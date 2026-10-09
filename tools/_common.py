"""
Shared foundation for the agent tools package.

Everything that makes a tool "production grade" lives here so each tool module stays small:

* uniform results      -> ERR / NO_RESULTS markers, err(), no_results(), ok_json()
* error guard          -> @guard turns any exception into "TOOL_ERROR: ..." (never crashes the agent)
* filesystem sandbox   -> safe_path() keeps file tools inside AGENT_WORKSPACE and blocks secret files
* network sandbox      -> check_url_safe() / safe_request() block SSRF (localhost, private IPs, metadata)
* durable state        -> db_connect() sqlite (WAL, transactions) + atomic file writes
* subprocess safety    -> run_command() (no shell, scrubbed env, timeout, output cap)
* LLM access           -> llm_text() (Gemini via langchain) with a test override and clean fallback

Environment variables (all optional):
    AGENT_WORKSPACE          root folder file tools may touch        (default: current directory)
    AGENT_DATA_DIR           where sqlite/state files live           (default: ~/.agent_tools)
    AGENT_ALLOW_PRIVATE_NET  "1" lets http tools reach private IPs  (default: blocked)
    GOOGLE_API_KEY           enables LLM-backed tools                (Gemini)
    AGENT_LLM_MODEL          LLM model name                          (default: gemini-2.5-flash)
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import ipaddress
import json
import logging
import os
import re
import socket
import sqlite3
import subprocess
import tempfile
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple

logger = logging.getLogger("agent_tools")

ERR = "TOOL_ERROR:"
NO_RESULTS = "NO_RESULTS:"
TEMPLATE_FALLBACK = "[TEMPLATE_FALLBACK: no LLM key configured, returning a basic template]"


# ---------------------------------------------------------------------------
# Exceptions (message is shown to the agent, so keep it short and actionable)
# ---------------------------------------------------------------------------
class ToolError(Exception):
    """Expected failure with a message safe to show the agent."""


class MissingDependency(ToolError):
    pass


class UnsafePath(ToolError):
    pass


class UnsafeURL(ToolError):
    pass


class LLMUnavailable(ToolError):
    pass


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def err(message: str) -> str:
    return f"{ERR} {redact_secrets(str(message))}"


def no_results(message: str) -> str:
    return f"{NO_RESULTS} {message}"


def is_error(text: str) -> bool:
    return isinstance(text, str) and text.startswith(ERR)


def ok_json(obj: Any) -> str:
    return json.dumps(obj, indent=2, default=str, ensure_ascii=False)


def truncate(text: str, limit: int = 6000, marker: str = "\n...[truncated]") -> str:
    if text is None:
        return ""
    return text if len(text) <= limit else text[:limit] + marker


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def slugify(text: str, default: str = "item") -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text)).strip("._-")
    return s[:80] or default


def safe_identifier(name: str) -> str:
    """Validate an SQL identifier (table / column). Raises ToolError if unsafe."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,62}", str(name or "")):
        raise ToolError(f"invalid identifier '{name}' (letters, digits, underscore; must not start with a digit)")
    return name


def no_newlines(value: str, field: str = "value") -> str:
    """Reject header-injection attempts (CR/LF) in email headers, HTTP headers, etc."""
    if "\r" in value or "\n" in value:
        raise ToolError(f"{field} must not contain line breaks")
    return value


def parse_json_arg(value: Any, default: Any = None, expect: Tuple[type, ...] = (dict, list)) -> Any:
    """Parse a JSON string argument coming from an LLM. Empty -> default."""
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    if isinstance(value, expect):
        return value
    try:
        parsed = json.loads(value)
    except (TypeError, ValueError) as e:
        raise ToolError(f"argument is not valid JSON: {e}")
    if not isinstance(parsed, expect):
        names = "/".join(t.__name__ for t in expect)
        raise ToolError(f"JSON argument must be of type {names}")
    return parsed


def extract_json(text: str) -> Any:
    """Pull the first JSON object/array out of an LLM answer (handles ``` fences)."""
    t = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", t)
    if m:
        t = m.group(1).strip()
    try:
        return json.loads(t)
    except ValueError:
        pass
    for open_c, close_c in (("{", "}"), ("[", "]")):
        a, b = t.find(open_c), t.rfind(close_c)
        if a != -1 and b > a:
            try:
                return json.loads(t[a : b + 1])
            except ValueError:
                continue
    raise ToolError("could not parse JSON from model output")


_SECRET_PATTERNS = [
    re.compile(r"AIza[0-9A-Za-z_\-]{30,}"),                    # Google API key
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),                      # OpenAI-style
    re.compile(r"gh[pousr]_[A-Za-z0-9]{30,}"),                  # GitHub tokens
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"),               # Slack
    re.compile(r"tvly-[A-Za-z0-9\-_]{20,}"),                    # Tavily
    re.compile(r"(?i)(api[_-]?key|token|secret|password)=([^&\s\"']{6,})"),
]


def redact_secrets(text: str) -> str:
    """Mask API keys / tokens that may leak through exception messages or URLs."""
    out = text
    for pat in _SECRET_PATTERNS:
        if pat.groups >= 2:
            out = pat.sub(lambda m: f"{m.group(1)}=***", out)
        else:
            out = pat.sub("***REDACTED***", out)
    return out


# ---------------------------------------------------------------------------
# Guard decorator
# ---------------------------------------------------------------------------
def guard(fn: Callable[..., Any]) -> Callable[..., Any]:
    """Put UNDER @tool:   @tool / @guard / def my_tool(...)

    Converts every exception into a "TOOL_ERROR: ..." string so one bad call can never
    crash the agent loop, and keeps the full traceback in the log.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
            return result if isinstance(result, str) else ok_json(result)
        except ToolError as e:
            return err(f"{fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001 - last line of defence by design
            logger.exception("tool %s failed", fn.__name__)
            return err(f"{fn.__name__}: {type(e).__name__}: {str(e)[:300]}")

    return wrapper


def require(module: str, pip_name: Optional[str] = None):
    """Import an optional dependency or raise a clear, actionable error."""
    try:
        return importlib.import_module(module)
    except ImportError:
        raise MissingDependency(
            f"missing package '{module}'. Install it with: python -m pip install {pip_name or module}"
        )


# ---------------------------------------------------------------------------
# Filesystem sandbox
# ---------------------------------------------------------------------------
_SECRET_NAMES = {
    ".env", ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519", "id_dsa", "known_hosts",
    "credentials", "credentials.json", "client_secret.json", "token.json", "secrets.json",
    "service_account.json",
}
_SECRET_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".kdbx", ".jks")
_SECRET_DIRS = {".ssh", ".aws", ".gnupg", ".kube"}


def workspace_root() -> Path:
    return Path(os.environ.get("AGENT_WORKSPACE") or os.getcwd()).resolve()


def is_secret_path(path: Path) -> bool:
    name = path.name.lower()
    if name in _SECRET_NAMES or name.startswith(".env") or name.endswith(_SECRET_SUFFIXES):
        return True
    return any(part.lower() in _SECRET_DIRS for part in path.parts)


def safe_path(path: str, *, must_exist: bool = False, allow_secret: bool = False,
              base: Optional[str] = None) -> Path:
    """Resolve `path` inside the workspace. Raises UnsafePath on escape or secret files."""
    if not path or not str(path).strip():
        raise ToolError("path is empty")
    root = Path(base).resolve() if base else workspace_root()
    p = Path(str(path)).expanduser()
    if not p.is_absolute():
        p = root / p
    p = p.resolve()
    try:
        p.relative_to(root)
    except ValueError:
        raise UnsafePath(
            f"path '{path}' is outside the workspace ({root}). "
            "Use a relative path, or set AGENT_WORKSPACE to widen access."
        )
    if not allow_secret and is_secret_path(p):
        raise UnsafePath("access to secret/credential files is blocked")
    if must_exist and not p.exists():
        raise ToolError(f"file not found: {path}")
    return p


def safe_output_path(path: str, default_name: str = "output.txt") -> Path:
    """Like safe_path, for files we are about to create; creates parent folders."""
    p = safe_path(path or default_name)
    if p.exists() and p.is_dir():
        p = p / default_name
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_bytes(path: Path | str, data: bytes) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path | str, text: str, encoding: str = "utf-8") -> None:
    atomic_write_bytes(path, text.encode(encoding))


def read_text_file(path: Path, max_bytes: int = 2_000_000) -> str:
    size = path.stat().st_size
    if size > max_bytes:
        raise ToolError(f"file is {size} bytes, larger than the {max_bytes} byte limit")
    return path.read_text(encoding="utf-8", errors="replace")


def data_dir(*parts: str) -> Path:
    root = Path(os.environ.get("AGENT_DATA_DIR") or (Path.home() / ".agent_tools"))
    d = root.joinpath(*parts)
    d.mkdir(parents=True, exist_ok=True)
    return d


@contextlib.contextmanager
def db_connect(name: str, schema: Optional[str] = None) -> Iterator[sqlite3.Connection]:
    """sqlite connection (WAL, 30s busy timeout, Row factory). Commits on success, rolls back on error."""
    path = data_dir("db") / f"{slugify(name)}.db"
    conn = sqlite3.connect(str(path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        if schema:
            conn.executescript(schema)
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Network sandbox (SSRF protection)
# ---------------------------------------------------------------------------
_BLOCKED_HOST_SUFFIXES = (".localhost", ".internal", ".local")
_BLOCKED_HOSTS = {"localhost", "metadata.google.internal", "metadata"}


def _ip_is_blocked(ip: ipaddress._BaseAddress) -> bool:
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped:
        ip = ip.ipv4_mapped
    return bool(
        ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
        or ip.is_multicast or ip.is_unspecified
    )


def check_url_safe(url: str, *, allow_private: Optional[bool] = None) -> str:
    """Raise UnsafeURL unless `url` is an http(s) URL pointing at a public host."""
    if allow_private is None:
        allow_private = env_flag("AGENT_ALLOW_PRIVATE_NET")
    parsed = urllib.parse.urlparse(str(url).strip())
    if parsed.scheme not in ("http", "https"):
        raise UnsafeURL("only http:// and https:// URLs are allowed")
    host = parsed.hostname
    if not host:
        raise UnsafeURL("URL has no host")
    if allow_private:
        return url
    h = host.lower()
    if h in _BLOCKED_HOSTS or h.endswith(_BLOCKED_HOST_SUFFIXES):
        raise UnsafeURL(f"host '{host}' is blocked (internal address)")
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80),
                                   proto=socket.IPPROTO_TCP)
    except socket.gaierror:
        raise ToolError(f"cannot resolve host '{host}'")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if _ip_is_blocked(ip):
            raise UnsafeURL(
                f"host '{host}' resolves to a private/internal address ({ip}). "
                "Set AGENT_ALLOW_PRIVATE_NET=1 only if you really need this."
            )
    return url


DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def safe_request(method: str, url: str, *, timeout: float = 15, max_bytes: int = 2_000_000,
                 max_redirects: int = 5, allow_private: Optional[bool] = None, **kwargs):
    """requests wrapper: validates every redirect hop, caps the body size, never raises on HTTP status.

    The returned response has `.truncated` (bool). Network errors raise requests exceptions
    (caught by @guard).
    """
    import requests

    headers = dict(DEFAULT_HEADERS)
    headers.update(kwargs.pop("headers", None) or {})
    for k, v in headers.items():
        no_newlines(str(k), "header name")
        no_newlines(str(v), "header value")

    current, verb = url, method.upper()
    for _ in range(max_redirects + 1):
        check_url_safe(current, allow_private=allow_private)
        resp = requests.request(verb, current, headers=headers, timeout=timeout,
                                allow_redirects=False, stream=True, **kwargs)
        if resp.status_code in (301, 302, 303, 307, 308):
            location = resp.headers.get("Location")
            resp.close()
            if not location:
                raise ToolError("redirect response without Location header")
            current = urllib.parse.urljoin(current, location)
            if resp.status_code == 303:
                verb = "GET"
                kwargs.pop("json", None)
                kwargs.pop("data", None)
            continue
        body = bytearray()
        for chunk in resp.iter_content(65536):
            body.extend(chunk)
            if len(body) > max_bytes:
                break
        resp._content = bytes(body[:max_bytes])
        resp._content_consumed = True
        resp.truncated = len(body) > max_bytes
        return resp
    raise ToolError("too many redirects")


# ---------------------------------------------------------------------------
# Subprocess safety
# ---------------------------------------------------------------------------
_SECRET_ENV_HINTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "PASSWD", "CREDENTIAL", "AUTH")


def clean_env(extra: Optional[dict] = None) -> dict:
    """Environment for child processes: current env minus anything that looks like a secret."""
    env = {k: v for k, v in os.environ.items() if not any(h in k.upper() for h in _SECRET_ENV_HINTS)}
    env["PYTHONIOENCODING"] = "utf-8"
    if extra:
        env.update(extra)
    return env


def run_command(args: Sequence[str], *, cwd: Optional[str] = None, timeout: int = 30,
                max_output: int = 20000, input_text: Optional[str] = None,
                env: Optional[dict] = None) -> Tuple[int, str, str]:
    """Run a program WITHOUT a shell. Returns (returncode, stdout, stderr), outputs truncated."""
    if not args:
        raise ToolError("empty command")
    try:
        res = subprocess.run(
            list(args), shell=False, cwd=cwd, capture_output=True, text=True, timeout=timeout,
            env=env if env is not None else clean_env(), input=input_text,
            encoding="utf-8", errors="replace",
        )
    except subprocess.TimeoutExpired:
        raise ToolError(f"command timed out after {timeout}s")
    except FileNotFoundError:
        raise ToolError(f"program not found: {args[0]}")
    return res.returncode, truncate(res.stdout, max_output), truncate(res.stderr, max_output)


# ---------------------------------------------------------------------------
# LLM access (used by tools that genuinely need language generation)
# ---------------------------------------------------------------------------
_LLM_OVERRIDE: Optional[Callable[[str, str], str]] = None


def set_llm_override(fn: Optional[Callable[[str, str], str]]) -> None:
    """Tests: make llm_text() return fn(prompt, system) without any network call."""
    global _LLM_OVERRIDE
    _LLM_OVERRIDE = fn


def message_text(resp: Any) -> str:
    content = getattr(resp, "content", resp)
    if isinstance(content, list):
        parts = []
        for p in content:
            parts.append(p.get("text", "") if isinstance(p, dict) else str(p))
        return "".join(parts).strip()
    return str(content).strip()


def llm_available() -> bool:
    if _LLM_OVERRIDE is not None:
        return True
    return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))


def llm_text(prompt: str, *, system: str = "", temperature: float = 0.4, max_tokens: int = 2048) -> str:
    """Single-shot LLM call. Raises LLMUnavailable when no key / package is configured."""
    if _LLM_OVERRIDE is not None:
        return _LLM_OVERRIDE(prompt, system)
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise LLMUnavailable("no GOOGLE_API_KEY (or GEMINI_API_KEY) set")
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage
    except ImportError:
        raise LLMUnavailable("langchain-google-genai is not installed")
    llm = ChatGoogleGenerativeAI(
        model=os.environ.get("AGENT_LLM_MODEL", "gemini-2.5-flash"),
        temperature=temperature,
        max_output_tokens=max_tokens,
        max_retries=3,
        google_api_key=key,
    )
    messages = ([SystemMessage(content=system)] if system else []) + [HumanMessage(content=prompt)]
    return message_text(llm.invoke(messages))


def llm_or_none(prompt: str, **kwargs) -> Optional[str]:
    """LLM text, or None when no LLM is configured (caller then returns a labelled template)."""
    try:
        return llm_text(prompt, **kwargs)
    except LLMUnavailable:
        return None
