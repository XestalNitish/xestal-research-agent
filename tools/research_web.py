"""
Module 2: Research & Web Intelligence Tools (Tools 16 - 30)  -- FIXED VERSION

All 15 original tool names are kept so tools/registry.py keeps working
(numbering 16-30 unchanged), plus wikipedia_search as an extra.

Every tool returns plain text. Failures start with "TOOL_ERROR:" and empty
results with "NO_RESULTS:", so the agent can tell "the tool broke" apart from
"nothing exists".

Install (inside your .venv):
    python -m pip install -U ddgs requests beautifulsoup4 trafilatura feedparser \
        pypdf yt-dlp youtube-transcript-api langchain-core

Optional (.env):  TAVILY_API_KEY=...   GITHUB_TOKEN=...
"""

import io
import json
import os
import re
import urllib.parse
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional

import requests
from ._common import safe_request, safe_path
from bs4 import BeautifulSoup
from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
API_HEADERS = {"User-Agent": "research-agent/1.0 (student project)"}

ERR = "TOOL_ERROR:"
NO_RESULTS = "NO_RESULTS:"


def _fmt_results(results: List[Dict[str, str]]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r.get('title', '').strip()}\n"
            f"    URL: {r.get('url', '').strip()}\n"
            f"    {r.get('snippet', '').strip()}"
        )
    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# 16. web_search  (Tavily -> ddgs -> DDG html; every result carries a URL)
# ---------------------------------------------------------------------------
def _search_tavily(query: str, n: int) -> Optional[List[Dict[str, str]]]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return None  # not configured, skip silently
    r = requests.post(
        "https://api.tavily.com/search",
        headers={"Authorization": f"Bearer {key}"},
        json={"query": query, "max_results": n, "search_depth": "basic"},
        timeout=25,
    )
    r.raise_for_status()
    return [
        {"title": x.get("title", ""), "url": x.get("url", ""), "snippet": (x.get("content") or "")[:500]}
        for x in r.json().get("results", [])
    ]


def _search_ddgs(query: str, n: int) -> List[Dict[str, str]]:
    try:
        from ddgs import DDGS
    except ImportError:
        # Do NOT fall back to the old duckduckgo_search package: it returns empty lists.
        raise RuntimeError("package 'ddgs' is not installed. Run: python -m pip install -U ddgs")
    rows = DDGS().text(query, max_results=n)
    return [
        {"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
        for r in rows
    ]


def _search_ddg_html(query: str, n: int) -> List[Dict[str, str]]:
    resp = requests.post(
        "https://html.duckduckgo.com/html/",
        data={"q": query},
        headers=BROWSER_HEADERS,
        timeout=15,
    )
    if resp.status_code == 202:
        raise RuntimeError("DuckDuckGo bot challenge (HTTP 202)")
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    out = []
    for r in soup.select("div.result"):
        a = r.select_one("a.result__a")
        if not a:
            continue
        href = a.get("href", "")
        if "uddg=" in href:  # DDG redirect wrapper -> real URL
            href = urllib.parse.unquote(href.split("uddg=")[1].split("&")[0])
        sn = r.select_one(".result__snippet")
        out.append(
            {
                "title": a.get_text(strip=True),
                "url": href,
                "snippet": sn.get_text(" ", strip=True) if sn else "",
            }
        )
        if len(out) >= n:
            break
    return out


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web. Returns title, URL and snippet for each result.

    Use SHORT, specific queries (3-8 words), e.g. 'Anthropic computer use tool docs'.
    Do not stuff many keywords into one query. After searching, call web_reader
    on the most relevant URL to get the full content.

    Args:
        query: Short search query.
        max_results: Number of results (default 5).
    """
    providers = [("tavily", _search_tavily), ("ddgs", _search_ddgs), ("ddg_html", _search_ddg_html)]
    errors: List[str] = []
    for name, fn in providers:
        try:
            results = fn(query, max_results)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {str(e)[:120]}")
            continue
        if results is None:
            continue
        if results:
            return f"(provider: {name})\n\n" + _fmt_results(results)
        errors.append(f"{name}: 0 results")

    if any("0 results" not in e for e in errors):
        return (
            f"{ERR} web_search failed for '{query}'. Details: {' | '.join(errors)}. "
            "This is a TOOL failure, NOT proof that the information does not exist."
        )
    return f"{NO_RESULTS} nothing found for '{query}'. Try a shorter or different query."


# ---------------------------------------------------------------------------
# 17. news_search
# ---------------------------------------------------------------------------
@tool
def news_search(topic: str = "technology", max_results: int = 5) -> str:
    """Latest news headlines on a topic via Google News RSS.

    Args:
        topic: Topic to search.
        max_results: Max stories (default 5).
    """
    try:
        import feedparser

        url = f"https://news.google.com/rss/search?q={urllib.parse.quote(topic)}&hl=en-IN&gl=IN&ceid=IN:en"
        feed = feedparser.parse(url)
        items = [
            {"title": e.title, "url": e.link, "snippet": f"Published: {getattr(e, 'published', '')}"}
            for e in feed.entries[:max_results]
        ]
        return _fmt_results(items) if items else f"{NO_RESULTS} no news for '{topic}'."
    except Exception as e:
        return f"{ERR} news_search: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 18. web_reader  (trafilatura for clean article text)
# ---------------------------------------------------------------------------
@tool
def web_reader(url: str, max_chars: int = 6000) -> str:
    """Read the main text content of a web page (docs, blog, article, GitHub README).

    Args:
        url: Full URL to read.
        max_chars: Max characters to return (default 6000).
    """
    try:
        resp = safe_request("GET", url, headers=BROWSER_HEADERS, timeout=15)
        if resp.status_code != 200:
            return f"{ERR} HTTP {resp.status_code} for {url}"
        ctype = resp.headers.get("Content-Type", "").lower()
        if "pdf" in ctype or url.lower().endswith(".pdf"):
            return f"{ERR} This URL is a PDF. Use pdf_reader with the same URL."

        text = ""
        try:
            import trafilatura

            text = trafilatura.extract(resp.text, include_tables=True) or ""
        except ImportError:
            pass

        if not text:  # fallback: plain BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "noscript", "aside", "form"]):
                tag.extract()
            text = " ".join(soup.get_text(separator=" ", strip=True).split())

        if not text:
            return f"{NO_RESULTS} no readable text at {url} (page may need JavaScript)."
        return f"Source: {url}\n\n{text[:max_chars]}"
    except Exception as e:
        return f"{ERR} web_reader({url}): {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 19. browser_agent  (honest: a page inspector, NOT a real browser)
# ---------------------------------------------------------------------------
@tool
def browser_agent(url: str, extract_type: str = "summary") -> str:
    """Inspect a web page WITHOUT running JavaScript: title, description, headings, links.
    This is not a real browser. Use web_reader to read the page text itself.

    Args:
        url: Page URL.
        extract_type: 'summary' (title, description, headings), 'links', or 'full_structure' (both).
    """
    try:
        resp = safe_request("GET", url, headers=BROWSER_HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")
        title = soup.title.get_text(strip=True) if soup.title else ""
        md = soup.find("meta", attrs={"name": "description"})
        desc = (md.get("content") or "").strip() if md else ""

        data = {"url": url, "status_code": resp.status_code, "title": title, "meta_description": desc}
        if extract_type in ("summary", "full_structure"):
            data["headings"] = [h.get_text(" ", strip=True) for h in soup.find_all(["h1", "h2", "h3"])][:15]
        if extract_type in ("links", "full_structure"):
            links = []
            for a in soup.find_all("a", href=True):
                href = urllib.parse.urljoin(url, a["href"])
                if href.startswith("http"):
                    links.append({"text": a.get_text(" ", strip=True)[:80] or "link", "url": href})
                if len(links) >= 20:
                    break
            data["links"] = links
        return json.dumps(data, indent=2)
    except Exception as e:
        return f"{ERR} browser_agent({url}): {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 20. url_extractor
# ---------------------------------------------------------------------------
@tool
def url_extractor(source: str) -> str:
    """Extract URLs and email addresses from raw text or from a web page URL.

    Args:
        source: Text string or a page URL starting with http(s)://.
    """
    try:
        text = source
        if source.startswith(("http://", "https://")):
            text = safe_request("GET", source, headers=BROWSER_HEADERS, timeout=12).text

        urls = [u.rstrip(".,;:") for u in re.findall(r'https?://[^\s<>"\')]+', text)]
        emails = re.findall(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text)
        urls = list(dict.fromkeys(urls))[:20]
        emails = list(dict.fromkeys(emails))[:10]
        return json.dumps(
            {"urls_count": len(urls), "urls": urls, "emails_count": len(emails), "emails": emails},
            indent=2,
        )
    except Exception as e:
        return f"{ERR} url_extractor: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 21. pdf_reader  (local path OR URL; arXiv abs links auto-converted)
# ---------------------------------------------------------------------------
@tool
def pdf_reader(source: str, max_pages: int = 8, max_chars: int = 8000) -> str:
    """Extract text from a PDF given a URL (e.g. an arXiv paper) or a local file path.

    For arXiv you can pass either arxiv.org/abs/<id> or arxiv.org/pdf/<id>.

    Args:
        source: PDF URL or local path.
        max_pages: Max pages to read (default 8).
        max_chars: Max characters to return (default 8000).
    """
    try:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader

        if source.startswith(("http://", "https://")):
            if "arxiv.org/abs/" in source:
                source = source.replace("/abs/", "/pdf/")
            r = safe_request("GET", source, headers=BROWSER_HEADERS, timeout=40, max_bytes=25_000_000)
            if r.status_code != 200:
                return f"{ERR} HTTP {r.status_code} for {source}"
            data = r.content
        else:
            local = safe_path(source, must_exist=True)  # workspace sandbox only
            with open(local, "rb") as f:
                data = f.read()

        reader = PdfReader(io.BytesIO(data))
        total = len(reader.pages)
        parts = []
        for i in range(min(total, max_pages)):
            parts.append(f"--- Page {i + 1} ---\n{(reader.pages[i].extract_text() or '').strip()}")
        text = "\n\n".join(parts)
        return f"Total pages: {total} (read {min(total, max_pages)})\n\n{text[:max_chars]}"
    except Exception as e:
        return f"{ERR} pdf_reader: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 22. youtube_search
# ---------------------------------------------------------------------------
@tool
def youtube_search(query: str, max_results: int = 5) -> str:
    """Search YouTube videos (title, URL, uploader).

    Args:
        query: Topic or title.
        max_results: Max videos (default 5).
    """
    try:
        import yt_dlp

        with yt_dlp.YoutubeDL({"quiet": True, "extract_flat": True}) as ydl:
            info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
        items = [
            {
                "title": e.get("title", ""),
                "url": f"https://www.youtube.com/watch?v={e.get('id')}",
                "snippet": f"Uploader: {e.get('uploader', 'Unknown')}",
            }
            for e in info.get("entries", [])
            if e
        ]
        return _fmt_results(items) if items else f"{NO_RESULTS} no videos for '{query}'."
    except Exception as e:
        return f"{ERR} youtube_search: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 23. youtube_transcript
# ---------------------------------------------------------------------------
@tool
def youtube_transcript(video_url_or_id: str, max_chars: int = 6000) -> str:
    """Fetch the transcript/captions of a YouTube video.

    Args:
        video_url_or_id: Video URL or 11-character ID.
        max_chars: Max characters to return (default 6000).
    """
    try:
        m = re.search(r"(?:v=|youtu\.be/|shorts/)([A-Za-z0-9_-]{11})", video_url_or_id)
        vid = m.group(1) if m else video_url_or_id.strip()

        from youtube_transcript_api import YouTubeTranscriptApi

        try:  # youtube-transcript-api >= 1.0
            fetched = YouTubeTranscriptApi().fetch(vid, languages=["en", "hi"])
            text = " ".join(s.text for s in fetched)
        except AttributeError:  # older versions
            data = YouTubeTranscriptApi.get_transcript(vid, languages=["en", "hi"])
            text = " ".join(i["text"] for i in data)
        return text[:max_chars] if text else f"{NO_RESULTS} empty transcript."
    except Exception as e:
        return f"{ERR} youtube_transcript: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 24. reddit_search
# ---------------------------------------------------------------------------
@tool
def reddit_search(query: str, max_results: int = 5) -> str:
    """Search Reddit posts (often blocked; may return TOOL_ERROR).

    Args:
        query: Keywords.
        max_results: Max posts (default 5).
    """
    try:
        resp = requests.get(
            "https://www.reddit.com/search.json",
            params={"q": query, "limit": max_results},
            headers=API_HEADERS,
            timeout=12,
        )
        if resp.status_code != 200:
            return f"{ERR} Reddit blocked the request (HTTP {resp.status_code})."
        posts = resp.json().get("data", {}).get("children", [])
        items = [
            {
                "title": f"{p['data'].get('title')} (r/{p['data'].get('subreddit')}, {p['data'].get('ups', 0)} upvotes)",
                "url": f"https://reddit.com{p['data'].get('permalink')}",
                "snippet": "",
            }
            for p in posts
        ]
        return _fmt_results(items) if items else f"{NO_RESULTS} no Reddit posts for '{query}'."
    except Exception as e:
        return f"{ERR} reddit_search: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 25. github_search
# ---------------------------------------------------------------------------
@tool
def github_search(query: str, max_results: int = 5) -> str:
    """Search GitHub repositories (good for finding official repos of benchmarks and tools).

    Args:
        query: Short repo keywords, e.g. 'OSWorld'.
        max_results: Max repos (default 5).
    """
    try:
        headers = dict(API_HEADERS)
        if os.getenv("GITHUB_TOKEN"):
            headers["Authorization"] = f"Bearer {os.getenv('GITHUB_TOKEN')}"
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": max_results},
            headers=headers,
            timeout=12,
        )
        if resp.status_code != 200:
            return f"{ERR} GitHub HTTP {resp.status_code} (rate limit? set GITHUB_TOKEN)."
        items = resp.json().get("items", [])
        if not items:
            return f"{NO_RESULTS} no GitHub repos for '{query}'."
        return _fmt_results(
            [
                {
                    "title": f"{i.get('full_name')} ({i.get('stargazers_count', 0)} stars)",
                    "url": i.get("html_url", ""),
                    "snippet": i.get("description") or "No description",
                }
                for i in items
            ]
        )
    except Exception as e:
        return f"{ERR} github_search: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 26. arxiv_search  (phrase-first query + timeout handling)
# ---------------------------------------------------------------------------
def _arxiv_fetch(search_query: str, n: int) -> List[Dict[str, str]]:
    resp = requests.get(
        "https://export.arxiv.org/api/query",
        params={
            "search_query": search_query,
            "start": 0,
            "max_results": n,
            "sortBy": "relevance",
            "sortOrder": "descending",
        },
        headers=API_HEADERS,
        timeout=(10, 40),
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)
    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers = []
    for e in root.findall("a:entry", ns):
        abs_url = (e.find("a:id", ns).text or "").strip()
        authors = [a.find("a:name", ns).text for a in e.findall("a:author", ns)]
        papers.append(
            {
                "title": " ".join((e.find("a:title", ns).text or "").split()),
                "url": abs_url,
                "snippet": (
                    f"Published: {(e.find('a:published', ns).text or '')[:10]} | "
                    f"Authors: {', '.join(authors[:3])} | PDF: {abs_url.replace('/abs/', '/pdf/')}\n"
                    f"    Abstract: {' '.join((e.find('a:summary', ns).text or '').split())[:450]}..."
                ),
            }
        )
    return papers


@tool
def arxiv_search(query: str, max_results: int = 3) -> str:
    """Search academic papers on arXiv. Use 1-4 SPECIFIC keywords or a paper name,
    e.g. 'OSWorld benchmark' or 'GUI grounding agent'. Long keyword lists give bad results.

    Args:
        query: Short topic or paper title.
        max_results: Max papers (default 3).
    """
    words = re.findall(r"[A-Za-z0-9\-]+", query)
    if not words:
        return f"{ERR} empty query."
    attempts = []
    if len(words) <= 5:
        attempts.append(f'all:"{" ".join(words)}"')  # exact phrase first
    attempts.append(" AND ".join(f"all:{w}" for w in words[:4]))  # then all words must match
    try:
        for sq in attempts:
            papers = _arxiv_fetch(sq, max_results)
            if papers:
                return _fmt_results(papers)
        return f"{NO_RESULTS} no arXiv papers for '{query}'. Try other keywords."
    except requests.exceptions.Timeout:
        return (
            f"{ERR} arXiv API timed out. Use web_search with '{query} site:arxiv.org' instead, "
            "then pdf_reader on the paper."
        )
    except Exception as e:
        return f"{ERR} arxiv_search: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 27. google_trends
# ---------------------------------------------------------------------------
@tool
def google_trends(region: str = "US") -> str:
    """Daily trending searches via Google Trends RSS (useful for content ideas, not technical research).

    Args:
        region: Country code like 'IN', 'US'.
    """
    try:
        import feedparser

        feed = feedparser.parse(f"https://trends.google.com/trending/rss?geo={region.upper()}")
        items = [
            {"title": e.title, "url": e.link, "snippet": f"Traffic: {getattr(e, 'ht_approx_traffic', 'n/a')}"}
            for e in feed.entries[:10]
        ]
        return _fmt_results(items) if items else f"{NO_RESULTS} no trends for '{region}'."
    except Exception as e:
        return f"{ERR} google_trends: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 28. social_search  (HackerNews only; Mastodon was never implemented)
# ---------------------------------------------------------------------------
@tool
def social_search(topic: str, platform: str = "hackernews") -> str:
    """Search developer discussions on HackerNews (the only supported platform).

    Args:
        topic: Keyword.
        platform: 'hackernews'.
    """
    if platform.lower() != "hackernews":
        return f"{ERR} unsupported platform '{platform}'. Supported: hackernews."
    try:
        resp = requests.get(
            "https://hn.algolia.com/api/v1/search",
            params={"query": topic, "tags": "story", "hitsPerPage": 5},
            timeout=12,
        ).json()
        items = [
            {
                "title": f"{h.get('title')} (points {h.get('points')}, comments {h.get('num_comments')})",
                "url": h.get("url") or f"https://news.ycombinator.com/item?id={h.get('objectID')}",
                "snippet": "",
            }
            for h in resp.get("hits", [])
        ]
        return _fmt_results(items) if items else f"{NO_RESULTS} no HN stories for '{topic}'."
    except Exception as e:
        return f"{ERR} social_search: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 29. source_validator  (honest: technical hints only, no fake trust score)
# ---------------------------------------------------------------------------
_KNOWN_PRIMARY = (
    "github.com", "arxiv.org", "wikipedia.org", "nih.gov", "nature.com", "ieee.org",
    "acm.org", "python.org", "openreview.net", "reuters.com", "bbc.com",
)
_INSTITUTIONAL_TLDS = (".edu", ".gov", ".ac.in", ".edu.in", ".gov.in", ".ac.uk")


@tool
def source_validator(url: str) -> str:
    """Quick technical check of a source URL: HTTPS, HTTP status, final URL and whether the
    domain is a well-known primary/institutional source. A hint only, NOT a fact-check.

    Args:
        url: The URL to check.
    """
    try:
        host = urllib.parse.urlparse(url).netloc.lower().split(":")[0]
        known = any(host == d or host.endswith("." + d) for d in _KNOWN_PRIMARY)
        institutional = host.endswith(_INSTITUTIONAL_TLDS)

        resp = safe_request("GET", url, headers=BROWSER_HEADERS, timeout=10, max_bytes=50_000)
        return json.dumps(
            {
                "url": url,
                "final_url": resp.url,
                "domain": host,
                "status_code": resp.status_code,
                "uses_https": resp.url.startswith("https://"),
                "reputation_hint": (
                    "well-known primary or institutional domain"
                    if (known or institutional)
                    else "unknown domain: confirm claims with a second source"
                ),
            },
            indent=2,
        )
    except Exception as e:
        return f"{ERR} source_validator: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# 30. research_summarizer  (honest: cleans notes, does NOT summarize)
# ---------------------------------------------------------------------------
@tool
def research_summarizer(research_data: str, focus_topic: str = "General") -> str:
    """Clean raw research text into a de-duplicated evidence list plus a source-URL list.
    It does NOT write a summary; the LLM synthesis step does that.

    Args:
        research_data: Raw text, findings or search snippets.
        focus_topic: The question being researched.
    """
    try:
        lines = list(dict.fromkeys(l.strip() for l in research_data.split("\n") if l.strip()))
        urls = list(dict.fromkeys(u.rstrip(".,;:") for u in re.findall(r"https?://[^\s<>\"')]+", research_data)))
        evidence = "\n".join(f"{i}. {l[:300]}" for i, l in enumerate(lines[:30], 1))
        sources = "\n".join(f"- {u}" for u in urls[:20]) or "- (no URLs found)"
        return f"# Evidence: {focus_topic}\n\n{evidence}\n\n## Sources\n{sources}"
    except Exception as e:
        return f"{ERR} research_summarizer: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# Extra (not in the numbered 100): wikipedia_search
# ---------------------------------------------------------------------------
@tool
def wikipedia_search(topic: str) -> str:
    """Get the Wikipedia summary for a topic. Best for definitions and background.

    Args:
        topic: Topic name, e.g. 'Large language model'.
    """
    try:
        s = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": topic, "srlimit": 1, "format": "json"},
            headers=API_HEADERS,
            timeout=12,
        ).json()
        hits = s.get("query", {}).get("search", [])
        if not hits:
            return f"{NO_RESULTS} no Wikipedia page for '{topic}'."
        title = hits[0]["title"]
        r = requests.get(
            "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title.replace(" ", "_")),
            headers=API_HEADERS,
            timeout=12,
        ).json()
        url = r.get("content_urls", {}).get("desktop", {}).get("page", "")
        return f"{r.get('title', title)}\nURL: {url}\n\n{r.get('extract', '')}"
    except Exception as e:
        return f"{ERR} wikipedia_search: {type(e).__name__}: {str(e)[:150]}"


# ---------------------------------------------------------------------------
# Tool lists
# ---------------------------------------------------------------------------
# Lean set that actually helps an LLM researcher. Bind THIS by default.
RESEARCH_WEB_TOOLS = [
    web_search,
    web_reader,
    arxiv_search,
    wikipedia_search,
    github_search,
    pdf_reader,
]

# Bigger set for deep dives. source_validator and research_summarizer are
# helpers, intentionally NOT bound to the LLM.
ALL_RESEARCH_TOOLS = RESEARCH_WEB_TOOLS + [
    news_search,
    social_search,
    reddit_search,
    youtube_search,
    youtube_transcript,
    google_trends,
    browser_agent,
    url_extractor,
]


# The 15 numbered tools (16-30) of the master registry, in the original, stable order.
# (wikipedia_search is available to agents via RESEARCH_WEB_TOOLS but is not a numbered registry slot.)
REGISTRY_RESEARCH_TOOLS = [
    web_search, news_search, web_reader, browser_agent, url_extractor,
    pdf_reader, youtube_search, youtube_transcript, reddit_search, github_search,
    arxiv_search, google_trends, social_search, source_validator, research_summarizer,
]


# ---------------------------------------------------------------------------
# Smoke test:  python tools/research_web.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        (web_search, {"query": "Anthropic computer use tool docs"}),
        (web_search, {"query": "OSWorld benchmark"}),
        (arxiv_search, {"query": "OSWorld benchmark"}),
        (wikipedia_search, {"topic": "Large language model"}),
        (github_search, {"query": "OSWorld"}),
        (web_reader, {"url": "https://os-world.github.io/"}),
    ]
    for t, args in tests:
        out = t.invoke(args)
        status = "FAIL " if out.startswith(ERR) else ("EMPTY" if out.startswith(NO_RESULTS) else "OK   ")
        print(f"[{status}] {t.name}({args})\n{out[:300]}\n" + "-" * 70)
