"""Master registry: indexed lookup of every tool by number, name and category.

Numbers 1-100 are the original catalogue (stable). 101-150 are the new categories; a category whose
module is not installed yet is simply skipped, so the registry never breaks on a partial install.
"""
from __future__ import annotations

import importlib
from typing import Dict, List

# (category, module, list attribute, expected tool count)  -- ORDER DEFINES THE TOOL NUMBERS
_CATEGORY_SPECS = [
    ("orchestration", "orchestration", "ORCHESTRATION_TOOLS", 15),
    ("research_web", "research_web", "REGISTRY_RESEARCH_TOOLS", 15),
    ("memory_context", "memory_context", "MEMORY_CONTEXT_TOOLS", 10),
    ("code_dev", "code_dev", "CODE_DEV_TOOLS", 15),
    ("database_data", "database_data", "DATABASE_DATA_TOOLS", 10),
    ("media_content", "media_content", "MEDIA_CONTENT_TOOLS", 15),
    ("social_growth", "social_growth", "SOCIAL_GROWTH_TOOLS", 10),
    ("integration_ops", "integration_ops", "INTEGRATION_OPS_TOOLS", 10),
    ("math_analysis", "math_analysis", "MATH_ANALYSIS_TOOLS", 5),
    ("doc_generation", "doc_generation", "DOC_GENERATION_TOOLS", 5),
    ("file_ops", "file_ops", "FILE_OPS_TOOLS", 5),
    ("text_nlp", "text_nlp", "TEXT_NLP_TOOLS", 5),
    ("security_privacy", "security_privacy", "SECURITY_PRIVACY_TOOLS", 5),
    ("education", "education", "EDUCATION_TOOLS", 5),
    ("finance_business", "finance_business", "FINANCE_BUSINESS_TOOLS", 5),
    ("time_productivity", "time_productivity", "TIME_PRODUCTIVITY_TOOLS", 5),
    ("ai_ml_engineering", "ai_ml_engineering", "AI_ML_TOOLS", 5),
    ("geo_weather", "geo_weather", "GEO_WEATHER_TOOLS", 5),
]


def _load():
    all_tools: List = []
    by_category: Dict[str, List] = {}
    missing: List[str] = []
    for i, (cat, mod, attr, _n) in enumerate(_CATEGORY_SPECS):
        try:
            module = importlib.import_module(f".{mod}", __package__)
        except ModuleNotFoundError as e:
            if e.name == f"{__package__}.{mod}":
                missing.append(cat)
                continue
            raise
        tools = list(getattr(module, attr, []))
        by_category[cat] = tools
        all_tools.extend(tools)
    return all_tools, by_category, missing


ALL_TOOLS, TOOLS_BY_CATEGORY, MISSING_CATEGORIES = _load()
ALL_100_TOOLS = ALL_TOOLS[:100]  # backwards compatible name
ALL_135_TOOLS = ALL_TOOLS[:135]  # backwards compatible name (older __init__ files import it)
TOOLS_BY_NUMBER = {i + 1: t for i, t in enumerate(ALL_TOOLS)}
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}

if len(TOOLS_BY_NAME) != len(ALL_TOOLS):  # duplicate names would silently shadow each other
    from collections import Counter

    _dups = sorted(n for n, c in Counter(t.name for t in ALL_TOOLS).items() if c > 1)
    raise RuntimeError(f"duplicate tool names in registry: {_dups}")


def get_tool_by_number(number: int):
    """Return the tool at catalogue position `number` (1-based)."""
    if number not in TOOLS_BY_NUMBER:
        raise KeyError(f"No tool numbered {number} (valid: 1-{len(ALL_TOOLS)})")
    return TOOLS_BY_NUMBER[number]


def get_tool_by_name(name: str):
    """Return the tool with this exact name."""
    if name not in TOOLS_BY_NAME:
        raise KeyError(f"No tool named {name!r}")
    return TOOLS_BY_NAME[name]


def list_all_tools() -> List[dict]:
    """Catalogue rows: number, name, category, first docstring line."""
    cat_of = {t.name: c for c, ts in TOOLS_BY_CATEGORY.items() for t in ts}
    return [
        {"number": i + 1, "name": t.name, "category": cat_of[t.name],
         "description": (t.description or "").strip().splitlines()[0] if t.description else ""}
        for i, t in enumerate(ALL_TOOLS)
    ]
