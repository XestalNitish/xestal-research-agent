"""Agentic AI tools suite: Modular LangChain tools for LangGraph agents.

Usage:  from tools import ALL_TOOLS   ->  llm.bind_tools(ALL_TOOLS)
        from tools import RESEARCH_WEB_TOOLS
"""
from .registry import (
    ALL_TOOLS, ALL_100_TOOLS, ALL_135_TOOLS, TOOLS_BY_NUMBER, TOOLS_BY_NAME, TOOLS_BY_CATEGORY,
    MISSING_CATEGORIES, get_tool_by_number, get_tool_by_name, list_all_tools,
)

def _safe_import(mod_name: str, attr_name: str, default=None):
    try:
        import importlib
        mod = importlib.import_module(f".{mod_name}", __package__)
        return getattr(mod, attr_name, default if default is not None else [])
    except Exception:
        return default if default is not None else []

ORCHESTRATION_TOOLS = _safe_import("orchestration", "ORCHESTRATION_TOOLS")
RESEARCH_WEB_TOOLS = _safe_import("research_web", "RESEARCH_WEB_TOOLS")
ALL_RESEARCH_TOOLS = _safe_import("research_web", "ALL_RESEARCH_TOOLS")
MEMORY_CONTEXT_TOOLS = _safe_import("memory_context", "MEMORY_CONTEXT_TOOLS")
CODE_DEV_TOOLS = _safe_import("code_dev", "CODE_DEV_TOOLS")
DATABASE_DATA_TOOLS = _safe_import("database_data", "DATABASE_DATA_TOOLS")
MEDIA_CONTENT_TOOLS = _safe_import("media_content", "MEDIA_CONTENT_TOOLS")
SOCIAL_GROWTH_TOOLS = _safe_import("social_growth", "SOCIAL_GROWTH_TOOLS")
INTEGRATION_OPS_TOOLS = _safe_import("integration_ops", "INTEGRATION_OPS_TOOLS")
MATH_ANALYSIS_TOOLS = _safe_import("math_analysis", "MATH_ANALYSIS_TOOLS")
DOC_GENERATION_TOOLS = _safe_import("doc_generation", "DOC_GENERATION_TOOLS")
FILE_OPS_TOOLS = _safe_import("file_ops", "FILE_OPS_TOOLS")
TEXT_NLP_TOOLS = _safe_import("text_nlp", "TEXT_NLP_TOOLS")
SECURITY_PRIVACY_TOOLS = _safe_import("security_privacy", "SECURITY_PRIVACY_TOOLS")
EDUCATION_TOOLS = _safe_import("education", "EDUCATION_TOOLS")
FINANCE_BUSINESS_TOOLS = _safe_import("finance_business", "FINANCE_BUSINESS_TOOLS")
TIME_PRODUCTIVITY_TOOLS = _safe_import("time_productivity", "TIME_PRODUCTIVITY_TOOLS")
AI_ML_TOOLS = _safe_import("ai_ml_engineering", "AI_ML_TOOLS")
GEO_WEATHER_TOOLS = _safe_import("geo_weather", "GEO_WEATHER_TOOLS")

__all__ = [
    "ALL_TOOLS", "ALL_100_TOOLS", "ALL_135_TOOLS", "TOOLS_BY_NUMBER", "TOOLS_BY_NAME", "TOOLS_BY_CATEGORY",
    "MISSING_CATEGORIES", "get_tool_by_number", "get_tool_by_name", "list_all_tools",
    "ORCHESTRATION_TOOLS", "RESEARCH_WEB_TOOLS", "ALL_RESEARCH_TOOLS", "MEMORY_CONTEXT_TOOLS",
    "CODE_DEV_TOOLS", "DATABASE_DATA_TOOLS", "MEDIA_CONTENT_TOOLS", "SOCIAL_GROWTH_TOOLS",
    "INTEGRATION_OPS_TOOLS", "MATH_ANALYSIS_TOOLS", "DOC_GENERATION_TOOLS", "FILE_OPS_TOOLS",
    "TEXT_NLP_TOOLS", "SECURITY_PRIVACY_TOOLS", "EDUCATION_TOOLS", "FINANCE_BUSINESS_TOOLS",
    "TIME_PRODUCTIVITY_TOOLS", "AI_ML_TOOLS", "GEO_WEATHER_TOOLS",
]
