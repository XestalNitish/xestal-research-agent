import os
import json
import re
import time
from datetime import datetime
from typing import TypedDict, Literal, Optional, List, Annotated
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END

from tools import (
    ALL_RESEARCH_TOOLS,
    TEXT_NLP_TOOLS,
    GEO_WEATHER_TOOLS,
    FILE_OPS_TOOLS,
    DOC_GENERATION_TOOLS
)

load_dotenv()

INVESTIGATION_TOOLS = (
    ALL_RESEARCH_TOOLS + 
    TEXT_NLP_TOOLS + 
    GEO_WEATHER_TOOLS + 
    FILE_OPS_TOOLS
)
DELIVERY_TOOLS = FILE_OPS_TOOLS + DOC_GENERATION_TOOLS


class WebSearchState(TypedDict):
    query: str
    attached_files: Optional[List[str]] 
    search_depth: Literal["basic", "medium", "advanced"]
    output_type: Literal["summary", "bullet_points", "deep_dive"]
    export_format: Literal["markdown", "pdf", "docx"]
    generated_search_queries: Optional[str]
    messages: Annotated[List[BaseMessage], add_messages]
    research_notes: Optional[str]
    draft_report: Optional[str]
    audit_feedback: Optional[str]
    audit_passed: bool
    delivery_messages: Annotated[List[BaseMessage], add_messages]
    iteration_count: int
    audit_retries: int
    final: Optional[str]


def sleep_rate_limit(seconds: float = 0.5):
    """Protects against API rate-limit spikes."""
    time.sleep(seconds)


def clamp_text(text: str, max_chars: int = 2500) -> str:
    """Guards LLM context window against massive raw tool dumps."""
    if len(text) > max_chars:
        return text[:max_chars] + "\n... [TRUNCATED FOR TOKEN SAFETY]"
    return text


def extract_content_text(response) -> str:
    """Extract plain text string safely from various LangChain message response formats."""
    if hasattr(response, "content"):
        content = response.content
    elif hasattr(response, "text"):
        content = response.text
    else:
        content = str(response)

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif hasattr(item, "text"):
                parts.append(getattr(item, "text", ""))
        return "\n".join(parts)
    return str(content or "")


def parse_json_safely(raw_text: str) -> dict:
    """Robust JSON extractor stripping markdown wrapping and extraneous noise."""
    clean = raw_text.strip()
    if "```" in clean:
        match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", clean)
        if match:
            clean = match.group(1).strip()
    try:
        return json.loads(clean)
    except Exception:
        start = clean.find("{")
        end = clean.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(clean[start : end + 1])
            except Exception:
                pass
    return {}


def route_investigation_loop(state: WebSearchState) -> str:
    """Routes between investigation tools and synthesizer."""
    last_message = state["messages"][-1]
    iteration = state.get("iteration_count", 0)

    if iteration >= 4:  # Allows file reading + web search hops
        return "synthesize"
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "investigation_tools"
    return "synthesize"


def route_audit_loop(state: WebSearchState) -> str:
    """Routes based on audit outcome."""
    passed = state.get("audit_passed", True)
    retries = state.get("audit_retries", 0)

    if not passed and retries <= 2:
        return "reporter"
    return "delivery_initiator"


def route_delivery_loop(state: WebSearchState) -> str:
    """Checks if delivery tools were invoked."""
    last_message = state["delivery_messages"][-1]
    if hasattr(last_message, "tool_calls") and len(last_message.tool_calls) > 0:
        return "delivery_tools"
    return "finalize"


def get_llm(
    provider: str = "Google Gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1
):
    """Instantiates a chat model based on provider preference and API keys."""
    prov = (provider or "").lower()

    if "gemini" in prov or "google" in prov:
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = api_key or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("Google API Key not found. Please provide it in the sidebar or .env file.")
        model = model_name or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=key,
            temperature=temperature,
            max_retries=3
        )

    elif "groq" in prov:
        from langchain_groq import ChatGroq
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Groq API Key not found. Please provide it in the sidebar or .env file.")
        model = model_name or "llama-3.3-70b-versatile"
        return ChatGroq(
            model_name=model,
            groq_api_key=key,
            temperature=temperature,
            max_retries=3
        )

    elif "openai" in prov:
        from langchain_openai import ChatOpenAI
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OpenAI API Key not found. Please provide it in the sidebar or .env file.")
        model = model_name or "gpt-4o-mini"
        return ChatOpenAI(
            model=model,
            api_key=key,
            temperature=temperature,
            max_retries=3
        )

    else:
        # Default try Google first, then Groq, then OpenAI
        if os.getenv("GOOGLE_API_KEY"):
            return get_llm("Google Gemini", model_name, api_key, temperature)
        if os.getenv("GROQ_API_KEY"):
            return get_llm("Groq", model_name, api_key, temperature)
        if os.getenv("OPENAI_API_KEY"):
            return get_llm("OpenAI", model_name, api_key, temperature)
        raise ValueError("No LLM API keys configured (GOOGLE_API_KEY, GROQ_API_KEY, or OPENAI_API_KEY).")


def build_research_graph(llm_base):
    """Compiles the full LangGraph research workflow with the bound LLM."""
    llm_investigator = llm_base.bind_tools(INVESTIGATION_TOOLS) if INVESTIGATION_TOOLS else llm_base
    llm_deliverer = llm_base.bind_tools(DELIVERY_TOOLS) if DELIVERY_TOOLS else llm_base

    def query_decomposer_node(state: WebSearchState) -> dict:
        user_query = state.get("query", "")
        attached = state.get("attached_files") or []
        depth = state.get("search_depth", "medium")
        current_time_str = datetime.now().strftime("%Y-%m-%d")
        file_note = f"\nUser attached files: {', '.join(attached)}" if attached else ""

        prompt = f"""You are a Principal Research Architect.
Current Date: {current_time_str}

Analyze the user intent:
Goal: {user_query}{file_note}
Target Depth: {depth}

Produce exactly TWO targeted web search queries that complement or verify any local materials.
Return ONLY valid JSON:
{{
  "query_1": "<first query>",
  "query_2": "<second query>"
}}"""

        try:
            response = llm_base.invoke(prompt)
            raw_text = extract_content_text(response)
            data = parse_json_safely(raw_text)
            q1 = data.get("query_1", user_query)
            q2 = data.get("query_2", "")
            combined_queries = f"{q1} | {q2}".strip(" |")
        except Exception:
            combined_queries = user_query

        task_instructions = f"User Request: {user_query}\nTarget Depth: {depth}\nWeb Search Focus: {combined_queries}"
        if attached:
            task_instructions += (
                f"\n\nPRIORITY DIRECTIVE: Local file(s) provided: {attached}. "
                "You MUST FIRST execute tool calls from FILE_OPS_TOOLS (e.g., read_file, read_pdf, extract_pdf) "
                "to inspect their contents before performing external searches."
            )

        return {
            "generated_search_queries": combined_queries,
            "iteration_count": 0,
            "audit_retries": 0,
            "delivery_messages": [],
            "messages": [
                SystemMessage(
                    content=(
                        f"You are an autonomous senior research agent. Current Date: {current_time_str}. "
                        "You have bound tools for inspecting local files/PDFs, searching the web, and extracting NLP entities. "
                        "Always execute tool calls to gather real, grounded evidence before returning your conclusions."
                    )
                ),
                HumanMessage(content=task_instructions)
            ]
        }

    def investigator_node(state: WebSearchState) -> dict:
        messages = state["messages"]
        current_iter = state.get("iteration_count", 0)

        sanitized_messages = []
        for msg in messages:
            if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
                sanitized_messages.append(ToolMessage(content=clamp_text(msg.content), tool_call_id=msg.tool_call_id))
            else:
                sanitized_messages.append(msg)

        response = llm_investigator.invoke(sanitized_messages)
        sleep_rate_limit(0.5)

        return {
            "messages": [response],
            "iteration_count": current_iter + 1
        }

    def research_synthesizer_node(state: WebSearchState) -> dict:
        messages = state["messages"]
        evidence_trail = []
        for m in messages:
            if isinstance(m, ToolMessage):
                evidence_trail.append(f"[TOOL DATA]: {clamp_text(str(m.content), 1200)}")
            elif isinstance(m, HumanMessage) and "User Request" in str(m.content):
                evidence_trail.append(f"[TASK]: {m.content}")

        synthesis_prompt = [
            SystemMessage(content="You are a Technical Research Synthesizer. Distill tool evidence into dense factual notes."),
            HumanMessage(
                content=(
                    f"Retrieved Tool Evidence:\n{chr(10).join(evidence_trail)}\n\n"
                    "CRITICAL DIRECTIVES:\n"
                    "1. Synthesize all extracted content from local files and web search tools.\n"
                    "2. Every metric, benchmark score, and technical claim MUST be backed by the tool data.\n"
                    "3. Retain verified source links, citations, and specific architectural mechanics."
                )
            )
        ]

        response = llm_investigator.invoke(synthesis_prompt)
        notes = extract_content_text(response)
        sleep_rate_limit(0.5)

        return {"research_notes": notes.strip()}

    def draft_reporter_node(state: WebSearchState) -> dict:
        user_query = state["query"]
        depth = state.get("search_depth", "medium")
        output_format = state.get("output_type", "deep_dive")
        notes = state.get("research_notes", "")
        audit_feedback = state.get("audit_feedback")

        feedback_block = f"\nCRITICAL CORRECTIONS FROM FACT-CHECK AUDITOR:\n{audit_feedback}\n" if audit_feedback else ""

        prompt = f"""You are a Lead Research Delivery Analyst.

Synthesize the gathered research notes into a finalized analytical report.

Original Query: {user_query}
Depth: {depth}
Format: {output_format}
Research Notes:
{notes}
{feedback_block}
STRICT ANTI-HALLUCINATION DIRECTIVES:
1. Every metric, benchmark score, and technical claim MUST be directly backed by the research notes.
2. Never fabricate parameter counts, benchmark percentages, or unverified claims.
3. Cite inline sources like [arxiv.org/abs/...], [github.com/...], or local file references directly next to claims.
4. If a specific metric was not found, explicitly state: 'Data point unverified in source search'.
5. Format cleanly using pure Markdown with headers, bold takeaways, and comparison tables."""

        response = llm_base.invoke(prompt)
        report_text = extract_content_text(response)
        sleep_rate_limit(0.5)

        return {"draft_report": report_text.strip()}

    def fact_checker_audit_node(state: WebSearchState) -> dict:
        notes = state.get("research_notes", "")
        draft = state.get("draft_report", "")
        retries = state.get("audit_retries", 0)

        prompt = f"""You are a Strict Fact-Checking & Grounding Auditor.
Validate the claims in the generated draft report against the raw ground-truth research notes.

Ground Truth Research Notes:
{notes[:3500]}

Generated Draft Report:
{draft}

Audit Criteria:
1. Are all benchmark numbers, model names, and claims backed by the notes?
2. Are URLs, citations, and document references legitimate and grounded?

Strict Output Constraints:
- Return ONLY valid JSON:
{{
  "is_grounded": true or false,
  "hallucination_detected": ["list of unbacked claims"],
  "audit_feedback": "<specific corrections for reporter, or 'PASS'>"
}}"""

        try:
            response = llm_base.invoke(prompt)
            raw_text = extract_content_text(response)
            data = parse_json_safely(raw_text)
            is_grounded = data.get("is_grounded", False)
            feedback = data.get("audit_feedback", "Auditor requires stricter citation verification.")
        except Exception as e:
            is_grounded = retries >= 2
            feedback = f"Parser issue during audit: {str(e)}"

        sleep_rate_limit(0.5)

        return {
            "audit_passed": is_grounded,
            "audit_feedback": feedback if not is_grounded else None,
            "audit_retries": retries + 1
        }

    def delivery_initiator_node(state: WebSearchState) -> dict:
        report = state.get("draft_report", "")
        query = state.get("query", "research_report")
        export_fmt = state.get("export_format", "markdown")
        clean_title = re.sub(r'[^a-zA-Z0-9_\- ]', '', query[:35]).strip().replace(' ', '_').lower() or "report"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_filename = f"research_vault/{clean_title}_{timestamp}.md"

        os.makedirs("research_vault", exist_ok=True)

        delivery_prompt = [
            SystemMessage(
                content=(
                    "You are an Autonomous Document Delivery & Storage Agent. "
                    "You have bound tools: FILE_OPS_TOOLS and DOC_GENERATION_TOOLS. "
                    "Your responsibility is to persist and export the final report to disk using these tools."
                )
            ),
            HumanMessage(
                content=(
                    f"PERSISTENCE DIRECTIVES:\n"
                    f"1. You MUST invoke `write_file` or a file-creation tool to save this verified markdown content to `{target_filename}`.\n"
                    f"2. Requested Export Format: {export_fmt.upper()}.\n"
                    f"If the export format is PDF or DOCX, invoke the appropriate tool from DOC_GENERATION_TOOLS.\n\n"
                    f"Report Content to Save:\n{report}"
                )
            )
        ]

        return {"delivery_messages": delivery_prompt}

    def delivery_agent_node(state: WebSearchState) -> dict:
        messages = state["delivery_messages"]
        response = llm_deliverer.invoke(messages)
        return {"delivery_messages": [response]}

    def finalize_output_node(state: WebSearchState) -> dict:
        report = state.get("draft_report", "")
        delivery_msgs = state.get("delivery_messages", [])

        tool_actions = []
        for msg in delivery_msgs:
            if isinstance(msg, ToolMessage):
                tool_actions.append(f"• {msg.content}")

        actions_summary = "\n".join(tool_actions) if tool_actions else "Deliverable processed by storage tools."

        final_text = (
            f"{report}\n\n"
            f"---\n"
            f"### 📦 Deliverable Persistence & Tool Execution Log\n"
            f"{actions_summary}"
        )

        return {"final": final_text}

    builder = StateGraph(WebSearchState)

    builder.add_node("decomposer", query_decomposer_node)
    builder.add_node("investigator", investigator_node)
    builder.add_node("investigation_tools", ToolNode(INVESTIGATION_TOOLS))
    builder.add_node("synthesizer", research_synthesizer_node)
    builder.add_node("reporter", draft_reporter_node)
    builder.add_node("auditor", fact_checker_audit_node)
    builder.add_node("delivery_initiator", delivery_initiator_node)
    builder.add_node("delivery_agent", delivery_agent_node)
    builder.add_node("delivery_tools", ToolNode(DELIVERY_TOOLS))
    builder.add_node("finalize", finalize_output_node)

    builder.add_edge(START, "decomposer")
    builder.add_edge("decomposer", "investigator")

    builder.add_conditional_edges(
        "investigator",
        route_investigation_loop,
        {
            "investigation_tools": "investigation_tools",
            "synthesize": "synthesizer"
        }
    )
    builder.add_edge("investigation_tools", "investigator")
    builder.add_edge("synthesizer", "reporter")
    builder.add_edge("reporter", "auditor")

    builder.add_conditional_edges(
        "auditor",
        route_audit_loop,
        {
            "reporter": "reporter",
            "delivery_initiator": "delivery_initiator"
        }
    )

    builder.add_edge("delivery_initiator", "delivery_agent")

    builder.add_conditional_edges(
        "delivery_agent",
        route_delivery_loop,
        {
            "delivery_tools": "delivery_tools",
            "finalize": "finalize"
        }
    )
    builder.add_edge("delivery_tools", "delivery_agent")
    builder.add_edge("finalize", END)

    return builder.compile()


def create_research_agent(
    llm=None,
    provider: str = "Google Gemini",
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.1
):
    """Factory creating a runnable LangGraph research agent instance."""
    if llm is None:
        llm = get_llm(provider=provider, model_name=model_name, api_key=api_key, temperature=temperature)
    return build_research_graph(llm)


# Lazily initialize default research agent if environment keys are present
research_web_agent = None
try:
    research_web_agent = create_research_agent()
except Exception:
    pass


if __name__ == "__main__":
    test_payload = {
        "query": (
            "Analyze GUI grounding architectures in Computer Use Agents (OmniParser vs Set-of-Marks). "
            "Compare latency, action space predictability, and token overhead."
        ),
        "attached_files": [],
        "search_depth": "medium",
        "output_type": "deep_dive",
        "export_format": "markdown",
        "generated_search_queries": None,
        "messages": [],
        "delivery_messages": [],
        "research_notes": None,
        "draft_report": None,
        "audit_feedback": None,
        "audit_passed": False,
        "iteration_count": 0,
        "audit_retries": 0,
        "final": None
    }

    print("\n🚀 Executing Production Web Research Agent...\n")
    agent = research_web_agent or create_research_agent()

    final_delivered_report = ""
    for step in agent.stream(test_payload):
        for node_name, state_update in step.items():
            print(f"[NODE COMPLETED: {node_name.upper()}]")
            if "generated_search_queries" in state_update:
                print(f"-> Queries: {state_update['generated_search_queries']}")
            if "messages" in state_update and hasattr(state_update["messages"][-1], "tool_calls"):
                tools_called = [tc["name"] for tc in state_update["messages"][-1].tool_calls]
                if tools_called:
                    print(f"-> Investigation Tools: {', '.join(tools_called)}")
            if "delivery_messages" in state_update and hasattr(state_update["delivery_messages"][-1], "tool_calls"):
                tools_called = [tc["name"] for tc in state_update["delivery_messages"][-1].tool_calls]
                if tools_called:
                    print(f"-> Delivery Tools: {', '.join(tools_called)}")
            if "audit_passed" in state_update:
                print(f"-> Grounding Audit Passed: {state_update['audit_passed']}")
            if "final" in state_update and state_update["final"]:
                final_delivered_report = state_update["final"]
            print("-" * 50)

    print("\n" + "=" * 60)
    print(final_delivered_report)
    print("=" * 60)