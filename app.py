import os
import re
import time
import glob
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import streamlit as st
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

from research_web import create_research_agent, WebSearchState, get_llm
from tools import (
    ALL_RESEARCH_TOOLS,
    FILE_OPS_TOOLS,
    DOC_GENERATION_TOOLS,
    TEXT_NLP_TOOLS,
    GEO_WEATHER_TOOLS,
)

# Page configuration
st.set_page_config(
    page_title="DeepResearch Agent | by x-estal nitish",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Liquid Glass Theme & Styling (Glassmorphism 3.0)
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Global Background & Atmospheric Mesh */
    .stApp {
        background: 
            radial-gradient(at 0% 0%, rgba(14, 165, 233, 0.16) 0px, transparent 45%),
            radial-gradient(at 100% 0%, rgba(168, 85, 247, 0.18) 0px, transparent 45%),
            radial-gradient(at 50% 50%, rgba(99, 102, 241, 0.08) 0px, transparent 55%),
            radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.14) 0px, transparent 45%),
            radial-gradient(at 0% 100%, rgba(236, 72, 153, 0.12) 0px, transparent 45%),
            #070B14;
        color: #F8FAFC;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar Glass Styling */
    section[data-testid="stSidebar"] {
        background: rgba(10, 16, 30, 0.72) !important;
        backdrop-filter: blur(24px) saturate(190%) !important;
        -webkit-backdrop-filter: blur(24px) saturate(190%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.4) !important;
    }

    /* Liquid Glass Cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(20px) saturate(180%);
        -webkit-backdrop-filter: blur(20px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 12px 36px 0 rgba(0, 0, 0, 0.35), inset 0 1px 0 0 rgba(255, 255, 255, 0.08);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .glass-card:hover {
        border-color: rgba(56, 189, 248, 0.35);
        box-shadow: 0 16px 42px 0 rgba(14, 165, 233, 0.15), inset 0 1px 0 0 rgba(255, 255, 255, 0.15);
        transform: translateY(-2px);
    }

    /* Hero Branding Header */
    .hero-container {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.015) 100%);
        backdrop-filter: blur(28px) saturate(200%);
        -webkit-backdrop-filter: blur(28px) saturate(200%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 24px;
        padding: 28px 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.15);
    }
    .hero-container::before {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 220px;
        height: 220px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.3) 0%, transparent 70%);
        filter: blur(35px);
        pointer-events: none;
    }
    .hero-title {
        font-size: 2.6rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #FFFFFF 0%, #38BDF8 40%, #818CF8 75%, #C084FC 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        line-height: 1.15;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 16px;
        max-width: 850px;
    }

    /* Creator Pill & Socials */
    .creator-bar {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 14px;
        padding-top: 16px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .creator-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(14, 165, 233, 0.12);
        border: 1px solid rgba(56, 189, 248, 0.3);
        color: #E0F2FE;
        padding: 6px 14px;
        border-radius: 9999px;
        font-size: 0.9rem;
        font-weight: 600;
        box-shadow: 0 0 16px rgba(56, 189, 248, 0.2);
    }
    .social-link {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #F1F5F9 !important;
        text-decoration: none !important;
        padding: 6px 16px;
        border-radius: 9999px;
        font-size: 0.86rem;
        font-weight: 500;
        backdrop-filter: blur(12px);
        transition: all 0.25s ease;
    }
    .social-link:hover {
        background: rgba(255, 255, 255, 0.12);
        border-color: rgba(255, 255, 255, 0.28);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.35);
    }
    .social-link.insta:hover {
        background: linear-gradient(135deg, rgba(225, 48, 108, 0.25), rgba(253, 29, 29, 0.25));
        border-color: #E1306C;
        box-shadow: 0 0 18px rgba(225, 48, 108, 0.35);
    }
    .social-link.linkedin:hover {
        background: rgba(10, 102, 194, 0.25);
        border-color: #0A66C2;
        box-shadow: 0 0 18px rgba(10, 102, 194, 0.35);
    }

    /* Creator Sidebar Profile Card */
    .sidebar-profile {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.08);
    }
    .sidebar-avatar {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0EA5E9, #8B5CF6, #EC4899);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 26px;
        font-weight: 800;
        color: white;
        margin-bottom: 10px;
        box-shadow: 0 0 20px rgba(14, 165, 233, 0.45);
        border: 2px solid rgba(255, 255, 255, 0.3);
    }
    .sidebar-name {
        font-size: 1.15rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 2px;
    }
    .sidebar-role {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-bottom: 12px;
        letter-spacing: 0.02em;
    }

    /* Liquid Glass Buttons */
    .stButton > button {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.85) 0%, rgba(99, 102, 241, 0.85) 50%, rgba(168, 85, 247, 0.85) 100%) !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        backdrop-filter: blur(12px) !important;
        box-shadow: 0 8px 24px rgba(14, 165, 233, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.01) !important;
        box-shadow: 0 14px 32px rgba(14, 165, 233, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.4) !important;
        border-color: rgba(255, 255, 255, 0.45) !important;
    }

    /* Tabs Custom Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.035) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 6px !important;
        gap: 6px !important;
        backdrop-filter: blur(16px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.2) !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px !important;
        color: #94A3B8 !important;
        font-weight: 500 !important;
        padding: 8px 18px !important;
        transition: all 0.25s ease !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.12), rgba(255, 255, 255, 0.04)) !important;
        color: #F8FAFC !important;
        font-weight: 700 !important;
        border: 1px solid rgba(255, 255, 255, 0.18) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3) !important;
    }

    /* Inputs, Textareas, Selectboxes */
    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {
        background: rgba(15, 23, 42, 0.55) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 12px !important;
        backdrop-filter: blur(14px) !important;
        color: #F8FAFC !important;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }
    .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 18px rgba(56, 189, 248, 0.3), inset 0 2px 4px rgba(0, 0, 0, 0.3) !important;
    }

    /* Expanders */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.025) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2) !important;
        margin-bottom: 12px !important;
    }

    /* Badges */
    .badge-pass {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.35);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 0 14px rgba(16, 185, 129, 0.2);
    }
    .badge-retry {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.35);
        padding: 4px 14px;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 0 14px rgba(245, 158, 11, 0.2);
    }
    .tool-tag {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #E2E8F0;
        padding: 4px 10px;
        border-radius: 8px;
        font-size: 0.82rem;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
        font-family: monospace;
    }

    /* Footer Liquid Glass Bar */
    .liquid-footer {
        margin-top: 50px;
        padding: 24px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px 20px 0 0;
        color: #94A3B8;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Helper to retrieve API key from st.secrets or os.environ
def get_env_or_secret(key_name: str, default: str = "") -> str:
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name])
    except Exception:
        pass
    return os.environ.get(key_name, default)

# Ensure workspaces & vault exist
VAULT_DIR = Path("research_vault")
UPLOADS_DIR = Path("uploads")
VAULT_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------
# Sidebar: Creator Identity & Engine Settings
# ---------------------------------------------------------
with st.sidebar:
    # Creator Profile Card
    st.markdown("""
    <div class="sidebar-profile">
        <div class="sidebar-avatar">💎</div>
        <div class="sidebar-name">x-estal nitish</div>
        <div class="sidebar-role">AI Architect & Systems Engineer</div>
        <div style="display:flex; justify-content:center; gap:8px; margin-top:10px;">
            <a href="https://instagram.com/x_estal_nitish" target="_blank" class="social-link insta">
                📸 @x_estal_nitish
            </a>
            <a href="https://www.linkedin.com/in/nitish-kumar-33714642b" target="_blank" class="social-link linkedin">
                💼 LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Engine Settings")

    provider = st.selectbox(
        "LLM Provider",
        options=["Google Gemini", "Groq", "OpenAI"],
        index=0,
        help="Select the AI reasoning backend. Google Gemini works seamlessly with free-tier keys."
    )

    # Provider specific default models & keys
    if provider == "Google Gemini":
        model_options = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        default_key = get_env_or_secret("GOOGLE_API_KEY")
        key_label = "Google API Key"
        key_help = "Get your key from https://aistudio.google.com/"
    elif provider == "Groq":
        model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "gemma2-9b-it"
        ]
        default_key = get_env_or_secret("GROQ_API_KEY")
        key_label = "Groq API Key"
        key_help = "Get your key from https://console.groq.com/keys"
    else:
        model_options = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
        default_key = get_env_or_secret("OPENAI_API_KEY")
        key_label = "OpenAI API Key"
        key_help = "Get your key from https://platform.openai.com/api-keys"

    selected_model = st.selectbox("Model Name", options=model_options)
    api_key_input = st.text_input(
        key_label,
        value=default_key,
        type="password",
        help=key_help
    )

    if api_key_input:
        st.caption("🟢 API Key active")
    else:
        st.caption("⚠️ API Key required to run the agent")

    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower values yield more factual, rigorous grounding."
    )

    st.markdown("---")
    st.markdown("### 🎯 Investigation Scope")

    search_depth = st.selectbox(
        "Search Depth",
        options=["basic", "medium", "advanced"],
        index=1,
        help="Controls number of tool cycles and depth of queries."
    )

    output_type = st.selectbox(
        "Output Format",
        options=["deep_dive", "summary", "bullet_points"],
        index=0,
        help="Format structure produced by the Lead Delivery Analyst."
    )

    export_format = st.selectbox(
        "Export Deliverable",
        options=["markdown", "pdf", "docx"],
        index=0,
        help="Document format automatically written to the research vault."
    )

    st.markdown("---")
    st.markdown("### 🧰 Bound Toolsets")
    total_tools = len(ALL_RESEARCH_TOOLS) + len(FILE_OPS_TOOLS) + len(DOC_GENERATION_TOOLS) + len(TEXT_NLP_TOOLS) + len(GEO_WEATHER_TOOLS)
    st.write(f"**Total Registered Tools:** `{total_tools}`")
    st.markdown(f"""
    - 🌐 **Web & Academic**: `{len(ALL_RESEARCH_TOOLS)}` tools
    - 📁 **File & PDF Ops**: `{len(FILE_OPS_TOOLS)}` tools
    - 📑 **Doc Generation**: `{len(DOC_GENERATION_TOOLS)}` tools
    - 🔤 **Text NLP**: `{len(TEXT_NLP_TOOLS)}` tools
    - ⛅ **Geo & Weather**: `{len(GEO_WEATHER_TOOLS)}` tools
    """)

# ---------------------------------------------------------
# Hero Liquid Glass Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🧭 DeepResearch Agent</div>
    <div class="hero-subtitle">
        Autonomous Multi-Step Investigation, Local Document Reading, Grounding Audit & Automated Deliverable Generation powered by LangGraph.
    </div>
    <div class="creator-bar">
        <span class="creator-badge">⚡ Created by <strong>x-estal nitish</strong></span>
        <a href="https://instagram.com/x_estal_nitish" target="_blank" class="social-link insta">
            📸 Instagram: @x_estal_nitish
        </a>
        <a href="https://www.linkedin.com/in/nitish-kumar-33714642b" target="_blank" class="social-link linkedin">
            💼 LinkedIn Profile
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

tabs = st.tabs(["🔬 New Research Mission", "📚 Research Vault", "🛠️ Tool Catalog", "🚀 Deployment Guide"])

# ---------------------------------------------------------
# Tab 1: New Research Mission
# ---------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown("**Quick Query Presets:**")
        p_col1, p_col2, p_col3 = st.columns(3)
        if p_col1.button("🤖 Computer Use Agents"):
            st.session_state["query_input"] = (
                "Analyze GUI grounding architectures in Computer Use Agents (OmniParser vs Set-of-Marks). "
                "Compare latency, action space predictability, and token overhead."
            )
        if p_col2.button("⚛️ Quantum Computing 2026"):
            st.session_state["query_input"] = (
                "Investigate recent breakthroughs in topological qubits and quantum error correction in 2025-2026. "
                "Highlight key metrics, physical qubit overhead, and commercial roadmap."
            )
        if p_col3.button("🧠 Long Context vs RAG"):
            st.session_state["query_input"] = (
                "Deep dive comparison: 1M+ token context windows vs Agentic Graph RAG for enterprise documentation retrieval. "
                "Evaluate retrieval accuracy, cost per query, and hallucination rates."
            )

        query = st.text_area(
            "Enter Research Objective or Investigation Query:",
            value=st.session_state.get("query_input", ""),
            height=130,
            placeholder="e.g. Compare modern vector search indexes (HNSW vs DiskANN) for billion-scale embeddings..."
        )

    with col2:
        st.markdown("**Attach Local Documents:**")
        uploaded_files = st.file_uploader(
            "Upload reference PDFs / files",
            type=["pdf", "txt", "md", "docx", "csv"],
            accept_multiple_files=True,
            help="Files are saved locally and passed to the agent's file reader tools before web queries run."
        )

        attached_paths = []
        if uploaded_files:
            for up_file in uploaded_files:
                save_path = UPLOADS_DIR / up_file.name
                with open(save_path, "wb") as f:
                    f.write(up_file.getbuffer())
                attached_paths.append(str(save_path))
            st.success(f"Attached {len(attached_paths)} file(s)")

    start_button = st.button("🚀 Launch Autonomous Research", type="primary", use_container_width=True)

    if start_button:
        if not query.strip():
            st.error("Please enter a research query before starting.")
        elif not api_key_input.strip():
            st.error(f"Please enter your {key_label} in the sidebar to proceed.")
        else:
            # Set key into environment temporarily for tool calls if needed
            if provider == "Google Gemini":
                os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
            elif provider == "Groq":
                os.environ["GROQ_API_KEY"] = api_key_input.strip()
            elif provider == "OpenAI":
                os.environ["OPENAI_API_KEY"] = api_key_input.strip()

            st.markdown("---")
            st.markdown("### 🔄 Execution Progress & Agent Thought Stream")

            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()
            log_container = st.container()

            # Initialize Agent
            with st.spinner("Compiling LangGraph State Machine & Binding Tools..."):
                try:
                    agent = create_research_agent(
                        provider=provider,
                        model_name=selected_model,
                        api_key=api_key_input.strip(),
                        temperature=temperature
                    )
                except Exception as e:
                    st.error(f"Failed to initialize agent: {e}")
                    agent = None

            if agent:
                payload = {
                    "query": query.strip(),
                    "attached_files": attached_paths,
                    "search_depth": search_depth,
                    "output_type": output_type,
                    "export_format": export_format,
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

                step_weights = {
                    "decomposer": 0.15,
                    "investigator": 0.35,
                    "investigation_tools": 0.45,
                    "synthesizer": 0.60,
                    "reporter": 0.75,
                    "auditor": 0.85,
                    "delivery_initiator": 0.90,
                    "delivery_agent": 0.95,
                    "delivery_tools": 0.98,
                    "finalize": 1.0
                }

                node_icons = {
                    "decomposer": "🎯 Decomposer",
                    "investigator": "🕵️ Investigator",
                    "investigation_tools": "🛠️ Tool Executor",
                    "synthesizer": "🧬 Synthesizer",
                    "reporter": "📝 Lead Reporter",
                    "auditor": "🛡️ Fact-Check Auditor",
                    "delivery_initiator": "📦 Delivery Initiator",
                    "delivery_agent": "🚚 Delivery Agent",
                    "delivery_tools": "💾 Storage Tools",
                    "finalize": "🏁 Final Deliverable"
                }

                final_report_text = ""
                research_notes_text = ""
                audit_passed_flag = False
                audit_feedback_text = ""
                start_time = time.time()

                try:
                    for step in agent.stream(payload):
                        for node_name, state_update in step.items():
                            progress_val = step_weights.get(node_name, 0.5)
                            progress_bar.progress(progress_val)
                            title_display = node_icons.get(node_name, f"Node: {node_name}")
                            status_placeholder.info(f"Currently active: **{title_display}**...")

                            with log_container:
                                with st.expander(f"{title_display} completed", expanded=(node_name in ["decomposer", "auditor", "finalize"])):
                                    if "generated_search_queries" in state_update:
                                        st.markdown(f"**Targeted Queries Formulated:**\n`{state_update['generated_search_queries']}`")

                                    if "messages" in state_update:
                                        last_msg = state_update["messages"][-1]
                                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                            st.markdown("**Invoking Tools:**")
                                            for tc in last_msg.tool_calls:
                                                st.markdown(f"- `<{tc.get('name')}>` with args: `{tc.get('args')}`")

                                    if "research_notes" in state_update and state_update["research_notes"]:
                                        research_notes_text = state_update["research_notes"]
                                        st.markdown("**Extracted Factual Evidence Trail:**")
                                        st.text_area("Research Notes", research_notes_text, height=180)

                                    if "draft_report" in state_update and state_update["draft_report"]:
                                        st.markdown("**Draft Report Compiled** (awaiting grounding audit)")

                                    if "audit_passed" in state_update:
                                        audit_passed_flag = state_update["audit_passed"]
                                        audit_feedback_text = state_update.get("audit_feedback") or "PASS"
                                        if audit_passed_flag:
                                            st.markdown('<span class="badge-pass">✅ AUDIT PASSED: Grounded in Verified Data</span>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'<span class="badge-retry">⚠️ AUDIT CORRECTION REQUESTED: {audit_feedback_text}</span>', unsafe_allow_html=True)

                                    if "final" in state_update and state_update["final"]:
                                        final_report_text = state_update["final"]
                                        st.success("Research Mission Finalized Successfully!")

                    total_elapsed = round(time.time() - start_time, 2)
                    progress_bar.progress(1.0)
                    status_placeholder.success(f"Mission Completed in {total_elapsed}s!")

                    # Render Final Result Section
                    if final_report_text:
                        st.markdown("---")
                        st.markdown("## 📑 Finalized Analytical Deliverable")
                        st.markdown(final_report_text)

                        # Download button
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        clean_q = re.sub(r'[^a-zA-Z0-9_\- ]', '', query[:25]).strip().replace(' ', '_').lower()
                        md_filename = f"report_{clean_q}_{timestamp_str}.md"

                        st.download_button(
                            label="📥 Download Verified Markdown Report",
                            data=final_report_text,
                            file_name=md_filename,
                            mime="text/markdown",
                            use_container_width=True
                        )

                except Exception as ex:
                    st.error(f"Error during agent execution: {str(ex)}")

# ---------------------------------------------------------
# Tab 2: Research Vault
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("### 🗄️ Persisted Deliverables Vault")
    st.markdown("All generated reports and exports are safely stored in `research_vault/`.")

    saved_files = list(VAULT_DIR.glob("*.*"))
    if not saved_files:
        st.info("No saved research reports yet. Launch a mission in the first tab to populate the vault.")
    else:
        st.write(f"**Found {len(saved_files)} persisted file(s):**")
        for fpath in sorted(saved_files, key=os.path.getmtime, reverse=True):
            col_icon, col_info, col_action = st.columns([1, 6, 3])
            ext = fpath.suffix.lower()
            icon = "📄" if ext == ".md" else ("📕" if ext == ".pdf" else "📘")

            with col_icon:
                st.markdown(f"### {icon}")
            with col_info:
                st.markdown(f"**{fpath.name}**")
                file_size_kb = round(os.path.getsize(fpath) / 1024, 2)
                mod_time = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Size: {file_size_kb} KB | Created: {mod_time}")
            with col_action:
                with open(fpath, "rb") as fl:
                    bytes_data = fl.read()
                st.download_button(
                    label=f"Download {fpath.name}",
                    data=bytes_data,
                    file_name=fpath.name,
                    key=f"dl_{fpath.name}"
                )

            with st.expander(f"Preview {fpath.name}"):
                if ext in [".md", ".txt"]:
                    try:
                        content = fpath.read_text(encoding="utf-8")
                        st.markdown(content)
                    except Exception:
                        st.text(fpath.read_text(errors="ignore"))
                else:
                    st.info("Binary document format. Use the download button to view.")
            st.markdown("---")

# ---------------------------------------------------------
# Tab 3: Tool Catalog
# ---------------------------------------------------------
with tabs[2]:
    st.markdown("### 🛠️ Agent Tools Suite & Capabilities")
    st.markdown("The agent has autonomous access to tools across 5 specialized domains:")

    t1, t2 = st.columns(2)
    with t1:
        st.markdown("#### 🌐 Research & Web Intelligence")
        for t in ALL_RESEARCH_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

        st.markdown("#### 📁 File & Workspace Operations")
        for t in FILE_OPS_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

    with t2:
        st.markdown("#### 📑 Document Generation")
        for t in DOC_GENERATION_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

        st.markdown("#### 🔤 Text & NLP")
        for t in TEXT_NLP_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

        st.markdown("#### ⛅ Geo & Weather")
        for t in GEO_WEATHER_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

# ---------------------------------------------------------
# Tab 4: Deployment Guide
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("### ☁️ Streamlit Community Cloud Deployment Guide")
    st.markdown("""
    This application is fully prepped for 1-click deployment on [Streamlit Community Cloud](https://streamlit.io/cloud).

    #### 1. Push to GitHub
    ```bash
    git init
    git add .
    git commit -m "Deploy DeepResearch Agent to Streamlit Cloud"
    git branch -M main
    git remote add origin https://github.com/<your-username>/<your-repo-name>.git
    git push -u origin main
    ```

    #### 2. Deploy on Streamlit Cloud
    1. Log in to **[share.streamlit.io](https://share.streamlit.io/)**.
    2. Click **New app** and select your GitHub repository.
    3. Set **Main file path** to `app.py`.
    4. Click **Advanced Settings** -> **Secrets** and paste your API keys:
    ```toml
    GOOGLE_API_KEY = "your-google-gemini-key"
    GROQ_API_KEY = "your-groq-key"
    OPENAI_API_KEY = "your-openai-key"
    ```
    5. Click **Deploy**! 🚀
    """)

# ---------------------------------------------------------
# Liquid Glass Footer
# ---------------------------------------------------------
st.markdown("""
<div class="liquid-footer">
    <div>⚡ <strong>DeepResearch Agent</strong> &mdash; Engineered by <strong>x-estal nitish</strong></div>
    <div style="margin-top: 8px; display: flex; justify-content: center; gap: 16px; flex-wrap: wrap;">
        <a href="https://instagram.com/x_estal_nitish" target="_blank" style="color: #38BDF8; text-decoration: none;">
            📸 Instagram: @x_estal_nitish
        </a>
        <span>&bull;</span>
        <a href="https://www.linkedin.com/in/nitish-kumar-33714642b" target="_blank" style="color: #818CF8; text-decoration: none;">
            💼 LinkedIn: Nitish Kumar
        </a>
    </div>
    <div style="margin-top: 6px; font-size: 0.8rem; color: #64748B;">
        Autonomous LangGraph Agent Architecture &bull; Grounded Fact-Checking &bull; Multi-Tool Intelligence
    </div>
</div>
""", unsafe_allow_html=True)
