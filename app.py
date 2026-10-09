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
    page_title="DeepResearch Intelligence | by x-estal nitish",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Cyber-Luxe Liquid Glass Styling (Glassmorphism 3.5)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Obsidian Liquid Canvas */
    .stApp {
        background: 
            radial-gradient(ellipse 90% 60% at 50% -20%, rgba(56, 189, 248, 0.22) 0%, transparent 70%),
            radial-gradient(ellipse 70% 50% at 100% 30%, rgba(168, 85, 247, 0.2) 0%, transparent 60%),
            radial-gradient(ellipse 80% 60% at 0% 70%, rgba(236, 72, 153, 0.16) 0%, transparent 65%),
            radial-gradient(ellipse 90% 70% at 70% 100%, rgba(14, 165, 233, 0.18) 0%, transparent 60%),
            #060913;
        color: #F8FAFC;
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Ultra-Frosted Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(8, 14, 28, 0.75) !important;
        backdrop-filter: blur(28px) saturate(210%) !important;
        -webkit-backdrop-filter: blur(28px) saturate(210%) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08) !important;
        box-shadow: 10px 0 35px rgba(0, 0, 0, 0.5) !important;
    }

    /* Executive Hero Container */
    .hero-glass-box {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06) 0%, rgba(255, 255, 255, 0.015) 100%);
        backdrop-filter: blur(30px) saturate(220%);
        -webkit-backdrop-filter: blur(30px) saturate(220%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 26px;
        padding: 32px 36px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 
            0 24px 60px rgba(0, 0, 0, 0.6),
            inset 0 1px 0 rgba(255, 255, 255, 0.2),
            inset 0 0 40px rgba(56, 189, 248, 0.05);
    }
    .hero-glass-box::before {
        content: '';
        position: absolute;
        top: -80px;
        right: -80px;
        width: 260px;
        height: 260px;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.35) 0%, rgba(168, 85, 247, 0.2) 50%, transparent 75%);
        filter: blur(40px);
        pointer-events: none;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(52, 211, 153, 0.35);
        color: #6EE7B7;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 14px;
        border-radius: 9999px;
        margin-bottom: 12px;
        box-shadow: 0 0 15px rgba(16, 185, 129, 0.2);
    }
    .status-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 8px #10B981;
        animation: pulseDot 2s infinite;
    }
    @keyframes pulseDot {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.85); }
    }
    .hero-title {
        font-size: 2.85rem;
        font-weight: 800;
        letter-spacing: -0.04em;
        line-height: 1.15;
        background: linear-gradient(135deg, #FFFFFF 20%, #7DD3FC 55%, #C084FC 85%, #F472B6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .hero-desc {
        font-size: 1.08rem;
        color: #94A3B8;
        line-height: 1.6;
        max-width: 860px;
        margin-bottom: 20px;
    }

    /* Creator Profile & Social Pill Bar */
    .author-pill-bar {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 12px;
        padding-top: 18px;
        border-top: 1px solid rgba(255, 255, 255, 0.08);
    }
    .author-chip {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: linear-gradient(135deg, rgba(56, 189, 248, 0.15), rgba(168, 85, 247, 0.12));
        border: 1px solid rgba(56, 189, 248, 0.35);
        color: #F0F9FF;
        padding: 7px 16px;
        border-radius: 9999px;
        font-size: 0.92rem;
        font-weight: 700;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.2);
    }
    .social-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.14);
        color: #F8FAFC !important;
        text-decoration: none !important;
        padding: 7px 18px;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 600;
        backdrop-filter: blur(14px);
        transition: all 0.28s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    .social-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
    }
    .social-btn.insta:hover {
        background: linear-gradient(135deg, rgba(225, 48, 108, 0.35), rgba(253, 29, 29, 0.3));
        border-color: #F43F5E;
        box-shadow: 0 0 22px rgba(244, 63, 94, 0.4);
    }
    .social-btn.linkedin:hover {
        background: linear-gradient(135deg, rgba(14, 165, 233, 0.35), rgba(10, 102, 194, 0.3));
        border-color: #38BDF8;
        box-shadow: 0 0 22px rgba(56, 189, 248, 0.4);
    }

    /* Metric Quick Bar */
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
        gap: 14px;
        margin-bottom: 24px;
    }
    .metric-pill {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 16px 20px;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255, 255, 255, 0.07);
        transition: all 0.25s ease;
    }
    .metric-pill:hover {
        border-color: rgba(56, 189, 248, 0.4);
        transform: translateY(-2px);
        box-shadow: 0 12px 30px rgba(14, 165, 233, 0.15);
    }
    .metric-num {
        font-size: 1.6rem;
        font-weight: 800;
        color: #F8FAFC;
        line-height: 1.2;
    }
    .metric-sub {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
        margin-top: 2px;
    }

    /* Sidebar Developer Profile */
    .dev-profile-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.06), rgba(255, 255, 255, 0.015));
        border: 1px solid rgba(255, 255, 255, 0.13);
        border-radius: 20px;
        padding: 22px 18px;
        margin-bottom: 24px;
        text-align: center;
        box-shadow: 0 14px 35px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .dev-avatar-ring {
        width: 72px;
        height: 72px;
        border-radius: 50%;
        background: linear-gradient(135deg, #0EA5E9, #8B5CF6, #EC4899);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 30px;
        color: white;
        margin-bottom: 12px;
        box-shadow: 0 0 25px rgba(14, 165, 233, 0.5);
        border: 2px solid rgba(255, 255, 255, 0.35);
    }
    .dev-name {
        font-size: 1.25rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #F8FAFC;
    }
    .dev-title {
        font-size: 0.82rem;
        color: #94A3B8;
        font-weight: 500;
        margin-bottom: 14px;
    }

    /* Liquid Glass Action Button */
    .stButton > button {
        background: linear-gradient(135deg, #0EA5E9 0%, #6366F1 50%, #A855F7 100%) !important;
        color: #FFFFFF !important;
        font-size: 1rem !important;
        font-weight: 700 !important;
        letter-spacing: 0.02em !important;
        border: 1px solid rgba(255, 255, 255, 0.35) !important;
        border-radius: 14px !important;
        padding: 14px 28px !important;
        backdrop-filter: blur(14px) !important;
        box-shadow: 0 10px 30px rgba(14, 165, 233, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.35) !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }
    .stButton > button:hover {
        transform: translateY(-3px) scale(1.008) !important;
        box-shadow: 0 18px 45px rgba(99, 102, 241, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
        border-color: rgba(255, 255, 255, 0.6) !important;
    }

    /* Custom Modern Segmented Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 18px !important;
        padding: 6px !important;
        gap: 8px !important;
        backdrop-filter: blur(20px) !important;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25) !important;
        margin-bottom: 22px !important;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 12px !important;
        color: #94A3B8 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 10px 24px !important;
        transition: all 0.25s ease !important;
        border: none !important;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15) 0%, rgba(255, 255, 255, 0.04) 100%) !important;
        color: #F8FAFC !important;
        border: 1px solid rgba(255, 255, 255, 0.22) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.18) !important;
    }

    /* Inputs, Textareas, Selectboxes */
    .stTextInput > div > div, .stTextArea > div > div, .stSelectbox > div > div {
        background: rgba(11, 18, 36, 0.65) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 14px !important;
        backdrop-filter: blur(16px) !important;
        color: #F8FAFC !important;
        box-shadow: inset 0 2px 5px rgba(0, 0, 0, 0.4) !important;
    }
    .stTextInput > div > div:focus-within, .stTextArea > div > div:focus-within {
        border-color: #38BDF8 !important;
        box-shadow: 0 0 20px rgba(56, 189, 248, 0.35), inset 0 2px 5px rgba(0, 0, 0, 0.4) !important;
    }

    /* Expanders & Deliverable Cards */
    div[data-testid="stExpander"] {
        background: rgba(255, 255, 255, 0.03) !important;
        border: 1px solid rgba(255, 255, 255, 0.09) !important;
        border-radius: 16px !important;
        backdrop-filter: blur(18px) !important;
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.25) !important;
        margin-bottom: 14px !important;
    }

    /* Badges */
    .badge-pass {
        background: rgba(16, 185, 129, 0.16);
        color: #34D399;
        border: 1px solid rgba(52, 211, 153, 0.4);
        padding: 5px 16px;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.25);
    }
    .badge-retry {
        background: rgba(245, 158, 11, 0.16);
        color: #FBBF24;
        border: 1px solid rgba(251, 191, 36, 0.4);
        padding: 5px 16px;
        border-radius: 9999px;
        font-size: 0.88rem;
        font-weight: 700;
        display: inline-block;
        box-shadow: 0 0 16px rgba(245, 158, 11, 0.25);
    }

    /* Vault Report Item */
    .vault-card {
        background: rgba(255, 255, 255, 0.035);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 18px 22px;
        margin-bottom: 14px;
        box-shadow: 0 8px 24px rgba(0,0,0,0.3);
        transition: all 0.25s ease;
    }
    .vault-card:hover {
        border-color: rgba(56, 189, 248, 0.4);
        box-shadow: 0 12px 32px rgba(14, 165, 233, 0.15);
        transform: translateY(-2px);
    }

    /* Footer */
    .liquid-footer {
        margin-top: 60px;
        padding: 30px 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(25px);
        -webkit-backdrop-filter: blur(25px);
        border-top: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 24px 24px 0 0;
        color: #94A3B8;
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
    st.markdown("""
    <div class="dev-profile-card">
        <div class="dev-avatar-ring">⚡</div>
        <div class="dev-name">x-estal nitish</div>
        <div class="dev-title">AI Research Architect</div>
        <div style="display:flex; justify-content:center; gap:8px; margin-top:8px;">
            <a href="https://instagram.com/x_estal_nitish" target="_blank" class="social-btn insta">
                📸 @x_estal_nitish
            </a>
            <a href="https://www.linkedin.com/in/nitish-kumar-33714642b" target="_blank" class="social-btn linkedin">
                💼 LinkedIn
            </a>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ Engine Parameters")

    provider = st.selectbox(
        "Inference Engine",
        options=["Google Gemini", "Groq", "OpenAI"],
        index=0,
        help="Select reasoning engine. Google Gemini is recommended with the active key."
    )

    if provider == "Google Gemini":
        model_options = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
        key_label = "Google API Key"
        key_help = "Get your free API key from https://aistudio.google.com/"
        placeholder_key = "Paste your Google Gemini API key here..."
    elif provider == "Groq":
        model_options = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "deepseek-r1-distill-llama-70b",
            "gemma2-9b-it"
        ]
        key_label = "Groq API Key"
        key_help = "Get your API key from https://console.groq.com/keys"
        placeholder_key = "Paste your Groq API key here (gsk_...)"
    else:
        model_options = ["gpt-4o-mini", "gpt-4o", "o3-mini"]
        key_label = "OpenAI API Key"
        key_help = "Get your API key from https://platform.openai.com/api-keys"
        placeholder_key = "Paste your OpenAI API key here (sk-...)"

    selected_model = st.selectbox("Active Model", options=model_options)
    api_key_input = st.text_input(
        key_label,
        value="",
        type="password",
        placeholder=placeholder_key,
        help=key_help
    )

    if api_key_input.strip():
        st.caption("🟢 API Key entered")
    else:
        st.caption("⚠️ Please enter your API Key above")

    temperature = st.slider(
        "Reasoning Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Lower values enforce rigorous grounded factual verification."
    )

    st.markdown("---")
    st.markdown("### 🎯 Investigation Controls")

    search_depth = st.selectbox(
        "Investigation Depth",
        options=["basic", "medium", "advanced"],
        index=1,
        help="Controls number of autonomous exploration hops."
    )

    output_type = st.selectbox(
        "Report Architecture",
        options=["deep_dive", "summary", "bullet_points"],
        index=0,
        help="Delivery analyst structuring."
    )

    export_format = st.selectbox(
        "Vault Export Format",
        options=["markdown", "pdf", "docx"],
        index=0,
        help="Document format persisted to the research vault."
    )

    st.markdown("---")
    total_tools = len(ALL_RESEARCH_TOOLS) + len(FILE_OPS_TOOLS) + len(DOC_GENERATION_TOOLS) + len(TEXT_NLP_TOOLS) + len(GEO_WEATHER_TOOLS)
    st.markdown(f"**⚡ Neural Tool Count:** `{total_tools} Active`")
    st.caption("Integrated across Web, ArXiv, Wikipedia, Local Readers, NLP, and Doc Exporters.")

# ---------------------------------------------------------
# Hero Master Liquid Glass Header
# ---------------------------------------------------------
st.markdown("""
<div class="hero-glass-box">
    <div class="status-badge">
        <span class="status-dot"></span>
        Neural Agent System Active
    </div>
    <div class="hero-title">DEEP RESEARCH INTELLIGENCE</div>
    <div class="hero-desc">
        Next-generation autonomous research engine powered by LangGraph. Conducts multi-hop web retrieval, 
        inspects local documents & PDFs, enforces anti-hallucination fact audits, and persists verified reports.
    </div>
    <div class="author-pill-bar">
        <span class="author-chip">⚡ Engineered by <strong>x-estal nitish</strong></span>
        <a href="https://instagram.com/x_estal_nitish" target="_blank" class="social-btn insta">
            📸 Instagram: @x_estal_nitish
        </a>
        <a href="https://www.linkedin.com/in/nitish-kumar-33714642b" target="_blank" class="social-btn linkedin">
            💼 LinkedIn: Nitish Kumar
        </a>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Quick Metrics Grid
# ---------------------------------------------------------
st.markdown(f"""
<div class="metric-grid">
    <div class="metric-pill">
        <div class="metric-num">14+</div>
        <div class="metric-sub">🌐 Web, ArXiv & Wiki Engines</div>
    </div>
    <div class="metric-pill">
        <div class="metric-num">5+</div>
        <div class="metric-sub">📁 Local Document & PDF Parsers</div>
    </div>
    <div class="metric-pill">
        <div class="metric-num">100%</div>
        <div class="metric-sub">🛡️ Grounding Fact-Audit Loop</div>
    </div>
    <div class="metric-pill">
        <div class="metric-num">3 Formats</div>
        <div class="metric-sub">💾 Markdown, PDF & DOCX Vault</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Main Tabs (Deployment guide removed as requested)
# ---------------------------------------------------------
tabs = st.tabs([
    "⚡ Autonomous Research Studio", 
    "🗄️ Intelligence Vault", 
    "🧩 Neural Tools Matrix"
])

# ---------------------------------------------------------
# Tab 1: Autonomous Research Studio
# ---------------------------------------------------------
with tabs[0]:
    col_main, col_side = st.columns([3, 1])

    with col_main:
        st.markdown("**💡 Mission Inspiration Presets:**")
        p1, p2, p3 = st.columns(3)
        if p1.button("🤖 Computer Use Agents"):
            st.session_state["query_input"] = (
                "Analyze GUI grounding architectures in Computer Use Agents (OmniParser vs Set-of-Marks). "
                "Compare latency, action space predictability, and token overhead."
            )
        if p2.button("⚛️ Quantum Computing 2026"):
            st.session_state["query_input"] = (
                "Investigate recent breakthroughs in topological qubits and quantum error correction in 2025-2026. "
                "Highlight key metrics, physical qubit overhead, and commercial roadmap."
            )
        if p3.button("🧠 Long Context vs Graph RAG"):
            st.session_state["query_input"] = (
                "Deep dive comparison: 1M+ token context windows vs Agentic Graph RAG for enterprise documentation retrieval. "
                "Evaluate retrieval accuracy, cost per query, and hallucination rates."
            )

        query = st.text_area(
            "Research Objective / Investigation Prompt:",
            value=st.session_state.get("query_input", ""),
            height=140,
            placeholder="Describe your research goal in detail (e.g., 'Compare HNSW vs DiskANN vector indexes on 100M embeddings...')"
        )

    with col_side:
        st.markdown("**📂 Attach Local Material:**")
        uploaded_files = st.file_uploader(
            "Upload reference documents",
            type=["pdf", "txt", "md", "docx", "csv"],
            accept_multiple_files=True,
            help="Files are stored locally and inspected by FILE_OPS_TOOLS before initiating external web queries."
        )

        attached_paths = []
        if uploaded_files:
            for up_file in uploaded_files:
                save_path = UPLOADS_DIR / up_file.name
                with open(save_path, "wb") as f:
                    f.write(up_file.getbuffer())
                attached_paths.append(str(save_path))
            st.success(f"Attached {len(attached_paths)} document(s)")

    launch_col1, launch_col2 = st.columns([1, 1])
    with launch_col1:
        start_button = st.button("🚀 INITIATE AUTONOMOUS INVESTIGATION", type="primary", use_container_width=True)

    if start_button:
        if not query.strip():
            st.error("Please enter a research objective before launching.")
        elif not api_key_input.strip():
            st.error(f"Please provide your {key_label} in the sidebar.")
        else:
            if provider == "Google Gemini":
                os.environ["GOOGLE_API_KEY"] = api_key_input.strip()
            elif provider == "Groq":
                os.environ["GROQ_API_KEY"] = api_key_input.strip()
            elif provider == "OpenAI":
                os.environ["OPENAI_API_KEY"] = api_key_input.strip()

            st.markdown("---")
            st.markdown("### 🔄 Autonomous Investigation Stream")

            progress_bar = st.progress(0.0)
            status_placeholder = st.empty()
            log_container = st.container()

            # Initialize Agent
            with st.spinner("Compiling LangGraph State Machine & Orchestrating Nodes..."):
                try:
                    agent = create_research_agent(
                        provider=provider,
                        model_name=selected_model,
                        api_key=api_key_input.strip(),
                        temperature=temperature
                    )
                except Exception as e:
                    st.error(f"Initialization Error: {e}")
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
                    "decomposer": "🎯 Query Decomposer",
                    "investigator": "🕵️ Investigator Agent",
                    "investigation_tools": "🛠️ Tool Executor",
                    "synthesizer": "🧬 Research Synthesizer",
                    "reporter": "📝 Lead Reporter",
                    "auditor": "🛡️ Fact-Check Auditor",
                    "delivery_initiator": "📦 Delivery Initiator",
                    "delivery_agent": "🚚 Storage Agent",
                    "delivery_tools": "💾 Document Exporter",
                    "finalize": "🏁 Intelligence Vault Delivery"
                }

                final_report_text = ""
                start_time = time.time()

                try:
                    for step in agent.stream(payload):
                        for node_name, state_update in step.items():
                            progress_val = step_weights.get(node_name, 0.5)
                            progress_bar.progress(progress_val)
                            title_display = node_icons.get(node_name, f"Node: {node_name}")
                            status_placeholder.info(f"Active Node: **{title_display}**...")

                            with log_container:
                                with st.expander(f"{title_display} completed", expanded=(node_name in ["decomposer", "auditor", "finalize"])):
                                    if "generated_search_queries" in state_update:
                                        st.markdown(f"**Targeted Search Queries:**\n`{state_update['generated_search_queries']}`")

                                    if "messages" in state_update:
                                        last_msg = state_update["messages"][-1]
                                        if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                                            st.markdown("**Dispatched Tool Invocations:**")
                                            for tc in last_msg.tool_calls:
                                                st.markdown(f"- `<{tc.get('name')}>` with parameters: `{tc.get('args')}`")

                                    if "research_notes" in state_update and state_update["research_notes"]:
                                        st.markdown("**Synthesized Evidence Trail:**")
                                        st.text_area("Research Notes", state_update["research_notes"], height=170)

                                    if "draft_report" in state_update and state_update["draft_report"]:
                                        st.markdown("**Draft Analytical Report Formulated** (Passing to Auditor)")

                                    if "audit_passed" in state_update:
                                        passed = state_update["audit_passed"]
                                        fb = state_update.get("audit_feedback") or "PASS"
                                        if passed:
                                            st.markdown('<span class="badge-pass">✅ GROUNDING AUDIT PASSED: 100% Verified Evidence</span>', unsafe_allow_html=True)
                                        else:
                                            st.markdown(f'<span class="badge-retry">⚠️ RE-EVALUATION REQUIRED: {fb}</span>', unsafe_allow_html=True)

                                    if "final" in state_update and state_update["final"]:
                                        final_report_text = state_update["final"]
                                        st.success("Investigation Finalized & Stored in Vault!")

                    total_elapsed = round(time.time() - start_time, 2)
                    progress_bar.progress(1.0)
                    status_placeholder.success(f"Investigation Complete in {total_elapsed}s!")

                    if final_report_text:
                        st.markdown("---")
                        st.markdown("## 📑 Verified Analytical Deliverable")
                        st.markdown(final_report_text)

                        # Download button
                        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                        clean_q = re.sub(r'[^a-zA-Z0-9_\- ]', '', query[:25]).strip().replace(' ', '_').lower()
                        md_filename = f"report_{clean_q}_{timestamp_str}.md"

                        st.download_button(
                            label="📥 Download Verified Markdown Deliverable",
                            data=final_report_text,
                            file_name=md_filename,
                            mime="text/markdown",
                            use_container_width=True
                        )

                except Exception as ex:
                    st.error(f"Execution Error: {str(ex)}")

# ---------------------------------------------------------
# Tab 2: Intelligence Vault
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("### 🗄️ Persisted Intelligence Vault")
    st.markdown("All completed research missions and generated documents are stored permanently in `research_vault/`.")

    saved_files = list(VAULT_DIR.glob("*.*"))
    if not saved_files:
        st.info("The intelligence vault is currently empty. Run an investigation in the studio to populate it.")
    else:
        st.write(f"**Found {len(saved_files)} persisted deliverable(s):**")
        for fpath in sorted(saved_files, key=os.path.getmtime, reverse=True):
            ext = fpath.suffix.lower()
            icon = "📄" if ext == ".md" else ("📕" if ext == ".pdf" else "📘")

            c_icon, c_info, c_action = st.columns([1, 6, 3])
            with c_icon:
                st.markdown(f"## {icon}")
            with c_info:
                st.markdown(f"**{fpath.name}**")
                file_size_kb = round(os.path.getsize(fpath) / 1024, 2)
                mod_time = datetime.fromtimestamp(os.path.getmtime(fpath)).strftime("%Y-%m-%d %H:%M:%S")
                st.caption(f"Size: {file_size_kb} KB &bull; Generated: {mod_time}")
            with c_action:
                with open(fpath, "rb") as fl:
                    bytes_data = fl.read()
                st.download_button(
                    label=f"Download {fpath.name}",
                    data=bytes_data,
                    file_name=fpath.name,
                    key=f"vault_dl_{fpath.name}"
                )

            with st.expander(f"Preview {fpath.name}"):
                if ext in [".md", ".txt"]:
                    try:
                        content = fpath.read_text(encoding="utf-8")
                        st.markdown(content)
                    except Exception:
                        st.text(fpath.read_text(errors="ignore"))
                else:
                    st.info("Binary format. Use the download button to inspect.")
            st.markdown("---")

# ---------------------------------------------------------
# Tab 3: Neural Tools Matrix
# ---------------------------------------------------------
with tabs[2]:
    st.markdown("### 🧩 Neural Tool Catalog & Registry")
    st.markdown("The autonomous agent dispatches tool calls dynamically from 5 specialized capability clusters:")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        st.markdown("#### 🌐 Web & Academic Intelligence")
        for t in ALL_RESEARCH_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

        st.markdown("#### 📁 File Ops & PDF Readers")
        for t in FILE_OPS_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

    with col_t2:
        st.markdown("#### 📑 Document Exporters")
        for t in DOC_GENERATION_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

        st.markdown("#### 🔤 NLP & Text Extraction")
        for t in TEXT_NLP_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

        st.markdown("#### ⛅ Environmental & Weather")
        for t in GEO_WEATHER_TOOLS:
            doc = (t.description or "No description").strip().splitlines()[0]
            st.markdown(f"- **`{t.name}`**: {doc}")

# ---------------------------------------------------------
# Liquid Glass Footer
# ---------------------------------------------------------
st.markdown("""
<div class="liquid-footer">
    <div style="font-size: 1.05rem; font-weight: 700; color: #F8FAFC;">
        ⚡ <strong>DeepResearch Agent</strong> &mdash; Engineered with pride by <strong>x-estal nitish</strong>
    </div>
    <div style="margin-top: 10px; display: flex; justify-content: center; gap: 18px; flex-wrap: wrap;">
        <a href="https://instagram.com/x_estal_nitish" target="_blank" style="color: #38BDF8; font-weight: 600; text-decoration: none;">
            📸 Instagram: @x_estal_nitish
        </a>
        <span style="color: #475569;">&bull;</span>
        <a href="https://www.linkedin.com/in/nitish-kumar-33714642b" target="_blank" style="color: #818CF8; font-weight: 600; text-decoration: none;">
            💼 LinkedIn: Nitish Kumar
        </a>
    </div>
    <div style="margin-top: 8px; font-size: 0.82rem; color: #64748B;">
        Autonomous LangGraph Agent Architecture &bull; Grounded Fact-Checking &bull; Multi-Tool Intelligence
    </div>
</div>
""", unsafe_allow_html=True)
