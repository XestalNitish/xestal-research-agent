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
    page_title="DeepResearch Agent | LangGraph Investigation System",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        border-radius: 10px;
        padding: 12px 16px;
        border: 1px solid #334155;
        margin-bottom: 10px;
    }
    .node-card {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-pass {
        background-color: #065F46;
        color: #34D399;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .badge-retry {
        background-color: #7C2D12;
        color: #FB923C;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .tool-tag {
        background-color: #1E293B;
        border: 1px solid #475569;
        color: #E2E8F0;
        padding: 2px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        margin-right: 6px;
        margin-bottom: 6px;
        display: inline-block;
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
# Sidebar Configuration
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
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
        st.caption("🟢 API Key configured")
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
# Main UI Layout
# ---------------------------------------------------------
st.markdown('<div class="main-title">🧭 DeepResearch Agent</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Autonomous Multi-Step Investigation, Local Document Reading, Grounding Audit & Automated Deliverable Generation</div>', unsafe_allow_html=True)

tabs = st.tabs(["🔬 New Research Mission", "📚 Research Vault", "🛠️ Tool Catalog", "🚀 Deployment Guide"])

# ---------------------------------------------------------
# Tab 1: New Research Mission
# ---------------------------------------------------------
with tabs[0]:
    col1, col2 = st.columns([3, 1])

    with col1:
        # Preset queries
        st.markdown("**Quick Presets:**")
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
    git commit -m "Initial commit: DeepResearch Agent Streamlit UI"
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
