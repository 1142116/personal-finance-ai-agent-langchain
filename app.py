"""
app.py
-------
Streamlit front-end for the Personal Finance AI Agent.

Responsibilities:
- Render a dark, professional chat UI.
- Maintain full conversation history in st.session_state.
- Provide sidebar example questions, a clear-chat button, and
  conversation history.
- Invoke the LangChain agent (agent.py) and display the response,
  the tool used, calculation details, and any warnings/errors.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import streamlit as st

from agent import AgentInitializationError, build_agent_executor, run_agent
from config import has_valid_api_key, settings

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Personal Finance AI Agent",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark professional theme (custom CSS)
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
<style>
:root {
    --bg-primary: #0e1117;
    --bg-secondary: #161b22;
    --bg-card: #1c2129;
    --accent: #22c55e;
    --accent-soft: #16a34a33;
    --text-primary: #e6edf3;
    --text-muted: #9aa5b1;
    --border: #2a303b;
}

.stApp {
    background-color: var(--bg-primary);
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary);
    border-right: 1px solid var(--border);
}

h1, h2, h3 {
    color: var(--text-primary) !important;
}

.app-subtitle {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin-top: -0.6rem;
    margin-bottom: 1.2rem;
}

.finance-card {
    background-color: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: 0.6rem;
    margin-bottom: 0.6rem;
}

.finance-card h4 {
    margin-top: 0;
    color: var(--accent);
}

.tool-badge {
    display: inline-block;
    background-color: var(--accent-soft);
    color: var(--accent);
    border: 1px solid var(--accent);
    border-radius: 999px;
    padding: 2px 12px;
    font-size: 0.8rem;
    font-weight: 600;
    margin-bottom: 0.5rem;
}

.warning-box {
    background-color: #3b1d1d;
    border: 1px solid #7f2d2d;
    color: #ffb4b4;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-top: 0.5rem;
}

.disclaimer {
    color: var(--text-muted);
    font-size: 0.82rem;
    margin-top: 1.5rem;
    border-top: 1px solid var(--border);
    padding-top: 0.8rem;
}

button[kind="secondary"] {
    border-color: var(--border) !important;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    # Each message: {"role": "human"|"ai", "content": str, "meta": dict|None}
    st.session_state.messages: List[dict] = []

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None

if "agent_error" not in st.session_state:
    st.session_state.agent_error = None

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


EXAMPLE_QUESTIONS = [
    ("🏠 Calculate EMI", "Calculate EMI for ₹15 lakh at 8.5% interest for 20 years."),
    ("📈 Calculate SIP Returns", "If I invest ₹7000 monthly for 15 years at 10% return, how much will I have?"),
    ("🎯 Savings Goal", "How much should I save monthly to reach ₹20 lakh in 10 years at 12% return?"),
    ("📊 Budget Planner", "Create a monthly budget plan for an income of ₹90,000."),
    ("🧮 Calculator", "What is 20% of ₹85,000?"),
]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 💰 Finance AI Agent")
    st.caption("LangChain Tool-Calling · Groq Llama 3.3 70B")

    st.markdown("---")
    st.markdown("### 💡 Example Questions")
    for label, question in EXAMPLE_QUESTIONS:
        if st.button(label, use_container_width=True, key=f"example_{label}"):
            st.session_state.pending_question = question

    st.markdown("---")
    st.markdown("### 🗂️ Conversation History")
    if st.session_state.messages:
        human_turns = [m for m in st.session_state.messages if m["role"] == "human"]
        if human_turns:
            for i, msg in enumerate(human_turns[-10:], start=1):
                preview = msg["content"][:45] + ("…" if len(msg["content"]) > 45 else "")
                st.caption(f"{i}. {preview}")
        else:
            st.caption("No questions yet.")
    else:
        st.caption("No conversation yet. Ask a question to get started!")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.session_state.pending_question = None
        st.rerun()

    st.markdown("---")
    api_status = "🟢 Connected" if has_valid_api_key() else "🔴 Missing API Key"
    st.caption(f"**Model:** {settings.model_name}")
    st.caption(f"**Status:** {api_status}")

    st.markdown(
        '<div class="disclaimer">⚠️ This assistant provides estimates only '
        "and does not constitute financial advice. Consult a licensed "
        "financial advisor before making decisions.</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

st.markdown(f"# {settings.app_title}")
st.markdown(f'<div class="app-subtitle">{settings.app_subtitle}</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# API key / agent initialization guard
# ---------------------------------------------------------------------------

if not has_valid_api_key():
    st.markdown(
        '<div class="warning-box">🔑 <b>Groq API key not found.</b> '
        "Add <code>GROQ_API_KEY</code> to a local <code>.env</code> file "
        "(see <code>.env.example</code>) or to your Streamlit Cloud secrets, "
        "then restart the app. Get a free key at "
        '<a href="https://console.groq.com/keys" style="color:#22c55e;">console.groq.com/keys</a>.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

if st.session_state.agent_executor is None and st.session_state.agent_error is None:
    try:
        st.session_state.agent_executor = build_agent_executor()
    except AgentInitializationError as exc:
        st.session_state.agent_error = str(exc)

if st.session_state.agent_error:
    st.markdown(
        f'<div class="warning-box">⚠️ <b>Agent initialization failed:</b> '
        f"{st.session_state.agent_error}</div>",
        unsafe_allow_html=True,
    )
    st.stop()

executor = st.session_state.agent_executor

# ---------------------------------------------------------------------------
# Render chat history
# ---------------------------------------------------------------------------


def render_tool_details(meta: dict) -> None:
    """Render Tool Used / Calculation Details / Warnings for an AI turn."""
    if not meta:
        return

    tool_used = meta.get("tool_used")
    tool_output = meta.get("tool_output")
    warning = meta.get("warning")

    if tool_used:
        st.markdown(f'<span class="tool-badge">🔧 Tool used: {tool_used}</span>', unsafe_allow_html=True)

    if isinstance(tool_output, dict) and tool_output:
        with st.expander("📋 Calculation Details", expanded=False):
            for key, value in tool_output.items():
                display_key = key.replace("_", " ").title()
                if isinstance(value, float):
                    st.write(f"**{display_key}:** {settings.currency_symbol}{value:,.2f}" if "percent" not in key else f"**{display_key}:** {value}%")
                else:
                    st.write(f"**{display_key}:** {value}")

    if warning:
        st.markdown(f'<div class="warning-box">⚠️ {warning}</div>', unsafe_allow_html=True)


for message in st.session_state.messages:
    role = "user" if message["role"] == "human" else "assistant"
    with st.chat_message(role):
        st.markdown(message["content"])
        if role == "assistant":
            render_tool_details(message.get("meta", {}))

# ---------------------------------------------------------------------------
# Chat input handling
# ---------------------------------------------------------------------------


def get_langchain_history() -> List[Tuple[str, str]]:
    """Build (role, content) history for the agent from session state,
    excluding the message currently being answered."""
    history = []
    for msg in st.session_state.messages:
        role = "human" if msg["role"] == "human" else "ai"
        history.append((role, msg["content"]))
    return history


def handle_user_message(user_text: str) -> None:
    user_text = user_text.strip()
    if not user_text:
        return

    history = get_langchain_history()
    st.session_state.messages.append({"role": "human", "content": user_text, "meta": None})

    with st.chat_message("user"):
        st.markdown(user_text)

    with st.chat_message("assistant"):
        with st.spinner("Thinking through the calculation…"):
            meta = {}
            try:
                start = time.time()
                result = run_agent(executor, user_text, history)
                elapsed = time.time() - start

                answer = result["output"]
                meta = {
                    "tool_used": result.get("tool_used"),
                    "tool_output": result.get("tool_output"),
                }
                if elapsed > settings.request_timeout:
                    meta["warning"] = "The response took longer than expected."

            except ValueError as exc:
                # Tool-level validation errors (invalid numbers, negative
                # values, division by zero, etc.)
                answer = f"I couldn't complete that calculation: {exc}"
                meta = {"warning": "Invalid input detected. Please check the numbers you provided."}

            except TimeoutError:
                answer = "The request timed out while waiting for a response."
                meta = {"warning": "LLM timeout — please try again in a moment."}

            except Exception as exc:  # noqa: BLE001 - surfaced safely to the user
                error_text = str(exc)
                if "api_key" in error_text.lower() or "authentication" in error_text.lower():
                    answer = "Authentication with Groq failed. Please check your GROQ_API_KEY."
                    meta = {"warning": "Missing or invalid Groq API key."}
                elif "rate limit" in error_text.lower():
                    answer = "Groq rate limit reached. Please wait a moment and try again."
                    meta = {"warning": "Rate limit exceeded."}
                else:
                    answer = "Something went wrong while processing your request."
                    meta = {"warning": f"Unexpected error: {error_text}"}

            st.markdown(answer)
            render_tool_details(meta)

    st.session_state.messages.append({"role": "ai", "content": answer, "meta": meta})


# Handle a sidebar example click first (takes priority for this run).
if st.session_state.pending_question:
    question = st.session_state.pending_question
    st.session_state.pending_question = None
    handle_user_message(question)

# Standard chat input box.
user_prompt = st.chat_input("Ask about EMI, SIP returns, savings goals, budgeting, or any calculation…")
if user_prompt:
    handle_user_message(user_prompt)
