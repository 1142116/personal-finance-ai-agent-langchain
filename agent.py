"""
agent.py
---------
Builds the LangChain Tool Calling Agent that powers the Personal
Finance AI Agent. The LLM (Groq Llama 3.3 70B Versatile, temperature=0)
automatically decides which of the five finance_tools to invoke based
on the user's natural-language question — there is no manual/rule
based routing in this file.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq

from config import settings
from finance_tools import ALL_TOOLS
from prompts import SYSTEM_PROMPT


class AgentInitializationError(RuntimeError):
    """Raised when the agent cannot be constructed (e.g. missing API key)."""


def _normalize_indian_number_words(text: str) -> str:
    """Best-effort normalization of 'lakh/lac' and 'crore' phrases so the
    LLM receives an unambiguous numeric hint alongside the original text.

    This does not replace the user's text — it appends a normalized
    hint the model can use, since rewriting user text in place is
    error-prone with mixed currency phrasing.
    """
    pattern = re.compile(
        r"(\d+(?:\.\d+)?)\s*(lakh|lac|crore|cr)\b", flags=re.IGNORECASE
    )

    hints: List[str] = []
    for match in pattern.finditer(text):
        number = float(match.group(1))
        unit = match.group(2).lower()
        multiplier = 100_000 if unit in ("lakh", "lac") else 10_000_000
        hints.append(f"{match.group(0)} = {number * multiplier:,.0f}")

    if not hints:
        return text

    hint_str = "; ".join(hints)
    return f"{text}\n\n[Normalization hint: {hint_str}]"


def build_agent_executor() -> AgentExecutor:
    """Construct and return a ready-to-invoke AgentExecutor."""
    if not settings.groq_api_key:
        raise AgentInitializationError(
            "GROQ_API_KEY is not set. Add it to your .env file or "
            "Streamlit secrets before starting the agent."
        )

    llm = ChatGroq(
        model=settings.model_name,
        temperature=settings.temperature,
        api_key=settings.groq_api_key,
        timeout=settings.request_timeout,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=ALL_TOOLS, prompt=prompt)

    executor = AgentExecutor(
        agent=agent,
        tools=ALL_TOOLS,
        verbose=False,
        handle_parsing_errors=True,
        max_iterations=6,
        return_intermediate_steps=True,
    )
    return executor


def run_agent(
    executor: AgentExecutor, user_input: str, chat_history: List[Tuple[str, str]]
) -> Dict[str, Any]:
    """Invoke the agent for a single turn.

    Args:
        executor: An AgentExecutor built by build_agent_executor().
        user_input: The user's latest natural-language question.
        chat_history: List of (role, content) tuples for prior turns,
            where role is "human" or "ai".

    Returns a dict with keys: "output", "tool_used", "tool_input",
    "tool_output".
    """
    normalized_input = _normalize_indian_number_words(user_input)

    # Convert simple (role, content) tuples into LangChain message tuples
    # understood by ChatPromptTemplate's MessagesPlaceholder.
    formatted_history = [(role, content) for role, content in chat_history]

    result = executor.invoke(
        {"input": normalized_input, "chat_history": formatted_history}
    )

    tool_used = None
    tool_input = None
    tool_output = None

    intermediate_steps = result.get("intermediate_steps", [])
    if intermediate_steps:
        # Use the last tool call as the "primary" one shown in the UI.
        last_action, last_observation = intermediate_steps[-1]
        tool_used = getattr(last_action, "tool", None)
        tool_input = getattr(last_action, "tool_input", None)
        tool_output = last_observation

    return {
        "output": result.get("output", ""),
        "tool_used": tool_used,
        "tool_input": tool_input,
        "tool_output": tool_output,
        "all_steps": intermediate_steps,
    }
