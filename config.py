"""
config.py
----------
Central configuration for the Personal Finance AI Agent.

Loads environment variables (via python-dotenv) and exposes typed
constants used across the application: model name, temperature,
app metadata and validation limits.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file (ignored by git) into the
# process environment. Safe to call even if .env does not exist.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Immutable application settings, resolved once at import time."""

    # --- LLM configuration -------------------------------------------------
    groq_api_key: str = os.getenv("GROQ_API_KEY", "")
    model_name: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    temperature: float = 0.0
    request_timeout: int = 30  # seconds, used to detect LLM timeouts

    # --- App metadata --------------------------------------------------
    app_title: str = "💰 Personal Finance AI Agent"
    app_subtitle: str = "AI-powered financial assistant using LangChain Tool Calling"
    currency_symbol: str = "₹"

    # --- Validation limits (used by finance_tools.py) -------------------
    max_amount: float = 1_000_000_000  # 100 crore ceiling, sanity guard
    max_years: int = 50
    min_years: int = 1
    max_rate_percent: float = 50.0


settings = Settings()


def has_valid_api_key() -> bool:
    """Return True if an Groq API key appears to be configured."""
    key = settings.groq_api_key
    return bool(key) and key.strip() != "" and key.strip().lower() != "your_groq_api_key_here"
