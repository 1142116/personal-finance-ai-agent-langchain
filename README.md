# 💰 Personal Finance AI Agent (LangChain)

An AI-powered personal finance assistant that understands natural-language
questions and automatically selects the correct financial calculation tool
using **LangChain Tool Calling Agents** and **Groq Llama 3.3 70B Versatile**.

Built as a GenAI / Agentic AI capstone project.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Folder Structure](#folder-structure)
4. [Tools Implemented](#tools-implemented)
5. [Installation](#installation)
6. [Groq API Key Setup](#groq-api-key-setup)
7. [Running Locally](#running-locally)
8. [Running Tests](#running-tests)
9. [GitHub Deployment](#github-deployment)
10. [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
11. [Features](#features)
12. [Screenshots](#screenshots)
13. [Future Improvements](#future-improvements)
14. [Disclaimer](#disclaimer)
15. [License](#license)

---

## Project Overview

Users ask plain-English financial questions such as:

> "Calculate EMI for ₹15 lakh at 8.5% interest for 20 years."
> "If I invest ₹7000 monthly for 15 years at 10% return, how much will I have?"
> "How much should I save monthly to reach ₹20 lakh in 10 years?"
> "Create a monthly budget for someone earning ₹80,000."

Instead of relying on the LLM to guess at arithmetic, the agent **always
delegates the actual math to a deterministic Python tool**. The LLM's job
is limited to understanding intent, picking the right tool, filling in
its arguments, and explaining the result in plain language.

## Architecture

```
                     ┌─────────────────────────┐
                     │   Streamlit UI (app.py) │
                     │  chat + sidebar + state │
                     └───────────┬─────────────┘
                                 │ user question + chat history
                                 ▼
                     ┌─────────────────────────┐
                     │   agent.py (LangChain)  │
                     │  create_tool_calling_   │
                     │  agent + AgentExecutor  │
                     └───────────┬─────────────┘
                                 │ tool-calling decision
                                 ▼
                     ┌─────────────────────────┐
                     │  Groq Llama 3.3 70B Versatile     │
                     │  temperature = 0        │
                     └───────────┬─────────────┘
                                 │ selects & calls one of:
                                 ▼
        ┌───────────┬────────────┬────────────┬───────────────┬────────────────┐
        │ Calculator │    EMI     │    SIP     │ Savings Goal  │ Budget Planner │
        │   Tool     │ Calculator │ Calculator │   Calculator  │      Tool      │
        └───────────┴────────────┴────────────┴───────────────┴────────────────┘
                       (finance_tools.py — pure, unit-tested Python)
```

Each tool is a pure Python function wrapped with `@tool`, so it can be
called by the LLM **and** unit tested independently, with no Groq key
required for the tests.

## Folder Structure

```
Personal_Finance_AI_Agent/
├── app.py                      # Streamlit UI, chat, sidebar, session state
├── agent.py                    # LangChain tool-calling agent construction
├── finance_tools.py            # 5 deterministic finance tools (AST-safe calculator, EMI, SIP, savings goal, budget)
├── prompts.py                  # System prompt / prompt engineering
├── config.py                   # Environment config, settings, validation limits
├── requirements.txt             # Runtime dependencies
├── requirements-notebook.txt    # Dev/test/notebook dependencies
├── .gitignore
├── .env.example
├── README.md
├── tests/
│   └── test_finance_tools.py   # pytest unit tests (no API key required)
├── docs/
│   └── Model_Development_Document.docx
├── screenshots/                # App screenshots (see screenshots/README.md)
└── assets/                     # Static assets (logos, icons)
```

## Tools Implemented

| # | Tool | Purpose |
|---|------|---------|
| 1 | **Calculator Tool** | Safe arithmetic (`+ − × ÷ % **` and parentheses) using an AST parser — no `eval()` |
| 2 | **EMI Calculator** | `EMI = P·r·(1+r)^n / ((1+r)^n − 1)` → monthly EMI, total interest, total payment |
| 3 | **SIP Calculator** | `FV = P·((1+r)^n − 1)/r` → future value, invested amount, profit |
| 4 | **Savings Goal Calculator** | Rearranges the SIP formula to find the required monthly investment for a target amount |
| 5 | **Budget Planner** | 50% Needs / 30% Wants / 20% Savings allocation from monthly income |

## Installation

```bash
git clone https://github.com/<your-username>/Personal_Finance_AI_Agent.git
cd Personal_Finance_AI_Agent

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## Groq API Key Setup

1. Create a free API key at [console.groq.com/keys](https://console.groq.com/keys).
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
3. Open `.env` and paste your key:
   ```
   GROQ_API_KEY=gsk_...
   ```

`.env` is listed in `.gitignore` and will never be committed.

## Running Locally

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`. Try a sidebar example question or
type your own in the chat box.

## Running Tests

```bash
pip install -r requirements-notebook.txt
pytest tests/ -v
```

Tests call the underlying calculation functions directly, so they run
without a Groq API key or network access.

## GitHub Deployment

```bash
git init
git add .
git commit -m "Initial commit: Personal Finance AI Agent"
git branch -M main
git remote add origin https://github.com/<your-username>/Personal_Finance_AI_Agent.git
git push -u origin main
```

Double-check that `.env` is **not** staged (`git status` should not list it —
`.gitignore` already excludes it).

## Streamlit Cloud Deployment

1. Push the repository to GitHub (above).
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
3. Click **New app**, select the repository, branch `main`, and set the
   main file path to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "sk-..."
   ```
5. Click **Deploy**. Streamlit Cloud installs `requirements.txt`
   automatically.

## Features

- 🧠 Fully automatic tool selection — no manual routing/if-else logic
- 🔒 Safe, `eval()`-free arithmetic via an AST-based calculator
- 💬 Full conversational memory across the session (follow-up questions work)
- 🎨 Dark, professional Streamlit UI with sidebar examples
- 📋 Transparent "Tool Used" badge and expandable "Calculation Details" per answer
- ⚠️ Graceful error handling for invalid numbers, negative values, division
  by zero, missing API keys, and LLM timeouts
- ✅ Pytest unit test suite covering all five tools

## Screenshots

See [`screenshots/`](screenshots/) — add your own captures after running the
app locally or on Streamlit Cloud:

- `screenshots/home_screen.png`
- `screenshots/emi_example.png`
- `screenshots/sip_example.png`
- `screenshots/budget_example.png`

## Future Improvements

- Add a REST API layer (FastAPI) alongside the Streamlit UI
- Support additional tools: tax estimators, retirement corpus planner,
  loan prepayment/refinance comparison
- Add multi-currency support beyond INR
- Persist conversation history to a database for multi-session continuity
- Add streaming token-by-token responses in the chat UI
- Add authentication for multi-user deployments

## Disclaimer

This application produces **illustrative estimates only** and does not
constitute financial, investment, tax, or lending advice. Always consult a
qualified, licensed financial professional before making financial
decisions.

## License

This project is released under the MIT License. See `LICENSE` for details
(add a `LICENSE` file with the standard MIT text before publishing, or
choose a different license as required by your institution).
