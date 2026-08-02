"""
finance_tools.py
-----------------
Deterministic financial calculation tools exposed to the LangChain
agent via @tool. Every tool wraps a pure Python function so that the
core math can be unit tested independently of the LLM (see
tests/test_finance_tools.py).

Design principles
-----------------
1. No unsafe eval(): the calculator tool parses expressions with the
   `ast` module and only permits a whitelisted set of nodes/operators.
2. Every function validates its inputs and raises ValueError with a
   clear, user-facing message on invalid input (negative amounts,
   zero/negative durations, absurd rates, etc.).
3. All monetary outputs are rounded to 2 decimal places.
4. Functions return plain dictionaries (JSON-serializable) so the
   Streamlit UI can render "Calculation Details" without re-deriving
   anything the tool already computed.
"""

from __future__ import annotations

import ast
import operator
from typing import Any, Dict

from langchain_core.tools import tool

from config import settings

# ---------------------------------------------------------------------------
# Shared validation helpers
# ---------------------------------------------------------------------------


def _validate_positive_amount(value: float, name: str) -> None:
    if value is None:
        raise ValueError(f"{name} is required.")
    if value <= 0:
        raise ValueError(f"{name} must be a positive number greater than zero.")
    if value > settings.max_amount:
        raise ValueError(f"{name} exceeds the maximum supported amount of {settings.max_amount:,.0f}.")


def _validate_years(years: float) -> None:
    if years is None:
        raise ValueError("Duration in years is required.")
    if years <= 0:
        raise ValueError("Duration in years must be greater than zero.")
    if years > settings.max_years:
        raise ValueError(f"Duration in years cannot exceed {settings.max_years} years.")


def _validate_rate(rate: float) -> None:
    if rate is None:
        raise ValueError("Annual interest/return rate is required.")
    if rate < 0:
        raise ValueError("Annual interest/return rate cannot be negative.")
    if rate > settings.max_rate_percent:
        raise ValueError(f"Annual rate looks unrealistic (> {settings.max_rate_percent}%). Please double-check it.")


def _round2(value: float) -> float:
    return round(float(value) + 0.0, 2)


# ---------------------------------------------------------------------------
# 1. Calculator Tool — safe AST-based arithmetic evaluator (no eval())
# ---------------------------------------------------------------------------

_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate a whitelisted AST node. Raises ValueError
    on any construct outside +, -, *, /, %, **, parentheses, and unary
    +/- on numeric literals."""

    if isinstance(node, ast.Expression):
        return _safe_eval_node(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            return node.value
        raise ValueError("Only numeric constants are allowed in the expression.")

    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise ValueError(f"Operator '{op_type.__name__}' is not permitted.")
        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)
        if op_type is ast.Div and right == 0:
            raise ValueError("Division by zero is not allowed.")
        if op_type is ast.Mod and right == 0:
            raise ValueError("Modulo by zero is not allowed.")
        if op_type is ast.Pow and (abs(right) > 1000):
            raise ValueError("Exponent is too large to evaluate safely.")
        return _ALLOWED_BINOPS[op_type](left, right)

    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise ValueError(f"Unary operator '{op_type.__name__}' is not permitted.")
        return _ALLOWED_UNARYOPS[op_type](_safe_eval_node(node.operand))

    raise ValueError(f"Expression contains a disallowed construct: {type(node).__name__}")


def safe_calculate(expression: str) -> Dict[str, Any]:
    """Evaluate a basic arithmetic expression safely.

    Supports +, -, *, /, %, ** (power) and parentheses. Does not use
    eval()/exec(); parses the expression into an AST and walks only a
    whitelisted set of nodes.
    """
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty.")

    # Reject anything that looks like it is trying to reach outside
    # plain arithmetic before even parsing (defense in depth).
    forbidden_tokens = ["__", "import", "lambda", "eval", "exec", "open", "os.", "sys."]
    lowered = expression.lower()
    if any(tok in lowered for tok in forbidden_tokens):
        raise ValueError("Expression contains disallowed keywords.")

    try:
        parsed = ast.parse(expression, mode="eval")
        result = _safe_eval_node(parsed)
    except (SyntaxError, ZeroDivisionError, TypeError) as exc:
        raise ValueError(f"Could not evaluate expression: {exc}") from exc

    return {
        "expression": expression,
        "result": _round2(result),
    }


@tool
def calculator_tool(expression: str) -> Dict[str, Any]:
    """Evaluate a general arithmetic expression safely.

    Use this for plain math such as percentages, sums, products, or
    quick numeric checks that are NOT an EMI, SIP, savings-goal, or
    budget question. Supports +, -, *, /, %, ** and parentheses.
    Example expressions: "20/100*85000", "(15000-5000)*0.08", "2**10".
    """
    return safe_calculate(expression)


# ---------------------------------------------------------------------------
# 2. EMI Calculator Tool
# ---------------------------------------------------------------------------


def calculate_emi(principal: float, annual_rate_percent: float, years: float) -> Dict[str, Any]:
    """Calculate loan EMI (Equated Monthly Installment).

    EMI = P * r * (1+r)^n / ((1+r)^n - 1)
    where r = monthly interest rate (annual_rate_percent / 12 / 100)
          n = number of monthly installments (years * 12)
    """
    _validate_positive_amount(principal, "Loan principal")
    _validate_rate(annual_rate_percent)
    _validate_years(years)

    n = int(round(years * 12))
    r = (annual_rate_percent / 12) / 100

    if r == 0:
        # Zero-interest edge case: EMI is simply principal spread evenly.
        emi = principal / n
    else:
        factor = (1 + r) ** n
        emi = principal * r * factor / (factor - 1)

    total_payment = emi * n
    total_interest = total_payment - principal

    return {
        "principal": _round2(principal),
        "annual_rate_percent": annual_rate_percent,
        "years": years,
        "months": n,
        "monthly_emi": _round2(emi),
        "total_payment": _round2(total_payment),
        "total_interest": _round2(total_interest),
    }


@tool
def emi_calculator_tool(principal: float, annual_rate_percent: float, years: float) -> Dict[str, Any]:
    """Calculate the monthly EMI (Equated Monthly Installment) for a loan.

    Use this whenever the user asks about a loan EMI, home loan,
    car loan, or "monthly installment" calculation.

    Args:
        principal: Loan amount in rupees (e.g. 1500000 for ₹15 lakh).
        annual_rate_percent: Annual interest rate as a percentage (e.g. 8.5).
        years: Loan tenure in years (e.g. 20).

    Returns monthly EMI, total interest payable, and total payment.
    """
    return calculate_emi(principal, annual_rate_percent, years)


# ---------------------------------------------------------------------------
# 3. SIP Calculator Tool
# ---------------------------------------------------------------------------


def calculate_sip(monthly_investment: float, annual_return_percent: float, years: float) -> Dict[str, Any]:
    """Calculate the future value of a monthly SIP (Systematic Investment Plan).

    FV = P * ((1+r)^n - 1) / r
    where r = monthly return rate (annual_return_percent / 12 / 100)
          n = number of monthly contributions (years * 12)
    """
    _validate_positive_amount(monthly_investment, "Monthly investment")
    _validate_rate(annual_return_percent)
    _validate_years(years)

    n = int(round(years * 12))
    r = (annual_return_percent / 12) / 100

    total_invested = monthly_investment * n

    if r == 0:
        future_value = total_invested
    else:
        future_value = monthly_investment * (((1 + r) ** n - 1) / r)

    profit = future_value - total_invested

    return {
        "monthly_investment": _round2(monthly_investment),
        "annual_return_percent": annual_return_percent,
        "years": years,
        "months": n,
        "future_value": _round2(future_value),
        "total_invested": _round2(total_invested),
        "estimated_profit": _round2(profit),
    }


@tool
def sip_calculator_tool(monthly_investment: float, annual_return_percent: float, years: float) -> Dict[str, Any]:
    """Calculate the future value of a monthly SIP / recurring investment.

    Use this when the user asks how much a monthly investment (SIP,
    mutual fund, recurring deposit) will grow to over time.

    Args:
        monthly_investment: Amount invested every month in rupees.
        annual_return_percent: Expected annual rate of return as a percentage (e.g. 12).
        years: Investment duration in years.

    Returns future value, total amount invested, and estimated profit.
    """
    return calculate_sip(monthly_investment, annual_return_percent, years)


# ---------------------------------------------------------------------------
# 4. Savings Goal Calculator Tool
# ---------------------------------------------------------------------------


def calculate_savings_goal(target_amount: float, annual_return_percent: float, years: float) -> Dict[str, Any]:
    """Calculate the required monthly investment to reach a target amount.

    Rearranges the SIP future-value formula for the monthly contribution:
    required_monthly = target * r / ((1+r)^n - 1)
    """
    _validate_positive_amount(target_amount, "Target amount")
    _validate_rate(annual_return_percent)
    _validate_years(years)

    n = int(round(years * 12))
    r = (annual_return_percent / 12) / 100

    if r == 0:
        required_monthly = target_amount / n
    else:
        required_monthly = target_amount * r / ((1 + r) ** n - 1)

    total_contribution = required_monthly * n
    projected_growth = target_amount - total_contribution

    return {
        "target_amount": _round2(target_amount),
        "annual_return_percent": annual_return_percent,
        "years": years,
        "months": n,
        "required_monthly_investment": _round2(required_monthly),
        "total_contribution": _round2(total_contribution),
        "projected_growth_from_returns": _round2(projected_growth),
    }


@tool
def savings_goal_tool(target_amount: float, annual_return_percent: float, years: float) -> Dict[str, Any]:
    """Calculate the monthly investment required to reach a savings goal.

    Use this when the user asks "how much should I save/invest monthly
    to reach ₹X" over a given number of years.

    Args:
        target_amount: The savings target in rupees (e.g. 2000000 for ₹20 lakh).
        annual_return_percent: Expected annual rate of return as a percentage.
        years: Time horizon in years.

    Returns the required monthly investment plus a breakdown of
    contributions versus projected growth from returns.
    """
    return calculate_savings_goal(target_amount, annual_return_percent, years)


# ---------------------------------------------------------------------------
# 5. Budget Planner Tool (50-30-20 rule)
# ---------------------------------------------------------------------------


def calculate_budget(monthly_income: float) -> Dict[str, Any]:
    """Suggest a budget allocation using the 50-30-20 rule.

    50% Needs, 30% Wants, 20% Savings/Investments.
    """
    _validate_positive_amount(monthly_income, "Monthly income")

    needs = monthly_income * 0.50
    wants = monthly_income * 0.30
    savings = monthly_income * 0.20

    return {
        "monthly_income": _round2(monthly_income),
        "needs_50_percent": _round2(needs),
        "wants_30_percent": _round2(wants),
        "savings_20_percent": _round2(savings),
    }


@tool
def budget_planner_tool(monthly_income: float) -> Dict[str, Any]:
    """Suggest a monthly budget allocation using the 50-30-20 rule.

    Use this when the user wants help planning or allocating a
    monthly income/salary into needs, wants, and savings.

    Args:
        monthly_income: Take-home monthly income in rupees.

    Returns the suggested rupee allocation for Needs (50%), Wants
    (30%), and Savings/Investments (20%).
    """
    return calculate_budget(monthly_income)


# ---------------------------------------------------------------------------
# Tool registry consumed by agent.py
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    calculator_tool,
    emi_calculator_tool,
    sip_calculator_tool,
    savings_goal_tool,
    budget_planner_tool,
]
