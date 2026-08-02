"""
tests/test_finance_tools.py
----------------------------
Pytest unit tests for the deterministic calculation functions in
finance_tools.py. These tests call the underlying pure Python
functions directly (safe_calculate, calculate_emi, calculate_sip,
calculate_savings_goal, calculate_budget) so they run without any
Groq API key or network access.
"""

import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finance_tools import (  # noqa: E402
    calculate_budget,
    calculate_emi,
    calculate_savings_goal,
    calculate_sip,
    safe_calculate,
)


# ---------------------------------------------------------------------------
# Calculator Tool
# ---------------------------------------------------------------------------

class TestCalculator:
    def test_basic_addition(self):
        assert safe_calculate("2 + 3")["result"] == 5

    def test_percentage_of_amount(self):
        result = safe_calculate("20/100*85000")
        assert result["result"] == 17000.0

    def test_operator_precedence_and_parentheses(self):
        result = safe_calculate("(15000-5000)*0.08")
        assert result["result"] == 800.0

    def test_power_operator(self):
        assert safe_calculate("2**10")["result"] == 1024

    def test_modulo_operator(self):
        assert safe_calculate("10 % 3")["result"] == 1

    def test_division_by_zero_raises(self):
        with pytest.raises(ValueError):
            safe_calculate("5/0")

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError):
            safe_calculate("")

    def test_disallowed_keyword_raises(self):
        with pytest.raises(ValueError):
            safe_calculate("__import__('os').system('ls')")

    def test_disallowed_function_call_raises(self):
        with pytest.raises(ValueError):
            safe_calculate("abs(-5)")


# ---------------------------------------------------------------------------
# EMI Calculator Tool
# ---------------------------------------------------------------------------

class TestEMICalculator:
    def test_known_emi_value(self):
        # ₹10,00,000 at 8% for 20 years ≈ ₹8,364.40 monthly EMI
        result = calculate_emi(1_000_000, 8.0, 20)
        assert math.isclose(result["monthly_emi"], 8364.40, abs_tol=1.0)

    def test_total_payment_exceeds_principal(self):
        result = calculate_emi(1_500_000, 8.5, 20)
        assert result["total_payment"] > result["principal"]
        assert result["total_interest"] == pytest.approx(
            result["total_payment"] - result["principal"], abs=0.01
        )

    def test_zero_interest_rate(self):
        result = calculate_emi(120_000, 0, 10)
        # 120 months, zero interest -> EMI = principal / months
        assert result["monthly_emi"] == pytest.approx(1000.0, abs=0.01)
        assert result["total_interest"] == pytest.approx(0.0, abs=0.01)

    def test_negative_principal_raises(self):
        with pytest.raises(ValueError):
            calculate_emi(-500000, 8.0, 20)

    def test_zero_years_raises(self):
        with pytest.raises(ValueError):
            calculate_emi(500000, 8.0, 0)

    def test_negative_rate_raises(self):
        with pytest.raises(ValueError):
            calculate_emi(500000, -5.0, 10)


# ---------------------------------------------------------------------------
# SIP Calculator Tool
# ---------------------------------------------------------------------------

class TestSIPCalculator:
    def test_future_value_greater_than_invested(self):
        result = calculate_sip(3000, 8.0, 5)
        assert result["future_value"] > result["total_invested"]
        assert result["total_invested"] == pytest.approx(3000 * 60, abs=0.01)

    def test_profit_matches_difference(self):
        result = calculate_sip(7000, 10.0, 15)
        assert result["estimated_profit"] == pytest.approx(
            result["future_value"] - result["total_invested"], abs=0.01
        )

    def test_zero_return_rate(self):
        result = calculate_sip(5000, 0, 10)
        assert result["future_value"] == pytest.approx(5000 * 120, abs=0.01)
        assert result["estimated_profit"] == pytest.approx(0.0, abs=0.01)

    def test_negative_investment_raises(self):
        with pytest.raises(ValueError):
            calculate_sip(-1000, 10.0, 5)


# ---------------------------------------------------------------------------
# Savings Goal Calculator Tool
# ---------------------------------------------------------------------------

class TestSavingsGoalCalculator:
    def test_required_monthly_investment_reaches_target(self):
        result = calculate_savings_goal(2_000_000, 12.0, 10)
        assert result["required_monthly_investment"] > 0
        # Reinvest the computed monthly amount through the SIP formula and
        # confirm it approximately reaches the original target.
        sip_check = calculate_sip(
            result["required_monthly_investment"], 12.0, 10
        )
        assert sip_check["future_value"] == pytest.approx(2_000_000, rel=0.01)

    def test_zero_return_rate_divides_evenly(self):
        result = calculate_savings_goal(120_000, 0, 10)
        assert result["required_monthly_investment"] == pytest.approx(1000.0, abs=0.01)

    def test_negative_target_raises(self):
        with pytest.raises(ValueError):
            calculate_savings_goal(-100000, 10.0, 5)


# ---------------------------------------------------------------------------
# Budget Planner Tool
# ---------------------------------------------------------------------------

class TestBudgetPlanner:
    def test_50_30_20_split(self):
        result = calculate_budget(90_000)
        assert result["needs_50_percent"] == pytest.approx(45_000, abs=0.01)
        assert result["wants_30_percent"] == pytest.approx(27_000, abs=0.01)
        assert result["savings_20_percent"] == pytest.approx(18_000, abs=0.01)

    def test_allocations_sum_to_income(self):
        result = calculate_budget(80_000)
        total = (
            result["needs_50_percent"]
            + result["wants_30_percent"]
            + result["savings_20_percent"]
        )
        assert total == pytest.approx(80_000, abs=0.01)

    def test_negative_income_raises(self):
        with pytest.raises(ValueError):
            calculate_budget(-5000)

    def test_zero_income_raises(self):
        with pytest.raises(ValueError):
            calculate_budget(0)
