import importlib.util
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import pytest

_STREAMLIT_APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
_spec = importlib.util.spec_from_file_location("streamlit_app", _STREAMLIT_APP_PATH)
assert _spec and _spec.loader  # type: ignore[truthy-bool]
streamlit_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(streamlit_app)

from goat_financial_model import GoatModel, InputSchedule


def _build_sample_schedule() -> pd.DataFrame:
    periods = pd.date_range("2024-01-31", periods=12, freq="M")
    revenue = pd.Series(100000 + (periods.month - 1) * 5000, index=periods)
    cogs = revenue * 0.45
    gross_profit = revenue - cogs
    variable_expenses = revenue * 0.12
    direct_wages = revenue * 0.08
    fixed_expenses = pd.Series(10000.0, index=periods)
    admin_wages = pd.Series(3000.0, index=periods)

    ebitda = gross_profit - variable_expenses - direct_wages - fixed_expenses - admin_wages
    depreciation = pd.Series(2000.0, index=periods)
    ebit = ebitda - depreciation
    interest = pd.Series(500.0, index=periods)
    npbt = ebit - interest
    tax = npbt * 0.25
    npat = npbt - tax

    cfo = ebitda - 1000
    capex = pd.Series(5000.0, index=periods)
    cfi = -capex
    cff = pd.Series(2000.0, index=periods)
    net_cash = cfo + cfi + cff

    opening_cash = pd.Series(50000.0, index=periods)
    opening_cash = opening_cash.cumsum().shift(1).fillna(50000.0)
    closing_cash = opening_cash + net_cash

    current_assets = closing_cash + 20000.0
    non_current_assets = pd.Series(100000.0, index=periods)
    current_liabilities = pd.Series(15000.0, index=periods)
    non_current_liabilities = pd.Series(50000.0, index=periods)
    equity = current_assets + non_current_assets - current_liabilities - non_current_liabilities

    data = pd.DataFrame(
        {
            "Revenue": revenue,
            "COGS": cogs,
            "Gross Margin": gross_profit,
            "Variable Expenses": variable_expenses,
            "Direct Wages": direct_wages,
            "Fixed Expenses": fixed_expenses,
            "Admin Wages": admin_wages,
            "EBITDA": ebitda,
            "Depreciation & Amortization": depreciation,
            "EBIT": ebit,
            "Interest Expense": interest,
            "NPBT": npbt,
            "Tax Expense": tax,
            "NPAT": npat,
            "CFO": cfo,
            "CFI": cfi,
            "CFF": cff,
            "Net Cash Flow": net_cash,
            "Capex": capex,
            "Opening Cash Balance": opening_cash,
            "Closing Cash Balance": closing_cash,
            "Cash and Cash Equivalents": closing_cash,
            "Current Assets": current_assets,
            "Non-current Assets": non_current_assets,
            "Current Liabilities": current_liabilities,
            "Non-current Liabilities": non_current_liabilities,
            "Equity": equity,
        }
    )

    return data


def _build_model() -> GoatModel:
    schedule = _build_sample_schedule()
    input_schedule = InputSchedule(data=schedule)
    return input_schedule.to_model()


def _build_model_with_loan_facilities(
    *, start_period: Optional[pd.Timestamp] = None
) -> GoatModel:
    schedule = _build_sample_schedule()
    first_period = start_period if start_period is not None else schedule.index[0]
    facilities = pd.DataFrame(
        {
            "Loan Name": ["Expansion Loan"],
            "Lender": ["Farm Bank"],
            "Start Period": [first_period],
            "Drawdown Amount": [1200.0],
            "Interest Rate %": [12.0],
            "Term (years)": [1.0],
            "Repayment Type": ["straight_line"],
            "Grace Periods": [0],
            "Balloon Amount": [0.0],
            "Fees": [0.0],
            "Active": [True],
        }
    )
    return InputSchedule(
        data=schedule,
        supplementary_tables={"Loan Facilities": facilities},
    ).to_model()


def _build_model_with_equity_facilities(
    *, start_period: Optional[pd.Timestamp] = None
) -> GoatModel:
    schedule = _build_sample_schedule()
    first_period = start_period if start_period is not None else schedule.index[0]
    facilities = pd.DataFrame(
        {
            "Investor Name": ["Growth Investor"],
            "Start Period": [first_period],
            "Contribution Amount": [1500.0],
            "Ownership %": [30.0],
            "Share Class": ["Ordinary"],
            "Issue Costs": [50.0],
            "Active": [True],
        }
    )
    return InputSchedule(
        data=schedule,
        supplementary_tables={"Equity Facilities": facilities},
    ).to_model()


def _build_valuation_model(
    ebit_values: list[float],
    *,
    depreciation_values: Optional[list[float]] = None,
    capex_values: Optional[list[float]] = None,
    valuation_inputs: Optional[dict[str, float]] = None,
    supplementary_tables: Optional[dict[str, pd.DataFrame]] = None,
) -> GoatModel:
    periods = pd.date_range("2024-12-31", periods=len(ebit_values), freq="Y")
    ebit = pd.Series(ebit_values, index=periods, dtype=float)
    depreciation = pd.Series(
        depreciation_values or [0.0] * len(ebit_values), index=periods, dtype=float
    )
    capex = pd.Series(capex_values or [0.0] * len(ebit_values), index=periods, dtype=float)
    zero = pd.Series(0.0, index=periods, dtype=float)

    ebitda = ebit + depreciation
    npbt = ebit.copy()
    tax = zero.copy()
    npat = npbt - tax
    cfo = ebitda
    cfi = -capex
    cff = zero.copy()
    net_cash = cfo + cfi + cff
    closing_cash = net_cash.cumsum()
    opening_cash = closing_cash.shift(1).fillna(0.0)
    current_assets = closing_cash.copy()
    non_current_assets = zero.copy()
    current_liabilities = zero.copy()
    non_current_liabilities = zero.copy()
    equity = current_assets + non_current_assets - current_liabilities - non_current_liabilities

    schedule = pd.DataFrame(
        {
            "Revenue": zero,
            "COGS": zero,
            "EBITDA": ebitda,
            "Depreciation & Amortization": depreciation,
            "EBIT": ebit,
            "Interest Expense": zero,
            "NPBT": npbt,
            "Tax Expense": tax,
            "NPAT": npat,
            "CFO": cfo,
            "CFI": cfi,
            "CFF": cff,
            "Net Cash Flow": net_cash,
            "Capex": capex,
            "Opening Cash Balance": opening_cash,
            "Closing Cash Balance": closing_cash,
            "Cash and Cash Equivalents": closing_cash,
            "Current Assets": current_assets,
            "Non-current Assets": non_current_assets,
            "Current Liabilities": current_liabilities,
            "Non-current Liabilities": non_current_liabilities,
            "Equity": equity,
        }
    )

    return InputSchedule(
        data=schedule,
        valuation_inputs=valuation_inputs or {},
        supplementary_tables=supplementary_tables or {},
    ).to_model()


def test_scenario_and_statements_pipeline():
    model = _build_model()

    scenario = model.scenario(milk_price_pct=0.05, feed_cost_pct=0.1)
    expected_columns = {
        "Revenue_adj",
        "COGS_adj",
        "Gross Margin_adj",
        "EBITDA_adj",
        "EBIT_adj",
        "NPAT_adj",
    }
    assert expected_columns.issubset(scenario.columns)

    performance = model.statement_of_financial_performance(scenario, annual=True)
    assert "Income – Revenue" in performance.columns
    assert "Cost of sales – Gross profit" in performance.columns
    assert "Operating profit – EBIT" in performance.columns
    assert "Profit – Profit for the period" in performance.columns
    assert not performance["Income – Revenue"].isna().all()
    assert not performance["Profit – Profit for the period"].isna().all()

    cash_flow = model.statement_of_cash_flow(scenario, annual=True)
    assert (
        "Operating activities – Net cash from operating activities"
        in cash_flow.columns
    )
    assert (
        "Net change – Cash and cash equivalents at end of period"
        in cash_flow.columns
    )

    position = model.statement_of_financial_position(scenario, annual=True)
    assert "Assets – Total assets" in position.columns
    assert (
        "Equity and liabilities – Total equity and liabilities"
        in position.columns
    )

    analytics = model.advanced_analytics(scenario, window=3, annual=True)
    assert {"sensitivity", "monte_carlo", "goal_seek"}.issubset(analytics.keys())
    sensitivity = analytics["sensitivity"]["tables"]["Impact Summary"]
    assert {"Revenue", "EBITDA", "NPAT"}.issubset(sensitivity.columns)
    monte_carlo = analytics["monte_carlo"]["tables"]["Summary Statistics"]
    assert "Mean" in monte_carlo.index
    assert "NPV" in monte_carlo.columns

    kpis = model.kpis(scenario, annual=True)
    assert "Gross Margin %" in kpis.columns

    break_even = model.break_even(scenario, annual=True)
    assert "Break-even Revenue" in break_even.columns


def test_annual_advanced_analytics_uses_costs():
    model = _build_model()
    scenario = model.scenario()

    analytics = model.advanced_analytics(scenario, window=3, annual=True)

    segments = analytics["segmentation"]["tables"]["Segment Contribution"]
    assert not segments.empty
    assert pytest.approx(0.55, rel=1e-6) == segments.iloc[0]["Margin %"]

    allocation = analytics["portfolio"]["tables"]["Allocation"]
    assert not allocation.empty
    assert pytest.approx(segments.iloc[0]["Margin %"], rel=1e-6) == allocation.iloc[0][
        "Expected Margin"
    ]


def test_input_schedule_rejects_non_numeric_columns():
    periods = pd.date_range("2024-01-31", periods=2, freq="M")
    schedule = pd.DataFrame({"Revenue": ["one", "two"]}, index=periods)

    with pytest.raises(ValueError):
        InputSchedule(data=schedule)


def test_ratios_mask_zero_or_negative_denominators():
    periods = pd.date_range("2024-01-31", periods=3, freq="M")
    schedule = pd.DataFrame(
        {
            "Revenue": [1000.0, 0.0, -500.0],
            "COGS": [400.0, 100.0, -200.0],
            "Gross Margin": [600.0, -100.0, -300.0],
            "EBITDA": [200.0, -50.0, -150.0],
            "NPAT": [100.0, -80.0, -200.0],
        },
        index=periods,
    )

    model = InputSchedule(data=schedule).to_model()

    kpis = model.kpis(annual=False)
    assert pytest.approx(kpis.loc[periods[0], "Gross Margin %"], rel=1e-6) == 0.6
    assert np.isnan(kpis.loc[periods[1], "Gross Margin %"])
    assert np.isnan(kpis.loc[periods[2], "Gross Margin %"])

    break_even = model.break_even(annual=False)
    assert pytest.approx(break_even.loc[periods[0], "Break-even Revenue"], rel=1e-6) == 666.6666667
    assert np.isnan(break_even.loc[periods[1], "Break-even Revenue"])
    assert np.isnan(break_even.loc[periods[2], "Break-even Revenue"])


def test_discounted_cash_flow_uses_model_ufcf_and_manual_terminal_value():
    periods = pd.date_range("2024-12-31", periods=3, freq="Y")
    valuation_inputs = {"WACC": 0.1, "Terminal Value": 200.0}
    supplementary_ufcf = pd.DataFrame({"Period": periods, "UFCF": [999.0, 999.0, 999.0]})

    model = _build_valuation_model(
        [100.0, 110.0, 120.0],
        valuation_inputs=valuation_inputs,
        supplementary_tables={"UFCF": supplementary_ufcf},
    )

    summary = model.discounted_cash_flow()

    cash_flows = summary["cash_flows"]
    expected_discount = [1 / (1.1**i) for i in range(1, 4)]
    expected_ufcf = [100.0, 110.0, 120.0]
    expected_terminal_pv = 200.0 / (1.1**4)
    expected_enterprise_value = sum(
        cash_flow * discount for cash_flow, discount in zip(expected_ufcf, expected_discount)
    ) + expected_terminal_pv

    assert list(cash_flows.columns) == ["UFCF", "Discount Factor", "Present Value"]
    assert cash_flows["UFCF"].tolist() == pytest.approx(expected_ufcf, rel=1e-6)
    assert cash_flows["Discount Factor"].tolist() == pytest.approx(expected_discount, rel=1e-6)
    assert summary["terminal_value"] == pytest.approx(200.0, rel=1e-6)
    assert summary["terminal_value_pv"] == pytest.approx(expected_terminal_pv, rel=1e-6)
    assert summary["enterprise_value"] == pytest.approx(expected_enterprise_value, rel=1e-6)


def test_discounted_cash_flow_irr_matches_explicit_cash_flows():
    model = _build_valuation_model(
        [-100.0, 60.0, 80.0],
        valuation_inputs={"WACC": 0.1, "Terminal Value": 0.0},
    )

    summary = model.discounted_cash_flow()
    roots = np.roots([-100.0, 60.0, 80.0])
    expected_root = max(root.real for root in roots if abs(root.imag) < 1e-9 and root.real > 0)
    expected_irr = expected_root - 1.0

    assert summary["terminal_value"] == pytest.approx(0.0, rel=1e-6)
    assert summary["irr"] == pytest.approx(expected_irr, rel=1e-6)


def test_model_audit_returns_structured_results():
    model = _build_model()

    audit = model.model_audit()

    assert {"status", "score", "headline", "summary", "issues", "reasoning"}.issubset(
        audit.keys()
    )
    assert isinstance(audit["summary"], pd.DataFrame)
    assert isinstance(audit["issues"], pd.DataFrame)
    assert isinstance(audit["reasoning"], list)


def test_model_audit_detects_reconciliation_and_valuation_errors():
    schedule = _build_sample_schedule()
    schedule["Gross Margin"] = schedule["Gross Margin"] + 500.0
    schedule["Closing Cash Balance"] = schedule["Closing Cash Balance"] + 2000.0

    model = InputSchedule(
        data=schedule,
        valuation_inputs={"WACC": 0.02, "Terminal Growth Rate": 0.03},
    ).to_model()

    audit = model.model_audit()
    issues = audit["issues"]

    assert audit["status"] == "critical"
    assert not issues.empty
    assert (
        (issues["Category"] == "Reconciliation") & (issues["Metric"] == "Gross Margin")
    ).any()
    assert (
        (issues["Category"] == "Reconciliation") & (issues["Metric"] == "Closing Cash")
    ).any()
    assert (
        (issues["Category"] == "Valuation")
        & (issues["Metric"] == "Terminal Growth Rate")
    ).any()
    assert any("structurally unstable" in str(note).lower() for note in issues["Reasoning"])


def test_execute_scenario_suite_runs_presets():
    schedule = _build_sample_schedule()
    suite = streamlit_app._build_scenario_suite()

    model, base, results = streamlit_app._execute_scenario_suite(
        schedule,
        {},
        {},
        suite,
    )

    pd.testing.assert_frame_equal(base, model.to_tidy())
    assert {
        "Base Case Scenario",
        "Best Case Scenario",
        "Worst Case Scenario",
    }.issubset(results.keys())

    for payload in results.values():
        assert "scenario" in payload
        assert "model_audit" in payload
        assert "Revenue_adj" in payload["scenario"].columns
        assert "Model Audit Summary" in payload["supplementary"]
        assert "Biological Herd Summary" in payload["supplementary"]


def test_biological_engine_drives_schedule_and_pricing_quantities():
    assumptions = streamlit_app._default_assumption_tables()
    pricing = streamlit_app._ensure_pricing_table(assumptions["Pricing"])
    pricing.loc[pricing["Product"] == "Cheese", "Active"] = False
    pricing.loc[pricing["Product"] == "Milk", "Active"] = True
    pricing.loc[pricing["Product"] == "Live Herd", "Active"] = True
    pricing.loc[pricing["Product"] == "Meat", "Active"] = True
    assumptions["Pricing"] = pricing

    core, detail_tables = streamlit_app._default_schedule_components(
        production_horizon=assumptions["Production Horizon"],
        assumptions=assumptions,
    )
    schedule = streamlit_app._build_schedule_dataframe(core, detail_tables, assumptions)
    synced = streamlit_app._sync_commercial_assumptions_to_core(assumptions, core)
    pricing_output = synced["Pricing"]

    first_period = schedule.index[0].strftime("%Y-%m-%d")
    first_schedule_row = schedule.iloc[0]

    assert {
        "Breeding Does",
        "Pregnant Does",
        "Milk Production (L)",
        "Slaughter Heads",
        "Live Herd Sales (heads)",
    }.issubset(schedule.columns)
    assert first_schedule_row["Herd Size (heads)"] > 0
    assert first_schedule_row["Milk Production (L)"] >= 0

    milk_row = pricing_output.loc[
        (pricing_output["Period"] == first_period)
        & (pricing_output["Product"] == "Milk")
    ].iloc[0]
    live_row = pricing_output.loc[
        (pricing_output["Period"] == first_period)
        & (pricing_output["Product"] == "Live Herd")
    ].iloc[0]
    meat_row = pricing_output.loc[
        (pricing_output["Period"] == first_period)
        & (pricing_output["Product"] == "Meat")
    ].iloc[0]

    assert milk_row["Quantity per Period"] == pytest.approx(
        first_schedule_row["Milk Production (L)"], rel=1e-6
    )
    assert live_row["Quantity per Period"] == pytest.approx(
        first_schedule_row["Live Herd Sales (heads)"], rel=1e-6
    )
    assert meat_row["Quantity per Period"] == pytest.approx(
        first_schedule_row["Meat Output Kg"], rel=1e-6
    )


def test_loan_facilities_drive_tidy_schedule_and_debt_schedule():
    model = _build_model_with_loan_facilities()

    tidy = model.to_tidy()
    debt_schedule = model.debt_schedule(annual=False)

    assert not debt_schedule.empty
    assert "Loan Name" in debt_schedule.columns
    assert pytest.approx(1200.0, rel=1e-6) == tidy.iloc[0]["Debt Drawdown"]
    assert pytest.approx(100.0, rel=1e-6) == tidy.iloc[0]["Principal Repayment"]
    assert pytest.approx(12.0, rel=1e-6) == tidy.iloc[0]["Interest Expense"]
    assert pytest.approx(1100.0, rel=1e-6) == tidy.iloc[0]["CFF"]
    assert pytest.approx(1100.0, rel=1e-6) == tidy.iloc[0]["Term Debt"]


def test_loan_facilities_support_mid_horizon_drawdown_in_debt_capacity():
    schedule = _build_sample_schedule()
    model = _build_model_with_loan_facilities(start_period=schedule.index[2])

    debt_capacity = model.debt_capacity_schedule(annual=False)

    assert pytest.approx(0.0, rel=1e-6) == debt_capacity.iloc[0]["Debt Drawdown"]
    assert pytest.approx(0.0, rel=1e-6) == debt_capacity.iloc[1]["Debt Drawdown"]
    assert pytest.approx(1200.0, rel=1e-6) == debt_capacity.iloc[2]["Debt Drawdown"]
    assert pytest.approx(1200.0, rel=1e-6) == debt_capacity.iloc[2]["Opening Debt"]


def test_annual_debt_capacity_recomputes_ratios_from_annual_totals():
    model = _build_model_with_loan_facilities()

    monthly = model.debt_capacity_schedule(annual=False)
    annual = model.debt_capacity_schedule(annual=True)
    tidy = model.to_tidy()

    assert len(annual) == 1
    annual_row = annual.iloc[0]
    expected_dscr = monthly["CFADS"].sum() / monthly["Debt Service"].sum()
    expected_interest_coverage = tidy["EBIT"].sum() / monthly["Interest Expense (Debt)"].sum()
    expected_cash_headroom = monthly["Closing Cash"].iloc[-1] - annual_row["Minimum Cash Reserve"]
    expected_net_debt_to_ebitda = annual_row["Net Debt"] / tidy["EBITDA"].sum()

    assert annual_row["Opening Debt"] == pytest.approx(monthly["Opening Debt"].iloc[0], rel=1e-6)
    assert annual_row["DSCR"] == pytest.approx(expected_dscr, rel=1e-6)
    assert annual_row["Interest Coverage"] == pytest.approx(
        expected_interest_coverage, rel=1e-6
    )
    assert annual_row["Cash Reserve Headroom"] == pytest.approx(
        expected_cash_headroom, rel=1e-6
    )
    assert annual_row["Net Debt / EBITDA"] == pytest.approx(
        expected_net_debt_to_ebitda, rel=1e-6
    )


def test_loan_facilities_flow_into_scenario_outputs():
    model = _build_model_with_loan_facilities()

    scenario = model.scenario()

    assert pytest.approx(12.0, rel=1e-6) == scenario.iloc[0]["Interest Expense_adj"]
    assert pytest.approx(1100.0, rel=1e-6) == scenario.iloc[0]["CFF_adj"]
    expected_close = (
        scenario.iloc[0]["Opening Cash Balance"]
        + scenario.iloc[0]["CFO_adj"]
        + scenario.iloc[0]["CFI_adj"]
        + scenario.iloc[0]["CFF_adj"]
    )
    assert pytest.approx(expected_close, rel=1e-6) == scenario.iloc[0]["Closing Cash Balance_adj"]


def test_equity_facilities_drive_tidy_schedule_and_equity_schedule():
    model = _build_model_with_equity_facilities()
    baseline = _build_sample_schedule()

    tidy = model.to_tidy()
    equity_schedule = model.equity_schedule(annual=False)

    assert not equity_schedule.empty
    assert pytest.approx(1500.0, rel=1e-6) == tidy.iloc[0]["Equity Contribution"]
    assert pytest.approx(50.0, rel=1e-6) == tidy.iloc[0]["Equity Issue Costs"]
    assert pytest.approx(1450.0, rel=1e-6) == tidy.iloc[0]["Net Equity Proceeds"]
    assert pytest.approx(1450.0, rel=1e-6) == tidy.iloc[0]["CFF"]
    assert pytest.approx(1450.0 - baseline.iloc[0]["CFF"], rel=1e-6) == (
        tidy.iloc[0]["Equity"] - baseline.iloc[0]["Equity"]
    )


def test_equity_facilities_support_mid_horizon_contribution():
    schedule = _build_sample_schedule()
    model = _build_model_with_equity_facilities(start_period=schedule.index[2])

    tidy = model.to_tidy()

    assert pytest.approx(0.0, rel=1e-6) == tidy.iloc[0]["Equity Contribution"]
    assert pytest.approx(0.0, rel=1e-6) == tidy.iloc[1]["Equity Contribution"]
    assert pytest.approx(1500.0, rel=1e-6) == tidy.iloc[2]["Equity Contribution"]
    assert pytest.approx(1450.0, rel=1e-6) == tidy.iloc[2]["CFF"]


def test_equity_facilities_flow_into_scenario_outputs():
    model = _build_model_with_equity_facilities()
    baseline = _build_sample_schedule()

    scenario = model.scenario()

    assert pytest.approx(1450.0, rel=1e-6) == scenario.iloc[0]["CFF_adj"]
    assert pytest.approx(1450.0 - baseline.iloc[0]["CFF"], rel=1e-6) == (
        scenario.iloc[0]["Equity_adj"] - scenario.iloc[0]["Equity"]
    )
    expected_close = (
        scenario.iloc[0]["Opening Cash Balance"]
        + scenario.iloc[0]["CFO_adj"]
        + scenario.iloc[0]["CFI_adj"]
        + scenario.iloc[0]["CFF_adj"]
    )
    assert pytest.approx(expected_close, rel=1e-6) == scenario.iloc[0]["Closing Cash Balance_adj"]
