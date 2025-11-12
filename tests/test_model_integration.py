import numpy as np
import pandas as pd
import pytest

from goat_financial_model import GoatModel, InputSchedule


def _build_sample_schedule() -> pd.DataFrame:
    periods = pd.date_range("2024-01-31", periods=12, freq="ME")
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
    assert list(performance.columns) == [
        "Revenue",
        "COGS",
        "Gross Profit",
        "Gross Profit Margin",
        "Variable Expenses",
        "Direct Wages",
        "EBITDA",
        "Fixed Expenses",
        "Admin Wages",
        "Depreciation",
        "EBIT",
        "Interest",
        "Tax",
        "Net Profit",
    ]
    assert not performance.isna().all().any()

    cash_flow = model.statement_of_cash_flow(scenario, annual=True)
    assert "Net cash from operating activities" in cash_flow.columns
    assert "Closing cash and cash equivalents" in cash_flow.columns

    position = model.statement_of_financial_position(scenario, annual=True)
    assert "Total Assets" in position.columns
    assert "Total Liabilities & Equity" in position.columns

    analytics = model.advanced_analytics(scenario, window=3, annual=True)
    assert {"sensitivity", "monte_carlo", "goal_seek"}.issubset(analytics.keys())
    sensitivity = analytics["sensitivity"]["tables"]["Impact Summary"]
    assert {"Revenue", "EBITDA", "NPAT"}.issubset(sensitivity.columns)
    monte_carlo = analytics["monte_carlo"]["tables"]["Summary Statistics"]
    assert "Mean" in monte_carlo.index

    kpis = model.kpis(scenario, annual=True)
    assert "Gross Margin %" in kpis.columns

    break_even = model.break_even(scenario, annual=True)
    assert "Break-even Revenue" in break_even.columns


def test_input_schedule_rejects_non_numeric_columns():
    periods = pd.date_range("2024-01-31", periods=2, freq="ME")
    schedule = pd.DataFrame({"Revenue": ["one", "two"]}, index=periods)

    with pytest.raises(ValueError):
        InputSchedule(data=schedule)


def test_ratios_mask_zero_or_negative_denominators():
    periods = pd.date_range("2024-01-31", periods=3, freq="ME")
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


def test_discounted_cash_flow_summary():
    periods = pd.date_range("2024-12-31", periods=3, freq="YE")
    schedule = pd.DataFrame({"Revenue": [1.0, 1.0, 1.0]}, index=periods)

    valuation_inputs = {"WACC": 0.1, "Terminal Value": 200000.0}
    ufcf_table = pd.DataFrame({"Period": periods, "UFCF": [10000.0, 12000.0, 15000.0]})

    model = InputSchedule(
        data=schedule,
        valuation_inputs=valuation_inputs,
        supplementary_tables={"UFCF": ufcf_table},
    ).to_model()

    summary = model.discounted_cash_flow()

    cash_flows = summary["cash_flows"]
    assert list(cash_flows.columns) == ["UFCF", "Discount Factor", "Present Value"]
    assert cash_flows["Discount Factor"].iloc[0] < 1
    assert "terminal_value_pv" in summary

    expected_discount = [1 / (1.1**i) for i in range(1, 4)]
    assert cash_flows["Discount Factor"].tolist() == pytest.approx(expected_discount, rel=1e-3)

    pv_total = cash_flows["Present Value"].sum() + summary["terminal_value_pv"]
    assert summary["enterprise_value"] == pytest.approx(pv_total, rel=1e-6)
