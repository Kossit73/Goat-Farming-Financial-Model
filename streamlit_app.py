"""Interactive dashboard for the goat farming financial model."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st

from goat_financial_model import GoatModel, InputSchedule


st.set_page_config(page_title="Goat Farm Financial Model", layout="wide")


def _default_income_schedule(periods: int = 12, start: str = "2024-01-31") -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="M")
    revenue = np.linspace(45000, 70000, periods)
    cogs = revenue * 0.45
    variable = revenue * 0.12
    fixed = np.full(periods, 9000.0)
    direct = revenue * 0.08
    admin = np.full(periods, 3500.0)
    gross_margin = revenue - cogs
    ebitda = gross_margin - variable - fixed - direct - admin
    depreciation = np.full(periods, 2000.0)
    ebit = ebitda - depreciation
    npbt = ebit - 1500.0
    npat = npbt * 0.72
    cfo = ebitda - 2500.0
    cfi = np.full(periods, -3000.0)
    cff = np.full(periods, 1500.0)
    capex = np.full(periods, 2500.0)
    net_cf = cfo + cfi + cff
    current_assets = np.linspace(95000, 120000, periods)
    non_current_assets = np.linspace(210000, 245000, periods)
    current_liabilities = np.linspace(40000, 45000, periods)
    non_current_liabilities = np.linspace(85000, 90000, periods)
    equity = current_assets + non_current_assets - current_liabilities - non_current_liabilities

    df = pd.DataFrame(
        {
            "Period": dates.strftime("%Y-%m-%d"),
            "Revenue": revenue,
            "COGS": cogs,
            "Variable Expenses": variable,
            "Fixed Expenses": fixed,
            "Direct Wages": direct,
            "Admin Wages": admin,
            "Gross Margin": gross_margin,
            "EBITDA": ebitda,
            "Depreciation & Amortization": depreciation,
            "EBIT": ebit,
            "NPBT": npbt,
            "NPAT": npat,
            "CFO": cfo,
            "CFI": cfi,
            "CFF": cff,
            "Capex": capex,
            "Net Cash Flow": net_cf,
            "Current Assets": current_assets,
            "Non-current Assets": non_current_assets,
            "Current Liabilities": current_liabilities,
            "Non-current Liabilities": non_current_liabilities,
            "Equity": equity,
        }
    )
    return df


def _default_supplementary_tables() -> Dict[str, pd.DataFrame]:
    return {
        "Capitalisation Table": pd.DataFrame(
            {
                "Shareholder": ["Founder", "Investor"],
                "Ownership %": [60.0, 40.0],
                "Investment": [0.0, 250000.0],
            }
        ),
        "Capex Schedule": pd.DataFrame(
            {
                "Year": [2024, 2025],
                "Category": ["Milking Equipment", "Housing Upgrades"],
                "Spend": [45000.0, 38000.0],
            }
        ),
        "Asset Schedules": pd.DataFrame(
            {
                "Asset": ["Barn", "Parlour"],
                "Opening NBV": [120000.0, 65000.0],
                "Additions": [10000.0, 5000.0],
                "Depreciation": [8000.0, 4200.0],
            }
        ),
        "Outputs": pd.DataFrame(
            {
                "Metric": ["IRR", "Payback (years)"],
                "Value": [0.17, 4.2],
            }
        ),
        "Benchmark KPIs": pd.DataFrame(
            {
                "KPI": ["Milk Yield per Doe", "Feed Cost per Litre"],
                "Benchmark": [3.6, 0.18],
            }
        ),
    }


def _default_assumption_tables() -> Dict[str, pd.DataFrame]:
    return {
        "Production Horizon": pd.DataFrame(
            {
                "Start Year": [2024],
                "End Year": [2030],
            }
        ),
        "Pricing": pd.DataFrame(
            {
                "Product": ["Milk", "Cheese"],
                "Unit": ["Litre", "Kg"],
                "Base Price": [1.85, 12.50],
                "Price Growth %": [3.0, 2.5],
            }
        ),
        "Operating Costs": pd.DataFrame(
            {
                "Category": ["Feed", "Healthcare", "Utilities"],
                "Monthly Cost": [8500.0, 1800.0, 1200.0],
                "Inflation %": [4.0, 3.5, 2.0],
            }
        ),
        "Capital & Financing": pd.DataFrame(
            {
                "Source": ["Bank Loan", "Equity"],
                "Amount": [250000.0, 150000.0],
                "Interest/Return %": [6.5, 0.0],
                "Term (years)": [7, None],
            }
        ),
    }


def _prepare_schedule(df: pd.DataFrame) -> pd.DataFrame:
    if "Period" not in df.columns:
        raise ValueError("The schedule must include a 'Period' column with dates.")

    work = df.copy()
    work = work[work["Period"].astype(str).str.strip() != ""]
    periods = pd.to_datetime(work["Period"], errors="coerce")
    if periods.isna().any():
        raise ValueError("One or more period values could not be parsed as dates.")

    values = work.drop(columns=["Period"]).apply(pd.to_numeric, errors="coerce")
    values.index = pd.DatetimeIndex(periods)
    values.index.name = "Period"

    mask = values.notna().any(axis=1)
    values = values.loc[mask]
    if values.empty:
        raise ValueError("No numeric data supplied in the schedule.")
    if values.index.has_duplicates:
        raise ValueError("Each period in the schedule must be unique.")
    return values.sort_index()


def _clean_editor_table(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    if df is None or df.empty:
        return None
    work = df.dropna(how="all").dropna(axis=1, how="all")
    if work.empty:
        return None
    return work.reset_index(drop=True)


def _run_model(
    schedule_df: pd.DataFrame,
    valuation_inputs: Dict[str, float],
    supplementary_tables: Dict[str, pd.DataFrame],
    milk_pct: float,
    feed_pct: float,
):
    schedule = InputSchedule(
        data=schedule_df,
        valuation_inputs=valuation_inputs,
        supplementary_tables=supplementary_tables,
    )
    model = schedule.to_model()
    base = model.to_tidy()
    scenario = model.scenario(milk_price_pct=milk_pct, feed_cost_pct=feed_pct)

    kpis = model.kpis(scenario, annual=True)
    break_even = model.break_even(scenario, annual=True)

    return (
        model,
        base,
        scenario,
        kpis,
        break_even,
    )


def _render_table(title: str, table: Optional[pd.DataFrame]) -> None:
    if table is None:
        st.info(f"No **{title}** data was provided.")
        return
    st.subheader(title)
    st.dataframe(table)


def main() -> None:
    st.title("🐐 Goat Farm Financial Model — Interactive Scenario Dashboard")

    if "schedule" not in st.session_state:
        st.session_state.schedule = _default_income_schedule()
    if "supplementary" not in st.session_state:
        st.session_state.supplementary = _default_supplementary_tables()
    if "assumptions" not in st.session_state:
        st.session_state.assumptions = _default_assumption_tables()
    if "results" not in st.session_state:
        st.session_state.results = None

    milk_price = 0
    feed_cost = 0
    valuation_inputs: Dict[str, float] = {}
    include_valuation = False
    run_clicked = False

    tabs = st.tabs(["Input Schedule", "Assumptions"])

    with tabs[0]:
        st.subheader("Input Schedule")
        schedule_editor = st.data_editor(
            st.session_state.schedule,
            num_rows="dynamic",
            use_container_width=True,
            key="income_schedule",
        )
        st.session_state.schedule = schedule_editor

        st.markdown("### Supplementary Tables")
        supplementary_tables: Dict[str, pd.DataFrame] = {}
        for name, default_table in st.session_state.supplementary.items():
            with st.expander(name, expanded=False):
                table = st.data_editor(
                    default_table,
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"supp_{name}",
                )
            cleaned = _clean_editor_table(table)
            if cleaned is not None:
                supplementary_tables[name] = cleaned
            st.session_state.supplementary[name] = table

    assumption_tables: Dict[str, pd.DataFrame] = {}

    with tabs[1]:
        st.subheader("Assumptions")
        assumption_tabs = st.tabs(
            [
                "Scenario Controls",
                "Production Horizon",
                "Pricing",
                "Operating Costs",
                "Capital & Financing",
                "Valuation Inputs",
            ]
        )

        with assumption_tabs[0]:
            milk_price = st.slider(
                "Milk price change (%)", min_value=-50, max_value=50, value=0, step=1
            )
            feed_cost = st.slider(
                "Feed cost change (%)", min_value=-50, max_value=50, value=0, step=1
            )
            run_clicked = st.button("Run Scenario", type="primary")

        with assumption_tabs[1]:
            st.markdown("#### Production Time Horizon")
            production_editor = st.data_editor(
                st.session_state.assumptions["Production Horizon"],
                num_rows="dynamic",
                use_container_width=True,
                key="assump_production",
            )
            st.session_state.assumptions["Production Horizon"] = production_editor
            assumption_tables["Production Horizon"] = production_editor

        with assumption_tabs[2]:
            st.markdown("#### Pricing Assumptions")
            pricing_editor = st.data_editor(
                st.session_state.assumptions["Pricing"],
                num_rows="dynamic",
                use_container_width=True,
                key="assump_pricing",
            )
            st.session_state.assumptions["Pricing"] = pricing_editor
            assumption_tables["Pricing"] = pricing_editor

        with assumption_tabs[3]:
            st.markdown("#### Operating Cost Assumptions")
            op_cost_editor = st.data_editor(
                st.session_state.assumptions["Operating Costs"],
                num_rows="dynamic",
                use_container_width=True,
                key="assump_operating",
            )
            st.session_state.assumptions["Operating Costs"] = op_cost_editor
            assumption_tables["Operating Costs"] = op_cost_editor

        with assumption_tabs[4]:
            st.markdown("#### Capital & Financing Assumptions")
            capital_editor = st.data_editor(
                st.session_state.assumptions["Capital & Financing"],
                num_rows="dynamic",
                use_container_width=True,
                key="assump_capital",
            )
            st.session_state.assumptions["Capital & Financing"] = capital_editor
            assumption_tables["Capital & Financing"] = capital_editor

        with assumption_tabs[5]:
            include_valuation = st.checkbox("Include valuation inputs", value=True)
            if include_valuation:
                wacc_pct = st.number_input("WACC (%)", value=12.0, step=0.1)
                npv_value = st.number_input("NPV", value=750000.0, step=10000.0)
                terminal_value = st.number_input(
                    "Terminal Value", value=1500000.0, step=10000.0
                )
                valuation_inputs = {
                    "WACC": wacc_pct / 100.0,
                    "NPV": npv_value,
                    "Terminal Value": terminal_value,
                }

    # ensure supplementary_tables defined even if tabs[0] not executed (Streamlit rerun)
    supplementary_tables = locals().get("supplementary_tables", {})

    if run_clicked:
        try:
            schedule_df = _prepare_schedule(schedule_editor)
        except ValueError as exc:
            st.error(str(exc))
            return

        production_horizon = assumption_tables.get("Production Horizon")
        horizon_filtered = schedule_df
        if production_horizon is not None and not production_horizon.empty:
            start_year = pd.to_numeric(
                production_horizon.get("Start Year"), errors="coerce"
            ).dropna()
            end_year = pd.to_numeric(
                production_horizon.get("End Year"), errors="coerce"
            ).dropna()
            if not start_year.empty and not end_year.empty:
                start = int(start_year.iloc[0])
                end = int(end_year.iloc[0])
                if start > end:
                    st.error("Production start year must be before the end year.")
                    return
                mask = (horizon_filtered.index.year >= start) & (
                    horizon_filtered.index.year <= end
                )
                horizon_filtered = horizon_filtered.loc[mask]
                if horizon_filtered.empty:
                    st.error(
                        "No schedule periods fall within the selected production horizon."
                    )
                    return
                schedule_df = horizon_filtered

        combined_supplementary = dict(supplementary_tables)
        for name, table in assumption_tables.items():
            cleaned = _clean_editor_table(table)
            if cleaned is not None:
                combined_supplementary[f"Assumptions - {name}"] = cleaned

        try:
            (
                model,
                base,
                scenario,
                kpis,
                break_even,
            ) = _run_model(
                schedule_df,
                valuation_inputs,
                combined_supplementary,
                milk_price / 100.0,
                feed_cost / 100.0,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success("Scenario complete")

        st.session_state.results = {
            "model": model,
            "base": base,
            "scenario": scenario,
            "kpis": kpis,
            "break_even": break_even,
            "supplementary": combined_supplementary,
        }

    results = st.session_state.results

    if results is None:
        st.info(
            "Update the input schedule, adjust the sliders, and press *Run Scenario* "
            "to evaluate alternative assumptions."
        )
    else:
        model = results["model"]
        base = results["base"]
        scenario = results["scenario"]
        kpis = results["kpis"]
        break_even = results["break_even"]

        valuation_metrics = {
            "WACC": model.wacc(),
            "NPV": model.npv(),
            "Terminal Value": model.terminal_value(),
        }
        non_null_metrics = [val for val in valuation_metrics.values() if val is not None]
        if non_null_metrics:
            summary_cols = st.columns(len(non_null_metrics))
            idx = 0
            for label, value in valuation_metrics.items():
                if value is None:
                    continue
                if label == "WACC":
                    summary_cols[idx].metric(label, f"{value * 100:.2f}%")
                else:
                    summary_cols[idx].metric(label, f"{value:,.2f}")
                idx += 1

        st.subheader("KPIs (Annual)")
        st.dataframe(kpis.mul(100).round(2))

    display_tabs = st.tabs(
        [
            "Dashboard",
            "Statement of Financial Performance",
            "Statement of Financial Position",
            "Statement of Cash Flow",
            "Advanced Analytics",
            "Supplementary Schedules",
        ]
    )

    if results is None:
        with display_tabs[0]:
            st.info("Run the scenario to populate the dashboard charts.")
        with display_tabs[1]:
            st.info("Run the scenario to generate the Statement of Financial Performance.")
        with display_tabs[2]:
            st.info("Run the scenario to generate the Statement of Financial Position.")
        with display_tabs[3]:
            st.info("Run the scenario to generate the Statement of Cash Flow.")
        with display_tabs[4]:
            st.info("Run the scenario to view advanced analytics.")
        with display_tabs[5]:
            st.info("Supplementary schedules will appear once a scenario has been run.")
        return

    # Results available for rendering
    scenario = results["scenario"]
    break_even = results["break_even"]
    model = results["model"]
    base = results["base"]

    with display_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Revenue vs NPAT")
            st.line_chart(scenario[["Revenue_adj", "NPAT_adj"]])
            st.markdown("#### Expense Breakdown")
            expense_cols = [
                col
                for col in [
                    "COGS_adj",
                    "Variable Expenses",
                    "Fixed Expenses",
                    "Direct Wages",
                    "Admin Wages",
                ]
                if col in scenario
            ]
            if expense_cols:
                st.area_chart(scenario[expense_cols])
            else:
                st.info("Add expense series to the schedule to view this chart.")

        with col2:
            st.markdown("#### Gross Margin vs EBITDA")
            st.line_chart(scenario[["Gross Margin_adj", "EBITDA_adj"]])
            st.markdown("#### Break-even Revenue")
            st.bar_chart(break_even["Break-even Revenue"])

    with display_tabs[1]:
        try:
            sop_base = model.statement_of_financial_performance(base, annual=True)
            sop_scenario = model.statement_of_financial_performance(
                scenario, annual=True
            )
            st.dataframe(
                pd.concat({"Base": sop_base, "Scenario": sop_scenario}, axis=1)
                .swaplevel(axis=1)
                .sort_index(axis=1, level=0)
            )
        except ValueError as exc:
            st.info(str(exc))

    with display_tabs[2]:
        try:
            sofp = model.statement_of_financial_position(base, annual=True)
            st.dataframe(sofp)
        except ValueError as exc:
            st.info(str(exc))

    with display_tabs[3]:
        try:
            socf = model.statement_of_cash_flow(base, annual=True)
            st.dataframe(socf)
        except ValueError as exc:
            st.info(str(exc))

    with display_tabs[4]:
        try:
            adv_monthly = model.advanced_analytics(scenario, window=3, annual=False)
            adv_annual = model.advanced_analytics(scenario, window=3, annual=True)
            st.markdown("#### Monthly Advanced Analytics")
            st.dataframe(adv_monthly)
            st.markdown("#### Annual Advanced Analytics")
            st.dataframe(adv_annual)
        except ValueError as exc:
            st.info(str(exc))

    with display_tabs[5]:
        supplementary_render = results.get("supplementary", {})
        for name in [
            "Capitalisation Table",
            "Capex Schedule",
            "Asset Schedules",
            "Outputs",
            "Benchmark KPIs",
        ]:
            _render_table(name, supplementary_render.get(name))

    st.download_button(
        "Download Scenario CSV",
        scenario.to_csv().encode("utf-8"),
        file_name="scenario_timeseries.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
