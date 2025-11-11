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

    st.sidebar.header("Assumptions")
    milk_price = st.sidebar.slider(
        "Milk price change (%)", min_value=-50, max_value=50, value=0, step=1
    )
    feed_cost = st.sidebar.slider(
        "Feed cost change (%)", min_value=-50, max_value=50, value=0, step=1
    )

    include_valuation = st.sidebar.checkbox("Include valuation inputs", value=True)
    valuation_inputs: Dict[str, float] = {}
    if include_valuation:
        wacc_pct = st.sidebar.number_input("WACC (%)", value=12.0, step=0.1)
        npv_value = st.sidebar.number_input("NPV", value=750000.0, step=10000.0)
        terminal_value = st.sidebar.number_input(
            "Terminal Value", value=1500000.0, step=10000.0
        )
        valuation_inputs = {
            "WACC": wacc_pct / 100.0,
            "NPV": npv_value,
            "Terminal Value": terminal_value,
        }

    st.subheader("Input Schedule")
    schedule_editor = st.data_editor(
        st.session_state.schedule,
        num_rows="dynamic",
        use_container_width=True,
        key="income_schedule",
    )
    st.session_state.schedule = schedule_editor

    st.subheader("Supplementary Tables")
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

    run_clicked = st.sidebar.button("Run Scenario", type="primary")

    if run_clicked:
        try:
            schedule_df = _prepare_schedule(schedule_editor)
        except ValueError as exc:
            st.error(str(exc))
            return

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
                supplementary_tables,
                milk_price / 100.0,
                feed_cost / 100.0,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success("Scenario complete")

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

        tabs = st.tabs(
            [
                "Scenario Charts",
                "Income Statement",
                "Cash Flow",
                "Supplementary Schedules",
            ]
        )

        with tabs[0]:
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

        with tabs[1]:
            st.markdown("#### Base vs Scenario (Income Statement)")
            st.dataframe(
                pd.concat(
                    {
                        "Base": base,
                        "Scenario": scenario,
                    },
                    axis=1,
                ).swaplevel(axis=1)
            )

        with tabs[2]:
            cash_cols = [
                col
                for col in ["CFO", "CFI", "CFF", "Net Cash Flow", "Capex"]
                if col in base
            ]
            if cash_cols:
                st.markdown("#### Cash Flow Series")
                st.line_chart(base[cash_cols])
                st.dataframe(base[cash_cols])
            else:
                st.info("Add cash flow series to the schedule to unlock this view.")

        with tabs[3]:
            for name in [
                "Capitalisation Table",
                "Capex Schedule",
                "Asset Schedules",
                "Outputs",
                "Benchmark KPIs",
            ]:
                _render_table(name, supplementary_tables.get(name))

        st.download_button(
            "Download Scenario CSV",
            scenario.to_csv().encode("utf-8"),
            file_name="scenario_timeseries.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Update the input schedule, adjust the sliders, and press *Run Scenario* "
            "to evaluate alternative assumptions."
        )


if __name__ == "__main__":
    main()
