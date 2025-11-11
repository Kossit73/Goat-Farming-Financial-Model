"""Interactive dashboard for the goat farming financial model."""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd
import streamlit as st

from goat_financial_model import GoatModel


st.set_page_config(page_title="Goat Farm Financial Model", layout="wide")


@st.cache_data(show_spinner=False)
def _run_model(
    workbook_bytes: Optional[bytes],
    workbook_path: Optional[str],
    milk_pct: float,
    feed_pct: float,
):
    """Load the workbook, compute scenario outputs, and fetch schedules."""

    if workbook_bytes is not None:
        source = io.BytesIO(workbook_bytes)
    elif workbook_path:
        source = workbook_path
    else:
        raise FileNotFoundError("No workbook provided")

    model = GoatModel(source)
    base = model.to_tidy()
    scen = model.scenario(milk_price_pct=milk_pct, feed_cost_pct=feed_pct)
    kpis = model.kpis(scen, annual=True)
    break_even = model.break_even(scen, annual=True)

    capitalisation = model.capitalisation_table()
    capex_schedule = model.capex_schedule()
    asset_schedules = model.asset_schedules()
    outputs = model.outputs()
    benchmarks = model.benchmark_kpis()

    valuation_metrics = {
        "WACC": model.wacc(),
        "NPV": model.npv(),
    }

    return (
        base,
        scen,
        kpis,
        break_even,
        capitalisation,
        capex_schedule,
        asset_schedules,
        outputs,
        benchmarks,
        valuation_metrics,
    )


def _render_table(title: str, table: Optional[pd.DataFrame]) -> None:
    if table is None:
        st.info(f"No **{title}** data found in the workbook.")
        return
    st.subheader(title)
    st.dataframe(table)


def main() -> None:
    st.title("🐐 Goat Farm Financial Model — Interactive Scenario Dashboard")

    st.sidebar.header("Workbook & Assumptions")
    uploaded = st.sidebar.file_uploader(
        "Upload Excel workbook", type=["xlsx", "xls", "xlsm"], accept_multiple_files=False
    )
    excel_path = st.sidebar.text_input(
        "Or provide a server path", "", help="Useful when running the app next to the workbook."
    )

    milk_price = st.sidebar.slider(
        "Milk price change (%)", min_value=-50, max_value=50, value=0, step=1
    )
    feed_cost = st.sidebar.slider(
        "Feed cost change (%)", min_value=-50, max_value=50, value=0, step=1
    )

    run_clicked = st.sidebar.button("Run Scenario", type="primary")

    if run_clicked:
        workbook_bytes = uploaded.getvalue() if uploaded is not None else None
        try:
            with st.spinner("Processing workbook and running scenario..."):
                (
                    base,
                    scen,
                    kpis,
                    break_even,
                    capitalisation,
                    capex_schedule,
                    asset_schedules,
                    outputs,
                    benchmarks,
                    valuation,
                ) = _run_model(
                    workbook_bytes,
                    excel_path or None,
                    milk_price / 100.0,
                    feed_cost / 100.0,
                )
        except FileNotFoundError:
            st.error(
                "No workbook supplied. Upload a file or provide a valid path in the sidebar."
            )
            return
        except Exception as exc:  # noqa: BLE001
            st.error(
                "Something went wrong while processing the workbook. "
                "Please verify the template matches the GoatModel expectations."
            )
            st.exception(exc)
            return

        st.success("Scenario complete")

        summary_cols = st.columns(len([v for v in valuation.values() if v is not None]) or 1)
        summary_idx = 0
        for label, value in valuation.items():
            if value is None:
                continue
            summary_cols[summary_idx].metric(label, f"{value:,.2f}")
            summary_idx += 1

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
                st.line_chart(scen[["Revenue_adj", "NPAT_adj"]])
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
                    if col in scen
                ]
                if expense_cols:
                    st.area_chart(scen[expense_cols])
                else:
                    st.info("Expense series were not found in the workbook.")

            with col2:
                st.markdown("#### Gross Margin vs EBITDA")
                st.line_chart(scen[["Gross Margin_adj", "EBITDA_adj"]])
                st.markdown("#### Break-even Revenue")
                st.bar_chart(break_even["Break-even Revenue"])

        with tabs[1]:
            st.markdown("#### Base vs Scenario (Income Statement)")
            st.dataframe(
                pd.concat(
                    {
                        "Base": base,
                        "Scenario": scen,
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
                st.info("No cash flow series detected in the workbook.")

        with tabs[3]:
            _render_table("Capitalisation Table", capitalisation)
            _render_table("Capex Schedule", capex_schedule)
            _render_table("Asset Schedules", asset_schedules)
            _render_table("Model Outputs", outputs)
            _render_table("Benchmark KPIs", benchmarks)

        st.download_button(
            "Download Scenario CSV",
            scen.to_csv().encode("utf-8"),
            file_name="scenario_timeseries.csv",
            mime="text/csv",
        )

    else:
        st.info(
            "Upload a workbook or provide a path, adjust the sliders, and press *Run Scenario* "
            "to evaluate alternative assumptions."
        )


if __name__ == "__main__":
    main()
