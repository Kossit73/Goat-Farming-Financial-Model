"""Interactive dashboard for the goat farming financial model."""

from __future__ import annotations

import streamlit as st

from goat_financial_model import GoatModel


st.set_page_config(page_title="Goat Farm Financial Model", layout="wide")


@st.cache_data(show_spinner=False)
def _run_model(path: str, milk_pct: float, feed_pct: float):
    """Load the workbook and compute scenario outputs.

    The cached helper ensures we only hit the Excel file when inputs change,
    keeping the app responsive on repeated runs.
    """

    model = GoatModel(path)
    base = model.to_tidy()
    scen = model.scenario(milk_price_pct=milk_pct, feed_cost_pct=feed_pct)
    kpis = model.kpis(scen, annual=True)
    break_even = model.break_even(scen, annual=True)
    return base, scen, kpis, break_even


def main() -> None:
    st.title("🐐 Goat Farm Financial Model — Interactive Scenario Dashboard")

    st.sidebar.header("Workbook & Assumptions")
    excel_path = st.sidebar.text_input(
        "Excel file path",
        "/mnt/data/Goat-Farming-Financial-Model-Excel-Template-v1.1.xlsx",
    )
    milk_price = st.sidebar.slider(
        "Milk price change (%)", min_value=-50, max_value=50, value=0, step=1
    )
    feed_cost = st.sidebar.slider(
        "Feed cost change (%)", min_value=-50, max_value=50, value=0, step=1
    )

    if st.button("Run Scenario", type="primary"):
        try:
            with st.spinner("Running scenario..."):
                _base, scen, kpis, be = _run_model(
                    excel_path, milk_price / 100.0, feed_cost / 100.0
                )
        except FileNotFoundError:
            st.error(
                "Could not find the Excel workbook. Please check the path and try again."
            )
        except Exception as exc:  # noqa: BLE001 - surface full error to the analyst
            st.error(f"Something went wrong while processing the workbook: {exc}")
        else:
            st.subheader("KPIs (Annual)")
            st.dataframe(kpis.mul(100).round(2))

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Revenue vs NPAT (Scenario)")
                st.line_chart(scen[["Revenue_adj", "NPAT_adj"]])

            with col2:
                st.subheader("Gross Margin vs EBITDA (Scenario)")
                st.line_chart(scen[["Gross Margin_adj", "EBITDA_adj"]])

            st.subheader("Break-even (Annual)")
            st.dataframe(be.round(2))

            st.download_button(
                "Download Scenario CSV",
                scen.to_csv().encode("utf-8"),
                file_name="scenario_timeseries.csv",
                mime="text/csv",
            )
    else:
        st.info("Adjust the sliders and click *Run Scenario* to evaluate alternative assumptions.")


if __name__ == "__main__":
    main()
