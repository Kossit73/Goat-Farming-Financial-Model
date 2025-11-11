"""Interactive dashboard for the goat farming financial model."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd
import streamlit as st
from pandas.tseries.offsets import MonthEnd

from goat_financial_model import GoatModel, InputSchedule


st.set_page_config(page_title="Goat Farm Financial Model", layout="wide")


DETAIL_SCHEDULE_COLUMNS = {
    "COGS Schedule": ["COGS"],
    "Variable Expenses Schedule": ["Variable Expenses"],
    "Direct Wages Schedule": ["Direct Wages"],
    "Admin Wages Schedule": ["Admin Wages"],
    "Capex Schedule": ["Capex"],
}


def _normalize_period(series: pd.Series) -> pd.Series:
    periods = pd.to_datetime(series, errors="coerce")
    formatted = periods.dt.strftime("%Y-%m-%d")
    return formatted.where(~periods.isna(), series.astype(str))


def _revenue_map(core: pd.DataFrame) -> Dict[str, float]:
    if "Period" not in core.columns or "Revenue" not in core.columns:
        return {}
    periods = _normalize_period(core["Period"])
    revenue = pd.to_numeric(core["Revenue"], errors="coerce")
    return {
        period: float(value)
        for period, value in zip(periods, revenue)
        if period and not np.isnan(value)
    }


def _sync_cogs_table(
    table: pd.DataFrame, core: pd.DataFrame, default_pct: float = 45.0
) -> pd.DataFrame:
    if table is None or table.empty:
        return table

    work = table.copy()
    work["Period"] = _normalize_period(work.get("Period", pd.Series(dtype=str)))

    if "COGS" not in work.columns:
        work["COGS"] = np.nan
    if "COGS %" not in work.columns:
        work["COGS %"] = np.nan

    revenue_lookup = _revenue_map(core)
    revenue_series = work["Period"].map(revenue_lookup)
    amount = pd.to_numeric(work["COGS"], errors="coerce")
    percent = pd.to_numeric(work["COGS %"], errors="coerce")

    # Treat fractions (0.45) as percentages (45%)
    fraction_mask = percent.notna() & (percent <= 1.0)
    percent.loc[fraction_mask] = percent.loc[fraction_mask] * 100.0

    percent_provided = percent.notna()
    amount_provided = amount.notna()
    revenue_available = revenue_series.notna() & (revenue_series != 0)

    # Where amount is provided alongside revenue, derive the percentage
    mask = revenue_available & amount_provided
    percent.loc[mask] = (amount.loc[mask] / revenue_series.loc[mask]) * 100.0

    # Where only the percentage is provided, back into the amount
    mask = revenue_available & percent_provided & ~amount_provided
    amount.loc[mask] = revenue_series.loc[mask] * (percent.loc[mask] / 100.0)

    # Where neither is supplied but revenue exists, fall back to default
    mask = revenue_available & ~amount_provided & ~percent_provided
    amount.loc[mask] = revenue_series.loc[mask] * (default_pct / 100.0)
    percent.loc[mask] = default_pct

    work["COGS"] = amount
    work["COGS %"] = percent

    ordered_cols = ["Period", "COGS %", "COGS"]
    remainder = [col for col in work.columns if col not in ordered_cols]
    return work[ordered_cols + remainder].reset_index(drop=True)


def _ensure_cogs_schedule(
    table: Optional[pd.DataFrame], core: pd.DataFrame, default_pct: float = 45.0
) -> pd.DataFrame:
    if table is None or table.empty:
        base_periods = _normalize_period(core.get("Period", pd.Series(dtype=str)))
        table = pd.DataFrame({"Period": base_periods, "COGS": np.nan})

    return _sync_cogs_table(table, core, default_pct=default_pct)


def _apply_cogs_percentage(
    table: pd.DataFrame, core: pd.DataFrame, percent: float
) -> pd.DataFrame:
    work = table.copy()
    work["COGS %"] = percent
    return _sync_cogs_table(work, core, default_pct=percent)


def _apply_yearly_increment(
    table: pd.DataFrame, core: pd.DataFrame, increment_pct: float, default_pct: float
) -> pd.DataFrame:
    if table is None or table.empty or increment_pct == 0:
        return table

    work = _sync_cogs_table(table, core, default_pct=default_pct)
    temp = work.copy()
    temp["__period_dt"] = pd.to_datetime(temp["Period"], errors="coerce")
    temp = temp.sort_values("__period_dt", kind="stable")

    increment_factor = 1 + (increment_pct / 100.0)
    current_pct = None
    current_year = None

    for idx, row in temp.iterrows():
        period_dt = row["__period_dt"]
        pct_value = row["COGS %"]
        if pd.isna(period_dt):
            continue
        if current_pct is None:
            current_pct = default_pct if pd.isna(pct_value) else float(pct_value)
            current_year = period_dt.year
            temp.at[idx, "COGS %"] = current_pct
            continue

        year_gap = period_dt.year - current_year if current_year is not None else 0
        if year_gap > 0:
            current_pct = current_pct * (increment_factor ** year_gap)
            current_year = period_dt.year
        temp.at[idx, "COGS %"] = current_pct

    temp = temp.drop(columns="__period_dt")
    work.loc[temp.index, "COGS %"] = temp["COGS %"]
    return _sync_cogs_table(work, core, default_pct=default_pct)


def _add_cogs_row(
    table: pd.DataFrame, core: pd.DataFrame, default_pct: float
) -> pd.DataFrame:
    work = _sync_cogs_table(table, core, default_pct=default_pct)
    periods = pd.to_datetime(work["Period"], errors="coerce")

    if periods.notna().any():
        last_period = periods.max()
    else:
        core_periods = pd.to_datetime(core.get("Period", pd.Series(dtype=str)), errors="coerce")
        if core_periods.notna().any():
            last_period = core_periods.max()
        else:
            last_period = pd.Timestamp.today() + MonthEnd(0)

    next_period = (last_period + MonthEnd(1)) if last_period is not None else pd.Timestamp.today() + MonthEnd(0)
    existing_periods = set(work["Period"].astype(str))
    while next_period.strftime("%Y-%m-%d") in existing_periods:
        next_period += MonthEnd(1)

    next_period_str = next_period.strftime("%Y-%m-%d")

    revenue_lookup = _revenue_map(core)
    revenue = revenue_lookup.get(next_period_str)

    last_pct_series = pd.to_numeric(work.get("COGS %"), errors="coerce").dropna()
    pct_value = float(last_pct_series.iloc[-1]) if not last_pct_series.empty else default_pct
    amount = revenue * (pct_value / 100.0) if revenue is not None else np.nan

    new_row = {"Period": next_period_str, "COGS %": pct_value, "COGS": amount}
    return _sync_cogs_table(pd.concat([work, pd.DataFrame([new_row])], ignore_index=True), core, default_pct=pct_value)


def _remove_cogs_row(table: pd.DataFrame, period: str) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    mask = table["Period"].astype(str) != str(period)
    return table.loc[mask].reset_index(drop=True)


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
    interest_expense = np.full(periods, 1500.0)
    npbt = ebit - interest_expense
    tax_expense = npbt * 0.28
    npat = npbt - tax_expense
    cfo = ebitda - 2500.0
    cfi = np.full(periods, -3000.0)
    cff = np.full(periods, 1500.0)
    capex = np.full(periods, 2500.0)
    net_cf = cfo + cfi + cff
    opening_cash = np.empty(periods)
    opening_cash[0] = 25000.0
    for i in range(1, periods):
        opening_cash[i] = opening_cash[i - 1] + net_cf[i - 1]
    closing_cash = opening_cash + net_cf
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
            "Interest Expense": interest_expense,
            "NPBT": npbt,
            "Tax Expense": tax_expense,
            "NPAT": npat,
            "CFO": cfo,
            "CFI": cfi,
            "CFF": cff,
            "Capex": capex,
            "Net Cash Flow": net_cf,
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
    return df


def _default_schedule_components(
    periods: int = 12, start: str = "2024-01-31"
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    base = _default_income_schedule(periods=periods, start=start)

    core_columns = [
        "Period",
        "Revenue",
        "Fixed Expenses",
        "Depreciation & Amortization",
        "EBIT",
        "Interest Expense",
        "NPBT",
        "Tax Expense",
        "NPAT",
        "CFO",
        "CFI",
        "CFF",
        "Net Cash Flow",
        "Opening Cash Balance",
        "Closing Cash Balance",
        "Cash and Cash Equivalents",
        "Current Assets",
        "Non-current Assets",
        "Current Liabilities",
        "Non-current Liabilities",
        "Equity",
    ]
    core_columns = [col for col in core_columns if col in base.columns]
    core = base[core_columns].copy()

    detail_tables: Dict[str, pd.DataFrame] = {}
    for name, cols in DETAIL_SCHEDULE_COLUMNS.items():
        detail_cols = ["Period"] + [col for col in cols if col in base.columns]
        detail_tables[name] = base[detail_cols].copy()

    return core, detail_tables


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


def _prepare_timeline_table(df: pd.DataFrame) -> pd.DataFrame:
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


def _assemble_schedule(
    core: pd.DataFrame, detail_tables: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    combined = core.copy()
    for table in detail_tables.values():
        combined = combined.join(table, how="outer")

    required_columns = [
        "Revenue",
        "COGS",
        "Gross Margin",
        "Variable Expenses",
        "Fixed Expenses",
        "Direct Wages",
        "Admin Wages",
        "EBITDA",
        "Depreciation & Amortization",
        "EBIT",
        "Interest Expense",
        "NPBT",
        "Tax Expense",
        "NPAT",
        "CFO",
        "CFI",
        "CFF",
        "Capex",
        "Net Cash Flow",
        "Opening Cash Balance",
        "Closing Cash Balance",
        "Cash and Cash Equivalents",
        "Current Assets",
        "Non-current Assets",
        "Current Liabilities",
        "Non-current Liabilities",
        "Equity",
    ]

    for col in required_columns:
        if col not in combined.columns:
            combined[col] = np.nan

    if {"Revenue", "COGS"}.issubset(combined.columns):
        combined["Gross Margin"] = combined["Revenue"] - combined["COGS"]

    if {
        "Gross Margin",
        "Variable Expenses",
        "Fixed Expenses",
        "Direct Wages",
        "Admin Wages",
    }.issubset(combined.columns):
        combined["EBITDA"] = (
            combined["Gross Margin"]
            - combined["Variable Expenses"].fillna(0)
            - combined["Fixed Expenses"].fillna(0)
            - combined["Direct Wages"].fillna(0)
            - combined["Admin Wages"].fillna(0)
        )

    if {"EBITDA", "Depreciation & Amortization"}.issubset(combined.columns):
        combined["EBIT"] = combined["EBITDA"] - combined["Depreciation & Amortization"].fillna(0)

    if "Interest Expense" not in combined.columns or combined["Interest Expense"].isna().all():
        if {"EBIT", "NPBT"}.issubset(combined.columns):
            combined["Interest Expense"] = (
                combined["EBIT"] - combined["NPBT"]
            )

    if "Tax Expense" not in combined.columns or combined["Tax Expense"].isna().all():
        if {"NPBT", "NPAT"}.issubset(combined.columns):
            combined["Tax Expense"] = combined["NPBT"] - combined["NPAT"]

    if {"CFO", "CFI", "CFF"}.issubset(combined.columns):
        combined["Net Cash Flow"] = combined[["CFO", "CFI", "CFF"]].sum(
            axis=1, min_count=1
        )

    ordered_columns = [
        "Revenue",
        "COGS",
        "Gross Margin",
        "Variable Expenses",
        "Fixed Expenses",
        "Direct Wages",
        "Admin Wages",
        "EBITDA",
        "Depreciation & Amortization",
        "EBIT",
        "Interest Expense",
        "NPBT",
        "Tax Expense",
        "NPAT",
        "CFO",
        "CFI",
        "CFF",
        "Capex",
        "Net Cash Flow",
        "Opening Cash Balance",
        "Closing Cash Balance",
        "Cash and Cash Equivalents",
        "Current Assets",
        "Non-current Assets",
        "Current Liabilities",
        "Non-current Liabilities",
        "Equity",
    ]

    ordered = [col for col in ordered_columns if col in combined.columns]
    remaining = [col for col in combined.columns if col not in ordered]

    combined = combined[ordered + remaining]
    return combined.sort_index()


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

    if "schedule" in st.session_state:
        st.session_state.pop("schedule")

    if "core_schedule" not in st.session_state or "detail_schedules" not in st.session_state:
        core_default, detail_defaults = _default_schedule_components()
        if "core_schedule" not in st.session_state:
            st.session_state.core_schedule = core_default
        if "detail_schedules" not in st.session_state:
            st.session_state.detail_schedules = detail_defaults
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

    supplementary_tables: Dict[str, pd.DataFrame] = {}
    detail_tables_for_run: Dict[str, pd.DataFrame] = {}
    core_editor: Optional[pd.DataFrame] = None

    tabs = st.tabs(
        [
            "Input Schedule",
            "Assumptions",
            "Financials",
            "Dashboard",
            "Advanced Analytics",
            "Supplementary Schedules",
        ]
    )

    with tabs[0]:
        st.subheader("Input Schedule")
        schedule_tab_names = [
            "Core Schedule",
            "COGS Schedule",
            "Variable Expenses Schedule",
            "Direct Wages Schedule",
            "Admin Wages Schedule",
            "Capex Schedule",
        ]
        schedule_tabs = st.tabs(schedule_tab_names)

        with schedule_tabs[0]:
            core_editor = st.data_editor(
                st.session_state.core_schedule,
                num_rows="dynamic",
                use_container_width=True,
                key="schedule_core",
            )
            st.session_state.core_schedule = core_editor

        for idx, name in enumerate(schedule_tab_names[1:], start=1):
            with schedule_tabs[idx]:
                if name == "COGS Schedule":
                    st.markdown("#### Cost of Goods Sold Schedule")
                    st.caption(
                        "Adjust COGS as a percentage of revenue, add or remove periods, and apply "
                        "automatic yearly increments. Amounts update automatically when revenue is available."
                    )

                    cogs_table = _ensure_cogs_schedule(
                        st.session_state.detail_schedules.get(name, pd.DataFrame()),
                        st.session_state.core_schedule,
                    )

                    inferred_pct = pd.to_numeric(
                        cogs_table.get("COGS %"), errors="coerce"
                    )
                    base_pct = (
                        float(inferred_pct.dropna().iloc[0])
                        if inferred_pct.notna().any()
                        else 45.0
                    )

                    st.session_state.setdefault("cogs_pct_input", round(base_pct, 2))
                    st.session_state.setdefault("cogs_increment_pct", 0.0)
                    st.session_state.setdefault("cogs_remove_choice", "Select a period")

                    controls = st.columns((1, 1, 1, 1))

                    pct_input = controls[0].number_input(
                        "Default COGS %",
                        min_value=0.0,
                        max_value=200.0,
                        step=0.1,
                        key="cogs_pct_input",
                    )
                    if controls[0].button("Apply % to all rows", key="cogs_apply_pct"):
                        cogs_table = _apply_cogs_percentage(
                            cogs_table, st.session_state.core_schedule, pct_input
                        )
                        st.session_state.detail_schedules[name] = cogs_table

                    increment_input = controls[1].number_input(
                        "Yearly increment %",
                        min_value=-100.0,
                        max_value=100.0,
                        step=0.1,
                        key="cogs_increment_pct",
                    )
                    if controls[1].button(
                        "Apply yearly increment", key="cogs_apply_increment"
                    ):
                        cogs_table = _apply_yearly_increment(
                            cogs_table,
                            st.session_state.core_schedule,
                            increment_input,
                            default_pct=pct_input,
                        )
                        st.session_state.detail_schedules[name] = cogs_table

                    if controls[2].button("Add Row", key="cogs_add_row"):
                        cogs_table = _add_cogs_row(
                            cogs_table,
                            st.session_state.core_schedule,
                            default_pct=pct_input,
                        )
                        st.session_state.detail_schedules[name] = cogs_table

                    remove_options = ["Select a period"] + cogs_table["Period"].astype(str).tolist()
                    controls[3].selectbox(
                        "Remove row",
                        options=remove_options,
                        key="cogs_remove_choice",
                    )
                    if controls[3].button("Remove", key="cogs_remove_row"):
                        remove_choice = st.session_state.get("cogs_remove_choice")
                        if (
                            remove_choice
                            and remove_choice in cogs_table["Period"].astype(str).values
                        ):
                            cogs_table = _remove_cogs_row(cogs_table, remove_choice)
                            st.session_state.detail_schedules[name] = cogs_table
                            st.session_state.cogs_remove_choice = "Select a period"

                    cogs_table = _sync_cogs_table(
                        cogs_table, st.session_state.core_schedule, default_pct=pct_input
                    )

                    editor = st.data_editor(
                        cogs_table,
                        num_rows="dynamic",
                        use_container_width=True,
                        column_config={
                            "COGS %": st.column_config.NumberColumn(
                                "COGS % of Revenue", format="%.2f %%", step=0.1
                            ),
                            "COGS": st.column_config.NumberColumn(
                                "COGS Amount", format="%.2f"
                            ),
                        },
                        key="schedule_cogs_schedule",
                    )
                    synced_editor = _sync_cogs_table(
                        editor, st.session_state.core_schedule, default_pct=pct_input
                    )
                    st.session_state.detail_schedules[name] = synced_editor
                    detail_tables_for_run[name] = synced_editor
                else:
                    table = st.data_editor(
                        st.session_state.detail_schedules.get(name, pd.DataFrame()),
                        num_rows="dynamic",
                        use_container_width=True,
                        key=f"schedule_{name.lower().replace(' ', '_')}",
                    )
                    st.session_state.detail_schedules[name] = table
                    detail_tables_for_run[name] = table

        st.markdown("### Supplementary Tables")
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

    if core_editor is None:
        core_editor = st.session_state.core_schedule

    for name, table in st.session_state.detail_schedules.items():
        detail_tables_for_run.setdefault(name, table)

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

    with tabs[2]:
        st.subheader("Financial Statements")
        if st.session_state.results is None:
            st.info("Run the scenario to generate the financial statements.")
        else:
            results = st.session_state.results
            financial_tabs = st.tabs(
                [
                    "Statement of Financial Performance",
                    "Statement of Financial Position",
                    "Statement of Cash Flow",
                ]
            )

            with financial_tabs[0]:
                try:
                    sop_base = results["model"].statement_of_financial_performance(
                        results["base"], annual=True
                    )
                    sop_scenario = results["model"].statement_of_financial_performance(
                        results["scenario"], annual=True
                    )
                    st.dataframe(
                        pd.concat({"Base": sop_base, "Scenario": sop_scenario}, axis=1)
                        .swaplevel(axis=1)
                        .sort_index(axis=1, level=0)
                    )
                except ValueError as exc:
                    st.info(str(exc))

            with financial_tabs[1]:
                try:
                    sofp = results["model"].statement_of_financial_position(
                        results["base"], annual=True
                    )
                    st.dataframe(sofp)
                except ValueError as exc:
                    st.info(str(exc))

            with financial_tabs[2]:
                try:
                    socf = results["model"].statement_of_cash_flow(
                        results["base"], annual=True
                    )
                    st.dataframe(socf)
                except ValueError as exc:
                    st.info(str(exc))

    if run_clicked:
        core_clean = _clean_editor_table(core_editor)
        if core_clean is None:
            st.error("Provide at least one period in the core schedule before running the scenario.")
            return

        try:
            core_prepared = _prepare_timeline_table(core_clean)
        except ValueError as exc:
            st.error(f"Core Schedule: {exc}")
            return

        prepared_details: Dict[str, pd.DataFrame] = {}
        for name, table in detail_tables_for_run.items():
            cleaned = _clean_editor_table(table)
            if cleaned is None:
                continue
            try:
                prepared = _prepare_timeline_table(cleaned)
            except ValueError as exc:
                st.error(f"{name}: {exc}")
                return

            expected_cols = DETAIL_SCHEDULE_COLUMNS.get(name)
            if expected_cols:
                missing = [col for col in expected_cols if col not in prepared.columns]
                if missing:
                    st.error(
                        f"{name} is missing required column(s): {', '.join(missing)}"
                    )
                    return
                prepared = prepared[expected_cols]
            prepared_details[name] = prepared

        schedule_df = _assemble_schedule(core_prepared, prepared_details)

        if schedule_df["Revenue"].isna().all():
            st.error("Core Schedule must include revenue values.")
            return
        if schedule_df["COGS"].isna().all():
            st.error("COGS Schedule must include at least one value.")
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

    with tabs[3]:
        st.subheader("Dashboard")
        if results is None:
            st.info("Run the scenario to populate the dashboard charts.")
        else:
            scenario = results["scenario"]
            break_even = results["break_even"]
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

            st.download_button(
                "Download Scenario CSV",
                scenario.to_csv().encode("utf-8"),
                file_name="scenario_timeseries.csv",
                mime="text/csv",
            )

    with tabs[4]:
        st.subheader("Advanced Analytics")
        if results is None:
            st.info("Run the scenario to view advanced analytics.")
        else:
            scenario = results["scenario"]
            model = results["model"]
            try:
                adv_monthly = model.advanced_analytics(scenario, window=3, annual=False)
                adv_annual = model.advanced_analytics(scenario, window=3, annual=True)
                st.markdown("#### Monthly Advanced Analytics")
                st.dataframe(adv_monthly)
                st.markdown("#### Annual Advanced Analytics")
                st.dataframe(adv_annual)
            except ValueError as exc:
                st.info(str(exc))

    with tabs[5]:
        st.subheader("Supplementary Schedules")
        if results is None:
            st.info("Supplementary schedules will appear once a scenario has been run.")
        else:
            supplementary_render = results.get("supplementary", {})
            for name in [
                "Capitalisation Table",
                "Capex Schedule",
                "Asset Schedules",
                "Outputs",
                "Benchmark KPIs",
            ]:
                _render_table(name, supplementary_render.get(name))


if __name__ == "__main__":
    main()
