from __future__ import annotations

from typing import Callable

import pandas as pd
import streamlit as st


def render_schedule_summary(title: str, table: pd.DataFrame) -> None:
    st.markdown(f"##### {title}")
    st.dataframe(table)


def render_cogs_schedule_editor(
    *,
    cogs_table: pd.DataFrame,
    core_schedule: pd.DataFrame,
    save_table: Callable[[pd.DataFrame], None],
    render_row_editor: Callable[[str, pd.DataFrame, Callable[[pd.DataFrame], None]], None],
    clear_editor_state: Callable[[str], None],
    apply_pct_fn: Callable[[pd.DataFrame, pd.DataFrame, float], pd.DataFrame],
    apply_increment_fn: Callable[[pd.DataFrame, pd.DataFrame, float, float], pd.DataFrame],
    add_row_fn: Callable[[pd.DataFrame, pd.DataFrame, float], pd.DataFrame],
    remove_row_fn: Callable[[pd.DataFrame, str], pd.DataFrame],
    sync_fn: Callable[[pd.DataFrame, pd.DataFrame, float], pd.DataFrame],
    ensure_fn: Callable[[pd.DataFrame, pd.DataFrame, float], pd.DataFrame],
) -> pd.DataFrame:
    inferred_pct = pd.to_numeric(cogs_table.get("COGS %"), errors="coerce")
    base_pct = float(inferred_pct.dropna().iloc[0]) if inferred_pct.notna().any() else 45.0
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
        cogs_table = apply_pct_fn(cogs_table, core_schedule, pct_input)
        save_table(cogs_table)
        clear_editor_state("detail::cogs_schedule")

    increment_input = controls[1].number_input(
        "Yearly increment %",
        min_value=-100.0,
        max_value=100.0,
        step=0.1,
        key="cogs_increment_pct",
    )
    if controls[1].button("Apply yearly increment", key="cogs_apply_increment"):
        cogs_table = apply_increment_fn(cogs_table, core_schedule, increment_input, pct_input)
        save_table(cogs_table)
        clear_editor_state("detail::cogs_schedule")

    if controls[2].button("Add Row", key="cogs_add_row"):
        cogs_table = add_row_fn(cogs_table, core_schedule, pct_input)
        save_table(cogs_table)
        clear_editor_state("detail::cogs_schedule")

    remove_options = ["Select a period"] + cogs_table["Period"].astype(str).tolist()
    controls[3].selectbox("Remove row", options=remove_options, key="cogs_remove_choice")
    if controls[3].button("Remove", key="cogs_remove_row"):
        remove_choice = st.session_state.get("cogs_remove_choice")
        if remove_choice and remove_choice in cogs_table["Period"].astype(str).values:
            cogs_table = remove_row_fn(cogs_table, remove_choice)
            save_table(cogs_table)
            st.session_state.cogs_remove_choice = "Select a period"
            clear_editor_state("detail::cogs_schedule")

    cogs_table = sync_fn(cogs_table, core_schedule, pct_input)
    save_table(cogs_table)
    render_row_editor(
        "detail::cogs_schedule",
        cogs_table,
        lambda updated: save_table(ensure_fn(updated, core_schedule, pct_input)),
    )
    return cogs_table
