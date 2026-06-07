from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st


EnsureSchedule = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
AddScheduleRow = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
RemoveScheduleRow = Callable[[pd.DataFrame, int], pd.DataFrame]
ApplyIncrement = Callable[[pd.DataFrame, float, Optional[str]], pd.DataFrame]
AggregateSchedule = Callable[[pd.DataFrame, pd.DataFrame], pd.DataFrame]
LabelBuilder = Callable[[pd.Series], str]
SaveEditor = Callable[[str, pd.DataFrame], None]
RenderRowEditor = Callable[[str, pd.DataFrame, Callable[[pd.DataFrame], None]], None]
ClearEditorState = Callable[[str], None]


@dataclass(frozen=True)
class ScheduleEditorSpec:
    schedule_name: str
    editor_key: str
    remove_choice_key: str
    increment_target_key: str
    increment_pct_key: str
    add_button_key: str
    remove_button_key: str
    increment_button_key: str
    ensure_fn: EnsureSchedule
    add_row_fn: AddScheduleRow
    remove_row_fn: RemoveScheduleRow
    apply_increment_fn: ApplyIncrement
    aggregate_fn: AggregateSchedule
    label_builder: LabelBuilder
    target_options_builder: Callable[[pd.DataFrame], list[str]]
    empty_remove_label: str = "-- Select Row --"
    all_items_label: str = "All items"


def build_remove_options(
    table: pd.DataFrame,
    label_builder: LabelBuilder,
) -> tuple[list[str], dict[str, int]]:
    labels: list[str] = []
    index_lookup: dict[str, int] = {}
    for row_index, row in table.iterrows():
        label = label_builder(row)
        labels.append(label)
        index_lookup[label] = row_index
    return labels, index_lookup


def render_incremental_schedule_editor(
    *,
    spec: ScheduleEditorSpec,
    table: pd.DataFrame,
    core_schedule: pd.DataFrame,
    save_table: SaveEditor,
    render_row_editor: RenderRowEditor,
    clear_editor_state: ClearEditorState,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensured = spec.ensure_fn(table, core_schedule)
    save_table(spec.schedule_name, ensured)

    st.session_state.setdefault(spec.remove_choice_key, spec.empty_remove_label)
    st.session_state.setdefault(spec.increment_target_key, spec.all_items_label)
    st.session_state.setdefault(spec.increment_pct_key, 0.0)

    add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])
    if add_col.button("Add Row", key=spec.add_button_key):
        ensured = spec.add_row_fn(ensured, core_schedule)
        save_table(spec.schedule_name, ensured)
        clear_editor_state(spec.editor_key)

    option_labels, option_index = build_remove_options(ensured, spec.label_builder)
    remove_select_col.selectbox(
        "Select row",
        options=[spec.empty_remove_label] + option_labels,
        key=spec.remove_choice_key,
    )
    if remove_btn_col.button("Remove Row", key=spec.remove_button_key):
        choice = st.session_state.get(spec.remove_choice_key)
        if choice in option_index:
            ensured = spec.remove_row_fn(ensured, option_index[choice])
            save_table(spec.schedule_name, ensured)
            st.session_state[spec.remove_choice_key] = spec.empty_remove_label
            clear_editor_state(spec.editor_key)

    inc_target_col, inc_pct_col, inc_btn_col = st.columns([2, 1, 1])
    inc_target_col.selectbox(
        "Apply increment to",
        options=spec.target_options_builder(ensured),
        key=spec.increment_target_key,
    )
    inc_pct_col.number_input(
        "Yearly increment (%)",
        min_value=-100.0,
        max_value=100.0,
        step=0.1,
        key=spec.increment_pct_key,
    )
    if inc_btn_col.button("Apply increment", key=spec.increment_button_key):
        ensured = spec.apply_increment_fn(
            ensured,
            float(st.session_state.get(spec.increment_pct_key, 0.0)),
            st.session_state.get(spec.increment_target_key),
        )
        save_table(spec.schedule_name, ensured)
        clear_editor_state(spec.editor_key)

    def _save(updated: pd.DataFrame) -> None:
        save_table(spec.schedule_name, spec.ensure_fn(updated, core_schedule))

    render_row_editor(spec.editor_key, ensured, _save)
    refreshed = spec.ensure_fn(st.session_state.get("detail_schedules", {}).get(spec.schedule_name, ensured), core_schedule)
    aggregated = spec.aggregate_fn(refreshed, core_schedule)
    return refreshed, aggregated
