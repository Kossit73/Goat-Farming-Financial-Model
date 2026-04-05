"""Interactive dashboard for the goat farming financial model."""

from __future__ import annotations

from copy import deepcopy
from importlib.util import find_spec
from io import BytesIO
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from copy import deepcopy

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
)
from pandas.tseries.offsets import MonthEnd
from streamlit.delta_generator import DeltaGenerator

from goat_financial_model import GoatModel, InputSchedule


try:
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except Exception:  # pragma: no cover - fallback for older Streamlit builds
    get_script_run_ctx = None

try:  # pragma: no cover - import guard for Streamlit API variations
    from streamlit.errors import StreamlitAPIException
except Exception:  # pragma: no cover - older versions exposed the exception elsewhere
    StreamlitAPIException = Exception


def _can_rerun() -> bool:
    """Return True when the app is executing within a Streamlit runtime."""

    if get_script_run_ctx is None:
        return False
    try:
        return get_script_run_ctx() is not None
    except Exception:  # pragma: no cover - defensive guard for API changes
        return False


def _maybe_rerun() -> None:
    """Invoke Streamlit rerun when a runtime context is active."""

    rerun_fn = getattr(st, "experimental_rerun", None) or getattr(st, "rerun", None)
    if rerun_fn is None or not _can_rerun():
        return
    rerun_fn()


def _safe_session_state_get(key: str, default: Any = None) -> Any:
    """Return a session state value without raising outside a Streamlit run."""

    try:
        return st.session_state.get(key, default)
    except StreamlitAPIException:
        return default


def _safe_session_state_setdefault(key: str, value: Any) -> Any:
    """Set a default session state value when a runtime context exists."""

    try:
        return st.session_state.setdefault(key, value)
    except StreamlitAPIException:
        return value


def _safe_session_state_set(key: str, value: Any) -> None:
    """Assign a session state value when supported by the runtime."""

    try:
        st.session_state[key] = value
    except StreamlitAPIException:
        pass


def _safe_session_state_contains(key: str) -> bool:
    """Return True when the session state currently tracks the key."""

    try:
        return key in st.session_state
    except StreamlitAPIException:
        return False


def _safe_session_state_pop(key: str, default: Any = None) -> Any:
    """Remove a session state key without raising when unavailable."""

    try:
        return st.session_state.pop(key, default)
    except StreamlitAPIException:
        return default


st.set_page_config(page_title="Goat Farm Financial Model", layout="wide")


AI_PROVIDER_OPTIONS = ("OpenAI", "Azure OpenAI", "Anthropic")

DEFAULT_VALUATION_INPUTS = {
    "WACC": 0.12,
    "NPV": 750000.0,
    "Terminal Value": 1500000.0,
}

ML_METHOD_LABELS = {
    "linear_regression": "Linear Regression",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
}

ML_LABEL_TO_CODE = {label: code for code, label in ML_METHOD_LABELS.items()}

GEN_AI_FEATURE_LABELS = {
    "summary": "Executive Summary",
    "risk_analysis": "Risk Analysis",
    "opportunity_analysis": "Opportunity Analysis",
}

GEN_AI_LABEL_TO_CODE = {label: code for code, label in GEN_AI_FEATURE_LABELS.items()}


DEFAULT_MODEL_AUTHOR = "Goat Farmers United"
MODEL_AUTHOR_KEY = "model_author"
MODEL_AUTHOR_WIDGET_KEY = "_model_author_widget"
MODEL_AUTHOR_CACHE_KEY = "_model_author_cached"


def _sanitize_model_author_value(value: Any) -> str:
    """Return a cleaned author string using the default when empty."""

    if not isinstance(value, str):
        value = str(value)
    return value.strip() or DEFAULT_MODEL_AUTHOR


def _handle_model_author_change() -> None:
    """Persist author edits and clear cached exports when updated."""

    if MODEL_AUTHOR_WIDGET_KEY not in st.session_state:
        return

    raw_value = st.session_state.get(MODEL_AUTHOR_WIDGET_KEY, "")
    sanitized = _sanitize_model_author_value(raw_value)
    previous = st.session_state.get(MODEL_AUTHOR_CACHE_KEY)

    # Streamlit raises when callbacks mutate keys that have not been declared yet
    # in bare execution. Ensure the storage key exists before assignment.
    if MODEL_AUTHOR_KEY not in st.session_state:
        st.session_state.setdefault(MODEL_AUTHOR_KEY, sanitized)
    else:
        st.session_state[MODEL_AUTHOR_KEY] = sanitized

    st.session_state[MODEL_AUTHOR_CACHE_KEY] = sanitized

    if sanitized != raw_value:
        st.session_state[MODEL_AUTHOR_WIDGET_KEY] = sanitized

    if previous is not None and sanitized != previous:
        st.session_state.pop("excel_bytes_map", None)


def _current_model_author() -> str:
    """Return the active model author, applying defaults when necessary."""

    current = st.session_state.get(MODEL_AUTHOR_KEY, DEFAULT_MODEL_AUTHOR)
    sanitized = _sanitize_model_author_value(current)
    if sanitized != current:
        st.session_state[MODEL_AUTHOR_KEY] = sanitized
    st.session_state.setdefault(MODEL_AUTHOR_CACHE_KEY, sanitized)
    st.session_state.setdefault(MODEL_AUTHOR_WIDGET_KEY, sanitized)
    widget_value = st.session_state.get(MODEL_AUTHOR_WIDGET_KEY)
    if widget_value != sanitized:
        st.session_state[MODEL_AUTHOR_WIDGET_KEY] = sanitized
    return sanitized


def _render_model_author_editor() -> None:
    """Display an inline editor for the model author name."""

    author_value = _current_model_author()
    st.session_state.setdefault(MODEL_AUTHOR_WIDGET_KEY, author_value)
    st.text_input(
        "Model author",
        value=st.session_state.get(MODEL_AUTHOR_WIDGET_KEY, author_value),
        key=MODEL_AUTHOR_WIDGET_KEY,
        help=(
            "Name recorded in scenario outputs and Excel downloads. "
            "Leave blank to reset to the default."
        ),
    )
    _handle_model_author_change()


def _statement_series_by_suffix(
    df: Optional[pd.DataFrame], suffixes: Sequence[str]
) -> Optional[pd.Series]:
    """Return the first numeric series whose column label ends with any suffix."""

    if df is None or df.empty:
        return None

    for suffix in suffixes:
        for column in df.columns:
            if not isinstance(column, str):
                continue
            if column.endswith(suffix):
                series = pd.to_numeric(df[column], errors="coerce")
                if series.notna().any():
                    return series
    return None


def _statement_scenario_frames(
    base_df: Optional[pd.DataFrame],
    scenario_df: Optional[pd.DataFrame],
    scenario_label: str,
) -> Dict[str, pd.DataFrame]:
    """Assemble labelled dataframes for the base case and selected scenario."""

    frames: Dict[str, pd.DataFrame] = {}

    if base_df is not None and not base_df.empty:
        frames["Base Case"] = base_df

    if scenario_df is not None and not scenario_df.empty:
        label = (scenario_label or "Selected Scenario").strip()
        if label.lower() in {"base", "base case", "base case scenario"}:
            label = "Selected Scenario"
        if label in frames:
            label = f"{label} (Selected)"
        frames[label] = scenario_df

    return frames


def _build_statement_chart_frame(
    frames: Dict[str, pd.DataFrame],
    metrics: Sequence[Tuple[str, Sequence[str]]],
) -> Optional[pd.DataFrame]:
    """Construct a combined dataframe for charting the requested metrics."""

    chart_frames: list[pd.DataFrame] = []

    for scenario_label, df in frames.items():
        if df is None or df.empty:
            continue

        scenario_columns: Dict[str, pd.Series] = {}
        for display_name, suffixes in metrics:
            series = _statement_series_by_suffix(df, suffixes)
            if series is None:
                continue
            scenario_columns[f"{scenario_label} – {display_name}"] = series

        if scenario_columns:
            chart_frames.append(pd.DataFrame(scenario_columns))

    if not chart_frames:
        return None

    combined = pd.concat(chart_frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.sort_index()


def _build_margin_chart_frame(
    frames: Dict[str, pd.DataFrame]
) -> Optional[pd.DataFrame]:
    """Compute gross and profit margin percentages for available scenarios."""

    margin_frames: list[pd.DataFrame] = []

    for scenario_label, df in frames.items():
        revenue = _statement_series_by_suffix(df, ("Income – Revenue",))
        if revenue is None:
            continue
        revenue = revenue.replace({0.0: np.nan})

        gross_profit = _statement_series_by_suffix(
            df, ("Cost of sales – Gross profit",)
        )
        profit_for_period = _statement_series_by_suffix(
            df, ("Profit – Profit for the period",)
        )

        margin_columns: Dict[str, pd.Series] = {}

        if gross_profit is not None:
            gross_margin = gross_profit.divide(revenue) * 100.0
            margin_columns[f"{scenario_label} – Gross margin (%)"] = gross_margin

        if profit_for_period is not None:
            profit_margin = profit_for_period.divide(revenue) * 100.0
            margin_columns[f"{scenario_label} – Profit margin (%)"] = profit_margin

        if margin_columns:
            margin_frames.append(pd.DataFrame(margin_columns))

    if not margin_frames:
        return None

    combined = pd.concat(margin_frames, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    return combined.sort_index()


def _render_financial_performance_charts(
    base_df: Optional[pd.DataFrame],
    scenario_df: Optional[pd.DataFrame],
    scenario_label: str,
) -> None:
    frames = _statement_scenario_frames(base_df, scenario_df, scenario_label)
    if not frames:
        return

    trend_metrics = [
        ("Revenue", ("Income – Revenue",)),
        ("Gross profit", ("Cost of sales – Gross profit",)),
        ("Profit for the period", ("Profit – Profit for the period",)),
    ]

    expense_metrics = [
        ("Distribution costs", ("Operating expenses – Distribution costs",)),
        ("Administrative expenses", ("Operating expenses – Administrative expenses",)),
        (
            "Depreciation and amortisation",
            ("Operating expenses – Depreciation and amortisation",),
        ),
        ("Other operating expenses", ("Operating expenses – Other operating expenses",)),
    ]

    trend_data = _build_statement_chart_frame(frames, trend_metrics)
    expense_data = _build_statement_chart_frame(frames, expense_metrics)
    margin_data = _build_margin_chart_frame(frames)

    if trend_data is not None:
        st.markdown("###### Income and profit trends")
        st.line_chart(trend_data)

    if expense_data is not None:
        st.markdown("###### Operating expense mix")
        st.bar_chart(expense_data)

    if margin_data is not None:
        st.markdown("###### Margin analysis")
        st.line_chart(margin_data)


def _render_financial_position_charts(
    base_df: Optional[pd.DataFrame],
    scenario_df: Optional[pd.DataFrame],
    scenario_label: str,
) -> None:
    frames = _statement_scenario_frames(base_df, scenario_df, scenario_label)
    if not frames:
        return

    balance_metrics = [
        ("Total assets", ("Assets – Total assets",)),
        ("Total liabilities", ("Equity and liabilities – Total liabilities",)),
        ("Equity", ("Equity and liabilities – Equity",)),
    ]

    net_metrics = [
        ("Net assets", ("Key metrics – Net assets",)),
        ("Net current assets", ("Key metrics – Net current assets",)),
    ]

    balance_data = _build_statement_chart_frame(frames, balance_metrics)
    net_data = _build_statement_chart_frame(frames, net_metrics)

    if balance_data is not None:
        st.markdown("###### Balance sheet totals")
        st.line_chart(balance_data)

    if net_data is not None:
        st.markdown("###### Net asset metrics")
        st.bar_chart(net_data)


def _render_cash_flow_charts(
    base_df: Optional[pd.DataFrame],
    scenario_df: Optional[pd.DataFrame],
    scenario_label: str,
) -> None:
    frames = _statement_scenario_frames(base_df, scenario_df, scenario_label)
    if not frames:
        return

    activity_metrics = [
        (
            "Operating activities",
            ("Operating activities – Net cash from operating activities",),
        ),
        (
            "Investing activities",
            ("Investing activities – Net cash used in investing activities",),
        ),
        (
            "Financing activities",
            ("Financing activities – Net cash from financing activities",),
        ),
    ]

    cash_balance_metrics = [
        (
            "Opening cash",
            ("Net change – Cash and cash equivalents at beginning of period",),
        ),
        (
            "Closing cash",
            ("Net change – Cash and cash equivalents at end of period",),
        ),
        (
            "Net change",
            (
                "Net change – Net increase/(decrease) in cash and cash equivalents",
            ),
        ),
    ]

    activity_data = _build_statement_chart_frame(frames, activity_metrics)
    cash_balance_data = _build_statement_chart_frame(frames, cash_balance_metrics)

    if activity_data is not None:
        st.markdown("###### Cash flow by activity")
        st.bar_chart(activity_data)

    if cash_balance_data is not None:
        st.markdown("###### Cash balance reconciliation")
        st.line_chart(cash_balance_data)


def _payload_to_ai_settings(payload: dict) -> Dict[str, Any]:
    ai_payload = payload.get("ai") or {}
    ml_methods = ai_payload.get("ml_methods") or ["linear_regression"]
    features = ai_payload.get("generative_features") or ["summary"]
    return {
        "enabled": bool(ai_payload.get("enabled", False)),
        "provider": ai_payload.get("provider", "OpenAI"),
        "model": ai_payload.get("model", "gpt-4"),
        "forecast_horizon": int(ai_payload.get("forecast_horizon", 3)),
        "ml_methods": [str(method) for method in ml_methods],
        "generative_features": [str(feature) for feature in features],
        "api_key": ai_payload.get("api_key", ""),
    }


def _ai_settings_to_payload(settings: Dict[str, Any], payload: dict) -> None:
    payload.setdefault("ai", {})
    payload["ai"].update(
        {
            "enabled": bool(settings.get("enabled", False)),
            "provider": settings.get("provider", "OpenAI"),
            "model": settings.get("model", "gpt-4"),
            "forecast_horizon": int(settings.get("forecast_horizon", 3)),
            "ml_methods": list(settings.get("ml_methods", ["linear_regression"])),
            "generative_features": list(
                settings.get("generative_features", ["summary"])
            ),
            "api_key": settings.get("api_key", ""),
        }
    )


def _render_ai_settings(payload: dict, container: Optional[DeltaGenerator] = None) -> None:
    target = container or st
    settings = st.session_state.setdefault(
        "ai_settings", _payload_to_ai_settings(payload)
    )
    st.session_state.setdefault("ai_api_key", settings.get("api_key", ""))

    if st.session_state.pop("ai_settings_saved", False):
        target.success("AI configuration updated. Rerunning the model with the new settings.")

    provider_options = list(AI_PROVIDER_OPTIONS)
    if settings.get("provider") and settings["provider"] not in provider_options:
        provider_options.append(settings["provider"])

    current_provider = settings.get("provider", provider_options[0])
    try:
        provider_index = provider_options.index(current_provider)
    except ValueError:
        provider_index = 0

    ml_defaults = [
        ML_METHOD_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("ml_methods", ["linear_regression"])
    ]
    feature_defaults = [
        GEN_AI_FEATURE_LABELS.get(code, code.replace("_", " ").title())
        for code in settings.get("generative_features", ["summary"])
    ]

    form = target.form("ai_settings_form")
    with form:
        enabled = form.checkbox(
            "Enable AI Enhancements",
            value=bool(settings.get("enabled", False)),
            help="Toggle machine-learning forecasts and generative commentary.",
        )
        provider = form.selectbox(
            "Provider",
            provider_options,
            index=provider_index,
            help="Select the API provider powering generative insights.",
        )
        model = form.text_input(
            "Model",
            value=settings.get("model", "gpt-4"),
            help="Name of the deployed model (for example `gpt-4o-mini`).",
        )
        horizon = form.number_input(
            "Forecast Horizon (years)",
            min_value=0,
            max_value=20,
            value=int(settings.get("forecast_horizon", 3)),
            step=1,
            help="Number of additional years used for machine-learning revenue forecasts.",
        )

        ml_selection = form.multiselect(
            "Machine Learning Methods",
            list(ML_METHOD_LABELS.values()),
            default=ml_defaults,
            help="Choose algorithms applied to projected net revenue.",
        )
        feature_selection = form.multiselect(
            "Generative Features",
            list(GEN_AI_FEATURE_LABELS.values()),
            default=feature_defaults,
            help="Pick the narrative focus areas generated by the AI summary.",
        )
        api_key = form.text_input(
            "API Key",
            value=st.session_state.get("ai_api_key", ""),
            type="password",
            help="Store your provider API key securely. Keys are retained only for the current session.",
        )

        submitted = form.form_submit_button("Save AI Configuration")

    if submitted:
        ml_codes = [
            ML_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower())
            for label in ml_selection
        ]
        feature_codes = [
            GEN_AI_LABEL_TO_CODE.get(label, label.replace(" ", "_").lower())
            for label in feature_selection
        ]

        settings.update(
            {
                "enabled": enabled,
                "provider": provider,
                "model": model.strip() or "gpt-4",
                "forecast_horizon": int(horizon),
                "ml_methods": ml_codes or ["linear_regression"],
                "generative_features": feature_codes or ["summary"],
                "api_key": api_key.strip(),
            }
        )
        st.session_state["ai_settings"] = settings
        st.session_state["ai_api_key"] = settings.get("api_key", "")
        _ai_settings_to_payload(settings, payload)
        st.session_state["ai_settings_saved"] = True
        _maybe_rerun()


def _analytics_override_store() -> Dict[str, Any]:
    """Return (and create) the session-backed override cache."""

    return st.session_state.setdefault("advanced_analytics_overrides", {})


def _get_analytics_override(
    scenario: str, block: str, analysis: str, table: str
) -> Optional[pd.DataFrame]:
    store = _analytics_override_store()
    return (
        store.get(scenario, {})
        .get(block, {})
        .get(analysis, {})
        .get(table)
    )


def _set_analytics_override(
    scenario: str, block: str, analysis: str, table: str, value: pd.DataFrame
) -> None:
    store = _analytics_override_store()
    scenario_store = store.setdefault(scenario, {})
    block_store = scenario_store.setdefault(block, {})
    analysis_store = block_store.setdefault(analysis, {})
    analysis_store[table] = value.copy(deep=True)
    st.session_state["advanced_analytics_overrides"] = store


def _clear_analytics_override(
    scenario: str, block: str, analysis: str, table: str
) -> None:
    store = _analytics_override_store()
    scenario_store = store.get(scenario)
    if not scenario_store:
        return
    block_store = scenario_store.get(block)
    if not block_store:
        return
    analysis_store = block_store.get(analysis)
    if not analysis_store:
        return
    analysis_store.pop(table, None)
    if not analysis_store:
        block_store.pop(analysis, None)
    if not block_store:
        scenario_store.pop(block, None)
    if not scenario_store:
        store.pop(scenario, None)
    st.session_state["advanced_analytics_overrides"] = store


def _analytics_edit_flag_key(
    scenario: str, block: str, analysis: str, table: str
) -> str:
    return f"analytics_edit::{scenario}::{block}::{analysis}::{table}"


def _analytics_editor_key(
    scenario: str, block: str, analysis: str, table: str
) -> str:
    return f"analytics_editor::{scenario}::{block}::{analysis}::{table}"


def _prepare_editor_table(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Return an editable copy and metadata to restore the original index."""

    working = df.copy(deep=True)
    if isinstance(working.index, pd.MultiIndex):
        orig_names = list(working.index.names)
        fallback_names = [
            name if name is not None else f"Index Level {idx + 1}"
            for idx, name in enumerate(orig_names)
        ]
        working.index.set_names(fallback_names, inplace=True)
        editor_df = working.reset_index()
        return editor_df, {
            "type": "multi",
            "names": fallback_names,
            "orig_names": orig_names,
        }

    if isinstance(working.index, pd.RangeIndex):
        editor_df = working.reset_index(drop=True)
        return editor_df, {
            "type": "range",
            "start": working.index.start,
            "step": working.index.step,
        }

    index_name = working.index.name or "Index"
    working.index.name = index_name
    editor_df = working.reset_index()
    return editor_df, {
        "type": "single",
        "name": index_name,
        "orig_name": df.index.name,
    }


def _restore_editor_table(edited: pd.DataFrame, meta: Dict[str, Any]) -> pd.DataFrame:
    """Rebuild a DataFrame using the stored index metadata."""

    restored = edited.copy(deep=True)
    meta_type = meta.get("type")

    if meta_type == "multi":
        names = [name for name in meta.get("names", []) if name in restored.columns]
        if names:
            restored = restored.set_index(names)
            orig_names = meta.get("orig_names", names)
            if len(orig_names) == restored.index.nlevels:
                restored.index.names = orig_names
        else:
            restored.index = pd.RangeIndex(len(restored))
        return restored

    if meta_type == "single":
        column = meta.get("name")
        if column and column in restored.columns:
            restored = restored.set_index(column)
            restored.index.name = meta.get("orig_name")
        else:
            restored.index = pd.RangeIndex(len(restored))
            restored.index.name = meta.get("orig_name")
        return restored

    # Default to a simple RangeIndex
    restored.index = pd.RangeIndex(len(restored))
    return restored


def _format_row_label(df: pd.DataFrame, idx: int) -> str:
    """Return a compact label describing a row for the row selector."""

    if df.empty:
        return "Row"

    row = df.iloc[idx]
    parts: list[str] = []
    for column in df.columns[:3]:
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value)
        if text.strip() == "":
            continue
        parts.append(f"{column}: {text}")
        if len(parts) == 2:
            break

    if not parts:
        return f"Row {idx + 1}"
    return " | ".join(parts)


def _coerce_row_value(raw: Any, dtype: pd.Series.dtype) -> Any:
    """Coerce a raw editor input back to the column's dtype."""

    if is_bool_dtype(dtype):
        return bool(raw)

    if isinstance(raw, str):
        text = raw.strip()
        if text == "":
            if is_numeric_dtype(dtype) or is_datetime64_any_dtype(dtype):
                return np.nan
            return ""

        if is_datetime64_any_dtype(dtype):
            try:
                return pd.to_datetime(text)
            except (TypeError, ValueError):
                return text

        if is_numeric_dtype(dtype):
            coerced = pd.to_numeric([text], errors="coerce")[0]
            if pd.isna(coerced):
                return np.nan
            if is_integer_dtype(dtype):
                return int(round(coerced))
            return float(coerced)

        return text

    return raw


def _render_row_input(
    column: str, value: Any, dtype: pd.Series.dtype, widget_key: str
) -> Any:
    """Render an appropriate widget for a single row cell."""

    if is_bool_dtype(dtype):
        default = bool(value) if not pd.isna(value) else False
        return st.checkbox(column, value=default, key=widget_key)

    display_value = "" if pd.isna(value) else str(value)
    return st.text_input(column, value=display_value, key=widget_key)


def _schedule_edit_flag_key(identifier: str) -> str:
    return f"schedule_edit::{identifier}"


def _schedule_working_key(identifier: str) -> str:
    return f"schedule_editor::{identifier}::working"


def _schedule_meta_key(identifier: str) -> str:
    return f"schedule_editor::{identifier}::meta"


def _schedule_row_selector_key(identifier: str) -> str:
    return f"schedule_editor::{identifier}::row"


def _clear_schedule_editor_state(identifier: str) -> None:
    """Remove any cached working copies for a schedule editor."""

    st.session_state.pop(_schedule_working_key(identifier), None)
    st.session_state.pop(_schedule_meta_key(identifier), None)
    st.session_state.pop(_schedule_row_selector_key(identifier), None)


def _default_value_for_dtype(dtype: pd.Series.dtype) -> Any:
    if is_bool_dtype(dtype):
        return False
    if is_numeric_dtype(dtype):
        return np.nan
    return ""


def _blank_row_like(df: pd.DataFrame) -> Dict[str, Any]:
    """Return a dictionary representing a blank row for the dataframe."""

    blanks: Dict[str, Any] = {}
    for column in df.columns:
        blanks[column] = _default_value_for_dtype(df[column].dtype)
    return blanks


def _render_schedule_row_editor(
    identifier: str, table: pd.DataFrame, save_callback: Callable[[pd.DataFrame], None]
) -> None:
    """Render a row-focused editor for schedule dataframes."""

    st.dataframe(table, use_container_width=True)

    edit_flag_key = _schedule_edit_flag_key(identifier)
    editing = st.session_state.get(edit_flag_key, False)
    toggle_label = "Edit rows" if not editing else "Close row editor"
    if st.button(toggle_label, key=f"{identifier}::toggle"):
        if editing:
            _clear_schedule_editor_state(identifier)
        st.session_state[edit_flag_key] = not editing
        _maybe_rerun()
        return

    editing = st.session_state.get(edit_flag_key, False)
    if not editing:
        return

    editor_df, meta = _prepare_editor_table(table)
    working_key = _schedule_working_key(identifier)
    meta_key = _schedule_meta_key(identifier)

    if working_key not in st.session_state:
        st.session_state[working_key] = editor_df.copy(deep=True)
        st.session_state[meta_key] = meta

    working_df = st.session_state.get(working_key, editor_df.copy(deep=True))
    stored_meta = st.session_state.get(meta_key, meta)

    st.caption(
        "Update one row at a time. Use the controls below to add new rows or remove the selected row."
    )
    st.dataframe(working_df, use_container_width=True)

    template_df = working_df if not working_df.empty else editor_df

    controls = st.columns(3)
    if controls[0].button("Add row", key=f"{identifier}::add_row"):
        blank_row = _blank_row_like(template_df)
        new_row = pd.DataFrame([blank_row], columns=template_df.columns)
        updated_df = pd.concat([working_df, new_row], ignore_index=True)
        st.session_state[working_key] = updated_df
        _maybe_rerun()
        return

    selector_key = _schedule_row_selector_key(identifier)
    if working_df.empty:
        selected_row = None
        controls[1].write("No rows to delete.")
    else:
        selected_row = st.selectbox(
            "Select a row to edit",
            list(range(len(working_df))),
            format_func=lambda idx: _format_row_label(working_df, idx),
            key=selector_key,
        )
        if controls[1].button("Delete row", key=f"{identifier}::delete_row"):
            if selected_row is not None and 0 <= selected_row < len(working_df):
                updated_df = working_df.drop(index=selected_row).reset_index(drop=True)
                st.session_state[working_key] = updated_df
                _maybe_rerun()
                return

    if controls[2].button("Reset changes", key=f"{identifier}::reset"):
        st.session_state[working_key] = editor_df.copy(deep=True)
        st.session_state[meta_key] = meta
        _maybe_rerun()
        return

    working_df = st.session_state.get(working_key, editor_df.copy(deep=True))

    if working_df.empty:
        st.info("There are no rows to edit. Add a new row to begin.")
    else:
        selected_row = st.session_state.get(selector_key, 0)
        if selected_row >= len(working_df):
            selected_row = 0
            st.session_state[selector_key] = selected_row

        dtype_map = working_df.dtypes.to_dict()
        row_series = working_df.iloc[selected_row]

        with st.form(f"{identifier}::row_form"):
            updated_values: Dict[str, Any] = {}
            for column in working_df.columns:
                widget_key = f"{identifier}::{selected_row}::{column}"
                updated_values[column] = _render_row_input(
                    column, row_series[column], dtype_map[column], widget_key
                )

            submitted = st.form_submit_button("Apply row changes")
            if submitted:
                for column, raw_value in updated_values.items():
                    coerced = _coerce_row_value(raw_value, dtype_map[column])
                    working_df.iat[
                        selected_row, working_df.columns.get_loc(column)
                    ] = coerced
                st.session_state[working_key] = working_df
                _maybe_rerun()
                return

    action_cols = st.columns(3)
    if action_cols[0].button("Save changes", key=f"{identifier}::save"):
        try:
            current_df = st.session_state.get(working_key, editor_df.copy(deep=True))
            current_meta = st.session_state.get(meta_key, stored_meta)
            restored = _restore_editor_table(current_df, current_meta)
            save_callback(restored)
            _clear_schedule_editor_state(identifier)
            st.session_state[edit_flag_key] = False
            _maybe_rerun()
            return
        except Exception as exc:  # pragma: no cover - user feedback path
            action_cols[0].error(f"Unable to save changes: {exc}")

    if action_cols[1].button("Discard edits", key=f"{identifier}::cancel"):
        _clear_schedule_editor_state(identifier)
        st.session_state[edit_flag_key] = False
        _maybe_rerun()
        return

    if action_cols[2].button("Close editor", key=f"{identifier}::close"):
        _clear_schedule_editor_state(identifier)
        st.session_state[edit_flag_key] = False
        _maybe_rerun()


DETAIL_SCHEDULE_COLUMNS = {
    "COGS Schedule": ["COGS"],
    "Variable Expenses Schedule": ["Variable Expenses"],
    "Direct Wages Schedule": ["Direct Wages"],
    "Admin Wages Schedule": ["Admin Wages"],
    "Capex Schedule": ["Capex"],
}


SCENARIO_PRESETS: Dict[str, Dict[str, Any]] = {
    "Base Case Scenario": {
        "adjustments": {
            "Milk price change (%)": 0.0,
            "Feed cost change (%)": 0.0,
        },
        "description": "Baseline view using the model inputs without additional shocks.",
    },
    "Best Case Scenario": {
        "adjustments": {
            "Milk price change (%)": 12.0,
            "Feed cost change (%)": -8.0,
        },
        "description": "Upside case with stronger milk pricing and more efficient feed spend.",
    },
    "Worst Case Scenario": {
        "adjustments": {
            "Milk price change (%)": -12.0,
            "Feed cost change (%)": 10.0,
        },
        "description": "Downside case featuring pricing pressure and higher feed costs.",
    },
}


def _default_scenario_preset_table(name: str) -> pd.DataFrame:
    preset = SCENARIO_PRESETS.get(name, {})
    adjustments = preset.get("adjustments", {})
    if not adjustments:
        return pd.DataFrame(columns=["Driver", "Change %"])
    return pd.DataFrame(
        {
            "Driver": list(adjustments.keys()),
            "Change %": [float(value) for value in adjustments.values()],
        }
    )


def _scenario_preset_removed_store() -> Dict[str, List[str]]:
    store = st.session_state.setdefault("scenario_preset_removed_drivers", {})
    normalised: Dict[str, List[str]] = {}
    updated = False

    for name in SCENARIO_PRESETS.keys():
        entries = store.get(name, [])
        cleaned = sorted(
            {
                str(entry).casefold()
                for entry in entries
                if str(entry).strip()
            }
        )
        normalised[name] = cleaned
        if entries != cleaned:
            updated = True

    if set(store.keys()) != set(normalised.keys()):
        updated = True

    if updated:
        st.session_state["scenario_preset_removed_drivers"] = normalised
        store = normalised
    else:
        st.session_state["scenario_preset_removed_drivers"] = store

    return store


def _unmark_removed_scenario_drivers(name: str, drivers: Iterable[str]) -> None:
    store = _scenario_preset_removed_store()
    current = set(store.get(name, []))
    lower_drivers = {str(driver).casefold() for driver in drivers if str(driver).strip()}
    new_removed = sorted(current - lower_drivers)
    if new_removed != sorted(current):
        store[name] = new_removed
        st.session_state["scenario_preset_removed_drivers"] = store


def _mark_removed_scenario_driver(name: str, driver: str) -> None:
    store = _scenario_preset_removed_store()
    cleaned = str(driver).strip()
    if not cleaned:
        return
    lowered = cleaned.casefold()
    current = set(store.get(name, []))
    if lowered not in current:
        current.add(lowered)
        store[name] = sorted(current)
        st.session_state["scenario_preset_removed_drivers"] = store


def _scenario_preset_tables_store() -> Dict[str, pd.DataFrame]:
    return st.session_state.setdefault("scenario_preset_tables", {})


def _ensure_scenario_preset_table(
    name: str, table: Optional[pd.DataFrame]
) -> pd.DataFrame:
    default_table = _default_scenario_preset_table(name)
    if table is None or table.empty:
        work = default_table.copy()
    else:
        work = table.copy()

    if "Driver" not in work.columns:
        work["Driver"] = ""
    work["Driver"] = work.get("Driver", "").astype(str).str.strip()
    work.loc[work["Driver"] == "", "Driver"] = "Driver"

    if "Change %" not in work.columns:
        work["Change %"] = np.nan
    work["Change %"] = pd.to_numeric(work.get("Change %"), errors="coerce")

    removed_store = _scenario_preset_removed_store()
    removed = set(removed_store.get(name, []))

    required_rows = []
    for _, row in default_table.iterrows():
        driver = str(row.get("Driver", "")).strip()
        if not driver:
            continue
        if driver.casefold() in removed:
            continue
        mask = work["Driver"].str.casefold() == driver.casefold()
        if not mask.any():
            required_rows.append({
                "Driver": driver,
                "Change %": float(row.get("Change %", 0.0)),
            })
        else:
            existing_index = work.index[mask][0]
            if pd.isna(work.at[existing_index, "Change %"]):
                work.at[existing_index, "Change %"] = float(row.get("Change %", 0.0))

    if required_rows:
        work = pd.concat([work, pd.DataFrame(required_rows)], ignore_index=True)

    driver_order = [
        str(val).casefold()
        for val in default_table.get("Driver", pd.Series(dtype=str)).tolist()
        if str(val).strip() and str(val).casefold() not in removed
    ]
    order_map = {driver: idx for idx, driver in enumerate(driver_order)}
    work["__order__"] = work["Driver"].str.casefold().map(order_map)
    work = work.sort_values(["__order__", "Driver"], na_position="last").drop(
        columns=["__order__"], errors="ignore"
    )

    ordered_cols = ["Driver", "Change %"]
    remainder = [col for col in work.columns if col not in ordered_cols]
    return work[ordered_cols + remainder].reset_index(drop=True)


def _set_scenario_preset_table(name: str, table: pd.DataFrame) -> None:
    store = _scenario_preset_tables_store()
    store[name] = table.copy(deep=True).reset_index(drop=True)
    drivers = store[name].get("Driver", pd.Series(dtype=str)).tolist()
    _unmark_removed_scenario_drivers(name, drivers)
    st.session_state["scenario_preset_tables"] = store


def _get_scenario_preset_table(name: str) -> pd.DataFrame:
    store = _scenario_preset_tables_store()
    ensured = _ensure_scenario_preset_table(name, store.get(name))
    store[name] = ensured
    st.session_state["scenario_preset_tables"] = store
    return ensured


def _scenario_preset_descriptions_store() -> Dict[str, str]:
    store = st.session_state.setdefault("scenario_preset_descriptions", {})
    for name, preset in SCENARIO_PRESETS.items():
        store.setdefault(name, preset.get("description", ""))
    st.session_state["scenario_preset_descriptions"] = store
    return store


def _set_scenario_preset_description(name: str, description: str) -> Optional[str]:
    store = _scenario_preset_descriptions_store()
    normalized = str(description or "").strip()
    previous = store.get(name, "")
    if normalized == previous:
        return None
    store[name] = normalized
    st.session_state["scenario_preset_descriptions"] = store
    _reset_cached_results()
    return normalized


def _scenario_preset_table_to_adjustments(
    name: str, table: pd.DataFrame
) -> Dict[str, float]:
    ensured = _ensure_scenario_preset_table(name, table)
    adjustments: Dict[str, float] = {}
    for _, row in ensured.iterrows():
        driver = str(row.get("Driver", "")).strip()
        if not driver:
            continue
        value = pd.to_numeric(pd.Series([row.get("Change %")]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        adjustments[driver] = float(value)

    removed_store = _scenario_preset_removed_store()
    removed = set(removed_store.get(name, []))
    for driver in list(adjustments.keys()):
        if driver.casefold() in removed:
            adjustments.pop(driver, None)

    return adjustments


def _add_scenario_preset_driver(name: str, driver: str, change: float) -> None:
    cleaned = str(driver).strip()
    if not cleaned:
        return

    table = _get_scenario_preset_table(name)
    mask = table["Driver"].str.casefold() == cleaned.casefold()
    if mask.any():
        table.loc[mask, "Change %"] = float(change)
    else:
        new_row = pd.DataFrame(
            [{"Driver": cleaned, "Change %": float(change)}],
            columns=table.columns,
        )
        table = pd.concat([table, new_row], ignore_index=True)

    _set_scenario_preset_table(name, table)
    _unmark_removed_scenario_drivers(name, [cleaned])
    _reset_cached_results()


def _remove_scenario_preset_driver(name: str, driver: str) -> None:
    cleaned = str(driver).strip()
    if not cleaned:
        return

    table = _get_scenario_preset_table(name)
    mask = table["Driver"].str.casefold() == cleaned.casefold()
    if not mask.any():
        return

    updated = table.loc[~mask].reset_index(drop=True)
    _set_scenario_preset_table(name, updated)
    _mark_removed_scenario_driver(name, cleaned)
    _reset_cached_results()


def _current_scenario_presets() -> Dict[str, Dict[str, Any]]:
    tables = _scenario_preset_tables_store()
    descriptions = _scenario_preset_descriptions_store()
    presets: Dict[str, Dict[str, Any]] = {}

    for name in SCENARIO_PRESETS.keys():
        table = _ensure_scenario_preset_table(name, tables.get(name))
        tables[name] = table
        adjustments = _scenario_preset_table_to_adjustments(name, table)
        description = descriptions.get(name, SCENARIO_PRESETS[name].get("description", ""))
        presets[name] = {
            "adjustments": adjustments,
            "description": description,
        }

    st.session_state["scenario_preset_tables"] = tables
    st.session_state["scenario_preset_descriptions"] = descriptions
    return presets


def _render_scenario_preset_editors() -> None:
    st.markdown("#### Scenario Presets")
    preset_names = list(SCENARIO_PRESETS.keys())
    tabs = st.tabs(preset_names)

    for tab, name in zip(tabs, preset_names):
        with tab:
            table = _get_scenario_preset_table(name)

            add_cols = st.columns([2, 1, 1])
            driver_key = f"scenario_preset::{_scenario_key_suffix(name)}::new_driver"
            change_key = f"scenario_preset::{_scenario_key_suffix(name)}::new_change"
            if not _safe_session_state_contains(driver_key):
                _safe_session_state_setdefault(driver_key, "")
            if not _safe_session_state_contains(change_key):
                _safe_session_state_setdefault(change_key, 0.0)

            new_driver = add_cols[0].text_input(
                "Driver name",
                key=driver_key,
            )
            new_change = add_cols[1].number_input(
                "Change (%)",
                key=change_key,
                step=0.25,
                format="%.2f",
            )
            if add_cols[2].button("Add variable", key=f"{driver_key}::add"):
                _add_scenario_preset_driver(name, new_driver, float(new_change))
                _safe_session_state_set(driver_key, "")
                _safe_session_state_set(change_key, 0.0)
                _maybe_rerun()
                return

            remove_cols = st.columns([2, 1])
            driver_options = table.get("Driver", pd.Series(dtype=str)).fillna("")
            option_labels = [value for value in driver_options.astype(str).tolist() if value.strip()]
            remove_key = f"scenario_preset::{_scenario_key_suffix(name)}::remove_choice"
            remove_selection = remove_cols[0].selectbox(
                "Select variable to remove",
                options=["-- Select --"] + option_labels,
                key=remove_key,
            )
            if remove_cols[1].button("Remove variable", key=f"{remove_key}::remove"):
                if remove_selection and remove_selection != "-- Select --":
                    _remove_scenario_preset_driver(name, remove_selection)
                    _safe_session_state_set(remove_key, "-- Select --")
                    _maybe_rerun()
                    return

            def _save(updated: pd.DataFrame, preset_name: str = name) -> None:
                ensured = _ensure_scenario_preset_table(preset_name, updated)
                _set_scenario_preset_table(preset_name, ensured)
                _reset_cached_results()

            _render_schedule_row_editor(
                f"scenario_preset::{_scenario_key_suffix(name)}",
                table,
                _save,
            )

            description_store = _scenario_preset_descriptions_store()
            default_description = SCENARIO_PRESETS[name].get("description", "")
            stored_value = description_store.get(name, default_description)
            desc_key = f"scenario_desc::{_scenario_key_suffix(name)}"
            if not _safe_session_state_contains(desc_key):
                _safe_session_state_setdefault(desc_key, stored_value)

            st.text_area("Description", key=desc_key)
            new_value = _safe_session_state_get(desc_key, "")
            updated = _set_scenario_preset_description(name, new_value)
            if updated is not None:
                _safe_session_state_set(desc_key, updated)

CAP_TABLE_COLUMNS = ["Year", "Shareholder", "Ownership %", "Investment"]


CAPEX_TABLE_COLUMNS = [
    "Year",
    "Category",
    "Spend",
    "Depreciation Rate %",
    "Depreciation",
]


ASSET_SCHEDULE_COLUMNS = [
    "Asset",
    "Year",
    "Opening NBV",
    "Additions",
    "Depreciation Rate %",
    "Depreciation",
    "Closing NBV",
]


def _normalize_period(series: pd.Series) -> pd.Series:
    periods = pd.to_datetime(series, errors="coerce")
    formatted = periods.dt.strftime("%Y-%m-%d")
    return formatted.where(~periods.isna(), series.astype(str))


def _format_scenario_label(milk_pct: int, feed_pct: int) -> str:
    if milk_pct == 0 and feed_pct == 0:
        return "Base Scenario"
    return f"Milk {milk_pct:+d}%, Feed {feed_pct:+d}%"


def _build_scenario_suite(
    custom_label: Optional[str] = None,
    custom_adjustments: Optional[Dict[str, float]] = None,
) -> Dict[str, Dict[str, Any]]:
    suite: Dict[str, Dict[str, Any]] = {}
    presets = _current_scenario_presets()
    for name, preset in presets.items():
        suite[name] = {
            "adjustments": deepcopy(preset["adjustments"]),
            "description": preset.get("description", ""),
        }

    if custom_label and custom_adjustments:
        suite[custom_label] = {
            "adjustments": {
                key: float(value)
                for key, value in custom_adjustments.items()
            },
            "description": "Custom scenario defined via scenario controls.",
        }

    return suite


def _execute_scenario_suite(
    schedule_df: pd.DataFrame,
    valuation_inputs: Dict[str, float],
    supplementary_tables: Dict[str, pd.DataFrame],
    scenario_suite: Dict[str, Dict[str, Any]],
) -> Tuple[GoatModel, pd.DataFrame, Dict[str, Dict[str, Any]]]:
    schedule = InputSchedule(
        data=schedule_df,
        valuation_inputs=valuation_inputs,
        supplementary_tables=supplementary_tables,
    )
    model = schedule.to_model()
    base = model.to_tidy()

    base_supplementary = {
        name: table.copy()
        for name, table in (supplementary_tables or {}).items()
        if isinstance(table, pd.DataFrame)
    }

    results: Dict[str, Dict[str, Any]] = {}
    author_name = _current_model_author()
    for name, config in scenario_suite.items():
        adjustments = config.get("adjustments", {})
        milk_pct = float(adjustments.get("Milk price change (%)", 0.0))
        feed_pct = float(adjustments.get("Feed cost change (%)", 0.0))

        scenario_df = model.scenario(
            milk_price_pct=milk_pct / 100.0,
            feed_cost_pct=feed_pct / 100.0,
        )

        scenario_supplementary = {
            key: value.copy() for key, value in base_supplementary.items()
        }

        scenario_inputs: Dict[str, Any] = {
            "Milk price change (%)": milk_pct,
            "Feed cost change (%)": feed_pct,
        }
        if author_name:
            scenario_inputs["Model author"] = author_name

        results[name] = {
            "model": model,
            "base": base,
            "scenario": scenario_df,
            "kpis": model.kpis(scenario_df, annual=True),
            "break_even": model.break_even(scenario_df, annual=True),
            "supplementary": scenario_supplementary,
            "selected_scenario": name,
            "scenario_inputs": scenario_inputs,
            "model_author": author_name,
            "preset_description": config.get("description", ""),
        }

    return model, base, results


def _ensure_active_scenario_selection() -> None:
    scenario_results = st.session_state.get("all_scenario_results") or {}
    if not scenario_results:
        return

    selected = st.session_state.get("selected_scenario_name")
    if selected not in scenario_results:
        selected = next(iter(scenario_results))
        st.session_state.selected_scenario_name = selected

    current = st.session_state.get("results")
    if not isinstance(current, dict) or current.get("selected_scenario") != selected:
        st.session_state.results = scenario_results[selected]


def _render_scenario_selector(prefix: str = "main") -> None:
    scenario_results = st.session_state.get("all_scenario_results") or {}
    if not scenario_results:
        st.info("Run the scenario suite to enable comparisons.")
        return

    options = list(scenario_results.keys())
    selected_name = st.session_state.get("selected_scenario_name")
    try:
        default_index = options.index(selected_name)
    except ValueError:
        default_index = 0
        st.session_state.selected_scenario_name = options[0]
        selected_name = options[0]

    selected = st.selectbox(
        "Select scenario",
        options,
        index=default_index,
        key=f"scenario_selector::{prefix}",
    )

    if selected != selected_name:
        st.session_state.selected_scenario_name = selected
        st.session_state.results = scenario_results[selected]
        selected_name = selected

    description = scenario_results[selected_name].get("preset_description")
    adjustments = scenario_results[selected_name].get("scenario_inputs", {})
    author_name = scenario_results[selected_name].get("model_author")

    details: list[str] = []
    if description:
        details.append(description)

    if adjustments:
        numeric_adjustments = [
            f"{driver}: {value:+.1f}%"
            for driver, value in adjustments.items()
            if isinstance(value, (int, float)) and not pd.isna(value)
        ]
        if numeric_adjustments:
            details.append(f"Adjustments – {', '.join(numeric_adjustments)}")

    if isinstance(author_name, str) and author_name.strip():
        details.append(f"Prepared by – {author_name.strip()}")

    if details:
        st.caption(" \n".join(details))


def _scenario_key_suffix(label: str) -> str:
    normalized = re.sub(r"[^0-9a-zA-Z]+", "_", label).strip("_").lower()
    return normalized or "scenario"


def _sanitize_sheet_name(name: str, existing: set[str]) -> str:
    invalid = set('[]:*?/\\')
    cleaned = "".join("_" if ch in invalid else ch for ch in name).strip()
    cleaned = cleaned or "Sheet"
    if len(cleaned) > 31:
        cleaned = cleaned[:31]
    base = cleaned
    suffix = 1
    while cleaned in existing:
        candidate = f"{base[:31 - len(str(suffix)) - 1]}_{suffix}" if len(base) >= 30 else f"{base}_{suffix}"
        cleaned = candidate[:31]
        suffix += 1
    existing.add(cleaned)
    return cleaned


def _prepare_dataframe_for_excel(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    frame = df.copy()
    if isinstance(frame, pd.Series):
        frame = frame.to_frame()

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = [
            " - ".join(str(part) for part in tup if str(part))
            for tup in frame.columns.to_flat_index()
        ]

    index = frame.index
    if isinstance(index, pd.MultiIndex):
        frame = frame.reset_index()
    elif isinstance(index, pd.DatetimeIndex):
        frame = frame.reset_index()
        first_col = frame.columns[0]
        frame[first_col] = _normalize_period(frame[first_col])
        if first_col == "index" or not first_col:
            frame = frame.rename(columns={first_col: "Period"})
    elif not isinstance(index, pd.RangeIndex) or index.name:
        frame = frame.reset_index()
        first_col = frame.columns[0]
        if first_col == "index" and index.name:
            frame = frame.rename(columns={first_col: index.name})

    return frame


def _resolve_excel_writer_engine() -> str:
    if find_spec("xlsxwriter") is not None:
        return "xlsxwriter"
    if find_spec("openpyxl") is not None:
        return "openpyxl"
    raise RuntimeError(
        "Excel export requires the XlsxWriter or openpyxl package. "
        "Install one of them to enable downloads."
    )


def _generate_excel_bytes(
    model: GoatModel,
    results: Dict[str, Any],
    scenario_name: str,
    author_name: Optional[str] = None,
) -> bytes:
    buffer = BytesIO()

    scenario_df = results.get("scenario")
    base_df = results.get("base")
    kpis = results.get("kpis")
    break_even = results.get("break_even")
    supplementary = results.get("supplementary", {})
    scenario_inputs = results.get("scenario_inputs", {})

    used_sheets: set[str] = set()

    def write_sheet(name: str, df: Optional[pd.DataFrame]) -> None:
        if df is None:
            return
        frame = _prepare_dataframe_for_excel(df)
        if frame.empty and frame.columns.empty:
            return
        frame.to_excel(
            writer,
            sheet_name=_sanitize_sheet_name(name, used_sheets),
            index=False,
        )

    engine = _resolve_excel_writer_engine()
    author = (author_name or "").strip() or DEFAULT_MODEL_AUTHOR

    with pd.ExcelWriter(buffer, engine=engine) as writer:
        workbook = getattr(writer, "book", None)
        if workbook is not None and author:
            if engine == "xlsxwriter" and hasattr(workbook, "set_properties"):
                workbook.set_properties({"author": author})
            elif engine == "openpyxl":
                try:
                    workbook.properties.creator = author
                except Exception:  # pragma: no cover - best effort
                    pass

        if base_df is not None:
            write_sheet("Input Schedule", base_df)

        if scenario_df is not None:
            write_sheet(f"{scenario_name} Timeline", scenario_df)
            try:
                annual = scenario_df.copy()
                annual.index = pd.to_datetime(annual.index)
                annual = annual.groupby(annual.index.year).sum(min_count=1)
                annual.index.name = "Year"
                write_sheet(f"{scenario_name} Annual", annual)
            except Exception:
                pass

            try:
                sop = model.statement_of_financial_performance(scenario_df, annual=True)
                write_sheet("Statement of Financial Performance", sop)
            except ValueError:
                pass

            try:
                sofp = model.statement_of_financial_position(scenario_df, annual=True)
                write_sheet("Statement of Financial Position", sofp)
            except ValueError:
                pass

            try:
                socf = model.statement_of_cash_flow(scenario_df, annual=True)
                write_sheet("Statement of Cash Flows", socf)
            except ValueError:
                pass

        if kpis is not None:
            write_sheet("KPIs (Annual)", kpis.mul(100))

        if break_even is not None:
            write_sheet("Break-even Analysis", break_even)

        if scenario_inputs:
            inputs_df = pd.DataFrame(
                {
                    "Input": list(scenario_inputs.keys()),
                    "Value": list(scenario_inputs.values()),
                }
            )
            write_sheet("Scenario Inputs", inputs_df)

        for name, table in supplementary.items():
            if table is None:
                continue
            write_sheet(f"Supplementary - {name}", table)

        metadata_df = pd.DataFrame(
            {
                "Scenario": [scenario_name],
            }
        )
        write_sheet("Scenario Details", metadata_df)

    buffer.seek(0)
    return buffer.getvalue()


DEFAULT_VARIABLE_ITEMS = [
    {"Item": "Feed & Supplements", "Share %": 5.0},
    {"Item": "Veterinary & Healthcare", "Share %": 4.0},
    {"Item": "Distribution & Logistics", "Share %": 3.0},
]


DEFAULT_DIRECT_WAGE_ITEMS = [
    {"Role": "Milking Crew", "Share %": 60.0},
    {"Role": "Herd Management", "Share %": 40.0},
]


DEFAULT_ADMIN_WAGE_ITEMS = [
    {"Function": "Administration", "Share %": 40.0},
    {"Function": "Finance & Compliance", "Share %": 35.0},
    {"Function": "Sales & Support", "Share %": 25.0},
]


DEFAULT_PRICING_ROWS = [
    {
        "Year": 2024,
        "Product": "Milk",
        "Unit": "Litre",
        "Base Price": 1.85,
        "Price Growth %": 3.0,
    },
    {
        "Year": 2025,
        "Product": "Cheese",
        "Unit": "Kg",
        "Base Price": 12.50,
        "Price Growth %": 2.5,
    },
    {
        "Year": 2024,
        "Product": "Pelt",
        "Unit": "Kg",
        "Base Price": 8.00,
        "Price Growth %": 2.0,
    },
    {
        "Year": 2024,
        "Product": "Meat",
        "Unit": "Kg",
        "Base Price": 10.50,
        "Price Growth %": 2.8,
    },
]


DEFAULT_OPERATING_COST_ROWS = [
    {"Year": 2024, "Category": "Feed", "Monthly Cost": 8500.0, "Inflation %": 4.0},
    {"Year": 2025, "Category": "Feed", "Monthly Cost": 8840.0, "Inflation %": 4.0},
    {
        "Year": 2024,
        "Category": "Healthcare",
        "Monthly Cost": 1800.0,
        "Inflation %": 3.5,
    },
    {
        "Year": 2025,
        "Category": "Healthcare",
        "Monthly Cost": 1863.0,
        "Inflation %": 3.5,
    },
    {
        "Year": 2024,
        "Category": "Utilities",
        "Monthly Cost": 1200.0,
        "Inflation %": 2.0,
    },
    {
        "Year": 2025,
        "Category": "Utilities",
        "Monthly Cost": 1224.0,
        "Inflation %": 2.0,
    },
]


DEFAULT_INPUT_CONFIG_KEY = "default_input_templates"


def _default_input_template_config() -> Dict[str, list[dict[str, object]]]:
    return {
        "variable_items": deepcopy(DEFAULT_VARIABLE_ITEMS),
        "direct_wage_items": deepcopy(DEFAULT_DIRECT_WAGE_ITEMS),
        "admin_wage_items": deepcopy(DEFAULT_ADMIN_WAGE_ITEMS),
        "pricing_rows": deepcopy(DEFAULT_PRICING_ROWS),
        "operating_rows": deepcopy(DEFAULT_OPERATING_COST_ROWS),
    }


def _ensure_default_templates() -> Dict[str, list[dict[str, object]]]:
    if DEFAULT_INPUT_CONFIG_KEY not in st.session_state:
        st.session_state[DEFAULT_INPUT_CONFIG_KEY] = _default_input_template_config()
    return st.session_state[DEFAULT_INPUT_CONFIG_KEY]


def _template_copy(template: list[dict[str, object]]) -> list[dict[str, object]]:
    return [dict(row) for row in template]


def _get_template(name: str, fallback: list[dict[str, object]]) -> list[dict[str, object]]:
    templates = _ensure_default_templates()
    rows = templates.get(name)
    if rows:
        return deepcopy(rows)
    return deepcopy(fallback)


def _set_template(name: str, rows: list[dict[str, object]]) -> None:
    templates = _ensure_default_templates()
    templates[name] = deepcopy(rows)
    st.session_state[DEFAULT_INPUT_CONFIG_KEY] = templates


def _template_to_dataframe(
    rows: list[dict[str, object]], columns: list[str]
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=columns)

    df = pd.DataFrame(rows)
    for column in columns:
        if column not in df.columns:
            df[column] = np.nan
    return df[columns]


def _dataframe_to_template(df: pd.DataFrame, columns: list[str]) -> list[dict[str, object]]:
    if df is None or df.empty:
        return []

    work = df.copy()
    for column in columns:
        if column not in work.columns:
            work[column] = np.nan

    records: list[dict[str, object]] = []
    for _, row in work[columns].iterrows():
        record: dict[str, object] = {}
        empty = True
        for column in columns:
            value = row[column]
            if isinstance(value, str):
                cleaned = value.strip()
                record[column] = cleaned or None
                if cleaned:
                    empty = False
            else:
                if pd.isna(value):
                    record[column] = None
                else:
                    record[column] = float(value)
                    empty = False
        if not empty:
            records.append(record)
    return records


def _default_pricing_table() -> pd.DataFrame:
    rows = _get_template("pricing_rows", DEFAULT_PRICING_ROWS)
    table = _template_to_dataframe(
        rows,
        ["Year", "Product", "Unit", "Base Price", "Price Growth %"],
    )

    if table.empty:
        return pd.DataFrame(
            {
                "Year": [pd.Timestamp.today().year],
                "Product": ["Product"],
                "Unit": ["Unit"],
                "Base Price": [np.nan],
                "Price Growth %": [np.nan],
            }
        )

    return table.reset_index(drop=True)


def _variable_default_items() -> list[tuple[str, Optional[float]]]:
    items: list[tuple[str, Optional[float]]] = []
    for row in _get_template("variable_items", DEFAULT_VARIABLE_ITEMS):
        item = str(row.get("Item", "")).strip() or "Variable Expense"
        share_value = pd.to_numeric(
            pd.Series([row.get("Share %")]), errors="coerce"
        ).iloc[0]
        share = float(share_value) / 100.0 if not pd.isna(share_value) else None
        items.append((item, share))

    if not items:
        items.append(("Variable Expense", None))

    return items


def _direct_wage_default_items() -> list[tuple[str, Optional[float]]]:
    roles: list[tuple[str, Optional[float]]] = []
    for row in _get_template("direct_wage_items", DEFAULT_DIRECT_WAGE_ITEMS):
        role = str(row.get("Role", "")).strip() or "Direct Wage"
        share_value = pd.to_numeric(
            pd.Series([row.get("Share %")]), errors="coerce"
        ).iloc[0]
        share = float(share_value) / 100.0 if not pd.isna(share_value) else None
        roles.append((role, share))

    if not roles:
        roles.append(("Direct Wage", None))

    return roles


def _admin_wage_default_items() -> list[tuple[str, Optional[float]]]:
    functions: list[tuple[str, Optional[float]]] = []
    for row in _get_template("admin_wage_items", DEFAULT_ADMIN_WAGE_ITEMS):
        function = str(row.get("Function", "")).strip() or "Admin Wage"
        share_value = pd.to_numeric(
            pd.Series([row.get("Share %")]), errors="coerce"
        ).iloc[0]
        share = float(share_value) / 100.0 if not pd.isna(share_value) else None
        functions.append((function, share))

    if not functions:
        functions.append(("Admin Wage", None))

    return functions


def _ensure_pricing_table(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if table is None or table.empty:
        return _default_pricing_table()

    work = table.copy()

    if "Year" not in work.columns:
        work["Year"] = np.nan

    work["Year"] = pd.to_numeric(work.get("Year"), errors="coerce")
    work["Product"] = work.get("Product", "").astype(str).str.strip()
    work.loc[work["Product"] == "", "Product"] = "Product"
    work["Unit"] = work.get("Unit", "").astype(str).str.strip()
    work.loc[work["Unit"] == "", "Unit"] = "Unit"
    work["Base Price"] = pd.to_numeric(work.get("Base Price"), errors="coerce")
    work["Price Growth %"] = pd.to_numeric(
        work.get("Price Growth %"), errors="coerce"
    )

    work = work.dropna(how="all")
    if work.empty:
        return _default_pricing_table()

    required_cols = [
        "Year",
        "Product",
        "Unit",
        "Base Price",
        "Price Growth %",
    ]
    for col in required_cols:
        if col not in work.columns:
            work[col] = np.nan

    ordered = work[required_cols + [c for c in work.columns if c not in required_cols]]
    return ordered.reset_index(drop=True)


def _add_pricing_row(table: pd.DataFrame) -> pd.DataFrame:
    work = _ensure_pricing_table(table)

    years = pd.to_numeric(work.get("Year"), errors="coerce")
    if years.notna().any():
        default_year = int(years.dropna().max())
        default_year += 1
    else:
        default_year = pd.Timestamp.today().year

    new_row = {
        "Year": default_year,
        "Product": f"Product {len(work) + 1}",
        "Unit": "Unit",
        "Base Price": np.nan,
        "Price Growth %": np.nan,
    }

    return pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)


def _remove_pricing_row(table: pd.DataFrame, index: int) -> pd.DataFrame:
    if table is None or table.empty:
        return table

    work = table.copy()
    if 0 <= index < len(work):
        work = work.drop(index=index).reset_index(drop=True)
    return work


def _apply_pricing_yearly_increment(
    table: pd.DataFrame,
    column: str,
    increment_pct: float,
    target_product: Optional[str] = None,
) -> pd.DataFrame:
    if table is None or table.empty or increment_pct == 0:
        return table

    work = _ensure_pricing_table(table)
    if column not in work.columns:
        return work

    work["Year"] = pd.to_numeric(work.get("Year"), errors="coerce")
    work["Product"] = work.get("Product", "").astype(str).str.strip()

    increment_factor = 1 + (increment_pct / 100.0)
    is_percent_column = column.endswith("%")

    for product, group in work.groupby("Product", dropna=False):
        product_key = product if isinstance(product, str) else ""
        if target_product and target_product != "All products" and product_key != target_product:
            continue

        group_sorted = group.sort_values("Year", kind="stable")
        last_value = None
        for idx, row in group_sorted.iterrows():
            current_value = pd.to_numeric(row.get(column), errors="coerce")
            if last_value is None:
                if not np.isnan(current_value):
                    last_value = current_value
                continue

            if np.isnan(last_value):
                continue

            if is_percent_column:
                last_value = last_value + increment_pct
            else:
                last_value = last_value * increment_factor

            work.at[idx, column] = last_value

    return work


def _default_operating_cost_table() -> pd.DataFrame:
    rows = _get_template("operating_rows", DEFAULT_OPERATING_COST_ROWS)
    table = _template_to_dataframe(
        rows,
        ["Year", "Category", "Monthly Cost", "Inflation %"],
    )

    if table.empty:
        return pd.DataFrame(
            {
                "Year": [pd.Timestamp.today().year],
                "Category": ["Operating Item"],
                "Monthly Cost": [np.nan],
                "Inflation %": [np.nan],
            }
        )

    return table.reset_index(drop=True)


def _ensure_operating_cost_table(
    table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if table is None or table.empty:
        return _default_operating_cost_table()

    work = table.copy()
    work["Year"] = pd.to_numeric(work.get("Year"), errors="coerce")
    work["Category"] = work.get("Category", "").astype(str).str.strip()
    work.loc[work["Category"] == "", "Category"] = np.nan
    work["Monthly Cost"] = pd.to_numeric(work.get("Monthly Cost"), errors="coerce")
    work["Inflation %"] = pd.to_numeric(work.get("Inflation %"), errors="coerce")

    work = work.dropna(how="all")

    if work.empty:
        return _default_operating_cost_table()

    if work["Category"].isna().any():
        for idx in work.index:
            if pd.isna(work.at[idx, "Category"]):
                work.at[idx, "Category"] = f"Operating Item {idx + 1}"

    if work["Year"].isna().all():
        work["Year"] = pd.Timestamp.today().year
    else:
        work["Year"] = work["Year"].ffill().fillna(work["Year"].dropna().min())
        work["Year"] = work["Year"].fillna(pd.Timestamp.today().year)
    work["Year"] = work["Year"].round().astype("Int64")

    required_cols = ["Year", "Category", "Monthly Cost", "Inflation %"]
    for col in required_cols:
        if col not in work.columns:
            work[col] = np.nan

    ordered = work[required_cols + [c for c in work.columns if c not in required_cols]]
    return ordered.sort_values(["Category", "Year"], kind="stable").reset_index(drop=True)


def _add_operating_cost_row(table: pd.DataFrame) -> pd.DataFrame:
    work = _ensure_operating_cost_table(table)
    years = pd.to_numeric(work.get("Year"), errors="coerce")
    if years.notna().any():
        default_year = int(years.dropna().max())
        default_year += 1
    else:
        default_year = pd.Timestamp.today().year
    new_row = {
        "Year": default_year,
        "Category": f"Operating Item {len(work) + 1}",
        "Monthly Cost": np.nan,
        "Inflation %": np.nan,
    }
    return pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)


def _remove_operating_cost_row(table: pd.DataFrame, index: int) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    work = table.copy()
    if 0 <= index < len(work):
        work = work.drop(index=index).reset_index(drop=True)
    return work


def _apply_operating_cost_increment(
    table: pd.DataFrame,
    increment_pct: float,
    target_category: Optional[str] = None,
    column: str = "Monthly Cost",
) -> pd.DataFrame:
    if table is None or table.empty or increment_pct == 0:
        return table

    work = _ensure_operating_cost_table(table)
    if column not in work.columns:
        return work

    work["Year"] = pd.to_numeric(work.get("Year"), errors="coerce")
    work["Category"] = work.get("Category", "").astype(str).str.strip()

    increment_factor = 1 + (increment_pct / 100.0)
    is_percent_column = column.endswith("%")

    for category, group in work.groupby("Category", dropna=False):
        category_key = category if isinstance(category, str) else ""
        if (
            target_category
            and target_category != "All categories"
            and category_key != target_category
        ):
            continue

        group_sorted = group.sort_values("Year", kind="stable")
        last_value = None
        for idx, row in group_sorted.iterrows():
            current_value = pd.to_numeric(row.get(column), errors="coerce")
            if last_value is None:
                if not np.isnan(current_value):
                    last_value = current_value
                continue

            if np.isnan(last_value):
                continue

            if is_percent_column:
                last_value = last_value + increment_pct
            else:
                last_value = last_value * increment_factor

            work.at[idx, column] = last_value

    return work.sort_values(["Category", "Year"], kind="stable").reset_index(drop=True)


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


# ---------- Direct wages helpers ----------


def _default_direct_wage_table(core: pd.DataFrame) -> pd.DataFrame:
    periods = _normalize_period(core.get("Period", pd.Series(dtype=str))).tolist()
    totals_raw = core.get("Direct Wages", pd.Series(dtype=float))
    totals = pd.to_numeric(totals_raw, errors="coerce")

    if isinstance(totals, pd.Series):
        total_series = totals.reset_index(drop=True)
    else:
        # When a scalar or unsupported type is returned, broadcast across periods
        total_series = pd.Series([totals])

    total_series = total_series.reindex(range(len(periods)), fill_value=np.nan)
    total_values = total_series.to_list()

    rows: list[dict[str, object]] = []
    if periods:
        for idx, period in enumerate(periods):
            total = total_values[idx] if idx < len(total_values) else np.nan
            for role, share in _direct_wage_default_items():
                amount = (
                    total * share
                    if share is not None and total is not None and not np.isnan(total)
                    else np.nan
                )
                rows.append(
                    {
                        "Period": period,
                        "Role": role,
                        "Amount": amount,
                    }
                )

    if not rows:
        today = (pd.Timestamp.today() + MonthEnd(0)).strftime("%Y-%m-%d")
        rows.append({"Period": today, "Role": "Direct Wage", "Amount": np.nan})

    return pd.DataFrame(rows)


def _ensure_direct_wage_table(
    table: Optional[pd.DataFrame], core: pd.DataFrame
) -> pd.DataFrame:
    if table is None or table.empty:
        return _default_direct_wage_table(core)

    work = table.copy()
    work["Period"] = _normalize_period(work.get("Period", pd.Series(dtype=str)))
    work["Role"] = work.get("Role", "").astype(str).str.strip()
    work.loc[work["Role"] == "", "Role"] = "Direct Wage"
    work["Amount"] = pd.to_numeric(work.get("Amount"), errors="coerce")

    work = work.dropna(how="all")
    work = work[(work["Role"].notna()) | (work["Amount"].notna())]
    work = work.dropna(subset=["Period"], how="all")

    if work.empty:
        return _default_direct_wage_table(core)

    return work.reset_index(drop=True)


def _add_direct_wage_row(table: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    work = _ensure_direct_wage_table(table, core)
    periods = work.get("Period", pd.Series(dtype=str))
    default_period = None
    if not periods.empty:
        default_period = periods.iloc[-1]
    else:
        core_periods = _normalize_period(core.get("Period", pd.Series(dtype=str)))
        if not core_periods.empty:
            default_period = core_periods.iloc[-1]
    if default_period is None or pd.isna(default_period):
        default_period = (pd.Timestamp.today() + MonthEnd(0)).strftime("%Y-%m-%d")

    new_row = {
        "Period": default_period,
        "Role": f"Direct Wage Item {len(work) + 1}",
        "Amount": np.nan,
    }
    return pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)


def _remove_direct_wage_row(table: pd.DataFrame, index: int) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    work = table.copy()
    if 0 <= index < len(work):
        work = work.drop(index=index).reset_index(drop=True)
    return work


def _apply_direct_wage_increment(
    table: pd.DataFrame, increment_pct: float, target_role: Optional[str] = None
) -> pd.DataFrame:
    if table is None or table.empty or increment_pct == 0:
        return table

    work = table.copy()
    work["Period_dt"] = pd.to_datetime(work.get("Period"), errors="coerce")
    work["Amount"] = pd.to_numeric(work.get("Amount"), errors="coerce")
    work["Role"] = work.get("Role", "").astype(str).str.strip()

    increment_factor = 1 + (increment_pct / 100.0)

    def _should_update(role: str) -> bool:
        if not target_role or target_role == "All roles":
            return True
        return role == target_role

    for role, group in work.groupby("Role", dropna=False):
        role_key = role if isinstance(role, str) else ""
        if not _should_update(role_key):
            continue
        group = group.sort_values("Period_dt", kind="stable")
        prev_amount = None
        prev_year = None
        for idx, row in group.iterrows():
            period_dt = row["Period_dt"]
            amount = row["Amount"]
            if pd.isna(period_dt):
                continue
            year = int(period_dt.year)
            if prev_amount is None and not pd.isna(amount):
                prev_amount = amount
                prev_year = year
                continue
            if prev_amount is None or prev_year is None:
                continue
            year_gap = year - prev_year
            if year_gap <= 0:
                if not pd.isna(amount):
                    prev_amount = amount
                    prev_year = year
                continue
            new_amount = prev_amount * (increment_factor ** year_gap)
            work.at[idx, "Amount"] = new_amount
            prev_amount = new_amount
            prev_year = year

    return work.drop(columns="Period_dt")


def _aggregate_direct_wages(
    table: pd.DataFrame, core: pd.DataFrame
) -> pd.DataFrame:
    work = _ensure_direct_wage_table(table, core)
    periods = _normalize_period(core.get("Period", pd.Series(dtype=str)))
    summary = (
        work.groupby("Period", as_index=False)["Amount"].sum(min_count=1)
        if not work.empty
        else pd.DataFrame(columns=["Period", "Amount"])
    )
    result = pd.DataFrame({"Period": periods})
    summary_map = dict(zip(summary.get("Period", []), summary.get("Amount", [])))
    result["Direct Wages"] = result["Period"].map(summary_map)
    if result["Direct Wages"].notna().any():
        result["Direct Wages"] = result["Direct Wages"].astype(float)
    else:
        result["Direct Wages"] = 0.0
    return result


# ---------- Admin wages helpers ----------


def _default_admin_wage_table(core: pd.DataFrame) -> pd.DataFrame:
    periods = _normalize_period(core.get("Period", pd.Series(dtype=str))).tolist()
    totals = pd.to_numeric(core.get("Admin Wages"), errors="coerce")
    total_values = totals.tolist() if totals is not None else []

    rows: list[dict[str, object]] = []
    if periods:
        for idx, period in enumerate(periods):
            total = total_values[idx] if idx < len(total_values) else np.nan
            for function, share in _admin_wage_default_items():
                amount = (
                    total * share
                    if share is not None and total is not None and not np.isnan(total)
                    else np.nan
                )
                rows.append(
                    {
                        "Period": period,
                        "Function": function,
                        "Amount": amount,
                    }
                )

    if not rows:
        today = (pd.Timestamp.today() + MonthEnd(0)).strftime("%Y-%m-%d")
        rows.append({"Period": today, "Function": "Admin Wage", "Amount": np.nan})

    return pd.DataFrame(rows)


def _ensure_admin_wage_table(
    table: Optional[pd.DataFrame], core: pd.DataFrame
) -> pd.DataFrame:
    if table is None or table.empty:
        return _default_admin_wage_table(core)

    work = table.copy()
    if "Admin Wages" in work.columns and "Amount" not in work.columns:
        periods = _normalize_period(work.get("Period", pd.Series(dtype=str)))
        totals = pd.to_numeric(work.get("Admin Wages"), errors="coerce")
        reconstructed: list[dict[str, object]] = []
        for idx, period in enumerate(periods):
            total = totals.iloc[idx] if idx < len(totals) else np.nan
            for function, share in _admin_wage_default_items():
                amount = (
                    total * share
                    if share is not None and total is not None and not np.isnan(total)
                    else np.nan
                )
                reconstructed.append(
                    {
                        "Period": period,
                        "Function": function,
                        "Amount": amount,
                    }
                )
        work = pd.DataFrame(reconstructed)

    work["Period"] = _normalize_period(work.get("Period", pd.Series(dtype=str)))
    if "Function" not in work.columns and "Role" in work.columns:
        work["Function"] = work["Role"]
    work["Function"] = work.get("Function", "").astype(str).str.strip()
    work.loc[work["Function"] == "", "Function"] = "Admin Wage"
    work["Amount"] = pd.to_numeric(work.get("Amount"), errors="coerce")

    work = work.dropna(how="all")
    work = work[(work["Function"].notna()) | (work["Amount"].notna())]
    work = work.dropna(subset=["Period"], how="all")

    if work.empty:
        return _default_admin_wage_table(core)

    return work.reset_index(drop=True)


def _add_admin_wage_row(table: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    work = _ensure_admin_wage_table(table, core)
    periods = work.get("Period", pd.Series(dtype=str))
    default_period = None
    if not periods.empty:
        default_period = periods.iloc[-1]
    else:
        core_periods = _normalize_period(core.get("Period", pd.Series(dtype=str)))
        if not core_periods.empty:
            default_period = core_periods.iloc[-1]
    if default_period is None or pd.isna(default_period):
        default_period = (pd.Timestamp.today() + MonthEnd(0)).strftime("%Y-%m-%d")

    new_row = {
        "Period": default_period,
        "Function": f"Admin Wage Item {len(work) + 1}",
        "Amount": np.nan,
    }
    return pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)


def _remove_admin_wage_row(table: pd.DataFrame, index: int) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    work = table.copy()
    if 0 <= index < len(work):
        work = work.drop(index=index).reset_index(drop=True)
    return work


def _apply_admin_wage_increment(
    table: pd.DataFrame, increment_pct: float, target_function: Optional[str] = None
) -> pd.DataFrame:
    if table is None or table.empty or increment_pct == 0:
        return table

    work = table.copy()
    work["Period_dt"] = pd.to_datetime(work.get("Period"), errors="coerce")
    work["Amount"] = pd.to_numeric(work.get("Amount"), errors="coerce")
    work["Function"] = work.get("Function", "").astype(str).str.strip()

    increment_factor = 1 + (increment_pct / 100.0)

    def _should_update(function: str) -> bool:
        if not target_function or target_function == "All functions":
            return True
        return function == target_function

    for function, group in work.groupby("Function", dropna=False):
        function_key = function if isinstance(function, str) else ""
        if not _should_update(function_key):
            continue
        group = group.sort_values("Period_dt", kind="stable")
        prev_amount = None
        prev_year = None
        for idx, row in group.iterrows():
            period_dt = row["Period_dt"]
            amount = row["Amount"]
            if pd.isna(period_dt):
                continue
            year = int(period_dt.year)
            if prev_amount is None and not pd.isna(amount):
                prev_amount = amount
                prev_year = year
                continue
            if prev_amount is None or prev_year is None:
                continue
            year_gap = year - prev_year
            if year_gap <= 0:
                if not pd.isna(amount):
                    prev_amount = amount
                    prev_year = year
                continue
            new_amount = prev_amount * (increment_factor ** year_gap)
            work.at[idx, "Amount"] = new_amount
            prev_amount = new_amount
            prev_year = year

    return work.drop(columns="Period_dt")


def _aggregate_admin_wages(
    table: pd.DataFrame, core: pd.DataFrame
) -> pd.DataFrame:
    work = _ensure_admin_wage_table(table, core)
    periods = _normalize_period(core.get("Period", pd.Series(dtype=str)))
    summary = (
        work.groupby("Period", as_index=False)["Amount"].sum(min_count=1)
        if not work.empty
        else pd.DataFrame(columns=["Period", "Amount"])
    )
    result = pd.DataFrame({"Period": periods})
    summary_map = dict(zip(summary.get("Period", []), summary.get("Amount", [])))
    result["Admin Wages"] = result["Period"].map(summary_map)
    if result["Admin Wages"].notna().any():
        result["Admin Wages"] = result["Admin Wages"].astype(float)
    else:
        result["Admin Wages"] = 0.0
    return result


# ---------- Variable expenses helpers ----------


def _default_variable_expense_table(core: pd.DataFrame) -> pd.DataFrame:
    periods = _normalize_period(core.get("Period", pd.Series(dtype=str))).tolist()
    revenue = pd.to_numeric(core.get("Revenue"), errors="coerce")
    revenue_values = revenue.tolist() if revenue is not None else []

    rows: list[dict[str, object]] = []
    if periods:
        for idx, period in enumerate(periods):
            rev = revenue_values[idx] if idx < len(revenue_values) else np.nan
            for item, share in _variable_default_items():
                amount = (
                    rev * share if share is not None and rev is not None and not np.isnan(rev)
                    else np.nan
                )
                rows.append({
                    "Period": period,
                    "Item": item,
                    "Amount": amount,
                })

    if not rows:
        today = (pd.Timestamp.today() + MonthEnd(0)).strftime("%Y-%m-%d")
        rows.append({"Period": today, "Item": "Variable Expense", "Amount": np.nan})

    return pd.DataFrame(rows)


def _ensure_variable_expense_table(
    table: Optional[pd.DataFrame], core: pd.DataFrame
) -> pd.DataFrame:
    if table is None or table.empty:
        return _default_variable_expense_table(core)

    work = table.copy()
    work["Period"] = _normalize_period(work.get("Period", pd.Series(dtype=str)))
    work["Item"] = work.get("Item", "").astype(str).str.strip()
    work.loc[work["Item"] == "", "Item"] = "Variable Expense"
    work["Amount"] = pd.to_numeric(work.get("Amount"), errors="coerce")

    work = work.dropna(how="all")
    work = work[(work["Item"].notna()) | (work["Amount"].notna())]
    work = work.dropna(subset=["Period"], how="all")

    if work.empty:
        return _default_variable_expense_table(core)

    return work.reset_index(drop=True)


def _add_variable_expense_row(table: pd.DataFrame, core: pd.DataFrame) -> pd.DataFrame:
    work = _ensure_variable_expense_table(table, core)
    periods = work.get("Period", pd.Series(dtype=str))
    default_period = None
    if not periods.empty:
        default_period = periods.iloc[-1]
    else:
        core_periods = _normalize_period(core.get("Period", pd.Series(dtype=str)))
        if not core_periods.empty:
            default_period = core_periods.iloc[-1]
    if default_period is None or pd.isna(default_period):
        default_period = (pd.Timestamp.today() + MonthEnd(0)).strftime("%Y-%m-%d")

    new_row = {
        "Period": default_period,
        "Item": f"Variable Item {len(work) + 1}",
        "Amount": np.nan,
    }
    return pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)


def _remove_variable_expense_row(table: pd.DataFrame, index: int) -> pd.DataFrame:
    if table is None or table.empty:
        return table
    work = table.copy()
    if 0 <= index < len(work):
        work = work.drop(index=index).reset_index(drop=True)
    return work


def _apply_variable_expense_increment(
    table: pd.DataFrame, increment_pct: float, target_item: Optional[str] = None
) -> pd.DataFrame:
    if table is None or table.empty or increment_pct == 0:
        return table

    work = table.copy()
    work["Period_dt"] = pd.to_datetime(work.get("Period"), errors="coerce")
    work["Amount"] = pd.to_numeric(work.get("Amount"), errors="coerce")
    work["Item"] = work.get("Item", "").astype(str).str.strip()

    increment_factor = 1 + (increment_pct / 100.0)

    def _should_update(item: str) -> bool:
        if not target_item or target_item == "All items":
            return True
        return item == target_item

    for item, group in work.groupby("Item", dropna=False):
        if not _should_update(item if isinstance(item, str) else ""):
            continue
        group = group.sort_values("Period_dt", kind="stable")
        prev_amount = None
        prev_year = None
        for idx, row in group.iterrows():
            period_dt = row["Period_dt"]
            amount = row["Amount"]
            if pd.isna(period_dt):
                continue
            year = int(period_dt.year)
            if prev_amount is None and not pd.isna(amount):
                prev_amount = amount
                prev_year = year
                continue
            if prev_amount is None or prev_year is None:
                continue
            year_gap = year - prev_year
            if year_gap <= 0:
                if not pd.isna(amount):
                    prev_amount = amount
                    prev_year = year
                continue
            new_amount = prev_amount * (increment_factor ** year_gap)
            work.at[idx, "Amount"] = new_amount
            prev_amount = new_amount
            prev_year = year

    return work.drop(columns="Period_dt")


def _aggregate_variable_expenses(
    table: pd.DataFrame, core: pd.DataFrame
) -> pd.DataFrame:
    work = _ensure_variable_expense_table(table, core)
    periods = _normalize_period(core.get("Period", pd.Series(dtype=str)))
    summary = (
        work.groupby("Period", as_index=False)["Amount"].sum(min_count=1)
        if not work.empty
        else pd.DataFrame(columns=["Period", "Amount"])
    )
    result = pd.DataFrame({"Period": periods})
    summary_map = dict(zip(summary.get("Period", []), summary.get("Amount", [])))
    result["Variable Expenses"] = result["Period"].map(summary_map)
    if result["Variable Expenses"].notna().any():
        result["Variable Expenses"] = result["Variable Expenses"].astype(float)
    else:
        result["Variable Expenses"] = 0.0
    return result


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


def _ensure_asset_schedule(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if table is None or table.empty:
        work = pd.DataFrame(columns=ASSET_SCHEDULE_COLUMNS)
    else:
        work = table.copy()

    for column in ASSET_SCHEDULE_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan

    work["Asset"] = work["Asset"].fillna("").astype(str)
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce")

    numeric_cols = [
        "Opening NBV",
        "Additions",
        "Depreciation Rate %",
        "Depreciation",
        "Closing NBV",
    ]
    for column in numeric_cols:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    ordered = [col for col in ASSET_SCHEDULE_COLUMNS if col in work.columns]
    remaining = [col for col in work.columns if col not in ordered]

    work = work[ordered + remaining].reset_index(drop=True)
    return _recalculate_asset_schedule(work)


def _recalculate_asset_schedule(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame(columns=ASSET_SCHEDULE_COLUMNS)

    work = table.copy()

    opening = pd.to_numeric(work.get("Opening NBV"), errors="coerce").fillna(0.0)
    additions = pd.to_numeric(work.get("Additions"), errors="coerce").fillna(0.0)
    rate = pd.to_numeric(work.get("Depreciation Rate %"), errors="coerce")
    depreciation = pd.to_numeric(work.get("Depreciation"), errors="coerce")

    total_basis = opening + additions

    if not rate.isna().all():
        calc_dep = total_basis * (rate.fillna(0.0) / 100.0)
        dep_mask = depreciation.isna() & rate.notna()
        depreciation.loc[dep_mask] = calc_dep.loc[dep_mask]

    with np.errstate(divide="ignore", invalid="ignore"):
        rate_mask = rate.isna() & depreciation.notna() & (total_basis != 0)
        rate.loc[rate_mask] = (depreciation.loc[rate_mask] / total_basis.loc[rate_mask]) * 100.0

    closing = opening + additions - depreciation.fillna(0.0)

    work["Depreciation Rate %"] = rate
    work["Depreciation"] = depreciation
    work["Closing NBV"] = closing

    return work


def _apply_asset_rate(table: pd.DataFrame, rate: float) -> pd.DataFrame:
    work = _ensure_asset_schedule(table)
    work["Depreciation Rate %"] = rate
    return _recalculate_asset_schedule(work)


def _apply_asset_additions_pattern(
    table: pd.DataFrame, base_amount: float, increment_pct: float
) -> pd.DataFrame:
    work = _ensure_asset_schedule(table)
    if work.empty:
        return work

    temp = work.copy()
    temp["__year"] = pd.to_numeric(temp["Year"], errors="coerce")
    temp = temp.sort_values("__year", kind="stable")

    factor = 1 + (increment_pct / 100.0)
    position = 0
    for idx, row in temp.iterrows():
        year = row["__year"]
        if pd.isna(year):
            continue
        temp.at[idx, "Additions"] = base_amount * (factor ** position)
        position += 1

    temp = temp.drop(columns="__year")
    work.loc[temp.index, "Additions"] = temp["Additions"]
    return _recalculate_asset_schedule(work)


def _apply_asset_yearly_increment(
    table: pd.DataFrame, column: str, increment_pct: float
) -> pd.DataFrame:
    work = _ensure_asset_schedule(table)
    if work.empty or column not in work.columns or increment_pct == 0:
        return work

    temp = work.copy()
    temp["__year"] = pd.to_numeric(temp["Year"], errors="coerce")
    temp = temp.sort_values("__year", kind="stable")

    factor = 1 + (increment_pct / 100.0)
    base_value = None
    last_year = None

    for idx, row in temp.iterrows():
        year = row["__year"]
        if pd.isna(year):
            continue

        current = pd.to_numeric(row[column], errors="coerce")
        if base_value is None:
            base_value = 0.0 if pd.isna(current) else float(current)
            last_year = int(year)
            temp.at[idx, column] = base_value
            continue

        year_gap = int(year) - int(last_year)
        if year_gap > 0:
            base_value = base_value * (factor ** year_gap)
            last_year = int(year)

        temp.at[idx, column] = base_value

    temp = temp.drop(columns="__year")
    work.loc[temp.index, column] = temp[column]
    return _recalculate_asset_schedule(work)


def _add_asset_row(
    table: pd.DataFrame, default_rate: float = 5.0, default_additions: float = 0.0
) -> pd.DataFrame:
    work = _ensure_asset_schedule(table)

    if work.empty:
        next_year = pd.Timestamp.today().year
    else:
        years = pd.to_numeric(work["Year"], errors="coerce")
        next_year = int(years.dropna().max()) + 1 if years.notna().any() else pd.Timestamp.today().year

    new_row = {
        "Asset": "New Asset",
        "Year": next_year,
        "Opening NBV": 0.0,
        "Additions": default_additions,
        "Depreciation Rate %": default_rate,
        "Depreciation": np.nan,
        "Closing NBV": np.nan,
    }

    work = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)
    return _recalculate_asset_schedule(work)


def _remove_asset_row(table: pd.DataFrame, index: int) -> pd.DataFrame:
    work = _ensure_asset_schedule(table)
    if index not in work.index:
        return work
    reduced = work.drop(index=index).reset_index(drop=True)
    return _recalculate_asset_schedule(reduced)


def _ensure_capitalisation_table(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame({col: [] for col in CAP_TABLE_COLUMNS})

    work = table.copy()
    for column in CAP_TABLE_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan

    work["Year"] = pd.to_numeric(work["Year"], errors="coerce").astype("Int64")
    shareholders = work.get("Shareholder")
    if shareholders is None:
        shareholders = pd.Series(["" for _ in range(len(work))])
    else:
        shareholders = shareholders.astype("string")
    work["Shareholder"] = shareholders.fillna("").replace({pd.NA: "", "<NA>": "", "nan": ""})
    work["Ownership %"] = pd.to_numeric(work["Ownership %"], errors="coerce")
    work["Investment"] = pd.to_numeric(work["Investment"], errors="coerce")

    ordered = CAP_TABLE_COLUMNS + [col for col in work.columns if col not in CAP_TABLE_COLUMNS]
    return work[ordered].reset_index(drop=True)


def _add_capitalisation_row(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    work = _ensure_capitalisation_table(table)

    years = pd.to_numeric(work.get("Year"), errors="coerce")
    if years.notna().any():
        next_year = int(years.max()) + 1
    else:
        next_year = pd.Timestamp.today().year

    new_row = {
        "Year": next_year,
        "Shareholder": "New Investor",
        "Ownership %": np.nan,
        "Investment": np.nan,
    }

    combined = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)
    return _ensure_capitalisation_table(combined)


def _remove_capitalisation_row(table: Optional[pd.DataFrame], index: int) -> pd.DataFrame:
    work = _ensure_capitalisation_table(table)
    if work.empty or index not in work.index:
        return work
    reduced = work.drop(index=index)
    return _ensure_capitalisation_table(reduced)


def _apply_capitalisation_increment(
    table: Optional[pd.DataFrame], column: str, increment_pct: float
) -> pd.DataFrame:
    if column not in {"Ownership %", "Investment"}:
        return _ensure_capitalisation_table(table)

    work = _ensure_capitalisation_table(table)
    if work.empty or increment_pct == 0:
        return work

    temp = work.copy()
    temp["__year"] = pd.to_numeric(temp["Year"], errors="coerce")
    temp = temp.sort_values(["Shareholder", "__year"], kind="stable")

    increment_factor = 1 + (increment_pct / 100.0)

    last_values: Dict[str, tuple[Optional[float], Optional[float]]] = {}
    for idx, row in temp.iterrows():
        shareholder = row["Shareholder"] or "Unnamed"
        year = row["__year"]
        if pd.isna(year):
            continue

        previous_value, previous_year = last_values.get(shareholder, (None, None))
        current_value = row[column]

        if previous_year is None or pd.isna(previous_year):
            if pd.notna(current_value):
                last_values[shareholder] = (float(current_value), float(year))
            else:
                last_values[shareholder] = (None, float(year))
            continue

        years_elapsed = int(float(year) - previous_year)
        if years_elapsed <= 0:
            if pd.notna(current_value):
                last_values[shareholder] = (float(current_value), float(year))
            continue

        base_value = previous_value
        if base_value is None:
            if pd.isna(current_value):
                continue
            base_value = float(current_value)

        new_value = float(base_value) * (increment_factor ** years_elapsed)
        temp.at[idx, column] = new_value
        last_values[shareholder] = (new_value, float(year))

    temp = temp.drop(columns="__year")
    work.loc[temp.index, column] = temp[column]
    return _ensure_capitalisation_table(work)


def _ensure_capex_schedule(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if table is None or table.empty:
        work = pd.DataFrame(columns=CAPEX_TABLE_COLUMNS)
    else:
        work = table.copy()

    for column in CAPEX_TABLE_COLUMNS:
        if column not in work.columns:
            work[column] = np.nan

    work["Category"] = work["Category"].fillna("").astype(str)
    work["Year"] = pd.to_numeric(work["Year"], errors="coerce").astype("Int64")

    work["Spend"] = pd.to_numeric(work["Spend"], errors="coerce")
    rate = pd.to_numeric(work["Depreciation Rate %"], errors="coerce")

    fraction_mask = rate.notna() & (rate.abs() <= 1.0)
    rate.loc[fraction_mask] = rate.loc[fraction_mask] * 100.0

    work["Depreciation Rate %"] = rate
    work["Depreciation"] = pd.to_numeric(work["Depreciation"], errors="coerce")

    ordered = [col for col in CAPEX_TABLE_COLUMNS if col in work.columns]
    remaining = [col for col in work.columns if col not in ordered]

    return _recalculate_capex_schedule(work[ordered + remaining].reset_index(drop=True))


def _recalculate_capex_schedule(table: pd.DataFrame) -> pd.DataFrame:
    if table is None or table.empty:
        return pd.DataFrame(columns=CAPEX_TABLE_COLUMNS)

    work = table.copy()

    spend = pd.to_numeric(work.get("Spend"), errors="coerce")
    rate = pd.to_numeric(work.get("Depreciation Rate %"), errors="coerce").fillna(0.0)

    depreciation = spend * (rate / 100.0)
    override = pd.to_numeric(work.get("Depreciation"), errors="coerce")
    depreciation = depreciation.where(override.isna(), override)

    work["Spend"] = spend
    work["Depreciation Rate %"] = rate
    work["Depreciation"] = depreciation

    return work


def _apply_capex_rate(table: Optional[pd.DataFrame], rate: float) -> pd.DataFrame:
    work = _ensure_capex_schedule(table)
    if work.empty:
        return work
    work["Depreciation Rate %"] = rate
    return _recalculate_capex_schedule(work)


def _apply_capex_yearly_increment(
    table: Optional[pd.DataFrame], column: str, increment_pct: float
) -> pd.DataFrame:
    work = _ensure_capex_schedule(table)
    if work.empty or column not in work.columns or increment_pct == 0:
        return work

    temp = work.copy()
    temp["__year"] = pd.to_numeric(temp["Year"], errors="coerce")
    temp = temp.sort_values(["__year"], kind="stable")

    factor = 1 + (increment_pct / 100.0)
    base_value = None
    last_year = None

    for idx, row in temp.iterrows():
        year = row["__year"]
        if pd.isna(year):
            continue

        current_value = pd.to_numeric(row[column], errors="coerce")
        if base_value is None:
            base_value = 0.0 if pd.isna(current_value) else float(current_value)
            last_year = int(year)
            temp.at[idx, column] = base_value
            continue

        year_gap = int(year) - int(last_year)
        if year_gap > 0:
            base_value = base_value * (factor ** year_gap)
            last_year = int(year)

        temp.at[idx, column] = base_value

    temp = temp.drop(columns="__year")
    work.loc[temp.index, column] = temp[column]
    return _recalculate_capex_schedule(work)


def _add_capex_row(
    table: Optional[pd.DataFrame], default_rate: float = 10.0, default_spend: float = 0.0
) -> pd.DataFrame:
    work = _ensure_capex_schedule(table)

    years = pd.to_numeric(work.get("Year"), errors="coerce")
    if years.notna().any():
        next_year = int(years.max()) + 1
    else:
        next_year = pd.Timestamp.today().year

    new_row = {
        "Year": next_year,
        "Category": "New Capex",
        "Spend": default_spend,
        "Depreciation Rate %": default_rate,
        "Depreciation": np.nan,
    }

    combined = pd.concat([work, pd.DataFrame([new_row])], ignore_index=True)
    return _recalculate_capex_schedule(combined)


def _remove_capex_row(table: Optional[pd.DataFrame], index: int) -> pd.DataFrame:
    work = _ensure_capex_schedule(table)
    if work.empty or index not in work.index:
        return work
    reduced = work.drop(index=index).reset_index(drop=True)
    return _recalculate_capex_schedule(reduced)


def _default_income_schedule(periods: int = 12, start: str = "2024-01-31") -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq=MonthEnd(1))
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


def _derive_horizon_years(
    production_horizon: Optional[pd.DataFrame],
) -> tuple[int, int]:
    """Return start and end years inferred from the production horizon table."""

    default_start = 2024
    default_end = 2024

    if production_horizon is None or production_horizon.empty:
        return default_start, default_end

    start_years = pd.to_numeric(
        production_horizon.get("Start Year"), errors="coerce"
    ).dropna()
    end_years = pd.to_numeric(
        production_horizon.get("End Year"), errors="coerce"
    ).dropna()

    start_year = int(start_years.iloc[0]) if not start_years.empty else default_start
    end_year = int(end_years.iloc[0]) if not end_years.empty else default_end

    if end_year < start_year:
        end_year = start_year

    return start_year, end_year


def _default_schedule_components(
    periods: Optional[int] = None,
    start: Optional[str] = None,
    production_horizon: Optional[pd.DataFrame] = None,
) -> tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    if production_horizon is None:
        production_horizon = _default_production_horizon_table()

    start_year, end_year = _derive_horizon_years(production_horizon)

    if periods is None:
        periods = max(1, (end_year - start_year + 1) * 12)

    if start is None:
        start_date = pd.Timestamp(start_year, 1, 1) + pd.offsets.MonthEnd(0)
        start = start_date.strftime("%Y-%m-%d")

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
        if name == "Variable Expenses Schedule":
            detail_tables[name] = _default_variable_expense_table(base)
            continue
        if name == "Direct Wages Schedule":
            detail_tables[name] = _default_direct_wage_table(base)
            continue
        if name == "Admin Wages Schedule":
            detail_tables[name] = _default_admin_wage_table(base)
            continue
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
                "Depreciation Rate %": [8.0, 6.5],
                "Depreciation": [3600.0, 2470.0],
            }
        ),
        "Asset Schedules": pd.DataFrame(
            {
                "Asset": ["Barn", "Parlour"],
                "Year": [2024, 2025],
                "Opening NBV": [120000.0, 65000.0],
                "Additions": [10000.0, 5000.0],
                "Depreciation Rate %": [6.0, 5.5],
                "Depreciation": [8000.0, 3600.0],
                "Closing NBV": [122000.0, 66400.0],
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


def _default_scenario_controls_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Driver": [
                "Milk price change (%)",
                "Feed cost change (%)",
            ],
            "Change %": [0.0, 0.0],
        }
    )


def _ensure_scenario_controls_table(
    table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if table is None or table.empty:
        work = _default_scenario_controls_table()
    else:
        work = table.copy()

    if "Driver" not in work.columns:
        work["Driver"] = ""
    work["Driver"] = work.get("Driver", "").astype(str).str.strip()
    work.loc[work["Driver"] == "", "Driver"] = "Driver"

    if "Change %" not in work.columns:
        work["Change %"] = np.nan
    work["Change %"] = pd.to_numeric(work.get("Change %"), errors="coerce")

    defaults = _default_scenario_controls_table()
    for _, default_row in defaults.iterrows():
        driver = str(default_row["Driver"])
        if not work["Driver"].str.casefold().eq(driver.casefold()).any():
            work = pd.concat(
                [
                    work,
                    pd.DataFrame(
                        {
                            "Driver": [driver],
                            "Change %": [float(default_row["Change %"])],
                        }
                    ),
                ],
                ignore_index=True,
            )

    ordered_cols = ["Driver", "Change %"]
    remainder = [col for col in work.columns if col not in ordered_cols]
    return work[ordered_cols + remainder].reset_index(drop=True)


def _scenario_controls_value_map(table: pd.DataFrame) -> Dict[str, float]:
    work = _ensure_scenario_controls_table(table)
    values: Dict[str, float] = {}
    for _, row in work.iterrows():
        driver = str(row.get("Driver", "")).strip()
        change = pd.to_numeric(pd.Series([row.get("Change %")]), errors="coerce").iloc[0]
        if driver:
            values[driver] = float(change) if not pd.isna(change) else 0.0
    return values


def _update_scenario_control_value(
    table: pd.DataFrame, driver: str, value: float
) -> pd.DataFrame:
    work = _ensure_scenario_controls_table(table)
    driver_key = str(driver).strip()
    if not driver_key:
        return work

    mask = work["Driver"].str.casefold() == driver_key.casefold()
    if mask.any():
        work.loc[mask, "Change %"] = float(value)
    else:
        work = pd.concat(
            [
                work,
                pd.DataFrame({"Driver": [driver_key], "Change %": [float(value)]}),
            ],
            ignore_index=True,
        )
    return _ensure_scenario_controls_table(work)


def _default_production_horizon_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Start Year": [2024],
            "End Year": [2030],
        }
    )


def _ensure_production_horizon_table(
    table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if table is None or table.empty:
        work = _default_production_horizon_table()
    else:
        work = table.copy()

    for column in ["Start Year", "End Year"]:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work.get(column), errors="coerce")

    work = work.dropna(how="all")
    if work.empty:
        return _default_production_horizon_table()

    defaults = _default_production_horizon_table().iloc[0]
    work["Start Year"] = work["Start Year"].fillna(defaults["Start Year"])
    work["End Year"] = work["End Year"].fillna(defaults["End Year"])

    for column in ["Start Year", "End Year"]:
        work[column] = work[column].round().astype("Int64")

    ordered_cols = ["Start Year", "End Year"]
    remainder = [col for col in work.columns if col not in ordered_cols]
    return work[ordered_cols + remainder].reset_index(drop=True)


def _production_year_options(start_year: int, end_year: int) -> list[int]:
    """Return a flexible list of year options covering the supplied range."""

    current_year = pd.Timestamp.today().year
    minimum = min(start_year, end_year, current_year - 10, 2000)
    maximum = max(start_year, end_year, current_year + 20)
    return list(range(minimum, maximum + 1))


def _safe_timeline(table: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Convert a schedule dataframe into a timeline index, ignoring errors."""

    if isinstance(table, pd.DataFrame) and not table.empty:
        try:
            return _prepare_timeline_table(table)
        except ValueError:
            return pd.DataFrame(index=pd.DatetimeIndex([], name="Period"))
    return pd.DataFrame(index=pd.DatetimeIndex([], name="Period"))


def _timeline_to_schedule_frame(
    timeline: pd.DataFrame, column_order: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    """Convert a datetime-indexed timeline back into a schedule dataframe."""

    if timeline.empty:
        base_columns: list[str] = ["Period"]
        if column_order:
            base_columns.extend(
                [col for col in column_order if col not in {"Period"}]
            )
        return pd.DataFrame(columns=base_columns)

    ordered = timeline.sort_index()
    frame = ordered.reset_index()
    first_column = frame.columns[0]
    if first_column != "Period":
        frame = frame.rename(columns={first_column: "Period"})

    frame["Period"] = _normalize_period(frame.get("Period", pd.Series(dtype=str)))

    if column_order:
        normalized_order = [
            col for col in column_order if col not in {"Period"} and col in frame.columns
        ]
    else:
        normalized_order = []

    remainder = [
        col
        for col in frame.columns
        if col not in {"Period", *normalized_order}
    ]
    return frame[["Period", *normalized_order, *remainder]]


def _merge_schedule_table(
    existing: Optional[pd.DataFrame],
    defaults: Optional[pd.DataFrame],
    period_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Merge an existing schedule with defaults to match the target horizon."""

    column_order: list[str] = []
    if isinstance(existing, pd.DataFrame) and not existing.empty:
        column_order = list(existing.columns)
    elif isinstance(defaults, pd.DataFrame) and not defaults.empty:
        column_order = list(defaults.columns)

    existing_timeline = _safe_timeline(existing)
    default_timeline = _safe_timeline(defaults)

    aligned_defaults = default_timeline.reindex(period_index)
    aligned_existing = existing_timeline.reindex(period_index)

    merged = aligned_defaults.copy()
    if not aligned_existing.empty:
        merged = merged.combine_first(aligned_existing)
        merged.update(aligned_existing)

    merged = merged.reindex(period_index)
    return _timeline_to_schedule_frame(merged, column_order)


def _rebase_schedule_to_horizon(
    core: Optional[pd.DataFrame],
    detail_tables: Optional[Dict[str, pd.DataFrame]],
    start_year: int,
    end_year: int,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Return schedule tables that span the requested production horizon."""

    if start_year > end_year:
        start_year, end_year = end_year, start_year

    horizon_table = pd.DataFrame({"Start Year": [start_year], "End Year": [end_year]})
    default_core, default_details = _default_schedule_components(
        production_horizon=horizon_table
    )

    start_date = (pd.Timestamp(start_year, 1, 1) + MonthEnd(0))
    end_date = (pd.Timestamp(end_year, 12, 1) + MonthEnd(0))
    period_index = pd.date_range(start=start_date, end=end_date, freq=MonthEnd())

    merged_core = _merge_schedule_table(core, default_core, period_index)

    existing_details = detail_tables or {}
    detail_names = set(default_details.keys()) | set(existing_details.keys())

    merged_details: Dict[str, pd.DataFrame] = {}
    for name in detail_names:
        merged_details[name] = _merge_schedule_table(
            existing_details.get(name),
            default_details.get(name),
            period_index,
        )

    return merged_core, merged_details


def _reset_cached_results() -> None:
    """Clear cached scenario results so defaults regenerate with fresh inputs."""

    st.session_state.all_scenario_results = {}
    st.session_state.results = None
    st.session_state.selected_scenario_name = next(iter(SCENARIO_PRESETS))


def _sync_production_horizon(start_year: int, end_year: int) -> None:
    """Ensure schedules and cached results reflect the selected horizon."""

    core_table = st.session_state.get("core_schedule")
    detail_tables = st.session_state.get("detail_schedules")

    merged_core, merged_details = _rebase_schedule_to_horizon(
        core_table, detail_tables, start_year, end_year
    )

    st.session_state.core_schedule = merged_core
    st.session_state.detail_schedules = merged_details

    _clear_schedule_editor_state("core_schedule")
    for name in merged_details:
        identifier = f"detail::{_scenario_key_suffix(name)}"
        _clear_schedule_editor_state(identifier)

    _reset_cached_results()

def _default_capital_financing_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Source": ["Bank Loan", "Equity"],
            "Amount": [250000.0, 150000.0],
            "Interest/Return %": [6.5, 0.0],
            "Term (years)": [7, None],
        }
    )


def _ensure_capital_financing_table(
    table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if table is None or table.empty:
        work = _default_capital_financing_table()
    else:
        work = table.copy()

    if "Source" not in work.columns:
        work["Source"] = ""
    work["Source"] = work.get("Source", "").astype(str).str.strip()
    work.loc[work["Source"] == "", "Source"] = "Source"

    for column in ["Amount", "Interest/Return %", "Term (years)"]:
        if column not in work.columns:
            work[column] = np.nan
        work[column] = pd.to_numeric(work.get(column), errors="coerce")

    ordered_cols = ["Source", "Amount", "Interest/Return %", "Term (years)"]
    remainder = [col for col in work.columns if col not in ordered_cols]
    return work[ordered_cols + remainder].reset_index(drop=True)


def _default_valuation_inputs_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Metric": list(DEFAULT_VALUATION_INPUTS.keys()),
            "Value": list(DEFAULT_VALUATION_INPUTS.values()),
        }
    )


def _ensure_valuation_inputs_table(
    table: Optional[pd.DataFrame],
) -> pd.DataFrame:
    if table is None or table.empty:
        work = _default_valuation_inputs_table()
    else:
        work = table.copy()

    if "Metric" not in work.columns:
        work["Metric"] = ""
    work["Metric"] = work.get("Metric", "").astype(str).str.strip()
    work.loc[work["Metric"] == "", "Metric"] = "Metric"

    if "Value" not in work.columns:
        work["Value"] = np.nan
    work["Value"] = pd.to_numeric(work.get("Value"), errors="coerce")

    ordered_cols = ["Metric", "Value"]
    remainder = [col for col in work.columns if col not in ordered_cols]
    return work[ordered_cols + remainder].reset_index(drop=True)


def _valuation_table_to_inputs(table: pd.DataFrame) -> Dict[str, float]:
    work = _ensure_valuation_inputs_table(table)
    inputs: Dict[str, float] = {}
    for _, row in work.iterrows():
        metric = str(row.get("Metric", "")).strip()
        value = pd.to_numeric(pd.Series([row.get("Value")]), errors="coerce").iloc[0]
        if metric and not pd.isna(value):
            inputs[metric] = float(value)
    return inputs


def _default_assumption_tables() -> Dict[str, pd.DataFrame]:
    return {
        "Scenario Controls": _default_scenario_controls_table(),
        "Production Horizon": _default_production_horizon_table(),
        "Pricing": _default_pricing_table(),
        "Operating Costs": _default_operating_cost_table(),
        "Capital & Financing": _default_capital_financing_table(),
        "Valuation Inputs": _default_valuation_inputs_table(),
    }


def _ensure_default_results_loaded() -> None:
    """Populate the dashboard with default results for the initial view."""

    if st.session_state.get("all_scenario_results"):
        return

    core_table = st.session_state.get("core_schedule")
    detail_tables = st.session_state.get("detail_schedules")
    supplementary_tables = st.session_state.get("supplementary")

    if core_table is None or detail_tables is None:
        return

    try:
        core_clean = _clean_editor_table(core_table)
        if core_clean is None:
            return
        core_prepared = _prepare_timeline_table(core_clean)
    except ValueError:
        return

    prepared_details: Dict[str, pd.DataFrame] = {}
    for name, table in detail_tables.items():
        cleaned = _clean_editor_table(table)
        if cleaned is None:
            continue
        try:
            prepared = _prepare_timeline_table(cleaned)
        except ValueError:
            continue

        expected_cols = DETAIL_SCHEDULE_COLUMNS.get(name)
        if expected_cols:
            missing = [col for col in expected_cols if col not in prepared.columns]
            if missing:
                continue
            prepared = prepared[expected_cols]

        prepared_details[name] = prepared

    try:
        schedule_df = _assemble_schedule(core_prepared, prepared_details)
    except ValueError:
        return

    valuation_inputs = dict(DEFAULT_VALUATION_INPUTS)
    supplementary_copy = {
        name: table.copy()
        for name, table in (supplementary_tables or {}).items()
        if isinstance(table, pd.DataFrame)
    }

    try:
        scenario_suite = _build_scenario_suite()
        model, _, scenario_results = _execute_scenario_suite(
            schedule_df,
            valuation_inputs,
            supplementary_copy,
            scenario_suite,
        )
    except ValueError:
        return

    st.session_state.all_scenario_results = scenario_results

    preferred = st.session_state.get("selected_scenario_name")
    if preferred not in scenario_results:
        preferred = next(iter(scenario_results))

    st.session_state.selected_scenario_name = preferred
    st.session_state.results = scenario_results[preferred]


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


def _render_table(title: str, table: Optional[pd.DataFrame]) -> None:
    if table is None:
        st.info(f"No **{title}** data was provided.")
        return
    st.subheader(title)
    st.dataframe(table)


ANALYTICS_FRAMEWORK_TOOLS: List[Dict[str, str]] = [
    {
        "key": "sensitivity_analysis",
        "title": "Sensitivity Analysis",
        "methodology": "One-at-a-time and multi-driver elasticity tests against profitability and cash flow outputs.",
        "visualization": "Spider plot + elasticity heatmap",
    },
    {
        "key": "scenario_stress_testing",
        "title": "Scenario & Stress Testing",
        "methodology": "Apply severe but plausible shocks and compare resilience across operating, liquidity, and valuation KPIs.",
        "visualization": "Scenario bridge chart + downside table",
    },
    {
        "key": "trend_seasonality",
        "title": "Trend & Seasonality Decomposition",
        "methodology": "Decompose historical production/revenue signals into trend, seasonal, and residual components.",
        "visualization": "Trend-season-residual line charts",
    },
    {
        "key": "customer_product_segmentation",
        "title": "Customer & Product Segmentation",
        "methodology": "Segment products/channels by margin, growth, and volatility to prioritize strategic focus areas.",
        "visualization": "Segment matrix + contribution waterfall",
    },
    {
        "key": "monte_carlo_simulation",
        "title": "Monte Carlo Simulation",
        "methodology": "Run probabilistic simulations for revenue, costs, NPV, and IRR based on configured distributions.",
        "visualization": "Distribution histogram + percentile fan chart",
    },
    {
        "key": "what_if_analysis",
        "title": "What-If Analysis",
        "methodology": "Interactive assumption perturbation with immediate recalc of key KPI outputs.",
        "visualization": "Delta KPI cards + before/after bars",
    },
    {
        "key": "goal_seek",
        "title": "Goal Seek Routines",
        "methodology": "Solve for required input values that satisfy target profitability, liquidity, and return constraints.",
        "visualization": "Target vs solved-input table",
    },
    {
        "key": "tornado_spider",
        "title": "Tornado Charts & Spider Diagrams",
        "methodology": "Rank assumptions by marginal impact on NPV/IRR and display response curves.",
        "visualization": "Tornado bar chart + spider line chart",
    },
    {
        "key": "regression_modeling",
        "title": "Regression Modeling",
        "methodology": "Estimate explanatory relationships between historical drivers and financial outcomes.",
        "visualization": "Coefficient chart + fitted vs actual scatter",
    },
    {
        "key": "time_series_models",
        "title": "Time Series (ARIMA/Prophet/LSTM)",
        "methodology": "Forecast cyclical and seasonal patterns in revenues, prices, and expenses using multiple model classes.",
        "visualization": "Forecast bands + model comparison table",
    },
    {
        "key": "classification_models",
        "title": "Classification Models",
        "methodology": "Classify credit risk/churn/segment outcomes from labeled features and operational indicators.",
        "visualization": "Confusion matrix + lift chart",
    },
    {
        "key": "linear_nonlinear_optimization",
        "title": "Linear/Nonlinear Optimization",
        "methodology": "Optimize objective functions under operational and capital constraints.",
        "visualization": "Optimal allocation table + constraint slack chart",
    },
    {
        "key": "portfolio_optimization",
        "title": "Portfolio Optimization",
        "methodology": "Balance risk and return across herds/product lines using mean-variance and robust alternatives.",
        "visualization": "Efficient frontier + allocation pie",
    },
    {
        "key": "real_options_analysis",
        "title": "Real Options Analysis",
        "methodology": "Value defer/expand/abandon flexibility embedded in strategic initiatives.",
        "visualization": "Option decision tree + value uplift table",
    },
    {
        "key": "var_cvar",
        "title": "VaR & Conditional VaR",
        "methodology": "Estimate downside tail risk at configurable confidence levels.",
        "visualization": "Loss distribution + tail expectation chart",
    },
    {
        "key": "copula_models",
        "title": "Copula Models",
        "methodology": "Model joint tail dependencies across multiple risk factors.",
        "visualization": "Dependence heatmap + tail copula diagnostics",
    },
    {
        "key": "macroeconomic_linking",
        "title": "Macroeconomic Linking",
        "methodology": "Link inflation/GDP/rates/FX assumptions into model drivers for macro-consistent projections.",
        "visualization": "Macro-to-driver linkage table",
    },
    {
        "key": "esg_sustainability",
        "title": "ESG & Sustainability Metrics",
        "methodology": "Quantify financial impact of emissions, carbon pricing, and renewable adoption pathways.",
        "visualization": "ESG scorecard + cost-benefit trend",
    },
    {
        "key": "market_intelligence",
        "title": "Market Intelligence Integration",
        "methodology": "Blend external sentiment and industry outlook data into dynamic demand forecasts.",
        "visualization": "Sentiment index + demand response chart",
    },
    {
        "key": "probabilistic_valuation",
        "title": "Probabilistic Valuation",
        "methodology": "Produce valuation ranges and confidence intervals instead of single-point estimates.",
        "visualization": "Valuation percentile chart",
    },
    {
        "key": "comparative_valuation_clustering",
        "title": "Comparative Valuation with Clustering",
        "methodology": "Benchmark against statistically similar peers using clustering and relative multiples.",
        "visualization": "Peer cluster map + valuation spread",
    },
    {
        "key": "ml_based_valuation",
        "title": "Machine Learning–Based Valuation",
        "methodology": "Predict fair value/multiples from historical market and operational feature sets.",
        "visualization": "Feature importance + predicted range chart",
    },
]


def _default_framework_table(
    columns: Sequence[str], rows: Sequence[Sequence[Any]]
) -> pd.DataFrame:
    return pd.DataFrame(list(rows), columns=list(columns))


def _analytics_framework_store() -> Dict[str, Dict[str, Any]]:
    store = st.session_state.setdefault("analytics_framework", {})
    for tool in ANALYTICS_FRAMEWORK_TOOLS:
        key = tool["key"]
        if key in store:
            continue
        store[key] = {
            "enabled": False,
            "data_sources": ["Scenario Output"],
            "inputs": _default_framework_table(
                ["Input", "Source", "Transform", "Active"],
                [
                    ("Milk Price Series", "Scenario Output", "none", True),
                    ("Feed Cost Series", "Scenario Output", "none", True),
                    ("Herd Productivity", "Input Schedule", "rolling_mean", True),
                ],
            ),
            "assumptions": _default_framework_table(
                ["Assumption", "Value", "Units"],
                [("Confidence Level", 95.0, "%"), ("Lookback Window", 12.0, "months")],
            ),
            "drivers": _default_framework_table(
                ["Driver", "Base", "Low", "High"],
                [
                    ("Milk Price", 1.0, -20.0, 20.0),
                    ("Feed Cost", 1.0, -20.0, 20.0),
                    ("Herd Productivity", 1.0, -15.0, 15.0),
                ],
            ),
            "scenarios": _default_framework_table(
                ["Scenario", "Shock %", "Probability %", "Active"],
                [
                    ("Base", 0.0, 60.0, True),
                    ("Downside", -15.0, 25.0, True),
                    ("Severe Stress", -35.0, 15.0, False),
                ],
            ),
        }
    st.session_state["analytics_framework"] = store
    return store


def _numeric_column_mean(df: pd.DataFrame, column: str) -> float:
    if column not in df.columns:
        return 0.0
    numeric = pd.to_numeric(df[column], errors="coerce")
    if numeric.dropna().empty:
        return 0.0
    return float(numeric.mean())


def _analytics_framework_output(
    tool_config: Dict[str, Any], results: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    assumptions = tool_config.get("assumptions", pd.DataFrame())
    drivers = tool_config.get("drivers", pd.DataFrame())
    scenarios = tool_config.get("scenarios", pd.DataFrame())

    assumption_level = _numeric_column_mean(assumptions, "Value")
    driver_range = (
        abs(_numeric_column_mean(drivers, "High"))
        + abs(_numeric_column_mean(drivers, "Low"))
    ) / 2.0
    avg_shock = _numeric_column_mean(scenarios, "Shock %")
    stress_probability = _numeric_column_mean(scenarios, "Probability %")

    base_npv = np.nan
    base_irr = np.nan
    if results is not None and isinstance(results.get("kpis"), pd.DataFrame):
        kpi_df = results["kpis"]
        if "NPV" in kpi_df.columns:
            base_npv = float(kpi_df["NPV"].iloc[0])
        if "IRR" in kpi_df.columns:
            base_irr = float(kpi_df["IRR"].iloc[0])

    impact_score = (driver_range * 0.4) + (assumption_level * 0.2) + (avg_shock * -0.3)
    resilience_score = max(0.0, 100.0 - abs(avg_shock) - (stress_probability * 0.2))

    return pd.DataFrame(
        {
            "Metric": [
                "Configured Data Sources",
                "Average Assumption Level",
                "Average Driver Stress Range",
                "Average Scenario Shock",
                "Resilience Score",
                "Indicative Impact Score",
                "Reference NPV",
                "Reference IRR",
            ],
            "Value": [
                len(tool_config.get("data_sources", [])),
                round(assumption_level, 2),
                round(driver_range, 2),
                round(avg_shock, 2),
                round(resilience_score, 2),
                round(impact_score, 2),
                round(base_npv, 2) if pd.notna(base_npv) else np.nan,
                round(base_irr, 4) if pd.notna(base_irr) else np.nan,
            ],
        }
    )


def _render_analytics_framework(results: Optional[Dict[str, Any]]) -> None:
    st.markdown("### Editable Analytics Schedule Framework")
    st.caption(
        "Each analytical capability has editable inputs, assumptions, model drivers, and "
        "scenario settings. Outputs refresh automatically on every change."
    )
    framework = _analytics_framework_store()
    linked_sources = [
        "Input Schedule",
        "Assumptions",
        "Scenario Output",
        "Financial Statements",
        "Dashboard KPIs",
        "Supplementary Schedules",
        "External Market Data",
        "ESG Data",
        "Peer Benchmark Data",
    ]

    for tool in ANALYTICS_FRAMEWORK_TOOLS:
        tool_key = tool["key"]
        config = framework[tool_key]
        with st.expander(tool["title"], expanded=False):
            left, right = st.columns([1, 2])
            config["enabled"] = left.checkbox(
                "Enable tool",
                value=bool(config.get("enabled", False)),
                key=f"framework_enabled::{tool_key}",
            )
            selected_sources = right.multiselect(
                "Linked data inputs",
                options=linked_sources,
                default=config.get("data_sources", ["Scenario Output"]),
                key=f"framework_sources::{tool_key}",
                help="Choose datasets this tool should consume.",
            )
            config["data_sources"] = selected_sources

            st.markdown("**Methodology**")
            st.write(tool["methodology"])

            st.markdown("**Configurable Input Mapping**")
            config["inputs"] = st.data_editor(
                config.get("inputs", pd.DataFrame()),
                num_rows="dynamic",
                use_container_width=True,
                key=f"framework_inputs::{tool_key}",
            )

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Adjustable Assumptions**")
                config["assumptions"] = st.data_editor(
                    config.get("assumptions", pd.DataFrame()),
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"framework_assumptions::{tool_key}",
                )
            with c2:
                st.markdown("**Model Drivers & Ranges**")
                config["drivers"] = st.data_editor(
                    config.get("drivers", pd.DataFrame()),
                    num_rows="dynamic",
                    use_container_width=True,
                    key=f"framework_drivers::{tool_key}",
                )

            st.markdown("**Scenario Settings**")
            config["scenarios"] = st.data_editor(
                config.get("scenarios", pd.DataFrame()),
                num_rows="dynamic",
                use_container_width=True,
                key=f"framework_scenarios::{tool_key}",
            )

            output_df = _analytics_framework_output(config, results)
            st.markdown("**Dynamic Outputs**")
            st.dataframe(output_df, use_container_width=True)
            chart_df = output_df.set_index("Metric")
            if "Value" in chart_df.columns:
                st.bar_chart(chart_df[["Value"]])
            st.caption(f"Suggested visualisation: {tool['visualization']}")

            framework[tool_key] = config

    st.session_state["analytics_framework"] = framework



def main() -> None:
    st.title("🐐 Goat Farm Financial Model — Interactive Scenario Dashboard")

    if "schedule" in st.session_state:
        st.session_state.pop("schedule")

    if DEFAULT_INPUT_CONFIG_KEY not in st.session_state:
        st.session_state[DEFAULT_INPUT_CONFIG_KEY] = _default_input_template_config()

    if "assumptions" not in st.session_state:
        st.session_state.assumptions = _default_assumption_tables()
    else:
        defaults = _default_assumption_tables()
        for name, table in defaults.items():
            st.session_state.assumptions.setdefault(name, table.copy())

    production_horizon_defaults = st.session_state.assumptions.get("Production Horizon")

    if "core_schedule" not in st.session_state or "detail_schedules" not in st.session_state:
        core_default, detail_defaults = _default_schedule_components(
            production_horizon=production_horizon_defaults
        )
        if "core_schedule" not in st.session_state:
            st.session_state.core_schedule = core_default
        if "detail_schedules" not in st.session_state:
            st.session_state.detail_schedules = detail_defaults
    if "supplementary" not in st.session_state:
        st.session_state.supplementary = _default_supplementary_tables()
    if "all_scenario_results" not in st.session_state:
        st.session_state.all_scenario_results = {}
    if "selected_scenario_name" not in st.session_state:
        st.session_state.selected_scenario_name = next(iter(SCENARIO_PRESETS))
    if "results" not in st.session_state:
        st.session_state.results = None

    _ensure_default_results_loaded()

    _ensure_active_scenario_selection()

    excel_download_container = st.container()

    ai_payload = st.session_state.setdefault("ai_payload", {})

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
            "AI Decision Making",
            "Assumptions",
            "Financials",
            "Dashboard",
            "Advanced Analytics",
        ]
    )

    with tabs[0]:
        st.subheader("Input Schedule")
        st.markdown("### Scenario Explorer")
        _render_model_author_editor()
        _render_scenario_selector()
        _render_scenario_preset_editors()

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
            core_table = st.session_state.get("core_schedule")
            if not isinstance(core_table, pd.DataFrame):
                core_table = pd.DataFrame()
                st.session_state.core_schedule = core_table

            _render_schedule_row_editor(
                "core_schedule",
                st.session_state.core_schedule,
                lambda updated: st.session_state.__setitem__(
                    "core_schedule", updated
                ),
            )
            core_editor = st.session_state.core_schedule
            st.markdown("### Supplementary Tables")
            for name in list(st.session_state.supplementary.keys()):
                if name == "Capitalisation Table":
                    st.markdown("#### Capitalisation Table Schedule")
                    cap_table = _ensure_capitalisation_table(
                        st.session_state.supplementary.get(name)
                    )
                    st.session_state.supplementary[name] = cap_table

                    add_col, remove_select_col, remove_btn_col, inc_col_col, inc_pct_col = st.columns(
                        [1, 2, 1, 2, 2]
                    )

                    with add_col:
                        if st.button("Add Row", key="cap_table_add_row"):
                            cap_table = _add_capitalisation_row(cap_table)
                            st.session_state.supplementary[name] = cap_table
                            _clear_schedule_editor_state("supp::capitalisation_table")

                    option_labels = []
                    option_map: Dict[str, int] = {}
                    for idx, row in cap_table.iterrows():
                        year = row.get("Year")
                        shareholder = row.get("Shareholder") or "Unnamed"
                        if pd.isna(year):
                            label = f"{shareholder}"
                        else:
                            label = f"{int(year)} – {shareholder}"
                        option_labels.append(label)
                        option_map[label] = idx

                    remove_choice = None
                    with remove_select_col:
                        if option_labels:
                            remove_choice = st.selectbox(
                                "Select row",
                                options=["-- Select Row --"] + option_labels,
                                key="cap_table_remove_choice",
                            )
                        else:
                            st.write("No rows available to remove.")

                    with remove_btn_col:
                        if (
                            st.button("Remove Row", key="cap_table_remove_button")
                            and remove_choice
                            and remove_choice in option_map
                        ):
                            cap_table = _remove_capitalisation_row(
                                cap_table, option_map[remove_choice]
                            )
                            st.session_state.supplementary[name] = cap_table
                            _clear_schedule_editor_state("supp::capitalisation_table")

                    with inc_col_col:
                        increment_column = st.selectbox(
                            "Increment column",
                            options=["Ownership %", "Investment"],
                            key="cap_table_increment_column",
                        )

                    with inc_pct_col:
                        increment_pct = st.number_input(
                            "Yearly increment (%)",
                            value=0.0,
                            step=0.5,
                            key="cap_table_increment_pct",
                        )
                        if st.button("Apply", key="cap_table_increment_button"):
                            cap_table = _apply_capitalisation_increment(
                                cap_table, increment_column, increment_pct
                            )

                            st.session_state.supplementary[name] = cap_table
                            _clear_schedule_editor_state("supp::capitalisation_table")

                    cap_table = st.session_state.supplementary[name]

                    def _save_capitalisation(updated: pd.DataFrame) -> None:
                        ensured = _ensure_capitalisation_table(updated)
                        st.session_state.supplementary[name] = ensured

                    _render_schedule_row_editor(
                        "supp::capitalisation_table", cap_table, _save_capitalisation
                    )

                    cleaned_cap = _clean_editor_table(
                        st.session_state.supplementary[name]
                    )
                    if cleaned_cap is not None:
                        supplementary_tables[name] = _ensure_capitalisation_table(cleaned_cap)
                    continue
                if name == "Capex Schedule":
                    st.markdown("#### Capex Schedule")
                    capex_table = _ensure_capex_schedule(
                        st.session_state.supplementary.get(name)
                    )
                    st.session_state.supplementary[name] = capex_table

                    rate_series = pd.to_numeric(
                        capex_table.get("Depreciation Rate %"), errors="coerce"
                    ).dropna()
                    spend_series = pd.to_numeric(
                        capex_table.get("Spend"), errors="coerce"
                    ).dropna()

                    default_rate = float(rate_series.iloc[0]) if not rate_series.empty else 10.0
                    default_spend = float(spend_series.iloc[0]) if not spend_series.empty else 0.0

                    st.session_state.setdefault("capex_rate_default", round(default_rate, 2))
                    st.session_state.setdefault("capex_default_spend", default_spend)
                    st.session_state.setdefault("capex_remove_choice", "-- Select Row --")
                    st.session_state.setdefault("capex_increment_column", "Spend")
                    st.session_state.setdefault("capex_increment_pct", 0.0)

                    rate_col, rate_btn_col, spend_col = st.columns([1, 1, 1])

                    rate_value = rate_col.number_input(
                        "Default depreciation rate (%)",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        key="capex_rate_default",
                    )
                    if rate_btn_col.button("Apply rate to all", key="capex_apply_rate"):
                        capex_table = _apply_capex_rate(capex_table, rate_value)
                        st.session_state.supplementary[name] = capex_table
                        _clear_schedule_editor_state("supp::capex_schedule")

                    spend_default = spend_col.number_input(
                        "Default spend for new row",
                        min_value=0.0,
                        step=100.0,
                        key="capex_default_spend",
                    )

                    add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])

                    if add_col.button("Add Row", key="capex_add_row"):
                        capex_table = _add_capex_row(
                            capex_table,
                            default_rate=rate_value,
                            default_spend=spend_default,
                        )
                        st.session_state.supplementary[name] = capex_table
                        _clear_schedule_editor_state("supp::capex_schedule")

                    option_labels: list[str] = []
                    option_map: Dict[str, int] = {}
                    for idx, row in capex_table.iterrows():
                        label_year = row.get("Year")
                        label_category = row.get("Category") or "Unnamed"
                        if pd.notna(label_year):
                            label = f"{int(label_year)} – {label_category}"
                        else:
                            label = str(label_category)
                        option_labels.append(label)
                        option_map[label] = idx

                    remove_choice = remove_select_col.selectbox(
                        "Select row",
                        options=["-- Select Row --"] + option_labels,
                        key="capex_remove_choice",
                    )
                    if remove_btn_col.button("Remove Row", key="capex_remove_row"):
                        if remove_choice in option_map:
                            capex_table = _remove_capex_row(
                                capex_table, option_map[remove_choice]
                            )
                            st.session_state.capex_remove_choice = "-- Select Row --"
                            st.session_state.supplementary[name] = capex_table
                            _clear_schedule_editor_state("supp::capex_schedule")

                    inc_col, inc_pct_col, inc_btn_col = st.columns([1.5, 1, 1])

                    increment_column = inc_col.selectbox(
                        "Increment column",
                        options=["Spend", "Depreciation Rate %"],
                        key="capex_increment_column",
                    )
                    increment_pct = inc_pct_col.number_input(
                        "Yearly increment (%)",
                        min_value=-100.0,
                        max_value=100.0,
                        step=0.1,
                        key="capex_increment_pct",
                    )
                    if inc_btn_col.button("Apply increment", key="capex_apply_increment"):
                        capex_table = _apply_capex_yearly_increment(
                            capex_table, increment_column, increment_pct
                        )
                        st.session_state.supplementary[name] = capex_table
                        _clear_schedule_editor_state("supp::capex_schedule")

                    capex_table = st.session_state.supplementary[name]

                    def _save_capex(updated: pd.DataFrame) -> None:
                        ensured = _ensure_capex_schedule(updated)
                        st.session_state.supplementary[name] = ensured

                    _render_schedule_row_editor(
                        "supp::capex_schedule", capex_table, _save_capex
                    )

                    cleaned_capex = _clean_editor_table(
                        st.session_state.supplementary[name]
                    )
                    if cleaned_capex is not None:
                        supplementary_tables[name] = _ensure_capex_schedule(cleaned_capex)
                    continue
                if name == "Asset Schedules":
                    st.markdown("#### Asset Schedule")
                    asset_table = _ensure_asset_schedule(
                        st.session_state.supplementary.get(name)
                    )
                    st.session_state.supplementary[name] = asset_table

                    rate_series = pd.to_numeric(
                        asset_table.get("Depreciation Rate %"), errors="coerce"
                    ).dropna()
                    addition_series = pd.to_numeric(
                        asset_table.get("Additions"), errors="coerce"
                    ).dropna()

                    default_rate = float(rate_series.iloc[0]) if not rate_series.empty else 5.0
                    default_addition = (
                        float(addition_series.iloc[0]) if not addition_series.empty else 0.0
                    )

                    st.session_state.setdefault("asset_rate_default", round(default_rate, 2))
                    st.session_state.setdefault("asset_add_base", default_addition)
                    st.session_state.setdefault("asset_add_increment", 0.0)
                    st.session_state.setdefault("asset_increment_column", "Depreciation Rate %")
                    st.session_state.setdefault("asset_increment_pct", 0.0)
                    st.session_state.setdefault("asset_remove_choice", "-- Select Row --")

                    rate_col, rate_btn_col, add_base_col, add_inc_col, add_btn_col = st.columns(
                        [1, 1, 1, 1, 1]
                    )

                    rate_value = rate_col.number_input(
                        "Default depreciation rate (%)",
                        min_value=0.0,
                        max_value=100.0,
                        step=0.1,
                        key="asset_rate_default",
                    )
                    if rate_btn_col.button("Apply rate to all", key="asset_apply_rate"):
                        asset_table = _apply_asset_rate(asset_table, rate_value)
                        st.session_state.supplementary[name] = asset_table
                        _clear_schedule_editor_state("supp::asset_schedule")

                    base_add_value = add_base_col.number_input(
                        "Base additions",
                        min_value=0.0,
                        step=100.0,
                        key="asset_add_base",
                    )
                    add_increment = add_inc_col.number_input(
                        "Additions yearly increment (%)",
                        min_value=-100.0,
                        max_value=100.0,
                        step=0.1,
                        key="asset_add_increment",
                    )
                    if add_btn_col.button(
                        "Apply additions pattern", key="asset_apply_additions"
                    ):
                        asset_table = _apply_asset_additions_pattern(
                            asset_table, base_add_value, add_increment
                        )
                        st.session_state.supplementary[name] = asset_table
                        _clear_schedule_editor_state("supp::asset_schedule")

                    (
                        add_row_col,
                        remove_select_col,
                        remove_btn_col,
                        inc_column_col,
                        inc_pct_col,
                        inc_btn_col,
                    ) = st.columns([1, 2, 1, 1.5, 1, 1])

                    if add_row_col.button("Add Asset", key="asset_add_row"):
                        asset_table = _add_asset_row(
                            asset_table,
                            default_rate=rate_value,
                            default_additions=base_add_value,
                        )
                        st.session_state.supplementary[name] = asset_table
                        _clear_schedule_editor_state("supp::asset_schedule")

                    option_labels: list[str] = []
                    option_map: Dict[str, int] = {}
                    for idx, row in asset_table.iterrows():
                        label_year = row.get("Year")
                        label_asset = row.get("Asset") or "Unnamed"
                        if pd.notna(label_year):
                            label = f"{int(label_year)} – {label_asset}"
                        else:
                            label = str(label_asset)
                        option_labels.append(label)
                        option_map[label] = idx

                    remove_choice = remove_select_col.selectbox(
                        "Select asset",
                        options=["-- Select Row --"] + option_labels,
                        key="asset_remove_choice",
                    )
                    if remove_btn_col.button("Remove Asset", key="asset_remove_row"):
                        if remove_choice in option_map:
                            asset_table = _remove_asset_row(
                                asset_table, option_map[remove_choice]
                            )
                            st.session_state.asset_remove_choice = "-- Select Row --"
                            st.session_state.supplementary[name] = asset_table
                            _clear_schedule_editor_state("supp::asset_schedule")

                    increment_column = inc_column_col.selectbox(
                        "Increment column",
                        options=[
                            "Depreciation Rate %",
                            "Additions",
                            "Opening NBV",
                            "Depreciation",
                        ],
                        key="asset_increment_column",
                    )
                    increment_pct = inc_pct_col.number_input(
                        "Yearly increment (%)",
                        min_value=-100.0,
                        max_value=100.0,
                        step=0.1,
                        key="asset_increment_pct",
                    )
                    if inc_btn_col.button("Apply increment", key="asset_apply_increment"):
                        asset_table = _apply_asset_yearly_increment(
                            asset_table, increment_column, increment_pct
                        )
                        st.session_state.supplementary[name] = asset_table
                        _clear_schedule_editor_state("supp::asset_schedule")

                    asset_table = st.session_state.supplementary[name]

                    def _save_asset(updated: pd.DataFrame) -> None:
                        ensured = _ensure_asset_schedule(updated)
                        st.session_state.supplementary[name] = ensured

                    _render_schedule_row_editor(
                        "supp::asset_schedule", asset_table, _save_asset
                    )

                    cleaned_asset = _clean_editor_table(
                        st.session_state.supplementary[name]
                    )
                    if cleaned_asset is not None:
                        supplementary_tables[name] = _ensure_asset_schedule(cleaned_asset)
                    continue

                with st.expander(name, expanded=False):
                    table = st.session_state.supplementary.get(name, pd.DataFrame())
                    if not isinstance(table, pd.DataFrame):
                        table = pd.DataFrame()
                        st.session_state.supplementary[name] = table

                    def _save_generic(updated: pd.DataFrame, table_name: str = name) -> None:
                        st.session_state.supplementary[table_name] = updated

                    _render_schedule_row_editor(
                        f"supp::{_scenario_key_suffix(name)}",
                        table,
                        _save_generic,
                    )

                cleaned = _clean_editor_table(
                    st.session_state.supplementary.get(name, pd.DataFrame())
                )
                if cleaned is not None:
                    supplementary_tables[name] = cleaned
                else:
                    st.session_state.supplementary[name] = (
                        st.session_state.supplementary.get(name, pd.DataFrame())
                    )

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
                    st.session_state.detail_schedules[name] = cogs_table

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
                        _clear_schedule_editor_state("detail::cogs_schedule")

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
                        _clear_schedule_editor_state("detail::cogs_schedule")

                    if controls[2].button("Add Row", key="cogs_add_row"):
                        cogs_table = _add_cogs_row(
                            cogs_table,
                            st.session_state.core_schedule,
                            default_pct=pct_input,
                        )
                        st.session_state.detail_schedules[name] = cogs_table
                        _clear_schedule_editor_state("detail::cogs_schedule")

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
                            _clear_schedule_editor_state("detail::cogs_schedule")

                    cogs_table = _sync_cogs_table(
                        cogs_table, st.session_state.core_schedule, default_pct=pct_input
                    )
                    st.session_state.detail_schedules[name] = cogs_table
                    def _save_cogs(updated: pd.DataFrame) -> None:
                        ensured = _ensure_cogs_schedule(
                            updated,
                            st.session_state.core_schedule,
                            default_pct=pct_input,
                        )
                        st.session_state.detail_schedules[name] = ensured

                    _render_schedule_row_editor(
                        "detail::cogs_schedule",
                        st.session_state.detail_schedules[name],
                        _save_cogs,
                    )

                    detail_tables_for_run[name] = st.session_state.detail_schedules[name]
                elif name == "Variable Expenses Schedule":
                    st.markdown("#### Variable Expenses Schedule")
                    st.caption(
                        "Manage individual variable cost items, add or remove rows, and apply yearly increments "
                        "to quickly escalate recurring expenses. Amounts roll up into the income statement automatically."
                    )

                    variable_table = _ensure_variable_expense_table(
                        st.session_state.detail_schedules.get(name, pd.DataFrame()),
                        st.session_state.core_schedule,
                    )
                    st.session_state.detail_schedules[name] = variable_table

                    st.session_state.setdefault("var_exp_remove_choice", "-- Select Row --")
                    st.session_state.setdefault("var_exp_increment_target", "All items")
                    st.session_state.setdefault("var_exp_increment_pct", 0.0)

                    add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])

                    if add_col.button("Add Row", key="var_exp_add_row"):
                        variable_table = _add_variable_expense_row(
                            variable_table, st.session_state.core_schedule
                        )
                        st.session_state.detail_schedules[name] = variable_table
                        _clear_schedule_editor_state("detail::variable_expenses")

                    option_labels: list[str] = []
                    option_index: Dict[str, int] = {}
                    for idx_row, row in variable_table.iterrows():
                        label_period = row.get("Period") or "Unknown Period"
                        label_item = row.get("Item") or "Item"
                        label = f"{label_period} – {label_item}"
                        option_labels.append(label)
                        option_index[label] = idx_row

                    remove_select_col.selectbox(
                        "Select row", options=["-- Select Row --"] + option_labels, key="var_exp_remove_choice"
                    )
                    if remove_btn_col.button("Remove Row", key="var_exp_remove_row"):
                        choice = st.session_state.get("var_exp_remove_choice")
                        if choice in option_index:
                            variable_table = _remove_variable_expense_row(
                                variable_table, option_index[choice]
                            )
                            st.session_state.detail_schedules[name] = variable_table
                            st.session_state.var_exp_remove_choice = "-- Select Row --"
                            _clear_schedule_editor_state("detail::variable_expenses")

                    inc_target_col, inc_pct_col, inc_btn_col = st.columns([2, 1, 1])
                    target_options = ["All items"] + sorted(
                        {
                            str(item)
                            for item in variable_table.get("Item", pd.Series(dtype=str)).dropna().unique().tolist()
                            if str(item).strip()
                        }
                    )
                    inc_target_col.selectbox(
                        "Apply increment to", options=target_options, key="var_exp_increment_target"
                    )
                    inc_pct_col.number_input(
                        "Yearly increment (%)",
                        min_value=-100.0,
                        max_value=100.0,
                        step=0.1,
                        key="var_exp_increment_pct",
                    )
                    if inc_btn_col.button("Apply increment", key="var_exp_apply_increment"):
                        variable_table = _apply_variable_expense_increment(
                            variable_table,
                            st.session_state.get("var_exp_increment_pct", 0.0),
                            st.session_state.get("var_exp_increment_target"),
                        )
                        st.session_state.detail_schedules[name] = variable_table
                        _clear_schedule_editor_state("detail::variable_expenses")

                    def _save_variable(updated: pd.DataFrame) -> None:
                        ensured = _ensure_variable_expense_table(
                            updated, st.session_state.core_schedule
                        )
                        st.session_state.detail_schedules[name] = ensured

                    _render_schedule_row_editor(
                        "detail::variable_expenses",
                        st.session_state.detail_schedules[name],
                        _save_variable,
                    )

                    variable_table = st.session_state.detail_schedules[name]

                    st.session_state.setdefault(
                        "variable_defaults_edit_mode", False
                    )
                    toggle_label = (
                        "Hide default variable expense template"
                        if st.session_state.variable_defaults_edit_mode
                        else "Edit default variable expense template"
                    )
                    if st.button(toggle_label, key="toggle_variable_defaults"):
                        st.session_state.variable_defaults_edit_mode = not st.session_state[
                            "variable_defaults_edit_mode"
                        ]

                    if st.session_state.variable_defaults_edit_mode:
                        st.markdown("##### Default Variable Expense Template")
                        st.caption(
                            "Update the baseline mix of variable expenses that populates new schedules."
                        )

                        variable_columns = ["Item", "Share %"]
                        default_frame = st.session_state.get(
                            "default_variable_items_editor"
                        )
                        if not isinstance(default_frame, pd.DataFrame):
                            default_frame = _template_to_dataframe(
                                _get_template("variable_items", DEFAULT_VARIABLE_ITEMS),
                                variable_columns,
                            )

                        template_editor = st.data_editor(
                            default_frame,
                            num_rows="dynamic",
                            use_container_width=True,
                            key="default_variable_items_editor",
                            column_config={
                                "Share %": st.column_config.NumberColumn(
                                    "Share (%)", format="%.2f", step=0.1
                                )
                            },
                        )

                        button_col, save_col, restore_col, apply_col = st.columns(4)

                        if button_col.button(
                            "Close editor", key="close_variable_defaults"
                        ):
                            st.session_state.variable_defaults_edit_mode = False

                        if save_col.button(
                            "Save defaults", key="save_variable_defaults"
                        ):
                            records = _dataframe_to_template(
                                template_editor, variable_columns
                            )
                            _set_template("variable_items", records)
                            st.success("Variable expense defaults updated.")

                        if restore_col.button(
                            "Restore baseline", key="reset_variable_defaults"
                        ):
                            baseline_template = _template_copy(DEFAULT_VARIABLE_ITEMS)
                            _set_template("variable_items", baseline_template)
                            variable_table = _default_variable_expense_table(
                                st.session_state.core_schedule
                            )
                            st.session_state.detail_schedules[name] = variable_table
                            st.session_state["default_variable_items_editor"] = _template_to_dataframe(
                                baseline_template, variable_columns
                            )
                            st.success(
                                "Variable expense defaults restored and schedule refreshed."
                            )
                            _clear_schedule_editor_state("detail::variable_expenses")

                        if apply_col.button(
                            "Apply to schedule", key="apply_variable_defaults"
                        ):
                            records = _dataframe_to_template(
                                template_editor, variable_columns
                            )
                            _set_template("variable_items", records)
                            variable_table = _default_variable_expense_table(
                                st.session_state.core_schedule
                            )
                            st.session_state.detail_schedules[name] = variable_table
                            st.session_state["default_variable_items_editor"] = template_editor
                            st.success(
                                "Variable expenses schedule regenerated from defaults."
                            )
                            _clear_schedule_editor_state("detail::variable_expenses")

                    aggregated_variable = _aggregate_variable_expenses(
                        variable_table, st.session_state.core_schedule
                    )
                    detail_tables_for_run[name] = aggregated_variable

                    st.markdown("##### Variable Expenses Summary")
                    st.dataframe(aggregated_variable)
                elif name == "Direct Wages Schedule":
                    st.markdown("#### Direct Wages Schedule")
                    st.caption(
                        "Capture individual direct labour cost items, manage rows, and escalate pay levels with yearly "
                        "increments. Totals automatically feed into the model's EBITDA calculations."
                    )

                    direct_table = _ensure_direct_wage_table(
                        st.session_state.detail_schedules.get(name, pd.DataFrame()),
                        st.session_state.core_schedule,
                    )
                    st.session_state.detail_schedules[name] = direct_table

                    st.session_state.setdefault("direct_wage_remove_choice", "-- Select Row --")
                    st.session_state.setdefault("direct_wage_increment_target", "All roles")
                    st.session_state.setdefault("direct_wage_increment_pct", 0.0)

                    add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])

                    if add_col.button("Add Row", key="direct_wage_add_row"):
                        direct_table = _add_direct_wage_row(
                            direct_table, st.session_state.core_schedule
                        )
                        st.session_state.detail_schedules[name] = direct_table
                        _clear_schedule_editor_state("detail::direct_wages")

                    option_labels: list[str] = []
                    option_index: Dict[str, int] = {}
                    for idx_row, row in direct_table.iterrows():
                        label_period = row.get("Period") or "Unknown Period"
                        label_role = row.get("Role") or "Role"
                        label = f"{label_period} – {label_role}"
                        option_labels.append(label)
                        option_index[label] = idx_row

                    remove_select_col.selectbox(
                        "Select row",
                        options=["-- Select Row --"] + option_labels,
                        key="direct_wage_remove_choice",
                    )
                    if remove_btn_col.button("Remove Row", key="direct_wage_remove_row"):
                        choice = st.session_state.get("direct_wage_remove_choice")
                        if choice in option_index:
                            direct_table = _remove_direct_wage_row(
                                direct_table, option_index[choice]
                            )
                            st.session_state.detail_schedules[name] = direct_table
                            st.session_state.direct_wage_remove_choice = "-- Select Row --"
                            _clear_schedule_editor_state("detail::direct_wages")

                    inc_target_col, inc_pct_col, inc_btn_col = st.columns([2, 1, 1])
                    target_options = ["All roles"] + sorted(
                        {
                            str(role)
                            for role in direct_table.get("Role", pd.Series(dtype=str))
                            .dropna()
                            .unique()
                            .tolist()
                            if str(role).strip()
                        }
                    )
                    inc_target_col.selectbox(
                        "Apply increment to",
                        options=target_options,
                        key="direct_wage_increment_target",
                    )
                    inc_pct_col.number_input(
                        "Yearly increment (%)",
                        min_value=-100.0,
                        max_value=100.0,
                        step=0.1,
                        key="direct_wage_increment_pct",
                    )
                    if inc_btn_col.button("Apply increment", key="direct_wage_apply_increment"):
                        direct_table = _apply_direct_wage_increment(
                            direct_table,
                            st.session_state.get("direct_wage_increment_pct", 0.0),
                            st.session_state.get("direct_wage_increment_target"),
                        )
                        st.session_state.detail_schedules[name] = direct_table
                        _clear_schedule_editor_state("detail::direct_wages")

                    def _save_direct(updated: pd.DataFrame) -> None:
                        ensured = _ensure_direct_wage_table(
                            updated, st.session_state.core_schedule
                        )
                        st.session_state.detail_schedules[name] = ensured

                    _render_schedule_row_editor(
                        "detail::direct_wages",
                        st.session_state.detail_schedules[name],
                        _save_direct,
                    )

                    direct_table = st.session_state.detail_schedules[name]

                    st.session_state.setdefault("direct_defaults_edit_mode", False)
                    toggle_label = (
                        "Hide default direct wage template"
                        if st.session_state.direct_defaults_edit_mode
                        else "Edit default direct wage template"
                    )
                    if st.button(toggle_label, key="toggle_direct_defaults"):
                        st.session_state.direct_defaults_edit_mode = not st.session_state[
                            "direct_defaults_edit_mode"
                        ]

                    if st.session_state.direct_defaults_edit_mode:
                        st.markdown("##### Default Direct Wage Template")
                        st.caption(
                            "Adjust the baseline allocation of direct labour that seeds future schedules."
                        )

                        direct_columns = ["Role", "Share %"]
                        default_frame = st.session_state.get(
                            "default_direct_wage_editor"
                        )
                        if not isinstance(default_frame, pd.DataFrame):
                            default_frame = _template_to_dataframe(
                                _get_template(
                                    "direct_wage_items", DEFAULT_DIRECT_WAGE_ITEMS
                                ),
                                direct_columns,
                            )

                        template_editor = st.data_editor(
                            default_frame,
                            num_rows="dynamic",
                            use_container_width=True,
                            key="default_direct_wage_editor",
                            column_config={
                                "Share %": st.column_config.NumberColumn(
                                    "Share (%)", format="%.2f", step=0.1
                                )
                            },
                        )

                        button_col, save_col, restore_col, apply_col = st.columns(4)

                        if button_col.button(
                            "Close editor", key="close_direct_defaults"
                        ):
                            st.session_state.direct_defaults_edit_mode = False

                        if save_col.button(
                            "Save defaults", key="save_direct_wage_defaults"
                        ):
                            records = _dataframe_to_template(
                                template_editor, direct_columns
                            )
                            _set_template("direct_wage_items", records)
                            st.success("Direct wage defaults updated.")

                        if restore_col.button(
                            "Restore baseline", key="reset_direct_wage_defaults"
                        ):
                            baseline_template = _template_copy(DEFAULT_DIRECT_WAGE_ITEMS)
                            _set_template("direct_wage_items", baseline_template)
                            direct_table = _default_direct_wage_table(
                                st.session_state.core_schedule
                            )
                            st.session_state.detail_schedules[name] = direct_table
                            st.session_state["default_direct_wage_editor"] = _template_to_dataframe(
                                baseline_template, direct_columns
                            )
                            st.success(
                                "Direct wage defaults restored and schedule refreshed."
                            )
                            _clear_schedule_editor_state("detail::direct_wages")

                        if apply_col.button(
                            "Apply to schedule", key="apply_direct_wage_defaults"
                        ):
                            records = _dataframe_to_template(
                                template_editor, direct_columns
                            )
                            _set_template("direct_wage_items", records)
                            direct_table = _default_direct_wage_table(
                                st.session_state.core_schedule
                            )
                            st.session_state.detail_schedules[name] = direct_table
                            st.session_state["default_direct_wage_editor"] = template_editor
                            st.success(
                                "Direct wages schedule regenerated from defaults."
                            )
                            _clear_schedule_editor_state("detail::direct_wages")

                    aggregated_direct = _aggregate_direct_wages(
                        direct_table, st.session_state.core_schedule
                    )
                    detail_tables_for_run[name] = aggregated_direct

                    st.markdown("##### Direct Wages Summary")
                    st.dataframe(aggregated_direct)
                elif name == "Admin Wages Schedule":
                    st.markdown("#### Admin Wages Schedule")
                    st.caption(
                        "Detail administrative wage items, add or remove roles, and apply yearly increments to keep overhead "
                        "assumptions in sync with your operating plan. Totals automatically roll into the income statement."
                    )

                    admin_table = _ensure_admin_wage_table(
                        st.session_state.detail_schedules.get(name, pd.DataFrame()),
                        st.session_state.core_schedule,
                    )
                    st.session_state.detail_schedules[name] = admin_table

                    st.session_state.setdefault("admin_wage_remove_choice", "-- Select Row --")
                    st.session_state.setdefault("admin_wage_increment_target", "All functions")
                    st.session_state.setdefault("admin_wage_increment_pct", 0.0)

                    add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])

                    if add_col.button("Add Row", key="admin_wage_add_row"):
                        admin_table = _add_admin_wage_row(
                            admin_table, st.session_state.core_schedule
                        )
                        st.session_state.detail_schedules[name] = admin_table
                        _clear_schedule_editor_state("detail::admin_wages")

                    option_labels: list[str] = []
                    option_index: Dict[str, int] = {}
                    for idx_row, row in admin_table.iterrows():
                        label_period = row.get("Period") or "Unknown Period"
                        label_role = row.get("Function") or "Function"
                        label = f"{label_period} – {label_role}"
                        option_labels.append(label)
                        option_index[label] = idx_row

                    remove_select_col.selectbox(
                        "Select row",
                        options=["-- Select Row --"] + option_labels,
                        key="admin_wage_remove_choice",
                    )
                    if remove_btn_col.button("Remove Row", key="admin_wage_remove_row"):
                        choice = st.session_state.get("admin_wage_remove_choice")
                        if choice in option_index:
                            admin_table = _remove_admin_wage_row(
                                admin_table, option_index[choice]
                            )
                            st.session_state.detail_schedules[name] = admin_table
                            st.session_state.admin_wage_remove_choice = "-- Select Row --"
                            _clear_schedule_editor_state("detail::admin_wages")

                    inc_target_col, inc_pct_col, inc_btn_col = st.columns([2, 1, 1])
                    target_options = ["All functions"] + sorted(
                        {
                            str(function)
                            for function in admin_table.get("Function", pd.Series(dtype=str))
                            .dropna()
                            .unique()
                            .tolist()
                            if str(function).strip()
                        }
                    )
                    inc_target_col.selectbox(
                        "Apply increment to",
                        options=target_options,
                        key="admin_wage_increment_target",
                    )
                    inc_pct_col.number_input(
                        "Yearly increment (%)",
                        min_value=-100.0,
                        max_value=100.0,
                        step=0.1,
                        key="admin_wage_increment_pct",
                    )
                    if inc_btn_col.button("Apply increment", key="admin_wage_apply_increment"):
                        admin_table = _apply_admin_wage_increment(
                            admin_table,
                            st.session_state.get("admin_wage_increment_pct", 0.0),
                            st.session_state.get("admin_wage_increment_target"),
                        )
                        st.session_state.detail_schedules[name] = admin_table
                        _clear_schedule_editor_state("detail::admin_wages")

                    def _save_admin(updated: pd.DataFrame) -> None:
                        ensured = _ensure_admin_wage_table(
                            updated, st.session_state.core_schedule
                        )
                        st.session_state.detail_schedules[name] = ensured

                    _render_schedule_row_editor(
                        "detail::admin_wages",
                        st.session_state.detail_schedules[name],
                        _save_admin,
                    )

                    admin_table = st.session_state.detail_schedules[name]

                    st.session_state.setdefault("admin_defaults_edit_mode", False)
                    toggle_label = (
                        "Hide default admin wage template"
                        if st.session_state.admin_defaults_edit_mode
                        else "Edit default admin wage template"
                    )
                    if st.button(toggle_label, key="toggle_admin_defaults"):
                        st.session_state.admin_defaults_edit_mode = not st.session_state[
                            "admin_defaults_edit_mode"
                        ]

                    if st.session_state.admin_defaults_edit_mode:
                        st.markdown("##### Default Admin Wage Template")
                        st.caption(
                            "Maintain the default administrative wage allocation used when rebuilding schedules."
                        )

                        admin_columns = ["Function", "Share %"]
                        default_frame = st.session_state.get("default_admin_wage_editor")
                        if not isinstance(default_frame, pd.DataFrame):
                            default_frame = _template_to_dataframe(
                                _get_template("admin_wage_items", DEFAULT_ADMIN_WAGE_ITEMS),
                                admin_columns,
                            )

                        template_editor = st.data_editor(
                            default_frame,
                            num_rows="dynamic",
                            use_container_width=True,
                            key="default_admin_wage_editor",
                            column_config={
                                "Share %": st.column_config.NumberColumn(
                                    "Share (%)", format="%.2f", step=0.1
                                )
                            },
                        )

                        button_col, save_col, restore_col, apply_col = st.columns(4)

                        if button_col.button(
                            "Close editor", key="close_admin_defaults"
                        ):
                            st.session_state.admin_defaults_edit_mode = False

                        if save_col.button(
                            "Save defaults", key="save_admin_wage_defaults"
                        ):
                            records = _dataframe_to_template(
                                template_editor, admin_columns
                            )
                            _set_template("admin_wage_items", records)
                            st.success("Admin wage defaults updated.")

                        if restore_col.button(
                            "Restore baseline", key="reset_admin_wage_defaults"
                        ):
                            baseline_template = _template_copy(DEFAULT_ADMIN_WAGE_ITEMS)
                            _set_template("admin_wage_items", baseline_template)
                            admin_table = _default_admin_wage_table(
                                st.session_state.core_schedule
                            )
                            st.session_state.detail_schedules[name] = admin_table
                            st.session_state["default_admin_wage_editor"] = _template_to_dataframe(
                                baseline_template, admin_columns
                            )
                            st.success(
                                "Admin wage defaults restored and schedule refreshed."
                            )
                            _clear_schedule_editor_state("detail::admin_wages")

                        if apply_col.button(
                            "Apply to schedule", key="apply_admin_wage_defaults"
                        ):
                            records = _dataframe_to_template(
                                template_editor, admin_columns
                            )
                            _set_template("admin_wage_items", records)
                            admin_table = _default_admin_wage_table(
                                st.session_state.core_schedule
                            )
                            st.session_state.detail_schedules[name] = admin_table
                            st.session_state["default_admin_wage_editor"] = template_editor
                            st.success(
                                "Admin wages schedule regenerated from defaults."
                            )
                            _clear_schedule_editor_state("detail::admin_wages")

                    aggregated_admin = _aggregate_admin_wages(
                        admin_table, st.session_state.core_schedule
                    )
                    detail_tables_for_run[name] = aggregated_admin

                    st.markdown("##### Admin Wages Summary")
                    st.dataframe(aggregated_admin)
                else:
                    table = st.session_state.detail_schedules.get(name, pd.DataFrame())
                    if not isinstance(table, pd.DataFrame):
                        table = pd.DataFrame()
                        st.session_state.detail_schedules[name] = table

                    def _save_other(updated: pd.DataFrame, schedule_name: str = name) -> None:
                        st.session_state.detail_schedules[schedule_name] = updated

                    _render_schedule_row_editor(
                        f"detail::{_scenario_key_suffix(name)}",
                        st.session_state.detail_schedules.get(name, table),
                        _save_other,
                    )
                    detail_tables_for_run[name] = st.session_state.detail_schedules.get(
                        name, table
                    )

    if core_editor is None:
        core_editor = st.session_state.core_schedule

    for name, table in st.session_state.detail_schedules.items():
        detail_tables_for_run.setdefault(name, table)

    assumption_tables: Dict[str, pd.DataFrame] = {}

    with tabs[1]:
        st.subheader("AI Decision Making")
        st.markdown("Configure AI provider, model, ML methods, and narrative outputs.")
        _render_ai_settings(ai_payload)

    with tabs[2]:
        st.subheader("Assumptions")
        st.caption(
            "All core model assumption categories are consolidated below on a single page."
        )

        st.markdown("#### Scenario Controls")
        scenario_table = _ensure_scenario_controls_table(
            st.session_state.assumptions.get("Scenario Controls")
        )
        st.session_state.assumptions["Scenario Controls"] = scenario_table

        def _save_scenario_controls(updated: pd.DataFrame) -> None:
            ensured = _ensure_scenario_controls_table(updated)
            st.session_state.assumptions["Scenario Controls"] = ensured

        _render_schedule_row_editor(
            "assump::scenario_controls",
            st.session_state.assumptions["Scenario Controls"],
            _save_scenario_controls,
        )

        control_values = _scenario_controls_value_map(
            st.session_state.assumptions["Scenario Controls"]
        )
        milk_default = control_values.get("Milk price change (%)", 0.0)
        feed_default = control_values.get("Feed cost change (%)", 0.0)

        milk_price = st.slider(
            "Milk price change (%)",
            min_value=-50,
            max_value=50,
            value=int(round(milk_default)),
            step=1,
        )
        if float(milk_price) != float(milk_default):
            updated_table = _update_scenario_control_value(
                st.session_state.assumptions["Scenario Controls"],
                "Milk price change (%)",
                float(milk_price),
            )
            st.session_state.assumptions["Scenario Controls"] = updated_table
            _clear_schedule_editor_state("assump::scenario_controls")

        feed_cost = st.slider(
            "Feed cost change (%)",
            min_value=-50,
            max_value=50,
            value=int(round(feed_default)),
            step=1,
        )
        if float(feed_cost) != float(feed_default):
            updated_table = _update_scenario_control_value(
                st.session_state.assumptions["Scenario Controls"],
                "Feed cost change (%)",
                float(feed_cost),
            )
            st.session_state.assumptions["Scenario Controls"] = updated_table
            _clear_schedule_editor_state("assump::scenario_controls")

        assumption_tables["Scenario Controls"] = st.session_state.assumptions[
            "Scenario Controls"
        ]
        run_clicked = st.button("Run Scenarios", type="primary")

        st.markdown("---")
        st.markdown("#### Production Time Horizon")
        production_table = _ensure_production_horizon_table(
            st.session_state.assumptions.get("Production Horizon")
        )
        st.session_state.assumptions["Production Horizon"] = production_table

        defaults = production_table.iloc[0]
        start_default = int(defaults.get("Start Year", 2024))
        end_default = int(defaults.get("End Year", start_default))

        st.session_state.setdefault("production_start_year", start_default)
        st.session_state.setdefault("production_end_year", end_default)

        year_options = _production_year_options(start_default, end_default)
        if st.session_state.production_start_year not in year_options:
            st.session_state.production_start_year = start_default

        start_col, end_col = st.columns(2)

        start_value = start_col.selectbox(
            "Start year",
            options=year_options,
            key="production_start_year",
        )

        valid_end_options = [year for year in year_options if year >= start_value]
        if not valid_end_options:
            valid_end_options = [start_value]

        if st.session_state.production_end_year not in valid_end_options:
            st.session_state.production_end_year = valid_end_options[0]

        end_value = end_col.selectbox(
            "End year",
            options=valid_end_options,
            key="production_end_year",
        )

        if end_value < start_value:
            st.session_state.production_end_year = start_value
            end_value = start_value

        if start_value != start_default or end_value != end_default:
            updated_table = pd.DataFrame({"Start Year": [start_value], "End Year": [end_value]})
            st.session_state.assumptions["Production Horizon"] = updated_table
            assumption_tables["Production Horizon"] = updated_table
            _sync_production_horizon(start_value, end_value)
            _ensure_default_results_loaded()
            _ensure_active_scenario_selection()
            _maybe_rerun()
        else:
            assumption_tables["Production Horizon"] = production_table

        st.markdown("---")
        st.markdown("#### Pricing Assumptions")
        pricing_table = _ensure_pricing_table(
            st.session_state.assumptions.get("Pricing", pd.DataFrame())
        )
        st.session_state.assumptions["Pricing"] = pricing_table

        st.session_state.setdefault("pricing_remove_choice", "-- Select Row --")
        st.session_state.setdefault("pricing_increment_target", "All products")
        st.session_state.setdefault("pricing_increment_column", "Base Price")
        st.session_state.setdefault("pricing_increment_pct", 0.0)

        add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])

        if add_col.button("Add Product", key="pricing_add_row"):
            pricing_table = _add_pricing_row(pricing_table)
            st.session_state.assumptions["Pricing"] = pricing_table
            _clear_schedule_editor_state("assump::pricing")

        option_labels: list[str] = []
        option_index: Dict[str, int] = {}
        for idx_row, row in pricing_table.iterrows():
            label_year = row.get("Year")
            label_product = row.get("Product") or "Product"
            if pd.notna(label_year):
                label = f"{int(label_year)} – {label_product}"
            else:
                label = str(label_product)
            option_labels.append(label)
            option_index[label] = idx_row

        remove_select_col.selectbox(
            "Select row",
            options=["-- Select Row --"] + option_labels,
            key="pricing_remove_choice",
        )

        if remove_btn_col.button("Remove Row", key="pricing_remove_row"):
            choice = st.session_state.get("pricing_remove_choice")
            if choice in option_index:
                pricing_table = _remove_pricing_row(
                    pricing_table, option_index[choice]
                )
                st.session_state.assumptions["Pricing"] = pricing_table
                st.session_state.pricing_remove_choice = "-- Select Row --"
                _clear_schedule_editor_state("assump::pricing")

        inc_target_col, inc_column_col, inc_pct_col, inc_btn_col = st.columns(
            [2, 1.5, 1, 1]
        )

        target_options = ["All products"] + sorted(
            {
                str(product)
                for product in pricing_table.get("Product", pd.Series(dtype=str))
                .dropna()
                .tolist()
                if str(product).strip()
            }
        )
        inc_target_col.selectbox(
            "Apply increment to",
            options=target_options,
            key="pricing_increment_target",
        )

        inc_column_col.selectbox(
            "Column",
            options=["Base Price", "Price Growth %"],
            key="pricing_increment_column",
        )

        inc_pct_col.number_input(
            "Yearly increment (%)",
            min_value=-100.0,
            max_value=100.0,
            step=0.1,
            key="pricing_increment_pct",
        )

        if inc_btn_col.button("Apply increment", key="pricing_apply_increment"):
            pricing_table = _apply_pricing_yearly_increment(
                pricing_table,
                st.session_state.get("pricing_increment_column", "Base Price"),
                st.session_state.get("pricing_increment_pct", 0.0),
                st.session_state.get("pricing_increment_target"),
            )
            st.session_state.assumptions["Pricing"] = pricing_table
            _clear_schedule_editor_state("assump::pricing")

        def _save_pricing(updated: pd.DataFrame) -> None:
            ensured = _ensure_pricing_table(updated)
            st.session_state.assumptions["Pricing"] = ensured

        _render_schedule_row_editor(
            "assump::pricing",
            st.session_state.assumptions["Pricing"],
            _save_pricing,
        )

        st.session_state.setdefault("pricing_defaults_edit_mode", False)
        toggle_label = (
            "Hide default pricing assumptions"
            if st.session_state.pricing_defaults_edit_mode
            else "Edit default pricing assumptions"
        )
        if st.button(toggle_label, key="toggle_pricing_defaults"):
            st.session_state.pricing_defaults_edit_mode = not st.session_state[
                "pricing_defaults_edit_mode"
            ]

        if st.session_state.pricing_defaults_edit_mode:
            st.markdown("##### Default Pricing Assumptions")
            st.caption(
                "Edit the baseline pricing table applied when resetting these assumptions."
            )

            pricing_columns = [
                "Year",
                "Product",
                "Unit",
                "Base Price",
                "Price Growth %",
            ]
            default_frame = st.session_state.get("default_pricing_editor")
            if not isinstance(default_frame, pd.DataFrame):
                default_frame = _template_to_dataframe(
                    _get_template("pricing_rows", DEFAULT_PRICING_ROWS), pricing_columns
                )

            template_editor = st.data_editor(
                default_frame,
                num_rows="dynamic",
                use_container_width=True,
                key="default_pricing_editor",
                column_config={
                    "Year": st.column_config.NumberColumn("Year", step=1),
                    "Base Price": st.column_config.NumberColumn(
                        "Base Price", format="%.2f"
                    ),
                    "Price Growth %": st.column_config.NumberColumn(
                        "Price Growth (%)", format="%.2f"
                    ),
                },
            )

            button_col, save_col, restore_col, apply_col = st.columns(4)

            if button_col.button("Close editor", key="close_pricing_defaults"):
                st.session_state.pricing_defaults_edit_mode = False

            if save_col.button("Save defaults", key="save_pricing_defaults"):
                records = _dataframe_to_template(template_editor, pricing_columns)
                _set_template("pricing_rows", records)
                st.success("Pricing defaults updated.")

            if restore_col.button("Restore baseline", key="reset_pricing_defaults"):
                baseline_template = _template_copy(DEFAULT_PRICING_ROWS)
                _set_template("pricing_rows", baseline_template)
                pricing_table = _default_pricing_table()
                st.session_state.assumptions["Pricing"] = pricing_table
                st.session_state["default_pricing_editor"] = _template_to_dataframe(
                    baseline_template, pricing_columns
                )
                st.success(
                    "Pricing defaults restored and assumptions refreshed."
                )
                _clear_schedule_editor_state("assump::pricing")

            if apply_col.button("Apply to assumptions", key="apply_pricing_defaults"):
                records = _dataframe_to_template(template_editor, pricing_columns)
                _set_template("pricing_rows", records)
                pricing_table = _default_pricing_table()
                st.session_state.assumptions["Pricing"] = pricing_table
                st.session_state["default_pricing_editor"] = template_editor
                st.success(
                    "Pricing assumptions refreshed from updated defaults."
                )
                _clear_schedule_editor_state("assump::pricing")

        assumption_tables["Pricing"] = st.session_state.assumptions["Pricing"]

        st.markdown("---")
        st.markdown("#### Operating Cost Assumptions")
        operating_table = _ensure_operating_cost_table(
            st.session_state.assumptions.get("Operating Costs")
        )
        st.session_state.assumptions["Operating Costs"] = operating_table

        st.session_state.setdefault("operating_remove_choice", "-- Select Item --")
        st.session_state.setdefault("operating_increment_target", "All categories")
        st.session_state.setdefault("operating_increment_column", "Monthly Cost")
        st.session_state.setdefault("operating_increment_pct", 0.0)

        add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])

        if add_col.button("Add Item", key="operating_add_row"):
            operating_table = _add_operating_cost_row(operating_table)
            st.session_state.assumptions["Operating Costs"] = operating_table
            _clear_schedule_editor_state("assump::operating_costs")

        option_labels: list[str] = []
        option_index: Dict[str, int] = {}
        for idx_row, row in operating_table.iterrows():
            category = str(row.get("Category", "")).strip() or f"Item {idx_row + 1}"
            year_value = row.get("Year")
            if pd.notna(year_value):
                label = f"{category} ({int(year_value)})"
            else:
                label = category
            option_labels.append(label)
            option_index[label] = idx_row

        remove_select_col.selectbox(
            "Select item",
            options=["-- Select Item --"] + option_labels,
            key="operating_remove_choice",
        )

        if remove_btn_col.button("Remove Item", key="operating_remove_row"):
            choice = st.session_state.get("operating_remove_choice")
            if choice in option_index:
                operating_table = _remove_operating_cost_row(
                    operating_table, option_index[choice]
                )
                st.session_state.assumptions["Operating Costs"] = operating_table
                st.session_state["operating_remove_choice"] = "-- Select Item --"
                _clear_schedule_editor_state("assump::operating_costs")

        inc_target_col, inc_column_col, inc_pct_col, inc_btn_col = st.columns(
            [2, 1.5, 1, 1]
        )

        target_options = ["All categories"] + sorted(
            {
                str(cat).strip()
                for cat in operating_table.get("Category", pd.Series(dtype=str))
                .dropna()
                .tolist()
                if str(cat).strip()
            }
        )
        inc_target_col.selectbox(
            "Apply increment to",
            options=target_options,
            key="operating_increment_target",
        )

        inc_column_col.selectbox(
            "Column",
            options=["Monthly Cost", "Inflation %"],
            key="operating_increment_column",
        )

        inc_pct_col.number_input(
            "Yearly increment (%)",
            min_value=-100.0,
            max_value=100.0,
            step=0.1,
            key="operating_increment_pct",
        )

        if inc_btn_col.button("Apply increment", key="operating_apply_increment"):
            operating_table = _apply_operating_cost_increment(
                operating_table,
                st.session_state.get("operating_increment_pct", 0.0),
                st.session_state.get("operating_increment_target"),
                st.session_state.get("operating_increment_column", "Monthly Cost"),
            )
            st.session_state.assumptions["Operating Costs"] = operating_table
            _clear_schedule_editor_state("assump::operating_costs")

        def _save_operating(updated: pd.DataFrame) -> None:
            ensured = _ensure_operating_cost_table(updated)
            st.session_state.assumptions["Operating Costs"] = ensured

        _render_schedule_row_editor(
            "assump::operating_costs",
            st.session_state.assumptions["Operating Costs"],
            _save_operating,
        )

        st.session_state.setdefault("operating_defaults_edit_mode", False)
        toggle_label = (
            "Hide default operating cost assumptions"
            if st.session_state.operating_defaults_edit_mode
            else "Edit default operating cost assumptions"
        )
        if st.button(toggle_label, key="toggle_operating_defaults"):
            st.session_state.operating_defaults_edit_mode = not st.session_state[
                "operating_defaults_edit_mode"
            ]

        if st.session_state.operating_defaults_edit_mode:
            st.markdown("##### Default Operating Cost Assumptions")
            st.caption(
                "Update the baseline operating cost table used when refreshing these assumptions."
            )

            operating_columns = ["Year", "Category", "Monthly Cost", "Inflation %"]
            default_frame = st.session_state.get("default_operating_editor")
            if not isinstance(default_frame, pd.DataFrame):
                default_frame = _template_to_dataframe(
                    _get_template("operating_rows", DEFAULT_OPERATING_COST_ROWS),
                    operating_columns,
                )

            template_editor = st.data_editor(
                default_frame,
                num_rows="dynamic",
                use_container_width=True,
                key="default_operating_editor",
                column_config={
                    "Year": st.column_config.NumberColumn("Year", step=1),
                    "Monthly Cost": st.column_config.NumberColumn(
                        "Monthly Cost", format="%.2f"
                    ),
                    "Inflation %": st.column_config.NumberColumn(
                        "Inflation (%)", format="%.2f"
                    ),
                },
            )

            button_col, save_col, restore_col, apply_col = st.columns(4)

            if button_col.button("Close editor", key="close_operating_defaults"):
                st.session_state.operating_defaults_edit_mode = False

            if save_col.button("Save defaults", key="save_operating_defaults"):
                records = _dataframe_to_template(template_editor, operating_columns)
                _set_template("operating_rows", records)
                st.success("Operating cost defaults updated.")

            if restore_col.button("Restore baseline", key="reset_operating_defaults"):
                baseline_template = _template_copy(DEFAULT_OPERATING_COST_ROWS)
                _set_template("operating_rows", baseline_template)
                operating_table = _default_operating_cost_table()
                st.session_state.assumptions["Operating Costs"] = operating_table
                st.session_state["default_operating_editor"] = _template_to_dataframe(
                    baseline_template, operating_columns
                )
                st.success(
                    "Operating cost defaults restored and assumptions refreshed."
                )
                _clear_schedule_editor_state("assump::operating_costs")

            if apply_col.button("Apply to assumptions", key="apply_operating_defaults"):
                records = _dataframe_to_template(template_editor, operating_columns)
                _set_template("operating_rows", records)
                operating_table = _default_operating_cost_table()
                st.session_state.assumptions["Operating Costs"] = operating_table
                st.session_state["default_operating_editor"] = template_editor
                st.success(
                    "Operating cost assumptions refreshed from updated defaults."
                )
                _clear_schedule_editor_state("assump::operating_costs")

        assumption_tables["Operating Costs"] = st.session_state.assumptions[
            "Operating Costs"
        ]
    
        st.markdown("---")
        st.markdown("#### Capital & Financing Assumptions")
        capital_table = _ensure_capital_financing_table(
            st.session_state.assumptions.get("Capital & Financing")
        )
        st.session_state.assumptions["Capital & Financing"] = capital_table
    
        def _save_capital(updated: pd.DataFrame) -> None:
            ensured = _ensure_capital_financing_table(updated)
            st.session_state.assumptions["Capital & Financing"] = ensured
    
        _render_schedule_row_editor(
            "assump::capital_financing",
            st.session_state.assumptions["Capital & Financing"],
            _save_capital,
        )
        assumption_tables["Capital & Financing"] = st.session_state.assumptions[
            "Capital & Financing"
        ]
    
        st.markdown("---")
        st.markdown("#### Valuation Inputs")
        include_valuation = st.checkbox("Include valuation inputs", value=True)
        valuation_table = _ensure_valuation_inputs_table(
            st.session_state.assumptions.get("Valuation Inputs")
        )
        st.session_state.assumptions["Valuation Inputs"] = valuation_table
    
        def _save_valuation(updated: pd.DataFrame) -> None:
            ensured = _ensure_valuation_inputs_table(updated)
            st.session_state.assumptions["Valuation Inputs"] = ensured
    
        _render_schedule_row_editor(
            "assump::valuation_inputs",
            st.session_state.assumptions["Valuation Inputs"],
            _save_valuation,
        )
    
        if include_valuation:
            valuation_inputs = _valuation_table_to_inputs(
                st.session_state.assumptions["Valuation Inputs"]
            )
        else:
            valuation_inputs = {}
    
        assumption_tables["Valuation Inputs"] = st.session_state.assumptions[
            "Valuation Inputs"
        ]

    with tabs[3]:
        st.subheader("Financial Statements")
        if st.session_state.results is None:
            st.info("Run the scenarios to generate the financial statements.")
        else:
            results = st.session_state.results
            financial_tabs = st.tabs(
                [
                        "Statement of Financial Performance",
                        "Statement of Financial Position",
                        "Statement of Cash Flow",
                    ]
                )
    
            scenario_label = results.get("selected_scenario", "Scenario")

            with financial_tabs[0]:
                try:
                    sop_base = results["model"].statement_of_financial_performance(
                        results["base"], annual=True
                    )
                    sop_scenario = results["model"].statement_of_financial_performance(
                        results["scenario"], annual=True
                    )
                    st.dataframe(
                        pd.concat(
                            {"Base": sop_base, scenario_label: sop_scenario}, axis=1
                        )
                        .swaplevel(axis=1)
                        .sort_index(axis=1, level=0)
                    )
                    _render_financial_performance_charts(
                        sop_base, sop_scenario, scenario_label
                    )
                except ValueError as exc:
                    st.info(str(exc))

            with financial_tabs[1]:
                try:
                    sofp_base = results["model"].statement_of_financial_position(
                        results["base"], annual=True
                    )
                    sofp_scenario = results["model"].statement_of_financial_position(
                        results["scenario"], annual=True
                    )
                    st.dataframe(
                        pd.concat(
                            {"Base": sofp_base, scenario_label: sofp_scenario}, axis=1
                        )
                        .swaplevel(axis=1)
                        .sort_index(axis=1, level=0)
                    )
                    _render_financial_position_charts(
                        sofp_base, sofp_scenario, scenario_label
                    )
                except ValueError as exc:
                    st.info(str(exc))

            with financial_tabs[2]:
                try:
                    socf_base = results["model"].statement_of_cash_flow(
                        results["base"], annual=True
                    )
                    socf_scenario = results["model"].statement_of_cash_flow(
                        results["scenario"], annual=True
                    )
                    st.dataframe(
                        pd.concat(
                            {"Base": socf_base, scenario_label: socf_scenario}, axis=1
                        )
                        .swaplevel(axis=1)
                        .sort_index(axis=1, level=0)
                    )
                    _render_cash_flow_charts(
                        socf_base, socf_scenario, scenario_label
                    )
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

        custom_adjustments = {
            "Milk price change (%)": float(milk_price),
            "Feed cost change (%)": float(feed_cost),
        }

        current_presets = _current_scenario_presets()
        matches_preset = any(
            np.isclose(
                custom_adjustments["Milk price change (%)"],
                preset["adjustments"].get("Milk price change (%)", 0.0),
            )
            and np.isclose(
                custom_adjustments["Feed cost change (%)"],
                preset["adjustments"].get("Feed cost change (%)", 0.0),
            )
            for preset in current_presets.values()
        )

        scenario_suite = _build_scenario_suite()

        if not matches_preset:
            suffix = _format_scenario_label(
                int(round(custom_adjustments["Milk price change (%)"])),
                int(round(custom_adjustments["Feed cost change (%)"])),
            )
            custom_label = (
                "Custom Scenario"
                if suffix == "Base Scenario"
                else f"Custom Scenario – {suffix}"
            )
            scenario_suite = _build_scenario_suite(custom_label, custom_adjustments)

        try:
            model, _, scenario_results = _execute_scenario_suite(
                schedule_df,
                valuation_inputs,
                combined_supplementary,
                scenario_suite,
            )
        except ValueError as exc:
            st.error(str(exc))
            return

        st.success("Scenario suite complete")

        st.session_state.all_scenario_results = scenario_results

        previous_selection = st.session_state.get("selected_scenario_name")
        if previous_selection not in scenario_results:
            previous_selection = "Base Case Scenario"
            if previous_selection not in scenario_results:
                previous_selection = next(iter(scenario_results))

        st.session_state.selected_scenario_name = previous_selection
        st.session_state.results = scenario_results[previous_selection]
        st.session_state.excel_bytes_map = {}

    results = st.session_state.results

    if results is None:
        with excel_download_container:
            st.info("Run a scenario to enable the Excel model download.")

        st.info(
            "Update the input schedule, adjust the sliders, and press *Run Scenarios* "
            "to evaluate alternative assumptions."
        )
    else:
        model = results["model"]
        scenario = results["scenario"]
        kpis = results["kpis"]
        break_even = results["break_even"]
        selected_scenario = results.get("selected_scenario", "Scenario")
        model.scenario_name = selected_scenario

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

        excel_map: Dict[str, bytes] = st.session_state.setdefault("excel_bytes_map", {})
        excel_bytes = excel_map.get(selected_scenario)
        key_suffix = _scenario_key_suffix(selected_scenario)

        with excel_download_container:
            st.markdown("#### Excel Model Download")
            if not excel_bytes:
                if st.button(
                    "Prepare Excel Model",
                    key=f"prepare_excel_{key_suffix}",
                ):
                    with st.spinner("Preparing Excel workbook..."):
                        excel_bytes = _generate_excel_bytes(
                            model,
                            results,
                            selected_scenario,
                            _current_model_author(),
                        )
                    excel_map[selected_scenario] = excel_bytes
                    st.session_state.excel_bytes_map = excel_map
            if excel_bytes:
                st.download_button(
                    "Download Excel Model",
                    data=excel_bytes,
                    file_name="Goat_Farm_Financial_Model.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key=f"download_excel_{key_suffix}",
                )
                if st.button(
                    "Clear Prepared Excel",
                    key=f"clear_excel_{key_suffix}",
                ):
                    excel_map.pop(selected_scenario, None)
                    st.session_state.excel_bytes_map = excel_map
                    excel_bytes = None
            if not excel_bytes:
                st.info("Click 'Prepare Excel Model' to generate the workbook for download.")

    with tabs[4]:
        st.subheader("Dashboard")
        if results is None:
            st.info("Run the scenarios to populate the dashboard charts.")
            st.markdown("---")
            st.subheader("Supplementary Schedules")
            st.info("Supplementary schedules will appear once a scenario has been run.")
        else:
            st.subheader("KPIs (Annual)")
            st.dataframe(kpis.mul(100).round(2))

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

            st.markdown("---")
            st.subheader("Supplementary Schedules")
            supplementary_render = results.get("supplementary", {})
            for name in [
                "Capitalisation Table",
                "Capex Schedule",
                "Asset Schedules",
                "Outputs",
                "Benchmark KPIs",
            ]:
                _render_table(name, supplementary_render.get(name))

    with tabs[5]:
        st.subheader("Advanced Analytics")
        st.markdown(
            "Use the framework below to configure inputs, assumptions, model drivers, "
            "and scenarios for each analytics tool."
        )
        _render_analytics_framework(results)
        if results is None:
            st.info("Run the scenarios to view advanced analytics.")
        else:
            scenario = results["scenario"]
            model = results["model"]
            try:
                adv_annual = model.advanced_analytics(scenario, window=3, annual=True)

                def _render_analytics(
                    block_name: str, payload: Dict[str, object], scenario_label: str
                ) -> None:
                    st.markdown(f"#### {block_name} Advanced Analytics")
                    for key, item in payload.items():
                        title = item.get("title", key.replace("_", " ").title())
                        description = item.get("description", "")
                        tables = item.get("tables", {})
                        with st.expander(title, expanded=False):
                            if description:
                                st.caption(description)
                            if isinstance(tables, dict):
                                for table_name, table in tables.items():
                                    st.markdown(f"**{table_name}**")
                                    if not isinstance(table, pd.DataFrame):
                                        st.info("No data available for this table.")
                                        continue

                                    override = _get_analytics_override(
                                        scenario_label, block_name, key, table_name
                                    )
                                    display_df = override if override is not None else table
                                    edit_flag_key = _analytics_edit_flag_key(
                                        scenario_label, block_name, key, table_name
                                    )
                                    editor_key = _analytics_editor_key(
                                        scenario_label, block_name, key, table_name
                                    )

                                    if not isinstance(display_df, pd.DataFrame):
                                        st.info("No data available for this table.")
                                        continue

                                    editing = st.session_state.get(edit_flag_key, False)

                                    if editing:
                                        editor_df, meta = _prepare_editor_table(display_df)
                                        working_key = f"{editor_key}::working"
                                        working_df = st.session_state.get(working_key)
                                        if working_df is None:
                                            working_df = editor_df.copy(deep=True)
                                            st.session_state[working_key] = working_df

                                        st.dataframe(working_df)
                                        if working_df.empty:
                                            st.info(
                                                "This table has no rows to edit. Update the model inputs to populate it."
                                            )
                                        else:
                                            row_indices = list(range(len(working_df)))
                                            selected_row = st.selectbox(
                                                "Select a row to edit",
                                                row_indices,
                                                format_func=lambda idx: _format_row_label(
                                                    working_df, idx
                                                ),
                                                key=f"{editor_key}_row_selector",
                                            )
                                            dtype_map = working_df.dtypes.to_dict()
                                            row_series = working_df.iloc[selected_row]
                                            with st.form(f"{editor_key}_row_form"):
                                                updated_values: Dict[str, Any] = {}
                                                for column in working_df.columns:
                                                    widget_key = (
                                                        f"{editor_key}::{selected_row}::{column}"
                                                    )
                                                    cell_value = row_series[column]
                                                    updated_values[column] = _render_row_input(
                                                        column,
                                                        cell_value,
                                                        dtype_map[column],
                                                        widget_key,
                                                    )

                                                submitted = st.form_submit_button(
                                                    "Apply Row Changes"
                                                )
                                                if submitted:
                                                    for column, raw_value in (
                                                        updated_values.items()
                                                    ):
                                                        coerced = _coerce_row_value(
                                                            raw_value, dtype_map[column]
                                                        )
                                                        working_df.iat[
                                                            selected_row,
                                                            working_df.columns.get_loc(
                                                                column
                                                            ),
                                                        ] = coerced
                                                    st.session_state[working_key] = (
                                                        working_df
                                                    )
                                                    _maybe_rerun()

                                        action_cols = st.columns(3)
                                        if action_cols[0].button(
                                            "Save Changes",
                                            key=f"save_{editor_key}",
                                        ):
                                            current_df = st.session_state.get(
                                                working_key, editor_df
                                            )
                                            restored = _restore_editor_table(
                                                current_df, meta
                                            )
                                            _set_analytics_override(
                                                scenario_label,
                                                block_name,
                                                key,
                                                table_name,
                                                restored,
                                            )
                                            st.session_state.pop(working_key, None)
                                            st.session_state[edit_flag_key] = False
                                            _maybe_rerun()

                                        if action_cols[1].button(
                                            "Cancel",
                                            key=f"cancel_{editor_key}",
                                        ):
                                            st.session_state.pop(working_key, None)
                                            st.session_state[edit_flag_key] = False
                                            _maybe_rerun()

                                        if action_cols[2].button(
                                            "Restore Original",
                                            key=f"reset_{editor_key}",
                                        ):
                                            _clear_analytics_override(
                                                scenario_label,
                                                block_name,
                                                key,
                                                table_name,
                                            )
                                            st.session_state.pop(working_key, None)
                                            st.session_state[edit_flag_key] = False
                                            _maybe_rerun()
                                    else:
                                        st.dataframe(display_df)
                                        if override is not None:
                                            st.caption("Manual override applied.")
                                        button_cols = st.columns(2)
                                        if button_cols[0].button(
                                            "Edit Table",
                                            key=f"edit_{editor_key}",
                                        ):
                                            st.session_state[edit_flag_key] = True
                                            _maybe_rerun()

                                        if button_cols[1].button(
                                            "Clear Manual Override",
                                            key=f"clear_{editor_key}",
                                            disabled=override is None,
                                        ):
                                            _clear_analytics_override(
                                                scenario_label,
                                                block_name,
                                                key,
                                                table_name,
                                            )
                                            _maybe_rerun()
                            else:
                                st.info("No tables available for this analysis.")

                selected_scenario_name = results.get("selected_scenario", "Scenario")
                _render_analytics("Annual", adv_annual, selected_scenario_name)
            except ValueError as exc:
                st.info(str(exc))

if __name__ == "__main__":
    main()
