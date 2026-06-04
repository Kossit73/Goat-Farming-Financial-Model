import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd


_STREAMLIT_APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
_spec = importlib.util.spec_from_file_location("streamlit_app", _STREAMLIT_APP_PATH)
assert _spec and _spec.loader  # type: ignore[truthy-bool]
streamlit_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(streamlit_app)


def _reset_local_state() -> None:
    streamlit_app._LOCAL_SESSION_STATE.clear()


def test_default_schedule_spans_production_horizon():
    horizon = pd.DataFrame({"Start Year": [2025], "End Year": [2027]})

    core, details = streamlit_app._default_schedule_components(
        production_horizon=horizon
    )

    periods = pd.to_datetime(core["Period"], errors="coerce").dropna()

    assert not periods.empty
    assert periods.min().year == 2025
    assert periods.max().year == 2027
    assert len(periods) == (2027 - 2025 + 1) * 12

    # Ensure detail schedules inherit the same period coverage
    for table in details.values():
        detail_periods = pd.to_datetime(table["Period"], errors="coerce").dropna()
        if detail_periods.empty:
            continue
        assert detail_periods.min().year == 2025
        assert detail_periods.max().year == 2027


def test_default_schedule_uses_builtin_horizon():
    horizon = streamlit_app._default_production_horizon_table()
    core, _ = streamlit_app._default_schedule_components(
        production_horizon=horizon
    )

    periods = pd.to_datetime(core["Period"], errors="coerce").dropna()

    start_year = int(horizon["Start Year"].iloc[0])
    end_year = int(horizon["End Year"].iloc[0])

    assert periods.min().year == start_year
    assert periods.max().year == end_year
    assert len(periods) == (end_year - start_year + 1) * 12


def test_scenario_presets_cover_key_cases():
    names = set(streamlit_app.SCENARIO_PRESETS.keys())
    assert {"Base Case Scenario", "Best Case Scenario", "Worst Case Scenario"}.issubset(
        names
    )


def test_build_scenario_suite_supports_custom_entries():
    custom_adjustments = {"Milk price change (%)": 5.0, "Feed cost change (%)": -3.0}
    custom_label = "Custom Scenario – Milk +5%, Feed -3%"

    base_suite = streamlit_app._build_scenario_suite()
    assert custom_label not in base_suite

    custom_suite = streamlit_app._build_scenario_suite(custom_label, custom_adjustments)
    assert custom_label in custom_suite
    assert (
        custom_suite[custom_label]["adjustments"]["Milk price change (%)"]
        == custom_adjustments["Milk price change (%)"]
    )


def test_current_scenario_presets_respect_overrides():
    _reset_local_state()

    override_table = pd.DataFrame(
        {
            "Driver": ["Milk price change (%)", "Feed cost change (%)"],
            "Change %": [2.5, -1.25],
        }
    )

    streamlit_app._LOCAL_SESSION_STATE["scenario_preset_tables"] = {
        "Base Case Scenario": override_table
    }
    streamlit_app._LOCAL_SESSION_STATE["scenario_preset_descriptions"] = {
        "Base Case Scenario": "Custom base preset"
    }

    presets = streamlit_app._current_scenario_presets()

    base_preset = presets["Base Case Scenario"]
    assert base_preset["adjustments"]["Milk price change (%)"] == 2.5
    assert base_preset["adjustments"]["Feed cost change (%)"] == -1.25
    assert base_preset["description"] == "Custom base preset"

    _reset_local_state()


def test_scenario_preset_add_and_remove_variables():
    _reset_local_state()

    base_table = streamlit_app._get_scenario_preset_table("Base Case Scenario")
    assert "Milk price change (%)" in base_table["Driver"].tolist()

    streamlit_app._remove_scenario_preset_driver(
        "Base Case Scenario", "Milk price change (%)"
    )

    updated_table = streamlit_app._get_scenario_preset_table("Base Case Scenario")
    assert "Milk price change (%)" not in updated_table["Driver"].tolist()

    streamlit_app._add_scenario_preset_driver(
        "Base Case Scenario", "Herd productivity change (%)", 4.0
    )

    refreshed_table = streamlit_app._get_scenario_preset_table("Base Case Scenario")
    assert "Herd productivity change (%)" in refreshed_table["Driver"].tolist()

    presets = streamlit_app._current_scenario_presets()
    base_adjustments = presets["Base Case Scenario"]["adjustments"]

    assert "Milk price change (%)" not in base_adjustments
    assert base_adjustments["Herd productivity change (%)"] == 4.0

    _reset_local_state()


def test_rebase_schedule_to_horizon_extends_periods():
    short_horizon = pd.DataFrame({"Start Year": [2024], "End Year": [2025]})
    core, details = streamlit_app._default_schedule_components(
        production_horizon=short_horizon
    )

    core.at[0, "Revenue"] = 123456.0
    details["COGS Schedule"].at[0, "COGS"] = 654321.0

    extended_core, extended_details = streamlit_app._rebase_schedule_to_horizon(
        core, details, 2024, 2027
    )

    periods = pd.to_datetime(extended_core["Period"], errors="coerce").dropna()
    assert periods.min().year == 2024
    assert periods.max().year == 2027
    assert len(periods) == (2027 - 2024 + 1) * 12

    january_2024 = periods == pd.Timestamp(2024, 1, 31)
    assert not january_2024.empty
    assert (
        extended_core.loc[january_2024, "Revenue"].iloc[0]
        == core.loc[0, "Revenue"]
    )

    cogs_table = extended_details.get("COGS Schedule")
    assert cogs_table is not None
    cogs_periods = pd.to_datetime(cogs_table["Period"], errors="coerce").dropna()
    assert cogs_periods.max().year == 2027
    assert not cogs_table.loc[cogs_periods.dt.year == 2027, "COGS"].isna().all()


def test_ensure_operating_cost_table_forward_fills_years_without_fillna_method_kwarg():
    raw = pd.DataFrame(
        {
            "Year": [2024, None, 2026],
            "Category": ["Utilities", "Utilities", "Utilities"],
            "Monthly Cost": [1000.0, 1100.0, 1200.0],
            "Inflation %": [3.0, 3.0, 3.0],
        }
    )

    table = streamlit_app._ensure_operating_cost_table(raw)

    assert table["Year"].dtype.name == "Int64"
    assert table["Year"].tolist() == [2024, 2024, 2026]


def test_standalone_app_bootstraps_and_runs_without_top_level_exceptions():
    repo_root = _STREAMLIT_APP_PATH.parent
    command = [
        sys.executable,
        "-c",
        (
            "from streamlit.testing.v1 import AppTest; "
            "at = AppTest.from_file('streamlit_app.py'); "
            "at.run(timeout=20); "
            "print('exc_count', len(at.exception)); "
            "print('tab_count', len(at.tabs)); "
            "raise SystemExit(0 if len(at.exception) == 0 else 1)"
        ),
    ]
    completed = subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "exc_count 0" in completed.stdout
