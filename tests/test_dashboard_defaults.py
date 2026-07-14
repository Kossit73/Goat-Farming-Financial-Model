from contextlib import nullcontext
import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from goat_financial_model.editor_registry import build_remove_options


_STREAMLIT_APP_PATH = Path(__file__).resolve().parents[1] / "streamlit_app.py"
_spec = importlib.util.spec_from_file_location("streamlit_app", _STREAMLIT_APP_PATH)
assert _spec and _spec.loader  # type: ignore[truthy-bool]
streamlit_app = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(streamlit_app)


def _reset_local_state() -> None:
    streamlit_app._LOCAL_SESSION_STATE.clear()


def test_row_selector_labels_prioritize_operating_cost_fields():
    df = pd.DataFrame(
        [
            {
                "Year": 2027,
                "Business Unit": "General",
                "Field": "variable_feed_cost_per_herd",
                "Category": "Feed",
            }
        ]
    )

    label = streamlit_app._format_row_label(df, 0)

    assert "Field: variable_feed_cost_per_herd" in label
    assert "Category: Feed" in label


def test_build_remove_options_preserves_duplicate_labels():
    df = pd.DataFrame(
        [
            {"Field": "variable_feed_cost_per_herd", "Category": "Feed", "Year": 2027},
            {"Field": "variable_feed_cost_per_herd", "Category": "Feed", "Year": 2027},
        ]
    )

    labels, index_lookup = build_remove_options(
        df,
        lambda row: " | ".join(
            [
                str(row.get("Field", "")).strip(),
                str(row.get("Category", "")).strip(),
                str(int(row["Year"])) if pd.notna(row.get("Year")) else "",
            ]
        ).strip(" | "),
    )

    assert len(labels) == 2
    assert labels[0] == "variable_feed_cost_per_herd | Feed | 2027"
    assert labels[1] == "variable_feed_cost_per_herd | Feed | 2027 [2]"
    assert index_lookup[labels[0]] == 0
    assert index_lookup[labels[1]] == 1


def test_default_results_wait_for_explicit_run():
    _reset_local_state()

    streamlit_app._ensure_default_results_loaded()

    assert streamlit_app._LOCAL_SESSION_STATE.get("results") is None
    assert (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_RESULTS_STALE_KEY)
        is True
    )

    _reset_local_state()


def test_default_business_configuration_includes_breeding_transfer_controls():
    config = streamlit_app._default_business_configuration_table()

    assert config.columns.tolist() == [
        "Business Type",
        "Operating Model",
        "Transfer Destination",
        "Reporting View",
        "Transfer Pricing Method",
        "Allow External Kid Sales",
    ]
    assert config.loc[0, "Business Type"] == "Combined"
    assert config.loc[0, "Operating Model"] == "Standalone"


def test_default_assumptions_include_breeding_transfer_tables():
    assumptions = streamlit_app._default_assumption_tables()

    for name in [
        "Kid Routing Rules",
        "Internal Transfer Pricing",
        "Downstream Intake Rules",
        "Transfer Elimination Rules",
    ]:
        assert name in assumptions
        assert isinstance(assumptions[name], pd.DataFrame)
        assert not assumptions[name].empty


def test_breeding_to_unit_mode_emits_transfer_schedules():
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Business Configuration"] = pd.DataFrame(
        {
            "Business Type": ["Breeding"],
            "Operating Model": ["Breeding-to-Unit"],
            "Transfer Destination": ["Meat"],
            "Reporting View": ["Consolidated"],
            "Transfer Pricing Method": ["Cost"],
            "Allow External Kid Sales": [True],
        }
    )
    assumptions = streamlit_app._sync_transfer_tables_to_business_configuration(assumptions)
    core, details = streamlit_app._default_schedule_components(
        production_horizon=assumptions.get("Production Horizon"),
        assumptions=assumptions,
    )
    schedule = streamlit_app._build_schedule_dataframe(core, details, assumptions)
    biological = streamlit_app._derive_biological_schedules(schedule, assumptions)

    assert "Kid Availability Schedule" in biological
    assert "Internal Transfer Schedule" in biological
    assert "Downstream Intake Schedule" in biological
    assert "Breeding Unit Schedule" in biological
    assert "Destination Unit Schedule" in biological
    assert "Reporting Schedule - Breeding" in biological
    assert "Reporting Schedule - Consolidated" in biological
    assert not biological["Kid Availability Schedule"].empty


def test_reporting_schedules_apply_internal_transfer_elimination():
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Business Configuration"] = pd.DataFrame(
        {
            "Business Type": ["Breeding"],
            "Operating Model": ["Breeding-to-Unit"],
            "Transfer Destination": ["Meat"],
            "Reporting View": ["Consolidated"],
            "Transfer Pricing Method": ["Cost"],
            "Allow External Kid Sales": [True],
        }
    )
    assumptions = streamlit_app._sync_transfer_tables_to_business_configuration(assumptions)
    core, details = streamlit_app._default_schedule_components(
        production_horizon=assumptions.get("Production Horizon"),
        assumptions=assumptions,
    )
    schedule = streamlit_app._build_schedule_dataframe(core, details, assumptions)
    biological = streamlit_app._derive_biological_schedules(schedule, assumptions)

    breeding = biological["Reporting Schedule - Breeding"]
    destination = biological["Reporting Schedule - Meat"]
    consolidated = biological["Reporting Schedule - Consolidated"]
    elimination = biological["Internal Transfer Elimination Schedule"]

    combined_cogs = (
        pd.to_numeric(breeding["COGS"], errors="coerce").fillna(0.0)
        + pd.to_numeric(destination["COGS"], errors="coerce").fillna(0.0)
    )
    cost_elimination = pd.to_numeric(
        elimination["Cost Elimination"], errors="coerce"
    ).fillna(0.0)

    assert cost_elimination.abs().max() > 0
    pd.testing.assert_series_equal(
        pd.to_numeric(consolidated["COGS"], errors="coerce").fillna(0.0),
        combined_cogs.add(cost_elimination, fill_value=0.0),
        check_names=False,
    )


def test_unit_aggregations_preserve_business_unit_breakdown():
    core = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-02-29"],
        }
    )
    variable_table = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-01-31", "2024-02-29"],
            "Business Unit": ["Breeding", "Meat", "Breeding"],
            "Item": ["Health", "Health", "Health"],
            "Amount": [10.0, 20.0, 30.0],
        }
    )
    direct_table = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-02-29"],
            "Business Unit": ["Breeding", "Meat"],
            "Position": ["Farmhand", "Butcher"],
            "Head Count": [1.0, 1.0],
            "Monthly Salary per Head": [100.0, 200.0],
            "Total Salary": [100.0, 200.0],
        }
    )
    admin_table = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-02-29"],
            "Business Unit": ["Breeding", "Meat"],
            "Position": ["Clerk", "Manager"],
            "Head Count": [1.0, 1.0],
            "Monthly Salary per Head": [50.0, 75.0],
            "Total Salary": [50.0, 75.0],
        }
    )

    variable_agg = streamlit_app._aggregate_variable_expenses_by_business_unit(
        variable_table, core
    )
    direct_agg = streamlit_app._aggregate_direct_wages_by_business_unit(
        direct_table, core
    )
    admin_agg = streamlit_app._aggregate_admin_wages_by_business_unit(
        admin_table, core
    )

    assert variable_agg.columns.tolist() == ["Period", "Breeding", "Meat"]
    assert direct_agg.columns.tolist() == ["Period", "Breeding", "Meat"]
    assert admin_agg.columns.tolist() == ["Period", "Breeding", "Meat"]
    assert variable_agg.loc[0, "Breeding"] == 10.0
    assert variable_agg.loc[0, "Meat"] == 20.0
    assert direct_agg.loc[1, "Meat"] == 200.0
    assert admin_agg.loc[0, "Breeding"] == 50.0


def test_refresh_results_stale_state_tracks_input_changes():
    _reset_local_state()

    assumptions = streamlit_app._default_assumption_tables()
    core, details = streamlit_app._default_schedule_components(
        production_horizon=assumptions.get("Production Horizon"),
        assumptions=assumptions,
    )

    streamlit_app._LOCAL_SESSION_STATE["assumptions"] = assumptions
    streamlit_app._LOCAL_SESSION_STATE["core_schedule"] = core
    streamlit_app._LOCAL_SESSION_STATE["detail_schedules"] = details
    streamlit_app._LOCAL_SESSION_STATE["supplementary"] = (
        streamlit_app._default_supplementary_tables()
    )
    streamlit_app._LOCAL_SESSION_STATE["schedule_period_type"] = "monthly"
    streamlit_app._LOCAL_SESSION_STATE["results"] = {
        "selected_scenario": "Base Case Scenario"
    }
    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_INPUT_VERSION_KEY] = 3
    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_LAST_RUN_VERSION_KEY] = 3

    streamlit_app._refresh_results_stale_state()
    assert (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_RESULTS_STALE_KEY)
        is False
    )

    updated_controls = assumptions["Scenario Controls"].copy()
    updated_controls.loc[
        updated_controls["Driver"] == "Feed cost change (%)", "Change %"
    ] = 7.0
    assumptions["Scenario Controls"] = updated_controls
    streamlit_app._LOCAL_SESSION_STATE["assumptions"] = assumptions
    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_INPUT_VERSION_KEY] = 4

    streamlit_app._refresh_results_stale_state()
    assert (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_RESULTS_STALE_KEY)
        is True
    )

    _reset_local_state()


def test_reset_cached_results_marks_outputs_stale_but_keeps_last_run():
    _reset_local_state()

    scenario_results = {
        "Base Case Scenario": {"selected_scenario": "Base Case Scenario"}
    }
    streamlit_app._store_run_bundle(
        scenario_results,
        selected_scenario_name="Base Case Scenario",
    )
    streamlit_app._LOCAL_SESSION_STATE["excel_bytes_map"] = {"Base Case Scenario": b"test"}
    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_INPUT_VERSION_KEY] = 8
    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_LAST_RUN_VERSION_KEY] = 7

    streamlit_app._reset_cached_results()

    assert streamlit_app._LOCAL_SESSION_STATE.get("results") == {
        "selected_scenario": "Base Case Scenario"
    }
    assert "Base Case Scenario" in streamlit_app._LOCAL_SESSION_STATE.get(
        "all_scenario_results", {}
    )
    assert (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_RUN_BUNDLE_KEY, {})
        .get("selected_scenario_name")
        == "Base Case Scenario"
    )
    assert "Base Case Scenario" in (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_RUN_BUNDLE_KEY, {})
        .get("scenario_results", {})
    )
    assert streamlit_app._LOCAL_SESSION_STATE.get("excel_bytes_map") == {}
    assert (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_INPUT_VERSION_KEY)
        == 9
    )
    assert (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_LAST_RUN_VERSION_KEY)
        == 7
    )
    assert (
        streamlit_app._LOCAL_SESSION_STATE.get(streamlit_app.MODEL_RESULTS_STALE_KEY)
        is True
    )

    _reset_local_state()


def test_cached_result_view_reuses_values_within_same_run_version():
    _reset_local_state()

    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_LAST_RUN_VERSION_KEY] = 5
    call_count = {"value": 0}

    def _builder() -> dict[str, int]:
        call_count["value"] += 1
        return {"count": call_count["value"]}

    first = streamlit_app._cached_result_view(
        "financial_statements",
        "Base Case Scenario",
        _builder,
    )
    second = streamlit_app._cached_result_view(
        "financial_statements",
        "Base Case Scenario",
        _builder,
    )

    assert first == {"count": 1}
    assert second == {"count": 1}
    assert call_count["value"] == 1

    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_LAST_RUN_VERSION_KEY] = 6
    third = streamlit_app._cached_result_view(
        "financial_statements",
        "Base Case Scenario",
        _builder,
    )

    assert third == {"count": 2}
    assert call_count["value"] == 2

    _reset_local_state()


def test_store_run_bundle_aligns_selected_result_and_legacy_keys():
    _reset_local_state()

    scenario_results = {
        "Base Case Scenario": {"selected_scenario": "Base Case Scenario", "value": 1},
        "Downside Scenario": {"selected_scenario": "Downside Scenario", "value": 2},
    }

    bundle = streamlit_app._store_run_bundle(
        scenario_results,
        selected_scenario_name="Downside Scenario",
    )

    assert bundle == streamlit_app._LOCAL_SESSION_STATE.get(
        streamlit_app.MODEL_RUN_BUNDLE_KEY
    )
    assert streamlit_app._LOCAL_SESSION_STATE.get("all_scenario_results") == scenario_results
    assert streamlit_app._LOCAL_SESSION_STATE.get("selected_scenario_name") == (
        "Downside Scenario"
    )
    assert streamlit_app._LOCAL_SESSION_STATE.get("results") == scenario_results[
        "Downside Scenario"
    ]
    assert streamlit_app._current_selected_result() == scenario_results[
        "Downside Scenario"
    ]

    _reset_local_state()


def test_reporting_views_cache_by_run_version_and_scenario(monkeypatch: pytest.MonkeyPatch):
    _reset_local_state()

    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_LAST_RUN_VERSION_KEY] = 4
    call_log: list[str] = []

    def _fake_builder(result_payload: dict[str, object]) -> dict[str, object]:
        scenario_name = str(result_payload.get("selected_scenario", "Scenario"))
        call_log.append(scenario_name)
        return {
            "entity_options": ["Consolidated"],
            "default_entity": "Consolidated",
            "assumptions": {},
            "base_schedules": {},
            "scenario_schedules": {
                "Consolidated": pd.DataFrame({"Scenario": [scenario_name]})
            },
        }

    monkeypatch.setattr(streamlit_app, "_build_reporting_views_for_result", _fake_builder)

    base_payload = {"selected_scenario": "Base Case Scenario"}
    downside_payload = {"selected_scenario": "Downside Scenario"}

    first = streamlit_app._reporting_views_for_result(base_payload)
    second = streamlit_app._reporting_views_for_result(base_payload)
    third = streamlit_app._reporting_views_for_result(downside_payload)

    assert call_log == ["Base Case Scenario", "Downside Scenario"]
    assert first["scenario_schedules"]["Consolidated"].equals(
        second["scenario_schedules"]["Consolidated"]
    )
    assert third["scenario_schedules"]["Consolidated"].iloc[0, 0] == "Downside Scenario"

    _reset_local_state()


def test_dashboard_outputs_cache_by_run_version_scenario_and_entity(
    monkeypatch: pytest.MonkeyPatch,
):
    _reset_local_state()

    streamlit_app._LOCAL_SESSION_STATE[streamlit_app.MODEL_LAST_RUN_VERSION_KEY] = 6
    call_log: list[tuple[str, str]] = []

    def _fake_builder(
        result_payload: dict[str, object],
        entity: str,
    ) -> dict[str, object]:
        scenario_name = str(result_payload.get("selected_scenario", "Scenario"))
        call_log.append((scenario_name, entity))
        return {
            "scenario": pd.DataFrame({"Entity": [entity], "Scenario": [scenario_name]}),
            "valuation_summary": {"entity": entity},
            "model_audit": {},
            "working_capital_annual": pd.DataFrame(),
            "debt_capacity_annual": pd.DataFrame(),
            "ufcf_schedule_annual": pd.DataFrame(),
            "kpis": pd.DataFrame(),
            "break_even": pd.DataFrame(),
            "pricing_assumptions": pd.DataFrame(),
            "product_revenue_summary": pd.DataFrame(),
            "product_qty_summary": pd.DataFrame(),
            "supplementary": {},
        }

    monkeypatch.setattr(streamlit_app, "_build_dashboard_outputs_for_entity", _fake_builder)

    payload = {"selected_scenario": "Base Case Scenario"}

    first = streamlit_app._dashboard_outputs_for_entity(payload, "Consolidated")
    second = streamlit_app._dashboard_outputs_for_entity(payload, "Consolidated")
    third = streamlit_app._dashboard_outputs_for_entity(payload, "Breeding")

    assert call_log == [
        ("Base Case Scenario", "Consolidated"),
        ("Base Case Scenario", "Breeding"),
    ]
    assert first["scenario"].equals(second["scenario"])
    assert third["scenario"].iloc[0]["Entity"] == "Breeding"

    _reset_local_state()


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


def test_default_assumptions_include_biological_engine_tables():
    assumptions = streamlit_app._default_assumption_tables()

    required = {
        "Biological System Settings",
        "Breeding & Reproduction Biology",
        "Lactation Biology",
        "Finishing & Slaughter Biology",
        "Opening Herd Cohorts",
        "Cohort Allocation Rules",
        "Biological Cost Drivers",
    }

    assert required.issubset(assumptions.keys())
    assert not assumptions["Opening Herd Cohorts"].empty
    assert "Age at First Kidding (months)" in assumptions[
        "Breeding & Reproduction Biology"
    ].columns


def test_default_breeding_reproduction_biology_includes_breeder_offtake_controls():
    table = streamlit_app._default_breeding_reproduction_biology_table()

    assert {
        "Breeder Doe Cull Age (months)",
        "Breeder Doe Cull At Parity",
        "Breeder Doe Live Sale Share %",
        "Breeder Buck Replacement Age (months)",
        "Breeder Buck Live Sale Share %",
    }.issubset(table.columns)
    assert table.loc[0, "Breeder Doe Cull Age (months)"] == pytest.approx(72.0)
    assert pd.isna(table.loc[0, "Breeder Doe Cull At Parity"])
    assert table.loc[0, "Breeder Buck Replacement Age (months)"] == pytest.approx(60.0)


def test_breeding_pricing_sync_assigns_livestock_products_to_breeding_unit():
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Business Configuration"] = pd.DataFrame(
        {
            "Business Type": ["Breeding"],
            "Operating Model": ["Breeding-to-Unit"],
            "Transfer Destination": ["Meat"],
            "Reporting View": ["Consolidated"],
            "Transfer Pricing Method": ["Cost"],
            "Allow External Kid Sales": [True],
        }
    )
    assumptions = streamlit_app._sync_transfer_tables_to_business_configuration(assumptions)
    core, _ = streamlit_app._default_schedule_components(
        production_horizon=assumptions.get("Production Horizon"),
        assumptions=assumptions,
    )
    synced = streamlit_app._sync_commercial_assumptions_to_core(assumptions, core)
    pricing = synced["Pricing"]

    livestock_rows = pricing.loc[
        pricing["Product"].isin(["Meat", "Offal", "Pelt", "Live Herd"])
    ]

    assert not livestock_rows.empty
    assert set(livestock_rows["Business Unit"]) == {"Breeding"}


def test_default_biological_start_date_matches_production_horizon_start():
    assumptions = streamlit_app._default_assumption_tables()

    production_horizon = assumptions["Production Horizon"]
    biological_settings = assumptions["Biological System Settings"]
    settings_lookup = dict(
        zip(biological_settings["Setting"], biological_settings["Value"], strict=False)
    )

    expected_start = streamlit_app._opening_biological_start_date_for_horizon(
        production_horizon
    )

    assert settings_lookup["Opening Biological Start Date"] == expected_start


def test_sync_biological_start_date_updates_to_new_horizon_start():
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Production Horizon"] = pd.DataFrame(
        {"Start Year": [2027], "End Year": [2029]}
    )
    assumptions["Biological System Settings"] = pd.DataFrame(
        {
            "Setting": [
                "Model Grain",
                "Opening Biological Start Date",
                "Age Band Width (months)",
            ],
            "Value": ["monthly", "2024-01-31", "1"],
        }
    )

    synced = streamlit_app._sync_biological_start_date_in_assumptions(assumptions)
    biological_settings = synced["Biological System Settings"]
    settings_lookup = dict(
        zip(biological_settings["Setting"], biological_settings["Value"], strict=False)
    )

    assert settings_lookup["Opening Biological Start Date"] == "2027-01-31"


def test_opening_biological_start_date_uses_first_quarter_for_quarterly_horizon():
    horizon = pd.DataFrame({"Start Year": [2026], "End Year": [2028]})

    aligned = streamlit_app._opening_biological_start_date_for_horizon(
        horizon,
        period_type="quarterly",
    )

    assert aligned == "2026-03-31"


def test_opening_herd_cohorts_sync_to_first_herd_plan_target():
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Herd Plan"] = pd.DataFrame(
        {"Year": [2026, 2027], "Herd Size (heads)": [500.0, 525.0], "Herd Growth %": [pd.NA, 5.0]}
    )

    synced = streamlit_app._sync_opening_herd_cohorts_in_assumptions(assumptions)
    opening = synced["Opening Herd Cohorts"]
    total_heads = pd.to_numeric(opening["Head Count"], errors="coerce").sum()

    assert total_heads == pytest.approx(500.0)


def test_opening_herd_cohorts_sync_preserves_relative_mix():
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Herd Plan"] = pd.DataFrame(
        {"Year": [2026], "Herd Size (heads)": [640.0], "Herd Growth %": [pd.NA]}
    )
    assumptions["Opening Herd Cohorts"] = pd.DataFrame(
        {
            "Cohort ID": ["A", "B", "C"],
            "Sex": ["Female", "Male", "Female"],
            "Purpose": ["breeding_doe", "breeding_buck", "replacement_doe"],
            "Age in Months": [24.0, 36.0, 10.0],
            "Head Count": [100.0, 20.0, 40.0],
            "Parity": [2.0, 0.0, 0.0],
            "Pregnant": [True, False, False],
            "Days in Milk": [60.0, 0.0, 0.0],
            "Active": [True, True, True],
        }
    )

    synced = streamlit_app._sync_opening_herd_cohorts_in_assumptions(assumptions)
    opening = synced["Opening Herd Cohorts"].set_index("Cohort ID")

    assert opening.loc["A", "Head Count"] == pytest.approx(400.0)
    assert opening.loc["B", "Head Count"] == pytest.approx(80.0)
    assert opening.loc["C", "Head Count"] == pytest.approx(160.0)


def test_default_asset_schedule_is_derived_from_capex_schedule():
    supplementary = streamlit_app._default_supplementary_tables()

    capex = supplementary["Capex Schedule"]
    assets = supplementary["Asset Schedules"]
    derived = streamlit_app._derive_asset_schedule_from_capex(capex)

    pd.testing.assert_frame_equal(
        assets.reset_index(drop=True),
        derived.reset_index(drop=True),
        check_dtype=False,
    )


def test_default_opening_herd_acquisition_syncs_to_opening_herd():
    assumptions = streamlit_app._default_assumption_tables()
    supplementary = streamlit_app._default_supplementary_tables()

    acquisition = supplementary["Opening Herd Acquisition"]
    opening = assumptions["Opening Herd Cohorts"].set_index("Cohort ID")

    assert "Opening Herd Acquisition" in supplementary
    assert acquisition["Cohort ID"].tolist() == opening.reset_index()["Cohort ID"].tolist()
    for _, row in acquisition.iterrows():
        cohort_id = row["Cohort ID"]
        assert float(row["Quantity"]) == pytest.approx(float(opening.loc[cohort_id, "Head Count"]))
        assert row["Funding Source"] == "Equity"


def test_opening_herd_acquisition_preserves_pricing_and_rolls_into_capex():
    assumptions = streamlit_app._default_assumption_tables()
    supplementary = streamlit_app._default_supplementary_tables()
    acquisition = supplementary["Opening Herd Acquisition"].copy()

    acquisition.loc[acquisition["Cohort ID"] == "BD-001", "Unit Cost"] = 1200.0
    acquisition.loc[acquisition["Cohort ID"] == "BD-001", "Funding Source"] = "Equity"
    acquisition.loc[acquisition["Cohort ID"] == "BB-001", "Unit Cost"] = 2500.0
    acquisition.loc[acquisition["Cohort ID"] == "BB-001", "Funding Source"] = "Debt"
    acquisition["Purchase Year"] = 2026
    supplementary["Opening Herd Acquisition"] = acquisition

    synced = streamlit_app._sync_asset_schedule_from_capex_in_supplementary(
        supplementary,
        assumptions,
    )
    capex = synced["Capex Schedule"]
    acquisition_synced = synced["Opening Herd Acquisition"].set_index("Cohort ID")

    assert float(acquisition_synced.loc["BD-001", "Total Starter-Herd Cost"]) == pytest.approx(180000.0)
    assert float(acquisition_synced.loc["BB-001", "Total Starter-Herd Cost"]) == pytest.approx(20000.0)

    equity_row = capex.loc[capex["Category"] == "Opening Herd Acquisition - Equity"].iloc[0]
    debt_row = capex.loc[capex["Category"] == "Opening Herd Acquisition - Debt"].iloc[0]

    assert int(equity_row["Year"]) == 2026
    assert float(equity_row["Spend"]) == pytest.approx(180000.0)
    assert float(debt_row["Spend"]) == pytest.approx(20000.0)


def test_opening_herd_acquisition_updates_quantity_when_opening_herd_changes():
    assumptions = streamlit_app._default_assumption_tables()
    supplementary = streamlit_app._default_supplementary_tables()
    acquisition = supplementary["Opening Herd Acquisition"].copy()
    acquisition.loc[acquisition["Cohort ID"] == "BD-001", "Unit Cost"] = 900.0
    acquisition.loc[acquisition["Cohort ID"] == "BD-001", "Funding Source"] = "Mixed"
    supplementary["Opening Herd Acquisition"] = acquisition

    opening = assumptions["Opening Herd Cohorts"].copy()
    opening.loc[opening["Cohort ID"] == "BD-001", "Head Count"] = 175.0
    assumptions["Opening Herd Cohorts"] = opening

    synced = streamlit_app._sync_asset_schedule_from_capex_in_supplementary(
        supplementary,
        assumptions,
    )
    acquisition_synced = synced["Opening Herd Acquisition"].set_index("Cohort ID")

    assert float(acquisition_synced.loc["BD-001", "Quantity"]) == pytest.approx(175.0)
    assert float(acquisition_synced.loc["BD-001", "Unit Cost"]) == pytest.approx(900.0)
    assert acquisition_synced.loc["BD-001", "Funding Source"] == "Mixed"
    assert float(acquisition_synced.loc["BD-001", "Total Starter-Herd Cost"]) == pytest.approx(157500.0)


def test_asset_schedule_rolls_forward_from_capex_schedule():
    capex = pd.DataFrame(
        {
            "Year": [2026, 2027, 2027],
            "Category": ["Barn", "Barn", "Milking Line"],
            "Spend": [1000.0, 200.0, 300.0],
            "Depreciation Rate %": [10.0, 10.0, 20.0],
            "Depreciation": [100.0, 20.0, 60.0],
        }
    )

    assets = streamlit_app._derive_asset_schedule_from_capex(capex)

    barn_2026 = assets.loc[(assets["Asset"] == "Barn") & (assets["Year"] == 2026)].iloc[0]
    barn_2027 = assets.loc[(assets["Asset"] == "Barn") & (assets["Year"] == 2027)].iloc[0]
    line_2027 = assets.loc[
        (assets["Asset"] == "Milking Line") & (assets["Year"] == 2027)
    ].iloc[0]

    assert barn_2026["Opening NBV"] == pytest.approx(0.0)
    assert barn_2026["Closing NBV"] == pytest.approx(900.0)
    assert barn_2027["Opening NBV"] == pytest.approx(900.0)
    assert barn_2027["Closing NBV"] == pytest.approx(1080.0)
    assert line_2027["Opening NBV"] == pytest.approx(0.0)
    assert line_2027["Closing NBV"] == pytest.approx(240.0)


def test_capex_asset_reconciliation_has_no_issues_for_derived_schedule():
    supplementary = streamlit_app._default_supplementary_tables()

    issues = streamlit_app._capex_asset_reconciliation_issues(
        supplementary["Capex Schedule"],
        supplementary["Asset Schedules"],
    )

    assert issues == []


def test_capex_asset_reconciliation_flags_asset_mismatch():
    supplementary = streamlit_app._default_supplementary_tables()
    broken_assets = supplementary["Asset Schedules"].copy()
    broken_assets.loc[0, "Additions"] = float(broken_assets.loc[0, "Additions"]) + 1.0

    issues = streamlit_app._capex_asset_reconciliation_issues(
        supplementary["Capex Schedule"],
        broken_assets,
    )

    assert any("do not reconcile" in issue for issue in issues)


def test_default_valuation_inputs_exclude_derived_metrics():
    valuation = streamlit_app._default_valuation_inputs_table()

    assert "IRR" not in valuation["Metric"].tolist()
    assert "NPV" not in valuation["Metric"].tolist()


def test_valuation_table_to_inputs_ignores_derived_metrics():
    raw = pd.DataFrame(
        {
            "Metric": ["WACC", "IRR", "NPV", "Terminal Value"],
            "Value": [0.1, 0.55, 123456.0, 999.0],
        }
    )

    inputs = streamlit_app._valuation_table_to_inputs(raw)

    assert inputs["WACC"] == 0.1
    assert inputs["Terminal Value"] == 999.0
    assert "IRR" not in inputs
    assert "NPV" not in inputs


def test_scenario_presets_cover_key_cases():
    names = set(streamlit_app.SCENARIO_PRESETS.keys())
    assert {"Base Case Scenario", "Best Case Scenario", "Worst Case Scenario"}.issubset(
        names
    )


def test_build_scenario_suite_supports_custom_entries():
    custom_adjustments = {"Milk price change (%)": 5.0, "Feed cost change (%)": -3.0}
    custom_label = "Custom Scenario - Milk +5%, Feed -3%"

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


def test_sparse_loader_tables_do_not_crash_on_scalar_get_fallbacks():
    scenario = streamlit_app._ensure_scenario_preset_table(
        "Base Case Scenario",
        pd.DataFrame({"Change %": [5.0]}),
    )
    assert "Driver" in scenario.columns
    assert scenario["Driver"].astype(str).str.strip().ne("").all()

    merged_drivers = streamlit_app._merge_production_driver_subset(
        None,
        pd.DataFrame({"Milk Yield per Head per Day": [2.5]}),
        ["Milk"],
    )
    assert "Product" in merged_drivers.columns

    pricing = streamlit_app._apply_pricing_yearly_increment(
        pd.DataFrame(
            {
                "Period": ["2026-03-31"],
                "Base Price": [10.0],
                "Allocation %": [100.0],
                "Quantity per Period": [5.0],
                "Active": [True],
            }
        ),
        "Base Price",
        5.0,
    )
    assert pricing.loc[0, "Product"] == "Product"

    operating = streamlit_app._ensure_operating_cost_table(
        pd.DataFrame(
            {
                "Year": [2026],
                "unit_cost_per_head_per_month": [12.0],
                "Inflation %": [3.0],
            }
        )
    )
    assert operating.loc[0, "Field"] == "variable_feed_cost_per_herd"

    direct_wages = streamlit_app._apply_direct_wage_increment(
        pd.DataFrame(
            {
                "Period": ["2026-03-31"],
                "Head Count": [2.0],
                "Monthly Salary per Head": [1500.0],
            }
        ),
        4.0,
    )
    assert direct_wages.loc[0, "Position"] == "Direct Wage"

    admin_wages = streamlit_app._apply_admin_wage_increment(
        pd.DataFrame(
            {
                "Period": ["2026-03-31"],
                "Head Count": [1.0],
                "Monthly Salary per Head": [2200.0],
            }
        ),
        4.0,
    )
    assert admin_wages.loc[0, "Position"] == "Admin Wage"

    variable_expenses = streamlit_app._apply_variable_expense_increment(
        pd.DataFrame(
            {
                "Period": ["2026-03-31"],
                "Amount": [500.0],
            }
        ),
        4.0,
    )
    assert "Item" in variable_expenses.columns


def test_direct_wage_template_normalization_recomputes_itemised_totals():
    records = streamlit_app._normalize_direct_wage_template_records(
        [
            {
                "Position": "Milking Crew",
                "Head Count": 3.0,
                "Monthly Salary per Head": 1800.0,
                "Total Salary": 0.0,
            }
        ]
    )

    assert records == [
        {
            "Position": "Milking Crew",
            "Head Count": 3.0,
            "Monthly Salary per Head": 1800.0,
            "Total Salary": 5400.0,
        }
    ]


def test_default_direct_wage_schedule_rolls_up_itemised_positions():
    _reset_local_state()

    core = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28"],
            "Direct Wages": [8000.0, 8000.0],
        }
    )

    table = streamlit_app._default_direct_wage_table(core)
    summary = streamlit_app._aggregate_direct_wages(table, core)

    assert {
        "Period",
        "Position",
        "Head Count",
        "Monthly Salary per Head",
        "Total Salary",
    }.issubset(table.columns)
    assert table.loc[table["Period"] == "2026-01-31", "Total Salary"].sum() == 8000.0
    assert summary["Direct Wages"].tolist() == [8000.0, 8000.0]

    _reset_local_state()


def test_admin_wage_template_normalization_recomputes_itemised_totals():
    records = streamlit_app._normalize_admin_wage_template_records(
        [
            {
                "Position": "Administration",
                "Head Count": 2.0,
                "Monthly Salary per Head": 900.0,
                "Total Salary": 0.0,
            }
        ]
    )

    assert records == [
        {
            "Position": "Administration",
            "Head Count": 2.0,
            "Monthly Salary per Head": 900.0,
            "Total Salary": 1800.0,
        }
    ]


def test_default_admin_wage_schedule_rolls_up_itemised_positions():
    _reset_local_state()

    core = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28"],
            "Admin Wages": [3500.0, 3500.0],
        }
    )

    table = streamlit_app._default_admin_wage_table(core)
    summary = streamlit_app._aggregate_admin_wages(table, core)

    assert {
        "Period",
        "Position",
        "Head Count",
        "Monthly Salary per Head",
        "Total Salary",
    }.issubset(table.columns)
    assert table.loc[table["Period"] == "2026-01-31", "Total Salary"].sum() == 3500.0
    assert summary["Admin Wages"].tolist() == [3500.0, 3500.0]

    _reset_local_state()


def test_default_assumption_tables_include_master_schedule_inputs():
    assumptions = streamlit_app._default_assumption_tables()

    assert {"Variable Expenses", "Direct Wages", "Admin Wages", "Production Drivers"}.issubset(
        assumptions.keys()
    )
    assert {
        "Item",
        "Amount per Period",
        "Yearly Increase %",
    }.issubset(assumptions["Variable Expenses"].columns)
    assert {
        "Position",
        "Head Count",
        "Monthly Salary per Head",
        "Total Salary",
        "Yearly Increase %",
    }.issubset(assumptions["Direct Wages"].columns)
    assert {
        "Position",
        "Head Count",
        "Monthly Salary per Head",
        "Total Salary",
        "Yearly Increase %",
    }.issubset(assumptions["Admin Wages"].columns)
    assert {
        "Product",
        "Quantity Mode",
        "Lactating Herd Share %",
        "Annual Slaughter Rate % of Herd",
    }.issubset(assumptions["Production Drivers"].columns)


def test_variable_expense_master_inputs_propagate_quarterly_with_yearly_growth():
    core = pd.DataFrame({"Period": ["2026-03-31", "2026-06-30", "2027-03-31"]})
    assumptions = pd.DataFrame(
        {
            "Item": ["Vet Care"],
            "Amount per Period": [100.0],
            "Yearly Increase %": [10.0],
        }
    )

    propagated = streamlit_app._propagate_variable_expense_inputs_to_schedule(
        assumptions, core
    )

    assert propagated["Amount"].tolist() == [300.0, 300.0, 330.0]


def test_direct_wage_master_inputs_propagate_quarterly_and_recompute_totals():
    core = pd.DataFrame({"Period": ["2026-03-31", "2027-03-31"]})
    assumptions = pd.DataFrame(
        {
            "Position": ["Supervisor"],
            "Head Count": [2.0],
            "Monthly Salary per Head": [500.0],
            "Total Salary": [1000.0],
            "Yearly Increase %": [5.0],
        }
    )

    propagated = streamlit_app._propagate_direct_wage_inputs_to_schedule(
        assumptions, core
    )

    assert propagated["Monthly Salary per Head"].tolist() == [500.0, 525.0]
    assert propagated["Total Salary"].tolist() == [3000.0, 3150.0]


def test_admin_wage_master_inputs_feed_default_schedule_components():
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Admin Wages"] = pd.DataFrame(
        {
            "Position": ["Administration"],
            "Head Count": [1.0],
            "Monthly Salary per Head": [1200.0],
            "Total Salary": [1200.0],
            "Yearly Increase %": [0.0],
        }
    )

    _, detail_tables = streamlit_app._default_schedule_components(
        production_horizon=pd.DataFrame({"Start Year": [2026], "End Year": [2026]}),
        period_type="quarterly",
        assumptions=assumptions,
    )

    admin_table = detail_tables["Admin Wages Schedule"]
    assert admin_table["Position"].unique().tolist() == ["Administration"]
    assert admin_table["Total Salary"].iloc[0] == 3600.0


def test_default_pricing_table_uses_period_product_activation_structure():
    pricing = streamlit_app._default_pricing_table()

    assert {
        "Period",
        "Product",
        "Active",
        "Allocation %",
        "Quantity Mode",
        "Manual Quantity Override",
        "Quantity per Period",
        "Unit",
        "Base Price",
        "Price Growth %",
        "Revenue",
    }.issubset(pricing.columns)
    assert "Milk" in pricing["Product"].unique().tolist()


def test_production_drivers_derive_milk_and_cheese_quantities_from_herd():
    schedule = pd.DataFrame(
        {"Herd Size (heads)": [100.0]},
        index=pd.to_datetime(["2026-01-31"]),
    )
    pricing = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-01-31"],
            "Product": ["Milk", "Cheese"],
            "Active": [True, True],
            "Allocation %": [100.0, 100.0],
            "Quantity Mode": ["Derived", "Derived"],
            "Manual Quantity Override": [pd.NA, pd.NA],
            "Quantity per Period": [0.0, 0.0],
            "Unit": ["Litre", "Kg"],
            "Base Price": [2.0, 12.0],
            "Price Growth %": [0.0, 0.0],
        }
    )
    drivers = pd.DataFrame(
        {
            "Product": ["Milk", "Cheese"],
            "Unit": ["Litre", "Kg"],
            "Quantity Mode": ["Derived", "Derived"],
            "Lactating Herd Share %": [50.0, 50.0],
            "Litres per Lactating Doe per Day": [2.0, 2.0],
            "Milk Allocation to Cheese %": [0.0, 25.0],
            "Cheese Yield Kg per Litre": [0.0, 0.2],
            "Annual Slaughter Rate % of Herd": [0.0, 0.0],
            "Meat Yield Kg per Goat": [0.0, 0.0],
            "Pelt Units per Goat": [0.0, 0.0],
            "Driver Growth %": [0.0, 0.0],
        }
    )

    derived = streamlit_app._derive_pricing_quantities_from_production(
        pricing, schedule, drivers
    )

    milk_qty = derived.loc[derived["Product"] == "Milk", "Quantity per Period"].iloc[0]
    cheese_qty = derived.loc[derived["Product"] == "Cheese", "Quantity per Period"].iloc[0]

    assert milk_qty == 2283.0
    assert cheese_qty == pytest.approx(152.2, rel=1e-9)


def test_production_drivers_derive_meat_and_pelt_from_slaughter():
    schedule = pd.DataFrame(
        {"Herd Size (heads)": [100.0]},
        index=pd.to_datetime(["2026-03-31"]),
    )
    pricing = pd.DataFrame(
        {
            "Period": ["2026-03-31", "2026-03-31"],
            "Product": ["Meat", "Pelt"],
            "Active": [True, True],
            "Allocation %": [100.0, 100.0],
            "Quantity Mode": ["Derived", "Derived"],
            "Manual Quantity Override": [pd.NA, pd.NA],
            "Quantity per Period": [0.0, 0.0],
            "Unit": ["Kg", "Piece"],
            "Base Price": [9.0, 4.0],
            "Price Growth %": [0.0, 0.0],
        }
    )
    drivers = pd.DataFrame(
        {
            "Product": ["Meat", "Pelt"],
            "Unit": ["Kg", "Piece"],
            "Quantity Mode": ["Derived", "Derived"],
            "Lactating Herd Share %": [0.0, 0.0],
            "Litres per Lactating Doe per Day": [0.0, 0.0],
            "Milk Allocation to Cheese %": [0.0, 0.0],
            "Cheese Yield Kg per Litre": [0.0, 0.0],
            "Annual Slaughter Rate % of Herd": [5.0, 5.0],
            "Meat Yield Kg per Goat": [20.0, 0.0],
            "Pelt Units per Goat": [0.0, 1.0],
            "Driver Growth %": [0.0, 0.0],
        }
    )

    derived = streamlit_app._derive_pricing_quantities_from_production(
        pricing, schedule, drivers
    )

    expected_saleable_goats = 100.0 * (1.0 - (1.0 - 0.05) ** (1.0 / 12.0))
    assert derived.loc[derived["Product"] == "Meat", "Quantity per Period"].iloc[0] == pytest.approx(
        expected_saleable_goats * 20.0
    )
    assert derived.loc[derived["Product"] == "Pelt", "Quantity per Period"].iloc[0] == pytest.approx(
        expected_saleable_goats
    )


def test_production_drivers_derive_offal_and_live_herd_from_saleable_stream():
    schedule = pd.DataFrame(
        {"Herd Size (heads)": [100.0]},
        index=pd.to_datetime(["2026-03-31"]),
    )
    pricing = pd.DataFrame(
        {
            "Period": ["2026-03-31"] * 4,
            "Product": ["Meat", "Offal", "Pelt", "Live Herd"],
            "Active": [True, True, True, True],
            "Allocation %": [100.0, 100.0, 100.0, 100.0],
            "Quantity Mode": ["Derived", "Derived", "Derived", "Derived"],
            "Manual Quantity Override": [pd.NA, pd.NA, pd.NA, pd.NA],
            "Quantity per Period": [0.0, 0.0, 0.0, 0.0],
            "Unit": ["Kg", "Kg", "Piece", "Head"],
            "Base Price": [10.0, 4.0, 3.0, 85.0],
            "Price Growth %": [0.0, 0.0, 0.0, 0.0],
        }
    )
    drivers = pd.DataFrame(
        {
            "Product": ["Meat", "Offal", "Pelt", "Live Herd"],
            "Unit": ["Kg", "Kg", "Piece", "Head"],
            "Quantity Mode": ["Derived", "Derived", "Derived", "Derived"],
            "Lactating Herd Share %": [0.0, 0.0, 0.0, 0.0],
            "Litres per Lactating Doe per Day": [0.0, 0.0, 0.0, 0.0],
            "Milk Allocation to Cheese %": [0.0, 0.0, 0.0, 0.0],
            "Cheese Yield Kg per Litre": [0.0, 0.0, 0.0, 0.0],
            "Annual Slaughter Rate % of Herd": [5.0, 5.0, 5.0, 5.0],
            "Live Herd Sales Share %": [20.0, 20.0, 20.0, 20.0],
            "Meat Yield Kg per Goat": [20.0, 0.0, 0.0, 0.0],
            "Offal Yield Kg per Goat": [0.0, 4.0, 0.0, 0.0],
            "Pelt Units per Goat": [0.0, 0.0, 1.0, 0.0],
            "Driver Growth %": [0.0, 0.0, 0.0, 0.0],
        }
    )

    derived = streamlit_app._derive_pricing_quantities_from_production(
        pricing, schedule, drivers
    )

    expected_saleable_goats = 100.0 * (1.0 - (1.0 - 0.05) ** (1.0 / 12.0))
    expected_live_herd = expected_saleable_goats * 0.20
    expected_slaughter_heads = expected_saleable_goats - expected_live_herd
    assert derived.loc[derived["Product"] == "Meat", "Quantity per Period"].iloc[0] == pytest.approx(
        expected_slaughter_heads * 20.0
    )
    assert derived.loc[derived["Product"] == "Offal", "Quantity per Period"].iloc[0] == pytest.approx(
        expected_slaughter_heads * 4.0
    )
    assert derived.loc[derived["Product"] == "Pelt", "Quantity per Period"].iloc[0] == pytest.approx(
        expected_slaughter_heads
    )
    assert derived.loc[derived["Product"] == "Live Herd", "Quantity per Period"].iloc[0] == pytest.approx(
        expected_live_herd
    )


def test_add_production_driver_column_preserves_core_schema():
    drivers = streamlit_app._default_production_driver_table()

    updated = streamlit_app._add_production_driver_column(drivers, "Benchmark Note")

    assert "Benchmark Note" in updated.columns
    assert {
        "Product",
        "Quantity Mode",
        "Lactating Herd Share %",
        "Annual Slaughter Rate % of Herd",
    }.issubset(updated.columns)


def test_remove_production_driver_columns_only_drops_custom_fields():
    drivers = streamlit_app._add_production_driver_column(
        streamlit_app._default_production_driver_table(),
        "Benchmark Note",
    )

    updated = streamlit_app._remove_production_driver_columns(
        drivers,
        ["Benchmark Note", "Product"],
    )

    assert "Benchmark Note" not in updated.columns
    assert "Product" in updated.columns


def test_merge_production_driver_subset_updates_target_products_only():
    drivers = streamlit_app._add_production_driver_column(
        streamlit_app._default_production_driver_table(),
        "Benchmark Note",
    )
    drivers.loc[drivers["Product"] == "Meat", "Benchmark Note"] = "Keep"

    dairy_subset = drivers.loc[drivers["Product"].isin(["Milk", "Cheese"])].copy()
    dairy_subset.loc[dairy_subset["Product"] == "Milk", "Lactating Herd Share %"] = 62.5
    dairy_subset.loc[dairy_subset["Product"] == "Cheese", "Benchmark Note"] = "Stretch"

    merged = streamlit_app._merge_production_driver_subset(
        drivers,
        dairy_subset,
        ["Milk", "Cheese"],
    )

    assert (
        merged.loc[merged["Product"] == "Milk", "Lactating Herd Share %"].iloc[0]
        == 62.5
    )
    assert merged.loc[merged["Product"] == "Cheese", "Benchmark Note"].iloc[0] == "Stretch"
    assert merged.loc[merged["Product"] == "Meat", "Benchmark Note"].iloc[0] == "Keep"


def test_merge_production_driver_subset_dedupes_when_product_is_reassigned():
    drivers = streamlit_app._default_production_driver_table()
    slaughter_subset = drivers.loc[drivers["Product"].isin(["Meat", "Offal", "Pelt", "Live Herd"])].copy()
    slaughter_subset.loc[slaughter_subset["Product"] == "Meat", "Product"] = "Offal"
    slaughter_subset.loc[slaughter_subset["Product"] == "Offal", "Offal Yield Kg per Goat"] = 5.0

    merged = streamlit_app._merge_production_driver_subset(
        drivers,
        slaughter_subset,
        ["Meat", "Offal", "Pelt", "Live Herd"],
    )

    assert merged["Product"].value_counts().max() == 1
    assert merged.loc[merged["Product"] == "Offal", "Offal Yield Kg per Goat"].iloc[0] == 5.0


def test_manual_quantity_override_is_preserved_over_derived_drivers():
    schedule = pd.DataFrame(
        {"Herd Size (heads)": [100.0]},
        index=pd.to_datetime(["2026-01-31"]),
    )
    pricing = pd.DataFrame(
        {
            "Period": ["2026-01-31"],
            "Product": ["Milk"],
            "Active": [True],
            "Allocation %": [100.0],
            "Quantity Mode": ["Manual Override"],
            "Manual Quantity Override": [999.0],
            "Quantity per Period": [0.0],
            "Unit": ["Litre"],
            "Base Price": [2.0],
            "Price Growth %": [0.0],
        }
    )
    drivers = streamlit_app._default_production_driver_table()

    derived = streamlit_app._derive_pricing_quantities_from_production(
        pricing, schedule, drivers
    )

    assert derived["Quantity per Period"].iloc[0] == 999.0


def test_pricing_assumptions_only_count_active_products_in_revenue():
    schedule = pd.DataFrame(
        {
            "Revenue": [0.0, 0.0],
            "COGS": [10.0, 10.0],
            "Variable Expenses": [5.0, 5.0],
            "Fixed Expenses": [2.0, 2.0],
            "Direct Wages": [3.0, 3.0],
            "Admin Wages": [1.0, 1.0],
        },
        index=pd.to_datetime(["2026-01-31", "2026-02-28"]),
    )
    pricing = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-01-31", "2026-02-28"],
            "Product": ["Milk", "Meat", "Milk"],
            "Active": [True, False, True],
            "Allocation %": [100.0, 100.0, 50.0],
            "Quantity Mode": ["Manual Override", "Manual Override", "Manual Override"],
            "Manual Quantity Override": [100.0, 40.0, 80.0],
            "Quantity per Period": [100.0, 40.0, 80.0],
            "Unit": ["Litre", "Kg", "Litre"],
            "Base Price": [2.0, 10.0, 2.0],
            "Price Growth %": [0.0, 0.0, 0.0],
        }
    )

    updated = streamlit_app._apply_pricing_assumptions_to_schedule(schedule, pricing)

    assert updated["Revenue"].tolist() == [200.0, 80.0]
    assert updated["Gross Margin"].tolist() == [190.0, 70.0]


def test_sync_pricing_table_to_core_expands_products_to_new_periods():
    core = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28"],
            "Revenue": [1000.0, 1200.0],
        }
    )
    pricing = pd.DataFrame(
        {
            "Period": ["2026-01-31"],
            "Product": ["Milk"],
            "Active": [True],
            "Allocation %": [100.0],
            "Quantity Mode": ["Manual Override"],
            "Manual Quantity Override": [400.0],
            "Quantity per Period": [400.0],
            "Unit": ["Litre"],
            "Base Price": [2.5],
            "Price Growth %": [0.0],
        }
    )

    synced = streamlit_app._sync_pricing_table_to_core(pricing, core)

    period_product_pairs = set(zip(synced["Period"], synced["Product"]))
    assert ("2026-01-31", "Milk") in period_product_pairs
    assert ("2026-02-28", "Milk") in period_product_pairs
    assert ("2026-01-31", "Meat") in period_product_pairs


def test_sync_commercial_assumptions_to_core_rebases_pricing_periods_to_horizon():
    core = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28"],
            "Revenue": [1000.0, 1200.0],
        }
    )
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Herd Plan"] = pd.DataFrame(
        {
            "Year": [2026],
            "Herd Size (heads)": [100.0],
            "Herd Growth %": [0.0],
        }
    )
    assumptions["Pricing"] = pd.DataFrame(
        {
            "Period": ["2025-12-31"],
            "Product": ["Milk"],
            "Active": [True],
            "Allocation %": [100.0],
            "Quantity Mode": ["Derived"],
            "Manual Quantity Override": [pd.NA],
            "Quantity per Period": [0.0],
            "Unit": ["Litre"],
            "Base Price": [2.5],
            "Price Growth %": [0.0],
        }
    )

    synced = streamlit_app._sync_commercial_assumptions_to_core(assumptions, core)

    assert "2025-12-31" not in synced["Pricing"]["Period"].tolist()
    assert {"2026-01-31", "2026-02-28"} == set(synced["Pricing"]["Period"].tolist())
    assert synced["Pricing"]["Quantity per Period"].fillna(0.0).ge(0.0).all()


def test_default_assumptions_include_business_configuration():
    assumptions = streamlit_app._default_assumption_tables()

    assert "Business Configuration" in assumptions
    assert streamlit_app._selected_business_type(assumptions) == "Combined"


def test_model_npv_and_irr_ignore_stored_fallback_values():
    assumptions = streamlit_app._default_assumption_tables()
    core, detail_tables = streamlit_app._default_schedule_components(
        production_horizon=assumptions.get("Production Horizon"),
        assumptions=assumptions,
    )
    schedule_df = streamlit_app._build_schedule_dataframe(core, detail_tables, assumptions)

    valuation_inputs = streamlit_app._valuation_table_to_inputs(assumptions["Valuation Inputs"])
    valuation_inputs["NPV"] = -999.0
    valuation_inputs["IRR"] = -0.99

    supplementary_tables = streamlit_app._default_supplementary_tables()
    supplementary_tables["Capital & Financing"] = streamlit_app._ensure_capital_financing_table(
        assumptions.get("Capital & Financing")
    )
    for name, table in assumptions.items():
        supplementary_tables[f"Assumptions - {name}"] = table.copy()
    supplementary_tables["Assumptions - Pricing"] = pd.DataFrame()

    model = streamlit_app.InputSchedule(
        data=schedule_df,
        valuation_inputs=valuation_inputs,
        supplementary_tables=supplementary_tables,
    ).to_model()

    summary = model.valuation_summary()

    assert model.npv() == pytest.approx(float(summary["npv"]))
    assert model.irr() == pytest.approx(float(summary["irr"]))
    assert model.npv() != pytest.approx(-999.0)
    assert model.irr() != pytest.approx(-0.99)


def test_kpis_use_computed_npv_and_irr_only():
    assumptions = streamlit_app._default_assumption_tables()
    core, detail_tables = streamlit_app._default_schedule_components(
        production_horizon=assumptions.get("Production Horizon"),
        assumptions=assumptions,
    )
    schedule_df = streamlit_app._build_schedule_dataframe(core, detail_tables, assumptions)

    valuation_inputs = streamlit_app._valuation_table_to_inputs(assumptions["Valuation Inputs"])
    valuation_inputs["NPV"] = -999.0
    valuation_inputs["IRR"] = -0.99

    supplementary_tables = streamlit_app._default_supplementary_tables()
    supplementary_tables["Capital & Financing"] = streamlit_app._ensure_capital_financing_table(
        assumptions.get("Capital & Financing")
    )
    for name, table in assumptions.items():
        supplementary_tables[f"Assumptions - {name}"] = table.copy()
    supplementary_tables["Assumptions - Pricing"] = pd.DataFrame()

    model = streamlit_app.InputSchedule(
        data=schedule_df,
        valuation_inputs=valuation_inputs,
        supplementary_tables=supplementary_tables,
    ).to_model()

    kpis = model.kpis(schedule_df, annual=False)
    summary = model.valuation_summary(schedule_df)

    assert "NPV" in kpis.columns
    assert "IRR" in kpis.columns
    assert "Terminal Value" in kpis.columns
    assert "Payback Period (Years)" in kpis.columns
    assert float(kpis["NPV"].iloc[0]) == pytest.approx(float(summary["npv"]))
    assert float(kpis["IRR"].iloc[0]) == pytest.approx(float(summary["irr"]))
    assert float(kpis["Terminal Value"].iloc[0]) == pytest.approx(float(summary["terminal_value"]))
    assert float(kpis["Payback Period (Years)"].iloc[0]) == pytest.approx(float(summary["payback_years"]))
    assert float(kpis["NPV"].iloc[0]) != pytest.approx(-999.0)
    assert float(kpis["IRR"].iloc[0]) != pytest.approx(-0.99)


def test_valuation_diagnostic_messages_explain_missing_irr_and_payback():
    model = streamlit_app.GoatModel(
        pd.DataFrame(
            {
                "Revenue": [0.0, 0.0, 0.0],
                "COGS": [0.0, 0.0, 0.0],
                "EBITDA": [-100.0, -40.0, -10.0],
                "Depreciation & Amortization": [0.0, 0.0, 0.0],
                "EBIT": [-100.0, -40.0, -10.0],
                "Interest Expense": [0.0, 0.0, 0.0],
                "NPBT": [-100.0, -40.0, -10.0],
                "Tax Expense": [0.0, 0.0, 0.0],
                "NPAT": [-100.0, -40.0, -10.0],
                "CFO": [-100.0, -40.0, -10.0],
                "CFI": [0.0, 0.0, 0.0],
                "CFF": [0.0, 0.0, 0.0],
            },
            index=pd.date_range("2024-12-31", periods=3, freq="Y"),
        ),
        valuation_inputs={"WACC": 0.1, "Terminal Value": 0.0},
    )

    messages = streamlit_app._valuation_diagnostic_messages(model)

    assert any("IRR is mathematically unavailable" in message for message in messages)
    assert any("Payback Period is mathematically unavailable" in message for message in messages)


def test_sync_commercial_assumptions_filters_rows_to_business_type():
    core = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28"],
            "Revenue": [1000.0, 1200.0],
        }
    )
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Business Configuration"] = pd.DataFrame({"Business Type": ["Meat"]})

    synced = streamlit_app._sync_commercial_assumptions_to_core(assumptions, core)

    assert set(synced["Pricing"]["Product"].unique()) == {"Meat", "Offal", "Pelt", "Live Herd"}
    assert set(synced["Production Drivers"]["Product"].unique()) == {"Meat", "Offal", "Pelt", "Live Herd"}
    scenario_drivers = set(synced["Scenario Controls"]["Driver"].tolist())
    assert "Milk price change (%)" not in scenario_drivers
    assert "Meat price change (%)" in scenario_drivers
    assert "Feed cost change (%)" in scenario_drivers


def test_build_schedule_dataframe_rebases_commercial_periods_to_core_schedule():
    assumptions = streamlit_app._default_assumption_tables()
    core = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28"],
            "Revenue": [0.0, 0.0],
            "COGS": [0.0, 0.0],
            "Variable Expenses": [0.0, 0.0],
            "Fixed Expenses": [0.0, 0.0],
            "Direct Wages": [0.0, 0.0],
            "Admin Wages": [0.0, 0.0],
        }
    )
    assumptions["Herd Plan"] = pd.DataFrame(
        {
            "Year": [2026],
            "Herd Size (heads)": [100.0],
            "Herd Growth %": [0.0],
        }
    )
    assumptions["Pricing"] = pd.DataFrame(
        {
            "Period": ["2025-12-31"],
            "Product": ["Milk"],
            "Active": [True],
            "Allocation %": [100.0],
            "Quantity Mode": ["Manual Override"],
            "Manual Quantity Override": [50.0],
            "Quantity per Period": [50.0],
            "Unit": ["Litre"],
            "Base Price": [2.0],
            "Price Growth %": [0.0],
        }
    )

    built = streamlit_app._build_schedule_dataframe(core, {}, assumptions)

    built_periods = pd.to_datetime(built.index if built.index.name == "Period" else built["Period"], errors="coerce")
    assert set(built_periods.strftime("%Y-%m-%d").tolist()) == {"2026-01-31", "2026-02-28"}


def test_pricing_product_plan_can_target_a_period_range_only():
    pricing = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28", "2026-03-31"],
            "Product": ["Milk", "Milk", "Milk"],
            "Active": [False, False, False],
            "Allocation %": [0.0, 0.0, 0.0],
            "Quantity Mode": ["Derived", "Derived", "Derived"],
            "Manual Quantity Override": [pd.NA, pd.NA, pd.NA],
            "Quantity per Period": [0.0, 0.0, 0.0],
            "Unit": ["Litre", "Litre", "Litre"],
            "Base Price": [2.0, 2.0, 2.0],
            "Price Growth %": [0.0, 0.0, 0.0],
        }
    )

    updated = streamlit_app._apply_pricing_product_plan(
        pricing,
        "Milk",
        active=True,
        allocation_pct=100.0,
        quantity_mode="Manual Override",
        base_quantity=50.0,
        yearly_growth_pct=0.0,
        period_start="2026-02-28",
        period_end="2026-03-31",
    )

    assert updated["Active"].tolist() == [False, True, True]
    assert updated["Manual Quantity Override"].fillna(0.0).tolist() == [0.0, 50.0, 50.0]


def test_pricing_validation_messages_flag_inactive_quantities_and_zero_prices():
    pricing = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-01-31"],
            "Product": ["Milk", "Cheese"],
            "Active": [False, True],
            "Allocation %": [100.0, 100.0],
            "Quantity Mode": ["Manual Override", "Manual Override"],
            "Manual Quantity Override": [25.0, 10.0],
            "Quantity per Period": [25.0, 10.0],
            "Unit": ["Litre", "Kg"],
            "Base Price": [2.0, 0.0],
            "Price Growth %": [0.0, 0.0],
        }
    )

    issues = streamlit_app._pricing_validation_messages(
        pricing,
        streamlit_app._default_production_driver_table(),
    )

    assert any("Inactive products still carry quantities" in issue for issue in issues)
    assert any("zero or missing prices" in issue for issue in issues)


def test_commercial_shocks_apply_to_multiple_products():
    schedule = pd.DataFrame(
        {"Herd Size (heads)": [100.0]},
        index=pd.to_datetime(["2026-03-31"]),
    )
    pricing = pd.DataFrame(
        {
            "Period": ["2026-03-31", "2026-03-31"],
            "Product": ["Meat", "Pelt"],
            "Active": [True, True],
            "Allocation %": [100.0, 100.0],
            "Quantity Mode": ["Derived", "Derived"],
            "Manual Quantity Override": [pd.NA, pd.NA],
            "Quantity per Period": [0.0, 0.0],
            "Unit": ["Kg", "Piece"],
            "Base Price": [10.0, 4.0],
            "Price Growth %": [0.0, 0.0],
        }
    )
    drivers = pd.DataFrame(
        {
            "Product": ["Meat", "Pelt"],
            "Unit": ["Kg", "Piece"],
            "Quantity Mode": ["Derived", "Derived"],
            "Lactating Herd Share %": [0.0, 0.0],
            "Litres per Lactating Doe per Day": [0.0, 0.0],
            "Milk Allocation to Cheese %": [0.0, 0.0],
            "Cheese Yield Kg per Litre": [0.0, 0.0],
            "Annual Slaughter Rate % of Herd": [5.0, 5.0],
            "Meat Yield Kg per Goat": [20.0, 0.0],
            "Pelt Units per Goat": [0.0, 1.0],
            "Driver Growth %": [0.0, 0.0],
        }
    )

    shocked = streamlit_app._apply_commercial_shocks_to_pricing(
        pricing,
        schedule,
        drivers,
        {
            "Meat price change (%)": 10.0,
            "Meat quantity change (%)": 20.0,
            "Pelt price change (%)": -25.0,
        },
    )

    meat_row = shocked.loc[shocked["Product"] == "Meat"].iloc[0]
    pelt_row = shocked.loc[shocked["Product"] == "Pelt"].iloc[0]

    expected_saleable_goats = 100.0 * (1.0 - (1.0 - 0.05) ** (1.0 / 12.0))
    assert meat_row["Quantity per Period"] == pytest.approx(expected_saleable_goats * 1.2 * 20.0)
    assert meat_row["Base Price"] == pytest.approx(11.0)
    assert meat_row["Revenue"] == pytest.approx(meat_row["Quantity per Period"] * 11.0)
    assert pelt_row["Base Price"] == pytest.approx(3.0)


def test_standalone_app_bootstraps_and_runs_without_top_level_exceptions():
    repo_root = _STREAMLIT_APP_PATH.parent
    command = [
        sys.executable,
        "-c",
        (
            "from streamlit.testing.v1 import AppTest; "
            "at = AppTest.from_file('streamlit_app.py'); "
            "at.run(timeout=30); "
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


def test_assumptions_run_model_button_executes_without_duplicate_period_error():
    repo_root = _STREAMLIT_APP_PATH.parent
    command = [
        sys.executable,
        "-c",
        (
            "from streamlit.testing.v1 import AppTest; "
            "at = AppTest.from_file('streamlit_app.py'); "
            "at.run(timeout=60); "
            "run_buttons = [b for b in at.button if getattr(b, 'label', '') == 'Run model']; "
            "assert run_buttons, 'Run model button not found'; "
            "run_buttons[0].click(); "
            "at.run(timeout=180); "
            "errors = [getattr(x, 'value', '') for x in at.error]; "
            "has_results = 'results' in at.session_state; "
            "results = at.session_state['results'] if has_results else None; "
            "print('exc_count', len(at.exception)); "
            "print('errors', errors); "
            "print('has_results', has_results); "
            "print('results_type', type(results).__name__ if has_results else 'missing'); "
            "raise SystemExit(0 if len(at.exception) == 0 and has_results and isinstance(results, dict) and not any('Each period in the schedule must be unique.' in err for err in errors) else 1)"
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
    assert "has_results True" in completed.stdout


class _FakeExcelDownloadStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.button_labels: list[str] = []
        self.download_labels: list[str] = []
        self.info_messages: list[str] = []

    def markdown(self, *args: object, **kwargs: object) -> None:
        return None

    def info(self, message: str, **kwargs: object) -> None:
        self.info_messages.append(message)

    def button(self, label: str, **kwargs: object) -> bool:
        self.button_labels.append(label)
        return False

    def download_button(self, label: str, **kwargs: object) -> bool:
        self.download_labels.append(label)
        return False

    def spinner(self, *args: object, **kwargs: object):
        return nullcontext()


def test_excel_download_panel_shows_prepare_action_when_results_exist(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeExcelDownloadStreamlit()
    monkeypatch.setattr(streamlit_app, "st", fake_st)

    streamlit_app._render_excel_download_panel(
        nullcontext(),
        {"selected_scenario": "Base Case Scenario", "model": object()},
        False,
    )

    assert "Prepare Excel Model" in fake_st.button_labels


def test_model_input_fingerprint_is_stable_for_equivalent_tables():
    payload_a = {
        "schedule": pd.DataFrame({"Year": [2026, 2027], "Revenue": [10.0, 12.0]}),
        "settings": {"scenario": "Base", "enabled": True},
    }
    payload_b = {
        "settings": {"enabled": True, "scenario": "Base"},
        "schedule": pd.DataFrame({"Year": [2026, 2027], "Revenue": [10.0, 12.0]}),
    }

    assert streamlit_app._model_input_fingerprint(payload_a) == streamlit_app._model_input_fingerprint(payload_b)


def test_model_input_fingerprint_changes_when_a_model_input_changes():
    base = {"schedule": pd.DataFrame({"Revenue": [10.0, 12.0]})}
    changed = {"schedule": pd.DataFrame({"Revenue": [10.0, 13.0]})}

    assert streamlit_app._model_input_fingerprint(base) != streamlit_app._model_input_fingerprint(changed)
