import importlib.util
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


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
        "Slaughter Rate % of Herd per Period",
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
            "Slaughter Rate % of Herd per Period": [0.0, 0.0],
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
            "Slaughter Rate % of Herd per Period": [5.0, 5.0],
            "Meat Yield Kg per Goat": [20.0, 0.0],
            "Pelt Units per Goat": [0.0, 1.0],
            "Driver Growth %": [0.0, 0.0],
        }
    )

    derived = streamlit_app._derive_pricing_quantities_from_production(
        pricing, schedule, drivers
    )

    assert derived.loc[derived["Product"] == "Meat", "Quantity per Period"].iloc[0] == 100.0
    assert derived.loc[derived["Product"] == "Pelt", "Quantity per Period"].iloc[0] == 5.0


def test_add_production_driver_column_preserves_core_schema():
    drivers = streamlit_app._default_production_driver_table()

    updated = streamlit_app._add_production_driver_column(drivers, "Benchmark Note")

    assert "Benchmark Note" in updated.columns
    assert {
        "Product",
        "Quantity Mode",
        "Lactating Herd Share %",
        "Slaughter Rate % of Herd per Period",
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
            "Slaughter Rate % of Herd per Period": [5.0, 5.0],
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

    assert meat_row["Quantity per Period"] == pytest.approx(120.0)
    assert meat_row["Base Price"] == pytest.approx(11.0)
    assert meat_row["Revenue"] == pytest.approx(1320.0)
    assert pelt_row["Base Price"] == pytest.approx(3.0)


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
