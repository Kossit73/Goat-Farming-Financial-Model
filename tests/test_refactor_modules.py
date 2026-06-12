from __future__ import annotations

import pandas as pd
import pytest

import streamlit_app
from goat_financial_model.assumption_bundle import build_assumption_bundle
from goat_financial_model.commercial_services import (
    build_pricing_validation_messages,
    sync_commercial_assumptions_to_core,
)
from goat_financial_model.pages import assumptions_page, input_schedule_page
from goat_financial_model.pages.assumptions_page import (
    BiologicalEditorDefinition,
    render_biological_assumption_editor,
)
from goat_financial_model.pages.input_schedule_page import render_cogs_schedule_editor
from goat_financial_model.pricing_engine import build_pricing_context, derive_pricing_quantities
from goat_financial_model.scenario_runner import (
    ScenarioBuildHooks,
    ScenarioOutputSpec,
    collect_supplementary_outputs,
    run_scenario_suite,
)
from goat_financial_model.table_registry import ColumnSchema, TableSchema, build_default_table, ensure_table


def test_table_registry_restores_missing_columns_and_defaults() -> None:
    schema = TableSchema(
        name="Demo",
        columns=(
            ColumnSchema("Name", ""),
            ColumnSchema("Value", 0.0),
        ),
        default_rows=({"Name": "Base", "Value": 1.0},),
    )

    restored = ensure_table(schema, pd.DataFrame({"Name": ["Custom"]}))

    assert restored.columns.tolist() == ["Name", "Value"]
    assert restored.iloc[0]["Name"] == "Custom"
    assert restored.iloc[0]["Value"] == 0.0
    assert build_default_table(schema).iloc[0]["Name"] == "Base"


def test_assumption_bundle_groups_sections() -> None:
    ensure_map = {
        "Biological System Settings": lambda table: table if table is not None else pd.DataFrame({"Setting": [], "Value": []}),
        "Pricing": lambda table: table if table is not None else pd.DataFrame(),
        "Production Drivers": lambda table: table if table is not None else pd.DataFrame(),
        "Scenario Controls": lambda table: table if table is not None else pd.DataFrame(),
        "Capital & Financing": lambda table: table if table is not None else pd.DataFrame(),
        "Loan Facilities": lambda table: table if table is not None else pd.DataFrame(),
        "Equity Facilities": lambda table: table if table is not None else pd.DataFrame(),
        "Valuation Inputs": lambda table: table if table is not None else pd.DataFrame(),
        "Breeding & Reproduction Biology": lambda table: table if table is not None else pd.DataFrame(),
        "Lactation Biology": lambda table: table if table is not None else pd.DataFrame(),
        "Finishing & Slaughter Biology": lambda table: table if table is not None else pd.DataFrame(),
        "Opening Herd Cohorts": lambda table: table if table is not None else pd.DataFrame(),
        "Cohort Allocation Rules": lambda table: table if table is not None else pd.DataFrame(),
        "Biological Cost Drivers": lambda table: table if table is not None else pd.DataFrame(),
    }
    assumptions = {
        "Biological System Settings": pd.DataFrame({"Setting": ["Model Grain"], "Value": ["monthly"]}),
        "Pricing": pd.DataFrame({"Product": ["Milk"]}),
    }

    bundle = build_assumption_bundle(assumptions, ensure_map)

    assert bundle.biological.system_settings.iloc[0]["Setting"] == "Model Grain"
    assert bundle.commercial.pricing.iloc[0]["Product"] == "Milk"
    assert bundle.get("Pricing").equals(bundle.commercial.pricing)


def test_pricing_engine_uses_biological_quantities_when_available() -> None:
    pricing = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-01-31"],
            "Product": ["Milk", "Meat"],
            "Active": [True, True],
            "Allocation %": [100.0, 100.0],
            "Quantity Mode": ["Derived", "Derived"],
            "Manual Quantity Override": [None, None],
            "Quantity per Period": [0.0, 0.0],
            "Unit": ["L", "Kg"],
            "Base Price": [1.0, 1.0],
            "Price Growth %": [0.0, 0.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "Herd Size (heads)": [100.0],
            "Milk Production (L)": [250.0],
            "Meat Output Kg": [90.0],
            "Slaughter Heads": [5.0],
            "Live Herd Sales (heads)": [0.0],
        },
        index=pd.DatetimeIndex(["2024-01-31"]),
    )
    driver_lookup = {
        "Milk": {
            "Lactating Herd Share %": 20.0,
            "Litres per Lactating Doe per Day": 1.0,
            "Driver Growth %": 0.0,
        },
        "Meat": {
            "Meat Yield Kg per Goat": 10.0,
            "Driver Growth %": 0.0,
            "Annual Slaughter Rate % of Herd": 5.0,
        },
    }

    context = build_pricing_context(schedule, driver_lookup, ("Meat", "Offal", "Pelt", "Live Herd"))
    derived = derive_pricing_quantities(pricing, context)

    milk_qty = derived.loc[derived["Product"] == "Milk", "Quantity per Period"].iloc[0]
    meat_qty = derived.loc[derived["Product"] == "Meat", "Quantity per Period"].iloc[0]
    assert milk_qty == pytest.approx(250.0)
    assert meat_qty == pytest.approx(90.0)


def test_apply_product_base_price_updates_all_rows_for_selected_product() -> None:
    pricing = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-02-29", "2024-01-31"],
            "Product": ["Milk", "Milk", "Cheese"],
            "Base Price": [4.0, 4.1, 30.0],
        }
    )

    updated = assumptions_page._apply_product_base_price(pricing, "Milk", 4.75)

    assert updated.loc[updated["Product"] == "Milk", "Base Price"].tolist() == [4.75, 4.75]
    assert updated.loc[updated["Product"] == "Cheese", "Base Price"].tolist() == [30.0]


def test_apply_product_base_price_is_noop_when_required_columns_are_missing() -> None:
    pricing = pd.DataFrame({"Product": ["Milk"]})

    updated = assumptions_page._apply_product_base_price(pricing, "Milk", 4.75)

    assert updated.equals(pricing)


def test_apply_product_pricing_updates_updates_all_supported_fields() -> None:
    pricing = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-02-29", "2024-01-31"],
            "Product": ["Milk", "Milk", "Cheese"],
            "Active": [True, True, True],
            "Allocation %": [100.0, 100.0, 100.0],
            "Quantity Mode": ["Derived", "Derived", "Derived"],
            "Manual Quantity Override": [pd.NA, pd.NA, pd.NA],
            "Quantity per Period": [50.0, 55.0, 10.0],
            "Unit": ["Litre", "Litre", "Kg"],
            "Base Price": [4.0, 4.1, 30.0],
            "Price Growth %": [1.0, 1.0, 2.0],
            "Revenue": [200.0, 225.5, 300.0],
        }
    )

    updated = assumptions_page._apply_product_pricing_updates(
        pricing,
        "Milk",
        active=False,
        allocation_pct=65.0,
        quantity_mode="Manual Override",
        manual_quantity_override=120.0,
        unit="Bottle",
        base_price=5.5,
        price_growth_pct=3.0,
    )

    milk_rows = updated.loc[updated["Product"] == "Milk"]
    cheese_row = updated.loc[updated["Product"] == "Cheese"].iloc[0]

    assert milk_rows["Active"].tolist() == [False, False]
    assert milk_rows["Allocation %"].tolist() == [0.0, 0.0]
    assert milk_rows["Quantity Mode"].tolist() == ["Manual Override", "Manual Override"]
    assert milk_rows["Manual Quantity Override"].tolist() == [120.0, 120.0]
    assert milk_rows["Unit"].tolist() == ["Bottle", "Bottle"]
    assert milk_rows["Base Price"].tolist() == [5.5, 5.5]
    assert milk_rows["Price Growth %"].tolist() == [3.0, 3.0]
    assert cheese_row["Base Price"] == 30.0


def test_apply_product_pricing_updates_clears_manual_override_for_derived_mode() -> None:
    pricing = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-02-29"],
            "Product": ["Milk", "Milk"],
            "Active": [True, True],
            "Allocation %": [100.0, 100.0],
            "Quantity Mode": ["Manual Override", "Manual Override"],
            "Manual Quantity Override": [80.0, 90.0],
            "Quantity per Period": [80.0, 90.0],
            "Unit": ["Litre", "Litre"],
            "Base Price": [4.0, 4.0],
            "Price Growth %": [1.0, 1.0],
            "Revenue": [320.0, 360.0],
        }
    )

    updated = assumptions_page._apply_product_pricing_updates(
        pricing,
        "Milk",
        active=True,
        allocation_pct=75.0,
        quantity_mode="Derived",
        manual_quantity_override=120.0,
        unit="Litre",
        base_price=4.5,
        price_growth_pct=2.5,
    )

    assert updated["Allocation %"].tolist() == [75.0, 75.0]
    assert updated["Quantity Mode"].tolist() == ["Derived", "Derived"]
    assert updated["Manual Quantity Override"].isna().all()


def test_apply_product_pricing_updates_is_noop_when_product_column_missing() -> None:
    pricing = pd.DataFrame({"Base Price": [4.0]})

    updated = assumptions_page._apply_product_pricing_updates(
        pricing,
        "Milk",
        active=True,
        allocation_pct=100.0,
        quantity_mode="Derived",
        manual_quantity_override=None,
        unit="Litre",
        base_price=4.5,
        price_growth_pct=2.5,
    )

    assert updated.equals(pricing)


def test_commercial_service_sync_matches_existing_streamlit_wrapper() -> None:
    core = pd.DataFrame(
        {
            "Period": ["2026-01-31", "2026-02-28"],
            "Revenue": [1000.0, 1200.0],
        }
    )
    assumptions = streamlit_app._default_assumption_tables()
    assumptions["Business Configuration"] = pd.DataFrame({"Business Type": ["Meat"]})

    service_synced = sync_commercial_assumptions_to_core(
        assumptions,
        core,
        ensure_business_configuration_table=streamlit_app._ensure_business_configuration_table,
        selected_business_type=streamlit_app._selected_business_type,
        active_products_for_business_type=streamlit_app._active_products_for_business_type,
        build_app_assumption_bundle=streamlit_app._build_app_assumption_bundle,
        sync_production_driver_table_to_products=streamlit_app._sync_production_driver_table_to_products,
        sync_scenario_controls_to_products=streamlit_app._sync_scenario_controls_to_products,
        sync_pricing_table_to_core=streamlit_app._sync_pricing_table_to_core,
        derive_pricing_quantities_from_production=streamlit_app._derive_pricing_quantities_from_production,
        pricing_schedule_context=streamlit_app._pricing_schedule_context,
    )
    wrapper_synced = streamlit_app._sync_commercial_assumptions_to_core(assumptions, core)

    assert service_synced["Pricing"].equals(wrapper_synced["Pricing"])
    assert service_synced["Production Drivers"].equals(wrapper_synced["Production Drivers"])
    assert service_synced["Scenario Controls"].equals(wrapper_synced["Scenario Controls"])


def test_pricing_validation_service_matches_existing_streamlit_wrapper() -> None:
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
    production_drivers = streamlit_app._default_production_driver_table()

    service_issues = build_pricing_validation_messages(
        pricing,
        production_drivers,
        ensure_pricing_table=streamlit_app._ensure_pricing_table,
        ensure_production_driver_table=streamlit_app._ensure_production_driver_table,
        product_family_label=streamlit_app._product_family_label,
        slaughter_products=streamlit_app._PRODUCTION_DRIVER_SLAUGHTER_PRODUCTS,
    )
    wrapper_issues = streamlit_app._pricing_validation_messages(pricing, production_drivers)

    assert service_issues == wrapper_issues


class _FakeColumn:
    def __init__(self, streamlit: "_FakeStreamlit") -> None:
        self._streamlit = streamlit

    def button(self, label: str, key: str | None = None, **kwargs: object) -> bool:
        return self._streamlit.button(label, key=key, **kwargs)

    def number_input(self, label: str, key: str | None = None, **kwargs: object) -> float:
        return self._streamlit.number_input(label, key=key, **kwargs)

    def selectbox(
        self,
        label: str,
        options: list[object],
        key: str | None = None,
        index: int = 0,
        **kwargs: object,
    ) -> object:
        return self._streamlit.selectbox(label, options=options, key=key, index=index, **kwargs)


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self._button_returns: dict[str, bool] = {}
        self._selectbox_values: dict[str, object] = {}
        self._number_input_values: dict[str, float] = {}

    def columns(self, spec: int | list[int] | tuple[int, ...]) -> list[_FakeColumn]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_FakeColumn(self) for _ in range(count)]

    def selectbox(
        self,
        label: str,
        options: list[object],
        key: str | None = None,
        index: int = 0,
        **kwargs: object,
    ) -> object:
        value = self._selectbox_values.get(key or label, options[index])
        if key is not None:
            self.session_state[key] = value
        return value

    def number_input(
        self,
        label: str,
        key: str | None = None,
        min_value: float | None = None,
        **kwargs: object,
    ) -> float:
        default = float(min_value if min_value is not None else 0.0)
        value = float(self._number_input_values.get(key or label, default))
        if key is not None:
            self.session_state[key] = value
        return value

    def button(self, label: str, key: str | None = None, **kwargs: object) -> bool:
        return bool(self._button_returns.get(key or label, False))

    def markdown(self, *args: object, **kwargs: object) -> None:
        return None

    def caption(self, *args: object, **kwargs: object) -> None:
        return None

    def info(self, *args: object, **kwargs: object) -> None:
        return None

    def warning(self, *args: object, **kwargs: object) -> None:
        return None

    def success(self, *args: object, **kwargs: object) -> None:
        return None

    def dataframe(self, *args: object, **kwargs: object) -> None:
        return None

    def data_editor(self, data: pd.DataFrame, **kwargs: object) -> pd.DataFrame:
        return data


def test_render_biological_assumption_editor_updates_selected_table(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeStreamlit()
    fake_st._selectbox_values["biological_editor_name"] = "Lactation Biology"
    monkeypatch.setattr(assumptions_page, "st", fake_st)

    def _ensure_table(table: pd.DataFrame | None) -> pd.DataFrame:
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table
        return pd.DataFrame({"Setting": ["Default"], "Value": ["base"]})

    def _render_row_editor(
        key: str,
        table: pd.DataFrame,
        save_fn: callable,
    ) -> None:
        save_fn(pd.DataFrame({"Setting": ["Updated"], "Value": ["custom"]}))

    assumptions = {"Biological System Settings": pd.DataFrame({"Setting": ["Model Grain"], "Value": ["monthly"]})}
    definitions = [
        BiologicalEditorDefinition(
            "Biological System Settings",
            "System settings",
            _ensure_table,
            "assump::bio_system",
        ),
        BiologicalEditorDefinition(
            "Lactation Biology",
            "Lactation drivers",
            _ensure_table,
            "assump::lactation",
        ),
    ]

    updated = render_biological_assumption_editor(
        definitions=definitions,
        assumptions=assumptions,
        render_row_editor=_render_row_editor,
    )

    assert "Lactation Biology" in updated
    assert updated["Lactation Biology"].iloc[0]["Setting"] == "Updated"
    assert updated["Biological System Settings"].iloc[0]["Setting"] == "Model Grain"


def test_render_cogs_schedule_editor_syncs_and_saves(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _FakeStreamlit()
    fake_st._number_input_values["cogs_pct_input"] = 45.0
    monkeypatch.setattr(input_schedule_page, "st", fake_st)

    saved_tables: list[pd.DataFrame] = []

    def _save_table(updated: pd.DataFrame) -> None:
        saved_tables.append(updated.copy())

    def _render_row_editor(
        key: str,
        table: pd.DataFrame,
        save_fn: callable,
    ) -> None:
        save_fn(table.assign(Source="editor"))

    cogs_table = pd.DataFrame({"Period": ["2026-01-31"], "COGS %": [45.0], "COGS": [100.0]})
    core_schedule = pd.DataFrame({"Revenue": [200.0]}, index=pd.to_datetime(["2026-01-31"]))

    rendered = render_cogs_schedule_editor(
        cogs_table=cogs_table,
        core_schedule=core_schedule,
        save_table=_save_table,
        render_row_editor=_render_row_editor,
        clear_editor_state=lambda key: None,
        apply_pct_fn=lambda table, core, pct: table.assign(**{"COGS %": pct}),
        apply_increment_fn=lambda table, core, pct, default_pct: table.assign(**{"COGS %": default_pct + pct}),
        add_row_fn=lambda table, core, pct: pd.concat([table, table], ignore_index=True),
        remove_row_fn=lambda table, period: table.iloc[0:0],
        sync_fn=lambda table, core, pct: table.assign(Synced=pct),
        ensure_fn=lambda table, core, pct: table.assign(Ensured=pct),
    )

    assert rendered.iloc[0]["Synced"] == pytest.approx(45.0)
    assert saved_tables
    assert "Ensured" in saved_tables[-1].columns


def test_collect_supplementary_outputs_filters_empty_dataframes() -> None:
    outputs = collect_supplementary_outputs(
        [
            ScenarioOutputSpec("Keep", lambda ctx: pd.DataFrame({"Value": [1]})),
            ScenarioOutputSpec("Drop", lambda ctx: pd.DataFrame()),
        ],
        {"scenario": "Base"},
    )

    assert list(outputs) == ["Keep"]


def test_run_scenario_suite_uses_registry_hooks() -> None:
    class FakeModel:
        def __init__(self, data: pd.DataFrame) -> None:
            self.data = data.copy()

        def to_tidy(self) -> pd.DataFrame:
            return self.data.assign(BaseBuilt=True)

        def scenario(self, milk_price_pct: float, feed_cost_pct: float) -> pd.DataFrame:
            return self.data.assign(ScenarioFeedPct=feed_cost_pct)

        def valuation_summary(self, scenario_df: pd.DataFrame) -> dict[str, float]:
            return {"npv": 123.0}

        def model_audit(self, scenario_df: pd.DataFrame, annual: bool = True) -> dict[str, pd.DataFrame]:
            return {
                "summary": pd.DataFrame([{"Score": 100}]),
                "issues": pd.DataFrame(),
            }

        def kpis(self, scenario_df: pd.DataFrame, annual: bool = True) -> pd.DataFrame:
            return pd.DataFrame({"Metric": ["NPV"], "Value": [123.0]})

        def debt_schedule(self, annual: bool = False) -> pd.DataFrame:
            return pd.DataFrame({"Period": ["2026"], "Debt": [1.0]})

        def equity_schedule(self, annual: bool = False) -> pd.DataFrame:
            return pd.DataFrame({"Period": ["2026"], "Equity": [2.0]})

        def working_capital_schedule(self, scenario_df: pd.DataFrame, annual: bool = False) -> pd.DataFrame:
            return pd.DataFrame({"Period": ["2026"], "Working Capital": [3.0]})

        def debt_capacity_schedule(self, scenario_df: pd.DataFrame, annual: bool = False) -> pd.DataFrame:
            return pd.DataFrame({"Period": ["2026"], "DSCR": [1.5]})

        def ufcf_schedule(self, scenario_df: pd.DataFrame, annual: bool = False) -> pd.DataFrame:
            return pd.DataFrame({"Period": ["2026"], "UFCF": [4.0]})

        def break_even(self, scenario_df: pd.DataFrame, annual: bool = True) -> pd.DataFrame:
            return pd.DataFrame({"Period": ["2026"], "Break Even": [5.0]})

    class FakeInputSchedule:
        built_frames: list[pd.DataFrame] = []
        built_supplementaries: list[dict[str, pd.DataFrame]] = []

        def __init__(self, data: pd.DataFrame, valuation_inputs: dict[str, float], supplementary_tables: dict[str, pd.DataFrame]) -> None:
            self.data = data.copy()
            self.valuation_inputs = valuation_inputs
            self.supplementary_tables = supplementary_tables
            FakeInputSchedule.built_frames.append(self.data.copy())
            FakeInputSchedule.built_supplementaries.append(
                {
                    name: table.copy()
                    for name, table in supplementary_tables.items()
                    if isinstance(table, pd.DataFrame)
                }
            )

        def to_model(self) -> FakeModel:
            return FakeModel(self.data)

    schedule_df = pd.DataFrame({"Revenue": [0.0]}, index=pd.to_datetime(["2026-01-31"]))
    supplementary_tables = {
        "Assumptions - Pricing": pd.DataFrame({"Product": ["Milk"], "Revenue": [10.0]}),
        "Assumptions - Production Drivers": pd.DataFrame({"Product": ["Milk"]}),
        "Assumptions - Operating Costs": pd.DataFrame({"Category": ["Feed"]}),
        "Assumptions - Biological Cost Drivers": pd.DataFrame({"Applies To": ["breeding_doe"]}),
    }

    hooks = ScenarioBuildHooks(
        input_schedule_cls=FakeInputSchedule,
        biological_driver_defaults=("Conception rate change (%)",),
        apply_biological_shocks=lambda assumptions, adjustments: {
            **assumptions,
            "Operating Costs": assumptions["Operating Costs"].assign(Scenario="Bio"),
        },
        apply_biological_assumptions_to_schedule=lambda schedule, assumptions: schedule.assign(BioSeed=True),
        apply_operating_cost_assumptions_to_schedule=lambda schedule, operating, bio: schedule.assign(CostApplied=True),
        apply_commercial_shocks_to_pricing=lambda pricing, schedule, drivers, adjustments: pricing.assign(
            **{"Base Price": [4.0], "Revenue": [25.0]}
        ),
        apply_pricing_assumptions_to_schedule=lambda schedule, pricing, drivers: schedule.assign(Priced=True),
        derive_biological_schedules=lambda schedule, assumptions: {
            "Biological Herd Summary": pd.DataFrame({"Period": ["2026"], "Heads": [10.0]})
        },
        scenario_output_specs=lambda: [
            ScenarioOutputSpec("Outputs", lambda ctx: pd.DataFrame({"Value": [1.0]}))
        ],
        pricing_family_summary=lambda pricing: pd.DataFrame({"Product": ["Milk"], "Revenue": [25.0]}),
        pricing_quantity_by_period=lambda pricing: pd.DataFrame({"Period": ["2026-01-31"], "Quantity": [5.0]}),
    )

    model, base, results = run_scenario_suite(
        schedule_df=schedule_df,
        valuation_inputs={"WACC": 0.1},
        supplementary_tables=supplementary_tables,
        scenario_suite={
            "Scenario A": {
                "description": "Biological shock",
                "adjustments": {
                    "Feed cost change (%)": 5.0,
                    "Conception rate change (%)": -10.0,
                },
            }
        },
        author_name="Tester",
        hooks=hooks,
    )

    scenario_payload = results["Scenario A"]

    assert isinstance(model, FakeModel)
    assert "BaseBuilt" in base.columns
    assert FakeInputSchedule.built_frames[-1]["BioSeed"].all()
    assert FakeInputSchedule.built_frames[-1]["CostApplied"].all()
    assert FakeInputSchedule.built_frames[-1]["Priced"].all()
    assert (
        FakeInputSchedule.built_supplementaries[-1]["Assumptions - Pricing"]["Base Price"].iloc[0]
        == pytest.approx(4.0)
    )
    assert "Outputs" in scenario_payload["supplementary"]
    assert "Biological Herd Summary" in scenario_payload["supplementary"]
    assert "Commercial Revenue by Product" in scenario_payload["supplementary"]
    assert scenario_payload["supplementary"]["Assumptions - Pricing"]["Base Price"].iloc[0] == pytest.approx(4.0)
    assert scenario_payload["valuation"]["npv"] == pytest.approx(123.0)
