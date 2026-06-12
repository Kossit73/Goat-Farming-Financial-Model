from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

import pandas as pd


OutputBuilder = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ScenarioOutputSpec:
    name: str
    builder: OutputBuilder


@dataclass(frozen=True)
class ScenarioBuildHooks:
    input_schedule_cls: Callable[..., Any]
    biological_driver_defaults: Iterable[str]
    apply_biological_shocks: Callable[[dict[str, pd.DataFrame], dict[str, Any]], dict[str, pd.DataFrame]]
    apply_biological_assumptions_to_schedule: Callable[[pd.DataFrame, dict[str, pd.DataFrame]], pd.DataFrame]
    apply_operating_cost_assumptions_to_schedule: Callable[
        [pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame
    ]
    apply_commercial_shocks_to_pricing: Callable[
        [pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], dict[str, Any]], pd.DataFrame
    ]
    apply_pricing_assumptions_to_schedule: Callable[
        [pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame
    ]
    derive_biological_schedules: Callable[[pd.DataFrame, dict[str, pd.DataFrame]], dict[str, pd.DataFrame]]
    scenario_output_specs: Callable[[], list[ScenarioOutputSpec]]
    pricing_family_summary: Callable[[pd.DataFrame], pd.DataFrame]
    pricing_quantity_by_period: Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class ScenarioRunResult:
    scenario_name: str
    model: Any
    base: pd.DataFrame
    scenario_df: pd.DataFrame
    supplementary: dict[str, pd.DataFrame] = field(default_factory=dict)
    valuation: dict[str, Any] = field(default_factory=dict)
    kpis: pd.DataFrame = field(default_factory=pd.DataFrame)
    audit: dict[str, Any] = field(default_factory=dict)
    break_even: pd.DataFrame = field(default_factory=pd.DataFrame)
    debt_schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    debt_schedule_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_schedule_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    working_capital: pd.DataFrame = field(default_factory=pd.DataFrame)
    working_capital_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    debt_capacity: pd.DataFrame = field(default_factory=pd.DataFrame)
    debt_capacity_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    ufcf_schedule: pd.DataFrame = field(default_factory=pd.DataFrame)
    ufcf_schedule_annual: pd.DataFrame = field(default_factory=pd.DataFrame)
    pricing_assumptions: pd.DataFrame = field(default_factory=pd.DataFrame)
    selected_scenario: str = ""
    scenario_inputs: dict[str, Any] = field(default_factory=dict)
    model_author: str = ""
    preset_description: str = ""


def collect_supplementary_outputs(
    specs: list[ScenarioOutputSpec],
    context: dict[str, Any],
) -> dict[str, pd.DataFrame]:
    outputs: dict[str, pd.DataFrame] = {}
    for spec in specs:
        value = spec.builder(context)
        if isinstance(value, pd.DataFrame) and not value.empty:
            outputs[spec.name] = value
    return outputs


def _copy_supplementary_tables(
    supplementary_tables: dict[str, pd.DataFrame] | None,
) -> dict[str, pd.DataFrame]:
    return {
        name: table.copy()
        for name, table in (supplementary_tables or {}).items()
        if isinstance(table, pd.DataFrame)
    }


def _extract_biological_assumptions(
    supplementary_tables: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        name.replace("Assumptions - ", "", 1): table.copy()
        for name, table in supplementary_tables.items()
        if name.startswith("Assumptions - ") and isinstance(table, pd.DataFrame)
    }


def _prefixed_assumption_tables(
    assumptions: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    return {
        f"Assumptions - {name}": table.copy()
        for name, table in (assumptions or {}).items()
        if isinstance(table, pd.DataFrame) and not table.empty
    }


def build_scenario_seed(
    *,
    schedule_df: pd.DataFrame,
    adjustments: dict[str, Any],
    biological_assumptions: dict[str, pd.DataFrame],
    assumption_pricing: pd.DataFrame,
    production_drivers: pd.DataFrame,
    hooks: ScenarioBuildHooks,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    has_biological_adjustment = any(
        abs(float(adjustments.get(driver, 0.0) or 0.0)) > 1e-9
        for driver in hooks.biological_driver_defaults
    )
    if has_biological_adjustment:
        scenario_assumptions = hooks.apply_biological_shocks(
            biological_assumptions,
            adjustments,
        )
        scenario_seed = hooks.apply_biological_assumptions_to_schedule(
            schedule_df.copy(),
            scenario_assumptions,
        )
        operating_costs = scenario_assumptions.get("Operating Costs")
        if isinstance(operating_costs, pd.DataFrame) and not operating_costs.empty:
            scenario_seed = hooks.apply_operating_cost_assumptions_to_schedule(
                scenario_seed,
                operating_costs,
                scenario_assumptions.get("Biological Cost Drivers"),
            )
    else:
        scenario_assumptions = biological_assumptions
        scenario_seed = schedule_df.copy()

    scenario_pricing = pd.DataFrame()
    scenario_pricing_source = scenario_assumptions.get("Pricing", assumption_pricing)
    scenario_production_drivers = scenario_assumptions.get(
        "Production Drivers",
        production_drivers,
    )
    production_driver_table = (
        scenario_production_drivers
        if isinstance(scenario_production_drivers, pd.DataFrame)
        else None
    )
    if isinstance(scenario_pricing_source, pd.DataFrame) and not scenario_pricing_source.empty:
        scenario_pricing = hooks.apply_commercial_shocks_to_pricing(
            scenario_pricing_source,
            scenario_seed,
            production_driver_table,
            adjustments,
        )
        scenario_assumptions = dict(scenario_assumptions)
        scenario_assumptions["Pricing"] = scenario_pricing
        if production_driver_table is not None:
            scenario_assumptions["Production Drivers"] = production_driver_table
        scenario_seed = hooks.apply_pricing_assumptions_to_schedule(
            scenario_seed,
            scenario_pricing,
            production_driver_table,
        )

    return scenario_seed, scenario_assumptions, scenario_pricing


def run_single_scenario(
    *,
    scenario_name: str,
    config: dict[str, Any],
    schedule_df: pd.DataFrame,
    valuation_inputs: dict[str, float],
    base_supplementary: dict[str, pd.DataFrame],
    biological_assumptions: dict[str, pd.DataFrame],
    assumption_pricing: pd.DataFrame,
    production_drivers: pd.DataFrame,
    base: pd.DataFrame,
    author_name: str,
    hooks: ScenarioBuildHooks,
) -> ScenarioRunResult:
    adjustments = config.get("adjustments", {})
    feed_pct = float(adjustments.get("Feed cost change (%)", 0.0))
    scenario_seed, scenario_assumptions, scenario_pricing = build_scenario_seed(
        schedule_df=schedule_df,
        adjustments=adjustments,
        biological_assumptions=biological_assumptions,
        assumption_pricing=assumption_pricing,
        production_drivers=production_drivers,
        hooks=hooks,
    )

    scenario_model_supplementary = _copy_supplementary_tables(base_supplementary)
    scenario_model_supplementary.update(
        _prefixed_assumption_tables(scenario_assumptions)
    )

    scenario_schedule = hooks.input_schedule_cls(
        data=scenario_seed,
        valuation_inputs=valuation_inputs,
        supplementary_tables=scenario_model_supplementary,
    )
    scenario_model = scenario_schedule.to_model()
    scenario_df = scenario_model.scenario(
        milk_price_pct=0.0,
        feed_cost_pct=feed_pct / 100.0,
    )

    scenario_supplementary = _copy_supplementary_tables(
        scenario_model_supplementary
    )
    biological_bundle = hooks.derive_biological_schedules(
        scenario_seed,
        scenario_assumptions,
    )
    valuation_summary = scenario_model.valuation_summary(scenario_df)
    model_audit = scenario_model.model_audit(scenario_df, annual=True)
    scenario_kpis = scenario_model.kpis(scenario_df, annual=True)
    debt_schedule_detail = scenario_model.debt_schedule(annual=False)
    debt_schedule_annual = scenario_model.debt_schedule(annual=True)
    equity_schedule_detail = scenario_model.equity_schedule(annual=False)
    equity_schedule_annual = scenario_model.equity_schedule(annual=True)
    working_capital_detail = scenario_model.working_capital_schedule(scenario_df, annual=False)
    working_capital_annual = scenario_model.working_capital_schedule(scenario_df, annual=True)
    debt_capacity_detail = scenario_model.debt_capacity_schedule(scenario_df, annual=False)
    debt_capacity_annual = scenario_model.debt_capacity_schedule(scenario_df, annual=True)
    ufcf_detail = scenario_model.ufcf_schedule(scenario_df, annual=False)
    ufcf_annual = scenario_model.ufcf_schedule(scenario_df, annual=True)

    scenario_output_context = {
        "model": scenario_model,
        "scenario_df": scenario_df,
        "scenario_kpis": scenario_kpis,
        "debt_schedule_annual": debt_schedule_annual,
        "equity_schedule_annual": equity_schedule_annual,
        "working_capital_annual": working_capital_annual,
        "debt_capacity_annual": debt_capacity_annual,
        "ufcf_annual": ufcf_annual,
        "model_audit": model_audit,
    }
    scenario_supplementary.update(
        collect_supplementary_outputs(
            hooks.scenario_output_specs(),
            scenario_output_context,
        )
    )
    for schedule_name, biological_table in biological_bundle.items():
        if isinstance(biological_table, pd.DataFrame) and not biological_table.empty:
            scenario_supplementary[schedule_name] = biological_table
    if not scenario_pricing.empty:
        scenario_supplementary["Commercial Revenue by Product"] = hooks.pricing_family_summary(
            scenario_pricing
        )
        scenario_supplementary["Commercial Quantity by Period"] = hooks.pricing_quantity_by_period(
            scenario_pricing
        )
        scenario_supplementary["Assumptions - Pricing"] = scenario_pricing

    scenario_inputs: dict[str, Any] = {
        key: float(value) for key, value in adjustments.items()
    }
    if author_name:
        scenario_inputs["Model author"] = author_name

    return ScenarioRunResult(
        scenario_name=scenario_name,
        model=scenario_model,
        base=base,
        scenario_df=scenario_df,
        supplementary=scenario_supplementary,
        valuation=valuation_summary,
        kpis=scenario_kpis,
        audit=model_audit,
        break_even=scenario_model.break_even(scenario_df, annual=True),
        debt_schedule=debt_schedule_detail,
        debt_schedule_annual=debt_schedule_annual,
        equity_schedule=equity_schedule_detail,
        equity_schedule_annual=equity_schedule_annual,
        working_capital=working_capital_detail,
        working_capital_annual=working_capital_annual,
        debt_capacity=debt_capacity_detail,
        debt_capacity_annual=debt_capacity_annual,
        ufcf_schedule=ufcf_detail,
        ufcf_schedule_annual=ufcf_annual,
        pricing_assumptions=scenario_pricing,
        selected_scenario=scenario_name,
        scenario_inputs=scenario_inputs,
        model_author=author_name,
        preset_description=config.get("description", ""),
    )


def scenario_result_to_payload(result: ScenarioRunResult) -> dict[str, Any]:
    return {
        "model": result.model,
        "base": result.base,
        "scenario": result.scenario_df,
        "kpis": result.kpis,
        "break_even": result.break_even,
        "valuation": result.valuation,
        "model_audit": result.audit,
        "debt_schedule": result.debt_schedule,
        "debt_schedule_annual": result.debt_schedule_annual,
        "equity_schedule": result.equity_schedule,
        "equity_schedule_annual": result.equity_schedule_annual,
        "working_capital": result.working_capital,
        "working_capital_annual": result.working_capital_annual,
        "debt_capacity": result.debt_capacity,
        "debt_capacity_annual": result.debt_capacity_annual,
        "ufcf_schedule": result.ufcf_schedule,
        "ufcf_schedule_annual": result.ufcf_schedule_annual,
        "supplementary": result.supplementary,
        "pricing_assumptions": result.pricing_assumptions,
        "selected_scenario": result.selected_scenario,
        "scenario_inputs": result.scenario_inputs,
        "model_author": result.model_author,
        "preset_description": result.preset_description,
    }


def run_scenario_suite(
    *,
    schedule_df: pd.DataFrame,
    valuation_inputs: dict[str, float],
    supplementary_tables: dict[str, pd.DataFrame],
    scenario_suite: dict[str, dict[str, Any]],
    author_name: str,
    hooks: ScenarioBuildHooks,
) -> tuple[Any, pd.DataFrame, dict[str, dict[str, Any]]]:
    schedule = hooks.input_schedule_cls(
        data=schedule_df,
        valuation_inputs=valuation_inputs,
        supplementary_tables=supplementary_tables,
    )
    model = schedule.to_model()
    base = model.to_tidy()

    base_supplementary = _copy_supplementary_tables(supplementary_tables)
    biological_assumptions = _extract_biological_assumptions(base_supplementary)
    assumption_pricing = base_supplementary.get("Assumptions - Pricing", pd.DataFrame())
    production_drivers = base_supplementary.get(
        "Assumptions - Production Drivers",
        pd.DataFrame(),
    )

    results: dict[str, dict[str, Any]] = {}
    for scenario_name, config in scenario_suite.items():
        result = run_single_scenario(
            scenario_name=scenario_name,
            config=config,
            schedule_df=schedule_df,
            valuation_inputs=valuation_inputs,
            base_supplementary=base_supplementary,
            biological_assumptions=biological_assumptions,
            assumption_pricing=assumption_pricing,
            production_drivers=production_drivers,
            base=base,
            author_name=author_name,
            hooks=hooks,
        )
        results[scenario_name] = scenario_result_to_payload(result)

    return model, base, results
