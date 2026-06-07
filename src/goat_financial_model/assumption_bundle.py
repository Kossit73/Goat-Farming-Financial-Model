from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import pandas as pd


EnsureFn = Callable[[Optional[pd.DataFrame]], pd.DataFrame]


@dataclass(frozen=True)
class BiologicalConfig:
    system_settings: pd.DataFrame = field(default_factory=pd.DataFrame)
    breeding_reproduction: pd.DataFrame = field(default_factory=pd.DataFrame)
    lactation: pd.DataFrame = field(default_factory=pd.DataFrame)
    finishing_slaughter: pd.DataFrame = field(default_factory=pd.DataFrame)
    opening_herd_cohorts: pd.DataFrame = field(default_factory=pd.DataFrame)
    cohort_allocation_rules: pd.DataFrame = field(default_factory=pd.DataFrame)
    biological_cost_drivers: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class CommercialConfig:
    pricing: pd.DataFrame = field(default_factory=pd.DataFrame)
    production_drivers: pd.DataFrame = field(default_factory=pd.DataFrame)
    scenario_controls: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class FinancingConfig:
    capital_financing: pd.DataFrame = field(default_factory=pd.DataFrame)
    loan_facilities: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_facilities: pd.DataFrame = field(default_factory=pd.DataFrame)
    valuation_inputs: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass(frozen=True)
class AssumptionBundle:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    biological: BiologicalConfig = field(default_factory=BiologicalConfig)
    commercial: CommercialConfig = field(default_factory=CommercialConfig)
    financing: FinancingConfig = field(default_factory=FinancingConfig)

    def get(self, name: str, default: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        if name in self.tables:
            return self.tables[name]
        return pd.DataFrame() if default is None else default


def build_assumption_bundle(
    assumptions: dict[str, pd.DataFrame],
    ensure_map: dict[str, EnsureFn],
) -> AssumptionBundle:
    tables: dict[str, pd.DataFrame] = {}
    for name, ensure_fn in ensure_map.items():
        tables[name] = ensure_fn(assumptions.get(name))

    return AssumptionBundle(
        tables=tables,
        biological=BiologicalConfig(
            system_settings=tables.get("Biological System Settings", pd.DataFrame()),
            breeding_reproduction=tables.get("Breeding & Reproduction Biology", pd.DataFrame()),
            lactation=tables.get("Lactation Biology", pd.DataFrame()),
            finishing_slaughter=tables.get("Finishing & Slaughter Biology", pd.DataFrame()),
            opening_herd_cohorts=tables.get("Opening Herd Cohorts", pd.DataFrame()),
            cohort_allocation_rules=tables.get("Cohort Allocation Rules", pd.DataFrame()),
            biological_cost_drivers=tables.get("Biological Cost Drivers", pd.DataFrame()),
        ),
        commercial=CommercialConfig(
            pricing=tables.get("Pricing", pd.DataFrame()),
            production_drivers=tables.get("Production Drivers", pd.DataFrame()),
            scenario_controls=tables.get("Scenario Controls", pd.DataFrame()),
        ),
        financing=FinancingConfig(
            capital_financing=tables.get("Capital & Financing", pd.DataFrame()),
            loan_facilities=tables.get("Loan Facilities", pd.DataFrame()),
            equity_facilities=tables.get("Equity Facilities", pd.DataFrame()),
            valuation_inputs=tables.get("Valuation Inputs", pd.DataFrame()),
        ),
    )
