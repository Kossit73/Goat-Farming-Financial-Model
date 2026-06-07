from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import pandas as pd


OutputBuilder = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class ScenarioOutputSpec:
    name: str
    builder: OutputBuilder


@dataclass
class ScenarioRunResult:
    scenario_name: str
    scenario_df: pd.DataFrame
    supplementary: dict[str, pd.DataFrame] = field(default_factory=dict)
    valuation: dict[str, Any] = field(default_factory=dict)
    kpis: pd.DataFrame = field(default_factory=pd.DataFrame)
    audit: dict[str, Any] = field(default_factory=dict)


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
