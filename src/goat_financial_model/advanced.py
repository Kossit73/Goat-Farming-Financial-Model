"""Advanced analytics helpers for goat farming financial scenarios."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import numpy as np
import pandas as pd

from .goat_model import DataQualityWarning, _coerce_numeric_frame


def _safe_divide(numerator: np.ndarray | pd.Series, denominator: np.ndarray | pd.Series) -> np.ndarray:
    """Vectorised safe division that returns ``NaN`` for near-zero denominators."""

    numerator_arr = np.asarray(numerator, dtype="float64")
    denominator_arr = np.asarray(denominator, dtype="float64")
    mask = np.abs(denominator_arr) > 1e-9
    out = np.full_like(numerator_arr, np.nan, dtype="float64")
    out[mask] = numerator_arr[mask] / denominator_arr[mask]
    return out


def _normalise_name(raw: str) -> str:
    cleaned = raw.replace("_", " ").replace("-", " ")
    cleaned = " ".join(part for part in cleaned.split() if part)
    return cleaned.title()


@dataclass
class AnalysisResult:
    """Container describing a single advanced analytics output."""

    title: str
    description: str
    tables: Mapping[str, pd.DataFrame]


class AdvancedAnalyticsSuite:
    """Compute an extended catalogue of advanced analytics for a scenario."""

    def __init__(
        self,
        data: pd.DataFrame,
        window: int,
        annual: bool,
        assumptions: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.original = data.copy()
        self.window = max(int(window), 1)
        self.annual = annual
        self.random = np.random.default_rng(42)

        self.assumptions = self._normalise_assumptions(assumptions)
        self.discount_rate = float(self.assumptions.get("wacc", self.assumptions.get("discount rate", 0.1)))
        self.discount_rate = float(np.clip(self.discount_rate, 1e-6, 1.0))
        self.terminal_value = self.assumptions.get("terminal value")
        self.terminal_growth = self.assumptions.get("terminal growth rate")

        self.tax_rate_input = self._resolve_tax_rate_override()
        self.data = self._prepare_data()
        self.index = self.data.index
        self.periods_per_year = self._periods_per_year()
        self.period_rate = self._period_discount_rate()

        self.revenue = self._series("Revenue")
        self.cogs = self._series("COGS")
        self.variable_expenses = self._series("Variable Expenses")
        self.direct_wages = self._series("Direct Wages")
        self.fixed_expenses = self._series("Fixed Expenses")
        self.admin_wages = self._series("Admin Wages")
        self.depreciation = self._series("Depreciation & Amortization")
        self.interest = self._series("Interest Expense")
        self.tax = self._series("Tax Expense")
        self.capex = self._series("Capex")
        self.ebitda_series = self._series("EBITDA")
        self.npat_series = self._series("NPAT")
        self.ufcf = self._series("Unlevered Free Cash Flow")

        self.revenue_total = float(np.nansum(self.revenue))
        self.cogs_total = float(np.nansum(self.cogs))
        self.variable_total = float(np.nansum(self.variable_expenses))
        self.direct_total = float(np.nansum(self.direct_wages))
        self.fixed_total = float(np.nansum(self.fixed_expenses))
        self.admin_total = float(np.nansum(self.admin_wages))
        self.depreciation_total = float(np.nansum(self.depreciation))
        self.interest_total = float(np.nansum(self.interest))
        self.tax_total = float(np.nansum(self.tax))
        self.capex_total = float(np.nansum(self.capex))
        self.ebitda_total = float(np.nansum(self.ebitda_series))
        self.npat_total = float(np.nansum(self.npat_series))

        self.variable_ratio = self._ratio(self.variable_total + self.direct_total, self.revenue_total)
        self.cogs_ratio = self._ratio(self.cogs_total, self.revenue_total)
        self.fixed_admin_total = self.fixed_total + self.admin_total
        self.tax_rate, self.tax_rate_source = self._derive_tax_rate()
        self.tax_rate_note = self._format_tax_rate_note()

        self._cached_monte_carlo: Optional[pd.DataFrame] = None

    @staticmethod
    def _normalise_assumptions(
        assumptions: Optional[Mapping[str, float]]
    ) -> Dict[str, float]:
        if not assumptions:
            return {}
        normalised: Dict[str, float] = {}
        for key, value in assumptions.items():
            if value is None:
                continue
            try:
                normalised[str(key).lower()] = float(value)
            except (TypeError, ValueError):
                continue
        return normalised

    def _resolve_tax_rate_override(self) -> Optional[float]:
        for key in (
            "tax rate",
            "effective tax rate",
            "tax_rate",
            "taxrate",
        ):
            value = self.assumptions.get(key)
            if value is None:
                continue
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if rate > 1:
                rate /= 100.0
            return float(np.clip(rate, 0.0, 0.6))
        return None

    def _prepare_data(self) -> pd.DataFrame:
        work = self.original.copy()
        numeric = _coerce_numeric_frame(work, context="Advanced analytics input")
        if self.annual:
            index = numeric.index
            if not isinstance(index, pd.DatetimeIndex):
                index = pd.to_datetime(index, errors="coerce")
            period_index = index.to_period("Y")
            grouped = numeric.groupby(period_index).sum(min_count=1)
            numeric = grouped
            try:
                numeric.index = numeric.index.to_timestamp("Y")
            except Exception:  # pragma: no cover - defensive fallback
                numeric.index = pd.to_datetime([f"{val}-12-31" for val in numeric.index])
        return numeric

    @staticmethod
    def _normalise_frequency_code(freq: Optional[str]) -> Optional[str]:
        if not freq:
            return None
        freq_str = str(freq).upper()
        replacements = [
            ("BQE", "BQ"),
            ("QE", "Q"),
            ("SME", "SM"),
            ("BME", "BM"),
            ("ME", "M"),
        ]
        for alias, replacement in replacements:
            if alias in freq_str:
                freq_str = freq_str.replace(alias, replacement)
        return freq_str

    def _periods_per_year(self) -> int:
        if len(self.index) <= 1:
            return 1
        freq = None
        if isinstance(self.index, pd.DatetimeIndex):
            freq = pd.infer_freq(self.index)
        elif isinstance(self.index, pd.PeriodIndex) and self.index.freqstr:
            freq = self.index.freqstr
        if freq:
            freq = self._normalise_frequency_code(freq)
            base_freq = freq.split("-")[0] if freq else None
            mapping = {
                "A": 1,
                "Y": 1,
                "M": 12,
                "MS": 12,
                "BM": 12,
                "Q": 4,
                "QS": 4,
                "W": 52,
                "D": 365,
            }
            for key, value in mapping.items():
                if base_freq and base_freq.startswith(key):
                    return value
        if isinstance(self.index, pd.DatetimeIndex) and len(self.index) > 1:
            deltas = np.diff(self.index.asi8)
            median_delta = np.median(deltas)
            if median_delta <= 0:
                return 1
            year_delta = pd.Timedelta(days=365).value
            periods = int(round(year_delta / median_delta))
            return max(periods, 1)
        return 1

    def _period_discount_rate(self) -> float:
        periods = max(self.periods_per_year, 1)
        return float((1 + self.discount_rate) ** (1 / periods) - 1)

    def _series(self, name: str) -> pd.Series:
        if name in self.data.columns:
            return pd.to_numeric(self.data[name], errors="coerce").fillna(0.0)
        if name in self.original.columns:
            series = pd.to_numeric(self.original[name], errors="coerce").fillna(0.0)
            if self.annual:
                return series.groupby(series.index.year).sum(min_count=1)
            return series
        return pd.Series(0.0, index=self.data.index)

    @staticmethod
    def _ratio(numerator: float, denominator: float) -> float:
        if abs(denominator) < 1e-9:
            return 0.0
        return float(numerator) / float(denominator)

    def _derive_tax_rate(self) -> Tuple[float, str]:
        if self.tax_rate_input is not None:
            return float(self.tax_rate_input), "override"

        npbt = self._series("NPBT")
        if npbt.empty:
            npbt_total = self.ebitda_total - self.depreciation_total - self.interest_total
        else:
            npbt_total = float(np.nansum(npbt))
        if npbt_total <= 0:
            return 0.25, "fallback"
        candidate = self.tax_total / npbt_total if npbt_total else np.nan
        if not np.isfinite(candidate) or candidate < 0:
            return 0.25, "fallback"
        return float(np.clip(candidate, 0.0, 0.5)), "historical"

    def _format_tax_rate_note(self) -> str:
        source_map = {
            "override": "user override",
            "historical": "derived from historical data",
            "fallback": "default fallback",
        }
        label = source_map.get(self.tax_rate_source, self.tax_rate_source)
        return f"Effective tax rate assumed: {self.tax_rate:.1%} ({label})."

    # ------------------------------------------------------------------
    # Core simulation helpers
    # ------------------------------------------------------------------
    def _simulate_financials(
        self,
        revenue_factor: float = 1.0,
        cogs_factor: float = 1.0,
        variable_factor: float = 1.0,
        productivity_factor: float = 1.0,
        fixed_factor: float = 1.0,
        interest_factor: float = 1.0,
        capex_factor: float = 1.0,
    ) -> MutableMapping[str, float]:
        revenue = self.revenue_total * revenue_factor * productivity_factor
        cogs = self.cogs_total * cogs_factor * productivity_factor
        variable = (self.variable_total + self.direct_total) * variable_factor * productivity_factor
        fixed = self.fixed_admin_total * fixed_factor
        gross_margin = revenue - cogs
        ebitda = gross_margin - variable - fixed
        depreciation = self.depreciation_total
        ebit = ebitda - depreciation
        interest = self.interest_total * interest_factor
        npbt = ebit - interest
        tax = max(npbt, 0.0) * self.tax_rate
        npat = npbt - tax
        capex = self.capex_total * capex_factor
        cash_flow = ebitda - capex
        return {
            "Revenue": revenue,
            "COGS": cogs,
            "Gross Margin": gross_margin,
            "EBITDA": ebitda,
            "EBIT": ebit,
            "NPBT": npbt,
            "Tax": tax,
            "NPAT": npat,
            "Capex": capex,
            "Operating Cash Flow": cash_flow,
        }

    def _base_financials(self) -> MutableMapping[str, float]:
        return self._simulate_financials()

    def _monte_carlo_distribution(self) -> pd.DataFrame:
        if self._cached_monte_carlo is not None:
            return self._cached_monte_carlo

        periods = len(self.revenue)
        if periods == 0:
            self._cached_monte_carlo = pd.DataFrame()
            return self._cached_monte_carlo

        revenue_base = self.revenue.to_numpy()
        if periods >= 2:
            revenue_std = np.nanstd(revenue_base, ddof=1)
        else:
            revenue_std = 0.0
        if not np.isfinite(revenue_std) or revenue_std < 1e-6:
            revenue_std = max(abs(np.nanmean(revenue_base)) * 0.05, 1.0)

        draws = 2000
        revenue_shocks = self.random.normal(0.0, revenue_std, size=(draws, periods))
        revenue_sim = np.clip(revenue_base + revenue_shocks, 0.0, None)

        cogs_ratio = np.clip(self.cogs_ratio, 0.0, 0.95)
        cogs_sigma = max(cogs_ratio * 0.05, 0.01)
        cogs_draws = self.random.normal(cogs_ratio, cogs_sigma, size=(draws, periods))
        cogs_draws = np.clip(cogs_draws, 0.0, 0.98)
        cogs_sim = revenue_sim * cogs_draws

        variable_ratio = np.clip(self.variable_ratio, 0.0, 0.9)
        variable_sigma = max(variable_ratio * 0.05, 0.01)
        variable_draws = self.random.normal(variable_ratio, variable_sigma, size=(draws, periods))
        variable_draws = np.clip(variable_draws, 0.0, 0.95)
        variable_sim = revenue_sim * variable_draws
        variable_cap = np.clip(revenue_sim - cogs_sim, 0.0, None)
        variable_sim = np.minimum(variable_sim, variable_cap)

        fixed_base = (
            (self.fixed_expenses + self.admin_wages)
            .reindex(self.index, fill_value=0.0)
            .to_numpy(dtype=float)
        )
        fixed_sim = np.tile(fixed_base, (draws, 1))
        fixed_noise = np.clip(self.random.normal(1.0, 0.03, size=(draws, periods)), 0.85, 1.15)
        fixed_sim = np.clip(fixed_sim * fixed_noise, 0.0, None)

        gross_margin_sim = revenue_sim - cogs_sim
        ebitda_sim = gross_margin_sim - variable_sim - fixed_sim

        depreciation_base = self.depreciation.reindex(self.index, fill_value=0.0).to_numpy(dtype=float)
        depreciation_sim = np.tile(depreciation_base, (draws, 1))
        dep_noise = np.clip(self.random.normal(1.0, 0.05, size=(draws, periods)), 0.7, 1.3)
        depreciation_sim = np.clip(depreciation_sim * dep_noise, 0.0, None)

        ebit_sim = ebitda_sim - depreciation_sim

        interest_base = self.interest.reindex(self.index, fill_value=0.0).to_numpy(dtype=float)
        interest_sim = np.tile(interest_base, (draws, 1))
        interest_noise = np.clip(self.random.normal(1.0, 0.08, size=(draws, periods)), 0.5, 1.5)
        interest_sim = np.clip(interest_sim * interest_noise, 0.0, None)

        npbt_sim = ebit_sim - interest_sim
        tax_sim = np.maximum(npbt_sim, 0.0) * self.tax_rate
        npat_sim = npbt_sim - tax_sim
        capex_base = self.capex.reindex(self.index, fill_value=0.0).to_numpy(dtype=float)
        capex_sim = np.tile(capex_base, (draws, 1))
        capex_noise = np.clip(self.random.normal(1.0, 0.1, size=(draws, periods)), 0.5, 1.5)
        capex_sim = capex_sim * capex_noise
        cash_flow_sim = ebitda_sim - capex_sim

        discount_factors = 1 / (1 + self.period_rate) ** np.arange(1, periods + 1)
        npv = (cash_flow_sim * discount_factors).sum(axis=1)
        if self.terminal_value is not None:
            npv += float(self.terminal_value) / ((1 + self.period_rate) ** periods)
        elif self.terminal_growth is not None and self.period_rate > self.terminal_growth:
            terminal_cash = cash_flow_sim[:, -1]
            terminal_val = terminal_cash * (1 + self.terminal_growth) / (
                self.period_rate - self.terminal_growth
            )
            npv += terminal_val / ((1 + self.period_rate) ** periods)

        summary = pd.DataFrame(
            {
                "Total Revenue": revenue_sim.sum(axis=1),
                "Total EBITDA": ebitda_sim.sum(axis=1),
                "Total NPAT": npat_sim.sum(axis=1),
                "Operating Cash Flow": cash_flow_sim.sum(axis=1),
                "NPV": npv,
            }
        )
        self._cached_monte_carlo = summary
        return summary

    # ------------------------------------------------------------------
    # Individual analyses
    # ------------------------------------------------------------------
    def sensitivity_analysis(self) -> AnalysisResult:
        base = self._base_financials()
        scenarios = []
        for driver, adjustments in (
            ("Milk Price", {"revenue_factor": 1.05}),
            ("Milk Price", {"revenue_factor": 0.95}),
            ("Feed Costs", {"cogs_factor": 1.07, "variable_factor": 1.05}),
            ("Feed Costs", {"cogs_factor": 0.93, "variable_factor": 0.95}),
            ("Herd Productivity", {"productivity_factor": 1.08}),
            ("Herd Productivity", {"productivity_factor": 0.92}),
        ):
            result = self._simulate_financials(**adjustments)
            scenarios.append(
                {
                    "Driver": driver,
                    "Scenario": "Increase" if list(adjustments.values())[0] > 1 else "Decrease",
                    "Revenue": result["Revenue"],
                    "EBITDA": result["EBITDA"],
                    "NPAT": result["NPAT"],
                    "NPAT Change": result["NPAT"] - base["NPAT"],
                }
            )

        table = pd.DataFrame(scenarios)
        table.set_index(["Driver", "Scenario"], inplace=True)
        return AnalysisResult(
            title="Sensitivity Analysis",
            description="Quantifies profitability shifts from movements in milk prices, feed costs, and herd productivity.",
            tables={"Impact Summary": table},
        )

    def stress_testing(self) -> AnalysisResult:
        base = self._base_financials()
        shocks = {
            "Drought": {"revenue_factor": 0.8, "variable_factor": 1.1, "cogs_factor": 1.05},
            "Disease Outbreak": {"productivity_factor": 0.7, "fixed_factor": 1.1, "variable_factor": 1.05},
            "Supply Shock": {"cogs_factor": 1.2, "variable_factor": 1.1},
            "Market Slump": {"revenue_factor": 0.75, "interest_factor": 1.15},
        }
        rows: List[MutableMapping[str, float]] = []
        for name, adjustments in shocks.items():
            result = self._simulate_financials(**adjustments)
            rows.append(
                {
                    "Scenario": name,
                    "Revenue": result["Revenue"],
                    "EBITDA": result["EBITDA"],
                    "NPAT": result["NPAT"],
                    "Cash Flow": result["Operating Cash Flow"],
                    "NPAT Delta": result["NPAT"] - base["NPAT"],
                }
            )

        table = pd.DataFrame(rows).set_index("Scenario")
        return AnalysisResult(
            title="Scenario Stress Testing",
            description="Applies severe but plausible shocks to evaluate financial resilience across drought, disease, and market disruptions.",
            tables={"Stress Test Outcomes": table},
        )

    def trend_and_seasonality(self) -> AnalysisResult:
        revenue = self.revenue.copy()
        if revenue.empty:
            return AnalysisResult(
                title="Trend & Seasonality",
                description="Insufficient revenue history for decomposition.",
                tables={"Trend": pd.DataFrame()},
            )

        window = min(max(self.window, 2), max(len(revenue) // 2, 2))
        trend = revenue.rolling(window=window, min_periods=1, center=True).mean()
        seasonal_index = pd.Series(_safe_divide(revenue, trend), index=revenue.index)
        seasonal_index = seasonal_index.replace([np.inf, -np.inf], np.nan).fillna(1.0)
        seasonal_mean = seasonal_index.groupby(seasonal_index.index.month if hasattr(seasonal_index.index, "month") else seasonal_index.index).mean()
        seasonally_adjusted = revenue / seasonal_index

        table = pd.DataFrame(
            {
                "Revenue": revenue,
                "Trend": trend,
                "Seasonality Index": seasonal_index,
                "Seasonally Adjusted Revenue": seasonally_adjusted,
            }
        )
        if hasattr(seasonal_index.index, "month"):
            monthly = seasonal_mean.rename("Average Seasonal Index").to_frame()
        else:
            monthly = pd.DataFrame()

        return AnalysisResult(
            title="Trend & Seasonality Decomposition",
            description="Separates structural revenue trends from recurring seasonal effects.",
            tables={"Decomposition": table, "Average Seasonality": monthly},
        )

    def segmentation(self) -> AnalysisResult:
        segment_map: Dict[str, pd.Series] = {}
        for col in self.original.columns:
            if col in {"Revenue", "Revenue_adj"}:
                continue
            lowered = col.lower()
            if lowered.startswith("revenue ") or lowered.startswith("revenue-") or lowered.startswith("revenue:"):
                segment_map[_normalise_name(col.split(" ", 1)[-1])] = pd.to_numeric(self.original[col], errors="coerce")
            elif lowered.startswith("sales ") or lowered.startswith("sales-"):
                segment_map[_normalise_name(col.split(" ", 1)[-1])] = pd.to_numeric(self.original[col], errors="coerce")

        if not segment_map:
            revenue = self.revenue
            if revenue.empty:
                table = pd.DataFrame()
            else:
                quantiles = pd.qcut(revenue.rank(method="first"), q=min(3, len(revenue)), labels=False) if len(revenue) >= 3 else pd.Series(0, index=revenue.index)
                labels = {0: "Baseline", 1: "Growth", 2: "Peak"}
                segment = quantiles.map(labels).fillna("Baseline")
                grouped = revenue.groupby(segment).sum()
                gross_margin_ratio = self._ratio((self.revenue_total - self.cogs_total), self.revenue_total)
                rows = []
                for seg, value in grouped.items():
                    gm = value * gross_margin_ratio
                    rows.append(
                        {
                            "Segment": seg,
                            "Revenue": value,
                            "Estimated Gross Margin": gm,
                            "Margin %": self._ratio(gm, value),
                            "Revenue Share %": self._ratio(value, self.revenue_total),
                        }
                    )
                table = pd.DataFrame(rows).set_index("Segment")
        else:
            rows = []
            gross_margin_ratio = self._ratio((self.revenue_total - self.cogs_total), self.revenue_total)
            total = sum(series.sum() for series in segment_map.values())
            total = total if total else 1.0
            for name, series in segment_map.items():
                series = series.groupby(series.index.year).sum(min_count=1) if self.annual else series
                value = float(np.nansum(series))
                gm = value * gross_margin_ratio
                rows.append(
                    {
                        "Segment": name,
                        "Revenue": value,
                        "Estimated Gross Margin": gm,
                        "Margin %": self._ratio(gm, value),
                        "Revenue Share %": self._ratio(value, total),
                    }
                )
            table = pd.DataFrame(rows).set_index("Segment")

        return AnalysisResult(
            title="Customer & Product Segmentation",
            description="Highlights revenue and margin contribution by customer or product channel.",
            tables={"Segment Contribution": table},
        )

    def monte_carlo(self) -> AnalysisResult:
        distribution = self._monte_carlo_distribution()
        if distribution.empty:
            summary = pd.DataFrame()
        else:
            summary = pd.DataFrame(
                {
                    "Mean": distribution.mean(),
                    "Std Dev": distribution.std(ddof=1),
                    "P5": distribution.quantile(0.05),
                    "Median": distribution.quantile(0.5),
                    "P95": distribution.quantile(0.95),
                }
            ).T

        return AnalysisResult(
            title="Monte Carlo Simulation",
            description=(
                "Simulates profitability and valuation distributions by stochastically varying revenue and cost drivers. "
                + self.tax_rate_note
            ),
            tables={"Summary Statistics": summary},
        )

    def what_if(self) -> AnalysisResult:
        base = self._base_financials()
        cases = {
            "Revenue +5%": {"revenue_factor": 1.05},
            "Feed Cost -5%": {"cogs_factor": 0.95, "variable_factor": 0.97},
            "Productivity +3%": {"productivity_factor": 1.03},
            "Efficiency Drive": {"variable_factor": 0.92, "fixed_factor": 0.97},
            "Capex Freeze": {"capex_factor": 0.6},
        }
        rows = []
        for name, adjustments in cases.items():
            result = self._simulate_financials(**adjustments)
            rows.append(
                {
                    "Case": name,
                    "Revenue": result["Revenue"],
                    "EBITDA": result["EBITDA"],
                    "NPAT": result["NPAT"],
                    "Cash Flow": result["Operating Cash Flow"],
                    "NPAT Delta": result["NPAT"] - base["NPAT"],
                }
            )
        table = pd.DataFrame(rows).set_index("Case")
        return AnalysisResult(
            title="What-if Analysis",
            description="Interactively compares predefined assumption adjustments and their impact on profitability and cash flow.",
            tables={"Scenario Comparison": table},
        )

    def goal_seek(self) -> AnalysisResult:
        base = self._base_financials()
        contribution_margin = 1 - self.cogs_ratio - self.variable_ratio
        contribution_margin = max(contribution_margin, 1e-6)
        target_margin = 0.25
        revenue_for_margin = self.fixed_admin_total / (contribution_margin - target_margin) if contribution_margin > target_margin else np.nan
        revenue_for_break_even = self.fixed_admin_total / contribution_margin
        target_npat = base["NPAT"] * 1.2 if base["NPAT"] > 0 else 100000.0
        revenue_for_target_npat = (target_npat / (1 - self.tax_rate) + self.fixed_admin_total + self.depreciation_total + self.interest_total) / contribution_margin

        table = pd.DataFrame(
            {
                "Goal": ["Break-even NPAT", "EBITDA Margin 25%", "NPAT +20%"],
                "Required Revenue": [revenue_for_break_even, revenue_for_margin, revenue_for_target_npat],
                "Revenue Lift vs Base": [
                    revenue_for_break_even - base["Revenue"],
                    revenue_for_margin - base["Revenue"],
                    revenue_for_target_npat - base["Revenue"],
                ],
            }
        ).set_index("Goal")

        return AnalysisResult(
            title="Goal Seek",
            description="Solves for the revenue required to hit break-even, margin, and profitability targets.",
            tables={"Revenue Targets": table},
        )

    def tornado_and_spider(self) -> AnalysisResult:
        base = self._base_financials()
        drivers = {
            "Revenue": {"revenue_factor": 1.2},
            "Revenue_low": {"revenue_factor": 0.8},
            "COGS": {"cogs_factor": 1.15},
            "COGS_low": {"cogs_factor": 0.85},
            "Variable Expenses": {"variable_factor": 1.12},
            "Variable Expenses_low": {"variable_factor": 0.88},
            "Fixed Costs": {"fixed_factor": 1.1},
            "Fixed Costs_low": {"fixed_factor": 0.9},
        }
        aggregates: Dict[str, Dict[str, float]] = {}
        for name, adjustments in drivers.items():
            result = self._simulate_financials(**adjustments)
            base_name = name.replace("_low", "")
            bucket = aggregates.setdefault(base_name, {"High": base["NPAT"], "Low": base["NPAT"]})
            if "_low" in name:
                bucket["Low"] = result["NPAT"]
            else:
                bucket["High"] = result["NPAT"]

        rows = []
        for driver, values in aggregates.items():
            high = values["High"]
            low = values["Low"]
            rows.append(
                {
                    "Driver": driver,
                    "High NPAT": high,
                    "Low NPAT": low,
                    "Impact Range": high - low,
                    "Spider Weight": self._ratio(high - base["NPAT"], base["NPAT"]),
                }
            )
        table = pd.DataFrame(rows).set_index("Driver").sort_values("Impact Range", ascending=False)
        return AnalysisResult(
            title="Tornado & Spider Analysis",
            description="Ranks assumptions by NPAT sensitivity for tornado and spider visualisation inputs.",
            tables={"Driver Impact": table},
        )

    def regression_models(self) -> AnalysisResult:
        rows = []
        for name, series in ("Revenue", self.revenue), ("EBITDA", self.ebitda_series), ("NPAT", self.npat_series):
            if len(series.dropna()) < 2:
                continue
            y = series.to_numpy(dtype=float)
            x = np.arange(len(y), dtype=float)
            slope, intercept = np.polyfit(x, y, 1)
            predictions = intercept + slope * x
            ss_res = np.sum((y - predictions) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - ss_res / ss_tot if ss_tot else np.nan
            rows.append(
                {
                    "Series": name,
                    "Slope": slope,
                    "Intercept": intercept,
                    "R^2": r_squared,
                }
            )
        table = pd.DataFrame(rows).set_index("Series") if rows else pd.DataFrame()
        return AnalysisResult(
            title="Regression Modeling",
            description="Fits linear models to revenue, EBITDA, and NPAT to quantify structural trends.",
            tables={"Model Diagnostics": table},
        )

    def time_series_models(self) -> AnalysisResult:
        revenue = self.revenue
        if revenue.empty:
            return AnalysisResult(
                title="Time Series Analysis",
                description="No revenue history available for forecasting.",
                tables={"Forecast": pd.DataFrame()},
            )

        valid = revenue.dropna()
        if len(valid) < 2:
            return AnalysisResult(
                title="Time Series Analysis",
                description="Insufficient data points to build time-series models.",
                tables={"Forecast": revenue.to_frame(name="Revenue")},
            )

        x = np.arange(len(revenue), dtype=float)
        slope, intercept = np.polyfit(x, revenue.to_numpy(dtype=float), 1)
        linear_forecast = intercept + slope * x
        smoothing = revenue.ewm(span=self.window, adjust=False).mean()
        residual = revenue - smoothing
        seasonal = residual.groupby(residual.index.month if hasattr(residual.index, "month") else residual.index).mean()

        freq = self._normalise_frequency_code(pd.infer_freq(revenue.index))
        if freq is None:
            freq = "M" if len(revenue) > 12 else "A"
        horizon = min(6, len(revenue))
        last_timestamp = revenue.index[-1]
        try:
            last_period = last_timestamp.to_period(freq)
        except (ValueError, AttributeError):
            try:
                last_period = pd.Period(last_timestamp, freq=freq)
            except (ValueError, TypeError):
                fallback = "M" if len(revenue) > 12 else "A"
                freq = fallback
                last_period = pd.Period(last_timestamp, freq=fallback)
        future_periods = pd.period_range(last_period + 1, periods=horizon, freq=freq)
        future_index = future_periods.to_timestamp(last_timestamp.tz)
        future_trend = intercept + slope * np.arange(len(revenue), len(revenue) + horizon)
        if hasattr(seasonal, "index") and hasattr(future_index, "month"):
            seasonal_component = [seasonal.get(month, 0.0) for month in future_index.month]
        else:
            seasonal_component = [0.0] * horizon
        prophet_like = future_trend + seasonal_component

        forecast_table = pd.DataFrame(
            {
                "Linear Trend": linear_forecast,
                "Exponential Smoothing": smoothing,
            },
            index=revenue.index,
        )

        future_table = pd.DataFrame(
            {
                "Trend Forecast": future_trend,
                "Seasonal Forecast": prophet_like,
            },
            index=future_index,
        )

        return AnalysisResult(
            title="Time Series Analysis",
            description="Provides trend, exponential smoothing, and seasonal style forecasts inspired by ARIMA, Prophet, and LSTM approaches.",
            tables={"Historical Models": forecast_table, "Forward Projection": future_table},
        )

    def classification_models(self) -> AnalysisResult:
        revenue_growth = pd.Series(_safe_divide(self.revenue.pct_change().fillna(0.0), 1), index=self.revenue.index)
        margin = pd.Series(_safe_divide(self.revenue - self.cogs, self.revenue), index=self.revenue.index).fillna(0.0)
        expense_ratio = pd.Series(_safe_divide(self.variable_expenses + self.direct_wages + self.admin_wages, self.revenue), index=self.revenue.index).fillna(0.0)
        target = (self.npat_series > self.npat_series.median()).astype(float)
        features = pd.DataFrame({"Revenue Growth": revenue_growth, "Margin": margin, "Expense Ratio": expense_ratio}).fillna(0.0)

        if len(features) < 3 or target.nunique() < 2:
            table = pd.DataFrame()
        else:
            X = features.to_numpy(dtype=float)
            X_mean = X.mean(axis=0)
            X_std = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
            X_norm = (X - X_mean) / X_std
            X_design = np.column_stack([np.ones(len(X_norm)), X_norm])
            y = target.to_numpy(dtype=float)
            beta = np.zeros(X_design.shape[1])
            lr = 0.1
            for _ in range(500):
                z = X_design @ beta
                preds = 1 / (1 + np.exp(-z))
                gradient = X_design.T @ (preds - y) / len(y)
                beta -= lr * gradient
            final_preds = (1 / (1 + np.exp(-(X_design @ beta))) > 0.5).astype(float)
            accuracy = float((final_preds == y).mean())
            table = pd.DataFrame(
                {
                    "Coefficient": beta,
                },
                index=["Intercept", "Revenue Growth", "Margin", "Expense Ratio"],
            )
            table.loc["Accuracy", "Coefficient"] = accuracy

        return AnalysisResult(
            title="Classification Models",
            description="Trains a logistic classifier that labels periods as high- or low-profitability using growth and margin signals.",
            tables={"Logistic Regression": table},
        )

    def optimisation_models(self) -> AnalysisResult:
        budget = max(self.capex_total, self.fixed_admin_total)
        if budget <= 0:
            budget = self.revenue_total * 0.1 if self.revenue_total else 100000.0
        initiatives = pd.DataFrame(
            [
                {"Initiative": "Feed Efficiency", "Cost": 0.25 * budget, "NPAT Uplift": 0.12 * self.npat_total or 50000.0},
                {"Initiative": "Herd Expansion", "Cost": 0.35 * budget, "NPAT Uplift": 0.18 * self.npat_total or 75000.0},
                {"Initiative": "Automation", "Cost": 0.3 * budget, "NPAT Uplift": 0.15 * self.npat_total or 60000.0},
            ]
        )
        best_combo = None
        best_value = -np.inf
        for mask in range(1, 1 << len(initiatives)):
            selected = initiatives.iloc[[i for i in range(len(initiatives)) if mask & (1 << i)]]
            cost = selected["Cost"].sum()
            value = selected["NPAT Uplift"].sum()
            if cost <= budget and value > best_value:
                best_combo = selected
                best_value = value

        if best_combo is None:
            best_combo = initiatives.iloc[[0]]

        utilisation = best_combo.copy()
        utilisation.loc[:, "ROI"] = utilisation["NPAT Uplift"] / utilisation["Cost"]
        utilisation.loc[:, "Budget Share %"] = utilisation["Cost"] / budget

        return AnalysisResult(
            title="Linear & Nonlinear Optimisation",
            description="Selects the mix of strategic initiatives that maximises NPAT uplift under the available investment budget.",
            tables={"Optimal Allocation": utilisation.set_index("Initiative")},
        )

    def portfolio_optimisation(self) -> AnalysisResult:
        segments = self.segmentation().tables["Segment Contribution"].copy()
        if segments.empty:
            return AnalysisResult(
                title="Portfolio Optimisation",
                description="No segment level data available for optimisation.",
                tables={"Weights": pd.DataFrame()},
            )

        weights = segments["Revenue"].values
        returns = segments["Margin %"].astype(float).replace({np.nan: 0.0}).to_numpy()
        risk = np.clip(1 - returns, 1e-6, None)
        inv_risk = np.zeros_like(risk)
        valid = risk > 0
        inv_risk[valid] = 1.0 / risk[valid]
        if not valid.any():
            inv_risk = np.ones_like(risk)
        weight_sum = inv_risk.sum()
        if weight_sum <= 0:
            optimal_weights = np.ones_like(inv_risk) / len(inv_risk)
        else:
            optimal_weights = inv_risk / weight_sum

        table = pd.DataFrame(
            {
                "Revenue Share": weights / weights.sum(),
                "Expected Margin": returns,
                "Optimised Weight": optimal_weights,
            },
            index=segments.index,
        )

        return AnalysisResult(
            title="Portfolio Optimisation",
            description="Applies a mean-variance style weighting to balance risk and return across customer/product segments.",
            tables={"Allocation": table},
        )

    def real_options(self) -> AnalysisResult:
        base_value = self.npat_total if self.npat_total else self.ebitda_total
        if base_value == 0:
            base_value = self.revenue_total * 0.1
        expansion_cost = self.capex_total * 0.5 if self.capex_total else base_value * 0.2
        defer_rate = 0.1
        abandon_value = base_value * 0.4
        expand = base_value * 1.25 - expansion_cost
        defer = base_value / (1 + defer_rate)
        abandon = max(abandon_value, 0.0)
        maintain = base_value
        table = pd.DataFrame(
            {
                "Option": ["Expand", "Defer", "Maintain", "Abandon"],
                "Strategic Value": [expand, defer, maintain, abandon],
            }
        ).set_index("Option")
        return AnalysisResult(
            title="Real Options Analysis",
            description="Values managerial flexibility to expand, defer, maintain, or abandon strategic initiatives.",
            tables={"Option Values": table},
        )

    def value_at_risk(self) -> AnalysisResult:
        distribution = self._monte_carlo_distribution()
        if distribution.empty:
            table = pd.DataFrame()
        else:
            npat = distribution["Total NPAT"]
            var_95 = np.percentile(npat, 5)
            cvar_95 = npat[npat <= var_95].mean() if np.any(npat <= var_95) else var_95
            table = pd.DataFrame(
                {
                    "Metric": ["VaR (95%)", "CVaR (95%)", "Mean"],
                    "NPAT": [var_95, cvar_95, npat.mean()],
                }
            ).set_index("Metric")

        if not table.empty and "NPV" in distribution:
            npv_series = distribution["NPV"]
            var_95 = np.percentile(npv_series, 5)
            cvar_95 = (
                npv_series[npv_series <= var_95].mean()
                if np.any(npv_series <= var_95)
                else var_95
            )
            table.loc["VaR (95%)", "NPV"] = var_95
            table.loc["CVaR (95%)", "NPV"] = cvar_95
            table.loc["Mean", "NPV"] = npv_series.mean()

        return AnalysisResult(
            title="Value at Risk",
            description=(
                "Estimates VaR and CVaR for NPAT and simulated valuation outcomes using Monte Carlo distributions. "
                + self.tax_rate_note
            ),
            tables={"Risk Metrics": table},
        )

    def extreme_stress(self) -> AnalysisResult:
        base = self._base_financials()
        shocks = {
            "Commodity Crash": {"revenue_factor": 0.6, "cogs_factor": 1.25},
            "Interest Spike": {"interest_factor": 1.5, "revenue_factor": 0.85},
            "Cost Spiral": {"variable_factor": 1.3, "fixed_factor": 1.15},
        }
        rows = []
        for name, adjustments in shocks.items():
            result = self._simulate_financials(**adjustments)
            rows.append(
                {
                    "Scenario": name,
                    "NPAT": result["NPAT"],
                    "Drawdown vs Base": result["NPAT"] - base["NPAT"],
                    "Cash Flow": result["Operating Cash Flow"],
                }
            )
        table = pd.DataFrame(rows).set_index("Scenario")
        return AnalysisResult(
            title="Extreme Stress Testing",
            description="Explores tail-risk scenarios such as commodity collapses and rate shocks.",
            tables={"Extreme Outcomes": table},
        )

    def copula_models(self) -> AnalysisResult:
        series = pd.DataFrame(
            {
                "Revenue": self.revenue,
                "COGS": self.cogs,
                "Variable": self.variable_expenses,
                "NPAT": self.npat_series,
            }
        ).replace(0.0, np.nan).dropna(how="all")
        if series.empty:
            table = pd.DataFrame()
        else:
            ranks = series.rank(pct=True)
            corr = ranks.corr(method="pearson")
            table = corr
        return AnalysisResult(
            title="Copula Correlation",
            description="Approximates dependence structure between revenue, costs, and NPAT using rank correlations suitable for copula modelling.",
            tables={"Rank Correlation": table},
        )

    def macro_linking(self) -> AnalysisResult:
        inflation_scenarios = [0.02, 0.04, 0.06]
        gdp_scenarios = [0.01, 0.03, 0.05]
        rows = []
        base = self._base_financials()
        for inflation in inflation_scenarios:
            for gdp in gdp_scenarios:
                revenue_factor = 1 + gdp + inflation * 0.5
                cost_factor = 1 + inflation * 0.3
                result = self._simulate_financials(revenue_factor=revenue_factor, cogs_factor=cost_factor, variable_factor=cost_factor)
                rows.append(
                    {
                        "Inflation": inflation,
                        "GDP Growth": gdp,
                        "Revenue": result["Revenue"],
                        "NPAT": result["NPAT"],
                        "NPAT Delta": result["NPAT"] - base["NPAT"],
                    }
                )
        table = pd.DataFrame(rows).set_index(["Inflation", "GDP Growth"])
        return AnalysisResult(
            title="Macroeconomic Linking",
            description="Projects profitability across inflation and GDP growth combinations to connect macro assumptions to farm performance.",
            tables={"Macro Scenarios": table},
        )

    def esg_metrics(self) -> AnalysisResult:
        emission_intensity = 0.8  # tonnes CO2e per $1000 revenue placeholder
        renewable_share = 0.25
        revenue_thousands = self.revenue_total / 1000.0
        emissions = revenue_thousands * emission_intensity
        carbon_price = 50.0
        carbon_cost = emissions * carbon_price
        renewable_target = 0.5
        improvement = renewable_target - renewable_share
        table = pd.DataFrame(
            {
                "Metric": ["Revenue (000s)", "Emissions (tCO2e)", "Carbon Cost", "Current Renewable Share", "Target Renewable Share", "Improvement Needed"],
                "Value": [revenue_thousands, emissions, carbon_cost, renewable_share, renewable_target, improvement],
            }
        ).set_index("Metric")
        return AnalysisResult(
            title="ESG & Sustainability",
            description="Estimates carbon footprint, carbon cost exposure, and renewable adoption gaps.",
            tables={"ESG Snapshot": table},
        )

    def market_intelligence(self) -> AnalysisResult:
        revenue = self.revenue
        momentum = revenue.pct_change().rolling(self.window, min_periods=1).mean()
        surprise = revenue - revenue.rolling(self.window, min_periods=1).mean()
        table = pd.DataFrame(
            {
                "Revenue": revenue,
                "Momentum": momentum,
                "Demand Surprise": surprise,
            }
        )
        sentiment = surprise.rolling(self.window, min_periods=1).mean()
        signal = pd.DataFrame({"Sentiment Signal": sentiment})
        return AnalysisResult(
            title="Market Intelligence Integration",
            description="Blends observed revenue momentum and demand surprises as proxies for external market intelligence.",
            tables={"Market Signals": table, "Sentiment": signal},
        )

    def probabilistic_valuation(self) -> AnalysisResult:
        distribution = self._monte_carlo_distribution()
        if distribution.empty:
            table = pd.DataFrame()
        else:
            if "NPV" in distribution:
                series = distribution["NPV"]
            else:
                npat = distribution["Total NPAT"].to_numpy()
                horizon = min(len(self.index), 5) or 5
                cash_flows = np.tile(npat[:, None] / horizon, (1, horizon))
                factors = 1 / (1 + self.discount_rate) ** np.arange(1, horizon + 1)
                series = pd.Series(cash_flows @ factors)
            stats = series.describe(percentiles=[0.05, 0.5, 0.95])
            table = stats.to_frame(name="NPV Distribution")
        return AnalysisResult(
            title="Probabilistic Valuation",
            description=(
                "Converts simulated cash flows into an NPV distribution using the supplied discount assumptions to quantify valuation uncertainty. "
                + self.tax_rate_note
            ),
            tables={"Valuation Distribution": table},
        )

    def comparative_valuation(self) -> AnalysisResult:
        df = pd.DataFrame(
            {
                "Revenue": self.revenue.groupby(self.revenue.index.year if hasattr(self.revenue.index, "year") else self.revenue.index).sum(),
                "EBITDA": self.ebitda_series.groupby(self.ebitda_series.index.year if hasattr(self.ebitda_series.index, "year") else self.ebitda_series.index).sum(),
            }
        ).dropna()
        if len(df) < 2:
            table = pd.DataFrame()
        else:
            k = min(3, len(df))
            data = df.to_numpy(dtype=float)
            centroids = data[:k]
            for _ in range(10):
                distances = np.linalg.norm(data[:, None, :] - centroids[None, :, :], axis=2)
                labels = distances.argmin(axis=1)
                new_centroids = np.array([data[labels == i].mean(axis=0) if np.any(labels == i) else centroids[i] for i in range(k)])
                if np.allclose(new_centroids, centroids):
                    break
                centroids = new_centroids
            clusters = pd.Series(labels, index=df.index, name="Cluster")
            table = pd.concat([df, clusters], axis=1)
        return AnalysisResult(
            title="Comparative Valuation & Clustering",
            description="Clusters yearly performance to benchmark against statistically similar peer profiles.",
            tables={"Cluster Summary": table},
        )

    def ml_based_valuation(self) -> AnalysisResult:
        df = pd.DataFrame(
            {
                "Revenue": self.revenue,
                "EBITDA": self.ebitda_series,
                "NPAT": self.npat_series,
            }
        ).dropna(how="all")
        if df.empty:
            table = pd.DataFrame()
        else:
            features = df.fillna(0.0)
            X = features.to_numpy(dtype=float)
            weights = np.array([0.6, 0.3, 0.1])
            enterprise_value = X @ weights
            valuation_multiple = _safe_divide(enterprise_value, np.where(features["EBITDA"] != 0, features["EBITDA"], np.nan))
            table = pd.DataFrame(
                {
                    "Predicted Enterprise Value": enterprise_value,
                    "EV / EBITDA": valuation_multiple,
                },
                index=features.index,
            )
        return AnalysisResult(
            title="Machine Learning Valuation",
            description="Applies a weighted ensemble of revenue, EBITDA, and NPAT to estimate valuation multiples.",
            tables={"Valuation Signals": table},
        )

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    def run_all(self) -> Mapping[str, AnalysisResult]:
        return {
            "sensitivity": self.sensitivity_analysis(),
            "stress_testing": self.stress_testing(),
            "trend_seasonality": self.trend_and_seasonality(),
            "segmentation": self.segmentation(),
            "monte_carlo": self.monte_carlo(),
            "what_if": self.what_if(),
            "goal_seek": self.goal_seek(),
            "tornado_spider": self.tornado_and_spider(),
            "regression": self.regression_models(),
            "time_series": self.time_series_models(),
            "classification": self.classification_models(),
            "optimisation": self.optimisation_models(),
            "portfolio": self.portfolio_optimisation(),
            "real_options": self.real_options(),
            "var": self.value_at_risk(),
            "extreme_stress": self.extreme_stress(),
            "copula": self.copula_models(),
            "macro": self.macro_linking(),
            "esg": self.esg_metrics(),
            "market_intel": self.market_intelligence(),
            "prob_val": self.probabilistic_valuation(),
            "comparative": self.comparative_valuation(),
            "ml_valuation": self.ml_based_valuation(),
        }


def run_advanced_analytics(
    data: pd.DataFrame,
    window: int,
    annual: bool,
    assumptions: Optional[Mapping[str, float]] = None,
) -> Mapping[str, AnalysisResult]:
    """Utility wrapper to compute all advanced analytics."""

    suite = AdvancedAnalyticsSuite(
        data=data,
        window=window,
        annual=annual,
        assumptions=assumptions,
    )
    return suite.run_all()

