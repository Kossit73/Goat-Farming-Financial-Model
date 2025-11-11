"""Utilities for manipulating the goat farming financial model without Excel."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

SeriesLabels = Sequence[str]


@dataclass
class InputSchedule:
    """Container for manually entered time-series data and supplementary tables."""

    data: pd.DataFrame
    valuation_inputs: Dict[str, float] = field(default_factory=dict)
    supplementary_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Input schedule data must be indexed by datetimes.")
        if self.data.index.has_duplicates:
            raise ValueError("Input schedule periods must be unique.")
        self.data = self.data.sort_index()
        self.data.index.name = "Period"
        self.data = self.data.apply(pd.to_numeric, errors="coerce")

        cleaned_tables: Dict[str, pd.DataFrame] = {}
        for name, table in self.supplementary_tables.items():
            if table is None or table.empty:
                continue
            cleaned_tables[name] = _clean_table(table)
        self.supplementary_tables = cleaned_tables

    @property
    def timeline(self) -> pd.DatetimeIndex:
        return self.data.index

    @classmethod
    def from_frame(
        cls,
        frame: pd.DataFrame,
        period_col: str = "Period",
        valuation_inputs: Optional[Dict[str, float]] = None,
        supplementary_tables: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> "InputSchedule":
        if period_col not in frame.columns:
            raise ValueError(f"Expected a '{period_col}' column in the input schedule.")

        periods = pd.to_datetime(frame[period_col], errors="coerce")
        if periods.isna().any():
            raise ValueError("Unable to parse one or more period values into dates.")

        values = frame.drop(columns=[period_col]).apply(pd.to_numeric, errors="coerce")
        values.index = pd.DatetimeIndex(periods)
        values.index.name = "Period"

        return cls(
            data=values,
            valuation_inputs=valuation_inputs or {},
            supplementary_tables=supplementary_tables or {},
        )

    def to_model(self) -> "GoatModel":
        """Instantiate :class:`GoatModel` using the stored data."""

        return GoatModel(
            data=self.data.copy(),
            valuation_inputs=dict(self.valuation_inputs),
            supplementary_tables=dict(self.supplementary_tables),
        )


@dataclass
class GoatModel:
    """Helper for extracting series and performing analytics on manual inputs."""

    data: pd.DataFrame
    valuation_inputs: Dict[str, float] = field(default_factory=dict)
    supplementary_tables: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.data.index, pd.DatetimeIndex):
            raise ValueError("Model data must be indexed by datetimes.")
        if self.data.index.has_duplicates:
            raise ValueError("Model periods must be unique.")
        self.data = self.data.sort_index()
        self.data.index.name = "Period"
        self.data = self.data.apply(pd.to_numeric, errors="coerce")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.data.index

    # ---------- Internal helpers ----------
    def _get_series(self, labels: SeriesLabels) -> Optional[pd.Series]:
        for label in labels:
            if label in self.data.columns:
                series = pd.to_numeric(self.data[label], errors="coerce")
                return pd.Series(series.values, index=self.dates, name=label)
        return None

    # ---------- Base series ----------
    def revenue(self) -> Optional[pd.Series]:
        return self._get_series(("Revenue", "Total Revenue", "Sales Revenue"))

    def cogs(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "COGS",
                "Cost of Goods Sold",
                "Cost of Sales",
            )
        )

    def variable_expenses(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Variable Expenses",
                "Variable Operating Expenses",
                "Variable Costs",
            )
        )

    def fixed_expenses(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Fixed Expenses",
                "Fixed Operating Expenses",
                "Fixed Costs",
            )
        )

    def direct_wages(self) -> Optional[pd.Series]:
        return self._get_series(("Direct Wages", "Direct Labour", "Direct Labor"))

    def admin_wages(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Admin Wages",
                "Administrative Wages",
                "Admin Salaries",
            )
        )

    def gross_margin(self) -> Optional[pd.Series]:
        return self._get_series(("Gross Margin", "Gross Profit"))

    def ebitda(self) -> Optional[pd.Series]:
        return self._get_series(("EBITDA", "Operating EBITDA"))

    def depreciation(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Depreciation & Amortization",
                "Depreciation and Amortisation",
                "Depreciation",
            )
        )

    def ebit(self) -> Optional[pd.Series]:
        return self._get_series(("EBIT", "Operating Profit"))

    def npbt(self) -> Optional[pd.Series]:
        return self._get_series(("Net Profit Before Tax", "Profit Before Tax"))

    def npat(self) -> Optional[pd.Series]:
        return self._get_series(("Net Profit After Tax", "Net Income"))

    def cfo(self) -> Optional[pd.Series]:
        return self._get_series(("CFO", "Operating Cash Flow"))

    def cfi(self) -> Optional[pd.Series]:
        return self._get_series(("CFI", "Investing Cash Flow"))

    def cff(self) -> Optional[pd.Series]:
        return self._get_series(("CFF", "Financing Cash Flow"))

    def capex(self) -> Optional[pd.Series]:
        return self._get_series(("Capex", "Capital Expenditure"))

    def net_cash_flow(self) -> Optional[pd.Series]:
        if "Net Cash Flow" in self.data.columns:
            return self._get_series(("Net Cash Flow",))

        cfo = self.cfo()
        cfi = self.cfi()
        cff = self.cff()
        parts = [s for s in (cfo, cfi, cff) if s is not None]
        if not parts:
            return None
        return pd.concat(parts, axis=1).sum(axis=1, min_count=1).rename("Net Cash Flow")

    # ---------- Valuation ----------
    def wacc(self) -> Optional[float]:
        value = self.valuation_inputs.get("WACC")
        if value is None:
            return None
        return float(value)

    def npv(self) -> Optional[float]:
        value = self.valuation_inputs.get("NPV")
        if value is None:
            return None
        return float(value)

    def terminal_value(self) -> Optional[float]:
        value = self.valuation_inputs.get("Terminal Value")
        if value is None:
            return None
        return float(value)

    def ufcf(self) -> Optional[pd.Series]:
        table = None
        for key in ("UFCF", "Unlevered Free Cash Flow"):
            if key in self.supplementary_tables:
                table = self.supplementary_tables[key]
                break
        if table is None or table.empty:
            return None

        df = table.copy()
        if "Period" in df.columns:
            idx = pd.to_datetime(df["Period"], errors="coerce")
            values = pd.to_numeric(df.iloc[:, -1], errors="coerce")
        else:
            idx = pd.to_datetime(df.index, errors="coerce")
            values = pd.to_numeric(df.iloc[:, -1], errors="coerce")

        mask = idx.notna() & values.notna()
        if not mask.any():
            return None
        return pd.Series(values[mask].to_numpy(), index=pd.DatetimeIndex(idx[mask]), name="Unlevered Free Cash Flow")

    # ---------- Supplementary schedules ----------
    def capitalisation_table(self) -> Optional[pd.DataFrame]:
        return self.supplementary_tables.get("Capitalisation Table")

    def capex_schedule(self) -> Optional[pd.DataFrame]:
        return self.supplementary_tables.get("Capex Schedule")

    def asset_schedules(self) -> Optional[pd.DataFrame]:
        return self.supplementary_tables.get("Asset Schedules")

    def outputs(self) -> Optional[pd.DataFrame]:
        return self.supplementary_tables.get("Outputs")

    def benchmark_kpis(self) -> Optional[pd.DataFrame]:
        return self.supplementary_tables.get("Benchmark KPIs")

    # ---------- Scenario toggles ----------
    def scenario(self, milk_price_pct: float = 0.0, feed_cost_pct: float = 0.0) -> pd.DataFrame:
        """Apply shocks to milk price and feed cost, recomputing adjusted metrics."""

        base_cols = {
            "Revenue": self.revenue(),
            "COGS": self.cogs(),
            "Gross Margin": self.gross_margin(),
            "EBITDA": self.ebitda(),
            "Depreciation & Amortization": self.depreciation(),
            "EBIT": self.ebit(),
            "NPBT": self.npbt(),
            "NPAT": self.npat(),
            "Variable Expenses": self.variable_expenses(),
            "Fixed Expenses": self.fixed_expenses(),
            "Direct Wages": self.direct_wages(),
            "Admin Wages": self.admin_wages(),
        }
        valid = {k: v for k, v in base_cols.items() if v is not None}
        if "Revenue" not in valid or "COGS" not in valid:
            raise ValueError("Scenario analysis requires Revenue and COGS in the schedule.")

        df = pd.concat(valid, axis=1)
        df["Revenue_adj"] = df["Revenue"] * (1 + milk_price_pct)
        df["COGS_adj"] = df["COGS"] * (1 + feed_cost_pct)
        df["Gross Margin_adj"] = df["Revenue_adj"] - df["COGS_adj"]

        if "Gross Margin" in df and "EBITDA" in df:
            opex_ex_da = df["Gross Margin"] - df["EBITDA"]
        elif "EBITDA" in df:
            opex_ex_da = df["Revenue"] - df["COGS"] - df["EBITDA"]
        else:
            opex_ex_da = 0

        df["EBITDA_adj"] = df["Gross Margin_adj"] - opex_ex_da
        df["EBIT_adj"] = df["EBITDA_adj"] - df.get("Depreciation & Amortization", 0)

        npbt = self.npbt()
        npat = self.npat()
        if npbt is not None and npat is not None and npbt.notna().any() and npat.notna().any():
            idx = npbt.notna() & npat.notna()
            eff_tax = 1 - (npat[idx] / npbt[idx]).median()
            eff_tax = float(np.clip(eff_tax, 0.0, 0.5))
        else:
            eff_tax = 0.28
        df["NPAT_adj"] = df["EBIT_adj"] * (1 - eff_tax)
        return df

    # ---------- KPIs ----------
    def kpis(self, df: Optional[pd.DataFrame] = None, annual: bool = True) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        rev_col = "Revenue_adj" if "Revenue_adj" in df else "Revenue"
        gm_col = "Gross Margin_adj" if "Gross Margin_adj" in df else "Gross Margin"
        ebitda_col = "EBITDA_adj" if "EBITDA_adj" in df else "EBITDA"
        npat_col = "NPAT_adj" if "NPAT_adj" in df else "NPAT"
        cogs_col = "COGS_adj" if "COGS_adj" in df else "COGS"

        required = [rev_col, gm_col, ebitda_col, npat_col, cogs_col]
        missing = [col for col in required if col not in df]
        if missing:
            raise ValueError(f"Missing required columns for KPI calculation: {missing}")

        work = df[required].rename(
            columns={
                rev_col: "Revenue",
                gm_col: "Gross Margin",
                ebitda_col: "EBITDA",
                npat_col: "NPAT",
                cogs_col: "COGS",
            }
        )

        if annual:
            grp = work.groupby(work.index.year).sum(min_count=1)
        else:
            grp = work.copy()

        out = pd.DataFrame(index=grp.index)
        out["Gross Margin %"] = grp["Gross Margin"] / grp["Revenue"]
        out["EBITDA Margin %"] = grp["EBITDA"] / grp["Revenue"]
        out["Net Margin %"] = grp["NPAT"] / grp["Revenue"]
        out["COGS % of Revenue"] = grp["COGS"] / grp["Revenue"]
        out["Revenue YoY %"] = grp["Revenue"].pct_change()
        return out

    # ---------- Break-even ----------
    def break_even(self, df: Optional[pd.DataFrame] = None, annual: bool = True) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        rev = df["Revenue_adj"] if "Revenue_adj" in df else df["Revenue"]
        gm = df["Gross Margin_adj"] if "Gross Margin_adj" in df else df["Gross Margin"]
        ebitda = df["EBITDA_adj"] if "EBITDA_adj" in df else df["EBITDA"]

        cm_ratio = gm / rev
        fixed_costs = gm - ebitda

        if annual:
            idx = rev.index.year
            cm_ratio = (gm.groupby(idx).sum(min_count=1) / rev.groupby(idx).sum(min_count=1))
            fixed_costs = fixed_costs.groupby(idx).sum(min_count=1)

        be_rev = fixed_costs / cm_ratio
        return pd.DataFrame(
            {
                "Contribution Margin %": cm_ratio,
                "Fixed Costs (approx)": fixed_costs,
                "Break-even Revenue": be_rev,
            }
        )

    def to_tidy(self) -> pd.DataFrame:
        return self.data.copy()


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").dropna(axis=1, how="all")
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned.reset_index(drop=True)
