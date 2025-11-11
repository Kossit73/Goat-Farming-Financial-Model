"""Utilities for reading and manipulating the goat farming financial model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd


@dataclass
class GoatModel:
    excel_path: str
    ref_sheet: str = "IS"
    _df_is: Optional[pd.DataFrame] = None
    _df_cf: Optional[pd.DataFrame] = None
    _df_val: Optional[pd.DataFrame] = None
    _dates: Optional[pd.DatetimeIndex] = None

    # ---------- Internal loading & helpers ----------
    def _load(self) -> None:
        if self._df_is is None:
            self._df_is = pd.read_excel(self.excel_path, sheet_name="IS", header=None)
        if self._df_cf is None:
            self._df_cf = pd.read_excel(self.excel_path, sheet_name="CF", header=None)
        if self._df_val is None:
            self._df_val = pd.read_excel(self.excel_path, sheet_name="Valuation", header=None)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Monthly period end dates inferred from the reference sheet's header (row 6, columns 8+)."""
        self._load()
        if self._dates is None:
            hdr_row = 6
            date_cells = pd.to_datetime(self._df_is.iloc[hdr_row, 8:], errors="coerce").dropna()
            self._dates = pd.DatetimeIndex(date_cells.values)
        return self._dates

    def _extract_series(self, df: pd.DataFrame, label: str) -> Optional[pd.Series]:
        """Return numeric series across timeline columns by matching a row containing `label`."""
        idxs = df.index[(df == label).any(axis=1)].tolist()
        if not idxs:
            idxs = [
                i
                for i in range(len(df))
                if any(
                    str(x).strip().lower().startswith(label.lower())
                    for x in df.iloc[i, :6].tolist()
                )
            ]
        if not idxs:
            return None
        row = df.loc[idxs[0]]
        n_dates = len(self.dates)
        values = pd.to_numeric(row.iloc[8 : 8 + n_dates], errors="coerce")
        return pd.Series(values.values, index=self.dates, name=label)

    # ---------- Base series ----------
    def revenue(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_is, "Revenue")

    def cogs(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_is, "COGS")

    def gross_margin(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_is, "GROSS MARGIN")

    def ebitda(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_is, "EBITDA")

    def depreciation(self) -> Optional[pd.Series]:
        self._load()
        series = self._extract_series(self._df_is, "Total Depreciation & Amortization")
        if series is None:
            series = self._extract_series(self._df_is, "Depreciation & Amortization ")
        return series

    def ebit(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_is, "EBIT")

    def npbt(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_is, "Net Profit Before Tax")

    def npat(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_is, "Net Profit After Tax")

    def cfo(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_cf, "Net Cash Flow from Operating Activities")

    def cfi(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_cf, "Net Cash Flow from Investing Activities")

    def cff(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(self._df_cf, "Net Cash Flow from Financing Activities")

    def net_cash_flow(self) -> Optional[pd.Series]:
        cfo = self.cfo()
        cfi = self.cfi()
        cff = self.cff()
        parts = [s for s in (cfo, cfi, cff) if s is not None]
        if not parts:
            return None
        return pd.concat(parts, axis=1).sum(axis=1, min_count=1).rename("Net Cash Flow")

    # ---------- Valuation ----------
    def wacc(self) -> Optional[float]:
        self._load()
        row_idx = None
        for i in range(len(self._df_val)):
            if any(
                isinstance(v, str) and "weighted avg cost of capital" in v.lower()
                for v in self._df_val.iloc[i, :6].tolist()
            ):
                row_idx = i
                break
        if row_idx is None:
            return None
        nums = pd.to_numeric(self._df_val.loc[row_idx], errors="coerce").dropna()
        return float(nums.iloc[0]) if len(nums) else None

    def ufcf(self) -> Optional[pd.Series]:
        self._load()
        idxs = self._df_val.index[(self._df_val == "Unlevered Free Cash Flow").any(axis=1)].tolist()
        if not idxs:
            return None
        row = self._df_val.loc[idxs[0]]
        values = pd.to_numeric(row, errors="coerce").dropna()
        years = sorted(set(pd.DatetimeIndex(self.dates).year))
        n = min(len(years), len(values))
        if n == 0:
            return None
        return pd.Series(
            values.iloc[:n].values,
            index=pd.to_datetime([f"{y}-12-31" for y in years[:n]]),
            name="Unlevered Free Cash Flow",
        )

    def npv(self) -> Optional[float]:
        self._load()
        idxs = self._df_val.index[(self._df_val == "NPV based on year 5").any(axis=1)].tolist()
        if not idxs:
            return None
        row = self._df_val.loc[idxs[0]]
        nums = pd.to_numeric(row, errors="coerce").dropna()
        return float(nums.iloc[-1]) if len(nums) else None

    # ---------- Scenario toggles ----------
    def scenario(self, milk_price_pct: float = 0.0, feed_cost_pct: float = 0.0) -> pd.DataFrame:
        """Apply shocks to milk price (+/-) and feed cost (+/-), recompute IS metrics."""
        rev = self.revenue()
        cogs = self.cogs()
        gm = self.gross_margin()
        ebitda = self.ebitda()
        dep = self.depreciation()
        ebit = self.ebit()
        npbt = self.npbt()
        npat = self.npat()

        series_map: Dict[str, Optional[pd.Series]] = {
            "Revenue": rev,
            "COGS": cogs,
            "Gross Margin": gm,
            "EBITDA": ebitda,
            "Depreciation & Amortization": dep,
            "EBIT": ebit,
            "NPBT": npbt,
            "NPAT": npat,
        }
        valid_series = {k: v for k, v in series_map.items() if v is not None}
        if not valid_series:
            raise ValueError("Scenario analysis requires base income statement series.")

        base = pd.concat(valid_series, axis=1)

        if "Revenue" not in base or "COGS" not in base:
            raise ValueError("Scenario analysis requires both Revenue and COGS series.")

        base["Revenue_adj"] = base["Revenue"] * (1 + milk_price_pct)
        base["COGS_adj"] = base["COGS"] * (1 + feed_cost_pct)
        base["Gross Margin_adj"] = base["Revenue_adj"] - base["COGS_adj"]

        if "Gross Margin" in base:
            opex_ex_da = base["Gross Margin"] - base.get("EBITDA", 0)
        else:
            opex_ex_da = base["Revenue"] - base["COGS"] - base.get("EBITDA", 0)

        base["EBITDA_adj"] = base["Gross Margin_adj"] - opex_ex_da
        base["EBIT_adj"] = base["EBITDA_adj"] - base.get("Depreciation & Amortization", 0)

        if npbt is not None and npat is not None and npbt.notna().any() and npat.notna().any():
            idx = npbt.notna() & npat.notna()
            eff_tax = 1 - (npat[idx] / npbt[idx]).median()
            eff_tax = float(np.clip(eff_tax, 0.0, 0.5))
        else:
            eff_tax = 0.28
        base["NPAT_adj"] = base["EBIT_adj"] * (1 - eff_tax)
        return base

    # ---------- KPIs ----------
    def kpis(self, df: Optional[pd.DataFrame] = None, annual: bool = True) -> pd.DataFrame:
        """Compute Gross Margin %, EBITDA Margin %, Net Margin %, YoY growth."""
        if df is None:
            df = self.to_tidy()

        rev_col = "Revenue_adj" if "Revenue_adj" in df else "Revenue"
        gm_col = "Gross Margin_adj" if "Gross Margin_adj" in df else "Gross Margin"
        ebitda_col = "EBITDA_adj" if "EBITDA_adj" in df else "EBITDA"
        npat_col = "NPAT_adj" if "NPAT_adj" in df else "NPAT"
        cogs_col = "COGS_adj" if "COGS_adj" in df else "COGS"

        work = df[[rev_col, gm_col, ebitda_col, npat_col, cogs_col]].rename(
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
        """Compute break-even revenue per period."""
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
        series: Dict[str, Optional[pd.Series]] = {
            "Revenue": self.revenue(),
            "COGS": self.cogs(),
            "Gross Margin": self.gross_margin(),
            "EBITDA": self.ebitda(),
            "Depreciation & Amortization": self.depreciation(),
            "EBIT": self.ebit(),
            "NPBT": self.npbt(),
            "NPAT": self.npat(),
            "CFO": self.cfo(),
            "CFI": self.cfi(),
            "CFF": self.cff(),
            "Net Cash Flow": self.net_cash_flow(),
        }
        valid = {k: v for k, v in series.items() if v is not None}
        if not valid:
            raise ValueError("No financial series could be extracted from the workbook.")
        return pd.concat(valid, axis=1)
