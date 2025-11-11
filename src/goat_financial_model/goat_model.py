"""Tools for extracting a financial model from an Excel workbook.

The :class:`GoatModel` reads the expected workbook structure from the three
sheets ``IS`` (income statement), ``CF`` (cash flow), and ``Valuation`` to
provide easy to consume pandas ``Series`` objects and a tidy combined frame for
analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd


@dataclass
class GoatModel:
    """Helper around the Excel-based goat farming model.

    Parameters
    ----------
    excel_path:
        Path to the workbook that contains the three sheets.
    ref_sheet:
        Sheet to use as the reference for inferring the timeline.  Defaults to
        ``"IS"`` because that is the income statement sheet in the template.
    """

    excel_path: str
    ref_sheet: str = "IS"
    _df_is: Optional[pd.DataFrame] = None
    _df_cf: Optional[pd.DataFrame] = None
    _df_val: Optional[pd.DataFrame] = None
    _dates: Optional[pd.DatetimeIndex] = None

    def _load(self) -> None:
        """Lazily load the workbook sheets only when they are required."""

        if self._df_is is None:
            self._df_is = pd.read_excel(self.excel_path, sheet_name="IS", header=None)
        if self._df_cf is None:
            self._df_cf = pd.read_excel(self.excel_path, sheet_name="CF", header=None)
        if self._df_val is None:
            self._df_val = pd.read_excel(self.excel_path, sheet_name="Valuation", header=None)

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Monthly period end dates inferred from the reference sheet."""

        self._load()
        if self._dates is None:
            hdr_row = 6  # "Month" row with period end
            date_cells = pd.to_datetime(self._df_is.iloc[hdr_row, 8:], errors="coerce").dropna()
            self._dates = pd.DatetimeIndex(date_cells.values)
        return self._dates

    def _extract_series(self, df: pd.DataFrame, label: str) -> Optional[pd.Series]:
        """Return a numeric series for the first row that matches ``label``."""

        idxs = df.index[(df == label).any(axis=1)].tolist()
        if not idxs:
            idxs = [
                i
                for i in range(len(df))
                if any(
                    str(x).strip().lower().startswith(label.lower())
                    for x in df.iloc[i, :5].tolist()
                )
            ]
        if not idxs:
            return None
        row = df.loc[idxs[0]]
        n_dates = len(self.dates)
        values = pd.to_numeric(row.iloc[8 : 8 + n_dates], errors="coerce")
        return pd.Series(values.values, index=self.dates, name=label)

    # Income Statement series
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

    # Cash Flow categories
    def cfo(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_cf, "Net Cash Flow from Operating Activities"
        )

    def cfi(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_cf, "Net Cash Flow from Investing Activities"
        )

    def cff(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_cf, "Net Cash Flow from Financing Activities"
        )

    def net_cash_flow(self) -> Optional[pd.Series]:
        cfo = self.cfo()
        cfi = self.cfi()
        cff = self.cff()
        parts = [s for s in (cfo, cfi, cff) if s is not None]
        if not parts:
            return None
        return (
            pd.concat(parts, axis=1)
            .sum(axis=1, min_count=1)
            .rename("Net Cash Flow")
        )

    # Valuation metrics
    def wacc(self) -> Optional[float]:
        self._load()
        row_idx = None
        for i in range(len(self._df_val)):
            if any(
                isinstance(v, str) and "weighted avg cost of capital" in v.lower()
                for v in self._df_val.iloc[i, :5].tolist()
            ):
                row_idx = i
                break
        if row_idx is None:
            return None
        nums = pd.to_numeric(self._df_val.loc[row_idx], errors="coerce").dropna()
        return float(nums.iloc[0]) if len(nums) else None

    def ufcf(self) -> Optional[pd.Series]:
        self._load()
        idxs = self._df_val.index[
            (self._df_val == "Unlevered Free Cash Flow").any(axis=1)
        ].tolist()
        if not idxs:
            return None
        row = self._df_val.loc[idxs[0]]
        values = pd.to_numeric(row, errors="coerce").dropna()
        if values.empty:
            return None
        years = sorted(set(pd.DatetimeIndex(self.dates).year))
        n = min(len(years), len(values))
        return pd.Series(
            values.iloc[:n].values,
            index=pd.to_datetime([f"{y}-12-31" for y in years[:n]]),
            name="Unlevered Free Cash Flow",
        )

    def terminal_value(self) -> Optional[float]:
        self._load()
        idxs = self._df_val.index[(self._df_val == "Terminal Value").any(axis=1)].tolist()
        if not idxs:
            return None
        row = self._df_val.loc[idxs[0]]
        nums = pd.to_numeric(row, errors="coerce").dropna()
        return float(nums.iloc[-1]) if len(nums) else None

    def npv(self) -> Optional[float]:
        self._load()
        idxs = self._df_val.index[
            (self._df_val == "NPV based on year 5").any(axis=1)
        ].tolist()
        if not idxs:
            return None
        row = self._df_val.loc[idxs[0]]
        nums = pd.to_numeric(row, errors="coerce").dropna()
        return float(nums.iloc[-1]) if len(nums) else None

    def to_tidy(self) -> pd.DataFrame:
        """Return a tidy dataframe with the key extracted series stacked."""

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

        valid = {key: value for key, value in series.items() if value is not None}
        if not valid:
            raise ValueError("No financial series could be extracted from the workbook.")

        return pd.concat(valid, axis=1)
