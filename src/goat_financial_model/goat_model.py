"""Tools for extracting a financial model from an Excel workbook.

The :class:`GoatModel` reads the expected workbook structure from the three
sheets ``IS`` (income statement), ``CF`` (cash flow), and ``Valuation`` to
provide easy to consume pandas ``Series`` objects and a tidy combined frame for
analytics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
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

    # ------------------------------------------------------------------
    # Scenario and KPI helpers used by the Streamlit dashboard
    # ------------------------------------------------------------------
    def scenario(
        self,
        milk_price_pct: float = 0.0,
        feed_cost_pct: float = 0.0,
        base: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Apply simple revenue and feed cost shocks to the model.

        Parameters
        ----------
        milk_price_pct:
            Percentage change applied to revenue.  ``0.05`` corresponds to a
            +5% increase, ``-0.1`` to a -10% decrease.
        feed_cost_pct:
            Percentage change applied to COGS which acts as a proxy for feed
            costs in the simplified model.
        base:
            Optional tidy dataframe to start from.  When omitted the workbook is
            loaded through :meth:`to_tidy`.

        Returns
        -------
        pandas.DataFrame
            Copy of the tidy dataframe including ``*_adj`` columns with the
            adjusted values.
        """

        df = self.to_tidy() if base is None else base.copy()

        required_cols = {"Revenue", "COGS"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                "Scenario analysis requires the following columns: "
                + ", ".join(sorted(missing))
            )

        # Base gross margin is either supplied by the sheet or recomputed.
        if "Gross Margin" in df:
            gross_base = df["Gross Margin"].copy()
        else:
            gross_base = df["Revenue"] - df["COGS"]

        df["Revenue_adj"] = df["Revenue"] * (1.0 + float(milk_price_pct))
        df["COGS_adj"] = df["COGS"] * (1.0 + float(feed_cost_pct))

        df["Gross Margin_adj"] = df["Revenue_adj"] - df["COGS_adj"]
        delta_gross = (df["Gross Margin_adj"] - gross_base).fillna(0.0)

        for col in ["EBITDA", "EBIT", "NPBT", "NPAT"]:
            if col in df:
                df[f"{col}_adj"] = df[col] + delta_gross

        return df

    def _maybe_annualise(self, df: pd.DataFrame, annual: bool) -> pd.DataFrame:
        if not annual:
            return df
        if not isinstance(df.index, pd.DatetimeIndex):
            raise TypeError("Annual aggregation requires a DatetimeIndex.")
        aggregated = df.resample("A").sum(numeric_only=True)
        # Label with the period end for clearer display.
        aggregated.index = aggregated.index.to_period("A").to_timestamp("A")
        return aggregated

    def kpis(self, df: pd.DataFrame, annual: bool = False) -> pd.DataFrame:
        """Compute headline KPIs for the supplied (scenario) dataframe."""

        data = self._maybe_annualise(df, annual)

        required_cols = {"Revenue_adj", "COGS_adj", "Gross Margin_adj"}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(
                "KPI calculation requires scenario adjusted columns: "
                + ", ".join(sorted(missing))
            )

        revenue = data["Revenue_adj"]
        gross_margin = data["Gross Margin_adj"]

        kpis = pd.DataFrame(index=data.index)
        kpis["Revenue YoY"] = revenue.pct_change()
        kpis["Gross Margin"] = (gross_margin / revenue).replace([np.inf, -np.inf], np.nan)

        if "EBITDA_adj" in data:
            kpis["EBITDA Margin"] = (
                data["EBITDA_adj"] / revenue
            ).replace([np.inf, -np.inf], np.nan)
        if "NPAT_adj" in data:
            kpis["NPAT Margin"] = (
                data["NPAT_adj"] / revenue
            ).replace([np.inf, -np.inf], np.nan)
        if "CFO" in data:
            kpis["CFO Conversion"] = (
                data["CFO"] / revenue
            ).replace([np.inf, -np.inf], np.nan)

        return kpis

    def break_even(self, df: pd.DataFrame, annual: bool = False) -> pd.DataFrame:
        """Estimate break-even revenue based on contribution margin analysis."""

        data = self._maybe_annualise(df, annual)

        required_cols = {"Revenue_adj", "COGS_adj", "Gross Margin_adj"}
        missing = required_cols - set(data.columns)
        if missing:
            raise ValueError(
                "Break-even analysis requires scenario adjusted columns: "
                + ", ".join(sorted(missing))
            )

        revenue = data["Revenue_adj"]
        contribution = data["Gross Margin_adj"]
        contribution_ratio = (contribution / revenue).replace([np.inf, -np.inf], np.nan)

        if "EBITDA_adj" not in data:
            raise ValueError("Break-even analysis requires EBITDA_adj to approximate fixed costs.")

        operating_costs = contribution - data["EBITDA_adj"]
        breakeven_revenue = operating_costs / contribution_ratio
        breakeven_revenue = breakeven_revenue.mask(contribution_ratio <= 0)

        result = pd.DataFrame(index=data.index)
        result["Contribution Margin"] = contribution_ratio
        result["Operating Costs (approx)"] = operating_costs
        result["Break-even Revenue"] = breakeven_revenue
        result["Margin of Safety"] = revenue - breakeven_revenue

        return result
