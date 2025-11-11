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

    def interest_expense(self) -> Optional[pd.Series]:
        explicit = self._get_series(
            (
                "Interest Expense",
                "Finance Costs",
                "Interest",
            )
        )
        if explicit is not None:
            return explicit

        ebit = self.ebit()
        npbt = self.npbt()
        if ebit is None or npbt is None:
            return None
        aligned = pd.concat([ebit, npbt], axis=1)
        interest = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        interest.name = "Interest Expense"
        if interest.isna().all():
            return None
        return interest

    def tax_expense(self) -> Optional[pd.Series]:
        explicit = self._get_series(
            (
                "Tax Expense",
                "Income Tax Expense",
                "Tax",
            )
        )
        if explicit is not None:
            return explicit

        npbt = self.npbt()
        npat = self.npat()
        if npbt is None or npat is None:
            return None
        aligned = pd.concat([npbt, npat], axis=1)
        tax = aligned.iloc[:, 0] - aligned.iloc[:, 1]
        tax.name = "Tax Expense"
        if tax.isna().all():
            return None
        return tax

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

    def current_assets(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Current Assets",
                "Working Capital Assets",
                "Short-term Assets",
            )
        )

    def non_current_assets(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Non-current Assets",
                "Long-term Assets",
                "Fixed Assets",
            )
        )

    def current_liabilities(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Current Liabilities",
                "Short-term Liabilities",
                "Working Capital Liabilities",
            )
        )

    def non_current_liabilities(self) -> Optional[pd.Series]:
        return self._get_series(
            (
                "Non-current Liabilities",
                "Long-term Liabilities",
                "Term Debt",
            )
        )

    def equity(self) -> Optional[pd.Series]:
        return self._get_series(("Equity", "Shareholders' Equity", "Owner Equity"))

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
            "Interest Expense": self.interest_expense(),
            "Tax Expense": self.tax_expense(),
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

    # ---------- Financial statements ----------
    def _aggregate(self, df: pd.DataFrame, annual: bool) -> pd.DataFrame:
        if df.empty:
            return df
        if annual:
            return df.groupby(df.index.year).sum(min_count=1)
        return df

    def statement_of_financial_performance(
        self,
        df: Optional[pd.DataFrame] = None,
        annual: bool = True,
    ) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        rev_col = "Revenue_adj" if "Revenue_adj" in df else "Revenue"
        cogs_col = "COGS_adj" if "COGS_adj" in df else "COGS"
        gm_col = "Gross Margin_adj" if "Gross Margin_adj" in df else "Gross Margin"
        ebitda_col = "EBITDA_adj" if "EBITDA_adj" in df else "EBITDA"
        ebit_col = "EBIT_adj" if "EBIT_adj" in df else "EBIT"
        npbt_col = "NPBT"
        npat_col = "NPAT_adj" if "NPAT_adj" in df else "NPAT"
        work = pd.DataFrame(index=df.index)

        def _maybe_add(column: str, series: Optional[pd.Series]) -> None:
            if series is None:
                return
            work[column] = pd.to_numeric(series, errors="coerce")

        _maybe_add("Revenue", df.get(rev_col))
        _maybe_add("COGS", df.get(cogs_col))

        if "Revenue" in work and "COGS" in work:
            work["Gross Profit"] = work["Revenue"] - work["COGS"]
        else:
            _maybe_add("Gross Profit", df.get(gm_col))

        _maybe_add("Variable Expenses", df.get("Variable Expenses"))
        _maybe_add("Direct Wages", df.get("Direct Wages"))
        _maybe_add("EBITDA", df.get(ebitda_col))
        _maybe_add("Fixed Expenses", df.get("Fixed Expenses"))
        _maybe_add("Admin Wages", df.get("Admin Wages"))
        _maybe_add("Depreciation", df.get("Depreciation & Amortization"))
        _maybe_add("EBIT", df.get(ebit_col))

        npbt_series = df.get(npbt_col)
        npat_series = df.get(npat_col)
        _maybe_add("Net Profit", npat_series)

        interest_series = df.get("Interest Expense") or df.get("Interest") or df.get("Finance Costs")
        if interest_series is None and "EBIT" in work and npbt_series is not None:
            interest_series = pd.to_numeric(work["EBIT"], errors="coerce") - pd.to_numeric(
                npbt_series, errors="coerce"
            )
        _maybe_add("Interest", interest_series)

        tax_series = df.get("Tax Expense") or df.get("Income Tax Expense") or df.get("Tax")
        if tax_series is None and npbt_series is not None and npat_series is not None:
            tax_series = pd.to_numeric(npbt_series, errors="coerce") - pd.to_numeric(
                npat_series, errors="coerce"
            )
        _maybe_add("Tax", tax_series)

        if work.empty:
            raise ValueError("No income-statement data available in the schedule.")

        agg = self._aggregate(work, annual=annual)
        if agg.empty:
            raise ValueError("No income-statement data available in the schedule.")

        ordered = [
            "Revenue",
            "COGS",
            "Gross Profit",
            "Gross Profit Margin",
            "Variable Expenses",
            "Direct Wages",
            "EBITDA",
            "Fixed Expenses",
            "Admin Wages",
            "Depreciation",
            "EBIT",
            "Interest",
            "Tax",
            "Net Profit",
        ]

        out = pd.DataFrame(index=agg.index)
        for column in ordered:
            if column == "Gross Profit Margin":
                continue
            if column in agg:
                out[column] = agg[column]
            else:
                out[column] = np.nan

        if "Gross Profit" in agg and "Revenue" in agg:
            with np.errstate(divide="ignore", invalid="ignore"):
                margin = agg["Gross Profit"] / agg["Revenue"]
            margin = margin.replace({np.inf: np.nan, -np.inf: np.nan})
        else:
            margin = pd.Series(np.nan, index=agg.index)
        out["Gross Profit Margin"] = margin

        return out[ordered]

    def statement_of_cash_flow(
        self, df: Optional[pd.DataFrame] = None, annual: bool = True
    ) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        def _aggregate_sum(series: Optional[pd.Series]) -> Optional[pd.Series]:
            if series is None:
                return None
            cleaned = pd.to_numeric(series, errors="coerce")
            if annual:
                return cleaned.groupby(cleaned.index.year).sum(min_count=1)
            return cleaned

        def _aggregate_first(series: Optional[pd.Series]) -> Optional[pd.Series]:
            if series is None:
                return None
            cleaned = pd.to_numeric(series, errors="coerce")
            if annual:
                return cleaned.groupby(cleaned.index.year).first()
            return cleaned

        def _aggregate_last(series: Optional[pd.Series]) -> Optional[pd.Series]:
            if series is None:
                return None
            cleaned = pd.to_numeric(series, errors="coerce")
            if annual:
                return cleaned.groupby(cleaned.index.year).last()
            return cleaned

        flows = {
            "Net cash from operating activities": _aggregate_sum(df.get("CFO")),
            "Net cash used in investing activities": _aggregate_sum(df.get("CFI")),
            "Capital expenditure (included in investing activities)": _aggregate_sum(df.get("Capex")),
            "Net cash from financing activities": _aggregate_sum(df.get("CFF")),
        }
        flows = {name: series for name, series in flows.items() if series is not None}
        if not flows:
            raise ValueError("No cash-flow data available in the schedule.")

        out = pd.concat(flows, axis=1)

        net_cash_series = _aggregate_sum(df.get("Net Cash Flow"))
        if net_cash_series is None and {"Net cash from operating activities", "Net cash used in investing activities", "Net cash from financing activities"}.issubset(out.columns):
            net_cash_series = (
                out["Net cash from operating activities"]
                + out.get("Net cash used in investing activities", 0)
                + out.get("Net cash from financing activities", 0)
            )
        if net_cash_series is not None:
            out["Net increase/(decrease) in cash and cash equivalents"] = net_cash_series

        opening_candidates = [
            "Opening Cash Balance",
            "Opening Cash",
            "Cash at Beginning of Period",
        ]
        closing_candidates = [
            "Closing Cash Balance",
            "Closing Cash",
            "Cash and Cash Equivalents",
            "Cash at End of Period",
        ]

        opening_series = None
        for candidate in opening_candidates:
            if candidate in df:
                opening_series = _aggregate_first(df.get(candidate))
                break

        closing_series = None
        for candidate in closing_candidates:
            if candidate in df:
                closing_series = _aggregate_last(df.get(candidate))
                break

        if opening_series is not None:
            out = out.reindex(out.index.union(opening_series.index)).sort_index()
            out["Opening cash and cash equivalents"] = opening_series

        if closing_series is not None:
            out = out.reindex(out.index.union(closing_series.index)).sort_index()
            out["Closing cash and cash equivalents"] = closing_series
        elif opening_series is not None and net_cash_series is not None:
            out["Closing cash and cash equivalents"] = opening_series.add(
                net_cash_series, fill_value=np.nan
            )

        if (
            "Closing cash and cash equivalents" not in out
            and closing_series is None
            and "Net increase/(decrease) in cash and cash equivalents" in out
        ):
            out["Closing cash and cash equivalents"] = out[
                "Net increase/(decrease) in cash and cash equivalents"
            ].cumsum()

        order = [
            "Net cash from operating activities",
            "Net cash used in investing activities",
            "Capital expenditure (included in investing activities)",
            "Net cash from financing activities",
            "Net increase/(decrease) in cash and cash equivalents",
            "Opening cash and cash equivalents",
            "Closing cash and cash equivalents",
        ]

        available = [col for col in order if col in out.columns]
        if not available:
            raise ValueError("No cash-flow data available in the schedule.")

        return out[available]

    def statement_of_financial_position(
        self, df: Optional[pd.DataFrame] = None, annual: bool = True
    ) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        def _aggregate_balance(series: Optional[pd.Series]) -> Optional[pd.Series]:
            if series is None:
                return None
            cleaned = pd.to_numeric(series, errors="coerce")
            if annual:
                return cleaned.groupby(cleaned.index.year).last()
            return cleaned

        components = {
            "Cash and Cash Equivalents": df.get("Cash and Cash Equivalents")
            or df.get("Closing Cash Balance"),
            "Current Assets": df.get("Current Assets"),
            "Non-current Assets": df.get("Non-current Assets"),
            "Current Liabilities": df.get("Current Liabilities"),
            "Non-current Liabilities": df.get("Non-current Liabilities"),
            "Equity": df.get("Equity"),
        }

        aggregated = {
            name: _aggregate_balance(series) for name, series in components.items() if series is not None
        }

        if not aggregated:
            raise ValueError("No balance sheet data available in the schedule.")

        out = pd.concat(aggregated, axis=1)

        if {"Current Assets", "Non-current Assets"}.issubset(out.columns):
            out["Total Assets"] = out["Current Assets"] + out["Non-current Assets"]

        if {"Current Liabilities", "Non-current Liabilities"}.issubset(out.columns):
            out["Total Liabilities"] = (
                out["Current Liabilities"] + out["Non-current Liabilities"]
            )

        if "Equity" in out:
            out["Total Equity"] = out["Equity"]

        if {"Total Assets", "Total Liabilities"}.issubset(out.columns):
            out["Net Assets"] = out["Total Assets"] - out["Total Liabilities"]

        if {"Current Assets", "Current Liabilities"}.issubset(out.columns):
            out["Net Current Assets"] = out["Current Assets"] - out["Current Liabilities"]

        if {"Total Liabilities", "Total Equity"}.issubset(out.columns):
            out["Total Liabilities & Equity"] = (
                out["Total Liabilities"] + out["Total Equity"]
            )

        order = [
            "Cash and Cash Equivalents",
            "Current Assets",
            "Non-current Assets",
            "Total Assets",
            "Current Liabilities",
            "Non-current Liabilities",
            "Total Liabilities",
            "Equity",
            "Total Equity",
            "Net Assets",
            "Net Current Assets",
            "Total Liabilities & Equity",
        ]

        available = [col for col in order if col in out.columns]
        return out[available]

    def advanced_analytics(
        self,
        df: Optional[pd.DataFrame] = None,
        window: int = 3,
        annual: bool = False,
    ) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        rev_col = "Revenue_adj" if "Revenue_adj" in df else "Revenue"
        gm_col = "Gross Margin_adj" if "Gross Margin_adj" in df else "Gross Margin"
        ebitda_col = "EBITDA_adj" if "EBITDA_adj" in df else "EBITDA"
        npat_col = "NPAT_adj" if "NPAT_adj" in df else "NPAT"

        required = [col for col in [rev_col, gm_col, ebitda_col, npat_col] if col in df]
        if not required:
            raise ValueError("Insufficient data to compute advanced analytics.")

        work = df[required].rename(
            columns={
                rev_col: "Revenue",
                gm_col: "Gross Margin",
                ebitda_col: "EBITDA",
                npat_col: "NPAT",
            }
        )

        if annual:
            work = work.groupby(work.index.year).sum(min_count=1)

        out = pd.DataFrame(index=work.index)
        out["Revenue Growth %"] = work["Revenue"].pct_change()
        out["Rolling Revenue (window)"] = work["Revenue"].rolling(window, min_periods=1).mean()
        out["Gross Margin %"] = work["Gross Margin"] / work["Revenue"]
        out["EBITDA Margin %"] = work["EBITDA"] / work["Revenue"]
        out["Net Margin %"] = work["NPAT"] / work["Revenue"]
        if "Variable Expenses" in df:
            var = df["Variable Expenses"]
            var = var.groupby(var.index.year).sum(min_count=1) if annual else var
            out["Variable Cost %"] = var / work["Revenue"]
        if "Fixed Expenses" in df:
            fixed = df["Fixed Expenses"]
            fixed = fixed.groupby(fixed.index.year).sum(min_count=1) if annual else fixed
            out["Fixed Cost %"] = fixed / work["Revenue"]
        if "EBITDA" in work:
            out["EBITDA Conversion"] = work["EBITDA"] / work["Gross Margin"]
        return out

    def to_tidy(self) -> pd.DataFrame:
        return self.data.copy()


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").dropna(axis=1, how="all")
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned.reset_index(drop=True)
