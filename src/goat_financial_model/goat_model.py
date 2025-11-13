"""Utilities for manipulating the goat farming financial model without Excel."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

SeriesLabels = Sequence[str]


class DataQualityWarning(UserWarning):
    """Warning emitted when numeric coercion drops one or more values."""


def _coerce_numeric_frame(df: pd.DataFrame, *, context: str) -> pd.DataFrame:
    """Convert frame to numeric, raising if columns lose all data."""

    raw = df.copy()
    numeric = df.apply(pd.to_numeric, errors="coerce")

    coerced_mask = raw.notna() & numeric.isna()
    if coerced_mask.any().any():
        counts = coerced_mask.sum()
        details = ", ".join(
            f"{column}: {int(count)}"
            for column, count in counts[counts > 0].items()
        )
        total = int(coerced_mask.to_numpy().sum())
        warnings.warn(
            f"{context}: coerced {total} value(s) to NaN ({details}).",
            DataQualityWarning,
            stacklevel=2,
        )

    problematic = [
        column
        for column in numeric.columns
        if numeric[column].notna().sum() == 0 and raw[column].notna().sum() > 0
    ]
    if problematic:
        columns = ", ".join(problematic)
        raise ValueError(
            f"{context} columns contain no numeric values after coercion: {columns}."
        )

    return numeric


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
        self.data = _coerce_numeric_frame(self.data, context="Input schedule")

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

        values = frame.drop(columns=[period_col])
        values = _coerce_numeric_frame(values, context="Input schedule")
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
        self.data = _coerce_numeric_frame(self.data, context="Model data")

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.data.index

    # ---------- Internal helpers ----------
    @staticmethod
    def _safe_divide(
        numerator: pd.Series,
        denominator: pd.Series,
        *,
        min_abs: float = 1e-9,
        allow_negative: bool = False,
    ) -> pd.Series:
        """Safely divide two aligned series, masking unstable denominators."""

        num_aligned, denom_aligned = numerator.align(denominator, join="outer")
        result = pd.Series(np.nan, index=num_aligned.index, dtype=float)

        valid_mask = denom_aligned.notna() & (denom_aligned.abs() >= min_abs)
        if not allow_negative:
            valid_mask &= denom_aligned > 0

        if valid_mask.any():
            result.loc[valid_mask] = (
                num_aligned.loc[valid_mask] / denom_aligned.loc[valid_mask]
            )

        return result

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

    def ufcf(self, column: Optional[str] = None) -> Optional[pd.Series]:
        table = None
        for key in ("UFCF", "Unlevered Free Cash Flow"):
            if key in self.supplementary_tables:
                table = self.supplementary_tables[key]
                break
        if table is None or table.empty:
            return None

        df = table.copy()
        df.columns = [str(col).strip() for col in df.columns]
        preferred = column or str(self.valuation_inputs.get("UFCF Column", "")).strip()
        selected: Optional[str] = None
        if preferred:
            for candidate in df.columns:
                if candidate.lower() == preferred.lower():
                    selected = candidate
                    break
        if selected is None:
            candidates = [
                col
                for col in df.columns
                if "ufcf" in col.lower() or "free cash" in col.lower()
            ]
            if not candidates:
                candidates = [df.columns[-1]]
            elif len(candidates) > 1:
                raise ValueError(
                    "Multiple UFCF columns detected; specify the desired column explicitly."
                )
            selected = candidates[0]

        if "Period" in df.columns:
            idx = pd.to_datetime(df["Period"], errors="coerce")
        else:
            idx = pd.to_datetime(df.index, errors="coerce")

        values = pd.to_numeric(df[selected], errors="coerce")
        mask = idx.notna() & values.notna()
        if not mask.any():
            return None

        ordered = (
            pd.DataFrame({"Period": idx[mask], "Value": values[mask]})
            .sort_values("Period")
            .reset_index(drop=True)
        )
        if ordered["Period"].duplicated().any():
            raise ValueError("UFCF schedule contains duplicate periods.")
        if (ordered["Period"].diff().dt.total_seconds() <= 0).any():
            raise ValueError("UFCF schedule periods must be strictly increasing.")

        return pd.Series(
            ordered["Value"].to_numpy(),
            index=pd.DatetimeIndex(ordered["Period"].to_numpy()),
            name="Unlevered Free Cash Flow",
        )

    def discounted_cash_flow(self) -> Dict[str, object]:
        """Compute the discounted cash-flow valuation using stored assumptions."""

        cash_flows = self.ufcf()
        if cash_flows is None or cash_flows.empty:
            raise ValueError("UFCF schedule is required for discounted cash-flow analysis.")

        rate = self.wacc()
        if rate is None:
            raise ValueError("WACC is required for discounted cash-flow analysis.")

        if rate > 1:
            rate = rate / 100.0
        if rate <= 0:
            raise ValueError("WACC must be positive to compute discounted cash flow.")

        cash_flows = cash_flows.sort_index()
        if cash_flows.index.duplicated().any():
            raise ValueError("Cash-flow timeline contains duplicate periods.")
        if (cash_flows.index.to_series().diff().dt.total_seconds() <= 0).any():
            raise ValueError("Cash-flow timeline must be strictly increasing.")

        diffs_days = cash_flows.index.to_series().diff().dt.days.astype(float)
        valid_diffs = diffs_days.iloc[1:][np.isfinite(diffs_days.iloc[1:])]
        if not valid_diffs.empty:
            median_days = float(np.median(valid_diffs))
        else:
            median_days = 365.25
        if not np.isfinite(median_days) or median_days <= 0:
            median_days = 365.25

        irregular = valid_diffs[
            (valid_diffs - median_days).abs() > max(median_days * 0.25, 1.0)
        ]
        if not irregular.empty:
            warnings.warn(
                "Cash-flow timeline contains irregular step sizes; results may be approximate.",
                DataQualityWarning,
                stacklevel=2,
            )

        diffs_years = diffs_days / 365.25
        diffs_years.iloc[0] = median_days / 365.25
        diffs_years = diffs_years.fillna(median_days / 365.25).clip(lower=1e-9)

        periods = diffs_years.cumsum().to_numpy()
        discount_factors = 1 / np.power(1 + rate, periods)
        pv_cash_flows = cash_flows.to_numpy() * discount_factors

        cash_flow_df = pd.DataFrame(
            {
                "UFCF": cash_flows.to_numpy(),
                "Discount Factor": discount_factors,
                "Present Value": pv_cash_flows,
            },
            index=cash_flows.index,
        )

        terminal_value = self.terminal_value()
        terminal_value_pv = None
        if terminal_value is not None:
            last_step = diffs_years.iloc[-1]
            horizon = periods[-1] + last_step
            terminal_value_pv = float(terminal_value) / ((1 + rate) ** horizon)

        total_pv = float(np.nansum(pv_cash_flows))
        enterprise_value = total_pv + (terminal_value_pv or 0.0)

        summary: Dict[str, object] = {
            "cash_flows": cash_flow_df,
            "discount_rate": rate,
            "enterprise_value": enterprise_value,
        }

        if terminal_value_pv is not None:
            summary["terminal_value_pv"] = terminal_value_pv

        npv_value = self.npv()
        if npv_value is not None:
            summary["npv"] = float(npv_value)

        return summary

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
            "CFO": self.cfo(),
            "CFI": self.cfi(),
            "CFF": self.cff(),
            "Capex": self.capex(),
            "Net Cash Flow": self.net_cash_flow(),
            "Opening Cash Balance": self._get_series(
                ("Opening Cash Balance", "Opening Cash", "Cash at Beginning of Period")
            ),
            "Closing Cash Balance": self._get_series(
                ("Closing Cash Balance", "Closing Cash", "Cash at End of Period")
            ),
            "Cash and Cash Equivalents": self._get_series(
                (
                    "Cash and Cash Equivalents",
                    "Cash & Equivalents",
                    "Cash",
                    "Closing Cash",
                )
            ),
            "Current Assets": self.current_assets(),
            "Non-current Assets": self.non_current_assets(),
            "Current Liabilities": self.current_liabilities(),
            "Non-current Liabilities": self.non_current_liabilities(),
            "Equity": self.equity(),
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

        npbt_series = self.npbt()
        tax_series = self.tax_expense()
        npat_series = self.npat()
        eff_tax: Optional[float] = None
        if npbt_series is not None and tax_series is not None:
            aligned_tax = pd.concat([npbt_series, tax_series], axis=1).dropna()
            aligned_tax = aligned_tax[aligned_tax.iloc[:, 0] > 1e-9]
            if not aligned_tax.empty:
                ratios = aligned_tax.iloc[:, 1] / aligned_tax.iloc[:, 0]
                eff_tax = float(np.clip(ratios.median(), 0.0, 0.6))
        if eff_tax is None and npbt_series is not None and npat_series is not None:
            aligned = pd.concat([npbt_series, npat_series], axis=1).dropna()
            aligned = aligned[aligned.iloc[:, 0] > 1e-9]
            if not aligned.empty:
                eff_tax = float(
                    np.clip(1 - (aligned.iloc[:, 1] / aligned.iloc[:, 0]).median(), 0.0, 0.6)
                )
        if eff_tax is None:
            eff_tax = 0.28

        interest = df.get("Interest Expense", 0)
        df["NPBT_adj"] = df["EBIT_adj"] - interest
        tax_adj = np.maximum(df["NPBT_adj"], 0.0) * eff_tax
        df["Tax Expense_adj"] = tax_adj
        df["NPAT_adj"] = df["NPBT_adj"] - tax_adj

        notes: Dict[str, str] = {}
        if "CFO" in df:
            if "NPAT" in df:
                delta_npat = df["NPAT_adj"] - df["NPAT"]
                df["CFO_adj"] = df["CFO"] + delta_npat
                notes["CFO_adj"] = "Adjusted using NPAT delta"
            else:
                df["CFO_adj"] = df["CFO"]
                notes["CFO_adj"] = "Unchanged (missing NPAT baseline)"
        if "CFI" in df:
            capex = df.get("Capex")
            if capex is not None:
                candidate = (-capex).fillna(0.0)
                baseline = df["CFI"].fillna(0.0)
                if np.allclose(candidate.to_numpy(), baseline.to_numpy(), atol=1e-6):
                    df["CFI_adj"] = candidate
                    notes["CFI_adj"] = "Recomputed from capex"
                else:
                    df["CFI_adj"] = df["CFI"]
                    notes["CFI_adj"] = "Unchanged (capex does not reconcile)"
            else:
                df["CFI_adj"] = df["CFI"]
                notes["CFI_adj"] = "Unchanged (no capex data)"
        if "CFF" in df:
            df["CFF_adj"] = df["CFF"]
            notes["CFF_adj"] = "Unchanged (no financing schedule adjustments)"

        if {"CFO_adj", "CFI_adj", "CFF_adj"}.issubset(df.columns):
            df["Net Cash Flow_adj"] = df["CFO_adj"] + df["CFI_adj"] + df["CFF_adj"]
            notes["Net Cash Flow_adj"] = "Derived from adjusted cash flows"
        elif "Net Cash Flow" in df:
            df["Net Cash Flow_adj"] = df["Net Cash Flow"]
            notes["Net Cash Flow_adj"] = "Unchanged (incomplete cash flow drivers)"

        if "Opening Cash Balance" in df and "Net Cash Flow_adj" in df:
            opening = df["Opening Cash Balance"].ffill()
            df["Closing Cash Balance_adj"] = opening + df["Net Cash Flow_adj"].fillna(0.0)
            notes["Closing Cash Balance_adj"] = "Opening balance plus adjusted net cash flow"

        if notes:
            scenario_notes = dict(df.attrs.get("scenario_notes", {}))
            scenario_notes.update(notes)
            df.attrs["scenario_notes"] = scenario_notes

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
        out["Gross Margin %"] = self._safe_divide(grp["Gross Margin"], grp["Revenue"])
        out["EBITDA Margin %"] = self._safe_divide(grp["EBITDA"], grp["Revenue"])
        out["Net Margin %"] = self._safe_divide(grp["NPAT"], grp["Revenue"])
        out["COGS % of Revenue"] = self._safe_divide(grp["COGS"], grp["Revenue"])
        out["Revenue YoY %"] = grp["Revenue"].pct_change()
        return out

    # ---------- Break-even ----------
    def break_even(self, df: Optional[pd.DataFrame] = None, annual: bool = True) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        rev = df["Revenue_adj"] if "Revenue_adj" in df else df["Revenue"]
        gm = df["Gross Margin_adj"] if "Gross Margin_adj" in df else df["Gross Margin"]
        ebitda = df["EBITDA_adj"] if "EBITDA_adj" in df else df["EBITDA"]

        if annual:
            idx = rev.index.year
            rev = rev.groupby(idx).sum(min_count=1)
            gm = gm.groupby(idx).sum(min_count=1)
            ebitda = ebitda.groupby(idx).sum(min_count=1)

        cm_ratio = self._safe_divide(gm, rev)
        fixed_costs = gm - ebitda
        be_rev = self._safe_divide(fixed_costs, cm_ratio)
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
        """Return an IFRS-style statement of profit or loss."""

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
        presence: Dict[str, bool] = {}

        def _assign_first(target: str, *candidates: str) -> bool:
            for name in candidates:
                series = df.get(name)
                if series is None:
                    continue
                numeric = pd.to_numeric(series, errors="coerce")
                if target not in work:
                    work[target] = numeric
                    presence[target] = True
                    return True
            presence.setdefault(target, False)
            return False

        def _accumulate(target: str, *candidates: str) -> bool:
            found = False
            for name in candidates:
                series = df.get(name)
                if series is None:
                    continue
                numeric = pd.to_numeric(series, errors="coerce")
                if target in work:
                    work[target] = work[target].add(numeric, fill_value=0.0)
                else:
                    work[target] = numeric
                found = True
            if found:
                presence[target] = True
            else:
                presence.setdefault(target, False)
            return found

        _assign_first("Revenue", rev_col, "Revenue")
        _assign_first("Cost of sales", cogs_col, "Cost of Sales", "COGS")
        _assign_first("Gross profit", gm_col, "Gross Profit")
        _accumulate(
            "Other income",
            "Other Income",
            "Other Revenue",
            "Non-operating Income",
            "Investment Income",
        )
        _accumulate(
            "Distribution costs",
            "Variable Expenses",
            "Distribution Costs",
            "Selling Expenses",
            "Sales and Marketing",
            "Direct Wages",
        )
        _accumulate(
            "Administrative expenses",
            "Fixed Expenses",
            "Admin Wages",
            "Administrative Expenses",
            "General & Administrative Expenses",
            "Overheads",
        )
        _accumulate(
            "Depreciation and amortisation",
            "Depreciation & Amortization",
            "Depreciation",
            "Amortization",
        )
        _accumulate(
            "Other operating expenses",
            "Other Operating Expenses",
            "Operating Expenses",
            "Research and Development",
        )
        _assign_first("EBITDA", ebitda_col, "EBITDA")
        _assign_first("Operating profit (EBIT)", ebit_col, "EBIT", "Operating Profit", "Operating Income")
        _accumulate(
            "Finance income",
            "Finance Income",
            "Interest Income",
            "Investment Income",
        )
        _accumulate(
            "Finance costs",
            "Interest Expense",
            "Finance Costs",
            "Interest",
        )
        _assign_first("Profit before tax", npbt_col, "Profit Before Tax", "Earnings Before Tax")
        _accumulate(
            "Income tax expense",
            "Tax Expense",
            "Income Tax Expense",
            "Tax",
        )
        _assign_first(
            "Profit for the period",
            npat_col,
            "Net Profit",
            "Profit for the Period",
            "Profit After Tax",
            "Net Income",
        )

        if work.empty:
            raise ValueError("No income-statement data available in the schedule.")

        agg = self._aggregate(work, annual=annual)
        if agg.empty:
            raise ValueError("No income-statement data available in the schedule.")

        index = agg.index

        def _series(name: str, *, default: float = np.nan) -> pd.Series:
            if name in agg:
                return pd.to_numeric(agg[name], errors="coerce")
            if np.isnan(default):
                return pd.Series(np.nan, index=index, dtype=float)
            return pd.Series(default, index=index, dtype=float)

        def _series_with_presence(name: str, *, default: float = np.nan) -> pd.Series:
            series = _series(name, default=default)
            if not presence.get(name, False):
                return pd.Series(np.nan, index=index, dtype=float)
            return series

        revenue = _series_with_presence("Revenue")
        cost_of_sales = _series_with_presence("Cost of sales")
        gross_profit_reported = _series_with_presence("Gross profit")
        other_income = _series_with_presence("Other income", default=0.0)
        distribution_costs = _series_with_presence("Distribution costs", default=0.0)
        administrative_expenses = _series_with_presence("Administrative expenses", default=0.0)
        depreciation = _series_with_presence("Depreciation and amortisation", default=0.0)
        other_operating = _series_with_presence("Other operating expenses", default=0.0)
        finance_costs = _series_with_presence("Finance costs", default=0.0)
        profit_before_tax = _series_with_presence("Profit before tax")
        income_tax = _series_with_presence("Income tax expense", default=0.0)
        profit_for_period = _series_with_presence("Profit for the period")
        ebitda_series = _series_with_presence("EBITDA")
        operating_profit_reported = _series_with_presence("Operating profit (EBIT)")

        computed_gross = revenue.subtract(cost_of_sales, fill_value=0.0)
        if presence.get("Revenue", False) and presence.get("Cost of sales", False):
            gross_profit = computed_gross
        elif gross_profit_reported.notna().any():
            gross_profit = gross_profit_reported
        else:
            gross_profit = computed_gross

        expenses_components = pd.concat(
            [
                distribution_costs,
                administrative_expenses,
                depreciation,
                other_operating,
            ],
            axis=1,
        )
        expenses_components.columns = [
            "Distribution",
            "Administrative",
            "Depreciation",
            "Other",
        ]

        operating_expenses_total = expenses_components.sum(axis=1, min_count=1)
        operating_expenses_total = operating_expenses_total.where(
            expenses_components.notna().any(axis=1), np.nan
        )

        computed_operating = (
            gross_profit.fillna(0.0)
            + other_income.fillna(0.0)
            - operating_expenses_total.fillna(0.0)
        )
        has_operating_inputs = (
            gross_profit.notna()
            | other_income.notna()
            | expenses_components.notna().any(axis=1)
        )
        operating_profit = computed_operating.mask(~has_operating_inputs, np.nan)
        operating_profit = operating_profit.where(
            operating_profit.notna(), operating_profit_reported
        )

        computed_ebitda = operating_profit.add(
            depreciation.fillna(0.0), fill_value=0.0
        )
        if operating_profit.notna().any() or depreciation.notna().any():
            ebitda_series = ebitda_series.where(ebitda_series.notna(), computed_ebitda)

        computed_profit_before_tax = operating_profit.subtract(
            finance_costs.fillna(0.0), fill_value=0.0
        )
        if operating_profit.notna().any() or finance_costs.notna().any():
            profit_before_tax = profit_before_tax.where(
                profit_before_tax.notna(), computed_profit_before_tax
            )

        computed_profit_for_period = profit_before_tax.subtract(
            income_tax.fillna(0.0), fill_value=0.0
        )
        if profit_before_tax.notna().any() or income_tax.notna().any():
            profit_for_period = profit_for_period.where(
                profit_for_period.notna(), computed_profit_for_period
            )

        total_income = revenue.add(other_income, fill_value=0.0)
        if not (revenue.notna() | other_income.notna()).any():
            total_income[:] = np.nan

        finance_income_series = _series_with_presence("Finance income", default=0.0)
        finance_income_for_calc = finance_income_series.fillna(0.0)
        net_finance_result = finance_income_for_calc.subtract(
            finance_costs.fillna(0.0), fill_value=0.0
        )
        has_finance_activity = finance_income_series.notna() | finance_costs.notna()
        net_finance_result = net_finance_result.where(has_finance_activity, np.nan)

        ordered_sections = [
            (
                "Income",
                [
                    ("Revenue", revenue),
                    ("Other income", other_income),
                    ("Total income", total_income),
                ],
            ),
            (
                "Cost of sales",
                [
                    ("Cost of sales", cost_of_sales),
                    ("Gross profit", gross_profit),
                ],
            ),
            (
                "Operating expenses",
                [
                    ("Distribution costs", distribution_costs),
                    ("Administrative expenses", administrative_expenses),
                    ("Depreciation and amortisation", depreciation),
                    ("Other operating expenses", other_operating),
                    ("Total operating expenses", operating_expenses_total),
                ],
            ),
            (
                "Operating profit",
                [
                    ("EBIT", operating_profit),
                    ("EBITDA", ebitda_series),
                ],
            ),
            (
                "Finance",
                [
                    ("Finance income", finance_income_series),
                    ("Finance costs", finance_costs),
                    ("Net finance result", net_finance_result),
                ],
            ),
            (
                "Profit",
                [
                    ("Profit before tax", profit_before_tax),
                    ("Income tax expense", income_tax),
                    ("Profit for the period", profit_for_period),
                ],
            ),
        ]

        out = pd.DataFrame(index=index)
        column_order: List[str] = []

        def _add_column(section: str, label: str, series: pd.Series) -> None:
            if series is None:
                return
            if not isinstance(series, pd.Series):
                return
            if not series.notna().any():
                return
            column_name = f"{section} – {label}"
            out[column_name] = series
            column_order.append(column_name)

        for section, items in ordered_sections:
            for label, series in items:
                _add_column(section, label, series)

        if out.empty:
            raise ValueError("No income-statement data available in the schedule.")

        return out[column_order]

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
            "operating": _aggregate_sum(df.get("CFO")),
            "investing": _aggregate_sum(df.get("CFI")),
            "capex": _aggregate_sum(df.get("Capex")),
            "financing": _aggregate_sum(df.get("CFF")),
        }

        if not any(series is not None for series in flows.values()):
            raise ValueError("No cash-flow data available in the schedule.")

        net_cash_series = _aggregate_sum(df.get("Net Cash Flow"))
        if net_cash_series is None:
            available = [
                series
                for key, series in flows.items()
                if key in {"operating", "investing", "financing"} and series is not None
            ]
            if available:
                net_cash_series = sum(available)

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

        if closing_series is None and opening_series is not None and net_cash_series is not None:
            closing_series = opening_series.add(net_cash_series, fill_value=np.nan)

        if closing_series is None and net_cash_series is not None:
            closing_series = net_cash_series.cumsum()

        sections: List[Tuple[str, str, Optional[pd.Series]]] = []

        def _append(section: str, label: str, series: Optional[pd.Series]) -> None:
            if series is None:
                return
            if not isinstance(series, pd.Series):
                return
            if not series.notna().any():
                return
            sections.append((section, label, series))

        _append(
            "Operating activities",
            "Net cash from operating activities",
            flows["operating"],
        )
        if flows["capex"] is not None:
            _append("Investing activities", "Capital expenditure", flows["capex"])
        _append(
            "Investing activities",
            "Net cash used in investing activities",
            flows["investing"],
        )
        _append(
            "Financing activities",
            "Net cash from financing activities",
            flows["financing"],
        )
        _append(
            "Net change",
            "Net increase/(decrease) in cash and cash equivalents",
            net_cash_series,
        )
        _append(
            "Net change",
            "Cash and cash equivalents at beginning of period",
            opening_series,
        )
        _append(
            "Net change",
            "Cash and cash equivalents at end of period",
            closing_series,
        )

        if not sections:
            raise ValueError("No cash-flow data available in the schedule.")

        out = pd.DataFrame(index=pd.Index([], dtype=int))
        column_order: List[str] = []
        for section, label, series in sections:
            if out.empty:
                out = pd.DataFrame(index=series.index)
            column_name = f"{section} – {label}"
            out[column_name] = series
            column_order.append(column_name)

        return out[column_order]

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

        cash_candidates = (
            "Cash and Cash Equivalents",
            "Closing Cash Balance",
            "Closing Cash",
            "Cash at End of Period",
        )
        cash_series = None
        for candidate in cash_candidates:
            series = df.get(candidate)
            if series is not None:
                cash_series = series
                break

        components = {
            "Cash and Cash Equivalents": cash_series,
            "Current Assets": df.get("Current Assets"),
            "Non-current Assets": df.get("Non-current Assets"),
            "Current Liabilities": df.get("Current Liabilities"),
            "Non-current Liabilities": df.get("Non-current Liabilities"),
            "Equity": df.get("Equity"),
        }

        aggregated = {
            name: _aggregate_balance(series)
            for name, series in components.items()
            if series is not None
        }

        if not aggregated:
            raise ValueError("No balance sheet data available in the schedule.")

        out = pd.concat(aggregated, axis=1)

        total_assets = None
        if {"Current Assets", "Non-current Assets"}.issubset(out.columns):
            total_assets = out["Current Assets"].add(
                out["Non-current Assets"], fill_value=0.0
            )
            has_assets = (
                out["Current Assets"].notna() | out["Non-current Assets"].notna()
            )
            total_assets = total_assets.where(has_assets, np.nan)

        total_liabilities = None
        if {"Current Liabilities", "Non-current Liabilities"}.issubset(out.columns):
            total_liabilities = out["Current Liabilities"].add(
                out["Non-current Liabilities"], fill_value=0.0
            )
            has_liabilities = (
                out["Current Liabilities"].notna()
                | out["Non-current Liabilities"].notna()
            )
            total_liabilities = total_liabilities.where(has_liabilities, np.nan)

        total_equity = out.get("Equity")

        net_assets = None
        if total_assets is not None and total_liabilities is not None:
            net_assets = total_assets.subtract(total_liabilities, fill_value=0.0)
            has_net_assets = total_assets.notna() | total_liabilities.notna()
            net_assets = net_assets.where(has_net_assets, np.nan)

        net_current_assets = None
        if {"Current Assets", "Current Liabilities"}.issubset(out.columns):
            net_current_assets = out["Current Assets"].subtract(
                out["Current Liabilities"], fill_value=0.0
            )
            has_working_capital = (
                out["Current Assets"].notna() | out["Current Liabilities"].notna()
            )
            net_current_assets = net_current_assets.where(has_working_capital, np.nan)

        total_liabilities_and_equity = None
        if total_liabilities is not None and total_equity is not None:
            total_liabilities_and_equity = total_liabilities.add(
                total_equity, fill_value=0.0
            )
            has_balancing = total_liabilities.notna() | total_equity.notna()
            total_liabilities_and_equity = total_liabilities_and_equity.where(
                has_balancing, np.nan
            )

        sections: List[Tuple[str, str, Optional[pd.Series]]] = []

        def _append(section: str, label: str, series: Optional[pd.Series]) -> None:
            if series is None:
                return
            if not isinstance(series, pd.Series):
                return
            if not series.notna().any():
                return
            sections.append((section, label, series))

        _append("Assets", "Cash and cash equivalents", out.get("Cash and Cash Equivalents"))
        _append("Assets", "Current assets", out.get("Current Assets"))
        _append("Assets", "Non-current assets", out.get("Non-current Assets"))
        _append("Assets", "Total assets", total_assets)
        _append("Equity and liabilities", "Equity", total_equity)
        _append(
            "Equity and liabilities",
            "Non-current liabilities",
            out.get("Non-current Liabilities"),
        )
        _append(
            "Equity and liabilities",
            "Current liabilities",
            out.get("Current Liabilities"),
        )
        _append("Equity and liabilities", "Total liabilities", total_liabilities)
        _append(
            "Equity and liabilities",
            "Total equity and liabilities",
            total_liabilities_and_equity,
        )
        _append("Key metrics", "Net assets", net_assets)
        _append("Key metrics", "Net current assets", net_current_assets)

        if not sections:
            raise ValueError("No balance sheet data available in the schedule.")

        result = pd.DataFrame(index=out.index)
        column_order: List[str] = []
        for section, label, series in sections:
            column_name = f"{section} – {label}"
            result[column_name] = series
            column_order.append(column_name)

        return result[column_order]

    def advanced_analytics(
        self,
        df: Optional[pd.DataFrame] = None,
        window: int = 3,
        annual: bool = False,
    ) -> Dict[str, object]:
        if df is None:
            df = self.to_tidy()

        rev_col = "Revenue_adj" if "Revenue_adj" in df else "Revenue"
        gm_col = "Gross Margin_adj" if "Gross Margin_adj" in df else "Gross Margin"
        ebitda_col = "EBITDA_adj" if "EBITDA_adj" in df else "EBITDA"
        npat_col = "NPAT_adj" if "NPAT_adj" in df else "NPAT"

        required = [col for col in [rev_col, gm_col, ebitda_col, npat_col] if col in df]
        if not required:
            raise ValueError("Insufficient data to compute advanced analytics.")

        work = pd.DataFrame(
            {
                "Revenue": df[rev_col],
                "Gross Margin": df[gm_col],
                "EBITDA": df[ebitda_col],
                "NPAT": df[npat_col],
            }
        )

        cogs_col = "COGS_adj" if "COGS_adj" in df else "COGS"
        if cogs_col in df:
            work["COGS"] = df[cogs_col]
        for candidate in (
            "Variable Expenses",
            "Direct Wages",
            "Fixed Expenses",
            "Admin Wages",
            "Depreciation & Amortization",
            "Interest Expense",
            "Tax Expense",
            "Capex",
            "Unlevered Free Cash Flow",
        ):
            if candidate in df:
                work[candidate] = df[candidate]

        from .advanced import run_advanced_analytics

        results = run_advanced_analytics(
            work,
            window=window,
            annual=annual,
            assumptions=self.valuation_inputs,
        )
        payload: Dict[str, object] = {}
        for key, analysis in results.items():
            payload[key] = {
                "title": analysis.title,
                "description": analysis.description,
                "tables": analysis.tables,
            }
        return payload

    def to_tidy(self) -> pd.DataFrame:
        return self.data.copy()


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").dropna(axis=1, how="all")
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned.reset_index(drop=True)
