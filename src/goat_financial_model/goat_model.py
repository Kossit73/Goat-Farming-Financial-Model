"""Utilities for manipulating the goat farming financial model without Excel."""

from __future__ import annotations

from dataclasses import dataclass, field
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
        self._irr_cache: dict = {}

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
        """Return computed DCF-based NPV for the current model cash-flow profile."""

        summary = self.valuation_summary()
        value = summary.get("npv")
        if value is None or pd.isna(value):
            return None
        return float(value)

    def irr(self) -> Optional[float]:
        """Return computed IRR for the current model cash-flow profile."""

        summary = self.valuation_summary()
        value = summary.get("irr")
        if value is None or pd.isna(value):
            return None
        return float(value)

    def terminal_value(self) -> Optional[float]:
        value = self.valuation_inputs.get("Terminal Value")
        if value is None:
            return None
        return float(value)

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
            ("BYE", "BY"),
            ("YE", "Y"),
        ]
        for alias, replacement in replacements:
            if alias in freq_str:
                freq_str = freq_str.replace(alias, replacement)
        return freq_str

    def _periods_per_year(self, index: Optional[pd.DatetimeIndex] = None) -> int:
        idx = index if index is not None else self.dates
        if not isinstance(idx, pd.DatetimeIndex):
            coerced = pd.to_datetime(idx, errors="coerce")
            if isinstance(coerced, pd.Series):
                coerced = coerced.dropna().to_numpy()
            coerced_index = pd.DatetimeIndex(coerced)
            idx = coerced_index[coerced_index.notna()]
        if len(idx) <= 1:
            return 12

        freq = None
        if isinstance(idx, pd.DatetimeIndex) and len(idx) >= 3:
            try:
                freq = pd.infer_freq(idx)
            except ValueError:
                freq = None
        if freq:
            normalised = self._normalise_frequency_code(freq)
            base_freq = normalised.split("-")[0] if normalised else None
            mapping = {
                "A": 1,
                "Y": 1,
                "Q": 4,
                "M": 12,
                "BM": 12,
                "W": 52,
                "D": 365,
            }
            for key, value in mapping.items():
                if base_freq and base_freq.startswith(key):
                    return value

        deltas = np.diff(idx.asi8)
        if len(deltas) == 0:
            return 12
        median_delta = float(np.median(deltas))
        if median_delta <= 0:
            return 12
        year_delta = pd.Timedelta(days=365.25).value
        periods = int(round(year_delta / median_delta))
        return max(periods, 1)

    def _valuation_rate(self) -> float:
        rate = self.wacc()
        if rate is None:
            return 0.12
        if rate > 1:
            rate /= 100.0
        return float(np.clip(rate, 1e-6, 1.0))

    def _terminal_growth_rate(self) -> float:
        value = self.valuation_inputs.get("Terminal Growth Rate", 0.02)
        try:
            growth = float(value)
        except (TypeError, ValueError):
            growth = 0.02
        if growth > 1:
            growth /= 100.0
        return float(np.clip(growth, -0.2, 0.2))

    def _working_capital_metric(self, key: str, default: float) -> float:
        value = self.valuation_inputs.get(key, default)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(numeric, 0.0)

    def _capital_financing_assumptions(self) -> Optional[pd.DataFrame]:
        for key in ("Assumptions - Capital & Financing", "Capital & Financing"):
            table = self.supplementary_tables.get(key)
            if isinstance(table, pd.DataFrame) and not table.empty:
                return table.copy()
        return None

    def loan_facilities(self) -> Optional[pd.DataFrame]:
        table = self.supplementary_tables.get("Loan Facilities")
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table.copy()

        debt_table = self._capital_financing_assumptions()
        if debt_table is None or debt_table.empty:
            return None

        work = debt_table.copy()
        work["Source"] = work.get("Source", pd.Series("Loan", index=work.index)).astype(str).str.strip()
        work["Amount"] = pd.to_numeric(work.get("Amount"), errors="coerce")
        work["Interest/Return %"] = pd.to_numeric(work.get("Interest/Return %"), errors="coerce")
        work["Term (years)"] = pd.to_numeric(work.get("Term (years)"), errors="coerce")

        debt_mask = (
            work["Amount"].notna()
            & (work["Amount"] > 0)
            & work["Interest/Return %"].notna()
            & (work["Interest/Return %"] > 0)
            & work["Term (years)"].notna()
            & (work["Term (years)"] > 0)
        )
        if not debt_mask.any():
            return None

        start_period = self.dates[0] if len(self.dates) else pd.NaT
        return pd.DataFrame(
            {
                "Loan Name": work.loc[debt_mask, "Source"].replace("", "Loan Facility"),
                "Lender": work.loc[debt_mask, "Source"].replace("", "Lender"),
                "Start Period": start_period,
                "Drawdown Amount": work.loc[debt_mask, "Amount"].astype(float),
                "Interest Rate %": work.loc[debt_mask, "Interest/Return %"].astype(float),
                "Term (years)": work.loc[debt_mask, "Term (years)"].astype(float),
                "Repayment Type": "straight_line",
                "Grace Periods": 0,
                "Balloon Amount": 0.0,
                "Fees": 0.0,
                "Active": True,
            }
        ).reset_index(drop=True)

    def equity_facilities(self) -> Optional[pd.DataFrame]:
        table = self.supplementary_tables.get("Equity Facilities")
        if isinstance(table, pd.DataFrame) and not table.empty:
            return table.copy()

        capitalisation = self.capitalisation_table()
        if capitalisation is None or capitalisation.empty:
            return None

        work = capitalisation.copy()
        work["Year"] = pd.to_numeric(work.get("Year"), errors="coerce")
        work["Shareholder"] = work.get(
            "Shareholder", pd.Series("Investor", index=work.index)
        ).astype(str).str.strip()
        work["Ownership %"] = pd.to_numeric(work.get("Ownership %"), errors="coerce")
        work["Investment"] = pd.to_numeric(work.get("Investment"), errors="coerce")

        valid = work["Investment"].notna() & (work["Investment"] > 0)
        if not valid.any():
            return None

        start_periods: List[pd.Timestamp] = []
        for _, row in work.loc[valid].iterrows():
            year_value = row.get("Year")
            start_periods.append(self._resolve_year_on_index(year_value, self.dates))

        return pd.DataFrame(
            {
                "Investor Name": work.loc[valid, "Shareholder"].replace("", "Investor"),
                "Start Period": start_periods,
                "Contribution Amount": work.loc[valid, "Investment"].astype(float),
                "Ownership %": work.loc[valid, "Ownership %"].astype(float),
                "Share Class": "Ordinary",
                "Issue Costs": 0.0,
                "Active": True,
            }
        ).reset_index(drop=True)

    @staticmethod
    def _coerce_bool_series(values: pd.Series, default: bool = True) -> pd.Series:
        if values.dtype == bool:
            return values.fillna(default).astype(bool)

        mapping = {
            "true": True,
            "1": True,
            "yes": True,
            "y": True,
            "false": False,
            "0": False,
            "no": False,
            "n": False,
        }
        cleaned = values.map(
            lambda value: mapping.get(str(value).strip().lower(), default)
            if pd.notna(value)
            else default
        )
        return cleaned.astype(bool)

    @staticmethod
    def _normalise_repayment_type(value: object) -> str:
        text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
        if text in {"annuity", "amortising", "amortizing"}:
            return "annuity"
        if text in {"bullet", "bullet_payment"}:
            return "bullet"
        return "straight_line"

    @staticmethod
    def _resolve_period_on_index(
        raw_period: object, index: pd.DatetimeIndex
    ) -> Optional[pd.Timestamp]:
        if len(index) == 0:
            return None
        if pd.isna(raw_period) or str(raw_period).strip() == "":
            return pd.Timestamp(index[0])

        candidate = pd.to_datetime(raw_period, errors="coerce")
        if pd.isna(candidate):
            return pd.Timestamp(index[0])

        pos = int(index.searchsorted(candidate, side="left"))
        if pos >= len(index):
            return None
        return pd.Timestamp(index[pos])

    @staticmethod
    def _resolve_year_on_index(
        raw_year: object, index: pd.DatetimeIndex
    ) -> pd.Timestamp:
        if len(index) == 0:
            return pd.NaT

        year = pd.to_numeric(pd.Series([raw_year]), errors="coerce").iloc[0]
        if pd.isna(year):
            return pd.Timestamp(index[0])

        year_int = int(year)
        same_year = index[index.year == year_int]
        if len(same_year):
            return pd.Timestamp(same_year[0])

        future = index[index.year >= year_int]
        if len(future):
            return pd.Timestamp(future[0])
        return pd.Timestamp(index[-1])

    @staticmethod
    def _annuity_payment(
        principal_amount: float, period_rate: float, amortising_periods: int
    ) -> float:
        if amortising_periods <= 0 or principal_amount <= 0:
            return 0.0
        if abs(period_rate) <= 1e-12:
            return principal_amount / amortising_periods
        factor = (1 + period_rate) ** amortising_periods
        return principal_amount * period_rate * factor / (factor - 1)

    def _loan_schedule_detail(
        self, index: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        idx = index if index is not None else self.dates
        if len(idx) == 0:
            return pd.DataFrame()

        facilities = self.loan_facilities()
        if facilities is None or facilities.empty:
            return pd.DataFrame()

        work = facilities.copy()
        for column in [
            "Drawdown Amount",
            "Interest Rate %",
            "Term (years)",
            "Grace Periods",
            "Balloon Amount",
            "Fees",
        ]:
            if column not in work.columns:
                work[column] = np.nan
            work[column] = pd.to_numeric(work.get(column), errors="coerce")

        if "Loan Name" not in work.columns:
            work["Loan Name"] = "Loan Facility"
        work["Loan Name"] = work["Loan Name"].fillna("").astype(str).str.strip()
        work.loc[work["Loan Name"] == "", "Loan Name"] = "Loan Facility"

        if "Lender" not in work.columns:
            work["Lender"] = ""
        work["Lender"] = work["Lender"].fillna("").astype(str).str.strip()

        if "Repayment Type" not in work.columns:
            work["Repayment Type"] = "straight_line"
        work["Repayment Type"] = work["Repayment Type"].map(self._normalise_repayment_type)

        if "Active" not in work.columns:
            work["Active"] = True
        work["Active"] = self._coerce_bool_series(work["Active"], default=True)

        periods_per_year = max(self._periods_per_year(idx), 1)
        loan_dfs: List[pd.DataFrame] = []

        for _, row in work.iterrows():
            if not bool(row.get("Active", True)):
                continue

            drawdown_amount = float(row.get("Drawdown Amount") or 0.0)
            annual_rate = float(row.get("Interest Rate %") or 0.0)
            term_years = float(row.get("Term (years)") or 0.0)
            if drawdown_amount <= 0 or annual_rate <= 0 or term_years <= 0:
                continue

            if annual_rate > 1:
                annual_rate /= 100.0

            term_periods = max(int(round(term_years * periods_per_year)), 1)
            grace_periods = int(max(float(row.get("Grace Periods") or 0.0), 0.0))
            grace_periods = min(grace_periods, term_periods - 1) if term_periods > 1 else 0
            balloon_amount = max(float(row.get("Balloon Amount") or 0.0), 0.0)
            balloon_amount = min(balloon_amount, drawdown_amount)
            fees = max(float(row.get("Fees") or 0.0), 0.0)
            repayment_type = self._normalise_repayment_type(row.get("Repayment Type"))
            start_period = self._resolve_period_on_index(row.get("Start Period"), idx)
            if start_period is None:
                continue

            start_pos = int(idx.get_loc(start_period))
            active_periods = min(term_periods, len(idx) - start_pos)
            if active_periods <= 0:
                continue

            amortising_periods = max(active_periods - grace_periods, 1)
            amortising_principal = max(drawdown_amount - balloon_amount, 0.0)
            period_rate = annual_rate / periods_per_year
            annuity_payment = self._annuity_payment(
                amortising_principal,
                period_rate,
                amortising_periods,
            )
            straight_line_principal = (
                amortising_principal / amortising_periods if amortising_periods > 0 else 0.0
            )

            # Build per-loan arrays — avoids creating one dict per period
            n_max = active_periods
            opening_arr = np.empty(n_max, dtype=float)
            principal_arr = np.empty(n_max, dtype=float)

            outstanding = drawdown_amount
            n_actual = 0
            for offset in range(n_max):
                opening_arr[offset] = outstanding
                interest = outstanding * period_rate
                principal = 0.0

                if offset >= grace_periods and outstanding > 1e-9:
                    periods_remaining = n_max - offset
                    is_final_period = periods_remaining == 1

                    if repayment_type == "bullet":
                        principal = outstanding if is_final_period else 0.0
                    elif repayment_type == "annuity":
                        if is_final_period:
                            principal = outstanding
                        else:
                            principal = max(annuity_payment - interest, 0.0)
                    else:
                        principal = straight_line_principal
                        if is_final_period:
                            principal = outstanding

                    principal = min(principal, outstanding)

                principal_arr[offset] = principal
                outstanding = max(outstanding - principal, 0.0)
                n_actual = offset + 1
                if outstanding <= 1e-9 and offset >= grace_periods:
                    break

            n = n_actual
            periods_slice = idx[start_pos : start_pos + n]
            op = opening_arr[:n]
            pr = principal_arr[:n]
            interest_arr = op * period_rate
            ending_arr = np.maximum(op - pr, 0.0)
            drawdown_col = np.zeros(n, dtype=float)
            drawdown_col[0] = drawdown_amount
            fees_col = np.zeros(n, dtype=float)
            fees_col[0] = fees

            loan_dfs.append(pd.DataFrame({
                "Period": periods_slice,
                "Loan Name": row.get("Loan Name", "Loan Facility"),
                "Lender": row.get("Lender", ""),
                "Opening Debt": op,
                "Debt Drawdown": drawdown_col,
                "Principal Repayment": pr,
                "Interest Expense (Debt)": interest_arr,
                "Debt Service": pr + interest_arr,
                "Ending Debt": ending_arr,
                "Financing Fees": fees_col,
            }))

        if not loan_dfs:
            return pd.DataFrame()

        detail = pd.concat(loan_dfs, ignore_index=True)
        detail["Period"] = pd.to_datetime(detail["Period"], errors="coerce")
        return detail.sort_values(["Period", "Loan Name"], kind="stable").reset_index(drop=True)

    def debt_schedule(self, annual: bool = False) -> pd.DataFrame:
        detail = self._loan_schedule_detail(self.dates)
        if detail.empty:
            return detail
        if not annual:
            return detail

        work = detail.copy()
        work["Year"] = work["Period"].dt.year
        grouped = pd.DataFrame(index=pd.Index(sorted(work["Year"].unique()), name="Year"))
        for column in [
            "Opening Debt",
            "Debt Drawdown",
            "Principal Repayment",
            "Interest Expense (Debt)",
            "Debt Service",
            "Ending Debt",
            "Financing Fees",
        ]:
            if column == "Opening Debt":
                grouped[column] = work.groupby("Year")[column].first()
            elif column == "Ending Debt":
                grouped[column] = work.groupby("Year")[column].last()
            else:
                grouped[column] = work.groupby("Year")[column].sum(min_count=1)
        return grouped

    def _equity_schedule_detail(
        self, index: Optional[pd.DatetimeIndex] = None
    ) -> pd.DataFrame:
        idx = index if index is not None else self.dates
        if len(idx) == 0:
            return pd.DataFrame()

        facilities = self.equity_facilities()
        if facilities is None or facilities.empty:
            return pd.DataFrame()

        work = facilities.copy()
        for column in ["Contribution Amount", "Ownership %", "Issue Costs"]:
            if column not in work.columns:
                work[column] = np.nan
            work[column] = pd.to_numeric(work.get(column), errors="coerce")

        if "Investor Name" not in work.columns:
            work["Investor Name"] = "Investor"
        work["Investor Name"] = work["Investor Name"].fillna("").astype(str).str.strip()
        work.loc[work["Investor Name"] == "", "Investor Name"] = "Investor"

        if "Share Class" not in work.columns:
            work["Share Class"] = "Ordinary"
        work["Share Class"] = work["Share Class"].fillna("").astype(str).str.strip()
        work.loc[work["Share Class"] == "", "Share Class"] = "Ordinary"

        if "Active" not in work.columns:
            work["Active"] = True
        work["Active"] = self._coerce_bool_series(work["Active"], default=True)

        rows: List[Dict[str, object]] = []
        running_contributed_equity = 0.0
        ordered = work.sort_values(["Start Period", "Investor Name"], kind="stable")
        for _, row in ordered.iterrows():
            if not bool(row.get("Active", True)):
                continue

            contribution = max(float(row.get("Contribution Amount") or 0.0), 0.0)
            if contribution <= 0:
                continue

            issue_costs = max(float(row.get("Issue Costs") or 0.0), 0.0)
            start_period = self._resolve_period_on_index(row.get("Start Period"), idx)
            if start_period is None:
                continue

            net_proceeds = max(contribution - issue_costs, 0.0)
            running_contributed_equity += net_proceeds
            rows.append(
                {
                    "Period": pd.Timestamp(start_period),
                    "Investor Name": row.get("Investor Name", "Investor"),
                    "Share Class": row.get("Share Class", "Ordinary"),
                    "Ownership %": float(row.get("Ownership %") or 0.0),
                    "Equity Contribution": contribution,
                    "Equity Issue Costs": issue_costs,
                    "Net Equity Proceeds": net_proceeds,
                    "Cumulative Contributed Equity": running_contributed_equity,
                }
            )

        if not rows:
            return pd.DataFrame()

        detail = pd.DataFrame(rows)
        detail["Period"] = pd.to_datetime(detail["Period"], errors="coerce")
        detail = detail.sort_values(["Period", "Investor Name"], kind="stable").reset_index(
            drop=True
        )
        detail["Cumulative Contributed Equity"] = pd.to_numeric(
            detail["Net Equity Proceeds"], errors="coerce"
        ).fillna(0.0).cumsum()
        return detail

    def equity_schedule(self, annual: bool = False) -> pd.DataFrame:
        detail = self._equity_schedule_detail(self.dates)
        if detail.empty:
            return detail
        if not annual:
            return detail

        work = detail.copy()
        work["Year"] = work["Period"].dt.year
        grouped = pd.DataFrame(index=pd.Index(sorted(work["Year"].unique()), name="Year"))
        for column in [
            "Equity Contribution",
            "Equity Issue Costs",
            "Net Equity Proceeds",
        ]:
            grouped[column] = work.groupby("Year")[column].sum(min_count=1)
        grouped["Cumulative Contributed Equity"] = work.groupby("Year")[
            "Cumulative Contributed Equity"
        ].last()
        return grouped

    def _aggregated_debt_schedule(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        base = pd.DataFrame(index=index.copy())
        for column in [
            "Opening Debt",
            "Debt Drawdown",
            "Principal Repayment",
            "Interest Expense (Debt)",
            "Debt Service",
            "Ending Debt",
            "Financing Fees",
        ]:
            base[column] = 0.0

        detail = self._loan_schedule_detail(index)
        if detail.empty:
            return base

        grouped = detail.groupby("Period").agg(
            {
                "Opening Debt": "sum",
                "Debt Drawdown": "sum",
                "Principal Repayment": "sum",
                "Interest Expense (Debt)": "sum",
                "Debt Service": "sum",
                "Ending Debt": "sum",
                "Financing Fees": "sum",
            }
        )
        grouped.index = pd.to_datetime(grouped.index, errors="coerce")
        for column in grouped.columns:
            base.loc[grouped.index, column] = pd.to_numeric(grouped[column], errors="coerce")
        return base

    def _aggregated_equity_schedule(self, index: pd.DatetimeIndex) -> pd.DataFrame:
        base = pd.DataFrame(index=index.copy())
        for column in [
            "Equity Contribution",
            "Equity Issue Costs",
            "Net Equity Proceeds",
            "Cumulative Contributed Equity",
        ]:
            base[column] = 0.0

        detail = self._equity_schedule_detail(index)
        if detail.empty:
            return base

        grouped = detail.groupby("Period").agg(
            {
                "Equity Contribution": "sum",
                "Equity Issue Costs": "sum",
                "Net Equity Proceeds": "sum",
            }
        )
        grouped.index = pd.to_datetime(grouped.index, errors="coerce")
        for column in grouped.columns:
            base.loc[grouped.index, column] = pd.to_numeric(grouped[column], errors="coerce")
        base["Cumulative Contributed Equity"] = (
            pd.to_numeric(base["Net Equity Proceeds"], errors="coerce").fillna(0.0).cumsum()
        )
        return base

    @staticmethod
    def _effective_tax_rate_from_frame(df: pd.DataFrame) -> float:
        npbt = None
        for candidate in ["NPBT_adj", "NPBT", "Net Profit Before Tax", "Profit Before Tax"]:
            if candidate in df.columns:
                npbt = pd.to_numeric(df[candidate], errors="coerce")
                break

        tax = None
        for candidate in ["Tax Expense_adj", "Tax Expense", "Income Tax Expense", "Tax"]:
            if candidate in df.columns:
                tax = pd.to_numeric(df[candidate], errors="coerce")
                break

        if npbt is not None and tax is not None:
            aligned = pd.concat([npbt, tax], axis=1).dropna()
            aligned = aligned[aligned.iloc[:, 0] > 1e-9]
            if not aligned.empty:
                ratios = aligned.iloc[:, 1] / aligned.iloc[:, 0]
                return float(np.clip(ratios.median(), 0.0, 0.6))

        npat = None
        for candidate in ["NPAT_adj", "NPAT", "Net Profit After Tax", "Net Income"]:
            if candidate in df.columns:
                npat = pd.to_numeric(df[candidate], errors="coerce")
                break

        if npbt is not None and npat is not None:
            aligned = pd.concat([npbt, npat], axis=1).dropna()
            aligned = aligned[aligned.iloc[:, 0] > 1e-9]
            if not aligned.empty:
                ratios = 1 - (aligned.iloc[:, 1] / aligned.iloc[:, 0])
                return float(np.clip(ratios.median(), 0.0, 0.6))

        return 0.28

    def _apply_financing_schedule(
        self, df: pd.DataFrame, *, adjusted: bool = False
    ) -> pd.DataFrame:
        work = df.copy()
        debt = self._aggregated_debt_schedule(work.index)
        equity = self._aggregated_equity_schedule(work.index)
        has_debt = not debt.empty and bool(debt.to_numpy().any())
        has_equity = not equity.empty and bool(equity.to_numpy().any())
        if not has_debt and not has_equity:
            return work

        if has_debt:
            for column in debt.columns:
                work[column] = debt[column].to_numpy()
        if has_equity:
            for column in equity.columns:
                work[column] = equity[column].to_numpy()

        ebit_col = "EBIT_adj" if adjusted and "EBIT_adj" in work.columns else "EBIT"
        npbt_col = "NPBT_adj" if adjusted else "NPBT"
        npat_col = "NPAT_adj" if adjusted else "NPAT"
        tax_col = "Tax Expense_adj" if adjusted else "Tax Expense"
        interest_col = "Interest Expense_adj" if adjusted else "Interest Expense"
        cfo_col = "CFO_adj" if adjusted and "CFO_adj" in work.columns else "CFO"
        cfi_col = "CFI_adj" if adjusted and "CFI_adj" in work.columns else "CFI"
        cff_col = "CFF_adj" if adjusted else "CFF"
        net_cash_col = "Net Cash Flow_adj" if adjusted else "Net Cash Flow"
        closing_cash_col = "Closing Cash Balance_adj" if adjusted else "Closing Cash Balance"
        cash_col = "Cash and Cash Equivalents_adj" if adjusted else "Cash and Cash Equivalents"
        current_assets_col = "Current Assets_adj" if adjusted else "Current Assets"
        non_current_liabilities_col = (
            "Non-current Liabilities_adj" if adjusted else "Non-current Liabilities"
        )
        equity_col = "Equity_adj" if adjusted else "Equity"
        term_debt_col = "Term Debt_adj" if adjusted else "Term Debt"

        original_npat = (
            pd.to_numeric(work.get(npat_col), errors="coerce")
            if npat_col in work.columns
            else None
        )
        original_cfo = (
            pd.to_numeric(work.get(cfo_col), errors="coerce")
            if cfo_col in work.columns
            else None
        )
        baseline_closing_cash = (
            pd.to_numeric(work.get("Closing Cash Balance"), errors="coerce")
            if "Closing Cash Balance" in work.columns
            else None
        )
        baseline_current_assets = (
            pd.to_numeric(work.get("Current Assets"), errors="coerce")
            if "Current Assets" in work.columns
            else None
        )
        baseline_non_current_assets = (
            pd.to_numeric(work.get("Non-current Assets"), errors="coerce")
            if "Non-current Assets" in work.columns
            else None
        )
        baseline_current_liabilities = (
            pd.to_numeric(work.get("Current Liabilities"), errors="coerce")
            if "Current Liabilities" in work.columns
            else None
        )
        baseline_non_current_liabilities = (
            pd.to_numeric(work.get("Non-current Liabilities"), errors="coerce")
            if "Non-current Liabilities" in work.columns
            else None
        )
        baseline_term_debt = (
            pd.to_numeric(work.get("Term Debt"), errors="coerce")
            if "Term Debt" in work.columns
            else pd.Series(0.0, index=work.index, dtype=float)
        )

        if has_debt:
            work[interest_col] = debt["Interest Expense (Debt)"].to_numpy()
        if has_debt and ebit_col in work.columns:
            ebit = pd.to_numeric(work[ebit_col], errors="coerce")
            npbt = ebit - pd.to_numeric(work[interest_col], errors="coerce").fillna(0.0)
            work[npbt_col] = npbt

            tax_rate = self._effective_tax_rate_from_frame(df)
            tax = np.maximum(npbt.fillna(0.0), 0.0) * tax_rate
            work[tax_col] = tax
            work[npat_col] = npbt - tax

            if original_cfo is not None:
                if original_npat is not None:
                    delta_npat = (
                        pd.to_numeric(work[npat_col], errors="coerce").fillna(0.0)
                        - original_npat.fillna(0.0)
                    )
                    work[cfo_col] = original_cfo.fillna(0.0) + delta_npat
                else:
                    work[cfo_col] = original_cfo

        debt_financing_flow = pd.Series(0.0, index=work.index, dtype=float)
        if has_debt:
            debt_financing_flow = (
                debt["Debt Drawdown"] - debt["Principal Repayment"] - debt["Financing Fees"]
            )
        equity_financing_flow = pd.Series(0.0, index=work.index, dtype=float)
        if has_equity:
            equity_financing_flow = equity["Net Equity Proceeds"]
        work[cff_col] = (debt_financing_flow + equity_financing_flow).to_numpy()

        if cfo_col in work.columns and cfi_col in work.columns and cff_col in work.columns:
            work[net_cash_col] = (
                pd.to_numeric(work[cfo_col], errors="coerce").fillna(0.0)
                + pd.to_numeric(work[cfi_col], errors="coerce").fillna(0.0)
                + pd.to_numeric(work[cff_col], errors="coerce").fillna(0.0)
            )
        elif "Net Cash Flow" in work.columns and net_cash_col != "Net Cash Flow":
            work[net_cash_col] = pd.to_numeric(work["Net Cash Flow"], errors="coerce")

        if "Opening Cash Balance" in work.columns and net_cash_col in work.columns:
            opening = pd.to_numeric(work["Opening Cash Balance"], errors="coerce").ffill().fillna(0.0)
            work[closing_cash_col] = opening + pd.to_numeric(work[net_cash_col], errors="coerce").fillna(0.0)
            work[cash_col] = work[closing_cash_col]

        if has_debt:
            work[term_debt_col] = debt["Ending Debt"].to_numpy()

        if baseline_non_current_liabilities is not None and has_debt:
            debt_delta = pd.to_numeric(debt["Ending Debt"], errors="coerce").fillna(0.0) - baseline_term_debt.fillna(0.0)
            work[non_current_liabilities_col] = baseline_non_current_liabilities.fillna(0.0) + debt_delta
        elif has_debt:
            work[non_current_liabilities_col] = debt["Ending Debt"].to_numpy()

        closing_cash_for_delta = (
            pd.to_numeric(work.get(closing_cash_col), errors="coerce")
            if closing_cash_col in work.columns
            else None
        )
        if baseline_current_assets is not None and closing_cash_for_delta is not None and baseline_closing_cash is not None:
            cash_delta = closing_cash_for_delta.fillna(0.0) - baseline_closing_cash.fillna(0.0)
            work[current_assets_col] = baseline_current_assets.fillna(0.0) + cash_delta

        current_assets_for_equity = None
        for candidate in [current_assets_col, "Current Assets"]:
            if candidate in work.columns:
                current_assets_for_equity = pd.to_numeric(work[candidate], errors="coerce")
                break
        non_current_assets_for_equity = baseline_non_current_assets
        current_liabilities_for_equity = baseline_current_liabilities
        non_current_liabilities_for_equity = None
        for candidate in [non_current_liabilities_col, "Non-current Liabilities"]:
            if candidate in work.columns:
                non_current_liabilities_for_equity = pd.to_numeric(
                    work[candidate], errors="coerce"
                )
                break

        if (
            current_assets_for_equity is not None
            and non_current_assets_for_equity is not None
            and current_liabilities_for_equity is not None
            and non_current_liabilities_for_equity is not None
        ):
            work[equity_col] = (
                current_assets_for_equity.fillna(0.0)
                + non_current_assets_for_equity.fillna(0.0)
                - current_liabilities_for_equity.fillna(0.0)
                - non_current_liabilities_for_equity.fillna(0.0)
            )

        return work

    def _capex_spend_schedule(self, index: pd.DatetimeIndex) -> pd.Series:
        capex = self.capex()
        if capex is None:
            cfi = self.cfi()
            if cfi is not None:
                capex_series = (-pd.to_numeric(cfi, errors="coerce")).clip(lower=0.0)
            else:
                capex_series = pd.Series(0.0, index=index, dtype=float)
        else:
            capex_series = pd.to_numeric(capex, errors="coerce").reindex(index).fillna(0.0)

        capex_table = self.capex_schedule()
        if capex_table is None or capex_table.empty:
            return capex_series.reindex(index).fillna(0.0)

        work = capex_table.copy()
        if "Year" not in work.columns or "Spend" not in work.columns:
            return capex_series.reindex(index).fillna(0.0)

        work["Year"] = pd.to_numeric(work.get("Year"), errors="coerce")
        work["Spend"] = pd.to_numeric(work.get("Spend"), errors="coerce")
        by_year = work.dropna(subset=["Year", "Spend"]).groupby("Year")["Spend"].sum()

        supplemental = pd.Series(0.0, index=index, dtype=float)
        for year, spend in by_year.items():
            mask = index.year == int(year)
            if mask.any():
                first_period = index[mask][0]
                supplemental.loc[first_period] += float(spend)

        return capex_series.reindex(index).fillna(0.0).add(supplemental, fill_value=0.0)

    def _effective_tax_rate(self, df: pd.DataFrame) -> float:
        npbt_col = "NPBT_adj" if "NPBT_adj" in df.columns else "NPBT"
        tax_col = "Tax Expense_adj" if "Tax Expense_adj" in df.columns else "Tax Expense"
        npbt = (
            pd.to_numeric(df[npbt_col], errors="coerce")
            if npbt_col in df.columns
            else pd.Series(dtype=float)
        )
        tax = (
            pd.to_numeric(df[tax_col], errors="coerce")
            if tax_col in df.columns
            else pd.Series(dtype=float)
        )
        if not npbt.empty and not tax.empty:
            aligned = pd.concat([npbt, tax], axis=1).dropna()
            aligned = aligned[aligned.iloc[:, 0] > 1e-9]
            if not aligned.empty:
                ratios = aligned.iloc[:, 1] / aligned.iloc[:, 0]
                if ratios.notna().any():
                    return float(np.clip(ratios.median(), 0.0, 0.6))
        return 0.28

    def working_capital_schedule(
        self, df: Optional[pd.DataFrame] = None, annual: bool = False
    ) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        revenue_col = "Revenue_adj" if "Revenue_adj" in df.columns else "Revenue"
        cogs_col = "COGS_adj" if "COGS_adj" in df.columns else "COGS"
        if revenue_col not in df.columns or cogs_col not in df.columns:
            return pd.DataFrame()

        work = pd.DataFrame(index=df.index.copy())
        work["Revenue"] = pd.to_numeric(df.get(revenue_col), errors="coerce").fillna(0.0)
        work["COGS"] = pd.to_numeric(df.get(cogs_col), errors="coerce").fillna(0.0)

        periods_per_year = max(self._periods_per_year(work.index), 1)
        period_days = 365.25 / periods_per_year
        dso = self._working_capital_metric("Receivable Days", 30.0)
        dio = self._working_capital_metric("Inventory Days", 45.0)
        dpo = self._working_capital_metric("Payable Days", 30.0)

        work["Receivable Days"] = dso
        work["Inventory Days"] = dio
        work["Payable Days"] = dpo
        work["Accounts Receivable"] = work["Revenue"] * (dso / period_days)
        work["Inventory"] = work["COGS"] * (dio / period_days)
        work["Accounts Payable"] = work["COGS"] * (dpo / period_days)
        work["Net Working Capital"] = (
            work["Accounts Receivable"] + work["Inventory"] - work["Accounts Payable"]
        )
        work["Change in NWC"] = work["Net Working Capital"].diff().fillna(
            work["Net Working Capital"]
        )

        if not annual:
            return work

        grouped = pd.DataFrame(index=pd.Index(sorted(work.index.year.unique()), name="Year"))
        grouped["Revenue"] = work["Revenue"].groupby(work.index.year).sum(min_count=1)
        grouped["COGS"] = work["COGS"].groupby(work.index.year).sum(min_count=1)
        for column in [
            "Accounts Receivable",
            "Inventory",
            "Accounts Payable",
            "Net Working Capital",
        ]:
            grouped[column] = work[column].groupby(work.index.year).last()
        grouped["Change in NWC"] = work["Change in NWC"].groupby(work.index.year).sum(
            min_count=1
        )
        grouped["Receivable Days"] = dso
        grouped["Inventory Days"] = dio
        grouped["Payable Days"] = dpo
        return grouped

    def debt_capacity_schedule(
        self, df: Optional[pd.DataFrame] = None, annual: bool = False
    ) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        idx = df.index.copy()
        if len(idx) == 0:
            return pd.DataFrame()

        debt_totals = self._aggregated_debt_schedule(idx)
        opening_debt = pd.to_numeric(debt_totals["Opening Debt"], errors="coerce").fillna(0.0)
        drawdown = pd.to_numeric(debt_totals["Debt Drawdown"], errors="coerce").fillna(0.0)
        principal = pd.to_numeric(
            debt_totals["Principal Repayment"], errors="coerce"
        ).fillna(0.0)
        interest = pd.to_numeric(
            debt_totals["Interest Expense (Debt)"], errors="coerce"
        ).fillna(0.0)
        ending_debt = pd.to_numeric(debt_totals["Ending Debt"], errors="coerce").fillna(0.0)
        debt_service = pd.to_numeric(debt_totals["Debt Service"], errors="coerce").fillna(0.0)
        ebit_col = "EBIT_adj" if "EBIT_adj" in df.columns else "EBIT"
        ebitda_col = "EBITDA_adj" if "EBITDA_adj" in df.columns else "EBITDA"
        cfo_col = "CFO_adj" if "CFO_adj" in df.columns else "CFO"
        cash_col = (
            "Closing Cash Balance_adj"
            if "Closing Cash Balance_adj" in df.columns
            else "Closing Cash Balance"
        )

        if ebit_col in df.columns:
            ebit = pd.to_numeric(df[ebit_col], errors="coerce").reindex(idx).fillna(0.0)
        else:
            ebit = pd.Series(0.0, index=idx, dtype=float)
        if ebitda_col in df.columns:
            ebitda = (
                pd.to_numeric(df[ebitda_col], errors="coerce").reindex(idx).fillna(0.0)
            )
        else:
            ebitda = pd.Series(0.0, index=idx, dtype=float)
        if cfo_col in df.columns:
            cfo = pd.to_numeric(df[cfo_col], errors="coerce").reindex(idx).fillna(0.0)
        else:
            cfo = pd.Series(0.0, index=idx, dtype=float)
        if cash_col in df.columns:
            closing_cash = (
                pd.to_numeric(df[cash_col], errors="coerce").reindex(idx).fillna(0.0)
            )
        else:
            closing_cash = pd.Series(0.0, index=idx, dtype=float)

        dscr = self._safe_divide(cfo, debt_service, allow_negative=False)
        interest_coverage = self._safe_divide(ebit, interest, allow_negative=False)
        net_debt = ending_debt - closing_cash
        net_debt_to_ebitda = self._safe_divide(net_debt, ebitda, allow_negative=False)

        dscr_covenant = self._working_capital_metric("DSCR Covenant", 1.20)
        ic_covenant = self._working_capital_metric("Interest Coverage Covenant", 1.50)
        min_cash_reserve = self._working_capital_metric("Minimum Cash Reserve", 25000.0)

        out = pd.DataFrame(
            {
                "Opening Debt": opening_debt,
                "Debt Drawdown": drawdown,
                "Principal Repayment": principal,
                "Interest Expense (Debt)": interest,
                "Debt Service": debt_service,
                "Ending Debt": ending_debt,
                "CFADS": cfo,
                "DSCR": dscr,
                "DSCR Covenant": dscr_covenant,
                "DSCR Headroom": dscr - dscr_covenant,
                "Interest Coverage": interest_coverage,
                "Interest Coverage Covenant": ic_covenant,
                "Interest Coverage Headroom": interest_coverage - ic_covenant,
                "Closing Cash": closing_cash,
                "Minimum Cash Reserve": min_cash_reserve,
                "Cash Reserve Headroom": closing_cash - min_cash_reserve,
                "Net Debt": net_debt,
                "Net Debt / EBITDA": net_debt_to_ebitda,
            },
            index=idx,
        )
        out["Covenant Breach"] = (
            (out["DSCR"].fillna(0.0) < dscr_covenant)
            | (out["Interest Coverage"].fillna(0.0) < ic_covenant)
            | (out["Closing Cash"].fillna(0.0) < min_cash_reserve)
        )

        if not annual:
            return out

        grouped = pd.DataFrame(index=pd.Index(sorted(idx.year.unique()), name="Year"))
        year_groups = idx.year
        _sum_cols = [
            "Debt Drawdown",
            "Principal Repayment",
            "Interest Expense (Debt)",
            "Debt Service",
            "CFADS",
        ]
        _g = out.groupby(year_groups)
        grouped["Opening Debt"] = _g["Opening Debt"].first()
        for column in _sum_cols:
            grouped[column] = _g[column].sum(min_count=1)
        grouped["Ending Debt"] = _g["Ending Debt"].last()
        grouped["Closing Cash"] = _g["Closing Cash"].last()
        grouped["Net Debt"] = grouped["Ending Debt"] - grouped["Closing Cash"]

        _g_scalar = ebit.groupby(year_groups)
        annual_ebit = _g_scalar.sum(min_count=1)
        annual_ebitda = ebitda.groupby(year_groups).sum(min_count=1)

        grouped["DSCR"] = self._safe_divide(
            grouped["CFADS"], grouped["Debt Service"], allow_negative=False
        )
        grouped["DSCR Covenant"] = dscr_covenant
        grouped["DSCR Headroom"] = grouped["DSCR"] - dscr_covenant
        grouped["Interest Coverage"] = self._safe_divide(
            annual_ebit, grouped["Interest Expense (Debt)"], allow_negative=False
        )
        grouped["Interest Coverage Covenant"] = ic_covenant
        grouped["Interest Coverage Headroom"] = grouped["Interest Coverage"] - ic_covenant
        grouped["Minimum Cash Reserve"] = min_cash_reserve
        grouped["Cash Reserve Headroom"] = grouped["Closing Cash"] - min_cash_reserve
        grouped["Net Debt / EBITDA"] = self._safe_divide(
            grouped["Net Debt"], annual_ebitda, allow_negative=False
        )
        grouped["Covenant Breach"] = (
            (grouped["DSCR"].fillna(0.0) < dscr_covenant)
            | (grouped["Interest Coverage"].fillna(0.0) < ic_covenant)
            | (grouped["Closing Cash"].fillna(0.0) < min_cash_reserve)
        )
        return grouped

    def ufcf_schedule(
        self, df: Optional[pd.DataFrame] = None, annual: bool = False
    ) -> pd.DataFrame:
        if df is None:
            df = self.to_tidy()

        idx = df.index.copy()
        if len(idx) == 0:
            return pd.DataFrame()

        ebit_col = "EBIT_adj" if "EBIT_adj" in df.columns else "EBIT"
        dep_col = (
            "Depreciation & Amortization_adj"
            if "Depreciation & Amortization_adj" in df.columns
            else "Depreciation & Amortization"
        )
        if ebit_col in df.columns:
            ebit = pd.to_numeric(df[ebit_col], errors="coerce").reindex(idx).fillna(0.0)
        else:
            ebit = pd.Series(0.0, index=idx, dtype=float)
        if dep_col in df.columns:
            depreciation = (
                pd.to_numeric(df[dep_col], errors="coerce").reindex(idx).fillna(0.0)
            )
        else:
            depreciation = pd.Series(0.0, index=idx, dtype=float)
        tax_rate = self._effective_tax_rate(df)
        nopat = ebit * (1.0 - tax_rate)

        wc = self.working_capital_schedule(df, annual=False)
        capex = self._capex_spend_schedule(idx)

        out = pd.DataFrame(index=idx)
        out["EBIT"] = ebit
        out["Tax Rate"] = tax_rate
        out["NOPAT"] = nopat
        out["Depreciation"] = depreciation
        out["Capex"] = capex.reindex(idx).fillna(0.0)
        out["Change in NWC"] = (
            wc["Change in NWC"].reindex(idx).fillna(0.0) if not wc.empty else 0.0
        )
        out["UFCF"] = out["NOPAT"] + out["Depreciation"] - out["Capex"] - out["Change in NWC"]

        if not annual:
            return out

        grouped = pd.DataFrame(index=pd.Index(sorted(idx.year.unique()), name="Year"))
        for column in ["EBIT", "NOPAT", "Depreciation", "Capex", "Change in NWC", "UFCF"]:
            grouped[column] = out[column].groupby(idx.year).sum(min_count=1)
        grouped["Tax Rate"] = tax_rate
        return grouped

    def valuation_summary(
        self, df: Optional[pd.DataFrame] = None, annual: bool = False
    ) -> Dict[str, object]:
        working_df = df if df is not None else self.to_tidy()

        if working_df is None or working_df.empty:
            return {}

        ufcf_frame = self.ufcf_schedule(working_df, annual=annual)
        if ufcf_frame.empty or "UFCF" not in ufcf_frame.columns:
            ufcf_table = self.ufcf()
            if ufcf_table is None or annual:
                return {}
            ufcf_series = ufcf_table.sort_index()
            ufcf_frame = pd.DataFrame({"UFCF": ufcf_series}, index=ufcf_series.index)
        else:
            ufcf_series = pd.to_numeric(ufcf_frame["UFCF"], errors="coerce").dropna()

        if ufcf_series.empty:
            return {}

        rate = self._valuation_rate()
        configured_growth = self._terminal_growth_rate()
        manual_terminal_value = self.terminal_value()
        periods_per_year = max(self._periods_per_year(ufcf_series.index), 1)
        inferred_freq = None
        if len(ufcf_series.index) >= 3:
            try:
                inferred_freq = pd.infer_freq(ufcf_series.index)
            except ValueError:
                inferred_freq = None
        if inferred_freq:
            period_step = 1.0 / periods_per_year
            diffs_years = pd.Series(period_step, index=ufcf_series.index, dtype=float)
            periods = np.arange(1, len(ufcf_series) + 1, dtype=float) * period_step
        else:
            diffs_days = ufcf_series.index.to_series().diff().dt.days.astype(float)
            valid_diffs = diffs_days.iloc[1:][np.isfinite(diffs_days.iloc[1:])]
            median_days = (
                float(np.median(valid_diffs))
                if not valid_diffs.empty
                else 365.25 / periods_per_year
            )
            if not np.isfinite(median_days) or median_days <= 0:
                median_days = 365.25 / periods_per_year
            diffs_years = (diffs_days / 365.25).fillna(median_days / 365.25).clip(lower=1e-9)
            periods = diffs_years.cumsum().to_numpy()

        trailing_periods = min(periods_per_year, len(ufcf_series))
        trailing_ufcf = float(ufcf_series.iloc[-trailing_periods:].sum())
        if trailing_periods < periods_per_year and trailing_periods > 0:
            trailing_ufcf *= periods_per_year / trailing_periods

        terminal_period = float(periods[-1] + diffs_years.iloc[-1]) if len(periods) else 1.0

        def _terminal_value_at(discount_rate: float) -> Tuple[float, Optional[float]]:
            if manual_terminal_value is not None and np.isfinite(manual_terminal_value):
                return float(manual_terminal_value), None
            if trailing_ufcf <= 0:
                return 0.0, None

            growth_rate = configured_growth
            if discount_rate <= growth_rate:
                growth_rate = discount_rate - 1e-6
            if discount_rate <= growth_rate:
                return 0.0, None

            terminal = trailing_ufcf * (1.0 + growth_rate) / (discount_rate - growth_rate)
            return float(terminal), float(growth_rate)

        terminal_value, effective_growth = _terminal_value_at(rate)
        discount_factors = 1 / np.power(1 + rate, periods)
        pv_cash_flows = ufcf_series.to_numpy() * discount_factors
        terminal_value_pv = terminal_value / ((1 + rate) ** terminal_period)
        enterprise_value = float(np.nansum(pv_cash_flows) + terminal_value_pv)

        values = ufcf_series.to_numpy(dtype=float)

        def _npv_at(test_rate: float) -> float:
            if test_rate <= -0.9999:
                return np.nan
            discounted = np.nansum(values / np.power(1 + test_rate, periods))
            terminal_value_at_rate, _ = _terminal_value_at(test_rate)
            discounted += terminal_value_at_rate / ((1 + test_rate) ** terminal_period)
            return float(discounted)

        _irr_key = (
            int(pd.util.hash_pandas_object(ufcf_series).sum()),
            rate,
            configured_growth,
            manual_terminal_value,
        )
        _sentinel = object()
        irr_value: Optional[float] = self._irr_cache.get(_irr_key, _sentinel)  # type: ignore[assignment]
        if irr_value is _sentinel:
            irr_value = None
            search_grid = np.concatenate(([-0.95], np.linspace(-0.9, 5.0, 2000)))
            previous_rate: Optional[float] = None
            previous_npv: Optional[float] = None
            for candidate_rate in search_grid:
                candidate_npv = _npv_at(float(candidate_rate))
                if not np.isfinite(candidate_npv):
                    previous_rate = None
                    previous_npv = None
                    continue
                if (
                    previous_rate is not None
                    and previous_npv is not None
                    and previous_npv * candidate_npv <= 0
                ):
                    low, high = previous_rate, float(candidate_rate)
                    npv_low, npv_high = previous_npv, float(candidate_npv)
                    for _ in range(120):
                        mid = (low + high) / 2.0
                        npv_mid = _npv_at(mid)
                        if not np.isfinite(npv_mid):
                            break
                        if abs(npv_mid) < 1e-7:
                            irr_value = mid
                            break
                        if npv_low * npv_mid < 0:
                            high = mid
                            npv_high = npv_mid
                        else:
                            low = mid
                            npv_low = npv_mid
                    if irr_value is None:
                        irr_value = (low + high) / 2.0
                    break
                previous_rate = float(candidate_rate)
                previous_npv = float(candidate_npv)
            self._irr_cache[_irr_key] = irr_value

        cumulative = np.cumsum(values)
        payback_years: Optional[float] = None
        if cumulative[0] >= 0:
            payback_years = 0.0
        else:
            crossing = np.where(cumulative >= 0)[0]
            if len(crossing) > 0:
                cross_idx = int(crossing[0])
                if cross_idx == 0:
                    payback_years = float(periods[0])
                else:
                    prev_cum = float(cumulative[cross_idx - 1])
                    curr_cum = float(cumulative[cross_idx])
                    prev_t = float(periods[cross_idx - 1])
                    curr_t = float(periods[cross_idx])
                    if np.isclose(curr_cum, prev_cum):
                        payback_years = curr_t
                    else:
                        frac = float(np.clip((0.0 - prev_cum) / (curr_cum - prev_cum), 0.0, 1.0))
                        payback_years = prev_t + (curr_t - prev_t) * frac

        cash_flow_df = pd.DataFrame(
            {
                "UFCF": ufcf_series.to_numpy(),
                "Discount Factor": discount_factors,
                "Present Value": pv_cash_flows,
            },
            index=ufcf_series.index,
        )

        return {
            "cash_flows": cash_flow_df,
            "discount_rate": rate,
            "terminal_growth_rate": effective_growth,
            "terminal_value": terminal_value,
            "terminal_value_pv": terminal_value_pv,
            "enterprise_value": enterprise_value,
            "npv": enterprise_value,
            "irr": irr_value,
            "payback_years": payback_years,
            "ufcf_schedule": ufcf_frame,
        }

    def computed_npv(self) -> Optional[float]:
        """Return DCF-based NPV from UFCF timeline, WACC, and terminal value."""
        return self.npv()

    def computed_irr(self) -> Optional[float]:
        """Solve an IRR from irregular UFCF timing using a bisection search."""
        return self.irr()

    def payback_period_years(self) -> Optional[float]:
        """Estimate simple payback period (years) from UFCF timeline."""
        summary = self.valuation_summary()
        payback = summary.get("payback_years")
        return float(payback) if payback is not None else None

    def _reference_milk_price_per_litre(self) -> Optional[float]:
        pricing = self.supplementary_tables.get("Assumptions - Pricing")
        if pricing is None or pricing.empty:
            return None
        work = pricing.copy()
        if "Base Price" not in work.columns:
            return None
        prices = pd.to_numeric(work.get("Base Price"), errors="coerce")
        if "Unit" in work.columns:
            units = work["Unit"].astype(str).str.lower()
            litre_mask = units.str.contains("litre|liter|l", regex=True)
            filtered = prices[litre_mask]
            if filtered.notna().any():
                return float(filtered.median())
        if prices.notna().any():
            return float(prices.median())
        return None

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
        summary = self.valuation_summary()
        if not summary:
            raise ValueError("Unable to derive a UFCF schedule for discounted cash-flow analysis.")

        cash_flows = summary.get("cash_flows")
        if not isinstance(cash_flows, pd.DataFrame) or cash_flows.empty:
            raise ValueError("Discounted cash-flow analysis requires non-empty UFCF cash flows.")

        schedule = cash_flows.copy()
        if not isinstance(schedule.index, pd.DatetimeIndex):
            raise ValueError("Discounted cash-flow analysis requires dated UFCF periods.")

        diffs_days = schedule.index.to_series().diff().dt.days.astype(float)
        valid_diffs = diffs_days.iloc[1:][np.isfinite(diffs_days.iloc[1:])]
        if not valid_diffs.empty:
            median_days = float(np.median(valid_diffs))
            irregular = valid_diffs[
                (valid_diffs - median_days).abs() > max(median_days * 0.25, 1.0)
            ]
            if not irregular.empty:
                warnings.warn(
                    "Cash-flow timeline contains irregular step sizes; results may be approximate.",
                    DataQualityWarning,
                    stacklevel=2,
                )

        output: Dict[str, object] = {
            "cash_flows": schedule,
            "discount_rate": summary.get("discount_rate"),
            "enterprise_value": float(summary.get("enterprise_value", np.nan)),
            "npv": float(summary.get("npv", np.nan)),
        }
        if summary.get("terminal_value_pv") is not None:
            output["terminal_value_pv"] = float(summary["terminal_value_pv"])
        if summary.get("terminal_value") is not None:
            output["terminal_value"] = float(summary["terminal_value"])
        if summary.get("irr") is not None:
            output["irr"] = float(summary["irr"])
        return output

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

        df = self._apply_financing_schedule(df, adjusted=True)

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

        price_per_litre = self._reference_milk_price_per_litre()
        herd_series = (
            pd.to_numeric(df["Herd Size (heads)"], errors="coerce")
            if "Herd Size (heads)" in df
            else None
        )
        if price_per_litre is not None and price_per_litre > 0 and herd_series is not None:
            if annual:
                herd_group = herd_series.groupby(df.index.year).mean()
            else:
                herd_group = herd_series.copy()
            litres = self._safe_divide(grp["Revenue"], price_per_litre)
            out["Milk Yield per Doe"] = self._safe_divide(litres, herd_group)
            out["Feed Cost per Litre"] = self._safe_divide(grp["COGS"], litres)

        valuation_summary = self.valuation_summary(df)
        computed_npv = valuation_summary.get("npv")
        if computed_npv is not None:
            out["NPV"] = float(computed_npv)

        computed_irr = valuation_summary.get("irr")
        if computed_irr is not None:
            out["IRR"] = float(computed_irr)
        payback = valuation_summary.get("payback_years")
        if payback is not None:
            out["Payback Period (Years)"] = payback

        debt_capacity = self.debt_capacity_schedule(df, annual=annual)
        if not debt_capacity.empty:
            for metric in [
                "DSCR",
                "DSCR Headroom",
                "Interest Coverage",
                "Interest Coverage Headroom",
                "Cash Reserve Headroom",
            ]:
                if metric in debt_capacity.columns:
                    out[metric] = pd.to_numeric(
                        debt_capacity[metric], errors="coerce"
                    ).to_numpy()
            if "Covenant Breach" in debt_capacity.columns:
                out["Covenant Breach"] = (
                    debt_capacity["Covenant Breach"].astype(bool).to_numpy()
                )
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

    # ---------- Model audit ----------
    @staticmethod
    def _series_from_candidates(
        df: pd.DataFrame, *candidates: str
    ) -> Optional[pd.Series]:
        for candidate in candidates:
            if candidate in df.columns:
                return pd.to_numeric(df[candidate], errors="coerce")
        return None

    @staticmethod
    def _materiality_tolerance(*series: Optional[pd.Series]) -> float:
        scale = 1.0
        for series_item in series:
            if series_item is None:
                continue
            numeric = pd.to_numeric(series_item, errors="coerce")
            if numeric.notna().any():
                scale = max(scale, float(numeric.abs().max()))
        return max(1e-6, scale * 1e-4)

    @staticmethod
    def _format_audit_periods(index: pd.Index, limit: int = 3) -> str:
        labels: List[str] = []
        for value in list(index[:limit]):
            if isinstance(value, pd.Timestamp):
                labels.append(value.strftime("%Y-%m-%d"))
            else:
                labels.append(str(value))
        if len(index) > limit:
            labels.append("...")
        return ", ".join(labels)

    @staticmethod
    def _audit_issue(
        issues: List[Dict[str, Any]],
        *,
        severity: str,
        category: str,
        metric: str,
        message: str,
        reasoning: str,
        recommendation: str,
        periods: str = "",
        value: Optional[float] = None,
        threshold: Optional[float] = None,
    ) -> None:
        issues.append(
            {
                "Severity": severity,
                "Category": category,
                "Metric": metric,
                "Message": message,
                "Reasoning": reasoning,
                "Recommendation": recommendation,
                "Periods": periods,
                "Value": value,
                "Threshold": threshold,
            }
        )

    def model_audit(
        self, df: Optional[pd.DataFrame] = None, annual: bool = True
    ) -> Dict[str, object]:
        if df is None:
            df = self.to_tidy()
        if df is None or df.empty:
            empty_issues = pd.DataFrame(
                columns=[
                    "Severity",
                    "Category",
                    "Metric",
                    "Message",
                    "Reasoning",
                    "Recommendation",
                    "Periods",
                    "Value",
                    "Threshold",
                ]
            )
            return {
                "status": "critical",
                "score": 0,
                "headline": "Model audit could not run because the operating schedule is empty.",
                "summary": pd.DataFrame(
                    [{"Status": "critical", "Critical Issues": 1, "Warnings": 0, "Info": 0}]
                ),
                "issues": empty_issues,
                "reasoning": [
                    "The model has no time-series data, so statements, cash flow, and valuation cannot be validated."
                ],
            }

        work = df.copy()
        issues: List[Dict[str, Any]] = []

        def _check_reconciliation(
            *,
            metric: str,
            lhs: Optional[pd.Series],
            rhs: Optional[pd.Series],
            message: str,
            reasoning: str,
            recommendation: str,
            severity: str = "Critical",
        ) -> None:
            if lhs is None or rhs is None:
                return
            lhs_aligned, rhs_aligned = lhs.align(rhs, join="inner")
            valid = lhs_aligned.notna() & rhs_aligned.notna()
            if not valid.any():
                return
            residual = lhs_aligned[valid] - rhs_aligned[valid]
            tolerance = self._materiality_tolerance(lhs_aligned[valid], rhs_aligned[valid])
            breaches = residual.abs() > tolerance
            if breaches.any():
                self._audit_issue(
                    issues,
                    severity=severity,
                    category="Reconciliation",
                    metric=metric,
                    message=message,
                    reasoning=reasoning,
                    recommendation=recommendation,
                    periods=self._format_audit_periods(residual.index[breaches]),
                    value=float(residual.abs()[breaches].max()),
                    threshold=tolerance,
                )

        revenue = self._series_from_candidates(work, "Revenue_adj", "Revenue")
        cogs = self._series_from_candidates(work, "COGS_adj", "COGS")
        gross_margin = self._series_from_candidates(
            work, "Gross Margin_adj", "Gross Margin"
        )
        ebitda = self._series_from_candidates(work, "EBITDA_adj", "EBITDA")
        depreciation = self._series_from_candidates(
            work,
            "Depreciation & Amortization_adj",
            "Depreciation & Amortization",
            "Depreciation",
        )
        ebit = self._series_from_candidates(work, "EBIT_adj", "EBIT")
        interest = self._series_from_candidates(
            work, "Interest Expense_adj", "Interest Expense"
        )
        npbt = self._series_from_candidates(work, "NPBT_adj", "NPBT")
        tax = self._series_from_candidates(work, "Tax Expense_adj", "Tax Expense")
        npat = self._series_from_candidates(work, "NPAT_adj", "NPAT")
        cfo = self._series_from_candidates(work, "CFO_adj", "CFO")
        cfi = self._series_from_candidates(work, "CFI_adj", "CFI")
        cff = self._series_from_candidates(work, "CFF_adj", "CFF")
        net_cash_flow = self._series_from_candidates(
            work, "Net Cash Flow_adj", "Net Cash Flow"
        )
        opening_cash = self._series_from_candidates(
            work, "Opening Cash Balance_adj", "Opening Cash Balance"
        )
        closing_cash = self._series_from_candidates(
            work,
            "Closing Cash Balance_adj",
            "Cash and Cash Equivalents_adj",
            "Closing Cash Balance",
            "Cash and Cash Equivalents",
        )
        current_assets = self._series_from_candidates(
            work, "Current Assets_adj", "Current Assets"
        )
        non_current_assets = self._series_from_candidates(
            work, "Non-current Assets_adj", "Non-current Assets"
        )
        current_liabilities = self._series_from_candidates(
            work, "Current Liabilities_adj", "Current Liabilities"
        )
        non_current_liabilities = self._series_from_candidates(
            work, "Non-current Liabilities_adj", "Non-current Liabilities"
        )
        equity = self._series_from_candidates(work, "Equity_adj", "Equity")

        _check_reconciliation(
            metric="Gross Margin",
            lhs=gross_margin,
            rhs=revenue.subtract(cogs, fill_value=np.nan)
            if revenue is not None and cogs is not None
            else None,
            message="Gross margin does not reconcile to revenue minus COGS.",
            reasoning="This indicates the operating profit build is internally inconsistent, so downstream EBITDA and valuation metrics may be unreliable.",
            recommendation="Inspect the revenue, COGS, and gross margin series for manual overrides or missing mapping.",
        )
        _check_reconciliation(
            metric="EBIT",
            lhs=ebit,
            rhs=ebitda.subtract(depreciation, fill_value=np.nan)
            if ebitda is not None and depreciation is not None
            else None,
            message="EBIT does not reconcile to EBITDA less depreciation and amortization.",
            reasoning="The operating earnings bridge is broken, so profitability ratios and tax calculations can drift away from the model structure.",
            recommendation="Review EBITDA, depreciation, and EBIT lines for duplicate or missing expense treatment.",
        )
        _check_reconciliation(
            metric="NPBT",
            lhs=npbt,
            rhs=ebit.subtract(interest, fill_value=np.nan)
            if ebit is not None and interest is not None
            else None,
            message="Profit before tax does not reconcile to EBIT less interest expense.",
            reasoning="Financing charges are not flowing cleanly into pre-tax profit, which can distort both tax and investor-return outputs.",
            recommendation="Check interest expense sourcing and any manual NPBT overrides.",
        )
        _check_reconciliation(
            metric="NPAT",
            lhs=npat,
            rhs=npbt.subtract(tax, fill_value=np.nan)
            if npbt is not None and tax is not None
            else None,
            message="Net profit after tax does not reconcile to pre-tax profit less tax expense.",
            reasoning="After-tax profitability is inconsistent, so retained earnings and valuation metrics may not represent the modeled economics.",
            recommendation="Review tax expense logic and any direct NPAT inputs.",
        )
        _check_reconciliation(
            metric="Net Cash Flow",
            lhs=net_cash_flow,
            rhs=(cfo + cfi + cff) if cfo is not None and cfi is not None and cff is not None else None,
            message="Net cash flow does not reconcile to CFO plus CFI plus CFF.",
            reasoning="Cash flow statement sections are not summing to the reported net movement in cash, which breaks liquidity analysis.",
            recommendation="Inspect the operating, investing, and financing cash flow lines for missing adjustments or sign errors.",
        )
        _check_reconciliation(
            metric="Closing Cash",
            lhs=closing_cash,
            rhs=opening_cash.add(net_cash_flow, fill_value=np.nan)
            if opening_cash is not None and net_cash_flow is not None
            else None,
            message="Closing cash does not reconcile to opening cash plus net cash flow.",
            reasoning="The cash roll-forward is broken, so balance-sheet liquidity and funding headroom cannot be trusted.",
            recommendation="Review opening cash, cash flow totals, and any direct closing-cash overrides.",
        )
        _check_reconciliation(
            metric="Balance Sheet",
            lhs=current_assets.add(non_current_assets, fill_value=np.nan)
            if current_assets is not None and non_current_assets is not None
            else None,
            rhs=current_liabilities.add(non_current_liabilities, fill_value=np.nan).add(
                equity, fill_value=np.nan
            )
            if current_liabilities is not None
            and non_current_liabilities is not None
            and equity is not None
            else None,
            message="The balance sheet does not balance between assets and equity plus liabilities.",
            reasoning="This is a core structural model error; valuation, leverage, and liquidity outputs should be treated as unreliable until the balance sheet balances.",
            recommendation="Check asset, liability, and equity roll-forwards, especially cash and financing schedules.",
        )

        if closing_cash is not None and closing_cash.notna().any():
            negative_cash = closing_cash < -self._materiality_tolerance(closing_cash)
            if negative_cash.any():
                self._audit_issue(
                    issues,
                    severity="Warning",
                    category="Liquidity",
                    metric="Closing Cash",
                    message="Closing cash turns negative in one or more periods.",
                    reasoning="The model indicates a funding gap, so the operation cannot sustain its planned cash outflows without additional financing or working-capital improvement.",
                    recommendation="Add funding, reduce capex or costs, or revise operating assumptions to restore positive cash coverage.",
                    periods=self._format_audit_periods(closing_cash.index[negative_cash]),
                    value=float(closing_cash[negative_cash].min()),
                    threshold=0.0,
                )

        if revenue is not None and gross_margin is not None:
            gross_margin_pct = self._safe_divide(gross_margin, revenue)
            if gross_margin_pct.notna().any():
                excessive = gross_margin_pct > 1.0 + 1e-6
                negative = gross_margin_pct < -1.0 - 1e-6
                if excessive.any():
                    self._audit_issue(
                        issues,
                        severity="Warning",
                        category="Ratio",
                        metric="Gross Margin %",
                        message="Gross margin exceeds 100% in one or more periods.",
                        reasoning="This usually means COGS is negative or revenue/gross-profit mapping is broken, which is rarely economically valid for this operating model.",
                        recommendation="Review pricing, production, and COGS inputs for sign or linkage errors.",
                        periods=self._format_audit_periods(gross_margin_pct.index[excessive]),
                        value=float(gross_margin_pct[excessive].max()),
                        threshold=1.0,
                    )
                if negative.any():
                    self._audit_issue(
                        issues,
                        severity="Info",
                        category="Ratio",
                        metric="Gross Margin %",
                        message="Gross margin is worse than -100% in one or more periods.",
                        reasoning="The model is showing losses materially larger than revenue, which may be possible under stress but usually merits assumption review.",
                        recommendation="Inspect pricing and variable-cost assumptions for severe downside or bad input mapping.",
                        periods=self._format_audit_periods(gross_margin_pct.index[negative]),
                        value=float(gross_margin_pct[negative].min()),
                        threshold=-1.0,
                    )

        valuation_summary = self.valuation_summary(work)
        discount_rate = self._valuation_rate()
        terminal_growth_rate = self._terminal_growth_rate()
        enterprise_value = pd.to_numeric(
            pd.Series([valuation_summary.get("enterprise_value")]), errors="coerce"
        ).iloc[0]
        terminal_value_pv = pd.to_numeric(
            pd.Series([valuation_summary.get("terminal_value_pv")]), errors="coerce"
        ).iloc[0]
        irr_value = pd.to_numeric(
            pd.Series([valuation_summary.get("irr")]), errors="coerce"
        ).iloc[0]

        if self.wacc() is None:
            self._audit_issue(
                issues,
                severity="Info",
                category="Valuation",
                metric="WACC",
                message="WACC is not explicitly set, so the model is using the internal default discount rate.",
                reasoning="Valuation still computes, but investor-return outputs may not reflect the intended capital cost assumptions.",
                recommendation="Set WACC in the valuation inputs if you want valuation to reflect a specific hurdle rate.",
            )
        if (
            discount_rate is not None
            and terminal_growth_rate is not None
            and float(discount_rate) <= float(terminal_growth_rate) + 1e-9
        ):
            self._audit_issue(
                issues,
                severity="Critical",
                category="Valuation",
                metric="Terminal Growth Rate",
                message="Terminal growth is at or above the discount rate.",
                reasoning="That assumption makes a perpetuity valuation structurally unstable and can overstate enterprise value.",
                recommendation="Set terminal growth below WACC to keep the DCF mathematically and economically credible.",
                value=float(terminal_growth_rate),
                threshold=float(discount_rate),
            )
        if pd.notna(enterprise_value) and enterprise_value <= 0:
            self._audit_issue(
                issues,
                severity="Warning",
                category="Valuation",
                metric="Enterprise Value",
                message="Computed enterprise value is non-positive.",
                reasoning="The current free-cash-flow profile does not support positive economic value under the modeled assumptions.",
                recommendation="Review profitability, capex, working-capital, and discount-rate assumptions.",
                value=float(enterprise_value),
                threshold=0.0,
            )
        if (
            pd.notna(enterprise_value)
            and enterprise_value > 0
            and pd.notna(terminal_value_pv)
        ):
            terminal_share = float(terminal_value_pv / enterprise_value)
            if terminal_share > 0.85:
                self._audit_issue(
                    issues,
                    severity="Warning",
                    category="Valuation",
                    metric="Terminal Value Share",
                    message="Terminal value drives more than 85% of enterprise value.",
                    reasoning="The valuation is highly back-loaded, so small changes in long-run growth or discount rate can move NPV materially.",
                    recommendation="Stress-test WACC, terminal growth, and near-term cash generation to reduce valuation concentration in the terminal period.",
                    value=terminal_share,
                    threshold=0.85,
                )
        if pd.isna(irr_value):
            self._audit_issue(
                issues,
                severity="Info",
                category="Valuation",
                metric="IRR",
                message="IRR could not be solved from the current unlevered cash-flow profile.",
                reasoning="This usually means the cash flows do not cross the zero-NPV threshold within a reasonable discount-rate range or remain one-sided in sign.",
                recommendation="Review the UFCF profile, terminal value assumptions, and whether the project ever recovers its initial investment.",
            )

        debt_capacity = self.debt_capacity_schedule(work, annual=annual)
        if not debt_capacity.empty:
            if "Covenant Breach" in debt_capacity.columns:
                breaches = debt_capacity["Covenant Breach"].fillna(False).astype(bool)
                if breaches.any():
                    self._audit_issue(
                        issues,
                        severity="Warning",
                        category="Financing",
                        metric="Covenant Breach",
                        message="One or more debt covenant periods are breached.",
                        reasoning="The forecast violates lender protection thresholds, which implies refinancing risk or a need for remedial funding action.",
                        recommendation="Review debt sizing, repayment profile, cash generation, and reserve assumptions.",
                        periods=self._format_audit_periods(debt_capacity.index[breaches]),
                        value=float(breaches.sum()),
                        threshold=0.0,
                    )
            if "DSCR" in debt_capacity.columns:
                dscr = pd.to_numeric(debt_capacity["DSCR"], errors="coerce")
                weak_dscr = dscr < 1.0 - 1e-6
                if weak_dscr.any():
                    self._audit_issue(
                        issues,
                        severity="Warning",
                        category="Financing",
                        metric="DSCR",
                        message="Debt service coverage falls below 1.0x in one or more periods.",
                        reasoning="Operating cash flow is insufficient to cover scheduled debt service, signaling structural debt stress.",
                        recommendation="Restructure debt service, increase cash generation, or add equity support.",
                        periods=self._format_audit_periods(dscr.index[weak_dscr]),
                        value=float(dscr[weak_dscr].min()),
                        threshold=1.0,
                    )

        severity_order = {"Critical": 0, "Warning": 1, "Info": 2}
        issues_df = pd.DataFrame(issues)
        if not issues_df.empty:
            issues_df["Severity Rank"] = issues_df["Severity"].map(severity_order).fillna(99)
            issues_df = issues_df.sort_values(
                ["Severity Rank", "Category", "Metric"], kind="stable"
            ).drop(columns=["Severity Rank"])

        critical_count = int((issues_df["Severity"] == "Critical").sum()) if not issues_df.empty else 0
        warning_count = int((issues_df["Severity"] == "Warning").sum()) if not issues_df.empty else 0
        info_count = int((issues_df["Severity"] == "Info").sum()) if not issues_df.empty else 0

        if critical_count > 0:
            status = "critical"
            headline = (
                f"Model audit found {critical_count} critical issue(s) and {warning_count} warning(s)."
            )
        elif warning_count > 0:
            status = "warning"
            headline = f"Model audit found {warning_count} warning(s)."
        else:
            status = "pass"
            headline = "Model audit found no material structural issues."

        score = max(0, 100 - critical_count * 35 - warning_count * 10 - info_count * 3)
        reasoning_notes: List[str] = []
        if critical_count:
            reasoning_notes.append(
                "Core reconciliations or valuation assumptions are broken, so headline outputs should be reviewed before using the model for decisions."
            )
        if not issues_df.empty and (issues_df["Category"] == "Liquidity").any():
            reasoning_notes.append(
                "Liquidity risk is present because the cash roll-forward shows periods with insufficient cash coverage."
            )
        if not issues_df.empty and (issues_df["Category"] == "Financing").any():
            reasoning_notes.append(
                "The financing structure is under pressure because covenant headroom or debt service coverage falls below acceptable levels."
            )
        if not issues_df.empty and (issues_df["Category"] == "Valuation").any():
            reasoning_notes.append(
                "Valuation is assumption-sensitive, so NPV and IRR should be interpreted alongside the terminal-value and discount-rate diagnostics."
            )
        if not reasoning_notes:
            reasoning_notes.append(
                "The model is internally coherent on the tested checks, so the key statements, cash flow, and valuation outputs reconcile."
            )

        summary = pd.DataFrame(
            [
                {
                    "Status": status,
                    "Score": score,
                    "Critical Issues": critical_count,
                    "Warnings": warning_count,
                    "Info": info_count,
                }
            ]
        )

        return {
            "status": status,
            "score": score,
            "headline": headline,
            "summary": summary,
            "issues": issues_df,
            "reasoning": reasoning_notes,
        }

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
            "Interest Expense_adj",
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
            column_name = f"{section} - {label}"
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
            "operating": _aggregate_sum(df.get("CFO_adj") if "CFO_adj" in df else df.get("CFO")),
            "investing": _aggregate_sum(df.get("CFI_adj") if "CFI_adj" in df else df.get("CFI")),
            "capex": _aggregate_sum(df.get("Capex")),
            "financing": _aggregate_sum(df.get("CFF_adj") if "CFF_adj" in df else df.get("CFF")),
        }

        if not any(series is not None for series in flows.values()):
            raise ValueError("No cash-flow data available in the schedule.")

        net_cash_series = _aggregate_sum(
            df.get("Net Cash Flow_adj") if "Net Cash Flow_adj" in df else df.get("Net Cash Flow")
        )
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
            "Closing Cash Balance_adj",
            "Cash and Cash Equivalents_adj",
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
            column_name = f"{section} - {label}"
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
            "Cash and Cash Equivalents_adj",
            "Closing Cash Balance_adj",
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
            "Current Assets": (
                df.get("Current Assets_adj")
                if "Current Assets_adj" in df
                else df.get("Current Assets")
            ),
            "Non-current Assets": (
                df.get("Non-current Assets_adj")
                if "Non-current Assets_adj" in df
                else df.get("Non-current Assets")
            ),
            "Current Liabilities": (
                df.get("Current Liabilities_adj")
                if "Current Liabilities_adj" in df
                else df.get("Current Liabilities")
            ),
            "Non-current Liabilities": (
                df.get("Non-current Liabilities_adj")
                if "Non-current Liabilities_adj" in df
                else df.get("Non-current Liabilities")
            ),
            "Equity": df.get("Equity_adj") if "Equity_adj" in df else df.get("Equity"),
            "Term Debt": df.get("Term Debt_adj") if "Term Debt_adj" in df else df.get("Term Debt"),
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
        _append("Equity and liabilities", "Term debt", out.get("Term Debt"))
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
            column_name = f"{section} - {label}"
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
            "Interest Expense_adj",
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
        return self._apply_financing_schedule(self.data.copy(), adjusted=False)


def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.dropna(how="all").dropna(axis=1, how="all")
    cleaned.columns = [str(col).strip() for col in cleaned.columns]
    return cleaned.reset_index(drop=True)
