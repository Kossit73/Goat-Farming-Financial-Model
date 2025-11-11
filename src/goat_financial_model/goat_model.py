"""Utilities for reading and manipulating the goat farming financial model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

SeriesLabels = Union[str, Sequence[str]]


@dataclass
class GoatModel:
    """Helper for extracting time series and schedules from the Excel template."""

    excel_path: str
    ref_sheet: str = "IS"
    _df_is: Optional[pd.DataFrame] = None
    _df_cf: Optional[pd.DataFrame] = None
    _df_val: Optional[pd.DataFrame] = None
    _dates: Optional[pd.DatetimeIndex] = None
    _workbook: Optional[pd.ExcelFile] = None
    _sheet_cache: Dict[str, pd.DataFrame] = field(default_factory=dict)

    _SHEET_ALIASES: ClassVar[Dict[str, Sequence[str]]] = {
        "IS": ("Income Statement", "P&L", "Profit and Loss"),
        "CF": ("Cash Flow", "Cashflow", "Cash Flow Statement"),
        "Valuation": ("Valuation Summary", "Valuation Model"),
        "Capex": ("Capex", "CAPEX", "Capital Expenditure", "Capital Expenditures"),
        "Asset Schedules": (
            "Asset Schedules",
            "Asset Schedule",
            "Assets Schedule",
            "Asset Register",
        ),
        "Outputs": ("Outputs", "Output Summary", "Model Outputs"),
        "Benchmark KPI": (
            "Benchmark KPI",
            "Benchmark KPIs",
            "Capita Benchmark KPI",
            "Capita- Benchmark KPI",
            "Benchmark KPI's",
        ),
    }

    # ---------- Internal loading & helpers ----------
    def _load(self) -> None:
        if self._df_is is None:
            self._df_is = self._require_sheet("IS")
        if self._df_cf is None:
            self._df_cf = self._require_sheet("CF")
        if self._df_val is None:
            self._df_val = self._require_sheet("Valuation")

    def _ensure_workbook(self) -> None:
        if self._workbook is None:
            self._workbook = pd.ExcelFile(self.excel_path)

    @staticmethod
    def _normalise_sheet_name(name: str) -> str:
        return "".join(ch for ch in name.lower() if ch.isalnum())

    def _load_sheet(self, canonical: str) -> Optional[pd.DataFrame]:
        self._ensure_workbook()
        assert self._workbook is not None

        aliases = self._SHEET_ALIASES.get(canonical, ())
        candidates: Iterable[str] = (canonical,) + tuple(aliases)
        available = {
            self._normalise_sheet_name(sheet_name): sheet_name
            for sheet_name in self._workbook.sheet_names
        }

        for candidate in candidates:
            key = self._normalise_sheet_name(candidate)
            sheet_name = available.get(key)
            if sheet_name:
                if sheet_name not in self._sheet_cache:
                    self._sheet_cache[sheet_name] = self._workbook.parse(
                        sheet_name, header=None
                    )
                return self._sheet_cache[sheet_name]
        return None

    def _require_sheet(self, canonical: str) -> pd.DataFrame:
        sheet = self._load_sheet(canonical)
        if sheet is None:
            raise ValueError(
                f"Workbook is missing the expected '{canonical}' sheet or a known alias."
            )
        return sheet

    @property
    def dates(self) -> pd.DatetimeIndex:
        """Monthly period end dates inferred from the reference sheet's header."""

        self._load()
        if self._dates is None:
            hdr_row = 6
            date_cells = pd.to_datetime(
                self._df_is.iloc[hdr_row, 8:], errors="coerce"
            ).dropna()
            self._dates = pd.DatetimeIndex(date_cells.values)
        return self._dates

    def _coerce_labels(self, labels: SeriesLabels) -> Sequence[str]:
        if isinstance(labels, str):
            return (labels,)
        return tuple(labels)

    def _extract_series(
        self,
        df: Optional[pd.DataFrame],
        labels: SeriesLabels,
        search_cols: int = 6,
    ) -> Optional[pd.Series]:
        """Return numeric series across timeline columns by matching label variants."""

        if df is None:
            return None

        matched_label: Optional[str] = None
        for label in self._coerce_labels(labels):
            idxs = df.index[(df == label).any(axis=1)].tolist()
            if not idxs:
                idxs = [
                    i
                    for i in range(len(df))
                    if any(
                        isinstance(val, str)
                        and val.strip().lower().startswith(label.lower())
                        for val in df.iloc[i, :search_cols].tolist()
                    )
                ]
            if idxs:
                row = df.loc[idxs[0]]
                matched_label = label
                break
        else:
            return None

        n_dates = len(self.dates)
        values = pd.to_numeric(row.iloc[8 : 8 + n_dates], errors="coerce")
        return pd.Series(values.values, index=self.dates, name=matched_label)

    def _extract_named_table(
        self,
        df: Optional[pd.DataFrame],
        anchors: Sequence[str],
        max_rows: int = 100,
        search_cols: int = 6,
    ) -> Optional[pd.DataFrame]:
        """Extract a table that appears below a title row matching one of `anchors`."""

        if df is None:
            return None

        start_row: Optional[int] = None
        for anchor in anchors:
            for i in range(len(df)):
                row = df.iloc[i, :search_cols]
                if any(
                    isinstance(val, str) and anchor.lower() in val.lower()
                    for val in row.tolist()
                ):
                    start_row = i
                    break
            if start_row is not None:
                break

        if start_row is None:
            return None

        header_row = start_row + 1
        while header_row < len(df) and df.iloc[header_row].isna().all():
            header_row += 1
        if header_row >= len(df):
            return None

        header_values = df.iloc[header_row].tolist()
        header = [
            str(val).strip() if pd.notna(val) else f"Column {idx+1}"
            for idx, val in enumerate(header_values)
        ]

        records = []
        for row_idx in range(header_row + 1, min(len(df), header_row + 1 + max_rows)):
            row = df.iloc[row_idx]
            values = row.iloc[: len(header)]
            if values.isna().all():
                break
            records.append(values.tolist())

        if not records:
            return None

        table = pd.DataFrame(records, columns=header)
        table = table.dropna(axis=1, how="all")
        table = table.apply(pd.to_numeric, errors="ignore")
        return table.reset_index(drop=True)

    @staticmethod
    def _clean_table(df: pd.DataFrame) -> pd.DataFrame:
        """Drop empty rows/columns and reset the index for easier consumption."""

        cleaned = df.dropna(how="all").dropna(axis=1, how="all")
        cleaned.columns = [str(col).strip() for col in cleaned.columns]
        return cleaned.reset_index(drop=True)

    # ---------- Base series ----------
    def revenue(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            ("Revenue", "Total Revenue", "Sales Revenue", "Total Sales"),
        )

    def cogs(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            (
                "COGS",
                "Cost of Goods Sold",
                "Cost of Sales",
                "Total COGS",
                "Total Cost of Sales",
            ),
        )

    def variable_expenses(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            (
                "Variable Expenses",
                "Variable Operating Expenses",
                "Variable Costs",
                "Variable Overheads",
            ),
        )

    def fixed_expenses(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            (
                "Fixed Expenses",
                "Fixed Operating Expenses",
                "Fixed Costs",
                "Fixed Overheads",
            ),
        )

    def direct_wages(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            (
                "Direct Wages",
                "Direct Labour",
                "Direct Labor",
                "Production Wages",
            ),
        )

    def admin_wages(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            (
                "Admin Wages",
                "Administrative Wages",
                "Administration Salaries",
                "Admin Salaries",
            ),
        )

    def gross_margin(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            ("GROSS MARGIN", "Gross Profit", "Gross Margin"),
        )

    def ebitda(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            ("EBITDA", "EBITDA (Operating)", "Operating EBITDA"),
        )

    def depreciation(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            (
                "Total Depreciation & Amortization",
                "Depreciation & Amortization",
                "Depreciation",
                "Depreciation and Amortisation",
            ),
        )

    def ebit(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            ("EBIT", "Operating Profit", "EBIT (Operating)"),
        )

    def npbt(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            (
                "Net Profit Before Tax",
                "Profit Before Tax",
                "Net Profit Before Income Tax",
            ),
        )

    def npat(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_is,
            ("Net Profit After Tax", "Profit After Tax", "Net Income"),
        )

    def cfo(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_cf,
            (
                "Net Cash Flow from Operating Activities",
                "Net Cash Provided by Operating Activities",
                "Operating Cash Flow",
            ),
        )

    def cfi(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_cf,
            (
                "Net Cash Flow from Investing Activities",
                "Net Cash Used in Investing Activities",
                "Investing Cash Flow",
            ),
        )

    def cff(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_cf,
            (
                "Net Cash Flow from Financing Activities",
                "Net Cash Provided by Financing Activities",
                "Financing Cash Flow",
            ),
        )

    def capex(self) -> Optional[pd.Series]:
        self._load()
        return self._extract_series(
            self._df_cf,
            (
                "Capital Expenditure",
                "Capital Expenditures",
                "Purchase of Property & Equipment",
                "Purchase of Property, Plant & Equipment",
                "Capex",
            ),
        )

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
        idxs = self._df_val.index[
            (self._df_val == "Unlevered Free Cash Flow").any(axis=1)
        ].tolist()
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
        idxs = self._df_val.index[
            (self._df_val == "NPV based on year 5").any(axis=1)
        ].tolist()
        if not idxs:
            return None
        row = self._df_val.loc[idxs[0]]
        nums = pd.to_numeric(row, errors="coerce").dropna()
        return float(nums.iloc[-1]) if len(nums) else None

    # ---------- Supplementary schedules ----------
    def capitalisation_table(self) -> Optional[pd.DataFrame]:
        """Return the capitalisation table if present on the valuation sheet."""

        self._load()
        return self._extract_named_table(
            self._df_val,
            (
                "Capitalisation Table",
                "Capitalization Table",
                "Cap Table",
                "Capitalisation",
            ),
        )

    def capex_schedule(self) -> Optional[pd.DataFrame]:
        """Return the dedicated Capex schedule if a specific sheet exists."""

        sheet = self._load_sheet("Capex")
        if sheet is None:
            return None
        return self._extract_named_table(
            sheet,
            ("Capex", "Capital Expenditure", "Capital Expenditure Schedule"),
            search_cols=sheet.shape[1],
        ) or self._clean_table(sheet)

    def asset_schedules(self) -> Optional[pd.DataFrame]:
        """Return the asset schedule sheet cleaned of empty rows and columns."""

        sheet = self._load_sheet("Asset Schedules")
        if sheet is None:
            return None
        return self._clean_table(sheet)

    def outputs(self) -> Optional[pd.DataFrame]:
        """Return the model outputs sheet with empty structure removed."""

        sheet = self._load_sheet("Outputs")
        if sheet is None:
            return None
        return self._clean_table(sheet)

    def benchmark_kpis(self) -> Optional[pd.DataFrame]:
        """Return the benchmark KPI sheet, often titled 'Capita Benchmark KPI'."""

        sheet = self._load_sheet("Benchmark KPI")
        if sheet is None:
            return None
        return self._clean_table(sheet)

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
            "Variable Expenses": self.variable_expenses(),
            "Fixed Expenses": self.fixed_expenses(),
            "Direct Wages": self.direct_wages(),
            "Admin Wages": self.admin_wages(),
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
            "Variable Expenses": self.variable_expenses(),
            "Fixed Expenses": self.fixed_expenses(),
            "Direct Wages": self.direct_wages(),
            "Admin Wages": self.admin_wages(),
            "Gross Margin": self.gross_margin(),
            "EBITDA": self.ebitda(),
            "Depreciation & Amortization": self.depreciation(),
            "EBIT": self.ebit(),
            "NPBT": self.npbt(),
            "NPAT": self.npat(),
            "Capex": self.capex(),
            "CFO": self.cfo(),
            "CFI": self.cfi(),
            "CFF": self.cff(),
            "Net Cash Flow": self.net_cash_flow(),
        }
        valid = {k: v for k, v in series.items() if v is not None}
        if not valid:
            raise ValueError("No financial series could be extracted from the workbook.")
        return pd.concat(valid, axis=1)
