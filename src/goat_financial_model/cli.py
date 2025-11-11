"""Command line entry-points for working with :class:`GoatModel`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .goat_model import GoatModel, InputSchedule


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a tidy financial model from a manual input schedule.",
    )
    parser.add_argument(
        "schedule",
        type=Path,
        help=(
            "CSV file containing a 'Period' column plus financial metrics such as "
            "Revenue, COGS, EBITDA, etc."
        ),
    )
    parser.add_argument(
        "--period-column",
        default="Period",
        help="Column name that contains the period labels (default: Period).",
    )
    parser.add_argument(
        "--wacc",
        type=float,
        help="Weighted average cost of capital in percent (e.g. 12 for 12%%).",
    )
    parser.add_argument(
        "--npv",
        type=float,
        help="Net present value of the plan (currency units).",
    )
    parser.add_argument(
        "--terminal-value",
        type=float,
        help="Terminal value used in the valuation (currency units).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to write the tidy data to (CSV or Parquet based on extension).",
    )
    return parser


def _write_output(df: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".parquet":
        df.to_parquet(output)
    else:
        df.to_csv(output, index=True)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    frame = pd.read_csv(args.schedule)
    valuation_inputs = {}
    if args.wacc is not None:
        valuation_inputs["WACC"] = args.wacc / 100.0
    if args.npv is not None:
        valuation_inputs["NPV"] = args.npv
    if args.terminal_value is not None:
        valuation_inputs["Terminal Value"] = args.terminal_value

    schedule = InputSchedule.from_frame(
        frame,
        period_col=args.period_column,
        valuation_inputs=valuation_inputs,
    )
    model = schedule.to_model()
    tidy = model.to_tidy()

    print("=== Timeline ===")
    print(tidy.index.min(), "to", tidy.index.max())
    print()

    print("=== Key Metrics ===")
    valuation_bits = {
        "WACC": model.wacc(),
        "NPV": model.npv(),
        "Terminal Value": model.terminal_value(),
    }
    for key, value in valuation_bits.items():
        if value is not None:
            if key == "WACC":
                print(f"{key}: {value * 100:.2f}%")
            else:
                print(f"{key}: {value:,.2f}")
    print()

    print("=== Financial Series (first 5 rows) ===")
    print(tidy.head())

    if args.output:
        _write_output(tidy, args.output)
        print()
        print(f"Saved tidy data to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover - direct invocation
    raise SystemExit(main())
