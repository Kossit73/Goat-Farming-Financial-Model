"""Command line entry-points for working with :class:`GoatModel`."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import pandas as pd

from .goat_model import GoatModel


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a tidy financial model from the goat farming workbook.",
    )
    parser.add_argument(
        "excel_path",
        type=Path,
        help="Path to the Excel workbook containing the goat financial model.",
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

    model = GoatModel(str(args.excel_path))
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
