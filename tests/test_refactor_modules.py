from __future__ import annotations

import pandas as pd
import pytest

from goat_financial_model.assumption_bundle import build_assumption_bundle
from goat_financial_model.pricing_engine import build_pricing_context, derive_pricing_quantities
from goat_financial_model.table_registry import ColumnSchema, TableSchema, build_default_table, ensure_table


def test_table_registry_restores_missing_columns_and_defaults() -> None:
    schema = TableSchema(
        name="Demo",
        columns=(
            ColumnSchema("Name", ""),
            ColumnSchema("Value", 0.0),
        ),
        default_rows=({"Name": "Base", "Value": 1.0},),
    )

    restored = ensure_table(schema, pd.DataFrame({"Name": ["Custom"]}))

    assert restored.columns.tolist() == ["Name", "Value"]
    assert restored.iloc[0]["Name"] == "Custom"
    assert restored.iloc[0]["Value"] == 0.0
    assert build_default_table(schema).iloc[0]["Name"] == "Base"


def test_assumption_bundle_groups_sections() -> None:
    ensure_map = {
        "Biological System Settings": lambda table: table if table is not None else pd.DataFrame({"Setting": [], "Value": []}),
        "Pricing": lambda table: table if table is not None else pd.DataFrame(),
        "Production Drivers": lambda table: table if table is not None else pd.DataFrame(),
        "Scenario Controls": lambda table: table if table is not None else pd.DataFrame(),
        "Capital & Financing": lambda table: table if table is not None else pd.DataFrame(),
        "Loan Facilities": lambda table: table if table is not None else pd.DataFrame(),
        "Equity Facilities": lambda table: table if table is not None else pd.DataFrame(),
        "Valuation Inputs": lambda table: table if table is not None else pd.DataFrame(),
        "Breeding & Reproduction Biology": lambda table: table if table is not None else pd.DataFrame(),
        "Lactation Biology": lambda table: table if table is not None else pd.DataFrame(),
        "Finishing & Slaughter Biology": lambda table: table if table is not None else pd.DataFrame(),
        "Opening Herd Cohorts": lambda table: table if table is not None else pd.DataFrame(),
        "Cohort Allocation Rules": lambda table: table if table is not None else pd.DataFrame(),
        "Biological Cost Drivers": lambda table: table if table is not None else pd.DataFrame(),
    }
    assumptions = {
        "Biological System Settings": pd.DataFrame({"Setting": ["Model Grain"], "Value": ["monthly"]}),
        "Pricing": pd.DataFrame({"Product": ["Milk"]}),
    }

    bundle = build_assumption_bundle(assumptions, ensure_map)

    assert bundle.biological.system_settings.iloc[0]["Setting"] == "Model Grain"
    assert bundle.commercial.pricing.iloc[0]["Product"] == "Milk"
    assert bundle.get("Pricing").equals(bundle.commercial.pricing)


def test_pricing_engine_uses_biological_quantities_when_available() -> None:
    pricing = pd.DataFrame(
        {
            "Period": ["2024-01-31", "2024-01-31"],
            "Product": ["Milk", "Meat"],
            "Active": [True, True],
            "Allocation %": [100.0, 100.0],
            "Quantity Mode": ["Derived", "Derived"],
            "Manual Quantity Override": [None, None],
            "Quantity per Period": [0.0, 0.0],
            "Unit": ["L", "Kg"],
            "Base Price": [1.0, 1.0],
            "Price Growth %": [0.0, 0.0],
        }
    )
    schedule = pd.DataFrame(
        {
            "Herd Size (heads)": [100.0],
            "Milk Production (L)": [250.0],
            "Meat Output Kg": [90.0],
            "Slaughter Heads": [5.0],
            "Live Herd Sales (heads)": [0.0],
        },
        index=pd.DatetimeIndex(["2024-01-31"]),
    )
    driver_lookup = {
        "Milk": {
            "Lactating Herd Share %": 20.0,
            "Litres per Lactating Doe per Day": 1.0,
            "Driver Growth %": 0.0,
        },
        "Meat": {
            "Meat Yield Kg per Goat": 10.0,
            "Driver Growth %": 0.0,
            "Slaughter Rate % of Herd per Period": 5.0,
        },
    }

    context = build_pricing_context(schedule, driver_lookup, ("Meat", "Offal", "Pelt", "Live Herd"))
    derived = derive_pricing_quantities(pricing, context)

    milk_qty = derived.loc[derived["Product"] == "Milk", "Quantity per Period"].iloc[0]
    meat_qty = derived.loc[derived["Product"] == "Meat", "Quantity per Period"].iloc[0]
    assert milk_qty == pytest.approx(250.0)
    assert meat_qty == pytest.approx(90.0)
