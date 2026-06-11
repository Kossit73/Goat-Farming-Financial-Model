from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = pd.to_numeric(value, errors="coerce")
    except Exception:
        return float(default)
    if isinstance(numeric, (pd.Series, pd.Index, np.ndarray, list, tuple)):
        array = np.asarray(numeric, dtype="float64").reshape(-1)
        numeric = array[0] if array.size else np.nan
    return float(numeric) if pd.notna(numeric) else float(default)


@dataclass(frozen=True)
class PricingContext:
    schedule_df: pd.DataFrame
    driver_lookup: dict[str, dict[str, Any]]
    slaughter_products: tuple[str, ...]
    period_days_map: dict[str, float]
    herd_map: dict[str, float]
    biological_maps: dict[str, dict[str, float]]
    base_year: int


def period_days_from_index(index: pd.DatetimeIndex) -> pd.Series:
    if index.empty:
        return pd.Series(dtype=float)
    deltas = index.to_series().diff().dt.days.astype(float)
    valid_days = deltas.iloc[1:][np.isfinite(deltas.iloc[1:])]
    default_days = float(np.median(valid_days)) if not valid_days.empty else 30.44
    if not np.isfinite(default_days) or default_days <= 0:
        default_days = 30.44
    return deltas.fillna(default_days).clip(lower=1.0)


def annual_pct_to_period_rate(percent: float, period_days: float) -> float:
    """Convert an annualized rate into the active schedule period rate."""

    annual_rate = max(0.0, min(float(percent) / 100.0, 1.0))
    months_factor = max(float(period_days) / 30.44, 1e-6)
    return 1.0 - ((1.0 - annual_rate) ** (months_factor / 12.0))


def build_pricing_context(
    schedule_df: pd.DataFrame,
    driver_lookup: dict[str, dict[str, Any]],
    slaughter_products: tuple[str, ...],
) -> PricingContext:
    schedule_index = pd.to_datetime(schedule_df.index, errors="coerce")
    period_days_map = dict(zip(schedule_index.strftime("%Y-%m-%d"), period_days_from_index(schedule_index).tolist()))

    def _schedule_numeric_map(column: str) -> dict[str, float]:
        series = schedule_df.get(column)
        if series is None:
            values = pd.Series(np.nan, index=schedule_df.index, dtype=float)
        else:
            values = pd.to_numeric(series, errors="coerce")
        return {
            idx.strftime("%Y-%m-%d"): float(value)
            for idx, value in zip(schedule_index, values)
            if pd.notna(idx) and pd.notna(value)
        }

    herd_series = schedule_df.get("Herd Size (heads)")
    if herd_series is None:
        herd_series = pd.Series(np.nan, index=schedule_df.index, dtype=float)
    else:
        herd_series = pd.to_numeric(herd_series, errors="coerce")
    herd_map = {
        idx.strftime("%Y-%m-%d"): float(value)
        for idx, value in zip(schedule_index, herd_series)
        if pd.notna(idx) and pd.notna(value)
    }

    biological_maps = {
        "milk": _schedule_numeric_map("Milk Production (L)"),
        "live_herd": _schedule_numeric_map("Live Herd Sales (heads)"),
        "slaughter": _schedule_numeric_map("Slaughter Heads"),
        "meat": _schedule_numeric_map("Meat Output Kg"),
        "offal": _schedule_numeric_map("Offal Output Kg"),
        "pelt": _schedule_numeric_map("Pelt Output Units"),
    }
    base_year = int(schedule_index.min().year) if len(schedule_index) else pd.Timestamp.today().year
    return PricingContext(
        schedule_df=schedule_df,
        driver_lookup=driver_lookup,
        slaughter_products=slaughter_products,
        period_days_map=period_days_map,
        herd_map=herd_map,
        biological_maps=biological_maps,
        base_year=base_year,
    )


def derive_pricing_quantities(
    pricing_table: pd.DataFrame,
    context: PricingContext,
) -> pd.DataFrame:
    work = pricing_table.copy()
    work["Period_dt"] = pd.to_datetime(work["Period"], errors="coerce")
    milk_driver = context.driver_lookup.get("Milk", {})
    cheese_driver = context.driver_lookup.get("Cheese", {})
    livestock_driver = next(
        (context.driver_lookup.get(product, {}) for product in context.slaughter_products if context.driver_lookup.get(product)),
        {},
    )
    live_herd_driver = context.driver_lookup.get("Live Herd", livestock_driver)
    milk_growth = 1 + (_coerce_float(milk_driver.get("Driver Growth %"), 0.0) / 100.0)
    cheese_growth = 1 + (_coerce_float(cheese_driver.get("Driver Growth %"), 0.0) / 100.0)
    lactating_share = _coerce_float(milk_driver.get("Lactating Herd Share %"), 0.0) / 100.0
    litres_per_day = _coerce_float(milk_driver.get("Litres per Lactating Doe per Day"), 0.0)
    cheese_allocation = _coerce_float(cheese_driver.get("Milk Allocation to Cheese %"), 0.0) / 100.0
    base_cheese_yield = _coerce_float(cheese_driver.get("Cheese Yield Kg per Litre"), 0.0)
    slaughter_growth = 1 + (_coerce_float(livestock_driver.get("Driver Growth %"), 0.0) / 100.0)
    annual_slaughter_pct = _coerce_float(livestock_driver.get("Slaughter Rate % of Herd per Period"), 0.0)
    live_herd_share = _coerce_float(live_herd_driver.get("Live Herd Sales Share %"), 0.0) / 100.0

    for period, period_group in work.groupby("Period", sort=False):
        herd_size = context.herd_map.get(period, 0.0)
        period_days = context.period_days_map.get(period, 30.44)
        period_dt = pd.to_datetime(period, errors="coerce")
        year_offset = max(0, int(period_dt.year) - context.base_year) if pd.notna(period_dt) else 0

        milk_output = herd_size * lactating_share * litres_per_day * period_days * (milk_growth ** year_offset)
        if period in context.biological_maps["milk"]:
            milk_output = float(context.biological_maps["milk"][period])

        cheese_yield = base_cheese_yield * (cheese_growth ** year_offset)
        cheese_active = bool(
            period_group.loc[period_group["Product"].astype(str).str.strip() == "Cheese", "Active"].fillna(False).any()
        )
        milk_available_for_sale = milk_output * (1.0 - cheese_allocation if cheese_active else 1.0)
        cheese_milk_input = milk_output * cheese_allocation if cheese_active else 0.0

        slaughter_rate = annual_pct_to_period_rate(annual_slaughter_pct, period_days)
        saleable_goats = herd_size * slaughter_rate * (slaughter_growth ** year_offset)
        live_herd_units = saleable_goats * live_herd_share
        goats_for_slaughter = max(0.0, saleable_goats - live_herd_units)
        if period in context.biological_maps["live_herd"] or period in context.biological_maps["slaughter"]:
            live_herd_units = max(0.0, float(context.biological_maps["live_herd"].get(period, 0.0)))
            goats_for_slaughter = max(0.0, float(context.biological_maps["slaughter"].get(period, 0.0)))

        for idx, row in period_group.iterrows():
            quantity_mode = str(row.get("Quantity Mode", "Derived")).strip()
            if quantity_mode == "Manual Override":
                manual_qty = _coerce_float(row.get("Manual Quantity Override"), np.nan)
                work.at[idx, "Quantity per Period"] = 0.0 if pd.isna(manual_qty) else float(manual_qty)
                continue
            active = bool(row.get("Active", False))
            if not active:
                work.at[idx, "Quantity per Period"] = 0.0
                continue

            product = str(row.get("Product", "")).strip()
            quantity = calculate_product_quantity(
                product,
                row,
                period,
                milk_available_for_sale,
                cheese_milk_input,
                cheese_yield,
                live_herd_units,
                goats_for_slaughter,
                context,
            )
            work.at[idx, "Quantity per Period"] = max(0.0, quantity)

    return work.drop(columns="Period_dt")


def calculate_product_quantity(
    product: str,
    row: pd.Series,
    period: str,
    milk_available_for_sale: float,
    cheese_milk_input: float,
    cheese_yield: float,
    live_herd_units: float,
    goats_for_slaughter: float,
    context: PricingContext,
) -> float:
    if product == "Milk":
        return milk_available_for_sale
    if product == "Cheese":
        return cheese_milk_input * cheese_yield
    if product == "Live Herd":
        return live_herd_units
    if product == "Meat" and period in context.biological_maps["meat"]:
        return float(context.biological_maps["meat"].get(period, 0.0))
    if product == "Offal" and period in context.biological_maps["offal"]:
        return float(context.biological_maps["offal"].get(period, 0.0))
    if product == "Pelt" and period in context.biological_maps["pelt"]:
        return float(context.biological_maps["pelt"].get(period, 0.0))

    driver = context.driver_lookup.get(product, {})
    if product == "Meat":
        yield_value = _coerce_float(driver.get("Meat Yield Kg per Goat"), 0.0)
        return goats_for_slaughter * yield_value
    if product == "Offal":
        yield_value = _coerce_float(driver.get("Offal Yield Kg per Goat"), 0.0)
        return goats_for_slaughter * yield_value
    if product == "Pelt":
        yield_value = _coerce_float(driver.get("Pelt Units per Goat"), 0.0)
        return goats_for_slaughter * yield_value
    return 0.0
