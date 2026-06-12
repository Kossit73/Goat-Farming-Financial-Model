from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence

import pandas as pd

ANNUAL_SLAUGHTER_RATE_COLUMN = "Annual Slaughter Rate % of Herd"
LEGACY_SLAUGHTER_RATE_COLUMN = "Slaughter Rate % of Herd per Period"


def build_pricing_validation_messages(
    pricing_table: pd.DataFrame,
    production_drivers: Optional[pd.DataFrame],
    *,
    ensure_pricing_table: Callable[[pd.DataFrame], pd.DataFrame],
    ensure_production_driver_table: Callable[[Optional[pd.DataFrame]], pd.DataFrame],
    product_family_label: Callable[[Any], str],
    slaughter_products: Sequence[str],
) -> list[str]:
    """Validate pricing rows against current product and production-driver rules."""

    pricing = ensure_pricing_table(pricing_table)
    drivers = ensure_production_driver_table(production_drivers)
    messages: list[str] = []
    if pricing.empty:
        return messages

    inactive_qty = pricing.loc[
        ~pricing["Active"].fillna(False).astype(bool)
        & pd.to_numeric(pricing["Quantity per Period"], errors="coerce").fillna(0.0).gt(0.0)
    ]
    if not inactive_qty.empty:
        labels = inactive_qty[["Period", "Product"]].astype(str).agg(" / ".join, axis=1).tolist()
        messages.append(
            "Inactive products still carry quantities for: " + ", ".join(labels[:4]) + ("..." if len(labels) > 4 else "")
        )

    zero_price = pricing.loc[
        pricing["Active"].fillna(False).astype(bool)
        & pd.to_numeric(pricing["Base Price"], errors="coerce").fillna(0.0).le(0.0)
    ]
    if not zero_price.empty:
        labels = zero_price["Product"].astype(str).drop_duplicates().tolist()
        messages.append("Active products with zero or missing prices: " + ", ".join(labels))

    zero_qty = pricing.loc[
        pricing["Active"].fillna(False).astype(bool)
        & pd.to_numeric(pricing["Quantity per Period"], errors="coerce").fillna(0.0).le(0.0)
    ]
    if not zero_qty.empty:
        labels = zero_qty[["Period", "Product"]].astype(str).agg(" / ".join, axis=1).tolist()
        messages.append(
            "Active products with zero derived quantity: " + ", ".join(labels[:4]) + ("..." if len(labels) > 4 else "")
        )

    dairy = pricing.loc[
        pricing["Active"].fillna(False).astype(bool)
        & pricing["Product"].astype(str).map(product_family_label).eq("Dairy")
    ].copy()
    if not dairy.empty:
        dairy_totals = dairy.groupby("Period")["Allocation %"].sum(min_count=1)
        over_allocated = dairy_totals[dairy_totals > 100.0 + 1e-9]
        if not over_allocated.empty:
            period_labels = over_allocated.index.astype(str).tolist()
            messages.append(
                "Dairy allocation exceeds 100% for: " + ", ".join(period_labels[:4]) + ("..." if len(period_labels) > 4 else "")
            )

    driver_lookup = {
        str(row.get("Product", "")).strip(): dict(row)
        for _, row in drivers.iterrows()
        if str(row.get("Product", "")).strip()
    }
    cheese_alloc = pd.to_numeric(
        pd.Series([driver_lookup.get("Cheese", {}).get("Milk Allocation to Cheese %")]),
        errors="coerce",
    ).iloc[0]
    if pd.notna(cheese_alloc) and (cheese_alloc < 0 or cheese_alloc > 100):
        messages.append("Cheese driver `Milk Allocation to Cheese %` must stay between 0% and 100%.")

    live_herd_share = pd.to_numeric(
        pd.Series([driver_lookup.get("Live Herd", {}).get("Live Herd Sales Share %")]),
        errors="coerce",
    ).iloc[0]
    if pd.notna(live_herd_share) and (live_herd_share < 0 or live_herd_share > 100):
        messages.append("Live Herd driver `Live Herd Sales Share %` must stay between 0% and 100%.")

    active_livestock_products = [
        product
        for product in slaughter_products
        if pricing.loc[pricing["Product"].astype(str).str.strip() == product, "Active"].fillna(False).any()
    ]
    slaughter_rates = {
        product: pd.to_numeric(
            pd.Series(
                [
                    driver_lookup.get(product, {}).get(
                        ANNUAL_SLAUGHTER_RATE_COLUMN,
                        driver_lookup.get(product, {}).get(
                            LEGACY_SLAUGHTER_RATE_COLUMN
                        ),
                    )
                ]
            ),
            errors="coerce",
        ).iloc[0]
        for product in active_livestock_products
    }
    valid_rates = {product: rate for product, rate in slaughter_rates.items() if pd.notna(rate)}
    if len({round(float(rate), 8) for rate in valid_rates.values()}) > 1:
        messages.append(
            "Livestock products use different slaughter rates. Align Meat, Offal, Pelt, and Live Herd if they share the same saleable herd stream."
        )

    return messages


def sync_commercial_assumptions_to_core(
    assumptions: Optional[Dict[str, pd.DataFrame]],
    core: pd.DataFrame,
    *,
    ensure_business_configuration_table: Callable[[Optional[pd.DataFrame]], pd.DataFrame],
    selected_business_type: Callable[[pd.DataFrame], Any],
    active_products_for_business_type: Callable[[Any], list[str]],
    build_app_assumption_bundle: Callable[[Dict[str, pd.DataFrame]], Any],
    sync_production_driver_table_to_products: Callable[[Optional[pd.DataFrame], Sequence[str]], pd.DataFrame],
    sync_scenario_controls_to_products: Callable[[Optional[pd.DataFrame], Sequence[str]], pd.DataFrame],
    sync_pricing_table_to_core: Callable[[pd.DataFrame, pd.DataFrame, Any], pd.DataFrame],
    derive_pricing_quantities_from_production: Callable[[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame],
    pricing_schedule_context: Callable[[pd.DataFrame, Optional[pd.DataFrame], Optional[Dict[str, pd.DataFrame]]], pd.DataFrame],
) -> Dict[str, pd.DataFrame]:
    """Align commercial assumptions to the active core schedule without changing model rules."""

    synced: Dict[str, pd.DataFrame] = dict(assumptions or {})
    business_config = ensure_business_configuration_table(synced.get("Business Configuration"))
    business_type = selected_business_type(business_config)
    active_products = active_products_for_business_type(business_type)
    bundle = build_app_assumption_bundle(synced)
    synced["Business Configuration"] = business_config
    synced["Biological System Settings"] = bundle.biological.system_settings
    synced["Breeding & Reproduction Biology"] = bundle.biological.breeding_reproduction
    synced["Lactation Biology"] = bundle.biological.lactation
    synced["Finishing & Slaughter Biology"] = bundle.biological.finishing_slaughter
    synced["Opening Herd Cohorts"] = bundle.biological.opening_herd_cohorts
    synced["Cohort Allocation Rules"] = bundle.biological.cohort_allocation_rules
    synced["Biological Cost Drivers"] = bundle.biological.biological_cost_drivers
    synced["Production Drivers"] = sync_production_driver_table_to_products(
        synced.get("Production Drivers"),
        active_products,
    )
    synced["Scenario Controls"] = sync_scenario_controls_to_products(
        synced.get("Scenario Controls"),
        active_products,
    )
    pricing_table = synced.get("Pricing")
    pricing_df = pricing_table if isinstance(pricing_table, pd.DataFrame) else pd.DataFrame()
    synced_pricing = sync_pricing_table_to_core(pricing_df, core, business_type)
    synced["Pricing"] = derive_pricing_quantities_from_production(
        synced_pricing,
        pricing_schedule_context(core, synced.get("Herd Plan"), synced),
        synced.get("Production Drivers"),
    )
    return synced
