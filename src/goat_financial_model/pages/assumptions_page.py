from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd
import streamlit as st

from goat_financial_model.editor_registry import build_remove_options


EnsureFn = Callable[[Optional[pd.DataFrame]], pd.DataFrame]
RenderEditor = Callable[[str, pd.DataFrame, Callable[[pd.DataFrame], None]], None]


@dataclass(frozen=True)
class BiologicalEditorDefinition:
    name: str
    caption: str
    ensure_fn: EnsureFn
    editor_key: str


def _apply_product_base_price(
    pricing_table: pd.DataFrame,
    product: str,
    base_price: float,
) -> pd.DataFrame:
    updated = pricing_table.copy()
    if "Product" not in updated.columns or "Base Price" not in updated.columns:
        return updated

    product_key = str(product).strip()
    if not product_key:
        return updated

    product_mask = updated["Product"].astype(str).str.strip() == product_key
    if product_mask.any():
        updated.loc[product_mask, "Base Price"] = float(base_price)
    return updated


def render_biological_assumption_editor(
    *,
    definitions: list[BiologicalEditorDefinition],
    assumptions: dict[str, pd.DataFrame],
    render_row_editor: RenderEditor,
) -> dict[str, pd.DataFrame]:
    for definition in definitions:
        assumptions[definition.name] = definition.ensure_fn(assumptions.get(definition.name))

    selected_name = st.selectbox(
        "Biological assumption table",
        options=[definition.name for definition in definitions],
        key="biological_editor_name",
    )
    selected = next(definition for definition in definitions if definition.name == selected_name)
    st.markdown(f"#### {selected.name}")
    st.caption(selected.caption)
    render_row_editor(
        selected.editor_key,
        assumptions[selected.name],
        lambda updated, assumption_name=selected.name, ensure_fn=selected.ensure_fn: assumptions.__setitem__(
            assumption_name,
            ensure_fn(updated),
        ),
    )
    return assumptions


def render_herd_plan_editor(
    *,
    herd_plan: pd.DataFrame,
    save_table: Callable[[pd.DataFrame], None],
    render_row_editor: RenderEditor,
    clear_editor_state: Callable[[str], None],
    apply_increment_fn: Callable[[pd.DataFrame, float], pd.DataFrame],
    ensure_fn: Callable[[pd.DataFrame], pd.DataFrame],
) -> pd.DataFrame:
    st.caption(
        "Set herd size by year and optional growth %. Revenue and key variable costs are scaled from the baseline herd level."
    )
    st.session_state.setdefault("herd_yearly_increment_percent", 0.0)
    herd_inc_col, herd_inc_btn_col = st.columns([2, 1])
    herd_inc_col.number_input(
        "Yearly Increment (%)",
        min_value=-100.0,
        max_value=300.0,
        step=0.1,
        key="herd_yearly_increment_percent",
    )
    if herd_inc_btn_col.button("Apply Increment Across Years", key="apply_herd_yearly_increment"):
        herd_plan = apply_increment_fn(
            herd_plan,
            st.session_state.get("herd_yearly_increment_percent", 0.0),
        )
        save_table(herd_plan)
        clear_editor_state("assump::herd_plan")

    herd_add_col, herd_remove_select_col, herd_remove_btn_col = st.columns([1, 2, 1])
    if herd_add_col.button("Add Herd Year", key="herd_plan_add_row"):
        years = pd.to_numeric(herd_plan.get("Year"), errors="coerce")
        next_year = int(years.dropna().max() + 1) if years.notna().any() else pd.Timestamp.today().year
        herd_plan = pd.concat(
            [
                herd_plan,
                pd.DataFrame(
                    {
                        "Year": [next_year],
                        "Herd Size (heads)": [pd.NA],
                        "Herd Growth %": [pd.NA],
                    }
                ),
            ],
            ignore_index=True,
        )
        save_table(herd_plan)
        clear_editor_state("assump::herd_plan")

    labels, label_index = build_remove_options(
        herd_plan,
        lambda row: str(int(row["Year"])) if pd.notna(row.get("Year")) else f"Row {row.name + 1}",
    )
    herd_remove_select_col.selectbox(
        "Remove year",
        options=["-- Select Year --"] + labels,
        key="herd_plan_remove_choice",
    )
    if herd_remove_btn_col.button("Remove", key="herd_plan_remove_row"):
        choice = st.session_state.get("herd_plan_remove_choice")
        if choice in label_index:
            herd_plan = herd_plan.drop(index=label_index[choice]).reset_index(drop=True)
            save_table(ensure_fn(herd_plan))
            st.session_state.herd_plan_remove_choice = "-- Select Year --"
            clear_editor_state("assump::herd_plan")

    render_row_editor(
        "assump::herd_plan",
        herd_plan,
        lambda updated: save_table(ensure_fn(updated)),
    )
    return herd_plan


def render_operating_cost_editor(
    *,
    operating_table: pd.DataFrame,
    save_table: Callable[[pd.DataFrame], None],
    render_row_editor: RenderEditor,
    clear_editor_state: Callable[[str], None],
    add_row_fn: Callable[[pd.DataFrame], pd.DataFrame],
    remove_row_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
    apply_increment_fn: Callable[[pd.DataFrame, float, Optional[str], str], pd.DataFrame],
    ensure_fn: Callable[[pd.DataFrame], pd.DataFrame],
    get_default_frame: Callable[[], pd.DataFrame],
    save_defaults_fn: Callable[[pd.DataFrame], None],
    apply_defaults_fn: Callable[[pd.DataFrame], pd.DataFrame],
    restore_defaults_fn: Callable[[], pd.DataFrame],
) -> pd.DataFrame:
    st.caption(
        "Fields `variable_feed_cost_per_herd`, `variable_healthcare_cost_per_herd`, and "
        "`fixed_utility_cost_per_herd` are treated as unit_cost_per_head_per_month values. "
        "Monthly total cost = unit cost x herd heads x months."
    )
    st.session_state.setdefault("operating_remove_choice", "-- Select Item --")
    st.session_state.setdefault("operating_increment_target", "All categories")
    st.session_state.setdefault("operating_increment_column", "unit_cost_per_head_per_month")
    st.session_state.setdefault("operating_increment_pct", 0.0)

    add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])
    if add_col.button("Add Item", key="operating_add_row"):
        save_table(add_row_fn(operating_table))
        clear_editor_state("assump::operating_costs")

    labels, label_index = build_remove_options(
        operating_table,
        lambda row: (
            f"{str(row.get('Category', '')).strip() or f'Item {row.name + 1}'} ({int(row['Year'])})"
            if pd.notna(row.get("Year"))
            else str(row.get("Category", "")).strip() or f"Item {row.name + 1}"
        ),
    )
    remove_select_col.selectbox(
        "Select item",
        options=["-- Select Item --"] + labels,
        key="operating_remove_choice",
    )
    if remove_btn_col.button("Remove Item", key="operating_remove_row"):
        choice = st.session_state.get("operating_remove_choice")
        if choice in label_index:
            save_table(remove_row_fn(operating_table, label_index[choice]))
            st.session_state["operating_remove_choice"] = "-- Select Item --"
            clear_editor_state("assump::operating_costs")

    inc_target_col, inc_column_col, inc_pct_col, inc_btn_col = st.columns([2, 1.5, 1, 1])
    target_options = ["All categories"] + sorted(
        {
            str(cat).strip()
            for cat in operating_table.get("Category", pd.Series(dtype=str)).dropna().tolist()
            if str(cat).strip()
        }
    )
    inc_target_col.selectbox("Apply increment to", options=target_options, key="operating_increment_target")
    inc_column_col.selectbox(
        "Column",
        options=["unit_cost_per_head_per_month", "Inflation %"],
        key="operating_increment_column",
    )
    inc_pct_col.number_input(
        "Yearly increment (%)",
        min_value=-100.0,
        max_value=100.0,
        step=0.1,
        key="operating_increment_pct",
    )
    if inc_btn_col.button("Apply increment", key="operating_apply_increment"):
        save_table(
            apply_increment_fn(
                operating_table,
                st.session_state.get("operating_increment_pct", 0.0),
                st.session_state.get("operating_increment_target"),
                st.session_state.get("operating_increment_column", "unit_cost_per_head_per_month"),
            )
        )
        clear_editor_state("assump::operating_costs")

    render_row_editor(
        "assump::operating_costs",
        operating_table,
        lambda updated: save_table(ensure_fn(updated)),
    )

    st.session_state.setdefault("operating_defaults_edit_mode", False)
    toggle_label = (
        "Hide default operating cost assumptions"
        if st.session_state.operating_defaults_edit_mode
        else "Edit default operating cost assumptions"
    )
    if st.button(toggle_label, key="toggle_operating_defaults"):
        st.session_state.operating_defaults_edit_mode = not st.session_state["operating_defaults_edit_mode"]

    if st.session_state.operating_defaults_edit_mode:
        st.markdown("##### Default Operating Cost Assumptions")
        st.caption("Update the baseline operating cost table used when refreshing these assumptions.")
        default_frame = get_default_frame()
        template_editor = st.data_editor(
            default_frame,
            num_rows="dynamic",
            use_container_width=True,
            key="default_operating_editor",
            column_config={
                "Year": st.column_config.NumberColumn("Year", step=1),
                "Field": st.column_config.TextColumn("Field"),
                "unit_cost_per_head_per_month": st.column_config.NumberColumn(
                    "Unit Cost / Head / Month", format="%.4f"
                ),
                "Inflation %": st.column_config.NumberColumn("Inflation (%)", format="%.2f"),
            },
        )
        save_col, apply_col, restore_col, close_col = st.columns(4)
        if save_col.button("Save Defaults", key="save_operating_defaults"):
            save_defaults_fn(template_editor)
            st.success("Operating cost defaults updated.")
        if apply_col.button("Apply to Assumptions", key="apply_operating_defaults"):
            save_table(apply_defaults_fn(template_editor))
            st.success("Operating cost assumptions refreshed from updated defaults.")
            clear_editor_state("assump::operating_costs")
        if restore_col.button("Restore Baseline", key="reset_operating_defaults"):
            save_table(restore_defaults_fn())
            st.success("Operating cost defaults restored and assumptions refreshed.")
            clear_editor_state("assump::operating_costs")
        if close_col.button("Close Editor", key="close_operating_defaults"):
            st.session_state.operating_defaults_edit_mode = False

    return operating_table


def render_pricing_manual_editor(
    *,
    pricing_table: pd.DataFrame,
    assumptions: dict[str, pd.DataFrame],
    core_schedule: pd.DataFrame,
    sync_assumptions_fn: Callable[[dict[str, pd.DataFrame], pd.DataFrame], dict[str, pd.DataFrame]],
    refresh_quantities_fn: Callable[[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame]], pd.DataFrame],
    pricing_context_fn: Callable[[pd.DataFrame, Optional[pd.DataFrame], Optional[dict[str, pd.DataFrame]]], pd.DataFrame],
    pricing_validation_fn: Callable[[pd.DataFrame, Optional[pd.DataFrame]], list[str]],
    revenue_by_period_fn: Callable[[pd.DataFrame], pd.DataFrame],
    family_summary_fn: Callable[[pd.DataFrame], pd.DataFrame],
    quantity_by_period_fn: Callable[[pd.DataFrame], pd.DataFrame],
    add_row_fn: Callable[[pd.DataFrame], pd.DataFrame],
    remove_row_fn: Callable[[pd.DataFrame, int], pd.DataFrame],
    apply_increment_fn: Callable[[pd.DataFrame, str, float, Optional[str]], pd.DataFrame],
    render_row_editor: RenderEditor,
    clear_editor_state: Callable[[str], None],
    active_products: list[str],
    period_label: str,
) -> dict[str, pd.DataFrame]:
    refresh_context = pricing_context_fn(core_schedule, assumptions.get("Herd Plan"), assumptions)
    if st.button("Refresh derived quantities", key="pricing_refresh_quantities"):
        assumptions["Pricing"] = refresh_quantities_fn(
            assumptions["Pricing"],
            refresh_context,
            assumptions.get("Production Drivers"),
        )

    product_options = sorted(
        {
            str(product)
            for product in assumptions["Pricing"].get("Product", pd.Series(dtype=str)).dropna().tolist()
            if str(product).strip()
        }
    )
    if not product_options:
        product_options = sorted({product for product in active_products if str(product).strip()})

    st.markdown("##### Base Price Update")
    if product_options:
        st.session_state.setdefault("pricing_base_price_product", product_options[0])
        selected_product = st.session_state.get("pricing_base_price_product", "")
        if selected_product not in product_options:
            selected_product = product_options[0]
            st.session_state.pricing_base_price_product = selected_product

        default_base_price = 0.0
        if selected_product and "Product" in assumptions["Pricing"].columns:
            matching_rows = assumptions["Pricing"].loc[
                assumptions["Pricing"]["Product"].astype(str).str.strip() == selected_product,
                "Base Price",
            ]
            if not matching_rows.empty:
                current_price = pd.to_numeric(matching_rows, errors="coerce").dropna()
                if not current_price.empty:
                    default_base_price = float(current_price.iloc[0])

        last_product = st.session_state.get("pricing_base_price_last_product")
        if last_product != selected_product:
            st.session_state.pricing_base_price_amount = default_base_price
            st.session_state.pricing_base_price_last_product = selected_product
        else:
            st.session_state.setdefault("pricing_base_price_amount", default_base_price)

        st.caption("Choose a product and apply a new base price across its pricing rows.")
        base_price_product_col, base_price_amount_col, base_price_apply_col = st.columns([2, 1.5, 1])
        base_price_product_col.selectbox(
            "Product",
            options=product_options,
            key="pricing_base_price_product",
        )
        base_price_amount_col.number_input(
            "New base price",
            min_value=0.0,
            step=0.1,
            format="%.2f",
            key="pricing_base_price_amount",
        )
        if base_price_apply_col.button("Apply base price", key="pricing_apply_base_price"):
            updated_assumptions = dict(assumptions)
            updated_assumptions["Pricing"] = _apply_product_base_price(
                assumptions["Pricing"],
                str(st.session_state.get("pricing_base_price_product", "")),
                float(st.session_state.get("pricing_base_price_amount", 0.0)),
            )
            assumptions.update(sync_assumptions_fn(updated_assumptions, core_schedule))
            clear_editor_state("assump::pricing")
    else:
        st.caption("No products are available yet. Add pricing rows first to apply a base price update.")

    def _save_pricing_matrix(updated: pd.DataFrame) -> None:
        refreshed_assumptions = dict(assumptions)
        refreshed_assumptions["Pricing"] = updated
        assumptions.update(sync_assumptions_fn(refreshed_assumptions, core_schedule))

    pricing_matrix = st.data_editor(
        assumptions["Pricing"],
        use_container_width=True,
        key="assump::pricing_matrix",
        column_config={
            "Period": st.column_config.TextColumn("Period"),
            "Product": st.column_config.TextColumn("Product"),
            "Active": st.column_config.CheckboxColumn("Active"),
            "Allocation %": st.column_config.NumberColumn("Allocation (%)", format="%.2f", step=1.0),
            "Quantity Mode": st.column_config.SelectboxColumn("Quantity Mode", options=["Derived", "Manual Override"]),
            "Manual Quantity Override": st.column_config.NumberColumn(f"Manual qty / {period_label}", format="%.2f", step=1.0),
            "Quantity per Period": st.column_config.NumberColumn(f"Quantity per {period_label}", format="%.2f", step=1.0),
            "Unit": st.column_config.TextColumn("Unit"),
            "Base Price": st.column_config.NumberColumn("Base Price", format="%.2f", step=0.1),
            "Price Growth %": st.column_config.NumberColumn("Price Growth (%)", format="%.2f", step=0.1),
            "Revenue": st.column_config.NumberColumn("Revenue", format="%.2f"),
        },
        disabled=["Period", "Product", "Quantity per Period", "Revenue"],
    )
    _save_pricing_matrix(pricing_matrix)

    pricing_validation = pricing_validation_fn(
        assumptions["Pricing"],
        assumptions.get("Production Drivers"),
    )
    if pricing_validation:
        st.warning("Commercial validation: " + " ".join(f"- {msg}" for msg in pricing_validation))
    else:
        st.success("Commercial validation: active products, allocations, and production drivers are aligned.")

    st.markdown("##### Revenue Driven by Active Products")
    st.dataframe(revenue_by_period_fn(assumptions["Pricing"]), use_container_width=True)

    st.session_state.setdefault("pricing_remove_choice", "-- Select Row --")
    st.session_state.setdefault("pricing_increment_target", "All products")
    st.session_state.setdefault("pricing_increment_column", "Base Price")
    st.session_state.setdefault("pricing_increment_pct", 0.0)

    add_col, remove_select_col, remove_btn_col = st.columns([1, 2, 1])
    if add_col.button("Add Product", key="pricing_add_row"):
        updated_assumptions = dict(assumptions)
        updated_assumptions["Pricing"] = add_row_fn(pricing_table)
        assumptions.update(sync_assumptions_fn(updated_assumptions, core_schedule))
        clear_editor_state("assump::pricing")

    labels, label_index = build_remove_options(
        pricing_table,
        lambda row: (
            f"{int(row['Year'])} - {row.get('Product') or 'Product'}"
            if pd.notna(row.get("Year"))
            else str(row.get("Product") or "Product")
        ),
    )
    remove_select_col.selectbox(
        "Select row",
        options=["-- Select Row --"] + labels,
        key="pricing_remove_choice",
    )
    if remove_btn_col.button("Remove Row", key="pricing_remove_row"):
        choice = st.session_state.get("pricing_remove_choice")
        if choice in label_index:
            updated_assumptions = dict(assumptions)
            updated_assumptions["Pricing"] = remove_row_fn(pricing_table, label_index[choice])
            assumptions.update(sync_assumptions_fn(updated_assumptions, core_schedule))
            st.session_state.pricing_remove_choice = "-- Select Row --"
            clear_editor_state("assump::pricing")

    inc_target_col, inc_column_col, inc_pct_col, inc_btn_col = st.columns([2, 1.5, 1, 1])
    target_options = ["All products"] + sorted(
        {str(product) for product in pricing_table.get("Product", pd.Series(dtype=str)).dropna().tolist() if str(product).strip()}
    )
    inc_target_col.selectbox("Apply increment to", options=target_options, key="pricing_increment_target")
    inc_column_col.selectbox("Column", options=["Base Price", "Price Growth %"], key="pricing_increment_column")
    inc_pct_col.number_input(
        "Yearly increment (%)",
        min_value=-100.0,
        max_value=100.0,
        step=0.1,
        key="pricing_increment_pct",
    )
    if inc_btn_col.button("Apply increment", key="pricing_apply_increment"):
        updated_assumptions = dict(assumptions)
        updated_assumptions["Pricing"] = apply_increment_fn(
            pricing_table,
            st.session_state.get("pricing_increment_column", "Base Price"),
            st.session_state.get("pricing_increment_pct", 0.0),
            st.session_state.get("pricing_increment_target"),
        )
        assumptions.update(sync_assumptions_fn(updated_assumptions, core_schedule))
        clear_editor_state("assump::pricing")

    render_row_editor(
        "assump::pricing",
        assumptions["Pricing"],
        lambda updated: assumptions.update(
            sync_assumptions_fn({**assumptions, "Pricing": updated}, core_schedule)
        ),
    )

    st.info("Use the product planner and pricing matrix above as the source of truth for period-based product activation and revenue planning.")
    st.caption("The add/remove row tools below remain as a manual fallback, but the planner and matrix above should be the primary commercial workflow.")

    summary_col1, summary_col2 = st.columns(2)
    with summary_col1:
        st.markdown("**Revenue by Product**")
        st.dataframe(family_summary_fn(assumptions["Pricing"]), use_container_width=True)
    with summary_col2:
        st.markdown("**Quantity by Product and Period**")
        st.dataframe(quantity_by_period_fn(assumptions["Pricing"]), use_container_width=True)

    return assumptions
