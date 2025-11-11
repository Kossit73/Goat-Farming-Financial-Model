# Goat Model Improvement Recommendations

## 1. Increase Data Robustness
- **Explicit field validation:** Before returning `to_tidy()` output, surface which critical series are missing instead of raising a generic error. This can be done by comparing the keys in `series` against a required set (e.g., revenue, COGS, EBITDA) and raising a descriptive exception or warning so template deviations are obvious during ingestion.
- **Scenario safeguards:** `scenario()` currently assumes fixed `Revenue`/`COGS` availability and clamps the effective tax rate between 0 and 50%. Consider surfacing validation messages when the resulting adjusted metrics become negative or when the implied tax rate hits a bound, which signals the workbook history may be inconsistent.
- **Workbook schema tests:** Add a lightweight test suite that exercises `_SHEET_ALIASES`, `_extract_series`, and `_extract_named_table` against synthetic DataFrames to prevent regressions when refactoring extraction logic.

## 2. Expand Financial Coverage
- **Driver-level expenses:** The template exposes variable, fixed, direct wage, and admin wage series individually (`variable_expenses()`, `fixed_expenses()`, `direct_wages()`, `admin_wages()`). Add derived metrics such as cost per litre or per goat by combining these series with production volumes to give operators more actionable benchmarks.
- **Capex and depreciation alignment:** Tie `capex()` output to the depreciation schedule by reconciling the cash outflow (`CF` sheet) with the asset schedules. This will improve the accuracy of long-term EBITDA and cash-flow planning.
- **Capitalisation analytics:** The `capitalisation_table()` helper extracts the cap table but does not summarise dilution or ownership trends. Introduce convenience functions that compute post-money valuation, investor ownership, and dilution impacts per round to support fundraising analysis.

## 3. Deepen Scenario & Sensitivity Analysis
- **Multi-factor toggles:** Extend `scenario()` to accept additional levers (e.g., herd size, milk yield, wage inflation). These can scale the variable and wage series in tandem with revenue/COGS to simulate operational shifts.
- **Scenario comparison:** Provide a method that compares multiple scenario DataFrames and returns key deltas (revenue, EBITDA, cash flow). This would pair well with the Streamlit dashboard to visualise upside/downside cases side by side.
- **Monte Carlo / tornado analysis:** For strategic planning, implement stochastic simulations around milk prices and feed costs, or a tornado chart summarising sensitivity coefficients for the major drivers.

## 4. Streamlit UX Enhancements
- **Input validation & status:** In `streamlit_app.py`, centralise the file-loading errors surfaced by `_run_model` and show inline callouts that guide users through fixing missing sheets or mismatched headers. This reduces the need to check server logs during deployment.
- **Downloadable schedule packs:** Offer multi-tab Excel exports that bundle the scenario output, KPI table, break-even analysis, and supplementary schedules so decision makers can share full scenario packages offline.
- **Performance profiling:** Cache individual table extractions and large chart transformations (e.g., stacked expenses) with `st.cache_data` to cut down on re-computation when users tweak minor assumptions.

## 5. Benchmarking & KPI Storytelling
- **Normalise benchmark data:** The `benchmark_kpis()` helper merely cleans the sheet. Add logic to align the benchmark KPIs with the model’s actual metrics (e.g., converting percentages to decimals, standardising period labels) so they can be plotted alongside the farm’s performance.
- **Alert thresholds:** Introduce rules-based alerts when actual metrics deviate materially from benchmark KPIs (for instance, when COGS% or admin wages exceed targets). These alerts can surface as Streamlit status boxes or be exposed via the CLI for automated reporting.
- **Narrative summaries:** Generate text insights that describe trends (e.g., "EBITDA margin compresses 3pp under the -10% milk price scenario") by comparing KPI deltas. This helps non-technical stakeholders interpret the charts quickly.
