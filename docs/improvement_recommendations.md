# Goat Farming Financial Model – Investor-Readiness Review

This review is written from a senior financial analyst perspective with one goal: make the model easier for equity/debt investors to underwrite quickly and confidently.

## What is already strong

- The model enforces strong **data hygiene** with datetime-index validation and numeric coercion checks, reducing silent spreadsheet errors.
- It already supports broad **core statements and KPI extraction** (revenue, cost buckets, EBITDA/EBIT/NPAT, cash flow) and includes an advanced analytics module for sensitivity and Monte Carlo workflows.
- The Streamlit interface allows non-technical stakeholders to run scenarios and export outputs.

These are a solid base. The biggest remaining gap is not calculation breadth, but **investor packaging**: proving assumptions, downside resilience, and capital efficiency in a repeatable way.

---

## Priority 1 — Build an investor-grade assumptions bridge (highest ROI)

### Why investors care
Institutional investors typically discount management forecasts when assumptions are not fully auditable. They want to trace every major output to a driver and see historical support.

### Improvement actions
1. **Create a formal assumptions table** with fields for:
   - driver name,
   - base value,
   - source (historical average, contract, market quote, management estimate),
   - update cadence,
   - confidence level.
2. **Link assumptions directly to scenario levers** (milk price, feed cost, labour inflation, herd productivity, capex timing).
3. **Add an “assumption variance” report** showing forecast vs. actual deviations by period.

### Investor impact
- Faster diligence process.
- Higher confidence in forecast integrity.
- Better support for valuation multiples and discount-rate discussions.

---

## Priority 2 — Add debt capacity and covenant visibility

### Why investors care
Even equity investors assess refinancing and liquidity risk. Debt providers need covenant headroom under downside scenarios.

### Improvement actions
1. Add core funding metrics per period:
   - DSCR,
   - Interest Coverage,
   - Net Debt / EBITDA,
   - Minimum cash runway (months).
2. Add a covenant dashboard with:
   - threshold,
   - actual,
   - headroom,
   - breach flag.
3. Include these metrics in base, downside, and severe downside cases.

### Investor impact
- Demonstrates balance-sheet discipline.
- Reduces perceived financing risk.
- Supports larger check sizes at better terms.

---

## Priority 3 — Upgrade scenario design from “single shock” to “operating system”

### Why investors care
Real downside does not happen one variable at a time. Investors need stacked shocks and management response logic.

### Improvement actions
1. Define standard scenario packs:
   - Base,
   - Management Case,
   - Downside (price + cost + yield pressure),
   - Recovery (operational turnaround assumptions).
2. Add **response levers** (cost cuts, capex deferral, hiring freeze, hedging assumptions).
3. Add **scenario delta tables** for revenue, EBITDA margin, free cash flow, and valuation.

### Investor impact
- Shows operational maturity.
- Strengthens downside credibility.
- Improves IC (investment committee) readability.

---

## Priority 4 — Tighten free-cash-flow credibility

### Why investors care
Enterprise value is often underwritten against sustainable free cash flow, not accounting earnings.

### Improvement actions
1. Split capex into:
   - maintenance capex,
   - expansion capex.
2. Add explicit working-capital drivers:
   - receivable days,
   - payable days,
   - inventory turns (or feed-stock days).
3. Include a **cash conversion bridge**:
   EBITDA → operating cash flow → free cash flow.

### Investor impact
- More reliable DCF outcomes.
- Better comparability to listed agribusiness peers.
- Clearer path to distributable cash and exit value.

---

## Priority 5 — Make valuation output board/investor ready

### Why investors care
Current valuation fields are useful but not yet packaged like an investment memo.

### Improvement actions
1. Provide a valuation panel with:
   - DCF (base/downside/upside),
   - EV/EBITDA and EV/Revenue ranges,
   - implied IRR and MOIC at entry valuation.
2. Add a **waterfall chart** from enterprise value to equity value:
   EV → net debt → minority interests → equity.
3. Add a simple dilution schedule for future capital rounds.

### Investor impact
- Shortens path from model to term-sheet discussion.
- Frames valuation debate around transparent drivers.
- Improves confidence in ownership-outcome planning.

---

## Priority 6 — Add benchmark narrative and KPI alerts

### Why investors care
Investors want to know whether performance is “good enough” versus market standards, not only versus plan.

### Improvement actions
1. Convert benchmark KPI ingestion into normalized peer comparison cards.
2. Add threshold alerts for key risk indicators:
   - gross margin compression,
   - feed-cost ratio creep,
   - labour-cost inflation above plan,
   - covenant headroom deterioration.
3. Auto-generate short commentary per scenario (2–5 lines) for IC packs.

### Investor impact
- Better executive communication quality.
- Faster interpretation for non-operators.
- Stronger credibility in fundraising decks.

---

## Suggested implementation order (90-day plan)

### Days 1–30
- Implement assumptions bridge + audit trail.
- Add debt metrics and covenant headroom calculations.
- Standardize scenario pack definitions.

### Days 31–60
- Add working-capital and capex split logic.
- Introduce cash conversion bridge visuals/tables.
- Expand scenario comparison outputs in Streamlit export.

### Days 61–90
- Complete valuation panel and equity waterfall.
- Add dilution schedule and investor one-page summary export.
- Add benchmark alerts + narrative commentary generator.

---

## What this changes in fundraising outcomes

If implemented, the model moves from “good internal planning tool” to “institutional investor decision tool.”

Expected practical benefits:
- More credible forecast defense during diligence,
- fewer diligence follow-up cycles,
- improved perception of risk management quality,
- stronger negotiating position on valuation and structure.
