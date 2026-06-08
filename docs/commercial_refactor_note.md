# Commercial Refactor Note

## Scope

This refactor extracts the pricing/commercial assumption synchronization and pricing validation rules from
`streamlit_app.py` into `src/goat_financial_model/commercial_services.py`.

## Old Structure

- `streamlit_app.py` owned UI rendering, session-state orchestration, pricing validation, and commercial assumption
  synchronization.
- The pricing/commercial business rules were embedded directly in the app entrypoint, which made the lineage from
  assumptions to synchronized pricing tables harder to audit and harder to test outside the Streamlit runtime.

## New Structure

- `streamlit_app.py` remains the orchestration entrypoint and preserves the existing public helper names.
- `commercial_services.py` now contains:
  - `build_pricing_validation_messages(...)`
  - `sync_commercial_assumptions_to_core(...)`
- `streamlit_app.py` calls those extracted functions through thin wrappers, preserving the existing app behavior and
  test surface.

## Why This Improves Auditability

- The commercial business rules now live in a dedicated module instead of being mixed into the main UI file.
- The extracted functions are deterministic and explicit about their dependencies.
- The service module can be tested directly for parity against the existing wrappers.
- The wrapper layer keeps current user workflows and internal call sites stable while allowing further refactor work
  to proceed incrementally.

## Preserved Behavior

- Pricing assumptions are still rebased to the active core schedule.
- Production drivers and scenario controls are still filtered to the active business type.
- Derived pricing quantities are still recalculated from the biological and production context.
- Pricing validation messages retain the same wording and rules.

## Recommended Next Steps

1. Extract pricing summary helpers and revenue application logic into the same service layer.
2. Apply the same pattern to biological synchronization helpers.
3. Add explicit reconciliation utilities for commercial assumptions and downstream revenue schedules.
