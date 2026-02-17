# Phase 13 - Run Review And E2E QA Hardening (Implemented)

## Objective

Improve experiment visibility and operational confidence by adding:
- GUI-native training/evaluation report review and comparison
- reusable workflow profile persistence helpers with tests
- stronger automated coverage for report comparison and profile roundtrips

## Implemented

1. Run Review GUI tab
- Added Workflow Hub tab in:
  - `hydride_segmentation/qt/main_window.py`
- Features:
  - load two report JSON files (baseline/candidate)
  - summarize each report (schema, backend, status, runtime, device, config hash, metrics)
  - compute metric deltas in comparison table
  - compare metadata consistency (`kind`, `schema`, `backend`, `config_sha256`)

2. Report review backend module
- Added:
  - `src/microseg/app/report_review.py`
- Utilities:
  - `summarize_run_report(...)`
  - `compare_run_reports(...)`

3. Workflow profile backend module
- Added:
  - `src/microseg/app/workflow_profiles.py`
- Utilities:
  - `write_workflow_profile(...)`
  - `read_workflow_profile(...)`
- Schema:
  - `microseg.workflow_profile.v1`

4. GUI integration updates
- `main_window.py` now uses backend profile helpers instead of ad hoc YAML handling.
- Correction guide and workflow notes updated to include Run Review usage.

5. Test coverage expansion
- Added:
  - `tests/test_phase13_report_review.py`
- Covers:
  - evaluation report summarization and metric-delta comparison
  - training report summarization from history/latest metrics
  - workflow profile save/load roundtrip validation

## Exit Criteria

- Users can inspect and compare report outputs directly in GUI.
- Report-review/profile logic has dedicated automated tests.
- Docs and phase tracking are synchronized with implementation.
