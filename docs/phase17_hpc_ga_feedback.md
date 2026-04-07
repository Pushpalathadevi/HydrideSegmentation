# Phase 17 - HPC GA Feedback-Hybrid Optimization

## Goals

- Extend GA planning from diversity-only to feedback-informed ranking.
- Use previous run metrics to guide the next HPC sweep.
- Keep GUI, CLI, config, and profile workflows aligned.

## Implemented

1. Planner feedback ingestion and hybrid scoring:
- `src/microseg/app/hpc_ga.py`
- new config controls:
  - `fitness_mode`
  - `feedback_sources`
  - `feedback_min_samples`
  - `feedback_k`
  - `exploration_weight`
  - metric/runtime fitness weights
- new feedback contracts:
  - `HpcGaHistoricalSample`
  - `microseg.hpc_ga_feedback_summary.v1`
- candidate enrichment:
  - `predicted_fitness`
  - `selection_score`

2. CLI workflow extensions:
- `scripts/microseg_cli.py`
- updated `hpc-ga-generate` options for feedback-hybrid controls
- new `hpc-ga-feedback-report` command (JSON + markdown outputs)

3. GUI workflow extensions:
- `hydride_segmentation/qt/main_window.py`
- HPC GA Planner fields for feedback mode, sources, kNN and weights
- new `Analyze Feedback` action
- workflow profile save/load for `hpc_ga` now includes feedback fields

4. Orchestration builder extension:
- `src/microseg/app/orchestration.py`
- `OrchestrationCommandBuilder.hpc_ga_feedback_report(...)`

5. Config template update:
- `configs/hpc_ga.default.yml` includes feedback controls

6. Tests:
- `tests/test_phase15_hpc_ga_planner.py`
  - feedback source parsing
  - feedback summary generation
  - hybrid planning with predicted fitness fields
- `tests/test_phase4_orchestration.py`
  - command-builder coverage for `hpc-ga-feedback-report`

## Output Contracts

- planner bundle manifest: `microseg.hpc_ga_bundle.v1`
  - optional `feedback_summary` block
- feedback report: `microseg.hpc_ga_feedback_summary.v1`
  - JSON output + markdown table summary

## Validation

Run:

```bash
PYTHONPATH=. pytest -q tests/test_phase15_hpc_ga_planner.py tests/test_phase4_orchestration.py
```

Expected:
- All tests pass.
- Feedback-aware planner fields are covered by regression checks.

## Notes

- `feedback_hybrid` automatically falls back to novelty ranking when usable feedback sample count is below `feedback_min_samples`.
- Full Pareto-front ranking is still a future extension.
