# HPC GA Developer Guide

## Objective

Provide a reproducible, code-driven way to generate HPC experiment bundles for multi-model/multi-hyperparameter comparisons.

## Main Modules

- Planner and bundle writer:
  - `src/microseg/app/hpc_ga.py`
- CLI integration:
  - `scripts/microseg_cli.py` (`hpc-ga-generate`, `hpc-ga-feedback-report`)
- GUI integration:
  - `hydride_segmentation/qt/main_window.py` (`HPC GA Planner` tab)
- Command-builder integration:
  - `src/microseg/app/orchestration.py`
- Profile persistence:
  - `src/microseg/app/workflow_profiles.py` (scope `hpc_ga`)

## Data Contracts

`HpcGaPlanConfig`
- complete planner + scheduler + runtime config

`HpcGaCandidate`
- one candidate's backend/hyperparameters/seed/novelty score
- includes optional `predicted_fitness` and `selection_score` fields for feedback mode

`HpcGaHistoricalSample`
- one parsed feedback sample from prior `candidates/cand_*.json` + `runs/cand_*/eval_report.json`
- includes derived `fitness_score`

`HpcGaBundleResult`
- generated bundle and manifest paths plus selected candidates

## GA Strategy (Current)

Supported modes:

- `novelty`
  - initial random population from configured ranges
  - tournament selection
  - crossover + mutation
  - final ranking by novelty distance in normalized parameter space

- `feedback_hybrid`
  - initial random population from configured ranges
  - tournament selection
  - crossover + mutation
  - kNN predicted fitness from historical runs
  - blended ranking: `exploration_weight * novelty + (1-exploration_weight) * predicted_fitness`
  - automatic fallback to novelty when feedback sample count is below `feedback_min_samples`

Feedback fitness function uses weighted metrics:
- `mean_iou`
- `macro_f1`
- `pixel_accuracy`
- runtime penalty (`runtime_seconds`, normalized)

## Generated Artifact Schemas

Manifest:
- `microseg.hpc_ga_bundle.v1` (`ga_plan_manifest.json`)
- includes optional `feedback_summary`

Candidate payload:
- `microseg.hpc_ga_candidate.v1` (`candidates/cand_XXX.json`)
- may include `predicted_fitness` and `selection_score`

Feedback summary:
- `microseg.hpc_ga_feedback_summary.v1`
- produced by `microseg-cli hpc-ga-feedback-report`

## Script Generation Rules

Each candidate script:
- creates candidate run folder
- runs `microseg-cli train`
- optionally runs `microseg-cli evaluate` when `run_mode=train_eval`

Model path fallback for evaluation:
1. `best_checkpoint.pt`
2. `last_checkpoint.pt`
3. `torch_pixel_classifier.pt`

Master launcher:
- `submit_all.sh`
- dispatches via `sbatch`, `qsub`, or local `bash`

## Extending The Planner

1. Add parameter fields to `HpcGaPlanConfig`.
2. Update:
- `_sample_candidate`
- `_candidate_features`
- `_crossover`
- `_mutate`
- `_compute_selection_scores`
- `_job_script_lines` (`--set` overrides)
3. Update CLI parser defaults and GUI controls.
4. Add/adjust tests in `tests/test_phase15_hpc_ga_planner.py`.
5. Update docs in same change.

## Testing

Core tests:
- `tests/test_phase15_hpc_ga_planner.py`
- `tests/test_phase4_orchestration.py` (command builder path)

Run:
```bash
PYTHONPATH=. pytest -q tests/test_phase15_hpc_ga_planner.py tests/test_phase4_orchestration.py
```

## Design Constraints

- Keep generated scripts explicit and inspectable.
- Keep all paths reproducible and repo-relative where possible.
- Keep scheduler support generic (Slurm/PBS/local).
- Keep GUI and core planning logic decoupled.
- Keep feedback ingestion failure-tolerant but non-silent in user reports (sample counts and sources must be visible).

## Future Work

- multi-objective Pareto ranking (quality vs runtime)
- scheduler-specific array-job generation
- optional conda/module environment bootstrap preamble
