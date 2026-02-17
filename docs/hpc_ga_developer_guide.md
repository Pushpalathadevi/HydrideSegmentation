# HPC GA Developer Guide

## Objective

Provide a reproducible, code-driven way to generate HPC experiment bundles for multi-model/multi-hyperparameter comparisons.

## Main Modules

- Planner and bundle writer:
  - `src/microseg/app/hpc_ga.py`
- CLI integration:
  - `scripts/microseg_cli.py` (`hpc-ga-generate`)
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

`HpcGaBundleResult`
- generated bundle and manifest paths plus selected candidates

## GA Strategy (Current)

Current GA is novelty-oriented:
- initial random population from configured ranges
- tournament selection
- crossover + mutation
- final ranking by novelty (distance in normalized parameter space)

This is intentionally diversity-first for first-pass sweep generation.
Future phases can add metric-driven fitness from prior run reports.

## Generated Artifact Schemas

Manifest:
- `microseg.hpc_ga_bundle.v1` (`ga_plan_manifest.json`)

Candidate payload:
- `microseg.hpc_ga_candidate.v1` (`candidates/cand_XXX.json`)

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

## Future Work

- fitness integration from prior evaluation reports
- multi-objective Pareto ranking (quality vs runtime)
- scheduler-specific array-job generation
- optional conda/module environment bootstrap preamble
