# Phase 15 - GA-Based HPC Orchestration Bundle

## Goals

- Add GA-based configuration manipulator for architecture/hyperparameter sweeps.
- Generate one-click HPC script bundles from GUI and CLI.
- Support Slurm/PBS/local execution styles.
- Preserve reproducibility via manifested candidate metadata.

## Implemented

1. New GA planner and bundle writer:
- `src/microseg/app/hpc_ga.py`
- novelty-oriented GA candidate generation
- script generation for train/train+eval pipelines

2. New CLI command:
- `microseg-cli hpc-ga-generate`
- config template: `configs/hpc_ga.default.yml`

3. GUI integration:
- Workflow Hub tab: `HPC GA Planner`
- one-click bundle generation job via orchestration runtime
- profile save/load support for `hpc_ga` scope

4. Command builder integration:
- `OrchestrationCommandBuilder.hpc_ga_generate(...)`

5. Test coverage:
- `tests/test_phase15_hpc_ga_planner.py`
- updated command-builder test in `tests/test_phase4_orchestration.py`

## Output Contracts

Bundle root includes:
- `ga_plan_manifest.json` (`microseg.hpc_ga_bundle.v1`)
- `submit_all.sh`
- `jobs/*.sh`
- `candidates/*.json` + `candidates/*.yml`
- `README.txt`

## Notes

- Current GA is diversity-first (novelty search), suitable for initial sweep setup.
- Metric-feedback optimization can be layered in later phases using report-derived fitness.
