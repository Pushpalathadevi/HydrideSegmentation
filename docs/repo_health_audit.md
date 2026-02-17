# Repository Health Audit (Updated)

## High-priority architecture actions completed

- MicroSeg now owns its own versioning and image-encoding utilities (`src/microseg/version.py`, `src/microseg/utils/encoding.py`), reducing internal dependence on `hydride_segmentation` for core orchestration and reporting code paths.
- Hydride analysis helpers were formalized in `src/microseg/evaluation/hydride_metrics.py` and are consumed by the MicroSeg analyzer/evaluation pipeline.
- Remaining `hydride_segmentation` imports in `src/microseg/inference/predictors.py` are explicitly treated as legacy model adapters; future work can replace them with native `src/microseg/inference` implementations.

## Dependency and packaging hardening

- Introduced profile-based requirements:
  - `requirements-core.txt` (headless/HPC default, uses `opencv-python-headless`)
  - `requirements-gui.txt` (desktop GUI stack)
  - `requirements.txt` now points to GUI profile + test extras.
- `pyproject.toml` now uses `opencv-python-headless` and exposes optional extras (`core`, `gui`, `transformers`).
- Added pinned reproducibility baseline lock file: `envs/microseg-core.lock.txt`.

## Training harness upgrades (benchmark-grade foundations)

`UNetBinaryTrainingConfig` and trainer now support:

- mixed precision (`amp_enabled`)
- gradient accumulation (`grad_accum_steps`)
- controlled dataloader behavior (`num_workers`, `pin_memory`, `persistent_workers`)
- deterministic mode (`deterministic`)

These settings are exposed in `microseg-cli train` and included in resolved configs/reports.

## Evaluation scientific metrics expansion

`PixelModelEvaluator` now reports scientific summary metrics (mean across evaluated samples):

- area-fraction GT/pred and absolute error
- hydride count GT/pred and absolute error
- size distribution distances (Wasserstein, KS)
- orientation distribution distances (Wasserstein, KS)

Report schema is bumped to `microseg.pixel_eval.v3`.

## Dataset freeze enforcement in benchmark mode

`scripts/hydride_benchmark_suite.py` now supports benchmark hard-fail checks:

- requires `dataset_manifest.json` when `benchmark_mode=true`
- optional strict manifest hash check via `expected_dataset_manifest_sha256`
- optional strict split-ID membership check via `expected_split_id_file`

This hardens split reproducibility and leakage-guard assumptions for HPC sweeps.
