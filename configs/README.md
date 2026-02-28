# Configurations

Target configuration groups:
- `models/` model registry entries and model-specific defaults
- `pipelines/` inference/training pipeline definitions
- `app/` desktop app defaults

All configs should be versioned and reproducible.

Related metadata:
- frozen model metadata registry: `frozen_checkpoints/model_registry.json`
- smoke-checkpoint generator script: `scripts/generate_smoke_checkpoint.py`
- local pretrained registry (air-gap transfer): `pre_trained_weights/registry.json`
- pretrained bundle bootstrap script: `scripts/download_pretrained_weights.py`
- hydride benchmark presets: `configs/hydride/README.md`

Additional workflow config:
- `phase_gate.default.yml` for end-of-phase closeout checks
- `preflight.default.yml` for unified train/eval/benchmark/deploy preflight checks
- `deployment_package.default.yml` for deployment bundle creation inputs
- `deploy_health.default.yml` for runtime deployment health checks and queue-style batch validation
- `promotion_policy.default.yml` for objective model-promotion thresholds
- `support_bundle.default.yml` for run diagnostics/support artifact bundling
- `registry_validation.default.yml` for frozen registry validation
- `dataset_split.default.yml` for leakage-aware split planning
- `dataset_qa.default.yml` for packaged dataset QA checks
- `dataset_prepare.default.yml` for unsplit source/masks -> split dataset preparation
  - includes `split_strategy`, leakage-group controls, RGB `mask_input_type`, `mask_colormap`, optional `binary_mask_normalization`, and optional `class_map_path` (fallback: `configs/segmentation_classes.json`)
- `data_prep.default.yml` for `prep-dataset` binary segmentation data preparation (`src/microseg/data_preparation`)
  - includes binarization mode/threshold controls, red-dominance RGB fallback, auto-Otsu fallback for noisy near-binary grayscale masks, empty-mask warn/error policy, resizing policy, debug inspection options, and warning-related raw-mask expectations
- `segmentation_classes.json` repo-level default class definitions used by correction/export and RGB-mask class-color fallback
- `hpc_ga.default.yml` for GA-based HPC multi-candidate script bundle generation
  - includes feedback-hybrid planning controls and metric/runtime fitness weights
- `hpc_ga.airgap_pretrained.default.yml` for low-friction air-gapped HPC sweeps
  - enables `pretrained_init_mode=auto` with backend-to-model mapping from `pre_trained_weights/registry.json`
- `hpc_ga.top5_scratch.default.yml` for the canonical top-5 scratch-only hydride sweep
- `hpc_ga.top5_airgap_pretrained.default.yml` for the canonical top-5 air-gapped local-pretrained sweep
