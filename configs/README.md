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
- `registry_validation.default.yml` for frozen registry validation
- `dataset_split.default.yml` for leakage-aware split planning
- `dataset_qa.default.yml` for packaged dataset QA checks
- `dataset_prepare.default.yml` for unsplit source/masks -> split dataset preparation
  - includes `split_strategy`, leakage-group controls, RGB `mask_input_type`, `mask_colormap`, and optional `binary_mask_normalization`
- `hpc_ga.default.yml` for GA-based HPC multi-candidate script bundle generation
  - includes feedback-hybrid planning controls and metric/runtime fitness weights
- `hpc_ga.airgap_pretrained.default.yml` for low-friction air-gapped HPC sweeps
  - enables `pretrained_init_mode=auto` with backend-to-model mapping from `pre_trained_weights/registry.json`
