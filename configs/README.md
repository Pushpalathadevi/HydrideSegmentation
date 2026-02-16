# Configurations

Target configuration groups:
- `models/` model registry entries and model-specific defaults
- `pipelines/` inference/training pipeline definitions
- `app/` desktop app defaults

All configs should be versioned and reproducible.

Related metadata:
- frozen model metadata registry: `frozen_checkpoints/model_registry.json`

Additional workflow config:
- `phase_gate.default.yml` for end-of-phase closeout checks
- `registry_validation.default.yml` for frozen registry validation
- `dataset_split.default.yml` for leakage-aware split planning
- `dataset_qa.default.yml` for packaged dataset QA checks
- `dataset_prepare.default.yml` for unsplit source/masks -> split dataset preparation
  - includes `split_strategy`, leakage-group controls, RGB `mask_input_type`, and `mask_colormap` options
