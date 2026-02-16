# Phase 11 - Dataset Policy Alignment (Implemented)

## Objective

Finalize dataset auto-prepare behavior with explicit policy decisions:
- leakage-aware split as the default
- optional RGB color-mask ingestion through configurable colormap conversion
- globally unique file IDs across prepared outputs

## Implemented

1. Leakage-aware auto-split for unsplit datasets
- `src/microseg/dataops/training_dataset.py`
- Added `split_strategy`, `leakage_group_mode`, and `leakage_group_regex`.
- Added suffix-aware grouping mode to keep augmented variants together.

2. RGB mask conversion support
- Added `mask_input_type` and `mask_colormap` options in dataset preparation config.
- Supports index->RGB and RGB->index colormap definitions.
- Strict unknown-color validation enabled by default (`mask_colormap_strict: true`).

3. Global ID contract
- Prepared samples keep original stem and append global IDs (`_000001`, `_000002`, ...).
- Manifest mapping now includes `global_id`, `source_group`, and split/group metadata.

4. CLI and config integration
- `scripts/microseg_cli.py` and default YAML templates updated.
- New options available in `train`, `evaluate`, and `dataset-prepare` auto-prepare paths.

5. Test coverage
- Extended `tests/test_phase10_dataset_prepare.py` for:
  - leakage-aware group preservation
  - RGB mask colormap conversion
  - global ID coverage across splits
- Extended `tests/test_phase4_foundation_features.py` for JSON object parsing in `--set` overrides.

## Exit Criteria

- Dataset preparation behavior is policy-aligned, tested, and documented for user/developer workflows.
