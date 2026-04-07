# Phase 10 - Training Dataset Auto-Prepare Contract

## Objective

Formalize and implement training data expectations so the pipeline can accept:
- explicit split datasets (`train/val/test`)
- unsplit datasets (`source/masks` or `images/masks`)

with deterministic default split behavior and programmatic file IDs.

## Implemented

1. Training dataset preparation module
- Added `src/microseg/dataops/training_dataset.py`.
- Supports:
  - split layout detection and direct use
  - unsplit layout detection and auto split
  - format checks and indexed mask checks
  - deterministic split and ID-suffixed renaming

2. Default split behavior
- If split folders are absent, auto split defaults to:
  - `train=0.8`
  - `val=0.1`
  - `test=0.1`
- Ratios and seed are configurable via YAML/CLI.

3. Naming and mapping
- Keeps original stem and appends `_ID`:
  - `name_000001.png`
- Writes `dataset_prepare_manifest.json` with original/new path mapping and split assignment.

4. CLI integration
- Added explicit command:
  - `microseg-cli dataset-prepare`
- Training/evaluation (`microseg-cli train/evaluate`) now auto-prepare by default when split folders are missing.

5. Documentation
- Added `docs/training_data_requirements.md`.
- Updated config/workflow docs and command references.
- Added optional binary normalization modes:
  - `two_value_zero_background` for two-value indexed masks (for example `0/255` -> `0/1`)
  - `nonzero_foreground` for robust legacy binary masks (`0` background, any non-zero foreground)
- Added class-map fallback resolution for RGB masks via `--class-map-path`, `MICROSEG_CLASS_MAP_PATH`, and `configs/segmentation_classes.json`.

## Example

```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
microseg-cli train --config configs/train.default.yml
microseg-cli evaluate --config configs/evaluate.default.yml --set split=val
```

## Tests Added

- `tests/test_phase10_dataset_prepare.py`

## Follow-up Decisions Implemented

- Auto-prepare split strategy is now leakage-aware by default (`split_strategy=leakage_aware`).
- Optional RGB mask ingestion is supported through configurable `mask_colormap`.
- Prepared IDs are global (not split-scoped) and persisted in manifest mapping.
