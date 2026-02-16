# Phase 9 - Model Lifecycle And Dataset Operations Foundation

## Objective

Implement production-grade foundations for:
- frozen-model metadata governance
- leakage-safe dataset split operations
- dataset quality checks before training/evaluation

## Implemented

1. Strict registry validator
- Added `src/microseg/plugins/registry_validation.py`.
- Validates:
  - schema version
  - required model fields
  - unique `model_id` and `model_nickname`
  - class index integrity
  - non-absolute checkpoint path hints
- CLI:
```bash
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```

2. Leakage-aware split planner
- Added `src/microseg/dataops/split_planner.py`.
- Builds train/val/test splits from correction exports with deterministic seed control.
- Supports leakage grouping (`source_stem` or `sample_id`).
- Materializes packaged dataset and writes:
  - `dataset_manifest.json` (`microseg.dataset_split_manifest.v1`)
- CLI:
```bash
microseg-cli dataset-split --config configs/dataset_split.default.yml
```

3. Dataset QA checks
- Added `src/microseg/dataops/quality.py`.
- Checks:
  - missing image/mask pairs
  - image/mask dimension mismatches
  - duplicate file content hashes
  - class imbalance warning threshold
- Writes:
  - `dataset_qa_report.json` (`microseg.dataset_qa.v1`)
- CLI:
```bash
microseg-cli dataset-qa --config configs/dataset_qa.default.yml --strict
```

## Tests Added

- `tests/test_phase9_registry_validation.py`
- `tests/test_phase9_dataops.py`
