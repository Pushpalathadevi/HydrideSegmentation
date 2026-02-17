# Configuration Workflow

## Objective

All major pipelines are config-driven with YAML base files and `--set` overrides.

## YAML Files

Reference templates:
- `configs/inference.default.yml`
- `configs/train.default.yml`
- `configs/evaluate.default.yml`
- `configs/package.default.yml`
- `configs/phase_gate.default.yml`
- `configs/registry_validation.default.yml`
- `configs/dataset_split.default.yml`
- `configs/dataset_qa.default.yml`
- `configs/dataset_prepare.default.yml`

## CLI Usage

Inference:
```bash
microseg-cli infer --config configs/inference.default.yml --set params.area_threshold=120 --set include_analysis=true
```

Dataset packaging:
```bash
microseg-cli package --config configs/package.default.yml --set train_ratio=0.75 --set val_ratio=0.15
```

Leakage-aware split planning:
```bash
microseg-cli dataset-split --config configs/dataset_split.default.yml
```

Unsplit source/masks to split-layout preparation:
```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

Leakage-aware split controls for unsplit auto-prepare:
```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set split_strategy=leakage_aware \
  --set leakage_group_mode=suffix_aware
```

RGB mask colormap conversion example:
```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set mask_input_type=rgb_colormap \
  --set 'mask_colormap={"0":[0,0,0],"1":[255,0,0],"2":[0,255,0]}'
```

Training:
```bash
microseg-cli train --config configs/train.default.yml --set max_samples=300000 --set epochs=12
```
Common training backend options:
- `backend=unet_binary` (default)
- `backend=torch_pixel`
- `backend=sklearn_pixel`

UNet resume example:
```bash
microseg-cli train --config configs/train.default.yml --set resume_checkpoint=outputs/training/last_checkpoint.pt
```

UNet validation tracking + reporting example:
```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set val_tracking_samples=8 \
  --set "val_tracking_fixed_samples=val_000.png|val_123.png" \
  --set write_html_report=true \
  --set progress_log_interval_pct=10
```

Evaluation:
```bash
microseg-cli evaluate --config configs/evaluate.default.yml --set split=test
```

Evaluation HTML + tracked sample panel example:
```bash
microseg-cli evaluate \
  --config configs/evaluate.default.yml \
  --set tracking_samples=12 \
  --set write_html_report=true
```

Dataset QA:
```bash
microseg-cli dataset-qa --config configs/dataset_qa.default.yml --strict
```

Frozen registry validation:
```bash
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```

GPU-enabled runs (auto policy with CPU fallback):
```bash
microseg-cli train --config configs/train.default.yml --enable-gpu --device-policy auto
microseg-cli infer --config configs/inference.default.yml --enable-gpu --device-policy auto
microseg-cli evaluate --config configs/evaluate.default.yml --enable-gpu --device-policy auto
```

## GUI Usage

- Provide optional config path in the top `Config` field.
- Add comma-separated overrides in `key=value` form.
- Run segmentation; final parameters are merged as:
  - YAML base
  - GUI override entries
  - runtime image path
- Workflow Hub supports YAML profile save/load for:
  - `dataset_prepare`
  - `training`
  - `evaluation`

## Override Conventions

- Dotted keys create nested structures (`params.crop=true`).
- Scalars parse into bool/int/float/null/string automatically.
- JSON objects/lists can be passed in `--set` values (for example `mask_colormap={...}`).

## Reproducibility

CLI inference/package commands persist `resolved_config.json` with outputs.
CLI training/evaluation commands also persist `resolved_config.json` beside artifacts/reports.
Training additionally writes:
- `report.json` with status/progress/ETA/history
- `training_report.html` for rapid visual review
- `eval_samples/epoch_XXX` tracked validation panels

Phase closeout checks:
```bash
microseg-cli phase-gate --config configs/phase_gate.default.yml --set phase_label=\"Phase N\"
```
