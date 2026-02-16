# Configuration Workflow

## Objective

All major pipelines are config-driven with YAML base files and `--set` overrides.

## YAML Files

Reference templates:
- `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/configs/inference.default.yml`
- `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/configs/package.default.yml`

## CLI Usage

Inference:
```bash
microseg-cli infer --config configs/inference.default.yml --set params.area_threshold=120 --set include_analysis=true
```

Dataset packaging:
```bash
microseg-cli package --config configs/package.default.yml --set train_ratio=0.75 --set val_ratio=0.15
```

## GUI Usage

- Provide optional config path in the top `Config` field.
- Add comma-separated overrides in `key=value` form.
- Run segmentation; final parameters are merged as:
  - YAML base
  - GUI override entries
  - runtime image path

## Override Conventions

- Dotted keys create nested structures (`params.crop=true`).
- Scalars parse into bool/int/float/null/string automatically.

## Reproducibility

CLI inference/package commands persist `resolved_config.json` with outputs.
