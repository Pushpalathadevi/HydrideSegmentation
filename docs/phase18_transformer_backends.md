# Phase 18 - Transformer Segmentation Trial Backends

## Goals

- Introduce transformer-based segmentation options for hydride benchmarking.
- Preserve existing UNet and baseline backend behavior.
- Expose trial-ready configurations for HPC experiments.

## Implemented

1. New backend variants (shared trainer path):
- `transunet_tiny`
- `segformer_mini`

2. Training stack updates:
- `src/microseg/training/unet_binary.py`
  - model architecture factory
  - transformer hyperparameter config support
  - new checkpoint schema for non-UNet transformer variants
  - backward compatibility with existing UNet checkpoints

3. Evaluation compatibility:
- `src/microseg/evaluation/pixel_model_eval.py`
  - supports loading both:
    - `microseg.torch_unet_binary.v1`
    - `microseg.torch_segmentation_binary.v2`

4. CLI/GUI updates:
- `scripts/microseg_cli.py`
  - backend choices include `transunet_tiny`, `segformer_mini`
  - transformer hyperparameter CLI flags added
- `hydride_segmentation/qt/main_window.py`
  - training backend dropdown includes transformer options
  - HPC architecture default list includes transformer variants

5. Config presets:
- `configs/hydride/train.unet_binary.baseline.yml`
- `configs/hydride/train.transunet_tiny.yml`
- `configs/hydride/train.transunet_tiny_deep.yml`
- `configs/hydride/train.segformer_mini.yml`
- `configs/hydride/train.segformer_mini_wide.yml`
- `configs/hydride/train.torch_pixel.yml`
- `configs/hydride/train.sklearn_pixel.yml`
- `configs/hydride/evaluate.hydride.yml`

6. Tests:
- `tests/test_phase18_transformer_backends.py`

## Validation

```bash
PYTHONPATH=. pytest -q
microseg-cli phase-gate --phase-label "Phase 18 Transformer Backends" --strict
```

Result snapshot:
- tests: `55 passed`
- strict phase gate: `pass`
