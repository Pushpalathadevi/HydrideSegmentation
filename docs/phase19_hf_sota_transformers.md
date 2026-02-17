# Phase 19 - SOTA External Transformer Integration (Scratch-Only)

## Goals

- Integrate state-of-the-art transformer segmentation backends through maintained external libraries.
- Ensure strict scratch initialization (no transfer learning, no pretrained weight downloads).
- Enable fair comparisons against U-Net on identical fixed hydride splits.

## Implemented

1. Hugging Face SegFormer backends:
- `hf_segformer_b0`
- `hf_segformer_b2`
- `hf_segformer_b5`

2. Training/evaluation integration:
- `src/microseg/training/unet_binary.py`
  - architecture factory includes HF SegFormer variants
  - checkpoints include `model_initialization` metadata
  - HF variants initialize from architecture configs only (scratch)
- `src/microseg/evaluation/pixel_model_eval.py`
  - supports `microseg.hf_transformer_segmentation.v1`
  - propagates model initialization metadata in reports

3. CLI/GUI/HPC updates:
- `scripts/microseg_cli.py` backend choices include HF SegFormer variants
- `hydride_segmentation/qt/main_window.py` training backend dropdown includes HF variants
- `configs/hpc_ga.default.yml` default architectures include HF SegFormer options

4. Hydride benchmark config presets:
- `configs/hydride/train.hf_segformer_b0_scratch.yml`
- `configs/hydride/train.hf_segformer_b2_scratch.yml`
- `configs/hydride/train.hf_segformer_b5_scratch.yml`

5. Dependency updates:
- `requirements.txt`
- `pyproject.toml`

6. Tests:
- `tests/test_phase19_hf_transformer_backends.py`

## Validation

```bash
PYTHONPATH=. pytest -q
microseg-cli phase-gate --phase-label "Phase 19 HF SOTA Transformers" --strict
```
