# Phase 5 - GPU-Compatible Runtime (CPU-First)

Date: 2026-02-16
Branch: `codex/microstructure-foundation-scaffold`

## Objective

Enable GPU-compatible training and inference while keeping CPU as the default runtime path.

## Implemented

- Central runtime device resolver with automatic CPU fallback:
  - `src/microseg/core/device.py`
- Hydride ML inference now supports device controls:
  - `enable_gpu` + `device_policy` (`cpu|auto|cuda|mps`)
  - falls back to CPU if GPU runtime unavailable
- Torch-based baseline training pipeline:
  - `src/microseg/training/torch_pixel_classifier.py`
- Evaluation supports both sklearn and torch baseline artifacts:
  - `src/microseg/evaluation/pixel_model_eval.py`
- Unified CLI GPU switches added:
  - `microseg-cli infer/train/evaluate --enable-gpu --device-policy auto`
- GUI Workflow Hub now exposes per-job GPU controls for inference/training/evaluation.

## Runtime Policy

- Default behavior: CPU
- GPU usage: opt-in (`--enable-gpu` / GUI checkbox)
- If selected GPU backend is unavailable, runtime logs reason and falls back to CPU.

## Validation

- Added tests:
  - `tests/test_phase5_gpu_runtime.py`
- Full test suite passing on CPU-only environment.
