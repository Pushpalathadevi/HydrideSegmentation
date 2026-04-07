# Phase 4 Orchestration Pane Implementation

Date: 2026-02-16
Branch: `codex/microstructure-foundation-scaffold`

## Scope Delivered

Implemented a centralized orchestration pane for:
- inference jobs
- training jobs
- evaluation jobs
- dataset packaging jobs

GPU runtime policy:
- default execution is CPU
- GPU usage is opt-in
- if GPU runtime is unavailable, jobs automatically fall back to CPU

All jobs execute through `scripts/microseg_cli.py` using YAML configs plus `--set` overrides.

## Backend Additions

- Command builder for reproducible orchestration commands:
  - `src/microseg/app/orchestration.py`
- Baseline CPU training pipeline:
- Baseline torch training pipeline (GPU-compatible):
  - `src/microseg/training/pixel_classifier.py`
  - `src/microseg/training/torch_pixel_classifier.py`
- Baseline evaluation pipeline:
  - `src/microseg/evaluation/pixel_model_eval.py`
- Unified CLI expanded with `train` and `evaluate`:
  - `scripts/microseg_cli.py`

## GUI Integration

Updated workflow hub in:
- `hydride_segmentation/qt/main_window.py`

Features:
- Tabbed orchestration panel (`Inference`, `Training`, `Evaluation`, `Packaging`)
- Asynchronous subprocess execution via `QProcess`
- One-active-job guard
- Live merged stdout/stderr log stream
- Completion/failure dialogs with exit status

## Config Templates

Added defaults:
- `configs/train.default.yml`
- `configs/evaluate.default.yml`

## Validation

- Unit/integration tests added for orchestration and baseline train/eval.
- Test status: `pytest -q` passing on CPU-only (`26 passed`).
