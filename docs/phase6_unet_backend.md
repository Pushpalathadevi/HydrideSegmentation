# Phase 6 - UNet Training Backend

Note: observability/reporting extensions after this phase are documented in `docs/phase7_observability_and_registry.md`.

Date: 2026-02-16
Branch: `codex/microstructure-foundation-scaffold`

## Objective

Add a full UNet-style segmentation training backend with production-grade lifecycle controls:
- checkpointing
- early stopping
- resume from checkpoint
- GPU-compatible runtime with CPU fallback

## Implemented

- UNet training module:
  - `src/microseg/training/unet_binary.py`
- Features:
  - train/val split usage from packaged dataset layout
  - BCEWithLogits loss
  - train/val IoU tracking
  - `best_checkpoint.pt` + `last_checkpoint.pt` + periodic epoch checkpoints
  - resume support via checkpoint path
  - training manifest with run history

## Integration

- CLI `train` now supports backend `unet_binary`:
  - `scripts/microseg_cli.py`
- CLI `evaluate` automatically detects UNet checkpoint schema and evaluates appropriately.
- GUI workflow hub training tab includes backend and UNet lifecycle controls.

## Validation

- Added tests:
  - `tests/test_phase6_unet_training.py`
- Full suite passing in CPU-only environment.
