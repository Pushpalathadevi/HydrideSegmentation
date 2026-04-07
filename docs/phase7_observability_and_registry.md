# Phase 7 - Observability And Model Registry Hardening

## Objective

Strengthen the platform foundation for scientific publication and field deployment by improving:
- model metadata governance
- long-run job visibility
- interruption-safe artifacts
- human-readable and machine-consumable reporting

## Implemented In This Phase

1. Frozen checkpoint registry
- Added `frozen_checkpoints/model_registry.json` with schema `microseg.frozen_checkpoint_registry.v1`.
- Added loader utilities in `src/microseg/plugins/frozen_checkpoints.py`.
- Integrated metadata into GUI model descriptions via `hydride_segmentation/microseg_adapter.py`.
- Added CLI model listing command: `microseg-cli models`.

2. UNet training observability
- Extended `UNetBinaryTrainingConfig` with:
  - `val_tracking_samples`
  - `val_tracking_fixed_samples`
  - `val_tracking_seed`
  - `write_html_report`
  - `progress_log_interval_pct`
- Added progress logging with percentage and ETA during epochs.
- Added interruption-safe run reporting:
  - `report.json` (`microseg.training_report.v1`)
  - `epoch_history.jsonl`
  - `events.jsonl`
  - `training_report.html`
- Added per-epoch tracked validation sample exports:
  - fixed examples from config
  - random remainder
  - panel images: input | GT | pred | diff
- `training_report.html` now renders tracked sample panels with per-image metrics and an epoch-by-epoch sample section.

3. Evaluation reporting upgrades
- Extended evaluation config with HTML and tracked samples controls.
- Added progress logging and tracked sample panels.
- Added evaluation HTML summary sidecar report.

4. GUI orchestration controls
- Training tab now supports val tracking counts, fixed names, reporting toggle, and progress interval.
- Evaluation tab now supports tracked sample count/seed and HTML report toggle.

## Artifacts And Schemas

- Training:
  - `training_manifest.json` (`microseg.training_manifest.v2`)
  - `report.json` (`microseg.training_report.v1`)
  - `training_report.html`
  - `eval_samples/epoch_XXX/*.png`
- Evaluation:
  - JSON report (`microseg.pixel_eval.v4`)
  - HTML sidecar report (`.html`)
  - tracked sample panels under sibling `samples/`

## Config Example

```yaml
backend: unet_binary
epochs: 20
val_tracking_samples: 8
val_tracking_fixed_samples:
  - val_000.png
  - val_123.png
val_tracking_seed: 17
write_html_report: true
progress_log_interval_pct: 10
```

GUI override equivalent for fixed names:
- `val_000.png|val_123.png`

## Tests Added

- `tests/test_phase7_frozen_registry.py`
- `tests/test_phase7_training_reporting.py`
