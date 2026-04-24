# Frozen Checkpoint Registry Guide

## Purpose

`frozen_checkpoints/model_registry.json` is the single source of truth for frozen model metadata used in:
- GUI model guidance
- CLI model listing/help
- future model loading validation and compatibility checks

It stores metadata only. Model binaries stay outside git tracking.
Committed registry entries must use repository-relative paths under `frozen_checkpoints/`.
The default trained hydride checkpoint is resolved from:

```text
frozen_checkpoints/candidates/U_net_binary_best_checkpoint.pt
```

Absolute checkpoint paths are not valid in the committed registry. They are only appropriate for explicit direct runtime overrides such as `params.checkpoint_path` / `params.weights_path`, or for machine-local experiments in `frozen_checkpoints/model_registry.local.json`.

The loader also reads an optional local overlay file:

- `frozen_checkpoints/model_registry.local.json`

Use the overlay for machine-specific additions when you do not want to commit the model entry yet. The overlay is merged on top of the canonical registry at runtime, so the GUI and CLI see the same model list without changing source control history.

## Required Metadata Fields

Each model entry must include:
- `model_id`
- `model_nickname`
- `model_type`
- `framework`
- `input_size`
- `input_dimensions`
- `checkpoint_path_hint`
- `application_remarks`

Recommended additional fields:
- `short_description`
- `detailed_description`
- `classes` (index/name/color)
- `artifact_stage` (`smoke`, `candidate`, `promoted`, `builtin`, `deprecated`)
- `source_run_manifest` (repo-relative path when applicable)
- `quality_report_path` (repo-relative path when applicable)
- `file_sha256`
- `file_size_bytes`

## How Discovery Works

The app discovers trained models from two sources:

1. successful run folders under `outputs/runs/`
2. frozen registry entries from `frozen_checkpoints/model_registry.json` and the optional `.local` overlay

Each discovered model can appear in:

- the GUI model selector,
- `microseg-cli infer --model ...`,
- `microseg-cli models --details`

The default trained hydride model in this repository is `hydride_ml`, which maps to `Hydride ML (UNet)` in the GUI.

## Adding A New Trained Model

If you trained a new binary UNet checkpoint named `unet_binary`, add a registry entry like this:

```json
{
  "model_id": "my_unet_v2",
  "model_nickname": "my_unet_v2_optical",
  "model_type": "unet_binary",
  "framework": "pytorch",
  "input_size": "variable",
  "input_dimensions": "H x W x 3",
  "checkpoint_path_hint": "frozen_checkpoints/candidates/my_unet_v2/best_checkpoint.pt",
  "application_remarks": "Binary hydride segmentation checkpoint for optical microscopy.",
  "short_description": "Local candidate checkpoint for GUI and CLI inference.",
  "detailed_description": "Use the same preprocessing and class map that were used during training.",
  "artifact_stage": "candidate",
  "source_run_manifest": "outputs/training/my_unet_v2/training_manifest.json",
  "quality_report_path": "outputs/training/my_unet_v2/report.json",
  "file_sha256": "PUT_THE_CHECKSUM_HERE",
  "file_size_bytes": 12345678,
  "classes": [
    { "index": 0, "name": "background", "color_hex": "#000000" },
    { "index": 1, "name": "hydride", "color_hex": "#00FFFF" }
  ]
}
```

Edit the following file(s):

- `frozen_checkpoints/model_registry.json` for a repository-wide committed model
- `frozen_checkpoints/model_registry.local.json` for a machine-local model

Then restart the GUI or rerun the CLI. The selector order is:

1. discovered trained model `hydride_ml` first
2. additional discovered trained models
3. `Hydride Conventional` last as the fallback

## Beginner Checklist

Before you hand a model to another machine, verify:

- the checkpoint file exists at `checkpoint_path_hint`
- `checkpoint_path_hint` is repo-relative, not an absolute local path
- `model_type` is exactly one of the supported loader tokens
- the class indices and colors match training
- the SHA-256 checksum matches the packaged artifact
- the model appears in `microseg-cli models --details`

## Typical Update Flow

1. For smoke tests, generate a tiny deterministic local checkpoint:
```bash
python scripts/generate_smoke_checkpoint.py --force
```
2. Add or update local checkpoint file under lifecycle folders (`frozen_checkpoints/smoke`, `frozen_checkpoints/candidates`, `frozen_checkpoints/promoted`), outside git tracking.
3. Update `frozen_checkpoints/model_registry.json`.
   The canonical default trained hydride model in this repo is `hydride_ml`, which currently points at `frozen_checkpoints/candidates/U_net_binary_best_checkpoint.pt`. Keep the binary itself out of git, but keep its metadata entry committed.
4. Verify with:
```bash
microseg-cli models --details
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```
5. Open GUI and confirm model help text updates.

For a beginner-friendly, air-gapped walkthrough with concrete examples, see [`docs/gui_model_integration_guide.md`](gui_model_integration_guide.md).

## Design Rules

- Do not hardcode absolute local filesystem paths.
- Keep committed `checkpoint_path_hint` values repository-relative under `frozen_checkpoints/`; strict registry validation rejects absolute paths.
- Keep `artifact_stage` and `checkpoint_path_hint` folder aligned:
  - `smoke` -> `frozen_checkpoints/smoke/...`
  - `candidate` -> `frozen_checkpoints/candidates/...`
  - `promoted` -> `frozen_checkpoints/promoted/...`
- Keep `source_run_manifest` and `quality_report_path` repository-relative when provided.
- Keep descriptions concise, user-oriented, and scientifically specific.
- Keep metadata stable and versioned for reproducibility.
