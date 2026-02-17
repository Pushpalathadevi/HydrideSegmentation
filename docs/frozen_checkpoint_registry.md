# Frozen Checkpoint Registry Guide

## Purpose

`frozen_checkpoints/model_registry.json` is the single source of truth for frozen model metadata used in:
- GUI model guidance
- CLI model listing/help
- future model loading validation and compatibility checks

It stores metadata only. Model binaries stay outside git tracking.

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

## Typical Update Flow

1. For smoke tests, generate a tiny deterministic local checkpoint:
```bash
python scripts/generate_smoke_checkpoint.py --force
```
2. Add or update local checkpoint file under lifecycle folders (`frozen_checkpoints/smoke`, `frozen_checkpoints/candidates`, `frozen_checkpoints/promoted`), outside git tracking.
3. Update `frozen_checkpoints/model_registry.json`.
4. Verify with:
```bash
microseg-cli models --details
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```
5. Open GUI and confirm model help text updates.

## Design Rules

- Do not hardcode absolute local filesystem paths.
- Keep `checkpoint_path_hint` repository-relative.
- Keep `source_run_manifest` and `quality_report_path` repository-relative when provided.
- Keep descriptions concise, user-oriented, and scientifically specific.
- Keep metadata stable and versioned for reproducibility.
