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

## Typical Update Flow

1. Add local checkpoint file outside git tracking.
2. Update `frozen_checkpoints/model_registry.json`.
3. Verify with:
```bash
microseg-cli models --details
```
4. Open GUI and confirm model help text updates.

## Design Rules

- Do not hardcode absolute local filesystem paths.
- Keep `checkpoint_path_hint` repository-relative.
- Keep descriptions concise, user-oriented, and scientifically specific.
- Keep metadata stable and versioned for reproducibility.
