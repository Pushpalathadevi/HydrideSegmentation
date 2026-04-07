# Phase 14 - Checkpoint Lifecycle And Smoke Artifact Baseline

## Scope

This phase formalizes how model checkpoints are handled to prevent repository bloat while preserving reproducibility:
- metadata stays versioned in git
- binary checkpoints remain local by default
- tiny deterministic smoke checkpoint is available for pipeline sanity checks

## Implemented Changes

1. Added smoke-checkpoint generator:
   - `scripts/generate_smoke_checkpoint.py`
   - Produces a tiny random-weight torch pixel-classifier checkpoint (`.pth`) and sidecar metadata JSON.

2. Added lifecycle directory structure:
   - `frozen_checkpoints/smoke/`
   - `frozen_checkpoints/candidates/`
   - `frozen_checkpoints/promoted/`

3. Extended frozen-checkpoint metadata fields (optional):
   - `artifact_stage`
   - `source_run_manifest`
   - `quality_report_path`
   - `file_sha256`
   - `file_size_bytes`

4. Registry validator checks:
   - accepted `artifact_stage` values
   - relative path constraints for source/quality report paths
   - numeric validation for `file_size_bytes`

5. Improved runtime behavior:
   - evaluation now accepts `.pt`, `.pth`, `.ckpt` torch checkpoints
   - hydride ML predictor tries `checkpoint_path_hint` from registry when `weights_path` is not passed and file exists locally

## Usage

Generate tiny smoke checkpoint:

```bash
python scripts/generate_smoke_checkpoint.py --force
```

Validate registry:

```bash
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```

## Notes

- Smoke checkpoint is for code-path validation only and is not scientifically valid for production use.
- Promoted deployment models should be referenced via metadata and validated quality reports before field use.
