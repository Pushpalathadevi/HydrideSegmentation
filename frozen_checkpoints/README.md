# Frozen Checkpoints Registry

This folder is the canonical in-repo registry for deployable model checkpoints.

Rules:
- Track metadata in `model_registry.json`.
- Keep heavy binary checkpoint files (`.pt`, `.pth`, `.ckpt`, `.onnx`) outside git tracking.
- Use `checkpoint_path_hint` values that point to expected local paths for field deployment.
- Keep lifecycle directories available:
  - `smoke/` tiny debug-only checkpoints
  - `candidates/` local quality-evaluation candidates
  - `promoted/` approved deployment checkpoints (still ignored by git unless policy changes)

The registry is used by:
- Qt GUI model help panel (dynamic model selection guidance)
- CLI model listing (`microseg-cli models`)
- Future model loading and validation workflows

Recommended pattern:
1. Generate a tiny smoke checkpoint for pipeline tests:
   - `python scripts/generate_smoke_checkpoint.py --force`
2. Place candidate/approved checkpoints locally under `frozen_checkpoints/candidates/` or `frozen_checkpoints/promoted/`.
3. Update `model_registry.json` with dimensions, class mapping, lifecycle stage, and usage notes.
4. Keep `short_description` and `detailed_description` focused on user decision support.
