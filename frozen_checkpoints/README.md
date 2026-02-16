# Frozen Checkpoints Registry

This folder is the canonical in-repo registry for deployable model checkpoints.

Rules:
- Track metadata in `model_registry.json`.
- Keep heavy binary checkpoint files (`.pt`, `.pth`, `.ckpt`, `.onnx`) outside git tracking.
- Use `checkpoint_path_hint` values that point to expected local paths for field deployment.

The registry is used by:
- Qt GUI model help panel (dynamic model selection guidance)
- CLI model listing (`microseg-cli models`)
- Future model loading and validation workflows

Recommended pattern:
1. Place checkpoint files locally under `frozen_checkpoints/<model_nickname>/`.
2. Update `model_registry.json` with dimensions, class mapping, and domain usage notes.
3. Keep `short_description` and `detailed_description` focused on user decision support.
