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
2. Install real checkpoints with the installer rather than by hand:
   - GUI: `Settings > Installed Models...` then `Install Model...`
   - CLI: `microseg-cli install-model --checkpoint path/to/best_checkpoint.pth --model-id my_unet_v1 --nickname my_unet_v1_optical`
   - The installer copies the file into the lifecycle folder for the chosen stage, verifies it loads and runs, and writes a complete entry into the untracked `model_registry.local.json` overlay.
3. Edit `model_registry.json` directly only for entries that ship with the repository; keep dimensions, class mapping, lifecycle stage, and usage notes accurate.
4. Keep `short_description` and `detailed_description` focused on user decision support.

See `docs/gui_model_integration_guide.md` for the full walkthrough.
