# Phase 34 Closeout: Local Model Installation

## Goal and delivered behavior

Trained checkpoint binaries are not tracked in git, so a fresh clone or a newly deployed air-gapped machine has no `.pt` file. Making a model usable previously required copying the binary by hand, hand-writing a registry overlay, hand-computing a checksum, and knowing the exact loader architecture token. Phase 34 replaces that with one mechanical operation available from both the GUI and the CLI.

Delivered:

- `src/microseg/plugins/model_install.py`, a GUI-free module providing checkpoint introspection, runtime verification, install, uninstall, and catalog status.
- Checkpoint introspection that recovers architecture, backend, training config, input size, parameter count, SHA-256, size, training timestamp, epoch, and best validation loss from the file itself.
- Install that copies the checkpoint into the lifecycle folder matching its artifact stage, verifies it with a real CPU load and one synthetic forward pass, records it in `frozen_checkpoints/model_registry.local.json`, and validates the merged registry.
- Full rollback of both the copied file and the registry overlay when any step fails.
- `Settings > Installed Models...` in the Qt desktop app: availability table, install form pre-filled from introspection, re-verify, remove with an explicit choice about deleting the binary, and open-folder.
- Live catalog reload, so an installed model becomes selectable without restarting the application.
- CLI parity through `inspect-checkpoint`, `install-model`, and `uninstall-model`, sharing the same module.
- Machine-readable install reports under `outputs/model_install/<model_id>/install_report.json`.

Availability corrections shipped in the same change:

- `microseg-cli models` now reports an availability line per model.
- GUI selector entries whose checkpoint is missing or whose architecture is unsupported are disabled, carry an explanatory tooltip, and show an `Unavailable` notice in the guidance panel.
- Running an unavailable model is refused with an explanation instead of failing mid-run.
- `DesktopWorkflowManager.preferred_default_model_name` previously fell back to `specs[0]` rather than the conventional entry, so a machine with no checkpoint started on an unusable ML model. It now resolves the conventional entry by identifier and skips models reported as unavailable.

The canonical `frozen_checkpoints/model_registry.json` is never written by the installer. Only the untracked local overlay is modified.

## Verification and traceability

- New test module: `tests/test_phase34_model_install.py` (18 tests), covering introspection, non-checkpoint rejection, runtime verification, install, stage-folder placement, duplicate and reserved and malformed identifiers, verification rollback, unsupported architecture rejection, uninstall with and without file deletion, catalog status, canonical-registry immutability, install reports, GUI availability metadata, conventional fallback selection, Qt selector disabling, and Qt install-form prefill.
- Full repository suite: 214 passed.
- User documentation: [`gui_model_integration_guide.md`](gui_model_integration_guide.md), rewritten around the installer.
- Registry reference: [`frozen_checkpoint_registry.md`](frozen_checkpoint_registry.md).
- Machine-readable closeout: `docs/phase34_model_installation.report.json`.

## Remaining gaps

- Installed checkpoints must live inside the repository tree because registry hints are repository-relative; the packaged Windows build has no writable repository root, so a user-data models root is still required for the frozen executable.
- Install runs synchronously behind a wait cursor rather than on the background worker used by inference; very large checkpoints will block the dialog while loading.
- The class map cannot be recovered from a checkpoint and still has to be confirmed by the user for non-binary models.
- `hydride_ml_Unet` remains a compatibility alias pointing at the same checkpoint as `hydride_ml`, so it appears as a second selector entry.
- Pre-existing scikit-image morphology deprecation warnings remain outside this phase.
