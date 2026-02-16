# AGENTS.md - Repository Working Contract

This file defines how developers and automation agents must work in this repository.
The long-term mission is microstructural segmentation as a local deployable scientific application.
Hydride segmentation is one validated target, not the platform limit.

If a future task conflicts with this document, use this priority order:
1. Safety, correctness, and scientific traceability
2. Reproducibility and non-regression behavior
3. Backward compatibility for existing users
4. Speed of delivery

## 1. Mission Alignment (Mandatory)

All changes must align with `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/mission_statement.md`.
The repository must evolve toward:
- General microstructural feature segmentation (hydrides, grain boundaries, phases, inclusions, cracks, pores, etc.)
- CPU-first local inference workflows
- Human-in-the-loop review and correction
- Export of corrected data for future model retraining / active learning loops
- Field-deployable desktop usage with robust logging and error handling

## 2. Scope And Product Direction

In-scope (current):
- Local desktop GUI and local CLI workflows
- Inference, correction, export, packaging, evaluation orchestration
- Config-driven execution (`.yml` + `--set` overrides)
- Session/project save and resume

Out-of-scope until explicitly reprioritized:
- Web-first productization
- Cloud-only assumptions
- GPU-required core paths

## 3. Non-Negotiable Engineering Terms

- Keep architecture modular: domain contracts, pipelines, models, UI, and I/O are separate.
- Keep GUI and core computation decoupled.
- Do not hardcode model weights, class maps, or environment-specific paths.
- Keep inference deterministic where possible and seed stochastic operations.
- Make every pipeline configurable through versioned config files.
- Prefer explicit data contracts over ad hoc dictionaries.
- No silent fallback behavior for critical scientific steps.
- Persist run metadata: model ID, config, code version, timestamp, and hardware profile where feasible.

## 4. Annotation And Correction Standards

The correction workflow must support:
- Deleting wrongly segmented connected features
- Redrawing corrected annotations with multiple tools
- Class index and color mapping (background=0, feature classes >=1)
- Export in indexed and colorized forms
- Undo/redo, zoom, layered transparency, synchronized view for inspection efficiency

Correction exports are scientific artifacts and must include:
- Versioned schema
- Source run linkage
- Class map used during correction
- Annotator + timestamp provenance

## 5. Training/Inference/Evaluation Workflow Standards

- Inference, correction export, and dataset packaging must be callable from both GUI and CLI.
- Config strategy is mandatory:
  - YAML base configs
  - `--set` dotted-key overrides
  - GUI editable entries that map to same config model
- Split generation must be deterministic with explicit seed control.
- Validation/test isolation must be documented and preserved.

## 6. Coding And Packaging Expectations

- Python 3.10+.
- Type hints for public APIs.
- NumPy docstrings for public functions and classes.
- No import-time side effects in library modules.
- No GUI-only dependencies imported by package top-level modules.
- Keep entry points thin; heavy logic belongs in library modules.

## 7. Testing Expectations (Release Gate)

- Unit tests for pure functions and data contracts.
- Integration tests for inference and correction export pipelines.
- End-to-end smoke tests for desktop app in debug mode.
- CPU-only tests must pass in CI.
- Existing behavior must be captured in regression tests before major rewrites.
- Behavior-changing PRs must add or update tests in the same change.

## 8. Documentation Sync (Mandatory)

Any behavior change must update docs in the same change.
At minimum update:
- `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/README.md` for user-facing usage changes
- `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/` architecture/workflow docs for internal behavior changes
- `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/tests/README.md` when validation protocol changes

## 9. Migration Rules

The current `hydride_segmentation` package is base-zero implementation.
During migration:
- Preserve old entry points until replacements are available.
- Introduce compatibility wrappers where needed.
- Remove legacy paths only after equivalent tests and docs exist.

## 10. Local Deployment Focus

This repository is desktop/local-app-first.
Do not prioritize web deployment or cloud-only assumptions until local CPU workflows are complete and stable.

## 11. What To Avoid

- Monolithic scripts that mix UI, inference, and file I/O.
- Hidden global state for model instances without lifecycle control.
- Unversioned output formats.
- Undocumented algorithmic constants.
- Changes that break reproducibility logging.
- New features without corresponding user/developer documentation.
