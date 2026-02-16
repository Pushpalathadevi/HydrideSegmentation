# AGENTS.md - Repository Working Contract

This file defines how developers and automation agents should work in this repository.
The long-term mission is microstructural segmentation as a local deployable scientific application.
Hydride segmentation is one concrete target, not the only one.

If a future task conflicts with this document, use this priority order:
1. Safety, correctness, and scientific traceability
2. Reproducibility and non-regression behavior
3. Backward compatibility for existing users
4. Speed of delivery

## 1. Mission Alignment

All changes must align with `docs/mission_statement.md`.
The repository must evolve toward:
- General microstructural feature segmentation (hydrides, grain boundaries, phases, inclusions, cracks, pores, etc.)
- CPU-first local inference workflows
- Human-in-the-loop review and correction
- Export of corrected data for future model retraining

## 2. Development Principles

- Keep architecture modular: domain contracts, pipelines, models, UI, and I/O are separate.
- Keep GUI and core computation decoupled.
- Do not hardcode model weights or paths.
- Keep inference deterministic where possible and seed all stochastic operations.
- Make every pipeline configurable through versioned config files.
- Prefer explicit data contracts over ad hoc dictionaries.

## 3. Scientific Robustness Requirements

- Every metric and transformation must have a documented definition.
- Persist run metadata: model ID, config hash, code version, timestamp, and hardware profile.
- Track uncertainty and failure modes where feasible.
- Keep validation sets isolated from training and correction-derived data.
- Avoid silent fallback behavior for critical scientific steps.

## 4. Coding and Packaging Expectations

- Python 3.10+.
- Type hints for public APIs.
- NumPy docstrings for public functions and classes.
- No import-time side effects in library modules.
- No GUI-only dependencies imported by package top-level modules.
- Keep entry points thin; heavy logic belongs in library modules.

## 5. Testing Expectations

- Unit tests for pure functions and data contracts.
- Integration tests for inference and correction export pipelines.
- End-to-end smoke tests for the desktop app in debug mode.
- CPU-only tests must pass in CI.
- Existing behavior must be captured in regression tests before major rewrites.

## 6. Documentation Sync (Mandatory)

Any behavior change must update docs in the same change.
At minimum update:
- `README.md` for user-facing usage changes
- `docs/` architecture or workflow docs for internal behavior changes
- test docs when validation protocol changes

## 7. Migration Rules

The current `hydride_segmentation` package is the base-zero implementation.
During migration:
- Preserve old entry points until replacements are available.
- Introduce compatibility wrappers where needed.
- Remove legacy paths only after equivalent tests and docs exist.

## 8. Local Deployment Focus

This repository is desktop/local-app-first.
Do not prioritize web deployment or cloud-only assumptions until local CPU workflows are complete and stable.

## 9. What to Avoid

- Monolithic scripts that mix UI, inference, and file I/O.
- Hidden global state for model instances without lifecycle control.
- Unversioned output formats.
- Undocumented algorithmic constants.
- Changes that break reproducibility logging.
