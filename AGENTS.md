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

All changes must align with `docs/mission_statement.md`.
The repository must evolve toward:
- General microstructural feature segmentation (hydrides, grain boundaries, phases, inclusions, cracks, pores, etc.)
- CPU-first local inference workflows
- Human-in-the-loop review and correction
- Export of corrected data for future model retraining / active learning loops
- Field-deployable desktop usage with robust logging and error handling
- GPU-compatible training/inference with safe CPU fallback

## 2. Scope And Product Direction

In-scope (current):
- Local desktop GUI and local CLI workflows
- Inference, correction, export, packaging, training, and evaluation orchestration
- HPC script-bundle generation for offloaded GPU training/evaluation runs
- Config-driven execution (`.yml` + `--set` overrides)
- Session/project save and resume
- Frozen-checkpoint metadata registry for dynamic user guidance

Out-of-scope until explicitly reprioritized:
- Web-first productization
- Cloud-only assumptions
- GPU-required core paths

## 3. Policy Baseline (DeepImageDeconvolution-Inspired)

This repository adopts and extends standards inspired by `kvmani/DeepImageDeconvolution`:
- Thin scripts, heavy library modules
- Debuggable and traceable run artifacts
- Documentation sync in the same change as behavior updates
- Structured logs and machine-readable manifests/reports
- Beginner-friendly user and developer documentation

Where this repository differs:
- Segmentation-centric architecture (not deconvolution)
- Human correction loops with export-ready annotations
- Local desktop deployment with annotation-first UX

## 4. Non-Negotiable Engineering Terms

- Keep architecture modular: domain contracts, pipelines, models, UI, and I/O are separate.
- Keep GUI and core computation decoupled.
- Do not hardcode model weights, class maps, or environment-specific paths.
- Keep inference deterministic where possible and seed stochastic operations.
- Make every pipeline configurable through versioned config files.
- Prefer explicit data contracts over ad hoc dictionaries.
- No silent fallback behavior for critical scientific steps.
- Persist run metadata: model ID, config, code version, timestamp, and hardware profile where feasible.
- Training and inference must be GPU-compatible but CPU-safe:
  - default runtime path is CPU
  - GPU use must be explicit/configurable (or opt-in auto mode)
  - missing GPU runtime must automatically and visibly fall back to CPU

## 5. Observability And Run Reporting Standards

Long-running jobs (training/evaluation/packaging) must:
- Log timestamped progress with counts, percent complete, and ETA
- Persist restart/useful intermediate artifacts during execution
- Write machine-readable report files (`report.json`) for automated summarization
- Emit structured epoch/sample artifacts for human inspection
- Write HTML summaries when practical
- On interruption/failure, preserve partial outputs and write failure context

Validation tracking for training must support:
- `n` tracked validation samples per epoch
- fixed named samples from config
- random sampled remainder for coverage
- stored panels/metrics per tracked sample

## 6. Frozen Checkpoint Registry Standards

`frozen_checkpoints/model_registry.json` is the canonical metadata registry.
Each entry must include:
- model ID and nickname
- model type and framework
- input dimensions and size assumptions
- class index mapping
- checkpoint path hint
- short and detailed user guidance
- application suitability remarks

Binary weights (`.pt`, `.pth`, `.ckpt`, `.onnx`) are not tracked in git.
Model lifecycle folders must be used consistently:
- `frozen_checkpoints/smoke` for tiny debug-only checkpoints
- `frozen_checkpoints/candidates` for non-approved evaluation checkpoints
- `frozen_checkpoints/promoted` for approved deployment checkpoints

## 7. Annotation And Correction Standards

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

## 8. Workflow Configuration Standards

- Inference, correction export, dataset packaging, training, and evaluation must be callable from both GUI and CLI.
- Config strategy is mandatory:
  - YAML base configs
  - `--set` dotted-key overrides
  - GUI editable entries mapped to the same config model
- Split generation must be deterministic with explicit seed control.
- Split generation must include leakage guards (at minimum source-group-aware split constraints).
- Validation/test isolation must be documented and preserved.

## 9. Coding And Packaging Expectations

- Python 3.10+.
- Type hints for public APIs.
- NumPy docstrings for public functions and classes.
- No import-time side effects in library modules.
- No GUI-only dependencies imported by package top-level modules.
- Keep entry points thin; heavy logic belongs in library modules.

## 10. Testing Expectations (Release Gate)

- Unit tests for pure functions and data contracts.
- Integration tests for inference and correction export pipelines.
- End-to-end smoke tests for desktop app in debug mode.
- CPU-only tests must pass in CI.
- Existing behavior must be captured in regression tests before major rewrites.
- Behavior-changing PRs must add or update tests in the same change.

## 10A. Mandatory End-Of-Phase Closeout

At the end of each development phase, the following are mandatory:
- Run full repository tests and ensure all pass.
- Take stock of implemented features vs phase goals.
- Identify and document remaining gaps explicitly.
- Update `README.md`, roadmap/gap docs, and relevant phase docs in the same change.
- Produce a closeout artifact (machine-readable + human-readable summary) for traceability.

## 11. Documentation Sync (Mandatory)

Any behavior change must update docs in the same change.
At minimum update:
- `README.md` for user-facing usage changes
- `docs/` architecture/workflow docs for internal behavior changes
- `tests/README.md` when validation protocol changes

All markdown links must be repository-relative, not absolute local filesystem paths.

## 12. Migration Rules

The current `hydride_segmentation` package is base-zero implementation.
During migration:
- Preserve old entry points until replacements are available.
- Introduce compatibility wrappers where needed.
- Remove legacy paths only after equivalent tests and docs exist.

## 13. Local Deployment Focus

This repository is desktop/local-app-first.
Do not prioritize web deployment or cloud-only assumptions until local CPU workflows are complete and stable.

## 14. What To Avoid

- Monolithic scripts that mix UI, inference, and file I/O.
- Hidden global state for model instances without lifecycle control.
- Unversioned output formats.
- Undocumented algorithmic constants.
- Changes that break reproducibility logging.
- New features without corresponding user/developer documentation.

## 15. Documentation And Scientific Traceability Requirements

- Any change that modifies a segmentation stage, model backend, default parameter, output schema, or registry entry must update the relevant documentation in the same change.
- Flow sheets, schematics, and publication figures in user-facing docs must be committed as static SVG assets under `docs/diagrams/`; inline Mermaid blocks should be treated as temporary authoring artifacts only.
- Classical segmentation docs must include:
  - an end-to-end flow sheet,
  - a parameter table,
  - typical values and valid ranges,
  - the scientific purpose of each stage,
  - failure modes and tuning guidance.
- Model-family docs must include:
  - the architecture family and what is internal versus externally published,
  - the original publication citation,
  - a comparison table across model families,
  - the main performance factors,
  - a clear note when a backend is an internal variant rather than an official reproduction.
- When legacy and modern code paths disagree on defaults, documentation must name both and identify the canonical runtime path.
- New user-facing workflows should be accompanied by a beginner or on-ramp guide and added to the docs index.
- If a docs change alters navigation or entry points, update `docs/index.md`, `docs/README.md`, and any user-facing top-level references in the same change.
- Keep `docs/documentation_principles.md` synchronized with this contract; if the contract changes, update both files together.
