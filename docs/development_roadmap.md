# Phase-Wise Development Roadmap

Global phase-close rule (mandatory for every phase):
- run full tests and ensure pass
- run phase-gate closeout checks
- publish stocktake + gap review
- synchronize README/roadmap/phase docs in the same change

## Phase 0 - Baseline Freeze and Safety Net

Goals:
- Preserve current hydride behavior and interfaces
- Add regression tests around current outputs and APIs

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-15)

Deliverables:
- Baseline audit complete (`docs/base_zero_audit.md`)
- Non-GUI import path fixed for tests
- Snapshot tests for representative images and metrics

Exit criteria:
- Existing user flows run unchanged
- CI passes in CPU-only environment

## Phase 1 - Core Architecture Extraction

Goals:
- Introduce model-agnostic core contracts
- Decouple GUI, inference, and analysis modules

Deliverables:
- `src/microseg` core scaffolding
- predictor/analyzer/exporter interfaces
- compatibility adapter for existing `hydride_segmentation`

Status:
- Implemented baseline extraction on branch `codex/microstructure-foundation-scaffold` (2026-02-15)
- Added model registry, contracts, hydride adapters, and pipeline orchestration
- GUI segmentation path now routes through the compatibility adapter

Exit criteria:
- New core can execute hydride inference via adapter
- no duplicate logic for orchestration paths

## Phase 2 - Desktop App Refactor

Goals:
- Build a robust local app workflow around the new core

Deliverables:
- model registry UI
- batch run manager
- improved result browsing and export UX

Status:
- Implemented baseline desktop refactor on branch `codex/microstructure-foundation-scaffold` (2026-02-15)
- GUI model options now sourced from registry metadata
- Batch run flow and run-history browsing added
- Save results now exports structured run package with manifest and metrics

Exit criteria:
- end-to-end local inference + analysis on CPU for hydride model

## Phase 3 - Human Correction and Data Export Loop

Goals:
- Add manual correction and retraining dataset export

Deliverables:
- correction session model and edit tools
- versioned correction export schema
- dataset packaging CLI

Status:
- Implemented baseline correction/export loop on branch `codex/microstructure-foundation-scaffold` (2026-02-15)
- Added correction session with undo/redo and brush/polygon edits
- Added schema version `microseg.correction.v1` and corrected sample export
- Added deterministic correction dataset packaging CLI
- Added Qt GUI foundation for correction workflow
- Added advanced correction UX: polygon/lasso tools, split-view synchronized zoom/pan, layer transparency controls, and keyboard shortcuts
- Added connected-feature deletion/relabel workflow for correcting wrong segmented objects
- Added class index/color map editing and class-aware annotation overlays
- Added export format options: indexed PNG, color PNG, NumPy mask

Exit criteria:
- corrected outputs export to training-ready folder layout with manifests

## Phase 4 - Training and Active Improvement Pipeline

Goals:
- unify training, fine-tuning, and iterative data growth

Deliverables:
- standardized training pipeline
- experiment tracking metadata
- retraining guide from correction datasets
- config-first orchestration (`.yml` + `--set`) across GUI and CLI
- restartable session persistence for iterative annotation projects

Status:
- Foundation scaffolding implemented in v0.9.0:
  - YAML configuration + `--set` merge engine
  - unified CLI (`microseg-cli`) for inference and dataset packaging
  - project session save/load schema `microseg.project.v1`
  - GUI workflow hub for split packaging control
- Orchestration implementation expanded:
  - GUI orchestration tabs for inference/training/evaluation/packaging
  - asynchronous subprocess job execution with live logs
  - `microseg-cli` extended with `train` and `evaluate` subcommands
  - baseline CPU pixel-classifier training/evaluation pipeline
  - opt-in GPU runtime support for training/inference/evaluation with CPU fallback
  - UNet binary training backend with early stopping/checkpoint/resume

Exit criteria:
- documented and testable loop: infer -> correct -> export -> train -> deploy

## Phase 5 - Packaging and Deployment Hardening

Goals:
- local installable application distribution

Deliverables:
- reproducible build scripts
- installation guides per OS
- operational diagnostics and support docs
- semantic versioning policy and release note discipline
- GPU-runtime deployment guidance for Windows/HPC while preserving CPU default

Exit criteria:
- install and run on clean target machines with CPU-only setup

## Phase 6 - UNet Backend Expansion (Completed)

Goals:
- Add a trainable torch UNet baseline backend with checkpoint lifecycle controls.

Deliverables:
- `unet_binary` backend with early stopping, periodic checkpointing, and resume support
- GUI/CLI orchestration integration for UNet training/evaluation
- GPU compatibility with CPU fallback for runtime portability

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-16)

Exit criteria:
- reproducible UNet train/eval loop with test coverage and docs

## Phase 7 - Observability, Registry, and Reporting Hardening (Implemented)

Goals:
- Make long-running jobs interruption-safe and deeply inspectable.
- Add dynamic model metadata guidance for GUI and CLI model selection.
- Enforce repository-relative documentation links and publication-ready reporting quality.

Deliverables:
- frozen checkpoint metadata registry (`frozen_checkpoints/model_registry.json`)
- GUI/CLI model help integration from registry metadata
- training `report.json` + `training_report.html` with progress/ETA and tracked val sample panels
- evaluation JSON + HTML reports with tracked sample outputs
- fixed + random validation sample tracking controls in YAML/CLI/GUI

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-16)

Exit criteria:
- training/evaluation runs provide machine-readable summaries and human-readable visual reports
- interruption leaves usable partial artifacts and status reports
- docs and AGENTS policies synchronized with implementation

## Phase 8 - Phase Gate And Quality Governance Automation (Implemented)

Goals:
- Enforce end-of-phase closure discipline with reproducible checks and artifacts.
- Make test pass, gap stocktaking, and docs synchronization explicit and auditable.

Deliverables:
- phase-gate checker module: `src/microseg/quality/phase_gate.py`
- script wrapper: `scripts/run_phase_gate.py`
- CLI integration: `microseg-cli phase-gate`
- default config template: `configs/phase_gate.default.yml`
- policy/workflow docs updated with mandatory closeout steps

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-16)

Exit criteria:
- phase completion now includes explicit test pass + stocktake + gap + docs gate
- machine-readable and markdown closeout artifacts are generated

## Phase 9 - Model Lifecycle And Dataset Operations Foundation (Implemented)

Goals:
- Enforce strict frozen-model metadata quality for safe model selection.
- Add deterministic leakage-aware dataset split planning for correction exports.
- Add dataset QA checks for missing pairs, dimensional mismatches, duplicates, and imbalance warnings.

Deliverables:
- strict registry validator: `src/microseg/plugins/registry_validation.py`
- dataset split planner: `src/microseg/dataops/split_planner.py`
- dataset QA module: `src/microseg/dataops/quality.py`
- CLI commands:
  - `microseg-cli validate-registry`
  - `microseg-cli dataset-split`
  - `microseg-cli dataset-qa`
- config templates:
  - `configs/registry_validation.default.yml`
  - `configs/dataset_split.default.yml`
  - `configs/dataset_qa.default.yml`

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-16)

Exit criteria:
- registry validation and dataset operations are test-covered, documented, and runnable from CLI

## Phase 10 - Training Dataset Auto-Prepare Contract (Implemented)

Goals:
- Support both explicit split datasets and unsplit source/masks datasets.
- Auto-generate default `80:10:10` split when split folders are absent.
- Preserve original stem names while adding programmatic `_id` suffixes.

Deliverables:
- dataset preparation module: `src/microseg/dataops/training_dataset.py`
- CLI command: `microseg-cli dataset-prepare`
- train/evaluate integration with auto-prepare defaults
- training data requirements documentation

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-16)

Exit criteria:
- train/eval pipelines accept unsplit datasets and produce deterministic split-ready layouts

## Phase 11 - Dataset Policy Alignment (Implemented)

Goals:
- Make unsplit auto-prepare leakage-aware by default.
- Support optional RGB color-mask conversion via configurable colormap.
- Enforce globally unique prepared IDs for consistent sample tracking.

Deliverables:
- extended dataset-prepare contracts in `src/microseg/dataops/training_dataset.py`
- CLI/config support in `scripts/microseg_cli.py` and `configs/*.default.yml`
- expanded dataset-prepare tests for leakage-aware grouping and RGB conversion
- updated user/developer docs for training data requirements and config workflow

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-16)

Exit criteria:
- dataset auto-prepare policy decisions are code-complete, test-covered, and documented
