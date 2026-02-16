# Phase-Wise Development Roadmap

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
- Foundation scaffolding implemented in v0.6.0:
  - YAML configuration + `--set` merge engine
  - unified CLI (`microseg-cli`) for inference and dataset packaging
  - project session save/load schema `microseg.project.v1`
  - GUI workflow hub for split packaging control

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

Exit criteria:
- install and run on clean target machines with CPU-only setup
