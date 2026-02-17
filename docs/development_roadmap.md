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

## Phase 12 - GUI Dataset Workspace And QA Gate (Implemented)

Goals:
- Bring dataset onboarding and validation fully into the Qt app workflow hub.
- Provide preview-first controls for split planning and RGB mask conversion.
- Enforce optional QA gate before launching training jobs.

Deliverables:
- dataset preview API in `src/microseg/dataops/training_dataset.py`
- Workflow Hub `Dataset Prep + QA` tab in `hydride_segmentation/qt/main_window.py`
- training preflight QA gate and launch blocking on critical dataset failures
- YAML workflow profile save/load for dataset-prepare, training, and evaluation panes
- orchestration builder support for dataset-prepare and dataset-qa command paths

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-16)

Exit criteria:
- users can preview, prepare, QA-check, and gate training from GUI without ad hoc CLI steps

## Phase 13 - Run Review And E2E QA Hardening (Implemented)

Goals:
- Add GUI-native training/evaluation report review and comparison.
- Strengthen testable backend support for report analytics and workflow profile persistence.
- Increase automated coverage for high-value post-run inspection workflows.

Deliverables:
- Run Review tab in `hydride_segmentation/qt/main_window.py`
- report summary/compare module: `src/microseg/app/report_review.py`
- workflow profile module: `src/microseg/app/workflow_profiles.py`
- tests: `tests/test_phase13_report_review.py`

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- report review/comparison and profile roundtrip workflows are code-complete, documented, and test-covered

## Phase 14 - Checkpoint Lifecycle And Smoke Artifact Baseline (Implemented)

Goals:
- Enable tiny deterministic checkpoint generation for quick pipeline debugging on clean machines.
- Formalize checkpoint lifecycle folders without committing heavy binaries.
- Improve ML checkpoint path resolution through metadata hints.

Deliverables:
- smoke checkpoint generator: `scripts/generate_smoke_checkpoint.py`
- lifecycle folders: `frozen_checkpoints/smoke`, `frozen_checkpoints/candidates`, `frozen_checkpoints/promoted`
- registry metadata extension with optional lifecycle/provenance fields
- evaluator support for `.pt`/`.pth`/`.ckpt` torch checkpoint suffixes
- hydride ML predictor auto-resolution of weights via `checkpoint_path_hint` when local file exists

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- local smoke checkpoint can be generated and loaded by evaluation pipeline
- frozen checkpoint registry validates with lifecycle metadata fields
- docs and workflow guidance reflect metadata-first, binary-outside-git policy

## Phase 15 - GA HPC Orchestration Bundle (Implemented)

Goals:
- Provide GUI/CLI-driven generation of HPC job bundles for multi-candidate model sweeps.
- Support architecture/hyperparameter comparisons with deterministic GA-based candidate synthesis.
- Produce one launcher script to submit all jobs on Slurm/PBS/local environments.

Deliverables:
- planner + bundle module: `src/microseg/app/hpc_ga.py`
- CLI command: `microseg-cli hpc-ga-generate`
- GUI Workflow Hub tab: `HPC GA Planner`
- config template: `configs/hpc_ga.default.yml`
- user/dev docs for end-to-end HPC bundle workflow

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- users can generate and inspect scheduler-ready bundle from GUI and CLI
- profile save/load supports `hpc_ga` scope
- tests and docs updated in the same phase change

## Phase 16 - Repository Health Hardening Pass (Implemented)

Goals:
- Run end-to-end health audit over code, tests, and documentation policy.
- Resolve concrete deprecation and dependency risks.
- Publish explicit audit findings and remaining strategic gaps.

Deliverables:
- full test + strict phase-gate validation pass
- API validator modernization (`@field_validator`) and deprecated `imghdr` replacement
- dependency alignment (`pydantic` in `requirements.txt`)
- audit report doc: `docs/repo_health_audit.md`

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- no failing tests
- no known deprecation warnings from repository test run
- audit findings documented with clear next hardening step

## Phase 17 - HPC GA Feedback-Hybrid Optimization (Implemented)

Goals:
- Add feedback-aware candidate planning for later HPC sweep iterations.
- Ingest prior run artifacts and summarize reusable optimization signals.
- Keep GUI/CLI/profile workflows synchronized with new planning controls.

Deliverables:
- feedback-hybrid ranking in planner: `src/microseg/app/hpc_ga.py`
- CLI command: `microseg-cli hpc-ga-feedback-report`
- GUI `HPC GA Planner` updates:
  - fitness mode controls
  - feedback-source controls
  - report generation action (`Analyze Feedback`)
- command builder method: `OrchestrationCommandBuilder.hpc_ga_feedback_report(...)`
- expanded tests for feedback ingestion/hybrid ranking and command builder coverage

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- planner supports `novelty` and `feedback_hybrid` modes
- feedback report command generates JSON + markdown outputs
- GUI workflow profile (`hpc_ga`) roundtrip includes feedback fields
- docs and tests updated in the same phase change

## Phase 18 - Transformer Segmentation Trial Backends (Implemented)

Goals:
- Add transformer-based segmentation backends for hydride trials.
- Keep training/evaluation orchestration compatible with existing pipelines.
- Provide HPC-ready benchmark configs and user-facing trial guidance.

Deliverables:
- training architecture variants in `src/microseg/training/unet_binary.py`:
  - `transunet_tiny`
  - `segformer_mini`
- evaluation compatibility for new torch checkpoint schema in `src/microseg/evaluation/pixel_model_eval.py`
- CLI backend support in `scripts/microseg_cli.py`
- GUI backend selection updates in `hydride_segmentation/qt/main_window.py`
- hydride benchmark config set in `configs/hydride/`
- end-to-end trial matrix table in `docs/hydride_research_workflow.md`

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- transformer variants train and evaluate in regression tests
- existing UNet checkpoint compatibility preserved
- docs include explicit HPC trial matrix and config references

## Phase 19 - SOTA External Transformer Backends (Implemented)

Goals:
- Integrate external, publication-grade transformer segmentation models via pip-installable libraries.
- Guarantee scratch-only initialization for offline HPC workflows (no transfer learning).
- Expand fair-comparison hydride benchmark matrix against U-Net baseline.

Deliverables:
- HF SegFormer backends:
  - `hf_segformer_b0`
  - `hf_segformer_b2`
  - `hf_segformer_b5`
- scratch-only initialization path in `src/microseg/training/unet_binary.py`
- evaluation support for `microseg.hf_transformer_segmentation.v1`
- CLI/GUI/HPC architecture list updates for new backends
- hydride benchmark presets for HF backends in `configs/hydride/`
- workflow table update in `docs/hydride_research_workflow.md`

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- HF transformer backends train/evaluate without downloading pretrained weights
- benchmark matrix documents scratch-only runs and HPC trial guidance
- tests and strict phase-gate pass

## Phase 20 - Top-5 Benchmark Suite Orchestration (Implemented)

Goals:
- Add one-command orchestration for top-5 hydride benchmark runs.
- Produce run-level and aggregate summary artifacts for model comparison.
- Emit a manuscript-friendly dashboard artifact from the same pipeline.

Deliverables:
- suite runner script: `scripts/hydride_benchmark_suite.py`
- top-5 suite config: `configs/hydride/benchmark_suite.top5.yml`
- workflow documentation updates with top-5 rationale/compatibility/metrics contract
- dry-run orchestration regression test: `tests/test_phase20_benchmark_suite_script.py`

Status:
- Implemented on branch `codex/microstructure-foundation-scaffold` (2026-02-17)

Exit criteria:
- script can execute or dry-run full model/seed matrix from one config file
- outputs include JSON, CSV, aggregate CSV, and HTML dashboard
- docs explicitly describe remaining gaps and next hardening targets
