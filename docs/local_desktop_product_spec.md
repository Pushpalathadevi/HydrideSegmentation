# Local Desktop Product Specification

## Product Scope

A local installable desktop application for microstructural segmentation workflows.

## Primary User Workflow

1. Load one or more microscopy images.
2. Select a segmentation model from a registry.
3. Run inference locally.
4. Inspect segmentation and quantitative analysis.
5. Apply manual corrections where needed.
6. Export final results and optional training-ready correction datasets.

## Required Functional Areas

1. Model management
- Discover registered models and expected input formats
- Validate model files and metadata before inference

2. Inference
- Single-image and batch processing
- Progress, runtime, and failure visibility

3. Analysis
- Feature counts, area fraction, size and orientation distributions
- Pluggable analysis modules by feature type

4. Correction UI
- Brush/eraser, polygon, lasso, connected-feature delete/relabel
- Class index/color map editing
- Session save/load and intermediate state recovery
- Overlay controls and comparison view (raw, predicted, corrected)

5. Export
- Results export (images, masks, metrics, reports)
- Correction export for retraining (image + corrected mask + provenance)
- Multi-format mask outputs (indexed PNG, color PNG, NumPy)

6. Configuration and Orchestration
- YAML parameter files for inference/packaging pipelines
- `--set` runtime overrides for reproducible command-line runs
- GUI config entries mapping to same config semantics
- dataset onboarding workspace with split preview, optional RGB colormap conversion, and QA checks

## Non-Functional Requirements

- CPU-first runtime path
- GPU-compatible runtime for train/infer/eval with explicit enable switch and fallback
- Offline usability
- Reproducible outputs
- Cross-platform packaging target (macOS, Linux, Windows)

## Current Implementation Snapshot (Phase 12)

Implemented now:
- Registry-backed model selector in GUI
- Single and batch local segmentation execution
- Run history list for result browsing
- Structured result export package with manifest + metrics
- Correction UI with brush, polygon, and lasso workflows
- Feature-select correction tool for delete/relabel of wrong connected objects
- Class index selector and class map editor (index/name/color)
- Split-view correction workspace with synchronized pan/zoom
- Layer toggles and transparency controls (predicted/corrected/diff)
- Keyboard shortcut set for high-throughput annotation
- Session save/load for restartable projects
- Workflow hub tab for dataset split packaging
- Workflow hub orchestration tabs for inference/training/evaluation/packaging jobs
- Unified `microseg-cli` supporting YAML + `--set` flow
- UNet training backend with early stopping/checkpoint/resume controls
- Validation sample tracking controls for training (fixed + random)
- Training/evaluation JSON + HTML report generation
- Dynamic model help from frozen checkpoint metadata registry
- Phase-gate closeout automation for test pass + stocktake + docs synchronization
- Leakage-aware correction split planner and packaged dataset QA checks
- Deterministic unsplit source/masks to train/val/test preparation with ID-mapped filenames
- Workflow Hub Dataset Prep + QA tab with searchable preview table and class histogram summary
- Optional training preflight gate to block launches when dataset QA fails
- YAML workflow profile save/load for dataset-prepare, training, and evaluation panes

Pending:
- Advanced correction ergonomics (shape libraries, smarter snapping, uncertainty-driven guidance)
- Active-learning policy integration over exported corrections
- Installer-grade packaging and signed release workflows

## GUI Framework Decision

- Primary GUI framework: Qt (`PySide6`)
- Rationale: richer annotation canvas and scalable desktop architecture for correction-heavy workflows.
- Tkinter path remains as compatibility fallback during migration.
