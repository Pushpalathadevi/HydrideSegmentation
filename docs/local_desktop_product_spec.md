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
- Brush/eraser, contour edits, undo/redo
- Overlay controls and comparison view (raw, predicted, corrected)

5. Export
- Results export (images, masks, metrics, reports)
- Correction export for retraining (image + corrected mask + provenance)

## Non-Functional Requirements

- CPU-first runtime path
- Offline usability
- Reproducible outputs
- Cross-platform packaging target (macOS, Linux, Windows)

## Current Implementation Snapshot (Phase 3)

Implemented now:
- Registry-backed model selector in GUI
- Single and batch local segmentation execution
- Run history list for result browsing
- Structured result export package with manifest + metrics
- Correction UI with brush, polygon, and lasso workflows
- Split-view correction workspace with synchronized pan/zoom
- Layer toggles and transparency controls (predicted/corrected/diff)
- Keyboard shortcut set for high-throughput annotation

Pending:
- Advanced correction ergonomics (shape libraries, smarter snapping, uncertainty-driven guidance)
- Active-learning policy integration over exported corrections
- Installer-grade packaging and signed release workflows

## GUI Framework Decision

- Primary GUI framework: Qt (`PySide6`)
- Rationale: richer annotation canvas and scalable desktop architecture for correction-heavy workflows.
- Tkinter path remains as compatibility fallback during migration.
