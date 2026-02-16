# Target Architecture (Microstructural Segmentation Platform)

## Architectural Goals

- Domain-first, model-agnostic segmentation platform
- Local desktop app + reusable backend library
- Explicit contracts for inference, correction, metrics, and exports
- CPU-first reliability with optional acceleration

## Proposed Logical Layers

1. Domain layer
- Data contracts: image sample, mask, feature class, run metadata, correction event
- Validation and schema versioning

2. Pipeline layer
- Inference orchestration
- Post-processing and measurement extraction
- Batch processing and job manifests

3. Model layer
- Model registry and adapters
- Weight resolution and lifecycle management
- Standardized predictor interface

4. Correction layer
- Interactive correction objects (paint, erase, polygon, lasso, connected-component delete/relabel)
- Class-index and color-map contracts
- Change logs and reversible history
- Quality flags and reviewer attribution

5. Training data export layer
- Convert reviewed outputs into train/val/test-ready datasets
- Store correction provenance and split policies

6. Application layer
- Desktop GUI: upload, model selection, run, inspect, correct, export
- Headless CLI for automation and batch workflows
- Project/session persistence for resume workflows
- Workflow hub for inference/correction/packaging orchestration

7. Configuration layer
- YAML config loading
- CLI and GUI override reconciliation (`--set` semantics)
- Persisted resolved-config artifacts for reproducibility

## Canonical Interfaces (to implement)

- `Predictor.predict(image) -> SegmentationResult`
- `PostProcessor.apply(mask, config) -> mask`
- `Analyzer.compute(mask, image, config) -> MeasurementReport`
- `CorrectionSession.apply(action) -> CorrectionState`
- `DatasetExporter.export(samples, format, split_policy) -> ExportReport`

## Cross-Cutting Requirements

- Structured logging and run manifests
- Versioned file formats
- Deterministic debug mode
- Backward-compatible adapters for current hydride workflows
- GUI and CLI parity for core local workflows
