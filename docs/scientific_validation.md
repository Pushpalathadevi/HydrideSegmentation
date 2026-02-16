# Scientific Validation and Reproducibility Plan

## Validation Philosophy

Segmentation quality must be demonstrated with transparent, repeatable metrics and dataset governance.

## Required Validation Outputs

1. Segmentation performance
- IoU, Dice, precision, recall, F1
- Per-class and aggregate metrics where applicable

2. Scientific task metrics
- Orientation statistics stability
- Area fraction accuracy
- Size distribution consistency

3. Robustness testing
- Noise sensitivity
- Contrast variation
- Resolution scaling

4. Calibration and uncertainty (phase-wise)
- Confidence histograms
- Error concentration analysis by feature size/type

## Reproducibility Controls

- Run manifest with code version, config, model ID, and environment
- Fixed seeds for debug and benchmark modes
- Versioned dataset splits and correction sources

## Data Governance for Human Corrections

- Store original prediction and corrected mask together
- Record annotator/reviewer metadata
- Prevent leakage of corrected samples into evaluation sets unless explicitly versioned

## Correction Export Schema

- Current schema: `microseg.correction.v1`
- Required fields include:
  - source/run/model provenance
  - annotator identity and notes
  - corrected and predicted mask foreground counts
  - metrics snapshot from inference run

All downstream training pipelines must validate schema version before ingestion.
