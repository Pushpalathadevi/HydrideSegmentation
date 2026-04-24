# Results Analysis And File Layout

This page explains where the system writes outputs and how to inspect them.

## Desktop Result Packages

Per-image desktop exports land in a dedicated folder with the pattern:

```text
<output_dir>/<image_stem>_<run_id>_results/
```

Files written by the desktop result exporter typically include:

- `input.png`
- `predicted_mask_indexed.png`
- `corrected_mask_indexed.png`
- `predicted_mask_color.png`
- `corrected_mask_color.png`
- `predicted_overlay.png`
- `corrected_overlay.png`
- `predicted_orientation_map.png`
- `predicted_size_distribution.png` when distribution chart export is enabled
- `predicted_orientation_distribution.png` when distribution chart export is enabled
- `corrected_orientation_map.png`
- `corrected_size_distribution.png` when distribution chart export is enabled
- `corrected_orientation_distribution.png` when distribution chart export is enabled
- `diff_mask.png`
- `results_summary.json`
- `results_report.html`
- `results_report.pdf`
- `results_metrics.csv`
- `artifacts_manifest.json`

The JSON summary contains:

- run provenance
- model identifiers
- analysis configuration
- predicted and corrected scalar metrics
- selected metric rows and key summary rows
- artifact names
- optional artifact-manifest metadata
- `analysis_config.postprocessing_options`, which records whether extended metrics, distribution charts, orientation maps, and physical-calibration metrics were enabled

Default postprocessing keeps the required scientific summaries fast: hydride count, area fraction, total/feature size, orientation values, and orientation color maps. Distribution chart PNGs, extended scalar summaries, density/equivalent-diameter metrics, histogram vectors, and micron-based metrics are opt-in.

## Batch Result Packages

Batch export folders use the pattern:

```text
<output_dir>/batch_results_<timestamp>/
```

Batch artifacts include:

- `batch_results_summary.json`
- `batch_results_report.html`
- `batch_results_report.pdf`
- `batch_metrics.csv`

## Training Runs

Training run folders typically contain:

- `report.json`
- `report.html`
- `error_report.json` on failure
- checkpoint files for each saved epoch or resume point
- structured log output

The report files are the primary source for:

- epoch metrics
- tracked validation sample summaries
- runtime progress
- resume metadata

## Evaluation Runs

Evaluation outputs default to `outputs/evaluation/` and commonly include:

- `pixel_eval_report.json`
- optional HTML summaries or report panels

The evaluation report records:

- per-run and aggregate metric values
- confusion-matrix-style breakdowns where relevant
- scientific distance metrics for size and orientation distributions

## Dataset Preparation Outputs

Dataset preparation and QA outputs commonly land under `outputs/dataops/` and may include:

- `dataset_qa_report.json`
- manifest files
- split planning reports
- preview artifacts when debug mode is enabled

## Desktop Sessions And Project State

Saved GUI project folders include:

- `project_state.json`
- `input.png`
- `prediction_indexed.png`
- `overlay.png`
- `corrected_indexed.png`

These folders are restartable session artifacts and should be preserved when sharing a correction session.

## Logs And Support Bundles

Useful roots:

- `outputs/logs/desktop/`
- `outputs/support_bundles/`
- `outputs/feedback_records/`

Support bundles should be used when a run needs to be reconstructed with its runtime context, manifests, and failure details.

## How To Analyze A Result

1. Open the HTML report first.
2. Check the scalar tables for drift in area fraction, count, and orientation summaries.
3. Inspect overlay and diff panels to understand geometric disagreements.
4. If spatial calibration exists, compare pixel and micron-based outputs together.
5. Use the CSV export when you need batch comparisons in a spreadsheet or notebook.
6. Keep the JSON summary for automated downstream processing and traceability.

