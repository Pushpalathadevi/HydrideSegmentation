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
- `predicted_fn_classification.png` and `corrected_fn_classification.png` when Fn debug export is enabled
- `predicted_fn_angle_distribution.png` and `corrected_fn_angle_distribution.png` when Fn debug export is enabled
- `predicted_fn_feature_table.csv` and `corrected_fn_feature_table.csv` when Fn debug export is enabled
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

## Fn orientation metrics

Fn is computed over retained connected hydride components after the configured `min_feature_pixels` filter. Each component has an orientation angle `theta_i` in `[0, 90]` degrees, measured from the horizontal using the existing skeleton/PCA orientation convention. The default threshold is 45 degrees and the comparison is inclusive:

```text
Fn_count = sum_i I(theta_i >= theta_threshold) / N
```

The numerator is the number of hydrides meeting the threshold and `N` is the retained hydride count. The modified length-weighted metric uses the component's projected major-axis extent, measured on component pixels along the same principal direction:

```text
Fn_length = sum_i L_i I(theta_i >= theta_threshold) / sum_i L_i
```

`Fn_count` and `Fn_length_weighted` are dimensionless values in `[0, 1]`. The report also stores the threshold, both numerators and denominators, component lengths in pixels, and calibrated length totals when `microns_per_pixel` is available. An empty or fully filtered mask returns zero for both ratios and a zero denominator rather than NaN.

The length weighting is intentionally not based on component area. It reduces the influence of many short hydrides and exposes cases where a small number of long, high-angle hydrides dominate the morphology. It remains sensitive to segmentation fragmentation and merged components, so the per-component debug table and annotated overlay should be inspected before interpreting a value scientifically.

When `write_fn_debug_artifacts` is enabled, green outlines identify components counted in the numerator and red outlines identify components excluded from it. Labels show the component index, angle, and projected length. The companion angle plot shows the threshold line; the CSV is the auditable source for reproducing both ratios.

Every generated visual report image includes a white, top-right Fn summary box containing both `Fn (number)` and `Fn (length)`. Indexed mask files remain machine-readable and are not annotated. The `report_decimal_places` setting controls displayed and serialized report precision, including JSON metrics, CSV tables, debug feature tables, and image labels. It defaults to `2` and is clamped to the range `0` to `8` in the desktop YAML configuration.

### Estimation method and caveats

The component orientation is estimated after connected-component labeling with 8-connectivity. The retained component is filled, dilated by the existing one-pixel analysis footprint, skeletonized, and evaluated by principal-component analysis of skeleton coordinates. The resulting line orientation is folded into the acute angle from the horizontal, so a horizontal hydride is near `0` degrees and a vertical hydride is near `90` degrees. A component exactly on the threshold is included.

The length `L_i` is the inclusive projected extent of the original component pixels along its principal direction. It is a geometric length estimate, not a centerline arc length. For a one-pixel or degenerate component the minimum reported length is one pixel. A calibration changes the reported numerator and denominator units but does not change `Fn_length`, because the same scale factor appears in both terms.

Interpret results with these failure modes in mind:

- A fragmented hydride is counted as several components and can inflate the count-based numerator.
- A merged or crossing group is counted as one component and can receive a misleading single PCA angle and a very long projected length.
- Round or very short components have unstable orientation; use `min_feature_pixels` and inspect the debug table.
- Image borders can truncate a hydride and bias both its angle and length.
- The segmentation mask, not the raw grayscale appearance, determines the result. Incorrect segmentation produces a numerically valid but scientifically invalid Fn.
- The threshold is a morphology convention, not a universal material constant. Report it with every comparison.

For audit, compare `fn_count` with `fn_length_weighted`. A substantially lower length-weighted value means the excluded hydrides carry more total projected length. Always inspect the color-coded classification image and `*_fn_feature_table.csv` before drawing a materials conclusion.

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

