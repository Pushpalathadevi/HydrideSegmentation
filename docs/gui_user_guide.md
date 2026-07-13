# GUI User Guide (Qt Desktop)

## Primary Workflows

1. Load image (or bundled sample) and select model.
2. Run segmentation.
3. Inspect prediction in split view and Results Dashboard.
4. Inspect the Results Dashboard and Fn metrics.
5. Export mask artifacts and/or full results package (`json` + `html` + `pdf` + `csv`).
7. Save session and resume later.
8. Review model-specific frozen-checkpoint tips before selecting ML models.

While inference is running, the top status banner shows the current stage, elapsed time, and an ETA estimate when the app has enough history to infer one.
During recursive batch jobs it also shows processed-image counts, percent complete, and rolling ETA updates as inference, quantitative analysis, and export steps finish.
Single-image and batch inference now both run through the in-process background worker path, which keeps the GUI responsive while allowing warmed ML bundles to be reused across runs.
The main model selector is intentionally ordered for deployment use: discovered trained models appear first, `Hydride ML (UNet)` remains the default trained checkpoint, and `Hydride Conventional` remains available as the fallback baseline.
The desktop uses a split layout: the left sidebar holds project/model/review controls, while the right `Segmentation` workspace shows the input image and predicted mask together. A newly loaded image leaves the right pane empty with `Result not ready yet`, preventing an old mask from being mistaken for the current result.
The left control rail now defaults to a narrower progressive-disclosure layout:
- `Quick Start` stays visible with load/select/run controls
- `Run Setup / Status` carries model metadata, preprocessing summary, warm-load status, and segmentation progress
- `Active Run` appears after inference with review/export shortcuts plus optional notes
- quantification settings are visible and grouped; advanced display tools, export/session, and logs remain expandable

The desktop log now appears in a shared bottom strip under the main workspace instead of consuming sidebar width, and it is shown by default on startup.
Input, mask, overlay, and batch-summary image views all expose local zoom, pan, fit, and display-contrast controls. The main active-run image views keep pan and zoom synchronized so inspection stays aligned across tabs.

The control rail keeps image loading, sample selection, and model selection on separate rows so the ML model list remains readable without forcing the image workspace to collapse.
Advanced controls are grouped behind collapsible sections:
- `Inference Setup` for config and calibration
- `Advanced Display Tools` for optional display/layer controls
- `Export & Session` for exports, saves, and report options
- `Quantification Settings` for Fn metrics, distributions, precision, and debug artifacts

The desktop application intentionally keeps training and active-learning data capture out of the primary inference workflow. Use the dedicated command-line training/evaluation tools when those workflows are needed.

## Review Workflow

Inspection controls:
- zoom in/out/reset
- synchronized pan/zoom
- transparency sliders for prediction layers

Conventional controls (`Hydride Conventional` model):

The controls appear directly above the side-by-side input/result panes with canonical defaults pre-filled. Hover over any control for its scientific purpose and tuning guidance. Run once with `Run Segmentation`; subsequent parameter edits automatically queue a new background result after a 400 ms debounce. Rapid edits are coalesced, and an edit made while inference is running is applied as soon as the current run finishes.
- CLAHE clip limit and tile grid
- adaptive threshold block size and `C`
- morphology kernel and iterations
- area threshold and optional crop percentage

## Exporting Results

`Export Results Package` writes deployment-facing outputs:
- `results_summary.json` with prediction statistics and analysis config
- `results_report.html`
- `results_report.pdf`
- `results_metrics.csv`
- `artifacts_manifest.json`
- input/mask/overlay/orientation-map/distribution images for predicted masks

Report customization controls:
- report profile: `balanced`, `full`, `audit`
- section toggles: metadata, calibration, key summary, scalar table, distributions, overlays, artifact manifest
- metric checklist (advanced): select exact scalar metrics for export
- key-metric cutoff (`Top-K`) and CSV output toggle

Batch export:
- `Export Batch Summary` (menu/button) exports selected history runs or all runs if none selected.
- outputs: `batch_results_summary.json`, `batch_results_report.html`, optional `batch_results_report.pdf`, `batch_metrics.csv`
- batch summary now also includes `artifacts_manifest.json` plus `runs/` with one full per-image result package each; the root HTML links directly into those per-run summaries.
- batch export skips PDF generation by default to keep large folder exports responsive; enable PDF only when you specifically need a printable report.
- desktop scalar displays and report tables round floating metrics to two decimals for consistent scientific readability.

## Recursive Folder Inference (GUI + CLI parity)

`Run Batch` supports folder-first operation:

1. Click `Run Batch`.
2. Select an input folder (or cancel to fall back to manual file selection).
3. The app scans recursively (default) for configured image globs (`*.png`, `*.jpg`, `*.jpeg`, `*.tif`, `*.tiff`, `*.bmp`).
4. Each discovered image is inferred, quantitatively analyzed, and exported under the final batch package in one pass.
5. The app writes `runs/` per-image artifacts plus `batch_results_summary.json`, `batch_results_report.html`, optional PDF/CSV outputs, `artifacts_manifest.json`, and `resolved_config.json`.
6. The batch summary inspector opens automatically at the end of the run for immediate review.

The exported batch HTML includes one aligned row per image with:

- input image preview
- predicted mask preview
- overlay preview
- key scalar metrics (including hydride area fraction/count when available)
- direct links to the corresponding per-run `results_summary.json` / HTML report package under `runs/`

ML preprocessing preview:

- when `Hydride ML (UNet)` is selected, the live input preview shows the actual processed image being fed to inference rather than only the raw file
- `Adjust Contrast Before Inference` now shows a two-panel split view: raw source on the left, processed-for-inference image on the right
- desktop logs now write explicit preprocessing records for original size, resized size, resize scale, contrast mode/parameters, channel duplication, and mask rescaling back to source size

GUI-native batch summary inspector:

- open from `File -> Open Batch Results Summary...` or `Results Dashboard -> Open Batch Summary`
- load any exported `batch_results_summary.json`
- inspect aggregate batch summary at top, select per-image rows on the left, and review large input/mask/overlay panels with detailed per-image statistics on the right

## Session Persistence

- `Save Session` writes a restartable project folder with images, masks, class map, notes, and UI state.
- `Load Session` restores run state and correction workspace.

## Appearance And UI Config

- Use `Settings -> Appearance & Export Settings` to adjust readability and defaults:
  - base/heading/monospace font size
  - menu, tab, toolbar, and status-bar font size
  - control padding, panel spacing, table row density
  - high-contrast mode
  - startup window geometry and screen clamping
  - default export profile and output toggles
- Load/save YAML from the dialog.
- Default config file: `configs/app/desktop_ui.default.yml`
- Startup override: `hydride-gui --ui-config configs/app/desktop_ui.default.yml`
- The main workspace now keeps advanced panels behind the gear button near the top controls.
- The image workspace is fit-to-view aware on resize and tab changes, so small test images fill the available viewport better.
- Each image viewport includes its own zoom controls, plus Ctrl+mouse-wheel zoom and drag-based scrolling.
- The application restores or clamps its window size so it stays on-screen on single or dual-monitor setups.

## Sidebar Redesign

The left control rail now uses grouped cards instead of a dense button strip:

![Sidebar redesign comparison](diagrams/gui_sidebar_redesign_comparison.svg)

The revised layout puts the most common actions first:

- quick start: load image, sample, and model selection
- correction tools: interaction and overlay controls
- export and session: output packaging and persistence

The goal is to keep the controls readable without forcing the image workspace to become too narrow.

## Results Dashboard

`Results Dashboard` provides:
- side-by-side predicted/corrected scalar metrics
- hydride fraction, hydride count, feature density, orientation summary, and size summary
- orientation map + size/orientation distributions for predicted and corrected masks
- adjustable plotting parameters:
  - orientation bins
  - size bins
  - minimum feature-pixel threshold
  - size axis scale (`linear`/`log`)
  - orientation colormap
  - Fn inclusion and angle threshold
  - report decimal precision
  - distribution charts, orientation maps, and Fn debug artifacts

## Spatial Calibration (Optional)

Default reporting units are pixels.

To enable micron-based reporting:
- click `Scan Metadata Scale` to auto-detect TIFF/DPI scale metadata when available
- or click `Calibrate Scale...`, draw a known line, and enter its real-world length

When calibration is active, size-related metrics and report outputs include micron-based values (`um`, `um^2`) in addition to pixel metrics.

## CLI For Training And Evaluation

Training, evaluation, dataset preparation, and deployment operations are intentionally documented and run through the CLI rather than competing with the inference workspace. Use [`usage_commands.md`](usage_commands.md) for copy-paste Windows and Linux commands.

## Model Guidance Panel

The model description area now includes metadata pulled from `frozen_checkpoints/model_registry.json` when available:
- model nickname and type
- expected input dimensions
- checkpoint path hint
- lifecycle stage (`smoke`, `candidate`, `promoted`, `builtin`)
- application suitability remarks
- short user tips
- optional quality report path

This helps users select the right model for optical/TEM or other microstructural contexts.
Smoke-stage models are debug-only and are not intended for scientific reporting.

For a step-by-step beginner tutorial on copying a trained `.pth` checkpoint into the GUI workflow on an air-gapped machine, see [`docs/gui_model_integration_guide.md`](gui_model_integration_guide.md).

## Sample Onboarding And Logs

- Bundled sample images are available from:
  - `File -> Open Sample`
  - top-bar `Load Sample`
- Desktop logs are written to:
  - `outputs/logs/desktop/`


## Trained model discovery (architecture-aware)

The model dropdown now includes inference-capable trained models discovered from:

- `outputs/runs/<run_name>/` (successful runs only)
- frozen checkpoint registry entries in `frozen_checkpoints/model_registry.json`
- optional machine-local overlay entries in `frozen_checkpoints/model_registry.local.json`

A trained run is considered inference-eligible when it has:

- `report.json` with status `ok`/`success`/`completed`
- a resolvable model checkpoint path
- architecture metadata (`model_architecture`) compatible with repo-supported trainable families

Failed/incomplete runs are skipped and not shown as runnable model options.

When troubleshooting model loading:

1. confirm run status in `report.json`
2. confirm checkpoint file exists at declared `model_path`
3. confirm architecture is one of the supported trainable families
4. verify required backend dependencies are installed (for example `transformers` for HF backends)
5. for local-only checkpoints, confirm the registry overlay and GUI model label match the copied file
