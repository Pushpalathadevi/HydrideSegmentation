# Benchmark Metrics and Dashboard Reference

## Purpose

This reference defines the expanded metrics used in benchmark reports and how to read the benchmark dashboard for faster model-selection decisions.

## Evaluation Report Schema

- `microseg.pixel_eval.v4`

## Metric Groups

### Core quality

- `pixel_accuracy`
- `macro_f1`
- `mean_iou`

### Robustness and class balance

- `macro_precision`
- `macro_recall`
- `weighted_f1`
- `balanced_accuracy`
- `cohen_kappa`
- `frequency_weighted_iou`

### Binary hydride diagnostics (labels `{0,1}`)

- `foreground_precision`
- `foreground_recall` (sensitivity)
- `foreground_specificity`
- `foreground_iou`
- `foreground_dice`
- `false_positive_rate`
- `false_negative_rate`
- `matthews_corrcoef`
- `gt_foreground_fraction`
- `pred_foreground_fraction`

### Per-class explainability

- `per_class_iou`
- `per_class_precision`
- `per_class_recall`
- `per_class_f1`
- `class_support`
- `confusion_matrix` (`counts`, `row_normalized`, `column_normalized`)

### Hydride scientific error metrics

- `mask_area_fraction_abs_error`
- `hydride_count_abs_error`
- `hydride_size_wasserstein`
- `hydride_orientation_wasserstein`

## Benchmark Dashboard Sections

The benchmark dashboard (`benchmark_dashboard.html`) consolidates:

1. High-level cards:
   - run counts
   - best quality model
   - best efficiency model
   - fastest model
2. Model summary ranking:
   - quality/efficiency/runtime/robustness ranks
   - quality and diagnostic metrics with mean±std
   - runtime and model-size summaries
3. Scientific error summary:
   - area/count/size/orientation error aggregates
4. Run-level table:
   - all metrics per seed/model run
   - resolved hyperparameters and artifact pointers
5. Training curve gallery:
   - loss/accuracy/IoU curves per run
6. Tracked sample evolution:
   - IoU-vs-epoch curves per tracked sample
   - first/last/delta/best/worst IoU summaries
7. Validation sample panels:
   - latest tracked sample visual panels

## Recommended Selection Logic

1. Filter out failed/unstable runs first.
2. Rank by quality (`mean_iou`, `macro_f1`, `foreground_dice` for binary).
3. Check robustness diagnostics (`FPR/FNR`, `MCC`, `Cohen kappa`, overfit gap, tracked-sample IoU deltas).
4. Apply runtime/size constraints.
5. Use scientific error metrics to reject models with physically implausible morphology behavior.

## Remaining Gaps

- Statistical significance module (paired tests/confidence intervals) is not yet implemented.
- Dashboard is static HTML by design (air-gap safe), not interactive.
