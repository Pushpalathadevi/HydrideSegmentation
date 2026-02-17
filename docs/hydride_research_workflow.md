# Hydride Research Workflow (End-to-End)

## Purpose

This guide defines a complete, reproducible workflow for a hydride-segmentation research project where multiple networks are trained and compared fairly on the same dataset and the same train/val/test split.

Use this workflow when your objective is:
- rigorous model comparison
- scientifically defensible conclusions
- manuscript-ready evidence and traceability

This project assumes a real annotated dataset (>10,000 examples, from prior publication) and requires strict split consistency across all compared models.

## Non-Negotiable Rules For Fair Comparison

1. Use one fixed split definition for all models.
2. Never tune on test results.
3. Keep preprocessing and label policy identical across models.
4. Log all configs, seeds, runtime context, and output artifacts.
5. Compare models on the same metrics and same evaluation split.
6. Keep train/val/test leakage checks strict.

## Recommended Project Layout

```text
project_root/
  data/
    hydride_raw/                       # source + masks or pre-split dataset
  outputs/
    prepared_dataset_hydride_v1/       # frozen split used by all models
    benchmarks/
      unet_binary_seed42/
      unet_binary_seed43/
      torch_pixel_seed42/
      sklearn_pixel_seed42/
    eval/
      unet_binary_seed42_test.json
      ...
    phase_gates/
  configs/
    hydride/
      train.unet_binary.yml
      train.torch_pixel.yml
      train.sklearn_pixel.yml
      evaluate.hydride.yml
```

## Phase 0 - Freeze Dataset Policy And Splits

### Goal
Create one canonical prepared dataset split and reuse it everywhere.

### Option A: Dataset Already Pre-Split

If your dataset already has:
- `train/images`, `train/masks`
- `val/images`, `val/masks`
- `test/images`, `test/masks`

Then use that as canonical and run QA only:

```bash
microseg-cli dataset-qa \
  --config configs/dataset_qa.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --strict
```

### Option B: Dataset Is `source/` + `masks/`

Prepare once, then freeze:

```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --dataset-dir data/hydride_raw \
  --output-dir outputs/prepared_dataset_hydride_v1 \
  --set split_train_ratio=0.8 \
  --set split_val_ratio=0.1 \
  --set split_test_ratio=0.1 \
  --set split_strategy=leakage_aware
```

Then QA:

```bash
microseg-cli dataset-qa \
  --config configs/dataset_qa.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --strict
```

### Cross-Checks

- Confirm split counts are stable and documented.
- Confirm split manifest exists and includes global IDs.
- Record the split manifest path and checksum in your experiment logbook.
- Confirm no leakage warnings/errors in QA report.
- Confirm class index policy is fixed and documented.

### Do / Don’t

- Do freeze this dataset path for all downstream runs.
- Do not re-run split preparation with different seeds after benchmarking starts.

## Phase 1 - Define Benchmark Matrix

### Goal
Define exactly which models and settings will be compared.

Recommended starting matrix:
- `unet_binary`
- `hf_segformer_b0` (scratch)
- `hf_segformer_b2` (scratch)
- `transunet_tiny`
- `segformer_mini`

For each model, define:
- training config (`configs/hydride/train.<model>.yml`)
- seed list (for example 42, 43, 44)
- training budget policy (epochs, max_samples, early stop policy)

### Cross-Checks

- Same dataset path for all models.
- Same label semantics and mask format.
- Explicit seed list captured before training.

### Do / Don’t

- Do run at least 3 seeds for robust comparison when feasible.
- Do not change hyperparameter search space mid-comparison without versioning that as a new experiment set.

### HPC Trial Configuration Table (Hydride)

Use these ready templates under `configs/hydride/` with the same frozen split dataset.

All `_scratch` transformer trials below are initialized from architecture configs only (no transfer learning, no pretrained checkpoint download).

| Trial ID | Backend | Config File | Expected Input Contract | Starter Hyperparameters | Suggested HPC Profile | Why This Model Is In Top-5 |
|---|---|---|---|---|---|---|
| H1 | `unet_binary` | `configs/hydride/train.unet_binary.baseline.yml` | RGB image (`H,W,3`) + indexed binary mask (`H,W`, `0/1`) | `epochs=20, batch=8, lr=1e-3, wd=1e-5` | 1 GPU, 8 CPU, 32 GB RAM, 8-12h | Essential strong CNN baseline; best reference for classical segmentation behavior |
| H2 | `hf_segformer_b0` | `configs/hydride/train.hf_segformer_b0_scratch.yml` | RGB image (`H,W,3`) + indexed binary mask (`H,W`, `0/1`) | `epochs=24, batch=6, lr=6e-4, wd=1e-5` | 1 GPU, 10-12 CPU, 40 GB RAM, 10-18h | Efficient SOTA transformer; good quality/compute trade-off |
| H3 | `hf_segformer_b2` | `configs/hydride/train.hf_segformer_b2_scratch.yml` | RGB image (`H,W,3`) + indexed binary mask (`H,W`, `0/1`) | `epochs=30, batch=4, lr=4e-4, wd=2e-5` | 1 GPU, 12 CPU, 48 GB RAM, 14-28h | Higher-capacity SOTA transformer to test scale-up gains |
| H4 | `transunet_tiny` | `configs/hydride/train.transunet_tiny.yml` | RGB image (`H,W,3`) + indexed binary mask (`H,W`, `0/1`) | `epochs=24, batch=6, lr=7e-4, wd=1e-5, depth=2, heads=4` | 1 GPU, 8-12 CPU, 40 GB RAM, 10-16h | Hybrid transformer+U-Net; useful architectural bridge model |
| H5 | `segformer_mini` | `configs/hydride/train.segformer_mini.yml` | RGB image (`H,W,3`) + indexed binary mask (`H,W`, `0/1`) | `epochs=24, batch=6, lr=8e-4, wd=1e-5, patch=4` | 1 GPU, 8-12 CPU, 40 GB RAM, 10-16h | Internal lightweight transformer baseline for ablation and debugging |

Evaluation config for all trials:
- `configs/hydride/evaluate.hydride.yml`

Run pattern:

```bash
microseg-cli train \
  --config configs/hydride/train.hf_segformer_b0_scratch.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/benchmarks/hf_segformer_b0_seed42 \
  --set seed=42 \
  --no-auto-prepare-dataset
```

## Phase 2 - Run Training

### Goal
Train each model with reproducible artifacts and logs.

Example command pattern:

```bash
microseg-cli train \
  --config configs/hydride/train.unet_binary.baseline.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/benchmarks/unet_binary_seed42 \
  --set seed=42 \
  --no-auto-prepare-dataset
```

Repeat for all models and seeds.

### Optional: HPC Sweep Planning

Use HPC bundle generation for large sweeps:

```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_hydride \
  --set architectures=unet_binary,hf_segformer_b0,hf_segformer_b2,transunet_tiny,segformer_mini
```

### Cross-Checks

- Training output contains `resolved_config.json`.
- Training report artifacts exist (`report.json`, optional HTML).
- Checkpoints exist (best/last or backend-equivalent).
- No interrupted runs left untracked.

### Do / Don’t

- Do keep `--no-auto-prepare-dataset` for benchmark fairness.
- Do not compare runs where training data path differs.

## Phase 3 - Run Standardized Evaluation

### Goal
Evaluate every trained model on the same split (normally `test`) with the same policy.

Example:

```bash
microseg-cli evaluate \
  --config configs/hydride/evaluate.hydride.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --model-path outputs/benchmarks/unet_binary_seed42/best_checkpoint.pt \
  --split test \
  --output-path outputs/eval/unet_binary_seed42_test.json \
  --no-auto-prepare-dataset
```

### Cross-Checks

- All evaluations use `split=test`.
- Metrics are present: `pixel_accuracy`, `macro_f1`, `mean_iou`.
- Runtime fields are present and comparable.
- Schema versions in reports are valid and consistent.

### Do / Don’t

- Do use validation split for tuning and test split for final reporting.
- Do not use test metrics to adjust model settings during selection.

## Unified Data Contract (All Top-5 Models)

All top-5 models consume the identical dataset layout:

```text
<dataset_root>/
  train/
    images/*.png|*.jpg|*.tif
    masks/*.png|*.jpg|*.tif
  val/
    images/*.png|*.jpg|*.tif
    masks/*.png|*.jpg|*.tif
  test/
    images/*.png|*.jpg|*.tif
    masks/*.png|*.jpg|*.tif
```

Rules:
- image and mask filenames must match per split.
- all models train on the same `train` split and tune on the same `val` split.
- all models are evaluated on the same `test` split using identical evaluator logic.
- no model-specific alternate split is allowed in benchmark runs.

This is already compatible with the current training/evaluation stack for:
- `unet_binary`
- `hf_segformer_b0`
- `hf_segformer_b2`
- `transunet_tiny`
- `segformer_mini`

## Metrics Used For Comparison

For every evaluated run (`microseg.pixel_eval.v2` reports):
- `pixel_accuracy`
- `macro_f1`
- `mean_iou`
- `per_class_iou`
- `runtime_seconds`

Primary ranking metrics:
- `mean_iou`
- `macro_f1`

Secondary selection metrics:
- `runtime_seconds`
- stability across seeds (mean + spread over repeated runs)

## Phase 4 - Collate Results And Compare Critically

### Goal
Build a unified comparison table across models and seeds.

Collect from each evaluation JSON:
- model/backend
- seed
- pixel accuracy
- macro F1
- mean IoU
- runtime (seconds)
- checkpoint used

Recommended summary views:
- per-run table (all seeds)
- per-model mean and standard deviation
- Pareto view: quality (`mean_iou`) vs runtime

Use GUI `Workflow Hub -> Run Review` for pairwise comparisons and narrative checks, then build final aggregate tables from all runs.

Example collation snippet (JSON -> CSV):

```bash
python - <<'PY'
import csv
import json
import re
from glob import glob
from pathlib import Path

rows = []
for path in sorted(glob("outputs/eval/*_test.json")):
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    m = re.match(r"(?P<model>.+)_seed(?P<seed>\\d+)_test$", p.stem)
    rows.append(
        {
            "run_file": p.name,
            "model": m.group("model") if m else p.stem,
            "seed": int(m.group("seed")) if m else -1,
            "pixel_accuracy": metrics.get("pixel_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "mean_iou": metrics.get("mean_iou"),
            "runtime_seconds": payload.get("runtime_seconds"),
        }
    )

out = Path("outputs/eval/benchmark_summary.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "run_file",
            "model",
            "seed",
            "pixel_accuracy",
            "macro_f1",
            "mean_iou",
            "runtime_seconds",
        ],
    )
    w.writeheader()
    w.writerows(rows)
print(f"Wrote: {out}")
PY
```

### Cross-Checks

- Every row maps to a unique run directory and config.
- No missing runs for a planned seed/model.
- Outliers are investigated, not silently removed.

### Do / Don’t

- Do report both central tendency and variability.
- Do not cherry-pick single best seed only.

## Single-Script Benchmark + Dashboard Status

Current status: **available**.

You can run the top-5 benchmark suite end-to-end (train + eval + summary dashboard) with:

```bash
python scripts/hydride_benchmark_suite.py \
  --config configs/hydride/benchmark_suite.top5.yml \
  --strict
```

Outputs:
- run-level summary JSON: `outputs/benchmarks/top5_suite/benchmark_summary.json`
- run-level CSV: `outputs/benchmarks/top5_suite/benchmark_summary.csv`
- aggregated CSV: `outputs/benchmarks/top5_suite/benchmark_aggregate.csv`
- HTML dashboard: `outputs/benchmarks/top5_suite/benchmark_dashboard.html`

What this already includes:
- model name, seed, status
- metrics (`pixel_accuracy`, `macro_f1`, `mean_iou`, runtime)
- backend + resolved architecture fields
- resolved key hyperparameters (`epochs`, `batch_size`, `learning_rate`, `weight_decay`, plus architecture knobs)
- train/eval config references and artifact paths

Remaining gaps (known):
- no statistical significance testing module yet (for example paired tests/CI bands).
- dashboard is table-first (no advanced interactive plots yet).
- HPC-distributed execution graph (job dependencies/retries) is not yet centralized in one scheduler-native workflow file.
- `segmentation_models_pytorch` family backbones are installed but not yet first-class train/eval backends in the unified benchmark runner.

## Phase 5 - Error Analysis And Scientific Interpretation

### Goal
Understand failure modes, not only headline metrics.

Suggested analysis:
- class-wise false positives/false negatives
- boundary-quality failures
- low-contrast hydride regions
- artifact sensitivity (noise, illumination, texture)

Use correction GUI for visual audit of representative successes/failures and document patterns in a structured review log.

### Cross-Checks

- Include at least fixed sample IDs for qualitative panels.
- Confirm examples are drawn from test set for final manuscript figures.

### Do / Don’t

- Do include both strengths and weaknesses per model.
- Do not present only visually favorable cases.

## Phase 6 - Final Model Selection

### Goal
Select the model for deployment/publication based on transparent criteria.

Recommended decision criteria:
- primary: `mean_iou` and `macro_f1` on test
- secondary: runtime, stability across seeds, operational simplicity
- tertiary: correction burden in human review

Record decision rationale in a short selection memo:
- winner
- runner-up
- why winner is preferred
- where winner still fails

## Phase 7 - Manuscript Preparation Package

Prepare a manuscript-ready bundle with:
- dataset policy statement and split protocol
- model comparison methodology
- final config files used
- aggregate metrics table (mean ± std)
- qualitative figure panel with sample IDs
- failure-mode discussion
- reproducibility appendix (commands + versions + schema references)

Minimum claims checklist:
- all compared models trained on identical split
- no test leakage into tuning
- seed variability reported
- caveats/limitations explicitly stated

## Caveats And Common Pitfalls

- Hidden leakage from near-duplicate micrographs can invalidate results.
- Different preprocessing pipelines across models can make comparisons unfair.
- Comparing checkpoints from different training stages without policy control can bias results.
- Runtime comparisons are hardware-dependent; report hardware context.
- Small metric differences without seed-level stability should be treated cautiously.

## End-Of-Phase Gates (Mandatory)

For each major phase change, run:

```bash
PYTHONPATH=. pytest -q
microseg-cli phase-gate --phase-label "Hydride Benchmark Phase X" --strict
```

Then update:
- `README.md` (if user workflow changed)
- `docs/current_state_gap_analysis.md` (progress and open gaps)
- relevant phase/status docs under `docs/`

## Related Documents

- `docs/training_data_requirements.md`
- `docs/configuration_workflow.md`
- `docs/gui_user_guide.md`
- `docs/hpc_ga_user_guide.md`
- `docs/scientific_validation.md`
- `docs/versioning_and_release_policy.md`

## External Model References (For Manuscript Methods)

- SegFormer paper: Xie et al., *SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers*, arXiv:2105.15203
- Hugging Face SegFormer docs: https://huggingface.co/docs/transformers/model_doc/segformer
- Hugging Face implementation classes used in this repo:
  - `transformers.SegformerConfig`
  - `transformers.SegformerForSemanticSegmentation`
