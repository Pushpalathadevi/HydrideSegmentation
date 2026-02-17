# HPC Air-Gap Top-5 Real-Data Runbook

## Purpose

This is the single end-to-end runbook to execute the full hydride top-5 model comparison on an air-gapped GPU HPC system, for both:
- scratch initialization
- local-pretrained initialization

It produces one consolidated benchmark dashboard with training curves, validation sample panels, and evaluation metrics.

## Canonical Top-5 Model Matrix

1. `unet_binary`
2. `hf_segformer_b0`
3. `hf_segformer_b2`
4. `transunet_tiny`
5. `segformer_mini`

Local-pretrained model IDs used in this runbook:
- `unet_binary` -> `unet_binary_resnet18_imagenet_partial`
- `hf_segformer_b0` -> `hf_segformer_b0_ade20k`
- `hf_segformer_b2` -> `hf_segformer_b2_ade20k`
- `transunet_tiny` -> `transunet_tiny_vit_tiny_patch16_imagenet`
- `segformer_mini` -> `segformer_mini_vit_tiny_patch16_imagenet`

## 1. Connected Machine Preparation (Internet Available)

Install dependencies:
```bash
pip install -r requirements-core.txt
pip install -e .
pip install segmentation-models-pytorch
```

Download all pretrained bundles (writes to `pre_trained_weights/`):
```bash
python scripts/download_pretrained_weights.py --targets all --force
```

Validate registry + checksums:
```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

Generate provenance inventory report (for reporting/manuscript traceability):
```bash
python scripts/pretrained_inventory_report.py \
  --registry-path pre_trained_weights/registry.json \
  --output-path outputs/pretrained_weights/inventory_report.json
```

## 2. Dataset Contract (Real Data)

Expected dataset layout on HPC:
```text
<dataset_root>/
  train/
    images/*
    masks/*
  val/
    images/*
    masks/*
  test/
    images/*
    masks/*
```

Rules:
- image/mask filenames must match per split.
- masks must be indexed class maps for binary segmentation (`0` background, `1` hydride).
- no split leakage across `train`, `val`, `test`.

Optional quick count check before transfer:
```bash
python - <<'PY'
from pathlib import Path
root = Path('/path/to/real_dataset_root')
for split in ['train', 'val', 'test']:
    ni = len(list((root / split / 'images').glob('*')))
    nm = len(list((root / split / 'masks').glob('*')))
    print(split, 'images=', ni, 'masks=', nm)
PY
```

## 3. Transfer to Air-Gapped HPC

Copy these to HPC (external disk or secure transfer path):
- full repository working tree
- `pre_trained_weights/` folder
- real dataset root (train/val/test layout)

On HPC, verify pretrained registry after copy:
```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

## 4. HPC Environment Bootstrap

Inside repo on HPC:
```bash
pip install -r requirements-core.txt
pip install -e .
```

If missing in your environment:
```bash
pip install segmentation-models-pytorch
```

## 5. Mandatory Debug Integrity Run (Before Real Data)

Build tiny debug dataset by duplicating repository sample image:
```bash
python scripts/build_debug_duplicate_dataset.py \
  --image-path test_data/syntheticHydrides.png \
  --output-dir outputs/debug_pretrained_dataset \
  --resize-width 64 \
  --resize-height 64 \
  --train-count 4 \
  --val-count 2
```

Run combined scratch+pretrained top-5 debug benchmark:
```bash
python scripts/hydride_benchmark_suite.py \
  --config configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.debug.yml \
  --strict
```

Debug dashboard output:
- `outputs/benchmarks/top5_suite_scratch_vs_pretrained_debug/benchmark_dashboard.html`

## 6. Real Data Full Benchmark (Single Dashboard)

Use this template config:
- `configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.template.yml`

Copy and edit only these fields:
1. `dataset_dir`
2. `output_root`
3. optionally `seeds` (`[42]` for first run, `[42, 43, 44]` for publication-grade comparison)
4. `benchmark_mode` (`false` for externally pre-split datasets without `dataset_manifest.json`)

Run:
```bash
python scripts/hydride_benchmark_suite.py \
  --config configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.template.yml \
  --strict
```

Primary outputs:
- `benchmark_summary.json`
- `benchmark_summary.csv`
- `benchmark_aggregate.csv`
- `benchmark_dashboard.html`
- `curves/*_loss_curve.png`
- `curves/*_accuracy_curve.png`
- `curves/*_iou_curve.png`

The dashboard includes:
- per-run and per-model metrics (`pixel_accuracy`, `macro_f1`, `mean_iou`)
- runtime and model-size summaries
- training curves for every run
- tracked validation sample IoU summaries and validation panel gallery

## 7. Optional GA HPC Sweep Profiles (If You Want Candidate Search)

Scratch top-5 GA profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_scratch.default.yml \
  --dataset-dir /path/to/real_dataset_root \
  --output-dir /path/to/hpc_outputs/hpc_ga_top5_scratch
```

Local-pretrained top-5 GA profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_airgap_pretrained.default.yml \
  --dataset-dir /path/to/real_dataset_root \
  --output-dir /path/to/hpc_outputs/hpc_ga_top5_local_pretrained
```

## 8. Pretrained Provenance + Citation Sources

- Runtime registry: `pre_trained_weights/registry.json`
- Tracked metadata companions: `pre_trained_weights/metadata/*.meta.json`
- Catalog for reporting: `docs/pretrained_model_catalog.md`
- Catalog JSON: `docs/pretrained_model_catalog.json`
- BibTeX: `docs/pretrained_model_citations.bib`

## 9. Success Criteria Checklist

1. `microseg-cli validate-pretrained --strict` passes on HPC.
2. Debug suite passes and dashboard is generated.
3. Real-data suite completes without failed runs (`--strict`).
4. Combined dashboard exists and includes all 10 experiments (5 scratch + 5 local-pretrained).
5. Summary JSON/CSV and aggregate CSV are archived with run config and environment notes.
