# HPC Air-Gap Top-5 Real-Data Runbook

## Purpose

Run one full campaign on an air-gapped GPU HPC system for both:
- scratch initialization
- local-pretrained initialization

The campaign writes one consolidated view with:
- run-level metrics
- training curves
- tracked-sample panels with per-image metrics
- model/weight/compute summary statistics

Primary outputs per campaign:
- `summary.json`
- `summary.html`

Backward-compatible outputs are still written:
- `benchmark_summary.json`
- `benchmark_dashboard.html`
- `benchmark_summary.csv`
- `benchmark_aggregate.csv`

## 0. Canonical Model Matrix

Top-5 models in this runbook:
1. `unet_binary`
2. `hf_segformer_b0`
3. `hf_segformer_b2`
4. `transunet_tiny`
5. `segformer_mini`

Local-pretrained model IDs:
- `unet_binary` -> `unet_binary_resnet18_imagenet_partial`
- `hf_segformer_b0` -> `hf_segformer_b0_ade20k`
- `hf_segformer_b2` -> `hf_segformer_b2_ade20k`
- `transunet_tiny` -> `transunet_tiny_vit_tiny_patch16_imagenet`
- `segformer_mini` -> `segformer_mini_vit_tiny_patch16_imagenet`

## 1. Connected Linux Machine (Internet + Browser)

All commands below assume repo root.

### 1.1 Environment bootstrap (`.venv`)

```bash
cd /path/to/HydrideSegmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -e .
python -m pip install segmentation-models-pytorch
```

### 1.2 Browser download step (manual)

Use browser downloads only and stage a complete `pre_trained_weights/` payload (bundle folders + `registry.json`).

Reference for required model IDs and source URLs:
- [docs/pretrained_model_catalog.md](docs/pretrained_model_catalog.md)

After browser downloads, copy them under repo-root `pre_trained_weights/` so that:
- `pre_trained_weights/registry.json` exists
- all referenced bundle folders/files exist

### 1.3 Validate pretrained payload before packaging

```bash
source .venv/bin/activate
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

Expected output includes:
- `pretrained registry valid: True`

### 1.4 Package transfer archive + checksum

```bash
mkdir -p transfer_bundle
zip -r transfer_bundle/hydrideseg_repo_with_weights.zip . \
  -x ".git/*" ".venv/*" "__pycache__/*" ".pytest_cache/*" ".ruff_cache/*" "outputs/*"
sha256sum transfer_bundle/hydrideseg_repo_with_weights.zip > transfer_bundle/hydrideseg_repo_with_weights.zip.sha256
```

Optional local verification before physical transfer:

```bash
sha256sum -c transfer_bundle/hydrideseg_repo_with_weights.zip.sha256
```

## 2. Air-Gapped HPC Machine

### 2.1 Copy + verify + extract

```bash
mkdir -p /path/to/hpc_work
cp /transfer_media/hydrideseg_repo_with_weights.zip /path/to/hpc_work/
cp /transfer_media/hydrideseg_repo_with_weights.zip.sha256 /path/to/hpc_work/
cd /path/to/hpc_work
sha256sum -c hydrideseg_repo_with_weights.zip.sha256
unzip -o hydrideseg_repo_with_weights.zip
cd /path/to/hpc_work/HydrideSegmentation
```

### 2.2 HPC environment bootstrap (`.venv`)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -e .
python -m pip install segmentation-models-pytorch
```

### 2.3 Validate pretrained payload on HPC

```bash
source .venv/bin/activate
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

### 2.4 Confirm GPU visibility (`nvidia-smi`)

```bash
nvidia-smi
```

## 3. Dataset Contract (Real Data)

Expected dataset layout:

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

Quick count sanity check:

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

## 4. Recommended Debug Integrity Gate

```bash
source .venv/bin/activate
python scripts/build_debug_duplicate_dataset.py \
  --image-path test_data/syntheticHydrides.png \
  --output-dir outputs/debug_pretrained_dataset \
  --resize-width 64 \
  --resize-height 64 \
  --train-count 4 \
  --val-count 2

python scripts/hydride_benchmark_suite.py \
  --config configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.debug.yml \
  --strict
```

Debug outputs:
- `outputs/benchmarks/top5_suite_scratch_vs_pretrained_debug/summary.html`
- `outputs/benchmarks/top5_suite_scratch_vs_pretrained_debug/summary.json`

## 5. Real Campaign (Scratch + Local-Pretrained in One Run)

Copy template and edit only:
- `dataset_dir`
- `output_root`
- `seeds`
- `benchmark_mode`

```bash
cp configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.template.yml \
   configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.yml
```

Run campaign:

```bash
source .venv/bin/activate
python scripts/hydride_benchmark_suite.py \
  --config configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.yml \
  --strict
```

Optional Slurm single-job wrapper from repo root (robust to Slurm script staging/spooling):

```bash
./submitJob_1GPU.sh ./run_training_jobs.sh --dataset tiny --profile smoke
./submitJob_1GPU.sh ./run_training_jobs.sh --dataset custom --dataset_dir /path/to/HydrideData6.0/mado_style --profile full
```

The wrapper exports `HYDRIDE_REPO_ROOT` at submission time. `run_training_jobs.sh` uses that value (or `SLURM_SUBMIT_DIR`) to restore repo-root working directory and enforce `./.venv/bin/python` inside the job.

Optional anti-hang watchdog settings (add to the suite YAML for long HPC jobs):
- `command_idle_timeout_seconds`: timeout when no new bytes are written to a run log for N seconds.
- `command_wall_timeout_seconds`: timeout after N total seconds of a run.
- `command_terminate_grace_seconds`: graceful terminate window before force-kill (default `30`).
- `command_poll_interval_seconds`: watchdog polling interval (default `1`).

During execution, each run writes live logs to:
- `output_root/logs/<run_tag>/train.log`
- `output_root/logs/<run_tag>/eval.log`

## 6. Output Interpretation

Main deliverables (single campaign view):
- `summary.json`
- `summary.html`

Also written:
- `benchmark_summary.json`
- `benchmark_dashboard.html`
- `benchmark_summary.csv`
- `benchmark_aggregate.csv`
- `curves/*_loss_curve.png`
- `curves/*_accuracy_curve.png`
- `curves/*_iou_curve.png`

`summary.html` and `summary.json` include:
- scratch + local-pretrained runs in one campaign
- total runtime and per-run runtime
- parameter count + trainable parameter count
- checkpoint size and weight statistics
- compute-effort estimates (FLOPs estimates) and runtime GPU fields
- tracked validation images with per-image metric blocks under each image

## 7. Missing-Pretrained Behavior (Runner Hardening)

If pretrained artifacts are missing for a local-pretrained run:
- run is marked `pretrained_missing`
- actionable fix text is written in that run log
- remaining runs continue (scratch runs are not blocked)

Status appears in run-level table and JSON row `status`/`status_message`.

## 8. Success Checklist

1. `validate-pretrained --strict` passes on HPC.
2. Debug campaign passes.
3. Real campaign completes with expected rows for all models/seeds.
4. `summary.html` and `summary.json` exist in campaign output root.
5. Logs and output folder are archived with the exact config used.
