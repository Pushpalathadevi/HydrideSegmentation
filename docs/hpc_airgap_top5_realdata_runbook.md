# HPC Air-Gapped Top-10 End-to-End Runbook

## Purpose

Run the full 10-model hydride segmentation benchmark suite on HPC with:

- scratch initialization
- local-pretrained initialization
- one consolidated comparative report (`summary.json`, `summary.html`)

This runbook is copy-paste oriented and assumes you set `REPO_ROOT` once.

## 0. One-Time Variables (HPC + connected machine)

```bash
export REPO_ROOT=/home/kvmani/ml_works/hydride_segmentation
cd "$REPO_ROOT"
```

If your folder name differs, set `REPO_ROOT` to the actual path first.

## 1. Canonical 10-Model Matrix

Scratch + local-pretrained comparisons are configured for:

1. `unet_binary`
2. `smp_deeplabv3plus_resnet101`
3. `smp_unetplusplus_resnet101`
4. `smp_pspnet_resnet101`
5. `smp_fpn_resnet101`
6. `hf_segformer_b0`
7. `hf_segformer_b2`
8. `hf_upernet_swin_large` (Swin-Large encoder family)
9. `transunet_tiny`
10. `segformer_mini`

Config files still use `top5` in filenames for backward compatibility; those configs now contain top-10 experiments.

## 2. Connected Machine (Internet) - Build Transfer Bundle

### 2.1 Environment

```bash
cd "$REPO_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -e .
python -m pip install segmentation-models-pytorch
```

### 2.2 Download all pretrained bundles (recommended path)

```bash
source .venv/bin/activate
python scripts/download_pretrained_weights.py --targets all --force
```

This command creates registry + bundle folders with exact filenames expected by training configs.

### 2.3 Validate pretrained registry and checksums

```bash
source .venv/bin/activate
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

### 2.4 Optional: verify exact expected local weight filenames

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("pre_trained_weights")
reg = json.loads((root / "registry.json").read_text(encoding="utf-8"))
ids = {
    "unet_binary_resnet18_imagenet_partial",
    "smp_deeplabv3plus_resnet101_imagenet",
    "smp_unetplusplus_resnet101_imagenet",
    "smp_pspnet_resnet101_imagenet",
    "smp_fpn_resnet101_imagenet",
    "hf_segformer_b0_ade20k",
    "hf_segformer_b2_ade20k",
    "hf_upernet_swin_large_ade20k",
    "transunet_tiny_vit_tiny_patch16_imagenet",
    "segformer_mini_vit_tiny_patch16_imagenet",
}
for model in reg.get("models", []):
    mid = str(model.get("model_id", ""))
    if mid not in ids:
        continue
    bundle = root / str(model.get("bundle_dir", ""))
    rel = str(model.get("weights_path", ""))
    target = bundle / rel
    print(f"{mid:50s} -> {target}")
PY
```

### 2.5 Package repo for air-gap transfer

```bash
mkdir -p transfer_bundle
zip -r transfer_bundle/hydrideseg_repo_with_weights.zip . \
  -x ".git/*" ".venv/*" "__pycache__/*" ".pytest_cache/*" ".ruff_cache/*" "outputs/*"
sha256sum transfer_bundle/hydrideseg_repo_with_weights.zip > transfer_bundle/hydrideseg_repo_with_weights.zip.sha256
```

## 3. Air-Gapped HPC Machine - Restore and Validate

### 3.1 Copy, verify, extract

```bash
mkdir -p "$(dirname "$REPO_ROOT")"
cd "$(dirname "$REPO_ROOT")"
cp /transfer_media/hydrideseg_repo_with_weights.zip .
cp /transfer_media/hydrideseg_repo_with_weights.zip.sha256 .
sha256sum -c hydrideseg_repo_with_weights.zip.sha256
unzip -o hydrideseg_repo_with_weights.zip
cd "$REPO_ROOT"
```

### 3.2 HPC python environment

```bash
cd "$REPO_ROOT"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -e .
python -m pip install segmentation-models-pytorch
```

### 3.3 Validate pretrained payload in HPC

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

## 4. Dataset Contract

Expected dataset:

```text
${DATASET_DIR}/
  train/images/*.png
  train/masks/*.png
  val/images/*.png
  val/masks/*.png
  test/images/*.png
  test/masks/*.png
```

Quick check:

```bash
export DATASET_DIR=/path/to/your/mado_style
python - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["DATASET_DIR"])
for split in ["train", "val", "test"]:
    ni = len(list((root / split / "images").glob("*")))
    nm = len(list((root / split / "masks").glob("*")))
    print(f"{split}: images={ni} masks={nm}")
PY
```

## 5. Optional Integrity Gate (small debug run)

```bash
cd "$REPO_ROOT"
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

## 6. Real Campaign - All 10 Models (Scratch + Pretrained in one suite)

### 6.1 Create editable real-data suite config

```bash
cd "$REPO_ROOT"
cp configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.template.yml \
   configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.yml
```

### 6.2 Set dataset/output/seeds without manual YAML editing

```bash
export DATASET_DIR=/path/to/your/mado_style
export OUTPUT_ROOT=/path/to/hpc_outputs/top10_suite_scratch_vs_pretrained
export SUITE_CFG=$REPO_ROOT/configs/hydride/benchmark_suite.top5_scratch_vs_pretrained.realdata.yml

python - <<'PY'
import os, yaml
from pathlib import Path

cfg_path = Path(os.environ["SUITE_CFG"])
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
cfg["dataset_dir"] = os.environ["DATASET_DIR"]
cfg["output_root"] = os.environ["OUTPUT_ROOT"]
cfg["seeds"] = [42, 43, 44]
cfg["benchmark_mode"] = True
cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(cfg_path)
PY
```

### 6.3 Run full suite

```bash
cd "$REPO_ROOT"
source .venv/bin/activate
python scripts/hydride_benchmark_suite.py \
  --config "$SUITE_CFG" \
  --strict
```

## 7. Logs and Diagnostics (queue-friendly)

Per-suite:

- `${OUTPUT_ROOT}/logs/suite_events.jsonl` (structured suite timeline)

Per-run (`<run_tag>` = model + seed):

- `${OUTPUT_ROOT}/logs/<run_tag>/run_events.jsonl` (preflight/train/eval timings + status)
- `${OUTPUT_ROOT}/logs/<run_tag>/train.log` (full training stdout/stderr)
- `${OUTPUT_ROOT}/logs/<run_tag>/eval.log` (full evaluation stdout/stderr)

Live tail examples:

```bash
tail -f "${OUTPUT_ROOT}/logs/suite_events.jsonl"
tail -f "${OUTPUT_ROOT}/logs/unet_binary_scratch_seed42/train.log"
tail -f "${OUTPUT_ROOT}/logs/hf_upernet_swin_large_local_pretrained_seed42/eval.log"
```

## 8. Comparative Outputs and Key Metrics

Main outputs:

- `${OUTPUT_ROOT}/summary.json`
- `${OUTPUT_ROOT}/summary.html`

Compatibility outputs also written:

- `${OUTPUT_ROOT}/benchmark_summary.json`
- `${OUTPUT_ROOT}/benchmark_dashboard.html`
- `${OUTPUT_ROOT}/benchmark_summary.csv`
- `${OUTPUT_ROOT}/benchmark_aggregate.csv`

Important metrics now tracked across train/eval/benchmark tables:

- `pixel_accuracy`, `macro_f1`, `mean_iou`
- `weighted_f1`, `balanced_accuracy`, `cohen_kappa`
- binary diagnostics: `foreground_*`, `false_positive_rate`, `false_negative_rate`, `matthews_corrcoef`
- hydride science metrics: area/count/size/orientation error

Quick top-model view from aggregate CSV:

```bash
python - <<'PY'
import csv, os
from pathlib import Path

agg = Path(os.environ["OUTPUT_ROOT"]) / "benchmark_aggregate.csv"
rows = list(csv.DictReader(agg.open(encoding="utf-8")))
rows.sort(key=lambda r: float(r.get("quality_score", 0.0)), reverse=True)
for r in rows[:10]:
    print(
        r["model"],
        "quality=", f'{float(r.get("quality_score", 0.0)):.6f}',
        "miou=", f'{float(r.get("mean_mean_iou", 0.0)):.6f}',
        "dice=", f'{float(r.get("mean_foreground_dice", 0.0)):.6f}',
        "kappa=", f'{float(r.get("mean_cohen_kappa", 0.0)):.6f}',
        "runtime_s=", f'{float(r.get("mean_total_runtime_seconds", 0.0)):.2f}',
    )
PY
```

## 9. Direct Download Links + Exact Filenames (manual fallback)

Preferred workflow is `python scripts/download_pretrained_weights.py --targets all --force` because it assembles exact bundle structure automatically.

If manual download is required, use these URLs and keep the shown filenames unchanged:

| Model ID | Direct link(s) | Filename(s) from URL | Final expected local weight target used by config |
|---|---|---|---|
| `unet_binary_resnet18_imagenet_partial` | `https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet18.a1_in1k/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/unet_binary_resnet18_imagenet_partial/weights/unet_binary_resnet18_imagenet_partial_state_dict.pt` |
| `smp_deeplabv3plus_resnet101_imagenet` | `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/smp_deeplabv3plus_resnet101_imagenet/weights/smp_deeplabv3plus_resnet101_imagenet_state_dict.pt` |
| `smp_unetplusplus_resnet101_imagenet` | `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/smp_unetplusplus_resnet101_imagenet/weights/smp_unetplusplus_resnet101_imagenet_state_dict.pt` |
| `smp_pspnet_resnet101_imagenet` | `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/smp_pspnet_resnet101_imagenet/weights/smp_pspnet_resnet101_imagenet_state_dict.pt` |
| `smp_fpn_resnet101_imagenet` | `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/resnet101.a1h_in1k/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/smp_fpn_resnet101_imagenet/weights/smp_fpn_resnet101_imagenet_state_dict.pt` |
| `hf_segformer_b0_ade20k` | `https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/model.safetensors` ; `https://huggingface.co/nvidia/segformer-b0-finetuned-ade-512-512/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/hf_segformer_b0_ade20k/hf_model/model.safetensors` (preferred) |
| `hf_segformer_b2_ade20k` | `https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512/resolve/main/pytorch_model.bin` | `pytorch_model.bin` | `pre_trained_weights/hf_segformer_b2_ade20k/hf_model/pytorch_model.bin` |
| `hf_upernet_swin_large_ade20k` | `https://huggingface.co/openmmlab/upernet-swin-large/resolve/main/model.safetensors` ; `https://huggingface.co/openmmlab/upernet-swin-large/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/hf_upernet_swin_large_ade20k/hf_model/model.safetensors` (preferred) |
| `transunet_tiny_vit_tiny_patch16_imagenet` | `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/transunet_tiny_vit_tiny_patch16_imagenet/weights/transunet_tiny_vit_tiny_patch16_imagenet_state_dict.pt` |
| `segformer_mini_vit_tiny_patch16_imagenet` | `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/model.safetensors` ; `https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k/resolve/main/pytorch_model.bin` | `model.safetensors` or `pytorch_model.bin` | `pre_trained_weights/segformer_mini_vit_tiny_patch16_imagenet/weights/segformer_mini_vit_tiny_patch16_imagenet_state_dict.pt` |

For provenance/citations see:

- `docs/pretrained_model_catalog.md`
- `docs/pretrained_model_catalog.json`
- `pre_trained_weights/metadata/*.meta.json`

