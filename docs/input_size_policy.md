# Input Size Policy For Segmentation Training

Variable image sizes in a single training batch cause PyTorch default collation (`torch.stack`) to fail because tensors must have matching dimensions. This repository now uses a configurable **Input Size Policy** so image/mask tensors are normalized to a consistent shape before batching.

## Why this matters

- **Stability:** prevents DataLoader stack errors from mixed-size microscopy images.
- **Memory safety:** transformer backends (e.g., HF SegFormer) can OOM on large raw images; fixed `input_hw` bounds memory.
- **Reproducibility:** deterministic val/test transforms keep metrics stable across runs.

## Config fields

Training configs (e.g. `configs/train.default.yml`) support:

- `input_hw: [512, 512]`
- `input_policy: random_crop | resize | letterbox | center_crop`
- `val_input_policy: random_crop | resize | letterbox | center_crop`
- `keep_aspect: true` (used by `letterbox`)
- `pad_value_image: 0.0`
- `pad_value_mask: 0`
- `image_interpolation: bilinear | bicubic | nearest`
- `mask_interpolation: nearest` (**must remain nearest**)
- `require_divisible_by: 32` (pads to next multiple after policy)
- `dataloader_collate: default | pad_to_max`

## Recommended microscopy defaults

- **Train:** `input_policy=random_crop`, `input_hw=[512,512]`
- **Val/Test:** `val_input_policy=letterbox` (or `center_crop` if desired)
- **Collate:** keep `dataloader_collate=default`; use `pad_to_max` only as fallback/debug mode.

## CLI examples

UNet fixed input training:

```bash
python3 scripts/microseg_cli.py train \
  --config configs/hydride/train.unet_binary.baseline.yml \
  --dataset-dir ./data/HydrideData6.0/mado_style \
  --output-dir ./outputs/tmp_unet \
  --set input_hw=[512,512] \
  --set input_policy=random_crop
```

SegFormer memory-safer training (same input cap + AMP + accumulation):

```bash
python3 scripts/microseg_cli.py train \
  --config configs/hydride/train.hf_segformer_b0_scratch.yml \
  --dataset-dir ./data/HydrideData6.0/mado_style \
  --output-dir ./outputs/tmp_segformer \
  --set input_hw=[512,512] \
  --set input_policy=random_crop \
  --set amp_enabled=true \
  --set grad_accum_steps=4
```

## AMP and CUDA allocator guidance (HPC)

When using deterministic/large jobs, these environment settings can improve reproducibility and memory behavior:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

These are **not forced in code**; set them in your shell, scheduler script, or run wrapper.

## Scientific quantification note

Resizing and cropping can change the pixel-to-physical-scale relationship if metadata is not carried through. For quantitative workflows, record pixel size metadata and make transform choices explicit in run manifests and reports.
