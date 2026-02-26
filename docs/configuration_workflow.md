# Configuration Workflow

## Objective

All major pipelines are config-driven with YAML base files and `--set` overrides.

## YAML Files

Reference templates:
- `configs/inference.default.yml`
- `configs/train.default.yml`
- `configs/evaluate.default.yml`
- `configs/package.default.yml`
- `configs/phase_gate.default.yml`
- `configs/registry_validation.default.yml`
- `configs/dataset_split.default.yml`
- `configs/dataset_qa.default.yml`
- `configs/dataset_prepare.default.yml`
- `configs/hpc_ga.default.yml`

## CLI Usage

Inference:
```bash
microseg-cli infer --config configs/inference.default.yml --set params.area_threshold=120 --set include_analysis=true
```

Dataset packaging:
```bash
microseg-cli package --config configs/package.default.yml --set train_ratio=0.75 --set val_ratio=0.15
```

Leakage-aware split planning:
```bash
microseg-cli dataset-split --config configs/dataset_split.default.yml
```

Unsplit source/masks to split-layout preparation:
```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

Leakage-aware split controls for unsplit auto-prepare:
```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set split_strategy=leakage_aware \
  --set leakage_group_mode=suffix_aware
```

RGB mask colormap conversion example:
```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set mask_input_type=rgb_colormap \
  --set 'mask_colormap={"0":[0,0,0],"1":[255,0,0],"2":[0,255,0]}'
```
If `mask_colormap` is omitted for RGB masks, the class-color mapping is resolved from:
1. `--class-map-path` override
2. `MICROSEG_CLASS_MAP_PATH` environment variable
3. `configs/segmentation_classes.json`
4. builtin fallback (`background`/`feature`)

Training:
```bash
microseg-cli train --config configs/train.default.yml --set max_samples=300000 --set epochs=12
```
Common training backend options:
- `backend=unet_binary` (default)
- `backend=smp_unet_resnet18` (SMP U-Net with ResNet18 encoder)
- `backend=hf_segformer_b0` (Hugging Face SegFormer-B0, scratch init)
- `backend=hf_segformer_b2` (Hugging Face SegFormer-B2, scratch init)
- `backend=hf_segformer_b5` (Hugging Face SegFormer-B5, scratch init)
- `backend=transunet_tiny`
- `backend=segformer_mini`
- `backend=torch_pixel`
- `backend=sklearn_pixel`

UNet resume example:
```bash
microseg-cli train --config configs/train.default.yml --set resume_checkpoint=outputs/training/last_checkpoint.pt
```

Transformer backend example (scratch, no transfer learning):
```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set backend=hf_segformer_b0 \
  --set model_architecture=hf_segformer_b0
```

Local pretrained transfer-learning example (air-gap ready):
```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set backend=hf_segformer_b0 \
  --set model_architecture=hf_segformer_b0 \
  --set pretrained_init_mode=local \
  --set pretrained_model_id=hf_segformer_b0_ade20k \
  --set pretrained_registry_path=pre_trained_weights/registry.json
```

Local pretrained transformer variants:
```bash
microseg-cli train --config configs/train.default.yml \
  --set backend=hf_segformer_b2 \
  --set model_architecture=hf_segformer_b2 \
  --set pretrained_init_mode=local \
  --set pretrained_model_id=hf_segformer_b2_ade20k \
  --set pretrained_registry_path=pre_trained_weights/registry.json

microseg-cli train --config configs/train.default.yml \
  --set backend=hf_segformer_b5 \
  --set model_architecture=hf_segformer_b5 \
  --set pretrained_init_mode=local \
  --set pretrained_model_id=hf_segformer_b5_ade20k \
  --set pretrained_registry_path=pre_trained_weights/registry.json
```

Local pretrained SMP U-Net example:
```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set backend=smp_unet_resnet18 \
  --set model_architecture=smp_unet_resnet18 \
  --set pretrained_init_mode=local \
  --set pretrained_model_id=smp_unet_resnet18_imagenet \
  --set pretrained_registry_path=pre_trained_weights/registry.json
```

Local pretrained internal U-Net example:
```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set backend=unet_binary \
  --set model_architecture=unet_binary \
  --set pretrained_init_mode=local \
  --set pretrained_model_id=unet_binary_resnet18_imagenet_partial \
  --set pretrained_registry_path=pre_trained_weights/registry.json
```

Local pretrained internal transformer bootstrap examples:
```bash
microseg-cli train --config configs/train.default.yml \
  --set backend=transunet_tiny \
  --set model_architecture=transunet_tiny \
  --set pretrained_init_mode=local \
  --set pretrained_model_id=transunet_tiny_vit_tiny_patch16_imagenet \
  --set pretrained_registry_path=pre_trained_weights/registry.json

microseg-cli train --config configs/train.default.yml \
  --set backend=segformer_mini \
  --set model_architecture=segformer_mini \
  --set pretrained_init_mode=local \
  --set pretrained_model_id=segformer_mini_vit_tiny_patch16_imagenet \
  --set pretrained_registry_path=pre_trained_weights/registry.json
```
These two bundles are partial warm-start mappings from ViT-tiny; see `docs/pretrained_model_catalog.md` and
`pre_trained_weights/metadata/*.meta.json` for mapped/unmapped component details.

Binary mask normalization override example:
```bash
microseg-cli train --config configs/train.default.yml --set binary_mask_normalization=nonzero_foreground
```
`nonzero_foreground` maps any non-zero indexed pixel to class `1` (foreground), while preserving `0` as background.
`two_value_zero_background` is stricter and only remaps when masks contain exactly two values where one is `0`.

UNet validation tracking + reporting example:
```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set val_tracking_samples=8 \
  --set "val_tracking_fixed_samples=val_000.png|val_123.png" \
  --set write_html_report=true \
  --set progress_log_interval_pct=10
```

Evaluation:
```bash
microseg-cli evaluate --config configs/evaluate.default.yml --set split=test
```
Evaluation accepts torch checkpoints with `.pt`, `.pth`, or `.ckpt` suffixes.

Evaluation HTML + tracked sample panel example:
```bash
microseg-cli evaluate \
  --config configs/evaluate.default.yml \
  --set tracking_samples=12 \
  --set write_html_report=true
```

Dataset QA:
```bash
microseg-cli dataset-qa --config configs/dataset_qa.default.yml --strict
```

Frozen registry validation:
```bash
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```

Pretrained registry validation:
```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

Download/stage pretrained bundles on connected machine:
```bash
python scripts/download_pretrained_weights.py --targets all --force
```

Generate pretrained inventory report for reporting/manuscript provenance:
```bash
python scripts/pretrained_inventory_report.py \
  --registry-path pre_trained_weights/registry.json \
  --output-path outputs/pretrained_weights/inventory_report.json
```

Generate tiny smoke checkpoint for pipeline sanity checks:
```bash
python scripts/generate_smoke_checkpoint.py --force
```

HPC GA bundle generation:
```bash
microseg-cli hpc-ga-generate --config configs/hpc_ga.default.yml --dataset-dir outputs/prepared_dataset --output-dir outputs/hpc_ga_bundle
```

Air-gapped low-friction HPC GA bundle generation with backend->pretrained mapping:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset \
  --output-dir outputs/hpc_ga_bundle_airgap_pretrained
```

Top-5 scratch-only HPC GA profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_scratch.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_scratch
```

Top-5 local-pretrained HPC GA profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_airgap_pretrained
```

HPC GA feedback report generation:
```bash
microseg-cli hpc-ga-feedback-report \
  --config configs/hpc_ga.default.yml \
  --feedback-sources outputs/hpc_ga_bundle \
  --output-path outputs/hpc_ga_feedback/feedback_report.json
```

GPU-enabled runs (auto policy with CPU fallback):
```bash
microseg-cli train --config configs/train.default.yml --enable-gpu --device-policy auto
microseg-cli infer --config configs/inference.default.yml --enable-gpu --device-policy auto
microseg-cli evaluate --config configs/evaluate.default.yml --enable-gpu --device-policy auto
```

## GUI Usage

- Provide optional config path in the top `Config` field.
- Add comma-separated overrides in `key=value` form.
- Run segmentation; final parameters are merged as:
  - YAML base
  - GUI override entries
  - runtime image path
- Workflow Hub supports YAML profile save/load for:
  - `dataset_prepare`
  - `training`
  - `evaluation`
  - `hpc_ga`

## Override Conventions

- Dotted keys create nested structures (`params.crop=true`).
- Scalars parse into bool/int/float/null/string automatically.
- JSON objects/lists can be passed in `--set` values (for example `mask_colormap={...}`).
- JSON-like override values with malformed syntax now fail fast with a clear config error (no silent string fallback).

## Reproducibility

CLI inference/package commands persist `resolved_config.json` with outputs.
CLI training/evaluation commands also persist `resolved_config.json` beside artifacts/reports.
Training additionally writes:
- `report.json` with status/progress/ETA/history
- `training_report.html` for rapid visual review (tracked sample panels include per-image metrics, including epoch-by-epoch section)
- `eval_samples/epoch_XXX` tracked validation panels

Phase closeout checks:
```bash
microseg-cli phase-gate --config configs/phase_gate.default.yml --set phase_label=\"Phase N\"
```

HPC pretrained controls (in `configs/hpc_ga*.yml`):
- `pretrained_init_mode`: `scratch`, `auto`, `local`
- `pretrained_registry_path`: local pretrained registry path
- `pretrained_model_map`: backend->model_id mapping used by generated candidate scripts
