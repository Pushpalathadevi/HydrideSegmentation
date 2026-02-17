# Offline Pretrained Transfer Workflow

## Objective

Enable transfer learning on air-gapped systems using locally staged pretrained bundles.

This workflow supports the following local-pretrained backends:
- U-Net:
  - `smp_unet_resnet18` (local pretrained state dict)
- Transformers:
  - `hf_segformer_b0` (local Hugging Face model directory)
  - `hf_segformer_b2` (local Hugging Face model directory)
  - `hf_segformer_b5` (local Hugging Face model directory)

## Storage Contract

Local staging root:
- `pre_trained_weights/`

Required registry:
- `pre_trained_weights/registry.json`

Tracked template:
- `pre_trained_weights/registry.template.json`

Each model bundle contains:
- `metadata.json` with source/revision/framework details
- model artifacts (ignored by git)
- `SHA256SUMS.json` with file checksums

Validation command:
```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

Manuscript/reporting metadata contract:
- each model record should include `source`, `source_url`, `source_revision`
- each model record should include `license`
- each model record should include `citation_key` and/or `citation`
- training/evaluation artifacts persist selected pretrained provenance under `pretrained_init`

## Connected Machine Steps (Download + Stage)

1. Install dependencies:
```bash
pip install -r requirements-core.txt
pip install -e .
pip install segmentation-models-pytorch
```

2. Download/stage pretrained bundles:
```bash
python scripts/download_pretrained_weights.py --targets all --force
```

This materializes:
- `pre_trained_weights/hf_segformer_b0_ade20k/`
- `pre_trained_weights/hf_segformer_b2_ade20k/`
- `pre_trained_weights/hf_segformer_b5_ade20k/`
- `pre_trained_weights/smp_unet_resnet18_imagenet/`
- `pre_trained_weights/registry.json`

3. Validate staged artifacts:
```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

## Air-Gapped Machine Steps

1. Copy `pre_trained_weights/` from connected machine (external drive).
2. Validate copied artifacts:
```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```
3. Run training with local-pretrained initialization:
```bash
microseg-cli train --config configs/hydride/train.smp_unet_resnet18_local_pretrained.debug.yml
microseg-cli train --config configs/hydride/train.hf_segformer_b0_local_pretrained.debug.yml
microseg-cli train --config configs/hydride/train.hf_segformer_b2_local_pretrained.debug.yml
microseg-cli train --config configs/hydride/train.hf_segformer_b5_local_pretrained.debug.yml
```

## Debug End-To-End Integrity Run

Build tiny split dataset by duplicating one repository test image:
```bash
python scripts/build_debug_duplicate_dataset.py \
  --image-path test_data/syntheticHydrides.png \
  --output-dir outputs/debug_pretrained_dataset \
  --train-count 4 \
  --val-count 2
```

Train local-pretrained debug configs:
```bash
microseg-cli train --config configs/hydride/train.smp_unet_resnet18_local_pretrained.debug.yml
microseg-cli train --config configs/hydride/train.hf_segformer_b0_local_pretrained.debug.yml
microseg-cli train --config configs/hydride/train.hf_segformer_b2_local_pretrained.debug.yml
microseg-cli train --config configs/hydride/train.hf_segformer_b5_local_pretrained.debug.yml
```

Evaluate each run:
```bash
microseg-cli evaluate \
  --config configs/evaluate.default.yml \
  --dataset-dir outputs/debug_pretrained_dataset \
  --model-path outputs/debug_runs/smp_unet_resnet18_local/best_checkpoint.pt \
  --split val \
  --output-path outputs/debug_runs/smp_unet_resnet18_local/eval_val.json \
  --no-auto-prepare-dataset

microseg-cli evaluate \
  --config configs/evaluate.default.yml \
  --dataset-dir outputs/debug_pretrained_dataset \
  --model-path outputs/debug_runs/hf_segformer_b0_local/best_checkpoint.pt \
  --split val \
  --output-path outputs/debug_runs/hf_segformer_b0_local/eval_val.json \
  --no-auto-prepare-dataset

microseg-cli evaluate \
  --config configs/evaluate.default.yml \
  --dataset-dir outputs/debug_pretrained_dataset \
  --model-path outputs/debug_runs/hf_segformer_b2_local/best_checkpoint.pt \
  --split val \
  --output-path outputs/debug_runs/hf_segformer_b2_local/eval_val.json \
  --no-auto-prepare-dataset

microseg-cli evaluate \
  --config configs/evaluate.default.yml \
  --dataset-dir outputs/debug_pretrained_dataset \
  --model-path outputs/debug_runs/hf_segformer_b5_local/best_checkpoint.pt \
  --split val \
  --output-path outputs/debug_runs/hf_segformer_b5_local/eval_val.json \
  --no-auto-prepare-dataset
```

## Low-Friction HPC Sweep (Air-Gapped)

Use the dedicated config with backend-to-model mapping pre-filled:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset \
  --output-dir outputs/hpc_ga_bundle_airgap_pretrained
```

Pretrained modes:
- `scratch`: always scratch init
- `auto`: use local pretrained only when `pretrained_model_map[backend]` exists
- `local`: require pretrained mapping for each pretrained-capable backend in `architectures`

Generated HPC job scripts now set:
- `backend`
- `model_architecture` (matches backend to avoid config drift)
- pretrained init settings (`pretrained_init_mode`, `pretrained_model_id`, registry/checksum flags) when applicable

## Training Config Keys (New)

- `pretrained_init_mode`: `scratch` or `local`
- `pretrained_model_id`: model identifier in `pre_trained_weights/registry.json`
- `pretrained_bundle_dir`: optional direct local bundle path (fallback when `pretrained_model_id` is not used)
- `pretrained_registry_path`: registry JSON path
- `pretrained_strict`: strict state-dict loading for torch checkpoints
- `pretrained_ignore_mismatched_sizes`: HF load option (classifier-head mismatch handling)
- `pretrained_verify_sha256`: validate checksums before training startup

HPC GA config keys for low-friction air-gap runs:
- `pretrained_init_mode`: `scratch`, `auto`, or `local`
- `pretrained_registry_path`: local registry path used in generated scripts
- `pretrained_model_map`: backend-to-model mapping (YAML map, JSON object string, or `backend=model_id` CSV)
- `pretrained_verify_sha256`: generated script override for registry checksum validation
- `pretrained_ignore_mismatched_sizes`: generated script override for HF local load behavior
- `pretrained_strict`: generated script override for strict torch state-dict loading

## Reproducibility Notes

- Training reports/manifests include `model_initialization` and `pretrained_init` provenance.
- All pretrained path references are local and do not require network at train time.
- Use pinned copied bundles to keep runs repeatable across air-gapped machines.
- For publication/reporting metadata, use `docs/pretrained_model_catalog.md`.
