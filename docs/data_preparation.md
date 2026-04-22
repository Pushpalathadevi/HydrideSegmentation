# Data Preparation

This page answers the four questions a beginner usually has first.

## 1. What input folder is expected?

For the primary beginner workflow, use one paired folder that contains both images and masks:

```text
raw_pairs/
  image_001.jpg
  image_001_mask.png
  image_002.png
  image_002_mask.png
```

The paired-folder CLI also supports alternate mask patterns defined by `mask_name_patterns`.

## 2. What command should I run?

Primary beginner path:

```bash
python scripts/microseg_cli.py prepare_dataset --config configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml
```

Backward-compatible wrapper:

```bash
python hydride_segmentation/prepare_dataset.py --input-dir tmp/tutorial_demo/raw_pairs --output tmp/tutorial_demo/prepared_dataset
```

Secondary layout-based path:

```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

## 3. Which YAML file controls behavior?

Primary beginner tutorial config:

- [`configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml`](../configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml)

Secondary layout-based config:

- [`configs/dataset_prepare.default.yml`](../configs/dataset_prepare.default.yml)

Training config that consumes the prepared tutorial dataset:

- [`configs/tutorials/train.tiny_unet_from_prepared.yml`](../configs/tutorials/train.tiny_unet_from_prepared.yml)

## 4. What output layout is produced?

The paired-folder beginner config writes a MaDo-style dataset plus reports:

```text
prepared_dataset/
  manifest.json
  dataset_qa_report.json
  dataset_qa_report.html
  debug_augmentation/
  mado/
    train/images
    train/masks
    val/images
    val/masks
    test/images
    test/masks
```

## Which Preparation Path Should I Use?

| Workflow | Use it when | Main command | What it does |
|---|---|---|---|
| `prepare_dataset` | You have raw image+mask pairs in one folder | `python scripts/microseg_cli.py prepare_dataset ...` | Pairing, mask conversion, leak-aware split creation, resize/crop, augmentation, export |
| `dataset-prepare` | You already have `source/masks` or split folders | `microseg-cli dataset-prepare ...` | Split planning or augmentation for an already organized dataset |
| `train --auto_prepare_dataset` | You want training to prepare an unsplit dataset on the fly | `microseg-cli train --config ...` | Training-time preparation before the trainer starts |

Beginner default:

- use `prepare_dataset` first
- use `dataset-prepare` only when your files are already organized into dataset layouts

## Paired-Folder Naming Rules

Default paired-folder matching is controlled by:

- `image_extensions`
- `mask_extensions`
- `mask_name_patterns`

Common accepted pairs:

- `sample.jpg` + `sample_mask.png`
- `sample.png` + `sample_mask.png`
- `sample.tif` + `sample_mask.tif`
- `sample.jpg` + `sample.png` only when `same_stem_pairing.enabled: true`

The tutorial config uses:

```yaml
mask_name_patterns:
  - "{stem}_mask.png"
  - "{stem}.png"
```

## How Pair Discovery Works

Pair discovery is explicit and deterministic.
The paired-folder path does not guess silently between ambiguous files.

Current precedence order:

1. list candidate images from `image_extensions`
2. exclude stems containing `_mask`
3. try `mask_name_patterns` in the order you wrote them
4. if no pattern match exists and `same_stem_pairing.enabled: true`, try same-stem cross-extension matching
5. if `strict_pairing: true`, fail when any image or mask remains unmatched

This means the default beginner workflow remains unchanged for datasets like:

- `sample_001.jpg`
- `sample_001_mask.png`

It also means a folder like this is intentionally treated as ambiguous unless you opt in:

- `sample_001.jpg`
- `sample_001.png`

Without the opt-in block, the `.png` file is still a valid image extension candidate in many workflows, so the pipeline refuses to silently assume it is the mask.

## Same-Stem Cross-Extension Pairing

Use this only when your real raw data follows this pattern:

- `hydride_plate_family_a_aug00.jpg` as the raw image
- `hydride_plate_family_a_aug00.png` as the mask

Enable it in YAML like this:

```yaml
image_extensions:
  - .jpg
  - .jpeg
  - .png
mask_extensions:
  - .png
mask_name_patterns:
  - "{stem}_mask.png"
  - "{stem}.png"

same_stem_pairing:
  enabled: true
  image_extensions:
    - .jpg
    - .jpeg
  mask_extensions:
    - .png
```

What this does:

- keeps the normal `_mask` pattern workflow working first
- only falls back to same-stem pairing when pattern matching did not find a mask
- treats `.jpg` and `.jpeg` as image-role files in the ambiguous same-stem case
- treats `.png` as the mask-role file in the ambiguous same-stem case
- prevents the same-stem `.png` mask from also being counted as a source image

Recommended rule:

- keep `same_stem_pairing.enabled: false` unless your dataset really uses extension-only role separation

That makes the dataset contract obvious to the next user and keeps debugging simple.

## Leak-Aware Split Policy

The paired-folder CLI now supports the same split-policy concepts the layout-based workflow already had:

- `split_strategy: leakage_aware`
- `leakage_group_mode: suffix_aware | stem | regex`
- `leakage_group_regex`

With `suffix_aware`, names like these stay together:

- `family_a_aug00`
- `family_a_aug01`
- `family_a_rot90`

This prevents near-duplicate samples from leaking across train, val, and test.

## What The Reports Mean

`manifest.json` includes:

- resolved config
- `source_split_counts` before augmentation
- `split_counts` after augmentation
- `group_to_split`
- per-record export paths
- per-record mask statistics and warnings

`dataset_qa_report.json` includes:

- pairing diagnostics
- `base_split_counts`
- final `split_counts`
- read failures
- empty-output-mask summary
- augmentation summary
- elapsed seconds

This split-count distinction matters.
If train-only augmentation is enabled, the final training file count will be larger than the original split count.

## Augmentation Schema

The paired-folder and layout-based preparation workflows use the same augmentation block:

```yaml
augmentation:
  enabled: true
  seed: 42
  stage: post_resize
  apply_splits: [train]
  variants_per_sample: 1
  operations:
    - name: shadow
      probability: 1.0
      parameters:
        intensity_range: [20, 25]
        count_range: [1, 1]
    - name: blur
      probability: 1.0
      parameters:
        kernel_size_range: [3, 3]
        count_range: [1, 1]
```

Important fields:

- `apply_splits`: which splits can be augmented
- `variants_per_sample`: how many augmented samples are created per original sample
- `stage`: `pre_resize` or `post_resize`
- `debug.enabled`: writes `debug_augmentation/` panels and metadata

The beginner default is train-only augmentation.

## Why The Tutorial Uses YAML-First Control

The repository is designed around config-driven workflows.
Editing a small YAML file is easier to review, repeat, and share than a long command with many overrides.

For the full step-by-step beginner workflow, continue to:

- [`tutorials/05_paired_dataset_preparation_and_training_cli.md`](tutorials/05_paired_dataset_preparation_and_training_cli.md)
- [`cli_windows_linux.md`](cli_windows_linux.md)
