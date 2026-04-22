# 05 Paired Dataset Preparation And Training (CLI)

This tutorial is the canonical beginner workflow for machine-learning dataset preparation in this repository.
It starts from one raw paired folder, creates a tiny reproducible tutorial dataset, prepares leak-aware train/val/test splits, applies train-only shadow and blur augmentation, and runs a tiny CPU-safe UNet training job.

The tutorial is driven by YAML files:

- dataset prep config: [`configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml`](../../configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml)
- training config: [`configs/tutorials/train.tiny_unet_from_prepared.yml`](../../configs/tutorials/train.tiny_unet_from_prepared.yml)

If you need environment setup first, read [`../cli_windows_linux.md`](../cli_windows_linux.md).

## Quick Success Path

From the repository root:

```bash
python scripts/generate_tutorial_dataset.py
python scripts/microseg_cli.py prepare_dataset --config configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml
python scripts/microseg_cli.py train --config configs/tutorials/train.tiny_unet_from_prepared.yml
```

Expected artifact roots:

- raw pairs: `tmp/tutorial_demo/raw_pairs/`
- prepared dataset: `tmp/tutorial_demo/prepared_dataset/`
- training run: `tmp/tutorial_demo/training_unet/`

## What The Raw Paired Input Folder Looks Like

The paired-folder CLI expects one folder containing image files and their masks together.

Accepted naming examples:

- `image_001.jpg` with `image_001_mask.png`
- `image_002.png` with `image_002_mask.png`
- `image_003.jpg` with `image_003.png` only when `same_stem_pairing.enabled: true`

The tutorial generator writes 12 pairs using grouped names such as:

- `hydride_plate_family_a_aug00.png`
- `hydride_plate_family_a_aug00_mask.png`
- `hydride_plate_family_a_aug01.png`
- `hydride_plate_family_a_aug01_mask.png`

Those suffixes are deliberate.
The paired prep command uses `split_strategy: leakage_aware` and `leakage_group_mode: suffix_aware`, so the two variants from the same family stay in the same split.

Some real lab folders use same-stem cross-extension pairs instead:

- `hydride_plate_family_a_aug00.jpg`
- `hydride_plate_family_a_aug00.png`

That format is also supported, but only with an explicit YAML opt-in:

```yaml
same_stem_pairing:
  enabled: true
  image_extensions:
    - .jpg
    - .jpeg
  mask_extensions:
    - .png
```

Pairing precedence is deliberate:

1. try `mask_name_patterns` first
2. only if no pattern match exists, try same-stem cross-extension pairing
3. fail under `strict_pairing: true` if ambiguity remains

That keeps the default `_mask` workflow stable while supporting `.jpg` image plus same-stem `.png` mask datasets.

## Step 1. Generate The Tiny Tutorial Dataset

Run:

```bash
python scripts/generate_tutorial_dataset.py
```

What this does:

- reads the bundled test image at `test_data/3PB_SRT_data_generation_1817_OD_side1_8.png`
- generates a pseudo-mask with the repository's conventional hydride segmentation path
- duplicates the example into 6 leakage groups with 2 variants each
- writes 12 paired image/mask examples into `tmp/tutorial_demo/raw_pairs/`
- writes `tutorial_dataset_manifest.json`

Verify these files exist:

- `tmp/tutorial_demo/raw_pairs/tutorial_dataset_manifest.json`
- `tmp/tutorial_demo/raw_pairs/hydride_plate_family_a_aug00.png`
- `tmp/tutorial_demo/raw_pairs/hydride_plate_family_a_aug00_mask.png`

Expected raw folder shape:

```text
tmp/tutorial_demo/raw_pairs/
  hydride_plate_family_a_aug00.png
  hydride_plate_family_a_aug00_mask.png
  ...
  hydride_plate_family_f_aug01.png
  hydride_plate_family_f_aug01_mask.png
  tutorial_dataset_manifest.json
```

## Step 2. Prepare The Dataset From The Paired Folder

Run:

```bash
python scripts/microseg_cli.py prepare_dataset --config configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml
```

This uses the paired-folder beginner path, not `dataset-prepare`.

What the command does:

1. reads image+mask pairs from `tmp/tutorial_demo/raw_pairs/`
2. applies leakage-aware split planning
3. creates an original split policy of 8 train, 2 val, 2 test
4. resizes to `64x64` with `short_side_to_target_crop`
5. converts masks to binary training masks
6. applies one shadow+blur augmentation variant to each training sample only
7. exports a MaDo-style dataset under `tmp/tutorial_demo/prepared_dataset/mado/`
8. writes manifests, QA reports, and augmentation debug panels

Important note:

- `source_split_counts` means the original split policy before augmentation
- `split_counts` means the final materialized counts after augmentation

With this tutorial config you should see:

- `source_split_counts = {"train": 8, "val": 2, "test": 2}`
- `split_counts = {"train": 16, "val": 2, "test": 2}`

Expected prepared folder shape:

```text
tmp/tutorial_demo/prepared_dataset/
  manifest.json
  dataset_qa_report.json
  dataset_qa_report.html
  debug_augmentation/
  mado/
    train/
      images/
      masks/
    val/
      images/
      masks/
    test/
      images/
      masks/
```

Verify these artifacts:

- `tmp/tutorial_demo/prepared_dataset/manifest.json`
- `tmp/tutorial_demo/prepared_dataset/dataset_qa_report.json`
- `tmp/tutorial_demo/prepared_dataset/debug_augmentation/`
- `tmp/tutorial_demo/prepared_dataset/mado/train/images/`

## Step 3. Train From The Prepared Dataset

Run:

```bash
python scripts/microseg_cli.py train --config configs/tutorials/train.tiny_unet_from_prepared.yml
```

This tutorial training config uses:

- backend: `unet_binary`
- CPU runtime
- `epochs: 1`
- `batch_size: 2`
- `model_base_channels: 8`
- `input_hw: [64, 64]`
- `auto_prepare_dataset: false`

That last setting matters.
The training job consumes the dataset you already prepared.
It does not run another preparation stage internally.

Verify these outputs:

- `tmp/tutorial_demo/training_unet/report.json`
- `tmp/tutorial_demo/training_unet/training_manifest.json`
- `tmp/tutorial_demo/training_unet/training_report.html`
- `tmp/tutorial_demo/training_unet/last_checkpoint.pt`

This is a smoke/tutorial training run.
It proves the end-to-end mechanics, not final scientific model quality.

## Dataset Prep Config, Option By Option

Primary file:
[`configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml`](../../configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml)

### Input and pairing

- `input_dir`: raw paired folder that contains both images and masks
- `output_dir`: root where manifests, QA, and exported dataset are written
- `styles`: export layout, here `mado`
- `strict_pairing`: fail if an image is missing its mask or vice versa
- `image_extensions`, `mask_extensions`, `mask_name_patterns`: control which files are treated as images and masks
- `same_stem_pairing.enabled`: opt-in support for `sample.jpg` + `sample.png` style raw folders
- `same_stem_pairing.image_extensions`: image-role extensions for same-stem ambiguous folders
- `same_stem_pairing.mask_extensions`: mask-role extensions for same-stem ambiguous folders

Why these exist:
They let you adapt the same CLI to real lab folders without changing code.

### Split policy

- `split_strategy: leakage_aware`: keep near-duplicate samples together
- `leakage_group_mode: suffix_aware`: treat names ending in `_augNN`, `_cropNN`, `_flip`, `_rotNN` as one family
- `train_pct`, `val_pct`: base split policy
- `max_val_examples`, `max_test_examples`: caps used here to force an exact 8/2/2 split from 12 samples
- `seed`: deterministic split seed

Why these exist:
They prevent subtle train/val/test leakage when one physical specimen is duplicated or augmented into multiple files.

### Mask handling

- `rgb_mask_mode`: read RGB masks using red-dominance rules
- `mask_r_min`, `mask_g_max`, `mask_b_max`: thresholds for red-foreground masks
- `allow_red_dominance_fallback`, `mask_red_*`: tolerate imperfect RGB masks
- `auto_otsu_for_noisy_grayscale`: rescue slightly noisy near-binary masks
- `empty_mask_action`: warn or fail if preprocessing destroys all foreground

Why these exist:
Real mask files are often messy.
These options make the conversion explicit and traceable instead of silently guessing.

### Resize and crop

- `target_size: [64, 64]`: tutorial-friendly output size
- `resize_policy: short_side_to_target_crop`: scale short side to target, then crop
- `crop_mode_train: random`: allow training diversity
- `crop_mode_eval: center`: stable validation/test crop

Why these exist:
Training backends usually need a consistent input shape, but the crop policy should still be deliberate.

### Augmentation

- `augmentation.enabled`: master switch
- `augmentation.apply_splits: [train]`: never augment val/test in the beginner default
- `variants_per_sample: 1`: one extra training sample per original training sample
- `shadow` and `blur`: image-only augmentations
- `augmentation.debug.enabled`: write before/after inspection panels

Why these exist:
The tutorial demonstrates a safe default: split first, then augment train only.

## Training Config, Focused Explanation

Primary file:
[`configs/tutorials/train.tiny_unet_from_prepared.yml`](../../configs/tutorials/train.tiny_unet_from_prepared.yml)

Important fields:

- `dataset_dir`: must point to the prepared dataset root containing `train/`, `val/`, and `test/`
- `output_dir`: where reports and checkpoints are written
- `backend`, `model_architecture`: selects the actual model path
- `auto_prepare_dataset: false`: consume the prepared dataset directly
- `epochs`, `batch_size`, `learning_rate`: basic training controls
- `input_hw`: expected model input size
- `enable_gpu: false`, `device_policy: cpu`: tutorial-safe runtime defaults
- `write_html_report: true`: keep one human-readable report artifact

Why this file is separate:
It makes the handoff between data preparation and training explicit.
The beginner can inspect and edit each phase independently.

## How To Change Behavior

Use YAML edits first.
That keeps the workflow reproducible and beginner-readable.

Common edits:

- change `input_dir` to your own paired folder
- change `target_size` to match the model you intend to train
- change `variants_per_sample` or disable augmentation
- change `output_dir` if you want a different scratch location
- increase `epochs` only after the smoke workflow works

You can still use CLI overrides, but treat them as temporary:

```bash
python scripts/microseg_cli.py train \
  --config configs/tutorials/train.tiny_unet_from_prepared.yml \
  --set epochs=5 \
  --set batch_size=4
```

## When To Use `dataset-prepare` Instead

Use `microseg-cli dataset-prepare` when your dataset is already organized as:

- `dataset/source` + `dataset/masks`, or
- `dataset/train|val|test/images|masks`

That command is the layout-based path.
This tutorial intentionally starts from the raw paired-folder path because it is the most common beginner situation.
