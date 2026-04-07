# Training Data Requirements

## Overview

The training/evaluation pipeline supports both:
- binary segmentation
- multi-class indexed segmentation

The expected canonical dataset layout is split-based:

```text
<dataset_dir>/
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

Image/mask pairing is by identical filename inside each split.

## Supported File Formats

Supported image and mask input extensions:
- `.png`
- `.jpg`
- `.jpeg`
- `.tif`
- `.tiff`
- `.bmp`

Internally, prepared training datasets are written as PNG files for consistency.

## Mask Requirements

Indexed mask mode (`mask_input_type: indexed`):
- masks must be 2D indexed/grayscale arrays
- background is usually index `0`
- foreground/classes are integer indices `>= 1`

Binary masks:
- both `{0,1}` and `{0,255}` are accepted
- `{0,255}` is normalized to `{0,1}` by the pipeline

Multi-class masks:
- use indexed values like `0,1,2,...`
- keep class-index semantics consistent across all splits

Optional RGB colormap mode (`mask_input_type: rgb_colormap`):
- masks can be 3-channel RGB masks
- you must supply `mask_colormap`
- conversion is strict by default (`mask_colormap_strict: true`): unknown colors fail fast

Supported `mask_colormap` forms:
- index-to-color:
  - `{"0":[0,0,0], "1":[255,0,0], "2":[0,255,0]}`
- color-to-index:
  - `{"0,0,0":0, "255,0,0":1, "0,255,0":2}`

`mask_input_type: auto`:
- accepts 2D indexed masks directly
- converts RGB masks only when `mask_colormap` is provided

## If You Already Have Train/Val/Test Folders

If all three split folders (`train`, `val`, `test`) exist with `images/` + `masks/`, the pipeline uses them directly.

## If You Provide Unsplit Data (Supported)

If split folders are not provided, the pipeline can auto-prepare from unsplit folders:

Supported unsplit patterns:
- `<dataset_dir>/source` + `<dataset_dir>/masks`
- `<dataset_dir>/images` + `<dataset_dir>/masks`
- `<dataset_dir>/data/source` + `<dataset_dir>/data/masks`
- `<dataset_dir>/data/images` + `<dataset_dir>/data/masks`

Default split:
- `80:10:10` (`train:val:test`)
- configurable via YAML/CLI

Default behavior:
- leakage-aware split with deterministic seed (configurable)
- original stem is retained
- a stable global ID suffix is added:
  - `originalName_000001.png`
  - `originalName_000002.png`
- mapping is recorded in `dataset_prepare_manifest.json`

Leakage-aware split controls:
- `split_strategy`: `leakage_aware` (default) or `random`
- `leakage_group_mode`:
  - `suffix_aware` (default): groups common augmentation variants like `_aug1`, `_crop3`, `_tile10`, `_rot90`, `_flip`
  - `stem`: groups by exact stem
  - `regex`: groups by `leakage_group_regex` capture

## Programmatic Tracking IDs

Prepared filenames include global `_ID` suffix for consistent referencing in:
- training/evaluation summaries
- tracked sample panels
- automated downstream scripts

Mapping fields include:
- original image path
- original mask path
- new filename
- split assignment
- source leakage group
- global numeric ID (`id` and `global_id`)

## Commands

Prepare dataset explicitly:

```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

GUI workflow equivalent:
- Open `Workflow Hub` -> `Dataset Prep + QA`
- Use `Preview Dataset Plan` before running prepare
- Optionally run QA and then launch training with QA gate enabled

Prepare dataset with RGB colormap masks:

```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set mask_input_type=rgb_colormap \
  --set 'mask_colormap={"0":[0,0,0],"1":[255,0,0],"2":[0,255,0]}'
```

Train (auto-prepare enabled by default):

```bash
microseg-cli train --config configs/train.default.yml
```

Evaluate (auto-prepare enabled by default):

```bash
microseg-cli evaluate --config configs/evaluate.default.yml
```

## Recommended Validation

Run dataset QA before training:

```bash
microseg-cli dataset-qa --config configs/dataset_qa.default.yml --strict
```

## Feedback-Derived Active-Learning Dataset

Centralized feedback records (`microseg.feedback_record.v1`) can be converted into
train/val/test datasets using:

```bash
microseg-cli feedback-build-dataset --config configs/feedback_build_dataset.default.yml
```

Policy defaults:
- corrected masks: included with weight `1.0`
- thumbs-up without correction: included as pseudo-labels with low weight (`0.2`)
- thumbs-down without correction: excluded from training dataset and routed to review queue during ingest

Generated layout matches canonical split format:

```text
<output_dir>/
  train/{images,masks,metadata}
  val/{images,masks,metadata}
  test/{images,masks,metadata}
  sample_weights.csv
  dataset_manifest.json
```

Threshold-based retraining trigger (corrected-count or elapsed-days gate):

```bash
microseg-cli feedback-train-trigger --config configs/feedback_train_trigger.default.yml
```

This trigger never auto-promotes models; promotion remains human-gated through benchmark/run-review policy flow.


Enable binary auto-normalization for legacy masks (`0` background, any non-zero foreground):

```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set binary_mask_normalization=nonzero_foreground
```
The same setting is available in `train` and `evaluate`. When non-canonical foreground values
(for example `78`, `80`, `255`) are remapped to class `1`, a warning is emitted and execution continues.
