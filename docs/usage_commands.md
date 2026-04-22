# Usage Commands

This page is the short command index.
If you are new to the repository, start with:

- [`cli_windows_linux.md`](cli_windows_linux.md) for environment activation and import troubleshooting
- [`tutorials/05_paired_dataset_preparation_and_training_cli.md`](tutorials/05_paired_dataset_preparation_and_training_cli.md) for the complete paired-folder dataset and training walkthrough
- [`data_preparation.md`](data_preparation.md) for the paired-vs-layout preparation comparison

## Beginner Quick Path

Build the docs:

```bash
python scripts/build_docs.py --html-only
```

Generate the tiny tutorial dataset:

```bash
python scripts/generate_tutorial_dataset.py
```

Prepare the dataset from raw paired files:

```bash
python scripts/microseg_cli.py prepare_dataset --config configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml
```

Train the tiny CPU-safe UNet tutorial run:

```bash
python scripts/microseg_cli.py train --config configs/tutorials/train.tiny_unet_from_prepared.yml
```

## Inference

Single image:

```bash
microseg-cli infer --config configs/inference.default.yml
```

Recursive folder inference:

```bash
microseg-cli infer \
  --config configs/inference.default.yml \
  --image-dir data/sample_images \
  --recursive \
  --glob-patterns "*.png,*.tif,*.tiff,*.jpg,*.jpeg"
```

## Dataset Preparation

Primary beginner path, raw paired folder:

```bash
python scripts/microseg_cli.py prepare_dataset --config configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml
```

For raw folders shaped like `sample.jpg` + `sample.png`, enable the `same_stem_pairing` block in the YAML config first.

Backward-compatible wrapper:

```bash
python hydride_segmentation/prepare_dataset.py --input-dir tmp/tutorial_demo/raw_pairs --output tmp/tutorial_demo/prepared_dataset
```

Secondary path, already organized `source/masks` or split layout:

```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

## Training

Default training config:

```bash
microseg-cli train --config configs/train.default.yml
```

Tutorial training config:

```bash
python scripts/microseg_cli.py train --config configs/tutorials/train.tiny_unet_from_prepared.yml
```

## Desktop GUI

```bash
hydride-gui
```

Qt with explicit UI config:

```bash
hydride-gui --ui-config configs/app/desktop_ui.default.yml
```

## Docs

HTML:

```bash
python scripts/build_docs.py --html-only
```

HTML + PDF:

```bash
python scripts/build_docs.py
```
