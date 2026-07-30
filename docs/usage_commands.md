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

## Model Installation

Inspect a trained checkpoint before installing it:

```bash
microseg-cli inspect-checkpoint --checkpoint path/to/best_checkpoint.pth
```

Install it for GUI and CLI inference:

```bash
microseg-cli install-model --checkpoint path/to/best_checkpoint.pth --model-id my_unet_v1 --nickname my_unet_v1_optical
```

List models with availability:

```bash
microseg-cli models --details
```

Remove a locally installed model:

```bash
microseg-cli uninstall-model --model-id my_unet_v1 --delete-checkpoint
```

Walkthrough: [`gui_model_integration_guide.md`](gui_model_integration_guide.md)

## Inference

Single image:

```bash
microseg-cli infer --config configs/inference.default.yml
```

If you omit `--model` / `--model-name`, the CLI defaults to the first discovered trained model.
For ML-backed models, the CLI now uses the same default preprocessing as the GUI unless you override it in YAML:

- resize preserves aspect ratio to a `512` long side by default
- auto-contrast is enabled by default
- the preprocessing block is recorded in the exported manifests

To disable that behavior for a specific run, edit `gui_preprocess.enabled: false` in the inference YAML or override it with `--set`.

Recursive folder inference:

```bash
microseg-cli infer \
  --config configs/inference.default.yml \
  --image-dir data/sample_images \
  --recursive \
  --glob-patterns "*.png,*.tif,*.tiff,*.jpg,*.jpeg" \
  --model "Hydride ML (UNet)"
```

To add a new model so it appears here and in the GUI, edit `frozen_checkpoints/model_registry.json` or `frozen_checkpoints/model_registry.local.json` and restart the app.

## Complete ML And Quantification Pipeline

The `infer` command performs model inference and automatically exports quantification reports, including area fraction, hydride count, size/orientation distributions, and Fn metrics. Enable Fn debug artifacts and distribution charts explicitly as shown below.

### Windows PowerShell

Run from the repository root after activating `.venv`:

```powershell
python scripts\microseg_cli.py dataset-prepare --config configs\dataset_prepare.default.yml --dataset-dir data\my_dataset --output-dir outputs\packaged_dataset
python scripts\microseg_cli.py train --config configs\train.default.yml --dataset-dir outputs\packaged_dataset --output-dir outputs\training
python scripts\microseg_cli.py evaluate --config configs\evaluate.default.yml --dataset-dir outputs\packaged_dataset --model-path outputs\training\best_checkpoint.pt --output-path outputs\evaluation\model_eval.json
python scripts\microseg_cli.py infer --config configs\inference.default.yml --image-dir data\sample_images --output-dir outputs\inference\ml --model-name "Hydride ML (UNet)" --set result_export.write_distribution_charts=true --set result_export.write_fn_debug_artifacts=true --set result_export.report_decimal_places=2
```

For one image, replace `--image-dir data\sample_images` with `--image test_data\syntheticHydrides.png`.

### Linux Or macOS

Run from the repository root after activating `.venv`:

```bash
python scripts/microseg_cli.py dataset-prepare --config configs/dataset_prepare.default.yml --dataset-dir data/my_dataset --output-dir outputs/packaged_dataset
python scripts/microseg_cli.py train --config configs/train.default.yml --dataset-dir outputs/packaged_dataset --output-dir outputs/training
python scripts/microseg_cli.py evaluate --config configs/evaluate.default.yml --dataset-dir outputs/packaged_dataset --model-path outputs/training/best_checkpoint.pt --output-path outputs/evaluation/model_eval.json
python scripts/microseg_cli.py infer --config configs/inference.default.yml --image-dir data/sample_images --output-dir outputs/inference/ml --model-name "Hydride ML (UNet)" --set result_export.write_distribution_charts=true --set result_export.write_fn_debug_artifacts=true --set result_export.report_decimal_places=2
```

For one image, replace `--image-dir data/sample_images` with `--image test_data/syntheticHydrides.png`.

The prepared dataset must contain an organized source/mask layout accepted by `dataset-prepare`. For raw paired files, use `prepare_dataset` first; see [`cli_windows_linux.md`](cli_windows_linux.md) for the paired-folder recipe. The ML checkpoint must be available at the path supplied to `evaluate` and registered or otherwise discoverable by the inference workflow.

## Conventional Segmentation And Quantification

The conventional path requires no trained checkpoint. It produces the same result-package structure and quantification outputs, using deterministic image-processing parameters from the inference configuration.

### Windows PowerShell

```powershell
python scripts\microseg_cli.py infer --config configs\inference.default.yml --image-dir data\sample_images --output-dir outputs\inference\conventional --model-name "Hydride Conventional" --set params.area_threshold=95 --set result_export.write_distribution_charts=true --set result_export.write_fn_debug_artifacts=true --set result_export.report_decimal_places=2
```

### Linux Or macOS

```bash
python scripts/microseg_cli.py infer --config configs/inference.default.yml --image-dir data/sample_images --output-dir outputs/inference/conventional --model-name "Hydride Conventional" --set params.area_threshold=95 --set result_export.write_distribution_charts=true --set result_export.write_fn_debug_artifacts=true --set result_export.report_decimal_places=2
```

Each exported image receives the top-right Fn box. Inspect `results_summary.json` for machine-readable scalar metrics, `results_report.html` for the report, `results_metrics.csv` for tabular comparison, and the `*_fn_feature_table.csv` files for per-hydride audit. Both `fn_count` and `fn_length_weighted` use retained components after `params.area_threshold` and `result_export.min_feature_pixels` filtering.

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
