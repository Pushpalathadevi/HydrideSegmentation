# CLI Quickstart On Windows And Linux

This page is the operational reference for running the repository from the command line.
The commands assume you are in the repository root.

For the full beginner workflow, continue to [`tutorials/05_paired_dataset_preparation_and_training_cli.md`](tutorials/05_paired_dataset_preparation_and_training_cli.md).

## 1. Activate The Environment

### Windows PowerShell Setup

```powershell
Set-Location C:\Users\kvman\HydrideSegmentation
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gui.txt
python -m pip install -r requirements-docs.txt
python -m pip install -e .
```

### Windows CMD Setup

```bat
cd C:\Users\kvman\HydrideSegmentation
python -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gui.txt
python -m pip install -r requirements-docs.txt
python -m pip install -e .
```

### Linux Or macOS Setup

```bash
cd /path/to/HydrideSegmentation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-core.txt
python -m pip install -r requirements-gui.txt
python -m pip install -r requirements-docs.txt
python -m pip install -e .
```

## 2. Verify The Environment

Run these checks before debugging imports:

### Windows PowerShell Verification

```powershell
Get-Location
python --version
python -m pip show hydride-segmentation
python -c "import src.microseg, scripts.microseg_cli; print('imports ok')"
```

### Linux Or macOS Verification

```bash
pwd
python --version
python -m pip show hydride-segmentation
python -c "import src.microseg, scripts.microseg_cli; print('imports ok')"
```

If `python -m pip show hydride-segmentation` returns nothing, rerun `python -m pip install -e .`.

## 3. Avoid `ModuleNotFoundError: src`

Use one of these supported run modes:

1. Preferred installed CLI:

```bash
microseg-cli models --details
```

2. Script form from repo root:

```bash
python scripts/microseg_cli.py models --details
```

3. Backward-compatible paired dataset prep wrapper:

```bash
python hydride_segmentation/prepare_dataset.py --input-dir tmp/tutorial_demo/raw_pairs --output tmp/tutorial_demo/prepared_dataset
```

Do not run commands from outside the repository root unless you already installed the package editable.

## 4. Exact Command Recipes

### Generate The Tiny Tutorial Dataset

```bash
python scripts/generate_tutorial_dataset.py
```

This writes paired image/mask files under `tmp/tutorial_demo/raw_pairs/`.

### Prepare A Dataset From Raw Paired Files

Primary beginner path:

```bash
python scripts/microseg_cli.py prepare_dataset --config configs/tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml
```

The YAML file controls:

- input folder
- split policy
- resize and crop policy
- RGB-mask handling
- train-only shadow and blur augmentation
- output folder

If your raw folder uses the same stem for image and mask, such as `sample_001.jpg` and `sample_001.png`, enable this YAML block:

```yaml
same_stem_pairing:
  enabled: true
  image_extensions:
    - .jpg
    - .jpeg
  mask_extensions:
    - .png
```

This fallback is opt-in so the default `_mask` naming workflow remains unambiguous.

### Prepare An Already Organized `source/masks` Or Split Dataset

```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

Use this when your data already exists in either:

- `dataset/source` + `dataset/masks`
- `dataset/train|val|test/images|masks`

### Train From The Prepared Tutorial Dataset

```bash
python scripts/microseg_cli.py train --config configs/tutorials/train.tiny_unet_from_prepared.yml
```

### Single-Image Inference

```bash
microseg-cli infer --config configs/inference.default.yml --image test_data/syntheticHydrides.png
```

### Recursive Folder Inference

```bash
microseg-cli infer \
  --config configs/inference.default.yml \
  --image-dir data/sample_images \
  --recursive \
  --glob-patterns "*.png,*.tif,*.tiff,*.jpg,*.jpeg"
```

### Build The Sphinx HTML Site

```bash
python scripts/build_docs.py --html-only
```

## 5. Windows Notes

- Prefer PowerShell when copying commands from the docs.
- Keep the repository path free of trailing quotes when using `Set-Location`.
- When a command contains JSON for `--set`, wrap the whole JSON payload in single quotes in PowerShell.
- If execution policy blocks activation, either use `activate.bat` from CMD or run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

## 6. Troubleshooting

### `No module named 'src'`

- confirm you are in the repository root
- rerun `python -m pip install -e .`
- retry with `python scripts/microseg_cli.py ...`

### `No module named 'sphinx'`

```bash
python -m pip install -r requirements-docs.txt
```

### `No module named 'PySide6'`

```bash
python -m pip install -r requirements-gui.txt
```

### Training runs but writes no report

Check:

- `output_dir/report.json`
- `output_dir/training_manifest.json`
- `output_dir/training_report.html`
- `output_dir/last_checkpoint.pt`

### Dataset prep created more training files than expected

That is normal when augmentation is enabled.
The paired prep reports now show both:

- `source_split_counts`: original split counts before augmentation
- `split_counts`: final materialized counts after augmentation
