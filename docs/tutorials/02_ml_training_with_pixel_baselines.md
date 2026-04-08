# 02 Ml Training With Pixel Baselines

This page mirrors <a href="../02_ml_training_with_pixel_baselines.ipynb">02_ml_training_with_pixel_baselines.ipynb</a> so the tutorial is searchable in Sphinx while the raw notebook remains downloadable.

## Notebook Transcript

# 02 - Actual ML Training and Inference Loops

This notebook trains a lightweight CPU-first pixel classifier on the prepared hydride study dataset and then runs inference on the sample image.
It also shows how to repeat inference over deliberate input variants so students can see how preprocessing choices affect the output.

## Why This Notebook Exists

The repo is not just about running a model.
It is about teaching the workflow shape: preprocess the data, train a baseline, inspect the manifest, and then compare the result with the input image.

The inference-loop section matters because real workflows rarely stop at one image.
You usually want to run the same trained model across several inputs, compare the outputs, and inspect which preprocessing choices were helpful or harmful.

```python
from __future__ import annotations

import json
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from skimage import filters, morphology


def find_repo_root(start: Path | None = None) -> Path:
    start = Path.cwd() if start is None else Path(start)
    for candidate in [start, *start.parents]:
        if (candidate / 'docs').exists() and (candidate / 'src').exists():
            return candidate
    raise FileNotFoundError('Could not find the repository root from the current working directory.')


repo_root = find_repo_root()
sample_path = repo_root / 'data' / 'sample_images' / 'hydride_optical_sample.png'
if not sample_path.exists():
    sample_path = repo_root / 'test_data' / '3PB_SRT_data_generation_1817_OD_side1_8.png'
assert sample_path.exists(), sample_path

study_root = repo_root / 'outputs' / 'notebook_tutorials'
study_root.mkdir(parents=True, exist_ok=True)

plt.rcParams['figure.dpi'] = 120
plt.rcParams['image.cmap'] = 'gray'
```

```python
def make_pseudo_mask(image: Image.Image) -> np.ndarray:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.uint8)
    threshold = filters.threshold_otsu(gray)
    mask = gray < threshold
    mask = morphology.remove_small_objects(mask, min_size=64)
    mask = morphology.binary_opening(mask, morphology.disk(1))
    mask = morphology.binary_closing(mask, morphology.disk(2))
    mask = morphology.remove_small_holes(mask, area_threshold=256)
    return mask.astype(np.uint8)


def show_side_by_side(image: np.ndarray, mask: np.ndarray, *, title: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].imshow(image)
    axes[0].set_title(f'{title} - image')
    axes[0].axis('off')
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title(f'{title} - mask')
    axes[1].axis('off')
    plt.tight_layout()


def show_grid(pairs: list[tuple[str, np.ndarray, np.ndarray]], *, title: str) -> None:
    rows = len(pairs)
    fig, axes = plt.subplots(rows, 2, figsize=(11, 3.2 * rows))
    if rows == 1:
        axes = np.array([axes])
    fig.suptitle(title)
    for row, (label, image, mask) in enumerate(pairs):
        axes[row, 0].imshow(image)
        axes[row, 0].set_title(f'{label} - image')
        axes[row, 0].axis('off')
        axes[row, 1].imshow(mask, cmap='gray')
        axes[row, 1].set_title(f'{label} - mask')
        axes[row, 1].axis('off')
    plt.tight_layout()
```

```python
# Reuse the prepared study dataset from notebook 01. Create it here if needed so the notebook is self-contained.
from src.microseg.dataops import DatasetPrepareConfig, prepare_training_dataset_layout

study_dir = study_root / '01_data_preparation'
prepared_dir = study_dir / 'prepared'
if not prepared_dir.exists() or not any((prepared_dir / 'train' / 'images').glob('*')):
    raw = Image.open(sample_path).convert('RGB')
    base_mask = make_pseudo_mask(raw)
    demo_root = study_dir
    source_dir = demo_root / 'source'
    mask_dir = demo_root / 'masks'
    for folder in [source_dir, mask_dir, prepared_dir]:
        folder.mkdir(parents=True, exist_ok=True)
    families = {
        'hydride_plate_family_a': [
            ('aug00', lambda img: img, lambda msk: msk),
            ('aug01', lambda img: ImageOps.mirror(img), lambda msk: ImageOps.mirror(msk)),
        ],
        'hydride_plate_family_b': [
            ('aug00', lambda img: img.rotate(90, expand=True), lambda msk: msk.rotate(90, expand=True)),
            ('aug01', lambda img: img.rotate(270, expand=True), lambda msk: msk.rotate(270, expand=True)),
        ],
        'hydride_plate_family_c': [
            ('aug00', lambda img: ImageEnhance.Contrast(img).enhance(1.25), lambda msk: msk),
            ('aug01', lambda img: ImageEnhance.Sharpness(img).enhance(1.5), lambda msk: msk),
        ],
    }
    for family_name, variants in families.items():
        for suffix, image_fn, mask_fn in variants:
            image = image_fn(raw)
            mask_img = mask_fn(Image.fromarray(base_mask * 255)).convert('L').point(lambda v: 255 if v > 0 else 0)
            stem = f'{family_name}_{suffix}'
            image.save(source_dir / f'{stem}.png')
            mask_img.save(mask_dir / f'{stem}.png')
    cfg = DatasetPrepareConfig(
        dataset_dir=str(demo_root),
        output_dir=str(prepared_dir),
        train_ratio=0.67,
        val_ratio=0.17,
        test_ratio=0.16,
        seed=13,
        id_width=4,
        split_strategy='leakage_aware',
        leakage_group_mode='suffix_aware',
        binary_mask_normalization='off',
    )
    prepare_training_dataset_layout(cfg)

print('Prepared study dataset:', prepared_dir)
print('Train images:')
for path in sorted((prepared_dir / 'train' / 'images').glob('*')):
    print(' -', path.name)
```

```python
# Train the CPU-first baseline pixel classifier.
from src.microseg.training.pixel_classifier import PixelClassifierTrainer, PixelTrainingConfig, infer_image_with_pixel_classifier

run_root = study_root / '02_ml_training'
pixel_run_dir = run_root / 'pixel_classifier'
pixel_run_dir.mkdir(parents=True, exist_ok=True)

trainer = PixelClassifierTrainer()
train_cfg = PixelTrainingConfig(
    dataset_dir=str(prepared_dir),
    output_dir=str(pixel_run_dir),
    train_split='train',
    max_samples=20000,
    max_iter=250,
    seed=13,
    binary_mask_normalization='off',
)
train_summary = trainer.train(train_cfg)
print(json.dumps(train_summary, indent=2))

prediction_dir = run_root / 'prediction_demo'
pred_summary = infer_image_with_pixel_classifier(sample_path, train_summary['model_path'], prediction_dir)
print('\nInference outputs:')
print(json.dumps(pred_summary, indent=2))
```

```python
# Display the inference result and the training manifest.
from PIL import Image

input_image = np.asarray(Image.open(pred_summary['input']).convert('RGB'))
pred_mask = np.asarray(Image.open(pred_summary['prediction_indexed']).convert('L'))

show_side_by_side(input_image, pred_mask, title='Pixel classifier inference')

manifest_path = Path(train_summary['manifest_path'])
print('\nTraining manifest excerpt:')
print(manifest_path.read_text(encoding='utf-8')[:1800])
```

## Batch Inference Loop

This example reuses the trained pixel classifier and evaluates it on a small set of input variants.
That gives you a compact pattern you can later expand into a folder sweep, a preprocessing ablation, or a parameter search.

```python
# Run an inference loop over simple input variants to study robustness.
from src.microseg.training.pixel_classifier import load_pixel_classifier, predict_index_mask

model = load_pixel_classifier(train_summary['model_path'])
base = np.asarray(Image.open(pred_summary['input']).convert('RGB'), dtype=np.uint8)

variants = [
    ('original', base),
    ('contrast 1.10x', np.asarray(ImageEnhance.Contrast(Image.fromarray(base)).enhance(1.10), dtype=np.uint8)),
    ('sharpness 1.40x', np.asarray(ImageEnhance.Sharpness(Image.fromarray(base)).enhance(1.40), dtype=np.uint8)),
    ('mirror', np.asarray(ImageOps.mirror(Image.fromarray(base)).convert('RGB'), dtype=np.uint8)),
]

loop_rows = []
fig, axes = plt.subplots(len(variants), 2, figsize=(11, 3.0 * len(variants)))
if len(variants) == 1:
    axes = np.array([axes])

for row, (label, image) in enumerate(variants):
    pred = predict_index_mask(image, model)
    loop_rows.append({
        'variant': label,
        'foreground_pixels': int((pred > 0).sum()),
        'foreground_fraction': float((pred > 0).mean()),
        'shape': list(pred.shape),
    })
    axes[row, 0].imshow(image)
    axes[row, 0].set_title(f'{label} - input')
    axes[row, 0].axis('off')
    axes[row, 1].imshow(pred, cmap='gray')
    axes[row, 1].set_title(f'{label} - prediction')
    axes[row, 1].axis('off')

plt.tight_layout()
print(json.dumps(loop_rows, indent=2))
```

```python
# Optional: train the torch baseline if torch is installed.
try:
    from src.microseg.training.torch_pixel_classifier import TorchPixelClassifierTrainer, TorchPixelTrainingConfig
except Exception as exc:
    print('Torch baseline skipped:', exc)
else:
    torch_run_dir = run_root / 'torch_pixel_classifier'
    torch_run_dir.mkdir(parents=True, exist_ok=True)
    torch_summary = TorchPixelClassifierTrainer().train(
        TorchPixelTrainingConfig(
            dataset_dir=str(prepared_dir),
            output_dir=str(torch_run_dir),
            train_split='train',
            max_samples=12000,
            epochs=3,
            batch_size=2048,
            learning_rate=1e-2,
            seed=13,
            enable_gpu=False,
            device_policy='cpu',
            binary_mask_normalization='off',
        )
    )
    print(json.dumps(torch_summary, indent=2))
```

## What To Notice

- The baseline model is intentionally small and CPU-friendly.
- The output manifest captures the exact config, output paths, and training runtime.
- The batch inference loop shows how a trained model behaves across deliberate input variants.
- Once the workflow is clear, you can swap in the deep-learning backend without changing the dataset contract.
- Alternative next steps include a folder-level inference sweep, a different preprocessing chain, or a larger model family.

## Raw Notebook

- Download the notebook file: <a href="../02_ml_training_with_pixel_baselines.ipynb">02_ml_training_with_pixel_baselines.ipynb</a>
