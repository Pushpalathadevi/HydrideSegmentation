# 01 Data Preparation And Dataset Planning

This page mirrors <a href="../01_data_preparation_and_dataset_planning.ipynb">01_data_preparation_and_dataset_planning.ipynb</a> so the tutorial is searchable in Sphinx while the raw notebook remains downloadable.

## Notebook Transcript

# 01 - Data Preparation, Classical Preprocessing, and Dataset Planning

This notebook builds a small teaching dataset from the bundled hydride sample image, then uses the repository's real split planner and QA checks.
The example is intentionally small so students can see how the data contract works before they scale up to a real project.

## What You Will Learn

- how the repo expects source/mask folders to be laid out
- how classical preprocessing steps can create a first-pass baseline mask
- how leakage-aware split planning keeps augmented variants together
- how the preparation manifest and QA report are written
- why grouping augmented variants matters before training
- which alternative split strategies exist and when to prefer them

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
# Build a tiny hydride-style study dataset from the bundled sample image.
from PIL import ImageOps

raw = Image.open(sample_path).convert('RGB')
base_mask = make_pseudo_mask(raw)

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

demo_root = study_root / '01_data_preparation'
source_dir = demo_root / 'source'
mask_dir = demo_root / 'masks'
prepared_dir = demo_root / 'prepared'
for folder in [source_dir, mask_dir, prepared_dir]:
    folder.mkdir(parents=True, exist_ok=True)

saved = []
for family_name, variants in families.items():
    for suffix, image_fn, mask_fn in variants:
        image = image_fn(raw)
        mask_img = mask_fn(Image.fromarray(base_mask * 255)).convert('L').point(lambda v: 255 if v > 0 else 0)
        stem = f'{family_name}_{suffix}'
        image_path = source_dir / f'{stem}.png'
        mask_path = mask_dir / f'{stem}.png'
        image.save(image_path)
        mask_img.save(mask_path)
        saved.append((stem, np.asarray(image), np.asarray(mask_img)))

print('Saved files:')
for stem, *_ in saved:
    print(' -', stem)

show_grid(saved[:4], title='Demo source images and masks')
```

```python
# Preview the split plan before materializing anything.
from src.microseg.dataops import DatasetPrepareConfig, preview_training_dataset_layout

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
preview = preview_training_dataset_layout(cfg)
print(json.dumps({
    'source_layout': preview.source_layout,
    'used_existing_splits': preview.used_existing_splits,
    'split_counts': preview.split_counts,
    'total_pairs': preview.total_pairs,
    'leakage_groups': preview.leakage_groups,
    'class_histogram': preview.class_histogram,
}, indent=2))
print('\nFirst few mapping rows:')
for row in preview.mapping[:6]:
    pprint(row)
```

```python
# Materialize the split layout and run dataset QA.
from src.microseg.dataops import DatasetQualityConfig, prepare_training_dataset_layout, run_dataset_quality_checks

result = prepare_training_dataset_layout(cfg)
qa = run_dataset_quality_checks(
    DatasetQualityConfig(
        dataset_dir=str(prepared_dir),
        output_path=str(prepared_dir / 'dataset_qa_report.json'),
        imbalance_ratio_warn=0.98,
        strict=False,
    )
)

print(json.dumps({
    'prepared': result.prepared,
    'source_layout': result.source_layout,
    'split_counts': result.split_counts,
    'manifest_path': result.manifest_path,
}, indent=2))
print('\nQA status:', qa.ok)
print('QA warnings:')
for item in qa.warnings:
    print(' -', item)
print('\nPrepared files:')
for path in sorted((prepared_dir / 'train' / 'images').glob('*'))[:4]:
    print(' -', path.name)
```

## Interpretation Notes

- The split planner keeps suffix-aware augmentation families together so near-duplicates do not leak across train, val, and test.
- The QA report is the first place to look when a dataset is missing a pair or has a dimension mismatch.
- The classical preprocessing step here is deliberately simple because it is a teaching baseline, not a claim that Otsu-based segmentation is always sufficient.
- Alternatives include hand-labeled masks, more aggressive morphology, or a model-generated pseudo-labeling pass when you already have a trained baseline.
- If the pseudo-mask is unstable under tiny contrast changes, that is a signal to improve the data, not to jump straight to a larger model.

## Raw Notebook

- Download the notebook file: <a href="../01_data_preparation_and_dataset_planning.ipynb">01_data_preparation_and_dataset_planning.ipynb</a>
