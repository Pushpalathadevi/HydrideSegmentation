# 04 Evaluation Testing And Report Reading

This page mirrors <a href="../04_evaluation_testing_and_report_reading.ipynb">04_evaluation_testing_and_report_reading.ipynb</a> so the tutorial is searchable in Sphinx while the raw notebook remains downloadable.

## Notebook Transcript

# 04 - Evaluation, Testing, and Report Reading

This notebook ties together dataset QA, hydride statistics, visualization, and report inspection.
It is the final study step because it shows how to read the artifacts produced by the earlier notebooks.

## What You Will Learn

- how to run dataset QA as a test gate
- how to compute hydride statistics from predicted and corrected masks
- how to render the scientific plots that the desktop app uses
- how to read manifests and reports as reproducibility evidence
- why evaluation is separated from training
- what other validation styles you could use if you were running a different experiment design

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
# Locate the artifacts produced by the earlier notebooks and create any missing demo data.
from src.microseg.dataops import DatasetQualityConfig, run_dataset_quality_checks

prep_dir = study_root / '01_data_preparation' / 'prepared'
train_dir = study_root / '02_ml_training' / 'pixel_classifier'
correction_root = study_root / '03_post_processing' / 'exports'

if not prep_dir.exists():
    raise FileNotFoundError('Run notebook 01 first so the prepared dataset exists.')
if not train_dir.exists():
    raise FileNotFoundError('Run notebook 02 first so the training artifacts exist.')
if not correction_root.exists():
    raise FileNotFoundError('Run notebook 03 first so the correction export exists.')

qa_report = run_dataset_quality_checks(
    DatasetQualityConfig(
        dataset_dir=str(prep_dir),
        output_path=str(study_root / '04_evaluation' / 'dataset_qa_report.json'),
        imbalance_ratio_warn=0.98,
        strict=False,
    )
)
print('QA ok:', qa_report.ok)
print('Split counts:', qa_report.split_counts)
print('Warnings:')
for item in qa_report.warnings:
    print(' -', item)
```

```python
# Load the latest correction export and compare prediction vs correction statistics.
from src.microseg.evaluation import HydrideVisualizationConfig, compute_hydride_statistics, render_hydride_visualizations

sample_exports = sorted(correction_root.glob('*'))
if not sample_exports:
    raise FileNotFoundError('No correction exports found.')
export_dir = sample_exports[-1]
pred_mask = np.asarray(Image.open(export_dir / 'predicted_mask_indexed.png').convert('L'))
corr_mask = np.asarray(Image.open(export_dir / 'corrected_mask_indexed.png').convert('L'))

pred_stats = compute_hydride_statistics(pred_mask, orientation_bins=18, size_bins=20, min_feature_pixels=1)
corr_stats = compute_hydride_statistics(corr_mask, orientation_bins=18, size_bins=20, min_feature_pixels=1)

keys = [
    'hydride_count',
    'hydride_area_fraction_percent',
    'hydride_total_area_pixels',
    'orientation_alignment_index',
    'orientation_entropy_bits',
]
print('Metric comparison:')
for key in keys:
    print(f" - {key}: pred={pred_stats.scalar_metrics.get(key)} corr={corr_stats.scalar_metrics.get(key)}")
```

```python
# Render the hydride analysis images for a visual readout.
vis_cfg = HydrideVisualizationConfig(orientation_bins=18, size_bins=20, min_feature_pixels=1, orientation_cmap='viridis', size_scale='linear')
pred_vis = render_hydride_visualizations(pred_stats, vis_cfg)
corr_vis = render_hydride_visualizations(corr_stats, vis_cfg)

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
axes[0, 0].imshow(pred_vis['orientation_map_rgb'])
axes[0, 0].set_title('Prediction orientation map')
axes[0, 1].imshow(pred_vis['size_distribution_rgb'])
axes[0, 1].set_title('Prediction size distribution')
axes[0, 2].imshow(pred_vis['orientation_distribution_rgb'])
axes[0, 2].set_title('Prediction orientation distribution')
axes[1, 0].imshow(corr_vis['orientation_map_rgb'])
axes[1, 0].set_title('Correction orientation map')
axes[1, 1].imshow(corr_vis['size_distribution_rgb'])
axes[1, 1].set_title('Correction size distribution')
axes[1, 2].imshow(corr_vis['orientation_distribution_rgb'])
axes[1, 2].set_title('Correction orientation distribution')
for ax in axes.flat:
    ax.axis('off')
plt.tight_layout()
```

```python
# Inspect the manifest and correction record as reproducibility evidence.
training_manifest = json.loads((train_dir / 'training_manifest.json').read_text(encoding='utf-8'))
correction_record = json.loads((export_dir / 'correction_record.json').read_text(encoding='utf-8'))

print('Training schema:', training_manifest['schema_version'])
print('Training backend/runtime samples:', training_manifest.get('train_samples'))
print('Correction schema:', correction_record['schema_version'])
print('Annotator:', correction_record.get('annotator'))
print('Export formats:', correction_record.get('export_formats'))
```

## How To Use This In Practice

- Treat QA failures as a data problem first, not a model problem.
- Compare metrics and images together; never trust the table alone.
- Keep manifests and correction records because they are the traceability chain for the work.
- If you need a different validation style, use this notebook as the pattern and swap in your own dataset or metrics.

## Raw Notebook

- Download the notebook file: <a href="../04_evaluation_testing_and_report_reading.ipynb">04_evaluation_testing_and_report_reading.ipynb</a>
