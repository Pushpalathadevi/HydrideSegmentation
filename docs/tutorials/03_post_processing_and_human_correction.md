# 03 Post Processing And Human Correction

This page mirrors <a href="../03_post_processing_and_human_correction.ipynb">03_post_processing_and_human_correction.ipynb</a> so the tutorial is searchable in Sphinx while the raw notebook remains downloadable.

## Notebook Transcript

# 03 - Post-Processing, Human Correction, and Versioned Export

This notebook shows the correction path that sits between model prediction and export.
It uses the same correction session and exporter objects that the desktop GUI relies on.

## What You Will Learn

- how a prediction becomes an editable correction session
- how brush, delete, and undo/redo work in the session object
- how annotation overlays are composed
- how the correction export schema records provenance
- why correction is a separate stage instead of an in-place image edit
- which alternatives exist if you want manual annotation only or a different labeling tool

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
# Run one local inference so we have a real predicted mask to correct.
from src.microseg.app.desktop_workflow import DesktopWorkflowManager

run_root = study_root / '03_post_processing'
run_root.mkdir(parents=True, exist_ok=True)
workflow = DesktopWorkflowManager()
model_name = next((name for name in workflow.model_options() if 'conventional' in name.lower()), workflow.model_options()[0])
record = workflow.run_single(
    str(sample_path),
    model_name=model_name,
    params={'image_path': str(sample_path)},
    include_analysis=True,
)
print('Model:', model_name)
print('Run ID:', record.run_id)
print('Metrics:')
pprint(record.metrics)
```

```python
# Create a correction session and perform a few simple edits.
import numpy as np
from scipy import ndimage
from src.microseg.corrections import CorrectionSession

session = CorrectionSession(np.asarray(record.mask_image))

component_map, component_count = ndimage.label(session.current_mask > 0)
if component_count > 0:
    ids, counts = np.unique(component_map[component_map > 0], return_counts=True)
    target_id = int(ids[np.argmin(counts)])
    y0, x0 = np.argwhere(component_map == target_id)[0]
    deleted = session.delete_feature(int(x0), int(y0))
    print('Deleted one connected component:', deleted)

session.apply_brush(
    x=int(session.current_mask.shape[1] // 2),
    y=int(session.current_mask.shape[0] // 2),
    radius=12,
    mode='add',
    class_index=1,
)
print('Undo works:', session.undo())
print('Redo works:', session.redo())
print('Session report:')
pprint(session.report())
```

```python
# Compose the annotation view before and after the correction.
from src.microseg.ui import AnnotationLayerSettings, compose_annotation_view

base_rgb = np.asarray(record.input_image)
pred_mask = np.asarray(record.mask_image)
corr_mask = np.asarray(session.current_mask)

before = compose_annotation_view(base_rgb, pred_mask, pred_mask, AnnotationLayerSettings())
after = compose_annotation_view(base_rgb, pred_mask, corr_mask, AnnotationLayerSettings())

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(before)
axes[0].set_title('Before correction')
axes[0].axis('off')
axes[1].imshow(after)
axes[1].set_title('After correction')
axes[1].axis('off')
plt.tight_layout()
```

```python
# Export the corrected sample using the repository's versioned correction schema.
from src.microseg.corrections import CorrectionExporter

export_dir = CorrectionExporter().export_sample(
    record,
    session.current_mask,
    run_root / 'exports',
    annotator='student',
    notes='Notebook demo correction',
    formats={'indexed_png', 'color_png', 'numpy_npy'},
    feedback_record_id='notebook-demo',
    feedback_record_dir=str(run_root),
)
print('Exported to:', export_dir)
print('Files:')
for path in sorted(export_dir.iterdir()):
    print(' -', path.name)

record_json = export_dir / 'correction_record.json'
print('\nCorrection record excerpt:')
print(record_json.read_text(encoding='utf-8')[:1800])
```

## Key Point

The notebook is doing the same work as the GUI correction flow: session state, edit history, overlay composition, and export provenance.
That makes it safe to prototype correction ideas in a notebook before you change the interactive UI.

The main alternative is direct mask editing without a session object, but that loses the structured undo/redo and provenance trail that the repo uses for traceability.

## Raw Notebook

- Download the notebook file: <a href="../03_post_processing_and_human_correction.ipynb">03_post_processing_and_human_correction.ipynb</a>
