# Data Preparation Subsystem (Binary Segmentation)

## Mission

The data preparation subsystem converts paired source images and masks into reproducible training datasets for semantic segmentation with robust traceability.
It currently targets binary segmentation and is designed to extend to multiclass without breaking run manifests.

## Supported Export Layouts

- **Oxford-like**: `images/`, `annotations/trimaps/`, split files (`train.txt`, `val.txt`, `test.txt`, `trainval.txt`).
- **MaDo-like**: `train|val|test/{images,masks}/`.

## Key Features

- Config model with YAML loading fallback to in-code dictionary defaults.
- Default config file: `configs/data_prep.default.yml` (auto-loaded by `prep-dataset` when `--config` is omitted).
- Pairing with strict/permissive modes and configurable mask naming patterns.
- **Paired single-folder ingestion** (`{stem}.jpg` + `{stem}_mask.png` or `{stem}.png`).
- Binarization modes:
  - `nonzero`
  - `threshold` (`>=T` or `>T`)
  - `value_equals`
  - `otsu`
  - `percentile`
  - `rgb_mask_mode` (red-channel threshold with optional G/B suppression)
- Robust mask normalization extras:
  - grayscale binary-like masks (`0/1` or `0/255`) map with non-zero foreground
  - red-dominance fallback for imperfect RGB masks (low red / small non-zero G/B noise)
  - auto-Otsu for noisy near-binary grayscale masks (for JPEG/compression contamination patterns)
- Empty output-mask sanity policy (`empty_mask_action=warn|error`) to flag masks with no foreground after preprocessing.
- Optional morphology: open/close, small-component removal, hole filling.
- Raw mask quality checks for unexpected non-binary values (default expected values: `0`, `255`).
- Resizing policies:
  - `letterbox_pad`
  - `center_crop`
  - `stretch`
  - `keep_aspect_no_pad`
  - `short_side_to_target_crop` (scale shortest side to target, then crop)
- Split-aware crop modes (`crop_mode_train`, `crop_mode_eval`).
- Multi-format export (`.png`, `.tif/.tiff`) with Pillow TIFF fallback.
- Deterministic `manifest.json` with split counts, record-level stats, and warnings.
- Dataset QA artifacts: `dataset_qa_report.json` and `dataset_qa_report.html`.
- Progress + ETA logging plus `dataset_prepare.log` in output folder.
- Debug mode subset processing plus inspection artifacts:
  - input image + output image
  - input mask + processed mask
  - input-vs-output mask difference
  - overlay
  - per-sample criteria JSON (`mode`, threshold details, foreground ratio, warnings)
  - combined panel image.

## CLI Usage

```bash
prep-dataset \
  --input-dir path/to/paired_data \
  --output-root outputs/prepared_binary \
  --style mado \
  --target-size 512 \
  --crop-train random \
  --crop-eval center \
  --mask-r-min 200 \
  --mask-g-max 60 \
  --mask-b-max 60 \
  --allow-red-dominance-fallback \
  --auto-otsu-for-noisy-grayscale \
  --empty-mask-action warn \
  --seed 42
```

Dry run:

```bash
python scripts/microseg_cli.py prepare_dataset \
  --input-dir D:/data/hydride_pairs \
  --output-root D:/data/HydrideData7.0 \
  --style mado \
  --target-size 512 \
  --crop-train random \
  --crop-eval center \
  --mask-r-min 200 \
  --mask-g-max 60 \
  --mask-b-max 60 \
  --allow-red-dominance-fallback \
  --auto-otsu-for-noisy-grayscale \
  --empty-mask-action warn \
  --seed 42 \
  --train-frac 0.8 \
  --val-frac 0.1 \
  --dry-run
```

Debug run with detailed visuals/criteria:

```bash
python scripts/microseg_cli.py prepare_dataset \
  --input-dir D:/data/hydride_pairs \
  --output-root D:/data/HydrideData7.0 \
  --style mado \
  --target-size 512 \
  --debug \
  --debug-limit 100 \
  --num-debug 12 \
  --debug-draw-contours \
  --mask-r-min 200 --mask-g-max 60 --mask-b-max 60 \
  --allow-red-dominance-fallback \
  --auto-otsu-for-noisy-grayscale
```

Python module invocation:

```bash
python -m src.microseg.data_preparation.cli --input ... --output ... --debug
```

## Programmatic Usage

```python
from src.microseg.data_preparation.config import DatasetPrepConfig
from src.microseg.data_preparation.pipeline import DatasetPreparer

cfg = DatasetPrepConfig.from_dict({
    "input_dir": "data/paired",
    "output_dir": "outputs/prepared",
    "styles": ["mado"],
    "rgb_mask_mode": True,
    "mask_r_min": 200,
    "mask_g_max": 60,
    "mask_b_max": 60,
    "allow_red_dominance_fallback": True,
    "auto_otsu_for_noisy_grayscale": True,
    "empty_mask_action": "warn",
    "resize_policy": "short_side_to_target_crop",
    "target_size": 512,
    "crop_mode_train": "random",
    "crop_mode_eval": "center",
})
result = DatasetPreparer(cfg).run()
print(result.manifest_path)
```

## Manifest / QA Fields

`manifest.json` includes:

- UTC timestamp
- tool version and git commit (when available)
- resolved config
- split counts
- per-record source paths and export paths
- original/output shapes
- mask stats (raw/binary unique values, foreground count/ratio)
- mask criteria (`mode`, thresholds, auto-otsu usage, all-zero output marker)
- item-level and overall warnings summary

`dataset_qa_report.json` includes:

- pair/missing diagnostics
- split counts
- read failure list
- empty output mask summary (`count`, `stems`, configured action)
- aggregate foreground coverage stats
- elapsed stage timing

## Extending To Multiclass

To evolve toward multiclass preparation:

1. replace binary binarizer with indexed-class converter,
2. add class-map metadata to the manifest,
3. keep record schema stable and version increments explicit.
