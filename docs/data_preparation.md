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
- Binarization modes:
  - `nonzero`
  - `threshold` (`>=T` or `>T`)
  - `value_equals`
  - `otsu`
  - `percentile`
- Optional morphology: open/close, small-component removal, hole filling.
- Raw mask quality checks for unexpected non-binary values (default expected values: `0`, `255`).
- Resizing policies:
  - `letterbox_pad` (default)
  - `center_crop`
  - `stretch`
  - `keep_aspect_no_pad`
- Multi-format export (`.png`, `.tif/.tiff`) with Pillow TIFF fallback.
- Deterministic `manifest.json` with split counts, record-level stats, and warnings.
- Debug mode subset processing plus inspection artifacts (image/raw mask/binary/difference/overlay/panel).
- Run-time warnings and manifest warnings for any raw-mask pixels outside expected binary values.

## CLI Usage

```bash
prep-dataset \
  --input path/to/paired_data \
  --output outputs/prepared_binary \
  --style oxford,mado \
  --config configs/data_prep.default.yml \
  --seed 42
```

Python module invocation:

```bash
python -m src.microseg.data_preparation.cli --input ... --output ... --debug
```

Dry run (manifest-only planning):

```bash
prep-dataset --input ... --output ... --dry-run --style oxford,mado
```

## Programmatic Usage

```python
from src.microseg.data_preparation.config import DatasetPrepConfig
from src.microseg.data_preparation.pipeline import DatasetPreparer

cfg = DatasetPrepConfig.from_dict({
    "input_dir": "data/paired",
    "output_dir": "outputs/prepared",
    "styles": ["oxford", "mado"],
    "binarization_mode": "threshold",
    "threshold": 128,
})
result = DatasetPreparer(cfg).run()
print(result.manifest_path)
```

## Manifest Fields

`manifest.json` includes:

- UTC timestamp
- tool version and git commit (when available)
- resolved config
- split counts
- per-record source paths and export paths
- original/output shapes
- mask stats (raw/binary unique values, foreground count/ratio)
- non-binary raw-value diagnostics (unexpected values, affected pixel count/ratio, warning text)
- item-level and overall warnings summary

## Extending To Multiclass

To evolve toward multiclass preparation:

1. replace binary binarizer with indexed-class converter,
2. add class-map metadata to the manifest,
3. keep record schema stable and version increments explicit.
