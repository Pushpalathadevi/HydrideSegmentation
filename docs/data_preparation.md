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
- Shared augmentation subsystem used by both `microseg-cli dataset-prepare` and the paired `prepare_dataset` / `prep-dataset` path.
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
- Optional split caps for evaluation sets (`max_val_examples`, `max_test_examples`) with remainder routed to training.
- Multi-format export (`.png`, `.tif/.tiff`) with Pillow TIFF fallback.
- Deterministic `manifest.json` with split counts, record-level stats, and warnings.
- Dataset QA artifacts: `dataset_qa_report.json` and `dataset_qa_report.html`.
- Progress + ETA logging plus `dataset_prepare.log` in output folder.
- Optional split-targeted augmentation with deterministic seeds and machine-readable provenance.
- Debug mode subset processing plus inspection artifacts:
  - input image + output image
  - input mask + processed mask
  - input-vs-output mask difference
  - overlay
  - per-sample criteria JSON (`mode`, threshold details, foreground ratio, warnings)
  - combined panel image.
- Augmentation debug mode writes `debug_augmentation/` before/after panels, overlays, delta views, and per-variant metadata JSON.

## Canonical Leakage Policy

The canonical split-layout preparation path is `microseg-cli dataset-prepare`.
When augmentation is enabled there, the workflow is:

1. pair source images and masks,
2. create deterministic train/val/test assignments,
3. materialize the original samples,
4. generate augmented variants only inside the configured split targets.

The default policy is train-only augmentation. This avoids leakage of near-duplicate augmented views across train/val/test.

## Augmentation Schema

Both dataset-preparation subsystems accept the same YAML block:

```yaml
augmentation:
  enabled: true
  seed: 42
  stage: post_resize
  apply_splits: [train]
  variants_per_sample: 2
  operations:
    - name: shadow
      probability: 0.9
      parameters:
        radius: 150
        sigma: 500
        intensity_range: [40, 50]
        count_range: [1, 3]
    - name: blur
      probability: 0.8
      parameters:
        sigma: 120
        kernel_size_range: [3, 9]
        min_center_distance_ratio: 0.4
        count_range: [1, 3]
  debug:
    enabled: true
    max_samples: 6
```

Supported fields:

- `enabled`: master on/off switch.
- `seed`: deterministic augmentation seed.
- `stage`: `pre_resize` or `post_resize`.
  - In the paired `prep-dataset` path, this controls whether image-only augmentation runs before or after resize/crop.
  - In the split-layout `dataset-prepare` path, there is no resize/crop stage, so the resolved stage is recorded as source-native in the manifest.
- `apply_splits`: split names to augment.
- `variants_per_sample`: number of augmented variants attempted per source sample.
- `operations`: ordered augmentation list.
- `operations[].probability`: Bernoulli application probability per variant.
- `operations[].parameters`: per-operation parameter mapping.
- `debug.enabled` and `debug.max_samples`: before/after inspection exports.

Current built-in operations:

- `shadow`: localized subtractive shadow fields.
- `blur`: localized peripheral Gaussian blur.

These two are image-only augmentations, so masks remain unchanged. The augmentation registry is structured to admit future paired geometry transforms without rewriting the orchestration layer.

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
  --seed 42 \
  --max-val-examples 200 \
  --max-test-examples 200
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
  --max-val-examples 200 \
  --max-test-examples 200 \
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

Canonical split-layout auto-prepare with augmentation:

```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.augmentation.shadow_blur.yml
```

Ad hoc CLI override example:

```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set 'augmentation={"enabled":true,"seed":42,"apply_splits":["train"],"variants_per_sample":1,"operations":[{"name":"shadow","probability":1.0,"parameters":{"count_range":[1,2],"intensity_range":[35,45]}},{"name":"blur","probability":0.7,"parameters":{"kernel_size_range":[3,7],"count_range":[1,2],"min_center_distance_ratio":0.4}}],"debug":{"enabled":true,"max_samples":4}}'
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
- augmentation metadata for generated variants when enabled

`dataset_qa_report.json` includes:

- pair/missing diagnostics
- split counts
- read failure list
- empty output mask summary (`count`, `stems`, configured action)
- aggregate foreground coverage stats
- elapsed stage timing
- augmentation summary (`enabled`, split targets, variants per sample, generated count, debug count)

## Debug Behavior

- Classical debug artifacts remain under `debug_inspection/`.
- Augmentation-specific before/after review artifacts are written under `debug_augmentation/`.
- Each augmentation debug sample includes:
  - source image
  - augmented image
  - mask
  - before/after overlays
  - difference image
  - comparison panel
  - metadata JSON with requested stage, resolved stage, seed, and applied operation parameters

## Failure Modes And Tuning Notes

- If `apply_splits` includes validation or test, you are explicitly trading away the safest leakage-default policy.
- Large `variants_per_sample` values can inflate dataset size quickly; check split counts in `dataset_prepare_manifest.json` or `manifest.json`.
- Strong shadows can erase faint microstructural evidence; lower `intensity_range` first before reducing `count_range`.
- Large blur kernels can remove edge fidelity needed for thin-feature segmentation; keep `kernel_size_range` small for fine structures.
- `pre_resize` is useful when you want resize/crop to act on already degraded images; `post_resize` is safer when you want augmentation strength controlled in model-input space.

## Extension Guide

Add new augmentations in `src/microseg/data_preparation/augmentation.py` by:

1. implementing the augmentation contract (`apply(...) -> image, mask, metadata`),
2. declaring whether it is `image_only` or `paired_geometry`,
3. registering it in `DEFAULT_AUGMENTATION_REGISTRY`,
4. adding config/tests/docs for the new operation.

## Extending To Multiclass

To evolve toward multiclass preparation:

1. replace binary binarizer with indexed-class converter,
2. add class-map metadata to the manifest,
3. keep record schema stable and version increments explicit.
