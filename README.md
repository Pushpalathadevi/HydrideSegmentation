# HydrideSegmentation

HydrideSegmentation is a collection of object‑oriented Python tools for segmenting
hydrides in zirconium alloys.  The modules form a modular pipeline that can be
used directly or integrated with GUI applications.  Both real and synthetic
images are supported.

## Key Features

- **HydrideSegmentation** (`segmentationMaskCreation.py`) – conventional image
  processing pipeline that produces an OpenRaster (ORA) file and optional PNG
  mask【F:segmentationMaskCreation.py†L13-L18】.
- **SegmentationEvaluator** (`segmentationEvlauater.py`) – generate synthetic
  data or load real results and compute IoU, Dice, precision, recall and other
  metrics【F:segmentationEvlauater.py†L28-L38】.
- **Orientation analysis** (`hydrideOrientationAnalyzer.py`) – compute mean
  plate orientation for each hydride region with optional debug plots.
- **Dataset utilities** – dataset preparation and augmentation scripts to help
  create training data for deep learning (`prepare_segmentation_dataset_for_training.py`,
  `Augment_Hydride_Dataset.py`).
- **GUI integration** – `GUI.py` provides a desktop interface supporting both the
  conventional and ML models.
- **Debug mode** – most scripts accept a `debug` flag to save or plot annotated
  intermediate results.

## Folder Structure

```
HydrideSegmentation/
├── Augment_Hydride_Dataset.py       # dataset augmentation
├── Export_hydrideMaskXCF_to_PNG.py  # convert XCF layers to PNG/JPG
├── GUI.py                           # desktop interface
├── applyMaskToMatrix.py             # merge hydride patches with matrix images
├── collectAllHydrideImagesAtOnePlace.py
├── prepare_segmentation_dataset_for_training.py
├── segmentationEvlauater.py         # evaluation utilities
├── segmentationMaskCreation.py      # conventional segmentation
├── hydrideOrientationAnalyzer.py    # orientation measurement
├── test_data/                       # sample and synthetic images
└── README.md
```

## Usage

1. **Environment** – install the required packages.  A minimal setup is:

```bash
pip install -r requirements.txt  # or install cv2, numpy, matplotlib, scikit-learn, albumentations, etc.
```

2. **Segment an image** using the conventional pipeline:

```bash
python segmentationMaskCreation.py --image_path path/to/input.png --debug
```

3. **Run the GUI** for interactive segmentation:

```bash
python GUI.py
```

4. **Prepare datasets** for deep learning:

```bash
python prepare_segmentation_dataset_for_training.py
```

5. **Analyse hydride orientations** on the provided synthetic image:

```bash
python hydrideOrientationAnalyzer.py --debug
```

## Testing

Example images are provided in `test_data/`.  To validate the segmentation
pipeline, run the evaluator in simulation mode which generates synthetic
shapes, predicts a tampered mask and prints evaluation metrics:

```bash
python segmentationEvlauater.py --simulate
```

The `SegmentationEvaluator` class can also be configured to load real
`input_image`, `ground_truth` and `predicted_mask` from this folder.

## Debug Mode

Many modules accept a `debug` or `plot` flag.  When enabled, additional figures
and annotated images are produced, as seen in `segmentationMaskCreation.py`
which logs intermediate steps such as CLAHE, adaptive thresholding and filtering
of small regions【F:segmentationMaskCreation.py†L59-L97】.

## Contribution

Please see [CONTRIBUTE.md](CONTRIBUTE.md) for coding standards, documentation
requirements and testing guidelines.  Updates to documentation (including this
README) are expected with any major code change.

## Credits

Original code authors, contributors and reviewers are gratefully acknowledged.
