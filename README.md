# HydrideSegmentation -> Microstructural Segmentation Platform (Transition)

Current release version: `0.6.0`

This repository is transitioning from a hydride-specific toolkit into a general-purpose microstructural segmentation platform.
Hydride segmentation remains the baseline implemented workflow.

## Mission

Build a scientifically robust local application and backend stack for microstructural segmentation with:
- pluggable segmentation models
- quantitative analysis pipelines
- human-in-the-loop correction
- correction export for future model retraining

See `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/mission_statement.md`.

## Current Capabilities

- Registry-backed segmentation orchestration (`src/microseg`)
- Qt desktop GUI as default (`hydride-gui`)
- Batch inference + run history
- Advanced correction tools:
  - brush, polygon, lasso
  - connected-feature delete/relabel
  - class index and color map editing
  - split-view synchronized zoom/pan and layer transparency
- Correction export schema `microseg.correction.v1`
- Export mask formats: indexed PNG, color PNG, NumPy
- Session persistence schema `microseg.project.v1`
- Deterministic correction dataset packaging
- YAML config + `--set` override support in unified CLI

## Transition Roadmap and Architecture

- Docs index: `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/README.md`
- Target architecture: `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/target_architecture.md`
- Phase plan: `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/development_roadmap.md`
- Foundation strategy: `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/foundation_strategy.md`
- Gap analysis: `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/docs/current_state_gap_analysis.md`

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Primary Usage

Qt GUI (default):
```bash
hydride-gui
```

Legacy Tk GUI fallback:
```bash
hydride-gui --framework tk
```

Unified CLI inference (YAML + overrides):
```bash
microseg-cli infer --config configs/inference.default.yml --set params.area_threshold=120
```

Unified CLI dataset packaging:
```bash
microseg-cli package --config configs/package.default.yml --set train_ratio=0.75
```

Legacy packaging script remains supported:
```bash
python scripts/package_corrections_dataset.py --input-dir <correction_exports> --output-dir <dataset_out>
```

## Model Weights

ML inference expects model weights from:
- `HYDRIDE_MODEL_PATH` environment variable, or
- `/opt/models/hydride_segmentation/model.pt` by default.

## GUI Dependency Note

Qt GUI requires `PySide6`.
Install with:
```bash
pip install PySide6
```

## Contributing

- Contributor guide: `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/CONTRIBUTE.md`
- Repository working contract: `/Users/anantatamukalaamrutha/python_projects/HydrideSegmentation/AGENTS.md`

## License

MIT (see `LICENSE`).
