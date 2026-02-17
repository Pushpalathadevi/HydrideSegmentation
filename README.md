# HydrideSegmentation -> MicroSeg Platform (Transition)

Current release version: `0.15.0`

This repository is transitioning from a hydride-specific toolkit into a general local platform for microstructural segmentation.
Hydride segmentation is the first validated workflow.

## Mission

Build a scientifically robust, CPU-first desktop + CLI platform for microstructural segmentation with:
- pluggable model backends
- quantitative analysis pipelines
- human-in-the-loop correction
- correction export for retraining loops
- reproducible experiment and deployment artifacts

See `docs/mission_statement.md`.

## Core Capabilities

- Registry-backed segmentation orchestration (`src/microseg`)
- Qt desktop GUI (`hydride-gui`) with:
  - brush/polygon/lasso tools
  - connected-feature delete/relabel
  - class index + color map editing
  - split-view synchronized zoom/pan and layer transparency
  - project/session save-load
  - Dataset Prep + QA workspace (preview, prepare, QA, training gate)
- Correction export schema `microseg.correction.v1`
- Deterministic correction dataset packaging
- Unified CLI (`microseg-cli`) for infer/train/evaluate/package/models
- GPU-compatible training/inference/evaluation with CPU default + safe fallback
- UNet training backend with checkpoint/resume + fixed/random validation sample tracking
- JSON + HTML run reports for training and evaluation
- Frozen checkpoint metadata registry for model selection guidance

## Standards Baseline

The documentation and implementation policies are aligned with and adapted from the standards style used in
[DeepImageDeconvolution](https://github.com/kvmani/DeepImageDeconvolution), then extended for segmentation-specific
annotation, correction, and deployment needs.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

Qt GUI dependency:
```bash
pip install PySide6
```

## Primary Usage

Qt GUI:
```bash
hydride-gui
```

Legacy Tk GUI fallback:
```bash
hydride-gui --framework tk
```

Model listing with dynamic metadata:
```bash
microseg-cli models --details
```

Inference:
```bash
microseg-cli infer --config configs/inference.default.yml --set params.area_threshold=120
```

Training (UNet):
```bash
microseg-cli train --config configs/train.default.yml --set epochs=20
```

Training with tracked validation samples:
```bash
microseg-cli train \
  --config configs/train.default.yml \
  --set val_tracking_samples=8 \
  --set "val_tracking_fixed_samples=val_000.png|val_123.png"
```

Evaluation:
```bash
microseg-cli evaluate --config configs/evaluate.default.yml --set split=test
```

Dataset packaging:
```bash
microseg-cli package --config configs/package.default.yml --set train_ratio=0.75
```

Leakage-aware split planning (v2):
```bash
microseg-cli dataset-split --config configs/dataset_split.default.yml
```

Unsplit `source/masks` auto-prepare (leakage-aware default + global IDs):
```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```

RGB mask colormap conversion during auto-prepare:
```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set mask_input_type=rgb_colormap \
  --set 'mask_colormap={"0":[0,0,0],"1":[255,0,0]}'
```

Dataset QA:
```bash
microseg-cli dataset-qa --config configs/dataset_qa.default.yml --strict
```

Registry validation:
```bash
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```

Phase closeout gate:
```bash
microseg-cli phase-gate --phase-label "Phase N" --strict
```

## Frozen Checkpoints

- Metadata registry: `frozen_checkpoints/model_registry.json`
- Guidance: `docs/frozen_checkpoint_registry.md`
- Binary weights are intentionally excluded from git tracking.

## Documentation

- Docs index: `docs/README.md`
- Mission: `docs/mission_statement.md`
- Phase roadmap: `docs/development_roadmap.md`
- Foundation strategy: `docs/foundation_strategy.md`
- Current gap analysis: `docs/current_state_gap_analysis.md`
- Training data requirements: `docs/training_data_requirements.md`
- GUI user guide: `docs/gui_user_guide.md`
- Configuration workflow: `docs/configuration_workflow.md`
- Development workflow + phase closeout gate: `docs/development_workflow.md`
- Developer guide: `developer_guide.md`
- Repository contract: `AGENTS.md`

## Contributing

- Contributor guide: `CONTRIBUTE.md`
- Working contract: `AGENTS.md`

## License

MIT (see `LICENSE`).
