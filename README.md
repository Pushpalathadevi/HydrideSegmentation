# HydrideSegmentation -> MicroSeg Platform (Transition)

Current release version: `0.22.0`

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
  - Run Review workspace for report summary + metric-delta comparison
  - HPC GA Planner for scheduler-ready multi-candidate bundle generation and feedback analysis
- Correction export schema `microseg.correction.v1`
- Deterministic correction dataset packaging
- Unified CLI (`microseg-cli`) for infer/train/evaluate/package/models
- GPU-compatible training/inference/evaluation with CPU default + safe fallback
- UNet + transformer segmentation backends (`hf_segformer_b0/b2/b5` scratch-init, `transunet_tiny`, `segformer_mini`) with checkpoint/resume + fixed/random validation sample tracking
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

HPC GA bundle generation:
```bash
microseg-cli hpc-ga-generate --config configs/hpc_ga.default.yml --dataset-dir outputs/prepared_dataset --output-dir outputs/hpc_ga_bundle
```

HPC GA feedback summary report:
```bash
microseg-cli hpc-ga-feedback-report \
  --config configs/hpc_ga.default.yml \
  --feedback-sources outputs/hpc_ga_bundle \
  --output-path outputs/hpc_ga_feedback/feedback_report.json
```

Single-script top-5 hydride benchmark run + dashboard:
```bash
python scripts/hydride_benchmark_suite.py --config configs/hydride/benchmark_suite.top5.yml --strict
```
- Outputs include consolidated JSON/CSV summaries, aggregate mean/std tables, and HTML dashboard sections for run-level training curves (`loss`, `accuracy`, `IoU` vs epoch), model size, parameter count, and runtime effort metrics.

## Beginner End-To-End Workflow

1. Prepare data:
- Start with `docs/training_data_requirements.md`.
- Use GUI `Dataset Prep + QA` tab or CLI `dataset-prepare` / `dataset-qa`.
2. Run baseline inference:
- GUI `Input` + `Run Segmentation` or CLI `microseg-cli infer`.
3. Correct masks:
- Use GUI correction tools and export corrected samples.
4. Train and evaluate:
- Use GUI `Training` + `Evaluation` tabs or CLI `train` + `evaluate`.
5. Compare runs:
- Use GUI `Run Review` tab for metric deltas.
6. Scale on HPC:
- Use GUI `HPC GA Planner` or CLI `hpc-ga-generate`.
- Optionally run `Analyze Feedback` in GUI or `hpc-ga-feedback-report` in CLI before the next sweep.
- Upload bundle and run `submit_all.sh` on scheduler environment.

## Frozen Checkpoints

- Metadata registry: `frozen_checkpoints/model_registry.json`
- Guidance: `docs/frozen_checkpoint_registry.md`
- Binary weights are intentionally excluded from git tracking.
- Tiny smoke-checkpoint generator: `python scripts/generate_smoke_checkpoint.py --force`
- Lifecycle folders: `frozen_checkpoints/smoke`, `frozen_checkpoints/candidates`, `frozen_checkpoints/promoted`

## Documentation

- Docs index: `docs/README.md`
- Mission: `docs/mission_statement.md`
- Phase roadmap: `docs/development_roadmap.md`
- Foundation strategy: `docs/foundation_strategy.md`
- Current gap analysis: `docs/current_state_gap_analysis.md`
- Repository health audit: `docs/repo_health_audit.md`
- Training data requirements: `docs/training_data_requirements.md`
- GUI user guide: `docs/gui_user_guide.md`
- HPC GA user guide: `docs/hpc_ga_user_guide.md`
- HPC GA developer guide: `docs/hpc_ga_developer_guide.md`
- Hydride end-to-end research workflow: `docs/hydride_research_workflow.md`
- Phase 17 HPC GA feedback status: `docs/phase17_hpc_ga_feedback.md`
- Phase 18 transformer backend status: `docs/phase18_transformer_backends.md`
- Phase 19 SOTA HF transformer integration status: `docs/phase19_hf_sota_transformers.md`
- Phase 20 benchmark suite orchestration status: `docs/phase20_benchmark_suite_orchestration.md`
- Configuration workflow: `docs/configuration_workflow.md`
- Development workflow + phase closeout gate: `docs/development_workflow.md`
- Developer guide: `developer_guide.md`
- Repository contract: `AGENTS.md`

## Contributing

- Contributor guide: `CONTRIBUTE.md`
- Working contract: `AGENTS.md`

## License

MIT (see `LICENSE`).
