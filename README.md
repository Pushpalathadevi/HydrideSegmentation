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
- UNet + transformer segmentation backends (`hf_segformer_b0/b2/b5`, `smp_unet_resnet18`, `transunet_tiny`, `segformer_mini`) with checkpoint/resume + fixed/random validation sample tracking
- Air-gapped local transfer-learning support via `pre_trained_weights/` registry + bundle validation (`unet_binary`, `smp_unet_resnet18`, `hf_segformer_b0/b2/b5`, `transunet_tiny`, `segformer_mini`)
- JSON + HTML run reports for training and evaluation
- Frozen checkpoint metadata registry for model selection guidance

## Standards Baseline

The documentation and implementation policies are aligned with and adapted from the standards style used in
[DeepImageDeconvolution](https://github.com/kvmani/DeepImageDeconvolution), then extended for segmentation-specific
annotation, correction, and deployment needs.

## Installation

```bash
pip install -r requirements-core.txt
pip install -e .

# Desktop GUI profile
pip install -r requirements-gui.txt

# Fully pinned CPU-first reproducible baseline
pip install -r envs/microseg-core.lock.txt
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

Training with AMP + gradient accumulation + deterministic controls:
```bash
microseg-cli train --config configs/train.default.yml \
  --set amp_enabled=true \
  --set grad_accum_steps=2 \
  --set num_workers=4 \
  --set pin_memory=true \
  --set persistent_workers=true \
  --set deterministic=true
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
If `mask_colormap` is omitted, class colors are resolved from `--class-map-path`, `MICROSEG_CLASS_MAP_PATH`,
then `configs/segmentation_classes.json`.

Binary mask auto-normalization (`0` stays background, non-zero can be foreground):
```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.default.yml \
  --set binary_mask_normalization=nonzero_foreground
```
`two_value_zero_background` remains available for stricter two-value-only remapping.

Dataset QA:
```bash
microseg-cli dataset-qa --config configs/dataset_qa.default.yml --strict
```

Registry validation:
```bash
microseg-cli validate-registry --config configs/registry_validation.default.yml --strict
```

Pretrained bundle validation:
```bash
microseg-cli validate-pretrained --registry-path pre_trained_weights/registry.json --strict
```

Download local pretrained bundles (connected machine):
```bash
python scripts/download_pretrained_weights.py --targets all --force
```

Generate pretrained inventory report (JSON + markdown):
```bash
python scripts/pretrained_inventory_report.py \
  --registry-path pre_trained_weights/registry.json \
  --output-path outputs/pretrained_weights/inventory_report.json
```

Phase closeout gate:
```bash
microseg-cli phase-gate --phase-label "Phase N" --strict
```

HPC GA bundle generation:
```bash
microseg-cli hpc-ga-generate --config configs/hpc_ga.default.yml --dataset-dir outputs/prepared_dataset --output-dir outputs/hpc_ga_bundle
```

Air-gapped low-friction HPC sweep generation:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset \
  --output-dir outputs/hpc_ga_bundle_airgap_pretrained
```

Top-5 scratch HPC sweep profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_scratch.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_scratch
```

Top-5 local-pretrained HPC sweep profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_airgap_pretrained
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
- Benchmark mode now supports hard-fail dataset freeze checks (`expected_dataset_manifest_sha256`, `expected_split_id_file`).
- When `benchmark_mode=true` and `dataset_manifest.json` is missing, the suite auto-generates it from `train/val/test`.
- Outputs include consolidated JSON/CSV summaries, aggregate mean/std tables, and HTML dashboard sections for run-level training curves (`loss`, `accuracy`, `IoU` vs epoch), tracked validation sample panels/IoU summaries, model size, parameter count, runtime effort metrics, and evaluation scientific metrics.

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
- Offline pretrained transfer workflow: `docs/offline_pretrained_transfer_workflow.md`
- Single-file HPC real-data runbook (scratch + local-pretrained + dashboard): `docs/hpc_airgap_top5_realdata_runbook.md`
- Pretrained model catalog + citations: `docs/pretrained_model_catalog.md`
- Model architecture manuscript foundation (Mermaid diagrams + critical comparison): `docs/model_architecture_manuscript_foundation.md`
- Pretrained citation BibTeX: `docs/pretrained_model_citations.bib`
- Configuration workflow: `docs/configuration_workflow.md`
- Development workflow + phase closeout gate: `docs/development_workflow.md`
- Developer guide: `developer_guide.md`
- Repository contract: `AGENTS.md`

## Contributing

- Contributor guide: `CONTRIBUTE.md`
- Working contract: `AGENTS.md`

## License

MIT (see `LICENSE`).
