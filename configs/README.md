# Configurations

Target configuration groups:
- `models/` model registry entries and model-specific defaults
- `pipelines/` inference/training pipeline definitions
- `app/` desktop app defaults
  - `app/desktop_ui.default.yml` Qt desktop appearance and export-default settings (`microseg.desktop_ui_config.v1`)

All configs should be versioned and reproducible.

Related metadata:
- frozen model metadata registry: `frozen_checkpoints/model_registry.json`
- smoke-checkpoint generator script: `scripts/generate_smoke_checkpoint.py`
- local pretrained registry (air-gap transfer): `pre_trained_weights/registry.json`
- pretrained bundle bootstrap script: `scripts/download_pretrained_weights.py`
- hydride benchmark presets: `configs/hydride/README.md`

Additional workflow config:
- `inference.default.yml` for default CLI inference
  - defaults to `Hydride ML (UNet)` and resolves the trained checkpoint through repo-relative `frozen_checkpoints/model_registry.json`
  - includes `result_export` switches for optional extended metrics and distribution charts
- `phase_gate.default.yml` for end-of-phase closeout checks
- `preflight.default.yml` for unified train/eval/benchmark/deploy preflight checks
- `deployment_package.default.yml` for deployment bundle creation inputs
- `deploy_health.default.yml` for runtime deployment health checks and queue-style batch validation
- `deploy_worker.default.yml` for queue-safe deployment service worker batch execution
- `deploy_canary_shadow.default.yml` for candidate-vs-baseline canary/shadow deployment comparison
- `deploy_perf.default.yml` for deployment latency/throughput harness runs
- `promotion_policy.default.yml` for objective model-promotion thresholds
- `support_bundle.default.yml` for run diagnostics/support artifact bundling
- `registry_validation.default.yml` for frozen registry validation
- `dataset_split.default.yml` for leakage-aware split planning
- `dataset_qa.default.yml` for packaged dataset QA checks
- `dataset_prepare.default.yml` for unsplit source/masks -> split dataset preparation
  - includes `split_strategy`, leakage-group controls, RGB `mask_input_type`, `mask_colormap`, optional `binary_mask_normalization`, optional `class_map_path` (fallback: `configs/segmentation_classes.json`), and the shared `augmentation` block
  - augmentation examples: `dataset_prepare.augmentation.disabled.yml`, `dataset_prepare.augmentation.shadow_blur.yml`, `dataset_prepare.augmentation.debug.yml`, `dataset_prepare.augmentation.multi.yml`
  - scalar-or-range augmentation magnitudes are supported, such as `radius: [100, 300]` and `kernel_size: [3, 9]`
- `phaseid_oh5_benchmark.default.yml` for the single-command raw `.oh5` phaseId workflow
  - includes `.oh5` dataset-path resolution, phase-ID foreground mapping, dataset split/QA policy, benchmark template selection, and PPTX generation settings
- `data_prep.default.yml` for `prep-dataset` binary segmentation data preparation (`src/microseg/data_preparation`)
  - includes binarization mode/threshold controls, red-dominance RGB fallback, auto-Otsu fallback for noisy near-binary grayscale masks, empty-mask warn/error policy, resizing policy, debug inspection options, warning-related raw-mask expectations, and the shared `augmentation` block
- `tutorials/prepare_dataset.paired_tutorial.shadow_blur.yml` for the beginner paired-folder dataset tutorial
- `tutorials/train.tiny_unet_from_prepared.yml` for the matching tiny UNet training smoke/tutorial run
- `segmentation_classes.json` repo-level default class definitions used by correction/export and RGB-mask class-color fallback
- `hpc_ga.default.yml` for GA-based HPC multi-candidate script bundle generation
  - includes feedback-hybrid planning controls and metric/runtime fitness weights
- `hpc_ga.airgap_pretrained.default.yml` for low-friction air-gapped HPC sweeps
  - enables `pretrained_init_mode=auto` with backend-to-model mapping from `pre_trained_weights/registry.json`
- `hpc_ga.top5_scratch.default.yml` for the canonical top-5 scratch-only hydride sweep
- `hpc_ga.top5_airgap_pretrained.default.yml` for the canonical top-5 air-gapped local-pretrained sweep
- `feedback_capture.default.yml` for local per-inference feedback capture defaults
- `feedback_bundle.default.yml` for deployment-side feedback bundle export
- `feedback_ingest.default.yml` for central feedback bundle ingest + dedup + review queue
- `feedback_build_dataset.default.yml` for weighted active-learning dataset build policy
- `feedback_train_trigger.default.yml` for threshold-based retraining trigger orchestration
