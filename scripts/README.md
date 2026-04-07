# Scripts

This folder is the target for thin CLI orchestration scripts.

Rules:
- Keep business logic in library modules.
- Scripts should parse args, load config, call core APIs, and persist outputs.

Current scripts:
- `build_docs.py` builds the Sphinx HTML site and browser-rendered PDF archive from the repository docs.
  - use `microseg-docs` after installing package entry points
  - use `python scripts/build_docs.py` for a direct local build
- `package_corrections_dataset.py` packages exported correction samples into train/val/test layout.
- `microseg_cli.py` unified CLI for inference, training, evaluation, model listing, and correction-dataset packaging with YAML + `--set` overrides.
  - supports optional GPU runtime selection for inference/training/evaluation with CPU fallback.
  - training supports `unet_binary`, SMP backends (`smp_unet_resnet18`, `smp_deeplabv3plus_resnet101`, `smp_unetplusplus_resnet101`, `smp_pspnet_resnet101`, `smp_fpn_resnet101`), HF transformer backends (`hf_segformer_b0/b2/b5`, `hf_upernet_swin_large`), internal transformer variants, `torch_pixel`, and `sklearn_pixel`.
  - training/evaluation can emit HTML reports and tracked sample panels.
  - `models` command reads frozen checkpoint metadata for dynamic help text.
  - `validate-registry` validates frozen checkpoint metadata schema/constraints.
  - `dataset-split` builds leakage-aware deterministic train/val/test splits from correction exports.
  - `dataset-prepare` converts unsplit source/masks datasets into train/val/test layout with global ID-suffixed names.
  - `dataset-prepare` uses leakage-aware auto-split by default and optionally supports RGB mask conversion via configurable colormap.
  - `dataset-qa` runs packaged dataset quality checks.
  - `phaseid-benchmark` runs the raw `.oh5` phase-ID workflow end to end: extraction -> split prep -> QA -> benchmark suite -> PPTX deck.
  - `phase-gate` command runs mandatory end-of-phase checks and writes closeout artifacts.
  - `preflight` runs unified workflow checks for train/eval/benchmark/deploy modes.
  - `deploy-package` creates a deployment handoff bundle (`deployment_manifest.json` + checksums).
  - `deploy-validate` validates package integrity and optional SHA256 checks.
  - `deploy-smoke` runs one-image smoke inference from a deployment package.
  - `deploy-health` runs runtime health checks (`package_validation`, `model_load`, `preprocess`, `inference`, `output_write`) and supports queue-style concurrent batch validation (`--image-dir`, `--max-workers`).
  - `deploy-worker-run` runs queue-safe service-mode worker batches with bounded queue controls (`--max-workers`, `--max-queue-size`).
    - service-mode runs can also write per-inference feedback evidence records (`microseg.feedback_record.v1`).
  - `deploy-canary-shadow` compares candidate and baseline packages on the same images (with optional GT-mask quality gain reporting).
  - `deploy-perf` runs deployment latency/throughput harness (`warmup`, `repeat`, `p95`, `throughput`).
  - `feedback-bundle` exports unsent deployment feedback records into transfer bundles.
  - `feedback-ingest` validates + ingests bundles centrally with record dedup and review-queue generation.
  - `feedback-build-dataset` builds weighted train/val/test datasets from ingested feedback records.
  - `feedback-train-trigger` evaluates/executes threshold-based retraining cycles without auto-promotion.
  - `promote-model` applies threshold policy to benchmark aggregate rows and can update registry stage metadata.
  - `support-bundle` collects run diagnostics/log artifacts into a zipped support bundle.
  - `compatibility-matrix` writes environment/runtime fingerprint JSON for reproducibility audits.
  - `validate-pretrained` validates local pretrained registry metadata, paths, and optional checksums.
  - `hpc-ga-generate` creates GA-planned Slurm/PBS/local script bundles for multi-candidate GPU HPC sweeps.
  - `hpc-ga-generate` supports low-friction pretrained controls (`pretrained_init_mode`, `pretrained_model_map`) for air-gapped transfer learning.
- `download_pretrained_weights.py` stages local pretrained bundles for offline transfer learning.
  - supports HF SegFormer `b0/b2/b5`, HF UPerNet Swin-Large, SMP decoder-family backends (U-Net/DeepLabV3+/U-Net++/PSPNet/FPN), `unet_binary` ResNet18-derived bootstrap, and internal ViT-tiny bootstrap bundles for `transunet_tiny`/`segformer_mini`.
- `pretrained_inventory_report.py` builds JSON/markdown inventory reports from `pre_trained_weights/registry.json` for reporting/manuscript traceability.
- `run_phase_gate.py` standalone wrapper for phase closeout checks/stocktake reporting.
  - installed console entry point: `microseg-phase-gate`
- `generate_smoke_checkpoint.py` creates deterministic tiny random-weight `.pth` checkpoints for
  smoke-testing model loading/evaluation paths without large binary artifacts.
- `build_debug_duplicate_dataset.py` creates tiny duplicated train/val datasets from one image for
  end-to-end pipeline integrity checks.
  - supports optional `--resize-width/--resize-height` downscaling to keep transformer debug runs fast on CPU.
- `hydride_benchmark_suite.py` runs multi-model hydride benchmark suites (train+eval) and writes consolidated
  JSON/CSV summaries and concise HTML comparison dashboards.
  - writes queue-debuggable structured logs: `logs/suite_events.jsonl` and `logs/<run_tag>/run_events.jsonl`.
  - executes model families in fixed order: transformer -> deeplab -> advanced -> unet.
  - supports `--single-seed` override (or `single_seed_only` in YAML) to run only the first configured seed.
  - includes per-run training curves (`loss`, `accuracy`, `IoU` vs epoch), model artifact size, parameter count, train/eval/total runtime, and mean epoch timing fields (`mean_train_epoch_seconds`, `mean_validation_epoch_seconds`, `mean_epoch_runtime_seconds`).
  - emits one per-run detail page (`runs/<run_tag>/inside.html`) with links to plots/panels/logs; images are opened on demand rather than embedded at summary load.
  - includes trainable-parameter count, checkpoint weight statistics, runtime hardware fields, and FLOPs estimates when available.
  - includes `cohen_kappa` in run-level and aggregate benchmark metrics.
  - emits canonical campaign artifacts `summary.json` + `summary.html` (plus compatibility files `benchmark_summary.json` + `benchmark_dashboard.html`).
  - local-pretrained runs with missing weights are marked `pretrained_missing`, logged with actionable fixes, and remaining runs continue.
  - failed runs generate explicit skip/failure logs and the suite continues by default (`continue_on_failure=true`).
  - aggregate output includes mean/std rollups for quality and runtime metrics across seeds.
  - installed console entry point: `microseg-benchmark-suite`
- `generate_benchmark_lab_meeting_ppt.py` converts `benchmark_summary.json` into a deck manifest and `.pptx`.
- `build_benchmark_lab_meeting_ppt.js` is the PptxGenJS authoring script for the generated benchmark deck.
- `build_windows_installer.ps1` builds the Qt desktop executable (`PyInstaller`) and optional single offline installer (`Inno Setup`).
