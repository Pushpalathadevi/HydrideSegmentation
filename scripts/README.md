# Scripts

This folder is the target for thin CLI orchestration scripts.

Rules:
- Keep business logic in library modules.
- Scripts should parse args, load config, call core APIs, and persist outputs.

Current scripts:
- `package_corrections_dataset.py` packages exported correction samples into train/val/test layout.
- `microseg_cli.py` unified CLI for inference, training, evaluation, model listing, and correction-dataset packaging with YAML + `--set` overrides.
  - supports optional GPU runtime selection for inference/training/evaluation with CPU fallback.
  - training supports `unet_binary`, HF SegFormer scratch backends (`hf_segformer_b0/b2/b5`), internal transformer variants, `torch_pixel`, and `sklearn_pixel`.
  - training/evaluation can emit HTML reports and tracked sample panels.
  - `models` command reads frozen checkpoint metadata for dynamic help text.
  - `validate-registry` validates frozen checkpoint metadata schema/constraints.
  - `dataset-split` builds leakage-aware deterministic train/val/test splits from correction exports.
  - `dataset-prepare` converts unsplit source/masks datasets into train/val/test layout with global ID-suffixed names.
  - `dataset-prepare` uses leakage-aware auto-split by default and optionally supports RGB mask conversion via configurable colormap.
  - `dataset-qa` runs packaged dataset quality checks.
  - `phase-gate` command runs mandatory end-of-phase checks and writes closeout artifacts.
  - `validate-pretrained` validates local pretrained registry metadata, paths, and optional checksums.
  - `hpc-ga-generate` creates GA-planned Slurm/PBS/local script bundles for multi-candidate GPU HPC sweeps.
  - `hpc-ga-generate` supports low-friction pretrained controls (`pretrained_init_mode`, `pretrained_model_map`) for air-gapped transfer learning.
- `download_pretrained_weights.py` stages local pretrained bundles for offline transfer learning.
  - supports HF SegFormer `b0/b2/b5`, SMP U-Net ResNet18, `unet_binary` ResNet18-derived bootstrap, and internal ViT-tiny bootstrap bundles for `transunet_tiny`/`segformer_mini`.
- `pretrained_inventory_report.py` builds JSON/markdown inventory reports from `pre_trained_weights/registry.json` for reporting/manuscript traceability.
- `run_phase_gate.py` standalone wrapper for phase closeout checks/stocktake reporting.
  - installed console entry point: `microseg-phase-gate`
- `generate_smoke_checkpoint.py` creates deterministic tiny random-weight `.pth` checkpoints for
  smoke-testing model loading/evaluation paths without large binary artifacts.
- `build_debug_duplicate_dataset.py` creates tiny duplicated train/val datasets from one image for
  end-to-end pipeline integrity checks.
  - supports optional `--resize-width/--resize-height` downscaling to keep transformer debug runs fast on CPU.
- `hydride_benchmark_suite.py` runs multi-model hydride benchmark suites (train+eval) and writes consolidated
  JSON/CSV summaries and a single HTML comparison dashboard.
  - includes per-run training curves (`loss`, `accuracy`, `IoU` vs epoch), model artifact size, parameter count, and train/eval/total runtime fields.
  - includes tracked validation-sample panel galleries with per-image metric blocks from training reports.
  - includes trainable-parameter count, checkpoint weight statistics, runtime hardware fields, and FLOPs estimates when available.
  - emits canonical campaign artifacts `summary.json` + `summary.html` (plus compatibility files `benchmark_summary.json` + `benchmark_dashboard.html`).
  - local-pretrained runs with missing weights are marked `pretrained_missing`, logged with actionable fixes, and remaining runs continue.
  - aggregate output includes mean/std rollups for quality and runtime metrics across seeds.
  - installed console entry point: `microseg-benchmark-suite`
- `build_windows_installer.ps1` builds the Qt desktop executable (`PyInstaller`) and optional single offline installer (`Inno Setup`).
