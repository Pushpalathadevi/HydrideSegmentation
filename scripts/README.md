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
  - `hpc-ga-generate` creates GA-planned Slurm/PBS/local script bundles for multi-candidate GPU HPC sweeps.
- `run_phase_gate.py` standalone wrapper for phase closeout checks/stocktake reporting.
  - installed console entry point: `microseg-phase-gate`
- `generate_smoke_checkpoint.py` creates deterministic tiny random-weight `.pth` checkpoints for
  smoke-testing model loading/evaluation paths without large binary artifacts.
- `hydride_benchmark_suite.py` runs multi-model hydride benchmark suites (train+eval) and writes consolidated
  JSON/CSV summaries and a single HTML comparison dashboard.
  - includes per-run training curves (`loss`, `accuracy`, `IoU` vs epoch), model artifact size, parameter count, and train/eval/total runtime fields.
  - aggregate output includes mean/std rollups for quality and runtime metrics across seeds.
  - installed console entry point: `microseg-benchmark-suite`
