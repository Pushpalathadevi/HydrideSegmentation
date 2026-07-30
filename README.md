# HydrideSegmentation -> MicroSeg Platform (Transition)

Current release version: `0.22.0`

This repository is transitioning from a hydride-specific toolkit into a general local platform for microstructural segmentation.
Hydride segmentation is the first validated workflow.

## Mission

Build a scientifically robust, CPU-first desktop + CLI platform for microstructural segmentation with:
- pluggable model backends
- quantitative analysis pipelines
- local prediction review and export
- reproducible experiment and deployment artifacts

See `docs/mission_statement.md`.

## Core Capabilities

- Registry-backed segmentation orchestration (`src/microseg`)
- Browser-based intranet app (`microseg-web`) for colleagues who should not have to install anything:
  - one air-gapped host serves the whole team; clients need only a browser
  - drag-and-drop upload plus bundled example images for testing without private data
  - immediate upload preview and clickable thumbnail previews for bundled examples
  - strict 5 MB image validation, memory-only processing, and asynchronous progress with a live log
  - both the conventional pipeline and any installed trained model, from the same registry the desktop app uses
  - radial hydride fraction (Fn) as the headline result, length-weighted and count-based, with a user-controlled angle threshold and opt-in views showing which hydrides were counted
  - per-parameter in-app help and a dedicated help page
  - trained models preloaded at startup so the first request is not slowed by model loading
  - see `docs/intranet_web_app.md`
- Qt desktop GUI (`hydride-gui`) with:
  - a unified `Segmentation` workspace showing input on the left and predicted mask on the right, with a clear not-ready state before inference
  - live, pre-filled conventional-segmentation controls beside the images; hover help explains every parameter and edits trigger a debounced background refresh
  - file/edit/view/help desktop-style menus
  - bundled sample image onboarding (`Load Sample` / `File -> Open Sample`)
  - split dashboard layout with a fixed-width, scrollable control sidebar and a large central visual workspace
  - collapsible sidebar groups for inference setup, quantification, display, and export/session actions
  - in-view zoom, pan, fit-to-view, and 100% controls on each image canvas
  - class index + color map editing
  - conventional-pipeline controls (CLAHE/adaptive/morphology/crop/area threshold)
  - optional spatial calibration (manual line-draw or TIFF metadata scan) for micron-based reporting
  - synchronized zoom/pan and layer transparency for review
  - Results Dashboard with adjustable orientation/size plotting controls
  - scalar statistics panel (fraction/count/density/orientation/size summaries) with number- and length-weighted Fn metrics
  - full results-package export (`results_summary.json`, `results_report.html`, `results_report.pdf`, `results_metrics.csv`, `artifacts_manifest.json`)
  - configurable report profiles (`balanced`/`full`/`audit`) with section + metric selection
  - one-action recursive batch inference + export with per-run manifests under `runs/`, aggregate outputs (`batch_results_summary.json`, `batch_results_report.html`, `batch_results_report.pdf`, `batch_metrics.csv`), and auto-opened summary review in the GUI
  - YAML-driven desktop appearance settings (font sizes, contrast, spacing, startup geometry) with in-app settings dialog
  - gear menu for secondary panels and a real status bar so the workspace stays image-first
  - project/session save-load
  - focused inference, mask review, quantification, and results-review workspace
  - persistent desktop logs under `outputs/logs/desktop/`
- Student notebook labs under `docs/student_notebooks.md`, `docs/tutorials/*.md`, and `docs/notebooks/*.ipynb` for preprocessing, training, evaluation, and inference-loop walkthroughs
- Unified CLI (`microseg-cli`) for infer/train/evaluate/models/data preparation
- Default trained hydride inference checkpoint is registered via `frozen_checkpoints/model_registry.json` and resolved from `frozen_checkpoints/candidates/U_net_binary_best_checkpoint.pt` when present locally; additional trained models can be added through `frozen_checkpoints/model_registry.local.json` and will appear in GUI/CLI discovery automatically
- Committed model discovery metadata uses repository-relative `checkpoint_path_hint` values under `frozen_checkpoints/`; absolute paths are reserved for explicit direct `checkpoint_path` / `weights_path` overrides or machine-local overlays.
- Deployment operations tooling (`preflight`, `deploy-package`, `deploy-validate`, `deploy-smoke`, `promote-model`, `support-bundle`)
- GPU-compatible training/inference/evaluation with CPU default + safe fallback
- UNet + transformer segmentation backends (`hf_segformer_b0/b2/b5`, `hf_upernet_swin_large`, `smp_unet_resnet18`, `smp_deeplabv3plus_resnet101`, `smp_unetplusplus_resnet101`, `smp_pspnet_resnet101`, `smp_fpn_resnet101`, `transunet_tiny`, `segformer_mini`) with checkpoint/resume + fixed/random validation sample tracking
- Air-gapped local transfer-learning support via `pre_trained_weights/` registry + bundle validation (`unet_binary`, SMP decoder family, `hf_segformer_b0/b2/b5`, `hf_upernet_swin_large`, `transunet_tiny`, `segformer_mini`)
- JSON + HTML run reports for training and evaluation
- Frozen checkpoint metadata registry for model selection guidance
- Local-only checkpoint overlays can be staged in `frozen_checkpoints/model_registry.local.json` with binaries under `frozen_checkpoints/candidates/`; the overlay is ignored by git but still available to the GUI and CLI at runtime.

## Standards Baseline

The documentation and implementation policies are aligned with and adapted from the standards style used in
[DeepImageDeconvolution](https://github.com/kvmani/DeepImageDeconvolution), then extended for segmentation-specific
annotation, quantitative review, and deployment needs.

The repository now ships a Sphinx-based documentation system that is treated as part of the product surface:

- source markdown, SVG diagrams, and math live under `docs/`
- the docs build prefers the vendored MathJax bundle in `docs/_static/mathjax/es5/tex-mml-chtml.js`, so offline HTML rendering works without a CDN
- HTML docs build with `python scripts/build_docs.py --html-only`
- HTML + PDF docs build with `python scripts/build_docs.py`
- the generated HTML can be served locally from `docs/_build/html/`

Documentation build dependencies live in `requirements-docs.txt`.

## Installation

```bash
pip install -r requirements-core.txt
pip install -e .

# PPTX generation for benchmark decks
npm install

# Desktop GUI profile
pip install -r requirements-gui.txt

# Intranet web server profile
pip install -r requirements-web.txt

# Fully pinned CPU-first reproducible baseline
pip install -r envs/microseg-core.lock.txt
```

Qt GUI dependency:
```bash
pip install PySide6
```

Windows offline installer build tooling:
```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_windows_installer.ps1
```
See `docs/windows_offline_installer.md`.

## Primary Usage

Intranet web app (serves the whole team from one host):
```bash
python scripts/run_web_server.py
```

The command prints the `http://<host>:5005/` links to share with colleagues. Open the host firewall for the port, and the team needs nothing installed. Deployment, configuration, service setup, and the offline wheelhouse install are covered in `docs/intranet_web_app.md`.

Qt GUI:
```bash
hydride-gui
```

Qt GUI with explicit appearance config:
```bash
hydride-gui --ui-config configs/app/desktop_ui.default.yml
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

`configs/inference.default.yml` selects `Hydride ML (UNet)` by default. Use `--model-name "Hydride Conventional"` only when you intentionally want the deterministic classical fallback.

The ML inference path uses GUI-style preprocessing by default unless disabled in the YAML:
- aspect-ratio-preserving resize to a `512` long side
- auto-contrast enabled
- preprocessing metadata recorded in manifests

## Documentation

The canonical documentation landing page is [`docs/index.md`](docs/index.md).
For students and new contributors, start with [`docs/cli_windows_linux.md`](docs/cli_windows_linux.md), [`docs/tutorials/05_paired_dataset_preparation_and_training_cli.md`](docs/tutorials/05_paired_dataset_preparation_and_training_cli.md), and [`docs/learning_path.md`](docs/learning_path.md).
For core terminology, use [`docs/glossary.md`](docs/glossary.md).
For copy-paste Windows/Linux CLI workflows, use [`docs/usage_commands.md`](docs/usage_commands.md). For Fn formulation, estimation caveats, and audit artifacts, use [`docs/results_analysis.md`](docs/results_analysis.md).

Build the docs locally:

```bash
pip install -r requirements-docs.txt
python scripts/build_docs.py
python -m http.server 8000 -d docs/_build/html
```

HTML-only build:

```bash
python scripts/build_docs.py --html-only
```

The docs site includes:

- exact command recipes
- output-path guidance
- mathematical algorithm notes
- SVG diagrams for architecture, workflow, and GUI schematics
- a classical segmentation flow sheet with parameter meaning and tuning guidance
- model-family architecture comparisons with original citations
- a beginner on-ramp for students and new contributors
- a beginner learning path and glossary
- rendered notebook tutorials for preprocessing, training, inference, correction, and evaluation
- repository policy updates that make documentation part of the change contract


## Unified trained-model inference loading

Model-based inference now uses one architecture-aware loader shared by GUI, legacy API/service adapters, and ML inference entry points.

- Discovery sources:
  - training runs under `outputs/runs/<run_name>/`
  - frozen registry entries in `frozen_checkpoints/model_registry.json`
  - optional local registry overlay `frozen_checkpoints/model_registry.local.json`
- Run eligibility requires successful metadata + checkpoint artifacts (`report.json` status ok/success/completed and resolvable `model_path`).
- Failed/incomplete runs (for example folders with only `error_report.json`) are excluded from inference-capable discovery with explicit diagnostics.
- Architecture is reconstructed from run metadata (`model_architecture` in report/manifest/config), then loaded via the unified binary-backend loader used in training/evaluation (`unet_binary`, SMP families, HF SegFormer/UPerNet, `transunet_tiny`, `segformer_mini`).
- The CLI `infer` command accepts `--model` / `--model-name` and defaults to the first discovered trained model when omitted.

For legacy service/API callers, pass one of:
- `run_dir`
- `registry_model_id`
- `checkpoint_path`

to select the exact inference artifact and avoid architecture mismatches.


Training (UNet):
```bash
microseg-cli train --config configs/train.default.yml --set epochs=20
```

Training with fixed input-size policy + AMP + gradient accumulation:
```bash
microseg-cli train --config configs/train.default.yml   --set input_hw=[512,512]   --set input_policy=random_crop   --set val_input_policy=letterbox   --set amp_enabled=true   --set grad_accum_steps=2   --set deterministic=true
```
See [docs/input_size_policy.md](docs/input_size_policy.md) for policy details, collation fallback mode, and HPC memory guidance.

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

Unsplit `source/masks` auto-prepare (leakage-aware default + global IDs):
```bash
microseg-cli dataset-prepare --config configs/dataset_prepare.default.yml
```
See [docs/data_preparation.md](docs/data_preparation.md) for the dedicated binary segmentation data preparation subsystem (pairing, binarization, resizing, manifesting, and Oxford/MaDo exports).

Leakage-safe train-only shadow + blur augmentation during dataset preparation:
```bash
microseg-cli dataset-prepare \
  --config configs/dataset_prepare.augmentation.shadow_blur.yml
```
The canonical `dataset-prepare` path now supports a shared YAML augmentation block with deterministic seeds, split targeting, per-operation probabilities, per-sample variant counts, and debug inspection artifacts. The same augmentation block also flows through `train` / `evaluate` when `auto_prepare_dataset=true`.
Augmentation magnitude fields accept either scalars or inclusive ranges where meaningful: for example `radius: [100, 300]`, `sigma: [80, 160]`, and `kernel_size: [3, 9]` samples valid odd blur kernels.

Paired JPG + RGB PNG mask to MaDo-style dataset prep:
```bash
python scripts/microseg_cli.py prepare_dataset \
  --config configs/hydride/prepare_dataset.paired_rgb_mask.mado.yml \
  --input-dir D:/data/hydride_pairs \
  --output-root D:/data/HydrideData7.0 \
  --style mado \
  --target-size 512 \
  --crop-train random \
  --crop-eval center \
  --mask-r-min 200 --mask-g-max 60 --mask-b-max 60 \
  --allow-red-dominance-fallback \
  --auto-otsu-for-noisy-grayscale \
  --empty-mask-action warn \
  --seed 42 --train-frac 0.8 --val-frac 0.1 \
  --max-val-examples 200 \
  --max-test-examples 200
```
Use `--dry-run` to validate pairing and inspect planned outputs without writing dataset files.
`--max-val-examples` and `--max-test-examples` are optional caps; any remainder is assigned to `train`.

`prepare_dataset` supports `{stem}.jpg + {stem}_mask.png` by default, and it can also support `{stem}.jpg + {stem}.png` with the explicit `same_stem_pairing` YAML block documented in [docs/data_preparation.md](docs/data_preparation.md).
In debug mode (`--debug --num-debug N`), dataset preparation exports input/output images, input/processed masks, mask-difference views, overlay panels, and per-sample criteria JSON.
All-zero output masks are now explicitly flagged (`--empty-mask-action warn|error`), and noisy near-binary grayscale masks can auto-switch to Otsu (`--auto-otsu-for-noisy-grayscale`).


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

Single-command raw `.oh5` -> benchmark -> PPTX workflow:
```bash
microseg-cli phaseid-benchmark \
  --config configs/phaseid_oh5_benchmark.default.yml \
  --raw-input-dir D:/phaseid/raw_oh5 \
  --working-dir D:/phaseid/run_001
```
This workflow extracts image/mask pairs from `.oh5`, prepares a frozen split dataset, runs dataset QA, launches the configured benchmark suite, and generates a lab-meeting PPTX from `benchmark_summary.json`.

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

Unified workflow preflight:
```bash
microseg-cli preflight --config configs/preflight.default.yml --mode train --strict
```

Deployment package creation + validation + smoke:
```bash
microseg-cli deploy-package --config configs/deployment_package.default.yml --model-path outputs/training/model.pth
microseg-cli deploy-validate --package-dir outputs/deployments/<package_dir> --strict
microseg-cli deploy-smoke --package-dir outputs/deployments/<package_dir> --image-path test_data/sample.png
```

Deployment runtime health checks (queue-style batch capable):
```bash
microseg-cli deploy-health \
  --config configs/deploy_health.default.yml \
  --package-dir outputs/deployments/<package_dir> \
  --image-dir test_data \
  --max-workers 4 --strict
```

Deployment service-mode worker batch (bounded queue):
```bash
microseg-cli deploy-worker-run \
  --config configs/deploy_worker.default.yml \
  --package-dir outputs/deployments/<package_dir> \
  --image-dir test_data \
  --max-workers 4 --max-queue-size 64 --strict
```

Canary-shadow package comparison:
```bash
microseg-cli deploy-canary-shadow \
  --config configs/deploy_canary_shadow.default.yml \
  --baseline-package-dir outputs/deployments/<baseline_pkg> \
  --candidate-package-dir outputs/deployments/<candidate_pkg> \
  --image-dir test_data \
  --mask-dir test_data/masks \
  --strict
```

Deployment latency/throughput benchmark:
```bash
microseg-cli deploy-perf \
  --config configs/deploy_perf.default.yml \
  --package-dir outputs/deployments/<package_dir> \
  --image-dir test_data \
  --warmup-runs 1 --repeat 3 --max-workers 4 --strict
```

Promotion gate from benchmark summary:
```bash
microseg-cli promote-model \
  --summary-json outputs/hydride_benchmark_suite/summary.json \
  --model-name hf_segformer_b2 \
  --registry-model-id hf_segformer_b2 \
  --policy-config configs/promotion_policy.default.yml \
  --update-registry --strict
```

Support diagnostics bundle + environment fingerprint:
```bash
microseg-cli support-bundle --config configs/support_bundle.default.yml
microseg-cli compatibility-matrix --output-path outputs/support_bundles/compatibility_matrix.json
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

Top-10 scratch HPC sweep profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_scratch.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_scratch
```

Top-10 local-pretrained HPC sweep profile:
```bash
microseg-cli hpc-ga-generate \
  --config configs/hpc_ga.top5_airgap_pretrained.default.yml \
  --dataset-dir outputs/prepared_dataset_hydride_v1 \
  --output-dir outputs/hpc_ga_bundle_top5_airgap_pretrained
```

Single-script top-10 hydride benchmark run + dashboard:
```bash
python scripts/hydride_benchmark_suite.py --config configs/hydride/benchmark_suite.top5.yml --strict
```
- Optional single-seed override from CLI (uses the first seed in suite YAML):
```bash
python scripts/hydride_benchmark_suite.py --config configs/hydride/benchmark_suite.top5.yml --single-seed
```
- Slurm single-job wrapper (repo-root-safe, enforces `./.venv` in job runtime):
```bash
./submitJob_1GPU.sh ./run_training_jobs.sh --dataset tiny --profile smoke
./submitJob_1GPU.sh ./run_training_jobs.sh --dataset tiny --profile full --single_seed true
./submitJob_1GPU.sh ./run_training_jobs.sh --dataset custom --dataset_dir /path/to/HydrideData6.0/mado_style --profile full
```
- The wrapper exports `HYDRIDE_REPO_ROOT` from submission time, so `run_training_jobs.sh` can recover repo root even when Slurm stages the script into a spool directory.
- `run_training_jobs.sh` hard-fails if `./.venv/bin/python` is missing or if dependency sanity checks fail (including `pydantic>=2`).
- Benchmark mode now supports hard-fail dataset freeze checks (`expected_dataset_manifest_sha256`, `expected_split_id_file`).
- When `benchmark_mode=true` and `dataset_manifest.json` is missing, the suite auto-generates it from `train/val/test`.
- Outputs include consolidated JSON/CSV summaries, aggregate mean/std tables, concise HTML summary pages, and per-run `inside.html` detail pages with links for training curves (`loss`, `accuracy`, `IoU`), tracked validation evolution curves, and validation sample panels.
- Summary/inside pages avoid embedded image preloading; heavy plots/panels open only through links.
- Canonical campaign artifacts: `summary.json` and `summary.html` (compatibility files `benchmark_summary.json` and `benchmark_dashboard.html` are still emitted).
- Suite execution order is now deterministic by model family: transformers first, then DeepLab, then other advanced models, and `unet_binary` last.
- If a local-pretrained run is missing required weights/registry artifacts, it is marked `pretrained_missing` with actionable fix text in the run log, and remaining runs continue.
- Failed train/eval runs write explicit skip/failure logs and do not stop remaining runs unless `continue_on_failure` is disabled in suite YAML.
- `run_training_jobs.sh` now defaults to non-strict completion (`--strict false`) so partial-success campaigns still finish and archive successful runs.
- Per-suite and per-run structured event logs are written (`logs/suite_events.jsonl`, `logs/<run_tag>/run_events.jsonl`) along with continuous `train.log` / `eval.log`.
- Suite scheduler auto-detects visible GPUs and runs serially on 0/1 GPU, or in parallel on multi-GPU allocations with one benchmark unit pinned per GPU (`CUDA_VISIBLE_DEVICES` scoped per subjob).
- New optional CLI controls: `--max-parallel-gpus auto|N`, `--parallel-jobs auto|N`, `--failure-policy continue|fail-fast`.
- Subjob diagnostics are written under `output_root/subjobs/<run_tag>/` (`stdout.log`, `stderr.log`, `metadata.json`, `command.sh`) for HPC debugging and failure triage.
- Metrics include `cohen_kappa` in evaluation, benchmark CSV, and dashboard summaries.
- Optional suite YAML watchdog keys (`command_idle_timeout_seconds`, `command_wall_timeout_seconds`) can auto-terminate stuck runs and continue the campaign.
- Bundled hydride benchmark suite configs now default both watchdog thresholds to `10800` seconds (`3` hours).

## Beginner End-To-End Workflow

1. Prepare data:
- Start with `docs/training_data_requirements.md`.
- Use GUI `Dataset Prep + QA` tab or CLI `dataset-prepare` / `dataset-qa`.
2. Run baseline inference:
- GUI `Segmentation` + `Run Segmentation` or CLI `microseg-cli infer`.
  - The desktop `Run Segmentation` action now uses the same in-process background worker path as batch inference, so the window stays responsive while warmed ML checkpoints and cached bundles are reused across runs.
  - The primary selector now exposes discovered trained models first, with `Hydride ML (UNet)` as the default trained checkpoint and `Hydride Conventional` as the deterministic fallback.
  - The Qt sidebar now defaults to a compact `Quick Start` + `Active Run` rail, with a separate `Run Setup / Status` card for model metadata, preprocessing summary, warm-load state, and progress.
  - The desktop log now lives in a shared bottom workspace strip instead of the left sidebar, and it is visible on startup.
  - `Run Batch` now performs recursive folder inference, writes the full batch export package in one pass (`runs/`, `batch_results_summary.json`, `batch_results_report.html`, `artifacts_manifest.json`, `resolved_config.json`), and opens the batch summary inspector automatically when the job finishes.
  - Batch export packages now place one complete per-image result bundle under `runs/` and link those per-run summaries from the root batch HTML/JSON package.
  - ML contrast adjustment now previews the actual processed-for-inference image in split view against the raw source, and desktop preprocessing logs explicitly report resize, contrast, channel duplication, and mask rescaling steps.
  - Desktop result tables and reports now round floating scientific metrics to two decimals for consistent operator-facing output.
  - The `Segmentation` tab keeps input and predicted mask side by side; before inference the result pane says `Result not ready yet`.
  - With `Hydride Conventional`, hover-documented parameters appear above the comparison and changes queue a 400 ms debounced rerun.
  - Input, mask, overlay, and batch-inspector views ship with local zoom/pan/display-contrast tools; active-run image views keep pan/zoom synchronized during review.
  - The main window opens maximized by default when the UI config keeps `start_maximized: true`, and the image canvas now re-fits on tab switches and resize events.
- A live status banner shows the current stage, processed-image counts, elapsed time, percent complete, and ETA during batch jobs.
3. Review masks and metrics:
- Use GUI mask/overlay/results views, then export result packages as needed.
4. Train and evaluate:
- Use GUI `Training` + `Evaluation` tabs or CLI `train` + `evaluate`.
5. Compare runs:
- Use GUI `Run Review` tab for metric deltas.
6. Scale on HPC:
- Use the CLI `hpc-ga-generate` command for scheduler-ready bundles.
- Upload bundle and run `submit_all.sh` on scheduler environment.
7. Review benchmark and deployment reports, then keep promotion human-gated through policy checks.

## Frozen Checkpoints

- Metadata registry: `frozen_checkpoints/model_registry.json`
- Guidance: `docs/frozen_checkpoint_registry.md`
- Binary weights are intentionally excluded from git tracking.

### Installing A Trained Model

Because checkpoint binaries are not tracked in git, a fresh clone has no `.pt` file and the ML models are shown as unavailable until one is installed. The conventional pipeline works without any checkpoint.

From the GUI: `Settings > Installed Models...` then `Install Model...`, pick the checkpoint, confirm the name, and click `Install`. The model appears in the selector immediately, with no restart.

From the CLI:

```bash
microseg-cli install-model --checkpoint path/to/best_checkpoint.pth --model-id my_unet_v1 --nickname my_unet_v1_optical
```

The installer reads the architecture, input size, checksum, and training provenance from the checkpoint itself, copies the file into the lifecycle folder matching its stage, verifies it with a real load and one forward pass, and records it in the untracked `frozen_checkpoints/model_registry.local.json` overlay. Failed installs roll back completely. Related commands: `microseg-cli inspect-checkpoint`, `microseg-cli uninstall-model`, `microseg-cli models`.

Full walkthrough: `docs/gui_model_integration_guide.md`.
- Tiny smoke-checkpoint generator: `python scripts/generate_smoke_checkpoint.py --force`
- Lifecycle folders: `frozen_checkpoints/smoke`, `frozen_checkpoints/candidates`, `frozen_checkpoints/promoted`

## Documentation

- Docs index: `docs/README.md`
- Code architecture + data flow map: `docs/code_architecture_map.md`
- Mission: `docs/mission_statement.md`
- Phase roadmap: `docs/development_roadmap.md`
- Deployment + productization master roadmap: `docs/deployment_productization_master_roadmap.md`
- Foundation strategy: `docs/foundation_strategy.md`
- Current gap analysis: `docs/current_state_gap_analysis.md`
- Repository health audit: `docs/repo_health_audit.md`
- Training data requirements: `docs/training_data_requirements.md`
- GUI user guide: `docs/gui_user_guide.md`
- Intranet web app deployment: `docs/intranet_web_app.md`
- Deferred Hydride Connectivity Index specification and audit: `docs/hydride_connectivity_index.md`
- Windows offline installer workflow: `docs/windows_offline_installer.md`
- HPC GA user guide: `docs/hpc_ga_user_guide.md`
- HPC GA developer guide: `docs/hpc_ga_developer_guide.md`
- Hydride end-to-end research workflow: `docs/hydride_research_workflow.md`
- Phase 18 transformer backend status: `docs/phase18_transformer_backends.md`
- Phase 19 SOTA HF transformer integration status: `docs/phase19_hf_sota_transformers.md`
- Phase 20 benchmark suite orchestration status: `docs/phase20_benchmark_suite_orchestration.md`
- Phase 26 deployment runtime modes status: `docs/phase26_deployment_runtime_modes.md`
- Offline pretrained transfer workflow: `docs/offline_pretrained_transfer_workflow.md`
- Single-file HPC top-10 real-data runbook (scratch + local-pretrained + dashboard): `docs/hpc_airgap_top5_realdata_runbook.md`
- Pretrained model catalog + citations: `docs/pretrained_model_catalog.md`
- Model architecture manuscript foundation (Mermaid diagrams + critical comparison): `docs/model_architecture_manuscript_foundation.md`
- Pretrained citation BibTeX: `docs/pretrained_model_citations.bib`
- Configuration workflow: `docs/configuration_workflow.md`
- Deployment operations workflow: `docs/deployment_ops_workflow.md`
- Failure taxonomy and error codes: `docs/failure_taxonomy.md`
- Development workflow + phase closeout gate: `docs/development_workflow.md`
- Developer guide: `developer_guide.md`
- Repository contract: `AGENTS.md`

## Contributing

- Contributor guide: `CONTRIBUTE.md`
- Working contract: `AGENTS.md`

## License

MIT (see `LICENSE`).
