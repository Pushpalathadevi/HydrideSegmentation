# Current State Gap Analysis (v0.22.0)

## Baseline Status

Implemented:
- Registry-backed inference orchestration
- Qt desktop correction GUI with brush/polygon/lasso tools
- Connected-feature delete/relabel workflow (`feature_select` tool)
- Class index/color map editing
- Correction export schema (`microseg.correction.v1`) with indexed/color/npy formats
- Dataset packaging with deterministic train/val/test split
- Project/session save-load (`microseg.project.v1`)
- YAML config loading + `--set` override support
- GUI orchestration pane for inference/training/evaluation/packaging
- Baseline CPU training and evaluation pipelines (pixel classifier)
- Torch-based baseline training/evaluation with opt-in GPU runtime and CPU fallback
- UNet binary training backend with checkpoint/resume lifecycle controls
- Transformer segmentation backends for trials (`hf_segformer_b0/b2/b5` scratch-init, `transunet_tiny`, `segformer_mini`)
- Local pretrained transfer-learning path for `unet_binary`, `smp_unet_resnet18`, HF SegFormer (`hf_segformer_b0/b2/b5`), `transunet_tiny`, and `segformer_mini` via air-gap-friendly `pre_trained_weights/` bundles
- Frozen checkpoint metadata registry with GUI/CLI model guidance
- Tiny smoke-checkpoint generation for local pipeline sanity checks without large model binaries
- Frozen-checkpoint lifecycle folders (`smoke`, `candidates`, `promoted`) with metadata-only tracking
- Training `report.json` + HTML summaries with fixed/random val sample tracking
- Evaluation JSON + HTML reports with tracked sample panels
- Phase-gate automation for end-of-phase test pass + stocktake + gap + docs checks
- Strict frozen registry validation and leakage-aware dataset split/QA tooling
- Training dataset auto-prepare from unsplit source/masks with deterministic 80:10:10 default split
- Leakage-aware auto-prepare grouping policies with optional RGB-colormap mask conversion and global IDs
- GUI Dataset Prep + QA workspace with searchable preview table and optional training QA gate
- GUI Run Review workspace for training/evaluation report summaries and metric-delta comparison
- GA-based HPC bundle generation (Slurm/PBS/local) from GUI/CLI for architecture/hyperparameter comparison sweeps
- Feedback-aware HPC GA planning mode with kNN fitness estimation and report summarization (`hpc-ga-feedback-report`)
- HPC GA pretrained-init hardening (`scratch/auto/local` modes, backend-to-model mapping, and explicit backend/model_architecture script overrides)
- Single-script top-5 hydride benchmark orchestration with consolidated JSON/CSV/HTML summary outputs
- Single dashboard benchmark analytics now includes tracked validation sample IoU summaries and panel galleries
- Results Dashboard in Qt GUI with adjustable plotting controls and predicted/corrected distribution panels
- Full desktop results-package export with JSON + HTML + PDF reports
- Conventional-model parameter controls exposed directly in Qt GUI
- Bundled sample image onboarding (`Load Sample` + `File -> Open Sample`)
- Persistent desktop operational logs in `outputs/logs/desktop/`
- Windows packaging assets for offline installer build (PyInstaller spec + Inno Setup script + build script)
- Sphinx-based documentation source tree with HTML/PDF build helper, SVG diagrams, and math-aware narrative docs

## Remaining Gaps To World-Class Target

High-priority gaps:
- Advanced model-training pipelines beyond baseline pixel classifier
- Advanced augmentation workflow authoring and preview in GUI
- Hardware profile capture beyond current run metadata basics
- Formal uncertainty quantification pathways in inference outputs
- Comprehensive GUI-native visualization for training/evaluation reports

Medium-priority gaps:
- Multi-feature default registries beyond hydrides
- Rich experiment tracking backends
- Signed installer and update channels for field distribution
- More GUI schematic screenshots and captured visual QA artifacts for future release notes

## Risks

- Current ML model wrappers still depend on legacy paths; modernization is in progress.
- Some legacy modules still use older style validation/logging patterns.
- GUI-heavy paths require expanded e2e coverage over time.

## Mitigation Strategy

- Keep compatibility adapters while introducing contract-first replacements.
- Continue phase-based migration with regression snapshots.
- Enforce doc+test updates in each behavior-changing change.

## Latest Health Audit Snapshot

- Full tests pass (`76 passed`).
- Strict phase gate pass confirmed (`Airgap Pretrained HPC Hardening`).
- API validation warnings removed (Pydantic v2 validators and image-type probing modernization).
- Detailed audit record: `repo_health_audit.md`.

## 2026-02 HPC-readiness hardening update

Addressed gaps:
- Reduced core MicroSeg coupling to legacy package for version/encoding/analysis pathways.
- Added profile-based dependency strategy and pinned lock baseline for reproducibility.
- Added AMP, gradient accumulation, deterministic mode, and dataloader runtime knobs to UNet training.
- Expanded evaluation outputs with hydride-scientific distribution/error metrics.
- Added benchmark-mode hard-fail checks for dataset manifest freeze and split-ID consistency.

Remaining intentional gap:
- Native MicroSeg inference adapters still include compatibility imports from `hydride_segmentation` for conventional/legacy ML paths.
