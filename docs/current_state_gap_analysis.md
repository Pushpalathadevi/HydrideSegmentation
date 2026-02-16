# Current State Gap Analysis (v0.14.0)

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
- Frozen checkpoint metadata registry with GUI/CLI model guidance
- Training `report.json` + HTML summaries with fixed/random val sample tracking
- Evaluation JSON + HTML reports with tracked sample panels
- Phase-gate automation for end-of-phase test pass + stocktake + gap + docs checks
- Strict frozen registry validation and leakage-aware dataset split/QA tooling
- Training dataset auto-prepare from unsplit source/masks with deterministic 80:10:10 default split
- Leakage-aware auto-prepare grouping policies with optional RGB-colormap mask conversion and global IDs

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
- Installer and update channels for field distribution

## Risks

- Current ML model wrappers still depend on legacy paths; modernization is in progress.
- Some legacy modules still use older style validation/logging patterns.
- GUI-heavy paths require expanded e2e coverage over time.

## Mitigation Strategy

- Keep compatibility adapters while introducing contract-first replacements.
- Continue phase-based migration with regression snapshots.
- Enforce doc+test updates in each behavior-changing change.
