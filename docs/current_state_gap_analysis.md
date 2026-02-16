# Current State Gap Analysis (v0.7.0)

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

## Remaining Gaps To World-Class Target

High-priority gaps:
- Advanced model-training pipelines beyond baseline pixel classifier
- Rich evaluation visualization/reporting in GUI
- Advanced augmentation workflow authoring and preview in GUI
- Hardware profile capture beyond current run metadata basics
- Formal uncertainty quantification pathways in inference outputs

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
