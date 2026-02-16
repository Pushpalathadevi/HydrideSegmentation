# Testing Strategy (Target)

Target test groups:
- `unit/` pure module behavior
- `integration/` multi-module and API behavior
- `e2e/` desktop and CLI workflows

Legacy tests at repository root will be migrated incrementally.

Current phase coverage includes:
- `test_phase0_regression.py` baseline non-regression snapshots
- `test_phase1_microseg_core.py` registry/pipeline compatibility
- `test_phase2_desktop_workflow.py` run history and export packaging
- `test_phase3_corrections.py` correction session and dataset packaging
- `test_phase3_annotation_view.py` layered correction visualization composition
- `test_phase4_foundation_features.py` class maps, feature delete/relabel, config overrides, project save/load
- `test_phase4_orchestration.py` command builder + baseline training/evaluation orchestration
- `test_phase5_gpu_runtime.py` GPU-compatible training/evaluation with CPU fallback behavior
