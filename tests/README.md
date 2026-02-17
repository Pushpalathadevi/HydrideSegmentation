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
- `test_phase6_unet_training.py` UNet training backend, checkpoint resume, and evaluation path
- `test_phase7_frozen_registry.py` frozen checkpoint metadata registry and GUI integration
- `test_phase7_training_reporting.py` UNet reporting outputs and val tracking artifact generation
- `test_phase8_phase_gate.py` phase closeout gate checks and stocktake artifact generation
- `test_phase9_registry_validation.py` strict frozen model registry validation rules
- `test_phase9_dataops.py` leakage-aware split planner and dataset QA checks
- `test_phase10_dataset_prepare.py` unsplit source/masks auto-prepare with leakage-aware grouping, global IDs, optional RGB-colormap conversion, and preview contract checks
- `test_phase13_report_review.py` run-report summary/compare and workflow profile persistence roundtrips
- `test_phase14_checkpoint_lifecycle.py` checkpoint lifecycle metadata validation and registry-hint checkpoint resolution
- `test_phase15_hpc_ga_planner.py` GA candidate planning, HPC script-bundle generation, and HPC workflow-profile scope roundtrips
