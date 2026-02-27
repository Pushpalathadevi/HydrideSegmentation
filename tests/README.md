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
- `test_input_size_policy.py` mixed-size DataLoader failure reproduction, fixed-size policy transforms, letterbox/padding semantics, and pad-to-max collate fallback
- `test_phase7_frozen_registry.py` frozen checkpoint metadata registry and GUI integration
- `test_phase7_training_reporting.py` UNet reporting outputs and val tracking artifact generation
- `test_phase8_phase_gate.py` phase closeout gate checks and stocktake artifact generation
- `test_phase9_registry_validation.py` strict frozen model registry validation rules
- `test_phase9_dataops.py` leakage-aware split planner and dataset QA checks
- `test_phase10_dataset_prepare.py` unsplit source/masks auto-prepare with leakage-aware grouping, global IDs, optional RGB-colormap conversion, and preview contract checks
- `test_phase13_report_review.py` run-report summary/compare and workflow profile persistence roundtrips
- `test_phase14_checkpoint_lifecycle.py` checkpoint lifecycle metadata validation and registry-hint checkpoint resolution
- `test_phase15_hpc_ga_planner.py` GA candidate planning, feedback-hybrid ranking inputs, HPC script-bundle generation, pretrained mapping/script overrides, and HPC workflow-profile scope roundtrips
- `test_phase18_transformer_backends.py` transformer segmentation backend smoke training/evaluation (`transunet_tiny`, `segformer_mini`)
- `test_phase19_hf_transformer_backends.py` Hugging Face SegFormer transformer backend smoke training/evaluation from scratch (no pretrained weights)
- `test_phase20_benchmark_suite_script.py` hydride benchmark suite single-script dry-run orchestration + summary artifact generation
- `test_phase21_benchmark_dashboard_enrichment.py` benchmark dashboard enrichment with training curves, training-history metrics, and aggregate comparison fields
- `test_phase22_pretrained_offline.py` offline local pretrained registry validation, checksum verification, and local-bundle initialization smoke tests for `unet_binary`, `smp_unet_resnet18`, `hf_segformer_b0`, `transunet_tiny`, and `segformer_mini`

- `test_data_preparation_module.py` paired image/mask collector, RGB red-threshold mask binarization, short-side resize+crop alignment, manifests, QA reports, and dry-run behavior
- `test_mask_binary_normalization.py` binary-mask auto-normalization option (`two_value_zero_background`) for 2-value indexed masks
- `test_service.py` legacy Flask service request validation for model selection, parameter parsing, and response behavior
