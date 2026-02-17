# Phase 12 - GUI Dataset Workspace And QA Gate (Implemented)

## Objective

Add a user-facing dataset onboarding workspace inside the Qt app to support:
- dataset structure preview before running jobs
- leakage-aware split controls and RGB colormap conversion options
- dataset QA execution and training launch gating
- reusable YAML workflow profiles for dataset-prepare, training, and evaluation panes

## Implemented

1. Dataset Prep + QA GUI tab
- Added a dedicated Workflow Hub tab in:
  - `hydride_segmentation/qt/main_window.py`
- Includes controls for:
  - split ratios, seed, ID width
  - split strategy and leakage grouping mode/regex
  - mask input type and colormap JSON
  - strict colormap handling
  - QA report path, imbalance threshold, strict mode

2. Preview-first onboarding
- Added preview API:
  - `src/microseg/dataops/training_dataset.py`
  - `preview_training_dataset_layout(...)`
- GUI now shows:
  - source layout classification
  - planned split counts
  - leakage group count
  - class histogram
  - searchable mapping table (`global_id`, split, source group, names, paths)

3. Training QA gate
- Training tab now includes `Require dataset QA pass before launch`.
- When enabled:
  - dataset preflight prepare runs (if needed)
  - dataset QA runs
  - training launch is blocked on QA-critical failures

4. Workflow profiles
- Added save/load profile support in Workflow Hub for:
  - `dataset_prepare`
  - `training`
  - `evaluation`
- Profiles are YAML files with schema:
  - `microseg.workflow_profile.v1`

5. Orchestration and tests
- Extended command builder with:
  - `dataset_prepare(...)`
  - `dataset_qa(...)`
- Added/extended tests:
  - `tests/test_phase10_dataset_prepare.py` (preview coverage)
  - `tests/test_phase4_orchestration.py` (new command builders)

## Exit Criteria

- GUI supports dataset onboarding and QA gating without requiring manual CLI steps.
- Preview + QA workflow is documented, test-covered, and integrated with phase closeout standards.
