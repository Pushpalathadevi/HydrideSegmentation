# Versioning and Release Policy

## Current Version

- Software version: `0.22.0`
- Version source of truth:
  - `hydride_segmentation/version.py`
  - `pyproject.toml`
  - `setup.py`

## Semantic Versioning Rules

Use `MAJOR.MINOR.PATCH`.

- `MAJOR`: breaking API or workflow changes for users/integrators.
- `MINOR`: backward-compatible feature additions (new tools, GUI enhancements, exporters).
- `PATCH`: bug fixes and internal quality improvements without changing expected behavior.

## Release Requirements

Before release:

1. Tests
- `pytest -q` must pass on CPU-only environment.
- `microseg-cli phase-gate --phase-label "Release Gate" --strict` must pass.

2. Documentation synchronization
- Update `README.md` and relevant `docs/*.md` files.
- Ensure migration/roadmap status reflects actual code state.

3. Change summary
- Maintain release notes with:
  - new features
  - bug fixes
  - migration notes
  - known limitations

4. Reproducibility checks
- Validate correction schema compatibility (`microseg.correction.v1`)
- Validate project schema compatibility (`microseg.project.v1`)
- Validate dataset packaging manifest generation
- Validate resolved configuration artifacts for CLI workflows
- Validate training report artifacts (`microseg.training_report.v1`, `microseg.training_manifest.v2`)
- Validate evaluation report artifacts (`microseg.pixel_eval.v4`) and optional HTML summary output
- Validate frozen model metadata (`microseg-cli validate-registry --config configs/registry_validation.default.yml --strict`)
- Keep checkpoint binaries out of git; track only metadata and promotion evidence paths

## Deployment Guidance

Field deployments should pin explicit versions (for example `hydride-segmentation==0.22.0`) and avoid floating upgrades.

## Patch And Rollback Protocol

1. Patch release workflow
- Branch from the latest promoted tag.
- Keep scope constrained to the defect and required regression tests.
- Run:
  - `PYTHONPATH=. pytest -q`
  - `microseg-cli phase-gate --config configs/phase_gate.default.yml --set phase_label=\"Release Gate\" --strict`
- Publish release notes with root cause, fix summary, and validation evidence paths.

2. Rollback workflow
- Keep the last known-good deployment package manifest and checksum report.
- Validate candidate package before activation:
  - `microseg-cli deploy-validate --package-dir <pkg_dir> --strict`
- If production incident occurs, rollback by re-activating the prior known-good package directory and recording:
  - previous package id
  - restored package id
  - incident timestamp and operator
- After rollback, generate a support artifact bundle:
  - `microseg-cli support-bundle --run-root <run_root> --output-dir outputs/support_bundles`
