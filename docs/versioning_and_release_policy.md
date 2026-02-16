# Versioning and Release Policy

## Current Version

- Software version: `0.6.0`
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

## Deployment Guidance

Field deployments should pin explicit versions (for example `hydride-segmentation==0.6.0`) and avoid floating upgrades.
