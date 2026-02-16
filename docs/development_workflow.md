# Development Workflow Procedure

## Working Procedure for New Features

1. Define the scientific and user objective.
2. Check alignment with mission and architecture docs.
3. Add or update ADR if a major design choice is involved.
4. Implement smallest vertical slice with tests.
5. Add regression tests for preserved behavior.
6. Update docs in the same change.
7. Validate CPU-only local run path.
8. Verify YAML + `--set` configuration paths for changed workflows.
9. Verify project/session persistence if correction UI behavior changed.
10. Run end-of-phase closeout checks before declaring phase completion.

## Mandatory End-Of-Phase Closeout

1. Run full tests:
- `PYTHONPATH=. pytest -q`

2. Run phase gate check:
- `microseg-cli phase-gate --phase-label "Phase N" --strict`
or
- `python scripts/run_phase_gate.py --phase-label "Phase N" --strict`

3. Produce stocktake and gap review:
- confirm implemented vs planned deliverables
- list remaining gaps (if any) in `docs/current_state_gap_analysis.md`
- add/update phase status doc under `docs/`

4. Documentation synchronization:
- update `README.md`, `docs/development_roadmap.md`, and impacted user/developer docs
- ensure markdown links are repository-relative

## Pull Request Checklist

- Scope clearly stated
- Tests added or updated
- Backward compatibility addressed
- Documentation synchronized
- Reproducibility metadata preserved
- Config schema or defaults updated (if behavior changed)

## Release Readiness Checklist

- Core workflows verified on reference sample images
- Local app install/run validated on clean environment
- Known limitations listed in release notes
- Latest phase closeout artifacts generated under `outputs/phase_gates/`
