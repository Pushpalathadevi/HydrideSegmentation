# Phase 8 - Phase Gate And Quality Governance Automation

## Objective

Make end-of-phase closure operational and auditable by enforcing:
- full test pass checks
- stocktaking of progress
- explicit gap capture
- synchronized documentation updates

## Implemented

1. Phase-gate module
- Added `src/microseg/quality/phase_gate.py` with schema `microseg.phase_gate.v1`.
- Checks include:
  - full test execution (optional but enabled by default)
  - absolute markdown path reference scan
  - required governance docs presence

2. Closeout artifacts
- Machine-readable report:
  - `outputs/phase_gates/<phase>_phase_gate.json`
- Human-readable stocktake:
  - `outputs/phase_gates/<phase>_stocktake.md`

3. CLI and script integration
- Added `microseg-cli phase-gate`
- Added `python scripts/run_phase_gate.py`
- Added config template: `configs/phase_gate.default.yml`

4. Policy and workflow synchronization
- Updated `AGENTS.md` with mandatory end-of-phase closeout section.
- Updated `docs/development_workflow.md` with concrete closeout procedure.
- Updated roadmap with global phase-close rule and Phase 8 status.

## Example Usage

```bash
microseg-cli phase-gate --phase-label "Phase 8" --strict
```

or

```bash
python scripts/run_phase_gate.py --phase-label "Phase 8" --strict
```

## Test Coverage

- `tests/test_phase8_phase_gate.py`
