# Phase 20 - Hydride Top-5 Benchmark Suite Orchestration

## Goals

- Provide a single script to run top-5 hydride benchmark experiments end-to-end.
- Emit consolidated run-level and aggregate reports for manuscript analysis.
- Keep model/data/metric comparison contracts explicit and reproducible.

## Implemented

1. New orchestration script:
- `scripts/hydride_benchmark_suite.py`
- supports:
  - train + eval execution loops across models and seeds
  - strict mode fail handling
  - dry-run planning mode
  - consolidated JSON/CSV summary outputs
  - HTML dashboard generation

2. Top-5 suite config:
- `configs/hydride/benchmark_suite.top5.yml`
- includes:
  - fixed model list
  - fixed split dataset path
  - seed set
  - train/eval config paths

3. Workflow and benchmark guidance updates:
- `docs/hydride_research_workflow.md`
- `configs/hydride/README.md`

4. Test coverage:
- `tests/test_phase20_benchmark_suite_script.py`

## Validation

```bash
PYTHONPATH=. pytest -q
microseg-cli phase-gate --phase-label "Phase 20 Benchmark Suite Orchestration" --strict
```
