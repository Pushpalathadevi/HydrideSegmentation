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
  - deterministic execution ordering by model family:
    - transformer first
    - deeplab next
    - advanced non-unet models next
    - `unet_binary` last
  - single-seed CLI override (`--single-seed`) to run only the first configured seed
  - default resilient continuation on per-run failures (`continue_on_failure=true`)
  - strict mode fail handling
  - dry-run planning mode
  - live per-run log streaming to `logs/<run_tag>/train.log|eval.log`
  - optional watchdog timeouts (`command_idle_timeout_seconds`, `command_wall_timeout_seconds`)
  - consolidated JSON/CSV summary outputs
  - concise HTML summary generation + per-run `inside.html` detail pages with on-demand links to curves/panels/logs

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
