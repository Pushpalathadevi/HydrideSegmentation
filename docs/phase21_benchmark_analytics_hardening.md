# Phase 21 - Benchmark Analytics Hardening

## Goals

- Expand hydride benchmark summaries beyond final eval metrics to include training dynamics and runtime/resource footprint.
- Provide objective comparison artifacts suitable for manuscript-grade model selection.
- Keep outputs machine-readable and human-readable from a single suite run.

## Implemented

1. Enriched suite analytics in `scripts/hydride_benchmark_suite.py`:
- ingest per-run `report.json` training history
- compute and persist:
  - training runtime, eval runtime, total runtime
  - model artifact size (bytes/MB)
  - parameter count (from torch checkpoint `model_state_dict`, when available)
  - final-epoch training/validation loss, accuracy, IoU snapshots
- aggregate model-level statistics (mean/std across seeds)

2. Training-curve generation:
- per-run curve PNGs for:
  - loss vs epoch
  - accuracy vs epoch
  - IoU vs epoch
- links consolidated in per-run `inside.html` pages generated under `runs/<run_tag>/inside.html`
- top-level `summary.html` remains concise and routes heavy visual artifacts through links (no eager image embedding)

3. UNet/transformer trainer history enrichment:
- `src/microseg/training/unet_binary.py` now records:
  - `train_accuracy`
  - `val_accuracy`
- HTML training report table updated accordingly.

4. Docs updates:
- `docs/hydride_research_workflow.md`
- `scripts/README.md`
- `README.md`

5. Test coverage:
- updated: `tests/test_phase7_training_reporting.py`
- added: `tests/test_phase21_benchmark_dashboard_enrichment.py`

## Validation

```bash
PYTHONPATH=. pytest -q tests/test_phase7_training_reporting.py tests/test_phase20_benchmark_suite_script.py tests/test_phase21_benchmark_dashboard_enrichment.py
```

## Remaining Gaps

- no built-in statistical significance testing yet (paired tests / confidence intervals)
- dashboard remains static HTML (not interactive plotting)
- hardware energy/consumption accounting is not yet integrated
