# Phase 26 Deployment Runtime Modes

## Scope

Phase 26 extends deployment operations with:

1. Queue-safe service-mode worker execution.
2. Canary/shadow comparison between baseline and candidate deployment packages.
3. Deployment latency/throughput benchmark harness.
4. Expanded machine-readable failure taxonomy for service operations.

## Implemented Commands

- `microseg-cli deploy-worker-run`
- `microseg-cli deploy-canary-shadow`
- `microseg-cli deploy-perf`

## Core Modules

- `src/microseg/deployment/service_worker.py`
- `src/microseg/deployment/canary_shadow.py`
- `src/microseg/deployment/perf_benchmark.py`
- `src/microseg/quality/failure_codes.py` (service codes)

## Queue-Safe Service Worker

`deploy-worker-run` provides bounded queue controls:

- `max_workers`
- `max_queue_size`
- optional timeout handling per wait path

Outputs:

- `service_batch_report.json` with per-job status and failure codes.

## Canary/Shadow Comparison

`deploy-canary-shadow` computes per-image:

- disagreement fraction
- baseline/candidate foreground fraction
- optional GT-based IoU/Dice for both packages and candidate gain

Outputs:

- `canary_shadow_report.json`
- per-image `*_diff.png` disagreement masks

## Perf Harness

`deploy-perf` reports:

- throughput (images/sec)
- latency mean + p50/p90/p95/p99
- per-request CSV rows

Outputs:

- `deployment_perf_report.json`
- `deployment_perf_report.csv`

## Validation

Added test module:

- `tests/test_phase26_deployment_runtime_modes.py`

Covers:

1. queue capacity rejection (`MICROSEG_SERVICE_QUEUE_FULL`)
2. successful service batch completion
3. canary-shadow positive gain path
4. perf harness report generation and metrics fields
