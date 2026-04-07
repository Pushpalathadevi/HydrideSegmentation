# Failure Taxonomy And Error Codes

This project now emits machine-readable failure codes for deployment/runtime-preflight diagnostics.

Source of truth:
- `src/microseg/quality/failure_codes.py`

## Primary Codes

| Code | Category | Meaning |
|---|---|---|
| `MICROSEG_INPUT_INVALID` | input | Invalid user/config input payload |
| `MICROSEG_INPUT_NOT_FOUND` | input | Required path or artifact is missing |
| `MICROSEG_IO_WRITE_FAILED` | io | Failed to write output artifact |
| `MICROSEG_PREFLIGHT_DATASET_INVALID` | preflight | Dataset contract or QA preflight failed |
| `MICROSEG_PREFLIGHT_PRETRAINED_MISSING` | preflight | Pretrained artifact preflight failed |
| `MICROSEG_PREFLIGHT_MODEL_MISSING` | preflight | Model path missing in preflight |
| `MICROSEG_PREFLIGHT_BENCHMARK_CONFIG_INVALID` | preflight | Benchmark config parsing/contract failed |
| `MICROSEG_DEPLOY_PACKAGE_INVALID` | deployment | Deployment manifest/checksum contract failed |
| `MICROSEG_DEPLOY_MODEL_RESOLVE_FAILED` | deployment | Model path could not be resolved from package |
| `MICROSEG_DEPLOY_MODEL_LOAD_FAILED` | deployment | Model loading failed before inference |
| `MICROSEG_DEPLOY_PREPROCESS_FAILED` | deployment | Input preprocessing failed |
| `MICROSEG_DEPLOY_INFERENCE_FAILED` | deployment | Inference execution failed |
| `MICROSEG_DEPLOY_OUTPUT_WRITE_FAILED` | deployment | Runtime output write failed |
| `MICROSEG_SERVICE_QUEUE_FULL` | service | Worker queue capacity exceeded |
| `MICROSEG_SERVICE_JOB_NOT_FOUND` | service | Requested job id does not exist |
| `MICROSEG_SERVICE_JOB_TIMEOUT` | service | Wait timeout while job still running |
| `MICROSEG_UNKNOWN_INTERNAL` | internal | Unclassified internal error |

## Where Codes Are Emitted

1. `microseg-cli preflight`
- each issue entry includes semantic `code` and canonical `error_code`.

2. `microseg-cli deploy-health`
- global and per-image steps include `error_code` fields in `runtime_health_report.json`.

3. `microseg-cli deploy-worker-run`
- per-job records include `error_code` for rejection/failure/timeout paths.

## Operational Guidance

- Parse `error_code` first for automation/alerting.
- Use human-readable `message` for incident triage.
- Keep alert rules stable at code level; avoid brittle message matching.
