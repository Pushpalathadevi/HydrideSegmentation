# Deployment Operations Workflow

This runbook standardizes benchmark-to-deployment handoff and post-run diagnostics.

## 1. Preflight Checks

Train/eval/benchmark/deploy modes share one preflight contract.

```bash
microseg-cli preflight --config configs/preflight.default.yml --mode train --strict
microseg-cli preflight --config configs/preflight.default.yml --mode benchmark --benchmark-config configs/hydride/benchmark_suite.top5.yml --strict
```

## 2. Build Deployment Package

```bash
microseg-cli deploy-package \
  --config configs/deployment_package.default.yml \
  --model-path outputs/training/model.pth
```

Package output includes:
- `deployment_manifest.json` (schema + runtime contract)
- copied model artifact and optional config/report files
- per-file checksums (`sha256`)

## 3. Validate Package Integrity

```bash
microseg-cli deploy-validate --package-dir outputs/deployments/<package_dir> --strict
```

## 4. Smoke Inference Before Rollout

```bash
microseg-cli deploy-smoke \
  --package-dir outputs/deployments/<package_dir> \
  --image-path test_data/sample.png \
  --output-dir outputs/deployments/smoke
```

Smoke output includes predicted mask, overlay PNG, and smoke JSON report.

## 4b. Runtime Health And Queue-Style Batch Validation

```bash
microseg-cli deploy-health \
  --config configs/deploy_health.default.yml \
  --package-dir outputs/deployments/<package_dir> \
  --image-dir test_data \
  --max-workers 4 --strict
```

Runtime health report contains global and per-image step checks with machine-readable error codes for:
- package validation
- model load
- preprocess
- inference
- output write

## 4c. Service-Mode Worker Batch (Queue-Safe Controls)

```bash
microseg-cli deploy-worker-run \
  --config configs/deploy_worker.default.yml \
  --package-dir outputs/deployments/<package_dir> \
  --image-dir test_data \
  --max-workers 4 \
  --max-queue-size 64 \
  --strict
```

This command exercises bounded queue behavior and produces `service_batch_report.json`.
When `capture_feedback=true` (default), each completed inference also writes a per-inference
feedback evidence folder under `feedback_root` (`microseg.feedback_record.v1`).

## 4d. Canary/Shadow Comparison

```bash
microseg-cli deploy-canary-shadow \
  --config configs/deploy_canary_shadow.default.yml \
  --baseline-package-dir outputs/deployments/<baseline_pkg> \
  --candidate-package-dir outputs/deployments/<candidate_pkg> \
  --image-dir test_data \
  --mask-dir test_data/masks \
  --strict
```

Output includes per-image disagreement masks and aggregate disagreement/gain metrics.

## 4e. Performance Harness

```bash
microseg-cli deploy-perf \
  --config configs/deploy_perf.default.yml \
  --package-dir outputs/deployments/<package_dir> \
  --image-dir test_data \
  --warmup-runs 1 \
  --repeat 3 \
  --max-workers 4 \
  --strict
```

Reports include throughput and latency (`mean`, `p50`, `p90`, `p95`, `p99`) plus per-request CSV rows.

## 4f. Feedback Bundle Export + Central Ingest

Deployment-side bundle export (recommended cadence: weekly or 200 records):

```bash
microseg-cli feedback-bundle --config configs/feedback_bundle.default.yml
```

Central ingest with checksum validation + dedup:

```bash
microseg-cli feedback-ingest \
  --config configs/feedback_ingest.default.yml \
  --bundle-path outputs/feedback_bundles/<bundle>.zip \
  --strict
```

Ingest writes:
- centralized record copy under `ingest_root`
- ingest report JSON (`accepted`, `duplicates`, `rejected`)
- review queue JSONL for thumbs-down records without corrected masks

## 5. Promotion Gate (Benchmark Evidence)

```bash
microseg-cli promote-model \
  --summary-json outputs/hydride_benchmark_suite/summary.json \
  --model-name hf_segformer_b2 \
  --registry-model-id hf_segformer_b2 \
  --policy-config configs/promotion_policy.default.yml \
  --update-registry --strict
```

This writes a decision JSON/markdown report and optionally updates registry stage metadata.

## 6. Support Diagnostics Bundle

After long benchmark jobs or incidents, collect one transferable bundle.

```bash
microseg-cli support-bundle --config configs/support_bundle.default.yml
microseg-cli compatibility-matrix --output-path outputs/support_bundles/compatibility_matrix.json
```

Support bundle output includes key run logs/reports and an environment fingerprint.

## 7. Phase Closeout Enforcement

```bash
microseg-cli phase-gate --config configs/phase_gate.default.yml --set phase_label="Deployment Gate" --strict
```

Default phase-gate template now supports:
- release-policy verification (`docs/versioning_and_release_policy.md`)
- required rollback keyword check
- optional deployment package validation list
