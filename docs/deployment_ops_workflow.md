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
