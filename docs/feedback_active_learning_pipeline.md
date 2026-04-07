# Feedback Active-Learning Pipeline

## Purpose

Define the low-friction, deployment-safe feedback loop that captures per-inference evidence,
collects bundles from field deployments, ingests/validates centrally, builds weighted datasets,
and triggers retraining while keeping promotion human-gated.

## Per-Inference Record Contract

Schema: `microseg.feedback_record.v1`

Canonical folder:

```text
<feedback_root>/<deployment_id>/<YYYY>/<MM>/<DD>/<record_id>/
  input.png
  predicted_mask_indexed.png
  predicted_overlay.png
  corrected_mask_indexed.png                 # optional
  analysis/<name>.png                        # optional
  inference_manifest.json
  resolved_config.json
  feedback_record.json
  feedback_events.jsonl
  artifacts_manifest.json
```

`feedback_record.json` stores:
- record/run IDs
- deployment + operator ID
- source image path + SHA256
- model ID/name + artifact hint
- runtime info (`enable_gpu`, `device_policy`, worker controls when relevant)
- inference manifest and resolved config hash
- feedback rating (`unrated|thumbs_up|thumbs_down`) + optional comment
- correction linkage (corrected mask flag/path, optional correction export linkage)
- artifact path map + artifact manifest hash

## Capture Paths

- Desktop GUI (`hydride-gui`)
  - creates record immediately after inference success
  - `👍`/`👎` + optional comment auto-update same record
  - corrected masks are attached to same record when edits differ from prediction
- CLI inference (`microseg-cli infer`)
  - creates same record schema (`source=cli_infer`)
- Deployment worker (`microseg-cli deploy-worker-run`)
  - completed jobs emit same record schema (`source=service_worker`)

## Bundle Export (Deployment Side)

Command:

```bash
microseg-cli feedback-bundle --config configs/feedback_bundle.default.yml
```

Behavior:
- selects unsent records (state cursor in `.bundle_state.json` or configured `state_path`)
- default cap: 200 records per bundle
- writes zip + `feedback_bundle_manifest.json` (`microseg.feedback_bundle.v1`)
- recommended cadence: weekly or when 200 new records accumulated

## Central Ingest + Review Queue

Command:

```bash
microseg-cli feedback-ingest \
  --config configs/feedback_ingest.default.yml \
  --bundle-path <bundle.zip> \
  --strict
```

Behavior:
- validates bundle schema + per-record artifact checksums
- deduplicates by `record_id` + record/artifact hashes
- copies accepted records to central lake (`ingest_root`)
- writes ingest report (`microseg.feedback_ingest_report.v1`)
- appends thumbs-down-without-correction records to review queue JSONL

## Weighted Dataset Build

Command:

```bash
microseg-cli feedback-build-dataset --config configs/feedback_build_dataset.default.yml
```

Label policy:
- corrected masks: supervised label, weight `1.0`
- thumbs-up without correction: pseudo-label, weight `0.2`
- thumbs-down without correction: excluded from training dataset

Outputs:
- split dataset (`train/val/test` with `images,masks,metadata`)
- `sample_weights.csv`
- `dataset_manifest.json` (`microseg.feedback_dataset_manifest.v1`)

Split policy is deterministic and leakage-aware by source stem (configurable).

## Retrain Trigger

Command:

```bash
microseg-cli feedback-train-trigger --config configs/feedback_train_trigger.default.yml
```

Trigger rule defaults:
- corrected records since last trigger >= `500`
- OR days since last trigger >= `14`

When triggered:
- builds feedback dataset
- prepares train/evaluate commands
- optional execution via `execute=true` / `--execute`
- writes trigger report (`microseg.feedback_train_trigger_report.v1`)

Promotion remains human-gated; this command does not perform automatic model promotion.

## Recommended Rollout

1. Enable capture in pilot deployments (2-3 sites).
2. Validate bundle quality and ingest dedup behavior.
3. Build first weighted datasets and compare model deltas in run review.
4. Keep promotion policy strict until feedback volume and label quality stabilize.
