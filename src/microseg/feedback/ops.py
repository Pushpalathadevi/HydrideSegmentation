"""Bundle export, ingest, dataset build, and trigger operations for feedback loops."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import tempfile
from typing import Any
from zipfile import ZipFile

from .contracts import (
    FEEDBACK_BUNDLE_RESULT_SCHEMA,
    FEEDBACK_BUNDLE_SCHEMA,
    FEEDBACK_DATASET_MANIFEST_SCHEMA,
    FEEDBACK_INGEST_REPORT_SCHEMA,
    FEEDBACK_RECORD_SCHEMA,
    FEEDBACK_TRIGGER_REPORT_SCHEMA,
    FeedbackBundleConfig,
    FeedbackBundleManifest,
    FeedbackBundleResult,
    FeedbackDatasetBuildConfig,
    FeedbackDatasetBuildResult,
    FeedbackIngestConfig,
    FeedbackIngestReport,
    FeedbackTrainTriggerConfig,
    FeedbackTrainTriggerReport,
)
from .writer import discover_feedback_record_dirs, load_feedback_record, sha256_file, utc_now


def _safe_name(text: str, *, fallback: str = "item") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _find_manifest(path: Path, name: str) -> Path | None:
    if path.is_file() and path.name == name:
        return path
    if path.is_dir():
        direct = path / name
        if direct.exists():
            return direct
        matches = list(path.rglob(name))
        if matches:
            return matches[0]
    return None


def _extract_bundle_root(path: Path) -> tuple[Path, tempfile.TemporaryDirectory[str] | None]:
    if path.is_dir():
        return path, None
    if path.is_file() and path.suffix.lower() == ".zip":
        tmp = tempfile.TemporaryDirectory(prefix="microseg_feedback_bundle_")
        with ZipFile(path, "r") as zf:
            zf.extractall(tmp.name)
        return Path(tmp.name), tmp
    raise ValueError(f"unsupported bundle path: {path}")


def _parse_timestamp(text: object) -> datetime | None:
    raw = str(text or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _planned_counts(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    if n <= 0:
        return 0, 0
    n_train = min(n, int(round(n * train_ratio)))
    n_val = min(max(0, n - n_train), int(round(n * val_ratio)))
    return n_train, n_val


def _assign_groups(
    grouped: dict[str, list[dict[str, Any]]],
    *,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> dict[str, str]:
    groups = list(grouped.keys())
    rng = random.Random(seed)
    rng.shuffle(groups)
    groups.sort(key=lambda g: len(grouped[g]), reverse=True)
    sample_count = sum(len(v) for v in grouped.values())

    target_train, target_val = _planned_counts(sample_count, train_ratio, val_ratio)
    target_test = sample_count - target_train - target_val
    targets = {"train": target_train, "val": target_val, "test": target_test}
    current = {"train": 0, "val": 0, "test": 0}

    assignments: dict[str, str] = {}
    for group in groups:
        size = len(grouped[group])
        deficits = {split: targets[split] - current[split] for split in ("train", "val", "test")}
        best = max(deficits.items(), key=lambda kv: (kv[1], kv[0]))[0]
        if deficits[best] < 0:
            best = min(current.items(), key=lambda kv: kv[1])[0]
        assignments[group] = best
        current[best] += size
    return assignments


def export_feedback_bundle(config: FeedbackBundleConfig) -> FeedbackBundleResult:
    """Bundle unsent local feedback records for centralized ingest."""

    root = Path(config.feedback_root).resolve()
    out_root = Path(config.output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    state_path = Path(config.state_path).resolve() if str(config.state_path).strip() else (root / ".bundle_state.json")
    state = _read_json(state_path)
    sent_ids = {str(v) for v in state.get("sent_record_ids", []) if str(v).strip()}

    candidates: list[tuple[datetime, str, Path, dict[str, Any]]] = []
    for record_dir in discover_feedback_record_dirs(root):
        try:
            payload = load_feedback_record(record_dir)
        except Exception:
            continue
        record_id = str(payload.get("record_id", record_dir.name)).strip()
        if not record_id or record_id in sent_ids:
            continue
        created = _parse_timestamp(payload.get("created_utc")) or datetime.fromtimestamp(0, tz=timezone.utc)
        candidates.append((created, record_id, record_dir, payload))

    candidates.sort(key=lambda row: (row[0], row[1]))
    selected = candidates[: max(0, int(config.max_records))]
    pending = max(0, len(candidates) - len(selected))
    created_utc = utc_now()
    if not selected:
        return FeedbackBundleResult(
            schema_version=FEEDBACK_BUNDLE_RESULT_SCHEMA,
            created_utc=created_utc,
            deployment_id=str(config.deployment_id),
            bundle_id="",
            bundle_dir="",
            bundle_zip_path="",
            manifest_path="",
            selected_records=0,
            pending_records=pending,
            state_path=str(state_path),
        )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    bundle_id = f"{_safe_name(config.deployment_id, fallback='deployment')}_feedback_{stamp}"
    bundle_dir = out_root / bundle_id
    records_dir = bundle_dir / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for created, record_id, src_dir, payload in selected:
        dst = records_dir / _safe_name(record_id, fallback="record")
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src_dir, dst)
        rows.append(
            {
                "record_id": record_id,
                "record_rel_path": dst.relative_to(bundle_dir).as_posix(),
                "created_utc": payload.get("created_utc", ""),
                "rating": payload.get("feedback", {}).get("rating", "unrated"),
                "artifact_manifest_sha256": str(payload.get("artifact_manifest_sha256", "")),
                "record_sha256": sha256_file(dst / "feedback_record.json"),
            }
        )

    manifest = FeedbackBundleManifest(
        schema_version=FEEDBACK_BUNDLE_SCHEMA,
        bundle_id=bundle_id,
        created_utc=created_utc,
        deployment_id=str(config.deployment_id),
        source_feedback_root=str(root),
        record_count=len(rows),
        records=rows,
        cadence_policy={
            "cadence_days": int(config.cadence_days),
            "cadence_record_count": int(config.cadence_record_count),
        },
    )
    manifest_path = bundle_dir / "feedback_bundle_manifest.json"
    _write_json_atomic(manifest_path, asdict(manifest))
    zip_path = Path(shutil.make_archive(str(bundle_dir), "zip", root_dir=bundle_dir))

    sent_ids.update(row["record_id"] for row in rows)
    state_payload = {
        "schema_version": "microseg.feedback_bundle_state.v1",
        "updated_utc": created_utc,
        "deployment_id": str(config.deployment_id),
        "sent_record_ids": sorted(sent_ids),
        "last_bundle_id": bundle_id,
        "last_bundle_zip_path": str(zip_path),
    }
    _write_json_atomic(state_path, state_payload)

    return FeedbackBundleResult(
        schema_version=FEEDBACK_BUNDLE_RESULT_SCHEMA,
        created_utc=created_utc,
        deployment_id=str(config.deployment_id),
        bundle_id=bundle_id,
        bundle_dir=str(bundle_dir),
        bundle_zip_path=str(zip_path),
        manifest_path=str(manifest_path),
        selected_records=len(rows),
        pending_records=pending,
        state_path=str(state_path),
    )


def _validate_record_dir(record_dir: Path) -> tuple[dict[str, Any], str, str]:
    record_path = record_dir / "feedback_record.json"
    artifact_manifest_path = record_dir / "artifacts_manifest.json"
    if not record_path.exists():
        raise FileNotFoundError(f"missing feedback_record.json in {record_dir}")
    if not artifact_manifest_path.exists():
        raise FileNotFoundError(f"missing artifacts_manifest.json in {record_dir}")

    payload = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"feedback_record.json is not an object in {record_dir}")
    if str(payload.get("schema_version", "")) != FEEDBACK_RECORD_SCHEMA:
        raise ValueError(f"unsupported feedback record schema in {record_dir}")

    manifest = json.loads(artifact_manifest_path.read_text(encoding="utf-8"))
    rows = manifest.get("files", [])
    if not isinstance(rows, list):
        raise ValueError(f"artifacts manifest files list missing in {record_dir}")
    for row in rows:
        if not isinstance(row, dict):
            continue
        rel = str(row.get("path", "")).strip()
        if not rel:
            continue
        file_path = record_dir / rel
        if not file_path.exists():
            raise FileNotFoundError(f"artifact missing ({rel}) in {record_dir}")
        expected = str(row.get("sha256", "")).strip()
        if expected:
            got = sha256_file(file_path)
            if got != expected:
                raise ValueError(f"artifact checksum mismatch ({rel}) in {record_dir}")

    record_sha = sha256_file(record_path)
    artifact_sha = sha256_file(artifact_manifest_path)
    return payload, record_sha, artifact_sha


def ingest_feedback_bundles(config: FeedbackIngestConfig) -> FeedbackIngestReport:
    """Ingest bundles into central lake with dedup + checksum validation."""

    ingest_root = Path(config.ingest_root).resolve()
    ingest_root.mkdir(parents=True, exist_ok=True)
    dedup_index_path = Path(config.dedup_index_path).resolve()
    review_queue_path = Path(config.review_queue_path).resolve()
    output_path = Path(config.output_path).resolve()

    index = _read_json(dedup_index_path)
    records_index = index.get("records", {})
    if not isinstance(records_index, dict):
        records_index = {}

    accepted: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    review_queue_records = 0
    processed = 0

    for raw_path in config.bundle_paths:
        bundle_path = Path(str(raw_path)).resolve()
        if not bundle_path.exists():
            rejected.append({"bundle_path": str(bundle_path), "reason": "bundle_path_missing"})
            continue
        tmp_ctx: tempfile.TemporaryDirectory[str] | None = None
        try:
            root, tmp_ctx = _extract_bundle_root(bundle_path)
            manifest_path = _find_manifest(root, "feedback_bundle_manifest.json")
            if manifest_path is None:
                rejected.append({"bundle_path": str(bundle_path), "reason": "manifest_missing"})
                continue
            manifest = _read_json(manifest_path)
            if str(manifest.get("schema_version", "")) != FEEDBACK_BUNDLE_SCHEMA:
                rejected.append({"bundle_path": str(bundle_path), "reason": "unsupported_bundle_schema"})
                continue
            processed += 1

            for row in manifest.get("records", []):
                if not isinstance(row, dict):
                    continue
                record_rel = str(row.get("record_rel_path", "")).strip()
                record_id = str(row.get("record_id", "")).strip()
                if not record_rel:
                    record_rel = f"records/{record_id}"
                record_dir = manifest_path.parent / record_rel
                try:
                    payload, record_sha, artifact_sha = _validate_record_dir(record_dir)
                except Exception as exc:
                    rejected.append(
                        {
                            "bundle_path": str(bundle_path),
                            "record_id": record_id or record_dir.name,
                            "reason": "record_validation_failed",
                            "detail": str(exc),
                        }
                    )
                    continue

                rid = str(payload.get("record_id", "")).strip() or record_dir.name
                existing = records_index.get(rid)
                if isinstance(existing, dict):
                    old_record_sha = str(existing.get("record_sha256", ""))
                    old_artifact_sha = str(existing.get("artifact_manifest_sha256", ""))
                    if old_record_sha == record_sha and old_artifact_sha == artifact_sha:
                        duplicates.append(
                            {
                                "record_id": rid,
                                "bundle_path": str(bundle_path),
                                "reason": "already_ingested",
                            }
                        )
                        continue
                    rejected.append(
                        {
                            "record_id": rid,
                            "bundle_path": str(bundle_path),
                            "reason": "record_id_conflict_hash_mismatch",
                        }
                    )
                    continue

                deployment_id = _safe_name(str(payload.get("deployment_id", "unknown")), fallback="deployment")
                created = _parse_timestamp(payload.get("created_utc")) or datetime.now(timezone.utc)
                target_dir = (
                    ingest_root
                    / deployment_id
                    / f"{created.year:04d}"
                    / f"{created.month:02d}"
                    / f"{created.day:02d}"
                    / _safe_name(rid, fallback="record")
                )
                if target_dir.exists():
                    shutil.rmtree(target_dir)
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(record_dir, target_dir)

                records_index[rid] = {
                    "record_sha256": record_sha,
                    "artifact_manifest_sha256": artifact_sha,
                    "ingested_utc": utc_now(),
                    "source_bundle_path": str(bundle_path),
                    "ingest_path": str(target_dir),
                }
                accepted.append(
                    {
                        "record_id": rid,
                        "ingest_path": str(target_dir),
                        "bundle_path": str(bundle_path),
                    }
                )

                rating = str(payload.get("feedback", {}).get("rating", "unrated"))
                has_corrected = bool(payload.get("correction", {}).get("has_corrected_mask", False))
                if rating == "thumbs_down" and not has_corrected:
                    review_row = {
                        "schema_version": "microseg.feedback_review_queue.v1",
                        "queued_utc": utc_now(),
                        "record_id": rid,
                        "ingest_path": str(target_dir),
                        "source_image_path": str(payload.get("source_image_path", "")),
                        "model_id": str(payload.get("model_id", "")),
                        "comment": str(payload.get("feedback", {}).get("comment", "")),
                    }
                    review_queue_path.parent.mkdir(parents=True, exist_ok=True)
                    with review_queue_path.open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(review_row, separators=(",", ":")) + "\n")
                    review_queue_records += 1
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    index_payload = {
        "schema_version": "microseg.feedback_ingest_index.v1",
        "updated_utc": utc_now(),
        "records": records_index,
    }
    _write_json_atomic(dedup_index_path, index_payload)

    report = FeedbackIngestReport(
        schema_version=FEEDBACK_INGEST_REPORT_SCHEMA,
        created_utc=utc_now(),
        ingest_root=str(ingest_root),
        bundles_processed=processed,
        accepted_records=len(accepted),
        duplicate_records=len(duplicates),
        rejected_records=len(rejected),
        review_queue_records=review_queue_records,
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
    )
    _write_json_atomic(output_path, asdict(report))
    return report


def build_feedback_training_dataset(config: FeedbackDatasetBuildConfig) -> FeedbackDatasetBuildResult:
    """Build deterministic train/val/test dataset from ingested feedback records."""

    root = Path(config.feedback_root).resolve()
    out = Path(config.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)

    scanned = 0
    corrected_samples = 0
    pseudo_samples = 0
    excluded_unrated = 0
    excluded_downvote = 0

    collected: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = {}

    for record_dir in discover_feedback_record_dirs(root):
        scanned += 1
        try:
            payload = load_feedback_record(record_dir)
        except Exception:
            continue
        artifacts = payload.get("artifacts", {})
        if not isinstance(artifacts, dict):
            continue
        rating = str(payload.get("feedback", {}).get("rating", "unrated"))
        corr = payload.get("correction", {})
        has_corrected = bool(corr.get("has_corrected_mask", False)) if isinstance(corr, dict) else False
        input_rel = str(artifacts.get("input", "input.png"))
        pred_rel = str(artifacts.get("predicted_mask_indexed", "predicted_mask_indexed.png"))
        corr_rel = str(artifacts.get("corrected_mask_indexed", "corrected_mask_indexed.png"))
        input_path = record_dir / input_rel
        predicted_mask_path = record_dir / pred_rel
        corrected_mask_path = record_dir / corr_rel

        if not input_path.exists() or not predicted_mask_path.exists():
            continue

        label_source = ""
        mask_path = predicted_mask_path
        weight = float(config.thumbs_up_weight)
        include = False
        if has_corrected and corrected_mask_path.exists():
            label_source = "human_corrected"
            mask_path = corrected_mask_path
            weight = float(config.corrected_weight)
            include = True
            corrected_samples += 1
        elif rating == "thumbs_up":
            label_source = "pseudo_accepted"
            include = True
            pseudo_samples += 1
        elif rating == "thumbs_down":
            excluded_downvote += 1
        else:
            excluded_unrated += 1

        if not include:
            continue
        record_id = str(payload.get("record_id", record_dir.name)).strip() or record_dir.name
        source_image_path = str(payload.get("source_image_path", ""))
        source_group = record_id
        if config.leakage_group == "source_stem":
            source_group = Path(source_image_path).stem or record_id
        row = {
            "record_id": record_id,
            "record_dir": record_dir,
            "input_path": input_path,
            "mask_path": mask_path,
            "weight": weight,
            "label_source": label_source,
            "payload": payload,
            "source_group": source_group,
        }
        collected.append(row)
        grouped.setdefault(source_group, []).append(row)

    group_to_split = _assign_groups(
        grouped,
        train_ratio=float(config.train_ratio),
        val_ratio=float(config.val_ratio),
        seed=int(config.seed),
    )
    split_counts = {"train": 0, "val": 0, "test": 0}
    sample_to_split: dict[str, str] = {}
    sample_weights: dict[str, float] = {}
    sample_label_source: dict[str, str] = {}

    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "masks").mkdir(parents=True, exist_ok=True)
        (out / split / "metadata").mkdir(parents=True, exist_ok=True)

    for row in collected:
        split = group_to_split.get(str(row["source_group"]), "test")
        sample_id = _safe_name(str(row["record_id"]), fallback="sample")
        dst_img = out / split / "images" / f"{sample_id}.png"
        dst_mask = out / split / "masks" / f"{sample_id}.png"
        dst_meta = out / split / "metadata" / f"{sample_id}.json"
        shutil.copy2(Path(row["input_path"]), dst_img)
        shutil.copy2(Path(row["mask_path"]), dst_mask)

        meta_payload = dict(row["payload"])
        meta_payload["feedback_dataset"] = {
            "sample_id": sample_id,
            "split": split,
            "label_source": str(row["label_source"]),
            "sample_weight": float(row["weight"]),
        }
        _write_json_atomic(dst_meta, meta_payload)
        split_counts[split] += 1
        sample_to_split[sample_id] = split
        sample_weights[sample_id] = float(row["weight"])
        sample_label_source[sample_id] = str(row["label_source"])

    weights_csv = out / "sample_weights.csv"
    lines = ["sample_id,split,label_source,weight"]
    for sample_id in sorted(sample_to_split.keys()):
        lines.append(
            f"{sample_id},{sample_to_split[sample_id]},{sample_label_source[sample_id]},{sample_weights[sample_id]:.6f}"
        )
    weights_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": FEEDBACK_DATASET_MANIFEST_SCHEMA,
        "created_utc": utc_now(),
        "config": asdict(config),
        "total_records_scanned": scanned,
        "included_samples": len(collected),
        "corrected_samples": corrected_samples,
        "pseudo_labeled_samples": pseudo_samples,
        "excluded_unrated": excluded_unrated,
        "excluded_downvote_without_correction": excluded_downvote,
        "split_counts": split_counts,
        "sample_to_split": sample_to_split,
        "sample_weights": sample_weights,
        "sample_label_source": sample_label_source,
        "sample_weights_csv": weights_csv.name,
    }
    manifest_path = out / "dataset_manifest.json"
    _write_json_atomic(manifest_path, manifest)

    return FeedbackDatasetBuildResult(
        schema_version="microseg.feedback_dataset_build_result.v1",
        created_utc=utc_now(),
        output_dir=str(out),
        manifest_path=str(manifest_path),
        total_records_scanned=scanned,
        included_samples=len(collected),
        corrected_samples=corrected_samples,
        pseudo_labeled_samples=pseudo_samples,
        excluded_unrated=excluded_unrated,
        excluded_downvote_without_correction=excluded_downvote,
        split_counts=split_counts,
    )


def _run_command(cmd: list[str]) -> int:
    proc = subprocess.run(cmd, check=False)
    return int(proc.returncode)


def evaluate_feedback_train_trigger(config: FeedbackTrainTriggerConfig) -> FeedbackTrainTriggerReport:
    """Evaluate/optionally execute threshold-based retraining trigger."""

    now = datetime.now(timezone.utc)
    state_path = Path(config.state_path).resolve()
    report_path = Path(config.output_path).resolve()
    state = _read_json(state_path)
    last_trigger = _parse_timestamp(state.get("last_trigger_utc"))

    corrected_since_last = 0
    for record_dir in discover_feedback_record_dirs(config.feedback_root):
        try:
            payload = load_feedback_record(record_dir)
        except Exception:
            continue
        corr = payload.get("correction", {})
        has_corrected = bool(corr.get("has_corrected_mask", False)) if isinstance(corr, dict) else False
        if not has_corrected:
            continue
        created = _parse_timestamp(payload.get("created_utc"))
        if created is None:
            continue
        if last_trigger is None or created > last_trigger:
            corrected_since_last += 1

    if last_trigger is None:
        days_since = float(config.max_days_since_last_trigger + 1)
    else:
        days_since = max(0.0, float((now - last_trigger).total_seconds() / 86400.0))

    should_trigger = (
        corrected_since_last >= int(config.corrected_threshold)
        or days_since >= float(config.max_days_since_last_trigger)
    )
    reason = "none"
    if should_trigger:
        if corrected_since_last >= int(config.corrected_threshold):
            reason = "corrected_threshold"
        else:
            reason = "max_days_elapsed"

    dataset_manifest_path = ""
    train_exit_code: int | None = None
    evaluate_exit_code: int | None = None
    commands: list[list[str]] = []

    if should_trigger:
        dataset_result = build_feedback_training_dataset(
            FeedbackDatasetBuildConfig(
                feedback_root=str(config.feedback_root),
                output_dir=str(config.dataset_output_dir),
            )
        )
        dataset_manifest_path = str(dataset_result.manifest_path)
        train_cmd = [
            sys.executable,
            "scripts/microseg_cli.py",
            "train",
            "--config",
            str(config.train_config),
            "--dataset-dir",
            str(config.dataset_output_dir),
            "--output-dir",
            str(config.train_output_dir),
        ]
        for item in config.train_overrides:
            train_cmd.extend(["--set", str(item)])
        eval_cmd = [
            sys.executable,
            "scripts/microseg_cli.py",
            "evaluate",
            "--config",
            str(config.evaluate_config),
            "--dataset-dir",
            str(config.dataset_output_dir),
            "--model-path",
            str(Path(config.train_output_dir) / "best_checkpoint.pt"),
            "--output-path",
            str(config.evaluate_output_path),
        ]
        for item in config.evaluate_overrides:
            eval_cmd.extend(["--set", str(item)])
        commands = [train_cmd, eval_cmd]

        if bool(config.execute):
            train_exit_code = _run_command(train_cmd)
            if train_exit_code == 0:
                evaluate_exit_code = _run_command(eval_cmd)
            else:
                evaluate_exit_code = None

            if train_exit_code == 0 and evaluate_exit_code == 0:
                state_payload = {
                    "schema_version": "microseg.feedback_train_trigger_state.v1",
                    "updated_utc": utc_now(),
                    "last_trigger_utc": utc_now(),
                    "last_trigger_reason": reason,
                    "last_dataset_manifest_path": dataset_manifest_path,
                    "last_train_output_dir": str(config.train_output_dir),
                    "last_evaluate_output_path": str(config.evaluate_output_path),
                }
                _write_json_atomic(state_path, state_payload)

    report = FeedbackTrainTriggerReport(
        schema_version=FEEDBACK_TRIGGER_REPORT_SCHEMA,
        created_utc=utc_now(),
        should_trigger=bool(should_trigger),
        trigger_reason=reason,
        corrected_records_since_last_trigger=corrected_since_last,
        days_since_last_trigger=days_since,
        state_path=str(state_path),
        report_path=str(report_path),
        dataset_manifest_path=dataset_manifest_path,
        train_exit_code=train_exit_code,
        evaluate_exit_code=evaluate_exit_code,
        commands=commands,
    )
    _write_json_atomic(report_path, asdict(report))
    return report
