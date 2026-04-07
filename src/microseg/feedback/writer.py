"""Per-inference feedback artifact writer and update helpers."""

from __future__ import annotations

import base64
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid
from typing import Any

import numpy as np
from PIL import Image

from .contracts import (
    FEEDBACK_RECORD_SCHEMA,
    FeedbackCaptureConfig,
    FeedbackCaptureResult,
    FeedbackRating,
)


def utc_now() -> str:
    """Return ISO-8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_name(text: str, *, fallback: str = "item") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(text).strip())
    cleaned = cleaned.strip("._-")
    return cleaned or fallback


def sha256_file(path: Path) -> str:
    """Return SHA256 checksum for a file."""

    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, separators=(",", ":")) + "\n")


def _to_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.name


def discover_feedback_record_dirs(feedback_root: str | Path) -> list[Path]:
    """Return sorted feedback record directories under root."""

    root = Path(feedback_root)
    if not root.exists():
        return []
    out: list[Path] = []
    for rec in root.rglob("feedback_record.json"):
        if rec.is_file():
            out.append(rec.parent)
    return sorted(set(out))


def load_feedback_record(record_dir: str | Path) -> dict[str, Any]:
    """Load ``feedback_record.json`` from record directory."""

    path = Path(record_dir) / "feedback_record.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"feedback record is not a JSON object: {path}")
    return payload


def _build_artifacts_manifest(record_dir: Path, artifacts: dict[str, str]) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    for key, rel in sorted(artifacts.items()):
        path = (record_dir / rel).resolve()
        exists = path.exists() and path.is_file()
        files.append(
            {
                "artifact_key": key,
                "path": rel,
                "exists": bool(exists),
                "size_bytes": int(path.stat().st_size) if exists else 0,
                "sha256": sha256_file(path) if exists else "",
            }
        )
    payload = {
        "schema_version": "microseg.feedback_artifacts_manifest.v1",
        "generated_utc": utc_now(),
        "record_dir": str(record_dir),
        "file_count": len(files),
        "files": files,
    }
    return payload


def _normalize_index_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError(f"predicted/corrected mask must be 2D indexed array, got shape={arr.shape}")
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)
    return arr


class FeedbackArtifactWriter:
    """Create and update canonical per-inference feedback record folders."""

    def __init__(self, config: FeedbackCaptureConfig) -> None:
        self.config = config

    @property
    def feedback_root(self) -> Path:
        return Path(self.config.feedback_root).resolve()

    def _new_record_id(self, run_id: str, image_path: str) -> str:
        stem = _safe_name(Path(str(image_path)).stem, fallback="image")
        return f"{stem}_{_safe_name(run_id, fallback='run')}_{uuid.uuid4().hex[:8]}"

    def _record_dir_for(self, record_id: str, created_utc: str) -> Path:
        dt = datetime.fromisoformat(created_utc.replace("Z", "+00:00")).astimezone(timezone.utc)
        return (
            self.feedback_root
            / _safe_name(self.config.deployment_id, fallback="deployment")
            / f"{dt.year:04d}"
            / f"{dt.month:02d}"
            / f"{dt.day:02d}"
            / _safe_name(record_id, fallback="record")
        )

    def create_from_inference_arrays(
        self,
        *,
        run_id: str,
        image_path: str,
        input_image_rgb: np.ndarray,
        predicted_mask_indexed: np.ndarray,
        predicted_overlay_rgb: np.ndarray,
        model_id: str,
        model_name: str,
        started_utc: str,
        finished_utc: str,
        inference_manifest: dict[str, Any] | None = None,
        resolved_config: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        analysis_images: dict[str, np.ndarray] | None = None,
        analysis_images_b64: dict[str, str] | None = None,
        source: str | None = None,
        operator_id: str | None = None,
        model_artifact_hint: str = "",
    ) -> FeedbackCaptureResult:
        """Create one full per-inference feedback record folder."""

        created = utc_now()
        record_id = self._new_record_id(run_id=run_id, image_path=image_path)
        record_dir = self._record_dir_for(record_id=record_id, created_utc=created)
        record_dir.mkdir(parents=True, exist_ok=True)

        input_path = record_dir / "input.png"
        pred_path = record_dir / "predicted_mask_indexed.png"
        overlay_path = record_dir / "predicted_overlay.png"
        inference_manifest_path = record_dir / "inference_manifest.json"
        resolved_cfg_path = record_dir / "resolved_config.json"
        events_path = record_dir / "feedback_events.jsonl"
        record_path = record_dir / "feedback_record.json"
        artifact_manifest_path = record_dir / "artifacts_manifest.json"

        Image.fromarray(np.asarray(input_image_rgb, dtype=np.uint8)).save(input_path)
        Image.fromarray(_normalize_index_mask(np.asarray(predicted_mask_indexed))).save(pred_path)
        Image.fromarray(np.asarray(predicted_overlay_rgb, dtype=np.uint8)).save(overlay_path)

        inf_manifest = dict(inference_manifest or {})
        _write_json_atomic(inference_manifest_path, inf_manifest)
        resolved_cfg = dict(resolved_config or {})
        _write_json_atomic(resolved_cfg_path, resolved_cfg)

        artifacts: dict[str, str] = {
            "input": input_path.name,
            "predicted_mask_indexed": pred_path.name,
            "predicted_overlay": overlay_path.name,
            "inference_manifest_json": inference_manifest_path.name,
            "resolved_config_json": resolved_cfg_path.name,
        }

        analysis_dir = record_dir / "analysis"
        if analysis_images:
            analysis_dir.mkdir(parents=True, exist_ok=True)
            for key, arr in analysis_images.items():
                name = _safe_name(str(key), fallback="analysis")
                rel = f"analysis/{name}.png"
                out = record_dir / rel
                Image.fromarray(np.asarray(arr, dtype=np.uint8)).save(out)
                artifacts[f"analysis_{name}"] = rel

        if analysis_images_b64:
            analysis_dir.mkdir(parents=True, exist_ok=True)
            for key, payload in analysis_images_b64.items():
                name = _safe_name(str(key), fallback="analysis")
                rel = f"analysis/{name}.png"
                out = record_dir / rel
                raw = base64.b64decode(str(payload).encode("utf-8"))
                out.write_bytes(raw)
                artifacts[f"analysis_{name}"] = rel

        artifact_manifest = _build_artifacts_manifest(record_dir, artifacts)
        _write_json_atomic(artifact_manifest_path, artifact_manifest)
        artifact_manifest_sha256 = sha256_file(artifact_manifest_path)

        src_sha = ""
        source_path = Path(str(image_path))
        if source_path.exists() and source_path.is_file():
            src_sha = sha256_file(source_path)
        cfg_text = json.dumps(resolved_cfg, sort_keys=True, separators=(",", ":"))
        payload = {
            "schema_version": FEEDBACK_RECORD_SCHEMA,
            "record_id": record_id,
            "run_id": str(run_id),
            "created_utc": created,
            "updated_utc": created,
            "deployment_id": str(self.config.deployment_id),
            "operator_id": str(operator_id or self.config.operator_id),
            "source": str(source or self.config.source),
            "source_image_path": str(image_path),
            "source_image_sha256": src_sha,
            "model_id": str(model_id),
            "model_name": str(model_name),
            "model_artifact_hint": str(model_artifact_hint),
            "started_utc": str(started_utc),
            "finished_utc": str(finished_utc),
            "runtime": dict(runtime or {}),
            "params": dict(params or {}),
            "resolved_config_sha256": _sha256_text(cfg_text),
            "inference_manifest": inf_manifest,
            "feedback": {
                "rating": "unrated",
                "comment": "",
            },
            "correction": {
                "has_corrected_mask": False,
                "corrected_mask_path": "",
                "correction_record_path": "",
                "linked_utc": "",
            },
            "artifacts": artifacts,
            "artifact_manifest_sha256": artifact_manifest_sha256,
        }
        _write_json_atomic(record_path, payload)
        _append_jsonl(
            events_path,
            {
                "event": "record_created",
                "created_utc": created,
                "record_id": record_id,
                "rating": "unrated",
            },
        )
        return FeedbackCaptureResult(
            record_id=record_id,
            record_dir=str(record_dir),
            record_path=str(record_path),
            events_path=str(events_path),
            artifacts_manifest_path=str(artifact_manifest_path),
        )

    def create_from_desktop_run(
        self,
        run: Any,
        *,
        source: str = "desktop_gui",
        resolved_config: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        runtime: dict[str, Any] | None = None,
        operator_id: str | None = None,
        model_artifact_hint: str = "",
    ) -> FeedbackCaptureResult:
        """Create record from desktop run-like object."""

        return self.create_from_inference_arrays(
            run_id=str(run.run_id),
            image_path=str(run.image_path),
            input_image_rgb=np.asarray(run.input_image),
            predicted_mask_indexed=np.asarray(run.mask_image),
            predicted_overlay_rgb=np.asarray(run.overlay_image),
            model_id=str(run.model_id),
            model_name=str(run.model_name),
            started_utc=str(run.started_utc),
            finished_utc=str(run.finished_utc),
            inference_manifest=dict(getattr(run, "manifest", {})),
            resolved_config=resolved_config,
            params=params,
            runtime=runtime,
            analysis_images_b64=dict(getattr(run, "analysis_images_b64", {})),
            source=source,
            operator_id=operator_id,
            model_artifact_hint=model_artifact_hint,
        )

    @staticmethod
    def _update_record(record_dir: Path, update_fn) -> dict[str, Any]:
        record_path = record_dir / "feedback_record.json"
        payload = json.loads(record_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"invalid feedback record payload: {record_path}")
        update_fn(payload)
        payload["updated_utc"] = utc_now()

        artifacts = payload.get("artifacts", {})
        if isinstance(artifacts, dict):
            manifest = _build_artifacts_manifest(record_dir, {str(k): str(v) for k, v in artifacts.items()})
            _write_json_atomic(record_dir / "artifacts_manifest.json", manifest)
            payload["artifact_manifest_sha256"] = sha256_file(record_dir / "artifacts_manifest.json")

        _write_json_atomic(record_path, payload)
        return payload

    def update_feedback(
        self,
        record_dir: str | Path,
        *,
        rating: FeedbackRating | None = None,
        comment: str | None = None,
        operator_id: str | None = None,
    ) -> dict[str, Any]:
        """Update feedback rating/comment for an existing record."""

        root = Path(record_dir)
        rating_value = rating
        if rating_value is not None and rating_value not in {"unrated", "thumbs_up", "thumbs_down"}:
            raise ValueError(f"unsupported feedback rating: {rating_value!r}")

        def _apply(payload: dict[str, Any]) -> None:
            feedback = payload.get("feedback")
            if not isinstance(feedback, dict):
                feedback = {"rating": "unrated", "comment": ""}
                payload["feedback"] = feedback
            if rating_value is not None:
                feedback["rating"] = str(rating_value)
            if comment is not None:
                feedback["comment"] = str(comment)
            if operator_id:
                payload["operator_id"] = str(operator_id)

        payload = self._update_record(root, _apply)
        _append_jsonl(
            root / "feedback_events.jsonl",
            {
                "event": "feedback_updated",
                "created_utc": utc_now(),
                "record_id": str(payload.get("record_id", "")),
                "rating": str(payload.get("feedback", {}).get("rating", "unrated")),
                "comment_len": len(str(payload.get("feedback", {}).get("comment", ""))),
                "operator_id": str(payload.get("operator_id", "")),
            },
        )
        return payload

    def attach_corrected_mask(
        self,
        record_dir: str | Path,
        corrected_mask: np.ndarray,
        *,
        correction_record_path: str = "",
    ) -> dict[str, Any]:
        """Attach corrected mask and optional correction-export linkage."""

        root = Path(record_dir)
        corrected_path = root / "corrected_mask_indexed.png"
        Image.fromarray(_normalize_index_mask(corrected_mask)).save(corrected_path)

        def _apply(payload: dict[str, Any]) -> None:
            artifacts = payload.get("artifacts")
            if not isinstance(artifacts, dict):
                artifacts = {}
                payload["artifacts"] = artifacts
            artifacts["corrected_mask_indexed"] = _to_rel(corrected_path, root)

            corr = payload.get("correction")
            if not isinstance(corr, dict):
                corr = {}
                payload["correction"] = corr
            corr["has_corrected_mask"] = True
            corr["corrected_mask_path"] = _to_rel(corrected_path, root)
            if correction_record_path:
                corr["correction_record_path"] = str(correction_record_path)
            corr["linked_utc"] = utc_now()

        payload = self._update_record(root, _apply)
        _append_jsonl(
            root / "feedback_events.jsonl",
            {
                "event": "correction_attached",
                "created_utc": utc_now(),
                "record_id": str(payload.get("record_id", "")),
                "corrected_mask_path": str(payload.get("correction", {}).get("corrected_mask_path", "")),
                "correction_record_path": str(payload.get("correction", {}).get("correction_record_path", "")),
            },
        )
        return payload

    def link_correction_export(
        self,
        record_dir: str | Path,
        *,
        correction_record_path: str,
    ) -> dict[str, Any]:
        """Link existing feedback record to exported correction metadata path."""

        root = Path(record_dir)

        def _apply(payload: dict[str, Any]) -> None:
            corr = payload.get("correction")
            if not isinstance(corr, dict):
                corr = {}
                payload["correction"] = corr
            corr["correction_record_path"] = str(correction_record_path)
            corr["linked_utc"] = utc_now()

        payload = self._update_record(root, _apply)
        _append_jsonl(
            root / "feedback_events.jsonl",
            {
                "event": "correction_export_linked",
                "created_utc": utc_now(),
                "record_id": str(payload.get("record_id", "")),
                "correction_record_path": str(correction_record_path),
            },
        )
        return payload
