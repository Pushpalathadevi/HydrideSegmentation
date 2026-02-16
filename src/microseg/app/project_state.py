"""Persistence helpers for GUI project/session save and resume."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hydride_segmentation.version import __version__
from src.microseg.app.desktop_workflow import DesktopRunRecord
from src.microseg.corrections import DEFAULT_CLASS_MAP, SegmentationClassMap, to_index_mask


SCHEMA_VERSION = "microseg.project.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ProjectSaveRequest:
    """Serializable GUI project snapshot data."""

    record: DesktopRunRecord
    corrected_mask: np.ndarray
    class_map: SegmentationClassMap
    annotator: str = ""
    notes: str = ""
    ui_state: dict[str, Any] | None = None


@dataclass
class ProjectLoadResult:
    """Deserialized project snapshot result."""

    record: DesktopRunRecord
    corrected_mask: np.ndarray
    class_map: SegmentationClassMap
    annotator: str
    notes: str
    ui_state: dict[str, Any]
    root_dir: Path


class ProjectStateStore:
    """Save/load local GUI project folders for restartable sessions."""

    def save(self, req: ProjectSaveRequest, output_dir: str | Path) -> Path:
        """Persist project state into a folder with artifacts + metadata."""

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        input_path = out / "input.png"
        pred_path = out / "prediction_indexed.png"
        overlay_path = out / "overlay.png"
        corr_path = out / "corrected_indexed.png"

        req.record.input_image.save(input_path)
        Image.fromarray(to_index_mask(np.array(req.record.mask_image))).save(pred_path)
        req.record.overlay_image.save(overlay_path)
        Image.fromarray(to_index_mask(req.corrected_mask)).save(corr_path)

        payload = {
            "schema_version": SCHEMA_VERSION,
            "saved_utc": _utc_now(),
            "app_version": __version__,
            "record": {
                "run_id": req.record.run_id,
                "image_path": req.record.image_path,
                "image_name": req.record.image_name,
                "model_name": req.record.model_name,
                "model_id": req.record.model_id,
                "started_utc": req.record.started_utc,
                "finished_utc": req.record.finished_utc,
                "metrics": req.record.metrics,
                "manifest": req.record.manifest,
                "analysis_images_b64": req.record.analysis_images_b64,
            },
            "files": {
                "input": input_path.name,
                "prediction_indexed": pred_path.name,
                "overlay": overlay_path.name,
                "corrected_indexed": corr_path.name,
            },
            "annotator": req.annotator,
            "notes": req.notes,
            "class_map": req.class_map.as_dict(),
            "ui_state": req.ui_state or {},
        }
        (out / "project_state.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    def load(self, project_dir: str | Path) -> ProjectLoadResult:
        """Load previously saved project state folder."""

        root = Path(project_dir)
        state_path = root / "project_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"missing project_state.json in {root}")

        payload = json.loads(state_path.read_text(encoding="utf-8"))
        schema = payload.get("schema_version")
        if schema != SCHEMA_VERSION:
            raise ValueError(f"unsupported project schema: {schema}")

        files = payload["files"]
        input_img = Image.open(root / files["input"]).convert("RGB")
        pred_mask = Image.open(root / files["prediction_indexed"]).convert("L")
        overlay_img = Image.open(root / files["overlay"]).convert("RGB")
        corrected_idx = to_index_mask(np.array(Image.open(root / files["corrected_indexed"]).convert("L")))

        rec = payload["record"]
        record = DesktopRunRecord(
            run_id=rec["run_id"],
            image_path=rec["image_path"],
            image_name=rec["image_name"],
            model_name=rec["model_name"],
            model_id=rec["model_id"],
            started_utc=rec["started_utc"],
            finished_utc=rec["finished_utc"],
            input_image=input_img,
            mask_image=pred_mask,
            overlay_image=overlay_img,
            metrics=rec.get("metrics", {}),
            manifest=rec.get("manifest", {}),
            analysis_images_b64=rec.get("analysis_images_b64", {}),
        )
        class_map = SegmentationClassMap.from_dict(payload.get("class_map") or DEFAULT_CLASS_MAP.as_dict())

        return ProjectLoadResult(
            record=record,
            corrected_mask=corrected_idx,
            class_map=class_map,
            annotator=str(payload.get("annotator", "")),
            notes=str(payload.get("notes", "")),
            ui_state=dict(payload.get("ui_state", {})),
            root_dir=root,
        )
