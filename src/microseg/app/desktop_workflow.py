"""Desktop workflow manager for local segmentation app flows."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from hydride_segmentation.microseg_adapter import (
    get_gui_model_options,
    resolve_gui_model_id,
    run_pipeline_from_gui,
)
from src.microseg.utils import mask_overlay, to_rgb


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DesktopRunRecord:
    """In-memory record for one desktop segmentation run."""

    run_id: str
    image_path: str
    image_name: str
    model_name: str
    model_id: str
    started_utc: str
    finished_utc: str
    input_image: Image.Image
    mask_image: Image.Image
    overlay_image: Image.Image
    metrics: dict[str, float | int] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)
    analysis_images_b64: dict[str, str] = field(default_factory=dict)

    @property
    def history_label(self) -> str:
        return f"{self.image_name} | {self.model_name} | {self.run_id}"


class DesktopWorkflowManager:
    """Orchestrates single and batch runs for the local desktop application."""

    def __init__(self, max_history: int = 100) -> None:
        self.max_history = max_history
        self._history: list[DesktopRunRecord] = []

    def model_options(self) -> list[str]:
        return get_gui_model_options()

    def history(self) -> list[DesktopRunRecord]:
        return list(self._history)

    def latest(self) -> DesktopRunRecord | None:
        return self._history[-1] if self._history else None

    def get(self, index: int) -> DesktopRunRecord:
        return self._history[index]

    def clear(self) -> None:
        self._history.clear()

    def run_single(
        self,
        image_path: str,
        *,
        model_name: str,
        params: dict | None = None,
        include_analysis: bool = True,
    ) -> DesktopRunRecord:
        start_ts = _utc_now()
        result = run_pipeline_from_gui(
            image_path=image_path,
            model_name=model_name,
            params=params,
            include_analysis=include_analysis,
        )
        finish_ts = _utc_now()

        image_name = Path(image_path).name
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

        input_img = Image.fromarray(to_rgb(result.image))
        mask_img = Image.fromarray(result.mask)
        overlay_img = Image.fromarray(mask_overlay(result.image, result.mask))

        analysis_images = {
            k: v
            for k, v in result.images_b64.items()
            if k not in {"input_png_b64", "mask_png_b64", "overlay_png_b64"}
        }

        record = DesktopRunRecord(
            run_id=run_id,
            image_path=image_path,
            image_name=image_name,
            model_name=model_name,
            model_id=resolve_gui_model_id(model_name),
            started_utc=start_ts,
            finished_utc=finish_ts,
            input_image=input_img,
            mask_image=mask_img,
            overlay_image=overlay_img,
            metrics=result.metrics,
            manifest=result.manifest,
            analysis_images_b64=analysis_images,
        )
        self._history.append(record)
        if len(self._history) > self.max_history:
            self._history.pop(0)
        return record

    def run_batch(
        self,
        image_paths: list[str],
        *,
        model_name: str,
        params: dict | None = None,
        include_analysis: bool = False,
    ) -> list[DesktopRunRecord]:
        records: list[DesktopRunRecord] = []
        for path in image_paths:
            records.append(
                self.run_single(
                    path,
                    model_name=model_name,
                    params=params,
                    include_analysis=include_analysis,
                )
            )
        return records

    def export_run(self, record: DesktopRunRecord, output_dir: str | Path) -> Path:
        """Export run artifacts and metadata to a dedicated folder."""
        out_root = Path(output_dir)
        run_dir = out_root / f"{Path(record.image_name).stem}_{record.run_id}"
        run_dir.mkdir(parents=True, exist_ok=True)

        record.input_image.save(run_dir / "input.png")
        record.mask_image.save(run_dir / "prediction.png")
        record.overlay_image.save(run_dir / "overlay.png")

        for key, value in record.analysis_images_b64.items():
            raw = base64.b64decode(value.encode("utf-8"))
            (run_dir / f"{key}.png").write_bytes(raw)

        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(record.metrics, indent=2), encoding="utf-8")

        manifest_payload = {
            "run_id": record.run_id,
            "image_path": record.image_path,
            "image_name": record.image_name,
            "model_name": record.model_name,
            "model_id": record.model_id,
            "started_utc": record.started_utc,
            "finished_utc": record.finished_utc,
            "metrics": record.metrics,
            "manifest": record.manifest,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )
        return run_dir
