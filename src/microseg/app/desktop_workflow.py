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
    get_gui_model_specs,
    resolve_gui_model_id,
    resolve_gui_model_reference,
    run_pipeline_from_gui,
)
from src.microseg.inference import ModelWarmLoadStatus, warm_load_reference_bundle
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
    feedback_record_dir: str = ""
    feedback_record_id: str = ""

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

    def model_specs(self) -> list[dict[str, str]]:
        return get_gui_model_specs()

    def preferred_default_model_name(self) -> str:
        """Return the first usable ML model, otherwise the conventional fallback."""

        specs = self.model_specs()
        conventional_name = specs[0]["display_name"] if specs else ""
        for spec in specs:
            model_id = str(spec.get("model_id", "")).strip()
            name = str(spec.get("display_name", "")).strip()
            if not name or model_id in {"", "hydride_conventional"}:
                if model_id == "hydride_conventional" and not conventional_name:
                    conventional_name = name
                continue
            try:
                if resolve_gui_model_reference(name, {}) is not None:
                    return name
            except Exception:
                continue
        return conventional_name

    def warm_model(self, model_name: str, *, params: dict | None = None) -> ModelWarmLoadStatus | None:
        """Warm-load an ML model bundle for GUI responsiveness."""

        reference = resolve_gui_model_reference(model_name, params)
        if reference is None:
            return None
        cfg = dict(params or {})
        return warm_load_reference_bundle(
            reference,
            enable_gpu=bool(cfg.get("enable_gpu", False)),
            device_policy=str(cfg.get("device_policy", "cpu")),
        )

    def history(self) -> list[DesktopRunRecord]:
        return list(self._history)

    def latest(self) -> DesktopRunRecord | None:
        return self._history[-1] if self._history else None

    def get(self, index: int) -> DesktopRunRecord:
        return self._history[index]

    def clear(self) -> None:
        self._history.clear()

    def append_history(self, record: DesktopRunRecord) -> None:
        """Append externally loaded run record into in-memory history."""

        self._history.append(record)
        if len(self._history) > self.max_history:
            self._history.pop(0)

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

        run_manifest = dict(result.manifest)
        run_manifest.setdefault("model_name", str(model_name))
        run_manifest.setdefault("image_path", str(image_path))
        run_manifest.setdefault("params", dict(params or {}))
        run_manifest.setdefault("runtime", {})
        if params:
            run_manifest["runtime"].update(
                {
                    "enable_gpu": bool(params.get("enable_gpu", False)),
                    "device_policy": str(params.get("device_policy", "cpu")),
                }
            )

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
            manifest=run_manifest,
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
            "feedback_record_dir": record.feedback_record_dir,
            "feedback_record_id": record.feedback_record_id,
        }
        (run_dir / "manifest.json").write_text(
            json.dumps(manifest_payload, indent=2),
            encoding="utf-8",
        )
        return run_dir


def load_exported_run(run_dir: str | Path) -> DesktopRunRecord:
    """Load a CLI-exported inference run folder into an in-memory record.

    Parameters
    ----------
    run_dir:
        Path to a directory produced by ``microseg-cli infer``.

    Returns
    -------
    DesktopRunRecord
        Reconstructed run record suitable for GUI display and correction.
    """

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    metrics_path = root / "metrics.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest.json in {root}")
    if not (root / "input.png").exists() or not (root / "prediction.png").exists() or not (root / "overlay.png").exists():
        raise FileNotFoundError(f"missing required exported image artifacts in {root}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"run manifest is not a JSON object: {manifest_path}")

    metrics: dict[str, float | int] = {}
    if metrics_path.exists():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        if isinstance(metrics_payload, dict):
            metrics = metrics_payload
    elif isinstance(manifest.get("metrics"), dict):
        metrics = dict(manifest["metrics"])

    input_img = Image.open(root / "input.png").convert("RGB")
    mask_img = Image.open(root / "prediction.png").convert("L")
    overlay_img = Image.open(root / "overlay.png").convert("RGB")

    standard_pngs = {"input.png", "prediction.png", "overlay.png"}
    analysis_images_b64: dict[str, str] = {}
    for extra in sorted(root.glob("*.png")):
        if extra.name in standard_pngs:
            continue
        analysis_images_b64[extra.stem] = base64.b64encode(extra.read_bytes()).decode("utf-8")

    nested_manifest = manifest.get("manifest")
    if not isinstance(nested_manifest, dict):
        nested_manifest = {}

    return DesktopRunRecord(
        run_id=str(manifest.get("run_id", root.name)),
        image_path=str(manifest.get("image_path", "")),
        image_name=str(manifest.get("image_name", Path(str(manifest.get("image_path", root.name))).name)),
        model_name=str(manifest.get("model_name", nested_manifest.get("model_name", ""))),
        model_id=str(manifest.get("model_id", nested_manifest.get("model_id", ""))),
        started_utc=str(manifest.get("started_utc", "")),
        finished_utc=str(manifest.get("finished_utc", "")),
        input_image=input_img,
        mask_image=mask_img,
        overlay_image=overlay_img,
        metrics=metrics,
        manifest=dict(nested_manifest),
        analysis_images_b64=analysis_images_b64,
        feedback_record_dir=str(manifest.get("feedback_record_dir", "")),
        feedback_record_id=str(manifest.get("feedback_record_id", "")),
    )
