"""Desktop result-package exporter with JSON/HTML/PDF summaries."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
import statistics
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.microseg.app.desktop_workflow import DesktopRunRecord
from src.microseg.app.desktop_ui_config import BALANCED_METRIC_KEYS, REPORT_PROFILES, REPORT_SECTIONS
from src.microseg.corrections.classes import (
    DEFAULT_CLASS_MAP,
    SegmentationClassMap,
    colorize_index_mask,
    resolve_class_map,
    to_index_mask,
)
from src.microseg.evaluation.hydride_statistics import (
    HydrideVisualizationConfig,
    compute_hydride_statistics,
    render_hydride_visualizations,
    statistics_to_json,
)
from src.microseg.utils import mask_overlay, to_rgb


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fmt_metric(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(to_rgb(image).astype(np.uint8)).save(path)


def _to_rel(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except Exception:
        return path.name


def _safe_float(value: object) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


@dataclass(frozen=True)
class DesktopResultExportConfig:
    """Report-export controls for desktop run packages."""

    orientation_bins: int = 18
    size_bins: int = 20
    min_feature_pixels: int = 1
    orientation_cmap: str = "coolwarm"
    size_scale: str = "linear"
    microns_per_pixel: float | None = None
    calibration_source: str = "none"
    calibration_notes: str = ""
    write_html_report: bool = True
    write_pdf_report: bool = True
    write_csv_report: bool = True
    write_batch_summary: bool = True
    report_profile: str = "balanced"
    selected_metric_keys: tuple[str, ...] = BALANCED_METRIC_KEYS
    include_sections: tuple[str, ...] = REPORT_SECTIONS
    sort_metrics: str = "name"
    top_k_key_metrics: int = 12
    include_artifact_manifest: bool = True

    def visualization_config(self) -> HydrideVisualizationConfig:
        """Return analysis-plot configuration object."""

        return HydrideVisualizationConfig(
            orientation_bins=max(1, int(self.orientation_bins)),
            size_bins=max(1, int(self.size_bins)),
            min_feature_pixels=max(1, int(self.min_feature_pixels)),
            orientation_cmap=str(self.orientation_cmap),
            size_scale=str(self.size_scale),
        )

    def normalized_profile(self) -> str:
        profile = str(self.report_profile).strip().lower()
        if profile in REPORT_PROFILES:
            return profile
        return "balanced"

    def normalized_sections(self) -> tuple[str, ...]:
        allowed = set(REPORT_SECTIONS)
        selected: list[str] = []
        for item in self.include_sections:
            key = str(item).strip()
            if key and key in allowed and key not in selected:
                selected.append(key)
        if selected:
            return tuple(selected)
        if self.normalized_profile() == "audit":
            return (
                "metadata",
                "calibration",
                "key_summary",
                "scalar_table",
                "overlays",
                "diff_panel",
                "artifact_manifest",
            )
        if self.normalized_profile() == "full":
            return REPORT_SECTIONS
        return (
            "metadata",
            "calibration",
            "key_summary",
            "scalar_table",
            "distribution_charts",
            "overlays",
            "diff_panel",
            "artifact_manifest",
        )

    def normalized_selected_metric_keys(self) -> tuple[str, ...]:
        out: list[str] = []
        seen: set[str] = set()
        for key in self.selected_metric_keys:
            text = str(key).strip()
            if not text or text in seen:
                continue
            seen.add(text)
            out.append(text)
        if out:
            return tuple(out)
        if self.normalized_profile() == "full":
            return ()
        return BALANCED_METRIC_KEYS

    def normalized_sort_metrics(self) -> str:
        mode = str(self.sort_metrics).strip().lower()
        if mode in {"name", "as_is", "value_desc"}:
            return mode
        return "name"

    def normalized_top_k_key_metrics(self) -> int:
        return max(1, min(200, int(self.top_k_key_metrics)))


class DesktopResultExporter:
    """Export segmentation results as a deployment-grade report package."""

    schema_version = "microseg.desktop_results.v2"
    batch_schema_version = "microseg.desktop_batch_results.v1"

    @staticmethod
    def _profile_metric_defaults(profile: str) -> tuple[str, ...]:
        if profile == "audit":
            return (
                "hydride_area_fraction_percent",
                "hydride_count",
                "hydride_total_area_pixels",
                "equivalent_diameter_mean_px",
                "orientation_mean_deg",
                "orientation_std_deg",
                "orientation_alignment_index",
                "orientation_entropy_bits",
                "excluded_small_features",
                "min_feature_pixels",
            )
        if profile == "full":
            return ()
        return BALANCED_METRIC_KEYS

    @staticmethod
    def _metric_rows(
        metric_keys: list[str],
        predicted_metrics: dict[str, Any],
        corrected_metrics: dict[str, Any],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in metric_keys:
            pred = predicted_metrics.get(key)
            corr = corrected_metrics.get(key)
            pred_f = _safe_float(pred)
            corr_f = _safe_float(corr)
            delta = None
            delta_pct = None
            if pred_f is not None and corr_f is not None:
                delta = float(corr_f - pred_f)
                if abs(pred_f) > 1e-12:
                    delta_pct = float((delta / pred_f) * 100.0)
            rows.append(
                {
                    "metric": str(key),
                    "predicted": pred,
                    "corrected": corr,
                    "delta": delta,
                    "delta_pct": delta_pct,
                }
            )
        return rows

    @staticmethod
    def _pick_metric_keys(
        *,
        config: DesktopResultExportConfig,
        predicted_metrics: dict[str, Any],
        corrected_metrics: dict[str, Any],
    ) -> list[str]:
        available = sorted(set(predicted_metrics.keys()) | set(corrected_metrics.keys()))
        available_set = set(available)
        profile = config.normalized_profile()
        selected = list(config.normalized_selected_metric_keys())
        if selected:
            keys = [key for key in selected if key in available_set]
        else:
            defaults = DesktopResultExporter._profile_metric_defaults(profile)
            if defaults:
                keys = [key for key in defaults if key in available_set]
            else:
                keys = list(available)
        if not keys:
            keys = list(available)

        mode = config.normalized_sort_metrics()
        if mode == "name":
            keys = sorted(keys)
        elif mode == "value_desc":
            keys = sorted(
                keys,
                key=lambda key: float(
                    _safe_float(corrected_metrics.get(key))
                    if _safe_float(corrected_metrics.get(key)) is not None
                    else (_safe_float(predicted_metrics.get(key)) or float("-inf"))
                ),
                reverse=True,
            )
        return keys

    @staticmethod
    def _key_summary_keys(metric_rows: list[dict[str, Any]], *, top_k: int) -> list[str]:
        numeric_rows = [
            row for row in metric_rows if _safe_float(row.get("corrected")) is not None or _safe_float(row.get("predicted")) is not None
        ]
        if not numeric_rows:
            return [str(row.get("metric", "")) for row in metric_rows[: max(1, top_k)]]
        ranked = sorted(
            numeric_rows,
            key=lambda row: abs(_safe_float(row.get("delta")) or (_safe_float(row.get("corrected")) or 0.0)),
            reverse=True,
        )
        out: list[str] = []
        seen: set[str] = set()
        for row in ranked:
            key = str(row.get("metric", "")).strip()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) >= int(top_k):
                break
        return out

    @staticmethod
    def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["metric", "predicted", "corrected", "delta", "delta_pct"],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        "metric": str(row.get("metric", "")),
                        "predicted": row.get("predicted"),
                        "corrected": row.get("corrected"),
                        "delta": row.get("delta"),
                        "delta_pct": row.get("delta_pct"),
                    }
                )

    @staticmethod
    def _build_artifact_manifest(*, run_dir: Path, artifacts: dict[str, str]) -> dict[str, Any]:
        files: list[dict[str, Any]] = []
        for key, rel in sorted(artifacts.items()):
            path = (run_dir / rel).resolve()
            if not path.exists() or not path.is_file():
                files.append(
                    {
                        "artifact_key": key,
                        "path": rel,
                        "exists": False,
                        "size_bytes": 0,
                    }
                )
                continue
            files.append(
                {
                    "artifact_key": key,
                    "path": rel,
                    "exists": True,
                    "size_bytes": int(path.stat().st_size),
                }
            )
        return {
            "schema_version": "microseg.artifact_manifest.v1",
            "generated_utc": _utc_now(),
            "file_count": len(files),
            "files": files,
        }

    @staticmethod
    def write_batch_artifact_manifest(batch_dir: Path) -> Path:
        """Write or refresh the aggregate batch artifact manifest for a batch export root."""

        rows_manifest: list[dict[str, Any]] = []
        for path in sorted(p for p in batch_dir.rglob("*") if p.is_file() and p.name != "artifacts_manifest.json"):
            rel = _to_rel(path, batch_dir)
            rows_manifest.append(
                {
                    "path": rel,
                    "size_bytes": int(path.stat().st_size),
                }
            )
        manifest_path = batch_dir / "artifacts_manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": "microseg.desktop_batch_artifacts_manifest.v1",
                    "created_utc": _utc_now(),
                    "file_count": len(rows_manifest),
                    "files": rows_manifest,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest_path

    def export(
        self,
        run: DesktopRunRecord,
        *,
        output_dir: str | Path,
        corrected_mask: np.ndarray | None = None,
        annotator: str = "",
        notes: str = "",
        class_map: SegmentationClassMap | None = None,
        config: DesktopResultExportConfig | None = None,
    ) -> Path:
        """Export report package for one desktop run.

        Parameters
        ----------
        run:
            Desktop run record to export.
        output_dir:
            Root folder where the package directory will be created.
        corrected_mask:
            Optional corrected mask. When omitted, predicted mask is reused.
        annotator:
            Optional annotator name stored in export metadata.
        notes:
            Optional notes stored in export metadata.
        class_map:
            Optional class map used for colorized outputs.
        config:
            Optional export/report configuration.

        Returns
        -------
        pathlib.Path
            Path to the created package directory.
        """

        cfg = config or DesktopResultExportConfig()
        profile = cfg.normalized_profile()
        include_sections = set(cfg.normalized_sections())
        if not bool(cfg.include_artifact_manifest):
            include_sections.discard("artifact_manifest")
        vis_cfg = cfg.visualization_config()
        cmap = class_map
        if cmap is None:
            try:
                cmap, _ = resolve_class_map()
            except Exception:
                cmap = DEFAULT_CLASS_MAP

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / f"{Path(run.image_name).stem}_{run.run_id}_results"
        run_dir.mkdir(parents=True, exist_ok=True)

        base = to_rgb(np.array(run.input_image))
        predicted_mask = to_index_mask(np.array(run.mask_image))
        corrected = to_index_mask(corrected_mask) if corrected_mask is not None else predicted_mask.copy()

        predicted_stats = compute_hydride_statistics(
            predicted_mask,
            orientation_bins=vis_cfg.orientation_bins,
            size_bins=vis_cfg.size_bins,
            min_feature_pixels=vis_cfg.min_feature_pixels,
            microns_per_pixel=cfg.microns_per_pixel,
        )
        corrected_stats = compute_hydride_statistics(
            corrected,
            orientation_bins=vis_cfg.orientation_bins,
            size_bins=vis_cfg.size_bins,
            min_feature_pixels=vis_cfg.min_feature_pixels,
            microns_per_pixel=cfg.microns_per_pixel,
        )
        predicted_visuals = render_hydride_visualizations(predicted_stats, vis_cfg)
        corrected_visuals = render_hydride_visualizations(corrected_stats, vis_cfg)

        input_path = run_dir / "input.png"
        predicted_mask_indexed_path = run_dir / "predicted_mask_indexed.png"
        corrected_mask_indexed_path = run_dir / "corrected_mask_indexed.png"
        predicted_mask_color_path = run_dir / "predicted_mask_color.png"
        corrected_mask_color_path = run_dir / "corrected_mask_color.png"
        predicted_overlay_path = run_dir / "predicted_overlay.png"
        corrected_overlay_path = run_dir / "corrected_overlay.png"
        predicted_orientation_map_path = run_dir / "predicted_orientation_map.png"
        predicted_size_hist_path = run_dir / "predicted_size_distribution.png"
        predicted_orientation_hist_path = run_dir / "predicted_orientation_distribution.png"
        corrected_orientation_map_path = run_dir / "corrected_orientation_map.png"
        corrected_size_hist_path = run_dir / "corrected_size_distribution.png"
        corrected_orientation_hist_path = run_dir / "corrected_orientation_distribution.png"
        diff_mask_path = run_dir / "diff_mask.png"

        _save_rgb(input_path, base)
        Image.fromarray(predicted_mask.astype(np.uint8)).save(predicted_mask_indexed_path)
        Image.fromarray(corrected.astype(np.uint8)).save(corrected_mask_indexed_path)
        Image.fromarray(colorize_index_mask(predicted_mask, cmap)).save(predicted_mask_color_path)
        Image.fromarray(colorize_index_mask(corrected, cmap)).save(corrected_mask_color_path)
        _save_rgb(predicted_overlay_path, mask_overlay(base, (predicted_mask > 0).astype(np.uint8) * 255))
        _save_rgb(corrected_overlay_path, mask_overlay(base, (corrected > 0).astype(np.uint8) * 255))
        _save_rgb(predicted_orientation_map_path, predicted_visuals["orientation_map_rgb"])
        _save_rgb(predicted_size_hist_path, predicted_visuals["size_distribution_rgb"])
        _save_rgb(predicted_orientation_hist_path, predicted_visuals["orientation_distribution_rgb"])
        _save_rgb(corrected_orientation_map_path, corrected_visuals["orientation_map_rgb"])
        _save_rgb(corrected_size_hist_path, corrected_visuals["size_distribution_rgb"])
        _save_rgb(corrected_orientation_hist_path, corrected_visuals["orientation_distribution_rgb"])
        Image.fromarray(np.where(corrected != predicted_mask, 255, 0).astype(np.uint8)).save(diff_mask_path)

        predicted_metrics = dict(predicted_stats.scalar_metrics)
        corrected_metrics = dict(corrected_stats.scalar_metrics)
        metric_keys = self._pick_metric_keys(
            config=cfg,
            predicted_metrics=predicted_metrics,
            corrected_metrics=corrected_metrics,
        )
        metric_rows = self._metric_rows(metric_keys, predicted_metrics, corrected_metrics)
        key_summary_keys = self._key_summary_keys(
            metric_rows,
            top_k=cfg.normalized_top_k_key_metrics(),
        )
        key_summary_rows = [row for row in metric_rows if str(row.get("metric")) in set(key_summary_keys)]

        artifacts = {
            "input": input_path.name,
            "predicted_mask_indexed": predicted_mask_indexed_path.name,
            "corrected_mask_indexed": corrected_mask_indexed_path.name,
            "predicted_mask_color": predicted_mask_color_path.name,
            "corrected_mask_color": corrected_mask_color_path.name,
            "predicted_overlay": predicted_overlay_path.name,
            "corrected_overlay": corrected_overlay_path.name,
            "predicted_orientation_map": predicted_orientation_map_path.name,
            "predicted_size_distribution": predicted_size_hist_path.name,
            "predicted_orientation_distribution": predicted_orientation_hist_path.name,
            "corrected_orientation_map": corrected_orientation_map_path.name,
            "corrected_size_distribution": corrected_size_hist_path.name,
            "corrected_orientation_distribution": corrected_orientation_hist_path.name,
            "diff_mask": diff_mask_path.name,
        }

        summary_payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "created_utc": _utc_now(),
            "run_id": run.run_id,
            "source_image_path": run.image_path,
            "image_name": run.image_name,
            "model_id": run.model_id,
            "model_name": run.model_name,
            "started_utc": run.started_utc,
            "finished_utc": run.finished_utc,
            "annotator": annotator or "unknown",
            "notes": notes,
            "analysis_config": {
                "orientation_bins": vis_cfg.orientation_bins,
                "size_bins": vis_cfg.size_bins,
                "min_feature_pixels": vis_cfg.min_feature_pixels,
                "orientation_cmap": vis_cfg.orientation_cmap,
                "size_scale": vis_cfg.size_scale,
            },
            "spatial_calibration": {
                "microns_per_pixel": None if cfg.microns_per_pixel is None else float(cfg.microns_per_pixel),
                "source": str(cfg.calibration_source),
                "notes": str(cfg.calibration_notes),
            },
            "pipeline_metrics": run.metrics,
            "predicted_stats": statistics_to_json(predicted_stats),
            "corrected_stats": statistics_to_json(corrected_stats),
            "selected_metric_rows": metric_rows,
            "key_summary_rows": key_summary_rows,
            "applied_export_criteria": {
                "report_profile": profile,
                "selected_metric_keys": [str(v) for v in cfg.normalized_selected_metric_keys()],
                "resolved_metric_keys": metric_keys,
                "include_sections": sorted(include_sections),
                "sort_metrics": cfg.normalized_sort_metrics(),
                "top_k_key_metrics": cfg.normalized_top_k_key_metrics(),
                "write_html_report": bool(cfg.write_html_report),
                "write_pdf_report": bool(cfg.write_pdf_report),
                "write_csv_report": bool(cfg.write_csv_report),
                "include_artifact_manifest": bool(cfg.include_artifact_manifest),
            },
            "artifacts": artifacts,
            "report_outputs": {
                "summary_json": "results_summary.json",
                "html_report": "results_report.html" if bool(cfg.write_html_report) else "",
                "pdf_report": "results_report.pdf" if bool(cfg.write_pdf_report) else "",
                "metrics_csv": "results_metrics.csv" if bool(cfg.write_csv_report) else "",
                "artifact_manifest_json": "artifacts_manifest.json" if bool(cfg.include_artifact_manifest) else "",
            },
        }

        if bool(cfg.include_artifact_manifest):
            artifact_manifest = self._build_artifact_manifest(run_dir=run_dir, artifacts=artifacts)
            manifest_path = run_dir / "artifacts_manifest.json"
            manifest_path.write_text(json.dumps(artifact_manifest, indent=2), encoding="utf-8")
            summary_payload["artifact_manifest"] = artifact_manifest

        if bool(cfg.write_csv_report):
            self._write_metrics_csv(run_dir / "results_metrics.csv", metric_rows)

        compatibility_manifest = {
            "run_id": run.run_id,
            "image_path": run.image_path,
            "image_name": run.image_name,
            "model_name": run.model_name,
            "model_id": run.model_id,
            "started_utc": run.started_utc,
            "finished_utc": run.finished_utc,
            "metrics": corrected_metrics,
            "manifest": run.manifest,
            "feedback_record_dir": run.feedback_record_dir,
            "feedback_record_id": run.feedback_record_id,
        }
        (run_dir / "manifest.json").write_text(json.dumps(compatibility_manifest, indent=2), encoding="utf-8")
        (run_dir / "metrics.json").write_text(json.dumps(corrected_metrics, indent=2), encoding="utf-8")

        summary_path = run_dir / "results_summary.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        if bool(cfg.write_html_report):
            html_path = run_dir / "results_report.html"
            html_path.write_text(self._build_html(summary_payload), encoding="utf-8")
        if bool(cfg.write_pdf_report):
            pdf_path = run_dir / "results_report.pdf"
            self._write_pdf(
                pdf_path=pdf_path,
                payload=summary_payload,
                predicted_visuals=predicted_visuals,
                corrected_visuals=corrected_visuals,
                base=base,
                predicted_mask=predicted_mask,
                corrected_mask=corrected,
            )
        return run_dir

    @staticmethod
    def _build_html(payload: dict[str, Any]) -> str:
        criteria = payload.get("applied_export_criteria", {})
        include_sections = set(criteria.get("include_sections", []))
        metric_rows = payload.get("selected_metric_rows", [])
        key_rows = payload.get("key_summary_rows", [])
        artifacts = payload.get("artifacts", {})
        artifact_manifest = payload.get("artifact_manifest", {})

        rows_html = []
        for row in metric_rows if isinstance(metric_rows, list) else []:
            if not isinstance(row, dict):
                continue
            rows_html.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('metric', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('predicted', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('corrected', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('delta', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('delta_pct', '')))}</td>"
                "</tr>"
            )

        key_rows_html = []
        for row in key_rows if isinstance(key_rows, list) else []:
            if not isinstance(row, dict):
                continue
            key_rows_html.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('metric', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('predicted', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('corrected', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('delta', '')))}</td>"
                "</tr>"
            )

        overlay_grid = "\n".join(
            [
                f"<figure><img src='{html.escape(str(artifacts.get(name, '')))}' alt='{html.escape(name)}'/><figcaption>{html.escape(name)}</figcaption></figure>"
                for name in [
                    "input",
                    "predicted_mask_color",
                    "corrected_mask_color",
                    "predicted_overlay",
                    "corrected_overlay",
                ]
                if str(artifacts.get(name, "")).strip()
            ]
        )
        distribution_grid = "\n".join(
            [
                f"<figure><img src='{html.escape(str(artifacts.get(name, '')))}' alt='{html.escape(name)}'/><figcaption>{html.escape(name)}</figcaption></figure>"
                for name in [
                    "predicted_orientation_map",
                    "predicted_size_distribution",
                    "predicted_orientation_distribution",
                    "corrected_orientation_map",
                    "corrected_size_distribution",
                    "corrected_orientation_distribution",
                ]
                if str(artifacts.get(name, "")).strip()
            ]
        )
        manifest_rows = []
        if isinstance(artifact_manifest, dict):
            for file_row in artifact_manifest.get("files", []):
                if not isinstance(file_row, dict):
                    continue
                manifest_rows.append(
                    "<tr>"
                    f"<td>{html.escape(str(file_row.get('artifact_key', '')))}</td>"
                    f"<td>{html.escape(str(file_row.get('path', '')))}</td>"
                    f"<td>{html.escape(str(file_row.get('size_bytes', '')))}</td>"
                    "</tr>"
                )

        sections: list[str] = []
        if "metadata" in include_sections:
            sections.append(
                "<h2>Metadata</h2>"
                f"<div class='meta'><div>Run ID</div><div>{html.escape(str(payload.get('run_id', '')))}</div>"
                f"<div>Model</div><div>{html.escape(str(payload.get('model_name', '')))} ({html.escape(str(payload.get('model_id', '')))} )</div>"
                f"<div>Source Image</div><div>{html.escape(str(payload.get('source_image_path', '')))}</div>"
                f"<div>Generated UTC</div><div>{html.escape(str(payload.get('created_utc', '')))}</div>"
                f"<div>Annotator</div><div>{html.escape(str(payload.get('annotator', '')))}</div>"
                f"<div>Notes</div><div>{html.escape(str(payload.get('notes', '')))}</div></div>"
            )
        if "calibration" in include_sections:
            cal = payload.get("spatial_calibration", {})
            sections.append(
                "<h2>Calibration</h2>"
                f"<p>microns_per_pixel={html.escape(str(cal.get('microns_per_pixel', 'None')))} | "
                f"source={html.escape(str(cal.get('source', 'none')))} | "
                f"notes={html.escape(str(cal.get('notes', '')))}</p>"
            )
        if "key_summary" in include_sections:
            sections.append(
                "<h2>Key Metrics</h2>"
                "<table><thead><tr><th>Metric</th><th>Predicted</th><th>Corrected</th><th>Delta</th></tr></thead>"
                f"<tbody>{''.join(key_rows_html) if key_rows_html else '<tr><td colspan=4>n/a</td></tr>'}</tbody></table>"
            )
        if "scalar_table" in include_sections:
            sections.append(
                "<h2>Scalar Statistics</h2>"
                "<table><thead><tr><th>Metric</th><th>Predicted</th><th>Corrected</th><th>Delta</th><th>Delta %</th></tr></thead>"
                f"<tbody>{''.join(rows_html) if rows_html else '<tr><td colspan=5>n/a</td></tr>'}</tbody></table>"
            )
        if "overlays" in include_sections:
            sections.append("<h2>Overlays</h2>" + f"<div class='grid'>{overlay_grid}</div>")
        if "distribution_charts" in include_sections:
            sections.append("<h2>Distributions</h2>" + f"<div class='grid'>{distribution_grid}</div>")
        if "diff_panel" in include_sections and str(artifacts.get("diff_mask", "")).strip():
            sections.append(
                "<h2>Diff Panel</h2>"
                f"<figure><img src='{html.escape(str(artifacts.get('diff_mask', '')))}' alt='diff_mask'/>"
                "<figcaption>diff_mask</figcaption></figure>"
            )
        if "artifact_manifest" in include_sections and manifest_rows:
            sections.append(
                "<h2>Artifact Manifest</h2>"
                "<table><thead><tr><th>Key</th><th>Path</th><th>Size (bytes)</th></tr></thead>"
                f"<tbody>{''.join(manifest_rows)}</tbody></table>"
            )

        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>MicroSeg Result Report</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:20px;background:#f4f6f8;color:#162029;}"
            "h1,h2{margin:10px 0 8px 0;}"
            "table{border-collapse:collapse;width:100%;margin:14px 0;background:#fff;}"
            "th,td{border:1px solid #d6dde4;padding:8px;text-align:left;font-size:13px;}"
            "th{background:#eef2f6;}"
            ".meta{display:grid;grid-template-columns:220px 1fr;gap:6px;background:#fff;border:1px solid #d6dde4;padding:10px;}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;}"
            "figure{background:#fff;border:1px solid #d6dde4;padding:8px;margin:0;}"
            "img{max-width:100%;height:auto;display:block;margin:0 auto;}"
            "figcaption{font-size:12px;color:#2a3946;margin-top:6px;text-align:center;}"
            ".criteria{background:#fff;border:1px solid #d6dde4;padding:10px;font-size:12px;}"
            "</style></head><body>"
            "<h1>MicroSeg Result Report</h1>"
            "<div class='criteria'>"
            f"profile={html.escape(str(criteria.get('report_profile', '')))} | "
            f"sections={html.escape(','.join(criteria.get('include_sections', []) if isinstance(criteria.get('include_sections', []), list) else []))} | "
            f"sort={html.escape(str(criteria.get('sort_metrics', '')))}"
            "</div>"
            + "".join(sections)
            + "</body></html>"
        )

    @staticmethod
    def _write_pdf(
        *,
        pdf_path: Path,
        payload: dict[str, Any],
        predicted_visuals: dict[str, np.ndarray],
        corrected_visuals: dict[str, np.ndarray],
        base: np.ndarray,
        predicted_mask: np.ndarray,
        corrected_mask: np.ndarray,
    ) -> None:
        criteria = payload.get("applied_export_criteria", {})
        include_sections = set(criteria.get("include_sections", []))
        pred_overlay = mask_overlay(base, (predicted_mask > 0).astype(np.uint8) * 255)
        corr_overlay = mask_overlay(base, (corrected_mask > 0).astype(np.uint8) * 255)
        pred_color = np.array(colorize_index_mask(predicted_mask, DEFAULT_CLASS_MAP))
        corr_color = np.array(colorize_index_mask(corrected_mask, DEFAULT_CLASS_MAP))
        diff_mask = np.where(corrected_mask != predicted_mask, 255, 0).astype(np.uint8)

        metric_rows = payload.get("selected_metric_rows", [])
        key_rows = payload.get("key_summary_rows", [])

        with PdfPages(pdf_path) as pdf:
            if "metadata" in include_sections or "calibration" in include_sections or "key_summary" in include_sections:
                fig_meta = plt.figure(figsize=(11.0, 8.5))
                fig_meta.suptitle("MicroSeg Result Report", fontsize=15, fontweight="bold")
                lines = [
                    f"Run ID: {payload.get('run_id', '')}",
                    f"Model: {payload.get('model_name', '')} ({payload.get('model_id', '')})",
                    f"Source: {payload.get('source_image_path', '')}",
                    f"Generated UTC: {payload.get('created_utc', '')}",
                    f"Annotator: {payload.get('annotator', '')}",
                    f"Notes: {payload.get('notes', '')}",
                    (
                        "Calibration: {} um/px ({})".format(
                            payload.get("spatial_calibration", {}).get("microns_per_pixel", None),
                            payload.get("spatial_calibration", {}).get("source", "none"),
                        )
                    ),
                    "",
                    "Report criteria:",
                    f"- profile={criteria.get('report_profile', '')}",
                    f"- sections={','.join(criteria.get('include_sections', []))}",
                    "",
                    "Key metrics (predicted -> corrected):",
                ]
                for row in key_rows if isinstance(key_rows, list) else []:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        f"- {row.get('metric', '')}: {_fmt_metric(row.get('predicted'))} -> {_fmt_metric(row.get('corrected'))}"
                    )
                fig_meta.text(0.05, 0.92, "\n".join(lines), va="top", family="monospace", fontsize=10)
                pdf.savefig(fig_meta)
                plt.close(fig_meta)

            if "overlays" in include_sections or "diff_panel" in include_sections:
                fig_comp, ax = plt.subplots(2, 3, figsize=(11.0, 8.5))
                panel = [
                    ("Input", base),
                    ("Predicted Mask", pred_color),
                    ("Predicted Overlay", pred_overlay),
                    ("Corrected Mask", corr_color),
                    ("Corrected Overlay", corr_overlay),
                    ("Diff (Corrected - Predicted)", diff_mask),
                ]
                for i, (title, image) in enumerate(panel):
                    r, c = divmod(i, 3)
                    ax[r, c].imshow(to_rgb(image))
                    ax[r, c].set_title(title)
                    ax[r, c].axis("off")
                fig_comp.tight_layout()
                pdf.savefig(fig_comp)
                plt.close(fig_comp)

            if "distribution_charts" in include_sections:
                fig_pred, ax_pred = plt.subplots(1, 3, figsize=(11.0, 4.0))
                pred_panels = [
                    ("Predicted Orientation Map", predicted_visuals["orientation_map_rgb"]),
                    ("Predicted Size Distribution", predicted_visuals["size_distribution_rgb"]),
                    ("Predicted Orientation Distribution", predicted_visuals["orientation_distribution_rgb"]),
                ]
                for idx, (title, image) in enumerate(pred_panels):
                    ax_pred[idx].imshow(image)
                    ax_pred[idx].set_title(title, fontsize=10)
                    ax_pred[idx].axis("off")
                fig_pred.tight_layout()
                pdf.savefig(fig_pred)
                plt.close(fig_pred)

                fig_corr, ax_corr = plt.subplots(1, 3, figsize=(11.0, 4.0))
                corr_panels = [
                    ("Corrected Orientation Map", corrected_visuals["orientation_map_rgb"]),
                    ("Corrected Size Distribution", corrected_visuals["size_distribution_rgb"]),
                    ("Corrected Orientation Distribution", corrected_visuals["orientation_distribution_rgb"]),
                ]
                for idx, (title, image) in enumerate(corr_panels):
                    ax_corr[idx].imshow(image)
                    ax_corr[idx].set_title(title, fontsize=10)
                    ax_corr[idx].axis("off")
                fig_corr.tight_layout()
                pdf.savefig(fig_corr)
                plt.close(fig_corr)

            if "scalar_table" in include_sections and isinstance(metric_rows, list):
                fig_tbl = plt.figure(figsize=(11.0, 8.5))
                fig_tbl.suptitle("Selected Scalar Metrics", fontsize=13, fontweight="bold")
                lines = ["metric | predicted | corrected | delta | delta_pct"]
                for row in metric_rows[:80]:
                    if not isinstance(row, dict):
                        continue
                    lines.append(
                        "{} | {} | {} | {} | {}".format(
                            row.get("metric", ""),
                            _fmt_metric(row.get("predicted", "")),
                            _fmt_metric(row.get("corrected", "")),
                            _fmt_metric(row.get("delta", "")),
                            _fmt_metric(row.get("delta_pct", "")),
                        )
                    )
                fig_tbl.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=8)
                pdf.savefig(fig_tbl)
                plt.close(fig_tbl)

    def export_batch(
        self,
        runs: list[DesktopRunRecord],
        *,
        output_dir: str | Path,
        corrected_masks: dict[str, np.ndarray] | None = None,
        annotator: str = "",
        notes: str = "",
        class_map: SegmentationClassMap | None = None,
        config: DesktopResultExportConfig | None = None,
    ) -> Path:
        """Export aggregate summary package for multiple runs."""

        if not runs:
            raise ValueError("runs is empty")
        cfg = config or DesktopResultExportConfig()
        profile = cfg.normalized_profile()
        include_sections = set(cfg.normalized_sections())
        if not bool(cfg.include_artifact_manifest):
            include_sections.discard("artifact_manifest")
        vis_cfg = cfg.visualization_config()

        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        batch_dir = root / f"batch_results_{batch_id}"
        batch_dir.mkdir(parents=True, exist_ok=True)

        rows: list[dict[str, Any]] = []
        model_counts: dict[str, int] = {}
        metric_value_map: dict[str, list[float]] = {}
        selected_keys = cfg.normalized_selected_metric_keys()
        selected_set = set(selected_keys)
        runs_dir = batch_dir / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        for run in runs:
            per_run_dir = self.export(
                run,
                output_dir=runs_dir,
                corrected_mask=corrected_masks.get(run.run_id) if corrected_masks else None,
                annotator=annotator,
                notes=notes,
                class_map=class_map,
                config=cfg,
            )
            um_per_px = None if cfg.microns_per_pixel is None else float(cfg.microns_per_pixel)
            pred_metrics = dict(run.metrics)
            if not pred_metrics:
                pred_mask = to_index_mask(np.array(run.mask_image))
                pred_stats = compute_hydride_statistics(
                    pred_mask,
                    orientation_bins=vis_cfg.orientation_bins,
                    size_bins=vis_cfg.size_bins,
                    min_feature_pixels=vis_cfg.min_feature_pixels,
                    microns_per_pixel=um_per_px,
                )
                pred_metrics = dict(pred_stats.scalar_metrics)
            corr_metrics = dict(pred_metrics)
            if corrected_masks and run.run_id in corrected_masks:
                corr_mask = to_index_mask(np.asarray(corrected_masks[run.run_id]))
                corr_stats = compute_hydride_statistics(
                    corr_mask,
                    orientation_bins=vis_cfg.orientation_bins,
                    size_bins=vis_cfg.size_bins,
                    min_feature_pixels=vis_cfg.min_feature_pixels,
                    microns_per_pixel=um_per_px,
                )
                corr_metrics = dict(corr_stats.scalar_metrics)
            keys = self._pick_metric_keys(config=cfg, predicted_metrics=pred_metrics, corrected_metrics=corr_metrics)
            if selected_set:
                keys = [k for k in keys if k in selected_set]
            row: dict[str, Any] = {
                "run_id": run.run_id,
                "image_name": run.image_name,
                "image_path": run.image_path,
                "model_name": run.model_name,
                "model_id": run.model_id,
                "started_utc": run.started_utc,
                "finished_utc": run.finished_utc,
                "run_export_dir": _to_rel(per_run_dir, batch_dir),
                "run_summary_path": _to_rel(per_run_dir / "results_summary.json", batch_dir),
                "run_html_report_path": _to_rel(per_run_dir / "results_report.html", batch_dir) if bool(cfg.write_html_report) else "",
                "run_pdf_report_path": _to_rel(per_run_dir / "results_report.pdf", batch_dir) if bool(cfg.write_pdf_report) else "",
                "run_metrics_csv_path": _to_rel(per_run_dir / "results_metrics.csv", batch_dir) if bool(cfg.write_csv_report) else "",
                "input_preview_path": _to_rel(per_run_dir / "input.png", batch_dir),
                "mask_preview_path": _to_rel(per_run_dir / "predicted_mask_color.png", batch_dir),
                "overlay_preview_path": _to_rel(per_run_dir / "predicted_overlay.png", batch_dir),
            }
            model_counts[run.model_name] = int(model_counts.get(run.model_name, 0)) + 1
            for key in keys:
                p_val = pred_metrics.get(key)
                c_val = corr_metrics.get(key)
                row[f"predicted_{key}"] = p_val
                row[f"corrected_{key}"] = c_val
                use_val = _safe_float(c_val)
                if use_val is None:
                    use_val = _safe_float(p_val)
                if use_val is not None:
                    metric_value_map.setdefault(key, []).append(float(use_val))
            rows.append(row)

        aggregate_rows: list[dict[str, Any]] = []
        for key in sorted(metric_value_map.keys()):
            vals = metric_value_map[key]
            aggregate_rows.append(
                {
                    "metric": key,
                    "count": len(vals),
                    "mean": float(sum(vals) / len(vals)) if vals else None,
                    "median": float(statistics.median(vals)) if vals else None,
                    "std": float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0,
                    "min": float(min(vals)) if vals else None,
                    "max": float(max(vals)) if vals else None,
                }
            )

        summary_payload: dict[str, Any] = {
            "schema_version": self.batch_schema_version,
            "created_utc": _utc_now(),
            "batch_id": batch_id,
            "run_count": len(rows),
            "annotator": annotator or "unknown",
            "notes": notes,
            "analysis_config": {
                "orientation_bins": vis_cfg.orientation_bins,
                "size_bins": vis_cfg.size_bins,
                "min_feature_pixels": vis_cfg.min_feature_pixels,
                "orientation_cmap": vis_cfg.orientation_cmap,
                "size_scale": vis_cfg.size_scale,
            },
            "applied_export_criteria": {
                "report_profile": profile,
                "selected_metric_keys": [str(v) for v in selected_keys],
                "include_sections": sorted(include_sections),
                "sort_metrics": cfg.normalized_sort_metrics(),
                "top_k_key_metrics": cfg.normalized_top_k_key_metrics(),
                "write_html_report": bool(cfg.write_html_report),
                "write_pdf_report": bool(cfg.write_pdf_report),
                "write_csv_report": bool(cfg.write_csv_report),
            },
            "model_counts": model_counts,
            "rows": rows,
            "aggregate_metrics": aggregate_rows,
            "report_outputs": {
                "summary_json": "batch_results_summary.json",
                "html_report": "batch_results_report.html" if bool(cfg.write_html_report) else "",
                "pdf_report": "batch_results_report.pdf" if bool(cfg.write_pdf_report) else "",
                "metrics_csv": "batch_metrics.csv" if bool(cfg.write_csv_report) else "",
                "artifacts_manifest": "artifacts_manifest.json" if bool(cfg.include_artifact_manifest) else "",
                "runs_dir": "runs",
            },
        }

        summary_path = batch_dir / "batch_results_summary.json"
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

        if bool(cfg.write_csv_report):
            csv_path = batch_dir / "batch_metrics.csv"
            all_fields: list[str] = []
            for row in rows:
                for key in row.keys():
                    if key not in all_fields:
                        all_fields.append(key)
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=all_fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, "") for k in all_fields})

        if bool(cfg.write_html_report):
            html_path = batch_dir / "batch_results_report.html"
            html_path.write_text(self._build_batch_html(summary_payload), encoding="utf-8")

        if bool(cfg.write_pdf_report):
            pdf_path = batch_dir / "batch_results_report.pdf"
            self._write_batch_pdf(pdf_path=pdf_path, payload=summary_payload)

        if bool(cfg.include_artifact_manifest):
            self.write_batch_artifact_manifest(batch_dir)

        return batch_dir

    @staticmethod
    def _build_batch_html(payload: dict[str, Any]) -> str:
        rows = payload.get("rows", [])
        aggregate = payload.get("aggregate_metrics", [])
        telemetry = payload.get("telemetry", {})
        header_fields: list[str] = []
        hidden_fields = {
            "input_preview_path",
            "mask_preview_path",
            "overlay_preview_path",
            "run_export_dir",
            "run_summary_path",
            "run_html_report_path",
            "run_pdf_report_path",
            "run_metrics_csv_path",
        }
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                if key in hidden_fields:
                    continue
                if key not in header_fields:
                    header_fields.append(key)
        run_rows = []
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict):
                continue
            metric_bits = []
            for key in (
                "corrected_hydride_area_fraction_percent",
                "predicted_hydride_area_fraction_percent",
                "corrected_hydride_count",
                "predicted_hydride_count",
            ):
                if key in row:
                    metric_bits.append(f"{html.escape(key)}={html.escape(_fmt_metric(row.get(key, '')))}")
            metrics_inline = "<br/>".join(metric_bits) if metric_bits else "n/a"
            links_inline = " | ".join(
                [
                    item
                    for item in [
                        (
                            f"<a href='{html.escape(str(row.get('run_summary_path', '')))}'>summary</a>"
                            if str(row.get("run_summary_path", "")).strip()
                            else ""
                        ),
                        (
                            f"<a href='{html.escape(str(row.get('run_html_report_path', '')))}'>html</a>"
                            if str(row.get("run_html_report_path", "")).strip()
                            else ""
                        ),
                        (
                            f"<a href='{html.escape(str(row.get('run_pdf_report_path', '')))}'>pdf</a>"
                            if str(row.get("run_pdf_report_path", "")).strip()
                            else ""
                        ),
                        (
                            f"<a href='{html.escape(str(row.get('run_metrics_csv_path', '')))}'>csv</a>"
                            if str(row.get("run_metrics_csv_path", "")).strip()
                            else ""
                        ),
                    ]
                    if item
                ]
            )
            run_rows.append(
                "<tr>"
                f"<td><a href='{html.escape(str(row.get('run_summary_path', '')))}'><img src='{html.escape(str(row.get('input_preview_path', '')))}' alt='input' style='max-width:220px;max-height:140px;'/></a></td>"
                f"<td><a href='{html.escape(str(row.get('run_summary_path', '')))}'><img src='{html.escape(str(row.get('mask_preview_path', '')))}' alt='mask' style='max-width:220px;max-height:140px;'/></a></td>"
                f"<td><a href='{html.escape(str(row.get('run_summary_path', '')))}'><img src='{html.escape(str(row.get('overlay_preview_path', '')))}' alt='overlay' style='max-width:220px;max-height:140px;'/></a></td>"
                f"<td>{metrics_inline}</td>"
                f"<td>{links_inline or 'n/a'}</td>"
                + "".join(f"<td>{html.escape(_fmt_metric(row.get(k, '')))}</td>" for k in header_fields)
                + "</tr>"
            )
        agg_rows = []
        for row in aggregate if isinstance(aggregate, list) else []:
            if not isinstance(row, dict):
                continue
            agg_rows.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('metric', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('count', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('mean', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('median', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('std', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('min', '')))}</td>"
                f"<td>{html.escape(_fmt_metric(row.get('max', '')))}</td>"
                "</tr>"
            )
        telemetry_rows = []
        if isinstance(telemetry, dict):
            for key in (
                "job_elapsed_human",
                "job_elapsed_seconds",
                "throughput_images_per_second",
                "total_images",
                "completed_images",
                "total_steps",
                "completed_steps",
                "run_duration_seconds_total",
                "run_duration_seconds_mean",
                "run_duration_seconds_min",
                "run_duration_seconds_max",
                "earliest_run_started_utc",
                "latest_run_finished_utc",
                "batch_completed_utc",
                "model_name",
            ):
                value = telemetry.get(key)
                if value in ("", None):
                    continue
                telemetry_rows.append(
                    f"<tr><th>{html.escape(key)}</th><td>{html.escape(_fmt_metric(value))}</td></tr>"
                )
        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>MicroSeg Batch Results Report</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:20px;background:#f4f6f8;color:#162029;}"
            "table{border-collapse:collapse;width:100%;margin:14px 0;background:#fff;}"
            "th,td{border:1px solid #d6dde4;padding:6px;text-align:left;font-size:12px;}"
            "th{background:#eef2f6;}"
            ".cards{display:flex;gap:12px;flex-wrap:wrap;}"
            ".card{background:#fff;border:1px solid #d6dde4;padding:10px;min-width:220px;}"
            "</style></head><body>"
            "<h1>MicroSeg Batch Results Report</h1>"
            "<div class='cards'>"
            f"<div class='card'><b>Batch ID</b><div>{html.escape(str(payload.get('batch_id', '')))}</div></div>"
            f"<div class='card'><b>Run Count</b><div>{html.escape(str(payload.get('run_count', 0)))}</div></div>"
            f"<div class='card'><b>Annotator</b><div>{html.escape(str(payload.get('annotator', '')))}</div></div>"
            f"<div class='card'><b>Per-run packages</b><div><a href='runs/'>Open runs folder</a></div></div>"
            "</div>"
            + (
                "<h2>Batch Telemetry</h2>"
                "<table><tbody>"
                + ("".join(telemetry_rows) if telemetry_rows else "<tr><td>n/a</td><td></td></tr>")
                + "</tbody></table>"
            )
            + "<h2>Aggregate Metrics</h2>"
            "<table><thead><tr><th>Metric</th><th>Count</th><th>Mean</th><th>Median</th><th>Std</th><th>Min</th><th>Max</th></tr></thead>"
            f"<tbody>{''.join(agg_rows) if agg_rows else '<tr><td colspan=7>n/a</td></tr>'}</tbody></table>"
            "<h2>Run Rows</h2>"
            "<table><thead><tr>"
            "<th>Input</th><th>Mask</th><th>Overlay</th><th>Key Stats</th><th>Links</th>"
            + "".join(f"<th>{html.escape(field)}</th>" for field in header_fields)
            + "</tr></thead><tbody>"
            + ("".join(run_rows) if run_rows else "<tr><td>n/a</td></tr>")
            + "</tbody></table></body></html>"
        )

    @staticmethod
    def _write_batch_pdf(*, pdf_path: Path, payload: dict[str, Any]) -> None:
        rows = payload.get("rows", [])
        aggregate = payload.get("aggregate_metrics", [])
        telemetry = payload.get("telemetry", {})
        with PdfPages(pdf_path) as pdf:
            fig_meta = plt.figure(figsize=(11.0, 8.5))
            fig_meta.suptitle("MicroSeg Batch Results", fontsize=15, fontweight="bold")
            lines = [
                f"Batch ID: {payload.get('batch_id', '')}",
                f"Run count: {payload.get('run_count', 0)}",
                f"Annotator: {payload.get('annotator', '')}",
                f"Notes: {payload.get('notes', '')}",
                "",
                "Telemetry:",
            ]
            if isinstance(telemetry, dict):
                for key in (
                    "job_elapsed_human",
                    "job_elapsed_seconds",
                    "throughput_images_per_second",
                    "total_images",
                    "completed_images",
                    "total_steps",
                    "completed_steps",
                    "run_duration_seconds_total",
                    "run_duration_seconds_mean",
                    "run_duration_seconds_min",
                    "run_duration_seconds_max",
                    "batch_completed_utc",
                ):
                    value = telemetry.get(key)
                    if value in ("", None):
                        continue
                    lines.append(f"- {key}: {_fmt_metric(value)}")
            lines.extend(
                [
                    "",
                    "Model counts:",
                ]
            )
            model_counts = payload.get("model_counts", {})
            if isinstance(model_counts, dict):
                for key in sorted(model_counts.keys()):
                    lines.append(f"- {key}: {model_counts.get(key)}")
            lines.append("")
            lines.append("Top aggregate metrics:")
            for row in aggregate[:40] if isinstance(aggregate, list) else []:
                if not isinstance(row, dict):
                    continue
                lines.append(
                    "- {} mean={} median={} std={}".format(
                        row.get("metric", ""),
                        _fmt_metric(row.get("mean", "")),
                        _fmt_metric(row.get("median", "")),
                        _fmt_metric(row.get("std", "")),
                    )
                )
            fig_meta.text(0.05, 0.95, "\n".join(lines), va="top", family="monospace", fontsize=9)
            pdf.savefig(fig_meta)
            plt.close(fig_meta)

            fig_rows = plt.figure(figsize=(11.0, 8.5))
            fig_rows.suptitle("Batch Run Rows", fontsize=13, fontweight="bold")
            lines_rows = []
            for idx, row in enumerate(rows[:100] if isinstance(rows, list) else [], start=1):
                if not isinstance(row, dict):
                    continue
                lines_rows.append(
                    "{}. {} | {} | {} | {}".format(
                        idx,
                        row.get("run_id", ""),
                        row.get("image_name", ""),
                        row.get("model_name", ""),
                        row.get("model_id", ""),
                    )
                )
            fig_rows.text(0.05, 0.95, "\n".join(lines_rows) if lines_rows else "n/a", va="top", family="monospace", fontsize=8)
            pdf.savefig(fig_rows)
            plt.close(fig_rows)
