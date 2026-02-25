"""Desktop result-package exporter with JSON/HTML/PDF summaries."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import html
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

from src.microseg.app.desktop_workflow import DesktopRunRecord
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
        return f"{value:.6f}"
    return str(value)


def _save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(to_rgb(image).astype(np.uint8)).save(path)


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

    def visualization_config(self) -> HydrideVisualizationConfig:
        """Return analysis-plot configuration object."""

        return HydrideVisualizationConfig(
            orientation_bins=max(1, int(self.orientation_bins)),
            size_bins=max(1, int(self.size_bins)),
            min_feature_pixels=max(1, int(self.min_feature_pixels)),
            orientation_cmap=str(self.orientation_cmap),
            size_scale=str(self.size_scale),
        )


class DesktopResultExporter:
    """Export segmentation results as a deployment-grade report package."""

    schema_version = "microseg.desktop_results.v1"

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
            "artifacts": {
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
            },
        }

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
        metrics_pred = payload.get("predicted_stats", {}).get("scalar_metrics", {})
        metrics_corr = payload.get("corrected_stats", {}).get("scalar_metrics", {})
        keys = sorted(set(metrics_pred.keys()) | set(metrics_corr.keys()))
        rows = []
        for key in keys:
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(key))}</td>"
                f"<td>{html.escape(_fmt_metric(metrics_pred.get(key, '')))}</td>"
                f"<td>{html.escape(_fmt_metric(metrics_corr.get(key, '')))}</td>"
                "</tr>"
            )

        artifacts = payload.get("artifacts", {})
        image_grid = "\n".join(
            [
                f"<figure><img src='{html.escape(str(artifacts.get(name, '')))}' alt='{html.escape(name)}'/><figcaption>{html.escape(name)}</figcaption></figure>"
                for name in [
                    "input",
                    "predicted_overlay",
                    "corrected_overlay",
                    "predicted_orientation_map",
                    "predicted_size_distribution",
                    "predicted_orientation_distribution",
                    "corrected_orientation_map",
                    "corrected_size_distribution",
                    "corrected_orientation_distribution",
                ]
            ]
        )

        return (
            "<!doctype html><html><head><meta charset='utf-8'/>"
            "<title>MicroSeg Result Report</title>"
            "<style>"
            "body{font-family:Arial,sans-serif;margin:20px;background:#f4f6f8;color:#162029;}"
            "h1,h2{margin:8px 0;}"
            "table{border-collapse:collapse;width:100%;margin:14px 0;background:#fff;}"
            "th,td{border:1px solid #d6dde4;padding:8px;text-align:left;font-size:13px;}"
            "th{background:#eef2f6;}"
            ".meta{display:grid;grid-template-columns:220px 1fr;gap:6px;background:#fff;border:1px solid #d6dde4;padding:10px;}"
            ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;}"
            "figure{background:#fff;border:1px solid #d6dde4;padding:8px;margin:0;}"
            "img{max-width:100%;height:auto;display:block;margin:0 auto;}"
            "figcaption{font-size:12px;color:#2a3946;margin-top:6px;text-align:center;}"
            "</style></head><body>"
            "<h1>MicroSeg Result Report</h1>"
            f"<div class='meta'><div>Run ID</div><div>{html.escape(str(payload.get('run_id', '')))}</div>"
            f"<div>Model</div><div>{html.escape(str(payload.get('model_name', '')))} ({html.escape(str(payload.get('model_id', '')))} )</div>"
            f"<div>Source Image</div><div>{html.escape(str(payload.get('source_image_path', '')))}</div>"
            f"<div>Generated UTC</div><div>{html.escape(str(payload.get('created_utc', '')))}</div>"
            f"<div>Annotator</div><div>{html.escape(str(payload.get('annotator', '')))}</div>"
            f"<div>Notes</div><div>{html.escape(str(payload.get('notes', '')))}</div></div>"
            f"<div style='margin-top:8px;font-size:12px;'>Calibration: "
            f"{html.escape(str(payload.get('spatial_calibration', {}).get('microns_per_pixel', 'None')))} um/px "
            f"({html.escape(str(payload.get('spatial_calibration', {}).get('source', 'none')))})</div>"
            "<h2>Scalar Statistics</h2>"
            "<table><thead><tr><th>Metric</th><th>Predicted</th><th>Corrected</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<h2>Visual Outputs</h2>"
            f"<div class='grid'>{image_grid}</div>"
            "</body></html>"
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
        pred_overlay = mask_overlay(base, (predicted_mask > 0).astype(np.uint8) * 255)
        corr_overlay = mask_overlay(base, (corrected_mask > 0).astype(np.uint8) * 255)
        pred_color = np.array(colorize_index_mask(predicted_mask, DEFAULT_CLASS_MAP))
        corr_color = np.array(colorize_index_mask(corrected_mask, DEFAULT_CLASS_MAP))

        metrics_pred = payload.get("predicted_stats", {}).get("scalar_metrics", {})
        metrics_corr = payload.get("corrected_stats", {}).get("scalar_metrics", {})
        keys = sorted(set(metrics_pred.keys()) | set(metrics_corr.keys()))

        with PdfPages(pdf_path) as pdf:
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
                "Key metrics (predicted -> corrected):",
            ]
            for key in keys:
                lines.append(
                    f"- {key}: {_fmt_metric(metrics_pred.get(key, ''))} -> {_fmt_metric(metrics_corr.get(key, ''))}"
                )
            fig_meta.text(0.05, 0.92, "\n".join(lines), va="top", family="monospace", fontsize=10)
            pdf.savefig(fig_meta)
            plt.close(fig_meta)

            fig_comp, ax = plt.subplots(2, 3, figsize=(11.0, 8.5))
            diff_mask = np.where(corrected_mask != predicted_mask, 255, 0).astype(np.uint8)
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
