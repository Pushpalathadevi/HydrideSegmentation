"""In-memory scientific report exports for browser segmentation jobs."""

from __future__ import annotations

import base64
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
from typing import Any
import zipfile

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402

REPORT_SCHEMA_VERSION = "microseg.web_report.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_base_name(value: str) -> str:
    stem = Path(str(value or "micrograph")).stem
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    return clean or "micrograph"


def _decode_png(value: str | None) -> np.ndarray | None:
    if not value:
        return None
    try:
        return np.asarray(Image.open(BytesIO(base64.b64decode(value))).convert("RGB"))
    except Exception:
        return None


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.1f}"
        return f"{value:.4f}"
    return str(value)


def _report_manifest(result: dict[str, Any], *, app_version: str, job_meta: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "software": {"name": "MicroSeg", "version": app_version},
        "job": job_meta,
        "source": result.get("source_name"),
        "model": {
            "id": result.get("model_id"),
            "display_name": result.get("model_display_name"),
        },
        "image": (result.get("manifest") or {}).get("image", {}),
        "quantification": (result.get("manifest") or {}).get("quantification", {}),
        "privacy": result.get("privacy") or (result.get("manifest") or {}).get("privacy", {}),
        "timing": result.get("timing", {}),
        "fn": result.get("fn", {}),
        "metrics": result.get("metrics", {}),
        "analysis_data": result.get("analysis_data", {}),
    }


def build_pdf_report(result: dict[str, Any], *, app_version: str, job_meta: dict[str, Any]) -> bytes:
    """Build a compact, two-page PDF report entirely in memory."""

    images = result.get("images") or {}
    manifest = _report_manifest(result, app_version=app_version, job_meta=job_meta)
    metrics = manifest["metrics"]
    fn = manifest["fn"]
    image_meta = manifest["image"]
    quant = manifest["quantification"]
    buffer = BytesIO()

    with PdfPages(buffer, metadata={
        "Title": f"MicroSeg segmentation report — {result.get('source_name', 'micrograph')}",
        "Author": "MicroSeg",
        "Subject": "Microstructural segmentation and hydride quantification",
        "Keywords": "MicroSeg, segmentation, hydrides, Fn",
    }) as pdf:
        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        grid = fig.add_gridspec(3, 2, height_ratios=[0.32, 1.0, 0.32], hspace=0.24, wspace=0.08)
        header = fig.add_subplot(grid[0, :]); header.axis("off")
        header.text(0, 0.9, "MicroSeg — Detailed Segmentation Report", fontsize=18, weight="bold", color="#123f5a", transform=header.transAxes)
        header.text(0, 0.54, str(result.get("source_name", "micrograph")), fontsize=11, weight="bold", transform=header.transAxes)
        meta_line = (
            f"Software v{app_version}  |  Model: {result.get('model_display_name', result.get('model_id', '—'))} "
            f"[{result.get('model_id', '—')}]  |  Generated: {manifest['generated_utc']}"
        )
        header.text(0, 0.23, meta_line, fontsize=8.5, color="#4f6272", transform=header.transAxes)
        header.plot([0, 1], [0.06, 0.06], color="#2d789e", linewidth=1.5, transform=header.transAxes)

        for column, (key, title) in enumerate((("input_png_b64", "Input micrograph"), ("mask_png_b64", "Predicted mask"))):
            ax = fig.add_subplot(grid[1, column]); ax.axis("off"); ax.set_title(title, fontsize=11, weight="bold", pad=6)
            image = _decode_png(images.get(key))
            if image is not None:
                ax.imshow(image)
            else:
                ax.text(0.5, 0.5, "View unavailable", ha="center", va="center", color="#777777")

        summary = fig.add_subplot(grid[2, :]); summary.axis("off")
        left = (
            f"Fn (length-weighted): {_fmt(fn.get('fn_length_weighted'))}\n"
            f"Fn (count-based): {_fmt(fn.get('fn_count'))}\n"
            f"Hydrides measured: {_fmt(metrics.get('hydride_count', fn.get('count_denominator')))}"
        )
        middle = (
            f"Area fraction: {_fmt(metrics.get('hydride_area_fraction', metrics.get('area_fraction')))}\n"
            f"Mean orientation: {_fmt(metrics.get('orientation_mean_deg'))}°\n"
            f"Mean feature area: {_fmt(metrics.get('size_mean_pixels'))} px"
        )
        right = (
            f"Processed size: {_fmt(image_meta.get('width'))} × {_fmt(image_meta.get('height'))} px\n"
            f"Fn threshold: {_fmt(quant.get('fn_angle_threshold_deg'))}°\n"
            f"Total processing time: {_fmt((manifest.get('timing') or {}).get('total_seconds'))} s"
        )
        for x, text_value in ((0.0, left), (0.35, middle), (0.69, right)):
            summary.text(x, 0.94, text_value, va="top", fontsize=9, linespacing=1.45)
        summary.text(0, 0.02, "Scientific-use note: verify the predicted mask against the input before interpretation; report the model and quantification settings.", fontsize=7.8, color="#6b4b16")
        pdf.savefig(fig); plt.close(fig)

        fig = plt.figure(figsize=(11.69, 8.27), facecolor="white")
        grid = fig.add_gridspec(2, 2, height_ratios=[1.08, 0.92], hspace=0.26, wspace=0.16)
        for cell, key, title in (
            ((0, 0), "overlay_png_b64", "Segmentation overlay"),
            ((0, 1), "orientation_map_png_b64", "Feature orientation map"),
            ((1, 0), "size_histogram_png_b64", "Feature-size distribution"),
            ((1, 1), "angle_histogram_png_b64", "Orientation distribution"),
        ):
            ax = fig.add_subplot(grid[cell]); ax.axis("off"); ax.set_title(title, fontsize=10, weight="bold", pad=5)
            image = _decode_png(images.get(key))
            if image is not None:
                ax.imshow(image)
            else:
                ax.text(0.5, 0.5, "Not generated for this run", ha="center", va="center", color="#777777")
        fig.suptitle("Quality control and distributions", x=0.04, ha="left", fontsize=16, weight="bold", color="#123f5a")
        fig.text(0.04, 0.015, f"Report schema {REPORT_SCHEMA_VERSION}  |  Job {job_meta.get('job_id', '—')}  |  Results retained only in server memory.", fontsize=7.5, color="#4f6272")
        pdf.savefig(fig); plt.close(fig)

    return buffer.getvalue()


def build_excel_workbook(result: dict[str, Any], *, app_version: str, job_meta: dict[str, Any]) -> bytes:
    """Build a formatted XLSX workbook with source data and editable charts."""

    try:
        import xlsxwriter
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("XLSX export requires the xlsxwriter package") from exc

    manifest = _report_manifest(result, app_version=app_version, job_meta=job_meta)
    analysis = manifest.get("analysis_data") or {}
    buffer = BytesIO()
    workbook = xlsxwriter.Workbook(buffer, {"in_memory": True})
    title = workbook.add_format({"bold": True, "font_size": 16, "font_color": "#123F5A"})
    section = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#12557C", "border": 1})
    header = workbook.add_format({"bold": True, "font_color": "#FFFFFF", "bg_color": "#2D789E", "border": 1})
    cell = workbook.add_format({"border": 1, "border_color": "#D3DBE3"})
    number = workbook.add_format({"border": 1, "border_color": "#D3DBE3", "num_format": "0.0000"})
    note = workbook.add_format({"font_color": "#5A6B7C", "italic": True, "text_wrap": True})

    summary = workbook.add_worksheet("Summary")
    summary.set_column("A:A", 30); summary.set_column("B:B", 54)
    summary.write("A1", "MicroSeg Detailed Results", title)
    summary.write("A3", "Field", section); summary.write("B3", "Value", section)
    summary_rows = [
        ("Software version", app_version), ("Source", manifest.get("source")),
        ("Model", manifest.get("model", {}).get("display_name")), ("Model ID", manifest.get("model", {}).get("id")),
        ("Generated UTC", manifest.get("generated_utc")), ("Job ID", job_meta.get("job_id")),
        ("Fn, length-weighted", manifest.get("fn", {}).get("fn_length_weighted")),
        ("Fn, count-based", manifest.get("fn", {}).get("fn_count")),
        ("Fn angle threshold (deg)", manifest.get("quantification", {}).get("fn_angle_threshold_deg")),
        ("Minimum feature size (px)", manifest.get("quantification", {}).get("min_feature_pixels")),
        ("Processing time (s)", manifest.get("timing", {}).get("total_seconds")),
    ]
    for row, (label, value) in enumerate(summary_rows, 3):
        summary.write(row, 0, label, cell); summary.write(row, 1, value, number if isinstance(value, (int, float)) else cell)
    summary.write("A16", "Interpretation note", section); summary.merge_range("B16:B18", "Verify the predicted mask against the input image before scientific interpretation. Record the model ID, software version, and quantification settings with reported values.", note)
    summary.freeze_panes(3, 0)

    metrics_sheet = workbook.add_worksheet("Metrics")
    metrics_sheet.set_column("A:A", 38); metrics_sheet.set_column("B:B", 18)
    metrics_sheet.write_row(0, 0, ["Metric", "Value"], header)
    metric_items = sorted((manifest.get("metrics") or {}).items())
    for row, (key, value) in enumerate(metric_items, 1):
        metrics_sheet.write(row, 0, key, cell); metrics_sheet.write(row, 1, value, number if isinstance(value, (int, float)) else cell)
    metrics_sheet.autofilter(0, 0, max(1, len(metric_items)), 1); metrics_sheet.freeze_panes(1, 0)

    features = workbook.add_worksheet("Features")
    features.set_column("A:E", 20); features.write_row(0, 0, ["Feature", "Area (px)", "Length (px)", "Orientation (deg)", "Counted toward Fn"], header)
    sizes = analysis.get("sizes_px") or []; lengths = analysis.get("lengths_px") or []
    orientations = analysis.get("orientations_deg") or []; selected = analysis.get("fn_exceeding_angle") or []
    feature_count = max(len(sizes), len(lengths), len(orientations), len(selected))
    for index in range(feature_count):
        values = [index + 1, sizes[index] if index < len(sizes) else None, lengths[index] if index < len(lengths) else None, orientations[index] if index < len(orientations) else None, selected[index] if index < len(selected) else None]
        for column, value in enumerate(values):
            features.write(index + 1, column, value, number if isinstance(value, float) else cell)
    features.autofilter(0, 0, max(1, feature_count), 4); features.freeze_panes(1, 0)

    hist = workbook.add_worksheet("Histograms")
    hist.set_column("A:F", 22)
    hist.write_row(0, 0, ["Orientation lower (deg)", "Orientation upper (deg)", "Count", "Size lower (px)", "Size upper (px)", "Count"], header)
    orient_hist = analysis.get("orientation_histogram") or {}; size_hist = analysis.get("size_histogram") or {}
    orient_counts = orient_hist.get("counts") or []; orient_edges = orient_hist.get("bin_edges_deg") or []
    size_counts = size_hist.get("counts") or []; size_edges = size_hist.get("bin_edges_px") or []
    rows = max(len(orient_counts), len(size_counts))
    for index in range(rows):
        values = [
            orient_edges[index] if index < len(orient_counts) and index < len(orient_edges) else None,
            orient_edges[index + 1] if index < len(orient_counts) and index + 1 < len(orient_edges) else None,
            orient_counts[index] if index < len(orient_counts) else None,
            size_edges[index] if index < len(size_counts) and index < len(size_edges) else None,
            size_edges[index + 1] if index < len(size_counts) and index + 1 < len(size_edges) else None,
            size_counts[index] if index < len(size_counts) else None,
        ]
        for column, value in enumerate(values): hist.write(index + 1, column, value, number if isinstance(value, float) else cell)
    if orient_counts:
        chart = workbook.add_chart({"type": "column"}); chart.add_series({"name": "Orientation count", "categories": ["Histograms", 1, 0, len(orient_counts), 0], "values": ["Histograms", 1, 2, len(orient_counts), 2], "fill": {"color": "#F28E2B"}}); chart.set_title({"name": "Orientation distribution"}); chart.set_x_axis({"name": "Orientation bin lower edge (deg)"}); chart.set_y_axis({"name": "Count"}); hist.insert_chart("H2", chart)
    if size_counts:
        chart = workbook.add_chart({"type": "column"}); chart.add_series({"name": "Feature count", "categories": ["Histograms", 1, 3, len(size_counts), 3], "values": ["Histograms", 1, 5, len(size_counts), 5], "fill": {"color": "#4E79A7"}}); chart.set_title({"name": "Feature-size distribution"}); chart.set_x_axis({"name": "Area bin lower edge (px)"}); chart.set_y_axis({"name": "Count"}); hist.insert_chart("H19", chart)
    hist.freeze_panes(1, 0)

    metadata = workbook.add_worksheet("Metadata")
    metadata.set_column("A:A", 34); metadata.set_column("B:B", 80)
    metadata.write_row(0, 0, ["Key", "JSON value"], header)
    metadata_rows = [(key, json.dumps(value, ensure_ascii=False, sort_keys=True)) for key, value in manifest.items() if key not in {"metrics", "analysis_data"}]
    for row, (key, value) in enumerate(metadata_rows, 1): metadata.write(row, 0, key, cell); metadata.write(row, 1, value, cell)
    metadata.freeze_panes(1, 0)

    workbook.set_properties({"title": "MicroSeg detailed segmentation data", "author": "MicroSeg", "comments": REPORT_SCHEMA_VERSION})
    workbook.close()
    return buffer.getvalue()


def build_report_bundle(result: dict[str, Any], *, app_version: str, job_meta: dict[str, Any]) -> bytes:
    """Build a ZIP containing the PDF, source data workbook, images, and JSON manifest."""

    base = _safe_base_name(str(result.get("source_name", "micrograph")))
    report_pdf = build_pdf_report(result, app_version=app_version, job_meta=job_meta)
    workbook = build_excel_workbook(result, app_version=app_version, job_meta=job_meta)
    manifest = _report_manifest(result, app_version=app_version, job_meta=job_meta)
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(f"{base}_detailed_report.pdf", report_pdf)
        archive.writestr(f"{base}_scientific_data.xlsx", workbook)
        archive.writestr(f"{base}_results.json", json.dumps(manifest, indent=2, ensure_ascii=False))
        for key, encoded in sorted((result.get("images") or {}).items()):
            if not encoded:
                continue
            filename = key.removesuffix("_png_b64") + ".png"
            archive.writestr(f"images/{filename}", base64.b64decode(encoded))
        archive.writestr("README.txt", "MicroSeg scientific result package\n\nThe PDF is the compact report. The XLSX workbook contains scalar metrics, per-feature measurements, histogram source data, and editable charts. PNG files are the individual rendered views. results.json is the machine-readable provenance record.\n")
    return buffer.getvalue()


def report_download_name(result: dict[str, Any], suffix: str) -> str:
    """Return a safe attachment filename for one report artifact."""

    return f"{_safe_base_name(str(result.get('source_name', 'micrograph')))}_{suffix}"
