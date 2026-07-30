"""Segmentation execution for the intranet web application.

Requests are routed through the same :class:`SegmentationPipeline` the desktop
GUI and the CLI use, so a result produced in the browser matches a result
produced on the workstation for the same image, model, and parameters.

Uploaded bytes are decoded, validated, preprocessed, and inferred entirely in
memory. Nothing is persisted by the web application.
"""

from __future__ import annotations

import logging
import threading
import time
import hashlib
import io
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from hydride_segmentation.legacy_api import DEFAULT_CONVENTIONAL_PARAMS
from hydride_segmentation.microseg_adapter import run_pipeline_array
from src.microseg.evaluation.hydride_statistics import (
    HydrideVisualizationConfig,
    compute_hydride_statistics,
    render_fn_debug_visualizations,
    render_hydride_visualizations,
)
from src.microseg.utils import image_to_png_base64

_LOGGER = logging.getLogger(__name__)

#: Upload extensions the server accepts.
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "tif", "tiff", "bmp"})

#: Pillow format names mapped onto accepted extensions.
_FORMAT_ALIASES = {"jpeg": "jpg", "tif": "tiff"}


class SegmentationRequestError(ValueError):
    """Raised when a browser request cannot be served as submitted."""


@dataclass
class ConventionalControl:
    """One conventional parameter exposed in the browser, with in-app help."""

    key: str
    label: str
    kind: str
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    help_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of this control."""

        return {
            "key": self.key,
            "label": self.label,
            "kind": self.kind,
            "default": self.default,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "step": self.step,
            "help_text": self.help_text,
        }


#: Conventional controls offered in the browser. Defaults mirror the desktop app.
CONVENTIONAL_CONTROLS: tuple[ConventionalControl, ...] = (
    ConventionalControl(
        key="clahe_clip_limit",
        label="CLAHE clip limit",
        kind="float",
        default=float(DEFAULT_CONVENTIONAL_PARAMS["clahe"]["clip_limit"]),
        minimum=0.1,
        maximum=20.0,
        step=0.1,
        help_text=(
            "Controls how aggressively local contrast is stretched before thresholding. "
            "Raise it when hydrides are faint against the matrix. Too high amplifies noise "
            "and polishing scratches into false detections."
        ),
    ),
    ConventionalControl(
        key="clahe_tile_grid",
        label="CLAHE tile grid",
        kind="int",
        default=int(DEFAULT_CONVENTIONAL_PARAMS["clahe"]["tile_grid_size"][0]),
        minimum=1,
        maximum=32,
        step=1,
        help_text=(
            "Number of tiles per image side used for local contrast equalisation. "
            "Larger values adapt to finer illumination variation; smaller values behave "
            "more like a global contrast stretch."
        ),
    ),
    ConventionalControl(
        key="adaptive_block_size",
        label="Adaptive threshold block size",
        kind="odd_int",
        default=int(DEFAULT_CONVENTIONAL_PARAMS["adaptive"]["block_size"]),
        minimum=3,
        maximum=255,
        step=2,
        help_text=(
            "Neighbourhood size, in pixels, used to compute the local threshold. Must be odd. "
            "Increase it for thick or widely spaced hydrides, decrease it for fine features."
        ),
    ),
    ConventionalControl(
        key="adaptive_offset",
        label="Adaptive threshold offset",
        kind="int",
        default=int(DEFAULT_CONVENTIONAL_PARAMS["adaptive"]["C"]),
        minimum=-50,
        maximum=50,
        step=1,
        help_text=(
            "Constant subtracted from the local mean. Higher values make the threshold "
            "stricter and detect less; lower values detect more, including more noise."
        ),
    ),
    ConventionalControl(
        key="morph_kernel",
        label="Morphology kernel",
        kind="int",
        default=int(DEFAULT_CONVENTIONAL_PARAMS["morph"]["kernel_size"][0]),
        minimum=1,
        maximum=31,
        step=2,
        help_text=(
            "Size of the structuring element used to close small gaps along hydride traces. "
            "Larger kernels join broken segments but also merge features that are genuinely separate."
        ),
    ),
    ConventionalControl(
        key="morph_iterations",
        label="Morphology iterations",
        kind="int",
        default=int(DEFAULT_CONVENTIONAL_PARAMS["morph"]["iterations"]),
        minimum=0,
        maximum=10,
        step=1,
        help_text=(
            "How many times the morphological closing is applied. Zero disables it. "
            "Start at zero and increase only if hydride traces are visibly fragmented."
        ),
    ),
    ConventionalControl(
        key="area_threshold",
        label="Minimum feature area (px)",
        kind="int",
        default=int(DEFAULT_CONVENTIONAL_PARAMS["area_threshold"]),
        minimum=0,
        maximum=100000,
        step=10,
        help_text=(
            "Connected regions smaller than this are discarded as noise. Raise it to clean up "
            "speckle; lower it when you need to keep genuinely small precipitates."
        ),
    ),
)


#: Quantification controls. These act on the segmented mask, so they apply to
#: the conventional and trained routes alike and stay visible for both.
QUANTIFICATION_CONTROLS: tuple[ConventionalControl, ...] = (
    ConventionalControl(
        key="fn_angle_threshold_deg",
        label="Fn angle threshold (deg)",
        kind="float",
        default=45.0,
        minimum=0.0,
        maximum=90.0,
        step=1.0,
        help_text=(
            "A hydride counts towards Fn when its angle from the horizontal reaches this "
            "threshold. 45 degrees is the common convention for separating radial from "
            "circumferential hydrides. Change it only if your specimen convention differs, "
            "and report the value you used alongside the number."
        ),
    ),
    ConventionalControl(
        key="min_feature_pixels",
        label="Minimum feature size for statistics (px)",
        kind="int",
        default=1,
        minimum=1,
        maximum=100000,
        step=1,
        help_text=(
            "Connected features smaller than this are excluded from Fn and the distributions. "
            "Raising it stops single-pixel noise from dominating the count-based Fn. It does "
            "not change the mask itself, only which features are measured."
        ),
    ),
    ConventionalControl(
        key="orientation_bins",
        label="Orientation histogram bins",
        kind="int",
        default=18,
        minimum=2,
        maximum=180,
        step=1,
        help_text="Number of bins in the orientation distribution chart. Purely a display choice.",
    ),
    ConventionalControl(
        key="size_bins",
        label="Size histogram bins",
        kind="int",
        default=20,
        minimum=2,
        maximum=200,
        step=1,
        help_text="Number of bins in the size distribution chart. Purely a display choice.",
    ),
)

#: Metric groups used to present the measurements table in a readable order.
METRIC_GROUPS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "fn",
        "Radial hydride fraction (Fn)",
        (
            "fn_count",
            "fn_length_weighted",
            "fn_count_numerator",
            "fn_count_denominator",
            "fn_length_numerator_px",
            "fn_length_denominator_px",
            "fn_angle_threshold_deg",
            "fn_excluded_small_features",
        ),
    ),
    (
        "coverage",
        "Coverage",
        (
            "hydride_area_fraction",
            "hydride_area_fraction_percent",
            "hydride_total_area_pixels",
            "hydride_count",
            "hydride_density_per_megapixel",
        ),
    ),
    (
        "orientation",
        "Orientation",
        (
            "orientation_mean_deg",
            "orientation_median_deg",
            "orientation_std_deg",
            "orientation_p10_deg",
            "orientation_p90_deg",
            "orientation_min_deg",
            "orientation_max_deg",
            "orientation_alignment_index",
            "orientation_entropy_bits",
        ),
    ),
    (
        "size",
        "Feature size",
        (
            "size_mean_pixels",
            "size_median_pixels",
            "size_std_pixels",
            "size_p10_pixels",
            "size_p90_pixels",
            "size_min_pixels",
            "size_max_pixels",
            "equivalent_diameter_mean_px",
            "equivalent_diameter_median_px",
            "equivalent_diameter_std_px",
            "equivalent_diameter_min_px",
            "equivalent_diameter_max_px",
        ),
    ),
)


@dataclass
class PreparedImage:
    """An uploaded image after validation and optional downscaling."""

    array: np.ndarray
    original_width: int
    original_height: int
    width: int
    height: int
    downscaled: bool
    scale: float
    source_format: str
    byte_size: int
    sha256: str
    original_mode: str
    frame_count: int

    def to_metadata(self) -> dict[str, Any]:
        """Return a JSON-serializable description of the prepared image."""

        return {
            "original_width": int(self.original_width),
            "original_height": int(self.original_height),
            "width": int(self.width),
            "height": int(self.height),
            "downscaled": bool(self.downscaled),
            "scale": round(float(self.scale), 4),
            "source_format": self.source_format,
            "byte_size": int(self.byte_size),
            "sha256": self.sha256,
            "original_mode": self.original_mode,
            "frame_count": int(self.frame_count),
            "input_transport": "memory",
        }


def validate_upload_name(filename: str) -> str:
    """Validate an upload filename and return its normalized extension.

    Parameters
    ----------
    filename:
        Client-supplied filename.

    Returns
    -------
    str
        Lowercase extension without the leading dot.

    Raises
    ------
    SegmentationRequestError
        If the name has no extension or the extension is not accepted.
    """

    name = str(filename or "").strip()
    if not name or "." not in name:
        raise SegmentationRequestError(
            "The file needs a recognisable image extension. Accepted types: "
            + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
    ext = name.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise SegmentationRequestError(
            f"'{ext}' files are not supported. Accepted types: " + ", ".join(sorted(ALLOWED_EXTENSIONS))
        )
    return ext


def prepare_image(
    data: bytes,
    *,
    max_long_side_px: int = 0,
    max_image_pixels: int = 40_000_000,
    expected_extension: str = "",
) -> PreparedImage:
    """Decode uploaded bytes and downscale them when they exceed the limit.

    Parameters
    ----------
    data:
        Raw uploaded bytes.
    max_long_side_px:
        Longest allowed image side. ``0`` disables downscaling.

    Returns
    -------
    PreparedImage
        Decoded RGB image plus the metadata shown to the user.

    Raises
    ------
    SegmentationRequestError
        If the bytes cannot be decoded as a supported image.
    """

    if not data:
        raise SegmentationRequestError("The uploaded file is empty.")

    try:
        with Image.open(io.BytesIO(data)) as probe:
            source_format = str(probe.format or "").strip().lower()
            original_mode = str(probe.mode or "")
            frame_count = int(getattr(probe, "n_frames", 1) or 1)
            width, height = int(probe.width), int(probe.height)
            if width <= 0 or height <= 0:
                raise SegmentationRequestError("The image has invalid zero-sized dimensions.")
            pixel_count = width * height
            if int(max_image_pixels) > 0 and pixel_count > int(max_image_pixels):
                raise SegmentationRequestError(
                    f"The image contains {pixel_count:,} pixels, above the "
                    f"{int(max_image_pixels):,}-pixel safety limit."
                )
            if frame_count != 1:
                raise SegmentationRequestError(
                    "Multi-frame or animated images are not accepted. Export one micrograph frame "
                    "as PNG, JPEG, TIFF, or BMP and try again."
                )
            probe.verify()
        with Image.open(io.BytesIO(data)) as handle:
            handle.load()
            rgb = handle.convert("RGB")
            array = np.asarray(rgb, dtype=np.uint8)
    except UnidentifiedImageError as exc:
        raise SegmentationRequestError(
            "The file could not be read as an image. It may be corrupt or not an image at all."
        ) from exc
    except Exception as exc:
        raise SegmentationRequestError(f"The image could not be decoded: {exc}") from exc

    normalized_format = _FORMAT_ALIASES.get(source_format, source_format)
    if normalized_format and normalized_format not in ALLOWED_EXTENSIONS:
        raise SegmentationRequestError(
            f"The file contents look like '{normalized_format}', which is not a supported image type."
        )
    normalized_expected = _FORMAT_ALIASES.get(
        str(expected_extension or "").strip().lower(),
        str(expected_extension or "").strip().lower(),
    )
    if normalized_expected and normalized_format != normalized_expected:
        raise SegmentationRequestError(
            f"The filename says '{normalized_expected}' but the file contents are "
            f"'{normalized_format}'. Rename or re-export the image so its extension matches its format."
        )

    original_height, original_width = int(array.shape[0]), int(array.shape[1])
    long_side = max(original_width, original_height)
    limit = int(max_long_side_px or 0)

    if limit and long_side > limit:
        scale = float(limit) / float(long_side)
        new_size = (max(1, int(round(original_width * scale))), max(1, int(round(original_height * scale))))
        array = np.asarray(Image.fromarray(array).resize(new_size, Image.BILINEAR), dtype=np.uint8)
        return PreparedImage(
            array=array,
            original_width=original_width,
            original_height=original_height,
            width=int(array.shape[1]),
            height=int(array.shape[0]),
            downscaled=True,
            scale=scale,
            source_format=normalized_format or "unknown",
            byte_size=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
            original_mode=original_mode,
            frame_count=frame_count,
        )

    return PreparedImage(
        array=array,
        original_width=original_width,
        original_height=original_height,
        width=original_width,
        height=original_height,
        downscaled=False,
        scale=1.0,
        source_format=normalized_format or "unknown",
        byte_size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        original_mode=original_mode,
        frame_count=frame_count,
    )


def build_conventional_params(form: dict[str, Any]) -> dict[str, Any]:
    """Translate browser form values into conventional pipeline parameters.

    Parameters
    ----------
    form:
        Raw form mapping from the request.

    Returns
    -------
    dict
        Parameters shaped like ``DEFAULT_CONVENTIONAL_PARAMS``.

    Raises
    ------
    SegmentationRequestError
        If a submitted value is not numeric or violates a documented constraint.
    """

    def _number(key: str, default: float) -> float:
        raw = form.get(key, None)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            control = next((item for item in CONVENTIONAL_CONTROLS if item.key == key), None)
            label = control.label if control else key
            raise SegmentationRequestError(f"{label} must be a number, got {raw!r}.") from exc

    clip_limit = _number("clahe_clip_limit", DEFAULT_CONVENTIONAL_PARAMS["clahe"]["clip_limit"])
    if clip_limit <= 0:
        raise SegmentationRequestError("CLAHE clip limit must be greater than zero.")

    tile_grid = int(_number("clahe_tile_grid", DEFAULT_CONVENTIONAL_PARAMS["clahe"]["tile_grid_size"][0]))
    if tile_grid < 1:
        raise SegmentationRequestError("CLAHE tile grid must be at least 1.")

    block_size = int(_number("adaptive_block_size", DEFAULT_CONVENTIONAL_PARAMS["adaptive"]["block_size"]))
    if block_size < 3 or block_size % 2 == 0:
        raise SegmentationRequestError("Adaptive threshold block size must be an odd number of at least 3.")

    offset = int(_number("adaptive_offset", DEFAULT_CONVENTIONAL_PARAMS["adaptive"]["C"]))

    morph_kernel = int(_number("morph_kernel", DEFAULT_CONVENTIONAL_PARAMS["morph"]["kernel_size"][0]))
    if morph_kernel < 1:
        raise SegmentationRequestError("Morphology kernel must be at least 1.")

    morph_iterations = int(_number("morph_iterations", DEFAULT_CONVENTIONAL_PARAMS["morph"]["iterations"]))
    if morph_iterations < 0:
        raise SegmentationRequestError("Morphology iterations cannot be negative.")

    area_threshold = int(_number("area_threshold", DEFAULT_CONVENTIONAL_PARAMS["area_threshold"]))
    if area_threshold < 0:
        raise SegmentationRequestError("Minimum feature area cannot be negative.")

    return {
        "clahe": {"clip_limit": float(clip_limit), "tile_grid_size": [tile_grid, tile_grid]},
        "adaptive": {"block_size": block_size, "C": offset},
        "morph": {"kernel_size": [morph_kernel, morph_kernel], "iterations": morph_iterations},
        "area_threshold": area_threshold,
        "crop": False,
        "crop_percent": 0,
    }


def build_quantification_config(form: dict[str, Any]) -> HydrideVisualizationConfig:
    """Translate browser form values into a hydride quantification configuration.

    Parameters
    ----------
    form:
        Raw form mapping from the request.

    Returns
    -------
    HydrideVisualizationConfig
        Configuration controlling Fn classification and distribution charts.

    Raises
    ------
    SegmentationRequestError
        If a submitted value is not numeric or falls outside its valid range.
    """

    def _number(key: str, default: float) -> float:
        raw = form.get(key, None)
        if raw is None or str(raw).strip() == "":
            return float(default)
        try:
            return float(raw)
        except (TypeError, ValueError) as exc:
            control = next((item for item in QUANTIFICATION_CONTROLS if item.key == key), None)
            label = control.label if control else key
            raise SegmentationRequestError(f"{label} must be a number, got {raw!r}.") from exc

    threshold = _number("fn_angle_threshold_deg", 45.0)
    if not 0.0 <= threshold <= 90.0:
        raise SegmentationRequestError(
            "Fn angle threshold must be between 0 and 90 degrees, because orientations are "
            "measured as the angle from horizontal and folded into that range."
        )

    min_feature_pixels = int(_number("min_feature_pixels", 1))
    if min_feature_pixels < 1:
        raise SegmentationRequestError("Minimum feature size for statistics must be at least 1 pixel.")

    orientation_bins = int(_number("orientation_bins", 18))
    if orientation_bins < 2:
        raise SegmentationRequestError("Orientation histogram bins must be at least 2.")

    size_bins = int(_number("size_bins", 20))
    if size_bins < 2:
        raise SegmentationRequestError("Size histogram bins must be at least 2.")

    return HydrideVisualizationConfig(
        orientation_bins=orientation_bins,
        size_bins=size_bins,
        min_feature_pixels=min_feature_pixels,
        include_fn_metrics=True,
        fn_angle_threshold_deg=threshold,
    )


def summarize_fn(metrics: dict[str, Any]) -> dict[str, Any]:
    """Extract the Fn headline summary shown above the measurements table.

    Parameters
    ----------
    metrics:
        Scalar metric mapping produced by the analysis.

    Returns
    -------
    dict
        Mapping with ``available`` plus the count-based and length-weighted Fn
        values and the inputs they were derived from.
    """

    if "fn_count" not in metrics:
        return {"available": False}

    denominator = int(metrics.get("fn_count_denominator", 0) or 0)
    return {
        "available": True,
        "fn_count": float(metrics.get("fn_count", 0.0)),
        "fn_length_weighted": float(metrics.get("fn_length_weighted", 0.0)),
        "count_numerator": int(metrics.get("fn_count_numerator", 0) or 0),
        "count_denominator": denominator,
        "length_numerator_px": float(metrics.get("fn_length_numerator_px", 0.0)),
        "length_denominator_px": float(metrics.get("fn_length_denominator_px", 0.0)),
        "angle_threshold_deg": float(metrics.get("fn_angle_threshold_deg", 45.0)),
        "excluded_small_features": int(metrics.get("fn_excluded_small_features", 0) or 0),
        "measured_features": denominator,
    }


def group_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Arrange scalar metrics into labelled groups for display.

    Parameters
    ----------
    metrics:
        Scalar metric mapping produced by the analysis.

    Returns
    -------
    list of dict
        Groups in presentation order. Metrics not belonging to a known group are
        collected into a trailing "Other" group so nothing is silently dropped.
    """

    remaining = dict(metrics)
    groups: list[dict[str, Any]] = []
    for key, title, metric_keys in METRIC_GROUPS:
        entries = [
            {"key": name, "value": remaining.pop(name)}
            for name in metric_keys
            if name in remaining
        ]
        if entries:
            groups.append({"key": key, "title": title, "metrics": entries})
    if remaining:
        groups.append(
            {
                "key": "other",
                "title": "Other",
                "metrics": [{"key": name, "value": remaining[name]} for name in sorted(remaining)],
            }
        )
    return groups


def run_web_segmentation(
    prepared: PreparedImage,
    *,
    model_id: str,
    params: dict[str, Any] | None = None,
    include_analysis: bool = True,
    quantification: HydrideVisualizationConfig | None = None,
    include_fn_classification: bool = False,
    source_name: str = "in-memory image",
    progress_hook: Callable[[str, int, str], None] | None = None,
) -> dict[str, Any]:
    """Run one segmentation request and return a browser-ready payload.

    Parameters
    ----------
    prepared:
        Validated image from :func:`prepare_image`.
    model_id:
        Registry model identifier to run.
    params:
        Backend parameters. Conventional runs receive threshold settings; trained
        runs receive device settings.
    include_analysis:
        Whether to compute orientation and distribution figures.
    quantification:
        Fn and distribution settings. Defaults are used when omitted.
    include_fn_classification:
        Whether to also render the annotated Fn classification view, which is
        noticeably more expensive than the standard figures.

    Returns
    -------
    dict
        Mapping with ``metrics``, ``metric_groups``, ``fn``, ``images`` (base64
        PNG strings), ``manifest`` and ``timing`` keys.
    """

    started = time.perf_counter()
    config = quantification or HydrideVisualizationConfig()

    # The pipeline's built-in analysis cannot take a quantification config, so
    # inference runs without it and analysis is computed below with the exact
    # settings submitted by the user. The source image never leaves memory.
    def pipeline_progress(stage: str, percent: int, message: str) -> None:
        if progress_hook is None:
            return
        mapped = min(75, max(10, int(round(float(percent) * 0.75))))
        mapped_stage = "inference" if str(stage).lower() == "complete" else stage
        mapped_message = (
            "Core segmentation completed; preparing scientific analysis."
            if str(stage).lower() == "complete"
            else message
        )
        progress_hook(mapped_stage, mapped, mapped_message)

    result = run_pipeline_array(
        prepared.array,
        source_name=source_name,
        model_id=model_id,
        params=dict(params or {}),
        include_analysis=False,
        progress_hook=pipeline_progress,
    )

    predictor_seconds = max(0.0, time.perf_counter() - started)
    images = dict(result.images_b64)
    images.setdefault("input_png_b64", image_to_png_base64(prepared.array))

    metrics = dict(result.metrics or {})
    mask = np.asarray(result.mask)
    if mask.size:
        metrics.setdefault("area_fraction", float(np.count_nonzero(mask) / mask.size))

    analysis_started = time.perf_counter()
    if include_analysis and mask.size:
        if progress_hook is not None:
            progress_hook("analysis", 78, "Measuring connected features and orientations.")
        stats = compute_hydride_statistics(
            mask,
            orientation_bins=config.orientation_bins,
            size_bins=config.size_bins,
            min_feature_pixels=config.min_feature_pixels,
            include_fn_metrics=config.include_fn_metrics,
            fn_angle_threshold_deg=config.fn_angle_threshold_deg,
        )
        metrics.update(stats.scalar_metrics)
        if progress_hook is not None:
            progress_hook("analysis", 88, "Rendering the scientific analysis views.")
        visuals = render_hydride_visualizations(stats, config)
        images["orientation_map_png_b64"] = image_to_png_base64(visuals["orientation_map_rgb"])
        images["size_histogram_png_b64"] = image_to_png_base64(visuals["size_distribution_rgb"])
        images["angle_histogram_png_b64"] = image_to_png_base64(visuals["orientation_distribution_rgb"])
        if include_fn_classification:
            if progress_hook is not None:
                progress_hook("analysis", 94, "Rendering the optional Fn audit views.")
            fn_visuals = render_fn_debug_visualizations(stats, base_image=prepared.array)
            images["fn_classification_png_b64"] = image_to_png_base64(fn_visuals["fn_classification_rgb"])
            images["fn_angle_threshold_png_b64"] = image_to_png_base64(
                fn_visuals["fn_angle_distribution_rgb"]
            )
    analysis_seconds = max(0.0, time.perf_counter() - analysis_started)

    manifest = dict(result.manifest or {})
    manifest["image"] = prepared.to_metadata()
    manifest["quantification"] = {
        "fn_angle_threshold_deg": float(config.fn_angle_threshold_deg),
        "min_feature_pixels": int(config.min_feature_pixels),
        "orientation_bins": int(config.orientation_bins),
        "size_bins": int(config.size_bins),
        "include_analysis": bool(include_analysis),
        "include_fn_classification": bool(include_fn_classification),
    }
    manifest["privacy"] = {
        "input_transport": "memory",
        "source_persisted": False,
        "result_persisted": False,
    }

    return {
        "model_id": result.model_id,
        "metrics": metrics,
        "metric_groups": group_metrics(metrics),
        "fn": summarize_fn(metrics),
        "images": images,
        "manifest": manifest,
        "timing": {
            "total_seconds": round(predictor_seconds + analysis_seconds, 3),
            "inference_seconds": round(predictor_seconds, 3),
            "analysis_seconds": round(analysis_seconds, 3),
        },
    }


class JobLimiter:
    """Bounds how many segmentation jobs run at once.

    A CPU-bound job pool keeps the server responsive when several colleagues
    submit images at the same time, instead of letting every request compete for
    the same cores.
    """

    def __init__(self, max_concurrent_jobs: int = 2, timeout_seconds: float = 300.0) -> None:
        self._semaphore = threading.BoundedSemaphore(max(1, int(max_concurrent_jobs)))
        self._timeout = float(timeout_seconds)
        self._active = 0
        self._lock = threading.Lock()

    @property
    def active_jobs(self) -> int:
        """Return how many jobs are currently running."""

        with self._lock:
            return self._active

    def acquire(self) -> bool:
        """Reserve a job slot, waiting up to the configured timeout."""

        acquired = self._semaphore.acquire(timeout=self._timeout)
        if acquired:
            with self._lock:
                self._active += 1
        return acquired

    def release(self) -> None:
        """Return a previously reserved job slot."""

        with self._lock:
            self._active = max(0, self._active - 1)
        try:
            self._semaphore.release()
        except ValueError:  # pragma: no cover - defensive
            pass
