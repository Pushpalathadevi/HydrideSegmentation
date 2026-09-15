"""Downloadable encodings of a binary segmentation mask: semantic labels and a display preview.

A mask means one thing and is shown another way:

* ``<stem>_mask_labels.png`` stores **class numbers** -- 0 = background, 1 = hydride -- for model
  training and annotation tools. OnlineAnnotator, for example, keeps ``{0, 1}`` exactly as the
  class numbers they are.
* ``<stem>_mask_preview.png`` stores **display values** -- 0 and 255 -- so the mask is visible in
  any image viewer. It carries no extra information and must be normalized before training.

Both files are written from one label array, so they have identical size and geometry. The
normalization reuses the repository's binary mask rules in
:mod:`src.microseg.corrections.classes`: the canonical ``{0, 255} -> {0, 1}`` of
:func:`~src.microseg.corrections.classes.to_index_mask` and the ``two_value_zero_background``
mode of :func:`~src.microseg.corrections.classes.normalize_binary_index_mask`. A mask with more
than two values, without background 0, or in colour is refused rather than silently binarized.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from src.microseg.corrections.classes import normalize_binary_index_mask, to_index_mask

MASK_DOWNLOAD_SCHEMA = "microseg.mask_download.v1"
PREVIEW_FOREGROUND_VALUE = 255
CLASS_MAP: tuple[dict[str, Any], ...] = (
    {"index": 0, "name": "background"},
    {"index": 1, "name": "hydride"},
)


class MaskEncodingError(ValueError):
    """The mask cannot be written as binary class labels without guessing."""


@dataclass(frozen=True)
class MaskDownloads:
    """The label PNG, preview PNG and metadata for one mask, all from the same label array."""

    base_name: str
    labels: np.ndarray
    preview: np.ndarray
    labels_png: bytes
    preview_png: bytes
    metadata: dict[str, Any]

    @property
    def labels_name(self) -> str:
        return f"{self.base_name}_mask_labels.png"

    @property
    def preview_name(self) -> str:
        return f"{self.base_name}_mask_preview.png"

    @property
    def metadata_name(self) -> str:
        return f"{self.base_name}_mask_metadata.json"

    @property
    def bundle_name(self) -> str:
        return f"{self.base_name}_masks.zip"


def binary_labels(mask: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Normalize a binary mask to ``uint8`` class indices ``{0, 1}``.

    Parameters
    ----------
    mask:
        2-D mask, or an RGB(A) mask whose colour channels are identical (grey saved as RGB).

    Returns
    -------
    tuple[np.ndarray, dict]
        ``(labels, normalization)``: ``labels`` holds only 0 and 1; ``normalization`` records
        the source values, the rule applied and any notes.

    Raises
    ------
    MaskEncodingError
        For a colour mask, a non-integer mask, more than two values, or values without
        background 0 that are not a display foreground.
    """

    arr = np.asarray(mask)
    notes: list[str] = []
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
            notes.append("alpha channel ignored")
        if arr.shape[2] == 3 and np.array_equal(arr[:, :, 0], arr[:, :, 1]) and np.array_equal(arr[:, :, 1], arr[:, :, 2]):
            arr = arr[:, :, 0]
            notes.append("identical RGB channels read as greyscale")
        else:
            raise MaskEncodingError("the mask is a colour image; binary class labels need a single channel")
    if arr.ndim != 2:
        raise MaskEncodingError(f"the mask must be two-dimensional, got shape {arr.shape}")
    if arr.dtype == bool:
        arr = arr.astype(np.uint8)
    if arr.dtype.kind not in "iu":
        if arr.size and not np.all(np.mod(arr, 1) == 0):
            raise MaskEncodingError("the mask holds non-integer values (a probability map); threshold it explicitly first")
        arr = arr.astype(np.int64)

    values = [int(v) for v in np.unique(arr)]
    shown = f"{values[:10]}{' ...' if len(values) > 10 else ''}"
    if len(values) > 2 or (len(values) == 2 and values[0] != 0):
        raise MaskEncodingError(
            f"the mask holds the values {shown}; a binary mask has background 0 and a single foreground value"
        )
    if values and values[-1] > 255:
        index = (arr > 0).astype(np.uint8)
        rule = f"two_value_zero_background: {values[-1]} -> 1"
    else:
        index = to_index_mask(arr.astype(np.uint8))
        if set(np.unique(index).tolist()) <= {0, 1}:
            rule = "canonical {0,255}->{0,1} (to_index_mask)" if 255 in values else "none: values already {0,1}"
        else:
            index = normalize_binary_index_mask(index, mode="two_value_zero_background")
            rule = f"two_value_zero_background: {values[-1]} -> 1"
    if not set(np.unique(index).tolist()) <= {0, 1}:
        raise MaskEncodingError(
            f"the mask holds only the value {values[0]}, which is neither background 0 nor a foreground of 1 or 255"
        )
    return index.astype(np.uint8), {"source_values": values, "rule": rule, "notes": notes}


def preview_values(labels: np.ndarray) -> np.ndarray:
    """Display encoding of ``{0, 1}`` labels: ``{0, 255}``."""

    return np.asarray(labels, dtype=np.uint8) * np.uint8(PREVIEW_FOREGROUND_VALUE)


def decode_mask_png(encoded: str) -> np.ndarray:
    """Decode a base64 PNG mask (as carried in web results) into an array without resampling."""

    with Image.open(io.BytesIO(base64.b64decode(encoded))) as image:
        if image.mode in ("1", "LA"):
            image = image.convert("L")
        elif image.mode in ("P", "PA"):
            image = image.convert("RGB")
        return np.asarray(image)


def label_sha256(labels: np.ndarray) -> str:
    """Hash of the label values (``"<w>x<h>:"`` + raw bytes), the convention OnlineAnnotator uses."""

    digest = hashlib.sha256(f"{labels.shape[1]}x{labels.shape[0]}:".encode())
    digest.update(np.ascontiguousarray(labels, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _png(arr: np.ndarray, text: dict[str, str]) -> bytes:
    info = PngInfo()
    for key, value in text.items():
        info.add_text(key, value)
    buffer = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr, dtype=np.uint8)).save(buffer, format="PNG", pnginfo=info)
    return buffer.getvalue()


def build_mask_downloads(
    mask: np.ndarray,
    *,
    source_name: str,
    base_name: str,
    app_version: str,
    model_id: str = "",
    job_meta: dict[str, Any] | None = None,
) -> MaskDownloads:
    """Encode a binary mask as label and preview PNGs with a metadata record.

    Parameters
    ----------
    mask:
        Binary mask in any encoding accepted by :func:`binary_labels`.
    source_name:
        File name of the analysed micrograph, recorded in the metadata.
    base_name:
        Safe file stem for the download names.
    app_version:
        Application version, recorded in the metadata and the PNG ``Software`` text chunk.
    model_id, job_meta:
        Run provenance copied into the metadata.
    """

    labels, normalization = binary_labels(mask)
    preview = preview_values(labels)
    software = f"HydrideSegmentation {app_version}"
    labels_png = _png(labels, {"Software": software, "microseg.mask_encoding": "labels",
                               "Description": "Semantic class labels: 0 = background, 1 = hydride."})
    preview_png = _png(preview, {"Software": software, "microseg.mask_encoding": "preview",
                                 "Description": "Display preview: 0 = background, 255 = hydride. Not class labels."})
    foreground = int(labels.sum())
    metadata = {
        "schema_version": MASK_DOWNLOAD_SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "app": {"name": "HydrideSegmentation", "version": app_version},
        "source_name": source_name,
        "model_id": model_id,
        "job": dict(job_meta or {}),
        "width": int(labels.shape[1]),
        "height": int(labels.shape[0]),
        "class_map": [dict(entry) for entry in CLASS_MAP],
        "display_mapping": {"0": 0, "1": PREVIEW_FOREGROUND_VALUE},
        "files": {
            "labels": {
                "name": f"{base_name}_mask_labels.png",
                "purpose": "Semantic class labels for model training and annotation tools such as OnlineAnnotator.",
                "encoding": "8-bit greyscale PNG; pixel value = class index (0 = background, 1 = hydride)",
                "values": sorted(int(v) for v in np.unique(labels)),
                "sha256": hashlib.sha256(labels_png).hexdigest(),
            },
            "preview": {
                "name": f"{base_name}_mask_preview.png",
                "purpose": "Visual preview for image viewers. Normalize 255 -> 1 before any training use.",
                "encoding": "8-bit greyscale PNG; 0 = background, 255 = hydride (display values)",
                "values": sorted(int(v) for v in np.unique(preview)),
                "sha256": hashlib.sha256(preview_png).hexdigest(),
            },
        },
        "label_sha256": label_sha256(labels),
        "normalization": normalization,
        "foreground_pixels": foreground,
        "foreground_fraction": round(foreground / labels.size, 8) if labels.size else 0.0,
        "resized": False,
    }
    return MaskDownloads(base_name=base_name, labels=labels, preview=preview, labels_png=labels_png,
                         preview_png=preview_png, metadata=metadata)


def build_mask_bundle(downloads: MaskDownloads) -> bytes:
    """ZIP with the label PNG, the preview PNG, the metadata JSON and a short README."""

    readme = (
        "HydrideSegmentation mask package\n\n"
        f"{downloads.labels_name}\n"
        "  Semantic class labels: 0 = background, 1 = hydride. Use this file for model training and\n"
        "  for annotation tools such as OnlineAnnotator (it is imported as class numbers, unchanged).\n\n"
        f"{downloads.preview_name}\n"
        "  Visual preview: 0 = background, 255 = hydride. For looking at the mask only; normalize\n"
        "  255 -> 1 before any training use.\n\n"
        f"{downloads.metadata_name}\n"
        f"  {MASK_DOWNLOAD_SCHEMA}: class IDs, display mapping, dimensions, normalization applied,\n"
        "  SHA-256 of both PNGs and of the label values, and the run that produced the mask.\n\n"
        "Both PNGs come from the same label array and have identical dimensions. Nothing is resized.\n"
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(downloads.labels_name, downloads.labels_png)
        archive.writestr(downloads.preview_name, downloads.preview_png)
        archive.writestr(downloads.metadata_name, json.dumps(downloads.metadata, indent=2, ensure_ascii=False))
        archive.writestr("README.txt", readme)
    return buffer.getvalue()
