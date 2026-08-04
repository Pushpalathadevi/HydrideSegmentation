"""Browsable library of test micrographs stored on the server.

Deployments drop images into a folder (``test_library`` by default) so that
colleagues who have nothing of their own on hand can still try the application.
The folder is supplied per host and is not tracked in git, so every operation
here treats it as optional: when it is missing, empty, or unreadable the caller
falls back to the example images listed in the configuration.

The folder is scanned per request rather than at startup, because images are
added on a running server. A short time-to-live cache keeps a page full of
thumbnails from re-reading the directory once per image.
"""

from __future__ import annotations

import io
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .segmentation import ALLOWED_EXTENSIONS

_LOGGER = logging.getLogger(__name__)

#: Seconds a directory listing is reused before the folder is read again. Short
#: enough that an image copied onto the server shows up almost immediately.
LISTING_CACHE_SECONDS = 2.0

#: Long side, in pixels, of the thumbnails shown in the picker grid. Full-size
#: micrographs are far too large to send once per grid cell.
THUMBNAIL_LONG_SIDE_PX = 320

#: Number of rendered thumbnails held in memory. Bounded so a large library
#: cannot grow the cache without limit.
THUMBNAIL_CACHE_ENTRIES = 256

#: Thumbnails are JPEG rather than PNG. Micrographs are noisy, so a PNG preview
#: of one costs tens of kilobytes; across a large grid that dominates page load
#: for no benefit, since these are previews the user never analyses.
THUMBNAIL_JPEG_QUALITY = 80

#: Media type of the bytes returned by :meth:`ImageLibrary.thumbnail_bytes`.
THUMBNAIL_MIMETYPE = "image/jpeg"


@dataclass(frozen=True)
class LibraryImage:
    """One selectable micrograph found in the library folder."""

    image_id: str
    filename: str
    label: str
    size_bytes: int

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable view of this entry."""

        return {
            "id": self.image_id,
            "filename": self.filename,
            "label": self.label,
            "size_bytes": self.size_bytes,
        }


def _humanize(stem: str) -> str:
    """Turn a file stem into a label for the picker grid."""

    text = stem.replace("_", " ").replace("-", " ").strip()
    return " ".join(text.split()) or stem


def library_root(library_dir: str | Path | None) -> Path | None:
    """Return the library folder as a path, or ``None`` when unusable.

    Parameters
    ----------
    library_dir:
        Configured folder location. Empty or missing values disable the library.

    Returns
    -------
    Path | None
        The resolved directory, or ``None`` when it is not configured, does not
        exist, or is not a directory.
    """

    if not library_dir:
        return None
    try:
        root = Path(library_dir).resolve()
    except OSError:
        return None
    return root if root.is_dir() else None


def resolve_library_image(library_dir: str | Path | None, image_id: str) -> Path | None:
    """Resolve a requested library id to a file inside the library folder.

    The id arrives from the browser, so it is validated rather than trusted. Ids
    must be bare filenames: anything carrying a path separator, a parent
    reference, or a drive prefix is rejected before the filesystem is touched,
    and the resolved path is confirmed to still sit inside the library so a
    symlink cannot lead outside it.

    Parameters
    ----------
    library_dir:
        Configured library folder.
    image_id:
        Identifier supplied by the client, normally the bare filename.

    Returns
    -------
    Path | None
        The image path, or ``None`` when the id is invalid or absent.
    """

    root = library_root(library_dir)
    if root is None:
        return None

    name = str(image_id or "").strip()
    if not name or name in {".", ".."}:
        return None
    # Both separators are checked explicitly: Path only treats a backslash as a
    # separator on Windows, so a POSIX host would otherwise accept "..\\etc".
    if "/" in name or "\\" in name or "\x00" in name:
        return None
    if name != Path(name).name:
        return None
    if Path(name).suffix.lower().lstrip(".") not in ALLOWED_EXTENSIONS:
        return None

    try:
        candidate = (root / name).resolve()
    except OSError:
        return None
    if not candidate.is_relative_to(root):
        return None
    if not candidate.is_file():
        return None
    return candidate


def _scan(root: Path, limit: int) -> list[LibraryImage]:
    entries: list[LibraryImage] = []
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name.lower())
    except OSError as exc:
        _LOGGER.warning("Image library could not be read: %s", exc)
        return []

    for child in children:
        if len(entries) >= limit:
            _LOGGER.info(
                "Image library holds more than %d images; listing was truncated.", limit
            )
            break
        # Subfolders are ignored by design: the library is a flat drop folder.
        if not child.is_file():
            continue
        if child.suffix.lower().lstrip(".") not in ALLOWED_EXTENSIONS:
            continue
        try:
            size_bytes = child.stat().st_size
        except OSError:
            continue
        entries.append(
            LibraryImage(
                image_id=child.name,
                filename=child.name,
                label=_humanize(child.stem),
                size_bytes=size_bytes,
            )
        )
    return entries


class ImageLibrary:
    """Cached view of the library folder, safe to share across worker threads."""

    def __init__(self, library_dir: str | Path | None, max_images: int = 50) -> None:
        self._library_dir = str(library_dir or "")
        self._max_images = max(1, int(max_images))
        self._lock = threading.Lock()
        self._listing: list[LibraryImage] = []
        self._listing_expires_at = 0.0
        self._thumbnails: OrderedDict[tuple[str, int, int], bytes] = OrderedDict()

    @property
    def library_dir(self) -> str:
        """Return the configured library folder as a string."""

        return self._library_dir

    def available(self) -> bool:
        """Return whether the folder exists and currently holds usable images."""

        return bool(self.list_images())

    def list_images(self) -> list[LibraryImage]:
        """Return the current library contents, using the short-lived cache."""

        now = time.monotonic()
        with self._lock:
            if now < self._listing_expires_at:
                return list(self._listing)

        root = library_root(self._library_dir)
        entries = _scan(root, self._max_images) if root is not None else []

        with self._lock:
            self._listing = entries
            self._listing_expires_at = time.monotonic() + LISTING_CACHE_SECONDS
        return list(entries)

    def resolve(self, image_id: str) -> Path | None:
        """Resolve one library id to a path, or ``None`` when it is not valid."""

        return resolve_library_image(self._library_dir, image_id)

    def thumbnail_bytes(self, path: Path) -> bytes:
        """Render and cache a small preview of one library image.

        Parameters
        ----------
        path:
            An image path already validated by :meth:`resolve`.

        Returns
        -------
        bytes
            JPEG-encoded thumbnail bytes, media type :data:`THUMBNAIL_MIMETYPE`.
        """

        try:
            stat = path.stat()
            key = (str(path), stat.st_mtime_ns, stat.st_size)
        except OSError:
            key = (str(path), 0, 0)

        with self._lock:
            cached = self._thumbnails.get(key)
            if cached is not None:
                self._thumbnails.move_to_end(key)
                return cached

        payload = _render_thumbnail(path)

        with self._lock:
            self._thumbnails[key] = payload
            self._thumbnails.move_to_end(key)
            while len(self._thumbnails) > THUMBNAIL_CACHE_ENTRIES:
                self._thumbnails.popitem(last=False)
        return payload


def _to_displayable(image: Image.Image) -> Image.Image:
    """Convert any decoded micrograph into a mode JPEG can store."""

    if image.mode in {"RGB", "L"}:
        return image
    if image.mode in {"RGBA", "LA", "P"}:
        # JPEG has no alpha channel; compose onto white so a transparent
        # background does not turn black in the picker.
        rgba = image.convert("RGBA")
        flattened = Image.new("RGB", rgba.size, (255, 255, 255))
        flattened.paste(rgba, mask=rgba.split()[3])
        return flattened
    # Scientific images are often 16-bit or float, which PNG cannot hold
    # directly. Stretch to the actual data range so a dark 16-bit scan does not
    # render as a black square.
    if image.mode.startswith("I") or image.mode == "F":
        try:
            scaled = image.convert("I")
            low, high = scaled.getextrema()
            span = float(high) - float(low)
            if span <= 0:
                return scaled.point(lambda value: 0).convert("L")
            return scaled.point(lambda value: int((value - low) * 255.0 / span)).convert("L")
        except (OSError, ValueError):
            pass
    return image.convert("RGB")


def _render_thumbnail(path: Path) -> bytes:
    with Image.open(path) as handle:
        handle.draft("L", (THUMBNAIL_LONG_SIDE_PX, THUMBNAIL_LONG_SIDE_PX))
        image = _to_displayable(handle)
        image.thumbnail(
            (THUMBNAIL_LONG_SIDE_PX, THUMBNAIL_LONG_SIDE_PX), Image.LANCZOS
        )
        buffer = io.BytesIO()
        image.save(
            buffer, format="JPEG", quality=THUMBNAIL_JPEG_QUALITY, optimize=True
        )
    return buffer.getvalue()
