"""Metadata-driven download catalog for the local web application."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import mimetypes
from pathlib import Path
from typing import Any

_LOGGER = logging.getLogger(__name__)
SCHEMA_VERSION = "microseg.download.v1"


@dataclass(frozen=True)
class DownloadAsset:
    """One downloadable repository asset described by a JSON sidecar."""

    asset_id: str
    display_name: str
    help_text: str
    description: str
    category: str
    version: str
    repo_path: str
    download_name: str
    media_type: str
    order: int
    featured: bool
    path: Path
    available: bool
    size_bytes: int | None
    sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return presentation-safe metadata without exposing local paths."""

        size_label = None
        if self.size_bytes is not None:
            size_label = (
                f"{self.size_bytes / 1048576:,.1f} MB"
                if self.size_bytes >= 1048576
                else f"{self.size_bytes / 1024:,.1f} KB"
            )
        return {
            "asset_id": self.asset_id,
            "display_name": self.display_name,
            "help_text": self.help_text,
            "description": self.description,
            "category": self.category,
            "version": self.version,
            "download_name": self.download_name,
            "media_type": self.media_type,
            "order": self.order,
            "featured": self.featured,
            "available": self.available,
            "size_bytes": self.size_bytes,
            "size_label": size_label,
            "sha256": self.sha256,
        }


class DownloadCatalog:
    """Load validated download records from ``downloads/metadata/*.json``."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.metadata_dir = self.repo_root / "downloads" / "metadata"

    def _resolve_path(self, repo_path: str) -> Path:
        candidate = (self.repo_root / repo_path).resolve()
        try:
            candidate.relative_to(self.repo_root)
        except ValueError as exc:
            raise ValueError("repo_path must remain inside the repository") from exc
        return candidate

    @staticmethod
    def _digest(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_one(self, path: Path) -> DownloadAsset:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        asset_id = str(payload.get("asset_id", "")).strip()
        repo_path = str(payload.get("repo_path", "")).strip()
        display_name = str(payload.get("display_name", "")).strip()
        if not asset_id or not repo_path or not display_name:
            raise ValueError("asset_id, display_name, and repo_path are required")
        resolved = self._resolve_path(repo_path)
        available = resolved.is_file()
        media_type = str(payload.get("media_type", "")).strip()
        if not media_type:
            media_type = mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        return DownloadAsset(
            asset_id=asset_id,
            display_name=display_name,
            help_text=str(payload.get("help_text", "")).strip(),
            description=str(payload.get("description", "")).strip(),
            category=str(payload.get("category", "Other")).strip() or "Other",
            version=str(payload.get("version", "")).strip(),
            repo_path=repo_path,
            download_name=str(payload.get("download_name", "")).strip() or resolved.name,
            media_type=media_type,
            order=int(payload.get("order", 100)),
            featured=bool(payload.get("featured", False)),
            path=resolved,
            available=available,
            size_bytes=resolved.stat().st_size if available else None,
            sha256=self._digest(resolved) if available else None,
        )

    def assets(self) -> list[DownloadAsset]:
        """Return all valid catalog entries in display order."""

        if not self.metadata_dir.is_dir():
            return []
        records: list[DownloadAsset] = []
        seen: set[str] = set()
        for path in sorted(self.metadata_dir.glob("*.json")):
            try:
                asset = self._load_one(path)
                if asset.asset_id in seen:
                    raise ValueError(f"duplicate asset_id {asset.asset_id!r}")
                seen.add(asset.asset_id)
                records.append(asset)
            except Exception as exc:
                _LOGGER.warning("Ignoring invalid download metadata %s: %s", path, exc)
        return sorted(records, key=lambda item: (not item.featured, item.order, item.display_name.lower()))

    def get(self, asset_id: str) -> DownloadAsset | None:
        """Return one asset by stable ID."""

        wanted = str(asset_id).strip()
        return next((item for item in self.assets() if item.asset_id == wanted), None)
