"""Qt viewer state and caching helpers for desktop image workspaces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PySide6.QtCore import QObject


@dataclass
class ModelWarmLoadState:
    """UI-facing warm-load state for the currently selected ML model."""

    status: str = "idle"
    model_id: str = ""
    display_name: str = ""
    device_policy: str = "cpu"
    enable_gpu: bool = False
    message: str = ""
    cache_hit: bool = False
    elapsed_seconds: float = 0.0


@dataclass
class DisplayAssetCacheEntry:
    """Cached display assets derived from an in-memory run record."""

    run_id: str
    input_pixmap: Any
    mask_pixmap: Any
    overlay_pixmap: Any
    base_image_rgb: Any
    predicted_mask: Any
    overlay_image_rgb: Any


@dataclass
class ResultsDashboardCacheEntry:
    """Cached metrics and rendered plots for one dashboard configuration."""

    run_id: str
    cache_key: str
    predicted_metrics: dict[str, Any]
    corrected_metrics: dict[str, Any]
    predicted_visuals: dict[str, Any]
    corrected_visuals: dict[str, Any]
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ViewerState:
    """Shared linked-view state for synchronized image viewers."""

    zoom: float = 1.0
    pan_x: int = 0
    pan_y: int = 0
    fit_mode: bool = True
    black_percentile: float = 0.0
    white_percentile: float = 100.0
    gamma: float = 1.0
    sync_group_id: str = ""


class LinkedViewerController(QObject):
    """Synchronize pan, zoom, and display-contrast state across viewers."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._viewers: dict[str, Any] = {}
        self._active_names: set[str] = set()
        self._syncing = False

    def register(self, name: str, viewer: Any) -> None:
        """Register one viewer that supports the viewer-state contract."""

        self._viewers[str(name)] = viewer
        if hasattr(viewer, "view_state_changed"):
            viewer.view_state_changed.connect(lambda state, src=str(name): self._propagate(src, state))

    def set_active_names(self, names: set[str] | list[str] | tuple[str, ...]) -> None:
        """Restrict synchronization to the provided viewer names."""

        self._active_names = {str(name) for name in names}

    def restore_all(self, state: ViewerState) -> None:
        """Apply a view state to all active viewers without feedback loops."""

        self._syncing = True
        try:
            for name in self._active_names:
                viewer = self._viewers.get(name)
                if viewer is None or not hasattr(viewer, "restore_view_state"):
                    continue
                viewer.restore_view_state(state, emit_change=False)
        finally:
            self._syncing = False

    def snapshot(self, preferred_name: str | None = None) -> ViewerState:
        """Return the current state from one active viewer."""

        ordered = [str(preferred_name)] if preferred_name else []
        ordered.extend(sorted(self._active_names))
        for name in ordered:
            viewer = self._viewers.get(name)
            if viewer is None or not hasattr(viewer, "snapshot_view_state"):
                continue
            return viewer.snapshot_view_state()
        return ViewerState()

    def _propagate(self, source: str, state: ViewerState) -> None:
        if self._syncing:
            return
        if self._active_names and source not in self._active_names:
            return
        self._syncing = True
        try:
            for name in self._active_names:
                if name == source:
                    continue
                viewer = self._viewers.get(name)
                if viewer is None or not hasattr(viewer, "restore_view_state"):
                    continue
                viewer.restore_view_state(state, emit_change=False)
        finally:
            self._syncing = False
