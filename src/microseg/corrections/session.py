"""Correction session utilities for editable segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from skimage.draw import disk, polygon

from src.microseg.domain.corrections import CorrectionAction, CorrectionSessionReport


Mode = Literal["add", "erase"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CorrectionSession:
    """In-memory editable mask session with undo/redo support."""

    initial_mask: np.ndarray
    max_history: int = 50

    def __post_init__(self) -> None:
        mask = self.initial_mask
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)
        self.initial_mask = (mask > 0).astype(np.uint8) * 255
        self.current_mask = self.initial_mask.copy()
        self._undo: list[np.ndarray] = []
        self._redo: list[np.ndarray] = []
        self.actions: list[CorrectionAction] = []
        self._stroke_active = False

    def _push_undo(self) -> None:
        self._undo.append(self.current_mask.copy())
        if len(self._undo) > self.max_history:
            self._undo.pop(0)
        self._redo.clear()

    def _apply_mode(self, rows: np.ndarray, cols: np.ndarray, mode: Mode) -> None:
        if mode == "add":
            self.current_mask[rows, cols] = 255
        else:
            self.current_mask[rows, cols] = 0

    def begin_stroke(self) -> None:
        """Begin a brush stroke transaction for efficient undo behavior."""
        if not self._stroke_active:
            self._push_undo()
            self._stroke_active = True

    def end_stroke(self) -> None:
        """End current brush stroke transaction."""
        self._stroke_active = False

    def reset_to_initial(self) -> None:
        """Reset corrected mask to initial prediction state."""
        self._push_undo()
        self.current_mask = self.initial_mask.copy()
        self.actions.append(
            CorrectionAction(
                action_type="polygon",
                mode="erase",
                params={"reset_to_initial": True},
                timestamp_utc=_utc_now(),
            )
        )

    def apply_brush(
        self,
        x: int,
        y: int,
        radius: int = 6,
        mode: Mode = "add",
        *,
        push_undo: bool = True,
        record_action: bool = True,
    ) -> None:
        """Apply circular brush action centered at (x, y)."""

        if push_undo:
            self._push_undo()
        rr, cc = disk((y, x), max(1, int(radius)), shape=self.current_mask.shape)
        self._apply_mode(rr, cc, mode)
        if record_action:
            self.actions.append(
                CorrectionAction(
                    action_type="brush",
                    mode=mode,
                    params={"x": x, "y": y, "radius": radius},
                    timestamp_utc=_utc_now(),
                )
            )

    def apply_polygon(self, points: list[tuple[int, int]], mode: Mode = "add") -> None:
        """Apply polygon fill action using image-space points."""

        if len(points) < 3:
            return
        self._push_undo()
        xs = np.array([p[0] for p in points], dtype=np.int32)
        ys = np.array([p[1] for p in points], dtype=np.int32)
        rr, cc = polygon(ys, xs, shape=self.current_mask.shape)
        self._apply_mode(rr, cc, mode)
        self.actions.append(
            CorrectionAction(
                action_type="polygon",
                mode=mode,
                params={"points": points},
                timestamp_utc=_utc_now(),
            )
        )

    def undo(self) -> bool:
        """Undo latest correction if available."""

        if not self._undo:
            return False
        self._redo.append(self.current_mask.copy())
        self.current_mask = self._undo.pop()
        return True

    def redo(self) -> bool:
        """Redo latest undone correction if available."""

        if not self._redo:
            return False
        self._undo.append(self.current_mask.copy())
        self.current_mask = self._redo.pop()
        return True

    def report(self) -> CorrectionSessionReport:
        """Return correction session summary report."""

        return CorrectionSessionReport(
            initial_foreground_pixels=int(np.count_nonzero(self.initial_mask)),
            current_foreground_pixels=int(np.count_nonzero(self.current_mask)),
            actions_applied=len(self.actions),
            undo_depth=len(self._undo),
            redo_depth=len(self._redo),
        )
