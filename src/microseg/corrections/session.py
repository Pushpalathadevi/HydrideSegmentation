"""Correction session utilities for editable segmentation masks."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
from skimage.draw import disk, polygon
from skimage.measure import label

from src.microseg.corrections.classes import to_index_mask

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
        self.initial_mask = to_index_mask(self.initial_mask)
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

    def _apply_mode(
        self,
        rows: np.ndarray,
        cols: np.ndarray,
        mode: Mode,
        class_index: int,
    ) -> None:
        if mode == "add":
            self.current_mask[rows, cols] = np.uint8(class_index)
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
        class_index: int = 1,
        *,
        push_undo: bool = True,
        record_action: bool = True,
    ) -> None:
        """Apply circular brush action centered at (x, y)."""

        if push_undo:
            self._push_undo()
        rr, cc = disk((y, x), max(1, int(radius)), shape=self.current_mask.shape)
        self._apply_mode(rr, cc, mode, class_index)
        if record_action:
            self.actions.append(
                CorrectionAction(
                    action_type="brush",
                    mode=mode,
                    params={"x": x, "y": y, "radius": radius, "class_index": class_index},
                    timestamp_utc=_utc_now(),
                )
            )

    def apply_polygon(
        self,
        points: list[tuple[int, int]],
        mode: Mode = "add",
        class_index: int = 1,
    ) -> None:
        """Apply polygon fill action using image-space points."""

        if len(points) < 3:
            return
        self._push_undo()
        xs = np.array([p[0] for p in points], dtype=np.int32)
        ys = np.array([p[1] for p in points], dtype=np.int32)
        rr, cc = polygon(ys, xs, shape=self.current_mask.shape)
        self._apply_mode(rr, cc, mode, class_index)
        self.actions.append(
            CorrectionAction(
                action_type="polygon",
                mode=mode,
                params={"points": points, "class_index": class_index},
                timestamp_utc=_utc_now(),
            )
        )

    def delete_feature(self, x: int, y: int) -> bool:
        """Delete connected feature component at the clicked pixel."""

        h, w = self.current_mask.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        cls = int(self.current_mask[y, x])
        if cls == 0:
            return False

        target = self.current_mask == cls
        cc_map = label(target, connectivity=2)
        cc_id = int(cc_map[y, x])
        if cc_id == 0:
            return False

        self._push_undo()
        pix = cc_map == cc_id
        self.current_mask[pix] = 0
        self.actions.append(
            CorrectionAction(
                action_type="feature_delete",
                mode="erase",
                params={"x": x, "y": y, "deleted_class_index": cls, "deleted_pixels": int(np.count_nonzero(pix))},
                timestamp_utc=_utc_now(),
            )
        )
        return True

    def relabel_feature(self, x: int, y: int, class_index: int) -> bool:
        """Relabel connected feature component at clicked pixel to class index."""

        h, w = self.current_mask.shape
        if x < 0 or y < 0 or x >= w or y >= h:
            return False
        src_cls = int(self.current_mask[y, x])
        if src_cls == 0:
            return False

        target = self.current_mask == src_cls
        cc_map = label(target, connectivity=2)
        cc_id = int(cc_map[y, x])
        if cc_id == 0:
            return False

        self._push_undo()
        pix = cc_map == cc_id
        self.current_mask[pix] = np.uint8(class_index)
        self.actions.append(
            CorrectionAction(
                action_type="feature_relabel",
                mode="add",
                params={
                    "x": x,
                    "y": y,
                    "from_class_index": src_cls,
                    "to_class_index": class_index,
                    "relabeled_pixels": int(np.count_nonzero(pix)),
                },
                timestamp_utc=_utc_now(),
            )
        )
        return True

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
