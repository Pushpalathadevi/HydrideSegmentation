"""Companion descriptor: path continuity Π_u (minimum matrix ligament).

For every pixel ``p`` the cheapest path between the two field edges normal to
direction ``u`` that passes through ``p`` crosses a matrix ligament
``ℓ(p) = D_in(p) + D_out(p)``, where ``D`` are geodesic cumulative costs with
cost ``ε`` in hydride and 1 in matrix (Dijkstra on the 8-neighbour grid). Then

* ``Π_u = mean over hydride pixels of max(0, 1 − ℓ(p)/L_u)``;
* ``Π_u^best = max(0, 1 − min_p ℓ(p)/L_u)``, an RHCP-type weakest path
  (Simon et al., J. Nucl. Mater. 547 (2021) 152817).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PathContinuity:
    direction: str
    mean: float
    best: float
    best_path_rc: np.ndarray  # (k, 2) pixel coordinates of the weakest path
    ligament_px: float


def _sweep(cost: np.ndarray, starts: list[tuple[int, int]]) -> tuple[np.ndarray, object]:
    from skimage.graph import MCP_Geometric

    mcp = MCP_Geometric(cost)
    cum, _ = mcp.find_costs(starts)
    return cum, mcp


def path_continuity(mask: np.ndarray, direction: str, reference_px: float, hydride_cost: float = 0.0) -> PathContinuity:
    """Π_u for ``direction`` in {"radial", "circumferential"} on a non-empty boolean mask."""

    h, w = mask.shape
    cost = np.where(mask, float(hydride_cost), 1.0)
    if direction == "radial":
        a = [(0, j) for j in range(w)]
        b = [(h - 1, j) for j in range(w)]
    elif direction == "circumferential":
        a = [(i, 0) for i in range(h)]
        b = [(i, w - 1) for i in range(h)]
    else:
        raise ValueError(f"path continuity is directional; got {direction!r}")
    da, mcp_a = _sweep(cost, a)
    db, mcp_b = _sweep(cost, b)
    # MCP_Geometric charges each step the mean cost of its two pixels, so each sweep
    # contains half of p's own cost and their sum contains it exactly once.
    lig = da + db
    span = float(reference_px)
    weights = np.clip(1.0 - lig[mask] / span, 0.0, 1.0)
    idx = np.unravel_index(int(np.argmin(lig)), lig.shape)
    try:
        first = np.array(mcp_a.traceback(idx))
        second = np.array(mcp_b.traceback(idx))[::-1]
        best_path = np.vstack([first, second[1:]]) if len(second) else first
    except Exception:  # pragma: no cover - traceback failure only affects the overlay
        best_path = np.zeros((0, 2), int)
    return PathContinuity(
        direction=direction,
        mean=float(weights.mean()),
        best=float(np.clip(1.0 - lig[idx] / span, 0.0, 1.0)),
        best_path_rc=best_path.astype(np.int32),
        ligament_px=float(lig[idx]),
    )
