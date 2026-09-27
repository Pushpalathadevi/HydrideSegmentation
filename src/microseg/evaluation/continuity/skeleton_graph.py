"""Companion descriptor: skeleton-graph topology of the real hydride pixels.

The skeleton is reduced to a graph whose nodes are free ends and collapsed
junctions. Junction pixels are merged within a thickness-scaled radius (the
local distance-transform value), so a thick four-way crossing is one degree-4
node rather than two degree-3 nodes. Loops come from the Euler characteristic,
independently of the node bookkeeping, so the identity

    N_E = β + 2C − 2μ,    β = Σ_j (d_j − 2)

provides a built-in consistency check (``identity_residual``). The network
closure ``κ = 2μ / (β + 2C) = 1 − N_E/(β + 2C)`` lies in [0, 1].
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

_K8 = np.ones((3, 3), int)
_K8[1, 1] = 0


@dataclass
class TopologyResult:
    n_endpoints: int
    n_junctions: int
    junction_degree_histogram: dict[str, int]
    branching_excess: int
    n_components: int
    n_loops: int
    closure: float | None
    identity_residual: int
    skeleton_length_px: float
    endpoints_rc: np.ndarray
    junctions_rc: np.ndarray
    skeleton: np.ndarray
    per_component: dict[int, dict[str, int]] = field(default_factory=dict)


def _skeleton_length(skel: np.ndarray) -> float:
    """Length of an 8-connected skeleton: orthogonal steps count 1 and diagonal steps √2.

    A diagonal step is ignored when the two pixels are already joined through a
    shared orthogonal neighbour, which would otherwise double-count corners.
    """

    s = skel.astype(bool)
    ortho = int((s[:, 1:] & s[:, :-1]).sum() + (s[1:, :] & s[:-1, :]).sum())
    d1 = s[1:, 1:] & s[:-1, :-1] & ~(s[1:, :-1] | s[:-1, 1:])
    d2 = s[1:, :-1] & s[:-1, 1:] & ~(s[1:, 1:] | s[:-1, :-1])
    return float(ortho + math.sqrt(2.0) * (int(d1.sum()) + int(d2.sum())))


def skeleton_topology(mask: np.ndarray, component_labels: np.ndarray | None = None) -> TopologyResult:
    """Topology of the skeleton of a boolean mask.

    Parameters
    ----------
    mask:
        Cleaned boolean hydride mask.
    component_labels:
        Optional 8-connected component labels of ``mask`` for per-component
        aggregation.
    """

    from scipy import ndimage as ndi
    from skimage.measure import euler_number, label
    from skimage.morphology import skeletonize

    skel = skeletonize(mask)
    if not skel.any():
        empty = np.zeros((0, 2), int)
        return TopologyResult(0, 0, {}, 0, 0, 0, None, 0, 0.0, empty, empty, skel)
    deg = ndi.convolve(skel.astype(int), _K8, mode="constant") * skel
    dt = ndi.distance_transform_edt(mask)

    # Collapse junction pixels within a thickness-scaled radius.
    jpix = np.argwhere(skel & (deg >= 3))
    region = np.zeros_like(skel)
    h, w = skel.shape
    for r, c in jpix:
        rad = max(1, int(math.ceil(dt[r, c])))
        r0, r1 = max(0, r - rad), min(h, r + rad + 1)
        c0, c1 = max(0, c - rad), min(w, c + rad + 1)
        yy, xx = np.ogrid[r0:r1, c0:c1]
        region[r0:r1, c0:c1] |= (yy - r) ** 2 + (xx - c) ** 2 <= rad * rad
    region &= skel
    jlab, n_nodes = ndi.label(region, structure=np.ones((3, 3)))

    chains = skel & ~region
    clab, n_chains = ndi.label(chains, structure=np.ones((3, 3)))
    cdeg = ndi.convolve(chains.astype(int), _K8, mode="constant") * chains
    grown = ndi.grey_dilation(jlab, footprint=np.ones((3, 3)))

    node_degree = np.zeros(n_nodes + 1, int)
    endpoints: list[tuple[int, int]] = []
    for r, c in np.argwhere(chains & (cdeg <= 1)):
        if cdeg[r, c] == 1:
            # Ordinary chain end: attached to the adjacent node, or a free end.
            k = grown[r, c]
            if k > 0:
                node_degree[k] += 1
            else:
                endpoints.append((int(r), int(c)))
            continue
        # A single-pixel chain has two ends at one pixel. Each distinct adjacent
        # node takes one end (a bridge between two nodes takes both); ends left
        # over are free (a one-pixel spur, or an isolated dot with two).
        window = jlab[max(0, r - 1) : r + 2, max(0, c - 1) : c + 2]
        adjacent = [int(k) for k in np.unique(window) if k > 0][:2]
        for k in adjacent:
            node_degree[k] += 1
        endpoints.extend([(int(r), int(c))] * (2 - len(adjacent)))

    # A small skeleton loop can lie entirely inside a collapsed node. It is still
    # a loop (counted by the Euler characteristic), so it adds two branch ends
    # to its node, as a self-loop would, keeping N_E = β + 2C − 2μ exact.
    if n_nodes:
        from skimage.measure import regionprops

        for prop in regionprops(jlab):
            holes = 1 - int(prop.euler_number)
            if holes > 0:
                node_degree[prop.label] += 2 * holes

    # Collapsed nodes of degree < 3 are pass-through (2) or free ends (1, 0).
    junction_rc = []
    hist: dict[str, int] = {}
    beta = 0
    node_centroids = ndi.center_of_mass(region, jlab, range(1, n_nodes + 1)) if n_nodes else []
    for k in range(1, n_nodes + 1):
        d = int(node_degree[k])
        cy, cx = node_centroids[k - 1]
        if d >= 3:
            junction_rc.append((cy, cx))
            hist[str(d)] = hist.get(str(d), 0) + 1
            beta += d - 2
        elif d == 1:
            endpoints.append((int(round(cy)), int(round(cx))))
        elif d == 0:
            endpoints.extend([(int(round(cy)), int(round(cx)))] * 2)

    n_comp = int(label(skel, connectivity=2).max())
    chi = int(euler_number(skel, connectivity=2))
    loops = n_comp - chi
    n_end = len(endpoints)
    denom = beta + 2 * n_comp
    closure = (2.0 * loops / denom) if denom > 0 else None
    residual = n_end - (beta + 2 * n_comp - 2 * loops)

    per_component: dict[int, dict[str, int]] = {}
    if component_labels is not None:
        for r, c in endpoints:
            lab = int(component_labels[r, c])
            per_component.setdefault(lab, {"endpoints": 0, "junctions": 0, "branching_excess": 0})["endpoints"] += 1
        for (cy, cx), k in zip(node_centroids, range(1, n_nodes + 1)):
            d = int(node_degree[k])
            if d >= 3:
                rr, cc = np.argwhere(jlab == k)[0]
                lab = int(component_labels[rr, cc])
                entry = per_component.setdefault(lab, {"endpoints": 0, "junctions": 0, "branching_excess": 0})
                entry["junctions"] += 1
                entry["branching_excess"] += d - 2

    return TopologyResult(
        n_endpoints=n_end,
        n_junctions=len(junction_rc),
        junction_degree_histogram=dict(sorted(hist.items())),
        branching_excess=int(beta),
        n_components=n_comp,
        n_loops=int(loops),
        closure=None if closure is None else float(min(max(closure, 0.0), 1.0)),
        identity_residual=int(residual),
        skeleton_length_px=_skeleton_length(skel),
        endpoints_rc=np.array(endpoints, float).reshape(-1, 2),
        junctions_rc=np.array(junction_rc, float).reshape(-1, 2),
        skeleton=skel,
        per_component=per_component,
    )
