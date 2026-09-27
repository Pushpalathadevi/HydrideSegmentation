"""Single-linkage connectivity: gap graph, exact connectivity function and HCI integral.

Notation follows ``docs/hci_specification.md`` §5:

* components ``C_j`` are 8-connected hydride components with areas ``A_j``;
* the surface gap ``g_jk`` is the shortest centre-to-centre pixel distance
  between two components minus one pixel, i.e. the length of the matrix
  ligament on the straight segment joining their nearest pixels;
* clusters at level ``δ`` are connected components of the graph whose edges
  join components with ``g_jk ≤ δ`` (single linkage);
* ``C_u(δ) = Σ_K A_K min(1, S_K(u)/L_u) / Σ_K A_K`` with ``S_K`` the projected
  hydride coverage (radial, circumferential) or the Feret diameter (isotropic);
* ``HCI_u = (1/δ_max) ∫_0^{δ_max} C_u(δ) dδ``.

All lengths inside this module are in pixels; the analyzer converts to µm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

DIRS = ("radial", "circumferential", "isotropic")


# ---------------------------------------------------------------------------
# Components
# ---------------------------------------------------------------------------


@dataclass
class ComponentTable:
    """Per-component geometry of a cleaned binary mask (index 0 is unused)."""

    labels: np.ndarray
    n: int
    area_px: np.ndarray
    rows: list[np.ndarray]
    cols: list[np.ndarray]
    hulls: list[np.ndarray]
    feret_px: np.ndarray
    touches_border: np.ndarray
    centroid_rc: np.ndarray


def _hull_vertices(points: np.ndarray) -> np.ndarray:
    """Convex-hull vertices of pixel-centre points (row, col); degenerate sets returned as-is."""

    from scipy.spatial import ConvexHull, QhullError

    if len(points) < 3:
        return points.astype(float)
    try:
        return points[ConvexHull(points).vertices].astype(float)
    except QhullError:
        # Collinear pixels: keep the two extreme points along the principal axis.
        centred = points - points.mean(axis=0)
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        proj = centred @ axis
        return points[[int(np.argmin(proj)), int(np.argmax(proj))]].astype(float)


def _feret(points: np.ndarray) -> float:
    """Maximum Feret diameter in pixels, measured between pixel edges (+1)."""

    if len(points) == 0:
        return 0.0
    if len(points) == 1:
        return 1.0
    diff = points[:, None, :] - points[None, :, :]
    return math.sqrt(float((diff**2).sum(-1).max())) + 1.0


def build_components(mask: np.ndarray) -> ComponentTable:
    """Label 8-connected components and collect the geometry the analysis needs."""

    from skimage.measure import label, regionprops

    labels = label(mask, connectivity=2).astype(np.int32)
    n = int(labels.max())
    h, w = mask.shape
    area = np.zeros(n + 1)
    rows: list[np.ndarray] = [np.zeros(0, np.int32)] * (n + 1)
    cols: list[np.ndarray] = [np.zeros(0, np.int32)] * (n + 1)
    hulls: list[np.ndarray] = [np.zeros((0, 2))] * (n + 1)
    feret = np.zeros(n + 1)
    border = np.zeros(n + 1, bool)
    centroid = np.zeros((n + 1, 2))
    for p in regionprops(labels):
        coords = p.coords
        area[p.label] = p.area
        rows[p.label] = np.unique(coords[:, 0]).astype(np.int32)
        cols[p.label] = np.unique(coords[:, 1]).astype(np.int32)
        hull = _hull_vertices(coords)
        hulls[p.label] = hull
        feret[p.label] = _feret(hull)
        r0, c0, r1, c1 = p.bbox
        border[p.label] = r0 == 0 or c0 == 0 or r1 == h or c1 == w
        centroid[p.label] = p.centroid
    return ComponentTable(labels, n, area, rows, cols, hulls, feret, border, centroid)


# ---------------------------------------------------------------------------
# Gap graph
# ---------------------------------------------------------------------------


@dataclass
class GapEdges:
    """Minimum surface gap between Voronoi-neighbouring components."""

    a: np.ndarray
    b: np.ndarray
    gap_px: np.ndarray
    p_rc: np.ndarray  # nearest pixel of component a (row, col)
    q_rc: np.ndarray  # nearest pixel of component b (row, col)

    def __len__(self) -> int:
        return int(len(self.a))


def gap_edges(labels: np.ndarray) -> GapEdges:
    """Sparse gap graph from the Euclidean distance transform of the matrix.

    For every pair of 8-adjacent pixels whose nearest hydride components
    differ, the distance between their two nearest hydride pixels is a
    candidate gap. Keeping the minimum per component pair yields the
    Voronoi-neighbour graph, which contains the minimum spanning tree and is
    therefore sufficient for exact single-linkage clustering. Cost is O(HW)
    time and memory; no pairwise pixel-distance matrix is formed.
    """

    from scipy.ndimage import distance_transform_edt

    empty = GapEdges(np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros((0, 2), np.int64), np.zeros((0, 2), np.int64))
    if labels.max() < 2:
        return empty
    _, (iy, ix) = distance_transform_edt(labels == 0, return_indices=True)
    near = labels[iy, ix]
    h, w = labels.shape
    parts = []
    for dy, dx in ((0, 1), (1, 0), (1, 1), (1, -1)):
        ys0, ys1 = slice(0, h - dy), slice(dy, h)
        xs0, xs1 = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        la, lb = near[ys0, xs0], near[ys1, xs1]
        diff = la != lb
        if not diff.any():
            continue
        pr, pc = iy[ys0, xs0][diff], ix[ys0, xs0][diff]
        qr, qc = iy[ys1, xs1][diff], ix[ys1, xs1][diff]
        a, b = la[diff], lb[diff]
        swap = a > b
        a2 = np.where(swap, b, a)
        b2 = np.where(swap, a, b)
        p_r = np.where(swap, qr, pr)
        p_c = np.where(swap, qc, pc)
        q_r = np.where(swap, pr, qr)
        q_c = np.where(swap, pc, qc)
        d = np.hypot(p_r - q_r, p_c - q_c) - 1.0
        parts.append((a2, b2, d, p_r, p_c, q_r, q_c))
    if not parts:
        return empty
    a, b, d, p_r, p_c, q_r, q_c = (np.concatenate(x) for x in zip(*parts))
    key = a.astype(np.int64) * (int(labels.max()) + 1) + b
    order = np.lexsort((d, key))
    first = order[np.r_[True, key[order][1:] != key[order][:-1]]]
    return GapEdges(
        a[first].astype(np.int64),
        b[first].astype(np.int64),
        np.maximum(d[first], 0.0),
        np.stack([p_r[first], p_c[first]], 1).astype(np.int64),
        np.stack([q_r[first], q_c[first]], 1).astype(np.int64),
    )


# ---------------------------------------------------------------------------
# Kruskal merge events and the exact connectivity function
# ---------------------------------------------------------------------------


@dataclass
class MergeEvent:
    gap_px: float
    a: int
    b: int
    edge_index: int


@dataclass
class LinkageResult:
    """Exact step functions ``C_u(δ)`` and the data to replay clustering at any ``δ``."""

    components: ComponentTable
    edges: GapEdges
    events: list[MergeEvent]
    reference_px: dict[str, float]
    breaks_px: np.ndarray  # δ at which each step starts; breaks_px[0] = 0
    values: dict[str, np.ndarray]  # C_u on [breaks[k], breaks[k+1])
    n_clusters: np.ndarray
    total_area_px: float
    extras: dict[str, object] = field(default_factory=dict)

    def value_at(self, direction: str, delta_px: float) -> float:
        k = int(np.searchsorted(self.breaks_px, delta_px, side="right")) - 1
        return float(self.values[direction][max(k, 0)])

    def clusters_at_count(self, delta_px: float) -> int:
        k = int(np.searchsorted(self.breaks_px, delta_px, side="right")) - 1
        return int(self.n_clusters[max(k, 0)])

    def integral_mean(self, direction: str, dmax_px: float) -> float:
        """``(1/δ_max) ∫_0^{δ_max} C_u(δ) dδ`` evaluated exactly on the step function."""

        if dmax_px <= 0:
            return float(self.values[direction][0])
        starts = np.minimum(self.breaks_px, dmax_px)
        ends = np.minimum(np.r_[self.breaks_px[1:], np.inf], dmax_px)
        return float(np.sum(self.values[direction] * (ends - starts)) / dmax_px)

    def delta_half(self, direction: str) -> float | None:
        hit = np.nonzero(self.values[direction] >= 0.5)[0]
        return None if len(hit) == 0 else float(self.breaks_px[hit[0]])

    def roots_at(self, delta_px: float) -> np.ndarray:
        """Cluster id (a component label) for every component label at level ``δ``."""

        parent = np.arange(self.components.n + 1)

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        for ev in self.events:
            if ev.gap_px > delta_px:
                break
            ra, rb = find(ev.a), find(ev.b)
            if ra != rb:
                parent[ra] = rb
        return np.array([find(i) for i in range(self.components.n + 1)])


def _weights(rows: np.ndarray, cols: np.ndarray, feret: float, ref: dict[str, float]) -> dict[str, float]:
    return {
        "radial": min(1.0, len(rows) / ref["radial"]),
        "circumferential": min(1.0, len(cols) / ref["circumferential"]),
        "isotropic": min(1.0, feret / ref["isotropic"]),
    }


def compute_linkage(mask: np.ndarray, reference_px: dict[str, float] | None = None) -> LinkageResult:
    """Components, gap graph and the exact connectivity step functions.

    Parameters
    ----------
    mask:
        Cleaned boolean hydride mask with at least one foreground pixel.
    reference_px:
        Reference length ``L_u`` in pixels per direction; defaults to the field
        extent (height, width, min of both).
    """

    h, w = mask.shape
    ref = {"radial": float(h), "circumferential": float(w), "isotropic": float(min(h, w))}
    if reference_px:
        ref.update({k: float(v) for k, v in reference_px.items() if v})
    comp = build_components(mask)
    if comp.n == 0:
        raise ValueError("compute_linkage requires a non-empty mask")
    edges = gap_edges(comp.labels)
    order = np.argsort(edges.gap_px, kind="stable")

    # Per-root cluster state.
    rows = {i: comp.rows[i] for i in range(1, comp.n + 1)}
    cols = {i: comp.cols[i] for i in range(1, comp.n + 1)}
    hulls = {i: comp.hulls[i] for i in range(1, comp.n + 1)}
    feret = {i: float(comp.feret_px[i]) for i in range(1, comp.n + 1)}
    area = {i: float(comp.area_px[i]) for i in range(1, comp.n + 1)}
    wts = {i: _weights(rows[i], cols[i], feret[i], ref) for i in rows}
    total = float(comp.area_px[1:].sum())
    num = {d: sum(area[i] * wts[i][d] for i in rows) for d in DIRS}

    parent = list(range(comp.n + 1))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    breaks = [0.0]
    values = {d: [num[d] / total] for d in DIRS}
    n_clusters = [comp.n]
    events: list[MergeEvent] = []
    clusters = comp.n
    for idx in order:
        a, b, g = int(edges.a[idx]), int(edges.b[idx]), float(edges.gap_px[idx])
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        for d in DIRS:
            num[d] -= area[ra] * wts[ra][d] + area[rb] * wts[rb][d]
        parent[ra] = rb
        rows[rb] = np.union1d(rows[rb], rows.pop(ra))
        cols[rb] = np.union1d(cols[rb], cols.pop(ra))
        hull = _hull_vertices(np.vstack([hulls[rb], hulls.pop(ra)]))
        hulls[rb] = hull
        feret[rb] = _feret(hull)
        feret.pop(ra)
        area[rb] += area.pop(ra)
        wts.pop(ra)
        wts[rb] = _weights(rows[rb], cols[rb], feret[rb], ref)
        for d in DIRS:
            num[d] += area[rb] * wts[rb][d]
        clusters -= 1
        events.append(MergeEvent(g, a, b, int(idx)))
        if g == breaks[-1]:
            for d in DIRS:
                values[d][-1] = num[d] / total
            n_clusters[-1] = clusters
        else:
            breaks.append(g)
            for d in DIRS:
                values[d].append(num[d] / total)
            n_clusters.append(clusters)
    return LinkageResult(
        components=comp,
        edges=edges,
        events=events,
        reference_px=ref,
        breaks_px=np.array(breaks),
        values={d: np.clip(np.array(v), 0.0, 1.0) for d, v in values.items()},
        n_clusters=np.array(n_clusters),
        total_area_px=total,
    )
