"""Controlled synthetic segmented-hydride fields for HCI validation.

The benchmark isolates one morphological factor at a time — gap size,
fragmentation, arrangement, topology, orientation — while holding the hydride
area fraction constant. Any candidate Hydride Connectivity Index (HCI) can then
be checked against a known expected ordering instead of against ambiguous real
micrographs.

Every sample is built from an explicit centerline network (nodes and straight
edges) that is rasterised as a union of capsules of fixed thickness. The
network provides exact ground truth (length, endpoints, junction degrees,
independent loops, connected components) for validating skeleton-graph
extraction, independently of any HCI formula.

Coordinate convention: points are ``(x, y)`` in pixels, ``x`` is the image
column (circumferential direction) and ``y`` the image row (radial
direction), matching the Fn convention in ``hydride_statistics``. Pixel centres
lie at integer coordinates.

The module has no import-time side effects and no GUI dependencies.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

BENCHMARK_SCHEMA_VERSION = "microseg.hci_synthetic_benchmark.v1"
GENERATOR_VERSION = "1.0.0"

Point = tuple[float, float]
Builder = Callable[[float, np.random.Generator], "CenterlineNetwork | None"]


# ---------------------------------------------------------------------------
# Configuration and data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SyntheticBenchmarkConfig:
    """Parameters shared by every synthetic benchmark family.

    Parameters
    ----------
    height, width:
        Field size in pixels.
    thickness_px:
        Hydride platelet thickness. Each centerline edge is rasterised as a
        capsule of diameter ``thickness_px``.
    target_area_fraction:
        Hydride area fraction that every area-matched sample reproduces.
    area_fraction_tolerance:
        Maximum absolute deviation from ``target_area_fraction`` accepted for
        area-matched samples. Generation fails instead of silently accepting a
        larger deviation.
    clearance_px:
        Minimum surface-to-surface matrix ligament between objects that are
        meant to be separate. It is kept larger than any association distance
        the benchmark is meant to test, so separate objects stay separate.
    margin_px:
        Minimum distance between any hydride pixel and the field border, except
        in samples that are explicitly border-truncated.
    pixel_size_um:
        Nominal physical pixel size recorded in the manifest so physical-unit
        formulations can be exercised. It does not change the raster.
    base_seed:
        Root seed. Every sample seed is derived deterministically from it.
    replicates:
        Default number of random replicates per condition.
    families:
        Families to generate. ``None`` generates all of them.
    max_placement_attempts:
        Rejection-sampling attempts per object before a placement is declared
        infeasible.
    """

    height: int = 512
    width: int = 512
    thickness_px: float = 4.0
    target_area_fraction: float = 0.05
    area_fraction_tolerance: float = 0.0005
    clearance_px: float = 12.0
    margin_px: float = 16.0
    pixel_size_um: float = 0.5
    base_seed: int = 20260927
    replicates: int = 5
    families: tuple[str, ...] | None = None
    max_placement_attempts: int = 2000

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SyntheticBenchmarkConfig":
        """Build a config from a YAML-style mapping, rejecting unknown keys."""

        known = set(cls.__dataclass_fields__)
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown synthetic benchmark config keys: {unknown}")
        data = dict(payload)
        if data.get("families") is not None:
            data["families"] = tuple(str(v) for v in data["families"])
        return cls(**data)


@dataclass
class CenterlineNetwork:
    """Straight-edge centerline graph describing hydride midlines.

    ``object_ids`` tags each node with the object that created it so ground
    truth can distinguish intended objects from raster-level merges.
    """

    nodes: list[Point] = field(default_factory=list)
    edges: list[tuple[int, int]] = field(default_factory=list)
    object_ids: list[int] = field(default_factory=list)

    @property
    def n_objects(self) -> int:
        return (max(self.object_ids) + 1) if self.object_ids else 0

    def add_object(self, points: list[Point], edges: list[tuple[int, int]]) -> None:
        """Append one object given local node coordinates and local edges."""

        obj = self.n_objects
        offset = len(self.nodes)
        self.nodes.extend((float(x), float(y)) for x, y in points)
        self.object_ids.extend([obj] * len(points))
        self.edges.extend((a + offset, b + offset) for a, b in edges)

    def scaled(self, factor: float) -> "CenterlineNetwork":
        return CenterlineNetwork(
            nodes=[(x * factor, y * factor) for x, y in self.nodes],
            edges=list(self.edges),
            object_ids=list(self.object_ids),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "nodes": [[round(x, 3), round(y, 3)] for x, y in self.nodes],
            "edges": [list(e) for e in self.edges],
            "object_ids": list(self.object_ids),
        }


@dataclass
class SyntheticSample:
    """One generated mask with provenance, ground truth and expectations."""

    sample_id: str
    family: str
    condition: str
    condition_value: float | str | None
    replicate: int
    seed: int
    mask: np.ndarray
    network: CenterlineNetwork | None
    thickness_px: float
    target_area_fraction: float | None
    parameters: dict[str, Any]
    ground_truth: dict[str, Any]

    @property
    def area_fraction(self) -> float:
        return float(self.mask.mean())


# ---------------------------------------------------------------------------
# Rasterisation and geometry primitives
# ---------------------------------------------------------------------------


def _capsule_window(
    shape: tuple[int, int], p0: Point, p1: Point, radius: float
) -> tuple[slice, slice, np.ndarray] | None:
    """Return the window and boolean footprint of a capsule, clipped to shape."""

    h, w = shape
    x0, y0 = p0
    x1, y1 = p1
    xmin = max(int(math.floor(min(x0, x1) - radius)), 0)
    xmax = min(int(math.ceil(max(x0, x1) + radius)), w - 1)
    ymin = max(int(math.floor(min(y0, y1) - radius)), 0)
    ymax = min(int(math.ceil(max(y0, y1) + radius)), h - 1)
    if xmin > xmax or ymin > ymax:
        return None
    ys, xs = np.mgrid[ymin : ymax + 1, xmin : xmax + 1]
    dx, dy = x1 - x0, y1 - y0
    length2 = dx * dx + dy * dy
    if length2 == 0.0:
        t = np.zeros(xs.shape, dtype=float)
    else:
        t = np.clip(((xs - x0) * dx + (ys - y0) * dy) / length2, 0.0, 1.0)
    d2 = (xs - (x0 + t * dx)) ** 2 + (ys - (y0 + t * dy)) ** 2
    return slice(ymin, ymax + 1), slice(xmin, xmax + 1), d2 <= radius * radius


def _segments(points: list[Point], edges: list[tuple[int, int]]) -> list[tuple[Point, Point]]:
    """Edge segments plus zero-length segments for isolated nodes."""

    segs = [(points[a], points[b]) for a, b in edges]
    used = {i for e in edges for i in e}
    segs.extend((points[i], points[i]) for i in range(len(points)) if i not in used)
    return segs


def _paint(mask: np.ndarray, segs: list[tuple[Point, Point]], radius: float) -> None:
    for p0, p1 in segs:
        window = _capsule_window(mask.shape, p0, p1, radius)
        if window is not None:
            sy, sx, fp = window
            mask[sy, sx] |= fp


def _hits(mask: np.ndarray, segs: list[tuple[Point, Point]], radius: float) -> bool:
    for p0, p1 in segs:
        window = _capsule_window(mask.shape, p0, p1, radius)
        if window is not None:
            sy, sx, fp = window
            if np.any(mask[sy, sx] & fp):
                return True
    return False


def rasterize_network(
    network: CenterlineNetwork, shape: tuple[int, int], thickness_px: float
) -> np.ndarray:
    """Rasterise a centerline network as a union of capsules.

    Parameters
    ----------
    network:
        Centerline graph in pixel coordinates.
    shape:
        Output ``(height, width)``.
    thickness_px:
        Capsule diameter.

    Returns
    -------
    numpy.ndarray
        Boolean hydride mask.
    """

    mask = np.zeros(shape, dtype=bool)
    _paint(mask, _segments(network.nodes, network.edges), thickness_px / 2.0)
    return mask


def _rotate(points: list[Point], angle_rad: float) -> list[Point]:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [(x * c - y * s, x * s + y * c) for x, y in points]


def _centered(points: list[Point]) -> list[Point]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    return [(x - cx, y - cy) for x, y in points]


class _Placer:
    """Rejection sampler that places objects with a guaranteed matrix clearance."""

    def __init__(self, cfg: SyntheticBenchmarkConfig, rng: np.random.Generator, clearance: float | None = None):
        self.cfg = cfg
        self.rng = rng
        self.shape = (cfg.height, cfg.width)
        self.radius = cfg.thickness_px / 2.0
        self.clearance = cfg.clearance_px if clearance is None else clearance
        self.forbidden = np.zeros(self.shape, dtype=bool)
        self.network = CenterlineNetwork()

    def _inside(self, pts: list[Point]) -> bool:
        lo = self.cfg.margin_px + self.radius
        return all(
            lo <= x <= self.cfg.width - 1 - lo and lo <= y <= self.cfg.height - 1 - lo for x, y in pts
        )

    def place(self, template: list[Point], edges: list[tuple[int, int]], angle: float | None = None) -> bool:
        """Place a centred template at a random position.

        ``angle`` fixes the orientation in radians; ``None`` draws it uniformly.
        """

        template = _centered(template)
        for _ in range(self.cfg.max_placement_attempts):
            theta = self.rng.uniform(0.0, math.pi) if angle is None else angle
            pts = _rotate(template, theta)
            cx = self.rng.uniform(0.0, self.cfg.width - 1)
            cy = self.rng.uniform(0.0, self.cfg.height - 1)
            pts = [(x + cx, y + cy) for x, y in pts]
            if not self._inside(pts):
                continue
            segs = _segments(pts, edges)
            if _hits(self.forbidden, segs, self.radius):
                continue
            self.network.add_object(pts, edges)
            _paint(self.forbidden, segs, self.radius + self.clearance)
            return True
        return False


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def network_ground_truth(network: CenterlineNetwork) -> dict[str, Any]:
    """Exact topological and metric ground truth of a centerline network.

    Returns
    -------
    dict
        ``total_length_px``, ``n_components``, ``n_endpoints``,
        ``n_junctions``, ``junction_degree_histogram``, ``branching_excess``
        (sum of ``degree - 2`` over junctions), ``n_independent_loops``
        (cyclomatic number ``E - V + C``) and ``network_closure``
        (``2 * loops / (branching_excess + 2 * open_components)``, see the HCI
        specification; ``None`` when undefined).
    """

    n = len(network.nodes)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    degree = [0] * n
    length = 0.0
    for a, b in network.edges:
        degree[a] += 1
        degree[b] += 1
        (xa, ya), (xb, yb) = network.nodes[a], network.nodes[b]
        length += math.hypot(xb - xa, yb - ya)
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = {find(i) for i in range(n)}
    n_components = len(roots)
    loops = len(network.edges) - n + n_components
    endpoints = sum(1 for d in degree if d == 1)
    junction_degrees = [d for d in degree if d >= 3]
    hist: dict[str, int] = {}
    for d in junction_degrees:
        hist[str(d)] = hist.get(str(d), 0) + 1
    branching_excess = sum(d - 2 for d in junction_degrees)
    # Components that contain at least one endpoint or isolated node are "open".
    comp_has_end = {r: False for r in roots}
    for i in range(n):
        if degree[i] <= 1:
            comp_has_end[find(i)] = True
    open_components = sum(1 for v in comp_has_end.values() if v)
    denom = branching_excess + 2 * n_components
    closure = (2.0 * loops / denom) if denom > 0 else None
    return {
        "total_length_px": round(length, 3),
        "n_objects": network.n_objects,
        "n_components": n_components,
        "n_open_components": open_components,
        "n_endpoints": endpoints,
        "n_junctions": len(junction_degrees),
        "junction_degree_histogram": dict(sorted(hist.items())),
        "branching_excess": branching_excess,
        "n_independent_loops": loops,
        "network_closure": None if closure is None else round(closure, 6),
    }


def _raster_components(mask: np.ndarray) -> int:
    from skimage.measure import label

    return int(label(mask, connectivity=2).max())


# ---------------------------------------------------------------------------
# Area-fraction matching
# ---------------------------------------------------------------------------


def _solve_area(
    build: Builder,
    cfg: SyntheticBenchmarkConfig,
    seed: int,
    lo: float,
    hi: float,
    *,
    target_af: float | None = None,
    increasing: bool = True,
    max_iter: int = 60,
) -> tuple[CenterlineNetwork, np.ndarray, float]:
    """Bisect a scalar size parameter until the rasterised area fraction matches.

    The builder is re-run with a fresh generator from the same seed at each
    iteration, so placement is deterministic for a given parameter value. A
    builder returning ``None`` (placement infeasible) is treated as "too much
    hydride". Failure to meet the tolerance raises ``RuntimeError``.
    """

    shape = (cfg.height, cfg.width)
    target = (cfg.target_area_fraction if target_af is None else target_af) * shape[0] * shape[1]
    tol = cfg.area_fraction_tolerance * shape[0] * shape[1]
    best: tuple[float, CenterlineNetwork, np.ndarray, float] | None = None
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        net = build(mid, np.random.default_rng(seed))
        if net is None:
            too_big = True
        else:
            mask = rasterize_network(net, shape, cfg.thickness_px)
            area = float(mask.sum())
            err = abs(area - target)
            if best is None or err < best[0]:
                best = (err, net, mask, mid)
            if err <= tol:
                return net, mask, mid
            too_big = area > target
        if too_big == increasing:
            hi = mid
        else:
            lo = mid
    # Random placement makes area a slightly noisy function of the size
    # parameter, so bisection can stall within one raster jitter of the target.
    # A deterministic scan of small relative offsets around the best value then
    # finds a parameter inside tolerance.
    if best is not None:
        centre = best[3]
        for k in range(1, 401):
            for sign in (1, -1):
                size = centre * (1.0 + sign * k * 5e-4)
                net = build(size, np.random.default_rng(seed))
                if net is None:
                    continue
                mask = rasterize_network(net, shape, cfg.thickness_px)
                err = abs(float(mask.sum()) - target)
                if err < best[0]:
                    best = (err, net, mask, size)
                if err <= tol:
                    return net, mask, size
    detail = "no feasible placement" if best is None else f"closest error {best[0]:.0f} px > tolerance {tol:.0f} px"
    raise RuntimeError(f"area-fraction matching failed (seed={seed}): {detail}")


# ---------------------------------------------------------------------------
# Object templates (centred later by the placer)
# ---------------------------------------------------------------------------


def _line(length: float) -> tuple[list[Point], list[tuple[int, int]]]:
    return [(0.0, 0.0), (0.0, length)], [(0, 1)]


def _chain(seg_len: float, n_seg: int, surface_gap: float, thickness: float) -> tuple[list[Point], list[tuple[int, int]]]:
    """Collinear radial segments separated by a surface-to-surface gap."""

    pitch = seg_len + surface_gap + thickness
    pts: list[Point] = []
    edges: list[tuple[int, int]] = []
    for k in range(n_seg):
        y0 = k * pitch
        pts.extend([(0.0, y0), (0.0, y0 + seg_len)])
        edges.append((2 * k, 2 * k + 1))
    return pts, edges


def _staircase(seg_len: float, n_seg: int, step: float) -> tuple[list[Point], list[tuple[int, int]]]:
    """Radial segments linked by short circumferential steps (degree-2 corners)."""

    pts: list[Point] = [(0.0, 0.0)]
    for k in range(n_seg):
        x = k * step
        pts.append((x, (k + 1) * seg_len))
        if k < n_seg - 1:
            pts.append((x + step, (k + 1) * seg_len))
    edges = [(i, i + 1) for i in range(len(pts) - 1)]
    return pts, edges


def _star(arm: float, n_arms: int) -> tuple[list[Point], list[tuple[int, int]]]:
    pts: list[Point] = [(0.0, 0.0)]
    for k in range(n_arms):
        a = 2.0 * math.pi * k / n_arms
        pts.append((arm * math.cos(a), arm * math.sin(a)))
    return pts, [(0, k + 1) for k in range(n_arms)]


def _polygon(side: float, n_sides: int = 6) -> tuple[list[Point], list[tuple[int, int]]]:
    r = side / (2.0 * math.sin(math.pi / n_sides))
    pts = [(r * math.cos(2 * math.pi * k / n_sides), r * math.sin(2 * math.pi * k / n_sides)) for k in range(n_sides)]
    return pts, [(k, (k + 1) % n_sides) for k in range(n_sides)]


def _branched_tree(trunk: float, n_branches: int = 4) -> tuple[list[Point], list[tuple[int, int]]]:
    """Radial trunk with side branches alternating at +/-60 degrees."""

    branch = trunk / 3.0
    pts: list[Point] = []
    edges: list[tuple[int, int]] = []
    ys = [trunk * (k + 1) / (n_branches + 1) for k in range(n_branches)]
    trunk_nodes = [0.0] + ys + [trunk]
    for y in trunk_nodes:
        pts.append((0.0, y))
    edges.extend((i, i + 1) for i in range(len(trunk_nodes) - 1))
    for k, y in enumerate(ys):
        side = 1.0 if k % 2 == 0 else -1.0
        pts.append((side * branch * math.sin(math.radians(60)), y + branch * math.cos(math.radians(60))))
        edges.append((k + 1, len(pts) - 1))
    return pts, edges


def _honeycomb(side: float, n_cols: int, n_rows: int, centre: Point) -> CenterlineNetwork:
    """Hexagonal network of ``n_cols`` x ``n_rows`` whole cells centred on ``centre``.

    Boundary vertices have degree 2 and interior vertices degree 3, so the
    network has no free ends. The cell count is fixed and ``side`` is
    continuous, so area varies continuously with ``side``.
    """

    dx = 1.5 * side
    dy = math.sqrt(3.0) * side
    key_to_idx: dict[tuple[int, int], int] = {}
    nodes: list[Point] = []
    edge_set: set[tuple[int, int]] = set()

    def node(p: Point) -> int:
        key = (round(p[0] * 1000), round(p[1] * 1000))
        if key not in key_to_idx:
            key_to_idx[key] = len(nodes)
            nodes.append(p)
        return key_to_idx[key]

    for col in range(n_cols):
        for row in range(n_rows):
            cx = col * dx
            cy = row * dy + (dy / 2.0 if col % 2 else 0.0)
            verts = [(cx + side * math.cos(math.pi * k / 3), cy + side * math.sin(math.pi * k / 3)) for k in range(6)]
            ids = [node(v) for v in verts]
            for k in range(6):
                a, b = ids[k], ids[(k + 1) % 6]
                edge_set.add((min(a, b), max(a, b)))
    xs = [p[0] for p in nodes]
    ys = [p[1] for p in nodes]
    ox = centre[0] - (min(xs) + max(xs)) / 2.0
    oy = centre[1] - (min(ys) + max(ys)) / 2.0
    net = CenterlineNetwork()
    net.add_object([(x + ox, y + oy) for x, y in nodes], sorted(edge_set))
    return net


def _fits(net: CenterlineNetwork, cfg: SyntheticBenchmarkConfig) -> bool:
    lo = cfg.margin_px + cfg.thickness_px / 2.0
    return all(lo <= x <= cfg.width - 1 - lo and lo <= y <= cfg.height - 1 - lo for x, y in net.nodes)


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Condition:
    name: str
    value: float | str | None
    builder: Callable[[SyntheticBenchmarkConfig, int], tuple[CenterlineNetwork, np.ndarray, dict[str, Any]]]
    area_matched: bool = True


_PLACEMENT_RESTARTS = 20


def _placed(templates: Callable[[float], list[tuple[list[Point], list[tuple[int, int]]]]], angle: Callable[[np.random.Generator], float | None], cfg: SyntheticBenchmarkConfig) -> Builder:
    def build(size: float, rng: np.random.Generator) -> CenterlineNetwork | None:
        # Random sequential placement can jam before the geometric packing
        # limit; restart from the continuing generator stream (deterministic).
        for _restart in range(_PLACEMENT_RESTARTS):
            placer = _Placer(cfg, rng)
            if all(placer.place(pts, edges, angle(rng)) for pts, edges in templates(size)):
                return placer.network
        return None

    return build


def _const(angle: float | None) -> Callable[[np.random.Generator], float | None]:
    return lambda _rng: angle


def _matched(
    templates: Callable[[float], list[tuple[list[Point], list[tuple[int, int]]]]],
    angle: float | None,
    lo: float,
    hi: float,
    extra: dict[str, Any] | None = None,
) -> Callable[[SyntheticBenchmarkConfig, int], tuple[CenterlineNetwork, np.ndarray, dict[str, Any]]]:
    def run(cfg: SyntheticBenchmarkConfig, seed: int) -> tuple[CenterlineNetwork, np.ndarray, dict[str, Any]]:
        net, mask, size = _solve_area(_placed(templates, _const(angle), cfg), cfg, seed, lo, hi)
        params = {"solved_size_px": round(size, 4)}
        params.update(extra or {})
        return net, mask, params

    return run


VERTICAL = 0.0  # templates are drawn along +y (radial); zero rotation keeps them radial
HORIZONTAL = math.pi / 2.0


def _gap_sweep(cfg: SyntheticBenchmarkConfig) -> list[_Condition]:
    n_chains, n_seg = 10, 3
    t = cfg.thickness_px
    conds: list[_Condition] = []
    for gap in (40.0, 20.0, 10.0, 6.0, 3.0, 0.0):
        if gap > 0:
            tmpl = (lambda g: (lambda L: [_chain(L, n_seg, g, t)] * n_chains))(gap)
        else:
            tmpl = lambda L: [_line(n_seg * L)] * n_chains
        conds.append(
            _Condition(
                f"gap_{int(gap):02d}px",
                gap,
                _matched(tmpl, VERTICAL, 5.0, 150.0, {"n_chains": n_chains, "segments_per_chain": n_seg, "surface_gap_px": gap}),
            )
        )
    return conds


def _fragmentation(cfg: SyntheticBenchmarkConfig) -> list[_Condition]:
    conds = []
    for n in (8, 16, 32, 64, 128):
        tmpl = (lambda k: (lambda L: [_line(L)] * k))(n)
        conds.append(_Condition(f"n_{n:03d}", n, _matched(tmpl, VERTICAL, 2.0, 470.0, {"n_segments": n})))
    return conds


def _arrangement(cfg: SyntheticBenchmarkConfig) -> list[_Condition]:
    n_obj, n_seg, t = 12, 3, cfg.thickness_px
    seg = lambda L: [_line(L)] * (n_obj * n_seg)
    return [
        _Condition("random_circumferential", "circumferential", _matched(seg, HORIZONTAL, 5.0, 150.0, {"n_segments": n_obj * n_seg})),
        _Condition("random_isotropic", "isotropic", _matched(seg, None, 5.0, 150.0, {"n_segments": n_obj * n_seg})),
        _Condition("random_radial", "radial", _matched(seg, VERTICAL, 5.0, 150.0, {"n_segments": n_obj * n_seg})),
        _Condition(
            "radial_stringers",
            "stringers",
            _matched(lambda L: [_chain(L, n_seg, 6.0, t)] * n_obj, VERTICAL, 5.0, 150.0, {"n_chains": n_obj, "surface_gap_px": 6.0}),
        ),
        _Condition(
            "radial_stepped",
            "stepped",
            _matched(lambda L: [_staircase(L, n_seg, 10.0)] * n_obj, VERTICAL, 5.0, 150.0, {"n_chains": n_obj, "step_px": 10.0}),
        ),
        _Condition(
            "radial_continuous",
            "continuous",
            _matched(lambda L: [_line(n_seg * L)] * n_obj, VERTICAL, 5.0, 150.0, {"n_lines": n_obj}),
        ),
    ]


def _topology(cfg: SyntheticBenchmarkConfig) -> list[_Condition]:
    n = 24

    def mesh(cfg_: SyntheticBenchmarkConfig, seed: int) -> tuple[CenterlineNetwork, np.ndarray, dict[str, Any]]:
        centre = ((cfg_.width - 1) / 2.0, (cfg_.height - 1) / 2.0)

        def build(side: float, _rng: np.random.Generator) -> CenterlineNetwork | None:
            net = _honeycomb(side, 4, 3, centre)
            return net if _fits(net, cfg_) else None

        net, mask, side = _solve_area(build, cfg_, seed, 5.0, 200.0)
        return net, mask, {"cell_side_px": round(side, 4), "cells": [4, 3]}

    return [
        _Condition("lines", "lines", _matched(lambda s: [_line(s)] * n, None, 20.0, 400.0, {"n_objects": n})),
        _Condition("y_junctions", "Y", _matched(lambda s: [_star(s / 3.0, 3)] * n, None, 20.0, 400.0, {"n_objects": n})),
        _Condition("crosses", "X", _matched(lambda s: [_star(s / 4.0, 4)] * n, None, 20.0, 400.0, {"n_objects": n})),
        _Condition(
            "branched_trees",
            "tree",
            _matched(lambda s: [_branched_tree(s * 3.0 / 7.0)] * n, VERTICAL, 20.0, 400.0, {"n_objects": n, "branches_per_tree": 4}),
        ),
        _Condition("rings", "ring", _matched(lambda s: [_polygon(s / 6.0)] * n, None, 20.0, 400.0, {"n_objects": n})),
        _Condition("honeycomb_mesh", "mesh", mesh),
    ]


def _orientation(cfg: SyntheticBenchmarkConfig) -> list[_Condition]:
    conds = []
    n = 12
    for deg in (0, 15, 30, 45, 60, 75, 90):
        # Angle is measured from the circumferential (x) axis as in Fn; templates
        # point along +y, so rotate by (deg - 90).
        rot = math.radians(deg - 90.0)
        conds.append(
            _Condition(f"theta_{deg:02d}deg", float(deg), _matched(lambda L: [_line(L)] * n, rot, 20.0, 460.0, {"n_lines": n, "angle_from_circumferential_deg": deg}))
        )
    return conds


def _boolean_model(cfg: SyntheticBenchmarkConfig) -> list[_Condition]:
    """Overlapping random segments (continuum-percolation Boolean model)."""

    conds = []
    seg_len = 60.0
    for mode, angle in (("isotropic", None), ("radial", VERTICAL)):
        for af in (0.02, 0.05, 0.10, 0.15, 0.20):

            def run(cfg_: SyntheticBenchmarkConfig, seed: int, af=af, angle=angle, mode=mode) -> tuple[CenterlineNetwork, np.ndarray, dict[str, Any]]:
                rng = np.random.default_rng(seed)
                shape = (cfg_.height, cfg_.width)
                target = af * shape[0] * shape[1]
                net = CenterlineNetwork()
                mask = np.zeros(shape, dtype=bool)
                lo = cfg_.margin_px + cfg_.thickness_px
                while True:
                    theta = rng.uniform(0.0, math.pi) if angle is None else angle
                    cx = rng.uniform(lo, cfg_.width - 1 - lo)
                    cy = rng.uniform(lo, cfg_.height - 1 - lo)
                    dx, dy = -math.sin(theta) * seg_len / 2, math.cos(theta) * seg_len / 2
                    p0, p1 = (cx - dx, cy - dy), (cx + dx, cy + dy)
                    if not all(lo <= p[0] <= cfg_.width - 1 - lo and lo <= p[1] <= cfg_.height - 1 - lo for p in (p0, p1)):
                        continue
                    trial = mask.copy()
                    _paint(trial, [(p0, p1)], cfg_.thickness_px / 2)
                    if trial.sum() >= target:
                        # Shorten the final segment by bisection to land on target.
                        a, b = 0.0, 1.0
                        for _ in range(40):
                            f = 0.5 * (a + b)
                            q1 = (p0[0] + f * (p1[0] - p0[0]), p0[1] + f * (p1[1] - p0[1]))
                            trial = mask.copy()
                            _paint(trial, [(p0, q1)], cfg_.thickness_px / 2)
                            if abs(trial.sum() - target) <= cfg_.area_fraction_tolerance * trial.size:
                                break
                            if trial.sum() > target:
                                b = f
                            else:
                                a = f
                        net.add_object([p0, q1], [(0, 1)])
                        return net, trial, {"segment_length_px": seg_len, "orientation": mode, "overlaps_allowed": True, "n_segments": net.n_objects}
                    net.add_object([p0, p1], [(0, 1)])
                    mask = trial

            conds.append(_Condition(f"{mode}_af{int(round(af * 100)):02d}", af, run, area_matched=True))
    return conds


def _degenerate(cfg: SyntheticBenchmarkConfig) -> list[_Condition]:
    h, w = cfg.height, cfg.width
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0

    def fixed(net_fn: Callable[[], CenterlineNetwork], extra: dict[str, Any]) -> Callable[[SyntheticBenchmarkConfig, int], tuple[CenterlineNetwork, np.ndarray, dict[str, Any]]]:
        def run(cfg_: SyntheticBenchmarkConfig, _seed: int) -> tuple[CenterlineNetwork, np.ndarray, dict[str, Any]]:
            net = net_fn()
            return net, rasterize_network(net, (cfg_.height, cfg_.width), cfg_.thickness_px), extra

        return run

    def single(points: list[Point], edges: list[tuple[int, int]]) -> CenterlineNetwork:
        net = CenterlineNetwork()
        net.add_object(points, edges)
        return net

    def two_lines() -> CenterlineNetwork:
        net = CenterlineNetwork()
        net.add_object([(cx - 30, cy - 100), (cx - 30, cy + 100)], [(0, 1)])
        net.add_object([(cx + 30, cy - 100), (cx + 30, cy + 100)], [(0, 1)])
        return net

    ring_pts, ring_edges = _polygon(60.0)
    return [
        _Condition("empty", "empty", fixed(CenterlineNetwork, {}), area_matched=False),
        _Condition("speck_below_min_area", "speck", fixed(lambda: single([(cx, cy)], []), {"note": "single capsule of diameter thickness_px"}), area_matched=False),
        _Condition("single_segment", "segment", fixed(lambda: single([(cx, cy - 50), (cx, cy + 50)], [(0, 1)]), {"length_px": 100}), area_matched=False),
        _Condition(
            "single_spanning_radial",
            "spanning",
            fixed(lambda: single([(cx, 0.0), (cx, h - 1.0)], [(0, 1)]), {"border_truncated": True}),
            area_matched=False,
        ),
        _Condition("single_ring", "ring", fixed(lambda: single([(x + cx, y + cy) for x, y in ring_pts], ring_edges), {"hexagon_side_px": 60}), area_matched=False),
        _Condition("single_mesh_cluster", "mesh", fixed(lambda: _honeycomb(60.0, 4, 3, (cx, cy)), {"cell_side_px": 60, "cells": [4, 3]}), area_matched=False),
        _Condition("two_parallel_lines", "pair", fixed(two_lines, {"surface_gap_px": 60 - cfg.thickness_px}), area_matched=False),
    ]


FAMILY_BUILDERS: dict[str, Callable[[SyntheticBenchmarkConfig], list[_Condition]]] = {
    "gap_sweep": _gap_sweep,
    "fragmentation": _fragmentation,
    "arrangement": _arrangement,
    "topology": _topology,
    "orientation": _orientation,
    "boolean_model": _boolean_model,
    "degenerate": _degenerate,
}

#: Replicate overrides; ``None`` uses ``SyntheticBenchmarkConfig.replicates``.
FAMILY_REPLICATES: dict[str, int | None] = {
    "gap_sweep": None,
    "fragmentation": None,
    "arrangement": None,
    "topology": None,
    "orientation": 3,
    "boolean_model": 3,
    "degenerate": 1,
}

FAMILY_DESCRIPTIONS: dict[str, dict[str, Any]] = {
    "gap_sweep": {
        "question": "Does the index increase monotonically as collinear hydrides approach and finally link?",
        "varied": "surface-to-surface gap between collinear radial segments (40, 20, 10, 6, 3, 0 px)",
        "controlled": ["area fraction", "number of chains", "segments per chain", "orientation", "thickness"],
        "expected": [
            {"descriptor": "continuity", "relation": "nondecreasing_in_condition_order", "note": "0 px means one merged line per chain"},
            {"descriptor": "radial_path_continuity", "relation": "nondecreasing_in_condition_order"},
        ],
    },
    "fragmentation": {
        "question": "Does the index decrease as the same total hydride length is split into more, shorter pieces?",
        "varied": "number of radial segments (8 to 128) at a fixed total area",
        "controlled": ["area fraction", "orientation", "thickness", "minimum clearance"],
        "expected": [{"descriptor": "continuity", "relation": "nonincreasing_in_condition_order"}],
    },
    "arrangement": {
        "question": "At fixed area fraction and segment count, does the index separate random, aligned, chained, stepped and continuous arrangements?",
        "varied": "spatial arrangement of 36 equal radial segments",
        "controlled": ["area fraction", "segment count (before merging)", "thickness"],
        "expected": [
            {"descriptor": "radial_path_continuity", "relation": "nondecreasing_in_condition_order", "note": "stepped and continuous may tie"},
            {"descriptor": "isotropic_linkage", "relation": "random_* < radial_stringers < radial_stepped ~ radial_continuous"},
        ],
    },
    "topology": {
        "question": "Does the index distinguish free-ended, branched, looped and meshed hydrides of equal total length?",
        "varied": "object topology (line, Y, X, branched tree, ring, honeycomb mesh)",
        "controlled": ["area fraction", "object count (except mesh)", "thickness"],
        "expected": [
            {"descriptor": "network_closure", "relation": "0 for lines, Y, X and trees; 1 for rings and mesh"},
            {"descriptor": "continuity", "relation": "honeycomb_mesh is the maximum; rings are not more continuous than lines"},
        ],
    },
    "orientation": {
        "question": "Does a directional index follow the hydride angle while an isotropic index stays constant?",
        "varied": "angle of identical lines from the circumferential axis (0 to 90 degrees)",
        "controlled": ["area fraction", "line count", "line length", "thickness"],
        "expected": [
            {"descriptor": "radial_path_continuity", "relation": "nondecreasing_in_condition_order"},
            {"descriptor": "isotropic_continuity", "relation": "approximately_constant"},
        ],
    },
    "boolean_model": {
        "question": "How does the index respond to area fraction itself, across the continuum-percolation transition?",
        "varied": "area fraction (2 to 20 percent) of overlapping random 60 px segments, isotropic and radial",
        "controlled": ["segment length", "thickness"],
        "expected": [{"descriptor": "continuity", "relation": "nondecreasing_in_area_fraction within each orientation"}],
    },
    "degenerate": {
        "question": "Are empty, sub-threshold, single-cluster, looped and border-truncated inputs handled explicitly?",
        "varied": "edge cases (not area matched)",
        "controlled": [],
        "expected": [
            {"descriptor": "all", "relation": "explicit status: empty and speck -> not_applicable; single clusters -> finite value with no zero-weight fallback"}
        ],
    },
}


# ---------------------------------------------------------------------------
# Perturbations for invariance and robustness checks
# ---------------------------------------------------------------------------


def _perturb(mask: np.ndarray, kind: str, rng: np.random.Generator, thickness: float) -> np.ndarray:
    """Apply one documented perturbation to a mask."""

    from scipy import ndimage as ndi

    out = mask.copy()
    h, w = out.shape
    if kind == "transpose":
        return out.T.copy()
    if kind == "flip_lr":
        return out[:, ::-1].copy()
    if kind == "flip_ud":
        return out[::-1, :].copy()
    if kind == "boundary_roughness":
        eroded = ndi.binary_erosion(out)
        dilated = ndi.binary_dilation(out)
        inner = out & ~eroded
        outer = dilated & ~out
        out[inner & (rng.random(out.shape) < 0.10)] = False
        out[outer & (rng.random(out.shape) < 0.10)] = True
        return out
    if kind == "speckle":
        # 40 isolated 2x2 specks (4 px, below a 20 px minimum feature area).
        far = ~ndi.binary_dilation(out, iterations=6)
        ys, xs = np.nonzero(far[2 : h - 3, 2 : w - 3])
        idx = rng.choice(len(ys), size=min(40, len(ys)), replace=False)
        for i in idx:
            out[ys[i] + 2 : ys[i] + 4, xs[i] + 2 : xs[i] + 4] = True
        return out
    if kind == "pinholes":
        core = ndi.binary_erosion(out, iterations=1)
        ys, xs = np.nonzero(core)
        idx = rng.choice(len(ys), size=min(40, len(ys)), replace=False)
        out[ys[idx], xs[idx]] = False
        return out
    if kind == "spurs":
        edge = out & ~ndi.binary_erosion(out)
        ys, xs = np.nonzero(edge)
        idx = rng.choice(len(ys), size=min(30, len(ys)), replace=False)
        for i in idx:
            dy, dx = [(0, 1), (0, -1), (1, 0), (-1, 0)][int(rng.integers(4))]
            for k in range(1, 5):
                y, x = ys[i] + k * dy, xs[i] + k * dx
                if 0 <= y < h and 0 <= x < w:
                    out[y, x] = True
        return out
    if kind == "breaks":
        # Ten 2-pixel-wide cuts across randomly chosen hydride pixels.
        skel_like = ndi.binary_erosion(out, iterations=max(1, int(thickness // 2) - 1))
        ys, xs = np.nonzero(skel_like)
        idx = rng.choice(len(ys), size=min(10, len(ys)), replace=False)
        half = int(math.ceil(thickness)) + 1
        for i in idx:
            y, x = ys[i], xs[i]
            win = out[max(0, y - half) : y + half + 1, max(0, x - half) : x + half + 1]
            vertical_extent = win.any(axis=1).sum()
            horizontal_extent = win.any(axis=0).sum()
            if vertical_extent >= horizontal_extent:
                out[y : y + 2, max(0, x - half) : x + half + 1] = False
            else:
                out[max(0, y - half) : y + half + 1, x : x + 2] = False
        return out
    raise ValueError(f"unknown perturbation: {kind}")


PERTURBATIONS: dict[str, dict[str, Any]] = {
    "transpose": {"exact": True, "expectation": "isotropic indices unchanged; radial and circumferential indices swap"},
    "flip_lr": {"exact": True, "expectation": "all indices unchanged"},
    "flip_ud": {"exact": True, "expectation": "all indices unchanged"},
    "scale_0.5": {"exact": False, "expectation": "unchanged within discretisation tolerance when thresholds are physical"},
    "scale_2.0": {"exact": False, "expectation": "unchanged within discretisation tolerance when thresholds are physical"},
    "boundary_roughness": {"exact": False, "expectation": "small change; 10 percent boundary pixel flips"},
    "speckle": {"exact": False, "expectation": "unchanged after minimum-area cleaning"},
    "pinholes": {"exact": False, "expectation": "unchanged after hole filling"},
    "spurs": {"exact": False, "expectation": "unchanged after spur pruning"},
    "breaks": {"exact": False, "expectation": "unchanged when association distance exceeds 2 px"},
}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def _seed(base: int, *parts: int) -> int:
    return int(np.random.SeedSequence([base, *parts]).generate_state(1)[0])


def _make_sample(
    cfg: SyntheticBenchmarkConfig,
    family: str,
    cond: _Condition,
    replicate: int,
    seed: int,
    net: CenterlineNetwork | None,
    mask: np.ndarray,
    params: dict[str, Any],
    target_af: float | None,
) -> SyntheticSample:
    gt: dict[str, Any] = network_ground_truth(net) if net is not None else {}
    gt["raster_components"] = _raster_components(mask)
    if net is not None:
        gt["topology_consistent"] = gt["raster_components"] == gt["n_components"]
    return SyntheticSample(
        sample_id=f"{family}__{cond.name}__r{replicate:02d}",
        family=family,
        condition=cond.name,
        condition_value=cond.value,
        replicate=replicate,
        seed=seed,
        mask=mask,
        network=net,
        thickness_px=float(params.pop("_thickness", cfg.thickness_px)),
        target_area_fraction=target_af,
        parameters=params,
        ground_truth=gt,
    )


def generate_benchmark(cfg: SyntheticBenchmarkConfig | None = None, progress: Callable[[str], None] | None = None) -> list[SyntheticSample]:
    """Generate every configured family plus the invariance set.

    Parameters
    ----------
    cfg:
        Benchmark configuration; defaults to ``SyntheticBenchmarkConfig()``.
    progress:
        Optional callback receiving one human-readable line per sample.

    Returns
    -------
    list of SyntheticSample
        Samples in deterministic order.
    """

    cfg = cfg or SyntheticBenchmarkConfig()
    names = list(FAMILY_BUILDERS) if cfg.families is None else list(cfg.families)
    unknown = [n for n in names if n not in FAMILY_BUILDERS and n != "invariance"]
    if unknown:
        raise ValueError(f"unknown benchmark families: {unknown}")
    samples: list[SyntheticSample] = []
    for f_idx, family in enumerate(FAMILY_BUILDERS):
        if family not in names:
            continue
        n_rep = FAMILY_REPLICATES.get(family) or cfg.replicates
        for c_idx, cond in enumerate(FAMILY_BUILDERS[family](cfg)):
            for rep in range(n_rep):
                seed = _seed(cfg.base_seed, f_idx, c_idx, rep)
                net, mask, params = cond.builder(cfg, seed)
                target = (cond.value if family == "boolean_model" else cfg.target_area_fraction) if cond.area_matched else None
                sample = _make_sample(cfg, family, cond, rep, seed, net, mask, dict(params), target)
                samples.append(sample)
                if progress:
                    progress(f"{sample.sample_id}: area fraction {sample.area_fraction:.4f}")
    if cfg.families is None or "invariance" in names:
        samples.extend(_invariance(cfg, progress))
    return samples


def _invariance(cfg: SyntheticBenchmarkConfig, progress: Callable[[str], None] | None) -> list[SyntheticSample]:
    """Perturbed copies of three reference samples (replicate 0)."""

    bases = [("arrangement", "radial_stringers"), ("arrangement", "random_isotropic"), ("topology", "honeycomb_mesh")]
    out: list[SyntheticSample] = []
    fam_names = list(FAMILY_BUILDERS)
    for b_idx, (family, cond_name) in enumerate(bases):
        f_idx = fam_names.index(family)
        conds = FAMILY_BUILDERS[family](cfg)
        c_idx = [c.name for c in conds].index(cond_name)
        seed = _seed(cfg.base_seed, f_idx, c_idx, 0)
        net, mask, _ = conds[c_idx].builder(cfg, seed)
        for p_idx, kind in enumerate(PERTURBATIONS):
            rng = np.random.default_rng(_seed(cfg.base_seed, 99, b_idx, p_idx))
            thickness = cfg.thickness_px
            p_net: CenterlineNetwork | None = net
            if kind.startswith("scale_"):
                factor = float(kind.split("_")[1])
                thickness = cfg.thickness_px * factor
                shape = (int(round(cfg.height * factor)), int(round(cfg.width * factor)))
                p_net = net.scaled(factor)
                p_mask = rasterize_network(p_net, shape, thickness)
            else:
                p_mask = _perturb(mask, kind, rng, cfg.thickness_px)
                if kind == "transpose":
                    p_net = CenterlineNetwork([(y, x) for x, y in net.nodes], list(net.edges), list(net.object_ids))
                elif kind == "flip_lr":
                    p_net = CenterlineNetwork([(cfg.width - 1 - x, y) for x, y in net.nodes], list(net.edges), list(net.object_ids))
                elif kind == "flip_ud":
                    p_net = CenterlineNetwork([(x, cfg.height - 1 - y) for x, y in net.nodes], list(net.edges), list(net.object_ids))
            cond = _Condition(f"{cond_name}__{kind}", kind, lambda *_: None, area_matched=False)
            params = {"base_sample": f"{family}__{cond_name}__r00", "perturbation": kind, "_thickness": thickness}
            if kind.startswith("scale_"):
                # Same physical microstructure imaged at a different resolution.
                params["pixel_size_um"] = cfg.pixel_size_um / float(kind.split("_")[1])
            params.update(PERTURBATIONS[kind])
            sample = _make_sample(cfg, "invariance", cond, 0, seed, p_net, p_mask, params, None)
            out.append(sample)
            if progress:
                progress(f"{sample.sample_id}: area fraction {sample.area_fraction:.4f}")
    return out


def write_benchmark(
    samples: list[SyntheticSample],
    output_dir: str | Path,
    cfg: SyntheticBenchmarkConfig,
    *,
    code_version: str | None = None,
    generated_at: str | None = None,
) -> Path:
    """Write indexed masks, per-family geometry and ``manifest.json``.

    Masks are single-channel PNGs with background 0 and hydride 1, matching the
    MicroSeg indexed-mask convention.

    Returns
    -------
    pathlib.Path
        Path of the written manifest.
    """

    from PIL import Image

    root = Path(output_dir)
    (root / "masks").mkdir(parents=True, exist_ok=True)
    (root / "geometry").mkdir(parents=True, exist_ok=True)
    geometry: dict[str, dict[str, Any]] = {}
    entries = []
    for s in samples:
        rel = Path("masks") / s.family / f"{s.sample_id}.png"
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(s.mask.astype(np.uint8)).save(root / rel, optimize=True)
        if s.network is not None:
            geometry.setdefault(s.family, {})[s.sample_id] = s.network.to_json()
        af = s.area_fraction
        entries.append(
            {
                "sample_id": s.sample_id,
                "family": s.family,
                "condition": s.condition,
                "condition_value": s.condition_value,
                "replicate": s.replicate,
                "seed": s.seed,
                "mask_path": rel.as_posix(),
                "height": int(s.mask.shape[0]),
                "width": int(s.mask.shape[1]),
                "thickness_px": s.thickness_px,
                "pixel_size_um": float(s.parameters.get("pixel_size_um", cfg.pixel_size_um)),
                "target_area_fraction": s.target_area_fraction,
                "area_fraction": round(af, 6),
                "area_fraction_matched": s.target_area_fraction is not None
                and abs(af - s.target_area_fraction) <= cfg.area_fraction_tolerance + 1e-12,
                "parameters": s.parameters,
                "ground_truth": s.ground_truth,
            }
        )
    for family, payload in geometry.items():
        (root / "geometry" / f"{family}.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    cfg_payload = asdict(cfg)
    cfg_payload["families"] = list(cfg.families) if cfg.families is not None else None
    manifest = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "generator": "microseg.evaluation.continuity.synthetic",
        "generator_version": GENERATOR_VERSION,
        "code_version": code_version,
        "generated_at": generated_at,
        "config": cfg_payload,
        "conventions": {
            "mask_encoding": "single-channel PNG, 0 = matrix, 1 = hydride",
            "coordinates": "(x, y) pixels; x = column = circumferential, y = row = radial",
            "thickness": "capsule diameter around each centerline edge",
            "gap": "surface-to-surface matrix ligament between capsules",
        },
        "families": FAMILY_DESCRIPTIONS
        | {"invariance": {"question": "Is the index invariant (or correctly equivariant) under exact symmetries, resolution change and segmentation noise?", "perturbations": PERTURBATIONS}},
        "samples": entries,
    }
    path = root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return path


def render_overview(samples: list[SyntheticSample], path: str | Path, *, replicate: int = 0, dpi: int = 110) -> Path:
    """Render one row per family with replicate ``replicate`` of every condition.

    The figure format follows the file suffix (``.png`` or ``.svg``).
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families: dict[str, list[SyntheticSample]] = {}
    for s in samples:
        if s.replicate == replicate:
            families.setdefault(s.family, []).append(s)
    n_rows = len(families)
    n_cols = max(len(v) for v in families.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.85 * n_rows), squeeze=False)
    for r, (family, items) in enumerate(families.items()):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(items):
                ax.axis("off")
                continue
            s = items[c]
            ax.imshow(s.mask, cmap="gray_r", interpolation="nearest")
            label = s.condition.split("__")[-1]
            ax.set_title(f"{label}\nAf={s.area_fraction:.3f}", fontsize=6)
            if c == 0:
                ax.set_ylabel(family, fontsize=7)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def render_overview_from_manifest(
    root: str | Path, path: str | Path, *, exclude: tuple[str, ...] = ("invariance",), dpi: int = 110
) -> Path:
    """Render the overview figure from a written benchmark directory."""

    from PIL import Image

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    samples = []
    for e in manifest["samples"]:
        if e["family"] in exclude:
            continue
        mask = np.array(Image.open(root / e["mask_path"])) > 0
        samples.append(
            SyntheticSample(e["sample_id"], e["family"], e["condition"], e["condition_value"], e["replicate"], e["seed"], mask, None, e["thickness_px"], e["target_area_fraction"], e["parameters"], e["ground_truth"])
        )
    return render_overview(samples, path, dpi=dpi)
