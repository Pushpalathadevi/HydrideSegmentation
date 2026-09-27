"""Overlays and plots for HCI results (Matplotlib Agg, imported lazily).

Synthetic association bridges are always drawn in a colour distinct from real
hydride, so they can never be mistaken for segmented material.
"""

from __future__ import annotations

from io import BytesIO

import numpy as np

from .contracts import ContinuityAnalysisResult

BRIDGE_RGB = (230, 57, 70)
PATH_RGB = (255, 183, 3)
ENDPOINT_RGB = (230, 57, 70)
JUNCTION_RGB = (29, 53, 87)

_PALETTE = np.array(
    [
        (31, 119, 180), (44, 160, 44), (148, 103, 189), (23, 190, 207), (140, 86, 75),
        (227, 119, 194), (188, 189, 34), (255, 127, 14), (57, 106, 177), (62, 150, 81),
        (107, 76, 154), (146, 36, 40), (148, 139, 61), (83, 81, 84), (0, 128, 128),
    ],
    dtype=np.uint8,
)


def _line(img: np.ndarray, p: tuple[int, int], q: tuple[int, int], rgb: tuple[int, int, int], width: int) -> None:
    from skimage.draw import disk, line

    rr, cc = line(int(p[0]), int(p[1]), int(q[0]), int(q[1]))
    h, w = img.shape[:2]
    rad = max(1, width // 2)
    for r, c in zip(rr[:: max(1, rad)], cc[:: max(1, rad)]):
        dr, dc = disk((r, c), rad, shape=(h, w))
        img[dr, dc] = rgb
    img[np.clip(rr, 0, h - 1), np.clip(cc, 0, w - 1)] = rgb


def _stroke(shape: tuple[int, int]) -> int:
    return max(1, int(round(max(shape) / 400)))


def cluster_overlay(result: ContinuityAnalysisResult, *, background: int = 255) -> np.ndarray:
    """Clusters at the report distance in distinct colours, with bridges in red."""

    cleaned = result.arrays["cleaned"]
    img = np.full(cleaned.shape + (3,), background, np.uint8)
    if not result.ok:
        img[cleaned] = (60, 60, 60)
        return img
    labels = result.arrays["labels"]
    roots = result.arrays["roots"]
    root_img = roots[labels]
    uniq = [r for r in np.unique(root_img) if r != 0]
    # Colour clusters by size rank so the largest clusters get the first colours.
    order = sorted(uniq, key=lambda r: -int((root_img == r).sum()))
    lut = np.zeros((int(roots.max()) + 1, 3), np.uint8)
    for i, r in enumerate(order):
        lut[r] = _PALETTE[i % len(_PALETTE)]
    img[cleaned] = lut[root_img[cleaned]]
    width = _stroke(cleaned.shape) + 1
    for cl in result.clusters:
        for br in cl.bridges:
            _line(img, tuple(br["p_rc"]), tuple(br["q_rc"]), BRIDGE_RGB, width)
    return img


def path_overlay(result: ContinuityAnalysisResult, direction: str = "radial") -> np.ndarray:
    """Hydride in grey with the weakest edge-to-edge path in amber."""

    cleaned = result.arrays["cleaned"]
    img = np.full(cleaned.shape + (3,), 255, np.uint8)
    img[cleaned] = (120, 120, 120)
    path = result.arrays.get("best_paths", {}).get(direction)
    if path is not None and len(path):
        from skimage.draw import disk

        rad = _stroke(cleaned.shape)
        for r, c in path[:: max(1, rad)]:
            dr, dc = disk((r, c), rad + 0.5, shape=cleaned.shape)
            img[dr, dc] = PATH_RGB
    return img


def topology_overlay(result: ContinuityAnalysisResult) -> np.ndarray:
    """Skeleton in black, free ends in red and junctions in navy."""

    from skimage.draw import disk

    cleaned = result.arrays["cleaned"]
    img = np.full(cleaned.shape + (3,), 255, np.uint8)
    img[cleaned] = (205, 205, 205)
    topo = result.arrays.get("topology")
    if topo is None:
        return img
    from scipy.ndimage import binary_dilation

    img[binary_dilation(topo.skeleton, iterations=max(0, _stroke(cleaned.shape) - 1)) if _stroke(cleaned.shape) > 1 else topo.skeleton] = (0, 0, 0)
    rad = 2 * _stroke(cleaned.shape) + 1
    for (r, c) in topo.endpoints_rc:
        dr, dc = disk((r, c), rad, shape=cleaned.shape)
        img[dr, dc] = ENDPOINT_RGB
    for (r, c) in topo.junctions_rc:
        dr, dc = disk((r, c), rad + 1, shape=cleaned.shape)
        img[dr, dc] = JUNCTION_RGB
    return img


def curve_figure(result: ContinuityAnalysisResult, *, fmt: str = "png", dpi: int = 130, title: str | None = None) -> bytes:
    """Connectivity function C_u(δ) with the HCI integral shaded; returns image bytes."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    unit = "µm" if result.unit == "um" else "px"
    if result.ok:
        breaks = np.array(result.curve["step_breaks"])
        dmax = float(result.specimen["max_bridge_distance"])
        right = max(dmax * 1.6, float(breaks[-1]) if len(breaks) else dmax)
        colors = {"radial": "#e76f51", "circumferential": "#2a9d8f", "isotropic": "#6a4c93"}
        names = {"radial": "radial", "circumferential": "circumferential", "isotropic": "isotropic"}
        for d, col in colors.items():
            vals = np.array(result.curve[f"step_C_{d}"])
            xs = np.r_[breaks, right]
            ys = np.r_[vals, vals[-1]]
            ax.step(xs, ys, where="post", color=col, lw=2, label=f"C {names[d]}  (HCI = {result.specimen['HCI'][d]:.3f})")
            if d == "radial":
                sel = xs <= dmax
                xx = np.r_[xs[sel], dmax]
                yy = np.r_[ys[sel], ys[sel][-1] if sel.any() else ys[0]]
                ax.fill_between(xx, 0, yy, step="post", color=col, alpha=0.15)
            half = result.specimen["critical_linking_distance"][d]
            if half is not None and half <= right:
                ax.plot([half], [0.5], "o", color=col, ms=6)
        ax.axvline(dmax, color="k", ls="--", lw=1)
        ax.text(dmax, 1.02, f" δ_max = {dmax:.2f} {unit}", fontsize=9, va="bottom")
        ax.axhline(0.5, color="grey", lw=0.8, ls=":")
        ax.set_xlim(0, right)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel(f"association distance δ ({unit})")
    ax.set_ylabel("connectivity C(δ)")
    ax.set_title(title or "Connectivity function; shaded area / δ_max = HCI (radial)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi)
    plt.close(fig)
    return buf.getvalue()


def to_png_bytes(rgb: np.ndarray) -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", optimize=True)
    return buf.getvalue()
