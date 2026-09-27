"""Phase 38: production HCI core — contracts, properties and the benchmark axiom suite."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.microseg.evaluation.continuity import (
    Calibration,
    ContinuityAnalysisConfig,
    analyze_continuity,
    binarize,
)
from src.microseg.evaluation.continuity.linkage import compute_linkage
from src.microseg.evaluation.continuity.skeleton_graph import skeleton_topology

ROOT = Path(__file__).resolve().parents[1] / "test_data" / "hci_synthetic_v1"
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
ENTRIES = {e["sample_id"]: e for e in MANIFEST["samples"]}
FIXED = ContinuityAnalysisConfig(max_bridge_distance=10.0)


def _mask(sample_id: str) -> np.ndarray:
    return np.array(Image.open(ROOT / ENTRIES[sample_id]["mask_path"]))


def _run(sample_id: str, cfg: ContinuityAnalysisConfig | None = None):
    e = ENTRIES[sample_id]
    return analyze_continuity(_mask(sample_id), cfg or FIXED, Calibration(e["pixel_size_um"], "manifest"))


# ---------------------------------------------------------------------------
# Configuration and input handling
# ---------------------------------------------------------------------------


def test_config_validation_and_mapping() -> None:
    cfg = ContinuityAnalysisConfig.from_mapping({"max_bridge_distance": "auto", "path_enabled": "false", "foreground_class_indices": 2})
    assert cfg.max_bridge_distance == "auto" and cfg.path_enabled is False and cfg.foreground_class_indices == (2,)
    assert ContinuityAnalysisConfig.from_mapping({"max_bridge_distance": "7.5"}).max_bridge_distance == 7.5
    with pytest.raises(ValueError, match="unknown"):
        ContinuityAnalysisConfig.from_mapping({"delta": 3})
    with pytest.raises(ValueError):
        ContinuityAnalysisConfig(max_bridge_distance=0.0)
    with pytest.raises(ValueError):
        ContinuityAnalysisConfig(auto_size_statistic="mean")
    with pytest.raises(ValueError):
        ContinuityAnalysisConfig(formulation_id="hci.prototype")


def test_binarize_variants() -> None:
    idx = np.array([[0, 1], [2, 1]], np.uint8)
    assert binarize(idx, (1,))[0].sum() == 2
    assert binarize(idx, (1, 2))[0].sum() == 3
    preview, warn = binarize(np.array([[0, 255], [255, 0]], np.uint8), (1,))
    assert preview.sum() == 2 and warn
    rgb = np.zeros((2, 2, 3), np.uint8)
    rgb[0, 0, 2] = 9
    assert binarize(rgb)[0].sum() == 1


# ---------------------------------------------------------------------------
# Degenerate cases and contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sample_id", ["degenerate__empty__r00", "degenerate__speck_below_min_area__r00"])
def test_not_applicable_has_no_numeric_zeros(sample_id: str) -> None:
    res = _run(sample_id)
    assert res.status == "not_applicable"
    payload = res.to_dict()
    assert all(v is None for v in payload["specimen"]["HCI"].values())
    json.dumps(payload)


def test_single_clusters_are_finite_and_correct() -> None:
    spanning = _run("degenerate__single_spanning_radial__r00")
    assert spanning.specimen["HCI"]["radial"] == pytest.approx(1.0)
    segment = _run("degenerate__single_segment__r00")
    # 100 px centreline + 4 px thickness in a 512 px field.
    assert segment.specimen["HCI"]["radial"] == pytest.approx(104 / 512, abs=0.01)
    mesh = _run("degenerate__single_mesh_cluster__r00")
    assert mesh.specimen["HCI"]["isotropic"] > 0.9


def test_result_contract_is_json_and_deterministic() -> None:
    a = _run("arrangement__radial_stringers__r00").to_dict()
    b = _run("arrangement__radial_stringers__r00").to_dict()
    for d in (a, b):
        for key in ("timestamp", "runtime_s", "stage_times_s"):
            d["provenance"].pop(key)
    assert a == b
    assert a["schema_version"] == "microseg.hci_result.v1" and a["formulation_id"] == "hci.v1"
    assert a["clusters"] and a["clusters"][0]["cluster_id"] == "K0001"
    json.dumps(a)


# ---------------------------------------------------------------------------
# δ_max rule
# ---------------------------------------------------------------------------


def test_auto_max_bridge_distance_is_one_fifth_of_smallest_hydride() -> None:
    res = _run("gap_sweep__gap_40px__r00", ContinuityAnalysisConfig())
    s = res.specimen
    assert s["max_bridge_distance_source"] == "auto:min"
    assert s["max_bridge_distance"] == pytest.approx(0.2 * s["smallest_hydride_length"])
    user = _run("gap_sweep__gap_40px__r00", ContinuityAnalysisConfig(max_bridge_distance=4.0))
    assert user.specimen["max_bridge_distance"] == pytest.approx(4.0)
    assert user.specimen["max_bridge_distance_source"] == "user"


def test_auto_rule_is_resolution_invariant() -> None:
    base = _run("invariance__radial_stringers__scale_2.0__r00", ContinuityAnalysisConfig())
    half = _run("invariance__radial_stringers__scale_0.5__r00", ContinuityAnalysisConfig())
    assert base.specimen["max_bridge_distance"] == pytest.approx(half.specimen["max_bridge_distance"], rel=0.05)


# ---------------------------------------------------------------------------
# Mathematical properties
# ---------------------------------------------------------------------------


def test_connectivity_function_is_bounded_and_monotone() -> None:
    link = compute_linkage(_mask("boolean_model__isotropic_af10__r00") > 0)
    for d in ("radial", "circumferential", "isotropic"):
        v = link.values[d]
        assert np.all(v > 0) and np.all(v <= 1)
        assert np.all(np.diff(v) >= -1e-12)
    assert np.all(np.diff(link.breaks_px) > 0)


def test_linking_never_decreases_hci() -> None:
    mask = np.zeros((200, 200), bool)
    mask[20:80, 98:102] = True
    mask[95:160, 98:102] = True
    linked = mask.copy()
    linked[80:95, 99:101] = True  # a thin bridge of negligible area
    cfg = ContinuityAnalysisConfig(max_bridge_distance=5.0)
    a = analyze_continuity(mask, cfg, Calibration(1.0))
    b = analyze_continuity(linked, cfg, Calibration(1.0))
    assert b.specimen["HCI"]["radial"] > a.specimen["HCI"]["radial"]


def test_exact_symmetries() -> None:
    ref = _run("arrangement__radial_stringers__r00").specimen
    tr = _run("invariance__radial_stringers__transpose__r00").specimen
    assert tr["HCI"]["radial"] == pytest.approx(ref["HCI"]["circumferential"], abs=1e-12)
    assert tr["HCI"]["isotropic"] == pytest.approx(ref["HCI"]["isotropic"], abs=1e-12)
    for kind in ("flip_lr", "flip_ud"):
        s = _run(f"invariance__radial_stringers__{kind}__r00").specimen
        assert s["HCI"] == pytest.approx(ref["HCI"], abs=1e-12)


@pytest.mark.parametrize("kind", ["scale_0.5", "scale_2.0", "speckle", "pinholes", "spurs", "breaks"])
def test_resolution_and_noise_tolerance(kind: str) -> None:
    for base, family in (("radial_stringers", "arrangement"), ("honeycomb_mesh", "topology")):
        ref = _run(f"{family}__{base}__r00").specimen["HCI"]["radial"]
        val = _run(f"invariance__{base}__{kind}__r00").specimen["HCI"]["radial"]
        assert abs(val - ref) / ref <= 0.05, (base, kind, ref, val)


# ---------------------------------------------------------------------------
# Benchmark ordering axioms (fixed δ_max so that conditions are comparable)
# ---------------------------------------------------------------------------

ORDERINGS = {
    "gap_sweep": ["gap_40px", "gap_20px", "gap_10px", "gap_06px", "gap_03px", "gap_00px"],
    "fragmentation": ["n_128", "n_064", "n_032", "n_016", "n_008"],
    "arrangement": ["random_circumferential", "random_isotropic", "random_radial", "radial_stringers", "radial_stepped", "radial_continuous"],
    "orientation": [f"theta_{d:02d}deg" for d in (0, 15, 30, 45, 60, 75, 90)],
}


@pytest.mark.parametrize("family", list(ORDERINGS))
@pytest.mark.parametrize("replicate", [0, 1])
def test_ordering_axiom(family: str, replicate: int) -> None:
    values = [_run(f"{family}__{c}__r{replicate:02d}").specimen["HCI"]["radial"] for c in ORDERINGS[family]]
    assert np.all(np.diff(values) >= -0.01), values


def test_gap_sweep_critical_distance_recovers_designed_gap() -> None:
    # Valid while the designed gap is below the lateral chain spacing (about 36 px);
    # at 40 px neighbouring chains link sideways first and δ½ reads 18 µm.
    for gap_px in (20, 10, 6, 3):
        s = _run(f"gap_sweep__gap_{gap_px:02d}px__r00").specimen
        assert s["critical_linking_distance"]["radial"] == pytest.approx(gap_px * 0.5, abs=0.5)


def test_isotropic_index_ignores_orientation() -> None:
    vals = [_run(f"orientation__{c}__r00").specimen["HCI"]["isotropic"] for c in ORDERINGS["orientation"]]
    assert np.std(vals) / np.mean(vals) < 0.05


# ---------------------------------------------------------------------------
# Topology against exact ground truth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", ["lines", "y_junctions", "crosses", "branched_trees", "rings", "honeycomb_mesh"])
def test_topology_matches_ground_truth(shape: str) -> None:
    sid = f"topology__{shape}__r00"
    gt = ENTRIES[sid]["ground_truth"]
    topo = skeleton_topology(_mask(sid) > 0)
    assert topo.n_endpoints == gt["n_endpoints"]
    assert topo.n_junctions == gt["n_junctions"]
    assert topo.junction_degree_histogram == gt["junction_degree_histogram"]
    assert topo.n_loops == gt["n_independent_loops"]
    assert topo.identity_residual == 0
    assert topo.closure == pytest.approx(gt["network_closure"])
