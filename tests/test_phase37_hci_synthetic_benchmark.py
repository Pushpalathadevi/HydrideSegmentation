"""Phase 37: synthetic constant-area-fraction benchmark for HCI validation."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.microseg.evaluation.continuity.synthetic import (
    BENCHMARK_SCHEMA_VERSION,
    FAMILY_BUILDERS,
    CenterlineNetwork,
    SyntheticBenchmarkConfig,
    _honeycomb,
    _perturb,
    _polygon,
    _star,
    generate_benchmark,
    network_ground_truth,
    rasterize_network,
    write_benchmark,
)

REPO = Path(__file__).resolve().parents[1]
COMMITTED = REPO / "test_data" / "hci_synthetic_v1"


def _net(points, edges) -> CenterlineNetwork:
    net = CenterlineNetwork()
    net.add_object(points, edges)
    return net


def test_config_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        SyntheticBenchmarkConfig.from_mapping({"thickness": 3})
    cfg = SyntheticBenchmarkConfig.from_mapping({"families": ["gap_sweep"], "replicates": 2})
    assert cfg.families == ("gap_sweep",) and cfg.replicates == 2


def test_ground_truth_line_star_ring_mesh() -> None:
    line = network_ground_truth(_net([(0, 0), (0, 10)], [(0, 1)]))
    assert (line["n_endpoints"], line["n_junctions"], line["n_independent_loops"]) == (2, 0, 0)
    assert line["total_length_px"] == pytest.approx(10.0)
    assert line["network_closure"] == 0.0

    y = network_ground_truth(_net(*_star(10.0, 3)))
    assert y["n_endpoints"] == 3 and y["junction_degree_histogram"] == {"3": 1}
    assert y["branching_excess"] == 1

    ring = network_ground_truth(_net(*_polygon(10.0)))
    assert ring["n_endpoints"] == 0 and ring["n_independent_loops"] == 1
    assert ring["network_closure"] == pytest.approx(1.0)

    mesh = network_ground_truth(_honeycomb(20.0, 4, 3, (200.0, 200.0)))
    assert mesh["n_components"] == 1 and mesh["n_endpoints"] == 0
    assert mesh["network_closure"] == pytest.approx(1.0)
    assert mesh["n_independent_loops"] == 12  # one per hexagonal cell


def test_endpoint_identity_holds() -> None:
    """N_E = beta + 2C - 2*mu for any graph without isolated rings or nodes."""

    for points, edges in (_star(10, 3), _star(10, 5), _polygon(8)):
        gt = network_ground_truth(_net(points, edges))
        assert gt["n_endpoints"] == gt["branching_excess"] + 2 * gt["n_components"] - 2 * gt["n_independent_loops"]


def test_capsule_area_matches_geometry() -> None:
    # A half-pixel offset avoids the integer-aligned case, whose raster width is t + 1.
    mask = rasterize_network(_net([(50.5, 20), (50.5, 120)], [(0, 1)]), (160, 100), thickness_px=6.0)
    expected = 100 * 6 + math.pi * 9
    assert abs(mask.sum() - expected) / expected < 0.05


@pytest.fixture(scope="module")
def small_benchmark():
    cfg = SyntheticBenchmarkConfig(replicates=1, families=("gap_sweep", "arrangement", "degenerate"))
    return cfg, generate_benchmark(cfg)


def test_area_fraction_is_matched_and_topology_consistent(small_benchmark) -> None:
    cfg, samples = small_benchmark
    for s in samples:
        if s.target_area_fraction is not None:
            assert abs(s.area_fraction - s.target_area_fraction) <= cfg.area_fraction_tolerance, s.sample_id
        if s.family in {"gap_sweep", "arrangement"}:
            assert s.ground_truth["topology_consistent"], s.sample_id


def test_gap_sweep_ground_truth_components(small_benchmark) -> None:
    _, samples = small_benchmark
    comps = {s.condition: s.ground_truth["raster_components"] for s in samples if s.family == "gap_sweep"}
    assert comps["gap_00px"] == 10
    assert all(v == 30 for k, v in comps.items() if k != "gap_00px")


def test_generation_is_deterministic() -> None:
    cfg = SyntheticBenchmarkConfig(replicates=1, families=("gap_sweep",))
    a = generate_benchmark(cfg)
    b = generate_benchmark(cfg)
    assert all(np.array_equal(x.mask, y.mask) for x, y in zip(a, b))


def test_exact_symmetry_perturbations() -> None:
    mask = np.zeros((20, 30), bool)
    mask[2:5, 3:20] = True
    rng = np.random.default_rng(0)
    assert np.array_equal(_perturb(mask, "transpose", rng, 3.0), mask.T)
    assert np.array_equal(_perturb(mask, "flip_lr", rng, 3.0), mask[:, ::-1])
    with pytest.raises(ValueError):
        _perturb(mask, "unknown", rng, 3.0)


def test_unknown_family_rejected() -> None:
    with pytest.raises(ValueError, match="unknown benchmark families"):
        generate_benchmark(SyntheticBenchmarkConfig(families=("nope",)))


def test_write_benchmark_round_trip(tmp_path, small_benchmark) -> None:
    cfg, samples = small_benchmark
    manifest_path = write_benchmark(samples, tmp_path, cfg, code_version="test", generated_at="2026-09-27T00:00:00+00:00")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == BENCHMARK_SCHEMA_VERSION
    entry = manifest["samples"][0]
    stored = np.array(Image.open(tmp_path / entry["mask_path"]))
    assert set(np.unique(stored)) <= {0, 1}
    assert np.array_equal(stored.astype(bool), samples[0].mask)


@pytest.mark.skipif(not (COMMITTED / "manifest.json").exists(), reason="committed benchmark not present")
def test_committed_benchmark_is_complete_and_reproducible() -> None:
    manifest = json.loads((COMMITTED / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == BENCHMARK_SCHEMA_VERSION
    families = {e["family"] for e in manifest["samples"]}
    assert families == set(FAMILY_BUILDERS) | {"invariance"}
    for e in manifest["samples"]:
        assert (COMMITTED / e["mask_path"]).exists(), e["mask_path"]
        if e["target_area_fraction"] is not None:
            assert e["area_fraction_matched"], e["sample_id"]
    # The committed gap-sweep masks regenerate bit-for-bit from the manifest config.
    cfg_payload = dict(manifest["config"])
    cfg_payload["families"] = ["gap_sweep"]
    cfg_payload["replicates"] = 1
    regenerated = generate_benchmark(SyntheticBenchmarkConfig.from_mapping(cfg_payload))
    for s in regenerated:
        stored = np.array(Image.open(COMMITTED / "masks" / "gap_sweep" / f"{s.sample_id}.png")).astype(bool)
        assert np.array_equal(stored, s.mask), s.sample_id
