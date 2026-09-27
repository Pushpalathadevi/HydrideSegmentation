# HCI Implementation and Publication Plan (Phases 38–40)

| Field | Value |
|---|---|
| Status | **Phases 38–39 largely delivered in MicroSeg 2.0.0** (see the table below and [`phase38_hci_progress.md`](phase38_hci_progress.md)); Phase 40 is pending field data |
| Inputs | [`hci_specification.md`](hci_specification.md), [`hci_synthetic_benchmark.md`](hci_synthetic_benchmark.md), `test_data/hci_synthetic_v1/`, `scripts/hci_candidate_study.py` |
| Governing rules | `AGENTS.md` (repository root) §4–§11 and §15, and the HCI principles in [`hydride_connectivity_index.md`](hydride_connectivity_index.md) §5 |

## 0. Delivery Status (2.0.0)

| Item | Status |
|---|---|
| Core library: `config`, `contracts`, `linkage`, `path`, `skeleton_graph`, `analyzer`, `visualization`, `batch` (preprocessing folded into `analyzer`) | Delivered |
| Axiom, topology and property tests (`tests/test_phase38_hci_core.py`) | Delivered (33 tests) |
| CLI `hci`, web integration (on by default), PDF/XLSX/ZIP export, KaTeX Help, Sphinx theory | Delivered (`tests/test_phase38_hci_interfaces.py`) |
| δ_max rule | Owner default `auto` (1/5 of smallest hydride); `percentile` and fixed modes available |
| Desktop (Qt) card, inference-export hook, correction-loop recomputation | **Open**: the library call is ready; UI wiring remains |
| 2048² performance target (≤ 2 s) | **Open**: 8.4 s measured, 6 s of it in path continuity |
| Phase 40 field validation and manuscript | Pending owner field data |

## 1. Goals

1. Implement `hci.v1-candidate` as a tested, experimental, off-by-default post-processing
   analysis in `src/microseg/evaluation/continuity/`, with identical results through the CLI,
   desktop and web adapters.
2. Turn the synthetic benchmark into an automated **axiom test suite**, the release gate G1.
3. Run the real-data validation protocol (specification §11) and write the validation report.
4. Prepare a journal manuscript. Target: *Journal of Nuclear Materials*; alternatives are
   *Materials Characterization* and *Integrating Materials and Manufacturing Innovation*.

## 2. Phase Breakdown

### Phase 38 — Core library and axiom suite (CPU, no UI)

| WP | Deliverable | Detail | Done when |
|---|---|---|---|
| 38.1 | `continuity/config.py` | `ContinuityAnalysisConfig`, a frozen dataclass mirroring specification §7 with µm-first thresholds. Validation rejects unknown keys and inconsistent ranges. YAML `configs/hci.default.yml` supports `--set` | Unit tests for defaults, overrides and rejection |
| 38.2 | `continuity/contracts.py` | `ContinuityAnalysisResult`, `ContinuityClusterResult` and `ConnectivityCurve`, JSON-serialisable, schema `microseg.hci_result.v1`, with explicit `status` | Schema round-trip test; the `not_applicable` path has no numeric zeros |
| 38.3 | `continuity/preprocess.py` | Class-set binarisation, cleaning in µm², area bookkeeping and quality flags | Tests on `degenerate`, speckle and pinholes |
| 38.4 | `continuity/linkage.py` | EDT Voronoi gap graph, Kruskal merge events, interval-set coverage, Feret merge, then C_u(δ), HCI_u (exact integral), δ½ and ξ | Matches the study harness to 1e-9 on the whole benchmark; unit tests on hand-built masks with analytic answers |
| 38.5 | `continuity/path.py` | Π_u and Π_u^best with configurable ε; best-path extraction for the overlay | Monotonicity under set inclusion (property test); RHCP-comparable mode ε = 0.02 |
| 38.6 | `continuity/skeleton_graph.py` | Skeleton, thickness-scaled junction collapse, branch tracing, N_E, N_J, β, μ, κ and densities. Implemented in-house; `skan` is not added to the core dependencies | Exact match to benchmark ground truth for lines, Y, X, trees, rings and mesh (gate G3), including the degree-4 crossing case |
| 38.7 | `continuity/analyzer.py` | `analyze_continuity(mask, config, calibration) -> ContinuityAnalysisResult`. Pure function, deterministic, with progress hook and timing | Determinism test (repeat runs are bitwise equal) |
| 38.8 | `continuity/visualization.py` | Cluster colour map, synthetic bridges in a distinct colour, best path, nodes, curve plot. Matplotlib Agg, lazily imported | Golden-image smoke tests (size and non-empty only) |
| 38.9 | `tests/test_phase38_hci_axioms.py` | Reads `manifest.json` and asserts the specification §4 axioms family by family, with declared tolerances (monotone 0.01; resolution 5 %; noise 5 %, with the documented roughness exception) | All pass on CPU in under 3 min |
| 38.10 | Performance | Benchmark 2048² and 4096² masks against the §6 target (≤ 2 s and ≤ 1 GB at 2048²). Profile MCP and coverage merges | `tests/` performance smoke test, marked slow |
| 38.11 | Prototype lineage | Keep `hci.prototype.v0` only in the study harness (unchanged prototype run in a subprocess). It is never shipped in the product | Documented; no production import |

**Module layout** (matching [`hydride_connectivity_index.md`](hydride_connectivity_index.md) §6):

```text
src/microseg/evaluation/continuity/
    __init__.py          # exports config, contracts, analyze_continuity (no import-time work)
    config.py
    contracts.py
    preprocess.py
    linkage.py
    path.py
    skeleton_graph.py
    analyzer.py
    visualization.py
    serialization.py
    synthetic.py         # already delivered in Phase 37
```

### Phase 39 — Interfaces and reporting (experimental flag)

| WP | Deliverable | Detail |
|---|---|---|
| 39.1 | CLI | `microseg-cli hci --mask <path or dir> --config configs/hci.default.yml --set ...`. Batch mode writes a per-image `hci_result.json`, `clusters.csv` and overlays, plus a run-level `report.json`, `summary.csv` and HTML. Progress lines give counts, percentage and ETA |
| 39.2 | Inference integration | Optional `result_export.continuity.enabled` in `inference.default.yml`. HCI runs on the exported indexed mask and records the run linkage |
| 39.3 | Correction loop | HCI is recomputed on the human-corrected mask with `mask_source = corrected`, so its before/after delta is visible in the correction export |
| 39.4 | Desktop (Qt) | A results-panel "Connectivity (experimental)" card: headline HCI_R and HCI_iso with δ½, a curve plot, and an overlay toggle with bridges in a distinct colour. **Help → Methods & Measurements** gains an HCI section. Computation runs off the UI thread with a progress card |
| 39.5 | Web (intranet) | Optional block in the result JSON and PDF report, behind a server flag that is off by default. Memory-only processing is preserved |
| 39.6 | Parity test | The same mask and config through the CLI, desktop service layer and web handler give identical JSON (gate G4) |
| 39.7 | Documentation | README usage; `algorithms.md`; GUI, CLI and web guides; a beginner on-ramp "Measuring hydride connectivity" added to `docs/index.md`; `tests/README.md` |

### Phase 40 — Scientific validation and manuscript

| WP | Deliverable | Detail |
|---|---|---|
| 40.1 | Realism benchmark v2 | Adds curved, tapered and stacked platelets, optical blur and noise before segmentation, and segmentation-model output instead of the ideal mask. This measures how segmentation error propagates into HCI |
| 40.2 | Real dataset | Specimens at three or more hydrogen levels (e.g. 25, 70, 200 ppm), plus the reorientation series (`book_chapter_files_on_hydriding/`). At least 10 fields per specimen, pixel size recorded, full wall coverage or stitching |
| 40.3 | Statistics | Bootstrap CIs over fields; mixed model (field in specimen); within-specimen CV; parameter elasticities; comparison of predicted and corrected masks |
| 40.4 | Literature baselines | RHF, RHCF, HCC and RHCP computed on the same masks. RHCP comes from Π_R^best with ε = 0.02, cross-checked against the MIT-licensed reference code of Simon et al. [2] on a subset |
| 40.5 | Expert ranking | Three or more experts rank blinded field sets; Kendall τ against HCI and each baseline |
| 40.6 | Property correlation | Ring-compression or DBTT/DHC data nominated by the owner. Incremental R² of HCI beyond Af and Fn |
| 40.7 | Validation report | `docs/hci_validation_report.md` plus a machine-readable report. Promotion decision from experimental to supported (gate G6) |
| 40.8 | Manuscript | Outline in §5; figures regenerated from committed scripts only |

## 3. Acceptance Gates (from the specification §12)

| Gate | Phase | Evidence artefact |
|---|---|---|
| G1 Axioms | 38 | `tests/test_phase38_hci_axioms.py` and a `report.json` from the harness |
| G2 Symmetry, resolution and noise | 38 | Same test module, invariance family |
| G3 Topology ground truth | 38 | `tests/test_phase38_hci_topology.py` |
| G4 Interface parity | 39 | `tests/test_phase39_hci_parity.py` |
| G5 Performance | 38 | Slow-marked performance test and a profile note |
| G6 Real-data validation | 40 | Validation report, owner sign-off |
| G7 Documentation sync | 39–40 | Docs navigation test extended to the new pages |

## 4. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| δ_max has no accepted physical value | Comparability across studies | Report δ½ (independent of δ_max) and the full curve; sensitivity study; fixed default recorded in every result |
| Field narrower than the wall thickness | HCI_R saturates or is truncated | Border-truncation flag; recommend stitched full-wall fields; L_u = wall thickness |
| Segmentation merges touching hydrides or breaks thin ones | Biased linkage | The integral over δ absorbs sub-δ breaks; realism benchmark v2 quantifies the effect; the correction loop allows repair |
| 8-neighbour metric anisotropy in Π | Up to about 8 % directional bias | Optional fast-marching solver; Π remains a companion, not the headline |
| Reviewers see overlap with RHCP | Novelty challenge | Position Π as RHCP-type; claim novelty only for linkage-integral HCI, δ½, the closure identity and the constant-Af benchmark with axioms (specification §2.3) |
| Name confusion (continuity or connectivity) | Communication | Owner decision D2; the specification recommends "Connectivity" for HCI and "continuity" for Π |

## 5. Manuscript Outline (working title)

**"A bounded, threshold-free hydride connectivity index validated on a constant-area-fraction
synthetic benchmark"**

1. Introduction: hydride embrittlement; amount, orientation and connectivity; limits of RHF,
   RHCF, HCC and RHCP; the need for axioms and controlled validation.
2. Methods:
   1. Connectivity function, integral HCI, δ½ and ξ.
   2. Path continuity (RHCP-type) and skeleton topology, including the closure identity.
   3. Synthetic benchmark design.
   4. Real specimens and imaging.
   5. Statistics.
3. Results:
   1. Axiom scorecard for HCI v1, Π, RHCF, RHCP and the prototype.
   2. δ½ recovering the designed gaps.
   3. Robustness to resolution and noise.
   4. Real specimens: repeatability and discrimination.
   5. Correlation with the mechanical response beyond Af and Fn.
4. Discussion: physical meaning of δ_max; loops and continuity; percolation behaviour; limits of
   2D sections; transfer to other networked microstructures.
5. Conclusions.
6. Data and code availability: MicroSeg release, the benchmark DOI (e.g. a Zenodo deposit of
   `hci_synthetic_v1` and the study harness).

**Planned figures**

1. Concept and algorithm flow (`docs/diagrams/hci_v1_algorithm_flow.svg`).
2. The benchmark overview.
3. The candidate evidence panel.
4. Connectivity curves with δ½ markers.
5. Real micrographs with cluster and bridge overlays.
6. HCI against the mechanical property.

## 6. Immediate Next Actions

1. The owner reviews the specification §13 decisions D1–D8 and signs off.
2. Owner and student nominate the real dataset (40.2) and the mechanical test series (40.6).
3. Open Phase 38 with WP 38.1–38.4, the shortest path to a usable HCI_R, and wire the axiom
   suite (38.9) immediately so every later change is gated.
