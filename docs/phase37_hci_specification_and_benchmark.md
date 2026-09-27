# Phase 37 Closeout: HCI Specification and Constant-Area-Fraction Benchmark

## Outcome

The intern's Hydride Connectivity Index work (`HARSHAD_WORK/`) has been converted into a
publication-grade specification backed by controlled evidence. The prototype formula is
**not** adopted; its intent is preserved in `hci.v1-candidate`. The v1 HCI is the integral,
over bridgeable matrix ligaments, of an area-weighted single-linkage connectivity function with
projected-coverage extents. It is reported together with the critical linking distance δ½,
RHCP-type path continuity and skeleton topology with an exact closure identity.

HCI is still **not** a product metric. No CLI, desktop or web behaviour changed in this phase.

## Delivered

| Item | Path |
|---|---|
| Specification and algorithm, with evidence | [`hci_specification.md`](hci_specification.md) |
| Synthetic benchmark documentation | [`hci_synthetic_benchmark.md`](hci_synthetic_benchmark.md) |
| Implementation, validation and manuscript plan (Phases 38–40) | [`hci_implementation_plan.md`](hci_implementation_plan.md) |
| Algorithm flow sheet | [`diagrams/hci_v1_algorithm_flow.svg`](diagrams/hci_v1_algorithm_flow.svg) |
| Benchmark overview figure | [`diagrams/hci_synthetic_benchmark_overview.svg`](diagrams/hci_synthetic_benchmark_overview.svg) |
| Candidate evidence figure | [`diagrams/hci_candidate_evidence.svg`](diagrams/hci_candidate_evidence.svg) |
| Generator library (no HCI metric exposed) | `src/microseg/evaluation/continuity/synthetic.py` |
| Generator CLI and config | `scripts/generate_hci_synthetic_benchmark.py`, `configs/hci_synthetic_benchmark.default.yml` |
| Committed benchmark (203 masks, about 2 MB) | `test_data/hci_synthetic_v1/` |
| Study harness (prototype run unchanged in a subprocess; v1 candidates) | `scripts/hci_candidate_study.py` |
| Study results | [`hci_candidate_study.report.json`](hci_candidate_study.report.json), [`hci_candidate_study.summary.json`](hci_candidate_study.summary.json) |
| Tests | `tests/test_phase37_hci_synthetic_benchmark.py` |

## Key Findings

- **Prototype.** At 5 % area fraction the prototype:
  - fails ordering in 4 of 5 gap-sweep replicates and in every arrangement replicate;
  - assigns 0 to rings, the honeycomb mesh and any single cluster;
  - varies by 15 % under pure rotation and by 7.7 % under a transpose;
  - times out (over 300 s) on percolating masks.
- **HCI_R v1.** It passes every ordering axiom in every replicate. Symmetries are exact;
  resolution changes it by ≤ 2.1 %; speckle, pinholes, spurs and breaks by ≤ 1.8 %. It
  separates random, stringer, stepped and continuous arrangements at equal area fraction. δ½
  recovers the designed gaps exactly in µm.
- **Open robustness point.** A harsh 10 % boundary-flip roughness changes HCI_R by 5.4 %, against
  a 5 % target. That is down from 17 % for a single-threshold index.
- **Topology extraction.** It matches ground truth except that degree-4 crossings occasionally
  split. Thickness-scaled junction collapse is specified for Phase 38.

## Owner Decisions Required

The specification §13 lists decisions D1–D8: adoption of v1, the public name, headline direction,
δ_max and δ0, reference length, topology reporting, the path-cost ratio, and the validation
dataset. Phase 38 starts after sign-off.

## Verification

See [`phase37_hci_specification_and_benchmark.report.json`](phase37_hci_specification_and_benchmark.report.json)
for the full-suite result, the new tests and the study statistics.

## Remaining Gaps

- The HCI core library, interfaces and axiom test suite are not yet implemented (Phases 38–39).
- There is no real-specimen validation, expert ranking or property correlation yet (Phase 40).
- The benchmark uses idealised straight capsules; realism benchmark v2 is planned.
- Citation details marked † in the specification must be re-verified before manuscript
  submission.
- The prototype source in `HARSHAD_WORK/` is untracked; its SHA-256 is recorded in the study
  report. The owner should decide whether to archive it in the repository.
