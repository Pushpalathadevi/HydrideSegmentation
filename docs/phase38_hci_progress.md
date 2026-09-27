# Phase 38 Progress Record: HCI Implementation, Documentation, Deck and Release 2.0.0

This is the running progress record for the HCI implementation. If work is interrupted, resume
from the first unchecked item. Each entry is updated as soon as a step completes.

## Owner decisions (2026-09-27)

1. Adopt `hci.v1` (integrated linkage connectivity) instead of the intern formula.
2. Public name: **Hydride Connectivity Index**.
3. δ_max default = **1/5 of the smallest accepted hydride** (after mask clean-up), easily
   editable by users. Implemented as `max_bridge_distance: auto` with `auto_fraction: 0.2`,
   size measure = maximum Feret length of each cleaned component.
4. Real-world validation data is out of scope; the owner's team does it later from field data.
5. Documentation must be exceptionally clear, with properly typeset maths, in both the web-UI Help
   (KaTeX) and the Sphinx documentation.
6. Dedicated PowerPoint deck: problem, synthetic images and rationale, the algorithm step by step
   with intermediate images, caveats, the same SVGs as the docs, and HCI applied to the real
   images in the intern's report. Style: black title bar (Arial 26, bold, white), black bottom
   line (Arial 20, white), content text ≥ 18 pt Arial, and the graphic occupying the maximum area.
7. Implement as an optional attribute, **enabled by default in the web UI**. Commit and push as a
   major version bump (1.2.0 → **2.0.0**).
8. Keep this running progress record.

## Work items

| # | Item | Status | Notes |
|---|---|---|---|
| 0 | Progress record plus memory pointer | done | this file |
| 1 | Core library `src/microseg/evaluation/continuity/` (config, contracts, linkage, path, skeleton_graph, analyzer, visualization) | done | Topology exact against ground truth on all controlled masks, including degree-4 crossings (thickness-scaled junction collapse). 2048² runtime 8.4 s (path continuity 6 s); the 2 s design target is not met and is recorded as a gap |
| 2 | Axiom and topology tests on `test_data/hci_synthetic_v1` | done | `tests/test_phase38_hci_core.py` (33 tests; orderings use fixed δ_max = 10 µm) |
| 3 | CLI `microseg_cli.py hci` plus `configs/hci.default.yml` | done | Batch logic in `continuity/batch.py` |
| 4 | Web integration: request option (default on), δ_max control, result card, overlays, curve, JSON/PDF/XLSX export | done | Verified live on `hydride_optical_sample.png`: HCI R/C/iso 0.053/0.112/0.155; auto δ_max = 1.54 px (the smallest speck is about 8 px), and the topology identity residual is non-zero on a thin mask — both go to the deck |
| 5 | Web Help HCI section with KaTeX maths and SVGs | done | 63 formulas, 0 KaTeX errors; `hci_concept.svg` plus the algorithm flow SVG |
| 6 | Sphinx documentation with maths (specification converted to LaTeX maths; theory and algorithm page) | done | `hci_theory.md` (proofs, algorithm, complexity), `hci_user_guide.md`, specification formulas in LaTeX, clean Sphinx build |
| 7 | Re-run the benchmark evidence with the production library and the auto δ_max; update the specification §10 | done | `scripts/hci_benchmark_evaluation.py` → `docs/hci_evidence/`. δ_max rule comparison in `docs/hci_evidence/delta_max_rules/`. **Finding:** auto:min is hypersensitive to breaks (−61 %) and not comparable across images (arrangement 3/5, orientation 1/3). A percentile rule (P50) fixes the breaks sensitivity only; fixed 10 µm passes everything. Owner default kept; `auto_size_statistic` option added. Specification text still to update |
| 8 | Real images from the intern report: extract, compute HCI, record the results | done | `test_data/hci_report_images/` (Table 2 optical + masks), `scripts/hci_real_image_study.py` → `docs/hci_evidence/real_report_images/`. Prototype reproduced (2.44/3.12/6.97 vs reported 2.59/3.38/8.02). Topology bookkeeping fixes (1-px spurs, loops inside nodes) and a 4 px resolution floor; identity now exact on real and benchmark masks |
| 9 | PowerPoint deck (`docs/presentations/hci_review/hci_v1_review.pptx`) plus build script | done | Assets: `docs/presentations/hci_review/build_assets.py` → `assets/` (done). UI captures: `capture_ui.mjs` (headless Edge over CDP; done). Next: `build_deck.py` (python-pptx, SVG with PNG fallback), then QA through PowerPoint export |
| 10 | Version 2.0.0: pyproject, setup, version.py, installer .iss, packaging test, CHANGELOG, release notes, README | done | `docs/releases/v2.0.0.md` |
| 11 | Full test suite, phase gate, commit, push | done | 437 passed, 1 skipped; phase gate pass (strict) |

## Log

- 2026-09-27: Phase 37 (specification, benchmark, study) completed; see
  [`phase37_hci_specification_and_benchmark.md`](phase37_hci_specification_and_benchmark.md).
  Owner decisions received; Phase 38 started.
- 2026-09-27: Core library implemented and smoke-tested (item 1).
- 2026-09-27: Items 2 and 7 (evidence run). Added `auto_size_statistic` (min | percentile) after the δ_max rule study.
- 2026-09-27: Items 3–5 done (CLI, web integration, Help). Next: interface tests, Sphinx, real images, deck, release.
- 2026-09-27: Items 8 and the interface tests done (40 HCI tests pass). Topology fixes applied. Next: Sphinx docs (6), specification update (7), deck (9), release (10–11).
- 2026-09-27: Items 6–7 done. Specification §8, §10.9, §10.10 and §13 updated. Next: deck (9), release (10–11).
- Assets and UI captures for the deck done. Web server restarted after the topology fix (the residual flag has gone from the UI).
- Deck built: 31 slides, validated, rendered through PowerPoint and checked visually (filter-free SVG copy for PowerPoint; all text ≥ 18 pt Arial; notes on every slide). Next: release 2.0.0.
- Version bumped to 2.0.0; release notes and CHANGELOG written. Next: full tests, phase gate, commit, push.

## Closeout (2026-09-27)

Everything decided by the owner is implemented and released as **MicroSeg 2.0.0**. The
machine-readable closeout is
[`phase38_hci_implementation.report.json`](phase38_hci_implementation.report.json). Remaining gaps:

- field-data validation (the owner's team);
- the desktop (Qt) card;
- a faster path solver;
- fixed-δ_max comparison protocols.
- 2026-09-27: Committed 81e8d8d ("Release 2.0.0 …") and pushed to origin/main.
