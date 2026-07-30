# Phase 36 Closeout: Memory-Safe Interfaces and HCI Candidate Specification

## Outcome

Phase 36 documents the proposed Hydride Connectivity Index (HCI) without exposing it as a
production metric, hardens the intranet application for memory-only processing, and improves
scientific guidance and job visibility in both user interfaces.

## Delivered

- [`hydride_connectivity_index.md`](hydride_connectivity_index.md) records the prototype
  formulation, implementation audit, unresolved scientific decisions, modular architecture,
  validation program, and promotion gates.
- [`diagrams/hydride_connectivity_index_candidate_workflow.svg`](diagrams/hydride_connectivity_index_candidate_workflow.svg)
  is the static, tracked development-path diagram.
- Array-native predictor and pipeline contracts let conventional and trained inference operate
  without temporary source files.
- The web application enforces a 5 MB image-byte limit, decoded-pixel ceiling, supported real
  image formats, matching content and filename extension, and single-frame input.
- Web jobs validate before queueing, run in bounded memory, publish ordered progress events, and
  expire terminal reports without an upload directory or result database.
- The web workspace shows a progress bar, stage, current message, and timestamped processing log.
  The downloadable JSON includes timing, privacy provenance, parameters, and the processing log.
- The web Help page includes the Fn formulations, workflow SVG, input contract, privacy model,
  algorithms, and reporting guidance.
- The Qt desktop application exposes **Help → Methods & Measurements** and retains its responsive
  progress/ETA card and continuous local desktop log.
- Qt shutdown now joins auxiliary warm-load and results threads and suppresses a deferred startup
  warm-load after a window has closed.

## HCI Status

HCI remains on a development hold. It is not registered as a metric and is not shown in CLI,
desktop, web, or exported production reports. Owner approval is still required for:

1. area and length definitions;
2. real versus synthetic topology rules;
3. branching and association semantics;
4. single-cluster and zero-denominator behavior;
5. pruning policy and calibrated units;
6. validation datasets, expert ranking, uncertainty, and acceptance thresholds.

## Verification

- Full repository suite: `268 passed, 15 warnings`.
- Web regression plus new memory-job tests: `54 passed`.
- Qt settings smoke after thread-lifecycle fix: `8 passed` with a successful process exit.
- Live browser verification: example-image conventional segmentation completed through the
  asynchronous endpoint; progress events, results, Fn values, downloads, Help SVG, and zero browser
  console errors were confirmed.
- Live Windows Qt verification: model warm-load, sample loading, progress/ETA, ML inference, mask
  rendering, desktop log, and the scientific guide were confirmed.
- Both new SVGs were rendered and inspected at their intended display sizes.

## Remaining Gaps

- HCI scientific decisions and validation are intentionally unresolved.
- Browser jobs are process-local; restarting the host discards them by design.
- Authentication and cross-process job coordination remain deployment concerns for a future
  multi-host or untrusted-network product.
- Existing `skimage.morphology.binary_dilation` deprecation warnings remain outside this phase.
