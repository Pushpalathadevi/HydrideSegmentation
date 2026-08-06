# Phase 35 Closeout: Intranet Web Application

## Goal and delivered behavior

Colleagues on the office intranet needed to run segmentation from a browser without installing anything, on an air-gapped network. Phase 35 delivers a self-contained Flask application served from one Windows or Ubuntu host.

Delivered:

- `hydride_segmentation/web/`, an application factory with YAML plus environment configuration, a model catalog with startup warm-loading, a segmentation service, and an HTTP blueprint.
- Browser workspace with drag-and-drop upload, bundled example images, model selection, conventional parameter controls with per-parameter in-app help, result tabs (overlay, input, mask, orientation, size, angle), a measurements table, and PNG downloads.
- A dedicated in-app Help page covering quick start, method choice, every conventional parameter, result interpretation, image requirements, and troubleshooting.
- Both routes available: the conventional pipeline and any installed trained model, selected from the same registry the desktop GUI and CLI use, so all three surfaces offer identical models and produce identical results for the same inputs.
- Radial hydride fraction (Fn) as the headline result for both routes, reported length-weighted and count-based with numerators and denominators shown, a user-controlled angle threshold and minimum feature size, opt-in Fn classification QA views, grouped measurements, and a JSON export carrying the settings that produced the numbers.
- `scripts/run_web_server.py` launching waitress on both platforms, printing the intranet URLs to share, and falling back to the development server with a clear warning when waitress is absent.
- `deploy/microseg-web.service` and `deploy/start_web_server.bat` for run-as-a-service on either platform, plus `requirements-web.txt` and a `microseg-web` console entry point.
- `configs/app/web_server.default.yml` documenting every option inline, overridable by `MICROSEG_WEB_*` environment variables and command-line flags.

Air-gap guarantees, verified by test and by inspecting live page loads: the pages request only same-origin assets, the bundled CSS and JS contain no remote URLs and no `@import`, and the server makes no outbound connections. Uploaded images are decoded in memory, written to a private temporary file only for the duration of one request, and deleted immediately.

Operational safety: uploads are capped by `MAX_CONTENT_LENGTH` before being read, extension-checked and content-checked by decoding, oversized images are downscaled, and a bounded job pool returns an explicit busy response instead of letting requests pile onto the CPU.

## Performance work

Model preloading warms trained checkpoints into the shared bundle cache on a background thread at startup, so `/health` answers immediately and the first user request does not pay load cost.

Profiling the request path found two hot loops that were quadratic in feature count, both of the form `labels == idx` inside a per-component loop, which allocates and scans a full-image array once per detected feature:

- `src/microseg/evaluation/hydride_statistics.py` coloured the orientation map one feature at a time. Replaced with a single label-indexed lookup. Output is bitwise identical; that step went from 1.90 s to 0.03 s on a mask with 1386 features.
- `src/microseg/evaluation/hydride_metrics.py` ran fill, dilation, and skeletonization over the whole image per component. Replaced with a padded per-component bounding-box crop, padded by two pixels so a `disk(1)` dilation can never be clipped. Sizes and skeletons are identical; angles differ only by floating-point noise up to 2.6e-13 degrees because the covariance is computed on translated coordinates. Measured 39x to 152x faster depending on feature count.

A third instance was a correctness bug, not just a slow path. `render_fn_debug_visualizations` called `ax.contour` once per component with a full-image boolean array, and raised `MemoryError` on a mask with 1386 features, which also breaks the desktop result-export path that renders the same figure. Counted and uncounted features are now contoured as two unions, which draws exactly the same boundaries, centroids come from one vectorized `center_of_mass` pass, and per-feature annotations are capped at the 150 largest counted features with the title recording how many were left unlabelled. The figure now renders in 0.64 s where it previously crashed.

All three changes benefit the desktop GUI and the CLI equally, since they share this analysis code.

End-to-end effect on a 1024 x 768 optical micrograph, CPU only, analysis figures included: conventional 0.95 s to 0.73 s, trained UNet 3.24 s to 1.32 s.

## 2026-08 professional workspace and reporting enhancement

The deployed browser surface now closes the main review and handoff gaps identified after the
original phase closeout:

- the primary Run action is directly below image selection;
- the header displays the software version and opens a metadata-driven Downloads catalog;
- the completed result begins with aligned input/mask panes and a synchronized, focal-point zoom
  tool that resets with `Esc`;
- completed retained jobs generate a compact two-page scientific PDF or a ZIP with the PDF,
  provenance JSON, individual PNG views, and a formatted XLSX workbook carrying scalar metrics,
  per-feature data, histogram bins, and editable charts;
- the catalog validates repository-bound paths and displays file size and SHA-256 metadata for the
  local Windows installer, research publication, release notes, and future JSON-described assets.

The report and workbook remain in memory until downloaded. PDF pages were rendered with Poppler for
visual QA; the workbook was imported, inspected, formula-error scanned, and rendered through the
artifact validation workflow. Live browser QA confirmed identical scale and transform origin in both
comparison panes and verified `Esc` returns each pane to 1x.

## Verification and traceability

- New test module: `tests/test_phase35_web_app.py` (49 tests) covering configuration loading and environment overrides, missing-file and missing-sample fallbacks, offline asset guarantees, page rendering and embedded bootstrap JSON, help-page coverage of every control, all API endpoints, conventional and trained segmentation, parameter effect, downscaling, upload validation across six failure modes, oversized uploads, catalog defaults and warm loading, job-limiter concurrency, and Fn quantification: presence on both routes, threshold monotonicity, minimum-feature-size exclusion, manifest recording, opt-in classification views, dense-mask rendering that previously exhausted memory, metric grouping completeness, and threshold validation.
- Original closeout suite: 263 passed. Current enhancement gate: 333 passed, 1 skipped.
- Focused enhancement suite: `tests/test_web_downloads_and_reports.py` (5 passed).
- Live browser verification against a running waitress server: both routes executed end to end, all six result tabs populated, measurements rendered, downloads enabled, no console errors, and `performance.getEntriesByType("resource")` reported zero external requests.
- User documentation: [`intranet_web_app.md`](intranet_web_app.md).
- Machine-readable closeout: `docs/phase35_intranet_web_app.report.json`.

## Remaining gaps

- No authentication or per-user access control; the app assumes a trusted intranet and documents putting a reverse proxy in front when that assumption does not hold.
- Results are not calibrated to physical units in the browser, because the server has no access to image scale metadata; Fn is dimensionless and therefore unaffected, but lengths and areas are reported in pixels, and the desktop app remains the path for calibrated measurements.
- Fn is computed from the mask as segmented, so it inherits any segmentation error; the UI directs users to check the overlay and the Fn classification view rather than trusting the number alone.
- No batch upload in the browser; one image per request.
- Corrections and annotation editing are desktop-only; the browser is inference and review.
- Analysis figures are computed on every request rather than on demand, so disabling them is a configuration choice rather than a per-user toggle.
- Pre-existing scikit-image morphology deprecation warnings remain outside this phase.
