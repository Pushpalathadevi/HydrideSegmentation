# Changelog

All notable changes to MicroSeg are documented here. The project follows
[Semantic Versioning](https://semver.org/), starting with the first stable
release, v1.0.0.

## [Unreleased]

## [1.2.0] - 2026-09-20

### Added

- The web app can rotate a micrograph before segmenting it. Fn reads the
  horizontal image axis as the tube's circumferential direction and the vertical
  axis as radial, an assumption nothing in a micrograph can confirm, so an image
  captured in another frame previously returned an Fn measured against the wrong
  axes. A new **Straighten and rotate** editor under the chosen image offers
  90-degree turns and a fine-angle slider over guide lines, with a live preview.
  Quarter turns transpose the pixel array losslessly; other angles are resampled
  once and cropped to the largest upright rectangle inside the turned image, so
  no invented corner pixels are ever measured. The applied angle travels in the
  request as `rotation_deg`, appears in the run manifest as `rotation_deg`,
  `rotation_applied`, `uploaded_width` and `uploaded_height`, and is printed on
  the detailed report.

### Changed

- A chosen image now shows its thumbnail immediately, while the rest of the form
  is still being filled in, instead of only appearing once the run finished.
  TIFF micrographs, which browsers cannot decode, are previewed through a new
  `POST /api/preview` endpoint that renders and discards a JPEG thumbnail
  in memory; library images fall back to their cached thumbnail.

## [1.1.1] - 2026-09-16

### Changed

- Web results for images larger than `limits.max_long_side_px` are now returned at
  the uploaded image size. Segmentation still runs on the downscaled copy for
  speed; the mask is enlarged back with nearest-neighbour interpolation (label
  values stay exact), the input and overlay views use the full-resolution upload,
  and every measurement (areas, lengths, Fn, minimum feature size) refers to
  original-image pixels. Mask boundary detail is that of the downscaled image.
  The run manifest adds `output_width`, `output_height` and
  `mask_restored_to_original`; `width`/`height` still give the segmented size.
- The web upload limit (`limits.max_upload_mb`) default rises from 5 MB to 10 MB.
- Version metadata across Python, package, installer and documentation now reads
  1.1.1; the `v1.0.1` and `v1.1.0` tags were cut while these files still read 1.0.0.

### Fixed

- Declared `pypdf` in both supported test-install paths so a clean contributor
  environment can run the PDF report verification suite without a manual
  dependency repair. Runtime report generation remains backed by Matplotlib.
- Repaired source-distribution wheel builds by making `pyproject.toml` the dependency authority
  and including the supported requirements profiles in the source archive.
- Replaced scikit-image morphology aliases scheduled for removal with their
  behavior-equivalent supported APIs; regression snapshots continue to pin the
  conventional segmentation output.

## [1.1.0] - 2026-09-15

### Added

- Web "Download mask" offers `<stem>_mask_labels.png` (class IDs 0/1),
  `<stem>_mask_preview.png` (display 0/255) and `<stem>_masks.zip` with
  `microseg.mask_download.v1` metadata; the previous mask download is unchanged.

## [1.0.0] - 2026-08-05

### Added

- Stable Qt desktop application for local, CPU-first segmentation, correction,
  quantification, batch review, project resume, and scientific result export.
- Local intranet web application with memory-only uploads, asynchronous jobs,
  bundled examples, registered ML and conventional methods, and Fn analysis.
- Unified CLI for inference, training, evaluation, dataset preparation,
  checkpoint management, deployment validation, and phase gates.
- Single-file, per-user Windows offline installer containing both the windowed
  desktop launcher and a console CLI companion.
- Writable installed-runtime workspace under the user's local application-data
  directory, preserving local model overlays and checkpoint binaries across
  application upgrades.
- Packaged and installed executable smoke reports plus SHA-256 installer release
  metadata.

### Changed

- Promoted the software version from the pre-stable 0.x series to `1.0.0`.
- Made `src/microseg/version.py` the canonical version source while retaining
  the legacy package re-export for compatibility.
- Reworked the PyInstaller layout into an installable application directory
  with dedicated GUI and CLI launchers.
- Made installer compilation fail visibly when required build tools or expected
  artifacts are missing.

### Fixed

- Fixed the PyInstaller spec's unsupported reliance on `__file__`.
- Fixed the mismatch between the former one-file PyInstaller output and the
  Inno Setup script's expected application directory.
- Fixed installed desktop startup and background-job orchestration so they no
  longer require a source checkout or attempt to execute the GUI as Python.
- Prevented installed applications from writing logs, outputs, and local model
  metadata beneath the read-only installation directory.
- Fixed the web status indicator so `--no-preload` is reported as ready for
  on-demand loading instead of remaining on "Loading models..." indefinitely.

### Known limitations

- The v1.0.0 installer is not Authenticode-signed; Windows may show an unknown
  publisher warning until a signing certificate and release signing pipeline
  are configured.
- Large trained checkpoint binaries remain external local artifacts by design
  and are not committed to Git or embedded in the base installer.

[1.1.1]: https://github.com/Pushpalathadevi/HydrideSegmentation/releases/tag/v1.1.1
[1.1.0]: https://github.com/Pushpalathadevi/HydrideSegmentation/releases/tag/v1.1.0
[1.0.0]: https://github.com/Pushpalathadevi/HydrideSegmentation/releases/tag/v1.0.0
