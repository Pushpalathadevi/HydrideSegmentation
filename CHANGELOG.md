# Changelog

All notable changes to MicroSeg are documented here. The project follows
[Semantic Versioning](https://semver.org/), starting with the first stable
release, v1.0.0.

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

[1.0.0]: https://github.com/Pushpalathadevi/HydrideSegmentation/releases/tag/v1.0.0
