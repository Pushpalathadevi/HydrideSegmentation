# Mission Statement

## Project Mission

Build a scientifically robust, extensible, and deployable platform for microstructural image segmentation and quantitative analysis.

Hydride segmentation is the first validated workflow. The architecture must support additional targets without rewriting core infrastructure.

## Vision

Deliver a field-ready local desktop product and backend toolkit that can:
- run inference from registered models
- support human correction with efficient annotation tools
- export corrections into training-ready datasets
- run reproducible experiments and evaluations
- continuously improve model quality through data curation loops

## Strategic Scope (Current)

- CPU-first local workflows
- GPU-compatible training and inference with safe CPU fallback
- Desktop GUI and CLI parity for core operations
- Config-driven execution with YAML and command-line overrides
- Reproducibility logging and schema-versioned outputs

## Non-Goals (Current Phases)

- Web-first deployment as primary interface
- Cloud dependence for mandatory workflows
- GPU-only assumptions for baseline usage

## Core Objectives

1. Deliver high-quality segmentation inference through a pluggable model registry.
2. Enable rapid and auditable human correction in GUI.
3. Export indexed/color correction outputs with provenance metadata.
4. Provide deterministic dataset packaging and split generation.
5. Support restartable project sessions with intermediate state persistence.
6. Keep deployment local, installable, and operationally reliable.
7. Maintain semantic versioning and release notes for deployed users.
8. Ensure GPU acceleration is opt-in/configurable and automatically falls back to CPU when unavailable.

## Success Criteria

- Any supported model can be registered and selected with short + detailed descriptions.
- Users can delete wrongly segmented features and redraw corrected annotations by class index.
- Exports include indexed masks, optional colorized masks, and correction metadata.
- YAML configs and `--set` overrides are consistently applied in CLI and GUI workflows.
- Saved project sessions can be reopened and resumed without data loss.
- Hydride workflows remain stable while generalization is implemented.
