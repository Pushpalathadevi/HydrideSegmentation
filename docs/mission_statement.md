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
- Frozen-checkpoint metadata registry for model selection and guidance
- Checkpoint lifecycle management (`smoke`, `candidate`, `promoted`) with git-tracked metadata and git-ignored binaries
- Human-readable + machine-readable run reporting (`json` + `html`)
- Deployment-grade result reporting (`json` + `html` + `pdf`) for audit and handoff
- Offline installer workflows for enterprise desktop deployment (single-file setup artifacts)
- Mandatory end-of-phase quality gates (tests + stocktake + gap review + docs sync)
- HPC-ready orchestration artifact generation (scheduler scripts + manifests) for GPU environments
- Feedback-informed HPC sweep planning from prior evaluation reports (quality/runtime-aware)
- External SOTA transformer-model support with scratch initialization for offline HPC training

## Non-Goals (Current Phases)

- Web-first deployment as primary interface
- Cloud dependence for mandatory workflows
- GPU-only assumptions for baseline usage

## Core Objectives

1. Deliver high-quality segmentation inference through a pluggable model registry.
2. Enable rapid and auditable human correction in GUI.
3. Export indexed/color correction outputs with provenance metadata.
4. Provide deterministic, leakage-aware dataset packaging and split generation with global sample IDs.
5. Support restartable project sessions with intermediate state persistence.
6. Keep deployment local, installable, and operationally reliable.
7. Maintain semantic versioning and release notes for deployed users.
8. Ensure GPU acceleration is opt-in/configurable and automatically falls back to CPU when unavailable.
9. Track fixed and random validation exemplars per epoch to support scientific model progression analysis.
10. Keep user-facing and developer-facing documentation beginner-friendly and synchronized with implementation.
11. Enforce strict model metadata validation and leakage-aware dataset split/QA checks before training cycles.
12. Support explicit split datasets and unsplit source/masks datasets through deterministic auto-prepare workflows, including optional RGB-mask colormap conversion.
13. Provide GUI-native dataset onboarding with preview-first split planning and optional QA-gated training launch controls.
14. Provide GUI-native run review and report comparison to support evidence-based model iteration decisions.
15. Maintain a deterministic tiny smoke-checkpoint path to validate model plumbing on fresh systems without large artifacts.
16. Provide GUI/CLI generation of multi-candidate HPC script bundles from architecture/hyperparameter search definitions.
17. Provide feedback-aware candidate ranking for later HPC sweeps using prior run metrics and reproducible reporting artifacts.
18. Deliver enterprise-ready desktop UX with professional menus, bundled sample inputs, and persistent operational logging.
19. Provide installer-grade packaging instructions/scripts for offline Windows deployment.

## Success Criteria

- Any supported model can be registered and selected with short + detailed descriptions.
- Users can delete wrongly segmented features and redraw corrected annotations by class index.
- Exports include indexed masks, optional colorized masks, and correction metadata.
- YAML configs and `--set` overrides are consistently applied in CLI and GUI workflows.
- Training and evaluation runs emit progress/ETA logs, interruption-safe reports, and HTML summaries.
- Model registry metadata validates cleanly and packaged datasets pass QA gates before training.
- Tiny smoke checkpoint generation remains available for quick pipeline sanity checks.
- Saved project sessions can be reopened and resumed without data loss.
- Hydride workflows remain stable while generalization is implemented.
- Desktop users can export full run evidence packages (images + masks + metrics + HTML/PDF reports) without external tooling.
- Desktop workflows support optional spatial calibration so size statistics can be reported in physical units when scale is available.
- Documentation is treated as a first-class deliverable: commands, outputs, algorithms, and current status must be updated in the same change as behavior.
- The Sphinx docs site must remain buildable into HTML and PDF artifacts from repository sources.
- SVG diagrams and schematic visuals should be used liberally when they improve understanding of architecture, workflows, or GUI behavior.
