# Foundation Strategy And Standards

## Non-Negotiables

- Scientific traceability for every exported artifact
- Deterministic behavior for splits and stochastic transforms
- Config-first workflows with explicit overrides
- Thin UI layer over reusable core library modules
- No hidden fallback for critical scientific logic
- Timestamped progress logging + ETA for long-running jobs
- Machine-readable reports and optional HTML summaries for inspectability
- Repository-relative documentation links only
- Documentation is a product surface: exact commands, outputs, and scientific math must be documented with code changes
- Sphinx-based HTML and PDF documentation builds must remain reproducible from source-controlled files
- SVG diagrams are preferred for architecture, workflow, and GUI explanations

## Must-Have Capabilities

- Local CPU inference and analysis
- Human correction with efficient, auditable tools
- Class-index aware annotation and export
- Session persistence and restart
- Dataset packaging for retraining loops
- Comprehensive user and developer documentation
- Frozen checkpoint metadata registry and model-selection guidance
- Fixed + random validation sample tracking during training
- Leakage-aware deterministic split planning and dataset QA gates
- Deterministic auto-prepare path for unsplit source/masks datasets
- GA-driven HPC bundle generation for scalable multi-candidate GPU experiments

## Desirable Capabilities

- Pluggable augmentation policy editor
- Active-learning prioritization over correction queues
- Rich experiment dashboards for model comparison
- Cross-platform packaged installers

## Development Strategy

1. Contract-first core modules (`src/microseg`).
2. Keep legacy wrappers until replacement parity and tests exist.
3. Build GUI as orchestration layer over tested services.
4. Add CLI parity for reproducibility and automation.
5. Promote phase exits only with tests and documentation complete.
6. Apply standards inspired by DeepImageDeconvolution and adapt them for segmentation workflows.
7. Use a standardized phase-gate closeout run before phase completion.

## Testing Strategy

- Unit tests for class maps, config merge, session actions
- Integration tests for inference->correction->export->package path
- GUI smoke workflows for load, run, correct, save, load, export

## Documentation Strategy

- User docs: GUI workflows, CLI examples, config conventions
- Developer docs: architecture, contracts, migration decisions
- Release docs: semver changes and behavior-impact notes
- Keep mission/roadmap docs as living strategy artifacts with phase-level updates
- Keep docs build commands visible in the repo root and in the documentation landing page
- Update status docs whenever a phase or workflow reaches a new state
