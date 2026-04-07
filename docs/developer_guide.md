# Developer Guide

This project is designed so that UI, orchestration, core logic, and report generation stay separate.

## Contribution Flow

1. Start from a targeted issue or phase objective.
2. Read the mission, architecture, and status docs.
3. Make the smallest vertical change that solves the problem.
4. Add or update tests for preserved behavior.
5. Update the docs in the same change.
6. Verify CPU-first behavior and reproducibility metadata.
7. Re-run the relevant command surface with the exact config shown in the docs.

## Architectural Boundaries

- `hydride_segmentation/` remains the compatibility layer.
- `src/microseg/` contains the contract-first reusable implementation.
- GUI code should orchestrate; it should not own heavy algorithmic logic.
- Output serialization belongs to dedicated exporter or workflow modules.
- Config resolution should stay versioned and deterministic.

## What To Update When Behavior Changes

- User-facing command syntax
- Output filenames or schemas
- Report structure
- Metrics definitions
- Validation gates
- Migration notes for legacy wrappers
- Documentation diagrams if the data flow changes

## Adding A New Workflow

When adding a new workflow, document:

- the entry point
- required configuration keys
- expected outputs
- failure modes
- reproducibility metadata
- downstream analysis path

If the workflow has a GUI representation, add:

- a schematic diagram
- a results-path description
- a short operator checklist

## Documentation Build Discipline

The docs are built with Sphinx and MyST Markdown from the repository sources.

Recommended local commands:

```bash
python scripts/build_docs.py --html-only
python scripts/build_docs.py
```

The generated HTML lives under `docs/_build/html/`.

## Review Checklist

- No hard-coded paths for user artifacts
- No hidden fallback for scientific steps
- No untracked schema changes
- Exact commands are documented
- Results paths are documented
- GUI model integration updates must be documented alongside registry, loader, and smoke-test changes
- New math is explained clearly
- SVG diagrams are updated when architecture or UX changes
