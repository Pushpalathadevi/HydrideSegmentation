# Documentation Principles

This repository treats documentation as a product surface, not an afterthought.

## Normative Rules

1. Every behavior change must update the relevant docs in the same change.
2. User-facing commands must be documented with exact invocations.
3. Research-facing algorithms must be documented with the math they implement.
4. Outputs must be documented with filenames, directory roots, and schema versions.
5. Diagrams should be SVG-first so they remain crisp in HTML and PDF builds.
6. Documentation links must remain repository-relative inside the repo.
7. New workflows should explain how to run, how to validate, and how to interpret results.
8. When GUI features are introduced, they need schematic visuals or equivalent structure diagrams.
9. The docs system itself must stay reproducible through source-controlled build commands.

## Required Documentation Layers

- Mission and scope: why the project exists.
- Operational guides: how to run the code exactly.
- Results guides: where outputs land and how to analyze them.
- Scientific notes: metrics, assumptions, and algorithmic limits.
- Developer notes: extension points, contracts, and migration rules.
- Status notes: current progress, known gaps, and phase-specific constraints.

## Build Targets

The repository documentation system supports:

- HTML for interactive reading
- single-page HTML for archival and PDF conversion
- PDF for offline review and sharing

Use:

```bash
python scripts/build_docs.py
```

to generate the site locally.

