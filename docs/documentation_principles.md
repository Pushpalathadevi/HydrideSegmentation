# Documentation Principles

This repository treats documentation as a product surface, not an afterthought.

## Normative Rules

1. Every behavior change must update the relevant docs in the same change.
2. User-facing commands must be documented with exact invocations.
3. Research-facing algorithms must be documented with the math they implement.
4. Outputs must be documented with filenames, directory roots, and schema versions.
5. Diagrams should be static SVG-first so they remain crisp in HTML and PDF builds.
6. Documentation links must remain repository-relative inside the repo.
7. New workflows should explain how to run, how to validate, and how to interpret results.
8. When GUI features are introduced, they need schematic visuals or equivalent structure diagrams.
9. The docs system itself must stay reproducible through source-controlled build commands.
10. Documentation builds must not require internet for core rendering features; required frontend assets (for example MathJax) should be vendored minimally and versioned in-repo with explicit fallback behavior.
11. Classical segmentation docs must include a flow sheet, parameter table, typical values, and failure modes.
12. Model-family docs must include the original publication citation, architecture summary, comparison notes, and performance factors.
13. Internal model variants must be labeled as internal variants, not as canonical reproductions of an external paper unless they truly are.
14. New user-facing workflows should have a beginner or on-ramp page that points to the shortest safe reading path.
15. When defaults differ between legacy and modern code paths, the docs must name both and explain which one is canonical.
16. When a workflow is best learned hands-on, provide a runnable notebook tutorial that uses sample data and links back to the canonical docs.

17. Publication figures, flow sheets, and schematics should be authored or exported as static SVG assets in `docs/diagrams/` and referenced from the markdown pages rather than embedded as inline Mermaid blocks.

## Required Documentation Layers

- Mission and scope: why the project exists.
- Operational guides: how to run the code exactly.
- Results guides: where outputs land and how to analyze them.
- Scientific notes: metrics, assumptions, and algorithmic limits.
- Developer notes: extension points, contracts, and migration rules.
- Status notes: current progress, known gaps, and phase-specific constraints.
- Learning notes: recommended reading order, glossary terms, and first-run guidance for students.
- Learning labs: runnable notebook tutorials that exercise the repo on sample data.

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
