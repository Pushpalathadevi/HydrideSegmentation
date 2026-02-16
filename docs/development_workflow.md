# Development Workflow Procedure

## Working Procedure for New Features

1. Define the scientific and user objective.
2. Check alignment with mission and architecture docs.
3. Add or update ADR if a major design choice is involved.
4. Implement smallest vertical slice with tests.
5. Add regression tests for preserved behavior.
6. Update docs in the same change.
7. Validate CPU-only local run path.

## Pull Request Checklist

- Scope clearly stated
- Tests added or updated
- Backward compatibility addressed
- Documentation synchronized
- Reproducibility metadata preserved

## Release Readiness Checklist

- Core workflows verified on reference sample images
- Local app install/run validated on clean environment
- Known limitations listed in release notes
