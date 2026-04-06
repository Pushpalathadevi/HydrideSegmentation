# Documentation Upgrade Plan

## Goal

This plan describes how the documentation should continue to evolve so the repository remains beginner-friendly, scientifically traceable, and easy to maintain as new segmentation methods are added.

## What Must Exist

Every major workflow should have all of the following:

- a user-facing how-to page,
- a scientific explanation page,
- a parameter reference table,
- a troubleshooting section,
- a comparison section,
- a beginner path link,
- an index entry on the docs landing page.

## Priority Order For New Documentation

1. Explain what the workflow is for.
2. Show the end-to-end flow sheet.
3. Explain each step in plain language.
4. List parameters, valid ranges, and typical values.
5. State failure modes and how to diagnose them.
6. Provide at least one publication-grade diagram.
7. Add citation links for every external architecture or method.
8. State whether the implementation is official, adapted, or internal-only.

## Near-Term Additions

### Conventional segmentation

- keep the detailed flow sheet current,
- add parameter examples for low-contrast and noisy images,
- add a short decision guide for when to use the rule-based baseline,
- make sure GUI defaults and library defaults stay clearly separated.

### ML model documentation

- keep the architecture foundation page synchronized with the actual supported backends,
- document the original paper for every family,
- call out internal variants explicitly,
- compare model families by capacity, context, and compute profile,
- document which settings affect performance the most.

### Beginner support

- keep the student on-ramp page short and practical,
- include a glossary of jargon,
- point new users to the simplest baseline first,
- show how to inspect outputs critically, not just how to run commands.

## Governance Rules

The following should be treated as documentation policy, not optional advice:

- any behavior change must update the relevant docs in the same change,
- any new backend must add or update a model reference entry,
- any new algorithm stage must be explained in the flow sheet,
- any new default parameter must be described in prose and in a table,
- any internal model variant must be labeled as internal, not as an original paper reproduction,
- any docs change that affects the site navigation must update the index page and docs README.

## File Ownership Map

The canonical documentation surfaces are:

- `AGENTS.md` for repository working rules,
- `docs/documentation_principles.md` for the docs contract,
- `docs/conventional_segmentation_pipeline.md` for the classical algorithm,
- `docs/model_architecture_manuscript_foundation.md` for model-family comparison,
- `docs/student_onramp.md` for student-friendly entry paths,
- `docs/usage_commands.md` for exact commands,
- `docs/results_analysis.md` for output interpretation,
- `docs/scientific_validation.md` for evaluation discipline,
- `docs/pretrained_model_catalog.md` for checkpoint provenance and citations.

## Suggested Future Enhancements

- add a glossary appendix to the docs site,
- add a "which model should I use?" decision tree,
- add family-level architecture diagrams as standalone SVGs,
- add short worked examples that compare the same image across multiple backends,
- add a documentation checklist for pull requests that change behavior.

