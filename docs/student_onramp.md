# Beginner On-Ramp And Study Guide

## Why This Exists

This repository is meant to be usable by students and new contributors, not only by the original authors. This page gives a safe reading order, a practical first workflow, and a short glossary so the rest of the documentation is easier to follow.

## Recommended Reading Order

If you are new to the project, read these in order:

1. [`docs/mission_statement.md`](mission_statement.md)
2. [`docs/documentation_principles.md`](documentation_principles.md)
3. [`docs/learning_path.md`](learning_path.md)
4. [`docs/glossary.md`](glossary.md)
5. [`docs/student_notebooks.md`](student_notebooks.md)
6. [`docs/usage_commands.md`](usage_commands.md)
7. [`docs/why_tradeoffs.md`](why_tradeoffs.md)
8. [`docs/results_analysis.md`](results_analysis.md)
9. [`docs/scientific_validation.md`](scientific_validation.md)

That sequence goes from purpose, to vocabulary, to guided practice, to method choice, to interpretation.

## First Practical Workflow

Use this sequence when you want your first successful run:

1. Open the sample image or the bundled notebook tutorial data.
2. Run the classical baseline or preprocessing step first.
3. Inspect the mask and overlay visually.
4. Train or load the lightweight ML baseline.
5. Compare the baseline with the ML result.
6. Read the metrics together with the image outputs.
7. Only then change hyperparameters or switch architectures.

This order prevents the common mistake of tuning a complex model before you understand the baseline.
If you want a lower-risk sandbox before opening the GUI, start with the student notebooks and work through the sample data there first.

## What Students Should Look For

When reading a result, do not ask only "Did the IoU go up?"

Also ask:

- Did the boundaries look more realistic?
- Did the model preserve thin plates?
- Did it merge nearby features?
- Did the size distribution become physically plausible?
- Did the orientation statistics remain sensible?
- Did the improvement hold across multiple runs or only one run?

## Use The Glossary

The short glossary now lives in [`glossary.md`](glossary.md) so this page can stay focused on the learning path and the first practical workflow.

## How To Critically Analyze A Model

Use this checklist when comparing models:

- Does the method match the data regime?
- Is the improvement visible or only numerical?
- Are the boundaries clean and stable?
- Are the outputs reproducible across seeds?
- Is the model explainable enough for the intended use?
- Is the runtime acceptable on the target hardware?
- Does the model require hidden tuning that is not documented?
- Are the artifacts sufficient to reproduce the run later?

## Best Practices For Students

- Start with the simplest baseline first.
- Keep a notebook of exact commands and configs.
- Change one parameter family at a time.
- Save both images and metrics.
- Read the failure cases, not just the best examples.
- Prefer documented defaults over guessed values.
- Treat hidden assumptions as experimental variables.

## Good Questions To Ask In A Lab Meeting

- Why did this architecture help on this dataset?
- Which failure mode did it reduce?
- What did it cost in runtime or memory?
- Was the result stable across seeds?
- Is the gain scientifically meaningful or just numerically small?
- Could the same gain be achieved with a simpler model?

## If You Only Have Ten Minutes

Read:

- the mission statement,
- the learning path,
- the glossary,
- the student notebooks,
- the usage commands page,
- the tradeoffs page.

That is enough to start using the repository responsibly.
