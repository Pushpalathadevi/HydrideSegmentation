# Contributing Guidelines

Thank you for considering a contribution to **HydrideSegmentation**.  This
project relies on clean object‑oriented Python code to remain compatible with
GUI applications and automated workflows.

## Coding Standards

- Use object‑oriented designs wherever possible.  Keep classes small and focused.
- Every public function or class must include a Python docstring.  Add inline
  comments where the logic is non‑trivial.
- Keep modules modular so they can be reused from the GUI and other tools.

## Testing Protocol

1. **Example images** – primary modules should be tested using the images in
   `test_data/`.
2. **Synthetic cases** – the `SegmentationEvaluator` can generate known ground
   truth masks for quick checks.
3. **Debug vs Regular modes** – when modules provide a `debug` flag, ensure both
   paths work:
   - *Regular mode* should run quickly and return only final results.
   - *Debug mode* should produce annotated intermediate outputs or plots for
     troubleshooting.
4. Document new test cases in the README whenever you add a new module or
   feature.

## Documentation Updates

Any enhancement or new script must be accompanied by updates to
`README.md` (project overview and usage) and, when relevant, this file.  Keep
the documentation clear so others can reproduce your workflow.
