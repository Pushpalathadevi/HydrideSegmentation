# Migration Strategy (From Hydride-Specific to General Platform)

## Strategy Summary

Use a strangler pattern:
- Keep current working package alive.
- Move shared logic into new core modules incrementally.
- Route existing entry points through adapters.
- Decommission legacy paths only after parity tests pass.

## Immediate Priority Changes

1. Make package imports resilient without GUI extras.
2. Unify inference routing through one orchestrator.
3. Introduce model registry abstraction.
4. Standardize output manifest and metadata schema.
5. Add correction export schema and file layout.

## Compatibility Contract

Current state after Phase 3:
- Existing CLI commands remain available.
- Existing API endpoints remain callable.
- Qt is the primary GUI path (`hydride-gui` default), with Tk available as compatibility fallback (`hydride-gui --framework tk`).
