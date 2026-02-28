"""Machine-readable failure taxonomy for operational diagnostics."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FailureCodeDescriptor:
    """Canonical failure code metadata."""

    code: str
    category: str
    description: str


UNKNOWN_INTERNAL = "MICROSEG_UNKNOWN_INTERNAL"
INPUT_INVALID = "MICROSEG_INPUT_INVALID"
INPUT_NOT_FOUND = "MICROSEG_INPUT_NOT_FOUND"
IO_WRITE_FAILED = "MICROSEG_IO_WRITE_FAILED"

PREFLIGHT_DATASET_INVALID = "MICROSEG_PREFLIGHT_DATASET_INVALID"
PREFLIGHT_PRETRAINED_MISSING = "MICROSEG_PREFLIGHT_PRETRAINED_MISSING"
PREFLIGHT_MODEL_MISSING = "MICROSEG_PREFLIGHT_MODEL_MISSING"
PREFLIGHT_BENCHMARK_CONFIG_INVALID = "MICROSEG_PREFLIGHT_BENCHMARK_CONFIG_INVALID"

DEPLOY_PACKAGE_INVALID = "MICROSEG_DEPLOY_PACKAGE_INVALID"
DEPLOY_MODEL_RESOLVE_FAILED = "MICROSEG_DEPLOY_MODEL_RESOLVE_FAILED"
DEPLOY_MODEL_LOAD_FAILED = "MICROSEG_DEPLOY_MODEL_LOAD_FAILED"
DEPLOY_PREPROCESS_FAILED = "MICROSEG_DEPLOY_PREPROCESS_FAILED"
DEPLOY_INFERENCE_FAILED = "MICROSEG_DEPLOY_INFERENCE_FAILED"
DEPLOY_OUTPUT_WRITE_FAILED = "MICROSEG_DEPLOY_OUTPUT_WRITE_FAILED"


FAILURE_CODE_REGISTRY: dict[str, FailureCodeDescriptor] = {
    UNKNOWN_INTERNAL: FailureCodeDescriptor(
        code=UNKNOWN_INTERNAL,
        category="internal",
        description="Unexpected internal error.",
    ),
    INPUT_INVALID: FailureCodeDescriptor(
        code=INPUT_INVALID,
        category="input",
        description="Input argument or payload is invalid.",
    ),
    INPUT_NOT_FOUND: FailureCodeDescriptor(
        code=INPUT_NOT_FOUND,
        category="input",
        description="Required input path does not exist.",
    ),
    IO_WRITE_FAILED: FailureCodeDescriptor(
        code=IO_WRITE_FAILED,
        category="io",
        description="Failed to write output artifact to disk.",
    ),
    PREFLIGHT_DATASET_INVALID: FailureCodeDescriptor(
        code=PREFLIGHT_DATASET_INVALID,
        category="preflight",
        description="Dataset preflight check failed.",
    ),
    PREFLIGHT_PRETRAINED_MISSING: FailureCodeDescriptor(
        code=PREFLIGHT_PRETRAINED_MISSING,
        category="preflight",
        description="Pretrained readiness check failed.",
    ),
    PREFLIGHT_MODEL_MISSING: FailureCodeDescriptor(
        code=PREFLIGHT_MODEL_MISSING,
        category="preflight",
        description="Required model artifact missing in preflight.",
    ),
    PREFLIGHT_BENCHMARK_CONFIG_INVALID: FailureCodeDescriptor(
        code=PREFLIGHT_BENCHMARK_CONFIG_INVALID,
        category="preflight",
        description="Benchmark config preflight validation failed.",
    ),
    DEPLOY_PACKAGE_INVALID: FailureCodeDescriptor(
        code=DEPLOY_PACKAGE_INVALID,
        category="deployment",
        description="Deployment package manifest or checksums are invalid.",
    ),
    DEPLOY_MODEL_RESOLVE_FAILED: FailureCodeDescriptor(
        code=DEPLOY_MODEL_RESOLVE_FAILED,
        category="deployment",
        description="Deployment package model artifact could not be resolved.",
    ),
    DEPLOY_MODEL_LOAD_FAILED: FailureCodeDescriptor(
        code=DEPLOY_MODEL_LOAD_FAILED,
        category="deployment",
        description="Model load failed before inference.",
    ),
    DEPLOY_PREPROCESS_FAILED: FailureCodeDescriptor(
        code=DEPLOY_PREPROCESS_FAILED,
        category="deployment",
        description="Input preprocessing failed in deployment runtime.",
    ),
    DEPLOY_INFERENCE_FAILED: FailureCodeDescriptor(
        code=DEPLOY_INFERENCE_FAILED,
        category="deployment",
        description="Inference failed in deployment runtime.",
    ),
    DEPLOY_OUTPUT_WRITE_FAILED: FailureCodeDescriptor(
        code=DEPLOY_OUTPUT_WRITE_FAILED,
        category="deployment",
        description="Deployment runtime failed writing output artifacts.",
    ),
}


def normalize_failure_code(code: str | None) -> str:
    """Return known code or fallback to UNKNOWN_INTERNAL."""

    value = str(code or "").strip()
    if not value:
        return UNKNOWN_INTERNAL
    if value in FAILURE_CODE_REGISTRY:
        return value
    return UNKNOWN_INTERNAL


def classify_exception(exc: Exception, *, stage: str = "") -> str:
    """Map a Python exception to canonical failure taxonomy code."""

    text = str(exc).lower()
    if isinstance(exc, FileNotFoundError):
        return INPUT_NOT_FOUND
    if isinstance(exc, PermissionError):
        return IO_WRITE_FAILED
    if isinstance(exc, ValueError):
        if "package" in text or "manifest" in text:
            return DEPLOY_PACKAGE_INVALID
        return INPUT_INVALID
    if isinstance(exc, RuntimeError):
        if "validation failed" in text or "package" in text:
            return DEPLOY_PACKAGE_INVALID
        if "load" in text and "model" in text:
            return DEPLOY_MODEL_LOAD_FAILED
        if "inference" in text or "predict" in text:
            return DEPLOY_INFERENCE_FAILED
    if stage == "deploy_preprocess":
        return DEPLOY_PREPROCESS_FAILED
    if stage == "deploy_inference":
        return DEPLOY_INFERENCE_FAILED
    if stage == "deploy_output":
        return DEPLOY_OUTPUT_WRITE_FAILED
    return UNKNOWN_INTERNAL
