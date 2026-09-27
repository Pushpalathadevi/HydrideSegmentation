"""Hydride Connectivity Index (HCI): microstructural connectivity of segmented features.

Public API:

* :func:`analyze_continuity` — the single computational core used by every
  interface (CLI, desktop, web);
* :class:`ContinuityAnalysisConfig` and :class:`Calibration` — inputs;
* :class:`ContinuityAnalysisResult` — the ``microseg.hci_result.v1`` output;
* :mod:`.synthetic` — the constant-area-fraction validation benchmark.

The theory and algorithm are documented in ``docs/hci_specification.md`` and
``docs/hci_theory.md``. Importing this package has no side effects; heavy
dependencies are imported inside functions.
"""

from .analyzer import analyze_continuity, binarize, clean_mask
from .config import (
    DIRECTIONS,
    FORMULATION_ID,
    RESULT_SCHEMA_VERSION,
    Calibration,
    ContinuityAnalysisConfig,
)
from .contracts import ContinuityAnalysisResult, ContinuityClusterResult
from .synthetic import (
    BENCHMARK_SCHEMA_VERSION,
    GENERATOR_VERSION,
    CenterlineNetwork,
    SyntheticBenchmarkConfig,
    SyntheticSample,
    generate_benchmark,
    network_ground_truth,
    rasterize_network,
    write_benchmark,
)

__all__ = [
    "BENCHMARK_SCHEMA_VERSION",
    "DIRECTIONS",
    "FORMULATION_ID",
    "GENERATOR_VERSION",
    "RESULT_SCHEMA_VERSION",
    "Calibration",
    "CenterlineNetwork",
    "ContinuityAnalysisConfig",
    "ContinuityAnalysisResult",
    "ContinuityClusterResult",
    "SyntheticBenchmarkConfig",
    "SyntheticSample",
    "analyze_continuity",
    "binarize",
    "clean_mask",
    "generate_benchmark",
    "network_ground_truth",
    "rasterize_network",
    "write_benchmark",
]
