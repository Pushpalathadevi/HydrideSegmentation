"""Application assembly and runtime lifecycle helpers."""

from .desktop_workflow import DesktopRunRecord, DesktopWorkflowManager
from .facade import run_request
from .hpc_ga import (
    HpcGaBundleResult,
    HpcGaPlanConfig,
    generate_hpc_ga_bundle,
    parse_feedback_sources,
    summarize_feedback_sources,
)
from .orchestration import OrchestrationCommandBuilder
from .report_review import compare_run_reports, summarize_run_report
from .project_state import (
    ProjectLoadResult,
    ProjectSaveRequest,
    ProjectStateStore,
)
from .workflow_profiles import read_workflow_profile, write_workflow_profile

__all__ = [
    "DesktopRunRecord",
    "DesktopWorkflowManager",
    "generate_hpc_ga_bundle",
    "HpcGaBundleResult",
    "HpcGaPlanConfig",
    "parse_feedback_sources",
    "OrchestrationCommandBuilder",
    "compare_run_reports",
    "ProjectLoadResult",
    "ProjectSaveRequest",
    "ProjectStateStore",
    "read_workflow_profile",
    "run_request",
    "summarize_feedback_sources",
    "summarize_run_report",
    "write_workflow_profile",
]
