"""Application assembly and runtime lifecycle helpers."""

from .desktop_workflow import DesktopRunRecord, DesktopWorkflowManager
from .desktop_result_export import DesktopResultExportConfig, DesktopResultExporter
from .desktop_ui_config import (
    BALANCED_METRIC_KEYS,
    REPORT_PROFILES,
    REPORT_SECTIONS,
    DesktopUIConfig,
    build_qt_stylesheet,
    default_desktop_ui_config,
    default_desktop_ui_config_path,
    load_desktop_ui_config,
)
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
    "DesktopResultExportConfig",
    "DesktopResultExporter",
    "DesktopUIConfig",
    "DesktopWorkflowManager",
    "REPORT_PROFILES",
    "REPORT_SECTIONS",
    "BALANCED_METRIC_KEYS",
    "build_qt_stylesheet",
    "default_desktop_ui_config",
    "default_desktop_ui_config_path",
    "generate_hpc_ga_bundle",
    "HpcGaBundleResult",
    "HpcGaPlanConfig",
    "load_desktop_ui_config",
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
