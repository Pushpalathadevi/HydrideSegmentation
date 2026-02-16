"""Application assembly and runtime lifecycle helpers."""

from .desktop_workflow import DesktopRunRecord, DesktopWorkflowManager
from .facade import run_request

__all__ = ["DesktopRunRecord", "DesktopWorkflowManager", "run_request"]
