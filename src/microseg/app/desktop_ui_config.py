"""Desktop Qt UI config loading and stylesheet generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.microseg.io import load_yaml_config


SCHEMA_VERSION = "microseg.desktop_ui_config.v1"
REPORT_PROFILES = ("balanced", "full", "audit")
REPORT_SECTIONS = (
    "metadata",
    "calibration",
    "key_summary",
    "scalar_table",
    "distribution_charts",
    "overlays",
    "diff_panel",
    "artifact_manifest",
)

BALANCED_METRIC_KEYS = (
    "hydride_area_fraction_percent",
    "hydride_count",
    "hydride_total_area_pixels",
    "size_mean_pixels",
    "orientation_mean_deg",
    "excluded_small_features",
)

AUDIT_METRIC_KEYS = (
    "hydride_area_fraction_percent",
    "hydride_count",
    "hydride_total_area_pixels",
    "equivalent_diameter_mean_px",
    "orientation_mean_deg",
    "orientation_std_deg",
    "orientation_alignment_index",
    "orientation_entropy_bits",
    "excluded_small_features",
    "min_feature_pixels",
)


@dataclass(frozen=True)
class DesktopAppearanceConfig:
    """Qt desktop visual style defaults."""

    base_font_size: int = 16
    heading_font_size: int = 18
    monospace_font_size: int = 15
    menu_font_size: int = 15
    tab_font_size: int = 15
    toolbar_font_size: int = 15
    status_font_size: int = 14
    control_padding_px: int = 5
    panel_spacing_px: int = 8
    table_row_padding_px: int = 6
    table_min_row_height_px: int = 24
    high_contrast: bool = False


@dataclass(frozen=True)
class DesktopWindowConfig:
    """Desktop window sizing and geometry behavior defaults."""

    initial_width: int = 1880
    initial_height: int = 1180
    minimum_width: int = 1360
    minimum_height: int = 860
    left_dock_width: int = 320
    right_dock_width: int = 360
    workflow_dock_width: int = 1180
    remember_geometry: bool = True
    clamp_to_screen: bool = True
    start_maximized: bool = True
    start_fullscreen: bool = False
    show_workflow_dock_on_start: bool = False
    show_log_dock_on_start: bool = False


@dataclass(frozen=True)
class DesktopExportDefaultsConfig:
    """Default report/export behavior for desktop results package generation."""

    report_profile: str = "balanced"
    write_html_report: bool = True
    write_pdf_report: bool = True
    write_csv_report: bool = True
    write_batch_summary: bool = True
    selected_metric_keys: tuple[str, ...] = BALANCED_METRIC_KEYS
    include_sections: tuple[str, ...] = REPORT_SECTIONS
    sort_metrics: str = "name"
    top_k_key_metrics: int = 12
    include_artifact_manifest: bool = True
    compute_required_metrics: bool = True
    compute_extended_metrics: bool = False
    write_orientation_map: bool = True
    write_distribution_charts: bool = False
    write_physical_calibration_metrics: bool = False


@dataclass(frozen=True)
class DesktopUIConfig:
    """Complete desktop UI + reporting defaults payload."""

    schema_version: str = SCHEMA_VERSION
    appearance: DesktopAppearanceConfig = DesktopAppearanceConfig()
    window: DesktopWindowConfig = DesktopWindowConfig()
    export_defaults: DesktopExportDefaultsConfig = DesktopExportDefaultsConfig()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_desktop_ui_config() -> DesktopUIConfig:
    """Return built-in desktop UI config defaults."""

    return DesktopUIConfig()


def default_desktop_ui_config_path() -> Path:
    """Return repo default desktop UI config path."""

    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "configs" / "app" / "desktop_ui.default.yml"


def _as_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(default)


def _clamp_int(
    value: object,
    *,
    default: int,
    minimum: int,
    maximum: int,
    key: str,
    warnings: list[str],
) -> int:
    try:
        parsed = int(value)
    except Exception:
        warnings.append(f"{key}: invalid value {value!r}; using default={default}")
        return int(default)
    if parsed < int(minimum):
        warnings.append(f"{key}: clamped {parsed} -> {minimum}")
        return int(minimum)
    if parsed > int(maximum):
        warnings.append(f"{key}: clamped {parsed} -> {maximum}")
        return int(maximum)
    return int(parsed)


def _to_str_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        items = [part.strip() for part in value.split(",")]
        return tuple(part for part in items if part)
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            text = str(item).strip()
            if text:
                out.append(text)
        return tuple(out)
    return ()


def _sanitize_report_profile(value: object, *, default: str, warnings: list[str]) -> str:
    text = str(value).strip().lower()
    if text in REPORT_PROFILES:
        return text
    warnings.append(f"export_defaults.report_profile: unsupported value {value!r}; using {default!r}")
    return str(default)


def _sanitize_sort_metrics(value: object, *, default: str, warnings: list[str]) -> str:
    text = str(value).strip().lower()
    if text in {"name", "as_is", "value_desc"}:
        return text
    warnings.append(f"export_defaults.sort_metrics: unsupported value {value!r}; using {default!r}")
    return str(default)


def _sanitize_sections(value: object, *, default: tuple[str, ...], warnings: list[str]) -> tuple[str, ...]:
    requested = _to_str_tuple(value)
    if not requested:
        return tuple(default)
    allowed = set(REPORT_SECTIONS)
    selected: list[str] = []
    for item in requested:
        key = str(item).strip()
        if key in allowed:
            selected.append(key)
        else:
            warnings.append(f"export_defaults.include_sections: ignored unsupported section {item!r}")
    if not selected:
        warnings.append("export_defaults.include_sections: empty after filtering; using defaults")
        return tuple(default)
    return tuple(selected)


def _sanitize_metric_keys(value: object, *, fallback_profile: str) -> tuple[str, ...]:
    requested = _to_str_tuple(value)
    if requested:
        # preserve user order, remove duplicates
        dedup: list[str] = []
        seen: set[str] = set()
        for key in requested:
            if key in seen:
                continue
            seen.add(key)
            dedup.append(key)
        return tuple(dedup)
    if fallback_profile == "audit":
        return AUDIT_METRIC_KEYS
    if fallback_profile == "full":
        return ()
    return BALANCED_METRIC_KEYS


def _sanitize_appearance(payload: dict[str, Any], *, base: DesktopAppearanceConfig, warnings: list[str]) -> DesktopAppearanceConfig:
    return DesktopAppearanceConfig(
        base_font_size=_clamp_int(
            payload.get("base_font_size", base.base_font_size),
            default=base.base_font_size,
            minimum=10,
            maximum=30,
            key="appearance.base_font_size",
            warnings=warnings,
        ),
        heading_font_size=_clamp_int(
            payload.get("heading_font_size", base.heading_font_size),
            default=base.heading_font_size,
            minimum=11,
            maximum=34,
            key="appearance.heading_font_size",
            warnings=warnings,
        ),
        monospace_font_size=_clamp_int(
            payload.get("monospace_font_size", base.monospace_font_size),
            default=base.monospace_font_size,
            minimum=9,
            maximum=28,
            key="appearance.monospace_font_size",
            warnings=warnings,
        ),
        menu_font_size=_clamp_int(
            payload.get("menu_font_size", base.menu_font_size),
            default=base.menu_font_size,
            minimum=10,
            maximum=30,
            key="appearance.menu_font_size",
            warnings=warnings,
        ),
        tab_font_size=_clamp_int(
            payload.get("tab_font_size", base.tab_font_size),
            default=base.tab_font_size,
            minimum=10,
            maximum=30,
            key="appearance.tab_font_size",
            warnings=warnings,
        ),
        toolbar_font_size=_clamp_int(
            payload.get("toolbar_font_size", base.toolbar_font_size),
            default=base.toolbar_font_size,
            minimum=10,
            maximum=30,
            key="appearance.toolbar_font_size",
            warnings=warnings,
        ),
        status_font_size=_clamp_int(
            payload.get("status_font_size", base.status_font_size),
            default=base.status_font_size,
            minimum=10,
            maximum=30,
            key="appearance.status_font_size",
            warnings=warnings,
        ),
        control_padding_px=_clamp_int(
            payload.get("control_padding_px", base.control_padding_px),
            default=base.control_padding_px,
            minimum=2,
            maximum=20,
            key="appearance.control_padding_px",
            warnings=warnings,
        ),
        panel_spacing_px=_clamp_int(
            payload.get("panel_spacing_px", base.panel_spacing_px),
            default=base.panel_spacing_px,
            minimum=2,
            maximum=24,
            key="appearance.panel_spacing_px",
            warnings=warnings,
        ),
        table_row_padding_px=_clamp_int(
            payload.get("table_row_padding_px", base.table_row_padding_px),
            default=base.table_row_padding_px,
            minimum=2,
            maximum=20,
            key="appearance.table_row_padding_px",
            warnings=warnings,
        ),
        table_min_row_height_px=_clamp_int(
            payload.get("table_min_row_height_px", base.table_min_row_height_px),
            default=base.table_min_row_height_px,
            minimum=18,
            maximum=64,
            key="appearance.table_min_row_height_px",
            warnings=warnings,
        ),
        high_contrast=_as_bool(payload.get("high_contrast", base.high_contrast), base.high_contrast),
    )


def _sanitize_window(payload: dict[str, Any], *, base: DesktopWindowConfig, warnings: list[str]) -> DesktopWindowConfig:
    return DesktopWindowConfig(
        initial_width=_clamp_int(
            payload.get("initial_width", base.initial_width),
            default=base.initial_width,
            minimum=1024,
            maximum=5120,
            key="window.initial_width",
            warnings=warnings,
        ),
        initial_height=_clamp_int(
            payload.get("initial_height", base.initial_height),
            default=base.initial_height,
            minimum=768,
            maximum=4320,
            key="window.initial_height",
            warnings=warnings,
        ),
        minimum_width=_clamp_int(
            payload.get("minimum_width", base.minimum_width),
            default=base.minimum_width,
            minimum=800,
            maximum=3840,
            key="window.minimum_width",
            warnings=warnings,
        ),
        minimum_height=_clamp_int(
            payload.get("minimum_height", base.minimum_height),
            default=base.minimum_height,
            minimum=600,
            maximum=2160,
            key="window.minimum_height",
            warnings=warnings,
        ),
        left_dock_width=_clamp_int(
            payload.get("left_dock_width", base.left_dock_width),
            default=base.left_dock_width,
            minimum=220,
            maximum=700,
            key="window.left_dock_width",
            warnings=warnings,
        ),
        right_dock_width=_clamp_int(
            payload.get("right_dock_width", base.right_dock_width),
            default=base.right_dock_width,
            minimum=240,
            maximum=800,
            key="window.right_dock_width",
            warnings=warnings,
        ),
        workflow_dock_width=_clamp_int(
            payload.get("workflow_dock_width", base.workflow_dock_width),
            default=base.workflow_dock_width,
            minimum=600,
            maximum=2400,
            key="window.workflow_dock_width",
            warnings=warnings,
        ),
        remember_geometry=_as_bool(payload.get("remember_geometry", base.remember_geometry), base.remember_geometry),
        clamp_to_screen=_as_bool(payload.get("clamp_to_screen", base.clamp_to_screen), base.clamp_to_screen),
        start_maximized=_as_bool(payload.get("start_maximized", base.start_maximized), base.start_maximized),
        start_fullscreen=_as_bool(payload.get("start_fullscreen", base.start_fullscreen), base.start_fullscreen),
        show_workflow_dock_on_start=_as_bool(
            payload.get("show_workflow_dock_on_start", base.show_workflow_dock_on_start),
            base.show_workflow_dock_on_start,
        ),
        show_log_dock_on_start=_as_bool(
            payload.get("show_log_dock_on_start", base.show_log_dock_on_start),
            base.show_log_dock_on_start,
        ),
    )


def _sanitize_export_defaults(
    payload: dict[str, Any],
    *,
    base: DesktopExportDefaultsConfig,
    warnings: list[str],
) -> DesktopExportDefaultsConfig:
    profile = _sanitize_report_profile(payload.get("report_profile", base.report_profile), default=base.report_profile, warnings=warnings)
    return DesktopExportDefaultsConfig(
        report_profile=profile,
        write_html_report=_as_bool(payload.get("write_html_report", base.write_html_report), base.write_html_report),
        write_pdf_report=_as_bool(payload.get("write_pdf_report", base.write_pdf_report), base.write_pdf_report),
        write_csv_report=_as_bool(payload.get("write_csv_report", base.write_csv_report), base.write_csv_report),
        write_batch_summary=_as_bool(payload.get("write_batch_summary", base.write_batch_summary), base.write_batch_summary),
        selected_metric_keys=_sanitize_metric_keys(payload.get("selected_metric_keys"), fallback_profile=profile),
        include_sections=_sanitize_sections(payload.get("include_sections"), default=base.include_sections, warnings=warnings),
        sort_metrics=_sanitize_sort_metrics(payload.get("sort_metrics", base.sort_metrics), default=base.sort_metrics, warnings=warnings),
        top_k_key_metrics=_clamp_int(
            payload.get("top_k_key_metrics", base.top_k_key_metrics),
            default=base.top_k_key_metrics,
            minimum=1,
            maximum=200,
            key="export_defaults.top_k_key_metrics",
            warnings=warnings,
        ),
        include_artifact_manifest=_as_bool(
            payload.get("include_artifact_manifest", base.include_artifact_manifest),
            base.include_artifact_manifest,
        ),
        compute_required_metrics=_as_bool(
            payload.get("compute_required_metrics", base.compute_required_metrics),
            base.compute_required_metrics,
        ),
        compute_extended_metrics=_as_bool(
            payload.get("compute_extended_metrics", base.compute_extended_metrics),
            base.compute_extended_metrics,
        ),
        write_orientation_map=_as_bool(
            payload.get("write_orientation_map", base.write_orientation_map),
            base.write_orientation_map,
        ),
        write_distribution_charts=_as_bool(
            payload.get("write_distribution_charts", base.write_distribution_charts),
            base.write_distribution_charts,
        ),
        write_physical_calibration_metrics=_as_bool(
            payload.get("write_physical_calibration_metrics", base.write_physical_calibration_metrics),
            base.write_physical_calibration_metrics,
        ),
    )


def load_desktop_ui_config(path: str | Path | None = None) -> tuple[DesktopUIConfig, list[str], Path | None]:
    """Load desktop UI config with validation, clamping, and fallbacks."""

    base = default_desktop_ui_config()
    warnings: list[str] = []
    source_path: Path | None
    if path is None or not str(path).strip():
        source_path = default_desktop_ui_config_path()
    else:
        source_path = Path(path).expanduser().resolve()
    if source_path is None:
        return base, warnings, None
    if not source_path.exists():
        warnings.append(f"desktop ui config not found: {source_path}; using built-in defaults")
        return base, warnings, source_path
    try:
        payload = load_yaml_config(source_path)
    except Exception as exc:
        warnings.append(f"failed to read desktop ui config {source_path}: {exc}; using defaults")
        return base, warnings, source_path

    schema = str(payload.get("schema_version", SCHEMA_VERSION)).strip()
    if schema != SCHEMA_VERSION:
        warnings.append(f"desktop ui config schema mismatch {schema!r}; expected {SCHEMA_VERSION!r}; applying tolerant parse")

    appearance_payload = payload.get("appearance", {})
    window_payload = payload.get("window", {})
    export_payload = payload.get("export_defaults", {})
    if not isinstance(appearance_payload, dict):
        warnings.append("appearance payload must be a mapping; using defaults")
        appearance_payload = {}
    if not isinstance(window_payload, dict):
        warnings.append("window payload must be a mapping; using defaults")
        window_payload = {}
    if not isinstance(export_payload, dict):
        warnings.append("export_defaults payload must be a mapping; using defaults")
        export_payload = {}

    appearance = _sanitize_appearance(appearance_payload, base=base.appearance, warnings=warnings)
    window = _sanitize_window(window_payload, base=base.window, warnings=warnings)
    export_defaults = _sanitize_export_defaults(export_payload, base=base.export_defaults, warnings=warnings)
    return (
        DesktopUIConfig(schema_version=SCHEMA_VERSION, appearance=appearance, window=window, export_defaults=export_defaults),
        warnings,
        source_path,
    )


def build_qt_stylesheet(config: DesktopUIConfig) -> str:
    """Build Qt stylesheet text from desktop appearance config."""

    a = config.appearance
    if a.high_contrast:
        main_bg = "#090D13"
        panel_bg = "#121821"
        surface_bg = "#0C1219"
        fg = "#F5FAFF"
        muted_fg = "#D0DAE5"
        border = "#58708A"
        border_strong = "#8AA7C4"
        accent = "#FFD166"
        accent_soft = "#33415A"
        hover_bg = "#1A2330"
        selected_bg = "#2E445D"
    else:
        main_bg = "#0D1117"
        panel_bg = "#161B22"
        surface_bg = "#0F141B"
        fg = "#E6EDF5"
        muted_fg = "#AAB6C4"
        border = "#334155"
        border_strong = "#52657A"
        accent = "#7CC3FF"
        accent_soft = "#203C57"
        hover_bg = "#1F2833"
        selected_bg = "#243447"

    button_pad_y = max(2, int(a.control_padding_px))
    button_pad_x = max(4, int(a.control_padding_px + 4))
    radius = int(max(4, a.control_padding_px // 2))
    return (
        "QWidget {"
        f" font-size: {int(a.base_font_size)}px;"
        f" color: {fg};"
        f" background: {main_bg};"
        "}"
        "QMainWindow {"
        f" background: {main_bg};"
        "}"
        "QGroupBox {"
        f" font-size: {int(a.heading_font_size)}px;"
        f" margin-top: {int(max(4, a.panel_spacing_px))}px;"
        f" padding-top: {int(max(4, a.panel_spacing_px // 2))}px;"
        f" border: 1px solid {border};"
        f" border-radius: {radius + 4}px;"
        f" background: {panel_bg};"
        "}"
        "QGroupBox::title {"
        " subcontrol-origin: margin;"
        " subcontrol-position: top left;"
        f" left: {int(max(4, a.control_padding_px))}px;"
        f" padding: 0px {int(max(4, a.control_padding_px))}px;"
        f" color: {accent};"
        "}"
        "QPushButton, QToolButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {"
        f" padding: {button_pad_y}px {button_pad_x}px;"
        " min-height: 28px;"
        f" border: 1px solid {border};"
        f" background: {surface_bg};"
        f" border-radius: {radius}px;"
        f" color: {fg};"
        "}"
        "QPushButton:hover, QToolButton:hover, QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover,"
        " QPlainTextEdit:hover, QTextEdit:hover {"
        f" background: {hover_bg};"
        f" border-color: {accent};"
        "}"
        "QPushButton:pressed, QToolButton:pressed {"
        f" background: {selected_bg};"
        f" border-color: {border_strong};"
        "}"
        "QPushButton:disabled, QToolButton:disabled, QLineEdit:disabled, QComboBox:disabled, "
        "QSpinBox:disabled, QDoubleSpinBox:disabled, QPlainTextEdit:disabled, QTextEdit:disabled {"
        f" color: {muted_fg};"
        f" background: {panel_bg};"
        f" border-color: {border};"
        "}"
        "QPlainTextEdit {"
        " font-family: Menlo, Monaco, monospace;"
        f" font-size: {int(a.monospace_font_size)}px;"
        "}"
        "QComboBox::drop-down {"
        f" border-left: 1px solid {border};"
        " width: 24px;"
        "}"
        "QComboBox QAbstractItemView, QListWidget, QTableWidget {"
        f" background: {panel_bg};"
        f" border: 1px solid {border};"
        f" selection-background-color: {selected_bg};"
        f" selection-color: {fg};"
        "}"
        "QMenuBar, QMenu {"
        f" background: {panel_bg};"
        f" color: {fg};"
        f" font-size: {int(a.menu_font_size)}px;"
        "}"
        "QMenuBar::item:selected, QMenu::item:selected {"
        f" background: {accent_soft};"
        "}"
        "QToolBar, QToolButton {"
        f" font-size: {int(a.toolbar_font_size)}px;"
        "}"
        "QTabBar::tab {"
        f" background: {surface_bg};"
        f" color: {muted_fg};"
        f" border: 1px solid {border};"
        f" border-bottom-color: {border};"
        f" font-size: {int(a.tab_font_size)}px;"
        f" padding: {int(max(4, a.control_padding_px // 2))}px {int(max(6, a.control_padding_px))}px;"
        " margin-right: 2px;"
        " min-width: 110px;"
        " min-height: 30px;"
        f" border-top-left-radius: {radius}px;"
        f" border-top-right-radius: {radius}px;"
        "}"
        "QTabBar::tab:hover {"
        f" background: {hover_bg};"
        f" color: {fg};"
        "}"
        "QTabBar::tab:selected {"
        f" background: {panel_bg};"
        f" color: {fg};"
        f" border-color: {accent};"
        f" border-bottom-color: {panel_bg};"
        f" font-weight: 600;"
        "}"
        "QTabWidget::pane {"
        f" background: {panel_bg};"
        f" border: 1px solid {border};"
        f" top: -1px;"
        f" border-radius: {radius + 2}px;"
        "}"
        "QGroupBox {"
        f" background: {panel_bg};"
        f" border: 1px solid {border};"
        f" border-radius: {radius + 2}px;"
        " margin-top: 14px;"
        f" padding: {int(max(6, a.control_padding_px))}px;"
        "}"
        "QGroupBox::title {"
        f" color: {fg};"
        " subcontrol-origin: margin;"
        " subcontrol-position: top left;"
        f" padding: 0 {int(max(6, a.control_padding_px))}px;"
        " font-weight: 600;"
        "}"
        "QStatusBar {"
        f" font-size: {int(a.status_font_size)}px;"
        f" background: {panel_bg};"
        f" color: {muted_fg};"
        "}"
        "QDockWidget::title {"
        f" background: {panel_bg};"
        f" color: {fg};"
        f" font-size: {int(a.menu_font_size)}px;"
        f" padding: {int(max(4, a.control_padding_px))}px;"
        "}"
        "QHeaderView::section {"
        f" padding: {int(max(4, a.table_row_padding_px // 2))}px;"
        f" background: {surface_bg};"
        f" color: {fg};"
        f" border: 1px solid {border};"
        "}"
        "QTableView::item:selected, QTableWidget::item:selected, QListWidget::item:selected {"
        f" background: {selected_bg};"
        f" color: {fg};"
        "}"
        "QTableView::item, QTableWidget::item {"
        f" padding: {int(a.table_row_padding_px)}px;"
        f" min-height: {int(a.table_min_row_height_px)}px;"
        "}"
        "QScrollArea {"
        f" background: {panel_bg};"
        f" border: 1px solid {border};"
        "}"
        "QFrame#segmentationStatusPanel {"
        f" background: {surface_bg};"
        f" border: 1px solid {border};"
        f" border-radius: {radius + 2}px;"
        "}"
        "QLabel[class='segmentation-status-title'] {"
        f" color: {fg};"
        " font-weight: 600;"
        "}"
        "QProgressBar {"
        f" background: {main_bg};"
        f" color: {fg};"
        f" border: 1px solid {border};"
        f" border-radius: {radius}px;"
        f" padding: {int(max(4, a.control_padding_px // 2))}px;"
        " text-align: center;"
        "}"
        "QProgressBar::chunk {"
        f" background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {accent}, stop:1 {accent_soft});"
        f" border-radius: {radius}px;"
        "}"
        "QScrollBar:vertical, QScrollBar:horizontal {"
        f" background: {main_bg};"
        "}"
    )
