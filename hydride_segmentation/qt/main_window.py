"""Qt desktop GUI for segmentation, correction, and correction export."""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from math import hypot
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.draw import disk, line

from PySide6.QtCore import QEvent, QProcess, QSettings, QTimer, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QScrollArea,
    QToolBar,
    QToolButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QHeaderView,
    QGroupBox,
)

from hydride_segmentation.version import __version__
from hydride_segmentation.legacy_api import DEFAULT_CONVENTIONAL_PARAMS
from src.microseg.app import (
    DesktopUIConfig,
    DesktopRunRecord,
    DesktopResultExportConfig,
    DesktopResultExporter,
    OrchestrationCommandBuilder,
    ProjectSaveRequest,
    ProjectStateStore,
    build_qt_stylesheet,
    compare_run_reports,
    default_desktop_ui_config,
    default_desktop_ui_config_path,
    load_desktop_ui_config,
    REPORT_SECTIONS,
    read_workflow_profile,
    summarize_run_report,
    write_workflow_profile,
)
from src.microseg.app.desktop_ui_config import DesktopAppearanceConfig, DesktopExportDefaultsConfig, DesktopWindowConfig
from src.microseg.app.desktop_workflow import DesktopRunRecord, DesktopWorkflowManager
from src.microseg.corrections import (
    DEFAULT_CLASS_MAP,
    CorrectionExporter,
    CorrectionSession,
    SegmentationClass,
    SegmentationClassMap,
    colorize_index_mask,
    resolve_class_map,
    to_index_mask,
)
from src.microseg.dataops import (
    DatasetPrepareConfig,
    DatasetQualityConfig,
    preview_training_dataset_layout,
    prepare_training_dataset_layout,
    run_dataset_quality_checks,
)
from src.microseg.evaluation import (
    HydrideVisualizationConfig,
    compute_hydride_statistics,
    render_hydride_visualizations,
)
from src.microseg.feedback import FeedbackArtifactWriter, FeedbackCaptureConfig, load_feedback_record
from src.microseg.io import resolve_config
from src.microseg.quality import PreflightConfig, run_preflight
from src.microseg.ui import AnnotationLayerSettings, compose_annotation_view
from src.microseg.utils import (
    SpatialCalibration,
    calibration_from_manual_line,
    metadata_calibration_from_image,
    to_rgb,
)


@dataclass
class _UiState:
    image_path: str | None = None
    current_run: DesktopRunRecord | None = None
    correction_session: CorrectionSession | None = None
    class_map: SegmentationClassMap = field(default_factory=lambda: DEFAULT_CLASS_MAP)
    spatial_calibration: SpatialCalibration | None = None
    calibration_image_path: str | None = None
    current_feedback_record_dir: str = ""
    current_feedback_rating: str = "unrated"


def _discover_sample_images(repo_root: Path) -> list[Path]:
    sample_dirs = [
        repo_root / "data" / "sample_images",
        repo_root / "test_data",
    ]
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
    files: list[Path] = []
    for folder in sample_dirs:
        if not folder.exists():
            continue
        for pattern in patterns:
            files.extend(sorted(folder.glob(pattern)))
    unique: dict[str, Path] = {}
    for path in files:
        unique[str(path.resolve())] = path.resolve()
    return list(unique.values())


def _rgb_to_pixmap(arr: np.ndarray) -> QPixmap:
    rgb = to_rgb(arr).astype(np.uint8)
    h, w, _ = rgb.shape
    image = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image.copy())


def _scale_pixmap(pix: QPixmap, zoom: float) -> QPixmap:
    if abs(zoom - 1.0) < 1e-6:
        return pix
    w = max(1, int(pix.width() * zoom))
    h = max(1, int(pix.height() * zoom))
    return pix.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)


def _mask_to_pixmap(mask: np.ndarray, class_map: SegmentationClassMap) -> QPixmap:
    return _rgb_to_pixmap(colorize_index_mask(to_index_mask(mask), class_map))


def _fmt_metric(value: object) -> str:
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


class _UiLogHandler(logging.Handler):
    """Logging handler that forwards records to a GUI callback."""

    def __init__(self, emit_callback):
        super().__init__()
        self.emit_callback = emit_callback

    def emit(self, record):  # noqa: D401
        msg = self.format(record)
        self.emit_callback(msg)


class ZoomableImageViewport(QWidget):
    """Compact image viewport with in-panel zoom controls and scroll-based panning."""

    zoom_changed = Signal(float)

    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self._title = title
        self._base_pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._fit_mode = True

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        toolbar = QHBoxLayout()
        toolbar.setContentsMargins(0, 0, 0, 0)
        toolbar.setSpacing(4)
        self.title_label = QLabel(title)
        self.title_label.setProperty("class", "viewport-title")
        toolbar.addWidget(self.title_label)
        toolbar.addStretch(1)

        self.btn_fit = QToolButton()
        self.btn_fit.setText("Fit")
        self.btn_fit.setToolTip("Fit image to view")
        self.btn_fit.clicked.connect(self.fit_to_view)
        toolbar.addWidget(self.btn_fit)

        self.btn_reset = QToolButton()
        self.btn_reset.setText("100%")
        self.btn_reset.setToolTip("Reset to 100%")
        self.btn_reset.clicked.connect(self.zoom_reset)
        toolbar.addWidget(self.btn_reset)

        self.btn_zoom_out = QToolButton()
        self.btn_zoom_out.setText("−")
        self.btn_zoom_out.setToolTip("Zoom out")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        toolbar.addWidget(self.btn_zoom_out)

        self.btn_zoom_in = QToolButton()
        self.btn_zoom_in.setText("+")
        self.btn_zoom_in.setToolTip("Zoom in")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        toolbar.addWidget(self.btn_zoom_in)

        self.zoom_label = QLabel("100%")
        toolbar.addWidget(self.zoom_label)
        root.addLayout(toolbar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(False)
        self.scroll_area.setFrameShape(QFrame.StyledPanel)
        self.scroll_area.viewport().installEventFilter(self)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.image_label.setFrameShape(QFrame.NoFrame)
        self.image_label.setSizePolicy(self.sizePolicy())
        self.scroll_area.setWidget(self.image_label)
        root.addWidget(self.scroll_area, stretch=1)

    def eventFilter(self, obj, event):  # noqa: D401, N802
        if obj is self.scroll_area.viewport() and event.type() == QEvent.Wheel:
            if event.modifiers() & Qt.ControlModifier:
                if event.angleDelta().y() > 0:
                    self.zoom_in()
                else:
                    self.zoom_out()
                return True
        return super().eventFilter(obj, event)

    def resizeEvent(self, event):  # noqa: N802
        super().resizeEvent(event)
        if self._fit_mode:
            self._apply_fit_zoom()

    def set_image(self, image: np.ndarray | QPixmap) -> None:
        if isinstance(image, QPixmap):
            pix = image
        else:
            pix = _rgb_to_pixmap(np.asarray(image))
        self._base_pixmap = pix
        self._fit_mode = True
        self.fit_to_view()

    def current_zoom(self) -> float:
        return float(self._zoom)

    def set_zoom(self, value: float) -> None:
        self._fit_mode = False
        self._zoom = max(0.05, min(20.0, float(value)))
        self._refresh()
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.15)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.15)

    def zoom_reset(self) -> None:
        self._fit_mode = False
        self.set_zoom(1.0)

    def fit_to_view(self) -> None:
        self._fit_mode = True
        self._apply_fit_zoom()

    def _apply_fit_zoom(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            return
        viewport = self.scroll_area.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        zoom = min(
            viewport.width() / max(1, self._base_pixmap.width()),
            viewport.height() / max(1, self._base_pixmap.height()),
        )
        self._zoom = max(0.05, min(20.0, float(zoom)))
        self._refresh()
        self.zoom_changed.emit(self._zoom)

    def _refresh(self) -> None:
        if self._base_pixmap is None or self._base_pixmap.isNull():
            self.image_label.clear()
            self.zoom_label.setText("0%")
            return
        pix = _scale_pixmap(self._base_pixmap, self._zoom)
        self.image_label.setPixmap(pix)
        self.image_label.setFixedSize(pix.size())
        self.zoom_label.setText(f"{int(round(self._zoom * 100))}%")


class CalibrationLineCanvas(QLabel):
    """Simple interactive canvas for drawing one calibration line."""

    line_changed = Signal(float)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMouseTracking(True)
        self._pixmap_base: QPixmap | None = None
        self._p0: tuple[int, int] | None = None
        self._p1: tuple[int, int] | None = None

    def set_image(self, image: np.ndarray) -> None:
        self._pixmap_base = _rgb_to_pixmap(image)
        self._p0 = None
        self._p1 = None
        self._refresh()

    def clear_line(self) -> None:
        self._p0 = None
        self._p1 = None
        self._refresh()
        self.line_changed.emit(0.0)

    def line_distance_px(self) -> float:
        if self._p0 is None or self._p1 is None:
            return 0.0
        return float(hypot(self._p1[0] - self._p0[0], self._p1[1] - self._p0[1]))

    def _in_bounds(self, x: int, y: int) -> bool:
        if self._pixmap_base is None:
            return False
        return 0 <= x < self._pixmap_base.width() and 0 <= y < self._pixmap_base.height()

    def mousePressEvent(self, event):  # noqa: N802
        if self._pixmap_base is None:
            return
        p = event.position().toPoint()
        x = int(p.x())
        y = int(p.y())
        if not self._in_bounds(x, y):
            return
        if event.button() == Qt.LeftButton:
            if self._p0 is None or (self._p0 is not None and self._p1 is not None):
                self._p0 = (x, y)
                self._p1 = None
            else:
                self._p1 = (x, y)
            self._refresh()
            self.line_changed.emit(self.line_distance_px())
            return
        if event.button() == Qt.RightButton:
            self.clear_line()

    def _refresh(self) -> None:
        if self._pixmap_base is None:
            return
        pix = self._pixmap_base.copy()
        if self._p0 is not None:
            painter = QPainter(pix)
            painter.setRenderHint(QPainter.Antialiasing, True)
            painter.setPen(QPen(QColor(255, 40, 40), 2))
            painter.drawEllipse(self._p0[0] - 3, self._p0[1] - 3, 6, 6)
            if self._p1 is not None:
                painter.drawLine(self._p0[0], self._p0[1], self._p1[0], self._p1[1])
                painter.drawEllipse(self._p1[0] - 3, self._p1[1] - 3, 6, 6)
            painter.end()
        self.setPixmap(pix)
        self.setFixedSize(pix.size())


class CorrectedMaskCanvas(QLabel):
    """Interactive canvas for correction editing with layered overlays and zoom."""

    zoom_changed = Signal(float)
    cursor_changed = Signal(int, int)
    correction_changed = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.setMouseTracking(True)

        self._base_image: np.ndarray | None = None
        self._predicted_mask: np.ndarray | None = None
        self._session: CorrectionSession | None = None
        self._settings = AnnotationLayerSettings()
        self._class_map = DEFAULT_CLASS_MAP
        self._class_index = 1

        self._tool = "brush"
        self._mode = "add"
        self._radius = 6
        self._painting = False
        self._zoom = 1.0

        self._poly_points: list[tuple[int, int]] = []
        self._lasso_points: list[tuple[int, int]] = []
        self._lasso_active = False

    def set_tool(self, tool: str) -> None:
        self._tool = tool
        self._poly_points = []
        self._lasso_points = []
        self._lasso_active = False
        self._refresh()

    def set_mode(self, mode: str) -> None:
        self._mode = mode

    def set_class_index(self, class_index: int) -> None:
        self._class_index = max(0, int(class_index))

    def set_class_map(self, class_map: SegmentationClassMap) -> None:
        self._class_map = class_map
        self._refresh()

    def set_radius(self, radius: int) -> None:
        self._radius = max(1, int(radius))

    def set_zoom(self, value: float) -> None:
        self._zoom = max(0.1, min(8.0, float(value)))
        self._refresh()
        self.zoom_changed.emit(self._zoom)

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * 1.15)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / 1.15)

    def zoom_reset(self) -> None:
        self.set_zoom(1.0)

    def zoom_value(self) -> float:
        return self._zoom

    def update_layer_settings(self, settings: AnnotationLayerSettings) -> None:
        self._settings = settings
        self._refresh()

    def bind(
        self,
        base_image: np.ndarray,
        predicted_mask: np.ndarray,
        session: CorrectionSession,
        class_map: SegmentationClassMap,
    ) -> None:
        self._base_image = base_image
        self._predicted_mask = to_index_mask(predicted_mask)
        self._session = session
        self._class_map = class_map
        self._painting = False
        self._poly_points = []
        self._lasso_points = []
        self._lasso_active = False
        self._zoom = 1.0
        self._refresh()
        self.zoom_changed.emit(self._zoom)

    def _image_shape(self) -> tuple[int, int] | None:
        if self._session is None:
            return None
        return self._session.current_mask.shape

    def _compose(self) -> np.ndarray:
        if self._base_image is None or self._predicted_mask is None or self._session is None:
            return np.zeros((10, 10, 3), dtype=np.uint8)
        return compose_annotation_view(
            self._base_image,
            self._predicted_mask,
            self._session.current_mask,
            self._settings,
            class_map=self._class_map,
        )

    @staticmethod
    def _draw_preview_points(image: np.ndarray, points: list[tuple[int, int]], color=(0, 255, 255)) -> np.ndarray:
        if len(points) < 1:
            return image
        out = image.copy()
        h, w = out.shape[:2]
        for i in range(1, len(points)):
            x0, y0 = points[i - 1]
            x1, y1 = points[i]
            rr, cc = line(y0, x0, y1, x1)
            rr = np.clip(rr, 0, h - 1)
            cc = np.clip(cc, 0, w - 1)
            out[rr, cc] = color
        for x, y in points:
            rr, cc = disk((y, x), 2, shape=(h, w))
            out[rr, cc] = (255, 255, 255)
        return out

    def _refresh(self) -> None:
        if self._base_image is None or self._session is None:
            return
        composed = self._compose()
        if self._tool == "polygon" and self._poly_points:
            composed = self._draw_preview_points(composed, self._poly_points)
        if self._tool == "lasso" and self._lasso_points:
            composed = self._draw_preview_points(composed, self._lasso_points, color=(0, 200, 255))

        pix = _rgb_to_pixmap(composed)
        pix = _scale_pixmap(pix, self._zoom)
        self.setPixmap(pix)
        self.setFixedSize(pix.size())

    def _to_image_coords(self, p) -> tuple[int, int]:
        x = int(p.x() / self._zoom)
        y = int(p.y() / self._zoom)
        return x, y

    def _in_bounds(self, x: int, y: int) -> bool:
        shape = self._image_shape()
        if shape is None:
            return False
        h, w = shape
        return 0 <= x < w and 0 <= y < h

    def _apply_brush(self, x: int, y: int, *, push_undo: bool, record_action: bool) -> None:
        if self._session is None or not self._in_bounds(x, y):
            return
        self._session.apply_brush(
            x=x,
            y=y,
            radius=self._radius,
            mode=self._mode,
            class_index=self._class_index,
            push_undo=push_undo,
            record_action=record_action,
        )
        self._refresh()
        self.correction_changed.emit()

    def _finish_polygon(self) -> None:
        if self._session is None:
            return
        if len(self._poly_points) >= 3:
            self._session.apply_polygon(self._poly_points, mode=self._mode, class_index=self._class_index)
            self.correction_changed.emit()
        self._poly_points = []
        self._refresh()

    def _finish_lasso(self) -> None:
        if self._session is None:
            return
        if len(self._lasso_points) >= 3:
            self._session.apply_polygon(self._lasso_points, mode=self._mode, class_index=self._class_index)
            self.correction_changed.emit()
        self._lasso_points = []
        self._lasso_active = False
        self._refresh()

    def cancel_preview(self) -> None:
        self._poly_points = []
        self._lasso_points = []
        self._lasso_active = False
        self._refresh()

    def wheelEvent(self, event):  # noqa: N802
        if event.modifiers() & Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        p = event.position().toPoint()
        x, y = self._to_image_coords(p)
        self.cursor_changed.emit(x, y)

        if self._tool == "brush" and event.button() == Qt.LeftButton:
            self._painting = True
            self._apply_brush(x, y, push_undo=True, record_action=True)
            return

        if self._tool == "polygon":
            if event.button() == Qt.LeftButton and self._in_bounds(x, y):
                self._poly_points.append((x, y))
                self._refresh()
            elif event.button() == Qt.RightButton:
                self._finish_polygon()
            return

        if self._tool == "lasso" and event.button() == Qt.LeftButton:
            self._lasso_active = True
            self._lasso_points = [(x, y)] if self._in_bounds(x, y) else []
            self._refresh()
            return

        if self._tool == "feature_select" and event.button() == Qt.LeftButton and self._session is not None:
            changed = False
            if self._mode == "erase":
                changed = self._session.delete_feature(x, y)
            else:
                changed = self._session.relabel_feature(x, y, class_index=self._class_index)
            if changed:
                self._refresh()
                self.correction_changed.emit()

    def mouseMoveEvent(self, event):  # noqa: N802
        p = event.position().toPoint()
        x, y = self._to_image_coords(p)
        self.cursor_changed.emit(x, y)

        if self._tool == "brush" and self._painting:
            self._apply_brush(x, y, push_undo=False, record_action=False)
            return

        if self._tool == "lasso" and self._lasso_active and self._in_bounds(x, y):
            self._lasso_points.append((x, y))
            self._refresh()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if self._tool == "brush" and event.button() == Qt.LeftButton:
            self._painting = False
            return
        if self._tool == "lasso" and event.button() == Qt.LeftButton and self._lasso_active:
            self._finish_lasso()


class AppearanceExportSettingsDialog(QDialog):
    """Dialog for desktop appearance + export default controls."""

    def __init__(self, *, config: DesktopUIConfig, source_path: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Appearance & Export Settings")
        self.resize(720, 560)
        self._config = config
        self._source_path = str(source_path or "")

        layout = QVBoxLayout(self)

        self.path_edit = QLineEdit(self._source_path)
        self.path_edit.setPlaceholderText(str(default_desktop_ui_config_path()))
        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("YAML"))
        path_row.addWidget(self.path_edit, stretch=1)
        self.btn_load = QPushButton("Load")
        self.btn_load.clicked.connect(self.on_load_yaml)
        path_row.addWidget(self.btn_load)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.on_save_yaml)
        path_row.addWidget(self.btn_save)
        self.btn_defaults = QPushButton("Restore Defaults")
        self.btn_defaults.clicked.connect(self.on_restore_defaults)
        path_row.addWidget(self.btn_defaults)
        layout.addLayout(path_row)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        appearance_box = QGroupBox("Appearance")
        appearance_form = QFormLayout(appearance_box)
        self.base_font_spin = QSpinBox()
        self.base_font_spin.setRange(10, 28)
        appearance_form.addRow("Base font", self.base_font_spin)
        self.heading_font_spin = QSpinBox()
        self.heading_font_spin.setRange(11, 34)
        appearance_form.addRow("Heading font", self.heading_font_spin)
        self.mono_font_spin = QSpinBox()
        self.mono_font_spin.setRange(9, 28)
        appearance_form.addRow("Monospace font", self.mono_font_spin)
        self.menu_font_spin = QSpinBox()
        self.menu_font_spin.setRange(10, 30)
        appearance_form.addRow("Menu font", self.menu_font_spin)
        self.tab_font_spin = QSpinBox()
        self.tab_font_spin.setRange(10, 30)
        appearance_form.addRow("Tab font", self.tab_font_spin)
        self.toolbar_font_spin = QSpinBox()
        self.toolbar_font_spin.setRange(10, 30)
        appearance_form.addRow("Toolbar font", self.toolbar_font_spin)
        self.status_font_spin = QSpinBox()
        self.status_font_spin.setRange(10, 30)
        appearance_form.addRow("Status font", self.status_font_spin)
        self.control_pad_spin = QSpinBox()
        self.control_pad_spin.setRange(2, 20)
        appearance_form.addRow("Control padding", self.control_pad_spin)
        self.panel_spacing_spin = QSpinBox()
        self.panel_spacing_spin.setRange(2, 24)
        appearance_form.addRow("Panel spacing", self.panel_spacing_spin)
        self.table_row_padding_spin = QSpinBox()
        self.table_row_padding_spin.setRange(2, 20)
        appearance_form.addRow("Table row padding", self.table_row_padding_spin)
        self.table_row_min_height_spin = QSpinBox()
        self.table_row_min_height_spin.setRange(18, 64)
        appearance_form.addRow("Table min row height", self.table_row_min_height_spin)
        self.high_contrast_check = QCheckBox("High contrast")
        appearance_form.addRow(self.high_contrast_check)
        layout.addWidget(appearance_box)

        window_box = QGroupBox("Window")
        window_form = QFormLayout(window_box)
        self.window_width_spin = QSpinBox()
        self.window_width_spin.setRange(1024, 5120)
        window_form.addRow("Initial width", self.window_width_spin)
        self.window_height_spin = QSpinBox()
        self.window_height_spin.setRange(768, 4320)
        window_form.addRow("Initial height", self.window_height_spin)
        self.window_min_width_spin = QSpinBox()
        self.window_min_width_spin.setRange(800, 3840)
        window_form.addRow("Minimum width", self.window_min_width_spin)
        self.window_min_height_spin = QSpinBox()
        self.window_min_height_spin.setRange(600, 2160)
        window_form.addRow("Minimum height", self.window_min_height_spin)
        self.left_dock_width_spin = QSpinBox()
        self.left_dock_width_spin.setRange(220, 700)
        window_form.addRow("Left dock width", self.left_dock_width_spin)
        self.right_dock_width_spin = QSpinBox()
        self.right_dock_width_spin.setRange(240, 800)
        window_form.addRow("Right dock width", self.right_dock_width_spin)
        self.workflow_dock_width_spin = QSpinBox()
        self.workflow_dock_width_spin.setRange(600, 2400)
        window_form.addRow("Workflow dock width", self.workflow_dock_width_spin)
        self.remember_geometry_check = QCheckBox("Remember geometry")
        window_form.addRow(self.remember_geometry_check)
        self.clamp_to_screen_check = QCheckBox("Clamp to active screen")
        window_form.addRow(self.clamp_to_screen_check)
        self.start_maximized_check = QCheckBox("Start maximized")
        window_form.addRow(self.start_maximized_check)
        self.start_fullscreen_check = QCheckBox("Start fullscreen")
        window_form.addRow(self.start_fullscreen_check)
        layout.addWidget(window_box)

        export_box = QGroupBox("Export Defaults")
        export_form = QFormLayout(export_box)
        self.profile_combo = QComboBox()
        self.profile_combo.addItems(["balanced", "full", "audit"])
        export_form.addRow("Report profile", self.profile_combo)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 200)
        export_form.addRow("Top-K key metrics", self.top_k_spin)
        self.write_html_check = QCheckBox("Write HTML by default")
        self.write_pdf_check = QCheckBox("Write PDF by default")
        self.write_csv_check = QCheckBox("Write CSV by default")
        self.write_batch_check = QCheckBox("Enable batch summary default")
        export_form.addRow(self.write_html_check)
        export_form.addRow(self.write_pdf_check)
        export_form.addRow(self.write_csv_check)
        export_form.addRow(self.write_batch_check)
        layout.addWidget(export_box)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate_from_config(config)

    def _populate_from_config(self, config: DesktopUIConfig) -> None:
        self._config = config
        app = config.appearance
        win = config.window
        exp = config.export_defaults
        self.base_font_spin.setValue(int(app.base_font_size))
        self.heading_font_spin.setValue(int(app.heading_font_size))
        self.mono_font_spin.setValue(int(app.monospace_font_size))
        self.menu_font_spin.setValue(int(app.menu_font_size))
        self.tab_font_spin.setValue(int(app.tab_font_size))
        self.toolbar_font_spin.setValue(int(app.toolbar_font_size))
        self.status_font_spin.setValue(int(app.status_font_size))
        self.control_pad_spin.setValue(int(app.control_padding_px))
        self.panel_spacing_spin.setValue(int(app.panel_spacing_px))
        self.table_row_padding_spin.setValue(int(app.table_row_padding_px))
        self.table_row_min_height_spin.setValue(int(app.table_min_row_height_px))
        self.high_contrast_check.setChecked(bool(app.high_contrast))
        self.window_width_spin.setValue(int(win.initial_width))
        self.window_height_spin.setValue(int(win.initial_height))
        self.window_min_width_spin.setValue(int(win.minimum_width))
        self.window_min_height_spin.setValue(int(win.minimum_height))
        self.left_dock_width_spin.setValue(int(win.left_dock_width))
        self.right_dock_width_spin.setValue(int(win.right_dock_width))
        self.workflow_dock_width_spin.setValue(int(win.workflow_dock_width))
        self.remember_geometry_check.setChecked(bool(win.remember_geometry))
        self.clamp_to_screen_check.setChecked(bool(win.clamp_to_screen))
        self.start_maximized_check.setChecked(bool(win.start_maximized))
        self.start_fullscreen_check.setChecked(bool(win.start_fullscreen))
        self.profile_combo.setCurrentText(str(exp.report_profile))
        self.top_k_spin.setValue(int(exp.top_k_key_metrics))
        self.write_html_check.setChecked(bool(exp.write_html_report))
        self.write_pdf_check.setChecked(bool(exp.write_pdf_report))
        self.write_csv_check.setChecked(bool(exp.write_csv_report))
        self.write_batch_check.setChecked(bool(exp.write_batch_summary))

    def _build_config(self) -> DesktopUIConfig:
        appearance = DesktopAppearanceConfig(
            base_font_size=int(self.base_font_spin.value()),
            heading_font_size=int(self.heading_font_spin.value()),
            monospace_font_size=int(self.mono_font_spin.value()),
            menu_font_size=int(self.menu_font_spin.value()),
            tab_font_size=int(self.tab_font_spin.value()),
            toolbar_font_size=int(self.toolbar_font_spin.value()),
            status_font_size=int(self.status_font_spin.value()),
            control_padding_px=int(self.control_pad_spin.value()),
            panel_spacing_px=int(self.panel_spacing_spin.value()),
            table_row_padding_px=int(self.table_row_padding_spin.value()),
            table_min_row_height_px=int(self.table_row_min_height_spin.value()),
            high_contrast=bool(self.high_contrast_check.isChecked()),
        )
        window = DesktopWindowConfig(
            initial_width=int(self.window_width_spin.value()),
            initial_height=int(self.window_height_spin.value()),
            minimum_width=int(self.window_min_width_spin.value()),
            minimum_height=int(self.window_min_height_spin.value()),
            left_dock_width=int(self.left_dock_width_spin.value()),
            right_dock_width=int(self.right_dock_width_spin.value()),
            workflow_dock_width=int(self.workflow_dock_width_spin.value()),
            remember_geometry=bool(self.remember_geometry_check.isChecked()),
            clamp_to_screen=bool(self.clamp_to_screen_check.isChecked()),
            start_maximized=bool(self.start_maximized_check.isChecked()),
            start_fullscreen=bool(self.start_fullscreen_check.isChecked()),
            show_workflow_dock_on_start=bool(self._config.window.show_workflow_dock_on_start),
            show_log_dock_on_start=bool(self._config.window.show_log_dock_on_start),
        )
        exp_base = self._config.export_defaults
        export_defaults = DesktopExportDefaultsConfig(
            report_profile=str(self.profile_combo.currentText()),
            write_html_report=bool(self.write_html_check.isChecked()),
            write_pdf_report=bool(self.write_pdf_check.isChecked()),
            write_csv_report=bool(self.write_csv_check.isChecked()),
            write_batch_summary=bool(self.write_batch_check.isChecked()),
            selected_metric_keys=tuple(exp_base.selected_metric_keys),
            include_sections=tuple(exp_base.include_sections),
            sort_metrics=str(exp_base.sort_metrics),
            top_k_key_metrics=int(self.top_k_spin.value()),
            include_artifact_manifest=bool(exp_base.include_artifact_manifest),
        )
        return DesktopUIConfig(
            schema_version="microseg.desktop_ui_config.v1",
            appearance=appearance,
            window=window,
            export_defaults=export_defaults,
        )

    def selected_config(self) -> DesktopUIConfig:
        return self._build_config()

    def selected_path(self) -> str:
        return str(self.path_edit.text().strip())

    def on_load_yaml(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            path = str(default_desktop_ui_config_path())
            self.path_edit.setText(path)
        cfg, warnings, _ = load_desktop_ui_config(path)
        self._populate_from_config(cfg)
        if warnings:
            self.status_label.setText("Loaded with warnings: " + "; ".join(warnings[:4]))
        else:
            self.status_label.setText("Loaded configuration successfully.")

    def on_save_yaml(self) -> None:
        out_path = self.path_edit.text().strip()
        if not out_path:
            out_path = str(default_desktop_ui_config_path())
            self.path_edit.setText(out_path)
        path = Path(out_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        cfg = self._build_config()
        try:
            import yaml

            path.write_text(yaml.safe_dump(cfg.as_dict(), sort_keys=False), encoding="utf-8")
            self.status_label.setText(f"Saved: {path}")
        except Exception as exc:
            self.status_label.setText(f"Save failed: {exc}")

    def on_restore_defaults(self) -> None:
        cfg = default_desktop_ui_config()
        self._populate_from_config(cfg)
        self.status_label.setText("Restored built-in defaults.")


class QtSegmentationMainWindow(QMainWindow):
    """Qt main window for phase-3 correction workflow."""

    def __init__(self, *, ui_config_path: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle(f"MicroSeg Desktop v{__version__}")
        self.resize(1700, 1050)

        self.workflow = DesktopWorkflowManager(max_history=400)
        self.exporter = CorrectionExporter()
        self.result_exporter = DesktopResultExporter()
        self.project_store = ProjectStateStore()
        self.orchestrator = OrchestrationCommandBuilder.discover(start=Path(__file__))
        self._job_process: QProcess | None = None
        self._job_name: str = ""
        self._sample_images: list[Path] = _discover_sample_images(self.orchestrator.repo_root)
        self._dataset_preview_rows: list[dict[str, object]] = []
        self._dataset_preview_split_counts: dict[str, int] = {}
        self._last_dataset_qa_ok: bool | None = None
        self._last_dataset_qa_dir: str = ""
        self._review_summary_a = None
        self._review_summary_b = None
        self._results_dirty = False
        self._latest_results_payload: dict[str, object] = {}
        self.state = _UiState()
        self._ui_config_path = str(ui_config_path or "").strip()
        self._ui_config_source: str = ""
        self._ui_config_warnings: list[str] = []
        self._ui_config: DesktopUIConfig = default_desktop_ui_config()
        self._suppress_feedback_note_events = False

        self.logger = logging.getLogger("MicroSegQtGUI")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        if not self.logger.handlers:
            stream = logging.StreamHandler()
            stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(stream)
        log_dir = self.orchestrator.repo_root / "outputs" / "logs" / "desktop"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"microseg_desktop_{datetime.now().strftime('%Y%m%d')}.log"
        if not any(
            isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == log_path
            for h in self.logger.handlers
        ):
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(file_handler)
        self.logger.info("Desktop log file: %s", log_path)
        self._load_ui_config(self._ui_config_path or None)

        self.feedback_writer = FeedbackArtifactWriter(
            FeedbackCaptureConfig(
                feedback_root=str(self.orchestrator.repo_root / "outputs" / "feedback_records"),
                deployment_id=str(os.environ.get("MICROSEG_DEPLOYMENT_ID", "desktop_local")),
                operator_id=str(os.environ.get("MICROSEG_OPERATOR_ID", "unknown_operator")),
                source="desktop_gui",
            )
        )

        self._results_refresh_timer = QTimer(self)
        self._results_refresh_timer.setSingleShot(True)
        self._results_refresh_timer.timeout.connect(self._update_results_dashboard)
        self._feedback_comment_timer = QTimer(self)
        self._feedback_comment_timer.setSingleShot(True)
        self._feedback_comment_timer.timeout.connect(self._flush_feedback_comment)
        self._feedback_correction_timer = QTimer(self)
        self._feedback_correction_timer.setSingleShot(True)
        self._feedback_correction_timer.timeout.connect(self._flush_feedback_correction)

        try:
            resolved_class_map, class_map_source = resolve_class_map()
            self.state.class_map = resolved_class_map
            self.logger.info("Class map source: %s", class_map_source)
        except Exception as exc:
            self.logger.warning("Failed to load configured class map; falling back to builtin defaults: %s", exc)
        self._model_specs = {spec["display_name"]: spec for spec in self.workflow.model_specs()}

        self._sync_scroll_guard = False

        self._build_ui()
        self._configure_menu()
        self._bind_shortcuts()
        self._apply_style()
        self._apply_application_fonts()
        self._apply_window_geometry()

        self._ui_handler = _UiLogHandler(self._log)
        self._ui_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(self._ui_handler)
        self._refresh_sample_picker()
        self._update_calibration_status_label()

    def _load_ui_config(self, ui_config_path: str | None) -> None:
        cfg, warnings, source_path = load_desktop_ui_config(ui_config_path)
        self._ui_config = cfg
        self._ui_config_source = str(source_path) if source_path is not None else ""
        self._ui_config_warnings = list(warnings)
        self.logger.info(
            "Desktop UI config source=%s base_font=%d heading_font=%d mono_font=%d high_contrast=%s",
            self._ui_config_source or "<builtin-defaults>",
            int(cfg.appearance.base_font_size),
            int(cfg.appearance.heading_font_size),
            int(cfg.appearance.monospace_font_size),
            bool(cfg.appearance.high_contrast),
        )
        for warning in warnings:
            self.logger.warning("Desktop UI config warning: %s", warning)

    def _apply_style(self) -> None:
        self.setStyleSheet(build_qt_stylesheet(self._ui_config))

    def _apply_application_fonts(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        base_font = QFont(app.font())
        base_font.setPointSize(int(self._ui_config.appearance.base_font_size))
        app.setFont(base_font)

    def _apply_window_geometry(self) -> None:
        window_cfg = self._ui_config.window
        self.resize(int(window_cfg.initial_width), int(window_cfg.initial_height))

        if window_cfg.start_fullscreen:
            self.showFullScreen()
            return
        if window_cfg.start_maximized:
            self.showMaximized()
            return

        settings = QSettings()
        restored = False
        if window_cfg.remember_geometry:
            geometry = settings.value("desktop/window_geometry")
            state = settings.value("desktop/window_state")
            if geometry is not None:
                try:
                    restored = bool(self.restoreGeometry(geometry))
                except Exception:
                    restored = False
            if state is not None:
                try:
                    self.restoreState(state)
                except Exception:
                    pass

        screen = QApplication.primaryScreen() or self.screen()
        available = screen.availableGeometry() if screen is not None else None
        max_width = max(200, (available.width() - 40) if available is not None else int(window_cfg.initial_width))
        max_height = max(200, (available.height() - 40) if available is not None else int(window_cfg.initial_height))
        min_width = int(window_cfg.minimum_width)
        min_height = int(window_cfg.minimum_height)
        if available is not None and window_cfg.clamp_to_screen:
            min_width = min(min_width, max_width)
            min_height = min(min_height, max_height)
        self.setMinimumSize(min_width, min_height)

        if not restored and window_cfg.clamp_to_screen:
            if available is not None:
                width = min(max(int(window_cfg.initial_width), int(window_cfg.minimum_width)), max_width)
                height = min(max(int(window_cfg.initial_height), int(window_cfg.minimum_height)), max_height)
                self.resize(width, height)
                frame = self.frameGeometry()
                frame.moveCenter(available.center())
                self.move(frame.topLeft())

    def closeEvent(self, event):  # noqa: N802
        window_cfg = self._ui_config.window
        if window_cfg.remember_geometry:
            settings = QSettings()
            settings.setValue("desktop/window_geometry", self.saveGeometry())
            settings.setValue("desktop/window_state", self.saveState())
        super().closeEvent(event)

    def _configure_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        act_open = QAction("Open Image", self)
        act_open.triggered.connect(self.on_load_image)
        file_menu.addAction(act_open)

        self._sample_menu = file_menu.addMenu("Open Sample")
        self._populate_sample_menu()

        act_batch = QAction("Open Batch", self)
        act_batch.triggered.connect(self.on_run_batch)
        file_menu.addAction(act_batch)

        file_menu.addSeparator()
        act_scan_cal = QAction("Scan Metadata Scale", self)
        act_scan_cal.triggered.connect(self.on_scan_metadata_calibration)
        file_menu.addAction(act_scan_cal)
        act_manual_cal = QAction("Calibrate Scale...", self)
        act_manual_cal.triggered.connect(self.on_calibrate_scale)
        file_menu.addAction(act_manual_cal)
        act_clear_cal = QAction("Clear Scale", self)
        act_clear_cal.triggered.connect(self.on_clear_calibration)
        file_menu.addAction(act_clear_cal)

        file_menu.addSeparator()
        act_export_results = QAction("Export Results Package", self)
        act_export_results.triggered.connect(self.on_export_results_package)
        file_menu.addAction(act_export_results)
        act_export_batch_results = QAction("Export Batch Summary", self)
        act_export_batch_results.triggered.connect(self.on_export_batch_results)
        file_menu.addAction(act_export_batch_results)

        act_export = QAction("Export Corrected Sample", self)
        act_export.triggered.connect(self.on_export_correction)
        file_menu.addAction(act_export)

        act_save_project = QAction("Save Project Session", self)
        act_save_project.triggered.connect(self.on_save_project)
        file_menu.addAction(act_save_project)

        act_load_project = QAction("Load Project Session", self)
        act_load_project.triggered.connect(self.on_load_project)
        file_menu.addAction(act_load_project)

        act_export_log = QAction("Open Log Folder", self)
        act_export_log.triggered.connect(self.on_open_log_folder)
        file_menu.addAction(act_export_log)

        file_menu.addSeparator()
        act_exit = QAction("Exit", self)
        act_exit.triggered.connect(self.close)
        file_menu.addAction(act_exit)

        edit_menu = menu.addMenu("Edit")
        edit_menu.addAction("Undo", self.on_undo)
        edit_menu.addAction("Redo", self.on_redo)
        edit_menu.addAction("Reset Corrections", self.on_reset_corrections)

        view_menu = menu.addMenu("View")
        view_menu.addAction("Zoom In", self.corrected_canvas.zoom_in)
        view_menu.addAction("Zoom Out", self.corrected_canvas.zoom_out)
        view_menu.addAction("Zoom Reset", self.corrected_canvas.zoom_reset)
        view_menu.addAction("Results Dashboard", self.on_open_results_dashboard)
        view_menu.addAction("Workflow Hub", self.on_open_workflow_hub)

        settings_menu = menu.addMenu("Settings")
        settings_menu.addAction("Appearance & Export Settings", self.on_open_appearance_settings)

        help_menu = menu.addMenu("Help")
        help_menu.addAction("Shortcuts", self.on_show_shortcuts)
        help_menu.addAction("Guide", self.on_show_guide)
        help_menu.addAction("Model Details", self.on_show_model_details)
        help_menu.addAction("About", self.on_show_about)

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        layout = QVBoxLayout(root)

        controls = QHBoxLayout()
        layout.addLayout(controls)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Image path")
        controls.addWidget(self.path_edit, stretch=5)

        self.btn_load = QPushButton("Load Image")
        self.btn_load.clicked.connect(self.on_load_image)
        controls.addWidget(self.btn_load)

        self.sample_combo = QComboBox()
        self.sample_combo.setMinimumWidth(260)
        controls.addWidget(self.sample_combo)

        self.btn_load_sample = QPushButton("Load Sample")
        self.btn_load_sample.clicked.connect(self.on_load_sample)
        controls.addWidget(self.btn_load_sample)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.workflow.model_options())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        controls.addWidget(self.model_combo)

        self.btn_model_details = QPushButton("Model Details")
        self.btn_model_details.clicked.connect(self.on_show_model_details)
        controls.addWidget(self.btn_model_details)

        self.btn_run = QPushButton("Run Segmentation")
        self.btn_run.clicked.connect(self.on_run_segmentation)
        controls.addWidget(self.btn_run)

        self.btn_batch = QPushButton("Run Batch")
        self.btn_batch.clicked.connect(self.on_run_batch)
        controls.addWidget(self.btn_batch)

        self.btn_workspace_gear = QToolButton()
        self.btn_workspace_gear.setText("⚙")
        self.btn_workspace_gear.setToolTip("Show or hide advanced panels")
        self.btn_workspace_gear.setPopupMode(QToolButton.InstantPopup)
        controls.addWidget(self.btn_workspace_gear)

        self.model_desc = QLabel("")
        self.model_desc.setWordWrap(True)
        layout.addWidget(self.model_desc)
        self._on_model_changed(self.model_combo.currentText())

        config_row = QHBoxLayout()
        layout.addLayout(config_row)
        self.config_path_edit = QLineEdit()
        self.config_path_edit.setPlaceholderText("Optional YAML config path")
        config_row.addWidget(self.config_path_edit, stretch=4)
        self.btn_config_browse = QPushButton("Config...")
        self.btn_config_browse.clicked.connect(self.on_pick_config)
        config_row.addWidget(self.btn_config_browse)
        self.config_overrides_edit = QLineEdit()
        self.config_overrides_edit.setPlaceholderText("Overrides: key=value,key2=value2")
        config_row.addWidget(self.config_overrides_edit, stretch=3)

        calibration_row = QHBoxLayout()
        layout.addLayout(calibration_row)
        self.calibration_status_label = QLabel("Scale: pixels (no calibration)")
        self.calibration_status_label.setWordWrap(True)
        calibration_row.addWidget(self.calibration_status_label, stretch=4)
        self.btn_scan_metadata_calibration = QPushButton("Scan Metadata Scale")
        self.btn_scan_metadata_calibration.clicked.connect(self.on_scan_metadata_calibration)
        calibration_row.addWidget(self.btn_scan_metadata_calibration)
        self.btn_calibrate_line = QPushButton("Calibrate Scale...")
        self.btn_calibrate_line.clicked.connect(self.on_calibrate_scale)
        calibration_row.addWidget(self.btn_calibrate_line)
        self.btn_clear_calibration = QPushButton("Clear Scale")
        self.btn_clear_calibration.clicked.connect(self.on_clear_calibration)
        calibration_row.addWidget(self.btn_clear_calibration)

        self.conventional_row_widget = QWidget()
        conventional_row = QHBoxLayout(self.conventional_row_widget)
        conventional_row.setContentsMargins(0, 0, 0, 0)
        conventional_row.addWidget(QLabel("Conventional Controls"))

        conventional_row.addWidget(QLabel("CLAHE clip"))
        self.conv_clip_spin = QDoubleSpinBox()
        self.conv_clip_spin.setDecimals(2)
        self.conv_clip_spin.setRange(0.1, 20.0)
        self.conv_clip_spin.setSingleStep(0.1)
        self.conv_clip_spin.setValue(float(DEFAULT_CONVENTIONAL_PARAMS["clahe"]["clip_limit"]))
        conventional_row.addWidget(self.conv_clip_spin)

        conventional_row.addWidget(QLabel("Tile"))
        self.conv_tile_x = QSpinBox()
        self.conv_tile_x.setRange(1, 64)
        self.conv_tile_x.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["clahe"]["tile_grid_size"][0]))
        conventional_row.addWidget(self.conv_tile_x)
        self.conv_tile_y = QSpinBox()
        self.conv_tile_y.setRange(1, 64)
        self.conv_tile_y.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["clahe"]["tile_grid_size"][1]))
        conventional_row.addWidget(self.conv_tile_y)

        conventional_row.addWidget(QLabel("Adaptive block"))
        self.conv_block_spin = QSpinBox()
        self.conv_block_spin.setRange(3, 401)
        self.conv_block_spin.setSingleStep(2)
        self.conv_block_spin.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["adaptive"]["block_size"]))
        conventional_row.addWidget(self.conv_block_spin)

        conventional_row.addWidget(QLabel("Adaptive C"))
        self.conv_c_spin = QSpinBox()
        self.conv_c_spin.setRange(-200, 200)
        self.conv_c_spin.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["adaptive"]["C"]))
        conventional_row.addWidget(self.conv_c_spin)

        conventional_row.addWidget(QLabel("Kernel"))
        self.conv_kernel_x = QSpinBox()
        self.conv_kernel_x.setRange(1, 64)
        self.conv_kernel_x.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["morph"]["kernel_size"][0]))
        conventional_row.addWidget(self.conv_kernel_x)
        self.conv_kernel_y = QSpinBox()
        self.conv_kernel_y.setRange(1, 64)
        self.conv_kernel_y.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["morph"]["kernel_size"][1]))
        conventional_row.addWidget(self.conv_kernel_y)

        conventional_row.addWidget(QLabel("Morph iters"))
        self.conv_iterations_spin = QSpinBox()
        self.conv_iterations_spin.setRange(0, 20)
        self.conv_iterations_spin.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["morph"]["iterations"]))
        conventional_row.addWidget(self.conv_iterations_spin)

        conventional_row.addWidget(QLabel("Area >="))
        self.conv_area_spin = QSpinBox()
        self.conv_area_spin.setRange(0, 100000)
        self.conv_area_spin.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["area_threshold"]))
        conventional_row.addWidget(self.conv_area_spin)

        self.conv_crop_check = QCheckBox("Crop")
        self.conv_crop_check.setChecked(bool(DEFAULT_CONVENTIONAL_PARAMS["crop"]))
        conventional_row.addWidget(self.conv_crop_check)
        self.conv_crop_percent = QSpinBox()
        self.conv_crop_percent.setRange(0, 80)
        self.conv_crop_percent.setValue(int(DEFAULT_CONVENTIONAL_PARAMS["crop_percent"]))
        conventional_row.addWidget(self.conv_crop_percent)
        conventional_row.addStretch(1)
        layout.addWidget(self.conventional_row_widget)

        self.corrected_canvas = CorrectedMaskCanvas()
        self.corrected_canvas.zoom_changed.connect(self._on_zoom_changed)
        self.corrected_canvas.cursor_changed.connect(self._on_cursor_changed)
        self.corrected_canvas.correction_changed.connect(self._on_correction_changed)

        tool_row = QHBoxLayout()
        layout.addLayout(tool_row)

        tool_row.addWidget(QLabel("Tool"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["brush", "polygon", "lasso", "feature_select"])
        self.tool_combo.currentTextChanged.connect(self._on_tool_changed)
        tool_row.addWidget(self.tool_combo)

        tool_row.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["add", "erase"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tool_row.addWidget(self.mode_combo)

        tool_row.addWidget(QLabel("Class"))
        self.class_combo = QComboBox()
        self.class_combo.currentTextChanged.connect(self._on_class_changed)
        tool_row.addWidget(self.class_combo)

        self.btn_classes = QPushButton("Edit Classes")
        self.btn_classes.clicked.connect(self.on_edit_classes)
        tool_row.addWidget(self.btn_classes)

        tool_row.addWidget(QLabel("Brush"))
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 150)
        self.radius_spin.setValue(6)
        self.radius_spin.valueChanged.connect(self.corrected_canvas.set_radius)
        tool_row.addWidget(self.radius_spin)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self.on_undo)
        tool_row.addWidget(self.btn_undo)

        self.btn_redo = QPushButton("Redo")
        self.btn_redo.clicked.connect(self.on_redo)
        tool_row.addWidget(self.btn_redo)

        self.btn_reset_corr = QPushButton("Reset Corrections")
        self.btn_reset_corr.clicked.connect(self.on_reset_corrections)
        tool_row.addWidget(self.btn_reset_corr)

        tool_row.addWidget(QLabel("Zoom"))
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.clicked.connect(self.corrected_canvas.zoom_out)
        tool_row.addWidget(self.btn_zoom_out)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.clicked.connect(self.corrected_canvas.zoom_in)
        tool_row.addWidget(self.btn_zoom_in)

        self.btn_zoom_reset = QPushButton("100%")
        self.btn_zoom_reset.clicked.connect(self.corrected_canvas.zoom_reset)
        tool_row.addWidget(self.btn_zoom_reset)

        layer_row = QHBoxLayout()
        layout.addLayout(layer_row)

        self.chk_pred = QCheckBox("Predicted")
        self.chk_pred.setChecked(True)
        self.chk_pred.stateChanged.connect(self._update_layer_settings)
        layer_row.addWidget(self.chk_pred)

        self.slider_pred = QSlider(Qt.Horizontal)
        self.slider_pred.setRange(0, 100)
        self.slider_pred.setValue(35)
        self.slider_pred.valueChanged.connect(self._update_layer_settings)
        layer_row.addWidget(QLabel("Pred α"))
        layer_row.addWidget(self.slider_pred)

        self.chk_corr = QCheckBox("Corrected")
        self.chk_corr.setChecked(True)
        self.chk_corr.stateChanged.connect(self._update_layer_settings)
        layer_row.addWidget(self.chk_corr)

        self.slider_corr = QSlider(Qt.Horizontal)
        self.slider_corr.setRange(0, 100)
        self.slider_corr.setValue(45)
        self.slider_corr.valueChanged.connect(self._update_layer_settings)
        layer_row.addWidget(QLabel("Corr α"))
        layer_row.addWidget(self.slider_corr)

        self.chk_diff = QCheckBox("Diff")
        self.chk_diff.setChecked(True)
        self.chk_diff.stateChanged.connect(self._update_layer_settings)
        layer_row.addWidget(self.chk_diff)

        self.slider_diff = QSlider(Qt.Horizontal)
        self.slider_diff.setRange(0, 100)
        self.slider_diff.setValue(70)
        self.slider_diff.valueChanged.connect(self._update_layer_settings)
        layer_row.addWidget(QLabel("Diff α"))
        layer_row.addWidget(self.slider_diff)

        self.annotator_edit = QLineEdit()
        self.annotator_edit.setPlaceholderText("Annotator")
        layer_row.addWidget(self.annotator_edit, stretch=1)

        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Correction notes")
        self.notes_edit.textChanged.connect(self._on_feedback_comment_changed)
        layer_row.addWidget(self.notes_edit, stretch=2)

        self.btn_thumb_up = QPushButton("👍")
        self.btn_thumb_up.setToolTip("Rate segmentation as acceptable")
        self.btn_thumb_up.clicked.connect(lambda: self._on_feedback_rating_clicked("thumbs_up"))
        layer_row.addWidget(self.btn_thumb_up)

        self.btn_thumb_down = QPushButton("👎")
        self.btn_thumb_down.setToolTip("Rate segmentation as poor")
        self.btn_thumb_down.clicked.connect(lambda: self._on_feedback_rating_clicked("thumbs_down"))
        layer_row.addWidget(self.btn_thumb_down)

        self.feedback_rating_label = QLabel("Feedback: unrated")
        layer_row.addWidget(self.feedback_rating_label)

        self.chk_fmt_indexed = QCheckBox("indexed")
        self.chk_fmt_indexed.setChecked(True)
        layer_row.addWidget(self.chk_fmt_indexed)

        self.chk_fmt_color = QCheckBox("color")
        self.chk_fmt_color.setChecked(True)
        layer_row.addWidget(self.chk_fmt_color)

        self.chk_fmt_npy = QCheckBox("npy")
        self.chk_fmt_npy.setChecked(False)
        layer_row.addWidget(self.chk_fmt_npy)

        self.chk_report_html = QCheckBox("report.html")
        self.chk_report_html.setChecked(bool(self._ui_config.export_defaults.write_html_report))
        layer_row.addWidget(self.chk_report_html)

        self.chk_report_pdf = QCheckBox("report.pdf")
        self.chk_report_pdf.setChecked(bool(self._ui_config.export_defaults.write_pdf_report))
        layer_row.addWidget(self.chk_report_pdf)

        self.chk_report_csv = QCheckBox("report.csv")
        self.chk_report_csv.setChecked(bool(self._ui_config.export_defaults.write_csv_report))
        layer_row.addWidget(self.chk_report_csv)

        layer_row.addWidget(QLabel("Profile"))
        self.report_profile_combo = QComboBox()
        self.report_profile_combo.addItems(["balanced", "full", "audit"])
        self.report_profile_combo.setCurrentText(str(self._ui_config.export_defaults.report_profile))
        self.report_profile_combo.currentTextChanged.connect(lambda *_: self.on_reset_profile_report_metrics())
        layer_row.addWidget(self.report_profile_combo)

        layer_row.addWidget(QLabel("Top-K"))
        self.report_top_k_spin = QSpinBox()
        self.report_top_k_spin.setRange(1, 200)
        self.report_top_k_spin.setValue(int(self._ui_config.export_defaults.top_k_key_metrics))
        layer_row.addWidget(self.report_top_k_spin)

        self.chk_artifact_manifest = QCheckBox("artifact manifest")
        self.chk_artifact_manifest.setChecked(bool(self._ui_config.export_defaults.include_artifact_manifest))
        layer_row.addWidget(self.chk_artifact_manifest)

        self.btn_export_batch = QPushButton("Export Batch Summary")
        self.btn_export_batch.clicked.connect(self.on_export_batch_results)
        layer_row.addWidget(self.btn_export_batch)

        self.btn_export = QPushButton("Export Corrected Sample")
        self.btn_export.clicked.connect(self.on_export_correction)
        layer_row.addWidget(self.btn_export)

        self.btn_export_results = QPushButton("Export Results Package")
        self.btn_export_results.clicked.connect(self.on_export_results_package)
        layer_row.addWidget(self.btn_export_results)

        self.btn_save_project = QPushButton("Save Session")
        self.btn_save_project.clicked.connect(self.on_save_project)
        layer_row.addWidget(self.btn_save_project)

        self.btn_load_project = QPushButton("Load Session")
        self.btn_load_project.clicked.connect(self.on_load_project)
        layer_row.addWidget(self.btn_load_project)

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        self.history_list = QListWidget()
        self.history_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.history_list.currentRowChanged.connect(self.on_history_selected)
        body.addWidget(self.history_list, stretch=1)

        self.report_advanced_group = QGroupBox("Advanced Report Selection")
        self.report_advanced_group.setCheckable(True)
        self.report_advanced_group.setChecked(False)
        advanced_layout = QVBoxLayout(self.report_advanced_group)
        sections_row = QHBoxLayout()
        advanced_layout.addLayout(sections_row)
        self.report_section_checks: dict[str, QCheckBox] = {}
        for section in REPORT_SECTIONS:
            chk = QCheckBox(section)
            chk.setChecked(section in set(self._ui_config.export_defaults.include_sections))
            self.report_section_checks[section] = chk
            sections_row.addWidget(chk)
        sections_row.addStretch(1)

        metrics_row = QHBoxLayout()
        advanced_layout.addLayout(metrics_row)
        self.btn_metrics_select_all = QPushButton("Select All Metrics")
        self.btn_metrics_select_all.clicked.connect(self.on_select_all_report_metrics)
        metrics_row.addWidget(self.btn_metrics_select_all)
        self.btn_metrics_clear = QPushButton("Clear Metrics")
        self.btn_metrics_clear.clicked.connect(self.on_clear_report_metrics)
        metrics_row.addWidget(self.btn_metrics_clear)
        self.btn_metrics_reset_profile = QPushButton("Reset Profile Metrics")
        self.btn_metrics_reset_profile.clicked.connect(self.on_reset_profile_report_metrics)
        metrics_row.addWidget(self.btn_metrics_reset_profile)
        metrics_row.addStretch(1)

        self.report_metric_list = QListWidget()
        self.report_metric_list.setMinimumHeight(130)
        advanced_layout.addWidget(self.report_metric_list)
        layout.addWidget(self.report_advanced_group)

        self.tabs = QTabWidget()
        body.addWidget(self.tabs, stretch=6)

        self.input_view = ZoomableImageViewport("Input")
        self.mask_view = ZoomableImageViewport("Predicted Mask")
        self.overlay_view = ZoomableImageViewport("Overlay")

        self.tabs.addTab(self.input_view, "Input")
        self.tabs.addTab(self.mask_view, "Predicted Mask")
        self.tabs.addTab(self.overlay_view, "Overlay")

        self.split_widget = QWidget()
        split_layout = QHBoxLayout(self.split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Horizontal)
        split_layout.addWidget(self.splitter)

        self.raw_corr_view = ZoomableImageViewport("Raw Input")
        self.raw_corr_scroll = self.raw_corr_view.scroll_area
        self.corrected_scroll = self._in_scroll(self.corrected_canvas)

        self.splitter.addWidget(self.raw_corr_scroll)
        self.splitter.addWidget(self.corrected_scroll)
        self.splitter.setSizes([700, 900])

        self.tabs.addTab(self.split_widget, "Correction Split View")

        self.results_widget = QWidget()
        results_root = QVBoxLayout(self.results_widget)

        results_controls = QHBoxLayout()
        results_root.addLayout(results_controls)
        results_controls.addWidget(QLabel("Orientation bins"))
        self.results_orientation_bins = QSpinBox()
        self.results_orientation_bins.setRange(6, 180)
        self.results_orientation_bins.setValue(18)
        results_controls.addWidget(self.results_orientation_bins)

        results_controls.addWidget(QLabel("Size bins"))
        self.results_size_bins = QSpinBox()
        self.results_size_bins.setRange(4, 180)
        self.results_size_bins.setValue(20)
        results_controls.addWidget(self.results_size_bins)

        results_controls.addWidget(QLabel("Min feature px"))
        self.results_min_feature = QSpinBox()
        self.results_min_feature.setRange(1, 100000)
        self.results_min_feature.setValue(1)
        results_controls.addWidget(self.results_min_feature)

        results_controls.addWidget(QLabel("Size scale"))
        self.results_size_scale = QComboBox()
        self.results_size_scale.addItems(["linear", "log"])
        self.results_size_scale.setCurrentText("linear")
        results_controls.addWidget(self.results_size_scale)

        results_controls.addWidget(QLabel("Orientation map"))
        self.results_cmap = QComboBox()
        self.results_cmap.addItems(["coolwarm", "viridis", "plasma", "turbo", "magma", "cividis"])
        self.results_cmap.setCurrentText("coolwarm")
        results_controls.addWidget(self.results_cmap)

        self.btn_results_refresh = QPushButton("Recompute Stats")
        self.btn_results_refresh.clicked.connect(self._update_results_dashboard)
        results_controls.addWidget(self.btn_results_refresh)

        self.btn_results_export = QPushButton("Export Results")
        self.btn_results_export.clicked.connect(self.on_export_results_package)
        results_controls.addWidget(self.btn_results_export)
        results_controls.addStretch(1)

        self.results_summary_label = QLabel("Results: run segmentation to populate dashboard")
        self.results_summary_label.setWordWrap(True)
        results_root.addWidget(self.results_summary_label)

        self.results_table = QTableWidget(0, 3)
        self.results_table.setHorizontalHeaderLabels(["Metric", "Predicted", "Corrected"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.results_table.setAlternatingRowColors(True)
        results_root.addWidget(self.results_table, stretch=1)

        self.results_plot_tabs = QTabWidget()
        results_root.addWidget(self.results_plot_tabs, stretch=2)

        self.results_pred_widget = QWidget()
        pred_layout = QHBoxLayout(self.results_pred_widget)
        self.results_pred_orientation_view = ZoomableImageViewport("Orientation map")
        self.results_pred_size_view = ZoomableImageViewport("Size distribution")
        self.results_pred_angle_view = ZoomableImageViewport("Orientation distribution")
        pred_layout.addWidget(self.results_pred_orientation_view, stretch=1)
        pred_layout.addWidget(self.results_pred_size_view, stretch=1)
        pred_layout.addWidget(self.results_pred_angle_view, stretch=1)
        self.results_plot_tabs.addTab(self.results_pred_widget, "Predicted")

        self.results_corr_widget = QWidget()
        corr_layout = QHBoxLayout(self.results_corr_widget)
        self.results_corr_orientation_view = ZoomableImageViewport("Orientation map")
        self.results_corr_size_view = ZoomableImageViewport("Size distribution")
        self.results_corr_angle_view = ZoomableImageViewport("Orientation distribution")
        corr_layout.addWidget(self.results_corr_orientation_view, stretch=1)
        corr_layout.addWidget(self.results_corr_size_view, stretch=1)
        corr_layout.addWidget(self.results_corr_angle_view, stretch=1)
        self.results_plot_tabs.addTab(self.results_corr_widget, "Corrected")

        self.tabs.addTab(self.results_widget, "Results Dashboard")

        self.workflow_widget = QWidget()
        wf_root = QVBoxLayout(self.workflow_widget)
        self.workflow_tabs = QTabWidget()
        wf_root.addWidget(self.workflow_tabs)

        infer_tab = QWidget()
        infer_form = QFormLayout(infer_tab)
        self.orch_infer_config_edit = QLineEdit("configs/inference.default.yml")
        self.orch_infer_set_edit = QLineEdit()
        self.orch_infer_set_edit.setPlaceholderText("key=value,key2=value2")
        self.orch_infer_image_edit = QLineEdit()
        self.orch_infer_image_edit.setPlaceholderText("Image path (optional if in config)")
        self.orch_infer_model_edit = QLineEdit()
        self.orch_infer_model_edit.setPlaceholderText("Model name (optional, defaults selected GUI model)")
        self.orch_infer_output_edit = QLineEdit("outputs/inference")
        self.orch_infer_enable_gpu = QCheckBox("Enable GPU")
        self.orch_infer_enable_gpu.setChecked(False)
        self.orch_infer_device_policy = QComboBox()
        self.orch_infer_device_policy.addItems(["cpu", "auto", "cuda", "mps"])
        self.orch_infer_device_policy.setCurrentText("cpu")
        self.btn_orch_infer = QPushButton("Run Inference Job")
        self.btn_orch_infer.clicked.connect(self.on_orchestrate_inference)
        infer_form.addRow("Config", self.orch_infer_config_edit)
        infer_form.addRow("Overrides", self.orch_infer_set_edit)
        infer_form.addRow("Image", self.orch_infer_image_edit)
        infer_form.addRow("Model", self.orch_infer_model_edit)
        infer_form.addRow("Output Dir", self.orch_infer_output_edit)
        infer_form.addRow(self.orch_infer_enable_gpu, self.orch_infer_device_policy)
        infer_form.addRow(self.btn_orch_infer)
        self.workflow_tabs.addTab(infer_tab, "Inference")

        train_tab = QWidget()
        train_form = QFormLayout(train_tab)
        self.orch_train_config_edit = QLineEdit("configs/train.default.yml")
        self.orch_train_set_edit = QLineEdit()
        self.orch_train_set_edit.setPlaceholderText("key=value,key2=value2")
        self.orch_train_dataset_edit = QLineEdit("outputs/packaged_dataset")
        self.orch_train_output_edit = QLineEdit("outputs/training")
        self.orch_train_backend = QComboBox()
        self.orch_train_backend.addItems(
            [
                "unet_binary",
                "smp_unet_resnet18",
                "transunet_tiny",
                "segformer_mini",
                "hf_segformer_b0",
                "hf_segformer_b2",
                "hf_segformer_b5",
                "torch_pixel",
                "sklearn_pixel",
            ]
        )
        self.orch_train_backend.setCurrentText("unet_binary")
        self.orch_train_enable_gpu = QCheckBox("Enable GPU")
        self.orch_train_enable_gpu.setChecked(False)
        self.orch_train_device_policy = QComboBox()
        self.orch_train_device_policy.addItems(["cpu", "auto", "cuda", "mps"])
        self.orch_train_device_policy.setCurrentText("cpu")
        self.orch_train_max_samples = QSpinBox()
        self.orch_train_max_samples.setRange(1000, 2000000)
        self.orch_train_max_samples.setValue(250000)
        self.orch_train_epochs = QSpinBox()
        self.orch_train_epochs.setRange(1, 1000)
        self.orch_train_epochs.setValue(8)
        self.orch_train_batch_size = QSpinBox()
        self.orch_train_batch_size.setRange(1, 131072)
        self.orch_train_batch_size.setValue(8)
        self.orch_train_learning_rate = QDoubleSpinBox()
        self.orch_train_learning_rate.setDecimals(6)
        self.orch_train_learning_rate.setRange(0.000001, 1.0)
        self.orch_train_learning_rate.setValue(0.001)
        self.orch_train_weight_decay = QDoubleSpinBox()
        self.orch_train_weight_decay.setDecimals(6)
        self.orch_train_weight_decay.setRange(0.0, 1.0)
        self.orch_train_weight_decay.setValue(0.00001)
        self.orch_train_patience = QSpinBox()
        self.orch_train_patience.setRange(1, 200)
        self.orch_train_patience.setValue(5)
        self.orch_train_min_delta = QDoubleSpinBox()
        self.orch_train_min_delta.setDecimals(6)
        self.orch_train_min_delta.setRange(0.0, 1.0)
        self.orch_train_min_delta.setValue(0.0001)
        self.orch_train_checkpoint_every = QSpinBox()
        self.orch_train_checkpoint_every.setRange(1, 100)
        self.orch_train_checkpoint_every.setValue(1)
        self.orch_train_resume_checkpoint = QLineEdit()
        self.orch_train_resume_checkpoint.setPlaceholderText("Optional checkpoint path for resume")
        self.orch_train_val_tracking_samples = QSpinBox()
        self.orch_train_val_tracking_samples.setRange(0, 200)
        self.orch_train_val_tracking_samples.setValue(6)
        self.orch_train_val_tracking_fixed = QLineEdit()
        self.orch_train_val_tracking_fixed.setPlaceholderText("Fixed val names, separated by | (e.g. val_000.png|val_123.png)")
        self.orch_train_val_tracking_seed = QSpinBox()
        self.orch_train_val_tracking_seed.setRange(0, 100000)
        self.orch_train_val_tracking_seed.setValue(17)
        self.orch_train_write_html_report = QCheckBox("Write HTML report")
        self.orch_train_write_html_report.setChecked(True)
        self.orch_train_progress_interval = QSpinBox()
        self.orch_train_progress_interval.setRange(1, 50)
        self.orch_train_progress_interval.setValue(10)
        self.orch_train_seed = QSpinBox()
        self.orch_train_seed.setRange(0, 100000)
        self.orch_train_seed.setValue(42)
        self.orch_train_require_qa = QCheckBox("Require dataset QA pass before launch")
        self.orch_train_require_qa.setChecked(True)
        self.btn_orch_train = QPushButton("Run Training Job")
        self.btn_orch_train.clicked.connect(self.on_orchestrate_training)
        train_form.addRow("Config", self.orch_train_config_edit)
        train_form.addRow("Overrides", self.orch_train_set_edit)
        train_form.addRow("Backend", self.orch_train_backend)
        train_form.addRow("Dataset Dir", self.orch_train_dataset_edit)
        train_form.addRow("Output Dir", self.orch_train_output_edit)
        train_form.addRow(self.orch_train_enable_gpu, self.orch_train_device_policy)
        train_form.addRow("Max Samples", self.orch_train_max_samples)
        train_form.addRow("Epochs", self.orch_train_epochs)
        train_form.addRow("Batch Size", self.orch_train_batch_size)
        train_form.addRow("Learning Rate", self.orch_train_learning_rate)
        train_form.addRow("Weight Decay", self.orch_train_weight_decay)
        train_form.addRow("Early Stop Patience", self.orch_train_patience)
        train_form.addRow("Early Stop Min Δ", self.orch_train_min_delta)
        train_form.addRow("Checkpoint Every", self.orch_train_checkpoint_every)
        train_form.addRow("Resume Checkpoint", self.orch_train_resume_checkpoint)
        train_form.addRow("Track Val Samples", self.orch_train_val_tracking_samples)
        train_form.addRow("Fixed Val Names", self.orch_train_val_tracking_fixed)
        train_form.addRow("Val Tracking Seed", self.orch_train_val_tracking_seed)
        train_form.addRow("Progress Log Interval (%)", self.orch_train_progress_interval)
        train_form.addRow(self.orch_train_write_html_report)
        train_form.addRow("Seed", self.orch_train_seed)
        train_form.addRow(self.orch_train_require_qa)
        train_form.addRow(self.btn_orch_train)
        self.workflow_tabs.addTab(train_tab, "Training")

        eval_tab = QWidget()
        eval_form = QFormLayout(eval_tab)
        self.orch_eval_config_edit = QLineEdit("configs/evaluate.default.yml")
        self.orch_eval_set_edit = QLineEdit()
        self.orch_eval_set_edit.setPlaceholderText("key=value,key2=value2")
        self.orch_eval_dataset_edit = QLineEdit("outputs/packaged_dataset")
        self.orch_eval_model_edit = QLineEdit("outputs/training/torch_pixel_classifier.pt")
        self.orch_eval_enable_gpu = QCheckBox("Enable GPU")
        self.orch_eval_enable_gpu.setChecked(False)
        self.orch_eval_device_policy = QComboBox()
        self.orch_eval_device_policy.addItems(["cpu", "auto", "cuda", "mps"])
        self.orch_eval_device_policy.setCurrentText("cpu")
        self.orch_eval_split_combo = QComboBox()
        self.orch_eval_split_combo.addItems(["val", "test", "train"])
        self.orch_eval_output_edit = QLineEdit("outputs/evaluation/pixel_eval_report.json")
        self.orch_eval_tracking_samples = QSpinBox()
        self.orch_eval_tracking_samples.setRange(0, 200)
        self.orch_eval_tracking_samples.setValue(8)
        self.orch_eval_tracking_seed = QSpinBox()
        self.orch_eval_tracking_seed.setRange(0, 100000)
        self.orch_eval_tracking_seed.setValue(17)
        self.orch_eval_write_html_report = QCheckBox("Write HTML report")
        self.orch_eval_write_html_report.setChecked(True)
        self.btn_orch_eval = QPushButton("Run Evaluation Job")
        self.btn_orch_eval.clicked.connect(self.on_orchestrate_evaluation)
        eval_form.addRow("Config", self.orch_eval_config_edit)
        eval_form.addRow("Overrides", self.orch_eval_set_edit)
        eval_form.addRow("Dataset Dir", self.orch_eval_dataset_edit)
        eval_form.addRow("Model Path", self.orch_eval_model_edit)
        eval_form.addRow(self.orch_eval_enable_gpu, self.orch_eval_device_policy)
        eval_form.addRow("Split", self.orch_eval_split_combo)
        eval_form.addRow("Output Path", self.orch_eval_output_edit)
        eval_form.addRow("Track Sample Panels", self.orch_eval_tracking_samples)
        eval_form.addRow("Tracking Seed", self.orch_eval_tracking_seed)
        eval_form.addRow(self.orch_eval_write_html_report)
        eval_form.addRow(self.btn_orch_eval)
        self.workflow_tabs.addTab(eval_tab, "Evaluation")

        package_tab = QWidget()
        package_form = QFormLayout(package_tab)
        self.dataset_input_edit = QLineEdit()
        self.dataset_input_edit.setPlaceholderText("Correction exports directory")
        self.dataset_output_edit = QLineEdit()
        self.dataset_output_edit.setPlaceholderText("Packaged dataset output directory")
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.1, 0.95)
        self.train_ratio_spin.setDecimals(2)
        self.train_ratio_spin.setValue(0.8)
        self.val_ratio_spin = QDoubleSpinBox()
        self.val_ratio_spin.setRange(0.0, 0.8)
        self.val_ratio_spin.setDecimals(2)
        self.val_ratio_spin.setValue(0.1)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 100000)
        self.seed_spin.setValue(42)
        self.btn_package = QPushButton("Package Dataset Job")
        self.btn_package.clicked.connect(self.on_package_dataset)
        package_form.addRow("Corrections Dir", self.dataset_input_edit)
        package_form.addRow("Output Dir", self.dataset_output_edit)
        package_form.addRow("Train Ratio", self.train_ratio_spin)
        package_form.addRow("Val Ratio", self.val_ratio_spin)
        package_form.addRow("Seed", self.seed_spin)
        package_form.addRow(self.btn_package)
        self.workflow_tabs.addTab(package_tab, "Packaging")

        prep_tab = QWidget()
        self.workflow_prep_tab = prep_tab
        prep_root = QVBoxLayout(prep_tab)
        prep_form = QFormLayout()
        self.orch_prepare_config_edit = QLineEdit("configs/dataset_prepare.default.yml")
        self.orch_prepare_set_edit = QLineEdit()
        self.orch_prepare_set_edit.setPlaceholderText("key=value,key2=value2")
        self.orch_prepare_dataset_edit = QLineEdit("data")
        self.orch_prepare_output_edit = QLineEdit("outputs/prepared_dataset")
        self.orch_prepare_train_ratio = QDoubleSpinBox()
        self.orch_prepare_train_ratio.setRange(0.05, 0.95)
        self.orch_prepare_train_ratio.setDecimals(2)
        self.orch_prepare_train_ratio.setValue(0.8)
        self.orch_prepare_val_ratio = QDoubleSpinBox()
        self.orch_prepare_val_ratio.setRange(0.0, 0.8)
        self.orch_prepare_val_ratio.setDecimals(2)
        self.orch_prepare_val_ratio.setValue(0.1)
        self.orch_prepare_test_ratio = QDoubleSpinBox()
        self.orch_prepare_test_ratio.setRange(0.0, 0.8)
        self.orch_prepare_test_ratio.setDecimals(2)
        self.orch_prepare_test_ratio.setValue(0.1)
        self.orch_prepare_seed = QSpinBox()
        self.orch_prepare_seed.setRange(0, 100000)
        self.orch_prepare_seed.setValue(42)
        self.orch_prepare_id_width = QSpinBox()
        self.orch_prepare_id_width.setRange(1, 12)
        self.orch_prepare_id_width.setValue(6)
        self.orch_prepare_strategy = QComboBox()
        self.orch_prepare_strategy.addItems(["leakage_aware", "random"])
        self.orch_prepare_strategy.setCurrentText("leakage_aware")
        self.orch_prepare_group_mode = QComboBox()
        self.orch_prepare_group_mode.addItems(["suffix_aware", "stem", "regex"])
        self.orch_prepare_group_mode.setCurrentText("suffix_aware")
        self.orch_prepare_group_regex = QLineEdit()
        self.orch_prepare_group_regex.setPlaceholderText("Optional when leakage group mode=regex")
        self.orch_prepare_mask_type = QComboBox()
        self.orch_prepare_mask_type.addItems(["indexed", "rgb_colormap", "auto"])
        self.orch_prepare_mask_type.setCurrentText("indexed")
        self.orch_prepare_colormap = QTextEdit()
        self.orch_prepare_colormap.setPlaceholderText('JSON object, e.g. {"0":[0,0,0],"1":[255,0,0]}')
        self.orch_prepare_colormap.setMaximumHeight(90)
        self.orch_prepare_colormap_strict = QCheckBox("Strict colormap (unknown RGB colors fail)")
        self.orch_prepare_colormap_strict.setChecked(True)

        self.orch_qa_config_edit = QLineEdit("configs/dataset_qa.default.yml")
        self.orch_qa_output_edit = QLineEdit("outputs/dataops/dataset_qa_report.json")
        self.orch_qa_imbalance_warn = QDoubleSpinBox()
        self.orch_qa_imbalance_warn.setRange(0.5, 1.0)
        self.orch_qa_imbalance_warn.setDecimals(3)
        self.orch_qa_imbalance_warn.setValue(0.98)
        self.orch_qa_strict = QCheckBox("Strict QA")
        self.orch_qa_strict.setChecked(True)

        prep_form.addRow("Prepare Config", self.orch_prepare_config_edit)
        prep_form.addRow("Prepare Overrides", self.orch_prepare_set_edit)
        prep_form.addRow("Dataset Dir", self.orch_prepare_dataset_edit)
        prep_form.addRow("Prepared Output Dir", self.orch_prepare_output_edit)
        prep_form.addRow("Train Ratio", self.orch_prepare_train_ratio)
        prep_form.addRow("Val Ratio", self.orch_prepare_val_ratio)
        prep_form.addRow("Test Ratio", self.orch_prepare_test_ratio)
        prep_form.addRow("Seed", self.orch_prepare_seed)
        prep_form.addRow("ID Width", self.orch_prepare_id_width)
        prep_form.addRow("Split Strategy", self.orch_prepare_strategy)
        prep_form.addRow("Leakage Group Mode", self.orch_prepare_group_mode)
        prep_form.addRow("Leakage Group Regex", self.orch_prepare_group_regex)
        prep_form.addRow("Mask Input Type", self.orch_prepare_mask_type)
        prep_form.addRow("Mask Colormap JSON", self.orch_prepare_colormap)
        prep_form.addRow(self.orch_prepare_colormap_strict)
        prep_form.addRow("QA Config", self.orch_qa_config_edit)
        prep_form.addRow("QA Output Path", self.orch_qa_output_edit)
        prep_form.addRow("QA Imbalance Warn", self.orch_qa_imbalance_warn)
        prep_form.addRow(self.orch_qa_strict)
        prep_root.addLayout(prep_form)

        prep_actions = QHBoxLayout()
        self.btn_orch_prepare_preview = QPushButton("Preview Dataset Plan")
        self.btn_orch_prepare_preview.clicked.connect(self.on_preview_dataset_prepare)
        self.btn_orch_prepare_run = QPushButton("Run Dataset Prepare Job")
        self.btn_orch_prepare_run.clicked.connect(self.on_orchestrate_dataset_prepare)
        self.btn_orch_run_qa = QPushButton("Run QA Check")
        self.btn_orch_run_qa.clicked.connect(self.on_run_dataset_qa)
        self.btn_orch_apply_train_dataset = QPushButton("Use Prepared Dir In Training")
        self.btn_orch_apply_train_dataset.clicked.connect(self.on_apply_prepared_dataset_to_training)
        prep_actions.addWidget(self.btn_orch_prepare_preview)
        prep_actions.addWidget(self.btn_orch_prepare_run)
        prep_actions.addWidget(self.btn_orch_run_qa)
        prep_actions.addWidget(self.btn_orch_apply_train_dataset)
        prep_root.addLayout(prep_actions)

        self.dataset_preview_summary = QLabel("Preview: not generated")
        prep_root.addWidget(self.dataset_preview_summary)
        self.dataset_preview_filter = QLineEdit()
        self.dataset_preview_filter.setPlaceholderText("Filter preview rows by id/stem/split/group/path")
        self.dataset_preview_filter.textChanged.connect(self._refresh_dataset_preview_table)
        prep_root.addWidget(self.dataset_preview_filter)

        self.dataset_preview_table = QTableWidget(0, 7)
        self.dataset_preview_table.setHorizontalHeaderLabels(
            ["Global ID", "Split", "Source Group", "Original Stem", "New Name", "Image Path", "Mask Path"]
        )
        self.dataset_preview_table.horizontalHeader().setStretchLastSection(True)
        self.dataset_preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.dataset_preview_table.setAlternatingRowColors(True)
        prep_root.addWidget(self.dataset_preview_table, stretch=1)
        self.workflow_tabs.addTab(prep_tab, "Dataset Prep + QA")

        review_tab = QWidget()
        review_root = QVBoxLayout(review_tab)
        review_form = QFormLayout()
        self.review_report_a_edit = QLineEdit("outputs/training/report.json")
        self.review_report_b_edit = QLineEdit("outputs/evaluation/pixel_eval_report.json")
        pick_a_row = QWidget()
        pick_a_layout = QHBoxLayout(pick_a_row)
        pick_a_layout.setContentsMargins(0, 0, 0, 0)
        pick_a_layout.addWidget(self.review_report_a_edit)
        self.btn_review_pick_a = QPushButton("Browse")
        self.btn_review_pick_a.clicked.connect(self.on_pick_review_report_a)
        pick_a_layout.addWidget(self.btn_review_pick_a)
        pick_b_row = QWidget()
        pick_b_layout = QHBoxLayout(pick_b_row)
        pick_b_layout.setContentsMargins(0, 0, 0, 0)
        pick_b_layout.addWidget(self.review_report_b_edit)
        self.btn_review_pick_b = QPushButton("Browse")
        self.btn_review_pick_b.clicked.connect(self.on_pick_review_report_b)
        pick_b_layout.addWidget(self.btn_review_pick_b)
        review_form.addRow("Baseline Report", pick_a_row)
        review_form.addRow("Candidate Report", pick_b_row)
        review_root.addLayout(review_form)

        review_actions = QHBoxLayout()
        self.btn_review_load_a = QPushButton("Load Baseline Summary")
        self.btn_review_load_a.clicked.connect(self.on_load_review_report_a)
        self.btn_review_load_b = QPushButton("Load Candidate Summary")
        self.btn_review_load_b.clicked.connect(self.on_load_review_report_b)
        self.btn_review_compare = QPushButton("Compare Reports")
        self.btn_review_compare.clicked.connect(self.on_compare_review_reports)
        review_actions.addWidget(self.btn_review_load_a)
        review_actions.addWidget(self.btn_review_load_b)
        review_actions.addWidget(self.btn_review_compare)
        review_actions.addStretch(1)
        review_root.addLayout(review_actions)

        self.review_compare_meta = QLabel("Comparison: not computed")
        review_root.addWidget(self.review_compare_meta)

        review_split = QSplitter(Qt.Horizontal)
        self.review_summary_a_text = QPlainTextEdit()
        self.review_summary_a_text.setReadOnly(True)
        self.review_summary_b_text = QPlainTextEdit()
        self.review_summary_b_text.setReadOnly(True)
        review_split.addWidget(self.review_summary_a_text)
        review_split.addWidget(self.review_summary_b_text)
        review_split.setSizes([800, 800])
        review_root.addWidget(review_split, stretch=1)

        self.review_compare_table = QTableWidget(0, 5)
        self.review_compare_table.setHorizontalHeaderLabels(["Metric", "Baseline", "Candidate", "Delta", "Delta %"])
        self.review_compare_table.horizontalHeader().setStretchLastSection(True)
        self.review_compare_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.review_compare_table.setAlternatingRowColors(True)
        review_root.addWidget(self.review_compare_table, stretch=1)
        self.workflow_tabs.addTab(review_tab, "Run Review")

        hpc_tab = QWidget()
        self.workflow_hpc_tab = hpc_tab
        hpc_root = QVBoxLayout(hpc_tab)
        hpc_form = QFormLayout()
        self.orch_hpc_config_edit = QLineEdit("configs/hpc_ga.default.yml")
        self.orch_hpc_set_edit = QLineEdit()
        self.orch_hpc_set_edit.setPlaceholderText("key=value,key2=value2")
        self.orch_hpc_dataset_edit = QLineEdit("outputs/prepared_dataset")
        self.orch_hpc_output_edit = QLineEdit("outputs/hpc_ga_bundle")
        self.orch_hpc_experiment_name = QLineEdit("microseg_hpc_ga_sweep")
        self.orch_hpc_scheduler_combo = QComboBox()
        self.orch_hpc_scheduler_combo.addItems(["slurm", "pbs", "local"])
        self.orch_hpc_scheduler_combo.setCurrentText("slurm")
        self.orch_hpc_run_mode_combo = QComboBox()
        self.orch_hpc_run_mode_combo.addItems(["train_eval", "train_only"])
        self.orch_hpc_run_mode_combo.setCurrentText("train_eval")
        self.orch_hpc_architectures_edit = QLineEdit(
            "unet_binary,smp_unet_resnet18,hf_segformer_b0,hf_segformer_b2,hf_segformer_b5,transunet_tiny,segformer_mini,torch_pixel"
        )
        self.orch_hpc_num_candidates = QSpinBox()
        self.orch_hpc_num_candidates.setRange(1, 128)
        self.orch_hpc_num_candidates.setValue(8)
        self.orch_hpc_population = QSpinBox()
        self.orch_hpc_population.setRange(2, 512)
        self.orch_hpc_population.setValue(24)
        self.orch_hpc_generations = QSpinBox()
        self.orch_hpc_generations.setRange(1, 100)
        self.orch_hpc_generations.setValue(8)
        self.orch_hpc_mutation = QDoubleSpinBox()
        self.orch_hpc_mutation.setRange(0.0, 1.0)
        self.orch_hpc_mutation.setDecimals(3)
        self.orch_hpc_mutation.setValue(0.2)
        self.orch_hpc_crossover = QDoubleSpinBox()
        self.orch_hpc_crossover.setRange(0.0, 1.0)
        self.orch_hpc_crossover.setDecimals(3)
        self.orch_hpc_crossover.setValue(0.7)
        self.orch_hpc_seed = QSpinBox()
        self.orch_hpc_seed.setRange(0, 1000000)
        self.orch_hpc_seed.setValue(42)
        self.orch_hpc_lr_min = QDoubleSpinBox()
        self.orch_hpc_lr_min.setDecimals(8)
        self.orch_hpc_lr_min.setRange(0.0000001, 1.0)
        self.orch_hpc_lr_min.setValue(0.0001)
        self.orch_hpc_lr_max = QDoubleSpinBox()
        self.orch_hpc_lr_max.setDecimals(8)
        self.orch_hpc_lr_max.setRange(0.0000001, 1.0)
        self.orch_hpc_lr_max.setValue(0.01)
        self.orch_hpc_batch_sizes = QLineEdit("4,8,16,32")
        self.orch_hpc_epochs_min = QSpinBox()
        self.orch_hpc_epochs_min.setRange(1, 1000)
        self.orch_hpc_epochs_min.setValue(8)
        self.orch_hpc_epochs_max = QSpinBox()
        self.orch_hpc_epochs_max.setRange(1, 1000)
        self.orch_hpc_epochs_max.setValue(40)
        self.orch_hpc_wd_min = QDoubleSpinBox()
        self.orch_hpc_wd_min.setDecimals(8)
        self.orch_hpc_wd_min.setRange(0.000000001, 1.0)
        self.orch_hpc_wd_min.setValue(0.000001)
        self.orch_hpc_wd_max = QDoubleSpinBox()
        self.orch_hpc_wd_max.setDecimals(8)
        self.orch_hpc_wd_max.setRange(0.000000001, 1.0)
        self.orch_hpc_wd_max.setValue(0.001)
        self.orch_hpc_ms_min = QSpinBox()
        self.orch_hpc_ms_min.setRange(1, 1000000000)
        self.orch_hpc_ms_min.setValue(50000)
        self.orch_hpc_ms_max = QSpinBox()
        self.orch_hpc_ms_max.setRange(1, 1000000000)
        self.orch_hpc_ms_max.setValue(250000)
        self.orch_hpc_fitness_mode = QComboBox()
        self.orch_hpc_fitness_mode.addItems(["novelty", "feedback_hybrid"])
        self.orch_hpc_fitness_mode.setCurrentText("novelty")
        self.orch_hpc_feedback_sources = QLineEdit()
        self.orch_hpc_feedback_sources.setPlaceholderText("Comma-separated prior bundle dirs or ga_plan_manifest.json paths")
        self.orch_hpc_feedback_min_samples = QSpinBox()
        self.orch_hpc_feedback_min_samples.setRange(1, 2000)
        self.orch_hpc_feedback_min_samples.setValue(3)
        self.orch_hpc_feedback_k = QSpinBox()
        self.orch_hpc_feedback_k.setRange(1, 200)
        self.orch_hpc_feedback_k.setValue(5)
        self.orch_hpc_exploration_weight = QDoubleSpinBox()
        self.orch_hpc_exploration_weight.setRange(0.0, 1.0)
        self.orch_hpc_exploration_weight.setDecimals(3)
        self.orch_hpc_exploration_weight.setValue(0.55)
        self.orch_hpc_w_iou = QDoubleSpinBox()
        self.orch_hpc_w_iou.setRange(0.0, 10.0)
        self.orch_hpc_w_iou.setDecimals(3)
        self.orch_hpc_w_iou.setValue(0.50)
        self.orch_hpc_w_f1 = QDoubleSpinBox()
        self.orch_hpc_w_f1.setRange(0.0, 10.0)
        self.orch_hpc_w_f1.setDecimals(3)
        self.orch_hpc_w_f1.setValue(0.30)
        self.orch_hpc_w_acc = QDoubleSpinBox()
        self.orch_hpc_w_acc.setRange(0.0, 10.0)
        self.orch_hpc_w_acc.setDecimals(3)
        self.orch_hpc_w_acc.setValue(0.20)
        self.orch_hpc_w_runtime = QDoubleSpinBox()
        self.orch_hpc_w_runtime.setRange(0.0, 10.0)
        self.orch_hpc_w_runtime.setDecimals(3)
        self.orch_hpc_w_runtime.setValue(0.05)
        self.orch_hpc_enable_gpu = QCheckBox("Enable GPU")
        self.orch_hpc_enable_gpu.setChecked(True)
        self.orch_hpc_device_policy = QComboBox()
        self.orch_hpc_device_policy.addItems(["auto", "cpu", "cuda", "mps"])
        self.orch_hpc_device_policy.setCurrentText("auto")
        self.orch_hpc_queue = QLineEdit()
        self.orch_hpc_queue.setPlaceholderText("partition/queue (optional)")
        self.orch_hpc_account = QLineEdit()
        self.orch_hpc_account.setPlaceholderText("project/account (optional)")
        self.orch_hpc_qos = QLineEdit()
        self.orch_hpc_qos.setPlaceholderText("QoS (optional)")
        self.orch_hpc_gpus = QSpinBox()
        self.orch_hpc_gpus.setRange(0, 16)
        self.orch_hpc_gpus.setValue(1)
        self.orch_hpc_cpus = QSpinBox()
        self.orch_hpc_cpus.setRange(1, 512)
        self.orch_hpc_cpus.setValue(8)
        self.orch_hpc_mem = QSpinBox()
        self.orch_hpc_mem.setRange(1, 4096)
        self.orch_hpc_mem.setValue(32)
        self.orch_hpc_time = QLineEdit("08:00:00")
        self.orch_hpc_job_prefix = QLineEdit("microseg")
        self.orch_hpc_python = QLineEdit("python")
        self.orch_hpc_cli_path = QLineEdit("scripts/microseg_cli.py")
        self.orch_hpc_base_train = QLineEdit("configs/train.default.yml")
        self.orch_hpc_base_eval = QLineEdit("configs/evaluate.default.yml")
        self.orch_hpc_eval_split = QComboBox()
        self.orch_hpc_eval_split.addItems(["val", "test", "train"])
        self.orch_hpc_eval_split.setCurrentText("val")
        self.orch_hpc_feedback_top_k = QSpinBox()
        self.orch_hpc_feedback_top_k.setRange(1, 500)
        self.orch_hpc_feedback_top_k.setValue(10)
        self.orch_hpc_feedback_report_output = QLineEdit("outputs/hpc_ga_feedback/feedback_report.json")

        hpc_form.addRow("Config", self.orch_hpc_config_edit)
        hpc_form.addRow("Overrides", self.orch_hpc_set_edit)
        hpc_form.addRow("Dataset Dir", self.orch_hpc_dataset_edit)
        hpc_form.addRow("Bundle Output Dir", self.orch_hpc_output_edit)
        hpc_form.addRow("Experiment Name", self.orch_hpc_experiment_name)
        hpc_form.addRow("Scheduler", self.orch_hpc_scheduler_combo)
        hpc_form.addRow("Run Mode", self.orch_hpc_run_mode_combo)
        hpc_form.addRow("Architectures", self.orch_hpc_architectures_edit)
        hpc_form.addRow("# Candidates", self.orch_hpc_num_candidates)
        hpc_form.addRow("Population Size", self.orch_hpc_population)
        hpc_form.addRow("Generations", self.orch_hpc_generations)
        hpc_form.addRow("Mutation Rate", self.orch_hpc_mutation)
        hpc_form.addRow("Crossover Rate", self.orch_hpc_crossover)
        hpc_form.addRow("Seed", self.orch_hpc_seed)
        hpc_form.addRow("Learning Rate Min", self.orch_hpc_lr_min)
        hpc_form.addRow("Learning Rate Max", self.orch_hpc_lr_max)
        hpc_form.addRow("Batch Size Choices", self.orch_hpc_batch_sizes)
        hpc_form.addRow("Epochs Min", self.orch_hpc_epochs_min)
        hpc_form.addRow("Epochs Max", self.orch_hpc_epochs_max)
        hpc_form.addRow("Weight Decay Min", self.orch_hpc_wd_min)
        hpc_form.addRow("Weight Decay Max", self.orch_hpc_wd_max)
        hpc_form.addRow("Max Samples Min", self.orch_hpc_ms_min)
        hpc_form.addRow("Max Samples Max", self.orch_hpc_ms_max)
        hpc_form.addRow("Fitness Mode", self.orch_hpc_fitness_mode)
        hpc_form.addRow("Feedback Sources", self.orch_hpc_feedback_sources)
        hpc_form.addRow("Feedback Min Samples", self.orch_hpc_feedback_min_samples)
        hpc_form.addRow("Feedback K (kNN)", self.orch_hpc_feedback_k)
        hpc_form.addRow("Exploration Weight", self.orch_hpc_exploration_weight)
        hpc_form.addRow("Fitness Weight Mean IoU", self.orch_hpc_w_iou)
        hpc_form.addRow("Fitness Weight Macro F1", self.orch_hpc_w_f1)
        hpc_form.addRow("Fitness Weight Pixel Accuracy", self.orch_hpc_w_acc)
        hpc_form.addRow("Fitness Weight Runtime", self.orch_hpc_w_runtime)
        hpc_form.addRow(self.orch_hpc_enable_gpu, self.orch_hpc_device_policy)
        hpc_form.addRow("Queue/Partition", self.orch_hpc_queue)
        hpc_form.addRow("Account", self.orch_hpc_account)
        hpc_form.addRow("QoS", self.orch_hpc_qos)
        hpc_form.addRow("GPUs per Job", self.orch_hpc_gpus)
        hpc_form.addRow("CPUs per Task", self.orch_hpc_cpus)
        hpc_form.addRow("Memory (GB)", self.orch_hpc_mem)
        hpc_form.addRow("Time Limit", self.orch_hpc_time)
        hpc_form.addRow("Job Prefix", self.orch_hpc_job_prefix)
        hpc_form.addRow("Python Executable", self.orch_hpc_python)
        hpc_form.addRow("microseg_cli.py Path", self.orch_hpc_cli_path)
        hpc_form.addRow("Base Train Config", self.orch_hpc_base_train)
        hpc_form.addRow("Base Eval Config", self.orch_hpc_base_eval)
        hpc_form.addRow("Eval Split", self.orch_hpc_eval_split)
        hpc_form.addRow("Feedback Top K", self.orch_hpc_feedback_top_k)
        hpc_form.addRow("Feedback Report Output", self.orch_hpc_feedback_report_output)
        hpc_root.addLayout(hpc_form)

        hpc_actions = QHBoxLayout()
        self.btn_orch_hpc_generate = QPushButton("Generate HPC GA Bundle")
        self.btn_orch_hpc_generate.clicked.connect(self.on_orchestrate_hpc_ga)
        hpc_actions.addWidget(self.btn_orch_hpc_generate)
        self.btn_orch_hpc_feedback = QPushButton("Analyze Feedback")
        self.btn_orch_hpc_feedback.clicked.connect(self.on_orchestrate_hpc_feedback_report)
        hpc_actions.addWidget(self.btn_orch_hpc_feedback)
        hpc_actions.addStretch(1)
        hpc_root.addLayout(hpc_actions)

        self.orch_hpc_preview = QPlainTextEdit()
        self.orch_hpc_preview.setReadOnly(True)
        self.orch_hpc_preview.setMaximumBlockCount(800)
        self.orch_hpc_preview.setPlainText(
            "HPC GA Planner\n"
            "- Configure architectures + GA search ranges.\n"
            "- Optional: set feedback sources and switch fitness mode to feedback_hybrid.\n"
            "- Use 'Analyze Feedback' to build a ranked report from previous runs.\n"
            "- Click 'Generate HPC GA Bundle' to write scheduler scripts.\n"
            "- Upload the bundle to HPC and run submit_all.sh."
        )
        hpc_root.addWidget(self.orch_hpc_preview)
        self.workflow_tabs.addTab(hpc_tab, "HPC GA Planner")

        profile_bar = QHBoxLayout()
        self.workflow_profile_scope = QComboBox()
        self.workflow_profile_scope.addItems(["dataset_prepare", "training", "evaluation", "hpc_ga"])
        self.btn_workflow_profile_save = QPushButton("Save Workflow Profile")
        self.btn_workflow_profile_save.clicked.connect(self.on_save_workflow_profile)
        self.btn_workflow_profile_load = QPushButton("Load Workflow Profile")
        self.btn_workflow_profile_load.clicked.connect(self.on_load_workflow_profile)
        profile_bar.addWidget(QLabel("Profile Scope"))
        profile_bar.addWidget(self.workflow_profile_scope)
        profile_bar.addWidget(self.btn_workflow_profile_save)
        profile_bar.addWidget(self.btn_workflow_profile_load)
        profile_bar.addStretch(1)
        wf_root.addLayout(profile_bar)

        self.workflow_notes = QTextEdit()
        self.workflow_notes.setReadOnly(True)
        self.workflow_notes.setPlainText(
            "Orchestration Log\\n"
            "- One active job at a time\\n"
            "- Commands run through scripts/microseg_cli.py\\n"
            "- Use YAML config + overrides for reproducibility.\\n"
            "- GPU is opt-in; fallback to CPU is automatic if unavailable.\\n"
            "- Dataset Prep + QA tab supports preview, prepare, QA gating, and profile save/load.\\n"
            "- Run Review tab supports report loading and metric-delta comparison.\\n"
            "- HPC GA Planner generates multi-candidate scheduler bundles for HPC runs."
        )
        wf_root.addWidget(self.workflow_notes)

        self.tabs.addTab(self.workflow_widget, "Workflow Hub")

        self._connect_scroll_sync()
        self._reload_class_combo()
        self._toggle_conventional_controls(self.model_combo.currentText())

        self.results_orientation_bins.valueChanged.connect(self._queue_results_refresh)
        self.results_size_bins.valueChanged.connect(self._queue_results_refresh)
        self.results_min_feature.valueChanged.connect(self._queue_results_refresh)
        self.results_size_scale.currentTextChanged.connect(self._queue_results_refresh)
        self.results_cmap.currentTextChanged.connect(self._queue_results_refresh)

        status = QHBoxLayout()
        layout.addLayout(status)

        self.zoom_label = QLabel("Zoom: 100%")
        self.cursor_label = QLabel("Cursor: -,-")
        self.action_label = QLabel("Actions: 0")
        self.tool_label = QLabel("Tool: brush/add")
        status.addWidget(self.zoom_label)
        status.addWidget(self.cursor_label)
        status.addWidget(self.action_label)
        status.addWidget(self.tool_label)
        status.addStretch(1)

        self.log_box = QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(1200)
        layout.addWidget(self.log_box, stretch=0)

        self.model_desc.hide()
        for widget in [
            self.config_path_edit,
            self.btn_config_browse,
            self.config_overrides_edit,
            self.calibration_status_label,
            self.btn_scan_metadata_calibration,
            self.btn_calibrate_line,
            self.btn_clear_calibration,
            self.chk_fmt_indexed,
            self.chk_fmt_color,
            self.chk_fmt_npy,
            self.chk_report_html,
            self.chk_report_pdf,
            self.chk_report_csv,
            self.report_profile_combo,
            self.report_top_k_spin,
            self.chk_artifact_manifest,
            self.btn_export_batch,
            self.btn_save_project,
            self.btn_load_project,
            self.report_advanced_group,
            self.log_box,
        ]:
            widget.hide()

        if bool(self._ui_config.window.show_log_dock_on_start):
            self.log_box.show()
        if bool(self._ui_config.window.show_workflow_dock_on_start):
            self.tabs.setCurrentWidget(self.workflow_widget)

        workspace_menu = QMenu(self)
        workspace_menu.addAction("Appearance & Export Settings", self.on_open_appearance_settings)
        workspace_menu.addSeparator()

        def _add_toggle_action(title: str, widgets: list[QWidget], *, checked: bool = False) -> None:
            action = QAction(title, self)
            action.setCheckable(True)
            action.setChecked(checked)
            action.triggered.connect(lambda state, ws=widgets: [w.setVisible(bool(state)) for w in ws])
            workspace_menu.addAction(action)

        _add_toggle_action("Model summary", [self.model_desc], checked=False)
        _add_toggle_action("Config row", [self.config_path_edit, self.btn_config_browse, self.config_overrides_edit], checked=False)
        _add_toggle_action(
            "Calibration row",
            [self.calibration_status_label, self.btn_scan_metadata_calibration, self.btn_calibrate_line, self.btn_clear_calibration],
            checked=False,
        )
        _add_toggle_action("Conventional controls", [self.conventional_row_widget], checked=False)
        _add_toggle_action(
            "Advanced report options",
            [
                self.chk_fmt_indexed,
                self.chk_fmt_color,
                self.chk_fmt_npy,
                self.chk_report_html,
                self.chk_report_pdf,
                self.chk_report_csv,
                self.report_profile_combo,
                self.report_top_k_spin,
                self.chk_artifact_manifest,
                self.btn_export_batch,
                self.btn_save_project,
                self.btn_load_project,
                self.report_advanced_group,
            ],
            checked=False,
        )
        _add_toggle_action("Desktop log", [self.log_box], checked=False)
        workspace_menu.addSeparator()
        workspace_menu.addAction("Workflow Hub", self.on_open_workflow_hub)
        workspace_menu.addAction("Results Dashboard", self.on_open_results_dashboard)
        self.btn_workspace_gear.setMenu(workspace_menu)

    @staticmethod
    def _in_scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(False)
        area.setWidget(widget)
        return area

    def _connect_scroll_sync(self) -> None:
        def bind_pair(a: QScrollArea, b: QScrollArea) -> None:
            a.horizontalScrollBar().valueChanged.connect(
                lambda val: self._sync_scroll(a.horizontalScrollBar(), b.horizontalScrollBar(), val)
            )
            a.verticalScrollBar().valueChanged.connect(
                lambda val: self._sync_scroll(a.verticalScrollBar(), b.verticalScrollBar(), val)
            )

        bind_pair(self.raw_corr_view.scroll_area, self.corrected_scroll)
        bind_pair(self.corrected_scroll, self.raw_corr_view.scroll_area)

    def _sync_scroll(self, src, dst, value: int) -> None:
        if self._sync_scroll_guard:
            return
        self._sync_scroll_guard = True
        try:
            dst.setValue(value)
        finally:
            self._sync_scroll_guard = False

    def _bind_shortcuts(self) -> None:
        QShortcut(QKeySequence("B"), self, activated=lambda: self.tool_combo.setCurrentText("brush"))
        QShortcut(QKeySequence("P"), self, activated=lambda: self.tool_combo.setCurrentText("polygon"))
        QShortcut(QKeySequence("L"), self, activated=lambda: self.tool_combo.setCurrentText("lasso"))
        QShortcut(QKeySequence("F"), self, activated=lambda: self.tool_combo.setCurrentText("feature_select"))

        QShortcut(QKeySequence("A"), self, activated=lambda: self.mode_combo.setCurrentText("add"))
        QShortcut(QKeySequence("R"), self, activated=lambda: self.mode_combo.setCurrentText("erase"))

        QShortcut(QKeySequence("Ctrl+Z"), self, activated=self.on_undo)
        QShortcut(QKeySequence("Ctrl+Y"), self, activated=self.on_redo)
        QShortcut(QKeySequence("Ctrl++"), self, activated=self.corrected_canvas.zoom_in)
        QShortcut(QKeySequence("Ctrl+-"), self, activated=self.corrected_canvas.zoom_out)
        QShortcut(QKeySequence("Ctrl+0"), self, activated=self.corrected_canvas.zoom_reset)

        QShortcut(QKeySequence("Escape"), self, activated=self.corrected_canvas.cancel_preview)

    def _log(self, message: str) -> None:
        self.log_box.appendPlainText(message)

    def _set_image_preview(self, label: QLabel, arr: np.ndarray | QPixmap, *, zoom: float = 1.0) -> None:
        if hasattr(label, "set_image"):
            pix = arr if isinstance(arr, QPixmap) else _rgb_to_pixmap(arr)
            if zoom != 1.0 and hasattr(label, "set_zoom"):
                label.set_image(pix)
                label.set_zoom(zoom)
                return
            label.set_image(pix)
            return
        pix = arr if isinstance(arr, QPixmap) else _rgb_to_pixmap(arr)
        pix = _scale_pixmap(pix, zoom)
        label.setPixmap(pix)
        label.setFixedSize(pix.size())

    def _layer_settings(self) -> AnnotationLayerSettings:
        return AnnotationLayerSettings(
            show_predicted=self.chk_pred.isChecked(),
            show_corrected=self.chk_corr.isChecked(),
            show_difference=self.chk_diff.isChecked(),
            predicted_alpha=self.slider_pred.value() / 100.0,
            corrected_alpha=self.slider_corr.value() / 100.0,
            difference_alpha=self.slider_diff.value() / 100.0,
        )

    def _update_layer_settings(self) -> None:
        self.corrected_canvas.update_layer_settings(self._layer_settings())

    def _reload_class_combo(self) -> None:
        prev = self.class_combo.currentText()
        self.class_combo.blockSignals(True)
        self.class_combo.clear()
        for cls in sorted(self.state.class_map.classes, key=lambda c: c.index):
            self.class_combo.addItem(f"{cls.index}:{cls.name}")
        self.class_combo.blockSignals(False)
        if prev:
            idx = self.class_combo.findText(prev)
            if idx >= 0:
                self.class_combo.setCurrentIndex(idx)
            elif self.class_combo.count() > 0:
                self.class_combo.setCurrentIndex(min(1, self.class_combo.count() - 1))
        elif self.class_combo.count() > 0:
            self.class_combo.setCurrentIndex(min(1, self.class_combo.count() - 1))
        self._on_class_changed(self.class_combo.currentText())

    @staticmethod
    def _parse_class_line(line: str) -> SegmentationClass:
        parts = [p.strip() for p in line.split(",", 3)]
        if len(parts) < 3:
            raise ValueError("expected: index,name,#RRGGBB[,description]")
        idx = int(parts[0])
        name = parts[1]
        color_hex = parts[2].lstrip("#")
        if len(color_hex) != 6:
            raise ValueError(f"invalid color hex '{parts[2]}'")
        color = tuple(int(color_hex[i:i + 2], 16) for i in (0, 2, 4))
        desc = parts[3] if len(parts) > 3 else ""
        return SegmentationClass(index=idx, name=name, color_rgb=(color[0], color[1], color[2]), description=desc)

    @staticmethod
    def _class_map_to_text(class_map: SegmentationClassMap) -> str:
        lines = []
        for cls in sorted(class_map.classes, key=lambda c: c.index):
            color_hex = "#{:02X}{:02X}{:02X}".format(*cls.color_rgb)
            line = f"{cls.index},{cls.name},{color_hex}"
            if cls.description:
                line += f",{cls.description}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _class_map_from_text(text: str) -> SegmentationClassMap:
        classes: list[SegmentationClass] = []
        for raw in text.splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            classes.append(QtSegmentationMainWindow._parse_class_line(line))
        return SegmentationClassMap(tuple(classes))

    def _selected_class_index(self) -> int:
        label = self.class_combo.currentText().strip()
        if not label:
            return 1
        try:
            return int(label.split(":", 1)[0])
        except Exception:
            return 1

    def _selected_export_formats(self) -> set[str]:
        fmts: set[str] = set()
        if self.chk_fmt_indexed.isChecked():
            fmts.add("indexed_png")
        if self.chk_fmt_color.isChecked():
            fmts.add("color_png")
        if self.chk_fmt_npy.isChecked():
            fmts.add("numpy_npy")
        return fmts or {"indexed_png"}

    def _selected_model_spec(self, model_name: str | None = None) -> dict[str, str]:
        selected = model_name or self.model_combo.currentText()
        spec = self._model_specs.get(selected, {})
        return {str(k): str(v) for k, v in spec.items()}

    def _selected_model_id(self, model_name: str | None = None) -> str:
        return str(self._selected_model_spec(model_name).get("model_id", ""))

    def _toggle_conventional_controls(self, model_name: str) -> None:
        if not hasattr(self, "conventional_row_widget"):
            return
        visible = self._selected_model_id(model_name) == "hydride_conventional"
        self.conventional_row_widget.setVisible(visible)
        self.conventional_row_widget.setEnabled(visible)

    def _collect_conventional_params(self) -> dict[str, object]:
        block_size = int(self.conv_block_spin.value())
        if block_size % 2 == 0:
            block_size += 1
            self.conv_block_spin.setValue(block_size)
        return {
            "clahe": {
                "clip_limit": float(self.conv_clip_spin.value()),
                "tile_grid_size": [int(self.conv_tile_x.value()), int(self.conv_tile_y.value())],
            },
            "adaptive": {
                "block_size": int(block_size),
                "C": int(self.conv_c_spin.value()),
            },
            "morph": {
                "kernel_size": [int(self.conv_kernel_x.value()), int(self.conv_kernel_y.value())],
                "iterations": int(self.conv_iterations_spin.value()),
            },
            "area_threshold": int(self.conv_area_spin.value()),
            "crop": bool(self.conv_crop_check.isChecked()),
            "crop_percent": int(self.conv_crop_percent.value()),
        }

    def _analysis_config_from_ui(self) -> HydrideVisualizationConfig:
        return HydrideVisualizationConfig(
            orientation_bins=int(self.results_orientation_bins.value()),
            size_bins=int(self.results_size_bins.value()),
            min_feature_pixels=int(self.results_min_feature.value()),
            orientation_cmap=self.results_cmap.currentText(),
            size_scale=self.results_size_scale.currentText(),
        )

    def _selected_report_sections(self) -> tuple[str, ...]:
        if not hasattr(self, "report_section_checks"):
            return tuple(REPORT_SECTIONS)
        selected = [name for name, chk in self.report_section_checks.items() if chk.isChecked()]
        return tuple(selected or list(REPORT_SECTIONS))

    def _selected_report_metric_keys(self) -> tuple[str, ...]:
        if not hasattr(self, "report_metric_list"):
            return tuple()
        keys: list[str] = []
        for idx in range(self.report_metric_list.count()):
            item = self.report_metric_list.item(idx)
            if item is None:
                continue
            if item.checkState() == Qt.Checked:
                text = str(item.text()).strip()
                if text:
                    keys.append(text)
        return tuple(keys)

    def _refresh_report_metric_checklist(self, keys: list[str], *, selected: tuple[str, ...] | None = None) -> None:
        if not hasattr(self, "report_metric_list"):
            return
        selected_set = set(selected or self._selected_report_metric_keys())
        self.report_metric_list.blockSignals(True)
        self.report_metric_list.clear()
        for key in sorted({str(k).strip() for k in keys if str(k).strip()}):
            item = QListWidgetItem(key)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEnabled | Qt.ItemIsSelectable)
            item.setCheckState(Qt.Checked if (not selected_set or key in selected_set) else Qt.Unchecked)
            self.report_metric_list.addItem(item)
        self.report_metric_list.blockSignals(False)

    def on_select_all_report_metrics(self) -> None:
        if not hasattr(self, "report_metric_list"):
            return
        for idx in range(self.report_metric_list.count()):
            item = self.report_metric_list.item(idx)
            if item is not None:
                item.setCheckState(Qt.Checked)

    def on_clear_report_metrics(self) -> None:
        if not hasattr(self, "report_metric_list"):
            return
        for idx in range(self.report_metric_list.count()):
            item = self.report_metric_list.item(idx)
            if item is not None:
                item.setCheckState(Qt.Unchecked)

    def on_reset_profile_report_metrics(self) -> None:
        profile = str(self.report_profile_combo.currentText()).strip().lower()
        defaults = tuple(self._ui_config.export_defaults.selected_metric_keys)
        if profile == "full":
            defaults = tuple()
        elif profile == "audit":
            defaults = tuple(
                [
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
                ]
            )
        available = [str(self.report_metric_list.item(i).text()) for i in range(self.report_metric_list.count())]
        if not available and isinstance(self._latest_results_payload, dict):
            pred = self._latest_results_payload.get("predicted_metrics", {})
            corr = self._latest_results_payload.get("corrected_metrics", {})
            keys = sorted(set(pred.keys()) | set(corr.keys())) if isinstance(pred, dict) and isinstance(corr, dict) else []
            self._refresh_report_metric_checklist(keys, selected=defaults)
            return
        self._refresh_report_metric_checklist(available, selected=defaults)

    def _results_export_config_from_ui(self) -> DesktopResultExportConfig:
        cal = self.state.spatial_calibration
        return DesktopResultExportConfig(
            orientation_bins=int(self.results_orientation_bins.value()),
            size_bins=int(self.results_size_bins.value()),
            min_feature_pixels=int(self.results_min_feature.value()),
            orientation_cmap=self.results_cmap.currentText(),
            size_scale=self.results_size_scale.currentText(),
            microns_per_pixel=None if cal is None else float(cal.microns_per_pixel),
            calibration_source="none" if cal is None else cal.source,
            calibration_notes="" if cal is None else cal.notes,
            write_html_report=bool(self.chk_report_html.isChecked()),
            write_pdf_report=bool(self.chk_report_pdf.isChecked()),
            write_csv_report=bool(self.chk_report_csv.isChecked()),
            write_batch_summary=True,
            report_profile=str(self.report_profile_combo.currentText()).strip().lower(),
            selected_metric_keys=self._selected_report_metric_keys(),
            include_sections=self._selected_report_sections(),
            sort_metrics="name",
            top_k_key_metrics=int(self.report_top_k_spin.value()),
            include_artifact_manifest=bool(self.chk_artifact_manifest.isChecked()),
        )

    def _apply_calibration(self, calibration: SpatialCalibration | None, *, image_path: str | None = None) -> None:
        self.state.spatial_calibration = calibration
        if image_path is not None:
            self.state.calibration_image_path = str(image_path)
        self._update_calibration_status_label()
        self._queue_results_refresh()

    def _update_calibration_status_label(self) -> None:
        if not hasattr(self, "calibration_status_label"):
            return
        cal = self.state.spatial_calibration
        if cal is None:
            self.calibration_status_label.setText("Scale: pixels (no calibration)")
            return
        details = f"{float(cal.microns_per_pixel):.6g} um/px ({cal.source})"
        if cal.notes:
            details += f" | {cal.notes}"
        self.calibration_status_label.setText(f"Scale: {details}")

    def _try_auto_calibration_from_metadata(self, image_path: str, *, user_initiated: bool = False) -> None:
        if (
            self.state.spatial_calibration is not None
            and self.state.spatial_calibration.source == "manual_line"
            and str(self.state.calibration_image_path or "") == str(image_path)
            and not user_initiated
        ):
            self._update_calibration_status_label()
            return
        if (
            self.state.spatial_calibration is not None
            and self.state.spatial_calibration.source == "manual_line"
            and self.state.calibration_image_path
            and str(self.state.calibration_image_path) != str(image_path)
        ):
            self.state.spatial_calibration = None
        cal = metadata_calibration_from_image(image_path)
        if cal is None:
            if user_initiated:
                QMessageBox.information(
                    self,
                    "Metadata Scale",
                    "No usable micron-per-pixel metadata was found. Scale remains in pixels.",
                )
            if self.state.spatial_calibration is None or self.state.spatial_calibration.source != "manual_line":
                self._apply_calibration(None, image_path=image_path)
            return
        self._apply_calibration(cal, image_path=image_path)
        self.logger.info(
            "Loaded metadata calibration for %s: %.6g um/px (%s)",
            image_path,
            cal.microns_per_pixel,
            cal.source,
        )
        if user_initiated:
            QMessageBox.information(
                self,
                "Metadata Scale",
                f"Detected scale: {cal.microns_per_pixel:.6g} um/px\nSource: {cal.source}",
            )

    def _queue_results_refresh(self, *_args) -> None:
        self._results_dirty = True
        self._results_refresh_timer.start(250)

    def _update_results_dashboard(self) -> None:
        run = self.state.current_run
        if run is None:
            self.results_summary_label.setText("Results: run segmentation to populate dashboard")
            self.results_table.setRowCount(0)
            return
        sess = self.state.correction_session
        pred_mask = to_index_mask(np.array(run.mask_image))
        corr_mask = to_index_mask(sess.current_mask) if sess is not None else pred_mask
        cfg = self._analysis_config_from_ui()
        cal = self.state.spatial_calibration
        um_per_px = None if cal is None else float(cal.microns_per_pixel)
        try:
            pred_stats = compute_hydride_statistics(
                pred_mask,
                orientation_bins=cfg.orientation_bins,
                size_bins=cfg.size_bins,
                min_feature_pixels=cfg.min_feature_pixels,
                microns_per_pixel=um_per_px,
            )
            corr_stats = compute_hydride_statistics(
                corr_mask,
                orientation_bins=cfg.orientation_bins,
                size_bins=cfg.size_bins,
                min_feature_pixels=cfg.min_feature_pixels,
                microns_per_pixel=um_per_px,
            )
            pred_visuals = render_hydride_visualizations(pred_stats, cfg)
            corr_visuals = render_hydride_visualizations(corr_stats, cfg)

            self._set_image_preview(self.results_pred_orientation_view, pred_visuals["orientation_map_rgb"])
            self._set_image_preview(self.results_pred_size_view, pred_visuals["size_distribution_rgb"])
            self._set_image_preview(self.results_pred_angle_view, pred_visuals["orientation_distribution_rgb"])
            self._set_image_preview(self.results_corr_orientation_view, corr_visuals["orientation_map_rgb"])
            self._set_image_preview(self.results_corr_size_view, corr_visuals["size_distribution_rgb"])
            self._set_image_preview(self.results_corr_angle_view, corr_visuals["orientation_distribution_rgb"])

            pred_metrics = dict(pred_stats.scalar_metrics)
            corr_metrics = dict(corr_stats.scalar_metrics)
            preferred = [
                "hydride_area_fraction_percent",
                "hydride_count",
                "hydride_total_area_um2" if um_per_px is not None else "hydride_total_area_pixels",
                "equivalent_diameter_mean_um" if um_per_px is not None else "equivalent_diameter_mean_px",
                "hydride_density_per_megapixel",
                "size_mean_um2" if um_per_px is not None else "size_mean_pixels",
                "size_p90_um2" if um_per_px is not None else "size_p90_pixels",
                "orientation_mean_deg",
                "orientation_std_deg",
                "orientation_alignment_index",
                "orientation_entropy_bits",
                "excluded_small_features",
            ]
            metric_keys = [k for k in preferred if k in pred_metrics or k in corr_metrics]
            extra_keys = sorted((set(pred_metrics.keys()) | set(corr_metrics.keys())) - set(metric_keys))
            metric_keys.extend(extra_keys)
            selected_defaults = (
                self._selected_report_metric_keys()
                if hasattr(self, "report_metric_list") and self.report_metric_list.count() > 0
                else tuple(self._ui_config.export_defaults.selected_metric_keys)
            )
            self._refresh_report_metric_checklist(metric_keys, selected=selected_defaults)
            self.results_table.setRowCount(len(metric_keys))
            for r, key in enumerate(metric_keys):
                self.results_table.setItem(r, 0, QTableWidgetItem(str(key)))
                self.results_table.setItem(r, 1, QTableWidgetItem(_fmt_metric(pred_metrics.get(key, ""))))
                self.results_table.setItem(r, 2, QTableWidgetItem(_fmt_metric(corr_metrics.get(key, ""))))

            self.results_summary_label.setText(
                "Results | predicted area={:.3f}% count={} | corrected area={:.3f}% count={} | bins(o={}, s={}) | units={}".format(
                    float(pred_metrics.get("hydride_area_fraction_percent", 0.0)),
                    int(pred_metrics.get("hydride_count", 0)),
                    float(corr_metrics.get("hydride_area_fraction_percent", 0.0)),
                    int(corr_metrics.get("hydride_count", 0)),
                    int(cfg.orientation_bins),
                    int(cfg.size_bins),
                    "um (calibrated)" if um_per_px is not None else "pixels",
                )
            )
            self._latest_results_payload = {
                "predicted_metrics": pred_metrics,
                "corrected_metrics": corr_metrics,
                "analysis_config": {
                    "orientation_bins": cfg.orientation_bins,
                    "size_bins": cfg.size_bins,
                    "min_feature_pixels": cfg.min_feature_pixels,
                    "orientation_cmap": cfg.orientation_cmap,
                    "size_scale": cfg.size_scale,
                },
                "spatial_calibration": None if cal is None else cal.as_dict(),
            }
            self._results_dirty = False
        except Exception as exc:
            self.logger.exception("Results dashboard update failed")
            self.results_summary_label.setText(f"Results update failed: {exc}")

    def _refresh_sample_picker(self) -> None:
        if not hasattr(self, "sample_combo"):
            return
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        for path in self._sample_images:
            rel = path
            try:
                rel = path.relative_to(self.orchestrator.repo_root)
            except Exception:
                rel = path
            self.sample_combo.addItem(str(rel), str(path))
        self.sample_combo.blockSignals(False)
        self._populate_sample_menu()

    def _populate_sample_menu(self) -> None:
        if not hasattr(self, "_sample_menu"):
            return
        self._sample_menu.clear()
        if not self._sample_images:
            disabled = QAction("(no sample images found)", self)
            disabled.setEnabled(False)
            self._sample_menu.addAction(disabled)
            return
        for sample_path in self._sample_images:
            label = sample_path.name
            action = QAction(label, self)
            action.triggered.connect(lambda checked=False, p=sample_path: self._load_sample_path(p))
            self._sample_menu.addAction(action)

    def _load_sample_path(self, sample_path: Path) -> None:
        if not sample_path.exists():
            QMessageBox.warning(self, "Missing sample", f"Sample path not found:\n{sample_path}")
            return
        self.path_edit.setText(str(sample_path))
        self.orch_infer_image_edit.setText(str(sample_path))
        self.state.image_path = str(sample_path)
        self._preview_input_image(str(sample_path))
        self._try_auto_calibration_from_metadata(str(sample_path), user_initiated=False)
        self.tabs.setCurrentIndex(0)
        self.logger.info("Loaded sample image: %s", sample_path)

    def on_load_sample(self) -> None:
        if self.sample_combo.count() == 0:
            QMessageBox.warning(self, "No samples", "No sample images are available in data/sample_images or test_data.")
            return
        sample_raw = self.sample_combo.currentData()
        if not sample_raw:
            QMessageBox.warning(self, "No sample", "Select a sample image first.")
            return
        sample_path = Path(str(sample_raw))
        self._load_sample_path(sample_path)

    def _on_model_changed(self, model_name: str) -> None:
        spec = self._model_specs.get(model_name)
        self._toggle_conventional_controls(model_name)
        if not spec:
            self.model_desc.setText("")
            return
        lines = [
            f"<b>{spec['display_name']}</b> | {spec.get('description', '')}",
            spec.get("details", ""),
        ]
        if spec.get("model_nickname"):
            lines.append(
                f"<b>Frozen model:</b> {spec.get('model_nickname')} | {spec.get('framework', '')} | "
                f"{spec.get('model_type', '')} | input={spec.get('input_dimensions', '')}"
            )
        if spec.get("checkpoint_path_hint"):
            lines.append(f"<b>Checkpoint hint:</b> {spec.get('checkpoint_path_hint')}")
        if spec.get("artifact_stage"):
            lines.append(f"<b>Stage:</b> {spec.get('artifact_stage')}")
        if spec.get("application_remarks"):
            lines.append(f"<b>Application:</b> {spec.get('application_remarks')}")
        if spec.get("short_description"):
            lines.append(f"<b>User tip:</b> {spec.get('short_description')}")
        if spec.get("quality_report_path") and str(spec.get("quality_report_path")).lower() != "n/a":
            lines.append(f"<b>Quality report:</b> {spec.get('quality_report_path')}")
        self.model_desc.setText("<br>".join([line for line in lines if line]))
        self.logger.info("Selected model: %s (%s)", model_name, spec.get("model_id", ""))

    def _on_class_changed(self, class_label: str) -> None:
        class_index = self._selected_class_index()
        self.corrected_canvas.set_class_index(class_index)
        self.logger.info("Active class index: %s (%s)", class_index, class_label)

    def on_pick_config(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select YAML config", "", "YAML (*.yml *.yaml)")
        if path:
            self.config_path_edit.setText(path)

    @staticmethod
    def _parse_override_text(raw: str) -> list[str]:
        txt = raw.strip()
        if not txt:
            return []
        return [part.strip() for part in txt.split(",") if part.strip()]

    @staticmethod
    def _parse_json_mapping_text(raw: str) -> dict[str, object]:
        text = raw.strip()
        if not text:
            return {}
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("Mask colormap must be a JSON object")
        return {str(k): v for k, v in payload.items()}

    def _build_dataset_prepare_config(
        self,
        *,
        dataset_dir_override: str | None = None,
        output_dir_override: str | None = None,
    ) -> DatasetPrepareConfig:
        cfg_path = self.orch_prepare_config_edit.text().strip() or None
        cfg_overrides = self._parse_override_text(self.orch_prepare_set_edit.text())
        cfg = resolve_config(cfg_path, cfg_overrides)
        dataset_dir = dataset_dir_override or self.orch_prepare_dataset_edit.text().strip() or str(cfg.get("dataset_dir", ""))
        output_dir = output_dir_override or self.orch_prepare_output_edit.text().strip() or str(cfg.get("output_dir", ""))
        if not dataset_dir or not output_dir:
            raise ValueError("Dataset Dir and Prepared Output Dir are required for dataset preparation")

        mask_colormap = dict(cfg.get("mask_colormap", {}))
        if self.orch_prepare_colormap.toPlainText().strip():
            mask_colormap = self._parse_json_mapping_text(self.orch_prepare_colormap.toPlainText())

        return DatasetPrepareConfig(
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            train_ratio=float(self.orch_prepare_train_ratio.value()),
            val_ratio=float(self.orch_prepare_val_ratio.value()),
            test_ratio=float(self.orch_prepare_test_ratio.value()),
            seed=int(self.orch_prepare_seed.value()),
            id_width=int(self.orch_prepare_id_width.value()),
            split_strategy=str(self.orch_prepare_strategy.currentText()),
            leakage_group_mode=str(self.orch_prepare_group_mode.currentText()),
            leakage_group_regex=self.orch_prepare_group_regex.text().strip(),
            mask_input_type=str(self.orch_prepare_mask_type.currentText()),
            mask_colormap=mask_colormap,
            mask_colormap_strict=bool(self.orch_prepare_colormap_strict.isChecked()),
        )

    def _dataset_prepare_overrides(self, config: DatasetPrepareConfig) -> list[str]:
        overrides = self._parse_override_text(self.orch_prepare_set_edit.text())
        overrides.extend(
            [
                f"split_train_ratio={float(config.train_ratio)}",
                f"split_val_ratio={float(config.val_ratio)}",
                f"split_test_ratio={float(config.test_ratio)}",
                f"split_seed={int(config.seed)}",
                f"split_id_width={int(config.id_width)}",
                f"split_strategy={config.split_strategy}",
                f"leakage_group_mode={config.leakage_group_mode}",
                f"leakage_group_regex={config.leakage_group_regex}",
                f"mask_input_type={config.mask_input_type}",
                f"mask_colormap_strict={str(bool(config.mask_colormap_strict)).lower()}",
            ]
        )
        if config.mask_colormap:
            overrides.append(f"mask_colormap={json.dumps(config.mask_colormap, separators=(',', ':'))}")
        return overrides

    def _set_dataset_preview_payload(self, rows: list[dict[str, object]], split_counts: dict[str, int], summary: str) -> None:
        self._dataset_preview_rows = rows
        self._dataset_preview_split_counts = split_counts
        self.dataset_preview_summary.setText(summary)
        self._refresh_dataset_preview_table()

    def _refresh_dataset_preview_table(self, *_args) -> None:
        query = self.dataset_preview_filter.text().strip().lower()
        rows = self._dataset_preview_rows
        if query:
            rows = [
                row
                for row in rows
                if query in " ".join(
                    [
                        str(row.get("global_id", "")),
                        str(row.get("split", "")),
                        str(row.get("source_group", "")),
                        str(row.get("original_stem", "")),
                        str(row.get("new_name", "")),
                        str(row.get("original_image_path", "")),
                        str(row.get("original_mask_path", "")),
                    ]
                ).lower()
            ]
        self.dataset_preview_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            vals = [
                str(row.get("global_id", row.get("id", ""))),
                str(row.get("split", "")),
                str(row.get("source_group", "")),
                str(row.get("original_stem", "")),
                str(row.get("new_name", "")),
                str(row.get("original_image_path", "")),
                str(row.get("original_mask_path", "")),
            ]
            for c, value in enumerate(vals):
                self.dataset_preview_table.setItem(r, c, QTableWidgetItem(value))

    def _run_dataset_qa(self, dataset_dir: str) -> tuple[bool, str]:
        qa_cfg_path = self.orch_qa_config_edit.text().strip() or None
        qa_cfg = resolve_config(qa_cfg_path, None)
        output_path = self.orch_qa_output_edit.text().strip() or str(qa_cfg.get("output_path", "outputs/dataops/dataset_qa_report.json"))
        imbalance_warn = float(self.orch_qa_imbalance_warn.value())
        strict = bool(self.orch_qa_strict.isChecked())
        report = run_dataset_quality_checks(
            DatasetQualityConfig(
                dataset_dir=dataset_dir,
                output_path=output_path,
                imbalance_ratio_warn=imbalance_warn,
                strict=False,
            )
        )
        self._last_dataset_qa_ok = bool(report.ok)
        self._last_dataset_qa_dir = str(dataset_dir)
        self.workflow_notes.append(
            f"[Dataset QA] ok={report.ok} errors={len(report.errors)} warnings={len(report.warnings)} report={output_path}"
        )
        if report.errors:
            self.workflow_notes.append("[Dataset QA] critical errors:\n- " + "\n- ".join(report.errors[:8]))
        if report.warnings:
            self.workflow_notes.append("[Dataset QA] warnings:\n- " + "\n- ".join(report.warnings[:8]))
        if strict and not report.ok:
            return False, output_path
        return True, output_path

    def _config_overrides(self) -> list[str]:
        return self._parse_override_text(self.config_overrides_edit.text())

    def _resolve_run_config(self) -> dict:
        cfg_path = self.config_path_edit.text().strip() or None
        return resolve_config(cfg_path, self._config_overrides())

    def _start_orchestration_job(self, command: list[str], job_name: str) -> None:
        if self._job_process is not None and self._job_process.state() != QProcess.NotRunning:
            QMessageBox.warning(self, "Job Running", "Another orchestration job is already running.")
            return

        self._job_name = job_name
        self.workflow_notes.append(f"$ {' '.join(command)}")
        self.logger.info("Starting %s job", job_name)

        proc = QProcess(self)
        proc.setProgram(command[0])
        proc.setArguments(command[1:])
        proc.setWorkingDirectory(str(self.orchestrator.repo_root))
        proc.setProcessChannelMode(QProcess.MergedChannels)
        proc.readyReadStandardOutput.connect(self._on_orchestration_output)
        proc.finished.connect(self._on_orchestration_finished)
        proc.errorOccurred.connect(self._on_orchestration_error)
        self._job_process = proc
        proc.start()

    def _on_orchestration_output(self) -> None:
        if self._job_process is None:
            return
        text = bytes(self._job_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if text:
            self.workflow_notes.append(text.rstrip("\n"))

    def _on_orchestration_error(self, error) -> None:  # noqa: ANN001
        _ = error
        if self._job_process is None:
            return
        self.logger.error("Orchestration process error: %s", self._job_process.errorString())
        self.workflow_notes.append(f"[ERROR] {self._job_process.errorString()}")

    def _on_orchestration_finished(self, exit_code: int, exit_status) -> None:  # noqa: ANN001
        status_label = "normal" if exit_status == QProcess.NormalExit else "crash"
        self.logger.info("%s job finished (code=%s, status=%s)", self._job_name, exit_code, status_label)
        self.workflow_notes.append(
            f"[{self._job_name}] finished with exit_code={exit_code}, status={status_label}"
        )
        if exit_code == 0 and exit_status == QProcess.NormalExit:
            QMessageBox.information(self, "Job Complete", f"{self._job_name} job completed successfully.")
        else:
            QMessageBox.critical(
                self,
                "Job Failed",
                f"{self._job_name} job failed (exit_code={exit_code}, status={status_label}).",
            )
        self._job_process = None

    def on_preview_dataset_prepare(self) -> None:
        try:
            config = self._build_dataset_prepare_config()
            preview = preview_training_dataset_layout(config)
            rows = list(preview.mapping)
            split_counts = preview.split_counts
            summary = (
                f"Preview | layout={preview.source_layout} | pairs={preview.total_pairs} | "
                f"split={split_counts} | leakage_groups={preview.leakage_groups} | "
                f"class_hist={preview.class_histogram}"
            )
            self._set_dataset_preview_payload(rows, split_counts, summary)
            self.workflow_notes.append(f"[Dataset Preview] {summary}")
        except Exception as exc:
            self.logger.exception("Dataset preview failed")
            QMessageBox.critical(self, "Dataset Preview Error", str(exc))

    def on_orchestrate_dataset_prepare(self) -> None:
        try:
            config = self._build_dataset_prepare_config()
        except Exception as exc:
            QMessageBox.critical(self, "Dataset Prepare Config Error", str(exc))
            return
        command = self.orchestrator.dataset_prepare(
            config=self.orch_prepare_config_edit.text().strip() or None,
            overrides=self._dataset_prepare_overrides(config),
            dataset_dir=config.dataset_dir,
            output_dir=config.output_dir,
        )
        self._start_orchestration_job(command, "DatasetPrepare")

    def on_run_dataset_qa(self) -> None:
        dataset_dir = self.orch_prepare_output_edit.text().strip() or self.orch_prepare_dataset_edit.text().strip()
        if not dataset_dir:
            QMessageBox.warning(self, "Missing path", "Set Prepared Output Dir or Dataset Dir for QA.")
            return
        try:
            ok, report_path = self._run_dataset_qa(dataset_dir)
            if ok:
                QMessageBox.information(self, "Dataset QA", f"QA passed for:\n{dataset_dir}\n\nReport:\n{report_path}")
            else:
                QMessageBox.critical(self, "Dataset QA Failed", f"QA failed for:\n{dataset_dir}\n\nReport:\n{report_path}")
        except Exception as exc:
            self.logger.exception("Dataset QA failed")
            QMessageBox.critical(self, "Dataset QA Error", str(exc))

    def on_apply_prepared_dataset_to_training(self) -> None:
        target = self.orch_prepare_output_edit.text().strip()
        if not target:
            QMessageBox.warning(self, "Missing path", "Set Prepared Output Dir first.")
            return
        self.orch_train_dataset_edit.setText(target)
        self.workflow_notes.append(f"[Workflow] Training dataset path set to prepared output: {target}")
        QMessageBox.information(self, "Training Dataset Updated", f"Training Dataset Dir set to:\n{target}")

    def _preflight_training_dataset_gate(self, dataset_dir: str, output_dir: str) -> tuple[str, list[str]] | None:
        if not self.orch_train_require_qa.isChecked():
            return dataset_dir, []
        prep_out = self.orch_prepare_output_edit.text().strip() or str(Path(output_dir or "outputs/training") / "prepared_dataset")
        try:
            prep_cfg = self._build_dataset_prepare_config(dataset_dir_override=dataset_dir, output_dir_override=prep_out)
            prepared = prepare_training_dataset_layout(prep_cfg)
            qa_dataset_dir = str(prepared.dataset_dir)
            ok, report_path = self._run_dataset_qa(qa_dataset_dir)
            if not ok:
                QMessageBox.critical(
                    self,
                    "Training Blocked (Dataset QA)",
                    f"Dataset QA failed; training launch is blocked.\n\nDataset: {qa_dataset_dir}\nReport: {report_path}",
                )
                return None
            train_cfg = self.orch_train_config_edit.text().strip()
            if train_cfg:
                preflight_report = Path(output_dir or "outputs/training") / "preflight_train_gate.json"
                train_overrides = self._parse_override_text(self.orch_train_set_edit.text())
                preflight = run_preflight(
                    PreflightConfig(
                        mode="train",
                        dataset_dir=qa_dataset_dir,
                        train_config=train_cfg,
                        train_overrides=tuple(train_overrides),
                        require_dataset_qa=False,
                        output_path=str(preflight_report),
                    )
                )
                if not preflight.ok:
                    blocking = [
                        f"{issue.code}: {issue.message}"
                        for issue in preflight.issues
                        if issue.severity == "error"
                    ]
                    body = "\n".join(blocking[:5]) if blocking else "Preflight reported errors."
                    QMessageBox.critical(
                        self,
                        "Training Blocked (Preflight)",
                        "Training preflight failed after dataset QA.\n\n"
                        f"Dataset: {qa_dataset_dir}\n"
                        f"Preflight report: {preflight_report}\n\n"
                        f"{body}",
                    )
                    return None
            self.workflow_notes.append(
                f"[Training Gate] QA passed | dataset={qa_dataset_dir} | used_existing_splits={prepared.used_existing_splits}"
            )
            return qa_dataset_dir, ["auto_prepare_dataset=false"]
        except Exception as exc:
            self.logger.exception("Training dataset preflight failed")
            QMessageBox.critical(self, "Training Dataset Gate Error", str(exc))
            return None

    @staticmethod
    def _report_summary_text(summary) -> str:
        lines = [
            f"path: {summary.path}",
            f"kind: {summary.report_kind}",
            f"schema: {summary.schema_version}",
            f"backend: {summary.backend}",
            f"status: {summary.status}",
            f"runtime: {summary.runtime_human or summary.runtime_seconds}",
            f"device: {summary.device}",
            f"config_sha256: {summary.config_sha256}",
            f"samples_evaluated: {summary.samples_evaluated}",
            f"tracked_samples: {summary.tracked_samples}",
            f"html_report_path: {summary.html_report_path}",
            "",
            "metrics:",
        ]
        if summary.metrics:
            for key in sorted(summary.metrics.keys()):
                lines.append(f"  - {key}: {summary.metrics[key]:.6f}")
        else:
            lines.append("  - (none)")
        return "\n".join(lines)

    def on_pick_review_report_a(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select baseline report JSON", "", "JSON (*.json)")
        if path:
            self.review_report_a_edit.setText(path)

    def on_pick_review_report_b(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select candidate report JSON", "", "JSON (*.json)")
        if path:
            self.review_report_b_edit.setText(path)

    def on_load_review_report_a(self) -> None:
        path = self.review_report_a_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing report", "Set Baseline Report path.")
            return
        try:
            summary = summarize_run_report(path)
            self._review_summary_a = summary
            self.review_summary_a_text.setPlainText(self._report_summary_text(summary))
            self.workflow_notes.append(f"[Run Review] Loaded baseline summary: {path}")
        except Exception as exc:
            self.logger.exception("Failed to load baseline report summary")
            QMessageBox.critical(self, "Run Review Error", str(exc))

    def on_load_review_report_b(self) -> None:
        path = self.review_report_b_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing report", "Set Candidate Report path.")
            return
        try:
            summary = summarize_run_report(path)
            self._review_summary_b = summary
            self.review_summary_b_text.setPlainText(self._report_summary_text(summary))
            self.workflow_notes.append(f"[Run Review] Loaded candidate summary: {path}")
        except Exception as exc:
            self.logger.exception("Failed to load candidate report summary")
            QMessageBox.critical(self, "Run Review Error", str(exc))

    def on_compare_review_reports(self) -> None:
        if self._review_summary_a is None:
            self.on_load_review_report_a()
        if self._review_summary_b is None:
            self.on_load_review_report_b()
        if self._review_summary_a is None or self._review_summary_b is None:
            return
        comparison = compare_run_reports(self._review_summary_a, self._review_summary_b)
        rows = comparison.get("rows", [])
        self.review_compare_table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            metric = str(row.get("metric", ""))
            baseline = row.get("baseline")
            candidate = row.get("candidate")
            delta = row.get("delta")
            delta_pct = row.get("delta_pct")
            vals = [
                metric,
                "" if baseline is None else f"{float(baseline):.6f}",
                "" if candidate is None else f"{float(candidate):.6f}",
                "" if delta is None else f"{float(delta):+.6f}",
                "" if delta_pct is None else f"{float(delta_pct):+.2f}%",
            ]
            for c, value in enumerate(vals):
                self.review_compare_table.setItem(r, c, QTableWidgetItem(value))

        self.review_compare_meta.setText(
            "Comparison | same_kind={} same_schema={} same_backend={} same_config={}".format(
                comparison.get("same_kind"),
                comparison.get("same_schema"),
                comparison.get("same_backend"),
                comparison.get("same_config_sha256"),
            )
        )
        self.workflow_notes.append(
            "[Run Review] Compared reports | same_kind={} same_schema={} metrics={}".format(
                comparison.get("same_kind"),
                comparison.get("same_schema"),
                len(rows),
            )
        )

    def _collect_workflow_profile(self, scope: str) -> dict[str, object]:
        if scope == "dataset_prepare":
            return {
                "prepare_config": self.orch_prepare_config_edit.text().strip(),
                "prepare_overrides": self.orch_prepare_set_edit.text().strip(),
                "dataset_dir": self.orch_prepare_dataset_edit.text().strip(),
                "output_dir": self.orch_prepare_output_edit.text().strip(),
                "split_train_ratio": float(self.orch_prepare_train_ratio.value()),
                "split_val_ratio": float(self.orch_prepare_val_ratio.value()),
                "split_test_ratio": float(self.orch_prepare_test_ratio.value()),
                "split_seed": int(self.orch_prepare_seed.value()),
                "split_id_width": int(self.orch_prepare_id_width.value()),
                "split_strategy": self.orch_prepare_strategy.currentText(),
                "leakage_group_mode": self.orch_prepare_group_mode.currentText(),
                "leakage_group_regex": self.orch_prepare_group_regex.text().strip(),
                "mask_input_type": self.orch_prepare_mask_type.currentText(),
                "mask_colormap_json": self.orch_prepare_colormap.toPlainText().strip(),
                "mask_colormap_strict": bool(self.orch_prepare_colormap_strict.isChecked()),
                "qa_config": self.orch_qa_config_edit.text().strip(),
                "qa_output": self.orch_qa_output_edit.text().strip(),
                "qa_imbalance_warn": float(self.orch_qa_imbalance_warn.value()),
                "qa_strict": bool(self.orch_qa_strict.isChecked()),
            }
        if scope == "training":
            return {
                "config": self.orch_train_config_edit.text().strip(),
                "overrides": self.orch_train_set_edit.text().strip(),
                "backend": self.orch_train_backend.currentText(),
                "dataset_dir": self.orch_train_dataset_edit.text().strip(),
                "output_dir": self.orch_train_output_edit.text().strip(),
                "enable_gpu": bool(self.orch_train_enable_gpu.isChecked()),
                "device_policy": self.orch_train_device_policy.currentText(),
                "max_samples": int(self.orch_train_max_samples.value()),
                "epochs": int(self.orch_train_epochs.value()),
                "batch_size": int(self.orch_train_batch_size.value()),
                "learning_rate": float(self.orch_train_learning_rate.value()),
                "weight_decay": float(self.orch_train_weight_decay.value()),
                "patience": int(self.orch_train_patience.value()),
                "min_delta": float(self.orch_train_min_delta.value()),
                "checkpoint_every": int(self.orch_train_checkpoint_every.value()),
                "resume_checkpoint": self.orch_train_resume_checkpoint.text().strip(),
                "val_tracking_samples": int(self.orch_train_val_tracking_samples.value()),
                "val_tracking_fixed": self.orch_train_val_tracking_fixed.text().strip(),
                "val_tracking_seed": int(self.orch_train_val_tracking_seed.value()),
                "write_html_report": bool(self.orch_train_write_html_report.isChecked()),
                "progress_interval": int(self.orch_train_progress_interval.value()),
                "seed": int(self.orch_train_seed.value()),
                "require_qa": bool(self.orch_train_require_qa.isChecked()),
            }
        if scope == "evaluation":
            return {
                "config": self.orch_eval_config_edit.text().strip(),
                "overrides": self.orch_eval_set_edit.text().strip(),
                "dataset_dir": self.orch_eval_dataset_edit.text().strip(),
                "model_path": self.orch_eval_model_edit.text().strip(),
                "enable_gpu": bool(self.orch_eval_enable_gpu.isChecked()),
                "device_policy": self.orch_eval_device_policy.currentText(),
                "split": self.orch_eval_split_combo.currentText(),
                "output_path": self.orch_eval_output_edit.text().strip(),
                "tracking_samples": int(self.orch_eval_tracking_samples.value()),
                "tracking_seed": int(self.orch_eval_tracking_seed.value()),
                "write_html_report": bool(self.orch_eval_write_html_report.isChecked()),
            }
        if scope == "hpc_ga":
            return {
                "config": self.orch_hpc_config_edit.text().strip(),
                "overrides": self.orch_hpc_set_edit.text().strip(),
                "dataset_dir": self.orch_hpc_dataset_edit.text().strip(),
                "output_dir": self.orch_hpc_output_edit.text().strip(),
                "experiment_name": self.orch_hpc_experiment_name.text().strip(),
                "scheduler": self.orch_hpc_scheduler_combo.currentText(),
                "run_mode": self.orch_hpc_run_mode_combo.currentText(),
                "architectures": self.orch_hpc_architectures_edit.text().strip(),
                "num_candidates": int(self.orch_hpc_num_candidates.value()),
                "population_size": int(self.orch_hpc_population.value()),
                "generations": int(self.orch_hpc_generations.value()),
                "mutation_rate": float(self.orch_hpc_mutation.value()),
                "crossover_rate": float(self.orch_hpc_crossover.value()),
                "seed": int(self.orch_hpc_seed.value()),
                "learning_rate_min": float(self.orch_hpc_lr_min.value()),
                "learning_rate_max": float(self.orch_hpc_lr_max.value()),
                "batch_size_choices": self.orch_hpc_batch_sizes.text().strip(),
                "epochs_min": int(self.orch_hpc_epochs_min.value()),
                "epochs_max": int(self.orch_hpc_epochs_max.value()),
                "weight_decay_min": float(self.orch_hpc_wd_min.value()),
                "weight_decay_max": float(self.orch_hpc_wd_max.value()),
                "max_samples_min": int(self.orch_hpc_ms_min.value()),
                "max_samples_max": int(self.orch_hpc_ms_max.value()),
                "fitness_mode": self.orch_hpc_fitness_mode.currentText(),
                "feedback_sources": self.orch_hpc_feedback_sources.text().strip(),
                "feedback_min_samples": int(self.orch_hpc_feedback_min_samples.value()),
                "feedback_k": int(self.orch_hpc_feedback_k.value()),
                "exploration_weight": float(self.orch_hpc_exploration_weight.value()),
                "fitness_weight_mean_iou": float(self.orch_hpc_w_iou.value()),
                "fitness_weight_macro_f1": float(self.orch_hpc_w_f1.value()),
                "fitness_weight_pixel_accuracy": float(self.orch_hpc_w_acc.value()),
                "fitness_weight_runtime": float(self.orch_hpc_w_runtime.value()),
                "enable_gpu": bool(self.orch_hpc_enable_gpu.isChecked()),
                "device_policy": self.orch_hpc_device_policy.currentText(),
                "queue": self.orch_hpc_queue.text().strip(),
                "account": self.orch_hpc_account.text().strip(),
                "qos": self.orch_hpc_qos.text().strip(),
                "gpus_per_job": int(self.orch_hpc_gpus.value()),
                "cpus_per_task": int(self.orch_hpc_cpus.value()),
                "mem_gb": int(self.orch_hpc_mem.value()),
                "time_limit": self.orch_hpc_time.text().strip(),
                "job_prefix": self.orch_hpc_job_prefix.text().strip(),
                "python_executable": self.orch_hpc_python.text().strip(),
                "microseg_cli_path": self.orch_hpc_cli_path.text().strip(),
                "base_train_config": self.orch_hpc_base_train.text().strip(),
                "base_eval_config": self.orch_hpc_base_eval.text().strip(),
                "eval_split": self.orch_hpc_eval_split.currentText(),
                "feedback_top_k": int(self.orch_hpc_feedback_top_k.value()),
                "feedback_report_output": self.orch_hpc_feedback_report_output.text().strip(),
            }
        raise ValueError(f"Unsupported profile scope: {scope}")

    def _apply_workflow_profile(self, scope: str, values: dict[str, object]) -> None:
        if scope == "dataset_prepare":
            self.orch_prepare_config_edit.setText(str(values.get("prepare_config", self.orch_prepare_config_edit.text())))
            self.orch_prepare_set_edit.setText(str(values.get("prepare_overrides", self.orch_prepare_set_edit.text())))
            self.orch_prepare_dataset_edit.setText(str(values.get("dataset_dir", self.orch_prepare_dataset_edit.text())))
            self.orch_prepare_output_edit.setText(str(values.get("output_dir", self.orch_prepare_output_edit.text())))
            self.orch_prepare_train_ratio.setValue(float(values.get("split_train_ratio", self.orch_prepare_train_ratio.value())))
            self.orch_prepare_val_ratio.setValue(float(values.get("split_val_ratio", self.orch_prepare_val_ratio.value())))
            self.orch_prepare_test_ratio.setValue(float(values.get("split_test_ratio", self.orch_prepare_test_ratio.value())))
            self.orch_prepare_seed.setValue(int(values.get("split_seed", self.orch_prepare_seed.value())))
            self.orch_prepare_id_width.setValue(int(values.get("split_id_width", self.orch_prepare_id_width.value())))
            self.orch_prepare_strategy.setCurrentText(str(values.get("split_strategy", self.orch_prepare_strategy.currentText())))
            self.orch_prepare_group_mode.setCurrentText(str(values.get("leakage_group_mode", self.orch_prepare_group_mode.currentText())))
            self.orch_prepare_group_regex.setText(str(values.get("leakage_group_regex", self.orch_prepare_group_regex.text())))
            self.orch_prepare_mask_type.setCurrentText(str(values.get("mask_input_type", self.orch_prepare_mask_type.currentText())))
            self.orch_prepare_colormap.setPlainText(str(values.get("mask_colormap_json", self.orch_prepare_colormap.toPlainText())))
            self.orch_prepare_colormap_strict.setChecked(bool(values.get("mask_colormap_strict", self.orch_prepare_colormap_strict.isChecked())))
            self.orch_qa_config_edit.setText(str(values.get("qa_config", self.orch_qa_config_edit.text())))
            self.orch_qa_output_edit.setText(str(values.get("qa_output", self.orch_qa_output_edit.text())))
            self.orch_qa_imbalance_warn.setValue(float(values.get("qa_imbalance_warn", self.orch_qa_imbalance_warn.value())))
            self.orch_qa_strict.setChecked(bool(values.get("qa_strict", self.orch_qa_strict.isChecked())))
            self.workflow_tabs.setCurrentIndex(self.workflow_tabs.indexOf(self.workflow_prep_tab))
            return
        if scope == "training":
            self.orch_train_config_edit.setText(str(values.get("config", self.orch_train_config_edit.text())))
            self.orch_train_set_edit.setText(str(values.get("overrides", self.orch_train_set_edit.text())))
            self.orch_train_backend.setCurrentText(str(values.get("backend", self.orch_train_backend.currentText())))
            self.orch_train_dataset_edit.setText(str(values.get("dataset_dir", self.orch_train_dataset_edit.text())))
            self.orch_train_output_edit.setText(str(values.get("output_dir", self.orch_train_output_edit.text())))
            self.orch_train_enable_gpu.setChecked(bool(values.get("enable_gpu", self.orch_train_enable_gpu.isChecked())))
            self.orch_train_device_policy.setCurrentText(str(values.get("device_policy", self.orch_train_device_policy.currentText())))
            self.orch_train_max_samples.setValue(int(values.get("max_samples", self.orch_train_max_samples.value())))
            self.orch_train_epochs.setValue(int(values.get("epochs", self.orch_train_epochs.value())))
            self.orch_train_batch_size.setValue(int(values.get("batch_size", self.orch_train_batch_size.value())))
            self.orch_train_learning_rate.setValue(float(values.get("learning_rate", self.orch_train_learning_rate.value())))
            self.orch_train_weight_decay.setValue(float(values.get("weight_decay", self.orch_train_weight_decay.value())))
            self.orch_train_patience.setValue(int(values.get("patience", self.orch_train_patience.value())))
            self.orch_train_min_delta.setValue(float(values.get("min_delta", self.orch_train_min_delta.value())))
            self.orch_train_checkpoint_every.setValue(int(values.get("checkpoint_every", self.orch_train_checkpoint_every.value())))
            self.orch_train_resume_checkpoint.setText(str(values.get("resume_checkpoint", self.orch_train_resume_checkpoint.text())))
            self.orch_train_val_tracking_samples.setValue(int(values.get("val_tracking_samples", self.orch_train_val_tracking_samples.value())))
            self.orch_train_val_tracking_fixed.setText(str(values.get("val_tracking_fixed", self.orch_train_val_tracking_fixed.text())))
            self.orch_train_val_tracking_seed.setValue(int(values.get("val_tracking_seed", self.orch_train_val_tracking_seed.value())))
            self.orch_train_write_html_report.setChecked(bool(values.get("write_html_report", self.orch_train_write_html_report.isChecked())))
            self.orch_train_progress_interval.setValue(int(values.get("progress_interval", self.orch_train_progress_interval.value())))
            self.orch_train_seed.setValue(int(values.get("seed", self.orch_train_seed.value())))
            self.orch_train_require_qa.setChecked(bool(values.get("require_qa", self.orch_train_require_qa.isChecked())))
            self.workflow_tabs.setCurrentIndex(1)
            return
        if scope == "evaluation":
            self.orch_eval_config_edit.setText(str(values.get("config", self.orch_eval_config_edit.text())))
            self.orch_eval_set_edit.setText(str(values.get("overrides", self.orch_eval_set_edit.text())))
            self.orch_eval_dataset_edit.setText(str(values.get("dataset_dir", self.orch_eval_dataset_edit.text())))
            self.orch_eval_model_edit.setText(str(values.get("model_path", self.orch_eval_model_edit.text())))
            self.orch_eval_enable_gpu.setChecked(bool(values.get("enable_gpu", self.orch_eval_enable_gpu.isChecked())))
            self.orch_eval_device_policy.setCurrentText(str(values.get("device_policy", self.orch_eval_device_policy.currentText())))
            self.orch_eval_split_combo.setCurrentText(str(values.get("split", self.orch_eval_split_combo.currentText())))
            self.orch_eval_output_edit.setText(str(values.get("output_path", self.orch_eval_output_edit.text())))
            self.orch_eval_tracking_samples.setValue(int(values.get("tracking_samples", self.orch_eval_tracking_samples.value())))
            self.orch_eval_tracking_seed.setValue(int(values.get("tracking_seed", self.orch_eval_tracking_seed.value())))
            self.orch_eval_write_html_report.setChecked(bool(values.get("write_html_report", self.orch_eval_write_html_report.isChecked())))
            self.workflow_tabs.setCurrentIndex(2)
            return
        if scope == "hpc_ga":
            self.orch_hpc_config_edit.setText(str(values.get("config", self.orch_hpc_config_edit.text())))
            self.orch_hpc_set_edit.setText(str(values.get("overrides", self.orch_hpc_set_edit.text())))
            self.orch_hpc_dataset_edit.setText(str(values.get("dataset_dir", self.orch_hpc_dataset_edit.text())))
            self.orch_hpc_output_edit.setText(str(values.get("output_dir", self.orch_hpc_output_edit.text())))
            self.orch_hpc_experiment_name.setText(str(values.get("experiment_name", self.orch_hpc_experiment_name.text())))
            self.orch_hpc_scheduler_combo.setCurrentText(str(values.get("scheduler", self.orch_hpc_scheduler_combo.currentText())))
            self.orch_hpc_run_mode_combo.setCurrentText(str(values.get("run_mode", self.orch_hpc_run_mode_combo.currentText())))
            self.orch_hpc_architectures_edit.setText(str(values.get("architectures", self.orch_hpc_architectures_edit.text())))
            self.orch_hpc_num_candidates.setValue(int(values.get("num_candidates", self.orch_hpc_num_candidates.value())))
            self.orch_hpc_population.setValue(int(values.get("population_size", self.orch_hpc_population.value())))
            self.orch_hpc_generations.setValue(int(values.get("generations", self.orch_hpc_generations.value())))
            self.orch_hpc_mutation.setValue(float(values.get("mutation_rate", self.orch_hpc_mutation.value())))
            self.orch_hpc_crossover.setValue(float(values.get("crossover_rate", self.orch_hpc_crossover.value())))
            self.orch_hpc_seed.setValue(int(values.get("seed", self.orch_hpc_seed.value())))
            self.orch_hpc_lr_min.setValue(float(values.get("learning_rate_min", self.orch_hpc_lr_min.value())))
            self.orch_hpc_lr_max.setValue(float(values.get("learning_rate_max", self.orch_hpc_lr_max.value())))
            self.orch_hpc_batch_sizes.setText(str(values.get("batch_size_choices", self.orch_hpc_batch_sizes.text())))
            self.orch_hpc_epochs_min.setValue(int(values.get("epochs_min", self.orch_hpc_epochs_min.value())))
            self.orch_hpc_epochs_max.setValue(int(values.get("epochs_max", self.orch_hpc_epochs_max.value())))
            self.orch_hpc_wd_min.setValue(float(values.get("weight_decay_min", self.orch_hpc_wd_min.value())))
            self.orch_hpc_wd_max.setValue(float(values.get("weight_decay_max", self.orch_hpc_wd_max.value())))
            self.orch_hpc_ms_min.setValue(int(values.get("max_samples_min", self.orch_hpc_ms_min.value())))
            self.orch_hpc_ms_max.setValue(int(values.get("max_samples_max", self.orch_hpc_ms_max.value())))
            self.orch_hpc_fitness_mode.setCurrentText(str(values.get("fitness_mode", self.orch_hpc_fitness_mode.currentText())))
            self.orch_hpc_feedback_sources.setText(str(values.get("feedback_sources", self.orch_hpc_feedback_sources.text())))
            self.orch_hpc_feedback_min_samples.setValue(int(values.get("feedback_min_samples", self.orch_hpc_feedback_min_samples.value())))
            self.orch_hpc_feedback_k.setValue(int(values.get("feedback_k", self.orch_hpc_feedback_k.value())))
            self.orch_hpc_exploration_weight.setValue(float(values.get("exploration_weight", self.orch_hpc_exploration_weight.value())))
            self.orch_hpc_w_iou.setValue(float(values.get("fitness_weight_mean_iou", self.orch_hpc_w_iou.value())))
            self.orch_hpc_w_f1.setValue(float(values.get("fitness_weight_macro_f1", self.orch_hpc_w_f1.value())))
            self.orch_hpc_w_acc.setValue(float(values.get("fitness_weight_pixel_accuracy", self.orch_hpc_w_acc.value())))
            self.orch_hpc_w_runtime.setValue(float(values.get("fitness_weight_runtime", self.orch_hpc_w_runtime.value())))
            self.orch_hpc_enable_gpu.setChecked(bool(values.get("enable_gpu", self.orch_hpc_enable_gpu.isChecked())))
            self.orch_hpc_device_policy.setCurrentText(str(values.get("device_policy", self.orch_hpc_device_policy.currentText())))
            self.orch_hpc_queue.setText(str(values.get("queue", self.orch_hpc_queue.text())))
            self.orch_hpc_account.setText(str(values.get("account", self.orch_hpc_account.text())))
            self.orch_hpc_qos.setText(str(values.get("qos", self.orch_hpc_qos.text())))
            self.orch_hpc_gpus.setValue(int(values.get("gpus_per_job", self.orch_hpc_gpus.value())))
            self.orch_hpc_cpus.setValue(int(values.get("cpus_per_task", self.orch_hpc_cpus.value())))
            self.orch_hpc_mem.setValue(int(values.get("mem_gb", self.orch_hpc_mem.value())))
            self.orch_hpc_time.setText(str(values.get("time_limit", self.orch_hpc_time.text())))
            self.orch_hpc_job_prefix.setText(str(values.get("job_prefix", self.orch_hpc_job_prefix.text())))
            self.orch_hpc_python.setText(str(values.get("python_executable", self.orch_hpc_python.text())))
            self.orch_hpc_cli_path.setText(str(values.get("microseg_cli_path", self.orch_hpc_cli_path.text())))
            self.orch_hpc_base_train.setText(str(values.get("base_train_config", self.orch_hpc_base_train.text())))
            self.orch_hpc_base_eval.setText(str(values.get("base_eval_config", self.orch_hpc_base_eval.text())))
            self.orch_hpc_eval_split.setCurrentText(str(values.get("eval_split", self.orch_hpc_eval_split.currentText())))
            self.orch_hpc_feedback_top_k.setValue(int(values.get("feedback_top_k", self.orch_hpc_feedback_top_k.value())))
            self.orch_hpc_feedback_report_output.setText(
                str(values.get("feedback_report_output", self.orch_hpc_feedback_report_output.text()))
            )
            self.workflow_tabs.setCurrentIndex(self.workflow_tabs.indexOf(self.workflow_hpc_tab))
            return
        raise ValueError(f"Unsupported profile scope: {scope}")

    def on_save_workflow_profile(self) -> None:
        scope = self.workflow_profile_scope.currentText()
        path, _ = QFileDialog.getSaveFileName(self, "Save workflow profile", "", "YAML (*.yml *.yaml)")
        if not path:
            return
        try:
            write_workflow_profile(path, scope=scope, values=self._collect_workflow_profile(scope))
            self.workflow_notes.append(f"[Profile] Saved {scope} profile to {path}")
            QMessageBox.information(self, "Profile Saved", f"Saved profile:\n{path}")
        except Exception as exc:
            self.logger.exception("Failed to save workflow profile")
            QMessageBox.critical(self, "Save Profile Error", str(exc))

    def on_load_workflow_profile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load workflow profile", "", "YAML (*.yml *.yaml)")
        if not path:
            return
        try:
            payload = read_workflow_profile(path)
            scope = str(payload.get("scope", self.workflow_profile_scope.currentText()))
            values = payload.get("values", {})
            if not isinstance(values, dict):
                raise ValueError("Profile values must be a mapping")
            self.workflow_profile_scope.setCurrentText(scope)
            self._apply_workflow_profile(scope, values)
            self.workflow_notes.append(f"[Profile] Loaded {scope} profile from {path}")
            QMessageBox.information(self, "Profile Loaded", f"Loaded profile:\n{path}")
        except Exception as exc:
            self.logger.exception("Failed to load workflow profile")
            QMessageBox.critical(self, "Load Profile Error", str(exc))

    def on_orchestrate_inference(self) -> None:
        overrides = self._parse_override_text(self.orch_infer_set_edit.text())
        overrides.extend(
            [
                f"enable_gpu={str(self.orch_infer_enable_gpu.isChecked()).lower()}",
                f"device_policy={self.orch_infer_device_policy.currentText()}",
            ]
        )
        command = self.orchestrator.infer(
            config=self.orch_infer_config_edit.text().strip() or None,
            overrides=overrides,
            image=self.orch_infer_image_edit.text().strip() or None,
            model_name=self.orch_infer_model_edit.text().strip() or self.model_combo.currentText(),
            output_dir=self.orch_infer_output_edit.text().strip() or None,
        )
        self._start_orchestration_job(command, "Inference")

    def on_orchestrate_training(self) -> None:
        overrides = self._parse_override_text(self.orch_train_set_edit.text())
        overrides.extend(
            [
                f"backend={self.orch_train_backend.currentText()}",
                f"max_samples={int(self.orch_train_max_samples.value())}",
                f"epochs={int(self.orch_train_epochs.value())}",
                f"batch_size={int(self.orch_train_batch_size.value())}",
                f"learning_rate={float(self.orch_train_learning_rate.value())}",
                f"weight_decay={float(self.orch_train_weight_decay.value())}",
                f"early_stopping_patience={int(self.orch_train_patience.value())}",
                f"early_stopping_min_delta={float(self.orch_train_min_delta.value())}",
                f"checkpoint_every={int(self.orch_train_checkpoint_every.value())}",
                f"val_tracking_samples={int(self.orch_train_val_tracking_samples.value())}",
                f"val_tracking_seed={int(self.orch_train_val_tracking_seed.value())}",
                f"write_html_report={str(self.orch_train_write_html_report.isChecked()).lower()}",
                f"progress_log_interval_pct={int(self.orch_train_progress_interval.value())}",
                f"seed={int(self.orch_train_seed.value())}",
                f"enable_gpu={str(self.orch_train_enable_gpu.isChecked()).lower()}",
                f"device_policy={self.orch_train_device_policy.currentText()}",
            ]
        )
        resume = self.orch_train_resume_checkpoint.text().strip()
        if resume:
            overrides.append(f"resume_checkpoint={resume}")
        fixed = self.orch_train_val_tracking_fixed.text().strip()
        if fixed:
            overrides.append(f"val_tracking_fixed_samples={fixed}")
        dataset_dir = self.orch_train_dataset_edit.text().strip() or None
        output_dir = self.orch_train_output_edit.text().strip() or "outputs/training"
        if self.orch_train_require_qa.isChecked() and not dataset_dir:
            train_cfg = resolve_config(self.orch_train_config_edit.text().strip() or None, overrides)
            cfg_dataset = str(train_cfg.get("dataset_dir", "")).strip()
            cfg_output = str(train_cfg.get("output_dir", "")).strip()
            dataset_dir = cfg_dataset or None
            if cfg_output:
                output_dir = cfg_output
        if dataset_dir:
            gate = self._preflight_training_dataset_gate(dataset_dir, output_dir)
            if gate is None:
                return
            dataset_dir, gate_overrides = gate
            overrides.extend(gate_overrides)
            self.orch_train_dataset_edit.setText(dataset_dir)
        command = self.orchestrator.train(
            config=self.orch_train_config_edit.text().strip() or None,
            overrides=overrides,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
        self._start_orchestration_job(command, "Training")

    def on_orchestrate_evaluation(self) -> None:
        overrides = self._parse_override_text(self.orch_eval_set_edit.text())
        overrides.extend(
            [
                f"split={self.orch_eval_split_combo.currentText()}",
                f"enable_gpu={str(self.orch_eval_enable_gpu.isChecked()).lower()}",
                f"device_policy={self.orch_eval_device_policy.currentText()}",
                f"tracking_samples={int(self.orch_eval_tracking_samples.value())}",
                f"tracking_seed={int(self.orch_eval_tracking_seed.value())}",
                f"write_html_report={str(self.orch_eval_write_html_report.isChecked()).lower()}",
            ]
        )
        command = self.orchestrator.evaluate(
            config=self.orch_eval_config_edit.text().strip() or None,
            overrides=overrides,
            dataset_dir=self.orch_eval_dataset_edit.text().strip() or None,
            model_path=self.orch_eval_model_edit.text().strip() or None,
            split=self.orch_eval_split_combo.currentText(),
            output_path=self.orch_eval_output_edit.text().strip() or None,
        )
        self._start_orchestration_job(command, "Evaluation")

    def on_orchestrate_hpc_ga(self) -> None:
        dataset_dir = self.orch_hpc_dataset_edit.text().strip()
        output_dir = self.orch_hpc_output_edit.text().strip()
        if not dataset_dir or not output_dir:
            QMessageBox.warning(self, "Missing Paths", "Dataset Dir and Bundle Output Dir are required.")
            return

        overrides = self._parse_override_text(self.orch_hpc_set_edit.text())
        overrides.extend(
            [
                f"experiment_name={self.orch_hpc_experiment_name.text().strip()}",
                f"scheduler={self.orch_hpc_scheduler_combo.currentText()}",
                f"run_mode={self.orch_hpc_run_mode_combo.currentText()}",
                f"architectures={self.orch_hpc_architectures_edit.text().strip()}",
                f"num_candidates={int(self.orch_hpc_num_candidates.value())}",
                f"population_size={int(self.orch_hpc_population.value())}",
                f"generations={int(self.orch_hpc_generations.value())}",
                f"mutation_rate={float(self.orch_hpc_mutation.value())}",
                f"crossover_rate={float(self.orch_hpc_crossover.value())}",
                f"seed={int(self.orch_hpc_seed.value())}",
                f"learning_rate_min={float(self.orch_hpc_lr_min.value())}",
                f"learning_rate_max={float(self.orch_hpc_lr_max.value())}",
                f"batch_size_choices={self.orch_hpc_batch_sizes.text().strip()}",
                f"epochs_min={int(self.orch_hpc_epochs_min.value())}",
                f"epochs_max={int(self.orch_hpc_epochs_max.value())}",
                f"weight_decay_min={float(self.orch_hpc_wd_min.value())}",
                f"weight_decay_max={float(self.orch_hpc_wd_max.value())}",
                f"max_samples_min={int(self.orch_hpc_ms_min.value())}",
                f"max_samples_max={int(self.orch_hpc_ms_max.value())}",
                f"fitness_mode={self.orch_hpc_fitness_mode.currentText()}",
                f"feedback_sources={self.orch_hpc_feedback_sources.text().strip()}",
                f"feedback_min_samples={int(self.orch_hpc_feedback_min_samples.value())}",
                f"feedback_k={int(self.orch_hpc_feedback_k.value())}",
                f"exploration_weight={float(self.orch_hpc_exploration_weight.value())}",
                f"fitness_weight_mean_iou={float(self.orch_hpc_w_iou.value())}",
                f"fitness_weight_macro_f1={float(self.orch_hpc_w_f1.value())}",
                f"fitness_weight_pixel_accuracy={float(self.orch_hpc_w_acc.value())}",
                f"fitness_weight_runtime={float(self.orch_hpc_w_runtime.value())}",
                f"enable_gpu={str(self.orch_hpc_enable_gpu.isChecked()).lower()}",
                f"device_policy={self.orch_hpc_device_policy.currentText()}",
                f"queue={self.orch_hpc_queue.text().strip()}",
                f"account={self.orch_hpc_account.text().strip()}",
                f"qos={self.orch_hpc_qos.text().strip()}",
                f"gpus_per_job={int(self.orch_hpc_gpus.value())}",
                f"cpus_per_task={int(self.orch_hpc_cpus.value())}",
                f"mem_gb={int(self.orch_hpc_mem.value())}",
                f"time_limit={self.orch_hpc_time.text().strip()}",
                f"job_prefix={self.orch_hpc_job_prefix.text().strip()}",
                f"python_executable={self.orch_hpc_python.text().strip()}",
                f"microseg_cli_path={self.orch_hpc_cli_path.text().strip()}",
                f"base_train_config={self.orch_hpc_base_train.text().strip()}",
                f"base_eval_config={self.orch_hpc_base_eval.text().strip()}",
                f"eval_split={self.orch_hpc_eval_split.currentText()}",
            ]
        )
        command = self.orchestrator.hpc_ga_generate(
            config=self.orch_hpc_config_edit.text().strip() or None,
            overrides=overrides,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
        )
        self.orch_hpc_preview.appendPlainText("$ " + " ".join(command))
        self._start_orchestration_job(command, "HPC-GA-Bundle")

    def on_orchestrate_hpc_feedback_report(self) -> None:
        feedback_sources = self.orch_hpc_feedback_sources.text().strip()
        if not feedback_sources:
            QMessageBox.warning(
                self,
                "Missing Feedback Sources",
                "Set one or more feedback sources (bundle directory or ga_plan_manifest.json path).",
            )
            return
        overrides = self._parse_override_text(self.orch_hpc_set_edit.text())
        overrides.extend(
            [
                f"dataset_dir={self.orch_hpc_dataset_edit.text().strip()}",
                f"output_dir={self.orch_hpc_output_edit.text().strip()}",
                f"architectures={self.orch_hpc_architectures_edit.text().strip()}",
                f"batch_size_choices={self.orch_hpc_batch_sizes.text().strip()}",
                f"fitness_mode={self.orch_hpc_fitness_mode.currentText()}",
                f"feedback_min_samples={int(self.orch_hpc_feedback_min_samples.value())}",
                f"feedback_k={int(self.orch_hpc_feedback_k.value())}",
                f"exploration_weight={float(self.orch_hpc_exploration_weight.value())}",
                f"fitness_weight_mean_iou={float(self.orch_hpc_w_iou.value())}",
                f"fitness_weight_macro_f1={float(self.orch_hpc_w_f1.value())}",
                f"fitness_weight_pixel_accuracy={float(self.orch_hpc_w_acc.value())}",
                f"fitness_weight_runtime={float(self.orch_hpc_w_runtime.value())}",
                f"top_k={int(self.orch_hpc_feedback_top_k.value())}",
            ]
        )
        command = self.orchestrator.hpc_ga_feedback_report(
            config=self.orch_hpc_config_edit.text().strip() or None,
            overrides=overrides,
            feedback_sources=feedback_sources,
            output_path=self.orch_hpc_feedback_report_output.text().strip() or None,
        )
        self.orch_hpc_preview.appendPlainText("$ " + " ".join(command))
        self._start_orchestration_job(command, "HPC-GA-Feedback")

    def on_edit_classes(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("Edit Class Map")
        dlg.resize(700, 480)
        v = QVBoxLayout(dlg)
        help_label = QLabel("One class per line: index,name,#RRGGBB[,description]")
        v.addWidget(help_label)
        text = QTextEdit()
        text.setPlainText(self._class_map_to_text(self.state.class_map))
        v.addWidget(text)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        v.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return
        try:
            cmap = self._class_map_from_text(text.toPlainText())
            self.state.class_map = cmap
            self.corrected_canvas.set_class_map(cmap)
            self._reload_class_combo()
            self.logger.info("Updated class map with %d classes", len(cmap.classes))
        except Exception as exc:
            QMessageBox.critical(self, "Class Map Error", str(exc))

    def _update_split_input_view(self) -> None:
        run = self.state.current_run
        if run is None:
            return
        self._set_image_preview(self.raw_corr_view, np.array(run.input_image), zoom=self.corrected_canvas.zoom_value())

    def _update_action_label(self) -> None:
        sess = self.state.correction_session
        if sess is None:
            self.action_label.setText("Actions: 0")
            return
        rep = sess.report()
        self.action_label.setText(f"Actions: {rep.actions_applied} | FG: {rep.current_foreground_pixels}")

    def _on_zoom_changed(self, zoom: float) -> None:
        self.zoom_label.setText(f"Zoom: {int(round(zoom * 100))}%")
        self._update_split_input_view()

    def _on_cursor_changed(self, x: int, y: int) -> None:
        self.cursor_label.setText(f"Cursor: {x},{y}")

    def _on_correction_changed(self) -> None:
        self._update_action_label()
        self._queue_results_refresh()
        self._feedback_correction_timer.start(700)

    def _on_tool_changed(self, tool: str) -> None:
        self.corrected_canvas.set_tool(tool)
        self.radius_spin.setEnabled(tool == "brush")
        self.tool_label.setText(f"Tool: {tool}/{self.mode_combo.currentText()}")
        self.logger.info("Tool changed to %s", tool)

    def _on_mode_changed(self, mode: str) -> None:
        self.corrected_canvas.set_mode(mode)
        self.tool_label.setText(f"Tool: {self.tool_combo.currentText()}/{mode}")
        self.logger.info("Mode changed to %s", mode)

    def on_show_shortcuts(self) -> None:
        QMessageBox.information(
            self,
            "Shortcuts",
            "B: Brush tool\n"
            "P: Polygon tool\n"
            "L: Lasso tool\n"
            "F: Feature-select tool (delete/relabel connected feature)\n"
            "A: Add mode\n"
            "R: Erase mode\n"
            "Ctrl+Z/Ctrl+Y: Undo/Redo\n"
            "Ctrl+Wheel or Ctrl+/-, Ctrl+0: Zoom\n"
            "Esc: Cancel polygon/lasso preview",
        )

    def on_show_guide(self) -> None:
        QMessageBox.information(
            self,
            "Correction Guide",
            "1. Load a file or sample image and run segmentation.\n"
            "2. Open 'Correction Split View'.\n"
            "3. Pick class index/color map and select tool/mode.\n"
            "4. For wrong objects: feature-select + erase to delete component.\n"
            "5. Redraw with brush/polygon/lasso in add mode.\n"
            "6. Tune layer transparency and inspect 'Results Dashboard'.\n"
            "7. Optionally calibrate scale (manual line or TIFF metadata) for micron-based reporting.\n"
            "8. Export correction masks and full JSON/HTML/PDF result packages.\n"
            "9. Tune conventional model controls when running Hydride Conventional.\n"
            "10. Use Workflow Hub for train/infer/evaluate/package orchestration jobs.\n"
            "11. Use Dataset Prep + QA for split preview, colormap conversion, and QA gating.\n"
            "12. Use Run Review to compare training/evaluation reports.\n"
            "13. Use HPC GA Planner to generate Slurm/PBS/local job bundles.",
        )

    def on_show_about(self) -> None:
        QMessageBox.information(
            self,
            "About MicroSeg Desktop",
            f"MicroSeg Desktop v{__version__}\n"
            "Qt-based local application for segmentation review and correction\n"
            "Designed for field deployment workflows.",
        )

    def on_show_model_details(self) -> None:
        spec = self._selected_model_spec()
        if not spec:
            QMessageBox.information(self, "Model Details", "No model metadata available.")
            return
        lines = []
        for key in [
            "display_name",
            "model_id",
            "feature_family",
            "description",
            "details",
            "model_nickname",
            "model_type",
            "framework",
            "input_dimensions",
            "checkpoint_path_hint",
            "artifact_stage",
            "application_remarks",
            "short_description",
            "detailed_description",
            "quality_report_path",
        ]:
            value = str(spec.get(key, "")).strip()
            if not value:
                continue
            lines.append(f"<b>{key}</b>: {value}")
        dlg = QDialog(self)
        dlg.setWindowTitle("Model Details")
        dlg.resize(760, 480)
        root = QVBoxLayout(dlg)
        body = QTextEdit()
        body.setReadOnly(True)
        body.setHtml("<br>".join(lines))
        root.addWidget(body)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(dlg.reject)
        buttons.accepted.connect(dlg.accept)
        root.addWidget(buttons)
        dlg.exec()

    def on_open_results_dashboard(self) -> None:
        idx = self.tabs.indexOf(self.results_widget)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)
            self._update_results_dashboard()

    def on_open_workflow_hub(self) -> None:
        idx = self.tabs.indexOf(self.workflow_widget)
        if idx >= 0:
            self.tabs.setCurrentIndex(idx)

    def on_open_log_folder(self) -> None:
        log_dir = self.orchestrator.repo_root / "outputs" / "logs" / "desktop"
        opened = False
        try:
            if sys.platform.startswith("darwin"):
                opened = bool(QProcess.startDetached("open", [str(log_dir)]))
            elif os.name == "nt":
                opened = bool(QProcess.startDetached("explorer", [str(log_dir)]))
            else:
                opened = bool(QProcess.startDetached("xdg-open", [str(log_dir)]))
        except Exception:
            opened = False
        if not opened:
            QMessageBox.information(self, "Log Folder", f"Desktop logs:\n{log_dir}")

    def on_clear_calibration(self) -> None:
        self._apply_calibration(None, image_path=self.state.image_path)
        self.logger.info("Spatial calibration cleared; reporting units reverted to pixels.")

    def on_scan_metadata_calibration(self) -> None:
        image_path = self.path_edit.text().strip() or (self.state.image_path or "")
        if not image_path:
            QMessageBox.warning(self, "Missing image", "Load an image first to scan metadata calibration.")
            return
        self._try_auto_calibration_from_metadata(image_path, user_initiated=True)

    def on_calibrate_scale(self) -> None:
        image_path = self.path_edit.text().strip() or (self.state.image_path or "")
        if not image_path:
            QMessageBox.warning(self, "Missing image", "Load an image first before calibration.")
            return
        try:
            image_arr = np.array(Image.open(image_path))
        except Exception as exc:
            QMessageBox.critical(self, "Calibration Error", f"Failed to load image:\n{exc}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Spatial Calibration")
        dlg.resize(980, 760)
        root = QVBoxLayout(dlg)

        info = QLabel(
            "Draw a known reference line with two left-clicks. Right-click clears the line.\n"
            "Then enter the real length and apply calibration."
        )
        info.setWordWrap(True)
        root.addWidget(info)

        canvas = CalibrationLineCanvas()
        canvas.set_image(image_arr)
        scroll = self._in_scroll(canvas)
        root.addWidget(scroll, stretch=1)

        form = QFormLayout()
        known_length = QDoubleSpinBox()
        known_length.setDecimals(6)
        known_length.setRange(0.000001, 1_000_000.0)
        known_length.setValue(100.0)
        known_unit = QComboBox()
        known_unit.addItems(["um", "mm", "nm"])
        known_unit.setCurrentText("um")
        distance_label = QLabel("Line distance: 0.000 px")
        derived_label = QLabel("Derived scale: n/a")
        form.addRow("Known length", known_length)
        form.addRow("Length unit", known_unit)
        form.addRow("Measured line", distance_label)
        form.addRow("Scale", derived_label)
        root.addLayout(form)

        def _refresh_labels(distance_px: float) -> None:
            distance_label.setText(f"Line distance: {distance_px:.3f} px")
            if distance_px <= 0:
                derived_label.setText("Derived scale: n/a")
                return
            try:
                cal = calibration_from_manual_line(distance_px, known_length.value(), known_unit.currentText())
                derived_label.setText(f"Derived scale: {cal.microns_per_pixel:.6g} um/px")
            except Exception as exc:
                derived_label.setText(f"Derived scale: invalid ({exc})")

        canvas.line_changed.connect(_refresh_labels)
        known_length.valueChanged.connect(lambda *_args: _refresh_labels(canvas.line_distance_px()))
        known_unit.currentTextChanged.connect(lambda *_args: _refresh_labels(canvas.line_distance_px()))

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        root.addWidget(buttons)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)

        if dlg.exec() != QDialog.Accepted:
            return

        distance_px = canvas.line_distance_px()
        if distance_px <= 0:
            QMessageBox.warning(self, "Calibration", "Draw a calibration line before applying.")
            return
        try:
            cal = calibration_from_manual_line(distance_px, known_length.value(), known_unit.currentText())
            self._apply_calibration(cal, image_path=image_path)
            self.logger.info(
                "Applied manual calibration for %s: %.6g um/px",
                image_path,
                cal.microns_per_pixel,
            )
            QMessageBox.information(
                self,
                "Calibration Applied",
                f"Scale set to {cal.microns_per_pixel:.6g} um/px\nSource: manual line",
            )
        except Exception as exc:
            QMessageBox.critical(self, "Calibration Error", str(exc))

    def _preview_input_image(self, path: str) -> None:
        try:
            with Image.open(path) as img:
                arr = np.array(img)
            self._set_image_preview(self.input_view, arr)
            self._set_image_preview(self.raw_corr_view, arr, zoom=self.corrected_canvas.zoom_value())
        except Exception as exc:
            self.logger.warning("Failed to preview input image: %s", exc)

    def on_load_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select image",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not path:
            return
        self.path_edit.setText(path)
        self.orch_infer_image_edit.setText(path)
        self.state.image_path = path
        self._preview_input_image(path)
        self._try_auto_calibration_from_metadata(path, user_initiated=False)
        self.tabs.setCurrentIndex(0)
        self.logger.info("Loaded image path: %s", path)

    def on_run_segmentation(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing image", "Select an image first")
            return
        self.state.image_path = path
        self._try_auto_calibration_from_metadata(path, user_initiated=False)

        try:
            cfg = self._resolve_run_config()
            model_name = str(cfg.get("model_name") or self.model_combo.currentText())
            self.model_combo.setCurrentText(model_name)
            include_analysis = bool(cfg.get("include_analysis", True))
            params = dict(cfg.get("params", {}))
            if self._selected_model_id(model_name) == "hydride_conventional":
                params.update(self._collect_conventional_params())
            params["image_path"] = path
            self.logger.info("Running segmentation on %s with %s", path, model_name)
            record = self.workflow.run_single(
                path,
                model_name=model_name,
                params=params,
                include_analysis=include_analysis,
            )
            self._capture_feedback_for_record(record, resolved_config=cfg, params=params, source="desktop_gui")
            self._add_record(record)
            self._show_record(record)
        except Exception as exc:
            self.logger.exception("Segmentation failed")
            QMessageBox.critical(self, "Segmentation Error", str(exc))

    def on_run_batch(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select batch images",
            "",
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp)",
        )
        if not paths:
            return
        self.state.image_path = str(paths[0])
        self._try_auto_calibration_from_metadata(str(paths[0]), user_initiated=False)
        try:
            cfg = self._resolve_run_config()
            model_name = str(cfg.get("model_name") or self.model_combo.currentText())
            self.model_combo.setCurrentText(model_name)
            include_analysis = bool(cfg.get("include_analysis", False))
            params = dict(cfg.get("params", {}))
            if self._selected_model_id(model_name) == "hydride_conventional":
                params.update(self._collect_conventional_params())
            params.setdefault("image_path", paths[0])
            self.logger.info("Running batch of %d images with %s", len(paths), model_name)
            records = self.workflow.run_batch(
                list(paths),
                model_name=model_name,
                params=params,
                include_analysis=include_analysis,
            )
            for rec in records:
                params_row = dict(params)
                params_row["image_path"] = rec.image_path
                self._capture_feedback_for_record(rec, resolved_config=cfg, params=params_row, source="desktop_gui")
                self._add_record(rec)
            if records:
                self._show_record(records[-1])
                self.history_list.setCurrentRow(self.history_list.count() - 1)
        except Exception as exc:
            self.logger.exception("Batch run failed")
            QMessageBox.critical(self, "Batch Error", str(exc))

    def _add_record(self, record: DesktopRunRecord) -> None:
        self.history_list.addItem(record.history_label)

    def _active_operator_id(self) -> str:
        text = self.annotator_edit.text().strip()
        if text:
            return text
        return str(self.feedback_writer.config.operator_id)

    def _capture_feedback_for_record(
        self,
        record: DesktopRunRecord,
        *,
        resolved_config: dict[str, object] | None,
        params: dict[str, object] | None,
        source: str,
    ) -> None:
        if str(record.feedback_record_dir).strip():
            return
        try:
            capture = self.feedback_writer.create_from_desktop_run(
                record,
                source=source,
                resolved_config=dict(resolved_config or {}),
                params=dict(params or {}),
                runtime={
                    "enable_gpu": bool((params or {}).get("enable_gpu", False)),
                    "device_policy": str((params or {}).get("device_policy", "cpu")),
                },
                operator_id=self._active_operator_id(),
            )
            record.feedback_record_dir = str(capture.record_dir)
            record.feedback_record_id = str(capture.record_id)
            self.logger.info("Feedback record captured: %s", record.feedback_record_dir)
        except Exception:
            self.logger.exception("Failed to capture feedback record for run_id=%s", record.run_id)

    def _apply_feedback_rating_ui(self) -> None:
        rating = str(self.state.current_feedback_rating or "unrated")
        self.feedback_rating_label.setText(f"Feedback: {rating}")
        up_active = rating == "thumbs_up"
        down_active = rating == "thumbs_down"
        self.btn_thumb_up.setStyleSheet("background-color: #dff0d8;" if up_active else "")
        self.btn_thumb_down.setStyleSheet("background-color: #f2dede;" if down_active else "")

    def _load_feedback_for_record(self, record: DesktopRunRecord) -> None:
        self.state.current_feedback_record_dir = str(record.feedback_record_dir or "")
        self.state.current_feedback_rating = "unrated"
        comment = ""
        if self.state.current_feedback_record_dir:
            try:
                payload = load_feedback_record(self.state.current_feedback_record_dir)
                self.state.current_feedback_rating = str(payload.get("feedback", {}).get("rating", "unrated"))
                comment = str(payload.get("feedback", {}).get("comment", ""))
                record.feedback_record_id = str(payload.get("record_id", record.feedback_record_id))
            except Exception:
                self.logger.exception("Failed to load feedback record: %s", self.state.current_feedback_record_dir)
        self._suppress_feedback_note_events = True
        self.notes_edit.setText(comment)
        self._suppress_feedback_note_events = False
        self._apply_feedback_rating_ui()

    def _persist_feedback(self, *, rating: str | None = None, comment: str | None = None) -> None:
        record_dir = str(self.state.current_feedback_record_dir or "").strip()
        if not record_dir:
            return
        try:
            payload = self.feedback_writer.update_feedback(
                record_dir,
                rating=rating if rating in {"unrated", "thumbs_up", "thumbs_down"} else None,
                comment=comment,
                operator_id=self._active_operator_id(),
            )
            self.state.current_feedback_rating = str(payload.get("feedback", {}).get("rating", "unrated"))
            self._apply_feedback_rating_ui()
        except Exception:
            self.logger.exception("Failed to persist feedback: %s", record_dir)

    def _on_feedback_rating_clicked(self, rating: str) -> None:
        if rating not in {"thumbs_up", "thumbs_down"}:
            return
        self._persist_feedback(rating=rating, comment=self.notes_edit.text().strip())
        self._maybe_attach_current_correction()

    def _on_feedback_comment_changed(self) -> None:
        if self._suppress_feedback_note_events:
            return
        self._feedback_comment_timer.start(350)

    def _flush_feedback_comment(self) -> None:
        self._persist_feedback(comment=self.notes_edit.text().strip())

    def _flush_feedback_correction(self) -> None:
        self._maybe_attach_current_correction()

    def _maybe_attach_current_correction(self) -> None:
        record_dir = str(self.state.current_feedback_record_dir or "").strip()
        sess = self.state.correction_session
        run = self.state.current_run
        if not record_dir or sess is None or run is None:
            return
        try:
            pred = to_index_mask(np.array(run.mask_image))
            corr = to_index_mask(np.array(sess.current_mask))
            if np.array_equal(pred, corr):
                return
            self.feedback_writer.attach_corrected_mask(record_dir, corr)
        except Exception:
            self.logger.exception("Failed to attach corrected mask to feedback record: %s", record_dir)

    def _link_feedback_correction_export(self, correction_record_path: str) -> None:
        record_dir = str(self.state.current_feedback_record_dir or "").strip()
        if not record_dir:
            return
        try:
            self.feedback_writer.link_correction_export(
                record_dir,
                correction_record_path=str(correction_record_path),
            )
        except Exception:
            self.logger.exception("Failed to link correction export to feedback record: %s", record_dir)

    def _show_record(self, record: DesktopRunRecord, corrected_mask: np.ndarray | None = None) -> None:
        self.state.current_run = record
        if not str(record.feedback_record_dir).strip():
            params = {}
            if isinstance(record.manifest, dict):
                raw = record.manifest.get("params", {})
                if isinstance(raw, dict):
                    params = dict(raw)
            self._capture_feedback_for_record(
                record,
                resolved_config={"from_history": True, "manifest": record.manifest},
                params=params,
                source="desktop_gui",
            )
        self.path_edit.setText(record.image_path)
        self.state.image_path = record.image_path
        self.orch_infer_image_edit.setText(record.image_path)
        self._try_auto_calibration_from_metadata(record.image_path, user_initiated=False)
        if self.model_combo.findText(record.model_name) >= 0:
            self.model_combo.setCurrentText(record.model_name)
        base = np.array(record.input_image)
        pred_mask = to_index_mask(np.array(record.mask_image))
        self.state.correction_session = CorrectionSession(pred_mask)
        if corrected_mask is not None:
            self.state.correction_session.current_mask = to_index_mask(corrected_mask)

        self._set_image_preview(self.input_view, base)
        self._set_image_preview(self.mask_view, _mask_to_pixmap(pred_mask, self.state.class_map))
        self._set_image_preview(self.overlay_view, np.array(record.overlay_image))

        self.corrected_canvas.bind(base, pred_mask, self.state.correction_session, self.state.class_map)
        self.corrected_canvas.update_layer_settings(self._layer_settings())
        self._update_split_input_view()
        self._update_action_label()
        self._update_results_dashboard()
        self._load_feedback_for_record(record)

        self.logger.info("Active run: %s", record.history_label)
        if record.metrics:
            self.logger.info("Metrics: %s", record.metrics)

    def on_history_selected(self, index: int) -> None:
        if index < 0:
            return
        try:
            records = self.workflow.history()
            if index >= len(records):
                return
            self._show_record(records[index])
        except Exception:
            self.logger.exception("Failed to load history selection")

    def on_undo(self) -> None:
        sess = self.state.correction_session
        if sess is None:
            return
        if sess.undo():
            self.corrected_canvas._refresh()
            self._update_action_label()
            self._queue_results_refresh()
            self.logger.info("Correction undo")

    def on_redo(self) -> None:
        sess = self.state.correction_session
        if sess is None:
            return
        if sess.redo():
            self.corrected_canvas._refresh()
            self._update_action_label()
            self._queue_results_refresh()
            self.logger.info("Correction redo")

    def on_reset_corrections(self) -> None:
        sess = self.state.correction_session
        if sess is None:
            return
        sess.reset_to_initial()
        self.corrected_canvas._refresh()
        self._update_action_label()
        self._queue_results_refresh()
        self.logger.info("Corrections reset to initial prediction")

    def on_export_correction(self) -> None:
        run = self.state.current_run
        sess = self.state.correction_session
        if run is None or sess is None:
            QMessageBox.warning(self, "No run", "Run segmentation first")
            return

        out_dir = QFileDialog.getExistingDirectory(self, "Select export directory")
        if not out_dir:
            return

        try:
            self._maybe_attach_current_correction()
            sample_dir = self.exporter.export_sample(
                run,
                sess.current_mask,
                out_dir,
                annotator=self.annotator_edit.text().strip() or "unknown",
                notes=self.notes_edit.text().strip(),
                class_map=self.state.class_map,
                formats=self._selected_export_formats(),
                feedback_record_id=str(run.feedback_record_id or ""),
                feedback_record_dir=str(run.feedback_record_dir or ""),
            )
            self._link_feedback_correction_export(str(Path(sample_dir) / "correction_record.json"))
            self.logger.info("Exported corrected sample: %s", sample_dir)
            QMessageBox.information(self, "Export complete", f"Saved to:\n{sample_dir}")
        except Exception as exc:
            self.logger.exception("Correction export failed")
            QMessageBox.critical(self, "Export Error", str(exc))

    def on_export_results_package(self) -> None:
        run = self.state.current_run
        if run is None:
            QMessageBox.warning(self, "No run", "Run segmentation first")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select results export directory")
        if not out_dir:
            return
        sess = self.state.correction_session
        corrected_mask = sess.current_mask if sess is not None else None
        self._maybe_attach_current_correction()
        try:
            export_cfg = self._results_export_config_from_ui()
            export_dir = self.result_exporter.export(
                run,
                output_dir=out_dir,
                corrected_mask=corrected_mask,
                annotator=self.annotator_edit.text().strip() or "unknown",
                notes=self.notes_edit.text().strip(),
                class_map=self.state.class_map,
                config=export_cfg,
            )
            self.logger.info(
                "Exported desktop results package: %s profile=%s metrics_selected=%d sections=%s html=%s pdf=%s csv=%s",
                export_dir,
                export_cfg.report_profile,
                len(export_cfg.selected_metric_keys),
                ",".join(export_cfg.include_sections),
                bool(export_cfg.write_html_report),
                bool(export_cfg.write_pdf_report),
                bool(export_cfg.write_csv_report),
            )
            QMessageBox.information(self, "Results Exported", f"Saved results package:\n{export_dir}")
        except Exception as exc:
            self.logger.exception("Results package export failed")
            QMessageBox.critical(self, "Results Export Error", str(exc))

    def _selected_history_records(self) -> list[DesktopRunRecord]:
        selected = self.history_list.selectedIndexes() if hasattr(self, "history_list") else []
        if not selected:
            return self.workflow.history()
        rows = sorted({int(idx.row()) for idx in selected})
        out: list[DesktopRunRecord] = []
        for row in rows:
            try:
                out.append(self.workflow.get(row))
            except Exception:
                continue
        return out

    def on_export_batch_results(self) -> None:
        records = self._selected_history_records()
        if not records:
            QMessageBox.warning(self, "No runs", "No history runs available for batch export.")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select batch export directory")
        if not out_dir:
            return
        corrected_map: dict[str, np.ndarray] = {}
        if self.state.current_run is not None and self.state.correction_session is not None:
            corrected_map[str(self.state.current_run.run_id)] = np.asarray(self.state.correction_session.current_mask)
            self._maybe_attach_current_correction()
        try:
            export_dir = self.result_exporter.export_batch(
                records,
                output_dir=out_dir,
                corrected_masks=corrected_map,
                annotator=self.annotator_edit.text().strip() or "unknown",
                notes=self.notes_edit.text().strip(),
                class_map=self.state.class_map,
                config=self._results_export_config_from_ui(),
            )
            self.logger.info(
                "Exported batch results package: %s runs=%d profile=%s sections=%s",
                export_dir,
                len(records),
                self.report_profile_combo.currentText(),
                ",".join(self._selected_report_sections()),
            )
            QMessageBox.information(self, "Batch Results Exported", f"Saved batch results package:\n{export_dir}")
        except Exception as exc:
            self.logger.exception("Batch results package export failed")
            QMessageBox.critical(self, "Batch Results Export Error", str(exc))

    def on_open_appearance_settings(self) -> None:
        dialog = AppearanceExportSettingsDialog(
            config=self._ui_config,
            source_path=self._ui_config_source,
            parent=self,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self._ui_config = dialog.selected_config()
        self._ui_config_source = dialog.selected_path()
        self._apply_style()
        self._apply_application_fonts()
        self.chk_report_html.setChecked(bool(self._ui_config.export_defaults.write_html_report))
        self.chk_report_pdf.setChecked(bool(self._ui_config.export_defaults.write_pdf_report))
        self.chk_report_csv.setChecked(bool(self._ui_config.export_defaults.write_csv_report))
        self.report_profile_combo.setCurrentText(str(self._ui_config.export_defaults.report_profile))
        self.report_top_k_spin.setValue(int(self._ui_config.export_defaults.top_k_key_metrics))
        self.chk_artifact_manifest.setChecked(bool(self._ui_config.export_defaults.include_artifact_manifest))
        for section, chk in self.report_section_checks.items():
            chk.setChecked(section in set(self._ui_config.export_defaults.include_sections))
        self.on_reset_profile_report_metrics()
        self.logger.info(
            "Applied UI settings from dialog source=%s base_font=%d heading_font=%d mono_font=%d high_contrast=%s",
            self._ui_config_source or "<unspecified>",
            int(self._ui_config.appearance.base_font_size),
            int(self._ui_config.appearance.heading_font_size),
            int(self._ui_config.appearance.monospace_font_size),
            bool(self._ui_config.appearance.high_contrast),
        )

    def on_save_project(self) -> None:
        run = self.state.current_run
        sess = self.state.correction_session
        if run is None or sess is None:
            QMessageBox.warning(self, "No run", "Run segmentation first")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select project save directory")
        if not out_dir:
            return
        self._maybe_attach_current_correction()
        try:
            req = ProjectSaveRequest(
                record=run,
                corrected_mask=sess.current_mask,
                class_map=self.state.class_map,
                annotator=self.annotator_edit.text().strip(),
                notes=self.notes_edit.text().strip(),
                ui_state={
                    "tool": self.tool_combo.currentText(),
                    "mode": self.mode_combo.currentText(),
                    "class_index": self._selected_class_index(),
                    "radius": self.radius_spin.value(),
                    "show_pred": self.chk_pred.isChecked(),
                    "show_corr": self.chk_corr.isChecked(),
                    "show_diff": self.chk_diff.isChecked(),
                    "pred_alpha": self.slider_pred.value(),
                    "corr_alpha": self.slider_corr.value(),
                    "diff_alpha": self.slider_diff.value(),
                    "config_path": self.config_path_edit.text().strip(),
                    "config_overrides": self.config_overrides_edit.text().strip(),
                    "conv_clip": float(self.conv_clip_spin.value()),
                    "conv_tile_x": int(self.conv_tile_x.value()),
                    "conv_tile_y": int(self.conv_tile_y.value()),
                    "conv_block_size": int(self.conv_block_spin.value()),
                    "conv_c": int(self.conv_c_spin.value()),
                    "conv_kernel_x": int(self.conv_kernel_x.value()),
                    "conv_kernel_y": int(self.conv_kernel_y.value()),
                    "conv_iterations": int(self.conv_iterations_spin.value()),
                    "conv_area_threshold": int(self.conv_area_spin.value()),
                    "conv_crop": bool(self.conv_crop_check.isChecked()),
                    "conv_crop_percent": int(self.conv_crop_percent.value()),
                    "results_orientation_bins": int(self.results_orientation_bins.value()),
                    "results_size_bins": int(self.results_size_bins.value()),
                    "results_min_feature": int(self.results_min_feature.value()),
                    "results_size_scale": self.results_size_scale.currentText(),
                    "results_cmap": self.results_cmap.currentText(),
                    "report_html": bool(self.chk_report_html.isChecked()),
                    "report_pdf": bool(self.chk_report_pdf.isChecked()),
                    "report_csv": bool(self.chk_report_csv.isChecked()),
                    "report_profile": self.report_profile_combo.currentText(),
                    "report_top_k": int(self.report_top_k_spin.value()),
                    "report_sections": list(self._selected_report_sections()),
                    "report_selected_metrics": list(self._selected_report_metric_keys()),
                    "report_include_artifact_manifest": bool(self.chk_artifact_manifest.isChecked()),
                    "report_advanced_enabled": bool(self.report_advanced_group.isChecked()),
                    "ui_config_source": self._ui_config_source,
                    "ui_base_font_size": int(self._ui_config.appearance.base_font_size),
                    "ui_heading_font_size": int(self._ui_config.appearance.heading_font_size),
                    "ui_monospace_font_size": int(self._ui_config.appearance.monospace_font_size),
                    "ui_menu_font_size": int(self._ui_config.appearance.menu_font_size),
                    "ui_tab_font_size": int(self._ui_config.appearance.tab_font_size),
                    "ui_toolbar_font_size": int(self._ui_config.appearance.toolbar_font_size),
                    "ui_status_font_size": int(self._ui_config.appearance.status_font_size),
                    "ui_control_padding_px": int(self._ui_config.appearance.control_padding_px),
                    "ui_panel_spacing_px": int(self._ui_config.appearance.panel_spacing_px),
                    "ui_table_row_padding_px": int(self._ui_config.appearance.table_row_padding_px),
                    "ui_table_min_row_height_px": int(self._ui_config.appearance.table_min_row_height_px),
                    "ui_high_contrast": bool(self._ui_config.appearance.high_contrast),
                    "ui_window_initial_width": int(self._ui_config.window.initial_width),
                    "ui_window_initial_height": int(self._ui_config.window.initial_height),
                    "ui_window_minimum_width": int(self._ui_config.window.minimum_width),
                    "ui_window_minimum_height": int(self._ui_config.window.minimum_height),
                    "ui_window_left_dock_width": int(self._ui_config.window.left_dock_width),
                    "ui_window_right_dock_width": int(self._ui_config.window.right_dock_width),
                    "ui_window_workflow_dock_width": int(self._ui_config.window.workflow_dock_width),
                    "ui_window_remember_geometry": bool(self._ui_config.window.remember_geometry),
                    "ui_window_clamp_to_screen": bool(self._ui_config.window.clamp_to_screen),
                    "ui_window_start_maximized": bool(self._ui_config.window.start_maximized),
                    "ui_window_start_fullscreen": bool(self._ui_config.window.start_fullscreen),
                    "calibration": (
                        None if self.state.spatial_calibration is None else self.state.spatial_calibration.as_dict()
                    ),
                    "calibration_image_path": self.state.calibration_image_path or "",
                },
            )
            out = self.project_store.save(req, out_dir)
            self.logger.info("Project session saved: %s", out)
            QMessageBox.information(self, "Session Saved", f"Saved project state in:\n{out}")
        except Exception as exc:
            self.logger.exception("Project save failed")
            QMessageBox.critical(self, "Save Error", str(exc))

    def on_load_project(self) -> None:
        project_dir = QFileDialog.getExistingDirectory(self, "Select project directory")
        if not project_dir:
            return
        try:
            loaded = self.project_store.load(project_dir)
            self.state.class_map = loaded.class_map
            self._reload_class_combo()
            self.annotator_edit.setText(loaded.annotator)
            self.notes_edit.setText(loaded.notes)
            self.config_path_edit.setText(str(loaded.ui_state.get("config_path", "")))
            self.config_overrides_edit.setText(str(loaded.ui_state.get("config_overrides", "")))

            self.workflow.append_history(loaded.record)
            self._add_record(loaded.record)
            self._show_record(loaded.record, corrected_mask=loaded.corrected_mask)
            self.history_list.setCurrentRow(self.history_list.count() - 1)

            self.tool_combo.setCurrentText(str(loaded.ui_state.get("tool", "brush")))
            self.mode_combo.setCurrentText(str(loaded.ui_state.get("mode", "add")))
            self.radius_spin.setValue(int(loaded.ui_state.get("radius", 6)))
            self.chk_pred.setChecked(bool(loaded.ui_state.get("show_pred", True)))
            self.chk_corr.setChecked(bool(loaded.ui_state.get("show_corr", True)))
            self.chk_diff.setChecked(bool(loaded.ui_state.get("show_diff", True)))
            self.slider_pred.setValue(int(loaded.ui_state.get("pred_alpha", 35)))
            self.slider_corr.setValue(int(loaded.ui_state.get("corr_alpha", 45)))
            self.slider_diff.setValue(int(loaded.ui_state.get("diff_alpha", 70)))
            self.conv_clip_spin.setValue(float(loaded.ui_state.get("conv_clip", self.conv_clip_spin.value())))
            self.conv_tile_x.setValue(int(loaded.ui_state.get("conv_tile_x", self.conv_tile_x.value())))
            self.conv_tile_y.setValue(int(loaded.ui_state.get("conv_tile_y", self.conv_tile_y.value())))
            self.conv_block_spin.setValue(int(loaded.ui_state.get("conv_block_size", self.conv_block_spin.value())))
            self.conv_c_spin.setValue(int(loaded.ui_state.get("conv_c", self.conv_c_spin.value())))
            self.conv_kernel_x.setValue(int(loaded.ui_state.get("conv_kernel_x", self.conv_kernel_x.value())))
            self.conv_kernel_y.setValue(int(loaded.ui_state.get("conv_kernel_y", self.conv_kernel_y.value())))
            self.conv_iterations_spin.setValue(int(loaded.ui_state.get("conv_iterations", self.conv_iterations_spin.value())))
            self.conv_area_spin.setValue(int(loaded.ui_state.get("conv_area_threshold", self.conv_area_spin.value())))
            self.conv_crop_check.setChecked(bool(loaded.ui_state.get("conv_crop", self.conv_crop_check.isChecked())))
            self.conv_crop_percent.setValue(int(loaded.ui_state.get("conv_crop_percent", self.conv_crop_percent.value())))
            self.results_orientation_bins.setValue(
                int(loaded.ui_state.get("results_orientation_bins", self.results_orientation_bins.value()))
            )
            self.results_size_bins.setValue(int(loaded.ui_state.get("results_size_bins", self.results_size_bins.value())))
            self.results_min_feature.setValue(int(loaded.ui_state.get("results_min_feature", self.results_min_feature.value())))
            self.results_size_scale.setCurrentText(
                str(loaded.ui_state.get("results_size_scale", self.results_size_scale.currentText()))
            )
            self.results_cmap.setCurrentText(str(loaded.ui_state.get("results_cmap", self.results_cmap.currentText())))
            self.chk_report_html.setChecked(bool(loaded.ui_state.get("report_html", self.chk_report_html.isChecked())))
            self.chk_report_pdf.setChecked(bool(loaded.ui_state.get("report_pdf", self.chk_report_pdf.isChecked())))
            self.chk_report_csv.setChecked(bool(loaded.ui_state.get("report_csv", self.chk_report_csv.isChecked())))
            self.report_profile_combo.setCurrentText(
                str(loaded.ui_state.get("report_profile", self.report_profile_combo.currentText()))
            )
            self.report_top_k_spin.setValue(int(loaded.ui_state.get("report_top_k", self.report_top_k_spin.value())))
            self.chk_artifact_manifest.setChecked(
                bool(loaded.ui_state.get("report_include_artifact_manifest", self.chk_artifact_manifest.isChecked()))
            )
            self.report_advanced_group.setChecked(
                bool(loaded.ui_state.get("report_advanced_enabled", self.report_advanced_group.isChecked()))
            )
            loaded_sections = loaded.ui_state.get("report_sections", [])
            if isinstance(loaded_sections, list):
                loaded_section_set = {str(v) for v in loaded_sections}
                for name, chk in self.report_section_checks.items():
                    chk.setChecked(name in loaded_section_set)
            loaded_metrics = loaded.ui_state.get("report_selected_metrics", [])
            if isinstance(loaded_metrics, list):
                self._refresh_report_metric_checklist(
                    [str(v) for v in loaded_metrics if str(v).strip()],
                    selected=tuple(str(v) for v in loaded_metrics if str(v).strip()),
                )

            ui_base_font_size = loaded.ui_state.get("ui_base_font_size")
            if ui_base_font_size is not None:
                exp_defaults = self._ui_config.export_defaults
                self._ui_config = DesktopUIConfig(
                    schema_version="microseg.desktop_ui_config.v1",
                    appearance=DesktopAppearanceConfig(
                        base_font_size=int(loaded.ui_state.get("ui_base_font_size", self._ui_config.appearance.base_font_size)),
                        heading_font_size=int(loaded.ui_state.get("ui_heading_font_size", self._ui_config.appearance.heading_font_size)),
                        monospace_font_size=int(
                            loaded.ui_state.get("ui_monospace_font_size", self._ui_config.appearance.monospace_font_size)
                        ),
                        menu_font_size=int(loaded.ui_state.get("ui_menu_font_size", self._ui_config.appearance.menu_font_size)),
                        tab_font_size=int(loaded.ui_state.get("ui_tab_font_size", self._ui_config.appearance.tab_font_size)),
                        toolbar_font_size=int(
                            loaded.ui_state.get("ui_toolbar_font_size", self._ui_config.appearance.toolbar_font_size)
                        ),
                        status_font_size=int(
                            loaded.ui_state.get("ui_status_font_size", self._ui_config.appearance.status_font_size)
                        ),
                        control_padding_px=int(
                            loaded.ui_state.get("ui_control_padding_px", self._ui_config.appearance.control_padding_px)
                        ),
                        panel_spacing_px=int(
                            loaded.ui_state.get("ui_panel_spacing_px", self._ui_config.appearance.panel_spacing_px)
                        ),
                        table_row_padding_px=int(
                            loaded.ui_state.get("ui_table_row_padding_px", self._ui_config.appearance.table_row_padding_px)
                        ),
                        table_min_row_height_px=int(
                            loaded.ui_state.get(
                                "ui_table_min_row_height_px",
                                self._ui_config.appearance.table_min_row_height_px,
                            )
                        ),
                        high_contrast=bool(loaded.ui_state.get("ui_high_contrast", self._ui_config.appearance.high_contrast)),
                    ),
                    window=DesktopWindowConfig(
                        initial_width=int(loaded.ui_state.get("ui_window_initial_width", self._ui_config.window.initial_width)),
                        initial_height=int(loaded.ui_state.get("ui_window_initial_height", self._ui_config.window.initial_height)),
                        minimum_width=int(loaded.ui_state.get("ui_window_minimum_width", self._ui_config.window.minimum_width)),
                        minimum_height=int(loaded.ui_state.get("ui_window_minimum_height", self._ui_config.window.minimum_height)),
                        left_dock_width=int(loaded.ui_state.get("ui_window_left_dock_width", self._ui_config.window.left_dock_width)),
                        right_dock_width=int(loaded.ui_state.get("ui_window_right_dock_width", self._ui_config.window.right_dock_width)),
                        workflow_dock_width=int(
                            loaded.ui_state.get("ui_window_workflow_dock_width", self._ui_config.window.workflow_dock_width)
                        ),
                        remember_geometry=bool(
                            loaded.ui_state.get("ui_window_remember_geometry", self._ui_config.window.remember_geometry)
                        ),
                        clamp_to_screen=bool(
                            loaded.ui_state.get("ui_window_clamp_to_screen", self._ui_config.window.clamp_to_screen)
                        ),
                        start_maximized=bool(
                            loaded.ui_state.get("ui_window_start_maximized", self._ui_config.window.start_maximized)
                        ),
                        start_fullscreen=bool(
                            loaded.ui_state.get("ui_window_start_fullscreen", self._ui_config.window.start_fullscreen)
                        ),
                        show_workflow_dock_on_start=bool(self._ui_config.window.show_workflow_dock_on_start),
                        show_log_dock_on_start=bool(self._ui_config.window.show_log_dock_on_start),
                    ),
                    export_defaults=exp_defaults,
                )
                self._apply_style()
                self._apply_application_fonts()
            self._ui_config_source = str(loaded.ui_state.get("ui_config_source", self._ui_config_source or ""))
            cal_payload = loaded.ui_state.get("calibration")
            cal_obj = None
            if isinstance(cal_payload, dict):
                try:
                    cal_obj = SpatialCalibration.from_dict(cal_payload)
                except Exception:
                    cal_obj = None
            self.state.calibration_image_path = str(
                loaded.ui_state.get("calibration_image_path", loaded.record.image_path)
            )
            self._apply_calibration(cal_obj, image_path=self.state.calibration_image_path)

            wanted_cls = int(loaded.ui_state.get("class_index", 1))
            for i in range(self.class_combo.count()):
                if self.class_combo.itemText(i).startswith(f"{wanted_cls}:"):
                    self.class_combo.setCurrentIndex(i)
                    break
            self._update_layer_settings()
            self._update_results_dashboard()

            self.logger.info("Loaded project session from %s", loaded.root_dir)
            QMessageBox.information(self, "Session Loaded", f"Loaded project:\n{loaded.root_dir}")
        except Exception as exc:
            self.logger.exception("Project load failed")
            QMessageBox.critical(self, "Load Error", str(exc))

    def on_package_dataset(self) -> None:
        input_dir = self.dataset_input_edit.text().strip()
        output_dir = self.dataset_output_edit.text().strip()
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Missing paths", "Set corrections input and dataset output directories")
            return
        overrides = [
            f"train_ratio={float(self.train_ratio_spin.value())}",
            f"val_ratio={float(self.val_ratio_spin.value())}",
            f"seed={int(self.seed_spin.value())}",
        ]
        command = self.orchestrator.package(
            config=None,
            overrides=overrides,
            input_dir=input_dir,
            output_dir=output_dir,
        )
        self._start_orchestration_job(command, "Packaging")
