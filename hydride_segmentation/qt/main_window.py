"""Qt desktop GUI for segmentation, correction, and correction export."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage.draw import disk, line

from PySide6.QtCore import QProcess, Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
    QHeaderView,
)

from hydride_segmentation.version import __version__
from src.microseg.app import (
    OrchestrationCommandBuilder,
    ProjectSaveRequest,
    ProjectStateStore,
    compare_run_reports,
    read_workflow_profile,
    summarize_run_report,
    write_workflow_profile,
)
from src.microseg.app.desktop_workflow import DesktopRunRecord, DesktopWorkflowManager
from src.microseg.corrections import (
    DEFAULT_CLASS_MAP,
    CorrectionExporter,
    CorrectionSession,
    SegmentationClass,
    SegmentationClassMap,
    colorize_index_mask,
    to_index_mask,
)
from src.microseg.dataops import (
    DatasetPrepareConfig,
    DatasetQualityConfig,
    preview_training_dataset_layout,
    prepare_training_dataset_layout,
    run_dataset_quality_checks,
)
from src.microseg.io import resolve_config
from src.microseg.ui import AnnotationLayerSettings, compose_annotation_view
from src.microseg.utils import to_rgb


@dataclass
class _UiState:
    image_path: str | None = None
    current_run: DesktopRunRecord | None = None
    correction_session: CorrectionSession | None = None
    class_map: SegmentationClassMap = field(default_factory=lambda: DEFAULT_CLASS_MAP)


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


class _UiLogHandler(logging.Handler):
    """Logging handler that forwards records to a GUI callback."""

    def __init__(self, emit_callback):
        super().__init__()
        self.emit_callback = emit_callback

    def emit(self, record):  # noqa: D401
        msg = self.format(record)
        self.emit_callback(msg)


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


class QtSegmentationMainWindow(QMainWindow):
    """Qt main window for phase-3 correction workflow."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"MicroSeg Desktop v{__version__}")
        self.resize(1700, 1050)

        self.workflow = DesktopWorkflowManager(max_history=400)
        self.exporter = CorrectionExporter()
        self.project_store = ProjectStateStore()
        self.orchestrator = OrchestrationCommandBuilder.discover(start=Path(__file__))
        self._job_process: QProcess | None = None
        self._job_name: str = ""
        self._dataset_preview_rows: list[dict[str, object]] = []
        self._dataset_preview_split_counts: dict[str, int] = {}
        self._last_dataset_qa_ok: bool | None = None
        self._last_dataset_qa_dir: str = ""
        self._review_summary_a = None
        self._review_summary_b = None
        self.state = _UiState()
        self._model_specs = {spec["display_name"]: spec for spec in self.workflow.model_specs()}

        self._sync_scroll_guard = False
        self.logger = logging.getLogger("MicroSegQtGUI")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            stream = logging.StreamHandler()
            stream.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self.logger.addHandler(stream)

        self._build_ui()
        self._configure_menu()
        self._bind_shortcuts()
        self._apply_style()

        self._ui_handler = _UiLogHandler(self._log)
        self._ui_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        self.logger.addHandler(self._ui_handler)

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QWidget { font-size: 12px; }
            QMainWindow { background: #f3f5f7; }
            QPushButton { padding: 6px 10px; }
            QLineEdit, QComboBox, QSpinBox { padding: 4px; }
            QPlainTextEdit { background: #10161f; color: #d8e1ea; font-family: Menlo, Monaco, monospace; }
            QListWidget { background: #ffffff; }
            """
        )

    def _configure_menu(self) -> None:
        menu = self.menuBar()

        file_menu = menu.addMenu("File")
        act_open = QAction("Open Image", self)
        act_open.triggered.connect(self.on_load_image)
        file_menu.addAction(act_open)

        act_batch = QAction("Open Batch", self)
        act_batch.triggered.connect(self.on_run_batch)
        file_menu.addAction(act_batch)

        act_export = QAction("Export Corrected Sample", self)
        act_export.triggered.connect(self.on_export_correction)
        file_menu.addAction(act_export)

        act_save_project = QAction("Save Project Session", self)
        act_save_project.triggered.connect(self.on_save_project)
        file_menu.addAction(act_save_project)

        act_load_project = QAction("Load Project Session", self)
        act_load_project.triggered.connect(self.on_load_project)
        file_menu.addAction(act_load_project)

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

        help_menu = menu.addMenu("Help")
        help_menu.addAction("Shortcuts", self.on_show_shortcuts)
        help_menu.addAction("Guide", self.on_show_guide)
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

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.workflow.model_options())
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        controls.addWidget(self.model_combo)

        self.btn_run = QPushButton("Run Segmentation")
        self.btn_run.clicked.connect(self.on_run_segmentation)
        controls.addWidget(self.btn_run)

        self.btn_batch = QPushButton("Run Batch")
        self.btn_batch.clicked.connect(self.on_run_batch)
        controls.addWidget(self.btn_batch)

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
        layer_row.addWidget(self.notes_edit, stretch=2)

        self.chk_fmt_indexed = QCheckBox("indexed")
        self.chk_fmt_indexed.setChecked(True)
        layer_row.addWidget(self.chk_fmt_indexed)

        self.chk_fmt_color = QCheckBox("color")
        self.chk_fmt_color.setChecked(True)
        layer_row.addWidget(self.chk_fmt_color)

        self.chk_fmt_npy = QCheckBox("npy")
        self.chk_fmt_npy.setChecked(False)
        layer_row.addWidget(self.chk_fmt_npy)

        self.btn_export = QPushButton("Export Corrected Sample")
        self.btn_export.clicked.connect(self.on_export_correction)
        layer_row.addWidget(self.btn_export)

        self.btn_save_project = QPushButton("Save Session")
        self.btn_save_project.clicked.connect(self.on_save_project)
        layer_row.addWidget(self.btn_save_project)

        self.btn_load_project = QPushButton("Load Session")
        self.btn_load_project.clicked.connect(self.on_load_project)
        layer_row.addWidget(self.btn_load_project)

        body = QHBoxLayout()
        layout.addLayout(body, stretch=1)

        self.history_list = QListWidget()
        self.history_list.currentRowChanged.connect(self.on_history_selected)
        body.addWidget(self.history_list, stretch=1)

        self.tabs = QTabWidget()
        body.addWidget(self.tabs, stretch=6)

        self.input_label = QLabel("Input")
        self.mask_label = QLabel("Prediction")
        self.overlay_label = QLabel("Overlay")

        self.tabs.addTab(self._in_scroll(self.input_label), "Input")
        self.tabs.addTab(self._in_scroll(self.mask_label), "Predicted Mask")
        self.tabs.addTab(self._in_scroll(self.overlay_label), "Overlay")

        self.split_widget = QWidget()
        split_layout = QHBoxLayout(self.split_widget)
        split_layout.setContentsMargins(0, 0, 0, 0)

        self.splitter = QSplitter(Qt.Horizontal)
        split_layout.addWidget(self.splitter)

        self.raw_corr_label = QLabel("Raw")
        self.raw_corr_label.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.raw_corr_scroll = self._in_scroll(self.raw_corr_label)
        self.corrected_scroll = self._in_scroll(self.corrected_canvas)

        self.splitter.addWidget(self.raw_corr_scroll)
        self.splitter.addWidget(self.corrected_scroll)
        self.splitter.setSizes([700, 900])

        self.tabs.addTab(self.split_widget, "Correction Split View")

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
            "unet_binary,hf_segformer_b0,hf_segformer_b2,transunet_tiny,segformer_mini,torch_pixel"
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

        bind_pair(self.raw_corr_scroll, self.corrected_scroll)
        bind_pair(self.corrected_scroll, self.raw_corr_scroll)

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

    def _on_model_changed(self, model_name: str) -> None:
        spec = self._model_specs.get(model_name)
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
        self._set_image_preview(self.raw_corr_label, np.array(run.input_image), zoom=self.corrected_canvas.zoom_value())

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
            "1. Run segmentation.\n"
            "2. Open 'Correction Split View'.\n"
            "3. Pick class index/color map and select tool/mode.\n"
            "4. For wrong objects: feature-select + erase to delete component.\n"
            "5. Redraw with brush/polygon/lasso in add mode.\n"
            "6. Tune layer transparency and export indexed/color/npy masks.\n"
            "7. Use Workflow Hub for train/infer/evaluate/package orchestration jobs.\n"
            "8. Use Dataset Prep + QA for split preview, colormap conversion, and QA gating.\n"
            "9. Use Run Review to compare training/evaluation reports.\n"
            "10. Use HPC GA Planner to generate Slurm/PBS/local job bundles.",
        )

    def on_show_about(self) -> None:
        QMessageBox.information(
            self,
            "About MicroSeg Desktop",
            f"MicroSeg Desktop v{__version__}\n"
            "Qt-based local application for segmentation review and correction\n"
            "Designed for field deployment workflows.",
        )

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
        self.state.image_path = path
        self.logger.info("Loaded image path: %s", path)

    def on_run_segmentation(self) -> None:
        path = self.path_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Missing image", "Select an image first")
            return

        try:
            cfg = self._resolve_run_config()
            model_name = cfg.get("model_name", self.model_combo.currentText())
            include_analysis = bool(cfg.get("include_analysis", True))
            params = dict(cfg.get("params", {}))
            params["image_path"] = path
            self.logger.info("Running segmentation on %s with %s", path, model_name)
            record = self.workflow.run_single(
                path,
                model_name=model_name,
                params=params,
                include_analysis=include_analysis,
            )
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
        try:
            cfg = self._resolve_run_config()
            model_name = cfg.get("model_name", self.model_combo.currentText())
            include_analysis = bool(cfg.get("include_analysis", False))
            params = dict(cfg.get("params", {}))
            params.setdefault("image_path", paths[0])
            self.logger.info("Running batch of %d images with %s", len(paths), model_name)
            records = self.workflow.run_batch(
                list(paths),
                model_name=model_name,
                params=params,
                include_analysis=include_analysis,
            )
            for rec in records:
                self._add_record(rec)
            if records:
                self._show_record(records[-1])
                self.history_list.setCurrentRow(self.history_list.count() - 1)
        except Exception as exc:
            self.logger.exception("Batch run failed")
            QMessageBox.critical(self, "Batch Error", str(exc))

    def _add_record(self, record: DesktopRunRecord) -> None:
        self.history_list.addItem(record.history_label)

    def _show_record(self, record: DesktopRunRecord, corrected_mask: np.ndarray | None = None) -> None:
        self.state.current_run = record
        self.path_edit.setText(record.image_path)
        if self.model_combo.findText(record.model_name) >= 0:
            self.model_combo.setCurrentText(record.model_name)
        base = np.array(record.input_image)
        pred_mask = to_index_mask(np.array(record.mask_image))
        self.state.correction_session = CorrectionSession(pred_mask)
        if corrected_mask is not None:
            self.state.correction_session.current_mask = to_index_mask(corrected_mask)

        self._set_image_preview(self.input_label, base)
        self._set_image_preview(self.mask_label, _mask_to_pixmap(pred_mask, self.state.class_map))
        self._set_image_preview(self.overlay_label, np.array(record.overlay_image))

        self.corrected_canvas.bind(base, pred_mask, self.state.correction_session, self.state.class_map)
        self.corrected_canvas.update_layer_settings(self._layer_settings())
        self._update_split_input_view()
        self._update_action_label()

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
            self.logger.info("Correction undo")

    def on_redo(self) -> None:
        sess = self.state.correction_session
        if sess is None:
            return
        if sess.redo():
            self.corrected_canvas._refresh()
            self._update_action_label()
            self.logger.info("Correction redo")

    def on_reset_corrections(self) -> None:
        sess = self.state.correction_session
        if sess is None:
            return
        sess.reset_to_initial()
        self.corrected_canvas._refresh()
        self._update_action_label()
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
            sample_dir = self.exporter.export_sample(
                run,
                sess.current_mask,
                out_dir,
                annotator=self.annotator_edit.text().strip() or "unknown",
                notes=self.notes_edit.text().strip(),
                class_map=self.state.class_map,
                formats=self._selected_export_formats(),
            )
            self.logger.info("Exported corrected sample: %s", sample_dir)
            QMessageBox.information(self, "Export complete", f"Saved to:\n{sample_dir}")
        except Exception as exc:
            self.logger.exception("Correction export failed")
            QMessageBox.critical(self, "Export Error", str(exc))

    def on_save_project(self) -> None:
        run = self.state.current_run
        sess = self.state.correction_session
        if run is None or sess is None:
            QMessageBox.warning(self, "No run", "Run segmentation first")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "Select project save directory")
        if not out_dir:
            return
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

            wanted_cls = int(loaded.ui_state.get("class_index", 1))
            for i in range(self.class_combo.count()):
                if self.class_combo.itemText(i).startswith(f"{wanted_cls}:"):
                    self.class_combo.setCurrentIndex(i)
                    break
            self._update_layer_settings()

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
