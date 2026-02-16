"""Qt desktop GUI for segmentation, correction, and correction export."""

from __future__ import annotations

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
    QTextEdit,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from hydride_segmentation.version import __version__
from src.microseg.app import OrchestrationCommandBuilder, ProjectSaveRequest, ProjectStateStore
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
        self.orch_train_backend.addItems(["unet_binary", "torch_pixel", "sklearn_pixel"])
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

        self.workflow_notes = QTextEdit()
        self.workflow_notes.setReadOnly(True)
        self.workflow_notes.setPlainText(
            "Orchestration Log\\n"
            "- One active job at a time\\n"
            "- Commands run through scripts/microseg_cli.py\\n"
            "- Use YAML config + overrides for reproducibility.\\n"
            "- GPU is opt-in; fallback to CPU is automatic if unavailable."
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
        if spec.get("application_remarks"):
            lines.append(f"<b>Application:</b> {spec.get('application_remarks')}")
        if spec.get("short_description"):
            lines.append(f"<b>User tip:</b> {spec.get('short_description')}")
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
        command = self.orchestrator.train(
            config=self.orch_train_config_edit.text().strip() or None,
            overrides=overrides,
            dataset_dir=self.orch_train_dataset_edit.text().strip() or None,
            output_dir=self.orch_train_output_edit.text().strip() or None,
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
            "7. Use Workflow Hub for train/infer/evaluate/package orchestration jobs.",
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
