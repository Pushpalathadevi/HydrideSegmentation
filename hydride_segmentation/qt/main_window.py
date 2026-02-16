"""Qt desktop GUI for segmentation, correction, and correction export."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from skimage.draw import disk, line

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QImage, QKeySequence, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from hydride_segmentation.version import __version__
from src.microseg.app.desktop_workflow import DesktopRunRecord, DesktopWorkflowManager
from src.microseg.corrections import CorrectionExporter, CorrectionSession
from src.microseg.ui import AnnotationLayerSettings, compose_annotation_view
from src.microseg.utils import to_rgb


@dataclass
class _UiState:
    image_path: str | None = None
    current_run: DesktopRunRecord | None = None
    correction_session: CorrectionSession | None = None


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


def _mask_to_pixmap(mask: np.ndarray) -> QPixmap:
    m = (mask > 0).astype(np.uint8) * 255
    return _rgb_to_pixmap(np.stack([m] * 3, axis=-1))


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

    def bind(self, base_image: np.ndarray, predicted_mask: np.ndarray, session: CorrectionSession) -> None:
        self._base_image = base_image
        self._predicted_mask = (predicted_mask > 0).astype(np.uint8) * 255
        self._session = session
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
            push_undo=push_undo,
            record_action=record_action,
        )
        self._refresh()
        self.correction_changed.emit()

    def _finish_polygon(self) -> None:
        if self._session is None:
            return
        if len(self._poly_points) >= 3:
            self._session.apply_polygon(self._poly_points, mode=self._mode)
            self.correction_changed.emit()
        self._poly_points = []
        self._refresh()

    def _finish_lasso(self) -> None:
        if self._session is None:
            return
        if len(self._lasso_points) >= 3:
            self._session.apply_polygon(self._lasso_points, mode=self._mode)
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
        self.state = _UiState()

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
        controls.addWidget(self.model_combo)

        self.btn_run = QPushButton("Run Segmentation")
        self.btn_run.clicked.connect(self.on_run_segmentation)
        controls.addWidget(self.btn_run)

        self.btn_batch = QPushButton("Run Batch")
        self.btn_batch.clicked.connect(self.on_run_batch)
        controls.addWidget(self.btn_batch)

        self.corrected_canvas = CorrectedMaskCanvas()
        self.corrected_canvas.zoom_changed.connect(self._on_zoom_changed)
        self.corrected_canvas.cursor_changed.connect(self._on_cursor_changed)
        self.corrected_canvas.correction_changed.connect(self._on_correction_changed)

        tool_row = QHBoxLayout()
        layout.addLayout(tool_row)

        tool_row.addWidget(QLabel("Tool"))
        self.tool_combo = QComboBox()
        self.tool_combo.addItems(["brush", "polygon", "lasso"])
        self.tool_combo.currentTextChanged.connect(self._on_tool_changed)
        tool_row.addWidget(self.tool_combo)

        tool_row.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["add", "erase"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        tool_row.addWidget(self.mode_combo)

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

        self.btn_export = QPushButton("Export Corrected Sample")
        self.btn_export.clicked.connect(self.on_export_correction)
        layer_row.addWidget(self.btn_export)

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

        self._connect_scroll_sync()

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
            "3. Select tool/mode and adjust brush size.\n"
            "4. Tune layer transparency to inspect differences.\n"
            "5. Export corrected sample with annotator and notes.",
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
        model_name = self.model_combo.currentText()

        try:
            self.logger.info("Running segmentation on %s with %s", path, model_name)
            record = self.workflow.run_single(
                path,
                model_name=model_name,
                params={"image_path": path},
                include_analysis=True,
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
        model_name = self.model_combo.currentText()
        try:
            self.logger.info("Running batch of %d images with %s", len(paths), model_name)
            records = self.workflow.run_batch(
                list(paths),
                model_name=model_name,
                params={"image_path": paths[0]},
                include_analysis=False,
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

    def _show_record(self, record: DesktopRunRecord) -> None:
        self.state.current_run = record
        base = np.array(record.input_image)
        pred_mask = np.array(record.mask_image)
        self.state.correction_session = CorrectionSession(pred_mask)

        self._set_image_preview(self.input_label, base)
        self._set_image_preview(self.mask_label, _mask_to_pixmap(pred_mask))
        self._set_image_preview(self.overlay_label, np.array(record.overlay_image))

        self.corrected_canvas.bind(base, pred_mask, self.state.correction_session)
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
            )
            self.logger.info("Exported corrected sample: %s", sample_dir)
            QMessageBox.information(self, "Export complete", f"Saved to:\n{sample_dir}")
        except Exception as exc:
            self.logger.exception("Correction export failed")
            QMessageBox.critical(self, "Export Error", str(exc))
