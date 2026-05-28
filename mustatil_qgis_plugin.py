import os
import sys
import tempfile
import subprocess
from datetime import datetime
import re
import json
from pathlib import Path

from qgis.PyQt.QtCore import Qt, QProcess, QSize, QTimer, QPointF
from qgis.PyQt.QtGui import QIcon, QColor, QPixmap, QPainter, QPen, QWheelEvent, QImage
from qgis.PyQt.QtWidgets import (
    QAction, QMenu, QToolBar, QDockWidget, QWidget, QVBoxLayout, QHBoxLayout, QApplication,
    QLabel, QPushButton, QTextEdit, QFileDialog, QGroupBox, QFormLayout,
    QProgressBar, QMessageBox, QSpinBox, QDoubleSpinBox, QCheckBox, QTabWidget,
    QLineEdit, QComboBox, QScrollArea, QSlider, QListWidget, QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
)
from qgis.core import (
    QgsProject, QgsRasterLayer, QgsVectorLayer, QgsRectangle, QgsWkbTypes, QgsCoordinateReferenceSystem, QgsCoordinateTransform,
    QgsPointXY, QgsRasterFileWriter, QgsMapLayerProxyModel, QgsFillSymbol, QgsMapSettings, QgsMapRendererParallelJob,
    QgsVectorFileWriter, QgsCoordinateTransformContext
)
from qgis.gui import QgsMapToolEmitPoint, QgsRubberBand, QgsMapLayerComboBox

MENU_TITLE = "&Mustatil AI"
TOOLBAR_OBJECT_NAME = "MustatilAIToolbar"

class RectTool(QgsMapToolEmitPoint):
    def __init__(self, canvas, callback):
        super().__init__(canvas)
        self.canvas = canvas
        self.callback = callback
        self.start = None
        self.rubber = QgsRubberBand(canvas, QgsWkbTypes.LineGeometry)
        self.rubber.setColor(QColor(255, 0, 0, 255))
        self.rubber.setWidth(2)

    def canvasPressEvent(self, e):
        self.start = self.toMapCoordinates(e.pos())
        self.clear()

    def canvasMoveEvent(self, e):
        if self.start is None:
            return
        end = self.toMapCoordinates(e.pos())
        self._draw_box(QgsRectangle(self.start, end))

    def canvasReleaseEvent(self, e):
        if self.start is None:
            return
        end = self.toMapCoordinates(e.pos())
        rect = QgsRectangle(self.start, end)
        self._draw_box(rect)
        self.callback(rect)
        self.start = None

    def clear(self):
        try:
            self.rubber.reset(QgsWkbTypes.LineGeometry)
        except Exception:
            pass

    def deactivate(self):
        self.clear()
        super().deactivate()

    def _draw_box(self, rect):
        self.rubber.reset(QgsWkbTypes.LineGeometry)
        x1 = rect.xMinimum()
        x2 = rect.xMaximum()
        y1 = rect.yMinimum()
        y2 = rect.yMaximum()

        points = [
            QgsPointXY(x1, y2),
            QgsPointXY(x2, y2),
            QgsPointXY(x2, y1),
            QgsPointXY(x1, y1),
            QgsPointXY(x1, y2),
        ]

        for i, p in enumerate(points):
            self.rubber.addPoint(p, i == len(points) - 1)

        self.rubber.show()
        self.canvas.refresh()

class MustatilDock(QDockWidget):
    def __init__(self, iface, plugin_dir):
        super().__init__("Mustatil AI Detection", iface.mainWindow())
        self.iface = iface
        self.plugin_dir = Path(plugin_dir)

        self.runtime_python_txt = self.plugin_dir / "runtime_python.txt"
        self.canvas = iface.mapCanvas()
        self.rect_tool = None
        self.last_clip = ""
        self.last_output = ""
        self.last_filtered_layer_id = ""
        self.process = None
        self.download_tile_start_time = None
        self.download_tile_last = 0
        self.detection_heartbeat_value = 0
        self.detection_heartbeat_timer = QTimer(self)
        self.detection_heartbeat_timer.timeout.connect(self.detection_heartbeat_tick)

        self.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        self._build_ui()

    def detect_existing_runtime(self):
        """
        Detect runtime python path from runtime_python.txt or fallback paths.
        """
        try:
            if self.runtime_python_txt.exists():
                p = self.runtime_python_txt.read_text(encoding="utf-8").strip()
                if p and Path(p).exists():
                    return p
        except Exception:
            pass

        embedded = self.plugin_dir / "runtime" / "python" / "python.exe"
        if embedded.exists():
            return str(embedded)

        return ""

    def _build_ui(self):
        outer = QWidget()
        root = QVBoxLayout(outer)

        title = QLabel("<b>Mustatil AI Detection</b><br>Built-in models with YOLO/ONNX models")
        root.addWidget(title)

        top_buttons = QHBoxLayout()
        self.install_runtime_top = QPushButton("Install Runtime")
        self.install_runtime_top.clicked.connect(self.install_runtime)
        top_buttons.addWidget(self.install_runtime_top)
        top_buttons.addStretch(1)
        root.addLayout(top_buttons)

        self.tabs = QTabWidget()
        root.addWidget(self.tabs)

        self.detect_tab = self._make_detection_tab()
        self.training_tab = self._make_training_tab()
        self.annotator_tab = self._make_annotator_tab()
        self.toolkit_tab = self._make_toolkit_tab()
        self.runtime_tab = self.toolkit_tab  # runtime controls are embedded in 
        self.log_tab = self._make_log_tab()

        self.tabs.addTab(self.detect_tab, "Detection")
        self.tabs.addTab(self.training_tab, "YOLO Trainer")
        self.tabs.addTab(self.annotator_tab, "Annotator")
        self.tabs.addTab(self.toolkit_tab, "Toolkit / Runtime")
        self.tabs.addTab(self.log_tab, "Log")

        # Now all UI fields exist.
        self.update_preset_fields()

        self.setWidget(outer)

    def _wrap_scroll(self, widget):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        return scroll

    def _make_detection_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        layer_group = QGroupBox("Input")
        layer_form = QFormLayout(layer_group)
        self.layer_combo = QgsMapLayerComboBox()
        self.layer_combo.setFilters(QgsMapLayerProxyModel.RasterLayer)
        layer_form.addRow("Raster layer", self.layer_combo)

        self.output_dir = QLineEdit(str(Path.home() / "Desktop"))
        browse_out = QPushButton("Browse")
        browse_out.clicked.connect(self.pick_output_dir)
        row = QHBoxLayout()
        row.addWidget(self.output_dir)
        row.addWidget(browse_out)
        layer_form.addRow("Output folder", row)
        layout.addWidget(layer_group)

        
        model_group = QGroupBox("Custom Model")
        model_form = QFormLayout(model_group)

        self.custom_model_path = QLineEdit("")
        self.custom_model_button = QPushButton("Load Custom YOLOv8 .pt Model")
        self.device_combo = QComboBox()
        self.device_combo.addItems(["CPU", "CUDA"])
        self.custom_model_button.clicked.connect(self.pick_custom_model)
        custom_row = QHBoxLayout()
        custom_row.addWidget(self.custom_model_path)
        custom_row.addWidget(self.custom_model_button)
        model_form.addRow("Load Custom YOLOv8 .pt Model", custom_row)

        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(25)
        self.conf_label = QLabel("0.25")
        self.conf_slider.valueChanged.connect(
            lambda v: self.conf_label.setText(f"{v/100.0:.2f}")
        )
        conf_row = QHBoxLayout()
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(self.conf_label)
        model_form.addRow("Device", self.device_combo)

        model_form.addRow("YOLO confidence", conf_row)

        self.tile_size = QSpinBox()
        self.tile_size.setRange(256, 2048)
        self.tile_size.setSingleStep(64)
        self.tile_size.setValue(1024)
        model_form.addRow("Tile size", self.tile_size)

        self.overlap = QSpinBox()
        self.overlap.setRange(0, 1024)
        self.overlap.setSingleStep(32)
        self.overlap.setValue(160)
        model_form.addRow("Overlap", self.overlap)

        self.max_clip_size = QSpinBox()
        self.max_clip_size.setRange(1024, 16384)
        self.max_clip_size.setSingleStep(512)
        self.max_clip_size.setValue(6144)
        model_form.addRow("Max clip size px", self.max_clip_size)

        self.add_output = QCheckBox("Load GeoPackage result into QGIS")
        self.add_output.setChecked(True)
        model_form.addRow("", self.add_output)

        layout.addWidget(model_group)

        buttons = QHBoxLayout()
        self.select_btn = QPushButton("Select map area + detect")
        self.select_btn.clicked.connect(self.select_area)
        self.run_full_btn = QPushButton("Run on selected raster extent")
        self.run_full_btn.clicked.connect(self.run_full_extent)
        self.cancel_detection_btn = QPushButton("Cancel Detection")
        self.cancel_detection_btn.clicked.connect(self.cancel_detection)
        buttons.addWidget(self.select_btn)
        buttons.addWidget(self.run_full_btn)
        buttons.addWidget(self.cancel_detection_btn)
        layout.addLayout(buttons)

        self.status = QLabel("Ready.")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.hide()
        self.progress_percent = QLabel("0% | Raster 0/0")
        self.progress_percent.hide()
        self.detection_count_label = QLabel("Detections: 0")
        self.download_tiles_label = QLabel("Tiles: 0/0")
        self.download_speed_label = QLabel("Speed: 0.00 tiles/sec")
        self.download_phase_label = QLabel("Phase: idle")

        progress_row = QHBoxLayout()
        progress_row.addWidget(self.progress)
        progress_row.addWidget(self.progress_percent)
        layout.addWidget(self.status)
        layout.addLayout(progress_row)
        layout.addWidget(self.detection_count_label)
        layout.addWidget(self.download_tiles_label)
        layout.addWidget(self.download_speed_label)
        layout.addWidget(self.download_phase_label)

        rerun_group = QGroupBox("Post-detection adjustment")
        rerun_layout = QVBoxLayout(rerun_group)

        rerun_info = QLabel(
            "Choose the model above, then run detection or rerun the last selected area with the current slider values."
        )
        rerun_info.setWordWrap(True)
        rerun_layout.addWidget(rerun_info)

        self.rerun_btn = QPushButton("Rerun detection with current sliders")
        self.rerun_btn.clicked.connect(self.rerun_last_detection)
        rerun_layout.addWidget(self.rerun_btn)


        layout.addWidget(rerun_group)

        layout.addStretch(1)
        return self._wrap_scroll(page)

    def _make_training_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        project_group = QGroupBox("Project")
        pf = QFormLayout(project_group)

        self.train_project_folder = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_Project"))
        choose_project = QPushButton("Choose Project")
        choose_project.clicked.connect(self.choose_train_project)
        prow = QHBoxLayout()
        prow.addWidget(self.train_project_folder)
        prow.addWidget(choose_project)
        pf.addRow("Project folder", prow)

        create_project = QPushButton("Create Project")
        create_project.clicked.connect(self.create_train_project)
        pf.addRow("", create_project)

        create_yaml = QPushButton("Create / Update data.yaml + sort images/labels")
        create_yaml.clicked.connect(self.create_train_yaml)
        pf.addRow("", create_yaml)

        split_btn = QPushButton("Split images into train/val")
        split_btn.clicked.connect(self.split_train_project)
        pf.addRow("", split_btn)

        self.train_source_images = QLineEdit(str(Path(self.train_project_folder.text()) / "images"))
        browse_source_images = QPushButton("Browse")
        browse_source_images.clicked.connect(self.pick_train_source_images)
        imgrow = QHBoxLayout()
        imgrow.addWidget(self.train_source_images)
        imgrow.addWidget(browse_source_images)
        pf.addRow("Training images", imgrow)

        self.train_source_labels = QLineEdit(str(Path(self.train_project_folder.text()) / "labels"))
        browse_source_labels = QPushButton("Browse")
        browse_source_labels.clicked.connect(self.pick_train_source_labels)
        labrow = QHBoxLayout()
        labrow.addWidget(self.train_source_labels)
        labrow.addWidget(browse_source_labels)
        pf.addRow("Training labels", labrow)

        self.train_classes = QLineEdit("mustatil,false_positive")
        pf.addRow("Classes", self.train_classes)

        layout.addWidget(project_group)

        group = QGroupBox("YOLO Trainer")
        form = QFormLayout(group)

        self.train_data_yaml = QLineEdit(str(Path(self.train_project_folder.text()) / "yolo_datasets" / "data.yaml"))
        browse_data = QPushButton("Browse")
        browse_data.clicked.connect(self.pick_train_data_yaml)
        row = QHBoxLayout()
        row.addWidget(self.train_data_yaml)
        row.addWidget(browse_data)
        form.addRow("data.yaml", row)

        self.train_base_model = QLineEdit("yolov8n.pt")
        browse_model = QPushButton("Browse")
        browse_model.clicked.connect(self.pick_train_base_model)
        row2 = QHBoxLayout()
        row2.addWidget(self.train_base_model)
        row2.addWidget(browse_model)
        form.addRow("Base model", row2)

        self.train_epochs = QSpinBox()
        self.train_epochs.setRange(1, 2000)
        self.train_epochs.setValue(80)
        form.addRow("Epochs", self.train_epochs)

        self.train_imgsz = QSpinBox()
        self.train_imgsz.setRange(128, 4096)
        self.train_imgsz.setSingleStep(32)
        self.train_imgsz.setValue(640)
        form.addRow("Image size", self.train_imgsz)

        self.train_batch = QSpinBox()
        self.train_batch.setRange(1, 256)
        self.train_batch.setValue(2)
        form.addRow("Batch", self.train_batch)

        self.train_device = QComboBox()
        self.train_device.addItems([
            "cpu",
            "cuda",
            "directml",
            "opencl",
        ])
        self.train_device.setCurrentText("cpu")
        form.addRow("Device", self.train_device)

        self.train_runs_folder = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_YOLO_Runs"))
        form.addRow("Runs folder", self.train_runs_folder)

        btn = QPushButton("Start YOLO training")
        btn.clicked.connect(self.start_yolo_training)
        form.addRow("", btn)

        layout.addWidget(group)

        info = QLabel(
            "Create/choose a project first, then create data.yaml. "
            "Create/Update data.yaml also copies/sorts files from the selected training image/label folders into yolo_datasets/train and yolo_datasets/val. "
            "Device dropdown supports CPU, CUDA, DirectML and OpenCL labels. "
            "Actual acceleration depends on installed Torch/ONNX runtime backend."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch(1)
        return self._wrap_scroll(page)

    def _make_toolkit_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        project_group = QGroupBox("Project / Dataset")
        form = QFormLayout(project_group)

        self.toolkit_project = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_Project"))
        browse_project = QPushButton("Browse")
        browse_project.clicked.connect(self.pick_toolkit_project)
        prow = QHBoxLayout()
        prow.addWidget(self.toolkit_project)
        prow.addWidget(browse_project)
        form.addRow("Project folder", prow)

        create_btn = QPushButton("Create Mustatil project folders + data.yaml")
        create_btn.clicked.connect(self.create_mustatil_project)
        form.addRow("", create_btn)

        open_btn = QPushButton("Open project folder")
        open_btn.clicked.connect(lambda: self.open_folder(self.toolkit_project.text()))
        form.addRow("", open_btn)

        layout.addWidget(project_group)

        full_group = QGroupBox("Original workflow modules")
        fl = QVBoxLayout(full_group)


        annotator_note = QLabel(
            "Crop annotator, SAM2 image-list workflow and full interactive preview are kept as external workflow modules. "
            "This avoids mixing Tkinter/SAM2/Torch directly into the QGIS event loop."
        )
        annotator_note.setWordWrap(True)
        fl.addWidget(annotator_note)

        layout.addWidget(full_group)

        export_group = QGroupBox("QGIS area tools")
        el = QVBoxLayout(export_group)
        export_clip_btn = QPushButton("Select map area and export GeoTIFF only")
        export_clip_btn.clicked.connect(self.select_area_export_only)
        el.addWidget(export_clip_btn)
        layout.addWidget(export_group)

        runtime_group = QGroupBox("External Python Runtime")
        rform = QFormLayout(runtime_group)

        self.python_path = QLineEdit(self.detect_existing_runtime())
        browse_py = QPushButton("Browse")
        browse_py.clicked.connect(self.pick_python)
        pyrow = QHBoxLayout()
        pyrow.addWidget(self.python_path)
        pyrow.addWidget(browse_py)
        rform.addRow("Python.exe", pyrow)

        self.model_path = QLineEdit(str(self.plugin_dir / "models" / "bestf260.onnx"))
        rform.addRow("ONNX model", self.model_path)

        self.runtime_training_images = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_Project" / "images"))
        rt_img_btn = QPushButton("Browse")
        rt_img_btn.clicked.connect(self.pick_runtime_training_images)
        rt_img_row = QHBoxLayout()
        rt_img_row.addWidget(self.runtime_training_images)
        rt_img_row.addWidget(rt_img_btn)
        rform.addRow("Training images", rt_img_row)

        self.runtime_training_labels = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_Project" / "labels"))
        rt_lab_btn = QPushButton("Browse")
        rt_lab_btn.clicked.connect(self.pick_runtime_training_labels)
        rt_lab_row = QHBoxLayout()
        rt_lab_row.addWidget(self.runtime_training_labels)
        rt_lab_row.addWidget(rt_lab_btn)
        rform.addRow("Training labels", rt_lab_row)

        use_training_btn = QPushButton("Use selected training data in YOLO Trainer")
        use_training_btn.clicked.connect(self.use_runtime_training_data)
        rform.addRow("", use_training_btn)

        install_btn = QPushButton("Install external runtime")
        install_btn.clicked.connect(self.install_runtime)
        rform.addRow("", install_btn)

        manual_btn = QPushButton("Open plugin folder")
        manual_btn.clicked.connect(self.open_plugin_folder)
        rform.addRow("", manual_btn)

        runtime_info = QLabel(
            "The AI runtime is external to QGIS. Training image/label folders can be selected here and applied to the YOLO Trainer. "
            "This avoids DLL conflicts with QGIS Python, GDAL, ONNX Runtime, Torch, Fiona and Rasterio."
        )
        runtime_info.setWordWrap(True)
        layout.addWidget(runtime_group)
        layout.addWidget(runtime_info)

        layout.addStretch(1)
        return self._wrap_scroll(page)

    def choose_train_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose YOLO project folder", self.train_project_folder.text())
        if d:
            self.train_project_folder.setText(d)
            self.train_data_yaml.setText(str(Path(d) / "yolo_datasets" / "data.yaml"))
            if hasattr(self, "toolkit_project"):
                self.toolkit_project.setText(d)
            if hasattr(self, "train_source_images"):
                self.train_source_images.setText(str(Path(d) / "images"))
            if hasattr(self, "train_source_labels"):
                self.train_source_labels.setText(str(Path(d) / "labels"))
            if hasattr(self, "runtime_training_images"):
                self.runtime_training_images.setText(str(Path(d) / "images"))
            if hasattr(self, "runtime_training_labels"):
                self.runtime_training_labels.setText(str(Path(d) / "labels"))
            if hasattr(self, "annotator_images"):
                self.annotator_images.setText(str(Path(d) / "train" / "images"))
            if hasattr(self, "annotator_labels"):
                self.annotator_labels.setText(str(Path(d) / "train" / "labels"))

    def create_train_project(self):
        root = self.train_project_folder.text().strip()
        if not root:
            QMessageBox.warning(self, "Missing project folder", "Enter or choose a project folder.")
            return
        args = [
            self.plugin_dir / "scripts" / "mustatil_project_tools.py",
            "--create-project", root,
        ]
        self._start_external_process(args, "Create YOLO project")
        self.train_data_yaml.setText(str(Path(root) / "yolo_datasets" / "data.yaml"))

    def create_train_yaml(self):
        root = self.train_project_folder.text().strip()
        if not root:
            QMessageBox.warning(self, "Missing project folder", "Enter or choose a project folder.")
            return
        args = [
            self.plugin_dir / "scripts" / "mustatil_project_tools.py",
            "--create-yaml", root,
            "--classes", self.train_classes.text().strip() or "mustatil,false_positive",
        ]
        if hasattr(self, "train_source_images") and self.train_source_images.text().strip():
            args += ["--images-dir", self.train_source_images.text().strip()]
        if hasattr(self, "train_source_labels") and self.train_source_labels.text().strip():
            args += ["--labels-dir", self.train_source_labels.text().strip()]
        self._start_external_process(args, "Create data.yaml")
        self.train_data_yaml.setText(str(Path(root) / "yolo_datasets" / "data.yaml"))

    def split_train_project(self):
        root = self.train_project_folder.text().strip()
        if not root:
            QMessageBox.warning(self, "Missing project folder", "Enter or choose a project folder.")
            return
        args = [
            self.plugin_dir / "scripts" / "mustatil_project_tools.py",
            "--split-project", root,
            "--classes", self.train_classes.text().strip() or "mustatil,false_positive",
        ]
        if hasattr(self, "train_source_images") and self.train_source_images.text().strip():
            args += ["--images-dir", self.train_source_images.text().strip()]
        if hasattr(self, "train_source_labels") and self.train_source_labels.text().strip():
            args += ["--labels-dir", self.train_source_labels.text().strip()]
        self._start_external_process(args, "Split YOLO dataset")
        self.train_data_yaml.setText(str(Path(root) / "yolo_datasets" / "data.yaml"))

    def pick_train_source_images(self):
        d = QFileDialog.getExistingDirectory(self, "Select training image folder", self.train_source_images.text())
        if d:
            self.train_source_images.setText(d)

    def pick_train_source_labels(self):
        d = QFileDialog.getExistingDirectory(self, "Select training label folder", self.train_source_labels.text())
        if d:
            self.train_source_labels.setText(d)

    def pick_train_data_yaml(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select YOLO data.yaml", "", "YAML (*.yaml *.yml);;All files (*.*)")
        if p:
            self.train_data_yaml.setText(p)

    def pick_train_base_model(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "YOLO models (*.pt *.onnx);;All files (*.*)")
        if p:
            self.train_base_model.setText(p)

    def pick_toolkit_project(self):
        d = QFileDialog.getExistingDirectory(self, "Select project folder", self.toolkit_project.text())
        if d:
            self.toolkit_project.setText(d)
            if hasattr(self, "train_data_yaml"):
                self.train_data_yaml.setText(str(Path(d) / "yolo_datasets" / "data.yaml"))

    def open_folder(self, path):
        try:
            os.startfile(str(Path(path).expanduser()))
        except Exception as e:
            QMessageBox.warning(self, "Open folder failed", str(e))

    def _external_python(self):
        py_text = self.python_path.text().strip() if hasattr(self, "python_path") else ""
        if py_text:
            p = Path(py_text)
            if p.exists() and not p.is_dir() and str(p) != ".":
                return p
        detected = self.detect_existing_runtime()
        if detected:
            p = Path(detected)
            if p.exists() and not p.is_dir() and str(p) != ".":
                return p
        return self.plugin_dir / "runtime" / "python" / "python.exe"

    def _start_external_process(self, args, title="External process", switch_to_log=True, on_success=None):
        py = self._external_python()
        if not py.exists():
            QMessageBox.warning(self, "Runtime missing", "Python runtime not found. Use Install Runtime in the  tab first.")
            self.tabs.setCurrentWidget(self.toolkit_tab)
            return

        cmd = [str(py)] + [str(a) for a in args]
        self.append_log(title + " started:", switch_to_log=switch_to_log)
        self.append_log(" ".join(f'"{c}"' if " " in c else c for c in cmd), switch_to_log=switch_to_log)

        proc = QProcess(self)
        proc.setProgram(cmd[0])
        proc.setArguments(cmd[1:])
        proc.readyReadStandardOutput.connect(lambda p=proc, sw=switch_to_log: self.append_log(bytes(p.readAllStandardOutput()).decode("utf-8", errors="replace"), switch_to_log=sw))
        proc.readyReadStandardError.connect(lambda p=proc, sw=switch_to_log: self.append_log(bytes(p.readAllStandardError()).decode("utf-8", errors="replace"), switch_to_log=sw))

        def _finished(code, status, name=title, sw=switch_to_log, cb=on_success):
            self.append_log(f"{name} finished with code {code}.", switch_to_log=sw)
            if code == 0 and cb is not None:
                try:
                    QTimer.singleShot(0, cb)
                except Exception as exc:
                    self.append_log(f"{name} post-processing failed: {exc}", switch_to_log=sw)

        proc.finished.connect(_finished)
        proc.start()
        self.process = proc

    def start_yolo_training(self):
        data = self.train_data_yaml.text().strip()
        if not data:
            root = self.train_project_folder.text().strip() if hasattr(self, "train_project_folder") else ""
            data = str(Path(root) / "yolo_datasets" / "data.yaml") if root else ""
            if hasattr(self, "train_data_yaml") and data:
                self.train_data_yaml.setText(data)

        data_path = Path(data) if data else None

        if not data_path:
            QMessageBox.warning(self, "Missing data.yaml", "Select a YOLO project folder or data.yaml first.")
            return

        # If data.yaml does not exist yet, create it first and then start training automatically.
        if not data_path.exists():
            root = self.train_project_folder.text().strip() if hasattr(self, "train_project_folder") else str(data_path.parent.parent)
            create_args = [
                self.plugin_dir / "scripts" / "mustatil_project_tools.py",
                "--create-yaml", root,
                "--classes", self.train_classes.text().strip() or "mustatil,false_positive",
            ]
            if hasattr(self, "train_source_images") and self.train_source_images.text().strip():
                create_args += ["--images-dir", self.train_source_images.text().strip()]
            if hasattr(self, "train_source_labels") and self.train_source_labels.text().strip():
                create_args += ["--labels-dir", self.train_source_labels.text().strip()]

            self.append_log("data.yaml missing. Creating dataset first, then training will start automatically.", switch_to_log=True)
            self._start_external_process(create_args, "Create data.yaml before YOLO training", on_success=self.start_yolo_training)
            return

        args = [
            self.plugin_dir / "scripts" / "mustatil_yolo_trainer.py",
            "--data", str(data_path),
            "--model", self.train_base_model.text().strip() or "yolov8n.pt",
            "--epochs", str(self.train_epochs.value()),
            "--imgsz", str(self.train_imgsz.value()),
            "--batch", str(self.train_batch.value()),
            "--device", self.train_device.currentText().strip() or "cpu",
            "--project", self.train_runs_folder.text().strip(),
            "--name", "mustatil_qgis_train",
        ]
        self._start_external_process(args, "YOLO training")

    def create_mustatil_project(self):
        root = self.toolkit_project.text().strip()
        if not root:
            QMessageBox.warning(self, "Missing project folder", "Select or enter a project folder.")
            return
        args = [
            self.plugin_dir / "scripts" / "mustatil_project_tools.py",
            "--create-project", root,
        ]
        self._start_external_process(args, "Project creation")
        if hasattr(self, "train_data_yaml"):
            self.train_data_yaml.setText(str(Path(root) / "yolo_datasets" / "data.yaml"))


    def select_area_export_only(self):
        layer = self.layer_combo.currentLayer()
        if not layer:
            QMessageBox.warning(self, "No raster", "Select a raster layer in the Detection tab first.")
            return
        self.export_only_mode = True
        self.clear_selection_outline()
        self.rect_tool = RectTool(self.canvas, self._area_selected_export_only)
        self._active_selection_rubber = self.rect_tool.rubber
        self.canvas.setMapTool(self.rect_tool)
        self.status.setText("Draw a rectangle to export a georeferenced GeoTIFF only.")

    def _area_selected_export_only(self, rect):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return
        out_dir = Path(self.output_dir.text()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_dir = out_dir / "Mustatil_QGIS_Clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = clip_dir / f"mustatil_qgis_clip_only_{stamp}.tif"
        try:
            self.export_clip(layer, rect, str(clip_path))
            self.append_log(f"GeoTIFF exported: {clip_path}")
            QMessageBox.information(self, "Exported", f"GeoTIFF exported:\n{clip_path}")
        except Exception as e:
            self.append_log("Raster clip failed:")
            self.append_log(str(e))
            QMessageBox.critical(self, "Export failed", str(e))

    def _make_annotator_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        settings = QGroupBox("Annotator Project")
        form = QFormLayout(settings)

        self.annotator_project = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_Project"))
        b_proj = QPushButton("Choose Project for Annotator")
        b_proj.clicked.connect(self.choose_annotator_project)
        projrow = QHBoxLayout()
        projrow.addWidget(self.annotator_project)
        projrow.addWidget(b_proj)
        form.addRow("Project", projrow)

        self.annotator_images = QLineEdit(str(Path(self.annotator_project.text()) / "yolo_datasets" / "train" / "images"))
        b_img = QPushButton("Choose images")
        b_img.clicked.connect(self.pick_annotator_images)
        row = QHBoxLayout()
        row.addWidget(self.annotator_images)
        row.addWidget(b_img)
        form.addRow("Image folder", row)

        self.annotator_labels = QLineEdit("")
        b_lab = QPushButton("Choose labels")
        b_lab.clicked.connect(self.pick_annotator_labels)
        row2 = QHBoxLayout()
        row2.addWidget(self.annotator_labels)
        row2.addWidget(b_lab)
        form.addRow("Label folder", row2)

        self.annotator_classes = QLineEdit("positive,false")
        form.addRow("Classes", self.annotator_classes)

        self.annotator_box_class = QComboBox()
        self.annotator_box_class.addItems(["0 positive", "1 false"])
        form.addRow("Draw box as", self.annotator_box_class)

        load_btn = QPushButton("Load images into preview")
        load_btn.clicked.connect(self.load_annotator_images)
        form.addRow("", load_btn)

        layout.addWidget(settings)

        body = QHBoxLayout()

        self.annotator_list = QListWidget()
        self.annotator_list.currentRowChanged.connect(self.annotator_select_index)
        body.addWidget(self.annotator_list, 1)

        right = QVBoxLayout()

        zoom_row = QHBoxLayout()
        self.annotator_zoom_label = QLabel("100%")
        zoom_in_btn = QPushButton("+")
        zoom_out_btn = QPushButton("-")
        zoom_fit_btn = QPushButton("Fit")
        zoom_in_btn.clicked.connect(lambda: self.annotator_zoom(1.25))
        zoom_out_btn.clicked.connect(lambda: self.annotator_zoom(0.8))
        zoom_fit_btn.clicked.connect(self.annotator_zoom_fit)

        zoom_row.addWidget(QLabel("Zoom"))
        zoom_row.addWidget(zoom_out_btn)
        zoom_row.addWidget(zoom_in_btn)
        zoom_row.addWidget(zoom_fit_btn)
        zoom_row.addWidget(self.annotator_zoom_label)
        zoom_row.addStretch(1)
        right.addLayout(zoom_row)

        self.annotator_scene = QGraphicsScene()
        self.annotator_view = QGraphicsView(self.annotator_scene)
        self.annotator_view.setMinimumHeight(420)
        self.annotator_view.setDragMode(QGraphicsView.NoDrag)
        self.annotator_view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.annotator_view.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.annotator_view.setMouseTracking(True)
        # Override mouse events for direct drawing inside the QGIS dock.
        self.annotator_view.mousePressEvent = self.annotator_mouse_press
        self.annotator_view.mouseMoveEvent = self.annotator_mouse_move
        self.annotator_view.mouseReleaseEvent = self.annotator_mouse_release
        right.addWidget(self.annotator_view, 5)

        buttons = QHBoxLayout()
        save_btn = QPushButton("Save label")
        save_btn.clicked.connect(self.annotator_save_current)
        prev_btn = QPushButton("Prev")
        prev_btn.clicked.connect(self.annotator_prev)
        next_btn = QPushButton("Next")
        next_btn.clicked.connect(self.annotator_next)
        del_btn = QPushButton("Delete last box")
        del_btn.clicked.connect(self.annotator_delete_last_box)

        buttons.addWidget(prev_btn)
        buttons.addWidget(next_btn)
        buttons.addWidget(del_btn)
        buttons.addWidget(save_btn)
        right.addLayout(buttons)

        sam_group = QGroupBox("SAM2")
        sf = QFormLayout(sam_group)
        self.sam2_checkpoint = QLineEdit("")
        sam_ckpt_btn = QPushButton("Checkpoint")
        sam_ckpt_btn.clicked.connect(self.pick_sam2_checkpoint)
        ckrow = QHBoxLayout()
        ckrow.addWidget(self.sam2_checkpoint)
        ckrow.addWidget(sam_ckpt_btn)
        sf.addRow("SAM2 checkpoint", ckrow)

        self.sam2_model_cfg = QLineEdit("")
        sf.addRow("SAM2 config", self.sam2_model_cfg)

        self.sam2_model_choice = QComboBox()
        self.sam2_model_choice.addItems(["base_plus", "tiny", "small", "large"])
        self.sam2_model_choice.setCurrentText("base_plus")
        sf.addRow("Auto model", self.sam2_model_choice)

        sam_install = QPushButton("Install SAM2 automatically")
        sam_install.clicked.connect(self.install_sam2_auto)
        sf.addRow("", sam_install)

        sam_buttons = QHBoxLayout()
        sam_one = QPushButton("SAM2 current image")
        sam_one.clicked.connect(self.run_sam2_current)
        sam_all = QPushButton("SAM2 all pictures")
        sam_all.clicked.connect(self.run_sam2_all)
        sam_buttons.addWidget(sam_one)
        sam_buttons.addWidget(sam_all)
        sf.addRow("", sam_buttons)

        sam_label_buttons = QHBoxLayout()
        sam_use_current = QPushButton("Use SAM2 segments + YOLO negatives as labels")
        sam_use_current.setToolTip("Convert SAM2 polygons to YOLO boxes; segments overlapping false_positive YOLO boxes inherit the negative class.")
        sam_use_current.clicked.connect(self.use_sam2_segments_current)
        sam_use_all = QPushButton("Use all SAM2 segments + YOLO negatives")
        sam_use_all.setToolTip("Convert all SAM2 polygons; boxes overlapping false_positive YOLO labels are saved as negative.")
        sam_use_all.clicked.connect(self.use_sam2_segments_all)
        sam_label_buttons.addWidget(sam_use_current)
        sam_label_buttons.addWidget(sam_use_all)
        sf.addRow("", sam_label_buttons)
        right.addWidget(sam_group)

        hint = QLabel(
            "Draw boxes directly with the mouse: choose Positive or False, then left-drag on the image. Use mouse wheel or +/- buttons to zoom. Drag with the middle mouse button to pan. "
            "Right click deletes the last box. Save writes YOLO labels. Draw red/False boxes for negative  samples. "
            "SAM2 conversion now inherits negative/false_positive classes from YOLO labels and preserves unmatched negative boxes. "
            "SAM2 uses the external runtime. Press 'Install SAM2 automatically' first; checkpoint/config are then filled automatically."
        )
        hint.setWordWrap(True)
        right.addWidget(hint)

        body.addLayout(right, 4)
        layout.addLayout(body)

        self.annotator_image_paths = []
        self.annotator_current_index = -1
        self.annotator_boxes = []
        self.annotator_pixmap_item = None
        self.annotator_drag_start = None
        self.annotator_preview_item = None
        self.annotator_zoom_factor = 1.0
        self.annotator_middle_dragging = False
        self.annotator_middle_last_pos = None

        # Mouse wheel zoom support.
        self.annotator_view.wheelEvent = self.annotator_wheel_event
        return page

    def choose_annotator_project(self):
        d = QFileDialog.getExistingDirectory(self, "Choose annotator project folder", self.annotator_project.text())
        if d:
            self.annotator_project.setText(d)
            self.annotator_images.setText(str(Path(d) / "yolo_datasets" / "train" / "images"))
            self.annotator_labels.setText(str(Path(d) / "yolo_datasets" / "train" / "labels"))
            if hasattr(self, "train_project_folder"):
                self.train_project_folder.setText(d)
            if hasattr(self, "train_data_yaml"):
                self.train_data_yaml.setText(str(Path(d) / "yolo_datasets" / "data.yaml"))
            self.load_annotator_images()

    def pick_annotator_images(self):
        d = QFileDialog.getExistingDirectory(self, "Select image folder", self.annotator_images.text())
        if d:
            self.annotator_images.setText(d)
            if not self.annotator_labels.text().strip():
                self.annotator_labels.setText(str(Path(d).parent / "labels"))
            self.load_annotator_images()

    def pick_annotator_labels(self):
        d = QFileDialog.getExistingDirectory(self, "Select label folder", self.annotator_labels.text())
        if d:
            self.annotator_labels.setText(d)

    def pick_sam2_checkpoint(self):
        p, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM2 checkpoint",
            "",
            "SAM2 checkpoints (*.pt *.pth *.ckpt);;All files (*.*)"
        )
        if p:
            self.sam2_checkpoint.setText(p)

    def _annotator_image_exts(self):
        return {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

    def load_annotator_images(self):
        folder = Path(self.annotator_images.text()).expanduser()
        self.annotator_list.clear()
        self.annotator_image_paths = []
        if not folder.exists():
            QMessageBox.warning(self, "Missing folder", "Image folder does not exist.")
            return

        self.annotator_image_paths = [
            p for p in sorted(folder.rglob("*"))
            if p.suffix.lower() in self._annotator_image_exts()
        ]

        for p in self.annotator_image_paths:
            self.annotator_list.addItem(p.name)

        if self.annotator_image_paths:
            self.annotator_list.setCurrentRow(0)
        self.append_annotator_log(f"Annotator loaded images: {len(self.annotator_image_paths)}")

    def annotator_update_zoom_label(self):
        try:
            self.annotator_zoom_label.setText(f"{int(self.annotator_zoom_factor * 100)}%")
        except Exception:
            pass

    def annotator_zoom(self, factor):
        if not hasattr(self, "annotator_view"):
            return
        self.annotator_zoom_factor *= factor
        self.annotator_zoom_factor = max(0.05, min(20.0, self.annotator_zoom_factor))
        self.annotator_view.scale(factor, factor)
        self.annotator_update_zoom_label()

    def annotator_zoom_fit(self):
        if not hasattr(self, "annotator_view"):
            return
        try:
            self.annotator_view.resetTransform()
            self.annotator_view.fitInView(self.annotator_scene.sceneRect(), Qt.KeepAspectRatio)
            # Estimate resulting scale.
            transform = self.annotator_view.transform()
            self.annotator_zoom_factor = max(0.01, transform.m11())
            self.annotator_update_zoom_label()
        except Exception:
            pass

    def annotator_wheel_event(self, event):
        try:
            delta = event.angleDelta().y()
            if delta > 0:
                self.annotator_zoom(1.15)
            else:
                self.annotator_zoom(0.87)
            event.accept()
        except Exception:
            try:
                QGraphicsView.wheelEvent(self.annotator_view, event)
            except Exception:
                pass

    def annotator_select_index(self, row):
        if row < 0 or row >= len(getattr(self, "annotator_image_paths", [])):
            return
        self.annotator_current_index = row
        self.annotator_load_current()

    def _annotator_label_path(self, img_path):
        labels = self.annotator_labels.text().strip()
        if not labels:
            labels = str(Path(img_path).parent.parent / "labels")
            self.annotator_labels.setText(labels)
        lab_dir = Path(labels)
        lab_dir.mkdir(parents=True, exist_ok=True)
        return lab_dir / (Path(img_path).stem + ".txt")

    def annotator_load_current(self):
        if self.annotator_current_index < 0:
            return
        img_path = self.annotator_image_paths[self.annotator_current_index]
        pix = QPixmap(str(img_path))
        if pix.isNull():
            self.append_annotator_log(f"Could not load image preview: {img_path}")
            return

        self.annotator_scene.clear()
        self.annotator_pixmap_item = self.annotator_scene.addPixmap(pix)
        self.annotator_scene.setSceneRect(self.annotator_pixmap_item.boundingRect())
        self.annotator_zoom_fit()

        self.annotator_boxes = []
        lab = self._annotator_label_path(img_path)
        if lab.exists():
            W = max(1, pix.width())
            H = max(1, pix.height())
            for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = (cx - bw / 2.0) * W
                y1 = (cy - bh / 2.0) * H
                x2 = (cx + bw / 2.0) * W
                y2 = (cy + bh / 2.0) * H
                self.annotator_boxes.append([cls, x1, y1, x2, y2])

        self.annotator_redraw_boxes()
        self.append_annotator_log(f"Annotator preview: {img_path}")

    def _annotator_scene_pos(self, event):
        return self.annotator_view.mapToScene(event.pos())

    def _annotator_clamp_point(self, p):
        if not self.annotator_pixmap_item:
            return QPointF(0, 0)
        rect = self.annotator_pixmap_item.boundingRect()
        x = max(rect.left(), min(rect.right(), p.x()))
        y = max(rect.top(), min(rect.bottom(), p.y()))
        return QPointF(x, y)

    def annotator_mouse_press(self, event):
        if self.annotator_current_index < 0 or self.annotator_pixmap_item is None:
            return

        if event.button() == Qt.MiddleButton:
            self.annotator_middle_dragging = True
            self.annotator_middle_last_pos = event.pos()
            self.annotator_view.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.RightButton:
            self.annotator_delete_last_box()
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            self.annotator_drag_start = self._annotator_clamp_point(self._annotator_scene_pos(event))
            if self.annotator_preview_item:
                self.annotator_scene.removeItem(self.annotator_preview_item)
                self.annotator_preview_item = None
            event.accept()
            return

    def annotator_mouse_move(self, event):
        if getattr(self, "annotator_middle_dragging", False):
            if self.annotator_middle_last_pos is not None:
                delta = event.pos() - self.annotator_middle_last_pos
                self.annotator_middle_last_pos = event.pos()
                self.annotator_view.horizontalScrollBar().setValue(
                    self.annotator_view.horizontalScrollBar().value() - delta.x()
                )
                self.annotator_view.verticalScrollBar().setValue(
                    self.annotator_view.verticalScrollBar().value() - delta.y()
                )
            event.accept()
            return

        if self.annotator_drag_start is None:
            return

        end = self._annotator_clamp_point(self._annotator_scene_pos(event))
        x1 = self.annotator_drag_start.x()
        y1 = self.annotator_drag_start.y()
        x2 = end.x()
        y2 = end.y()

        if self.annotator_preview_item:
            self.annotator_scene.removeItem(self.annotator_preview_item)

        pen = QPen(QColor(0, 255, 255, 255))
        pen.setWidth(2)
        self.annotator_preview_item = self.annotator_scene.addRect(
            min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1), pen
        )
        event.accept()

    def annotator_mouse_release(self, event):
        if event.button() == Qt.MiddleButton:
            self.annotator_middle_dragging = False
            self.annotator_middle_last_pos = None
            self.annotator_view.setCursor(Qt.ArrowCursor)
            event.accept()
            return

        if self.annotator_drag_start is None:
            return

        end = self._annotator_clamp_point(self._annotator_scene_pos(event))
        x1 = self.annotator_drag_start.x()
        y1 = self.annotator_drag_start.y()
        x2 = end.x()
        y2 = end.y()
        self.annotator_drag_start = None

        if self.annotator_preview_item:
            self.annotator_scene.removeItem(self.annotator_preview_item)
            self.annotator_preview_item = None

        if abs(x2-x1) < 4 or abs(y2-y1) < 4:
            event.accept()
            return

        cls = self.annotator_box_class.currentIndex()
        self.annotator_boxes.append([cls, x1, y1, x2, y2])
        self.annotator_redraw_boxes()
        event.accept()

    def annotator_redraw_boxes(self):
        for item in list(self.annotator_scene.items()):
            if item is not self.annotator_pixmap_item:
                self.annotator_scene.removeItem(item)

        classes = [c.strip() for c in self.annotator_classes.text().split(",") if c.strip()]
        for box in self.annotator_boxes:
            cls, x1, y1, x2, y2 = box
            color = QColor(0, 220, 0, 255) if cls == 0 else QColor(255, 0, 0, 255)
            pen = QPen(color)
            pen.setWidth(3)
            self.annotator_scene.addRect(min(x1, x2), min(y1, y2), abs(x2-x1), abs(y2-y1), pen)
            label = classes[cls] if 0 <= cls < len(classes) else str(cls)
            text_item = self.annotator_scene.addText(label)
            text_item.setDefaultTextColor(color)
            text_item.setPos(min(x1, x2), min(y1, y2))

    def annotator_save_current(self):
        if self.annotator_current_index < 0:
            return
        img_path = self.annotator_image_paths[self.annotator_current_index]
        pix = QPixmap(str(img_path))
        W = max(1, pix.width())
        H = max(1, pix.height())
        lab = self._annotator_label_path(img_path)

        lines = []
        for cls, x1, y1, x2, y2 in self.annotator_boxes:
            x1, x2 = sorted([max(0, min(W, x1)), max(0, min(W, x2))])
            y1, y2 = sorted([max(0, min(H, y1)), max(0, min(H, y2))])
            if x2-x1 < 2 or y2-y1 < 2:
                continue
            cx = ((x1+x2)/2.0)/W
            cy = ((y1+y2)/2.0)/H
            bw = (x2-x1)/W
            bh = (y2-y1)/H
            lines.append(f"{int(cls)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")

        lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.append_annotator_log(f"Saved label: {lab}")

    def annotator_prev(self):
        row = max(0, self.annotator_current_index - 1)
        self.annotator_list.setCurrentRow(row)

    def annotator_next(self):
        row = min(len(self.annotator_image_paths) - 1, self.annotator_current_index + 1)
        self.annotator_list.setCurrentRow(row)

    def annotator_delete_last_box(self):
        if self.annotator_boxes:
            self.annotator_boxes.pop()
            self.annotator_redraw_boxes()

    def _sam2_sidecar_path(self, img_path):
        """Return the SAM2 JSON sidecar used by the runner."""
        return Path(img_path).with_suffix(".sam2.json")

    def _sam2_polygon_bbox(self, item):
        """Extract a pixel bbox from a SAM2 polygon item or fallback bbox."""
        if isinstance(item, dict):
            if "polygon" in item:
                pts = item.get("polygon") or []
            elif "points" in item:
                pts = item.get("points") or []
            else:
                pts = []
            bbox = item.get("bbox")
        else:
            pts = item
            bbox = None

        xs, ys = [], []
        if isinstance(pts, list):
            for pt in pts:
                if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                    try:
                        xs.append(float(pt[0])); ys.append(float(pt[1]))
                    except Exception:
                        pass
                elif isinstance(pt, dict):
                    try:
                        xs.append(float(pt.get("x"))); ys.append(float(pt.get("y")))
                    except Exception:
                        pass

        if xs and ys:
            return min(xs), min(ys), max(xs), max(ys)

        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = map(float, bbox[:4])
                return x1, y1, x2, y2
            except Exception:
                return None
        return None

    def _load_sam2_boxes_for_image(self, img_path, W, H):
        sidecar = self._sam2_sidecar_path(img_path)
        if not sidecar.exists():
            return []
        try:
            data = json.loads(sidecar.read_text(encoding="utf-8", errors="ignore"))
        except Exception as exc:
            self.append_annotator_log(f"Could not read SAM2 sidecar {sidecar}: {exc}")
            return []

        if isinstance(data, dict):
            items = data.get("polygons") or data.get("segments") or data.get("masks") or []
        elif isinstance(data, list):
            items = data
        else:
            items = []

        boxes = []
        for item in items:
            bb = self._sam2_polygon_bbox(item)
            if not bb:
                continue
            x1, y1, x2, y2 = bb
            x1, x2 = sorted([max(0.0, min(float(W), x1)), max(0.0, min(float(W), x2))])
            y1, y2 = sorted([max(0.0, min(float(H), y1)), max(0.0, min(float(H), y2))])
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                boxes.append([0, x1, y1, x2, y2])
        return boxes

    def _yolo_line_from_box(self, cls, x1, y1, x2, y2, W, H):
        x1, x2 = sorted([max(0, min(W, x1)), max(0, min(W, x2))])
        y1, y2 = sorted([max(0, min(H, y1)), max(0, min(H, y2))])
        if x2 - x1 < 2 or y2 - y1 < 2:
            return None
        cx = ((x1 + x2) / 2.0) / W
        cy = ((y1 + y2) / 2.0) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H
        return f"{int(cls)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"

    def _read_yolo_label_lines(self, lab):
        if not Path(lab).exists():
            return []
        return [ln.strip() for ln in Path(lab).read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]

    def _negative_yolo_lines_from_current_boxes(self, img_path):
        if self.annotator_current_index < 0 or Path(img_path) != Path(self.annotator_image_paths[self.annotator_current_index]):
            return []
        pix = QPixmap(str(img_path))
        W = max(1, pix.width())
        H = max(1, pix.height())
        out = []
        for cls, x1, y1, x2, y2 in self.annotator_boxes:
            if int(cls) == 0:
                continue
            ln = self._yolo_line_from_box(cls, x1, y1, x2, y2, W, H)
            if ln:
                out.append(ln)
        return out

    def _write_sam2_positive_boxes_keep_negatives(self, img_path, positive_boxes):
        """Write SAM2 boxes and inherit negative YOLO classes.

         treats class 0 as positive and every non-zero class as negative.
        Existing YOLO/annotator false-positive boxes are used as class hints: when a
        SAM2 segment overlaps a negative YOLO box, that SAM2 segment is written as
        the same negative class instead of class 0. Unmatched negative YOLO boxes are
        still preserved so manual false-positive samples are never lost.
        """
        pix = QPixmap(str(img_path))
        W = max(1, pix.width())
        H = max(1, pix.height())
        lab = self._annotator_label_path(img_path)

        def _line_to_box(ln):
            parts = ln.split()
            if len(parts) != 5:
                return None
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = (cx - bw / 2.0) * W
                y1 = (cy - bh / 2.0) * H
                x2 = (cx + bw / 2.0) * W
                y2 = (cy + bh / 2.0) * H
                return [cls, x1, y1, x2, y2, ln]
            except Exception:
                return None

        def _overlap_score(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1 = max(ax1, bx1), max(ay1, by1)
            ix2, iy2 = min(ax2, bx2), min(ay2, by2)
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            if inter <= 0:
                return 0.0
            aa = max(1.0, (ax2 - ax1) * (ay2 - ay1))
            bb = max(1.0, (bx2 - bx1) * (by2 - by1))
            return max(inter / aa, inter / bb, inter / max(1.0, aa + bb - inter))

        def _center_inside(box, target):
            x1, y1, x2, y2 = box
            tx1, ty1, tx2, ty2 = target
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            return tx1 <= cx <= tx2 and ty1 <= cy <= ty2

        negative_boxes = []
        for ln in self._read_yolo_label_lines(lab):
            box = _line_to_box(ln)
            if box and int(box[0]) != 0:
                negative_boxes.append(box)

        # Keep negative boxes that were drawn in the visible annotator but not saved yet.
        for ln in self._negative_yolo_lines_from_current_boxes(img_path):
            box = _line_to_box(ln)
            if box and all(box[5] != old[5] for old in negative_boxes):
                negative_boxes.append(box)

        positive_lines = []
        sam_negative_lines = []
        matched_negative_indexes = set()
        for _cls, x1, y1, x2, y2 in positive_boxes:
            sam_box = (x1, y1, x2, y2)
            best_idx = None
            best_score = 0.0
            for idx, neg in enumerate(negative_boxes):
                neg_box = (neg[1], neg[2], neg[3], neg[4])
                score = _overlap_score(sam_box, neg_box)
                if _center_inside(sam_box, neg_box):
                    score = max(score, 1.0)
                if _center_inside(neg_box, sam_box):
                    score = max(score, 0.75)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            if best_idx is not None and best_score >= 0.10:
                neg_cls = int(negative_boxes[best_idx][0])
                ln = self._yolo_line_from_box(neg_cls, x1, y1, x2, y2, W, H)
                if ln and ln not in sam_negative_lines:
                    sam_negative_lines.append(ln)
                matched_negative_indexes.add(best_idx)
            else:
                ln = self._yolo_line_from_box(0, x1, y1, x2, y2, W, H)
                if ln and ln not in positive_lines:
                    positive_lines.append(ln)

        preserved_negative_lines = []
        for idx, neg in enumerate(negative_boxes):
            # If SAM2 produced a negative segment for this YOLO false-positive box,
            # the SAM2 box replaces the old YOLO rectangle. Otherwise preserve it.
            if idx not in matched_negative_indexes and neg[5] not in preserved_negative_lines:
                preserved_negative_lines.append(neg[5])

        lines = positive_lines + sam_negative_lines + preserved_negative_lines
        lab.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        return lab, len(positive_lines), len(sam_negative_lines) + len(preserved_negative_lines)

    def _yolo_lines_to_boxes(self, lines, W, H):
        boxes = []
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                continue
            try:
                cls = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:])
                x1 = (cx - bw / 2.0) * W
                y1 = (cy - bh / 2.0) * H
                x2 = (cx + bw / 2.0) * W
                y2 = (cy + bh / 2.0) * H
                boxes.append([cls, x1, y1, x2, y2])
            except Exception:
                continue
        return boxes

    def use_sam2_segments_current(self):
        if self.annotator_current_index < 0:
            QMessageBox.warning(self, "No image", "Load and select an image first.")
            return
        img_path = self.annotator_image_paths[self.annotator_current_index]
        pix = QPixmap(str(img_path))
        boxes = self._load_sam2_boxes_for_image(img_path, max(1, pix.width()), max(1, pix.height()))
        if not boxes:
            QMessageBox.warning(self, "No SAM2 segments", "No .sam2.json segments were found for the selected image. Run SAM2 first.")
            return
        lab, pos_n, neg_n = self._write_sam2_positive_boxes_keep_negatives(img_path, boxes)
        pix = QPixmap(str(img_path))
        self.annotator_boxes = self._yolo_lines_to_boxes(self._read_yolo_label_lines(lab), max(1, pix.width()), max(1, pix.height()))
        self.annotator_redraw_boxes()
        self.append_annotator_log(f"SAM2 labels written: {pos_n} positive boxes + {neg_n} negative boxes -> {lab}")

    def use_sam2_segments_all(self):
        folder = Path(self.annotator_images.text()).expanduser()
        if not folder.exists():
            QMessageBox.warning(self, "Missing image folder", "Choose an image folder first.")
            return
        if not getattr(self, "annotator_image_paths", None):
            self.load_annotator_images()
        total_images = 0
        total_boxes = 0
        total_negatives = 0
        for img_path in getattr(self, "annotator_image_paths", []):
            pix = QPixmap(str(img_path))
            boxes = self._load_sam2_boxes_for_image(img_path, max(1, pix.width()), max(1, pix.height()))
            if not boxes:
                continue
            lab, pos_n, neg_n = self._write_sam2_positive_boxes_keep_negatives(img_path, boxes)
            total_images += 1
            total_boxes += pos_n
            total_negatives += neg_n
        if total_images == 0:
            QMessageBox.warning(self, "No SAM2 segments", "No .sam2.json sidecars with polygons were found in the image folder. Run SAM2 all pictures first.")
            return
        if self.annotator_current_index >= 0:
            self.annotator_load_current()
        self.append_annotator_log(f"SAM2 labels written for /YOLO training: {total_boxes} positive boxes + {total_negatives} negative boxes from {total_images} images.")

    def _auto_use_sam2_segments_current_after_run(self):
        """After SAM2 succeeds, immediately convert the created sidecar to labels.

        This mirrors the standalone GUI workflow: SAM2 segments become class-0
        positive boxes automatically, while SAM2 boxes overlapping YOLO/manual false boxes inherit
        class-1 negative samples for  training.
        """
        try:
            self.append_annotator_log("SAM2 finished. Automatically converting current SAM2 segments to /YOLO labels...")
            self.use_sam2_segments_current()
        except Exception as exc:
            self.append_annotator_log(f"Automatic SAM2 label conversion failed: {exc}")

    def _auto_use_sam2_segments_all_after_run(self):
        """After batch SAM2 succeeds, convert all sidecars to training labels."""
        try:
            self.append_annotator_log("SAM2 finished. Automatically converting all SAM2 segments to /YOLO labels...")
            self.use_sam2_segments_all()
        except Exception as exc:
            self.append_annotator_log(f"Automatic SAM2 batch label conversion failed: {exc}")

    def install_sam2_auto(self):
        sam_dir = self.plugin_dir / "runtime_sam2"
        args = [
            self.plugin_dir / "scripts" / "mustatil_sam2_installer.py",
            "--dest", str(sam_dir),
            "--model", self.sam2_model_choice.currentText(),
        ]
        self._start_external_process(args, "Install SAM2")
        # Pre-fill likely checkpoint/config paths immediately.
        model = self.sam2_model_choice.currentText()
        ckpt_names = {
            "tiny": "sam2_hiera_tiny.pt",
            "small": "sam2_hiera_small.pt",
            "base_plus": "sam2_hiera_base_plus.pt",
            "large": "sam2_hiera_large.pt",
        }
        cfg_names = {
            "tiny": "sam2_hiera_t.yaml",
            "small": "sam2_hiera_s.yaml",
            "base_plus": "sam2_hiera_b+.yaml",
            "large": "sam2_hiera_l.yaml",
        }
        self.sam2_checkpoint.setText(str(sam_dir / ckpt_names.get(model, "sam2_hiera_base_plus.pt")))
        self.sam2_model_cfg.setText(cfg_names.get(model, "sam2_hiera_b+.yaml"))

    def run_sam2_current(self):
        if self.annotator_current_index < 0:
            QMessageBox.warning(self, "No image", "Load and select an image first.")
            return
        img = self.annotator_image_paths[self.annotator_current_index]
        out = Path(self.annotator_labels.text() or Path(img).parent / "sam2") / "sam2_current"
        ckpt = self.sam2_checkpoint.text().strip()
        if not ckpt or not Path(ckpt).exists():
            # Let the runner auto-resolve a bundled/default SAM2 checkpoint if present.
            self.append_annotator_log("SAM2 checkpoint not selected/found. Trying automatic base_plus/default checkpoint resolution.")
            ckpt = ""
        args = [
            self.plugin_dir / "scripts" / "mustatil_sam2_runner.py",
            "--images", str(img),
            "--output", str(out),
            "--mode", "one",
            "--checkpoint", ckpt,
            "--model-cfg", self.sam2_model_cfg.text().strip(),
            "--labels-dir", self.annotator_labels.text().strip(),
        ]
        self._start_external_process(
            args,
            "SAM2 current image",
            switch_to_log=False,
            on_success=self._auto_use_sam2_segments_current_after_run,
        )

    def run_sam2_all(self):
        folder = self.annotator_images.text().strip()
        if not folder:
            QMessageBox.warning(self, "No image folder", "Choose an image folder first.")
            return
        out = Path(self.annotator_labels.text() or Path(folder) / "sam2") / "sam2_all"
        ckpt = self.sam2_checkpoint.text().strip()
        if not ckpt or not Path(ckpt).exists():
            # Let the runner auto-resolve a bundled/default SAM2 checkpoint if present.
            self.append_annotator_log("SAM2 checkpoint not selected/found. Trying automatic base_plus/default checkpoint resolution.")
            ckpt = ""
        args = [
            self.plugin_dir / "scripts" / "mustatil_sam2_runner.py",
            "--images", folder,
            "--output", str(out),
            "--mode", "all",
            "--checkpoint", ckpt,
            "--model-cfg", self.sam2_model_cfg.text().strip(),
            "--labels-dir", self.annotator_labels.text().strip(),
        ]
        self._start_external_process(
            args,
            "SAM2 all pictures",
            switch_to_log=False,
            on_success=self._auto_use_sam2_segments_all_after_run,
        )

    def open_annotator(self):
        self.load_annotator_images()

    def _make_runtime_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)

        group = QGroupBox("External Python Runtime")
        form = QFormLayout(group)

        self.python_path = QLineEdit(self.detect_existing_runtime())
        browse_py = QPushButton("Browse")
        browse_py.clicked.connect(self.pick_python)
        pyrow = QHBoxLayout()
        pyrow.addWidget(self.python_path)
        pyrow.addWidget(browse_py)
        form.addRow("Python.exe", pyrow)

        self.model_path = QLineEdit(str(self.plugin_dir / "models" / "bestf260.onnx"))
        form.addRow("ONNX model", self.model_path)

        self.runtime_training_images = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_Project" / "images"))
        rt_img_btn = QPushButton("Browse")
        rt_img_btn.clicked.connect(self.pick_runtime_training_images)
        rt_img_row = QHBoxLayout()
        rt_img_row.addWidget(self.runtime_training_images)
        rt_img_row.addWidget(rt_img_btn)
        form.addRow("Training images", rt_img_row)

        self.runtime_training_labels = QLineEdit(str(Path.home() / "Desktop" / "Mustatil_Project" / "labels"))
        rt_lab_btn = QPushButton("Browse")
        rt_lab_btn.clicked.connect(self.pick_runtime_training_labels)
        rt_lab_row = QHBoxLayout()
        rt_lab_row.addWidget(self.runtime_training_labels)
        rt_lab_row.addWidget(rt_lab_btn)
        form.addRow("Training labels", rt_lab_row)

        use_training_btn = QPushButton("Use selected training data in YOLO Trainer")
        use_training_btn.clicked.connect(self.use_runtime_training_data)
        form.addRow("", use_training_btn)

        install_btn = QPushButton("Install external runtime")
        install_btn.clicked.connect(self.install_runtime)
        form.addRow("", install_btn)

        manual_btn = QPushButton("Open plugin folder")
        manual_btn.clicked.connect(self.open_plugin_folder)
        form.addRow("", manual_btn)

        layout.addWidget(group)

        info = QLabel(
            "The AI runtime is external to QGIS. Training image/label folders can be selected here and applied to the YOLO Trainer. This avoids DLL conflicts with "
            "QGIS Python, GDAL, ONNX Runtime, Torch, Fiona and Rasterio."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        layout.addStretch(1)
        return self._wrap_scroll(page)

    def _make_log_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        layout.addWidget(self.log)
        clear = QPushButton("Clear log")
        clear.clicked.connect(self.log.clear)
        layout.addWidget(clear)
        return page

    def update_preset_fields(self, *args):
        """Models removed: always use the loaded custom YOLOv8 .pt model."""
        if hasattr(self, "custom_model_path"):
            self.custom_model_path.setEnabled(True)
            p = self.custom_model_path.text().strip()
            if p and hasattr(self, "model_path"):
                self.model_path.setText(p)
        if hasattr(self, "custom_model_button"):
            self.custom_model_button.setEnabled(True)

    def pick_custom_model(self):
        start = self.custom_model_path.text().strip() if hasattr(self, "custom_model_path") else ""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Custom YOLOv8 .pt Model",
            start,
            "YOLOv8 model (*.pt);;Model files (*.onnx *.pt *.pth);;All files (*.*)"
        )
        if path:
            self.custom_model_path.setText(path)
            if hasattr(self, "model_path"):
                self.model_path.setText(path)
            try:
                self.append_log(f"Loaded custom YOLOv8 model: {path}", switch_to_log=False)
            except Exception:
                pass

    def pick_output_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_dir.text())
        if d:
            self.output_dir.setText(d)

    def pick_python(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select python.exe", "", "Python (*.exe)")
        if p:
            self.python_path.setText(p)

    def append_log(self, text, switch_to_log=False):
        self.log.append(str(text).rstrip())
        if switch_to_log:
            self.tabs.setCurrentWidget(self.log_tab)

    def append_annotator_log(self, text):
        # Do not steal focus from the annotator tab for routine annotator actions.
        self.append_log(text, switch_to_log=False)

    def set_detection_progress(self, value, text=None, busy=False):
        try:
            if busy:
                self.progress.setRange(0, 0)
                self.progress_percent.setText("working | Raster 0/0")
            else:
                self.progress.setRange(0, 100)
                value = max(0, min(100, int(value)))
                self.progress.setValue(value)
                self.progress_percent.setText(f"{value}%")
            self.progress.show()
            self.progress_percent.show()
            if text:
                self.status.setText(text)
        except Exception:
            pass

    def clear_selection_outline(self):
        try:
            if hasattr(self, "rect_tool") and self.rect_tool:
                try:
                    self.rect_tool.clear()
                except Exception:
                    pass
            if hasattr(self, "_active_selection_rubber") and self._active_selection_rubber:
                try:
                    self._active_selection_rubber.reset(QgsWkbTypes.LineGeometry)
                except Exception:
                    pass
                self._active_selection_rubber = None
        except Exception:
            pass

    def cancel_detection(self):
        try:
            if self.process is not None and self.process.state() != QProcess.NotRunning:
                self.process.kill()
                self.append_log("Detection cancelled by user.")
                self.status.setText("Detection cancelled.")
                self.stop_detection_heartbeat()
                self.progress.setRange(0, 100)
                self.progress.setValue(0)
                self.progress_percent.setText("cancelled")
        except Exception as e:
            self.append_log(f"Cancel failed: {e}")

    def select_area(self):
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            QMessageBox.warning(self, "Mustatil", "Please select a raster layer first.")
            return
        self.clear_selection_outline()
        self.rect_tool = RectTool(self.canvas, self._area_selected)
        self._active_selection_rubber = self.rect_tool.rubber
        self.canvas.setMapTool(self.rect_tool)
        self.status.setText("Draw a rectangle in the QGIS map canvas. Detection starts automatically after selection.")

    def run_full_extent(self):
        layer = self.layer_combo.currentLayer()
        if not layer or not isinstance(layer, QgsRasterLayer):
            QMessageBox.warning(self, "Mustatil", "Please select a raster layer first.")
            return
        self.set_detection_progress(1, "Running detection on selected raster extent...")
        self._area_selected(layer.extent())

    def _area_selected(self, rect):
        layer = self.layer_combo.currentLayer()
        if not layer:
            return
        out_dir = Path(self.output_dir.text()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        clip_dir = out_dir / "Mustatil_QGIS_Clips"
        clip_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clip_path = clip_dir / f"mustatil_qgis_clip_{stamp}.tif"
        gpkg_path = out_dir / "mustatil_qgis_detections.gpkg"

        try:
            self.set_detection_progress(5, "Exporting selected raster area...")
            QApplication.processEvents()
            self.export_clip(layer, rect, str(clip_path))
            QApplication.processEvents()
            self.set_detection_progress(25, "Clip exported. Starting detection...")
        except Exception as e:
            self.append_log("Raster clip failed:")
            self.append_log(str(e))
            QMessageBox.critical(self, "Export failed", str(e))
            return

        if not clip_path.exists() or clip_path.stat().st_size < 100:
            msg = f"Clip export failed: file was not created or is empty: {clip_path}"
            self.append_log(msg)
            QMessageBox.critical(self, "Clip missing", msg)
            return

        self.last_clip = str(clip_path)
        self.last_output = str(gpkg_path)
        self.last_detection_output = str(gpkg_path)
        self.append_log(f"Clip exported and verified: {clip_path} ({clip_path.stat().st_size} bytes)")
        self.run_analysis(str(clip_path), str(gpkg_path))

    def rerun_last_detection(self):
        if not self.last_clip:
            QMessageBox.warning(
                self,
                "No previous detection",
                "Run a detection first."
            )
            return

        out_dir = Path(self.output_dir.text()).expanduser()
        gpkg_path = out_dir / "mustatil_qgis_detections.gpkg"

        self.last_output = str(gpkg_path)
        self.append_log("Rerunning detection with updated slider values...")
        self.run_analysis(self.last_clip, str(gpkg_path))

    def _detection_filter_expression(self):
        conf = self.conf_slider.value() / 100.0
        form = 0.0  # Form scoring removed: do not filter by form score
        return f'"score" >= {conf:.6f} AND "formscore" >= {form:.6f}'

    def add_filtered_detection_layer(self, gpkg_path=None, layer_prefix="Mustatil detections"):
        gpkg_path = gpkg_path or getattr(self, "last_detection_output", "") or self.last_output
        if not gpkg_path or not Path(gpkg_path).exists():
            QMessageBox.warning(self, "No detections", "Run a detection first. The sliders can only filter an existing candidate GeoPackage.")
            return None

        conf = self.conf_slider.value() / 100.0
        form = 0.0  # Form scoring removed: do not filter by form score
        layer_name = f"{layer_prefix} conf≥{conf:.2f} form≥{form:.2f}"
        uri = f"{gpkg_path}|layername=mustatile_detections"
        layer = QgsVectorLayer(uri, layer_name, "ogr")
        if not layer.isValid():
            layer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
        if not layer.isValid():
            self.append_log(f"Could not load detection layer for filtering: {gpkg_path}")
            return None
        try:
            layer.setSubsetString(self._detection_filter_expression())
        except Exception as e:
            self.append_log(f"Could not apply detection filter expression: {e}")
        self.style_detection_layer_red_boxes(layer)
        QgsProject.instance().addMapLayer(layer)
        self.last_filtered_layer_id = layer.id()
        try:
            self.detection_count_label.setText(f"Detections: {layer.featureCount()}")
        except Exception:
            pass
        self.append_log(f"Filtered detection layer added: confidence >= {conf:.2f}")
        return layer

    def create_filtered_layer_from_sliders(self):
        # Avoid warnings while sliders are still being created during UI setup.
        if not hasattr(self, "last_detection_output") and not getattr(self, "last_output", ""):
            return
        return self.add_filtered_detection_layer()

    def export_visible_detections(self):
        """Export the currently filtered/visible detection layer using the
        same GeoPackage layer name/schema as the original detection output.

        Older plugin versions wrote detections to a GeoPackage layer named
        `mustatile_detections`. Keeping that layer name is important for QGIS
        reloads and for downstream tools that expect the old output format.
        """
        layer = None
        try:
            if getattr(self, "last_filtered_layer_id", ""):
                layer = QgsProject.instance().mapLayer(self.last_filtered_layer_id)
            if layer is None:
                active = self.iface.activeLayer()
                if isinstance(active, QgsVectorLayer):
                    layer = active
        except Exception:
            layer = None

        if layer is None or not isinstance(layer, QgsVectorLayer) or not layer.isValid():
            QMessageBox.warning(self, "No detection layer", "Create or select a filtered detection layer first.")
            return

        out_dir = Path(self.output_dir.text()).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        default_path = out_dir / f"mustatil_qgis_detections_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.gpkg"
        out_file, _ = QFileDialog.getSaveFileName(self, "Export detections", str(default_path), "GeoPackage (*.gpkg)")
        if not out_file:
            return
        if not out_file.lower().endswith(".gpkg"):
            out_file += ".gpkg"

        # Match the old detection GeoPackage format as closely as possible:
        # a single GPKG layer named `mustatile_detections` with the existing
        # detection attributes (score, formscore, class_id, model, model,
        # ) and georeferenced polygon geometry. The subset string
        # on the QGIS layer is respected, so only the currently filtered
        # detections are exported.
        try:
            out_path = Path(out_file)
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass

        opts = QgsVectorFileWriter.SaveVectorOptions()
        opts.driverName = "GPKG"
        opts.layerName = "mustatile_detections"
        opts.onlySelectedFeatures = False
        opts.actionOnExistingFile = QgsVectorFileWriter.CreateOrOverwriteFile

        result = QgsVectorFileWriter.writeAsVectorFormatV3(
            layer,
            out_file,
            QgsCoordinateTransformContext(),
            opts
        )
        err_code = result[0] if isinstance(result, tuple) else result
        if err_code == QgsVectorFileWriter.NoError:
            self.append_log(f"Detections exported in old GeoPackage format: {out_file}")
            QMessageBox.information(self, "Exported", f"Detections exported:\n{out_file}")
        else:
            self.append_log(f"Detection export failed: {result}")
            QMessageBox.critical(self, "Export failed", str(result))

    def export_clip(self, layer, rect, out_tif):
        """
        Export selected area as GeoTIFF without GDAL export calls inside QGIS.

        The selected map area is rendered by QGIS and written as a georeferenced
        GeoTIFF using tifffile. This avoids QGIS provider URI problems such as
        type=xyz&url=... for Google/XYZ layers.
        """
        out_tif = str(out_tif)
        out_path = Path(out_tif)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if out_path.exists():
                out_path.unlink()
        except Exception:
            pass

        def _valid_output():
            return out_path.exists() and out_path.stat().st_size > 100

        def _write_rendered_geotiff():
            import numpy as np
            try:
                from PIL import Image
            except Exception as exc:
                raise RuntimeError(
                    "Pillow (PIL) is required for TIFF export."
                ) from exc

            canvas_size = self.canvas.mapSettings().outputSize()

            canvas_extent = self.canvas.extent()
            map_units_per_px_x = canvas_extent.width() / max(1, canvas_size.width())
            map_units_per_px_y = canvas_extent.height() / max(1, canvas_size.height())

            width = max(256, int(round(rect.width() / max(map_units_per_px_x, 1e-12))))
            height = max(256, int(round(rect.height() / max(map_units_per_px_y, 1e-12))))

            # For large WebMercator-like coordinate ranges, approximate zoom 18.
            # Avoid CRS database lookups here; broken PROJ installations can fail before rendering.
            try:
                if max(abs(rect.xMinimum()), abs(rect.xMaximum()), abs(rect.yMinimum()), abs(rect.yMaximum())) > 1000:
                    meters_per_pixel_z18 = 156543.03392804097 / (2 ** 18)
                    width = max(width, int(round(rect.width() / meters_per_pixel_z18)))
                    height = max(height, int(round(rect.height() / meters_per_pixel_z18)))
            except Exception:
                pass

            max_side = self.max_clip_size.value() if hasattr(self, "max_clip_size") else 6144
            width = min(width, max_side)
            height = min(height, max_side)

            max_pixels = max_side * max_side
            if width * height > max_pixels:
                scale = (max_pixels / float(width * height)) ** 0.5
                width = max(256, int(width * scale))
                height = max(256, int(height * scale))

            try:
                self.update_download_stats(1, 1, "Rendering TIFF")
                self.append_log(
                    f"Rendering detection clip without GDAL at {width}x{height}px.",
                    switch_to_log=False
                )
                self.status.setText(f"Rendering clip {width}x{height}px...")
                self.progress_percent.setText(f"rendering | {width}x{height}px")
                QApplication.processEvents()
            except Exception:
                pass

            settings = QgsMapSettings()
            settings.setLayers([layer])
            # Important: do not force a destination CRS here.
            # Some systems have a conflicting PROJ database; QGIS can still render
            # the provider layer in its native/map coordinates without this call.
            settings.setExtent(rect)
            settings.setOutputSize(QSize(width, height))
            settings.setBackgroundColor(Qt.white)

            job = QgsMapRendererParallelJob(settings)
            job.start()
            job.waitForFinished()
            image = job.renderedImage()

            if image.isNull():
                raise RuntimeError("QGIS rendered image is null.")

            image = image.convertToFormat(QImage.Format_RGB888)
            ptr = image.bits()
            ptr.setsize(image.height() * image.width() * 3)
            arr = np.frombuffer(ptr, np.uint8).reshape((image.height(), image.width(), 3)).copy()

            img = Image.fromarray(arr)
            img.save(out_tif, format="TIFF", compression="tiff_deflate")

            # Write worldfile + CRS sidecar for the external analyzer.
            # This avoids GDAL/PROJ calls inside QGIS but preserves map placement.
            pixel_x = rect.width() / float(width)
            pixel_y = rect.height() / float(height)
            tfw_path = str(out_path.with_suffix(".tfw"))
            with open(tfw_path, "w", encoding="utf-8") as wf:
                wf.write(f"{pixel_x:.12f}\n")
                wf.write("0.0\n")
                wf.write("0.0\n")
                wf.write(f"{-pixel_y:.12f}\n")
                wf.write(f"{rect.xMinimum() + pixel_x / 2.0:.12f}\n")
                wf.write(f"{rect.yMaximum() - pixel_y / 2.0:.12f}\n")

            epsg = 3857 if max(abs(rect.xMinimum()), abs(rect.xMaximum()), abs(rect.yMinimum()), abs(rect.yMaximum())) > 1000 else 4326
            crs_sidecar = out_path.with_suffix(".crs.json")
            crs_sidecar.write_text(
                json.dumps({"epsg": epsg}, indent=2),
                encoding="utf-8"
            )

            if not _valid_output():
                raise RuntimeError("GeoTIFF was not created.")

            try:
                self.append_log(f"Raster exported without GDAL: {out_tif}", switch_to_log=False)
            except Exception:
                pass

        _write_rendered_geotiff()
        return

    def reset_download_stats(self):
        try:
            import time
            self.download_tile_start_time = time.time()
            self.download_tile_last = 0
            self.download_tiles_label.setText("Tiles: 0/0")
            self.download_speed_label.setText("Speed: 0.00 tiles/sec")
            self.download_phase_label.setText("Phase: preparing TIFF")
        except Exception:
            pass

    def update_download_stats(self, current, total, phase="Downloading TIFF"):
        try:
            import time
            current = int(current)
            total = max(1, int(total))

            self.download_tiles_label.setText(f"Tiles: {current}/{total}")
            self.download_phase_label.setText(f"Phase: {phase}")

            if self.download_tile_start_time is not None:
                elapsed = max(0.001, time.time() - self.download_tile_start_time)
                speed = current / elapsed
                self.download_speed_label.setText(
                    f"Speed: {speed:.2f} tiles/sec"
                )
        except Exception:
            pass

    def detection_heartbeat_tick(self):
        try:
            if self.process is None:
                self.detection_heartbeat_timer.stop()
                return
            if self.process.state() == QProcess.NotRunning:
                self.detection_heartbeat_timer.stop()
                return

            # Keep visible feedback before real TILE_PROGRESS arrives.
            self.detection_heartbeat_value = (self.detection_heartbeat_value + 1) % 100
            if self.progress.maximum() == 0:
                self.progress_percent.setText("running | waiting for raster progress")
            elif self.progress.value() < 5:
                self.progress.setValue(5)
                self.progress_percent.setText("5% | preparing detection")
            self.status.setText("Detection running... rendering/downloading tiles or waiting for raster progress")
        except Exception:
            pass

    def start_detection_heartbeat(self):
        try:
            self.detection_heartbeat_value = 0
            self.detection_heartbeat_timer.start(1000)
        except Exception:
            pass

    def stop_detection_heartbeat(self):
        try:
            self.detection_heartbeat_timer.stop()
        except Exception:
            pass

    def run_analysis(self, clip_path, gpkg_path):
        clip_file = Path(clip_path)
        if not clip_file.exists() or clip_file.stat().st_size < 100:
            msg = f"Input GeoTIFF missing before analysis: {clip_file}"
            self.append_log(msg)
            QMessageBox.critical(self, "Missing clip", msg)
            return

        # Resolve runtime Python robustly. Empty QLineEdit must not become Path(".").
        py_text = self.python_path.text().strip() if hasattr(self, "python_path") else ""
        py = Path(py_text) if py_text else None

        if py is None or str(py) == "." or not py.exists() or py.is_dir():
            detected = self.detect_existing_runtime()
            py = Path(detected) if detected else self.plugin_dir / "runtime" / "python" / "python.exe"

        if not py.exists() or py.is_dir():
            QMessageBox.warning(self, "Runtime missing", "Python runtime not found. Use the Runtime tab first.")
            self.tabs.setCurrentIndex(3 if self.tabs.count() > 3 else 1)
            return

        if hasattr(self, "python_path"):
            self.python_path.setText(str(py))

        analyzer = self.plugin_dir / "scripts" / "mustatil_run_analyzer_wrapper.py"
        if hasattr(self, "model_path") and self.model_path.text().strip():
            model = Path(self.model_path.text())
        elif hasattr(self, "custom_model_path") and self.custom_model_path.text().strip():
            model = Path(self.custom_model_path.text())
        else:
            QMessageBox.warning(self, "Model missing", "Please load a custom YOLOv8 .pt model first.")
            return


        cmd = [
            str(py), "-u", str(analyzer),
            "--input", clip_path,
            "--output", gpkg_path,
            "--model", str(model),
            "--conf", str(self.conf_slider.value() / 100.0),
            "--tile-size", str(self.tile_size.value()),
            "--overlap", str(self.overlap.value()),
        ]
        cmd.extend(["--preset-name", "Custom YOLOv8"])
        detection_device = "cpu"
        try:
            detection_device = "cuda" if self.device_combo.currentText().strip().lower() == "cuda" else "cpu"
        except Exception:
            detection_device = "cpu"
        cmd.extend(["--device", detection_device])
        self.append_log("Running Mustatil model... Runtime dependencies will be checked automatically.", switch_to_log=False)
        self.append_log(" ".join(f'"{c}"' if " " in c else c for c in cmd))
        self.set_detection_progress(30, "Analysis running...", busy=True)

        self.process = QProcess(self)
        self.process.setProgram(cmd[0])
        self.process.setArguments(cmd[1:])
        self.process.setProcessChannelMode(QProcess.MergedChannels)

        env = self.process.processEnvironment()
        if env.isEmpty():
            from qgis.PyQt.QtCore import QProcessEnvironment
            env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONUNBUFFERED", "1")
        env.insert("PYTHONIOENCODING", "utf-8")
        self.process.setProcessEnvironment(env)

        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)

        self.start_detection_heartbeat()
        self.append_log("Detection process started. Waiting for raster/tile progress...", switch_to_log=False)
        self.reset_download_stats()
        self.append_log("XYZ tile download started...", switch_to_log=False)
        self.process.start()

    def _read_stdout(self):
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        if data.strip():
            self.append_log(data)
            if self.progress.maximum() == 0:
                self.progress.setRange(0, 100)
                self.progress.setValue(max(5, self.progress.value()))
                self.progress_percent.setText("5% | Runtime dependency check")

            # Tile/raster progress parsing.
            wrapper_match = re.search(r"WRAPPER_STATUS:(.*)", data)
            if wrapper_match:
                msg = wrapper_match.group(1).strip()
                self.status.setText(f"Detection: {msg}")
                self.progress_percent.setText(f"running | {msg}")

            tile_match = re.search(r"TILE_PROGRESS:(\d+)/(\d+)", data)
            if tile_match:
                current_tile = int(tile_match.group(1))
                total_tiles = max(1, int(tile_match.group(2)))

                # Keep percentages sane and below 100 until finished callback.
                percent = int((current_tile / total_tiles) * 95.0)
                percent = max(0, min(95, percent))

                self.progress.setRange(0, 100)
                self.progress.setValue(percent)
                self.progress.show()
                self.progress_percent.show()
                self.progress_percent.setText(
                    f"{percent}% | Raster {current_tile}/{total_tiles}"
                )
                self.status.setText(
                    f"Detection running... Raster {current_tile}/{total_tiles}"
                )

            count_match = re.search(r"DETECTION_COUNT:(\d+)", data)
            if count_match:
                self.detection_count_label.setText(f"Detections: {count_match.group(1)}")

            prog_match = re.search(r"PROGRESS\\s*[:=]\\s*(\\d{1,3})", data)
            if prog_match:
                value = max(0, min(100, int(prog_match.group(1))))
                self.set_detection_progress(value, f"Analysis running... {value}%")

    def _read_stderr(self):
        data = bytes(self.process.readAllStandardError()).decode("utf-8", errors="replace")
        if data.strip():
            self.append_log(data)

    def style_detection_layer_red_boxes(self, layer):
        """
        Style detection polygons as red outline boxes with transparent fill.
        """
        try:
            symbol = QgsFillSymbol.createSimple({
                "color": "255,0,0,0",
                "outline_color": "255,0,0,255",
                "outline_width": "0.8",
                "outline_style": "solid",
                "style": "no",
            })
            layer.renderer().setSymbol(symbol)
            layer.triggerRepaint()
        except Exception as e:
            self.append_log(f"Could not style detection layer: {e}")



    def load_detection_gpkg_as_red_boxes(self, gpkg_path=None, layer_name="Mustatil detections", attempt=0):
        """
        Old-style QGIS import:
        Load file.gpkg|layername=mustatile_detections through iface.addVectorLayer,
        style as red boxes, and retry briefly if the GPKG was just written.
        """
        try:
            gpkg_path = gpkg_path or getattr(self, "last_detection_output", "") or getattr(self, "last_output", "")
            if not gpkg_path:
                self.append_log("No detection GeoPackage path available for map import.", switch_to_log=False)
                return False

            gpkg_path = str(gpkg_path)
            if not Path(gpkg_path).exists() or Path(gpkg_path).stat().st_size < 100:
                if attempt < 8:
                    QTimer.singleShot(500, lambda: self.load_detection_gpkg_as_red_boxes(gpkg_path, layer_name, attempt + 1))
                    return False
                self.append_log(f"Detection GeoPackage not ready or missing: {gpkg_path}", switch_to_log=False)
                return False

            # Remove old auto-loaded layer with same source to avoid duplicates.
            try:
                for lyr in list(QgsProject.instance().mapLayers().values()):
                    try:
                        if isinstance(lyr, QgsVectorLayer) and gpkg_path.replace("\\", "/") in lyr.source().replace("\\", "/"):
                            QgsProject.instance().removeMapLayer(lyr.id())
                    except Exception:
                        pass
            except Exception:
                pass

            uri = f"{gpkg_path}|layername=mustatile_detections"

            # Exactly the old QGIS way: iface.addVectorLayer adds it directly to the layer tree.
            vlayer = None
            try:
                vlayer = self.iface.addVectorLayer(uri, layer_name, "ogr")
            except Exception as e:
                self.append_log(f"iface.addVectorLayer failed, trying QgsVectorLayer fallback: {e}", switch_to_log=False)

            if vlayer is None or not vlayer.isValid():
                vlayer = QgsVectorLayer(uri, layer_name, "ogr")
                if vlayer.isValid():
                    QgsProject.instance().addMapLayer(vlayer)

            # Fallback: open first GPKG layer.
            if vlayer is None or not vlayer.isValid():
                vlayer = QgsVectorLayer(gpkg_path, layer_name, "ogr")
                if vlayer.isValid():
                    QgsProject.instance().addMapLayer(vlayer)

            if vlayer is None or not vlayer.isValid():
                if attempt < 8:
                    self.append_log(f"Detection layer not ready yet, retrying import... attempt {attempt + 1}", switch_to_log=False)
                    QTimer.singleShot(500, lambda: self.load_detection_gpkg_as_red_boxes(gpkg_path, layer_name, attempt + 1))
                    return False
                self.append_log(f"Could not load detections layer into QGIS: {uri}", switch_to_log=False)
                return False

            self.style_detection_layer_red_boxes(vlayer)

            try:
                vlayer.setName("Mustatil detections")
            except Exception:
                pass

            try:
                count = vlayer.featureCount()
                self.detection_count_label.setText(f"Detections: {count}")
                self.append_log(f"Detection layer loaded into QGIS: {count} red boxes", switch_to_log=False)
            except Exception:
                self.append_log("Detection layer loaded into QGIS as red boxes.", switch_to_log=False)

            try:
                extent = vlayer.extent()
                if extent and not extent.isEmpty():
                    self.canvas.setExtent(extent)
                    self.canvas.refresh()
            except Exception:
                pass

            return True

        except Exception as e:
            self.append_log(f"Automatic detection layer import failed: {e}", switch_to_log=False)
            if attempt < 8:
                QTimer.singleShot(500, lambda: self.load_detection_gpkg_as_red_boxes(gpkg_path, layer_name, attempt + 1))
            return False


    def update_detection_count_display(self):
        count = None
        gpkg_path = None
        try:
            if hasattr(self, "last_detection_output"):
                gpkg_path = str(self.last_detection_output)
            elif hasattr(self, "last_output"):
                gpkg_path = str(self.last_output)

            if gpkg_path:
                count_file = Path(gpkg_path).with_suffix(".count.txt")
                if count_file.exists():
                    count = int(count_file.read_text(encoding="utf-8").strip())
        except Exception:
            count = None

        try:
            if count is None and gpkg_path and Path(gpkg_path).exists():
                uri = gpkg_path + "|layername=mustatile_detections"
                lyr = QgsVectorLayer(uri, "count_tmp", "ogr")
                if not lyr.isValid():
                    lyr = QgsVectorLayer(gpkg_path, "count_tmp", "ogr")
                if lyr.isValid():
                    count = lyr.featureCount()
        except Exception:
            pass

        if count is None:
            count = 0

        try:
            self.detection_count_label.setText(f"Detections: {count}")
            self.append_log(f"Detection count: {count}")
        except Exception:
            pass

    def _finished(self, code, status):
        self.stop_detection_heartbeat()
        if code == 0:
            self.set_detection_progress(100, "Finished. Raster processing complete.")
            self.update_detection_count_display()
            self.download_phase_label.setText("Phase: finished")
            self.append_log("Analysis finished.", switch_to_log=False)

            gpkg_path = getattr(self, "last_detection_output", "") or getattr(self, "last_output", "")
            if not gpkg_path and hasattr(self, "output_path") and self.output_path.text().strip():
                gpkg_path = self.output_path.text().strip()

            # Always import detections into QGIS as red boxes, like the older workflow.
            QTimer.singleShot(700, lambda gpkg_path=gpkg_path: self.load_detection_gpkg_as_red_boxes(gpkg_path, "Mustatil detections"))

        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)
            self.progress_percent.setText("0%")
            self.progress.show()
            self.progress_percent.show()
            self.status.setText("Error.")
            self.append_log(f"Process exited with code {code}.")

    def pick_runtime_training_images(self):
        d = QFileDialog.getExistingDirectory(self, "Select training image folder", self.runtime_training_images.text())
        if d:
            self.runtime_training_images.setText(d)

    def pick_runtime_training_labels(self):
        d = QFileDialog.getExistingDirectory(self, "Select training label folder", self.runtime_training_labels.text())
        if d:
            self.runtime_training_labels.setText(d)

    def use_runtime_training_data(self):
        if hasattr(self, "train_source_images"):
            self.train_source_images.setText(self.runtime_training_images.text().strip())
        if hasattr(self, "train_source_labels"):
            self.train_source_labels.setText(self.runtime_training_labels.text().strip())
        if hasattr(self, "train_project_folder") and self.train_project_folder.text().strip():
            self.train_data_yaml.setText(str(Path(self.train_project_folder.text().strip()) / "yolo_datasets" / "data.yaml"))
        self.append_log("Selected runtime training data was applied to the YOLO Trainer tab.")

    def install_runtime(self):
        # Do not use sys.executable here: inside QGIS it can point to qgis.exe,
        # which opens a second QGIS instance instead of running the installer.
        bat = self.plugin_dir / "INSTALL_EXTERNAL_RUNTIME.bat"
        if not bat.exists():
            QMessageBox.critical(self, "Missing file", str(bat))
            return
        try:
            self._runtime_installer_process = subprocess.Popen(["cmd.exe", "/c", str(bat)], cwd=str(self.plugin_dir))
            self.append_log("Runtime installer started via INSTALL_EXTERNAL_RUNTIME.bat.")
        except Exception as exc:
            QMessageBox.critical(self, "Runtime installer failed", str(exc))
            self.append_log(f"Runtime installer failed: {exc}")

    def open_plugin_folder(self):
        try:
            os.startfile(str(self.plugin_dir))
        except Exception as e:
            QMessageBox.warning(self, "Open folder failed", str(e))

class MustatilQGISPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.plugin_dir = os.path.dirname(__file__)
        self.actions = []
        self.menu = None  # legacy, not used for QGIS Extensions menu
        self.toolbar = None
        self.dock = None

    def initGui(self):
        """Register the plugin cleanly under QGIS Extensions/Plugins menu."""
        self.menu_title = "&MustatilQGIS"

        self.toolbar = QToolBar("MustatilQGIS")
        self.toolbar.setObjectName(TOOLBAR_OBJECT_NAME)
        self.iface.addToolBar(self.toolbar)

        icon_path = os.path.join(self.plugin_dir, "icons", "icon.svg")
        if not os.path.exists(icon_path):
            icon_path = os.path.join(self.plugin_dir, "icons", "mustatil_logo.png")

        self.action_open = QAction(QIcon(icon_path), "Mustatil AI Detection", self.iface.mainWindow())
        self.action_open.triggered.connect(self.show_dock)
        self.toolbar.addAction(self.action_open)
        self.iface.addPluginToMenu(self.menu_title, self.action_open)
        self.actions.append(self.action_open)

        self.action_training = QAction("Open YOLO Trainer", self.iface.mainWindow())
        self.action_training.triggered.connect(self.open_training_direct)
        self.iface.addPluginToMenu(self.menu_title, self.action_training)
        self.actions.append(self.action_training)

        self.action_annotator = QAction("Open Annotator", self.iface.mainWindow())
        self.action_annotator.triggered.connect(self.open_annotator_direct)
        self.iface.addPluginToMenu(self.menu_title, self.action_annotator)
        self.actions.append(self.action_annotator)

        self.action_install = QAction("Install external runtime", self.iface.mainWindow())
        self.action_install.triggered.connect(self.install_runtime_direct)
        self.iface.addPluginToMenu(self.menu_title, self.action_install)
        self.actions.append(self.action_install)

    def unload(self):
        """Remove toolbar, dock and QGIS Extensions/Plugins menu entries."""
        menu_title = getattr(self, "menu_title", "&MustatilQGIS")
        for a in self.actions:
            try:
                self.iface.removePluginMenu(menu_title, a)
            except Exception:
                pass
            try:
                if self.toolbar:
                    self.toolbar.removeAction(a)
            except Exception:
                pass

        if self.toolbar:
            self.iface.mainWindow().removeToolBar(self.toolbar)
            self.toolbar = None

        if self.dock:
            self.iface.removeDockWidget(self.dock)
            self.dock = None

    def show_dock(self):
        if self.dock is None:
            self.dock = MustatilDock(self.iface, self.plugin_dir)
            self.iface.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.dock.show()
        self.dock.raise_()

    def install_runtime_direct(self):
        self.show_dock()
        self.dock.install_runtime()

    def open_training_direct(self):
        self.show_dock()
        try:
            self.dock.tabs.setCurrentWidget(self.dock.training_tab)
        except Exception:
            pass

    def open_annotator_direct(self):
        self.show_dock()
        try:
            self.dock.tabs.setCurrentWidget(self.dock.annotator_tab)
        except Exception:
            pass

    def open_toolkit_direct(self):
        self.show_dock()
        try:
            self.dock.tabs.setCurrentWidget(self.dock.toolkit_tab)
        except Exception:
            pass

        github_label = QLabel('<a href="https://github.com/tarekwasfy01/Mustatil-YOLO-AI-Model-Trainer-">Standalone Version on GitHub</a>')
        github_label.setOpenExternalLinks(True)
        github_label.setTextInteractionFlags(Qt.TextBrowserInteraction)
        github_label.setWordWrap(True)
        try:
            self.main_layout.addWidget(github_label)
        except Exception:
            try:
                layout.addWidget(github_label)
            except Exception:
                pass
