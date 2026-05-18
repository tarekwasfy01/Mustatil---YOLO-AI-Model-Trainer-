#!/usr/bin/env python3
"""
Mustatil Trainer & Detector
One-file GUI for annotating satellite/map imagery, training a YOLO detector,
and running tiled inference on very large maps. Exports detections as GeoJSON
when rasterio can read georeferencing; otherwise exports pixel-coordinate GeoJSON.

Install CPU-safe dependencies:
    python -m pip install --upgrade pip
    python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    python -m pip install ultralytics pillow opencv-python numpy pyyaml tqdm pandas matplotlib

Optional GeoTIFF georeferencing:
    python -m pip install rasterio shapely

Run:
    python mustatil_trainer_detector.py
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, simpledialog, ttk
except Exception as exc:
    raise SystemExit(f"Tkinter is required: {exc}")

try:
    from PIL import Image, ImageTk
except Exception:
    raise SystemExit("Missing Pillow. Install with: python -m pip install pillow")

try:
    import yaml
except Exception:
    yaml = None

APP_TITLE = "Mustatil Trainer & Detector"
SUPPORTED_IMAGES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def resolve_model_path(model_name: str, project_root: Path, log_fn=None) -> Path:
    """
    Keeps YOLO weights inside the project folder so Ultralytics never tries to
    download into a protected/current working directory.
    """
    model_name = (model_name or "yolov8n.pt").strip()
    p = Path(model_name)

    if p.exists():
        return p.resolve()

    weights_dir = project_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # If user entered only yolov8n.pt, store it in project/weights/yolov8n.pt
    target = weights_dir / p.name

    if target.exists() and target.stat().st_size > 100_000:
        if log_fn:
            log_fn(f"Using existing local model: {target}")
        return target

    known = {
        "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
        "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt",
        "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8m.pt",
        "yolov5nu.pt": "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov5nu.pt",
    }

    if p.name not in known:
        # Unknown model name: return project-local path. Ultralytics may still download if supported.
        return target

    url = known[p.name]
    if log_fn:
        log_fn(f"Downloading base model to writable project folder:")
        log_fn(f"{target}")
        log_fn(url)

    tmp = target.with_suffix(".tmp")
    urllib.request.urlretrieve(url, tmp)
    if tmp.stat().st_size < 100_000:
        raise RuntimeError(f"Downloaded model is too small/invalid: {tmp}")
    tmp.replace(target)
    return target.resolve()


def have_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def dependency_report() -> Tuple[bool, str]:
    """
    Checks runtime dependencies without crashing the GUI.
    """
    missing = []
    for mod in ["torch", "ultralytics", "yaml", "PIL", "cv2", "numpy"]:
        if not have_module(mod):
            missing.append(mod)

    py = sys.executable
    if missing:
        msg = (
            "Missing dependencies: " + ", ".join(missing) + "\n\n"
            "Install with the SAME Python used to start this GUI:\n\n"
            f'"{py}" -m pip install --upgrade pip\n'
            f'"{py}" -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu\n'
            f'"{py}" -m pip install ultralytics pillow opencv-python numpy pyyaml tqdm pandas matplotlib\n\n'
            "Then restart this app."
        )
        return False, msg

    try:
        import torch
        import ultralytics
        msg = (
            f"Python: {sys.executable}\n"
            f"torch: {getattr(torch, '__version__', 'unknown')}\n"
            f"ultralytics: {getattr(ultralytics, '__file__', 'unknown')}\n"
            f"CUDA available: {torch.cuda.is_available()}\n\n"
            "AMD R9 390X is not supported by PyTorch CUDA. Use CPU mode."
        )
        return True, msg
    except Exception as exc:
        return False, f"Dependency import failed:\n{exc}"


def show_dependency_help(parent=None):
    ok, msg = dependency_report()
    if parent is not None:
        messagebox.showinfo("Dependency Check" if ok else "Missing Dependencies", msg)
    return ok, msg



@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float

    def normalized_yolo(self, w: int, h: int) -> Tuple[int, float, float, float, float]:
        x1, x2 = sorted((max(0, self.x1), min(w - 1, self.x2)))
        y1, y2 = sorted((max(0, self.y1), min(h - 1, self.y2)))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        cx = x1 + bw / 2
        cy = y1 + bh / 2
        return self.cls, cx / w, cy / h, bw / w, bh / h

    @staticmethod
    def from_yolo(cls: int, cx: float, cy: float, bw: float, bh: float, w: int, h: int) -> "Box":
        x1 = (cx - bw / 2) * w
        y1 = (cy - bh / 2) * h
        x2 = (cx + bw / 2) * w
        y2 = (cy + bh / 2) * h
        return Box(cls, x1, y1, x2, y2)


class Project:
    def __init__(self, root: Path):
        self.root = root
        self.images_dir = root / "images"
        self.labels_dir = root / "labels"
        self.runs_dir = root / "runs"
        self.exports_dir = root / "exports"
        self.config_path = root / "project.json"
        self.classes = ["mustatil"]
        for d in [self.images_dir, self.labels_dir, self.runs_dir, self.exports_dir]:
            d.mkdir(parents=True, exist_ok=True)
        self.load()

    def load(self):
        if self.config_path.exists():
            data = json.loads(self.config_path.read_text(encoding="utf-8"))
            self.classes = data.get("classes", ["mustatil"])
        else:
            self.save()

    def save(self):
        self.config_path.write_text(json.dumps({"classes": self.classes}, indent=2), encoding="utf-8")

    def image_files(self) -> List[Path]:
        return sorted([p for p in self.images_dir.iterdir() if p.suffix.lower() in SUPPORTED_IMAGES])

    def label_path(self, image_path: Path) -> Path:
        return self.labels_dir / f"{image_path.stem}.txt"

    def load_boxes(self, image_path: Path) -> List[Box]:
        lp = self.label_path(image_path)
        if not lp.exists():
            return []
        with Image.open(image_path) as im:
            w, h = im.size
        boxes: List[Box] = []
        for line in lp.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls = int(float(parts[0])); cx, cy, bw, bh = map(float, parts[1:])
            boxes.append(Box.from_yolo(cls, cx, cy, bw, bh, w, h))
        return boxes

    def save_boxes(self, image_path: Path, boxes: List[Box]):
        with Image.open(image_path) as im:
            w, h = im.size
        lines = []
        for b in boxes:
            cls, cx, cy, bw, bh = b.normalized_yolo(w, h)
            if bw > 0 and bh > 0:
                lines.append(f"{cls} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        self.label_path(image_path).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def run_cmd_live(cmd: List[str], log_fn, cwd: Optional[Path] = None):
    """
    Runs a subprocess and streams output into the GUI.
    Windows/Anaconda often emits UTF-8 progress characters that crash cp1252
    decoding with: 'charmap' codec can't decode byte ...
    This reader forces UTF-8 and replaces undecodable bytes.
    """
    log_fn("$ " + " ".join(map(str, cmd)))

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"

    p = subprocess.Popen(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
        env=env,
    )
    assert p.stdout is not None
    for line in p.stdout:
        # Remove common terminal control sequences from tqdm/Ultralytics output.
        clean = line.replace("\x1b[K", "").rstrip()
        log_fn(clean)
    code = p.wait()
    if code != 0:
        raise RuntimeError(f"Command failed with exit code {code}")


class MustatilGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1280x820")
        self.project: Optional[Project] = None
        self.current_image: Optional[Path] = None
        self.boxes: List[Box] = []
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.drag_start: Optional[Tuple[int, int]] = None
        self.temp_rect = None
        self.photo = None
        self.det_preview_img = None
        self.det_preview_photo = None
        self.det_preview_features = []
        self._build_ui()
        self.after(500, self._startup_dependency_log)

    def _build_ui(self):
        nb = ttk.Notebook(self)
        nb.pack(fill=tk.BOTH, expand=True)
        self.annotate_tab = ttk.Frame(nb)
        self.train_tab = ttk.Frame(nb)
        self.detect_tab = ttk.Frame(nb)
        nb.add(self.annotate_tab, text="1 Annotate")
        nb.add(self.train_tab, text="2 Train / Export")
        nb.add(self.detect_tab, text="3 Detect Maps")
        self._build_annotate_tab()
        self._build_train_tab()
        self._build_detect_tab()

    def _build_annotate_tab(self):
        left = ttk.Frame(self.annotate_tab, width=280)
        left.pack(side=tk.LEFT, fill=tk.Y)
        right = ttk.Frame(self.annotate_tab)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        ttk.Button(left, text="New / Open Project", command=self.open_project).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(left, text="Import Images", command=self.import_images).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(left, text="Save Labels", command=self.save_current).pack(fill=tk.X, padx=8, pady=4)
        ttk.Button(left, text="Delete Selected Box", command=self.delete_selected_box).pack(fill=tk.X, padx=8, pady=4)
        ttk.Label(left, text="Images").pack(anchor="w", padx=8, pady=(12, 0))
        self.img_list = tk.Listbox(left)
        self.img_list.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
        self.img_list.bind("<<ListboxSelect>>", lambda e: self.select_image())
        ttk.Label(left, text="Boxes").pack(anchor="w", padx=8, pady=(12, 0))
        self.box_list = tk.Listbox(left, height=8)
        self.box_list.pack(fill=tk.X, padx=8, pady=4)

        self.canvas = tk.Canvas(right, bg="#222222", cursor="crosshair")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        ttk.Label(right, text="Draw rectangles around mustatils. Labels are saved in YOLO format.").pack(anchor="w")

    def _build_train_tab(self):
        frm = ttk.Frame(self.train_tab)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.epochs = tk.IntVar(value=80)
        self.imgsz = tk.IntVar(value=1024)
        self.batch = tk.IntVar(value=4)
        self.device = tk.StringVar(value="cpu")
        self.model_name = tk.StringVar(value="yolov8n.pt")
        row = 0
        for label, var in [("Base model", self.model_name), ("Epochs", self.epochs), ("Image size", self.imgsz), ("Batch", self.batch), ("Device", self.device)]:
            ttk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=4)
            ttk.Entry(frm, textvariable=var).grid(row=row, column=1, sticky="ew", pady=4)
            row += 1
        frm.columnconfigure(1, weight=1)
        ttk.Button(frm, text="Dependency Check", command=lambda: show_dependency_help(self)).grid(row=row, column=0, sticky="ew", pady=8)
        ttk.Button(frm, text="Prepare YOLO Dataset", command=self.prepare_dataset).grid(row=row, column=1, sticky="ew", pady=8)
        row += 1
        ttk.Button(frm, text="Train YOLO Model", command=self.train_model_thread).grid(row=row, column=0, columnspan=2, sticky="ew", pady=8)
        row += 1
        ttk.Button(frm, text="Export Best Model to ONNX", command=self.export_onnx_thread).grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        self.train_log = tk.Text(frm, height=28)
        self.train_log.grid(row=row, column=0, columnspan=2, sticky="nsew")
        frm.rowconfigure(row, weight=1)

    def _build_detect_tab(self):
        frm = ttk.Frame(self.detect_tab)
        frm.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.det_model = tk.StringVar()
        self.det_image = tk.StringVar()
        self.conf = tk.DoubleVar(value=0.25)
        self.tile = tk.IntVar(value=1024)
        self.overlap = tk.IntVar(value=128)
        self.out_geojson = tk.StringVar()

        top = ttk.LabelFrame(frm, text="Detection Settings", padding=8)
        top.pack(fill=tk.X)

        rows = [
            ("Model .pt/.onnx", self.det_model, self.pick_model),
            ("Map image/GeoTIFF", self.det_image, self.pick_detect_image),
            ("Output GeoJSON", self.out_geojson, self.pick_output_geojson),
        ]
        for r, (label, var, cmd) in enumerate(rows):
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Entry(top, textvariable=var).grid(row=r, column=1, sticky="ew", pady=4)
            ttk.Button(top, text="Browse", command=cmd).grid(row=r, column=2, sticky="ew", pady=4)

        r = 3
        for label, var in [("Confidence", self.conf), ("Tile size", self.tile), ("Overlap", self.overlap)]:
            ttk.Label(top, text=label).grid(row=r, column=0, sticky="w", pady=4)
            ttk.Entry(top, textvariable=var).grid(row=r, column=1, sticky="ew", pady=4)
            r += 1

        btnrow = ttk.Frame(top)
        btnrow.grid(row=r, column=0, columnspan=3, sticky="ew", pady=8)
        ttk.Button(btnrow, text="Run Tiled Detection", command=self.detect_thread).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btnrow, text="Show Image + GeoJSON", command=self.load_detection_preview).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        ttk.Button(btnrow, text="Clear Preview", command=self.clear_detection_preview).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        top.columnconfigure(1, weight=1)

        middle = ttk.PanedWindow(frm, orient=tk.HORIZONTAL)
        middle.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

        preview_frame = ttk.LabelFrame(middle, text="Image + Detection GeoJSON Preview", padding=4)
        log_frame = ttk.LabelFrame(middle, text="Detection Log", padding=4)
        middle.add(preview_frame, weight=3)
        middle.add(log_frame, weight=2)

        self.det_canvas = tk.Canvas(preview_frame, bg="#222222", highlightthickness=1, highlightbackground="#666")
        self.det_canvas.pack(fill=tk.BOTH, expand=True)
        self.det_canvas.bind("<Configure>", lambda e: self.redraw_detection_preview())

        self.detect_log = tk.Text(log_frame, height=30)
        self.detect_log.pack(fill=tk.BOTH, expand=True)

        ttk.Label(
            preview_frame,
            text="Preview shows pixel-space overlays. Geo-referenced GeoJSON is still exported for QGIS when the input image has CRS/geotransform.",
            foreground="#555",
        ).pack(anchor="w")

    def log_train(self, s): self.train_log.insert(tk.END, s + "\n"); self.train_log.see(tk.END); self.update_idletasks()
    def log_detect(self, s): self.detect_log.insert(tk.END, s + "\n"); self.detect_log.see(tk.END); self.update_idletasks()

    def _startup_dependency_log(self):
        ok, msg = dependency_report()
        try:
            self.log_train(msg)
        except Exception:
            pass

    def open_project(self):
        path = filedialog.askdirectory(title="Choose/create project folder")
        if not path: return
        self.project = Project(Path(path))
        self.refresh_images()

    def import_images(self):
        if not self.project:
            self.open_project()
            if not self.project: return
        files = filedialog.askopenfilenames(title="Import images", filetypes=[("Images", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.webp")])
        for f in files:
            src = Path(f)
            dst = self.project.images_dir / src.name
            if src.resolve() != dst.resolve():
                shutil.copy2(src, dst)
        self.refresh_images()

    def refresh_images(self):
        self.img_list.delete(0, tk.END)
        if not self.project: return
        for p in self.project.image_files():
            self.img_list.insert(tk.END, p.name)

    def select_image(self):
        if not self.project: return
        sel = self.img_list.curselection()
        if not sel: return
        if self.current_image:
            self.save_current(silent=True)
        self.current_image = self.project.images_dir / self.img_list.get(sel[0])
        self.boxes = self.project.load_boxes(self.current_image)
        self.redraw()

    def save_current(self, silent=False):
        if self.project and self.current_image:
            self.project.save_boxes(self.current_image, self.boxes)
            if not silent: messagebox.showinfo(APP_TITLE, "Labels saved.")

    def _image_to_canvas(self, x, y): return x * self.scale + self.offset_x, y * self.scale + self.offset_y
    def _canvas_to_image(self, x, y): return (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale

    def redraw(self):
        self.canvas.delete("all")
        self.box_list.delete(0, tk.END)
        if not self.current_image: return
        im = Image.open(self.current_image).convert("RGB")
        cw = max(1, self.canvas.winfo_width()); ch = max(1, self.canvas.winfo_height())
        self.scale = min(cw / im.width, ch / im.height, 1.0)
        show = im.resize((int(im.width * self.scale), int(im.height * self.scale)))
        self.offset_x = (cw - show.width) // 2; self.offset_y = (ch - show.height) // 2
        self.photo = ImageTk.PhotoImage(show)
        self.canvas.create_image(self.offset_x, self.offset_y, anchor="nw", image=self.photo)
        for i, b in enumerate(self.boxes):
            x1, y1 = self._image_to_canvas(b.x1, b.y1); x2, y2 = self._image_to_canvas(b.x2, b.y2)
            self.canvas.create_rectangle(x1, y1, x2, y2, outline="yellow", width=2)
            self.canvas.create_text(x1 + 4, y1 + 4, anchor="nw", text=str(i), fill="yellow")
            self.box_list.insert(tk.END, f"{i}: mustatil [{int(b.x1)},{int(b.y1)}]-[{int(b.x2)},{int(b.y2)}]")

    def on_mouse_down(self, e): self.drag_start = (e.x, e.y)
    def on_mouse_drag(self, e):
        if not self.drag_start: return
        if self.temp_rect: self.canvas.delete(self.temp_rect)
        x0, y0 = self.drag_start
        self.temp_rect = self.canvas.create_rectangle(x0, y0, e.x, e.y, outline="red", width=2)
    def on_mouse_up(self, e):
        if not self.drag_start or not self.current_image: return
        x0, y0 = self.drag_start; x1, y1 = e.x, e.y
        ix0, iy0 = self._canvas_to_image(x0, y0); ix1, iy1 = self._canvas_to_image(x1, y1)
        if abs(ix1 - ix0) > 8 and abs(iy1 - iy0) > 8:
            self.boxes.append(Box(0, ix0, iy0, ix1, iy1))
            self.save_current(silent=True)
        self.drag_start = None; self.temp_rect = None; self.redraw()

    def delete_selected_box(self):
        sel = self.box_list.curselection()
        if sel:
            del self.boxes[sel[0]]
            self.save_current(silent=True); self.redraw()

    def prepare_dataset(self):
        if not self.project: return messagebox.showerror(APP_TITLE, "Open a project first.")
        if yaml is None: return messagebox.showerror(APP_TITLE, "Install PyYAML: python -m pip install pyyaml")
        ds = self.project.root / "yolo_dataset"
        for split in ["train", "val"]:
            (ds / "images" / split).mkdir(parents=True, exist_ok=True)
            (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
        imgs = self.project.image_files()
        random.seed(42); random.shuffle(imgs)
        cut = max(1, int(len(imgs) * 0.8))
        for idx, img in enumerate(imgs):
            split = "train" if idx < cut else "val"
            shutil.copy2(img, ds / "images" / split / img.name)
            lp = self.project.label_path(img)
            outlp = ds / "labels" / split / f"{img.stem}.txt"
            if lp.exists(): shutil.copy2(lp, outlp)
            else: outlp.write_text("", encoding="utf-8")
        data = {"path": str(ds), "train": "images/train", "val": "images/val", "names": {0: "mustatil"}}
        (ds / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        self.log_train(f"Dataset ready: {ds}")
        return ds / "data.yaml"

    def train_model_thread(self): threading.Thread(target=self.train_model, daemon=True).start()
    def train_model(self):
        try:
            if not self.project:
                raise RuntimeError("Open a project first.")

            ok, msg = dependency_report()
            self.log_train(msg)
            if not ok:
                messagebox.showerror(APP_TITLE, msg)
                return

            data_yaml = self.prepare_dataset()
            device = self.device.get().strip() or "cpu"
            if device.lower() != "cpu":
                self.log_train("WARNING: Non-CPU device selected. AMD R9 390X is not supported by PyTorch CUDA.")

            local_model = resolve_model_path(self.model_name.get(), self.project.root, self.log_train)
            self.log_train(f"Model path: {local_model}")
            train_code = (
                "from ultralytics import YOLO\n"
                "import torch\n"
                f"model = YOLO(r'{local_model}')\n"
                "model.train("
                f"data=r'{data_yaml}', "
                f"epochs={int(self.epochs.get())}, "
                f"imgsz={int(self.imgsz.get())}, "
                f"batch={int(self.batch.get())}, "
                f"device=r'{device}', "
                f"project=r'{self.project.runs_dir}', "
                "name='mustatil', "
                "exist_ok=True, "
                "workers=0, "
                "cache=False, "
                "patience=20, "
                "save=True, "
                "plots=True, "
                "verbose=True)\n"
            )
            cmd = [sys.executable, "-c", train_code]
            run_cmd_live(cmd, self.log_train, cwd=self.project.root)
            self.log_train("Training complete. Best model is usually runs/mustatil/weights/best.pt")
        except Exception as exc:
            self.log_train(f"ERROR: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))

    def export_onnx_thread(self): threading.Thread(target=self.export_onnx, daemon=True).start()
    def export_onnx(self):
        try:
            if not self.project: raise RuntimeError("Open a project first.")
            best = self.project.runs_dir / "mustatil" / "weights" / "best.pt"
            if not best.exists():
                best = Path(filedialog.askopenfilename(title="Choose best.pt", filetypes=[("PyTorch model", "*.pt")]))
            export_code = (
                "from ultralytics import YOLO\n"
                f"model = YOLO(r'{best}')\n"
                f"model.export(format='onnx', imgsz={int(self.imgsz.get())})\n"
            )
            cmd = [sys.executable, "-c", export_code]
            run_cmd_live(cmd, self.log_train)
            self.log_train("ONNX export complete. Use .pt for this tool; use .onnx where QGIS/GeoAI supports ONNX.")
        except Exception as exc:
            self.log_train(f"ERROR: {exc}")

    def pick_model(self): self.det_model.set(filedialog.askopenfilename(filetypes=[("Models", "*.pt *.onnx")]))
    def pick_detect_image(self):
        p = filedialog.askopenfilename(filetypes=[("Images", "*.tif *.tiff *.jpg *.jpeg *.png *.bmp *.webp")])
        if p:
            self.det_image.set(p)
            self.load_detection_preview(image_only=True)

    def pick_output_geojson(self):
        p = filedialog.asksaveasfilename(defaultextension=".geojson", filetypes=[("GeoJSON", "*.geojson")])
        if p:
            self.out_geojson.set(p)


    def clear_detection_preview(self):
        self.det_preview_img = None
        self.det_preview_photo = None
        self.det_preview_features = []
        if hasattr(self, "det_canvas"):
            self.det_canvas.delete("all")

    def load_detection_preview(self, image_only=False):
        """
        Loads the map image and, if available, the output GeoJSON.
        For display we use pixel_bbox when present. This keeps the preview aligned
        even when the GeoJSON geometry itself is in EPSG:3857 or another CRS.
        """
        img_path = Path(self.det_image.get()) if self.det_image.get() else None
        if not img_path or not img_path.exists():
            messagebox.showerror(APP_TITLE, "Choose a map image first.")
            return

        try:
            self.det_preview_img = Image.open(img_path).convert("RGB")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open image:\\n{exc}")
            return

        self.det_preview_features = []

        if not image_only:
            gj_path = Path(self.out_geojson.get()) if self.out_geojson.get() else img_path.with_suffix(".detections.geojson")
            if gj_path.exists():
                try:
                    data = json.loads(gj_path.read_text(encoding="utf-8"))
                    self.det_preview_features = data.get("features", [])
                    self.log_detect(f"Loaded preview GeoJSON: {gj_path} ({len(self.det_preview_features)} features)")
                except Exception as exc:
                    self.log_detect(f"Could not load GeoJSON preview: {exc}")
            else:
                self.log_detect(f"No GeoJSON preview found yet: {gj_path}")

        self.redraw_detection_preview()

    def redraw_detection_preview(self):
        if not hasattr(self, "det_canvas"):
            return
        self.det_canvas.delete("all")
        if self.det_preview_img is None:
            self.det_canvas.create_text(12, 12, anchor="nw", fill="white", text="No preview loaded.")
            return

        cw = max(1, self.det_canvas.winfo_width())
        ch = max(1, self.det_canvas.winfo_height())
        im = self.det_preview_img
        scale = min(cw / im.width, ch / im.height, 1.0)
        sw, sh = max(1, int(im.width * scale)), max(1, int(im.height * scale))
        show = im.resize((sw, sh))
        ox = (cw - sw) // 2
        oy = (ch - sh) // 2

        self.det_preview_photo = ImageTk.PhotoImage(show)
        self.det_canvas.create_image(ox, oy, anchor="nw", image=self.det_preview_photo)

        count = 0
        for feat in self.det_preview_features:
            props = feat.get("properties", {})
            bbox = props.get("pixel_bbox")
            conf = props.get("confidence", None)

            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                sx1, sy1 = ox + x1 * scale, oy + y1 * scale
                sx2, sy2 = ox + x2 * scale, oy + y2 * scale
                self.det_canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="red", width=2)
                label = f"{conf:.2f}" if isinstance(conf, (int, float)) else ""
                if label:
                    self.det_canvas.create_text(sx1 + 3, sy1 + 3, anchor="nw", text=label, fill="red")
                count += 1
            else:
                # Fallback for non-georeferenced pixel-coordinate GeoJSON polygons
                geom = feat.get("geometry", {})
                coords = geom.get("coordinates", [])
                if geom.get("type") == "Polygon" and coords:
                    ring = coords[0]
                    pts = []
                    for p in ring:
                        if len(p) >= 2:
                            pts.extend([ox + float(p[0]) * scale, oy + float(p[1]) * scale])
                    if len(pts) >= 6:
                        self.det_canvas.create_line(*pts, fill="red", width=2)
                        count += 1

        self.det_canvas.create_text(
            10, 10,
            anchor="nw",
            fill="white",
            text=f"Image: {im.width}x{im.height} | Detections shown: {count}",
        )


    def detect_thread(self): threading.Thread(target=self.detect, daemon=True).start()
    def detect(self):
        ok, msg = dependency_report()
        self.log_detect(msg)
        if not ok:
            messagebox.showerror(APP_TITLE, msg)
            return
        try:
            from ultralytics import YOLO
            import numpy as np
            import cv2
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Install dependencies first.\n\n{exc}")
            return
        model_path = Path(self.det_model.get()); img_path = Path(self.det_image.get())
        out_path = Path(self.out_geojson.get() or (img_path.with_suffix(".detections.geojson")))
        if not model_path.exists() or not img_path.exists():
            return messagebox.showerror(APP_TITLE, "Choose model and image.")
        model = YOLO(str(model_path))
        tile = int(self.tile.get()); overlap = int(self.overlap.get()); stride = max(1, tile - overlap)
        conf = float(self.conf.get())
        im = Image.open(img_path).convert("RGB")
        W, H = im.size
        self.log_detect(f"Image size: {W} x {H}; tile={tile}; overlap={overlap}")
        transform = None; crs_name = None
        try:
            import rasterio
            with rasterio.open(img_path) as src:
                transform = src.transform
                crs_name = src.crs.to_string() if src.crs else None
        except Exception:
            transform = None
        features = []
        total = math.ceil(W / stride) * math.ceil(H / stride)
        n = 0
        for y in range(0, H, stride):
            for x in range(0, W, stride):
                n += 1
                crop = im.crop((x, y, min(W, x + tile), min(H, y + tile)))
                arr = np.array(crop)
                results = model.predict(arr, conf=conf, imgsz=tile, verbose=False)
                for r in results:
                    if r.boxes is None: continue
                    for b in r.boxes:
                        xyxy = b.xyxy.cpu().numpy()[0].tolist()
                        score = float(b.conf.cpu().numpy()[0])
                        cls = int(b.cls.cpu().numpy()[0])
                        gx1, gy1, gx2, gy2 = xyxy[0] + x, xyxy[1] + y, xyxy[2] + x, xyxy[3] + y
                        if transform is not None:
                            # rasterio affine: x_geo = a*col + b*row + c; y_geo = d*col + e*row + f
                            pts_px = [(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2), (gx1, gy1)]
                            pts = [(transform * (px, py)) for px, py in pts_px]
                        else:
                            pts = [(gx1, gy1), (gx2, gy1), (gx2, gy2), (gx1, gy2), (gx1, gy1)]
                        features.append({
                            "type": "Feature",
                            "properties": {"class": "mustatil", "class_id": cls, "confidence": score, "pixel_bbox": [gx1, gy1, gx2, gy2]},
                            "geometry": {"type": "Polygon", "coordinates": [[list(p) for p in pts]]}
                        })
                if n % 25 == 0:
                    self.log_detect(f"Processed {n}/{total} tiles; detections={len(features)}")
                if x + tile >= W and y + tile >= H: pass
            if y + tile >= H: break
        fc = {"type": "FeatureCollection", "name": "mustatil_detections", "crs": {"type":"name", "properties":{"name": crs_name}} if crs_name else None, "features": features}
        if fc["crs"] is None: del fc["crs"]
        out_path.write_text(json.dumps(fc, indent=2), encoding="utf-8")
        self.log_detect(f"Done. Wrote {len(features)} detections: {out_path}")
        self.out_geojson.set(str(out_path))
        self.load_detection_preview()
        messagebox.showinfo(APP_TITLE, f"Detection complete.\n{len(features)} detections\n{out_path}")


if __name__ == "__main__":
    app = MustatilGUI()
    app.mainloop()
