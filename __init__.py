#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple external YOLO box annotator for Mustatil QGIS plugin.

Controls:
- Open image folder
- Next / Previous image
- Left mouse drag: create rectangle
- Right click near/inside box: delete nearest box
- Class selector
- Save writes YOLO .txt labels
"""
from __future__ import annotations
import argparse
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

class Annotator(tk.Tk):
    def __init__(self, image_dir="", label_dir="", classes=None):
        super().__init__()
        self.title("Mustatil YOLO Annotator")
        self.geometry("1200x760")
        self.image_dir = Path(image_dir) if image_dir else None
        self.label_dir = Path(label_dir) if label_dir else None
        self.classes = classes or ["mustatil", "false_positive"]
        self.images = []
        self.index = 0
        self.pil = None
        self.tkimg = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.boxes = []
        self.drag_start = None
        self.preview_rect = None
        self._build()
        if self.image_dir:
            self.load_folder(self.image_dir)

    def _build(self):
        top = ttk.Frame(self, padding=6)
        top.pack(fill="x")
        ttk.Button(top, text="Open image folder", command=self.pick_folder).pack(side="left", padx=2)
        ttk.Button(top, text="Prev", command=self.prev_image).pack(side="left", padx=2)
        ttk.Button(top, text="Next", command=self.next_image).pack(side="left", padx=2)
        ttk.Button(top, text="Save", command=self.save_labels).pack(side="left", padx=2)

        self.cls = tk.IntVar(value=0)
        self.cls_combo = ttk.Combobox(top, state="readonly", values=[f"{i}: {c}" for i,c in enumerate(self.classes)], width=24)
        self.cls_combo.current(0)
        self.cls_combo.pack(side="left", padx=12)
        self.status = tk.StringVar(value="Ready")
        ttk.Label(top, textvariable=self.status).pack(side="left", padx=12)

        self.canvas = tk.Canvas(self, bg="#202020")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda e: self.redraw())
        self.canvas.bind("<ButtonPress-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)
        self.canvas.bind("<Button-3>", self.delete_box)

    def pick_folder(self):
        d = filedialog.askdirectory()
        if d:
            self.load_folder(Path(d))

    def load_folder(self, d):
        self.image_dir = Path(d)
        if not self.label_dir:
            self.label_dir = self.image_dir.parent / "labels"
        self.label_dir.mkdir(parents=True, exist_ok=True)
        self.images = [p for p in sorted(self.image_dir.rglob("*")) if p.suffix.lower() in IMG_EXT]
        self.index = 0
        self.load_image()

    def label_path(self):
        return self.label_dir / (self.images[self.index].stem + ".txt")

    def load_image(self):
        if not self.images:
            self.status.set("No images found")
            return
        self.pil = Image.open(self.images[self.index]).convert("RGB")
        self.load_labels()
        self.redraw()
        self.status.set(f"{self.index+1}/{len(self.images)} {self.images[self.index].name}")

    def load_labels(self):
        self.boxes = []
        lp = self.label_path()
        if not lp.exists() or self.pil is None:
            return
        W, H = self.pil.size
        for line in lp.read_text(encoding="utf-8", errors="ignore").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            cls = int(float(p[0])); cx, cy, bw, bh = map(float, p[1:])
            x1 = (cx - bw/2) * W; y1 = (cy - bh/2) * H
            x2 = (cx + bw/2) * W; y2 = (cy + bh/2) * H
            self.boxes.append([cls, x1, y1, x2, y2])

    def save_labels(self):
        if not self.images or self.pil is None:
            return
        W, H = self.pil.size
        lines = []
        for cls, x1, y1, x2, y2 in self.boxes:
            x1, x2 = sorted([max(0, min(W, x1)), max(0, min(W, x2))])
            y1, y2 = sorted([max(0, min(H, y1)), max(0, min(H, y2))])
            if x2-x1 < 2 or y2-y1 < 2:
                continue
            cx = ((x1+x2)/2)/W; cy = ((y1+y2)/2)/H
            bw = (x2-x1)/W; bh = (y2-y1)/H
            lines.append(f"{int(cls)} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}")
        self.label_path().write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.status.set(f"Saved: {self.label_path()}")

    def prev_image(self):
        self.save_labels()
        if self.images:
            self.index = max(0, self.index-1)
            self.load_image()

    def next_image(self):
        self.save_labels()
        if self.images:
            self.index = min(len(self.images)-1, self.index+1)
            self.load_image()

    def redraw(self):
        self.canvas.delete("all")
        if self.pil is None:
            return
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        W, H = self.pil.size
        self.scale = min(cw/W, ch/H)
        nw, nh = int(W*self.scale), int(H*self.scale)
        self.offset = ((cw-nw)//2, (ch-nh)//2)
        im = self.pil.resize((nw, nh), Image.LANCZOS)
        self.tkimg = ImageTk.PhotoImage(im)
        ox, oy = self.offset
        self.canvas.create_image(ox, oy, image=self.tkimg, anchor="nw")
        for box in self.boxes:
            self.draw_box(box)

    def to_img(self, sx, sy):
        ox, oy = self.offset
        return (sx-ox)/self.scale, (sy-oy)/self.scale

    def to_screen(self, x, y):
        ox, oy = self.offset
        return ox + x*self.scale, oy + y*self.scale

    def draw_box(self, box):
        cls, x1, y1, x2, y2 = box
        sx1, sy1 = self.to_screen(x1, y1)
        sx2, sy2 = self.to_screen(x2, y2)
        self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="red", width=2)
        name = self.classes[cls] if 0 <= cls < len(self.classes) else str(cls)
        self.canvas.create_text(sx1+3, sy1+3, text=name, fill="yellow", anchor="nw")

    def mouse_down(self, e):
        if self.pil is None:
            return
        self.drag_start = self.to_img(e.x, e.y)

    def mouse_drag(self, e):
        if not self.drag_start:
            return
        if self.preview_rect:
            self.canvas.delete(self.preview_rect)
        x1, y1 = self.drag_start
        x2, y2 = self.to_img(e.x, e.y)
        sx1, sy1 = self.to_screen(x1, y1)
        sx2, sy2 = self.to_screen(x2, y2)
        self.preview_rect = self.canvas.create_rectangle(sx1, sy1, sx2, sy2, outline="cyan", width=2)

    def mouse_up(self, e):
        if self.pil is None or not self.drag_start:
            return
        x1, y1 = self.drag_start
        x2, y2 = self.to_img(e.x, e.y)
        self.drag_start = None
        if self.preview_rect:
            self.canvas.delete(self.preview_rect)
            self.preview_rect = None
        if abs(x2-x1) < 3 or abs(y2-y1) < 3:
            return
        cls = self.cls_combo.current()
        self.boxes.append([cls, x1, y1, x2, y2])
        self.redraw()

    def delete_box(self, e):
        if self.pil is None or not self.boxes:
            return
        x, y = self.to_img(e.x, e.y)
        best_i, best_d = None, 1e18
        for i, (_, x1, y1, x2, y2) in enumerate(self.boxes):
            cx, cy = (x1+x2)/2, (y1+y2)/2
            inside = min(x1,x2) <= x <= max(x1,x2) and min(y1,y2) <= y <= max(y1,y2)
            d = 0 if inside else (cx-x)**2 + (cy-y)**2
            if d < best_d:
                best_i, best_d = i, d
        if best_i is not None:
            del self.boxes[best_i]
            self.redraw()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--classes", default="mustatil,false_positive")
    args = ap.parse_args()
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    Annotator(args.images, args.labels, classes).mainloop()

if __name__ == "__main__":
    main()