#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External YOLO trainer for Mustatil QGIS plugin.

Runs outside the QGIS Python runtime.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import random


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}


def _read_data_yaml_root(data_yaml: Path) -> Path:
    text = data_yaml.read_text(encoding="utf-8", errors="ignore")
    root = data_yaml.parent

    # Minimal YAML parse without requiring PyYAML.
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("path:"):
            val = s.split(":", 1)[1].strip().strip('"').strip("'")
            if val:
                root = Path(val)
                if not root.is_absolute():
                    root = (data_yaml.parent / root).resolve()
            break

    return root


def _image_files(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        p for p in folder.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def _label_for_image(img: Path, candidates):
    for base in candidates:
        p = base / (img.stem + ".txt")
        if p.exists():
            return p
    # Also allow label file next to image.
    p = img.with_suffix(".txt")
    if p.exists():
        return p
    return None


def _copy_image_and_label(img: Path, train_img: Path, train_lab: Path, label_candidates):
    dst_img = train_img / img.name
    if not dst_img.exists():
        shutil.copy2(img, dst_img)

    lab = _label_for_image(img, label_candidates)
    dst_lab = train_lab / (img.stem + ".txt")
    if lab and lab.exists():
        if not dst_lab.exists():
            shutil.copy2(lab, dst_lab)
    else:
        # Empty label is valid YOLO syntax and lets training start,
        # but users should normally have labels for useful training.
        if not dst_lab.exists():
            dst_lab.write_text("", encoding="utf-8")


def _autofill_yolo_dataset(root: Path, data_yaml: Path, train_img: Path, train_lab: Path):
    """
    If _yolo_dataset/train/images is empty, populate it from likely crop/export folders.

    Typical user path:
      C:/Users/.../Car/crops/_yolo_dataset/data.yaml

    Existing images are often in:
      C:/Users/.../Car/crops/*.jpg
      C:/Users/.../Car/crops/images/*.jpg
      C:/Users/.../Car/crops/train/images/*.jpg
      C:/Users/.../Car/images/*.jpg
    """
    search_roots = []
    for p in [
        root,
        data_yaml.parent,
        data_yaml.parent.parent,
        data_yaml.parent.parent.parent,
        data_yaml.parent.parent / "images",
        data_yaml.parent.parent / "crops",
        data_yaml.parent.parent / "_yolo_dataset",
        data_yaml.parent.parent / "_yolo_dataset" / "images",
    ]:
        if p and p.exists() and p not in search_roots:
            search_roots.append(p)

    image_candidates = []
    for base in search_roots:
        for sub in [
            base,
            base / "images",
            base / "crops",
            base / "train" / "images",
            base / "valid" / "images",
            base / "val" / "images",
        ]:
            if sub.exists():
                for img in _image_files(sub):
                    # Do not copy from the target folder into itself.
                    try:
                        if train_img in img.parents:
                            continue
                    except Exception:
                        pass
                    # Ignore validation target folder here; validation is split later.
                    try:
                        if (root / "val" / "images") in img.parents:
                            continue
                    except Exception:
                        pass
                    image_candidates.append(img)

    # Deduplicate by resolved path.
    unique = []
    seen = set()
    for img in image_candidates:
        try:
            key = str(img.resolve()).lower()
        except Exception:
            key = str(img).lower()
        if key not in seen:
            seen.add(key)
            unique.append(img)

    label_candidates = []
    for base in search_roots:
        for sub in [
            base / "labels",
            base / "train" / "labels",
            base / "valid" / "labels",
            base / "val" / "labels",
            base / "crops" / "labels",
            base,
        ]:
            if sub.exists() and sub not in label_candidates:
                label_candidates.append(sub)

    if unique:
        train_img.mkdir(parents=True, exist_ok=True)
        train_lab.mkdir(parents=True, exist_ok=True)
        for img in unique:
            _copy_image_and_label(img, train_img, train_lab, label_candidates)
        print("Auto-fill searched parent/crop/image folders.", flush=True)
        print(
            f"Auto-filled YOLO training folder from crop/image folders: {len(unique)} images",
            flush=True,
        )

    return len(unique)


def ensure_dataset_ready(data_yaml: Path):
    root = _read_data_yaml_root(data_yaml)

    train_img = root / "train" / "images"
    train_lab = root / "train" / "labels"
    val_img = root / "val" / "images"
    val_lab = root / "val" / "labels"

    for d in [train_img, train_lab, val_img, val_lab]:
        d.mkdir(parents=True, exist_ok=True)

    imgs = _image_files(train_img)

    if not imgs:
        _autofill_yolo_dataset(root, data_yaml, train_img, train_lab)
        imgs = _image_files(train_img)

    val_imgs = _image_files(val_img)

    if imgs and not val_imgs and len(imgs) > 1:
        random.seed(42)
        move_count = max(1, int(round(len(imgs) * 0.2)))
        selected = random.sample(imgs, move_count)
        for img in selected:
            dst = val_img / img.name
            if not dst.exists():
                shutil.copy2(img, dst)

            lab = train_lab / (img.stem + ".txt")
            if lab.exists():
                shutil.copy2(lab, val_lab / lab.name)
            else:
                (val_lab / (img.stem + ".txt")).write_text("", encoding="utf-8")

        print(f"Auto-created validation split: {move_count} images", flush=True)

    imgs = _image_files(train_img)
    val_imgs = _image_files(val_img)

    if not imgs:
        raise RuntimeError(
            "No YOLO training images found.\n\n"
            f"Expected images in:\n  {train_img}\n\n"
            "The trainer also searched the parent crop folders, but found no supported image files.\n"
            "Please export/create crops first, or place images in train/images.\n"
            "Supported image extensions: " + ", ".join(sorted(IMAGE_EXTS))
        )

    if not val_imgs:
        if len(imgs) == 1:
            # YOLO needs val data; duplicate the only image as validation fallback.
            img = imgs[0]
            shutil.copy2(img, val_img / img.name)
            lab = train_lab / (img.stem + ".txt")
            if lab.exists():
                shutil.copy2(lab, val_lab / lab.name)
            else:
                (val_lab / (img.stem + ".txt")).write_text("", encoding="utf-8")
            print("Only one training image found: duplicated it into val/images so training can start.", flush=True)
        else:
            raise RuntimeError(
                "No validation images found and automatic validation split failed.\n\n"
                f"Expected validation images in:\n  {val_img}"
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="YOLO data.yaml")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="")
    ap.add_argument("--name", default="qgis_train")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("PYTHONUTF8", "1")
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")

    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(f"data.yaml not found: {data}")

    ensure_dataset_ready(data)

    print("============================================================")
    print("Mustatil QGIS YOLO Trainer")
    print("============================================================")
    print(f"Data:   {data}")
    print(f"Model:  {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Image:  {args.imgsz}")
    print(f"Batch:  {args.batch}")
    print(f"Device: {args.device}")
    print("============================================================")

    model = YOLO(args.model)

    device = (args.device or "cpu").strip().lower()
    if device == "opencl":
        print("OpenCL selected: falling back to CPU because Torch training has no generic OpenCL backend.")
        device = "cpu"
    elif device == "directml":
        print("DirectML selected. Requires torch-directml runtime for acceleration.")

    kwargs = dict(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        name=args.name,
        verbose=True,
    )
    if args.project:
        kwargs["project"] = args.project
    if args.resume:
        kwargs["resume"] = True

    result = model.train(**kwargs)
    print("Training finished.")
    print(result)


if __name__ == "__main__":
    main()
