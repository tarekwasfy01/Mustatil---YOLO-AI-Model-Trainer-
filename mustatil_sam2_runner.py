#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
DEFAULT_CLASSES = ["mustatil", "false_positive"]


def yolo_root(project: Path) -> Path:
    return project / "yolo_datasets"


def ensure_yolo_dirs(project: Path):
    root = yolo_root(project)
    dirs = [
        root / "train" / "images",
        root / "train" / "labels",
        root / "val" / "images",
        root / "val" / "labels",
        project / "images",
        project / "labels",
        project / "crops",
        project / "exports",
        project / "weights",
        project / "runs",
        project / "sam2",
        project / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return root


def write_data_yaml(project: Path, classes=None):
    classes = classes or DEFAULT_CLASSES
    root = ensure_yolo_dirs(project)
    names = "\n".join([f"  {i}: {name}" for i, name in enumerate(classes)])
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        "path: " + str(root).replace("\\", "/") + "\n"
        "train: train/images\n"
        "val: val/images\n"
        "names:\n" + names + "\n",
        encoding="utf-8"
    )
    print(f"data.yaml written: {data_yaml}", flush=True)
    return data_yaml


def _is_image(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in IMG_EXT


def _dedupe_paths(paths):
    out = []
    seen = set()
    for p in paths:
        try:
            key = str(p.resolve()).lower()
        except Exception:
            key = str(p).lower()
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _candidate_image_dirs(project: Path, images_dir=None):
    candidates = []

    def add(p):
        if p:
            p = Path(p)
            candidates.append(p)
            # Common structures below and around the selected folder.
            candidates.extend([
                p / "images",
                p / "image",
                p / "imgs",
                p / "crops",
                p / "_yolo_dataset" / "images",
                p / "_yolo_dataset" / "train" / "images",
                p / "_yolo_dataset" / "val" / "images",
                p / "yolo_datasets" / "train" / "images",
                p / "yolo_datasets" / "val" / "images",
                p / "train" / "images",
                p / "valid" / "images",
                p / "val" / "images",
            ])
            if p.name.lower() in {"images", "image", "imgs", "crops"}:
                candidates.extend([
                    p.parent,
                    p.parent / "images",
                    p.parent / "crops",
                    p.parent / "_yolo_dataset" / "images",
                    p.parent / "_yolo_dataset" / "train" / "images",
                    p.parent / "_yolo_dataset" / "val" / "images",
                ])

    add(images_dir)
    add(project)
    candidates.extend([
        project / "images",
        project / "image",
        project / "imgs",
        project / "crops",
        project / "exports",
        project / "Mustatils" / "images",
        project / "Mustatils" / "image",
        yolo_root(project) / "train" / "images",
        yolo_root(project) / "val" / "images",
    ])
    return _dedupe_paths(candidates)


def _candidate_label_dirs(project: Path, labels_dir=None, images_dir=None):
    candidates = []

    def add(p):
        if p:
            p = Path(p)
            candidates.append(p)
            candidates.extend([
                p / "labels",
                p / "label",
                p / "_yolo_dataset" / "labels",
                p / "_yolo_dataset" / "train" / "labels",
                p / "_yolo_dataset" / "val" / "labels",
                p / "yolo_datasets" / "train" / "labels",
                p / "yolo_datasets" / "val" / "labels",
                p / "train" / "labels",
                p / "valid" / "labels",
                p / "val" / "labels",
            ])
            if p.name.lower() in {"labels", "label"}:
                candidates.extend([
                    p.parent,
                    p.parent / "labels",
                    p.parent / "_yolo_dataset" / "labels",
                    p.parent / "_yolo_dataset" / "train" / "labels",
                    p.parent / "_yolo_dataset" / "val" / "labels",
                ])
            if p.name.lower() in {"images", "image", "imgs", "crops"}:
                candidates.extend([
                    p.parent / "labels",
                    p.parent / "label",
                    p.parent / "_yolo_dataset" / "labels",
                    p.parent / "_yolo_dataset" / "train" / "labels",
                    p.parent / "_yolo_dataset" / "val" / "labels",
                ])

    add(labels_dir)
    add(images_dir)
    add(project)
    candidates.extend([
        project / "labels",
        project / "label",
        project / "crops" / "labels",
        project / "Mustatils" / "labels",
        project / "Mustatils" / "label",
        yolo_root(project) / "train" / "labels",
        yolo_root(project) / "val" / "labels",
    ])
    return _dedupe_paths(candidates)


def _find_images(project: Path, images_dir=None):
    dirs = _candidate_image_dirs(project, images_dir)
    images = []
    searched = []
    for d in dirs:
        if d.exists() and d.is_dir():
            searched.append(d)
            found = [p for p in d.rglob("*") if _is_image(p)]
            if found:
                print(f"Found {len(found)} images in: {d}", flush=True)
            images.extend(found)

    images = _dedupe_paths(images)
    print("Image folder search complete.", flush=True)
    if searched:
        print("Searched image folders:", flush=True)
        for d in searched[:30]:
            print(f"  - {d}", flush=True)
        if len(searched) > 30:
            print(f"  ... {len(searched) - 30} more", flush=True)
    else:
        print("No existing image folders found from the selected paths.", flush=True)

    return sorted(images)


def prepare_dataset_and_yaml(project: Path, classes=None, val_ratio=0.2, images_dir=None, labels_dir=None):
    data_yaml = write_data_yaml(project, classes)
    split_dataset(project, val_ratio=val_ratio, seed=42, copy=True, images_dir=images_dir, labels_dir=labels_dir)
    return data_yaml


def create_project(project: Path, classes=None):
    project.mkdir(parents=True, exist_ok=True)
    root = ensure_yolo_dirs(project)
    data_yaml = write_data_yaml(project, classes or DEFAULT_CLASSES)
    (project / "project.json").write_text(json.dumps({
        "created_by": "Mustatil QGIS Plugin",
        "dataset_root": str(root),
        "data_yaml": str(data_yaml),
        "classes": classes or DEFAULT_CLASSES,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Project created: {project}", flush=True)
    print(f"YOLO dataset root: {root}", flush=True)


def split_dataset(project: Path, val_ratio=0.2, seed=42, copy=True, images_dir=None, labels_dir=None):
    project = Path(project)
    root = ensure_yolo_dirs(project)

    images = _find_images(project, images_dir=images_dir)
    images = [p for p in images if root not in p.parents or "images" in p.parts]

    if not images:
        print("No images found to split.", flush=True)
        print(f"Selected images-dir: {images_dir}", flush=True)
        print(f"Project folder: {project}", flush=True)
        print("Supported extensions: " + ", ".join(sorted(IMG_EXT)), flush=True)
        print("Put images in one of these folders or select it directly:", flush=True)
        print(f"  {project / 'images'}", flush=True)
        print(f"  {project / 'crops'}", flush=True)
        print(f"  {root / 'train' / 'images'}", flush=True)
        return

    random.seed(seed)
    shuffled = images[:]
    random.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(val_ratio)))) if len(shuffled) > 1 else 0
    val_set = set(shuffled[:n_val])

    train_img = root / "train" / "images"
    train_lab = root / "train" / "labels"
    val_img = root / "val" / "images"
    val_lab = root / "val" / "labels"

    label_dirs = _candidate_label_dirs(project, labels_dir=labels_dir, images_dir=images_dir)
    label_dirs = [d for d in label_dirs if d.exists() and d.is_dir()]
    if label_dirs:
        print("Searched label folders:", flush=True)
        for d in label_dirs[:30]:
            print(f"  - {d}", flush=True)

    def find_label(img: Path):
        for lab_dir in label_dirs:
            c = lab_dir / (img.stem + ".txt")
            if c.exists():
                return c
        sidecar = img.with_suffix(".txt")
        if sidecar.exists():
            return sidecar
        return None

    def transfer(src, dst):
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            if src.resolve() == dst.resolve():
                return
        except Exception:
            pass
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    copied = 0
    empty_labels = 0
    for img in images:
        is_val = img in val_set
        dst_img_dir = val_img if is_val else train_img
        dst_lab_dir = val_lab if is_val else train_lab
        dst_img = dst_img_dir / img.name
        transfer(img, dst_img)
        copied += 1

        lab = find_label(img)
        if lab:
            transfer(lab, dst_lab_dir / (img.stem + ".txt"))
        else:
            (dst_lab_dir / (img.stem + ".txt")).write_text("", encoding="utf-8")
            empty_labels += 1

    print(f"Split complete. Images={len(images)} train={len(images)-len(val_set)} val={len(val_set)}", flush=True)
    print(f"Copied images: {copied}", flush=True)
    if empty_labels:
        print(f"Images without matching label: {empty_labels} (empty YOLO label files were created)", flush=True)
    print(f"Dataset root: {root}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--create-project", default="")
    ap.add_argument("--create-yaml", default="")
    ap.add_argument("--split-project", default="")
    ap.add_argument("--prepare-dataset", default="")
    ap.add_argument("--classes", default="mustatil,false_positive")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--images-dir", default="", help="Optional source folder containing training images")
    ap.add_argument("--labels-dir", default="", help="Optional source folder containing YOLO .txt labels")
    args = ap.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    if args.create_project:
        create_project(Path(args.create_project), classes)
    if args.split_project:
        split_dataset(Path(args.split_project), args.val_ratio, images_dir=args.images_dir or None, labels_dir=args.labels_dir or None)
    if args.create_yaml:
        prepare_dataset_and_yaml(Path(args.create_yaml), classes, args.val_ratio, images_dir=args.images_dir or None, labels_dir=args.labels_dir or None)
    if args.prepare_dataset:
        prepare_dataset_and_yaml(Path(args.prepare_dataset), classes, args.val_ratio, images_dir=args.images_dir or None, labels_dir=args.labels_dir or None)


if __name__ == "__main__":
    main()
