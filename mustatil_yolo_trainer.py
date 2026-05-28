#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mustatil QGIS SAM2 runner using the working Ultralytics SAM path.

This intentionally does NOT use:
- official facebookresearch/sam2 source
- hydra
- omegaconf
- sam2.build_sam / build_sam2

It uses the same working approach as the standalone GUI:
    from ultralytics import SAM
    SAM("sam2_b.pt").predict(image_array, bboxes=[...])

Outputs:
- image.with_suffix(".sam2.json") next to every segmented image
- sam2_manifest.json in the requested output folder
- optional mask PNG files in output/<image_stem>/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
import zipfile
from pathlib import Path

os.environ.setdefault("PYTHONUTF8", "1")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLBACKEND", "Agg")

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def list_images(path: str):
    p = Path(path)
    if p.is_file():
        return [p]
    return [x for x in sorted(p.rglob("*")) if x.suffix.lower() in IMG_EXT]

def resolve_sam_model_file(model_text: str, runtime_root: Path, log=print) -> str:
    """
    Resolve and validate a SAM/SAM2 .pt file.

    Matches the working standalone GUI behavior: local weights are preferred,
    but names such as sam2_b.pt may be passed to Ultralytics for auto-download.
    """
    name = (model_text or "sam2_b.pt").strip().strip('"')
    p = Path(name)

    search = []
    if p.is_absolute():
        search.append(p)
    else:
        base = Path(__file__).resolve().parents[1]
        search += [
            runtime_root / "weights" / p.name,
            runtime_root / p.name,
            base / "weights" / p.name,
            base / "models" / p.name,
            base / p.name,
            Path.cwd() / p.name,
            p,
        ]

    chosen = None
    for c in search:
        try:
            if c.exists():
                chosen = c.resolve()
                break
        except Exception:
            pass

    if chosen is None:
        log(f"SAM model not found locally: {name}. Ultralytics may auto-download it.")
        log("Recommended: place sam2_b.pt in plugin weights/ or models/ folder.")
        return name

    try:
        size = chosen.stat().st_size
    except Exception:
        size = 0

    if size < 1_000_000:
        bad = chosen.with_suffix(chosen.suffix + ".broken")
        try:
            chosen.replace(bad)
            moved = f" Moved to: {bad}"
        except Exception:
            moved = ""
        raise RuntimeError(
            f"SAM model file is too small/corrupt: {chosen} ({size} bytes).{moved}\n"
            "Put a complete sam2_b.pt/sam2_t.pt into weights/ or choose it manually."
        )

    if zipfile.is_zipfile(chosen):
        with zipfile.ZipFile(chosen, "r") as zf:
            bad_member = zf.testzip()
        if bad_member is not None:
            bad = chosen.with_suffix(chosen.suffix + ".broken")
            try:
                chosen.replace(bad)
                moved = f" Moved to: {bad}"
            except Exception:
                moved = ""
            raise RuntimeError(
                f"SAM model archive is corrupt at member {bad_member}: {chosen}.{moved}"
            )

    log(f"Using SAM model: {chosen} ({size/1024/1024:.1f} MB)")
    return str(chosen)

def load_sam_model(model_text: str, runtime_root: Path):
    from ultralytics import SAM
    model_path = resolve_sam_model_file(model_text, runtime_root, log=print)
    try:
        return SAM(model_path)
    except Exception as exc:
        msg = str(exc).lower()
        if "zip" in msg or "pickle" in msg or "failed reading zip archive" in msg:
            try:
                p = Path(model_path)
                if p.exists():
                    bad = p.with_suffix(p.suffix + ".broken")
                    p.replace(bad)
                    print(f"Corrupt SAM model moved to: {bad}", flush=True)
            except Exception:
                pass
            raise RuntimeError(
                "SAM model load failed because the .pt file is incomplete or corrupt. "
                "Replace it with a complete sam2_b.pt/sam2_t.pt."
            ) from exc
        raise

def yolo_boxes_for_image(img: Path, labels_dir: str | None, W: int, H: int):
    boxes = []
    candidates = [
        img.with_suffix(".txt"),
        img.parent.parent / "labels" / (img.stem + ".txt"),
    ]
    if labels_dir:
        candidates.append(Path(labels_dir) / (img.stem + ".txt"))

    for lab in candidates:
        try:
            if not lab.exists():
                continue
            for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 5:
                    cx, cy, bw, bh = map(float, parts[1:5])
                    x1 = (cx - bw / 2.0) * W
                    y1 = (cy - bh / 2.0) * H
                    x2 = (cx + bw / 2.0) * W
                    y2 = (cy + bh / 2.0) * H
                    boxes.append([max(0, x1), max(0, y1), min(W - 1, x2), min(H - 1, y2)])
            if boxes:
                return boxes
        except Exception:
            pass

    return [[0, 0, W - 1, H - 1]]

def segment_one(img_path: Path, sam, out_root: Path, labels_dir: str | None, padding: int, max_crop: int, save_masks: bool):
    from PIL import Image
    import numpy as np
    import cv2

    im = Image.open(img_path).convert("RGB")
    W, H = im.size
    boxes = yolo_boxes_for_image(img_path, labels_dir, W, H)

    print(f"ULTRALYTICS_SAM_START:{img_path.name}:prompts={len(boxes)} size={W}x{H}", flush=True)

    out_dir = out_root / img_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    polys = []
    for i, bb in enumerate(boxes, start=1):
        x1, y1, x2, y2 = map(float, bb)
        left = max(0, int(x1 - padding))
        top = max(0, int(y1 - padding))
        right = min(W, int(x2 + padding))
        bottom = min(H, int(y2 + padding))

        if right <= left or bottom <= top:
            continue

        crop = im.crop((left, top, right, bottom))
        rb = [x1 - left, y1 - top, x2 - left, y2 - top]

        scale = 1.0
        max_side = max(crop.size)
        if max_side > max_crop:
            scale = max_crop / float(max_side)
            crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
            rb = [v * scale for v in rb]

        try:
            res = sam.predict(np.asarray(crop), bboxes=[rb], verbose=False)
            if res and res[0].masks is not None and getattr(res[0].masks, "xy", None):
                inv = 1.0 / scale
                xy_list = res[0].masks.xy
                for mask_idx, xy in enumerate(xy_list):
                    poly = [(float(x) * inv + left, float(y) * inv + top) for x, y in xy]
                    item = {
                        "image": img_path.name,
                        "bbox": [x1, y1, x2, y2],
                        "polygon": poly,
                        "prompt_index": i,
                        "padding": padding,
                        "max_crop": max_crop,
                    }
                    polys.append(item)

                    if save_masks:
                        mask_img = None
                        try:
                            data = res[0].masks.data[mask_idx].cpu().numpy()
                            mask_img = (data.astype("uint8") * 255)
                        except Exception:
                            mask_img = None
                        if mask_img is not None:
                            cv2.imwrite(str(out_dir / f"mask_{i:04d}_{mask_idx:02d}.png"), mask_img)
        except Exception as exc:
            print(f"ULTRALYTICS_SAM_PROMPT_ERROR:{img_path.name}:{i}:{exc}", flush=True)

        if i % 5 == 0 or i == len(boxes):
            print(f"ULTRALYTICS_SAM_PROGRESS:{img_path.name}:{i}/{len(boxes)} masks={len(polys)}", flush=True)

    sidecar = img_path.with_suffix(".sam2.json")
    sidecar.write_text(json.dumps({"polygons": polys, "count": len(polys), "image": str(img_path)}, indent=2), encoding="utf-8")
    print(f"ULTRALYTICS_SAM_SAVED:{sidecar}", flush=True)
    return polys

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["one", "all"], default="one")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-cfg", default="")  # ignored, kept for QGIS compatibility
    ap.add_argument("--model", default="")
    ap.add_argument("--labels-dir", default="")
    ap.add_argument("--padding", type=int, default=96)
    ap.add_argument("--max-crop", type=int, default=1024)
    ap.add_argument("--save-masks", action="store_true")
    args = ap.parse_args()

    print("ULTRALYTICS_SAM_RUNNER=1", flush=True)
    print("NO_HYDRA=1", flush=True)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    runtime_root = Path(__file__).resolve().parents[1]
    model_text = args.model or args.checkpoint or "sam2_b.pt"

    manifest = {
        "runner": "ultralytics_sam",
        "hydra": False,
        "mode": args.mode,
        "model": model_text,
        "images": [],
        "outputs": [],
        "real_sam2": False,
    }

    try:
        sam = load_sam_model(model_text, runtime_root)
        images = list_images(args.images)
        if args.mode == "one" and images:
            images = images[:1]

        all_polys = []
        for idx, img in enumerate(images, start=1):
            print(f"ULTRALYTICS_SAM_IMAGE:{idx}/{len(images)} {img.name}", flush=True)
            polys = segment_one(img, sam, out, args.labels_dir or None, args.padding, args.max_crop, args.save_masks)
            all_polys.extend(polys)
            manifest["outputs"].append({"image": str(img), "count": len(polys), "sidecar": str(img.with_suffix(".sam2.json"))})

        manifest["images"] = [str(p) for p in images]
        manifest["count"] = len(all_polys)
        manifest["real_sam2"] = True
        print(f"ULTRALYTICS_SAM_FINISHED:masks={len(all_polys)}", flush=True)
    except Exception as exc:
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        print("ULTRALYTICS_SAM_FAILED", flush=True)
        traceback.print_exc()

    (out / "sam2_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Output: {out}", flush=True)

if __name__ == "__main__":
    main()
