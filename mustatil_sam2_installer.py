#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
External Mustatil ONNX +  analyzer for the QGIS plugin.

This script is intentionally executed outside the QGIS Python runtime to avoid
Torch/ONNX/GDAL DLL conflicts inside QGIS.
"""
import argparse
import math
import math
import json
import math
import os
from pathlib import Path
import json

def _read_worldfile_transform(path):
    """Read .tfw sidecar and return rasterio Affine transform, or None."""
    try:
        from affine import Affine
        p = Path(path)
        candidates = [
            p.with_suffix(".tfw"),
            p.with_suffix(".tifw"),
            p.with_suffix(".wld"),
        ]
        for wf in candidates:
            if wf.exists():
                vals = [float(x.strip()) for x in wf.read_text(encoding="utf-8", errors="ignore").splitlines()[:6]]
                if len(vals) == 6:
                    # World file order: A, D, B, E, C, F. C/F are pixel center.
                    A, D, B, E, C, F = vals
                    return Affine(A, B, C - A / 2.0 - B / 2.0, D, E, F - D / 2.0 - E / 2.0)
    except Exception:
        return None
    return None

def _read_sidecar_crs(path):
    """Read .crs.json sidecar and return rasterio CRS, or None."""
    try:
        from rasterio.crs import CRS
        p = Path(path).with_suffix(".crs.json")
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8", errors="ignore"))
            epsg = int(data.get("epsg", 0))
            if epsg:
                return CRS.from_epsg(epsg)
    except Exception:
        return None
    return None


import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-float(x)))

def load_image(path):
    import rasterio
    from rasterio.plot import reshape_as_image
    with rasterio.open(path) as src:
        arr = src.read()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
    img = reshape_as_image(arr)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if img.dtype != np.uint8:
        p2, p98 = np.percentile(img, (2, 98))
        if p98 > p2:
            img = np.clip((img - p2) * 255.0 / (p98 - p2), 0, 255).astype(np.uint8)
        else:
            img = np.clip(img, 0, 255).astype(np.uint8)
    return img, transform, crs, bounds

def preprocess_tile(tile, size):
    import cv2
    resized = cv2.resize(tile, (size, size), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None, ...]
    return arr

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = max(0, a[2]-a[0]) * max(0, a[3]-a[1])
    area_b = max(0, b[2]-b[0]) * max(0, b[3]-b[1])
    return inter / max(1e-9, area_a + area_b - inter)

def nms(boxes, scores, thr):
    order = np.argsort(scores)[::-1]
    keep = []
    while len(order):
        i = int(order[0]); keep.append(i)
        rem = []
        for j in order[1:]:
            if iou(boxes[i], boxes[int(j)]) <= thr:
                rem.append(int(j))
        order = np.array(rem, dtype=int)
    return keep

def parse_yolo_output(outputs, conf, tile_x, tile_y, scale_x, scale_y):
    # Handles common YOLOv8 ONNX output [1, n, 5+] or [1, 5+, n].
    out = outputs[0]
    if isinstance(out, list):
        out = out[0]
    out = np.asarray(out)
    out = np.squeeze(out)
    if out.ndim != 2:
        return []
    if out.shape[0] < out.shape[1] and out.shape[0] <= 100:
        out = out.T
    detections = []
    for row in out:
        if len(row) < 5:
            continue
        cx, cy, w, h = row[:4]
        if len(row) == 5:
            score = row[4]
            cls = 0
        else:
            cls_scores = row[4:]
            cls = int(np.argmax(cls_scores))
            score = float(cls_scores[cls])
        if score < conf:
            continue
        x1 = (cx - w / 2.0) * scale_x + tile_x
        y1 = (cy - h / 2.0) * scale_y + tile_y
        x2 = (cx + w / 2.0) * scale_x + tile_x
        y2 = (cy + h / 2.0) * scale_y + tile_y
        detections.append([float(x1), float(y1), float(x2), float(y2), float(score), int(cls)])
    return detections

def pixel_box_to_polygon(transform, box):
    x1, y1, x2, y2 = box
    pts_px = [(x1,y1), (x2,y1), (x2,y2), (x1,y2), (x1,y1)]
    pts = [transform * p for p in pts_px]
    return pts

def write_gpkg(out_path, crs, transform, detections, args_model_name="model", args_preset_name="Custom Model", args_text_prompt=""):
    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import Polygon

    rows = []

    for det in detections:
        box = det["box"]
        pts = pixel_box_to_polygon(transform, box)
        rows.append({
            "score": float(det["score"]),
            "formscore": float(det["formscore"]),
            "class_id": int(det["class_id"]),
            "model": Path(args_model_name).name,
            "preset": args_preset_name,
            "text_prompt": args_text_prompt,
            "geometry": Polygon(pts),
        })

    # Convert Rasterio CRS to a form GeoPandas/Fiona writes reliably.
    crs_out = None
    try:
        if crs is not None:
            crs_out = crs.to_wkt()
    except Exception:
        crs_out = crs

    if rows:
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs_out)
    else:
        # Valid empty polygon layer. This is important so QGIS can import it
        # as a layer even when no detections passed the thresholds.
        df = pd.DataFrame({
            "score": pd.Series(dtype="float64"),
            "formscore": pd.Series(dtype="float64"),
            "class_id": pd.Series(dtype="int64"),
            "model": pd.Series(dtype="str"),
            "preset": pd.Series(dtype="str"),
            "text_prompt": pd.Series(dtype="str"),
            "geometry": pd.Series(dtype="object"),
        })
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs_out)

    out_path = Path(out_path)
    if out_path.exists():
        try:
            out_path.unlink()
        except Exception:
            pass

    # Force a real GeoPackage vector layer name so QGIS can load it by URI:
    # file.gpkg|layername=mustatile_detections
    gdf.to_file(
        out_path,
        driver="GPKG",
        layer="mustatile_detections",
        index=False,
        engine="fiona"
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--preset-name", default="Mustatile_")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tile-size", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=160)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--class-filter", default="", help="Comma-separated class ids to keep, e.g. 2,5,7")
    ap.add_argument("--device", default="cpu", help="Inference device: cpu or cuda")
    args = ap.parse_args()

    class_filter = None
    if args.class_filter.strip():
        try:
            class_filter = {int(x.strip()) for x in args.class_filter.split(",") if x.strip()}
            print(f"Class filter active: {sorted(class_filter)}", flush=True)
        except Exception as exc:
            raise RuntimeError(f"Invalid --class-filter: {args.class_filter}") from exc

    img, transform, crs, bounds = load_image(args.input)
    h, w = img.shape[:2]

    model_suffix = Path(args.model).suffix.lower()
    step = max(1, args.tile_size - args.overlap)
    all_dets = []
    total_tiles = max(1, math.ceil(h / step) * math.ceil(w / step))
    tile_counter = 0
    print(f"TILE_PROGRESS:0/{total_tiles}", flush=True)
    total_tiles = max(1, ((h + step - 1) // step) * ((w + step - 1) // step))
    tile_counter = 0
    

    if model_suffix == ".onnx":
        import onnxruntime as ort
        sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        input_name = inp.name
        in_shape = inp.shape
        model_size = args.tile_size
        try:
            if isinstance(in_shape[2], int):
                model_size = int(in_shape[2])
        except Exception:
            pass

        for y in range(0, max(1, h), step):
            for x in range(0, max(1, w), step):
                tile_counter += 1
                print(f"TILE_PROGRESS:{tile_counter}/{total_tiles}", flush=True)
                x2 = min(w, x + args.tile_size)
                y2 = min(h, y + args.tile_size)
                tile = img[y:y2, x:x2]
                if tile.size == 0:
                    continue
                padded = np.zeros((args.tile_size, args.tile_size, 3), dtype=np.uint8)
                padded[:tile.shape[0], :tile.shape[1], :] = tile
                arr = preprocess_tile(padded, model_size)
                outs = sess.run(None, {input_name: arr})
                scale_x = args.tile_size / float(model_size)
                scale_y = args.tile_size / float(model_size)
                dets = parse_yolo_output(outs, args.conf, x, y, scale_x, scale_y)
                for d in dets:
                    d[0] = max(0, min(w, d[0])); d[2] = max(0, min(w, d[2]))
                    d[1] = max(0, min(h, d[1])); d[3] = max(0, min(h, d[3]))
                    if d[2] > d[0] and d[3] > d[1]:
                        all_dets.append(d)
    elif model_suffix in {".pt", ".pth"}:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("PyTorch/Ultralytics model selected, but ultralytics is not installed. Run the runtime installer again. Details: " + str(exc))

        model = YOLO(args.model)
        for y in range(0, max(1, h), step):
            for x in range(0, max(1, w), step):
                tile_counter += 1
                print(f"TILE_PROGRESS:{tile_counter}/{total_tiles}", flush=True)
                x2 = min(w, x + args.tile_size)
                y2 = min(h, y + args.tile_size)
                tile = img[y:y2, x:x2]
                if tile.size == 0:
                    continue
                result_list = model.predict(tile, conf=args.conf, imgsz=args.tile_size, verbose=False, device=args.device)
                for res in result_list:
                    if res.boxes is None:
                        continue
                    for box in res.boxes:
                        xyxy = box.xyxy.cpu().numpy().reshape(-1).tolist()
                        score = float(box.conf.cpu().numpy().reshape(-1)[0])
                        cls = int(box.cls.cpu().numpy().reshape(-1)[0]) if box.cls is not None else 0
                        d = [xyxy[0] + x, xyxy[1] + y, xyxy[2] + x, xyxy[3] + y, score, cls]
                        d[0] = max(0, min(w, d[0])); d[2] = max(0, min(w, d[2]))
                        d[1] = max(0, min(h, d[1])); d[3] = max(0, min(h, d[3]))
                        if d[2] > d[0] and d[3] > d[1]:
                            all_dets.append(d)
    else:
        raise RuntimeError("Unsupported model format. Use .onnx, .pt or .pth")

    

    if class_filter is not None:
        all_dets = [d for d in all_dets if int(d[5]) in class_filter]
        print(f"After class filter: {len(all_dets)} detections", flush=True)

    if all_dets:
        boxes = [d[:4] for d in all_dets]
        scores = [d[4] for d in all_dets]
        keep = nms(boxes, scores, args.nms)
        all_dets = [all_dets[i] for i in keep]

    final = []
    for d in all_dets:
        final.append({"box": d[:4], "score": d[4], "class_id": d[5], "formscore": 1.0})

    
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_gpkg(args.output, crs, transform, final, args.model, args.preset_name, args.prompt)

    try:
        count_path = Path(args.output).with_suffix(".count.txt")
        count_path.write_text(str(len(final)), encoding="utf-8")
        print(f"DETECTION_COUNT:{len(final)}", flush=True)
    except Exception as exc:
        print(f"Could not write detection count sidecar: {exc}", flush=True)

    
    print(f"Finished. Raw detections={len(all_dets)}  accepted={len(final)}")
    if args.prompt:
        print(f"Text prompt: {args.prompt}")
    if len(final) == 0:
        print("No detections passed the current confidence/ thresholds. Empty GeoPackage layer was written.")
    print(f"Output: {args.output}")

if __name__ == "__main__":
    main()