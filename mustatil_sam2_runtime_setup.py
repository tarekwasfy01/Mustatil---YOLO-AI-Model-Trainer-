#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper for mustatil_external_analyzer.py.

Checks and installs missing runtime dependencies before running detection.
This fixes cases where QGIS selected an existing Python that does not yet have
onnxruntime/rasterio/geopandas installed.
"""
from __future__ import annotations
import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path
import runpy

BASE_REQ = [
    ("numpy", "numpy"),
    ("PIL", "pillow"),
    ("cv2", "opencv-python"),
    ("rasterio", "rasterio"),
    ("geopandas", "geopandas"),
    ("shapely", "shapely"),
    ("pyproj", "pyproj"),
    ("fiona", "fiona"),
    ("pandas", "pandas"),
]

ONNX_REQ = [("onnxruntime", "onnxruntime")]
PT_REQ = [("ultralytics", "ultralytics")]

def missing(reqs):
    out = []
    for import_name, pip_name in reqs:
        if importlib.util.find_spec(import_name) is None:
            out.append(pip_name)
    return out

def pip_install(pkgs):
    if not pkgs:
        return
    print("Missing runtime packages: " + ", ".join(pkgs), flush=True)
    print("Installing missing packages into: " + sys.executable, flush=True)
    cmd = [sys.executable, "-m", "pip", "install", "--upgrade"] + pkgs
    print("$ " + " ".join(cmd), flush=True)
    subprocess.check_call(cmd)

def main():
    print("WRAPPER_STATUS:starting dependency check", flush=True)
    # Parse only model path; forward everything else unchanged.
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", default="")
    known, _ = ap.parse_known_args()

    # Fail fast if the exported clip is missing.
    ap2 = argparse.ArgumentParser(add_help=False)
    ap2.add_argument("--input", default="")
    known2, _ = ap2.parse_known_args()
    if known2.input and not Path(known2.input).exists():
        raise FileNotFoundError(f"Input GeoTIFF does not exist: {known2.input}")

    model_suffix = Path(known.model).suffix.lower()
    reqs = list(BASE_REQ)
    if model_suffix == ".onnx":
        reqs += ONNX_REQ
    elif model_suffix in {".pt", ".pth"}:
        reqs += PT_REQ

    miss = missing(reqs)
    if miss:
        print("WRAPPER_STATUS:installing missing packages", flush=True)
        pip_install(miss)
    print("WRAPPER_STATUS:starting analyzer", flush=True)
    print("DOWNLOAD_TILE:1/1", flush=True)

    analyzer = Path(__file__).with_name("mustatil_external_analyzer.py")
    sys.argv = [str(analyzer)] + sys.argv[1:]
    runpy.run_path(str(analyzer), run_name="__main__")

if __name__ == "__main__":
    main()