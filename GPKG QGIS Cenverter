#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GeoPackage QGIS Converter GUI

Purpose:
- Read a problematic/minimal GeoPackage
- Rewrite it as a QGIS-friendly GeoPackage using GeoPandas/Fiona/GDAL
- Optional GeoJSON export
- Tries to preserve CRS and attributes

Place this file on Desktop and run with START_GPKG_QGIS_CONVERTER.bat.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import shutil
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


APP_TITLE = "GeoPackage QGIS Converter"


def have_module(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception:
        return False


def inspect_gpkg(path: Path) -> dict:
    info = {
        "tables": [],
        "contents": [],
        "geometry_columns": [],
        "srs": [],
        "feature_tables": [],
    }
    con = sqlite3.connect(str(path))
    try:
        cur = con.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        info["tables"] = [r[0] for r in cur.fetchall()]

        if "gpkg_contents" in info["tables"]:
            cur.execute("SELECT * FROM gpkg_contents")
            info["contents"] = cur.fetchall()

        if "gpkg_geometry_columns" in info["tables"]:
            cur.execute("SELECT * FROM gpkg_geometry_columns")
            info["geometry_columns"] = cur.fetchall()
            info["feature_tables"] = [r[0] for r in info["geometry_columns"]]

        if "gpkg_spatial_ref_sys" in info["tables"]:
            cur.execute("SELECT srs_id, organization, organization_coordsys_id, srs_name FROM gpkg_spatial_ref_sys")
            info["srs"] = cur.fetchall()
    finally:
        con.close()
    return info


def convert_with_geopandas(src: Path, dst: Path, layer: str | None = None, log_fn=print):
    import geopandas as gpd
    import fiona

    layers = fiona.listlayers(str(src))
    if not layers:
        raise RuntimeError("No layers found in source GeoPackage.")

    if layer and layer in layers:
        layers_to_convert = [layer]
    else:
        layers_to_convert = layers

    if dst.exists():
        dst.unlink()

    written = []
    for i, lyr in enumerate(layers_to_convert):
        log_fn(f"Reading layer: {lyr}")
        gdf = gpd.read_file(str(src), layer=lyr)

        if gdf.empty:
            log_fn(f"Layer {lyr} is empty, skipping.")
            continue

        log_fn(f"  Features: {len(gdf)}")
        log_fn(f"  CRS: {gdf.crs}")

        # If CRS is missing but coordinates look like WebMercator, set EPSG:3857.
        if gdf.crs is None:
            try:
                bounds = gdf.total_bounds
                max_abs = max(abs(float(bounds[0])), abs(float(bounds[1])), abs(float(bounds[2])), abs(float(bounds[3])))
                if max_abs > 1000:
                    gdf = gdf.set_crs("EPSG:3857", allow_override=True)
                    log_fn("  CRS missing; coordinates look metric -> set EPSG:3857")
                else:
                    log_fn("  CRS missing; leaving undefined.")
            except Exception as exc:
                log_fn(f"  CRS heuristic failed: {exc}")

        # Clean unsupported/object columns by converting to string.
        for col in list(gdf.columns):
            if col == gdf.geometry.name:
                continue
            if str(gdf[col].dtype) == "object":
                gdf[col] = gdf[col].astype(str)

        out_layer = lyr if len(layers_to_convert) > 1 else "detections"
        log_fn(f"Writing layer: {out_layer}")
        gdf.to_file(str(dst), layer=out_layer, driver="GPKG")
        written.append(out_layer)

    if not written:
        raise RuntimeError("No features were written.")
    return written


def convert_to_geojson(src: Path, dst: Path, layer: str | None = None, log_fn=print):
    import geopandas as gpd
    import fiona

    layers = fiona.listlayers(str(src))
    if not layers:
        raise RuntimeError("No layers found in source GeoPackage.")
    lyr = layer if layer in layers else layers[0]
    log_fn(f"Reading layer for GeoJSON: {lyr}")
    gdf = gpd.read_file(str(src), layer=lyr)
    if gdf.crs is None:
        try:
            bounds = gdf.total_bounds
            max_abs = max(abs(float(bounds[0])), abs(float(bounds[1])), abs(float(bounds[2])), abs(float(bounds[3])))
            if max_abs > 1000:
                gdf = gdf.set_crs("EPSG:3857", allow_override=True)
                log_fn("CRS missing; coordinates look metric -> set EPSG:3857")
        except Exception:
            pass
    if dst.exists():
        dst.unlink()
    gdf.to_file(str(dst), driver="GeoJSON")
    return dst


class ConverterGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x650")

        self.src_path = tk.StringVar()
        self.dst_path = tk.StringVar()
        self.geojson_path = tk.StringVar()
        self.layer_name = tk.StringVar(value="")

        self._build_ui()
        self.after(100, self.log_deps)

    def _build_ui(self):
        main = ttk.Frame(self, padding=10)
        main.pack(fill=tk.BOTH, expand=True)

        box = ttk.LabelFrame(main, text="Files", padding=8)
        box.pack(fill=tk.X)

        ttk.Label(box, text="Source .gpkg").grid(row=0, column=0, sticky="w", pady=3)
        ttk.Entry(box, textvariable=self.src_path).grid(row=0, column=1, sticky="ew", pady=3)
        ttk.Button(box, text="Browse", command=self.pick_src).grid(row=0, column=2, padx=4)

        ttk.Label(box, text="Output QGIS .gpkg").grid(row=1, column=0, sticky="w", pady=3)
        ttk.Entry(box, textvariable=self.dst_path).grid(row=1, column=1, sticky="ew", pady=3)
        ttk.Button(box, text="Browse", command=self.pick_dst).grid(row=1, column=2, padx=4)

        ttk.Label(box, text="Optional layer name").grid(row=2, column=0, sticky="w", pady=3)
        ttk.Entry(box, textvariable=self.layer_name).grid(row=2, column=1, sticky="ew", pady=3)

        ttk.Label(box, text="Optional GeoJSON").grid(row=3, column=0, sticky="w", pady=3)
        ttk.Entry(box, textvariable=self.geojson_path).grid(row=3, column=1, sticky="ew", pady=3)
        ttk.Button(box, text="Browse", command=self.pick_geojson).grid(row=3, column=2, padx=4)

        box.columnconfigure(1, weight=1)

        btns = ttk.Frame(main)
        btns.pack(fill=tk.X, pady=8)
        ttk.Button(btns, text="Inspect Source", command=self.inspect_clicked).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(btns, text="Convert to QGIS GeoPackage", command=self.convert_clicked).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)
        ttk.Button(btns, text="Export GeoJSON", command=self.geojson_clicked).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=3)

        self.log = tk.Text(main, height=28, wrap=tk.WORD)
        self.log.pack(fill=tk.BOTH, expand=True)

    def log_msg(self, s=""):
        self.log.insert(tk.END, str(s) + "\n")
        self.log.see(tk.END)
        self.update_idletasks()

    def log_deps(self):
        self.log_msg(f"GeoPandas: {'OK' if have_module('geopandas') else 'missing'}")
        self.log_msg(f"Fiona: {'OK' if have_module('fiona') else 'missing'}")
        self.log_msg(f"Shapely: {'OK' if have_module('shapely') else 'missing'}")
        self.log_msg(f"PyProj: {'OK' if have_module('pyproj') else 'missing'}")

    def pick_src(self):
        p = filedialog.askopenfilename(filetypes=[("GeoPackage", "*.gpkg"), ("All", "*.*")])
        if p:
            self.src_path.set(p)
            src = Path(p)
            if not self.dst_path.get():
                self.dst_path.set(str(src.with_name(src.stem + "_qgis.gpkg")))
            if not self.geojson_path.get():
                self.geojson_path.set(str(src.with_name(src.stem + ".geojson")))

    def pick_dst(self):
        p = filedialog.asksaveasfilename(defaultextension=".gpkg", filetypes=[("GeoPackage", "*.gpkg")])
        if p:
            self.dst_path.set(p)

    def pick_geojson(self):
        p = filedialog.asksaveasfilename(defaultextension=".geojson", filetypes=[("GeoJSON", "*.geojson")])
        if p:
            self.geojson_path.set(p)

    def inspect_clicked(self):
        try:
            src = Path(self.src_path.get().strip().strip('"'))
            if not src.exists():
                raise RuntimeError("Source file not found.")
            self.log_msg("=" * 80)
            self.log_msg(f"Inspecting: {src}")
            info = inspect_gpkg(src)
            self.log_msg(f"Tables: {info['tables']}")
            self.log_msg(f"Geometry columns: {info['geometry_columns']}")
            self.log_msg(f"Contents: {info['contents']}")
            self.log_msg(f"SRS: {info['srs']}")
            self.log_msg(f"Feature layers: {info['feature_tables']}")
        except Exception as exc:
            self.log_msg(f"Inspect ERROR: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))

    def convert_clicked(self):
        try:
            src = Path(self.src_path.get().strip().strip('"'))
            dst = Path(self.dst_path.get().strip().strip('"'))
            if not src.exists():
                raise RuntimeError("Source file not found.")
            if not dst:
                raise RuntimeError("Choose output file.")
            layer = self.layer_name.get().strip() or None

            self.log_msg("=" * 80)
            self.log_msg(f"Converting: {src}")
            self.log_msg(f"Output: {dst}")
            written = convert_with_geopandas(src, dst, layer, self.log_msg)
            self.log_msg(f"Done. Written layers: {written}")
            self.log_msg("Try loading the new _qgis.gpkg in QGIS.")
        except Exception as exc:
            self.log_msg(f"Convert ERROR: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))

    def geojson_clicked(self):
        try:
            src = Path(self.src_path.get().strip().strip('"'))
            dst = Path(self.geojson_path.get().strip().strip('"'))
            if not src.exists():
                raise RuntimeError("Source file not found.")
            layer = self.layer_name.get().strip() or None
            self.log_msg("=" * 80)
            self.log_msg(f"Exporting GeoJSON: {dst}")
            convert_to_geojson(src, dst, layer, self.log_msg)
            self.log_msg("GeoJSON exported.")
        except Exception as exc:
            self.log_msg(f"GeoJSON ERROR: {exc}")
            messagebox.showerror(APP_TITLE, str(exc))


if __name__ == "__main__":
    app = ConverterGUI()
    app.mainloop()
