# QGIS Plugin Release Checklist

This package was prepared with the QGIS plugin repository checks in mind.

## Structure

- [x] The ZIP contains one top-level plugin folder: `MustatilQGIS/`.
- [x] The top-level plugin folder contains `metadata.txt`.
- [x] The top-level plugin folder contains `__init__.py`.
- [x] The top-level plugin folder contains `LICENSE`.
- [x] Source code is included in the plugin folder.
- [x] Compiled/cache files such as `__pycache__`, `.pyc`, and `.pyo` are excluded.

## Repository consistency

- [ ] The public plugin repository must contain the same source code as this ZIP file.
- [ ] The public plugin repository should exclude generated/compiled files.
- [ ] Large binary model files should be documented clearly if included, or distributed through releases if preferred.

## Metadata

- [x] The plugin includes a meaningful English description.
- [x] `homepage` is set to: https://github.com/tarekwasfy01
- [x] `repository` is set to: https://github.com/tarekwasfy01/Mustatil---YOLO-AI-Model-Trainer-
- [x] `tracker` is set to: https://github.com/tarekwasfy01/Mustatil---YOLO-AI-Model-Trainer-/issues
- [x] Homepage, tracker and repository URLs are public.
- [ ] Before official upload, verify that the repository contains the same final source code as this ZIP.

## Testing

- [ ] Install this exact ZIP in QGIS.
- [ ] Confirm the plugin opens without `initGui()` errors.
- [ ] Test Detection on a small raster/XYZ area.
- [ ] Test QGIS GeoPackage layer import.
- [ ] Test Annotator preview and zoom.
- [ ] Test YOLO project creation and `data.yaml`.
- [ ] Test Runtime installer on a clean Windows/QGIS setup.
- [ ] Confirm this version works as expected before uploading.

## Declaration text

Use the following confirmations only after the checks above are true:

- My plugin ZIP file follows the correct QGIS plugin structure, with a top-level folder containing `metadata.txt` and `__init__.py`.
- The plugin repository is not empty and contains the same source code as the ZIP file, excluding compiled files such as `__pycache__`.
- My plugin includes a meaningful description written in English.
- All metadata links (`homepage`, `tracker`, `repository`) are valid and publicly accessible.
- I have tested this plugin version with QGIS and it works as expected.
