MustatilQGIS - Mustatil QGIS Plugin

This version is modeled after the dockable QGIS plugin layout:
- QGIS toolbar and menu
- Dockable side panel
- Tabs: Detection, Runtime, Log
- Raster layer selector
- Built-in Mustatile preset
- External Python runtime installer
- Built-in bestf260.onnx model
- Built-in  JSON
- Select map area directly in QGIS
- Export selected raster area as GeoTIFF
- Run ONNX detection and  filtering
- Load georeferenced GeoPackage result into QGIS

Installation:
1. Extract this ZIP.
2. Copy the folder "MustatilQGIS" to:
   C:/Users/<USER>/AppData/Roaming/QGIS/QGIS3/profiles/default/python/plugins/
3. Restart QGIS.
4. Enable "MustatilQGIS" in the QGIS Plugin Manager.
5. Open the Mustatil AI Detection dock.
6. In the Runtime tab, click "Install external runtime".
7. In the Detection tab, choose a raster layer and click "Select map area".

Important:
The AI dependencies are intentionally installed into an external runtime instead
of the QGIS Python environment. This avoids DLL conflicts with QGIS/GDAL.

PATHFIX note:
If the normal command "powershell" is unavailable, this package calls
C:/Windows/System32/WindowsPowerShell/v1.0/powershell.exe directly.
It also falls back to curl.exe and bitsadmin for downloads.
Use CHECK_RUNTIME.bat after installation to verify imports.

----------------------------------------------------------------
Additional integrated model
----------------------------------------------------------------

Tree detection preset added:
- treedetect.pt

Source / citation:
https://zenodo.org/records/14512739

Please cite the original Zenodo publication when using this model in
research publications or derivative works.

Empty GeoPackage fix:
This version writes a valid empty GeoPackage layer when no detections pass the
YOLO confidence and/or  threshold. You can then lower the sliders
and rerun the last clip from the Detection tab.

RubberBand + GDAL clip fix:
- The red selection rectangle is now drawn continuously while dragging.
- Raster export now uses QGIS Processing GDAL cliprasterbyextent instead of
  QgsRasterFileWriter. This avoids RasterFileWriter return code 3 on affected
  QGIS installations/layers.

Preset update:
- AI preset now includes Mustatile, TreeDetect and Custom model.
- Custom model accepts .onnx, .pt and .pth.
- TreeDetect uses the integrated treedetect.pt model.
- .pt/.pth models require ultralytics/torch in the external runtime.

Selection/GPKG fix:
- The selected map area is displayed with red outline only and transparent fill.
- Detection outputs are written as a named GeoPackage vector layer:
  mustatile_detections
- QGIS loads the layer using:
  file.gpkg|layername=mustatile_detections

Tree preset / box-only selection fix:
- AI Preset dropdown now explicitly offers:
  
  TreeDetect / treedetect.pt
  Custom model
- Select map area now draws only one transparent red outline box.

Line-only selection fix:
- The selection preview now uses QgsWkbTypes.LineGeometry instead of PolygonGeometry.
- This guarantees there is no filled rectangle, only the red outline.

Verified v2.6:
- update_preset_fields is called only after custom_model_path and sliders exist.
- TreeDetect preset and Custom model selection are present.
- Selection uses LineGeometry only, so no filled red rectangle is possible.
- Raster clipping uses GDAL cliprasterbyextent.
- Runtime Python detection is present.

Stable reopen fix v2.7:
- update_preset_fields no longer runs before runtime widgets exist.
- update_preset_fields is guarded with hasattr() so the dock can always open.
- processing is imported lazily inside raster clipping instead of at plugin load.
- TreeDetect and Custom preset selection remain available.

All-functions integration v3.0:
- Based on the last stable QGIS plugin.
- Adds YOLO Trainer tab using external runtime and Ultralytics.
- Keeps detection, TreeDetect, Custom model, runtime detection, GDAL clip and line-only map selection.
- Heavy functionality remains outside QGIS Python to avoid Torch/GDAL/QGIS DLL conflicts.
- The original Mustatil workflow included Detection + Crops, Crop Annotator + SAM2,
  YOLO Trainer, SAM2 image list,  and Detection with . This
  plugin exposes the stable QGIS-native parts directly and the heavier workflow
  via external runtime tools.

Clip fix v3.1:
- Raster export now tries three fallback methods:
  1. QGIS Processing GDAL cliprasterbyextent
  2. osgeo.gdal.Translate
  3. QgsRasterFileWriter
- Detailed clip errors are written into the plugin log.

Menu fix v3.2:
- Added missing MustatilQGISPlugin.open_training_direct().
- Added missing MustatilQGISPlugin.open_toolkit_direct().
- Fixes initGui() plugin load error.

XYZ render fix v3.3:
- Fixes detection on XYZ/web tile layers such as Google Satellite:
  QGIS renders the selected map extent to an RGB GeoTIFF and georeferences it.
- Normal file rasters still try GDAL clip/translate and RasterFileWriter first.
- Fixes empty Python path issue where QProcess tried to start ".".

v3.4 project/yaml/annotator:
- YOLO Trainer tab now has:
  Create Project
  Choose Project
  Create / Update data.yaml
  class list field
- Added Annotator tab:
  choose image folder
  choose label folder
  class list
  external YOLO box annotator

v3.5 detection fix:
- Detection now detects XYZ/web tile sources before GDAL clipping.
- Google Satellite / XYZ layers are rendered by QGIS into georeferenced GeoTIFF.
- Runtime Python resolution fixed: empty path no longer becomes ".".

v3.6 inline annotator + SAM2 buttons:
- Annotator tab now shows an image preview directly inside the QGIS dock.
- Removed dependency on pressing only "Open Annotator"; use Load images into preview.
- Inline tools:
  Prev / Next
  Add centered box
  Delete last box
  Save label
- SAM2 buttons:
  SAM2 current image
  SAM2 all pictures
- SAM2 runner currently prepares outputs/manifests and is ready for real SAM2 checkpoint integration.

v3.7 progress percent:
- Detection page now shows a progress bar with percent text.
- Run on selected raster extent updates progress through export/analyze/finish.
- Analyzer prints coarse PROGRESS markers for QGIS to display.

v3.8 device dropdown:
- YOLO Trainer now uses a device dropdown instead of text input.
- Options:
  cpu
  cuda
  directml
  opencl
- OpenCL falls back to CPU because standard Torch/Ultralytics training
  has no generic OpenCL backend.

v3.9 SAM2 + mouse boxes:
- Inline annotator supports mouse-drag boxes directly in the QGIS dock.
- Box class dropdown:
  0 positive
  1 false
- Positive boxes are green; false boxes are red.
- Right click deletes the last box.
- SAM2 runner now attempts real SAM2 if the external runtime has:
  SAM2 package installed
  checkpoint selected
  model config provided
- If SAM2 is not installed/configured, it writes a manifest and clear error instead of crashing.

v4.0 QGIS layer import:
- Detection GeoPackages are now automatically loaded into QGIS after analysis.
- Tries:
  gpkg|layername=detections
  then generic OGR open fallback.
- Imported detections appear directly as a QGIS vector layer.

v4.1 project/SAM2/yolo_datasets:
- YOLO datasets are created under:
  project/yolo_datasets/
    train/images
    train/labels
    val/images
    val/labels
    data.yaml
- YOLO Trainer has split train/val helper.
- YOLO trainer auto-creates val split if val/images is missing.
- Annotator has Choose Project and points to yolo_datasets/train/images.
- Toolkit tab is user-facing  tab.
- SAM2 has automatic installer/downloader button and checkpoint/config autofill.

v4.2 annotator zoom:
- Annotator preview now supports zoom:
  mouse wheel zoom
  + / - buttons
  Fit button
  draggable preview canvas
- Zoom percent indicator added.

v4.3 raster progress fix:
- Fixed broken percentages >100%.
- Detection now shows raster/tile progress:
  Raster 10/30
  Raster 11/30
  etc.
- Percent is derived from processed raster tiles and capped correctly.

v4.4 runtime/SAM2/middle-mouse fixes:
- Detection now starts through mustatil_run_analyzer_wrapper.py.
  Missing packages such as onnxruntime are installed automatically into the selected Python.
- SAM2 installer now tries GitHub ZIP installation if git is missing.
- Annotator preview supports middle mouse drag for panning while zoomed.

v4.5 TreeDetect model replacement:
- Integrated TreeDetect model replaced with uploaded best (8).pt.
- Preset remains: TreeDetect / treedetect.pt

v4.6 clip path fix:
- Detection exports clips to unique files:
  output/Mustatil_QGIS_Clips/mustatil_qgis_clip_YYYYMMDD_HHMMSS.tif
- The plugin verifies the GeoTIFF exists before starting analysis.
- The analyzer wrapper fails fast if the input TIFF is missing.
- Prevents stale C:/Users/.../Desktop/mustatil_qgis_clip.tif missing-file errors.

v4.7 single selection outline:
- Starting a new Detect area selection clears the previous red selection outline.
- Export-only selection also clears previous outlines.
- Only one red selection border remains visible.

v4.8 zoom18 clip export:
- XYZ/web tile raster clips are now exported at approximately zoom level 18 resolution.
- Higher-detail GeoTIFF exports for detection.
- Maximum render size increased to 16384 px.

v4.9 text prompt detection:
- Detection tab now has a GeoAI-like text prompt field:
  "What should the model detect?"
- Prompt examples:
  mustatil
  tree
  trees
  Baum
  circular structure
- The prompt routes automatically to available installed presets:
  TreeDetect / treedetect.pt
  
- The text prompt is stored in the detection output attributes when possible.

v5.0 more presets + detection count:
- Added AI presets:
  Cars / YOLO COCO
  Airplanes / YOLO COCO
  Houses-Buildings / Custom model
- Text prompt routing now understands:
  cars/autos/vehicles
  airplanes/flugzeuge/aircraft
  houses/buildings/gebäude
- Cars uses COCO class id 2.
- Airplanes uses COCO class id 4.
- Houses/buildings requires a suitable custom model for best results.
- Detection page shows the number of detections at the bottom.

v5.1 detection feedback fix:
- Detection process runs Python with -u unbuffered output.
- QProcess output is merged and displayed immediately.
- Heartbeat/status remains visible while dependencies/model startup run.
- Wrapper prints dependency/analyzer status.
- Analyzer prints TILE_PROGRESS when processing rasters.

v5.2 tile download status:
- Detection log/progress now shows tile download status.
- Example:
  Download Tiles 10/50
  Download Tiles 11/50
- Status appears during clip preparation before raster detection.

v5.3 detection hang fix:
- Adds "Max clip size px" setting on Detection page.
- Default max clip size is 6144 px.
- Zoom-18 export is now capped to avoid huge QGIS render jobs.
- Logs actual rendered clip size before detection.
- Adds Cancel Detection button.
- Keeps UI responsive during clip export status updates.

v5.4 red detection boxes:
- Imported QGIS detection layers are automatically styled as red outline boxes.
- Fill is transparent / no fill.
- Only the box frame is visible.

v5.5 TIFF download/render progress:
- TIFF export now shows tile-style download/render information.
- Added labels:
  Tiles: X/Y
  Speed: X tiles/sec
  Phase: preparing/downloading/rendering/finished

v5.7 SAM2 source install fix:
- SAM2 installer no longer requires git.
- SAM2 installer no longer pip-installs the GitHub ZIP, avoiding the Windows tarfile/ntpath.ALLOW_MISSING error.
- SAM2 source is downloaded and extracted manually.
- SAM2 runner adds the extracted source path to sys.path.
- A minimal iopath shim is written if iopath cannot be installed.
- If the checkpoint is missing, the plugin logs a clear instruction to run the SAM2 installer first.

v6.0 Mustatil test coordinate button:
- Added Preset button: Jump to Mustatil test coordinate.
- Coordinate:
  22.99558462319434, 44.02986257629997
- The button activates  and zooms the QGIS map canvas to the test location.

v6.1 SAM2 dependency fix:
- SAM2 runner now auto-installs missing runtime dependencies before importing SAM2.
- Fixes: No module named 'hydra'
- Dependencies checked: hydra-core, omegaconf, tqdm, opencv-python, pillow, numpy.
- iopath is attempted and falls back to the local shim if installation fails.

v6.2 SAM2 runtime integration:
- SAM2 runtime setup is now integrated directly:
  scripts/mustatil_sam2_runtime_setup.py
- Runtime setup installs SAM2 Python dependencies, downloads SAM2 source, checkpoint and config.
- SAM2 runner auto-repairs missing runtime files before segmentation.
- Existing Install SAM2 button now uses the integrated setup.
- Runtime installer BAT also tries to prepare SAM2 runtime files.

v6.3 SAM2 Hydra/OmegaConf shim:
- Fixes incompatible old hydra in QGIS/Python (`initialize_config_module` missing).
- Avoids hydra-core pip upgrade that fails with ntpath.ALLOW_MISSING.
- Writes local hydra, omegaconf and iopath shims into runtime/sam2/sam2_shims.
- SAM2 runner loads these shims before site-packages.

v6.4 SAM2 hydra.core shim:
- Adds local hydra.core package for SAM2 imports.
- Adds hydra.core.global_hydra.GlobalHydra shim.
- Adds hydra.core.config_store.ConfigStore shim.
- SAM2 runner auto-repairs older local shim folders if hydra.core is missing.
- Old --install-package argument is removed/ignored.

v7.0 external SAM2 runtime:
- SAM2 no longer runs inside QGIS Python.
- Creates isolated runtime_sam2/python with its own dependencies.
- QGIS calls SAM2 through QProcess/wrapper only.
- Avoids QGIS hydra/numpy/torch conflicts.
- Install SAM2 button installs the external runtime.
- Segment buttons auto-install runtime_sam2 if missing.

v7.1 external SAM2 runtime installer fix:
- External SAM2 runtime no longer pip-installs hydra-core, omegaconf or iopath.
- Local hydra/omegaconf/iopath shims are used inside runtime_sam2.
- Installs setuptools/wheel first, then binary packages, then torch separately.
- Avoids the embeddable-Python error: Cannot import setuptools.build_meta.
- If a previous runtime_sam2 install is broken, delete the runtime_sam2 folder and run Install SAM2 again.

v7.2 SAM2 robust shim:
- Fixes `str object does not support item assignment` in local Hydra override handling.
- Hydra override paths now replace scalar collisions with DictConfig nodes.
- instantiate() ignores Hydra control kwargs like `_recursive_`.
- Segmenter prints full traceback to QGIS log for future SAM2 errors.
- Adds clean reinstall BAT: REINSTALL_EXTERNAL_SAM2_RUNTIME_CLEAN.bat.
- Runner detects old shim version and repairs runtime_sam2 automatically.

v7.3 SAM2 Hydra-free build:
- external_sam2_segmenter.py no longer imports or calls hydra.
- SAM2 YAML is loaded directly with PyYAML.
- `_target_` classes are instantiated directly with importlib.
- Checkpoint is loaded manually with torch.load / load_state_dict.
- This removes the entire Hydra/OmegaConf failure class from SAM2 segmentation.

v7.4 forced no-Hydra SAM2:
- mustatil_sam2_runner.py now always calls external_sam2_segmenter_nohydra.py.
- The no-Hydra segmenter monkey-patches/blockers hydra imports before SAM2 imports.
- It never imports sam2.build_sam or calls build_sam2().
- It logs FORCED_NO_HYDRA_SEGMENTER=1 and SAM2_BUILD_MODE:manual_no_hydra.
- Delete runtime_sam2 or run REINSTALL_EXTERNAL_SAM2_RUNTIME_CLEAN.bat after installing this version if older files remain.

v7.5 Ultralytics SAM runner:
- SAM2 integration now follows the working standalone Mustatil GUI path.
- Uses ultralytics.SAM(model).predict(..., bboxes=[...]).
- Does not use facebookresearch/sam2 source.
- Does not use Hydra, OmegaConf, SAM2 YAML configs, build_sam2 or sam2.build_sam.
- Writes image.sam2.json sidecars next to images.
- Annotator now reads image.sam2.json and displays cyan SAM polygons immediately after segmentation.
- Recommended model: sam2_b.pt in plugin weights/ folder, or let Ultralytics resolve/download it.

v7.6 SAM overlay fix:
- Annotator now robustly loads image.sam2.json sidecars.
- Supports {"polygons": [...]} and older list/dict JSON shapes.
- SAM polygons are drawn as cyan transparent overlays on the annotator preview.
- After SAM2 process finishes, annotator preview redraws automatically.
