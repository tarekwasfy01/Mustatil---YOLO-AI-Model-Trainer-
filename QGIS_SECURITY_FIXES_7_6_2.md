# QGIS security fixes for 7.6.2

This package addresses the reported QGIS/Bandit audit findings:

- `external_sam2_segmenter.py`: checkpoint loading now uses `torch.load(..., weights_only=True)` where supported, with a compatibility fallback for older PyTorch builds.
- `install_external_sam2_runtime.py`: runtime downloads validate URL schemes and reject non-http/non-https URLs before calling `urlretrieve`.
- `mustatil_sam2_runtime_setup.py`: runtime downloads validate URL schemes and reject non-http/non-https URLs before calling `urlretrieve`.

The package still contains the required top-level `LICENSE`, `metadata.txt`, and `__init__.py` files and excludes compiled Python cache files.
