# QGIS Security Hardening 7.6.3

Additional hardening for QGIS plugin review:

- Downloads are restricted to HTTPS only.
- Download hosts are explicitly allowlisted.
- Downloads use streamed requests with timeouts and atomic `.part` files.
- ZIP extraction now validates every archive member to block path traversal / zip-slip writes.
- SAM2 checkpoint loading now requires PyTorch `weights_only=True`; insecure fallback loading was removed.
- SAM2 checkpoints must use `.pt` or `.pth` and must be larger than 1 MB.
- Runtime installation is launched through the current Python interpreter instead of `cmd.exe /c`.
- Plugin structure, `LICENSE`, metadata and cache cleanup were rechecked.
