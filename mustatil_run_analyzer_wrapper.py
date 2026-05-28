#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Install a completely separate SAM2 runtime for MustatilQGIS.

v7.1 fix:
- Does NOT pip-install hydra-core / omegaconf / iopath.
- Uses local hydra/omegaconf/iopath shims instead.
- Installs pip/setuptools/wheel first in a separate step.
- Then installs binary deps.
- Then installs torch/torchvision separately.
This avoids the Windows embeddable Python build error:
  BackendUnavailable: Cannot import 'setuptools.build_meta'
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path
from urllib.parse import urlparse
import requests

PYTHON_VERSION = "3.12.10"
PYTHON_EMBED_URL = f"https://www.python.org/ftp/python/{PYTHON_VERSION}/python-{PYTHON_VERSION}-embed-amd64.zip"
GET_PIP_URL = "https://bootstrap.pypa.io/get-pip.py"
SAM2_ZIP_URL = "https://github.com/facebookresearch/sam2/archive/refs/heads/main.zip"

CHECKPOINTS = {
    "tiny": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "small": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "base_plus": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "large": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}
CFG = {
    "tiny": "sam2_hiera_t.yaml",
    "small": "sam2_hiera_s.yaml",
    "base_plus": "sam2_hiera_b+.yaml",
    "large": "sam2_hiera_l.yaml",
}

OMEGACONF_SHIM = """
import copy
import ast
import yaml

class DictConfig(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
    def __setattr__(self, name, value):
        self[name] = value

def _wrap(x):
    if isinstance(x, dict):
        return DictConfig({k: _wrap(v) for k, v in x.items()})
    if isinstance(x, list):
        return [_wrap(v) for v in x]
    return x

def _to_plain(x):
    if isinstance(x, dict):
        return {k: _to_plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_plain(v) for v in x]
    return x

class OmegaConf:
    @staticmethod
    def load(path):
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return _wrap(data)

    @staticmethod
    def create(obj):
        return _wrap(copy.deepcopy(obj))

    @staticmethod
    def to_container(obj, resolve=True):
        return _to_plain(obj)

    @staticmethod
    def resolve(obj):
        return obj

    @staticmethod
    def merge(*configs):
        def merge_two(a, b):
            a = _to_plain(a)
            b = _to_plain(b)
            if isinstance(a, dict) and isinstance(b, dict):
                out = dict(a)
                for k, v in b.items():
                    out[k] = merge_two(out.get(k), v) if k in out else v
                return out
            return b
        out = {}
        for cfg in configs:
            out = merge_two(out, cfg)
        return _wrap(out)

    @staticmethod
    def set_struct(obj, flag):
        return None

    @staticmethod
    def select(cfg, key, default=None):
        cur = cfg
        for part in key.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur
"""

HYDRA_INIT_SHIM = """
from pathlib import Path
import ast
from omegaconf import OmegaConf, DictConfig

_CONFIG_BASE = None

def initialize_config_module(*args, **kwargs):
    return None

def initialize_config_dir(config_dir=None, *args, **kwargs):
    global _CONFIG_BASE
    _CONFIG_BASE = config_dir
    return None

def _parse_value(v):
    if isinstance(v, str):
        s = v.strip()
        if s in ("true", "True"):
            return True
        if s in ("false", "False"):
            return False
        if s in ("null", "None", "~"):
            return None
        try:
            return ast.literal_eval(s)
        except Exception:
            return s
    return v

def _ensure_container(parent, key):
    existing = None
    if isinstance(parent, dict):
        existing = parent.get(key)
    if not isinstance(existing, dict):
        if isinstance(parent, dict):
            parent[key] = DictConfig()
            return parent[key]
        raise TypeError("Cannot create nested config node")
    return existing

def _set_nested(cfg, key, value):
    if key.startswith("++"):
        key = key[2:]
    if key.startswith("+"):
        key = key[1:]

    # Hydra supports deletion (~key). Ignore deletions for SAM2 compatibility.
    if key.startswith("~"):
        return

    parts = [p for p in key.split(".") if p]
    if not parts:
        return

    cur = cfg
    for p in parts[:-1]:
        if not isinstance(cur, dict):
            # Override path collided with scalar; replace with container.
            cur = DictConfig()
        if p not in cur or not isinstance(cur.get(p), dict):
            cur[p] = DictConfig()
        cur = cur[p]

    if not isinstance(cur, dict):
        return
    cur[parts[-1]] = _parse_value(value)

def compose(config_name=None, overrides=None, *args, **kwargs):
    path = Path(config_name)
    if not path.exists() and _CONFIG_BASE:
        path = Path(_CONFIG_BASE) / config_name
    cfg = OmegaConf.load(path)

    # SAM2 config files may include Hydra defaults; they are not needed here.
    if isinstance(cfg, dict) and "defaults" in cfg:
        try:
            del cfg["defaults"]
        except Exception:
            pass

    for ov in overrides or []:
        if not isinstance(ov, str):
            continue
        if "=" in ov:
            k, v = ov.split("=", 1)
            _set_nested(cfg, k, v)
    return cfg
"""

HYDRA_UTILS_SHIM = """
import importlib
from omegaconf import OmegaConf

def _plain(x):
    return OmegaConf.to_container(x, resolve=True)

def _locate(target):
    module_name, name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, name)

def instantiate(cfg, *args, **kwargs):
    cfg = _plain(cfg)

    # Hydra control kwargs; not constructor args.
    kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

    if isinstance(cfg, list):
        return [instantiate(x) for x in cfg]
    if not isinstance(cfg, dict):
        return cfg

    if "_target_" not in cfg:
        return {k: instantiate(v) for k, v in cfg.items()}

    target = cfg.get("_target_")
    cls = _locate(target)

    params = {}
    for k, v in cfg.items():
        if k.startswith("_"):
            continue
        params[k] = instantiate(v)

    params.update(kwargs)
    return cls(**params)
"""

HYDRA_GLOBAL_HYDRA_SHIM = """
class GlobalHydra:
    _instance = None
    def __init__(self):
        self._initialized = False
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = GlobalHydra()
        return cls._instance
    def is_initialized(self):
        return self._initialized
    def clear(self):
        self._initialized = False
    def initialize(self, *args, **kwargs):
        self._initialized = True
        return None
"""

HYDRA_CONFIG_STORE_SHIM = """
class ConfigStore:
    _instance = None
    def __init__(self):
        self.items = []
    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = ConfigStore()
        return cls._instance
    def store(self, *args, **kwargs):
        self.items.append((args, kwargs))
        return None
"""

IOPATH_SHIM = """
import os
class _PathManager:
    def open(self, path, mode="r", *args, **kwargs):
        return open(path, mode, *args, **kwargs)
    def exists(self, path):
        return os.path.exists(path)
    def isfile(self, path):
        return os.path.isfile(path)
    def isdir(self, path):
        return os.path.isdir(path)
    def mkdirs(self, path):
        os.makedirs(path, exist_ok=True)
    def get_local_path(self, path, *args, **kwargs):
        return str(path)
    def copy(self, src_path, dst_path, overwrite=False, **kwargs):
        import shutil
        if overwrite or not os.path.exists(dst_path):
            shutil.copy2(src_path, dst_path)
        return True
g_pathmgr = _PathManager()
PathManager = _PathManager
"""

def _validate_download_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        raise ValueError(f"Refusing non-HTTPS download URL scheme: {parsed.scheme!r}")
    allowed_hosts = {
        "www.python.org",
        "bootstrap.pypa.io",
        "github.com",
        "codeload.github.com",
        "dl.fbaipublicfiles.com",
        "files.pythonhosted.org",
        "pypi.org",
    }
    host = (parsed.hostname or "").lower()
    if not host or host not in allowed_hosts:
        raise ValueError(f"Refusing download from unapproved host: {host!r}")

def download(url: str, out: Path):
    _validate_download_url(url)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and out.stat().st_size > 100:
        print(f"Already present: {out}", flush=True)
        return
    print(f"Downloading: {url}", flush=True)
    tmp = out.with_suffix(out.suffix + ".part")
    with requests.get(url, stream=True, timeout=(10, 60), allow_redirects=True) as response:
        response.raise_for_status()
        _validate_download_url(response.url)
        with open(tmp, "wb") as fh:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    fh.write(chunk)
    tmp.replace(out)

def safe_extract_zip(zip_file: Path, destination: Path) -> None:
    destination = destination.resolve()
    with zipfile.ZipFile(zip_file, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if not str(target).startswith(str(destination) + os.sep) and target != destination:
                raise RuntimeError(f"Blocked unsafe ZIP member path: {member.filename}")
        archive.extractall(destination)

def run(cmd, required=True, cwd=None):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    try:
        subprocess.check_call(list(map(str, cmd)), cwd=str(cwd) if cwd else None)
        return True
    except Exception as exc:
        print(f"Command failed: {exc}", flush=True)
        if required:
            raise
        return False

def write_shims(root: Path):
    shim_root = root / "sam2_shims"
    hydra_pkg = shim_root / "hydra"
    hydra_core = hydra_pkg / "core"
    omega_pkg = shim_root / "omegaconf"
    iopath_common = shim_root / "iopath" / "common"

    hydra_core.mkdir(parents=True, exist_ok=True)
    omega_pkg.mkdir(parents=True, exist_ok=True)
    iopath_common.mkdir(parents=True, exist_ok=True)

    (omega_pkg / "__init__.py").write_text(OMEGACONF_SHIM.strip() + "\n", encoding="utf-8")
    (hydra_pkg / "__init__.py").write_text(HYDRA_INIT_SHIM.strip() + "\n", encoding="utf-8")
    (hydra_pkg / "utils.py").write_text(HYDRA_UTILS_SHIM.strip() + "\n", encoding="utf-8")
    (hydra_core / "__init__.py").write_text("", encoding="utf-8")
    (hydra_core / "global_hydra.py").write_text(HYDRA_GLOBAL_HYDRA_SHIM.strip() + "\n", encoding="utf-8")
    (hydra_core / "config_store.py").write_text(HYDRA_CONFIG_STORE_SHIM.strip() + "\n", encoding="utf-8")

    (shim_root / "iopath" / "__init__.py").write_text("", encoding="utf-8")
    (iopath_common / "__init__.py").write_text("", encoding="utf-8")
    (iopath_common / "file_io.py").write_text(IOPATH_SHIM.strip() + "\n", encoding="utf-8")

    (root / "sam2_shim_path.txt").write_text(str(shim_root), encoding="utf-8")
    (root / "sam2_shim_version_v72.txt").write_text("v72", encoding="utf-8")
    print(f"External SAM2 shims ready: {shim_root}", flush=True)

def ensure_embed_python(root: Path) -> Path:
    py_dir = root / "python"
    py_exe = py_dir / "python.exe"
    if py_exe.exists():
        print(f"External SAM2 Python already exists: {py_exe}", flush=True)
        return py_exe

    py_zip = root / f"python-{PYTHON_VERSION}-embed-amd64.zip"
    download(PYTHON_EMBED_URL, py_zip)

    py_dir.mkdir(parents=True, exist_ok=True)
    print(f"Extracting Python embed runtime to: {py_dir}", flush=True)
    safe_extract_zip(py_zip, py_dir)

    for pth in py_dir.glob("python*._pth"):
        txt = pth.read_text(encoding="utf-8", errors="ignore")
        txt = txt.replace("#import site", "import site")
        if "Lib\\site-packages" not in txt and "Lib/site-packages" not in txt:
            txt += "\nLib\\site-packages\n"
        pth.write_text(txt, encoding="utf-8")

    get_pip = root / "get-pip.py"
    download(GET_PIP_URL, get_pip)
    run([py_exe, str(get_pip)])
    return py_exe

def install_deps(py_exe: Path):
    # Step 1: toolchain first.
    run([py_exe, "-m", "pip", "install", "--upgrade", "setuptools", "wheel"], required=False)

    # Step 2: normal binary packages. Do not install hydra/omegaconf/iopath.
    run([py_exe, "-m", "pip", "install", "--upgrade", "--prefer-binary",
         "numpy", "pillow", "opencv-python", "pyyaml", "tqdm"], required=True)

    # Step 3: torch stack isolated from QGIS. If this fails, log but still allow setup to finish.
    run([py_exe, "-m", "pip", "install", "--upgrade", "--prefer-binary",
         "torch", "torchvision"], required=False)

def extract_sam2(root: Path) -> Path:
    zip_path = root / "sam2_main.zip"
    download(SAM2_ZIP_URL, zip_path)

    src_root = root / "sam2_source"
    if src_root.exists():
        shutil.rmtree(src_root, ignore_errors=True)
    src_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting SAM2 source: {zip_path}", flush=True)
    safe_extract_zip(zip_path, src_root)

    candidates = [p for p in src_root.iterdir() if p.is_dir() and (p / "sam2").exists()]
    if not candidates:
        raise RuntimeError("SAM2 source folder was not found after extraction.")

    src = candidates[0]
    (root / "sam2_source_path.txt").write_text(str(src), encoding="utf-8")
    print(f"SAM2 source path: {src}", flush=True)
    return src

def download_checkpoint(root: Path, model: str, src: Path):
    ckpt_dir = root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt = ckpt_dir / Path(CHECKPOINTS[model]).name
    download(CHECKPOINTS[model], ckpt)

    cfg_name = CFG[model]
    matches = list(src.rglob(cfg_name))
    cfg_path = matches[0] if matches else Path(cfg_name)

    (root / "sam2_checkpoint_path.txt").write_text(str(ckpt), encoding="utf-8")
    (root / "sam2_config_name.txt").write_text(cfg_name, encoding="utf-8")
    (root / "sam2_config_path.txt").write_text(str(cfg_path), encoding="utf-8")

    print(f"SAM2 checkpoint: {ckpt}", flush=True)
    print(f"SAM2 config: {cfg_path}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--model", choices=list(CHECKPOINTS), default="base_plus")
    ap.add_argument("--skip-deps", action="store_true")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    if args.clean and root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    write_shims(root)
    py_exe = ensure_embed_python(root)
    if not args.skip_deps:
        install_deps(py_exe)

    src = extract_sam2(root)
    download_checkpoint(root, args.model, src)

    print("EXTERNAL_SAM2_RUNTIME_READY=1", flush=True)
    print(f"python={py_exe}", flush=True)
    print(f"runtime_root={root}", flush=True)

if __name__ == "__main__":
    main()
tifffile
