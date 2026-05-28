#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse
import requests
import zipfile
import shutil

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

OMEGACONF_SHIM = '''
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
'''

HYDRA_INIT_SHIM = '''
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

def _set_nested(cfg, key, value):
    if key.startswith("++"):
        key = key[2:]
    if key.startswith("+"):
        key = key[1:]
    parts = key.split(".")
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or cur[p] is None:
            cur[p] = DictConfig()
        cur = cur[p]
    cur[parts[-1]] = _parse_value(value)

def compose(config_name=None, overrides=None, *args, **kwargs):
    path = Path(config_name)
    if not path.exists() and _CONFIG_BASE:
        path = Path(_CONFIG_BASE) / config_name
    cfg = OmegaConf.load(path)
    for ov in overrides or []:
        if "=" in ov:
            k, v = ov.split("=", 1)
            _set_nested(cfg, k, v)
    return cfg
'''

HYDRA_UTILS_SHIM = '''
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
'''


HYDRA_CORE_INIT_SHIM = """
""".strip() + "\n"

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
""".strip() + "\n"

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
""".strip() + "\n"

HYDRA_UTILS_PKG_SHIM = """
from . import instantiate
""".strip() + "\n"

IOPATH_SHIM = '''
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
'''

def run(cmd, required=False):
    print("$ " + " ".join(map(str, cmd)), flush=True)
    try:
        subprocess.check_call(list(map(str, cmd)))
        return True
    except Exception as exc:
        print(f"Command failed: {exc}", flush=True)
        if required:
            raise
        return False

def pip_install(packages, required=False):
    return run([sys.executable, "-m", "pip", "install", "--upgrade", "--prefer-binary"] + list(packages), required=required)

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

def write_shims(dest: Path):
    shim_root = dest / "sam2_shims"
    hydra_pkg = shim_root / "hydra"
    hydra_core_pkg = shim_root / "hydra" / "core"
    omega_pkg = shim_root / "omegaconf"
    iopath_pkg = shim_root / "iopath" / "common"
    hydra_pkg.mkdir(parents=True, exist_ok=True)
    hydra_core_pkg.mkdir(parents=True, exist_ok=True)
    omega_pkg.mkdir(parents=True, exist_ok=True)
    iopath_pkg.mkdir(parents=True, exist_ok=True)

    (omega_pkg / "__init__.py").write_text(OMEGACONF_SHIM.strip() + "\n", encoding="utf-8")
    (hydra_pkg / "__init__.py").write_text(HYDRA_INIT_SHIM.strip() + "\n", encoding="utf-8")
    (hydra_pkg / "utils.py").write_text(HYDRA_UTILS_SHIM.strip() + "\n", encoding="utf-8")
    (hydra_core_pkg / "__init__.py").write_text(HYDRA_CORE_INIT_SHIM, encoding="utf-8")
    (hydra_core_pkg / "global_hydra.py").write_text(HYDRA_GLOBAL_HYDRA_SHIM, encoding="utf-8")
    (hydra_core_pkg / "config_store.py").write_text(HYDRA_CONFIG_STORE_SHIM, encoding="utf-8")

    (shim_root / "iopath" / "__init__.py").write_text("", encoding="utf-8")
    (iopath_pkg / "__init__.py").write_text("", encoding="utf-8")
    (iopath_pkg / "file_io.py").write_text(IOPATH_SHIM.strip() + "\n", encoding="utf-8")

    (dest / "sam2_shim_path.txt").write_text(str(shim_root), encoding="utf-8")
    print(f"SAM2 local shims ready: {shim_root}", flush=True)

def extract_sam2_source(dest: Path) -> Path:
    zip_path = dest / "sam2_main.zip"
    download(SAM2_ZIP_URL, zip_path)

    src_root = dest / "sam2_source"
    if src_root.exists():
        shutil.rmtree(src_root, ignore_errors=True)
    src_root.mkdir(parents=True, exist_ok=True)

    print(f"Extracting SAM2 source: {zip_path}", flush=True)
    safe_extract_zip(zip_path, src_root)

    candidates = [p for p in src_root.iterdir() if p.is_dir() and (p / "sam2").exists()]
    if not candidates:
        raise RuntimeError(f"SAM2 extraction failed. No extracted folder contains sam2/: {src_root}")

    source_dir = candidates[0]
    (dest / "sam2_source_path.txt").write_text(str(source_dir), encoding="utf-8")
    print(f"SAM2 source path: {source_dir}", flush=True)
    return source_dir

def setup(dest: Path, model: str):
    dest.mkdir(parents=True, exist_ok=True)

    print("Preparing SAM2 local shims and runtime dependencies...", flush=True)
    write_shims(dest)
    pip_install(["tqdm", "opencv-python", "pillow", "numpy", "pyyaml"], required=False)

    source_dir = extract_sam2_source(dest)

    ckpt = dest / Path(CHECKPOINTS[model]).name
    download(CHECKPOINTS[model], ckpt)

    cfg_name = CFG[model]
    (dest / "sam2_config.txt").write_text(cfg_name, encoding="utf-8")
    matches = list(source_dir.rglob(cfg_name))
    if matches:
        (dest / "sam2_config_path.txt").write_text(str(matches[0]), encoding="utf-8")
        print(f"SAM2 config path: {matches[0]}", flush=True)
    else:
        print(f"SAM2 config file name stored: {cfg_name}", flush=True)

    (dest / "sam2_checkpoint_path.txt").write_text(str(ckpt), encoding="utf-8")
    print("SAM2_RUNTIME_READY=1", flush=True)
    print(f"checkpoint={ckpt}", flush=True)
    print(f"config={cfg_name}", flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", required=True)
    ap.add_argument("--model", choices=list(CHECKPOINTS), default="tiny")
    args = ap.parse_args()
    setup(Path(args.dest), args.model)

if __name__ == "__main__":
    main()
