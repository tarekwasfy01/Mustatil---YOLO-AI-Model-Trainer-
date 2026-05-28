#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib
import json
import sys
import traceback
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}

def list_images(path):
    p = Path(path)
    if p.is_file():
        return [p]
    return [x for x in sorted(p.rglob("*")) if x.suffix.lower() in IMG_EXT]

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""

def add_runtime_paths(root: Path):
    # SAM2 source first. No hydra import required.
    src = read_text(root / "sam2_source_path.txt")
    if src and Path(src).exists() and src not in sys.path:
        sys.path.insert(0, src)

def resolve_checkpoint(root: Path, checkpoint: str) -> str:
    if checkpoint and Path(checkpoint).exists():
        return checkpoint
    p = read_text(root / "sam2_checkpoint_path.txt")
    return p

def resolve_config(root: Path, cfg: str) -> str:
    if cfg and Path(cfg).exists():
        return cfg
    p = read_text(root / "sam2_config_path.txt")
    if p and Path(p).exists():
        return p
    return cfg or read_text(root / "sam2_config_name.txt")

def write_manifest(out, data):
    out.mkdir(parents=True, exist_ok=True)
    (out / "sam2_manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

def _plain(x):
    # Keep config plain Python types.
    if isinstance(x, dict):
        return {k: _plain(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_plain(v) for v in x]
    return x

def _locate(target: str):
    module_name, name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, name)

def _instantiate_no_hydra(obj):
    """
    Minimal Hydra-free instantiation for SAM2 YAML configs.

    It supports the SAM2 convention:
      _target_: module.ClassName
      key: value
    Recursively instantiates nested _target_ dictionaries.
    """
    obj = _plain(obj)

    if isinstance(obj, list):
        return [_instantiate_no_hydra(v) for v in obj]

    if not isinstance(obj, dict):
        return obj

    # Drop Hydra-only fields.
    obj = {k: v for k, v in obj.items() if k not in {"_recursive_", "_convert_", "_partial_"}}

    if "_target_" not in obj:
        return {k: _instantiate_no_hydra(v) for k, v in obj.items()}

    target = obj["_target_"]
    cls = _locate(target)

    kwargs = {}
    for k, v in obj.items():
        if k.startswith("_"):
            continue
        kwargs[k] = _instantiate_no_hydra(v)

    return cls(**kwargs)

def _apply_override(cfg: dict, override: str):
    """
    Apply a tiny subset of Hydra override syntax used by SAM2 build_sam2.
    """
    if not isinstance(override, str) or "=" not in override:
        return
    key, value = override.split("=", 1)
    key = key.lstrip("+")
    if key.startswith("~"):
        return

    def parse(v):
        s = v.strip()
        if s in ("true", "True"):
            return True
        if s in ("false", "False"):
            return False
        if s in ("null", "None", "~"):
            return None
        try:
            import ast
            return ast.literal_eval(s)
        except Exception:
            return s

    parts = [p for p in key.split(".") if p]
    cur = cfg
    for p in parts[:-1]:
        if p not in cur or not isinstance(cur[p], dict):
            cur[p] = {}
        cur = cur[p]
    if parts:
        cur[parts[-1]] = parse(value)

def _load_yaml(path: str) -> dict:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise RuntimeError(f"SAM2 config is not a dictionary: {path}")
    return data

def _load_checkpoint_no_hydra(model, ckpt_path: str, device: str):
    import torch
    ckpt = Path(ckpt_path)
    if ckpt.suffix.lower() not in {".pt", ".pth"}:
        raise RuntimeError(f"Refusing unsupported checkpoint extension: {ckpt.suffix}")
    if not ckpt.exists() or ckpt.stat().st_size < 1024 * 1024:
        raise RuntimeError(f"Refusing missing or suspiciously small checkpoint: {ckpt}")
    try:
        # Required for safer checkpoint deserialization. This avoids arbitrary object unpickling on supported PyTorch builds.
        sd = torch.load(str(ckpt), map_location=device, weights_only=True)  # nosec B614
    except TypeError as exc:
        raise RuntimeError("This runtime uses an older PyTorch without weights_only=True. Update the external SAM2 runtime and retry.") from exc

    # SAM2 checkpoints usually use {"model": state_dict}.
    if isinstance(sd, dict) and "model" in sd:
        state = sd["model"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        state = sd["state_dict"]
    else:
        state = sd

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"SAM2_CHECKPOINT_LOADED missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    if missing:
        print("SAM2_CHECKPOINT_MISSING_KEYS:" + ",".join(list(missing)[:20]), flush=True)
    if unexpected:
        print("SAM2_CHECKPOINT_UNEXPECTED_KEYS:" + ",".join(list(unexpected)[:20]), flush=True)
    return model

def build_sam2_without_hydra(cfg_path: str, checkpoint: str, device: str):
    """
    Direct SAM2 builder that avoids hydra and omegaconf completely.
    """
    cfg = _load_yaml(cfg_path)

    # SAM2 YAMLs define the model under "model".
    if "model" not in cfg:
        raise RuntimeError(f"SAM2 config has no 'model' section: {cfg_path}")

    # Equivalent of common build_sam2 postprocessing flags.
    # These are safe defaults and can be ignored by classes that do not use them.
    # Do NOT require Hydra/OmegaConf.
    model_cfg = cfg["model"]

    model = _instantiate_no_hydra(model_cfg)
    model = model.to(device)
    model.eval()

    _load_checkpoint_no_hydra(model, checkpoint, device)
    return model

def segment(images, out, root, checkpoint, cfg):
    add_runtime_paths(root)

    checkpoint = resolve_checkpoint(root, checkpoint)
    cfg = resolve_config(root, cfg)

    if not checkpoint or not Path(checkpoint).exists():
        raise RuntimeError(f"SAM2 checkpoint missing: {checkpoint}")
    if not cfg or not Path(cfg).exists():
        raise RuntimeError(f"SAM2 config missing or not found: {cfg}")

    try:
        import numpy as np
        from PIL import Image
        import cv2
        import torch
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    except Exception as exc:
        raise RuntimeError(
            "External SAM2 imports failed. The isolated runtime probably did not finish installing dependencies. "
            "Run REINSTALL_EXTERNAL_SAM2_RUNTIME_CLEAN.bat or press Install SAM2 again. "
            f"Import error: {exc}"
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SAM2_DEVICE:{device}", flush=True)
    print("SAM2_BUILD:hydra_free", flush=True)

    model = build_sam2_without_hydra(cfg, checkpoint, device)
    generator = SAM2AutomaticMaskGenerator(model)

    outputs = []
    for idx, img_path in enumerate(images, start=1):
        print(f"SAM2_IMAGE:{idx}/{len(images)} {img_path.name}", flush=True)
        img = np.array(Image.open(img_path).convert("RGB"))
        masks = generator.generate(img)

        img_out = out / img_path.stem
        img_out.mkdir(parents=True, exist_ok=True)

        infos = []
        for i, m in enumerate(masks):
            seg = m.get("segmentation")
            if seg is None:
                continue
            mask_path = img_out / f"mask_{i:04d}.png"
            cv2.imwrite(str(mask_path), (seg.astype("uint8") * 255))
            infos.append({
                "mask": str(mask_path),
                "area": int(m.get("area", 0)),
                "bbox": m.get("bbox", []),
                "score": float(m.get("predicted_iou", 0.0)),
            })
        print(f"SAM2_MASKS:{img_path.name}:{len(infos)}", flush=True)
        outputs.append({"image": str(img_path), "count": len(infos), "masks": infos})
    return outputs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-root", required=True)
    ap.add_argument("--images", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--mode", choices=["one", "all"], default="one")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--model-cfg", default="")
    args = ap.parse_args()

    root = Path(args.runtime_root)
    out = Path(args.output)
    images = list_images(args.images)
    if args.mode == "one" and images:
        images = images[:1]

    manifest = {
        "runtime_root": str(root),
        "mode": args.mode,
        "images": [str(p) for p in images],
        "real_sam2": False,
        "hydra_free": True,
        "outputs": [],
    }

    try:
        manifest["outputs"] = segment(images, out, root, args.checkpoint, args.model_cfg)
        manifest["real_sam2"] = True
        print("SAM2 segmentation finished.", flush=True)
    except Exception as exc:
        manifest["error"] = str(exc)
        manifest["traceback"] = traceback.format_exc()
        print("SAM2 segmentation failed.", flush=True)
        traceback.print_exc()
        print(str(exc), flush=True)

    write_manifest(out, manifest)
    print(f"Output: {out}", flush=True)

if __name__ == "__main__":
    main()
