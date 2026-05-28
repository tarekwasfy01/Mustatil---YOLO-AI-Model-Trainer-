#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import subprocess
import sys

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="")
    ap.add_argument("--model", default="sam2_b.pt")
    args = ap.parse_args()

    print("Installing/repairing Ultralytics SAM runtime dependencies.", flush=True)
    print("This path matches the working standalone Mustatil GUI: ultralytics.SAM(...).predict(...).", flush=True)

    # Keep it simple: no hydra, no facebookresearch/sam2 source.
    run([sys.executable, "-m", "pip", "install", "--upgrade", "--prefer-binary",
         "ultralytics", "torch", "torchvision", "opencv-python", "pillow", "numpy"], required=False)

    print("ULTRALYTICS_SAM_RUNTIME_READY=1", flush=True)
    print("Recommended model: sam2_b.pt or sam2_t.pt in plugin weights/ folder.", flush=True)

if __name__ == "__main__":
    main()
tifffile
