#!/usr/bin/env python3
"""Basic end-to-end smoke test for the Z-Image Turbo (fal.ai) backend.

Runs without torch, a GPU, or the notebook -- just the dependency-light client.
Makes two real, cheap API calls (text-to-image, then image-to-image on that
result) and saves the outputs so you can eyeball them.

Usage:
    # FAL_KEY from a .env at the repo root, or exported in your shell
    python scripts/smoke_test.py
    python scripts/smoke_test.py --prompt "a neon city street at night" --steps 8

Exits non-zero on failure so it can gate CI / pre-flight checks.
"""
import argparse
import os
import sys
import time

# Make `helpers` importable whether run from the repo root or elsewhere.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from helpers import zimage_client as zc  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default="a serene mountain lake at golden hour, painterly, sharp focus")
    parser.add_argument("--steps", type=int, default=8, help="1-8 (clamped by the client)")
    parser.add_argument("--size", type=int, default=512, help="square edge length")
    parser.add_argument("--strength", type=float, default=0.65,
                        help="Deforum convention: higher = more faithful to the init frame")
    parser.add_argument("--outdir", default=".", help="where to save smoke outputs")
    args = parser.parse_args()

    try:
        zc.resolve_fal_key()
    except RuntimeError as e:
        print(f"[smoke] {e}")
        return 1

    W = H = args.size
    os.makedirs(args.outdir, exist_ok=True)

    # 1) text-to-image
    t = time.time()
    t2i = zc.txt2img(args.prompt, W, H, steps=args.steps)[0]
    t2i_path = os.path.join(args.outdir, "smoke_txt2img.png")
    t2i.save(t2i_path)
    print(f"[smoke] txt2img OK  {t2i.size}  {time.time() - t:.1f}s  -> {t2i_path}")

    # 2) image-to-image on that result (exercises upload + strength inversion)
    fal_strength = zc.to_fal_strength(args.strength)
    t = time.time()
    i2i = zc.img2img(args.prompt + ", slightly more mist", t2i, args.strength, W, H, steps=args.steps)[0]
    i2i_path = os.path.join(args.outdir, "smoke_img2img.png")
    i2i.save(i2i_path)
    print(f"[smoke] img2img OK  {i2i.size}  {time.time() - t:.1f}s  "
          f"(deforum strength {args.strength} -> fal {fal_strength})  -> {i2i_path}")

    print("[smoke] PASS -- both paths work end-to-end. Open the two PNGs to eyeball quality.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
