# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Render one of the paper's experiment manifests and compare against reference renders.

    python scripts/run_manifest.py paper/manifest_supplemental_s1.json

References are looked up as <condition>.{png,jpeg,jpg} in --reference, else in the manifest's
output directory (for paper/*.json: the expected PyTorch renders in paper/expected/). Prints the
mean |difference| to each reference and a locality check: per condition, the mean |difference|
to the first (baseline) condition inside vs. outside the style mask, for both.
"""

import argparse
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from lcsc import LocalContentStyle
from lcsc import manifest as mf

ROOT = Path(__file__).resolve().parents[1]


def find_reference(dirs, cond_id, size):
    for d in dirs:
        for ext in ("png", "jpeg", "jpg"):
            p = d / f"{cond_id}.{ext}"
            if p.exists():
                return Image.open(p).convert("RGB").resize(size, Image.BICUBIC)
    return None


def region_diffs(img, base, mask):
    """Mean |img - base| (0..1) where mask >= 0.5 and where mask < 0.5."""
    a = np.asarray(img, dtype=np.float32) / 255.0
    b = np.asarray(base, dtype=np.float32) / 255.0
    d = np.abs(a - b).mean(axis=2)
    inside = mask >= 0.5
    return d[inside].mean(), d[~inside].mean()


def sheet(grids, labels, cols, t):
    """Stack one grid per source (reference, this run), each `cols` wide."""
    n = max(len(g) for g in grids)
    rows = math.ceil(n / cols)
    cell = t + 18
    out = Image.new("RGB", (cols * t, len(grids) * (rows * cell + 24)), "white")
    draw = ImageDraw.Draw(out)
    for g, (grid, label) in enumerate(zip(grids, labels)):
        y0 = g * (rows * cell + 24)
        draw.text((6, y0 + 6), label, fill="black")
        for i, (name, im) in enumerate(grid):
            x, y = (i % cols) * t, y0 + 24 + (i // cols) * cell
            if im is not None:
                out.paste(im.resize((t, t)), (x, y + 18))
            draw.text((x + 6, y + 4), name, fill="black")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest")
    ap.add_argument("--reference", type=Path, help="directory with <condition>.{png,jpeg,jpg}")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--seeds", type=int, nargs="+", help="subset of the manifest's seeds")
    ap.add_argument("--styles", nargs="+", help="subset of the manifest's style ids")
    ap.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"),
                    help="compare (and report locality on) this box of the full frame")
    ap.add_argument("--cols", type=int, help="comparison grid width (default: all conditions in one row)")
    ap.add_argument("--tile", type=int, default=384, help="comparison cell size in pixels")
    ap.add_argument("--skip-existing", action="store_true", help="reuse renders already in the output directory")
    ap.add_argument("--offload", action="store_true", help="model CPU offload (less device memory, slower)")
    ap.add_argument("--label", default="expected", help="caption for the reference row of the comparison")
    ap.add_argument("--noise", choices=("torch", "numpy"), default="torch",
                    help="initial noise generator: torch, as diffusers (default), or numpy, as the paper's "
                         "figures and paper/expected")
    args = ap.parse_args()

    m = mf.load(args.manifest)
    size = (args.size, args.size)
    out_root = args.out or ROOT / "outputs" / Path(args.manifest).stem
    styles = [s for s in m["styles"] if not args.styles or s["id"] in args.styles]
    seeds = [s for s in m["seeds"] if not args.seeds or s in args.seeds]
    crop = (lambda im: im.crop(tuple(args.crop))) if args.crop else (lambda im: im)
    lcs = None  # loaded on the first render, so --skip-existing reruns stay fast

    for content in mf.contents(m):
        image = Image.open(mf.path(m, content["image"])).convert("RGB").resize(size, Image.BICUBIC)
        style_mask, tile_mask = mf.load_masks(m, content, size)
        mask = np.asarray(crop(Image.fromarray((style_mask.numpy() * 255).astype(np.uint8)))) / 255.0
        for style in styles:
            style_image = Image.open(mf.path(m, style["image"])).convert("RGB")
            for seed in seeds:
                out = out_root / content["id"] / style["id"] / f"seed-{seed}"
                out.mkdir(parents=True, exist_ok=True)
                ref_dirs = ([args.reference] if args.reference else []) + [mf.run_dir(m, content, style, seed)]
                ours, refs = [], []
                for cond in m["conditions"]:
                    png = out / f"{cond['id']}.png"
                    if args.skip_existing and png.exists():
                        img = Image.open(png).convert("RGB")
                    else:
                        lcs = lcs or LocalContentStyle.load(offload=args.offload)
                        img = lcs(style["prompt"], image, style_image, seed=seed, steps=m["stepCount"],
                                  width=size[0], height=size[1], noise=args.noise,
                                  **mf.condition_kwargs(m, cond, style_mask, tile_mask))
                        img.save(png)
                        print("rendered", content["id"], style["id"], seed, cond["id"], flush=True)
                    ours.append((cond["id"], crop(img)))
                    ref = find_reference(ref_dirs, cond["id"], size)
                    refs.append((cond["id"], ref and crop(ref)))

                cols = args.cols or len(ours)
                sheet([refs, ours], [args.label, "this run"], cols, args.tile).save(out / "comparison.png")
                print("wrote", out / "comparison.png")

                diffs = [np.abs(np.asarray(o, np.float32) - np.asarray(r, np.float32)).mean()
                         for (_, o), (_, r) in zip(ours, refs) if r is not None]
                if diffs:
                    print(f"mean|this run - {args.label}| per condition (0..255): "
                          + " ".join(f"{d:.1f}" for d in diffs))

                print(f"locality vs '{ours[0][0]}': mean|diff| inside (mask white) / outside (mask black)")
                for (cid, o), (_, r) in list(zip(ours, refs))[1:]:
                    oi, oo = region_diffs(o, ours[0][1], mask)
                    line = f"  {cid:10s} this run {oi:.3f} / {oo:.3f}"
                    if r is not None and refs[0][1] is not None:
                        ri, ro = region_diffs(r, refs[0][1], mask)
                        line += f"   {args.label} {ri:.3f} / {ro:.3f}"
                    print(line)


if __name__ == "__main__":
    main()
