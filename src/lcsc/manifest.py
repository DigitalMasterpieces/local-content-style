# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Read the paper's experiment manifests (paper/*.json) and turn a condition into LocalContentStyle arguments.

Semantics follow the renderer that produced the paper's figures: masks are remapped as
outside + (inside - outside) * m with white = inside; the Tile map uses `contentMask` when
given (unless `tileUsesStyleMask`); per-block style maps multiply the block's base scale,
and blocks with base 0 stay off.
"""

import json
from pathlib import Path

from .blocks import NUM_BLOCKS
from .maps import load_mask, remap


def load(path):
    m = json.loads(Path(path).read_text())
    m["_dir"] = Path(path).resolve().parent
    return m


def contents(m):
    return m["contents"] if "contents" in m else [m["content"]]


def path(m, rel):
    return m["_dir"] / rel


def run_dir(m, content, style, seed):
    """The manifest's output directory for one (content, style, seed)."""
    root = path(m, m["output"])
    if "contents" in m:
        root = root / content["id"]
    return root / style["id"] / f"seed-{seed}"


def load_masks(m, content, size):
    """(style mask, tile mask) as [H, W] in [0, 1]; the tile mask falls back to the style mask."""
    style_mask = load_mask(path(m, content["mask"]), size)
    tile_mask = load_mask(path(m, content["contentMask"]), size) if "contentMask" in content else style_mask
    return style_mask, tile_mask


def condition_kwargs(m, cond, style_mask, tile_mask):
    """Keyword arguments for LocalContentStyle.__call__ reproducing one manifest condition."""
    if "featherSigma" in cond:
        raise NotImplementedError("featherSigma conditions are not ported")
    d = m["defaults"]
    base = cond.get("ipAdapterBlockScales") or d.get("ipAdapterBlockScales") or [d["ipAdapterScale"]] * NUM_BLOCKS

    if "tileInside" in cond or "tileOutside" in cond:
        region = style_mask if cond.get("tileUsesStyleMask") else tile_mask
        content = remap(region, cond.get("tileOutside", d["tileStrength"]), cond.get("tileInside", d["tileStrength"]))
    else:
        content = d["tileStrength"]

    shared = remap(style_mask, d["ipAdapterScale"], cond["styleInside"]) if "styleInside" in cond else None
    block_maps = {b["index"]: remap(style_mask, b["outside"], b["inside"]) for b in cond.get("ipAdapterBlockMaps", [])}
    active = [b for b in range(NUM_BLOCKS) if base[b] != 0]
    per_block = {}
    for b in active:
        mp = block_maps.get(b, shared)
        per_block[b] = base[b] if mp is None else mp * base[b]
    return {"content_strength": content, "blocks": active, "block_strength": per_block}
