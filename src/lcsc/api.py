# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Public entry point.

    from lcsc import LocalContentStyle, Region

    lcs = LocalContentStyle.load()
    image = lcs(
        "a young woman holding yellow flowers, cubist painting",
        content_image="photo.jpg",
        style_image="exemplar.jpg",
        content_strength=Region("subject.png", inside=0.7, outside=0.35),  # s(p)
        style_strength=Region("subject.png", inside=0.3, outside=1.0),     # lambda_b(p)
        seed=1,
    )

Strengths are a float (uniform), a map (image path, PIL image, 2-D array or tensor with
values in [0, 1] for 8-bit images, any non-negative float otherwise), or a Region that
remaps a mask: outside + (inside - outside) * mask, with white = inside.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .blocks import NUM_BLOCKS
from .maps import remap
from .pipeline import generate, load_pipeline

# Active IP-Adapter blocks (indices into blocks.BLOCKS) for the paper's operating points.
BLOCK_PRESETS = {
    "all": tuple(range(NUM_BLOCKS)),  # every block (Fig. 1, Fig. S2)
    "style": (6,),                    # InstantStyle's style block only (Fig. 4 left)
    "mid+style": (4, 6),              # mid block + style block (Fig. 4 middle, Fig. S1)
}


@dataclass(frozen=True)
class Region:
    """A mask remapped to strengths: `inside` where the mask is white, `outside` where black."""
    mask: object
    inside: float
    outside: float


def as_map(value, size):
    """float -> float; mask-like or Region -> [H, W] float tensor at size=(width, height)."""
    if value is None or isinstance(value, (int, float)):
        return value
    if isinstance(value, Region):
        return remap(_mask(value.mask, size), value.outside, value.inside)
    return _mask(value, size)


def _mask(value, size):
    w, h = size
    if isinstance(value, (str, Path)):
        value = Image.open(value)
    if isinstance(value, Image.Image):
        value = np.asarray(value.convert("L").resize((w, h), Image.BILINEAR))
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value.astype(np.float32) / 255.0 if value.dtype == np.uint8 else value.astype(np.float32))
    if not isinstance(value, torch.Tensor) or value.ndim != 2:
        raise TypeError("a map must be a path, PIL image, or 2-D array/tensor")
    value = value.float().cpu()
    if tuple(value.shape) != (h, w):
        value = F.interpolate(value[None, None], size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    return value


def _image(value):
    return (Image.open(value) if isinstance(value, (str, Path)) else value).convert("RGB")


class LocalContentStyle:
    """SDXL stylization with a spatial content map s(p) and per-block style maps lambda_b(p)."""

    def __init__(self, pipe, store):
        self.pipe = pipe
        self.store = store

    @classmethod
    def load(cls, device=None, dtype=torch.float16, offload=False):
        """offload=True: model CPU offload for GPUs with less memory (slower)."""
        return cls(*load_pipeline(device=device, dtype=dtype, offload=offload))

    @torch.no_grad()
    def __call__(self, prompt, content_image, style_image, *, content_strength=0.5, style_strength=1.0,
                 blocks="all", block_strength=None, seed=0, steps=4, width=1024, height=1024, noise="torch"):
        """Stylize `content_image` after `style_image`.

        content_strength: ControlNet Tile strength s(p); None disables the map (upstream behavior).
        style_strength:   IP-Adapter scale lambda(p) for every active block.
        blocks:           a BLOCK_PRESETS name or block indices; inactive blocks get no style.
        block_strength:   {block index: strength} overrides style_strength per active block.
        noise:            initial noise generator, "torch" (diffusers) or "numpy" (the paper's figures).
        """
        size = (width, height)
        active = BLOCK_PRESETS[blocks] if isinstance(blocks, str) else tuple(blocks)
        overrides = dict(block_strength or {})
        if not set(overrides) <= set(active):
            raise ValueError(f"block_strength sets inactive blocks {sorted(set(overrides) - set(active))}")

        self.store.image_size = (height, width)
        self.store.set_content(as_map(content_strength, size))
        self.store.set_style(0.0)
        for b in active:
            self.store.set_style(as_map(overrides.get(b, style_strength), size), blocks=[b])

        content = _image(content_image).resize(size, Image.BICUBIC)
        return generate(self.pipe, prompt, content, _image(style_image), seed, steps=steps, size=(height, width),
                        noise=noise)
