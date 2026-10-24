# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Content map s(p) and per-block style maps lambda_b(p).

Convention (matches the paper's experiment manifests): a region mask m(p) in [0, 1]
is remapped as  value(p) = outside + (inside - outside) * m(p),  i.e. "inside" is where
the mask is white. Maps are kept at pixel resolution and anti-alias (area) downsampled
to each modulation point: latent resolution for the ControlNet fusion, the block's
attention grid for each IP-Adapter block.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .blocks import BLOCKS, NUM_BLOCKS


def load_mask(path, size=None) -> torch.Tensor:
    """Grayscale mask -> float tensor [H, W] in [0, 1] (white = 1)."""
    im = Image.open(path).convert("L")
    if size is not None:
        im = im.resize(size, Image.BILINEAR)
    return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)


def remap(mask: torch.Tensor, outside: float, inside: float) -> torch.Tensor:
    return outside + (inside - outside) * mask


def _area_resize(m: torch.Tensor, hw, dtype, device) -> torch.Tensor:
    """[H, W] -> [1, 1, h, w] with an area (box) filter."""
    x = m.to(device=device, dtype=torch.float32)[None, None]
    if tuple(x.shape[-2:]) != tuple(hw):
        x = F.interpolate(x, size=tuple(hw), mode="area")
    return x.to(dtype)


class MapStore:
    """Holds the current s(p) and lambda_b(p); the hooks read from it at every call.

    content: None (no spatial ControlNet modulation), a float (uniform), or a [H, W] map.
    style:   per-block entries, each None (block keeps the uniform default), a float, or a [H, W] map.
    """

    def __init__(self, content=1.0, style_default=1.0, image_size=(1024, 1024)):
        self.image_size = tuple(image_size)  # (H, W) of the generated image
        self.content = content
        self.style = [style_default] * NUM_BLOCKS

    # -- setters ---------------------------------------------------------------
    def set_content(self, value):
        self.content = value

    def set_style(self, value, blocks=None):
        for b in range(NUM_BLOCKS) if blocks is None else blocks:
            self.style[b] = value

    # -- readers used by the hooks ---------------------------------------------
    def content_map(self, latent_hw, dtype, device):
        """[1, 1, h, w] at the ControlNet fusion (latent) resolution, or None."""
        if self.content is None:
            return None
        if isinstance(self.content, (int, float)):
            return torch.full((1, 1, *latent_hw), float(self.content), dtype=dtype, device=device)
        return _area_resize(self.content, latent_hw, dtype, device)

    def style_map(self, b, dtype=torch.float32, device="cpu"):
        """[1, 1, h_b, w_b] at block b's attention grid."""
        H, W = self.image_size
        f = 8 * BLOCKS[b][1]
        hw = (H // f, W // f)
        v = self.style[b]
        if isinstance(v, (int, float)):
            return torch.full((1, 1, *hw), float(v), dtype=dtype, device=device)
        return _area_resize(v, hw, dtype, device)
