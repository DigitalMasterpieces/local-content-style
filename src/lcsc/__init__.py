# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Local Content-Style Control: spatial ControlNet strength s(p) and per-block
IP-Adapter scale maps lambda_b(p) for diffusers SDXL pipelines."""

from .api import BLOCK_PRESETS, LocalContentStyle, Region
from .blocks import BLOCKS
from .maps import MapStore
from .controlnet import make_spatial_controlnet_union, install_spatial_controlnet
from .ip_adapter import BlockMapIPProcessor, install_spatial_ip_adapter

__all__ = [
    "LocalContentStyle",
    "Region",
    "BLOCK_PRESETS",
    "BLOCKS",
    "MapStore",
    "make_spatial_controlnet_union",
    "install_spatial_controlnet",
    "BlockMapIPProcessor",
    "install_spatial_ip_adapter",
]
