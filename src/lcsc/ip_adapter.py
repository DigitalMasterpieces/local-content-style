# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Style axis: per-block, per-location IP-Adapter scale lambda_b(p) (paper Eq. 2).

The stock IPAdapterAttnProcessor2_0 already computes
    h + scale * (Attn(Q, K', V') * mask)
for `ip_adapter_masks`, but takes one (binarized-by-default) mask shared by all blocks.
BlockMapIPProcessor wraps each stock processor and feeds it *its own block's* continuous
map, already area-downsampled to the block's attention grid (so upstream's bicubic resize
is a same-size no-op and cannot overshoot at hard brush edges). Upstream math and weights
are reused unchanged.
"""

import torch
from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0

from .blocks import block_index


class BlockMapIPProcessor(torch.nn.Module):
    def __init__(self, inner: IPAdapterAttnProcessor2_0, block: int, store):
        super().__init__()
        self.inner = inner
        self.block = block
        self.store = store

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        ip_adapter_masks=None,
    ):
        lam = self.store.style_map(self.block, dtype=hidden_states.dtype, device=hidden_states.device)
        return self.inner(
            attn,
            hidden_states,
            encoder_hidden_states,
            attention_mask,
            temb,
            ip_adapter_masks=[lam],  # one IP image -> [1, 1, h_b, w_b]
        )


def install_spatial_ip_adapter(pipe, store):
    """Wrap the IP-Adapter processors of the 11 blocks. Call after load_ip_adapter(); do not
    call pipe.set_ip_adapter_scale() afterwards (the maps carry the scale)."""
    pipe.set_ip_adapter_scale(1.0)
    original = dict(pipe.unet.attn_processors)
    procs = {}
    for name, proc in original.items():
        b = block_index(name)
        if b is not None and isinstance(proc, IPAdapterAttnProcessor2_0):
            procs[name] = BlockMapIPProcessor(proc, b, store)
        else:
            procs[name] = proc
    pipe.unet.set_attn_processor(procs)
    return original  # pass back to pipe.unet.set_attn_processor() to uninstall
