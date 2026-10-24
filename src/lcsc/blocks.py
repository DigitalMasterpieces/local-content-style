# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""The 11 IP-Adapter-injected SDXL attention blocks (Transformer2DModels), in the
index order used in the paper (b = 0..10)."""

# (diffusers module prefix, latent downsampling factor of the block's attention grid)
BLOCKS = [
    ("down_blocks.1.attentions.0", 2),  # b=0   64^2 at 1024^2
    ("down_blocks.1.attentions.1", 2),  # b=1
    ("down_blocks.2.attentions.0", 4),  # b=2   32^2
    ("down_blocks.2.attentions.1", 4),  # b=3
    ("mid_block.attentions.0", 4),      # b=4
    ("up_blocks.0.attentions.0", 4),    # b=5
    ("up_blocks.0.attentions.1", 4),    # b=6   InstantStyle's style block
    ("up_blocks.0.attentions.2", 4),    # b=7
    ("up_blocks.1.attentions.0", 2),    # b=8   64^2
    ("up_blocks.1.attentions.1", 2),    # b=9
    ("up_blocks.1.attentions.2", 2),    # b=10
]
NUM_BLOCKS = len(BLOCKS)


def block_index(processor_name: str):
    """Block index for an attention-processor name, or None if outside the 11 blocks."""
    for i, (prefix, _) in enumerate(BLOCKS):
        if processor_name.startswith(prefix + "."):
            return i
    return None
