# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Equivalence checks for the two hooks against diffusers' own code paths.

Run:  conda activate local-content-style && python tests/test_uniform_equivalence.py
(works under pytest too). Uses a 512^2 setup to stay fast; fp16 on MPS/CUDA.
"""

import sys
from pathlib import Path

import numpy as np
import torch
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel
from PIL import Image

from lcsc.ip_adapter import BlockMapIPProcessor
from lcsc.pipeline import TILE, load_pipeline

INPUTS = Path(__file__).resolve().parents[1] / "paper/Inputs"
CONTENT = INPUTS / "content/showcase/danila-1024.jpg"
STYLE = INPUTS / "styles/gleizes_paysage_cubiste_sq.png"
SIZE = 512
TOL = 5e-3

_ctx = {}


def ctx():
    if _ctx:
        return _ctx
    pipe, store = load_pipeline()
    store.image_size = (SIZE, SIZE)
    dev, dt = pipe.device, pipe.unet.dtype
    pe, _, pooled, _ = pipe.encode_prompt("a cubist painting", device=dev, num_images_per_prompt=1,
                                         do_classifier_free_guidance=False)
    style = Image.open(STYLE).convert("RGB")
    img_embeds = pipe.prepare_ip_adapter_image_embeds(style, None, dev, 1, False)
    time_ids = torch.tensor([[SIZE, SIZE, 0, 0, SIZE, SIZE]], device=dev, dtype=dt)

    def cond_tensor(path, flip=False):
        im = Image.open(path).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
        a = np.asarray(im, dtype=np.float32) / 255.0
        if flip:
            a = a[:, ::-1].copy()
        return torch.from_numpy(a).permute(2, 0, 1)[None].to(dev, dt)

    ct = torch.zeros(1, pipe.controlnet.config.num_control_type, device=dev, dtype=dt)
    ct[0, TILE] = 1
    _ctx.update(
        pipe=pipe, store=store, pe=pe, pooled=pooled, img_embeds=img_embeds, time_ids=time_ids,
        cond=cond_tensor(CONTENT), cond_other=cond_tensor(STYLE), ct=ct,
        latents=torch.randn(1, 4, SIZE // 8, SIZE // 8, generator=torch.Generator().manual_seed(0)).to(dev, dt),
        t=torch.tensor(999, device=dev),
        wrapped=dict(pipe.unet.attn_processors),
    )
    _ctx["stock"] = {k: (v.inner if isinstance(v, BlockMapIPProcessor) else v) for k, v in _ctx["wrapped"].items()}
    return _ctx


def _cn(c, cond, spatial=True):
    cn = c["pipe"].controlnet
    fwd = cn.forward if spatial else (lambda **kw: ControlNetUnionModel.forward(cn, **kw))
    down, mid = fwd(sample=c["latents"], timestep=c["t"], encoder_hidden_states=c["pe"],
                    controlnet_cond=[cond], control_type=c["ct"], control_type_idx=[TILE],
                    conditioning_scale=1.0,
                    added_cond_kwargs={"text_embeds": c["pooled"], "time_ids": c["time_ids"]},
                    return_dict=False)
    return torch.cat([x.flatten().float() for x in (*down, mid)])


def _unet(c):
    with torch.no_grad():
        return c["pipe"].unet(
            c["latents"], c["t"], encoder_hidden_states=c["pe"],
            added_cond_kwargs={"text_embeds": c["pooled"], "time_ids": c["time_ids"],
                               "image_embeds": c["img_embeds"]},
        ).sample.float()


def _stock_unet(c, scale):
    pipe = c["pipe"]
    pipe.unet.set_attn_processor(dict(c["stock"]))   # copies: set_attn_processor pops from the dict
    pipe.set_ip_adapter_scale(scale)
    out = _unet(c)
    pipe.set_ip_adapter_scale(1.0)               # restore inner scale for the wrappers
    pipe.unet.set_attn_processor(dict(c["wrapped"]))
    return out


def _report(name, a, b, tol=TOL, expect_equal=True):
    d = (a - b).abs()
    ok = (d.max().item() <= tol) if expect_equal else (d.max().item() > tol)
    print(f"{'PASS' if ok else 'FAIL'}  {name}: max|diff|={d.max().item():.3e}  mean={d.mean().item():.3e}")
    return ok


@torch.no_grad()
def test_controlnet_no_map_is_upstream():
    c = ctx(); c["store"].set_content(None)
    assert _report("ControlNet  s=None  == upstream forward", _cn(c, c["cond"]), _cn(c, c["cond"], spatial=False), tol=0.0)


@torch.no_grad()
def test_controlnet_uniform_one_is_upstream():
    c = ctx(); c["store"].set_content(1.0)
    assert _report("ControlNet  s(p)=1  == upstream, scale 1.0", _cn(c, c["cond"]), _cn(c, c["cond"], spatial=False))


@torch.no_grad()
def test_controlnet_zero_map_removes_control_image():
    c = ctx(); c["store"].set_content(0.0)
    assert _report("ControlNet  s(p)=0  -> control image has no effect", _cn(c, c["cond"]), _cn(c, c["cond_other"]))
    c["store"].set_content(1.0)


@torch.no_grad()
def test_ip_uniform_map_equals_scalar():
    c = ctx(); c["store"].set_style(0.6)
    assert _report("IP-Adapter  lambda_b(p)=0.6 all blocks == set_ip_adapter_scale(0.6)", _unet(c), _stock_unet(c, 0.6))


@torch.no_grad()
def test_ip_block6_map_equals_instantstyle():
    c = ctx(); c["store"].set_style(0.0); c["store"].set_style(1.0, blocks=[6])
    ref = _stock_unet(c, {"up": {"block_0": [0.0, 1.0, 0.0]}})
    assert _report("IP-Adapter  lambda_6=1 only == InstantStyle {up.block_0:[0,1,0]}", _unet(c), ref)


@torch.no_grad()
def test_ip_path_is_active():
    c = ctx()
    c["store"].set_style(0.6); a = _unet(c)
    c["store"].set_style(0.0); b = _unet(c)
    assert _report("IP-Adapter  lambda=0.6 vs 0 differ (path active)", a, b, tol=0.05, expect_equal=False)


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    results = []
    for t in tests:
        try:
            t(); results.append(True)
        except AssertionError:
            results.append(False)
    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)
