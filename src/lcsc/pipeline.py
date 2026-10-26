# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""The paper's model stack in diffusers: SDXL-Lightning 4-step UNet, ControlNet Union ProMax
(Tile), IP-Adapter Plus ViT-H, fp16-fix VAE, 4 trailing Euler steps, no CFG."""

import numpy as np
import torch
from accelerate import cpu_offload_with_hook, init_empty_weights
from diffusers import (
    AutoencoderKL,
    ControlNetUnionModel,
    EulerDiscreteScheduler,
    StableDiffusionXLControlNetUnionPipeline,
    UNet2DConditionModel,
)
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from .controlnet import make_spatial_controlnet_union
from .ip_adapter import install_spatial_ip_adapter
from .maps import MapStore

BASE = "stabilityai/stable-diffusion-xl-base-1.0"   # text encoders, tokenizers, configs only
LIGHTNING = ("ByteDance/SDXL-Lightning", "sdxl_lightning_4step_unet.safetensors")
# The original ProMax release; brad-twinkl/controlnet-union-sdxl-1.0-promax (used for the paper's
# Core ML conversion) is a byte-identical re-upload (sha256 9fae2e50..., same config).
CONTROLNET = ("xinsir/controlnet-union-sdxl-1.0", "config_promax.json", "diffusion_pytorch_model_promax.safetensors")
VAE = "madebyollin/sdxl-vae-fp16-fix"
IP_ADAPTER = ("h94/IP-Adapter", "sdxl_models", "ip-adapter-plus_sdxl_vit-h.safetensors")
TILE = 6  # ControlNet Union ProMax control mode for Tile


def default_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _from_checkpoint(cls, config, weights, dtype):
    """Build `cls` without allocating random weights, then take the checkpoint's tensors as they are:
    peak RAM stays near the (memory-mapped, fp16) checkpoint instead of a full fp32 model on top."""
    with init_empty_weights():
        model = cls.from_config(config)
    model.load_state_dict(load_file(weights), assign=True)
    return model.to(dtype)


ENCODERS = ("text_encoder", "text_encoder_2", "image_encoder")


def load_pipeline(device=None, dtype=torch.float16, store=None, offload=False):
    """offload controls where the weights live between uses:
    False      everything on the device (about 11.6 GB at fp16);
    "encoders" UNet, ControlNet and VAE stay on the device, the two text encoders and the image
               encoder (about 3 GB) move there only while they run: about 8.7 GB on the device and
               little extra system RAM, e.g. for 16 GB GPUs such as Colab's T4;
    True/"all" accelerate's model CPU offload: least device memory, but all weights in system RAM."""
    device = device or default_device()
    unet = _from_checkpoint(UNet2DConditionModel, UNet2DConditionModel.load_config(BASE, subfolder="unet"),
                            hf_hub_download(*LIGHTNING), dtype)
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=dtype)
    controlnet = _from_checkpoint(ControlNetUnionModel,
                                  ControlNetUnionModel.load_config(hf_hub_download(CONTROLNET[0], CONTROLNET[1])),
                                  hf_hub_download(CONTROLNET[0], CONTROLNET[2]), dtype)
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        IP_ADAPTER[0], subfolder="models/image_encoder", torch_dtype=dtype
    )
    # Build from components (not from_pretrained(BASE)): only SDXL base's text encoders,
    # tokenizers and scheduler config are needed, never its UNet/VAE weights.
    pipe = StableDiffusionXLControlNetUnionPipeline(
        vae=vae,
        text_encoder=CLIPTextModel.from_pretrained(BASE, subfolder="text_encoder", variant="fp16", torch_dtype=dtype),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
            BASE, subfolder="text_encoder_2", variant="fp16", torch_dtype=dtype),
        tokenizer=CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer"),
        tokenizer_2=CLIPTokenizer.from_pretrained(BASE, subfolder="tokenizer_2"),
        unet=unet,
        controlnet=controlnet,
        scheduler=EulerDiscreteScheduler.from_pretrained(BASE, subfolder="scheduler", timestep_spacing="trailing"),
        add_watermarker=False,
        feature_extractor=CLIPImageProcessor(),
        image_encoder=image_encoder,
    )
    pipe.load_ip_adapter(IP_ADAPTER[0], subfolder=IP_ADAPTER[1], weight_name=IP_ADAPTER[2],
                         image_encoder_folder=None)

    store = store or MapStore()
    # Install the hooks before device placement: accelerate's offload wrappers capture each
    # module's forward when they are attached.
    make_spatial_controlnet_union(pipe.controlnet, store)
    install_spatial_ip_adapter(pipe, store)
    if offload == "encoders":
        for name, module in pipe.components.items():
            if isinstance(module, torch.nn.Module) and name not in ENCODERS:
                module.to(device)
        hook = None
        for name in ENCODERS:  # each encoder moves to the device when it runs, the previous one back
            _, hook = cpu_offload_with_hook(getattr(pipe, name), device, prev_module_hook=hook)
    elif offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)
    return pipe, store


def numpy_latents(seed, size, dtype):
    """Initial noise as NumPy's legacy RandomState(seed).randn, the generator the paper's figures were
    rendered with. Unscaled; the pipeline multiplies by init_noise_sigma as for any supplied latents."""
    shape = (1, 4, size[0] // 8, size[1] // 8)
    return torch.from_numpy(np.random.RandomState(seed).randn(*shape).astype(np.float32)).to(dtype)


def generate(pipe, prompt, content_image, style_image, seed, steps=4, size=(1024, 1024), noise="torch"):
    """One generation with whatever maps are currently in the pipe's MapStore.

    noise="torch" draws the initial noise with diffusers' torch CPU generator; noise="numpy" draws it
    as the paper's figures did, so the paper's seeds give the paper's compositions."""
    g = torch.Generator("cpu").manual_seed(seed)
    latents = numpy_latents(seed, size, pipe.unet.dtype) if noise == "numpy" else None
    return pipe(
        prompt=prompt,
        control_image=[content_image],
        control_mode=[TILE],
        ip_adapter_image=style_image,
        controlnet_conditioning_scale=1.0,  # strength lives in s(p)
        num_inference_steps=steps,
        guidance_scale=0.0,                 # no CFG (SDXL-Lightning)
        height=size[0],
        width=size[1],
        generator=g,
        latents=latents,
    ).images[0]
