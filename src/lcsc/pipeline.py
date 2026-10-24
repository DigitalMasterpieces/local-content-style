# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""The paper's model stack in diffusers: SDXL-Lightning 4-step UNet, ControlNet Union ProMax
(Tile), IP-Adapter Plus ViT-H, fp16-fix VAE, 4 trailing Euler steps, no CFG."""

import torch
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


def load_pipeline(device=None, dtype=torch.float16, store=None, offload=False):
    """offload=True keeps each model on the CPU until it runs (accelerate's model CPU offload):
    slower, but peak device memory drops to the largest single model."""
    device = device or default_device()
    unet = UNet2DConditionModel.from_config(UNet2DConditionModel.load_config(BASE, subfolder="unet"))
    unet.load_state_dict(load_file(hf_hub_download(*LIGHTNING)))
    unet = unet.to(dtype)
    vae = AutoencoderKL.from_pretrained(VAE, torch_dtype=dtype)
    controlnet = ControlNetUnionModel.from_config(
        ControlNetUnionModel.load_config(hf_hub_download(CONTROLNET[0], CONTROLNET[1])))
    controlnet.load_state_dict(load_file(hf_hub_download(CONTROLNET[0], CONTROLNET[2])))
    controlnet = controlnet.to(dtype)
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
    if offload:
        pipe.enable_model_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)
    return pipe, store


def generate(pipe, prompt, content_image, style_image, seed, steps=4, size=(1024, 1024)):
    """One generation with whatever maps are currently in the pipe's MapStore."""
    g = torch.Generator("cpu").manual_seed(seed)
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
    ).images[0]
