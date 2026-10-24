# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0

"""Gradio demo: retouch content and style region by region.

    python demo/app.py [--share] [--port 7860] [--offload]

Examples use the paper's inputs in paper/Inputs.
"""

import argparse
import random
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageOps

from lcsc import BLOCK_PRESETS, LocalContentStyle, Region

ROOT = Path(__file__).resolve().parents[1]
PAPER = "https://doi.org/10.1145/3829339.3847814"
OFFLOAD = False
_lcs = None


def model():
    global _lcs
    if _lcs is None:
        _lcs = LocalContentStyle.load(offload=OFFLOAD)
    return _lcs


def sdxl_size(w, h, area=1024 * 1024):
    """Closest SDXL-friendly size (multiples of 64, about one megapixel) with the image's aspect."""
    scale = (area / (w * h)) ** 0.5
    return max(512, round(w * scale / 64) * 64), max(512, round(h * scale / 64) * 64)


def region_mask(editor, upload):
    """White-on-black region from an uploaded mask, else from the brushed layers; None if empty."""
    if upload is not None:
        return upload.convert("L")
    layers = (editor or {}).get("layers") or []
    if not layers:
        return None
    alpha = np.zeros(layers[0].size[::-1], np.uint8)
    for layer in layers:
        alpha = np.maximum(alpha, np.asarray(layer.convert("RGBA"))[..., 3])
    return Image.fromarray(np.where(alpha > 0, 255, 0).astype(np.uint8)) if alpha.any() else None


def strength(mask, inside, outside):
    return outside if mask is None else Region(mask, inside=inside, outside=outside)


def render(prompt, content, style, style_editor, style_upload, content_editor, content_upload,
           s_in, s_out, l_in, l_out, blocks, seed, randomize, baseline):
    if content is None or style is None:
        raise gr.Error("Add a content image and a style exemplar.")
    seed = random.randint(0, 2**32 - 1) if randomize else int(seed)
    style_mask = region_mask(style_editor, style_upload)
    content_mask = region_mask(content_editor, content_upload) or style_mask
    w, h = sdxl_size(*content.size)
    common = dict(blocks=blocks, seed=seed, width=w, height=h)

    lcs = model()
    images = [(lcs(prompt, content, style, content_strength=strength(content_mask, s_in, s_out),
                   style_strength=strength(style_mask, l_in, l_out), **common), "local maps")]
    if baseline:
        images.append((lcs(prompt, content, style, content_strength=s_out, style_strength=l_out, **common),
                       "uniform (outside values everywhere)"))
    return images, seed


def as_background(image):
    return None if image is None else {"background": image, "layers": [], "composite": image}


def examples():
    """Paper cases expressible with one content and one style map; masks inverted so white = the retouched region."""
    inp = ROOT / "paper/Inputs"
    if not inp.exists():
        return []
    cache = ROOT / "outputs/demo_examples"
    cache.mkdir(parents=True, exist_ok=True)

    def inverted(rel):
        out = cache / Path(rel).with_suffix(".png").name
        if not out.exists():
            ImageOps.invert(Image.open(inp / rel).convert("L")).save(out)
        return str(out)

    teaser = "masks/showcase/danila-perevoshchikov-77p2DAx00EA-unsplash-mask"
    return [
        # Fig. 1 CS: the retouched subject keeps more of the photo (higher s, lower lambda)
        ["a young woman holding a bouquet of yellow flowers, cubist painting, fragmented forms, Albert Gleizes inspired",
         str(inp / "content/showcase/danila-1024.jpg"), str(inp / "styles/gleizes_paysage_cubiste_sq.png"),
         inverted(teaser + ".jpeg"), None, 0.7, 0.35, 0.3, 1.0, "all", 1447777030],
        # Fig. 4 left: depth drives the style block, exemplar fades up close
        ["watercolor style", str(inp / "content/showcase/depth.jpg"), str(inp / "styles/homer_sloop_nassau.png"),
         str(inp / "masks/showcase/depth.jpeg"), None, 0.5, 0.5, 1.0, 0.0, "style", 590662726],
        # Fig. S2 cell s70_l25: the retouched face held at s=0.7 with a light exemplar
        ["A young woman with glasses, Kandinsky style.", str(inp / "content/supplemental/woman-glasses.png"),
         str(inp / "styles/kandinsky_black_grid.png"), inverted("masks/supplemental/woman-glasses.png"), None,
         0.7, 0.5, 0.25, 1.0, "all", 2907713393],
    ]


def build():
    brush = gr.Brush(colors=[("#ff3b30", 0.6)], color_mode="fixed", default_size=40)
    with gr.Blocks(title="Local Content-Style Control") as demo:
        gr.Markdown(
            "# Local Content-Style Control\n"
            f"Brush a region to retouch, then set how much **content** (ControlNet Tile strength *s*) and **style** "
            f"(IP-Adapter scale *λ*) it gets inside and outside. SIGGRAPH Asia 2026 Technical Communications "
            f"([paper]({PAPER}))."
        )
        with gr.Row():
            with gr.Column():
                content = gr.Image(type="pil", label="Content image")
                style = gr.Image(type="pil", label="Style exemplar")
                prompt = gr.Textbox(label="Prompt", placeholder="describe the image and the style")
            with gr.Column():
                with gr.Tab("Style region"):
                    style_editor = gr.ImageEditor(type="pil", brush=brush, sources=(), label="Brush the region")
                    style_upload = gr.Image(type="pil", image_mode="L", label="…or upload a mask (white = region)")
                with gr.Tab("Content region (optional)"):
                    gr.Markdown("Leave empty to reuse the style region.")
                    content_editor = gr.ImageEditor(type="pil", brush=brush, sources=(), label="Brush the region")
                    content_upload = gr.Image(type="pil", image_mode="L", label="…or upload a mask (white = region)")
                with gr.Row():
                    s_in = gr.Slider(0, 1.5, 0.7, step=0.05, label="content s · inside")
                    s_out = gr.Slider(0, 1.5, 0.35, step=0.05, label="content s · outside")
                with gr.Row():
                    l_in = gr.Slider(0, 1.5, 0.3, step=0.05, label="style λ · inside")
                    l_out = gr.Slider(0, 1.5, 1.0, step=0.05, label="style λ · outside")
                blocks = gr.Radio(list(BLOCK_PRESETS), value="all", label="IP-Adapter blocks")
                with gr.Row():
                    seed = gr.Number(1447777030, precision=0, label="seed")
                    randomize = gr.Checkbox(False, label="random seed")
                    baseline = gr.Checkbox(False, label="also render the uniform baseline")
                run = gr.Button("Render", variant="primary")
            with gr.Column():
                result = gr.Gallery(label="Result", columns=1, height="auto", format="png")

        content.change(as_background, content, style_editor)
        content.change(as_background, content, content_editor)
        run.click(render, [prompt, content, style, style_editor, style_upload, content_editor, content_upload,
                           s_in, s_out, l_in, l_out, blocks, seed, randomize, baseline], [result, seed])
        if ex := examples():
            gr.Examples(ex, [prompt, content, style, style_upload, content_upload,
                             s_in, s_out, l_in, l_out, blocks, seed], label="Paper examples")
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--offload", action="store_true", help="model CPU offload for GPUs with less memory")
    args = ap.parse_args()
    OFFLOAD = args.offload
    build().queue().launch(server_port=args.port, share=args.share)
