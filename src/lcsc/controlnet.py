# Copyright 2026 The HuggingFace Team. All rights reserved.
# Copyright 2026 Digital Masterpieces GmbH
# SPDX-License-Identifier: Apache-2.0
#
# SpatialControlNetUnionModel.forward is adapted from diffusers 0.41.0
# (models/controlnets/controlnet_union.py). Modified by Digital Masterpieces GmbH:
# the content map s(p) scales the control embedding at the fusion stage and the
# ControlNet output residuals; changes are marked "LCSC".

"""Content axis: spatial ControlNet strength s(p) (paper Eq. 1, h~ = h + s(p) c).

The map gates the *conditioning embedding at the input fusion*, not the output residuals.
For a vanilla ControlNetModel a forward hook on `controlnet_cond_embedding` is enough.
For ControlNet Union (what the paper used) the fusion also has a per-type bias `alpha`
and a pooled transformer path, so the forward is vendored from diffusers 0.41.0 with only
the fusion stage replaced, as in the paper's Core ML pipeline:
    pooled token   <- feat * mean(s)
    condition      <- condition * s
    fuser         += condition + alpha * s
and, with a single active control type, every output residual is also multiplied by s
resampled to its resolution (the paper pipeline's spatial generalization of diffusers'
single-control output scale). Call the pipeline with controlnet_conditioning_scale=1.0 so
the upstream output scaling is the identity; the strength lives entirely in the map.
"""

import torch
import torch.nn.functional as F
from diffusers.models.controlnets.controlnet import ControlNetOutput
from diffusers.models.controlnets.controlnet_union import ControlNetUnionModel
from diffusers.utils.torch_utils import maybe_adjust_dtype_for_device


def install_spatial_controlnet(controlnet, store):
    """Vanilla ControlNetModel: scale the conditioning embedding by s(p). Returns the hook handle."""

    def hook(module, args, c):
        s = store.content_map(c.shape[-2:], c.dtype, c.device)
        return c if s is None else c * s

    return controlnet.controlnet_cond_embedding.register_forward_hook(hook)


def make_spatial_controlnet_union(controlnet: ControlNetUnionModel, store):
    """Turn a loaded ControlNetUnionModel into the spatial variant in place (no weight reload)."""
    controlnet.__class__ = SpatialControlNetUnionModel
    controlnet.lcsc_store = store
    return controlnet


class SpatialControlNetUnionModel(ControlNetUnionModel):
    lcsc_store = None

    # Vendored from diffusers 0.41.0 models/controlnets/controlnet_union.py; changes are
    # marked "LCSC". Re-vendor when bumping diffusers.
    def forward(
        self,
        sample,
        timestep,
        encoder_hidden_states,
        controlnet_cond,
        control_type,
        control_type_idx,
        conditioning_scale=1.0,
        class_labels=None,
        timestep_cond=None,
        attention_mask=None,
        added_cond_kwargs=None,
        cross_attention_kwargs=None,
        from_multi=False,
        guess_mode=False,
        return_dict=True,
    ):
        if isinstance(conditioning_scale, float):
            conditioning_scale = [conditioning_scale] * len(controlnet_cond)

        channel_order = self.config.controlnet_conditioning_channel_order
        if channel_order != "rgb":
            raise ValueError(f"unknown `controlnet_conditioning_channel_order`: {channel_order}")

        if attention_mask is not None:
            attention_mask = (1 - attention_mask.to(sample.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            dtype = maybe_adjust_dtype_for_device(
                torch.float64 if isinstance(timestep, float) else torch.int64, sample.device
            )
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)
        timesteps = timesteps.expand(sample.shape[0])

        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=sample.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)
        aug_emb = None

        if self.class_embedding is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            if self.config.class_embed_type == "timestep":
                class_labels = self.time_proj(class_labels)
            class_emb = self.class_embedding(class_labels).to(dtype=self.dtype)
            emb = emb + class_emb

        if self.config.addition_embed_type is not None:
            if self.config.addition_embed_type == "text":
                aug_emb = self.add_embedding(encoder_hidden_states)
            elif self.config.addition_embed_type == "text_time":
                text_embeds = added_cond_kwargs.get("text_embeds")
                time_ids = added_cond_kwargs.get("time_ids")
                time_embeds = self.add_time_proj(time_ids.flatten())
                time_embeds = time_embeds.reshape((text_embeds.shape[0], -1))
                add_embeds = torch.concat([text_embeds, time_embeds], dim=-1)
                add_embeds = add_embeds.to(emb.dtype)
                aug_emb = self.add_embedding(add_embeds)

        control_embeds = self.control_type_proj(control_type.flatten())
        control_embeds = control_embeds.reshape((t_emb.shape[0], -1))
        control_embeds = control_embeds.to(emb.dtype)
        control_emb = self.control_add_embedding(control_embeds)
        emb = emb + control_emb
        emb = emb + aug_emb if aug_emb is not None else emb

        # 2. pre-process
        sample = self.conv_in(sample)

        # LCSC: the content map s(p) at fusion (latent) resolution; one map for all active types.
        store = self.lcsc_store
        s = None if store is None else store.content_map(sample.shape[-2:], sample.dtype, sample.device)

        inputs = []
        condition_list = []

        for cond, control_idx, scale in zip(controlnet_cond, control_type_idx, conditioning_scale):
            condition = self.controlnet_cond_embedding(cond)
            feat_seq = torch.mean(condition, dim=(2, 3))
            feat_seq = feat_seq + self.task_embedding[control_idx]
            if s is not None:  # LCSC: spatial fusion
                inputs.append(feat_seq.unsqueeze(1) * s.mean())
                condition_list.append(condition * s)
            elif from_multi or len(control_type_idx) == 1:
                inputs.append(feat_seq.unsqueeze(1))
                condition_list.append(condition)
            else:
                inputs.append(feat_seq.unsqueeze(1) * scale)
                condition_list.append(condition * scale)

        condition = sample
        feat_seq = torch.mean(condition, dim=(2, 3))
        inputs.append(feat_seq.unsqueeze(1))
        condition_list.append(condition)

        x = torch.cat(inputs, dim=1)
        for layer in self.transformer_layes:
            x = layer(x)

        controlnet_cond_fuser = sample * 0.0
        for (idx, condition), scale in zip(enumerate(condition_list[:-1]), conditioning_scale):
            alpha = self.spatial_ch_projs(x[:, idx])
            alpha = alpha.unsqueeze(-1).unsqueeze(-1)
            if s is not None:  # LCSC: h~ = h + s(p) * (condition + alpha)
                controlnet_cond_fuser += condition + alpha * s
            elif from_multi or len(control_type_idx) == 1:
                controlnet_cond_fuser += condition + alpha
            else:
                controlnet_cond_fuser += condition + alpha * scale

        sample = sample + controlnet_cond_fuser

        # 3. down
        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=attention_mask,
                    cross_attention_kwargs=cross_attention_kwargs,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)
            down_block_res_samples += res_samples

        # 4. mid
        if self.mid_block is not None:
            sample = self.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=attention_mask,
                cross_attention_kwargs=cross_attention_kwargs,
            )

        # 5. Control net blocks
        controlnet_down_block_res_samples = ()
        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples = controlnet_down_block_res_samples + (down_block_res_sample,)
        down_block_res_samples = controlnet_down_block_res_samples
        mid_block_res_sample = self.controlnet_mid_block(sample)

        # 6. scaling (upstream; identity with conditioning_scale=1.0)
        if guess_mode and not self.config.global_pool_conditions:
            scales = torch.logspace(-1, 0, len(down_block_res_samples) + 1, device=sample.device)
            if from_multi or len(control_type_idx) == 1:
                scales = scales * conditioning_scale[0]
            down_block_res_samples = [sample * scale for sample, scale in zip(down_block_res_samples, scales)]
            mid_block_res_sample = mid_block_res_sample * scales[-1]
        elif from_multi or len(control_type_idx) == 1:
            down_block_res_samples = [sample * conditioning_scale[0] for sample in down_block_res_samples]
            mid_block_res_sample = mid_block_res_sample * conditioning_scale[0]

        # LCSC: with a single active control, also scale each residual by s(p) resampled to its
        # own resolution -- the spatial form of the output scale above, as in the paper's
        # Core ML pipeline.
        if s is not None and (from_multi or len(control_type_idx) == 1):
            def _scaled(r):
                return r * (s if r.shape[-2:] == s.shape[-2:] else F.interpolate(s, size=r.shape[-2:], mode="area"))
            down_block_res_samples = [_scaled(r) for r in down_block_res_samples]
            mid_block_res_sample = _scaled(mid_block_res_sample)

        if self.config.global_pool_conditions:
            down_block_res_samples = [
                torch.mean(sample, dim=(2, 3), keepdim=True) for sample in down_block_res_samples
            ]
            mid_block_res_sample = torch.mean(mid_block_res_sample, dim=(2, 3), keepdim=True)

        if not return_dict:
            return (down_block_res_samples, mid_block_res_sample)
        return ControlNetOutput(
            down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
        )
