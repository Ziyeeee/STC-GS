import os
from typing import Tuple
import math
from functools import partial
import json
import copy
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import Tensor

torch.set_float32_matmul_precision("high")
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import trunc_normal_
from torch.optim import Optimizer, AdamW
from torch.utils.checkpoint import checkpoint
from transformers import get_cosine_schedule_with_warmup
from lightning import LightningModule

from mamba_ssm.modules.mamba_simple import Mamba
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import MHA
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.modules.block import Block
from mamba_ssm.utils.generation import GenerationMixin
from mamba_ssm.utils.hf import load_config_hf, load_state_dict_hf

from diff_gaussian_rasterization_radar import GaussianRasterizationSettings, GaussianRasterizer
from utils.metrics import Evaluator
from datasets.stcgs_dataset import GaussiansDataset


try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


def convert_str_color_to_rgb(color_str):
    color_str = color_str[1:]
    return tuple(int(color_str[i : i + 2], 16) / 255.0 for i in [0, 2, 4])


COLOR_MAP = [
    "#FFFFFF",
    "#FFFFFF",
    "#23ECEB",
    "#13A0DC",
    "#0029F0",
    "#2BFC3E",
    "#22C531",
    "#188E23",
    "#FFFC42",
    "#E5BD33",
    "#FD8D2A",
    "#FC0019",
    "#A30011",
    "#63000A",
    "#FC28F9",
    "#9559C3",
    "#FFFFFF",
    "#FFFFFF",
]
COLOR_MAP = np.array([convert_str_color_to_rgb(color) for color in COLOR_MAP])

PIXEL_SCALE = 80
BOUNDS = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, PIXEL_SCALE])


def normalize_to_rgb(array):
    array_scaled = array * PIXEL_SCALE
    indices = np.digitize(array_scaled, BOUNDS)
    return COLOR_MAP[indices]


def create_block(
    d_model,
    d_intermediate,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    norm_epsilon=1e-5,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    if layer_idx not in attn_layer_idx:
        # Create a copy of the config to modify
        ssm_cfg = copy.deepcopy(ssm_cfg) if ssm_cfg is not None else {}
        ssm_layer = ssm_cfg.pop("layer", "Mamba1")
        if ssm_layer not in ["Mamba1", "Mamba2"]:
            raise ValueError(f"Invalid ssm_layer: {ssm_layer}, only support Mamba1 and Mamba2")
        mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(MHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(GatedMLP, hidden_features=d_intermediate, out_features=d_model, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


class MixerModel(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm=False,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32

        # We change the order of residual and layer norm:
        # Instead of LN -> Attn / MLP -> Add, we do:
        # Add -> LN -> Attn / MLP / Mixer, returning both the residual branch (output of Add) and
        # the main branch (output of MLP / Mixer). The model definition is unchanged.
        # This is for performance reason: we can fuse add + layer_norm.
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Failed to import Triton LayerNorm / RMSNorm kernels")
        self.layers = nn.ModuleList(
            [
                create_block(
                    d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.linears_z = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model * 2, d_model, **factory_kwargs), nn.LayerNorm(d_model), nn.Sigmoid())
                for _ in range(n_layer // 2)
            ]
        )

        self.linears_r = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d_model * 2, d_model, **factory_kwargs), nn.LayerNorm(d_model), nn.Sigmoid())
                for _ in range(n_layer // 2)
            ]
        )

        self.linears_h = nn.ModuleList([nn.Linear(d_model * 2, d_model, **factory_kwargs) for _ in range(n_layer // 2)])

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(d_model, eps=norm_epsilon, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,  # 2 if we have MLP
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, hidden_states, pre_hidden_states_list=None, inference_params=None, **mixer_kwargs):
        residual = None
        hidden_states_list = []
        for i in range(len(self.layers) // 2):
            if pre_hidden_states_list is not None:
                pre_hidden_states = pre_hidden_states_list[i]
            else:
                pre_hidden_states = torch.zeros_like(hidden_states)

            gate_embed = torch.cat([hidden_states, pre_hidden_states], dim=-1)
            z = self.linears_z[i](gate_embed)
            r = self.linears_r[i](gate_embed)
            h = torch.cat([hidden_states, r * pre_hidden_states], dim=-1)
            hidden_states = self.linears_h[i](h)

            hidden_states_f, residual_f = self.layers[i * 2](
                hidden_states, residual, inference_params=inference_params, **mixer_kwargs
            )
            hidden_states_b, residual_b = self.layers[i * 2 + 1](
                hidden_states.flip([1]),
                None if residual == None else residual.flip([1]),
                inference_params=inference_params,
                **mixer_kwargs,
            )
            hidden_states = (hidden_states_f + hidden_states_b.flip([1])) / 2
            residual = (residual_f + residual_b.flip([1])) / 2

            hidden_states_list.append((1 - z) * hidden_states + z * pre_hidden_states)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )
        return hidden_states, hidden_states_list


class PredMamba(LightningModule):
    def __init__(
        self,
        seq_in,
        seq_out,
        image_size,
        view_depth,
        point_dim,
        d_model,
        n_layer,
        d_intermediate,
        ssm_cfg,
        attn_layer_idx,
        attn_cfg,
        rms_norm,
        residual_in_fp32,
        fused_add_norm,
        initializer_cfg=None,
        device=None,
        dtype=None,
        lr=1e-4,
        value_scale=80,
    ) -> None:
        super().__init__()

        self.seq_in = seq_in
        self.seq_out = seq_out
        self.image_height, self.image_width = image_size
        self.view_depths = [i for i in range(view_depth)]
        self.lr = lr
        self.value_scale = value_scale
        factory_kwargs = {"device": device, "dtype": dtype}

        self.linear_in = nn.Sequential(
            nn.Linear(point_dim, d_model, **factory_kwargs), nn.LayerNorm(d_model), nn.GELU()
        )
        self.embed = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            d_intermediate=d_intermediate,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.linear_out = nn.Linear(d_model, point_dim, **factory_kwargs)

        self.loss = nn.MSELoss(reduction="mean")

    def set_norm_denorm(self, dataset: GaussiansDataset):
        self.norm_points = dataset.norm_points
        self.denorm_points = dataset.denorm_points
        self.norm_delta = dataset.norm_delta
        self.denorm_delta = dataset.denorm_delta

    def forward(self, points, inference_params=None, **mixer_kwargs):
        embedding = self.linear_in(points)
        B, L, N, C = embedding.shape
        pre_hidden_states_list = None
        pred = []
        for i in range(L):
            # hidden_states, pre_hidden_states_list = self.embed(embedding[:, i], pre_hidden_states_list)
            hidden_states, pre_hidden_states_list = checkpoint(
                self.embed,
                embedding[:, i],
                pre_hidden_states_list,
                inference_params=inference_params,
                **mixer_kwargs,
                use_reentrant=False,
            )
            pred_delta = self.linear_out(hidden_states)
            pred.append(pred_delta)

        for i in range(self.seq_out - 1):
            # input_embedding = self.linear_in(points[:, 0] + pred_delta)
            input_embedding = embedding[:, 0].detach() + hidden_states.detach()
            hidden_states, pre_hidden_states_list = checkpoint(
                self.embed,
                input_embedding,
                pre_hidden_states_list,
                inference_params=inference_params,
                **mixer_kwargs,
                use_reentrant=False,
            )
            pred_delta = self.linear_out(hidden_states)
            pred.append(pred_delta)
        pred = torch.stack(pred, dim=1)
        return pred

    def predict(self, points, inference_params=None, **mixer_kwargs):
        # pdb.set_trace()
        embedding = self.linear_in(points)
        B, L, N, C = embedding.shape
        pre_hidden_states_list = None
        pred = []
        points_embedding = [embedding[:, i] for i in range(L)]
        for i in range(L):
            # hidden_states, pre_hidden_states_list = self.embed(embedding[:, i], pre_hidden_states_list)
            hidden_states, pre_hidden_states_list = checkpoint(
                self.embed,
                embedding[:, i],
                pre_hidden_states_list,
                inference_params=inference_params,
                **mixer_kwargs,
                use_reentrant=False,
            )
            pred_delta = self.linear_out(hidden_states)
            pred.append(pred_delta)
        points_embedding.append(embedding[:, 0] + hidden_states)

        for i in range(1, self.seq_out):
            pre_hidden_states_list = None
            for j in range(L):
                hidden_states, pre_hidden_states_list = checkpoint(
                    self.embed,
                    points_embedding[i + j],
                    pre_hidden_states_list,
                    inference_params=inference_params,
                    **mixer_kwargs,
                    use_reentrant=False,
                )
            pred_delta = self.linear_out(hidden_states)
            pred.append(pred_delta)
            points_embedding.append(embedding[:, 0] + hidden_states)
        # pdb.set_trace()
        pred = torch.stack(pred, dim=1)
        return pred

    def compute_loss(self, pred: Tensor, gt: Tensor) -> Tensor:
        return self.loss(pred, gt)

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        B, L, N, C = batch.shape
        seq_delta = []
        for i in range(1, L):
            seq_delta.append(batch[:, i] - batch[:, 0])
        seq_delta = torch.stack(seq_delta, dim=1)
        seq_delta = self.norm_delta(seq_delta)
        input = self.norm_points(batch[:, : self.seq_in])
        output = self(input)

        loss = self.compute_loss(output, seq_delta)

        self.log_dict(
            {"train_loss": loss, "lr": self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]},
            on_step=True,
            sync_dist=True,
            logger=True,
            prog_bar=True,
        )

        return loss

    def on_validation_epoch_start(self, *args, **kwargs):
        self.image_save_path = Path(self.trainer.logger.log_dir) / "images"
        self.image_save_path.mkdir(exist_ok=True)

    def validation_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        B, L, N, C = batch.shape
        seq_delta = []
        for i in range(1, L):
            seq_delta.append(batch[:, i] - batch[:, 0])
        seq_delta = torch.stack(seq_delta, dim=1)
        seq_delta = self.norm_delta(seq_delta)
        input = self.norm_points(batch[:, : self.seq_in])
        output = self(input)

        loss = self.compute_loss(output, seq_delta)
        self.log("val_loss", loss, sync_dist=True)

        gt_delta = self.denorm_delta(seq_delta)
        output = self.denorm_delta(output)
        gt_points = [batch[:, 0]]
        pred_points = [batch[:, 0]]
        for i in range(L - 1):
            gt_points.append(gt_points[0] + gt_delta[:, i])
            pred_points.append(pred_points[0] + output[:, i])
        gt_points = torch.stack(gt_points, dim=1)
        pred_points = torch.stack(pred_points, dim=1)
        if self.current_epoch % 10 == 0 and batch_idx < 2 and self.local_rank == 0:
            rendered_sequences = self.render(pred_points)
            gt_sequences = self.render(gt_points)
            for b in range(B):
                for seq_idx in range(self.seq_in + self.seq_out):
                    for view_idx in [2, 4, 8]:
                        rendered = torch.clamp(rendered_sequences[b][seq_idx][view_idx][0], 0, 1)
                        rendered = (rendered.detach().cpu().numpy() * 255).astype(np.uint8)
                        # rendered = (normalize_to_rgb(rendered.detach().cpu().numpy()) * 255).astype(np.uint8)
                        image = Image.fromarray(rendered)
                        image.save(
                            self.image_save_path
                            / f"img{b + batch_idx * B:0>4d}-view{view_idx}-seq{seq_idx:0>2d}-render.png"
                        )
                        gt = torch.clamp(gt_sequences[b][seq_idx][view_idx][0], 0, 1)
                        gt = (gt.detach().cpu().numpy() * 255).astype(np.uint8)
                        # gt = (normalize_to_rgb(gt.detach().cpu().numpy()) * 255).astype(np.uint8)
                        image = Image.fromarray(gt)
                        image.save(
                            self.image_save_path
                            / f"img{b + batch_idx * B:0>4d}-view{view_idx}-seq{seq_idx:0>2d}-gt.png"
                        )

        return loss

    @torch.no_grad()
    def render(self, seq_points: Tensor) -> list:
        B, L, N, C = seq_points.shape
        rendered_seqs = [[] for _ in range(B)]
        for seq_idx in range(L):
            points = seq_points[:, seq_idx]
            for b in range(B):
                means3D = points[b, :, :3]
                intensity = points[b, :, 3:-7]
                means2D = torch.zeros_like(means3D, dtype=points.dtype, requires_grad=True, device=points.device) + 0
                scales = F.softplus(points[b, :, -7:-4])
                rotations = F.normalize(points[b, :, -4:])
                cov3D_precomp = None

                rendered_tmp = []
                for view_depth in self.view_depths:
                    raster_settings = GaussianRasterizationSettings(
                        image_height=int(self.image_height),
                        image_width=int(self.image_width),
                        image_channel=intensity.shape[-1],
                        scale_modifier=1.0,
                        viewmatrix=torch.tensor(
                            [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]],
                            device=means3D.device,
                        ),
                        viewdepth=view_depth,
                        prefiltered=False,
                        debug=False,
                    )

                    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

                    rendered_radar, radii = rasterizer(
                        means3D=means3D,
                        means2D=means2D,
                        intensity=intensity,
                        scales=scales,
                        rotations=rotations,
                        cov3D_precomp=cov3D_precomp,
                    )
                    rendered_tmp.append(rendered_radar)

                rendered_seqs[b].append(rendered_tmp)
        return rendered_seqs

    def on_test_start(self) -> None:
        self.evaluator = Evaluator(seq_len=self.seq_out, value_scale=self.value_scale)

        self.image_save_path = Path(self.trainer.logger.log_dir) / "images"
        self.image_save_path.mkdir(exist_ok=True)

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        points, raw_data, seq = batch
        B, L, N, C = points.shape

        input = self.norm_points(points[:, : self.seq_in])
        output = self(input)

        output = self.denorm_delta(output)
        pred_points = [points[:, 0]]
        for i in range(L - 1):
            pred_points.append(points[:, 0] + output[:, i])
        pred_points = torch.stack(pred_points, dim=1)

        rendered = self.render(pred_points[:, self.seq_in :])
        gt = raw_data[:, self.seq_in :]

        render_imgs = []
        for batch_data in rendered:
            batch_imgs = []
            for temp in batch_data:
                temp = torch.stack(temp, dim=1)
                batch_imgs.append(temp)
            batch_imgs = torch.stack(batch_imgs, dim=0)
            render_imgs.append(batch_imgs)
        render_imgs = torch.stack(render_imgs, dim=0)

        torch.clamp_(render_imgs[:, :, :2], 0, 1)
        torch.clamp_(render_imgs[:, :, 2:], -1, 1)
        self.evaluator(render_imgs, gt)

    def on_test_end(self) -> None:
        metrics = self.evaluator.get_metrics()
        score_path = Path(self.trainer.logger.log_dir) / "score.json"
        with open(score_path, "w") as f:
            json.dump(metrics, f, indent=4)

    def configure_optimizers(self) -> dict:
        optimizer = AdamW(
            self.parameters(),
            lr=self.lr,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, num_warmup_steps=100, num_training_steps=self.trainer.estimated_stepping_batches
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
