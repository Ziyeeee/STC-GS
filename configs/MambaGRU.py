from dataclasses import dataclass, field


@dataclass
class PredMambaConfig:
    value_scale = 70
    image_height: int = 512
    image_width: int = 512
    view_depth: int = 36
    seq_len: int = 25
    seq_in: int = 5
    seq_out: int = 20

    point_dim: int = 16

    # d_model: int = 192
    d_intermediate: int = 0
    # n_layer: int = 32
    ssm_cfg: dict = field(default_factory=dict)
    attn_layer_idx: list = field(default_factory=list)
    attn_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    residual_in_fp32: bool = True
    fused_add_norm: bool = True

    down_block_types = (
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "DownBlock2D",
        "AttnDownBlock2D",
        "DownBlock2D",
    )
    up_block_types = (
        "UpBlock2D",
        "AttnUpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
        "UpBlock2D",
    )
    layers_per_block = 4  # how many ResNet layers to use per block
    block_out_channels = (4, 8, 16, 32, 64, 128)  # the number of output channels for each block
