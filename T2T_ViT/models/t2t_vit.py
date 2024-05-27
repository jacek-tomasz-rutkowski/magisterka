# Copyright (c) [2012]-[2021] Shanghai Yitu Technology Co., Ltd.
#
# This source code is licensed under the Clear BSD License
# LICENSE file in the root directory of this file
# All rights reserved.
"""
T2T-ViT
"""
import torch
import torch.nn as nn
from pathlib import Path

from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model, generate_default_cfgs
from timm.models.layers import trunc_normal_
import numpy as np
from models.token_transformer import Token_transformer
from models.token_performer import Token_performer
from models.transformer_block import Block, get_sinusoid_encoding


def _cfg(url="", **kwargs):
    return {
        "file": str(url),
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "pool_size": None,
        "crop_pct": 0.9,
        "interpolation": "bicubic",
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "classifier": "head",
        **kwargs,
    }


CHECKPOINTS_DIR = Path(__file__).parent.parent / "saved_models/downloaded/imagenet/"

default_cfgs = generate_default_cfgs(
    {
        "t2t_vit_7": _cfg(CHECKPOINTS_DIR / "71.7_T2T_ViT_7.pth"),
        "t2t_vit_10": _cfg(CHECKPOINTS_DIR / "75.2_T2T_ViT_10.pth"),
        "t2t_vit_12": _cfg(CHECKPOINTS_DIR / "76.5_T2T_ViT_12.pth"),
        "t2t_vit_14": _cfg(CHECKPOINTS_DIR / "81.5_T2T_ViT_14.pth"),
        "t2t_vit_19": _cfg(CHECKPOINTS_DIR / "81.9_T2T_ViT_19.pth"),
        "t2t_vit_24": _cfg(),
        "t2t_vit_t_14": _cfg(file=CHECKPOINTS_DIR / "81.7_T2T_ViTt_14.pth"),
        "t2t_vit_t_19": _cfg(),
        "t2t_vit_t_24": _cfg(),
        "t2t_vit_14_resnext": _cfg(),
        "t2t_vit_14_wide": _cfg(),
    }
)


class T2T_module(nn.Module):
    """
    Tokens-to-Token encoding module
    """

    def __init__(self, img_size=224, tokens_type="performer", in_chans=3, embed_dim=768, token_dim=64):
        super().__init__()

        self.attention1: Token_transformer | Token_performer
        self.attention2: Token_transformer | Token_performer
        self.soft_split0: nn.Unfold | nn.Conv2d
        self.soft_split1: nn.Unfold | nn.Conv2d
        self.soft_split2: nn.Unfold | nn.Conv2d
        self.project: nn.Linear | nn.Conv2d
        if tokens_type == "transformer":
            print("adopt transformer encoder for tokens-to-token")
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            self.attention1 = Token_transformer(dim=in_chans * 7 * 7, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.attention2 = Token_transformer(dim=token_dim * 3 * 3, in_dim=token_dim, num_heads=1, mlp_ratio=1.0)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == "performer":
            print("adopt performer encoder for tokens-to-token")
            self.soft_split0 = nn.Unfold(kernel_size=(7, 7), stride=(4, 4), padding=(2, 2))
            self.soft_split1 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
            self.soft_split2 = nn.Unfold(kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

            # self.attention1 = Token_performer(dim=token_dim, in_dim=in_chans*7*7, kernel_ratio=0.5)
            # self.attention2 = Token_performer(dim=token_dim, in_dim=token_dim*3*3, kernel_ratio=0.5)
            self.attention1 = Token_performer(dim=in_chans * 7 * 7, in_dim=token_dim, kernel_ratio=0.5)
            self.attention2 = Token_performer(dim=token_dim * 3 * 3, in_dim=token_dim, kernel_ratio=0.5)
            self.project = nn.Linear(token_dim * 3 * 3, embed_dim)

        elif tokens_type == "convolution":  # just for comparison with conolution, not our model
            # for this tokens type, you need change forward as three convolution operation
            print("adopt convolution layers for tokens-to-token")
            self.soft_split0 = nn.Conv2d(
                3, token_dim, kernel_size=(7, 7), stride=(4, 4), padding=(2, 2)
            )  # the 1st convolution
            self.soft_split1 = nn.Conv2d(
                token_dim, token_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )  # the 2nd convolution
            self.project = nn.Conv2d(
                token_dim, embed_dim, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )  # the 3rd convolution

        self.num_patches = (img_size // (4 * 2 * 2)) * (
            img_size // (4 * 2 * 2)
        )  # there are 3 sfot split, stride are 4,2,2 seperately

    def forward(self, x):
        # step0: soft split
        x = self.soft_split0(x).transpose(1, 2)

        # iteration1: re-structurization/reconstruction
        x = self.attention1(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))

        # iteration1: soft split
        x = self.soft_split1(x).transpose(1, 2)

        # iteration2: re-structurization/reconstruction
        x = self.attention2(x)
        B, new_HW, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, int(np.sqrt(new_HW)), int(np.sqrt(new_HW)))
        # iteration2: soft split
        x = self.soft_split2(x).transpose(1, 2)

        # final tokens
        x = self.project(x)

        return x


class T2T_ViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        tokens_type="performer",
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        token_dim=64,
        pretrained_cfg=None,
        pretrained_cfg_overlay=None,
        global_pool: str = "token"
    ):
        super().__init__()
        assert not pretrained_cfg_overlay, f"Received unexpected {pretrained_cfg_overlay=}"
        assert not pretrained_cfg, f"Received unexpected {pretrained_cfg=}"
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.tokens_to_token = T2T_module(
            img_size=img_size, tokens_type=tokens_type, in_chans=in_chans, embed_dim=embed_dim, token_dim=token_dim
        )
        num_patches = self.tokens_to_token.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            data=get_sinusoid_encoding(n_position=num_patches + 1, d_hid=embed_dim), requires_grad=False
        )
        # wcześniej było n_position=num_patches + 1
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)
        self.feature_info = [dict(num_chs=embed_dim, reduction=16, module=f'blocks.{i}') for i in range(depth)]

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"cls_token"}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=""):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        x = self.tokens_to_token(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False):
        x = x[:, 0]  # Take classification token only, we don't support other methods of global pooling.
        return x if pre_logits else self.head(x)

    # def forward_head(self, x, pre_logits: bool = False):
    #     return self.head(x, pre_logits=True) if pre_logits else self.head(x)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


@register_model
def t2t_vit_7(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 256**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=256, depth=7, num_heads=4, mlp_ratio=2.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_7", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_10(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 256**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=256, depth=10, num_heads=4, mlp_ratio=2.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_10", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_12(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 256**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=256, depth=12, num_heads=4, mlp_ratio=2.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_12", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_14(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 384**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.0, **kwargs)
    model = build_model_with_cfg(T2T_ViT, "t2t_vit_14", pretrained=pretrained, **kwargs)
    return model


@register_model
def t2t_vit_19(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 448**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=448, depth=19, num_heads=7, mlp_ratio=3.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_19", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_24(pretrained=False, **kwargs):  # adopt performer for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 512**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=512, depth=24, num_heads=8, mlp_ratio=3.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_24", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_t_14(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 384**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="transformer", embed_dim=384, depth=14, num_heads=6, mlp_ratio=3.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_t_14", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_t_19(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 448**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="transformer", embed_dim=448, depth=19, num_heads=7, mlp_ratio=3.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_t_19", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_t_24(pretrained=False, **kwargs):  # adopt transformers for tokens to token
    if pretrained:
        kwargs.setdefault("qk_scale", 512**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="transformer", embed_dim=512, depth=24, num_heads=8, mlp_ratio=3.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_t_24", pretrained=pretrained, **kwargs)


# rexnext and wide structure
@register_model
def t2t_vit_14_resnext(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault("qk_scale", 384**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=384, depth=14, num_heads=32, mlp_ratio=3.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_14_wide", pretrained=pretrained, **kwargs)


@register_model
def t2t_vit_14_wide(pretrained=False, **kwargs):
    if pretrained:
        kwargs.setdefault("qk_scale", 512**-0.5)
        kwargs.setdefault("in_chans", 3)
    kwargs = dict(tokens_type="performer", embed_dim=768, depth=4, num_heads=12, mlp_ratio=3.0, **kwargs)
    return build_model_with_cfg(T2T_ViT, "t2t_vit_14_wide", pretrained=pretrained, **kwargs)
