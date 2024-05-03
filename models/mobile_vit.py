"""
Adapted from https://github.com/lucidrains/vit-pytorch/blob/main/vit_pytorch/mobile_vit.py
"""

from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from einops.layers.torch import Reduce

from models.quantized_conv import QuantizedConv2d


def quantized_conv_1x1_bn(inp, oup, quantize_fn=None, bits=4):
    return nn.Sequential(
        QuantizedConv2d(
            inp, oup, 1, 1, 0, bias=False, quantize_fn=quantize_fn, bits=bits
        ),
        nn.SyncBatchNorm(oup),
        nn.SiLU(),
    )


def quantized_conv_nxn_bn(inp, oup, kernel_size=3, stride=1, quantize_fn=None, bits=4):
    return nn.Sequential(
        QuantizedConv2d(
            inp,
            oup,
            kernel_size,
            stride,
            1,
            bias=False,
            quantize_fn=quantize_fn,
            bits=bits,
        ),
        nn.SyncBatchNorm(oup),
        nn.SiLU(),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False), nn.SyncBatchNorm(oup), nn.SiLU()
    )


def conv_nxn_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size, stride, 1, bias=False),
        nn.SyncBatchNorm(oup),
        nn.SiLU(),
    )


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)

        q, k, v = map(
            lambda t: rearrange(t, "b p n (h d) -> b p h n d", h=self.heads), qkv
        )

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b p h n d -> b p n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """Transformer block described in ViT.
    Paper: https://arxiv.org/abs/2010.11929
    Based on: https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads, dim_head, dropout),
                        FeedForward(dim, mlp_dim, dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class MV2Block(nn.Module):
    """MV2 block described in MobileNetV2.
    Paper: https://arxiv.org/pdf/1801.04381
    Based on: https://github.com/tonylins/pytorch-mobilenet-v2
    """

    def __init__(
        self,
        inp,
        oup,
        stride=1,
        expansion=4,
        quantize_fn: Optional[Callable] = None,
        bits: int = 4,
    ):
        super().__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(inp * expansion)
        self.use_res_connect = self.stride == 1 and inp == oup

        if expansion == 1:
            self.conv = nn.Sequential(
                # dw
                QuantizedConv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                    quantize_fn=quantize_fn,
                    bits=bits,
                ),
                nn.SyncBatchNorm(hidden_dim),
                nn.SiLU(),
                # pw-linear
                QuantizedConv2d(
                    hidden_dim,
                    oup,
                    1,
                    1,
                    0,
                    bias=False,
                    quantize_fn=quantize_fn,
                    bits=bits,
                ),
                nn.SyncBatchNorm(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                QuantizedConv2d(
                    inp,
                    hidden_dim,
                    1,
                    1,
                    0,
                    bias=False,
                    quantize_fn=quantize_fn,
                    bits=bits,
                ),
                nn.SyncBatchNorm(hidden_dim),
                nn.SiLU(),
                # dw
                QuantizedConv2d(
                    hidden_dim,
                    hidden_dim,
                    3,
                    stride,
                    1,
                    groups=hidden_dim,
                    bias=False,
                    quantize_fn=quantize_fn,
                    bits=bits,
                ),
                nn.SyncBatchNorm(hidden_dim),
                nn.SiLU(),
                # pw-linear
                QuantizedConv2d(
                    hidden_dim,
                    oup,
                    1,
                    1,
                    0,
                    bias=False,
                    quantize_fn=quantize_fn,
                    bits=bits,
                ),
                nn.SyncBatchNorm(oup),
            )

    def forward(self, x):
        out = self.conv(x)
        return out + x if self.use_res_connect else out

    def get_quantization_error(self):
        quantization_error, numel = 0.0, 0

        for sub_layer in self.conv:
            if isinstance(sub_layer, QuantizedConv2d):
                qerror, numel_sub = sub_layer.get_quantization_error()
                quantization_error += qerror
                numel += numel_sub

        return quantization_error, numel


class MobileViTBlock(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        channel,
        kernel_size,
        patch_size,
        mlp_dim,
        dropout=0.0,
        quantize_fn: Optional[Callable] = None,
        bits: int = 4,
    ):
        super().__init__()
        self.ph, self.pw = patch_size

        self.conv1 = quantized_conv_nxn_bn(
            channel, channel, kernel_size, quantize_fn=quantize_fn, bits=bits
        )
        self.conv2 = quantized_conv_1x1_bn(
            channel, dim, quantize_fn=quantize_fn, bits=bits
        )

        self.transformer = Transformer(dim, depth, 4, 8, mlp_dim, dropout)

        self.conv3 = quantized_conv_1x1_bn(
            dim, channel, quantize_fn=quantize_fn, bits=bits
        )
        self.conv4 = quantized_conv_nxn_bn(
            2 * channel, channel, kernel_size, quantize_fn=quantize_fn, bits=bits
        )

    def forward(self, x):
        y = x.clone()

        # Local representations
        x = self.conv1(x)
        x = self.conv2(x)

        # Global representations
        _, _, h, w = x.shape
        x = rearrange(
            x, "b d (h ph) (w pw) -> b (ph pw) (h w) d", ph=self.ph, pw=self.pw
        )
        x = self.transformer(x)
        x = rearrange(
            x,
            "b (ph pw) (h w) d -> b d (h ph) (w pw)",
            h=h // self.ph,
            w=w // self.pw,
            ph=self.ph,
            pw=self.pw,
        )

        # Fusion
        x = self.conv3(x)
        x = torch.cat((x, y), 1)
        x = self.conv4(x)
        return x

    def get_quantization_error(self):
        quantization_error, numel = 0.0, 0

        for conv in [self.conv1, self.conv2, self.conv3, self.conv4]:
            for sub_layer in conv:
                if isinstance(sub_layer, QuantizedConv2d):
                    qerror, numel_sub = sub_layer.get_quantization_error()
                    quantization_error += qerror
                    numel += numel_sub

        return quantization_error, numel


class MobileViT(nn.Module):
    """MobileViT.
    Paper: https://arxiv.org/abs/2110.02178
    Based on: https://github.com/chinhsuanwu/mobilevit-pytorch
    """

    def __init__(
        self,
        image_size,
        dims,
        channels,
        num_classes,
        expansion=4,
        kernel_size=3,
        patch_size=(2, 2),
        depths=(2, 4, 3),
        quantize_fn: Optional[Callable] = None,
        bits: int = 4,
    ):
        super().__init__()
        assert len(dims) == 3, "dims must be a tuple of 3"
        assert len(depths) == 3, "depths must be a tuple of 3"

        ih, iw = image_size
        ph, pw = patch_size
        assert ih % ph == 0 and iw % pw == 0

        init_dim, *_, last_dim = channels

        # first conv not quantized
        self.conv1 = conv_nxn_bn(3, init_dim, stride=2)

        self.stem = nn.ModuleList([])
        self.stem.append(
            MV2Block(
                channels[0],
                channels[1],
                1,
                expansion,
                quantize_fn=quantize_fn,
                bits=bits,
            )
        )
        self.stem.append(
            MV2Block(
                channels[1],
                channels[2],
                2,
                expansion,
                quantize_fn=quantize_fn,
                bits=bits,
            )
        )
        self.stem.append(
            MV2Block(
                channels[2],
                channels[3],
                1,
                expansion,
                quantize_fn=quantize_fn,
                bits=bits,
            )
        )
        self.stem.append(
            MV2Block(
                channels[2],
                channels[3],
                1,
                expansion,
                quantize_fn=quantize_fn,
                bits=bits,
            )
        )

        self.trunk = nn.ModuleList([])
        self.trunk.append(
            nn.ModuleList(
                [
                    MV2Block(
                        channels[3],
                        channels[4],
                        2,
                        expansion,
                        quantize_fn=quantize_fn,
                        bits=bits,
                    ),
                    MobileViTBlock(
                        dims[0],
                        depths[0],
                        channels[5],
                        kernel_size,
                        patch_size,
                        int(dims[0] * 2),
                        quantize_fn=quantize_fn,
                        bits=bits,
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2Block(
                        channels[5],
                        channels[6],
                        2,
                        expansion,
                        quantize_fn=quantize_fn,
                        bits=bits,
                    ),
                    MobileViTBlock(
                        dims[1],
                        depths[1],
                        channels[7],
                        kernel_size,
                        patch_size,
                        int(dims[1] * 4),
                        quantize_fn=quantize_fn,
                        bits=bits,
                    ),
                ]
            )
        )

        self.trunk.append(
            nn.ModuleList(
                [
                    MV2Block(
                        channels[7],
                        channels[8],
                        2,
                        expansion,
                        quantize_fn=quantize_fn,
                        bits=bits,
                    ),
                    MobileViTBlock(
                        dims[2],
                        depths[2],
                        channels[9],
                        kernel_size,
                        patch_size,
                        int(dims[2] * 4),
                        quantize_fn=quantize_fn,
                        bits=bits,
                    ),
                ]
            )
        )

        # last conv not quantized
        self.to_logits = nn.Sequential(
            conv_1x1_bn(channels[-2], last_dim),
            Reduce("b c h w -> b c", "mean"),
            nn.Linear(channels[-1], num_classes, bias=False),
        )

    def forward(self, x):
        x = self.conv1(x)

        for conv in self.stem:
            x = conv(x)

        for conv, attn in self.trunk:
            x = conv(x)
            x = attn(x)

        return self.to_logits(x)

    def _get_quantization_error_for_layer(self, layer: nn.Module) -> Tuple[float, int]:
        quantization_error, numel = 0.0, 0

        for sub_layer in layer:
            if isinstance(sub_layer, MV2Block) or isinstance(sub_layer, MobileViTBlock):
                qerror, numel_sub = sub_layer.get_quantization_error()
                quantization_error += qerror
                numel += numel_sub

        return quantization_error, numel

    def get_quantization_error(self):
        quantization_error, numel = 0.0, 0

        qerror1, numel1 = self._get_quantization_error_for_layer(self.stem)
        qerror2, numel2 = self._get_quantization_error_for_layer(self.stem)

        quantization_error += qerror1 + qerror2
        numel += numel1 + numel2

        return quantization_error, numel


def MobileVIT(
    *,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    bits: int = 4,
    **kwargs: Any,
) -> MobileViT:
    return MobileViT(
        num_classes=num_classes, quantize_fn=quantize_fn, bits=bits, **kwargs
    )
