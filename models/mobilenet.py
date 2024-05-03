"""
Creates a MobileNetV2 Model as defined in:
Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen. (2018).
MobileNetV2: Inverted Residuals and Linear Bottlenecks
arXiv preprint arXiv:1801.04381.
Adapted from https://github.com/tonylins/pytorch-mobilenet-v2
"""

import math
from typing import Any, Callable, Optional, Tuple

import torch.nn as nn

from models.quantized_conv import QuantizedConv2d


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def quantized_conv_3x3_bn(inp, oup, stride, quantize_fn=None, bits=4):
    return nn.Sequential(
        QuantizedConv2d(
            inp, oup, 3, stride, 1, bias=False, quantize_fn=quantize_fn, bits=bits
        ),
        nn.SyncBatchNorm(oup),
        nn.ReLU6(inplace=True),
    )


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.SyncBatchNorm(oup),
        nn.ReLU6(inplace=True),
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.SyncBatchNorm(oup),
        nn.ReLU6(inplace=True),
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio, quantize_fn=None, bits=4):
        super().__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
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
                nn.ReLU6(inplace=True),
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
                nn.ReLU6(inplace=True),
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
                nn.ReLU6(inplace=True),
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
        return x + self.conv(x) if self.identity else self.conv(x)

    def get_quantization_error(self):
        quantization_error, numel = 0.0, 0

        for sub_layer in self.conv:
            if isinstance(sub_layer, QuantizedConv2d):
                qerror, numel_sub = sub_layer.get_quantization_error()
                quantization_error += qerror
                numel += numel_sub

        return quantization_error, numel


class MobileNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        quantize_fn: Optional[Callable] = None,
        bits: int = 4,
    ):
        super().__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        # first conv not quantized
        layers = [conv_3x3_bn(3, input_channel, 2)]
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(
                c * width_mult, 4 if width_mult == 0.1 else 8
            )
            for i in range(n):
                layers.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        s if i == 0 else 1,
                        t,
                        quantize_fn=quantize_fn,
                        bits=bits,
                    )
                )
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        output_channel = (
            _make_divisible(1280 * width_mult, 4 if width_mult == 0.1 else 8)
            if width_mult > 1.0
            else 1280
        )
        # last conv not quantized
        self.conv = conv_1x1_bn(input_channel, output_channel)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(output_channel, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.conv(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _get_quantization_error_for_layer(self, layer: nn.Module) -> Tuple[float, int]:
        quantization_error, numel = 0.0, 0

        for sub_layer in layer:
            if isinstance(sub_layer, InvertedResidual):
                qerror, numel_sub = sub_layer.get_quantization_error()
                quantization_error += qerror
                numel += numel_sub

        return quantization_error, numel

    def get_quantization_error(self):
        quantization_error, numel = 0.0, 0

        qerror, numel = self._get_quantization_error_for_layer(self.features)
        quantization_error += qerror
        numel += numel

        return quantization_error, numel


def MobileNetV2(
    *,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    bits: int = 4,
    **kwargs: Any,
) -> MobileNet:
    return MobileNet(
        num_classes=num_classes, quantize_fn=quantize_fn, bits=bits, **kwargs
    )
