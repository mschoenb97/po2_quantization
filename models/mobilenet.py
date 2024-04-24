from typing import Callable, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


class QuantizedConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=1,
        dilation=1,
        groups=1,
        bias=False,
        quantize_fn=None,
        fsr=1,
        bits=4,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.quantize_fn = quantize_fn
        self.bits = bits

    def forward(self, input):
        # apply quantize function for QAT, otherwise normal conv2d forward pass
        if self.quantize_fn is not None:
            quantized_weight = self.quantize_fn.apply(self.weight, self.bits)
            return self._conv_forward(input, quantized_weight, self.bias)
        else:
            return self._conv_forward(input, self.weight, self.bias)

    def get_quantization_error(self):
        if self.quantize_fn is not None:
            quantized_weight = self.quantize_fn.apply(self.weight, self.bits)
            return torch.sum((quantized_weight - self.weight) ** 2), self.weight.numel()
        else:
            return 0, self.weight.numel()


class InvertedResidual(nn.Module):
    def __init__(
        self,
        inp: int,
        oup: int,
        stride: int,
        expand_ratio: int,
        quantize_fn: Optional[Callable] = None,
        bits: int = 4,
    ):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                QuantizedConv2d(
                    inp, hidden_dim, kernel_size=1, quantize_fn=quantize_fn, bits=bits
                )
            )
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))

        layers.extend(
            [
                # dw
                QuantizedConv2d(
                    hidden_dim,
                    hidden_dim,
                    stride=stride,
                    padding=1,
                    groups=hidden_dim,
                    quantize_fn=quantize_fn,
                    bits=bits,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                QuantizedConv2d(
                    hidden_dim, oup, kernel_size=1, quantize_fn=quantize_fn, bits=bits
                ),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

    def get_quantization_error(self):
        quantization_error = 0.0
        numel = 0

        for layer in self.conv:
            if isinstance(layer, QuantizedConv2d):
                qerror, n = layer.get_quantization_error()
                quantization_error += qerror
                numel += n

        return quantization_error, numel


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        width_mult: float = 1.0,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        quantize_fn: Optional[Callable] = None,
        bits: int = 4,
    ) -> None:
        super(MobileNetV2, self).__init__()

        if block is None:
            block = InvertedResidual

        input_channel = 32
        last_channel = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if (
            len(inverted_residual_setting) == 0
            or len(inverted_residual_setting[0]) != 4
        ):
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 4-element list, got {inverted_residual_setting}"
            )

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(
            last_channel * max(1.0, width_mult), round_nearest
        )

        features: List[nn.Module] = [
            nn.Conv2d(3, input_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]

        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    block(
                        input_channel,
                        output_channel,
                        stride,
                        expand_ratio=t,
                        quantize_fn=quantize_fn,
                        bits=bits,
                    )
                )
                input_channel = output_channel

        # building last several layers
        features.append(
            nn.Conv2d(
                input_channel, self.last_channel, kernel_size=1, stride=1, bias=False
            )
        )
        features.append(nn.BatchNorm2d(self.last_channel))
        features.append(nn.ReLU6(inplace=True))

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        # building classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_quantization_error(self):
        quantization_error = 0.0
        numel = 0

        for layer in self.features:
            if isinstance(layer, InvertedResidual):
                qerror, n = layer.get_quantization_error()
                quantization_error += qerror
                numel += n

        return quantization_error, numel


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
