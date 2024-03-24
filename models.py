import torch
import torch.nn as nn
from torch import Tensor

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Tuple
from tqdm.notebook import trange
import matplotlib.pyplot as plt
from copy import deepcopy
import re
import os
import seaborn as sns
import pandas as pd
import random
import numpy as np
import pickle
import collections


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
        bitwidth=7,
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
        self.fsr = fsr
        self.bitwidth = bitwidth

    def forward(self, input):
        # apply quantize function for QAT, otherwise normal conv2d forward pass
        if self.quantize_fn is not None:
            quantized_weight = self.quantize_fn.apply(
                self.weight, self.fsr, self.bitwidth
            )
            return self._conv_forward(input, quantized_weight, self.bias)
        else:
            return self._conv_forward(input, self.weight, self.bias)

    def get_quantization_error(self):
        if self.quantize_fn is not None:
            quantized_weight = self.quantize_fn.apply(
                self.weight, self.fsr, self.bitwidth
            )
            return torch.sum((quantized_weight - self.weight) ** 2), self.weight.numel()
        else:
            return 0, self.weight.numel()


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        quantize_fn: Optional[Callable] = None,
        fsr: int = 1,
        bitwidth: int = 7,
    ) -> None:
        super().__init__()

        # both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = QuantizedConv2d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
            quantize_fn=quantize_fn,
            fsr=fsr,
            bitwidth=bitwidth,
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = QuantizedConv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
            bias=False,
            dilation=1,
            quantize_fn=quantize_fn,
            fsr=fsr,
            bitwidth=bitwidth,
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

    def get_quantization_error(self):

        quantization_error = 0.0
        numel = 0

        qerror1, numel1 = self.conv1.get_quantization_error()
        qerror2, numel2 = self.conv2.get_quantization_error()

        quantization_error += qerror1 + qerror2
        numel += numel1 + numel2

        return quantization_error, numel


class ResNet(nn.Module):

    def __init__(
        self,
        block: BasicBlock,
        num_blocks: List[int],
        num_filters: List[int],
        num_classes: int = 10,
        groups: int = 1,
        width_per_group: int = 64,
        quantize_fn: Optional[Callable] = None,
        fsr: int = 1,
        bitwidth: int = 7,
    ) -> None:
        super().__init__()

        self.inplanes = 16

        # don't apply QAT training to this first conv2d block, regardless
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            block,
            num_filters[0],
            num_blocks[0],
            stride=1,
            quantize_fn=quantize_fn,
            fsr=fsr,
            bitwidth=bitwidth,
        )
        self.layer2 = self._make_layer(
            block,
            num_filters[1],
            num_blocks[1],
            stride=2,
            quantize_fn=quantize_fn,
            fsr=fsr,
            bitwidth=bitwidth,
        )  # stride 2 creates subsampling
        self.layer3 = self._make_layer(
            block,
            num_filters[2],
            num_blocks[2],
            stride=2,
            quantize_fn=quantize_fn,
            fsr=fsr,
            bitwidth=bitwidth,
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: BasicBlock,
        planes: int,
        blocks: int,
        stride: int = 1,
        quantize_fn: Optional[Callable] = None,
        fsr: int = 1,
        bitwidth: int = 7,
    ) -> nn.Sequential:

        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                QuantizedConv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                    quantize_fn=quantize_fn,
                    fsr=fsr,
                    bitwidth=bitwidth,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                quantize_fn=quantize_fn,
                fsr=fsr,
                bitwidth=bitwidth,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    quantize_fn=quantize_fn,
                    fsr=fsr,
                    bitwidth=bitwidth,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)  # 64x3x32x32 -> 64x16x32x32
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)  # 64x16x32x32 (feature map size 32)
        x = self.layer2(x)  # 64x32x16x16
        x = self.layer3(x)  # 64x64x8x8

        x = self.avgpool(x)  # reduces feature map to get 64x64x1x1
        x = torch.flatten(x, 1)  # flatten spatial dimension to get 64x64
        return self.fc(x)  # 64x10

    def _get_quantization_error_for_layer(self, layer: nn.Module) -> Tuple[float, int]:

        quantization_error = 0.0
        numel = 0

        for sub_layer in layer:
            if isinstance(sub_layer, BasicBlock):
                qerror, numel = sub_layer.get_quantization_error()
                quantization_error += qerror
                numel += numel

        return quantization_error, numel

    def get_quantization_error(self):

        quantization_error = 0.0
        numel = 0

        qerror1, numel1 = self._get_quantization_error_for_layer(self.layer1)
        qerror2, numel2 = self._get_quantization_error_for_layer(self.layer2)
        qerror3, numel3 = self._get_quantization_error_for_layer(self.layer3)

        quantization_error += qerror1 + qerror2 + qerror3
        numel += numel1 + numel2 + numel3

        return quantization_error, numel


def resnet20(
    *,
    n: int = 3,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    fsr: int = 1,
    bitwidth: int = 7,
    **kwargs: Any
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        fsr=fsr,
        bitwidth=bitwidth,
        **kwargs
    )


def resnet32(
    *,
    n: int = 5,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    fsr: int = 1,
    bitwidth: int = 7,
    **kwargs: Any
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        fsr=fsr,
        bitwidth=bitwidth,
        **kwargs
    )


def resnet44(
    *,
    n: int = 7,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    fsr: int = 1,
    bitwidth: int = 7,
    **kwargs: Any
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        fsr=fsr,
        bitwidth=bitwidth,
        **kwargs
    )


def resnet56(
    *,
    n: int = 9,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    fsr: int = 1,
    bitwidth: int = 7,
    **kwargs: Any
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        fsr=fsr,
        bitwidth=bitwidth,
        **kwargs
    )
