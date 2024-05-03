from typing import Any, Callable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from models.quantized_conv import QuantizedConv2d


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        quantize_fn: Optional[Callable] = None,
        bits: int = 7,
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
            bits=bits,
        )
        self.bn1 = nn.SyncBatchNorm(planes)
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
            bits=bits,
        )
        self.bn2 = nn.SyncBatchNorm(planes)
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
        quantization_error, numel = 0.0, 0
        
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
        quantize_fn: Optional[Callable] = None,
        bits: int = 7,
    ) -> None:
        super().__init__()

        self.inplanes = 16

        # don't apply QAT training to this first conv2d block, regardless
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.SyncBatchNorm(self.inplanes)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(
            block,
            num_filters[0],
            num_blocks[0],
            stride=1,
            quantize_fn=quantize_fn,
            bits=bits,
        )
        self.layer2 = self._make_layer(
            block,
            num_filters[1],
            num_blocks[1],
            stride=2,
            quantize_fn=quantize_fn,
            bits=bits,
        )  # stride 2 creates subsampling
        self.layer3 = self._make_layer(
            block,
            num_filters[2],
            num_blocks[2],
            stride=2,
            quantize_fn=quantize_fn,
            bits=bits,
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
        bits: int = 7,
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
                    bits=bits,
                ),
                nn.SyncBatchNorm(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                quantize_fn=quantize_fn,
                bits=bits,
            )
        )
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    quantize_fn=quantize_fn,
                    bits=bits,
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
        quantization_error, numel = 0.0, 0

        for sub_layer in layer:
            if isinstance(sub_layer, BasicBlock):
                qerror, numel = sub_layer.get_quantization_error()
                quantization_error += qerror
                numel += numel

        return quantization_error, numel

    def get_quantization_error(self):
        quantization_error, numel = 0.0, 0

        qerror1, numel1 = self._get_quantization_error_for_layer(self.layer1)
        qerror2, numel2 = self._get_quantization_error_for_layer(self.layer2)
        qerror3, numel3 = self._get_quantization_error_for_layer(self.layer3)

        quantization_error += qerror1 + qerror2 + qerror3
        numel += numel1 + numel2 + numel3

        return quantization_error, numel


def ResNet20(
    *,
    n: int = 3,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    bits: int = 4,
    **kwargs: Any,
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        bits=bits,
        **kwargs,
    )


def ResNet32(
    *,
    n: int = 5,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    bits: int = 4,
    **kwargs: Any,
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        bits=bits,
        **kwargs,
    )


def ResNet44(
    *,
    n: int = 7,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    bits: int = 4,
    **kwargs: Any,
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        bits=bits,
        **kwargs,
    )


def ResNet56(
    *,
    n: int = 9,
    num_classes: int = 10,
    quantize_fn: Optional[Callable] = None,
    bits: int = 4,
    **kwargs: Any,
) -> ResNet:
    return ResNet(
        block=BasicBlock,
        num_blocks=[n, n, n],
        num_filters=[16, 32, 64],
        num_classes=num_classes,
        quantize_fn=quantize_fn,
        bits=bits,
        **kwargs,
    )
