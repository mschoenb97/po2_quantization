import torch
import torch.nn as nn


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
