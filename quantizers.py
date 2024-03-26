import torch
import torch.nn as nn
from typing import List
from copy import deepcopy


class PowerOfTwoQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int = 4, fsr: int = 1):
        sign = torch.sign(input)
        scale = torch.max(torch.abs(input))
        normalized_input = input / scale
        abs_normalized_input = torch.abs(normalized_input)
        quantized_output = torch.clamp(
            torch.round(torch.log2(abs_normalized_input)),
            fsr - 2 ** (bits - 1),
            fsr - 1,
        )
        log_quant = 2**quantized_output
        return log_quant * sign * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class PowerOfTwoPlusQuantizer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int = 4, fsr: int = 1):
        sign = torch.sign(input)
        scale = torch.max(torch.abs(input))
        normalized_input = input / scale
        abs_normalized_input = torch.abs(normalized_input)
        quantized_output = torch.clamp(
            torch.round(torch.log2(abs_normalized_input / 1.5) + 0.5),
            fsr - 2 ** (bits - 1),
            fsr - 1,
        )
        log_quant = 2**quantized_output
        return log_quant * sign * scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


def quantize_per_filter(
    x: torch.Tensor, delta: torch.Tensor, bits: int
) -> torch.Tensor:
    # x: (num_channels, N, N)
    # delta: (num_channels, 1)
    scaled = torch.round(x / delta.view(-1, 1, 1))
    max = (2 ** (bits - 1)) - 1  # for 8 bits [-127, 127]
    clip = torch.clamp(scaled, min=-max, max=max)
    return delta.view(-1, 1, 1) * clip


class LinearPowerOfTwoQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int = 4, num_iters: int = 10):

        max = torch.max(
            torch.max(torch.max(input, dim=3).values, dim=2).values, dim=0
        ).values  # find the max for each filter (dim = 1)
        min = torch.min(
            torch.min(torch.min(input, dim=3).values, dim=2).values, dim=0
        ).values
        delta = (max - min) / (2**bits - 1)  # 256 - 1
        assert delta.shape[0] == input.shape[1]

        q = quantize_per_filter(input, delta, bits) / delta.view(-1, 1, 1)
        assert q.shape == input.shape

        for _ in range(num_iters):

            # unconstrained optimal delta minimizing MSQE
            qTw = torch.sum(q * input, dim=[0, 2, 3])
            qTq = torch.sum(q * q, dim=[0, 2, 3])
            delta = qTw / qTq
            assert delta.shape[0] == input.shape[1]

            delta = 2 ** torch.round(
                torch.log2(delta)
            )  # constrain delta to be power of 2
            assert delta.shape[0] == input.shape[1]

            q = quantize_per_filter(input, delta, bits) / delta.view(
                -1, 1, 1
            )  # quantize input with new delta
            assert q.shape == input.shape

        return q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


class LinearPowerOfTwoPlusQuantizer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, bits: int = 4, num_iters: int = 10):

        max = torch.max(
            torch.max(torch.max(input, dim=3).values, dim=2).values, dim=0
        ).values  # find the max for each filter (dim = 1)
        min = torch.min(
            torch.min(torch.min(input, dim=3).values, dim=2).values, dim=0
        ).values
        delta = (max - min) / (2**bits - 1)  # 256 - 1
        assert delta.shape[0] == input.shape[1]

        q = quantize_per_filter(input, delta, bits) / delta.view(-1, 1, 1)
        assert q.shape == input.shape

        for _ in range(num_iters):

            # unconstrained optimal delta minimizing MSQE
            qTw = torch.sum(q * input, dim=[0, 2, 3])
            qTq = torch.sum(q * q, dim=[0, 2, 3])
            delta = qTw / qTq
            assert delta.shape[0] == input.shape[1]

            delta = 2 ** torch.round(
                torch.log2(torch.sqrt(torch.tensor(8.0 / 9.0)) * delta)
            )  # improved po2+ quantizer
            assert delta.shape[0] == input.shape[1]

            q = quantize_per_filter(input, delta, bits) / delta.view(
                -1, 1, 1
            )  # quantize input with new delta
            assert q.shape == input.shape

        return q

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None


if __name__ == "__main__":

    x = torch.randn(1, 1_000, 1_000)
    bits_to_try = [2, 3, 4, 5, 8]
    for bit in bits_to_try:
        mse_before = (
            torch.sum((PowerOfTwoQuantizer.forward(None, x, 1, bit) - x) ** 2)
            / x.numel()
        ).item()
        mse_after = (
            torch.sum((PowerOfTwoPlusQuantizer.forward(None, x, 1, bit) - x) ** 2)
            / x.numel()
        ).item()
        percent_improvement = 100 * (mse_before - mse_after) / mse_before
        print(
            f"Using {bit} bits, {mse_before = :.10f}, {mse_after = :.10f}, {percent_improvement = :.5f}%"
        )


def quantize_model(model, quantizer, fsr: int = 1, bits: int = 4):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "conv" in name and "layer" in name:
                quantized_param = quantizer.forward(None, param, bits=bits)
                param.copy_(quantized_param)


def quantize_loop(model: nn.Module, bits_to_try: List[int], quantizer):
    accuracies = []
    for bit in bits_to_try:
        model_copy = deepcopy(model)
        quantize_model(model_copy, quantizer, 1, bit - 1)
        accuracies.append(compute_test(model_copy))

    return accuracies
