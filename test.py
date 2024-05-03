import csv
import random
from copy import deepcopy
from typing import Callable

import fire
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.mobile_vit import MobileVIT
from models.mobilenet import MobileNetV2
from models.resnet import ResNet20, ResNet32, ResNet44, ResNet56
from utils.dataloaders import get_dataloaders
from utils.quantizers import (
    LinearPowerOfTwoPlusQuantizer,
    LinearPowerOfTwoQuantizer,
    PowerOfTwoPlusQuantizer,
    PowerOfTwoQuantizer,
)


def quantize_model(model, quantizer, bits: int = 4):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "conv" in name and "layer" in name:
                quantized_param = quantizer.forward(None, param, bits=bits)
                param.copy_(quantized_param)


def test_model(model: nn.Module, test_loader: DataLoader, device: str) -> None:
    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def load_distributed_state_dict(model: nn.Module, model_path: str) -> None:
    state_dict = torch.load(model_path)
    state_dict = {
        key.replace("module.", ""): value for key, value in state_dict.items()
    }
    model.load_state_dict(state_dict)


def get_model(model_type: str, num_classes: int, quantize_fn: Callable, bits: int):
    if model_type == "resnet20":
        model = ResNet20(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet32":
        model = ResNet32(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet44":
        model = ResNet44(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet56":
        model = ResNet56(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "mobilenet":
        model = MobileNetV2(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)
    elif model_type == "mobilevit":
        model = MobileVIT(num_classes=num_classes, quantize_fn=quantize_fn, bits=bits)

    return model


def main(
    model_type: str,
    dataset: str,
    batch_size: int = 128,
    seed: int = 8,
    data_dir: str = "./data",
    results_dir: str = "./results",
    skip_qat: bool = False,
) -> None:
    assert torch.cuda.is_available(), "invalid hardware"

    assert model_type in [
        "resnet20",
        "resnet32",
        "resnet44",
        "resnet56",
        "mobilenet",
        "mobilevit",
    ], "invalid model type"
    assert dataset in ["cifar", "imagenet"], "invalid dataset"

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda", 0)

    quantizer_dict = {
        "lin": LinearPowerOfTwoQuantizer,
        "lin+": LinearPowerOfTwoPlusQuantizer,
        "po2": PowerOfTwoQuantizer,
        "po2+": PowerOfTwoPlusQuantizer,
    }
    bits_to_try = [2, 3, 4]

    _, test_loader = get_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        distributed=False,
    )

    num_classes = len(test_loader.dataset.classes)

    # Post Training Quantization: load full precision model and quantize weights
    full_precision = f"{model_type}_{dataset}_full_precision"

    model = get_model(
        model_type=model_type, num_classes=num_classes, quantize_fn=None, bits=4
    )
    model.to(device)
    load_distributed_state_dict(
        model=model, model_path=f"{results_dir}/{full_precision}.pth"
    )

    test_results = []

    # Full Precision Training
    test_acc = test_model(model=model, test_loader=test_loader, device=device)
    print(f"full_precision = {test_acc * 100:.2f}%")
    test_results.append(("full_precision", test_acc))

    # Post Training Quantization
    for quantizer_type, quantizer in quantizer_dict.items():
        for bits in bits_to_try:
            model_copy = deepcopy(model)
            quantize_model(model_copy, quantizer, bits)
            test_acc = test_model(
                model=model_copy, test_loader=test_loader, device=device
            )
            test_results.append((f"ptq_{quantizer_type}_{bits}", test_acc))
            print(f"ptq_{quantizer_type}_{bits} = {test_acc * 100:.2f}%")

    # Quantization Aware Training
    if not skip_qat:
        for quantizer_type, quantizer in quantizer_dict.items():
            for bits in bits_to_try:
                train_config = f"{model_type}_{dataset}_{quantizer_type}_{bits}"
                model = get_model(
                    model_type=model_type,
                    num_classes=num_classes,
                    quantize_fn=quantizer,
                    bits=bits,
                )
                model.to(device)
                load_distributed_state_dict(
                    model=model, model_path=f"{results_dir}/{train_config}.pth"
                )
                test_acc = test_model(
                    model=model, test_loader=test_loader, device=device
                )
                test_results.append((f"qat_{quantizer_type}_{bits}", test_acc))
                print(f"qat_{quantizer_type}_{bits} = {test_acc * 100:.2f}%")

    with open(f"{results_dir}/{model_type}_{dataset}_results.csv", mode="w") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "test_acc"])
        writer.writerows(test_results)


if __name__ == "__main__":
    fire.Fire(main)
