import os
import random
from copy import deepcopy
from typing import Callable

import fire
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from models.resnet import ResNet20, ResNet32, ResNet44, ResNet56
from utils.dataloaders import get_dataloaders
from utils.quantizers import (
    LinearPowerOfTwoPlusQuantizer,
    LinearPowerOfTwoQuantizer,
    PowerOfTwoPlusQuantizer,
    PowerOfTwoQuantizer,
)


def quantize_model(
    model_type: str, model: nn.Module, quantizer: Callable, bits: int = 4
) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "resnet" in model_type:
                if "conv" in name and "layer" in name:
                    quantized_param = quantizer.forward(None, param, bits=bits)
                    param.copy_(quantized_param)


def test_model(model: nn.Module, test_loader: DataLoader, device: str) -> None:
    correct, total = 0, 0
    model.eval()

    with torch.no_grad():
        test_loader.sampler.set_epoch(0)
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    test_acc = correct / total
    test_acc_tensor = torch.tensor(test_acc, device=device)
    dist.all_reduce(test_acc_tensor, op=dist.ReduceOp.SUM)
    test_acc = test_acc_tensor.item() / dist.get_world_size()
    return test_acc


def load_model(
    model_type: str, quantize_fn: Callable, bits: int, model_path: str, device: str
) -> nn.Module:
    if model_type == "resnet20":
        model = ResNet20(quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet32":
        model = ResNet32(quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet44":
        model = ResNet44(quantize_fn=quantize_fn, bits=bits)
    elif model_type == "resnet56":
        model = ResNet56(quantize_fn=quantize_fn, bits=bits)

    model = model.to(device)
    model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def main(
    model_type: str,
    dataset: str,
    batch_size: int = 128,
    seed: int = 8,
    data_dir: str = "./data",
    results_dir: str = "./results",
) -> None:
    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= 4
    ), "invalid hardware"

    assert model_type in [
        "resnet20",
        "resnet32",
        "resnet44",
        "resnet56",
    ], "invalid model type"
    assert dataset in ["cifar", "imagenet"], "invalid dataset"

    quantizer_dict = {
        "lin": LinearPowerOfTwoQuantizer,
        "lin+": LinearPowerOfTwoPlusQuantizer,
        "po2": PowerOfTwoQuantizer,
        "po2+": PowerOfTwoPlusQuantizer,
    }
    bits_to_try = [2, 3, 4]

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print("initializing distributed environment 😇")
    dist.init_process_group(backend="nccl")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda", local_rank)

    _, test_loader = get_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        distributed=True,
    )

    full_precision = f"{model_type}_{dataset}_full_precision"
    model = load_model(
        model_type=model_type,
        quantize_fn=None,
        bits=4,
        model_path=f"{results_dir}/{full_precision}.pth",
        device=device,
    )
    test_results = []

    # Full Precision Training
    test_acc = test_model(model=model, test_loader=test_loader, device=device)
    if local_rank == 0:
        print(f"full_precision = {100 * test_acc:.2f}%")
    test_results.append(("full_precision", test_acc))

    # Post Training Quantization
    for quantizer_type, quantizer in quantizer_dict.items():
        for bits in bits_to_try:
            model_copy = deepcopy(model)
            quantize_model(model_type, model_copy, quantizer, bits)
            test_acc = test_model(
                model=model_copy, test_loader=test_loader, device=device
            )
            if local_rank == 0:
                print(f"ptq_{quantizer_type}_{bits} = {100 * test_acc:.2f}%")
                test_results.append((f"ptq_{quantizer_type}_{bits}", test_acc))

    # Quantization Aware Training
    # for quantizer_type, quantizer in quantizer_dict.items():
    #     for bits in bits_to_try:
    #         train_config = f"{model_type}_{dataset}_{quantizer_type}_{bits}"
    #         model = load_model(
    #             model_type=model_type,
    #             quantize_fn=quantizer,
    #             bits=bits,
    #             model_path=f"{results_dir}/{train_config}.pth",
    #         )
    #         test_acc = test_model(model=model, test_loader=test_loader, device=device)
    #         if local_rank == 0:
    #             print(f"qat_{quantizer_type}_{bits} = {100 * test_acc:.2f}%")
    #         test_results.append((f"qat_{quantizer_type}_{bits}", test_acc))

    # if local_rank == 0:
    #     with open(f"{results_dir}/{model_type}_{dataset}_results.csv", mode="w") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(["model", "test_acc"])
    #         writer.writerows(test_results)


if __name__ == "__main__":
    fire.Fire(main)
