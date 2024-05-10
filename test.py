import csv
import glob
import random
from copy import deepcopy
from pathlib import Path

import fire
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.model import get_model
from utils.dataloaders import get_dataloaders
from utils.quantizers import (
    quantize_model,
    quantizer_dict,
)

bits_to_try = [3, 4]


def init() -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def test_model(model: nn.Module, test_loader: DataLoader, device: str) -> float:
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


def main(
    model_type: str,
    dataset: str,
    batch_size: int = 128,
    data_dir: str = "./data",
    train_dir: str = "./train",
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

    device = torch.device("cuda", 0)

    _, test_loader, image_size = get_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        distributed=False,
    )

    num_classes = len(test_loader.dataset.classes)

    work_dir = f"{train_dir}/{dataset}/{model_type}"
    results_work_dir = f"{results_dir}/{dataset}/{model_type}"
    Path(results_work_dir).mkdir(parents=True, exist_ok=True)
    seeds = [seed.split("/")[-1] for seed in glob.glob(f"{work_dir}/*")]

    for seed in seeds:
        test_results = []

        model = get_model(
            model_type=model_type,
            num_classes=num_classes,
            quantize_fn=None,
            bits=4,
            image_size=image_size,
        )
        model.to(device)
        load_distributed_state_dict(
            model=model,
            model_path=f"{work_dir}/{seed}/model_state/full_precision.pth",
        )

        # Full Precision Training
        acc = test_model(model=model, test_loader=test_loader, device=device)
        print(f"full_precision = {acc * 100:.2f}%, q_error = 0.0")
        test_results.append(("full_precision", acc, 0.0))

        # Post Training Quantization
        for quant_type, quantizer in quantizer_dict.items():
            for bits in bits_to_try:
                model_copy = deepcopy(model)
                quant_error = quantize_model(
                    model=model_copy, quantizer=quantizer, bits=bits
                )
                acc = test_model(
                    model=model_copy, test_loader=test_loader, device=device
                )
                test_results.append((f"ptq_{quant_type}_{bits}", acc, quant_error))
                print(
                    f"ptq_{quant_type}_{bits} = {acc * 100:.2f}%, q_error = {quant_error:.10f}"
                )

        # Quantization Aware Training
        if not skip_qat:
            for quant_type, quantizer in quantizer_dict.items():
                for bits in bits_to_try:
                    train_config = f"{quant_type}_{bits}"
                    model = get_model(
                        model_type=model_type,
                        num_classes=num_classes,
                        quantize_fn=quantizer,
                        bits=bits,
                        image_size=image_size,
                    )
                    model.to(device)
                    load_distributed_state_dict(
                        model=model,
                        model_path=f"{work_dir}/{seed}/model_state/{train_config}.pth",
                    )
                    acc = test_model(
                        model=model, test_loader=test_loader, device=device
                    )

                    # get average quant_error during QAT
                    df = pd.read_csv(f"{work_dir}/{seed}/{train_config}.csv")
                    quant_error = df["quantization_error"].mean()
                    test_results.append((f"qat_{quant_type}_{bits}", acc, quant_error))
                    print(
                        f"qat_{quant_type}_{bits} = {acc * 100:.2f}%, q_error = {quant_error:.10f}"
                    )

        with open(f"{results_work_dir}/{seed}.csv", mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["model", "test_acc", "quantization_error"])
            writer.writerows(test_results)


if __name__ == "__main__":
    fire.Fire(main)
