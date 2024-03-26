import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from typing import List, Optional
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Union, Callable
from tqdm import tqdm
import argparse
import random
import numpy as np

from dataloading import get_cifar_dataloaders
from quantizers import quantize_model, PowerOfTwoQuantizer, PowerOfTwoPlusQuantizer
from train_state import load_dict, save_dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
bits_to_try = [2, 3, 4]
test_run = False


@dataclass
class TrainResult:

    losses: List[float]
    accuracies: List[float]
    quantization_errors: Optional[List[float]] = None


def train_model(
    model: nn.Module,
    model_name: str,
    dir: str,
    trainloader: DataLoader,
    num_epochs: int = 164,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    save_model: bool = True,
    percent_warmup_epochs: float = 0.1,
) -> TrainResult:

    model.to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    losses = []
    accuracies = []
    criterion = nn.CrossEntropyLoss()
    quantization_errors = []

    warmup_epochs = int(percent_warmup_epochs * num_epochs)

    # calculate lienar warmup steps
    warmup_steps = warmup_epochs * len(trainloader)
    lambda1 = lambda step: step / warmup_steps if step < warmup_steps else 1
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda1)

    # setup cosine annealing scheduler after the warmup
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=0
    )

    # terminate at 64k iterations or 164 epochs: each epoch has 391 iterations (50048/128)
    for epoch in tqdm(range(num_epochs)):

        running_loss = 0.0
        correct = 0
        samples = 0
        model.train()

        for images, labels in trainloader:
            inputs, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item

            # calculate accuracy
            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()
                samples += labels.size(0)

            # backward pass
            loss.backward()
            optimizer.step()
            if epoch < warmup_epochs:
                scheduler_warmup.step()
            else:
                # Only step the cosine scheduler after warmup
                scheduler_cosine.step()

        losses.append(running_loss / len(trainloader))
        accuracies.append(correct / samples)
        total_quantization_error, numel = model.get_quantization_error()
        quantization_errors.append(total_quantization_error / numel)

    if save_model:
        torch.save(model.state_dict(), f"{dir}/{model_name}_cifar10.pth")
    return TrainResult(
        losses=losses, accuracies=accuracies, quantization_errors=quantization_errors
    )


def compute_test(model: nn.Module, testloader: DataLoader) -> float:

    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return correct / total


def quantize_loop(
    state_dict: Dict[str, Dict[str, Union[nn.Module, List[int], str]]],
    model_name: str,
    quantizer: Callable,
    testloader: DataLoader,
) -> List[float]:

    fp_model_name = state_dict[model_name]["fp_model"]
    model = state_dict[fp_model_name]["model"]
    bits = state_dict[model_name]["bits"]

    accuracies = []
    model_copy = deepcopy(model)
    quantize_model(model_copy, quantizer, 1, bits)
    accuracies.append(compute_test(model_copy, testloader))

    return accuracies


def _train_models(
    state_dict: Dict[str, Dict[str, Union[nn.Module, List[int], bool, str]]],
    dir: str,
    trainloader: DataLoader,
    testloader: DataLoader,
    qat: bool,
) -> None:

    for model_name, model_dict in state_dict.items():

        if model_dict["is_quantized"] != qat:
            continue

        model = model_dict["model"]

        if model_dict["trained"]:
            print(f"{model_name} already trained, skipping...")
        else:
            print(f"training {model_name}...")
            num_epochs = 1 if test_run else 164
            train_res = train_model(
                model, model_name, dir, trainloader, num_epochs=num_epochs
            )
            model_dict["train_loss"] = train_res.losses
            model_dict["train_acc"] = train_res.accuracies
            model_dict["train_quantization_error"] = train_res.quantization_errors

        test_acc = compute_test(model, testloader)
        model_dict["test_acc"] = test_acc
        print(f"{model_name} test accuracy: {test_acc:.4f}")


def train_models(
    state_dict: Dict[str, Dict[str, Union[nn.Module, List[int], bool, str]]],
    dir: str,
    trainloader: DataLoader,
    testloader: DataLoader,
) -> None:

    # perform normal full precision training
    _train_models(state_dict, dir, trainloader, testloader, qat=False)

    # copy fp_model params to quantized model
    for model_dict in state_dict.values():
        if model_dict["quantized"]:
            fp_model_name = model_dict["fp_model"]
            fp_model = state_dict[fp_model_name]["model"]
            model_dict["model"].load_state_dict(fp_model.state_dict())

    # perform quantize aware training
    _train_models(state_dict, dir, trainloader, testloader, qat=True)
    save_dict(state_dict, dir)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train and quantize models")
    parser.add_argument("--test", action="store_true", help="run test")
    parser.add_argument("--dir", type=str, default=".", help="directory to save models")
    args = parser.parse_args()
    test_run = args.test
    dir = args.dir

    if test_run:
        print("Performing test run")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    state_dict = load_dict(dir, device, bits_to_try)
    trainloader, testloader = get_cifar_dataloaders(
        dir=".", batch_size=128, num_workers=2, test_run=test_run
    )

    # QAT: Quantize Aware Training

    train_models(state_dict, dir, trainloader, testloader)

    # PTQ: Post Training Quantization

    for model_name, model_dict in state_dict.items():

        if not model_dict["is_quantized"]:
            continue

        print(f"Quantizing {model_name} to {bits_to_try} bits")

        model = model_dict["model"]
        test_acc_po2 = quantize_loop(
            state_dict, model_name, PowerOfTwoQuantizer, testloader
        )
        test_acc_po2_plus = quantize_loop(
            state_dict, model_name, PowerOfTwoPlusQuantizer, testloader
        )

        model_dict["test_acc_po2"] = test_acc_po2
        model_dict["test_acc_po2+"] = test_acc_po2_plus

        model_dict["improvement"] = [
            (lambda new, old: (new - old) / old if old != 0 else 0)(new, old)
            for new, old in zip(test_acc_po2_plus, test_acc_po2)
        ]

        test_acc = model_dict["test_acc"]
        test_acc_po2 = " ".join([str(round(100 * acc, 2)) for acc in test_acc_po2])
        test_acc_po2_plus = " ".join(
            [str(round(100 * acc, 2)) for acc in test_acc_po2_plus]
        )
        improvement = " ".join(
            [str(round(100 * imp, 2)) for imp in model_dict["improvement"]]
        )
        print(f"\t{'test_acc':<15} = {test_acc}")
        print(f"\t{'test_acc_po2':<15} = {test_acc_po2}")
        print(f"\t{'test_acc_po2+':<15} = {test_acc_po2_plus}")
        print(f"\t{'improvement':<15} = {improvement}")
