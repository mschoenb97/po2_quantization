import torch
import torch.nn as nn
from torch import Tensor

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR

from functools import partial
from typing import Any, Callable, List, Optional, Type, Union, Tuple
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
from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm

from dataloading import get_cifar_dataloaders
from quantizers import quantize_model, PowerOfTwoQuantizer, PowerOfTwoPlusQuantizer
from train_state import load_dict, save_dict

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@dataclass
class TrainResult:

    losses: List[float]
    accuracies: List[float]
    quantization_errors: Optional[List[float]] = None


def train_model(
    model,
    model_name,
    dir,
    num_epochs=164,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    save_model=True,
    percent_warmup_epochs=0.1,
    test=False,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainloader, _ = get_cifar_dataloaders(
        dir=".", batch_size=128, num_workers=2, test=test
    )
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    losses = []
    accuracies = []
    criterion = nn.CrossEntropyLoss()
    quantization_errors = []

    warmup_epochs = int(percent_warmup_epochs * num_epochs)
    # Calculate the number of steps for warmup
    warmup_steps = warmup_epochs * len(
        trainloader
    )  # Assuming train_loader is your DataLoader

    # Lambda function for linear warmup
    lambda1 = lambda step: step / warmup_steps if step < warmup_steps else 1

    # Apply linear warmup with LambdaLR
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda1)

    # Setup the cosine annealing scheduler after the warmup
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
            running_loss += loss.item()

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


def compute_test(model, test=False):

    _, testloader = get_cifar_dataloaders(
        dir=".", batch_size=128, num_workers=2, test=test
    )

    correct = 0
    total = 0
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


def quantize_loop(state_dict, model_name, quantizer, test=False):
    fp_model_name = state_dict[model_name]["fp_model"]
    model = state_dict[fp_model_name]["model"]
    bits = state_dict[model_name]["bits"]

    accuracies = []
    model_copy = deepcopy(model)
    quantize_model(model_copy, quantizer, 1, bits)
    accuracies.append(compute_test(model_copy, test=test))

    return accuracies


def _train_models(state_dict, dir, qat, test=False):

    num_epochs = 1 if test else 164

    for model_name, model_dict in state_dict.items():

        if model_dict["is_quantized"] != qat:
            continue

        model = model_dict["model"]

        if model_dict["trained"]:
            print(f"{model_name} already trained, skipping...")
        else:
            print(f"training {model_name}...")
            train_res = train_model(
                model, model_name, dir, num_epochs=num_epochs, test=test
            )
            model_dict["train_loss"] = train_res.losses
            model_dict["train_acc"] = train_res.accuracies
            model_dict["train_quantization_error"] = train_res.quantization_errors

        test_acc = compute_test(model, test=test)
        model_dict["test_acc"] = test_acc

        # test_acc = model_dict["test_acc"]
        print(f"{model_name} test accuracy: {test_acc:.4f}")


def train_models(state_dict, dir, test=False):
    _train_models(state_dict, dir, qat=False, test=test)

    for model_dict in state_dict.values():
        if model_dict["quantized"]:
            fp_model_name = model_dict["fp_model"]
            fp_model = state_dict[fp_model_name]["model"]
            # copy fp_model params to quantized model
            model_dict["model"].load_state_dict(fp_model.state_dict())

    _train_models(state_dict, dir, qat=True, test=test)

    save_dict(state_dict, dir)


if __name__ == "__main__":

    # read in test parameter with argparse
    import argparse

    parser = argparse.ArgumentParser(description="Train and quantize models")
    parser.add_argument("--test", action="store_true", help="run test")
    parser.add_argument("--dir", type=str, default=".", help="directory to save models")
    args = parser.parse_args()
    test = args.test
    dir = args.dir

    # bits_to_try = [2, 3, 4, 5, 8]
    bits_to_try = [2, 4]
    state_dict = load_dict(dir, device, bits_to_try, test=test)

    # QAT

    train_models(state_dict, dir, test=test)

    # PTQ

    for model_name, model_dict in state_dict.items():

        if not model_dict["is_quantized"]:
            continue

        print(f"Quantizing {model_name} to {bits_to_try} bits")

        model = model_dict["model"]
        test_acc_po2 = quantize_loop(
            state_dict, model_name, PowerOfTwoQuantizer, test=test
        )
        test_acc_po2_plus = quantize_loop(
            state_dict, model_name, PowerOfTwoPlusQuantizer, test=test
        )

        model_dict["test_acc_po2"] = test_acc_po2
        model_dict["test_acc_po2+"] = test_acc_po2_plus

        def improvement(new, old):
            if old == 0:
                return 0
            return (new - old) / old

        model_dict["improvement"] = [
            improvement(new, old) for new, old in zip(test_acc_po2_plus, test_acc_po2)
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

    def plot_quantize(ax, model_name, bits_to_try, start):

        plt.style.use("default")
        unquantized_acc = [100 * state_dict[model_name]["test_acc"]] * len(
            bits_to_try[start:]
        )
        quantized_acc = [
            100 * acc for acc in state_dict[model_name]["test_acc_po2"][start:]
        ]
        quantized_plus_acc = [
            100 * acc for acc in state_dict[model_name]["test_acc_po2+"][start:]
        ]

        ax.plot(
            bits_to_try[start:],
            unquantized_acc,
            label="float32",
            color="purple",
            linestyle="--",
        )
        ax.plot(bits_to_try[start:], quantized_acc, label="quantized", color="orange")
        ax.plot(
            bits_to_try[start:], quantized_plus_acc, label="quantized+", color="green"
        )

        ax.set_xlabel("Bits Used")
        ax.set_ylabel("Percent")
        ax.set_title(f"{model_name}")
        ax.set_xticks(bits_to_try[start:])
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.set_ylim(70, 95)
        ax.legend()

    # Commented these out because of a bug
    # fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    # for i, model_name in enumerate(state_dict.keys()):
    #     plot_quantize(axs[i // 2, i % 2], model_name, bits_to_try, 1)

    # for ax in axs[0, :]:
    #     ax.set_xlabel('')

    # for ax in axs[:, 1]:
    #     ax.set_ylabel('')

    # plt.tight_layout()
    # plt.savefig(f'{dir}/quantization.png', bbox_inches='tight')

    def plot_improvement(ax, model_name, bits_to_try, start):

        plt.style.use("default")
        improvement = [
            100 * acc for acc in state_dict[model_name]["test_acc_po2+"][start:]
        ]

        ax.plot(
            bits_to_try[start:],
            improvement,
            label="quantized+ improvement",
            color="orange",
        )

        ax.set_xlabel("Bits Used")
        ax.set_ylabel("Percent")
        ax.set_title(f"{model_name}")
        ax.set_xticks(bits_to_try[start:])
        ax.grid(axis="y", linestyle="--", linewidth=0.5)
        ax.set_ylim(-3, 14)
        ax.legend()

    # Commented these out because of a bug
    # fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
    # for i, model_name in enumerate(state_dict.keys()):
    #     plot_quantize(axs[i // 2, i % 2], model_name, bits_to_try, 1)

    # for ax in axs[0, :]:
    #     ax.set_xlabel('')

    # for ax in axs[:, 1]:
    #     ax.set_ylabel('')

    # plt.tight_layout()
    # plt.savefig(f'{dir}/percent_improvement.png', bbox_inches='tight')
