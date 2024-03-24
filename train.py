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
from dataclasses import dataclass
from typing import List, Optional

from dataloading import get_cifar_dataloaders
from quantizers import quantize_model, PowerOfTwoQuantizer, PowerOfTwoPlusQuantizer
from train_state import load_dict, save_dict


@dataclass
class TrainResult:

    losses: List[float]
    accuracies: List[float]
    quantization_errors: Optional[List[float]] = None


def train_model(
    model,
    model_name,
    num_epochs=164,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0001,
    save_model=True,
):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    trainloader, _ = get_cifar_dataloaders(dir=".", batch_size=128, num_workers=2)
    model.to(device)
    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    iter = 0
    losses = []
    accuracies = []
    criterion = nn.CrossEntropyLoss()
    quantization_errors = []

    # terminate at 64k iterations or 164 epochs: each epoch has 391 iterations (50048/128)
    for epoch in trange(num_epochs):

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

            # adjust learning rate at specified iterations
            iter += 1
            if iter == 32000 or iter == 48000:
                for param_group in optimizer.param_groups:
                    param_group["lr"] /= 10

        losses.append(running_loss / len(trainloader))
        accuracies.append(correct / samples)
        total_quantization_error, numel = model.get_quantization_error()
        quantization_errors.append(total_quantization_error.item() / numel)

    if save_model:
        torch.save(model.state_dict(), f"{dir}/{model_name}_cifar10.pth")
    return TrainResult(
        losses=losses, accuracies=accuracies, quantization_errors=quantization_errors
    )


def compute_test(model):

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    _, testloader = get_cifar_dataloaders(dir=".", batch_size=128, num_workers=2)

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


def quantize_loop(model: nn.Module, bits_to_try: List[int], quantizer):
    accuracies = []
    for bit in bits_to_try:
        model_copy = deepcopy(model)
        quantize_model(model_copy, quantizer, 1, bit - 1)
        accuracies.append(compute_test(model_copy))

    return accuracies


if __name__ == "__main__":
    
    # PTQ
    
    state_dict = load_dict()
    bits_to_try = [2, 3, 4, 5, 8]
    
    for model_name, model_dict in state_dict.items():

        print(f"Quantizing {model_name} to {bits_to_try} bits")

        model = model_dict["model"]
        test_acc_po2 = quantize_loop(model, bits_to_try, PowerOfTwoQuantizer)
        test_acc_po2_plus = quantize_loop(model, bits_to_try, PowerOfTwoPlusQuantizer)

        model_dict["test_acc_po2"] = test_acc_po2
        model_dict["test_acc_po2+"] = test_acc_po2_plus
        model_dict["improvement"] = [(new - old) / old for new, old in zip(test_acc_po2_plus, test_acc_po2)]

        test_acc = model_dict["test_acc"]
        test_acc_po2 = " ".join([str(round(100 * acc, 2)) for acc in test_acc_po2])
        test_acc_po2_plus = " ".join([str(round(100 * acc, 2)) for acc in test_acc_po2_plus])
        improvement = " ".join([str(round(100 * imp, 2)) for imp in model_dict["improvement"]])
        print(f"\t{'test_acc':<15} = {test_acc}")
        print(f"\t{'test_acc_po2':<15} = {test_acc_po2}")
        print(f"\t{'test_acc_po2+':<15} = {test_acc_po2_plus}")
        print(f"\t{'improvement':<15} = {improvement}")
