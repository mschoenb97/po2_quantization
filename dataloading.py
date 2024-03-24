import torch
import torch.nn as nn
from torch import Tensor

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
import torch.nn.functional as F

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

DOWNLOAD = False


def get_cifar_dataloaders(dir: str, batch_size: int, num_workers: int, test=False):

    batch_size = 128
    num_workers = 2

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=f"{dir}/data", train=True, download=DOWNLOAD, transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=f"{dir}/data", train=False, download=DOWNLOAD, transform=test_transform
    )

    if test:
        test_subset_size = 10
        # For testing, sample a subset for both train and test
        train_indices = torch.randperm(len(trainset))[:test_subset_size]
        test_indices = torch.randperm(len(testset))[:test_subset_size]

        trainset = Subset(trainset, train_indices)
        testset = Subset(testset, test_indices)

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return trainloader, testloader
