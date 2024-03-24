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

from models import resnet20, resnet32, resnet44, resnet56
from quantizers import PowerOfTwoQuantizer, PowerOfTwoPlusQuantizer


def save_dict(state_dict, dir):
    print("saving state_dict...")
    dict_to_save = {
        k: {x: y for x, y in v.items() if x != "model"} for k, v in state_dict.items()
    }
    with open(f"{dir}/state_dict.pkl", "wb") as f:
        pickle.dump(dict_to_save, f)


def load_dict(dir, device):
    if os.path.exists(f"{dir}/state_dict.pkl"):
        print(f"loading state_dict {dir}")
        with open(f"{dir}/state_dict.pkl", "rb") as f:
            state_dict = pickle.load(f)
    else:
        state_dict = collections.defaultdict(dict)

    state_dict["resnet20"]["model"] = resnet20()
    state_dict["resnet32"]["model"] = resnet32()
    state_dict["resnet44"]["model"] = resnet44()
    state_dict["resnet56"]["model"] = resnet56()

    # QAT: 8 bit
    state_dict["resnet20_po2_ptq8"]["model"] = resnet20(
        quantize_fn=PowerOfTwoQuantizer, fsr=1, bitwidth=7
    )
    state_dict["resnet20_po2+_ptq8"]["model"] = resnet20(
        quantize_fn=PowerOfTwoPlusQuantizer, fsr=1, bitwidth=7
    )
    state_dict["resnet32_po2_ptq8"]["model"] = resnet32(
        quantize_fn=PowerOfTwoQuantizer, fsr=1, bitwidth=7
    )
    state_dict["resnet32_po2+_ptq8"]["model"] = resnet32(
        quantize_fn=PowerOfTwoPlusQuantizer, fsr=1, bitwidth=7
    )
    state_dict["resnet44_po2_ptq8"]["model"] = resnet44(
        quantize_fn=PowerOfTwoQuantizer, fsr=1, bitwidth=7
    )
    state_dict["resnet44_po2+_ptq8"]["model"] = resnet44(
        quantize_fn=PowerOfTwoPlusQuantizer, fsr=1, bitwidth=7
    )
    state_dict["resnet56_po2_ptq8"]["model"] = resnet56(
        quantize_fn=PowerOfTwoQuantizer, fsr=1, bitwidth=7
    )
    state_dict["resnet56_po2+_ptq8"]["model"] = resnet56(
        quantize_fn=PowerOfTwoPlusQuantizer, fsr=1, bitwidth=7
    )

    for model_name, model_dict in state_dict.items():
        model_dict["trained"] = False
        model_path = f"{dir}/{model_name}_cifar10.pth"
        if os.path.exists(model_path):
            print(f"loading model {model_name}")
            model = model_dict["model"]
            model.load_state_dict(
                torch.load(model_path, map_location=torch.device(device))
            )
            model.to(device)
            model_dict["trained"] = True

    return state_dict
