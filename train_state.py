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
from quantizers import (
    PowerOfTwoQuantizer,
    PowerOfTwoPlusQuantizer,
    LinearPowerOfTwoQuantizer,
    LinearPowerOfTwoPlusQuantizer,
)


def save_dict(state_dict, dir):
    print("saving state_dict...")
    dict_to_save = {
        k: {x: y for x, y in v.items() if x != "model"} for k, v in state_dict.items()
    }
    with open(f"{dir}/state_dict.pkl", "wb") as f:
        pickle.dump(dict_to_save, f)


def load_dict(dir, device, bits_to_try):
    if os.path.exists(f"{dir}/state_dict.pkl"):
        print(f"loading state_dict {dir}")
        with open(f"{dir}/state_dict.pkl", "rb") as f:
            state_dict = pickle.load(f)
    else:
        state_dict = collections.defaultdict(lambda: collections.defaultdict(dict))

    base_models = {
        "resnet20": resnet20,
        "resnet32": resnet32,
        "resnet44": resnet44,
        "resnet56": resnet56,
    }

    quantizers = {
        "po2": PowerOfTwoQuantizer,
        "po2+": PowerOfTwoPlusQuantizer,
        "linear": LinearPowerOfTwoQuantizer,
        "linear+": LinearPowerOfTwoPlusQuantizer,
    }

    for model_name, model_fn in base_models.items():
        state_dict[model_name]["model"] = model_fn()
        state_dict[model_name]["is_quantized"] = False

    for bits in bits_to_try:
        for base_model_name, model_fn in base_models.items():
            for quantizer_name, quantizer in quantizers.items():
                state_dict[f"{base_model_name}_{quantizer_name}_{bits}"]["model"] = (
                    model_fn(quantize_fn=quantizer, bits=bits)
                )
                state_dict[f"{base_model_name}_{quantizer_name}_{bits}"][
                    "fp_model"
                ] = base_model_name
                state_dict[f"{base_model_name}_{quantizer_name}_{bits}"][
                    "is_quantized"
                ] = True
                state_dict[f"{base_model_name}_{quantizer_name}_{bits}"]["bits"] = bits

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
