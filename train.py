import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from typing import List, Tuple, Optional, Callable
import random
import numpy as np
import fire
import os
import csv

from utils.dataloaders import get_dataloaders
from utils.quantizers import (
    PowerOfTwoQuantizer,
    PowerOfTwoPlusQuantizer,
    LinearPowerOfTwoQuantizer,
    LinearPowerOfTwoPlusQuantizer,
)
from models.resnet import ResNet20, ResNet32, ResNet44, ResNet56


def run_train_loop(
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    model_path: str,
    num_epochs: int = 164,
    lr: float = 0.1,
    momentum: float = 0.9,
    weight_decay: float = 0.0001,
    percent_warmup_epochs: float = 0.1,
) -> List[Tuple[int, float, float, float]]:

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()
    warmup_epochs = int(percent_warmup_epochs * num_epochs)
    warmup_steps = warmup_epochs * len(train_loader)
    lambda1 = lambda step: step / warmup_steps if step < warmup_steps else 1
    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lambda1)
    scheduler_cosine = CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=0
    )

    train_results = []
    dist.barrier()

    # terminate at 64k iterations or 164 epochs: each epoch has 391 iterations
    for epoch in range(num_epochs):

        # set the epoch ID for the DistributedSampler
        train_loader.sampler.set_epoch(epoch)
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        model.train()

        for images, labels in train_loader:
            inputs, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                total_correct += (predicted == labels).sum().item()
                total_samples += labels.size(0)

            loss.backward()
            optimizer.step()
            if epoch < warmup_epochs:
                scheduler_warmup.step()
            else:
                scheduler_cosine.step()

        # convert scalar values to tensors
        total_loss_tensor = torch.tensor(total_loss, device=device)
        total_correct_tensor = torch.tensor(total_correct, device=device)
        total_samples_tensor = torch.tensor(total_samples, device=device)

        # sum values across all processes
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)

        train_loss = total_loss_tensor.item() / (
        len(train_loader) * dist.get_world_size()
        )
        train_acc = total_correct_tensor.item() / total_samples_tensor.item()

        total_quantization_error, numel = model.module.get_quantization_error()
        quantization_error = total_quantization_error / numel

        if int(os.environ["LOCAL_RANK"]) == 0:
            print(
                f"epoch: {epoch}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}, quantization_error: {quantization_error:.4f}"
            )
        train_results.append((epoch, train_loss, train_acc, quantization_error))

    torch.save(model.state_dict(), model_path)

    return train_results


def train_model(
    model_type: str,
    quantizer: Optional[Callable[..., None]],
    bits: int,
    num_epochs: int,
    train_loader: DataLoader,
    model_path: str,
    full_precision_model_path: str,
) -> List[Tuple[int, float, float, float]]:

    if model_type == "resnet20":
        model = ResNet20(quantize_fn=quantizer, bits=bits)
    elif model_type == "resnet32":
        model = ResNet32(quantize_fn=quantizer, bits=bits)
    elif model_type == "resnet44":
        model = ResNet44(quantize_fn=quantizer, bits=bits)
    elif model_type == "resnet56":
        model = ResNet56(quantize_fn=quantizer, bits=bits)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(local_rank)  # set different seed for each process
    device = torch.device("cuda", local_rank)
    print(f"{device = } 👨‍💻")
    
    model = model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # if quantize aware training, copy full precision model weights
    if quantizer is not None:
        assert os.path.exists(full_precision_model_path), "QAT requires full precision model"
        model.load_state_dict(torch.load(full_precision_model_path))

    # perform full precision training or QAT
    train_res = run_train_loop(
            model=model, device=device, train_loader=train_loader, num_epochs=num_epochs, model_path=model_path
        )
    return train_res


def main(model_type: str, dataset: str, quantizer_type: str, bits: int, num_epochs: int = 164, batch_size: int = 128, seed: int = 8, data_dir: str = "./data", results_dir: str = "./results") -> None:

    """
    Training Configurations:
    - model_type: model to train (ie. resnet20, mobilenet)
    - dataset: dataset to train on (ie. cifar, imagenet)
    - quantizer_type: quantizer to use for QAT (ie. none, lin, lin+, po2, po2+)
    - bits: number of bits to quantize to (ie. 2, 3, 4)
    """
    
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 4, "invalid hardware"
    assert model_type in ["resnet20", "resnet32", "resnet44", "resnet56"], "invalid model type"
    assert dataset in ["cifar", "imagenet"], "invalid dataset"
    assert quantizer_type in ["none", "lin", "lin+", "po2", "po2+"], "invalid quantizer type"
    assert bits in [2, 3, 4], "invalid number of bits"

    config = f"{model_type}_{dataset}_full_precision" if quantizer_type == "none" else f"{model_type}_{dataset}_{quantizer_type}_{bits}bits"
    full_precision_model_path = f"{results_dir}/{model_type}_{dataset}_full_precision.pth"

    quantizer = {
        "none" : None,
        "lin" : LinearPowerOfTwoQuantizer,
        "lin+" : LinearPowerOfTwoPlusQuantizer,
        "po2" : PowerOfTwoQuantizer,
        "po2+" : PowerOfTwoPlusQuantizer,
    }[quantizer_type]
    
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

    train_loader, _ = get_dataloaders(
        dataset=dataset, data_dir=data_dir, batch_size=batch_size, num_workers=2
    )

    train_results = train_model(quantizer=quantizer, bits=bits, model_type=model_type, num_epochs=num_epochs, train_loader=train_loader, model_path=f"{results_dir}/{config}.pth", full_precision_model_path=full_precision_model_path)

    if int(os.environ["LOCAL_RANK"]) == 0:
        with open(f"{results_dir}/{config}.csv", mode='w') as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "quantization_error"])
            writer.writerows(train_results)

    if local_rank == 0:
        print("destroying distributed environment 😈")
    dist.destroy_process_group()

if __name__ == "__main__":
    fire.Fire(main)
