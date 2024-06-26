import csv
import os
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import fire
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import LambdaLR, MultiStepLR
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from models.model import get_model
from utils.dataloaders import get_dataloaders
from utils.quantizers import (
    quantizer_dict,
)


def init(seed: int) -> None:
    dist.init_process_group(backend="nccl")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def run_train_loop(
    model: nn.Module,
    device: str,
    train_loader: DataLoader,
    model_path: str,
    num_epochs: int,
    lr: float,
    train_acc: MulticlassAccuracy,
    momentum: float = 0.9,
    weight_decay: float = 1e-4,
    percent_warmup_epochs: float = 0.1,
) -> List[Tuple[int, float, float, float]]:
    local_rank = int(os.environ["LOCAL_RANK"])

    # scale learning rate to accomodate for larger effective batch_size
    lr *= dist.get_world_size()
    warmup_epochs = int(percent_warmup_epochs * num_epochs)

    optimizer = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )
    criterion = nn.CrossEntropyLoss()

    def lr_lambda(epoch):
        return (epoch + 1) / (warmup_epochs + 1)

    scheduler_warmup = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # divide lr at epoch 82 and 123 (391 iterations per epoch)
    scheduler_multistep = MultiStepLR(
        optimizer, milestones=[82 - warmup_epochs, 123 - warmup_epochs], gamma=0.1
    )

    train_results = []
    dist.barrier()

    for epoch in range(num_epochs):
        # set the epoch ID for the DistributedSampler
        train_loader.sampler.set_epoch(epoch)
        total_loss = torch.tensor(0.0, dtype=torch.float32, device=device)
        total_samples = torch.tensor(0, dtype=torch.int64, device=device)
        model.train()

        for images, labels in train_loader:
            inputs, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)

            with torch.no_grad():
                _, predicted = torch.max(outputs, 1)
                train_acc(predicted, labels)
                total_samples += labels.size(0)

            loss.backward()
            optimizer.step()

        if epoch < warmup_epochs:
            scheduler_warmup.step()
        else:
            scheduler_multistep.step()

        # sum values across all processes
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)

        loss = total_loss.item() / total_samples.item()
        acc = train_acc.compute().item()  # total acc over all batches

        total_quantization_error, numel = model.module.get_quantization_error()
        quant_error = total_quantization_error / numel
        if type(quant_error) == torch.Tensor:
            quant_error = quant_error.item()

        if local_rank == 0:
            print(
                f"epoch: {epoch}, train_loss: {loss:.4f}, train_acc: {acc:.4f}, quantization_error: {quant_error:.10f}"
            )
        train_results.append((epoch, loss, acc, quant_error))
        train_acc.reset()  # reset internal state

    if local_rank == 0:
        print(f"saving model at {model_path} 💾")
        torch.save(model.state_dict(), model_path)

    return train_results


def train_model(
    model_type: str,
    quantizer: Optional[Callable[..., None]],
    bits: int,
    num_epochs: int,
    lr: float,
    train_loader: DataLoader,
    image_size: Tuple[int],
    model_path: str,
    full_precision_model_path: str,
) -> List[Tuple[int, float, float, float]]:
    num_classes = len(train_loader.dataset.classes)

    model = get_model(
        model_type=model_type,
        num_classes=num_classes,
        quantize_fn=quantizer,
        bits=bits,
        image_size=image_size,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.manual_seed(local_rank)  # set different seed for each process
    device = torch.device("cuda", local_rank)
    print(f"{device = } 👨‍💻")

    train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    model = model.to(device)
    model = DistributedDataParallel(
        model, device_ids=[local_rank], output_device=local_rank
    )

    # if quantize aware training, copy full precision model weights
    if quantizer is not None:
        assert os.path.exists(
            full_precision_model_path
        ), "QAT requires full precision model"
        model.load_state_dict(torch.load(full_precision_model_path))

    # perform full precision training or QAT
    train_res = run_train_loop(
        model=model,
        device=device,
        train_loader=train_loader,
        model_path=model_path,
        num_epochs=num_epochs,
        lr=lr,
        train_acc=train_acc,
    )
    return train_res


def main(
    model_type: str,
    dataset: str,
    quantizer_type: str,
    bits: int,
    num_epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    train_dir: str = "./train",
    data_dir: str = "./data",
) -> None:
    """
    Training Configurations:
    - model_type: model to train (ie. resnet20, mobilenet)
    - dataset: dataset to train on (ie. cifar, imagenet)
    - quantizer_type: quantizer to use for QAT (ie. none, lin, lin+, po2, po2+)
    - bits: number of bits to quantize to (ie. 2, 3, 4)
    """

    assert (
        torch.cuda.is_available() and torch.cuda.device_count() >= 1
    ), "invalid hardware"
    assert model_type in [
        "resnet20",
        "resnet32",
        "resnet44",
        "resnet56",
        "mobilenet",
        "mobilevit",
    ], "invalid model type"
    assert dataset in ["cifar", "imagenet"], "invalid dataset"
    assert quantizer_type in [
        "none",
        "lin",
        "lin+",
        "po2",
        "po2+",
    ], "invalid quantizer type"
    assert bits in [2, 3, 4], "invalid number of bits"

    train_config = (
        "full_precision" if quantizer_type == "none" else f"{quantizer_type}_{bits}"
    )

    work_dir = f"{train_dir}/{dataset}/{model_type}/{seed}"
    Path(f"{work_dir}/model_state").mkdir(parents=True, exist_ok=True)

    quantizer = (
        quantizer_dict[quantizer_type] if quantizer_type in quantizer_dict else None
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    if local_rank == 0:
        print("initializing distributed environment 😇")
    init(seed=seed)

    train_loader, _, image_size = get_dataloaders(
        dataset=dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        num_workers=2,
        distributed=True,
    )

    dist.barrier()

    train_results = train_model(
        quantizer=quantizer,
        bits=bits,
        model_type=model_type,
        num_epochs=num_epochs,
        lr=lr,
        train_loader=train_loader,
        image_size=image_size,
        model_path=f"{work_dir}/model_state/{train_config}.pth",
        full_precision_model_path=f"{work_dir}/model_state/full_precision.pth",
    )

    if local_rank == 0:
        with open(f"{work_dir}/{train_config}.csv", mode="w") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "quantization_error"])
            writer.writerows(train_results)

    if local_rank == 0:
        print("destroying distributed environment 😈")

    dist.destroy_process_group()


if __name__ == "__main__":
    fire.Fire(main)
