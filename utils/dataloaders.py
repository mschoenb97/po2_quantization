import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_dataloaders(
    dataset: str,
    data_dir: str,
    batch_size: int,
    num_workers: int,
    distributed: bool,
) -> tuple[DataLoader, DataLoader]:
    if dataset == "cifar":
        return get_cifar_dataloaders(data_dir, batch_size, num_workers, distributed)


def get_cifar_dataloaders(
    data_dir: str, batch_size: int, num_workers: int, distributed: bool
) -> tuple[DataLoader, DataLoader]:
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

    train_data = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=False, transform=train_transform
    )

    test_data = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=False, transform=test_transform
    )

    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    if distributed:
        # create a DistributedSampler to handle data parallelism
        train_sampler = DistributedSampler(train_loader.dataset, shuffle=True)
        test_sampler = DistributedSampler(test_loader.dataset, shuffle=False)

        train_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=False,
        )

        test_loader = DataLoader(
            train_loader.dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=False,
        )

    return train_loader, test_loader, (32, 32)
