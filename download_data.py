import fire
import torchvision
import torchvision.transforms as transforms
from datasets import load_dataset
from pathlib import Path


def main(dataset: str, data_dir: str = "./data"):

    Path(data_dir).mkdir(parents=True, exist_ok=True)

    if dataset == "cifar":
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
            root=data_dir, train=True, download=True, transform=train_transform
        )

        test_data = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform
        )

    # saved at ~/.cache/huggingface/datasets/imagenet-1k
    elif dataset == "imagenet": 
        imagenet_dataset = load_dataset("imagenet-1k")


if __name__ == "__main__":
    fire.Fire(main)