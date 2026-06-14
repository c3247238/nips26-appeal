"""
Data loading utilities for CIFAR-10/100 with standard augmentation.
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T


# CIFAR normalization constants
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_transforms(dataset_name):
    """Get train/test transforms with standard CIFAR augmentation."""
    if dataset_name == "cifar10":
        mean, std = CIFAR10_MEAN, CIFAR10_STD
    elif dataset_name == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std),
    ])
    return train_transform, test_transform


def get_dataloaders(dataset_name, batch_size=128, data_dir="./data",
                    num_workers=4, pin_memory=True):
    """Create train and test dataloaders.

    Args:
        dataset_name: "cifar10" or "cifar100"
        batch_size: training batch size
        data_dir: directory for downloading/caching datasets
        num_workers: dataloader workers
        pin_memory: pin GPU memory

    Returns:
        train_loader, test_loader, num_classes
    """
    train_transform, test_transform = get_transforms(dataset_name)

    if dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=test_transform)
        num_classes = 10
    elif dataset_name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=test_transform)
        num_classes = 100
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader, num_classes


def get_large_batch_loader(dataset_name, batch_size=8192, data_dir="./data",
                           num_workers=4):
    """Get a large-batch train loader for alignment diagnostic."""
    train_transform, _ = get_transforms(dataset_name)

    if dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=train_transform)
    elif dataset_name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=train_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=False)
    return loader
