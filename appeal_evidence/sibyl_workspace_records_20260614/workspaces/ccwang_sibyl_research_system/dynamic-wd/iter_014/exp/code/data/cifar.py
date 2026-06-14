"""CIFAR-10/100 data loading with standard augmentation."""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np


def get_cifar10_loaders(batch_size=128, num_workers=4, data_root='./data',
                        max_samples=None, seed=42):
    """Get CIFAR-10 train/test data loaders.

    Args:
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        data_root: Root directory for data storage.
        max_samples: If set, use only this many training samples (for pilots).
        seed: Random seed for subset selection.
    """
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test)

    if max_samples is not None and max_samples < len(trainset):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(trainset))[:max_samples]
        trainset = Subset(trainset, indices)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader


def get_cifar100_loaders(batch_size=128, num_workers=4, data_root='./data',
                         max_samples=None, seed=42):
    """Get CIFAR-100 train/test data loaders."""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root=data_root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=data_root, train=False, download=True, transform=transform_test)

    if max_samples is not None and max_samples < len(trainset):
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(trainset))[:max_samples]
        trainset = Subset(trainset, indices)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)

    return train_loader, test_loader
