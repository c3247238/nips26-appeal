"""
Data utilities: CIFAR-10/100 and ImageNet data loaders with standard augmentation.

Provides factory functions for creating train/test data loaders with proper
augmentation, DDP-compatible samplers, and mixed precision support.
"""

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
import torchvision.transforms as transforms


# ---------- CIFAR ----------

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def get_cifar_transforms(dataset: str = "cifar10", train: bool = True):
    """Get standard CIFAR augmentation transforms.

    Train: RandomCrop(32, padding=4) + RandomHorizontalFlip + Normalize
    Test: Normalize only
    """
    if dataset == "cifar100":
        mean, std = CIFAR100_MEAN, CIFAR100_STD
    else:
        mean, std = CIFAR_MEAN, CIFAR_STD

    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


def get_cifar_loaders(
    dataset: str = "cifar10",
    batch_size: int = 128,
    num_workers: int = 4,
    data_dir: str = "./data",
    distributed: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 or CIFAR-100 train and test data loaders.

    Args:
        dataset: "cifar10" or "cifar100"
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        data_dir: Directory to download/store data
        distributed: If True, use DistributedSampler for DDP
        seed: Random seed for reproducibility

    Returns:
        (train_loader, test_loader)
    """
    train_transform = get_cifar_transforms(dataset, train=True)
    test_transform = get_cifar_transforms(dataset, train=False)

    DatasetClass = torchvision.datasets.CIFAR100 if dataset == "cifar100" else torchvision.datasets.CIFAR10

    train_dataset = DatasetClass(
        root=data_dir, train=True, download=True, transform=train_transform,
    )
    test_dataset = DatasetClass(
        root=data_dir, train=False, download=True, transform=test_transform,
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,  # Larger batch for eval
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


# ---------- ImageNet ----------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_imagenet_transforms(train: bool = True):
    """Get standard ImageNet augmentation transforms.

    Train: RandomResizedCrop(224) + RandomHorizontalFlip + ColorJitter + Normalize
    Test: Resize(256) + CenterCrop(224) + Normalize
    """
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class ParquetImageDataset(torch.utils.data.Dataset):
    """ImageNet dataset loaded from HuggingFace parquet files.

    Each parquet file has columns: image (struct{bytes, path}), label (int64).
    Images are stored as raw bytes in the parquet files.
    """

    def __init__(self, parquet_files, transform=None):
        import pyarrow.parquet as pq
        self.transform = transform
        self.tables = []
        self.cumulative_lengths = []
        total = 0
        for f in sorted(parquet_files):
            t = pq.read_table(f, columns=["image", "label"])
            self.tables.append(t)
            total += len(t)
            self.cumulative_lengths.append(total)
        self._length = total

    def __len__(self):
        return self._length

    def _find_table_and_row(self, idx):
        import bisect
        table_idx = bisect.bisect_right(self.cumulative_lengths, idx)
        if table_idx > 0:
            idx -= self.cumulative_lengths[table_idx - 1]
        return table_idx, idx

    def __getitem__(self, idx):
        from io import BytesIO
        from PIL import Image

        table_idx, row_idx = self._find_table_and_row(idx)
        table = self.tables[table_idx]
        image_struct = table.column("image")[row_idx].as_py()
        label = table.column("label")[row_idx].as_py()

        img = Image.open(BytesIO(image_struct["bytes"])).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_imagenet_loaders(
    data_dir: str = "/data/imagenet",
    batch_size: int = 256,
    num_workers: int = 8,
    distributed: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Create ImageNet-1K train and val data loaders.

    Supports two data formats:
    1. Standard ImageFolder layout (train/, val/ subdirs with class folders)
    2. HuggingFace parquet format (data/ subdir with train-*.parquet, test-*.parquet)

    Args:
        data_dir: Path to ImageNet directory
        batch_size: Batch size per GPU
        num_workers: Number of data loading workers
        distributed: If True, use DistributedSampler for DDP
        seed: Random seed for reproducibility

    Returns:
        (train_loader, val_loader)
    """
    import glob

    # Detect data format
    parquet_data_dir = os.path.join(data_dir, "data")
    train_parquets = sorted(glob.glob(os.path.join(parquet_data_dir, "train-*.parquet")))
    use_parquet = len(train_parquets) > 0

    if use_parquet:
        # Prefer validation split (has real labels), fall back to test/val
        test_parquets = sorted(glob.glob(os.path.join(parquet_data_dir, "validation-*.parquet")))
        if not test_parquets:
            test_parquets = sorted(glob.glob(os.path.join(parquet_data_dir, "val-*.parquet")))
        if not test_parquets:
            test_parquets = sorted(glob.glob(os.path.join(parquet_data_dir, "test-*.parquet")))
        print(f"[ImageNet] Loading from parquet: {len(train_parquets)} train, {len(test_parquets)} test shards")
        train_dataset = ParquetImageDataset(train_parquets, transform=get_imagenet_transforms(train=True))
        val_dataset = ParquetImageDataset(test_parquets, transform=get_imagenet_transforms(train=False))
    else:
        train_dir = os.path.join(data_dir, "train")
        val_dir = os.path.join(data_dir, "val")
        train_dataset = torchvision.datasets.ImageFolder(
            train_dir, transform=get_imagenet_transforms(train=True),
        )
        val_dataset = torchvision.datasets.ImageFolder(
            val_dir, transform=get_imagenet_transforms(train=False),
        )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(train_dataset, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        prefetch_factor=2,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return train_loader, val_loader


# ---------- Model Factory ----------

def get_model(arch: str, num_classes: int = 10, use_bn: bool = True):
    """Create a model by architecture name.

    Supported architectures:
    - resnet20: Small ResNet for CIFAR (0.27M params)
    - vgg16bn: VGG-16 with batch normalization for CIFAR (15M params)
    - vgg16: VGG-16 WITHOUT batch normalization for CIFAR
    - resnet50: Standard ResNet-50 for ImageNet (25.6M params)
    - resnet101: Standard ResNet-101 for ImageNet (44.5M params)
    """
    arch = arch.lower()

    if arch == "resnet20":
        return _resnet20(num_classes)
    elif arch == "vgg16bn":
        return _vgg16_cifar(num_classes, use_bn=True)
    elif arch == "vgg16":
        return _vgg16_cifar(num_classes, use_bn=False)
    elif arch == "resnet50":
        import torchvision.models as models
        return models.resnet50(weights=None, num_classes=num_classes)
    elif arch == "resnet101":
        import torchvision.models as models
        return models.resnet101(weights=None, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown architecture: {arch}")


# ---------- ResNet-20 for CIFAR ----------

import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.fc = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        return self.fc(out)


def _resnet20(num_classes=10):
    return ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes)


# ---------- VGG-16 for CIFAR ----------

class VGG(nn.Module):
    """VGG-16 adapted for CIFAR (32x32 input)."""

    def __init__(self, num_classes=10, use_bn=True):
        super().__init__()
        self.features = self._make_layers(use_bn)
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def _make_layers(self, use_bn):
        cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M',
               512, 512, 512, 'M', 512, 512, 512, 'M']
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers.append(nn.MaxPool2d(2, 2))
            else:
                layers.append(nn.Conv2d(in_channels, x, 3, padding=1))
                if use_bn:
                    layers.append(nn.BatchNorm2d(x))
                layers.append(nn.ReLU(True))
                in_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        return self.classifier(out)


def _vgg16_cifar(num_classes=10, use_bn=True):
    return VGG(num_classes=num_classes, use_bn=use_bn)


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
