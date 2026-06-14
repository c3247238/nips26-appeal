"""ImageNet-1K data loading from HuggingFace parquet format.

Data path: /home/ccwang/dataset/imagenet-1k
  - 294 train shards (train-NNNNN-of-00294.parquet)
  - 14 validation shards (validation-NNNNN-of-00014.parquet) with labels
  - 28 test shards (test-NNNNN-of-00028.parquet) without labels (not used)
"""

import os
import io
import glob
import torch
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from PIL import Image


class ImageNetParquetDataset(Dataset):
    """Load ImageNet from HuggingFace parquet shards."""

    def __init__(self, data_dir, split='train', transform=None, max_samples=None, seed=42):
        """
        Args:
            data_dir: Root directory containing parquet files.
            split: 'train' or 'test' (mapped to val).
            transform: Image transforms.
            max_samples: Limit number of samples for pilots.
            seed: Random seed for subset selection.
        """
        self.transform = transform

        # Find parquet files
        pattern = os.path.join(data_dir, 'data', f'{split}-*.parquet')
        shard_files = sorted(glob.glob(pattern))

        if not shard_files:
            raise FileNotFoundError(f"No parquet files found matching {pattern}")

        # Load all shards
        tables = []
        for f in shard_files:
            tables.append(pq.read_table(f))

        import pyarrow as pa
        full_table = pa.concat_tables(tables)

        self.images = full_table.column('image')
        self.labels = full_table.column('label').to_pylist()

        # Subsample if needed
        if max_samples is not None and max_samples < len(self.labels):
            rng = np.random.RandomState(seed)
            self._indices = rng.permutation(len(self.labels))[:max_samples].tolist()
        else:
            self._indices = list(range(len(self.labels)))

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]

        # The image column in HF parquet is a struct with 'bytes' and 'path'
        img_struct = self.images[real_idx].as_py()
        img_bytes = img_struct['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        label = self.labels[real_idx]

        if self.transform:
            img = self.transform(img)

        return img, label


def get_imagenet_loaders(batch_size=256, num_workers=8,
                         data_root='/home/ccwang/dataset/imagenet-1k',
                         max_samples=None, seed=42):
    """Get ImageNet train/val data loaders.

    Args:
        batch_size: Batch size per GPU.
        num_workers: Number of data loading workers.
        data_root: Root directory of ImageNet parquet data.
        max_samples: If set, use only this many training samples (for pilots).
        seed: Random seed.
    """
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = ImageNetParquetDataset(
        data_root, split='train', transform=transform_train,
        max_samples=max_samples, seed=seed
    )

    val_dataset = ImageNetParquetDataset(
        data_root, split='validation', transform=transform_val,
        max_samples=min(max_samples, 1000) if max_samples else None, seed=seed
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader
