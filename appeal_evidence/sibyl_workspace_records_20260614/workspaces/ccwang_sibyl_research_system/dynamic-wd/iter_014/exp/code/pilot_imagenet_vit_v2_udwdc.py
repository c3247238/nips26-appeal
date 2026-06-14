#!/usr/bin/env python3
"""Pilot v2 supplement: Run UDWDC-v2 alone on ViT-S/16 ImageNet.

The main pilot timed out before UDWDC-v2 could complete 2 epochs.
This script runs UDWDC-v2 independently to verify it completes.
"""

import argparse
import json
import os
import sys
import time
import math
import gc
import signal
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

# Add code dir to path
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from models import create_model
from optimizers import create_optimizer
from diagnostics.logger import DiagnosticLogger

# Reuse data loader from pilot
import io
import glob
import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class ImageNetPilotDataset(Dataset):
    """Fast ImageNet dataset that only loads a few parquet shards."""

    def __init__(self, data_dir, split='train', transform=None,
                 max_shards=4, max_samples=None, seed=42):
        self.transform = transform
        pattern = os.path.join(data_dir, 'data', f'{split}-*.parquet')
        shard_files = sorted(glob.glob(pattern))
        if not shard_files:
            raise FileNotFoundError(f"No parquet files found matching {pattern}")
        if max_shards and max_shards < len(shard_files):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(shard_files), max_shards, replace=False)
            shard_files = [shard_files[i] for i in sorted(indices)]
        print(f"  Loading {len(shard_files)} shards for {split}...")
        tables = [pq.read_table(f) for f in shard_files]
        full_table = pa.concat_tables(tables)
        self.images = full_table.column('image')
        self.labels = full_table.column('label').to_pylist()
        if max_samples and max_samples < len(self.labels):
            rng = np.random.RandomState(seed)
            self._indices = rng.permutation(len(self.labels))[:max_samples].tolist()
        else:
            self._indices = list(range(len(self.labels)))
        print(f"  {split}: {len(self._indices)} samples from {len(shard_files)} shards")

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        real_idx = self._indices[idx]
        img_struct = self.images[real_idx].as_py()
        img_bytes = img_struct['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.labels[real_idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cosine_lr_with_warmup(optimizer_wrapper, epoch, total_epochs, base_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    optimizer_wrapper.set_lr(lr)
    return lr


def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


def apply_cutmix_mixup(inputs, targets, cutmix_alpha=1.0, mixup_alpha=0.8):
    batch_size = inputs.size(0)
    use_cutmix = np.random.rand() < 0.5
    if use_cutmix and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        rand_index = torch.randperm(batch_size, device=inputs.device)
        targets_a, targets_b = targets, targets[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
    elif mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        rand_index = torch.randperm(batch_size, device=inputs.device)
        targets_a, targets_b = targets, targets[rand_index]
        inputs = lam * inputs + (1 - lam) * inputs[rand_index]
    else:
        targets_a, targets_b, lam = targets, targets, 1.0
    return inputs, targets_a, targets_b, lam


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=96)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='/home/ccwang/dataset/imagenet-1k')
    parser.add_argument('--save_dir', type=str, default='exp/results/pilots/imagenet_vit')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_base = Path('exp/results')

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")

    set_seed(args.seed)

    # Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(), normalize, transforms.RandomErasing(p=0.25),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(), normalize,
    ])

    print("Loading ImageNet data...")
    train_dataset = ImageNetPilotDataset(
        args.data_root, 'train', transform_train, max_shards=8,
        max_samples=50000, seed=args.seed
    )
    val_dataset = ImageNetPilotDataset(
        args.data_root, 'validation', transform_val, max_shards=2,
        max_samples=5000, seed=args.seed
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Model + optimizer
    model = create_model('vit_s_16', num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"ViT-S/16: {n_params:,} params")

    wd_optimizer = create_optimizer(
        'UDWDC-v2', model, lr=args.lr, momentum=0.9, weight_decay=0.05,
        use_adamw=True, K_p=0.5, K_i=0.1, K_d=0.3
    )
    criterion = nn.CrossEntropyLoss()
    logger = DiagnosticLogger(save_dir=str(save_dir), method='UDWDC-v2', seed=args.seed,
                              model_name='vit_s_16', dataset='imagenet')

    print(f"\n{'='*60}")
    print(f"  Method: UDWDC-v2 | ViT-S/16 | AdamW | LR: {args.lr} | WD: 0.05")
    print(f"  CutMix(1.0) + Mixup(0.8) + RandAugment(2,9) + RandomErasing(0.25)")
    print(f"{'='*60}")

    start = time.time()
    epoch_results = []
    wd_budget_per_epoch = []

    for epoch in range(args.epochs):
        lr_now = cosine_lr_with_warmup(wd_optimizer, epoch, args.epochs, args.lr, 1)
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            inputs, ta, tb, lam = apply_cutmix_mixup(inputs, targets)
            wd_optimizer.zero_grad()
            outputs = model(inputs)
            loss = lam * criterion(outputs, ta) + (1 - lam) * criterion(outputs, tb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in wd_optimizer.param_groups for p in g['params']], max_norm=1.0
            )
            wd_optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += ta.size(0)
            correct += (lam * predicted.eq(ta).float() + (1 - lam) * predicted.eq(tb).float()).sum().item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx+1}/{len(train_loader)} | loss: {loss.item():.4f}", flush=True)

        train_loss = total_loss / max(total, 1)
        train_acc = 100.0 * correct / max(total, 1)

        # Eval
        model.eval()
        val_loss, val_correct, val_correct5, val_total = 0.0, 0, 0, 0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                _, pred5 = outputs.topk(5, 1, True, True)
                val_correct5 += pred5.eq(targets.view(-1, 1).expand_as(pred5)).sum().item()

        val_loss_avg = val_loss / max(val_total, 1)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        val_top5 = 100.0 * val_correct5 / max(val_total, 1)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, val_loss_avg, train_acc, val_acc, lr=lr_now)

        epoch_wd_budget = wd_optimizer.end_epoch_check()
        wd_budget_per_epoch.append(epoch_wd_budget)

        elapsed = time.time() - start
        print(f"  Epoch {epoch+1}/{args.epochs} | LR: {lr_now:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Top5: {val_top5:.2f}%) | "
              f"Loss: {train_loss:.4f} | WD_budget: {epoch_wd_budget:.2e} | "
              f"Time: {elapsed:.0f}s")

        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss_avg,
            'val_acc': val_acc,
            'val_top5': val_top5,
            'lr': lr_now,
            'epoch_wd_budget': epoch_wd_budget,
        })

    total_time = time.time() - start
    logger.save()

    # Coverage verification
    attn_layers = [k for k in diagnostics.keys() if 'attn' in k.lower() or 'qkv' in k.lower() or 'proj' in k.lower()]
    mlp_layers = [k for k in diagnostics.keys() if 'mlp' in k.lower() or 'fc1' in k.lower() or 'fc2' in k.lower()]

    result = {
        'method': 'UDWDC-v2',
        'model': 'vit_s_16',
        'dataset': 'imagenet',
        'optimizer': 'AdamW',
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': 0.05,
        'best_val_acc': max(r['val_acc'] for r in epoch_results),
        'best_val_top5': max(r['val_top5'] for r in epoch_results),
        'final_val_acc': epoch_results[-1]['val_acc'],
        'final_val_top5': epoch_results[-1]['val_top5'],
        'final_train_acc': epoch_results[-1]['train_acc'],
        'total_wd_budget': logger.get_total_wd_budget(),
        'wd_budget_per_epoch': wd_budget_per_epoch,
        'wd_budget_all_positive': all(b > 0 for b in wd_budget_per_epoch),
        'elapsed_sec': total_time,
        'epoch_results': epoch_results,
        'n_params': n_params,
        'diagnostics_coverage': {
            'total_tracked': len(diagnostics),
            'attention_layers': len(attn_layers),
            'mlp_layers': len(mlp_layers),
            'covers_attn': len(attn_layers) > 0,
            'covers_mlp': len(mlp_layers) > 0,
            'passed': len(attn_layers) > 0 and len(mlp_layers) > 0,
        },
        'cutmix_mixup_enabled': True,
        'status': 'success',
    }

    (save_dir / 'udwdc_v2_supplement.json').write_text(json.dumps(result, indent=2, default=str))

    print(f"\n{'='*60}")
    print(f"  UDWDC-v2 Supplement Complete")
    print(f"{'='*60}")
    print(f"  Final Val Acc: {result['final_val_acc']:.2f}% (Top5: {result['final_val_top5']:.2f}%)")
    print(f"  WD budget per epoch: {wd_budget_per_epoch}")
    print(f"  All WD budgets positive: {result['wd_budget_all_positive']}")
    print(f"  Total WD budget: {result['total_wd_budget']:.4f}")
    print(f"  Diagnostics: attn={len(attn_layers)}, mlp={len(mlp_layers)}")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Status: {'PASS' if result['wd_budget_all_positive'] else 'FAIL'}")

    return 0 if result['wd_budget_all_positive'] else 1


if __name__ == '__main__':
    sys.exit(main())
