#!/usr/bin/env python3
"""ImageNet training script with all WD methods, DDP support, and diagnostic logging.

Usage (single GPU):
    python train_imagenet.py --method UDWDC --model resnet50 --epochs 90 \
        --seed 42 --save_dir exp/results/full/phase4_imagenet_main

Usage (DDP, 2 GPUs):
    torchrun --nproc_per_node=2 train_imagenet.py --method UDWDC --model resnet50 \
        --epochs 90 --seed 42 --save_dir exp/results/full/phase4_imagenet_main

For pilot mode:
    python train_imagenet.py --method UDWDC --model resnet50 --epochs 2 \
        --seed 42 --max_samples 100 --save_dir exp/results/pilots
"""

import argparse
import json
import os
import sys
import time
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from data.imagenet import ImageNetParquetDataset, get_imagenet_loaders
from diagnostics.logger import DiagnosticLogger
from torchvision import transforms


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # Faster for fixed-size inputs


def is_ddp():
    return dist.is_initialized()


def get_rank():
    if is_ddp():
        return dist.get_rank()
    return 0


def get_world_size():
    if is_ddp():
        return dist.get_world_size()
    return 1


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, sampler=None):
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


def cosine_lr(optimizer_wrapper, epoch, total_epochs, base_lr, warmup_epochs=0):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    optimizer_wrapper.set_lr(lr)
    return lr


def main():
    parser = argparse.ArgumentParser(description='ImageNet Training with Dynamic WD')
    parser.add_argument('--method', type=str, default='FixedWD')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=['resnet50', 'resnet101', 'vit_s_16'])
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--save_dir', type=str, default='exp/results/pilots')
    parser.add_argument('--data_root', type=str, default='/home/ccwang/dataset/imagenet-1k')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--warmup_epochs', type=int, default=0)
    parser.add_argument('--task_id', type=str, default='')

    # UDWDC args
    parser.add_argument('--K_p', type=float, default=0.5)
    parser.add_argument('--K_i', type=float, default=0.1)
    parser.add_argument('--K_d', type=float, default=0.3)

    # DDP
    parser.add_argument('--local_rank', type=int, default=-1)

    args = parser.parse_args()

    # DDP setup
    ddp = 'RANK' in os.environ
    if ddp:
        dist.init_process_group('nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    rank = get_rank()
    set_seed(args.seed + rank)

    if rank == 0:
        print(f"[{args.method}] Device: {device}, World: {get_world_size()}, Seed: {args.seed}")

    # PID file
    if args.task_id and rank == 0:
        results_dir = Path(args.save_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        pid_file = results_dir / f"{args.task_id}.pid"
        pid_file.write_text(str(os.getpid()))

    # ViT uses AdamW
    use_adamw = args.model == 'vit_s_16'
    if use_adamw:
        args.lr = 1e-3
        args.weight_decay = 0.05
        args.warmup_epochs = 5

    # Model
    num_classes = 1000
    model = create_model(args.model, num_classes=num_classes).to(device)
    if rank == 0:
        print(f"[{args.method}] Model: {args.model}, Params: {sum(p.numel() for p in model.parameters()):,}")

    if ddp:
        model = DDP(model, device_ids=[device.index])

    # Optimizer (wraps the raw model, not DDP wrapper)
    raw_model = model.module if ddp else model
    opt_kwargs = {'use_adamw': use_adamw}
    if args.method == 'UDWDC':
        opt_kwargs.update({'K_p': args.K_p, 'K_i': args.K_i, 'K_d': args.K_d})

    wd_optimizer = create_optimizer(
        args.method, raw_model, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, **opt_kwargs
    )

    criterion = nn.CrossEntropyLoss()

    # Data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if use_adamw:
        # ViT augmentation
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            normalize,
            transforms.RandomErasing(p=0.25),
        ])
    else:
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
        args.data_root, split='train', transform=transform_train,
        max_samples=args.max_samples, seed=args.seed
    )
    val_dataset = ImageNetParquetDataset(
        args.data_root, split='validation', transform=transform_val,
        max_samples=min(args.max_samples, 1000) if args.max_samples else None,
        seed=args.seed
    )

    train_sampler = DistributedSampler(train_dataset) if ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if ddp else None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True
    )

    # Logger (rank 0 only)
    logger = None
    if rank == 0:
        logger = DiagnosticLogger(
            save_dir=args.save_dir, method=args.method, seed=args.seed,
            model_name=args.model, dataset='imagenet'
        )

    # Training loop
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(args.epochs):
        lr = cosine_lr(wd_optimizer, epoch, args.epochs, args.lr, args.warmup_epochs)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device, epoch, train_sampler
        )

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        if rank == 0:
            diagnostics = wd_optimizer.get_diagnostics()
            logger.log_epoch(epoch, diagnostics, train_loss, val_loss, train_acc, val_acc, lr=lr)

            if val_acc > best_acc:
                best_acc = val_acc

            # Progress reporting
            if args.task_id:
                progress_file = Path(args.save_dir) / f"{args.task_id}_PROGRESS.json"
                progress_file.write_text(json.dumps({
                    'task_id': args.task_id,
                    'epoch': epoch + 1,
                    'total_epochs': args.epochs,
                    'loss': train_loss,
                    'metric': {'train_acc': train_acc, 'val_acc': val_acc, 'best_acc': best_acc},
                    'updated_at': datetime.now().isoformat(),
                }))

            if (epoch + 1) % 5 == 0 or epoch == 0:
                elapsed = time.time() - start_time
                print(f"[{args.method}] Epoch {epoch+1}/{args.epochs} | "
                      f"LR: {lr:.6f} | Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                      f"Loss: {train_loss:.4f} | Best: {best_acc:.2f}% | "
                      f"Time: {elapsed:.0f}s")

    elapsed = time.time() - start_time

    if rank == 0:
        logger.save()

        summary = {
            'method': args.method,
            'model': args.model,
            'dataset': 'imagenet',
            'seed': args.seed,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'weight_decay': args.weight_decay,
            'best_val_acc': best_acc,
            'final_val_acc': val_acc,
            'final_train_acc': train_acc,
            'generalization_gap': train_acc - val_acc,
            'total_wd_budget': logger.get_total_wd_budget(),
            'elapsed_sec': elapsed,
            'world_size': get_world_size(),
        }

        summary_file = Path(args.save_dir) / f'{args.method}_seed{args.seed}_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[{args.method}] DONE | Best: {best_acc:.2f}% | Time: {elapsed:.1f}s")

        # DONE marker
        if args.task_id:
            done_file = Path(args.save_dir) / f"{args.task_id}_DONE"
            pid_file = Path(args.save_dir) / f"{args.task_id}.pid"
            if pid_file.exists():
                pid_file.unlink()
            done_file.write_text(json.dumps({
                'task_id': args.task_id,
                'status': 'success',
                'summary': f"Best val acc: {best_acc:.2f}%",
                'final_progress': summary,
                'timestamp': datetime.now().isoformat(),
            }))

    if ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
