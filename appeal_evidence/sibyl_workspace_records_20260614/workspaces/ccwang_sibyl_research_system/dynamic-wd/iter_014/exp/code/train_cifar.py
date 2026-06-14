#!/usr/bin/env python3
"""CIFAR-10/100 training script with all WD methods and diagnostic logging.

Usage:
    python train_cifar.py --method UDWDC --dataset cifar10 --model resnet20 \
        --epochs 200 --seed 42 --save_dir exp/results/full/phase1_diagnostic

For pilot mode:
    python train_cifar.py --method UDWDC --dataset cifar10 --model resnet20 \
        --epochs 10 --seed 42 --max_samples 100 --save_dir exp/results/pilots
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
import numpy as np

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from data import get_cifar10_loaders, get_cifar100_loaders
from diagnostics.logger import DiagnosticLogger


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(model, loader, optimizer, criterion, device, epoch):
    """Train for one epoch. Returns (loss, accuracy)."""
    model.train()
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

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on test set. Returns (loss, accuracy)."""
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

    avg_loss = total_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def cosine_lr(optimizer_wrapper, epoch, total_epochs, base_lr):
    """Cosine annealing learning rate schedule."""
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    optimizer_wrapper.set_lr(lr)
    return lr


def main():
    parser = argparse.ArgumentParser(description='CIFAR Training with Dynamic WD')
    parser.add_argument('--method', type=str, default='FixedWD',
                        choices=['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective',
                                 'NoWD', 'UDWDC'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--model', type=str, default='resnet20',
                        choices=['resnet20', 'vgg16_bn'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Max training samples (for pilot)')
    parser.add_argument('--save_dir', type=str, default='exp/results/pilots')
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_id', type=str, default='',
                        help='Task ID for progress tracking')

    # UDWDC-specific args
    parser.add_argument('--K_p', type=float, default=0.5)
    parser.add_argument('--K_i', type=float, default=0.1)
    parser.add_argument('--K_d', type=float, default=0.3)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"[{args.method}] Device: {device}, Seed: {args.seed}")

    # PID file for process tracking
    if args.task_id:
        results_dir = Path(args.save_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        pid_file = results_dir / f"{args.task_id}.pid"
        pid_file.write_text(str(os.getpid()))

    # Data
    num_classes = 10 if args.dataset == 'cifar10' else 100
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(
            batch_size=args.batch_size, data_root=args.data_root,
            max_samples=args.max_samples, seed=args.seed
        )
    else:
        train_loader, test_loader = get_cifar100_loaders(
            batch_size=args.batch_size, data_root=args.data_root,
            max_samples=args.max_samples, seed=args.seed
        )

    # Model
    model = create_model(args.model, num_classes=num_classes).to(device)
    print(f"[{args.method}] Model: {args.model}, Params: {sum(p.numel() for p in model.parameters()):,}")

    # Optimizer
    opt_kwargs = {}
    if args.method == 'UDWDC':
        opt_kwargs = {'K_p': args.K_p, 'K_i': args.K_i, 'K_d': args.K_d}

    wd_optimizer = create_optimizer(
        args.method, model, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay, **opt_kwargs
    )

    criterion = nn.CrossEntropyLoss()

    # Diagnostic logger
    logger = DiagnosticLogger(
        save_dir=args.save_dir, method=args.method, seed=args.seed,
        model_name=args.model, dataset=args.dataset
    )

    # Training loop
    start_time = time.time()
    best_acc = 0.0

    for epoch in range(args.epochs):
        lr = cosine_lr(wd_optimizer, epoch, args.epochs, args.lr)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device, epoch
        )
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Collect diagnostics
        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(
            epoch, diagnostics, train_loss, test_loss, train_acc, test_acc, lr=lr
        )

        if test_acc > best_acc:
            best_acc = test_acc

        # Progress reporting
        if args.task_id:
            progress_file = Path(args.save_dir) / f"{args.task_id}_PROGRESS.json"
            progress_file.write_text(json.dumps({
                'task_id': args.task_id,
                'epoch': epoch + 1,
                'total_epochs': args.epochs,
                'loss': train_loss,
                'metric': {'train_acc': train_acc, 'test_acc': test_acc, 'best_acc': best_acc},
                'updated_at': datetime.now().isoformat(),
            }))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            elapsed = time.time() - start_time
            print(f"[{args.method}] Epoch {epoch+1}/{args.epochs} | "
                  f"LR: {lr:.6f} | Train: {train_acc:.2f}% | Test: {test_acc:.2f}% | "
                  f"Loss: {train_loss:.4f} | Best: {best_acc:.2f}% | "
                  f"Time: {elapsed:.0f}s")

    # Save diagnostics
    logger.save()
    elapsed = time.time() - start_time

    # Final summary
    summary = {
        'method': args.method,
        'model': args.model,
        'dataset': args.dataset,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'best_test_acc': best_acc,
        'final_test_acc': test_acc,
        'final_train_acc': train_acc,
        'generalization_gap': train_acc - test_acc,
        'total_wd_budget': logger.get_total_wd_budget(),
        'elapsed_sec': elapsed,
    }
    if args.method == 'UDWDC':
        summary.update({'K_p': args.K_p, 'K_i': args.K_i, 'K_d': args.K_d})

    summary_file = Path(args.save_dir) / f'{args.method}_seed{args.seed}_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n[{args.method}] DONE | Best: {best_acc:.2f}% | Final: {test_acc:.2f}% | "
          f"Gap: {train_acc - test_acc:.2f}% | Time: {elapsed:.1f}s")

    # DONE marker
    if args.task_id:
        done_file = Path(args.save_dir) / f"{args.task_id}_DONE"
        pid_file = Path(args.save_dir) / f"{args.task_id}.pid"
        if pid_file.exists():
            pid_file.unlink()
        done_file.write_text(json.dumps({
            'task_id': args.task_id,
            'status': 'success',
            'summary': f"Best acc: {best_acc:.2f}%, Final acc: {test_acc:.2f}%",
            'final_progress': summary,
            'timestamp': datetime.now().isoformat(),
        }))

    return summary


if __name__ == '__main__':
    main()
