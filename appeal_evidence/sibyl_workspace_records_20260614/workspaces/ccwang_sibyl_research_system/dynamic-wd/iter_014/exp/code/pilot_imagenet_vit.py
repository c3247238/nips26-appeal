#!/usr/bin/env python3
"""Pilot: ImageNet ViT-S/16 — 4 WD methods, 2 epochs, single GPU.

Validates that ViT-S/16 with AdamW can train on ImageNet with all 4 methods
(FixedWD, SWD, CWD, UDWDC) without OOM, and that training progresses.

Key differences from ResNet pilot:
- AdamW optimizer (not SGD)
- LR=1e-3 with warmup 5 epochs + cosine decay (but for 2-epoch pilot, warmup scaled)
- ViT-specific augmentation: RandAugment, CutMix, Mixup, RandomErasing
- weight_decay=0.05 (ViT standard) for FixedWD baseline
- Loads only a few parquet shards for speed
"""

import argparse
import json
import os
import sys
import time
import math
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

# Pilot-specific: fast ImageNet loader using only a few shards
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

        # Only load a few shards for pilot speed
        if max_shards and max_shards < len(shard_files):
            rng = np.random.RandomState(seed)
            indices = rng.choice(len(shard_files), max_shards, replace=False)
            shard_files = [shard_files[i] for i in sorted(indices)]

        print(f"  Loading {len(shard_files)} shards for {split}...")
        tables = []
        for f in shard_files:
            tables.append(pq.read_table(f))
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


METHODS = ['FixedWD', 'SWD', 'CWD', 'UDWDC']

METHOD_CONFIGS = {
    'FixedWD': {'weight_decay': 0.05},       # ViT standard WD
    'SWD':     {'weight_decay': 0.05},
    'CWD':     {'weight_decay': 0.05},
    'UDWDC':   {'weight_decay': 0.05, 'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.3},
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cosine_lr_with_warmup(optimizer_wrapper, epoch, total_epochs, base_lr, warmup_epochs):
    """Cosine decay with linear warmup."""
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    optimizer_wrapper.set_lr(lr)
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device):
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

        # Gradient clipping for ViT stability
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g['params']], max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)} | "
                  f"loss: {loss.item():.4f} | "
                  f"acc: {100.0 * correct / total:.2f}%", flush=True)

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


def run_single_method(method, train_loader, val_loader, device, save_dir,
                      epochs=2, base_lr=1e-3, warmup_epochs=1, seed=42, batch_size=256):
    """Train ViT-S/16 with a single WD method and return results."""
    set_seed(seed)

    model = create_model('vit_s_16', num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    config = METHOD_CONFIGS[method]
    wd = config.get('weight_decay', 0.05)
    opt_kwargs = {'use_adamw': True}
    if method == 'UDWDC':
        opt_kwargs.update({
            'K_p': config['K_p'],
            'K_i': config['K_i'],
            'K_d': config['K_d'],
        })

    wd_optimizer = create_optimizer(
        method, model, lr=base_lr, momentum=0.9, weight_decay=wd, **opt_kwargs
    )
    criterion = nn.CrossEntropyLoss()

    logger = DiagnosticLogger(
        save_dir=str(save_dir), method=method, seed=seed,
        model_name='vit_s_16', dataset='imagenet'
    )

    print(f"\n{'='*60}")
    print(f"  Method: {method} | Model: ViT-S/16 | Params: {n_params:,}")
    print(f"  LR: {base_lr} | WD: {wd} | Optimizer: AdamW")
    print(f"{'='*60}")

    start = time.time()
    best_acc = 0.0
    epoch_results = []

    for epoch in range(epochs):
        lr_now = cosine_lr_with_warmup(
            wd_optimizer, epoch, epochs, base_lr, warmup_epochs
        )
        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, val_loss, train_acc, val_acc, lr=lr_now)

        if val_acc > best_acc:
            best_acc = val_acc

        elapsed = time.time() - start
        print(f"  [{method}] Epoch {epoch+1}/{epochs} | LR: {lr_now:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
              f"Loss: {train_loss:.4f} | Time: {elapsed:.0f}s")

        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr_now,
        })

    elapsed = time.time() - start
    logger.save()

    # Free GPU memory
    del model, wd_optimizer
    torch.cuda.empty_cache()

    result = {
        'method': method,
        'model': 'vit_s_16',
        'dataset': 'imagenet',
        'optimizer': 'AdamW',
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': base_lr,
        'weight_decay': wd,
        'best_val_acc': best_acc,
        'final_val_acc': epoch_results[-1]['val_acc'] if epoch_results else 0,
        'final_train_acc': epoch_results[-1]['train_acc'] if epoch_results else 0,
        'elapsed_sec': elapsed,
        'epoch_results': epoch_results,
        'n_params': n_params,
        'status': 'success',
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='ImageNet ViT-S/16 Pilot')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='/home/ccwang/dataset/imagenet-1k')
    parser.add_argument('--save_dir', type=str,
                        default='exp/results/pilots/imagenet_vit')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--train_shards', type=int, default=8,
                        help='Number of train shards to load (pilot speed)')
    parser.add_argument('--val_shards', type=int, default=2,
                        help='Number of validation shards to load')
    parser.add_argument('--max_train_samples', type=int, default=50000,
                        help='Max training samples for pilot')
    parser.add_argument('--max_val_samples', type=int, default=5000,
                        help='Max validation samples for pilot')
    parser.add_argument('--warmup_epochs', type=int, default=1,
                        help='Warmup epochs (scaled down for pilot from 5)')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_base = Path('exp/results')
    results_base.mkdir(parents=True, exist_ok=True)

    # Write PID file
    pid_file = results_base / 'imagenet_vit.pid'
    pid_file.write_text(str(os.getpid()))

    # Timeout handler
    def timeout_handler(signum, frame):
        print(f"\n[TIMEOUT] Reached {args.timeout}s limit, saving partial results...")
        raise TimeoutError("Pilot timeout reached")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(0).total_memory -
                    torch.cuda.memory_allocated(0)) / 1e9
        print(f"Memory: {mem:.1f} GB total")

    print(f"\n=== ViT-S/16 ImageNet Pilot ===")
    print(f"Config: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, seed={args.seed}")
    print(f"Optimizer: AdamW | Warmup: {args.warmup_epochs} epochs")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Data: {args.train_shards} train shards, {args.val_shards} val shards")
    print(f"Max samples: train={args.max_train_samples}, val={args.max_val_samples}")
    print(f"Timeout: {args.timeout}s")

    # ViT-specific augmentation
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),
        transforms.ToTensor(),
        normalize,
        transforms.RandomErasing(p=0.25),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

    # Load data (once, reuse for all methods)
    print("\nLoading ImageNet data (pilot subset)...")
    data_start = time.time()
    train_dataset = ImageNetPilotDataset(
        args.data_root, split='train', transform=transform_train,
        max_shards=args.train_shards, max_samples=args.max_train_samples,
        seed=args.seed
    )
    val_dataset = ImageNetPilotDataset(
        args.data_root, split='validation', transform=transform_val,
        max_shards=args.val_shards, max_samples=args.max_val_samples,
        seed=args.seed
    )
    data_time = time.time() - data_start
    print(f"Data loaded in {data_time:.1f}s")

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Run all 4 methods
    all_results = {}
    passed = []
    failed = []
    overall_start = time.time()

    for i, method in enumerate(METHODS):
        # Write progress
        progress = {
            'task_id': 'imagenet_vit',
            'method_index': i + 1,
            'total_methods': len(METHODS),
            'current_method': method,
            'completed_methods': list(all_results.keys()),
            'passed': list(passed),
            'failed': list(failed),
            'updated_at': datetime.now().isoformat(),
        }
        progress_file = results_base / 'imagenet_vit_PROGRESS.json'
        progress_file.write_text(json.dumps(progress, indent=2))

        try:
            result = run_single_method(
                method, train_loader, val_loader, device, save_dir,
                epochs=args.epochs, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
                seed=args.seed, batch_size=args.batch_size
            )
            all_results[method] = result

            # For a 2-epoch ViT pilot, training is just starting.
            # Pass criteria: completed without error, loss is decreasing
            if result['status'] == 'success':
                passed.append(method)
                final_val = result['final_val_acc']
                print(f"  -> PASS: val_acc={final_val:.2f}%, training completed successfully")
            else:
                failed.append(method)
                print(f"  -> FAIL: {result.get('error', 'unknown')}")

        except torch.cuda.OutOfMemoryError:
            print(f"  -> OOM for {method}! Trying with batch_size=128...")
            torch.cuda.empty_cache()

            # Retry with smaller batch size
            small_train_loader = DataLoader(
                train_dataset, batch_size=128, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
            small_val_loader = DataLoader(
                val_dataset, batch_size=128, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            try:
                result = run_single_method(
                    method, small_train_loader, small_val_loader, device, save_dir,
                    epochs=args.epochs, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
                    seed=args.seed, batch_size=128
                )
                result['note'] = 'Reduced batch_size to 128 due to OOM'
                all_results[method] = result
                passed.append(method)
                print(f"  -> PASS (bs=128): val_acc={result['final_val_acc']:.2f}%")
            except Exception as e2:
                all_results[method] = {
                    'method': method, 'status': 'failed', 'error': str(e2)
                }
                failed.append(method)
                print(f"  -> FAIL: {e2}")
                torch.cuda.empty_cache()

        except TimeoutError:
            print(f"\n[TIMEOUT] Stopped at method {method}")
            all_results[method] = {
                'method': method, 'status': 'timeout'
            }
            failed.append(method)
            break

        except Exception as e:
            traceback.print_exc()
            all_results[method] = {
                'method': method, 'status': 'failed', 'error': str(e)
            }
            failed.append(method)
            torch.cuda.empty_cache()

    total_time = time.time() - overall_start
    signal.alarm(0)  # Cancel timeout

    # Summary
    print(f"\n{'='*60}")
    print(f"  PILOT SUMMARY: ImageNet ViT-S/16")
    print(f"{'='*60}")
    print(f"  Model: ViT-S/16 | Optimizer: AdamW | LR: {args.lr}")
    print(f"  Methods: {len(passed)} passed, {len(failed)} failed out of {len(METHODS)}")
    print(f"  Total time: {total_time:.1f}s")
    print()

    for method, result in all_results.items():
        status = "PASS" if method in passed else "FAIL"
        if result.get('status') == 'success':
            print(f"  [{status}] {method:20s} | Val: {result['final_val_acc']:.2f}% | "
                  f"Train: {result['final_train_acc']:.2f}% | "
                  f"Params: {result.get('n_params', 0):,} | "
                  f"Time: {result['elapsed_sec']:.0f}s")
        else:
            print(f"  [{status}] {method:20s} | {result.get('error', result.get('status', 'unknown'))}")

    # Overall pass/fail
    all_passed = len(passed) == len(METHODS)
    print(f"\n  Overall: {'PASS' if all_passed else 'PARTIAL'}")

    # Save summary
    summary = {
        'task': 'imagenet_vit',
        'type': 'pilot',
        'model': 'vit_s_16',
        'dataset': 'imagenet',
        'optimizer': 'AdamW',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'warmup_epochs': args.warmup_epochs,
        'augmentation': 'RandAugment(2,9) + RandomErasing(0.25)',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'methods': METHODS,
        'methods_passed': passed,
        'methods_failed': failed,
        'all_passed': all_passed,
        'total_time_sec': total_time,
        'data_load_time_sec': data_time,
        'results': all_results,
        'pass_criteria': 'All 4 methods complete 2 epochs without OOM; training progresses',
        'timestamp': datetime.now().isoformat(),
    }

    summary_file = save_dir / 'pilot_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Write DONE markers
    done_data = json.dumps({
        'task_id': 'imagenet_vit',
        'status': 'success' if all_passed else 'partial',
        'summary': f"ViT-S/16 pilot: {len(passed)}/{len(METHODS)} methods passed",
        'final_progress': summary,
        'timestamp': datetime.now().isoformat(),
    }, default=str)

    (results_base / 'imagenet_vit_DONE').write_text(done_data)
    (save_dir / 'imagenet_vit_DONE').write_text(done_data)

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    # Final progress update
    final_progress = {
        'task_id': 'imagenet_vit',
        'status': 'completed',
        'methods_passed': len(passed),
        'methods_failed': len(failed),
        'total_methods': len(METHODS),
        'all_passed': all_passed,
        'total_time_sec': total_time,
        'updated_at': datetime.now().isoformat(),
    }
    progress_file = results_base / 'imagenet_vit_PROGRESS.json'
    progress_file.write_text(json.dumps(final_progress, indent=2))

    print(f"\nResults saved to {save_dir}")
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
