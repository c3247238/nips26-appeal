#!/usr/bin/env python3
"""Pilot: ImageNet Budget-Matched FixedWD controls.

Train ResNet-50 on ImageNet-1K with FixedWD at 5 lambda values
{5e-4, 6e-4, 7e-4, 8e-4, 1e-3} to sweep through WD budgets.

Key metric: total WD budget = sum_{t} lambda_t * ||w_t||^2 across all steps
and all weight-decayed parameters. This measures the cumulative regularisation
pressure applied during training, and should differ across lambda values.

Pass criteria: All 5 configs complete 2 epochs; WD budgets differ across lambda values.
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


# 5 FixedWD lambda values to sweep
LAMBDA_VALUES = [5e-4, 6e-4, 7e-4, 8e-4, 1e-3]


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def cosine_lr(optimizer_wrapper, epoch, total_epochs, base_lr):
    progress = epoch / max(total_epochs, 1)
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * progress))
    optimizer_wrapper.set_lr(lr)
    return lr


def compute_wd_budget_step(model, wd_optimizer):
    """Compute the WD budget contribution for this step.

    WD budget per step = sum over all weight-decayed params of:
        lambda_effective * ||w||^2

    This is the amount of "regularisation pressure" applied in a single step.
    """
    budget = 0.0
    for group in wd_optimizer.param_groups:
        if not group['apply_wd']:
            continue
        wd = group.get('effective_wd', wd_optimizer.base_wd)
        for p in group['params']:
            w_norm_sq = torch.sum(p.data ** 2).item()
            budget += wd * w_norm_sq
    return budget


def train_one_epoch(model, loader, optimizer, criterion, device):
    """Train one epoch, tracking WD budget per step."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    epoch_wd_budget = 0.0
    step_wd_budgets = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()

        # Compute WD budget BEFORE the step (uses current weights and effective_wd)
        step_budget = compute_wd_budget_step(model, optimizer)
        epoch_wd_budget += step_budget
        step_wd_budgets.append(step_budget)

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy, epoch_wd_budget, step_wd_budgets


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


def run_single_lambda(lam, train_loader, val_loader, device, save_dir,
                      epochs=2, lr=0.1, momentum=0.9, seed=42, batch_size=256):
    """Train FixedWD with a specific lambda value and track WD budget."""
    set_seed(seed)

    model = create_model('resnet50', num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    wd_optimizer = create_optimizer(
        'FixedWD', model, lr=lr, momentum=momentum, weight_decay=lam
    )
    criterion = nn.CrossEntropyLoss()

    lam_str = f"{lam:.0e}".replace('+', '')
    logger = DiagnosticLogger(
        save_dir=str(save_dir), method=f'FixedWD_lam{lam_str}', seed=seed,
        model_name='resnet50', dataset='imagenet'
    )

    print(f"\n{'='*60}")
    print(f"  FixedWD lambda={lam} | Params: {n_params:,}")
    print(f"{'='*60}")

    start = time.time()
    best_acc = 0.0
    epoch_results = []
    total_wd_budget = 0.0
    all_step_wd_budgets = []

    for epoch in range(epochs):
        lr_now = cosine_lr(wd_optimizer, epoch, epochs, lr)
        train_loss, train_acc, epoch_wd_budget, step_budgets = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        total_wd_budget += epoch_wd_budget
        all_step_wd_budgets.extend(step_budgets)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, val_loss, train_acc, val_acc, lr=lr_now)

        if val_acc > best_acc:
            best_acc = val_acc

        elapsed = time.time() - start
        print(f"  [lam={lam}] Epoch {epoch+1}/{epochs} | LR: {lr_now:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
              f"Loss: {train_loss:.4f} | WD_budget: {epoch_wd_budget:.4f} | "
              f"Time: {elapsed:.0f}s")

        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': lr_now,
            'epoch_wd_budget': epoch_wd_budget,
            'cumulative_wd_budget': total_wd_budget,
        })

    elapsed = time.time() - start
    logger.save()

    # Compute WD budget statistics
    avg_step_budget = np.mean(all_step_wd_budgets) if all_step_wd_budgets else 0.0
    std_step_budget = np.std(all_step_wd_budgets) if all_step_wd_budgets else 0.0

    # Free GPU memory
    del model, wd_optimizer
    torch.cuda.empty_cache()

    result = {
        'lambda': lam,
        'method': 'FixedWD',
        'model': 'resnet50',
        'dataset': 'imagenet',
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': lam,
        'best_val_acc': best_acc,
        'final_val_acc': epoch_results[-1]['val_acc'] if epoch_results else 0,
        'final_train_acc': epoch_results[-1]['train_acc'] if epoch_results else 0,
        'total_wd_budget': total_wd_budget,
        'avg_step_wd_budget': avg_step_budget,
        'std_step_wd_budget': std_step_budget,
        'n_steps': len(all_step_wd_budgets),
        'elapsed_sec': elapsed,
        'epoch_results': epoch_results,
        'status': 'success',
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='ImageNet Budget-Matched FixedWD Pilot')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='/home/ccwang/dataset/imagenet-1k')
    parser.add_argument('--save_dir', type=str,
                        default='exp/results/pilots/imagenet_budget_matched')
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
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_base = Path('exp/results')

    # Write PID file
    pid_file = results_base / 'imagenet_budget_matched.pid'
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
        print(f"Memory: {mem:.1f} GB")

    print(f"\nPilot config: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, momentum={args.momentum}, seed={args.seed}")
    print(f"Lambda values: {LAMBDA_VALUES}")
    print(f"Data: {args.train_shards} train shards, {args.val_shards} val shards")
    print(f"Max samples: train={args.max_train_samples}, val={args.max_val_samples}")
    print(f"Timeout: {args.timeout}s")

    # Data transforms
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
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

    # Load data once, reuse for all lambda configs
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

    # Run all 5 lambda values
    all_results = {}
    passed = []
    failed = []
    overall_start = time.time()

    for i, lam in enumerate(LAMBDA_VALUES):
        lam_key = f"lam_{lam}"

        # Write progress
        progress = {
            'task_id': 'imagenet_budget_matched',
            'lambda_index': i + 1,
            'total_lambdas': len(LAMBDA_VALUES),
            'current_lambda': lam,
            'completed_lambdas': list(all_results.keys()),
            'passed': passed[:],
            'failed': failed[:],
            'updated_at': datetime.now().isoformat(),
        }
        progress_file = results_base / 'imagenet_budget_matched_PROGRESS.json'
        progress_file.write_text(json.dumps(progress, indent=2))

        try:
            result = run_single_lambda(
                lam, train_loader, val_loader, device, save_dir,
                epochs=args.epochs, lr=args.lr, momentum=args.momentum,
                seed=args.seed, batch_size=args.batch_size
            )
            all_results[lam_key] = result

            if result['status'] == 'success':
                passed.append(lam_key)
                print(f"  -> PASS: lambda={lam}, WD_budget={result['total_wd_budget']:.4f}, "
                      f"val_acc={result['final_val_acc']:.2f}%")
            else:
                failed.append(lam_key)

        except torch.cuda.OutOfMemoryError:
            print(f"  -> OOM for lambda={lam}! Trying with batch_size=128...")
            torch.cuda.empty_cache()

            small_train_loader = DataLoader(
                train_dataset, batch_size=128, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
            small_val_loader = DataLoader(
                val_dataset, batch_size=128, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            try:
                result = run_single_lambda(
                    lam, small_train_loader, small_val_loader, device, save_dir,
                    epochs=args.epochs, lr=args.lr, momentum=args.momentum,
                    seed=args.seed, batch_size=128
                )
                result['note'] = 'Reduced batch_size to 128 due to OOM'
                all_results[lam_key] = result
                passed.append(lam_key)
                print(f"  -> PASS (bs=128): lambda={lam}, WD_budget={result['total_wd_budget']:.4f}")
            except Exception as e2:
                all_results[lam_key] = {
                    'lambda': lam, 'status': 'failed', 'error': str(e2)
                }
                failed.append(lam_key)
                print(f"  -> FAIL: {e2}")

        except TimeoutError:
            print(f"\n[TIMEOUT] Stopped at lambda={lam}")
            all_results[lam_key] = {
                'lambda': lam, 'status': 'timeout'
            }
            failed.append(lam_key)
            break

        except Exception as e:
            traceback.print_exc()
            all_results[lam_key] = {
                'lambda': lam, 'status': 'failed', 'error': str(e)
            }
            failed.append(lam_key)
            torch.cuda.empty_cache()

    total_time = time.time() - overall_start
    signal.alarm(0)  # Cancel timeout

    # Verify WD budgets differ across lambda values
    wd_budgets = {}
    for lam_key, result in all_results.items():
        if result.get('status') == 'success':
            wd_budgets[lam_key] = result['total_wd_budget']

    budgets_differ = False
    if len(wd_budgets) >= 2:
        budget_values = list(wd_budgets.values())
        budget_range = max(budget_values) - min(budget_values)
        budgets_differ = budget_range > 0.001  # non-trivial difference

    # Summary
    print(f"\n{'='*60}")
    print(f"  PILOT SUMMARY: ImageNet Budget-Matched FixedWD")
    print(f"{'='*60}")
    print(f"  Configs: {len(passed)} passed, {len(failed)} failed out of {len(LAMBDA_VALUES)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  WD budgets differ: {budgets_differ}")
    print()

    print(f"  {'Lambda':>10s} | {'WD Budget':>12s} | {'Val Acc':>10s} | {'Train Acc':>10s} | {'Time':>8s}")
    print(f"  {'-'*10} | {'-'*12} | {'-'*10} | {'-'*10} | {'-'*8}")
    for lam_key, result in all_results.items():
        status = "PASS" if lam_key in passed else "FAIL"
        if result.get('status') == 'success':
            print(f"  [{status}] {result['lambda']:>8.0e} | {result['total_wd_budget']:>12.4f} | "
                  f"{result['final_val_acc']:>9.2f}% | {result['final_train_acc']:>9.2f}% | "
                  f"{result['elapsed_sec']:>7.0f}s")
        else:
            print(f"  [{status}] {result.get('lambda', '?'):>8} | "
                  f"{result.get('error', result.get('status', 'unknown'))}")

    all_passed = (len(passed) == len(LAMBDA_VALUES)) and budgets_differ
    print(f"\n  Overall: {'PASS' if all_passed else 'PARTIAL'}")
    if not budgets_differ and len(wd_budgets) >= 2:
        print(f"  WARNING: WD budgets do not differ significantly!")
        print(f"  Budget values: {wd_budgets}")

    # Save summary
    summary = {
        'task': 'imagenet_budget_matched',
        'type': 'pilot',
        'model': 'resnet50',
        'dataset': 'imagenet',
        'method': 'FixedWD',
        'lambda_values': LAMBDA_VALUES,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'momentum': args.momentum,
        'seed': args.seed,
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'configs_passed': passed,
        'configs_failed': failed,
        'all_passed': all_passed,
        'budgets_differ': budgets_differ,
        'wd_budgets': {k: float(v) for k, v in wd_budgets.items()},
        'total_time_sec': total_time,
        'data_load_time_sec': data_time,
        'results': all_results,
        'pass_criteria': 'All 5 configs complete 2 epochs; WD budgets differ across lambda values',
        'timestamp': datetime.now().isoformat(),
    }

    summary_file = save_dir / 'pilot_summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Write DONE markers
    done_data = json.dumps({
        'task_id': 'imagenet_budget_matched',
        'status': 'success' if all_passed else 'partial',
        'summary': f"{len(passed)}/{len(LAMBDA_VALUES)} configs passed, budgets_differ={budgets_differ}",
        'final_progress': summary,
        'timestamp': datetime.now().isoformat(),
    }, default=str)

    (results_base / 'imagenet_budget_matched_DONE').write_text(done_data)
    (save_dir / 'imagenet_budget_matched_DONE').write_text(done_data)

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    # Final progress update
    final_progress = {
        'task_id': 'imagenet_budget_matched',
        'status': 'completed',
        'configs_passed': len(passed),
        'configs_failed': len(failed),
        'total_configs': len(LAMBDA_VALUES),
        'all_passed': all_passed,
        'budgets_differ': budgets_differ,
        'wd_budgets': {k: float(v) for k, v in wd_budgets.items()},
        'total_time_sec': total_time,
        'updated_at': datetime.now().isoformat(),
    }
    progress_file.write_text(json.dumps(final_progress, indent=2))

    print(f"\nResults saved to {save_dir}")
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
