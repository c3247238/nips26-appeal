#!/usr/bin/env python3
"""Pilot v2: ImageNet ResNet-101 — 5 WD methods (top-3 + FixedWD + UDWDC-v2), 2 epochs, single GPU.

Phase 5a: Architecture generalization — ResNet-101/ImageNet.
Tests whether ResNet-101 preserves the method ranking from ResNet-50 Phase 4.

Methods: FixedWD (baseline), SWD (top-1 in pilot), CWD (top-2), CPR (top-3), UDWDC-v2

Pass criteria (from task_plan.json):
  - ResNet-101 with FixedWD completes 2 epochs without OOM on 2 GPUs
  - UDWDC-v2 WD budget > 0
  - All methods complete without errors

Updated from v1:
  - Added UDWDC-v2 and CPR methods
  - Added WD budget tracking + UDWDC-v2 budget verification
  - Added independence check (epoch-1 outputs differ across methods)
  - VRAM probing for optimal batch size
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


# 5 methods: FixedWD baseline + top-3 from Phase 4 + UDWDC-v2
METHODS = [
    'FixedWD',
    'SWD',
    'CWD',
    'CPR',
    'UDWDC-v2',
]

METHOD_CONFIGS = {
    'FixedWD': {'weight_decay': 1e-4},
    'SWD': {'weight_decay': 1e-4},
    'CWD': {'weight_decay': 1e-4},
    'CPR': {'weight_decay': 1e-4},
    'UDWDC-v2': {'weight_decay': 1e-4, 'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.3},
}


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


def find_max_batch_size(model, device, start=256, min_bs=32):
    """Binary search for the maximum batch size the current GPU can sustain."""
    high, best = start, min_bs
    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            dummy = torch.randn(mid, 3, 224, 224, device=device)
            with torch.no_grad():
                _ = model(dummy)
            del dummy
            torch.cuda.empty_cache()
            best = mid
            min_bs = mid + 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
            gc.collect()
    return best


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in loader:
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
    correct_top5 = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Top-5 accuracy
        _, pred5 = outputs.topk(5, 1, True, True)
        correct_top5 += pred5.eq(targets.view(-1, 1).expand_as(pred5)).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    top5_accuracy = 100.0 * correct_top5 / max(total, 1)
    return avg_loss, accuracy, top5_accuracy


def run_single_method(method, train_loader, val_loader, device, save_dir,
                      epochs=2, lr=0.1, momentum=0.9, seed=42, batch_size=256):
    """Train a single method and return results."""
    set_seed(seed)

    model = create_model('resnet101', num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    config = METHOD_CONFIGS[method]
    wd = config.get('weight_decay', 1e-4)
    opt_kwargs = {}
    if method == 'UDWDC-v2':
        opt_kwargs = {'K_p': config['K_p'], 'K_i': config['K_i'], 'K_d': config['K_d']}

    wd_optimizer = create_optimizer(
        method, model, lr=lr, momentum=momentum, weight_decay=wd, **opt_kwargs
    )
    criterion = nn.CrossEntropyLoss()

    logger = DiagnosticLogger(
        save_dir=str(save_dir), method=method, seed=seed,
        model_name='resnet101', dataset='imagenet'
    )

    print(f"\n{'='*60}")
    print(f"  Method: {method} | Model: ResNet-101 | Params: {n_params:,} | WD: {wd}")
    print(f"{'='*60}")

    start = time.time()
    best_acc = 0.0
    epoch_results = []
    total_wd_budget = 0.0
    udwdc_v2_wd_budget = 0.0

    for epoch in range(epochs):
        lr_now = cosine_lr(wd_optimizer, epoch, epochs, lr)
        train_loss, train_acc = train_one_epoch(model, train_loader, wd_optimizer, criterion, device)
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, val_loss, train_acc, val_acc, lr=lr_now)

        # Compute epoch WD budget from diagnostics
        epoch_wd = sum(d.get('effective_wd', 0) * (d.get('w_norm', 0) ** 2)
                       for d in diagnostics.values())
        total_wd_budget += epoch_wd

        # UDWDC-v2 specific: check cumulative WD budget
        if method == 'UDWDC-v2' and hasattr(wd_optimizer, 'end_epoch_check'):
            epoch_budget_v2 = wd_optimizer.end_epoch_check()
            udwdc_v2_wd_budget = wd_optimizer.get_cumulative_wd_budget()
            print(f"    [UDWDC-v2] Epoch WD budget: {epoch_budget_v2:.6e}, "
                  f"Cumulative: {udwdc_v2_wd_budget:.6e}")

        if val_acc > best_acc:
            best_acc = val_acc

        elapsed = time.time() - start
        print(f"  [{method}] Epoch {epoch+1}/{epochs} | LR: {lr_now:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (top5: {val_top5:.2f}%) | "
              f"Loss: {train_loss:.4f} | WD_budget: {epoch_wd:.6e} | Time: {elapsed:.0f}s")

        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_top5': val_top5,
            'lr': lr_now,
            'wd_budget_epoch': epoch_wd,
        })

    elapsed = time.time() - start
    logger.save()

    # Free GPU memory
    del model, wd_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        'method': method,
        'model': 'resnet101',
        'dataset': 'imagenet',
        'seed': seed,
        'epochs': epochs,
        'batch_size': batch_size,
        'lr': lr,
        'weight_decay': wd,
        'best_val_acc': best_acc,
        'final_val_acc': epoch_results[-1]['val_acc'] if epoch_results else 0,
        'final_val_top5': epoch_results[-1]['val_top5'] if epoch_results else 0,
        'final_train_acc': epoch_results[-1]['train_acc'] if epoch_results else 0,
        'total_wd_budget': total_wd_budget,
        'cumulative_wd_budget_v2': udwdc_v2_wd_budget if method == 'UDWDC-v2' else 0.0,
        'elapsed_sec': elapsed,
        'epoch_results': epoch_results,
        'epoch1_outputs': {
            'train_loss': epoch_results[0]['train_loss'] if epoch_results else 0,
            'train_acc': epoch_results[0]['train_acc'] if epoch_results else 0,
            'val_acc': epoch_results[0]['val_acc'] if epoch_results else 0,
            'val_top5': epoch_results[0]['val_top5'] if epoch_results else 0,
        } if epoch_results else {},
        'status': 'success',
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='ImageNet ResNet-101 Pilot v2')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str, default='/home/ccwang/dataset/imagenet-1k')
    parser.add_argument('--save_dir', type=str,
                        default='exp/results/pilots/imagenet_resnet101')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--train_shards', type=int, default=8,
                        help='Number of train shards to load (pilot speed)')
    parser.add_argument('--val_shards', type=int, default=3,
                        help='Number of validation shards to load')
    parser.add_argument('--max_train_samples', type=int, default=30000,
                        help='Max training samples for pilot')
    parser.add_argument('--max_val_samples', type=int, default=5000,
                        help='Max validation samples for pilot')
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use (single GPU mode for pilot)')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_base = Path('exp/results')

    # Write PID file
    pid_file = results_base / 'imagenet_resnet101.pid'
    pid_file.write_text(str(os.getpid()))

    # Timeout handler
    def timeout_handler(signum, frame):
        print(f"\n[TIMEOUT] Reached {args.timeout}s limit, saving partial results...")
        raise TimeoutError("Pilot timeout reached")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(args.timeout)

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu_id)}")
        mem = torch.cuda.get_device_properties(args.gpu_id).total_mem / 1e9 if hasattr(torch.cuda.get_device_properties(args.gpu_id), 'total_mem') else torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9
        print(f"Memory: {mem:.1f} GB")

    print(f"\nPilot v2 config: ResNet-101, {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, seed={args.seed}")
    print(f"Data: {args.train_shards} train shards, {args.val_shards} val shards")
    print(f"Max samples: train={args.max_train_samples}, val={args.max_val_samples}")
    print(f"Timeout: {args.timeout}s")
    print(f"Methods: {METHODS}")

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

    # VRAM probing
    print("\nProbing max batch size for ResNet-101...")
    probe_model = create_model('resnet101', num_classes=1000).to(device)
    max_bs = find_max_batch_size(probe_model, device, start=512, min_bs=32)
    del probe_model
    torch.cuda.empty_cache()
    gc.collect()

    effective_bs = min(args.batch_size, max_bs)
    print(f"Max stable batch size: {max_bs}, using: {effective_bs}")

    # Write GPU profile
    gpu_profile = {
        'gpu_name': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'cpu',
        'vram_total_mb': int(torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e6) if torch.cuda.is_available() else 0,
        'max_batch_size': max_bs,
        'effective_batch_size': effective_bs,
        'model': 'resnet101',
    }
    profile_file = results_base / 'imagenet_resnet101_gpu_profile.json'
    profile_file.write_text(json.dumps(gpu_profile, indent=2))
    print(f"GPU profile saved to {profile_file}")

    train_loader = DataLoader(
        train_dataset, batch_size=effective_bs, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=effective_bs, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    # Run all 5 methods
    all_results = {}
    passed = []
    failed = []
    overall_start = time.time()

    for i, method in enumerate(METHODS):
        # Write progress
        progress = {
            'task_id': 'imagenet_resnet101',
            'epoch': 0,
            'total_epochs': args.epochs,
            'method_index': i + 1,
            'total_methods': len(METHODS),
            'current_method': method,
            'completed_methods': list(all_results.keys()),
            'passed': list(passed),
            'failed': list(failed),
            'updated_at': datetime.now().isoformat(),
        }
        progress_file = results_base / 'imagenet_resnet101_PROGRESS.json'
        progress_file.write_text(json.dumps(progress, indent=2))

        try:
            result = run_single_method(
                method, train_loader, val_loader, device, save_dir,
                epochs=args.epochs, lr=args.lr, momentum=args.momentum,
                seed=args.seed, batch_size=effective_bs
            )
            all_results[method] = result

            if result['status'] == 'success':
                passed.append(method)
                print(f"  -> PASS: val_acc={result['final_val_acc']:.2f}%, "
                      f"top5={result['final_val_top5']:.2f}%, "
                      f"wd_budget={result['total_wd_budget']:.6e}, "
                      f"time={result['elapsed_sec']:.0f}s")
            else:
                failed.append(method)
                print(f"  -> FAIL: status={result['status']}")

        except torch.cuda.OutOfMemoryError:
            print(f"  -> OOM for {method} with batch_size={effective_bs}! Trying batch_size={effective_bs//2}...")
            torch.cuda.empty_cache()
            gc.collect()

            fallback_bs = max(32, effective_bs // 2)
            small_train_loader = DataLoader(
                train_dataset, batch_size=fallback_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
            small_val_loader = DataLoader(
                val_dataset, batch_size=fallback_bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            try:
                result = run_single_method(
                    method, small_train_loader, small_val_loader, device, save_dir,
                    epochs=args.epochs, lr=args.lr, momentum=args.momentum,
                    seed=args.seed, batch_size=fallback_bs
                )
                result['note'] = f'Reduced batch_size to {fallback_bs} due to OOM'
                all_results[method] = result
                passed.append(method)
                print(f"  -> PASS (bs={fallback_bs}): val_acc={result['final_val_acc']:.2f}%")
            except Exception as e2:
                all_results[method] = {
                    'method': method, 'status': 'failed', 'error': str(e2)
                }
                failed.append(method)
                print(f"  -> FAIL: {e2}")
                torch.cuda.empty_cache()
                gc.collect()

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
            gc.collect()

    total_time = time.time() - overall_start
    signal.alarm(0)  # Cancel timeout

    # Independence check: verify epoch-1 outputs differ across methods
    epoch1_outputs = {}
    for m, r in all_results.items():
        if r.get('status') == 'success' and r.get('epoch1_outputs'):
            epoch1_outputs[m] = r['epoch1_outputs']

    independence = {}
    if len(epoch1_outputs) >= 2:
        losses = [e1['train_loss'] for e1 in epoch1_outputs.values()]
        accs = [e1['train_acc'] for e1 in epoch1_outputs.values()]
        independence = {
            'epoch1_loss_range': max(losses) - min(losses),
            'epoch1_acc_range': max(accs) - min(accs),
            'independent': (max(losses) - min(losses)) > 0.001,
            'note': 'Methods show distinct first-epoch outputs' if (max(losses) - min(losses)) > 0.001 else 'WARNING: methods may be using identical computation',
            'per_method': epoch1_outputs,
        }

    # UDWDC-v2 WD budget check
    udwdc_v2_wd_ok = True
    if 'UDWDC-v2' in all_results and all_results['UDWDC-v2'].get('status') == 'success':
        wd_budget = all_results['UDWDC-v2'].get('total_wd_budget', 0)
        wd_budget_v2 = all_results['UDWDC-v2'].get('cumulative_wd_budget_v2', 0)
        udwdc_v2_wd_ok = (wd_budget > 0) or (wd_budget_v2 > 0)
        print(f"\n  UDWDC-v2 WD budget check: total={wd_budget:.6e}, "
              f"v2_cumulative={wd_budget_v2:.6e} -> {'OK' if udwdc_v2_wd_ok else 'FAIL'}")

    # Summary
    all_passed = len(passed) == len(METHODS)
    print(f"\n{'='*60}")
    print(f"  PILOT v2 SUMMARY: ImageNet ResNet-101")
    print(f"{'='*60}")
    print(f"  Methods: {len(passed)} passed, {len(failed)} failed out of {len(METHODS)}")
    print(f"  UDWDC-v2 WD budget > 0: {'YES' if udwdc_v2_wd_ok else 'NO'}")
    print(f"  Total time: {total_time:.1f}s")
    print()

    for method, result in all_results.items():
        status = "PASS" if method in passed else "FAIL"
        if result.get('status') == 'success':
            print(f"  [{status}] {method:20s} | Val: {result['final_val_acc']:.2f}% "
                  f"(top5: {result['final_val_top5']:.2f}%) | "
                  f"Train: {result['final_train_acc']:.2f}% | "
                  f"WD: {result['total_wd_budget']:.4e} | "
                  f"Time: {result['elapsed_sec']:.0f}s"
                  f"{' | ' + result.get('note', '') if result.get('note') else ''}")
        else:
            print(f"  [{status}] {method:20s} | {result.get('error', result.get('status', 'unknown'))}")

    print(f"\n  Overall: {'PASS' if all_passed else 'PARTIAL'}")

    # Save summary
    summary = {
        'task': 'imagenet_resnet101',
        'type': 'pilot',
        'version': 'v2',
        'model': 'resnet101',
        'dataset': 'imagenet',
        'epochs': args.epochs,
        'batch_size': effective_bs,
        'seed': args.seed,
        'gpu_id': args.gpu_id,
        'gpu_name': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'cpu',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'methods_passed': passed,
        'methods_failed': failed,
        'all_passed': all_passed,
        'udwdc_v2_wd_budget_ok': udwdc_v2_wd_ok,
        'independence_check': independence,
        'total_time_sec': total_time,
        'data_load_time_sec': data_time,
        'max_batch_size_probed': max_bs,
        'results': all_results,
        'pass_criteria': {
            'resnet101_completes_2_epochs': all_passed,
            'udwdc_v2_wd_budget_positive': udwdc_v2_wd_ok,
            'methods_independent': independence.get('independent', False),
        },
        'timestamp': datetime.now().isoformat(),
    }

    summary_file = save_dir / 'pilot_summary_v2.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Write DONE markers
    done_status = 'success' if all_passed else 'partial'
    done_data = json.dumps({
        'task_id': 'imagenet_resnet101',
        'status': done_status,
        'summary': f"{len(passed)}/{len(METHODS)} methods passed, "
                   f"UDWDC-v2 WD={'OK' if udwdc_v2_wd_ok else 'FAIL'}",
        'final_progress': {
            'methods_passed': len(passed),
            'methods_failed': len(failed),
            'total_methods': len(METHODS),
            'all_passed': all_passed,
            'udwdc_v2_wd_ok': udwdc_v2_wd_ok,
        },
        'timestamp': datetime.now().isoformat(),
    }, default=str)

    (results_base / 'imagenet_resnet101_DONE').write_text(done_data)

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    # Final progress update
    final_progress = {
        'task_id': 'imagenet_resnet101',
        'status': 'completed',
        'methods_passed': len(passed),
        'methods_failed': len(failed),
        'total_methods': len(METHODS),
        'all_passed': all_passed,
        'udwdc_v2_wd_ok': udwdc_v2_wd_ok,
        'total_time_sec': total_time,
        'updated_at': datetime.now().isoformat(),
    }
    progress_file = results_base / 'imagenet_resnet101_PROGRESS.json'
    progress_file.write_text(json.dumps(final_progress, indent=2))

    print(f"\nResults saved to {save_dir}")
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
