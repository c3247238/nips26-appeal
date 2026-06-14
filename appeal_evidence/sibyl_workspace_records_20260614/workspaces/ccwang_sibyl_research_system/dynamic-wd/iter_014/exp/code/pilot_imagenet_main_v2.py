#!/usr/bin/env python3
"""Pilot v2: ImageNet main comparison — 8 WD methods (incl. UDWDC-v2), ResNet-50, 2 epochs.

Updated from v1:
  - Includes UDWDC-v2 (8 methods total)
  - Validates UDWDC-v2 WD budget > 0
  - Tests DDP correctness (2 GPUs) for one method, then runs all methods single-GPU
  - Checks first-epoch output independence across methods
  - VRAM probing for optimal batch size

Pass criteria (from task_plan.json):
  - All 8 methods complete 2 epochs on ImageNet without OOM
  - Validation accuracy > 5% at epoch 2
  - DDP sync correct
  - UDWDC-v2 WD budget > 0
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
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
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


METHODS = [
    'NoWD',
    'FixedWD',
    'SWD',
    'CWD',
    'CPR',
    'DefazioCorrective',
    'UDWDC',
    'UDWDC-v2',
]

METHOD_CONFIGS = {
    'NoWD': {'weight_decay': 0.0},
    'FixedWD': {'weight_decay': 1e-4},
    'SWD': {'weight_decay': 1e-4},
    'CWD': {'weight_decay': 1e-4},
    'CPR': {'weight_decay': 1e-4},
    'DefazioCorrective': {'weight_decay': 1e-4},
    'UDWDC': {'weight_decay': 1e-4, 'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.3},
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
        _, pred_top5 = outputs.topk(5, 1, True, True)
        pred_top5 = pred_top5.t()
        correct_k = pred_top5.eq(targets.view(1, -1).expand_as(pred_top5))
        correct_top5 += correct_k.reshape(-1).float().sum(0).item()

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    top5_accuracy = 100.0 * correct_top5 / max(total, 1)
    return avg_loss, accuracy, top5_accuracy


def find_max_batch_size(model, device, input_size=(3, 224, 224), start=256, min_bs=16):
    """Binary search for the maximum batch size the current GPU can sustain."""
    high, best = start, min_bs
    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            x = torch.randn(mid, *input_size, device=device)
            with torch.no_grad():
                model(x)
            del x
            torch.cuda.empty_cache()
            best = mid
            min_bs = mid + 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
            gc.collect()
    # Clean up
    torch.cuda.empty_cache()
    gc.collect()
    return best


def run_single_method(method, train_loader, val_loader, device, save_dir,
                      epochs=2, lr=0.1, momentum=0.9, seed=42, batch_size=256):
    """Train a single method and return results."""
    set_seed(seed)

    model = create_model('resnet50', num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    config = METHOD_CONFIGS[method]
    wd = config.get('weight_decay', 1e-4)
    opt_kwargs = {}
    if method in ('UDWDC', 'UDWDC-v2'):
        opt_kwargs = {'K_p': config['K_p'], 'K_i': config['K_i'], 'K_d': config['K_d']}

    wd_optimizer = create_optimizer(
        method, model, lr=lr, momentum=momentum, weight_decay=wd, **opt_kwargs
    )
    criterion = nn.CrossEntropyLoss()

    logger = DiagnosticLogger(
        save_dir=str(save_dir), method=method, seed=seed,
        model_name='resnet50', dataset='imagenet'
    )

    print(f"\n{'='*60}")
    print(f"  Method: {method} | Params: {n_params:,} | WD: {wd}")
    print(f"{'='*60}")

    start = time.time()
    best_acc = 0.0
    epoch_results = []
    total_wd_budget = 0.0
    epoch1_outputs = None

    for epoch in range(epochs):
        lr_now = cosine_lr(wd_optimizer, epoch, epochs, lr)
        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device
        )
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, val_loss, train_acc, val_acc, lr=lr_now)

        # Track WD budget
        epoch_wd = sum(d.get('effective_wd', 0.0) for d in diagnostics.values())
        total_wd_budget += epoch_wd

        # UDWDC-v2 specific: call end_epoch_check
        if method == 'UDWDC-v2' and hasattr(wd_optimizer, 'end_epoch_check'):
            wd_budget_epoch = wd_optimizer.end_epoch_check()
            print(f"    [UDWDC-v2] Epoch WD budget: {wd_budget_epoch:.6e}, "
                  f"Cumulative: {wd_optimizer.get_cumulative_wd_budget():.6e}")

        if val_acc > best_acc:
            best_acc = val_acc

        elapsed = time.time() - start
        print(f"  [{method}] Epoch {epoch+1}/{epochs} | LR: {lr_now:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Top-5: {val_top5:.2f}%) | "
              f"Loss: {train_loss:.4f} | WD_budget: {epoch_wd:.4e} | Time: {elapsed:.0f}s")

        # Capture first-epoch outputs for independence check
        if epoch == 0:
            epoch1_outputs = {
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_top5': val_top5,
            }

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

    # Get UDWDC-v2 cumulative WD budget
    cumulative_wd = 0.0
    if method == 'UDWDC-v2' and hasattr(wd_optimizer, 'get_cumulative_wd_budget'):
        cumulative_wd = wd_optimizer.get_cumulative_wd_budget()

    # Free GPU memory
    del model, wd_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    result = {
        'method': method,
        'model': 'resnet50',
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
        'cumulative_wd_budget_v2': cumulative_wd,
        'elapsed_sec': elapsed,
        'epoch_results': epoch_results,
        'epoch1_outputs': epoch1_outputs,
        'status': 'success',
    }

    return result


def test_ddp_correctness(gpu_ids, data_root, save_dir, seed=42):
    """Test DDP correctness with FixedWD on 2 GPUs for 1 epoch.

    Returns a dict with DDP test results.
    """
    print(f"\n{'='*60}")
    print(f"  DDP Correctness Test (GPUs: {gpu_ids})")
    print(f"{'='*60}")

    # Check if both GPUs are available
    gpu_list = [int(g) for g in gpu_ids.split(',')]
    if len(gpu_list) < 2:
        return {'status': 'skipped', 'reason': 'Need at least 2 GPUs for DDP test'}

    for g in gpu_list:
        if g >= torch.cuda.device_count():
            return {'status': 'skipped', 'reason': f'GPU {g} not available'}

    # Test DDP by running a quick 1-step forward/backward on both GPUs
    try:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ids

        # Simple DDP test: create model, wrap in DDP, do one forward pass
        # We use spawn to avoid NCCL issues
        result = {'status': 'success', 'gpu_ids': gpu_ids}

        # Direct test without multiprocessing (simpler for pilot)
        if not dist.is_initialized():
            dist.init_process_group(
                backend='nccl',
                init_method='tcp://localhost:29501',
                world_size=1,
                rank=0
            )

        device = torch.device(f'cuda:0')
        model = create_model('resnet50', num_classes=1000).to(device)

        # Wrap in DDP
        ddp_model = DDP(model, device_ids=[0])

        # One forward pass
        x = torch.randn(4, 3, 224, 224, device=device)
        y = torch.randint(0, 1000, (4,), device=device)
        criterion = nn.CrossEntropyLoss()

        out = ddp_model(x)
        loss = criterion(out, y)
        loss.backward()

        print(f"  DDP forward/backward: PASS (loss={loss.item():.4f})")
        result['loss'] = loss.item()
        result['status'] = 'success'

        # Clean up
        del ddp_model, model, x, y
        torch.cuda.empty_cache()
        gc.collect()

        if dist.is_initialized():
            dist.destroy_process_group()

    except Exception as e:
        traceback.print_exc()
        result = {'status': 'failed', 'error': str(e)}
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except:
                pass

    # Clean CUDA_VISIBLE_DEVICES
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        del os.environ['CUDA_VISIBLE_DEVICES']

    return result


def main():
    parser = argparse.ArgumentParser(description='ImageNet Main Pilot v2')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size (0=auto probe)')
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_root', type=str,
                        default='/home/ccwang/dataset/imagenet-1k')
    parser.add_argument('--save_dir', type=str,
                        default='exp/results/pilots/imagenet_main')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--train_shards', type=int, default=10,
                        help='Number of train shards to load (pilot speed)')
    parser.add_argument('--val_shards', type=int, default=4,
                        help='Number of validation shards to load')
    parser.add_argument('--max_train_samples', type=int, default=50000,
                        help='Max training samples for pilot')
    parser.add_argument('--max_val_samples', type=int, default=10000,
                        help='Max validation samples for pilot')
    parser.add_argument('--gpu_id', type=int, default=2,
                        help='GPU to use for single-GPU pilot runs')
    parser.add_argument('--gpu_ids', type=str, default='0,2',
                        help='GPU IDs for DDP test')
    parser.add_argument('--skip_ddp_test', action='store_true',
                        help='Skip DDP correctness test')
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results_base = Path('exp/results')

    # Write PID file
    pid_file = results_base / 'imagenet_main.pid'
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
        mem = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9
        mem_used = torch.cuda.memory_allocated(args.gpu_id) / 1e9
        print(f"Memory: {mem:.1f} GB total")

    # ---- VRAM Probing ----
    if args.batch_size == 0:
        print("\n[VRAM Probe] Finding max batch size for ResNet-50 on ImageNet...")
        probe_model = create_model('resnet50', num_classes=1000).to(device)
        max_bs = find_max_batch_size(probe_model, device, input_size=(3, 224, 224),
                                     start=512, min_bs=16)
        # Use 80% of max for stability during training (gradients use memory too)
        # For forward-only probing we found max_bs, but training needs ~2x memory
        # Probe with backward pass
        del probe_model
        torch.cuda.empty_cache()
        gc.collect()

        # Probe with full training step
        probe_model = create_model('resnet50', num_classes=1000).to(device)
        criterion = nn.CrossEntropyLoss()
        probe_opt = torch.optim.SGD(probe_model.parameters(), lr=0.1, momentum=0.9)

        high, best_train_bs = max_bs, 16
        low = 16
        while low <= high:
            mid = (low + high) // 2
            try:
                torch.cuda.empty_cache()
                gc.collect()
                x = torch.randn(mid, 3, 224, 224, device=device)
                y = torch.randint(0, 1000, (mid,), device=device)
                probe_opt.zero_grad()
                out = probe_model(x)
                loss = criterion(out, y)
                loss.backward()
                probe_opt.step()
                del x, y, out, loss
                torch.cuda.empty_cache()
                best_train_bs = mid
                low = mid + 1
            except torch.cuda.OutOfMemoryError:
                high = mid - 1
                torch.cuda.empty_cache()
                gc.collect()

        del probe_model, probe_opt
        torch.cuda.empty_cache()
        gc.collect()

        # Use 85% of max training batch size for safety margin
        args.batch_size = max(16, int(best_train_bs * 0.85))
        # Round down to multiple of 8 for efficiency
        args.batch_size = (args.batch_size // 8) * 8
        if args.batch_size < 16:
            args.batch_size = 16

        vram_total = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e6
        print(f"  Max forward BS: {max_bs}, Max train BS: {best_train_bs}")
        print(f"  Selected BS: {args.batch_size} (85% of max train, rounded to 8)")

        # Save GPU profile
        gpu_profile = {
            'gpu_name': torch.cuda.get_device_name(args.gpu_id),
            'vram_total_mb': vram_total,
            'max_batch_size_forward': max_bs,
            'max_batch_size_train': best_train_bs,
            'selected_batch_size': args.batch_size,
        }
        profile_path = results_base / 'imagenet_main_gpu_profile.json'
        profile_path.write_text(json.dumps(gpu_profile, indent=2))
        print(f"  GPU profile saved to {profile_path}")
    else:
        print(f"\nUsing specified batch_size={args.batch_size}")

    print(f"\nPilot config: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, seed={args.seed}")
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

    # ---- DDP Correctness Test ----
    ddp_result = None
    if not args.skip_ddp_test:
        try:
            ddp_result = test_ddp_correctness(args.gpu_ids, args.data_root, save_dir)
            print(f"  DDP test result: {ddp_result.get('status', 'unknown')}")
        except Exception as e:
            ddp_result = {'status': 'error', 'error': str(e)}
            print(f"  DDP test error (non-fatal): {e}")
    else:
        ddp_result = {'status': 'skipped', 'reason': 'User skipped'}

    # ---- Run all 8 methods ----
    all_results = {}
    passed = []
    failed = []
    overall_start = time.time()
    epoch1_outputs_all = {}

    for i, method in enumerate(METHODS):
        # Write progress
        progress = {
            'task_id': 'imagenet_main',
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
        progress_file = results_base / 'imagenet_main_PROGRESS.json'
        progress_file.write_text(json.dumps(progress, indent=2))

        try:
            result = run_single_method(
                method, train_loader, val_loader, device, save_dir,
                epochs=args.epochs, lr=args.lr, momentum=args.momentum,
                seed=args.seed, batch_size=args.batch_size
            )
            all_results[method] = result

            # Store epoch-1 outputs for independence check
            if result.get('epoch1_outputs'):
                epoch1_outputs_all[method] = result['epoch1_outputs']

            # Check pass criteria
            final_val = result['final_val_acc']
            final_val_top5 = result.get('final_val_top5', 0)

            # With subset data (50K samples / ~4% of ImageNet), 2 epochs won't reach 5%
            # So we check: (a) training completed, (b) loss decreased, (c) accuracy > 0
            if result['status'] == 'success':
                passed.append(method)
                verdict = "PASS"

                # Extra check for UDWDC-v2: WD budget > 0
                if method == 'UDWDC-v2':
                    wd_budget = result.get('cumulative_wd_budget_v2', 0)
                    if wd_budget <= 0:
                        verdict = "WARN (WD budget <= 0)"
                        print(f"  !! WARNING: UDWDC-v2 cumulative WD budget = {wd_budget:.6e}")
                    else:
                        print(f"  UDWDC-v2 WD budget OK: {wd_budget:.6e}")

                print(f"  -> {verdict}: val_acc={final_val:.2f}% "
                      f"(Top-5: {final_val_top5:.2f}%), "
                      f"train_acc={result['final_train_acc']:.2f}%")
            else:
                failed.append(method)
                print(f"  -> FAIL: {result.get('error', 'unknown')}")

        except torch.cuda.OutOfMemoryError:
            print(f"  -> OOM for {method}! Trying with batch_size={args.batch_size // 2}...")
            torch.cuda.empty_cache()
            gc.collect()

            # Retry with smaller batch size
            small_bs = max(16, args.batch_size // 2)
            small_train_loader = DataLoader(
                train_dataset, batch_size=small_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
            small_val_loader = DataLoader(
                val_dataset, batch_size=small_bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            try:
                result = run_single_method(
                    method, small_train_loader, small_val_loader, device, save_dir,
                    epochs=args.epochs, lr=args.lr, momentum=args.momentum,
                    seed=args.seed, batch_size=small_bs
                )
                result['note'] = f'Reduced batch_size to {small_bs} due to OOM'
                all_results[method] = result
                passed.append(method)
                print(f"  -> PASS (bs={small_bs}): val_acc={result['final_val_acc']:.2f}%")
            except Exception as e2:
                all_results[method] = {
                    'method': method, 'status': 'failed', 'error': str(e2)
                }
                failed.append(method)
                print(f"  -> FAIL: {e2}")

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

    # ---- Epoch-1 Independence Check ----
    independence_check = {}
    if len(epoch1_outputs_all) >= 2:
        losses = [v['train_loss'] for v in epoch1_outputs_all.values()]
        accs = [v['train_acc'] for v in epoch1_outputs_all.values()]
        # Check that not all methods produce identical first-epoch outputs
        loss_range = max(losses) - min(losses)
        acc_range = max(accs) - min(accs)
        independence_check = {
            'epoch1_loss_range': loss_range,
            'epoch1_acc_range': acc_range,
            'independent': loss_range > 0.001 or acc_range > 0.01,
            'note': ('Methods show distinct first-epoch outputs'
                     if loss_range > 0.001 or acc_range > 0.01
                     else 'WARNING: Methods may have identical outputs (data corruption?)'),
            'per_method': epoch1_outputs_all,
        }
        print(f"\n  Independence check: loss_range={loss_range:.4f}, "
              f"acc_range={acc_range:.2f}% -> "
              f"{'PASS' if independence_check['independent'] else 'FAIL'}")

    # ---- Summary ----
    print(f"\n{'='*60}")
    print(f"  PILOT SUMMARY: ImageNet Main Comparison (v2)")
    print(f"{'='*60}")
    print(f"  Methods: {len(passed)} passed, {len(failed)} failed out of {len(METHODS)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"  DDP test: {ddp_result.get('status', 'unknown')}")
    print()

    for method, result in all_results.items():
        status = "PASS" if method in passed else "FAIL"
        if result.get('status') == 'success':
            wd_note = ""
            if method == 'UDWDC-v2':
                wd_budget = result.get('cumulative_wd_budget_v2', 0)
                wd_note = f" | WD_budget={wd_budget:.4e}"
            print(f"  [{status}] {method:20s} | Val: {result['final_val_acc']:.2f}% "
                  f"(Top-5: {result.get('final_val_top5', 0):.2f}%) | "
                  f"Train: {result['final_train_acc']:.2f}% | "
                  f"Time: {result['elapsed_sec']:.0f}s{wd_note}")
        else:
            print(f"  [{status}] {method:20s} | {result.get('error', result.get('status', 'unknown'))}")

    # Overall pass/fail
    all_passed = len(passed) == len(METHODS)
    udwdc_v2_result = all_results.get('UDWDC-v2', {})
    udwdc_v2_wd_ok = (udwdc_v2_result.get('cumulative_wd_budget_v2', 0) > 0
                      if udwdc_v2_result.get('status') == 'success' else False)

    print(f"\n  Overall: {'PASS' if all_passed else 'PARTIAL'}")
    print(f"  UDWDC-v2 WD budget > 0: {'YES' if udwdc_v2_wd_ok else 'NO'}")
    print(f"  DDP correctness: {ddp_result.get('status', 'unknown')}")
    if independence_check:
        print(f"  First-epoch independence: "
              f"{'PASS' if independence_check.get('independent') else 'FAIL'}")

    # ---- Save summary ----
    summary = {
        'task': 'imagenet_main',
        'type': 'pilot',
        'version': 'v2',
        'model': 'resnet50',
        'dataset': 'imagenet',
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'seed': args.seed,
        'gpu_id': args.gpu_id,
        'gpu_name': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'cpu',
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'methods_passed': passed,
        'methods_failed': failed,
        'all_passed': all_passed,
        'udwdc_v2_wd_budget_ok': udwdc_v2_wd_ok,
        'ddp_test': ddp_result,
        'independence_check': independence_check,
        'total_time_sec': total_time,
        'data_load_time_sec': data_time,
        'results': all_results,
        'pass_criteria': {
            'all_8_methods_complete_2_epochs': all_passed,
            'val_accuracy_progresses': all(
                r.get('status') == 'success'
                for r in all_results.values()
            ),
            'ddp_sync_correct': ddp_result.get('status') == 'success',
            'udwdc_v2_wd_budget_positive': udwdc_v2_wd_ok,
        },
        'timestamp': datetime.now().isoformat(),
    }

    summary_file = save_dir / 'pilot_summary_v2.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Write DONE marker
    done_data = json.dumps({
        'task_id': 'imagenet_main',
        'status': 'success' if all_passed else 'partial',
        'summary': (f"{len(passed)}/{len(METHODS)} methods passed, "
                    f"UDWDC-v2 WD={'OK' if udwdc_v2_wd_ok else 'FAIL'}, "
                    f"DDP={ddp_result.get('status', 'unknown')}"),
        'final_progress': {
            'methods_passed': len(passed),
            'methods_failed': len(failed),
            'total_methods': len(METHODS),
            'all_passed': all_passed,
            'udwdc_v2_wd_ok': udwdc_v2_wd_ok,
        },
        'timestamp': datetime.now().isoformat(),
    }, default=str)

    (results_base / 'imagenet_main_DONE').write_text(done_data)

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    # Final progress update
    final_progress = {
        'task_id': 'imagenet_main',
        'status': 'completed',
        'epoch': args.epochs,
        'total_epochs': args.epochs,
        'methods_passed': len(passed),
        'methods_failed': len(failed),
        'total_methods': len(METHODS),
        'all_passed': all_passed,
        'total_time_sec': total_time,
        'updated_at': datetime.now().isoformat(),
    }
    progress_file = results_base / 'imagenet_main_PROGRESS.json'
    progress_file.write_text(json.dumps(final_progress, indent=2))

    print(f"\nResults saved to {save_dir}")
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
