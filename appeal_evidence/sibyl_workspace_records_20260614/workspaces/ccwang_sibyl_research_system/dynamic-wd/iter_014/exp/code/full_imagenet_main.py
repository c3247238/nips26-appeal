#!/usr/bin/env python3
"""FULL experiment: imagenet_main — Phase 4 main ImageNet comparison.

Train ResNet-50 on ImageNet-1K for 90 epochs (STANDARD regime) with 8 methods:
  FixedWD, CWD, SWD, CPR, DefazioCorrective, NoWD, UDWDC, UDWDC-v2.
SGD momentum=0.9, LR=0.1, cosine annealing, batch_size=256.
Seeds: 42, 123, 456, 789, 2024 (5 seeds).

Log full per-layer diagnostics every epoch.

CRITICAL CONTROLS:
  (1) Verify first-epoch outputs differ across methods (prevent data corruption)
  (2) Track total WD budget for every run
  (3) UDWDC-v2 must have WD budget > 0

Usage:
    CUDA_VISIBLE_DEVICES=6 conda run -n sibyl_dynamic-wd python full_imagenet_main.py

Supports resume: skips completed (method, seed) pairs found in save_dir.
"""

import argparse
import json
import os
import sys
import time
import math
import gc
import traceback
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from diagnostics.logger import DiagnosticLogger

# ImageNet data loading
import io
import glob
import pyarrow.parquet as pq
import pyarrow as pa
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

# ── Task Configuration ──
TASK_ID = "imagenet_main"
TOTAL_EPOCHS = 90
SEEDS = [42, 123, 456, 789, 2024]
BATCH_SIZE = 256  # Will be probed
BASE_LR = 0.1
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
MODEL_NAME = "resnet50"
DATASET = "imagenet"
NUM_CLASSES = 1000

METHODS = ["FixedWD", "CWD", "SWD", "CPR", "DefazioCorrective", "NoWD", "UDWDC", "UDWDC-v2"]

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

WORKSPACE = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current")
SAVE_DIR = WORKSPACE / "exp" / "results" / "full" / "phase4_imagenet_main"
RESULTS_DIR = WORKSPACE / "exp" / "results"
DATA_ROOT = "/home/ccwang/dataset/imagenet-1k"


class ImageNetFullDataset(Dataset):
    """Full ImageNet dataset loading all parquet shards."""

    def __init__(self, data_dir, split='train', transform=None, seed=42):
        self.transform = transform
        self.split = split

        pattern = os.path.join(data_dir, 'data', f'{split}-*.parquet')
        shard_files = sorted(glob.glob(pattern))

        if not shard_files:
            raise FileNotFoundError(f"No parquet files found matching {pattern}")

        print(f"  Loading {len(shard_files)} shards for {split}...")
        load_start = time.time()

        tables = []
        for i, f in enumerate(shard_files):
            tables.append(pq.read_table(f))
            if (i + 1) % 50 == 0:
                print(f"    Loaded {i+1}/{len(shard_files)} shards...")

        full_table = pa.concat_tables(tables)

        self.images = full_table.column('image')
        self.labels = full_table.column('label').to_pylist()
        self.n_samples = len(self.labels)

        load_time = time.time() - load_start
        print(f"  {split}: {self.n_samples} samples from {len(shard_files)} shards "
              f"({load_time:.1f}s)")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img_struct = self.images[idx].as_py()
        img_bytes = img_struct['bytes']
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def set_seed(seed):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # True for consistent input sizes


def cosine_lr(optimizer_wrapper, epoch, total_epochs, base_lr):
    """Cosine annealing LR schedule."""
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
    optimizer_wrapper.set_lr(lr)
    return lr


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch with progress reporting."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = len(loader)

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # Print progress every 200 batches
        if (batch_idx + 1) % 200 == 0:
            avg_loss = total_loss / total
            acc = 100.0 * correct / total
            print(f"    Batch {batch_idx+1}/{n_batches} | "
                  f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%", flush=True)

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Evaluate on validation set with top-1 and top-5 accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    correct_top5 = 0
    total = 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device, non_blocking=True), targets.to(device, non_blocking=True)
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


def find_max_batch_size_train(device, input_size=(3, 224, 224), start=512, min_bs=32):
    """Find max batch size for full training step (forward + backward + optimizer)."""
    print(f"\n[VRAM Probe] Finding max batch size for ResNet-50 training...")
    model = create_model('resnet50', num_classes=1000).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    high, best = start, min_bs
    low = min_bs
    while low <= high:
        mid = (low + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            x = torch.randn(mid, *input_size, device=device)
            y = torch.randint(0, 1000, (mid,), device=device)
            opt.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            opt.step()
            del x, y, out, loss
            torch.cuda.empty_cache()
            best = mid
            low = mid + 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
            gc.collect()

    del model, opt, criterion
    torch.cuda.empty_cache()
    gc.collect()

    # Use 85% for safety, round to multiple of 8
    selected = max(min_bs, int(best * 0.85))
    selected = (selected // 8) * 8

    print(f"  Max train BS: {best}, Selected: {selected} (85%, rounded to 8)")
    return selected, best


def is_run_completed(save_dir, method, seed):
    """Check if a (method, seed) run is already completed."""
    result_file = save_dir / f"{method}_seed{seed}_result.json"
    if result_file.exists():
        try:
            data = json.loads(result_file.read_text())
            if data.get('status') == 'success' and data.get('epochs_completed', 0) >= TOTAL_EPOCHS:
                return True
        except (json.JSONDecodeError, KeyError):
            pass
    return False


def get_checkpoint_path(save_dir, method, seed):
    """Get checkpoint file path for a (method, seed) run."""
    return save_dir / f"{method}_seed{seed}_checkpoint.pt"


def run_single(method, seed, train_loader, val_loader, device, save_dir,
               batch_size, total_completed, total_runs):
    """Train a single (method, seed) configuration for 90 epochs."""
    set_seed(seed)

    run_save_dir = save_dir / f"{method}_seed{seed}"
    run_save_dir.mkdir(parents=True, exist_ok=True)

    model = create_model('resnet50', num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    config = METHOD_CONFIGS[method]
    wd = config.get('weight_decay', WEIGHT_DECAY)
    opt_kwargs = {}
    if method in ('UDWDC', 'UDWDC-v2'):
        opt_kwargs = {'K_p': config['K_p'], 'K_i': config['K_i'], 'K_d': config['K_d']}

    wd_optimizer = create_optimizer(
        method, model, lr=BASE_LR, momentum=MOMENTUM, weight_decay=wd, **opt_kwargs
    )
    criterion = nn.CrossEntropyLoss()

    logger = DiagnosticLogger(
        save_dir=str(run_save_dir), method=method, seed=seed,
        model_name=MODEL_NAME, dataset=DATASET
    )

    # Check for checkpoint to resume from
    start_epoch = 0
    ckpt_path = get_checkpoint_path(save_dir, method, seed)
    if ckpt_path.exists():
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt['model_state_dict'])
            wd_optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            start_epoch = ckpt['epoch'] + 1
            logger.epoch_metrics = ckpt.get('epoch_metrics', [])
            logger.trajectories = ckpt.get('trajectories', {})
            print(f"  Resuming from epoch {start_epoch}")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}, starting fresh")
            start_epoch = 0

    print(f"\n{'='*70}")
    print(f"  [{total_completed+1}/{total_runs}] Method: {method} | Seed: {seed} | "
          f"Params: {n_params:,} | WD: {wd} | Epochs: {start_epoch}->{TOTAL_EPOCHS}")
    print(f"{'='*70}")

    start_time = time.time()
    best_acc = 0.0
    best_top5 = 0.0
    epoch_results = []
    total_wd_budget = 0.0
    epoch1_outputs = None

    for epoch in range(start_epoch, TOTAL_EPOCHS):
        epoch_start = time.time()
        lr_now = cosine_lr(wd_optimizer, epoch, TOTAL_EPOCHS, BASE_LR)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device,
            epoch, TOTAL_EPOCHS
        )
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        # Log diagnostics
        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, val_loss, train_acc, val_acc, lr=lr_now)

        # Track WD budget
        epoch_wd = sum(d.get('effective_wd', 0.0) for d in diagnostics.values())
        total_wd_budget += epoch_wd

        # UDWDC-v2 specific checks
        if method == 'UDWDC-v2' and hasattr(wd_optimizer, 'end_epoch_check'):
            wd_budget_epoch = wd_optimizer.end_epoch_check()

        if val_acc > best_acc:
            best_acc = val_acc
            best_top5 = val_top5

        epoch_time = time.time() - epoch_start
        elapsed = time.time() - start_time

        # Capture epoch-1 outputs for independence check
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
            'epoch_time_sec': epoch_time,
        })

        # Print progress
        eta_total = epoch_time * (TOTAL_EPOCHS - epoch - 1)
        print(f"  [{method}|s{seed}] Epoch {epoch+1}/{TOTAL_EPOCHS} | LR: {lr_now:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Top5: {val_top5:.2f}%) | "
              f"Loss: {train_loss:.4f} | WD: {epoch_wd:.4e} | "
              f"Time: {epoch_time:.0f}s | ETA: {eta_total/60:.0f}min", flush=True)

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == TOTAL_EPOCHS - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': wd_optimizer.state_dict(),
                'epoch_metrics': logger.epoch_metrics,
                'trajectories': logger.trajectories,
                'best_acc': best_acc,
                'total_wd_budget': total_wd_budget,
            }, ckpt_path)

        # Write progress file for system monitor
        progress_data = {
            'task_id': TASK_ID,
            'epoch': epoch + 1,
            'total_epochs': TOTAL_EPOCHS,
            'step': 0,
            'total_steps': 0,
            'loss': train_loss,
            'metric': {
                'val_acc': val_acc,
                'val_top5': val_top5,
                'best_acc': best_acc,
                'method': method,
                'seed': seed,
                'total_runs': total_runs,
                'completed_runs': total_completed,
            },
            'updated_at': datetime.now().isoformat(),
        }
        (RESULTS_DIR / f'{TASK_ID}_PROGRESS.json').write_text(
            json.dumps(progress_data, indent=2))

    elapsed = time.time() - start_time
    logger.save()

    # Get UDWDC-v2 cumulative WD budget
    cumulative_wd = 0.0
    if method == 'UDWDC-v2' and hasattr(wd_optimizer, 'get_cumulative_wd_budget'):
        cumulative_wd = wd_optimizer.get_cumulative_wd_budget()

    # Build result
    result = {
        'method': method,
        'model': MODEL_NAME,
        'dataset': DATASET,
        'seed': seed,
        'epochs_completed': TOTAL_EPOCHS,
        'batch_size': batch_size,
        'lr': BASE_LR,
        'weight_decay': wd,
        'best_val_acc': best_acc,
        'best_val_top5': best_top5,
        'final_val_acc': epoch_results[-1]['val_acc'] if epoch_results else 0,
        'final_val_top5': epoch_results[-1]['val_top5'] if epoch_results else 0,
        'final_train_acc': epoch_results[-1]['train_acc'] if epoch_results else 0,
        'final_train_loss': epoch_results[-1]['train_loss'] if epoch_results else 0,
        'total_wd_budget': total_wd_budget,
        'cumulative_wd_budget_v2': cumulative_wd,
        'gen_gap': (epoch_results[-1]['train_acc'] - epoch_results[-1]['val_acc']
                    if epoch_results else 0),
        'elapsed_sec': elapsed,
        'epoch1_outputs': epoch1_outputs,
        'status': 'success',
        'timestamp': datetime.now().isoformat(),
    }

    # Save individual result
    result_file = save_dir / f"{method}_seed{seed}_result.json"
    result_file.write_text(json.dumps(result, indent=2))

    # Save epoch-by-epoch results
    epochs_file = save_dir / f"{method}_seed{seed}_epochs.json"
    epochs_file.write_text(json.dumps(epoch_results, indent=2))

    # Free GPU memory
    del model, wd_optimizer
    torch.cuda.empty_cache()
    gc.collect()

    # Remove checkpoint after successful completion
    if ckpt_path.exists():
        ckpt_path.unlink()

    return result


def update_gpu_progress(completed_runs, total_runs, start_time, save_dir):
    """Update gpu_progress.json with current task status."""
    gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gpu_progress_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    # Update running status with progress info
    if TASK_ID in gp.get('running', {}):
        gp['running'][TASK_ID]['progress'] = f"{completed_runs}/{total_runs} runs"

    gpu_progress_file.write_text(json.dumps(gp, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Full ImageNet Main Experiment')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='Batch size (0=auto probe)')
    parser.add_argument('--num_workers', type=int, default=12)
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU device ID (after CUDA_VISIBLE_DEVICES mapping)')
    parser.add_argument('--methods', type=str, default='',
                        help='Comma-separated method subset to run (empty=all)')
    parser.add_argument('--log_suffix', type=str, default='',
                        help='Suffix for log file to avoid overwriting main log')
    args = parser.parse_args()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID file
    pid_file = RESULTS_DIR / f'{TASK_ID}.pid'
    pid_file.write_text(str(os.getpid()))

    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(args.gpu_id)
        mem_total = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9
        print(f"GPU: {gpu_name}, {mem_total:.1f} GB")

    # ---- VRAM Probing ----
    if args.batch_size == 0:
        selected_bs, max_bs = find_max_batch_size_train(device)
        args.batch_size = selected_bs

        # Save GPU profile
        gpu_profile = {
            'gpu_name': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'cpu',
            'vram_total_mb': torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e6 if torch.cuda.is_available() else 0,
            'max_batch_size': max_bs,
            'selected_batch_size': selected_bs,
            'utilization_pct': round(selected_bs / max(max_bs, 1) * 100, 1),
        }
        (RESULTS_DIR / f'{TASK_ID}_gpu_profile.json').write_text(
            json.dumps(gpu_profile, indent=2))
    else:
        print(f"Using specified batch_size={args.batch_size}")

    batch_size = args.batch_size
    print(f"\nConfiguration: {TOTAL_EPOCHS} epochs, batch_size={batch_size}, "
          f"lr={BASE_LR}, seeds={SEEDS}")
    print(f"Methods: {METHODS}")
    print(f"Total runs: {len(METHODS) * len(SEEDS)} = {len(METHODS)} methods x {len(SEEDS)} seeds")

    # ---- Load full ImageNet data ----
    print("\n" + "="*70)
    print("  Loading full ImageNet-1K dataset...")
    print("="*70)

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

    data_start = time.time()
    train_dataset = ImageNetFullDataset(
        DATA_ROOT, split='train', transform=transform_train
    )
    val_dataset = ImageNetFullDataset(
        DATA_ROOT, split='validation', transform=transform_val
    )
    data_time = time.time() - data_start
    print(f"Data loaded in {data_time:.1f}s "
          f"({len(train_dataset)} train, {len(val_dataset)} val)")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=True,
    )

    # ---- Build run schedule ----
    total_runs = len(METHODS) * len(SEEDS)
    # Filter methods if --methods specified
    active_methods = METHODS
    if args.methods:
        active_methods = [m.strip() for m in args.methods.split(',')]
        print(f"Method filter active: {active_methods}")

    all_runs = [(m, s) for m in active_methods for s in SEEDS]
    total_runs = len(all_runs)

    # Check which runs are already completed
    completed_runs = []
    pending_runs = []
    for method, seed in all_runs:
        if is_run_completed(SAVE_DIR, method, seed):
            completed_runs.append((method, seed))
        else:
            pending_runs.append((method, seed))

    print(f"\nRun status: {len(completed_runs)} completed, {len(pending_runs)} pending")
    if completed_runs:
        print(f"  Completed: {', '.join(f'{m}/s{s}' for m, s in completed_runs)}")
    if pending_runs:
        print(f"  Pending: {', '.join(f'{m}/s{s}' for m, s in pending_runs)}")

    # ---- Execute pending runs ----
    overall_start = time.time()
    all_results = {}

    # Load completed results
    for method, seed in completed_runs:
        result_file = SAVE_DIR / f"{method}_seed{seed}_result.json"
        try:
            all_results[(method, seed)] = json.loads(result_file.read_text())
        except:
            pass

    for run_idx, (method, seed) in enumerate(pending_runs):
        n_completed = len(completed_runs) + run_idx
        try:
            result = run_single(
                method, seed, train_loader, val_loader, device, SAVE_DIR,
                batch_size, n_completed, total_runs
            )
            all_results[(method, seed)] = result

            print(f"\n  Completed {method}/s{seed}: "
                  f"Val {result['final_val_acc']:.2f}% "
                  f"(Top5: {result['final_val_top5']:.2f}%) | "
                  f"WD budget: {result['total_wd_budget']:.4e} | "
                  f"Time: {result['elapsed_sec']/60:.1f}min")

        except torch.cuda.OutOfMemoryError:
            print(f"\n  OOM on {method}/s{seed}! Reducing batch size...")
            torch.cuda.empty_cache()
            gc.collect()

            # Retry with smaller batch
            small_bs = max(32, batch_size // 2)
            small_train_loader = DataLoader(
                train_dataset, batch_size=small_bs, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True,
                persistent_workers=True,
            )
            small_val_loader = DataLoader(
                val_dataset, batch_size=small_bs, shuffle=False,
                num_workers=args.num_workers, pin_memory=True,
                persistent_workers=True,
            )
            try:
                result = run_single(
                    method, seed, small_train_loader, small_val_loader,
                    device, SAVE_DIR, small_bs, n_completed, total_runs
                )
                result['note'] = f'Reduced batch_size to {small_bs} due to OOM'
                all_results[(method, seed)] = result
            except Exception as e2:
                traceback.print_exc()
                error_result = {
                    'method': method, 'seed': seed, 'status': 'failed',
                    'error': str(e2), 'timestamp': datetime.now().isoformat()
                }
                all_results[(method, seed)] = error_result
                result_file = SAVE_DIR / f"{method}_seed{seed}_result.json"
                result_file.write_text(json.dumps(error_result, indent=2))

        except Exception as e:
            traceback.print_exc()
            error_result = {
                'method': method, 'seed': seed, 'status': 'failed',
                'error': str(e), 'timestamp': datetime.now().isoformat()
            }
            all_results[(method, seed)] = error_result
            result_file = SAVE_DIR / f"{method}_seed{seed}_result.json"
            result_file.write_text(json.dumps(error_result, indent=2))
            torch.cuda.empty_cache()
            gc.collect()

    total_time = time.time() - overall_start

    # ---- Compile summary ----
    print(f"\n{'='*70}")
    print(f"  FULL EXPERIMENT SUMMARY: ImageNet Main Comparison")
    print(f"  Model: ResNet-50 | Dataset: ImageNet-1K | Epochs: {TOTAL_EPOCHS}")
    print(f"  Seeds: {SEEDS} | Batch size: {batch_size}")
    print(f"{'='*70}")

    # Aggregate by method
    method_summary = {}
    for method in METHODS:
        seed_results = []
        for seed in SEEDS:
            r = all_results.get((method, seed), {})
            if r.get('status') == 'success':
                seed_results.append(r)

        if seed_results:
            accs = [r['best_val_acc'] for r in seed_results]
            top5s = [r['best_val_top5'] for r in seed_results]
            wd_budgets = [r['total_wd_budget'] for r in seed_results]
            gen_gaps = [r['gen_gap'] for r in seed_results]
            times = [r['elapsed_sec'] for r in seed_results]

            method_summary[method] = {
                'n_seeds': len(seed_results),
                'top1_mean': float(np.mean(accs)),
                'top1_std': float(np.std(accs)),
                'top5_mean': float(np.mean(top5s)),
                'top5_std': float(np.std(top5s)),
                'gen_gap_mean': float(np.mean(gen_gaps)),
                'gen_gap_std': float(np.std(gen_gaps)),
                'wd_budget_mean': float(np.mean(wd_budgets)),
                'wd_budget_std': float(np.std(wd_budgets)),
                'time_mean_min': float(np.mean(times)) / 60,
                'per_seed': {
                    str(r['seed']): {
                        'top1': r['best_val_acc'],
                        'top5': r['best_val_top5'],
                        'gen_gap': r['gen_gap'],
                        'wd_budget': r['total_wd_budget'],
                    }
                    for r in seed_results
                },
            }

            print(f"\n  {method:20s} | "
                  f"Top-1: {np.mean(accs):.2f} +/- {np.std(accs):.2f}% | "
                  f"Top-5: {np.mean(top5s):.2f} +/- {np.std(top5s):.2f}% | "
                  f"GenGap: {np.mean(gen_gaps):.2f}% | "
                  f"WD: {np.mean(wd_budgets):.4e} | "
                  f"({len(seed_results)}/{len(SEEDS)} seeds)")
        else:
            method_summary[method] = {
                'n_seeds': 0,
                'status': 'all_failed',
            }
            print(f"  {method:20s} | ALL FAILED")

    # Independence check: epoch-1 outputs across methods (seed 42)
    independence = {}
    for method in METHODS:
        r = all_results.get((method, 42), {})
        if r.get('epoch1_outputs'):
            independence[method] = r['epoch1_outputs']

    if len(independence) >= 2:
        losses = [v['train_loss'] for v in independence.values()]
        accs = [v['train_acc'] for v in independence.values()]
        loss_range = max(losses) - min(losses)
        acc_range = max(accs) - min(accs)
        print(f"\n  Independence check (seed 42): "
              f"loss_range={loss_range:.4f}, acc_range={acc_range:.2f}% -> "
              f"{'PASS' if loss_range > 0.001 or acc_range > 0.01 else 'FAIL'}")

    # UDWDC-v2 WD budget check
    for seed in SEEDS:
        r = all_results.get(('UDWDC-v2', seed), {})
        if r.get('status') == 'success':
            wd = r.get('total_wd_budget', 0)
            print(f"  UDWDC-v2 seed {seed}: WD budget = {wd:.4e} "
                  f"{'OK' if wd > 0 else 'WARNING: <= 0'}")

    # ---- Save comprehensive summary ----
    summary = {
        'task': TASK_ID,
        'type': 'full',
        'model': MODEL_NAME,
        'dataset': DATASET,
        'epochs': TOTAL_EPOCHS,
        'batch_size': batch_size,
        'lr': BASE_LR,
        'seeds': SEEDS,
        'methods': METHODS,
        'total_runs': total_runs,
        'successful_runs': sum(1 for r in all_results.values() if r.get('status') == 'success'),
        'failed_runs': sum(1 for r in all_results.values() if r.get('status') != 'success'),
        'method_summary': method_summary,
        'independence_check': independence,
        'total_time_sec': total_time,
        'total_time_hours': total_time / 3600,
        'timestamp': datetime.now().isoformat(),
    }

    summary_file = SAVE_DIR / 'summary.json'
    summary_file.write_text(json.dumps(summary, indent=2, default=str))

    # Write markdown summary
    md_lines = [
        "# ImageNet Main Experiment Results",
        "",
        f"**Model**: ResNet-50 | **Dataset**: ImageNet-1K | **Epochs**: {TOTAL_EPOCHS}",
        f"**Seeds**: {SEEDS} | **Batch size**: {batch_size} | **LR**: {BASE_LR}",
        f"**Total time**: {total_time/3600:.1f} hours",
        "",
        "## Results Table",
        "",
        "| Method | Top-1 (%) | Top-5 (%) | Gen Gap (%) | WD Budget | BEM |",
        "|--------|-----------|-----------|-------------|-----------|-----|",
    ]

    # Compute BEM (accuracy improvement per unit WD budget)
    fixed_wd_acc = method_summary.get('FixedWD', {}).get('top1_mean', 0)
    for method in METHODS:
        ms = method_summary.get(method, {})
        if ms.get('n_seeds', 0) > 0:
            acc = ms['top1_mean']
            top5 = ms['top5_mean']
            gg = ms['gen_gap_mean']
            wd_b = ms['wd_budget_mean']
            bem = (acc - fixed_wd_acc) / max(wd_b, 1e-10) if wd_b > 0 else 0
            md_lines.append(
                f"| {method} | {acc:.2f} +/- {ms['top1_std']:.2f} | "
                f"{top5:.2f} +/- {ms['top5_std']:.2f} | "
                f"{gg:.2f} +/- {ms['gen_gap_std']:.2f} | "
                f"{wd_b:.4e} | {bem:.4f} |"
            )
        else:
            md_lines.append(f"| {method} | FAILED | - | - | - | - |")

    md_lines.extend([
        "",
        "## Training Curves",
        "",
        "Per-method per-seed training curves saved in `phase4_imagenet_main/`.",
        "",
    ])

    summary_md = SAVE_DIR / 'summary.md'
    summary_md.write_text('\n'.join(md_lines))

    # ---- Write DONE marker ----
    n_success = sum(1 for r in all_results.values() if r.get('status') == 'success')
    done_data = {
        'task_id': TASK_ID,
        'status': 'success' if n_success == total_runs else 'partial',
        'summary': f"{n_success}/{total_runs} runs completed, "
                   f"{total_time/3600:.1f}h total",
        'final_progress': {
            'completed_runs': n_success,
            'total_runs': total_runs,
            'methods': len(METHODS),
            'seeds': len(SEEDS),
        },
        'timestamp': datetime.now().isoformat(),
    }
    (RESULTS_DIR / f'{TASK_ID}_DONE').write_text(json.dumps(done_data, indent=2))

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    # Final progress update
    final_progress = {
        'task_id': TASK_ID,
        'epoch': TOTAL_EPOCHS,
        'total_epochs': TOTAL_EPOCHS,
        'step': 0,
        'total_steps': 0,
        'loss': 0,
        'metric': {
            'completed_runs': n_success,
            'total_runs': total_runs,
            'status': 'completed',
        },
        'updated_at': datetime.now().isoformat(),
    }
    (RESULTS_DIR / f'{TASK_ID}_PROGRESS.json').write_text(
        json.dumps(final_progress, indent=2))

    # ---- Update gpu_progress.json ----
    gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gpu_progress_file.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp.get('completed', []):
        gp.setdefault('completed', []).append(TASK_ID)
    if TASK_ID in gp.get('running', {}):
        del gp['running'][TASK_ID]

    gp.setdefault('timings', {})[TASK_ID] = {
        'planned_min': 180,
        'actual_min': int(total_time / 60),
        'start_time': datetime.fromtimestamp(
            time.time() - total_time).isoformat(),
        'end_time': datetime.now().isoformat(),
        'config_snapshot': {
            'model': MODEL_NAME,
            'batch_size': batch_size,
            'dataset': DATASET,
            'epochs': TOTAL_EPOCHS,
            'n_methods': len(METHODS),
            'n_seeds': len(SEEDS),
            'total_runs': total_runs,
            'successful_runs': n_success,
            'gpu_model': torch.cuda.get_device_name(args.gpu_id) if torch.cuda.is_available() else 'cpu',
            'gpu_count': 1,
        },
    }

    gpu_progress_file.write_text(json.dumps(gp, indent=2))

    print(f"\n{'='*70}")
    print(f"  DONE: {n_success}/{total_runs} runs completed in {total_time/3600:.1f}h")
    print(f"  Results saved to {SAVE_DIR}")
    print(f"{'='*70}")

    return 0 if n_success == total_runs else 1


if __name__ == '__main__':
    sys.exit(main())
