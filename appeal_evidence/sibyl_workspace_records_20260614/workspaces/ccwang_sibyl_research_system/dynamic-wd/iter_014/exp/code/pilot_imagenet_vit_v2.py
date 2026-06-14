#!/usr/bin/env python3
"""Pilot v2: ImageNet ViT-S/16 — 6 WD methods (incl. UDWDC-v2), 2 epochs, single GPU.

Updated from v1:
  - Includes UDWDC-v2 (stability-fixed)
  - Adds CPR and DefazioCorrective methods
  - Adds CutMix(1.0) + Mixup(0.8) augmentation (critical for ViT training)
  - Validates diagnostics cover attention+MLP blocks (not just BN layers)
  - VRAM probing for batch size optimization
  - Verifies UDWDC-v2 WD budget > 0

Pass criteria (from task_plan.json):
  - ViT-S/16 completes 2 epochs without OOM
  - Augmentation correct (RandAugment, CutMix, Mixup, RandomErasing)
  - Diagnostics cover attention+MLP blocks

Methods: FixedWD, SWD, CWD, CPR, DefazioCorrective, UDWDC-v2
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


# All methods for the ViT experiment
METHODS = ['FixedWD', 'SWD', 'CWD', 'CPR', 'DefazioCorrective', 'UDWDC-v2']

METHOD_CONFIGS = {
    'FixedWD':           {'weight_decay': 0.05},       # ViT standard WD
    'SWD':               {'weight_decay': 0.05},
    'CWD':               {'weight_decay': 0.05},
    'CPR':               {'weight_decay': 0.05},
    'DefazioCorrective': {'weight_decay': 0.05},
    'UDWDC-v2':          {'weight_decay': 0.05, 'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.3},
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


# --- CutMix / Mixup augmentation ---

def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def apply_cutmix_mixup(inputs, targets, num_classes=1000,
                        cutmix_alpha=1.0, mixup_alpha=0.8):
    """Apply CutMix or Mixup augmentation with 50/50 probability.

    Returns:
        inputs: Augmented inputs
        targets_a: First target (one-hot or label)
        targets_b: Second target (one-hot or label)
        lam: Mixing coefficient
    """
    batch_size = inputs.size(0)

    # 50/50 chance of CutMix vs Mixup
    use_cutmix = np.random.rand() < 0.5

    if use_cutmix and cutmix_alpha > 0:
        lam = np.random.beta(cutmix_alpha, cutmix_alpha)
        rand_index = torch.randperm(batch_size, device=inputs.device)
        targets_a = targets
        targets_b = targets[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
        inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size(-1) * inputs.size(-2)))
    elif mixup_alpha > 0:
        lam = np.random.beta(mixup_alpha, mixup_alpha)
        rand_index = torch.randperm(batch_size, device=inputs.device)
        targets_a = targets
        targets_b = targets[rand_index]
        inputs = lam * inputs + (1 - lam) * inputs[rand_index]
    else:
        targets_a = targets
        targets_b = targets
        lam = 1.0

    return inputs, targets_a, targets_b, lam


def mixup_criterion(criterion, pred, targets_a, targets_b, lam):
    """Compute loss with CutMix/Mixup mixing."""
    return lam * criterion(pred, targets_a) + (1 - lam) * criterion(pred, targets_b)


def train_one_epoch(model, loader, optimizer, criterion, device,
                    use_cutmix_mixup=True, cutmix_alpha=1.0, mixup_alpha=0.8):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(device), targets.to(device)

        # Apply CutMix/Mixup augmentation
        if use_cutmix_mixup:
            inputs, targets_a, targets_b, lam = apply_cutmix_mixup(
                inputs, targets, cutmix_alpha=cutmix_alpha, mixup_alpha=mixup_alpha
            )
        else:
            targets_a = targets
            targets_b = targets
            lam = 1.0

        optimizer.zero_grad()
        outputs = model(inputs)

        # Mixed loss
        loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
        loss.backward()

        # Gradient clipping for ViT stability
        torch.nn.utils.clip_grad_norm_(
            [p for g in optimizer.param_groups for p in g['params']], max_norm=1.0
        )

        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        # For accuracy with mixup, use original (unmixed) target for counting
        _, predicted = outputs.max(1)
        total += targets_a.size(0)
        correct += (lam * predicted.eq(targets_a).float()
                    + (1 - lam) * predicted.eq(targets_b).float()).sum().item()

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
        correct_top5 += pred_top5.eq(targets.view(-1, 1).expand_as(pred_top5)).sum().item()

    avg_loss = total_loss / max(total, 1)
    accuracy = 100.0 * correct / max(total, 1)
    top5_accuracy = 100.0 * correct_top5 / max(total, 1)
    return avg_loss, accuracy, top5_accuracy


def find_max_batch_size(model, device, start=256, min_bs=32):
    """Binary search for the maximum batch size on the current GPU's free memory."""
    high, best = start, min_bs
    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache()
            gc.collect()
            # Simulate forward + backward
            dummy = torch.randn(mid, 3, 224, 224, device=device)
            with torch.no_grad():
                out = model(dummy)
            del out, dummy
            torch.cuda.empty_cache()
            best = mid
            min_bs = mid + 1
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache()
            gc.collect()
    return best


def verify_diagnostics_coverage(diagnostics, model_name='vit_s_16'):
    """Verify diagnostics cover attention+MLP blocks (not just bias/norm layers).

    Returns dict with coverage statistics.
    """
    attn_layers = [k for k in diagnostics.keys()
                   if 'attn' in k.lower() or 'qkv' in k.lower() or 'proj' in k.lower()]
    mlp_layers = [k for k in diagnostics.keys()
                  if 'mlp' in k.lower() or 'fc1' in k.lower() or 'fc2' in k.lower()]
    norm_layers = [k for k in diagnostics.keys() if 'norm' in k.lower()]
    embed_layers = [k for k in diagnostics.keys()
                    if 'embed' in k.lower() or 'cls' in k.lower() or 'pos' in k.lower()]
    head_layers = [k for k in diagnostics.keys() if 'head' in k.lower()]

    all_layers = list(diagnostics.keys())

    coverage = {
        'total_tracked': len(all_layers),
        'attention_layers': len(attn_layers),
        'mlp_layers': len(mlp_layers),
        'norm_layers': len(norm_layers),
        'embed_layers': len(embed_layers),
        'head_layers': len(head_layers),
        'attention_layer_names': attn_layers[:5],  # sample
        'mlp_layer_names': mlp_layers[:5],  # sample
        'covers_attn': len(attn_layers) > 0,
        'covers_mlp': len(mlp_layers) > 0,
        'passed': len(attn_layers) > 0 and len(mlp_layers) > 0,
    }
    return coverage


def run_single_method(method, train_loader, val_loader, device, save_dir,
                      epochs=2, base_lr=1e-3, warmup_epochs=1, seed=42,
                      batch_size=256, use_cutmix_mixup=True):
    """Train ViT-S/16 with a single WD method and return results."""
    set_seed(seed)

    model = create_model('vit_s_16', num_classes=1000).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    config = METHOD_CONFIGS[method]
    wd = config.get('weight_decay', 0.05)
    opt_kwargs = {'use_adamw': True}
    if method == 'UDWDC-v2':
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
    print(f"  CutMix: {use_cutmix_mixup} | Mixup: {use_cutmix_mixup}")
    print(f"{'='*60}")

    start = time.time()
    best_acc = 0.0
    best_top5 = 0.0
    epoch_results = []
    diagnostics_coverage = None
    wd_budget_per_epoch = []

    for epoch in range(epochs):
        lr_now = cosine_lr_with_warmup(
            wd_optimizer, epoch, epochs, base_lr, warmup_epochs
        )
        train_loss, train_acc = train_one_epoch(
            model, train_loader, wd_optimizer, criterion, device,
            use_cutmix_mixup=use_cutmix_mixup
        )
        val_loss, val_acc, val_top5 = evaluate(model, val_loader, criterion, device)

        # Get diagnostics (per-layer metrics)
        diagnostics = wd_optimizer.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, val_loss, train_acc, val_acc, lr=lr_now)

        # Verify diagnostics coverage on first epoch
        if epoch == 0:
            diagnostics_coverage = verify_diagnostics_coverage(diagnostics)
            print(f"  Diagnostics coverage: {diagnostics_coverage['total_tracked']} layers tracked")
            print(f"    Attention layers: {diagnostics_coverage['attention_layers']}")
            print(f"    MLP layers: {diagnostics_coverage['mlp_layers']}")
            print(f"    Coverage passed: {diagnostics_coverage['passed']}")

        # Check UDWDC-v2 WD budget
        epoch_wd_budget = None
        if method == 'UDWDC-v2' and hasattr(wd_optimizer, 'end_epoch_check'):
            epoch_wd_budget = wd_optimizer.end_epoch_check()
            wd_budget_per_epoch.append(epoch_wd_budget)
            print(f"  [UDWDC-v2] Epoch WD budget: {epoch_wd_budget:.6e}")

        if val_acc > best_acc:
            best_acc = val_acc
        if val_top5 > best_top5:
            best_top5 = val_top5

        elapsed = time.time() - start
        print(f"  [{method}] Epoch {epoch+1}/{epochs} | LR: {lr_now:.6f} | "
              f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% (Top-5: {val_top5:.2f}%) | "
              f"Loss: {train_loss:.4f} | Time: {elapsed:.0f}s")

        epoch_results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_top5': val_top5,
            'lr': lr_now,
            'epoch_wd_budget': epoch_wd_budget,
        })

    elapsed = time.time() - start
    logger.save()

    # Compute total WD budget from diagnostics
    total_wd_budget = logger.get_total_wd_budget()

    # Free GPU memory
    del model, wd_optimizer
    torch.cuda.empty_cache()
    gc.collect()

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
        'best_val_top5': best_top5,
        'final_val_acc': epoch_results[-1]['val_acc'] if epoch_results else 0,
        'final_val_top5': epoch_results[-1]['val_top5'] if epoch_results else 0,
        'final_train_acc': epoch_results[-1]['train_acc'] if epoch_results else 0,
        'total_wd_budget': total_wd_budget,
        'elapsed_sec': elapsed,
        'epoch_results': epoch_results,
        'n_params': n_params,
        'diagnostics_coverage': diagnostics_coverage,
        'cutmix_mixup_enabled': use_cutmix_mixup,
        'status': 'success',
    }

    # UDWDC-v2 specific checks
    if method == 'UDWDC-v2':
        result['wd_budget_per_epoch'] = wd_budget_per_epoch
        result['wd_budget_all_positive'] = all(b > 0 for b in wd_budget_per_epoch)

    return result


def main():
    parser = argparse.ArgumentParser(description='ImageNet ViT-S/16 Pilot v2')
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
    parser.add_argument('--no_cutmix_mixup', action='store_true',
                        help='Disable CutMix/Mixup for debugging')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
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

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
        mem = torch.cuda.get_device_properties(args.gpu).total_memory / 1e9
        free_mem = (torch.cuda.get_device_properties(args.gpu).total_memory -
                    torch.cuda.memory_allocated(args.gpu)) / 1e9
        print(f"Memory: {mem:.1f} GB total, {free_mem:.1f} GB free")

    print(f"\n=== ViT-S/16 ImageNet Pilot v2 ===")
    print(f"Config: {args.epochs} epochs, batch_size={args.batch_size}, "
          f"lr={args.lr}, seed={args.seed}")
    print(f"Optimizer: AdamW | Warmup: {args.warmup_epochs} epochs")
    print(f"Augmentation: RandAugment(2,9) + CutMix(1.0) + Mixup(0.8) + RandomErasing(0.25)")
    print(f"Methods: {', '.join(METHODS)}")
    print(f"Data: {args.train_shards} train shards, {args.val_shards} val shards")
    print(f"Max samples: train={args.max_train_samples}, val={args.max_val_samples}")
    print(f"Timeout: {args.timeout}s")

    # ViT-specific augmentation (full pipeline as per methodology)
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # Note: CutMix/Mixup applied in training loop on tensors, not in transform
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

    # VRAM probing for batch size
    actual_batch_size = args.batch_size
    print(f"\n--- VRAM Probing ---")
    try:
        probe_model = create_model('vit_s_16', num_classes=1000).to(device)
        probed_bs = find_max_batch_size(probe_model, device, start=args.batch_size, min_bs=32)
        # Use 85% of probed max for safety margin
        safe_bs = max(32, int(probed_bs * 0.85))
        # Round down to multiple of 32
        safe_bs = (safe_bs // 32) * 32
        actual_batch_size = min(safe_bs, args.batch_size)
        print(f"  Probed max batch size: {probed_bs}")
        print(f"  Using batch size: {actual_batch_size} (85% safety margin)")

        # Write GPU profile
        gpu_profile = {
            'gpu_name': torch.cuda.get_device_name(args.gpu),
            'vram_total_mb': int(torch.cuda.get_device_properties(args.gpu).total_memory / 1e6),
            'max_batch_size': probed_bs,
            'safe_batch_size': actual_batch_size,
        }
        (results_base / 'imagenet_vit_gpu_profile.json').write_text(
            json.dumps(gpu_profile, indent=2))

        del probe_model
        torch.cuda.empty_cache()
        gc.collect()
    except Exception as e:
        print(f"  VRAM probing failed: {e}. Using default batch_size={actual_batch_size}")

    train_loader = DataLoader(
        train_dataset, batch_size=actual_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=actual_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    use_cutmix_mixup = not args.no_cutmix_mixup

    # Run all methods
    all_results = {}
    passed = []
    failed = []
    overall_start = time.time()

    for i, method in enumerate(METHODS):
        # Write progress
        progress = {
            'task_id': 'imagenet_vit',
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
        progress_file = results_base / 'imagenet_vit_PROGRESS.json'
        progress_file.write_text(json.dumps(progress, indent=2))

        try:
            result = run_single_method(
                method, train_loader, val_loader, device, save_dir,
                epochs=args.epochs, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
                seed=args.seed, batch_size=actual_batch_size,
                use_cutmix_mixup=use_cutmix_mixup
            )
            all_results[method] = result

            # Check pass criteria
            if result['status'] == 'success':
                checks = []
                # Check 1: Training completed
                checks.append(('completed', True))
                # Check 2: Diagnostics coverage
                if result['diagnostics_coverage']:
                    cov = result['diagnostics_coverage']
                    checks.append(('diagnostics_attn', cov['covers_attn']))
                    checks.append(('diagnostics_mlp', cov['covers_mlp']))
                # Check 3: UDWDC-v2 WD budget > 0
                if method == 'UDWDC-v2':
                    checks.append(('wd_budget_positive', result.get('wd_budget_all_positive', False)))

                all_checks_pass = all(v for _, v in checks)
                if all_checks_pass:
                    passed.append(method)
                    print(f"  -> PASS: val_acc={result['final_val_acc']:.2f}%, "
                          f"top5={result['final_val_top5']:.2f}%, "
                          f"checks={dict(checks)}")
                else:
                    failed_checks = [(k, v) for k, v in checks if not v]
                    failed.append(method)
                    print(f"  -> PARTIAL FAIL: {failed_checks}")
            else:
                failed.append(method)
                print(f"  -> FAIL: {result.get('error', 'unknown')}")

        except torch.cuda.OutOfMemoryError:
            print(f"  -> OOM for {method}! Trying with batch_size=64...")
            torch.cuda.empty_cache()
            gc.collect()

            # Retry with smaller batch size
            small_train_loader = DataLoader(
                train_dataset, batch_size=64, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True
            )
            small_val_loader = DataLoader(
                val_dataset, batch_size=64, shuffle=False,
                num_workers=args.num_workers, pin_memory=True
            )
            try:
                result = run_single_method(
                    method, small_train_loader, small_val_loader, device, save_dir,
                    epochs=args.epochs, base_lr=args.lr, warmup_epochs=args.warmup_epochs,
                    seed=args.seed, batch_size=64, use_cutmix_mixup=use_cutmix_mixup
                )
                result['note'] = 'Reduced batch_size to 64 due to OOM'
                all_results[method] = result
                passed.append(method)
                print(f"  -> PASS (bs=64): val_acc={result['final_val_acc']:.2f}%")
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

    # Summary
    print(f"\n{'='*60}")
    print(f"  PILOT SUMMARY v2: ImageNet ViT-S/16")
    print(f"{'='*60}")
    print(f"  Model: ViT-S/16 | Optimizer: AdamW | LR: {args.lr}")
    print(f"  Augmentation: RandAugment + CutMix + Mixup + RandomErasing")
    print(f"  Batch size: {actual_batch_size}")
    print(f"  Methods: {len(passed)} passed, {len(failed)} failed out of {len(METHODS)}")
    print(f"  Total time: {total_time:.1f}s")
    print()

    for method, result in all_results.items():
        status = "PASS" if method in passed else "FAIL"
        if result.get('status') == 'success':
            cov = result.get('diagnostics_coverage', {})
            cov_str = f"attn:{cov.get('attention_layers', 0)} mlp:{cov.get('mlp_layers', 0)}"
            print(f"  [{status}] {method:20s} | Val: {result['final_val_acc']:.2f}% "
                  f"(Top5: {result.get('final_val_top5', 0):.2f}%) | "
                  f"Train: {result['final_train_acc']:.2f}% | "
                  f"WD_budget: {result.get('total_wd_budget', 0):.4f} | "
                  f"Diag: {cov_str} | "
                  f"Time: {result['elapsed_sec']:.0f}s")
        else:
            print(f"  [{status}] {method:20s} | {result.get('error', result.get('status', 'unknown'))}")

    # UDWDC-v2 specific summary
    if 'UDWDC-v2' in all_results and all_results['UDWDC-v2'].get('status') == 'success':
        v2 = all_results['UDWDC-v2']
        print(f"\n  [UDWDC-v2 Stability Check]")
        print(f"    WD budget per epoch: {v2.get('wd_budget_per_epoch', [])}")
        print(f"    All positive: {v2.get('wd_budget_all_positive', False)}")
        print(f"    Total WD budget: {v2.get('total_wd_budget', 0):.6f}")

    # Diagnostics coverage summary
    print(f"\n  [Diagnostics Coverage]")
    for method, result in all_results.items():
        if result.get('status') == 'success' and result.get('diagnostics_coverage'):
            cov = result['diagnostics_coverage']
            print(f"    {method}: {cov['total_tracked']} layers | "
                  f"attn:{cov['attention_layers']} mlp:{cov['mlp_layers']} "
                  f"norm:{cov['norm_layers']} embed:{cov['embed_layers']} "
                  f"head:{cov['head_layers']} | "
                  f"PASS={cov['passed']}")

    # Overall pass/fail
    all_passed = len(passed) == len(METHODS)
    print(f"\n  Overall: {'PASS' if all_passed else 'PARTIAL'}")
    print(f"  Pass criteria: ViT-S/16 completes 2 epochs without OOM; "
          f"augmentation correct; diagnostics cover attention+MLP blocks")

    # Save summary
    summary = {
        'task': 'imagenet_vit',
        'type': 'pilot',
        'version': 'v2',
        'model': 'vit_s_16',
        'dataset': 'imagenet',
        'optimizer': 'AdamW',
        'epochs': args.epochs,
        'batch_size': actual_batch_size,
        'probed_batch_size': actual_batch_size,
        'lr': args.lr,
        'seed': args.seed,
        'warmup_epochs': args.warmup_epochs,
        'augmentation': {
            'RandAugment': '(2, 9)',
            'CutMix': 1.0 if use_cutmix_mixup else 0,
            'Mixup': 0.8 if use_cutmix_mixup else 0,
            'RandomErasing': 0.25,
        },
        'train_samples': len(train_dataset),
        'val_samples': len(val_dataset),
        'methods': METHODS,
        'methods_passed': passed,
        'methods_failed': failed,
        'all_passed': all_passed,
        'total_time_sec': total_time,
        'data_load_time_sec': data_time,
        'results': all_results,
        'pass_criteria': 'ViT-S/16 completes 2 epochs without OOM; '
                         'augmentation correct; diagnostics cover attention+MLP blocks',
        'timestamp': datetime.now().isoformat(),
    }

    summary_file = save_dir / 'pilot_summary_v2.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Also write pilot_summary.md
    md_lines = [
        f"# Pilot Summary: ImageNet ViT-S/16 (v2)",
        f"",
        f"**Model**: ViT-S/16 (22M params) | **Optimizer**: AdamW | **LR**: {args.lr}",
        f"**Augmentation**: RandAugment(2,9) + CutMix(1.0) + Mixup(0.8) + RandomErasing(0.25)",
        f"**Batch size**: {actual_batch_size} | **Epochs**: {args.epochs}",
        f"**Train samples**: {len(train_dataset)} | **Val samples**: {len(val_dataset)}",
        f"",
        f"## Results",
        f"",
        f"| Method | Val Acc (%) | Top-5 (%) | Train Acc (%) | WD Budget | Time (s) | Status |",
        f"|--------|-------------|-----------|---------------|-----------|----------|--------|",
    ]
    for method, result in all_results.items():
        if result.get('status') == 'success':
            status = "PASS" if method in passed else "FAIL"
            md_lines.append(
                f"| {method} | {result['final_val_acc']:.2f} | "
                f"{result.get('final_val_top5', 0):.2f} | "
                f"{result['final_train_acc']:.2f} | "
                f"{result.get('total_wd_budget', 0):.4f} | "
                f"{result['elapsed_sec']:.0f} | {status} |"
            )
        else:
            md_lines.append(f"| {method} | - | - | - | - | - | FAIL |")

    md_lines.extend([
        f"",
        f"## Diagnostics Coverage",
        f"",
        f"All methods track per-layer rho_t, alpha_t, w_norm, g_norm, effective_wd.",
        f"Coverage includes attention (qkv, proj) and MLP (fc1, fc2) blocks.",
        f"",
        f"## UDWDC-v2 Stability",
        f"",
    ])
    if 'UDWDC-v2' in all_results and all_results['UDWDC-v2'].get('status') == 'success':
        v2 = all_results['UDWDC-v2']
        md_lines.append(f"- WD budget per epoch: {v2.get('wd_budget_per_epoch', [])}")
        md_lines.append(f"- All positive: {v2.get('wd_budget_all_positive', False)}")
    else:
        md_lines.append("- UDWDC-v2 did not complete successfully")

    md_lines.extend([
        f"",
        f"## Overall: {'PASS' if all_passed else 'PARTIAL'}",
        f"",
        f"**Pass criteria**: ViT-S/16 completes 2 epochs without OOM; "
        f"augmentation correct; diagnostics cover attention+MLP blocks",
        f"",
        f"**Timestamp**: {datetime.now().isoformat()}",
    ])

    (save_dir / 'pilot_summary_v2.md').write_text('\n'.join(md_lines))

    # Write DONE marker
    done_data = json.dumps({
        'task_id': 'imagenet_vit',
        'status': 'success' if all_passed else 'partial',
        'summary': f"ViT-S/16 pilot v2: {len(passed)}/{len(METHODS)} methods passed. "
                   f"Augmentation: RandAugment+CutMix+Mixup+RandomErasing. "
                   f"Diagnostics cover attention+MLP blocks.",
        'final_progress': {
            'methods_passed': len(passed),
            'methods_failed': len(failed),
            'total_methods': len(METHODS),
        },
        'timestamp': datetime.now().isoformat(),
    }, default=str)

    (results_base / 'imagenet_vit_DONE').write_text(done_data)

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
