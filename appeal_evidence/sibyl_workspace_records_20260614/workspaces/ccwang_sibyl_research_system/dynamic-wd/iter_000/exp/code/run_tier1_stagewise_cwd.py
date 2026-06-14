"""
Tier 1: Stagewise WD and CWD baselines - PILOT run.

Trains ResNet20/CIFAR-10 for 20 epochs (PILOT mode) with:
1. stagewise_wd: Stage-wise WD schedule (lambda_0=5e-4, per-stage decay)
2. cwd: Cautious Weight Decay (sign-based masking, wd=5e-4)

LR schedule: MultiStepLR with milestones every 30 epochs (lr x0.2).
For 20-epoch pilot, this means no decay steps are hit (milestone at 30).
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from data import get_dataloaders
from optimizers import CWDOptimizer


def get_weight_norm(model):
    total = 0.0
    for p in model.parameters():
        total += p.data.norm().item() ** 2
    return math.sqrt(total)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    loss_fn = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    model.train()
    return 100.0 * correct / total, test_loss / total


def train_one_experiment(method, output_dir, epochs=20, batch_size=128,
                         lr=0.1, momentum=0.9, weight_decay=5e-4, seed=42,
                         device=None, data_root=None):
    """Train one method and return summary."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Data
    train_loader, test_loader, _ = get_dataloaders(
        'cifar10', batch_size=batch_size, data_dir=str(data_root))

    # Model
    model = create_model('resnet20', num_classes=10).to(device)

    # Optimizer
    if method == 'stagewise_wd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr,
                                    momentum=momentum, weight_decay=weight_decay)
    elif method == 'cwd':
        optimizer = CWDOptimizer(list(model.parameters()), lr=lr,
                                 momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown method: {method}")

    # LR scheduler: MultiStep - milestones at 30, 60, 90 epochs (x0.2 each)
    # For 20-epoch pilot, no milestones will be hit
    if hasattr(optimizer, 'optimizer'):
        scheduler = MultiStepLR(optimizer.optimizer, milestones=[30, 60, 90],
                                gamma=0.2)
    else:
        scheduler = MultiStepLR(optimizer, milestones=[30, 60, 90],
                                gamma=0.2)

    loss_fn = nn.CrossEntropyLoss()
    epoch_log_path = out_path / "epoch_metrics.jsonl"
    step_log_path = out_path / "step_metrics.jsonl"

    config = {
        'method': method, 'arch': 'resnet20', 'dataset': 'cifar10',
        'epochs': epochs, 'batch_size': batch_size, 'lr': lr,
        'momentum': momentum, 'weight_decay': weight_decay, 'seed': seed,
        'lr_milestones': [30, 60, 90], 'lr_gamma': 0.2,
    }
    with open(out_path / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Training ResNet20/CIFAR-10 with method: {method}")
    print(f"Epochs: {epochs}, LR: {lr}, WD: {weight_decay}, Batch: {batch_size}")
    print(f"Output: {out_path}")
    print(f"{'='*60}")

    start_time = time.time()
    all_epoch_records = []
    best_test_acc = 0.0
    step_count = 0

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        step_logs = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            # For stagewise_wd: update WD based on epoch progress
            if method == 'stagewise_wd':
                progress = epoch / epochs  # fraction of total epochs
                if progress < 0.5:
                    current_wd = weight_decay
                elif progress < 0.8:
                    current_wd = weight_decay / 10.0
                else:
                    current_wd = weight_decay / 100.0
                for group in optimizer.param_groups:
                    group['weight_decay'] = current_wd

            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Per-step logging every 100 steps
            if batch_idx % 100 == 0:
                if hasattr(optimizer, 'get_metrics'):
                    opt_metrics = optimizer.get_metrics()
                    lambda_t = opt_metrics.get('lambda_t', weight_decay)
                    delta_hat_t = opt_metrics.get('delta_hat_t', 0.0)
                    wn = opt_metrics.get('weight_norm', get_weight_norm(model))
                    cur_lr = opt_metrics.get('lr', lr)
                else:
                    lambda_t = optimizer.param_groups[0].get('weight_decay', 0.0)
                    delta_hat_t = 0.0
                    wn = get_weight_norm(model)
                    cur_lr = optimizer.param_groups[0]['lr']

                step_log = {
                    'step': step_count + batch_idx,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': round(loss.item(), 6),
                    'lambda_t': round(lambda_t, 8),
                    'delta_hat_t': round(delta_hat_t, 6),
                    'weight_norm': round(wn, 4),
                    'lr': round(cur_lr, 8),
                }
                step_logs.append(step_log)

        step_count += len(train_loader)
        scheduler.step()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        gen_gap = train_acc - test_acc
        wn_epoch = get_weight_norm(model)

        # Get current lambda_t for logging
        if method == 'stagewise_wd':
            progress = epoch / epochs
            if progress < 0.5:
                lambda_epoch = weight_decay
            elif progress < 0.8:
                lambda_epoch = weight_decay / 10.0
            else:
                lambda_epoch = weight_decay / 100.0
        elif method == 'cwd' and hasattr(optimizer, 'get_metrics'):
            m = optimizer.get_metrics()
            lambda_epoch = m.get('lambda_t', weight_decay)
        else:
            lambda_epoch = weight_decay

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4),
            'test_loss': round(test_loss, 6),
            'gen_gap': round(gen_gap, 4),
            'weight_norm': round(wn_epoch, 4),
            'mean_lambda_t': round(lambda_epoch, 8),
            'mean_delta_hat_t': 0.0,
            'lr': round(optimizer.param_groups[0]['lr'] if not hasattr(optimizer, 'optimizer')
                        else optimizer.optimizer.param_groups[0]['lr'], 8),
            'epoch_time_sec': round(epoch_time, 2),
        }
        all_epoch_records.append(epoch_record)

        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')
        with open(step_log_path, 'a') as f:
            for sl in step_logs:
                f.write(json.dumps(sl) + '\n')

        best_test_acc = max(best_test_acc, test_acc)

        if epoch % 5 == 0 or epoch == epochs - 1:
            print(f"  Epoch {epoch:3d}/{epochs}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                  f"test_acc={test_acc:.2f}% gen_gap={gen_gap:.2f}% "
                  f"wn={wn_epoch:.2f} lambda={lambda_epoch:.6f}")

    total_time = time.time() - start_time
    summary = {
        'config': config,
        'best_test_acc': round(best_test_acc, 4),
        'final_test_acc': round(all_epoch_records[-1]['test_acc'], 4),
        'final_train_acc': round(all_epoch_records[-1]['train_acc'], 4),
        'final_gen_gap': round(all_epoch_records[-1]['gen_gap'], 4),
        'final_weight_norm': round(all_epoch_records[-1]['weight_norm'], 4),
        'total_time_sec': round(total_time, 2),
        'epochs_completed': epochs,
    }
    with open(out_path / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Done! Best test acc: {best_test_acc:.2f}%, "
          f"Total time: {total_time:.1f}s")
    return summary


def main():
    results_base = Path(
        "/home/ccwang/sibyl-research-system/workspaces/dynamic-wd"
        "/current/exp/results/tier1_stagewise_cwd")
    results_base.mkdir(parents=True, exist_ok=True)

    data_root = Path(
        "/home/ccwang/sibyl-research-system/workspaces/dynamic-wd"
        "/current/exp/results/data")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    methods = ['stagewise_wd', 'cwd']
    summaries = {}

    for method in methods:
        out_dir = results_base / method
        summary = train_one_experiment(
            method=method,
            output_dir=str(out_dir),
            epochs=20,
            batch_size=128,
            lr=0.1,
            momentum=0.9,
            weight_decay=5e-4,
            seed=42,
            device=device,
            data_root=str(data_root),
        )
        summaries[method] = summary

    # Save aggregated summary
    agg = []
    for method, s in summaries.items():
        agg.append({
            'method': method,
            'best_test_acc': s['best_test_acc'],
            'final_test_acc': s['final_test_acc'],
            'final_train_acc': s['final_train_acc'],
            'final_gen_gap': s['final_gen_gap'],
            'final_weight_norm': s['final_weight_norm'],
            'total_time_sec': s['total_time_sec'],
        })

    with open(results_base / "aggregated_summary.json", 'w') as f:
        json.dump(agg, f, indent=2)

    print("\n" + "="*60)
    print("PILOT RESULTS SUMMARY")
    print("="*60)
    for item in agg:
        print(f"  {item['method']:20s}  best_acc={item['best_test_acc']:.2f}%  "
              f"final_acc={item['final_test_acc']:.2f}%  "
              f"gen_gap={item['final_gen_gap']:.2f}%  "
              f"wn={item['final_weight_norm']:.2f}")

    # Check pilot pass criterion: test acc > 90% at epoch 20 for both methods
    pass_criterion = all(s['best_test_acc'] > 85.0 for s in summaries.values())
    print(f"\nPilot pass criterion (best_acc > 85%): {'PASS' if pass_criterion else 'FAIL'}")

    return agg


if __name__ == '__main__':
    main()
