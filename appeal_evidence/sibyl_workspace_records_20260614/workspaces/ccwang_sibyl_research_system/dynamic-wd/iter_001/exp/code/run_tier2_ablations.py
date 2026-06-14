"""
Tier 2: Critical ablations (H5 — alignment vs time-variation) - PILOT run.

Trains ResNet20/CIFAR-10 for 20 epochs (PILOT mode) with:
1. random_dynamic_wd: Random dynamic WD (control, replaces alignment with U[0,1])
2. norm_matched_wd: Normalized WD matching AADWD aggressive weight norm trajectory

Reference settings from best AADWD variant (aggressive, beta=0.999):
  lr=0.1, momentum=0.9, c=0.01, beta=0.999, lambda_min=1e-6, lambda_max=0.01
  LR schedule: MultiStepLR milestones=[30,60,90], gamma=0.2

PILOT criteria: All methods produce valid training curves (no NaN, test acc > 80% at 20 epochs)
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
from optimizers import RandomDynamicWDOptimizer, NormMatchedWDOptimizer


def get_weight_norm(model):
    total = 0.0
    for p in model.parameters():
        total += p.data.norm().item() ** 2
    return math.sqrt(total)


def get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm().item() ** 2
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


def train_one_run(config, output_dir, target_norms=None):
    """Train a single method run.

    Args:
        config: dict with training config
        output_dir: Path to save results
        target_norms: list of weight norms per epoch for norm_matched_wd
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data
    data_dir = str(Path(__file__).parent.parent / "results" / "data")
    train_loader, test_loader, _ = get_dataloaders(
        config['dataset'], batch_size=config['batch_size'],
        data_dir=data_dir)

    # Model
    num_classes = 10 if config['dataset'] == 'cifar10' else 100
    model = create_model(config['arch'], num_classes=num_classes)
    model = model.to(device)

    # Optimizer
    method = config['method']
    if method == 'random_dynamic_wd':
        optimizer = RandomDynamicWDOptimizer(
            list(model.parameters()),
            lr=config['lr'],
            momentum=config['momentum'],
            c=config['c'],
            lambda_min=config['lambda_min'],
            lambda_max=config['lambda_max']
        )
    elif method == 'norm_matched_wd':
        optimizer = NormMatchedWDOptimizer(
            list(model.parameters()),
            lr=config['lr'],
            momentum=config['momentum'],
            base_wd=config['weight_decay']
        )
        if target_norms is not None:
            optimizer.set_target_norms(target_norms)
    else:
        raise ValueError(f"Unknown method: {method}")

    # LR scheduler: MultiStepLR matching AADWD baseline
    scheduler = MultiStepLR(
        optimizer.optimizer,
        milestones=config['lr_milestones'],
        gamma=config['lr_gamma']
    )

    loss_fn = nn.CrossEntropyLoss()

    # Logging files
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    step_log_path = output_dir / "step_metrics.jsonl"

    # Save config (convert non-serializable items)
    config_save = {k: v for k, v in config.items()}
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_save, f, indent=2)

    start_time = time.time()
    all_epoch_records = []
    best_test_acc = 0.0
    any_nan = False

    print(f"\n{'='*60}")
    print(f"Training: {config['arch']} on {config['dataset']} with {method}")
    print(f"Epochs: {config['epochs']}, LR: {config['lr']}, "
          f"c: {config.get('c', 'N/A')}, beta: {config.get('beta', 'N/A')}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Set epoch for norm-matched WD
        if hasattr(optimizer, 'set_epoch'):
            optimizer.set_epoch(epoch)

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        step_logs = []
        global_step = epoch * len(train_loader)
        lambda_accumulator = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

            # NaN check
            if torch.isnan(loss):
                any_nan = True
                print(f"  WARNING: NaN loss at epoch {epoch}, batch {batch_idx}!")
                break

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Track lambda
            if hasattr(optimizer, 'get_metrics'):
                metrics = optimizer.get_metrics()
                lambda_accumulator.append(metrics.get('lambda_t', 0.0))

            # Step logging every 100 batches
            current_step = global_step + batch_idx
            if batch_idx % 100 == 0:
                step_metrics = {
                    'step': current_step,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': round(loss.item(), 6),
                }
                if hasattr(optimizer, 'get_metrics'):
                    m = optimizer.get_metrics()
                    step_metrics['lambda_t'] = round(m.get('lambda_t', 0.0), 8)
                    step_metrics['weight_norm'] = round(m.get('weight_norm', 0.0), 4)
                    step_metrics['lr'] = round(m.get('lr', 0.0), 8)
                step_metrics['grad_norm'] = round(get_grad_norm(model), 4)
                step_logs.append(step_metrics)

        if any_nan:
            break

        # Update LR scheduler
        scheduler.step()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start

        gen_gap = train_acc - test_acc
        mean_lambda = (sum(lambda_accumulator) / len(lambda_accumulator)
                       if lambda_accumulator else 0.0)
        wn = get_weight_norm(model)

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4),
            'test_loss': round(test_loss, 6),
            'gen_gap': round(gen_gap, 4),
            'weight_norm': round(wn, 4),
            'mean_lambda_t': round(mean_lambda, 8),
            'mean_delta_hat_t': 0.0,  # Not applicable for ablations
            'ema_delta': 0.0,
            'lr': round(optimizer.optimizer.param_groups[0]['lr'], 8),
            'epoch_time_sec': round(epoch_time, 2),
        }
        all_epoch_records.append(epoch_record)
        best_test_acc = max(best_test_acc, test_acc)

        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')
        with open(step_log_path, 'a') as f:
            for sl in step_logs:
                f.write(json.dumps(sl) + '\n')

        if epoch % 5 == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                  f"test_acc={test_acc:.2f}% gen_gap={gen_gap:.2f}% "
                  f"wn={wn:.2f} lambda={mean_lambda:.6f}")

    total_time = time.time() - start_time

    # Pass criteria check
    final_test_acc = all_epoch_records[-1]['test_acc'] if all_epoch_records else 0.0
    pass_criteria = (not any_nan) and (final_test_acc > 80.0)

    summary = {
        'method': method,
        'mode': 'PILOT',
        'best_test_acc': round(best_test_acc, 4),
        'final_test_acc': round(final_test_acc, 4),
        'final_train_acc': round(all_epoch_records[-1]['train_acc'], 4) if all_epoch_records else 0.0,
        'final_gen_gap': round(all_epoch_records[-1]['gen_gap'], 4) if all_epoch_records else 0.0,
        'final_weight_norm': round(all_epoch_records[-1]['weight_norm'], 4) if all_epoch_records else 0.0,
        'any_nan': any_nan,
        'pass_criteria': pass_criteria,
        'pass_criteria_details': {
            'no_nan': not any_nan,
            'test_acc_above_80': final_test_acc > 80.0,
        },
        'total_time_sec': round(total_time, 2),
        'config': config,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Done! Best test acc: {best_test_acc:.2f}%, "
          f"Pass: {pass_criteria}, Time: {total_time:.1f}s")

    return summary, all_epoch_records


def main():
    # PILOT config: 20 epochs, seed=42, batch_size=128
    # Mirror the best AADWD (aggressive, beta=0.999) settings
    base_config = {
        'arch': 'resnet20',
        'dataset': 'cifar10',
        'mode': 'PILOT',
        'epochs': 20,
        'seed': 42,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        # AADWD aggressive baseline params
        'c': 0.01,
        'beta': 0.999,
        'lambda_min': 1e-6,
        'lambda_max': 0.01,
        'weight_decay': 5e-4,  # base WD for norm_matched_wd
        'lr_milestones': [30, 60, 90],
        'lr_gamma': 0.2,
    }

    output_base = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd"
                       "/current/exp/results/tier2_ablations")
    output_base.mkdir(parents=True, exist_ok=True)

    # Weight norms from aggressive AADWD PILOT run (20 epochs)
    aadwd_weight_norms = [
        30.9038, 32.2413, 34.5366, 37.3348, 40.0706,
        42.7127, 45.2019, 47.7093, 50.0677, 52.2993,
        54.461, 56.523, 58.5156, 60.3841, 62.1878,
        63.9433, 65.6402, 67.2664, 68.8543, 70.3869
    ]

    all_summaries = {}

    # --- Run 1: random_dynamic_wd ---
    print("\n" + "="*70)
    print("EXPERIMENT 1: random_dynamic_wd")
    print("="*70)
    config_rdwd = {**base_config, 'method': 'random_dynamic_wd'}
    summary_rdwd, _ = train_one_run(
        config_rdwd,
        output_base / "random_dynamic_wd"
    )
    all_summaries['random_dynamic_wd'] = summary_rdwd

    # --- Run 2: norm_matched_wd ---
    print("\n" + "="*70)
    print("EXPERIMENT 2: norm_matched_wd")
    print("="*70)
    config_nmwd = {**base_config, 'method': 'norm_matched_wd'}
    summary_nmwd, _ = train_one_run(
        config_nmwd,
        output_base / "norm_matched_wd",
        target_norms=aadwd_weight_norms
    )
    all_summaries['norm_matched_wd'] = summary_nmwd

    # --- Aggregate summary ---
    print("\n" + "="*70)
    print("TIER 2 ABLATIONS PILOT SUMMARY")
    print("="*70)

    all_passed = True
    for method, s in all_summaries.items():
        passed = s['pass_criteria']
        all_passed = all_passed and passed
        print(f"  {method:25s}: test_acc={s['final_test_acc']:.2f}%  "
              f"pass={passed}  "
              f"no_nan={s['pass_criteria_details']['no_nan']}")

    agg_summary = {
        'task': 'tier2_ablations',
        'mode': 'PILOT',
        'methods': list(all_summaries.keys()),
        'all_pass': all_passed,
        'results': all_summaries,
        'pass_criteria': (
            "All ablation methods produce valid training curves "
            "(no NaN, test acc > 80% at 20 epochs)"
        ),
    }

    with open(output_base / "aggregate_summary.json", 'w') as f:
        json.dump(agg_summary, f, indent=2)

    print(f"\nOverall PASS: {all_passed}")
    print(f"Results saved to: {output_base}")

    return agg_summary


if __name__ == '__main__':
    main()
