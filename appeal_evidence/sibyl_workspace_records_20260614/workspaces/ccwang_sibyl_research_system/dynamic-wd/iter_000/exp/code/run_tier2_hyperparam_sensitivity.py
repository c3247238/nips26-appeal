"""
Tier 2: Hyperparameter Sensitivity Analysis for AADWD-aggressive.

Tests sensitivity to:
1. c (base coefficient): fixed beta=0.999, c in [0.001, 0.005, 0.01, 0.05, 0.1]
2. beta (EMA decay): fixed c=0.01, beta in [0.9, 0.99, 0.999, 0.9999]

Architecture: ResNet20 / CIFAR-10
Mode: PILOT (20 epochs, seed=42, batch_size=128)

LR schedule: MultiStepLR, milestones=[30,60,90], gamma=0.2
(milestones won't trigger in 20 epochs, LR stays at initial value for PILOT)
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
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from models import create_model
from data import get_dataloaders
from optimizers import AADWDOptimizer


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


def train_one_run(config, output_dir):
    """Train a single AADWD-aggressive run.

    Args:
        config: dict with training hyperparameters
        output_dir: Path to save results
    Returns:
        (summary dict, list of epoch records)
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
    data_dir = str(CODE_DIR.parent / "results" / "data")
    train_loader, test_loader, _ = get_dataloaders(
        config['dataset'], batch_size=config['batch_size'],
        data_dir=data_dir)

    # Model
    num_classes = 10 if config['dataset'] == 'cifar10' else 100
    model = create_model(config['arch'], num_classes=num_classes)
    model = model.to(device)

    # Optimizer: AADWD-aggressive
    optimizer = AADWDOptimizer(
        list(model.parameters()),
        lr=config['lr'],
        momentum=config['momentum'],
        c=config['c'],
        beta=config['beta'],
        lambda_min=config['lambda_min'],
        lambda_max=config['lambda_max'],
        variant='aggressive'
    )

    # LR scheduler: MultiStepLR
    scheduler = MultiStepLR(
        optimizer.optimizer,
        milestones=config['lr_milestones'],
        gamma=config['lr_gamma']
    )

    loss_fn = nn.CrossEntropyLoss()

    # Logging
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    step_log_path = output_dir / "step_metrics.jsonl"

    config_save = {k: v for k, v in config.items()}
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_save, f, indent=2)

    start_time = time.time()
    all_epoch_records = []
    best_test_acc = 0.0
    any_nan = False

    print(f"\n{'='*65}")
    print(f"AADWD-aggressive | c={config['c']}, beta={config['beta']}")
    print(f"Epochs: {config['epochs']}, LR: {config['lr']}")
    print(f"Device: {device} | Output: {output_dir.name}")
    print(f"{'='*65}")

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        step_logs = []
        global_step = epoch * len(train_loader)
        lambda_accum = []
        delta_hat_accum = []
        ema_delta_accum = []

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)

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

            # Track per-step metrics
            m = optimizer.get_metrics()
            lambda_accum.append(m['lambda_t'])
            delta_hat_accum.append(m['delta_hat_t'])
            ema_delta_accum.append(m['ema_delta'])

            # Step logging every 100 batches
            current_step = global_step + batch_idx
            if batch_idx % 100 == 0:
                step_metrics = {
                    'step': current_step,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': round(loss.item(), 6),
                    'lambda_t': round(m['lambda_t'], 8),
                    'delta_hat_t': round(m['delta_hat_t'], 6),
                    'ema_delta': round(m['ema_delta'], 6),
                    'weight_norm': round(m['weight_norm'], 4),
                    'lr': round(m['lr'], 8),
                    'grad_norm': round(get_grad_norm(model), 4),
                }
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
        mean_lambda = sum(lambda_accum) / len(lambda_accum) if lambda_accum else 0.0
        mean_delta_hat = sum(delta_hat_accum) / len(delta_hat_accum) if delta_hat_accum else 0.0
        mean_ema_delta = sum(ema_delta_accum) / len(ema_delta_accum) if ema_delta_accum else 0.0
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
            'mean_delta_hat_t': round(mean_delta_hat, 6),
            'mean_ema_delta': round(mean_ema_delta, 6),
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
            print(f"  Ep {epoch:3d}/{config['epochs']}: "
                  f"tr={train_acc:.1f}% ts={test_acc:.2f}% "
                  f"wn={wn:.2f} lam={mean_lambda:.6f} "
                  f"ema_d={mean_ema_delta:.3f}")

    total_time = time.time() - start_time
    final_test_acc = all_epoch_records[-1]['test_acc'] if all_epoch_records else 0.0

    summary = {
        'method': 'aadwd_aggressive',
        'c': config['c'],
        'beta': config['beta'],
        'mode': 'PILOT',
        'best_test_acc': round(best_test_acc, 4),
        'final_test_acc': round(final_test_acc, 4),
        'final_train_acc': round(all_epoch_records[-1]['train_acc'], 4) if all_epoch_records else 0.0,
        'final_gen_gap': round(all_epoch_records[-1]['gen_gap'], 4) if all_epoch_records else 0.0,
        'final_weight_norm': round(all_epoch_records[-1]['weight_norm'], 4) if all_epoch_records else 0.0,
        'final_mean_lambda_t': round(all_epoch_records[-1]['mean_lambda_t'], 8) if all_epoch_records else 0.0,
        'any_nan': any_nan,
        'total_time_sec': round(total_time, 2),
        'config': config,
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"  Done! best_acc={best_test_acc:.2f}%, time={total_time:.1f}s")

    return summary, all_epoch_records


def main():
    # Base config
    base_config = {
        'arch': 'resnet20',
        'dataset': 'cifar10',
        'mode': 'PILOT',
        'epochs': 20,
        'seed': 42,
        'batch_size': 128,
        'lr': 0.1,
        'momentum': 0.9,
        'lambda_min': 1e-6,
        'lambda_max': 0.01,
        'lr_milestones': [30, 60, 90],
        'lr_gamma': 0.2,
    }

    output_base = Path(
        "/home/ccwang/sibyl-research-system/workspaces/dynamic-wd"
        "/current/exp/results/tier2_hyperparam_sensitivity"
    )
    output_base.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    # ===== Sweep 1: vary c, fixed beta=0.999 =====
    print("\n" + "#"*70)
    print("SWEEP 1: Vary c | fixed beta=0.999")
    print("#"*70)

    c_values = [0.001, 0.005, 0.01, 0.05, 0.1]
    fixed_beta = 0.999

    for c_val in c_values:
        run_name = f"aadwd_agg_c{c_val}_beta{fixed_beta}"
        config = {
            **base_config,
            'c': c_val,
            'beta': fixed_beta,
            'run_name': run_name,
        }
        print(f"\nRunning: c={c_val}, beta={fixed_beta}")
        summary, _ = train_one_run(config, output_base / run_name)
        all_summaries[run_name] = summary

    # ===== Sweep 2: vary beta, fixed c=0.01 =====
    print("\n" + "#"*70)
    print("SWEEP 2: Vary beta | fixed c=0.01")
    print("#"*70)

    beta_values = [0.9, 0.99, 0.999, 0.9999]
    fixed_c = 0.01

    for beta_val in beta_values:
        run_name = f"aadwd_agg_c{fixed_c}_beta{beta_val}"
        # Skip if already done (c=0.01, beta=0.999 was in sweep 1)
        if run_name in all_summaries:
            print(f"\nSkipping (already done): {run_name}")
            continue
        config = {
            **base_config,
            'c': fixed_c,
            'beta': beta_val,
            'run_name': run_name,
        }
        print(f"\nRunning: c={fixed_c}, beta={beta_val}")
        summary, _ = train_one_run(config, output_base / run_name)
        all_summaries[run_name] = summary

    # ===== Aggregate Summary =====
    print("\n" + "="*70)
    print("HYPERPARAM SENSITIVITY PILOT SUMMARY")
    print("="*70)

    # Print sweep 1 table
    print("\n-- Sweep 1: Vary c (beta=0.999) --")
    print(f"  {'c':>8}  {'test_acc':>10}  {'weight_norm':>12}  {'mean_lambda_t':>14}")
    for c_val in c_values:
        run_name = f"aadwd_agg_c{c_val}_beta{fixed_beta}"
        s = all_summaries[run_name]
        print(f"  {c_val:>8}  {s['final_test_acc']:>10.4f}  "
              f"{s['final_weight_norm']:>12.4f}  {s['final_mean_lambda_t']:>14.8f}")

    # Print sweep 2 table
    print("\n-- Sweep 2: Vary beta (c=0.01) --")
    print(f"  {'beta':>8}  {'test_acc':>10}  {'weight_norm':>12}  {'mean_lambda_t':>14}")
    for beta_val in beta_values:
        run_name = f"aadwd_agg_c{fixed_c}_beta{beta_val}"
        s = all_summaries[run_name]
        print(f"  {beta_val:>8}  {s['final_test_acc']:>10.4f}  "
              f"{s['final_weight_norm']:>12.4f}  {s['final_mean_lambda_t']:>14.8f}")

    # Save aggregate summary
    agg_summary = {
        'task': 'tier2_hyperparam_sensitivity',
        'mode': 'PILOT',
        'sweep1_c_values': c_values,
        'sweep1_fixed_beta': fixed_beta,
        'sweep2_beta_values': beta_values,
        'sweep2_fixed_c': fixed_c,
        'results': all_summaries,
    }
    with open(output_base / "aggregate_summary.json", 'w') as f:
        json.dump(agg_summary, f, indent=2)

    print(f"\nAll results saved to: {output_base}")
    return agg_summary


if __name__ == '__main__':
    main()
