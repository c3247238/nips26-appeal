"""
Tier 1: AADWD 3 variants (Conservative, Aggressive, Square) - PILOT run.

Settings (PILOT mode):
- 20 epochs, seed=42, batch_size=128
- beta=0.999 (per tier0 diagnostic recommendation)
- lr=0.1, MultiStepLR milestones=[30, 60, 90] gamma=0.2 (every 30 epochs)
  (For 20 epoch pilot, no decay milestone reached, but scheduler is set correctly)
- c=5e-4 / lr_init = 5e-3 (so c * lr ~ 5e-4 when lr=0.1, matching WD scale)
  Actually: lambda_t = c * gamma_t * (1-ema_delta)
  With gamma_t=lr=0.1 and ema_delta~0.5 initially, lambda_t ~ c * 0.1 * 0.5
  To get lambda_t ~ 5e-4: c ~ 5e-4 / (0.1 * 0.5) = 0.01
  But c=1.0 is the default. Let's use c=5e-3 to keep lambda in [1e-6, 0.01] range.
  Actually: with c=1.0, lambda_t = 1.0 * 0.1 * 0.5 = 0.05 > lambda_max=0.01 -> clipped
  We need c * lr * (1-ema_delta) ~ 5e-4: c ~ 5e-4/(0.1*0.5) = 0.01
  Use c=0.01 to produce lambda_t ~ 5e-4 matching the best fixed WD scale.
  Alternatively, raise lambda_max to allow exploration. Use c=1.0, lambda_max=0.1 to
  allow meaningful variation. But task says lambda_max=0.01 so keep those.

  Per task spec: "使用 wd=5e-4 级别的 c 值" -> c should produce lambda_t around 5e-4.
  With lr=0.1, ema_delta~0.5: lambda_t = c * 0.1 * 0.5 = 5e-4 -> c = 0.01

Records: train_loss, test_acc, weight_norm, mean_lambda_t, mean_delta_hat_t, ema_delta
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Set GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

CODE_DIR = Path('/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/code')
sys.path.insert(0, str(CODE_DIR))

from models import create_model
from data import get_dataloaders
from optimizers import AADWDOptimizer


RESULTS_DIR = Path('/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/tier1_aadwd_variants')
DATA_DIR = Path('/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/data')

# PILOT config
EPOCHS = 20
SEED = 42
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
BETA = 0.999       # tier0 diagnostic recommendation
C = 0.01           # produces lambda_t ~ 5e-4 at lr=0.1, ema_delta~0.5
LAMBDA_MIN = 1e-6
LAMBDA_MAX = 0.01
LR_MILESTONES = [30, 60, 90]  # every 30 epochs, gamma=0.2
LR_GAMMA = 0.2

VARIANTS = ['conservative', 'aggressive', 'square']


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


def run_variant(variant):
    """Train ResNet20/CIFAR-10 with one AADWD variant."""
    print(f"\n{'='*60}")
    print(f"Running AADWD variant: {variant}")
    print(f"  beta={BETA}, c={C}, lambda_min={LAMBDA_MIN}, lambda_max={LAMBDA_MAX}")
    print(f"  epochs={EPOCHS}, lr={LR}, milestones={LR_MILESTONES}, gamma={LR_GAMMA}")
    print(f"{'='*60}")

    # Reproducibility
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Data
    train_loader, test_loader, _ = get_dataloaders(
        'cifar10', batch_size=BATCH_SIZE, data_dir=str(DATA_DIR))

    # Model
    model = create_model('resnet20', num_classes=10).to(device)

    # Optimizer: AADWD with specified variant
    optimizer = AADWDOptimizer(
        list(model.parameters()),
        lr=LR,
        momentum=MOMENTUM,
        c=C,
        beta=BETA,
        lambda_min=LAMBDA_MIN,
        lambda_max=LAMBDA_MAX,
        variant=variant
    )

    # LR scheduler: MultiStepLR, milestones every 30 epochs
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer.optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

    loss_fn = nn.CrossEntropyLoss()

    # Output directory
    out_dir = RESULTS_DIR / f'aadwd_{variant}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config = {
        'variant': variant,
        'method': f'aadwd_{variant}',
        'arch': 'resnet20',
        'dataset': 'cifar10',
        'mode': 'PILOT',
        'epochs': EPOCHS,
        'seed': SEED,
        'batch_size': BATCH_SIZE,
        'lr': LR,
        'momentum': MOMENTUM,
        'beta': BETA,
        'c': C,
        'lambda_min': LAMBDA_MIN,
        'lambda_max': LAMBDA_MAX,
        'lr_milestones': LR_MILESTONES,
        'lr_gamma': LR_GAMMA,
    }
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=2)

    epoch_log_path = out_dir / 'epoch_metrics.jsonl'
    step_log_path = out_dir / 'step_metrics.jsonl'

    # Clear existing logs
    for p in [epoch_log_path, step_log_path]:
        if p.exists():
            p.unlink()

    start_time = time.time()
    best_test_acc = 0.0
    all_epoch_records = []

    for epoch in range(EPOCHS):
        epoch_start = time.time()
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0
        step_lambda_sum = 0.0
        step_delta_hat_sum = 0.0
        step_ema_delta_sum = 0.0
        step_count_epoch = 0
        step_logs = []
        global_step = epoch * len(train_loader)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Accumulate per-step AADWD metrics
            metrics = optimizer.get_metrics()
            step_lambda_sum += metrics['lambda_t']
            step_delta_hat_sum += metrics['delta_hat_t']
            step_ema_delta_sum += metrics['ema_delta']
            step_count_epoch += 1

            # Log every 100 steps
            current_step = global_step + batch_idx
            if batch_idx % 100 == 0:
                step_log = {
                    'step': current_step,
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'loss': round(loss.item(), 6),
                    'lambda_t': round(metrics['lambda_t'], 8),
                    'delta_hat_t': round(metrics['delta_hat_t'], 6),
                    'ema_delta': round(metrics['ema_delta'], 6),
                    'weight_norm': round(metrics['weight_norm'], 4),
                    'lr': round(metrics['lr'], 8),
                }
                step_logs.append(step_log)

        # Step LR scheduler
        scheduler.step()

        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        gen_gap = train_acc - test_acc
        weight_norm = get_weight_norm(model)
        mean_lambda_t = step_lambda_sum / step_count_epoch if step_count_epoch > 0 else 0.0
        mean_delta_hat_t = step_delta_hat_sum / step_count_epoch if step_count_epoch > 0 else 0.0
        mean_ema_delta = step_ema_delta_sum / step_count_epoch if step_count_epoch > 0 else 0.0
        current_lr = optimizer.optimizer.param_groups[0]['lr']

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4),
            'test_loss': round(test_loss, 6),
            'gen_gap': round(gen_gap, 4),
            'weight_norm': round(weight_norm, 4),
            'mean_lambda_t': round(mean_lambda_t, 8),
            'mean_delta_hat_t': round(mean_delta_hat_t, 6),
            'ema_delta': round(mean_ema_delta, 6),
            'lr': round(current_lr, 8),
            'epoch_time_sec': round(epoch_time, 2),
        }
        all_epoch_records.append(epoch_record)

        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')
        with open(step_log_path, 'a') as f:
            for sl in step_logs:
                f.write(json.dumps(sl) + '\n')

        best_test_acc = max(best_test_acc, test_acc)

        if epoch % 5 == 0 or epoch == EPOCHS - 1:
            print(f"  Epoch {epoch:3d}/{EPOCHS}: "
                  f"loss={train_loss:.4f} train={train_acc:.2f}% "
                  f"test={test_acc:.2f}% gap={gen_gap:.2f}% "
                  f"wn={weight_norm:.2f} "
                  f"lambda={mean_lambda_t:.6f} "
                  f"delta_hat={mean_delta_hat_t:.4f} "
                  f"ema_d={mean_ema_delta:.4f} "
                  f"lr={current_lr:.6f}")

    total_time = time.time() - start_time

    # Check pass criteria: lambda_t not all clipped + test_acc > 85%
    all_lambdas = [r['mean_lambda_t'] for r in all_epoch_records]
    lambda_min_seen = min(all_lambdas)
    lambda_max_seen = max(all_lambdas)
    lambda_varied = (lambda_max_seen - lambda_min_seen) > 1e-7
    lambda_not_all_min = any(l > LAMBDA_MIN * 10 for l in all_lambdas)
    lambda_not_all_max = any(l < LAMBDA_MAX * 0.9 for l in all_lambdas)
    non_degenerate = lambda_varied and lambda_not_all_min and lambda_not_all_max
    pass_criteria = non_degenerate and best_test_acc > 85.0

    summary = {
        'variant': variant,
        'method': f'aadwd_{variant}',
        'mode': 'PILOT',
        'best_test_acc': round(best_test_acc, 4),
        'final_test_acc': round(all_epoch_records[-1]['test_acc'], 4),
        'final_train_acc': round(all_epoch_records[-1]['train_acc'], 4),
        'final_gen_gap': round(all_epoch_records[-1]['gen_gap'], 4),
        'final_weight_norm': round(all_epoch_records[-1]['weight_norm'], 4),
        'lambda_min_seen': round(lambda_min_seen, 8),
        'lambda_max_seen': round(lambda_max_seen, 8),
        'lambda_varied': lambda_varied,
        'non_degenerate_lambda': non_degenerate,
        'pass_criteria': pass_criteria,
        'pass_criteria_details': {
            'lambda_non_degenerate': non_degenerate,
            'test_acc_above_85': best_test_acc > 85.0,
        },
        'total_time_sec': round(total_time, 2),
        'config': config,
    }

    with open(out_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n  DONE variant={variant}: best_test_acc={best_test_acc:.2f}%, "
          f"lambda range=[{lambda_min_seen:.2e}, {lambda_max_seen:.2e}], "
          f"non_degenerate={non_degenerate}, PASS={pass_criteria}")

    return summary


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PILOT: tier1_aadwd_variants")
    print(f"Variants: {VARIANTS}")
    print(f"Config: epochs={EPOCHS}, seed={SEED}, beta={BETA}, c={C}")
    print(f"        lambda_min={LAMBDA_MIN}, lambda_max={LAMBDA_MAX}")
    print(f"        lr={LR}, milestones={LR_MILESTONES}, gamma={LR_GAMMA}")

    all_summaries = []
    for variant in VARIANTS:
        summary = run_variant(variant)
        all_summaries.append(summary)

    # Aggregate
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS:")
    print(f"{'Variant':<20} {'BestAcc':>8} {'FinalAcc':>9} {'LambdaMin':>12} {'LambdaMax':>12} {'NonDegen':>9} {'PASS':>6}")
    print('-' * 80)
    all_pass = True
    for s in all_summaries:
        print(f"  {s['variant']:<18} {s['best_test_acc']:>8.2f} {s['final_test_acc']:>9.2f} "
              f"{s['lambda_min_seen']:>12.2e} {s['lambda_max_seen']:>12.2e} "
              f"{str(s['non_degenerate_lambda']):>9} {str(s['pass_criteria']):>6}")
        if not s['pass_criteria']:
            all_pass = False

    overall = {
        'task': 'tier1_aadwd_variants',
        'mode': 'PILOT',
        'all_pass': all_pass,
        'variants': all_summaries,
        'pilot_config': {
            'epochs': EPOCHS,
            'seed': SEED,
            'beta': BETA,
            'c': C,
            'lambda_min': LAMBDA_MIN,
            'lambda_max': LAMBDA_MAX,
            'lr': LR,
            'lr_milestones': LR_MILESTONES,
            'lr_gamma': LR_GAMMA,
        }
    }

    with open(RESULTS_DIR / 'overall_summary.json', 'w') as f:
        json.dump(overall, f, indent=2)

    print(f"\nOverall PASS: {all_pass}")
    print(f"Results saved to: {RESULTS_DIR}")
    return all_pass


if __name__ == '__main__':
    success = main()

    # Write DONE marker
    done_path = Path('/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/tier1_aadwd_variants_DONE')
    done_path.write_text(json.dumps({
        'status': 'completed',
        'success': success,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
    }))
    print(f"\nDONE marker written: {done_path}")
