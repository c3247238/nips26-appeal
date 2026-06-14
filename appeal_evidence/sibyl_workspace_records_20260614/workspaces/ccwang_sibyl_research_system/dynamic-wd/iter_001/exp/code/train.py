"""
Configurable training script for AADWD experiments.

Supports all methods: no_wd, fixed_wd, stagewise_wd, cwd,
aadwd_conservative, aadwd_aggressive, aadwd_square,
random_dynamic_wd, norm_matched_wd, equiv_cumulative_wd.

Logs epoch-level and step-level metrics to JSONL files.
"""

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn

# Add code directory to path
sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from data import get_dataloaders
from optimizers import create_optimizer, AADWDOptimizer, CWDOptimizer


def compute_alignment(model, loss_fn, inputs, targets):
    """Compute gradient-parameter alignment delta_hat for a given batch."""
    model.zero_grad()
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    loss.backward()

    dot_gw = 0.0
    norm_g_sq = 0.0
    norm_w_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.data
        w = p.data
        dot_gw += torch.dot(g.flatten(), w.flatten()).item()
        norm_g_sq += g.norm().item() ** 2
        norm_w_sq += w.norm().item() ** 2

    norm_g = math.sqrt(norm_g_sq)
    norm_w = math.sqrt(norm_w_sq)
    delta = abs(dot_gw) / (norm_g * norm_w + 1e-8)
    return delta, loss.item()


def get_weight_norm(model):
    """Compute total L2 norm of all parameters."""
    total = 0.0
    for p in model.parameters():
        total += p.data.norm().item() ** 2
    return math.sqrt(total)


def get_grad_norm(model):
    """Compute total L2 norm of all gradients."""
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm().item() ** 2
    return math.sqrt(total)


def evaluate(model, test_loader, device):
    """Evaluate model on test set."""
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


def train_epoch(model, train_loader, optimizer, scheduler, loss_fn, device,
                epoch, total_epochs, step_log_interval=100, method="fixed_wd",
                weight_decay=5e-4):
    """Train for one epoch, returning metrics."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    step_logs = []
    global_step = epoch * len(train_loader)

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Handle stagewise WD adjustment
        if method == "stagewise_wd":
            progress = epoch / total_epochs
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

        # Per-step logging
        current_step = global_step + batch_idx
        if batch_idx % step_log_interval == 0:
            step_metrics = {
                'step': current_step,
                'epoch': epoch,
                'batch_idx': batch_idx,
                'loss': loss.item(),
            }

            # Get optimizer-specific metrics
            if hasattr(optimizer, 'get_metrics'):
                step_metrics.update(optimizer.get_metrics())
            else:
                # Standard SGD
                wd = optimizer.param_groups[0].get('weight_decay', 0.0)
                step_metrics['lambda_t'] = wd
                step_metrics['weight_norm'] = get_weight_norm(model)
                step_metrics['lr'] = optimizer.param_groups[0]['lr']

            step_metrics['grad_norm'] = get_grad_norm(model)
            step_logs.append(step_metrics)

    # Update LR scheduler
    if scheduler is not None:
        scheduler.step()

    train_acc = 100.0 * correct / total
    train_loss = running_loss / total

    # Get epoch-level optimizer metrics
    epoch_metrics = {}
    if hasattr(optimizer, 'get_metrics'):
        epoch_metrics = optimizer.get_metrics()
    else:
        wd = optimizer.param_groups[0].get('weight_decay', 0.0)
        epoch_metrics = {
            'lambda_t': wd,
            'weight_norm': get_weight_norm(model),
            'lr': optimizer.param_groups[0]['lr'],
        }

    return train_loss, train_acc, epoch_metrics, step_logs


def run_training(config):
    """Main training loop.

    Args:
        config: dict with keys:
            arch, dataset, method, epochs, batch_size, lr, momentum,
            weight_decay, c, beta, lambda_min, lambda_max,
            seed, output_dir, gpu_id, step_log_interval
    """
    # Setup
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{config.get('gpu_id', 0)}"
                          if torch.cuda.is_available() else "cpu")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    num_classes = 10 if config['dataset'] == 'cifar10' else 100
    train_loader, test_loader, _ = get_dataloaders(
        config['dataset'], batch_size=config['batch_size'],
        data_dir=str(output_dir.parent.parent / "data"))

    # Model
    model = create_model(config['arch'], num_classes=num_classes)
    model = model.to(device)

    # Optimizer
    optimizer = create_optimizer(
        model, config['method'], lr=config['lr'],
        momentum=config['momentum'], weight_decay=config['weight_decay'],
        c=config.get('c', 1.0), beta=config.get('beta', 0.99),
        lambda_min=config.get('lambda_min', 1e-6),
        lambda_max=config.get('lambda_max', 0.01),
        variant=config.get('variant', 'conservative'))

    # LR scheduler
    lr_schedule = config.get('lr_schedule', 'multistep')
    base_opt = optimizer.optimizer if hasattr(optimizer, 'optimizer') else optimizer
    if lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=config['epochs'])
    else:
        # MultiStepLR: standard CIFAR milestones [80, 120] with gamma=0.1
        milestones = config.get('lr_milestones', [80, 120])
        lr_gamma = config.get('lr_gamma', 0.1)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            base_opt, milestones=milestones, gamma=lr_gamma)

    # Norm-matched WD: set epoch tracker
    if hasattr(optimizer, 'set_epoch'):
        pass  # Will be set per epoch

    loss_fn = nn.CrossEntropyLoss()

    # Logging files
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    step_log_path = output_dir / "step_metrics.jsonl"

    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Training loop
    start_time = time.time()
    all_epoch_records = []
    best_test_acc = 0.0

    print(f"Training {config['arch']} on {config['dataset']} with {config['method']}")
    print(f"Epochs: {config['epochs']}, LR: {config['lr']}, WD: {config['weight_decay']}")
    print(f"Device: {device}")
    print(f"Output: {output_dir}")

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        # Set epoch for norm-matched WD
        if hasattr(optimizer, 'set_epoch'):
            optimizer.set_epoch(epoch)

        train_loss, train_acc, opt_metrics, step_logs = train_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            epoch, config['epochs'],
            step_log_interval=config.get('step_log_interval', 100),
            method=config['method'], weight_decay=config['weight_decay'])

        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start

        gen_gap = train_acc - test_acc

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4),
            'test_loss': round(test_loss, 6),
            'gen_gap': round(gen_gap, 4),
            'weight_norm': round(opt_metrics.get('weight_norm', get_weight_norm(model)), 4),
            'mean_lambda_t': round(opt_metrics.get('lambda_t', config['weight_decay']), 8),
            'mean_delta_hat_t': round(opt_metrics.get('delta_hat_t', 0.0), 6),
            'ema_delta': round(opt_metrics.get('ema_delta', 0.0), 6),
            'lr': round(opt_metrics.get('lr', config['lr']), 8),
            'epoch_time_sec': round(epoch_time, 2),
        }
        all_epoch_records.append(epoch_record)

        # Append to JSONL
        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')
        with open(step_log_path, 'a') as f:
            for sl in step_logs:
                # Round floats for readability
                sl_clean = {}
                for k, v in sl.items():
                    if isinstance(v, float):
                        sl_clean[k] = round(v, 8)
                    else:
                        sl_clean[k] = v
                f.write(json.dumps(sl_clean) + '\n')

        best_test_acc = max(best_test_acc, test_acc)

        # Save checkpoint every 50 epochs (if enabled)
        if config.get('save_checkpoint') and (epoch + 1) % 50 == 0:
            ckpt_path = output_dir / f"checkpoint_epoch{epoch+1}.pt"
            ckpt = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_test_acc': best_test_acc}
            if hasattr(optimizer, 'state_dict'):
                ckpt['optimizer_state_dict'] = optimizer.state_dict()
            torch.save(ckpt, ckpt_path)

        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']}: "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                  f"test_acc={test_acc:.2f}% gen_gap={gen_gap:.2f}% "
                  f"wn={epoch_record['weight_norm']:.2f} "
                  f"lambda={epoch_record['mean_lambda_t']:.6f}")

    total_time = time.time() - start_time

    # Save final summary
    summary = {
        'config': config,
        'best_test_acc': round(best_test_acc, 4),
        'final_test_acc': round(all_epoch_records[-1]['test_acc'], 4),
        'final_train_acc': round(all_epoch_records[-1]['train_acc'], 4),
        'final_gen_gap': round(all_epoch_records[-1]['gen_gap'], 4),
        'final_weight_norm': round(all_epoch_records[-1]['weight_norm'], 4),
        'total_time_sec': round(total_time, 2),
        'epochs_completed': config['epochs'],
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone! Best test acc: {best_test_acc:.2f}%, "
          f"Total time: {total_time:.1f}s")

    return summary, all_epoch_records


def main():
    parser = argparse.ArgumentParser(description="AADWD Training Script")
    parser.add_argument('--arch', type=str, default='resnet20',
                        choices=['resnet20', 'vgg16_bn'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--method', type=str, default='fixed_wd',
                        choices=['no_wd', 'fixed_wd', 'stagewise_wd', 'cwd',
                                 'aadwd_conservative', 'aadwd_aggressive',
                                 'aadwd_square', 'random_dynamic_wd',
                                 'norm_matched_wd', 'equiv_cumulative_wd'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--c', type=float, default=1.0, help='AADWD base coefficient')
    parser.add_argument('--beta', type=float, default=0.999, help='AADWD EMA decay')
    parser.add_argument('--lambda_min', type=float, default=1e-6)
    parser.add_argument('--lambda_max', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--step_log_interval', type=int, default=100)
    parser.add_argument('--lr_schedule', type=str, default='multistep',
                        choices=['multistep', 'cosine'],
                        help='LR schedule: multistep (default, milestones=[80,120]) or cosine')
    parser.add_argument('--lr_milestones', type=int, nargs='+', default=[80, 120],
                        help='LR milestones for multistep schedule')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='LR decay factor at milestones')
    parser.add_argument('--save_checkpoint', action='store_true',
                        help='Save model checkpoints every 50 epochs')

    args = parser.parse_args()

    config = vars(args)
    # For AADWD methods, extract variant from method name
    if args.method.startswith('aadwd_'):
        config['variant'] = args.method.split('_', 1)[1]

    run_training(config)


if __name__ == '__main__':
    main()
