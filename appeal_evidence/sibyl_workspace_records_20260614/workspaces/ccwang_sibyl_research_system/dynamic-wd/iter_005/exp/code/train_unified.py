"""
Unified Dynamic Weight Decay Framework - Training Script.

Supports all WD methods under the Phi modulator framework with AdamW.
Logs comprehensive diagnostics including CSI, AIS, BEM metrics.
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

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from data import get_dataloaders
from optimizers import create_unified_optimizer


def compute_per_layer_diagnostics(model):
    """Compute per-layer weight norms and gradient norms."""
    layers = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            w_norm = param.data.norm().item()
            g_norm = param.grad.data.norm().item() if param.grad is not None else 0.0
            # Alignment
            if param.grad is not None and w_norm > 1e-10 and g_norm > 1e-10:
                cos_sim = torch.nn.functional.cosine_similarity(
                    param.data.flatten().unsqueeze(0),
                    param.grad.data.flatten().unsqueeze(0)
                ).item()
            else:
                cos_sim = 0.0
            layers[name] = {
                'w_norm': w_norm,
                'g_norm': g_norm,
                'cos_sim': cos_sim,
            }
    return layers


def compute_csi(weight_norms_history, window=10):
    """Coupling Stability Index: CV of weight norm changes over a window.

    CSI_raw = std(delta_norms[-window:]) / (mean(|delta_norms[-window:]|) + eps)

    For relative normalization (CSI_rel = CSI_raw / CSI_constant), normalize
    at analysis time using the constant baseline's CSI value, so that
    CSI_rel(constant) = 1.0 by definition.

    Lower CSI = more stable coupling between LR and WD.
    """
    if len(weight_norms_history) < 2:
        return 0.0
    deltas = []
    for i in range(1, len(weight_norms_history)):
        deltas.append(weight_norms_history[i] - weight_norms_history[i - 1])
    recent = deltas[-window:]
    if not recent:
        return 0.0
    mean_abs = sum(abs(d) for d in recent) / len(recent)
    if mean_abs < 1e-10:
        return 0.0
    mean_d = sum(recent) / len(recent)
    var_d = sum((d - mean_d) ** 2 for d in recent) / len(recent)
    std_d = math.sqrt(var_d)
    return std_d / (mean_abs + 1e-10)


def compute_ais(per_layer_alignments):
    """Alignment Informativeness Score: normalized entropy of per-layer alignment.

    AIS = H(alignment_distribution) / H_max

    where H is the Shannon entropy of the binned distribution of per-layer
    |cos(g_l, w_l)| values, and H_max = log(n_bins) is the maximum entropy.

    Per-layer averaging: alignment_l = |cos(g_l, w_l)| for each layer l,
    then the distribution {alignment_1, ..., alignment_L} is binned into
    n_bins=10 equal-width bins over [0, 1].

    AIS in [0, 1]:
      - AIS -> 1: alignment varies maximally across layers (informative)
      - AIS -> 0: alignment is uniform across layers (uninformative)

    Input: list of per-layer |cos(g, w)| values for a single epoch.
    """
    if len(per_layer_alignments) < 3:
        return 0.0
    # Bin into 10 bins from 0 to 1
    n_bins = 10
    counts = [0] * n_bins
    for v in per_layer_alignments:
        v_clamp = max(0.0, min(1.0 - 1e-10, abs(v)))
        bin_idx = int(v_clamp * n_bins)
        counts[bin_idx] += 1
    total = sum(counts)
    if total == 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p + 1e-10)
    max_entropy = math.log(n_bins)
    return entropy / max_entropy if max_entropy > 0 else 0.0


def compute_bem(mean_wd_schedule, constant_wd, tolerance=0.01):
    """Budget Equivalence Metric (signed): (mean(lambda_t) - lambda_const) / lambda_const.

    BEM = 0 means budget-equivalent to constant.
    BEM = -0.5 means half the WD budget (e.g., half_lambda).
    BEM = -1.0 means no WD (e.g., no_wd).
    Signed BEM preserves direction: negative = less WD, positive = more WD.
    """
    if constant_wd < 1e-10:
        return 0.0
    return (mean_wd_schedule - constant_wd) / constant_wd


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


def train_one_epoch(model, train_loader, optimizer, scheduler, loss_fn, device,
                    epoch, total_epochs):
    """Train for one epoch, return epoch metrics."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    per_layer_alignments = []  # collected once per epoch at the last batch

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Collect per-layer alignment at the last batch for AIS
        if batch_idx == len(train_loader) - 1:
            for name, p in model.named_parameters():
                if p.grad is not None and p.data.norm().item() > 1e-10 and p.grad.norm().item() > 1e-10:
                    cos = torch.nn.functional.cosine_similarity(
                        p.data.flatten().unsqueeze(0),
                        p.grad.data.flatten().unsqueeze(0)
                    ).item()
                    per_layer_alignments.append(abs(cos))

        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    if scheduler is not None:
        scheduler.step()

    train_acc = 100.0 * correct / total
    train_loss = running_loss / total

    return train_loss, train_acc, per_layer_alignments


def run_training(config):
    """Main training loop with full diagnostics."""
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{config.get('gpu_id', 0)}"
                          if torch.cuda.is_available() else "cpu")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data - use shared data directory
    dataset = config['dataset']
    num_classes = 10 if dataset == 'cifar10' else 100
    # Shared data dir at exp/data level, falling back to torch cache
    code_dir = Path(__file__).parent
    data_dir = str(code_dir.parent / "data")
    train_loader, test_loader, num_classes_from_data = get_dataloaders(
        dataset, batch_size=config['batch_size'], data_dir=data_dir)
    dataset_size = 50000  # CIFAR-10/100 both have 50k training samples

    # Model
    model = create_model(config['arch'], num_classes=num_classes)
    model = model.to(device)

    # Optimizer with Phi modulator
    optimizer = create_unified_optimizer(
        model, config['wd_method'],
        lr=config['lr'], wd=config['wd'],
        epochs=config['epochs'], batch_size=config['batch_size'],
        dataset_size=dataset_size,
        wd_min=config.get('wd_min', 0.0),
        cwd_beta=config.get('cwd_beta', 100.0),
        swd_sensitivity=config.get('swd_sensitivity', 1.0),
        mask_prob=config.get('mask_prob', 0.5),
    )

    # LR scheduler
    lr_schedule = config.get('lr_schedule', 'cosine')
    if lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'])
    else:
        milestones = config.get('lr_milestones', [80, 120])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.1)

    loss_fn = nn.CrossEntropyLoss()

    # Save config
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    # Training state
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    weight_norms_history = []
    alignment_history = []
    wd_history = []
    best_test_acc = 0.0

    print(f"{'='*60}")
    print(f"Training {config['arch']} on {config['dataset']}")
    print(f"WD method: {config['wd_method']}, base WD: {config['wd']}")
    print(f"Epochs: {config['epochs']}, LR: {config['lr']}, Batch: {config['batch_size']}")
    print(f"Seed: {config['seed']}, Device: {device}")
    print(f"Output: {output_dir}")
    print(f"{'='*60}")

    start_time = time.time()

    for epoch in range(config['epochs']):
        epoch_start = time.time()

        train_loss, train_acc, per_layer_aligns = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            epoch, config['epochs'])

        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start

        # Get optimizer metrics
        opt_metrics = optimizer.get_metrics()
        weight_norm = opt_metrics.get('weight_norm', 0.0)
        weight_norms_history.append(weight_norm)
        alignment_history.extend(per_layer_aligns)  # accumulate per-layer alignments
        wd_history.append(opt_metrics.get('effective_wd', config['wd']))

        # Compute framework metrics
        csi = compute_csi(weight_norms_history)
        ais = compute_ais(per_layer_aligns)  # AIS from this epoch's per-layer alignment diversity
        mean_wd = sum(wd_history) / len(wd_history)
        bem = compute_bem(mean_wd, config['wd'])

        best_test_acc = max(best_test_acc, test_acc)

        epoch_record = {
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 4),
            'test_acc': round(test_acc, 4),
            'test_loss': round(test_loss, 6),
            'gen_gap': round(train_acc - test_acc, 4),
            'weight_norm': round(weight_norm, 4),
            'alignment': round(sum(per_layer_aligns) / max(len(per_layer_aligns), 1), 6),
            'csi': round(csi, 6),
            'ais': round(ais, 6),
            'bem': round(bem, 6),
            'mean_wd': round(mean_wd, 8),
            'lr': round(opt_metrics.get('lr', config['lr']), 8),
            'epoch_time_sec': round(epoch_time, 2),
        }
        # Add modulator-specific metrics
        for k, v in opt_metrics.items():
            if k not in epoch_record and isinstance(v, (int, float)):
                epoch_record[k] = round(v, 6) if isinstance(v, float) else v

        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')

        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"  E{epoch:3d}/{config['epochs']}: "
                  f"loss={train_loss:.4f} tr_acc={train_acc:.2f}% "
                  f"te_acc={test_acc:.2f}% wn={weight_norm:.2f} "
                  f"CSI={csi:.4f} AIS={ais:.4f} BEM={bem:.4f}")

    total_time = time.time() - start_time

    # Final summary
    summary = {
        'config': config,
        'best_test_acc': round(best_test_acc, 4),
        'final_test_acc': round(test_acc, 4),
        'final_train_acc': round(train_acc, 4),
        'final_gen_gap': round(train_acc - test_acc, 4),
        'final_weight_norm': round(weight_norm, 4),
        'final_csi': round(csi, 6),
        'final_ais': round(ais, 6),
        'final_bem': round(bem, 6),
        'mean_wd_actual': round(mean_wd, 8),
        'total_time_sec': round(total_time, 2),
        'epochs_completed': config['epochs'],
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Write completion marker
    with open(output_dir / "_DONE", 'w') as f:
        f.write(f"completed at {datetime.now().isoformat()}\n")

    print(f"\nDone! Best: {best_test_acc:.2f}%, Time: {total_time:.1f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Unified WD Framework Training")
    parser.add_argument('--arch', type=str, default='resnet20',
                        choices=['resnet20', 'resnet20_nobn', 'vgg16_bn'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--wd_method', type=str, default='constant',
                        choices=['constant', 'cosine_schedule', 'linear_schedule',
                                 'cwd_hard', 'cwd_soft', 'swd', 'random_mask',
                                 'half_lambda', 'no_wd'])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--wd_min', type=float, default=0.0)
    parser.add_argument('--cwd_beta', type=float, default=100.0)
    parser.add_argument('--swd_sensitivity', type=float, default=1.0)
    parser.add_argument('--mask_prob', type=float, default=0.5)
    parser.add_argument('--lr_schedule', type=str, default='cosine',
                        choices=['cosine', 'multistep'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    config = vars(args)
    run_training(config)


if __name__ == '__main__':
    main()
