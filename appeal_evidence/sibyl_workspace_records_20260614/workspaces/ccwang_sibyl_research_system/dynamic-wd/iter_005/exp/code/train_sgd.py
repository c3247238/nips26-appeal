"""
SGD Baseline Training Script for Unified Dynamic WD Framework.

Same as train_unified.py but uses SGD+momentum instead of AdamW.
Tests whether WD method irrelevance is specific to adaptive optimizers.
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
from optimizers import create_sgd_optimizer
from train_unified import (
    compute_csi, compute_ais, compute_bem,
    evaluate, train_one_epoch
)


def run_training(config):
    """Main training loop with SGD optimizer."""
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed_all(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device(f"cuda:{config.get('gpu_id', 0)}"
                          if torch.cuda.is_available() else "cpu")
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Data
    dataset = config['dataset']
    num_classes = 10 if dataset == 'cifar10' else 100
    code_dir = Path(__file__).parent
    data_dir = str(code_dir.parent / "data")
    train_loader, test_loader, _ = get_dataloaders(
        dataset, batch_size=config['batch_size'], data_dir=data_dir)
    dataset_size = 50000

    # Model
    model = create_model(config['arch'], num_classes=num_classes)
    model = model.to(device)

    # SGD Optimizer with Phi modulator
    optimizer = create_sgd_optimizer(
        model, config['wd_method'],
        lr=config['lr'], wd=config['wd'],
        epochs=config['epochs'], batch_size=config['batch_size'],
        dataset_size=dataset_size,
        wd_min=config.get('wd_min', 0.0),
        swd_sensitivity=config.get('swd_sensitivity', 1.0),
        mask_prob=config.get('mask_prob', 0.5),
    )

    # LR scheduler (cosine annealing, standard for SGD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs'])

    loss_fn = nn.CrossEntropyLoss()

    config['optimizer'] = 'sgd'
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)

    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    weight_norms_history = []
    wd_history = []
    best_test_acc = 0.0

    print(f"{'='*60}")
    print(f"Training {config['arch']} on {config['dataset']} [SGD]")
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

        opt_metrics = optimizer.get_metrics()
        weight_norm = opt_metrics.get('weight_norm', 0.0)
        weight_norms_history.append(weight_norm)
        wd_history.append(opt_metrics.get('effective_wd', config['wd']))

        csi = compute_csi(weight_norms_history)
        ais = compute_ais(per_layer_aligns)
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
            'csi': round(csi, 6),
            'ais': round(ais, 6),
            'bem': round(bem, 6),
            'lr': round(opt_metrics.get('lr', config['lr']), 8),
            'epoch_time_sec': round(epoch_time, 2),
        }

        with open(epoch_log_path, 'a') as f:
            f.write(json.dumps(epoch_record) + '\n')

        if epoch % 10 == 0 or epoch == config['epochs'] - 1:
            print(f"  E{epoch:3d}/{config['epochs']}: "
                  f"loss={train_loss:.4f} tr_acc={train_acc:.2f}% "
                  f"te_acc={test_acc:.2f}% wn={weight_norm:.2f} "
                  f"CSI={csi:.4f} AIS={ais:.4f} BEM={bem:.4f}")

    total_time = time.time() - start_time

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
        'total_time_sec': round(total_time, 2),
        'epochs_completed': config['epochs'],
    }
    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "_DONE", 'w') as f:
        f.write(f"completed at {datetime.now().isoformat()}\n")

    print(f"\nDone! Best: {best_test_acc:.2f}%, Time: {total_time:.1f}s")
    return summary


def main():
    parser = argparse.ArgumentParser(description="SGD Baseline Training")
    parser.add_argument('--arch', type=str, default='resnet20',
                        choices=['resnet20', 'vgg16_bn'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'])
    parser.add_argument('--wd_method', type=str, default='constant',
                        choices=['constant', 'cosine_schedule', 'cwd_hard',
                                 'swd', 'random_mask', 'half_lambda', 'no_wd'])
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--wd', type=float, default=5e-4)
    parser.add_argument('--wd_min', type=float, default=0.0)
    parser.add_argument('--swd_sensitivity', type=float, default=1.0)
    parser.add_argument('--mask_prob', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()
    config = vars(args)
    run_training(config)


if __name__ == '__main__':
    main()
