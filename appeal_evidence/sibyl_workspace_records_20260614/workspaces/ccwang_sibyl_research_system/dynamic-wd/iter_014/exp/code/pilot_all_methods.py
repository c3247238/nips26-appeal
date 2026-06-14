#!/usr/bin/env python3
"""Pilot validation: Run all 7 methods on CIFAR-10 for 1 epoch to verify end-to-end.

Also verifies that UDWDC(K=0) produces identical trajectory to FixedWD
(relative error < 1e-6) -- the key pilot pass criterion.
"""

import json
import os
import sys
import time
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from data import get_cifar10_loaders
from diagnostics.logger import DiagnosticLogger


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_method(method, model_name='resnet20', dataset='cifar10',
                     num_classes=10, epochs=1, batch_size=128,
                     max_samples=500, seed=42, save_dir='', device='cuda:0',
                     **kwargs):
    """Train one method and return results."""
    set_seed(seed)
    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    model = create_model(model_name, num_classes=num_classes).to(dev)
    opt = create_optimizer(method, model, lr=0.1, weight_decay=1e-4, **kwargs)

    train_loader, test_loader = get_cifar10_loaders(
        batch_size=batch_size, max_samples=max_samples, seed=seed
    )
    criterion = nn.CrossEntropyLoss()
    logger = DiagnosticLogger(save_dir=save_dir, method=method, seed=seed,
                              model_name=model_name, dataset=dataset)

    t0 = time.time()
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss, correct, total = 0, 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(dev), targets.to(dev)
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            opt.step()
            train_loss += loss.item() * inputs.size(0)
            _, pred = outputs.max(1)
            total += targets.size(0)
            correct += pred.eq(targets).sum().item()
        train_loss /= total
        train_acc = 100.0 * correct / total

        # Eval
        model.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(dev), targets.to(dev)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item() * inputs.size(0)
                _, pred = outputs.max(1)
                total += targets.size(0)
                correct += pred.eq(targets).sum().item()
        test_loss /= total
        test_acc = 100.0 * correct / total

        diagnostics = opt.get_diagnostics()
        logger.log_epoch(epoch, diagnostics, train_loss, test_loss, train_acc, test_acc)

    elapsed = time.time() - t0
    logger.save()

    return {
        'method': method,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'elapsed': elapsed,
        'n_layers': len(diagnostics),
        'diagnostics': diagnostics,
    }


def test_udwdc_identity():
    """Test UDWDC(K=0) == FixedWD trajectory."""
    print("\n--- Test: UDWDC(K=0) identity ---")
    set_seed(42)
    model_f = create_model('resnet20', num_classes=10)
    set_seed(42)
    model_u = create_model('resnet20', num_classes=10)

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model_f, model_u = model_f.to(dev), model_u.to(dev)

    opt_f = create_optimizer('FixedWD', model_f, lr=0.1, weight_decay=1e-4)
    opt_u = create_optimizer('UDWDC', model_u, lr=0.1, weight_decay=1e-4,
                             K_p=0.0, K_i=0.0, K_d=0.0)

    train_loader, _ = get_cifar10_loaders(batch_size=64, max_samples=500, seed=42)
    criterion = nn.CrossEntropyLoss()

    max_rel_err = 0.0
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(dev), targets.to(dev)

        # FixedWD step
        opt_f.zero_grad()
        loss_f = criterion(model_f(inputs), targets)
        loss_f.backward()
        opt_f.step()

        # UDWDC(K=0) step
        opt_u.zero_grad()
        loss_u = criterion(model_u(inputs), targets)
        loss_u.backward()
        opt_u.step()

        # Compare
        for (n1, p1), (n2, p2) in zip(model_f.named_parameters(),
                                       model_u.named_parameters()):
            if p1.norm().item() > 1e-8:
                rel = (p1.data - p2.data).norm().item() / p1.data.norm().item()
                max_rel_err = max(max_rel_err, rel)

    passed = max_rel_err < 1e-6
    print(f"  Max relative error: {max_rel_err:.2e}")
    print(f"  {'PASS' if passed else 'FAIL'} (threshold 1e-6)")
    return passed


def main():
    save_dir = '/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/pilots/setup'
    os.makedirs(save_dir, exist_ok=True)

    methods = ['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective', 'NoWD', 'UDWDC']

    print("=" * 70)
    print("Pilot Validation: All 7 Methods on CIFAR-10 (1 epoch, 500 samples)")
    print("=" * 70)

    results = {}
    for method in methods:
        print(f"\n--- {method} ---")
        r = train_one_method(method, epochs=1, max_samples=500,
                             save_dir=save_dir, device='cuda:0')
        results[method] = r
        print(f"  Train: {r['train_acc']:.1f}%, Test: {r['test_acc']:.1f}%, "
              f"Layers: {r['n_layers']}, Time: {r['elapsed']:.1f}s")

        # Check diagnostics
        for layer_name, diag in list(r['diagnostics'].items())[:1]:
            print(f"  Sample diag [{layer_name}]: rho={diag['rho_t']:.4f}, "
                  f"alpha={diag['alpha_t']:.4f}, wd={diag['effective_wd']:.6f}")

    # Identity test
    identity_pass = test_udwdc_identity()

    # Overall report
    print("\n" + "=" * 70)
    print("PILOT SUMMARY")
    print("=" * 70)

    all_pass = True
    checks = {
        'All methods complete': all(r['elapsed'] > 0 for r in results.values()),
        'Diagnostics logged': all(r['n_layers'] > 0 for r in results.values()),
        'rho_t non-trivial': all(
            any(d['rho_t'] > 0 for d in r['diagnostics'].values())
            for r in results.values()
        ),
        'UDWDC(K=0) == FixedWD': identity_pass,
    }

    for check, passed in checks.items():
        status = 'PASS' if passed else 'FAIL'
        print(f"  {status}: {check}")
        if not passed:
            all_pass = False

    # Save pilot summary
    summary = {
        'overall_recommendation': 'GO' if all_pass else 'NO_GO',
        'methods_tested': len(methods),
        'checks': {k: v for k, v in checks.items()},
        'per_method': {
            m: {
                'train_acc': r['train_acc'],
                'test_acc': r['test_acc'],
                'elapsed_sec': r['elapsed'],
                'n_layers_logged': r['n_layers'],
            }
            for m, r in results.items()
        },
    }

    with open(os.path.join(save_dir, 'pilot_validation_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'GO' if all_pass else 'NO_GO'}: Pilot validation {'passed' if all_pass else 'failed'}")
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
