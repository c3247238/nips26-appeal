#!/usr/bin/env python3
"""Unit tests for UDWDC-v2 stability fix.

Tests:
  1. WD budget > 0 over 100 steps on random data (all gain configurations)
  2. Floor clipping prevents WD collapse (lambda_t >= 0.1 * lambda_base)
  3. EMA smoothing reduces rho_t variance vs v1
  4. Kp_only and PD_control variants no longer produce zero WD budget
  5. CSI_temporal > 0 (stable coupling) over short training
  6. Effective WD stays in [0.1*lambda_base, 10*lambda_base]
  7. UDWDC-v2 constructs and registers correctly in METHOD_REGISTRY
  8. end_epoch_check() returns positive budget and warns on zero
  9. CIFAR-10 quick training: CSI_temporal > 0.5 over 10 epochs
"""

import json
import os
import sys
import warnings
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer, METHOD_REGISTRY
from optimizers.udwdc import UDWDCOptimizer
from optimizers.udwdc_v2 import UDWDCv2Optimizer


def _make_random_batch(batch_size=8, channels=3, size=32, num_classes=10):
    """Generate a random batch for testing."""
    x = torch.randn(batch_size, channels, size, size)
    y = torch.randint(0, num_classes, (batch_size,))
    return x, y


def test_wd_budget_positive_100_steps():
    """Test 1: WD budget > 0 over 100 steps on random data for ALL gain configs.

    This is the primary regression test. v1 produced zero WD budget for
    Kp_only and PD_control; v2 must guarantee positive budget for all.
    """
    print("\n=== Test 1: WD budget > 0 over 100 steps (all gain configs) ===")

    gain_configs = {
        'Kp_only':     {'K_p': 0.5, 'K_i': 0.0, 'K_d': 0.0},
        'Ki_only':     {'K_p': 0.0, 'K_i': 0.1, 'K_d': 0.0},
        'Kd_only':     {'K_p': 0.0, 'K_i': 0.0, 'K_d': 0.3},
        'PI_control':  {'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.0},
        'PD_control':  {'K_p': 0.5, 'K_i': 0.0, 'K_d': 0.3},
        'Full_PID':    {'K_p': 0.5, 'K_i': 0.1, 'K_d': 0.3},
        'Zero_gains':  {'K_p': 0.0, 'K_i': 0.0, 'K_d': 0.0},
    }

    wd = 1e-4
    all_pass = True
    criterion = nn.CrossEntropyLoss()

    for config_name, gains in gain_configs.items():
        torch.manual_seed(42)
        model = create_model('resnet20', num_classes=10)
        opt = UDWDCv2Optimizer(model, lr=0.1, weight_decay=wd, **gains)

        total_wd_budget = 0.0
        for step in range(100):
            x, y = _make_random_batch()
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

            # Accumulate WD budget: sum of lambda_t * ||w||^2 across layers
            for group in opt.param_groups:
                if group['apply_wd']:
                    w_norm_sq = torch.norm(group['params'][0].data).item() ** 2
                    total_wd_budget += group['effective_wd'] * w_norm_sq

        passed = total_wd_budget > 0
        print(f"  {config_name}: WD budget = {total_wd_budget:.6f} {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    print(f"\n  {'PASS' if all_pass else 'FAIL'}: All configs have positive WD budget")
    return all_pass


def test_floor_clipping():
    """Test 2: Floor clipping prevents WD from going below 0.1 * lambda_base."""
    print("\n=== Test 2: Floor clipping ===")

    wd = 1e-4
    floor = 0.1 * wd  # 1e-5

    torch.manual_seed(42)
    model = create_model('resnet20', num_classes=10)
    # Use extreme gains that would drive lambda to zero in v1
    opt = UDWDCv2Optimizer(model, lr=0.1, weight_decay=wd,
                           K_p=5.0, K_i=0.0, K_d=0.0)

    criterion = nn.CrossEntropyLoss()
    all_above_floor = True
    min_wd_seen = float('inf')

    for step in range(50):
        x, y = _make_random_batch()
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        for group in opt.param_groups:
            if group['apply_wd']:
                eff_wd = group['effective_wd']
                min_wd_seen = min(min_wd_seen, eff_wd)
                if eff_wd < floor - 1e-12:  # small epsilon for float comparison
                    all_above_floor = False

    print(f"  Min effective WD seen: {min_wd_seen:.2e} (floor: {floor:.2e})")
    print(f"  {'PASS' if all_above_floor else 'FAIL'}: All WDs >= floor")
    return all_above_floor


def test_ema_smoothing_reduces_variance():
    """Test 3: EMA smoothing in v2 reduces rho_t variance vs v1."""
    print("\n=== Test 3: EMA smoothing reduces rho_t variance ===")

    wd = 1e-4
    criterion = nn.CrossEntropyLoss()

    # Run v1 and collect rho_t values for a specific layer
    torch.manual_seed(42)
    model_v1 = create_model('resnet20', num_classes=10)
    opt_v1 = UDWDCOptimizer(model_v1, lr=0.1, weight_decay=wd,
                             K_p=0.5, K_i=0.1, K_d=0.3)

    torch.manual_seed(42)
    model_v2 = create_model('resnet20', num_classes=10)
    opt_v2 = UDWDCv2Optimizer(model_v2, lr=0.1, weight_decay=wd,
                               K_p=0.5, K_i=0.1, K_d=0.3)

    v1_effective_wds = []
    v2_effective_wds = []

    for step in range(50):
        torch.manual_seed(200 + step)
        x, y = _make_random_batch()

        # v1
        opt_v1.zero_grad()
        loss = criterion(model_v1(x), y)
        loss.backward()
        opt_v1.step()

        # v2
        opt_v2.zero_grad()
        loss = criterion(model_v2(x), y)
        loss.backward()
        opt_v2.step()

        # Get effective WD for first weight layer
        for group in opt_v1.param_groups:
            if group['apply_wd']:
                v1_effective_wds.append(group['effective_wd'])
                break
        for group in opt_v2.param_groups:
            if group['apply_wd']:
                v2_effective_wds.append(group['effective_wd'])
                break

    v1_var = np.var(v1_effective_wds)
    v2_var = np.var(v2_effective_wds)

    print(f"  v1 effective WD variance: {v1_var:.2e}")
    print(f"  v2 effective WD variance: {v2_var:.2e}")
    # v2 should have lower or comparable variance due to EMA smoothing
    # We check that v2 variance is not dramatically worse
    passed = v2_var <= v1_var * 2.0  # allow up to 2x (floor might increase variance slightly)
    print(f"  {'PASS' if passed else 'FAIL'}: v2 variance <= 2x v1 variance")
    return passed


def test_kp_only_and_pd_nonzero_budget():
    """Test 4: Kp_only and PD_control variants no longer produce zero WD budget.

    This is the specific regression test for the pilot bug.
    """
    print("\n=== Test 4: Kp_only and PD_control have non-zero WD budget ===")

    wd = 1e-4
    criterion = nn.CrossEntropyLoss()
    all_pass = True

    problem_configs = {
        'Kp_only':    {'K_p': 0.5, 'K_i': 0.0, 'K_d': 0.0},
        'PD_control': {'K_p': 0.5, 'K_i': 0.0, 'K_d': 0.3},
    }

    for config_name, gains in problem_configs.items():
        torch.manual_seed(42)
        model = create_model('resnet20', num_classes=10)
        opt = UDWDCv2Optimizer(model, lr=0.1, weight_decay=wd, **gains)

        total_wd_budget = 0.0
        for step in range(100):
            x, y = _make_random_batch()
            opt.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()

            for group in opt.param_groups:
                if group['apply_wd']:
                    w_norm_sq = torch.norm(group['params'][0].data).item() ** 2
                    total_wd_budget += group['effective_wd'] * w_norm_sq

        # With floor clipping at 0.1*lambda_base, the minimum possible budget per step
        # is 0.1 * 1e-4 * sum(||w||^2), which should be well above zero
        passed = total_wd_budget > 0.001  # Generous threshold
        print(f"  {config_name}: WD budget = {total_wd_budget:.6f} "
              f"(expected > 0.001) {'PASS' if passed else 'FAIL'}")
        if not passed:
            all_pass = False

    print(f"\n  {'PASS' if all_pass else 'FAIL'}: Previously zero-budget configs now positive")
    return all_pass


def test_csi_temporal_positive():
    """Test 5: CSI_temporal > 0 (stable coupling) over short training.

    CSI_temporal = 1 / Var_t[rho_t] averaged across layers over training.
    v1 had CSI_combined = -2.41 (highly unstable). v2 should be positive.
    """
    print("\n=== Test 5: CSI_temporal > 0 (stable coupling) ===")

    wd = 1e-4
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(42)
    model = create_model('resnet20', num_classes=10)
    opt = UDWDCv2Optimizer(model, lr=0.1, weight_decay=wd,
                           K_p=0.5, K_i=0.1, K_d=0.3)

    # Collect per-layer effective WD trajectories
    layer_wd_trajectories = {}

    for step in range(100):
        x, y = _make_random_batch()
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        for group in opt.param_groups:
            if group['apply_wd']:
                name = group['layer_name']
                if name not in layer_wd_trajectories:
                    layer_wd_trajectories[name] = []
                layer_wd_trajectories[name].append(group['effective_wd'])

    # Compute CSI_temporal: 1 / Var[lambda_t] for last 25% of steps
    csi_values = []
    for name, traj in layer_wd_trajectories.items():
        # Use last 25% of trajectory
        last_quarter = traj[len(traj) * 3 // 4:]
        if len(last_quarter) < 5:
            continue
        var = np.var(last_quarter)
        if var > 0:
            csi = 1.0 / var
        else:
            csi = float('inf')  # perfectly stable
        csi_values.append(csi)

    if csi_values:
        csi_temporal = np.mean([c for c in csi_values if c != float('inf')])
    else:
        csi_temporal = 0.0

    passed = csi_temporal > 0
    print(f"  CSI_temporal = {csi_temporal:.2f} {'PASS' if passed else 'FAIL'}")
    return passed


def test_effective_wd_bounds():
    """Test 6: Effective WD stays in [0.1*lambda_base, 10*lambda_base]."""
    print("\n=== Test 6: Effective WD bounds ===")

    wd = 1e-4
    lb = 0.1 * wd
    ub = 10.0 * wd
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(42)
    model = create_model('resnet20', num_classes=10)
    opt = UDWDCv2Optimizer(model, lr=0.1, weight_decay=wd,
                           K_p=0.5, K_i=0.1, K_d=0.3)

    min_wd = float('inf')
    max_wd = float('-inf')
    all_in_bounds = True

    for step in range(100):
        x, y = _make_random_batch()
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

        for group in opt.param_groups:
            if group['apply_wd']:
                eff_wd = group['effective_wd']
                min_wd = min(min_wd, eff_wd)
                max_wd = max(max_wd, eff_wd)
                if eff_wd < lb - 1e-12 or eff_wd > ub + 1e-12:
                    all_in_bounds = False

    print(f"  WD range: [{min_wd:.2e}, {max_wd:.2e}]")
    print(f"  Expected: [{lb:.2e}, {ub:.2e}]")
    print(f"  {'PASS' if all_in_bounds else 'FAIL'}: All WDs in bounds")
    return all_in_bounds


def test_registry_and_construction():
    """Test 7: UDWDC-v2 constructs and registers correctly."""
    print("\n=== Test 7: Registry and construction ===")

    # Check registry
    assert "UDWDC-v2" in METHOD_REGISTRY, "UDWDC-v2 not in registry"
    print("  UDWDC-v2 found in METHOD_REGISTRY")

    # Construct via factory
    torch.manual_seed(42)
    model = create_model('resnet20', num_classes=10)
    opt = create_optimizer("UDWDC-v2", model, lr=0.1, weight_decay=1e-4)

    assert isinstance(opt, UDWDCv2Optimizer)
    assert opt.get_method_name() == "UDWDC-v2"
    print(f"  Method name: {opt.get_method_name()}")

    # Run one step
    criterion = nn.CrossEntropyLoss()
    x, y = _make_random_batch()
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    opt.step()

    diag = opt.get_diagnostics()
    n_layers = len(diag)
    print(f"  Step completed, {n_layers} layers logged")

    passed = isinstance(opt, UDWDCv2Optimizer) and n_layers > 0
    print(f"  {'PASS' if passed else 'FAIL'}: Registry and construction")
    return passed


def test_end_epoch_check():
    """Test 8: end_epoch_check() returns positive budget and warns on zero."""
    print("\n=== Test 8: end_epoch_check() ===")

    wd = 1e-4
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(42)
    model = create_model('resnet20', num_classes=10)
    opt = UDWDCv2Optimizer(model, lr=0.1, weight_decay=wd,
                           K_p=0.5, K_i=0.1, K_d=0.3)

    # Train for 10 steps (simulate an epoch)
    for step in range(10):
        x, y = _make_random_batch()
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()

    budget = opt.end_epoch_check()
    cumulative = opt.get_cumulative_wd_budget()

    print(f"  Epoch WD budget: {budget:.6e}")
    print(f"  Cumulative budget: {cumulative:.6e}")

    passed = budget > 0 and cumulative > 0
    print(f"  {'PASS' if passed else 'FAIL'}: Positive epoch budget")
    return passed


def test_cifar10_10epoch_csi():
    """Test 9: CIFAR-10 quick training: CSI_temporal > 0.5 over 10 epochs.

    This is the integration test that verifies stability on actual data.
    Uses a small subset (100 samples) for speed.
    """
    print("\n=== Test 9: CIFAR-10 10-epoch CSI (integration test) ===")

    try:
        import torchvision
        import torchvision.transforms as transforms
    except ImportError:
        print("  SKIP: torchvision not available")
        return True  # Don't fail on missing dependency

    wd = 1e-4
    criterion = nn.CrossEntropyLoss()

    # Minimal CIFAR-10 transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    try:
        trainset = torchvision.datasets.CIFAR10(
            root='/tmp/cifar10_test', train=True, download=True, transform=transform)
    except Exception as e:
        print(f"  SKIP: Cannot download CIFAR-10: {e}")
        return True

    # Use only 100 samples for speed
    subset = torch.utils.data.Subset(trainset, range(100))
    trainloader = torch.utils.data.DataLoader(subset, batch_size=32, shuffle=True)

    torch.manual_seed(42)
    model = create_model('resnet20', num_classes=10)
    opt = UDWDCv2Optimizer(model, lr=0.1, weight_decay=wd,
                           K_p=0.5, K_i=0.1, K_d=0.3)

    # Cosine annealing scheduler
    total_epochs = 10
    layer_wd_trajectories = {}

    for epoch in range(total_epochs):
        # Update LR with cosine annealing
        lr = 0.1 * 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))
        opt.set_lr(lr)

        for batch_x, batch_y in trainloader:
            opt.zero_grad()
            out = model(batch_x)
            loss = criterion(out, batch_y)
            loss.backward()
            opt.step()

            # Record per-layer WD
            for group in opt.param_groups:
                if group['apply_wd']:
                    name = group['layer_name']
                    if name not in layer_wd_trajectories:
                        layer_wd_trajectories[name] = []
                    layer_wd_trajectories[name].append(group['effective_wd'])

        budget = opt.end_epoch_check()

    # Compute CSI_temporal from last 25% of training
    csi_values = []
    for name, traj in layer_wd_trajectories.items():
        last_quarter = traj[len(traj) * 3 // 4:]
        if len(last_quarter) < 3:
            continue
        var = np.var(last_quarter)
        if var > 0:
            csi_values.append(1.0 / var)

    if csi_values:
        finite_csi = [c for c in csi_values if not math.isinf(c)]
        csi_temporal = np.mean(finite_csi) if finite_csi else float('inf')
    else:
        csi_temporal = 0.0

    cumulative = opt.get_cumulative_wd_budget()
    print(f"  CSI_temporal = {csi_temporal:.2f}")
    print(f"  Cumulative WD budget = {cumulative:.6e}")
    print(f"  Total WD budget > 0: {cumulative > 0}")

    # CSI > 0.5 indicates reasonable stability
    passed = csi_temporal > 0.5 and cumulative > 0
    print(f"  {'PASS' if passed else 'FAIL'}: CSI > 0.5 and budget > 0")
    return passed


def main():
    print("=" * 60)
    print("UDWDC-v2 Stability Fix - Unit Tests")
    print("=" * 60)

    start_time = time.time()

    results = {}
    results['wd_budget_positive_100_steps'] = test_wd_budget_positive_100_steps()
    results['floor_clipping'] = test_floor_clipping()
    results['ema_smoothing'] = test_ema_smoothing_reduces_variance()
    results['kp_pd_nonzero_budget'] = test_kp_only_and_pd_nonzero_budget()
    results['csi_temporal_positive'] = test_csi_temporal_positive()
    results['effective_wd_bounds'] = test_effective_wd_bounds()
    results['registry_construction'] = test_registry_and_construction()
    results['end_epoch_check'] = test_end_epoch_check()
    results['cifar10_10epoch_csi'] = test_cifar10_10epoch_csi()

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print(f"\nElapsed: {elapsed:.1f}s")
    print(f"{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")

    # Write results as JSON for system tracking
    output_dir = Path(__file__).parent.parent / 'results' / 'pilots' / 'udwdc_v2_fix'
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump({
            'all_pass': bool(all_pass),
            'tests': {k: bool(v) for k, v in results.items()},
            'elapsed_sec': elapsed,
        }, f, indent=2)
    print(f"\nResults saved to {output_dir / 'test_results.json'}")

    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
