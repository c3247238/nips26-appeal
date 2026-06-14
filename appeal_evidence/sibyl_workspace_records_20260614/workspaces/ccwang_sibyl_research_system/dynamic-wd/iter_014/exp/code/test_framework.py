#!/usr/bin/env python3
"""Unit tests for the dynamic WD framework.

Tests:
  (a) UDWDC reduces to FixedWD when K_p=K_i=K_d=0
  (b) Gradient/weight norms computed correctly
  (c) Clipping works
  (d) All optimizer wrappers construct without error
  (e) Diagnostic logger saves correctly
  (f) Data loaders work
  (g) Models have correct parameter counts
"""

import json
import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from models import create_model
from optimizers import create_optimizer
from optimizers.udwdc import UDWDCOptimizer
from optimizers.fixed_wd import FixedWDOptimizer
from diagnostics.logger import DiagnosticLogger


def test_udwdc_reduces_to_fixed():
    """Test (a): UDWDC with K_p=K_i=K_d=0 should produce identical
    parameter trajectory to FixedWD."""
    print("\n=== Test (a): UDWDC reduces to FixedWD ===")

    torch.manual_seed(42)
    model_fixed = create_model('resnet20', num_classes=10)
    torch.manual_seed(42)
    model_udwdc = create_model('resnet20', num_classes=10)

    # Verify initial params are identical
    for (n1, p1), (n2, p2) in zip(model_fixed.named_parameters(),
                                   model_udwdc.named_parameters()):
        assert torch.allclose(p1.data, p2.data), f"Initial params differ at {n1}"

    lr = 0.1
    wd = 1e-4
    fixed_opt = FixedWDOptimizer(model_fixed, lr=lr, weight_decay=wd)
    udwdc_opt = UDWDCOptimizer(model_udwdc, lr=lr, weight_decay=wd,
                                K_p=0.0, K_i=0.0, K_d=0.0)

    # Run 5 steps with identical random inputs
    criterion = nn.CrossEntropyLoss()
    device = torch.device('cpu')

    max_rel_error = 0.0
    for step in range(5):
        torch.manual_seed(100 + step)
        x = torch.randn(4, 3, 32, 32)
        y = torch.randint(0, 10, (4,))

        # Fixed WD step
        fixed_opt.zero_grad()
        out_f = model_fixed(x)
        loss_f = criterion(out_f, y)
        loss_f.backward()
        fixed_opt.step()

        # UDWDC step (K=0)
        udwdc_opt.zero_grad()
        out_u = model_udwdc(x)
        loss_u = criterion(out_u, y)
        loss_u.backward()
        udwdc_opt.step()

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(model_fixed.named_parameters(),
                                       model_udwdc.named_parameters()):
            if p1.norm().item() > 1e-8:
                rel_err = (p1.data - p2.data).norm().item() / p1.data.norm().item()
                max_rel_error = max(max_rel_error, rel_err)

    print(f"  Max relative error after 5 steps: {max_rel_error:.2e}")
    passed = max_rel_error < 1e-6
    print(f"  {'PASS' if passed else 'FAIL'}: UDWDC(K=0) == FixedWD (threshold 1e-6)")
    return passed


def test_norm_computation():
    """Test (b): Gradient and weight norms computed correctly."""
    print("\n=== Test (b): Norm computation ===")

    model = create_model('resnet20', num_classes=10)
    opt = FixedWDOptimizer(model, lr=0.1, weight_decay=1e-4)

    # Forward + backward
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()

    diagnostics = opt.get_diagnostics()

    all_correct = True
    for name, param in model.named_parameters():
        if param.grad is None or name not in diagnostics:
            continue

        expected_w = torch.norm(param.data).item()
        expected_g = torch.norm(param.grad.data).item()

        actual_w = diagnostics[name]['w_norm']
        actual_g = diagnostics[name]['g_norm']

        w_ok = abs(expected_w - actual_w) < 1e-5
        g_ok = abs(expected_g - actual_g) < 1e-5

        if not w_ok or not g_ok:
            print(f"  FAIL at {name}: w={expected_w:.6f} vs {actual_w:.6f}, "
                  f"g={expected_g:.6f} vs {actual_g:.6f}")
            all_correct = False

    print(f"  Checked {len(diagnostics)} layers")
    print(f"  {'PASS' if all_correct else 'FAIL'}: Norm computation")
    return all_correct


def test_clipping():
    """Test (c): UDWDC clipping works correctly."""
    print("\n=== Test (c): Clipping ===")

    model = create_model('resnet20', num_classes=10)
    wd = 1e-4
    opt = UDWDCOptimizer(model, lr=0.1, weight_decay=wd,
                          K_p=100.0, K_i=100.0, K_d=100.0,  # Very large gains
                          lambda_max_factor=10.0)

    # Forward + backward to generate gradients
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    criterion = nn.CrossEntropyLoss()
    opt.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    opt._compute_effective_wd()

    # Check that all effective WDs are within bounds
    all_clipped = True
    for group in opt.param_groups:
        if group['apply_wd']:
            eff_wd = group['effective_wd']
            if eff_wd < 0 or eff_wd > 10.0 * wd:
                print(f"  FAIL at {group['layer_name']}: wd={eff_wd} out of [0, {10*wd}]")
                all_clipped = False

    print(f"  {'PASS' if all_clipped else 'FAIL'}: Clipping bounds [0, {10*wd}]")
    return all_clipped


def test_all_optimizers_construct():
    """Test (d): All 7 optimizer wrappers construct without error."""
    print("\n=== Test (d): All optimizers construct ===")

    methods = ['FixedWD', 'CWD', 'SWD', 'CPR', 'DefazioCorrective', 'NoWD', 'UDWDC']
    model = create_model('resnet20', num_classes=10)
    all_ok = True

    for method in methods:
        try:
            opt = create_optimizer(method, model, lr=0.1, weight_decay=1e-4)
            print(f"  OK: {method} -> {opt.get_method_name()}")
        except Exception as e:
            print(f"  FAIL: {method} -> {e}")
            all_ok = False

    # Also test one step with each
    print("\n  Running one training step with each method...")
    for method in methods:
        try:
            torch.manual_seed(42)
            m = create_model('resnet20', num_classes=10)
            opt = create_optimizer(method, m, lr=0.1, weight_decay=1e-4)
            criterion = nn.CrossEntropyLoss()
            x = torch.randn(4, 3, 32, 32)
            y = torch.randint(0, 10, (4,))
            opt.zero_grad()
            loss = criterion(m(x), y)
            loss.backward()
            opt.step()
            diag = opt.get_diagnostics()
            n_layers = len(diag)
            print(f"  OK: {method} step completed, {n_layers} layers logged")
        except Exception as e:
            print(f"  FAIL: {method} step -> {e}")
            all_ok = False

    print(f"\n  {'PASS' if all_ok else 'FAIL'}: All optimizers")
    return all_ok


def test_diagnostic_logger():
    """Test (e): DiagnosticLogger saves correctly."""
    print("\n=== Test (e): Diagnostic logger ===")

    with tempfile.TemporaryDirectory() as tmpdir:
        logger = DiagnosticLogger(
            save_dir=tmpdir, method='TestMethod', seed=42,
            model_name='resnet20', dataset='cifar10'
        )

        # Fake diagnostic data
        for epoch in range(3):
            diagnostics = {
                'conv1.weight': {
                    'w_norm': 1.0 + epoch * 0.1,
                    'g_norm': 0.5 - epoch * 0.01,
                    'rho_t': 0.5,
                    'alpha_t': 0.3,
                    'effective_wd': 1e-4,
                }
            }
            logger.log_epoch(epoch, diagnostics, 2.0 - epoch * 0.5, 2.1 - epoch * 0.4,
                             50.0 + epoch * 10, 45.0 + epoch * 8, lr=0.1)

        epoch_file, traj_file = logger.save()

        # Verify files exist and are valid JSON
        with open(epoch_file) as f:
            epoch_data = json.load(f)
        with open(traj_file) as f:
            traj_data = json.load(f)

        ok = (len(epoch_data['epochs']) == 3 and
              'conv1.weight' in traj_data['layers'] and
              len(traj_data['layers']['conv1.weight']['rho_t']) == 3)

        print(f"  Epoch file: {len(epoch_data['epochs'])} epochs")
        print(f"  Trajectory file: {len(traj_data['layers'])} layers")
        print(f"  Total WD budget: {logger.get_total_wd_budget():.6f}")
        print(f"  {'PASS' if ok else 'FAIL'}: Logger save/load")
        return ok


def test_models():
    """Test (g): Models have correct shapes and parameter counts."""
    print("\n=== Test (g): Model checks ===")

    tests = [
        ('resnet20', 10, (1, 3, 32, 32), 270000),    # ~270K
        ('vgg16_bn', 100, (1, 3, 32, 32), 15000000),  # ~15M
        ('resnet50', 1000, (1, 3, 224, 224), 25000000), # ~25.6M
        ('vit_s_16', 1000, (1, 3, 224, 224), 21000000), # ~22M
    ]

    all_ok = True
    for name, nc, input_shape, min_params in tests:
        try:
            model = create_model(name, num_classes=nc)
            x = torch.randn(*input_shape)
            out = model(x)
            n_params = sum(p.numel() for p in model.parameters())

            shape_ok = out.shape == (1, nc)
            params_ok = n_params > min_params * 0.5  # Rough check

            print(f"  {name}: params={n_params:,}, output={out.shape}, "
                  f"{'OK' if shape_ok and params_ok else 'FAIL'}")
            if not (shape_ok and params_ok):
                all_ok = False
        except Exception as e:
            print(f"  {name}: FAIL -> {e}")
            all_ok = False

    print(f"\n  {'PASS' if all_ok else 'FAIL'}: All models")
    return all_ok


def test_resnet101():
    """Test ResNet-101 constructs and runs."""
    print("\n=== Test: ResNet-101 ===")
    try:
        model = create_model('resnet101', num_classes=1000)
        x = torch.randn(1, 3, 224, 224)
        out = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        ok = out.shape == (1, 1000) and n_params > 40_000_000
        print(f"  ResNet-101: params={n_params:,}, output={out.shape}, {'OK' if ok else 'FAIL'}")
        return ok
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def main():
    print("=" * 60)
    print("Dynamic WD Framework - Unit Tests")
    print("=" * 60)

    results = {}
    results['udwdc_reduces_to_fixed'] = test_udwdc_reduces_to_fixed()
    results['norm_computation'] = test_norm_computation()
    results['clipping'] = test_clipping()
    results['all_optimizers'] = test_all_optimizers_construct()
    results['diagnostic_logger'] = test_diagnostic_logger()
    results['models'] = test_models()
    results['resnet101'] = test_resnet101()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_pass = False

    print(f"\n{'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    return all_pass


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
