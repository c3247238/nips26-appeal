"""
FULL 200-epoch experiment runner for all tiers.

Incorporates pilot learnings:
  - beta=0.999 (tier0 diagnostic recommended upgrade from 0.99)
  - MultiStepLR milestones=[80, 120], gamma=0.1 (standard CIFAR)
  - Conservative c=0.001 (pilot showed c=0.01 over-regularized with beta=0.999)
  - Aggressive c=0.01 (worked well in pilot)
  - Square c=0.01 (near-miss in pilot, same c as aggressive)
  - CWD vectorized (7x speedup from pilot)

Usage:
    python run_full.py --task TASK_ID --gpu GPU_ID --results_dir DIR
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from models import create_model
from data import get_dataloaders
from optimizers import create_optimizer


# ── Shared constants ──
SEED = 42
EPOCHS = 200
BATCH_SIZE = 128
LR = 0.1
MOMENTUM = 0.9
MILESTONES = [80, 120]
LR_GAMMA = 0.1
BETA = 0.999          # upgraded from 0.99 per tier0 pilot
LAMBDA_MIN = 1e-6
LAMBDA_MAX = 0.01
STEP_LOG_INTERVAL = 200  # less frequent for 200 epochs


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


def train_single(config, device, data_dir):
    """Train a single experiment configuration."""
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"  arch={config['arch']}, dataset={config['dataset']}")
    print(f"  method={config['method']}, epochs={config['epochs']}")
    print(f"  lr={config['lr']}, wd={config['weight_decay']}")
    if "aadwd" in config["method"]:
        print(f"  c={config.get('c')}, beta={config.get('beta')}")
    print(f"  milestones={config.get('milestones', MILESTONES)}")
    print(f"  Device: {device}")
    print(f"{'='*60}")

    # Data
    train_loader, test_loader, num_classes = get_dataloaders(
        config["dataset"], batch_size=config["batch_size"], data_dir=data_dir)

    # Model
    model = create_model(config["arch"], num_classes=num_classes).to(device)

    # Optimizer
    optimizer = create_optimizer(
        model, config["method"], lr=config["lr"],
        momentum=config["momentum"], weight_decay=config["weight_decay"],
        c=config.get("c", 1.0), beta=config.get("beta", BETA),
        lambda_min=config.get("lambda_min", LAMBDA_MIN),
        lambda_max=config.get("lambda_max", LAMBDA_MAX),
        variant=config.get("variant", "conservative"),
        decouple_lr=config.get("decouple_lr", False))

    # LR scheduler
    base_opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    milestones = config.get("milestones", MILESTONES)
    gamma = config.get("lr_gamma", LR_GAMMA)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        base_opt, milestones=milestones, gamma=gamma)

    loss_fn = nn.CrossEntropyLoss()
    epoch_log_path = output_dir / "epoch_metrics.jsonl"
    step_log_path = output_dir / "step_metrics.jsonl"

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    all_epoch_records = []
    best_test_acc = 0.0
    start_time = time.time()

    for epoch in range(config["epochs"]):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        step_logs = []
        global_step = epoch * len(train_loader)

        # Handle stagewise WD
        if config["method"] == "stagewise_wd":
            progress = epoch / config["epochs"]
            if progress < 0.5:
                current_wd = config["weight_decay"]
            elif progress < 0.8:
                current_wd = config["weight_decay"] / 10.0
            else:
                current_wd = config["weight_decay"] / 100.0
            for group in base_opt.param_groups:
                group["weight_decay"] = current_wd

        # Norm-matched WD epoch tracking
        if hasattr(optimizer, "set_epoch"):
            optimizer.set_epoch(epoch)

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

            if batch_idx % STEP_LOG_INTERVAL == 0:
                cs = global_step + batch_idx
                sm = {"step": cs, "epoch": epoch, "batch_idx": batch_idx,
                      "loss": round(loss.item(), 6)}
                if hasattr(optimizer, "get_metrics"):
                    m = optimizer.get_metrics()
                    for k in ("lambda_t", "delta_hat_t", "ema_delta", "lr"):
                        sm[k] = round(m.get(k, 0.0), 8)
                else:
                    sm["lambda_t"] = base_opt.param_groups[0].get("weight_decay", 0.0)
                    sm["lr"] = round(base_opt.param_groups[0]["lr"], 8)
                sm["grad_norm"] = round(get_grad_norm(model), 6)
                step_logs.append(sm)

        scheduler.step()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        gen_gap = train_acc - test_acc

        if hasattr(optimizer, "get_metrics"):
            om = optimizer.get_metrics()
            lambda_t = om.get("lambda_t", 0.0)
            delta_hat_t = om.get("delta_hat_t", 0.0)
            ema_delta = om.get("ema_delta", 0.0)
            wn = om.get("weight_norm", get_weight_norm(model))
            lr_now = om.get("lr", 0.0)
        else:
            lambda_t = base_opt.param_groups[0].get("weight_decay", 0.0)
            delta_hat_t = 0.0
            ema_delta = 0.0
            wn = get_weight_norm(model)
            lr_now = base_opt.param_groups[0]["lr"]

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "test_loss": round(test_loss, 6),
            "gen_gap": round(gen_gap, 4),
            "weight_norm": round(wn, 4),
            "mean_lambda_t": round(lambda_t, 8),
            "mean_delta_hat_t": round(delta_hat_t, 6),
            "ema_delta": round(ema_delta, 6),
            "lr": round(lr_now, 8),
            "epoch_time_sec": round(epoch_time, 2),
        }
        all_epoch_records.append(rec)

        with open(epoch_log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        with open(step_log_path, "a") as f:
            for sl in step_logs:
                f.write(json.dumps(sl) + "\n")

        best_test_acc = max(best_test_acc, test_acc)

        # Checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            ckpt = {"epoch": epoch, "model_state_dict": model.state_dict(),
                    "best_test_acc": best_test_acc}
            torch.save(ckpt, output_dir / f"ckpt_ep{epoch+1}.pt")

        if epoch % 20 == 0 or epoch == config["epochs"] - 1:
            print(f"  Epoch {epoch:3d}/{config['epochs']}: "
                  f"loss={train_loss:.4f} train={train_acc:.1f}% "
                  f"test={test_acc:.2f}% gap={gen_gap:.2f}% "
                  f"wn={wn:.2f} lr={lr_now:.5f} λ={lambda_t:.6f}")

    total_time = time.time() - start_time
    summary = {
        "config": config,
        "best_test_acc": round(best_test_acc, 4),
        "final_test_acc": round(all_epoch_records[-1]["test_acc"], 4),
        "final_train_acc": round(all_epoch_records[-1]["train_acc"], 4),
        "final_gen_gap": round(all_epoch_records[-1]["gen_gap"], 4),
        "final_weight_norm": round(all_epoch_records[-1]["weight_norm"], 4),
        "total_time_sec": round(total_time, 2),
        "epochs_completed": config["epochs"],
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Done! Best: {best_test_acc:.2f}%, Time: {total_time:.1f}s")
    return summary


# ── Task definitions ──

def task_tier0_diagnostic(results_dir, gpu_id, data_dir):
    """Tier0: alignment proxy reliability, 4 WD values × 200 epochs."""
    base = Path(results_dir) / "tier0_diagnostic"
    configs = []
    for wd in [1e-4, 5e-4, 1e-3, 3e-3]:
        configs.append({
            "name": f"tier0_fixed_wd_{wd}",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "fixed_wd", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": wd, "seed": SEED,
            "output_dir": str(base / f"wd_{wd}"),
        })
    return configs, base


def task_tier1_fixed_wd_grid(results_dir, gpu_id, data_dir):
    """Tier1: 5 fixed WD values + no-WD baseline."""
    base = Path(results_dir) / "tier1_fixed_wd_grid"
    configs = []
    # No-WD baseline
    configs.append({
        "name": "no_wd",
        "arch": "resnet20", "dataset": "cifar10",
        "method": "no_wd", "epochs": EPOCHS,
        "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
        "weight_decay": 0.0, "seed": SEED,
        "output_dir": str(base / "no_wd"),
    })
    # Fixed WD grid
    for wd in [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]:
        configs.append({
            "name": f"fixed_wd_{wd}",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "fixed_wd", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": wd, "seed": SEED,
            "output_dir": str(base / f"wd_{wd}"),
        })
    return configs, base


def task_tier1_stagewise_cwd(results_dir, gpu_id, data_dir):
    """Tier1: stagewise WD and CWD baselines."""
    base = Path(results_dir) / "tier1_stagewise_cwd"
    configs = [
        {
            "name": "stagewise_wd",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "stagewise_wd", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4, "seed": SEED,
            "output_dir": str(base / "stagewise_wd"),
        },
        {
            "name": "cwd",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "cwd", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4, "seed": SEED,
            "output_dir": str(base / "cwd"),
        },
    ]
    return configs, base


def task_tier1_aadwd_variants(results_dir, gpu_id, data_dir):
    """Tier1: 3 AADWD variants with pilot-corrected hyperparams."""
    base = Path(results_dir) / "tier1_aadwd_variants"
    # NOTE: alignment delta is O(10^{-3}) in practice. c must compensate:
    #   conservative: lambda ~ c * lr * 1.0  → c ≈ 0.005 for lambda ~ 5e-4 at lr=0.1
    #   aggressive:   lambda ~ c * lr * 0.001 → c ≈ 5.0 for lambda ~ 5e-4 at lr=0.1
    #   square:       lambda ~ c * lr^2 * 1.0 → c ≈ 0.05 for lambda ~ 5e-4 at lr=0.1
    configs = [
        {
            "name": "aadwd_conservative",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_conservative", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 0.005,        # lambda ~ c*lr*(1-delta) ~ c*lr ~ 5e-4
            "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": LAMBDA_MAX,
            "variant": "conservative", "seed": SEED,
            "output_dir": str(base / "conservative"),
        },
        {
            "name": "aadwd_aggressive",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_aggressive", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 2.5,          # lambda ~ c*lr*delta ~ 2.5*0.1*0.002 ≈ 5e-4
            "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": 0.05,
            "variant": "aggressive", "seed": SEED,
            "output_dir": str(base / "aggressive"),
        },
        {
            "name": "aadwd_square",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_square", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 0.05,         # lambda ~ c*lr^2*(1-delta) ~ c*lr^2 ~ 5e-4
            "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": LAMBDA_MAX,
            "variant": "square", "seed": SEED,
            "output_dir": str(base / "square"),
        },
    ]
    return configs, base


def task_tier2_cross_arch(results_dir, gpu_id, data_dir):
    """Tier2: cross-architecture (ResNet20/CIFAR-100 + VGG16/CIFAR-10)."""
    base = Path(results_dir) / "tier2_cross_arch"
    settings = [
        ("resnet20", "cifar100", [80, 120], 0.1),
        ("vgg16_bn", "cifar10", [80, 120], 0.1),
    ]
    methods = [
        ("aadwd_conservative", {"c": 0.005, "beta": BETA,
         "lambda_min": LAMBDA_MIN, "lambda_max": LAMBDA_MAX,
         "variant": "conservative"}),
        ("aadwd_aggressive", {"c": 2.5, "beta": BETA,
         "lambda_min": LAMBDA_MIN, "lambda_max": 0.05,
         "variant": "aggressive"}),
        ("fixed_wd", {}),
        ("cwd", {}),
        ("no_wd", {}),
    ]
    configs = []
    for arch, ds, ms, g in settings:
        nc = 100 if ds == "cifar100" else 10
        for method, extra in methods:
            wd = 5e-4 if method != "no_wd" else 0.0
            name = f"{arch}_{ds}_{method}"
            cfg = {
                "name": name,
                "arch": arch, "dataset": ds,
                "method": method, "epochs": EPOCHS,
                "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
                "weight_decay": wd,
                "milestones": ms, "lr_gamma": g,
                "seed": SEED,
                "output_dir": str(base / name),
            }
            cfg.update(extra)
            configs.append(cfg)
    return configs, base


def task_tier2_hyperparam_sensitivity(results_dir, gpu_id, data_dir):
    """Tier2: c sweep + beta sweep for AADWD-aggressive."""
    base = Path(results_dir) / "tier2_hyperparam_sensitivity"
    configs = []
    # c sweep with fixed beta=0.999 (aggressive: lambda ~ c*lr*delta ~ c*0.1*0.002)
    for c_val in [0.5, 1.0, 2.5, 5.0, 10.0]:
        configs.append({
            "name": f"aadwd_agg_c{c_val}",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_aggressive", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": c_val, "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": 0.05,
            "variant": "aggressive", "seed": SEED,
            "output_dir": str(base / f"c_{c_val}"),
        })
    # beta sweep with c=2.5 (calibrated aggressive center value)
    for beta_val in [0.9, 0.99, 0.999, 0.9999]:
        configs.append({
            "name": f"aadwd_agg_beta{beta_val}",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_aggressive", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 2.5, "beta": beta_val,
            "lambda_min": LAMBDA_MIN, "lambda_max": 0.05,
            "variant": "aggressive", "seed": SEED,
            "output_dir": str(base / f"beta_{beta_val}"),
        })
    return configs, base


def task_tier2_ablations(results_dir, gpu_id, data_dir):
    """Tier2: ablation controls."""
    base = Path(results_dir) / "tier2_ablations"
    configs = [
        {
            "name": "aadwd_conservative_ref",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_conservative", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 0.005, "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": LAMBDA_MAX,
            "variant": "conservative", "seed": SEED,
            "output_dir": str(base / "aadwd_conservative_ref"),
        },
        {
            "name": "aadwd_aggressive_ref",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_aggressive", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 2.5, "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": 0.05,
            "variant": "aggressive", "seed": SEED,
            "output_dir": str(base / "aadwd_aggressive_ref"),
        },
        {
            "name": "random_dynamic_wd",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "random_dynamic_wd", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 0.005,  # same c as conservative for fair ablation
            "lambda_min": LAMBDA_MIN, "lambda_max": LAMBDA_MAX,
            "seed": SEED,
            "output_dir": str(base / "random_dynamic_wd"),
        },
        {
            "name": "norm_matched_wd",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "norm_matched_wd", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "seed": SEED,
            "output_dir": str(base / "norm_matched_wd"),
        },
        {
            "name": "equiv_cumulative_wd",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "equiv_cumulative_wd", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,  # will be overridden by mean(lambda_t) from AADWD
            "seed": SEED,
            "output_dir": str(base / "equiv_cumulative_wd"),
        },
    ]
    return configs, base


def task_tier2_decoupled(results_dir, gpu_id, data_dir):
    """Tier2: LR-decoupled AADWD variants.

    Key insight from Tier 1: coupling lambda to gamma_t causes WD to vanish
    when LR drops at milestones. Decoupled variants remove this dependency.

    Calibration (decoupled, no gamma_t):
      conservative: lambda ~ c * (1-delta) ~ c * 1.0 → c ≈ 5e-4
      aggressive:   lambda ~ c * delta ~ c * 0.002 → c ≈ 0.25
    """
    base = Path(results_dir) / "tier2_decoupled"
    configs = [
        {
            "name": "aadwd_conservative_decoupled",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_conservative", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 5e-4,  # lambda ~ c*(1-delta) ~ c ~ 5e-4
            "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": LAMBDA_MAX,
            "variant": "conservative",
            "decouple_lr": True,
            "seed": SEED,
            "output_dir": str(base / "conservative_decoupled"),
        },
        {
            "name": "aadwd_aggressive_decoupled",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_aggressive", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 0.25,  # lambda ~ c*delta ~ 0.25*0.002 = 5e-4
            "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": 0.01,
            "variant": "aggressive",
            "decouple_lr": True,
            "seed": SEED,
            "output_dir": str(base / "aggressive_decoupled"),
        },
        # Reference: coupled versions for direct comparison
        {
            "name": "aadwd_conservative_coupled_ref",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_conservative", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 0.005, "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": LAMBDA_MAX,
            "variant": "conservative",
            "decouple_lr": False,
            "seed": SEED,
            "output_dir": str(base / "conservative_coupled_ref"),
        },
        {
            "name": "aadwd_aggressive_coupled_ref",
            "arch": "resnet20", "dataset": "cifar10",
            "method": "aadwd_aggressive", "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "lr": LR, "momentum": MOMENTUM,
            "weight_decay": 5e-4,
            "c": 2.5, "beta": BETA,
            "lambda_min": LAMBDA_MIN, "lambda_max": 0.05,
            "variant": "aggressive",
            "decouple_lr": False,
            "seed": SEED,
            "output_dir": str(base / "aggressive_coupled_ref"),
        },
    ]
    return configs, base


TASK_MAP = {
    "tier0_diagnostic": task_tier0_diagnostic,
    "tier1_fixed_wd_grid": task_tier1_fixed_wd_grid,
    "tier1_stagewise_cwd": task_tier1_stagewise_cwd,
    "tier1_aadwd_variants": task_tier1_aadwd_variants,
    "tier2_cross_arch": task_tier2_cross_arch,
    "tier2_hyperparam_sensitivity": task_tier2_hyperparam_sensitivity,
    "tier2_ablations": task_tier2_ablations,
    "tier2_decoupled": task_tier2_decoupled,
}


def write_progress(progress_file, task_id, current, total, status="running"):
    """Write progress JSON for monitoring daemon."""
    prog = {"task_id": task_id, "current": current, "total": total,
            "status": status, "pct": round(100 * current / max(total, 1), 1),
            "ts": time.time()}
    with open(progress_file, "w") as f:
        json.dump(prog, f)


def main():
    parser = argparse.ArgumentParser(description="FULL 200-epoch experiment runner")
    parser.add_argument("--task", required=True, choices=list(TASK_MAP.keys()))
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = str(Path(args.results_dir).parent / "data")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\n{'#'*60}")
    print(f"# FULL EXPERIMENT: {args.task}")
    print(f"# GPU: {args.gpu}, Device: {device}")
    print(f"# Results: {args.results_dir}")
    print(f"{'#'*60}")

    # Progress tracking — markers go in results_dir alongside task output dirs
    progress_file = str(Path(args.results_dir) / f"{args.task}_PROGRESS.json")
    done_file = str(Path(args.results_dir) / f"{args.task}_DONE")
    pid_file = str(Path(args.results_dir) / f"{args.task}.pid")

    # Write PID
    with open(pid_file, "w") as f:
        f.write(str(os.getpid()))

    task_fn = TASK_MAP[args.task]
    configs, base_dir = task_fn(args.results_dir, args.gpu, args.data_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}
    overall_start = time.time()

    for i, cfg in enumerate(configs):
        write_progress(progress_file, args.task, i, len(configs))
        summary = train_single(cfg, device, args.data_dir)
        all_summaries[cfg["name"]] = summary

    write_progress(progress_file, args.task, len(configs), len(configs), "completed")

    # Aggregate summary
    agg = {
        "task_id": args.task,
        "experiments": all_summaries,
        "total_wall_time_sec": round(time.time() - overall_start, 2),
        "total_experiments": len(configs),
    }
    with open(base_dir / "aggregate_summary.json", "w") as f:
        json.dump(agg, f, indent=2)

    # Write DONE marker
    done_data = {"task_id": args.task, "success": True,
                 "total_time_sec": agg["total_wall_time_sec"],
                 "experiments": len(configs)}
    with open(done_file, "w") as f:
        json.dump(done_data, f)

    print(f"\n{'='*60}")
    print(f"FULL {args.task} COMPLETE — {len(configs)} experiments")
    print(f"Total wall time: {agg['total_wall_time_sec']:.1f}s")
    for name, s in all_summaries.items():
        print(f"  {name}: best={s['best_test_acc']:.2f}% final={s['final_test_acc']:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
