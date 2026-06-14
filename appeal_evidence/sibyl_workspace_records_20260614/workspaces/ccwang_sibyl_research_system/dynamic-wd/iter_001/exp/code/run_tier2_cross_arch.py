"""
Tier2 Cross-Architecture PILOT experiment.

Tests AADWD-aggressive generalization on VGG16-BN / CIFAR-100.
Uses MultiStepLR milestones=[30,60,90], gamma=0.2 instead of CosineAnnealingLR.

Experiments:
  1. VGG16-BN / CIFAR-100, AADWD-aggressive (beta=0.999, c=0.01)
  2. VGG16-BN / CIFAR-100, Fixed WD (5e-4) baseline
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Add code directory to path
CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from models import create_model
from data import get_dataloaders
from optimizers import create_optimizer, AADWDOptimizer

RESULTS_BASE = Path(
    "/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/tier2_cross_arch"
)
DATA_DIR = str(
    Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/data")
)


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


def train_and_evaluate(config):
    """Run full training with MultiStepLR scheduler."""
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {config['name']}")
    print(f"  arch={config['arch']}, dataset={config['dataset']}")
    print(f"  method={config['method']}, epochs={config['epochs']}")
    print(f"  lr={config['lr']}, wd={config['weight_decay']}")
    if config["method"] == "aadwd_aggressive":
        print(f"  beta={config['beta']}, c={config['c']}")
    print(f"  Device: {device}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    # Data
    train_loader, test_loader, num_classes = get_dataloaders(
        config["dataset"],
        batch_size=config["batch_size"],
        data_dir=DATA_DIR,
    )

    # Model
    model = create_model(config["arch"], num_classes=num_classes)
    model = model.to(device)

    # Optimizer
    optimizer = create_optimizer(
        model,
        config["method"],
        lr=config["lr"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
        c=config.get("c", 1.0),
        beta=config.get("beta", 0.99),
        lambda_min=config.get("lambda_min", 1e-6),
        lambda_max=config.get("lambda_max", 0.01),
    )

    # MultiStepLR scheduler (as specified in task)
    base_opt = optimizer.optimizer if hasattr(optimizer, "optimizer") else optimizer
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        base_opt,
        milestones=config.get("milestones", [30, 60, 90]),
        gamma=config.get("gamma", 0.2),
    )

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

            # Per-step logging every 100 steps
            if batch_idx % 100 == 0:
                current_step = global_step + batch_idx
                step_metrics = {
                    "step": current_step,
                    "epoch": epoch,
                    "batch_idx": batch_idx,
                    "loss": round(loss.item(), 6),
                }
                if hasattr(optimizer, "get_metrics"):
                    m = optimizer.get_metrics()
                    step_metrics["lambda_t"] = round(m.get("lambda_t", 0.0), 8)
                    step_metrics["delta_hat_t"] = round(m.get("delta_hat_t", 0.0), 6)
                    step_metrics["ema_delta"] = round(m.get("ema_delta", 0.0), 6)
                    step_metrics["lr"] = round(m.get("lr", 0.0), 8)
                else:
                    step_metrics["lambda_t"] = config["weight_decay"]
                    step_metrics["lr"] = round(base_opt.param_groups[0]["lr"], 8)
                step_metrics["grad_norm"] = round(get_grad_norm(model), 6)
                step_logs.append(step_metrics)

        # Step scheduler after each epoch
        scheduler.step()

        train_acc = 100.0 * correct / total
        train_loss = running_loss / total
        test_acc, test_loss = evaluate(model, test_loader, device)
        epoch_time = time.time() - epoch_start
        gen_gap = train_acc - test_acc

        if hasattr(optimizer, "get_metrics"):
            opt_m = optimizer.get_metrics()
            lambda_t = opt_m.get("lambda_t", 0.0)
            delta_hat_t = opt_m.get("delta_hat_t", 0.0)
            ema_delta = opt_m.get("ema_delta", 0.0)
            wn = opt_m.get("weight_norm", get_weight_norm(model))
            lr_now = opt_m.get("lr", 0.0)
        else:
            lambda_t = config["weight_decay"]
            delta_hat_t = 0.0
            ema_delta = 0.0
            wn = get_weight_norm(model)
            lr_now = base_opt.param_groups[0]["lr"]

        epoch_record = {
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
        all_epoch_records.append(epoch_record)

        with open(epoch_log_path, "a") as f:
            f.write(json.dumps(epoch_record) + "\n")
        with open(step_log_path, "a") as f:
            for sl in step_logs:
                f.write(json.dumps(sl) + "\n")

        best_test_acc = max(best_test_acc, test_acc)

        if epoch % 5 == 0 or epoch == config["epochs"] - 1:
            print(
                f"  Epoch {epoch:3d}/{config['epochs']}: "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.1f}% "
                f"test_acc={test_acc:.2f}% gen_gap={gen_gap:.2f}% "
                f"wn={wn:.2f} lr={lr_now:.5f} lambda={lambda_t:.6f}"
            )

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

    print(f"\n  Done! Best test acc: {best_test_acc:.2f}%, Time: {total_time:.1f}s")
    return summary


def main():
    EXPERIMENTS = [
        {
            "name": "vgg16bn_cifar100_aadwd_aggressive",
            "arch": "vgg16_bn",
            "dataset": "cifar100",
            "method": "aadwd_aggressive",
            "epochs": 20,
            "batch_size": 128,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,  # fallback; AADWD uses dynamic
            "c": 0.01,
            "beta": 0.999,
            "lambda_min": 1e-6,
            "lambda_max": 0.01,
            "milestones": [30, 60, 90],
            "gamma": 0.2,
            "seed": 42,
            "output_dir": str(RESULTS_BASE / "vgg16bn_cifar100_aadwd_aggressive"),
        },
        {
            "name": "vgg16bn_cifar100_fixed_wd",
            "arch": "vgg16_bn",
            "dataset": "cifar100",
            "method": "fixed_wd",
            "epochs": 20,
            "batch_size": 128,
            "lr": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "milestones": [30, 60, 90],
            "gamma": 0.2,
            "seed": 42,
            "output_dir": str(RESULTS_BASE / "vgg16bn_cifar100_fixed_wd"),
        },
    ]

    all_summaries = {}
    overall_start = time.time()

    for cfg in EXPERIMENTS:
        summary = train_and_evaluate(cfg)
        all_summaries[cfg["name"]] = summary

    # Write combined summary
    combined = {
        "experiments": all_summaries,
        "total_wall_time_sec": round(time.time() - overall_start, 2),
    }
    with open(RESULTS_BASE / "combined_summary.json", "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 60)
    print("TIER2 CROSS-ARCH PILOT - ALL RESULTS")
    print("=" * 60)
    for name, s in all_summaries.items():
        print(
            f"  {name}: best_acc={s['best_test_acc']:.2f}%, "
            f"final_acc={s['final_test_acc']:.2f}%, "
            f"gen_gap={s['final_gen_gap']:.2f}%"
        )
    print("=" * 60)
    print(f"Combined summary: {RESULTS_BASE / 'combined_summary.json'}")


if __name__ == "__main__":
    main()
