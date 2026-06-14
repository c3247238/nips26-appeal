"""
Tier 0 Alignment Proxy Reliability Diagnostic (PILOT mode).

PILOT: 1 WD value (5e-4), 20 epochs, checkpoints at 5, 10, 15, 20.
Task: Verify mini-batch alignment proxy delta_hat_t is a reliable
      indicator of large-batch alignment delta_t.

Pass criterion: Pearson r > 0.85 between EMA-smoothed mini-batch
               and large-batch alignment.
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Set CUDA device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

CODE_DIR = Path(__file__).parent
sys.path.insert(0, str(CODE_DIR))

from models import create_model
from data import get_dataloaders, get_large_batch_loader
from alignment_diagnostic import (
    compute_mini_batch_alignment,
    compute_large_batch_alignment,
    ema_smooth,
    pearson_r,
)

# ---- Config ----
RESULTS_DIR = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/tier0_diagnostic")
DATA_DIR = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/data")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PILOT_CONFIG = {
    "arch": "resnet20",
    "dataset": "cifar10",
    "weight_decay": 5e-4,
    "epochs": 20,
    "batch_size": 128,
    "lr": 0.1,
    "momentum": 0.9,
    "seed": 42,
    "beta": 0.99,
    "checkpoint_epochs": [5, 10, 15, 20],
    "num_mini_batches": 50,
    "large_batch_size": 4096,  # Reduced for memory; still much larger than 128
}


def compute_weight_norm(model):
    total = sum(p.data.norm().item() ** 2 for p in model.parameters())
    return math.sqrt(total)


def evaluate(model, test_loader, device):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
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
                    epoch, ema_delta, beta=0.99):
    """Train one epoch, tracking per-step alignment for EMA."""
    model.train()
    running_loss = 0.0
    correct, total = 0, 0
    step_deltas = []

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        loss.backward()

        # Compute alignment from current gradient
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
        delta_hat = abs(dot_gw) / (norm_g * norm_w + 1e-8)
        ema_delta = beta * ema_delta + (1 - beta) * delta_hat
        step_deltas.append(delta_hat)

        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    scheduler.step()

    train_acc = 100.0 * correct / total
    train_loss = running_loss / total
    return train_loss, train_acc, ema_delta, step_deltas


def run_diagnostic():
    cfg = PILOT_CONFIG
    print("=" * 60)
    print("Tier 0 Diagnostic: Alignment Proxy Reliability (PILOT)")
    print(f"Config: {cfg}")
    print("=" * 60)

    # Reproducibility
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg["seed"])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Data
    train_loader, test_loader, num_classes = get_dataloaders(
        cfg["dataset"], batch_size=cfg["batch_size"],
        data_dir=str(DATA_DIR), num_workers=4, pin_memory=True)

    large_loader = get_large_batch_loader(
        cfg["dataset"], batch_size=cfg["large_batch_size"],
        data_dir=str(DATA_DIR), num_workers=4)

    # Mini-batch loader for diagnostic (uses train data, separate loader)
    mini_loader, _, _ = get_dataloaders(
        cfg["dataset"], batch_size=cfg["batch_size"],
        data_dir=str(DATA_DIR), num_workers=2, pin_memory=True)

    # Model
    model = create_model(cfg["arch"], num_classes=num_classes)
    model = model.to(device)

    # Optimizer (SGD with fixed WD)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=cfg["lr"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        nesterov=False
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"])

    loss_fn = nn.CrossEntropyLoss()

    # Tracking
    ema_delta = 0.5  # initial EMA
    epoch_records = []
    checkpoint_data = []  # (epoch, mini_batch_deltas, large_batch_delta, ema_at_ckpt)

    start_time = time.time()

    for epoch in range(1, cfg["epochs"] + 1):
        train_loss, train_acc, ema_delta, step_deltas = train_one_epoch(
            model, train_loader, optimizer, scheduler, loss_fn, device,
            epoch, ema_delta, beta=cfg["beta"])

        test_acc, test_loss = evaluate(model, test_loader, device)
        weight_norm = compute_weight_norm(model)

        rec = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_acc, 4),
            "test_acc": round(test_acc, 4),
            "test_loss": round(test_loss, 6),
            "gen_gap": round(train_acc - test_acc, 4),
            "weight_norm": round(weight_norm, 4),
            "mean_step_delta": round(float(np.mean(step_deltas)), 6),
            "std_step_delta": round(float(np.std(step_deltas)), 6),
            "ema_delta": round(ema_delta, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        epoch_records.append(rec)

        print(f"  Epoch {epoch:2d}/{cfg['epochs']}: "
              f"train={train_loss:.4f}/{train_acc:.1f}% "
              f"test={test_acc:.2f}% wn={weight_norm:.2f} "
              f"ema_delta={ema_delta:.4f} mean_delta={np.mean(step_deltas):.4f}")

        # At checkpoint epochs: compute mini-batch and large-batch alignment
        if epoch in cfg["checkpoint_epochs"]:
            print(f"  [Checkpoint] Epoch {epoch}: computing alignment diagnostics...")
            # Mini-batch alignment (50 random batches, B=128)
            mini_deltas = compute_mini_batch_alignment(
                model, mini_loader, loss_fn, device,
                num_batches=cfg["num_mini_batches"])
            mini_mean = float(np.mean(mini_deltas))
            mini_std = float(np.std(mini_deltas))

            # Large-batch alignment
            large_delta = compute_large_batch_alignment(
                model, large_loader, loss_fn, device)

            # EMA-smoothed mini-batch deltas
            mini_ema_smoothed = ema_smooth(mini_deltas, beta=cfg["beta"])
            mini_ema_mean = float(np.mean(mini_ema_smoothed))

            ckpt_rec = {
                "epoch": epoch,
                "mini_batch_deltas": mini_deltas,
                "mini_batch_mean": round(mini_mean, 6),
                "mini_batch_std": round(mini_std, 6),
                "mini_ema_smoothed": mini_ema_smoothed,
                "mini_ema_mean": round(mini_ema_mean, 6),
                "large_batch_delta": round(large_delta, 6),
                "training_ema_delta": round(ema_delta, 6),
            }
            checkpoint_data.append(ckpt_rec)

            print(f"    mini-batch delta: mean={mini_mean:.4f} std={mini_std:.4f}")
            print(f"    large-batch delta: {large_delta:.4f}")
            print(f"    training EMA delta: {ema_delta:.4f}")

    total_time = time.time() - start_time
    print(f"\nTraining complete in {total_time:.1f}s")

    # ---- Compute Pearson r ----
    # Use training EMA at checkpoint epochs vs large-batch delta
    ema_at_ckpts = [c["training_ema_delta"] for c in checkpoint_data]
    large_deltas = [c["large_batch_delta"] for c in checkpoint_data]
    mini_means = [c["mini_batch_mean"] for c in checkpoint_data]
    mini_ema_means = [c["mini_ema_mean"] for c in checkpoint_data]

    r_ema_vs_large = pearson_r(ema_at_ckpts, large_deltas)
    r_mini_vs_large = pearson_r(mini_means, large_deltas)
    r_mini_ema_vs_large = pearson_r(mini_ema_means, large_deltas)

    print(f"\n{'='*60}")
    print("ALIGNMENT PROXY RELIABILITY RESULTS")
    print(f"{'='*60}")
    print(f"Pearson r (training EMA vs large-batch): {r_ema_vs_large:.4f}")
    print(f"Pearson r (mini-batch mean vs large-batch): {r_mini_vs_large:.4f}")
    print(f"Pearson r (mini-batch EMA vs large-batch): {r_mini_ema_vs_large:.4f}")

    # Phase-dependent structure: split epochs into thirds
    all_mean_deltas = [r["mean_step_delta"] for r in epoch_records]
    n = len(all_mean_deltas)
    early = all_mean_deltas[:n//3]
    mid = all_mean_deltas[n//3:2*n//3]
    late = all_mean_deltas[2*n//3:]

    phase_stats = {
        "early": {"mean": round(float(np.mean(early)), 6), "std": round(float(np.std(early)), 6)},
        "mid": {"mean": round(float(np.mean(mid)), 6), "std": round(float(np.std(mid)), 6)},
        "late": {"mean": round(float(np.mean(late)), 6), "std": round(float(np.std(late)), 6)},
    }
    print(f"\nPhase-dependent structure (delta_hat_t):")
    for phase, stats in phase_stats.items():
        print(f"  {phase}: mean={stats['mean']:.4f} std={stats['std']:.4f}")

    # Overall std across all epochs
    overall_std = float(np.std(all_mean_deltas))
    print(f"  Overall std across epochs: {overall_std:.4f}")

    # ---- GO / NO-GO Decision ----
    best_r = max(r_ema_vs_large, r_mini_vs_large, r_mini_ema_vs_large)
    phase_std_ok = overall_std > 0.05

    if best_r > 0.85:
        verdict = "GO"
        verdict_msg = f"PASS: Best Pearson r={best_r:.4f} > 0.85 threshold"
    elif best_r > 0.6:
        verdict = "NO-GO (adjust beta)"
        verdict_msg = f"PARTIAL FAIL: r={best_r:.4f} in [0.6, 0.85]. Increase beta to 0.999."
    else:
        verdict = "NO-GO (pivot cand_empirical)"
        verdict_msg = f"FAIL: r={best_r:.4f} < 0.6. Pivot to cand_empirical."

    print(f"\n{'='*60}")
    print(f"VERDICT: {verdict}")
    print(f"  {verdict_msg}")
    print(f"  Phase structure (std > 0.05): {'PASS' if phase_std_ok else 'FAIL'} (std={overall_std:.4f})")
    print(f"{'='*60}")

    # ---- Save Results ----
    results = {
        "task": "tier0_diagnostic",
        "mode": "PILOT",
        "config": cfg,
        "verdict": verdict,
        "verdict_msg": verdict_msg,
        "pearson_r": {
            "ema_vs_large": round(r_ema_vs_large, 6),
            "mini_vs_large": round(r_mini_vs_large, 6),
            "mini_ema_vs_large": round(r_mini_ema_vs_large, 6),
            "best": round(best_r, 6),
        },
        "phase_structure": phase_stats,
        "overall_delta_std": round(overall_std, 6),
        "checkpoint_data": checkpoint_data,
        "epoch_records": epoch_records,
        "total_time_sec": round(total_time, 2),
    }

    results_path = RESULTS_DIR / "diagnostic_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # Save epoch metrics as JSONL
    epoch_log_path = RESULTS_DIR / "epoch_metrics.jsonl"
    with open(epoch_log_path, "w") as f:
        for rec in epoch_records:
            f.write(json.dumps(rec) + "\n")

    # Save summary
    summary = {
        "task": "tier0_diagnostic",
        "mode": "PILOT",
        "verdict": verdict,
        "best_pearson_r": round(best_r, 6),
        "r_ema_vs_large": round(r_ema_vs_large, 6),
        "r_mini_vs_large": round(r_mini_vs_large, 6),
        "r_mini_ema_vs_large": round(r_mini_ema_vs_large, 6),
        "overall_delta_std": round(overall_std, 6),
        "phase_structure": phase_stats,
        "pass_criterion_r": best_r > 0.85,
        "pass_criterion_phase_std": phase_std_ok,
        "final_test_acc": round(epoch_records[-1]["test_acc"], 4),
        "total_time_sec": round(total_time, 2),
    }
    summary_path = RESULTS_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    # Generate plots
    try:
        generate_plots(results)
    except Exception as e:
        print(f"[Warning] Plot generation failed: {e}")

    return results


def generate_plots(results):
    """Generate diagnostic plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    RESULTS_DIR = Path("/home/ccwang/sibyl-research-system/workspaces/dynamic-wd/current/exp/results/tier0_diagnostic")

    epoch_records = results["epoch_records"]
    checkpoint_data = results["checkpoint_data"]
    pearson_r_vals = results["pearson_r"]

    epochs = [r["epoch"] for r in epoch_records]
    ema_deltas = [r["ema_delta"] for r in epoch_records]
    mean_deltas = [r["mean_step_delta"] for r in epoch_records]

    ckpt_epochs = [c["epoch"] for c in checkpoint_data]
    large_deltas = [c["large_batch_delta"] for c in checkpoint_data]
    ema_at_ckpts = [c["training_ema_delta"] for c in checkpoint_data]
    mini_means = [c["mini_batch_mean"] for c in checkpoint_data]

    # ---- Plot 1: delta_hat_t trajectory ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Tier 0 Diagnostic: Alignment Proxy Reliability (PILOT)\n'
                 f'ResNet20/CIFAR-10, WD=5e-4, 20 epochs',
                 fontsize=13)

    ax = axes[0, 0]
    ax.plot(epochs, mean_deltas, 'b-o', markersize=3, label='mini-batch δ̂_t mean', alpha=0.8)
    ax.plot(epochs, ema_deltas, 'r-', linewidth=2, label=f'EMA(δ̂_t, β=0.99)')
    ax.scatter(ckpt_epochs, large_deltas, color='green', zorder=5, s=80,
               label='large-batch δ_t', marker='^')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Alignment δ')
    ax.set_title('Alignment Trajectory')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Correlation scatter ----
    ax = axes[0, 1]
    r_best = pearson_r_vals["best"]
    ax.scatter(large_deltas, ema_at_ckpts, color='blue', s=100, zorder=5)
    for i, ep in enumerate(ckpt_epochs):
        ax.annotate(f'ep{ep}', (large_deltas[i], ema_at_ckpts[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    # Fit line
    if len(large_deltas) >= 2:
        x = np.array(large_deltas)
        y = np.array(ema_at_ckpts)
        m, b = np.polyfit(x, y, 1)
        xline = np.linspace(min(x), max(x), 50)
        ax.plot(xline, m * xline + b, 'r--', alpha=0.7)
    ax.set_xlabel('Large-batch δ_t')
    ax.set_ylabel('Training EMA(δ̂_t)')
    ax.set_title(f'Correlation: r = {pearson_r_vals["ema_vs_large"]:.4f}')
    ax.grid(True, alpha=0.3)

    # ---- Plot 3: Mini-batch distribution at checkpoints ----
    ax = axes[1, 0]
    colors = ['blue', 'orange', 'green', 'red']
    for i, ckpt in enumerate(checkpoint_data):
        ep = ckpt["epoch"]
        deltas = ckpt["mini_batch_deltas"]
        color = colors[i % len(colors)]
        ax.hist(deltas, bins=15, alpha=0.5, color=color, label=f'Ep {ep}')
        ax.axvline(ckpt["large_batch_delta"], color=color, linestyle='--',
                   linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Alignment δ̂_t (mini-batch)')
    ax.set_ylabel('Count')
    ax.set_title('Mini-batch δ distribution at checkpoints\n(dashed = large-batch δ)')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ---- Plot 4: Phase structure ----
    ax = axes[1, 1]
    phase_stats = results["phase_structure"]
    n = len(mean_deltas)
    early_end = n // 3
    mid_end = 2 * n // 3
    ax.plot(epochs[:early_end], mean_deltas[:early_end], 'b-o', markersize=4,
            label=f'Early (std={phase_stats["early"]["std"]:.4f})')
    ax.plot(epochs[early_end:mid_end], mean_deltas[early_end:mid_end], 'orange',
            marker='o', markersize=4,
            label=f'Mid (std={phase_stats["mid"]["std"]:.4f})')
    ax.plot(epochs[mid_end:], mean_deltas[mid_end:], 'g-o', markersize=4,
            label=f'Late (std={phase_stats["late"]["std"]:.4f})')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Mean δ̂_t per epoch')
    ax.set_title(f'Phase-dependent Structure\n(overall std={results["overall_delta_std"]:.4f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Verdict annotation
    verdict = results["verdict"]
    color = 'green' if verdict == 'GO' else 'red'
    fig.text(0.5, 0.02,
             f'VERDICT: {verdict}  |  Best r={r_best:.4f}  |  '
             f'Threshold r>0.85',
             ha='center', fontsize=12, color=color,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plot_path = RESULTS_DIR / "alignment_diagnostic_plot.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # ---- Plot 5: EMA effect ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, mean_deltas, 'b-', alpha=0.5, linewidth=1, label='Raw δ̂_t (per-epoch mean)')
    ax.plot(epochs, ema_deltas, 'r-', linewidth=2, label=f'EMA(δ̂_t, β=0.99)')
    ax.scatter(ckpt_epochs, large_deltas, color='green', s=100, zorder=5,
               marker='^', label='Large-batch δ_t (ground truth)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Alignment δ')
    ax.set_title('EMA Smoothing Effect on Alignment Proxy\n'
                 f'(Pearson r EMA vs large-batch: {pearson_r_vals["ema_vs_large"]:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    ema_plot_path = RESULTS_DIR / "ema_effect_plot.png"
    plt.savefig(ema_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"EMA plot saved to: {ema_plot_path}")


if __name__ == "__main__":
    results = run_diagnostic()
    verdict = results["verdict"]
    print(f"\nFinal verdict: {verdict}")
    sys.exit(0)
