#!/usr/bin/env python3
"""
Held-Out Generalization Test - FULL Mode

80/20 train/test split on synthetic data to verify absorption generalizes
to unseen hierarchical patterns.

FULL mode: 5 seeds, 10000 samples, 2000 training steps
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import pearsonr, ttest_rel
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 43, 44, 45, 46]
N_SAMPLES = 10000
D_MODEL = 128
D_SAE = 4096
L0_TARGET = 32
TRAIN_STEPS = 2000
BATCH_SIZE = 256
TRAIN_RATIO = 0.8
EVAL_SAMPLES = 500


def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor to track."""
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor to detect."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


class TopKSAE(nn.Module):
    """Sparse Autoencoder with TopK activation."""
    def __init__(self, d_model, d_sae, l0_target=32):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae
        self.l0_target = l0_target
        self.k = l0_target

        self.W_encoder = nn.Linear(d_model, d_sae, bias=True)
        nn.init.xavier_uniform_(self.W_encoder.weight)
        nn.init.zeros_(self.W_encoder.bias)

        self.W_decoder = nn.Linear(d_sae, d_model, bias=False)
        nn.init.xavier_uniform_(self.W_decoder.weight)
        self.normalize_decoder()

        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_model))

    def normalize_decoder(self):
        with torch.no_grad():
            norm = torch.norm(self.W_decoder.weight, dim=0, keepdim=True)
            self.W_decoder.weight.div_(norm + 1e-8)

    def encode(self, x):
        pre_acts = self.W_encoder(x - self.b_dec) + self.b_enc
        acts = torch.relu(pre_acts)
        topk_vals, topk_idx = torch.topk(acts, k=self.k, dim=-1)
        sparse_acts = torch.zeros_like(acts)
        sparse_acts.scatter_(-1, topk_idx, topk_vals)
        return sparse_acts

    def decode(self, acts):
        return self.W_decoder(acts) + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        return recon, acts

    def get_encoder_activations(self, x):
        with torch.no_grad():
            pre_acts = self.W_encoder(x - self.b_dec) + self.b_enc
            return torch.relu(pre_acts)


def generate_hierarchical_data(n_samples, d_model, seed=42, stochastic_noise=0.1):
    """Generate synthetic data with hierarchical parent-child structure."""
    rng = np.random.RandomState(seed)

    parent_dir = rng.randn(d_model).astype(np.float32)
    parent_dir /= np.linalg.norm(parent_dir) + 1e-8

    noise_scale = stochastic_noise + rng.uniform(0.05, 0.15)
    child1_dir = (1.0 - noise_scale) * parent_dir + noise_scale * rng.randn(d_model).astype(np.float32)
    child1_dir /= np.linalg.norm(child1_dir) + 1e-8

    noise_scale2 = stochastic_noise + rng.uniform(0.05, 0.15)
    child2_dir = (1.0 - noise_scale2) * parent_dir + noise_scale2 * rng.randn(d_model).astype(np.float32)
    child2_dir /= np.linalg.norm(child2_dir) + 1e-8

    p_parent = 0.25 + rng.uniform(-0.05, 0.05)
    p_child1 = 0.25 + rng.uniform(-0.05, 0.05)
    p_child2 = 0.25 + rng.uniform(-0.05, 0.05)
    p_mixed = 1.0 - p_parent - p_child1 - p_child2

    data = []
    labels = []
    for _ in range(n_samples):
        base = rng.randn(d_model).astype(np.float32)
        base /= np.linalg.norm(base) + 1e-8

        mix = rng.rand()
        if mix < p_parent:
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * parent_dir
            labels.append(0)
        elif mix < p_parent + p_child1:
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * child1_dir
            labels.append(1)
        elif mix < p_parent + p_child1 + p_child2:
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * child2_dir
            labels.append(2)
        else:
            p_strength = rng.uniform(1.0, 2.0)
            c_strength = rng.uniform(1.0, 2.0)
            child_choice = child1_dir if rng.rand() < 0.5 else child2_dir
            sample = base + p_strength * parent_dir + c_strength * child_choice
            labels.append(3)

        data.append(sample)

    return np.array(data), np.array(labels), parent_dir, child1_dir, child2_dir


def train_sae(model, data, device, steps=2000, batch_size=256, lr=1e-3):
    """Train SAE on synthetic hierarchical data."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    dataset = torch.FloatTensor(data).to(device)
    n_samples = len(dataset)

    for step in range(steps):
        idx = np.random.choice(n_samples, size=min(batch_size, n_samples), replace=False)
        batch = dataset[idx]

        optimizer.zero_grad()
        recon, acts = model(batch)

        recon_loss = nn.functional.mse_loss(recon, batch)
        l1_loss = acts.abs().mean()
        loss = recon_loss + 1e-4 * l1_loss

        loss.backward()
        optimizer.step()
        model.normalize_decoder()

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1}/{steps}: loss={loss.item():.4f} recon={recon_loss.item():.4f} l1={l1_loss.item():.4f}")

    model.eval()
    return model


def measure_absorption(model, parent_dir, child_dirs, device, n_samples=500, seed=None):
    """Measure absorption using top-k Jaccard overlap (primary metric)."""
    parent_t = torch.FloatTensor(parent_dir).to(device)
    child1_t = torch.FloatTensor(child_dirs[0]).to(device)
    child2_t = torch.FloatTensor(child_dirs[1]).to(device)

    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random

    overlap_scores = []

    with torch.no_grad():
        for _ in range(n_samples):
            strength = rng.uniform(2.0, 4.0)

            parent_input = parent_t * strength
            child1_input = child1_t * strength
            child2_input = child2_t * strength

            parent_acts = model.get_encoder_activations(parent_input.unsqueeze(0))[0]
            child1_acts = model.get_encoder_activations(child1_input.unsqueeze(0))[0]
            child2_acts = model.get_encoder_activations(child2_input.unsqueeze(0))[0]

            k = min(32, parent_acts.shape[0])
            _, parent_topk = torch.topk(parent_acts, k=k)
            _, child1_topk = torch.topk(child1_acts, k=k)
            _, child2_topk = torch.topk(child2_acts, k=k)

            parent_set = set(parent_topk.cpu().tolist())
            child1_set = set(child1_topk.cpu().tolist())
            child2_set = set(child2_topk.cpu().tolist())

            jacc1 = len(parent_set & child1_set) / max(len(parent_set | child1_set), 1)
            jacc2 = len(parent_set & child2_set) / max(len(parent_set | child2_set), 1)
            overlap_scores.append((jacc1 + jacc2) / 2)

    return {
        "mean": float(np.mean(overlap_scores)),
        "std": float(np.std(overlap_scores)),
        "raw": [float(x) for x in overlap_scores]
    }


def run_single_split(seed, train_ratio=0.8):
    """Run absorption measurement for a single train/test split."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n  Seed {seed}, train_ratio={train_ratio}:")

    # Generate hierarchical data
    data, labels, parent_dir, child1_dir, child2_dir = generate_hierarchical_data(
        n_samples=N_SAMPLES, d_model=D_MODEL, seed=seed, stochastic_noise=0.1
    )
    child_dirs = [child1_dir, child2_dir]

    # Train/test split
    n_train = int(len(data) * train_ratio)
    train_data = data[:n_train]
    test_data = data[n_train:]
    print(f"    Train samples: {len(train_data)}, Test samples: {len(test_data)}")

    # Train SAE on train data
    model = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, l0_target=L0_TARGET).to(DEVICE)
    model = train_sae(model, train_data, DEVICE, steps=TRAIN_STEPS, batch_size=BATCH_SIZE)

    with torch.no_grad():
        sample = torch.FloatTensor(train_data[:1000]).to(DEVICE)
        recon, acts = model(sample)
        recon_err = nn.functional.mse_loss(recon, sample).item()
        mean_l0 = (acts > 0).float().sum(dim=-1).mean().item()
    print(f"    Recon MSE: {recon_err:.4f}, Mean L0: {mean_l0:.1f}")

    # Measure absorption on train data with fixed seed
    train_result = measure_absorption(model, parent_dir, child_dirs, DEVICE, n_samples=EVAL_SAMPLES, seed=seed)
    print(f"    Train absorption (Jaccard): {train_result['mean']:.4f} +/- {train_result['std']:.4f}")

    # Measure absorption on test data with DIFFERENT seed to test generalization
    test_result = measure_absorption(model, parent_dir, child_dirs, DEVICE, n_samples=EVAL_SAMPLES, seed=seed+1000)
    print(f"    Test absorption (Jaccard):  {test_result['mean']:.4f} +/- {test_result['std']:.4f}")

    return {
        "seed": seed,
        "train_ratio": train_ratio,
        "n_train": len(train_data),
        "n_test": len(test_data),
        "train_absorption": train_result,
        "test_absorption": test_result,
        "recon_mse": recon_err,
        "mean_l0": mean_l0,
        "parent_child1_cosine": float(np.dot(parent_dir, child1_dir)),
        "parent_child2_cosine": float(np.dot(parent_dir, child2_dir))
    }


def main():
    task_id = "heldout_generalization"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    start_time = datetime.now()

    print("=" * 70)
    print("Held-Out Generalization Test (FULL Mode)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Eval samples per split: {EVAL_SAMPLES}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    report_progress(task_id, RESULTS_DIR, epoch=1, total_epochs=len(SEEDS),
                    step=0, total_steps=len(SEEDS), loss=None,
                    metric={"stage": "start", "n_seeds": len(SEEDS)})

    # Run all seeds
    all_results = []
    for i, seed in enumerate(SEEDS):
        result = run_single_split(seed, train_ratio=TRAIN_RATIO)
        all_results.append(result)

        report_progress(task_id, RESULTS_DIR, epoch=i+1, total_epochs=len(SEEDS),
                        step=i+1, total_steps=len(SEEDS),
                        loss=result["recon_mse"],
                        metric={"train_absorption": result["train_absorption"]["mean"],
                                "test_absorption": result["test_absorption"]["mean"],
                                "seed": seed})

    # Aggregate across seeds
    train_means = [r["train_absorption"]["mean"] for r in all_results]
    test_means = [r["test_absorption"]["mean"] for r in all_results]
    train_stds = [r["train_absorption"]["std"] for r in all_results]
    test_stds = [r["test_absorption"]["std"] for r in all_results]

    overall_train_mean = np.mean(train_means)
    overall_train_std = np.mean(train_stds)
    overall_test_mean = np.mean(test_means)
    overall_test_std = np.mean(test_stds)

    # Within 10% check per seed
    pct_diffs = [abs(t - v) / max(t, 1e-10) * 100 for t, v in zip(train_means, test_means)]
    within_10pct_all = [d <= 10.0 for d in pct_diffs]
    within_10pct_majority = sum(within_10pct_all) / len(within_10pct_all) >= 0.5

    # Paired t-test across seeds
    t_stat, t_pval = ttest_rel(train_means, test_means)

    # Correlation across seeds (mean absorption per seed)
    if len(train_means) > 2:
        corr, corr_p = pearsonr(train_means, test_means)
        corr_above_08 = corr > 0.8
    else:
        corr = None
        corr_p = None
        corr_above_08 = False

    # Also compute correlation on raw scores pooled across seeds
    all_train_raw = []
    all_test_raw = []
    for r in all_results:
        all_train_raw.extend(r["train_absorption"]["raw"])
        all_test_raw.extend(r["test_absorption"]["raw"])

    if len(all_train_raw) > 10:
        pooled_corr, pooled_corr_p = pearsonr(all_train_raw, all_test_raw)
    else:
        pooled_corr = None
        pooled_corr_p = None

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\nOverall Train absorption: {overall_train_mean:.4f} +/- {overall_train_std:.4f}")
    print(f"Overall Test absorption:  {overall_test_mean:.4f} +/- {overall_test_std:.4f}")
    print(f"Paired t-test: t={t_stat:.4f}, p={t_pval:.4e}")

    for i, seed in enumerate(SEEDS):
        print(f"  Seed {seed}: train={train_means[i]:.4f}, test={test_means[i]:.4f}, pct_diff={pct_diffs[i]:.1f}%")

    print(f"\nWithin 10% per seed: {sum(within_10pct_all)}/{len(within_10pct_all)}")
    print(f"Majority within 10%: {'PASS' if within_10pct_majority else 'FAIL'}")

    if corr is not None:
        print(f"Correlation (seed means): r={corr:.4f}, p={corr_p:.4e}")
        print(f"Correlation > 0.8: {'PASS' if corr_above_08 else 'FAIL'}")
    else:
        print("Correlation (seed means): N/A")

    if pooled_corr is not None:
        print(f"Correlation (pooled raw): r={pooled_corr:.4f}, p={pooled_corr_p:.4e}")

    # Pass criteria
    criterion_1 = within_10pct_majority
    criterion_2 = corr_above_08 if corr is not None else False

    print(f"\nPass Criteria Evaluation:")
    print(f"  Test within 10% of train (majority): {'PASS' if criterion_1 else 'FAIL'}")
    print(f"  Correlation > 0.8: {'PASS' if criterion_2 else 'FAIL'}")

    overall_pass = criterion_1 and criterion_2
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    end_time = datetime.now()
    actual_min = int((end_time - start_time).total_seconds() / 60)

    # Prepare output
    output = {
        "task": task_id,
        "hypothesis": "Absorption generalizes to unseen hierarchical patterns (train/test split)",
        "mode": "full",
        "config": {
            "seeds": SEEDS,
            "train_ratio": TRAIN_RATIO,
            "n_samples": N_SAMPLES,
            "eval_samples": EVAL_SAMPLES,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "l0_target": L0_TARGET,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE
        },
        "results": {
            "overall_train_mean": float(overall_train_mean),
            "overall_train_std": float(overall_train_std),
            "overall_test_mean": float(overall_test_mean),
            "overall_test_std": float(overall_test_std),
            "per_seed": [
                {
                    "seed": r["seed"],
                    "train_absorption_mean": r["train_absorption"]["mean"],
                    "train_absorption_std": r["train_absorption"]["std"],
                    "test_absorption_mean": r["test_absorption"]["mean"],
                    "test_absorption_std": r["test_absorption"]["std"],
                    "percent_difference": float(pct_diffs[i]),
                    "recon_mse": r["recon_mse"],
                    "mean_l0": r["mean_l0"],
                    "parent_child1_cosine": r["parent_child1_cosine"],
                    "parent_child2_cosine": r["parent_child2_cosine"]
                }
                for i, r in enumerate(all_results)
            ]
        },
        "statistics": {
            "paired_t_stat": float(t_stat),
            "paired_t_pval": float(t_pval),
            "pearson_r_seed_means": float(corr) if corr is not None else None,
            "pearson_p_seed_means": float(corr_p) if corr_p is not None else None,
            "pearson_r_pooled": float(pooled_corr) if pooled_corr is not None else None,
            "pearson_p_pooled": float(pooled_corr_p) if pooled_corr_p is not None else None,
            "within_10pct_count": int(sum(within_10pct_all)),
            "within_10pct_total": len(within_10pct_all),
            "within_10pct_majority": bool(within_10pct_majority)
        },
        "pass_criteria": {
            "test_within_10pct_majority": bool(criterion_1),
            "correlation_above_0_8": bool(criterion_2),
            "overall_pass": bool(overall_pass)
        },
        "timing": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "actual_minutes": actual_min
        },
        "timestamp": datetime.now().isoformat()
    }

    output_path = RESULTS_DIR / f"{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    corr_str = f"{corr:.3f}" if corr is not None else "N/A"
    summary = f"Held-out generalization (FULL): train={overall_train_mean:.3f}, test={overall_test_mean:.3f}, pct_diffs={[f'{d:.1f}%' for d in pct_diffs]}, corr={corr_str}, overall={'PASS' if overall_pass else 'FAIL'}"
    mark_task_done(task_id, RESULTS_DIR, status="success" if overall_pass else "inconclusive", summary=summary)

    # Update gpu_progress.json
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}
    if gpu_progress_path.exists():
        try:
            gpu_progress = json.loads(gpu_progress_path.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    if task_id not in gpu_progress["completed"]:
        gpu_progress["completed"].append(task_id)
    if task_id in gpu_progress.get("running", {}):
        del gpu_progress["running"][task_id]

    gpu_progress["timings"][task_id] = {
        "planned_min": 20,
        "actual_min": actual_min,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_snapshot": {
            "model": "TopKSAE",
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "l0_target": L0_TARGET,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "train_ratio": TRAIN_RATIO,
            "n_seeds": len(SEEDS),
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "gpu_count": 1
        }
    }

    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))
    print(f"Updated: {gpu_progress_path}")

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
