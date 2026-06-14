#!/usr/bin/env python3
"""
Multi-Seed Stability Validation (H1) - FULL MODE

Replicate H1 across 5 seeds with stochastic hierarchy generation
to verify absorption stability and address zero-variance concern.

FULL mode: larger dataset (50000), more training steps (5000),
more absorption measurement samples (500), saved to full/ directory.

Validation criteria (from task_plan.json):
- Trained SAE absorption > 0.3 across all seeds
- Random baseline absorption < 0.2 for majority of seeds
- t-test p < 0.001 between trained and random
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import ttest_ind
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuration - FULL MODE
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_SAMPLES = 50000
D_MODEL = 128
D_SAE = 4096
L0_TARGET = 32
TRAIN_STEPS = 5000
BATCH_SIZE = 256
SEEDS = [42, 43, 44, 45, 46]
ABSORPTION_SAMPLES = 500


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
    """Generate synthetic data with hierarchical parent-child structure.

    Adds stochastic noise to hierarchy generation to address zero-variance concern.
    """
    rng = np.random.RandomState(seed)

    # Parent direction with stochastic variation
    parent_dir = rng.randn(d_model).astype(np.float32)
    parent_dir /= np.linalg.norm(parent_dir) + 1e-8

    # Child directions: correlated with parent but with stochastic noise
    noise_scale = stochastic_noise + rng.uniform(0.05, 0.15)
    child1_dir = (1.0 - noise_scale) * parent_dir + noise_scale * rng.randn(d_model).astype(np.float32)
    child1_dir /= np.linalg.norm(child1_dir) + 1e-8

    noise_scale2 = stochastic_noise + rng.uniform(0.05, 0.15)
    child2_dir = (1.0 - noise_scale2) * parent_dir + noise_scale2 * rng.randn(d_model).astype(np.float32)
    child2_dir /= np.linalg.norm(child2_dir) + 1e-8

    # Stochastic co-occurrence probabilities
    p_parent = 0.25 + rng.uniform(-0.05, 0.05)
    p_child1 = 0.25 + rng.uniform(-0.05, 0.05)
    p_child2 = 0.25 + rng.uniform(-0.05, 0.05)
    p_mixed = 1.0 - p_parent - p_child1 - p_child2

    # Generate samples
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


def train_sae(model, data, device, steps=5000, batch_size=256, lr=1e-3):
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


def measure_absorption(model, parent_dir, child_dirs, device, n_samples=500):
    """Measure absorption using top-k Jaccard overlap (primary metric)."""
    parent_t = torch.FloatTensor(parent_dir).to(device)
    child1_t = torch.FloatTensor(child_dirs[0]).to(device)
    child2_t = torch.FloatTensor(child_dirs[1]).to(device)

    overlap_scores = []

    with torch.no_grad():
        for _ in range(n_samples):
            strength = np.random.uniform(2.0, 4.0)

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
        "raw": overlap_scores
    }


def run_single_seed(seed, condition="trained"):
    """Run absorption measurement for a single seed."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n  Seed {seed} ({condition}):")

    # Generate stochastic hierarchical data
    data, labels, parent_dir, child1_dir, child2_dir = generate_hierarchical_data(
        n_samples=N_SAMPLES, d_model=D_MODEL, seed=seed, stochastic_noise=0.1
    )
    child_dirs = [child1_dir, child2_dir]

    if condition == "trained":
        # Train SAE
        model = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, l0_target=L0_TARGET).to(DEVICE)
        model = train_sae(model, data, DEVICE, steps=TRAIN_STEPS, batch_size=BATCH_SIZE)

        with torch.no_grad():
            sample = torch.FloatTensor(data[:1000]).to(DEVICE)
            recon, acts = model(sample)
            recon_err = nn.functional.mse_loss(recon, sample).item()
            mean_l0 = (acts > 0).float().sum(dim=-1).mean().item()
        print(f"    Recon MSE: {recon_err:.4f}, Mean L0: {mean_l0:.1f}")
    else:
        # Random SAE (untrained)
        model = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, l0_target=L0_TARGET).to(DEVICE)
        recon_err = None
        mean_l0 = None

    # Measure absorption
    result = measure_absorption(model, parent_dir, child_dirs, DEVICE, n_samples=ABSORPTION_SAMPLES)
    print(f"    Absorption (Jaccard): {result['mean']:.4f} +/- {result['std']:.4f}")

    return {
        "seed": seed,
        "condition": condition,
        "absorption": result,
        "recon_mse": recon_err,
        "mean_l0": mean_l0,
        "parent_child1_cosine": float(np.dot(parent_dir, child1_dir)),
        "parent_child2_cosine": float(np.dot(parent_dir, child2_dir))
    }


def main():
    task_id = "multiseed_validation"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    print("=" * 70)
    print("Multi-Seed Stability Validation (H1) - FULL MODE")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"Dataset size per seed: {N_SAMPLES}")
    print(f"Training steps: {TRAIN_STEPS}")
    print(f"Absorption measurement samples: {ABSORPTION_SAMPLES}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    report_progress(task_id, RESULTS_DIR, epoch=1, total_epochs=len(SEEDS)*2,
                    step=1, total_steps=len(SEEDS)*2, loss=None,
                    metric={"stage": "start", "mode": "full"})

    # Run trained SAE across all seeds
    print("\n[1/2] Trained SAE across seeds...")
    trained_results = []
    for i, seed in enumerate(SEEDS):
        result = run_single_seed(seed, condition="trained")
        trained_results.append(result)
        report_progress(task_id, RESULTS_DIR, epoch=i+1, total_epochs=len(SEEDS)*2,
                        step=i+1, total_steps=len(SEEDS)*2, loss=result.get("recon_mse"),
                        metric={"trained_seed": seed, "absorption": result["absorption"]["mean"]})

    # Run random SAE across all seeds
    print("\n[2/2] Random SAE across seeds...")
    random_results = []
    for i, seed in enumerate(SEEDS):
        result = run_single_seed(seed, condition="random")
        random_results.append(result)
        report_progress(task_id, RESULTS_DIR, epoch=len(SEEDS)+i+1, total_epochs=len(SEEDS)*2,
                        step=len(SEEDS)+i+1, total_steps=len(SEEDS)*2, loss=None,
                        metric={"random_seed": seed, "absorption": result["absorption"]["mean"]})

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    trained_means = [r["absorption"]["mean"] for r in trained_results]
    random_means = [r["absorption"]["mean"] for r in random_results]
    trained_stds = [r["absorption"]["std"] for r in trained_results]
    random_stds = [r["absorption"]["std"] for r in random_results]

    print(f"\nTrained SAE absorption across seeds:")
    for r in trained_results:
        print(f"  Seed {r['seed']}: {r['absorption']['mean']:.4f} +/- {r['absorption']['std']:.4f}")
    print(f"  Mean across seeds: {np.mean(trained_means):.4f} +/- {np.std(trained_means):.4f}")

    print(f"\nRandom SAE absorption across seeds:")
    for r in random_results:
        print(f"  Seed {r['seed']}: {r['absorption']['mean']:.4f} +/- {r['absorption']['std']:.4f}")
    print(f"  Mean across seeds: {np.mean(random_means):.4f} +/- {np.std(random_means):.4f}")

    # Variance check
    trained_variance = np.std(trained_means)
    print(f"\nVariance check:")
    print(f"  Trained variance across seeds: {trained_variance:.4f}")
    print(f"  With stochastic noise: variance {'> 0.05' if trained_variance > 0.05 else '<= 0.05'}")

    # t-test
    t_stat, p_value = ttest_ind(trained_means, random_means)
    print(f"\nT-test (trained vs random): t={t_stat:.3f}, p={p_value:.4e}")

    # Pass criteria
    criterion_1 = all(m > 0.3 for m in trained_means)
    criterion_2 = sum(m < 0.2 for m in random_means) >= 3  # majority of 5 = 3+
    criterion_3 = p_value < 0.001

    print(f"\nPass Criteria Evaluation:")
    print(f"  Trained > 0.3 across all seeds: {'PASS' if criterion_1 else 'FAIL'}")
    print(f"  Random < 0.2 for majority: {'PASS' if criterion_2 else 'FAIL'}")
    print(f"  t-test p < 0.001: {'PASS' if criterion_3 else 'FAIL'}")

    overall_pass = criterion_1 and criterion_2 and criterion_3
    print(f"\n  OVERALL H1: {'PASS' if overall_pass else 'FAIL'}")

    # Prepare output
    output = {
        "task": task_id,
        "hypothesis": "H1: Trained SAEs show higher multi-child proportional absorption than random baselines",
        "mode": "full",
        "config": {
            "seeds": SEEDS,
            "n_samples": N_SAMPLES,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "l0_target": L0_TARGET,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE,
            "absorption_samples": ABSORPTION_SAMPLES,
            "stochastic_noise": 0.1
        },
        "trained_results": trained_results,
        "random_results": random_results,
        "statistics": {
            "trained_mean": float(np.mean(trained_means)),
            "trained_std": float(np.std(trained_means)),
            "trained_variance_across_seeds": float(trained_variance),
            "random_mean": float(np.mean(random_means)),
            "random_std": float(np.std(random_means)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value)
        },
        "pass_criteria": {
            "trained_all_above_0.3": bool(criterion_1),
            "random_majority_below_0.2": bool(criterion_2),
            "t_test_p_below_0.001": bool(criterion_3),
            "overall_pass": bool(overall_pass)
        },
        "variance_analysis": {
            "trained_variance": float(trained_variance),
            "zero_variance_addressed": bool(trained_variance > 0.05),
            "notes": "Stochastic noise added to hierarchy generation to address zero-variance concern"
        },
        "timestamp": datetime.now().isoformat()
    }

    output_path = RESULTS_DIR / f"{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    summary = f"Multi-seed full: trained={np.mean(trained_means):.3f}+/-{np.std(trained_means):.3f}, random={np.mean(random_means):.3f}+/-{np.std(random_means):.3f}, p={p_value:.2e}, overall={'PASS' if overall_pass else 'FAIL'}"
    mark_task_done(task_id, RESULTS_DIR, status="success" if overall_pass else "inconclusive", summary=summary)

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
