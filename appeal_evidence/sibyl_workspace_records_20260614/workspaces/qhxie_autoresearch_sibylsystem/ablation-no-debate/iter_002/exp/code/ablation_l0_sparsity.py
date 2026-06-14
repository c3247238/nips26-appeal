#!/usr/bin/env python3
"""
Ablation: L0 Sparsity Level

Vary L0 sparsity target (20, 32, 50) to test sparsity effect on encoder-driven absorption.

Validation criteria (from task_plan.json):
- Higher sparsity -> higher absorption
- ANOVA p < 0.05

FULL mode: 5 seeds, 3 L0 levels, 10000 samples, 2000 training steps
"""

import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from scipy.stats import f_oneway
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 43, 44, 45, 46]
L0_LEVELS = [20, 32, 50]
D_MODEL = 128
D_SAE = 4096
TRAIN_STEPS = 2000
BATCH_SIZE = 256
N_SAMPLES_DATA = 10000
N_SAMPLES_MEASURE = 1000
HIERARCHY_SIMILARITY = 0.67  # Fixed at medium-high similarity


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


def generate_hierarchical_data(n_samples, d_model, target_similarity, seed=42):
    """Generate synthetic data with configurable parent-child cosine similarity."""
    rng = np.random.RandomState(seed)

    # Parent direction
    parent_dir = rng.randn(d_model).astype(np.float32)
    parent_dir /= np.linalg.norm(parent_dir) + 1e-8

    # Child directions: target cosine similarity with parent
    def make_child_dir(target_cos):
        orth = rng.randn(d_model).astype(np.float32)
        orth = orth - np.dot(orth, parent_dir) * parent_dir
        orth_norm = np.linalg.norm(orth)
        if orth_norm < 1e-8:
            orth = rng.randn(d_model).astype(np.float32)
            orth = orth - np.dot(orth, parent_dir) * parent_dir
            orth_norm = np.linalg.norm(orth)
        orth /= orth_norm + 1e-8

        child = target_cos * parent_dir + np.sqrt(max(0, 1 - target_cos**2)) * orth
        child /= np.linalg.norm(child) + 1e-8
        return child

    child1_dir = make_child_dir(target_similarity)
    child2_dir = make_child_dir(target_similarity)

    # Verify cosine
    cos1 = float(np.dot(parent_dir, child1_dir))
    cos2 = float(np.dot(parent_dir, child2_dir))

    # Generate samples
    data = []
    labels = []
    for _ in range(n_samples):
        base = rng.randn(d_model).astype(np.float32)
        base /= np.linalg.norm(base) + 1e-8

        mix = rng.rand()
        if mix < 0.25:
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * parent_dir
            labels.append(0)
        elif mix < 0.50:
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * child1_dir
            labels.append(1)
        elif mix < 0.75:
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

    return np.array(data), np.array(labels), parent_dir, child1_dir, child2_dir, cos1, cos2


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


def measure_absorption(model, parent_dir, child_dirs, device, n_samples=1000):
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


def run_single_condition(seed, l0_target):
    """Run absorption measurement for a single seed and L0 sparsity level."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"\n  Seed {seed}, L0={l0_target}:")

    # Generate hierarchical data with fixed similarity
    data, labels, parent_dir, child1_dir, child2_dir, actual_cos1, actual_cos2 = generate_hierarchical_data(
        n_samples=N_SAMPLES_DATA, d_model=D_MODEL, target_similarity=HIERARCHY_SIMILARITY, seed=seed
    )
    child_dirs = [child1_dir, child2_dir]
    print(f"    Actual cosines: p-c1={actual_cos1:.4f}, p-c2={actual_cos2:.4f}")

    # Train SAE with specified L0 target
    model = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, l0_target=l0_target).to(DEVICE)
    model = train_sae(model, data, DEVICE, steps=TRAIN_STEPS, batch_size=BATCH_SIZE)

    with torch.no_grad():
        sample = torch.FloatTensor(data[:1000]).to(DEVICE)
        recon, acts = model(sample)
        recon_err = nn.functional.mse_loss(recon, sample).item()
        mean_l0 = (acts > 0).float().sum(dim=-1).mean().item()
    print(f"    Recon MSE: {recon_err:.4f}, Mean L0: {mean_l0:.1f}")

    # Measure absorption
    result = measure_absorption(model, parent_dir, child_dirs, DEVICE, n_samples=N_SAMPLES_MEASURE)
    print(f"    Absorption (Jaccard): {result['mean']:.4f} +/- {result['std']:.4f}")

    return {
        "seed": seed,
        "l0_target": l0_target,
        "actual_cosine_similarity": (actual_cos1 + actual_cos2) / 2,
        "absorption": result,
        "recon_mse": recon_err,
        "mean_l0": mean_l0
    }


def main():
    task_id = "ablation_l0_sparsity"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    start_time = datetime.now()

    print("=" * 70)
    print("Ablation: L0 Sparsity Level (FULL Mode)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"L0 sparsity levels: {L0_LEVELS}")
    print(f"Fixed hierarchy similarity: {HIERARCHY_SIMILARITY}")
    print(f"Data samples: {N_SAMPLES_DATA}")
    print(f"Measurement samples: {N_SAMPLES_MEASURE}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    report_progress(task_id, RESULTS_DIR, epoch=1, total_epochs=len(SEEDS)*len(L0_LEVELS),
                    step=1, total_steps=len(SEEDS)*len(L0_LEVELS), loss=None,
                    metric={"stage": "start"})

    # Run all conditions
    all_results = []
    total_runs = len(SEEDS) * len(L0_LEVELS)
    run_idx = 0

    for l0_target in L0_LEVELS:
        print(f"\n{'='*70}")
        print(f"L0 sparsity level: {l0_target}")
        print(f"{'='*70}")

        for seed in SEEDS:
            run_idx += 1
            result = run_single_condition(seed, l0_target)
            all_results.append(result)

            report_progress(task_id, RESULTS_DIR, epoch=run_idx, total_epochs=total_runs,
                            step=run_idx, total_steps=total_runs,
                            loss=result["recon_mse"],
                            metric={"l0_target": l0_target, "seed": seed,
                                    "absorption": result["absorption"]["mean"]})

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Group by L0 level
    by_l0 = {l0: [] for l0 in L0_LEVELS}
    for r in all_results:
        by_l0[r["l0_target"]].append(r)

    print(f"\nAbsorption by L0 sparsity level:")
    l0_means = []
    l0_stds = []
    l0_raws = []

    for l0 in L0_LEVELS:
        means = [r["absorption"]["mean"] for r in by_l0[l0]]
        stds = [r["absorption"]["std"] for r in by_l0[l0]]
        raws = []
        for r in by_l0[l0]:
            raws.extend(r["absorption"]["raw"])

        mean = np.mean(means)
        std = np.std(means)
        print(f"  L0={l0}: {mean:.4f} +/- {std:.4f} (across {len(means)} seeds)")
        l0_means.append(mean)
        l0_stds.append(std)
        l0_raws.append(raws)

    # Monotonicity check (higher sparsity -> higher absorption)
    monotonic = all(l0_means[i] <= l0_means[i+1] for i in range(len(l0_means)-1))
    print(f"\nMonotonic increase with sparsity: {'PASS' if monotonic else 'FAIL'}")

    # ANOVA
    f_stat, p_value = f_oneway(*l0_raws)
    print(f"ANOVA: F={f_stat:.3f}, p={p_value:.4e} {'PASS' if p_value < 0.05 else 'FAIL'}")

    # Pass criteria
    criterion_1 = monotonic
    criterion_2 = p_value < 0.05

    print(f"\nPass Criteria Evaluation:")
    print(f"  Higher sparsity -> higher absorption: {'PASS' if criterion_1 else 'FAIL'}")
    print(f"  ANOVA p < 0.05: {'PASS' if criterion_2 else 'FAIL'}")

    overall_pass = criterion_1 and criterion_2
    print(f"\n  OVERALL: {'PASS' if overall_pass else 'FAIL'}")

    end_time = datetime.now()
    actual_min = int((end_time - start_time).total_seconds() / 60)

    # Prepare output
    results = {
        "task": task_id,
        "hypothesis": "Higher L0 sparsity leads to higher absorption in encoder-driven SAEs",
        "mode": "full",
        "config": {
            "seeds": SEEDS,
            "l0_levels": L0_LEVELS,
            "hierarchy_similarity": HIERARCHY_SIMILARITY,
            "n_samples_data": N_SAMPLES_DATA,
            "n_samples_measure": N_SAMPLES_MEASURE,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE
        },
        "results": {
            str(l0): [
                {
                    "seed": r["seed"],
                    "absorption_mean": r["absorption"]["mean"],
                    "absorption_std": r["absorption"]["std"],
                    "recon_mse": r["recon_mse"],
                    "mean_l0": r["mean_l0"],
                    "actual_cosine_similarity": r["actual_cosine_similarity"]
                }
                for r in by_l0[l0]
            ]
            for l0 in L0_LEVELS
        },
        "statistics": {
            "l0_means": {str(l0): float(l0_means[i]) for i, l0 in enumerate(L0_LEVELS)},
            "l0_stds": {str(l0): float(l0_stds[i]) for i, l0 in enumerate(L0_LEVELS)},
            "monotonic": bool(monotonic),
            "anova_f": float(f_stat),
            "anova_p": float(p_value)
        },
        "pass_criteria": {
            "monotonic_increase": bool(monotonic),
            "anova_p_below_0.05": bool(criterion_2),
            "overall_pass": bool(overall_pass)
        },
        "timing": {
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "actual_minutes": actual_min
        },
        "timestamp": datetime.now().isoformat()
    }

    output_file = RESULTS_DIR / f"{task_id}.json"
    output_file.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {output_file}")

    summary = f"L0 ablation: L0={L0_LEVELS}, means={[f'{m:.3f}' for m in l0_means]}, monotonic={'PASS' if monotonic else 'FAIL'}, ANOVA p={p_value:.2e}"
    mark_task_done(task_id, RESULTS_DIR, status="success" if overall_pass else "completed", summary=summary)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
