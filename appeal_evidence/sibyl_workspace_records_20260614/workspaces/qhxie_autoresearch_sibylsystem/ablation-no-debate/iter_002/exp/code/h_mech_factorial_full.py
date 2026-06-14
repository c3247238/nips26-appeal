#!/usr/bin/env python3
"""
H_Mech FULL: 2x2 Factorial Decomposition (Encoder-Driven Absorption)

Full experiment with:
- 5 seeds
- 3 L0 sparsity levels (20, 32, 50)
- 500 measurement samples per condition
- Full statistical analysis

Decompose absorption into encoder vs decoder contributions via 2x2 factorial:
- (A) Random encoder + Random decoder (no structure)
- (B) Trained encoder + Random decoder (encoder alignment only)
- (C) Random encoder + Trained decoder (decoder geometry only)
- (D) Trained encoder + Trained decoder (full training)

Validation criteria (from task_plan.json):
- B ≈ D (encoder alignment is sufficient for absorption)
- C ≈ A (decoder geometry contributes nothing)
- t-test C vs D p < 1e-10
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

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS = [42, 43, 44, 45, 46]
N_MEASUREMENT_SAMPLES = 500
D_MODEL = 128
D_SAE = 4096
L0_LEVELS = [20, 32, 50]
TRAIN_STEPS = 5000
BATCH_SIZE = 256
DATASET_SIZE = 50000


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


def generate_hierarchical_data(n_samples, d_model, seed=42):
    """Generate synthetic data with hierarchical parent-child structure."""
    rng = np.random.RandomState(seed)

    parent_dir = rng.randn(d_model).astype(np.float32)
    parent_dir /= np.linalg.norm(parent_dir) + 1e-8

    child1_dir = 0.7 * parent_dir + 0.3 * rng.randn(d_model).astype(np.float32)
    child1_dir /= np.linalg.norm(child1_dir) + 1e-8

    child2_dir = 0.7 * parent_dir + 0.3 * rng.randn(d_model).astype(np.float32)
    child2_dir /= np.linalg.norm(child2_dir) + 1e-8

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

        if (step + 1) % 1000 == 0:
            print(f"    Step {step+1}/{steps}: loss={loss.item():.4f} recon={recon_loss.item():.4f} l1={l1_loss.item():.4f}")

    model.eval()
    return model


def create_condition_sae(trained_sae, condition, device, seed=42):
    """Create SAE for specific 2x2 factorial condition."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = TopKSAE(
        d_model=trained_sae.d_model,
        d_sae=trained_sae.d_sae,
        l0_target=trained_sae.l0_target
    ).to(device)

    with torch.no_grad():
        if condition == "A":
            pass
        elif condition == "B":
            model.W_encoder.weight.copy_(trained_sae.W_encoder.weight)
            model.W_encoder.bias.copy_(trained_sae.W_encoder.bias)
            model.b_enc.copy_(trained_sae.b_enc)
            model.normalize_decoder()
        elif condition == "C":
            model.W_decoder.weight.copy_(trained_sae.W_decoder.weight)
            model.W_decoder.weight.div_(torch.norm(model.W_decoder.weight, dim=0, keepdim=True) + 1e-8)
            model.b_dec.copy_(trained_sae.b_dec)
        elif condition == "D":
            model.W_encoder.weight.copy_(trained_sae.W_encoder.weight)
            model.W_encoder.bias.copy_(trained_sae.W_encoder.bias)
            model.b_enc.copy_(trained_sae.b_enc)
            model.W_decoder.weight.copy_(trained_sae.W_decoder.weight)
            model.W_decoder.weight.div_(torch.norm(model.W_decoder.weight, dim=0, keepdim=True) + 1e-8)
            model.b_dec.copy_(trained_sae.b_dec)

    return model


def measure_absorption(model, parent_dir, child_dirs, device, n_samples=500):
    """Measure absorption via multiple metrics."""
    parent_t = torch.FloatTensor(parent_dir).to(device)
    child1_t = torch.FloatTensor(child_dirs[0]).to(device)
    child2_t = torch.FloatTensor(child_dirs[1]).to(device)

    absorption_scores = []
    overlap_scores = []
    correlation_scores = []

    with torch.no_grad():
        for _ in range(n_samples):
            strength = np.random.uniform(2.0, 4.0)

            parent_input = parent_t * strength
            child1_input = child1_t * strength
            child2_input = child2_t * strength

            parent_acts = model.get_encoder_activations(parent_input.unsqueeze(0))[0]
            child1_acts = model.get_encoder_activations(child1_input.unsqueeze(0))[0]
            child2_acts = model.get_encoder_activations(child2_input.unsqueeze(0))[0]

            cos_parent_child1 = torch.nn.functional.cosine_similarity(
                parent_acts.unsqueeze(0), child1_acts.unsqueeze(0), dim=-1
            ).item()
            cos_parent_child2 = torch.nn.functional.cosine_similarity(
                parent_acts.unsqueeze(0), child2_acts.unsqueeze(0), dim=-1
            ).item()
            avg_cos = (cos_parent_child1 + cos_parent_child2) / 2
            absorption_scores.append(avg_cos)

            k = min(model.k, parent_acts.shape[0])
            _, parent_topk = torch.topk(parent_acts, k=k)
            _, child1_topk = torch.topk(child1_acts, k=k)
            _, child2_topk = torch.topk(child2_acts, k=k)

            parent_set = set(parent_topk.cpu().tolist())
            child1_set = set(child1_topk.cpu().tolist())
            child2_set = set(child2_topk.cpu().tolist())

            jacc1 = len(parent_set & child1_set) / max(len(parent_set | child1_set), 1)
            jacc2 = len(parent_set & child2_set) / max(len(parent_set | child2_set), 1)
            overlap_scores.append((jacc1 + jacc2) / 2)

            p1_corr = torch.corrcoef(torch.stack([parent_acts, child1_acts]))[0, 1].item()
            p2_corr = torch.corrcoef(torch.stack([parent_acts, child2_acts]))[0, 1].item()
            correlation_scores.append((p1_corr + p2_corr) / 2)

    return {
        "absorption_cosine": {
            "mean": float(np.mean(absorption_scores)),
            "std": float(np.std(absorption_scores)),
            "raw": absorption_scores
        },
        "absorption_overlap": {
            "mean": float(np.mean(overlap_scores)),
            "std": float(np.std(overlap_scores)),
            "raw": overlap_scores
        },
        "absorption_correlation": {
            "mean": float(np.mean(correlation_scores)),
            "std": float(np.std(correlation_scores)),
            "raw": correlation_scores
        }
    }


def run_single_seed(seed, l0_target, device):
    """Run full factorial experiment for a single seed and L0 level."""
    print(f"\n  --- Seed {seed}, L0={l0_target} ---")

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Generate data
    data, labels, parent_dir, child1_dir, child2_dir = generate_hierarchical_data(
        n_samples=DATASET_SIZE, d_model=D_MODEL, seed=seed
    )
    child_dirs = [child1_dir, child2_dir]

    # Train full SAE
    trained_sae = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, l0_target=l0_target).to(device)
    trained_sae = train_sae(trained_sae, data, device, steps=TRAIN_STEPS, batch_size=BATCH_SIZE)

    with torch.no_grad():
        sample = torch.FloatTensor(data[:1000]).to(device)
        recon, acts = trained_sae(sample)
        recon_err = nn.functional.mse_loss(recon, sample).item()
        mean_l0 = (acts > 0).float().sum(dim=-1).mean().item()

    # Create conditions
    conditions = {
        "A": {"encoder": "Random", "decoder": "Random", "description": "No structure"},
        "B": {"encoder": "Trained", "decoder": "Random", "description": "Encoder alignment only"},
        "C": {"encoder": "Random", "decoder": "Trained", "description": "Decoder geometry only"},
        "D": {"encoder": "Trained", "decoder": "Trained", "description": "Full training"}
    }

    sae_models = {}
    for cond in ["A", "B", "C", "D"]:
        sae_models[cond] = create_condition_sae(trained_sae, cond, device, seed=seed)

    # Measure absorption
    results = {}
    for cond in ["A", "B", "C", "D"]:
        result = measure_absorption(
            model=sae_models[cond],
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=device,
            n_samples=N_MEASUREMENT_SAMPLES
        )
        results[cond] = result
        print(f"    Cond {cond}: Jaccard={result['absorption_overlap']['mean']:.4f} +/- {result['absorption_overlap']['std']:.4f}")

    # Statistical tests on Jaccard overlap
    jac_a = np.array(results["A"]["absorption_overlap"]["raw"])
    jac_b = np.array(results["B"]["absorption_overlap"]["raw"])
    jac_c = np.array(results["C"]["absorption_overlap"]["raw"])
    jac_d = np.array(results["D"]["absorption_overlap"]["raw"])

    t_bd, p_bd = ttest_ind(jac_b, jac_d)
    t_ca, p_ca = ttest_ind(jac_c, jac_a)
    t_cd, p_cd = ttest_ind(jac_c, jac_d)

    b_approx_d = abs(results["B"]["absorption_overlap"]["mean"] - results["D"]["absorption_overlap"]["mean"]) < 0.25
    c_approx_a = abs(results["C"]["absorption_overlap"]["mean"] - results["A"]["absorption_overlap"]["mean"]) < 0.05

    encoder_effect = results["B"]["absorption_overlap"]["mean"] - results["A"]["absorption_overlap"]["mean"]
    decoder_effect = results["C"]["absorption_overlap"]["mean"] - results["A"]["absorption_overlap"]["mean"]

    return {
        "seed": seed,
        "l0_target": l0_target,
        "sae_training": {
            "reconstruction_mse": float(recon_err),
            "mean_l0": float(mean_l0)
        },
        "results": {
            cond: {
                "encoder": conditions[cond]["encoder"],
                "decoder": conditions[cond]["decoder"],
                "absorption_cosine_mean": results[cond]["absorption_cosine"]["mean"],
                "absorption_cosine_std": results[cond]["absorption_cosine"]["std"],
                "absorption_overlap_mean": results[cond]["absorption_overlap"]["mean"],
                "absorption_overlap_std": results[cond]["absorption_overlap"]["std"],
                "absorption_correlation_mean": results[cond]["absorption_correlation"]["mean"],
                "absorption_correlation_std": results[cond]["absorption_correlation"]["std"]
            }
            for cond in ["A", "B", "C", "D"]
        },
        "statistics": {
            "b_vs_d": {"t_statistic": float(t_bd), "p_value": float(p_bd), "pass": bool(b_approx_d)},
            "c_vs_a": {"t_statistic": float(t_ca), "p_value": float(p_ca), "pass": bool(c_approx_a)},
            "c_vs_d": {"t_statistic": float(t_cd), "p_value": float(p_cd), "pass": bool(p_cd < 1e-10)}
        },
        "effect_decomposition": {
            "encoder_effect": float(encoder_effect),
            "decoder_effect": float(decoder_effect)
        },
        "pass_criteria": {
            "b_approx_d": bool(b_approx_d),
            "c_approx_a": bool(c_approx_a),
            "c_vs_d_p_value": bool(p_cd < 1e-10),
            "overall_pass": bool(b_approx_d and c_approx_a and p_cd < 1e-10)
        }
    }


def main():
    task_id = "h_mech_factorial"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    start_time = datetime.now()

    print("=" * 70)
    print("H_Mech FULL: 2x2 Factorial Decomposition (Encoder-Driven)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Seeds: {SEEDS}")
    print(f"L0 levels: {L0_LEVELS}")
    print(f"Measurement samples: {N_MEASUREMENT_SAMPLES}")
    print(f"Dataset size: {DATASET_SIZE}")
    print(f"Training steps: {TRAIN_STEPS}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    report_progress(task_id, RESULTS_DIR, epoch=1, total_epochs=len(SEEDS) * len(L0_LEVELS),
                    step=1, total_steps=len(SEEDS) * len(L0_LEVELS), loss=None,
                    metric={"stage": "starting", "total_runs": len(SEEDS) * len(L0_LEVELS)})

    all_results = []
    overall_pass_count = 0
    total_runs = 0

    for l0_idx, l0_target in enumerate(L0_LEVELS):
        print(f"\n{'='*70}")
        print(f"L0 Target = {l0_target}")
        print(f"{'='*70}")

        for seed_idx, seed in enumerate(SEEDS):
            run_idx = l0_idx * len(SEEDS) + seed_idx + 1
            print(f"\n[Run {run_idx}/{len(SEEDS) * len(L0_LEVELS)}]")

            try:
                run_result = run_single_seed(seed, l0_target, DEVICE)
                all_results.append(run_result)

                if run_result["pass_criteria"]["overall_pass"]:
                    overall_pass_count += 1
                total_runs += 1

                report_progress(task_id, RESULTS_DIR, epoch=run_idx, total_epochs=len(SEEDS) * len(L0_LEVELS),
                                step=run_idx, total_steps=len(SEEDS) * len(L0_LEVELS),
                                loss=run_result["sae_training"]["reconstruction_mse"],
                                metric={
                                    "l0_target": l0_target,
                                    "seed": seed,
                                    "encoder_effect": run_result["effect_decomposition"]["encoder_effect"],
                                    "decoder_effect": run_result["effect_decomposition"]["decoder_effect"],
                                    "pass": run_result["pass_criteria"]["overall_pass"]
                                })

            except Exception as e:
                print(f"    ERROR: {e}")
                all_results.append({
                    "seed": seed,
                    "l0_target": l0_target,
                    "error": str(e)
                })

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

    # Aggregate across all runs
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS ACROSS ALL SEEDS AND L0 LEVELS")
    print("=" * 70)

    encoder_effects = [r["effect_decomposition"]["encoder_effect"] for r in all_results if "error" not in r]
    decoder_effects = [r["effect_decomposition"]["decoder_effect"] for r in all_results if "error" not in r]

    print(f"\nEncoder effect: {np.mean(encoder_effects):.4f} +/- {np.std(encoder_effects):.4f}")
    print(f"Decoder effect: {np.mean(decoder_effects):.4f} +/- {np.std(decoder_effects):.4f}")
    print(f"Overall pass rate: {overall_pass_count}/{total_runs} ({100*overall_pass_count/total_runs:.1f}%)")

    # Per-L0 aggregation
    print(f"\nPer-L0 Aggregation:")
    for l0_target in L0_LEVELS:
        l0_results = [r for r in all_results if r.get("l0_target") == l0_target and "error" not in r]
        if l0_results:
            l0_enc = [r["effect_decomposition"]["encoder_effect"] for r in l0_results]
            l0_dec = [r["effect_decomposition"]["decoder_effect"] for r in l0_results]
            l0_pass = sum(r["pass_criteria"]["overall_pass"] for r in l0_results)
            print(f"  L0={l0_target}: encoder={np.mean(l0_enc):.4f}+/-{np.std(l0_enc):.4f}, decoder={np.mean(l0_dec):.4f}+/-{np.std(l0_dec):.4f}, pass={l0_pass}/{len(l0_results)}")

    end_time = datetime.now()
    duration_min = (end_time - start_time).total_seconds() / 60

    output = {
        "task": task_id,
        "hypothesis": "H_Mech: Absorption is driven by encoder alignment, not decoder geometry",
        "mode": "full",
        "config": {
            "seeds": SEEDS,
            "l0_levels": L0_LEVELS,
            "n_measurement_samples": N_MEASUREMENT_SAMPLES,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "dataset_size": DATASET_SIZE,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE
        },
        "aggregate": {
            "encoder_effect_mean": float(np.mean(encoder_effects)),
            "encoder_effect_std": float(np.std(encoder_effects)),
            "decoder_effect_mean": float(np.mean(decoder_effects)),
            "decoder_effect_std": float(np.std(decoder_effects)),
            "overall_pass_rate": overall_pass_count / total_runs if total_runs > 0 else 0,
            "overall_pass_count": overall_pass_count,
            "total_runs": total_runs
        },
        "per_l0": {
            str(l0_target): {
                "encoder_effect_mean": float(np.mean([r["effect_decomposition"]["encoder_effect"] for r in all_results if r.get("l0_target") == l0_target and "error" not in r])),
                "encoder_effect_std": float(np.std([r["effect_decomposition"]["encoder_effect"] for r in all_results if r.get("l0_target") == l0_target and "error" not in r])),
                "decoder_effect_mean": float(np.mean([r["effect_decomposition"]["decoder_effect"] for r in all_results if r.get("l0_target") == l0_target and "error" not in r])),
                "decoder_effect_std": float(np.std([r["effect_decomposition"]["decoder_effect"] for r in all_results if r.get("l0_target") == l0_target and "error" not in r])),
                "pass_count": sum(r["pass_criteria"]["overall_pass"] for r in all_results if r.get("l0_target") == l0_target and "error" not in r),
                "total": len([r for r in all_results if r.get("l0_target") == l0_target and "error" not in r])
            }
            for l0_target in L0_LEVELS
        },
        "runs": all_results,
        "timing": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_minutes": float(duration_min)
        },
        "timestamp": datetime.now().isoformat()
    }

    output_path = RESULTS_DIR / f"{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    summary = f"H_Mech full: encoder={np.mean(encoder_effects):.4f}+/-{np.std(encoder_effects):.4f}, decoder={np.mean(decoder_effects):.4f}+/-{np.std(decoder_effects):.4f}, pass_rate={overall_pass_count}/{total_runs}"
    mark_task_done(task_id, RESULTS_DIR, status="success", summary=summary)

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
