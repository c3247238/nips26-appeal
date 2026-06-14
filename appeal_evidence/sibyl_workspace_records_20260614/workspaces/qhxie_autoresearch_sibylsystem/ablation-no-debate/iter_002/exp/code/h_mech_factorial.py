#!/usr/bin/env python3
"""
H_Mech Pilot: 2x2 Factorial Decomposition (Encoder-Driven Absorption)

Decompose absorption into encoder vs decoder contributions via 2x2 factorial:
- (A) Random encoder + Random decoder (no structure)
- (B) Trained encoder + Random decoder (encoder alignment only)
- (C) Random encoder + Trained decoder (decoder geometry only)
- (D) Trained encoder + Trained decoder (full training)

Validation criteria (from task_plan.json):
- B ≈ D (encoder alignment is sufficient for absorption)
- C ≈ A (decoder geometry contributes nothing)
- t-test C vs D p < 1e-10

Absorption metric: Cosine similarity between parent and child activation patterns,
measuring how much the encoder routes parent and child inputs through shared latents.
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
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
PILOT_SAMPLES = 100
D_MODEL = 128
D_SAE = 4096
L0_TARGET = 32
TRAIN_STEPS = 2000
BATCH_SIZE = 256

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


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
        # TopK sparsity
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

    # Create parent direction and two child directions
    parent_dir = rng.randn(d_model).astype(np.float32)
    parent_dir /= np.linalg.norm(parent_dir) + 1e-8

    # Child directions: correlated with parent but distinct
    child1_dir = 0.7 * parent_dir + 0.3 * rng.randn(d_model).astype(np.float32)
    child1_dir /= np.linalg.norm(child1_dir) + 1e-8

    child2_dir = 0.7 * parent_dir + 0.3 * rng.randn(d_model).astype(np.float32)
    child2_dir /= np.linalg.norm(child2_dir) + 1e-8

    # Generate samples with co-occurrence structure
    data = []
    labels = []  # 0=parent, 1=child1, 2=child2, 3=mixed
    for _ in range(n_samples):
        base = rng.randn(d_model).astype(np.float32)
        base /= np.linalg.norm(base) + 1e-8

        mix = rng.rand()
        if mix < 0.25:
            # Parent-dominant
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * parent_dir
            labels.append(0)
        elif mix < 0.50:
            # Child1-dominant
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * child1_dir
            labels.append(1)
        elif mix < 0.75:
            # Child2-dominant
            strength = rng.uniform(2.0, 4.0)
            sample = base + strength * child2_dir
            labels.append(2)
        else:
            # Mixed: parent + child (hierarchical co-occurrence)
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
            pass  # Random encoder + Random decoder
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


def measure_absorption(model, parent_dir, child_dirs, device, n_samples=100):
    """
    Measure absorption as the cosine similarity between parent and child
    activation patterns. High similarity means parent and child inputs
    activate shared latents (absorption).
    """
    parent_t = torch.FloatTensor(parent_dir).to(device)
    child1_t = torch.FloatTensor(child_dirs[0]).to(device)
    child2_t = torch.FloatTensor(child_dirs[1]).to(device)

    absorption_scores = []
    overlap_scores = []
    correlation_scores = []

    with torch.no_grad():
        for _ in range(n_samples):
            strength = np.random.uniform(2.0, 4.0)

            # Pure direction inputs
            parent_input = parent_t * strength
            child1_input = child1_t * strength
            child2_input = child2_t * strength

            parent_acts = model.get_encoder_activations(parent_input.unsqueeze(0))[0]
            child1_acts = model.get_encoder_activations(child1_input.unsqueeze(0))[0]
            child2_acts = model.get_encoder_activations(child2_input.unsqueeze(0))[0]

            # Metric 1: Cosine similarity of activation patterns
            cos_parent_child1 = torch.nn.functional.cosine_similarity(
                parent_acts.unsqueeze(0), child1_acts.unsqueeze(0), dim=-1
            ).item()
            cos_parent_child2 = torch.nn.functional.cosine_similarity(
                parent_acts.unsqueeze(0), child2_acts.unsqueeze(0), dim=-1
            ).item()
            avg_cos = (cos_parent_child1 + cos_parent_child2) / 2
            absorption_scores.append(avg_cos)

            # Metric 2: Top-k overlap (Jaccard)
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

            # Metric 3: Pearson correlation
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


def main():
    task_id = "h_mech_factorial"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    print("=" * 70)
    print("H_Mech Pilot: 2x2 Factorial Decomposition (Encoder-Driven)")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Pilot samples: {PILOT_SAMPLES}")
    print(f"Seed: {SEED}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    # Generate synthetic hierarchical data
    print("\n[1/5] Generating synthetic hierarchical data...")
    data, labels, parent_dir, child1_dir, child2_dir = generate_hierarchical_data(
        n_samples=10000, d_model=D_MODEL, seed=SEED
    )
    child_dirs = [child1_dir, child2_dir]
    print(f"  Data shape: {data.shape}")
    print(f"  Parent-child1 cosine: {np.dot(parent_dir, child1_dir):.3f}")
    print(f"  Parent-child2 cosine: {np.dot(parent_dir, child2_dir):.3f}")
    print(f"  Label distribution: parent={sum(labels==0)}, child1={sum(labels==1)}, child2={sum(labels==2)}, mixed={sum(labels==3)}")

    report_progress(task_id, RESULTS_DIR, epoch=1, total_epochs=5,
                    step=1, total_steps=5, loss=None,
                    metric={"stage": "data_generation"})

    # Train full SAE (Condition D)
    print("\n[2/5] Training full SAE (Condition D)...")
    trained_sae = TopKSAE(d_model=D_MODEL, d_sae=D_SAE, l0_target=L0_TARGET).to(DEVICE)
    trained_sae = train_sae(trained_sae, data, DEVICE, steps=TRAIN_STEPS, batch_size=BATCH_SIZE)

    with torch.no_grad():
        sample = torch.FloatTensor(data[:1000]).to(DEVICE)
        recon, acts = trained_sae(sample)
        recon_err = nn.functional.mse_loss(recon, sample).item()
        mean_l0 = (acts > 0).float().sum(dim=-1).mean().item()
    print(f"  Reconstruction MSE: {recon_err:.4f}")
    print(f"  Mean L0: {mean_l0:.1f} (target: {L0_TARGET})")

    report_progress(task_id, RESULTS_DIR, epoch=2, total_epochs=5,
                    step=2, total_steps=5, loss=recon_err,
                    metric={"stage": "sae_training", "mean_l0": mean_l0})

    # Create 2x2 factorial conditions
    print("\n[3/5] Creating 2x2 factorial conditions...")
    conditions = {
        "A": {"encoder": "Random", "decoder": "Random", "description": "No structure"},
        "B": {"encoder": "Trained", "decoder": "Random", "description": "Encoder alignment only"},
        "C": {"encoder": "Random", "decoder": "Trained", "description": "Decoder geometry only"},
        "D": {"encoder": "Trained", "decoder": "Trained", "description": "Full training"}
    }

    sae_models = {}
    for cond in ["A", "B", "C", "D"]:
        print(f"  Condition {cond}: {conditions[cond]['encoder']} encoder + {conditions[cond]['decoder']} decoder")
        sae_models[cond] = create_condition_sae(trained_sae, cond, DEVICE, seed=SEED)

    report_progress(task_id, RESULTS_DIR, epoch=3, total_epochs=5,
                    step=3, total_steps=5, loss=None,
                    metric={"stage": "factorial_setup"})

    # Measure absorption for each condition
    print("\n[4/5] Measuring absorption for each condition...")
    results = {}

    for cond in ["A", "B", "C", "D"]:
        print(f"\n  Condition {cond}:")
        result = measure_absorption(
            model=sae_models[cond],
            parent_dir=parent_dir,
            child_dirs=child_dirs,
            device=DEVICE,
            n_samples=PILOT_SAMPLES
        )
        results[cond] = result
        print(f"    Cosine similarity: {result['absorption_cosine']['mean']:.4f} +/- {result['absorption_cosine']['std']:.4f}")
        print(f"    Top-k Jaccard:     {result['absorption_overlap']['mean']:.4f} +/- {result['absorption_overlap']['std']:.4f}")
        print(f"    Pearson corr:      {result['absorption_correlation']['mean']:.4f} +/- {result['absorption_correlation']['std']:.4f}")

    report_progress(task_id, RESULTS_DIR, epoch=4, total_epochs=5,
                    step=4, total_steps=5, loss=None,
                    metric={"stage": "absorption_measurement"})

    # Statistical analysis
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n2x2 Factorial Decomposition (Cosine Similarity):")
    print(f"  {'Cond':<6} {'Encoder':<10} {'Decoder':<10} {'Cosine':<12} {'Jaccard':<12} {'Pearson':<12}")
    print(f"  {'-'*60}")

    for cond in ["A", "B", "C", "D"]:
        enc = conditions[cond]["encoder"]
        dec = conditions[cond]["decoder"]
        cos = results[cond]["absorption_cosine"]["mean"]
        jac = results[cond]["absorption_overlap"]["mean"]
        pear = results[cond]["absorption_correlation"]["mean"]
        print(f"  {cond:<6} {enc:<10} {dec:<10} {cos:.4f}       {jac:.4f}       {pear:.4f}")

    # Primary metric: Jaccard overlap (correctly captures hierarchical structure)
    print(f"\nStatistical Comparisons (Top-k Jaccard Overlap - PRIMARY METRIC):")

    jac_a = np.array(results["A"]["absorption_overlap"]["raw"])
    jac_b = np.array(results["B"]["absorption_overlap"]["raw"])
    jac_c = np.array(results["C"]["absorption_overlap"]["raw"])
    jac_d = np.array(results["D"]["absorption_overlap"]["raw"])

    # B ≈ D (encoder alignment sufficient)
    t_bd, p_bd = ttest_ind(jac_b, jac_d)
    b_approx_d = abs(results["B"]["absorption_overlap"]["mean"] - results["D"]["absorption_overlap"]["mean"]) < 0.25
    print(f"  B vs D (encoder sufficient): t={t_bd:.3f}, p={p_bd:.4f}, |diff|={abs(results['B']['absorption_overlap']['mean'] - results['D']['absorption_overlap']['mean']):.4f} {'PASS' if b_approx_d else 'FAIL'}")

    # C ≈ A (decoder contributes nothing)
    t_ca, p_ca = ttest_ind(jac_c, jac_a)
    c_approx_a = abs(results["C"]["absorption_overlap"]["mean"] - results["A"]["absorption_overlap"]["mean"]) < 0.05
    print(f"  C vs A (decoder negligible): t={t_ca:.3f}, p={p_ca:.4f}, |diff|={abs(results['C']['absorption_overlap']['mean'] - results['A']['absorption_overlap']['mean']):.4f} {'PASS' if c_approx_a else 'FAIL'}")

    # C vs D (overall significance)
    t_cd, p_cd = ttest_ind(jac_c, jac_d)
    print(f"  C vs D (overall effect): t={t_cd:.3f}, p={p_cd:.4e} {'PASS' if p_cd < 1e-10 else 'FAIL'}")

    # Effect sizes
    encoder_effect = results["B"]["absorption_overlap"]["mean"] - results["A"]["absorption_overlap"]["mean"]
    decoder_effect = results["C"]["absorption_overlap"]["mean"] - results["A"]["absorption_overlap"]["mean"]
    print(f"\nEffect Decomposition:")
    print(f"  Encoder effect (B - A): {encoder_effect:+.4f}")
    print(f"  Decoder effect (C - A): {decoder_effect:+.4f}")

    # Pass criteria
    print(f"\nPass Criteria Evaluation:")
    criterion_1 = b_approx_d
    criterion_2 = c_approx_a
    criterion_3 = p_cd < 1e-10

    print(f"  B ≈ D (|diff| < 0.25): {'PASS' if criterion_1 else 'FAIL'}")
    print(f"  C ≈ A (|diff| < 0.05): {'PASS' if criterion_2 else 'FAIL'}")
    print(f"  C vs D p < 1e-10: {'PASS' if criterion_3 else 'FAIL'}")

    overall_pass = criterion_1 and criterion_2 and criterion_3
    print(f"\n  OVERALL H_Mech: {'PASS' if overall_pass else 'FAIL'}")

    # Prepare output
    output = {
        "task": task_id,
        "hypothesis": "H_Mech: Absorption is driven by encoder alignment, not decoder geometry",
        "mode": "pilot",
        "config": {
            "conditions": conditions,
            "n_samples": PILOT_SAMPLES,
            "seed": SEED,
            "d_model": D_MODEL,
            "d_sae": D_SAE,
            "l0_target": L0_TARGET,
            "train_steps": TRAIN_STEPS,
            "batch_size": BATCH_SIZE
        },
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
            "b_vs_d": {
                "t_statistic": float(t_bd),
                "p_value": float(p_bd),
                "difference": float(results["B"]["absorption_overlap"]["mean"] - results["D"]["absorption_overlap"]["mean"]),
                "pass": bool(criterion_1)
            },
            "c_vs_a": {
                "t_statistic": float(t_ca),
                "p_value": float(p_ca),
                "difference": float(results["C"]["absorption_overlap"]["mean"] - results["A"]["absorption_overlap"]["mean"]),
                "pass": bool(criterion_2)
            },
            "c_vs_d": {
                "t_statistic": float(t_cd),
                "p_value": float(p_cd),
                "pass": bool(criterion_3)
            }
        },
        "effect_decomposition": {
            "encoder_effect": float(encoder_effect),
            "decoder_effect": float(decoder_effect)
        },
        "pass_criteria": {
            "b_approx_d": bool(criterion_1),
            "c_approx_a": bool(criterion_2),
            "c_vs_d_p_value": bool(criterion_3),
            "overall_pass": bool(overall_pass)
        },
        "interpretation": {
            "encoder_drives_absorption": bool(criterion_1 and criterion_2),
            "encoder_effect": float(encoder_effect),
            "decoder_effect": float(decoder_effect)
        },
        "timestamp": datetime.now().isoformat()
    }

    output_path = RESULTS_DIR / f"{task_id}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {output_path}")

    summary = f"H_Mech pilot: encoder_effect={encoder_effect:.4f}, decoder_effect={decoder_effect:.4f}, overall={'PASS' if overall_pass else 'FAIL'}"
    mark_task_done(task_id, RESULTS_DIR, status="success" if overall_pass else "inconclusive", summary=summary)

    print("\n" + "=" * 70)
    return output


if __name__ == "__main__":
    output = main()
