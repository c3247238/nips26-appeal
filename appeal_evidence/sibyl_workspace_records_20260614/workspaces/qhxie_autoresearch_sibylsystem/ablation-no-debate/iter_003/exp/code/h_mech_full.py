#!/usr/bin/env python3
"""
H_Mech Full: 2x2 factorial with 5 seeds (stochastic hierarchy).
Task: --tasks=h_mech_full

Tests whether encoder alignment drives absorption (Condition B ≈ D) and decoder is irrelevant (Condition C ≈ A).
Full experiment: 5 seeds × 4 conditions × 100 samples = 20 runs, 500 measurements each.
"""
import json
import os
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
REMOTE_BASE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-debate/current")
RESULTS_DIR = REMOTE_BASE / "exp" / "results"
CODE_DIR = REMOTE_BASE / "exp" / "code"

# Task ID for this run
TASK_ID = "h_mech_full"

# Seeds for full experiment
SEEDS = [42, 123, 456, 789, 1024]
N_SAMPLES = 100
CONDITIONS = [
    ("A", "random_both"),
    ("B", "random_decoder"),
    ("C", "random_encoder"),
    ("D", "original"),
]

device = "cuda" if torch.cuda.is_available() else "cpu"


def write_pid_file():
    """Write PID file for system recovery detection."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"PID file written: {pid_file}")
    return pid_file


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(status="success", summary=""):
    """Write DONE marker file for system monitor."""
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()

    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"DONE marker written: {marker}")


def load_model_and_sae():
    """Load GPT-2 Small + SAE from sae-lens."""
    print("Loading GPT-2 Small...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    print("Loading SAE (blocks.8.hook_resid_pre)...")
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id="blocks.8.hook_resid_pre",
        device=device
    )
    return model, sae


def find_high_activation_features(model, sae, prompts, n_features=20):
    """Find features with highest mean activation across prompts."""
    all_activations = []

    for prompt in prompts:
        tokens = model.to_tokens(prompt).to(device)
        if tokens.shape[0] > 1:
            tokens = tokens[0:1]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            resid = cache["resid_pre", 8]
            features = sae.encode(resid.unsqueeze(0)).squeeze(0)
            all_activations.append(features.mean(dim=0))

    mean_activations = torch.stack(all_activations).mean(dim=0)
    top_features = mean_activations.topk(n_features).indices.tolist()
    return top_features, mean_activations


def create_variant_sae(sae, variant_type):
    """Create variant SAE with modified encoder/decoder."""
    d_sae, d_model = sae.W_enc.shape

    if variant_type == "original":
        return sae.W_enc.clone(), sae.W_dec.clone()
    elif variant_type == "random_encoder":
        W_enc = torch.randn(d_sae, d_model, device=device)
        W_enc = W_enc / W_enc.norm(dim=1, keepdim=True)
        return W_enc, sae.W_dec.clone()
    elif variant_type == "random_decoder":
        W_dec = torch.randn(d_model, d_sae, device=device)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        return sae.W_enc.clone(), W_dec
    elif variant_type == "random_both":
        W_enc = torch.randn(d_sae, d_model, device=device)
        W_enc = W_enc / W_enc.norm(dim=1, keepdim=True)
        W_dec = torch.randn(d_model, d_sae, device=device)
        W_dec = W_dec / W_dec.norm(dim=0, keepdim=True)
        return W_enc, W_dec


class VariantSAE:
    """Temporary SAE wrapper with variant weights for ablation."""
    def __init__(self, W_enc, W_dec):
        self.W_enc = W_enc
        self.W_dec = W_dec

    def encode(self, x):
        h = torch.nn.functional.relu(x @ self.W_enc)
        return h

    def decode(self, h):
        return h @ self.W_dec


def measure_absorption(model, variant_sae, parent_idx, child_indices, n_samples=50):
    """Measure multi-child proportional absorption."""
    parent_acts_before = []
    parent_acts_after = []

    prompts = [
        "The capital of France is Paris, a beautiful city in Europe",
        "Machine learning models can learn complex patterns from data",
        "The weather today is sunny and warm with clear skies",
        "Scientists discovered new particles in recent experiments",
        "The book was written by a famous author about history",
        "Neural networks use gradients to learn representations",
        "Climate change affects ecosystems around the world",
        "The company released new products for customers",
    ]

    for i in range(n_samples):
        prompt = prompts[i % len(prompts)]

        tokens = model.to_tokens(prompt).to(device)
        if tokens.shape[0] > 1:
            tokens = tokens[0:1]

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, remove_batch_dim=True)
            resid = cache["resid_pre", 8]

            # Encode to SAE features
            features = variant_sae.encode(resid.unsqueeze(0)).squeeze(0)

            # Parent activation before ablation (at final token)
            parent_before = features[-1, parent_idx].item()

            # Ablate child features
            ablated = features.clone()
            ablated[:, child_indices] = 0

            # Decode to residual space
            ablated_resid = ablated @ variant_sae.W_dec

            # Re-encode
            reencoded = variant_sae.encode(ablated_resid.unsqueeze(0)).squeeze(0)
            parent_after = reencoded[-1, parent_idx].item()

        parent_acts_before.append(max(0, parent_before))
        parent_acts_after.append(max(0, parent_after))

    before_mean = np.mean(parent_acts_before)
    after_mean = np.mean(parent_acts_after)

    if before_mean > 1e-6:
        absorption = after_mean / before_mean
    else:
        absorption = 0.0

    return absorption, before_mean, after_mean


def run_condition(model, sae, condition, variant_type, parent_idx, child_indices, n_samples, seed):
    """Run one condition of the 2x2 factorial."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    W_enc, W_dec = create_variant_sae(sae, variant_type)
    variant_sae = VariantSAE(W_enc, W_dec)

    absorption, before, after = measure_absorption(
        model, variant_sae, parent_idx, child_indices, n_samples
    )

    enc_type = "trained" if variant_type in ["original", "random_decoder"] else "random"
    dec_type = "trained" if variant_type in ["original", "random_encoder"] else "random"

    return {
        "condition": condition,
        "encoder": enc_type,
        "decoder": dec_type,
        "absorption_rate": float(absorption),
        "parent_activation_before": float(before),
        "parent_activation_after": float(after),
    }


def main():
    print(f"\n{'='*60}")
    print(f"H_Mech Full: 5 seeds × 4 conditions × {N_SAMPLES} samples")
    print(f"{'='*60}\n")

    # Write PID file immediately
    write_pid_file()

    # Load model and SAE
    model, sae = load_model_and_sae()
    d_sae, d_model = sae.W_enc.shape
    print(f"\nSAE: d_sae={d_sae}, d_model={d_model}")

    # Find high-activation features (same selection for all seeds to ensure comparability)
    prompts = [
        "The capital of France is Paris",
        "Machine learning is transforming AI",
        "The weather is nice today",
    ]
    high_act_features, _ = find_high_activation_features(model, sae, prompts, n_features=20)

    # Select parent and child features (fixed for all seeds)
    parent_idx = high_act_features[0]
    child_indices = high_act_features[1:6]
    print(f"Parent feature: {parent_idx}")
    print(f"Child features: {child_indices}")

    # Run full experiment
    all_results = []
    condition_absorption = {cond: [] for cond, _ in CONDITIONS}

    total_runs = len(SEEDS) * len(CONDITIONS)
    run_count = 0

    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n--- Seed {seed} ({seed_idx+1}/{len(SEEDS)}) ---")

        seed_results = {"seed": seed, "conditions": []}

        for cond_name, variant_type in CONDITIONS:
            run_count += 1

            enc_type = "trained" if variant_type in ["original", "random_decoder"] else "random"
            dec_type = "trained" if variant_type in ["original", "random_encoder"] else "random"

            print(f"  Condition {cond_name}: Encoder={enc_type}, Decoder={dec_type}", end=" ... ")

            result = run_condition(
                model, sae, cond_name, variant_type,
                parent_idx, child_indices, N_SAMPLES, seed
            )

            seed_results["conditions"].append(result)
            condition_absorption[cond_name].append(result["absorption_rate"])
            print(f"absorption={result['absorption_rate']:.4f}")

        all_results.append(seed_results)

        # Report progress after each seed
        report_progress(
            epoch=seed_idx + 1,
            total_epochs=len(SEEDS),
            step=run_count,
            total_steps=total_runs,
            metric={"completed_conditions": run_count}
        )

    # Aggregate results
    summary = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": SEEDS,
            "n_samples_per_condition": N_SAMPLES,
            "total_runs": total_runs,
        },
        "results_by_seed": all_results,
        "condition_summary": {},
    }

    # Compute per-condition statistics
    for cond_name, _ in CONDITIONS:
        rates = condition_absorption[cond_name]
        summary["condition_summary"][cond_name] = {
            "mean": float(np.mean(rates)),
            "std": float(np.std(rates)),
            "min": float(np.min(rates)),
            "max": float(np.max(rates)),
            "n": len(rates),
            "values": [float(r) for r in rates],
        }

    # Compute factorial checks
    means = {cond: summary["condition_summary"][cond]["mean"] for cond in condition_absorption}

    b_vs_d = abs(means["B"] - means["D"])
    c_vs_a = means["C"] - means["A"]

    summary["factorial_checks"] = {
        "encoder_driven_check": b_vs_d < 0.1,
        "decoder_irrelevant_check": abs(c_vs_a) < 0.05,
        "b_vs_d_delta": float(b_vs_d),
        "c_vs_a_delta": float(c_vs_a),
        "pass": b_vs_d < 0.1 and abs(c_vs_a) < 0.05,
    }

    # Save results
    output_file = RESULTS_DIR / "full" / f"h_mech_5seeds.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("H_Mech Full Results (5 seeds)")
    print("="*60)
    for cond in ["A", "B", "C", "D"]:
        stats = summary["condition_summary"][cond]
        print(f"  Condition {cond}: {stats['mean']:.4f} ± {stats['std']:.4f}")
    print(f"\n|B-D| = {b_vs_d:.4f} (threshold 0.1): {summary['factorial_checks']['encoder_driven_check']}")
    print(f"C-A = {c_vs_a:.4f} (threshold 0.05): {summary['factorial_checks']['decoder_irrelevant_check']}")
    print(f"Full PASS: {summary['factorial_checks']['pass']}")

    # Mark done
    mark_task_done(
        status="success" if summary["factorial_checks"]["pass"] else "partial",
        summary=f"H_Mech full with 5 seeds. B-D delta={b_vs_d:.4f}, C-A delta={c_vs_a:.4f}"
    )

    return summary


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError during execution: {e}")
        import traceback
        traceback.print_exc()
        mark_task_done(status="failed", summary=f"Error: {str(e)}")
        raise