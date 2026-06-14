#!/usr/bin/env python3
"""
Pilot P2: DFDA on GPT-2 Small
Dynamic Feature De-Absorption via tiny MLP compensation.
Target: mean improvement > 5%, runtime < 15 min.
"""

import os
import json
import time
import warnings
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE

warnings.filterwarnings("ignore")

# ── Config ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
TASK_ID = "p2_dfda_gpt2"
LAYER = 8
N_PAIRS = 4
N_TRAIN = 500
N_TEST = 200

# ── Process Tracking ────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "metric": metric or {}, "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file.unlink(missing_ok=True)
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress.exists():
        try:
            final_progress = json.loads(progress.read_text())
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    }))

# ── Set seeds ───────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Tiny MLP for compensation ───────────────────────────────────────
class TinyMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    def forward(self, x):
        return self.net(x)

# ── Helper: find absorbed pairs using UAD approach ──────────────────
def find_absorbed_pairs(model, sae, tokenizer, device, n_samples=500, n_pairs=4):
    """Find absorbed (parent, child) feature pairs."""
    print("[DFDA] Finding absorbed pairs...")

    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms artificial intelligence research.",
        "Scientists discovered new particles in the collider.",
        "The economy grew by three percent last quarter.",
        "She walked through the garden and picked flowers.",
    ] * 100

    tokenized = tokenizer(prompts[:n_samples], return_tensors="pt", padding=True,
                          truncation=True, max_length=64)
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', f'blocks.{LAYER}.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        flat = sae_acts.reshape(-1, sae_acts.shape[-1])
        binary = (flat > 0).float().cpu().numpy()

    # Find features with high co-occurrence but different marginal frequencies
    feature_freq = binary.mean(axis=0)
    alive = np.where(feature_freq > 0.001)[0]

    # Compute phi coefficient for alive features
    pairs = []
    n = binary.shape[0]
    for idx, i in enumerate(alive[:200]):
        fi = binary[:, i]
        n1_i = fi.sum()
        if n1_i < 10:
            continue
        for j in alive[idx+1:idx+50]:
            if i == j:
                continue
            fj = binary[:, j]
            n11 = (fi * fj).sum()
            if n11 < 5:
                continue
            n1_j = fj.sum()
            n00 = ((1-fi) * (1-fj)).sum()
            n10 = (fi * (1-fj)).sum()
            n01 = ((1-fi) * fj).sum()
            denom = np.sqrt(n1_i * (n-n1_i) * n1_j * (n-n1_j))
            if denom > 0:
                phi = (n11 * n00 - n10 * n01) / denom
                # Absorption: high phi, one feature fires more often
                if phi > 0.3 and abs(feature_freq[i] - feature_freq[j]) > 0.01:
                    parent = i if feature_freq[i] > feature_freq[j] else j
                    child = j if parent == i else i
                    pairs.append((int(parent), int(child), float(phi)))

    # Sort by phi and return top n_pairs
    pairs.sort(key=lambda x: x[2], reverse=True)
    selected = pairs[:n_pairs]
    print(f"[DFDA] Selected {len(selected)} absorbed pairs")
    for p, c, phi in selected:
        print(f"  Parent={p}, Child={c}, phi={phi:.3f}")
    return selected

# ── Helper: collect training data for a pair ────────────────────────
def collect_pair_data(model, sae, tokenizer, device, parent_feat, child_feat, n_samples=500):
    """Collect examples where child fires but parent does not (or weakly)."""
    prompts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning transforms artificial intelligence research daily.",
        "Scientists discovered new particles in the Large Hadron Collider.",
        "The economy grew by three percent last quarter despite challenges.",
        "She walked through the beautiful garden and picked some fresh flowers.",
        "Python is a popular programming language for data science and AI.",
        "The movie received critical acclaim at the international film festival.",
        "Climate change affects ecosystems and biodiversity worldwide significantly.",
        "Mathematics provides the foundation for modern physics and engineering.",
        "The company announced record profits this year exceeding all expectations.",
    ] * 50

    tokenized = tokenizer(prompts[:n_samples], return_tensors="pt", padding=True,
                          truncation=True, max_length=64)
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', f'blocks.{LAYER}.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        flat = sae_acts.reshape(-1, sae_acts.shape[-1])

    child_vals = flat[:, child_feat].cpu().numpy()
    parent_vals = flat[:, parent_feat].cpu().numpy()

    # Find positions where child fires strongly but parent doesn't
    child_positive = child_vals[child_vals > 0]
    parent_positive = parent_vals[parent_vals > 0]

    child_thresh = np.percentile(child_positive, 50) if len(child_positive) > 0 else 0
    parent_thresh = np.percentile(parent_positive, 50) if len(parent_positive) > 0 else 0

    mask = (child_vals > child_thresh) & (parent_vals < parent_thresh)

    # If mask is too small, relax
    if mask.sum() < 10:
        mask = child_vals > 0

    X = child_vals[mask].reshape(-1, 1)
    y = parent_vals[mask].reshape(-1, 1)

    return X, y

# ── Helper: train MLP compensation ──────────────────────────────────
def train_mlp_compensation(X_train, y_train, X_test, y_test, hidden_dim=64, epochs=100, lr=0.01):
    """Train tiny MLP to predict parent residual from child activation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.float32, device=device)

    mlp = TinyMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=1).to(device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(epochs):
        mlp.train()
        optimizer.zero_grad()
        pred = mlp(X_train_t)
        loss = criterion(pred, y_train_t)
        loss.backward()
        optimizer.step()

    # Evaluate
    mlp.eval()
    with torch.no_grad():
        baseline_mse = ((y_test_t - 0) ** 2).mean().item()
        pred_test = mlp(X_test_t)
        compensated_mse = ((y_test_t - pred_test) ** 2).mean().item()

    improvement = (baseline_mse - compensated_mse) / (baseline_mse + 1e-10) * 100
    n_params = sum(p.numel() for p in mlp.parameters())

    return {
        "baseline_mse": baseline_mse,
        "compensated_mse": compensated_mse,
        "improvement_pct": improvement,
        "n_params": n_params,
    }

# ── Main ────────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    print("=" * 60)
    print("Pilot P2: DFDA on GPT-2 Small")
    print("=" * 60)

    report_progress(1, 4, step=1, total_steps=4, metric={"phase": "init"})

    # Load model and SAE
    print("[1/4] Loading model and SAE...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE,
    )
    print(f"  Loaded SAE: d_sae={sae.cfg.d_sae}")

    # Find absorbed pairs
    print("[2/4] Finding absorbed pairs...")
    report_progress(2, 4, step=2, total_steps=4, metric={"phase": "finding_pairs"})
    absorbed_pairs = find_absorbed_pairs(model, sae, tokenizer, DEVICE, n_samples=500, n_pairs=N_PAIRS)

    if len(absorbed_pairs) == 0:
        print("[ERROR] No absorbed pairs found!")
        mark_done(status="failure", summary="No absorbed pairs found")
        return

    # Train DFDA for each pair
    print("[3/4] Training DFDA compensation...")
    report_progress(3, 4, step=3, total_steps=4, metric={"phase": "training"})

    pair_results = []
    for idx, (parent, child, phi) in enumerate(absorbed_pairs):
        print(f"\n  Pair {idx+1}/{len(absorbed_pairs)}: parent={parent}, child={child}")
        X_train, y_train = collect_pair_data(model, sae, tokenizer, DEVICE, parent, child, n_samples=N_TRAIN)
        X_test, y_test = collect_pair_data(model, sae, tokenizer, DEVICE, parent, child, n_samples=N_TEST)

        if len(X_train) < 10:
            print(f"    [SKIP] Too few training examples ({len(X_train)})")
            continue

        result = train_mlp_compensation(X_train, y_train, X_test, y_test, hidden_dim=64, epochs=100)
        result["pair_id"] = idx + 1
        result["parent_feature"] = parent
        result["child_feature"] = child
        result["phi"] = phi
        result["n_train"] = len(X_train)
        result["n_test"] = len(X_test)
        pair_results.append(result)

        print(f"    Baseline MSE: {result['baseline_mse']:.6f}")
        print(f"    Compensated MSE: {result['compensated_mse']:.6f}")
        print(f"    Improvement: {result['improvement_pct']:.2f}%")
        print(f"    Params: {result['n_params']}")

    # Aggregate
    print("\n[4/4] Aggregating results...")
    report_progress(4, 4, step=4, total_steps=4, metric={"phase": "aggregation"})

    if pair_results:
        improvements = [r["improvement_pct"] for r in pair_results]
        mean_improvement = np.mean(improvements)
        positive_ratio = sum(1 for x in improvements if x > 0) / len(improvements)
        total_params = sum(r["n_params"] for r in pair_results)
        sae_params = sum(p.numel() for p in sae.parameters())
        param_ratio = total_params / sae_params * 100
    else:
        mean_improvement = 0
        positive_ratio = 0
        total_params = 0
        param_ratio = 0

    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Pairs evaluated: {len(pair_results)}")
    print(f"Mean improvement: {mean_improvement:.2f}%")
    print(f"Positive ratio: {positive_ratio:.2%}")
    print(f"Total params: {total_params} ({param_ratio:.4f}% of SAE)")
    print(f"Runtime: {elapsed:.1f}s")

    results = {
        "task_id": TASK_ID,
        "model": "gpt2-small",
        "layer": LAYER,
        "seed": SEED,
        "absorbed_pairs": absorbed_pairs,
        "pair_results": pair_results,
        "aggregate": {
            "mean_improvement_pct": mean_improvement,
            "positive_ratio": positive_ratio,
            "total_params": total_params,
            "sae_params": sae_params,
            "param_ratio_pct": param_ratio,
        },
        "runtime_seconds": elapsed,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = RESULTS_DIR / f"{TASK_ID}_results.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[Save] Results saved to {output_path}")

    passed = mean_improvement > 5 and elapsed < 900
    print(f"\n[{'PASS' if passed else 'FAIL'}] Mean improvement={mean_improvement:.2f}% (target > 5%), Runtime={elapsed:.1f}s")

    mark_done(status="success" if passed else "failure",
              summary=f"DFDA: mean={mean_improvement:.2f}%, positive={positive_ratio:.2%}, params={total_params}, T={elapsed:.1f}s")

    return results

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failure", summary=error_msg[:500])
        raise
