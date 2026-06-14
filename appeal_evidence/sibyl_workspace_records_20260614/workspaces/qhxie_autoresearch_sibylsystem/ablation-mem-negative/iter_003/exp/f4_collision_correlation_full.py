#!/usr/bin/env python3
"""
Full E4: Collision Rate - Absorption Rate Correlation (Extended)

Extends P1 to multiple hierarchy types:
- Number words (one-eight)
- Punctuation marks
- Case features (upper/lower)

Tests whether collision rate (top-k feature overlap) correlates with
true absorption rate across ALL hierarchy types.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm

from sae_lens import SAE
from transformer_lens import HookedTransformer

SEED = 42
MODEL_NAME = "gpt2"
SAE_ID = "gpt2-small-res-jb"
LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"
DATASET_SAMPLES = 1000
TOPK = 5
PROBE_THRESHOLD = 0.1

RESULTS_DIR = Path("exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "f4_collision_correlation_full"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

# Hierarchy definitions
NUMBER_WORDS = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight']
PUNCTUATION = ['.', ',', '!', '?', ';', ':', '"', "'"]
CASE_PAIRS = [
    ('a', 'A'), ('b', 'B'), ('c', 'C'), ('d', 'D'), ('e', 'E'),
    ('f', 'F'), ('g', 'G'), ('h', 'H'), ('i', 'I'), ('j', 'J'),
    ('k', 'K'), ('l', 'L'), ('m', 'M'), ('n', 'N'), ('o', 'O'),
    ('p', 'P'), ('q', 'Q'), ('r', 'R'), ('s', 'S'), ('t', 'T'),
    ('u', 'U'), ('v', 'V'), ('w', 'W'), ('x', 'X'), ('y', 'Y'), ('z', 'Z')
]


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID, "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    }))


def probe_feature_for_token(model, sae, token_str, device="cuda"):
    """Find the top-k SAE features most activated by a single token."""
    tokens = model.to_tokens(token_str)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
        resid = cache[HOOK_NAME].squeeze(0)
        acts = sae.encode(resid)
    # Use last position
    last_acts = acts[-1].cpu().numpy()
    topk_idx = np.argsort(last_acts)[-TOPK:][::-1]
    topk_vals = last_acts[topk_idx]
    return {
        "token": token_str,
        "topk_idx": [int(x) for x in topk_idx],
        "topk_vals": [float(x) for x in topk_vals],
        "max_activation": float(last_acts[topk_idx[0]]),
    }


def compute_collision_rate(probe_a, probe_b):
    """Compute collision rate as Jaccard similarity of top-k feature sets."""
    set_a = set(probe_a["topk_idx"])
    set_b = set(probe_b["topk_idx"])
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    jaccard = intersection / union if union > 0 else 0.0
    # Also compute overlap ratio (intersection / min(len(a), len(b)))
    min_len = min(len(set_a), len(set_b))
    overlap = intersection / min_len if min_len > 0 else 0.0
    return {
        "jaccard": jaccard,
        "overlap": overlap,
        "shared_features": list(set_a & set_b),
        "n_shared": intersection,
    }


def compute_true_absorption(probes, parent_set, sae, model, device="cuda"):
    """
    Compute true absorption rate using Chanin et al. definition:
    A feature 'absorbs' multiple child concepts if it responds to ALL of them.
    True absorption rate for a pair = max feature activation across shared features,
    where a shared feature must have activation > threshold for BOTH tokens.
    """
    # Get full activation vectors for all tokens in the parent set
    full_acts = {}
    for token in parent_set:
        tokens_t = model.to_tokens(token)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens_t, names_filter=[HOOK_NAME])
            resid = cache[HOOK_NAME].squeeze(0)
            acts = sae.encode(resid)
        full_acts[token] = acts[-1].cpu().numpy()

    # Find features that respond to multiple tokens (absorption features)
    threshold = 5.0
    absorption_features = {}
    for feat_idx in range(sae.cfg.d_sae):
        responding_tokens = []
        for token in parent_set:
            if full_acts[token][feat_idx] > threshold:
                responding_tokens.append(token)
        if len(responding_tokens) > 1:
            absorption_features[feat_idx] = responding_tokens

    # For each pair, compute true absorption = fraction of absorption features they share
    pairs = []
    for i in range(len(parent_set)):
        for j in range(i + 1, len(parent_set)):
            token_i = parent_set[i]
            token_j = parent_set[j]
            probe_i = probes[token_i]
            probe_j = probes[token_j]
            collision = compute_collision_rate(probe_i, probe_j)

            # True absorption: how many absorption features respond to BOTH tokens
            shared_absorption = 0
            total_absorption = 0
            for feat_idx, tokens in absorption_features.items():
                if token_i in tokens or token_j in tokens:
                    total_absorption += 1
                    if token_i in tokens and token_j in tokens:
                        shared_absorption += 1

            true_absorption = shared_absorption / total_absorption if total_absorption > 0 else 0.0

            pairs.append({
                "pair": f"{token_i}-{token_j}",
                "token1": token_i,
                "token2": token_j,
                "collision_rate": collision["jaccard"],
                "overlap": collision["overlap"],
                "n_shared": collision["n_shared"],
                "shared_features": collision["shared_features"],
                "true_absorption": true_absorption,
                "shared_absorption_features": shared_absorption,
                "total_absorption_features": total_absorption,
            })
    return pairs


def bootstrap_ci(x, y, n_bootstrap=1000, ci=0.95):
    """Compute bootstrap confidence interval for Spearman correlation."""
    rng = np.random.RandomState(SEED)
    n = len(x)
    correlations = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        if len(np.unique(idx)) < 2:
            continue
        bx, by = x[idx], y[idx]
        if np.std(bx) == 0 or np.std(by) == 0:
            continue
        r, _ = spearmanr(bx, by)
        if not np.isnan(r):
            correlations.append(r)

    if len(correlations) == 0:
        return None, None, 0

    correlations = sorted(correlations)
    alpha = 1 - ci
    lower_idx = int(alpha / 2 * len(correlations))
    upper_idx = int((1 - alpha / 2) * len(correlations))
    return correlations[lower_idx], correlations[upper_idx], len(correlations)


def main():
    start_time = time.time()
    PID_FILE.write_text(str(os.getpid()))
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    report_progress(0, 4, metric={"stage": "loading_model"})
    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=device)
    print(f"Model loaded. Layers: {model.cfg.n_layers}")

    report_progress(1, 4, metric={"stage": "loading_sae"})
    print(f"Loading SAE: {SAE_ID}, layer {LAYER}...")
    sae = SAE.from_pretrained(release=SAE_ID, sae_id=f"blocks.{LAYER}.hook_resid_pre", device=device)
    print(f"SAE loaded. d_sae: {sae.cfg.d_sae}, d_in: {sae.cfg.d_in}")

    report_progress(2, 4, metric={"stage": "probing_hierarchies"})
    print("\n" + "=" * 60)
    print("PROBING FEATURES FOR ALL HIERARCHIES")
    print("=" * 60)

    # Probe number words
    print("\n--- Number Words ---")
    number_probes = {}
    for word in NUMBER_WORDS:
        probe = probe_feature_for_token(model, sae, word, device)
        number_probes[word] = probe
        print(f"  {word:8s}: max_act={probe['max_activation']:.2f}, topk={probe['topk_idx']}")

    # Probe punctuation
    print("\n--- Punctuation ---")
    punct_probes = {}
    for punct in PUNCTUATION:
        probe = probe_feature_for_token(model, sae, punct, device)
        punct_probes[punct] = probe
        print(f"  {punct:8s}: max_act={probe['max_activation']:.2f}, topk={probe['topk_idx']}")

    # Probe case pairs
    print("\n--- Case Pairs ---")
    case_probes = {}
    for lower, upper in CASE_PAIRS:
        probe_lower = probe_feature_for_token(model, sae, lower, device)
        probe_upper = probe_feature_for_token(model, sae, upper, device)
        case_probes[lower] = probe_lower
        case_probes[upper] = probe_upper

    report_progress(3, 4, metric={"stage": "computing_correlations"})
    print("\n" + "=" * 60)
    print("COMPUTING COLLISION RATES AND CORRELATIONS")
    print("=" * 60)

    # Number word pairs (all pairs)
    number_pairs = compute_true_absorption(number_probes, NUMBER_WORDS, sae, model, device)
    print(f"\nNumber word pairs: {len(number_pairs)}")

    # Punctuation pairs
    punct_pairs = compute_true_absorption(punct_probes, PUNCTUATION, sae, model, device)
    print(f"Punctuation pairs: {len(punct_pairs)}")

    # Case pairs (only lower-upper pairs, not all combinations)
    case_pair_results = []
    for lower, upper in CASE_PAIRS:
        collision = compute_collision_rate(case_probes[lower], case_probes[upper])
        case_pair_results.append({
            "pair": f"{lower}-{upper}",
            "token1": lower,
            "token2": upper,
            "collision_rate": collision["jaccard"],
            "overlap": collision["overlap"],
            "n_shared": collision["n_shared"],
            "shared_features": collision["shared_features"],
            "true_absorption": 0.0,  # Not computed for case pairs (different concept)
            "shared_absorption_features": 0,
            "total_absorption_features": 0,
        })
    print(f"Case pairs: {len(case_pair_results)}")

    # Combine all pairs
    all_pairs = number_pairs + punct_pairs + case_pair_results
    print(f"\nTotal pairs across all hierarchies: {len(all_pairs)}")

    # Compute correlations
    # Only use pairs with valid true_absorption (number and punctuation pairs)
    valid_pairs = [p for p in all_pairs if p.get("total_absorption_features", 0) > 0]
    case_only_pairs = [p for p in all_pairs if p.get("total_absorption_features", 0) == 0]

    collision_rates = np.array([p["collision_rate"] for p in valid_pairs])
    true_absorption_vals = np.array([p["true_absorption"] for p in valid_pairs])

    spearman_r, spearman_p = spearmanr(collision_rates, true_absorption_vals)
    pearson_r, pearson_p = pearsonr(collision_rates, true_absorption_vals)

    # Bootstrap CI
    ci_lower, ci_upper, n_valid = bootstrap_ci(collision_rates, true_absorption_vals)

    print(f"\n--- Correlation Results ---")
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
    if ci_lower is not None:
        print(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] (n_valid={n_valid})")

    # Per-hierarchy breakdown
    hierarchy_stats = {}
    for name, pairs in [("numbers", number_pairs), ("punctuation", punct_pairs)]:
        if len(pairs) == 0:
            continue
        c_rates = np.array([p["collision_rate"] for p in pairs])
        a_vals = np.array([p["true_absorption"] for p in pairs])
        if np.std(c_rates) > 0 and np.std(a_vals) > 0:
            r, p = spearmanr(c_rates, a_vals)
            hierarchy_stats[name] = {
                "n_pairs": len(pairs),
                "spearman_r": float(r) if not np.isnan(r) else None,
                "spearman_p": float(p) if not np.isnan(p) else None,
                "mean_collision": float(np.mean(c_rates)),
                "mean_true_absorption": float(np.mean(a_vals)),
            }
            print(f"\n{name:12s}: n={len(pairs)}, r={r:.4f}, mean_collision={np.mean(c_rates):.4f}, mean_absorption={np.mean(a_vals):.4f}")

    # Case pairs (no true absorption computed, just report collision)
    if case_pair_results:
        c_rates = np.array([p["collision_rate"] for p in case_pair_results])
        hierarchy_stats["case"] = {
            "n_pairs": len(case_pair_results),
            "mean_collision": float(np.mean(c_rates)),
            "note": "No true absorption computed (case pairs use different semantic relationship)",
        }
        print(f"\ncase        : n={len(case_pair_results)}, mean_collision={np.mean(c_rates):.4f} (no absorption computed)")

    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": MODEL_NAME, "sae_id": SAE_ID, "layer": LAYER,
            "dataset_samples": DATASET_SAMPLES, "topk": TOPK, "seed": SEED,
        },
        "overall": {
            "n_valid_pairs": len(valid_pairs),
            "n_case_pairs": len(case_only_pairs),
            "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
            "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else None,
            "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
            "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
            "bootstrap_ci_95": [float(ci_lower), float(ci_upper)] if ci_lower is not None else None,
            "n_bootstrap_valid": n_valid,
        },
        "hierarchy_stats": hierarchy_stats,
        "pair_results": {
            "numbers": number_pairs,
            "punctuation": punct_pairs,
            "case": case_pair_results,
        },
        "pass_criteria": {
            "r_ge_0.3": bool(spearman_r >= 0.3),
            "ci_excludes_0": bool(ci_lower is not None and ci_lower > 0),
        },
        "runtime_seconds": time.time() - start_time,
    }

    results_file = RESULTS_DIR / "f4_collision_correlation_results.json"
    results_file.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {results_file}")

    print("\n" + "=" * 60)
    print("COLLISION RATE - ABSORPTION RATE CORRELATION (FULL)")
    print("=" * 60)
    print(f"Valid pairs (with absorption): {len(valid_pairs)}")
    print(f"Case pairs (no absorption): {len(case_only_pairs)}")
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")
    if ci_lower is not None:
        print(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"\nPass criteria:")
    print(f"  r >= 0.3: {spearman_r:.4f} >= 0.3 -> {'PASS' if spearman_r >= 0.3 else 'FAIL'}")
    print(f"  CI excludes 0: {'PASS' if ci_lower is not None and ci_lower > 0 else 'FAIL'}")
    print("=" * 60)

    passed = spearman_r >= 0.3 and ci_lower is not None and ci_lower > 0
    mark_done(
        status="success" if passed else "partial",
        summary=f"Spearman r={spearman_r:.4f}, n_pairs={len(valid_pairs)}, CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
    )
    return output


if __name__ == "__main__":
    main()
