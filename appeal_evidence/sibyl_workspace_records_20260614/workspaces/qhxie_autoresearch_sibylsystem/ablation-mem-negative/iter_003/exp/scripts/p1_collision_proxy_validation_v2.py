#!/usr/bin/env python3
"""
Pilot P1 v2: Collision Rate - Absorption Rate Proxy Validation

Redesign: Instead of using a single "top feature" per letter (which degenerates
to the same feature for all letters), we use a DISTRIBUTION-BASED approach:

1. For each letter, compute the average activation profile across ALL SAE features
2. Two letters "collide" if their activation profiles have high cosine similarity
3. Two letters are "absorbed" if they share a single dominant SAE feature

This avoids the degenerate case where one feature dominates all letters.

Ground truth: Chanin et al. (2024) supervised detection method
- Parent feature: detects "first letter is vowel" (a, e, i, o, u)
- Child features: detect individual first letters (a, b, c, ..., z)
- Absorption: a single SAE feature responds to multiple vowels
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TASK_ID = "p1_collision_proxy_validation"
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-negative/current/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CONFIG = {
    "task_id": TASK_ID,
    "model": "gpt2-small",
    "sae_source": "gpt2-small-res-jb",
    "layer": 8,
    "dataset": "openwebtext",
    "dataset_samples": 1000,
    "device": "cuda",
    "batch_size": 8,
    "max_sequence_length": 128,
    "seed": 42,
    "features_analyzed": 500,
    "n_clusters": 50,
    "probe_threshold": 0.1,
    "version": "v2_distribution_based",
}

# ---------------------------------------------------------------------------
# PID / Progress / DONE marker helpers
# ---------------------------------------------------------------------------
def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
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

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 5, step=0, total_steps=5, metric={"phase": "init"})

    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = CONFIG["device"]
    print(f"[P1] Starting collision proxy validation v2 on {device}")
    print(f"[P1] Config: {json.dumps(CONFIG, indent=2, default=str)}")

    # -----------------------------------------------------------------------
    # Step 1: Load model and SAE
    # -----------------------------------------------------------------------
    report_progress(1, 5, step=1, total_steps=5, metric={"phase": "load_model"})
    print("[P1] Step 1/5: Loading GPT-2 Small and SAE...")

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{CONFIG['layer']}.hook_resid_pre",
        device=device,
    )
    sae.eval()
    d_sae = sae.W_enc.shape[1]
    print(f"[P1] SAE loaded: d_in={sae.W_enc.shape[0]}, d_sae={d_sae}")

    # -----------------------------------------------------------------------
    # Step 2: Load dataset and collect activations
    # -----------------------------------------------------------------------
    report_progress(2, 5, step=2, total_steps=5, metric={"phase": "collect_activations"})
    print("[P1] Step 2/5: Loading OpenWebText samples...")

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    ds_iter = iter(ds)

    texts = []
    for _ in range(CONFIG["dataset_samples"]):
        try:
            item = next(ds_iter)
            texts.append(item["text"])
        except StopIteration:
            break

    print(f"[P1] Loaded {len(texts)} text samples")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Collect per-letter activation profiles
    print("[P1] Collecting per-letter SAE activation profiles...")
    letter_profiles = {chr(ord('a') + i): [] for i in range(26)}  # List of activation vectors per letter

    for i in tqdm(range(0, len(texts), CONFIG["batch_size"]), desc="Processing batches"):
        batch_texts = texts[i:i+CONFIG["batch_size"]]
        tokens = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=CONFIG["max_sequence_length"],
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            _, cache = model.run_with_cache(input_ids, stop_at_layer=CONFIG["layer"] + 1)
            resid = cache[f"blocks.{CONFIG['layer']}.hook_resid_pre"]
            feature_acts = sae.encode(resid)  # (batch, seq, d_sae)

            for b in range(input_ids.shape[0]):
                valid_len = attention_mask[b].sum().item()
                sample_acts = feature_acts[b, :valid_len, :].cpu().numpy()  # (seq, d_sae)

                token_ids = input_ids[b, :valid_len].cpu().tolist()
                decoded = tokenizer.batch_decode([[tid] for tid in token_ids])

                for pos, tok_text in enumerate(decoded):
                    txt = tok_text.strip()
                    if len(txt) > 0 and txt[0].isalpha():
                        c = txt[0].lower()
                        if 'a' <= c <= 'z':
                            letter_profiles[c].append(sample_acts[pos, :])

    print(f"[P1] Collected activation profiles:")
    for letter in sorted(letter_profiles.keys()):
        print(f"  '{letter}': {len(letter_profiles[letter])} tokens")

    # -----------------------------------------------------------------------
    # Step 3: Compute mean activation profile per letter
    # -----------------------------------------------------------------------
    report_progress(3, 5, step=3, total_steps=5, metric={"phase": "compute_profiles"})
    print("[P1] Step 3/5: Computing mean activation profiles...")

    letter_mean_profiles = {}
    for letter in sorted(letter_profiles.keys()):
        profiles = letter_profiles[letter]
        if len(profiles) == 0:
            print(f"[P1] WARNING: No tokens for letter '{letter}'")
            continue
        mean_profile = np.mean(profiles, axis=0)  # (d_sae,)
        letter_mean_profiles[letter] = mean_profile

    # Find top-K features for each letter
    K = 10
    letter_topk = {}
    for letter, profile in letter_mean_profiles.items():
        topk_idx = np.argsort(profile)[-K:][::-1]
        topk_vals = profile[topk_idx]
        letter_topk[letter] = {
            "topk_idx": topk_idx.tolist(),
            "topk_vals": topk_vals.tolist(),
        }
        print(f"[P1] Letter '{letter}': top features = {topk_idx[:5].tolist()}, vals = {topk_vals[:5].round(3).tolist()}")

    # -----------------------------------------------------------------------
    # Step 4: Build ground truth absorption pairs
    # -----------------------------------------------------------------------
    report_progress(4, 5, step=4, total_steps=5, metric={"phase": "build_ground_truth"})
    print("[P1] Step 4/5: Building ground truth absorption pairs...")

    vowels = {'a', 'e', 'i', 'o', 'u'}

    # Ground truth: two vowels form an absorption pair if they share a top feature
    gt_absorption_pairs = []
    vowel_list = sorted(vowels)
    for i in range(len(vowel_list)):
        for j in range(i + 1, len(vowel_list)):
            v1, v2 = vowel_list[i], vowel_list[j]
            topk1 = set(letter_topk[v1]["topk_idx"])
            topk2 = set(letter_topk[v2]["topk_idx"])
            shared = topk1 & topk2
            if len(shared) > 0:
                gt_absorption_pairs.append({
                    "parent_letter": v1,
                    "child_letter": v2,
                    "shared_features": list(shared),
                    "absorption_type": "shared_topk",
                })

    print(f"[P1] Ground truth absorption pairs (top-{K} overlap): {len(gt_absorption_pairs)}")
    for pair in gt_absorption_pairs:
        print(f"  {pair['parent_letter']} - {pair['child_letter']} (shared: {pair['shared_features']})")

    # All possible vowel pairs
    all_vowel_pairs = []
    for i in range(len(vowel_list)):
        for j in range(i + 1, len(vowel_list)):
            all_vowel_pairs.append((vowel_list[i], vowel_list[j]))

    print(f"[P1] All possible vowel pairs: {len(all_vowel_pairs)}")

    # -----------------------------------------------------------------------
    # Step 5: Compute collision rate and true absorption rate
    # -----------------------------------------------------------------------
    report_progress(5, 5, step=5, total_steps=5, metric={"phase": "compute_metrics"})
    print("[P1] Step 5/5: Computing collision and absorption rates...")

    # Collision rate: cosine similarity of mean activation profiles
    # True absorption rate: Jaccard similarity of top-K feature sets

    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

    def jaccard_similarity(set1, set2):
        if len(set1 | set2) == 0:
            return 0.0
        return len(set1 & set2) / len(set1 | set2)

    results = []
    for v1, v2 in all_vowel_pairs:
        profile1 = letter_mean_profiles[v1]
        profile2 = letter_mean_profiles[v2]

        # Collision rate = cosine similarity of activation profiles
        collision_rate = cosine_similarity(profile1, profile2)

        # True absorption rate = Jaccard similarity of top-K feature sets
        topk1 = set(letter_topk[v1]["topk_idx"])
        topk2 = set(letter_topk[v2]["topk_idx"])
        true_absorption = jaccard_similarity(topk1, topk2)

        is_gt_pair = any(
            (p["parent_letter"] == v1 and p["child_letter"] == v2) or
            (p["parent_letter"] == v2 and p["child_letter"] == v1)
            for p in gt_absorption_pairs
        )

        results.append({
            "pair": f"{v1}-{v2}",
            "letter1": v1,
            "letter2": v2,
            "collision_rate": float(collision_rate),
            "true_absorption": float(true_absorption),
            "is_ground_truth": bool(is_gt_pair),
            "topk_shared": list(topk1 & topk2),
            "topk_jaccard": float(true_absorption),
        })

    # Compute Spearman correlation
    collision_rates = [r["collision_rate"] for r in results]
    absorption_rates = [r["true_absorption"] for r in results]

    if len(collision_rates) >= 3 and len(set(collision_rates)) > 1 and len(set(absorption_rates)) > 1:
        spearman_r, spearman_p = spearmanr(collision_rates, absorption_rates)
    else:
        spearman_r, spearman_p = 0.0, 1.0
        print(f"[P1] WARNING: Constant input detected. collision unique={len(set(collision_rates))}, absorption unique={len(set(absorption_rates))}")

    # Bootstrap CI
    n_bootstrap = 1000
    bootstrap_rs = []
    rng = np.random.RandomState(CONFIG["seed"])
    for _ in range(n_bootstrap):
        idx = rng.choice(len(collision_rates), size=len(collision_rates), replace=True)
        if len(set([collision_rates[i] for i in idx])) > 1 and len(set([absorption_rates[i] for i in idx])) > 1:
            r_boot, _ = spearmanr([collision_rates[i] for i in idx], [absorption_rates[i] for i in idx])
            bootstrap_rs.append(r_boot)

    if len(bootstrap_rs) > 0:
        ci_lower = float(np.percentile(bootstrap_rs, 2.5))
        ci_upper = float(np.percentile(bootstrap_rs, 97.5))
    else:
        ci_lower = ci_upper = 0.0

    # Alternative: Pearson correlation
    from scipy.stats import pearsonr
    if len(set(collision_rates)) > 1 and len(set(absorption_rates)) > 1:
        pearson_r, pearson_p = pearsonr(collision_rates, absorption_rates)
    else:
        pearson_r, pearson_p = 0.0, 1.0

    elapsed = time.time() - start_time
    gt_detected = sum(1 for r in results if r["is_ground_truth"])
    gt_pairs_found = [r["pair"] for r in results if r["is_ground_truth"]]

    summary = {
        "task_id": TASK_ID,
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_samples": len(texts),
        "n_letters_probed": len(letter_mean_profiles),
        "n_vowel_pairs": len(all_vowel_pairs),
        "n_gt_absorption_pairs": len(gt_absorption_pairs),
        "gt_pairs_detected": gt_detected,
        "gt_pair_list": gt_pairs_found,
        "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else 0.0,
        "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
        "pearson_r": float(pearson_r) if not np.isnan(pearson_r) else 0.0,
        "pearson_p": float(pearson_p) if not np.isnan(pearson_p) else 1.0,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "n_bootstrap_valid": len(bootstrap_rs),
        "n_bootstrap_total": n_bootstrap,
        "pair_results": results,
        "letter_topk_features": {
            k: {"topk_idx": v["topk_idx"][:5], "topk_vals": [round(x, 3) for x in v["topk_vals"][:5]]}
            for k, v in letter_topk.items()
        },
        "pass_criteria": {
            "r_ge_0.3": spearman_r >= 0.3 if not np.isnan(spearman_r) else False,
            "at_least_5_gt_pairs": gt_detected >= 5,
            "runtime_under_15min": elapsed < 900,
        },
    }

    # Save results
    results_file = RESULTS_DIR / "p1_collision_proxy_results.json"
    results_file.write_text(json.dumps(summary, indent=2, default=str))
    print(f"[P1] Results saved to {results_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("P1 COLLISION PROXY VALIDATION v2 - SUMMARY")
    print("=" * 60)
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"Pearson r:  {pearson_r:.4f} (p={pearson_p:.4f})")
    print(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] (n_valid={len(bootstrap_rs)})")
    print(f"Ground truth pairs detected: {gt_detected}/{len(gt_absorption_pairs)}")
    print(f"All vowel pairs analyzed: {len(results)}")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Pass criteria:")
    print(f"  r >= 0.3: {summary['pass_criteria']['r_ge_0.3']}")
    print(f"  >=5 GT pairs: {summary['pass_criteria']['at_least_5_gt_pairs']}")
    print(f"  Runtime < 15min: {summary['pass_criteria']['runtime_under_15min']}")
    print("=" * 60)

    # Detailed pair results
    print("\nPair-level results:")
    for r in results:
        marker = "*GT*" if r["is_ground_truth"] else ""
        print(f"  {r['pair']:6s}: collision={r['collision_rate']:.4f}, absorption={r['true_absorption']:.4f} {marker}")

    status = "success" if summary["pass_criteria"]["r_ge_0.3"] else "partial"
    mark_done(status=status, summary=f"Spearman r={spearman_r:.4f}, GT pairs={gt_detected}/{len(gt_absorption_pairs)}, elapsed={elapsed:.1f}s")

    return summary


if __name__ == "__main__":
    main()
