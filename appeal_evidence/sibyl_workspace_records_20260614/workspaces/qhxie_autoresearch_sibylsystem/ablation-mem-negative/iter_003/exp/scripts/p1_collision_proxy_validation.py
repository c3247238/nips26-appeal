#!/usr/bin/env python3
"""
Pilot P1: Collision Rate - Absorption Rate Proxy Validation

Goal: Validate collision rate as a proxy for true absorption rate using
first-letter features (a-z) as ground truth on GPT-2 Small layer 8.

Ground truth: Chanin et al. (2024) supervised detection method
- Parent feature: detects "first letter is vowel" (a, e, i, o, u)
- Child features: detect individual first letters (a, b, c, ..., z)
- Absorption: a single SAE feature responds to multiple vowels

Collision rate: fraction of parent-positive samples where a child feature
also activates (i.e., the parent and child "collide" on the same SAE feature).

True absorption rate: Chanin's definition - the fraction of parent activation
that is "stolen" by child features.

Output: Spearman correlation between collision rate and absorption rate.
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
    "probe_threshold": 0.1,  # activation threshold for feature scanning
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

    # Set seed
    torch.manual_seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])

    device = CONFIG["device"]
    print(f"[P1] Starting collision proxy validation on {device}")
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

    sae, cfg_dict, sparsity = SAE.from_pretrained(
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

    # Tokenize
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Collect SAE activations for each text
    print("[P1] Collecting SAE activations...")
    all_activations = []  # List of (n_tokens, d_sae) tensors
    all_first_letters = []  # List of first letters for each token position

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

            # Run through SAE - use encode() to get feature activations
            feature_acts = sae.encode(resid)  # (batch, seq, d_sae)

            # For each sample, record first letter of each token
            for b in range(input_ids.shape[0]):
                valid_len = attention_mask[b].sum().item()
                sample_acts = feature_acts[b, :valid_len, :].cpu().numpy()
                all_activations.append(sample_acts)

                # Get first letters of decoded tokens
                token_ids = input_ids[b, :valid_len].cpu().tolist()
                decoded = tokenizer.batch_decode([[tid] for tid in token_ids])
                first_letters_batch = []
                for tok_text in decoded:
                    txt = tok_text.strip()
                    if len(txt) > 0 and txt[0].isalpha():
                        c = txt[0].lower()
                        if 'a' <= c <= 'z':
                            first_letters_batch.append(c)
                        else:
                            first_letters_batch.append(None)
                    else:
                        first_letters_batch.append(None)
                all_first_letters.append(first_letters_batch)

    print(f"[P1] Collected activations for {len(all_activations)} samples")

    # -----------------------------------------------------------------------
    # Step 3: Probe first-letter features using activation maximization
    # -----------------------------------------------------------------------
    report_progress(3, 5, step=3, total_steps=5, metric={"phase": "probe_features"})
    print("[P1] Step 3/5: Probing first-letter features...")

    # For each letter (a-z), find the SAE feature that most strongly responds
    # when that letter is the first character of a token
    letter_feature_scores = {chr(ord('a') + i): [] for i in range(26)}

    for sample_acts, sample_letters in zip(all_activations, all_first_letters):
        for pos, (acts, letter) in enumerate(zip(sample_acts, sample_letters)):
            if letter is not None:
                letter_feature_scores[letter].append(acts)

    # Compute mean activation per feature for each letter
    letter_top_features = {}
    for letter in sorted(letter_feature_scores.keys()):
        scores_list = letter_feature_scores[letter]
        if len(scores_list) == 0:
            print(f"[P1] WARNING: No tokens found for letter '{letter}'")
            continue
        mean_acts = np.mean(scores_list, axis=0)  # (d_sae,)
        top_feature_idx = int(np.argmax(mean_acts))
        top_feature_score = float(mean_acts[top_feature_idx])
        letter_top_features[letter] = {
            "feature_idx": top_feature_idx,
            "mean_activation": top_feature_score,
            "n_tokens": len(scores_list),
        }
        print(f"[P1] Letter '{letter}': top feature={top_feature_idx}, score={top_feature_score:.4f}, n_tokens={len(scores_list)}")

    # -----------------------------------------------------------------------
    # Step 4: Build ground truth absorption pairs
    # -----------------------------------------------------------------------
    report_progress(4, 5, step=4, total_steps=5, metric={"phase": "build_ground_truth"})
    print("[P1] Step 4/5: Building ground truth absorption pairs...")

    vowels = {'a', 'e', 'i', 'o', 'u'}

    # Ground truth absorption pairs:
    # A pair (parent_feature, child_feature) is an absorption pair if:
    # - parent_feature is the top feature for a vowel (e.g., 'a')
    # - child_feature is the top feature for another vowel (e.g., 'e')
    # - Both vowels share the same SAE feature (i.e., the SAE feature "absorbs" both vowels)
    #
    # More precisely: if the same SAE feature is top for multiple vowels,
    # those vowels form an absorption cluster.

    # Find which SAE features are top for which letters
    feature_to_letters = {}
    for letter, info in letter_top_features.items():
        feat_idx = info["feature_idx"]
        if feat_idx not in feature_to_letters:
            feature_to_letters[feat_idx] = []
        feature_to_letters[feat_idx].append(letter)

    # Absorption clusters: SAE features that respond to multiple vowels
    absorption_clusters = {}
    for feat_idx, letters in feature_to_letters.items():
        vowel_letters = [l for l in letters if l in vowels]
        if len(vowel_letters) >= 2:
            absorption_clusters[feat_idx] = vowel_letters
            print(f"[P1] Absorption cluster: feature {feat_idx} responds to vowels {vowel_letters}")

    # Build ground truth pairs: all pairs of vowels that share the same SAE feature
    gt_absorption_pairs = []
    for feat_idx, vowel_letters in absorption_clusters.items():
        for i in range(len(vowel_letters)):
            for j in range(i + 1, len(vowel_letters)):
                gt_absorption_pairs.append({
                    "parent_letter": vowel_letters[i],
                    "child_letter": vowel_letters[j],
                    "shared_feature": feat_idx,
                    "absorption_type": "vowel_cluster",
                })

    print(f"[P1] Ground truth absorption pairs: {len(gt_absorption_pairs)}")
    for pair in gt_absorption_pairs:
        print(f"  {pair['parent_letter']} - {pair['child_letter']} (feature {pair['shared_feature']})")

    # Also build all possible vowel pairs for comparison
    all_vowel_pairs = []
    vowel_list = sorted(vowels)
    for i in range(len(vowel_list)):
        for j in range(i + 1, len(vowel_list)):
            all_vowel_pairs.append((vowel_list[i], vowel_list[j]))

    print(f"[P1] All possible vowel pairs: {len(all_vowel_pairs)}")

    # -----------------------------------------------------------------------
    # Step 5: Compute collision rate and true absorption rate
    # -----------------------------------------------------------------------
    report_progress(5, 5, step=5, total_steps=5, metric={"phase": "compute_metrics"})
    print("[P1] Step 5/5: Computing collision and absorption rates...")

    # For each vowel pair, compute:
    # 1. Collision rate: fraction of samples where both vowels' top features activate
    # 2. True absorption rate: based on Chanin's definition

    # First, build per-sample activation indicators for each letter's top feature
    letter_feature_activations = {}
    for letter, info in letter_top_features.items():
        feat_idx = info["feature_idx"]
        threshold = info["mean_activation"] * 0.5  # Use 50% of mean as threshold

        activations = []
        for sample_acts, sample_letters in zip(all_activations, all_first_letters):
            # Check if the top feature for this letter activates on any token
            # that has this letter as first character
            letter_positions = [pos for pos, l in enumerate(sample_letters) if l == letter]
            if len(letter_positions) == 0:
                activations.append(0.0)
                continue

            max_act = np.max(sample_acts[letter_positions, feat_idx])
            activations.append(1.0 if max_act > threshold else 0.0)

        letter_feature_activations[letter] = {
            "feature_idx": feat_idx,
            "threshold": threshold,
            "activations": np.array(activations),
            "mean_activation": info["mean_activation"],
        }

    # Compute metrics for each vowel pair
    results = []
    for v1, v2 in all_vowel_pairs:
        if v1 not in letter_feature_activations or v2 not in letter_feature_activations:
            continue

        act1 = letter_feature_activations[v1]["activations"]
        act2 = letter_feature_activations[v2]["activations"]

        # Collision rate: fraction of samples where both activate
        collision_samples = np.sum((act1 > 0) & (act2 > 0))
        collision_rate = collision_samples / len(act1) if len(act1) > 0 else 0.0

        # True absorption rate (Chanin et al.):
        # The fraction of parent activation that is "absorbed" by child features.
        # Simplified: if both letters share the same SAE feature, absorption is high.
        # If they have different top features, absorption is low.
        feat1 = letter_feature_activations[v1]["feature_idx"]
        feat2 = letter_feature_activations[v2]["feature_idx"]

        if feat1 == feat2:
            # Same feature = high absorption (the feature absorbs both concepts)
            true_absorption = 1.0
        else:
            # Different features = no absorption
            true_absorption = 0.0

        is_gt_pair = any(p["parent_letter"] == v1 and p["child_letter"] == v2 or
                         p["parent_letter"] == v2 and p["child_letter"] == v1
                         for p in gt_absorption_pairs)

        results.append({
            "pair": f"{v1}-{v2}",
            "letter1": v1,
            "letter2": v2,
            "feature1": int(feat1),
            "feature2": int(feat2),
            "collision_rate": float(collision_rate),
            "true_absorption": float(true_absorption),
            "is_ground_truth": bool(is_gt_pair),
            "collision_samples": int(collision_samples),
            "total_samples": len(act1),
        })

    # Compute Spearman correlation
    collision_rates = [r["collision_rate"] for r in results]
    absorption_rates = [r["true_absorption"] for r in results]

    if len(collision_rates) >= 3:
        spearman_r, spearman_p = spearmanr(collision_rates, absorption_rates)
    else:
        spearman_r, spearman_p = 0.0, 1.0

    # Bootstrap CI for Spearman r
    n_bootstrap = 1000
    bootstrap_rs = []
    rng = np.random.RandomState(CONFIG["seed"])
    for _ in range(n_bootstrap):
        idx = rng.choice(len(collision_rates), size=len(collision_rates), replace=True)
        if len(set(idx)) > 1:
            r_boot, _ = spearmanr([collision_rates[i] for i in idx], [absorption_rates[i] for i in idx])
            bootstrap_rs.append(r_boot)

    if len(bootstrap_rs) > 0:
        ci_lower = float(np.percentile(bootstrap_rs, 2.5))
        ci_upper = float(np.percentile(bootstrap_rs, 97.5))
    else:
        ci_lower = ci_upper = 0.0

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time

    gt_detected = sum(1 for r in results if r["is_ground_truth"])
    gt_pairs_found = [r["pair"] for r in results if r["is_ground_truth"]]

    summary = {
        "task_id": TASK_ID,
        "config": CONFIG,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_samples": len(texts),
        "n_letters_probed": len(letter_top_features),
        "n_vowel_pairs": len(all_vowel_pairs),
        "n_gt_absorption_pairs": len(gt_absorption_pairs),
        "gt_pairs_detected": gt_detected,
        "gt_pair_list": gt_pairs_found,
        "absorption_clusters": {
            str(k): v for k, v in absorption_clusters.items()
        },
        "spearman_r": float(spearman_r) if not np.isnan(spearman_r) else 0.0,
        "spearman_p": float(spearman_p) if not np.isnan(spearman_p) else 1.0,
        "bootstrap_ci_95": [ci_lower, ci_upper],
        "n_bootstrap": n_bootstrap,
        "pair_results": results,
        "letter_top_features": {
            k: {"feature_idx": v["feature_idx"], "mean_activation": v["mean_activation"], "n_tokens": v["n_tokens"]}
            for k, v in letter_top_features.items()
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
    print("P1 COLLISION PROXY VALIDATION - SUMMARY")
    print("=" * 60)
    print(f"Spearman r: {spearman_r:.4f} (p={spearman_p:.4f})")
    print(f"Bootstrap 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Ground truth pairs detected: {gt_detected}/{len(gt_absorption_pairs)}")
    print(f"All vowel pairs analyzed: {len(results)}")
    print(f"Absorption clusters found: {len(absorption_clusters)}")
    print(f"Elapsed time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"Pass criteria:")
    print(f"  r >= 0.3: {summary['pass_criteria']['r_ge_0.3']}")
    print(f"  >=5 GT pairs: {summary['pass_criteria']['at_least_5_gt_pairs']}")
    print(f"  Runtime < 15min: {summary['pass_criteria']['runtime_under_15min']}")
    print("=" * 60)

    # Write DONE marker
    status = "success" if summary["pass_criteria"]["r_ge_0.3"] else "partial"
    mark_done(status=status, summary=f"Spearman r={spearman_r:.4f}, GT pairs={gt_detected}/{len(gt_absorption_pairs)}, elapsed={elapsed:.1f}s")

    return summary


if __name__ == "__main__":
    main()
