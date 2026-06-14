#!/usr/bin/env python3
"""
Pilot P1: CAAB Feasibility
Load pre-trained SAE + train small TopK SAE on GPT-2 Small layer 8.
Run simplified absorption detection on first-letter features (a-z).
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import SAE, LanguageModelSAERunnerConfig, TopKTrainingSAEConfig, LanguageModelSAETrainingRunner
from sae_lens.config import LoggingConfig

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "p1_caab"

# Write PID file
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
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

# ── Set seeds ──────────────────────────────────────────────────────────────
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ── Helper: find first-letter features via simple probing ──────────────────
def find_first_letter_features(model, sae, tokenizer, device, n_samples=50):
    """Find SAE features that correspond to first-letter concepts."""
    letters = [chr(ord('a') + i) for i in range(26)]

    # Build dataset: sentences with words starting with each letter
    all_prompts = []
    all_labels = []
    for letter in letters:
        words = [f"{letter}{w}" for w in ["pple", "nimal", "lpha", "rt", "nt", "rea", "lone", "mazing", "wesome", "ble", "ctive", "rticle", "ward"]]
        for word in words[:4]:
            prompt = f"The word '{word}'"
            all_prompts.append(prompt)
            all_labels.append(ord(letter) - ord('a'))

    # Tokenize
    tokenized = tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    input_ids = tokenized["input_ids"].to(device)

    # Get SAE activations
    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        acts = cache[hook_name]  # [batch, pos, d_model]
        sae_acts = sae.encode(acts)  # [batch, pos, d_sae]
        # For each sample, take the max activation across positions
        sample_acts = sae_acts.max(dim=1).values  # [batch, d_sae]

    # Find features with highest correlation to letter labels
    labels = torch.tensor(all_labels, device=device)
    d_sae = sample_acts.shape[1]

    best_features = []
    for letter_idx in range(26):
        mask = labels == letter_idx
        if mask.sum() == 0:
            continue
        letter_acts = sample_acts[mask].mean(dim=0)  # [d_sae]
        other_acts = sample_acts[~mask].mean(dim=0)  # [d_sae]

        # Score: difference in mean activation
        scores = letter_acts - other_acts
        best_feat = scores.argmax().item()
        best_score = scores[best_feat].item()

        best_features.append({
            "letter": letters[letter_idx],
            "feature_idx": best_feat,
            "score": best_score,
        })

    return best_features


# ── Helper: measure absorption via feature collision ───────────────────────
def measure_absorption(model, sae, tokenizer, device, feature_infos, n_test=100):
    """
    Measure absorption by checking if first-letter features are 'absorbed'
    into more general features. A feature is absorbed if:
    1. Its activation is suppressed when related features fire
    2. Multiple letters share the same SAE feature (collision)
    """
    # Build test prompts for all letters
    letters = [f["letter"] for f in feature_infos]
    test_prompts = []
    for letter in letters:
        words = [f"{letter}{w}" for w in ["pple", "nimal", "lpha", "rt", "nt"]]
        for word in words[:3]:
            test_prompts.append(f"The word '{word}'")

    tokenized = tokenizer(test_prompts, return_tensors="pt", padding=True, truncation=True, max_length=32)
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(input_ids, names_filter=[hook_name])
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)  # [batch, pos, d_sae]
        # Max pool across positions
        sample_acts = sae_acts.max(dim=1).values  # [batch, d_sae]

    # For each first-letter feature, measure its activation on its own prompts
    # vs. on prompts for other letters
    absorption_scores = []
    for i, feat_info in enumerate(feature_infos):
        feat_idx = feat_info["feature_idx"]
        letter = feat_info["letter"]

        # Get prompts for this letter (every 3rd prompt starting at i*3)
        start_idx = i * 3
        end_idx = start_idx + 3
        own_acts = sample_acts[start_idx:end_idx, feat_idx].mean().item()

        # Get prompts for other letters
        other_indices = list(range(0, start_idx)) + list(range(end_idx, len(test_prompts)))
        other_acts = sample_acts[other_indices, feat_idx].mean().item()

        # Absorption score: how much is the feature suppressed on non-target prompts?
        # High own_acts + low other_acts = specific feature (not absorbed)
        # Low own_acts = potentially absorbed
        specificity = own_acts / (other_acts + 1e-8)
        absorption_scores.append({
            "letter": letter,
            "feature_idx": feat_idx,
            "own_activation": own_acts,
            "other_activation": other_acts,
            "specificity": specificity,
        })

    return absorption_scores


# ── Main experiment ────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    report_progress(0, 5, step=1, total_steps=5, metric={"phase": "init"})

    results = {
        "task_id": TASK_ID,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "seed": SEED,
        "start_time": datetime.now().isoformat(),
        "pretrained_sae": {},
        "topk_sae": {},
    }

    # ── Step 1: Load GPT-2 Small ───────────────────────────────────────────
    print("[1/5] Loading GPT-2 Small...")
    report_progress(1, 5, step=1, total_steps=5, metric={"phase": "load_model"})

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print(f"  Model loaded on {DEVICE}")

    # ── Step 2: Load pre-trained SAE ───────────────────────────────────────
    print("[2/5] Loading pre-trained SAE...")
    report_progress(2, 5, step=2, total_steps=5, metric={"phase": "load_sae"})

    try:
        sae_pretrained = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device=DEVICE,
        )
        print("  Loaded pre-trained GPT-2 SAE")
        results["pretrained_sae"]["source"] = "gpt2-small-res-jb"
        results["pretrained_sae"]["hook_name"] = sae_pretrained.cfg.hook_name
        results["pretrained_sae"]["d_sae"] = sae_pretrained.cfg.d_sae
    except Exception as e:
        print(f"  Could not load pre-trained SAE: {e}")
        sae_pretrained = None
        results["pretrained_sae"]["source"] = "none"
        results["pretrained_sae"]["error"] = str(e)

    # ── Step 3: Train a small TopK SAE on GPT-2 Small layer 8 ──────────────
    print("[3/5] Training TopK SAE on GPT-2 Small layer 8...")
    report_progress(3, 5, step=3, total_steps=5, metric={"phase": "train_topk"})

    # Use a small config for pilot - quick training
    topk_cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            d_in=model.cfg.d_model,  # 768
            d_sae=3072,  # 4x expansion
            k=25,
            device=DEVICE,
        ),
        model_name="gpt2-small",
        hook_name="blocks.8.hook_resid_pre",
        dataset_path="monology/pile-uncopyrighted",  # Default dataset used by SAELens
        training_tokens=500_000,  # Small for pilot
        train_batch_size_tokens=2048,
        store_batch_size_prompts=16,
        context_size=128,
        lr=3e-4,
        device=DEVICE,
        seed=SEED,
        logger=LoggingConfig(log_to_wandb=False),
        verbose=False,
        n_checkpoints=0,
        save_final_checkpoint=True,
        output_path=str(RESULTS_DIR / "topk_sae_output"),
    )

    runner = LanguageModelSAETrainingRunner(topk_cfg)
    sae_topk = runner.run()

    print("  TopK SAE training complete")
    results["topk_sae"]["d_sae"] = sae_topk.cfg.d_sae
    results["topk_sae"]["k"] = sae_topk.cfg.k

    # ── Step 4: Find first-letter features ─────────────────────────────────
    print("[4/5] Finding first-letter features...")
    report_progress(4, 5, step=4, total_steps=5, metric={"phase": "find_features"})

    topk_features = find_first_letter_features(model, sae_topk, tokenizer, DEVICE)
    results["topk_sae"]["first_letter_features"] = topk_features
    results["topk_sae"]["num_found"] = len(topk_features)
    print(f"  Found {len(topk_features)} first-letter features in TopK SAE")

    if sae_pretrained is not None:
        pretrained_features = find_first_letter_features(model, sae_pretrained, tokenizer, DEVICE)
        results["pretrained_sae"]["first_letter_features"] = pretrained_features
        results["pretrained_sae"]["num_found"] = len(pretrained_features)
        print(f"  Found {len(pretrained_features)} first-letter features in pre-trained SAE")

    # ── Step 5: Measure absorption patterns ────────────────────────────────
    print("[5/5] Measuring absorption patterns...")
    report_progress(5, 5, step=5, total_steps=5, metric={"phase": "measure_absorption"})

    # Measure absorption for TopK SAE
    topk_absorption = measure_absorption(model, sae_topk, tokenizer, DEVICE, topk_features)
    results["topk_sae"]["absorption_scores"] = topk_absorption

    # Compute collision rate (multiple letters sharing same feature)
    topk_feat_ids = [f["feature_idx"] for f in topk_features]
    topk_unique = len(set(topk_feat_ids))
    topk_collision_rate = (len(topk_features) - topk_unique) / len(topk_features) if topk_features else 0
    results["topk_sae"]["collision_rate"] = topk_collision_rate
    results["topk_sae"]["unique_features"] = topk_unique

    # Compute mean specificity (higher = less absorbed)
    topk_mean_specificity = np.mean([a["specificity"] for a in topk_absorption])
    results["topk_sae"]["mean_specificity"] = topk_mean_specificity

    print(f"  TopK: {topk_unique}/{len(topk_features)} unique features, collision rate: {topk_collision_rate:.2%}")
    print(f"  TopK: mean specificity = {topk_mean_specificity:.3f}")

    if sae_pretrained is not None:
        pretrained_absorption = measure_absorption(model, sae_pretrained, tokenizer, DEVICE, pretrained_features)
        results["pretrained_sae"]["absorption_scores"] = pretrained_absorption

        pretrained_feat_ids = [f["feature_idx"] for f in pretrained_features]
        pretrained_unique = len(set(pretrained_feat_ids))
        pretrained_collision_rate = (len(pretrained_features) - pretrained_unique) / len(pretrained_features) if pretrained_features else 0
        results["pretrained_sae"]["collision_rate"] = pretrained_collision_rate
        results["pretrained_sae"]["unique_features"] = pretrained_unique

        pretrained_mean_specificity = np.mean([a["specificity"] for a in pretrained_absorption])
        results["pretrained_sae"]["mean_specificity"] = pretrained_mean_specificity

        print(f"  Pretrained: {pretrained_unique}/{len(pretrained_features)} unique features, collision rate: {pretrained_collision_rate:.2%}")
        print(f"  Pretrained: mean specificity = {pretrained_mean_specificity:.3f}")

        # Compute difference
        diff_collision = abs(topk_collision_rate - pretrained_collision_rate)
        diff_specificity = abs(topk_mean_specificity - pretrained_mean_specificity)
        results["collision_rate_diff"] = diff_collision
        results["specificity_diff"] = diff_specificity
        print(f"  Collision rate difference: {diff_collision:.2%}")
        print(f"  Specificity difference: {diff_specificity:.3f}")

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = RESULTS_DIR / "p1_caab_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    # Summary for DONE marker
    summary = (
        f"TopK: collision={topk_collision_rate:.2%}, specificity={topk_mean_specificity:.3f}, "
        f"found {len(topk_features)} features, elapsed={elapsed:.1f}s"
    )
    if sae_pretrained is not None:
        summary += (
            f" | Pretrained: collision={pretrained_collision_rate:.2%}, "
            f"specificity={pretrained_mean_specificity:.3f}"
        )

    mark_done(status="success", summary=summary)
    print("\nPilot P1 complete!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
