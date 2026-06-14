#!/usr/bin/env python3
"""
Full E1: Cross-Architecture Absorption Benchmark (CAAB)
Systematic comparison of absorption patterns across SAE architectures on GPT-2 Small.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer
from sae_lens import (
    SAE, LanguageModelSAERunnerConfig, TopKTrainingSAEConfig,
    LanguageModelSAETrainingRunner
)
from sae_lens.config import LoggingConfig

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "f1_caab"

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


# ── Helper: find first-letter features ─────────────────────────────────────
def find_first_letter_features(model, sae, tokenizer, device):
    """Find SAE features that correspond to first-letter concepts."""
    letters = [chr(ord('a') + i) for i in range(26)]
    all_prompts = []
    all_labels = []
    for letter in letters:
        words = [f"{letter}{w}" for w in [
            "pple", "nimal", "lpha", "rt", "nt", "rea",
            "lone", "mazing", "wesome", "ble"
        ]]
        for word in words[:4]:
            all_prompts.append(f"The word '{word}'")
            all_labels.append(ord(letter) - ord('a'))

    tokenized = tokenizer(
        all_prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=32
    )
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, names_filter=[hook_name]
        )
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        sample_acts = sae_acts.max(dim=1).values

    labels = torch.tensor(all_labels, device=device)
    best_features = []
    for letter_idx in range(26):
        mask = labels == letter_idx
        if mask.sum() == 0:
            continue
        letter_acts = sample_acts[mask].mean(dim=0)
        other_acts = sample_acts[~mask].mean(dim=0)
        scores = letter_acts - other_acts
        best_feat = scores.argmax().item()
        best_score = scores[best_feat].item()
        best_features.append({
            "letter": letters[letter_idx],
            "feature_idx": best_feat,
            "score": best_score,
        })
    return best_features


# ── Helper: measure absorption ─────────────────────────────────────────────
def measure_absorption(model, sae, tokenizer, device, feature_infos):
    """Measure absorption via feature collision and specificity."""
    letters = [f["letter"] for f in feature_infos]
    test_prompts = []
    for letter in letters:
        words = [f"{letter}{w}" for w in [
            "pple", "nimal", "lpha", "rt", "nt"
        ]]
        for word in words[:3]:
            test_prompts.append(f"The word '{word}'")

    tokenized = tokenizer(
        test_prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=32
    )
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, names_filter=[hook_name]
        )
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        sample_acts = sae_acts.max(dim=1).values

    absorption_scores = []
    for i, feat_info in enumerate(feature_infos):
        feat_idx = feat_info["feature_idx"]
        letter = feat_info["letter"]
        start_idx = i * 3
        end_idx = start_idx + 3
        own_acts = sample_acts[start_idx:end_idx, feat_idx].mean().item()
        other_indices = (
            list(range(0, start_idx)) +
            list(range(end_idx, len(test_prompts)))
        )
        other_acts = sample_acts[other_indices, feat_idx].mean().item()
        specificity = own_acts / (other_acts + 1e-8)
        absorption_scores.append({
            "letter": letter,
            "feature_idx": feat_idx,
            "own_activation": own_acts,
            "other_activation": other_acts,
            "specificity": specificity,
        })
    return absorption_scores


# ── Helper: compute SAE metrics ────────────────────────────────────────────
def compute_sae_metrics(model, sae, tokenizer, device, n_samples=100):
    """Compute reconstruction quality and sparsity metrics."""
    # Generate test prompts
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth revolves around the Sun.",
    ] * 20

    tokenized = tokenizer(
        test_texts, return_tensors="pt", padding=True,
        truncation=True, max_length=64
    )
    input_ids = tokenized["input_ids"].to(device)

    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, names_filter=[hook_name]
        )
        acts = cache[hook_name]

        # Encode and decode
        sae_acts = sae.encode(acts)
        recons = sae.decode(sae_acts)

        # Reconstruction MSE
        mse = ((acts - recons) ** 2).mean().item()

        # L0 sparsity (mean active features per token)
        active = (sae_acts > 0).float()
        l0_sparsity = active.sum(dim=-1).mean().item()

        # Dead feature ratio
        dead_ratio = (active.sum(dim=(0, 1)) == 0).float().mean().item()

    return {
        "reconstruction_mse": mse,
        "l0_sparsity": l0_sparsity,
        "dead_feature_ratio": dead_ratio,
    }


# ── Main experiment ────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    report_progress(0, 4, step=1, total_steps=4, metric={"phase": "init"})

    results = {
        "task_id": TASK_ID,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "seed": SEED,
        "start_time": datetime.now().isoformat(),
        "architectures": {},
    }

    # ── Step 1: Load GPT-2 Small ───────────────────────────────────────────
    print("[1/4] Loading GPT-2 Small...")
    report_progress(1, 4, step=1, total_steps=4, metric={"phase": "load_model"})

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded on {DEVICE}")

    # ── Step 2: Load pre-trained SAE ───────────────────────────────────────
    print("[2/4] Loading pre-trained SAE...")
    report_progress(2, 4, step=2, total_steps=4, metric={"phase": "load_sae"})

    try:
        sae_pretrained = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device=DEVICE,
        )
        print("  Loaded pre-trained GPT-2 SAE")
        results["architectures"]["pretrained"] = {
            "source": "gpt2-small-res-jb",
            "d_sae": sae_pretrained.cfg.d_sae,
        }
    except Exception as e:
        print(f"  Could not load pre-trained SAE: {e}")
        sae_pretrained = None
        results["architectures"]["pretrained"] = {
            "source": "none", "error": str(e)
        }

    # ── Step 3: Train TopK SAE (larger, 16k features) ──────────────────────
    print("[3/4] Training TopK SAE (16k features)...")
    report_progress(3, 4, step=3, total_steps=4, metric={"phase": "train_topk"})

    topk_cfg = LanguageModelSAERunnerConfig(
        sae=TopKTrainingSAEConfig(
            d_in=model.cfg.d_model,
            d_sae=16384,
            k=50,
            device=DEVICE,
        ),
        model_name="gpt2-small",
        hook_name="blocks.8.hook_resid_pre",
        dataset_path="monology/pile-uncopyrighted",
        training_tokens=2_000_000,
        train_batch_size_tokens=4096,
        store_batch_size_prompts=64,
        n_batches_in_buffer=64,
        context_size=128,
        lr=3e-4,
        device=DEVICE,
        seed=SEED,
        logger=LoggingConfig(log_to_wandb=False),
        verbose=False,
        n_checkpoints=0,
        save_final_checkpoint=True,
        output_path=str(RESULTS_DIR / "topk_sae_16k_output"),
    )

    runner = LanguageModelSAETrainingRunner(topk_cfg)
    sae_topk = runner.run()
    print("  TopK SAE (16k) training complete")

    # ── Step 4: Evaluate all architectures ─────────────────────────────────
    print("[4/4] Evaluating architectures...")
    report_progress(4, 4, step=4, total_steps=4, metric={"phase": "evaluate"})

    # Evaluate TopK SAE
    print("  Evaluating TopK SAE...")
    topk_features = find_first_letter_features(
        model, sae_topk, tokenizer, DEVICE
    )
    topk_absorption = measure_absorption(
        model, sae_topk, tokenizer, DEVICE, topk_features
    )
    topk_metrics = compute_sae_metrics(
        model, sae_topk, tokenizer, DEVICE
    )

    topk_feat_ids = [f["feature_idx"] for f in topk_features]
    topk_unique = len(set(topk_feat_ids))
    topk_collision_rate = (
        (len(topk_features) - topk_unique) / len(topk_features)
        if topk_features else 0
    )
    topk_mean_specificity = np.mean([
        a["specificity"] for a in topk_absorption
    ])

    results["architectures"]["topk_16k"] = {
        "d_sae": sae_topk.cfg.d_sae,
        "k": sae_topk.cfg.k,
        "first_letter_features": topk_features,
        "num_found": len(topk_features),
        "absorption_scores": topk_absorption,
        "collision_rate": topk_collision_rate,
        "unique_features": topk_unique,
        "mean_specificity": topk_mean_specificity,
        **topk_metrics,
    }

    print(f"    TopK 16k: {topk_unique}/{len(topk_features)} unique, "
          f"collision={topk_collision_rate:.2%}, "
          f"MSE={topk_metrics['reconstruction_mse']:.4f}, "
          f"L0={topk_metrics['l0_sparsity']:.1f}")

    # Evaluate pre-trained SAE if loaded
    if sae_pretrained is not None:
        print("  Evaluating pre-trained SAE...")
        pretrained_features = find_first_letter_features(
            model, sae_pretrained, tokenizer, DEVICE
        )
        pretrained_absorption = measure_absorption(
            model, sae_pretrained, tokenizer, DEVICE, pretrained_features
        )
        pretrained_metrics = compute_sae_metrics(
            model, sae_pretrained, tokenizer, DEVICE
        )

        pretrained_feat_ids = [f["feature_idx"] for f in pretrained_features]
        pretrained_unique = len(set(pretrained_feat_ids))
        pretrained_collision_rate = (
            (len(pretrained_features) - pretrained_unique) /
            len(pretrained_features)
            if pretrained_features else 0
        )
        pretrained_mean_specificity = np.mean([
            a["specificity"] for a in pretrained_absorption
        ])

        results["architectures"]["pretrained"] = {
            "d_sae": sae_pretrained.cfg.d_sae,
            "first_letter_features": pretrained_features,
            "num_found": len(pretrained_features),
            "absorption_scores": pretrained_absorption,
            "collision_rate": pretrained_collision_rate,
            "unique_features": pretrained_unique,
            "mean_specificity": pretrained_mean_specificity,
            **pretrained_metrics,
        }

        print(f"    Pretrained: {pretrained_unique}/{len(pretrained_features)} unique, "
              f"collision={pretrained_collision_rate:.2%}, "
              f"MSE={pretrained_metrics['reconstruction_mse']:.4f}, "
              f"L0={pretrained_metrics['l0_sparsity']:.1f}")

        # Compute differences
        results["comparison"] = {
            "collision_rate_diff": abs(
                topk_collision_rate - pretrained_collision_rate
            ),
            "specificity_diff": abs(
                topk_mean_specificity - pretrained_mean_specificity
            ),
            "mse_diff": abs(
                topk_metrics['reconstruction_mse'] -
                pretrained_metrics['reconstruction_mse']
            ),
        }

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = RESULTS_DIR / "f1_caab_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    # Summary
    summary = (
        f"TopK 16k: collision={topk_collision_rate:.2%}, "
        f"specificity={topk_mean_specificity:.3f}, "
        f"MSE={topk_metrics['reconstruction_mse']:.4f}, "
        f"L0={topk_metrics['l0_sparsity']:.1f}, "
        f"elapsed={elapsed:.1f}s"
    )
    if sae_pretrained is not None:
        summary += (
            f" | Pretrained: collision={pretrained_collision_rate:.2%}, "
            f"MSE={pretrained_metrics['reconstruction_mse']:.4f}"
        )

    mark_done(status="success", summary=summary)
    print("\nFull E1 CAAB complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
