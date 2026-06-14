#!/usr/bin/env python3
"""
Full E3: Sparsity-Absorption Relationship
Train TopK SAEs with varying k values on GPT-2 Small layer 8.
Compute absorption rate and reconstruction quality at each k.
Fit monotonic trend and measure Spearman correlation.
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
    LanguageModelSAERunnerConfig, TopKTrainingSAEConfig,
    LanguageModelSAETrainingRunner
)
from sae_lens.config import LoggingConfig
from scipy.stats import spearmanr

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "f3_sparsity"

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


# ── Helper: compute absorption rate ────────────────────────────────────────
def compute_absorption_rate(model, sae, tokenizer, device):
    """Compute simplified absorption rate via feature collision."""
    letters = [chr(ord('a') + i) for i in range(26)]
    prompts = []
    labels = []
    for letter in letters:
        words = [f"{letter}{w}" for w in ["pple", "nimal", "lpha", "rt"]]
        for word in words:
            prompts.append(f"The word '{word}'")
            labels.append(ord(letter) - ord('a'))

    tokenized = tokenizer(
        prompts, return_tensors="pt", padding=True,
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

    labels_t = torch.tensor(labels, device=device)
    best_features = []
    for letter_idx in range(26):
        mask = labels_t == letter_idx
        if mask.sum() == 0:
            continue
        letter_acts = sample_acts[mask].mean(dim=0)
        other_acts = sample_acts[~mask].mean(dim=0)
        scores = letter_acts - other_acts
        best_feat = scores.argmax().item()
        best_features.append(best_feat)

    unique = len(set(best_features))
    collision_rate = (len(best_features) - unique) / len(best_features) if best_features else 0
    return collision_rate


# ── Helper: compute reconstruction metrics ─────────────────────────────────
def compute_reconstruction_metrics(model, sae, tokenizer, device):
    """Compute reconstruction MSE and loss recovered."""
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
        sae_acts = sae.encode(acts)
        recons = sae.decode(sae_acts)
        mse = ((acts - recons) ** 2).mean().item()
        active = (sae_acts > 0).float()
        l0 = active.sum(dim=-1).mean().item()

    return {"mse": mse, "l0": l0}


# ── Main experiment ────────────────────────────────────────────────────────
def main():
    start_time = time.time()
    report_progress(0, 3, step=1, total_steps=3, metric={"phase": "init"})

    results = {
        "task_id": TASK_ID,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "seed": SEED,
        "start_time": datetime.now().isoformat(),
        "k_values": [],
        "absorption_rates": [],
        "reconstruction_mses": [],
        "l0_sparsities": [],
    }

    # ── Step 1: Load GPT-2 Small ───────────────────────────────────────────
    print("[1/3] Loading GPT-2 Small...")
    report_progress(1, 3, step=1, total_steps=3, metric={"phase": "load_model"})

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded on {DEVICE}")

    # ── Step 2: Train SAEs with varying k ──────────────────────────────────
    print("[2/3] Training SAEs with varying k values...")
    report_progress(2, 3, step=2, total_steps=3, metric={"phase": "train_saes"})

    k_values = [10, 25, 50, 100, 200]
    saes = {}

    for k in k_values:
        print(f"  Training k={k}...")
        cfg = LanguageModelSAERunnerConfig(
            sae=TopKTrainingSAEConfig(
                d_in=model.cfg.d_model,
                d_sae=16384,
                k=k,
                device=DEVICE,
            ),
            model_name="gpt2-small",
            hook_name="blocks.8.hook_resid_pre",
            dataset_path="monology/pile-uncopyrighted",
            training_tokens=1_000_000,
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
            output_path=str(RESULTS_DIR / f"f3_k{k}_sae_output"),
        )
        runner = LanguageModelSAETrainingRunner(cfg)
        sae = runner.run()
        saes[k] = sae
        print(f"    k={k} complete")

    # ── Step 3: Evaluate absorption and reconstruction ─────────────────────
    print("[3/3] Evaluating absorption and reconstruction...")
    report_progress(3, 3, step=3, total_steps=3, metric={"phase": "evaluate"})

    for k in k_values:
        sae = saes[k]
        absorption = compute_absorption_rate(model, sae, tokenizer, DEVICE)
        metrics = compute_reconstruction_metrics(model, sae, tokenizer, DEVICE)

        results["k_values"].append(k)
        results["absorption_rates"].append(absorption)
        results["reconstruction_mses"].append(metrics["mse"])
        results["l0_sparsities"].append(metrics["l0"])

        print(f"  k={k}: absorption={absorption:.2%}, MSE={metrics['mse']:.4f}, L0={metrics['l0']:.1f}")

    # Compute Spearman correlation
    corr_abs, p_abs = spearmanr(results["k_values"], results["absorption_rates"])
    corr_mse, p_mse = spearmanr(results["k_values"], results["reconstruction_mses"])

    results["spearman"] = {
        "k_vs_absorption": {"r": corr_abs, "p_value": p_abs},
        "k_vs_mse": {"r": corr_mse, "p_value": p_mse},
    }

    print(f"\n  Spearman k vs absorption: r={corr_abs:.3f}, p={p_abs:.4f}")
    print(f"  Spearman k vs MSE: r={corr_mse:.3f}, p={p_mse:.4f}")

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = RESULTS_DIR / "f3_sparsity_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    # Summary
    summary = (
        f"k_vs_absorption: r={corr_abs:.3f} (p={p_abs:.4f}), "
        f"k_vs_mse: r={corr_mse:.3f} (p={p_mse:.4f}), "
        f"elapsed={elapsed:.1f}s"
    )

    mark_done(status="success", summary=summary)
    print("\nFull E3 Sparsity complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
