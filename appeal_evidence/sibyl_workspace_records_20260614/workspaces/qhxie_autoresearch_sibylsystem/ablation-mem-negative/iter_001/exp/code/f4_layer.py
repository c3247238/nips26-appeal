#!/usr/bin/env python3
"""
Full E4: Layer-Depth Absorption Pattern
Load pre-trained SAEs at multiple layers of GPT-2 Small.
Compute absorption rate at each layer using standardized concept set.
Analyze trend across depth.
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
from sae_lens import SAE

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "f4_layer"

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
        "layer_results": [],
    }

    # ── Step 1: Load GPT-2 Small ───────────────────────────────────────────
    print("[1/3] Loading GPT-2 Small...")
    report_progress(1, 3, step=1, total_steps=3, metric={"phase": "load_model"})

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded on {DEVICE}")

    # ── Step 2: Load SAEs at multiple layers ───────────────────────────────
    print("[2/3] Loading SAEs at multiple layers...")
    report_progress(2, 3, step=2, total_steps=3, metric={"phase": "load_saes"})

    # GPT-2 Small has 12 layers; test every 2nd layer
    layers = [0, 2, 4, 6, 8, 10]
    saes = {}

    for layer in layers:
        try:
            sae = SAE.from_pretrained(
                release="gpt2-small-res-jb",
                sae_id=f"blocks.{layer}.hook_resid_pre",
                device=DEVICE,
            )
            saes[layer] = sae
            print(f"  Loaded SAE for layer {layer}")
        except Exception as e:
            print(f"  Could not load SAE for layer {layer}: {e}")

    if not saes:
        print("  No SAEs loaded - marking as failed")
        mark_done(status="failed", summary="No SAEs could be loaded")
        return

    # ── Step 3: Evaluate absorption at each layer ──────────────────────────
    print("[3/3] Evaluating absorption at each layer...")
    report_progress(3, 3, step=3, total_steps=3, metric={"phase": "evaluate"})

    for layer in sorted(saes.keys()):
        sae = saes[layer]
        absorption = compute_absorption_rate(model, sae, tokenizer, DEVICE)

        results["layer_results"].append({
            "layer": layer,
            "absorption_rate": absorption,
            "d_sae": sae.cfg.d_sae,
        })
        print(f"  Layer {layer}: absorption={absorption:.2%}")

    # Compute trend
    layers_eval = [r["layer"] for r in results["layer_results"]]
    absorptions = [r["absorption_rate"] for r in results["layer_results"]]

    if len(layers_eval) >= 2:
        from scipy.stats import spearmanr
        corr, pvalue = spearmanr(layers_eval, absorptions)
        results["layer_trend"] = {
            "spearman_r": corr,
            "p_value": pvalue,
        }
        print(f"\n  Layer vs absorption: r={corr:.3f}, p={pvalue:.4f}")

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = RESULTS_DIR / "f4_layer_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    trend_str = ""
    if "layer_trend" in results:
        trend_str = f", trend_r={results['layer_trend']['spearman_r']:.3f}"

    summary = (
        f"Layers={len(results['layer_results'])}, "
        f"abs_range=[{min(absorptions):.2%}, {max(absorptions):.2%}]"
        f"{trend_str}, elapsed={elapsed:.1f}s"
    )

    mark_done(status="success", summary=summary)
    print("\nFull E4 Layer-Depth complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
