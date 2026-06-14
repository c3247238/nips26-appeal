#!/usr/bin/env python3
"""
Full E6: Dynamic Feature De-Absorption (DFDA)
Scale DFDA to more absorbed pairs on GPT-2 Small.
Train compensation MLPs. Measure reconstruction impact.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
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
TASK_ID = "f6_dfda"

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


# ── Helper: find absorbed pairs ────────────────────────────────────────────
def find_absorbed_pairs(model, sae, tokenizer, device, n_pairs=20):
    """Find pairs of features where one 'absorbs' the other."""
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
    letter_features = {}
    for letter_idx in range(26):
        mask = labels_t == letter_idx
        if mask.sum() == 0:
            continue
        letter_acts = sample_acts[mask].mean(dim=0)
        other_acts = sample_acts[~mask].mean(dim=0)
        scores = letter_acts - other_acts
        best_feat = scores.argmax().item()
        best_score = scores[best_feat].item()
        letter_features[letter_idx] = {"feature": best_feat, "score": best_score}

    feat_to_letters = {}
    for letter, info in letter_features.items():
        feat = info["feature"]
        if feat not in feat_to_letters:
            feat_to_letters[feat] = []
        feat_to_letters[feat].append(letter)

    absorbed_pairs = []
    for feat, letters_list in feat_to_letters.items():
        if len(letters_list) > 1:
            for i in range(1, len(letters_list)):
                absorbed_pairs.append({
                    "parent_letter": letters_list[0],
                    "child_letter": letters_list[i],
                    "shared_feature": feat,
                })

    return absorbed_pairs[:n_pairs]


# ── Helper: extract feature activations ────────────────────────────────────
def extract_pair_activations(model, sae, tokenizer, device, pair, n_samples=300):
    """Extract child and parent feature activations for training."""
    parent_letter = chr(ord('a') + pair["parent_letter"])
    child_letter = chr(ord('a') + pair["child_letter"])
    shared_feat = pair["shared_feature"]

    prompts = []
    for letter in [parent_letter, child_letter]:
        words = [f"{letter}{w}" for w in ["pple", "nimal", "lpha", "rt", "nt"]]
        for word in words[:3]:
            prompts.append(f"The word '{word}'")

    bg_prompts = [
        "The weather is nice today.",
        "Machine learning is fascinating.",
        "The cat sat on the mat.",
    ] * 50
    prompts.extend(bg_prompts)

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

    child_acts = sample_acts[:, shared_feat:shared_feat+1]
    parent_acts = sample_acts.mean(dim=1, keepdim=True)

    return child_acts, parent_acts


# ── Small MLP for de-absorption ────────────────────────────────────────────
class DeAbsorptionMLP(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=32, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


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
    }

    # ── Step 1: Load GPT-2 Small and SAE ───────────────────────────────────
    print("[1/4] Loading GPT-2 Small and SAE...")
    report_progress(1, 4, step=1, total_steps=4, metric={"phase": "load_model"})

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    try:
        sae = SAE.from_pretrained(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device=DEVICE,
        )
        print("  Loaded pre-trained GPT-2 SAE")
        results["sae_source"] = "gpt2-small-res-jb"
    except Exception as e:
        print(f"  Could not load pre-trained SAE: {e}")
        mark_done(status="failed", summary=f"SAE loading failed: {e}")
        return

    # ── Step 2: Find absorbed pairs ────────────────────────────────────────
    print("[2/4] Finding absorbed pairs...")
    report_progress(2, 4, step=2, total_steps=4, metric={"phase": "find_pairs"})

    absorbed_pairs = find_absorbed_pairs(model, sae, tokenizer, DEVICE, n_pairs=20)
    print(f"  Found {len(absorbed_pairs)} absorbed pairs")
    results["absorbed_pairs"] = absorbed_pairs

    if len(absorbed_pairs) == 0:
        results["note"] = "No absorbed pairs found"
        elapsed = time.time() - start_time
        results["elapsed_seconds"] = elapsed
        results["end_time"] = datetime.now().isoformat()
        output_file = RESULTS_DIR / "f6_dfda_results.json"
        output_file.write_text(json.dumps(results, indent=2, default=str))
        mark_done(status="success", summary="No absorbed pairs found")
        return

    # ── Step 3: Train de-absorption MLPs ───────────────────────────────────
    print("[3/4] Training de-absorption MLPs...")
    report_progress(3, 4, step=3, total_steps=4, metric={"phase": "train_mlp"})

    pair_results = []
    for i, pair in enumerate(absorbed_pairs):
        print(f"  Training MLP for pair {i+1}/{len(absorbed_pairs)}: "
              f"parent={chr(ord('a')+pair['parent_letter'])}, "
              f"child={chr(ord('a')+pair['child_letter'])}")

        child_acts, parent_acts = extract_pair_activations(
            model, sae, tokenizer, DEVICE, pair
        )

        n = len(child_acts)
        n_train = int(0.8 * n)
        child_train = child_acts[:n_train]
        parent_train = parent_acts[:n_train]
        child_test = child_acts[n_train:]
        parent_test = parent_acts[n_train:]

        mlp = DeAbsorptionMLP(input_dim=1, hidden_dim=32, output_dim=1).to(DEVICE)
        optimizer = torch.optim.Adam(mlp.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        for epoch in range(200):
            optimizer.zero_grad()
            pred = mlp(child_train)
            loss = criterion(pred, parent_train)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            pred_test = mlp(child_test)
            test_mse = criterion(pred_test, parent_test).item()
            baseline_pred = parent_train.mean()
            baseline_mse = ((parent_test - baseline_pred) ** 2).mean().item()

        improvement = (baseline_mse - test_mse) / (baseline_mse + 1e-8) if baseline_mse > 1e-8 else 0.0

        pair_results.append({
            "pair": pair,
            "test_mse": test_mse,
            "baseline_mse": baseline_mse,
            "improvement": improvement,
            "n_params": sum(p.numel() for p in mlp.parameters()),
        })

        print(f"    Test MSE: {test_mse:.4f}, Baseline: {baseline_mse:.4f}, Improvement: {improvement:.2%}")

    results["pair_results"] = pair_results
    results["mean_improvement"] = np.mean([r["improvement"] for r in pair_results])
    results["median_improvement"] = np.median([r["improvement"] for r in pair_results])
    results["total_params"] = sum(r["n_params"] for r in pair_results)

    print(f"\n  Mean improvement: {results['mean_improvement']:.2%}")
    print(f"  Median improvement: {results['median_improvement']:.2%}")
    print(f"  Total parameters: {results['total_params']}")

    # ── Step 4: Measure reconstruction impact ──────────────────────────────
    print("[4/4] Measuring reconstruction impact...")
    report_progress(4, 4, step=4, total_steps=4, metric={"phase": "reconstruction"})

    test_prompts = [
        "The word 'apple' is a fruit.",
        "The word 'animal' refers to living creatures.",
        "The word 'alpha' is the first Greek letter.",
    ]

    tokenized = tokenizer(
        test_prompts, return_tensors="pt", padding=True,
        truncation=True, max_length=32
    )
    input_ids = tokenized["input_ids"].to(DEVICE)

    hook_name = getattr(sae.cfg, 'hook_name', 'blocks.8.hook_resid_pre')
    with torch.no_grad():
        _, cache = model.run_with_cache(
            input_ids, names_filter=[hook_name]
        )
        acts = cache[hook_name]
        sae_acts = sae.encode(acts)
        recons = sae.decode(sae_acts)
        base_mse = ((acts - recons) ** 2).mean().item()

    results["base_reconstruction_mse"] = base_mse
    print(f"  Base reconstruction MSE: {base_mse:.4f}")

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = RESULTS_DIR / "f6_dfda_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    summary = (
        f"Pairs={len(absorbed_pairs)}, mean_improvement={results['mean_improvement']:.2%}, "
        f"median={results['median_improvement']:.2%}, total_params={results['total_params']}, "
        f"elapsed={elapsed:.1f}s"
    )

    mark_done(status="success", summary=summary)
    print("\nFull E6 DFDA complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
