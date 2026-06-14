#!/usr/bin/env python3
"""
Pilot P2: Causal Evaluation Feasibility
Train two SAEs with different sparsity levels (high vs low absorption).
Run sparse probing on 20 concepts (4 per domain).
Verify correlation between absorption rate and probe accuracy.
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

# ── Configuration ──────────────────────────────────────────────────────────
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "p2_causal"

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


# ── Concept sets for probing ───────────────────────────────────────────────
CONCEPTS = {
    "animals": ["dog", "cat", "bird", "fish"],
    "colors": ["red", "blue", "green", "yellow"],
    "numbers": ["one", "two", "three", "four"],
    "countries": ["USA", "China", "France", "Japan"],
    "emotions": ["happy", "sad", "angry", "calm"],
}


def build_concept_prompts(concepts, samples_per_concept=10):
    """Build prompts that elicit concept features."""
    prompts = []
    labels = []
    concept_list = []
    for domain, words in concepts.items():
        for word in words:
            templates = [
                f"The {word} is",
                f"A {word} can",
                f"I like {word}",
                f"The word '{word}' means",
                f"{word} is a",
            ]
            for tmpl in templates[:samples_per_concept]:
                prompts.append(tmpl)
                labels.append(len(concept_list))
            concept_list.append(word)
    return prompts, labels, concept_list


def train_linear_probe(features, labels, n_classes, device):
    """Train a simple linear probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    X = features.cpu().numpy()
    y = np.array(labels)

    # Split train/test
    n_train = int(0.8 * len(X))
    indices = np.random.permutation(len(X))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X[train_idx], y[train_idx])

    train_acc = accuracy_score(y[train_idx], clf.predict(X[train_idx]))
    test_acc = accuracy_score(y[test_idx], clf.predict(X[test_idx]))

    return {"train_acc": train_acc, "test_acc": test_acc}


def extract_sae_features(model, sae, tokenizer, device, prompts):
    """Extract SAE features for a list of prompts."""
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
        # Mean pool across positions
        features = sae_acts.mean(dim=1)
    return features


def compute_absorption_rate(model, sae, tokenizer, device):
    """Compute simplified absorption rate via feature collision."""
    letters = [chr(ord('a') + i) for i in range(26)]
    prompts = []
    labels = []
    for letter in letters:
        words = [f"{letter}{w}" for w in [
            "pple", "nimal", "lpha", "rt"
        ]]
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
    report_progress(0, 4, step=1, total_steps=4, metric={"phase": "init"})

    results = {
        "task_id": TASK_ID,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "seed": SEED,
        "start_time": datetime.now().isoformat(),
        "sae_configs": {},
        "probe_results": {},
    }

    # ── Step 1: Load GPT-2 Small ───────────────────────────────────────────
    print("[1/4] Loading GPT-2 Small...")
    report_progress(1, 4, step=1, total_steps=4, metric={"phase": "load_model"})

    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    print(f"  Model loaded on {DEVICE}")

    # ── Step 2: Train two SAEs with different sparsity ─────────────────────
    print("[2/4] Training SAEs with different sparsity levels...")
    report_progress(2, 4, step=2, total_steps=4, metric={"phase": "train_saes"})

    saes = {}
    configs = {
        "high_k": {"k": 100, "tokens": 500_000, "label": "low_absorption"},
        "low_k": {"k": 10, "tokens": 500_000, "label": "high_absorption"},
    }

    for cfg_name, cfg_info in configs.items():
        print(f"  Training {cfg_name} (k={cfg_info['k']})...")
        cfg = LanguageModelSAERunnerConfig(
            sae=TopKTrainingSAEConfig(
                d_in=model.cfg.d_model,
                d_sae=8192,
                k=cfg_info["k"],
                device=DEVICE,
            ),
            model_name="gpt2-small",
            hook_name="blocks.8.hook_resid_pre",
            dataset_path="monology/pile-uncopyrighted",
            training_tokens=cfg_info["tokens"],
            train_batch_size_tokens=2048,
            store_batch_size_prompts=32,
            n_batches_in_buffer=32,
            context_size=128,
            lr=3e-4,
            device=DEVICE,
            seed=SEED,
            logger=LoggingConfig(log_to_wandb=False),
            verbose=False,
            n_checkpoints=0,
            save_final_checkpoint=True,
            output_path=str(RESULTS_DIR / f"p2_{cfg_name}_sae_output"),
        )
        runner = LanguageModelSAETrainingRunner(cfg)
        sae = runner.run()
        saes[cfg_name] = sae
        results["sae_configs"][cfg_name] = {
            "k": cfg_info["k"],
            "d_sae": sae.cfg.d_sae,
            "training_tokens": cfg_info["tokens"],
        }
        print(f"    {cfg_name} complete: d_sae={sae.cfg.d_sae}, k={sae.cfg.k}")

    # ── Step 3: Compute absorption rates ───────────────────────────────────
    print("[3/4] Computing absorption rates...")
    report_progress(3, 4, step=3, total_steps=4, metric={"phase": "absorption"})

    for cfg_name, sae in saes.items():
        absorption_rate = compute_absorption_rate(
            model, sae, tokenizer, DEVICE
        )
        results["sae_configs"][cfg_name]["absorption_rate"] = absorption_rate
        print(f"  {cfg_name}: absorption_rate={absorption_rate:.2%}")

    # ── Step 4: Train probes and compare accuracy ──────────────────────────
    print("[4/4] Training sparse probes...")
    report_progress(4, 4, step=4, total_steps=4, metric={"phase": "probing"})

    prompts, labels, concept_list = build_concept_prompts(CONCEPTS)
    print(f"  Built {len(prompts)} prompts for {len(concept_list)} concepts")

    for cfg_name, sae in saes.items():
        print(f"  Probing {cfg_name}...")
        features = extract_sae_features(
            model, sae, tokenizer, DEVICE, prompts
        )
        probe_result = train_linear_probe(
            features, labels, len(concept_list), DEVICE
        )
        results["probe_results"][cfg_name] = probe_result
        print(f"    {cfg_name}: train_acc={probe_result['train_acc']:.3f}, "
              f"test_acc={probe_result['test_acc']:.3f}")

    # Compute correlation
    absorption_rates = [
        results["sae_configs"][k]["absorption_rate"] for k in configs.keys()
    ]
    test_accs = [
        results["probe_results"][k]["test_acc"] for k in configs.keys()
    ]

    # Simple correlation (only 2 points, so just report direction)
    if len(absorption_rates) == 2:
        diff_abs = absorption_rates[1] - absorption_rates[0]
        diff_acc = test_accs[1] - test_accs[0]
        results["correlation_direction"] = "negative" if diff_abs * diff_acc < 0 else "positive"
        results["correlation_magnitude"] = abs(diff_acc)
        print(f"\n  Correlation: {results['correlation_direction']}")
        print(f"  Accuracy difference: {abs(diff_acc):.3f}")

    # ── Save results ───────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_seconds"] = elapsed
    results["end_time"] = datetime.now().isoformat()

    output_file = RESULTS_DIR / "p2_causal_results.json"
    output_file.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nResults saved to {output_file}")
    print(f"Total time: {elapsed:.1f}s")

    # Summary
    high_k_acc = results["probe_results"]["high_k"]["test_acc"]
    low_k_acc = results["probe_results"]["low_k"]["test_acc"]
    high_k_abs = results["sae_configs"]["high_k"]["absorption_rate"]
    low_k_abs = results["sae_configs"]["low_k"]["absorption_rate"]

    summary = (
        f"high_k (k=100): abs={high_k_abs:.2%}, acc={high_k_acc:.3f} | "
        f"low_k (k=10): abs={low_k_abs:.2%}, acc={low_k_acc:.3f} | "
        f"corr={results.get('correlation_direction', 'N/A')}, elapsed={elapsed:.1f}s"
    )

    mark_done(status="success", summary=summary)
    print("\nPilot P2 complete!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
        print(f"ERROR: {error_msg}")
        mark_done(status="failed", summary=error_msg[:500])
        sys.exit(1)
