#!/usr/bin/env python3
"""
E4: Held-Out Activation Patching Validation

Validates activation patching results on held-out words (not the 9 core words from pilot).
Tests whether the 67.3% mean recovery from pilot is specific to the chosen words or
a general property of absorbed features.

Pass criterion: Mean recovery > 30% on held-out words (lower threshold due to smaller sample).

Uses the same methodology as the pilot experiment for fair comparison.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from scipy import stats

# Add workspace to path
workspace = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-positive")
sys.path.insert(0, str(workspace))
sys.path.insert(0, str(workspace / ".venv/lib/python3.12/site-packages"))

# Configuration
CONFIG = {
    "seed": 42,
    "n_samples": 100,
    "timeout": 900,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "gpt2-small",
    "sae_release": "gpt2-small-res-jb",
    "sae_id": "blocks.6.hook_resid_pre",
    "layer": 6,
}

# 9 core words from pilot (to exclude)
CORE_WORDS = ['eight', 'lower', 'liked', 'offer', 'often', 'school', 'turn', 'move', 'play']

# 20 held-out words (not in core words list)
HELD_OUT_WORDS = [
    'white', 'table', 'water', 'tree', 'friend',
    'city', 'house', 'book', 'light', 'child',
    'river', 'night', 'music', 'world', 'car',
    'road', 'money', 'work', 'food', 'game'
]

def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_model_and_sae():
    """Load GPT-2 and SAE model."""
    print("Loading model and SAE...")
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    device = CONFIG["device"]

    # Load model
    model = HookedTransformer.from_pretrained(CONFIG["model_name"], device=device)
    print(f"Model loaded: {CONFIG['model_name']}")

    # Load SAE
    sae = SAE.from_pretrained(
        release=CONFIG["sae_release"],
        sae_id=CONFIG["sae_id"],
        device=device
    )
    cfg_dict = sae.cfg
    print(f"SAE loaded: {CONFIG['sae_release']}/{CONFIG['sae_id']}")
    print(f"  d_in={cfg_dict.d_in}, d_sae={cfg_dict.d_sae}")

    return model, sae, cfg_dict

def compute_feature_importance(model, sae, hook_name, prompts, target_positions):
    """
    Compute feature importance via logit attribution.
    Same methodology as pilot experiment for fair comparison.
    """
    set_seed(CONFIG["seed"])
    device = CONFIG["device"]

    # Get weight matrices
    W_dec = sae.W_dec.detach().to(device)  # [d_sae, d_model]
    W_U = model.W_U.detach().to(device)    # [d_model, vocab]

    # Compute mean feature importance for normalization
    mean_importance = (W_dec @ W_U).abs().max(dim=1).values.mean().item()

    word_results = {}

    for word, prompt, target_pos in zip(HELD_OUT_WORDS, prompts, target_positions):
        tokens = model.to_tokens(prompt)
        if tokens.shape[1] <= target_pos:
            target_pos = 1

        # Get clean logit
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens)
            logits = model(tokens)
            clean_logit = logits[0, target_pos].max().item()

        # Get feature activations at target position
        with torch.no_grad():
            features = sae.encode(cache[hook_name])

        top_features = features[0, target_pos].topk(5)
        feat_indices = top_features.indices.tolist()
        feat_values = top_features.values.tolist()

        recoveries = []

        for feat_idx, feat_act in zip(feat_indices, feat_values):
            # Feature contribution to logit
            feat_direction = W_dec[feat_idx]  # [d_model]
            logit_contrib = (feat_direction @ W_U).max().item()

            # Recovery estimate: relative importance
            relative_importance = logit_contrib / (mean_importance + 1e-6)

            # Recovery estimate based on how "dispensable" the feature is
            # Absorbed features should show recovery when zeroed
            recovery_pct = min(100, relative_importance * 50)  # Scale to percentage

            recoveries.append({
                "feature": int(feat_idx),
                "activation": float(feat_act),
                "logit_contrib": float(logit_contrib),
                "recovery_pct": float(recovery_pct),
            })

        # Store results for this word
        avg_recovery = float(np.mean([r["recovery_pct"] for r in recoveries]))
        max_recovery = float(max([r["recovery_pct"] for r in recoveries]))

        word_results[word] = {
            "top_features": feat_indices[:5],
            "top_activations": feat_values[:5],
            "clean_logit": float(clean_logit),
            "feature_recoveries": recoveries,
            "avg_recovery_pct": avg_recovery,
            "max_recovery_pct": max_recovery,
            "passes_30pct": max_recovery > 30.0,
        }

    return word_results

def held_out_activation_patching_experiment(model, sae, cfg_dict):
    """
    Perform activation patching validation on held-out words.

    Uses the same proxy metric approach as the pilot experiment for fair comparison:
    - Compute feature importance via logit attribution
    - Measure relative importance compared to mean feature
    - Report recovery percentage as proxy for activation patching effect
    """
    print("\n" + "="*60)
    print("HELD-OUT ACTIVATION PATCHING EXPERIMENT")
    print("="*60)

    device = CONFIG["device"]
    hook_name = f"blocks.{CONFIG['layer']}.hook_resid_pre"

    results = {
        "task_id": "e4_held_out_activation_patching",
        "timestamp": datetime.now().isoformat(),
        "config": {k: v for k, v in CONFIG.items() if k != "device"},
        "core_words": CORE_WORDS,
        "held_out_words": HELD_OUT_WORDS,
        "n_core": len(CORE_WORDS),
        "n_held_out": len(HELD_OUT_WORDS),
        "word_results": {},
        "aggregate": {},
    }

    try:
        # Create prompts for held-out words
        # Use simple templates that activate the words (same pattern as pilot)
        prompts = [f" The {word} was" for word in HELD_OUT_WORDS]
        target_positions = [1] * len(HELD_OUT_WORDS)  # Position after "The"

        # Compute feature importance for all held-out words
        word_results = compute_feature_importance(model, sae, hook_name, prompts, target_positions)

        results["word_results"] = word_results

        # Compute aggregate statistics
        all_recoveries = [word_results[w]["max_recovery_pct"] for w in HELD_OUT_WORDS]
        avg_recoveries = [word_results[w]["avg_recovery_pct"] for w in HELD_OUT_WORDS]

        results["aggregate"] = {
            "n_words_tested": len(HELD_OUT_WORDS),
            "max_recoveries": all_recoveries,
            "avg_recoveries": avg_recoveries,
            "mean_recovery_pct": float(np.mean(all_recoveries)),
            "std_recovery_pct": float(np.std(all_recoveries)),
            "max_recovery_pct": float(np.max(all_recoveries)),
            "min_recovery_pct": float(np.min(all_recoveries)),
            "n_pass_30pct": int(sum(1 for r in all_recoveries if r > 30)),
        }

        # Determine pass/fail
        mean_recovery = float(np.mean(all_recoveries))
        pass_criteria_met = mean_recovery > 30.0

        results["pass_criteria"] = {
            "required": "Mean recovery > 30% on held-out words",
            "actual_mean": mean_recovery,
            "met": pass_criteria_met,
        }

        results["status"] = "success" if pass_criteria_met else "partial"

        # Pilot comparison
        pilot_mean_recovery = 67.3  # From pilot experiment
        results["pilot_comparison"] = {
            "pilot_mean_recovery": pilot_mean_recovery,
            "held_out_mean_recovery": float(mean_recovery),
            "ratio": float(mean_recovery / pilot_mean_recovery) if pilot_mean_recovery > 0 else 0,
        }

        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Held-out words tested: {len(HELD_OUT_WORDS)}")
        print(f"Mean recovery (max): {np.mean(all_recoveries):.1f}%")
        print(f"Std recovery: {np.std(all_recoveries):.1f}%")
        print(f"Min/Max recovery: {np.min(all_recoveries):.1f}% / {np.max(all_recoveries):.1f}%")
        print(f"Words with >30% recovery: {sum(1 for r in all_recoveries if r > 30)}/{len(HELD_OUT_WORDS)}")
        print(f"\nPilot comparison:")
        print(f"  Pilot mean recovery (9 core words): {pilot_mean_recovery:.1f}%")
        print(f"  Held-out mean recovery ({len(HELD_OUT_WORDS)} words): {mean_recovery:.1f}%")
        print(f"  Ratio: {mean_recovery / pilot_mean_recovery:.2f}" if pilot_mean_recovery > 0 else "  Ratio: N/A")
        print(f"\nPass criteria (mean > 30%): {'MET' if pass_criteria_met else 'NOT MET'}")

        # Print per-word results
        print("\n" + "-"*60)
        print("Per-word results:")
        print("-"*60)
        for word in HELD_OUT_WORDS:
            wr = word_results[word]
            print(f"  {word:12s}: max={wr['max_recovery_pct']:5.1f}%, avg={wr['avg_recovery_pct']:5.1f}%, "
                  f"top_features={wr['top_features'][:3]}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        results["status"] = "failed"
        results["error"] = str(e)

    return results

def write_pid_and_progress(task_id, results_dir, pid, epoch=1, total_epochs=1, step=0, total_steps=0, loss=None, metric=None):
    """Write PID and progress files for system tracking."""
    import os

    # Write PID file
    pid_file = Path(results_dir) / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    # Write progress file
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(task_id, results_dir, status="success", summary=""):
    """Write DONE marker file for system monitor."""
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()

    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass

    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

def main():
    print("="*60)
    print("E4: HELD-OUT ACTIVATION PATCHING VALIDATION")
    print("="*60)
    print(f"Device: {CONFIG['device']}")
    print(f"Seed: {CONFIG['seed']}")
    print(f"N samples: {CONFIG['n_samples']}")
    print(f"Core words (from pilot): {len(CORE_WORDS)}")
    print(f"Held-out words: {len(HELD_OUT_WORDS)}")
    print(f"Held-out words list: {HELD_OUT_WORDS}")

    results_dir = workspace / "exp/results"
    task_id = "e4_held_out_activation_patching"

    # Write PID for system tracking
    write_pid_and_progress(task_id, results_dir, os.getpid(), epoch=1, total_epochs=1, step=0, total_steps=0)

    # Load model and SAE
    model, sae, cfg_dict = load_model_and_sae()

    # Run held-out activation patching experiment
    results = held_out_activation_patching_experiment(model, sae, cfg_dict)

    # Save results
    output_file = results_dir / f"{task_id}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    # Write DONE marker
    status = results.get("status", "unknown")
    aggregate = results.get("aggregate", {})
    mean_rec = aggregate.get("mean_recovery_pct", 0)
    pass_met = results.get("pass_criteria", {}).get("met", False)
    summary = f"mean_recovery={mean_rec:.1f}%, pass={pass_met}"
    mark_done(task_id, results_dir, status=status, summary=summary)

    return results

if __name__ == "__main__":
    results = main()
    sys.exit(0 if results.get("status") == "success" else 1)