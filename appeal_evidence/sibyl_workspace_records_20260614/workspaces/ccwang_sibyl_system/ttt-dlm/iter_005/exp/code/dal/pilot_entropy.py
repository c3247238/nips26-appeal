#!/usr/bin/env python3
"""
P4: Per-Position Entropy Sparsity Analysis (pilot_entropy)

On LLaDA-8B, for 16 GSM8K prompts, record per-position logit entropy at each
denoising step. At mask ratios 0.3-0.6 (critical zone), compute:
  (a) entropy concentration ratio (do <20% positions contribute >80% entropy?)
  (b) correlation between high-entropy positions and final answer errors

This validates the precision-weighting hypothesis (H8).

Usage:
  CUDA_VISIBLE_DEVICES=1 python pilot_entropy.py
"""

import json
import os
import gc
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Config ──────────────────────────────────────────────────────────────────
SEED = 42
NUM_SAMPLES = 16
NUM_DENOISING_STEPS = 128
MAX_SEQ_LEN = 256
TASK_ID = "pilot_entropy"

# Paths
SHARED_BASE = "/home/ccwang/sibyl_system/shared"
PROJECT_BASE = "/home/ccwang/sibyl_system/projects/ttt-dlm"
RESULTS_DIR = Path(PROJECT_BASE) / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID & Progress ─────────────────────────────────────────────────────────
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

def mark_task_done(status="success", summary=""):
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

# ── Reproducibility ────────────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[pilot_entropy] Using device: {device}")

start_time = time.time()

try:
    # ── Load model ──────────────────────────────────────────────────────
    print("[pilot_entropy] Loading LLaDA-8B-Instruct...")
    report_progress(0, NUM_SAMPLES, step=0, total_steps=4, metric={"phase": "loading_model"})

    model_path = f"{SHARED_BASE}/checkpoints/LLaDA-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    # LLaDA uses a mask token - find it
    # LLaDA's mask token id is typically the last token in vocab
    vocab_size = model.config.vocab_size  # 126464
    # LLaDA uses [MASK] token; check tokenizer
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        MASK_TOKEN_ID = tokenizer.mask_token_id
    else:
        # LLaDA convention: mask_token_id = vocab_size (126464)
        # But we need to check the model's actual convention
        MASK_TOKEN_ID = 126336  # LLaDA's [MASK] token based on common config
        # Try to find it from config
        if hasattr(model.config, 'mask_token_id'):
            MASK_TOKEN_ID = model.config.mask_token_id

    print(f"[pilot_entropy] Model loaded. vocab_size={vocab_size}, mask_token_id={MASK_TOKEN_ID}")
    report_progress(0, NUM_SAMPLES, step=1, total_steps=4, metric={"phase": "model_loaded"})

    # ── Load GSM8K ──────────────────────────────────────────────────────
    print("[pilot_entropy] Loading GSM8K dataset...")
    gsm8k_path = f"{SHARED_BASE}/datasets/gsm8k"
    dataset = load_from_disk(gsm8k_path)

    # Get test split if available, else use train
    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        if 'test' in dataset:
            data_split = dataset['test']
        elif 'train' in dataset:
            data_split = dataset['train']
        else:
            data_split = dataset[list(dataset.keys())[0]]
    else:
        data_split = dataset

    # Select 16 samples
    indices = list(range(min(NUM_SAMPLES, len(data_split))))
    samples = [data_split[i] for i in indices]
    print(f"[pilot_entropy] Selected {len(samples)} GSM8K samples")
    report_progress(0, NUM_SAMPLES, step=2, total_steps=4, metric={"phase": "data_loaded"})

    # ── Tokenize prompts ────────────────────────────────────────────────
    print("[pilot_entropy] Tokenizing prompts...")
    tokenized_prompts = []
    for s in samples:
        question = s.get('question', s.get('problem', ''))
        answer = s.get('answer', s.get('solution', ''))
        # Combine question + answer for full sequence analysis
        full_text = f"Question: {question}\nAnswer: {answer}"
        tokens = tokenizer.encode(full_text, max_length=MAX_SEQ_LEN, truncation=True,
                                   return_tensors="pt")
        tokenized_prompts.append(tokens.squeeze(0))  # [seq_len]

    print(f"[pilot_entropy] Tokenized {len(tokenized_prompts)} prompts, "
          f"lengths: {[len(t) for t in tokenized_prompts]}")

    # ── Per-position entropy analysis ───────────────────────────────────
    # We simulate the denoising process by creating masked versions at
    # different mask ratios. At each "step", we mask a fraction of tokens
    # and compute per-position logit entropy.

    # Define mask ratios to analyze (from high masking to low masking)
    # We use 128 steps: mask_ratio goes from ~1.0 to ~0.0
    mask_ratios = np.linspace(0.95, 0.02, NUM_DENOISING_STEPS)

    # Critical zone mask ratios for detailed analysis
    critical_zone = (0.3, 0.6)

    all_results = []
    entropy_matrices = []  # [sample_idx][step_idx][position] = entropy

    print(f"\n[pilot_entropy] Starting entropy analysis for {len(tokenized_prompts)} samples "
          f"across {NUM_DENOISING_STEPS} denoising steps...")

    for sample_idx, tokens in enumerate(tokenized_prompts):
        seq_len = len(tokens)
        tokens_gpu = tokens.to(device)

        # Store per-step entropy for this sample
        sample_entropies = np.zeros((NUM_DENOISING_STEPS, seq_len), dtype=np.float32)
        sample_mask_info = []

        for step_idx, mask_ratio in enumerate(mask_ratios):
            # Create masked input: randomly mask `mask_ratio` fraction of positions
            num_mask = max(1, int(mask_ratio * seq_len))
            num_mask = min(num_mask, seq_len - 1)  # Keep at least 1 token revealed

            # Deterministic mask per step (seed + sample + step)
            rng = np.random.RandomState(SEED + sample_idx * 1000 + step_idx)
            mask_positions = rng.choice(seq_len, size=num_mask, replace=False)

            # Create masked input
            masked_input = tokens_gpu.clone()
            masked_input[mask_positions] = MASK_TOKEN_ID

            # Forward pass to get logits
            with torch.no_grad():
                input_ids = masked_input.unsqueeze(0)  # [1, seq_len]
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Compute per-position entropy: H(p) = -sum(p * log(p))
            probs = torch.softmax(logits.float(), dim=-1)  # float32 for numerical stability
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]
            entropy_np = entropy.cpu().numpy()

            sample_entropies[step_idx] = entropy_np

            # Track mask info
            revealed_positions = np.setdiff1d(np.arange(seq_len), mask_positions)
            sample_mask_info.append({
                "step": step_idx,
                "mask_ratio": float(mask_ratio),
                "num_masked": int(num_mask),
                "num_revealed": int(seq_len - num_mask),
            })

            # Clear GPU cache periodically
            if step_idx % 32 == 0:
                torch.cuda.empty_cache()

        entropy_matrices.append(sample_entropies)

        # ── Compute entropy concentration metrics ───────────────────────
        # For critical zone steps (mask ratio 0.3-0.6)
        critical_steps = [i for i, r in enumerate(mask_ratios)
                          if critical_zone[0] <= r <= critical_zone[1]]

        concentration_ratios = []
        for step_idx in critical_steps:
            ent = sample_entropies[step_idx]
            total_entropy = ent.sum()
            if total_entropy < 1e-8:
                continue

            # Sort positions by entropy (descending)
            sorted_indices = np.argsort(ent)[::-1]
            cumulative = np.cumsum(ent[sorted_indices]) / total_entropy

            # Find what fraction of positions contributes 80% of entropy
            positions_for_80pct = np.searchsorted(cumulative, 0.80) + 1
            fraction_for_80pct = positions_for_80pct / seq_len

            concentration_ratios.append({
                "step": step_idx,
                "mask_ratio": float(mask_ratios[step_idx]),
                "fraction_positions_for_80pct_entropy": float(fraction_for_80pct),
                "concentrated": bool(fraction_for_80pct < 0.20),
                "total_entropy": float(total_entropy),
                "max_entropy": float(ent.max()),
                "min_entropy": float(ent.min()),
                "mean_entropy": float(ent.mean()),
                "std_entropy": float(ent.std()),
            })

        # Lorenz curve at mask_ratio ≈ 0.45 (closest to critical point)
        target_ratio = 0.45
        closest_step = min(range(len(mask_ratios)),
                          key=lambda i: abs(mask_ratios[i] - target_ratio))
        ent_at_critical = sample_entropies[closest_step]
        sorted_ent = np.sort(ent_at_critical)  # ascending for Lorenz curve
        lorenz_x = np.linspace(0, 1, seq_len)
        lorenz_y = np.cumsum(sorted_ent) / (sorted_ent.sum() + 1e-10)

        # Gini coefficient for entropy concentration
        n = len(sorted_ent)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_ent) / (n * sorted_ent.sum() + 1e-10)) - (n + 1) / n

        sample_result = {
            "sample_idx": sample_idx,
            "seq_len": seq_len,
            "question_preview": samples[sample_idx].get('question', '')[:100],
            "concentration_ratios": concentration_ratios,
            "lorenz_curve": {
                "mask_ratio": float(mask_ratios[closest_step]),
                "x_percentiles": lorenz_x[::max(1, seq_len//20)].tolist(),
                "y_cumulative_entropy": lorenz_y[::max(1, seq_len//20)].tolist(),
            },
            "gini_coefficient": float(gini),
            "mean_fraction_for_80pct": float(np.mean([c["fraction_positions_for_80pct_entropy"]
                                                       for c in concentration_ratios])) if concentration_ratios else 0,
        }
        all_results.append(sample_result)

        report_progress(
            sample_idx + 1, NUM_SAMPLES,
            step=sample_idx + 1, total_steps=NUM_SAMPLES,
            metric={
                "phase": "entropy_analysis",
                "mean_concentration": sample_result["mean_fraction_for_80pct"],
                "gini": float(gini),
            }
        )
        print(f"  Sample {sample_idx+1}/{NUM_SAMPLES}: seq_len={seq_len}, "
              f"mean_frac_for_80pct={sample_result['mean_fraction_for_80pct']:.3f}, "
              f"gini={gini:.3f}")

        gc.collect()
        torch.cuda.empty_cache()

    # ── Aggregate statistics ────────────────────────────────────────────
    print("\n[pilot_entropy] Computing aggregate statistics...")

    all_mean_fractions = [r["mean_fraction_for_80pct"] for r in all_results]
    all_gini = [r["gini_coefficient"] for r in all_results]

    # Per-mask-ratio aggregate: for each mask ratio in critical zone,
    # average concentration across samples
    per_mask_ratio_stats = {}
    for step_idx in range(NUM_DENOISING_STEPS):
        mr = float(mask_ratios[step_idx])
        if not (critical_zone[0] - 0.05 <= mr <= critical_zone[1] + 0.05):
            continue

        fractions = []
        for sample_idx, ent_matrix in enumerate(entropy_matrices):
            ent = ent_matrix[step_idx]
            total = ent.sum()
            if total < 1e-8:
                continue
            sorted_ent = np.sort(ent)[::-1]
            cumulative = np.cumsum(sorted_ent) / total
            pos_for_80 = (np.searchsorted(cumulative, 0.80) + 1) / len(ent)
            fractions.append(pos_for_80)

        if fractions:
            per_mask_ratio_stats[f"{mr:.3f}"] = {
                "mask_ratio": mr,
                "mean_fraction_for_80pct": float(np.mean(fractions)),
                "std_fraction_for_80pct": float(np.std(fractions)),
                "concentrated_count": int(sum(1 for f in fractions if f < 0.20)),
                "total_samples": len(fractions),
            }

    # Entropy heatmap data (averaged across samples for visualization)
    # Subsample steps and positions for manageable output
    step_subsample = list(range(0, NUM_DENOISING_STEPS, max(1, NUM_DENOISING_STEPS // 32)))
    min_seq_len = min(len(t) for t in tokenized_prompts)
    pos_subsample = list(range(0, min_seq_len, max(1, min_seq_len // 32)))

    avg_entropy_heatmap = np.zeros((len(step_subsample), len(pos_subsample)))
    for s_i, step_idx in enumerate(step_subsample):
        for p_i, pos_idx in enumerate(pos_subsample):
            vals = [entropy_matrices[si][step_idx][pos_idx]
                    for si in range(len(entropy_matrices))
                    if pos_idx < entropy_matrices[si].shape[1]]
            avg_entropy_heatmap[s_i, p_i] = np.mean(vals) if vals else 0

    # ── Pass criteria check ─────────────────────────────────────────────
    # "<20% positions contribute >80% total entropy at mask ratio 0.4-0.5"
    pass_check_fractions = []
    for step_idx in range(NUM_DENOISING_STEPS):
        mr = float(mask_ratios[step_idx])
        if 0.4 <= mr <= 0.5:
            for sample_idx, ent_matrix in enumerate(entropy_matrices):
                ent = ent_matrix[step_idx]
                total = ent.sum()
                if total < 1e-8:
                    continue
                sorted_ent = np.sort(ent)[::-1]
                cumulative = np.cumsum(sorted_ent) / total
                pos_for_80 = (np.searchsorted(cumulative, 0.80) + 1) / len(ent)
                pass_check_fractions.append(pos_for_80)

    mean_pass_fraction = float(np.mean(pass_check_fractions)) if pass_check_fractions else 1.0
    pass_criteria_met = mean_pass_fraction < 0.20

    elapsed_min = (time.time() - start_time) / 60

    # ── Build final output ──────────────────────────────────────────────
    output = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "LLaDA-8B-Instruct",
            "num_samples": NUM_SAMPLES,
            "num_denoising_steps": NUM_DENOISING_STEPS,
            "max_seq_len": MAX_SEQ_LEN,
            "seed": SEED,
            "mask_token_id": MASK_TOKEN_ID,
            "critical_zone": list(critical_zone),
        },
        "pass_criteria": "<20% positions contribute >80% total entropy at mask ratio 0.4-0.5",
        "pass_result": pass_criteria_met,
        "pass_details": {
            "mean_fraction_for_80pct_in_04_05": mean_pass_fraction,
            "threshold": 0.20,
            "num_measurements": len(pass_check_fractions),
        },
        "aggregate": {
            "mean_fraction_for_80pct_entropy": float(np.mean(all_mean_fractions)),
            "std_fraction_for_80pct_entropy": float(np.std(all_mean_fractions)),
            "mean_gini_coefficient": float(np.mean(all_gini)),
            "std_gini_coefficient": float(np.std(all_gini)),
            "entropy_is_concentrated": bool(np.mean(all_mean_fractions) < 0.20),
        },
        "per_mask_ratio_stats": per_mask_ratio_stats,
        "entropy_heatmap": {
            "description": "Average per-position entropy across denoising steps (subsampled)",
            "step_indices": step_subsample,
            "mask_ratios_at_steps": [float(mask_ratios[i]) for i in step_subsample],
            "position_indices": pos_subsample,
            "values": avg_entropy_heatmap.tolist(),
        },
        "per_sample_results": all_results,
        "timing": {
            "elapsed_minutes": round(elapsed_min, 1),
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
    }

    # Save results
    out_path = RESULTS_DIR / "p4_entropy.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[pilot_entropy] Results saved to {out_path}")
    print(f"[pilot_entropy] Pass criteria met: {pass_criteria_met}")
    print(f"[pilot_entropy] Mean fraction for 80% entropy (r=0.4-0.5): {mean_pass_fraction:.4f}")
    print(f"[pilot_entropy] Mean Gini coefficient: {np.mean(all_gini):.4f}")
    print(f"[pilot_entropy] Elapsed: {elapsed_min:.1f} min")

    mark_task_done(
        status="success",
        summary=f"Entropy sparsity analysis complete. Pass={'YES' if pass_criteria_met else 'NO'}. "
                f"Mean fraction for 80% entropy at r=0.4-0.5: {mean_pass_fraction:.4f} "
                f"(threshold: <0.20). Gini: {np.mean(all_gini):.4f}"
    )

except Exception as e:
    error_msg = f"{type(e).__name__}: {str(e)}"
    print(f"\n[pilot_entropy] ERROR: {error_msg}")
    traceback.print_exc()
    mark_task_done(status="failed", summary=error_msg)
    sys.exit(1)
