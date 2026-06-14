#!/usr/bin/env python3
"""
P4: Per-Position Entropy Sparsity Analysis (pilot_entropy) — v2, 100 samples

On LLaDA-8B, for 100 GSM8K prompts, record per-position logit entropy at each
denoising step (128 steps). At mask ratios 0.3-0.6 (critical zone), compute:
  (a) entropy concentration ratio (do <20% positions contribute >80% entropy?)
  (b) correlation between high-entropy positions and final answer errors

Validates precision-weighting hypothesis (H4).

Changes from v1:
  - 100 samples (was 16)
  - Adds error-position correlation analysis
  - Memory-efficient: processes samples one at a time, stores only aggregates
  - Subsamples denoising steps for efficiency (every 4th step -> 32 steps analyzed)

Usage:
  CUDA_VISIBLE_DEVICES=2 python pilot_entropy_v2.py
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
NUM_SAMPLES = 100
NUM_DENOISING_STEPS = 128
STEP_SUBSAMPLE = 4  # Analyze every Nth step for efficiency -> 32 steps
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

    # LLaDA mask token
    vocab_size = model.config.vocab_size
    if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id is not None:
        MASK_TOKEN_ID = tokenizer.mask_token_id
    else:
        MASK_TOKEN_ID = 126336
        if hasattr(model.config, 'mask_token_id'):
            MASK_TOKEN_ID = model.config.mask_token_id

    print(f"[pilot_entropy] Model loaded. vocab_size={vocab_size}, mask_token_id={MASK_TOKEN_ID}")
    report_progress(0, NUM_SAMPLES, step=1, total_steps=4, metric={"phase": "model_loaded"})

    # ── Load GSM8K ──────────────────────────────────────────────────────
    print("[pilot_entropy] Loading GSM8K dataset...")
    gsm8k_path = f"{SHARED_BASE}/datasets/gsm8k"
    dataset = load_from_disk(gsm8k_path)

    if isinstance(dataset, dict) or hasattr(dataset, 'keys'):
        if 'test' in dataset:
            data_split = dataset['test']
        elif 'train' in dataset:
            data_split = dataset['train']
        else:
            data_split = dataset[list(dataset.keys())[0]]
    else:
        data_split = dataset

    indices = list(range(min(NUM_SAMPLES, len(data_split))))
    samples = [data_split[i] for i in indices]
    print(f"[pilot_entropy] Selected {len(samples)} GSM8K samples")
    report_progress(0, NUM_SAMPLES, step=2, total_steps=4, metric={"phase": "data_loaded"})

    # ── Tokenize prompts ────────────────────────────────────────────────
    print("[pilot_entropy] Tokenizing prompts...")
    tokenized_prompts = []
    answer_start_positions = []  # Track where answer starts for error correlation

    for s in samples:
        question = s.get('question', s.get('problem', ''))
        answer = s.get('answer', s.get('solution', ''))

        # Tokenize question and answer separately to find boundary
        q_text = f"Question: {question}\nAnswer: "
        q_tokens = tokenizer.encode(q_text, add_special_tokens=False)
        a_tokens = tokenizer.encode(answer, add_special_tokens=False)

        full_tokens = q_tokens + a_tokens
        if len(full_tokens) > MAX_SEQ_LEN:
            full_tokens = full_tokens[:MAX_SEQ_LEN]

        tokenized_prompts.append(torch.tensor(full_tokens))
        answer_start_positions.append(min(len(q_tokens), len(full_tokens) - 1))

    print(f"[pilot_entropy] Tokenized {len(tokenized_prompts)} prompts, "
          f"lengths: min={min(len(t) for t in tokenized_prompts)}, "
          f"max={max(len(t) for t in tokenized_prompts)}, "
          f"mean={np.mean([len(t) for t in tokenized_prompts]):.0f}")

    # ── Define analysis steps ────────────────────────────────────────────
    # Full 128 steps but only analyze every STEP_SUBSAMPLE-th for efficiency
    mask_ratios_full = np.linspace(0.95, 0.02, NUM_DENOISING_STEPS)
    analysis_step_indices = list(range(0, NUM_DENOISING_STEPS, STEP_SUBSAMPLE))
    analysis_mask_ratios = mask_ratios_full[analysis_step_indices]
    n_analysis_steps = len(analysis_step_indices)

    critical_zone = (0.3, 0.6)
    print(f"[pilot_entropy] Analyzing {n_analysis_steps} denoising steps "
          f"(every {STEP_SUBSAMPLE}th of {NUM_DENOISING_STEPS})")

    # ── Per-sample entropy analysis ──────────────────────────────────────
    # Accumulate statistics without storing full entropy matrices (memory efficient)

    # Per-mask-ratio accumulators
    per_ratio_fraction_accum = {}  # mask_ratio_key -> list of fractions
    per_ratio_entropy_stats = {}   # mask_ratio_key -> list of (mean, std, max, min)

    # Per-sample summary
    all_sample_results = []

    # Error-position correlation accumulators
    # For each sample in critical zone: track (high_entropy_positions, answer_positions)
    error_correlation_data = []

    # Heatmap accumulator: average entropy at subsampled (step, position)
    HEATMAP_POS_BINS = 32
    HEATMAP_STEP_BINS = n_analysis_steps
    heatmap_sum = np.zeros((HEATMAP_STEP_BINS, HEATMAP_POS_BINS), dtype=np.float64)
    heatmap_count = np.zeros((HEATMAP_STEP_BINS, HEATMAP_POS_BINS), dtype=np.int64)

    print(f"\n[pilot_entropy] Starting entropy analysis for {len(tokenized_prompts)} samples...")

    for sample_idx, tokens in enumerate(tokenized_prompts):
        seq_len = len(tokens)
        tokens_gpu = tokens.to(device)
        ans_start = answer_start_positions[sample_idx]

        sample_concentration_ratios = []
        sample_entropy_at_answer = []  # entropy at answer positions in critical zone

        for ai, step_idx in enumerate(analysis_step_indices):
            mask_ratio = mask_ratios_full[step_idx]

            # Create masked input
            num_mask = max(1, int(mask_ratio * seq_len))
            num_mask = min(num_mask, seq_len - 1)

            rng = np.random.RandomState(SEED + sample_idx * 1000 + step_idx)
            mask_positions = rng.choice(seq_len, size=num_mask, replace=False)

            masked_input = tokens_gpu.clone()
            masked_input[mask_positions] = MASK_TOKEN_ID

            # Forward pass
            with torch.no_grad():
                input_ids = masked_input.unsqueeze(0)
                outputs = model(input_ids=input_ids)
                logits = outputs.logits[0]  # [seq_len, vocab_size]

            # Per-position entropy
            probs = torch.softmax(logits.float(), dim=-1)
            log_probs = torch.log(probs + 1e-10)
            entropy = -(probs * log_probs).sum(dim=-1)  # [seq_len]
            entropy_np = entropy.cpu().numpy()

            # ── Heatmap accumulation ──
            for p_bin in range(HEATMAP_POS_BINS):
                p_start = int(p_bin * seq_len / HEATMAP_POS_BINS)
                p_end = int((p_bin + 1) * seq_len / HEATMAP_POS_BINS)
                if p_start < p_end:
                    heatmap_sum[ai, p_bin] += entropy_np[p_start:p_end].mean()
                    heatmap_count[ai, p_bin] += 1

            # ── Concentration ratio ──
            total_entropy = entropy_np.sum()
            mr_key = f"{mask_ratio:.3f}"

            if total_entropy > 1e-8:
                sorted_ent = np.sort(entropy_np)[::-1]
                cumulative = np.cumsum(sorted_ent) / total_entropy
                pos_for_80 = (np.searchsorted(cumulative, 0.80) + 1) / seq_len
                fraction_for_80pct = float(pos_for_80)

                if mr_key not in per_ratio_fraction_accum:
                    per_ratio_fraction_accum[mr_key] = []
                    per_ratio_entropy_stats[mr_key] = []
                per_ratio_fraction_accum[mr_key].append(fraction_for_80pct)
                per_ratio_entropy_stats[mr_key].append({
                    "mean": float(entropy_np.mean()),
                    "std": float(entropy_np.std()),
                    "max": float(entropy_np.max()),
                    "min": float(entropy_np.min()),
                })

                # Track for critical zone
                if critical_zone[0] <= mask_ratio <= critical_zone[1]:
                    sample_concentration_ratios.append({
                        "step": step_idx,
                        "mask_ratio": float(mask_ratio),
                        "fraction_positions_for_80pct_entropy": fraction_for_80pct,
                        "concentrated": fraction_for_80pct < 0.20,
                    })

                    # Error-position correlation: entropy at answer region vs question region
                    if ans_start < seq_len:
                        ent_question = entropy_np[:ans_start].mean() if ans_start > 0 else 0.0
                        ent_answer = entropy_np[ans_start:].mean() if ans_start < seq_len else 0.0

                        # Top-20% highest-entropy positions
                        top_k = max(1, int(0.2 * seq_len))
                        top_positions = np.argsort(entropy_np)[-top_k:]
                        frac_top_in_answer = np.mean(top_positions >= ans_start)

                        sample_entropy_at_answer.append({
                            "mask_ratio": float(mask_ratio),
                            "ent_question_mean": float(ent_question),
                            "ent_answer_mean": float(ent_answer),
                            "frac_high_entropy_in_answer": float(frac_top_in_answer),
                        })

            # Clear GPU periodically
            if ai % 8 == 0:
                del logits, probs, log_probs, entropy, outputs
                torch.cuda.empty_cache()

        # ── Lorenz curve at mask_ratio ~ 0.45 ──
        target_ratio = 0.45
        closest_ai = min(range(n_analysis_steps),
                         key=lambda i: abs(analysis_mask_ratios[i] - target_ratio))
        closest_step = analysis_step_indices[closest_ai]
        closest_mr = analysis_mask_ratios[closest_ai]

        # Re-compute entropy at this specific step for Lorenz curve
        num_mask_lz = max(1, int(closest_mr * seq_len))
        num_mask_lz = min(num_mask_lz, seq_len - 1)
        rng_lz = np.random.RandomState(SEED + sample_idx * 1000 + closest_step)
        mask_pos_lz = rng_lz.choice(seq_len, size=num_mask_lz, replace=False)
        masked_lz = tokens_gpu.clone()
        masked_lz[mask_pos_lz] = MASK_TOKEN_ID
        with torch.no_grad():
            out_lz = model(input_ids=masked_lz.unsqueeze(0))
            logits_lz = out_lz.logits[0]
        probs_lz = torch.softmax(logits_lz.float(), dim=-1)
        ent_lz = -(probs_lz * torch.log(probs_lz + 1e-10)).sum(dim=-1).cpu().numpy()

        sorted_ent_lz = np.sort(ent_lz)
        gini_n = len(sorted_ent_lz)
        gini = (2 * np.sum((np.arange(1, gini_n + 1)) * sorted_ent_lz) /
                (gini_n * sorted_ent_lz.sum() + 1e-10)) - (gini_n + 1) / gini_n

        # Lorenz curve (subsample for output size)
        lorenz_y = np.cumsum(sorted_ent_lz) / (sorted_ent_lz.sum() + 1e-10)
        lorenz_subsample = max(1, seq_len // 20)
        lorenz_x_sub = np.linspace(0, 1, seq_len)[::lorenz_subsample].tolist()
        lorenz_y_sub = lorenz_y[::lorenz_subsample].tolist()

        mean_conc = (np.mean([c["fraction_positions_for_80pct_entropy"]
                              for c in sample_concentration_ratios])
                     if sample_concentration_ratios else 0.0)

        # Error correlation for this sample
        if sample_entropy_at_answer:
            mean_frac_high_in_answer = np.mean([e["frac_high_entropy_in_answer"]
                                                 for e in sample_entropy_at_answer])
            error_correlation_data.append({
                "sample_idx": sample_idx,
                "answer_start_frac": ans_start / seq_len,
                "mean_frac_high_entropy_in_answer": float(mean_frac_high_in_answer),
                "ent_answer_vs_question_ratio": float(
                    np.mean([e["ent_answer_mean"] / (e["ent_question_mean"] + 1e-10)
                             for e in sample_entropy_at_answer])
                ),
            })

        sample_result = {
            "sample_idx": sample_idx,
            "seq_len": seq_len,
            "answer_start_pos": ans_start,
            "question_preview": samples[sample_idx].get('question', '')[:80],
            "mean_fraction_for_80pct": float(mean_conc),
            "gini_coefficient": float(gini),
            "lorenz_curve": {
                "mask_ratio": float(closest_mr),
                "x_percentiles": lorenz_x_sub,
                "y_cumulative_entropy": lorenz_y_sub,
            },
            "n_critical_steps": len(sample_concentration_ratios),
        }
        all_sample_results.append(sample_result)

        del logits_lz, probs_lz, out_lz, ent_lz
        gc.collect()
        torch.cuda.empty_cache()

        report_progress(
            sample_idx + 1, NUM_SAMPLES,
            step=sample_idx + 1, total_steps=NUM_SAMPLES,
            metric={
                "phase": "entropy_analysis",
                "mean_concentration": float(mean_conc),
                "gini": float(gini),
            }
        )
        if (sample_idx + 1) % 10 == 0 or sample_idx == 0:
            elapsed = (time.time() - start_time) / 60
            eta = elapsed / (sample_idx + 1) * (NUM_SAMPLES - sample_idx - 1)
            print(f"  Sample {sample_idx+1}/{NUM_SAMPLES}: "
                  f"frac_80pct={mean_conc:.3f}, gini={gini:.3f} "
                  f"[{elapsed:.1f}min elapsed, ~{eta:.1f}min remaining]")

    # ── Aggregate statistics ────────────────────────────────────────────
    print("\n[pilot_entropy] Computing aggregate statistics...")

    all_mean_fractions = [r["mean_fraction_for_80pct"] for r in all_sample_results]
    all_gini = [r["gini_coefficient"] for r in all_sample_results]

    # Per-mask-ratio stats
    per_mask_ratio_summary = {}
    for mr_key, fracs in per_ratio_fraction_accum.items():
        mr_float = float(mr_key)
        if not (critical_zone[0] - 0.05 <= mr_float <= critical_zone[1] + 0.05):
            continue
        per_mask_ratio_summary[mr_key] = {
            "mask_ratio": mr_float,
            "mean_fraction_for_80pct": float(np.mean(fracs)),
            "std_fraction_for_80pct": float(np.std(fracs)),
            "concentrated_count": int(sum(1 for f in fracs if f < 0.20)),
            "total_samples": len(fracs),
        }

    # ── Pass criteria check ─────────────────────────────────────────────
    pass_check_fractions = []
    for mr_key, fracs in per_ratio_fraction_accum.items():
        mr_float = float(mr_key)
        if 0.4 <= mr_float <= 0.5:
            pass_check_fractions.extend(fracs)

    mean_pass_fraction = float(np.mean(pass_check_fractions)) if pass_check_fractions else 1.0
    pass_criteria_met = mean_pass_fraction < 0.20

    # ── Error-position correlation analysis ──────────────────────────────
    print("[pilot_entropy] Computing error-position correlation...")
    if error_correlation_data:
        frac_high_in_answer_list = [e["mean_frac_high_entropy_in_answer"]
                                     for e in error_correlation_data]
        answer_frac_list = [e["answer_start_frac"] for e in error_correlation_data]
        ent_ratio_list = [e["ent_answer_vs_question_ratio"] for e in error_correlation_data]

        # If answer region is ~50% of sequence, random would give ~50% of top-entropy
        # positions in answer. If significantly higher, entropy concentrates in answer.
        answer_region_frac = 1.0 - np.mean(answer_frac_list)  # fraction of seq that is answer
        expected_random = answer_region_frac
        observed = np.mean(frac_high_in_answer_list)

        error_correlation_summary = {
            "n_samples": len(error_correlation_data),
            "mean_answer_region_fraction": float(answer_region_frac),
            "mean_frac_high_entropy_in_answer": float(observed),
            "expected_random_frac": float(expected_random),
            "concentration_ratio": float(observed / (expected_random + 1e-10)),
            "entropy_concentrates_in_answer": bool(observed > expected_random * 1.2),
            "mean_answer_question_entropy_ratio": float(np.mean(ent_ratio_list)),
            "std_answer_question_entropy_ratio": float(np.std(ent_ratio_list)),
        }
    else:
        error_correlation_summary = {"n_samples": 0, "note": "No error correlation data"}

    # ── Heatmap ──────────────────────────────────────────────────────────
    heatmap_avg = np.divide(heatmap_sum, heatmap_count,
                            out=np.zeros_like(heatmap_sum),
                            where=heatmap_count > 0)

    elapsed_min = (time.time() - start_time) / 60

    # ── Build final output ──────────────────────────────────────────────
    output = {
        "task_id": TASK_ID,
        "status": "success",
        "timestamp": datetime.now().isoformat(),
        "version": "v2",
        "config": {
            "model": "LLaDA-8B-Instruct",
            "num_samples": NUM_SAMPLES,
            "num_denoising_steps": NUM_DENOISING_STEPS,
            "step_subsample": STEP_SUBSAMPLE,
            "n_analysis_steps": n_analysis_steps,
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
        "error_position_correlation": error_correlation_summary,
        "per_mask_ratio_stats": per_mask_ratio_summary,
        "entropy_heatmap": {
            "description": "Average per-position entropy across denoising steps (binned)",
            "step_indices": analysis_step_indices,
            "mask_ratios_at_steps": [float(mr) for mr in analysis_mask_ratios],
            "position_bins": HEATMAP_POS_BINS,
            "values": heatmap_avg.tolist(),
        },
        "lorenz_curve_aggregate": {
            "description": "Aggregate Lorenz curve at mask_ratio ~0.45",
            "mean_gini": float(np.mean(all_gini)),
            "std_gini": float(np.std(all_gini)),
        },
        "per_sample_summary": [
            {
                "sample_idx": r["sample_idx"],
                "seq_len": r["seq_len"],
                "mean_fraction_for_80pct": r["mean_fraction_for_80pct"],
                "gini_coefficient": r["gini_coefficient"],
            }
            for r in all_sample_results
        ],
        "timing": {
            "elapsed_minutes": round(elapsed_min, 1),
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "samples_per_minute": round(NUM_SAMPLES / elapsed_min, 2) if elapsed_min > 0 else 0,
        },
    }

    # Save results
    out_path = RESULTS_DIR / "p4_entropy.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(f"\n[pilot_entropy] Results saved to {out_path}")
    print(f"[pilot_entropy] Pass criteria met: {pass_criteria_met}")
    print(f"[pilot_entropy] Mean fraction for 80% entropy (r=0.4-0.5): {mean_pass_fraction:.4f}")
    print(f"[pilot_entropy] Mean Gini coefficient: {np.mean(all_gini):.4f}")
    print(f"[pilot_entropy] Error correlation - high entropy in answer region: "
          f"{error_correlation_summary.get('mean_frac_high_entropy_in_answer', 'N/A')}")
    print(f"[pilot_entropy] Elapsed: {elapsed_min:.1f} min")

    mark_task_done(
        status="success",
        summary=(
            f"Entropy sparsity analysis (v2, n={NUM_SAMPLES}) complete. "
            f"Pass={'YES' if pass_criteria_met else 'NO'}. "
            f"Mean fraction for 80% entropy at r=0.4-0.5: {mean_pass_fraction:.4f} "
            f"(threshold: <0.20). Gini: {np.mean(all_gini):.4f}. "
            f"Error-position correlation: high-entropy positions "
            f"{'concentrate' if error_correlation_summary.get('entropy_concentrates_in_answer', False) else 'do NOT concentrate'} "
            f"in answer region."
        )
    )

except Exception as e:
    error_msg = f"{type(e).__name__}: {str(e)}"
    print(f"\n[pilot_entropy] ERROR: {error_msg}")
    traceback.print_exc()
    mark_task_done(status="failed", summary=error_msg)
    sys.exit(1)
