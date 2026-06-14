#!/usr/bin/env python3
"""
DLM Calibration Profiling (H1 + H2) — PILOT mode, v2
Task: calibration_study

Measures calibration of LLaDA-8B-Instruct during denoising with TWO methods:

METHOD A — "Self-consistency calibration":
  Run standard denoising. At each step, record confidence for each masked position.
  After full denoising completes, compare each step's top-1 prediction against the
  FINAL committed token at that position. This measures whether early-step confidence
  predicts eventual commitment stability.

METHOD B — "Oracle calibration" (teacher-forced):
  Given the reference answer, mask a fraction of tokens (matching each denoising stage),
  and measure whether the model can recover the masked tokens. This directly measures
  p(correct | confidence) in a controlled setting.

Both methods compute ECE per stage band and entropy-error correlation.

Usage:
    CUDA_VISIBLE_DEVICES=2 python calibration_study.py
"""

import os
import sys
import gc
import json
import time
import math
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from datasets import load_dataset

warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────
TASK_ID = "calibration_study"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
PILOTS_DIR = RESULTS_DIR / "pilots"
SEED = 42
NUM_SAMPLES = 100
NUM_STEPS = 64
GEN_LENGTH = 256
MASK_TOKEN_ID = 126336
EOS_TOKEN_ID = 126081
DEVICE = "cuda"

# Stage bands: fraction of positions still masked
STAGE_BANDS = [(0.80, 1.00), (0.50, 0.80), (0.20, 0.50), (0.00, 0.20)]
STAGE_BAND_NAMES = ["early_80_100", "mid_50_80", "mid_20_50", "late_0_20"]
ECE_BINS = 10

# Oracle calibration: masking ratios to test
ORACLE_MASK_RATIOS = [0.9, 0.7, 0.5, 0.3, 0.1]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0,
                    loss=None, metric=None):
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


def cosine_schedule(t, T):
    """Fraction unmasked after step t (0..T). t=0 -> 0, t=T -> 1."""
    return 1.0 - math.cos(math.pi * t / (2 * T))


def compute_ece(confidences, accuracies, n_bins=10):
    if len(confidences) == 0:
        return 0.0, [], [], []
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_ece = 0.0
    bin_confs, bin_accs, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == n_bins - 1 \
            else (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_confs.append((lo + hi) / 2)
            bin_accs.append(0.0)
            bin_counts.append(0)
            continue
        bc = confidences[mask].mean()
        ba = accuracies[mask].mean()
        bn = mask.sum()
        bin_ece += (bn / len(confidences)) * abs(ba - bc)
        bin_confs.append(float(bc))
        bin_accs.append(float(ba))
        bin_counts.append(int(bn))
    return float(bin_ece), bin_confs, bin_accs, bin_counts


def assign_stage_band(masking_ratio):
    for i, (lo, hi) in enumerate(STAGE_BANDS):
        if lo <= masking_ratio <= hi:
            return i
    return 0 if masking_ratio > 1.0 else 3


def get_gsm8k_prompts(tokenizer, num_samples=100):
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = []
    for idx in range(min(num_samples, len(dataset))):
        question = dataset[idx]["question"]
        answer = dataset[idx]["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        samples.append({
            "idx": idx,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids[:GEN_LENGTH],
            "final_answer": final_answer,
            "question": question,
        })
    return samples


# ══════════════════════════════════════════════════════════════════════
# METHOD A: Self-Consistency Calibration
# ══════════════════════════════════════════════════════════════════════

def run_denoising_self_consistency(model, prompt_ids, gen_len, num_steps=64):
    """
    Run denoising and compare each step's predictions to the FINAL tokens.
    Returns per-step records with confidence, entropy, and is_correct
    (where 'correct' = matches the final committed token, not any reference).
    """
    device = next(model.parameters()).device

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = torch.cat([
        prompt_tensor,
        torch.full((gen_len,), MASK_TOKEN_ID, dtype=torch.long, device=device)
    ]).unsqueeze(0)

    prompt_len = len(prompt_ids)
    gen_start = prompt_len
    gen_end = prompt_len + gen_len

    # Store predictions at each step for all positions
    # step_predictions[step] = {pos_idx: (confidence, token_id, entropy)}
    step_predictions = []

    for step in range(num_steps):
        frac_unmasked_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
        frac_unmasked_curr = cosine_schedule(step + 1, num_steps)

        gen_region = input_ids[0, gen_start:gen_end]
        is_masked = (gen_region == MASK_TOKEN_ID)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        masking_ratio = num_masked / gen_len

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]

        gen_logits = logits[gen_start:gen_end]
        masked_indices = torch.where(is_masked)[0]
        masked_logits = gen_logits[masked_indices]

        probs = F.softmax(masked_logits, dim=-1)
        top1_conf, top1_token = probs.max(dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Record predictions for masked positions
        step_preds = {}
        for j in range(len(masked_indices)):
            pos = masked_indices[j].item()
            step_preds[pos] = {
                "confidence": top1_conf[j].item(),
                "token": top1_token[j].item(),
                "entropy": entropy[j].item(),
                "masking_ratio": masking_ratio,
            }
        step_predictions.append(step_preds)

        # Unmask highest-confidence
        num_to_unmask = max(1, int(round((frac_unmasked_curr - frac_unmasked_prev) * gen_len)))
        num_to_unmask = min(num_to_unmask, num_masked)
        _, topk_indices = top1_conf.topk(num_to_unmask)
        positions_to_unmask = masked_indices[topk_indices]
        tokens_to_place = top1_token[topk_indices]
        input_ids[0, gen_start + positions_to_unmask] = tokens_to_place

    # Final committed tokens
    final_tokens = input_ids[0, gen_start:gen_end].cpu().numpy()

    # Now compute is_correct for each step's predictions
    records = []
    for step_idx, step_preds in enumerate(step_predictions):
        for pos, pred in step_preds.items():
            is_correct = (pred["token"] == final_tokens[pos])
            records.append({
                "step": step_idx,
                "position": pos,
                "masking_ratio": pred["masking_ratio"],
                "confidence": pred["confidence"],
                "entropy": pred["entropy"],
                "is_correct": int(is_correct),
            })

    return records, final_tokens


# ══════════════════════════════════════════════════════════════════════
# METHOD B: Oracle (Teacher-Forced) Calibration
# ══════════════════════════════════════════════════════════════════════

def run_oracle_calibration(model, prompt_ids, answer_ids, mask_ratios):
    """
    Given known reference tokens, mask a fraction and measure recovery accuracy.
    This gives the cleanest calibration signal.
    """
    device = next(model.parameters()).device
    gen_len = len(answer_ids)

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    answer_tensor = torch.tensor(answer_ids, dtype=torch.long, device=device)

    # Full sequence = prompt + answer
    full_seq = torch.cat([prompt_tensor, answer_tensor]).unsqueeze(0)
    prompt_len = len(prompt_ids)
    gen_start = prompt_len
    gen_end = prompt_len + gen_len

    records = []

    for mask_ratio in mask_ratios:
        num_to_mask = max(1, int(round(mask_ratio * gen_len)))

        # Randomly select positions to mask
        rng = np.random.RandomState(SEED + int(mask_ratio * 1000))
        mask_positions = rng.choice(gen_len, size=num_to_mask, replace=False)
        mask_positions = torch.tensor(mask_positions, dtype=torch.long, device=device)

        # Create masked input
        masked_input = full_seq.clone()
        masked_input[0, gen_start + mask_positions] = MASK_TOKEN_ID

        # Forward pass
        with torch.no_grad():
            outputs = model(input_ids=masked_input)
            logits = outputs.logits[0]

        # Get predictions at masked positions
        masked_logits = logits[gen_start + mask_positions]  # [num_masked, vocab]
        probs = F.softmax(masked_logits, dim=-1)
        top1_conf, top1_token = probs.max(dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Compare to reference
        ref_tokens = answer_tensor[mask_positions]
        is_correct = (top1_token == ref_tokens)

        for j in range(len(mask_positions)):
            records.append({
                "mask_ratio": mask_ratio,
                "position": mask_positions[j].item(),
                "confidence": top1_conf[j].item(),
                "entropy": entropy[j].item(),
                "is_correct": int(is_correct[j].item()),
                "predicted_token": top1_token[j].item(),
                "reference_token": ref_tokens[j].item(),
            })

    return records


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    write_pid()
    PILOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{TASK_ID}] Starting calibration study v2")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Samples: {NUM_SAMPLES}, Steps: {NUM_STEPS}")
    print(f"  Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print("\n[1/5] Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE).eval()
    vram_after_load = torch.cuda.memory_allocated() / 1e6
    print(f"  Model loaded. VRAM: {vram_after_load:.0f} MB")

    # Load GSM8K
    print("\n[2/5] Loading GSM8K samples...")
    samples = get_gsm8k_prompts(tokenizer, NUM_SAMPLES)
    print(f"  Loaded {len(samples)} samples")
    print(f"  Avg prompt length: {np.mean([len(s['prompt_ids']) for s in samples]):.0f} tokens")
    print(f"  Avg answer length: {np.mean([len(s['answer_ids']) for s in samples]):.0f} tokens")

    # ── METHOD A: Self-Consistency ────────────────────────────────────
    print(f"\n[3/5] Method A: Self-consistency calibration ({len(samples)} samples)...")
    sc_band_data = {i: {"confidences": [], "accuracies": [], "entropies": [], "is_errors": []}
                    for i in range(len(STAGE_BANDS))}

    for si, sample in enumerate(samples):
        if si % 10 == 0:
            elapsed = time.time() - start_time
            report_progress(epoch=si, total_epochs=len(samples) * 2,  # 2 methods
                            metric={"phase": "self_consistency", "samples_done": si,
                                    "elapsed_sec": round(elapsed, 1)})
            print(f"  [A] Sample {si}/{len(samples)}, elapsed: {elapsed:.0f}s")

        gen_len = min(len(sample["answer_ids"]), GEN_LENGTH)
        records, _ = run_denoising_self_consistency(
            model, sample["prompt_ids"], gen_len, num_steps=NUM_STEPS
        )

        for rec in records:
            band_idx = assign_stage_band(rec["masking_ratio"])
            sc_band_data[band_idx]["confidences"].append(rec["confidence"])
            sc_band_data[band_idx]["accuracies"].append(rec["is_correct"])
            sc_band_data[band_idx]["entropies"].append(rec["entropy"])
            sc_band_data[band_idx]["is_errors"].append(1 - rec["is_correct"])

        if si % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    elapsed_a = time.time() - start_time
    print(f"  Method A complete: {elapsed_a:.0f}s")

    # ── METHOD B: Oracle Calibration ──────────────────────────────────
    print(f"\n[4/5] Method B: Oracle (teacher-forced) calibration ({len(samples)} samples)...")
    oracle_data = {mr: {"confidences": [], "accuracies": [], "entropies": [], "is_errors": []}
                   for mr in ORACLE_MASK_RATIOS}

    for si, sample in enumerate(samples):
        if si % 10 == 0:
            elapsed = time.time() - start_time
            report_progress(epoch=len(samples) + si, total_epochs=len(samples) * 2,
                            metric={"phase": "oracle", "samples_done": si,
                                    "elapsed_sec": round(elapsed, 1)})
            print(f"  [B] Sample {si}/{len(samples)}, elapsed: {elapsed:.0f}s")

        records = run_oracle_calibration(
            model, sample["prompt_ids"], sample["answer_ids"], ORACLE_MASK_RATIOS
        )

        for rec in records:
            mr = rec["mask_ratio"]
            oracle_data[mr]["confidences"].append(rec["confidence"])
            oracle_data[mr]["accuracies"].append(rec["is_correct"])
            oracle_data[mr]["entropies"].append(rec["entropy"])
            oracle_data[mr]["is_errors"].append(1 - rec["is_correct"])

        if si % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    elapsed_b = time.time() - start_time
    print(f"  Method B complete: {elapsed_b:.0f}s")

    # ── Compute metrics ───────────────────────────────────────────────
    print("\n[5/5] Computing calibration metrics...")

    # Method A metrics
    sc_profile = {}
    sc_calibrators = {}
    for band_idx in range(len(STAGE_BANDS)):
        band_name = STAGE_BAND_NAMES[band_idx]
        bd = sc_band_data[band_idx]
        confs = np.array(bd["confidences"])
        accs = np.array(bd["accuracies"])
        ents = np.array(bd["entropies"])
        errs = np.array(bd["is_errors"])

        n = len(confs)
        if n == 0:
            sc_profile[band_name] = {"n_points": 0}
            continue

        ece, bin_confs, bin_accs, bin_counts = compute_ece(confs, accs, ECE_BINS)
        pearson_r, pearson_p = (stats.pearsonr(ents, errs)
                                if np.std(ents) > 1e-8 and np.std(errs) > 1e-8
                                else (0.0, 1.0))

        # Fit calibrator
        if n >= 20:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(confs, accs)
            sc_calibrators[band_name] = iso
            cal_confs = iso.predict(confs)
            cal_ece, cal_bconfs, cal_baccs, cal_bcounts = compute_ece(cal_confs, accs, ECE_BINS)
        else:
            cal_ece = None
            cal_bconfs, cal_baccs, cal_bcounts = [], [], []

        sc_profile[band_name] = {
            "stage_band": list(STAGE_BANDS[band_idx]),
            "n_points": n,
            "ece": ece,
            "ece_calibrated": cal_ece,
            "pearson_entropy_error": {"r": float(pearson_r), "p_value": float(pearson_p)},
            "mean_confidence": float(confs.mean()),
            "mean_accuracy": float(accs.mean()),
            "mean_entropy": float(ents.mean()),
            "mean_error_rate": float(errs.mean()),
            "reliability_diagram": {
                "bin_confidences": bin_confs, "bin_accuracies": bin_accs, "bin_counts": bin_counts,
            },
            "calibrated_reliability_diagram": {
                "bin_confidences": cal_bconfs, "bin_accuracies": cal_baccs, "bin_counts": cal_bcounts,
            },
        }

        print(f"  [A] {band_name}: N={n:,}, ECE={ece:.4f}, Cal-ECE={cal_ece if cal_ece else 'N/A'}")
        print(f"       Pearson(ent,err): r={pearson_r:.4f}, mean_acc={float(accs.mean()):.4f}")

    # Method B metrics
    oracle_profile = {}
    oracle_calibrators = {}
    for mr in ORACLE_MASK_RATIOS:
        bd = oracle_data[mr]
        confs = np.array(bd["confidences"])
        accs = np.array(bd["accuracies"])
        ents = np.array(bd["entropies"])
        errs = np.array(bd["is_errors"])

        n = len(confs)
        if n == 0:
            oracle_profile[f"mask_{mr:.0%}"] = {"n_points": 0}
            continue

        ece, bin_confs, bin_accs, bin_counts = compute_ece(confs, accs, ECE_BINS)
        pearson_r, pearson_p = (stats.pearsonr(ents, errs)
                                if np.std(ents) > 1e-8 and np.std(errs) > 1e-8
                                else (0.0, 1.0))

        if n >= 20:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(confs, accs)
            oracle_calibrators[f"mask_{mr:.0%}"] = iso
            cal_confs = iso.predict(confs)
            cal_ece, cal_bconfs, cal_baccs, cal_bcounts = compute_ece(cal_confs, accs, ECE_BINS)
        else:
            cal_ece = None
            cal_bconfs, cal_baccs, cal_bcounts = [], [], []

        key = f"mask_{int(mr*100)}pct"
        oracle_profile[key] = {
            "mask_ratio": mr,
            "n_points": n,
            "ece": ece,
            "ece_calibrated": cal_ece,
            "pearson_entropy_error": {"r": float(pearson_r), "p_value": float(pearson_p)},
            "mean_confidence": float(confs.mean()),
            "mean_accuracy": float(accs.mean()),
            "mean_entropy": float(ents.mean()),
            "mean_error_rate": float(errs.mean()),
            "reliability_diagram": {
                "bin_confidences": bin_confs, "bin_accuracies": bin_accs, "bin_counts": bin_counts,
            },
            "calibrated_reliability_diagram": {
                "bin_confidences": cal_bconfs, "bin_accuracies": cal_baccs, "bin_counts": cal_bcounts,
            },
        }

        print(f"  [B] mask={mr:.0%}: N={n:,}, ECE={ece:.4f}, Cal-ECE={cal_ece if cal_ece else 'N/A'}")
        print(f"       Pearson(ent,err): r={pearson_r:.4f}, mean_acc={float(accs.mean()):.4f}")

    # ── Overall ECE (self-consistency) ────────────────────────────────
    all_c = np.concatenate([np.array(bd["confidences"]) for bd in sc_band_data.values() if bd["confidences"]])
    all_a = np.concatenate([np.array(bd["accuracies"]) for bd in sc_band_data.values() if bd["accuracies"]])
    overall_sc_ece, _, _, _ = compute_ece(all_c, all_a, ECE_BINS)

    # Overall ECE (oracle)
    all_oc = np.concatenate([np.array(bd["confidences"]) for bd in oracle_data.values() if bd["confidences"]])
    all_oa = np.concatenate([np.array(bd["accuracies"]) for bd in oracle_data.values() if bd["accuracies"]])
    overall_oracle_ece, _, _, _ = compute_ece(all_oc, all_oa, ECE_BINS)

    # Save calibrators
    all_calibrators = {
        "self_consistency": sc_calibrators,
        "oracle": oracle_calibrators,
    }
    calibrator_path = PILOTS_DIR / "calibrators.pkl"
    with open(calibrator_path, "wb") as f:
        pickle.dump(all_calibrators, f)
    print(f"\n  Saved calibrators to {calibrator_path}")

    # ── Hypothesis Testing ────────────────────────────────────────────
    # H1: ECE > 0.10 at early stages
    # Use self-consistency for H1 (more relevant to actual denoising)
    early_sc_ece = sc_profile.get("early_80_100", {}).get("ece", 0)
    # Also check oracle at 90% masking
    early_oracle_ece = oracle_profile.get("mask_90pct", {}).get("ece", 0)

    # H2: Pearson corr(entropy, is_error) > 0.15 at late stages
    # Self-consistency
    late_sc_pearson = sc_profile.get("late_0_20", {}).get("pearson_entropy_error", {}).get("r", 0)
    # Oracle at 10% masking
    late_oracle_pearson = oracle_profile.get("mask_10pct", {}).get("pearson_entropy_error", {}).get("r", 0)

    h1_pass = (early_sc_ece is not None and early_sc_ece > 0.10) or \
              (early_oracle_ece is not None and early_oracle_ece > 0.10)
    h2_pass_sc = (late_sc_pearson is not None and late_sc_pearson > 0.15)
    h2_pass_oracle = (late_oracle_pearson is not None and late_oracle_pearson > 0.15)
    h2_pass = h2_pass_sc or h2_pass_oracle

    if h1_pass and h2_pass:
        go_no_go = "GO"
        overall_rec = "PROCEED"
    elif not h1_pass and early_sc_ece < 0.05 and early_oracle_ece < 0.05:
        go_no_go = "NO_GO"
        overall_rec = "PIVOT"
    else:
        go_no_go = "CONDITIONAL_GO"
        overall_rec = "REFINE"

    summary_text = (
        f"H1 (ECE>0.10 early): {'PASS' if h1_pass else 'FAIL'} "
        f"(SC-ECE={early_sc_ece:.4f}, Oracle-ECE={early_oracle_ece:.4f}). "
        f"H2 (corr>0.15 late): {'PASS' if h2_pass else 'FAIL'} "
        f"(SC-r={late_sc_pearson:.4f}, Oracle-r={late_oracle_pearson:.4f}). "
        f"Overall ECE: SC={overall_sc_ece:.4f}, Oracle={overall_oracle_ece:.4f}."
    )

    elapsed_total = time.time() - start_time

    # Build result
    result = {
        "task_id": TASK_ID,
        "candidate_id": "cand_card",
        "mode": "PILOT",
        "model": "LLaDA-8B-Instruct",
        "benchmark": "GSM8K",
        "num_samples": NUM_SAMPLES,
        "num_steps": NUM_STEPS,
        "gen_length": GEN_LENGTH,
        "seed": SEED,
        "elapsed_sec": round(elapsed_total, 1),
        "method_a_self_consistency": {
            "description": "Compare each step's prediction to final committed token",
            "overall_ece": float(overall_sc_ece),
            "calibration_profile": sc_profile,
        },
        "method_b_oracle": {
            "description": "Teacher-forced: mask fraction of reference, measure recovery",
            "overall_ece": float(overall_oracle_ece),
            "calibration_profile": oracle_profile,
        },
        "hypothesis_tests": {
            "H1_early_miscalibration": {
                "criterion": "ECE > 0.10 at early stages",
                "sc_value": early_sc_ece,
                "oracle_value": early_oracle_ece,
                "pass": h1_pass,
            },
            "H2_entropy_error_correlation": {
                "criterion": "Pearson corr(entropy, is_error) > 0.15 at late stages",
                "sc_value": late_sc_pearson,
                "oracle_value": late_oracle_pearson,
                "pass": h2_pass,
            },
        },
        "go_no_go": go_no_go,
        "recommendation": overall_rec,
        "summary": summary_text,
        "gpu_info": {
            "device": torch.cuda.get_device_name(0),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    result_path = RESULTS_DIR / "calibration_profile.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\n  Saved to {result_path}")

    # Pilot summary
    pilot_summary = {
        "task_id": TASK_ID,
        "candidate_id": "cand_card",
        "go_no_go": go_no_go,
        "recommendation": overall_rec,
        "summary": summary_text,
        "key_metrics": {
            "overall_sc_ece": float(overall_sc_ece),
            "overall_oracle_ece": float(overall_oracle_ece),
            "early_sc_ece": early_sc_ece,
            "early_oracle_ece": early_oracle_ece,
            "late_sc_pearson_r": late_sc_pearson,
            "late_oracle_pearson_r": late_oracle_pearson,
            "h1_pass": h1_pass,
            "h2_pass": h2_pass,
        },
        "elapsed_sec": round(elapsed_total, 1),
    }
    (PILOTS_DIR / "calibration_study_summary.json").write_text(json.dumps(pilot_summary, indent=2))

    # Print summary
    print(f"\n{'='*70}")
    print(f"CALIBRATION STUDY v2 RESULTS")
    print(f"{'='*70}")
    print(f"  Self-Consistency ECE: {overall_sc_ece:.4f}")
    print(f"  Oracle ECE:           {overall_oracle_ece:.4f}")
    print(f"  GO/NO-GO: {go_no_go}")
    print(f"  {summary_text}")
    print(f"  Elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*70}")

    report_progress(
        epoch=len(samples) * 2, total_epochs=len(samples) * 2,
        metric={"overall_sc_ece": float(overall_sc_ece),
                "overall_oracle_ece": float(overall_oracle_ece),
                "go_no_go": go_no_go, "elapsed_sec": round(elapsed_total, 1)}
    )
    mark_done(status="success", summary=summary_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n[FATAL] {error_msg}")
        traceback.print_exc()
        mark_done(status="failed", summary=error_msg)
        sys.exit(1)
