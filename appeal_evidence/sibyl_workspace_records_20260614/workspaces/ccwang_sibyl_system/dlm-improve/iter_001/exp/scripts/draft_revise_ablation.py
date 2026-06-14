#!/usr/bin/env python3
"""
Draft-Revise Factorial Ablation (H3 + H4) — PILOT mode
Task: draft_revise_ablation

2×4 factorial experiment on 100 GSM8K samples with LLaDA-8B-Instruct.
  Draft levels:  {standard ~10%/step, aggressive ~20%/step}
  Revision levels: {none, random-10%-remasking+3 steps, raw-entropy-top-10%+3 steps,
                    calibrated-entropy-top-10%+3 steps}

All conditions use 64 draft steps. Revision adds ~3 extra NFE.
Runs sequentially on single GPU (model loads once, ~25 min total).

Pass criteria:
  - Main effect of revision > 2% accuracy improvement over no-revision conditions
  - Calibrated entropy revision >= raw entropy revision
  - If revision < 1% improvement → H3 falsified

Usage:
    CUDA_VISIBLE_DEVICES=2 python draft_revise_ablation.py
"""

import os
import sys
import gc
import json
import time
import math
import re
import pickle
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset

warnings.filterwarnings("ignore")


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# ── Config ────────────────────────────────────────────────────────────
TASK_ID = "draft_revise_ablation"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
PILOTS_DIR = RESULTS_DIR / "pilots"
CALIBRATOR_PATH = PILOTS_DIR / "calibrators.pkl"
SEED = 42
NUM_SAMPLES = 100
NUM_DRAFT_STEPS = 64
GEN_LENGTH = 256
MASK_TOKEN_ID = 126336
REVISION_STEPS = 3
REVISION_FRACTION = 0.10  # remask top 10% of tokens
DEVICE = "cuda"

# Stage bands for calibrator lookup (matching calibration_study.py)
STAGE_BANDS = [(0.80, 1.00), (0.50, 0.80), (0.20, 0.50), (0.00, 0.20)]
STAGE_BAND_NAMES = ["early_80_100", "mid_50_80", "mid_20_50", "late_0_20"]

# Experimental conditions: (draft_type, revision_type)
CONDITIONS = [
    ("standard", "none"),
    ("standard", "random"),
    ("standard", "raw_entropy"),
    ("standard", "calibrated_entropy"),
    ("aggressive", "none"),
    ("aggressive", "random"),
    ("aggressive", "raw_entropy"),
    ("aggressive", "calibrated_entropy"),
]


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


# ── Schedule functions ────────────────────────────────────────────────
def cosine_schedule(t, T):
    """Fraction unmasked after step t. t=0 -> 0, t=T -> 1."""
    return 1.0 - math.cos(math.pi * t / (2 * T))


def aggressive_schedule(t, T):
    """Aggressive schedule: front-loads unmasking via power < 1 on cosine.
    This unmasks ~20% of total positions per step early on."""
    base = cosine_schedule(t, T)
    return min(1.0, base ** 0.6)


# ── GSM8K answer extraction (same as baseline_standard.py) ────────────
def extract_gsm8k_answer(text):
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', text)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'(?:the\s+)?answer\s+is\s*[:\s]*(-?[\d,]+\.?\d*)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(',', '').strip()
    m = re.search(r'\\boxed\{([^}]+)\}', text)
    if m:
        return m.group(1).strip()
    numbers = re.findall(r'-?[\d,]+\.?\d*', text)
    if numbers:
        return numbers[-1].replace(',', '').strip()
    return ""


def normalize_answer(ans):
    ans = str(ans).strip().replace(',', '')
    if '.' in ans:
        try:
            val = float(ans)
            if val == int(val):
                return str(int(val))
            return ans
        except ValueError:
            pass
    return ans


def check_gsm8k_correct(predicted_text, reference_answer):
    pred_ans = extract_gsm8k_answer(predicted_text)
    pred_norm = normalize_answer(pred_ans)
    ref_norm = normalize_answer(reference_answer)
    return bool(pred_norm == ref_norm and pred_norm != "")


# ── Calibrator helpers ────────────────────────────────────────────────
def load_calibrators():
    """Load pre-trained isotonic regression calibrators from pilot."""
    if not CALIBRATOR_PATH.exists():
        print(f"  WARNING: No calibrator found at {CALIBRATOR_PATH}")
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrators = pickle.load(f)
    return calibrators.get("self_consistency", {})


def get_stage_band_name(masking_ratio):
    """Return stage band name for given masking ratio."""
    for i, (lo, hi) in enumerate(STAGE_BANDS):
        if lo <= masking_ratio <= hi:
            return STAGE_BAND_NAMES[i]
    return STAGE_BAND_NAMES[0] if masking_ratio > 1.0 else STAGE_BAND_NAMES[-1]


# ── Draft phase ───────────────────────────────────────────────────────
def run_draft_phase(model, prompt_ids, gen_length, num_steps, draft_type):
    """Run the draft phase of denoising. Returns draft tokens and per-position info."""
    device = next(model.parameters()).device

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = torch.cat([
        prompt_tensor,
        torch.full((gen_length,), MASK_TOKEN_ID, dtype=torch.long, device=device)
    ]).unsqueeze(0)

    prompt_len = len(prompt_ids)
    gen_start = prompt_len
    gen_end = prompt_len + gen_length

    schedule_fn = cosine_schedule if draft_type == "standard" else aggressive_schedule
    nfe = 0

    # Track which step each position was unmasked at
    unmask_step = torch.full((gen_length,), -1, dtype=torch.long, device=device)
    # Track the confidence when each position was committed (use bfloat16 to match model output)
    commit_confidence = torch.zeros(gen_length, dtype=torch.bfloat16, device=device)

    for step in range(num_steps):
        frac_prev = schedule_fn(step, num_steps) if step > 0 else 0.0
        frac_curr = schedule_fn(step + 1, num_steps)

        gen_region = input_ids[0, gen_start:gen_end]
        is_masked = (gen_region == MASK_TOKEN_ID)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
        nfe += 1

        gen_logits = logits[gen_start:gen_end]
        masked_indices = torch.where(is_masked)[0]
        masked_logits = gen_logits[masked_indices]

        probs = F.softmax(masked_logits, dim=-1)
        top1_conf, top1_token = probs.max(dim=-1)

        num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
        num_to_unmask = min(num_to_unmask, num_masked)

        _, topk_indices = top1_conf.topk(num_to_unmask)
        positions_to_unmask = masked_indices[topk_indices]
        tokens_to_place = top1_token[topk_indices]
        input_ids[0, gen_start + positions_to_unmask] = tokens_to_place
        unmask_step[positions_to_unmask] = step
        commit_confidence[positions_to_unmask] = top1_conf[topk_indices]

    # One extra forward pass for revision targeting: compute entropy of completed draft
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]
    nfe += 1

    gen_logits = logits[gen_start:gen_end]
    probs = F.softmax(gen_logits, dim=-1)
    top1_conf_final, _ = probs.max(dim=-1)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    position_info = {
        "confidence": top1_conf_final.cpu(),
        "commit_confidence": commit_confidence.cpu(),
        "entropy": entropy.cpu(),
        "unmask_step": unmask_step.cpu(),
    }

    return position_info, nfe, input_ids, gen_start, gen_end


def select_revision_targets_random(gen_length, num_to_revise, seed_offset):
    """Randomly select positions to revise."""
    rng = np.random.RandomState(SEED + seed_offset)
    indices = rng.choice(gen_length, size=num_to_revise, replace=False)
    return torch.tensor(indices, dtype=torch.long)


def select_revision_targets_entropy(position_info, num_to_revise):
    """Select highest-entropy positions for revision (raw entropy)."""
    entropy = position_info["entropy"]
    _, topk = entropy.topk(num_to_revise)
    return topk


def select_revision_targets_calibrated(position_info, num_to_revise, calibrators, num_steps):
    """Select positions using calibration-corrected fragility score.

    Fragility = entropy * (1 + alpha * overconfidence)
    where overconfidence = max(0, raw_commit_conf - calibrated_conf)

    Tokens committed at poorly-calibrated stages get their entropy amplified.
    """
    if calibrators is None:
        return select_revision_targets_entropy(position_info, num_to_revise)

    entropy = position_info["entropy"]
    unmask_step = position_info["unmask_step"]
    commit_conf = position_info["commit_confidence"]

    fragility = torch.zeros_like(entropy)

    for pos in range(len(entropy)):
        step = unmask_step[pos].item()
        if step < 0:
            fragility[pos] = entropy[pos]
            continue

        # Estimate masking ratio when this token was committed
        masking_ratio = 1.0 - cosine_schedule(step + 1, num_steps)
        band_name = get_stage_band_name(masking_ratio)

        calibrator = calibrators.get(band_name)
        if calibrator is None:
            fragility[pos] = entropy[pos]
            continue

        raw_conf = commit_conf[pos].item()
        try:
            cal_conf = float(calibrator.predict(np.array([raw_conf]))[0])
        except Exception:
            cal_conf = raw_conf

        # Overconfidence: model thought it was right but calibration says less sure
        overconfidence = max(0.0, raw_conf - cal_conf)
        # Amplify entropy by overconfidence factor
        fragility[pos] = entropy[pos] * (1.0 + 5.0 * overconfidence)

    _, topk = fragility.topk(num_to_revise)
    return topk


def run_revision_phase(model, input_ids, gen_start, gen_end, revision_targets,
                       num_revision_steps):
    """Re-mask selected positions and re-denoise them."""
    device = input_ids.device
    gen_length = gen_end - gen_start
    nfe = 0

    revision_targets_device = revision_targets.to(device)
    num_to_revise = len(revision_targets)

    # Save original tokens at revision positions (for comparison later)
    original_tokens = input_ids[0, gen_start + revision_targets_device].clone()

    # Re-mask selected positions
    input_ids[0, gen_start + revision_targets_device] = MASK_TOKEN_ID

    for rev_step in range(num_revision_steps):
        gen_region = input_ids[0, gen_start:gen_end]
        is_masked = (gen_region == MASK_TOKEN_ID)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
        nfe += 1

        gen_logits = logits[gen_start:gen_end]
        masked_indices = torch.where(is_masked)[0]
        masked_logits = gen_logits[masked_indices]

        probs = F.softmax(masked_logits, dim=-1)
        top1_conf, top1_token = probs.max(dim=-1)

        # Unmask proportionally across revision steps
        target_unmasked = int(round((rev_step + 1) / num_revision_steps * num_to_revise))
        already_unmasked = num_to_revise - num_masked
        num_to_unmask = max(1, target_unmasked - already_unmasked)
        num_to_unmask = min(num_to_unmask, num_masked)

        if num_to_unmask > 0 and len(top1_conf) > 0:
            k = min(num_to_unmask, len(top1_conf))
            _, topk_indices = top1_conf.topk(k)
            positions_to_unmask = masked_indices[topk_indices]
            tokens_to_place = top1_token[topk_indices]
            input_ids[0, gen_start + positions_to_unmask] = tokens_to_place

    # Force-unmask any remaining
    gen_region = input_ids[0, gen_start:gen_end]
    remaining_masked = (gen_region == MASK_TOKEN_ID)
    if remaining_masked.any():
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
        nfe += 1
        masked_idx = torch.where(remaining_masked)[0]
        probs = F.softmax(logits[gen_start + masked_idx], dim=-1)
        _, tokens = probs.max(dim=-1)
        input_ids[0, gen_start + masked_idx] = tokens

    # Count how many tokens actually changed
    new_tokens = input_ids[0, gen_start + revision_targets_device]
    tokens_changed = (new_tokens != original_tokens).sum().item()

    return input_ids, nfe, tokens_changed


def run_single_condition(model, tokenizer, samples, draft_type, revision_type,
                         calibrators, condition_name):
    """Run a single experimental condition on all samples."""
    results = []
    correct = 0
    total = 0
    total_nfe = 0
    total_tokens_changed = 0
    condition_start = time.time()

    for si, sample in enumerate(samples):
        torch.manual_seed(SEED + sample["idx"])
        torch.cuda.manual_seed_all(SEED + sample["idx"])

        sample_start = time.time()
        gen_len = min(GEN_LENGTH, 256)

        # Phase 1: Draft
        position_info, draft_nfe, input_ids, gen_start, gen_end = run_draft_phase(
            model, sample["prompt_ids"], gen_len, NUM_DRAFT_STEPS, draft_type
        )

        # Phase 2: Revision
        revision_nfe = 0
        tokens_changed = 0
        num_to_revise = max(1, int(round(REVISION_FRACTION * gen_len)))

        if revision_type == "none":
            pass
        elif revision_type == "random":
            targets = select_revision_targets_random(gen_len, num_to_revise, sample["idx"])
            input_ids, revision_nfe, tokens_changed = run_revision_phase(
                model, input_ids, gen_start, gen_end, targets, REVISION_STEPS
            )
        elif revision_type == "raw_entropy":
            targets = select_revision_targets_entropy(position_info, num_to_revise)
            input_ids, revision_nfe, tokens_changed = run_revision_phase(
                model, input_ids, gen_start, gen_end, targets, REVISION_STEPS
            )
        elif revision_type == "calibrated_entropy":
            targets = select_revision_targets_calibrated(
                position_info, num_to_revise, calibrators, NUM_DRAFT_STEPS
            )
            input_ids, revision_nfe, tokens_changed = run_revision_phase(
                model, input_ids, gen_start, gen_end, targets, REVISION_STEPS
            )

        sample_nfe = draft_nfe + revision_nfe
        total_nfe += sample_nfe
        total_tokens_changed += tokens_changed

        # Decode
        gen_tokens = input_ids[0, gen_start:gen_end].cpu().tolist()
        gen_tokens_clean = [t for t in gen_tokens if t != MASK_TOKEN_ID]
        generated_text = tokenizer.decode(gen_tokens_clean, skip_special_tokens=True)

        is_correct = check_gsm8k_correct(generated_text, sample["final_answer"])
        if is_correct:
            correct += 1
        total += 1

        sample_time = time.time() - sample_start

        result_entry = {
            "idx": sample["idx"],
            "correct": is_correct,
            "nfe": sample_nfe,
            "draft_nfe": draft_nfe,
            "revision_nfe": revision_nfe,
            "tokens_changed": tokens_changed,
            "time_sec": round(sample_time, 3),
            "predicted_answer": extract_gsm8k_answer(generated_text),
            "reference_answer": sample["final_answer"],
        }

        if si < 5:
            result_entry["generated_text"] = generated_text[:300]

        results.append(result_entry)

        if si % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            acc_so_far = correct / total if total > 0 else 0
            elapsed = time.time() - condition_start
            print(f"    [{condition_name}] {si}/{len(samples)}, "
                  f"acc={acc_so_far:.3f}, elapsed={elapsed:.0f}s",
                  flush=True)

    condition_time = time.time() - condition_start
    accuracy = correct / total if total > 0 else 0
    avg_nfe = total_nfe / total if total > 0 else 0
    avg_tokens_changed = total_tokens_changed / total if total > 0 else 0

    condition_result = {
        "condition": condition_name,
        "draft_type": draft_type,
        "revision_type": revision_type,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_nfe": round(avg_nfe, 1),
        "avg_tokens_changed": round(avg_tokens_changed, 1),
        "wall_clock_sec": round(condition_time, 2),
        "avg_time_per_sample_sec": round(condition_time / total, 3),
        "per_sample": results,
    }

    return condition_result


def get_gsm8k_samples(tokenizer, num_samples=100):
    dataset = load_dataset("gsm8k", "main", split="test")
    samples = []
    for idx in range(min(num_samples, len(dataset))):
        question = dataset[idx]["question"]
        answer = dataset[idx]["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        samples.append({
            "idx": idx,
            "prompt_ids": prompt_ids,
            "final_answer": final_answer,
            "question": question[:80],
        })
    return samples


def main():
    global_start = time.time()
    write_pid()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PILOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{TASK_ID}] Starting draft-revise factorial ablation", flush=True)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Samples: {NUM_SAMPLES}, Draft steps: {NUM_DRAFT_STEPS}")
    print(f"  Revision: {REVISION_STEPS} steps, {REVISION_FRACTION*100:.0f}% remasking")
    print(f"  Conditions: {len(CONDITIONS)}")
    print(f"  Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
          flush=True)

    # Load calibrators
    print("\n[1/4] Loading calibrators...", flush=True)
    calibrators = load_calibrators()
    if calibrators:
        print(f"  Loaded calibrators for bands: {list(calibrators.keys())}")
    else:
        print("  WARNING: No calibrators. Calibrated conditions fall back to raw entropy.")

    # Load model & tokenizer
    print("\n[2/4] Loading model...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE).eval()
    vram_after = torch.cuda.memory_allocated() / 1e6
    print(f"  Model loaded. VRAM: {vram_after:.0f} MB", flush=True)

    # Load GSM8K
    print("\n[3/4] Loading GSM8K samples...", flush=True)
    samples = get_gsm8k_samples(tokenizer, NUM_SAMPLES)
    print(f"  Loaded {len(samples)} samples")

    # Run all conditions sequentially
    print(f"\n[4/4] Running {len(CONDITIONS)} conditions...", flush=True)

    all_condition_results = {}
    for ci, (draft_type, revision_type) in enumerate(CONDITIONS):
        cond_name = f"{draft_type}_{revision_type}"
        print(f"\n  ── Condition {ci+1}/{len(CONDITIONS)}: {cond_name} ──", flush=True)

        result = run_single_condition(
            model, tokenizer, samples, draft_type, revision_type,
            calibrators, cond_name
        )
        all_condition_results[cond_name] = result

        acc = result["accuracy"]
        nfe = result["avg_nfe"]
        t = result["wall_clock_sec"]
        tc = result["avg_tokens_changed"]
        print(f"  → {cond_name}: acc={acc:.4f}, NFE={nfe:.1f}, "
              f"tokens_changed={tc:.1f}, time={t:.1f}s", flush=True)

        report_progress(
            epoch=ci+1, total_epochs=len(CONDITIONS),
            metric={
                "condition": cond_name,
                "accuracy": acc,
                "avg_nfe": nfe,
                "elapsed_sec": round(time.time() - global_start, 1),
            }
        )

        torch.cuda.empty_cache()
        gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}", flush=True)
    print("ANALYSIS", flush=True)
    print(f"{'='*70}", flush=True)

    # Summary table
    summary_table = []
    for cond_name in [f"{d}_{r}" for d, r in CONDITIONS]:
        if cond_name not in all_condition_results:
            continue
        r = all_condition_results[cond_name]
        summary_table.append({
            "condition": cond_name,
            "draft_type": r["draft_type"],
            "revision_type": r["revision_type"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "avg_nfe": r["avg_nfe"],
            "avg_tokens_changed": r["avg_tokens_changed"],
            "wall_clock_sec": r["wall_clock_sec"],
            "avg_time_per_sample_sec": r["avg_time_per_sample_sec"],
        })

    # Main effects: revision type (averaged over draft types)
    revision_effects = {}
    for rev_type in ["none", "random", "raw_entropy", "calibrated_entropy"]:
        accs = []
        for draft_type in ["standard", "aggressive"]:
            cond = f"{draft_type}_{rev_type}"
            if cond in all_condition_results:
                accs.append(all_condition_results[cond]["accuracy"])
        revision_effects[rev_type] = np.mean(accs) if accs else 0.0

    # Main effects: draft type (averaged over revision types)
    draft_effects = {}
    for draft_type in ["standard", "aggressive"]:
        accs = []
        for rev_type in ["none", "random", "raw_entropy", "calibrated_entropy"]:
            cond = f"{draft_type}_{rev_type}"
            if cond in all_condition_results:
                accs.append(all_condition_results[cond]["accuracy"])
        draft_effects[draft_type] = np.mean(accs) if accs else 0.0

    # H3: Does revision help?
    no_revision_acc = revision_effects.get("none", 0)
    any_revision_acc = np.mean([
        revision_effects.get(r, 0) for r in ["random", "raw_entropy", "calibrated_entropy"]
    ])
    revision_improvement = any_revision_acc - no_revision_acc

    # H4: Does calibrated > raw entropy?
    raw_ent_acc = revision_effects.get("raw_entropy", 0)
    cal_ent_acc = revision_effects.get("calibrated_entropy", 0)
    calibration_improvement = cal_ent_acc - raw_ent_acc

    # Best vs worst revision (for effect size reporting)
    best_revision_type = max(["random", "raw_entropy", "calibrated_entropy"],
                             key=lambda r: revision_effects.get(r, 0))
    best_revision_acc = revision_effects[best_revision_type]

    # Per draft-type: does revision help within each draft strategy?
    per_draft_revision_effect = {}
    for draft_type in ["standard", "aggressive"]:
        none_cond = f"{draft_type}_none"
        none_acc = all_condition_results.get(none_cond, {}).get("accuracy", 0)
        rev_accs = []
        for rev_type in ["random", "raw_entropy", "calibrated_entropy"]:
            cond = f"{draft_type}_{rev_type}"
            if cond in all_condition_results:
                rev_accs.append(all_condition_results[cond]["accuracy"])
        avg_rev = np.mean(rev_accs) if rev_accs else 0
        per_draft_revision_effect[draft_type] = {
            "no_revision": round(none_acc, 4),
            "avg_revision": round(avg_rev, 4),
            "improvement": round(avg_rev - none_acc, 4),
        }

    # Pass criteria
    h3_pass = revision_improvement > 0.02
    h3_marginal = revision_improvement > 0.01
    h4_pass = cal_ent_acc >= raw_ent_acc

    if h3_pass and h4_pass:
        go_no_go = "GO"
        recommendation = "PROCEED"
    elif h3_marginal:
        go_no_go = "CONDITIONAL_GO"
        recommendation = "REFINE"
    elif revision_improvement < 0.01:
        go_no_go = "NO_GO"
        recommendation = "H3_FALSIFIED"
    else:
        go_no_go = "CONDITIONAL_GO"
        recommendation = "INVESTIGATE"

    summary_text = (
        f"Draft-Revise Ablation: "
        f"Revision effect={revision_improvement:+.4f} "
        f"(none={no_revision_acc:.4f}, avg_rev={any_revision_acc:.4f}). "
        f"Best revision={best_revision_type}({best_revision_acc:.4f}). "
        f"Calibration effect={calibration_improvement:+.4f} "
        f"(raw={raw_ent_acc:.4f}, cal={cal_ent_acc:.4f}). "
        f"Draft: std={draft_effects.get('standard', 0):.4f}, "
        f"agg={draft_effects.get('aggressive', 0):.4f}. "
        f"H3(rev>2%): {'PASS' if h3_pass else 'FAIL'}. "
        f"H4(cal>=raw): {'PASS' if h4_pass else 'FAIL'}. "
        f"GO/NO-GO={go_no_go}."
    )

    total_elapsed = time.time() - global_start

    # Qualitative samples
    qualitative = {}
    for cond_name, cond_result in all_condition_results.items():
        for entry in cond_result["per_sample"][:5]:
            if "generated_text" in entry:
                key = f"{cond_name}_sample_{entry['idx']}"
                qualitative[key] = {
                    "condition": cond_name,
                    "correct": entry["correct"],
                    "predicted": entry["predicted_answer"],
                    "reference": entry["reference_answer"],
                    "text": entry.get("generated_text", "")[:200],
                }

    # Build result
    result = {
        "task_id": TASK_ID,
        "candidate_id": "cand_card",
        "mode": "PILOT",
        "model": "LLaDA-8B-Instruct",
        "benchmark": "GSM8K",
        "num_samples": NUM_SAMPLES,
        "num_draft_steps": NUM_DRAFT_STEPS,
        "revision_steps": REVISION_STEPS,
        "revision_fraction": REVISION_FRACTION,
        "gen_length": GEN_LENGTH,
        "seed": SEED,
        "elapsed_sec": round(total_elapsed, 1),
        "summary_table": summary_table,
        "main_effects": {
            "revision_type": {k: round(v, 4) for k, v in revision_effects.items()},
            "draft_type": {k: round(v, 4) for k, v in draft_effects.items()},
            "per_draft_revision_effect": per_draft_revision_effect,
            "revision_improvement": round(revision_improvement, 4),
            "calibration_improvement": round(calibration_improvement, 4),
            "best_revision_type": best_revision_type,
        },
        "hypothesis_tests": {
            "H3_revision_helps": {
                "criterion": "revision accuracy > no_revision accuracy by > 2%",
                "no_revision_acc": round(no_revision_acc, 4),
                "avg_revision_acc": round(any_revision_acc, 4),
                "improvement": round(revision_improvement, 4),
                "pass": h3_pass,
                "marginal": h3_marginal,
            },
            "H4_calibration_helps": {
                "criterion": "calibrated_entropy >= raw_entropy",
                "raw_entropy_acc": round(raw_ent_acc, 4),
                "calibrated_entropy_acc": round(cal_ent_acc, 4),
                "improvement": round(calibration_improvement, 4),
                "pass": h4_pass,
            },
        },
        "condition_details": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
            for k, v in all_condition_results.items()
        },
        "per_sample_results": {
            k: v["per_sample"] for k, v in all_condition_results.items()
        },
        "qualitative_samples": qualitative,
        "go_no_go": go_no_go,
        "recommendation": recommendation,
        "summary": summary_text,
        "gpu_info": {
            "device": torch.cuda.get_device_name(0),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save
    result_path = RESULTS_DIR / "draft_revise_ablation.json"
    result_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n  Saved to {result_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"DRAFT-REVISE ABLATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<30} {'Acc':>8} {'NFE':>6} {'TokChg':>8} {'Time(s)':>8}")
    print(f"  {'-'*60}")
    for row in summary_table:
        print(f"  {row['condition']:<30} {row['accuracy']:>8.4f} "
              f"{row['avg_nfe']:>6.1f} {row['avg_tokens_changed']:>8.1f} "
              f"{row['wall_clock_sec']:>8.1f}")

    print(f"\n  Main Effects:")
    print(f"    Revision type:")
    for rev_type in ["none", "random", "raw_entropy", "calibrated_entropy"]:
        print(f"      {rev_type:<25} {revision_effects[rev_type]:.4f}")
    print(f"    Draft type:")
    for draft_type in ["standard", "aggressive"]:
        print(f"      {draft_type:<25} {draft_effects[draft_type]:.4f}")
    print(f"    Per-draft revision effect:")
    for dt, eff in per_draft_revision_effect.items():
        print(f"      {dt}: none={eff['no_revision']:.4f} → "
              f"avg_rev={eff['avg_revision']:.4f} ({eff['improvement']:+.4f})")

    print(f"\n  Hypotheses:")
    print(f"    H3 (revision > 2%): {'PASS' if h3_pass else 'FAIL'} "
          f"({revision_improvement:+.4f})")
    print(f"    H4 (cal >= raw): {'PASS' if h4_pass else 'FAIL'} "
          f"({calibration_improvement:+.4f})")
    print(f"  GO/NO-GO: {go_no_go}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}", flush=True)

    report_progress(
        epoch=len(CONDITIONS), total_epochs=len(CONDITIONS),
        metric={
            "revision_improvement": round(revision_improvement, 4),
            "calibration_improvement": round(calibration_improvement, 4),
            "go_no_go": go_no_go,
            "elapsed_sec": round(total_elapsed, 1),
        }
    )
    mark_done(status="success", summary=summary_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n[FATAL] {error_msg}", flush=True)
        traceback.print_exc()
        mark_done(status="failed", summary=error_msg)
        sys.exit(1)
