#!/usr/bin/env python3
"""
Temperature Annealing Ablation (H6) — PILOT mode
Task: annealing_ablation

Compare CARD with temperature annealing vs isothermal (T=1.0) on 100 GSM8K
samples. All conditions use 64 draft steps + calibrated entropy revision
(10% remasking, 3 revision steps).

Conditions:
  1. isothermal:  T=1.0 throughout draft
  2. linear:      T=T_max → T_min linearly
  3. quadratic:   T=T_max*(1-t/T)^2 + T_min
  4. cosine:      T=T_min + 0.5*(T_max-T_min)*(1+cos(pi*t/T))

T_max=2.0, T_min=0.1

Pass criteria:
  - Best annealing schedule > isothermal by >= 1% accuracy
  - If annealing degrades performance → H6 falsified, use isothermal

Usage:
    CUDA_VISIBLE_DEVICES=2 python annealing_ablation.py
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
TASK_ID = "annealing_ablation"
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
REVISION_FRACTION = 0.10
DEVICE = "cuda"

T_MAX = 2.0
T_MIN = 0.1

STAGE_BANDS = [(0.80, 1.00), (0.50, 0.80), (0.20, 0.50), (0.00, 0.20)]
STAGE_BAND_NAMES = ["early_80_100", "mid_50_80", "mid_20_50", "late_0_20"]

# Annealing conditions: (name, schedule_type)
CONDITIONS = [
    ("isothermal", "isothermal"),
    ("linear", "linear"),
    ("quadratic", "quadratic"),
    ("cosine_anneal", "cosine_anneal"),
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


# ── Temperature schedules ─────────────────────────────────────────────
def temperature_isothermal(step, total_steps):
    """Constant T=1.0 throughout."""
    return 1.0


def temperature_linear(step, total_steps):
    """Linear cooling: T_max → T_min."""
    t = step / max(total_steps - 1, 1)
    return T_MAX * (1.0 - t) + T_MIN * t


def temperature_quadratic(step, total_steps):
    """Quadratic cooling: T_max*(1-t/T)^2 + T_min."""
    t = step / max(total_steps - 1, 1)
    return T_MAX * (1.0 - t) ** 2 + T_MIN


def temperature_cosine_anneal(step, total_steps):
    """Cosine annealing: T_min + 0.5*(T_max-T_min)*(1+cos(pi*t/T))."""
    t = step / max(total_steps - 1, 1)
    return T_MIN + 0.5 * (T_MAX - T_MIN) * (1.0 + math.cos(math.pi * t))


TEMPERATURE_FNS = {
    "isothermal": temperature_isothermal,
    "linear": temperature_linear,
    "quadratic": temperature_quadratic,
    "cosine_anneal": temperature_cosine_anneal,
}


# ── Unmasking schedule ────────────────────────────────────────────────
def cosine_schedule(t, T):
    """Fraction unmasked after step t."""
    return 1.0 - math.cos(math.pi * t / (2 * T))


# ── GSM8K answer extraction ──────────────────────────────────────────
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
    if not CALIBRATOR_PATH.exists():
        print(f"  WARNING: No calibrator found at {CALIBRATOR_PATH}")
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        calibrators = pickle.load(f)
    return calibrators.get("self_consistency", {})


def get_stage_band_name(masking_ratio):
    for i, (lo, hi) in enumerate(STAGE_BANDS):
        if lo <= masking_ratio <= hi:
            return STAGE_BAND_NAMES[i]
    return STAGE_BAND_NAMES[0] if masking_ratio > 1.0 else STAGE_BAND_NAMES[-1]


# ── Draft phase with temperature annealing ────────────────────────────
def run_draft_phase_annealed(model, prompt_ids, gen_length, num_steps, temp_fn):
    """Run draft phase with temperature annealing.

    At each step:
    1. Forward pass to get logits
    2. Apply temperature T(step) to logits before softmax
    3. Use confidence (after temperature) to decide which tokens to unmask
    4. Unmask top-confidence tokens according to cosine schedule
    """
    device = next(model.parameters()).device

    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = torch.cat([
        prompt_tensor,
        torch.full((gen_length,), MASK_TOKEN_ID, dtype=torch.long, device=device)
    ]).unsqueeze(0)

    prompt_len = len(prompt_ids)
    gen_start = prompt_len
    gen_end = prompt_len + gen_length

    nfe = 0
    unmask_step = torch.full((gen_length,), -1, dtype=torch.long, device=device)
    commit_confidence = torch.zeros(gen_length, dtype=torch.bfloat16, device=device)
    temperature_at_commit = torch.zeros(gen_length, dtype=torch.float32, device=device)
    temperature_trace = []

    for step in range(num_steps):
        frac_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
        frac_curr = cosine_schedule(step + 1, num_steps)

        gen_region = input_ids[0, gen_start:gen_end]
        is_masked = (gen_region == MASK_TOKEN_ID)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        # Get temperature for this step
        temp = temp_fn(step, num_steps)
        temperature_trace.append({"step": step, "temperature": round(temp, 4),
                                  "num_masked": num_masked})

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]
        nfe += 1

        gen_logits = logits[gen_start:gen_end]
        masked_indices = torch.where(is_masked)[0]
        masked_logits = gen_logits[masked_indices]

        # Apply temperature scaling
        scaled_logits = masked_logits / temp
        probs = F.softmax(scaled_logits, dim=-1)
        top1_conf, top1_token = probs.max(dim=-1)

        num_to_unmask = max(1, int(round((frac_curr - frac_prev) * gen_length)))
        num_to_unmask = min(num_to_unmask, num_masked)

        _, topk_indices = top1_conf.topk(num_to_unmask)
        positions_to_unmask = masked_indices[topk_indices]
        tokens_to_place = top1_token[topk_indices]
        input_ids[0, gen_start + positions_to_unmask] = tokens_to_place
        unmask_step[positions_to_unmask] = step
        commit_confidence[positions_to_unmask] = top1_conf[topk_indices]
        temperature_at_commit[positions_to_unmask] = temp

    # Extra forward pass for revision targeting
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
        "temperature_at_commit": temperature_at_commit.cpu(),
    }

    return position_info, nfe, input_ids, gen_start, gen_end, temperature_trace


# ── Revision (calibrated entropy) ────────────────────────────────────
def select_revision_targets_calibrated(position_info, num_to_revise, calibrators, num_steps):
    """Select positions using calibration-corrected fragility score."""
    if calibrators is None:
        entropy = position_info["entropy"]
        _, topk = entropy.topk(num_to_revise)
        return topk

    entropy = position_info["entropy"]
    unmask_step = position_info["unmask_step"]
    commit_conf = position_info["commit_confidence"]

    fragility = torch.zeros_like(entropy)

    for pos in range(len(entropy)):
        step = unmask_step[pos].item()
        if step < 0:
            fragility[pos] = entropy[pos]
            continue

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

        overconfidence = max(0.0, raw_conf - cal_conf)
        fragility[pos] = entropy[pos] * (1.0 + 5.0 * overconfidence)

    _, topk = fragility.topk(num_to_revise)
    return topk


def run_revision_phase(model, input_ids, gen_start, gen_end, revision_targets,
                       num_revision_steps):
    """Re-mask selected positions and re-denoise them."""
    device = input_ids.device
    nfe = 0

    revision_targets_device = revision_targets.to(device)
    num_to_revise = len(revision_targets)

    original_tokens = input_ids[0, gen_start + revision_targets_device].clone()
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

    new_tokens = input_ids[0, gen_start + revision_targets_device]
    tokens_changed = (new_tokens != original_tokens).sum().item()

    return input_ids, nfe, tokens_changed


# ── Run single condition ──────────────────────────────────────────────
def run_single_condition(model, tokenizer, samples, schedule_type, calibrators,
                         condition_name):
    """Run one annealing schedule condition on all samples."""
    temp_fn = TEMPERATURE_FNS[schedule_type]
    results = []
    correct = 0
    total = 0
    total_nfe = 0
    total_tokens_changed = 0
    condition_start = time.time()
    all_temp_traces = []

    for si, sample in enumerate(samples):
        torch.manual_seed(SEED + sample["idx"])
        torch.cuda.manual_seed_all(SEED + sample["idx"])

        sample_start = time.time()
        gen_len = min(GEN_LENGTH, 256)

        # Phase 1: Draft with temperature annealing
        position_info, draft_nfe, input_ids, gen_start, gen_end, temp_trace = \
            run_draft_phase_annealed(
                model, sample["prompt_ids"], gen_len, NUM_DRAFT_STEPS, temp_fn
            )
        if si == 0:
            all_temp_traces.append(temp_trace)

        # Phase 2: Calibrated entropy revision (T=1.0 for revision, always)
        num_to_revise = max(1, int(round(REVISION_FRACTION * gen_len)))
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
        "schedule_type": schedule_type,
        "t_max": T_MAX if schedule_type != "isothermal" else 1.0,
        "t_min": T_MIN if schedule_type != "isothermal" else 1.0,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_nfe": round(avg_nfe, 1),
        "avg_tokens_changed": round(avg_tokens_changed, 1),
        "wall_clock_sec": round(condition_time, 2),
        "avg_time_per_sample_sec": round(condition_time / total, 3),
        "temperature_trace": all_temp_traces[0] if all_temp_traces else [],
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
        prompt = (f"Solve the following math problem step by step.\n\n"
                  f"Question: {question}\n\nAnswer:")
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

    print(f"[{TASK_ID}] Starting temperature annealing ablation (H6)", flush=True)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Samples: {NUM_SAMPLES}, Draft steps: {NUM_DRAFT_STEPS}")
    print(f"  Revision: {REVISION_STEPS} steps, {REVISION_FRACTION*100:.0f}% remasking")
    print(f"  T_max={T_MAX}, T_min={T_MIN}")
    print(f"  Conditions: {[c[0] for c in CONDITIONS]}")
    print(f"  Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
          flush=True)

    # Load calibrators
    print("\n[1/4] Loading calibrators...", flush=True)
    calibrators = load_calibrators()
    if calibrators:
        print(f"  Loaded calibrators for bands: {list(calibrators.keys())}")
    else:
        print("  WARNING: No calibrators. Falling back to raw entropy.")

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

    # Run all conditions
    print(f"\n[4/4] Running {len(CONDITIONS)} conditions...", flush=True)

    all_condition_results = {}
    for ci, (cond_name, schedule_type) in enumerate(CONDITIONS):
        print(f"\n  ── Condition {ci+1}/{len(CONDITIONS)}: {cond_name} "
              f"(schedule={schedule_type}) ──", flush=True)

        result = run_single_condition(
            model, tokenizer, samples, schedule_type, calibrators, cond_name
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
    for cond_name, _ in CONDITIONS:
        r = all_condition_results[cond_name]
        summary_table.append({
            "condition": cond_name,
            "schedule_type": r["schedule_type"],
            "accuracy": r["accuracy"],
            "correct": r["correct"],
            "total": r["total"],
            "avg_nfe": r["avg_nfe"],
            "avg_tokens_changed": r["avg_tokens_changed"],
            "wall_clock_sec": r["wall_clock_sec"],
            "avg_time_per_sample_sec": r["avg_time_per_sample_sec"],
        })

    # H6: annealing vs isothermal
    isothermal_acc = all_condition_results["isothermal"]["accuracy"]
    annealing_accs = {
        name: all_condition_results[name]["accuracy"]
        for name, stype in CONDITIONS if stype != "isothermal"
    }
    best_annealing_name = max(annealing_accs, key=annealing_accs.get)
    best_annealing_acc = annealing_accs[best_annealing_name]
    annealing_improvement = best_annealing_acc - isothermal_acc
    avg_annealing_acc = np.mean(list(annealing_accs.values()))

    # Per-sample agreement analysis: which samples flip between isothermal
    # and best annealing?
    iso_per_sample = {
        r["idx"]: r["correct"]
        for r in all_condition_results["isothermal"]["per_sample"]
    }
    best_per_sample = {
        r["idx"]: r["correct"]
        for r in all_condition_results[best_annealing_name]["per_sample"]
    }
    flipped_to_correct = []
    flipped_to_wrong = []
    for idx in iso_per_sample:
        if not iso_per_sample[idx] and best_per_sample.get(idx, False):
            flipped_to_correct.append(idx)
        elif iso_per_sample[idx] and not best_per_sample.get(idx, False):
            flipped_to_wrong.append(idx)

    # Pass criteria
    h6_pass = annealing_improvement >= 0.01
    h6_degrades = best_annealing_acc < isothermal_acc

    if h6_pass:
        go_no_go = "GO"
        recommendation = "PROCEED"
    elif h6_degrades:
        go_no_go = "NO_GO"
        recommendation = "H6_FALSIFIED_USE_ISOTHERMAL"
    else:
        go_no_go = "CONDITIONAL_GO"
        recommendation = "ANNEALING_MARGINAL_USE_ISOTHERMAL_AS_DEFAULT"

    summary_text = (
        f"Annealing Ablation (H6): "
        f"isothermal={isothermal_acc:.4f}, "
        f"best_annealing={best_annealing_name}({best_annealing_acc:.4f}), "
        f"improvement={annealing_improvement:+.4f}. "
        f"All schedules: {', '.join(f'{k}={v:.4f}' for k, v in annealing_accs.items())}. "
        f"avg_annealing={avg_annealing_acc:.4f}. "
        f"Flips: +{len(flipped_to_correct)}/-{len(flipped_to_wrong)} samples. "
        f"H6(anneal>iso by>=1%): {'PASS' if h6_pass else 'FAIL'}. "
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

    # Temperature schedule visualization data
    schedule_curves = {}
    for cond_name, schedule_type in CONDITIONS:
        temp_fn = TEMPERATURE_FNS[schedule_type]
        curve = [{"step": s, "temperature": round(temp_fn(s, NUM_DRAFT_STEPS), 4)}
                 for s in range(NUM_DRAFT_STEPS)]
        schedule_curves[cond_name] = curve

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
        "revision_type": "calibrated_entropy",
        "gen_length": GEN_LENGTH,
        "seed": SEED,
        "t_max": T_MAX,
        "t_min": T_MIN,
        "elapsed_sec": round(total_elapsed, 1),
        "summary_table": summary_table,
        "annealing_comparison": {
            "isothermal_accuracy": round(isothermal_acc, 4),
            "annealing_accuracies": {k: round(v, 4) for k, v in annealing_accs.items()},
            "best_annealing_schedule": best_annealing_name,
            "best_annealing_accuracy": round(best_annealing_acc, 4),
            "improvement_over_isothermal": round(annealing_improvement, 4),
            "avg_annealing_accuracy": round(avg_annealing_acc, 4),
            "flipped_to_correct": flipped_to_correct,
            "flipped_to_wrong": flipped_to_wrong,
            "net_flips": len(flipped_to_correct) - len(flipped_to_wrong),
        },
        "hypothesis_tests": {
            "H6_annealing_helps": {
                "criterion": "Best annealing schedule > isothermal by >= 1% accuracy",
                "isothermal_acc": round(isothermal_acc, 4),
                "best_annealing_acc": round(best_annealing_acc, 4),
                "best_schedule": best_annealing_name,
                "improvement": round(annealing_improvement, 4),
                "pass": h6_pass,
                "degrades": h6_degrades,
            },
        },
        "condition_details": {
            k: {kk: vv for kk, vv in v.items() if kk != "per_sample"}
            for k, v in all_condition_results.items()
        },
        "per_sample_results": {
            k: v["per_sample"] for k, v in all_condition_results.items()
        },
        "schedule_curves": schedule_curves,
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
    result_path = RESULTS_DIR / "annealing_ablation.json"
    result_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n  Saved to {result_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"ANNEALING ABLATION RESULTS (H6)")
    print(f"{'='*70}")
    print(f"\n  {'Condition':<20} {'Acc':>8} {'NFE':>6} {'TokChg':>8} {'Time(s)':>8}")
    print(f"  {'-'*50}")
    for row in summary_table:
        marker = " *" if row["condition"] == best_annealing_name else ""
        print(f"  {row['condition']:<20} {row['accuracy']:>8.4f} "
              f"{row['avg_nfe']:>6.1f} {row['avg_tokens_changed']:>8.1f} "
              f"{row['wall_clock_sec']:>8.1f}{marker}")

    print(f"\n  Comparison:")
    print(f"    Isothermal (T=1.0):       {isothermal_acc:.4f}")
    print(f"    Best annealing ({best_annealing_name}): {best_annealing_acc:.4f}")
    print(f"    Improvement:              {annealing_improvement:+.4f}")
    print(f"    Avg annealing:            {avg_annealing_acc:.4f}")
    print(f"    Flips: +{len(flipped_to_correct)} correct, "
          f"-{len(flipped_to_wrong)} wrong (net={len(flipped_to_correct)-len(flipped_to_wrong)})")

    print(f"\n  H6 (anneal > iso by >= 1%): {'PASS' if h6_pass else 'FAIL'}")
    print(f"  GO/NO-GO: {go_no_go}")
    print(f"  Recommendation: {recommendation}")
    print(f"  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}", flush=True)

    report_progress(
        epoch=len(CONDITIONS), total_epochs=len(CONDITIONS),
        metric={
            "isothermal_acc": round(isothermal_acc, 4),
            "best_annealing": best_annealing_name,
            "best_annealing_acc": round(best_annealing_acc, 4),
            "improvement": round(annealing_improvement, 4),
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
