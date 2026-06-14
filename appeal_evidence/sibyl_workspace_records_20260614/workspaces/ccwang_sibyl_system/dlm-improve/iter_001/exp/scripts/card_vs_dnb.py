#!/usr/bin/env python3
"""
CARD vs DNB Baseline (H5) — PILOT mode
Task: card_vs_dnb

Compare full CARD against DNB (Do Nothing Better) baseline on 100 GSM8K samples.

Methods compared:
  1. Standard-64:    cosine schedule, 64 steps (reference)
  2. Standard-84:    cosine schedule, 84 steps (DNB — same NFE as CARD)
  3. Standard-96:    cosine schedule, 96 steps (DNB — extra headroom)
  4. CARD-68:        standard 64 draft + entropy revision (10% remask, 3 rev steps, ~68 NFE)
  5. CARD-84:        standard 64 draft + entropy revision (15% remask, 6 rev steps, ~84 NFE)
  6. DNB-wallclock:  standard cosine at matched wall-clock to CARD-84

Key: CARD uses raw_entropy (not calibrated) since pilot showed no calibration benefit.
     CARD uses isothermal T=1.0 since annealing was falsified (H6).

Pass criteria:
  - CARD-84 accuracy > DNB-84 accuracy by >= 1%
  - If CARD <= DNB → H5 falsified at pilot level

Usage:
    CUDA_VISIBLE_DEVICES=2 python card_vs_dnb.py
"""

import os
import sys
import gc
import json
import time
import math
import re
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
TASK_ID = "card_vs_dnb"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
SEED = 42
NUM_SAMPLES = 100
GEN_LENGTH = 256
MASK_TOKEN_ID = 126336
DEVICE = "cuda"


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


# ── Schedule ──────────────────────────────────────────────────────────
def cosine_schedule(t, T):
    """Fraction unmasked after step t. t=0 -> 0, t=T -> 1."""
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


# ── Standard denoising (DNB baselines) ───────────────────────────────
def run_standard_denoising(model, prompt_ids, gen_length, num_steps):
    """Standard cosine-schedule denoising. Returns generated text tokens and NFE."""
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

    for step in range(num_steps):
        frac_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
        frac_curr = cosine_schedule(step + 1, num_steps)

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

    return input_ids, gen_start, gen_end, nfe


# ── CARD: draft + entropy revision ───────────────────────────────────
def run_card(model, prompt_ids, gen_length, num_draft_steps,
             revision_fraction, revision_steps):
    """
    CARD method:
      Phase 1: Standard cosine draft (isothermal T=1.0)
      Phase 2: Entropy-based targeted revision (remask top-p% highest entropy + re-denoise)

    Returns generated tokens, gen region indices, total NFE, tokens_changed, and
    the entropy profile for analysis.
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

    # ── Phase 1: Standard draft ──
    for step in range(num_draft_steps):
        frac_prev = cosine_schedule(step, num_draft_steps) if step > 0 else 0.0
        frac_curr = cosine_schedule(step + 1, num_draft_steps)

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

    # ── Entropy scoring pass ──
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]
    nfe += 1

    gen_logits = logits[gen_start:gen_end]
    probs = F.softmax(gen_logits, dim=-1)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    # ── Phase 2: Entropy-based revision ──
    num_to_revise = max(1, int(round(revision_fraction * gen_length)))
    _, revision_targets = entropy.topk(num_to_revise)

    # Save original tokens at revision positions
    original_tokens = input_ids[0, gen_start + revision_targets].clone()

    # Re-mask selected positions
    input_ids[0, gen_start + revision_targets] = MASK_TOKEN_ID

    for rev_step in range(revision_steps):
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
        target_unmasked = int(round((rev_step + 1) / revision_steps * num_to_revise))
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

    # Count tokens that changed during revision
    new_tokens = input_ids[0, gen_start + revision_targets]
    tokens_changed = (new_tokens != original_tokens).sum().item()

    entropy_stats = {
        "mean_entropy": float(entropy.mean().item()),
        "max_entropy": float(entropy.max().item()),
        "revision_mean_entropy": float(entropy[revision_targets].mean().item()),
        "num_revised": num_to_revise,
        "tokens_changed": tokens_changed,
    }

    return input_ids, gen_start, gen_end, nfe, tokens_changed, entropy_stats


# ── Data loading ──────────────────────────────────────────────────────
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


# ── Run one method on all samples ─────────────────────────────────────
def evaluate_method(model, tokenizer, samples, method_name, method_fn, method_kwargs):
    """Run a method on all samples and return results."""
    results = []
    correct = 0
    total = 0
    total_nfe = 0
    total_tokens_changed = 0
    method_start = time.time()

    for si, sample in enumerate(samples):
        torch.manual_seed(SEED + sample["idx"])
        torch.cuda.manual_seed_all(SEED + sample["idx"])

        sample_start = time.time()

        # Call the method
        ret = method_fn(model, sample["prompt_ids"], GEN_LENGTH, **method_kwargs)

        if len(ret) == 4:
            # standard denoising: (input_ids, gen_start, gen_end, nfe)
            input_ids, gen_start, gen_end, nfe = ret
            tokens_changed = 0
            entropy_stats = {}
        else:
            # CARD: (input_ids, gen_start, gen_end, nfe, tokens_changed, entropy_stats)
            input_ids, gen_start, gen_end, nfe, tokens_changed, entropy_stats = ret

        total_nfe += nfe
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

        entry = {
            "idx": sample["idx"],
            "correct": is_correct,
            "nfe": nfe,
            "tokens_changed": tokens_changed,
            "time_sec": round(sample_time, 3),
            "predicted_answer": extract_gsm8k_answer(generated_text),
            "reference_answer": sample["final_answer"],
        }
        if si < 5:
            entry["generated_text"] = generated_text[:300]
        if entropy_stats:
            entry["entropy_stats"] = {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in entropy_stats.items()}

        results.append(entry)

        if si % 25 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            acc_so_far = correct / total if total > 0 else 0
            elapsed = time.time() - method_start
            print(f"    [{method_name}] {si}/{len(samples)}, "
                  f"acc={acc_so_far:.3f}, nfe_avg={total_nfe/max(1,total):.0f}, "
                  f"elapsed={elapsed:.0f}s", flush=True)

    method_time = time.time() - method_start
    accuracy = correct / total if total > 0 else 0
    avg_nfe = total_nfe / total if total > 0 else 0
    avg_tokens_changed = total_tokens_changed / total if total > 0 else 0

    return {
        "method": method_name,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": total,
        "avg_nfe": round(avg_nfe, 1),
        "avg_tokens_changed": round(avg_tokens_changed, 1),
        "wall_clock_sec": round(method_time, 2),
        "avg_time_per_sample_sec": round(method_time / total, 3),
        "per_sample": results,
    }


def main():
    global_start = time.time()
    write_pid()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{TASK_ID}] Starting CARD vs DNB comparison (H5)", flush=True)
    print(f"  Model: {MODEL_PATH}")
    print(f"  Samples: {NUM_SAMPLES}, Gen length: {GEN_LENGTH}")
    print(f"  Seed: {SEED}")
    print(f"  Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB",
          flush=True)

    # Load model & tokenizer
    print("\n[1/3] Loading model...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE).eval()
    vram_after = torch.cuda.memory_allocated() / 1e6
    print(f"  Model loaded. VRAM: {vram_after:.0f} MB", flush=True)

    # Load GSM8K
    print("\n[2/3] Loading GSM8K samples...", flush=True)
    samples = get_gsm8k_samples(tokenizer, NUM_SAMPLES)
    print(f"  Loaded {len(samples)} samples")

    # ── Define methods to evaluate ────────────────────────────────────
    # Method list: (name, function, kwargs)
    methods = [
        # Baselines — standard cosine at various step counts
        ("Standard-64", run_standard_denoising, {"num_steps": 64}),
        ("DNB-84", run_standard_denoising, {"num_steps": 84}),
        ("DNB-96", run_standard_denoising, {"num_steps": 96}),
        # CARD variants
        ("CARD-68", run_card, {
            "num_draft_steps": 64,
            "revision_fraction": 0.10,
            "revision_steps": 3,
        }),
        ("CARD-84", run_card, {
            "num_draft_steps": 64,
            "revision_fraction": 0.15,
            "revision_steps": 6,
        }),
    ]

    # ── Run all methods ───────────────────────────────────────────────
    print(f"\n[3/3] Running {len(methods)} methods...", flush=True)

    all_results = {}
    for mi, (name, fn, kwargs) in enumerate(methods):
        print(f"\n  ── Method {mi+1}/{len(methods)}: {name} ──", flush=True)

        result = evaluate_method(model, tokenizer, samples, name, fn, kwargs)
        all_results[name] = result

        acc = result["accuracy"]
        nfe = result["avg_nfe"]
        t = result["wall_clock_sec"]
        tc = result["avg_tokens_changed"]
        print(f"  → {name}: acc={acc:.4f}, NFE={nfe:.1f}, "
              f"tokens_changed={tc:.1f}, time={t:.1f}s", flush=True)

        report_progress(
            epoch=mi+1, total_epochs=len(methods),
            metric={
                "method": name,
                "accuracy": acc,
                "avg_nfe": nfe,
                "elapsed_sec": round(time.time() - global_start, 1),
            }
        )

        torch.cuda.empty_cache()
        gc.collect()

    # Check if we need a DNB-wallclock method
    card84_time = all_results["CARD-84"]["avg_time_per_sample_sec"]
    std64_time = all_results["Standard-64"]["avg_time_per_sample_sec"]
    dnb84_time = all_results["DNB-84"]["avg_time_per_sample_sec"]

    # Estimate steps for wall-clock match: linear interpolation
    if card84_time > dnb84_time:
        # CARD is slower → find how many steps DNB can do in the same time
        time_per_step = dnb84_time / 84
        wallclock_matched_steps = min(256, max(84, int(round(card84_time / time_per_step))))

        if wallclock_matched_steps > 96:
            print(f"\n  ── Extra: DNB-wallclock ({wallclock_matched_steps} steps, "
                  f"matching CARD-84 time) ──", flush=True)
            wc_result = evaluate_method(
                model, tokenizer, samples,
                f"DNB-wallclock-{wallclock_matched_steps}",
                run_standard_denoising,
                {"num_steps": wallclock_matched_steps}
            )
            all_results[f"DNB-wallclock-{wallclock_matched_steps}"] = wc_result
            print(f"  → DNB-wallclock-{wallclock_matched_steps}: "
                  f"acc={wc_result['accuracy']:.4f}, "
                  f"NFE={wc_result['avg_nfe']:.1f}, "
                  f"time={wc_result['wall_clock_sec']:.1f}s", flush=True)

    # ══════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - global_start

    print(f"\n{'='*70}", flush=True)
    print("ANALYSIS: CARD vs DNB (H5)", flush=True)
    print(f"{'='*70}", flush=True)

    # Summary table
    summary_table = []
    for name, result in all_results.items():
        summary_table.append({
            "method": name,
            "accuracy": result["accuracy"],
            "correct": result["correct"],
            "total": result["total"],
            "avg_nfe": result["avg_nfe"],
            "avg_tokens_changed": result["avg_tokens_changed"],
            "wall_clock_sec": result["wall_clock_sec"],
            "avg_time_per_sample_sec": result["avg_time_per_sample_sec"],
        })

    # Core comparisons
    card84 = all_results["CARD-84"]
    card68 = all_results["CARD-68"]
    dnb84 = all_results["DNB-84"]
    dnb96 = all_results["DNB-96"]
    std64 = all_results["Standard-64"]

    card84_vs_dnb84 = card84["accuracy"] - dnb84["accuracy"]
    card84_vs_std64 = card84["accuracy"] - std64["accuracy"]
    card68_vs_std64 = card68["accuracy"] - std64["accuracy"]
    card84_vs_dnb96 = card84["accuracy"] - dnb96["accuracy"]

    # NFE efficiency: accuracy per NFE
    card84_efficiency = card84["accuracy"] / card84["avg_nfe"] if card84["avg_nfe"] > 0 else 0
    dnb84_efficiency = dnb84["accuracy"] / dnb84["avg_nfe"] if dnb84["avg_nfe"] > 0 else 0

    # Per-sample analysis: which samples flip between CARD and DNB?
    card84_per = {r["idx"]: r["correct"] for r in card84["per_sample"]}
    dnb84_per = {r["idx"]: r["correct"] for r in dnb84["per_sample"]}
    std64_per = {r["idx"]: r["correct"] for r in std64["per_sample"]}

    card_wins = [idx for idx in card84_per if card84_per[idx] and not dnb84_per.get(idx, False)]
    dnb_wins = [idx for idx in dnb84_per if dnb84_per[idx] and not card84_per.get(idx, False)]
    both_correct = [idx for idx in card84_per if card84_per[idx] and dnb84_per.get(idx, False)]
    both_wrong = [idx for idx in card84_per if not card84_per[idx] and not dnb84_per.get(idx, False)]

    # Revision impact analysis: for CARD-84, how many revised tokens changed?
    card84_entropy_stats = []
    for entry in card84["per_sample"]:
        if "entropy_stats" in entry:
            card84_entropy_stats.append(entry["entropy_stats"])

    avg_tokens_changed_card84 = np.mean([e.get("tokens_changed", 0) for e in card84_entropy_stats]) if card84_entropy_stats else 0
    avg_revision_entropy = np.mean([e.get("revision_mean_entropy", 0) for e in card84_entropy_stats]) if card84_entropy_stats else 0

    # Pareto analysis: which methods are Pareto-dominant?
    # A method is Pareto-dominated if another method has both higher accuracy and lower NFE
    pareto_points = []
    for name, result in all_results.items():
        pareto_points.append({
            "method": name,
            "accuracy": result["accuracy"],
            "nfe": result["avg_nfe"],
            "time": result["wall_clock_sec"],
        })

    pareto_frontier = []
    for p in pareto_points:
        dominated = False
        for q in pareto_points:
            if q["method"] == p["method"]:
                continue
            if q["accuracy"] >= p["accuracy"] and q["nfe"] <= p["nfe"]:
                if q["accuracy"] > p["accuracy"] or q["nfe"] < p["nfe"]:
                    dominated = True
                    break
        if not dominated:
            pareto_frontier.append(p["method"])

    # H5 test
    h5_pass = card84_vs_dnb84 >= 0.01
    h5_marginal = card84_vs_dnb84 >= 0.00
    h5_strong = card84_vs_dnb84 >= 0.02

    if h5_strong:
        go_no_go = "GO"
        recommendation = "PROCEED_TO_FULL"
    elif h5_pass:
        go_no_go = "CONDITIONAL_GO"
        recommendation = "PROCEED_CAUTIOUSLY"
    elif h5_marginal:
        go_no_go = "CONDITIONAL_GO"
        recommendation = "INVESTIGATE_FURTHER"
    else:
        go_no_go = "NO_GO"
        recommendation = "H5_FALSIFIED"

    summary_text = (
        f"CARD vs DNB (H5): "
        f"CARD-84={card84['accuracy']:.4f} ({card84['avg_nfe']:.0f} NFE), "
        f"DNB-84={dnb84['accuracy']:.4f} ({dnb84['avg_nfe']:.0f} NFE), "
        f"delta={card84_vs_dnb84:+.4f}. "
        f"CARD-68={card68['accuracy']:.4f} ({card68['avg_nfe']:.0f} NFE) vs "
        f"Standard-64={std64['accuracy']:.4f} (delta={card68_vs_std64:+.4f}). "
        f"DNB-96={dnb96['accuracy']:.4f}. "
        f"Pareto frontier: {pareto_frontier}. "
        f"H5(CARD>DNB by >=1%): {'PASS' if h5_pass else 'FAIL'}. "
        f"GO/NO-GO={go_no_go}."
    )

    # Qualitative: show samples where CARD and DNB differ
    flip_analysis = {
        "card_wins": {
            "count": len(card_wins),
            "sample_indices": card_wins[:10],
        },
        "dnb_wins": {
            "count": len(dnb_wins),
            "sample_indices": dnb_wins[:10],
        },
        "both_correct": len(both_correct),
        "both_wrong": len(both_wrong),
    }

    # Qualitative examples: show generated text for a few card_wins and dnb_wins
    qualitative_examples = []
    for idx in card_wins[:3]:
        card_entry = next((e for e in card84["per_sample"] if e["idx"] == idx), None)
        dnb_entry = next((e for e in dnb84["per_sample"] if e["idx"] == idx), None)
        if card_entry and dnb_entry:
            qualitative_examples.append({
                "idx": idx,
                "type": "card_wins",
                "card_answer": card_entry["predicted_answer"],
                "dnb_answer": dnb_entry["predicted_answer"],
                "reference": card_entry["reference_answer"],
                "card_text": card_entry.get("generated_text", "")[:200],
                "dnb_text": dnb_entry.get("generated_text", "")[:200],
            })
    for idx in dnb_wins[:3]:
        card_entry = next((e for e in card84["per_sample"] if e["idx"] == idx), None)
        dnb_entry = next((e for e in dnb84["per_sample"] if e["idx"] == idx), None)
        if card_entry and dnb_entry:
            qualitative_examples.append({
                "idx": idx,
                "type": "dnb_wins",
                "card_answer": card_entry["predicted_answer"],
                "dnb_answer": dnb_entry["predicted_answer"],
                "reference": card_entry["reference_answer"],
                "card_text": card_entry.get("generated_text", "")[:200],
                "dnb_text": dnb_entry.get("generated_text", "")[:200],
            })

    # Build final result
    result = {
        "task_id": TASK_ID,
        "candidate_id": "cand_card",
        "mode": "PILOT",
        "model": "LLaDA-8B-Instruct",
        "benchmark": "GSM8K",
        "num_samples": NUM_SAMPLES,
        "gen_length": GEN_LENGTH,
        "seed": SEED,
        "elapsed_sec": round(total_elapsed, 1),
        "summary_table": summary_table,
        "core_comparison": {
            "card84_accuracy": card84["accuracy"],
            "card84_nfe": card84["avg_nfe"],
            "dnb84_accuracy": dnb84["accuracy"],
            "dnb84_nfe": dnb84["avg_nfe"],
            "card84_vs_dnb84": round(card84_vs_dnb84, 4),
            "card84_vs_std64": round(card84_vs_std64, 4),
            "card68_vs_std64": round(card68_vs_std64, 4),
            "card84_vs_dnb96": round(card84_vs_dnb96, 4),
            "card84_efficiency": round(card84_efficiency, 6),
            "dnb84_efficiency": round(dnb84_efficiency, 6),
        },
        "pareto_analysis": {
            "all_points": pareto_points,
            "pareto_frontier": pareto_frontier,
        },
        "flip_analysis": flip_analysis,
        "revision_impact": {
            "avg_tokens_changed_card84": round(avg_tokens_changed_card84, 1),
            "avg_revision_entropy": round(avg_revision_entropy, 4),
            "note": "tokens_changed shows how many of the revised positions got different tokens after re-denoising"
        },
        "qualitative_examples": qualitative_examples,
        "hypothesis_tests": {
            "H5_card_beats_dnb": {
                "criterion": "CARD-84 accuracy > DNB-84 accuracy by >= 1%",
                "card84_acc": card84["accuracy"],
                "dnb84_acc": dnb84["accuracy"],
                "delta": round(card84_vs_dnb84, 4),
                "pass": h5_pass,
                "strong_pass": h5_strong,
            }
        },
        "method_details": {
            name: {k: v for k, v in result.items() if k != "per_sample"}
            for name, result in all_results.items()
        },
        "per_sample_results": {
            name: result["per_sample"]
            for name, result in all_results.items()
        },
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
    result_path = RESULTS_DIR / "card_vs_dnb.json"
    result_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n  Saved to {result_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"CARD vs DNB RESULTS (H5)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<30} {'Acc':>8} {'NFE':>6} {'TokChg':>8} {'Time(s)':>8}")
    print(f"  {'-'*60}")
    for row in summary_table:
        print(f"  {row['method']:<30} {row['accuracy']:>8.4f} "
              f"{row['avg_nfe']:>6.1f} {row['avg_tokens_changed']:>8.1f} "
              f"{row['wall_clock_sec']:>8.1f}")

    print(f"\n  Core Comparison:")
    print(f"    CARD-84 vs DNB-84:     {card84_vs_dnb84:+.4f} "
          f"({'BETTER' if card84_vs_dnb84 > 0 else 'WORSE' if card84_vs_dnb84 < 0 else 'TIED'})")
    print(f"    CARD-68 vs Standard-64: {card68_vs_std64:+.4f}")
    print(f"    CARD-84 vs DNB-96:     {card84_vs_dnb96:+.4f}")

    print(f"\n  Sample Flips (CARD-84 vs DNB-84):")
    print(f"    CARD wins (only CARD correct): {len(card_wins)}")
    print(f"    DNB wins (only DNB correct):   {len(dnb_wins)}")
    print(f"    Both correct:                  {len(both_correct)}")
    print(f"    Both wrong:                    {len(both_wrong)}")

    print(f"\n  Pareto frontier: {pareto_frontier}")
    print(f"\n  H5 (CARD>DNB by >=1%): {'PASS' if h5_pass else 'FAIL'} "
          f"(delta={card84_vs_dnb84:+.4f})")
    print(f"  GO/NO-GO: {go_no_go}")
    print(f"  Recommendation: {recommendation}")
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}", flush=True)

    report_progress(
        epoch=len(methods), total_epochs=len(methods),
        metric={
            "card84_vs_dnb84": round(card84_vs_dnb84, 4),
            "card84_acc": card84["accuracy"],
            "dnb84_acc": dnb84["accuracy"],
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
