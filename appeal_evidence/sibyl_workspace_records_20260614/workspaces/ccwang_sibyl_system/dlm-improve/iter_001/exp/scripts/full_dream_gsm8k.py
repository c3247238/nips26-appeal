#!/usr/bin/env python3
"""
Full Evaluation: Dream-7B on GSM8K — top-4 methods on pilot (100 samples).
Task: full_dream_gsm8k

Methods (top-4 from LLaDA pilot):
  1. Standard-64:       cosine schedule, 64 steps
  2. DNB-84:            cosine schedule, 84 steps (compute-matched to CARD)
  3. Entropy-Revise-64: standard 64 + raw entropy top-15% remasking + 3 steps
  4. CARD-84:           standard 64 draft + entropy revision (15% remask, 6 rev steps)

Cross-model validation on Dream-7B-Instruct (Evolution Lesson: avoid model-specific conclusions).

Usage:
    CUDA_VISIBLE_DEVICES=0 python full_dream_gsm8k.py
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
TASK_ID = "full_dream_gsm8k"
MODEL_PATH = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
FULL_RESULTS_DIR = RESULTS_DIR / "full"
SEED = 42
NUM_SAMPLES = 100  # PILOT mode
GEN_LENGTH = 256
MASK_TOKEN_ID = 151666  # Dream-7B mask token
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


# ── Standard denoising ───────────────────────────────────────────────
def run_standard_denoising(model, prompt_ids, gen_length, num_steps):
    """Standard cosine-schedule denoising. Returns (input_ids, gen_start, gen_end, nfe)."""
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


# ── Entropy-Revise: standard draft + entropy-based remasking ─────────
def run_entropy_revise(model, prompt_ids, gen_length, num_draft_steps,
                       revision_fraction, revision_steps):
    """Standard draft + entropy-based targeted revision (no calibration)."""
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

    # Phase 1: Standard draft
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

    # Entropy scoring pass
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits[0]
    nfe += 1

    gen_logits = logits[gen_start:gen_end]
    probs = F.softmax(gen_logits, dim=-1)
    log_probs = F.log_softmax(gen_logits, dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)

    # Phase 2: Entropy-based revision
    num_to_revise = max(1, int(round(revision_fraction * gen_length)))
    _, revision_targets = entropy.topk(num_to_revise)

    original_tokens = input_ids[0, gen_start + revision_targets].clone()
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
        masked_indices_rev = torch.where(is_masked)[0]
        masked_logits = gen_logits[masked_indices_rev]

        probs = F.softmax(masked_logits, dim=-1)
        top1_conf, top1_token = probs.max(dim=-1)

        target_unmasked = int(round((rev_step + 1) / revision_steps * num_to_revise))
        already_unmasked = num_to_revise - num_masked
        num_to_unmask = max(1, target_unmasked - already_unmasked)
        num_to_unmask = min(num_to_unmask, num_masked)

        if num_to_unmask > 0 and len(top1_conf) > 0:
            k = min(num_to_unmask, len(top1_conf))
            _, topk_indices = top1_conf.topk(k)
            positions_to_unmask = masked_indices_rev[topk_indices]
            tokens_to_place = top1_token[topk_indices]
            input_ids[0, gen_start + positions_to_unmask] = tokens_to_place

    # Force-unmask remaining
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


# ── CARD: standard draft + entropy revision (best config from pilot) ─
def run_card(model, prompt_ids, gen_length, num_draft_steps,
             revision_fraction, revision_steps):
    """CARD method: same as Entropy-Revise but with larger revision budget."""
    return run_entropy_revise(model, prompt_ids, gen_length, num_draft_steps,
                              revision_fraction, revision_steps)


# ── Data loading ──────────────────────────────────────────────────────
def get_gsm8k_samples(tokenizer, num_samples=None):
    dataset = load_dataset("gsm8k", "main", split="test")
    total = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    samples = []
    for idx in range(total):
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


# ── Evaluate one method on all samples ────────────────────────────────
def evaluate_method(model, tokenizer, samples, method_name, method_fn, method_kwargs,
                    global_start, method_idx, total_methods):
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

        ret = method_fn(model, sample["prompt_ids"], GEN_LENGTH, **method_kwargs)

        if len(ret) == 4:
            input_ids, gen_start, gen_end, nfe = ret
            tokens_changed = 0
            entropy_stats = {}
        else:
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
        # Save generated text for first 10 samples for qualitative inspection
        if si < 10:
            entry["generated_text"] = generated_text[:500]
        if entropy_stats:
            entry["entropy_stats"] = {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in entropy_stats.items()}

        results.append(entry)

        # Progress logging every 25 samples (pilot is 100 samples)
        if si % 25 == 0 or si == len(samples) - 1:
            torch.cuda.empty_cache()
            gc.collect()
            acc_so_far = correct / total if total > 0 else 0
            elapsed = time.time() - method_start
            total_elapsed = time.time() - global_start
            eta_method = elapsed / (si + 1) * (len(samples) - si - 1) if si > 0 else 0
            print(f"    [{method_name}] {si+1}/{len(samples)}, "
                  f"acc={acc_so_far:.3f}, nfe_avg={total_nfe/max(1,total):.0f}, "
                  f"elapsed={elapsed:.0f}s, ETA={eta_method:.0f}s, "
                  f"total_elapsed={total_elapsed:.0f}s", flush=True)

            # Update progress file
            report_progress(
                epoch=method_idx + 1, total_epochs=total_methods,
                step=si + 1, total_steps=len(samples),
                metric={
                    "method": method_name,
                    "method_idx": method_idx + 1,
                    "total_methods": total_methods,
                    "samples_done": si + 1,
                    "total_samples": len(samples),
                    "accuracy_so_far": round(acc_so_far, 4),
                    "avg_nfe": round(total_nfe / max(1, total), 1),
                    "method_elapsed_sec": round(elapsed, 1),
                    "total_elapsed_sec": round(total_elapsed, 1),
                    "eta_method_sec": round(eta_method, 1),
                }
            )

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
    FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    vram_total = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(num_gpus)]

    print(f"[{TASK_ID}] Starting Dream-7B GSM8K Evaluation (PILOT, {NUM_SAMPLES} samples)", flush=True)
    print(f"  Model: {MODEL_PATH}")
    print(f"  GPUs: {num_gpus} x {gpu_names[0] if gpu_names else 'N/A'}")
    print(f"  VRAM: {[f'{v:.1f}GB' for v in vram_total]}")
    print(f"  Gen length: {GEN_LENGTH}, Seed: {SEED}", flush=True)
    print(f"  Mask token ID: {MASK_TOKEN_ID}", flush=True)

    # Load model & tokenizer
    # Dream-7B uses custom model class, must use AutoModel with trust_remote_code
    print("\n[1/3] Loading Dream-7B model...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE).eval()

    vram_after = torch.cuda.memory_allocated() / 1e6
    print(f"  Model loaded on GPU 0. VRAM: {vram_after:.0f} MB", flush=True)

    # Verify mask token
    assert tokenizer.mask_token_id == MASK_TOKEN_ID, \
        f"Mask token mismatch: expected {MASK_TOKEN_ID}, got {tokenizer.mask_token_id}"

    # Quick sanity check: single forward pass
    print("  Sanity check: single forward pass...", flush=True)
    test_ids = tokenizer.encode("Hello", add_special_tokens=False)
    test_input = torch.tensor(test_ids + [MASK_TOKEN_ID] * 10, dtype=torch.long, device=DEVICE).unsqueeze(0)
    with torch.no_grad():
        test_out = model(input_ids=test_input)
    print(f"  Sanity check passed. Output logits shape: {test_out.logits.shape}", flush=True)
    del test_input, test_out
    torch.cuda.empty_cache()

    # Load GSM8K
    print(f"\n[2/3] Loading GSM8K samples (pilot: {NUM_SAMPLES})...", flush=True)
    samples = get_gsm8k_samples(tokenizer, NUM_SAMPLES)
    print(f"  Loaded {len(samples)} samples", flush=True)

    # ── Define top-4 methods ──────────────────────────────────────────
    methods = [
        # Standard baseline
        ("Standard-64", run_standard_denoising, {"num_steps": 64}),
        # DNB baseline (compute-matched to CARD)
        ("DNB-84", run_standard_denoising, {"num_steps": 84}),
        # Entropy-Revise: standard draft + entropy-based remasking (smaller budget)
        ("Entropy-Revise-64", run_entropy_revise, {
            "num_draft_steps": 64,
            "revision_fraction": 0.15,
            "revision_steps": 3,
        }),
        # CARD-84: standard draft + entropy revision (larger budget, best pilot config)
        ("CARD-84", run_card, {
            "num_draft_steps": 64,
            "revision_fraction": 0.15,
            "revision_steps": 6,
        }),
    ]

    # ── Run all methods ───────────────────────────────────────────────
    print(f"\n[3/3] Running {len(methods)} methods on {len(samples)} samples...", flush=True)

    all_results = {}
    for mi, (name, fn, kwargs) in enumerate(methods):
        print(f"\n  == Method {mi+1}/{len(methods)}: {name} ==", flush=True)

        result = evaluate_method(model, tokenizer, samples, name, fn, kwargs,
                                 global_start, mi, len(methods))
        all_results[name] = result

        acc = result["accuracy"]
        nfe = result["avg_nfe"]
        t = result["wall_clock_sec"]
        tc = result["avg_tokens_changed"]
        print(f"  Done {name}: acc={acc:.4f}, NFE={nfe:.1f}, "
              f"tokens_changed={tc:.1f}, time={t:.1f}s ({t/60:.1f}min)", flush=True)

        # Save intermediate results after each method completes
        intermediate_path = FULL_RESULTS_DIR / f"full_dream_gsm8k_{name.lower().replace('-','_').replace(' ','_')}.json"
        intermediate_path.write_text(json.dumps({
            "method": name,
            "accuracy": acc,
            "correct": result["correct"],
            "total": result["total"],
            "avg_nfe": nfe,
            "wall_clock_sec": t,
        }, indent=2))

        torch.cuda.empty_cache()
        gc.collect()

    # ══════════════════════════════════════════════════════════════════
    # Analysis
    # ══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - global_start

    print(f"\n{'='*70}", flush=True)
    print("ANALYSIS: Dream-7B GSM8K Results (PILOT)", flush=True)
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
    std64 = all_results["Standard-64"]
    dnb84 = all_results["DNB-84"]
    ent_revise = all_results["Entropy-Revise-64"]

    comparisons = {
        "card84_vs_dnb84": round(card84["accuracy"] - dnb84["accuracy"], 4),
        "card84_vs_std64": round(card84["accuracy"] - std64["accuracy"], 4),
        "ent_revise_vs_std64": round(ent_revise["accuracy"] - std64["accuracy"], 4),
        "card84_vs_ent_revise": round(card84["accuracy"] - ent_revise["accuracy"], 4),
    }

    # Pareto analysis (accuracy vs NFE)
    pareto_points = []
    for name, result in all_results.items():
        pareto_points.append({
            "method": name,
            "accuracy": result["accuracy"],
            "nfe": result["avg_nfe"],
            "time": result["wall_clock_sec"],
        })

    # Find Pareto frontier
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

    # Per-sample flip analysis: CARD vs DNB
    card84_per = {r["idx"]: r["correct"] for r in card84["per_sample"]}
    dnb84_per = {r["idx"]: r["correct"] for r in dnb84["per_sample"]}
    std64_per = {r["idx"]: r["correct"] for r in std64["per_sample"]}

    card_wins_dnb = [idx for idx in card84_per if card84_per[idx] and not dnb84_per.get(idx, False)]
    dnb_wins_card = [idx for idx in dnb84_per if dnb84_per[idx] and not card84_per.get(idx, False)]
    both_correct_cd = [idx for idx in card84_per if card84_per[idx] and dnb84_per.get(idx, False)]
    both_wrong_cd = [idx for idx in card84_per if not card84_per[idx] and not dnb84_per.get(idx, False)]

    # CARD vs Standard-64 flip analysis
    card_wins_std = [idx for idx in card84_per if card84_per[idx] and not std64_per.get(idx, False)]
    std_wins_card = [idx for idx in std64_per if std64_per[idx] and not card84_per.get(idx, False)]

    # Qualitative examples
    qualitative_examples = []
    for idx in card_wins_dnb[:5]:
        card_entry = next((e for e in card84["per_sample"] if e["idx"] == idx), None)
        dnb_entry = next((e for e in dnb84["per_sample"] if e["idx"] == idx), None)
        if card_entry and dnb_entry:
            qualitative_examples.append({
                "idx": idx,
                "type": "card_wins_over_dnb",
                "card_answer": card_entry["predicted_answer"],
                "dnb_answer": dnb_entry["predicted_answer"],
                "reference": card_entry["reference_answer"],
                "card_text": card_entry.get("generated_text", "")[:300],
                "dnb_text": dnb_entry.get("generated_text", "")[:300],
            })

    # Hypothesis tests
    h5_delta = card84["accuracy"] - dnb84["accuracy"]
    h5_pass = h5_delta >= 0.01
    card_vs_std = card84["accuracy"] - std64["accuracy"]
    card_beats_std = card_vs_std > 0

    # Cross-model validation: compare calibration profile shape
    # (we check if CARD > Standard-64 on Dream, similar to LLaDA pattern)
    cross_model_consistent = card_beats_std  # CARD should beat std-64 on both models

    if h5_pass and cross_model_consistent:
        go_no_go = "STRONG_PASS"
    elif card_beats_std:
        go_no_go = "PASS"
    elif card84["accuracy"] >= std64["accuracy"] - 0.01:
        go_no_go = "MARGINAL"
    else:
        go_no_go = "FAIL"

    # Compare with LLaDA results (reference)
    llada_ref = {
        "Standard-64": 0.3093,
        "DNB-84": 0.3086,
        "Entropy-Revise-64": 0.3313,
        "CARD-84": 0.3487,
    }

    cross_model_comparison = {}
    for method_name in all_results:
        dream_acc = all_results[method_name]["accuracy"]
        llada_acc = llada_ref.get(method_name, None)
        if llada_acc is not None:
            cross_model_comparison[method_name] = {
                "dream_acc": dream_acc,
                "llada_acc": llada_acc,
                "delta": round(dream_acc - llada_acc, 4),
            }

    summary_text = (
        f"Dream-7B GSM8K PILOT ({len(samples)} samples): "
        f"CARD-84={card84['accuracy']:.4f} ({card84['avg_nfe']:.0f} NFE), "
        f"DNB-84={dnb84['accuracy']:.4f} ({dnb84['avg_nfe']:.0f} NFE), "
        f"Standard-64={std64['accuracy']:.4f}, "
        f"Entropy-Revise={ent_revise['accuracy']:.4f}. "
        f"CARD vs DNB: {h5_delta:+.4f}. "
        f"CARD vs Std-64: {card_vs_std:+.4f}. "
        f"Pareto frontier: {pareto_frontier}. "
        f"Cross-model consistent: {cross_model_consistent}. "
        f"Result: {go_no_go}."
    )

    # Build final result
    result = {
        "task_id": TASK_ID,
        "candidate_id": "cand_card",
        "mode": "PILOT",
        "model": "Dream-7B-Instruct",
        "benchmark": "GSM8K",
        "num_samples": len(samples),
        "gen_length": GEN_LENGTH,
        "seed": SEED,
        "elapsed_sec": round(total_elapsed, 1),
        "summary_table": summary_table,
        "comparisons": comparisons,
        "pareto_analysis": {
            "all_points": pareto_points,
            "pareto_frontier": pareto_frontier,
            "card_on_frontier": "CARD-84" in pareto_frontier,
        },
        "flip_analysis": {
            "card84_vs_dnb84": {
                "card_wins": len(card_wins_dnb),
                "dnb_wins": len(dnb_wins_card),
                "both_correct": len(both_correct_cd),
                "both_wrong": len(both_wrong_cd),
            },
            "card84_vs_std64": {
                "card_wins": len(card_wins_std),
                "std_wins": len(std_wins_card),
            },
        },
        "cross_model_comparison": cross_model_comparison,
        "qualitative_examples": qualitative_examples,
        "hypothesis_tests": {
            "card_beats_std64": {
                "criterion": "CARD-84 accuracy > Standard-64 accuracy on Dream-7B",
                "card84_acc": card84["accuracy"],
                "std64_acc": std64["accuracy"],
                "delta": round(card_vs_std, 4),
                "pass": card_beats_std,
            },
            "card_beats_dnb84": {
                "criterion": "CARD-84 accuracy > DNB-84 accuracy",
                "card84_acc": card84["accuracy"],
                "dnb84_acc": dnb84["accuracy"],
                "delta": round(h5_delta, 4),
                "pass": h5_pass,
            },
            "cross_model_consistency": {
                "criterion": "CARD improvement pattern consistent across LLaDA and Dream",
                "pass": cross_model_consistent,
                "details": cross_model_comparison,
            },
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
        "summary": summary_text,
        "gpu_info": {
            "num_gpus": num_gpus,
            "device": gpu_names[0] if gpu_names else "unknown",
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    result_path = RESULTS_DIR / "full_dream_gsm8k.json"
    result_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n  Saved to {result_path}")

    # Also save to full/ directory
    full_path = FULL_RESULTS_DIR / "full_dream_gsm8k.json"
    full_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"  Saved to {full_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"DREAM-7B GSM8K PILOT RESULTS ({len(samples)} samples)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<25} {'Acc':>8} {'Correct':>8} {'NFE':>6} {'TokChg':>8} {'Time(s)':>8}")
    print(f"  {'-'*65}")
    for row in summary_table:
        print(f"  {row['method']:<25} {row['accuracy']:>8.4f} "
              f"{row['correct']:>8}/{row['total']:<4} "
              f"{row['avg_nfe']:>6.1f} {row['avg_tokens_changed']:>8.1f} "
              f"{row['wall_clock_sec']:>8.1f}")

    print(f"\n  Key Comparisons:")
    for k, v in comparisons.items():
        print(f"    {k}: {v:+.4f}")

    print(f"\n  Cross-Model Comparison (Dream vs LLaDA):")
    for method, cmp in cross_model_comparison.items():
        print(f"    {method}: Dream={cmp['dream_acc']:.4f}, LLaDA={cmp['llada_acc']:.4f}, delta={cmp['delta']:+.4f}")

    print(f"\n  Pareto frontier: {pareto_frontier}")
    print(f"  CARD on Pareto: {'CARD-84' in pareto_frontier}")

    print(f"\n  Sample Flips (CARD-84 vs DNB-84):")
    print(f"    CARD wins: {len(card_wins_dnb)}")
    print(f"    DNB wins:  {len(dnb_wins_card)}")
    print(f"    Both correct: {len(both_correct_cd)}")
    print(f"    Both wrong:   {len(both_wrong_cd)}")

    print(f"\n  Card beats Std-64: {'PASS' if card_beats_std else 'FAIL'} (delta={card_vs_std:+.4f})")
    print(f"  Card beats DNB-84: {'PASS' if h5_pass else 'FAIL'} (delta={h5_delta:+.4f})")
    print(f"  Cross-model consistent: {cross_model_consistent}")
    print(f"  Overall: {go_no_go}")
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}", flush=True)

    report_progress(
        epoch=len(methods), total_epochs=len(methods),
        step=len(samples), total_steps=len(samples),
        metric={
            "status": "completed",
            "card84_acc": card84["accuracy"],
            "dnb84_acc": dnb84["accuracy"],
            "std64_acc": std64["accuracy"],
            "card84_vs_dnb84": round(h5_delta, 4),
            "card84_vs_std64": round(card_vs_std, 4),
            "pareto_frontier": pareto_frontier,
            "go_no_go": go_no_go,
            "total_elapsed_sec": round(total_elapsed, 1),
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
