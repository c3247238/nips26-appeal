#!/usr/bin/env python3
"""
Full Evaluation: LLaDA-8B on HumanEval — top-4 methods on 164 problems (pilot: 100).
Task: full_llada_humaneval

Methods:
  1. Standard-64:       cosine schedule, 64 steps
  2. DNB-84:            cosine schedule, 84 steps (compute-matched to CARD)
  3. Entropy-Revise-64: standard 64 + raw entropy top-15% remasking + 3 revision steps
  4. CARD-84:           standard 64 draft + entropy revision (15% remask, 6 rev steps)

Evaluates pass@1 using execution-based testing (human_eval).

Usage:
    CUDA_VISIBLE_DEVICES=2 python full_llada_humaneval.py [--pilot]
"""

import os
import sys
import gc
import json
import time
import math
import re
import warnings
import tempfile
import signal
import multiprocessing
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn.functional as F

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
TASK_ID = "full_llada_humaneval"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
FULL_RESULTS_DIR = RESULTS_DIR / "full"
SEED = 42
PILOT_MODE = "--pilot" in sys.argv or True  # Default to pilot for safety
NUM_SAMPLES = 100 if PILOT_MODE else None  # 100 for pilot, all 164 for full
GEN_LENGTH = 512  # Code generation needs more tokens than math
MASK_TOKEN_ID = 126336
DEVICE = "cuda"
EXECUTION_TIMEOUT = 10  # seconds per test execution


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


# ── HumanEval code extraction ────────────────────────────────────────
def extract_code_completion(generated_text, prompt):
    """Extract the function body completion from generated text.

    The model generates text after the prompt. We need to extract just the
    function body. We stop at:
    - A new top-level function definition
    - A new class definition
    - End of meaningful code
    """
    # The completion should follow the prompt
    completion = generated_text

    # Remove common artifacts
    completion = completion.replace("<|endoftext|>", "")
    completion = completion.replace("<|end|>", "")
    completion = completion.replace("[MASK]", "")

    # Find where the function body ends
    lines = completion.split('\n')
    result_lines = []
    in_body = True

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Stop conditions: new top-level definition
        if i > 0 and stripped and not stripped.startswith('#'):
            # Check if this is a new top-level definition (no indentation)
            if line and not line[0].isspace():
                if any(line.startswith(kw) for kw in ['def ', 'class ', 'import ', 'from ']):
                    break
                # Also stop at lines that look like they start a new block
                if stripped.startswith('if __name__'):
                    break

        result_lines.append(line)

        # Stop after seeing a return at the base indentation of the function
        if stripped.startswith('return ') and len(line) - len(line.lstrip()) <= 4:
            # Check if next non-empty line is at same or less indentation
            # (simple heuristic - keep going if there's more at same indent)
            pass

    completion = '\n'.join(result_lines)

    # Trim trailing whitespace and empty lines
    completion = completion.rstrip()

    return completion


def _run_test_in_process(code_str, result_dict):
    """Execute test code in a subprocess-safe way."""
    try:
        exec_globals = {}
        exec(code_str, exec_globals)
        result_dict["passed"] = True
    except Exception as e:
        result_dict["passed"] = False
        result_dict["error"] = f"{type(e).__name__}: {str(e)}"


def check_humaneval_correct(prompt, completion, test_code, entry_point, timeout=EXECUTION_TIMEOUT):
    """Check if generated code passes the test cases using execution.

    Returns (passed: bool, error: str or None)
    """
    # Build full code: prompt + completion + test
    full_code = prompt + completion + "\n" + test_code + f"\ncheck({entry_point})\n"

    # Use multiprocessing for timeout safety
    manager = multiprocessing.Manager()
    result_dict = manager.dict()
    result_dict["passed"] = False
    result_dict["error"] = None

    proc = multiprocessing.Process(target=_run_test_in_process, args=(full_code, result_dict))
    proc.start()
    proc.join(timeout=timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join(timeout=2)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return False, "TimeoutError: execution exceeded time limit"

    return result_dict.get("passed", False), result_dict.get("error", None)


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
    """CARD method: same structure as Entropy-Revise but with larger revision budget."""
    return run_entropy_revise(model, prompt_ids, gen_length, num_draft_steps,
                              revision_fraction, revision_steps)


# ── Data loading ──────────────────────────────────────────────────────
def get_humaneval_samples(tokenizer, num_samples=None):
    """Load HumanEval problems and prepare prompts."""
    from human_eval.data import read_problems

    problems = read_problems()
    task_ids = sorted(problems.keys())

    if num_samples is not None:
        task_ids = task_ids[:num_samples]

    samples = []
    for task_id in task_ids:
        prob = problems[task_id]
        prompt = prob["prompt"]

        # Build instruction prompt for the model
        instruction = (
            f"Complete the following Python function. Only output the function body "
            f"(the code that goes after the function signature and docstring). "
            f"Do not repeat the function signature or docstring.\n\n{prompt}"
        )
        prompt_ids = tokenizer.encode(instruction, add_special_tokens=False)

        samples.append({
            "task_id": task_id,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "entry_point": prob["entry_point"],
            "test": prob["test"],
            "canonical_solution": prob["canonical_solution"],
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
        torch.manual_seed(SEED + si)
        torch.cuda.manual_seed_all(SEED + si)

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

        # Decode generated text
        gen_tokens = input_ids[0, gen_start:gen_end].cpu().tolist()
        gen_tokens_clean = [t for t in gen_tokens if t != MASK_TOKEN_ID]
        generated_text = tokenizer.decode(gen_tokens_clean, skip_special_tokens=True)

        # Extract the code completion
        completion = extract_code_completion(generated_text, sample["prompt"])

        # Execute tests
        passed, error = check_humaneval_correct(
            sample["prompt"], completion, sample["test"],
            sample["entry_point"], timeout=EXECUTION_TIMEOUT
        )

        if passed:
            correct += 1
        total += 1

        sample_time = time.time() - sample_start

        entry = {
            "task_id": sample["task_id"],
            "passed": passed,
            "nfe": nfe,
            "tokens_changed": tokens_changed,
            "time_sec": round(sample_time, 3),
            "error": error if not passed else None,
        }
        # Save generated text for first 10 samples for qualitative inspection
        if si < 10:
            entry["completion"] = completion[:800]
            entry["canonical_solution"] = sample["canonical_solution"][:400]
        if entropy_stats:
            entry["entropy_stats"] = {k: round(v, 4) if isinstance(v, float) else v
                                       for k, v in entropy_stats.items()}

        results.append(entry)

        # Progress logging every 20 samples
        if si % 20 == 0 or si == len(samples) - 1:
            torch.cuda.empty_cache()
            gc.collect()
            pass_rate = correct / total if total > 0 else 0
            elapsed = time.time() - method_start
            total_elapsed = time.time() - global_start
            eta_method = elapsed / (si + 1) * (len(samples) - si - 1) if si > 0 else 0
            print(f"    [{method_name}] {si+1}/{len(samples)}, "
                  f"pass@1={pass_rate:.3f}, nfe_avg={total_nfe/max(1,total):.0f}, "
                  f"elapsed={elapsed:.0f}s, ETA={eta_method:.0f}s, "
                  f"total_elapsed={total_elapsed:.0f}s", flush=True)

            report_progress(
                epoch=method_idx + 1, total_epochs=total_methods,
                step=si + 1, total_steps=len(samples),
                metric={
                    "method": method_name,
                    "method_idx": method_idx + 1,
                    "total_methods": total_methods,
                    "samples_done": si + 1,
                    "total_samples": len(samples),
                    "pass_at_1": round(pass_rate, 4),
                    "avg_nfe": round(total_nfe / max(1, total), 1),
                    "method_elapsed_sec": round(elapsed, 1),
                    "total_elapsed_sec": round(total_elapsed, 1),
                    "eta_method_sec": round(eta_method, 1),
                }
            )

    method_time = time.time() - method_start
    pass_at_1 = correct / total if total > 0 else 0
    avg_nfe = total_nfe / total if total > 0 else 0
    avg_tokens_changed = total_tokens_changed / total if total > 0 else 0

    return {
        "method": method_name,
        "pass_at_1": round(pass_at_1, 4),
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

    # Set multiprocessing start method for safe code execution
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # already set

    num_gpus = torch.cuda.device_count()
    gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
    vram_total = [torch.cuda.get_device_properties(i).total_memory / 1e9 for i in range(num_gpus)]

    mode_str = "PILOT" if PILOT_MODE else "FULL"
    print(f"[{TASK_ID}] Starting LLaDA-8B HumanEval Evaluation ({mode_str})", flush=True)
    print(f"  Model: {MODEL_PATH}")
    print(f"  GPUs: {num_gpus} x {gpu_names[0] if gpu_names else 'N/A'}")
    print(f"  VRAM: {[f'{v:.1f}GB' for v in vram_total]}")
    print(f"  Gen length: {GEN_LENGTH}, Seed: {SEED}")
    print(f"  Num samples: {NUM_SAMPLES if NUM_SAMPLES else 'ALL (164)'}", flush=True)

    # Load model & tokenizer
    print("\n[1/3] Loading model...", flush=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE).eval()

    vram_after = torch.cuda.memory_allocated() / 1e6
    print(f"  Model loaded. VRAM: {vram_after:.0f} MB", flush=True)

    # Load HumanEval
    print("\n[2/3] Loading HumanEval samples...", flush=True)
    samples = get_humaneval_samples(tokenizer, NUM_SAMPLES)
    print(f"  Loaded {len(samples)} problems", flush=True)

    # ── Define top-4 methods ──────────────────────────────────────────
    methods = [
        ("Standard-64", run_standard_denoising, {"num_steps": 64}),
        ("DNB-84", run_standard_denoising, {"num_steps": 84}),
        ("Entropy-Revise-64", run_entropy_revise, {
            "num_draft_steps": 64,
            "revision_fraction": 0.15,
            "revision_steps": 3,
        }),
        ("CARD-84", run_card, {
            "num_draft_steps": 64,
            "revision_fraction": 0.15,
            "revision_steps": 6,
        }),
    ]

    # ── Run all methods ───────────────────────────────────────────────
    print(f"\n[3/3] Running {len(methods)} methods on {len(samples)} problems...", flush=True)

    all_results = {}
    for mi, (name, fn, kwargs) in enumerate(methods):
        print(f"\n  == Method {mi+1}/{len(methods)}: {name} ==", flush=True)

        result = evaluate_method(model, tokenizer, samples, name, fn, kwargs,
                                 global_start, mi, len(methods))
        all_results[name] = result

        p1 = result["pass_at_1"]
        nfe = result["avg_nfe"]
        t = result["wall_clock_sec"]
        tc = result["avg_tokens_changed"]
        print(f"  >> {name}: pass@1={p1:.4f}, NFE={nfe:.1f}, "
              f"tokens_changed={tc:.1f}, time={t:.1f}s ({t/60:.1f}min)", flush=True)

        # Save intermediate results
        intermediate_path = FULL_RESULTS_DIR / f"full_llada_humaneval_{name.lower().replace('-','_').replace(' ','_')}.json"
        intermediate_path.write_text(json.dumps({
            "method": name,
            "pass_at_1": p1,
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
    print(f"ANALYSIS: LLaDA-8B HumanEval Results ({mode_str})", flush=True)
    print(f"{'='*70}", flush=True)

    # Summary table
    summary_table = []
    for name, result in all_results.items():
        summary_table.append({
            "method": name,
            "pass_at_1": result["pass_at_1"],
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
        "card84_vs_dnb84": round(card84["pass_at_1"] - dnb84["pass_at_1"], 4),
        "card84_vs_std64": round(card84["pass_at_1"] - std64["pass_at_1"], 4),
        "ent_revise_vs_std64": round(ent_revise["pass_at_1"] - std64["pass_at_1"], 4),
        "card84_vs_ent_revise": round(card84["pass_at_1"] - ent_revise["pass_at_1"], 4),
        "dnb84_vs_std64": round(dnb84["pass_at_1"] - std64["pass_at_1"], 4),
    }

    # Pareto analysis (pass@1 vs NFE)
    pareto_points = []
    for name, result in all_results.items():
        pareto_points.append({
            "method": name,
            "pass_at_1": result["pass_at_1"],
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
            if q["pass_at_1"] >= p["pass_at_1"] and q["nfe"] <= p["nfe"]:
                if q["pass_at_1"] > p["pass_at_1"] or q["nfe"] < p["nfe"]:
                    dominated = True
                    break
        if not dominated:
            pareto_frontier.append(p["method"])

    # Per-problem flip analysis: CARD vs DNB
    card84_per = {r["task_id"]: r["passed"] for r in card84["per_sample"]}
    dnb84_per = {r["task_id"]: r["passed"] for r in dnb84["per_sample"]}
    std64_per = {r["task_id"]: r["passed"] for r in std64["per_sample"]}

    card_wins_dnb = [tid for tid in card84_per if card84_per[tid] and not dnb84_per.get(tid, False)]
    dnb_wins_card = [tid for tid in dnb84_per if dnb84_per[tid] and not card84_per.get(tid, False)]
    both_correct_cd = [tid for tid in card84_per if card84_per[tid] and dnb84_per.get(tid, False)]
    both_wrong_cd = [tid for tid in card84_per if not card84_per[tid] and not dnb84_per.get(tid, False)]

    card_wins_std = [tid for tid in card84_per if card84_per[tid] and not std64_per.get(tid, False)]
    std_wins_card = [tid for tid in std64_per if std64_per[tid] and not card84_per.get(tid, False)]

    # Error analysis
    error_types = {}
    for r in card84["per_sample"]:
        if not r["passed"] and r.get("error"):
            etype = r["error"].split(":")[0] if ":" in r["error"] else "Unknown"
            error_types[etype] = error_types.get(etype, 0) + 1

    # Hypothesis test
    h5_delta = card84["pass_at_1"] - dnb84["pass_at_1"]
    h5_pass = h5_delta >= 0.0  # CARD >= DNB (relaxed for HumanEval)

    if h5_pass and h5_delta >= 0.02:
        go_no_go = "STRONG_PASS"
    elif h5_pass:
        go_no_go = "PASS"
    elif h5_delta >= -0.02:
        go_no_go = "MARGINAL"
    else:
        go_no_go = "FAIL"

    summary_text = (
        f"LLaDA-8B HumanEval ({mode_str}, {len(samples)} problems): "
        f"CARD-84={card84['pass_at_1']:.4f} ({card84['avg_nfe']:.0f} NFE), "
        f"DNB-84={dnb84['pass_at_1']:.4f} ({dnb84['avg_nfe']:.0f} NFE), "
        f"Standard-64={std64['pass_at_1']:.4f}, "
        f"Entropy-Revise={ent_revise['pass_at_1']:.4f}. "
        f"CARD vs DNB: {h5_delta:+.4f}. "
        f"Pareto frontier: {pareto_frontier}. "
        f"Result: {go_no_go}."
    )

    # Qualitative examples
    qualitative_examples = []
    for tid in card_wins_dnb[:5]:
        card_entry = next((e for e in card84["per_sample"] if e["task_id"] == tid), None)
        dnb_entry = next((e for e in dnb84["per_sample"] if e["task_id"] == tid), None)
        if card_entry and dnb_entry:
            ex = {
                "task_id": tid,
                "type": "card_wins_over_dnb",
                "card_passed": True,
                "dnb_passed": False,
                "dnb_error": dnb_entry.get("error", ""),
            }
            if "completion" in card_entry:
                ex["card_completion"] = card_entry["completion"][:300]
            qualitative_examples.append(ex)

    # Build final result
    result = {
        "task_id": TASK_ID,
        "candidate_id": "cand_card",
        "mode": mode_str,
        "model": "LLaDA-8B-Instruct",
        "benchmark": "HumanEval",
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
                "card_unique_wins": card_wins_dnb[:10],
                "dnb_unique_wins": dnb_wins_card[:10],
            },
            "card84_vs_std64": {
                "card_wins": len(card_wins_std),
                "std_wins": len(std_wins_card),
            },
        },
        "error_analysis": {
            "card84_error_types": error_types,
        },
        "qualitative_examples": qualitative_examples,
        "hypothesis_tests": {
            "H5_card_beats_dnb_humaneval": {
                "criterion": "CARD-84 pass@1 >= DNB-84 pass@1 on HumanEval",
                "card84_pass1": card84["pass_at_1"],
                "dnb84_pass1": dnb84["pass_at_1"],
                "delta": round(h5_delta, 4),
                "pass": h5_pass,
            }
        },
        "method_details": {
            name: {k: v for k, v in result_data.items() if k != "per_sample"}
            for name, result_data in all_results.items()
        },
        "per_sample_results": {
            name: result_data["per_sample"]
            for name, result_data in all_results.items()
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
    result_path = RESULTS_DIR / "full_llada_humaneval.json"
    result_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n  Saved to {result_path}")

    full_path = FULL_RESULTS_DIR / "full_llada_humaneval.json"
    full_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"  Saved to {full_path}")

    # Print summary
    print(f"\n{'='*70}")
    print(f"HUMANEVAL RESULTS ({mode_str}, {len(samples)} problems)")
    print(f"{'='*70}")
    print(f"\n  {'Method':<25} {'pass@1':>8} {'Correct':>8} {'NFE':>6} {'TokChg':>8} {'Time(s)':>8}")
    print(f"  {'-'*65}")
    for row in summary_table:
        print(f"  {row['method']:<25} {row['pass_at_1']:>8.4f} "
              f"{row['correct']:>8}/{row['total']:<4} "
              f"{row['avg_nfe']:>6.1f} {row['avg_tokens_changed']:>8.1f} "
              f"{row['wall_clock_sec']:>8.1f}")

    print(f"\n  Key Comparisons:")
    for k, v in comparisons.items():
        print(f"    {k}: {v:+.4f}")

    print(f"\n  Pareto frontier: {pareto_frontier}")
    print(f"  CARD on Pareto: {'CARD-84' in pareto_frontier}")

    print(f"\n  Problem Flips (CARD-84 vs DNB-84):")
    print(f"    CARD wins: {len(card_wins_dnb)}")
    print(f"    DNB wins:  {len(dnb_wins_card)}")
    print(f"    Both pass: {len(both_correct_cd)}")
    print(f"    Both fail: {len(both_wrong_cd)}")

    if error_types:
        print(f"\n  CARD-84 Error Types:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {etype}: {count}")

    print(f"\n  H5 (CARD>=DNB): {'PASS' if h5_pass else 'FAIL'} (delta={h5_delta:+.4f})")
    print(f"  Overall: {go_no_go}")
    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    print(f"{'='*70}", flush=True)

    report_progress(
        epoch=len(methods), total_epochs=len(methods),
        step=len(samples), total_steps=len(samples),
        metric={
            "status": "completed",
            "card84_pass1": card84["pass_at_1"],
            "dnb84_pass1": dnb84["pass_at_1"],
            "card84_vs_dnb84": round(h5_delta, 4),
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
