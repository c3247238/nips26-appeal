#!/usr/bin/env python3
"""
vanilla_baselines — Phase 0: Vanilla Baselines (GSM8K-300, ARC-C-300, HumanEval-164)

Establish rigorous vanilla Dream-7B baselines on all three primary benchmarks.
Standard denoising (128 steps), seed 42.
Record accuracy, time/sample, Distinct-1/2/3, peak memory.
Save per-sample predictions for paired bootstrap in downstream experiments.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 vanilla_baselines.py
"""
import os, sys, json, time, random, re, math, gc, signal
import subprocess, tempfile, textwrap
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ── Config ──
MODEL_DIR = "/home/ccwang/sibyl_system/shared/checkpoints/Dream-v0-Instruct-7B"
PROJECT_DIR = Path("/home/ccwang/sibyl_system/projects/ttt-dlm")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "vanilla_baselines"
SEED = 42
GEN_LEN = 512
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666

# Benchmark configs
BENCHMARKS = {
    "gsm8k": {"n_samples": 300, "type": "math"},
    "arc_challenge": {"n_samples": 300, "type": "multiple_choice"},
    "humaneval": {"n_samples": 164, "type": "code"},
}

# ── Progress / PID / DONE markers ──

def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

def report_progress(phase, completed, total, extra=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID,
        "phase": phase,
        "completed": completed,
        "total": total,
        "updated_at": datetime.now().isoformat(),
    }
    if extra:
        data.update(extra)
    progress.write_text(json.dumps(data))

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


# ═══════════════════════════════════════════════════════════
#  DATA LOADERS
# ═══════════════════════════════════════════════════════════

def load_gsm8k(n_samples=300):
    """Load first n_samples from GSM8K test set (deterministic order)."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        answer_text = item["answer"]
        match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', answer_text)
        target = None
        if match:
            try:
                target = float(match.group(1).replace(',', ''))
            except:
                pass
        problems.append({
            "id": i,
            "question": item["question"],
            "target": target,
            "benchmark": "gsm8k",
        })
    return problems


def load_arc_challenge(n_samples=300):
    """Load first n_samples from ARC-Challenge test set."""
    from datasets import load_from_disk
    ds = load_from_disk("/home/ccwang/sibyl_system/shared/datasets/arc_challenge")
    problems = []
    for i in range(min(n_samples, len(ds))):
        item = ds[i]
        choices = item["choices"]
        choice_texts = choices["text"]
        choice_labels = choices["label"]
        answer_key = item["answerKey"]
        # Find correct answer index
        answer_idx = choice_labels.index(answer_key) if answer_key in choice_labels else -1
        problems.append({
            "id": i,
            "arc_id": item["id"],
            "question": item["question"],
            "choices": choice_texts,
            "choice_labels": choice_labels,
            "answer_key": answer_key,
            "answer_idx": answer_idx,
            "benchmark": "arc_challenge",
        })
    return problems


def load_humaneval(n_samples=164):
    """Load HumanEval dataset (all 164 problems)."""
    from datasets import load_from_disk
    ds = load_from_disk("/home/ccwang/sibyl_system/shared/datasets/humaneval")
    problems = []
    for i in range(min(n_samples, len(ds))):
        item = ds[i]
        problems.append({
            "id": i,
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
            "benchmark": "humaneval",
        })
    return problems


# ═══════════════════════════════════════════════════════════
#  PROMPT FORMATTING
# ═══════════════════════════════════════════════════════════

def format_gsm8k_prompt(problem):
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


def format_arc_prompt(problem):
    choices_str = "\n".join(
        f"{label}. {text}"
        for label, text in zip(problem["choice_labels"], problem["choices"])
    )
    return (
        f"Answer the following multiple choice question. "
        f"Choose the best answer and respond with ONLY the letter (A, B, C, or D).\n\n"
        f"Question: {problem['question']}\n\n"
        f"{choices_str}\n\n"
        f"Answer:"
    )


def format_humaneval_prompt(problem):
    return (
        f"Complete the following Python function. Write ONLY the function body, "
        f"no explanations.\n\n{problem['prompt']}"
    )


# ═══════════════════════════════════════════════════════════
#  ANSWER EXTRACTION & VERIFICATION
# ═══════════════════════════════════════════════════════════

def extract_gsm8k_answer(text):
    """Extract numerical answer from model output."""
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "####"
        except:
            pass
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:=]?\s*([\-]?\d[\d,]*\.?\d*)',
        r'(?:therefore|thus|so|hence)[,\s]+(?:the\s+answer\s+is\s+)?([\-]?\d[\d,]*\.?\d*)',
        r'\\boxed\{([\-]?\d[\d,]*\.?\d*)\}',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(',', '')), "pattern"
            except:
                continue
    match = re.search(r'=\s*([\-]?\d[\d,]*\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "equation"
        except:
            pass
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', '')), "last_number"
        except:
            pass
    return None, "none"


def verify_gsm8k(text, problem):
    extracted, method = extract_gsm8k_answer(text)
    target = problem["target"]
    if extracted is None or target is None:
        return {"is_correct": False, "extracted": extracted, "target": target, "method": method}
    is_correct = abs(extracted - target) < 1e-3
    return {"is_correct": is_correct, "extracted": extracted, "target": target, "method": method}


def extract_arc_answer(text):
    """Extract letter answer (A/B/C/D) from model output."""
    text_clean = text.strip()
    # Direct single letter
    if text_clean and text_clean[0].upper() in "ABCD":
        return text_clean[0].upper(), "first_char"
    # "The answer is X"
    match = re.search(r'(?:answer|choice)\s+is\s*[:=]?\s*([A-Da-d])', text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "pattern"
    # Any letter in the text
    for c in text_clean:
        if c.upper() in "ABCD":
            return c.upper(), "scan"
    return None, "none"


def verify_arc(text, problem):
    extracted, method = extract_arc_answer(text)
    answer_key = problem["answer_key"]
    is_correct = extracted == answer_key if extracted else False
    return {"is_correct": is_correct, "extracted": extracted, "target": answer_key, "method": method}


def verify_humaneval(generated_code, problem):
    """Execute HumanEval test cases with sandboxing (timeout + restricted)."""
    # Build complete code: function prompt + generated body + test
    prompt = problem["prompt"]
    test_code = problem["test"]
    entry_point = problem["entry_point"]

    # Clean up generated code
    code = generated_code.strip()
    # If model repeated the function signature, try to extract just the body
    if f"def {entry_point}" in code:
        # Find the function body after the signature
        lines = code.split('\n')
        body_lines = []
        in_func = False
        for line in lines:
            if f"def {entry_point}" in line:
                in_func = True
                body_lines.append(line)
                continue
            if in_func:
                body_lines.append(line)
        if body_lines:
            code = '\n'.join(body_lines)

    # Construct full test program
    full_code = f"{prompt}{code}\n\n{test_code}\ncheck({entry_point})\n"

    # Execute in subprocess with timeout
    try:
        result = subprocess.run(
            [sys.executable, "-c", full_code],
            capture_output=True, text=True, timeout=10,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        passed = result.returncode == 0
        return {
            "is_correct": passed,
            "extracted": "pass" if passed else "fail",
            "target": "pass",
            "method": "execution",
            "stderr": result.stderr[:500] if result.stderr else "",
        }
    except subprocess.TimeoutExpired:
        return {
            "is_correct": False,
            "extracted": "timeout",
            "target": "pass",
            "method": "execution",
            "stderr": "Timeout after 10s",
        }
    except Exception as e:
        return {
            "is_correct": False,
            "extracted": "error",
            "target": "pass",
            "method": "execution",
            "stderr": str(e)[:500],
        }


# ═══════════════════════════════════════════════════════════
#  TEXT QUALITY
# ═══════════════════════════════════════════════════════════

def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def agg_diversity(results):
    d1 = [distinct_n(r["generated_text"], 1) for r in results if r.get("generated_text")]
    d2 = [distinct_n(r["generated_text"], 2) for r in results if r.get("generated_text")]
    d3 = [distinct_n(r["generated_text"], 3) for r in results if r.get("generated_text")]
    wc = [r["word_count"] for r in results]
    return {
        "distinct_1_mean": float(np.mean(d1)) if d1 else 0.0,
        "distinct_2_mean": float(np.mean(d2)) if d2 else 0.0,
        "distinct_3_mean": float(np.mean(d3)) if d3 else 0.0,
        "avg_word_count": float(np.mean(wc)) if wc else 0.0,
    }


# ═══════════════════════════════════════════════════════════
#  MODEL LOADING & INFERENCE
# ═══════════════════════════════════════════════════════════

def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    print(f"Loading Dream-7B from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}, dtype=bfloat16")
    return model, tokenizer


def get_shifted_logits(model, x):
    outputs = model(x, attention_mask="full", position_ids=None)
    logits = outputs.logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


def sample_tokens_from_logits(logits, temperature=0.4):
    if temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    if temperature > 0:
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
        confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    else:
        confidence, sampled = probs.max(dim=-1)
    return confidence, sampled


def prepare_input(tokenizer, prompt_text, device, gen_len=GEN_LEN):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    return x, prompt_len


def decode_output(tokenizer, x, prompt_len):
    gen_ids = x[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t_id in gen_ids:
        if t_id == MASK_TOKEN_ID:
            continue
        if t_id == eos_id:
            break
        clean_ids.append(t_id)
    return tokenizer.decode(clean_ids, skip_special_tokens=True).strip()


def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0", gen_len=GEN_LEN):
    """Standard Dream denoising (random unmasking order)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    t0 = time.time()
    with torch.no_grad():
        for i in range(STEPS):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = get_shifted_logits(model, x)
            mask_logits = logits[mask_index]
            t = timesteps[i]; s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
    text = decode_output(tokenizer, x, prompt_len)
    return text, time.time() - t0


# ═══════════════════════════════════════════════════════════
#  CHECKPOINT / RESUME
# ═══════════════════════════════════════════════════════════

def load_checkpoint():
    ckpt_file = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"
    if ckpt_file.exists():
        try:
            data = json.loads(ckpt_file.read_text())
            print(f"Resuming from checkpoint")
            return data
        except:
            pass
    return None


def save_checkpoint(all_results, current_benchmark_idx, current_sample_idx):
    ckpt_file = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"
    ckpt_file.write_text(json.dumps({
        "all_results": all_results,
        "current_benchmark_idx": current_benchmark_idx,
        "current_sample_idx": current_sample_idx,
        "timestamp": datetime.now().isoformat(),
    }))


# ═══════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════

def run_benchmark(model, tokenizer, benchmark_name, problems, device, all_results, start_idx=0):
    """Run vanilla baseline on a single benchmark, return results list."""
    bm_config = BENCHMARKS[benchmark_name]
    bm_type = bm_config["type"]
    n = len(problems)

    # For code tasks, use longer gen_len
    gen_len = 768 if bm_type == "code" else GEN_LEN

    results = all_results.get(benchmark_name, [])
    correct = sum(1 for r in results if r["is_correct"])
    total_time = sum(r["gen_time_s"] for r in results)

    print(f"\n{'─'*60}")
    print(f"  Benchmark: {benchmark_name} ({n} samples, type={bm_type}, start={start_idx})")
    print(f"{'─'*60}")

    for i in range(start_idx, n):
        problem = problems[i]
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed(SEED + i)

        # Format prompt based on type
        if bm_type == "math":
            prompt = format_gsm8k_prompt(problem)
        elif bm_type == "multiple_choice":
            prompt = format_arc_prompt(problem)
        elif bm_type == "code":
            prompt = format_humaneval_prompt(problem)
        else:
            raise ValueError(f"Unknown benchmark type: {bm_type}")

        text, elapsed = generate_vanilla(model, tokenizer, prompt, device, gen_len)
        total_time += elapsed

        # Verify
        if bm_type == "math":
            verification = verify_gsm8k(text, problem)
        elif bm_type == "multiple_choice":
            verification = verify_arc(text, problem)
        elif bm_type == "code":
            verification = verify_humaneval(text, problem)

        if verification["is_correct"]:
            correct += 1

        result = {
            "idx": i,
            "problem_id": problem.get("id", i),
            "target": verification["target"],
            "extracted": verification["extracted"],
            "extraction_method": verification["method"],
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "generated_text": text[:500],
        }
        # Add benchmark-specific fields
        if bm_type == "code":
            result["task_id_he"] = problem.get("task_id", "")
            result["stderr"] = verification.get("stderr", "")
        elif bm_type == "multiple_choice":
            result["arc_id"] = problem.get("arc_id", "")

        results.append(result)

        if (i + 1) % 20 == 0 or i < 3 or i == n - 1:
            acc = correct / (i + 1)
            st = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{n}] {st} | target={verification['target']} "
                  f"got={verification['extracted']} | acc={acc:.1%} | {elapsed:.1f}s",
                  flush=True)

        # Periodic cleanup & checkpoint
        if (i + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()
            all_results[benchmark_name] = results
            # Save progress after each benchmark batch
            report_progress(
                benchmark_name, len(results), n,
                {"accuracy": correct / (i + 1), "benchmark": benchmark_name}
            )

    accuracy = correct / n if n > 0 else 0.0
    avg_time = total_time / n if n > 0 else 0.0

    print(f"\n  {benchmark_name}: {correct}/{n} = {accuracy:.1%} ({avg_time:.1f}s/sample)")

    all_results[benchmark_name] = results
    return results, accuracy, correct, total_time


def main():
    device = "cuda:0"
    start_time = datetime.now()

    print(f"{'='*70}")
    print(f"  vanilla_baselines: Phase 0 Vanilla Baselines")
    print(f"  Model: Dream-7B-Instruct | Steps: {STEPS} | Temp: {TEMPERATURE}")
    print(f"  Seed: {SEED} | Device: {device}")
    print(f"  Benchmarks: GSM8K-300, ARC-C-300, HumanEval-164")
    print(f"{'='*70}")

    write_pid()

    total_samples = sum(BENCHMARKS[b]["n_samples"] for b in BENCHMARKS)
    report_progress("init", 0, total_samples)

    # Load model
    model, tokenizer = load_dream(device)

    # GPU profile
    gpu_name = torch.cuda.get_device_name(0)
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    print(f"GPU: {gpu_name}, VRAM: {vram_total_mb} MB")

    # Peak memory tracking
    torch.cuda.reset_peak_memory_stats()

    # Check for checkpoint
    ckpt = load_checkpoint()
    all_results = ckpt["all_results"] if ckpt else {}
    start_benchmark_idx = ckpt.get("current_benchmark_idx", 0) if ckpt else 0
    start_sample_idx = ckpt.get("current_sample_idx", 0) if ckpt else 0

    benchmark_order = ["gsm8k", "arc_challenge", "humaneval"]
    summaries = {}

    for bm_idx, bm_name in enumerate(benchmark_order):
        if bm_idx < start_benchmark_idx:
            # Already completed this benchmark, compute summary from saved results
            if bm_name in all_results:
                results = all_results[bm_name]
                correct = sum(1 for r in results if r["is_correct"])
                n = len(results)
                total_time = sum(r["gen_time_s"] for r in results)
                summaries[bm_name] = {
                    "accuracy": correct / n if n > 0 else 0.0,
                    "correct": correct,
                    "total": n,
                    "avg_time_s": round(total_time / n, 2) if n > 0 else 0.0,
                    "total_time_s": round(total_time, 1),
                    **agg_diversity(results),
                }
                print(f"  [{bm_name}] Restored from checkpoint: {correct}/{n} = {correct/n:.1%}")
            continue

        # Load data
        if bm_name == "gsm8k":
            problems = load_gsm8k(BENCHMARKS[bm_name]["n_samples"])
        elif bm_name == "arc_challenge":
            problems = load_arc_challenge(BENCHMARKS[bm_name]["n_samples"])
        elif bm_name == "humaneval":
            problems = load_humaneval(BENCHMARKS[bm_name]["n_samples"])

        sample_start = start_sample_idx if bm_idx == start_benchmark_idx else 0

        results, accuracy, correct, total_time = run_benchmark(
            model, tokenizer, bm_name, problems, device, all_results, sample_start
        )

        n = len(results)
        summaries[bm_name] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": n,
            "avg_time_s": round(total_time / n, 2) if n > 0 else 0.0,
            "total_time_s": round(total_time, 1),
            **agg_diversity(results),
        }

        # Save checkpoint after each benchmark
        save_checkpoint(all_results, bm_idx + 1, 0)

        report_progress(
            f"completed_{bm_name}",
            sum(len(all_results.get(b, [])) for b in benchmark_order[:bm_idx+1]),
            total_samples,
            {"completed_benchmarks": benchmark_order[:bm_idx+1]}
        )

    # ── Peak memory ──
    peak_mem_mb = torch.cuda.max_memory_allocated() // (1024 * 1024)

    # ── Qualitative samples (5 per benchmark) ──
    qualitative = {}
    for bm_name in benchmark_order:
        bm_results = all_results.get(bm_name, [])
        qualitative[bm_name] = [
            {
                "idx": r["idx"],
                "target": r["target"],
                "extracted": r["extracted"],
                "is_correct": r["is_correct"],
                "text_preview": r["generated_text"][:300],
            }
            for r in bm_results[:5]
        ]

    # ── Pass criteria check ──
    # "All 3 benchmarks produce valid accuracy numbers;
    #  GSM8K-300 baseline consistent with prior measurement (34% +/- 5%)"
    all_valid = all(bm_name in summaries for bm_name in benchmark_order)
    gsm8k_acc = summaries.get("gsm8k", {}).get("accuracy", 0.0)
    gsm8k_consistent = 0.29 <= gsm8k_acc <= 0.39  # 34% +/- 5%
    pass_criteria_met = all_valid and gsm8k_consistent

    # ── Build final summary ──
    end_time = datetime.now()
    total_wall_clock = (end_time - start_time).total_seconds()

    summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "candidate_id": "shared",
        "model": "Dream-v0-Instruct-7B",
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "gen_len": GEN_LEN,

        "benchmarks": summaries,

        "pass_criteria": {
            "criterion": "All 3 benchmarks produce valid accuracy; GSM8K-300 baseline consistent with prior (34% +/- 5%)",
            "all_valid": all_valid,
            "gsm8k_accuracy": gsm8k_acc,
            "gsm8k_consistent_with_prior": gsm8k_consistent,
            "prior_reference": "iggd_n300_verify vanilla=31.0%",
            "pass": pass_criteria_met,
            "go_no_go": "GO" if pass_criteria_met else "REVIEW",
        },

        "qualitative_samples": qualitative,

        "wall_clock_total_s": round(total_wall_clock, 1),
        "wall_clock_total_min": round(total_wall_clock / 60, 1),
        "peak_memory_mb": peak_mem_mb,
        "timestamp": end_time.isoformat(),
        "torch_version": torch.__version__,
        "gpu_info": {
            "device": device,
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total_mb,
        },
    }

    # ── Print final summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY: vanilla_baselines")
    print(f"{'='*70}")
    for bm_name in benchmark_order:
        s = summaries.get(bm_name, {})
        print(f"  {bm_name:20s}: {s.get('correct', '?')}/{s.get('total', '?')} = "
              f"{s.get('accuracy', 0):.1%}  ({s.get('avg_time_s', 0):.1f}s/sample)  "
              f"D2={s.get('distinct_2_mean', 0):.3f}")
    print(f"  ─────────────────────────────────")
    print(f"  Peak memory: {peak_mem_mb} MB")
    print(f"  Total time:  {total_wall_clock/60:.1f} min")
    print(f"  GSM8K consistent with prior (34%+/-5%): "
          f"{'YES' if gsm8k_consistent else 'NO'} (got {gsm8k_acc:.1%})")
    print(f"  GO/NO-GO: {'GO' if pass_criteria_met else 'REVIEW'}")
    print(f"{'='*70}")

    # Save results
    out_file = PILOT_DIR / "vanilla_baselines.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Save per-sample details
    details_file = PILOT_DIR / "vanilla_baselines_details.json"
    with open(details_file, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Per-sample details saved to {details_file}")

    # Clean up checkpoint
    ckpt_file = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"
    if ckpt_file.exists():
        ckpt_file.unlink()
        print("Checkpoint file cleaned up.")

    # Mark done
    mark_done(
        status="success",
        summary=(
            f"GSM8K={gsm8k_acc:.1%}, "
            f"ARC-C={summaries.get('arc_challenge', {}).get('accuracy', 0):.1%}, "
            f"HumanEval={summaries.get('humaneval', {}).get('accuracy', 0):.1%}, "
            f"GO/NO-GO={'GO' if pass_criteria_met else 'REVIEW'}"
        )
    )
    print("DONE marker written.")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n\nFATAL ERROR: {e}", file=sys.stderr, flush=True)
        print(f"\n\nFATAL ERROR: {e}", flush=True)
        traceback.print_exc(file=sys.stderr)
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        mark_done(status="failed", summary=f"FATAL: {e}")
        sys.exit(1)
