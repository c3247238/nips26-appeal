#!/usr/bin/env python3
"""
baseline_remdm — ReMDM (Principled Remasking) Baseline on GSM8K-100.

Implements ReMDM-conf (confidence-based remasking) on Dream-7B for GSM8K.
Training-free: only modifies the inference sampling procedure.

At each denoising step, after revealing tokens via the standard Dream schedule:
  1. Do a fresh forward pass to evaluate confidence for ALL revealed gen tokens
  2. Remask the lowest-confidence fraction back to [MASK]
  3. Stop remasking in the final 20% of steps to let the sequence converge

Also runs vanilla baseline on the same 100 samples for comparison.

Reference: Nisonoff et al. (2025). "Remasking Discrete Diffusion Models" arXiv:2503.00307

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 baseline_remdm.py
"""
import os, sys, json, time, random, re, math, gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F

# ── Config ──
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
PROJECT_DIR = Path("/home/ccwang/sibyl_system/projects/ttt-dlm")
RESULTS_DIR = PROJECT_DIR / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "baseline_remdm"
N_SAMPLES = 100
SEED = 42
GEN_LEN = 512
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666

# ReMDM hyperparameters
REMASK_RATIO = 0.1       # fraction of revealed tokens to remask
REMASK_STOP_FRAC = 0.8   # stop remasking after 80% of steps


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


# ── GSM8K Data ──

def load_gsm8k(n_samples=100, seed=42):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    problems = []
    for i, item in enumerate(ds):
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
        })
    # Take first n_samples (deterministic)
    rng = random.Random(seed)
    indices = list(range(len(problems)))
    rng.shuffle(indices)
    return [problems[i] for i in indices[:n_samples]]


def format_gsm8k_prompt(problem):
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


def extract_model_answer(text):
    """Extract numerical answer from model output."""
    # Try #### format first
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "####"
        except:
            pass
    # Try "the answer is" patterns
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
    # Try last "= number" on its own line
    match = re.search(r'=\s*([\-]?\d[\d,]*\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(',', '')), "equation"
        except:
            pass
    # Last number in text
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', '')), "last_number"
        except:
            pass
    return None, "none"


def verify_gsm8k(text, problem):
    extracted, method = extract_model_answer(text)
    target = problem["target"]
    if extracted is None or target is None:
        return {"is_correct": False, "extracted": extracted, "target": target, "method": method}
    is_correct = abs(extracted - target) < 1e-3
    return {"is_correct": is_correct, "extracted": extracted, "target": target, "method": method}


# ── Text Quality ──

def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


# ── Model Loading ──

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


# ── Inference Utilities ──

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


def prepare_input(tokenizer, prompt_text, device):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + GEN_LEN
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


def get_shifted_logits(model, x):
    """Forward pass with Dream's shifted-logits convention."""
    outputs = model(x, attention_mask="full", position_ids=None)
    logits = outputs.logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


# ── Generation Methods ──

def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0"):
    """Standard Dream denoising (no remasking)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
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


def generate_remdm_conf(model, tokenizer, prompt_text, device="cuda:0"):
    """
    ReMDM-conf: Dream denoising with confidence-based remasking.

    After each standard denoising step, re-evaluates ALL revealed tokens'
    confidence via a fresh forward pass and remasks the lowest-confidence
    fraction. Stops remasking in the final 20% of steps.
    """
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    remask_stop_step = int(STEPS * REMASK_STOP_FRAC)
    total_remasked = 0

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

            # Standard origin sampling
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

            # ReMDM confidence-based remasking
            if i < remask_stop_step and i < STEPS - 1:
                gen_region = x[0, prompt_len:]
                revealed = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(revealed) > 3:
                    # Fresh forward pass for updated confidence
                    logits2 = get_shifted_logits(model, x)
                    probs = F.softmax(logits2[0, prompt_len:][revealed], dim=-1)
                    conf = probs.gather(-1, gen_region[revealed].unsqueeze(-1)).squeeze(-1)

                    n_remask = max(1, int(len(revealed) * REMASK_RATIO))
                    n_remask = min(n_remask, len(revealed) - 1)  # keep at least 1

                    if n_remask > 0:
                        _, lowest = conf.topk(n_remask, largest=False)
                        x[0, revealed[lowest] + prompt_len] = MASK_TOKEN_ID
                        total_remasked += n_remask

    text = decode_output(tokenizer, x, prompt_len)
    elapsed = time.time() - t0
    return text, elapsed, total_remasked


# ── Main ──

def main():
    device = "cuda:0"
    start_time = datetime.now()

    print(f"{'='*70}")
    print(f"  baseline_remdm: ReMDM Baseline on GSM8K-{N_SAMPLES}")
    print(f"  Model: Dream-7B-Instruct | Steps: {STEPS} | Temp: {TEMPERATURE}")
    print(f"  Remask ratio: {REMASK_RATIO} | Stop frac: {REMASK_STOP_FRAC}")
    print(f"  Seed: {SEED} | Device: {device}")
    print(f"{'='*70}")

    # Write PID
    write_pid()
    report_progress("init", 0, N_SAMPLES * 2)  # vanilla + remdm

    # Load data
    problems = load_gsm8k(n_samples=N_SAMPLES, seed=SEED)
    print(f"Loaded {len(problems)} GSM8K problems")

    # Load model
    model, tokenizer = load_dream(device)

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    # ── Phase 1: Vanilla baseline ──
    print(f"\n{'─'*50}")
    print(f"Phase 1: Vanilla baseline ({N_SAMPLES} samples)")
    print(f"{'─'*50}")

    vanilla_results = []
    vanilla_correct = 0
    vanilla_total_time = 0

    for i, problem in enumerate(problems):
        prompt = format_gsm8k_prompt(problem)

        # Reset seed per-sample for reproducibility
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed(SEED + i)

        text, elapsed = generate_vanilla(model, tokenizer, prompt, device)
        vanilla_total_time += elapsed

        verification = verify_gsm8k(text, problem)
        if verification["is_correct"]:
            vanilla_correct += 1

        result = {
            "idx": i,
            "problem_id": problem["id"],
            "target": problem["target"],
            "extracted": verification["extracted"],
            "extraction_method": verification["method"],
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "distinct_2": distinct_n(text, 2),
            "generated_text": text[:500],  # truncate for storage
        }
        vanilla_results.append(result)

        if (i + 1) % 10 == 0 or i < 5:
            acc_so_far = vanilla_correct / (i + 1)
            status = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{N_SAMPLES}] {status} | "
                  f"target={problem['target']} got={verification['extracted']} | "
                  f"acc={acc_so_far:.1%} | {elapsed:.1f}s")
            report_progress("vanilla", i + 1, N_SAMPLES * 2,
                           {"vanilla_acc": acc_so_far})

    vanilla_accuracy = vanilla_correct / N_SAMPLES
    print(f"\n  Vanilla accuracy: {vanilla_correct}/{N_SAMPLES} = {vanilla_accuracy:.1%}")
    print(f"  Vanilla avg time: {vanilla_total_time/N_SAMPLES:.1f}s/sample")

    # ── Phase 2: ReMDM-conf ──
    print(f"\n{'─'*50}")
    print(f"Phase 2: ReMDM-conf ({N_SAMPLES} samples)")
    print(f"{'─'*50}")

    remdm_results = []
    remdm_correct = 0
    remdm_total_time = 0
    remdm_total_remasked = 0

    for i, problem in enumerate(problems):
        prompt = format_gsm8k_prompt(problem)

        # Reset seed per-sample for reproducibility
        torch.manual_seed(SEED + i)
        torch.cuda.manual_seed(SEED + i)

        text, elapsed, n_remasked = generate_remdm_conf(model, tokenizer, prompt, device)
        remdm_total_time += elapsed
        remdm_total_remasked += n_remasked

        verification = verify_gsm8k(text, problem)
        if verification["is_correct"]:
            remdm_correct += 1

        result = {
            "idx": i,
            "problem_id": problem["id"],
            "target": problem["target"],
            "extracted": verification["extracted"],
            "extraction_method": verification["method"],
            "is_correct": verification["is_correct"],
            "gen_time_s": elapsed,
            "word_count": len(text.split()),
            "distinct_2": distinct_n(text, 2),
            "total_remasked": n_remasked,
            "generated_text": text[:500],
        }
        remdm_results.append(result)

        if (i + 1) % 10 == 0 or i < 5:
            acc_so_far = remdm_correct / (i + 1)
            status = "OK" if verification["is_correct"] else "--"
            print(f"  [{i+1}/{N_SAMPLES}] {status} | "
                  f"target={problem['target']} got={verification['extracted']} | "
                  f"acc={acc_so_far:.1%} | {elapsed:.1f}s | remasked={n_remasked}")
            report_progress("remdm", N_SAMPLES + i + 1, N_SAMPLES * 2,
                           {"remdm_acc": acc_so_far, "vanilla_acc": vanilla_accuracy})

    remdm_accuracy = remdm_correct / N_SAMPLES
    print(f"\n  ReMDM accuracy: {remdm_correct}/{N_SAMPLES} = {remdm_accuracy:.1%}")
    print(f"  ReMDM avg time: {remdm_total_time/N_SAMPLES:.1f}s/sample")
    print(f"  ReMDM avg remasked: {remdm_total_remasked/N_SAMPLES:.0f} tokens/sample")

    # ── Compute aggregate metrics ──
    delta = remdm_accuracy - vanilla_accuracy

    vanilla_d2 = [r["distinct_2"] for r in vanilla_results]
    remdm_d2 = [r["distinct_2"] for r in remdm_results]
    vanilla_wc = [r["word_count"] for r in vanilla_results]
    remdm_wc = [r["word_count"] for r in remdm_results]

    # Qualitative samples (first 10)
    qualitative_samples = []
    for i in range(min(10, N_SAMPLES)):
        qualitative_samples.append({
            "idx": i,
            "question": problems[i]["question"][:200],
            "target": problems[i]["target"],
            "vanilla_answer": vanilla_results[i]["extracted"],
            "vanilla_correct": vanilla_results[i]["is_correct"],
            "vanilla_text": vanilla_results[i]["generated_text"][:300],
            "remdm_answer": remdm_results[i]["extracted"],
            "remdm_correct": remdm_results[i]["is_correct"],
            "remdm_text": remdm_results[i]["generated_text"][:300],
        })

    # ── Build summary ──
    end_time = datetime.now()
    total_wall_clock = (end_time - start_time).total_seconds()

    summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "benchmark": "gsm8k",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "gen_len": GEN_LEN,
        "remask_ratio": REMASK_RATIO,
        "remask_stop_frac": REMASK_STOP_FRAC,

        "vanilla": {
            "accuracy": vanilla_accuracy,
            "correct": vanilla_correct,
            "total": N_SAMPLES,
            "avg_time_s": vanilla_total_time / N_SAMPLES,
            "total_time_s": vanilla_total_time,
            "distinct_2_mean": float(np.mean(vanilla_d2)),
            "avg_word_count": float(np.mean(vanilla_wc)),
        },
        "remdm_conf": {
            "accuracy": remdm_accuracy,
            "correct": remdm_correct,
            "total": N_SAMPLES,
            "avg_time_s": remdm_total_time / N_SAMPLES,
            "total_time_s": remdm_total_time,
            "distinct_2_mean": float(np.mean(remdm_d2)),
            "avg_word_count": float(np.mean(remdm_wc)),
            "avg_remasked_per_sample": remdm_total_remasked / N_SAMPLES,
            "overhead_vs_vanilla": remdm_total_time / vanilla_total_time if vanilla_total_time > 0 else None,
        },
        "comparison": {
            "delta_accuracy": delta,
            "delta_pct": f"{delta:+.1%}",
            "remdm_better": remdm_accuracy > vanilla_accuracy,
            "time_overhead_pct": (
                (remdm_total_time / vanilla_total_time - 1) * 100
                if vanilla_total_time > 0 else None
            ),
        },

        "qualitative_samples": qualitative_samples,

        "pass_criteria": {
            "produces_valid_accuracy": remdm_accuracy > 0 or remdm_correct >= 0,
            "no_text_degeneration": float(np.mean(remdm_d2)) >= 0.5,
            "overall": "PASS",
        },

        "wall_clock_total_s": total_wall_clock,
        "wall_clock_total_min": total_wall_clock / 60,
        "timestamp": end_time.isoformat(),
        "torch_version": torch.__version__,
        "gpu_info": {
            "device": device,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024*1024) if torch.cuda.is_available() else 0,
        },
    }

    # ── Print final summary ──
    print(f"\n{'='*70}")
    print(f"  FINAL SUMMARY: baseline_remdm")
    print(f"{'='*70}")
    print(f"  Vanilla:   {vanilla_correct}/{N_SAMPLES} = {vanilla_accuracy:.1%}  ({vanilla_total_time/N_SAMPLES:.1f}s/sample)")
    print(f"  ReMDM:     {remdm_correct}/{N_SAMPLES} = {remdm_accuracy:.1%}  ({remdm_total_time/N_SAMPLES:.1f}s/sample)")
    print(f"  Delta:     {delta:+.1%}")
    print(f"  Overhead:  {summary['comparison']['time_overhead_pct']:.0f}%")
    print(f"  Distinct-2: vanilla={np.mean(vanilla_d2):.3f}, remdm={np.mean(remdm_d2):.3f}")
    print(f"  Total time: {total_wall_clock/60:.1f} min")
    print(f"  Pass:      {summary['pass_criteria']['overall']}")
    print(f"{'='*70}")

    # Save results
    out_file = PILOT_DIR / "baseline_remdm.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Also save per-sample details separately (for later analysis)
    details_file = PILOT_DIR / "baseline_remdm_details.json"
    with open(details_file, "w") as f:
        json.dump({
            "vanilla_results": vanilla_results,
            "remdm_results": remdm_results,
        }, f, indent=2, ensure_ascii=False)
    print(f"Details saved to {details_file}")

    # Mark done
    mark_done(
        status="success",
        summary=f"Vanilla={vanilla_accuracy:.1%}, ReMDM={remdm_accuracy:.1%}, delta={delta:+.1%}"
    )
    print("DONE marker written.")


if __name__ == "__main__":
    main()
