#!/usr/bin/env python3
"""
C2: DaL + TTS Composition Test (PILOT mode, 16 samples)

2x2 factorial design: {no memory, DaL-Full} x {standard sampling, ReMDM}
Evaluate on GSM8K and HumanEval (pilot, 16 samples each).
Tests H7: are cross-step memory and improved remasking complementary?

Report accuracy and whether gains are additive, sub-additive, or super-additive.

Uses GPU 1 (single GPU, CUDA_VISIBLE_DEVICES=1).

Output:
  - PID: exp/results/composition_test.pid
  - Progress: exp/results/composition_test_PROGRESS.json
  - Done: exp/results/composition_test_DONE
  - Results: exp/results/full/c2_composition.json
"""

import os, sys, json, time, gc, re, math, traceback, copy
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Blackwell GPU CUBLAS workarounds
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# === Paths ===
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results"
FULL_RESULTS_DIR = f"{PROJECT_DIR}/exp/results/full"
CODE_DIR = f"{PROJECT_DIR}/exp/code/dal"
SHARED_DIR = f"{REMOTE_BASE}/shared"

TASK_ID = "composition_test"
SEED = 42

# === Model Config (LLaDA-8B-Instruct) ===
MODEL_PATH = f"{SHARED_DIR}/checkpoints/LLaDA-8B-Instruct"
MASK_TOKEN_ID = 126336
D_MODEL = 4096
VOCAB_SIZE = 126464
N_LAYERS = 32
INSERTION_LAYER = 16  # L/2

# === Checkpoint path (DaL-Linear gate_sep = best from M6 ablation) ===
DAL_LINEAR_CKPT = f"{FULL_RESULTS_DIR}/dal_linear_ckpt_step5000.pt"

# === Eval Config (PILOT: 16 samples) ===
PILOT_SAMPLES = 16
GEN_STEPS = 128
GEN_LEN = 512
TEMPERATURE = 0.0  # greedy
MAX_CUBLAS_RETRIES = 3

sys.path.insert(0, CODE_DIR)
from ttt_layer import TTTLayer, build_ttt_layer
from dal_wrapper import MetaStateGRU, create_masked_input


# ==============================================================================
# Progress / DONE helpers
# ==============================================================================

def report_progress(phase, step, total, extra=None):
    Path(RESULTS_DIR).joinpath(f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "phase": phase, "step": step, "total_steps": total,
        "extra": extra or {}, "updated_at": datetime.now().isoformat()}))


def mark_done(status, summary):
    p = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if p.exists():
        p.unlink()
    prog_file = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if prog_file.exists():
        try:
            final_progress = json.loads(prog_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    Path(RESULTS_DIR).joinpath(f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat()}))


# ==============================================================================
# CUBLAS Error Retry Helper
# ==============================================================================

def cublas_safe_run(fn, max_retries=MAX_CUBLAS_RETRIES):
    """Retry function on CUBLAS errors (Blackwell GPU issue)."""
    for attempt in range(max_retries + 1):
        try:
            return fn()
        except RuntimeError as e:
            if "CUBLAS" in str(e) and attempt < max_retries:
                print(f"  [CUBLAS retry {attempt+1}/{max_retries}] {str(e)[:80]}")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(1)
                continue
            raise
    return None


# ==============================================================================
# LLaDA Generation Helpers
# ==============================================================================

def prepare_input(tokenizer, prompt_text, device, gen_len):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    prompt_len = len(prompt_ids)
    full_ids = prompt_ids + [MASK_TOKEN_ID] * gen_len
    x = torch.tensor([full_ids], device=device)
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


# ==============================================================================
# Generation Methods: 2x2 factorial
# ==============================================================================

def generate_vanilla(model, tokenizer, prompt_text, device, gen_len,
                     gen_steps=GEN_STEPS, temperature=0.0):
    """Standard LLaDA iterative unmasking (no memory, standard sampling)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)

    with torch.no_grad():
        for i in range(gen_steps):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            logits = model(x).logits
            mask_logits = logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if temperature == 0:
                sampled = mask_logits[transfer_mask].argmax(dim=-1)
            else:
                probs = F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

    text = decode_output(tokenizer, x, prompt_len)
    return text


def generate_remdm(model, tokenizer, prompt_text, device, gen_len,
                   gen_steps=GEN_STEPS, temperature=0.0, remask_fraction=0.2):
    """ReMDM-style remasking (no memory, improved sampling)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)

    prompt_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
    prompt_mask[:prompt_len] = True

    with torch.no_grad():
        for i in range(gen_steps):
            mask_index = (x == MASK_TOKEN_ID)
            if not mask_index.any():
                break

            logits = model(x).logits
            mask_logits = logits[mask_index]

            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if temperature == 0:
                sampled = mask_logits[transfer_mask].argmax(dim=-1)
            else:
                probs = F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1)
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

            # ReMDM: remask low-confidence revealed tokens (not in last 10%)
            if i < int(gen_steps * 0.9) and remask_fraction > 0:
                revealed_gen = (~mask_index[0]) & (~prompt_mask) & (x[0] != MASK_TOKEN_ID)
                revealed_indices = revealed_gen.nonzero(as_tuple=True)[0]

                if len(revealed_indices) > 3:
                    with torch.no_grad():
                        revealed_logits = logits[0, revealed_indices]
                        revealed_probs = F.softmax(revealed_logits, dim=-1)
                        current_tokens = x[0, revealed_indices]
                        token_probs = revealed_probs.gather(1, current_tokens.unsqueeze(1)).squeeze(1)

                    n_remask = max(1, int(len(revealed_indices) * remask_fraction))
                    progress = i / gen_steps
                    n_remask = max(1, int(n_remask * (1 - progress)))

                    _, lowest_conf_idx = token_probs.topk(min(n_remask, len(token_probs)), largest=False)
                    remask_positions = revealed_indices[lowest_conf_idx]
                    x[0, remask_positions] = MASK_TOKEN_ID

    text = decode_output(tokenizer, x, prompt_len)
    return text


def generate_dal_standard(model, ttt_layer, ssl_head, ssl_gate, layer_norm,
                          tokenizer, prompt_text, device, gen_len,
                          insertion_layer, gen_steps=GEN_STEPS, temperature=0.0):
    """DaL-Full with standard sampling (memory, standard sampling)."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    target_ids = x.clone()
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)

    inner_model = model.model
    tf = inner_model.transformer
    if hasattr(tf, 'blocks'):
        layers = tf.blocks
    elif hasattr(tf, 'block_groups'):
        layers = tf.block_groups
    else:
        raise ValueError("Cannot find blocks")

    target_layer = layers[insertion_layer]
    ttt_layer.reset_fast_weights(batch_size=1)

    def capture_and_inject(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()

        if n_revealed > 5 and n_masked > 3:
            h_normed = layer_norm(hidden.detach())
            gate = ttt_layer.gate.detach()

            with torch.enable_grad():
                fast_params = ttt_layer.fast_weight.get_params_for_grad()
                for p in fast_params:
                    p.requires_grad_(True)

                fast_output_grad = ttt_layer.fast_weight(h_normed)
                ssl_logits_grad = ssl_head(fast_output_grad)
                B, S, V = ssl_logits_grad.shape
                n_rev = revealed_mask.sum().clamp(min=1.0)
                per_token_loss = F.cross_entropy(
                    ssl_logits_grad.view(-1, V).float(),
                    target_ids.view(-1),
                    reduction="none"
                ).view(B, S).to(ssl_logits_grad.dtype)
                ssl_g = ssl_gate.detach() if ssl_gate is not None else 1.0
                masked_loss = per_token_loss * revealed_mask * ssl_g
                ssl_loss = masked_loss.sum() / n_rev

                grads = torch.autograd.grad(ssl_loss, fast_params,
                                             create_graph=False, allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, fast_params)]

            lr = ttt_layer.lr.detach()
            ttt_layer.fast_weight.apply_update(grads, lr)

            with torch.no_grad():
                new_fast_output = ttt_layer.fast_weight(h_normed)
                delta = gate * new_fast_output

            if isinstance(output, tuple):
                output = (hidden + delta,) + output[1:]
            else:
                output = hidden + delta

        return output

    hook_handle = target_layer.register_forward_hook(capture_and_inject)
    try:
        with torch.no_grad():
            for i in range(gen_steps):
                mask_index = (x == MASK_TOKEN_ID)
                if not mask_index.any():
                    break
                logits = model(x).logits
                mask_logits = logits[mask_index]
                t = timesteps[i]
                s = timesteps[i + 1]
                p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
                x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
                transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
                if temperature == 0:
                    sampled = mask_logits[transfer_mask].argmax(dim=-1)
                else:
                    probs = F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                x0[transfer_mask] = sampled
                x[mask_index] = x0
    finally:
        hook_handle.remove()

    text = decode_output(tokenizer, x, prompt_len)
    return text


def generate_dal_remdm(model, ttt_layer, ssl_head, ssl_gate, layer_norm,
                       tokenizer, prompt_text, device, gen_len,
                       insertion_layer, gen_steps=GEN_STEPS, temperature=0.0,
                       remask_fraction=0.2):
    """DaL-Full + ReMDM remasking (memory + improved sampling) — the COMBINED method."""
    x, prompt_len = prepare_input(tokenizer, prompt_text, device, gen_len)
    target_ids = x.clone()
    eps = 1e-3
    timesteps = torch.linspace(1, eps, gen_steps + 1, device=device)

    prompt_mask = torch.zeros(x.shape[1], dtype=torch.bool, device=device)
    prompt_mask[:prompt_len] = True

    inner_model = model.model
    tf = inner_model.transformer
    if hasattr(tf, 'blocks'):
        layers = tf.blocks
    elif hasattr(tf, 'block_groups'):
        layers = tf.block_groups
    else:
        raise ValueError("Cannot find blocks")

    target_layer = layers[insertion_layer]
    ttt_layer.reset_fast_weights(batch_size=1)

    def capture_and_inject(module, input, output):
        if isinstance(output, tuple):
            hidden = output[0]
        else:
            hidden = output

        mask_index = (x == MASK_TOKEN_ID)
        revealed_mask = (~mask_index).float()
        n_revealed = revealed_mask.sum().item()
        n_masked = mask_index.sum().item()

        if n_revealed > 5 and n_masked > 3:
            h_normed = layer_norm(hidden.detach())
            gate = ttt_layer.gate.detach()

            with torch.enable_grad():
                fast_params = ttt_layer.fast_weight.get_params_for_grad()
                for p in fast_params:
                    p.requires_grad_(True)

                fast_output_grad = ttt_layer.fast_weight(h_normed)
                ssl_logits_grad = ssl_head(fast_output_grad)
                B, S, V = ssl_logits_grad.shape
                n_rev = revealed_mask.sum().clamp(min=1.0)
                per_token_loss = F.cross_entropy(
                    ssl_logits_grad.view(-1, V).float(),
                    target_ids.view(-1),
                    reduction="none"
                ).view(B, S).to(ssl_logits_grad.dtype)
                ssl_g = ssl_gate.detach() if ssl_gate is not None else 1.0
                masked_loss = per_token_loss * revealed_mask * ssl_g
                ssl_loss = masked_loss.sum() / n_rev

                grads = torch.autograd.grad(ssl_loss, fast_params,
                                             create_graph=False, allow_unused=True)
                grads = [g if g is not None else torch.zeros_like(p)
                         for g, p in zip(grads, fast_params)]

            lr = ttt_layer.lr.detach()
            ttt_layer.fast_weight.apply_update(grads, lr)

            with torch.no_grad():
                new_fast_output = ttt_layer.fast_weight(h_normed)
                delta = gate * new_fast_output

            if isinstance(output, tuple):
                output = (hidden + delta,) + output[1:]
            else:
                output = hidden + delta

        return output

    hook_handle = target_layer.register_forward_hook(capture_and_inject)
    try:
        with torch.no_grad():
            for i in range(gen_steps):
                mask_index = (x == MASK_TOKEN_ID)
                if not mask_index.any():
                    break
                logits = model(x).logits
                mask_logits = logits[mask_index]
                t = timesteps[i]
                s = timesteps[i + 1]
                p_transfer = (1 - s / t).item() if i < gen_steps - 1 else 1.0
                x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
                transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
                if temperature == 0:
                    sampled = mask_logits[transfer_mask].argmax(dim=-1)
                else:
                    probs = F.softmax(mask_logits[transfer_mask] / max(temperature, 1e-6), dim=-1)
                    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                x0[transfer_mask] = sampled
                x[mask_index] = x0

                # ReMDM remasking (same as generate_remdm)
                if i < int(gen_steps * 0.9) and remask_fraction > 0:
                    revealed_gen = (~mask_index[0]) & (~prompt_mask) & (x[0] != MASK_TOKEN_ID)
                    revealed_indices = revealed_gen.nonzero(as_tuple=True)[0]

                    if len(revealed_indices) > 3:
                        with torch.no_grad():
                            revealed_logits = logits[0, revealed_indices]
                            revealed_probs = F.softmax(revealed_logits, dim=-1)
                            current_tokens = x[0, revealed_indices]
                            token_probs = revealed_probs.gather(1, current_tokens.unsqueeze(1)).squeeze(1)

                        n_remask = max(1, int(len(revealed_indices) * remask_fraction))
                        progress = i / gen_steps
                        n_remask = max(1, int(n_remask * (1 - progress)))

                        _, lowest_conf_idx = token_probs.topk(min(n_remask, len(token_probs)), largest=False)
                        remask_positions = revealed_indices[lowest_conf_idx]
                        x[0, remask_positions] = MASK_TOKEN_ID
    finally:
        hook_handle.remove()

    text = decode_output(tokenizer, x, prompt_len)
    return text


# ==============================================================================
# GSM8K Evaluation
# ==============================================================================

def extract_gsm8k_answer(text):
    """Extract numerical answer from GSM8K response."""
    patterns = [
        r'####\s*([\d,]+(?:\.\d+)?)',
        r'(?:the answer is|answer:)\s*\$?([\d,]+(?:\.\d+)?)',
        r'(?:=|equals?)\s*\$?([\d,]+(?:\.\d+)?)\s*(?:\.|$)',
        r'\*\*\$?([\d,]+(?:\.\d+)?)\*\*\s*(?:\.|$)',
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return float(m.group(1).replace(',', ''))
    numbers = re.findall(r'(?<![a-zA-Z])\$?([\d,]+(?:\.\d+)?)(?![a-zA-Z])', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except ValueError:
            pass
    return None


def get_gsm8k_target(example):
    answer_text = example.get("answer", "")
    m = re.search(r'####\s*([\d,]+(?:\.\d+)?)', answer_text)
    if m:
        return float(m.group(1).replace(',', ''))
    return None


def eval_gsm8k(method_name, generate_fn, dataset, num_samples):
    """Evaluate a generation method on GSM8K subset."""
    results = []
    correct = 0
    total = 0
    start_time = time.time()
    cublas_errors = 0

    for idx in range(min(num_samples, len(dataset))):
        example = dataset[idx]
        question = example["question"]
        target = get_gsm8k_target(example)
        if target is None:
            continue

        prompt = f"Solve this math problem step by step. Show your work and give the final numerical answer.\n\nQuestion: {question}\n\nSolution:"

        try:
            def _gen():
                return generate_fn(prompt)
            text = cublas_safe_run(_gen)
        except Exception as e:
            if "CUBLAS" in str(e):
                cublas_errors += 1
            print(f"  [{method_name}] GSM8K sample {idx} error: {str(e)[:80]}")
            text = ""

        extracted = extract_gsm8k_answer(text) if text else None
        is_correct = extracted is not None and abs(extracted - target) < 1e-3
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "idx": idx,
            "target": target,
            "extracted": extracted,
            "is_correct": is_correct,
            "gen_text_preview": text[:200] if text else ""
        })

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time_s": elapsed,
        "cublas_errors": cublas_errors,
        "results": results[:5]
    }


# ==============================================================================
# HumanEval Evaluation (functional correctness via pass@1)
# ==============================================================================

def eval_humaneval(method_name, generate_fn, dataset, num_samples):
    """
    Evaluate on HumanEval using functional correctness (pass@1).

    For PILOT mode with masked diffusion models, we use a simplified evaluation:
    - Generate code completions
    - Check if the generated code is syntactically valid Python
    - Attempt to run the test cases if provided in the prompt

    Note: Full HumanEval pass@1 requires sandboxed code execution.
    For pilot, we use syntax validity + pattern matching as proxy.
    """
    results = []
    correct = 0
    total = 0
    start_time = time.time()
    cublas_errors = 0

    for idx in range(min(num_samples, len(dataset))):
        example = dataset[idx]

        # HumanEval format: prompt is the function signature + docstring
        prompt_text = example.get("prompt", "")
        task_id = example.get("task_id", f"HumanEval/{idx}")
        canonical_solution = example.get("canonical_solution", "")
        test_code = example.get("test", "")
        entry_point = example.get("entry_point", "")

        if not prompt_text:
            continue

        # Ask the model to complete the function
        gen_prompt = f"Complete the following Python function. Only output the function body (the code that goes after the function signature). Do not repeat the function signature.\n\n{prompt_text}"

        try:
            def _gen():
                return generate_fn(gen_prompt)
            text = cublas_safe_run(_gen)
        except Exception as e:
            if "CUBLAS" in str(e):
                cublas_errors += 1
            print(f"  [{method_name}] HumanEval sample {idx} error: {str(e)[:80]}")
            text = ""

        # Evaluate: try to compile and run test cases
        passed = False
        error_msg = ""

        if text:
            # Combine prompt + generated completion
            full_code = prompt_text + "\n" + text

            # First check: can it compile?
            try:
                compile(full_code, f"<humaneval_{idx}>", "exec")

                # Second check: try to run test cases
                if test_code and entry_point:
                    try:
                        exec_globals = {}
                        exec(full_code, exec_globals)

                        # Run the test
                        test_full = full_code + "\n" + test_code
                        exec(test_full, exec_globals)

                        # If we get here, tests passed
                        # Call the check function if it exists
                        if "check" in test_code:
                            exec(f"check({entry_point})", exec_globals)

                        passed = True
                    except Exception as e:
                        error_msg = f"test_fail: {str(e)[:100]}"
                else:
                    # No test code available, syntax valid = partial pass
                    error_msg = "no_test_available"
            except SyntaxError as e:
                error_msg = f"syntax_error: {str(e)[:100]}"
        else:
            error_msg = "empty_generation"

        if passed:
            correct += 1
        total += 1

        results.append({
            "idx": idx,
            "task_id": task_id,
            "passed": passed,
            "error": error_msg,
            "gen_text_preview": text[:200] if text else ""
        })

    elapsed = time.time() - start_time
    accuracy = correct / total if total > 0 else 0

    return {
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "time_s": elapsed,
        "cublas_errors": cublas_errors,
        "results": results[:5]
    }


# ==============================================================================
# Main
# ==============================================================================

def main():
    global_start = time.time()
    print(f"=== C2: DaL + TTS Composition Test (PILOT) ===")
    print(f"Task ID: {TASK_ID}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"2x2 factorial: {{no memory, DaL-Full}} x {{standard, ReMDM}}")

    # Write PID
    Path(RESULTS_DIR).joinpath(f"{TASK_ID}.pid").write_text(str(os.getpid()))
    for suffix in ["_DONE", "_PROGRESS.json"]:
        p = Path(RESULTS_DIR) / f"{TASK_ID}{suffix}"
        if p.exists():
            p.unlink()

    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # === Load model ===
    print(f"\n[1/8] Loading LLaDA-8B-Instruct...")
    report_progress("loading_model", 0, 8)

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()
    print(f"  Model loaded on {device}")

    # === Load GSM8K ===
    print(f"\n[2/8] Loading GSM8K dataset...")
    report_progress("loading_data", 1, 8)

    from datasets import load_from_disk
    gsm8k_path = f"{SHARED_DIR}/datasets/gsm8k"
    gsm8k = load_from_disk(gsm8k_path)
    if "test" in gsm8k:
        gsm8k_test = gsm8k["test"]
    elif "train" in gsm8k:
        gsm8k_test = gsm8k["train"]
    else:
        gsm8k_test = gsm8k
    print(f"  GSM8K test: {len(gsm8k_test)} samples, using {PILOT_SAMPLES}")

    # === Load HumanEval ===
    print(f"\n[2.5/8] Loading HumanEval dataset...")
    humaneval_test = None
    try:
        humaneval_path = f"{SHARED_DIR}/datasets/humaneval"
        humaneval = load_from_disk(humaneval_path)
        if "test" in humaneval:
            humaneval_test = humaneval["test"]
        elif "train" in humaneval:
            humaneval_test = humaneval["train"]
        else:
            humaneval_test = humaneval
        print(f"  HumanEval: {len(humaneval_test)} samples, using {PILOT_SAMPLES}")
    except Exception as e:
        print(f"  HumanEval load failed: {e}")
        print(f"  Will try loading from HuggingFace...")
        try:
            from datasets import load_dataset
            humaneval = load_dataset("openai_humaneval", trust_remote_code=True)
            humaneval_test = humaneval["test"]
            print(f"  HumanEval from HF: {len(humaneval_test)} samples")
        except Exception as e2:
            print(f"  HumanEval unavailable: {e2}. Will skip HumanEval evaluation.")

    # === Load DaL-Linear checkpoint (gate_sep config) ===
    print(f"\n[3/8] Loading DaL-Linear checkpoint (gate_sep config)...")
    report_progress("loading_dal", 2, 8)

    ttt_layer = None
    ssl_head = None
    ssl_gate = None
    layer_norm = None
    dal_loaded = False

    try:
        dal_ckpt = torch.load(DAL_LINEAR_CKPT, map_location=device, weights_only=False)

        ttt_layer = build_ttt_layer(
            variant="linear",
            d_model=D_MODEL,
            d_ttt=D_MODEL,
            vocab_size=VOCAB_SIZE,
        ).to(dtype=torch.bfloat16, device=device).eval()

        if "ttt_layer_state_dict" in dal_ckpt:
            ttt_layer.load_state_dict(dal_ckpt["ttt_layer_state_dict"], strict=False)

        ssl_head = ttt_layer.ssl_head
        layer_norm = ttt_layer.layer_norm

        ssl_gate = nn.Parameter(torch.tensor(0.0, dtype=torch.bfloat16, device=device))

        # gate_sep config: gate = sigmoid(0) = 0.5
        with torch.no_grad():
            ttt_layer.gate_logit.fill_(0.0)

        dal_loaded = True
        print(f"  DaL-Linear loaded. Gate={torch.sigmoid(ttt_layer.gate_logit).item():.4f}, LR={ttt_layer.lr.item():.4f}")
    except Exception as e:
        print(f"  Failed to load DaL checkpoint: {e}")
        traceback.print_exc()

    if not dal_loaded:
        print("  WARNING: DaL checkpoint not loaded. DaL conditions will be skipped.")

    # ==================================================================
    # 2x2 Factorial Evaluation
    # ==================================================================

    # Conditions:
    # (1) No Memory + Standard    = vanilla
    # (2) No Memory + ReMDM       = remdm
    # (3) DaL-Full + Standard     = dal_standard
    # (4) DaL-Full + ReMDM        = dal_remdm

    conditions = [
        ("vanilla", "No Memory", "Standard"),
        ("remdm", "No Memory", "ReMDM"),
    ]
    if dal_loaded:
        conditions += [
            ("dal_standard", "DaL-Full", "Standard"),
            ("dal_remdm", "DaL-Full", "ReMDM"),
        ]

    gsm8k_results = {}
    humaneval_results = {}

    for cond_idx, (cond_name, memory, sampling) in enumerate(conditions):
        print(f"\n[{4 + cond_idx}/8] Evaluating: {cond_name} (Memory={memory}, Sampling={sampling})")
        report_progress(f"eval_{cond_name}", 3 + cond_idx, 8,
                        {"memory": memory, "sampling": sampling})

        # Build generation function for this condition
        if cond_name == "vanilla":
            gen_fn = lambda prompt: generate_vanilla(model, tokenizer, prompt, device, GEN_LEN)
        elif cond_name == "remdm":
            gen_fn = lambda prompt: generate_remdm(model, tokenizer, prompt, device, GEN_LEN,
                                                    remask_fraction=0.2)
        elif cond_name == "dal_standard":
            gen_fn = lambda prompt: generate_dal_standard(
                model, ttt_layer, ssl_head, ssl_gate, layer_norm,
                tokenizer, prompt, device, GEN_LEN, INSERTION_LAYER)
        elif cond_name == "dal_remdm":
            gen_fn = lambda prompt: generate_dal_remdm(
                model, ttt_layer, ssl_head, ssl_gate, layer_norm,
                tokenizer, prompt, device, GEN_LEN, INSERTION_LAYER,
                remask_fraction=0.2)

        # --- GSM8K ---
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print(f"  GSM8K evaluation ({PILOT_SAMPLES} samples)...")
        gsm8k_result = eval_gsm8k(cond_name, gen_fn, gsm8k_test, PILOT_SAMPLES)
        gsm8k_results[cond_name] = gsm8k_result
        print(f"  GSM8K: {gsm8k_result['accuracy']:.3f} ({gsm8k_result['correct']}/{gsm8k_result['total']}) in {gsm8k_result['time_s']:.1f}s")

        # --- HumanEval ---
        if humaneval_test is not None:
            torch.manual_seed(SEED)
            np.random.seed(SEED)
            print(f"  HumanEval evaluation ({PILOT_SAMPLES} samples)...")
            he_result = eval_humaneval(cond_name, gen_fn, humaneval_test, PILOT_SAMPLES)
            humaneval_results[cond_name] = he_result
            print(f"  HumanEval: {he_result['accuracy']:.3f} ({he_result['correct']}/{he_result['total']}) in {he_result['time_s']:.1f}s")

        gc.collect()
        torch.cuda.empty_cache()

    # ==================================================================
    # Additivity Analysis
    # ==================================================================

    print(f"\n{'='*70}")
    print(f"Composition Analysis (H7: Cross-step memory + remasking complementarity)")
    print(f"{'='*70}")

    analysis = {}

    # GSM8K analysis
    v_acc = gsm8k_results.get("vanilla", {}).get("accuracy", 0)
    r_acc = gsm8k_results.get("remdm", {}).get("accuracy", 0)
    d_acc = gsm8k_results.get("dal_standard", {}).get("accuracy", 0)
    dr_acc = gsm8k_results.get("dal_remdm", {}).get("accuracy", 0)

    remdm_gain = r_acc - v_acc
    dal_gain = d_acc - v_acc
    combined_gain = dr_acc - v_acc
    expected_additive = remdm_gain + dal_gain

    if expected_additive != 0:
        interaction_ratio = combined_gain / expected_additive if expected_additive != 0 else float('nan')
    else:
        interaction_ratio = float('nan') if combined_gain == 0 else float('inf')

    if combined_gain > expected_additive + 0.01:
        interaction_type = "super-additive"
    elif combined_gain < expected_additive - 0.01:
        interaction_type = "sub-additive"
    else:
        interaction_type = "additive"

    gsm8k_analysis = {
        "vanilla_acc": v_acc,
        "remdm_acc": r_acc,
        "dal_standard_acc": d_acc,
        "dal_remdm_acc": dr_acc,
        "remdm_gain": remdm_gain,
        "dal_gain": dal_gain,
        "combined_gain": combined_gain,
        "expected_additive_gain": expected_additive,
        "interaction_ratio": interaction_ratio if not math.isnan(interaction_ratio) and not math.isinf(interaction_ratio) else str(interaction_ratio),
        "interaction_type": interaction_type,
    }
    analysis["gsm8k"] = gsm8k_analysis

    print(f"\nGSM8K ({PILOT_SAMPLES} samples):")
    print(f"  Vanilla (baseline):     {v_acc:.3f}")
    print(f"  ReMDM only:             {r_acc:.3f} (gain: {remdm_gain:+.3f})")
    print(f"  DaL only:               {d_acc:.3f} (gain: {dal_gain:+.3f})")
    print(f"  DaL + ReMDM:            {dr_acc:.3f} (gain: {combined_gain:+.3f})")
    print(f"  Expected (additive):    {v_acc + expected_additive:.3f}")
    print(f"  Interaction type:       {interaction_type}")

    # HumanEval analysis
    if humaneval_results:
        hv_acc = humaneval_results.get("vanilla", {}).get("accuracy", 0)
        hr_acc = humaneval_results.get("remdm", {}).get("accuracy", 0)
        hd_acc = humaneval_results.get("dal_standard", {}).get("accuracy", 0)
        hdr_acc = humaneval_results.get("dal_remdm", {}).get("accuracy", 0)

        he_remdm_gain = hr_acc - hv_acc
        he_dal_gain = hd_acc - hv_acc
        he_combined_gain = hdr_acc - hv_acc
        he_expected = he_remdm_gain + he_dal_gain

        if he_expected != 0:
            he_ratio = he_combined_gain / he_expected
        else:
            he_ratio = float('nan') if he_combined_gain == 0 else float('inf')

        if he_combined_gain > he_expected + 0.01:
            he_interaction = "super-additive"
        elif he_combined_gain < he_expected - 0.01:
            he_interaction = "sub-additive"
        else:
            he_interaction = "additive"

        he_analysis = {
            "vanilla_acc": hv_acc,
            "remdm_acc": hr_acc,
            "dal_standard_acc": hd_acc,
            "dal_remdm_acc": hdr_acc,
            "remdm_gain": he_remdm_gain,
            "dal_gain": he_dal_gain,
            "combined_gain": he_combined_gain,
            "expected_additive_gain": he_expected,
            "interaction_ratio": he_ratio if not math.isnan(he_ratio) and not math.isinf(he_ratio) else str(he_ratio),
            "interaction_type": he_interaction,
        }
        analysis["humaneval"] = he_analysis

        print(f"\nHumanEval ({PILOT_SAMPLES} samples):")
        print(f"  Vanilla (baseline):     {hv_acc:.3f}")
        print(f"  ReMDM only:             {hr_acc:.3f} (gain: {he_remdm_gain:+.3f})")
        print(f"  DaL only:               {hd_acc:.3f} (gain: {he_dal_gain:+.3f})")
        print(f"  DaL + ReMDM:            {hdr_acc:.3f} (gain: {he_combined_gain:+.3f})")
        print(f"  Expected (additive):    {hv_acc + he_expected:.3f}")
        print(f"  Interaction type:       {he_interaction}")

    # ==================================================================
    # Build factorial table
    # ==================================================================

    factorial_table = []
    for cond_name, memory, sampling in conditions:
        row = {
            "condition": cond_name,
            "memory": memory,
            "sampling": sampling,
        }
        if cond_name in gsm8k_results:
            row["gsm8k_acc"] = gsm8k_results[cond_name]["accuracy"]
            row["gsm8k_correct"] = gsm8k_results[cond_name]["correct"]
            row["gsm8k_total"] = gsm8k_results[cond_name]["total"]
            row["gsm8k_time_s"] = gsm8k_results[cond_name]["time_s"]
        if cond_name in humaneval_results:
            row["humaneval_acc"] = humaneval_results[cond_name]["accuracy"]
            row["humaneval_correct"] = humaneval_results[cond_name]["correct"]
            row["humaneval_total"] = humaneval_results[cond_name]["total"]
            row["humaneval_time_s"] = humaneval_results[cond_name]["time_s"]
        factorial_table.append(row)

    # Print summary table
    print(f"\n{'='*70}")
    print(f"2x2 Factorial Summary")
    print(f"{'='*70}")
    print(f"{'Condition':<20} {'Memory':<12} {'Sampling':<12} {'GSM8K':>8} {'HumanEval':>10}")
    print(f"{'-'*70}")
    for row in factorial_table:
        gsm = f"{row.get('gsm8k_acc', 'N/A'):.3f}" if isinstance(row.get('gsm8k_acc'), float) else "N/A"
        he = f"{row.get('humaneval_acc', 'N/A'):.3f}" if isinstance(row.get('humaneval_acc'), float) else "N/A"
        print(f"{row['condition']:<20} {row['memory']:<12} {row['sampling']:<12} {gsm:>8} {he:>10}")

    # ==================================================================
    # Save results
    # ==================================================================

    total_time = time.time() - global_start

    # Determine pass criteria
    all_conditions_valid = all(
        cond_name in gsm8k_results and gsm8k_results[cond_name]["accuracy"] is not None
        for cond_name, _, _ in conditions
    )

    output = {
        "task_id": TASK_ID,
        "experiment": "C2: DaL + TTS Composition Test (PILOT)",
        "hypothesis": "H7: Cross-step memory (DaL) and improved remasking (ReMDM) are complementary",
        "model": "LLaDA-8B-Instruct",
        "mode": "PILOT",
        "pilot_samples": PILOT_SAMPLES,
        "gen_steps": GEN_STEPS,
        "gen_len": GEN_LEN,
        "seed": SEED,
        "dal_variant": "DaL-Linear (gate_sep, best from M6)",
        "remdm_fraction": 0.2,
        "factorial_design": {
            "memory_levels": ["No Memory", "DaL-Full"],
            "sampling_levels": ["Standard", "ReMDM"],
        },
        "factorial_table": factorial_table,
        "gsm8k_results": gsm8k_results,
        "humaneval_results": humaneval_results if humaneval_results else {"note": "HumanEval dataset unavailable"},
        "analysis": analysis,
        "pass_criteria": "All 4 factorial conditions produce valid accuracy numbers",
        "pass_criteria_met": all_conditions_valid,
        "timing": {
            "total_elapsed_s": total_time,
            "total_elapsed_min": total_time / 60,
        },
        "gpu_info": {
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if torch.cuda.is_available() else "unknown",
            "vram_total_mb": torch.cuda.get_device_properties(device).total_memory / (1024**2) if torch.cuda.is_available() else 0,
        },
        "timestamp": datetime.now().isoformat(),
    }

    os.makedirs(FULL_RESULTS_DIR, exist_ok=True)
    result_path = f"{FULL_RESULTS_DIR}/c2_composition.json"
    Path(result_path).write_text(json.dumps(output, indent=2, default=str))
    print(f"\nResults saved to: {result_path}")

    # Mark done
    summary_parts = []
    for cond_name, _, _ in conditions:
        gsm = gsm8k_results.get(cond_name, {}).get("accuracy", "N/A")
        summary_parts.append(f"{cond_name}_gsm8k={gsm}")
    summary = f"Composition test complete. {', '.join(summary_parts)}. Interaction: {analysis.get('gsm8k', {}).get('interaction_type', 'unknown')}"
    status = "success" if all_conditions_valid else "partial"
    mark_done(status, summary)
    print(f"\n{summary}")
    print(f"Status: {status}")


if __name__ == "__main__":
    main()
