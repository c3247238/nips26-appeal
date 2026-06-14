"""
Task 5c (PILOT): MBPP DTA Evaluation (16 samples).

Runs 3 methods on the same 16 MBPP problems with a single model load:
  1. Vanilla (standard Dream origin denoising)
  2. DTA (denoising-time adaptation - LoRA online update)
  3. DTA+ReMDM (DTA combined with ReMDM-conf)

Pass criteria (pilot): DTA Pass@1 >= vanilla.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 python3 task_5c_mbpp_dta.py
"""
import os, sys, json, time, random, re, math, gc, tempfile, subprocess
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# === Config ===
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 16
SEED = 42
GEN_LEN = 512       # Code can be longer
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666
SANDBOX_TIMEOUT = 15  # seconds per test execution

# DTA Hyperparams
LORA_RANK = 4
LORA_LAYERS = 2
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

# ReMDM-conf Hyperparams
REMASK_RATIO = 0.1
REMASK_STOP_FRAC = 0.8


# ──────────────────────────────────────────────────
# MBPP Data Loading
# ──────────────────────────────────────────────────

def load_mbpp_sanitized(n=None):
    """Load MBPP sanitized test split."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    problems = []
    for i, item in enumerate(ds):
        if n is not None and i >= n:
            break
        problems.append({
            "id": i,
            "task_id": item.get("task_id", i),
            "text": item["prompt"],
            "test_list": item["test_list"],
            "code": item.get("code", ""),
        })
    return problems


def format_mbpp_prompt(problem):
    """Format an MBPP problem as a 0-shot prompt for Dream-7B."""
    test_example = problem["test_list"][0] if problem["test_list"] else ""
    return (
        f"Write a Python function to solve the following problem.\n\n"
        f"Problem: {problem['text']}\n\n"
        f"Test case: {test_example}\n\n"
        f"Provide only the Python function code, no explanations."
    )


# ──────────────────────────────────────────────────
# Code Extraction & Sandbox
# ──────────────────────────────────────────────────

def extract_code_from_response(text):
    """Extract Python code from model response."""
    code_block = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    func_match = re.search(r'((?:def |class |import |from ).*)', text, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()
    return text.strip()


def run_code_sandbox(code, test_code, timeout=SANDBOX_TIMEOUT):
    """Run code + tests in a sandboxed subprocess."""
    full_code = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        runtime = time.time() - t0
        if result.returncode == 0:
            return {"passed": True, "error": None, "runtime": runtime}
        else:
            return {"passed": False, "error": result.stderr[:500], "runtime": runtime}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout ({timeout}s)", "runtime": timeout}
    except Exception as e:
        return {"passed": False, "error": str(e)[:500], "runtime": 0}
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ──────────────────────────────────────────────────
# Text Quality Metrics
# ──────────────────────────────────────────────────

def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0

def rep_n(text, n):
    return 1.0 - distinct_n(text, n)

def compute_text_metrics(text):
    return {
        "word_count": len(text.split()),
        "distinct_1": distinct_n(text, 1),
        "distinct_2": distinct_n(text, 2),
        "distinct_3": distinct_n(text, 3),
        "rep_2": rep_n(text, 2),
        "rep_3": rep_n(text, 3),
    }


# ──────────────────────────────────────────────────
# Sampling helper
# ──────────────────────────────────────────────────

def sample_tokens_from_logits(logits, temperature=1.0):
    if temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    confidence = probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
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
    """Forward pass + Dream's shifted logits."""
    outputs = model(x, attention_mask="full", position_ids=None)
    logits = outputs.logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


# ──────────────────────────────────────────────────
# Method 1: Vanilla
# ──────────────────────────────────────────────────

def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0"):
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

            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    return text, elapsed


# ──────────────────────────────────────────────────
# LoRA Implementation
# ──────────────────────────────────────────────────

class LoRALayer(nn.Module):
    def __init__(self, original: nn.Linear, rank: int, alpha: float = 1.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        self.lora_A = nn.Parameter(torch.zeros(rank, in_features,
                                               dtype=original.weight.dtype,
                                               device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank,
                                               dtype=original.weight.dtype,
                                               device=original.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x):
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base_out + lora_out

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_delta_norm(self):
        with torch.no_grad():
            delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
            return delta.norm().item()


def inject_lora(model, n_layers=2, rank=4, alpha=1.0):
    total_layers = len(model.model.layers)
    target_layers = list(range(total_layers - n_layers, total_layers))
    lora_params = []
    lora_modules = []
    for layer_idx in target_layers:
        layer = model.model.layers[layer_idx]
        mlp = layer.mlp
        for proj_name in ['gate_proj', 'up_proj', 'down_proj']:
            original = getattr(mlp, proj_name)
            lora_layer = LoRALayer(original, rank=rank, alpha=alpha)
            setattr(mlp, proj_name, lora_layer)
            lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])
            lora_modules.append(lora_layer)
    n_params = sum(p.numel() for p in lora_params)
    print(f"[LoRA] Injected into layers {target_layers}, {len(lora_modules)} modules, {n_params} params")
    return lora_params, lora_modules


def reset_lora(lora_modules, optimizer):
    for m in lora_modules:
        m.reset_parameters()
    optimizer.zero_grad()
    optimizer.state.clear()


def decay_lora_params(lora_params, gamma):
    with torch.no_grad():
        for p in lora_params:
            p.mul_(gamma)


# ──────────────────────────────────────────────────
# Method 2: DTA
# ──────────────────────────────────────────────────

def generate_dta(model, tokenizer, prompt_text, device="cuda:0",
                 lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)

    reset_lora(lora_modules, optimizer)

    n_updates = 0
    lora_losses = []
    max_norm = 0.0

    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break

        # E-step: standard denoising
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]

        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0

        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0

        # M-step: LoRA update
        gen_region = x[0, prompt_len:]
        revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        total_revealed = len(revealed_positions)

        if i >= warmup_steps and total_revealed >= 5:
            n_to_mask = max(2, total_revealed // 5)
            perm = torch.randperm(total_revealed, device=device)[:n_to_mask]
            mask_positions = revealed_positions[perm] + prompt_len

            target_tokens = x[0, mask_positions].clone()
            x_masked = x.clone()
            x_masked[0, mask_positions] = MASK_TOKEN_ID

            outputs_m = model(x_masked, attention_mask="full", position_ids=None)
            logits_m = outputs_m.logits
            logits_m = torch.cat([logits_m[:, :1], logits_m[:, :-1]], dim=1)

            loss_logits = logits_m[0, mask_positions]
            loss = F.cross_entropy(loss_logits, target_tokens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            lora_losses.append(loss.item())
            n_updates += 1

            decay_lora_params(lora_params, LORA_GAMMA)

        # Track norms
        norms = [m.get_delta_norm() for m in lora_modules]
        cur_max = max(norms) if norms else 0
        if cur_max > max_norm:
            max_norm = cur_max

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    avg_loss = float(np.mean(lora_losses)) if lora_losses else 0
    return text, elapsed, {"n_updates": n_updates, "max_norm": max_norm,
                           "avg_loss": avg_loss}


# ──────────────────────────────────────────────────
# Method 3: DTA + ReMDM-conf
# ──────────────────────────────────────────────────

def generate_dta_remdm(model, tokenizer, prompt_text, device="cuda:0",
                       lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)
    remask_stop_step = int(STEPS * REMASK_STOP_FRAC)

    reset_lora(lora_modules, optimizer)

    n_updates = 0
    lora_losses = []
    max_norm = 0.0
    total_remasked = 0

    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break

        # E-step: standard denoising
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]

        t = timesteps[i]
        s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0

        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0

        # ReMDM-conf step
        if i < remask_stop_step and i < STEPS - 1:
            gen_region = x[0, prompt_len:]
            revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(revealed_positions) > 3:
                with torch.no_grad():
                    logits2 = get_shifted_logits(model, x)
                gen_logits = logits2[0, prompt_len:]
                revealed_logits = gen_logits[revealed_positions]
                probs = F.softmax(revealed_logits, dim=-1)
                current_tokens = gen_region[revealed_positions]
                confidence = probs.gather(-1, current_tokens.unsqueeze(-1)).squeeze(-1)

                n_remask = max(1, int(len(revealed_positions) * REMASK_RATIO))
                _, lowest_idx = confidence.topk(n_remask, largest=False)
                remask_positions = revealed_positions[lowest_idx] + prompt_len
                x[0, remask_positions] = MASK_TOKEN_ID
                total_remasked += n_remask

        # M-step: LoRA update
        gen_region = x[0, prompt_len:]
        revealed_positions = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        total_revealed = len(revealed_positions)

        if i >= warmup_steps and total_revealed >= 5:
            n_to_mask = max(2, total_revealed // 5)
            perm = torch.randperm(total_revealed, device=device)[:n_to_mask]
            mask_positions = revealed_positions[perm] + prompt_len

            target_tokens = x[0, mask_positions].clone()
            x_masked = x.clone()
            x_masked[0, mask_positions] = MASK_TOKEN_ID

            outputs_m = model(x_masked, attention_mask="full", position_ids=None)
            logits_m = outputs_m.logits
            logits_m = torch.cat([logits_m[:, :1], logits_m[:, :-1]], dim=1)

            loss_logits = logits_m[0, mask_positions]
            loss = F.cross_entropy(loss_logits, target_tokens)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            lora_losses.append(loss.item())
            n_updates += 1

            decay_lora_params(lora_params, LORA_GAMMA)

        norms = [m.get_delta_norm() for m in lora_modules]
        cur_max = max(norms) if norms else 0
        if cur_max > max_norm:
            max_norm = cur_max

    elapsed = time.time() - t0
    text = decode_output(tokenizer, x, prompt_len)
    avg_loss = float(np.mean(lora_losses)) if lora_losses else 0
    return text, elapsed, {"n_updates": n_updates, "max_norm": max_norm,
                           "avg_loss": avg_loss, "total_remasked": total_remasked}


# ──────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────

def run_method(method_name, generate_fn, model, tokenizer, problems, device,
               lora_params=None, lora_modules=None, optimizer=None):
    """Run a method on all MBPP problems, return results list and summary."""
    results = []
    pass_count = 0
    n_nonempty = 0
    n_has_def = 0

    for idx, problem in enumerate(problems):
        prompt = format_mbpp_prompt(problem)

        if method_name in ("dta", "dta_remdm"):
            text, elapsed, extras = generate_fn(
                model, tokenizer, prompt, device,
                lora_params=lora_params, lora_modules=lora_modules,
                optimizer=optimizer
            )
        else:  # vanilla
            text, elapsed = generate_fn(model, tokenizer, prompt, device)
            extras = {}

        # Extract code
        code = extract_code_from_response(text)
        is_nonempty = len(code.strip()) > 0
        has_def = "def " in code
        if is_nonempty:
            n_nonempty += 1
        if has_def:
            n_has_def += 1

        # Sandbox evaluation
        test_code = "\n".join(problem["test_list"])
        sandbox_result = run_code_sandbox(code, test_code)
        passed = sandbox_result["passed"]
        if passed:
            pass_count += 1

        metrics = compute_text_metrics(text)
        result = {
            "idx": idx,
            "task_id": problem["task_id"],
            "problem_text": problem["text"][:200],
            "generated_text": text,
            "extracted_code": code[:500],
            "passed": passed,
            "sandbox_error": sandbox_result["error"],
            "sandbox_runtime": sandbox_result["runtime"],
            "gen_time_s": elapsed,
            "has_def": has_def,
            "is_nonempty": is_nonempty,
            **metrics,
            **extras,
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"  [{method_name} {idx+1}/{len(problems)}] {status} | "
              f"task_id={problem['task_id']} | has_def={has_def} | "
              f"{elapsed:.1f}s")

    pass_at_1 = pass_count / len(problems)
    summary = {
        "pass_at_1": pass_at_1,
        "pass_count": pass_count,
        "total": len(problems),
        "n_nonempty": n_nonempty,
        "n_has_def": n_has_def,
        "code_extraction_rate": n_nonempty / len(problems),
        "def_rate": n_has_def / len(problems),
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in results])),
        "distinct_2_mean": float(np.mean([r["distinct_2"] for r in results])),
        "rep_3_mean": float(np.mean([r["rep_3"] for r in results])),
    }
    print(f"  => {method_name}: {pass_count}/{len(problems)} = {pass_at_1:.1%} pass@1, "
          f"avg_time={summary['avg_time_s']:.1f}s")
    return results, summary


# ──────────────────────────────────────────────────
# DONE marker
# ──────────────────────────────────────────────────

def mark_task_done(task_id, results_dir, status="success", summary=""):
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    device = "cuda:0"
    start_time = datetime.now()
    print(f"=== Task 5c PILOT: MBPP DTA Evaluation ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Gen length: {GEN_LEN}")
    print(f"Start: {start_time.isoformat()}")

    # Load MBPP
    print(f"\nLoading MBPP sanitized test set (first {N_SAMPLES} samples)...")
    problems = load_mbpp_sanitized(n=N_SAMPLES)
    print(f"Loaded {len(problems)} problems")

    # Load model
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}")

    all_method_results = {}
    all_method_summaries = {}

    def set_seed():
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        np.random.seed(SEED)
        random.seed(SEED)

    # ── Method 1: Vanilla ──
    print(f"\n{'='*60}\n  Method 1/3: Vanilla\n{'='*60}")
    set_seed()
    results, summary = run_method("vanilla", generate_vanilla, model, tokenizer,
                                  problems, device)
    all_method_results["vanilla"] = results
    all_method_summaries["vanilla"] = summary
    torch.cuda.empty_cache(); gc.collect()

    # ── Methods with LoRA ──

    # Method 2: DTA
    print(f"\n{'='*60}\n  Method 2/3: DTA\n{'='*60}")
    set_seed()
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS,
                                            rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    results, summary = run_method("dta", generate_dta, model, tokenizer,
                                  problems, device,
                                  lora_params=lora_params,
                                  lora_modules=lora_modules,
                                  optimizer=optimizer)
    all_method_results["dta"] = results
    all_method_summaries["dta"] = summary
    dta_norms = [r.get("max_norm", 0) for r in results]
    all_method_summaries["dta"]["lora_norm_max"] = float(max(dta_norms)) if dta_norms else 0
    all_method_summaries["dta"]["lora_norm_mean"] = float(np.mean(dta_norms)) if dta_norms else 0
    torch.cuda.empty_cache(); gc.collect()

    # Method 3: DTA+ReMDM
    print(f"\n{'='*60}\n  Method 3/3: DTA+ReMDM\n{'='*60}")
    set_seed()
    results, summary = run_method("dta_remdm", generate_dta_remdm, model,
                                  tokenizer, problems, device,
                                  lora_params=lora_params,
                                  lora_modules=lora_modules,
                                  optimizer=optimizer)
    all_method_results["dta_remdm"] = results
    all_method_summaries["dta_remdm"] = summary
    combo_norms = [r.get("max_norm", 0) for r in results]
    all_method_summaries["dta_remdm"]["lora_norm_max"] = float(max(combo_norms)) if combo_norms else 0
    all_method_summaries["dta_remdm"]["lora_norm_mean"] = float(np.mean(combo_norms)) if combo_norms else 0
    torch.cuda.empty_cache(); gc.collect()

    end_time = datetime.now()

    # ── Summary Table ──
    print(f"\n{'='*80}")
    print(f"  FINAL SUMMARY: Task 5c Pilot - MBPP DTA ({N_SAMPLES} samples)")
    print(f"{'='*80}")
    print(f"  {'Method':<18} {'Pass@1':>8} {'Passed':>8} {'DefRate':>8} {'AvgTime':>10} {'Dist-2':>8} {'Rep-3':>8}")
    print(f"  {'-'*78}")

    methods_order = ["vanilla", "dta", "dta_remdm"]
    for m in methods_order:
        s = all_method_summaries[m]
        print(f"  {m:<18} {s['pass_at_1']:>7.1%} {s['pass_count']:>8} "
              f"{s['def_rate']:>7.1%} "
              f"{s['avg_time_s']:>9.1f}s {s['distinct_2_mean']:>7.3f} {s['rep_3_mean']:>7.3f}")

    # Pass criteria
    vanilla_pass = all_method_summaries["vanilla"]["pass_at_1"]
    dta_pass = all_method_summaries["dta"]["pass_at_1"]
    dta_remdm_pass = all_method_summaries["dta_remdm"]["pass_at_1"]
    all_ran = all(m in all_method_results for m in methods_order)
    all_have_results = all(len(all_method_results[m]) == N_SAMPLES for m in methods_order)
    dta_ge_vanilla = dta_pass >= vanilla_pass

    print(f"\n--- Pass Criteria ---")
    print(f"  All 3 methods ran:              {'PASS' if all_ran and all_have_results else 'FAIL'}")
    print(f"  DTA pass@1 >= vanilla:          {'PASS' if dta_ge_vanilla else 'FAIL'} "
          f"(DTA={dta_pass:.1%} vs Vanilla={vanilla_pass:.1%})")
    print(f"  DTA+ReMDM pass@1:               {dta_remdm_pass:.1%}")

    if all_ran and all_have_results and dta_ge_vanilla:
        overall = "GO"
    elif all_ran and all_have_results:
        overall = "CONDITIONAL-GO"
    else:
        overall = "NO-GO"
    print(f"  Overall: {overall}")
    print(f"\n  Wall-clock: {(end_time - start_time).total_seconds():.0f}s")

    # ── Save results ──
    output = {
        "task": "task_5c",
        "mode": "pilot",
        "benchmark": "mbpp_sanitized",
        "model": "Dream-v0-Instruct-7B",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "gen_len": GEN_LEN,
        "methods": methods_order,
        "summaries": all_method_summaries,
        "pass_criteria": {
            "all_methods_ran": all_ran and all_have_results,
            "dta_ge_vanilla": dta_ge_vanilla,
            "overall": overall,
        },
        "configs": {
            "dta": {"rank": LORA_RANK, "layers": LORA_LAYERS, "lr": LORA_LR,
                    "gamma": LORA_GAMMA, "warmup": WARMUP_FRAC, "alpha": LORA_ALPHA},
            "remdm": {"remask_ratio": REMASK_RATIO, "stop_frac": REMASK_STOP_FRAC},
        },
        "results": {m: all_method_results[m] for m in methods_order},
        "wall_clock_s": (end_time - start_time).total_seconds(),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "torch_version": torch.__version__,
    }

    out_file = RESULTS_DIR / "task_5c_mbpp_dta.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    # Write DONE marker
    try:
        mark_task_done("task_5c", "/home/ccwang/sibyl_system/exp/results",
                       status="success" if overall != "NO-GO" else "failed",
                       summary=f"MBPP pilot: vanilla={vanilla_pass:.1%}, dta={dta_pass:.1%}, "
                               f"dta+remdm={dta_remdm_pass:.1%}, verdict={overall}")
    except Exception as e:
        print(f"Warning: Failed to write DONE marker: {e}")


if __name__ == "__main__":
    main()
