"""
Task 5b (FULL): GSM8K DTA Evaluation.
1319 problems (full test set), parameterized seed, 4 methods.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 full_5b_gsm8k.py --seed 42
"""
import os, sys, json, time, random, re, math, gc, argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = RESULTS_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

GEN_LEN = 512
STEPS = 128
TEMPERATURE = 0.4
MASK_TOKEN_ID = 151666

LORA_RANK = 4
LORA_LAYERS = 2
LORA_LR = 5e-4
LORA_GAMMA = 0.95
WARMUP_FRAC = 0.20
LORA_ALPHA = 1.0
GRAD_CLIP = 1.0

REMASK_RATIO = 0.1
REMASK_STOP_FRAC = 0.8

CHECKPOINT_EVERY = 50


# ── GSM8K Data ──

def load_gsm8k():
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
    return problems


def format_gsm8k_prompt(problem):
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


def extract_model_answer(text):
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
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
                return float(match.group(1).replace(',', ''))
            except:
                continue
    match = re.search(r'=\s*([\-]?\d[\d,]*\.?\d*)\s*$', text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass
    return None


def verify_gsm8k(text, problem):
    extracted = extract_model_answer(text)
    target = problem["target"]
    if extracted is None or target is None:
        return {"is_correct": False, "extracted": extracted, "target": target}
    is_correct = abs(extracted - target) < 1e-3
    return {"is_correct": is_correct, "extracted": extracted, "target": target}


# ── Shared utilities (same as full_5a) ──

def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0

def compute_text_metrics(text):
    return {
        "word_count": len(text.split()),
        "distinct_2": distinct_n(text, 2),
        "rep_3": 1.0 - distinct_n(text, 3),
    }

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
    outputs = model(x, attention_mask="full", position_ids=None)
    logits = outputs.logits
    logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
    return logits


# ── Methods ──

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
            t = timesteps[i]; s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
    return decode_output(tokenizer, x, prompt_len), time.time() - t0

def generate_remdm_conf(model, tokenizer, prompt_text, device="cuda:0"):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    remask_stop_step = int(STEPS * REMASK_STOP_FRAC)
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
            if i < remask_stop_step and i < STEPS - 1:
                gen_region = x[0, prompt_len:]
                revealed = (gen_region != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
                if len(revealed) > 3:
                    logits2 = get_shifted_logits(model, x)
                    probs = F.softmax(logits2[0, prompt_len:][revealed], dim=-1)
                    conf = probs.gather(-1, gen_region[revealed].unsqueeze(-1)).squeeze(-1)
                    n_remask = max(1, int(len(revealed) * REMASK_RATIO))
                    _, lowest = conf.topk(n_remask, largest=False)
                    x[0, revealed[lowest] + prompt_len] = MASK_TOKEN_ID
    return decode_output(tokenizer, x, prompt_len), time.time() - t0


# ── LoRA ──

class LoRALayer(nn.Module):
    def __init__(self, original, rank, alpha=1.0):
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        self.lora_A = nn.Parameter(torch.zeros(rank, original.in_features,
                                               dtype=original.weight.dtype, device=original.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(original.out_features, rank,
                                               dtype=original.weight.dtype, device=original.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        original.weight.requires_grad_(False)
        if original.bias is not None:
            original.bias.requires_grad_(False)

    def forward(self, x):
        return self.original(x) + (x @ self.lora_A.T) @ self.lora_B.T * self.scaling

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def get_delta_norm(self):
        with torch.no_grad():
            return ((self.lora_B.float() @ self.lora_A.float()) * self.scaling).norm().item()

def inject_lora(model, n_layers=2, rank=4, alpha=1.0):
    total_layers = len(model.model.layers)
    target_layers = list(range(total_layers - n_layers, total_layers))
    lora_params, lora_modules = [], []
    for li in target_layers:
        mlp = model.model.layers[li].mlp
        for pn in ['gate_proj', 'up_proj', 'down_proj']:
            orig = getattr(mlp, pn)
            ll = LoRALayer(orig, rank=rank, alpha=alpha)
            setattr(mlp, pn, ll)
            lora_params.extend([ll.lora_A, ll.lora_B])
            lora_modules.append(ll)
    print(f"[LoRA] {len(lora_modules)} modules, {sum(p.numel() for p in lora_params)} params")
    return lora_params, lora_modules

def reset_lora(modules, opt):
    for m in modules:
        m.reset_parameters()
    opt.zero_grad(); opt.state.clear()

def decay_lora(params, gamma):
    with torch.no_grad():
        for p in params:
            p.mul_(gamma)


def generate_dta(model, tokenizer, prompt_text, device="cuda:0",
                 lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)
    reset_lora(lora_modules, optimizer)
    max_norm = 0.0
    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]
        t = timesteps[i]; s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0
        gen = x[0, prompt_len:]
        revealed = (gen != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        if i >= warmup_steps and len(revealed) >= 5:
            n_mask = max(2, len(revealed) // 5)
            perm = torch.randperm(len(revealed), device=device)[:n_mask]
            mpos = revealed[perm] + prompt_len
            tgt = x[0, mpos].clone()
            xm = x.clone(); xm[0, mpos] = MASK_TOKEN_ID
            out = model(xm, attention_mask="full", position_ids=None)
            lg = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            loss = F.cross_entropy(lg[0, mpos], tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            optimizer.step(); optimizer.zero_grad()
            decay_lora(lora_params, LORA_GAMMA)
        norms = [m.get_delta_norm() for m in lora_modules]
        cn = max(norms) if norms else 0
        if cn > max_norm:
            max_norm = cn
    text = decode_output(tokenizer, x, prompt_len)
    return text, time.time() - t0, {"max_norm": max_norm}


def generate_dta_remdm(model, tokenizer, prompt_text, device="cuda:0",
                       lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)
    remask_stop = int(STEPS * REMASK_STOP_FRAC)
    reset_lora(lora_modules, optimizer)
    max_norm = 0.0
    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        mask_logits = logits[mask_index]
        t = timesteps[i]; s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(mask_logits[transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0
        # ReMDM step
        if i < remask_stop and i < STEPS - 1:
            gen = x[0, prompt_len:]
            rev = (gen != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(rev) > 3:
                with torch.no_grad():
                    lg2 = get_shifted_logits(model, x)
                probs = F.softmax(lg2[0, prompt_len:][rev], dim=-1)
                conf = probs.gather(-1, gen[rev].unsqueeze(-1)).squeeze(-1)
                nr = max(1, int(len(rev) * REMASK_RATIO))
                _, lo = conf.topk(nr, largest=False)
                x[0, rev[lo] + prompt_len] = MASK_TOKEN_ID
        # DTA step
        gen = x[0, prompt_len:]
        revealed = (gen != MASK_TOKEN_ID).nonzero(as_tuple=True)[0]
        if i >= warmup_steps and len(revealed) >= 5:
            n_mask = max(2, len(revealed) // 5)
            perm = torch.randperm(len(revealed), device=device)[:n_mask]
            mpos = revealed[perm] + prompt_len
            tgt = x[0, mpos].clone()
            xm = x.clone(); xm[0, mpos] = MASK_TOKEN_ID
            out = model(xm, attention_mask="full", position_ids=None)
            lg = torch.cat([out.logits[:, :1], out.logits[:, :-1]], dim=1)
            loss = F.cross_entropy(lg[0, mpos], tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, GRAD_CLIP)
            optimizer.step(); optimizer.zero_grad()
            decay_lora(lora_params, LORA_GAMMA)
        norms = [m.get_delta_norm() for m in lora_modules]
        cn = max(norms) if norms else 0
        if cn > max_norm:
            max_norm = cn
    text = decode_output(tokenizer, x, prompt_len)
    return text, time.time() - t0, {"max_norm": max_norm}


# ── Bootstrap CI ──

def bootstrap_ci(correct_list, n_bootstrap=10000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(correct_list, dtype=float)
    n = len(arr)
    means = sorted([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return {"mean": float(arr.mean()),
            "ci_low": float(means[int(alpha * n_bootstrap)]),
            "ci_high": float(means[int((1 - alpha) * n_bootstrap)])}


# ── Main ──

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    device = "cuda:0"
    start_time = datetime.now()

    methods_to_run = ["vanilla", "remdm_conf", "dta", "dta_remdm"]

    print(f"=== Task 5b FULL: GSM8K (seed={seed}) ===")
    print(f"Start: {start_time.isoformat()}")

    problems = load_gsm8k()
    n_samples = len(problems)
    print(f"Loaded {n_samples} GSM8K problems")

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}")

    all_results = {}
    all_summaries = {}

    generate_fns = {
        "vanilla": generate_vanilla,
        "remdm_conf": generate_remdm_conf,
    }

    # Non-LoRA methods
    for method_name in ["vanilla", "remdm_conf"]:
        print(f"\n{'='*60}\n  {method_name} (seed={seed})\n{'='*60}")
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        np.random.seed(seed); random.seed(seed)

        results = []
        correct = 0
        ckpt_file = CHECKPOINT_DIR / f"task5b_{method_name}_s{seed}.json"
        start_idx = 0
        if ckpt_file.exists():
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            results = ckpt.get("results", [])
            start_idx = len(results)
            correct = sum(1 for r in results if r.get("is_correct"))

        for idx in range(start_idx, n_samples):
            problem = problems[idx]
            prompt = format_gsm8k_prompt(problem)
            torch.manual_seed(seed + idx); torch.cuda.manual_seed(seed + idx)
            try:
                text, elapsed = generate_fns[method_name](model, tokenizer, prompt, device)
            except Exception as e:
                text, elapsed = "", 0
                print(f"  [ERROR] {idx}: {e}")
            v = verify_gsm8k(text, problem)
            if v["is_correct"]:
                correct += 1
            metrics = compute_text_metrics(text) if text else {"word_count": 0, "distinct_2": 0, "rep_3": 0}
            results.append({"idx": idx, "is_correct": v["is_correct"], "gen_time_s": elapsed,
                           "extracted": v["extracted"], "target": v["target"], **metrics})
            if (idx + 1) % 50 == 0:
                print(f"  [{method_name} {idx+1}/{n_samples}] acc={correct/(idx+1):.1%} | {elapsed:.1f}s")
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                with open(ckpt_file, "w") as f:
                    json.dump({"method": method_name, "seed": seed, "results": results}, f)
            torch.cuda.empty_cache()

        acc = correct / n_samples
        all_results[method_name] = results
        all_summaries[method_name] = {
            "accuracy": acc, "correct": correct, "total": n_samples,
            "avg_time_s": float(np.mean([r["gen_time_s"] for r in results])),
            "bootstrap_ci": bootstrap_ci([r["is_correct"] for r in results]),
        }
        print(f"  => {method_name}: {correct}/{n_samples} = {acc:.1%}")
        torch.cuda.empty_cache(); gc.collect()

    # LoRA methods
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS, rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    lora_fns = {"dta": generate_dta, "dta_remdm": generate_dta_remdm}

    for method_name in ["dta", "dta_remdm"]:
        print(f"\n{'='*60}\n  {method_name} (seed={seed})\n{'='*60}")
        torch.manual_seed(seed); torch.cuda.manual_seed(seed)
        np.random.seed(seed); random.seed(seed)

        results = []
        correct = 0
        ckpt_file = CHECKPOINT_DIR / f"task5b_{method_name}_s{seed}.json"
        start_idx = 0
        if ckpt_file.exists():
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            results = ckpt.get("results", [])
            start_idx = len(results)
            correct = sum(1 for r in results if r.get("is_correct"))

        for idx in range(start_idx, n_samples):
            problem = problems[idx]
            prompt = format_gsm8k_prompt(problem)
            torch.manual_seed(seed + idx); torch.cuda.manual_seed(seed + idx)
            try:
                text, elapsed, extras = lora_fns[method_name](
                    model, tokenizer, prompt, device,
                    lora_params=lora_params, lora_modules=lora_modules, optimizer=optimizer
                )
            except Exception as e:
                text, elapsed, extras = "", 0, {}
                print(f"  [ERROR] {idx}: {e}")
            v = verify_gsm8k(text, problem)
            if v["is_correct"]:
                correct += 1
            metrics = compute_text_metrics(text) if text else {"word_count": 0, "distinct_2": 0, "rep_3": 0}
            results.append({"idx": idx, "is_correct": v["is_correct"], "gen_time_s": elapsed,
                           "extracted": v["extracted"], "target": v["target"],
                           "max_norm": extras.get("max_norm", 0), **metrics})
            if (idx + 1) % 50 == 0:
                print(f"  [{method_name} {idx+1}/{n_samples}] acc={correct/(idx+1):.1%} | {elapsed:.1f}s")
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                with open(ckpt_file, "w") as f:
                    json.dump({"method": method_name, "seed": seed, "results": results}, f)
            torch.cuda.empty_cache()

        acc = correct / n_samples
        all_results[method_name] = results
        all_summaries[method_name] = {
            "accuracy": acc, "correct": correct, "total": n_samples,
            "avg_time_s": float(np.mean([r["gen_time_s"] for r in results])),
            "bootstrap_ci": bootstrap_ci([r["is_correct"] for r in results]),
        }
        print(f"  => {method_name}: {correct}/{n_samples} = {acc:.1%}")
        torch.cuda.empty_cache(); gc.collect()

    end_time = datetime.now()

    # Summary
    print(f"\n{'='*80}")
    print(f"  GSM8K FULL seed={seed}")
    print(f"{'='*80}")
    for m in methods_to_run:
        s = all_summaries[m]
        ci = s["bootstrap_ci"]
        print(f"  {m:<18} {s['accuracy']:>7.1%} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
              f"{s['correct']:>5}/{s['total']}  {s['avg_time_s']:.1f}s")

    output = {
        "task": "task_5b", "mode": "full", "benchmark": "gsm8k",
        "model": "Dream-v0-Instruct-7B", "n_samples": n_samples, "seed": seed,
        "steps": STEPS, "temperature": TEMPERATURE, "gen_len": GEN_LEN,
        "methods": methods_to_run, "summaries": all_summaries,
        "results": all_results,
        "wall_clock_s": (end_time - start_time).total_seconds(),
        "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
    }

    out_file = RESULTS_DIR / f"task5b_gsm8k_s{seed}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    done_file = RESULTS_DIR / f"task5b_s{seed}_DONE"
    done_file.write_text(json.dumps({
        "task_id": f"task_5b_s{seed}", "status": "success",
        "summary": f"GSM8K {n_samples} x seed {seed}: " + ", ".join(
            f"{m}={all_summaries[m]['accuracy']:.1%}" for m in methods_to_run),
        "timestamp": datetime.now().isoformat(),
    }))


if __name__ == "__main__":
    main()
