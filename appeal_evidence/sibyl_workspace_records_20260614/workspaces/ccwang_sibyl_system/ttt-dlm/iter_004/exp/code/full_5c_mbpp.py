"""
Task 5c (FULL): MBPP DTA Evaluation.
500 problems (sanitized test), parameterized seed, 3 methods.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 full_5c_mbpp.py --seed 42
"""
import os, sys, json, time, random, re, math, gc, argparse, tempfile, subprocess
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
SANDBOX_TIMEOUT = 15

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


# ── MBPP Data ──

def load_mbpp_sanitized():
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    problems = []
    for i, item in enumerate(ds):
        problems.append({
            "id": i,
            "task_id": item.get("task_id", i),
            "text": item["prompt"],
            "test_list": item["test_list"],
        })
    return problems

def format_mbpp_prompt(problem):
    test_example = problem["test_list"][0] if problem["test_list"] else ""
    return (
        f"Write a Python function to solve the following problem.\n\n"
        f"Problem: {problem['text']}\n\n"
        f"Test case: {test_example}\n\n"
        f"Provide only the Python function code, no explanations."
    )

def extract_code(text):
    code_block = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    func_match = re.search(r'((?:def |class |import |from ).*)', text, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()
    return text.strip()

def run_sandbox(code, test_code, timeout=SANDBOX_TIMEOUT):
    full_code = code + "\n\n" + test_code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        return result.returncode == 0
    except:
        return False
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# ── Shared utilities ──

def sample_tokens_from_logits(logits, temperature=1.0):
    if temperature > 0:
        logits = logits / temperature
    probs = F.softmax(logits, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
    return probs.gather(-1, sampled.unsqueeze(-1)).squeeze(-1), sampled

def prepare_input(tokenizer, prompt_text, device):
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    x = F.pad(input_ids, (0, prompt_len + GEN_LEN - input_ids.shape[1]), value=MASK_TOKEN_ID)
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
    return torch.cat([logits[:, :1], logits[:, :-1]], dim=1)


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
            t = timesteps[i]; s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            _, sampled = sample_tokens_from_logits(logits[mask_index][transfer_mask], TEMPERATURE)
            x0[transfer_mask] = sampled
            x[mask_index] = x0
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
            ll = LoRALayer(getattr(mlp, pn), rank=rank, alpha=alpha)
            setattr(mlp, pn, ll)
            lora_params.extend([ll.lora_A, ll.lora_B])
            lora_modules.append(ll)
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
    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        t = timesteps[i]; s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(logits[mask_index][transfer_mask], TEMPERATURE)
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
    return decode_output(tokenizer, x, prompt_len), time.time() - t0

def generate_dta_remdm(model, tokenizer, prompt_text, device="cuda:0",
                       lora_params=None, lora_modules=None, optimizer=None):
    x, prompt_len = prepare_input(tokenizer, prompt_text, device)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, STEPS + 1, device=device)
    warmup_steps = int(STEPS * WARMUP_FRAC)
    remask_stop = int(STEPS * REMASK_STOP_FRAC)
    reset_lora(lora_modules, optimizer)
    t0 = time.time()
    for i in range(STEPS):
        mask_index = (x == MASK_TOKEN_ID)
        if not mask_index.any():
            break
        with torch.no_grad():
            logits = get_shifted_logits(model, x)
        t = timesteps[i]; s = timesteps[i + 1]
        p_transfer = (1 - s / t).item() if i < STEPS - 1 else 1.0
        x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
        transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
        _, sampled = sample_tokens_from_logits(logits[mask_index][transfer_mask], TEMPERATURE)
        x0[transfer_mask] = sampled
        x[mask_index] = x0
        # ReMDM
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
        # DTA
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
    return decode_output(tokenizer, x, prompt_len), time.time() - t0


def bootstrap_ci(correct_list, n_bootstrap=10000, seed=42):
    rng = np.random.RandomState(seed)
    arr = np.array(correct_list, dtype=float)
    n = len(arr)
    means = sorted([rng.choice(arr, size=n, replace=True).mean() for _ in range(n_bootstrap)])
    return {"mean": float(arr.mean()),
            "ci_low": float(means[int(0.025 * n_bootstrap)]),
            "ci_high": float(means[int(0.975 * n_bootstrap)])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    device = "cuda:0"
    start_time = datetime.now()

    methods = ["vanilla", "dta", "dta_remdm"]
    print(f"=== Task 5c FULL: MBPP (seed={seed}) ===")

    problems = load_mbpp_sanitized()
    n_samples = len(problems)
    print(f"Loaded {n_samples} MBPP problems")

    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    all_results = {}
    all_summaries = {}

    # Vanilla
    print(f"\n{'='*60}\n  vanilla (seed={seed})\n{'='*60}")
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    results = []
    passed = 0
    ckpt_file = CHECKPOINT_DIR / f"task5c_vanilla_s{seed}.json"
    start_idx = 0
    if ckpt_file.exists():
        with open(ckpt_file) as f:
            ckpt = json.load(f)
        results = ckpt.get("results", [])
        start_idx = len(results)
        passed = sum(1 for r in results if r.get("passed"))

    for idx in range(start_idx, n_samples):
        problem = problems[idx]
        prompt = format_mbpp_prompt(problem)
        torch.manual_seed(seed + idx); torch.cuda.manual_seed(seed + idx)
        try:
            text, elapsed = generate_vanilla(model, tokenizer, prompt, device)
        except Exception as e:
            text, elapsed = "", 0
        code = extract_code(text)
        test_code = "\n".join(problem["test_list"])
        ok = run_sandbox(code, test_code) if code else False
        if ok:
            passed += 1
        results.append({"idx": idx, "task_id": problem["task_id"], "passed": ok, "gen_time_s": elapsed})
        if (idx + 1) % 50 == 0:
            print(f"  [vanilla {idx+1}/{n_samples}] pass@1={passed/(idx+1):.1%}")
        if (idx + 1) % CHECKPOINT_EVERY == 0:
            with open(ckpt_file, "w") as f:
                json.dump({"method": "vanilla", "seed": seed, "results": results}, f)
        torch.cuda.empty_cache()

    all_results["vanilla"] = results
    all_summaries["vanilla"] = {
        "pass_at_1": passed / n_samples, "passed": passed, "total": n_samples,
        "avg_time_s": float(np.mean([r["gen_time_s"] for r in results])),
        "bootstrap_ci": bootstrap_ci([r["passed"] for r in results]),
    }
    print(f"  => vanilla: {passed}/{n_samples} = {passed/n_samples:.1%}")
    torch.cuda.empty_cache(); gc.collect()

    # LoRA methods
    lora_params, lora_modules = inject_lora(model, n_layers=LORA_LAYERS, rank=LORA_RANK, alpha=LORA_ALPHA)
    optimizer = torch.optim.AdamW(lora_params, lr=LORA_LR, weight_decay=0.01)

    for method_name, gen_fn in [("dta", generate_dta), ("dta_remdm", generate_dta_remdm)]:
        print(f"\n{'='*60}\n  {method_name} (seed={seed})\n{'='*60}")
        torch.manual_seed(seed); torch.cuda.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        results = []
        passed = 0
        ckpt_file = CHECKPOINT_DIR / f"task5c_{method_name}_s{seed}.json"
        start_idx = 0
        if ckpt_file.exists():
            with open(ckpt_file) as f:
                ckpt = json.load(f)
            results = ckpt.get("results", [])
            start_idx = len(results)
            passed = sum(1 for r in results if r.get("passed"))

        for idx in range(start_idx, n_samples):
            problem = problems[idx]
            prompt = format_mbpp_prompt(problem)
            torch.manual_seed(seed + idx); torch.cuda.manual_seed(seed + idx)
            try:
                text, elapsed = gen_fn(model, tokenizer, prompt, device,
                                       lora_params=lora_params, lora_modules=lora_modules, optimizer=optimizer)
            except Exception as e:
                text, elapsed = "", 0
            code = extract_code(text)
            test_code = "\n".join(problem["test_list"])
            ok = run_sandbox(code, test_code) if code else False
            if ok:
                passed += 1
            results.append({"idx": idx, "task_id": problem["task_id"], "passed": ok, "gen_time_s": elapsed})
            if (idx + 1) % 50 == 0:
                print(f"  [{method_name} {idx+1}/{n_samples}] pass@1={passed/(idx+1):.1%}")
            if (idx + 1) % CHECKPOINT_EVERY == 0:
                with open(ckpt_file, "w") as f:
                    json.dump({"method": method_name, "seed": seed, "results": results}, f)
            torch.cuda.empty_cache()

        all_results[method_name] = results
        all_summaries[method_name] = {
            "pass_at_1": passed / n_samples, "passed": passed, "total": n_samples,
            "avg_time_s": float(np.mean([r["gen_time_s"] for r in results])),
            "bootstrap_ci": bootstrap_ci([r["passed"] for r in results]),
        }
        print(f"  => {method_name}: {passed}/{n_samples} = {passed/n_samples:.1%}")
        torch.cuda.empty_cache(); gc.collect()

    end_time = datetime.now()
    print(f"\n{'='*80}\n  MBPP FULL seed={seed}\n{'='*80}")
    for m in methods:
        s = all_summaries[m]
        ci = s["bootstrap_ci"]
        print(f"  {m:<18} {s['pass_at_1']:>7.1%} [{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]")

    output = {
        "task": "task_5c", "mode": "full", "benchmark": "mbpp_sanitized",
        "model": "Dream-v0-Instruct-7B", "n_samples": n_samples, "seed": seed,
        "methods": methods, "summaries": all_summaries, "results": all_results,
        "wall_clock_s": (end_time - start_time).total_seconds(),
        "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
    }
    out_file = RESULTS_DIR / f"task5c_mbpp_s{seed}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    done_file = RESULTS_DIR / f"task5c_s{seed}_DONE"
    done_file.write_text(json.dumps({
        "task_id": f"task_5c_s{seed}", "status": "success",
        "summary": f"MBPP {n_samples} x seed {seed}: " + ", ".join(
            f"{m}={all_summaries[m]['pass_at_1']:.1%}" for m in methods),
        "timestamp": datetime.now().isoformat(),
    }))

if __name__ == "__main__":
    main()
