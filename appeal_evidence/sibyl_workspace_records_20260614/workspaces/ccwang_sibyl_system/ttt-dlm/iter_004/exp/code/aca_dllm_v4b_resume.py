"""
ACA-DLM v4b: Best-of-N Baseline - Resumable version.
Runs one method+seed at a time, saves results incrementally.
Can be called repeatedly to complete remaining configs.
"""
import os, sys, json, time, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

sys.path.insert(0, "/home/ccwang/sibyl_system/src/dllm")
sys.path.insert(0, "/home/ccwang/sibyl_system/exp/code")
import dllm
from ttt_dllm_v3_sweep import load_model
from ttt_dllm_v4_best import make_diverse_prompts
from aca_dllm_v2_sweep import remask_retry_v2

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

METHODS = {
    "vanilla": {"type": "vanilla"},
    "best_of_2": {"type": "best_of_n", "n": 2},
    "best_of_3": {"type": "best_of_n", "n": 3},
    "retry_30pct": {"type": "retry", "n_retries": 2, "remask_ratio": 0.3, "refine_steps": 32},
    "retry_70pct": {"type": "retry", "n_retries": 2, "remask_ratio": 0.7, "refine_steps": 32},
}
SEEDS = [42, 123, 456]


def get_remaining_configs():
    """Find which method+seed combos still need to run."""
    remaining = []
    for method_name in METHODS:
        for seed in SEEDS:
            fname = RESULTS_DIR / f"v4_partial_{method_name}_s{seed}.json"
            if not fname.exists():
                remaining.append((method_name, seed))
    return remaining


def best_of_n_sample(model, tokenizer, prompt, config, n=2, device=None):
    if device is None:
        device = next(model.parameters()).device
    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())
    candidates = []
    for i in range(n):
        with torch.no_grad():
            seq = sampler.sample([prompt], config=config)
        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)
        prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=device)
        pl = prompt_t.shape[0]
        gen_conf = token_conf[0, pl:pl+config.max_new_tokens].mean().item()
        candidates.append((seq, gen_conf))
    return max(candidates, key=lambda x: x[1])[0]


def run_single_config(method_name, seed):
    from transformers import AutoModelForCausalLM

    method_cfg = METHODS[method_name]
    print(f"\n{'='*60}")
    print(f"Running: {method_name}, seed={seed}")
    print(f"{'='*60}")

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_diverse_prompts(tokenizer, n=32)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    texts = []
    confs = []

    for pi, prompt in enumerate(prompts):
        if method_cfg["type"] == "vanilla":
            sampler = dllm.core.samplers.MDLMSampler(
                model=model, tokenizer=tokenizer,
                scheduler=dllm.core.schedulers.LinearAlphaScheduler())
            with torch.no_grad():
                seq = sampler.sample([prompt], config=config)
        elif method_cfg["type"] == "best_of_n":
            seq = best_of_n_sample(model, tokenizer, prompt, config,
                                   n=method_cfg["n"], device=device)
        elif method_cfg["type"] == "retry":
            seq, _ = remask_retry_v2(model, tokenizer, prompt, config,
                                     n_retries=method_cfg["n_retries"],
                                     remask_ratio=method_cfg["remask_ratio"],
                                     refine_steps=method_cfg["refine_steps"])

        text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
        texts.append(text)

        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)
        prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=device)
        pl = prompt_t.shape[0]
        gen_conf = token_conf[0, pl:pl+config.max_new_tokens].mean().item()
        confs.append(gen_conf)

        if (pi + 1) % 8 == 0:
            print(f"  {pi+1}/32 done")

    # Phase 2: PPL eval
    model.cpu()
    torch.cuda.empty_cache()
    eval_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, trust_remote_code=True).to(device)
    eval_model.eval()

    ppls = []
    for text in texts:
        if len(text) < 10:
            ppls.append(None)
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = eval_model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))

    del eval_model
    torch.cuda.empty_cache()

    valid_ppls = [p for p in ppls if p is not None]

    result = {
        "method": method_name,
        "seed": seed,
        "ppl_mean": float(np.mean(valid_ppls)),
        "ppl_std": float(np.std(valid_ppls)),
        "conf_mean": float(np.mean(confs)),
        "n_valid": len(valid_ppls),
        "all_ppls": [float(p) for p in valid_ppls],
    }

    fname = RESULTS_DIR / f"v4_partial_{method_name}_s{seed}.json"
    with open(fname, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  PPL={result['ppl_mean']:.3f}, Conf={result['conf_mean']:.4f}")
    print(f"  Saved to {fname}")
    return result


def summarize():
    """Summarize all completed results."""
    all_results = {}
    for method_name in METHODS:
        ppls = []
        confs = []
        for seed in SEEDS:
            fname = RESULTS_DIR / f"v4_partial_{method_name}_s{seed}.json"
            if fname.exists():
                with open(fname) as f:
                    data = json.load(f)
                if "all_ppls" in data:
                    ppls.extend(data["all_ppls"])
                else:
                    # Old format without all_ppls
                    ppls.append(data["ppl_mean"])
                confs.append(data["conf_mean"])
        if ppls:
            all_results[method_name] = {
                "ppl_mean": float(np.mean(ppls)),
                "conf_mean": float(np.mean(confs)),
                "n_seeds": len(confs),
                "all_ppls": ppls,
            }

    if "vanilla" not in all_results:
        print("No vanilla results yet")
        return

    print("\n" + "=" * 90)
    print(f"{'Method':<20} {'PPL':>7} {'Conf':>7} {'Δ%':>8} {'p-value':>12} {'Compute':>10} {'Seeds':>6}")
    print("-" * 90)

    vanilla = all_results["vanilla"]
    compute = {"vanilla": "1.0x", "best_of_2": "2.0x", "best_of_3": "3.0x",
               "retry_30pct": "1.28x", "retry_70pct": "1.28x"}

    for name in ["vanilla", "retry_30pct", "retry_70pct", "best_of_2", "best_of_3"]:
        if name not in all_results:
            continue
        res = all_results[name]
        delta = (res["ppl_mean"] - vanilla["ppl_mean"]) / vanilla["ppl_mean"] * 100
        if name != "vanilla" and len(res["all_ppls"]) == len(vanilla["all_ppls"]):
            _, p = scipy_stats.ttest_rel(vanilla["all_ppls"], res["all_ppls"])
            sig = f"p={p:.6f}"
        else:
            sig = ""
        c = compute.get(name, "?")
        print(f"  {name:<18} {res['ppl_mean']:>7.3f} {res['conf_mean']:>7.4f} "
              f"{delta:>+7.1f}% {sig:>12} {c:>10} {res['n_seeds']:>6}")
    print("=" * 90)


if __name__ == "__main__":
    remaining = get_remaining_configs()
    if not remaining:
        print("All configs complete! Summary:")
        summarize()
    else:
        print(f"Remaining configs: {len(remaining)}")
        for method_name, seed in remaining:
            print(f"  - {method_name} seed={seed}")

        # Run next one
        method_name, seed = remaining[0]
        run_single_config(method_name, seed)

        # Show progress
        remaining_after = get_remaining_configs()
        print(f"\nRemaining: {len(remaining_after)} configs")
        summarize()
