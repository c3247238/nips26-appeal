"""
ACA-DLM v2: Fine-grained sweep of ReMask-Retry + combination with TTT.

Based on v1: retry_2x_30pct is best (-7.9%, p=0.005).
Now explore:
1. More retry counts (1, 2, 3, 4)
2. More remask ratios (0.15, 0.25, 0.30, 0.35, 0.40)
3. Refinement step budget (8, 16, 32, 64)
4. ReMask-Retry + block-reset TTT combo
5. ReMask-Retry with higher steps budget for base denoising
"""
import os, sys, json, time, math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ccwang/sibyl_system/src/dllm")
import dllm
from dllm.core.samplers.utils import add_gumbel_noise, get_num_transfer_tokens

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, "/home/ccwang/sibyl_system/exp/code")
from ttt_dllm_v3_sweep import load_model, get_ttt_params
from ttt_dllm_v4_best import make_diverse_prompts
from ttt_dllm_v6_prompt import sample_with_block_reset_ttt


def remask_retry_v2(model, tokenizer, prompt, config,
                    n_retries=2, remask_ratio=0.3, refine_steps=32):
    """ReMask-Retry with configurable refinement steps."""
    mask_id = tokenizer.mask_token_id

    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    with torch.no_grad():
        x = sampler.sample([prompt], config=config)

    prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=model.device)
    pl = prompt_t.shape[0]
    max_new_tokens = config.max_new_tokens
    gen_region = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    gen_region[pl:pl+max_new_tokens] = True
    attention_mask = torch.ones_like(x)

    stats = {"retries": 0, "tokens_remasked": 0}

    for retry in range(n_retries):
        with torch.no_grad():
            logits = model(x, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)

        n_gen = gen_region.sum().item()
        n_remask = max(1, int(n_gen * remask_ratio))
        gen_indices = torch.where(gen_region)[0]
        gen_confs = token_conf[0, gen_indices]
        _, worst_idx = gen_confs.topk(n_remask, largest=False)

        low_conf_mask = torch.zeros_like(gen_region)
        low_conf_mask[gen_indices[worst_idx]] = True

        x[0, low_conf_mask] = mask_id
        stats["tokens_remasked"] += n_remask
        stats["retries"] += 1

        # Refinement
        for step in range(refine_steps):
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits

            still_masked = (x == mask_id)
            if not still_masked.any():
                break

            pred = torch.argmax(logits, dim=-1)
            probs = F.softmax(logits.float(), dim=-1)
            pred_conf = torch.gather(probs, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)
            pred_conf[~still_masked] = -float("inf")

            n_still = still_masked.sum().item()
            n_unmask = max(1, n_still // max(1, refine_steps - step))
            _, top_idx = pred_conf[0].topk(min(n_unmask, n_still))
            x[0, top_idx] = pred[0, top_idx]

    return x, stats


def remask_retry_then_ttt(model, tokenizer, prompt, config,
                          n_retries=2, remask_ratio=0.3, refine_steps=32,
                          ttt_lr=3e-4, confidence_threshold=0.1, ttt_remask_ratio=0.1):
    """ReMask-Retry followed by block-reset TTT."""
    # First: ReMask-Retry
    x, retry_stats = remask_retry_v2(model, tokenizer, prompt, config,
                                      n_retries=n_retries, remask_ratio=remask_ratio,
                                      refine_steps=refine_steps)
    # Then: block-reset TTT on the result
    # We use the retry result as starting point for another refinement pass
    # Actually, TTT needs to run during denoising, not after.
    # So instead: run block-reset TTT from scratch
    seq_ttt, ttt_stats = sample_with_block_reset_ttt(
        model, tokenizer, prompt, config,
        param_type="q_proj", ttt_lr=ttt_lr,
        confidence_threshold=confidence_threshold,
        ttt_remask_ratio=ttt_remask_ratio)

    # Pick the better one based on model confidence
    mask_id = tokenizer.mask_token_id
    attention_mask_retry = torch.ones_like(x)
    attention_mask_ttt = torch.ones_like(seq_ttt)

    with torch.no_grad():
        logits_retry = model(x, attention_mask=attention_mask_retry).logits
        logits_ttt = model(seq_ttt, attention_mask=attention_mask_ttt).logits

    prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=model.device)
    pl = prompt_t.shape[0]
    gen_region = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    gen_region[pl:pl+config.max_new_tokens] = True

    probs_retry = F.softmax(logits_retry.float(), dim=-1)
    conf_retry = torch.gather(probs_retry, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)
    avg_conf_retry = conf_retry[0, gen_region].mean().item()

    probs_ttt = F.softmax(logits_ttt.float(), dim=-1)
    conf_ttt = torch.gather(probs_ttt, dim=-1, index=seq_ttt.unsqueeze(-1)).squeeze(-1)
    avg_conf_ttt = conf_ttt[0, gen_region].mean().item()

    # Pick higher confidence output
    if avg_conf_retry >= avg_conf_ttt:
        return x, {"method": "retry", "retry_conf": avg_conf_retry, "ttt_conf": avg_conf_ttt}
    else:
        return seq_ttt, {"method": "ttt", "retry_conf": avg_conf_retry, "ttt_conf": avg_conf_ttt}


def run_one_config(model, tokenizer, prompts, config, sampler, cfg_type, seed, device, **kwargs):
    """Run one config with one seed."""
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    samples = []
    total_stats = {}
    t0 = time.time()

    for prompt in prompts:
        if cfg_type is None:
            with torch.no_grad():
                seq = sampler.sample([prompt], config=config)
        elif cfg_type == "retry":
            seq, stats = remask_retry_v2(model, tokenizer, prompt, config, **kwargs)
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    total_stats[k] = total_stats.get(k, 0) + v
        elif cfg_type == "retry_ttt":
            seq, stats = remask_retry_then_ttt(model, tokenizer, prompt, config, **kwargs)
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    total_stats[k] = total_stats.get(k, 0) + v
        elif cfg_type == "ttt_only":
            seq, stats = sample_with_block_reset_ttt(model, tokenizer, prompt, config, **kwargs)
            for k, v in stats.items():
                if isinstance(v, (int, float)):
                    total_stats[k] = total_stats.get(k, 0) + v

        text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
        samples.append(text)

    gen_time = time.time() - t0

    # Evaluate
    model.cpu()
    torch.cuda.empty_cache()
    eval_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, trust_remote_code=True).to(device)
    eval_model.eval()

    ppls = []
    for text in samples:
        if len(text) < 10:
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = eval_model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))

    del eval_model
    torch.cuda.empty_cache()
    model.to(device)

    return {
        "mean": float(np.mean(ppls)) if ppls else float("inf"),
        "median": float(np.median(ppls)) if ppls else float("inf"),
        "std": float(np.std(ppls)) if ppls else 0,
        "n": len(ppls),
        "gen_time": gen_time,
        "ppls": ppls,
    }


def run_experiment():
    from scipy import stats as scipy_stats

    print("=" * 70)
    print("ACA-DLM v2: ReMask-Retry Sweep + TTT Combo")
    print("=" * 70)

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_diverse_prompts(tokenizer, n=32)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    seeds = [42, 123, 456]
    all_results = {}

    configs = [
        ("vanilla", None, {}),
        # Remask ratio sweep (2 retries, 32 refine steps)
        ("r_15pct", "retry", dict(n_retries=2, remask_ratio=0.15, refine_steps=32)),
        ("r_25pct", "retry", dict(n_retries=2, remask_ratio=0.25, refine_steps=32)),
        ("r_30pct", "retry", dict(n_retries=2, remask_ratio=0.30, refine_steps=32)),
        ("r_35pct", "retry", dict(n_retries=2, remask_ratio=0.35, refine_steps=32)),
        ("r_40pct", "retry", dict(n_retries=2, remask_ratio=0.40, refine_steps=32)),
        # Retry count sweep (30%, 32 refine steps)
        ("r_1x", "retry", dict(n_retries=1, remask_ratio=0.30, refine_steps=32)),
        ("r_3x", "retry", dict(n_retries=3, remask_ratio=0.30, refine_steps=32)),
        ("r_4x", "retry", dict(n_retries=4, remask_ratio=0.30, refine_steps=32)),
        # Refine steps sweep
        ("rs_8", "retry", dict(n_retries=2, remask_ratio=0.30, refine_steps=8)),
        ("rs_16", "retry", dict(n_retries=2, remask_ratio=0.30, refine_steps=16)),
        ("rs_64", "retry", dict(n_retries=2, remask_ratio=0.30, refine_steps=64)),
        # Block-reset TTT only (for comparison)
        ("ttt_br", "ttt_only", dict(
            param_type="q_proj", ttt_lr=3e-4,
            confidence_threshold=0.1, ttt_remask_ratio=0.1)),
        # Combo: best-of retry vs TTT
        ("combo_best_of", "retry_ttt", dict(
            n_retries=2, remask_ratio=0.30, refine_steps=32,
            ttt_lr=3e-4, confidence_threshold=0.1, ttt_remask_ratio=0.1)),
    ]

    for cfg_name, cfg_type, kwargs in configs:
        print(f"\n=== {cfg_name} ===")
        seed_results = []

        for seed in seeds:
            result = run_one_config(model, tokenizer, prompts, config, sampler,
                                     cfg_type, seed, device, **kwargs)
            seed_results.append(result)
            print(f"  Seed {seed}: PPL={result['mean']:.3f} (med={result['median']:.3f}, {result['gen_time']:.1f}s)")

        means = [r["mean"] for r in seed_results]
        medians = [r["median"] for r in seed_results]
        all_ppls = []
        for r in seed_results:
            all_ppls.extend(r["ppls"])

        all_results[cfg_name] = {
            "overall_mean": float(np.mean(means)),
            "overall_std": float(np.std(means)),
            "overall_median": float(np.mean(medians)),
            "seed_means": means,
            "all_ppls": all_ppls,
        }

        with open(RESULTS_DIR / "aca_dllm_v2_sweep.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary
    print("\n" + "=" * 85)
    print(f"{'Method':<20} {'Mean PPL':>9} {'± Std':>7} {'Median':>8} {'vs Van':>8} {'p-val':>7}")
    print("-" * 85)

    vanilla_means = all_results["vanilla"]["seed_means"]
    vanilla_ppls = all_results["vanilla"]["all_ppls"]

    for name, res in sorted(all_results.items(), key=lambda x: x[1]["overall_mean"]):
        delta = ((res["overall_mean"] - all_results["vanilla"]["overall_mean"]) /
                 all_results["vanilla"]["overall_mean"] * 100)
        if name != "vanilla" and len(res["all_ppls"]) == len(vanilla_ppls):
            _, p_val = scipy_stats.ttest_rel(vanilla_ppls, res["all_ppls"])
            sig = " **" if p_val < 0.01 else (" *" if p_val < 0.05 else "")
            print(f"  {name:<18} {res['overall_mean']:>9.3f} {res['overall_std']:>6.3f} "
                  f"{res['overall_median']:>8.3f} {delta:>+7.1f}% {p_val:>6.4f}{sig}")
        else:
            print(f"  {name:<18} {res['overall_mean']:>9.3f} {res['overall_std']:>6.3f} "
                  f"{res['overall_median']:>8.3f} {'base':>8}")
    print("=" * 85)


if __name__ == "__main__":
    run_experiment()
