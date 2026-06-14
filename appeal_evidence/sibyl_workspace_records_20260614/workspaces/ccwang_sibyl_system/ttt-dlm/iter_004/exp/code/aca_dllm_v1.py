"""
ACA-DLM v1: Adaptive Compute Allocation for Discrete Diffusion LMs.

Instead of TTT (adapting parameters), adapt the DENOISING PROCESS itself.
Three approaches:

1. **ReMask-Retry**: After initial denoising, re-mask low-confidence tokens
   and re-denoise them. ReMDM-inspired but simpler.

2. **Confidence-Guided Ordering**: Instead of unmasking by schedule order,
   unmask high-confidence tokens first, leaving uncertain ones for later
   steps where they have more context.

3. **Iterative Refinement**: After full denoising, evaluate coherence,
   re-mask problematic regions, and re-denoise with full context.

All are training-free and work with any pretrained masked DLM.
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
from ttt_dllm_v3_sweep import load_model
from ttt_dllm_v4_best import make_diverse_prompts


def remask_retry_sample(model, tokenizer, prompt, config,
                        n_retries=2, remask_ratio=0.3, remask_threshold=None):
    """
    ReMask-Retry: Standard denoising → re-mask low-confidence tokens → re-denoise.

    After initial denoising, identify tokens the model is least confident about,
    re-mask them, and run additional denoising steps to refine.
    """
    mask_id = tokenizer.mask_token_id
    eos_id = tokenizer.eos_token_id

    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    # Initial denoising
    with torch.no_grad():
        x = sampler.sample([prompt], config=config)

    prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=model.device)
    pl = prompt_t.shape[0]
    max_new_tokens = config.max_new_tokens
    gen_region = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    gen_region[pl:pl+max_new_tokens] = True

    stats = {"retries": 0, "tokens_remasked": 0}

    for retry in range(n_retries):
        # Get confidence for generated tokens
        attention_mask = torch.ones_like(x)
        with torch.no_grad():
            logits = model(x, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            # Confidence = probability of the actual token at each position
            token_conf = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)

        # Only consider generated region
        gen_conf = token_conf[0, gen_region]

        if remask_threshold is not None:
            # Re-mask tokens below confidence threshold
            low_conf_mask = (token_conf[0] < remask_threshold) & gen_region
        else:
            # Re-mask bottom remask_ratio fraction
            n_gen = gen_region.sum().item()
            n_remask = max(1, int(n_gen * remask_ratio))
            gen_indices = torch.where(gen_region)[0]
            gen_confs = token_conf[0, gen_indices]
            _, worst_idx = gen_confs.topk(n_remask, largest=False)
            low_conf_mask = torch.zeros_like(gen_region)
            low_conf_mask[gen_indices[worst_idx]] = True

        n_remasked = low_conf_mask.sum().item()
        if n_remasked == 0:
            break

        stats["tokens_remasked"] += n_remasked
        stats["retries"] += 1

        # Re-mask
        x[0, low_conf_mask] = mask_id

        # Re-denoise with fewer steps (refinement)
        refine_steps = max(16, config.steps // 4)
        refine_config = dllm.core.samplers.MDLMSamplerConfig(
            max_new_tokens=max_new_tokens, steps=refine_steps, temperature=config.temperature,
            remasking=config.remasking, block_size=config.block_size)

        # Manual refinement denoising (only on masked positions)
        scheduler = dllm.core.schedulers.LinearAlphaScheduler()
        ntt_mask = (x == mask_id)

        # Simple greedy refinement: just predict masked tokens
        for step in range(refine_steps):
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits

            still_masked = (x == mask_id)
            if not still_masked.any():
                break

            probs = F.softmax(logits.float(), dim=-1)
            # Unmask highest confidence tokens progressively
            pred_tokens = torch.argmax(logits, dim=-1)
            pred_conf = torch.gather(probs, dim=-1, index=pred_tokens.unsqueeze(-1)).squeeze(-1)
            pred_conf[~still_masked] = -float("inf")

            # Unmask top fraction per step
            n_still = still_masked.sum().item()
            n_unmask = max(1, n_still // max(1, refine_steps - step))

            _, top_idx = pred_conf[0].topk(min(n_unmask, n_still))
            x[0, top_idx] = pred_tokens[0, top_idx]

    return x, stats


def iterative_refine_sample(model, tokenizer, prompt, config,
                            n_rounds=3, remask_fraction=0.2):
    """
    Iterative Refinement: Full denoise → assess → re-mask worst → re-denoise.
    Each round has full context from previous tokens, enabling better predictions.
    """
    mask_id = tokenizer.mask_token_id

    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=model.device)
    pl = prompt_t.shape[0]
    max_new_tokens = config.max_new_tokens

    # Round 0: Initial generation
    with torch.no_grad():
        x = sampler.sample([prompt], config=config)

    gen_region = torch.zeros(x.shape[1], dtype=torch.bool, device=x.device)
    gen_region[pl:pl+max_new_tokens] = True
    attention_mask = torch.ones_like(x)

    stats = {"rounds": 0, "total_remasked": 0}

    for round_idx in range(n_rounds):
        # Assess confidence
        with torch.no_grad():
            logits = model(x, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=x.unsqueeze(-1)).squeeze(-1)

        # Find worst tokens in generated region
        n_gen = gen_region.sum().item()
        n_remask = max(1, int(n_gen * remask_fraction))
        gen_indices = torch.where(gen_region)[0]
        gen_confs = token_conf[0, gen_indices]
        _, worst_idx = gen_confs.topk(n_remask, largest=False)

        # Check if minimum confidence is already high
        min_conf = gen_confs.min().item()
        if min_conf > 0.8:  # Already confident everywhere
            break

        # Re-mask worst tokens
        remask_positions = gen_indices[worst_idx]
        x[0, remask_positions] = mask_id
        stats["total_remasked"] += n_remask
        stats["rounds"] += 1

        # Re-denoise (with full context from good tokens)
        refine_steps = max(8, config.steps // 4)
        for step in range(refine_steps):
            with torch.no_grad():
                logits = model(x, attention_mask=attention_mask).logits

            still_masked = (x == mask_id)
            if not still_masked.any():
                break

            pred = torch.argmax(add_gumbel_noise(logits, temperature=config.temperature), dim=-1)
            probs = F.softmax(logits.float(), dim=-1)
            pred_conf = torch.gather(probs, dim=-1, index=pred.unsqueeze(-1)).squeeze(-1)
            pred_conf[~still_masked] = -float("inf")

            n_still = still_masked.sum().item()
            n_unmask = max(1, n_still // max(1, refine_steps - step))
            _, top_idx = pred_conf[0].topk(min(n_unmask, n_still))
            x[0, top_idx] = pred[0, top_idx]

    return x, stats


def run_experiment():
    from transformers import AutoModelForCausalLM

    print("=" * 70)
    print("ACA-DLM v1: Adaptive Compute Allocation")
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

    all_results = {}
    seeds = [42, 123, 456]  # 3 seeds for significance

    experiment_configs = [
        ("vanilla", None, {}),
        # ReMask-Retry variants
        ("retry_1x_30pct", "remask_retry", dict(n_retries=1, remask_ratio=0.3)),
        ("retry_2x_30pct", "remask_retry", dict(n_retries=2, remask_ratio=0.3)),
        ("retry_2x_20pct", "remask_retry", dict(n_retries=2, remask_ratio=0.2)),
        ("retry_2x_10pct", "remask_retry", dict(n_retries=2, remask_ratio=0.1)),
        ("retry_thresh_05", "remask_retry", dict(n_retries=2, remask_threshold=0.5)),
        # Iterative refinement
        ("refine_2r_20pct", "iterative", dict(n_rounds=2, remask_fraction=0.2)),
        ("refine_3r_20pct", "iterative", dict(n_rounds=3, remask_fraction=0.2)),
        ("refine_3r_10pct", "iterative", dict(n_rounds=3, remask_fraction=0.1)),
        ("refine_5r_20pct", "iterative", dict(n_rounds=5, remask_fraction=0.2)),
    ]

    for cfg_name, cfg_type, kwargs in experiment_configs:
        print(f"\n=== {cfg_name} ===")
        seed_results = []

        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            samples = []
            total_stats = {}
            t0 = time.time()

            for prompt in prompts:
                if cfg_type is None:
                    with torch.no_grad():
                        seq = sampler.sample([prompt], config=config)
                    text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
                    samples.append(text)
                elif cfg_type == "remask_retry":
                    seq, stats = remask_retry_sample(model, tokenizer, prompt, config, **kwargs)
                    text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
                    samples.append(text)
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            total_stats[k] = total_stats.get(k, 0) + v
                elif cfg_type == "iterative":
                    seq, stats = iterative_refine_sample(model, tokenizer, prompt, config, **kwargs)
                    text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
                    samples.append(text)
                    for k, v in stats.items():
                        if isinstance(v, (int, float)):
                            total_stats[k] = total_stats.get(k, 0) + v

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

            result = {
                "mean": float(np.mean(ppls)) if ppls else float("inf"),
                "median": float(np.median(ppls)) if ppls else float("inf"),
                "std": float(np.std(ppls)) if ppls else 0,
                "n": len(ppls),
                "gen_time": gen_time,
            }
            seed_results.append(result)
            print(f"  Seed {seed}: PPL={result['mean']:.3f} (med={result['median']:.3f}, {gen_time:.1f}s)")

        # Aggregate
        means = [r["mean"] for r in seed_results]
        medians = [r["median"] for r in seed_results]
        all_results[cfg_name] = {
            "overall_mean": float(np.mean(means)),
            "overall_std": float(np.std(means)),
            "overall_median": float(np.mean(medians)),
            "seed_means": means,
            "seed_medians": medians,
        }

        # Save incrementally
        with open(RESULTS_DIR / "aca_dllm_v1.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary with significance
    from scipy import stats as scipy_stats

    print("\n" + "=" * 80)
    print(f"{'Method':<25} {'Mean PPL':>9} {'± Std':>7} {'Median':>8} {'vs Van':>8} {'p-val':>7}")
    print("-" * 80)

    vanilla_means = all_results["vanilla"]["seed_means"]
    for name, res in sorted(all_results.items(), key=lambda x: x[1]["overall_mean"]):
        delta = ((res["overall_mean"] - all_results["vanilla"]["overall_mean"]) /
                 all_results["vanilla"]["overall_mean"] * 100)
        if name != "vanilla":
            _, p_val = scipy_stats.ttest_rel(vanilla_means, res["seed_means"])
            sig = " *" if p_val < 0.05 else ""
            print(f"  {name:<23} {res['overall_mean']:>9.3f} {res['overall_std']:>6.3f} "
                  f"{res['overall_median']:>8.3f} {delta:>+7.1f}% {p_val:>6.3f}{sig}")
        else:
            print(f"  {name:<23} {res['overall_mean']:>9.3f} {res['overall_std']:>6.3f} "
                  f"{res['overall_median']:>8.3f} {'base':>8}")
    print("=" * 80)


if __name__ == "__main__":
    run_experiment()
