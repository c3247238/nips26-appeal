"""
ACA-DLM v5: Token-level analysis of what gets remasked.

Analyzes:
1. Confidence distribution before/after retry
2. Token position analysis: where do low-confidence tokens cluster?
3. Token type analysis: function words vs content words
4. How many tokens actually change during retry
5. Confidence calibration: are low-confidence tokens actually wrong?
"""
import os, sys, json, math
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ccwang/sibyl_system/src/dllm")
sys.path.insert(0, "/home/ccwang/sibyl_system/exp/code")
import dllm
from ttt_dllm_v3_sweep import load_model
from ttt_dllm_v4_best import make_diverse_prompts

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Common English function words
FUNCTION_WORDS = set([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "can", "could", "must", "ought",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those", "who", "whom", "which", "what",
    "and", "but", "or", "nor", "for", "yet", "so", "if", "then",
    "of", "in", "to", "on", "at", "by", "with", "from", "up", "about",
    "into", "over", "after", "not", "no", "as", "than", "too", "very",
])


def analyze_remasking(model, tokenizer, prompt, config, remask_ratio=0.7, refine_steps=32):
    """Run ReMask-Retry with detailed token-level tracking."""
    device = next(model.parameters()).device

    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    # Phase 1: Standard denoising
    with torch.no_grad():
        seq_before = sampler.sample([prompt], config=config)

    # Get confidence for all generated positions
    prompt_len = len(prompt)
    gen_len = config.max_new_tokens

    attention_mask = torch.ones_like(seq_before)
    with torch.no_grad():
        logits_before = model(seq_before, attention_mask=attention_mask).logits
        probs_before = F.softmax(logits_before.float(), dim=-1)
        confs_before = torch.gather(probs_before, dim=-1, index=seq_before.unsqueeze(-1)).squeeze(-1)

    gen_confs = confs_before[0, prompt_len:prompt_len+gen_len].cpu().numpy()
    gen_tokens_before = seq_before[0, prompt_len:prompt_len+gen_len].cpu().numpy()

    # Phase 2: Identify low-confidence tokens
    n_remask = int(remask_ratio * gen_len)
    sorted_indices = np.argsort(gen_confs)
    remask_positions = sorted_indices[:n_remask]
    retain_positions = sorted_indices[n_remask:]

    # Phase 3: Remask and re-denoise
    seq_retry = seq_before.clone()
    mask_token_id = tokenizer.encode("[MASK]", add_special_tokens=False)
    if not mask_token_id:
        mask_token_id = tokenizer.encode("<mask>", add_special_tokens=False)
    if not mask_token_id:
        # Try common mask tokens
        for mt in ["[MASK]", "<mask>", "<|mask|>"]:
            ids = tokenizer.encode(mt, add_special_tokens=False)
            if len(ids) == 1:
                mask_token_id = ids
                break
    if not mask_token_id:
        mask_token_id = [tokenizer.mask_token_id] if hasattr(tokenizer, 'mask_token_id') and tokenizer.mask_token_id else [0]

    mtid = mask_token_id[0]
    for pos in remask_positions:
        seq_retry[0, prompt_len + pos] = mtid

    # Re-denoise
    refine_config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=gen_len, steps=refine_steps, temperature=0.2,
        remasking="low_confidence", block_size=32)

    with torch.no_grad():
        seq_after = sampler.sample([prompt], config=refine_config,
                                    x_initial=seq_retry)

    # Get confidence after retry
    attention_mask_after = torch.ones_like(seq_after)
    with torch.no_grad():
        logits_after = model(seq_after, attention_mask=attention_mask_after).logits
        probs_after = F.softmax(logits_after.float(), dim=-1)
        confs_after = torch.gather(probs_after, dim=-1, index=seq_after.unsqueeze(-1)).squeeze(-1)

    gen_confs_after = confs_after[0, prompt_len:prompt_len+gen_len].cpu().numpy()
    gen_tokens_after = seq_after[0, prompt_len:prompt_len+gen_len].cpu().numpy()

    # Analysis
    result = {
        "confs_before": gen_confs.tolist(),
        "confs_after": gen_confs_after.tolist(),
        "tokens_before": gen_tokens_before.tolist(),
        "tokens_after": gen_tokens_after.tolist(),
        "remask_positions": remask_positions.tolist(),
        "retain_positions": retain_positions.tolist(),
    }

    # Token change analysis
    n_changed = 0
    changed_positions = []
    for pos in remask_positions:
        if gen_tokens_before[pos] != gen_tokens_after[pos]:
            n_changed += 1
            changed_positions.append(int(pos))

    result["n_remasked"] = int(n_remask)
    result["n_changed"] = n_changed
    result["change_rate"] = n_changed / max(n_remask, 1)

    # Confidence improvement at remasked positions
    remask_conf_before = gen_confs[remask_positions].mean()
    remask_conf_after = gen_confs_after[remask_positions].mean()
    result["remask_conf_before"] = float(remask_conf_before)
    result["remask_conf_after"] = float(remask_conf_after)

    # Position distribution analysis
    result["remask_position_mean"] = float(np.mean(remask_positions))
    result["remask_position_std"] = float(np.std(remask_positions))
    result["gen_len"] = int(gen_len)

    # Token type analysis
    remasked_tokens = [tokenizer.decode([int(gen_tokens_before[p])]).strip().lower() for p in remask_positions]
    retained_tokens = [tokenizer.decode([int(gen_tokens_before[p])]).strip().lower() for p in retain_positions]

    n_func_remasked = sum(1 for t in remasked_tokens if t in FUNCTION_WORDS)
    n_func_retained = sum(1 for t in retained_tokens if t in FUNCTION_WORDS)

    result["func_word_ratio_remasked"] = n_func_remasked / max(len(remasked_tokens), 1)
    result["func_word_ratio_retained"] = n_func_retained / max(len(retained_tokens), 1)

    return result


def run_experiment():
    print("=" * 70)
    print("ACA-DLM v5: Token-Level Analysis of ReMask-Retry")
    print("=" * 70)

    model, tokenizer = load_model()
    prompts = make_diverse_prompts(tokenizer, n=32)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    all_results = []
    seeds = [42, 123]  # 2 seeds for analysis (less compute needed)

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        for pi, prompt in enumerate(prompts):
            result = analyze_remasking(model, tokenizer, prompt, config,
                                       remask_ratio=0.7, refine_steps=32)
            result["seed"] = seed
            result["prompt_idx"] = pi
            all_results.append(result)

            if (pi + 1) % 8 == 0:
                print(f"  Seed {seed}: {pi+1}/32 done")

        # Incremental save
        with open(RESULTS_DIR / "v5_token_analysis_partial.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Summary statistics
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_conf_before = [np.mean(r["confs_before"]) for r in all_results]
    all_conf_after = [np.mean(r["confs_after"]) for r in all_results]
    all_change_rates = [r["change_rate"] for r in all_results]
    all_remask_conf_before = [r["remask_conf_before"] for r in all_results]
    all_remask_conf_after = [r["remask_conf_after"] for r in all_results]
    all_func_remasked = [r["func_word_ratio_remasked"] for r in all_results]
    all_func_retained = [r["func_word_ratio_retained"] for r in all_results]
    all_pos_means = [r["remask_position_mean"] / r["gen_len"] for r in all_results]

    print(f"\n1. CONFIDENCE DISTRIBUTION")
    print(f"   Avg confidence (all tokens):")
    print(f"     Before retry: {np.mean(all_conf_before):.4f}")
    print(f"     After retry:  {np.mean(all_conf_after):.4f}")
    print(f"   Avg confidence (remasked tokens only):")
    print(f"     Before retry: {np.mean(all_remask_conf_before):.4f}")
    print(f"     After retry:  {np.mean(all_remask_conf_after):.4f}")
    print(f"     Improvement:  {np.mean(all_remask_conf_after)/np.mean(all_remask_conf_before):.1f}x")

    print(f"\n2. TOKEN CHANGE RATE")
    print(f"   Of {int(0.7*256)} remasked tokens:")
    print(f"     Changed:  {np.mean(all_change_rates)*100:.1f}% ({np.mean(all_change_rates)*0.7*256:.0f} tokens)")
    print(f"     Kept same: {(1-np.mean(all_change_rates))*100:.1f}%")

    print(f"\n3. POSITION DISTRIBUTION")
    print(f"   Remasked token position (relative):")
    print(f"     Mean: {np.mean(all_pos_means):.3f} (0=start, 1=end)")
    print(f"   {'Early tokens remasked more' if np.mean(all_pos_means) < 0.5 else 'Late tokens remasked more'}")

    print(f"\n4. FUNCTION vs CONTENT WORDS")
    print(f"   Function word ratio in remasked: {np.mean(all_func_remasked):.3f}")
    print(f"   Function word ratio in retained: {np.mean(all_func_retained):.3f}")
    ratio = np.mean(all_func_remasked) / max(np.mean(all_func_retained), 0.001)
    if ratio > 1.2:
        print(f"   -> Function words are {ratio:.1f}x MORE likely to be remasked")
    elif ratio < 0.8:
        print(f"   -> Content words are {1/ratio:.1f}x MORE likely to be remasked")
    else:
        print(f"   -> No strong preference (ratio: {ratio:.2f})")

    # Confidence histogram bins
    print(f"\n5. CONFIDENCE HISTOGRAM (before retry)")
    conf_flat = np.concatenate([r["confs_before"] for r in all_results])
    bins = [0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    hist, _ = np.histogram(conf_flat, bins=bins)
    total = len(conf_flat)
    for i in range(len(bins)-1):
        pct = hist[i] / total * 100
        bar = "#" * int(pct)
        print(f"   [{bins[i]:.2f}-{bins[i+1]:.2f}): {pct:5.1f}% {bar}")

    with open(RESULTS_DIR / "v5_token_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved to {RESULTS_DIR}/v5_token_analysis.json")


if __name__ == "__main__":
    run_experiment()
