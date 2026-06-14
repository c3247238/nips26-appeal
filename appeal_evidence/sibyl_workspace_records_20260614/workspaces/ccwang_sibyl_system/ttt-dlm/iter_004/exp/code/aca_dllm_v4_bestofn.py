"""
ACA-DLM v4: Best-of-N Baseline Comparison

Critical baseline: at 1.28x compute, generate ~1.28 sequences and pick best.
Does ReMask-Retry beat Best-of-N at matched compute budgets?

Also: Best-of-N at 2x and 3x compute for scaling comparison.
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


def best_of_n_sample(model, tokenizer, prompt, config, n=2, device=None):
    """Generate N sequences and return the one with lowest PPL."""
    if device is None:
        device = next(model.parameters()).device

    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    candidates = []
    for i in range(n):
        with torch.no_grad():
            seq = sampler.sample([prompt], config=config)
        # Compute model's own log-likelihood as proxy for quality
        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)

        prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=device)
        pl = prompt_t.shape[0]
        gen_conf = token_conf[0, pl:pl+config.max_new_tokens].mean().item()
        candidates.append((seq, gen_conf))

    # Pick highest confidence (proxy for best quality)
    best_seq = max(candidates, key=lambda x: x[1])[0]
    return best_seq


def run_experiment():
    from transformers import AutoModelForCausalLM

    print("=" * 70)
    print("ACA-DLM v4: Best-of-N Baseline Comparison")
    print("=" * 70)

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_diverse_prompts(tokenizer, n=32)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    seeds = [42, 123, 456]

    # Methods to compare at matched compute budgets:
    # - vanilla: 1.0x compute (128 steps)
    # - best_of_2: 2.0x compute (generate 2, pick best)
    # - best_of_3: 3.0x compute
    # - retry_30pct: ~1.28x compute (128 + 32 refine steps)
    # - retry_70pct: ~1.28x compute (128 + 32 refine steps)
    # - best_of_2_retry: 2.56x compute (2x retry)
    methods = {
        "vanilla": {"type": "vanilla"},
        "best_of_2": {"type": "best_of_n", "n": 2},
        "best_of_3": {"type": "best_of_n", "n": 3},
        "retry_30pct": {"type": "retry", "n_retries": 2, "remask_ratio": 0.3, "refine_steps": 32},
        "retry_70pct": {"type": "retry", "n_retries": 2, "remask_ratio": 0.7, "refine_steps": 32},
    }

    all_results = {}

    for method_name, method_cfg in methods.items():
        print(f"\n{'='*50}")
        print(f"Method: {method_name}")
        print(f"{'='*50}")

        all_ppls = []
        all_confs = []
        all_texts = []

        for seed in seeds:
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

                # Model confidence
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
                    print(f"  Seed {seed}: {pi+1}/32 done")

            # Phase 2: PPL evaluation with AR model
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
            model.to(device)

            valid_ppls = [p for p in ppls if p is not None]
            all_ppls.extend(valid_ppls)
            all_confs.extend(confs)
            all_texts.extend(texts)

            print(f"  Seed {seed}: PPL={np.mean(valid_ppls):.3f}, Conf={np.mean(confs):.4f}")

            # Incremental save
            partial = {
                "method": method_name,
                "seed": seed,
                "ppl_mean": float(np.mean(valid_ppls)),
                "conf_mean": float(np.mean(confs)),
            }
            with open(RESULTS_DIR / f"v4_partial_{method_name}_s{seed}.json", "w") as f:
                json.dump(partial, f, indent=2)

        all_results[method_name] = {
            "ppl_mean": float(np.mean(all_ppls)),
            "ppl_std": float(np.std(all_ppls)),
            "ppl_median": float(np.median(all_ppls)),
            "confidence_mean": float(np.mean(all_confs)),
            "n_samples": len(all_ppls),
            "all_ppls": [float(p) for p in all_ppls],
        }

    # Summary table
    print("\n" + "=" * 90)
    print(f"{'Method':<20} {'PPL':>7} {'±std':>7} {'Conf':>7} {'Δ vs Vanilla':>14} {'p-value':>12} {'Compute':>10}")
    print("-" * 90)

    vanilla_ppls = all_results["vanilla"]["all_ppls"]
    compute_cost = {
        "vanilla": "1.0x",
        "best_of_2": "2.0x",
        "best_of_3": "3.0x",
        "retry_30pct": "1.28x",
        "retry_70pct": "1.28x",
    }

    for name in ["vanilla", "retry_30pct", "retry_70pct", "best_of_2", "best_of_3"]:
        res = all_results[name]
        delta = ((res["ppl_mean"] - all_results["vanilla"]["ppl_mean"]) / all_results["vanilla"]["ppl_mean"] * 100)

        if name != "vanilla":
            _, p_val = scipy_stats.ttest_rel(vanilla_ppls[:len(res["all_ppls"])], res["all_ppls"])
            sig = f"p={p_val:.6f}"
        else:
            p_val = 1.0
            sig = ""

        cost = compute_cost.get(name, "?")
        print(f"  {name:<18} {res['ppl_mean']:>7.3f} {res['ppl_std']:>7.3f} "
              f"{res['confidence_mean']:>7.4f} {delta:>+13.1f}% {sig:>12} {cost:>10}")

    print("=" * 90)

    # Compute-normalized comparison
    print("\n--- Compute-Normalized Comparison ---")
    print("At ~1.28x compute: retry_30pct vs vanilla")
    print("At ~2.0x compute:  best_of_2 vs retry_70pct (retry is 1.28x, so BoN gets 2x)")
    print("At ~3.0x compute:  best_of_3")

    with open(RESULTS_DIR / "aca_dllm_v4_bestofn.json", "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != "all_ppls"}
                   for k, v in all_results.items()}, f, indent=2)

    # Also save full data for later analysis
    with open(RESULTS_DIR / "aca_dllm_v4_bestofn_full.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/aca_dllm_v4_bestofn*.json")


if __name__ == "__main__":
    run_experiment()
