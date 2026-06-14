"""
ACA-DLM v3: Text quality evaluation beyond PPL.

Metrics:
1. PPL (autoregressive model)
2. Self-BLEU (diversity: generate 3x, measure overlap - lower = more diverse)
3. Model confidence (average softmax probability of generated tokens)
4. Distinct-n (lexical diversity)
5. Length (valid output length)
"""
import os, sys, json, time, math
from pathlib import Path
from types import SimpleNamespace
from collections import Counter

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


def distinct_n(text, n=2):
    """Compute distinct-n: ratio of unique n-grams to total n-grams."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return len(set(ngrams)) / len(ngrams)


def compute_self_bleu(texts):
    """Compute average pairwise BLEU among multiple generations for same prompt."""
    import sacrebleu
    if len(texts) < 2:
        return 0.0
    scores = []
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            try:
                bleu = sacrebleu.sentence_bleu(texts[i], [texts[j]])
                scores.append(bleu.score)
            except:
                pass
    return float(np.mean(scores)) if scores else 0.0


def run_experiment():
    from transformers import AutoModelForCausalLM

    print("=" * 70)
    print("ACA-DLM v3: Multi-Metric Text Quality Evaluation")
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
    methods = {
        "vanilla": (None, {}),
        "retry_30pct": ("retry", dict(n_retries=2, remask_ratio=0.3, refine_steps=32)),
        "retry_50pct": ("retry", dict(n_retries=2, remask_ratio=0.5, refine_steps=32)),
        "retry_70pct": ("retry", dict(n_retries=2, remask_ratio=0.7, refine_steps=32)),
    }

    all_results = {}

    for method_name, (method_type, kwargs) in methods.items():
        print(f"\n=== {method_name} ===")

        all_ppls = []
        all_confs = []
        all_d1 = []
        all_d2 = []
        all_lengths = []
        # For self-BLEU: generate 3x per prompt with different seeds
        prompt_generations = {i: [] for i in range(32)}

        for seed in seeds:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            texts = []
            confs = []

            for pi, prompt in enumerate(prompts):
                if method_type is None:
                    with torch.no_grad():
                        seq = sampler.sample([prompt], config=config)
                else:
                    seq, _ = remask_retry_v2(model, tokenizer, prompt, config, **kwargs)

                text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
                texts.append(text)
                prompt_generations[pi].append(text)

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

                # Distinct-n
                all_d1.append(distinct_n(text, 1))
                all_d2.append(distinct_n(text, 2))
                all_lengths.append(len(text.split()))

            # PPL evaluation
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

            print(f"  Seed {seed}: PPL={np.mean(valid_ppls):.3f}, "
                  f"Conf={np.mean(confs):.4f}")

        # Self-BLEU (lower = more diverse across seeds)
        self_bleus = []
        for pi in range(32):
            sb = compute_self_bleu(prompt_generations[pi])
            self_bleus.append(sb)

        all_results[method_name] = {
            "ppl_mean": float(np.mean(all_ppls)),
            "ppl_std": float(np.std(all_ppls)),
            "ppl_median": float(np.median(all_ppls)),
            "confidence_mean": float(np.mean(all_confs)),
            "distinct_1": float(np.mean(all_d1)),
            "distinct_2": float(np.mean(all_d2)),
            "self_bleu_mean": float(np.mean(self_bleus)),
            "avg_length": float(np.mean(all_lengths)),
            "all_ppls": all_ppls,
        }

    # Summary
    print("\n" + "=" * 90)
    print(f"{'Method':<15} {'PPL':>7} {'Conf':>7} {'D-1':>6} {'D-2':>6} {'SelfBLEU':>9} {'AvgLen':>7} {'PPL Δ':>7}")
    print("-" * 90)
    vanilla = all_results["vanilla"]
    for name, res in sorted(all_results.items(), key=lambda x: x[1]["ppl_mean"]):
        delta = ((res["ppl_mean"] - vanilla["ppl_mean"]) / vanilla["ppl_mean"] * 100)

        # Significance
        if name != "vanilla":
            _, p_val = scipy_stats.ttest_rel(vanilla["all_ppls"], res["all_ppls"])
            sig = f"p={p_val:.5f}"
        else:
            sig = ""

        print(f"  {name:<13} {res['ppl_mean']:>7.3f} {res['confidence_mean']:>7.4f} "
              f"{res['distinct_1']:>6.3f} {res['distinct_2']:>6.3f} "
              f"{res['self_bleu_mean']:>9.1f} {res['avg_length']:>7.1f} {delta:>+6.1f}% {sig}")
    print("=" * 90)

    with open(RESULTS_DIR / "aca_dllm_v3_quality.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)


if __name__ == "__main__":
    run_experiment()
