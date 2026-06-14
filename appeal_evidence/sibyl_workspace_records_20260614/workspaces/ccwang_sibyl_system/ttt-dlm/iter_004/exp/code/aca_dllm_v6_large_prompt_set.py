"""
ACA-DLM v6: Large prompt set evaluation (256 prompts).

Uses WikiText-103 test set prefixes for standardized evaluation.
Runs vanilla and retry_70pct with 1 seed for fast turnaround.
If successful, can add more seeds.
"""
import os, sys, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats as scipy_stats

sys.path.insert(0, "/home/ccwang/sibyl_system/src/dllm")
sys.path.insert(0, "/home/ccwang/sibyl_system/exp/code")
import dllm
from ttt_dllm_v3_sweep import load_model
from aca_dllm_v2_sweep import remask_retry_v2

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def make_large_prompt_set(tokenizer, n=256, min_tokens=20, max_tokens=40):
    """Extract n diverse prompts from WikiText-103 or fallback to synthetic."""
    prompts = []

    # Try WikiText-103
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
        random.seed(42)
        random.shuffle(texts)

        for text in texts:
            if len(prompts) >= n:
                break
            # Take first 20-40 tokens as prompt
            tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
            if len(tokens) >= min_tokens:
                prompt_len = min(max_tokens, len(tokens) // 2)
                prompt_tokens = tokens[:prompt_len]
                # Verify it decodes cleanly
                decoded = tokenizer.decode(prompt_tokens)
                if len(decoded.strip()) > 20:
                    prompts.append(prompt_tokens)

        print(f"Loaded {len(prompts)} prompts from WikiText-103")
    except Exception as e:
        print(f"WikiText-103 failed ({e}), using synthetic prompts")

    # Fallback: generate diverse prompts synthetically
    if len(prompts) < n:
        # Diverse question templates
        topics = [
            "science", "history", "technology", "nature", "mathematics",
            "philosophy", "economics", "psychology", "medicine", "geography",
            "art", "music", "literature", "physics", "chemistry",
            "biology", "astronomy", "politics", "law", "education",
            "engineering", "architecture", "sports", "cooking", "language",
        ]
        templates = [
            "Explain the concept of {topic} and its importance in modern society.",
            "What are the key developments in {topic} during the past decade?",
            "Describe how {topic} has influenced our understanding of the world.",
            "The relationship between {topic} and daily life is often overlooked.",
            "In recent years, advances in {topic} have led to significant changes.",
            "Understanding {topic} requires knowledge of several fundamental principles.",
            "The history of {topic} reveals fascinating patterns of human innovation.",
            "Researchers in {topic} have discovered several surprising findings.",
            "One of the most important aspects of {topic} is its practical applications.",
            "The future of {topic} depends on continued investment in research.",
        ]

        while len(prompts) < n:
            topic = topics[len(prompts) % len(topics)]
            template = templates[len(prompts) % len(templates)]
            text = template.format(topic=topic)
            tokens = tokenizer.encode(text, add_special_tokens=False)[:max_tokens]
            prompts.append(tokens)

    return prompts[:n]


def run_experiment():
    from transformers import AutoModelForCausalLM

    print("=" * 70)
    print("ACA-DLM v6: Large Prompt Set (256 prompts)")
    print("=" * 70)

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_large_prompt_set(tokenizer, n=256)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    methods = {
        "vanilla": None,
        "retry_70pct": dict(n_retries=2, remask_ratio=0.7, refine_steps=32),
    }

    for method_name, retry_kwargs in methods.items():
        print(f"\n{'='*50}")
        print(f"Method: {method_name} (256 prompts, seed={seed})")
        print(f"{'='*50}")

        texts = []
        confs = []

        for pi, prompt in enumerate(prompts):
            if retry_kwargs is None:
                sampler = dllm.core.samplers.MDLMSampler(
                    model=model, tokenizer=tokenizer,
                    scheduler=dllm.core.schedulers.LinearAlphaScheduler())
                with torch.no_grad():
                    seq = sampler.sample([prompt], config=config)
            else:
                seq, _ = remask_retry_v2(model, tokenizer, prompt, config, **retry_kwargs)

            text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
            texts.append(text)

            # Confidence
            attention_mask = torch.ones_like(seq)
            with torch.no_grad():
                logits = model(seq, attention_mask=attention_mask).logits
                probs = F.softmax(logits.float(), dim=-1)
                token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)
            prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=device)
            pl = prompt_t.shape[0]
            gen_conf = token_conf[0, pl:pl+config.max_new_tokens].mean().item()
            confs.append(gen_conf)

            if (pi + 1) % 32 == 0:
                print(f"  {pi+1}/256 done, avg_conf={np.mean(confs):.4f}")

            # Save every 64 prompts
            if (pi + 1) % 64 == 0:
                partial = {
                    "method": method_name, "seed": seed,
                    "n_done": pi + 1, "conf_mean": float(np.mean(confs)),
                }
                with open(RESULTS_DIR / f"v6_partial_{method_name}.json", "w") as f:
                    json.dump(partial, f, indent=2)

        # PPL evaluation
        print(f"  Evaluating PPL for {len(texts)} texts...")
        model.cpu()
        torch.cuda.empty_cache()
        eval_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, trust_remote_code=True).to(device)
        eval_model.eval()

        ppls = []
        for ti, text in enumerate(texts):
            if len(text) < 10:
                ppls.append(None)
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = eval_model(**enc, labels=enc["input_ids"])
            ppls.append(math.exp(min(out.loss.item(), 20)))

            if (ti + 1) % 64 == 0:
                valid = [p for p in ppls if p is not None]
                print(f"  PPL eval: {ti+1}/256, running mean={np.mean(valid):.3f}")

        del eval_model
        torch.cuda.empty_cache()
        model.to(device)

        valid_ppls = [p for p in ppls if p is not None]
        result = {
            "method": method_name,
            "seed": seed,
            "n_prompts": len(prompts),
            "n_valid": len(valid_ppls),
            "ppl_mean": float(np.mean(valid_ppls)),
            "ppl_std": float(np.std(valid_ppls)),
            "ppl_median": float(np.median(valid_ppls)),
            "conf_mean": float(np.mean(confs)),
            "all_ppls": [float(p) for p in valid_ppls],
        }

        with open(RESULTS_DIR / f"v6_{method_name}.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"\n  {method_name}: PPL={result['ppl_mean']:.3f} ± {result['ppl_std']:.3f}, "
              f"Conf={result['conf_mean']:.4f}, N={result['n_valid']}")

    # Comparison
    with open(RESULTS_DIR / "v6_vanilla.json") as f:
        vanilla = json.load(f)
    with open(RESULTS_DIR / "v6_retry_70pct.json") as f:
        retry = json.load(f)

    delta = (retry["ppl_mean"] - vanilla["ppl_mean"]) / vanilla["ppl_mean"] * 100
    _, p_val = scipy_stats.ttest_rel(vanilla["all_ppls"][:min(len(vanilla["all_ppls"]), len(retry["all_ppls"]))],
                                      retry["all_ppls"][:min(len(vanilla["all_ppls"]), len(retry["all_ppls"]))])

    print(f"\n{'='*70}")
    print(f"COMPARISON (256 prompts)")
    print(f"  Vanilla:    PPL={vanilla['ppl_mean']:.3f}")
    print(f"  Retry 70%:  PPL={retry['ppl_mean']:.3f}")
    print(f"  Delta:      {delta:+.1f}%")
    print(f"  p-value:    {p_val:.8f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    run_experiment()
