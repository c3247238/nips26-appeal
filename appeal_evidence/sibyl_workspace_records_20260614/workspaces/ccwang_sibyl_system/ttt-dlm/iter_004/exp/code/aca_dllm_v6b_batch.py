"""
ACA-DLM v6b: Large prompt set - batch resumable version.
Processes BATCH_SIZE prompts per invocation, saves progress.
Call repeatedly to complete all 256 prompts.

Usage: python aca_dllm_v6b_batch.py [method] [batch_size]
  method: vanilla or retry_70pct (default: auto-detect next)
  batch_size: number of prompts per batch (default: 64)
"""
import os, sys, json, math, random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ccwang/sibyl_system/src/dllm")
sys.path.insert(0, "/home/ccwang/sibyl_system/exp/code")
import dllm
from ttt_dllm_v3_sweep import load_model
from aca_dllm_v2_sweep import remask_retry_v2

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PROMPTS = 256
SEED = 42


def make_large_prompt_set(tokenizer, n=256, min_tokens=20, max_tokens=40):
    prompts = []
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
        texts = [t for t in ds["text"] if len(t.strip()) > 100]
        random.seed(42)
        random.shuffle(texts)
        for text in texts:
            if len(prompts) >= n:
                break
            tokens = tokenizer.encode(text.strip(), add_special_tokens=False)
            if len(tokens) >= min_tokens:
                prompt_len = min(max_tokens, len(tokens) // 2)
                prompt_tokens = tokens[:prompt_len]
                decoded = tokenizer.decode(prompt_tokens)
                if len(decoded.strip()) > 20:
                    prompts.append(prompt_tokens)
        print(f"Loaded {len(prompts)} prompts from WikiText-103")
    except Exception as e:
        print(f"WikiText-103 failed ({e}), using synthetic")
        topics = ["science","history","technology","nature","mathematics",
                  "philosophy","economics","psychology","medicine","geography"]
        templates = [
            "Explain the concept of {t} and its importance.",
            "What are the key developments in {t}?",
            "The relationship between {t} and daily life.",
            "Understanding {t} requires fundamental knowledge.",
            "Researchers in {t} have discovered findings.",
        ]
        while len(prompts) < n:
            t = topics[len(prompts) % len(topics)]
            tmpl = templates[len(prompts) % len(templates)]
            tokens = tokenizer.encode(tmpl.format(t=t), add_special_tokens=False)[:max_tokens]
            prompts.append(tokens)
    return prompts[:n]


def load_progress(method_name):
    """Load existing progress for a method."""
    fname = RESULTS_DIR / f"v6b_{method_name}_progress.json"
    if fname.exists():
        with open(fname) as f:
            return json.load(f)
    return {"texts": [], "confs": [], "n_done": 0}


def save_progress(method_name, progress):
    fname = RESULTS_DIR / f"v6b_{method_name}_progress.json"
    with open(fname, "w") as f:
        json.dump(progress, f)


def run_generation_batch(method_name, batch_size=64):
    """Run one batch of generation for a method."""
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_large_prompt_set(tokenizer, n=N_PROMPTS)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    progress = load_progress(method_name)
    start_idx = progress["n_done"]

    if start_idx >= N_PROMPTS:
        print(f"{method_name}: All {N_PROMPTS} prompts done, skipping generation")
        return progress

    end_idx = min(start_idx + batch_size, N_PROMPTS)
    print(f"\n{method_name}: Processing prompts {start_idx}-{end_idx-1} of {N_PROMPTS}")

    torch.manual_seed(SEED + start_idx)  # Deterministic per batch
    torch.cuda.manual_seed(SEED + start_idx)

    for pi in range(start_idx, end_idx):
        prompt = prompts[pi]

        if method_name == "vanilla":
            sampler = dllm.core.samplers.MDLMSampler(
                model=model, tokenizer=tokenizer,
                scheduler=dllm.core.schedulers.LinearAlphaScheduler())
            with torch.no_grad():
                seq = sampler.sample([prompt], config=config)
        elif method_name == "retry_70pct":
            seq, _ = remask_retry_v2(model, tokenizer, prompt, config,
                                     n_retries=2, remask_ratio=0.7, refine_steps=32)

        text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
        progress["texts"].append(text)

        # Confidence
        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)
        prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=device)
        pl = prompt_t.shape[0]
        gen_conf = token_conf[0, pl:pl+config.max_new_tokens].mean().item()
        progress["confs"].append(gen_conf)

        if (pi - start_idx + 1) % 16 == 0:
            print(f"  {pi+1}/{N_PROMPTS} done")

    progress["n_done"] = end_idx
    save_progress(method_name, progress)
    print(f"  Batch complete. Total done: {end_idx}/{N_PROMPTS}")
    return progress


def run_ppl_eval(method_name):
    """Evaluate PPL for completed generation."""
    from transformers import AutoModelForCausalLM

    progress = load_progress(method_name)
    if progress["n_done"] < N_PROMPTS:
        print(f"{method_name}: Only {progress['n_done']}/{N_PROMPTS} done, need more generation first")
        return None

    ppl_file = RESULTS_DIR / f"v6b_{method_name}_ppls.json"
    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    print(f"\nEvaluating PPL for {method_name} ({len(progress['texts'])} texts)")

    # Load eval model directly (no diffusion model needed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, trust_remote_code=True).to(device)
    eval_model.eval()

    ppls = []
    for ti, text in enumerate(progress["texts"]):
        if len(text) < 10:
            ppls.append(None)
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = eval_model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))

        if (ti + 1) % 64 == 0:
            valid = [p for p in ppls if p is not None]
            print(f"  PPL eval: {ti+1}/{len(progress['texts'])}, mean={np.mean(valid):.3f}")

    del eval_model
    torch.cuda.empty_cache()

    valid_ppls = [float(p) for p in ppls if p is not None]
    result = {
        "method": method_name,
        "n_prompts": N_PROMPTS,
        "n_valid": len(valid_ppls),
        "ppl_mean": float(np.mean(valid_ppls)),
        "ppl_std": float(np.std(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)),
        "conf_mean": float(np.mean(progress["confs"])),
        "all_ppls": valid_ppls,
    }

    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {method_name}: PPL={result['ppl_mean']:.3f} ± {result['ppl_std']:.3f}")
    return result


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "auto"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    if method == "auto":
        # Find what needs to be done
        for m in ["vanilla", "retry_70pct"]:
            p = load_progress(m)
            if p["n_done"] < N_PROMPTS:
                method = m
                break
        else:
            # All generation done, do PPL eval
            for m in ["vanilla", "retry_70pct"]:
                ppl_file = RESULTS_DIR / f"v6b_{m}_ppls.json"
                if not ppl_file.exists():
                    method = f"eval_{m}"
                    break
            else:
                method = "summary"

    if method == "summary":
        from scipy import stats as scipy_stats
        v = json.load(open(RESULTS_DIR / "v6b_vanilla_ppls.json"))
        r = json.load(open(RESULTS_DIR / "v6b_retry_70pct_ppls.json"))
        delta = (r["ppl_mean"] - v["ppl_mean"]) / v["ppl_mean"] * 100
        n = min(len(v["all_ppls"]), len(r["all_ppls"]))
        _, p = scipy_stats.ttest_rel(v["all_ppls"][:n], r["all_ppls"][:n])
        print(f"\n{'='*60}")
        print(f"RESULTS (256 WikiText-103 prompts)")
        print(f"  Vanilla:    PPL={v['ppl_mean']:.3f} ± {v['ppl_std']:.3f}")
        print(f"  Retry 70%:  PPL={r['ppl_mean']:.3f} ± {r['ppl_std']:.3f}")
        print(f"  Delta:      {delta:+.1f}%")
        print(f"  p-value:    {p:.10f}")
        print(f"  N samples:  {n}")
        print(f"{'='*60}")

    elif method.startswith("eval_"):
        m = method[5:]
        run_ppl_eval(m)

    else:
        run_generation_batch(method, batch_size)

        # Check if we should also do PPL eval
        p = load_progress(method)
        if p["n_done"] >= N_PROMPTS:
            print(f"\n{method} generation complete! Running PPL eval...")
            run_ppl_eval(method)


if __name__ == "__main__":
    main()
