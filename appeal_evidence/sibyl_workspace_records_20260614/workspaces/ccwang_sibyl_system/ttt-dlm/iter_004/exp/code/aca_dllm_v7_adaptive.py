"""
ACA-DLM v7: Adaptive remask ratio.

Instead of fixed 70% remask, adapt the ratio based on initial generation quality:
- High avg confidence → lower remask ratio (don't destroy good content)
- Low avg confidence → higher remask ratio (more room for improvement)

Tests on 256 prompts using batch-resumable pattern.
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
from aca_dllm_v6c_largeset import make_256_prompts

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PROMPTS = 256
SEED = 42


def adaptive_remask_retry(model, tokenizer, prompt, config,
                          n_retries=2, refine_steps=32, max_ratio=0.7, min_ratio=0.3):
    """ReMask-Retry with adaptive ratio based on initial confidence."""
    device = next(model.parameters()).device
    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    # Phase 1: Standard denoising
    with torch.no_grad():
        seq = sampler.sample([prompt], config=config)

    prompt_len = len(prompt)
    gen_len = config.max_new_tokens

    for retry in range(n_retries):
        # Phase 2: Confidence assessment
        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            confs = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)

        gen_confs = confs[0, prompt_len:prompt_len+gen_len].cpu().numpy()
        avg_conf = float(np.mean(gen_confs))

        # Adaptive ratio: high confidence → lower ratio
        # Linear interpolation: conf=0 → max_ratio, conf=1 → min_ratio
        # But cap at reasonable bounds
        adaptive_ratio = max_ratio - (max_ratio - min_ratio) * min(avg_conf / 0.5, 1.0)
        adaptive_ratio = max(min_ratio, min(max_ratio, adaptive_ratio))

        n_remask = int(adaptive_ratio * gen_len)
        if n_remask < 10:
            break  # Too few to remask, skip

        sorted_indices = np.argsort(gen_confs)
        remask_positions = sorted_indices[:n_remask]

        # Phase 3: Remask
        # Find mask token
        for mt in ["[MASK]", "<mask>", "<|mask|>"]:
            ids = tokenizer.encode(mt, add_special_tokens=False)
            if len(ids) == 1:
                mtid = ids[0]
                break
        else:
            mtid = 0

        for pos in remask_positions:
            seq[0, prompt_len + pos] = mtid

        # Re-denoise
        refine_config = dllm.core.samplers.MDLMSamplerConfig(
            max_new_tokens=gen_len, steps=refine_steps, temperature=0.2,
            remasking="low_confidence", block_size=32)

        with torch.no_grad():
            seq = sampler.sample([prompt], config=refine_config, x_initial=seq)

    return seq, {"adaptive_ratio": adaptive_ratio, "avg_conf": avg_conf}


def load_progress(method_name):
    fname = RESULTS_DIR / f"v7_{method_name}_progress.json"
    if fname.exists():
        with open(fname) as f:
            return json.load(f)
    return {"texts": [], "confs": [], "ratios": [], "n_done": 0}


def save_progress(method_name, progress):
    fname = RESULTS_DIR / f"v7_{method_name}_progress.json"
    with open(fname, "w") as f:
        json.dump(progress, f)


def run_generation_batch(method_name, batch_size=64):
    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_256_prompts(tokenizer)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    progress = load_progress(method_name)
    start_idx = progress["n_done"]

    if start_idx >= N_PROMPTS:
        print(f"{method_name}: All {N_PROMPTS} done")
        return progress

    end_idx = min(start_idx + batch_size, N_PROMPTS)
    print(f"\n{method_name}: Prompts {start_idx}-{end_idx-1} of {N_PROMPTS}")

    torch.manual_seed(SEED + start_idx)
    torch.cuda.manual_seed(SEED + start_idx)

    for pi in range(start_idx, end_idx):
        prompt = prompts[pi]

        if method_name == "adaptive":
            seq, info = adaptive_remask_retry(model, tokenizer, prompt, config,
                                              n_retries=2, refine_steps=32,
                                              max_ratio=0.7, min_ratio=0.3)
            progress["ratios"].append(info["adaptive_ratio"])
        elif method_name == "adaptive_conservative":
            seq, info = adaptive_remask_retry(model, tokenizer, prompt, config,
                                              n_retries=2, refine_steps=32,
                                              max_ratio=0.6, min_ratio=0.2)
            progress["ratios"].append(info["adaptive_ratio"])

        text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
        progress["texts"].append(text)

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
            avg_ratio = np.mean(progress["ratios"][-16:]) if progress["ratios"] else 0
            print(f"  {pi+1}/{N_PROMPTS} done, avg_ratio={avg_ratio:.3f}")

    progress["n_done"] = end_idx
    save_progress(method_name, progress)
    print(f"  Batch done. Total: {end_idx}/{N_PROMPTS}")
    return progress


def run_ppl_eval(method_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress = load_progress(method_name)
    if progress["n_done"] < N_PROMPTS:
        print(f"{method_name}: Only {progress['n_done']}/{N_PROMPTS}")
        return None

    ppl_file = RESULTS_DIR / f"v7_{method_name}_ppls.json"
    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    print(f"\nPPL eval for {method_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            print(f"  {ti+1}/{len(progress['texts'])}, mean={np.mean(valid):.3f}")

    del eval_model; torch.cuda.empty_cache()

    valid_ppls = [float(p) for p in ppls if p is not None]
    result = {
        "method": method_name, "n_prompts": N_PROMPTS, "n_valid": len(valid_ppls),
        "ppl_mean": float(np.mean(valid_ppls)), "ppl_std": float(np.std(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)), "conf_mean": float(np.mean(progress["confs"])),
        "all_ppls": valid_ppls,
        "avg_ratio": float(np.mean(progress.get("ratios", [0]))),
    }
    if progress.get("ratios"):
        result["ratio_distribution"] = {
            "min": float(np.min(progress["ratios"])),
            "max": float(np.max(progress["ratios"])),
            "mean": float(np.mean(progress["ratios"])),
            "std": float(np.std(progress["ratios"])),
        }
    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {method_name}: PPL={result['ppl_mean']:.3f}, median={result['ppl_median']:.3f}")
    return result


def main():
    method = sys.argv[1] if len(sys.argv) > 1 else "auto"
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    if method == "auto":
        for m in ["adaptive"]:
            p = load_progress(m)
            if p["n_done"] < N_PROMPTS:
                method = m
                break
        else:
            for m in ["adaptive"]:
                if not (RESULTS_DIR / f"v7_{m}_ppls.json").exists():
                    method = f"eval_{m}"
                    break
            else:
                method = "summary"

    if method == "summary":
        from scipy import stats as scipy_stats
        # Load vanilla from v6c
        v = json.load(open(RESULTS_DIR / "v6c_vanilla_ppls.json"))
        a = json.load(open(RESULTS_DIR / "v7_adaptive_ppls.json"))
        # Also load fixed 70% from v6c
        r = json.load(open(RESULTS_DIR / "v6c_retry_70pct_ppls.json"))

        print(f"\n{'='*70}")
        print(f"ADAPTIVE vs FIXED REMASK (256 prompts)")
        for name, d in [("Vanilla", v), ("Fixed 70%", r), ("Adaptive", a)]:
            n_cat = sum(1 for p in d["all_ppls"] if p > 100)
            safe = [p for p in d["all_ppls"] if p <= 100]
            delta_med = (np.median(d["all_ppls"]) - np.median(v["all_ppls"])) / np.median(v["all_ppls"]) * 100
            print(f"  {name:<15} PPL(med)={np.median(d['all_ppls']):.3f}  "
                  f"PPL(safe mean)={np.mean(safe):.3f}  "
                  f"catastrophic={n_cat}  Δ(med)={delta_med:+.1f}%")
            if "avg_ratio" in d:
                print(f"                avg_ratio={d['avg_ratio']:.3f}")
        print(f"{'='*70}")

    elif method.startswith("eval_"):
        run_ppl_eval(method[5:])
    else:
        run_generation_batch(method, batch_size)
        p = load_progress(method)
        if p["n_done"] >= N_PROMPTS:
            print(f"\n{method} complete! Running PPL eval...")
            run_ppl_eval(method)


if __name__ == "__main__":
    main()
