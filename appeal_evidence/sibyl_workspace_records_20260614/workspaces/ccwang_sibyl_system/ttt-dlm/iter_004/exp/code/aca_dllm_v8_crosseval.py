"""
ACA-DLM v8: Cross-family PPL evaluation + Fixed-ratio-0.41 ablation.

Addresses critic review:
1. Cross-family PPL: Evaluate with GPT-2 (non-Qwen) to avoid same-family bias
2. Fixed-0.41 ablation: Compare adaptive (avg ratio=0.41) vs fixed 0.41 to isolate adaptation benefit
"""
import os, sys, json, math
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


# ========== Part 1: Cross-family PPL evaluation ==========

def eval_ppl_with_model(model_name, model_path, texts, output_key):
    """Evaluate PPL of generated texts using a specified AR model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ppl_file = RESULTS_DIR / f"v8_crosseval_{output_key}.json"
    if ppl_file.exists():
        print(f"Already done: {ppl_file}")
        with open(ppl_file) as f:
            return json.load(f)

    print(f"\nEvaluating PPL with {model_name} ({len(texts)} texts)")
    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)
    model.eval()

    # GPT-2 doesn't have a pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    for ti, text in enumerate(texts):
        if len(text.strip()) < 10:
            ppls.append(None)
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))
        if (ti + 1) % 64 == 0:
            valid = [p for p in ppls if p is not None]
            print(f"  {ti+1}/{len(texts)}, mean PPL={np.mean(valid):.3f}")

    del model
    torch.cuda.empty_cache()

    valid_ppls = [float(p) for p in ppls if p is not None]
    result = {
        "eval_model": model_name,
        "n_texts": len(texts),
        "n_valid": len(valid_ppls),
        "ppl_mean": float(np.mean(valid_ppls)),
        "ppl_std": float(np.std(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)),
        "all_ppls": valid_ppls,
    }

    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {output_key}: mean={result['ppl_mean']:.3f}, median={result['ppl_median']:.3f}")
    return result


def run_cross_family_eval():
    """Evaluate vanilla, fixed-70%, and adaptive texts with GPT-2."""
    # Load generated texts from existing results
    methods = {
        "vanilla": "v6c_vanilla_progress.json",
        "fixed70": "v6c_retry_70pct_progress.json",
        "adaptive": "v7_adaptive_progress.json",
    }

    gpt2_path = "/home/ccwang/sibyl_system/models/gpt2"

    for method_key, progress_file in methods.items():
        pf = RESULTS_DIR / progress_file
        if not pf.exists():
            print(f"Missing {pf}, skipping {method_key}")
            continue
        with open(pf) as f:
            progress = json.load(f)
        texts = progress["texts"]
        eval_ppl_with_model("GPT-2", gpt2_path, texts, f"gpt2_{method_key}")

    # Also eval with Qwen2.5-1.5B-Instruct (different size, still Qwen family but different gen)
    for method_key, progress_file in methods.items():
        pf = RESULTS_DIR / progress_file
        if not pf.exists():
            continue
        with open(pf) as f:
            progress = json.load(f)
        texts = progress["texts"]
        eval_ppl_with_model("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct",
                           texts, f"qwen25_{method_key}")


# ========== Part 2: Fixed-0.41 ablation ==========

def fixed_ratio_remask_retry(model, tokenizer, prompt, config,
                              n_retries=2, refine_steps=32, ratio=0.41):
    """ReMask-Retry with a fixed ratio (for comparison with adaptive)."""
    device = next(model.parameters()).device
    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    with torch.no_grad():
        seq = sampler.sample([prompt], config=config)

    prompt_len = len(prompt)
    gen_len = config.max_new_tokens

    for retry in range(n_retries):
        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            confs = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)

        gen_confs = confs[0, prompt_len:prompt_len+gen_len].cpu().numpy()

        n_remask = int(ratio * gen_len)
        if n_remask < 10:
            break

        sorted_indices = np.argsort(gen_confs)
        remask_positions = sorted_indices[:n_remask]

        for mt in ["[MASK]", "<mask>", "<|mask|>"]:
            ids = tokenizer.encode(mt, add_special_tokens=False)
            if len(ids) == 1:
                mtid = ids[0]
                break
        else:
            mtid = 0

        for pos in remask_positions:
            seq[0, prompt_len + pos] = mtid

        refine_config = dllm.core.samplers.MDLMSamplerConfig(
            max_new_tokens=gen_len, steps=refine_steps, temperature=0.2,
            remasking="low_confidence", block_size=32)

        with torch.no_grad():
            seq = sampler.sample([prompt], config=refine_config, x_initial=seq)

    return seq


def run_fixed041_generation(batch_size=64):
    """Generate with fixed ratio=0.41 for comparison with adaptive.
    Uses SEED+1000 offset to ensure independent random state from adaptive."""
    method_name = "fixed_041"
    progress_file = RESULTS_DIR / f"v8_{method_name}_progress.json"

    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"texts": [], "confs": [], "n_done": 0}

    if progress["n_done"] >= N_PROMPTS:
        print(f"{method_name}: All {N_PROMPTS} done")
        return progress

    model, tokenizer = load_model()
    device = next(model.parameters()).device
    prompts = make_256_prompts(tokenizer)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    start_idx = progress["n_done"]
    end_idx = min(start_idx + batch_size, N_PROMPTS)
    print(f"\n{method_name}: Prompts {start_idx}-{end_idx-1} of {N_PROMPTS}")

    # Use same seed as adaptive (SEED + start_idx) to get same initial generation,
    # but the ratio difference will cause different remasking patterns
    torch.manual_seed(SEED + start_idx)
    torch.cuda.manual_seed(SEED + start_idx)

    for pi in range(start_idx, end_idx):
        prompt = prompts[pi]
        seq = fixed_ratio_remask_retry(model, tokenizer, prompt, config,
                                        n_retries=2, refine_steps=32, ratio=0.41)

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
            print(f"  {pi+1}/{N_PROMPTS} done")

    progress["n_done"] = end_idx
    with open(progress_file, "w") as f:
        json.dump(progress, f)
    print(f"  Batch done. Total: {end_idx}/{N_PROMPTS}")
    return progress


def run_fixed041_ppl_eval():
    """PPL eval for fixed-0.41 method."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress_file = RESULTS_DIR / "v8_fixed_041_progress.json"
    if not progress_file.exists():
        print("No fixed-0.41 progress file")
        return None

    with open(progress_file) as f:
        progress = json.load(f)

    if progress["n_done"] < N_PROMPTS:
        print(f"fixed_041: Only {progress['n_done']}/{N_PROMPTS}")
        return None

    ppl_file = RESULTS_DIR / "v8_fixed_041_ppls.json"
    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    print("\nPPL eval for fixed_041")
    device = torch.device("cuda:0")
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

    del eval_model
    torch.cuda.empty_cache()

    valid_ppls = [float(p) for p in ppls if p is not None]
    result = {
        "method": "fixed_041", "n_prompts": N_PROMPTS, "n_valid": len(valid_ppls),
        "ppl_mean": float(np.mean(valid_ppls)), "ppl_std": float(np.std(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)), "conf_mean": float(np.mean(progress["confs"])),
        "all_ppls": valid_ppls,
    }
    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  fixed_041: PPL={result['ppl_mean']:.3f}, median={result['ppl_median']:.3f}")
    return result


# ========== Part 3: Summary ==========

def print_summary():
    """Print comprehensive comparison."""
    from scipy import stats as scipy_stats

    # Load all Qwen3-0.6B PPLs
    v = json.load(open(RESULTS_DIR / "v6c_vanilla_ppls.json"))
    r = json.load(open(RESULTS_DIR / "v6c_retry_70pct_ppls.json"))
    a = json.load(open(RESULTS_DIR / "v7_adaptive_ppls.json"))

    print(f"\n{'='*70}")
    print("QWEN3-0.6B AR EVALUATOR (original)")
    for name, d in [("Vanilla", v), ("Fixed 70%", r), ("Adaptive", a)]:
        n_cat = sum(1 for p in d["all_ppls"] if p > 100)
        safe = [p for p in d["all_ppls"] if p <= 100]
        delta = (np.median(d["all_ppls"]) - np.median(v["all_ppls"])) / np.median(v["all_ppls"]) * 100
        print(f"  {name:<15} median={np.median(d['all_ppls']):.3f}  "
              f"safe_mean={np.mean(safe):.3f}  cat={n_cat}  delta={delta:+.1f}%")

    # Fixed 0.41 if available
    f41_file = RESULTS_DIR / "v8_fixed_041_ppls.json"
    if f41_file.exists():
        f41 = json.load(open(f41_file))
        n_cat = sum(1 for p in f41["all_ppls"] if p > 100)
        safe = [p for p in f41["all_ppls"] if p <= 100]
        delta = (np.median(f41["all_ppls"]) - np.median(v["all_ppls"])) / np.median(v["all_ppls"]) * 100
        print(f"  {'Fixed 41%':<15} median={np.median(f41['all_ppls']):.3f}  "
              f"safe_mean={np.mean(safe):.3f}  cat={n_cat}  delta={delta:+.1f}%")

        # Adaptive vs Fixed-41% comparison
        stat, p = scipy_stats.wilcoxon(a["all_ppls"][:len(f41["all_ppls"])],
                                        f41["all_ppls"][:len(a["all_ppls"])])
        win = sum(1 for x, y in zip(a["all_ppls"], f41["all_ppls"]) if x < y)
        n = min(len(a["all_ppls"]), len(f41["all_ppls"]))
        print(f"\n  Adaptive vs Fixed-41%: Wilcoxon p={p:.2e}, win_rate={win/n*100:.1f}%")

    # Cross-family GPT-2 eval
    gpt2_files = {}
    for key in ["gpt2_vanilla", "gpt2_fixed70", "gpt2_adaptive"]:
        f = RESULTS_DIR / f"v8_crosseval_{key}.json"
        if f.exists():
            gpt2_files[key] = json.load(open(f))

    if gpt2_files:
        print(f"\n{'='*70}")
        print("GPT-2 EVALUATOR (cross-family)")
        gpt2_v = gpt2_files.get("gpt2_vanilla", {})
        for key, name in [("gpt2_vanilla", "Vanilla"), ("gpt2_fixed70", "Fixed 70%"),
                          ("gpt2_adaptive", "Adaptive")]:
            if key in gpt2_files:
                d = gpt2_files[key]
                delta = 0
                if gpt2_v.get("ppl_median"):
                    delta = (d["ppl_median"] - gpt2_v["ppl_median"]) / gpt2_v["ppl_median"] * 100
                print(f"  {name:<15} median={d['ppl_median']:.3f}  "
                      f"mean={d['ppl_mean']:.3f}  delta={delta:+.1f}%")

        # Wilcoxon on GPT-2 PPLs
        if "gpt2_vanilla" in gpt2_files and "gpt2_adaptive" in gpt2_files:
            va = gpt2_files["gpt2_vanilla"]["all_ppls"]
            aa = gpt2_files["gpt2_adaptive"]["all_ppls"]
            n = min(len(va), len(aa))
            stat, p = scipy_stats.wilcoxon(va[:n], aa[:n])
            win = sum(1 for x, y in zip(va, aa) if y < x)
            print(f"\n  Adaptive vs Vanilla (GPT-2): Wilcoxon p={p:.2e}, win_rate={win/n*100:.1f}%")

    # Qwen2.5-1.5B eval
    q25_files = {}
    for key in ["qwen25_vanilla", "qwen25_fixed70", "qwen25_adaptive"]:
        f = RESULTS_DIR / f"v8_crosseval_{key}.json"
        if f.exists():
            q25_files[key] = json.load(open(f))

    if q25_files:
        print(f"\n{'='*70}")
        print("QWEN2.5-1.5B-INSTRUCT EVALUATOR (larger model)")
        q25_v = q25_files.get("qwen25_vanilla", {})
        for key, name in [("qwen25_vanilla", "Vanilla"), ("qwen25_fixed70", "Fixed 70%"),
                          ("qwen25_adaptive", "Adaptive")]:
            if key in q25_files:
                d = q25_files[key]
                delta = 0
                if q25_v.get("ppl_median"):
                    delta = (d["ppl_median"] - q25_v["ppl_median"]) / q25_v["ppl_median"] * 100
                print(f"  {name:<15} median={d['ppl_median']:.3f}  "
                      f"mean={d['ppl_mean']:.3f}  delta={delta:+.1f}%")

    print(f"\n{'='*70}")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if task == "crosseval":
        run_cross_family_eval()
    elif task == "fixed041_gen":
        run_fixed041_generation()
        p_file = RESULTS_DIR / "v8_fixed_041_progress.json"
        with open(p_file) as f:
            p = json.load(f)
        if p["n_done"] >= N_PROMPTS:
            print("\nAll done! Running PPL eval...")
            run_fixed041_ppl_eval()
    elif task == "fixed041_eval":
        run_fixed041_ppl_eval()
    elif task == "summary":
        print_summary()
    elif task == "auto":
        # Auto: do cross-eval first (uses existing texts, no generation needed)
        run_cross_family_eval()
        print_summary()
    else:
        print(f"Unknown task: {task}")
        print("Usage: python v8_crosseval.py [crosseval|fixed041_gen|fixed041_eval|summary|auto]")


if __name__ == "__main__":
    main()
