"""
ACA-DLM v10: Temperature ablation for refinement step.

The core issue: low temperature (0.2) during refinement causes the model to
fill remasked positions with repetitive high-confidence tokens, gaming PPL
while degrading text quality.

Test: vary refinement temperature (0.2, 0.5, 0.7, 1.0) and measure both
PPL and text diversity (bigram diversity ratio).
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
N_PROMPTS = 64  # Quick test with 64 prompts
SEED = 42


def remask_retry_temp(model, tokenizer, prompt, config,
                       n_retries=1, refine_steps=32, ratio=0.5, refine_temp=0.7):
    """ReMask-Retry with configurable refinement temperature."""
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
            max_new_tokens=gen_len, steps=refine_steps, temperature=refine_temp,
            remasking="low_confidence", block_size=32)

        with torch.no_grad():
            seq = sampler.sample([prompt], config=refine_config, x_initial=seq)

    return seq


def bigram_diversity(text):
    tokens = text.split()
    if len(tokens) < 5:
        return 1.0
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


def run_config(config_name, ratio, refine_temp, batch_size=64):
    progress_file = RESULTS_DIR / f"v10_{config_name}_progress.json"

    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        if progress["n_done"] >= N_PROMPTS:
            print(f"{config_name}: Already done")
            return progress
    else:
        progress = {"texts": [], "confs": [], "diversities": [], "n_done": 0}

    model, tokenizer = load_model()
    prompts = make_256_prompts(tokenizer)[:N_PROMPTS]
    device = next(model.parameters()).device

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    start_idx = progress["n_done"]
    end_idx = min(start_idx + batch_size, N_PROMPTS)

    torch.manual_seed(SEED + start_idx)
    torch.cuda.manual_seed(SEED + start_idx)

    print(f"\n{config_name}: Prompts {start_idx}-{end_idx-1}")

    if config_name == "vanilla":
        sampler = dllm.core.samplers.MDLMSampler(
            model=model, tokenizer=tokenizer,
            scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    for pi in range(start_idx, end_idx):
        prompt = prompts[pi]

        if config_name == "vanilla":
            with torch.no_grad():
                seq = sampler.sample([prompt], config=config)
        else:
            seq = remask_retry_temp(model, tokenizer, prompt, config,
                                     n_retries=1, refine_steps=32,
                                     ratio=ratio, refine_temp=refine_temp)

        text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
        progress["texts"].append(text)
        progress["diversities"].append(bigram_diversity(text))

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
            avg_div = np.mean(progress["diversities"][-16:])
            print(f"  {pi+1}/{N_PROMPTS}, avg_diversity={avg_div:.3f}")

    progress["n_done"] = end_idx
    with open(progress_file, "w") as f:
        json.dump(progress, f)
    return progress


def run_ppl_eval(config_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress_file = RESULTS_DIR / f"v10_{config_name}_progress.json"
    ppl_file = RESULTS_DIR / f"v10_{config_name}_ppls.json"

    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    with open(progress_file) as f:
        progress = json.load(f)

    if progress["n_done"] < N_PROMPTS:
        return None

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", trust_remote_code=True)
    eval_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", dtype=torch.bfloat16, trust_remote_code=True).to(device)
    eval_model.eval()

    ppls = []
    for text in progress["texts"]:
        if len(text.strip()) < 10:
            ppls.append(None)
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = eval_model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))

    del eval_model; torch.cuda.empty_cache()

    valid_ppls = [float(p) for p in ppls if p is not None]
    result = {
        "method": config_name, "n_prompts": N_PROMPTS,
        "ppl_mean": float(np.mean(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)),
        "diversity_mean": float(np.mean(progress["diversities"])),
        "n_degenerate": sum(1 for d in progress["diversities"] if d < 0.3),
        "all_ppls": valid_ppls,
    }
    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "auto"

    configs = {
        "vanilla": (0, 0),
        "r50_t02": (0.5, 0.2),   # original low temp
        "r50_t05": (0.5, 0.5),   # medium temp
        "r50_t07": (0.5, 0.7),   # higher temp
        "r50_t10": (0.5, 1.0),   # full temp
        "r30_t05": (0.3, 0.5),   # conservative ratio, medium temp
        "r30_t07": (0.3, 0.7),   # conservative ratio, higher temp
    }

    if task == "auto":
        # Generate all configs
        for name, (ratio, temp) in configs.items():
            p_file = RESULTS_DIR / f"v10_{name}_progress.json"
            if p_file.exists():
                with open(p_file) as f:
                    p = json.load(f)
                if p["n_done"] >= N_PROMPTS:
                    continue
            run_config(name, ratio, temp)
            return

        # All generated, run eval
        for name in configs:
            ppl_file = RESULTS_DIR / f"v10_{name}_ppls.json"
            if not ppl_file.exists():
                p_file = RESULTS_DIR / f"v10_{name}_progress.json"
                if p_file.exists():
                    run_ppl_eval(name)
                    return

        # Print summary
        print_summary(configs)

    elif task == "summary":
        print_summary(configs)
    else:
        if task in configs:
            ratio, temp = configs[task]
            run_config(task, ratio, temp)
            p = json.load(open(RESULTS_DIR / f"v10_{task}_progress.json"))
            if p["n_done"] >= N_PROMPTS:
                run_ppl_eval(task)


def print_summary(configs):
    from scipy import stats as scipy_stats

    print(f"\n{'='*80}")
    print("TEMPERATURE ABLATION: PPL vs Diversity tradeoff")
    print(f"{'='*80}")
    print(f"{'Config':<15} {'Ratio':>5} {'Temp':>5} {'PPL(med)':>9} {'PPL(mean)':>10} "
          f"{'Diversity':>9} {'Degen':>5} {'Δ PPL':>8}")

    v_file = RESULTS_DIR / "v10_vanilla_ppls.json"
    v_ppl_med = None
    if v_file.exists():
        v = json.load(open(v_file))
        v_ppl_med = v["ppl_median"]

    for name, (ratio, temp) in configs.items():
        ppl_file = RESULTS_DIR / f"v10_{name}_ppls.json"
        if not ppl_file.exists():
            print(f"  {name:<15} (not available)")
            continue
        r = json.load(open(ppl_file))
        delta = ""
        if v_ppl_med and name != "vanilla":
            d = (r["ppl_median"] - v_ppl_med) / v_ppl_med * 100
            delta = f"{d:+.1f}%"
        print(f"  {name:<15} {ratio:>5.1f} {temp:>5.1f} {r['ppl_median']:>9.3f} "
              f"{r['ppl_mean']:>10.3f} {r['diversity_mean']:>9.3f} "
              f"{r['n_degenerate']:>3}/{N_PROMPTS} {delta:>8}")

    print(f"{'='*80}")


if __name__ == "__main__":
    main()
