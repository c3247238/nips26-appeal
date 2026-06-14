"""
ACA-DLM v11: Test ReMask-Retry on LLaDA-8B-Instruct.

The critical test: does ReMask-Retry improve text quality (not just PPL)
on a larger, more capable model?

Uses 32 prompts for quick validation, measures both PPL and diversity.
"""
import os, sys, json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PROMPTS = 32
SEED = 42
GEN_LEN = 128
STEPS = 64
MASK_ID = 126336


def load_llada():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True,
        torch_dtype=torch.bfloat16).to("cuda:0")
    model.eval()
    return model, tokenizer


def make_prompts(tokenizer, n=32):
    questions = [
        "What is machine learning?",
        "Explain the theory of general relativity.",
        "How does photosynthesis work?",
        "What are the main differences between DNA and RNA?",
        "Explain how black holes form.",
        "What is nuclear fusion?",
        "How do earthquakes occur?",
        "What is the structure of an atom?",
        "Explain dark matter.",
        "How does the immune system work?",
        "What causes climate change?",
        "Explain the water cycle.",
        "How do computers store information?",
        "What is blockchain technology?",
        "Explain how neural networks learn.",
        "What is the Heisenberg uncertainty principle?",
        "How do vaccines work?",
        "What is the difference between RAM and ROM?",
        "Explain the concept of entropy.",
        "How does GPS work?",
        "What is quantum computing?",
        "Explain the process of evolution by natural selection.",
        "How does the internet work?",
        "What is the standard model of particle physics?",
        "Explain how batteries work.",
        "What are prime numbers and why are they important?",
        "How does a microwave oven heat food?",
        "What is CRISPR gene editing?",
        "Explain the greenhouse effect.",
        "How do airplanes fly?",
        "What is the Turing test?",
        "Explain the concept of recursion in computer science.",
    ]
    prompts = []
    for q in questions[:n]:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def llada_generate(model, prompt_ids, gen_len=GEN_LEN, steps=STEPS, temperature=0.2):
    """Generate text with LLaDA using iterative unmasking."""
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor(
        [prompt_ids + [MASK_ID] * gen_len], device="cuda:0")

    for step in range(steps):
        with torch.no_grad():
            logits = model(input_ids).logits

        mask_positions = (input_ids[0] == MASK_ID).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            break

        mask_logits = logits[0, mask_positions] / max(temperature, 0.01)
        probs = torch.softmax(mask_logits, dim=-1)

        sampled = torch.multinomial(probs, 1).squeeze(-1)
        confidences = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)

        n_unmask = max(1, len(mask_positions) // max(1, steps - step))
        n_unmask = min(n_unmask, len(mask_positions))
        _, top_k = confidences.topk(n_unmask)

        for idx in top_k:
            pos = mask_positions[idx]
            input_ids[0, pos] = sampled[idx]

    return input_ids


def remask_retry(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                 steps=STEPS, temperature=0.2,
                 n_retries=1, refine_steps=32, ratio=0.5, refine_temp=0.2):
    """ReMask-Retry for LLaDA."""
    prompt_len = len(prompt_ids)

    # Phase 1: Standard generation
    input_ids = llada_generate(model, prompt_ids, gen_len, steps, temperature)

    for retry in range(n_retries):
        # Phase 2: Confidence assessment
        with torch.no_grad():
            logits = model(input_ids).logits
            probs = F.softmax(logits.float(), dim=-1)
            confs = torch.gather(probs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)

        gen_confs = confs[0, prompt_len:prompt_len+gen_len].cpu().numpy()
        avg_conf = float(np.mean(gen_confs))

        n_remask = int(ratio * gen_len)
        if n_remask < 5:
            break

        sorted_indices = np.argsort(gen_confs)
        remask_positions = sorted_indices[:n_remask]

        # Phase 3: Remask
        for pos in remask_positions:
            input_ids[0, prompt_len + pos] = MASK_ID

        # Phase 4: Re-denoise
        for step in range(refine_steps):
            with torch.no_grad():
                logits = model(input_ids).logits

            mask_positions = (input_ids[0] == MASK_ID).nonzero(as_tuple=True)[0]
            if len(mask_positions) == 0:
                break

            mask_logits = logits[0, mask_positions] / max(refine_temp, 0.01)
            probs = torch.softmax(mask_logits, dim=-1)

            sampled = torch.multinomial(probs, 1).squeeze(-1)
            confidences = probs.gather(1, sampled.unsqueeze(1)).squeeze(1)

            n_unmask = max(1, len(mask_positions) // max(1, refine_steps - step))
            n_unmask = min(n_unmask, len(mask_positions))
            _, top_k = confidences.topk(n_unmask)

            for idx in top_k:
                pos = mask_positions[idx]
                input_ids[0, pos] = sampled[idx]

    return input_ids, {"avg_conf": avg_conf, "ratio": ratio}


def bigram_diversity(text):
    tokens = text.split()
    if len(tokens) < 5:
        return 1.0
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


def run_experiment(method_name, ratio=0.5, refine_temp=0.2):
    """Run vanilla or retry experiment."""
    progress_file = RESULTS_DIR / f"v11_{method_name}_progress.json"

    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        if progress["n_done"] >= N_PROMPTS:
            print(f"{method_name}: Already done ({progress['n_done']}/{N_PROMPTS})")
            return progress
    else:
        progress = {"texts": [], "diversities": [], "confs": [], "n_done": 0}

    model, tokenizer = load_llada()
    prompts = make_prompts(tokenizer, N_PROMPTS)

    start_idx = progress["n_done"]
    end_idx = N_PROMPTS

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    print(f"\n{method_name}: Prompts {start_idx}-{end_idx-1}")

    for pi in range(start_idx, end_idx):
        prompt = prompts[pi]

        if method_name == "llada_vanilla":
            seq = llada_generate(model, prompt, GEN_LEN, STEPS, 0.2)
        else:
            seq, info = remask_retry(model, tokenizer, prompt, GEN_LEN, STEPS, 0.2,
                                      n_retries=1, refine_steps=32,
                                      ratio=ratio, refine_temp=refine_temp)

        prompt_len = len(prompt)
        gen_ids = seq[0, prompt_len:prompt_len+GEN_LEN].tolist()
        # Remove any remaining mask tokens
        gen_ids = [t for t in gen_ids if t != MASK_ID]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        progress["texts"].append(text)
        progress["diversities"].append(bigram_diversity(text))

        # Confidence
        with torch.no_grad():
            logits = model(seq).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)
        gen_conf = token_conf[0, prompt_len:prompt_len+GEN_LEN].mean().item()
        progress["confs"].append(gen_conf)

        if (pi + 1) % 8 == 0:
            avg_div = np.mean(progress["diversities"][-8:])
            print(f"  {pi+1}/{N_PROMPTS}, diversity={avg_div:.3f}")
            # Print last example
            print(f"    Example: {text[:150]}")

        progress["n_done"] = pi + 1
        with open(progress_file, "w") as f:
            json.dump(progress, f)

    del model
    torch.cuda.empty_cache()
    return progress


def run_ppl_eval(method_name):
    """PPL eval with GPT-2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress_file = RESULTS_DIR / f"v11_{method_name}_progress.json"
    ppl_file = RESULTS_DIR / f"v11_{method_name}_ppls.json"

    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    with open(progress_file) as f:
        progress = json.load(f)

    if progress["n_done"] < N_PROMPTS:
        print(f"{method_name}: Only {progress['n_done']}/{N_PROMPTS}")
        return None

    print(f"\nPPL eval for {method_name}")
    device = torch.device("cuda:0")

    # GPT-2 as cross-family evaluator
    tokenizer = AutoTokenizer.from_pretrained(
        "/home/ccwang/sibyl_system/models/gpt2")
    eval_model = AutoModelForCausalLM.from_pretrained(
        "/home/ccwang/sibyl_system/models/gpt2",
        dtype=torch.bfloat16).to(device)
    eval_model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    for text in progress["texts"]:
        if len(text.strip()) < 10:
            ppls.append(None)
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
        with torch.no_grad():
            out = eval_model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))

    del eval_model
    torch.cuda.empty_cache()

    valid_ppls = [float(p) for p in ppls if p is not None]
    n_deg = sum(1 for d in progress["diversities"] if d < 0.3)

    result = {
        "method": method_name, "n_prompts": N_PROMPTS,
        "ppl_mean": float(np.mean(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)),
        "diversity_mean": float(np.mean(progress["diversities"])),
        "n_degenerate": n_deg,
        "all_ppls": valid_ppls,
    }
    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  {method_name}: PPL(med)={result['ppl_median']:.3f}, "
          f"diversity={result['diversity_mean']:.3f}, degenerate={n_deg}/{N_PROMPTS}")
    return result


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "auto"

    configs = {
        "llada_vanilla": (0, 0),
        "llada_r50_t02": (0.5, 0.2),
        "llada_r30_t02": (0.3, 0.2),
    }

    if task == "auto":
        for name, (ratio, temp) in configs.items():
            p_file = RESULTS_DIR / f"v11_{name}_progress.json"
            if not p_file.exists() or json.load(open(p_file))["n_done"] < N_PROMPTS:
                run_experiment(name, ratio, temp)
                return

        for name in configs:
            if not (RESULTS_DIR / f"v11_{name}_ppls.json").exists():
                run_ppl_eval(name)
                return

        # Summary
        from scipy import stats as scipy_stats
        print(f"\n{'='*70}")
        print("LLaDA-8B-Instruct: ReMask-Retry Results")
        print(f"{'='*70}")

        v_file = RESULTS_DIR / "v11_llada_vanilla_ppls.json"
        if not v_file.exists():
            print("Missing vanilla results")
            return

        v = json.load(open(v_file))
        for name in configs:
            f = RESULTS_DIR / f"v11_{name}_ppls.json"
            if not f.exists():
                continue
            r = json.load(open(f))
            delta = ""
            if name != "llada_vanilla":
                d = (r["ppl_median"] - v["ppl_median"]) / v["ppl_median"] * 100
                delta = f"{d:+.1f}%"
                # Wilcoxon
                n = min(len(v["all_ppls"]), len(r["all_ppls"]))
                stat, p = scipy_stats.wilcoxon(v["all_ppls"][:n], r["all_ppls"][:n])
                delta += f" (p={p:.3f})"
            print(f"  {name:<20} PPL(med)={r['ppl_median']:.3f} "
                  f"div={r['diversity_mean']:.3f} deg={r['n_degenerate']}/{N_PROMPTS} {delta}")

        print(f"{'='*70}")

    elif task == "summary":
        sys.argv = ["", "auto"]
        # Force summary by pretending all configs done
        main()
    else:
        if task in configs:
            ratio, temp = configs[task]
            run_experiment(task, ratio, temp)
            p = json.load(open(RESULTS_DIR / f"v11_{task}_progress.json"))
            if p["n_done"] >= N_PROMPTS:
                run_ppl_eval(task)


if __name__ == "__main__":
    main()
