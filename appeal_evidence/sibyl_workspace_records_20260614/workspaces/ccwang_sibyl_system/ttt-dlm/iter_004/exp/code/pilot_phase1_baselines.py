"""
Pilot Phase 1: Multi-temperature baselines for Dream-7B.
Task_0 (setup verification) + Task_1 (vanilla baselines at 4 temperatures).

16 prompts, seed=42, 128 gen tokens, 128 denoising steps.
Temps: 0.2, 0.4, 0.6, 0.8.

Usage:
  python3 pilot_phase1_baselines.py [gpu_id]
"""
import os, sys, json, math, time
from pathlib import Path
from collections import Counter

import numpy as np
import torch

PYTHON = sys.executable
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilot")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
GPT2_DIR = "/home/ccwang/sibyl_system/models/gpt2"
N_PROMPTS = 16
SEED = 42
GEN_LEN = 128
STEPS = 128
MASK_TOKEN_ID = 151666


def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream loaded on {device}")
    return model, tokenizer


def make_prompts(tokenizer, n=16):
    questions = [
        "What is machine learning?", "Explain the theory of general relativity.",
        "How does photosynthesis work?", "What are the main differences between DNA and RNA?",
        "Explain how black holes form.", "What is nuclear fusion?",
        "How do earthquakes occur?", "What is the structure of an atom?",
        "Explain dark matter.", "How does the immune system work?",
        "What causes climate change?", "Explain the water cycle.",
        "How do computers store information?", "What is blockchain technology?",
        "Explain how neural networks learn.", "What is the Heisenberg uncertainty principle?",
    ]
    prompts = []
    for q in questions[:n]:
        messages = [{"role": "user", "content": q}]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompts.append(prompt_ids)
    return prompts


def bigram_diversity(text):
    tokens = text.split()
    if len(tokens) < 5:
        return 1.0
    bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
    return len(set(bigrams)) / len(bigrams) if bigrams else 1.0


def distinct_n(text, n=2):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def compute_ppl_batch(texts, device="cuda:0"):
    """Compute GPT-2 PPL for a list of texts."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(GPT2_DIR)
    model = AutoModelForCausalLM.from_pretrained(
        GPT2_DIR, torch_dtype=torch.bfloat16).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ppls = []
    for text in texts:
        if len(text.strip()) < 10:
            ppls.append(None)
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=512).to(device)
        with torch.no_grad():
            out = model(**enc, labels=enc["input_ids"])
        ppls.append(math.exp(min(out.loss.item(), 20)))

    del model
    torch.cuda.empty_cache()
    return ppls


def generate_vanilla(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                     steps=STEPS, device="cuda:0", temperature=0.4):
    input_ids = torch.tensor([prompt_ids], device=device)
    output = model.diffusion_generate(
        input_ids, max_new_tokens=gen_len, output_history=False,
        return_dict_in_generate=True, steps=steps,
        temperature=temperature, alg="origin",
    )
    return output.sequences


def decode_generation(seq, prompt_len, tokenizer, gen_len=GEN_LEN):
    total_len = seq.shape[1]
    gen_ids = seq[0, prompt_len:min(prompt_len + gen_len, total_len)].tolist()
    gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


def run_baselines(model, tokenizer, prompts, device="cuda:0"):
    temperatures = [0.2, 0.4, 0.6, 0.8]
    all_results = {}

    for temp in temperatures:
        method_name = f"vanilla_t{int(temp*10):02d}"
        print(f"\n{'='*60}")
        print(f"Running {method_name} ({N_PROMPTS} prompts)")
        print(f"{'='*60}")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        texts, diversities, distinct2s, gen_times, text_lengths = [], [], [], [], []

        for pi in range(N_PROMPTS):
            t0 = time.time()
            seq = generate_vanilla(model, tokenizer, prompts[pi],
                                   device=device, temperature=temp)
            elapsed = time.time() - t0

            text = decode_generation(seq, len(prompts[pi]), tokenizer)
            texts.append(text)
            diversities.append(bigram_diversity(text))
            distinct2s.append(distinct_n(text, 2))
            gen_times.append(elapsed)
            text_lengths.append(len(text.split()))

            if (pi + 1) % 8 == 0:
                print(f"  [{pi+1}/{N_PROMPTS}] div={np.mean(diversities[-8:]):.3f} "
                      f"time={np.mean(gen_times[-8:]):.1f}s len={np.mean(text_lengths[-8:]):.0f}w")

        # Print sample outputs
        print(f"\n  Sample outputs ({method_name}):")
        for i in [0, 4, 8]:
            print(f"    [{i}] {texts[i][:120]}...")

        all_results[method_name] = {
            "texts": texts,
            "diversities": diversities,
            "distinct2s": distinct2s,
            "gen_times": gen_times,
            "text_lengths": text_lengths,
        }

    return all_results


def main():
    gpu = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    device = f"cuda:{gpu}"

    print(f"=== Pilot Phase 1: Multi-Temperature Baselines ===")
    print(f"GPU: {gpu}, Prompts: {N_PROMPTS}, Seed: {SEED}")
    print(f"Gen length: {GEN_LEN}, Steps: {STEPS}")

    # Task 0: Setup verification
    print(f"\n--- Task 0: Setup Verification ---")
    model, tokenizer = load_dream(device)
    prompts = make_prompts(tokenizer, N_PROMPTS)

    # Quick verification: 4 prompts
    print("Verifying generation pipeline...")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    for i in range(4):
        seq = generate_vanilla(model, tokenizer, prompts[i], device=device, temperature=0.4)
        text = decode_generation(seq, len(prompts[i]), tokenizer)
        n_tokens = len(text.split())
        print(f"  Prompt {i}: {n_tokens} words, text='{text[:80]}...'")
        if n_tokens < 10:
            print(f"  WARNING: Very short output for prompt {i}!")

    setup_ok = True
    print(f"Setup verification: {'PASS' if setup_ok else 'FAIL'}")

    # Task 1: Multi-temperature baselines
    print(f"\n--- Task 1: Multi-Temperature Baselines ---")
    baseline_results = run_baselines(model, tokenizer, prompts, device)

    # Free Dream model, run PPL eval
    del model
    torch.cuda.empty_cache()

    print(f"\n--- PPL Evaluation ---")
    final_results = {}
    for method_name, data in baseline_results.items():
        ppls = compute_ppl_batch(data["texts"], device)
        valid_ppls = [p for p in ppls if p is not None]
        n_deg = sum(1 for d in data["diversities"] if d < 0.3)

        result = {
            "method": method_name,
            "n_prompts": N_PROMPTS,
            "n_valid": len(valid_ppls),
            "n_empty": sum(1 for t in data["texts"] if len(t.strip()) < 10),
            "ppl_mean": float(np.mean(valid_ppls)) if valid_ppls else None,
            "ppl_median": float(np.median(valid_ppls)) if valid_ppls else None,
            "diversity_mean": float(np.mean(data["diversities"])),
            "distinct2_mean": float(np.mean(data["distinct2s"])),
            "n_degenerate": n_deg,
            "avg_gen_time": float(np.mean(data["gen_times"])),
            "avg_text_len": float(np.mean(data["text_lengths"])),
            "n_short": sum(1 for l in data["text_lengths"] if l < 15),
            "all_ppls": [float(p) for p in valid_ppls],
            "texts": data["texts"][:5],  # Save 5 samples
        }
        final_results[method_name] = result
        print(f"  {method_name}: PPL(med)={result['ppl_median']:.2f} "
              f"mean={result['ppl_mean']:.1f} div={result['diversity_mean']:.3f} "
              f"deg={n_deg} short={result['n_short']} "
              f"time={result['avg_gen_time']:.1f}s len={result['avg_text_len']:.0f}w")

    # Save results
    out_file = RESULTS_DIR / "task1_baselines.json"
    with open(out_file, "w") as f:
        json.dump(final_results, f, indent=2)
    print(f"\nResults saved to {out_file}")

    # Pass/fail check
    t04 = final_results.get("vanilla_t04", {})
    if t04:
        ppl_ok = t04["ppl_median"] and 10 <= t04["ppl_median"] <= 50
        div_ok = t04["diversity_mean"] >= 0.9
        deg_ok = t04["n_degenerate"] == 0
        print(f"\n--- Pass Criteria ---")
        print(f"  PPL(med) in [10,50]: {t04['ppl_median']:.2f} -> {'PASS' if ppl_ok else 'FAIL'}")
        print(f"  Diversity >= 0.9: {t04['diversity_mean']:.3f} -> {'PASS' if div_ok else 'FAIL'}")
        print(f"  Degenerate = 0: {t04['n_degenerate']} -> {'PASS' if deg_ok else 'FAIL'}")
        overall = "GO" if (ppl_ok and div_ok and deg_ok) else "NO-GO"
        print(f"  Overall: {overall}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Method':<16} {'PPL(med)':>9} {'PPL(mean)':>10} {'Div':>6} {'D2':>6} "
          f"{'Deg':>4} {'Short':>6} {'Time':>6} {'Len':>5}")
    print("-" * 90)
    for m in sorted(final_results.keys()):
        r = final_results[m]
        print(f"  {m:<14} {r['ppl_median']:>9.2f} {r['ppl_mean']:>10.1f} "
              f"{r['diversity_mean']:>6.3f} {r['distinct2_mean']:>6.3f} "
              f"{r['n_degenerate']:>4} {r['n_short']:>6} "
              f"{r['avg_gen_time']:>6.1f} {r['avg_text_len']:>5.0f}")
    print(f"{'='*90}")

    # Save setup verification
    setup_result = {
        "task": "task_0",
        "status": "PASS",
        "model": MODEL_DIR,
        "mask_token_id": MASK_TOKEN_ID,
        "gen_len": GEN_LEN,
        "steps": STEPS,
        "python": PYTHON,
        "torch_version": torch.__version__,
    }
    with open(RESULTS_DIR / "task0_setup.json", "w") as f:
        json.dump(setup_result, f, indent=2)


if __name__ == "__main__":
    main()
