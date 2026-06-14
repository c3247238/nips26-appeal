"""
Trajectory-Consistent Remasking (TCR) for Dream-7B

Novel inference-time improvement for Masked Diffusion Language Models.

Key idea: During iterative denoising, track how predictions for masked positions
evolve across steps. Tokens whose predictions keep changing (unstable) are
unreliable and should be remasked after initial generation.

Uses Dream's native `diffusion_generate` with hook functions for clean integration.

Methods compared:
1. Vanilla: Standard Dream denoising (origin algorithm)
2. Vanilla-MaskGit: Dream with maskgit_plus algorithm
3. ReMDM-style: Principled remasking (random remask at each step via hook)
4. TCR (ours): Trajectory-consistent remasking

Evaluation: GPT-2 cross-family PPL, bigram diversity, Distinct-2, text inspection.
"""
import os, sys, json, math, time, copy
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/tcr_v1")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
N_PROMPTS = 32
SEED = 42
GEN_LEN = 128
STEPS = 128
MASK_TOKEN_ID = 151666


def load_dream(device="cuda:0"):
    """Load Dream-7B-Instruct model."""
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True,
        dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream loaded on {device}, mask_id={MASK_TOKEN_ID}")
    return model, tokenizer


def make_prompts(tokenizer, n=32):
    """Create diverse prompts for evaluation."""
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
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
        prompts.append(prompt_ids)
    return prompts


# ============================================================
# Evaluation utilities
# ============================================================
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


# ============================================================
# Method 1 & 2: Vanilla Dream (origin / maskgit_plus)
# ============================================================
def generate_vanilla(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                     steps=STEPS, alg="origin", device="cuda:0"):
    """Standard Dream generation using native API."""
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=gen_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        alg=alg,
        alg_temp=0.0 if alg != "origin" else None,
    )
    return output.sequences


# ============================================================
# Method 3: Trajectory-recording generation (for TCR)
# ============================================================
class TrajectoryRecorder:
    """Hook that records per-position predictions across denoising steps."""

    def __init__(self, prompt_len, gen_len, mask_id):
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.mask_id = mask_id
        # trajectory[relative_pos] = list of (step, top_token_id)
        self.trajectory = defaultdict(list)
        self.step_count = 0

    def logits_hook(self, step, x, logits):
        """Record top prediction for each still-masked position."""
        if logits is None:
            return logits

        gen_start = self.prompt_len
        gen_end = self.prompt_len + self.gen_len

        # For each masked position in the generation region
        for pos in range(gen_start, min(gen_end, x.shape[1])):
            if x[0, pos].item() == self.mask_id:
                rel_pos = pos - gen_start
                top_token = logits[0, pos].argmax().item()
                self.trajectory[rel_pos].append((self.step_count, top_token))

        self.step_count += 1
        return logits

    def get_stability(self):
        """Compute per-position trajectory stability.

        Stability = fraction of consecutive steps where top prediction was same.
        Low stability = unreliable token.
        """
        stability = np.ones(self.gen_len)
        for pos in range(self.gen_len):
            if pos not in self.trajectory or len(self.trajectory[pos]) < 2:
                continue
            entries = self.trajectory[pos]
            n_consistent = 0
            n_transitions = 0
            for i in range(1, len(entries)):
                n_transitions += 1
                if entries[i][1] == entries[i-1][1]:
                    n_consistent += 1
            stability[pos] = n_consistent / n_transitions if n_transitions > 0 else 1.0
        return stability


def generate_with_trajectory(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                              steps=STEPS, alg="origin", device="cuda:0"):
    """Generate with trajectory recording via logits hook."""
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    recorder = TrajectoryRecorder(prompt_len, gen_len, MASK_TOKEN_ID)

    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=gen_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        alg=alg,
        alg_temp=0.0 if alg != "origin" else None,
        generation_logits_hook_func=recorder.logits_hook,
    )
    return output.sequences, recorder


# ============================================================
# Method 4: TCR - Trajectory-Consistent Remasking
# ============================================================
def generate_tcr(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                  steps=STEPS, alg="origin", device="cuda:0",
                  remask_ratio=0.3, refine_steps=64):
    """Trajectory-Consistent Remasking.

    Phase 1: Standard generation with trajectory recording
    Phase 2: Compute per-position stability from trajectory
    Phase 3: Remask least-stable positions
    Phase 4: Re-denoise remasked positions
    """
    prompt_len = len(prompt_ids)

    # Phase 1: Generate with trajectory
    seq, recorder = generate_with_trajectory(
        model, tokenizer, prompt_ids, gen_len, steps, alg, device)

    # Phase 2: Compute stability
    stability = recorder.get_stability()

    # Phase 3: Remask unstable positions
    n_remask = int(remask_ratio * gen_len)
    if n_remask < 3:
        info = {"stability": stability.tolist(), "n_remasked": 0,
                "mean_stability": float(stability.mean())}
        return seq, info

    unstable_positions = np.argsort(stability)[:n_remask]
    for pos in unstable_positions:
        seq[0, prompt_len + pos] = MASK_TOKEN_ID

    # Phase 4: Re-denoise manually (Dream API doesn't support re-denoising
    # an already-full-length sequence due to max_length validation)
    eps = 1e-3
    timesteps = torch.linspace(1, eps, refine_steps + 1, device=device)
    for step_i in range(refine_steps):
        mask_index = (seq == MASK_TOKEN_ID)
        if not mask_index.any():
            break
        with torch.no_grad():
            logits = model(seq).logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

        mask_logits = logits[mask_index]
        t = timesteps[step_i]
        s = timesteps[step_i + 1]

        # Use origin algorithm for refinement
        p_transfer = 1 - s / t if step_i < refine_steps - 1 else 1
        x0 = torch.zeros_like(seq[mask_index], device=device, dtype=torch.long) + MASK_TOKEN_ID
        transfer_index = torch.rand(*x0.shape, device=device) < p_transfer
        probs = torch.softmax(mask_logits, dim=-1)
        sampled = probs.argmax(dim=-1)  # greedy for refinement
        x0[transfer_index] = sampled[transfer_index]
        seq[mask_index] = x0.clone()

    info = {
        "n_remasked": n_remask,
        "mean_stability": float(stability.mean()),
        "remasked_stability": float(stability[unstable_positions].mean()),
        "kept_stability": float(np.delete(stability, unstable_positions).mean()),
    }
    return seq, info


# ============================================================
# Method 5: ReMDM-style via token hook
# ============================================================
class ReMDMHook:
    """Token hook that randomly remasks already-generated tokens."""

    def __init__(self, prompt_len, mask_id, remask_prob=0.1):
        self.prompt_len = prompt_len
        self.mask_id = mask_id
        self.remask_prob = remask_prob
        self.step = 0

    def __call__(self, step, x, logits):
        if step is None:  # initial call
            return x
        self.step += 1

        # Randomly remask some already-unmasked generation tokens
        gen_region = x[0, self.prompt_len:]
        unmasked = (gen_region != self.mask_id).nonzero(as_tuple=True)[0]
        if len(unmasked) > 0:
            remask = torch.rand(len(unmasked), device=x.device) < self.remask_prob
            for idx in unmasked[remask]:
                x[0, self.prompt_len + idx] = self.mask_id

        return x


def generate_remdm(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                    steps=STEPS, alg="origin", device="cuda:0",
                    remask_prob=0.1):
    """ReMDM-style generation with random remasking via token hook."""
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)

    hook = ReMDMHook(prompt_len, MASK_TOKEN_ID, remask_prob)

    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=gen_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        alg=alg,
        alg_temp=0.0 if alg != "origin" else None,
        generation_tokens_hook_func=hook,
    )
    return output.sequences


# ============================================================
# Main experiment runner
# ============================================================
def run_experiment(method_name, model, tokenizer, prompts,
                   device="cuda:0", **method_kwargs):
    """Run a single method on all prompts with incremental saving."""
    progress_file = RESULTS_DIR / f"{method_name}_progress.json"

    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
        if progress["n_done"] >= N_PROMPTS:
            print(f"{method_name}: Already done ({progress['n_done']}/{N_PROMPTS})")
            return progress
    else:
        progress = {
            "method": method_name,
            "texts": [], "diversities": [], "distinct2s": [],
            "gen_times": [], "n_done": 0,
            "infos": [],
        }

    start_idx = progress["n_done"]
    print(f"\n{'='*60}")
    print(f"{method_name}: Prompts {start_idx}-{N_PROMPTS-1}")
    print(f"{'='*60}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    for pi in range(start_idx, N_PROMPTS):
        prompt = prompts[pi]
        t0 = time.time()

        if method_name.startswith("vanilla"):
            alg = method_kwargs.get("alg", "origin")
            seq = generate_vanilla(model, tokenizer, prompt, GEN_LEN, STEPS,
                                   alg=alg, device=device)
            info = {}
        elif method_name.startswith("tcr"):
            alg = method_kwargs.get("alg", "origin")
            remask_ratio = method_kwargs.get("remask_ratio", 0.3)
            refine_steps = method_kwargs.get("refine_steps", 64)
            seq, info = generate_tcr(model, tokenizer, prompt, GEN_LEN, STEPS,
                                      alg=alg, device=device,
                                      remask_ratio=remask_ratio,
                                      refine_steps=refine_steps)
        elif method_name.startswith("remdm"):
            alg = method_kwargs.get("alg", "origin")
            remask_prob = method_kwargs.get("remask_prob", 0.1)
            seq = generate_remdm(model, tokenizer, prompt, GEN_LEN, STEPS,
                                  alg=alg, device=device,
                                  remask_prob=remask_prob)
            info = {}
        else:
            raise ValueError(f"Unknown method: {method_name}")

        elapsed = time.time() - t0

        # Decode generated text
        prompt_len = len(prompt)
        total_len = seq.shape[1]
        gen_ids = seq[0, prompt_len:min(prompt_len+GEN_LEN, total_len)].tolist()
        gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        div = bigram_diversity(text)
        d2 = distinct_n(text, 2)

        progress["texts"].append(text)
        progress["diversities"].append(div)
        progress["distinct2s"].append(d2)
        progress["gen_times"].append(elapsed)
        progress["infos"].append(info)
        progress["n_done"] = pi + 1

        if (pi + 1) % 4 == 0 or pi == N_PROMPTS - 1:
            avg_div = np.mean(progress["diversities"][-4:])
            avg_time = np.mean(progress["gen_times"][-4:])
            print(f"  [{pi+1}/{N_PROMPTS}] div={avg_div:.3f} "
                  f"time={avg_time:.1f}s")
            print(f"    >> {text[:150]}...")

        # Incremental save
        with open(progress_file, "w") as f:
            json.dump(progress, f)

    return progress


def run_ppl_eval(method_name, device="cuda:0"):
    """Cross-family PPL evaluation with GPT-2."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress_file = RESULTS_DIR / f"{method_name}_progress.json"
    ppl_file = RESULTS_DIR / f"{method_name}_ppls.json"

    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    with open(progress_file) as f:
        progress = json.load(f)

    if progress["n_done"] < N_PROMPTS:
        print(f"{method_name}: Only {progress['n_done']}/{N_PROMPTS}")
        return None

    print(f"\nPPL eval for {method_name}")
    gpt2_path = "/home/ccwang/sibyl_system/models/gpt2"
    tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
    eval_model = AutoModelForCausalLM.from_pretrained(
        gpt2_path, torch_dtype=torch.bfloat16).to(device)
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
        "distinct2_mean": float(np.mean(progress["distinct2s"])),
        "n_degenerate": n_deg,
        "avg_gen_time": float(np.mean(progress["gen_times"])),
        "all_ppls": valid_ppls,
    }

    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {method_name}: PPL(med)={result['ppl_median']:.3f} "
          f"div={result['diversity_mean']:.3f} "
          f"d2={result['distinct2_mean']:.3f} "
          f"deg={n_deg}/{N_PROMPTS} "
          f"time={result['avg_gen_time']:.1f}s")
    return result


def print_summary():
    """Print comparison table of all methods."""
    from scipy import stats as scipy_stats

    print(f"\n{'='*90}")
    print("TCR v1: Trajectory-Consistent Remasking for Dream-7B")
    print(f"{'='*90}")

    methods = ["vanilla_origin", "vanilla_maskgit", "remdm_p10",
               "tcr_r30", "tcr_r50"]
    results = {}
    for m in methods:
        f = RESULTS_DIR / f"{m}_ppls.json"
        if f.exists():
            results[m] = json.load(open(f))

    if not results:
        print("No results yet")
        return

    # Use first available as baseline
    baseline_name = "vanilla_origin" if "vanilla_origin" in results else list(results.keys())[0]
    v = results[baseline_name]

    print(f"\n{'Method':<22} {'PPL(med)':>10} {'PPL(mean)':>10} {'Diversity':>10} "
          f"{'Distinct-2':>10} {'Degen':>8} {'Time(s)':>8} {'vs Baseline':>14}")
    print("-" * 104)

    for m in methods:
        if m not in results:
            continue
        r = results[m]
        delta = ""
        if m != baseline_name:
            d = (r["ppl_median"] - v["ppl_median"]) / v["ppl_median"] * 100
            n = min(len(v["all_ppls"]), len(r["all_ppls"]))
            try:
                stat, p = scipy_stats.wilcoxon(v["all_ppls"][:n], r["all_ppls"][:n])
                delta = f"{d:+.1f}% (p={p:.3f})"
            except Exception:
                delta = f"{d:+.1f}%"

        print(f"  {m:<20} {r['ppl_median']:>10.3f} {r['ppl_mean']:>10.3f} "
              f"{r['diversity_mean']:>10.3f} {r['distinct2_mean']:>10.3f} "
              f"{r['n_degenerate']:>8d} {r['avg_gen_time']:>8.1f} {delta:>14}")

    print(f"\n{'='*90}")

    # Print example texts
    for m in methods:
        pf = RESULTS_DIR / f"{m}_progress.json"
        if pf.exists():
            prog = json.load(open(pf))
            print(f"\n--- {m} example (prompt 0) ---")
            if prog["texts"]:
                print(prog["texts"][0][:400])


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "auto"
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    device = f"cuda:{gpu}"

    configs = {
        "vanilla_origin": {"alg": "origin"},
        "vanilla_maskgit": {"alg": "maskgit_plus"},
        "remdm_p10": {"alg": "origin", "remask_prob": 0.1},
        "tcr_r30": {"alg": "origin", "remask_ratio": 0.3, "refine_steps": 64},
        "tcr_r50": {"alg": "origin", "remask_ratio": 0.5, "refine_steps": 64},
    }

    if task == "auto":
        # Find next incomplete experiment
        for name in configs:
            p_file = RESULTS_DIR / f"{name}_progress.json"
            if not p_file.exists():
                need_gen = True
            else:
                with open(p_file) as f:
                    need_gen = json.load(f)["n_done"] < N_PROMPTS
            if need_gen:
                model, tokenizer = load_dream(device)
                prompts = make_prompts(tokenizer, N_PROMPTS)
                run_experiment(name, model, tokenizer, prompts,
                             device=device, **configs[name])
                del model
                torch.cuda.empty_cache()
                return

        # All generated, do PPL eval
        for name in configs:
            if not (RESULTS_DIR / f"{name}_ppls.json").exists():
                run_ppl_eval(name, device)
                return

        print_summary()

    elif task == "summary":
        print_summary()

    elif task == "ppl_all":
        for name in configs:
            if not (RESULTS_DIR / f"{name}_ppls.json").exists():
                run_ppl_eval(name, device)

    elif task in configs:
        model, tokenizer = load_dream(device)
        prompts = make_prompts(tokenizer, N_PROMPTS)
        run_experiment(task, model, tokenizer, prompts,
                     device=device, **configs[task])
        del model
        torch.cuda.empty_cache()
        p = json.load(open(RESULTS_DIR / f"{task}_progress.json"))
        if p["n_done"] >= N_PROMPTS:
            run_ppl_eval(task, device)

    else:
        print(f"Unknown task: {task}")
        print(f"Available: auto, summary, ppl_all, {', '.join(configs.keys())}")


if __name__ == "__main__":
    main()
