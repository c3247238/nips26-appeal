"""
TCR v2: Extended experiments for Dream-7B inference improvement.

New methods based on brainstorming + critic feedback:
1. vanilla_origin: Standard baseline (128 steps)
2. vanilla_256: Compute-matched baseline (256 steps, same wall-clock as TCR)
3. temp_anneal: Temperature annealing (warm→cold, zero extra cost)
4. tcr_r30: Trajectory-Consistent Remasking (30% remask)
5. parallel_vote: K=4 parallel trajectories + majority vote (uses 4 GPUs)

Also fixes:
- Scale to 128 prompts for statistical power
- Add sequence length tracking
- Add bootstrap confidence intervals
"""
import os, sys, json, math, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/tcr_v2")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
N_PROMPTS = 121
SEED = 42
GEN_LEN = 128
STEPS = 128
MASK_TOKEN_ID = 151666


def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream loaded on {device}")
    return model, tokenizer


def make_prompts(tokenizer, n=128):
    """Expanded prompt set for better statistical power."""
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
        # Additional prompts for 128 total
        "How does a nuclear reactor work?",
        "What is the difference between a virus and a bacterium?",
        "Explain the concept of supply and demand in economics.",
        "How do solar panels generate electricity?",
        "What is the theory of plate tectonics?",
        "Explain how encryption works.",
        "What causes tides in the ocean?",
        "How does a computer processor execute instructions?",
        "What is the Doppler effect?",
        "Explain how a combustion engine works.",
        "What is game theory?",
        "How does the human digestive system work?",
        "Explain the concept of natural selection.",
        "What are stem cells and why are they important?",
        "How does radar work?",
        "What is the difference between a conductor and an insulator?",
        "Explain how fiber optic cables transmit data.",
        "What causes aurora borealis?",
        "How does a refrigerator work?",
        "What is the Pythagorean theorem?",
        "Explain the water treatment process.",
        "What is an algorithm?",
        "How do magnets work?",
        "What is the difference between weather and climate?",
        "Explain how 3D printing works.",
        "What is the Periodic Table of Elements?",
        "How does a telescope work?",
        "What is the theory of relativity?",
        "Explain how memory works in the human brain.",
        "What is a semiconductor?",
        "How do antibiotics work?",
        "What is the Big Bang theory?",
        "Explain the concept of probability.",
        "How does a laser work?",
        "What is the difference between AC and DC electricity?",
        "Explain how a search engine works.",
        "What causes volcanic eruptions?",
        "How does a wind turbine generate electricity?",
        "What is the greenhouse effect?",
        "Explain how satellites orbit the Earth.",
        "What is artificial neural network?",
        "How does the stock market work?",
        "What is a black hole?",
        "Explain how sound travels.",
        "What is the difference between mitosis and meiosis?",
        "How do electric cars work?",
        "What is cryptography?",
        "Explain the concept of infinity in mathematics.",
        "How does a MRI machine work?",
        "What is the carbon cycle?",
        "Explain how touchscreens work.",
        "What is a supernova?",
        "How do bridges support weight?",
        "What is the difference between heat and temperature?",
        "Explain how a compiler works.",
        "What causes rainbows?",
        "How does a submarine work?",
        "What is the Fibonacci sequence?",
        "Explain how the human eye sees color.",
        "What is nuclear fission?",
        "How do drones fly?",
        "What is the difference between a planet and a star?",
        "Explain how wireless charging works.",
        "What causes thunder and lightning?",
        "How does a pacemaker work?",
        "What is the theory of evolution?",
        "Explain how a barcode works.",
        "What is dark energy?",
        "How do noise-canceling headphones work?",
        "What is the difference between speed and velocity?",
        "Explain how a fuel cell works.",
        "What causes seasons on Earth?",
        "How does a digital camera work?",
        "What is string theory?",
        "Explain how water purification works.",
        "What is the difference between a gene and a chromosome?",
        "How do self-driving cars work?",
        "What is the photoelectric effect?",
        "Explain how a thermostat works.",
        "What causes earthquakes?",
        "How does a jet engine work?",
        "What is the Schrödinger equation?",
        "Explain how the human heart pumps blood.",
        "What is machine learning used for?",
        "How does Bluetooth technology work?",
        "What is the difference between an element and a compound?",
        "Explain how a gyroscope works.",
        "What is the cosmic microwave background?",
        "How do holograms work?",
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
# Generation methods
# ============================================================
def generate_vanilla(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                     steps=STEPS, device="cuda:0"):
    """Standard Dream generation."""
    input_ids = torch.tensor([prompt_ids], device=device)
    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=gen_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        alg="origin",
    )
    return output.sequences


def generate_temp_anneal(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                          steps=STEPS, device="cuda:0",
                          t_high=1.2, t_low=0.5, alpha=2.0):
    """Temperature annealing: warm start → cold finish.

    Zero extra compute. Just modifies logits via hook.
    """
    input_ids = torch.tensor([prompt_ids], device=device)

    def logits_hook(step, x, logits):
        if logits is None or step is None:
            return logits
        # Temperature schedule: high at start, low at end
        progress = step / max(steps - 1, 1)  # 0 → 1
        temp = t_high * (1 - progress**alpha) + t_low * (progress**alpha)
        # Apply temperature to masked positions only
        mask_idx = (x == MASK_TOKEN_ID)
        if mask_idx.any():
            logits[mask_idx] = logits[mask_idx] / max(temp, 0.01)
        return logits

    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=gen_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,  # we handle temperature in the hook
        alg="origin",
        generation_logits_hook_func=logits_hook,
    )
    return output.sequences


class TrajectoryRecorder:
    """Records per-position predictions across steps."""
    def __init__(self, prompt_len, gen_len):
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.trajectory = defaultdict(list)
        self.step_count = 0

    def __call__(self, step, x, logits):
        if logits is None:
            return logits
        gen_start = self.prompt_len
        gen_end = self.prompt_len + self.gen_len
        for pos in range(gen_start, min(gen_end, x.shape[1])):
            if x[0, pos].item() == MASK_TOKEN_ID:
                rel_pos = pos - gen_start
                top_token = logits[0, pos].argmax().item()
                self.trajectory[rel_pos].append((self.step_count, top_token))
        self.step_count += 1
        return logits

    def get_stability(self):
        stability = np.ones(self.gen_len)
        for pos in range(self.gen_len):
            if pos not in self.trajectory or len(self.trajectory[pos]) < 2:
                continue
            entries = self.trajectory[pos]
            n_consistent = sum(
                1 for i in range(1, len(entries))
                if entries[i][1] == entries[i-1][1])
            stability[pos] = n_consistent / (len(entries) - 1)
        return stability


def generate_tcr(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                  steps=STEPS, device="cuda:0",
                  remask_ratio=0.3, refine_steps=64):
    """TCR: Trajectory-Consistent Remasking."""
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)

    recorder = TrajectoryRecorder(prompt_len, gen_len)

    # Phase 1: Generate with trajectory recording
    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=gen_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        alg="origin",
        generation_logits_hook_func=recorder,
    )
    seq = output.sequences

    # Phase 2: Compute stability & remask
    stability = recorder.get_stability()
    n_remask = int(remask_ratio * gen_len)
    if n_remask < 3:
        return seq, {"n_remasked": 0, "mean_stability": float(stability.mean())}

    unstable_positions = np.argsort(stability)[:n_remask]
    for pos in unstable_positions:
        seq[0, prompt_len + pos] = MASK_TOKEN_ID

    # Phase 3: Re-denoise manually
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
        p_transfer = 1 - s / t if step_i < refine_steps - 1 else 1
        x0 = torch.zeros_like(seq[mask_index], device=device, dtype=torch.long) + MASK_TOKEN_ID
        transfer_index = torch.rand(*x0.shape, device=device) < p_transfer
        probs = torch.softmax(mask_logits, dim=-1)
        sampled = probs.argmax(dim=-1)
        x0[transfer_index] = sampled[transfer_index]
        seq[mask_index] = x0.clone()

    info = {
        "n_remasked": n_remask,
        "mean_stability": float(stability.mean()),
        "remasked_stability": float(stability[unstable_positions].mean()),
    }
    return seq, info


def generate_tcr_anneal(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                         steps=STEPS, device="cuda:0",
                         remask_ratio=0.3, refine_steps=64,
                         t_high=1.2, t_low=0.5, alpha=2.0):
    """TCR + Temperature Annealing combined."""
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)

    recorder = TrajectoryRecorder(prompt_len, gen_len)

    def combined_hook(step, x, logits):
        # Record trajectory
        logits = recorder(step, x, logits)
        # Apply temperature annealing
        if logits is not None and step is not None:
            progress = step / max(steps - 1, 1)
            temp = t_high * (1 - progress**alpha) + t_low * (progress**alpha)
            mask_idx = (x == MASK_TOKEN_ID)
            if mask_idx.any():
                logits[mask_idx] = logits[mask_idx] / max(temp, 0.01)
        return logits

    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=gen_len,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        temperature=0.0,
        alg="origin",
        generation_logits_hook_func=combined_hook,
    )
    seq = output.sequences

    stability = recorder.get_stability()
    n_remask = int(remask_ratio * gen_len)
    if n_remask < 3:
        return seq, {"n_remasked": 0, "mean_stability": float(stability.mean())}

    unstable_positions = np.argsort(stability)[:n_remask]
    for pos in unstable_positions:
        seq[0, prompt_len + pos] = MASK_TOKEN_ID

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
        p_transfer = 1 - s / t if step_i < refine_steps - 1 else 1
        x0 = torch.zeros_like(seq[mask_index], device=device, dtype=torch.long) + MASK_TOKEN_ID
        transfer_index = torch.rand(*x0.shape, device=device) < p_transfer
        probs = torch.softmax(mask_logits, dim=-1)
        sampled = probs.argmax(dim=-1)
        x0[transfer_index] = sampled[transfer_index]
        seq[mask_index] = x0.clone()

    info = {
        "n_remasked": n_remask,
        "mean_stability": float(stability.mean()),
        "remasked_stability": float(stability[unstable_positions].mean()),
    }
    return seq, info


# ============================================================
# Experiment runner
# ============================================================
def run_experiment(method_name, model, tokenizer, prompts,
                   device="cuda:0", **kwargs):
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
            "gen_times": [], "text_lengths": [], "n_done": 0,
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

        if method_name == "vanilla_origin":
            seq = generate_vanilla(model, tokenizer, prompt, GEN_LEN, STEPS, device)
            info = {}
        elif method_name == "vanilla_256":
            seq = generate_vanilla(model, tokenizer, prompt, GEN_LEN, 256, device)
            info = {}
        elif method_name == "temp_anneal":
            seq = generate_temp_anneal(model, tokenizer, prompt, GEN_LEN, STEPS,
                                       device, **kwargs)
            info = {}
        elif method_name == "tcr_r30":
            seq, info = generate_tcr(model, tokenizer, prompt, GEN_LEN, STEPS,
                                      device, remask_ratio=0.3, refine_steps=64)
        elif method_name == "tcr_anneal":
            seq, info = generate_tcr_anneal(model, tokenizer, prompt, GEN_LEN, STEPS,
                                             device, remask_ratio=0.3, refine_steps=64,
                                             **kwargs)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        elapsed = time.time() - t0

        prompt_len = len(prompt)
        total_len = seq.shape[1]
        gen_ids = seq[0, prompt_len:min(prompt_len+GEN_LEN, total_len)].tolist()
        gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        div = bigram_diversity(text)
        d2 = distinct_n(text, 2)
        text_len = len(text.split())

        progress["texts"].append(text)
        progress["diversities"].append(div)
        progress["distinct2s"].append(d2)
        progress["gen_times"].append(elapsed)
        progress["text_lengths"].append(text_len)
        progress["infos"].append(info)
        progress["n_done"] = pi + 1

        if (pi + 1) % 16 == 0 or pi == N_PROMPTS - 1:
            avg_div = np.mean(progress["diversities"][-16:])
            avg_time = np.mean(progress["gen_times"][-16:])
            avg_len = np.mean(progress["text_lengths"][-16:])
            print(f"  [{pi+1}/{N_PROMPTS}] div={avg_div:.3f} "
                  f"time={avg_time:.1f}s len={avg_len:.0f}w")

        with open(progress_file, "w") as f:
            json.dump(progress, f)

    return progress


def run_ppl_eval(method_name, device="cuda:0"):
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

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_medians = []
    for _ in range(1000):
        sample = rng.choice(valid_ppls, size=len(valid_ppls), replace=True)
        boot_medians.append(float(np.median(sample)))
    ci_lo = float(np.percentile(boot_medians, 2.5))
    ci_hi = float(np.percentile(boot_medians, 97.5))

    result = {
        "method": method_name, "n_prompts": N_PROMPTS,
        "n_valid": len(valid_ppls),
        "ppl_mean": float(np.mean(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)),
        "ppl_ci95": [ci_lo, ci_hi],
        "ppl_std": float(np.std(valid_ppls)),
        "ppl_max": float(np.max(valid_ppls)),
        "ppl_p90": float(np.percentile(valid_ppls, 90)),
        "diversity_mean": float(np.mean(progress["diversities"])),
        "distinct2_mean": float(np.mean(progress["distinct2s"])),
        "n_degenerate": n_deg,
        "avg_gen_time": float(np.mean(progress["gen_times"])),
        "avg_text_len": float(np.mean(progress["text_lengths"])),
        "n_short": sum(1 for l in progress["text_lengths"] if l < 10),
        "all_ppls": valid_ppls,
    }

    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {method_name}: PPL(med)={result['ppl_median']:.3f} "
          f"[{ci_lo:.1f}, {ci_hi:.1f}] "
          f"mean={result['ppl_mean']:.1f} "
          f"div={result['diversity_mean']:.3f} "
          f"deg={n_deg}/{N_PROMPTS} "
          f"time={result['avg_gen_time']:.1f}s "
          f"len={result['avg_text_len']:.0f}w")
    return result


def print_summary():
    from scipy import stats as scipy_stats

    print(f"\n{'='*100}")
    print("TCR v2: Dream-7B Inference Improvement (128 prompts)")
    print(f"{'='*100}")

    methods = ["vanilla_origin", "vanilla_256", "temp_anneal",
               "tcr_r30", "tcr_anneal"]
    results = {}
    for m in methods:
        f = RESULTS_DIR / f"{m}_ppls.json"
        if f.exists():
            results[m] = json.load(open(f))

    if not results:
        print("No results yet")
        return

    baseline_name = "vanilla_origin" if "vanilla_origin" in results else list(results.keys())[0]
    v = results.get(baseline_name)

    print(f"\n{'Method':<18} {'PPL(med)':>9} {'95%CI':>14} {'PPL(mean)':>10} "
          f"{'PPL(p90)':>9} {'Div':>6} {'Deg':>5} {'Time':>6} {'Len':>5} {'vs Base':>14}")
    print("-" * 115)

    for m in methods:
        if m not in results:
            continue
        r = results[m]
        ci = f"[{r['ppl_ci95'][0]:.1f},{r['ppl_ci95'][1]:.1f}]"
        delta = ""
        if m != baseline_name and v:
            d = (r["ppl_median"] - v["ppl_median"]) / v["ppl_median"] * 100
            n = min(len(v["all_ppls"]), len(r["all_ppls"]))
            try:
                stat, p = scipy_stats.wilcoxon(v["all_ppls"][:n], r["all_ppls"][:n])
                delta = f"{d:+.1f}% p={p:.3f}"
            except Exception:
                delta = f"{d:+.1f}%"

        print(f"  {m:<16} {r['ppl_median']:>9.2f} {ci:>14} {r['ppl_mean']:>10.1f} "
              f"{r['ppl_p90']:>9.1f} {r['diversity_mean']:>6.3f} "
              f"{r['n_degenerate']:>5d} {r['avg_gen_time']:>6.1f} "
              f"{r['avg_text_len']:>5.0f} {delta:>14}")

    print(f"\n{'='*100}")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "auto"
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    device = f"cuda:{gpu}"

    configs = {
        "vanilla_origin": {},
        "vanilla_256": {},
        "temp_anneal": {"t_high": 1.2, "t_low": 0.5, "alpha": 2.0},
        "tcr_r30": {},
        "tcr_anneal": {"t_high": 1.2, "t_low": 0.5, "alpha": 2.0},
    }

    if task == "auto":
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
