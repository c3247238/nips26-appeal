"""
TCR v3: Proper temperature Dream-7B experiments.

Key insight from v2: Dream with temp=0 produces short/empty texts, making PPL
comparisons meaningless. Use temp=0.4 for proper-length generation.

Methods:
1. vanilla_t04: Standard Dream, temp=0.4
2. tcr_t04: TCR with temp=0.4
3. anneal_warm: Anneal from temp=0.8→0.2
4. parallel_vote: K=4 parallel trajectories + vote (all 4 GPUs)

Evaluation: GPT-2 PPL, bigram diversity, Distinct-2, text length.
121 prompts for statistical power.
"""
import os, sys, json, math, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/tcr_v3")
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


def make_prompts(tokenizer, n=121):
    questions = [
        "What is machine learning?", "Explain the theory of general relativity.",
        "How does photosynthesis work?", "What are the main differences between DNA and RNA?",
        "Explain how black holes form.", "What is nuclear fusion?",
        "How do earthquakes occur?", "What is the structure of an atom?",
        "Explain dark matter.", "How does the immune system work?",
        "What causes climate change?", "Explain the water cycle.",
        "How do computers store information?", "What is blockchain technology?",
        "Explain how neural networks learn.", "What is the Heisenberg uncertainty principle?",
        "How do vaccines work?", "What is the difference between RAM and ROM?",
        "Explain the concept of entropy.", "How does GPS work?",
        "What is quantum computing?", "Explain the process of evolution by natural selection.",
        "How does the internet work?", "What is the standard model of particle physics?",
        "Explain how batteries work.", "What are prime numbers and why are they important?",
        "How does a microwave oven heat food?", "What is CRISPR gene editing?",
        "Explain the greenhouse effect.", "How do airplanes fly?",
        "What is the Turing test?", "Explain the concept of recursion in computer science.",
        "How does a nuclear reactor work?", "What is the difference between a virus and a bacterium?",
        "Explain the concept of supply and demand in economics.",
        "How do solar panels generate electricity?", "What is the theory of plate tectonics?",
        "Explain how encryption works.", "What causes tides in the ocean?",
        "How does a computer processor execute instructions?", "What is the Doppler effect?",
        "Explain how a combustion engine works.", "What is game theory?",
        "How does the human digestive system work?", "Explain the concept of natural selection.",
        "What are stem cells and why are they important?", "How does radar work?",
        "What is the difference between a conductor and an insulator?",
        "Explain how fiber optic cables transmit data.", "What causes aurora borealis?",
        "How does a refrigerator work?", "What is the Pythagorean theorem?",
        "Explain the water treatment process.", "What is an algorithm?",
        "How do magnets work?", "What is the difference between weather and climate?",
        "Explain how 3D printing works.", "What is the Periodic Table of Elements?",
        "How does a telescope work?", "What is the theory of relativity?",
        "Explain how memory works in the human brain.", "What is a semiconductor?",
        "How do antibiotics work?", "What is the Big Bang theory?",
        "Explain the concept of probability.", "How does a laser work?",
        "What is the difference between AC and DC electricity?",
        "Explain how a search engine works.", "What causes volcanic eruptions?",
        "How does a wind turbine generate electricity?", "What is the greenhouse effect?",
        "Explain how satellites orbit the Earth.", "What is artificial neural network?",
        "How does the stock market work?", "What is a black hole?",
        "Explain how sound travels.", "What is the difference between mitosis and meiosis?",
        "How do electric cars work?", "What is cryptography?",
        "Explain the concept of infinity in mathematics.", "How does a MRI machine work?",
        "What is the carbon cycle?", "Explain how touchscreens work.",
        "What is a supernova?", "How do bridges support weight?",
        "What is the difference between heat and temperature?",
        "Explain how a compiler works.", "What causes rainbows?",
        "How does a submarine work?", "What is the Fibonacci sequence?",
        "Explain how the human eye sees color.", "What is nuclear fission?",
        "How do drones fly?", "What is the difference between a planet and a star?",
        "Explain how wireless charging works.", "What causes thunder and lightning?",
        "How does a pacemaker work?", "What is the theory of evolution?",
        "Explain how a barcode works.", "What is dark energy?",
        "How do noise-canceling headphones work?",
        "What is the difference between speed and velocity?",
        "Explain how a fuel cell works.", "What causes seasons on Earth?",
        "How does a digital camera work?", "What is string theory?",
        "Explain how water purification works.",
        "What is the difference between a gene and a chromosome?",
        "How do self-driving cars work?", "What is the photoelectric effect?",
        "Explain how a thermostat works.", "What causes earthquakes?",
        "How does a jet engine work?", "What is the Schrödinger equation?",
        "Explain how the human heart pumps blood.", "What is machine learning used for?",
        "How does Bluetooth technology work?",
        "What is the difference between an element and a compound?",
        "Explain how a gyroscope works.", "What is the cosmic microwave background?",
        "How do holograms work?",
    ]
    prompts = []
    for q in questions[:n]:
        messages = [{"role": "user", "content": q}]
        prompt_ids = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True)
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


# ============================================================
# Generation methods
# ============================================================
def generate_vanilla(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                     steps=STEPS, device="cuda:0", temperature=0.4):
    input_ids = torch.tensor([prompt_ids], device=device)
    output = model.diffusion_generate(
        input_ids, max_new_tokens=gen_len, output_history=False,
        return_dict_in_generate=True, steps=steps,
        temperature=temperature, alg="origin",
    )
    return output.sequences


class TrajectoryRecorder:
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
                  steps=STEPS, device="cuda:0", temperature=0.4,
                  remask_ratio=0.3, refine_steps=64):
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)
    recorder = TrajectoryRecorder(prompt_len, gen_len)

    output = model.diffusion_generate(
        input_ids, max_new_tokens=gen_len, output_history=False,
        return_dict_in_generate=True, steps=steps,
        temperature=temperature, alg="origin",
        generation_logits_hook_func=recorder,
    )
    seq = output.sequences

    stability = recorder.get_stability()
    n_remask = int(remask_ratio * gen_len)
    if n_remask < 3:
        return seq, {"n_remasked": 0, "mean_stability": float(stability.mean())}

    unstable_positions = np.argsort(stability)[:n_remask]
    for pos in unstable_positions:
        seq[0, prompt_len + pos] = MASK_TOKEN_ID

    # Re-denoise manually
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
        if temperature > 0:
            mask_logits = mask_logits / temperature
        t = timesteps[step_i]
        s = timesteps[step_i + 1]
        p_transfer = 1 - s / t if step_i < refine_steps - 1 else 1
        x0 = torch.zeros_like(seq[mask_index], device=device, dtype=torch.long) + MASK_TOKEN_ID
        transfer_index = torch.rand(*x0.shape, device=device) < p_transfer
        probs = torch.softmax(mask_logits, dim=-1)
        if temperature > 0:
            import torch.distributions as dists
            try:
                sampled = dists.Categorical(probs=probs).sample()
            except:
                sampled = probs.argmax(dim=-1)
        else:
            sampled = probs.argmax(dim=-1)
        x0[transfer_index] = sampled[transfer_index]
        seq[mask_index] = x0.clone()

    info = {
        "n_remasked": n_remask,
        "mean_stability": float(stability.mean()),
        "remasked_stability": float(stability[unstable_positions].mean()),
    }
    return seq, info


def generate_anneal(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                     steps=STEPS, device="cuda:0",
                     t_high=0.8, t_low=0.2, alpha=2.0):
    """Temperature annealing: warm→cold. Using logits hook."""
    input_ids = torch.tensor([prompt_ids], device=device)

    def logits_hook(step, x, logits):
        if logits is None or step is None:
            return logits
        progress = step / max(steps - 1, 1)
        temp = t_high * (1 - progress**alpha) + t_low * (progress**alpha)
        mask_idx = (x == MASK_TOKEN_ID)
        if mask_idx.any():
            logits[mask_idx] = logits[mask_idx] / max(temp, 0.01)
        return logits

    output = model.diffusion_generate(
        input_ids, max_new_tokens=gen_len, output_history=False,
        return_dict_in_generate=True, steps=steps,
        temperature=0.0,  # handled in hook
        alg="origin",
        generation_logits_hook_func=logits_hook,
    )
    return output.sequences


def generate_parallel_vote(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                            steps=STEPS, device="cuda:0", temperature=0.4,
                            n_trajectories=4):
    """Run multiple trajectories with different seeds, majority vote."""
    prompt_len = len(prompt_ids)
    all_gen_ids = []

    for traj_i in range(n_trajectories):
        torch.manual_seed(SEED + traj_i * 1000)
        torch.cuda.manual_seed(SEED + traj_i * 1000)
        seq = generate_vanilla(model, tokenizer, prompt_ids, gen_len, steps,
                                device, temperature)
        gen_ids = seq[0, prompt_len:prompt_len+gen_len].tolist()
        all_gen_ids.append(gen_ids)

    # Position-wise majority vote
    voted_ids = []
    for pos in range(gen_len):
        tokens_at_pos = [all_gen_ids[t][pos] for t in range(n_trajectories)]
        counter = Counter(tokens_at_pos)
        winner = counter.most_common(1)[0][0]
        voted_ids.append(winner)

    # Build output sequence
    result = torch.tensor([prompt_ids + voted_ids], device=device)

    # Optional: run a few healing steps to fix inconsistencies
    eps = 1e-3
    heal_steps = 16
    # Find positions where there was disagreement
    disagreement = []
    for pos in range(gen_len):
        tokens_at_pos = set(all_gen_ids[t][pos] for t in range(n_trajectories))
        if len(tokens_at_pos) > 1:
            disagreement.append(pos)

    # Remask disagreement positions and re-denoise
    if len(disagreement) > 0:
        for pos in disagreement:
            result[0, prompt_len + pos] = MASK_TOKEN_ID

        timesteps = torch.linspace(1, eps, heal_steps + 1, device=device)
        for step_i in range(heal_steps):
            mask_index = (result == MASK_TOKEN_ID)
            if not mask_index.any():
                break
            with torch.no_grad():
                logits = model(result).logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            mask_logits = logits[mask_index]
            if temperature > 0:
                mask_logits = mask_logits / temperature
            t = timesteps[step_i]
            s = timesteps[step_i + 1]
            p_transfer = 1 - s / t if step_i < heal_steps - 1 else 1
            x0 = torch.zeros_like(result[mask_index], device=device, dtype=torch.long) + MASK_TOKEN_ID
            transfer_idx = torch.rand(*x0.shape, device=device) < p_transfer
            probs = torch.softmax(mask_logits, dim=-1)
            import torch.distributions as dists
            try:
                sampled = dists.Categorical(probs=probs).sample()
            except:
                sampled = probs.argmax(dim=-1)
            x0[transfer_idx] = sampled[transfer_idx]
            result[mask_index] = x0.clone()

    info = {
        "n_disagreement": len(disagreement),
        "n_trajectories": n_trajectories,
        "disagreement_ratio": len(disagreement) / gen_len,
    }
    return result, info


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
            print(f"{method_name}: Already done")
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

        if method_name == "vanilla_t04":
            seq = generate_vanilla(model, tokenizer, prompt, device=device,
                                    temperature=0.4)
            info = {}
        elif method_name == "vanilla_t08":
            seq = generate_vanilla(model, tokenizer, prompt, device=device,
                                    temperature=0.8)
            info = {}
        elif method_name == "tcr_t04":
            seq, info = generate_tcr(model, tokenizer, prompt, device=device,
                                      temperature=0.4, remask_ratio=0.3)
        elif method_name == "anneal_warm":
            seq = generate_anneal(model, tokenizer, prompt, device=device,
                                   t_high=0.8, t_low=0.2, alpha=2.0)
            info = {}
        elif method_name == "parallel_vote":
            seq, info = generate_parallel_vote(
                model, tokenizer, prompt, device=device,
                temperature=0.4, n_trajectories=4)
        else:
            raise ValueError(f"Unknown: {method_name}")

        elapsed = time.time() - t0

        prompt_len = len(prompt)
        total_len = seq.shape[1]
        gen_ids = seq[0, prompt_len:min(prompt_len+GEN_LEN, total_len)].tolist()
        gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        progress["texts"].append(text)
        progress["diversities"].append(bigram_diversity(text))
        progress["distinct2s"].append(distinct_n(text, 2))
        progress["gen_times"].append(elapsed)
        progress["text_lengths"].append(len(text.split()))
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

    rng = np.random.RandomState(42)
    boot_medians = [float(np.median(rng.choice(valid_ppls, len(valid_ppls), True)))
                    for _ in range(1000)]

    result = {
        "method": method_name, "n_prompts": N_PROMPTS,
        "n_valid": len(valid_ppls),
        "n_empty": sum(1 for t in progress["texts"] if len(t.strip()) < 10),
        "ppl_mean": float(np.mean(valid_ppls)),
        "ppl_median": float(np.median(valid_ppls)),
        "ppl_ci95": [float(np.percentile(boot_medians, 2.5)),
                     float(np.percentile(boot_medians, 97.5))],
        "ppl_p90": float(np.percentile(valid_ppls, 90)),
        "diversity_mean": float(np.mean(progress["diversities"])),
        "distinct2_mean": float(np.mean(progress["distinct2s"])),
        "n_degenerate": n_deg,
        "avg_gen_time": float(np.mean(progress["gen_times"])),
        "avg_text_len": float(np.mean(progress["text_lengths"])),
        "n_short": sum(1 for l in progress["text_lengths"] if l < 15),
        "all_ppls": valid_ppls,
    }

    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {method_name}: PPL(med)={result['ppl_median']:.2f} "
          f"[{result['ppl_ci95'][0]:.1f},{result['ppl_ci95'][1]:.1f}] "
          f"mean={result['ppl_mean']:.1f} div={result['diversity_mean']:.3f} "
          f"empty={result['n_empty']} short={result['n_short']} "
          f"time={result['avg_gen_time']:.1f}s len={result['avg_text_len']:.0f}w")
    return result


def print_summary():
    from scipy import stats as scipy_stats
    methods = ["vanilla_t04", "vanilla_t08", "tcr_t04",
               "anneal_warm", "parallel_vote"]
    results = {}
    for m in methods:
        f = RESULTS_DIR / f"{m}_ppls.json"
        if f.exists():
            results[m] = json.load(open(f))

    if not results:
        print("No results yet")
        return

    baseline = "vanilla_t04" if "vanilla_t04" in results else list(results.keys())[0]
    v = results.get(baseline)

    print(f"\n{'='*115}")
    print(f"TCR v3: Dream-7B with temp=0.4 (121 prompts)")
    print(f"{'='*115}")
    print(f"\n{'Method':<18} {'PPL(med)':>9} {'95%CI':>14} {'Mean':>8} {'P90':>8} "
          f"{'Div':>6} {'Empty':>6} {'Short':>6} {'Time':>6} {'Len':>5} {'vs Base':>14}")
    print("-" * 115)

    for m in methods:
        if m not in results: continue
        r = results[m]
        ci = f"[{r['ppl_ci95'][0]:.1f},{r['ppl_ci95'][1]:.1f}]"
        delta = ""
        if m != baseline and v:
            d = (r["ppl_median"] - v["ppl_median"]) / v["ppl_median"] * 100
            n = min(len(v["all_ppls"]), len(r["all_ppls"]))
            try:
                _, p = scipy_stats.wilcoxon(v["all_ppls"][:n], r["all_ppls"][:n])
                delta = f"{d:+.1f}% p={p:.3f}"
            except: delta = f"{d:+.1f}%"
        print(f"  {m:<16} {r['ppl_median']:>9.2f} {ci:>14} {r['ppl_mean']:>8.1f} "
              f"{r['ppl_p90']:>8.1f} {r['diversity_mean']:>6.3f} "
              f"{r['n_empty']:>6} {r['n_short']:>6} {r['avg_gen_time']:>6.1f} "
              f"{r['avg_text_len']:>5.0f} {delta:>14}")
    print(f"{'='*115}")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "auto"
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    device = f"cuda:{gpu}"

    configs = {
        "vanilla_t04": {},
        "vanilla_t08": {},
        "tcr_t04": {},
        "anneal_warm": {},
        "parallel_vote": {},
    }

    if task == "auto":
        for name in configs:
            p_file = RESULTS_DIR / f"{name}_progress.json"
            need = not p_file.exists()
            if not need:
                with open(p_file) as f:
                    need = json.load(f)["n_done"] < N_PROMPTS
            if need:
                model, tokenizer = load_dream(device)
                prompts = make_prompts(tokenizer, N_PROMPTS)
                run_experiment(name, model, tokenizer, prompts,
                             device=device, **configs[name])
                del model; torch.cuda.empty_cache()
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
        del model; torch.cuda.empty_cache()
        p = json.load(open(RESULTS_DIR / f"{task}_progress.json"))
        if p["n_done"] >= N_PROMPTS:
            run_ppl_eval(task, device)
    else:
        print(f"Unknown: {task}. Available: {', '.join(configs.keys())}")


if __name__ == "__main__":
    main()
