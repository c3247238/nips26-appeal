"""
Pilot Phase 2: Method evaluation for Dream-7B.
Task_2a (TCR ablation), Task_2b (temp annealing), Task_2c (parallel vote),
Task_2d (entropy-guided remasking).

16 prompts, seed=42, 128 gen tokens, 128 denoising steps.

Usage:
  python3 pilot_phase2_methods.py <task> [gpu_id]

  task: tcr | anneal | vote | entropy | all
  gpu_id: 0-7 (default: 0)

Examples:
  python3 pilot_phase2_methods.py tcr 0      # TCR ablation on GPU 0
  python3 pilot_phase2_methods.py anneal 1    # Temp annealing on GPU 1
  python3 pilot_phase2_methods.py entropy 0   # Entropy remasking on GPU 0
  python3 pilot_phase2_methods.py vote 0      # Parallel vote on GPU 0
"""
import os, sys, json, math, time
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilot")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
GPT2_DIR = "/home/ccwang/sibyl_system/models/gpt2"
N_PROMPTS = 16
SEED = 42
GEN_LEN = 128
STEPS = 128
MASK_TOKEN_ID = 151666


# ============================================================
# Shared utilities
# ============================================================
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


def decode_generation(seq, prompt_len, tokenizer, gen_len=GEN_LEN):
    total_len = seq.shape[1]
    gen_ids = seq[0, prompt_len:min(prompt_len + gen_len, total_len)].tolist()
    gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]
    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()


def compute_ppl_batch(texts, device="cuda:0"):
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


def summarize_results(method_results, baseline_ppls=None):
    """Print summary table."""
    print(f"\n{'Method':<30} {'PPL(med)':>9} {'PPL(mean)':>10} {'Div':>6} "
          f"{'Deg':>4} {'Short':>6} {'Time':>6} {'Len':>5} {'vs Base':>10}")
    print("-" * 100)
    for name, r in sorted(method_results.items()):
        delta = ""
        if baseline_ppls and r.get("ppl_median"):
            base_med = float(np.median(baseline_ppls))
            d = (r["ppl_median"] - base_med) / base_med * 100
            delta = f"{d:+.1f}%"
        ppl_med = r.get("ppl_median", 0) or 0
        ppl_mean = r.get("ppl_mean", 0) or 0
        print(f"  {name:<28} {ppl_med:>9.2f} {ppl_mean:>10.1f} "
              f"{r['diversity_mean']:>6.3f} {r['n_degenerate']:>4} "
              f"{r['n_short']:>6} {r['avg_gen_time']:>6.1f} "
              f"{r['avg_text_len']:>5.0f} {delta:>10}")


def evaluate_method(name, texts, diversities, distinct2s, gen_times,
                    text_lengths, infos, device):
    """Compute PPL and package results."""
    ppls = compute_ppl_batch(texts, device)
    valid_ppls = [p for p in ppls if p is not None]
    n_deg = sum(1 for d in diversities if d < 0.3)

    result = {
        "method": name,
        "n_prompts": len(texts),
        "n_valid": len(valid_ppls),
        "n_empty": sum(1 for t in texts if len(t.strip()) < 10),
        "ppl_mean": float(np.mean(valid_ppls)) if valid_ppls else None,
        "ppl_median": float(np.median(valid_ppls)) if valid_ppls else None,
        "diversity_mean": float(np.mean(diversities)),
        "distinct2_mean": float(np.mean(distinct2s)),
        "n_degenerate": n_deg,
        "avg_gen_time": float(np.mean(gen_times)),
        "avg_text_len": float(np.mean(text_lengths)),
        "n_short": sum(1 for l in text_lengths if l < 15),
        "all_ppls": [float(p) for p in valid_ppls],
        "texts": texts[:5],
        "infos": infos[:3] if infos else [],
    }
    return result


# ============================================================
# Task 2a: TCR Ablation
# ============================================================
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


def redenoise(model, seq, mask_token_id, refine_steps, temperature, device):
    """Re-denoise masked positions in a sequence."""
    eps = 1e-3
    timesteps = torch.linspace(1, eps, refine_steps + 1, device=device)
    for step_i in range(refine_steps):
        mask_index = (seq == mask_token_id)
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
        x0 = torch.zeros_like(seq[mask_index], device=device, dtype=torch.long) + mask_token_id
        transfer_idx = torch.rand(*x0.shape, device=device) < p_transfer
        probs = torch.softmax(mask_logits, dim=-1)
        if temperature > 0:
            import torch.distributions as dists
            try:
                sampled = dists.Categorical(probs=probs).sample()
            except:
                sampled = probs.argmax(dim=-1)
        else:
            sampled = probs.argmax(dim=-1)
        x0[transfer_idx] = sampled[transfer_idx]
        seq[mask_index] = x0.clone()
    return seq


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

    seq = redenoise(model, seq, MASK_TOKEN_ID, refine_steps, temperature, device)

    info = {
        "n_remasked": n_remask,
        "mean_stability": float(stability.mean()),
        "remasked_stability": float(stability[unstable_positions].mean()),
    }
    return seq, info


def run_tcr_ablation(model, tokenizer, prompts, device="cuda:0"):
    """Task 2a: TCR ablation over remask_ratio x refine_steps."""
    configs = [
        {"remask_ratio": 0.2, "refine_steps": 32},
        {"remask_ratio": 0.2, "refine_steps": 64},
        {"remask_ratio": 0.3, "refine_steps": 32},
        {"remask_ratio": 0.3, "refine_steps": 64},
        {"remask_ratio": 0.5, "refine_steps": 32},
        {"remask_ratio": 0.5, "refine_steps": 64},
    ]

    results = {}
    for cfg in configs:
        rr = cfg["remask_ratio"]
        rs = cfg["refine_steps"]
        name = f"tcr_r{int(rr*100)}_s{rs}"
        print(f"\n{'='*60}")
        print(f"TCR: {name} ({N_PROMPTS} prompts)")
        print(f"{'='*60}")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        texts, divs, d2s, times, lens, infos = [], [], [], [], [], []
        for pi in range(N_PROMPTS):
            t0 = time.time()
            seq, info = generate_tcr(model, tokenizer, prompts[pi],
                                     device=device, temperature=0.4,
                                     remask_ratio=rr, refine_steps=rs)
            elapsed = time.time() - t0
            text = decode_generation(seq, len(prompts[pi]), tokenizer)
            texts.append(text)
            divs.append(bigram_diversity(text))
            d2s.append(distinct_n(text, 2))
            times.append(elapsed)
            lens.append(len(text.split()))
            infos.append(info)

            if (pi + 1) % 8 == 0:
                print(f"  [{pi+1}/{N_PROMPTS}] div={np.mean(divs[-8:]):.3f} "
                      f"time={np.mean(times[-8:]):.1f}s")

        results[name] = evaluate_method(name, texts, divs, d2s, times, lens, infos, device)
        print(f"  {name}: PPL(med)={results[name]['ppl_median']:.2f} "
              f"div={results[name]['diversity_mean']:.3f}")

    out_file = RESULTS_DIR / "task2a_tcr_ablation.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTCR ablation saved to {out_file}")
    return results


# ============================================================
# Task 2b: Temperature Annealing
# ============================================================
def generate_anneal(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                    steps=STEPS, device="cuda:0",
                    t_high=0.8, t_low=0.2, schedule="linear"):
    """Temperature annealing: warm->cold during denoising."""
    input_ids = torch.tensor([prompt_ids], device=device)

    def logits_hook(step, x, logits):
        if logits is None or step is None:
            return logits
        progress = step / max(steps - 1, 1)
        if schedule == "linear":
            temp = t_high * (1 - progress) + t_low * progress
        elif schedule == "cosine":
            # Cosine decay from t_high to t_low
            temp = t_low + (t_high - t_low) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            temp = t_high * (1 - progress) + t_low * progress
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


def run_temp_annealing(model, tokenizer, prompts, device="cuda:0"):
    """Task 2b: Temperature annealing with 3 schemes."""
    configs = [
        {"t_high": 0.8, "t_low": 0.2, "schedule": "linear", "name": "anneal_lin_08_02"},
        {"t_high": 0.6, "t_low": 0.2, "schedule": "linear", "name": "anneal_lin_06_02"},
        {"t_high": 0.8, "t_low": 0.2, "schedule": "cosine", "name": "anneal_cos_08_02"},
    ]

    results = {}
    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Annealing: {name} ({N_PROMPTS} prompts)")
        print(f"{'='*60}")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        texts, divs, d2s, times, lens = [], [], [], [], []
        for pi in range(N_PROMPTS):
            t0 = time.time()
            seq = generate_anneal(model, tokenizer, prompts[pi],
                                  device=device,
                                  t_high=cfg["t_high"], t_low=cfg["t_low"],
                                  schedule=cfg["schedule"])
            elapsed = time.time() - t0
            text = decode_generation(seq, len(prompts[pi]), tokenizer)
            texts.append(text)
            divs.append(bigram_diversity(text))
            d2s.append(distinct_n(text, 2))
            times.append(elapsed)
            lens.append(len(text.split()))

            if (pi + 1) % 8 == 0:
                print(f"  [{pi+1}/{N_PROMPTS}] div={np.mean(divs[-8:]):.3f} "
                      f"time={np.mean(times[-8:]):.1f}s")

        results[name] = evaluate_method(name, texts, divs, d2s, times, lens, [], device)
        print(f"  {name}: PPL(med)={results[name]['ppl_median']:.2f} "
              f"div={results[name]['diversity_mean']:.3f}")

    out_file = RESULTS_DIR / "task2b_temp_annealing.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nTemp annealing saved to {out_file}")
    return results


# ============================================================
# Task 2c: Parallel Vote
# ============================================================
def generate_parallel_vote(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                           steps=STEPS, device="cuda:0", temperature=0.4,
                           n_trajectories=4, heal_steps=16):
    """K parallel trajectories with majority vote + healing."""
    prompt_len = len(prompt_ids)
    all_gen_ids = []

    for traj_i in range(n_trajectories):
        torch.manual_seed(SEED + traj_i * 1000)
        torch.cuda.manual_seed(SEED + traj_i * 1000)
        seq = generate_vanilla(model, tokenizer, prompt_ids, gen_len, steps,
                               device, temperature)
        gen_ids = seq[0, prompt_len:prompt_len + gen_len].tolist()
        all_gen_ids.append(gen_ids)

    # Position-wise majority vote
    voted_ids = []
    for pos in range(gen_len):
        tokens_at_pos = [all_gen_ids[t][pos] for t in range(n_trajectories)
                        if pos < len(all_gen_ids[t])]
        if not tokens_at_pos:
            voted_ids.append(MASK_TOKEN_ID)
            continue
        counter = Counter(tokens_at_pos)
        winner = counter.most_common(1)[0][0]
        voted_ids.append(winner)

    result = torch.tensor([prompt_ids + voted_ids], device=device)

    # Find disagreement positions
    disagreement = []
    for pos in range(gen_len):
        tokens_at_pos = set(
            all_gen_ids[t][pos] for t in range(n_trajectories)
            if pos < len(all_gen_ids[t]))
        if len(tokens_at_pos) > 1:
            disagreement.append(pos)

    # Remask disagreement positions and heal
    if len(disagreement) > 0 and heal_steps > 0:
        for pos in disagreement:
            result[0, prompt_len + pos] = MASK_TOKEN_ID
        result = redenoise(model, result, MASK_TOKEN_ID, heal_steps, temperature, device)

    info = {
        "n_disagreement": len(disagreement),
        "n_trajectories": n_trajectories,
        "disagreement_ratio": len(disagreement) / gen_len,
    }
    return result, info


def run_parallel_vote(model, tokenizer, prompts, device="cuda:0"):
    """Task 2c: Parallel vote with K=4."""
    configs = [
        {"n_trajectories": 4, "heal_steps": 0, "name": "vote_k4_noheal"},
        {"n_trajectories": 4, "heal_steps": 16, "name": "vote_k4_heal16"},
    ]

    results = {}
    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Parallel Vote: {name} ({N_PROMPTS} prompts)")
        print(f"{'='*60}")

        texts, divs, d2s, times, lens, infos = [], [], [], [], [], []
        for pi in range(N_PROMPTS):
            torch.manual_seed(SEED)
            torch.cuda.manual_seed(SEED)
            t0 = time.time()
            seq, info = generate_parallel_vote(
                model, tokenizer, prompts[pi], device=device,
                temperature=0.4, n_trajectories=cfg["n_trajectories"],
                heal_steps=cfg["heal_steps"])
            elapsed = time.time() - t0
            text = decode_generation(seq, len(prompts[pi]), tokenizer)
            texts.append(text)
            divs.append(bigram_diversity(text))
            d2s.append(distinct_n(text, 2))
            times.append(elapsed)
            lens.append(len(text.split()))
            infos.append(info)

            if (pi + 1) % 8 == 0:
                avg_disagree = np.mean([inf["disagreement_ratio"] for inf in infos[-8:]])
                print(f"  [{pi+1}/{N_PROMPTS}] div={np.mean(divs[-8:]):.3f} "
                      f"time={np.mean(times[-8:]):.1f}s disagree={avg_disagree:.2f}")

        results[name] = evaluate_method(name, texts, divs, d2s, times, lens, infos, device)
        print(f"  {name}: PPL(med)={results[name]['ppl_median']:.2f} "
              f"div={results[name]['diversity_mean']:.3f}")

    out_file = RESULTS_DIR / "task2c_parallel_vote.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nParallel vote saved to {out_file}")
    return results


# ============================================================
# Task 2d: Entropy-Guided Remasking
# ============================================================
class EntropyRecorder:
    """Hook that records per-position entropy of predictions at each step."""
    def __init__(self, prompt_len, gen_len):
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.entropies = defaultdict(list)  # pos -> list of entropy values
        self.step_count = 0

    def __call__(self, step, x, logits):
        if logits is None:
            return logits
        gen_start = self.prompt_len
        gen_end = self.prompt_len + self.gen_len
        for pos in range(gen_start, min(gen_end, x.shape[1])):
            if x[0, pos].item() == MASK_TOKEN_ID:
                rel_pos = pos - gen_start
                # Compute entropy of the prediction distribution
                probs = F.softmax(logits[0, pos], dim=-1)
                # Use top-k for efficiency (full vocab entropy is noisy)
                topk_probs, _ = probs.topk(min(100, probs.shape[0]))
                topk_probs = topk_probs / topk_probs.sum()  # renormalize
                entropy = -(topk_probs * topk_probs.log().clamp(min=-20)).sum().item()
                self.entropies[rel_pos].append(entropy)
        self.step_count += 1
        return logits

    def get_final_entropy(self):
        """Get the entropy from the last recorded step for each position."""
        final_entropy = np.zeros(self.gen_len)
        for pos in range(self.gen_len):
            if pos in self.entropies and len(self.entropies[pos]) > 0:
                # Use the last (final step) entropy
                final_entropy[pos] = self.entropies[pos][-1]
            else:
                final_entropy[pos] = 0.0  # Already decided, low uncertainty
        return final_entropy

    def get_mean_entropy(self):
        """Get mean entropy across all steps for each position."""
        mean_entropy = np.zeros(self.gen_len)
        for pos in range(self.gen_len):
            if pos in self.entropies and len(self.entropies[pos]) > 0:
                mean_entropy[pos] = np.mean(self.entropies[pos])
            else:
                mean_entropy[pos] = 0.0
        return mean_entropy


def generate_entropy_remask(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                            steps=STEPS, device="cuda:0", temperature=0.4,
                            remask_ratio=0.3, refine_steps=64,
                            entropy_type="mean"):
    """Entropy-guided remasking: remask positions with highest prediction entropy."""
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], device=device)
    recorder = EntropyRecorder(prompt_len, gen_len)

    output = model.diffusion_generate(
        input_ids, max_new_tokens=gen_len, output_history=False,
        return_dict_in_generate=True, steps=steps,
        temperature=temperature, alg="origin",
        generation_logits_hook_func=recorder,
    )
    seq = output.sequences

    if entropy_type == "mean":
        entropy = recorder.get_mean_entropy()
    else:
        entropy = recorder.get_final_entropy()

    n_remask = int(remask_ratio * gen_len)
    if n_remask < 3:
        return seq, {"n_remasked": 0, "mean_entropy": float(entropy.mean())}

    # High entropy = most uncertain -> remask
    high_entropy_positions = np.argsort(entropy)[-n_remask:]  # highest entropy
    for pos in high_entropy_positions:
        seq[0, prompt_len + pos] = MASK_TOKEN_ID

    seq = redenoise(model, seq, MASK_TOKEN_ID, refine_steps, temperature, device)

    info = {
        "n_remasked": n_remask,
        "mean_entropy": float(entropy.mean()),
        "remasked_entropy": float(entropy[high_entropy_positions].mean()),
        "kept_entropy": float(np.delete(entropy, high_entropy_positions).mean()),
        "entropy_type": entropy_type,
    }
    return seq, info


def run_entropy_remasking(model, tokenizer, prompts, device="cuda:0"):
    """Task 2d: Entropy-guided remasking ablation."""
    configs = [
        {"remask_ratio": 0.2, "refine_steps": 64, "entropy_type": "mean",
         "name": "entropy_r20_mean"},
        {"remask_ratio": 0.3, "refine_steps": 64, "entropy_type": "mean",
         "name": "entropy_r30_mean"},
        {"remask_ratio": 0.5, "refine_steps": 64, "entropy_type": "mean",
         "name": "entropy_r50_mean"},
    ]

    results = {}
    for cfg in configs:
        name = cfg["name"]
        print(f"\n{'='*60}")
        print(f"Entropy Remasking: {name} ({N_PROMPTS} prompts)")
        print(f"{'='*60}")

        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)

        texts, divs, d2s, times, lens, infos = [], [], [], [], [], []
        for pi in range(N_PROMPTS):
            t0 = time.time()
            seq, info = generate_entropy_remask(
                model, tokenizer, prompts[pi], device=device,
                temperature=0.4, remask_ratio=cfg["remask_ratio"],
                refine_steps=cfg["refine_steps"],
                entropy_type=cfg["entropy_type"])
            elapsed = time.time() - t0
            text = decode_generation(seq, len(prompts[pi]), tokenizer)
            texts.append(text)
            divs.append(bigram_diversity(text))
            d2s.append(distinct_n(text, 2))
            times.append(elapsed)
            lens.append(len(text.split()))
            infos.append(info)

            if (pi + 1) % 8 == 0:
                avg_ent = np.mean([inf["mean_entropy"] for inf in infos[-8:]])
                print(f"  [{pi+1}/{N_PROMPTS}] div={np.mean(divs[-8:]):.3f} "
                      f"time={np.mean(times[-8:]):.1f}s entropy={avg_ent:.2f}")

        results[name] = evaluate_method(name, texts, divs, d2s, times, lens, infos, device)
        print(f"  {name}: PPL(med)={results[name]['ppl_median']:.2f} "
              f"div={results[name]['diversity_mean']:.3f}")

    out_file = RESULTS_DIR / "task2d_entropy_remasking.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nEntropy remasking saved to {out_file}")
    return results


# ============================================================
# Main
# ============================================================
def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "all"
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    device = f"cuda:{gpu}"

    print(f"=== Pilot Phase 2: Method Evaluation ===")
    print(f"Task: {task}, GPU: {gpu}, Prompts: {N_PROMPTS}, Seed: {SEED}")

    # Load baseline for comparison
    baseline_ppls = None
    baseline_file = RESULTS_DIR / "task1_baselines.json"
    if baseline_file.exists():
        with open(baseline_file) as f:
            baselines = json.load(f)
        if "vanilla_t04" in baselines:
            baseline_ppls = baselines["vanilla_t04"].get("all_ppls")

    model, tokenizer = load_dream(device)
    prompts = make_prompts(tokenizer, N_PROMPTS)

    all_results = {}

    if task in ("tcr", "all"):
        results = run_tcr_ablation(model, tokenizer, prompts, device)
        all_results.update(results)

    if task in ("anneal", "all"):
        results = run_temp_annealing(model, tokenizer, prompts, device)
        all_results.update(results)

    if task in ("vote", "all"):
        results = run_parallel_vote(model, tokenizer, prompts, device)
        all_results.update(results)

    if task in ("entropy", "all"):
        results = run_entropy_remasking(model, tokenizer, prompts, device)
        all_results.update(results)

    del model
    torch.cuda.empty_cache()

    if all_results:
        print(f"\n{'='*100}")
        print(f"Phase 2 Summary")
        print(f"{'='*100}")
        summarize_results(all_results, baseline_ppls)

        # Save combined results
        combined_file = RESULTS_DIR / f"task2_{task}_combined.json"
        with open(combined_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to {combined_file}")

    # Pass/fail check
    if baseline_ppls:
        base_med = float(np.median(baseline_ppls))
        print(f"\n--- Pass/Fail Check (vs vanilla_t04 median PPL={base_med:.2f}) ---")
        any_pass = False
        for name, r in sorted(all_results.items()):
            if r.get("ppl_median") and r["ppl_median"] < base_med:
                div_ok = r["diversity_mean"] >= 0.93
                deg_ok = r["n_degenerate"] == 0
                status = "GO" if (div_ok and deg_ok) else "CAUTION"
                print(f"  {name}: PPL={r['ppl_median']:.2f} ({(r['ppl_median']-base_med)/base_med*100:+.1f}%) "
                      f"div={r['diversity_mean']:.3f} -> {status}")
                if status == "GO":
                    any_pass = True
            else:
                ppl = r.get("ppl_median", 0) or 0
                print(f"  {name}: PPL={ppl:.2f} ({(ppl-base_med)/base_med*100:+.1f}%) -> NO-GO")
        print(f"\n  Overall: {'At least one GO' if any_pass else 'No method passed'}")


if __name__ == "__main__":
    main()
