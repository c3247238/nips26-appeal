"""
Full-Scale Validation (Task 3a): Best methods from pilot phase.
121 prompts x 3 seeds = 363 samples per method.

Methods:
  1. vanilla_t04 (baseline)
  2. entropy_r20_mean (best pilot: -24.9% PPL)
  3. tcr_r30_s32 (2nd best: -22.9% PPL)
  4. anneal_lin_06_02 (3rd best: -16.5% PPL, zero overhead)

Usage:
  python3 full_task3a_validation.py <method> <gpu_id> [seed]

  method: vanilla | entropy | tcr | anneal | all
  gpu_id: 0-7
  seed: 42 | 123 | 456 | all (default: all)

Examples:
  python3 full_task3a_validation.py vanilla 0 all    # vanilla on GPU 0, all seeds
  python3 full_task3a_validation.py entropy 1 42     # entropy on GPU 1, seed=42
  python3 full_task3a_validation.py all 0 all        # all methods sequentially
"""
import os, sys, json, math, time, hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# Configuration
# ============================================================
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
GPT2_DIR = "/home/ccwang/sibyl_system/models/gpt2"

ALL_SEEDS = [42, 123, 456]
GEN_LEN = 128
STEPS = 128
MASK_TOKEN_ID = 151666

# 121 diverse prompts for full-scale evaluation
QUESTIONS = [
    # Science (20)
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
    "How does quantum entanglement work?",
    "What is the theory of evolution?",
    "Explain the concept of entropy in thermodynamics.",
    "How does CRISPR gene editing work?",
    # Technology (20)
    "What is cloud computing?",
    "How does encryption protect data?",
    "Explain the difference between AI and machine learning.",
    "What is the Internet of Things?",
    "How do self-driving cars work?",
    "What is 5G technology?",
    "Explain how a compiler works.",
    "What is virtualization in computing?",
    "How does a search engine rank results?",
    "What is edge computing?",
    "Explain the concept of microservices architecture.",
    "How does natural language processing work?",
    "What is a neural network transformer?",
    "Explain how recommendation systems work.",
    "What is federated learning?",
    "How does a database index improve query performance?",
    "What is container orchestration?",
    "Explain the CAP theorem in distributed systems.",
    "What is homomorphic encryption?",
    "How does a garbage collector work in programming languages?",
    # History & Society (20)
    "What caused World War I?",
    "Explain the significance of the Renaissance.",
    "How did the Industrial Revolution change society?",
    "What was the Cold War?",
    "Explain the French Revolution.",
    "What is democracy and how did it develop?",
    "How did ancient Rome fall?",
    "What was the Silk Road?",
    "Explain the impact of the printing press.",
    "What caused the Great Depression?",
    "How did the Internet change communication?",
    "What is globalization?",
    "Explain the civil rights movement.",
    "What was the Space Race?",
    "How did agriculture begin?",
    "What is the United Nations and what does it do?",
    "Explain the concept of human rights.",
    "What was the Enlightenment?",
    "How has urbanization affected the world?",
    "What is the significance of the Magna Carta?",
    # Philosophy & Abstract (20)
    "What is the meaning of consciousness?",
    "Explain the trolley problem in ethics.",
    "What is existentialism?",
    "How do we define knowledge?",
    "What is the mind-body problem?",
    "Explain utilitarianism.",
    "What is free will?",
    "How does language shape thought?",
    "What is the social contract theory?",
    "Explain the concept of justice.",
    "What is moral relativism?",
    "How do we distinguish truth from belief?",
    "What is the nature of time?",
    "Explain Plato's allegory of the cave.",
    "What is the paradox of the Ship of Theseus?",
    "How does art reflect society?",
    "What is the role of education in society?",
    "Explain the concept of infinity.",
    "What makes something beautiful?",
    "How do cultural values differ across societies?",
    # Practical & Applied (21)
    "How does a vaccine work?",
    "What are the benefits of exercise?",
    "Explain how solar panels generate electricity.",
    "What is a balanced diet?",
    "How does recycling help the environment?",
    "What is mental health and why is it important?",
    "How do airplanes fly?",
    "What is sustainable development?",
    "Explain how a battery stores energy.",
    "What are the causes and effects of deforestation?",
    "How does the stock market work?",
    "What is inflation and how does it affect the economy?",
    "Explain the greenhouse effect.",
    "How does a microwave oven heat food?",
    "What is renewable energy?",
    "How do bridges support weight?",
    "What is the scientific method?",
    "Explain how GPS navigation works.",
    "What are antibiotics and how do they work?",
    "How does a refrigerator keep food cold?",
    "What is the role of sleep in human health?",
]

N_PROMPTS = len(QUESTIONS)  # 121


# ============================================================
# Shared utilities (same as pilot, but with multi-seed support)
# ============================================================
def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream loaded on {device}, VRAM: {torch.cuda.memory_allocated(device)/1e9:.1f}GB")
    return model, tokenizer


def make_prompts(tokenizer):
    prompts = []
    for q in QUESTIONS:
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


def compute_ppl_batch(texts, device="cuda:0", batch_size=16):
    """Compute GPT-2 PPL for a list of texts. More efficient batched version."""
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


# ============================================================
# Method implementations (from pilot, verified)
# ============================================================

# --- TCR ---
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
                 remask_ratio=0.3, refine_steps=32):
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


# --- Temperature Annealing ---
def generate_anneal(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                    steps=STEPS, device="cuda:0",
                    t_high=0.6, t_low=0.2, schedule="linear"):
    input_ids = torch.tensor([prompt_ids], device=device)

    def logits_hook(step, x, logits):
        if logits is None or step is None:
            return logits
        progress = step / max(steps - 1, 1)
        if schedule == "linear":
            temp = t_high * (1 - progress) + t_low * progress
        elif schedule == "cosine":
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


# --- Entropy-Guided Remasking ---
class EntropyRecorder:
    def __init__(self, prompt_len, gen_len):
        self.prompt_len = prompt_len
        self.gen_len = gen_len
        self.entropies = defaultdict(list)
        self.step_count = 0

    def __call__(self, step, x, logits):
        if logits is None:
            return logits
        gen_start = self.prompt_len
        gen_end = self.prompt_len + self.gen_len
        for pos in range(gen_start, min(gen_end, x.shape[1])):
            if x[0, pos].item() == MASK_TOKEN_ID:
                rel_pos = pos - gen_start
                probs = F.softmax(logits[0, pos], dim=-1)
                topk_probs, _ = probs.topk(min(100, probs.shape[0]))
                topk_probs = topk_probs / topk_probs.sum()
                entropy = -(topk_probs * topk_probs.log().clamp(min=-20)).sum().item()
                self.entropies[rel_pos].append(entropy)
        self.step_count += 1
        return logits

    def get_mean_entropy(self):
        mean_entropy = np.zeros(self.gen_len)
        for pos in range(self.gen_len):
            if pos in self.entropies and len(self.entropies[pos]) > 0:
                mean_entropy[pos] = np.mean(self.entropies[pos])
            else:
                mean_entropy[pos] = 0.0
        return mean_entropy


def generate_entropy_remask(model, tokenizer, prompt_ids, gen_len=GEN_LEN,
                            steps=STEPS, device="cuda:0", temperature=0.4,
                            remask_ratio=0.2, refine_steps=64):
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

    entropy = recorder.get_mean_entropy()
    n_remask = int(remask_ratio * gen_len)
    if n_remask < 3:
        return seq, {"n_remasked": 0, "mean_entropy": float(entropy.mean())}

    high_entropy_positions = np.argsort(entropy)[-n_remask:]
    for pos in high_entropy_positions:
        seq[0, prompt_len + pos] = MASK_TOKEN_ID

    seq = redenoise(model, seq, MASK_TOKEN_ID, refine_steps, temperature, device)

    info = {
        "n_remasked": n_remask,
        "mean_entropy": float(entropy.mean()),
        "remasked_entropy": float(entropy[high_entropy_positions].mean()),
    }
    return seq, info


# ============================================================
# Full-scale runner with checkpoint/resume
# ============================================================
def get_checkpoint_path(method, seed):
    return RESULTS_DIR / f"checkpoint_{method}_s{seed}.json"


def load_checkpoint(method, seed):
    cp_path = get_checkpoint_path(method, seed)
    if cp_path.exists():
        with open(cp_path) as f:
            return json.load(f)
    return None


def save_checkpoint(method, seed, data):
    cp_path = get_checkpoint_path(method, seed)
    with open(cp_path, "w") as f:
        json.dump(data, f, indent=2)


def run_method_seed(model, tokenizer, prompts, method, seed, device):
    """Run one method for one seed across all prompts. Supports resume."""
    checkpoint = load_checkpoint(method, seed)
    if checkpoint and checkpoint.get("complete"):
        print(f"  [{method} seed={seed}] Already complete, skipping.")
        return checkpoint

    start_idx = 0
    texts, divs, d2s, times, lens, infos = [], [], [], [], [], []

    if checkpoint:
        start_idx = checkpoint.get("last_idx", 0) + 1
        texts = checkpoint.get("texts", [])
        divs = checkpoint.get("diversities", [])
        d2s = checkpoint.get("distinct2s", [])
        times = checkpoint.get("gen_times", [])
        lens = checkpoint.get("text_lengths", [])
        infos = checkpoint.get("infos", [])
        print(f"  [{method} seed={seed}] Resuming from prompt {start_idx}/{N_PROMPTS}")

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # Burn through RNG to match position if resuming
    if start_idx > 0:
        # Re-seed to ensure deterministic generation from this point
        torch.manual_seed(seed + start_idx * 7919)
        torch.cuda.manual_seed(seed + start_idx * 7919)

    t_total_start = time.time()
    for pi in range(start_idx, N_PROMPTS):
        # Per-prompt deterministic seed
        torch.manual_seed(seed + pi * 7919)
        torch.cuda.manual_seed(seed + pi * 7919)

        t0 = time.time()
        try:
            if method == "vanilla":
                seq = generate_vanilla(model, tokenizer, prompts[pi],
                                       device=device, temperature=0.4)
                info = {}
            elif method == "entropy":
                seq, info = generate_entropy_remask(
                    model, tokenizer, prompts[pi], device=device,
                    temperature=0.4, remask_ratio=0.2, refine_steps=64)
            elif method == "tcr":
                seq, info = generate_tcr(
                    model, tokenizer, prompts[pi], device=device,
                    temperature=0.4, remask_ratio=0.3, refine_steps=32)
            elif method == "anneal":
                seq = generate_anneal(
                    model, tokenizer, prompts[pi], device=device,
                    t_high=0.6, t_low=0.2, schedule="linear")
                info = {}
            else:
                raise ValueError(f"Unknown method: {method}")

            elapsed = time.time() - t0
            text = decode_generation(seq, len(prompts[pi]), tokenizer)

        except Exception as e:
            elapsed = time.time() - t0
            text = ""
            info = {"error": str(e)}
            print(f"  WARNING: prompt {pi} failed: {e}")

        texts.append(text)
        divs.append(bigram_diversity(text))
        d2s.append(distinct_n(text, 2))
        times.append(elapsed)
        lens.append(len(text.split()))
        infos.append(info)

        # Progress logging
        if (pi + 1) % 20 == 0 or pi == N_PROMPTS - 1:
            elapsed_total = time.time() - t_total_start
            rate = (pi - start_idx + 1) / elapsed_total
            remaining = (N_PROMPTS - pi - 1) / rate if rate > 0 else 0
            print(f"  [{method} s{seed}] {pi+1}/{N_PROMPTS} "
                  f"div={np.mean(divs[-20:]):.3f} "
                  f"time={np.mean(times[-20:]):.1f}s/prompt "
                  f"ETA={remaining/60:.1f}min")

        # Checkpoint every 30 prompts
        if (pi + 1) % 30 == 0:
            save_checkpoint(method, seed, {
                "method": method, "seed": seed,
                "last_idx": pi,
                "texts": texts, "diversities": divs, "distinct2s": d2s,
                "gen_times": times, "text_lengths": lens,
                "infos": [i for i in infos if not isinstance(i, dict) or "error" not in i][:5],
                "complete": False,
            })

    # Mark complete
    result = {
        "method": method, "seed": seed,
        "last_idx": N_PROMPTS - 1,
        "texts": texts, "diversities": divs, "distinct2s": d2s,
        "gen_times": times, "text_lengths": lens,
        "infos": infos[:5],
        "complete": True,
    }
    save_checkpoint(method, seed, result)
    return result


def compute_statistics(method, all_seed_data, baseline_all_ppls=None):
    """Compute aggregate statistics across seeds with statistical tests."""
    from scipy import stats

    all_ppls_by_prompt = defaultdict(list)  # prompt_idx -> [ppl_s1, ppl_s2, ...]
    all_divs_by_prompt = defaultdict(list)
    all_texts = []
    flat_ppls = []
    flat_divs = []
    flat_d2s = []
    flat_times = []
    flat_lens = []

    for seed, data in all_seed_data.items():
        ppls = data["ppls"]
        for pi, ppl in enumerate(ppls):
            if ppl is not None:
                all_ppls_by_prompt[pi].append(ppl)
        for pi, div in enumerate(data["diversities"]):
            all_divs_by_prompt[pi].append(div)
        flat_ppls.extend([p for p in ppls if p is not None])
        flat_divs.extend(data["diversities"])
        flat_d2s.extend(data["distinct2s"])
        flat_times.extend(data["gen_times"])
        flat_lens.extend(data["text_lengths"])
        all_texts.extend(data["texts"])

    # Per-prompt mean PPL (averaged across seeds)
    prompt_mean_ppls = []
    for pi in range(N_PROMPTS):
        if pi in all_ppls_by_prompt and all_ppls_by_prompt[pi]:
            prompt_mean_ppls.append(np.mean(all_ppls_by_prompt[pi]))

    result = {
        "method": method,
        "n_seeds": len(all_seed_data),
        "seeds": list(all_seed_data.keys()),
        "n_prompts": N_PROMPTS,
        "n_total_samples": len(flat_ppls),
        # PPL stats
        "ppl_median": float(np.median(flat_ppls)) if flat_ppls else None,
        "ppl_mean": float(np.mean(flat_ppls)) if flat_ppls else None,
        "ppl_std": float(np.std(flat_ppls)) if flat_ppls else None,
        "ppl_prompt_mean_median": float(np.median(prompt_mean_ppls)) if prompt_mean_ppls else None,
        "ppl_prompt_mean_mean": float(np.mean(prompt_mean_ppls)) if prompt_mean_ppls else None,
        # Diversity stats
        "diversity_mean": float(np.mean(flat_divs)),
        "diversity_std": float(np.std(flat_divs)),
        "distinct2_mean": float(np.mean(flat_d2s)),
        "n_degenerate": sum(1 for d in flat_divs if d < 0.3),
        "degenerate_pct": sum(1 for d in flat_divs if d < 0.3) / len(flat_divs) * 100,
        # Time stats
        "avg_gen_time": float(np.mean(flat_times)),
        "total_time_min": float(sum(flat_times)) / 60,
        # Length stats
        "avg_text_len": float(np.mean(flat_lens)),
        "n_short": sum(1 for l in flat_lens if l < 15),
        # Per-seed PPL medians (consistency check)
        "per_seed_ppl_medians": {},
        # Raw data for statistical tests
        "prompt_mean_ppls": [float(p) for p in prompt_mean_ppls],
        # Sample texts (5 random)
        "sample_texts": all_texts[:5],
    }

    for seed, data in all_seed_data.items():
        valid = [p for p in data["ppls"] if p is not None]
        result["per_seed_ppl_medians"][str(seed)] = float(np.median(valid)) if valid else None

    # Statistical tests vs baseline
    if baseline_all_ppls is not None:
        baseline_prompt_means = []
        for pi in range(N_PROMPTS):
            if pi in baseline_all_ppls and baseline_all_ppls[pi]:
                baseline_prompt_means.append(np.mean(baseline_all_ppls[pi]))

        # Align: only prompts with both method and baseline data
        paired_method = []
        paired_baseline = []
        for pi in range(N_PROMPTS):
            if (pi in all_ppls_by_prompt and all_ppls_by_prompt[pi] and
                pi in baseline_all_ppls and baseline_all_ppls[pi]):
                paired_method.append(np.mean(all_ppls_by_prompt[pi]))
                paired_baseline.append(np.mean(baseline_all_ppls[pi]))

        if len(paired_method) >= 10:
            paired_method = np.array(paired_method)
            paired_baseline = np.array(paired_baseline)

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(paired_method, paired_baseline)
            # Wilcoxon signed-rank test
            try:
                w_stat, w_pval = stats.wilcoxon(paired_method, paired_baseline)
            except:
                w_stat, w_pval = None, None

            # Bootstrap 95% CI for mean difference
            diffs = paired_method - paired_baseline
            n_boot = 1000
            np.random.seed(42)
            boot_means = []
            for _ in range(n_boot):
                idx = np.random.choice(len(diffs), len(diffs), replace=True)
                boot_means.append(np.mean(diffs[idx]))
            boot_means = np.array(boot_means)
            ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

            base_med = float(np.median([p for pp in baseline_all_ppls.values() for p in pp if p is not None]))
            result["vs_baseline"] = {
                "baseline_ppl_median": base_med,
                "ppl_change_pct": (result["ppl_median"] - base_med) / base_med * 100 if result["ppl_median"] else None,
                "mean_diff": float(np.mean(diffs)),
                "paired_ttest": {"t_stat": float(t_stat), "p_value": float(t_pval)},
                "wilcoxon": {"w_stat": float(w_stat) if w_stat else None,
                            "p_value": float(w_pval) if w_pval else None},
                "bootstrap_95ci": {"low": float(ci_low), "high": float(ci_high)},
                "n_paired": len(paired_method),
                "n_improved": int(np.sum(paired_method < paired_baseline)),
                "n_worsened": int(np.sum(paired_method > paired_baseline)),
                "significant_p005": bool(t_pval < 0.05) if t_pval else False,
            }

    return result


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 3:
        print("Usage: python3 full_task3a_validation.py <method> <gpu_id> [seed]")
        print("  method: vanilla | entropy | tcr | anneal")
        print("  gpu_id: 0-7")
        print("  seed: 42 | 123 | 456 | all (default: all)")
        sys.exit(1)

    method = sys.argv[1]
    gpu = int(sys.argv[2])
    seed_arg = sys.argv[3] if len(sys.argv) > 3 else "all"

    if seed_arg == "all":
        seeds = ALL_SEEDS
    else:
        seeds = [int(seed_arg)]

    device = f"cuda:{gpu}"

    valid_methods = ["vanilla", "entropy", "tcr", "anneal"]
    if method == "all":
        methods = valid_methods
    elif method in valid_methods:
        methods = [method]
    else:
        print(f"Unknown method: {method}. Valid: {valid_methods}")
        sys.exit(1)

    print(f"=== Full-Scale Validation (Task 3a) ===")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds}")
    print(f"GPU: {gpu}, Prompts: {N_PROMPTS}")
    print(f"Gen length: {GEN_LEN}, Steps: {STEPS}")

    # Load model
    model, tokenizer = load_dream(device)
    prompts = make_prompts(tokenizer)
    print(f"Prepared {len(prompts)} prompts")

    # Run each method x seed
    for meth in methods:
        print(f"\n{'='*70}")
        print(f"Method: {meth}")
        print(f"{'='*70}")

        all_seed_data = {}
        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            result = run_method_seed(model, tokenizer, prompts, meth, seed, device)
            all_seed_data[seed] = result

        # Free model for PPL evaluation
        del model
        torch.cuda.empty_cache()

        # Compute PPL for each seed
        print(f"\n--- Computing PPL for {meth} ---")
        baseline_all_ppls = defaultdict(list)

        for seed in seeds:
            data = all_seed_data[seed]
            ppl_cache_path = RESULTS_DIR / f"ppls_{meth}_s{seed}.json"

            if ppl_cache_path.exists():
                with open(ppl_cache_path) as f:
                    ppls = json.load(f)
                print(f"  Loaded cached PPLs for {meth} seed={seed}")
            else:
                print(f"  Computing PPLs for {meth} seed={seed} ({len(data['texts'])} texts)...")
                ppls = compute_ppl_batch(data["texts"], device)
                ppls = [float(p) if p is not None else None for p in ppls]
                with open(ppl_cache_path, "w") as f:
                    json.dump(ppls, f)
                print(f"  Saved PPLs to {ppl_cache_path}")

            data["ppls"] = ppls

            valid_ppls = [p for p in ppls if p is not None]
            print(f"  {meth} seed={seed}: PPL(med)={np.median(valid_ppls):.2f} "
                  f"mean={np.mean(valid_ppls):.1f} n_valid={len(valid_ppls)}")

        # Load baseline PPLs for comparison if available
        baseline_ppls_by_prompt = None
        if meth != "vanilla":
            baseline_ppls_by_prompt = defaultdict(list)
            all_have_baseline = True
            for seed in seeds:
                bp_path = RESULTS_DIR / f"ppls_vanilla_s{seed}.json"
                if bp_path.exists():
                    with open(bp_path) as f:
                        bp = json.load(f)
                    for pi, ppl in enumerate(bp):
                        if ppl is not None:
                            baseline_ppls_by_prompt[pi].append(ppl)
                else:
                    all_have_baseline = False
                    print(f"  WARNING: No baseline PPLs for seed={seed}. Run vanilla first!")
            if not all_have_baseline:
                baseline_ppls_by_prompt = None

        # Compute statistics
        stats_result = compute_statistics(meth, all_seed_data, baseline_ppls_by_prompt)

        # Save full results
        out_path = RESULTS_DIR / f"full_{meth}.json"
        with open(out_path, "w") as f:
            json.dump(stats_result, f, indent=2, default=str)
        print(f"\n  Full results saved to {out_path}")

        # Print summary
        print(f"\n  === {meth} Summary ===")
        print(f"  PPL(median): {stats_result['ppl_median']:.2f}")
        print(f"  PPL(mean):   {stats_result['ppl_mean']:.1f} +/- {stats_result['ppl_std']:.1f}")
        print(f"  Diversity:   {stats_result['diversity_mean']:.3f}")
        print(f"  Degenerate:  {stats_result['n_degenerate']} ({stats_result['degenerate_pct']:.1f}%)")
        print(f"  Avg time:    {stats_result['avg_gen_time']:.1f}s")
        print(f"  Per-seed medians: {stats_result['per_seed_ppl_medians']}")
        if "vs_baseline" in stats_result:
            vb = stats_result["vs_baseline"]
            print(f"  vs Baseline: {vb['ppl_change_pct']:+.1f}%")
            print(f"    Paired t-test: t={vb['paired_ttest']['t_stat']:.3f}, p={vb['paired_ttest']['p_value']:.4f}")
            if vb["wilcoxon"]["p_value"] is not None:
                print(f"    Wilcoxon: p={vb['wilcoxon']['p_value']:.4f}")
            print(f"    Bootstrap 95% CI: [{vb['bootstrap_95ci']['low']:.2f}, {vb['bootstrap_95ci']['high']:.2f}]")
            print(f"    Improved: {vb['n_improved']}/{vb['n_paired']} prompts")
            print(f"    Significant (p<0.05): {vb['significant_p005']}")

        # Reload model for next method
        if meth != methods[-1]:
            model, tokenizer = load_dream(device)
            prompts = make_prompts(tokenizer)

    # Final combined summary
    print(f"\n{'='*80}")
    print(f"FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"{'Method':<20} {'PPL(med)':>9} {'PPL(mean)':>10} {'Div':>6} "
          f"{'Deg%':>5} {'Time':>6} {'vs Base':>10}")
    print("-" * 80)
    for meth in methods:
        fp = RESULTS_DIR / f"full_{meth}.json"
        if fp.exists():
            with open(fp) as f:
                r = json.load(f)
            delta = ""
            if "vs_baseline" in r and r["vs_baseline"].get("ppl_change_pct") is not None:
                delta = f"{r['vs_baseline']['ppl_change_pct']:+.1f}%"
            pmed = r.get("ppl_median") or 0
            pmean = r.get("ppl_mean") or 0
            print(f"  {meth:<18} {pmed:>9.2f} {pmean:>10.1f} "
                  f"{r['diversity_mean']:>6.3f} {r['degenerate_pct']:>5.1f} "
                  f"{r['avg_gen_time']:>6.1f} {delta:>10}")


if __name__ == "__main__":
    main()
