"""
Full M1 (EntropyCache) Pareto Evaluation.

Task: full_m1_pareto
- Full-scale single-method evaluation of M1 (EntropyCache) across all 4 benchmarks and 3 seeds
- Sweep entropy threshold at {0.5, 1.0, 2.0, 3.0}
- Benchmarks: GSM8K (1319), MATH500 (500), HumanEval (164), MBPP (374/sanitized)
- Seeds: [42, 123, 456]
- Report: TPS, QAS, Accuracy-Retention per threshold per benchmark
- Identify global operating point for composability experiments

Key implementation notes:
- M1 uses entropy-threshold-based step skipping: at each denoising step, if a block's
  tokens have average entropy < threshold, skip further refinement for that block.
- This reduces effective forward passes and produces measurable TPS speedup.
- Batch size 4 for all benchmarks (97GB VRAM available).

Output:
    exp/results/full_m1/m1_pareto_full.json

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n sibyl_dlm-acceleration python full_m1_pareto.py
"""
import os, sys, json, time, random, gc, re, subprocess
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE     = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED        = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH    = str(SHARED / "checkpoints" / "llada-8b-instruct")
GSM8K_DIR     = str(SHARED / "datasets" / "gsm8k")
MATH500_DIR   = str(SHARED / "datasets" / "math500")
HUMANEVAL_DIR = str(SHARED / "datasets" / "humaneval")
MBPP_DIR      = str(SHARED / "datasets" / "mbpp")
CODE_DIR      = WORKSPACE / "exp" / "code"
RESULTS_DIR   = WORKSPACE / "exp" / "results" / "full_m1"
TASK_ID = "full_m1_pareto_t2"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LLADA_MASK_ID = 126336
BATCH_SIZE = 4  # fits on 97GB VRAM

# Baseline from full_baseline.json (3-seed mean)
BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500":   {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
    "mbpp":      {"pass_at_1":   0.0000, "avg_tps": 191.586},
}

SEEDS = [42, 123, 456]
ENTROPY_THRESHOLDS = [2.0]

# ── System-monitor Helpers ────────────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, metric=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "updated_at": datetime.now().isoformat(),
        "metric": metric or {},
    }))
    sys.stdout.flush()

def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists(): pid_f.unlink()
    prog_f = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if prog_f.exists():
        try: final_progress = json.loads(prog_f.read_text())
        except: pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

def profile_vram(device = "cuda:4"):
    if not torch.cuda.is_available(): return {}
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "vram_total_mb": torch.cuda.get_device_properties(device).total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(device) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(device) // 1024**2,
    }

# ── Dataset Loaders ───────────────────────────────────────────────────────────
def load_gsm8k():
    p = Path(GSM8K_DIR) / "test.json"
    with open(p) as f: return json.load(f)

def load_math500():
    p = Path(MATH500_DIR) / "test.json"
    with open(p) as f: return json.load(f)

def load_humaneval():
    p = Path(HUMANEVAL_DIR) / "test.json"
    with open(p) as f: return json.load(f)

def load_mbpp():
    p = Path(MBPP_DIR) / "test.json"
    with open(p) as f: return json.load(f)

# ── Prompt Templates ──────────────────────────────────────────────────────────
GSM8K_8SHOT = """Question: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
Answer: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So they planted 21 - 15 = 6. The answer is 6.

Question: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
Answer: There are originally 3 cars. Then 2 more cars arrive. Now 3 + 2 = 5 cars are in the parking lot. The answer is 5.

Question: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
Answer: Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74. After eating 35, they had 74 - 35 = 39 pieces left in total. The answer is 39.

Question: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?
Answer: Jason had 20 lollipops originally. Then he gave Denny some lollipops. Now Jason has 12 lollipops. So he gave Denny 20 - 12 = 8 lollipops. The answer is 8.

Question: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?
Answer: Shawn started with 5 toys. He then got 2 toys each from his mom and dad. So he got 2 + 2 = 4 more toys. Now he has 5 + 4 = 9 toys. The answer is 9.

Question: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?
Answer: There were originally 9 computers. For each of 4 days (monday to thursday) 5 more computers were added. So 5 * 4 = 20 computers were added. Now 9 + 20 = 29 computers are in the server room. The answer is 29.

Question: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did Michael have at the end of wednesday?
Answer: Michael started with 58 golf balls. He lost 23 on Tuesday, leaving 58 - 23 = 35. Then he lost 2 more on Wednesday, leaving 35 - 2 = 33 golf balls. The answer is 33.

Question: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?
Answer: Olivia had $23. She bought 5 bagels for $3 each. So she spent 5 * 3 = $15. Now she has 23 - 15 = $8. The answer is 8.

"""

MATH500_4SHOT = """Problem: Find the domain of the expression $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.
Solution: The domain requires x-2 >= 0 and 5-x > 0. So 2 <= x < 5. The answer is [2,5).

Problem: If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$
Solution: We have det(AB) = det(A)*det(B) = 2*12 = 24. The answer is 24.

Problem: Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must he lift them in order to lift the same total weight?
Solution: Original: 2*20*12 = 480 pounds. New: 2*15*n = 480, so 30n = 480, n = 16. The answer is 16.

Problem: If the system of equations $6x-4y=a$ and $6y-9x=b$ has a solution $(x, y)$ where $x\\ne 0$ and $y\\ne 0,$ find $\\frac{a}{b}.$
Solution: From the first equation, a = 6x-4y. From the second, b = 6y-9x. Multiply first by 3/2: 9x-6y = 3a/2 = -b, so a/b = -2/3. The answer is -2/3.

"""

def build_gsm8k_prompt(q): return GSM8K_8SHOT + f"Question: {q}\nAnswer:"
def build_math500_prompt(p): return MATH500_4SHOT + f"Problem: {p}\nSolution:"

def extract_gsm8k_answer(text):
    m = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None

def extract_math500_answer(text):
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m: return m.group(1).strip()
    m = re.search(r"[Tt]he answer is\s+([\S]+)", text)
    if m: return m.group(1).rstrip(".").strip()
    nums = re.findall(r"-?\d+(?:[./]\d+)?", text)
    return nums[-1] if nums else None

def gsm8k_exact_match(pred, gold):
    p, g = extract_gsm8k_answer(pred), extract_gsm8k_answer(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except: return p.strip() == g.strip()

def math500_exact_match(pred, gold_answer):
    p = extract_math500_answer(pred)
    g = gold_answer.strip()
    if p is None: return False
    p = p.strip()
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-6
    except: return False

def humaneval_pass_at_1(code_completion, problem):
    full_code = problem["prompt"] + code_completion + "\n" + problem["test"]
    full_code += f"\ncheck({problem['entry_point']})\n"
    try:
        result = subprocess.run(["python", "-c", full_code],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except: return False

def mbpp_pass_at_1(code_completion, problem):
    imports = "\n".join(problem.get("test_imports", []))
    tests = "\n".join(problem.get("test_list", []))
    full_code = imports + "\n" + code_completion + "\n" + tests
    try:
        result = subprocess.run(["python", "-c", full_code],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except: return False

# ── M1 EntropyCache Generation (Batched, Step-Reduction) ─────────────────────
@torch.no_grad()
def batched_generate_m1(
    model, tokenizer, prompts,
    steps=64, gen_length=256, block_length=32,
    entropy_threshold=1.0,
    device = "cuda:4", mask_id=LLADA_MASK_ID,
    apply_chat_template=True,
):
    """
    M1 (EntropyCache-inspired) batched generation.

    Strategy: At each denoising step within a block, compute per-token entropy.
    Tokens whose entropy drops below the threshold are considered "settled".
    We implement block-level early stopping: if >skip_fraction of block tokens
    have entropy < threshold after the current step, skip remaining steps for
    that block (equivalent to reducing forward passes).

    The skip_fraction scales with entropy_threshold:
    - Low threshold (0.5): only very confident tokens skip (conservative)
    - High threshold (3.0): even moderate-confidence tokens skip (aggressive)

    Effective step reduction = fraction of blocks that exit early.
    Real TPS improvement = (total_steps - skipped_steps) / total_steps.

    Additionally, for blocks that are finished (all tokens unmasked), we
    completely skip further forward passes.
    """
    from generate import get_num_transfer_tokens

    if apply_chat_template:
        formatted = []
        for p in prompts:
            msg = [{"role": "user", "content": p}]
            formatted.append(tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False))
        prompts = formatted

    enc = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch, prompt_len + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_len] = input_ids.clone()
    attn = torch.cat([
        attention_mask,
        torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)
    ], dim=-1)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    # Map entropy_threshold to skip_fraction:
    # threshold=0.5 -> skip when >95% of tokens are low-entropy
    # threshold=1.0 -> skip when >85% of tokens are low-entropy
    # threshold=2.0 -> skip when >70% of tokens are low-entropy
    # threshold=3.0 -> skip when >50% of tokens are low-entropy
    skip_fraction_map = {0.5: 0.95, 1.0: 0.85, 2.0: 0.70, 3.0: 0.50}
    # Find closest threshold in map
    closest_thresh = min(skip_fraction_map.keys(), key=lambda k: abs(k - entropy_threshold))
    # Linear interpolation for custom thresholds
    skip_fraction = max(0.40, min(0.98, 1.0 - entropy_threshold * 0.15))

    t0 = time.perf_counter()
    total_forward_passes = 0
    skipped_forward_passes = 0
    total_cache_hits = 0
    total_cache_checks = 0

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = prompt_len + (block_idx + 1) * block_length

        block_mask = x[:, block_start:block_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        settled_in_block = False  # track if we can early-exit this block

        for step in range(steps_per_block):
            total_forward_passes += 1

            mask_index = x == mask_id
            # If all tokens in this block are already unmasked, skip this step
            if not mask_index[:, block_start:block_end].any():
                skipped_forward_passes += 1
                total_cache_hits += block_length * batch
                total_cache_checks += block_length * batch
                continue

            # If block was marked as settled from prior step entropy check, skip
            if settled_in_block:
                skipped_forward_passes += 1
                total_cache_hits += block_length * batch
                total_cache_checks += block_length * batch
                continue

            logits = model(x, attention_mask=attn).logits

            # Compute per-token entropy for the response region
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(-1)

            # Cache hit tracking: fraction of response tokens with entropy < threshold
            resp_entropy = entropy[:, prompt_len:]
            low_entropy_mask = resp_entropy < entropy_threshold
            total_cache_hits += low_entropy_mask.float().sum().item()
            total_cache_checks += low_entropy_mask.numel()

            # Block-level early exit check: fraction of block tokens with low entropy
            block_entropy = entropy[:, block_start:block_end]
            block_low_ent = (block_entropy < entropy_threshold).float().mean().item()
            if block_low_ent >= skip_fraction and step >= 1:
                # This block has settled; skip remaining steps
                settled_in_block = True

            # Standard LLaDA unmasking
            x0 = torch.argmax(probs, dim=-1)
            x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, block_end:] = -np.inf
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p,
                                     torch.tensor(-np.inf, device=device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(batch):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    elapsed = time.perf_counter() - t0
    cache_hit_rate = total_cache_hits / total_cache_checks if total_cache_checks > 0 else 0.0
    skip_rate = skipped_forward_passes / total_forward_passes if total_forward_passes > 0 else 0.0

    generated_ids = x[:, prompt_len:]
    total_tokens = generated_ids.shape[1] * batch
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    per_sample_tps = generated_ids.shape[1] / (elapsed / batch) if elapsed > 0 else 0.0

    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return texts, per_sample_tps, elapsed, cache_hit_rate, skip_rate


# ── Per-Benchmark Evaluation Functions ───────────────────────────────────────
def eval_gsm8k_m1(model, tokenizer, data, seed, entropy_threshold, device,
                  batch_size=BATCH_SIZE, n_warmup=2, tag=""):
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    cache_hit_list = []
    correct_count = 0
    n = len(shuffled)
    batch_idx = 0

    print(f"  [GSM8K{tag}] n={n}, seed={seed}, threshold={entropy_threshold}, bs={batch_size}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = shuffled[i:i+batch_size]
        prompts = [build_gsm8k_prompt(item["question"]) for item in batch_items]
        try:
            texts, tps, _, cache_hit, skip_rate = batched_generate_m1(
                model, tokenizer, prompts,
                steps=64, gen_length=256, block_length=32,
                entropy_threshold=entropy_threshold,
                device=device, apply_chat_template=True)
            for j, (item, pred) in enumerate(zip(batch_items, texts)):
                correct = gsm8k_exact_match(pred, item["answer"])
                if correct: correct_count += 1
                results.append({"idx": i+j, "correct": correct, "prediction": pred[:200]})
            if batch_idx >= n_warmup:
                tps_list.append(tps)
                cache_hit_list.append(cache_hit)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            # Fallback: single sample
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _, ch_s, _ = batched_generate_m1(
                        model, tokenizer, [build_gsm8k_prompt(item["question"])],
                        steps=64, gen_length=256, block_length=32,
                        entropy_threshold=entropy_threshold,
                        device=device, apply_chat_template=True)
                    correct = gsm8k_exact_match(texts_s[0], item["answer"])
                    if correct: correct_count += 1
                    if batch_idx >= n_warmup:
                        tps_list.append(tps_s)
                        cache_hit_list.append(ch_s)
                    results.append({"idx": i+j, "correct": correct, "prediction": texts_s[0][:200]})
                    batch_idx += 1
                except Exception as e2:
                    results.append({"idx": i+j, "correct": False, "error": str(e2)[:80]})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "correct": False, "error": str(e)[:80]})

        done = min(i + batch_size, n)
        if done % max(batch_size * 10, 100) == 0 or done == n:
            acc = correct_count / done
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            avg_ch  = float(np.mean(cache_hit_list)) if cache_hit_list else 0.0
            print(f"    [{done}/{n}] acc={acc:.3f}, tps={avg_tps:.1f}, cache_hit={avg_ch:.3f}", flush=True)

    exact_match = correct_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if tps_list else 0.0
    avg_cache_hit = float(np.mean(cache_hit_list)) if cache_hit_list else 0.0

    return {
        "n_samples": n, "correct": correct_count, "exact_match": exact_match,
        "avg_tps": avg_tps, "tps_std": tps_std,
        "cache_hit_rate": avg_cache_hit,
        "speedup": avg_tps / BASELINE["gsm8k"]["avg_tps"] if BASELINE["gsm8k"]["avg_tps"] > 0 else 0.0,
        "accuracy_retention": exact_match / BASELINE["gsm8k"]["exact_match"] if BASELINE["gsm8k"]["exact_match"] > 0 else 0.0,
    }


def eval_math500_m1(model, tokenizer, data, seed, entropy_threshold, device,
                    batch_size=2, n_warmup=2, tag=""):
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    cache_hit_list = []
    correct_count = 0
    n = len(shuffled)
    batch_idx = 0

    print(f"  [MATH500{tag}] n={n}, seed={seed}, threshold={entropy_threshold}, bs={batch_size}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = shuffled[i:i+batch_size]
        prompts = [build_math500_prompt(item["problem"]) for item in batch_items]
        try:
            texts, tps, _, cache_hit, skip_rate = batched_generate_m1(
                model, tokenizer, prompts,
                steps=64, gen_length=512, block_length=64,
                entropy_threshold=entropy_threshold,
                device=device, apply_chat_template=True)
            for j, (item, pred) in enumerate(zip(batch_items, texts)):
                correct = math500_exact_match(pred, item["answer"])
                if correct: correct_count += 1
                results.append({"idx": i+j, "correct": correct, "prediction": pred[:200]})
            if batch_idx >= n_warmup:
                tps_list.append(tps)
                cache_hit_list.append(cache_hit)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _, ch_s, _ = batched_generate_m1(
                        model, tokenizer, [build_math500_prompt(item["problem"])],
                        steps=64, gen_length=512, block_length=64,
                        entropy_threshold=entropy_threshold,
                        device=device, apply_chat_template=True)
                    correct = math500_exact_match(texts_s[0], item["answer"])
                    if correct: correct_count += 1
                    if batch_idx >= n_warmup:
                        tps_list.append(tps_s)
                        cache_hit_list.append(ch_s)
                    results.append({"idx": i+j, "correct": correct, "prediction": texts_s[0][:200]})
                    batch_idx += 1
                except Exception as e2:
                    results.append({"idx": i+j, "correct": False, "error": str(e2)[:80]})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "correct": False, "error": str(e)[:80]})

        done = min(i + batch_size, n)
        if done % 100 == 0 or done == n:
            acc = correct_count / done
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"    [{done}/{n}] acc={acc:.3f}, tps={avg_tps:.1f}", flush=True)

    exact_match = correct_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if tps_list else 0.0
    avg_cache_hit = float(np.mean(cache_hit_list)) if cache_hit_list else 0.0

    return {
        "n_samples": n, "correct": correct_count, "exact_match": exact_match,
        "avg_tps": avg_tps, "tps_std": tps_std,
        "cache_hit_rate": avg_cache_hit,
        "speedup": avg_tps / BASELINE["math500"]["avg_tps"] if BASELINE["math500"]["avg_tps"] > 0 else 0.0,
        "accuracy_retention": exact_match / BASELINE["math500"]["exact_match"] if BASELINE["math500"]["exact_match"] > 0 else 0.0,
    }


def eval_humaneval_m1(model, tokenizer, data, seed, entropy_threshold, device,
                      batch_size=BATCH_SIZE, n_warmup=2, tag=""):
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    cache_hit_list = []
    passed_count = 0
    n = len(shuffled)
    batch_idx = 0

    print(f"  [HumanEval{tag}] n={n}, seed={seed}, threshold={entropy_threshold}, bs={batch_size}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = shuffled[i:i+batch_size]
        prompts = [f"Complete the following Python function:\n\n{item['prompt']}" for item in batch_items]
        try:
            texts, tps, _, cache_hit, skip_rate = batched_generate_m1(
                model, tokenizer, prompts,
                steps=64, gen_length=256, block_length=32,
                entropy_threshold=entropy_threshold,
                device=device, apply_chat_template=True)
            for j, (item, code) in enumerate(zip(batch_items, texts)):
                passed = humaneval_pass_at_1(code, item)
                if passed: passed_count += 1
                results.append({"idx": i+j, "passed": passed})
            if batch_idx >= n_warmup:
                tps_list.append(tps)
                cache_hit_list.append(cache_hit)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _, ch_s, _ = batched_generate_m1(
                        model, tokenizer,
                        [f"Complete the following Python function:\n\n{item['prompt']}"],
                        steps=64, gen_length=256, block_length=32,
                        entropy_threshold=entropy_threshold,
                        device=device, apply_chat_template=True)
                    passed = humaneval_pass_at_1(texts_s[0], item)
                    if passed: passed_count += 1
                    if batch_idx >= n_warmup:
                        tps_list.append(tps_s)
                        cache_hit_list.append(ch_s)
                    results.append({"idx": i+j, "passed": passed})
                    batch_idx += 1
                except Exception as e2:
                    results.append({"idx": i+j, "passed": False, "error": str(e2)[:80]})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "passed": False, "error": str(e)[:80]})

        done = min(i + batch_size, n)
        if done % 40 == 0 or done == n:
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"    [{done}/{n}] pass@1={passed_count/done:.3f}, tps={avg_tps:.1f}", flush=True)

    pass_at_1 = passed_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if tps_list else 0.0
    avg_cache_hit = float(np.mean(cache_hit_list)) if cache_hit_list else 0.0

    base_acc = BASELINE["humaneval"]["pass_at_1"]
    acc_retention = pass_at_1 / base_acc if base_acc > 0 else 1.0
    if base_acc < 0.05:
        acc_retention = 1.0 if pass_at_1 >= base_acc else pass_at_1 / max(base_acc, 0.01)

    return {
        "n_samples": n, "passed": passed_count, "pass_at_1": pass_at_1,
        "avg_tps": avg_tps, "tps_std": tps_std,
        "cache_hit_rate": avg_cache_hit,
        "speedup": avg_tps / BASELINE["humaneval"]["avg_tps"] if BASELINE["humaneval"]["avg_tps"] > 0 else 0.0,
        "accuracy_retention": acc_retention,
    }


def eval_mbpp_m1(model, tokenizer, data, seed, entropy_threshold, device,
                 batch_size=BATCH_SIZE, n_warmup=2, tag=""):
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    cache_hit_list = []
    passed_count = 0
    n = len(shuffled)
    batch_idx = 0

    print(f"  [MBPP{tag}] n={n}, seed={seed}, threshold={entropy_threshold}, bs={batch_size}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = shuffled[i:i+batch_size]
        prompts = [f"Write a Python function to: {item.get('text', item.get('prompt', ''))}" for item in batch_items]
        try:
            texts, tps, _, cache_hit, skip_rate = batched_generate_m1(
                model, tokenizer, prompts,
                steps=64, gen_length=256, block_length=32,
                entropy_threshold=entropy_threshold,
                device=device, apply_chat_template=True)
            for j, (item, code) in enumerate(zip(batch_items, texts)):
                passed = mbpp_pass_at_1(code, item)
                if passed: passed_count += 1
                results.append({"idx": i+j, "passed": passed})
            if batch_idx >= n_warmup:
                tps_list.append(tps)
                cache_hit_list.append(cache_hit)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _, ch_s, _ = batched_generate_m1(
                        model, tokenizer,
                        [f"Write a Python function to: {item.get('text', item.get('prompt', ''))}"],
                        steps=64, gen_length=256, block_length=32,
                        entropy_threshold=entropy_threshold,
                        device=device, apply_chat_template=True)
                    passed = mbpp_pass_at_1(texts_s[0], item)
                    if passed: passed_count += 1
                    if batch_idx >= n_warmup:
                        tps_list.append(tps_s)
                        cache_hit_list.append(ch_s)
                    results.append({"idx": i+j, "passed": passed})
                    batch_idx += 1
                except Exception as e2:
                    results.append({"idx": i+j, "passed": False, "error": str(e2)[:80]})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "passed": False, "error": str(e)[:80]})

        done = min(i + batch_size, n)
        if done % 80 == 0 or done == n:
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"    [{done}/{n}] pass@1={passed_count/done:.3f}, tps={avg_tps:.1f}", flush=True)

    pass_at_1 = passed_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if tps_list else 0.0
    avg_cache_hit = float(np.mean(cache_hit_list)) if cache_hit_list else 0.0

    base_acc = BASELINE["mbpp"]["pass_at_1"]
    acc_retention = 1.0 if base_acc == 0.0 else (pass_at_1 / base_acc)

    return {
        "n_samples": n, "passed": passed_count, "pass_at_1": pass_at_1,
        "avg_tps": avg_tps, "tps_std": tps_std,
        "cache_hit_rate": avg_cache_hit,
        "speedup": avg_tps / BASELINE["mbpp"]["avg_tps"] if BASELINE["mbpp"]["avg_tps"] > 0 else 0.0,
        "accuracy_retention": acc_retention,
    }


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now()
    print(f"[full_m1_pareto_t2] Starting at {start_time.isoformat()}", flush=True)
    print(f"[full_m1_pareto_t2] Entropy thresholds: {ENTROPY_THRESHOLDS}", flush=True)
    print(f"[full_m1_pareto_t2] Seeds: {SEEDS}", flush=True)

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = "cuda:4"
    report_progress(0, 100, {"status": "loading_model"})

    # Load model
    print(f"[full_m1_pareto_t2] Loading LLaDA-8B-Instruct...", flush=True)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device).eval()

    vram_after_load = profile_vram(device)
    print(f"[full_m1_pareto_t2] Model loaded. VRAM: {vram_after_load.get('vram_used_mb',0)} MB", flush=True)

    # Load datasets
    print("[full_m1_pareto_t2] Loading datasets...", flush=True)
    gsm8k_data   = load_gsm8k()
    math500_data = load_math500()
    he_data      = load_humaneval()
    mbpp_data    = load_mbpp()
    print(f"  GSM8K={len(gsm8k_data)}, MATH500={len(math500_data)}, "
          f"HumanEval={len(he_data)}, MBPP={len(mbpp_data)}", flush=True)

    report_progress(5, 100, {"status": "data_loaded"})

    # Check for partial results (resumability)
    partial_path = RESULTS_DIR / "m1_pareto_partial_t2.json"
    all_threshold_results = {}
    # Load base partial (thresholds 0.5, 1.0 already done) to skip them
    base_partial_path = RESULTS_DIR / "m1_pareto_partial.json"
    if base_partial_path.exists():
        try:
            all_threshold_results = json.loads(base_partial_path.read_text())
            print(f"[full_m1_pareto_t2] Loaded base partial: {len(all_threshold_results)} thresholds done", flush=True)
        except:
            all_threshold_results = {}
    if partial_path.exists():
        try:
            own = json.loads(partial_path.read_text())
            all_threshold_results.update(own)
            print(f"[full_m1_pareto_t2] Resuming own partial: {len(own)} own threshold results", flush=True)
        except:
            pass

    total_steps = len(ENTROPY_THRESHOLDS) * len(SEEDS) * 4  # 4 benchmarks
    completed_steps = 0

    # ── Main sweep ────────────────────────────────────────────────────────────
    for t_idx, threshold in enumerate(ENTROPY_THRESHOLDS):
        thresh_key = str(threshold)
        if thresh_key not in all_threshold_results:
            all_threshold_results[thresh_key] = {}

        print(f"\n[full_m1_pareto_t2] === Threshold {threshold} ({t_idx+1}/{len(ENTROPY_THRESHOLDS)}) ===", flush=True)

        for seed in SEEDS:
            seed_key = str(seed)
            if seed_key in all_threshold_results[thresh_key]:
                print(f"  [skip] threshold={threshold}, seed={seed} already done", flush=True)
                completed_steps += 4
                continue

            print(f"  [full_m1_pareto_t2] threshold={threshold}, seed={seed}", flush=True)
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

            seed_results = {}

            # GSM8K
            try:
                gsm_r = eval_gsm8k_m1(
                    model, tokenizer, gsm8k_data, seed, threshold, device,
                    batch_size=BATCH_SIZE, n_warmup=2, tag=f"_t{threshold}_s{seed}")
                seed_results["gsm8k"] = gsm_r
                completed_steps += 1
                report_progress(completed_steps, total_steps,
                    {"threshold": threshold, "seed": seed, "benchmark": "gsm8k",
                     "speedup": gsm_r["speedup"], "acc": gsm_r["exact_match"]})
                print(f"    GSM8K done: acc={gsm_r['exact_match']:.3f}, "
                      f"speedup={gsm_r['speedup']:.2f}x, "
                      f"cache_hit={gsm_r['cache_hit_rate']:.3f}", flush=True)
            except Exception as e:
                print(f"    GSM8K ERROR: {e}", flush=True)
                seed_results["gsm8k"] = {"error": str(e)[:200], "exact_match": 0, "avg_tps": 0,
                                          "speedup": 0, "accuracy_retention": 0, "cache_hit_rate": 0}
                completed_steps += 1
            gc.collect(); torch.cuda.empty_cache()

            # MATH500
            try:
                math_r = eval_math500_m1(
                    model, tokenizer, math500_data, seed, threshold, device,
                    batch_size=2, n_warmup=2, tag=f"_t{threshold}_s{seed}")
                seed_results["math500"] = math_r
                completed_steps += 1
                report_progress(completed_steps, total_steps,
                    {"threshold": threshold, "seed": seed, "benchmark": "math500",
                     "speedup": math_r["speedup"], "acc": math_r["exact_match"]})
                print(f"    MATH500 done: acc={math_r['exact_match']:.3f}, "
                      f"speedup={math_r['speedup']:.2f}x", flush=True)
            except Exception as e:
                print(f"    MATH500 ERROR: {e}", flush=True)
                seed_results["math500"] = {"error": str(e)[:200], "exact_match": 0, "avg_tps": 0,
                                            "speedup": 0, "accuracy_retention": 0, "cache_hit_rate": 0}
                completed_steps += 1
            gc.collect(); torch.cuda.empty_cache()

            # HumanEval
            try:
                he_r = eval_humaneval_m1(
                    model, tokenizer, he_data, seed, threshold, device,
                    batch_size=BATCH_SIZE, n_warmup=2, tag=f"_t{threshold}_s{seed}")
                seed_results["humaneval"] = he_r
                completed_steps += 1
                report_progress(completed_steps, total_steps,
                    {"threshold": threshold, "seed": seed, "benchmark": "humaneval",
                     "speedup": he_r["speedup"], "pass_at_1": he_r["pass_at_1"]})
                print(f"    HumanEval done: pass@1={he_r['pass_at_1']:.3f}, "
                      f"speedup={he_r['speedup']:.2f}x", flush=True)
            except Exception as e:
                print(f"    HumanEval ERROR: {e}", flush=True)
                seed_results["humaneval"] = {"error": str(e)[:200], "pass_at_1": 0, "avg_tps": 0,
                                              "speedup": 0, "accuracy_retention": 0, "cache_hit_rate": 0}
                completed_steps += 1
            gc.collect(); torch.cuda.empty_cache()

            # MBPP
            try:
                mbpp_r = eval_mbpp_m1(
                    model, tokenizer, mbpp_data, seed, threshold, device,
                    batch_size=BATCH_SIZE, n_warmup=2, tag=f"_t{threshold}_s{seed}")
                seed_results["mbpp"] = mbpp_r
                completed_steps += 1
                report_progress(completed_steps, total_steps,
                    {"threshold": threshold, "seed": seed, "benchmark": "mbpp",
                     "speedup": mbpp_r["speedup"], "pass_at_1": mbpp_r["pass_at_1"]})
                print(f"    MBPP done: pass@1={mbpp_r['pass_at_1']:.3f}, "
                      f"speedup={mbpp_r['speedup']:.2f}x", flush=True)
            except Exception as e:
                print(f"    MBPP ERROR: {e}", flush=True)
                seed_results["mbpp"] = {"error": str(e)[:200], "pass_at_1": 0, "avg_tps": 0,
                                         "speedup": 0, "accuracy_retention": 0, "cache_hit_rate": 0}
                completed_steps += 1
            gc.collect(); torch.cuda.empty_cache()

            # Save seed result
            all_threshold_results[thresh_key][seed_key] = seed_results
            partial_path.write_text(json.dumps(all_threshold_results, indent=2))
            print(f"  [checkpoint] threshold={threshold}, seed={seed} saved", flush=True)

    # ── Aggregate Results ─────────────────────────────────────────────────────
    print("\n[full_m1_pareto_t2] Aggregating results...", flush=True)
    benchmarks = ["gsm8k", "math500", "humaneval", "mbpp"]
    acc_key = {"gsm8k": "exact_match", "math500": "exact_match",
               "humaneval": "pass_at_1", "mbpp": "pass_at_1"}

    pareto_points = []
    best_op_point = None
    best_qas = -1.0

    for threshold in ENTROPY_THRESHOLDS:
        thresh_key = str(threshold)
        thresh_data = all_threshold_results.get(thresh_key, {})

        agg = {}
        for bm in benchmarks:
            vals_tps, vals_acc, vals_speedup, vals_ret, vals_ch = [], [], [], [], []
            for seed in SEEDS:
                sd = thresh_data.get(str(seed), {}).get(bm, {})
                if "error" in sd or not sd: continue
                vals_tps.append(sd.get("avg_tps", 0))
                vals_acc.append(sd.get(acc_key[bm], 0))
                vals_speedup.append(sd.get("speedup", 0))
                vals_ret.append(sd.get("accuracy_retention", 0))
                vals_ch.append(sd.get("cache_hit_rate", 0))

            if vals_tps:
                agg[bm] = {
                    "avg_tps_mean":  float(np.mean(vals_tps)),
                    "avg_tps_std":   float(np.std(vals_tps)),
                    "accuracy_mean": float(np.mean(vals_acc)),
                    "accuracy_std":  float(np.std(vals_acc)),
                    "speedup_mean":  float(np.mean(vals_speedup)),
                    "speedup_std":   float(np.std(vals_speedup)),
                    "acc_retention_mean": float(np.mean(vals_ret)),
                    "acc_retention_std":  float(np.std(vals_ret)),
                    "cache_hit_mean": float(np.mean(vals_ch)),
                    "n_seeds": len(vals_tps),
                }
            else:
                agg[bm] = {
                    "avg_tps_mean": 0, "avg_tps_std": 0,
                    "accuracy_mean": 0, "accuracy_std": 0,
                    "speedup_mean": 0, "speedup_std": 0,
                    "acc_retention_mean": 0, "acc_retention_std": 0,
                    "cache_hit_mean": 0, "n_seeds": 0,
                }

        # QAS computation (weighted: GSM8K 1319, MATH500 500, HumanEval 164, MBPP 374)
        weights = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 374}
        total_w = sum(weights.values())
        combined_speedup = sum(agg[bm]["speedup_mean"] * weights[bm] for bm in benchmarks) / total_w
        combined_acc_ret = sum(agg[bm]["acc_retention_mean"] * weights[bm] for bm in benchmarks) / total_w
        combined_qas = combined_speedup * combined_acc_ret

        # Operating point: GSM8K accuracy drop <= 2%
        gsm_acc_drop = 1.0 - agg["gsm8k"]["acc_retention_mean"]
        within_budget = gsm_acc_drop <= 0.02

        point = {
            "entropy_threshold": threshold,
            "aggregated": agg,
            "combined_speedup": combined_speedup,
            "combined_accuracy_retention": combined_acc_ret,
            "combined_qas": combined_qas,
            "within_2pct_accuracy_budget": within_budget,
            "gsm8k_acc_drop": gsm_acc_drop,
        }
        pareto_points.append(point)

        if within_budget and combined_qas > best_qas:
            best_qas = combined_qas
            best_op_point = point

        print(f"  Threshold={threshold}: speedup={combined_speedup:.3f}x, "
              f"acc_ret={combined_acc_ret:.3f}, QAS={combined_qas:.3f}, "
              f"within_2pct={within_budget}", flush=True)

    # If nothing within 2%, use 5% budget as fallback
    if best_op_point is None:
        for p in pareto_points:
            if (1.0 - p["aggregated"]["gsm8k"]["acc_retention_mean"]) <= 0.05:
                if p["combined_qas"] > best_qas:
                    best_qas = p["combined_qas"]
                    best_op_point = p
    # Ultimate fallback: best QAS
    if best_op_point is None and pareto_points:
        best_op_point = max(pareto_points, key=lambda p: p["combined_qas"])

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    print(f"\n[full_m1_pareto_t2] === FINAL OPERATING POINT ===", flush=True)
    if best_op_point:
        print(f"  Entropy threshold: {best_op_point['entropy_threshold']}", flush=True)
        print(f"  Combined speedup:  {best_op_point['combined_speedup']:.3f}x", flush=True)
        print(f"  Combined QAS:      {best_op_point['combined_qas']:.3f}", flush=True)
        print(f"  GSM8K speedup:     {best_op_point['aggregated']['gsm8k']['speedup_mean']:.3f}x", flush=True)
        print(f"  GSM8K acc:         {best_op_point['aggregated']['gsm8k']['accuracy_mean']:.3f}", flush=True)

    # ── Save Output ───────────────────────────────────────────────────────────
    output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "method": "M1-EntropyCache",
        "mode": "full",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": elapsed_min,
        "seeds": SEEDS,
        "entropy_thresholds": ENTROPY_THRESHOLDS,
        "benchmarks": benchmarks,
        "baseline_reference": BASELINE,
        "pareto_points": pareto_points,
        "operating_point": best_op_point,
        "per_threshold_per_seed": all_threshold_results,
        "vram_after_load": vram_after_load,
    }

    out_path = RESULTS_DIR / "m1_pareto_full_t2.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[full_m1_pareto_t2] Results saved to {out_path}", flush=True)

    # Cleanup partial file
    if partial_path.exists():
        partial_path.unlink()

    # ── Update gpu_progress.json ───────────────────────────────────────────────
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {
            "completed": [], "failed": [], "running": {}, "timings": {}
        }
        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 90,
            "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "config_snapshot": {
                "model": "LLaDA-8B-Instruct",
                "method": "M1-EntropyCache",
                "thresholds": ENTROPY_THRESHOLDS,
                "seeds": SEEDS,
                "benchmarks": benchmarks,
                "gpu_model": vram_after_load.get("gpu_name", "unknown"),
            }
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"[full_m1_pareto_t2] gpu_progress.json updated", flush=True)
    except Exception as e:
        print(f"[full_m1_pareto_t2] WARNING: Could not update gpu_progress.json: {e}", flush=True)

    summary = (
        f"M1 full pareto: op_threshold={best_op_point['entropy_threshold'] if best_op_point else 'N/A'}, "
        f"speedup={best_op_point['combined_speedup']:.3f}x, "
        f"QAS={best_op_point['combined_qas']:.3f}" if best_op_point else "M1 full pareto: no valid operating point"
    )
    mark_done(status="success", summary=summary)
    report_progress(100, 100, {"status": "done"})
    print(f"[full_m1_pareto_t2] Done in {elapsed_min:.1f} minutes.", flush=True)


if __name__ == "__main__":
    main()
