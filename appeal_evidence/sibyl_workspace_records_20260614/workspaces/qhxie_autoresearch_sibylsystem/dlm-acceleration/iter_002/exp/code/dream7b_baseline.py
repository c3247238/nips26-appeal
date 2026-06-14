"""
Full Baseline Evaluation: LLaDA-8B-Instruct + Dream-7B-Instruct.

Task: full_baseline
- GSM8K (1319 examples), MATH500 (500 examples), HumanEval (164 examples), MBPP (374 examples)
- Seeds: [42, 123, 456], report mean ± std
- Report: TPS, exact_match (GSM8K/MATH500), pass@1 (HumanEval/MBPP)
- Also run Dream-7B-Instruct baseline for cross-model validation

Key optimization: batch inference to fully utilize 97GB VRAM GPU.
With model using ~17GB, batch_size=4 uses ~36GB (well within limits).

Output:
    exp/results/full_baseline/llada_baseline_full.json
    exp/results/full_baseline/dream_baseline_full.json

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dlm-acceleration python full_baseline.py
"""
import os, sys, json, time, random, gc, re, subprocess
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED    = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH       = str(SHARED / "checkpoints" / "llada-8b-instruct")
DREAM_MODEL_PATH = str(SHARED / "checkpoints" / "dream-7b-instruct")
GSM8K_DIR        = str(SHARED / "datasets" / "gsm8k")
HUMANEVAL_DIR    = str(SHARED / "datasets" / "humaneval")
MATH500_DIR      = str(SHARED / "datasets" / "math500")
MBPP_DIR         = str(SHARED / "datasets" / "mbpp")
CODE_DIR         = WORKSPACE / "exp" / "code"
RESULTS_DIR      = WORKSPACE / "exp" / "results" / "full_baseline"
TASK_ID          = "full_baseline"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LLADA_MASK_ID = 126336

# Batch size for inference - use 4 for ~70GB VRAM usage on 97GB GPU
BATCH_SIZE = 4

# ── System-monitor Helpers ────────────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, metric=None):
    prog = {
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "updated_at": datetime.now().isoformat(),
        "metric": metric or {},
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(prog))
    # Force flush
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

# ── VRAM Profiler ─────────────────────────────────────────────────────────────
def profile_vram(device="cuda:0"):
    if not torch.cuda.is_available(): return {}
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "vram_total_mb": torch.cuda.get_device_properties(device).total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(device) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(device) // 1024**2,
    }

# ── Dataset Preparation ───────────────────────────────────────────────────────
def prepare_gsm8k():
    path = Path(GSM8K_DIR) / "test.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"[data] GSM8K loaded from local: {len(data)} examples", flush=True)
        return data
    else:
        print("[data] GSM8K not found locally, loading from HuggingFace...", flush=True)
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        data = [{"question": x["question"], "answer": x["answer"]} for x in ds]
        Path(GSM8K_DIR).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[data] GSM8K saved: {len(data)} examples", flush=True)
        return data

def prepare_math500():
    path = Path(MATH500_DIR) / "test.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"[data] MATH500 loaded from local: {len(data)} examples", flush=True)
        return data
    else:
        print("[data] MATH500 not found locally, loading from HuggingFace...", flush=True)
        from datasets import load_dataset
        Path(MATH500_DIR).mkdir(parents=True, exist_ok=True)
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        data = [{"problem": x["problem"], "solution": x["solution"],
                 "answer": x["answer"], "subject": x["subject"],
                 "level": x["level"]} for x in ds]
        with open(path, "w") as f:
            json.dump(data, f)
        _update_registry("datasets", "math500", {
            "path": "shared/datasets/math500",
            "abs_path": str(SHARED / "datasets" / "math500"),
            "splits": ["test"], "test_size": len(data)
        })
        print(f"[data] MATH500 saved: {len(data)} examples", flush=True)
        return data

def prepare_humaneval():
    path = Path(HUMANEVAL_DIR) / "test.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"[data] HumanEval loaded from local: {len(data)} examples", flush=True)
        return data
    else:
        print("[data] HumanEval not found locally, loading from HuggingFace...", flush=True)
        from datasets import load_dataset
        ds = load_dataset("openai_humaneval", split="test")
        data = [dict(x) for x in ds]
        Path(HUMANEVAL_DIR).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[data] HumanEval saved: {len(data)} examples", flush=True)
        return data

def prepare_mbpp():
    path = Path(MBPP_DIR) / "test.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"[data] MBPP loaded from local: {len(data)} examples", flush=True)
        return data
    else:
        print("[data] MBPP not found locally, loading from HuggingFace...", flush=True)
        from datasets import load_dataset
        Path(MBPP_DIR).mkdir(parents=True, exist_ok=True)
        ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        data = [dict(x) for x in ds]
        with open(path, "w") as f:
            json.dump(data, f)
        _update_registry("datasets", "mbpp", {
            "path": "shared/datasets/mbpp",
            "abs_path": str(SHARED / "datasets" / "mbpp"),
            "splits": ["test"], "test_size": len(data)
        })
        print(f"[data] MBPP saved: {len(data)} examples", flush=True)
        return data

def _update_registry(section, key, value):
    registry_path = SHARED / "registry.json"
    if registry_path.exists():
        try: registry = json.loads(registry_path.read_text())
        except: registry = {"checkpoints": {}, "datasets": {}}
    else:
        registry = {"checkpoints": {}, "datasets": {}}
    if section not in registry: registry[section] = {}
    registry[section][key] = value
    registry_path.write_text(json.dumps(registry, indent=2))

def prepare_dream_7b():
    dream_path = Path(DREAM_MODEL_PATH)
    if dream_path.exists() and any(dream_path.iterdir()):
        print(f"[model] Dream-7B-Instruct found at {DREAM_MODEL_PATH}", flush=True)
        return True
    print("[model] Dream-7B-Instruct not found, downloading from HuggingFace...", flush=True)
    try:
        from huggingface_hub import snapshot_download
        dream_path.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="hkunlp/dream-7b-instruct",
            local_dir=str(dream_path),
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"]
        )
        _update_registry("checkpoints", "dream-7b-instruct", {
            "path": "shared/checkpoints/dream-7b-instruct",
            "abs_path": str(dream_path),
            "hf_repo": "hkunlp/dream-7b-instruct",
            "downloaded_at": datetime.now().strftime("%Y-%m-%d"),
        })
        print("[model] Dream-7B-Instruct downloaded successfully", flush=True)
        return True
    except Exception as e:
        print(f"[model] Dream-7B-Instruct download failed: {e}", flush=True)
        return False

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

# ── Batched LLaDA Generate ────────────────────────────────────────────────────
def batched_generate(model, tokenizer, prompts, steps=64, gen_length=256,
                     block_length=32, device="cuda:0", mask_id=LLADA_MASK_ID,
                     apply_chat_template=True):
    """
    Generate completions for a batch of prompts.
    Returns: list of (text, tps) tuples.
    """
    from generate import generate as llada_generate

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

    t0 = time.perf_counter()
    with torch.no_grad():
        out = llada_generate(
            model, input_ids, attention_mask=attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
            mask_id=mask_id,
        )
    elapsed = time.perf_counter() - t0

    prompt_len = input_ids.shape[1]
    generated_ids = out[:, prompt_len:]
    # Per-sample TPS
    per_sample_tokens = generated_ids.shape[1]
    batch_tps = (per_sample_tokens * len(prompts)) / elapsed if elapsed > 0 else 0.0

    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    per_sample_tps = per_sample_tokens / (elapsed / len(prompts)) if elapsed > 0 else 0.0
    return texts, per_sample_tps, elapsed


def probe_batch_size(model, tokenizer, device="cuda:0", gen_length=256,
                     block_length=32, start=8, min_bs=1):
    """Find max safe batch size via binary search."""
    from generate import generate as llada_generate
    print(f"[probe] Probing max batch size (start={start})...", flush=True)

    sample_prompt = "Question: What is 2+2?\nAnswer:"
    enc = tokenizer([sample_prompt], add_special_tokens=False, return_tensors="pt")
    sample_len = enc["input_ids"].shape[1]

    high, best = start, min_bs
    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache(); gc.collect()
            # Create dummy batch
            dummy_input = torch.zeros((mid, sample_len + gen_length), dtype=torch.long).to(device)
            dummy_mask = torch.ones((mid, sample_len + gen_length), dtype=torch.long).to(device)
            dummy_input[:, :sample_len] = enc["input_ids"].expand(mid, -1).to(device)
            with torch.no_grad():
                _ = model(dummy_input, attention_mask=dummy_mask)
            best = mid
            min_bs = mid + 1
            del dummy_input, dummy_mask
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache(); gc.collect()
        except Exception:
            high = mid - 1

    print(f"[probe] Max batch size: {best}", flush=True)
    return max(1, best)


# ── Core Evaluation Functions ────────────────────────────────────────────────
def eval_gsm8k_batched(model, tokenizer, data, seed=42, device="cuda:0",
                       batch_size=BATCH_SIZE, n_warmup=2, tag=""):
    """Batched GSM8K evaluation."""
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    correct_count = 0
    n = len(shuffled)
    batch_idx = 0

    print(f"\n[eval{tag}] GSM8K: {n} examples, seed={seed}, batch_size={batch_size}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = shuffled[i:i+batch_size]
        prompts = [build_gsm8k_prompt(item["question"]) for item in batch_items]
        try:
            texts, tps, _ = batched_generate(
                model, tokenizer, prompts,
                steps=64, gen_length=256, block_length=32,
                device=device, mask_id=LLADA_MASK_ID,
                apply_chat_template=True
            )
            for j, (item, pred) in enumerate(zip(batch_items, texts)):
                correct = gsm8k_exact_match(pred, item["answer"])
                if correct: correct_count += 1
                results.append({"idx": i+j, "correct": correct,
                                "prediction": pred[:200]})

            if batch_idx >= n_warmup:
                tps_list.append(tps)
            batch_idx += 1

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [OOM] batch {i//batch_size}, reducing to individual...", flush=True)
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _ = batched_generate(
                        model, tokenizer, [build_gsm8k_prompt(item["question"])],
                        steps=64, gen_length=256, block_length=32,
                        device=device, apply_chat_template=True)
                    correct = gsm8k_exact_match(texts_s[0], item["answer"])
                    if correct: correct_count += 1
                    if batch_idx >= n_warmup: tps_list.append(tps_s)
                    results.append({"idx": i+j, "correct": correct})
                    batch_idx += 1
                except Exception as e:
                    results.append({"idx": i+j, "error": str(e)[:80], "correct": False})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "error": str(e)[:80], "correct": False})

        done = min(i + batch_size, n)
        if done % max(batch_size*5, 100) == 0 or done == n:
            acc = correct_count / done
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"  [{done}/{n}] acc={acc:.3f}, avg_tps={avg_tps:.1f}", flush=True)

    exact_match = correct_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if tps_list else 0.0
    return {
        "n_samples": n, "correct": correct_count,
        "exact_match": exact_match, "avg_tps": avg_tps,
        "tps_std": tps_std, "n_warmup": n_warmup,
    }, results


def eval_math500_batched(model, tokenizer, data, seed=42, device="cuda:0",
                         batch_size=BATCH_SIZE, n_warmup=2, tag=""):
    """Batched MATH500 evaluation."""
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    correct_count = 0
    n = len(shuffled)
    batch_idx = 0

    # MATH500 needs longer gen_length - use smaller batch
    math_batch = max(1, batch_size // 2)
    print(f"\n[eval{tag}] MATH500: {n} examples, seed={seed}, batch_size={math_batch}", flush=True)

    for i in range(0, n, math_batch):
        batch_items = shuffled[i:i+math_batch]
        prompts = [build_math500_prompt(item["problem"]) for item in batch_items]
        try:
            texts, tps, _ = batched_generate(
                model, tokenizer, prompts,
                steps=64, gen_length=512, block_length=64,
                device=device, apply_chat_template=True
            )
            for j, (item, pred) in enumerate(zip(batch_items, texts)):
                correct = math500_exact_match(pred, item["answer"])
                if correct: correct_count += 1
                results.append({"idx": i+j, "correct": correct,
                                "gold_answer": item["answer"], "prediction": pred[:200]})
            if batch_idx >= n_warmup: tps_list.append(tps)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _ = batched_generate(
                        model, tokenizer, [build_math500_prompt(item["problem"])],
                        steps=64, gen_length=512, block_length=64,
                        device=device, apply_chat_template=True)
                    correct = math500_exact_match(texts_s[0], item["answer"])
                    if correct: correct_count += 1
                    if batch_idx >= n_warmup: tps_list.append(tps_s)
                    results.append({"idx": i+j, "correct": correct})
                    batch_idx += 1
                except Exception as e:
                    results.append({"idx": i+j, "error": str(e)[:80], "correct": False})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "error": str(e)[:80], "correct": False})

        done = min(i + math_batch, n)
        if done % 100 == 0 or done == n:
            acc = correct_count / done
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"  [{done}/{n}] acc={acc:.3f}, avg_tps={avg_tps:.1f}", flush=True)

    exact_match = correct_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return {
        "n_samples": n, "correct": correct_count,
        "exact_match": exact_match, "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
    }, results


def eval_humaneval_batched(model, tokenizer, data, seed=42, device="cuda:0",
                           batch_size=BATCH_SIZE, n_warmup=1, tag=""):
    """Batched HumanEval evaluation."""
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    passed_count = 0
    n = len(shuffled)
    batch_idx = 0

    print(f"\n[eval{tag}] HumanEval: {n} examples, seed={seed}, batch_size={batch_size}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = shuffled[i:i+batch_size]
        prompts = [p["prompt"] for p in batch_items]
        try:
            texts, tps, _ = batched_generate(
                model, tokenizer, prompts,
                steps=64, gen_length=256, block_length=32,
                device=device, apply_chat_template=False
            )
            for j, (problem, code) in enumerate(zip(batch_items, texts)):
                passed = humaneval_pass_at_1(code, problem)
                if passed: passed_count += 1
                results.append({
                    "task_id": problem.get("task_id", str(i+j)),
                    "passed": passed, "completion": code[:300]
                })
            if batch_idx >= n_warmup: tps_list.append(tps)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            for j, problem in enumerate(batch_items):
                try:
                    texts_s, tps_s, _ = batched_generate(
                        model, tokenizer, [problem["prompt"]],
                        steps=64, gen_length=256, block_length=32,
                        device=device, apply_chat_template=False)
                    passed = humaneval_pass_at_1(texts_s[0], problem)
                    if passed: passed_count += 1
                    if batch_idx >= n_warmup: tps_list.append(tps_s)
                    results.append({"task_id": problem.get("task_id", str(i+j)), "passed": passed})
                    batch_idx += 1
                except Exception as e:
                    results.append({"task_id": problem.get("task_id", str(i+j)), "error": str(e)[:80], "passed": False})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"task_id": batch_items[j].get("task_id", str(i+j)), "error": str(e)[:80], "passed": False})

        done = min(i + batch_size, n)
        if done % 50 == 0 or done == n:
            p1 = passed_count / done
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"  [{done}/{n}] pass@1={p1:.3f}, avg_tps={avg_tps:.1f}", flush=True)

    pass_at_1 = passed_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return {
        "n_samples": n, "passed": passed_count,
        "pass_at_1": pass_at_1, "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
    }, results


def eval_mbpp_batched(model, tokenizer, data, seed=42, device="cuda:0",
                      batch_size=BATCH_SIZE, n_warmup=1, tag=""):
    """Batched MBPP evaluation."""
    rng = random.Random(seed)
    shuffled = data[:]
    rng.shuffle(shuffled)

    results = []
    tps_list = []
    passed_count = 0
    n = len(shuffled)
    batch_idx = 0

    print(f"\n[eval{tag}] MBPP: {n} examples, seed={seed}, batch_size={batch_size}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = shuffled[i:i+batch_size]
        prompts = [f"Write a Python function to: {p['prompt']}\n\n# Your solution:\n"
                   for p in batch_items]
        try:
            texts, tps, _ = batched_generate(
                model, tokenizer, prompts,
                steps=64, gen_length=256, block_length=32,
                device=device, apply_chat_template=False
            )
            for j, (problem, code) in enumerate(zip(batch_items, texts)):
                passed = mbpp_pass_at_1(code, problem)
                if passed: passed_count += 1
                results.append({
                    "task_id": problem.get("task_id", str(i+j)),
                    "passed": passed, "completion": code[:300]
                })
            if batch_idx >= n_warmup: tps_list.append(tps)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            for j, problem in enumerate(batch_items):
                try:
                    prompt_s = f"Write a Python function to: {problem['prompt']}\n\n# Your solution:\n"
                    texts_s, tps_s, _ = batched_generate(
                        model, tokenizer, [prompt_s],
                        steps=64, gen_length=256, block_length=32,
                        device=device, apply_chat_template=False)
                    passed = mbpp_pass_at_1(texts_s[0], problem)
                    if passed: passed_count += 1
                    if batch_idx >= n_warmup: tps_list.append(tps_s)
                    results.append({"task_id": problem.get("task_id", str(i+j)), "passed": passed})
                    batch_idx += 1
                except Exception as e:
                    results.append({"task_id": problem.get("task_id", str(i+j)), "error": str(e)[:80], "passed": False})
        except Exception as e:
            for j in range(len(batch_items)):
                results.append({"task_id": batch_items[j].get("task_id", str(i+j)), "error": str(e)[:80], "passed": False})

        done = min(i + batch_size, n)
        if done % 100 == 0 or done == n:
            p1 = passed_count / done
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"  [{done}/{n}] pass@1={p1:.3f}, avg_tps={avg_tps:.1f}", flush=True)

    pass_at_1 = passed_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return {
        "n_samples": n, "passed": passed_count,
        "pass_at_1": pass_at_1, "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
    }, results


def aggregate_seeds(results_by_seed, bench, metric):
    values = [v[bench].get(metric) for v in results_by_seed.values()
              if bench in v and v[bench].get(metric) is not None]
    if not values:
        return {"mean": None, "std": None, "n_seeds": 0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values)),
            "n_seeds": len(values), "per_seed": values}


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now()
    print(f"[full_baseline] Starting at {start_time.isoformat()}", flush=True)
    print(f"[full_baseline] Task: Full baseline evaluation (3 seeds, 4 benchmarks)", flush=True)
    print(f"[full_baseline] Batch size: {BATCH_SIZE}", flush=True)
    if torch.cuda.is_available():
        print(f"[full_baseline] GPU: {torch.cuda.get_device_name(0)}", flush=True)

    seeds = [42, 123, 456]

    # ── Step 1: Prepare datasets ────────────────────────────────────────────────
    print("\n[full_baseline] === Preparing Datasets ===", flush=True)
    report_progress(0, 10, {"phase": "data_prep"})
    gsm8k_data   = prepare_gsm8k()
    math500_data = prepare_math500()
    he_data      = prepare_humaneval()
    mbpp_data    = prepare_mbpp()
    report_progress(1, 10, {"phase": "data_ready",
                            "gsm8k": len(gsm8k_data), "math500": len(math500_data),
                            "humaneval": len(he_data), "mbpp": len(mbpp_data)})
    print(f"\nDataset sizes: GSM8K={len(gsm8k_data)}, MATH500={len(math500_data)}, "
          f"HumanEval={len(he_data)}, MBPP={len(mbpp_data)}", flush=True)

    # ── Step 2: Load LLaDA ─────────────────────────────────────────────────────
    print("\n[full_baseline] === Loading LLaDA-8B-Instruct ===", flush=True)
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to("cuda:0").eval()
    vram_after_load = profile_vram()
    print(f"[full_baseline] Model loaded. VRAM: {vram_after_load}", flush=True)

    # Probe batch size
    actual_batch = probe_batch_size(model, tokenizer, start=8)
    actual_batch = max(1, actual_batch)
    print(f"[full_baseline] Using batch_size={actual_batch}", flush=True)

    # Save GPU profile
    gpu_profile = {
        **vram_after_load,
        "max_batch_size": actual_batch,
        "vram_used_mb": vram_after_load.get("vram_used_mb", 0),
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2))

    # ── Step 3: LLaDA evaluation across seeds ─────────────────────────────────
    print("\n[full_baseline] === LLaDA-8B-Instruct Baseline ===", flush=True)
    report_progress(2, 10, {"phase": "llada_eval"})

    results_by_seed = {}
    for seed_idx, seed in enumerate(seeds):
        print(f"\n{'='*50}", flush=True)
        print(f"  Seed {seed} ({seed_idx+1}/{len(seeds)})", flush=True)
        print(f"{'='*50}", flush=True)
        torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        tag = f"[seed={seed}]"
        seed_results = {}

        gsm_m, _ = eval_gsm8k_batched(model, tokenizer, gsm8k_data, seed=seed,
                                       batch_size=actual_batch, tag=tag)
        seed_results["gsm8k"] = gsm_m
        report_progress(2 + seed_idx, 10,
                        {"seed": seed, "gsm8k_acc": gsm_m["exact_match"],
                         "gsm8k_tps": gsm_m["avg_tps"]})
        print(f"  GSM8K done: acc={gsm_m['exact_match']:.3f}, tps={gsm_m['avg_tps']:.1f}", flush=True)

        math_m, _ = eval_math500_batched(model, tokenizer, math500_data, seed=seed,
                                         batch_size=actual_batch, tag=tag)
        seed_results["math500"] = math_m
        print(f"  MATH500 done: acc={math_m['exact_match']:.3f}, tps={math_m['avg_tps']:.1f}", flush=True)

        he_m, _ = eval_humaneval_batched(model, tokenizer, he_data, seed=seed,
                                         batch_size=actual_batch, tag=tag)
        seed_results["humaneval"] = he_m
        print(f"  HumanEval done: pass@1={he_m['pass_at_1']:.3f}, tps={he_m['avg_tps']:.1f}", flush=True)

        mbpp_m, _ = eval_mbpp_batched(model, tokenizer, mbpp_data, seed=seed,
                                      batch_size=actual_batch, tag=tag)
        seed_results["mbpp"] = mbpp_m
        print(f"  MBPP done: pass@1={mbpp_m['pass_at_1']:.3f}, tps={mbpp_m['avg_tps']:.1f}", flush=True)

        results_by_seed[str(seed)] = seed_results

        # Save intermediate seed result
        (RESULTS_DIR / f"llada_seed_{seed}.json").write_text(
            json.dumps({"seed": seed, **seed_results}, indent=2)
        )

    # Aggregate
    aggregated = {}
    for bench in ["gsm8k", "math500", "humaneval", "mbpp"]:
        main_m = "exact_match" if bench in ["gsm8k", "math500"] else "pass_at_1"
        aggregated[bench] = {
            main_m: aggregate_seeds(results_by_seed, bench, main_m),
            "avg_tps": aggregate_seeds(results_by_seed, bench, "avg_tps"),
        }

    vram_final = profile_vram()
    llada_metrics = {
        "task_id": TASK_ID, "model": "LLaDA-8B-Instruct", "model_path": MODEL_PATH,
        "mode": "baseline", "steps": 64, "seeds": seeds,
        "batch_size": actual_batch,
        "benchmarks": {b: {"n_total": len(d)} for b, d in
                       zip(["gsm8k","math500","humaneval","mbpp"],
                           [gsm8k_data, math500_data, he_data, mbpp_data])},
        "aggregated": aggregated,
        "per_seed": results_by_seed,
        "vram": {"after_load": vram_after_load, "final": vram_final},
        "timestamp": datetime.now().isoformat(),
    }
    llada_out = RESULTS_DIR / "llada_baseline_full.json"
    llada_out.write_text(json.dumps(llada_metrics, indent=2))
    print(f"\n[full_baseline] LLaDA results saved to {llada_out}", flush=True)

    agg = llada_metrics["aggregated"]
    print("\n[full_baseline] === LLaDA-8B-Instruct Summary (mean ± std) ===", flush=True)
    for bench, mkey in [("gsm8k","exact_match"),("math500","exact_match"),
                         ("humaneval","pass_at_1"),("mbpp","pass_at_1")]:
        v = agg[bench][mkey]
        tps = agg[bench]["avg_tps"]
        print(f"  {bench:12s}: {v['mean']:.3f} ± {v['std']:.3f}  |  TPS={tps['mean']:.1f}", flush=True)

    # Unload LLaDA
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()
    report_progress(8, 10, {"phase": "llada_done"})

    # ── Step 4: Dream-7B Baseline ───────────────────────────────────────────────
    print("\n[full_baseline] === Dream-7B-Instruct Baseline ===", flush=True)
    dream_available = prepare_dream_7b()
    dream_metrics = None

    if dream_available:
        try:
            print("[Dream] Loading model...", flush=True)
            from transformers import AutoTokenizer as ATok, AutoModel as AMod
            d_tok = ATok.from_pretrained(DREAM_MODEL_PATH, trust_remote_code=True)
            if d_tok.padding_side != "left": d_tok.padding_side = "left"
            d_model = AMod.from_pretrained(
                DREAM_MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
            ).to("cuda:0").eval()
            d_vram = profile_vram()

            # Detect Dream mask ID
            d_mask_id = getattr(d_tok, "mask_token_id", None) or (d_tok.vocab_size - 1)
            print(f"[Dream] Loaded. mask_id={d_mask_id}, VRAM={d_vram}", flush=True)

            # Probe batch size for Dream
            d_batch = probe_batch_size(d_model, d_tok, start=8)

            d_results = {}
            for seed in seeds:
                torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
                tag = f"[Dream,seed={seed}]"
                sr = {}

                # Use subset for Dream (validate on 200 GSM8K, 50 HE)
                rng = random.Random(seed)
                g_sub = rng.sample(gsm8k_data, min(200, len(gsm8k_data)))
                h_sub = rng.sample(he_data, min(50, len(he_data)))

                # Create temporary wrapper functions that use Dream's mask_id
                def d_gen(model_, tokenizer_, prompts_, **kw):
                    from generate import generate as llada_generate
                    fmtd = []
                    for p_ in prompts_:
                        if kw.get("apply_chat_template", True):
                            try:
                                msg = [{"role": "user", "content": p_}]
                                fmtd.append(tokenizer_.apply_chat_template(
                                    msg, add_generation_prompt=True, tokenize=False))
                            except Exception:
                                fmtd.append(p_)
                        else:
                            fmtd.append(p_)
                    enc = tokenizer_(fmtd, add_special_tokens=False, padding=True, return_tensors="pt")
                    iids = enc["input_ids"].to("cuda:0")
                    amsk = enc["attention_mask"].to("cuda:0")
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = llada_generate(model_, iids, attention_mask=amsk,
                                             steps=kw.get("steps",64),
                                             gen_length=kw.get("gen_length",256),
                                             block_length=kw.get("block_length",32),
                                             temperature=0.0, cfg_scale=0.0,
                                             remasking="low_confidence",
                                             mask_id=d_mask_id)
                    el = time.perf_counter() - t0
                    plen = iids.shape[1]
                    gids = out[:, plen:]
                    tps_ = (gids.shape[1] * len(prompts_)) / el if el > 0 else 0
                    return tokenizer_.batch_decode(gids, skip_special_tokens=True), tps_, el

                # GSM8K subset
                gsm_tps, gsm_correct = [], 0
                for ii in range(0, len(g_sub), d_batch):
                    batch_ = g_sub[ii:ii+d_batch]
                    try:
                        txts, tps_, _ = d_gen(d_model, d_tok,
                                              [build_gsm8k_prompt(x["question"]) for x in batch_],
                                              steps=64, gen_length=256, block_length=32,
                                              apply_chat_template=True)
                        for x_, t_ in zip(batch_, txts):
                            if gsm8k_exact_match(t_, x_["answer"]): gsm_correct += 1
                        if ii // d_batch >= 1: gsm_tps.append(tps_)
                    except Exception: pass
                sr["gsm8k"] = {
                    "n_samples": len(g_sub), "correct": gsm_correct,
                    "exact_match": gsm_correct/len(g_sub) if g_sub else 0,
                    "avg_tps": float(np.mean(gsm_tps)) if gsm_tps else 0.0,
                    "tps_std": float(np.std(gsm_tps)) if gsm_tps else 0.0,
                }
                print(f"[Dream,seed={seed}] GSM8K: acc={sr['gsm8k']['exact_match']:.3f}", flush=True)

                # HumanEval subset
                he_tps, he_passed = [], 0
                for ii in range(0, len(h_sub), d_batch):
                    batch_ = h_sub[ii:ii+d_batch]
                    try:
                        txts, tps_, _ = d_gen(d_model, d_tok,
                                              [p_["prompt"] for p_ in batch_],
                                              steps=64, gen_length=256, block_length=32,
                                              apply_chat_template=False)
                        for prob_, code_ in zip(batch_, txts):
                            if humaneval_pass_at_1(code_, prob_): he_passed += 1
                        if ii // d_batch >= 1: he_tps.append(tps_)
                    except Exception: pass
                sr["humaneval"] = {
                    "n_samples": len(h_sub), "passed": he_passed,
                    "pass_at_1": he_passed/len(h_sub) if h_sub else 0,
                    "avg_tps": float(np.mean(he_tps)) if he_tps else 0.0,
                    "tps_std": float(np.std(he_tps)) if he_tps else 0.0,
                }
                print(f"[Dream,seed={seed}] HumanEval: pass@1={sr['humaneval']['pass_at_1']:.3f}", flush=True)
                d_results[str(seed)] = sr

            d_agg = {}
            for bench, mkey in [("gsm8k","exact_match"),("humaneval","pass_at_1")]:
                d_agg[bench] = {
                    mkey: aggregate_seeds(d_results, bench, mkey),
                    "avg_tps": aggregate_seeds(d_results, bench, "avg_tps"),
                }

            dream_metrics = {
                "task_id": TASK_ID + "_dream", "model": "Dream-7B-Instruct",
                "model_path": DREAM_MODEL_PATH, "mode": "baseline",
                "steps": 64, "seeds": seeds, "batch_size": d_batch,
                "aggregated": d_agg, "per_seed": d_results,
                "vram": {"after_load": d_vram},
                "timestamp": datetime.now().isoformat(),
            }
            del d_model, d_tok
            gc.collect(); torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"[Dream] Evaluation failed: {e}", flush=True)
            print(traceback.format_exc(), flush=True)
            dream_metrics = {
                "status": "failed", "error": str(e),
                "model": "Dream-7B-Instruct",
                "timestamp": datetime.now().isoformat(),
            }
    else:
        dream_metrics = {
            "status": "unavailable",
            "reason": "Model download failed or not available",
            "model": "Dream-7B-Instruct",
            "timestamp": datetime.now().isoformat(),
        }

    dream_out = RESULTS_DIR / "dream_baseline_full.json"
    dream_out.write_text(json.dumps(dream_metrics, indent=2))
    print(f"[full_baseline] Dream results saved to {dream_out}", flush=True)

    # ── Step 5: Update GPU progress ─────────────────────────────────────────────
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    if gpu_progress_path.exists():
        try: gp = json.loads(gpu_progress_path.read_text())
        except: gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
    else:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp["completed"]: gp["completed"].append(TASK_ID)
    gp["running"].pop(TASK_ID, None)
    elapsed_min = (datetime.now() - start_time).total_seconds() / 60
    gp["timings"][TASK_ID] = {
        "planned_min": 90, "actual_min": round(elapsed_min),
        "start_time": start_time.isoformat(), "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "LLaDA-8B-Instruct", "steps": 64, "gen_length": 256,
            "seeds": seeds, "batch_size": actual_batch,
            "benchmarks": ["gsm8k", "math500", "humaneval", "mbpp"],
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_count": 1,
        }
    }
    gpu_progress_path.parent.mkdir(parents=True, exist_ok=True)
    gpu_progress_path.write_text(json.dumps(gp, indent=2))

    # ── Done ─────────────────────────────────────────────────────────────────────
    report_progress(10, 10, {"phase": "done"})
    agg = llada_metrics["aggregated"]
    dream_status = (dream_metrics.get("status", "done")
                    if isinstance(dream_metrics, dict) else "done")
    summary = (
        f"LLaDA-8B: GSM8K={agg['gsm8k']['exact_match']['mean']:.3f}, "
        f"MATH500={agg['math500']['exact_match']['mean']:.3f}, "
        f"HumanEval={agg['humaneval']['pass_at_1']['mean']:.3f}, "
        f"MBPP={agg['mbpp']['pass_at_1']['mean']:.3f}; "
        f"Dream={dream_status}"
    )
    mark_done(status="success", summary=summary)
    elapsed_total = (datetime.now() - start_time).total_seconds() / 60
    print(f"\n[full_baseline] DONE in {elapsed_total:.1f} min. {summary}", flush=True)


if __name__ == "__main__":
    main()
