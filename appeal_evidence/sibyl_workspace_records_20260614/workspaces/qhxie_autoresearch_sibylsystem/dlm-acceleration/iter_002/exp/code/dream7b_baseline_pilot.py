"""
Dream-7B-Instruct Baseline Pilot Evaluation.

Task: dream7b_baseline (PILOT mode)
- Download Dream-7B-Instruct (Dream-org/Dream-v0-Instruct-7B)
- Run 64-step baseline on 100 GSM8K + 100 MATH500, seed=42
- Establish reference TPS and accuracy for cross-model comparison
- FALLBACK: if Dream-7B fails at inference, report error and mark NO_GO

Pass criteria: Model loaded successfully AND GSM8K accuracy > 0.30

Output:
    exp/results/dream7b_baseline/dream7b_baseline_pilot.json
    exp/results/pilots/dream7b_baseline_pilot_summary.json

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n sibyl_dlm-acceleration python dream7b_baseline_pilot.py
"""
import os, sys, json, time, random, gc, re
from pathlib import Path
from datetime import datetime

import torch
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED    = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
DREAM_MODEL_PATH = str(SHARED / "checkpoints" / "dream-7b-instruct")
LLADA_MODEL_PATH = str(SHARED / "checkpoints" / "llada-8b-instruct")
GSM8K_DIR   = str(SHARED / "datasets" / "gsm8k")
MATH500_DIR = str(SHARED / "datasets" / "math500")
CODE_DIR    = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "dream7b_baseline"
PILOT_DIR   = WORKSPACE / "exp" / "results" / "pilots"
TASK_ID     = "dream7b_baseline"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

# Dream mask token ID from config.json
DREAM_MASK_ID = 151666

# Pilot configuration
PILOT_SAMPLES_GSM8K = 100
PILOT_SAMPLES_MATH500 = 100
PILOT_SEED = 42
PILOT_STEPS = 64
PILOT_TIMEOUT = 900  # 15 minutes

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
        print(f"[data] GSM8K loaded: {len(data)} examples", flush=True)
        return data
    else:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split="test")
        data = [{"question": x["question"], "answer": x["answer"]} for x in ds]
        Path(GSM8K_DIR).mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[data] GSM8K downloaded and saved: {len(data)} examples", flush=True)
        return data

def prepare_math500():
    path = Path(MATH500_DIR) / "test.json"
    if path.exists():
        with open(path) as f:
            data = json.load(f)
        print(f"[data] MATH500 loaded: {len(data)} examples", flush=True)
        return data
    else:
        from datasets import load_dataset
        Path(MATH500_DIR).mkdir(parents=True, exist_ok=True)
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        data = [{"problem": x["problem"], "solution": x["solution"],
                 "answer": x["answer"], "subject": x["subject"],
                 "level": x["level"]} for x in ds]
        with open(path, "w") as f:
            json.dump(data, f)
        print(f"[data] MATH500 downloaded and saved: {len(data)} examples", flush=True)
        return data

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

# ── Dream-7B Generation ──────────────────────────────────────────────────────
def dream_generate_batch(model, tokenizer, prompts, steps=64, max_new_tokens=256,
                         device="cuda:0", apply_chat_template=True,
                         alg="entropy", temperature=0.0):
    """
    Generate completions with Dream-7B using model.diffusion_generate().
    Returns: list of texts, TPS, elapsed time
    """
    if apply_chat_template:
        formatted = []
        for p in prompts:
            msg = [{"role": "user", "content": p}]
            try:
                formatted.append(tokenizer.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=False))
            except Exception:
                formatted.append(p)
        prompts = formatted

    enc = tokenizer(prompts, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            steps=steps,
            temperature=temperature,
            alg=alg,
            alg_temp=0.0,
        )
    elapsed = time.perf_counter() - t0

    # Extract generated tokens (output may be DreamModelOutput or tensor)
    if hasattr(out, 'sequences'):
        output_ids = out.sequences
    else:
        output_ids = out

    generated_ids = output_ids[:, prompt_len:]
    gen_len = generated_ids.shape[1]
    batch_size = len(prompts)
    tps = (gen_len * batch_size) / elapsed if elapsed > 0 else 0.0
    per_sample_tps = gen_len / (elapsed / batch_size) if elapsed > 0 else 0.0

    texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return texts, per_sample_tps, elapsed

# ── VRAM Probing for Dream ────────────────────────────────────────────────────
def probe_dream_batch_size(model, tokenizer, device="cuda:0",
                           max_new_tokens=256, start=4, min_bs=1):
    """Find max batch size for Dream via binary search."""
    print(f"[probe] Probing max batch size for Dream (start={start})...", flush=True)

    high, best = start, min_bs
    while min_bs <= high:
        mid = (min_bs + high) // 2
        try:
            torch.cuda.empty_cache(); gc.collect()
            # Create dummy batch with short prompt
            dummy_prompts = ["What is 2+2?"] * mid
            enc = tokenizer(dummy_prompts, padding=True, return_tensors="pt")
            iids = enc["input_ids"].to(device)
            amsk = enc["attention_mask"].to(device)
            with torch.no_grad():
                _ = model.diffusion_generate(
                    iids,
                    attention_mask=amsk,
                    max_new_tokens=max_new_tokens,
                    steps=4,  # minimal steps for probing
                    temperature=0.0,
                    alg="origin",
                )
            best = mid
            min_bs = mid + 1
            del iids, amsk
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            high = mid - 1
            torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"[probe] Error at bs={mid}: {e}", flush=True)
            high = mid - 1

    print(f"[probe] Max batch size for Dream: {best}", flush=True)
    return max(1, best)

# ── Core Evaluation Functions ────────────────────────────────────────────────
def eval_gsm8k_dream(model, tokenizer, data, seed=42, device="cuda:0",
                     batch_size=1, n_warmup=2, steps=64):
    """Evaluate Dream-7B on GSM8K subset."""
    rng = random.Random(seed)
    subset = rng.sample(data, min(PILOT_SAMPLES_GSM8K, len(data)))

    results = []
    tps_list = []
    correct_count = 0
    n = len(subset)
    batch_idx = 0
    sample_outputs = []

    print(f"\n[eval] GSM8K: {n} examples, seed={seed}, batch_size={batch_size}, steps={steps}", flush=True)

    for i in range(0, n, batch_size):
        batch_items = subset[i:i+batch_size]
        prompts = [build_gsm8k_prompt(item["question"]) for item in batch_items]
        try:
            texts, tps, elapsed = dream_generate_batch(
                model, tokenizer, prompts,
                steps=steps, max_new_tokens=256,
                device=device, apply_chat_template=True
            )
            for j, (item, pred) in enumerate(zip(batch_items, texts)):
                correct = gsm8k_exact_match(pred, item["answer"])
                if correct: correct_count += 1
                result_item = {
                    "idx": i+j, "correct": correct,
                    "prediction": pred[:300],
                    "gold": extract_gsm8k_answer(item["answer"]),
                    "extracted_pred": extract_gsm8k_answer(pred),
                }
                results.append(result_item)
                # Save first 10 sample outputs for qualitative inspection
                if len(sample_outputs) < 10:
                    sample_outputs.append({
                        "question": item["question"][:200],
                        "prediction": pred[:500],
                        "gold_answer": item["answer"][:200],
                        "correct": correct,
                    })
            if batch_idx >= n_warmup:
                tps_list.append(tps)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [OOM] batch {i//batch_size}, falling back to batch=1...", flush=True)
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _ = dream_generate_batch(
                        model, tokenizer, [build_gsm8k_prompt(item["question"])],
                        steps=steps, max_new_tokens=256,
                        device=device, apply_chat_template=True)
                    correct = gsm8k_exact_match(texts_s[0], item["answer"])
                    if correct: correct_count += 1
                    if batch_idx >= n_warmup: tps_list.append(tps_s)
                    results.append({"idx": i+j, "correct": correct})
                    batch_idx += 1
                except Exception as e:
                    results.append({"idx": i+j, "error": str(e)[:100], "correct": False})
        except Exception as e:
            print(f"  [ERR] batch {i//batch_size}: {str(e)[:100]}", flush=True)
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "error": str(e)[:100], "correct": False})

        done = min(i + batch_size, n)
        if done % max(batch_size * 5, 20) == 0 or done == n:
            acc = correct_count / done if done > 0 else 0
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"  [{done}/{n}] acc={acc:.3f}, avg_tps={avg_tps:.1f}", flush=True)

    exact_match = correct_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if tps_list else 0.0
    return {
        "n_samples": n, "correct": correct_count,
        "exact_match": exact_match, "avg_tps": avg_tps,
        "tps_std": tps_std, "n_warmup": n_warmup,
        "sample_outputs": sample_outputs,
    }, results


def eval_math500_dream(model, tokenizer, data, seed=42, device="cuda:0",
                       batch_size=1, n_warmup=2, steps=64):
    """Evaluate Dream-7B on MATH500 subset."""
    rng = random.Random(seed)
    subset = rng.sample(data, min(PILOT_SAMPLES_MATH500, len(data)))

    results = []
    tps_list = []
    correct_count = 0
    n = len(subset)
    batch_idx = 0
    sample_outputs = []

    # MATH500 needs longer generation
    math_batch = max(1, batch_size // 2)
    print(f"\n[eval] MATH500: {n} examples, seed={seed}, batch_size={math_batch}, steps={steps}", flush=True)

    for i in range(0, n, math_batch):
        batch_items = subset[i:i+math_batch]
        prompts = [build_math500_prompt(item["problem"]) for item in batch_items]
        try:
            texts, tps, elapsed = dream_generate_batch(
                model, tokenizer, prompts,
                steps=steps, max_new_tokens=512,
                device=device, apply_chat_template=True
            )
            for j, (item, pred) in enumerate(zip(batch_items, texts)):
                correct = math500_exact_match(pred, item["answer"])
                if correct: correct_count += 1
                results.append({
                    "idx": i+j, "correct": correct,
                    "gold_answer": item["answer"],
                    "prediction": pred[:300],
                })
                if len(sample_outputs) < 5:
                    sample_outputs.append({
                        "problem": item["problem"][:200],
                        "prediction": pred[:500],
                        "gold_answer": item["answer"],
                        "correct": correct,
                    })
            if batch_idx >= n_warmup: tps_list.append(tps)
            batch_idx += 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); gc.collect()
            print(f"  [OOM] batch {i//math_batch}, falling back to batch=1...", flush=True)
            for j, item in enumerate(batch_items):
                try:
                    texts_s, tps_s, _ = dream_generate_batch(
                        model, tokenizer, [build_math500_prompt(item["problem"])],
                        steps=steps, max_new_tokens=512,
                        device=device, apply_chat_template=True)
                    correct = math500_exact_match(texts_s[0], item["answer"])
                    if correct: correct_count += 1
                    if batch_idx >= n_warmup: tps_list.append(tps_s)
                    results.append({"idx": i+j, "correct": correct})
                    batch_idx += 1
                except Exception as e:
                    results.append({"idx": i+j, "error": str(e)[:100], "correct": False})
        except Exception as e:
            print(f"  [ERR] batch {i//math_batch}: {str(e)[:100]}", flush=True)
            for j in range(len(batch_items)):
                results.append({"idx": i+j, "error": str(e)[:100], "correct": False})

        done = min(i + math_batch, n)
        if done % 50 == 0 or done == n:
            acc = correct_count / done if done > 0 else 0
            avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"  [{done}/{n}] acc={acc:.3f}, avg_tps={avg_tps:.1f}", flush=True)

    exact_match = correct_count / n if n else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return {
        "n_samples": n, "correct": correct_count,
        "exact_match": exact_match, "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
        "sample_outputs": sample_outputs,
    }, results


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now()
    device = "cuda:0"
    print(f"[dream7b_baseline] PILOT starting at {start_time.isoformat()}", flush=True)
    print(f"[dream7b_baseline] Seed={PILOT_SEED}, Steps={PILOT_STEPS}", flush=True)
    print(f"[dream7b_baseline] GSM8K samples={PILOT_SAMPLES_GSM8K}, MATH500 samples={PILOT_SAMPLES_MATH500}", flush=True)
    if torch.cuda.is_available():
        print(f"[dream7b_baseline] GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"[dream7b_baseline] VRAM: {torch.cuda.get_device_properties(0).total_memory // 1024**2} MB", flush=True)

    # ── Step 1: Prepare datasets ──────────────────────────────────────────────
    print("\n=== Step 1: Preparing Datasets ===", flush=True)
    report_progress(0, 8, {"phase": "data_prep"})
    gsm8k_data = prepare_gsm8k()
    math500_data = prepare_math500()
    report_progress(1, 8, {"phase": "data_ready"})

    # ── Step 2: Load Dream-7B-Instruct ─────────────────────────────────────────
    print("\n=== Step 2: Loading Dream-7B-Instruct ===", flush=True)
    print(f"  Model path: {DREAM_MODEL_PATH}", flush=True)
    report_progress(2, 8, {"phase": "loading_dream"})

    model_loaded = False
    load_error = None

    try:
        from transformers import AutoTokenizer, AutoModel

        print("  Loading tokenizer...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(
            DREAM_MODEL_PATH, trust_remote_code=True)

        # Ensure left padding for batched generation
        if tokenizer.padding_side != "left":
            tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Verify mask token
        mask_id = getattr(tokenizer, "mask_token_id", None)
        config_mask_id = DREAM_MASK_ID
        print(f"  Tokenizer mask_token_id={mask_id}, config mask_token_id={config_mask_id}", flush=True)
        print(f"  Vocab size={tokenizer.vocab_size}", flush=True)

        print("  Loading model (bfloat16)...", flush=True)
        model = AutoModel.from_pretrained(
            DREAM_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ).to(device).eval()

        vram_after_load = profile_vram(device)
        print(f"  Model loaded! VRAM: {vram_after_load}", flush=True)

        # Verify the model has diffusion_generate
        if not hasattr(model, 'diffusion_generate'):
            raise RuntimeError("Dream model does not have diffusion_generate method!")

        model_loaded = True
        report_progress(3, 8, {"phase": "model_loaded", "vram": vram_after_load})

    except Exception as e:
        import traceback
        load_error = str(e)
        print(f"  [FAIL] Dream-7B loading failed: {e}", flush=True)
        print(traceback.format_exc(), flush=True)

    if not model_loaded:
        # Write failure result
        fail_result = {
            "task_id": TASK_ID,
            "mode": "pilot",
            "model": "Dream-7B-Instruct",
            "model_repo": "Dream-org/Dream-v0-Instruct-7B",
            "status": "failed",
            "error": load_error,
            "go_no_go": "NO_GO",
            "confidence": 0.0,
            "timestamp": datetime.now().isoformat(),
        }
        (RESULTS_DIR / "dream7b_baseline_pilot.json").write_text(json.dumps(fail_result, indent=2))
        _write_pilot_summary(fail_result)
        _update_gpu_progress(start_time, status="failed")
        mark_done(status="failed", summary=f"Dream-7B loading failed: {load_error[:200]}")
        return

    # ── Step 3: Quick smoke test ─────────────────────────────────────────────
    print("\n=== Step 3: Smoke Test ===", flush=True)
    report_progress(3, 8, {"phase": "smoke_test"})
    try:
        test_texts, test_tps, test_elapsed = dream_generate_batch(
            model, tokenizer, ["What is 2+2? Answer briefly."],
            steps=16, max_new_tokens=32, device=device,
            apply_chat_template=True
        )
        print(f"  Smoke test output: {test_texts[0][:100]}", flush=True)
        print(f"  Smoke test TPS: {test_tps:.1f}, elapsed: {test_elapsed:.2f}s", flush=True)
    except Exception as e:
        print(f"  [WARN] Smoke test failed: {e}", flush=True)
        print("  Continuing with evaluation anyway...", flush=True)

    # ── Step 4: Probe batch size ─────────────────────────────────────────────
    print("\n=== Step 4: VRAM Probing ===", flush=True)
    report_progress(4, 8, {"phase": "vram_probe"})
    actual_batch = probe_dream_batch_size(model, tokenizer, device=device, start=4)
    print(f"  Using batch_size={actual_batch}", flush=True)

    # Save GPU profile
    gpu_profile = {
        **profile_vram(device),
        "max_batch_size": actual_batch,
        "model": "Dream-7B-Instruct",
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2))

    # ── Step 5: GSM8K Evaluation ─────────────────────────────────────────────
    print("\n=== Step 5: GSM8K Evaluation ===", flush=True)
    torch.manual_seed(PILOT_SEED); np.random.seed(PILOT_SEED); random.seed(PILOT_SEED)
    report_progress(5, 8, {"phase": "gsm8k_eval"})

    gsm8k_metrics, gsm8k_results = eval_gsm8k_dream(
        model, tokenizer, gsm8k_data,
        seed=PILOT_SEED, device=device,
        batch_size=actual_batch, steps=PILOT_STEPS
    )
    print(f"\n  GSM8K Results: acc={gsm8k_metrics['exact_match']:.3f}, "
          f"tps={gsm8k_metrics['avg_tps']:.1f}", flush=True)
    report_progress(6, 8, {"phase": "gsm8k_done",
                           "gsm8k_acc": gsm8k_metrics['exact_match'],
                           "gsm8k_tps": gsm8k_metrics['avg_tps']})

    # ── Step 6: MATH500 Evaluation ───────────────────────────────────────────
    print("\n=== Step 6: MATH500 Evaluation ===", flush=True)
    torch.manual_seed(PILOT_SEED); np.random.seed(PILOT_SEED); random.seed(PILOT_SEED)
    report_progress(6, 8, {"phase": "math500_eval"})

    math500_metrics, math500_results = eval_math500_dream(
        model, tokenizer, math500_data,
        seed=PILOT_SEED, device=device,
        batch_size=actual_batch, steps=PILOT_STEPS
    )
    print(f"\n  MATH500 Results: acc={math500_metrics['exact_match']:.3f}, "
          f"tps={math500_metrics['avg_tps']:.1f}", flush=True)
    report_progress(7, 8, {"phase": "math500_done",
                           "math500_acc": math500_metrics['exact_match'],
                           "math500_tps": math500_metrics['avg_tps']})

    # ── Step 7: Compare with LLaDA baseline ───────────────────────────────────
    print("\n=== Step 7: Cross-Model Comparison ===", flush=True)
    llada_baseline_path = WORKSPACE / "iter_001" / "exp" / "results" / "full_baseline" / "llada_baseline_full.json"
    llada_ref = {}
    if llada_baseline_path.exists():
        try:
            llada_data = json.loads(llada_baseline_path.read_text())
            # Extract seed-42 or aggregated results
            if "aggregated" in llada_data:
                agg = llada_data["aggregated"]
                llada_ref = {
                    "gsm8k_acc": agg.get("gsm8k", {}).get("exact_match", {}).get("mean"),
                    "gsm8k_tps": agg.get("gsm8k", {}).get("avg_tps", {}).get("mean"),
                    "math500_acc": agg.get("math500", {}).get("exact_match", {}).get("mean"),
                    "math500_tps": agg.get("math500", {}).get("avg_tps", {}).get("mean"),
                }
            elif "per_seed" in llada_data:
                s42 = llada_data["per_seed"].get("42", {})
                llada_ref = {
                    "gsm8k_acc": s42.get("gsm8k", {}).get("exact_match"),
                    "gsm8k_tps": s42.get("gsm8k", {}).get("avg_tps"),
                    "math500_acc": s42.get("math500", {}).get("exact_match"),
                    "math500_tps": s42.get("math500", {}).get("avg_tps"),
                }
            print(f"  LLaDA reference: GSM8K acc={llada_ref.get('gsm8k_acc')}, "
                  f"TPS={llada_ref.get('gsm8k_tps')}", flush=True)
        except Exception as e:
            print(f"  [WARN] Could not load LLaDA reference: {e}", flush=True)
    else:
        # Try current iter results
        alt_path = WORKSPACE / "exp" / "results" / "full_baseline" / "llada_baseline_full.json"
        if alt_path.exists():
            try:
                llada_data = json.loads(alt_path.read_text())
                if "aggregated" in llada_data:
                    agg = llada_data["aggregated"]
                    llada_ref = {
                        "gsm8k_acc": agg.get("gsm8k", {}).get("exact_match", {}).get("mean"),
                        "gsm8k_tps": agg.get("gsm8k", {}).get("avg_tps", {}).get("mean"),
                        "math500_acc": agg.get("math500", {}).get("exact_match", {}).get("mean"),
                        "math500_tps": agg.get("math500", {}).get("avg_tps", {}).get("mean"),
                    }
                print(f"  LLaDA reference (from current iter): "
                      f"GSM8K acc={llada_ref.get('gsm8k_acc')}", flush=True)
            except: pass
        else:
            print("  [INFO] No LLaDA baseline reference found. Using known value: GSM8K=0.712", flush=True)
            llada_ref = {"gsm8k_acc": 0.712, "gsm8k_tps": 31.0,
                         "math500_acc": 0.111, "math500_tps": None}

    # ── Step 8: Compile Results ─────────────────────────────────────────────────
    print("\n=== Step 8: Compiling Results ===", flush=True)
    elapsed_min = (datetime.now() - start_time).total_seconds() / 60

    # GO/NO_GO decision
    gsm8k_acc = gsm8k_metrics['exact_match']
    go_verdict = "GO" if gsm8k_acc > 0.30 else "NO_GO"
    confidence = min(1.0, gsm8k_acc / 0.30) if gsm8k_acc <= 0.30 else min(1.0, gsm8k_acc)

    vram_final = profile_vram(device)

    pilot_result = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "model": "Dream-7B-Instruct",
        "model_repo": "Dream-org/Dream-v0-Instruct-7B",
        "model_path": DREAM_MODEL_PATH,
        "status": "success",
        "go_no_go": go_verdict,
        "confidence": confidence,
        "seed": PILOT_SEED,
        "steps": PILOT_STEPS,
        "batch_size": actual_batch,
        "mask_token_id": DREAM_MASK_ID,
        "benchmarks": {
            "gsm8k": gsm8k_metrics,
            "math500": math500_metrics,
        },
        "cross_model_comparison": {
            "dream_gsm8k_acc": gsm8k_acc,
            "dream_gsm8k_tps": gsm8k_metrics['avg_tps'],
            "dream_math500_acc": math500_metrics['exact_match'],
            "dream_math500_tps": math500_metrics['avg_tps'],
            "llada_gsm8k_acc": llada_ref.get('gsm8k_acc'),
            "llada_gsm8k_tps": llada_ref.get('gsm8k_tps'),
            "llada_math500_acc": llada_ref.get('math500_acc'),
            "llada_math500_tps": llada_ref.get('math500_tps'),
        },
        "vram": {"after_load": vram_after_load, "final": vram_final},
        "gpu_profile": gpu_profile,
        "elapsed_min": round(elapsed_min, 1),
        "timestamp": datetime.now().isoformat(),
        "notes": [
            f"Correct repo: Dream-org/Dream-v0-Instruct-7B (iter_001 used wrong name hkunlp/dream-7b-instruct)",
            f"Dream mask_token_id={DREAM_MASK_ID} (vs LLaDA mask_id=126336)",
            f"Dream uses diffusion_generate() with alg='entropy' (not LLaDA-style generate())",
        ]
    }

    # Save results
    out_path = RESULTS_DIR / "dream7b_baseline_pilot.json"
    out_path.write_text(json.dumps(pilot_result, indent=2))
    print(f"\n  Results saved to {out_path}", flush=True)

    # Save detailed results
    detailed = {
        "gsm8k_results": gsm8k_results[:50],  # first 50 for space
        "math500_results": math500_results[:50],
    }
    (RESULTS_DIR / "dream7b_baseline_pilot_detailed.json").write_text(
        json.dumps(detailed, indent=2))

    # Write pilot summary
    _write_pilot_summary(pilot_result)

    # Update GPU progress
    _update_gpu_progress(start_time)

    # ── Final Summary ──────────────────────────────────────────────────────────
    print("\n" + "="*60, flush=True)
    print("  Dream-7B-Instruct Pilot Summary", flush=True)
    print("="*60, flush=True)
    print(f"  Verdict: {go_verdict} (confidence={confidence:.2f})", flush=True)
    print(f"  GSM8K:   acc={gsm8k_acc:.3f}, tps={gsm8k_metrics['avg_tps']:.1f}", flush=True)
    print(f"  MATH500: acc={math500_metrics['exact_match']:.3f}, tps={math500_metrics['avg_tps']:.1f}", flush=True)
    if llada_ref.get('gsm8k_acc'):
        print(f"  LLaDA ref: GSM8K acc={llada_ref['gsm8k_acc']:.3f}", flush=True)
        acc_ratio = gsm8k_acc / llada_ref['gsm8k_acc'] if llada_ref['gsm8k_acc'] > 0 else 0
        print(f"  Dream/LLaDA GSM8K accuracy ratio: {acc_ratio:.2f}", flush=True)
    print(f"  Elapsed: {elapsed_min:.1f} min", flush=True)
    print("="*60, flush=True)

    # Mark done
    summary_text = (
        f"Dream-7B Pilot: {go_verdict}, "
        f"GSM8K acc={gsm8k_acc:.3f} (tps={gsm8k_metrics['avg_tps']:.1f}), "
        f"MATH500 acc={math500_metrics['exact_match']:.3f} (tps={math500_metrics['avg_tps']:.1f}), "
        f"{elapsed_min:.1f}min"
    )
    report_progress(8, 8, {"phase": "done", "verdict": go_verdict})
    mark_done(status="success", summary=summary_text)
    print(f"\n[dream7b_baseline] DONE. {summary_text}", flush=True)

    # Cleanup
    del model, tokenizer
    gc.collect(); torch.cuda.empty_cache()


def _write_pilot_summary(pilot_result):
    """Write pilot_summary.json and pilot_summary.md."""
    is_success = pilot_result.get("status") == "success"
    go = pilot_result.get("go_no_go", "NO_GO")

    summary_json = {
        "overall_recommendation": go,
        "selected_candidate_id": "cand_composeaccel",
        "candidates": [{
            "candidate_id": "cand_composeaccel",
            "go_no_go": go,
            "confidence": pilot_result.get("confidence", 0.0),
            "supported_hypotheses": ["Dream-7B cross-model validation feasible"] if go == "GO" else [],
            "failed_assumptions": [] if go == "GO" else ["Dream-7B GSM8K accuracy < 0.30"],
            "key_metrics": {
                "gsm8k_acc": pilot_result.get("benchmarks", {}).get("gsm8k", {}).get("exact_match", 0),
                "gsm8k_tps": pilot_result.get("benchmarks", {}).get("gsm8k", {}).get("avg_tps", 0),
                "math500_acc": pilot_result.get("benchmarks", {}).get("math500", {}).get("exact_match", 0),
                "math500_tps": pilot_result.get("benchmarks", {}).get("math500", {}).get("avg_tps", 0),
            } if is_success else {},
            "notes": pilot_result.get("error", "Dream-7B baseline established") if not is_success
                     else f"Dream-7B baseline established: GSM8K={pilot_result['benchmarks']['gsm8k']['exact_match']:.3f}",
        }],
    }
    (PILOT_DIR / "dream7b_baseline_pilot_summary.json").write_text(
        json.dumps(summary_json, indent=2))

    # Also write markdown summary
    if is_success:
        benchmarks = pilot_result.get("benchmarks", {})
        gsm_m = benchmarks.get("gsm8k", {})
        math_m = benchmarks.get("math500", {})
        comparison = pilot_result.get("cross_model_comparison", {})
        md = f"""# Dream-7B-Instruct Baseline Pilot Summary

## Verdict: {go}

## Benchmarks (seed={pilot_result.get('seed', 42)}, steps={pilot_result.get('steps', 64)})

| Benchmark | Accuracy | TPS |
|-----------|----------|-----|
| GSM8K ({gsm_m.get('n_samples', 0)} samples) | {gsm_m.get('exact_match', 0):.3f} | {gsm_m.get('avg_tps', 0):.1f} |
| MATH500 ({math_m.get('n_samples', 0)} samples) | {math_m.get('exact_match', 0):.3f} | {math_m.get('avg_tps', 0):.1f} |

## Cross-Model Comparison

| Model | GSM8K Acc | GSM8K TPS | MATH500 Acc |
|-------|-----------|-----------|-------------|
| Dream-7B | {comparison.get('dream_gsm8k_acc', 'N/A')} | {comparison.get('dream_gsm8k_tps', 'N/A')} | {comparison.get('dream_math500_acc', 'N/A')} |
| LLaDA-8B | {comparison.get('llada_gsm8k_acc', 'N/A')} | {comparison.get('llada_gsm8k_tps', 'N/A')} | {comparison.get('llada_math500_acc', 'N/A')} |

## Notes
- Model repo: Dream-org/Dream-v0-Instruct-7B (corrected from iter_001 wrong name)
- Dream mask_token_id={pilot_result.get('mask_token_id', 'N/A')}
- Elapsed: {pilot_result.get('elapsed_min', 0):.1f} min
"""
    else:
        md = f"""# Dream-7B-Instruct Baseline Pilot Summary

## Verdict: NO_GO

## Error
{pilot_result.get('error', 'Unknown error')}
"""
    (PILOT_DIR / "dream7b_baseline_pilot_summary.md").write_text(md)


def _update_gpu_progress(start_time, status="completed"):
    """Update gpu_progress.json."""
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / ".gpu_progress.lock"

    # Simple file-based lock
    import fcntl
    try:
        lock_fd = open(lock_path, 'w')
        fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except (IOError, OSError):
        # If lock fails, proceed anyway (better than blocking)
        lock_fd = None

    try:
        if gpu_progress_path.exists():
            try: gp = json.loads(gpu_progress_path.read_text())
            except: gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if status == "completed":
            if TASK_ID not in gp["completed"]:
                gp["completed"].append(TASK_ID)
        elif status == "failed":
            if TASK_ID not in gp.get("failed", []):
                gp.setdefault("failed", []).append(TASK_ID)

        # Remove from running
        gp.get("running", {}).pop(TASK_ID, None)

        elapsed_min = (datetime.now() - start_time).total_seconds() / 60
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 45,
            "actual_min": round(elapsed_min, 1),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "Dream-7B-Instruct",
                "model_repo": "Dream-org/Dream-v0-Instruct-7B",
                "steps": PILOT_STEPS,
                "mode": "pilot",
                "benchmarks": ["gsm8k", "math500"],
                "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "gpu_count": 1,
            }
        }

        gpu_progress_path.write_text(json.dumps(gp, indent=2))
    finally:
        if lock_fd:
            try:
                fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
                lock_fd.close()
            except: pass


if __name__ == "__main__":
    main()
