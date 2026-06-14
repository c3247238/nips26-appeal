"""
Pilot Baseline Evaluation: LLaDA-8B-Instruct (64-step, no acceleration).

Task: pilot_baseline
- 100 GSM8K + 50 HumanEval samples (pilot scale), seed=42
- Record: TPS, exact_match (GSM8K), pass@1 (HumanEval), VRAM usage
- Establishes denominator for all subsequent Speedup / Accuracy-Retention calculations.

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dlm-acceleration python pilot_baseline.py
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
MODEL_PATH   = str(SHARED / "checkpoints" / "llada-8b-instruct")
GSM8K_DIR    = str(SHARED / "datasets" / "gsm8k")
HUMANEVAL_DIR = str(SHARED / "datasets" / "humaneval")
CODE_DIR     = WORKSPACE / "exp" / "code"
RESULTS_DIR  = WORKSPACE / "exp" / "results" / "baseline"
TASK_ID      = "pilot_baseline"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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

# ── Data Loaders ──────────────────────────────────────────────────────────────
def load_gsm8k(n_samples=100, seed=42):
    path = Path(GSM8K_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n_samples, len(data)))

def load_humaneval(n_samples=50, seed=42):
    path = Path(HUMANEVAL_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n_samples, len(data)))

# ── GSM8K 8-shot Prompt ───────────────────────────────────────────────────────
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

def build_gsm8k_prompt(question):
    return GSM8K_8SHOT + f"Question: {question}\nAnswer:"

def extract_gsm8k_answer(text):
    match = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None

def gsm8k_exact_match(pred, gold):
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except: return p.strip() == g.strip()

def humaneval_pass_at_1(code_completion, problem):
    full_code = problem["prompt"] + code_completion + "\n" + problem["test"]
    full_code += f"\ncheck({problem['entry_point']})\n"
    try:
        result = subprocess.run(["python", "-c", full_code],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except: return False

# ── VRAM Profiler ─────────────────────────────────────────────────────────────
def profile_vram(device):
    if not torch.cuda.is_available(): return {}
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "vram_total_mb": props.total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(device) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(device) // 1024**2,
    }

# ── Main Evaluation ───────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now().isoformat()
    print(f"[pilot_baseline] Starting at {start_time}")

    # Seed
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # GPU
    device = "cuda:0"

    # Load model
    print(f"[pilot_baseline] Loading LLaDA-8B-Instruct from {MODEL_PATH}...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    MASK_ID = 126336

    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_after_load = profile_vram(device)
    print(f"[pilot_baseline] Model loaded. VRAM used: {vram_after_load.get('vram_used_mb',0)} MB")

    # Import LLaDA generate
    from generate import generate as llada_generate

    def generate_one(prompt_text, steps=64, gen_length=256, block_length=32,
                     apply_chat_template=True):
        if apply_chat_template:
            msg = [{"role": "user", "content": prompt_text}]
            prompt_text = tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
        enc = tokenizer([prompt_text], add_special_tokens=False,
                        padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        t0 = time.perf_counter()
        out = llada_generate(
            model, input_ids, attention_mask=attention_mask,
            steps=steps, gen_length=gen_length, block_length=block_length,
            temperature=0.0, cfg_scale=0.0, remasking="low_confidence",
            mask_id=MASK_ID,
        )
        elapsed = time.perf_counter() - t0

        prompt_len = input_ids.shape[1]
        generated_ids = out[:, prompt_len:]
        total_tokens = generated_ids.numel()
        tps = total_tokens / elapsed if elapsed > 0 else 0.0
        text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text, tps, elapsed

    # ── GSM8K Evaluation ─────────────────────────────────────────────────────
    print("\n[pilot_baseline] === GSM8K Evaluation ===")
    gsm8k_data = load_gsm8k(n_samples=100, seed=42)
    n_warmup = 5
    gsm8k_results = []
    gsm8k_tps_list = []
    gsm8k_correct = 0

    for i, item in enumerate(gsm8k_data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            pred, tps, elapsed = generate_one(prompt, steps=64, gen_length=256,
                                              block_length=32, apply_chat_template=True)
            correct = gsm8k_exact_match(pred, item["answer"])
            if correct: gsm8k_correct += 1
            if i >= n_warmup:
                gsm8k_tps_list.append(tps)
            gsm8k_results.append({
                "id": i, "question": item["question"][:100],
                "gold": item["answer"][-50:], "prediction": pred[:300],
                "correct": correct, "tps": round(tps, 2), "elapsed_sec": round(elapsed, 2),
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Sample {i}, skipping")
            torch.cuda.empty_cache(); gc.collect()
            gsm8k_results.append({"id": i, "error": "OOM", "correct": False, "tps": 0})
        except Exception as e:
            print(f"  [ERR] Sample {i}: {e}")
            gsm8k_results.append({"id": i, "error": str(e), "correct": False, "tps": 0})

        if (i + 1) % 10 == 0:
            acc = gsm8k_correct / (i + 1)
            avg_tps = np.mean(gsm8k_tps_list) if gsm8k_tps_list else 0.0
            print(f"  [{i+1}/100] acc={acc:.3f}, avg_tps={avg_tps:.1f}")
            report_progress(i+1, 150, {"gsm8k_accuracy": acc, "gsm8k_avg_tps": avg_tps})

    gsm8k_exact_match_score = gsm8k_correct / len(gsm8k_data) if gsm8k_data else 0.0
    gsm8k_avg_tps = float(np.mean(gsm8k_tps_list)) if gsm8k_tps_list else 0.0
    gsm8k_tps_std = float(np.std(gsm8k_tps_list)) if gsm8k_tps_list else 0.0

    print(f"\n[pilot_baseline] GSM8K Results:")
    print(f"  exact_match = {gsm8k_exact_match_score:.3f} ({gsm8k_correct}/{len(gsm8k_data)})")
    print(f"  avg_tps = {gsm8k_avg_tps:.1f} ± {gsm8k_tps_std:.1f}")

    # ── HumanEval Evaluation ─────────────────────────────────────────────────
    print("\n[pilot_baseline] === HumanEval Evaluation ===")
    humaneval_data = load_humaneval(n_samples=50, seed=42)
    he_tps_list = []
    he_passed = 0
    he_results = []

    for i, problem in enumerate(humaneval_data):
        prompt = problem["prompt"]
        try:
            code, tps, elapsed = generate_one(prompt, steps=64, gen_length=256,
                                              block_length=32, apply_chat_template=False)
            passed = humaneval_pass_at_1(code, problem)
            if passed: he_passed += 1
            if i >= 3:  # fewer warmup for humaneval
                he_tps_list.append(tps)
            he_results.append({
                "task_id": problem["task_id"], "completion": code[:400],
                "passed": passed, "tps": round(tps, 2),
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Problem {i}")
            torch.cuda.empty_cache(); gc.collect()
            he_results.append({"task_id": problem.get("task_id", str(i)), "error": "OOM", "passed": False, "tps": 0})
        except Exception as e:
            print(f"  [ERR] Problem {i}: {e}")
            he_results.append({"task_id": problem.get("task_id", str(i)), "error": str(e), "passed": False, "tps": 0})

        if (i + 1) % 10 == 0:
            p1 = he_passed / (i + 1)
            avg_tps = np.mean(he_tps_list) if he_tps_list else 0.0
            print(f"  [{i+1}/50] pass@1={p1:.3f}, avg_tps={avg_tps:.1f}")
            report_progress(100 + i + 1, 150, {"humaneval_pass_at_1": p1})

    he_pass_at_1 = he_passed / len(humaneval_data) if humaneval_data else 0.0
    he_avg_tps = float(np.mean(he_tps_list)) if he_tps_list else 0.0
    he_tps_std = float(np.std(he_tps_list)) if he_tps_list else 0.0

    print(f"\n[pilot_baseline] HumanEval Results:")
    print(f"  pass@1 = {he_pass_at_1:.3f} ({he_passed}/{len(humaneval_data)})")
    print(f"  avg_tps = {he_avg_tps:.1f} ± {he_tps_std:.1f}")

    # ── Final VRAM Profile ────────────────────────────────────────────────────
    vram_final = profile_vram(device)

    # ── Write Results ─────────────────────────────────────────────────────────
    end_time = datetime.now().isoformat()
    metrics = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "mode": "baseline",
        "steps": 64,
        "gen_length": 256,
        "block_length": 32,
        "seed": 42,
        "start_time": start_time,
        "end_time": end_time,
        "gsm8k": {
            "n_samples": len(gsm8k_data),
            "correct": gsm8k_correct,
            "exact_match": gsm8k_exact_match_score,
            "avg_tps": gsm8k_avg_tps,
            "tps_std": gsm8k_tps_std,
            "n_warmup": n_warmup,
        },
        "humaneval": {
            "n_samples": len(humaneval_data),
            "passed": he_passed,
            "pass_at_1": he_pass_at_1,
            "avg_tps": he_avg_tps,
            "tps_std": he_tps_std,
        },
        "overall_avg_tps": float(np.mean(
            [t for t in gsm8k_tps_list + he_tps_list if t > 0]
        )) if (gsm8k_tps_list or he_tps_list) else 0.0,
        "vram": {
            "after_load": vram_after_load,
            "final": vram_final,
        },
    }

    # Save baseline_metrics.json (canonical output path)
    out_path = RESULTS_DIR / "baseline_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[pilot_baseline] Saved metrics to {out_path}")

    # Save samples for qualitative inspection
    samples_path = RESULTS_DIR / "baseline_samples.json"
    samples_path.write_text(json.dumps({
        "gsm8k_samples": gsm8k_results[:20],
        "humaneval_samples": he_results[:20],
    }, indent=2))

    summary = (
        f"GSM8K exact_match={gsm8k_exact_match_score:.3f}, "
        f"HumanEval pass@1={he_pass_at_1:.3f}, "
        f"avg_tps={metrics['overall_avg_tps']:.1f}"
    )
    mark_done(status="success", summary=summary)

    print(f"\n[pilot_baseline] DONE. {summary}")
    return metrics


if __name__ == "__main__":
    main()
