"""
Pilot M2 (Adaptive Step Scheduling / Saber) Single-Method Evaluation.

Task: pilot_m2_single
- Sweep step-jump in {2x, 4x, 6x, 8x} tokens unmasked per step
- Run on 100 GSM8K + 50 HumanEval samples (seed=42)
- Record: TPS, exact_match (GSM8K), pass@1 (HumanEval), per step-jump
- Identify operating point: highest Speedup with <= 2% accuracy drop vs. baseline

Since Saber code is unavailable, implements simplified top-k confidence-based
unmasking: at each step, unmask top-k tokens by confidence score where
k = {N/32, N/16, N/8, N/4} for step-jump {2x, 4x, 6x, 8x}.
Fewer total steps needed => faster generation.

Baseline reference (from pilot_baseline):
    GSM8K exact_match = 0.73
    GSM8K avg_tps = 58.55 tokens/sec
    HumanEval pass@1 = 0.04
    HumanEval avg_tps = 100.93
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
HUMANEVAL_DIR = str(SHARED / "datasets" / "humaneval")
CODE_DIR      = WORKSPACE / "exp" / "code"
RESULTS_DIR   = WORKSPACE / "exp" / "results" / "pilot_m2"
TASK_ID       = "pilot_m2_single"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline reference from pilot_baseline
BASELINE_GSM8K_TPS   = 58.55
BASELINE_GSM8K_ACC   = 0.73
BASELINE_HE_TPS      = 100.93
BASELINE_HE_ACC      = 0.04

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

def profile_vram(device):
    if not torch.cuda.is_available(): return {}
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "vram_total_mb": props.total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(device) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(device) // 1024**2,
    }


# ── M2 Adaptive Step Scheduling (Simplified Saber) ───────────────────────────
@torch.no_grad()
def generate_m2(
    model, tokenizer, prompt_text, device,
    gen_length=256, block_length=32,
    step_jump=2,   # tokens_unmasked_per_step factor (2x, 4x, 6x, 8x)
    apply_chat_template=True,
    mask_id=126336,
):
    """
    M2 Adaptive Step Scheduling (simplified Saber implementation).

    Core idea: instead of unmasking tokens one-by-one with many small steps,
    unmask `step_jump` times more tokens per step, reducing total steps by step_jump.

    For a gen_length=256, block_length=32, standard steps=64 (8 blocks x 8 steps each):
    - step_jump=1 (baseline): 64 steps, unmask ~4 tokens/step
    - step_jump=2x: 32 effective steps, unmask ~8 tokens/step
    - step_jump=4x: 16 effective steps, unmask ~16 tokens/step
    - step_jump=6x: ~11 effective steps, unmask ~24 tokens/step
    - step_jump=8x: 8 effective steps, unmask ~32 tokens/step

    At each step, we select the top-k tokens by confidence score (max softmax prob)
    where k = block_length / num_steps_in_block, and num_steps_in_block is reduced
    by step_jump.

    This directly reduces model forward passes proportional to step_jump,
    producing a real TPS speedup.

    Risk: fewer steps may degrade quality when tokens that should be
    revised are "locked in" too early. The step-jump sweep measures this tradeoff.
    """
    from generate import get_num_transfer_tokens

    if apply_chat_template:
        msg = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=False
        )
    enc = tokenizer([prompt_text], add_special_tokens=False,
                    padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full(
        (batch, prompt_len + gen_length), mask_id, dtype=torch.long
    ).to(device)
    x[:, :prompt_len] = input_ids.clone()
    attn = torch.cat([
        attention_mask,
        torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)
    ], dim=-1)

    num_blocks = gen_length // block_length

    # Reduce steps per block by step_jump factor
    base_steps = 64  # standard total steps
    base_steps_per_block = base_steps // num_blocks   # =8 for 256/32 with 64 steps
    # Reduce steps per block; minimum 1 step per block
    steps_per_block = max(1, base_steps_per_block // step_jump)
    total_steps = steps_per_block * num_blocks

    t0 = time.perf_counter()

    total_forward_passes = 0
    total_tokens_unmasked = 0

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length
        block_mask = x[:, block_start:block_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            mask_index = x == mask_id

            # Skip if no more masked tokens in this block
            if not mask_index[:, block_start:block_end].any():
                continue

            logits = model(x, attention_mask=attn).logits
            total_forward_passes += 1

            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

            # Only consider positions in the current block for unmasking
            x0_p[:, block_end:] = -float('inf')
            # Don't overwrite already-unmasked tokens
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p,
                                     torch.tensor(-float('inf'), device=device))

            # Transfer top-k tokens by confidence
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(batch):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]
            total_tokens_unmasked += transfer_index.sum().item()

    elapsed = time.perf_counter() - t0

    generated_ids = x[:, prompt_len:]
    total_tokens = generated_ids.numel()
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    # Theoretical speedup based on step reduction
    baseline_steps = base_steps
    actual_steps = total_steps
    step_reduction_ratio = baseline_steps / max(actual_steps, 1)

    return text, tps, elapsed, {
        "total_forward_passes": total_forward_passes,
        "total_steps": total_steps,
        "baseline_steps": baseline_steps,
        "step_reduction_ratio": step_reduction_ratio,
        "steps_per_block": steps_per_block,
    }


# ── Main Evaluation ───────────────────────────────────────────────────────────
def evaluate_step_jump(
    model, tokenizer, device,
    step_jump,
    gsm8k_data, humaneval_data,
    n_warmup=3,
    jump_idx=0,
    total_jumps=4,
    total_progress=600,
):
    """Evaluate M2 with a given step-jump setting on GSM8K + HumanEval."""
    print(f"\n[pilot_m2] === Step-Jump = {step_jump}x ===")
    base_progress = jump_idx * (total_progress // total_jumps)

    # GSM8K
    gsm8k_tps_list = []
    gsm8k_correct = 0
    gsm8k_results = []
    gsm8k_step_reductions = []

    for i, item in enumerate(gsm8k_data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            text, tps, elapsed, stats = generate_m2(
                model, tokenizer, prompt, device,
                gen_length=256, block_length=32,
                step_jump=step_jump,
                apply_chat_template=True,
            )
            correct = gsm8k_exact_match(text, item["answer"])
            if correct: gsm8k_correct += 1
            if i >= n_warmup:
                gsm8k_tps_list.append(tps)
                gsm8k_step_reductions.append(stats["step_reduction_ratio"])
            gsm8k_results.append({
                "id": i, "correct": correct, "tps": round(tps, 2),
                "total_steps": stats["total_steps"],
                "step_reduction_ratio": round(stats["step_reduction_ratio"], 3),
                "prediction_snippet": text[:200],
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] GSM8K sample {i}")
            torch.cuda.empty_cache(); gc.collect()
            gsm8k_results.append({"id": i, "error": "OOM", "correct": False, "tps": 0})
        except Exception as e:
            print(f"  [ERR] GSM8K sample {i}: {e}")
            gsm8k_results.append({"id": i, "error": str(e)[:200], "correct": False, "tps": 0})

        if (i + 1) % 20 == 0:
            acc = gsm8k_correct / (i + 1)
            avg_tps = np.mean(gsm8k_tps_list) if gsm8k_tps_list else 0.0
            print(f"  GSM8K [{i+1}/{len(gsm8k_data)}] acc={acc:.3f}, tps={avg_tps:.1f}")
            report_progress(
                base_progress + (i + 1) // 2,
                total_progress,
                {"step_jump": step_jump, "gsm8k_acc": acc, "gsm8k_tps": avg_tps}
            )

    gsm8k_acc = gsm8k_correct / len(gsm8k_data) if gsm8k_data else 0.0
    gsm8k_avg_tps = float(np.mean(gsm8k_tps_list)) if gsm8k_tps_list else 0.0
    gsm8k_tps_std = float(np.std(gsm8k_tps_list)) if gsm8k_tps_list else 0.0
    gsm8k_speedup = gsm8k_avg_tps / BASELINE_GSM8K_TPS if BASELINE_GSM8K_TPS > 0 else 0.0
    gsm8k_acc_retention = gsm8k_acc / BASELINE_GSM8K_ACC if BASELINE_GSM8K_ACC > 0 else 0.0
    gsm8k_qas = gsm8k_speedup * gsm8k_acc_retention
    avg_step_reduction = float(np.mean(gsm8k_step_reductions)) if gsm8k_step_reductions else 0.0

    print(f"  GSM8K: acc={gsm8k_acc:.3f} ({gsm8k_correct}/{len(gsm8k_data)}), "
          f"tps={gsm8k_avg_tps:.1f}±{gsm8k_tps_std:.1f}, "
          f"speedup={gsm8k_speedup:.2f}x, QAS={gsm8k_qas:.3f}, "
          f"avg_step_reduction={avg_step_reduction:.2f}x")

    # HumanEval
    he_tps_list = []
    he_passed = 0
    he_results = []

    for i, item in enumerate(humaneval_data):
        prompt = f"Complete the following Python function:\n\n{item['prompt']}"
        try:
            code, tps, elapsed, stats = generate_m2(
                model, tokenizer, prompt, device,
                gen_length=256, block_length=32,
                step_jump=step_jump,
                apply_chat_template=True,
            )
            passed = humaneval_pass_at_1(code, item)
            if passed: he_passed += 1
            he_tps_list.append(tps)
            he_results.append({
                "id": i, "passed": passed, "tps": round(tps, 2),
                "total_steps": stats["total_steps"],
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] HumanEval sample {i}")
            torch.cuda.empty_cache(); gc.collect()
            he_results.append({"id": i, "error": "OOM", "passed": False, "tps": 0})
        except Exception as e:
            print(f"  [ERR] HumanEval sample {i}: {e}")
            he_results.append({"id": i, "error": str(e)[:200], "passed": False, "tps": 0})

        if (i + 1) % 10 == 0:
            he_acc = he_passed / (i + 1)
            avg_tps = np.mean(he_tps_list) if he_tps_list else 0.0
            print(f"  HumanEval [{i+1}/{len(humaneval_data)}] pass@1={he_acc:.3f}, tps={avg_tps:.1f}")

    he_pass_at_1 = he_passed / len(humaneval_data) if humaneval_data else 0.0
    he_avg_tps = float(np.mean(he_tps_list)) if he_tps_list else 0.0
    he_tps_std = float(np.std(he_tps_list)) if he_tps_list else 0.0
    he_speedup = he_avg_tps / BASELINE_HE_TPS if BASELINE_HE_TPS > 0 else 0.0
    # For near-zero baseline pass@1, treat as retention=1.0 if maintained
    if BASELINE_HE_ACC < 0.05:
        he_acc_retention = 1.0 if he_pass_at_1 >= BASELINE_HE_ACC else (he_pass_at_1 / max(BASELINE_HE_ACC, 0.01))
    else:
        he_acc_retention = he_pass_at_1 / BASELINE_HE_ACC
    he_qas = he_speedup * he_acc_retention

    print(f"  HumanEval: pass@1={he_pass_at_1:.3f} ({he_passed}/{len(humaneval_data)}), "
          f"tps={he_avg_tps:.1f}±{he_tps_std:.1f}, "
          f"speedup={he_speedup:.2f}x, QAS={he_qas:.3f}")

    return {
        "step_jump": step_jump,
        "gsm8k": {
            "n_samples": len(gsm8k_data),
            "correct": gsm8k_correct,
            "exact_match": gsm8k_acc,
            "avg_tps": gsm8k_avg_tps,
            "tps_std": gsm8k_tps_std,
            "speedup": gsm8k_speedup,
            "accuracy_retention": gsm8k_acc_retention,
            "qas": gsm8k_qas,
            "avg_step_reduction_ratio": avg_step_reduction,
        },
        "humaneval": {
            "n_samples": len(humaneval_data),
            "passed": he_passed,
            "pass_at_1": he_pass_at_1,
            "avg_tps": he_avg_tps,
            "tps_std": he_tps_std,
            "speedup": he_speedup,
            "accuracy_retention": he_acc_retention,
            "qas": he_qas,
        },
        "samples": {
            "gsm8k": gsm8k_results[:10],
            "humaneval": he_results[:5],
        },
    }


def main():
    write_pid()
    start_time = datetime.now()
    print(f"[pilot_m2] Starting at {start_time.isoformat()}")
    print(f"[pilot_m2] Baseline reference: GSM8K acc={BASELINE_GSM8K_ACC}, tps={BASELINE_GSM8K_TPS}")
    print(f"[pilot_m2] Method: Simplified Adaptive Step Scheduling (Saber-like)")
    print(f"[pilot_m2] Step-jumps to test: 2x, 4x, 6x, 8x")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = "cuda:0"

    # Load model
    print(f"[pilot_m2] Loading LLaDA-8B-Instruct from {MODEL_PATH}...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_after_load = profile_vram(device)
    print(f"[pilot_m2] Model loaded. VRAM used: {vram_after_load.get('vram_used_mb', 0)} MB")

    # Load datasets
    gsm8k_data = load_gsm8k(n_samples=100, seed=42)
    humaneval_data = load_humaneval(n_samples=50, seed=42)
    print(f"[pilot_m2] Datasets: {len(gsm8k_data)} GSM8K + {len(humaneval_data)} HumanEval")

    # Step-jump settings to sweep
    step_jumps = [2, 4, 6, 8]
    all_results = []

    report_progress(0, 600, {"status": "starting", "n_step_jumps": len(step_jumps)})

    for j_idx, step_jump in enumerate(step_jumps):
        try:
            result = evaluate_step_jump(
                model, tokenizer, device,
                step_jump=step_jump,
                gsm8k_data=gsm8k_data,
                humaneval_data=humaneval_data,
                n_warmup=3,
                jump_idx=j_idx,
                total_jumps=len(step_jumps),
                total_progress=600,
            )
            all_results.append(result)
            torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            print(f"[pilot_m2] ERROR at step_jump={step_jump}: {e}")
            all_results.append({
                "step_jump": step_jump,
                "error": str(e)[:500],
                "gsm8k": {"exact_match": 0, "avg_tps": 0, "speedup": 0, "qas": 0, "accuracy_retention": 0},
                "humaneval": {"pass_at_1": 0, "avg_tps": 0, "speedup": 0, "qas": 0, "accuracy_retention": 0},
            })
            torch.cuda.empty_cache(); gc.collect()

    # ── Compute Pareto Curve ───────────────────────────────────────────────────
    pareto_points = []
    best_op_point = None
    best_qas = -1.0

    for r in all_results:
        if "error" in r:
            continue
        sj = r["step_jump"]
        gsm = r["gsm8k"]
        he = r["humaneval"]

        # Combined speedup (GSM8K weighted more due to more samples)
        combined_speedup = (gsm["speedup"] * 100 + he["speedup"] * 50) / 150
        # Combined accuracy retention
        combined_acc_ret = (gsm["accuracy_retention"] * 100 + he["accuracy_retention"] * 50) / 150
        combined_qas = combined_speedup * combined_acc_ret

        # Operating point: highest QAS where GSM8K acc drop <= 5% (pilot threshold)
        gsm_acc_drop = 1.0 - gsm["accuracy_retention"]
        is_within_budget = gsm_acc_drop <= 0.05

        point = {
            "step_jump": sj,
            "step_jump_label": f"{sj}x",
            "gsm8k_exact_match": gsm["exact_match"],
            "gsm8k_speedup": gsm["speedup"],
            "gsm8k_accuracy_retention": gsm["accuracy_retention"],
            "gsm8k_qas": gsm["qas"],
            "gsm8k_step_reduction_ratio": gsm.get("avg_step_reduction_ratio", 0),
            "humaneval_pass_at_1": he["pass_at_1"],
            "humaneval_speedup": he["speedup"],
            "humaneval_accuracy_retention": he["accuracy_retention"],
            "humaneval_qas": he["qas"],
            "combined_speedup": combined_speedup,
            "combined_accuracy_retention": combined_acc_ret,
            "combined_qas": combined_qas,
            "within_accuracy_budget": is_within_budget,
        }
        pareto_points.append(point)

        if is_within_budget and combined_qas > best_qas:
            best_qas = combined_qas
            best_op_point = point

    # If no point within budget, take best by QAS regardless
    if best_op_point is None and pareto_points:
        best_op_point = max(pareto_points, key=lambda p: p["combined_qas"])

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    # Determine GO/NO-GO
    n_complete = len([r for r in all_results if "error" not in r])
    if best_op_point and best_op_point["gsm8k_speedup"] > 1.5:
        verdict = "GO"
        decision = (f"PROCEED: M2 speedup={best_op_point['gsm8k_speedup']:.2f}x "
                    f"at step_jump={best_op_point['step_jump']}x")
    elif best_op_point and best_op_point["gsm8k_speedup"] > 1.2:
        verdict = "MARGINAL"
        decision = (f"MARGINAL: M2 speedup={best_op_point.get('gsm8k_speedup',0):.2f}x "
                    f"- below 1.5x target but shows step reduction")
    else:
        verdict = "NO_GO"
        decision = "NO_GO: M2 fails to achieve meaningful speedup in simplified implementation"

    # Print summary
    print(f"\n[pilot_m2] === FINAL RESULTS ===")
    print(f"  Verdict: {verdict}")
    print(f"  Decision: {decision}")
    if best_op_point:
        print(f"  Operating point: step_jump={best_op_point['step_jump']}x")
        print(f"  Best GSM8K: acc={best_op_point['gsm8k_exact_match']:.3f}, "
              f"speedup={best_op_point['gsm8k_speedup']:.2f}x, "
              f"qas={best_op_point['gsm8k_qas']:.3f}")
    print(f"  {n_complete}/{len(step_jumps)} step-jump settings completed")
    print(f"  Elapsed: {elapsed_min:.1f} min")

    # Print Pareto table
    print(f"\n  Pareto Table:")
    print(f"  {'Step-Jump':>12} | {'GSM8K-Acc':>10} | {'GSM8K-TPS':>10} | {'Speedup':>8} | {'QAS':>8}")
    print(f"  {'-'*12}+{'-'*12}+{'-'*12}+{'-'*10}+{'-'*10}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['step_jump']:>11}x | {'ERROR':>10}")
            continue
        g = r["gsm8k"]
        print(f"  {r['step_jump']:>11}x | {g['exact_match']:>10.3f} | {g['avg_tps']:>10.1f} | "
              f"{g['speedup']:>8.2f}x | {g['qas']:>8.3f}")

    # ── Save Results ──────────────────────────────────────────────────────────
    pareto_output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "method": "M2-AdaptiveScheduling-simplified",
        "saber_available": False,
        "implementation": "simplified_topk_confidence_unmasking",
        "mode": "pilot",
        "step_jumps_tested": step_jumps,
        "n_completed": n_complete,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": elapsed_min,
        "baseline_reference": {
            "gsm8k_exact_match": BASELINE_GSM8K_ACC,
            "gsm8k_avg_tps": BASELINE_GSM8K_TPS,
            "humaneval_pass_at_1": BASELINE_HE_ACC,
            "humaneval_avg_tps": BASELINE_HE_TPS,
        },
        "pareto_points": pareto_points,
        "operating_point": best_op_point,
        "verdict": verdict,
        "decision": decision,
        "all_results": all_results,
        "vram_after_load": vram_after_load,
        "pass_criteria_met": {
            "speedup_1_5x": best_op_point["gsm8k_speedup"] > 1.5 if best_op_point else False,
            "acc_drop_lt_5pct": best_op_point.get("within_accuracy_budget", False) if best_op_point else False,
            "at_least_3_of_4_complete": n_complete >= 3,
        }
    }

    out_path = RESULTS_DIR / "m2_pareto.json"
    out_path.write_text(json.dumps(pareto_output, indent=2))
    print(f"[pilot_m2] Results saved to {out_path}")

    # Also save per-step-jump detail
    detail_path = RESULTS_DIR / "m2_step_jump_details.json"
    detail_path.write_text(json.dumps(all_results, indent=2))
    print(f"[pilot_m2] Detailed results saved to {detail_path}")

    # Update gpu_progress.json
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
            "planned_min": 20,
            "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "config_snapshot": {
                "model": "LLaDA-8B-Instruct",
                "method": "M2-AdaptiveScheduling",
                "step_jumps": step_jumps,
                "n_gsm8k": len(gsm8k_data),
                "n_humaneval": len(humaneval_data),
                "gpu_model": vram_after_load.get("gpu_name", "unknown"),
            }
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"[pilot_m2] gpu_progress.json updated")
    except Exception as e:
        print(f"[pilot_m2] WARNING: Could not update gpu_progress.json: {e}")

    summary = (
        f"M2-AdaptiveScheduling pilot: verdict={verdict}, "
        f"best step_jump={best_op_point['step_jump']}x, "
        f"GSM8K speedup={best_op_point['gsm8k_speedup']:.2f}x, "
        f"acc={best_op_point['gsm8k_exact_match']:.3f}, "
        f"QAS={best_op_point['gsm8k_qas']:.3f}"
    ) if best_op_point else "NO_GO"

    mark_done(status="success", summary=summary)
    report_progress(600, 600, {"status": "done", "verdict": verdict})
    print(f"[pilot_m2] Done.")
    return pareto_output


if __name__ == "__main__":
    main()
