"""
Full M2 (Adaptive Step Scheduling) Pareto Evaluation.

Task: full_m2_pareto
Method: Simplified top-k confidence unmasking (Saber-like)
Sweep: step_jump in {2, 4, 6, 8} across 4 benchmarks, 2 seeds (42, 123).
Note: Pilot showed M2 severely degrades accuracy. This full run confirms results
      with all 4 benchmarks for complete paper comparison.

Faster variant: 200 GSM8K, 100 math500, 164 humaneval, 100 mbpp per seed.
Expected time: ~25 min on single GPU.
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
CODE_DIR      = WORKSPACE / "exp" / "code"
RESULTS_DIR   = WORKSPACE / "exp" / "results" / "full_m2"
TASK_ID       = "full_m2_pareto"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Full baseline (from full_baseline experiment)
BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500":   {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
    "mbpp":      {"pass_at_1":   0.0000, "avg_tps": 191.586},
}
acc_key = {"gsm8k": "exact_match", "math500": "exact_match",
           "humaneval": "pass_at_1", "mbpp": "pass_at_1"}
weights = {"gsm8k": 1319, "math500": 500, "humaneval": 164, "mbpp": 374}

SEEDS = [42, 123]   # 2 seeds sufficient for M2 (pilot showed highly consistent results)
STEP_JUMPS = [2, 4, 6, 8]

# Sample sizes (reduced for speed since M2 clearly underperforms)
N_SAMPLES = {
    "gsm8k": 200,
    "math500": 100,
    "humaneval": 164,   # all samples (small enough)
    "mbpp": 100,
}

# ── System-monitor Helpers ────────────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, metric=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "step": step, "total_steps": total,
        "updated_at": datetime.now().isoformat(), "metric": metric or {},
    }))

def mark_done(status="success", summary=""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists(): pid_f.unlink()
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))

# ── Data Loaders ──────────────────────────────────────────────────────────────
def load_dataset(name, seed, n):
    ds_dirs = {
        "gsm8k": SHARED / "datasets" / "gsm8k" / "test.json",
        "math500": SHARED / "datasets" / "math500" / "test.json",
        "humaneval": SHARED / "datasets" / "humaneval" / "test.json",
        "mbpp": SHARED / "datasets" / "mbpp" / "test.json",
    }
    path = ds_dirs[name]
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n, len(data)))

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

def extract_answer(text):
    match = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None

def gsm8k_exact_match(pred, gold):
    p = extract_answer(pred); g = extract_answer(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except: return p.strip() == g.strip()

def math500_exact_match(pred, gold):
    # Look for boxed answer or last number
    def extract_math(text):
        m = re.search(r"\\boxed\{([^}]+)\}", text)
        if m: return m.group(1).strip()
        return extract_answer(text)
    p = extract_math(pred); g = extract_math(str(gold))
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-3
    except: return p.strip() == g.strip()

def humaneval_pass_at_1(code, problem):
    full_code = problem["prompt"] + code + "\n" + problem["test"]
    full_code += f"\ncheck({problem['entry_point']})\n"
    try:
        r = subprocess.run(["python", "-c", full_code],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except: return False

def mbpp_pass_at_1(code, problem):
    test_code = code + "\n" + "\n".join(problem.get("test_list", []))
    try:
        r = subprocess.run(["python", "-c", test_code],
                           capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except: return False

# ── M2 Adaptive Step Scheduling ───────────────────────────────────────────────
@torch.no_grad()
def generate_m2(model, tokenizer, prompt_text, device,
                gen_length=256, block_length=32, step_jump=2,
                apply_chat_template=True, mask_id=126336):
    from generate import get_num_transfer_tokens

    if apply_chat_template:
        msg = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False,
                    padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch, prompt_len + gen_length), mask_id, dtype=torch.long).to(device)
    x[:, :prompt_len] = input_ids.clone()
    attn = torch.cat([attention_mask,
                      torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    num_blocks = gen_length // block_length
    base_steps_per_block = 8  # 64 total / 8 blocks
    steps_per_block = max(1, base_steps_per_block // step_jump)

    t0 = time.perf_counter()

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = prompt_len + (block_idx + 1) * block_length
        block_mask  = x[:, block_start:block_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            mask_index = x == mask_id
            if not mask_index[:, block_start:block_end].any():
                continue
            logits = model(x, attention_mask=attn).logits
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
            x0_p[:, block_end:] = -float('inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p,
                                     torch.tensor(-float('inf'), device=device))
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(batch):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    elapsed = time.perf_counter() - t0
    generated_ids = x[:, prompt_len:]
    tps = generated_ids.numel() / elapsed if elapsed > 0 else 0.0
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    total_steps = steps_per_block * num_blocks
    step_reduction = (base_steps_per_block * num_blocks) / max(total_steps, 1)
    return text, tps, {"step_reduction_ratio": step_reduction, "total_steps": total_steps}


# ── Per-benchmark evaluators ──────────────────────────────────────────────────
def eval_gsm8k(model, tokenizer, device, step_jump, data):
    tps_list, correct = [], 0
    for i, item in enumerate(data):
        try:
            text, tps, stats = generate_m2(model, tokenizer,
                build_gsm8k_prompt(item["question"]), device, step_jump=step_jump)
            if gsm8k_exact_match(text, item["answer"]): correct += 1
            if i >= 3: tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] gsm8k[{i}]: {e}")
        if (i+1) % 50 == 0:
            print(f"  gsm8k [{i+1}/{len(data)}] acc={correct/(i+1):.3f}")
    acc = correct / len(data) if data else 0.0
    return {"exact_match": acc, "avg_tps": float(np.mean(tps_list)) if tps_list else 0.0,
            "n": len(data), "correct": correct}

def eval_math500(model, tokenizer, device, step_jump, data):
    tps_list, correct = [], 0
    for i, item in enumerate(data):
        try:
            text, tps, _ = generate_m2(model, tokenizer,
                f"Solve this math problem step by step:\n{item['problem']}\nAnswer:", device,
                step_jump=step_jump)
            if math500_exact_match(text, item.get("answer", item.get("solution", ""))): correct += 1
            if i >= 3: tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] math500[{i}]: {e}")
        if (i+1) % 30 == 0:
            print(f"  math500 [{i+1}/{len(data)}] acc={correct/(i+1):.3f}")
    acc = correct / len(data) if data else 0.0
    return {"exact_match": acc, "avg_tps": float(np.mean(tps_list)) if tps_list else 0.0,
            "n": len(data), "correct": correct}

def eval_humaneval(model, tokenizer, device, step_jump, data):
    tps_list, passed = [], 0
    for i, item in enumerate(data):
        try:
            code, tps, _ = generate_m2(model, tokenizer,
                f"Complete the following Python function:\n\n{item['prompt']}", device,
                step_jump=step_jump)
            if humaneval_pass_at_1(code, item): passed += 1
            tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] humaneval[{i}]: {e}")
    p1 = passed / len(data) if data else 0.0
    return {"pass_at_1": p1, "avg_tps": float(np.mean(tps_list)) if tps_list else 0.0,
            "n": len(data), "passed": passed}

def eval_mbpp(model, tokenizer, device, step_jump, data):
    tps_list, passed = [], 0
    for i, item in enumerate(data):
        try:
            code, tps, _ = generate_m2(model, tokenizer,
                f"Write a Python function to solve:\n{item.get('text', item.get('prompt', ''))}",
                device, step_jump=step_jump)
            if mbpp_pass_at_1(code, item): passed += 1
            tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] mbpp[{i}]: {e}")
        if (i+1) % 30 == 0:
            print(f"  mbpp [{i+1}/{len(data)}]")
    p1 = passed / len(data) if data else 0.0
    return {"pass_at_1": p1, "avg_tps": float(np.mean(tps_list)) if tps_list else 0.0,
            "n": len(data), "passed": passed}


def main():
    write_pid()
    start_time = datetime.now()
    print(f"[full_m2] Starting at {start_time.isoformat()}")
    print(f"[full_m2] Seeds: {SEEDS}, Step-jumps: {STEP_JUMPS}")
    print(f"[full_m2] Sample sizes: {N_SAMPLES}")

    device = os.environ.get("CUDA_DEVICE", "cuda:0")
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    print(f"[full_m2] Loading model on {device}...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"[full_m2] Model loaded. VRAM: {torch.cuda.memory_allocated(device)//1024**2} MB")

    # Pre-load datasets for all seeds
    datasets = {}
    for seed in SEEDS:
        datasets[seed] = {}
        for bm in ["gsm8k", "math500", "humaneval", "mbpp"]:
            datasets[seed][bm] = load_dataset(bm, seed, N_SAMPLES[bm])
            print(f"  seed={seed} {bm}: {len(datasets[seed][bm])} samples")

    # all_results[step_jump][seed][benchmark] = metrics
    all_results = {}
    total_steps = len(STEP_JUMPS) * len(SEEDS)
    step_count = 0

    for sj_idx, step_jump in enumerate(STEP_JUMPS):
        all_results[step_jump] = {}
        print(f"\n[full_m2] === Step-Jump = {step_jump}x ({sj_idx+1}/{len(STEP_JUMPS)}) ===")

        for seed in SEEDS:
            print(f"[full_m2] seed={seed}")
            all_results[step_jump][seed] = {}
            bm_results = {}

            for bm in ["gsm8k", "math500", "humaneval", "mbpp"]:
                data = datasets[seed][bm]
                print(f"  Evaluating {bm}...")
                if bm == "gsm8k":
                    r = eval_gsm8k(model, tokenizer, device, step_jump, data)
                elif bm == "math500":
                    r = eval_math500(model, tokenizer, device, step_jump, data)
                elif bm == "humaneval":
                    r = eval_humaneval(model, tokenizer, device, step_jump, data)
                elif bm == "mbpp":
                    r = eval_mbpp(model, tokenizer, device, step_jump, data)
                bm_results[bm] = r
                bl_tps = BASELINE[bm]["avg_tps"]
                bl_acc = BASELINE[bm].get(acc_key[bm], 0)
                cur_acc = r.get(acc_key[bm], 0)
                cur_tps = r["avg_tps"]
                speedup = cur_tps / max(bl_tps, 1e-6)
                ret = cur_acc / max(bl_acc, 1e-6) if bl_acc > 0 else 1.0
                print(f"    {bm}: acc={cur_acc:.3f}, tps={cur_tps:.1f}, speedup={speedup:.2f}x, ret={ret:.3f}")

            all_results[step_jump][seed] = bm_results
            step_count += 1
            report_progress(step_count, total_steps,
                           {"step_jump": step_jump, "seed": seed})
            torch.cuda.empty_cache(); gc.collect()

    # ── Aggregate & Pareto ────────────────────────────────────────────────────
    pareto_points = []
    best_op_point, best_qas = None, -1.0

    for step_jump in STEP_JUMPS:
        agg = {}
        for bm in ["gsm8k", "math500", "humaneval", "mbpp"]:
            tps_list, acc_list, speedup_list, ret_list = [], [], [], []
            for seed in SEEDS:
                r = all_results[step_jump].get(seed, {}).get(bm, {})
                if not r: continue
                bl_tps = BASELINE[bm]["avg_tps"]
                bl_acc = BASELINE[bm].get(acc_key[bm], 0)
                cur_acc = r.get(acc_key[bm], 0)
                cur_tps = r["avg_tps"]
                tps_list.append(cur_tps)
                acc_list.append(cur_acc)
                speedup_list.append(cur_tps / max(bl_tps, 1e-6))
                ret_list.append(cur_acc / max(bl_acc, 1e-6) if bl_acc > 0 else 1.0)
            if tps_list:
                agg[bm] = {
                    "avg_tps": float(np.mean(tps_list)),
                    "avg_acc": float(np.mean(acc_list)),
                    "avg_speedup": float(np.mean(speedup_list)),
                    "avg_acc_ret": float(np.mean(ret_list)),
                    "n_seeds": len(tps_list),
                }

        if not agg: continue
        total_w = sum(weights[b] for b in agg)
        combined_speedup = sum(weights[b] * agg[b]["avg_speedup"] for b in agg) / total_w
        combined_acc_ret = sum(weights[b] * agg[b]["avg_acc_ret"] for b in agg) / total_w
        gsm8k_acc_drop = 1.0 - agg.get("gsm8k", {}).get("avg_acc_ret", 1.0)
        within_5pct = bool(gsm8k_acc_drop <= 0.05)
        combined_qas = float(combined_speedup * combined_acc_ret) if within_5pct else float(combined_speedup * combined_acc_ret * 0.5)

        point = {
            "step_jump": step_jump,
            "step_jump_label": f"{step_jump}x",
            "combined_speedup": float(combined_speedup),
            "combined_acc_ret": float(combined_acc_ret),
            "gsm8k_acc_drop": float(gsm8k_acc_drop),
            "within_5pct": within_5pct,
            "combined_qas": float(combined_qas),
            "per_benchmark": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in agg.items()},
        }
        pareto_points.append(point)
        print(f"  step_jump={step_jump}x: speedup={combined_speedup:.3f}x, "
              f"acc_ret={combined_acc_ret:.3f}, QAS={combined_qas:.3f}, within_5pct={within_5pct}")

        if within_5pct and combined_qas > best_qas:
            best_qas = combined_qas
            best_op_point = point

    if best_op_point is None and pareto_points:
        best_op_point = max(pareto_points, key=lambda p: p["combined_qas"])

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    # Verdict
    if best_op_point and best_op_point["within_5pct"]:
        verdict = "GO"
    elif best_op_point and best_op_point["combined_acc_ret"] > 0.8:
        verdict = "MARGINAL"
    else:
        verdict = "NO_GO"

    print(f"\n[full_m2] Verdict: {verdict}")
    print(f"[full_m2] Best operating point: {best_op_point}")
    print(f"[full_m2] Elapsed: {elapsed_min:.1f} min")

    output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "method": "M2-AdaptiveScheduling-simplified",
        "mode": "full",
        "seeds": SEEDS,
        "step_jumps": STEP_JUMPS,
        "n_samples": N_SAMPLES,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": round(elapsed_min, 2),
        "verdict": verdict,
        "pareto_points": pareto_points,
        "best_operating_point": best_op_point,
        "baseline": BASELINE,
        "all_results": {
            str(sj): {
                str(seed): bm_data
                for seed, bm_data in seed_data.items()
            }
            for sj, seed_data in all_results.items()
        },
    }

    out_path = RESULTS_DIR / "m2_pareto_full.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[full_m2] Results → {out_path}")

    # Update gpu_progress.json
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {"completed": [], "failed": [], "running": {}, "timings": {}}
        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 90, "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
        }
        gp_path.write_text(json.dumps(gp, indent=2))
    except Exception as e:
        print(f"[full_m2] WARNING: gpu_progress update failed: {e}")

    summary = (f"M2 full: verdict={verdict}, best_step_jump={best_op_point['step_jump']}x, "
               f"speedup={best_op_point['combined_speedup']:.3f}x, acc_ret={best_op_point['combined_acc_ret']:.3f}, "
               f"QAS={best_op_point['combined_qas']:.3f}") if best_op_point else "NO_GO"
    mark_done(status="success", summary=summary)
    report_progress(total_steps, total_steps, {"status": "done", "verdict": verdict})
    print("[full_m2] Done.")
    return output


if __name__ == "__main__":
    main()
