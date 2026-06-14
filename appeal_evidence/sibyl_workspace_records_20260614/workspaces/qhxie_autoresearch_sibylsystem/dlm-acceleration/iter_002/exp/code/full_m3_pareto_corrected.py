"""
Full M3 (AR-Guided Unmasking) Corrected Pareto Evaluation -- Iteration 2.

Task: m3_pareto_corrected (FULL mode)
Method: AR-Guided Unmasking using Qwen2.5-0.5B as supervisor.
Pilot showed GO verdict (all gw achieve AccRet > 0.95, speedup > 1.0x).

Changes from iter_001:
- QAS formula: Speedup * AccRet (no 0.5x penalty)
- Combined metric: 0.7*GSM8K + 0.3*MATH500 (HumanEval reported separately)
- Dropped gw=1.0 (diminishing returns from iter_001)
- Full scale: 1319 GSM8K + 500 MATH500 + 164 HumanEval
- 3 seeds: [42, 123, 456]
- Baseline data reused from iter_001 (llada_baseline_full.json)
"""
import os, sys, json, time, random, re, subprocess, gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

# Use the actual iter directory (NOT the `current` symlink) to prevent
# breakage when `current` is relinked mid-experiment.
WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/iter_002")

SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
QWEN_PATH   = str(SHARED / "checkpoints" / "qwen2.5-0.5b")
CODE_DIR    = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "m3_pareto"
TASK_ID     = "m3_pareto_corrected"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline from iter_001 full run (3 seeds, full benchmark scale)
BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500":   {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
}
acc_key = {"gsm8k": "exact_match", "math500": "exact_match", "humaneval": "pass_at_1"}

# Corrected: combined metric weights (0.7*GSM8K + 0.3*MATH500)
COMBINED_WEIGHTS = {"gsm8k": 0.7, "math500": 0.3}

SEEDS = [42, 123, 456]
GUIDANCE_WEIGHTS = [0.3, 0.5, 0.7]
N_SAMPLES = {"gsm8k": 1319, "math500": 500, "humaneval": 164}
MASK_ID = 126336
WARMUP_SAMPLES = 5
_LLADA_TOKENIZER = None


# -- Helpers ------------------------------------------------------------------
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
    progress_f = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_f.exists():
        try: final_progress = json.loads(progress_f.read_text())
        except: pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# -- Data ---------------------------------------------------------------------
def load_dataset(name, seed, n):
    paths = {
        "gsm8k": SHARED / "datasets" / "gsm8k" / "test.json",
        "math500": SHARED / "datasets" / "math500" / "test.json",
        "humaneval": SHARED / "datasets" / "humaneval" / "test.json",
    }
    with open(paths[name]) as f:
        data = json.load(f)
    if n >= len(data):
        return data  # Use full dataset
    return random.Random(seed).sample(data, min(n, len(data)))


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

def extract_answer(text):
    m = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None

def gsm8k_match(pred, gold):
    p, g = extract_answer(pred), extract_answer(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except: return p.strip() == g.strip()

def math500_match(pred, gold):
    def extract_math(t):
        m = re.search(r"\\boxed\{([^}]+)\}", t)
        if m: return m.group(1).strip()
        return extract_answer(t)
    p, g = extract_math(pred), extract_math(str(gold))
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-3
    except: return p.strip() == g.strip()

def he_pass_at_1(code, problem):
    full = problem["prompt"] + code + "\n" + problem["test"]
    full += f"\ncheck({problem['entry_point']})\n"
    try:
        r = subprocess.run(["python", "-c", full], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except: return False


# -- Qwen Score ---------------------------------------------------------------
def get_qwen_scores(qwen_model, qwen_tokenizer, current_ids, llada_x0, device, max_ctx=512):
    batch, seq_len = current_ids.shape
    qwen_scores = torch.zeros(batch, seq_len, device=device)
    global _LLADA_TOKENIZER

    for b in range(batch):
        ids = current_ids[b]
        mask_positions = (ids == MASK_ID).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            continue
        pad_tok = qwen_tokenizer.pad_token_id or qwen_tokenizer.eos_token_id
        unmasked = ids.clone()
        unmasked[ids == MASK_ID] = pad_tok
        seq = unmasked.cpu().tolist()
        offset = max(0, len(seq) - max_ctx)
        seq = seq[offset:]
        try:
            text = _LLADA_TOKENIZER.decode(seq, skip_special_tokens=True)
            qenc = qwen_tokenizer(text, return_tensors="pt", max_length=max_ctx,
                                  truncation=True, add_special_tokens=False)
            qinput = qenc["input_ids"].to(device)
            with torch.no_grad():
                qout = qwen_model(qinput)
            ql = qout.logits
            for pos in mask_positions:
                pi = pos.item()
                if pi < offset: continue
                qp = pi - offset
                if qp >= ql.shape[1]: continue
                ltok = llada_x0[b, pi].item()
                if ltok == MASK_ID: continue
                tok_txt = _LLADA_TOKENIZER.decode([ltok], skip_special_tokens=True)
                if not tok_txt.strip(): continue
                qtok_ids = qwen_tokenizer.encode(tok_txt, add_special_tokens=False)
                if not qtok_ids: continue
                qtok = qtok_ids[0]
                if qtok >= ql.shape[-1]: continue
                qwen_scores[b, pi] = float(F.softmax(ql[0, qp, :], dim=-1)[qtok].item())
        except Exception:
            pass
    return qwen_scores


# -- M3 Generation ------------------------------------------------------------
@torch.no_grad()
def generate_m3(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                prompt, device, gen_length=256, block_length=32,
                guidance_weight=0.5, steps=64):
    from generate import get_num_transfer_tokens

    msg = [{"role": "user", "content": prompt}]
    prompt_text = llada_tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = llada_tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks

    t0 = time.perf_counter()

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = prompt_len + (block_idx + 1) * block_length
        block_mask  = x[:, block_start:block_end] == MASK_ID
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            mask_index = x == MASK_ID
            if not mask_index[:, block_start:block_end].any():
                continue

            logits = llada_model(x, attention_mask=attn).logits
            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)
            x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

            if guidance_weight > 0:
                qwen_scores = get_qwen_scores(qwen_model, qwen_tokenizer, x, x0, device)
                blended = (1 - guidance_weight) * x0_p + guidance_weight * qwen_scores
            else:
                blended = x0_p

            x0_p[:, block_end:] = -float('inf')
            blended[:, block_end:] = -float('inf')
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, blended,
                                     torch.tensor(-float('inf'), device=device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(batch):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(confidence[j], k=int(k))
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    elapsed = time.perf_counter() - t0
    text = llada_tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    return text, tps


# -- Benchmark evaluators ------------------------------------------------------
def eval_bm(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
            device, gw, bm, data, warmup=5):
    """Evaluate a single benchmark. Returns metrics dict."""
    tps_list, n_correct, n_passed = [], 0, 0

    for i, item in enumerate(data):
        if bm == "gsm8k":
            prompt = GSM8K_8SHOT + f"Question: {item['question']}\nAnswer:"
        elif bm == "math500":
            prompt = f"Solve this math problem:\n{item['problem']}\nAnswer:"
        elif bm == "humaneval":
            prompt = f"Complete the following Python function:\n\n{item['prompt']}"

        try:
            text, tps = generate_m3(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                                    prompt, device, guidance_weight=gw)
            if bm == "gsm8k":
                if gsm8k_match(text, item["answer"]): n_correct += 1
            elif bm == "math500":
                if math500_match(text, item.get("answer", item.get("solution", ""))): n_correct += 1
            elif bm == "humaneval":
                if he_pass_at_1(text, item): n_passed += 1

            # Skip warmup samples for TPS measurement
            if i >= warmup:
                tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] {bm}[{i}]: {e}")

        if (i+1) % 100 == 0:
            cur_acc = n_correct / (i+1) if bm in ["gsm8k", "math500"] else n_passed / (i+1)
            avg_tps_so_far = float(np.mean(tps_list)) if tps_list else 0.0
            print(f"  {bm} [{i+1}/{len(data)}] acc={cur_acc:.4f} tps={avg_tps_so_far:.2f}")

    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if len(tps_list) > 1 else 0.0

    if bm in ["gsm8k", "math500"]:
        acc = n_correct / len(data) if data else 0.0
        return {acc_key[bm]: acc, "avg_tps": avg_tps, "tps_std": tps_std,
                "n": len(data), "n_correct": n_correct}
    else:
        p1 = n_passed / len(data) if data else 0.0
        return {acc_key[bm]: p1, "avg_tps": avg_tps, "tps_std": tps_std,
                "n": len(data), "n_passed": n_passed}


# -- Save qualitative samples -------------------------------------------------
def save_qualitative_samples(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                              device, gw, data, n=10):
    """Generate and save qualitative samples for inspection."""
    samples = []
    for i, item in enumerate(data[:n]):
        prompt = GSM8K_8SHOT + f"Question: {item['question']}\nAnswer:"
        try:
            text, tps = generate_m3(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                                    prompt, device, guidance_weight=gw)
            gold_answer = extract_answer(item["answer"])
            pred_answer = extract_answer(text)
            samples.append({
                "question": item["question"],
                "gold_answer": gold_answer,
                "pred_answer": pred_answer,
                "correct": gsm8k_match(text, item["answer"]),
                "generated_text": text[:500],
                "tps": tps,
            })
        except Exception as e:
            samples.append({"question": item["question"], "error": str(e)})
    return samples


# -- Main ---------------------------------------------------------------------
def main():
    global _LLADA_TOKENIZER
    write_pid()
    start_time = datetime.now()
    device_id = os.environ.get("CUDA_VISIBLE_DEVICES", "4")
    device = "cuda:0"  # After CUDA_VISIBLE_DEVICES, device 0 is our target
    print(f"[m3_full] Starting FULL M3 Pareto (corrected) at {start_time.isoformat()}")
    print(f"[m3_full] CUDA_VISIBLE_DEVICES={device_id}, device={device}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # -- Load models --
    print("[m3_full] Loading LLaDA-8B-Instruct...")
    from transformers import AutoTokenizer, AutoModel
    llada_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llada_tokenizer.padding_side = "left"
    _LLADA_TOKENIZER = llada_tokenizer
    llada_model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    print("[m3_full] Loading Qwen2.5-0.5B...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    from transformers import AutoModelForCausalLM
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_used = torch.cuda.memory_allocated(device) // 1024**2
    vram_total = torch.cuda.get_device_properties(device).total_memory // 1024**2
    print(f"[m3_full] Both models loaded. VRAM: {vram_used} / {vram_total} MB")

    # Write GPU profile
    gpu_profile = {
        "gpu_name": torch.cuda.get_device_name(device),
        "vram_total_mb": vram_total,
        "vram_used_mb": vram_used,
        "utilization_pct": round(100.0 * vram_used / max(vram_total, 1), 1),
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2))

    # -- Pre-load datasets --
    print("[m3_full] Loading datasets...")
    datasets = {}
    for seed in SEEDS:
        datasets[seed] = {}
        for bm in ["gsm8k", "math500", "humaneval"]:
            datasets[seed][bm] = load_dataset(bm, seed, N_SAMPLES[bm])
        print(f"  seed={seed}: gsm8k={len(datasets[seed]['gsm8k'])}, "
              f"math500={len(datasets[seed]['math500'])}, humaneval={len(datasets[seed]['humaneval'])}")

    # -- Run evaluations --
    all_results = {}
    total_steps = len(GUIDANCE_WEIGHTS) * len(SEEDS) * 3  # 3 benchmarks per (gw, seed)
    prog_step = 0

    for gw in GUIDANCE_WEIGHTS:
        all_results[gw] = {}
        print(f"\n{'='*60}")
        print(f"[m3_full] === guidance_weight={gw} ===")
        print(f"{'='*60}")

        for seed in SEEDS:
            print(f"\n[m3_full] seed={seed}")
            # Set seed for reproducibility within each run
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            all_results[gw][seed] = {}
            for bm in ["gsm8k", "math500", "humaneval"]:
                bm_start = time.time()
                print(f"\n  [m3_full] Running {bm} (n={len(datasets[seed][bm])})...")
                r = eval_bm(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                            device, gw, bm, datasets[seed][bm], warmup=WARMUP_SAMPLES)
                all_results[gw][seed][bm] = r
                bm_elapsed = time.time() - bm_start

                # Compute metrics
                bl_tps = BASELINE[bm]["avg_tps"]
                bl_acc = BASELINE[bm].get(acc_key[bm], 0)
                cur_acc = r.get(acc_key[bm], 0)
                speedup = r["avg_tps"] / max(bl_tps, 1e-6)
                ret = cur_acc / max(bl_acc, 1e-6) if bl_acc > 0 else 1.0

                print(f"    {bm}: acc={cur_acc:.4f}, speedup={speedup:.3f}x, "
                      f"ret={ret:.3f}, tps={r['avg_tps']:.2f}, elapsed={bm_elapsed/60:.1f}min")

                prog_step += 1
                report_progress(prog_step, total_steps, {
                    "gw": gw, "seed": seed, "benchmark": bm,
                    "accuracy": cur_acc, "speedup": speedup, "acc_ret": ret,
                })
                torch.cuda.empty_cache(); gc.collect()

    # -- Save qualitative samples (gw=0.3, seed=42, GSM8K first 10) --
    print("\n[m3_full] Generating qualitative samples...")
    qualitative = save_qualitative_samples(
        llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
        device, 0.3, datasets[42]["gsm8k"], n=10
    )
    (RESULTS_DIR / "m3_qualitative_samples.json").write_text(json.dumps(qualitative, indent=2))

    # -- Aggregate and compute Pareto --
    print("\n[m3_full] Aggregating results...")
    pareto_points = []
    best_op_point, best_qas = None, -1.0

    for gw in GUIDANCE_WEIGHTS:
        # Per-benchmark per-seed breakdown
        per_bm = {}
        for bm in ["gsm8k", "math500", "humaneval"]:
            tps_list, speedup_list, ret_list, acc_list = [], [], [], []
            for seed in SEEDS:
                r = all_results[gw].get(seed, {}).get(bm, {})
                if not r: continue
                bl_tps = BASELINE[bm]["avg_tps"]
                bl_acc = BASELINE[bm].get(acc_key[bm], 0)
                cur_acc = r.get(acc_key[bm], 0)
                tps_list.append(r["avg_tps"])
                speedup_list.append(r["avg_tps"] / max(bl_tps, 1e-6))
                ret_list.append(cur_acc / max(bl_acc, 1e-6) if bl_acc > 0 else 1.0)
                acc_list.append(cur_acc)
            if tps_list:
                per_bm[bm] = {
                    "avg_tps": float(np.mean(tps_list)),
                    "tps_std": float(np.std(tps_list)),
                    "avg_speedup": float(np.mean(speedup_list)),
                    "speedup_std": float(np.std(speedup_list)),
                    "avg_acc": float(np.mean(acc_list)),
                    "acc_std": float(np.std(acc_list)),
                    "avg_acc_ret": float(np.mean(ret_list)),
                    "acc_ret_std": float(np.std(ret_list)),
                    "per_seed_acc": acc_list,
                    "per_seed_speedup": speedup_list,
                    "per_seed_acc_ret": ret_list,
                }

        if not per_bm: continue

        # Corrected combined metric: 0.7*GSM8K + 0.3*MATH500
        combined_speedup, combined_acc_ret = 0.0, 0.0
        for bm, w in COMBINED_WEIGHTS.items():
            if bm in per_bm:
                combined_speedup += w * per_bm[bm]["avg_speedup"]
                combined_acc_ret += w * per_bm[bm]["avg_acc_ret"]

        gsm8k_acc_ret = per_bm.get("gsm8k", {}).get("avg_acc_ret", 0.0)
        gsm8k_drop = 1.0 - gsm8k_acc_ret

        # Corrected QAS: no penalty
        combined_qas = float(combined_speedup * combined_acc_ret)

        point = {
            "guidance_weight": float(gw),
            "combined_speedup": float(combined_speedup),
            "combined_acc_ret": float(combined_acc_ret),
            "combined_qas": float(combined_qas),
            "gsm8k_acc_drop": float(gsm8k_drop),
            "gsm8k_acc_ret": float(gsm8k_acc_ret),
            "within_5pct": bool(gsm8k_drop <= 0.05),
            "per_benchmark": per_bm,
        }
        pareto_points.append(point)

        print(f"  gw={gw}: combined_speedup={combined_speedup:.3f}x, "
              f"combined_acc_ret={combined_acc_ret:.3f}, QAS={combined_qas:.3f}, "
              f"gsm8k_acc_ret={gsm8k_acc_ret:.3f}")

        if combined_qas > best_qas:
            best_qas = combined_qas
            best_op_point = point

    # Determine verdict
    any_within_5pct = any(p["within_5pct"] for p in pareto_points)
    any_speedup_above_1 = any(p["combined_speedup"] > 1.0 for p in pareto_points)
    verdict = "GO" if (any_within_5pct and any_speedup_above_1) else (
        "MARGINAL" if any_speedup_above_1 else "NO_GO")

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    print(f"\n{'='*60}")
    print(f"[m3_full] Verdict: {verdict}, elapsed: {elapsed_min:.1f} min")
    if best_op_point:
        print(f"[m3_full] Best: gw={best_op_point['guidance_weight']}, "
              f"speedup={best_op_point['combined_speedup']:.3f}x, "
              f"acc_ret={best_op_point['combined_acc_ret']:.3f}, "
              f"QAS={best_op_point['combined_qas']:.3f}")
    print(f"{'='*60}")

    # -- Construct output --
    output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "guide_model": "Qwen2.5-0.5B",
        "method": "M3-ARGuidedUnmasking",
        "mode": "full",
        "iteration": 2,
        "corrections": [
            "QAS formula: Speedup * AccRet (no 0.5x penalty)",
            "Combined metric: 0.7*GSM8K + 0.3*MATH500",
            "Dropped gw=1.0 (diminishing returns)",
            "HumanEval reported separately, not in combined metric",
        ],
        "seeds": SEEDS,
        "guidance_weights": GUIDANCE_WEIGHTS,
        "n_samples": N_SAMPLES,
        "warmup_samples": WARMUP_SAMPLES,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": round(elapsed_min, 2),
        "verdict": verdict,
        "pareto_points": pareto_points,
        "best_operating_point": best_op_point,
        "baseline": BASELINE,
        "all_results": {
            str(gw): {
                str(s): bm_data for s, bm_data in seed_data.items()
            } for gw, seed_data in all_results.items()
        },
        "gpu_profile": gpu_profile,
    }

    out_path = RESULTS_DIR / "m3_pareto_corrected_full.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[m3_full] Results -> {out_path}")

    # -- Update gpu_progress --
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
            "planned_min": 60,
            "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "config_snapshot": {
                "model": "LLaDA-8B-Instruct",
                "guide_model": "Qwen2.5-0.5B",
                "guidance_weights": GUIDANCE_WEIGHTS,
                "seeds": SEEDS,
                "n_gsm8k": N_SAMPLES["gsm8k"],
                "n_math500": N_SAMPLES["math500"],
                "n_humaneval": N_SAMPLES["humaneval"],
                "gpu_model": gpu_profile.get("gpu_name", "unknown"),
                "gpu_count": 1,
            },
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print("[m3_full] gpu_progress.json updated")
    except Exception as e:
        print(f"WARNING: gpu_progress update failed: {e}")

    # -- Mark done --
    summary = (f"M3 full corrected: verdict={verdict}, "
               f"best_gw={best_op_point['guidance_weight']}, "
               f"combined_speedup={best_op_point['combined_speedup']:.3f}x, "
               f"combined_acc_ret={best_op_point['combined_acc_ret']:.3f}, "
               f"QAS={best_op_point['combined_qas']:.3f}") if best_op_point else f"verdict={verdict}"
    mark_done(status="success", summary=summary)
    report_progress(total_steps, total_steps, {"status": "done", "verdict": verdict})
    print("[m3_full] Done.")
    return output


if __name__ == "__main__":
    main()
