"""
M3 (AR-Guided Unmasking) Corrected Pareto Curve -- PILOT MODE

Task: m3_pareto_corrected
Method: AR-Guided Unmasking using Qwen2.5-0.5B as guide model.

Iteration 2 corrections:
- Guidance weights: {0.3, 0.5, 0.7} (drop gw=1.0 per iter_001 diminishing returns)
- Combined metric: 0.7*GSM8K + 0.3*MATH500 (drop HumanEval/MBPP from combined)
- QAS formula: Speedup * AccRet (NO 0.5x penalty)
- HumanEval reported separately (not in combined metric)
- Reuse iter_001 baseline data as reference

Pilot config: 200 GSM8K + 100 MATH500 + 50 HumanEval, seed=42
Pass criteria: At least one gw achieves accuracy_retention >= 0.95 on GSM8K AND speedup > 1.0x
"""
import os, sys, json, time, random, re, subprocess, gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
QWEN_PATH   = str(SHARED / "checkpoints" / "qwen2.5-0.5b")
CODE_DIR    = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "m3_pareto"
TASK_ID     = "m3_pareto_corrected"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Baseline from iter_001 (full-scale, 3-seed) ─────────────────────────────
BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500":   {"exact_match": 0.1107, "avg_tps": 79.221},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
}
acc_key = {"gsm8k": "exact_match", "math500": "exact_match",
           "humaneval": "pass_at_1"}

# ── Iter_002 config ─────────────────────────────────────────────────────────
SEEDS = [42]  # Pilot: single seed
GUIDANCE_WEIGHTS = [0.3, 0.5, 0.7]  # Drop gw=1.0
N_SAMPLES = {"gsm8k": 200, "math500": 100, "humaneval": 50}
WARMUP_SAMPLES = 5  # Discard first 5 for TPS measurement
MASK_ID = 126336
_LLADA_TOKENIZER = None

# Combined metric weights (iter_002: 0.7*GSM8K + 0.3*MATH500)
COMBINED_WEIGHTS = {"gsm8k": 0.7, "math500": 0.3}


# ── Helpers ────────────────────────────────────────────────────────────────────
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


# ── Data ───────────────────────────────────────────────────────────────────────
def load_dataset(name, seed, n):
    paths = {
        "gsm8k": SHARED / "datasets" / "gsm8k" / "test.json",
        "math500": SHARED / "datasets" / "math500" / "test.json",
        "humaneval": SHARED / "datasets" / "humaneval" / "test.json",
    }
    with open(paths[name]) as f:
        data = json.load(f)
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


# ── Qwen Score ─────────────────────────────────────────────────────────────────
def get_qwen_scores(qwen_model, qwen_tokenizer, current_ids, llada_x0, device, max_ctx=512):
    """Get Qwen2.5-0.5B confidence scores for AR-guided unmasking."""
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


# ── M3 Generation ─────────────────────────────────────────────────────────────
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


# ── Benchmark evaluators ───────────────────────────────────────────────────────
def eval_bm(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
            device, gw, bm, data, warmup=0):
    """Evaluate M3 on a single benchmark.

    Returns dict with accuracy metric, avg_tps, and sample texts.
    Discards first `warmup` samples from TPS computation.
    """
    tps_list, n_correct, n_passed = [], 0, 0
    sample_texts = []  # Collect 5 samples for qualitative inspection

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
                correct = gsm8k_match(text, item["answer"])
                if correct: n_correct += 1
            elif bm == "math500":
                correct = math500_match(text, item.get("answer", item.get("solution", "")))
                if correct: n_correct += 1
            elif bm == "humaneval":
                passed = he_pass_at_1(text, item)
                if passed: n_passed += 1
                correct = passed

            # Only count TPS after warmup
            if i >= warmup:
                tps_list.append(tps)

            # Save first 5 non-warmup samples for qualitative inspection
            if len(sample_texts) < 5 and i >= warmup:
                sample_texts.append({
                    "index": i,
                    "correct": correct if bm != "humaneval" else passed,
                    "output_preview": text[:300],
                })
        except Exception as e:
            print(f"  [ERR] {bm}[{i}]: {e}")

        if (i+1) % 50 == 0:
            print(f"  {bm} [{i+1}/{len(data)}]")

    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if len(tps_list) > 1 else 0.0

    if bm in ["gsm8k", "math500"]:
        acc = n_correct / len(data) if data else 0.0
        return {acc_key[bm]: acc, "avg_tps": avg_tps, "tps_std": tps_std,
                "n": len(data), "n_correct": n_correct, "sample_texts": sample_texts}
    else:
        p1 = n_passed / len(data) if data else 0.0
        return {acc_key[bm]: p1, "avg_tps": avg_tps, "tps_std": tps_std,
                "n": len(data), "n_passed": n_passed, "sample_texts": sample_texts}


def compute_combined_metric(gsm8k_result, math500_result):
    """Compute 0.7*GSM8K + 0.3*MATH500 combined metrics."""
    gsm8k_acc = gsm8k_result.get("exact_match", 0)
    math500_acc = math500_result.get("exact_match", 0)
    gsm8k_tps = gsm8k_result.get("avg_tps", 0)
    math500_tps = math500_result.get("avg_tps", 0)

    gsm8k_speedup = gsm8k_tps / max(BASELINE["gsm8k"]["avg_tps"], 1e-6)
    math500_speedup = math500_tps / max(BASELINE["math500"]["avg_tps"], 1e-6)

    gsm8k_ret = gsm8k_acc / max(BASELINE["gsm8k"]["exact_match"], 1e-6)
    math500_ret = math500_acc / max(BASELINE["math500"]["exact_match"], 1e-6)

    # Combined metric: weighted average
    combined_acc_ret = 0.7 * gsm8k_ret + 0.3 * math500_ret
    combined_speedup = 0.7 * gsm8k_speedup + 0.3 * math500_speedup

    # QAS = Speedup * AccRet (NO penalty)
    combined_qas = combined_speedup * combined_acc_ret

    return {
        "gsm8k_acc": gsm8k_acc,
        "gsm8k_speedup": gsm8k_speedup,
        "gsm8k_acc_ret": gsm8k_ret,
        "math500_acc": math500_acc,
        "math500_speedup": math500_speedup,
        "math500_acc_ret": math500_ret,
        "combined_acc_ret": combined_acc_ret,
        "combined_speedup": combined_speedup,
        "combined_qas": combined_qas,
    }


def main():
    global _LLADA_TOKENIZER
    write_pid()
    start_time = datetime.now()

    # Use CUDA_VISIBLE_DEVICES from env, map to cuda:0 within the process
    device = "cuda:0"
    print(f"[m3_pareto_corrected PILOT] Starting at {start_time.isoformat()}")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"  Guidance weights: {GUIDANCE_WEIGHTS}")
    print(f"  Samples: {N_SAMPLES}")
    print(f"  Seeds: {SEEDS}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    # ── Load models ───────────────────────────────────────────────────────────
    print("[m3_pilot] Loading LLaDA-8B-Instruct...")
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    llada_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llada_tokenizer.padding_side = "left"
    _LLADA_TOKENIZER = llada_tokenizer
    llada_model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_after_llada = torch.cuda.memory_allocated(device) // 1024**2
    print(f"  LLaDA loaded. VRAM: {vram_after_llada} MB")

    print("[m3_pilot] Loading Qwen2.5-0.5B...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    # Use AutoModelForCausalLM for Qwen (it's an AR model)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_after_both = torch.cuda.memory_allocated(device) // 1024**2
    print(f"  Qwen loaded. Total VRAM: {vram_after_both} MB")

    # ── GPU profile ───────────────────────────────────────────────────────────
    gpu_name = torch.cuda.get_device_name(device)
    vram_total = torch.cuda.get_device_properties(device).total_memory // 1024**2
    gpu_profile = {
        "gpu_name": gpu_name,
        "vram_total_mb": vram_total,
        "vram_used_mb": vram_after_both,
        "utilization_pct": round(100 * vram_after_both / max(vram_total, 1), 1),
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2))

    # ── Pre-load datasets ─────────────────────────────────────────────────────
    datasets = {}
    for seed in SEEDS:
        datasets[seed] = {}
        for bm in ["gsm8k", "math500", "humaneval"]:
            datasets[seed][bm] = load_dataset(bm, seed, N_SAMPLES[bm])
        print(f"  seed={seed}: gsm8k={len(datasets[seed]['gsm8k'])}, "
              f"math500={len(datasets[seed]['math500'])}, "
              f"humaneval={len(datasets[seed]['humaneval'])}")

    # ── Run evaluations ───────────────────────────────────────────────────────
    all_results = {}  # gw -> seed -> bm -> metrics
    total_steps = len(GUIDANCE_WEIGHTS) * len(SEEDS) * 3  # 3 benchmarks each
    prog_step = 0

    for gw in GUIDANCE_WEIGHTS:
        all_results[str(gw)] = {}
        print(f"\n[m3_pilot] === guidance_weight={gw} ===")
        for seed in SEEDS:
            print(f"[m3_pilot] seed={seed}")
            all_results[str(gw)][str(seed)] = {}

            for bm in ["gsm8k", "math500", "humaneval"]:
                print(f"  Evaluating {bm}...")
                r = eval_bm(llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                            device, gw, bm, datasets[seed][bm],
                            warmup=WARMUP_SAMPLES if bm in ["gsm8k", "math500"] else 0)
                all_results[str(gw)][str(seed)][bm] = r

                bl_tps = BASELINE[bm]["avg_tps"]
                bl_acc = BASELINE[bm].get(acc_key[bm], 0)
                cur_acc = r.get(acc_key[bm], 0)
                speedup = r["avg_tps"] / max(bl_tps, 1e-6)
                ret = cur_acc / max(bl_acc, 1e-6) if bl_acc > 0 else 1.0
                print(f"    {bm}: acc={cur_acc:.4f}, TPS={r['avg_tps']:.1f}, "
                      f"speedup={speedup:.3f}x, ret={ret:.4f}")

                prog_step += 1
                report_progress(prog_step, total_steps, {
                    "gw": gw, "seed": seed, "bm": bm,
                    "acc": cur_acc, "speedup": round(speedup, 3)
                })

            torch.cuda.empty_cache(); gc.collect()

    # ── Compute Pareto points with corrected metrics ────────────────────────
    print("\n[m3_pilot] Computing corrected Pareto points...")
    pareto_points = []
    best_op_point, best_qas = None, -1.0

    for gw in GUIDANCE_WEIGHTS:
        gw_key = str(gw)
        seed_key = str(SEEDS[0])

        gsm8k_r = all_results[gw_key][seed_key].get("gsm8k", {})
        math500_r = all_results[gw_key][seed_key].get("math500", {})
        humaneval_r = all_results[gw_key][seed_key].get("humaneval", {})

        if not gsm8k_r or not math500_r:
            continue

        # Corrected combined metric
        combined = compute_combined_metric(gsm8k_r, math500_r)

        # HumanEval reported separately
        he_acc = humaneval_r.get("pass_at_1", 0) if humaneval_r else 0
        he_tps = humaneval_r.get("avg_tps", 0) if humaneval_r else 0
        he_speedup = he_tps / max(BASELINE["humaneval"]["avg_tps"], 1e-6)
        he_ret = he_acc / max(BASELINE["humaneval"]["pass_at_1"], 1e-6) if BASELINE["humaneval"]["pass_at_1"] > 0 else 0

        point = {
            "guidance_weight": float(gw),
            # Combined metric (0.7*GSM8K + 0.3*MATH500)
            "combined_speedup": combined["combined_speedup"],
            "combined_acc_ret": combined["combined_acc_ret"],
            "combined_qas": combined["combined_qas"],
            # Per-benchmark breakdown
            "gsm8k": {
                "accuracy": combined["gsm8k_acc"],
                "speedup": combined["gsm8k_speedup"],
                "acc_ret": combined["gsm8k_acc_ret"],
                "avg_tps": gsm8k_r["avg_tps"],
                "tps_std": gsm8k_r.get("tps_std", 0),
            },
            "math500": {
                "accuracy": combined["math500_acc"],
                "speedup": combined["math500_speedup"],
                "acc_ret": combined["math500_acc_ret"],
                "avg_tps": math500_r["avg_tps"],
                "tps_std": math500_r.get("tps_std", 0),
            },
            "humaneval": {
                "accuracy": he_acc,
                "speedup": he_speedup,
                "acc_ret": he_ret,
                "avg_tps": he_tps,
            },
        }
        pareto_points.append(point)

        print(f"  gw={gw}: combined_speedup={combined['combined_speedup']:.3f}x, "
              f"combined_acc_ret={combined['combined_acc_ret']:.4f}, "
              f"QAS={combined['combined_qas']:.4f}")
        print(f"    GSM8K: acc={combined['gsm8k_acc']:.4f}, ret={combined['gsm8k_acc_ret']:.4f}, "
              f"speedup={combined['gsm8k_speedup']:.3f}x")
        print(f"    MATH500: acc={combined['math500_acc']:.4f}, ret={combined['math500_acc_ret']:.4f}, "
              f"speedup={combined['math500_speedup']:.3f}x")

        if combined["combined_qas"] > best_qas:
            best_qas = combined["combined_qas"]
            best_op_point = point

    # ── Pass criteria check ────────────────────────────────────────────────────
    # "At least one gw achieves accuracy_retention >= 0.95 on GSM8K AND speedup > 1.0x"
    pass_criteria_met = False
    for p in pareto_points:
        if p["gsm8k"]["acc_ret"] >= 0.95 and p["gsm8k"]["speedup"] > 1.0:
            pass_criteria_met = True
            break

    # Also check if any gw achieves speedup > 1.0 at all
    any_speedup = any(p["combined_speedup"] > 1.0 for p in pareto_points)

    verdict = "GO" if pass_criteria_met else ("MARGINAL" if any_speedup else "NO_GO")

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    # ── Print summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[m3_pareto_corrected PILOT] VERDICT: {verdict}")
    print(f"  Pass criteria (GSM8K ret>=0.95 AND speedup>1.0): {'MET' if pass_criteria_met else 'NOT MET'}")
    print(f"  Elapsed: {elapsed_min:.1f} min")
    if best_op_point:
        print(f"  Best operating point: gw={best_op_point['guidance_weight']}")
        print(f"    Combined QAS={best_op_point['combined_qas']:.4f}")
        print(f"    Combined speedup={best_op_point['combined_speedup']:.3f}x")
        print(f"    Combined AccRet={best_op_point['combined_acc_ret']:.4f}")
    print(f"{'='*60}")

    # ── Save results ───────────────────────────────────────────────────────────
    # Strip sample_texts from all_results for compact JSON (save separately)
    all_results_compact = {}
    for gw_key, seeds_data in all_results.items():
        all_results_compact[gw_key] = {}
        for s_key, bm_data in seeds_data.items():
            all_results_compact[gw_key][s_key] = {}
            for bm_name, metrics in bm_data.items():
                compact = {k: v for k, v in metrics.items() if k != "sample_texts"}
                all_results_compact[gw_key][s_key][bm_name] = compact

    output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "guide_model": "Qwen2.5-0.5B",
        "method": "M3-ARGuidedUnmasking",
        "mode": "pilot",
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
        "pass_criteria": "At least one gw achieves accuracy_retention >= 0.95 on GSM8K AND speedup > 1.0x",
        "pass_criteria_met": pass_criteria_met,
        "pareto_points": pareto_points,
        "best_operating_point": best_op_point,
        "baseline": BASELINE,
        "all_results": all_results_compact,
        "gpu_profile": gpu_profile,
    }

    out_path = RESULTS_DIR / "m3_pareto_corrected.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[m3_pilot] Results saved to {out_path}")

    # Save sample texts separately for qualitative inspection
    sample_path = RESULTS_DIR / "m3_pilot_samples.json"
    samples_out = {}
    for gw_key, seeds_data in all_results.items():
        samples_out[gw_key] = {}
        for s_key, bm_data in seeds_data.items():
            samples_out[gw_key][s_key] = {}
            for bm_name, metrics in bm_data.items():
                if "sample_texts" in metrics:
                    samples_out[gw_key][s_key][bm_name] = metrics["sample_texts"]
    sample_path.write_text(json.dumps(samples_out, indent=2))
    print(f"[m3_pilot] Sample texts saved to {sample_path}")

    # ── Update gpu_progress ────────────────────────────────────────────────────
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
                "n_samples": N_SAMPLES,
                "mode": "pilot",
                "gpu_model": gpu_name,
                "gpu_count": 1,
            }
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"[m3_pilot] gpu_progress.json updated")
    except Exception as e:
        print(f"WARNING: gpu_progress update failed: {e}")

    # ── Mark done ─────────────────────────────────────────────────────────────
    summary = (f"M3 pilot: verdict={verdict}, best_gw={best_op_point['guidance_weight']}, "
               f"combined_QAS={best_op_point['combined_qas']:.4f}, "
               f"combined_speedup={best_op_point['combined_speedup']:.3f}x, "
               f"GSM8K_ret={best_op_point['gsm8k']['acc_ret']:.4f}") if best_op_point else f"verdict={verdict}"
    mark_done(status="success", summary=summary)
    report_progress(total_steps, total_steps, {"status": "done", "verdict": verdict})
    print("[m3_pilot] Done.")
    return output


if __name__ == "__main__":
    main()
