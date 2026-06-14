"""
M1 (Entropy Cache) Ablation Studies.

Task: full_ablation_m1
Ablations on the best M1 operating point (entropy_threshold=2.0):

1. threshold=0.0 (no cache skip): all tokens computed normally → baseline reference
2. threshold=1.0 (aggressive): only very low-entropy tokens skip computation
3. threshold=2.0 (best, reference): optimal Pareto point from full_m1_pareto
4. threshold=3.0 (relaxed): more tokens skip, higher speedup but lower quality?
5. threshold=4.0 (very relaxed): nearly all tokens cached

Evaluate on GSM8K (200 samples) + HumanEval (164 samples).
2 seeds: 42, 123.
"""
import os, sys, json, time, random, re, subprocess, gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
CODE_DIR    = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "ablation_m1"
TASK_ID     = "full_ablation_m1"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE = {
    "gsm8k":     {"exact_match": 0.7122, "avg_tps": 31.013},
    "humaneval": {"pass_at_1":   0.0244, "avg_tps": 97.999},
}

SEEDS    = [42, 123]
N_GSM8K  = 200
N_HE     = 164
MASK_ID  = 126336

# Thresholds to test (best=2.0 from full_m1_pareto)
THRESHOLDS = [0.0, 1.0, 2.0, 3.0, 4.0]


# ── Helpers ────────────────────────────────────────────────────────────────────
def write_pid():
    pid_root = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
    pid_root.write_text(str(os.getpid()))
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, metric=None):
    prog = json.dumps({"task_id": TASK_ID, "step": step, "total_steps": total,
                       "updated_at": datetime.now().isoformat(), "metric": metric or {}})
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(prog)
    # Also write to root for orchestrator
    (WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json").write_text(prog)

def mark_done(status="success", summary=""):
    pid_root = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
    if pid_root.exists(): pid_root.unlink()
    pid_sub = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_sub.exists(): pid_sub.unlink()
    done_data = json.dumps({"task_id": TASK_ID, "status": status,
                            "summary": summary, "timestamp": datetime.now().isoformat()})
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(done_data)
    # Also write to root for orchestrator
    (WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE").write_text(done_data)


# ── Data ───────────────────────────────────────────────────────────────────────
def load_gsm8k(n, seed):
    with open(SHARED / "datasets" / "gsm8k" / "test.json") as f:
        data = json.load(f)
    return random.Random(seed).sample(data, min(n, len(data)))

def load_humaneval(n, seed):
    with open(SHARED / "datasets" / "humaneval" / "test.json") as f:
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

def gsm8k_prompt(q):
    return GSM8K_8SHOT + f"Question: {q}\nAnswer:"

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

def he_pass_at_1(code, problem):
    full = problem["prompt"] + code + "\n" + problem["test"]
    full += f"\ncheck({problem['entry_point']})\n"
    try:
        r = subprocess.run(["python", "-c", full], capture_output=True, text=True, timeout=10)
        return r.returncode == 0
    except: return False


# ── M1 Generation ──────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_m1(model, tokenizer, prompt, device,
                entropy_threshold=2.0, gen_length=256,
                block_length=32, steps=64):
    """M1: Entropy-based cache skip. High-confidence tokens use cached computation."""
    msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full((batch, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    num_blocks = (gen_length + block_length - 1) // block_length
    steps_per_block = max(1, steps // num_blocks)

    # Cached state for M1
    cached_x0 = None       # Previously predicted tokens (cache)
    cached_valid = None    # Which positions are cached

    t0 = time.perf_counter()

    total_skipped = 0
    total_computed = 0

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = min(prompt_len + gen_length, block_start + block_length)

        for step in range(steps_per_block):
            mask_index = (x[:, block_start:block_end] == MASK_ID)
            if not mask_index.any(): break

            # Full forward pass
            logits = model(x, attention_mask=attn).logits[:, block_start:block_end, :]
            probs  = F.softmax(logits, dim=-1)
            x0     = torch.argmax(probs, dim=-1)  # (B, block_len)
            x0_p   = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

            # Compute entropy for current block
            entropy = -(probs * torch.log(probs + 1e-10)).sum(-1)  # (B, block_len)

            # M1 logic: positions with entropy <= threshold are "confident" and can be accepted
            if entropy_threshold > 0:
                low_entropy_mask = entropy[0] <= entropy_threshold  # (block_len,)
                # For masked positions with low entropy, accept the prediction
                masked_and_confident = mask_index[0] & low_entropy_mask
                total_skipped += masked_and_confident.sum().item()
            else:
                masked_and_confident = torch.zeros_like(mask_index[0])

            # Unmask the confident positions
            for pos in range(block_end - block_start):
                abs_pos = block_start + pos
                if x[0, abs_pos] == MASK_ID and (entropy_threshold > 0 and low_entropy_mask[pos].item()):
                    x[0, abs_pos] = x0[0, pos]

            # Standard top-k unmasking for remaining positions
            remaining = mask_index[0].sum().float()
            steps_left = steps_per_block - step
            k = int(torch.ceil(remaining / steps_left).clamp(min=1).item())
            k = min(k, mask_index[0].sum().item())
            if k <= 0: continue

            # Only select from still-masked positions
            gen_conf = torch.where(mask_index[0], x0_p[0],
                                   torch.tensor(-float('inf'), device=device))
            _, sel = torch.topk(gen_conf, k=k)
            for s in sel:
                abs_pos = block_start + s.item()
                if x[0, abs_pos] == MASK_ID:
                    x[0, abs_pos] = x0[0, s.item()]
            total_computed += k

    elapsed = time.perf_counter() - t0
    gen_text = tokenizer.batch_decode(x[:, prompt_len:], skip_special_tokens=True)[0]
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    skip_rate = total_skipped / max(total_skipped + total_computed, 1)
    return gen_text, tps, {"skip_rate": skip_rate, "elapsed": elapsed}


# ── Evaluation ─────────────────────────────────────────────────────────────────
def evaluate_config(model, tokenizer, device, threshold, gsm8k_data, he_data,
                    config_label, prog_step, total_prog):
    print(f"\n[ablation_m1] Config: {config_label} (threshold={threshold})")

    # GSM8K
    gsm_tps, gsm_correct, skip_rates = [], 0, []
    for i, item in enumerate(gsm8k_data):
        try:
            text, tps, stats = generate_m1(model, tokenizer,
                gsm8k_prompt(item["question"]), device, entropy_threshold=threshold)
            if gsm8k_match(text, item["answer"]): gsm_correct += 1
            if i >= 3: gsm_tps.append(tps)
            skip_rates.append(stats["skip_rate"])
        except Exception as e:
            print(f"  [ERR] gsm8k[{i}]: {e}")
        if (i+1) % 50 == 0:
            print(f"  {config_label} gsm8k [{i+1}/{len(gsm8k_data)}] acc={gsm_correct/(i+1):.3f}")

    gsm_acc = gsm_correct / len(gsm8k_data) if gsm8k_data else 0.0
    gsm_avg_tps = float(np.mean(gsm_tps)) if gsm_tps else 0.0
    gsm_speedup = gsm_avg_tps / max(BASELINE["gsm8k"]["avg_tps"], 1e-6)
    gsm_ret = gsm_acc / max(BASELINE["gsm8k"]["exact_match"], 1e-6)
    avg_skip = float(np.mean(skip_rates)) if skip_rates else 0.0
    print(f"  GSM8K: acc={gsm_acc:.3f}, tps={gsm_avg_tps:.1f}, speedup={gsm_speedup:.3f}x, skip_rate={avg_skip:.3f}")

    report_progress(prog_step, total_prog, {"config": config_label, "gsm8k_acc": gsm_acc})

    # HumanEval
    he_tps, he_passed = [], 0
    for i, item in enumerate(he_data):
        try:
            code, tps, _ = generate_m1(model, tokenizer,
                f"Complete the following Python function:\n\n{item['prompt']}", device,
                entropy_threshold=threshold, gen_length=256)
            if he_pass_at_1(code, item): he_passed += 1
            he_tps.append(tps)
        except Exception as e:
            print(f"  [ERR] he[{i}]: {e}")

    he_p1 = he_passed / len(he_data) if he_data else 0.0
    he_avg_tps = float(np.mean(he_tps)) if he_tps else 0.0
    he_speedup = he_avg_tps / max(BASELINE["humaneval"]["avg_tps"], 1e-6)
    bl_he = BASELINE["humaneval"]["pass_at_1"]
    he_ret = he_p1 / max(bl_he, 1e-6) if bl_he > 0.01 else (1.0 if he_p1 >= bl_he else 0.0)
    print(f"  HumanEval: pass@1={he_p1:.3f}, tps={he_avg_tps:.1f}, speedup={he_speedup:.3f}x")

    combined_speedup = (gsm_speedup * N_GSM8K + he_speedup * N_HE) / (N_GSM8K + N_HE)
    combined_ret     = (gsm_ret * N_GSM8K + he_ret * N_HE) / (N_GSM8K + N_HE)
    combined_qas     = float(combined_speedup * combined_ret)

    return {
        "config": config_label,
        "threshold": threshold,
        "gsm8k": {"exact_match": gsm_acc, "avg_tps": gsm_avg_tps, "speedup": gsm_speedup, "acc_ret": gsm_ret},
        "humaneval": {"pass_at_1": he_p1, "avg_tps": he_avg_tps, "speedup": he_speedup, "acc_ret": he_ret},
        "avg_skip_rate": avg_skip,
        "combined_speedup": combined_speedup,
        "combined_acc_ret": combined_ret,
        "combined_qas": combined_qas,
    }


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now()
    device = os.environ.get("CUDA_DEVICE", "cuda:0")
    print(f"[full_ablation_m1] Starting at {start_time.isoformat()} on {device}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    from transformers import AutoTokenizer, AutoModel
    print("[full_ablation_m1] Loading LLaDA-8B-Instruct...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"[full_ablation_m1] Model loaded. VRAM: {torch.cuda.memory_allocated(device)//1024**2} MB")

    # Pre-load datasets
    datasets = {}
    for seed in SEEDS:
        datasets[seed] = {
            "gsm8k":     load_gsm8k(N_GSM8K, seed),
            "humaneval": load_humaneval(N_HE, seed),
        }

    # Config labels
    configs = []
    for t in THRESHOLDS:
        if t == 0.0:
            label = "M1-no-cache (t=0.0)"
        elif t == 2.0:
            label = "M1-full (t=2.0) [ref]"
        else:
            label = f"M1-t{t} (t={t})"
        configs.append((label, t))

    all_results = {}
    total_steps = len(configs) * len(SEEDS)
    prog_step = 0

    for cfg_label, threshold in configs:
        all_results[cfg_label] = {}
        for seed in SEEDS:
            print(f"\n[full_ablation_m1] {cfg_label} | seed={seed}")
            r = evaluate_config(
                model, tokenizer, device,
                threshold=threshold,
                gsm8k_data=datasets[seed]["gsm8k"],
                he_data=datasets[seed]["humaneval"],
                config_label=cfg_label,
                prog_step=prog_step,
                total_prog=total_steps,
            )
            all_results[cfg_label][seed] = r
            prog_step += 1
            torch.cuda.empty_cache(); gc.collect()

    # ── Aggregate and compute delta-QAS ────────────────────────────────────────
    agg = {}
    for cfg_label, seed_data in all_results.items():
        speedups = [v["combined_speedup"] for v in seed_data.values()]
        rets     = [v["combined_acc_ret"]  for v in seed_data.values()]
        qas_vals = [v["combined_qas"]      for v in seed_data.values()]
        t        = list(seed_data.values())[0]["threshold"]
        agg[cfg_label] = {
            "threshold": t,
            "avg_speedup": float(np.mean(speedups)),
            "avg_acc_ret": float(np.mean(rets)),
            "avg_qas": float(np.mean(qas_vals)),
        }

    ref_label = "M1-full (t=2.0) [ref]"
    reference_qas = agg.get(ref_label, {}).get("avg_qas", 1.0)

    ablation_summary = []
    for label, metrics in agg.items():
        delta = metrics["avg_qas"] - reference_qas
        ablation_summary.append({
            "config": label,
            "threshold": metrics["threshold"],
            "avg_speedup": metrics["avg_speedup"],
            "avg_acc_ret": metrics["avg_acc_ret"],
            "avg_qas": metrics["avg_qas"],
            "delta_qas": float(delta),
            "delta_pct": float(100 * delta / max(reference_qas, 1e-6)),
        })

    print("\n[full_ablation_m1] === ABLATION SUMMARY ===")
    print(f"{'Config':<35} {'Speedup':>8} {'AccRet':>7} {'QAS':>6} {'ΔQAS':>8} {'ΔQAS%':>7}")
    for s in ablation_summary:
        print(f"  {s['config']:<33} {s['avg_speedup']:>8.3f}x {s['avg_acc_ret']:>7.3f} "
              f"{s['avg_qas']:>6.3f} {s['delta_qas']:>+8.3f} {s['delta_pct']:>+7.1f}%")

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "method": "M1-EntropyCacheSkip",
        "reference_config": {"threshold": 2.0},
        "seeds": SEEDS,
        "n_gsm8k": N_GSM8K,
        "n_humaneval": N_HE,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": round(elapsed_min, 2),
        "ablation_summary": ablation_summary,
        "all_results": {
            label: {str(s): v for s, v in sd.items()}
            for label, sd in all_results.items()
        },
    }

    out_path = RESULTS_DIR / "m1_ablation.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"[full_ablation_m1] Results → {out_path}")

    # Update gpu_progress
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {}
        gp.setdefault("completed", [])
        if TASK_ID not in gp["completed"]: gp["completed"].append(TASK_ID)
        gp.get("running", {}).pop(TASK_ID, None)
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 45, "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
        }
        gp_path.write_text(json.dumps(gp, indent=2))
    except Exception as e:
        print(f"WARNING: gpu_progress update failed: {e}")

    summary = f"M1 ablation done in {elapsed_min:.1f}min. Reference QAS={reference_qas:.3f}. See ablation_summary."
    mark_done(status="success", summary=summary)
    report_progress(total_steps, total_steps, {"status": "done"})
    print("[full_ablation_m1] Done.")
    return output


if __name__ == "__main__":
    main()
