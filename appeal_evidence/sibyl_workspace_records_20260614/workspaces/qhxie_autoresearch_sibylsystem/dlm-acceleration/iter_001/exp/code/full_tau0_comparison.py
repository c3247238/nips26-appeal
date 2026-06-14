"""
Full tau=0.0 Paradox Resolution: CD-SSD vs naive-T16 Full Validation.

Task: full_tau0_comparison
- Full validation of pilot results on 200 GSM8K, seeds {42, 123}
- Compares 5 conditions:
  1. CD-SSD(tau=0.9): IGSD with tau=0.9, t_draft=16, t_full=64 (reuse pairwise data)
  2. CD-SSD(tau=0.0): IGSD with tau=0.0, all tokens accepted in draft -> no refine
  3. naive-T16: Standard LLaDA denoising with T=16 steps (no IGSD structure)
  4. M1+CD-SSD(tau=0.9): Already measured in pairwise experiment (reuse)
  5. M1+naive-T16: M1 entropy caching + 16-step denoising

Key decision rules:
  - tau0_beats_naiveT16: accept gate has inherent value (frozen-token partition matters)
  - tau0_approx_naiveT16: accept gate adds no value vs equal-steps baseline
  - M1_cdssd_beats_M1_naiveT16: synergy from frozen-token KV anchor effect
  - M1_naiveT16_ortho_gt_1: synergy from step-reduction + KV, not CD-SSD specific

Benchmarks: GSM8K (200 samples)
Seeds: 42, 123
Output: exp/results/full_tau0_comparison/full_tau0_comparison.json

Usage:
    CUDA_VISIBLE_DEVICES=2 conda run -n sibyl_dlm-acceleration python full_tau0_comparison.py
"""
import os, sys, json, time, random, re, gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
CODE_DIR    = WORKSPACE / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full_tau0_comparison"
TASK_ID     = "full_tau0_comparison"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MASK_ID = 126336

# Reference baselines from full_baseline.json (3-seed mean)
BASELINE = {
    "gsm8k": {"exact_match": 0.7122, "avg_tps": 31.013},
}

# Existing M1+CD-SSD(tau=0.9) data from full_pairwise_ortho
# Source: exp/results/full_pairwise/full_pairwise_ortho.json > all_results > M1+IGSD
EXISTING_M1_CDSSD = {
    "42": {"gsm8k_acc": 0.39, "gsm8k_tps": 206.60547395378595,
           "combined_speedup": 5.128902631108715, "gsm_ret": 0.5475989890480202,
           "qas": 1.5431768657806937, "ortho": 1.2924429361647352},
    "123": {"gsm8k_acc": 0.445, "gsm8k_tps": 207.5181159706292,
            "combined_speedup": 5.1401352824580435, "gsm_ret": 0.6248244875035102,
            "qas": 1.7646606558025042, "ortho": 1.4779402477407908},
}

# Existing CD-SSD(tau=0.9) reference from full IGSD pareto (seed 42, seed 123)
# Source: exp/results/full_igsd/igsd_p2_tau09_td16_s123.json
EXISTING_CDSSD_TAU09 = {
    "123": {"gsm8k_acc": 0.4533737680060652, "gsm8k_tps": 140.71027461004348,
            "speedup": 4.537138445492003, "acc_ret": 0.6365820949256742},
}

SEEDS = [42, 123]
N_GSM8K = 200

# IGSD params
IGSD_T_DRAFT = 16
IGSD_T_FULL  = 64
GEN_LENGTH   = 256

M1_THRESHOLD = 2.0  # entropy threshold from M1 operating point


# ── System monitor helpers ─────────────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, metric=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "step": step, "total_steps": total,
        "updated_at": datetime.now().isoformat(), "metric": metric or {},
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


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_gsm8k(n, seed):
    with open(SHARED / "datasets" / "gsm8k" / "test.json") as f:
        data = json.load(f)
    return random.Random(seed).sample(data, min(n, len(data)))


# ── Prompt / answer helpers ───────────────────────────────────────────────────
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


# ── Generation functions ───────────────────────────────────────────────────────

@torch.no_grad()
def generate_cdssd(model, tokenizer, prompt, device, tau, t_draft=16, t_full=64,
                   gen_length=256):
    """
    CD-SSD (IGSD) generation.

    tau=0.9  → standard operating point: ~88% tokens accepted in draft
    tau=0.0  → all tokens accepted (confidence always >= 0), refine phase is empty
               → effectively equivalent to T_draft steps of denoising only
    """
    msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    t0 = time.perf_counter()

    # === Phase 1: Draft phase (T_draft steps, whole-sequence) ===
    tokens_per_step = max(1, gen_length // t_draft)
    remainder = gen_length % t_draft
    for step in range(t_draft):
        n_masked = int((x[0, prompt_len:] == MASK_ID).sum().item())
        if n_masked == 0:
            break
        logits = model(x, attention_mask=attn).logits
        p = F.softmax(logits, dim=-1)
        x0 = torch.argmax(p, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        mask_index = (x == MASK_ID)
        mask_index[:, :prompt_len] = False
        conf = torch.where(mask_index, x0_p, torch.tensor(-float("inf"), device=device))
        k = tokens_per_step + (1 if step < remainder else 0)
        k = min(k, n_masked)
        if k > 0:
            _, sel = torch.topk(conf[0], k=k)
            ti = torch.zeros_like(x[0], dtype=torch.bool)
            ti[sel] = True
            ti = ti.unsqueeze(0) & mask_index
            x[ti] = x0[ti]

    # === Phase 2: Partition ===
    final_logits = model(x, attention_mask=attn).logits
    final_p = F.softmax(final_logits[:, prompt_len:, :], dim=-1)
    gen_region = x[:, prompt_len:].clone()
    still_masked = (gen_region == MASK_ID)
    draft_pred = torch.argmax(final_p, dim=-1)
    draft_conf = torch.gather(final_p, -1, draft_pred.unsqueeze(-1)).squeeze(-1)
    filled_conf = torch.gather(final_p, -1, gen_region.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    final_confidence = torch.where(still_masked, draft_conf, filled_conf)
    gen_region_filled = torch.where(still_masked, draft_pred, gen_region)
    x[:, prompt_len:] = gen_region_filled

    # Accept gate: tau=0.0 -> all accepted; tau=0.9 -> ~88% accepted
    s_accept_gen = (final_confidence >= tau)
    n_accept = int(s_accept_gen.sum().item())
    n_total = s_accept_gen.numel()
    n_refine = n_total - n_accept
    accept_rate = n_accept / n_total

    # === Phase 3: Refine phase (only if any tokens to refine) ===
    x_refine = x.clone()
    s_accept_full = torch.cat([
        torch.ones(1, prompt_len, dtype=torch.bool, device=device),
        s_accept_gen
    ], dim=1)
    s_refine_full = ~s_accept_full
    x_refine[s_refine_full] = MASK_ID

    if n_refine > 0:
        tokens_per_refine_step = max(1, n_refine // t_full)
        remainder_refine = n_refine % t_full
        for step in range(t_full):
            n_masked_now = int((x_refine[0, prompt_len:] == MASK_ID).sum().item())
            if n_masked_now == 0:
                break
            logits = model(x_refine, attention_mask=attn).logits
            p = F.softmax(logits, dim=-1)
            x0 = torch.argmax(p, dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
            mask_index = (x_refine == MASK_ID) & s_refine_full
            conf = torch.where(mask_index, x0_p, torch.tensor(-float("inf"), device=device))
            k = tokens_per_refine_step + (1 if step < remainder_refine else 0)
            k = min(k, n_masked_now)
            if k > 0:
                _, sel = torch.topk(conf[0], k=k)
                ti = torch.zeros_like(x_refine[0], dtype=torch.bool)
                ti[sel] = True
                ti = ti.unsqueeze(0) & mask_index
                x_refine[ti] = x0[ti]

    # Fill any remaining masked tokens
    still_left = (x_refine[:, prompt_len:] == MASK_ID)
    if still_left.any():
        lf = model(x_refine, attention_mask=attn).logits
        pf = F.softmax(lf[:, prompt_len:, :], dim=-1)
        fp = torch.argmax(pf, dim=-1)
        x_refine[:, prompt_len:] = torch.where(still_left, fp, x_refine[:, prompt_len:])

    elapsed = time.perf_counter() - t0
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    text = tokenizer.decode(x_refine[0, prompt_len:].tolist(), skip_special_tokens=True)
    return text, tps, accept_rate


@torch.no_grad()
def generate_naive_t16(model, tokenizer, prompt, device, t_steps=16, gen_length=256):
    """
    Naive T=16 denoising: standard LLaDA denoising with only 16 steps.
    No draft/refine split - just 16 uniform denoising steps using the standard
    top-k-per-step schedule (same as LLaDA's default but with T=16 instead of T=64).
    """
    msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    t0 = time.perf_counter()

    tokens_per_step = max(1, gen_length // t_steps)
    remainder = gen_length % t_steps

    for step in range(t_steps):
        n_masked = int((x[0, prompt_len:] == MASK_ID).sum().item())
        if n_masked == 0:
            break
        logits = model(x, attention_mask=attn).logits
        p = F.softmax(logits, dim=-1)
        x0 = torch.argmax(p, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        mask_index = (x == MASK_ID)
        mask_index[:, :prompt_len] = False
        conf = torch.where(mask_index, x0_p, torch.tensor(-float("inf"), device=device))
        k = tokens_per_step + (1 if step < remainder else 0)
        k = min(k, n_masked)
        if k > 0:
            _, sel = torch.topk(conf[0], k=k)
            ti = torch.zeros_like(x[0], dtype=torch.bool)
            ti[sel] = True
            ti = ti.unsqueeze(0) & mask_index
            x[ti] = x0[ti]

    elapsed = time.perf_counter() - t0
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    text = tokenizer.decode(x[0, prompt_len:].tolist(), skip_special_tokens=True)
    return text, tps


@torch.no_grad()
def generate_m1_naive_t16(model, tokenizer, prompt, device, t_steps=16,
                           m1_thresh=M1_THRESHOLD, gen_length=256):
    """
    M1 + naive T=16: entropy-threshold acceptance gate applied per-step,
    combined with 16-step denoising schedule.
    Tokens with entropy < m1_thresh are 'cached' (accepted early and frozen).
    """
    msg = [{"role": "user", "content": prompt}]
    prompt_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
    enc = tokenizer([prompt_text], add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
    x[:, :prompt_len] = input_ids
    attn = torch.cat([attention_mask,
                      torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)], dim=-1)

    t0 = time.perf_counter()

    tokens_per_step = max(1, gen_length // t_steps)
    remainder = gen_length % t_steps

    for step in range(t_steps):
        n_masked = int((x[0, prompt_len:] == MASK_ID).sum().item())
        if n_masked == 0:
            break
        logits = model(x, attention_mask=attn).logits
        p = F.softmax(logits, dim=-1)
        x0 = torch.argmax(p, dim=-1)
        x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)
        # Entropy for M1 gate
        entropy = -(p * torch.log(p + 1e-10)).sum(-1)

        mask_index = (x == MASK_ID)
        mask_index[:, :prompt_len] = False

        # M1: low-entropy tokens (high confidence) get priority
        # Combined score: use confidence boosted by low entropy
        entropy_boost = torch.where(entropy < m1_thresh,
                                    torch.tensor(1.0, device=device),
                                    torch.tensor(0.0, device=device))
        # Score = confidence + entropy_boost to prioritize low-entropy tokens
        score = torch.where(mask_index, x0_p + entropy_boost, torch.tensor(-float("inf"), device=device))
        score[:, :prompt_len] = -float("inf")

        k = tokens_per_step + (1 if step < remainder else 0)
        k = min(k, n_masked)
        if k > 0:
            _, sel = torch.topk(score[0], k=k)
            ti = torch.zeros_like(x[0], dtype=torch.bool)
            ti[sel] = True
            ti = ti.unsqueeze(0) & mask_index
            x[ti] = x0[ti]

    elapsed = time.perf_counter() - t0
    tps = gen_length / elapsed if elapsed > 0 else 0.0
    text = tokenizer.decode(x[0, prompt_len:].tolist(), skip_special_tokens=True)
    return text, tps


# ── Evaluation ─────────────────────────────────────────────────────────────────
def eval_gsm8k_condition(gen_fn, gsm8k_data, tag, warmup=3):
    """Evaluate a generation function on GSM8K data."""
    tps_list, correct = [], 0
    for i, item in enumerate(gsm8k_data):
        try:
            result = gen_fn(gsm8k_prompt(item["question"]))
            if isinstance(result, tuple):
                text = result[0]
                tps = result[1]
            else:
                text, tps = result, 0.0
            if gsm8k_match(text, item["answer"]):
                correct += 1
            if i >= warmup:
                tps_list.append(tps)
        except Exception as e:
            print(f"  [ERR] {tag}[{i}]: {e}")
        if (i + 1) % 50 == 0:
            acc_so_far = correct / (i + 1)
            print(f"  {tag}: [{i+1}/{len(gsm8k_data)}] acc={acc_so_far:.3f}")
    acc = correct / len(gsm8k_data) if gsm8k_data else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return acc, avg_tps


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now()
    device = os.environ.get("CUDA_DEVICE", "cuda:0")
    print(f"[{TASK_ID}] Starting at {start_time.isoformat()} on {device}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    from transformers import AutoTokenizer, AutoModel

    print(f"[{TASK_ID}] Loading LLaDA-8B-Instruct from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()
    print(f"[{TASK_ID}] Model loaded. VRAM: {torch.cuda.memory_allocated(device) // 1024**2} MB")

    # Load datasets per seed
    datasets = {seed: load_gsm8k(N_GSM8K, seed) for seed in SEEDS}

    # Conditions to evaluate (only the new ones; M1+CD-SSD(tau=0.9) reused from pairwise)
    CONDITIONS = [
        ("cdssd_tau00", "CD-SSD(tau=0.0)"),
        ("naive_t16",   "naive-T16"),
        ("m1_naive_t16","M1+naive-T16"),
        ("cdssd_tau09", "CD-SSD(tau=0.9)"),  # also measure fresh to cross-validate
    ]

    all_results = {}
    total_steps = len(CONDITIONS) * len(SEEDS)
    prog_step = 0

    for cond_id, cond_name in CONDITIONS:
        all_results[cond_id] = {}
        for seed in SEEDS:
            print(f"\n[{TASK_ID}] === {cond_name} | seed={seed} ===")
            gsm8k_data = datasets[seed]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            if cond_id == "cdssd_tau09":
                def gen_fn(prompt):
                    text, tps, ar = generate_cdssd(model, tokenizer, prompt, device,
                                                    tau=0.9, t_draft=IGSD_T_DRAFT,
                                                    t_full=IGSD_T_FULL, gen_length=GEN_LENGTH)
                    return text, tps
            elif cond_id == "cdssd_tau00":
                def gen_fn(prompt):
                    text, tps, ar = generate_cdssd(model, tokenizer, prompt, device,
                                                    tau=0.0, t_draft=IGSD_T_DRAFT,
                                                    t_full=IGSD_T_FULL, gen_length=GEN_LENGTH)
                    return text, tps
            elif cond_id == "naive_t16":
                def gen_fn(prompt):
                    return generate_naive_t16(model, tokenizer, prompt, device,
                                             t_steps=IGSD_T_DRAFT, gen_length=GEN_LENGTH)
            elif cond_id == "m1_naive_t16":
                def gen_fn(prompt):
                    return generate_m1_naive_t16(model, tokenizer, prompt, device,
                                                  t_steps=IGSD_T_DRAFT, gen_length=GEN_LENGTH)

            gsm_acc, gsm_tps = eval_gsm8k_condition(gen_fn, gsm8k_data, cond_name)
            gsm_speedup = gsm_tps / max(BASELINE["gsm8k"]["avg_tps"], 1e-6)
            gsm_ret = gsm_acc / max(BASELINE["gsm8k"]["exact_match"], 1e-6)
            qas = gsm_speedup * gsm_ret

            print(f"  {cond_name}: acc={gsm_acc:.4f}, tps={gsm_tps:.2f}, "
                  f"speedup={gsm_speedup:.3f}x, acc_ret={gsm_ret:.3f}, QAS={qas:.3f}")

            all_results[cond_id][str(seed)] = {
                "gsm8k_acc": gsm_acc,
                "gsm8k_tps": gsm_tps,
                "gsm_speedup": gsm_speedup,
                "gsm_ret": gsm_ret,
                "qas": qas,
            }
            prog_step += 1
            report_progress(prog_step, total_steps, {
                "cond": cond_id, "seed": seed,
                "gsm_acc": gsm_acc, "speedup": gsm_speedup
            })
            torch.cuda.empty_cache()
            gc.collect()

    # ── Aggregate across seeds ─────────────────────────────────────────────────
    def agg(cond_id):
        rows = list(all_results[cond_id].values())
        return {
            "avg_gsm_acc":    float(np.mean([r["gsm8k_acc"]   for r in rows])),
            "avg_gsm_tps":    float(np.mean([r["gsm8k_tps"]   for r in rows])),
            "avg_speedup":    float(np.mean([r["gsm_speedup"] for r in rows])),
            "avg_acc_ret":    float(np.mean([r["gsm_ret"]     for r in rows])),
            "avg_qas":        float(np.mean([r["qas"]          for r in rows])),
        }

    condition_summaries = {}
    for cond_id, cond_name in CONDITIONS:
        condition_summaries[cond_id] = {"name": cond_name, **agg(cond_id)}

    # Add existing M1+CD-SSD(tau=0.9) from full pairwise (pre-computed)
    # Note: pairwise used combined metric (GSM8K+HumanEval); we use GSM8K-only for direct comparison
    m1_cdssd_gsm_accs = [EXISTING_M1_CDSSD["42"]["gsm8k_acc"],
                          EXISTING_M1_CDSSD["123"]["gsm8k_acc"]]
    m1_cdssd_gsm_tps  = [EXISTING_M1_CDSSD["42"]["gsm8k_tps"],
                          EXISTING_M1_CDSSD["123"]["gsm8k_tps"]]
    m1_cdssd_speedups = [t / BASELINE["gsm8k"]["avg_tps"] for t in m1_cdssd_gsm_tps]
    m1_cdssd_rets     = [a / BASELINE["gsm8k"]["exact_match"] for a in m1_cdssd_gsm_accs]
    m1_cdssd_qas      = [s * r for s, r in zip(m1_cdssd_speedups, m1_cdssd_rets)]
    condition_summaries["m1_cdssd_tau09"] = {
        "name": "M1+CD-SSD(tau=0.9)",
        "avg_gsm_acc": float(np.mean(m1_cdssd_gsm_accs)),
        "avg_gsm_tps": float(np.mean(m1_cdssd_gsm_tps)),
        "avg_speedup": float(np.mean(m1_cdssd_speedups)),
        "avg_acc_ret": float(np.mean(m1_cdssd_rets)),
        "avg_qas":     float(np.mean(m1_cdssd_qas)),
        "source":      "full_pairwise_ortho (reused)",
    }

    # ── Decision rules ─────────────────────────────────────────────────────────
    cdssd_tau09_qas = condition_summaries["cdssd_tau09"]["avg_qas"]
    cdssd_tau00_qas = condition_summaries["cdssd_tau00"]["avg_qas"]
    naive_t16_qas   = condition_summaries["naive_t16"]["avg_qas"]
    m1_naive_t16_qas = condition_summaries["m1_naive_t16"]["avg_qas"]
    m1_cdssd_tau09_qas = condition_summaries["m1_cdssd_tau09"]["avg_qas"]

    # tau0_beats_naiveT16: does CD-SSD(tau=0.0) beat naive-T16?
    # Note: tau=0.0 means all tokens accepted (skip refine), should be nearly identical to naive-T16
    # but token scheduling during draft is top-k by confidence vs LLaDA's masked scheduling
    tau0_vs_naive = cdssd_tau00_qas - naive_t16_qas
    tau09_vs_naive = cdssd_tau09_qas - naive_t16_qas
    m1_cdssd_vs_m1_naive = m1_cdssd_tau09_qas - m1_naive_t16_qas

    # Ortho for M1+naive-T16
    # Individual QAS for M1 from full_m1_pareto: 0.836
    # Individual QAS for naive-T16 from this experiment
    m1_ref_qas = 0.836
    naive_t16_ind_qas = naive_t16_qas  # naive-T16 alone QAS
    m1_naive_ortho = m1_naive_t16_qas / max(m1_ref_qas, naive_t16_ind_qas, 1e-6)

    decisions = {
        "tau0_vs_naive_delta_qas": tau0_vs_naive,
        "tau0_beats_naiveT16": bool(tau0_vs_naive > 0.05),
        "tau0_approx_naiveT16": bool(abs(tau0_vs_naive) <= 0.05),
        "tau09_vs_naive_delta_qas": tau09_vs_naive,
        "tau09_beats_naiveT16": bool(tau09_vs_naive > 0.05),
        "m1_cdssd_vs_m1_naive_delta_qas": m1_cdssd_vs_m1_naive,
        "M1_cdssd_beats_M1_naiveT16": bool(m1_cdssd_vs_m1_naive > 0.05),
        "M1_naive_ortho": m1_naive_ortho,
        "M1_naiveT16_ortho_gt_1": bool(m1_naive_ortho > 1.0),
        "interpretation": (
            "CD-SSD accept gate adds value beyond step-reduction"
            if cdssd_tau09_qas > cdssd_tau00_qas + 0.05
            else "CD-SSD can be reframed as a step-reduction method"
        ),
    }

    # ── Print summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"[{TASK_ID}] === TAU=0.0 PARADOX RESOLUTION SUMMARY ===")
    print(f"{'Condition':<25} {'Acc':>7} {'TPS':>7} {'Speedup':>8} {'AccRet':>7} {'QAS':>7}")
    print("-" * 70)
    order = ["cdssd_tau09", "cdssd_tau00", "naive_t16", "m1_cdssd_tau09", "m1_naive_t16"]
    for cond_id in order:
        s = condition_summaries[cond_id]
        print(f"  {s['name']:<23} {s['avg_gsm_acc']:>7.4f} {s['avg_gsm_tps']:>7.1f} "
              f"{s['avg_speedup']:>8.3f}x {s['avg_acc_ret']:>7.3f} {s['avg_qas']:>7.3f}")
    print("=" * 70)
    print("\nDecision rules:")
    print(f"  tau=0.0 vs naive-T16 QAS delta: {tau0_vs_naive:+.3f} → "
          f"{'tau=0.0 beats naive' if tau0_vs_naive > 0.05 else 'approx equal or naive wins'}")
    print(f"  tau=0.9 vs naive-T16 QAS delta: {tau09_vs_naive:+.3f} → "
          f"{'accept gate adds value' if tau09_vs_naive > 0.05 else 'accept gate marginal'}")
    print(f"  M1+CDSSD vs M1+naive delta: {m1_cdssd_vs_m1_naive:+.3f} → "
          f"{'CDSSD frozen-token synergy confirmed' if m1_cdssd_vs_m1_naive > 0.05 else 'step-reduction drives synergy'}")
    print(f"  M1+naive-T16 Ortho: {m1_naive_ortho:.3f} → "
          f"{'SYNERGY (>1.0)' if m1_naive_ortho > 1.0 else 'sub-multiplicative'}")

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "seeds": SEEDS,
        "n_gsm8k": N_GSM8K,
        "baseline": BASELINE,
        "igsd_params": {"t_draft": IGSD_T_DRAFT, "t_full": IGSD_T_FULL, "gen_length": GEN_LENGTH},
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": round(elapsed_min, 2),
        "condition_summaries": condition_summaries,
        "per_seed_results": all_results,
        "decisions": decisions,
        "existing_m1_cdssd_pairwise": EXISTING_M1_CDSSD,
    }

    out_path = RESULTS_DIR / "full_tau0_comparison.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n[{TASK_ID}] Results saved to {out_path}")

    mark_done(
        status="success",
        summary=(
            f"tau0_comparison: CDSSD(tau=0.9)_QAS={cdssd_tau09_qas:.3f}, "
            f"CDSSD(tau=0.0)_QAS={cdssd_tau00_qas:.3f}, "
            f"naive-T16_QAS={naive_t16_qas:.3f}, "
            f"M1+naive-T16_QAS={m1_naive_t16_qas:.3f}, "
            f"M1+CDSSD(tau=0.9)_QAS={m1_cdssd_tau09_qas:.3f}; "
            f"Decisions: tau0_beats_naive={decisions['tau0_beats_naiveT16']}, "
            f"M1_naive_ortho={m1_naive_ortho:.3f}"
        )
    )
    # Also update gpu_progress.json
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else \
            {"completed": [], "failed": [], "running": {}, "timings": {}}
        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)
        gp["running"].pop(TASK_ID, None)
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 90, "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(), "end_time": end_time.isoformat(),
            "config_snapshot": {"n_gsm8k": N_GSM8K, "n_seeds": len(SEEDS),
                                 "n_conditions": len(CONDITIONS), "gen_length": GEN_LENGTH}
        }
        gp_path.write_text(json.dumps(gp, indent=2))
    except Exception as e:
        print(f"[{TASK_ID}] Warning: could not update gpu_progress.json: {e}")

    print(f"[{TASK_ID}] Done in {elapsed_min:.1f} min.")


if __name__ == "__main__":
    main()
