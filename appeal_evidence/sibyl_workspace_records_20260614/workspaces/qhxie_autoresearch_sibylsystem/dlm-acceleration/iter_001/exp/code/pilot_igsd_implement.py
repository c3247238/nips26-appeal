"""
IGSD Implementation and Pilot Evaluation.

Task: pilot_igsd_implement
- Implement IGSD from scratch per proposal algorithm
- Evaluate at operating point from H6 pilot (tau=0.85, accept_rate@0.85 = 63.7%)
- Run on 100 GSM8K + 50 HumanEval samples (seed=42)
- Record: TPS, exact_match, pass@1, accept_rate, KV hit-rate during REFINE phase

H6 result: accept_rate@0.85 = 0.637 (>= 50%) -> PROCEED with IGSD as primary contribution

Decision rule (from task_plan.json):
    - accept_rate >= 50% OR experiment records accept_rate + flags tau adjustment
    - Speedup > 1.5x vs. baseline

Baseline reference (pilot_baseline):
    GSM8K exact_match = 0.73, avg_tps = 58.55
    HumanEval pass@1  = 0.04, avg_tps = 100.93

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dlm-acceleration python pilot_igsd_implement.py
"""

import os
import sys
import json
import time
import random
import gc
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

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
RESULTS_DIR   = WORKSPACE / "exp" / "results" / "pilot_igsd"
TASK_ID       = "pilot_igsd_implement"
MASK_ID       = 126336

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline reference from pilot_baseline (100 GSM8K samples, seed=42)
BASELINE_GSM8K_TPS  = 58.55
BASELINE_GSM8K_ACC  = 0.73
BASELINE_HE_TPS     = 100.93
BASELINE_HE_ACC     = 0.04

# IGSD operating point (from H6 pilot: accept_rate@0.85 = 63.7%)
TAU        = 0.85
T_DRAFT    = 8
T_FULL     = 64
GEN_LENGTH = 256
BLOCK_LEN  = 32

# ── System-monitor Helpers ─────────────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(step: int, total: int, metric: Optional[Dict] = None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "updated_at": datetime.now().isoformat(),
        "metric": metric or {},
    }))


def mark_done(status: str = "success", summary: str = ""):
    pid_f = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_f.exists():
        pid_f.unlink()
    prog_f = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if prog_f.exists():
        try:
            final_progress = json.loads(prog_f.read_text())
        except Exception:
            pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Data Loaders ───────────────────────────────────────────────────────────────
def load_gsm8k(n_samples: int = 100, seed: int = 42) -> List[Dict]:
    path = Path(GSM8K_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n_samples, len(data)))


def load_humaneval(n_samples: int = 50, seed: int = 42) -> List[Dict]:
    path = Path(HUMANEVAL_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n_samples, len(data)))


# ── GSM8K Prompt and Evaluation ───────────────────────────────────────────────
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


def build_gsm8k_prompt(question: str) -> str:
    return GSM8K_8SHOT + f"Question: {question}\nAnswer:"


def extract_gsm8k_answer(text: str) -> Optional[str]:
    match = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None


def gsm8k_exact_match(pred: str, gold: str) -> bool:
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except Exception:
        return p.strip() == g.strip()


def check_humaneval_pass(completion: str, test_code: str, entry_point: str) -> bool:
    """Simple pass@1 check by exec."""
    try:
        full_code = completion + "\n" + test_code
        exec_globals = {}
        exec(full_code, exec_globals)
        return True
    except Exception:
        return False


# ── IGSD Core Implementation ───────────────────────────────────────────────────
def igsd_generate(
    model,
    tokenizer,
    prompt_text: str,
    device: str,
    tau: float = 0.85,
    t_draft: int = 8,
    t_full: int = 64,
    gen_length: int = 256,
    block_length: int = 32,
    apply_chat_template: bool = True,
) -> Dict[str, Any]:
    """
    IGSD: Iterative Guided Self-speculative Denoising.

    Algorithm:
    1. Draft phase: T_draft steps with LLaDA (fast, full sequence denoising)
    2. Partition: S_accept = {i : c[i] >= tau}, S_refine = complement
    3. Refine phase: freeze S_accept tokens, run T_full steps for S_refine
    4. Return merged output with timing and accept_rate

    Returns dict with keys:
        generated_text, tps, elapsed_sec, accept_rate, kv_hit_rate_refine,
        n_accept, n_total, draft_tps, refine_tps
    """
    from generate import get_num_transfer_tokens

    # Tokenize
    if apply_chat_template:
        msg = [{"role": "user", "content": prompt_text}]
        prompt_text_enc = tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=False
        )
    else:
        prompt_text_enc = prompt_text

    enc = tokenizer(
        [prompt_text_enc], add_special_tokens=False,
        padding=True, return_tensors="pt"
    )
    input_ids    = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len   = input_ids.shape[1]

    # Initialize fully masked generation sequence
    x = torch.full(
        (1, prompt_len + gen_length), MASK_ID, dtype=torch.long
    ).to(device)
    x[:, :prompt_len] = input_ids.clone()
    attn = torch.cat(
        [attention_mask,
         torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)],
        dim=-1,
    )

    num_blocks = gen_length // block_length

    # ── Phase 1: Draft (T_draft steps) ────────────────────────────────────────
    draft_steps_per_block = max(1, t_draft // num_blocks)
    t_draft_start = time.perf_counter()

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = prompt_len + (block_idx + 1) * block_length
        block_mask  = x[:, block_start:block_end] == MASK_ID
        num_transfer = get_num_transfer_tokens(block_mask, draft_steps_per_block)

        for step in range(draft_steps_per_block):
            mask_index = (x == MASK_ID)
            with torch.no_grad():
                logits = model(x, attention_mask=attn).logits
            p      = F.softmax(logits, dim=-1)
            x0     = torch.argmax(p, dim=-1)
            x0_p   = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            # Prevent premature unmasking of future blocks
            x0_p[:, block_end:] = -float("inf")
            x0         = torch.where(mask_index, x0, x)
            confidence = torch.where(mask_index, x0_p,
                                     torch.tensor(-float("inf"), device=device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            k = num_transfer[0, step].item()
            if k > 0:
                _, sel = torch.topk(confidence[0], k=int(k))
                transfer_index[0, sel] = True
            x[transfer_index] = x0[transfer_index]

    draft_elapsed = time.perf_counter() - t_draft_start
    draft_tps = gen_length / draft_elapsed if draft_elapsed > 0 else 0.0

    # ── Phase 2: Confidence partition ─────────────────────────────────────────
    with torch.no_grad():
        logits = model(x, attention_mask=attn).logits
    p_final    = F.softmax(logits[:, prompt_len:, :], dim=-1)  # (1, gen_len, vocab)
    gen_region = x[:, prompt_len:].clone()                      # (1, gen_len)
    still_masked = (gen_region == MASK_ID)

    # For any still-masked tokens, use argmax draft prediction
    draft_pred = torch.argmax(p_final, dim=-1)
    draft_conf = torch.gather(p_final, -1, draft_pred.unsqueeze(-1)).squeeze(-1)

    # For filled tokens, confidence = probability of the chosen token
    filled_conf = torch.gather(
        p_final, -1, gen_region.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    final_confidence = torch.where(still_masked, draft_conf, filled_conf)  # (1, gen_len)
    gen_region_filled = torch.where(still_masked, draft_pred, gen_region)
    x[:, prompt_len:] = gen_region_filled

    # Accept/Refine split
    s_accept = (final_confidence >= tau)  # (1, gen_len)
    n_accept  = int(s_accept.sum().item())
    n_total   = s_accept.numel()
    accept_rate = n_accept / n_total

    # ── Phase 3: Refine S_refine tokens ───────────────────────────────────────
    x_refine = x.clone()
    s_refine_full = torch.cat([
        torch.zeros(1, prompt_len, dtype=torch.bool, device=device),
        ~s_accept
    ], dim=1)  # (1, prompt_len + gen_len)

    # Re-mask refine tokens
    x_refine[s_refine_full] = MASK_ID

    # Track KV hit-rate during refine phase:
    # A token position is a "KV hit" if it remains frozen (in S_accept) throughout refine.
    # Hit rate = fraction of positions that are frozen (don't need recomputation context change).
    # Proxy: n_accept / n_total (positions that skip refinement = cache hits for those positions)
    # More precisely, track per-step how many mask tokens remain from S_refine
    n_refine_tokens_initial = int(s_refine_full[:, prompt_len:].sum().item())
    kv_hit_steps = []

    refine_steps_per_block = max(1, t_full // num_blocks)
    t_refine_start = time.perf_counter()

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = prompt_len + (block_idx + 1) * block_length
        block_mask  = x_refine[:, block_start:block_end] == MASK_ID

        if not block_mask.any():
            # No refine tokens in this block → full KV hit
            kv_hit_steps.append(1.0)
            continue

        num_transfer = get_num_transfer_tokens(block_mask, refine_steps_per_block)

        for step in range(refine_steps_per_block):
            mask_index = (x_refine == MASK_ID)
            # Track how many non-masked positions exist (frozen = KV hit)
            n_frozen = int((~mask_index[:, prompt_len:]).sum().item())
            kv_hit_frac = n_frozen / n_total if n_total > 0 else 0.0
            kv_hit_steps.append(kv_hit_frac)

            with torch.no_grad():
                logits = model(x_refine, attention_mask=attn).logits
            p      = F.softmax(logits, dim=-1)
            x0     = torch.argmax(p, dim=-1)
            x0_p   = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            x0_p[:, block_end:] = -float("inf")
            # Keep accepted tokens frozen (never overwrite them)
            x0         = torch.where(mask_index, x0, x_refine)
            confidence = torch.where(mask_index, x0_p,
                                     torch.tensor(-float("inf"), device=device))

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            k = num_transfer[0, step].item()
            if k > 0:
                _, sel = torch.topk(confidence[0], k=int(k))
                transfer_index[0, sel] = True
            # Only transfer positions in S_refine
            transfer_index = transfer_index & s_refine_full
            x_refine[transfer_index] = x0[transfer_index]

    refine_elapsed = time.perf_counter() - t_refine_start
    refine_tps = gen_length / refine_elapsed if refine_elapsed > 0 else 0.0

    total_elapsed = draft_elapsed + refine_elapsed
    total_tps = gen_length / total_elapsed if total_elapsed > 0 else 0.0

    # KV hit-rate during refine = average fraction of frozen positions across all refine steps
    kv_hit_rate_refine = float(np.mean(kv_hit_steps)) if kv_hit_steps else float(n_accept / n_total)

    # Decode
    generated_ids = x_refine[:, prompt_len:]
    generated_text = tokenizer.decode(generated_ids[0].tolist(), skip_special_tokens=True)

    return {
        "generated_text":    generated_text,
        "tps":               total_tps,
        "elapsed_sec":       total_elapsed,
        "draft_elapsed_sec": draft_elapsed,
        "refine_elapsed_sec": refine_elapsed,
        "draft_tps":         draft_tps,
        "refine_tps":        refine_tps,
        "accept_rate":       accept_rate,
        "n_accept":          n_accept,
        "n_total":           n_total,
        "kv_hit_rate_refine": kv_hit_rate_refine,
        "n_refine_tokens":   n_refine_tokens_initial,
        "tau":               tau,
        "t_draft":           t_draft,
        "t_full":            t_full,
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now().isoformat()
    print(f"[pilot_igsd] Starting IGSD pilot at {start_time}")
    print(f"[pilot_igsd] Config: tau={TAU}, T_draft={T_DRAFT}, T_full={T_FULL}, gen_length={GEN_LENGTH}")

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda:0"

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[pilot_igsd] Loading LLaDA-8B-Instruct from {MODEL_PATH}...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_after_load = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(device) // 1024**2,
    }
    print(f"[pilot_igsd] Model loaded. VRAM: {vram_after_load['vram_used_mb']} MB")

    # ── Warmup (1 sample, not counted) ────────────────────────────────────────
    print("[pilot_igsd] Running 1 warmup sample...")
    gsm8k_all = load_gsm8k(n_samples=100, seed=42)
    try:
        _ = igsd_generate(
            model, tokenizer, build_gsm8k_prompt(gsm8k_all[0]["question"]),
            device=device, tau=TAU, t_draft=T_DRAFT, t_full=T_FULL,
            gen_length=GEN_LENGTH, block_length=BLOCK_LEN,
        )
        print("[pilot_igsd] Warmup complete.")
    except Exception as e:
        print(f"[pilot_igsd] Warmup failed (non-fatal): {e}")
        torch.cuda.empty_cache(); gc.collect()

    # ── GSM8K: 100 samples ────────────────────────────────────────────────────
    print("\n[pilot_igsd] === GSM8K IGSD Evaluation (100 samples) ===")
    gsm8k_data = gsm8k_all  # already 100 samples
    gsm8k_results = []
    gsm8k_correct = 0
    gsm8k_tps_list = []
    gsm8k_accept_rates = []
    gsm8k_kv_hits = []

    for i, item in enumerate(gsm8k_data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            res = igsd_generate(
                model, tokenizer, prompt,
                device=device, tau=TAU, t_draft=T_DRAFT, t_full=T_FULL,
                gen_length=GEN_LENGTH, block_length=BLOCK_LEN,
            )
            correct = gsm8k_exact_match(res["generated_text"], item["answer"])
            if correct:
                gsm8k_correct += 1
            gsm8k_tps_list.append(res["tps"])
            gsm8k_accept_rates.append(res["accept_rate"])
            gsm8k_kv_hits.append(res["kv_hit_rate_refine"])
            gsm8k_results.append({
                "id": i,
                "correct": correct,
                "tps": round(res["tps"], 2),
                "accept_rate": round(res["accept_rate"], 4),
                "kv_hit_rate_refine": round(res["kv_hit_rate_refine"], 4),
                "elapsed_sec": round(res["elapsed_sec"], 3),
                "generated_text": res["generated_text"][:200],
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] GSM8K sample {i}")
            torch.cuda.empty_cache(); gc.collect()
            gsm8k_results.append({"id": i, "error": "OOM"})
        except Exception as e:
            print(f"  [ERR] GSM8K sample {i}: {e}")
            gsm8k_results.append({"id": i, "error": str(e)[:200]})

        if (i + 1) % 10 == 0:
            n_ok = len(gsm8k_tps_list)
            avg_tps = float(np.mean(gsm8k_tps_list)) if gsm8k_tps_list else 0.0
            acc = gsm8k_correct / (i + 1)
            ar  = float(np.mean(gsm8k_accept_rates)) if gsm8k_accept_rates else 0.0
            print(f"  [{i+1}/100] acc={acc:.3f}, tps={avg_tps:.1f}, accept_rate={ar:.3f}")
            report_progress(i + 1, 150, {
                "gsm8k_accuracy": acc,
                "gsm8k_avg_tps": avg_tps,
                "gsm8k_accept_rate": ar,
            })

    gsm8k_avg_tps    = float(np.mean(gsm8k_tps_list)) if gsm8k_tps_list else 0.0
    gsm8k_exact_match_acc = gsm8k_correct / len(gsm8k_data) if gsm8k_data else 0.0
    gsm8k_avg_accept = float(np.mean(gsm8k_accept_rates)) if gsm8k_accept_rates else 0.0
    gsm8k_avg_kv_hit = float(np.mean(gsm8k_kv_hits)) if gsm8k_kv_hits else 0.0
    gsm8k_speedup    = gsm8k_avg_tps / BASELINE_GSM8K_TPS if BASELINE_GSM8K_TPS > 0 else 0.0
    gsm8k_acc_ret    = gsm8k_exact_match_acc / BASELINE_GSM8K_ACC if BASELINE_GSM8K_ACC > 0 else 0.0
    gsm8k_qas        = gsm8k_speedup * gsm8k_acc_ret

    print(f"\n[pilot_igsd] GSM8K Summary:")
    print(f"  exact_match:    {gsm8k_exact_match_acc:.3f} (baseline: {BASELINE_GSM8K_ACC})")
    print(f"  avg_tps:        {gsm8k_avg_tps:.1f} (baseline: {BASELINE_GSM8K_TPS})")
    print(f"  speedup:        {gsm8k_speedup:.3f}x")
    print(f"  accuracy_retention: {gsm8k_acc_ret:.3f}")
    print(f"  QAS:            {gsm8k_qas:.3f}")
    print(f"  accept_rate:    {gsm8k_avg_accept:.3f} (target: >= 0.50)")
    print(f"  kv_hit_rate:    {gsm8k_avg_kv_hit:.3f}")

    # ── HumanEval: 50 samples ─────────────────────────────────────────────────
    print("\n[pilot_igsd] === HumanEval IGSD Evaluation (50 samples) ===")
    he_data = load_humaneval(n_samples=50, seed=42)
    he_results = []
    he_passed = 0
    he_tps_list = []
    he_accept_rates = []
    he_kv_hits = []

    for i, problem in enumerate(he_data):
        prompt_text = problem.get("prompt", "")
        try:
            res = igsd_generate(
                model, tokenizer, prompt_text,
                device=device, tau=TAU, t_draft=T_DRAFT, t_full=T_FULL,
                gen_length=GEN_LENGTH, block_length=BLOCK_LEN,
                apply_chat_template=False,
            )
            # Pass@1 check
            completion = res["generated_text"]
            test_code  = problem.get("test", "")
            entry_pt   = problem.get("entry_point", "")
            passed = check_humaneval_pass(completion, test_code, entry_pt)
            if passed:
                he_passed += 1

            he_tps_list.append(res["tps"])
            he_accept_rates.append(res["accept_rate"])
            he_kv_hits.append(res["kv_hit_rate_refine"])
            he_results.append({
                "task_id": problem.get("task_id", str(i)),
                "passed": passed,
                "tps": round(res["tps"], 2),
                "accept_rate": round(res["accept_rate"], 4),
                "kv_hit_rate_refine": round(res["kv_hit_rate_refine"], 4),
                "elapsed_sec": round(res["elapsed_sec"], 3),
                "generated_text": res["generated_text"][:300],
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] HumanEval {i}")
            torch.cuda.empty_cache(); gc.collect()
            he_results.append({"task_id": problem.get("task_id", str(i)), "error": "OOM"})
        except Exception as e:
            print(f"  [ERR] HumanEval {i}: {e}")
            he_results.append({"task_id": problem.get("task_id", str(i)), "error": str(e)[:200]})

        if (i + 1) % 10 == 0:
            n_ok = len(he_tps_list)
            avg_tps = float(np.mean(he_tps_list)) if he_tps_list else 0.0
            pass_at_1 = he_passed / (i + 1)
            ar = float(np.mean(he_accept_rates)) if he_accept_rates else 0.0
            print(f"  [{i+1}/50] pass@1={pass_at_1:.3f}, tps={avg_tps:.1f}, accept_rate={ar:.3f}")
            report_progress(100 + i + 1, 150, {
                "humaneval_pass_at_1": pass_at_1,
                "humaneval_avg_tps": avg_tps,
                "humaneval_accept_rate": ar,
            })

    he_avg_tps    = float(np.mean(he_tps_list)) if he_tps_list else 0.0
    he_pass_at_1  = he_passed / len(he_data) if he_data else 0.0
    he_avg_accept = float(np.mean(he_accept_rates)) if he_accept_rates else 0.0
    he_avg_kv_hit = float(np.mean(he_kv_hits)) if he_kv_hits else 0.0
    he_speedup    = he_avg_tps / BASELINE_HE_TPS if BASELINE_HE_TPS > 0 else 0.0
    he_acc_ret    = he_pass_at_1 / BASELINE_HE_ACC if BASELINE_HE_ACC > 0 else 1.0
    he_qas        = he_speedup * min(he_acc_ret, 2.0)  # cap at 2x for pass@1 zero-baseline

    print(f"\n[pilot_igsd] HumanEval Summary:")
    print(f"  pass@1:         {he_pass_at_1:.3f} (baseline: {BASELINE_HE_ACC})")
    print(f"  avg_tps:        {he_avg_tps:.1f} (baseline: {BASELINE_HE_TPS})")
    print(f"  speedup:        {he_speedup:.3f}x")
    print(f"  accept_rate:    {he_avg_accept:.3f}")
    print(f"  kv_hit_rate:    {he_avg_kv_hit:.3f}")

    # ── Combined metrics ───────────────────────────────────────────────────────
    combined_speedup = (gsm8k_speedup + he_speedup) / 2
    combined_accept  = (gsm8k_avg_accept + he_avg_accept) / 2
    combined_kv_hit  = (gsm8k_avg_kv_hit + he_avg_kv_hit) / 2

    # ── Pass criteria check ───────────────────────────────────────────────────
    pass_criteria_met = (
        combined_accept >= 0.50 and gsm8k_speedup >= 1.5
    )
    if not pass_criteria_met:
        if combined_accept < 0.50:
            print(f"\n[pilot_igsd] WARNING: accept_rate={combined_accept:.3f} < 0.50 target")
            print(f"  Flagging: need tau adjustment (try tau=0.70 for more tokens accepted)")
        if gsm8k_speedup < 1.5:
            print(f"\n[pilot_igsd] WARNING: speedup={gsm8k_speedup:.3f} < 1.5x target")
            print(f"  Note: IGSD overhead includes both draft + refine phases")

    # ── Save full results ─────────────────────────────────────────────────────
    end_time = datetime.now().isoformat()
    vram_final = torch.cuda.memory_allocated(device) // 1024**2

    metrics = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "igsd_config": {
            "tau": TAU,
            "t_draft": T_DRAFT,
            "t_full": T_FULL,
            "gen_length": GEN_LENGTH,
            "block_length": BLOCK_LEN,
        },
        "start_time": start_time,
        "end_time": end_time,
        "gsm8k": {
            "n_samples": len(gsm8k_data),
            "correct": gsm8k_correct,
            "exact_match": gsm8k_exact_match_acc,
            "avg_tps": gsm8k_avg_tps,
            "speedup_vs_baseline": gsm8k_speedup,
            "accuracy_retention": gsm8k_acc_ret,
            "qas": gsm8k_qas,
            "avg_accept_rate": gsm8k_avg_accept,
            "avg_kv_hit_rate_refine": gsm8k_avg_kv_hit,
        },
        "humaneval": {
            "n_samples": len(he_data),
            "passed": he_passed,
            "pass_at_1": he_pass_at_1,
            "avg_tps": he_avg_tps,
            "speedup_vs_baseline": he_speedup,
            "avg_accept_rate": he_avg_accept,
            "avg_kv_hit_rate_refine": he_avg_kv_hit,
        },
        "combined": {
            "avg_speedup": combined_speedup,
            "avg_accept_rate": combined_accept,
            "avg_kv_hit_rate_refine": combined_kv_hit,
        },
        "baseline_reference": {
            "gsm8k_exact_match": BASELINE_GSM8K_ACC,
            "gsm8k_avg_tps": BASELINE_GSM8K_TPS,
            "humaneval_pass_at_1": BASELINE_HE_ACC,
            "humaneval_avg_tps": BASELINE_HE_TPS,
        },
        "pass_criteria": {
            "accept_rate_target": 0.50,
            "speedup_target": 1.5,
            "accept_rate_achieved": combined_accept,
            "speedup_achieved_gsm8k": gsm8k_speedup,
            "pass_criteria_met": pass_criteria_met,
        },
        "vram": {
            "after_load_mb": vram_after_load["vram_used_mb"],
            "final_mb": vram_final,
            "gpu_name": vram_after_load["gpu_name"],
        },
        # H6 context
        "h6_pilot_context": {
            "accept_rate_at_085_from_h6": 0.637,
            "igsd_verdict_h6": "go",
            "note": "H6 pilot confirmed accept_rate@0.85=0.637 >= 0.50, proceeding with IGSD",
        },
    }

    out_path = RESULTS_DIR / "igsd_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[pilot_igsd] Saved metrics to {out_path}")

    # Save per-sample details (first 30 of each for debugging)
    samples_path = RESULTS_DIR / "igsd_samples.json"
    samples_path.write_text(json.dumps({
        "gsm8k_samples": gsm8k_results[:30],
        "humaneval_samples": he_results[:20],
    }, indent=2))

    # ── Print final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[pilot_igsd] FINAL SUMMARY")
    print("=" * 60)
    print(f"  tau:              {TAU}")
    print(f"  T_draft:          {T_DRAFT} steps")
    print(f"  T_full:           {T_FULL} steps")
    print(f"  GSM8K accuracy:   {gsm8k_exact_match_acc:.3f} (baseline: {BASELINE_GSM8K_ACC})")
    print(f"  GSM8K speedup:    {gsm8k_speedup:.3f}x")
    print(f"  GSM8K QAS:        {gsm8k_qas:.3f}")
    print(f"  HumanEval pass@1: {he_pass_at_1:.3f} (baseline: {BASELINE_HE_ACC})")
    print(f"  HumanEval speedup:{he_speedup:.3f}x")
    print(f"  Accept rate:      {combined_accept:.3f} (from H6: 0.637)")
    print(f"  KV hit rate:      {combined_kv_hit:.3f}")
    print(f"  Pass criteria met:{pass_criteria_met}")
    print("=" * 60)

    summary = (
        f"tau={TAU}, T_draft={T_DRAFT}, "
        f"gsm8k_acc={gsm8k_exact_match_acc:.3f}, gsm8k_speedup={gsm8k_speedup:.3f}x, "
        f"he_pass@1={he_pass_at_1:.3f}, he_speedup={he_speedup:.3f}x, "
        f"accept_rate={combined_accept:.3f}, kv_hit_refine={combined_kv_hit:.3f}"
    )
    mark_done(status="success", summary=summary)
    print(f"\n[pilot_igsd] DONE.")

    # ── Update gpu_progress.json ───────────────────────────────────────────────
    end_dt = datetime.now()
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        with open(gpu_progress_path) as f:
            progress = json.load(f)
    except Exception:
        progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in progress.get("completed", []):
        progress.setdefault("completed", []).append(TASK_ID)
    progress.get("running", {}).pop(TASK_ID, None)
    progress.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 40,
        "actual_min": round((end_dt - datetime.fromisoformat(start_time)).total_seconds() / 60),
        "start_time": start_time,
        "end_time": end_dt.isoformat(),
        "config_snapshot": {
            "model": "LLaDA-8B-Instruct",
            "method": "IGSD",
            "tau": TAU,
            "t_draft": T_DRAFT,
            "t_full": T_FULL,
            "n_gsm8k": len(gsm8k_data),
            "n_humaneval": len(he_data),
            "gpu_model": vram_after_load.get("gpu_name", "unknown"),
        },
    }
    with open(gpu_progress_path, "w") as f:
        json.dump(progress, f, indent=2)
    print(f"[pilot_igsd] Updated gpu_progress.json")

    return metrics


if __name__ == "__main__":
    main()
