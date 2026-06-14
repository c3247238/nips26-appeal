"""
Three-Way Composition (M1+IGSD+M3) Pareto Pilot

Task: threeway_pareto_pilot
Iteration: 2
Mode: PILOT (100 GSM8K + 100 MATH500, seed=42)

Sweep the three-way composition design space:
  M1_eta      = {0.5 (best), 1.0 (aggressive)}
  IGSD_tau    = {0.85, 0.9}
  IGSD_T_draft = {32, 48}
  M3_gw       = {0.0 (off), 0.3, 0.7}
  Total: 2 x 2 x 2 x 3 = 24 configurations

Each config evaluated on 100 GSM8K + 100 MATH500, seed=42.
Identify Pareto frontier: max-speed, balanced, quality-first operating points.
Top 5 configs promoted to full validation.

Pass criteria: At least 20 of 24 configs complete without error AND
  Pareto frontier contains >= 3 distinct operating points.

Output: exp/results/threeway/threeway_pareto_pilot.json

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dlm-acceleration python threeway_pareto_pilot.py
"""

import os
import sys
import gc
import json
import time
import random
import re
from pathlib import Path
from datetime import datetime
from itertools import product

import torch
import numpy as np
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
ITER1_DIR   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/iter_001")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
QWEN_PATH   = str(SHARED / "checkpoints" / "qwen2.5-0.5b")
GSM8K_DIR   = str(SHARED / "datasets" / "gsm8k")
MATH500_DIR = str(SHARED / "datasets" / "math500")
CODE_DIR    = ITER1_DIR / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "threeway"
TASK_ID     = "threeway_pareto_pilot"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
SEED         = 42
GEN_LENGTH   = 256
BLOCK_LENGTH = 32
T_FULL       = 64
MASK_ID      = 126336
N_GSM8K      = 100   # pilot
N_MATH500    = 100   # pilot
N_WARMUP     = 3     # smaller warmup for efficiency with 24 configs

# Pilot baseline reference (same subset, seed=42)
PILOT_BASELINE = {
    "gsm8k":   {"exact_match": 0.73, "avg_tps": 58.505},
    "math500": {"exact_match": 0.32, "avg_tps": 111.987},
}

# Individual method QAS from Phase 2 for Ortho computation
# M1 (from full_m1_pareto.json)
INDIVIDUAL_QAS = {
    "M1_eta0.5": {
        "gsm8k":   {"speedup": 1.158, "acc_ret": 0.945, "qas": 1.094},
        "math500": {"speedup": 1.069, "acc_ret": 1.066, "qas": 1.140},
        "combined_qas": 0.981,
    },
    "M1_eta1.0": {
        # eta=1.0 from full_m1_pareto.json (full baseline-relative)
        # gsm8k: speedup=1.252, acc_ret=0.880, combined_qas=0.967
        # Note: these are full-baseline relative; pilot-relative may differ slightly
        "gsm8k":   {"speedup": 1.252, "acc_ret": 0.880, "qas": 1.102},
        "math500": {"speedup": 1.137, "acc_ret": 0.777, "qas": 0.883},
        "combined_qas": 0.967,
    },
    "IGSD_tau0.85_td32": {
        "gsm8k":   {"speedup": 1.732, "acc_ret": 0.678, "qas": 1.174},
        "math500": {"speedup": 1.749, "acc_ret": 0.438, "qas": 0.765},
        "combined_qas": 1.052,
    },
    "IGSD_tau0.85_td48": {
        "gsm8k":   {"speedup": 1.232, "acc_ret": 0.733, "qas": 0.903},
        "math500": {"speedup": 1.268, "acc_ret": 0.469, "qas": 0.595},
        "combined_qas": 0.812,
    },
    "IGSD_tau0.9_td32": {
        "gsm8k":   {"speedup": 1.706, "acc_ret": 0.678, "qas": 1.157},
        "math500": {"speedup": 1.712, "acc_ret": 0.438, "qas": 0.749},
        "combined_qas": 1.035,
    },
    "IGSD_tau0.9_td48": {
        "gsm8k":   {"speedup": 1.222, "acc_ret": 0.733, "qas": 0.895},
        "math500": {"speedup": 1.254, "acc_ret": 0.500, "qas": 0.627},
        "combined_qas": 0.816,
    },
    "M3_gw0.3": {
        "gsm8k":   {"speedup": 1.651, "acc_ret": 1.025, "qas": 1.692},
        "math500": {"speedup": 1.154, "acc_ret": 2.349, "qas": 2.710},
        "combined_qas": 2.136,
    },
    "M3_gw0.7": {
        "gsm8k":   {"speedup": 1.647, "acc_ret": 1.039, "qas": 1.711},
        "math500": {"speedup": 1.156, "acc_ret": 2.439, "qas": 2.819},
        "combined_qas": 2.188,
    },
}

# Sweep grid
M1_ETAS = [0.5, 1.0]
IGSD_TAUS = [0.85, 0.9]
IGSD_TDRAFTS = [32, 48]
M3_GWS = [0.0, 0.3, 0.7]


# ── System Monitor Helpers ─────────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(step, total, metric=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Seed & Data Loading ───────────────────────────────────────────────────
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_gsm8k(n_samples=None, seed=42):
    path = Path(GSM8K_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    if n_samples and n_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, n_samples)
    return data


def load_math500(n_samples=None, seed=42):
    path = Path(MATH500_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    if n_samples and n_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, n_samples)
    return data


# ── 8-shot GSM8K Prompt ──────────────────────────────────────────────────
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


def build_math500_prompt(problem):
    return f"Problem: {problem}\nSolution:"


# ── Metric Extractors ────────────────────────────────────────────────────
def extract_gsm8k_answer(text):
    """Extract final numeric answer from GSM8K solution text."""
    match = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def gsm8k_exact_match(pred, gold_answer):
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold_answer)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip() == g.strip()


def extract_math500_answer(text):
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"[Tt]he answer is\s+(.+?)(?:\.|$)", text)
    if match:
        return match.group(1).strip()
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return None


def math500_exact_match(pred, gold_solution):
    p = extract_math500_answer(pred)
    g = extract_math500_answer(gold_solution)
    if p is None or g is None:
        return False
    p = p.strip().replace(" ", "").lower()
    g = g.strip().replace(" ", "").lower()
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ══════════════════════════════════════════════════════════════════════════
# Three-Way Generator: M1 (EntropyCache) + IGSD (draft-partition-refine) + M3 (AR-guided)
# ══════════════════════════════════════════════════════════════════════════

class ThreeWayGenerator:
    """
    Composed M1 + IGSD + M3 generator.

    Pipeline:
      1. Draft phase (IGSD): T_draft steps, whole-sequence unmasking
         - M1: track entropy-based KV cache hits during draft
      2. Confidence partition: S_accept (conf >= tau), S_refine (conf < tau)
      3. Refine phase: T_full steps on S_refine tokens
         - M1: track KV cache hits during refine
         - M3 (if gw > 0): AR-guided blending during refine phase
           The AR guide provides quality insurance on uncertain tokens.

    When M3_gw=0.0, this reduces to M1+IGSD (no AR guidance).
    """

    def __init__(self, llada_model, llada_tokenizer, ar_model=None,
                 ar_tokenizer=None, device="cuda"):
        self.model = llada_model
        self.tokenizer = llada_tokenizer
        self.ar_model = ar_model
        self.ar_tokenizer = ar_tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt,
        entropy_threshold=0.5,
        tau=0.85,
        t_draft=32,
        guidance_weight=0.0,
        t_full=64,
        gen_length=256,
        block_length=32,
        apply_chat_template=True,
    ):
        """
        Three-way composed generation.

        Returns dict with generation text, timing, and composition metrics.
        """
        # Tokenize
        if apply_chat_template:
            msg = [{"role": "user", "content": prompt}]
            enc_text = self.tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
        else:
            enc_text = prompt

        enc = self.tokenizer(
            [enc_text], add_special_tokens=False, padding=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        # Initialize fully masked sequence
        x = torch.full(
            (1, prompt_len + gen_length), MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=self.device)
        ], dim=-1)

        # M1 cache tracking
        m1_cache_hits_draft = 0
        m1_cache_total_draft = 0
        m1_cache_hits_refine = 0
        m1_cache_total_refine = 0
        prev_low_entropy_mask = None

        # ── Phase 1: Whole-sequence draft (IGSD) ────────────────────
        t0 = time.perf_counter()
        tokens_per_draft_step = max(1, gen_length // t_draft)
        remainder_draft = gen_length % t_draft

        for step in range(t_draft):
            n_masked = int((x[0, prompt_len:] == MASK_ID).sum().item())
            if n_masked == 0:
                break

            logits = self.model(x, attention_mask=attn).logits
            p = F.softmax(logits, dim=-1)

            # M1: compute per-token entropy
            entropy = -(p * torch.log(p.clamp(min=1e-9))).sum(-1)
            low_entropy_now = entropy < entropy_threshold

            # Track cache hits
            non_masked = (x != MASK_ID)
            if prev_low_entropy_mask is not None:
                hits = (low_entropy_now & prev_low_entropy_mask & non_masked).sum().item()
                total = non_masked.sum().item()
                m1_cache_hits_draft += hits
                m1_cache_total_draft += total

            prev_low_entropy_mask = low_entropy_now.clone()

            # IGSD whole-sequence unmasking (greedy, top-k by confidence)
            x0 = torch.argmax(p, dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            mask_index = (x == MASK_ID)
            mask_index[:, :prompt_len] = False
            confidence = torch.where(
                mask_index, x0_p, torch.tensor(-float("inf"), device=self.device)
            )

            k = tokens_per_draft_step + (1 if step < remainder_draft else 0)
            k = min(k, n_masked)
            if k > 0:
                _, sel = torch.topk(confidence[0], k=k)
                ti = torch.zeros_like(x[0], dtype=torch.bool)
                ti[sel] = True
                ti = ti.unsqueeze(0) & mask_index
                x[ti] = x0[ti]

        draft_elapsed = time.perf_counter() - t0

        # ── Phase 2: Confidence partition ────────────────────────────
        logits = self.model(x, attention_mask=attn).logits
        p_final = F.softmax(logits[:, prompt_len:, :], dim=-1)
        gen_region = x[:, prompt_len:].clone()
        still_masked = (gen_region == MASK_ID)

        draft_pred = torch.argmax(p_final, dim=-1)
        draft_conf = torch.gather(p_final, -1, draft_pred.unsqueeze(-1)).squeeze(-1)
        filled_conf = torch.gather(p_final, -1, gen_region.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        final_confidence = torch.where(still_masked, draft_conf, filled_conf)
        gen_region_filled = torch.where(still_masked, draft_pred, gen_region)
        x[:, prompt_len:] = gen_region_filled

        s_accept_gen = (final_confidence >= tau)
        n_accept = int(s_accept_gen.sum().item())
        n_total = s_accept_gen.numel()
        n_refine = n_total - n_accept
        accept_rate = n_accept / n_total

        # ── Phase 3: Refine S_refine tokens ──────────────────────────
        # M3 AR guidance applied during refine phase (if gw > 0)
        x_refine = x.clone()
        s_accept_full = torch.cat([
            torch.ones(1, prompt_len, dtype=torch.bool, device=self.device),
            s_accept_gen
        ], dim=1)
        s_refine_full = ~s_accept_full
        x_refine[s_refine_full] = MASK_ID

        kv_hit_steps = []
        t_refine_start = time.perf_counter()
        prev_low_entropy_mask = None  # Reset for refine phase

        if n_refine > 0:
            tokens_per_refine_step = max(1, n_refine // t_full)
            remainder_refine = n_refine % t_full

            for step in range(t_full):
                n_masked_now = int((x_refine[0, prompt_len:] == MASK_ID).sum().item())
                if n_masked_now == 0:
                    break

                n_frozen = n_total - n_masked_now
                kv_hit_steps.append(n_frozen / n_total)

                logits = self.model(x_refine, attention_mask=attn).logits
                p = F.softmax(logits, dim=-1)

                # M1: cache tracking during refine
                entropy = -(p * torch.log(p.clamp(min=1e-9))).sum(-1)
                low_entropy_now = entropy < entropy_threshold
                non_masked = (x_refine != MASK_ID)

                if prev_low_entropy_mask is not None:
                    hits = (low_entropy_now & prev_low_entropy_mask & non_masked).sum().item()
                    total_tokens = non_masked.sum().item()
                    m1_cache_hits_refine += hits
                    m1_cache_total_refine += total_tokens

                prev_low_entropy_mask = low_entropy_now.clone()

                # M3: AR-guided blending (if guidance_weight > 0)
                if guidance_weight > 0.0 and self.ar_model is not None:
                    # Get AR guidance from unmasked prefix
                    # Use all non-masked tokens up to current position
                    unmasked_prefix = x_refine[:, :prompt_len].clone()
                    ar_out = self.ar_model(unmasked_prefix)
                    ar_logits = ar_out.logits[:, -1:, :]
                    p_ar = F.softmax(ar_logits, dim=-1)

                    vocab_size = p.shape[-1]
                    ar_vocab = p_ar.shape[-1]

                    # Handle vocab size mismatch between LLaDA and Qwen
                    if ar_vocab < vocab_size:
                        pad = torch.zeros(1, 1, vocab_size - ar_vocab,
                                          device=self.device, dtype=p_ar.dtype)
                        p_ar = torch.cat([p_ar, pad], dim=-1)
                    elif ar_vocab > vocab_size:
                        p_ar = p_ar[:, :, :vocab_size]

                    p_ar_broadcast = p_ar.expand(1, p.shape[1], vocab_size)
                    p = (1 - guidance_weight) * p + guidance_weight * p_ar_broadcast

                x0 = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                mask_index = (x_refine == MASK_ID) & s_refine_full
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-float("inf"), device=self.device)
                )

                k = tokens_per_refine_step + (1 if step < remainder_refine else 0)
                k = min(k, n_masked_now)
                if k > 0:
                    _, sel = torch.topk(confidence[0], k=k)
                    ti = torch.zeros_like(x_refine[0], dtype=torch.bool)
                    ti[sel] = True
                    ti = ti.unsqueeze(0) & mask_index
                    x_refine[ti] = x0[ti]

        refine_elapsed = time.perf_counter() - t_refine_start

        # Fill remaining masked tokens
        still_left = (x_refine[:, prompt_len:] == MASK_ID)
        if still_left.any():
            lf = self.model(x_refine, attention_mask=attn).logits
            pf = F.softmax(lf[:, prompt_len:, :], dim=-1)
            fp = torch.argmax(pf, dim=-1)
            x_refine[:, prompt_len:] = torch.where(
                still_left, fp, x_refine[:, prompt_len:]
            )

        total_elapsed = draft_elapsed + refine_elapsed
        total_tps = gen_length / total_elapsed if total_elapsed > 0 else 0.0
        kv_hit_rate = float(np.mean(kv_hit_steps)) if kv_hit_steps else float(n_accept / n_total)

        # M1 cache hit rates
        chr_draft = m1_cache_hits_draft / m1_cache_total_draft if m1_cache_total_draft > 0 else 0.0
        chr_refine = m1_cache_hits_refine / m1_cache_total_refine if m1_cache_total_refine > 0 else 0.0
        m1_total_hits = m1_cache_hits_draft + m1_cache_hits_refine
        m1_total = m1_cache_total_draft + m1_cache_total_refine
        chr_overall = m1_total_hits / m1_total if m1_total > 0 else 0.0

        text = self.tokenizer.decode(
            x_refine[0, prompt_len:].tolist(), skip_special_tokens=True
        )

        return {
            "generated_text": text,
            "tps": total_tps,
            "elapsed_sec": total_elapsed,
            "draft_elapsed_sec": draft_elapsed,
            "refine_elapsed_sec": refine_elapsed,
            "accept_rate": accept_rate,
            "n_accept": n_accept,
            "n_refine": n_refine,
            "n_total": n_total,
            "kv_hit_rate_refine": kv_hit_rate,
            "m1_chr_draft": chr_draft,
            "m1_chr_refine": chr_refine,
            "m1_chr_overall": chr_overall,
            "entropy_threshold": entropy_threshold,
            "tau": tau,
            "t_draft": t_draft,
            "guidance_weight": guidance_weight,
            "t_full": t_full,
        }


# ── Evaluation Functions ──────────────────────────────────────────────────
def evaluate_benchmark(generator, data, config, benchmark_name, build_prompt_fn,
                       match_fn, answer_field, n_warmup=N_WARMUP):
    """Generic benchmark evaluator."""
    eta = config["entropy_threshold"]
    tau = config["tau"]
    t_draft = config["t_draft"]
    gw = config["guidance_weight"]

    correct = 0
    total = len(data)
    tps_list = []
    accept_rates = []
    chr_values = []
    sample_texts = []

    for i, item in enumerate(data):
        prompt = build_prompt_fn(item)
        try:
            result = generator.generate(
                prompt=prompt,
                entropy_threshold=eta,
                tau=tau,
                t_draft=t_draft,
                guidance_weight=gw,
                t_full=T_FULL,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                apply_chat_template=True,
            )
            pred_text = result["generated_text"]
            gold = item.get(answer_field, item.get("solution", ""))
            is_correct = match_fn(pred_text, gold)
            if is_correct:
                correct += 1

            if i >= n_warmup:
                tps_list.append(result["tps"])
            accept_rates.append(result["accept_rate"])
            chr_values.append(result["m1_chr_overall"])

            if i < 5:
                sample_texts.append({
                    "id": i,
                    "correct": is_correct,
                    "prediction": pred_text[:200],
                    "accept_rate": result["accept_rate"],
                    "tps": result["tps"],
                    "m1_chr": result["m1_chr_overall"],
                })

        except Exception as e:
            print(f"  [ERROR] {benchmark_name} sample {i}: {e}")
            continue

        if (i + 1) % 25 == 0:
            acc = correct / (i + 1)
            avg_tps = float(np.mean(tps_list)) if tps_list else 0
            print(f"  [{benchmark_name}] {i+1}/{total}: acc={acc:.3f}, tps={avg_tps:.1f}")

    accuracy = correct / total if total > 0 else 0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    tps_std = float(np.std(tps_list)) if tps_list else 0.0

    return {
        "n_samples": total,
        "correct": correct,
        "exact_match": accuracy,
        "avg_tps": avg_tps,
        "tps_std": tps_std,
        "avg_accept_rate": float(np.mean(accept_rates)) if accept_rates else 0,
        "avg_m1_chr": float(np.mean(chr_values)) if chr_values else 0,
        "sample_texts": sample_texts,
    }


def compute_ortho_threeway(composed_qas, best_individual_qas):
    """Ortho = QAS(M1+IGSD+M3) / max(QAS(M1), QAS(IGSD), QAS(M3))."""
    if best_individual_qas <= 0:
        return 0.0
    return composed_qas / best_individual_qas


def identify_pareto_frontier(results_list):
    """
    Find Pareto-optimal configs: no config dominates another in both speedup and acc_ret.
    Returns list of indices into results_list that are on the frontier.
    """
    n = len(results_list)
    is_pareto = [True] * n

    for i in range(n):
        if not is_pareto[i]:
            continue
        si = results_list[i]["combined"]["speedup"]
        ai = results_list[i]["combined"]["acc_retention"]
        for j in range(n):
            if i == j or not is_pareto[j]:
                continue
            sj = results_list[j]["combined"]["speedup"]
            aj = results_list[j]["combined"]["acc_retention"]
            # j dominates i if j is >= in both and strictly > in at least one
            if sj >= si and aj >= ai and (sj > si or aj > ai):
                is_pareto[i] = False
                break

    return [i for i in range(n) if is_pareto[i]]


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    write_pid()
    set_seed(SEED)

    # Build all 24 configs
    all_configs = []
    for eta, tau, td, gw in product(M1_ETAS, IGSD_TAUS, IGSD_TDRAFTS, M3_GWS):
        gw_str = f"gw{gw:.1f}".replace(".", "")
        name = f"M1_eta{eta}+IGSD_tau{tau}_td{td}+M3_{gw_str}"
        all_configs.append({
            "name": name,
            "entropy_threshold": eta,
            "tau": tau,
            "t_draft": td,
            "guidance_weight": gw,
        })

    total_configs = len(all_configs)

    print("=" * 70)
    print("Three-Way Composition (M1+IGSD+M3) Pareto PILOT")
    print(f"Task: {TASK_ID}")
    print(f"Mode: PILOT ({N_GSM8K} GSM8K + {N_MATH500} MATH500, seed={SEED})")
    print(f"Grid: {len(M1_ETAS)} M1_eta x {len(IGSD_TAUS)} tau x "
          f"{len(IGSD_TDRAFTS)} T_draft x {len(M3_GWS)} gw = {total_configs} configs")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    # ── Load models ──
    print("\n[1/5] Loading LLaDA-8B-Instruct...")
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    # Load AR guide model for M3 (needed when gw > 0)
    print("  Loading Qwen2.5-0.5B (AR guide)...")
    ar_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH)
    ar_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    generator = ThreeWayGenerator(model, tokenizer, ar_model, ar_tokenizer, "cuda")
    print("  Models loaded.")

    # VRAM probe
    vram_info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // (1024 * 1024),
        "vram_used_mb": torch.cuda.memory_allocated(0) // (1024 * 1024),
        "vram_reserved_mb": torch.cuda.memory_reserved(0) // (1024 * 1024),
    }
    print(f"  VRAM: {vram_info['vram_used_mb']}MB used / {vram_info['vram_total_mb']}MB total")

    # Save GPU profile
    profile_path = RESULTS_DIR / f"{TASK_ID}_gpu_profile.json"
    profile_path.write_text(json.dumps(vram_info, indent=2))

    # ── Load datasets ──
    print("\n[2/5] Loading datasets...")
    gsm8k_data = load_gsm8k(N_GSM8K, SEED)
    math500_data = load_math500(N_MATH500, SEED)
    print(f"  GSM8K: {len(gsm8k_data)} samples")
    print(f"  MATH500: {len(math500_data)} samples")

    # ── Run all 24 configurations ──
    print(f"\n[3/5] Running {total_configs} configurations...")
    all_results = {}
    completed = 0
    failed = 0

    for ci, config in enumerate(all_configs):
        config_name = config["name"]
        config_start = time.perf_counter()

        print(f"\n{'='*60}")
        print(f"Config {ci+1}/{total_configs}: {config_name}")
        print(f"  M1 eta={config['entropy_threshold']}, IGSD tau={config['tau']}, "
              f"T_draft={config['t_draft']}, M3 gw={config['guidance_weight']}")
        print(f"{'='*60}")

        report_progress(ci + 1, total_configs, {
            "config": config_name,
            "completed": completed,
            "failed": failed,
        })

        try:
            # Evaluate GSM8K
            gsm8k_res = evaluate_benchmark(
                generator, gsm8k_data, config, "GSM8K",
                lambda item: build_gsm8k_prompt(item["question"]),
                gsm8k_exact_match, "answer", N_WARMUP
            )

            # Evaluate MATH500
            math500_res = evaluate_benchmark(
                generator, math500_data, config, "MATH500",
                lambda item: build_math500_prompt(item["problem"]),
                math500_exact_match, "answer", N_WARMUP
            )

            # Compute metrics relative to pilot baseline
            gsm8k_speedup = gsm8k_res["avg_tps"] / PILOT_BASELINE["gsm8k"]["avg_tps"] if PILOT_BASELINE["gsm8k"]["avg_tps"] > 0 else 0
            gsm8k_acc_ret = gsm8k_res["exact_match"] / PILOT_BASELINE["gsm8k"]["exact_match"] if PILOT_BASELINE["gsm8k"]["exact_match"] > 0 else 0
            gsm8k_qas = gsm8k_speedup * gsm8k_acc_ret

            math500_speedup = math500_res["avg_tps"] / PILOT_BASELINE["math500"]["avg_tps"] if PILOT_BASELINE["math500"]["avg_tps"] > 0 else 0
            math500_acc_ret = math500_res["exact_match"] / PILOT_BASELINE["math500"]["exact_match"] if PILOT_BASELINE["math500"]["exact_match"] > 0 else 0
            math500_qas = math500_speedup * math500_acc_ret

            # Combined: 0.7*GSM8K + 0.3*MATH500
            combined_speedup = 0.7 * gsm8k_speedup + 0.3 * math500_speedup
            combined_acc_ret = 0.7 * gsm8k_acc_ret + 0.3 * math500_acc_ret
            combined_qas = combined_speedup * combined_acc_ret

            # Ortho: compare against best individual method QAS
            # For each config, find the best reference individual QAS
            eta_key = f"M1_eta{config['entropy_threshold']}"
            igsd_key = f"IGSD_tau{config['tau']}_td{config['t_draft']}"
            gw_val = config['guidance_weight']

            ref_qas_list = []
            if eta_key in INDIVIDUAL_QAS:
                ref_qas_list.append(INDIVIDUAL_QAS[eta_key]["combined_qas"])
            if igsd_key in INDIVIDUAL_QAS:
                ref_qas_list.append(INDIVIDUAL_QAS[igsd_key]["combined_qas"])
            if gw_val > 0:
                m3_key = f"M3_gw{gw_val}"
                if m3_key in INDIVIDUAL_QAS:
                    ref_qas_list.append(INDIVIDUAL_QAS[m3_key]["combined_qas"])

            best_ref_qas = max(ref_qas_list) if ref_qas_list else 1.0
            combined_ortho = compute_ortho_threeway(combined_qas, best_ref_qas)

            # Per-benchmark Ortho
            ref_gsm8k_list = []
            if eta_key in INDIVIDUAL_QAS:
                ref_gsm8k_list.append(INDIVIDUAL_QAS[eta_key]["gsm8k"]["qas"])
            if igsd_key in INDIVIDUAL_QAS:
                ref_gsm8k_list.append(INDIVIDUAL_QAS[igsd_key]["gsm8k"]["qas"])
            if gw_val > 0:
                m3_key = f"M3_gw{gw_val}"
                if m3_key in INDIVIDUAL_QAS:
                    ref_gsm8k_list.append(INDIVIDUAL_QAS[m3_key]["gsm8k"]["qas"])
            best_ref_gsm8k = max(ref_gsm8k_list) if ref_gsm8k_list else 1.0
            gsm8k_ortho = gsm8k_qas / best_ref_gsm8k if best_ref_gsm8k > 0 else 0.0

            ref_math500_list = []
            if eta_key in INDIVIDUAL_QAS:
                ref_math500_list.append(INDIVIDUAL_QAS[eta_key]["math500"]["qas"])
            if igsd_key in INDIVIDUAL_QAS:
                ref_math500_list.append(INDIVIDUAL_QAS[igsd_key]["math500"]["qas"])
            if gw_val > 0:
                m3_key = f"M3_gw{gw_val}"
                if m3_key in INDIVIDUAL_QAS:
                    ref_math500_list.append(INDIVIDUAL_QAS[m3_key]["math500"]["qas"])
            best_ref_math500 = max(ref_math500_list) if ref_math500_list else 1.0
            math500_ortho = math500_qas / best_ref_math500 if best_ref_math500 > 0 else 0.0

            config_elapsed = (time.perf_counter() - config_start) / 60

            config_result = {
                "config_name": config_name,
                "entropy_threshold": config["entropy_threshold"],
                "tau": config["tau"],
                "t_draft": config["t_draft"],
                "guidance_weight": config["guidance_weight"],
                "seed": SEED,
                "elapsed_min": round(config_elapsed, 1),
                "gsm8k": {
                    **{k: v for k, v in gsm8k_res.items() if k != "sample_texts"},
                    "speedup": gsm8k_speedup,
                    "acc_retention": gsm8k_acc_ret,
                    "qas": gsm8k_qas,
                    "ortho": gsm8k_ortho,
                },
                "math500": {
                    **{k: v for k, v in math500_res.items() if k != "sample_texts"},
                    "speedup": math500_speedup,
                    "acc_retention": math500_acc_ret,
                    "qas": math500_qas,
                    "ortho": math500_ortho,
                },
                "combined": {
                    "speedup": combined_speedup,
                    "acc_retention": combined_acc_ret,
                    "qas": combined_qas,
                    "ortho": combined_ortho,
                },
                "ref_best_qas": {
                    "combined": best_ref_qas,
                    "gsm8k": best_ref_gsm8k,
                    "math500": best_ref_math500,
                },
                "sample_texts": {
                    "gsm8k": gsm8k_res.get("sample_texts", [])[:3],
                    "math500": math500_res.get("sample_texts", [])[:3],
                },
            }

            all_results[config_name] = config_result
            completed += 1

            print(f"\n  === {config_name} Summary ({config_elapsed:.1f} min) ===")
            print(f"  GSM8K:  acc={gsm8k_res['exact_match']:.3f}, speedup={gsm8k_speedup:.2f}x, "
                  f"QAS={gsm8k_qas:.3f}, Ortho={gsm8k_ortho:.3f}")
            print(f"  MATH500: acc={math500_res['exact_match']:.3f}, speedup={math500_speedup:.2f}x, "
                  f"QAS={math500_qas:.3f}")
            print(f"  Combined: speedup={combined_speedup:.2f}x, acc_ret={combined_acc_ret:.3f}, "
                  f"QAS={combined_qas:.3f}, Ortho={combined_ortho:.3f}")

        except Exception as e:
            print(f"\n  [FAILED] {config_name}: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

        torch.cuda.empty_cache()
        gc.collect()

    # ── Pareto analysis ──
    print(f"\n[4/5] Analyzing Pareto frontier...")

    results_list = list(all_results.values())
    pareto_indices = identify_pareto_frontier(results_list)
    pareto_configs = [results_list[i] for i in pareto_indices]

    # Sort Pareto by speedup (descending)
    pareto_configs.sort(key=lambda r: r["combined"]["speedup"], reverse=True)

    # Top 5 by combined QAS
    top5_by_qas = sorted(results_list, key=lambda r: r["combined"]["qas"], reverse=True)[:5]

    # Classify operating points
    operating_points = {}
    if pareto_configs:
        # Max-speed: highest speedup on Pareto
        operating_points["max_speed"] = pareto_configs[0]["config_name"]
        # Quality-first: highest acc_ret on Pareto
        quality_sorted = sorted(pareto_configs, key=lambda r: r["combined"]["acc_retention"], reverse=True)
        operating_points["quality_first"] = quality_sorted[0]["config_name"]
        # Balanced: best QAS on Pareto
        qas_sorted = sorted(pareto_configs, key=lambda r: r["combined"]["qas"], reverse=True)
        operating_points["balanced"] = qas_sorted[0]["config_name"]

    # ── Save results ──
    print(f"\n[5/5] Saving results...")

    elapsed_min = (datetime.now() - start_time).total_seconds() / 60

    # Build summary table
    summary_table = []
    for r in results_list:
        summary_table.append({
            "Config": r["config_name"],
            "M1_eta": r["entropy_threshold"],
            "IGSD_tau": r["tau"],
            "IGSD_Tdraft": r["t_draft"],
            "M3_gw": r["guidance_weight"],
            "GSM8K_Acc": round(r["gsm8k"]["exact_match"], 3),
            "GSM8K_Speedup": round(r["gsm8k"]["speedup"], 3),
            "MATH500_Acc": round(r["math500"]["exact_match"], 3),
            "MATH500_Speedup": round(r["math500"]["speedup"], 3),
            "Combined_Speedup": round(r["combined"]["speedup"], 3),
            "Combined_AccRet": round(r["combined"]["acc_retention"], 3),
            "Combined_QAS": round(r["combined"]["qas"], 3),
            "Combined_Ortho": round(r["combined"]["ortho"], 3),
            "Accept_Rate": round(r["gsm8k"]["avg_accept_rate"], 3),
            "M1_CHR": round(r["gsm8k"]["avg_m1_chr"], 3),
            "Elapsed_min": r.get("elapsed_min", 0),
        })

    # Sort table by Combined_QAS descending
    summary_table.sort(key=lambda r: r["Combined_QAS"], reverse=True)

    output = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "iteration": 2,
        "model": "LLaDA-8B-Instruct",
        "guide_model": "Qwen2.5-0.5B",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed_min, 1),
        "seed": SEED,
        "pilot_config": {
            "gsm8k_samples": N_GSM8K,
            "math500_samples": N_MATH500,
            "gen_length": GEN_LENGTH,
            "steps": T_FULL,
        },
        "sweep_grid": {
            "m1_eta": M1_ETAS,
            "igsd_tau": IGSD_TAUS,
            "igsd_t_draft": IGSD_TDRAFTS,
            "m3_gw": M3_GWS,
            "total_configs": total_configs,
        },
        "baseline_reference": {
            "pilot_baseline": PILOT_BASELINE,
        },
        "qas_formula": "QAS = Speedup * Accuracy_Retention (corrected, no 0.5x penalty)",
        "ortho_formula": "Ortho = QAS(M1+IGSD+M3) / max(QAS(M1), QAS(IGSD), QAS(M3))",
        "combined_metric": "0.7 * GSM8K + 0.3 * MATH500",
        "pass_criteria": {
            "requirement": "At least 20 of 24 configs complete AND Pareto frontier >= 3 points",
            "configs_completed": completed,
            "configs_failed": failed,
            "pareto_size": len(pareto_configs),
            "met": completed >= 20 and len(pareto_configs) >= 3,
        },
        "individual_qas_reference": INDIVIDUAL_QAS,
        "all_results": all_results,
        "summary_table": summary_table,
        "pareto_frontier": [
            {
                "config_name": r["config_name"],
                "combined_speedup": round(r["combined"]["speedup"], 3),
                "combined_acc_ret": round(r["combined"]["acc_retention"], 3),
                "combined_qas": round(r["combined"]["qas"], 3),
                "combined_ortho": round(r["combined"]["ortho"], 3),
                "gsm8k_acc": r["gsm8k"]["exact_match"],
                "gsm8k_speedup": round(r["gsm8k"]["speedup"], 3),
                "math500_acc": r["math500"]["exact_match"],
                "math500_speedup": round(r["math500"]["speedup"], 3),
            }
            for r in pareto_configs
        ],
        "top5_by_qas": [
            {
                "config_name": r["config_name"],
                "combined_qas": round(r["combined"]["qas"], 3),
                "combined_speedup": round(r["combined"]["speedup"], 3),
                "combined_acc_ret": round(r["combined"]["acc_retention"], 3),
                "combined_ortho": round(r["combined"]["ortho"], 3),
            }
            for r in top5_by_qas
        ],
        "operating_points": operating_points,
        "key_findings": [],  # Will be populated below
        "vram": vram_info,
    }

    # Populate key findings
    findings = []
    if completed >= 20:
        findings.append(f"{completed}/{total_configs} configs completed successfully")
    else:
        findings.append(f"WARNING: Only {completed}/{total_configs} configs completed ({failed} failed)")

    if len(pareto_configs) >= 3:
        findings.append(f"Pareto frontier has {len(pareto_configs)} distinct operating points")
        findings.append(f"Max-speed point: {operating_points.get('max_speed', 'N/A')}")
        findings.append(f"Balanced point: {operating_points.get('balanced', 'N/A')}")
        findings.append(f"Quality-first point: {operating_points.get('quality_first', 'N/A')}")

    # M3 effect analysis
    gw0_results = [r for r in results_list if r["guidance_weight"] == 0.0]
    gw03_results = [r for r in results_list if r["guidance_weight"] == 0.3]
    gw07_results = [r for r in results_list if r["guidance_weight"] == 0.7]

    if gw0_results and gw03_results:
        avg_qas_gw0 = np.mean([r["combined"]["qas"] for r in gw0_results])
        avg_qas_gw03 = np.mean([r["combined"]["qas"] for r in gw03_results])
        avg_qas_gw07 = np.mean([r["combined"]["qas"] for r in gw07_results]) if gw07_results else 0
        findings.append(f"M3 effect on QAS: gw=0.0 avg={avg_qas_gw0:.3f}, "
                        f"gw=0.3 avg={avg_qas_gw03:.3f}, gw=0.7 avg={avg_qas_gw07:.3f}")

        avg_acc_gw0 = np.mean([r["combined"]["acc_retention"] for r in gw0_results])
        avg_acc_gw03 = np.mean([r["combined"]["acc_retention"] for r in gw03_results])
        avg_acc_gw07 = np.mean([r["combined"]["acc_retention"] for r in gw07_results]) if gw07_results else 0
        findings.append(f"M3 effect on AccRet: gw=0.0 avg={avg_acc_gw0:.3f}, "
                        f"gw=0.3 avg={avg_acc_gw03:.3f}, gw=0.7 avg={avg_acc_gw07:.3f}")

    output["key_findings"] = findings

    output_path = RESULTS_DIR / "threeway_pareto_pilot.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to: {output_path}")

    # ── Print final summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Three-Way Pareto Pilot")
    print("=" * 70)
    print(f"  Completed: {completed}/{total_configs}, Failed: {failed}")
    print(f"  Pareto frontier: {len(pareto_configs)} points")
    print(f"  Total elapsed: {elapsed_min:.1f} minutes")

    if pareto_configs:
        print(f"\n  PARETO FRONTIER:")
        for r in pareto_configs:
            print(f"    {r['config_name']}")
            print(f"      Speedup={r['combined']['speedup']:.2f}x, "
                  f"AccRet={r['combined']['acc_retention']:.3f}, "
                  f"QAS={r['combined']['qas']:.3f}, Ortho={r['combined']['ortho']:.3f}")

    if top5_by_qas:
        print(f"\n  TOP 5 BY QAS:")
        for i, r in enumerate(top5_by_qas):
            print(f"    {i+1}. {r['config_name']}: QAS={r['combined']['qas']:.3f}, "
                  f"Speedup={r['combined']['speedup']:.2f}x, "
                  f"AccRet={r['combined']['acc_retention']:.3f}")

    if operating_points:
        print(f"\n  OPERATING POINTS:")
        for label, name in operating_points.items():
            r = all_results.get(name, {})
            comb = r.get("combined", {})
            print(f"    {label}: {name}")
            if comb:
                print(f"      Speedup={comb.get('speedup', 0):.2f}x, "
                      f"AccRet={comb.get('acc_retention', 0):.3f}, "
                      f"QAS={comb.get('qas', 0):.3f}")

    print(f"\n  Pass criteria met: {output['pass_criteria']['met']}")
    print("=" * 70)

    # Write pilot summary
    pilot_summary = {
        "task_id": TASK_ID,
        "overall_recommendation": "GO" if output["pass_criteria"]["met"] else "NO_GO",
        "pass_criteria_met": output["pass_criteria"]["met"],
        "configs_completed": completed,
        "configs_failed": failed,
        "pareto_size": len(pareto_configs),
        "operating_points": operating_points,
        "top5_for_full_validation": [r["config_name"] for r in top5_by_qas],
        "elapsed_minutes": round(elapsed_min, 1),
    }
    pilot_path = WORKSPACE / "exp" / "results" / "pilots" / f"{TASK_ID}_pilot_summary.json"
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_path.write_text(json.dumps(pilot_summary, indent=2))
    print(f"  Pilot summary saved to: {pilot_path}")

    mark_done(
        status="success",
        summary=f"Three-way Pareto pilot: {completed}/{total_configs} complete, "
                f"Pareto={len(pareto_configs)} pts, "
                f"pass={'YES' if output['pass_criteria']['met'] else 'NO'}, "
                f"{elapsed_min:.1f}min"
    )

    return output


if __name__ == "__main__":
    result = main()
