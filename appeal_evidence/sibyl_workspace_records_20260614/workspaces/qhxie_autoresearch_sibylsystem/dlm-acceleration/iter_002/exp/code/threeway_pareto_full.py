"""
Three-Way Composition Top-5 Full Validation (PILOT mode)

Task: threeway_pareto_full
Iteration: 2
Mode: PILOT (100 GSM8K + 100 MATH500, seeds=[42, 123, 456])

Full-scale validation of top 5 configurations from threeway_pareto_pilot.
Run on 100 GSM8K + 100 MATH500 (pilot), seeds=[42, 123, 456].
Report per-seed stability and final Pareto operating points.
These become the 'acceleration recipes' recommended in the paper.

Top 5 configs (by combined QAS from pilot):
  1. M1_eta0.5+IGSD_tau0.85_td32+M3_gw00  (QAS=0.952)
  2. M1_eta1.0+IGSD_tau0.9_td32+M3_gw00   (QAS=0.952)
  3. M1_eta1.0+IGSD_tau0.85_td32+M3_gw00  (QAS=0.947)
  4. M1_eta0.5+IGSD_tau0.9_td32+M3_gw00   (QAS=0.940)
  5. M1_eta0.5+IGSD_tau0.85_td32+M3_gw03  (QAS=0.934)

Pass criteria: All 5 configs complete on 3 seeds AND per-seed QAS std < 30% of mean.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 conda run -n sibyl_dlm-acceleration python threeway_pareto_full.py

Output: exp/results/threeway/threeway_pareto_full.json
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
TASK_ID     = "threeway_pareto_full"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS        = [42, 123, 456]
GEN_LENGTH   = 256
BLOCK_LENGTH = 32
T_FULL       = 64
MASK_ID      = 126336
N_GSM8K      = 100   # pilot
N_MATH500    = 100   # pilot
N_WARMUP     = 3

# Pilot baseline reference (from threeway_pareto_pilot.json, seed=42)
PILOT_BASELINE = {
    "gsm8k":   {"exact_match": 0.73, "avg_tps": 58.505},
    "math500": {"exact_match": 0.32, "avg_tps": 111.987},
}

# Individual method QAS from Phase 2 for Ortho computation (same as pilot)
INDIVIDUAL_QAS = {
    "M1_eta0.5": {
        "gsm8k":   {"speedup": 1.158, "acc_ret": 0.945, "qas": 1.094},
        "math500": {"speedup": 1.069, "acc_ret": 1.066, "qas": 1.140},
        "combined_qas": 0.981,
    },
    "M1_eta1.0": {
        "gsm8k":   {"speedup": 1.252, "acc_ret": 0.880, "qas": 1.102},
        "math500": {"speedup": 1.137, "acc_ret": 0.777, "qas": 0.883},
        "combined_qas": 0.967,
    },
    "IGSD_tau0.85_td32": {
        "gsm8k":   {"speedup": 1.732, "acc_ret": 0.678, "qas": 1.174},
        "math500": {"speedup": 1.749, "acc_ret": 0.438, "qas": 0.765},
        "combined_qas": 1.052,
    },
    "IGSD_tau0.9_td32": {
        "gsm8k":   {"speedup": 1.706, "acc_ret": 0.678, "qas": 1.157},
        "math500": {"speedup": 1.712, "acc_ret": 0.438, "qas": 0.749},
        "combined_qas": 1.035,
    },
    "M3_gw0.3": {
        "gsm8k":   {"speedup": 1.651, "acc_ret": 1.025, "qas": 1.692},
        "math500": {"speedup": 1.154, "acc_ret": 2.349, "qas": 2.710},
        "combined_qas": 2.136,
    },
}

# Top 5 configs from pilot
TOP5_CONFIGS = [
    {"name": "M1_eta0.5+IGSD_tau0.85_td32+M3_gw00",
     "entropy_threshold": 0.5, "tau": 0.85, "t_draft": 32, "guidance_weight": 0.0},
    {"name": "M1_eta1.0+IGSD_tau0.9_td32+M3_gw00",
     "entropy_threshold": 1.0, "tau": 0.9, "t_draft": 32, "guidance_weight": 0.0},
    {"name": "M1_eta1.0+IGSD_tau0.85_td32+M3_gw00",
     "entropy_threshold": 1.0, "tau": 0.85, "t_draft": 32, "guidance_weight": 0.0},
    {"name": "M1_eta0.5+IGSD_tau0.9_td32+M3_gw00",
     "entropy_threshold": 0.5, "tau": 0.9, "t_draft": 32, "guidance_weight": 0.0},
    {"name": "M1_eta0.5+IGSD_tau0.85_td32+M3_gw03",
     "entropy_threshold": 0.5, "tau": 0.85, "t_draft": 32, "guidance_weight": 0.3},
]

# Recipe labels for final paper
RECIPE_LABELS = {
    "M1_eta0.5+IGSD_tau0.85_td32+M3_gw00": "Max-Speed",
    "M1_eta1.0+IGSD_tau0.9_td32+M3_gw00":  "Balanced-A",
    "M1_eta1.0+IGSD_tau0.85_td32+M3_gw00": "Balanced-B",
    "M1_eta0.5+IGSD_tau0.9_td32+M3_gw00":  "Conservative",
    "M1_eta0.5+IGSD_tau0.85_td32+M3_gw03": "Quality-First",
}


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
        x_refine = x.clone()
        s_accept_full = torch.cat([
            torch.ones(1, prompt_len, dtype=torch.bool, device=self.device),
            s_accept_gen
        ], dim=1)
        s_refine_full = ~s_accept_full
        x_refine[s_refine_full] = MASK_ID

        kv_hit_steps = []
        t_refine_start = time.perf_counter()
        prev_low_entropy_mask = None

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
                    unmasked_prefix = x_refine[:, :prompt_len].clone()
                    ar_out = self.ar_model(unmasked_prefix)
                    ar_logits = ar_out.logits[:, -1:, :]
                    p_ar = F.softmax(ar_logits, dim=-1)

                    vocab_size = p.shape[-1]
                    ar_vocab = p_ar.shape[-1]

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

            if i < 3:
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

        if (i + 1) % 50 == 0:
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
    if best_individual_qas <= 0:
        return 0.0
    return composed_qas / best_individual_qas


def get_ref_qas(config):
    """Get individual reference QAS for Ortho computation."""
    eta_key = f"M1_eta{config['entropy_threshold']}"
    igsd_key = f"IGSD_tau{config['tau']}_td{config['t_draft']}"
    gw_val = config['guidance_weight']

    ref_combined = []
    ref_gsm8k = []
    ref_math500 = []

    for key in [eta_key, igsd_key]:
        if key in INDIVIDUAL_QAS:
            ref_combined.append(INDIVIDUAL_QAS[key]["combined_qas"])
            ref_gsm8k.append(INDIVIDUAL_QAS[key]["gsm8k"]["qas"])
            ref_math500.append(INDIVIDUAL_QAS[key]["math500"]["qas"])

    if gw_val > 0:
        m3_key = f"M3_gw{gw_val}"
        if m3_key in INDIVIDUAL_QAS:
            ref_combined.append(INDIVIDUAL_QAS[m3_key]["combined_qas"])
            ref_gsm8k.append(INDIVIDUAL_QAS[m3_key]["gsm8k"]["qas"])
            ref_math500.append(INDIVIDUAL_QAS[m3_key]["math500"]["qas"])

    return {
        "combined": max(ref_combined) if ref_combined else 1.0,
        "gsm8k": max(ref_gsm8k) if ref_gsm8k else 1.0,
        "math500": max(ref_math500) if ref_math500 else 1.0,
    }


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    write_pid()

    total_evals = len(TOP5_CONFIGS) * len(SEEDS)  # 5 configs * 3 seeds = 15

    print("=" * 70)
    print("Three-Way Composition Top-5 Full Validation (PILOT mode)")
    print(f"Task: {TASK_ID}")
    print(f"Mode: PILOT ({N_GSM8K} GSM8K + {N_MATH500} MATH500)")
    print(f"Seeds: {SEEDS}")
    print(f"Configs: {len(TOP5_CONFIGS)}, Total evaluations: {total_evals}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print("=" * 70)

    # ── Load models ──
    print("\n[1/4] Loading LLaDA-8B-Instruct...")
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    # Load AR guide model for M3 (needed for config #5 which has gw=0.3)
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

    # ── Run all configs x seeds ──
    print(f"\n[2/4] Running {total_evals} evaluations (5 configs x 3 seeds)...")

    # Structure: {config_name: {seed: {gsm8k: {...}, math500: {...}, combined: {...}}}}
    all_results = {}
    eval_count = 0
    failed_count = 0

    for ci, config in enumerate(TOP5_CONFIGS):
        config_name = config["name"]
        recipe_label = RECIPE_LABELS.get(config_name, f"Config-{ci+1}")
        all_results[config_name] = {
            "config": config,
            "recipe_label": recipe_label,
            "per_seed": {},
        }

        for si, seed in enumerate(SEEDS):
            eval_count += 1
            eval_start = time.perf_counter()

            print(f"\n{'='*60}")
            print(f"[{eval_count}/{total_evals}] {config_name} | seed={seed}")
            print(f"  Recipe: {recipe_label}")
            print(f"  M1 eta={config['entropy_threshold']}, IGSD tau={config['tau']}, "
                  f"T_draft={config['t_draft']}, M3 gw={config['guidance_weight']}")
            print(f"{'='*60}")

            report_progress(eval_count, total_evals, {
                "config": config_name,
                "seed": seed,
                "completed": eval_count - 1,
                "failed": failed_count,
            })

            # Set seed and load data for this seed
            set_seed(seed)
            gsm8k_data = load_gsm8k(N_GSM8K, seed)
            math500_data = load_math500(N_MATH500, seed)

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

                # Ortho
                ref_qas = get_ref_qas(config)
                combined_ortho = compute_ortho_threeway(combined_qas, ref_qas["combined"])
                gsm8k_ortho = gsm8k_qas / ref_qas["gsm8k"] if ref_qas["gsm8k"] > 0 else 0.0
                math500_ortho = math500_qas / ref_qas["math500"] if ref_qas["math500"] > 0 else 0.0

                eval_elapsed = (time.perf_counter() - eval_start) / 60

                seed_result = {
                    "seed": seed,
                    "elapsed_min": round(eval_elapsed, 1),
                    "gsm8k": {
                        "n_samples": gsm8k_res["n_samples"],
                        "correct": gsm8k_res["correct"],
                        "exact_match": gsm8k_res["exact_match"],
                        "avg_tps": gsm8k_res["avg_tps"],
                        "tps_std": gsm8k_res["tps_std"],
                        "avg_accept_rate": gsm8k_res["avg_accept_rate"],
                        "avg_m1_chr": gsm8k_res["avg_m1_chr"],
                        "speedup": gsm8k_speedup,
                        "acc_retention": gsm8k_acc_ret,
                        "qas": gsm8k_qas,
                        "ortho": gsm8k_ortho,
                    },
                    "math500": {
                        "n_samples": math500_res["n_samples"],
                        "correct": math500_res["correct"],
                        "exact_match": math500_res["exact_match"],
                        "avg_tps": math500_res["avg_tps"],
                        "tps_std": math500_res["tps_std"],
                        "avg_accept_rate": math500_res["avg_accept_rate"],
                        "avg_m1_chr": math500_res["avg_m1_chr"],
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
                    "ref_best_qas": ref_qas,
                    "sample_texts": {
                        "gsm8k": gsm8k_res.get("sample_texts", [])[:2],
                        "math500": math500_res.get("sample_texts", [])[:2],
                    },
                }

                all_results[config_name]["per_seed"][str(seed)] = seed_result

                print(f"\n  === {config_name} seed={seed} ({eval_elapsed:.1f} min) ===")
                print(f"  GSM8K:  acc={gsm8k_res['exact_match']:.3f}, speedup={gsm8k_speedup:.2f}x, "
                      f"QAS={gsm8k_qas:.3f}")
                print(f"  MATH500: acc={math500_res['exact_match']:.3f}, speedup={math500_speedup:.2f}x, "
                      f"QAS={math500_qas:.3f}")
                print(f"  Combined: speedup={combined_speedup:.2f}x, acc_ret={combined_acc_ret:.3f}, "
                      f"QAS={combined_qas:.3f}, Ortho={combined_ortho:.3f}")

            except Exception as e:
                print(f"\n  [FAILED] {config_name} seed={seed}: {e}")
                failed_count += 1
                import traceback
                traceback.print_exc()

            torch.cuda.empty_cache()
            gc.collect()

    # ── Aggregate results per config ──
    print(f"\n[3/4] Aggregating multi-seed statistics...")

    aggregated = {}
    for config_name, config_data in all_results.items():
        per_seed = config_data["per_seed"]
        seeds_completed = list(per_seed.keys())

        if len(seeds_completed) == 0:
            continue

        # Collect per-seed values
        gsm8k_accs = [per_seed[s]["gsm8k"]["exact_match"] for s in seeds_completed]
        gsm8k_speedups = [per_seed[s]["gsm8k"]["speedup"] for s in seeds_completed]
        gsm8k_qas_vals = [per_seed[s]["gsm8k"]["qas"] for s in seeds_completed]

        math500_accs = [per_seed[s]["math500"]["exact_match"] for s in seeds_completed]
        math500_speedups = [per_seed[s]["math500"]["speedup"] for s in seeds_completed]
        math500_qas_vals = [per_seed[s]["math500"]["qas"] for s in seeds_completed]

        combined_speedups = [per_seed[s]["combined"]["speedup"] for s in seeds_completed]
        combined_acc_rets = [per_seed[s]["combined"]["acc_retention"] for s in seeds_completed]
        combined_qas_vals = [per_seed[s]["combined"]["qas"] for s in seeds_completed]
        combined_ortho_vals = [per_seed[s]["combined"]["ortho"] for s in seeds_completed]

        agg = {
            "config_name": config_name,
            "recipe_label": config_data["recipe_label"],
            "config": config_data["config"],
            "seeds_completed": len(seeds_completed),
            "gsm8k": {
                "acc_mean": float(np.mean(gsm8k_accs)),
                "acc_std": float(np.std(gsm8k_accs)),
                "speedup_mean": float(np.mean(gsm8k_speedups)),
                "speedup_std": float(np.std(gsm8k_speedups)),
                "qas_mean": float(np.mean(gsm8k_qas_vals)),
                "qas_std": float(np.std(gsm8k_qas_vals)),
                "per_seed_acc": {s: per_seed[s]["gsm8k"]["exact_match"] for s in seeds_completed},
                "per_seed_speedup": {s: per_seed[s]["gsm8k"]["speedup"] for s in seeds_completed},
                "per_seed_qas": {s: per_seed[s]["gsm8k"]["qas"] for s in seeds_completed},
            },
            "math500": {
                "acc_mean": float(np.mean(math500_accs)),
                "acc_std": float(np.std(math500_accs)),
                "speedup_mean": float(np.mean(math500_speedups)),
                "speedup_std": float(np.std(math500_speedups)),
                "qas_mean": float(np.mean(math500_qas_vals)),
                "qas_std": float(np.std(math500_qas_vals)),
                "per_seed_acc": {s: per_seed[s]["math500"]["exact_match"] for s in seeds_completed},
                "per_seed_speedup": {s: per_seed[s]["math500"]["speedup"] for s in seeds_completed},
                "per_seed_qas": {s: per_seed[s]["math500"]["qas"] for s in seeds_completed},
            },
            "combined": {
                "speedup_mean": float(np.mean(combined_speedups)),
                "speedup_std": float(np.std(combined_speedups)),
                "acc_ret_mean": float(np.mean(combined_acc_rets)),
                "acc_ret_std": float(np.std(combined_acc_rets)),
                "qas_mean": float(np.mean(combined_qas_vals)),
                "qas_std": float(np.std(combined_qas_vals)),
                "ortho_mean": float(np.mean(combined_ortho_vals)),
                "ortho_std": float(np.std(combined_ortho_vals)),
                "per_seed_speedup": {s: per_seed[s]["combined"]["speedup"] for s in seeds_completed},
                "per_seed_acc_ret": {s: per_seed[s]["combined"]["acc_retention"] for s in seeds_completed},
                "per_seed_qas": {s: per_seed[s]["combined"]["qas"] for s in seeds_completed},
                "per_seed_ortho": {s: per_seed[s]["combined"]["ortho"] for s in seeds_completed},
            },
            # Stability check: QAS std < 30% of mean
            "stability": {
                "qas_cv": float(np.std(combined_qas_vals) / np.mean(combined_qas_vals)) if np.mean(combined_qas_vals) > 0 else float("inf"),
                "stable": bool(np.std(combined_qas_vals) < 0.3 * np.mean(combined_qas_vals)) if np.mean(combined_qas_vals) > 0 else False,
            },
        }

        aggregated[config_name] = agg

    # ── Build summary table (paper-ready) ──
    summary_table = []
    for config_name, agg in aggregated.items():
        summary_table.append({
            "Recipe": agg["recipe_label"],
            "Config": config_name,
            "Speedup_mean": round(agg["combined"]["speedup_mean"], 3),
            "Speedup_std": round(agg["combined"]["speedup_std"], 3),
            "AccRet_mean": round(agg["combined"]["acc_ret_mean"], 3),
            "AccRet_std": round(agg["combined"]["acc_ret_std"], 3),
            "QAS_mean": round(agg["combined"]["qas_mean"], 3),
            "QAS_std": round(agg["combined"]["qas_std"], 3),
            "Ortho_mean": round(agg["combined"]["ortho_mean"], 3),
            "Ortho_std": round(agg["combined"]["ortho_std"], 3),
            "GSM8K_Acc_mean": round(agg["gsm8k"]["acc_mean"], 3),
            "MATH500_Acc_mean": round(agg["math500"]["acc_mean"], 3),
            "Seeds_completed": agg["seeds_completed"],
            "Stable": agg["stability"]["stable"],
            "QAS_CV": round(agg["stability"]["qas_cv"], 3),
        })

    summary_table.sort(key=lambda r: r["QAS_mean"], reverse=True)

    # ── Pass criteria ──
    all_configs_complete = all(
        agg["seeds_completed"] == len(SEEDS) for agg in aggregated.values()
    )
    all_stable = all(agg["stability"]["stable"] for agg in aggregated.values())
    configs_complete_count = sum(
        1 for agg in aggregated.values() if agg["seeds_completed"] == len(SEEDS)
    )
    pass_met = (configs_complete_count == len(TOP5_CONFIGS)) and all_stable

    # ── Save results ──
    print(f"\n[4/4] Saving results...")

    elapsed_min = (datetime.now() - start_time).total_seconds() / 60

    output = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "iteration": 2,
        "model": "LLaDA-8B-Instruct",
        "guide_model": "Qwen2.5-0.5B",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed_min, 1),
        "seeds": SEEDS,
        "pilot_config": {
            "gsm8k_samples": N_GSM8K,
            "math500_samples": N_MATH500,
            "gen_length": GEN_LENGTH,
            "steps": T_FULL,
        },
        "baseline_reference": PILOT_BASELINE,
        "qas_formula": "QAS = Speedup * Accuracy_Retention (corrected, no 0.5x penalty)",
        "ortho_formula": "Ortho = QAS(M1+IGSD+M3) / max(QAS(M1), QAS(IGSD), QAS(M3))",
        "combined_metric": "0.7 * GSM8K + 0.3 * MATH500",
        "pass_criteria": {
            "requirement": "All 5 configs complete on 3 seeds AND per-seed QAS std < 30% of mean",
            "configs_with_3_seeds": configs_complete_count,
            "all_stable": all_stable,
            "met": pass_met,
        },
        "individual_qas_reference": INDIVIDUAL_QAS,
        "top5_pilot_source": {
            config["name"]: {
                "pilot_combined_qas": qas,
                "recipe_label": RECIPE_LABELS.get(config["name"], ""),
            }
            for config, qas in zip(TOP5_CONFIGS,
                                   [0.952, 0.952, 0.947, 0.940, 0.934])
        },
        "aggregated_results": aggregated,
        "per_config_per_seed": all_results,
        "summary_table": summary_table,
        "vram": vram_info,
    }

    # Key findings
    findings = []
    findings.append(f"Completed {configs_complete_count}/5 configs with {len(SEEDS)} seeds each")
    findings.append(f"Total evaluations: {eval_count} attempted, {failed_count} failed")

    stable_count = sum(1 for agg in aggregated.values() if agg["stability"]["stable"])
    findings.append(f"Stability: {stable_count}/5 configs have QAS CV < 30%")

    if aggregated:
        best_config = max(aggregated.values(), key=lambda a: a["combined"]["qas_mean"])
        findings.append(
            f"Best config by QAS: {best_config['config_name']} "
            f"(QAS={best_config['combined']['qas_mean']:.3f} +/- "
            f"{best_config['combined']['qas_std']:.3f})"
        )

        # Compare with pilot QAS
        pilot_qas_map = {
            "M1_eta0.5+IGSD_tau0.85_td32+M3_gw00": 0.952,
            "M1_eta1.0+IGSD_tau0.9_td32+M3_gw00": 0.952,
            "M1_eta1.0+IGSD_tau0.85_td32+M3_gw00": 0.947,
            "M1_eta0.5+IGSD_tau0.9_td32+M3_gw00": 0.940,
            "M1_eta0.5+IGSD_tau0.85_td32+M3_gw03": 0.934,
        }
        for cname, agg in aggregated.items():
            pilot_qas = pilot_qas_map.get(cname, 0)
            drift = abs(agg["combined"]["qas_mean"] - pilot_qas) / pilot_qas if pilot_qas > 0 else 0
            if drift > 0.15:
                findings.append(
                    f"WARNING: {cname} QAS drifted {drift:.1%} from pilot "
                    f"({pilot_qas:.3f} -> {agg['combined']['qas_mean']:.3f})"
                )

    findings.append(f"Pass criteria met: {pass_met}")
    output["key_findings"] = findings

    output_path = RESULTS_DIR / "threeway_pareto_full.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to: {output_path}")

    # ── Print final summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: Three-Way Top-5 Full Validation (PILOT)")
    print("=" * 70)
    print(f"  Total elapsed: {elapsed_min:.1f} minutes")
    print(f"  Pass criteria met: {pass_met}")

    print(f"\n  ACCELERATION RECIPES (sorted by QAS):")
    print(f"  {'Recipe':<16} {'Config':<50} {'Speedup':>10} {'AccRet':>10} {'QAS':>10} {'Ortho':>10} {'Stable':>8}")
    print(f"  {'-'*114}")
    for row in summary_table:
        print(f"  {row['Recipe']:<16} {row['Config']:<50} "
              f"{row['Speedup_mean']:>7.3f}x   {row['AccRet_mean']:>7.3f}   "
              f"{row['QAS_mean']:>7.3f}   {row['Ortho_mean']:>7.3f}   "
              f"{'YES' if row['Stable'] else 'NO':>8}")

    print(f"\n  PER-SEED BREAKDOWN:")
    for config_name, agg in aggregated.items():
        print(f"\n  {config_name} ({agg['recipe_label']}):")
        for s in [str(seed) for seed in SEEDS]:
            if s in agg["combined"]["per_seed_qas"]:
                print(f"    seed={s}: Speedup={agg['combined']['per_seed_speedup'][s]:.3f}x, "
                      f"AccRet={agg['combined']['per_seed_acc_ret'][s]:.3f}, "
                      f"QAS={agg['combined']['per_seed_qas'][s]:.3f}, "
                      f"Ortho={agg['combined']['per_seed_ortho'][s]:.3f}")
        print(f"    Mean QAS={agg['combined']['qas_mean']:.3f} +/- {agg['combined']['qas_std']:.3f} "
              f"(CV={agg['stability']['qas_cv']:.1%})")

    print("=" * 70)

    # Mark task as done
    mark_done(
        status="success" if pass_met else "partial",
        summary=f"Three-way top-5 validation: {configs_complete_count}/5 configs, "
                f"{len(SEEDS)} seeds, stable={stable_count}/5, "
                f"pass={'YES' if pass_met else 'NO'}, {elapsed_min:.1f}min"
    )

    return output


if __name__ == "__main__":
    result = main()
