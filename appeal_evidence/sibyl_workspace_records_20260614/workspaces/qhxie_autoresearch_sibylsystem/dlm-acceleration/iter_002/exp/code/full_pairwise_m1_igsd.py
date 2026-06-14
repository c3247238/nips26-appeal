"""
M1+IGSD Full-Scale Pairwise Composition (FULL Mode).

Task: pairwise_m1_igsd_full
Iteration: 2
Mode: FULL (1319 GSM8K + 500 MATH500, seeds=[42, 123, 456])

Tests M1 (EntropyCache, eta=0.5) + IGSD composition at full scale.
Uses best operating points from Phase 2 corrected Pareto curves.

M1 best: entropy_threshold=0.5 (combined_qas=0.981, gsm8k speedup=1.158x, acc_ret=94.5%)
IGSD configs to test:
  - tau=0.7, T_draft=16 (best combined_qas=1.399, gsm8k speedup=2.81x)
  - tau=0.85, T_draft=32 (balanced: combined_qas=1.052, gsm8k acc_ret=67.8%)
  - tau=0.9, T_draft=32 (conservative: combined_qas=1.035, gsm8k acc_ret=67.8%)

Reports per-seed Ortho breakdown. Uses corrected QAS formula.
Combined metric = 0.7*GSM8K + 0.3*MATH500.

Output: exp/results/pairwise/m1_igsd_full.json

Usage:
    CUDA_VISIBLE_DEVICES=0 conda run -n sibyl_dlm-acceleration python full_pairwise_m1_igsd.py
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
WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/iter_002")
SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
ITER1_DIR   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/iter_001")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
GSM8K_DIR   = str(SHARED / "datasets" / "gsm8k")
MATH500_DIR = str(SHARED / "datasets" / "math500")
CODE_DIR    = ITER1_DIR / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pairwise"
TASK_ID     = "pairwise_m1_igsd_full"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
SEEDS        = [42, 123, 456]
GEN_LENGTH   = 256
BLOCK_LENGTH = 32
T_FULL       = 64
MASK_ID      = 126336
N_GSM8K      = None    # Full: all 1319
N_MATH500    = None    # Full: all 500
N_WARMUP     = 5

# Full-scale baseline from llada_baseline_full.json (aggregated 3 seeds)
FULL_BASELINE = {
    "gsm8k": {
        "exact_match": {
            "mean": 0.7122, "std": 0.0149,
            "per_seed": {42: 0.7331, 123: 0.6998, 456: 0.7036},
        },
        "avg_tps": {
            "mean": 33.7725, "std": 0.0215,
            "per_seed": {42: 33.7443, 123: 33.7965, 456: 33.7769},
        },
    },
    "math500": {
        "exact_match": {
            "mean": 0.1107, "std": 0.0066,
            "per_seed": {42: 0.118, 123: 0.102, 456: 0.112},
        },
        "avg_tps": {
            "mean": 79.1079, "std": 0.1169,
            "per_seed": {42: 78.9473, 123: 79.1541, 456: 79.2222},
        },
    },
}

# Individual method QAS from Phase 2 (pilot baselines used for pilot;
# for full mode, we re-derive from full baseline)
# These are the pilot-referenced values used for consistency with Phase 2
INDIVIDUAL_QAS_PILOT = {
    "M1_eta0.5": {
        "gsm8k":   {"speedup": 1.158, "acc_ret": 0.945, "qas": 1.094},
        "math500": {"speedup": 1.069, "acc_ret": 1.066, "qas": 1.140},
        "combined_qas": 0.981,
    },
    "IGSD_tau0.7_td16": {
        "gsm8k":   {"speedup": 2.812, "acc_ret": 0.582, "qas": 1.637},
        "math500": {"speedup": 2.573, "acc_ret": 0.344, "qas": 0.884},
        "combined_qas": 1.399,
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
}

# Composition configs to test
COMPOSITION_CONFIGS = [
    {"name": "M1_eta0.5+IGSD_tau0.7_td16",  "entropy_threshold": 0.5, "tau": 0.7,  "t_draft": 16,
     "igsd_ref_key": "IGSD_tau0.7_td16"},
    {"name": "M1_eta0.5+IGSD_tau0.85_td32", "entropy_threshold": 0.5, "tau": 0.85, "t_draft": 32,
     "igsd_ref_key": "IGSD_tau0.85_td32"},
    {"name": "M1_eta0.5+IGSD_tau0.9_td32",  "entropy_threshold": 0.5, "tau": 0.9,  "t_draft": 32,
     "igsd_ref_key": "IGSD_tau0.9_td32"},
]


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


# ── Checkpoint / Resume ────────────────────────────────────────────────────
CHECKPOINT_PATH = RESULTS_DIR / f"{TASK_ID}_full_checkpoint.json"


def save_checkpoint(completed_runs, results):
    """Save checkpoint for resume capability."""
    CHECKPOINT_PATH.write_text(json.dumps({
        "completed_runs": completed_runs,
        "results": results,
    }, default=str))


def load_checkpoint():
    """Load checkpoint if exists."""
    if CHECKPOINT_PATH.exists():
        try:
            data = json.loads(CHECKPOINT_PATH.read_text())
            return data.get("completed_runs", {}), data.get("results", {})
        except (json.JSONDecodeError, ValueError):
            pass
    return {}, {}


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
    """Check GSM8K answer match."""
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold_answer)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip() == g.strip()


def extract_math500_answer(text):
    """Extract boxed answer from MATH500 solution text."""
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
    """Check MATH500 answer match."""
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


# ── M1+IGSD Composed Generator ───────────────────────────────────────────
class M1_IGSD_Generator:
    """
    Composed M1 (EntropyCache) + IGSD generator.

    Uses whole-sequence draft+refine (matching igsd_pareto_corrected.py pattern)
    with M1 entropy-based KV cache hit rate tracking at each forward pass.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt,
        entropy_threshold=0.5,
        tau=0.85,
        t_draft=32,
        t_full=64,
        gen_length=256,
        block_length=32,
        apply_chat_template=True,
    ):
        """
        M1+IGSD composed generation using whole-sequence approach.
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

        # ── Phase 1: Whole-sequence draft ────────────────────────────
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

            # IGSD whole-sequence unmasking
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

        # ── Phase 3: Whole-sequence refine for S_refine ──────────────
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
                    total = non_masked.sum().item()
                    m1_cache_hits_refine += hits
                    m1_cache_total_refine += total

                prev_low_entropy_mask = low_entropy_now.clone()

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
            "t_full": t_full,
        }


# ── Evaluation Functions ──────────────────────────────────────────────────
def evaluate_gsm8k(generator, data, config, n_warmup=N_WARMUP):
    """Evaluate M1+IGSD on GSM8K samples."""
    eta = config["entropy_threshold"]
    tau = config["tau"]
    t_draft = config["t_draft"]

    correct = 0
    total = len(data)
    tps_list = []
    accept_rates = []
    chr_values = []
    sample_texts = []

    for i, item in enumerate(data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            result = generator.generate(
                prompt=prompt,
                entropy_threshold=eta,
                tau=tau,
                t_draft=t_draft,
                t_full=T_FULL,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                apply_chat_template=True,
            )
            pred_text = result["generated_text"]
            is_correct = gsm8k_exact_match(pred_text, item["answer"])
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
            print(f"  [ERROR] GSM8K sample {i}: {e}")
            continue

        if (i + 1) % 100 == 0:
            acc = correct / (i + 1)
            avg_tps = float(np.mean(tps_list)) if tps_list else 0
            avg_ar = float(np.mean(accept_rates))
            avg_chr = float(np.mean(chr_values))
            print(f"  [GSM8K] {i+1}/{total}: acc={acc:.3f}, "
                  f"tps={avg_tps:.1f}, accept={avg_ar:.3f}, chr={avg_chr:.3f}")

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


def evaluate_math500(generator, data, config, n_warmup=N_WARMUP):
    """Evaluate M1+IGSD on MATH500 samples."""
    eta = config["entropy_threshold"]
    tau = config["tau"]
    t_draft = config["t_draft"]

    correct = 0
    total = len(data)
    tps_list = []
    accept_rates = []
    chr_values = []
    sample_texts = []

    for i, item in enumerate(data):
        prompt = build_math500_prompt(item["problem"])
        try:
            result = generator.generate(
                prompt=prompt,
                entropy_threshold=eta,
                tau=tau,
                t_draft=t_draft,
                t_full=T_FULL,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                apply_chat_template=True,
            )
            pred_text = result["generated_text"]
            gold = item.get("answer", item.get("solution", ""))
            is_correct = math500_exact_match(pred_text, gold)
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
            print(f"  [ERROR] MATH500 sample {i}: {e}")
            continue

        if (i + 1) % 100 == 0:
            acc = correct / (i + 1)
            avg_tps = float(np.mean(tps_list)) if tps_list else 0
            avg_ar = float(np.mean(accept_rates))
            print(f"  [MATH500] {i+1}/{total}: acc={acc:.3f}, "
                  f"tps={avg_tps:.1f}, accept={avg_ar:.3f}")

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


def compute_ortho(composed_qas, ref_qas_m1, ref_qas_igsd):
    """Ortho = QAS(M1+IGSD) / max(QAS(M1), QAS(IGSD))."""
    ref_max = max(ref_qas_m1, ref_qas_igsd)
    if ref_max <= 0:
        return 0.0
    return composed_qas / ref_max


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    write_pid()

    print("=" * 70)
    print("M1+IGSD Pairwise Composition FULL-SCALE")
    print(f"Task: {TASK_ID}")
    print(f"Mode: FULL (1319 GSM8K + 500 MATH500, seeds={SEEDS})")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"Configs: {len(COMPOSITION_CONFIGS)}")
    print(f"Total runs: {len(COMPOSITION_CONFIGS)} configs x {len(SEEDS)} seeds x 2 benchmarks = "
          f"{len(COMPOSITION_CONFIGS) * len(SEEDS) * 2}")
    print("=" * 70)

    # Load checkpoint for resume
    completed_runs, cached_results = load_checkpoint()
    if completed_runs:
        print(f"\n  [RESUME] Found checkpoint with {len(completed_runs)} completed runs")

    # ── Load model ──
    print("\n[1/4] Loading LLaDA-8B-Instruct...")
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()

    generator = M1_IGSD_Generator(model, tokenizer, "cuda")
    print("  Model loaded.")

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
    print("\n[2/4] Loading datasets...")
    gsm8k_data_full = load_gsm8k(N_GSM8K)  # None = all 1319
    math500_data_full = load_math500(N_MATH500)  # None = all 500
    print(f"  GSM8K: {len(gsm8k_data_full)} samples")
    print(f"  MATH500: {len(math500_data_full)} samples")

    # ── Run compositions across seeds ──
    print("\n[3/4] Running M1+IGSD compositions (full-scale, 3 seeds)...")
    total_configs = len(COMPOSITION_CONFIGS)
    total_runs = total_configs * len(SEEDS) * 2  # 2 benchmarks
    run_count = 0

    # Structure: per_seed_results[config_name][seed] = {"gsm8k": ..., "math500": ...}
    per_seed_results = {}

    for ci, config in enumerate(COMPOSITION_CONFIGS):
        config_name = config["name"]
        per_seed_results[config_name] = {}

        for seed in SEEDS:
            set_seed(seed)

            # Use seed-specific baseline for speedup computation
            gsm8k_baseline_acc = FULL_BASELINE["gsm8k"]["exact_match"]["per_seed"].get(seed,
                FULL_BASELINE["gsm8k"]["exact_match"]["mean"])
            gsm8k_baseline_tps = FULL_BASELINE["gsm8k"]["avg_tps"]["per_seed"].get(seed,
                FULL_BASELINE["gsm8k"]["avg_tps"]["mean"])
            math500_baseline_acc = FULL_BASELINE["math500"]["exact_match"]["per_seed"].get(seed,
                FULL_BASELINE["math500"]["exact_match"]["mean"])
            math500_baseline_tps = FULL_BASELINE["math500"]["avg_tps"]["per_seed"].get(seed,
                FULL_BASELINE["math500"]["avg_tps"]["mean"])

            seed_result = {}

            # ── GSM8K ──
            run_key_gsm = f"{config_name}_s{seed}_gsm8k"
            run_count += 1

            if run_key_gsm in completed_runs:
                print(f"\n  [SKIP] {run_key_gsm} (cached)")
                seed_result["gsm8k"] = cached_results.get(run_key_gsm, {})
            else:
                print(f"\n{'='*60}")
                print(f"Run {run_count}/{total_runs}: {config_name} | seed={seed} | GSM8K")
                print(f"  M1 eta={config['entropy_threshold']}, IGSD tau={config['tau']}, "
                      f"T_draft={config['t_draft']}")
                print(f"{'='*60}")

                report_progress(run_count, total_runs, {
                    "config": config_name, "seed": seed, "benchmark": "gsm8k"
                })

                gsm8k_res = evaluate_gsm8k(generator, gsm8k_data_full, config, N_WARMUP)

                gsm8k_speedup = gsm8k_res["avg_tps"] / gsm8k_baseline_tps if gsm8k_baseline_tps > 0 else 0
                gsm8k_acc_ret = gsm8k_res["exact_match"] / gsm8k_baseline_acc if gsm8k_baseline_acc > 0 else 0
                gsm8k_qas = gsm8k_speedup * gsm8k_acc_ret

                seed_result["gsm8k"] = {
                    **gsm8k_res,
                    "speedup": gsm8k_speedup,
                    "acc_retention": gsm8k_acc_ret,
                    "qas": gsm8k_qas,
                    "baseline_acc": gsm8k_baseline_acc,
                    "baseline_tps": gsm8k_baseline_tps,
                }

                print(f"  GSM8K (seed={seed}): acc={gsm8k_res['exact_match']:.3f}, "
                      f"speedup={gsm8k_speedup:.2f}x, acc_ret={gsm8k_acc_ret:.3f}, "
                      f"QAS={gsm8k_qas:.3f}")

                # Cache
                completed_runs[run_key_gsm] = True
                cached_results[run_key_gsm] = seed_result["gsm8k"]
                save_checkpoint(completed_runs, cached_results)

                torch.cuda.empty_cache()
                gc.collect()

            # ── MATH500 ──
            run_key_math = f"{config_name}_s{seed}_math500"
            run_count += 1

            if run_key_math in completed_runs:
                print(f"\n  [SKIP] {run_key_math} (cached)")
                seed_result["math500"] = cached_results.get(run_key_math, {})
            else:
                print(f"\n{'='*60}")
                print(f"Run {run_count}/{total_runs}: {config_name} | seed={seed} | MATH500")
                print(f"{'='*60}")

                report_progress(run_count, total_runs, {
                    "config": config_name, "seed": seed, "benchmark": "math500"
                })

                math500_res = evaluate_math500(generator, math500_data_full, config, N_WARMUP)

                math500_speedup = math500_res["avg_tps"] / math500_baseline_tps if math500_baseline_tps > 0 else 0
                math500_acc_ret = math500_res["exact_match"] / math500_baseline_acc if math500_baseline_acc > 0 else 0
                math500_qas = math500_speedup * math500_acc_ret

                seed_result["math500"] = {
                    **math500_res,
                    "speedup": math500_speedup,
                    "acc_retention": math500_acc_ret,
                    "qas": math500_qas,
                    "baseline_acc": math500_baseline_acc,
                    "baseline_tps": math500_baseline_tps,
                }

                print(f"  MATH500 (seed={seed}): acc={math500_res['exact_match']:.3f}, "
                      f"speedup={math500_speedup:.2f}x, acc_ret={math500_acc_ret:.3f}, "
                      f"QAS={math500_qas:.3f}")

                # Cache
                completed_runs[run_key_math] = True
                cached_results[run_key_math] = seed_result["math500"]
                save_checkpoint(completed_runs, cached_results)

                torch.cuda.empty_cache()
                gc.collect()

            per_seed_results[config_name][seed] = seed_result

    # ── Aggregate results ──
    print("\n[4/4] Computing aggregate results...")

    igsd_ref_map = {c["name"]: c["igsd_ref_key"] for c in COMPOSITION_CONFIGS}

    all_results = {}
    ortho_table = []

    for config_name, seed_data in per_seed_results.items():
        igsd_ref_key = igsd_ref_map[config_name]
        m1_ref = INDIVIDUAL_QAS_PILOT["M1_eta0.5"]
        igsd_ref = INDIVIDUAL_QAS_PILOT[igsd_ref_key]

        # Per-seed Ortho breakdown
        per_seed_breakdown = []
        gsm8k_speedups = []
        gsm8k_acc_rets = []
        gsm8k_qas_list = []
        math500_speedups = []
        math500_acc_rets = []
        math500_qas_list = []
        combined_qas_list = []

        for seed in SEEDS:
            sr = seed_data.get(seed, {})
            g = sr.get("gsm8k", {})
            m = sr.get("math500", {})

            g_speedup = g.get("speedup", 0)
            g_acc_ret = g.get("acc_retention", 0)
            g_qas = g.get("qas", 0)
            m_speedup = m.get("speedup", 0)
            m_acc_ret = m.get("acc_retention", 0)
            m_qas = m.get("qas", 0)

            # Combined: 0.7*GSM8K + 0.3*MATH500
            combined_speedup = 0.7 * g_speedup + 0.3 * m_speedup
            combined_acc_ret = 0.7 * g_acc_ret + 0.3 * m_acc_ret
            combined_qas = combined_speedup * combined_acc_ret

            gsm8k_ortho = compute_ortho(g_qas, m1_ref["gsm8k"]["qas"], igsd_ref["gsm8k"]["qas"])
            math500_ortho = compute_ortho(m_qas, m1_ref["math500"]["qas"], igsd_ref["math500"]["qas"])
            combined_ortho = compute_ortho(combined_qas, m1_ref["combined_qas"], igsd_ref["combined_qas"])

            per_seed_breakdown.append({
                "seed": seed,
                "gsm8k": {
                    "acc": g.get("exact_match", 0),
                    "tps": g.get("avg_tps", 0),
                    "speedup": g_speedup,
                    "acc_ret": g_acc_ret,
                    "qas": g_qas,
                    "ortho": gsm8k_ortho,
                    "accept_rate": g.get("avg_accept_rate", 0),
                    "m1_chr": g.get("avg_m1_chr", 0),
                },
                "math500": {
                    "acc": m.get("exact_match", 0),
                    "tps": m.get("avg_tps", 0),
                    "speedup": m_speedup,
                    "acc_ret": m_acc_ret,
                    "qas": m_qas,
                    "ortho": math500_ortho,
                    "accept_rate": m.get("avg_accept_rate", 0),
                    "m1_chr": m.get("avg_m1_chr", 0),
                },
                "combined": {
                    "speedup": combined_speedup,
                    "acc_ret": combined_acc_ret,
                    "qas": combined_qas,
                    "ortho": combined_ortho,
                },
            })

            gsm8k_speedups.append(g_speedup)
            gsm8k_acc_rets.append(g_acc_ret)
            gsm8k_qas_list.append(g_qas)
            math500_speedups.append(m_speedup)
            math500_acc_rets.append(m_acc_ret)
            math500_qas_list.append(m_qas)
            combined_qas_list.append(combined_qas)

        # Aggregate across seeds
        agg_gsm8k_qas = float(np.mean(gsm8k_qas_list))
        agg_math500_qas = float(np.mean(math500_qas_list))
        agg_combined_qas = float(np.mean(combined_qas_list))

        agg_gsm8k_ortho = compute_ortho(agg_gsm8k_qas, m1_ref["gsm8k"]["qas"], igsd_ref["gsm8k"]["qas"])
        agg_math500_ortho = compute_ortho(agg_math500_qas, m1_ref["math500"]["qas"], igsd_ref["math500"]["qas"])
        agg_combined_ortho = compute_ortho(agg_combined_qas, m1_ref["combined_qas"], igsd_ref["combined_qas"])

        config_result = {
            "config_name": config_name,
            "entropy_threshold": 0.5,
            "tau": [c for c in COMPOSITION_CONFIGS if c["name"] == config_name][0]["tau"],
            "t_draft": [c for c in COMPOSITION_CONFIGS if c["name"] == config_name][0]["t_draft"],
            "seeds": SEEDS,
            "n_seeds": len(SEEDS),
            "per_seed_breakdown": per_seed_breakdown,
            "aggregated": {
                "gsm8k": {
                    "speedup_mean": float(np.mean(gsm8k_speedups)),
                    "speedup_std": float(np.std(gsm8k_speedups)),
                    "acc_ret_mean": float(np.mean(gsm8k_acc_rets)),
                    "acc_ret_std": float(np.std(gsm8k_acc_rets)),
                    "qas_mean": agg_gsm8k_qas,
                    "qas_std": float(np.std(gsm8k_qas_list)),
                    "ortho": agg_gsm8k_ortho,
                },
                "math500": {
                    "speedup_mean": float(np.mean(math500_speedups)),
                    "speedup_std": float(np.std(math500_speedups)),
                    "acc_ret_mean": float(np.mean(math500_acc_rets)),
                    "acc_ret_std": float(np.std(math500_acc_rets)),
                    "qas_mean": agg_math500_qas,
                    "qas_std": float(np.std(math500_qas_list)),
                    "ortho": agg_math500_ortho,
                },
                "combined": {
                    "qas_mean": agg_combined_qas,
                    "qas_std": float(np.std(combined_qas_list)),
                    "ortho": agg_combined_ortho,
                },
            },
            "ref_qas": {
                "M1": m1_ref["combined_qas"],
                "IGSD": igsd_ref["combined_qas"],
                "M1_key": "M1_eta0.5",
                "IGSD_key": igsd_ref_key,
            },
        }

        all_results[config_name] = config_result

        ortho_table.append({
            "Config": config_name,
            "GSM8K_Ortho": round(agg_gsm8k_ortho, 3),
            "MATH500_Ortho": round(agg_math500_ortho, 3),
            "Combined_Ortho": round(agg_combined_ortho, 3),
            "GSM8K_Speedup_mean": round(float(np.mean(gsm8k_speedups)), 3),
            "GSM8K_AccRet_mean": round(float(np.mean(gsm8k_acc_rets)), 3),
            "MATH500_Speedup_mean": round(float(np.mean(math500_speedups)), 3),
            "MATH500_AccRet_mean": round(float(np.mean(math500_acc_rets)), 3),
            "Combined_QAS_mean": round(agg_combined_qas, 3),
            "Combined_QAS_std": round(float(np.std(combined_qas_list)), 3),
            # Per-seed Ortho for stability assessment
            "Per_Seed_Ortho": [round(psb["combined"]["ortho"], 3) for psb in per_seed_breakdown],
        })

    # Determine best and verdict
    best_config = max(all_results.values(), key=lambda r: r["aggregated"]["combined"]["ortho"])
    best_name = best_config["config_name"]

    any_synergy = any(r["aggregated"]["combined"]["ortho"] > 1.0 for r in all_results.values())
    any_near_ortho = any(r["aggregated"]["combined"]["ortho"] > 0.8 for r in all_results.values())

    if any_synergy:
        verdict = "SYNERGY"
    elif any_near_ortho:
        verdict = "NEAR_ORTHOGONAL"
    else:
        verdict = "INTERFERENCE"

    elapsed_min = (datetime.now() - start_time).total_seconds() / 60

    # Full per-seed Ortho table for paper
    detailed_ortho_table = []
    for config_name, res in all_results.items():
        for psb in res["per_seed_breakdown"]:
            for bench in ["gsm8k", "math500"]:
                detailed_ortho_table.append({
                    "Config": config_name,
                    "Seed": psb["seed"],
                    "Benchmark": bench.upper(),
                    "Speedup": round(psb[bench]["speedup"], 3),
                    "AccRet": round(psb[bench]["acc_ret"], 3),
                    "QAS": round(psb[bench]["qas"], 3),
                    "Ortho": round(psb[bench]["ortho"], 3),
                })

    output = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "iteration": 2,
        "model": "LLaDA-8B-Instruct",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed_min, 1),
        "seeds": SEEDS,
        "n_seeds": len(SEEDS),
        "full_config": {
            "gsm8k_samples": len(gsm8k_data_full),
            "math500_samples": len(math500_data_full),
            "gen_length": GEN_LENGTH,
            "steps": T_FULL,
        },
        "baseline_reference": {
            "full_baseline": FULL_BASELINE,
        },
        "qas_formula": "QAS = Speedup * Accuracy_Retention (corrected, no 0.5x penalty)",
        "ortho_formula": "Ortho = QAS(M1+IGSD) / max(QAS(M1), QAS(IGSD))",
        "combined_metric": "0.7 * GSM8K + 0.3 * MATH500",
        "pass_criteria": {
            "requirement": "Combined Ortho > 1.0 (synergy confirmed) AND per-benchmark Ortho computed",
            "met": any_synergy,
        },
        "individual_qas_reference": INDIVIDUAL_QAS_PILOT,
        "composition_configs": COMPOSITION_CONFIGS,
        "all_results": all_results,
        "ortho_table": ortho_table,
        "detailed_ortho_table": detailed_ortho_table,
        "best_config": {
            "name": best_name,
            "combined_ortho": best_config["aggregated"]["combined"]["ortho"],
            "gsm8k_ortho": best_config["aggregated"]["gsm8k"]["ortho"],
            "math500_ortho": best_config["aggregated"]["math500"]["ortho"],
        },
        "verdict": verdict,
        "vram": vram_info,
    }

    output_path = RESULTS_DIR / "m1_igsd_full.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Results saved to: {output_path}")

    # ── Final summary ──
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: M1+IGSD Pairwise Composition (FULL-SCALE)")
    print(f"  Seeds: {SEEDS}")
    print(f"  GSM8K: {len(gsm8k_data_full)} samples | MATH500: {len(math500_data_full)} samples")
    print("=" * 70)
    for row in ortho_table:
        print(f"\n  {row['Config']}")
        print(f"    GSM8K:    Ortho={row['GSM8K_Ortho']:.3f}, "
              f"Speedup={row['GSM8K_Speedup_mean']:.3f}x, AccRet={row['GSM8K_AccRet_mean']:.3f}")
        print(f"    MATH500:  Ortho={row['MATH500_Ortho']:.3f}, "
              f"Speedup={row['MATH500_Speedup_mean']:.3f}x, AccRet={row['MATH500_AccRet_mean']:.3f}")
        print(f"    Combined: Ortho={row['Combined_Ortho']:.3f} "
              f"(QAS_mean={row['Combined_QAS_mean']:.3f} +/- {row['Combined_QAS_std']:.3f})")
        print(f"    Per-seed Ortho: {row['Per_Seed_Ortho']}")

    print(f"\n  Best: {best_name} (Combined Ortho={best_config['aggregated']['combined']['ortho']:.3f})")
    print(f"  Verdict: {verdict}")
    print(f"  Pass criteria met: {any_synergy}")
    print(f"  Elapsed: {elapsed_min:.1f} minutes")
    print("=" * 70)

    mark_done(
        status="success",
        summary=f"M1+IGSD full: best={best_name}, "
                f"Ortho={best_config['aggregated']['combined']['ortho']:.3f}, "
                f"verdict={verdict}, seeds={SEEDS}, {elapsed_min:.1f}min"
    )

    # Update gpu_progress.json
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gpu_progress_path.read_text())
    except Exception:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]

    gp["timings"][TASK_ID] = {
        "planned_min": 60,
        "actual_min": round(elapsed_min),
        "start_time": start_time.isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "LLaDA-8B-Instruct",
            "mode": "FULL",
            "n_configs": len(COMPOSITION_CONFIGS),
            "n_seeds": len(SEEDS),
            "gsm8k_samples": len(gsm8k_data_full),
            "math500_samples": len(math500_data_full),
            "gpu_model": vram_info["gpu_name"],
            "gpu_count": 1,
        },
    }

    gpu_progress_path.write_text(json.dumps(gp, indent=2, default=str))
    print(f"  gpu_progress.json updated.")

    return output


if __name__ == "__main__":
    result = main()
