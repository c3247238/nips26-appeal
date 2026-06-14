"""
Batch Size Sensitivity Analysis (PILOT Mode).

Task: batch_sensitivity
Iteration: 2
Mode: PILOT (100 GSM8K, seed=42)

Tests whether composition patterns (M1+IGSD pairwise and M1+IGSD+M3 three-way)
hold at higher batch sizes relevant to production serving.

Configurations tested:
  1. M1_eta0.5+IGSD_tau0.85_td32 (best pairwise: near-orthogonal, balanced)
  2. M1_eta0.5+IGSD_tau0.85_td32+M3_gw00 (best three-way: max-speed recipe)

Batch sizes: {1, 4, 8}

Hypothesis: at higher batch sizes, mixed confidence profiles across samples
in a batch reduce cache hit rates and accept rates, potentially changing
Ortho values and composition dynamics.

Pass criteria: All batch sizes produce valid TPS measurements AND accuracy
does not drop > 5% from batch=1 to batch=8.

Output: exp/results/batch_sensitivity/batch_sensitivity.json

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n sibyl_dlm-acceleration python batch_sensitivity.py
"""

import os
import sys
import gc
import json
import time
import random
import re
import subprocess
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
GSM8K_DIR   = str(SHARED / "datasets" / "gsm8k")
CODE_DIR    = ITER1_DIR / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "batch_sensitivity"
TASK_ID     = "batch_sensitivity"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
SEED         = 42
GEN_LENGTH   = 256
BLOCK_LENGTH = 32
T_FULL       = 64
MASK_ID      = 126336
N_GSM8K      = 100     # pilot
N_WARMUP     = 3       # discard first N for TPS measurement
BATCH_SIZES  = [1, 4, 8]

# Pilot baseline (same 100-sample subset, seed=42)
PILOT_BASELINE = {
    "gsm8k": {"exact_match": 0.73, "avg_tps": 58.505},
}

# Individual method QAS reference for Ortho computation
INDIVIDUAL_QAS = {
    "M1_eta0.5": {
        "gsm8k": {"speedup": 1.158, "acc_ret": 0.945, "qas": 1.094},
        "combined_qas": 0.981,
    },
    "IGSD_tau0.85_td32": {
        "gsm8k": {"speedup": 1.732, "acc_ret": 0.678, "qas": 1.174},
        "combined_qas": 1.052,
    },
    "M3_gw0.3": {
        "gsm8k": {"speedup": 1.651, "acc_ret": 1.025, "qas": 1.692},
        "combined_qas": 2.136,
    },
}

# Configs to evaluate
CONFIGS = [
    {
        "name": "M1+IGSD (pairwise)",
        "config_key": "M1_eta0.5+IGSD_tau0.85_td32",
        "entropy_threshold": 0.5,
        "tau": 0.85,
        "t_draft": 32,
        "guidance_weight": 0.0,
        "ref_best_qas_key": "IGSD_tau0.85_td32",  # max(M1, IGSD) for Ortho
    },
    {
        "name": "M1+IGSD+M3 (three-way)",
        "config_key": "M1_eta0.5+IGSD_tau0.85_td32+M3_gw00",
        "entropy_threshold": 0.5,
        "tau": 0.85,
        "t_draft": 32,
        "guidance_weight": 0.0,  # M3 gw=0.0 (Max-Speed recipe)
        "ref_best_qas_key": "IGSD_tau0.85_td32",
    },
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


# ── GPU Utilization ───────────────────────────────────────────────────────
def get_gpu_utilization():
    """Get GPU memory and compute utilization via nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            # Get the GPU we're using (CUDA_VISIBLE_DEVICES maps to index 0)
            parts = lines[0].split(",")
            return {
                "memory_used_mb": int(parts[0].strip()),
                "memory_total_mb": int(parts[1].strip()),
                "gpu_utilization_pct": int(parts[2].strip()),
            }
    except Exception:
        pass
    return {"memory_used_mb": 0, "memory_total_mb": 0, "gpu_utilization_pct": 0}


# ── Seed & Data Loading ─────────────────────────────────────────────────
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


# ── 8-shot GSM8K Prompt ─────────────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
# Batched Three-Way Generator: M1 + IGSD + M3 with batch support
# ══════════════════════════════════════════════════════════════════════════

class BatchedThreeWayGenerator:
    """
    Composed M1 + IGSD (+ optional M3) generator with batch support.

    Pipeline (per batch):
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
    def generate_batch(
        self,
        prompts,
        entropy_threshold=0.5,
        tau=0.85,
        t_draft=32,
        guidance_weight=0.0,
        t_full=64,
        gen_length=256,
        block_length=32,
        apply_chat_template=True,
    ):
        """Generate for a batch of prompts. Returns list of per-sample dicts."""
        batch_size = len(prompts)

        # Tokenize all prompts
        if apply_chat_template:
            enc_texts = []
            for prompt in prompts:
                msg = [{"role": "user", "content": prompt}]
                enc_texts.append(self.tokenizer.apply_chat_template(
                    msg, add_generation_prompt=True, tokenize=False
                ))
        else:
            enc_texts = prompts

        enc = self.tokenizer(
            enc_texts, add_special_tokens=False, padding=True, return_tensors="pt"
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        # Initialize fully masked sequence
        x = torch.full(
            (batch_size, prompt_len + gen_length), MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat([
            attention_mask,
            torch.ones((batch_size, gen_length), dtype=attention_mask.dtype,
                        device=self.device)
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
            n_masked = int((x[:, prompt_len:] == MASK_ID).sum(dim=1).float().mean().item())
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

            # Per-sample topk
            for b in range(batch_size):
                n_masked_b = int(mask_index[b].sum().item())
                k_b = min(k, n_masked_b)
                if k_b > 0:
                    _, sel = torch.topk(confidence[b], k=k_b)
                    ti = torch.zeros_like(x[b], dtype=torch.bool)
                    ti[sel] = True
                    ti = ti & mask_index[b]
                    x[b, ti] = x0[b, ti]

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

        s_accept_gen = (final_confidence >= tau)  # (batch, gen_length)
        per_sample_accept = s_accept_gen.float().mean(dim=1)  # (batch,)
        n_total_gen = gen_length

        # ── Phase 3: Refine S_refine tokens ──────────────────────────
        x_refine = x.clone()
        s_accept_full = torch.cat([
            torch.ones(batch_size, prompt_len, dtype=torch.bool, device=self.device),
            s_accept_gen
        ], dim=1)
        s_refine_full = ~s_accept_full
        x_refine[s_refine_full] = MASK_ID

        kv_hit_steps = []
        t_refine_start = time.perf_counter()
        prev_low_entropy_mask = None

        # Average tokens to refine across batch
        avg_n_refine = int(s_refine_full[:, prompt_len:].float().sum(dim=1).mean().item())

        if avg_n_refine > 0:
            tokens_per_refine_step = max(1, avg_n_refine // t_full)
            remainder_refine = avg_n_refine % t_full

            for step in range(t_full):
                n_masked_now = int((x_refine[:, prompt_len:] == MASK_ID).sum(dim=1).float().mean().item())
                if n_masked_now == 0:
                    break

                n_frozen = n_total_gen - n_masked_now
                kv_hit_steps.append(n_frozen / n_total_gen)

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
                        pad = torch.zeros(batch_size, 1, vocab_size - ar_vocab,
                                          device=self.device, dtype=p_ar.dtype)
                        p_ar = torch.cat([p_ar, pad], dim=-1)
                    elif ar_vocab > vocab_size:
                        p_ar = p_ar[:, :, :vocab_size]

                    p_ar_broadcast = p_ar.expand(batch_size, p.shape[1], vocab_size)
                    p = (1 - guidance_weight) * p + guidance_weight * p_ar_broadcast

                x0 = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                mask_index = (x_refine == MASK_ID) & s_refine_full
                confidence = torch.where(
                    mask_index, x0_p, torch.tensor(-float("inf"), device=self.device)
                )

                k = tokens_per_refine_step + (1 if step < remainder_refine else 0)

                # Per-sample topk
                for b in range(batch_size):
                    n_masked_b = int(mask_index[b].sum().item())
                    k_b = min(k, n_masked_b)
                    if k_b > 0:
                        _, sel = torch.topk(confidence[b], k=k_b)
                        ti = torch.zeros_like(x_refine[b], dtype=torch.bool)
                        ti[sel] = True
                        ti = ti & mask_index[b]
                        x_refine[b, ti] = x0[b, ti]

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
        total_tokens = gen_length * batch_size
        total_tps = total_tokens / total_elapsed if total_elapsed > 0 else 0.0
        per_sample_latency = total_elapsed / batch_size

        kv_hit_rate = float(np.mean(kv_hit_steps)) if kv_hit_steps else float(per_sample_accept.mean().item())
        chr_draft = m1_cache_hits_draft / m1_cache_total_draft if m1_cache_total_draft > 0 else 0.0
        chr_refine = m1_cache_hits_refine / m1_cache_total_refine if m1_cache_total_refine > 0 else 0.0
        m1_total_hits = m1_cache_hits_draft + m1_cache_hits_refine
        m1_total = m1_cache_total_draft + m1_cache_total_refine
        chr_overall = m1_total_hits / m1_total if m1_total > 0 else 0.0

        # Decode all sequences
        texts = []
        for b in range(batch_size):
            text = self.tokenizer.decode(
                x_refine[b, prompt_len:].tolist(), skip_special_tokens=True
            )
            texts.append(text)

        return {
            "generated_texts": texts,
            "batch_tps": total_tps,
            "per_sample_latency_sec": per_sample_latency,
            "total_elapsed_sec": total_elapsed,
            "draft_elapsed_sec": draft_elapsed,
            "refine_elapsed_sec": refine_elapsed,
            "per_sample_accept_rate": per_sample_accept.cpu().numpy().tolist(),
            "avg_accept_rate": float(per_sample_accept.mean().item()),
            "kv_hit_rate_refine": kv_hit_rate,
            "m1_chr_draft": chr_draft,
            "m1_chr_refine": chr_refine,
            "m1_chr_overall": chr_overall,
            "batch_size": batch_size,
        }


# ── Evaluation Function ──────────────────────────────────────────────────

def evaluate_batched(generator, data, config, batch_size, n_warmup=N_WARMUP):
    """Evaluate on GSM8K with specified batch size."""
    eta = config["entropy_threshold"]
    tau_val = config["tau"]
    t_draft = config["t_draft"]
    gw = config["guidance_weight"]

    correct = 0
    total = len(data)
    batch_tps_list = []
    per_sample_latency_list = []
    accept_rates = []
    chr_values = []
    gpu_util_samples = []
    sample_texts = []

    # Process in batches
    batch_count = 0
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_items = data[start:end]
        actual_bs = len(batch_items)

        prompts = [build_gsm8k_prompt(item["question"]) for item in batch_items]

        try:
            result = generator.generate_batch(
                prompts=prompts,
                entropy_threshold=eta,
                tau=tau_val,
                t_draft=t_draft,
                guidance_weight=gw,
                t_full=T_FULL,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                apply_chat_template=True,
            )

            # Check correctness for each sample
            for j, (item, pred_text) in enumerate(zip(batch_items, result["generated_texts"])):
                gold = item.get("answer", "")
                is_correct = gsm8k_exact_match(pred_text, gold)
                if is_correct:
                    correct += 1

                if start + j < 3:
                    sample_texts.append({
                        "id": start + j,
                        "correct": is_correct,
                        "prediction": pred_text[:200],
                        "accept_rate": result["per_sample_accept_rate"][j] if j < len(result["per_sample_accept_rate"]) else result["avg_accept_rate"],
                    })

            if batch_count >= n_warmup:
                batch_tps_list.append(result["batch_tps"])
                per_sample_latency_list.append(result["per_sample_latency_sec"])

            accept_rates.append(result["avg_accept_rate"])
            chr_values.append(result["m1_chr_overall"])

            # Sample GPU utilization every few batches
            if batch_count % 3 == 0:
                gpu_info = get_gpu_utilization()
                gpu_util_samples.append(gpu_info)

        except Exception as e:
            print(f"  [ERROR] batch starting at {start} (bs={actual_bs}): {e}")
            import traceback
            traceback.print_exc()
            continue

        batch_count += 1
        if (start + actual_bs) % 20 == 0 or start + actual_bs >= total:
            acc = correct / (start + actual_bs)
            avg_tps = float(np.mean(batch_tps_list)) if batch_tps_list else 0
            print(f"  [bs={batch_size}] {start+actual_bs}/{total}: "
                  f"acc={acc:.3f}, tps={avg_tps:.1f}")

    accuracy = correct / total if total > 0 else 0
    avg_tps = float(np.mean(batch_tps_list)) if batch_tps_list else 0.0
    tps_std = float(np.std(batch_tps_list)) if batch_tps_list else 0.0
    avg_latency = float(np.mean(per_sample_latency_list)) if per_sample_latency_list else 0.0
    latency_std = float(np.std(per_sample_latency_list)) if per_sample_latency_list else 0.0

    avg_gpu_util = float(np.mean([g["gpu_utilization_pct"] for g in gpu_util_samples])) if gpu_util_samples else 0
    avg_gpu_mem = float(np.mean([g["memory_used_mb"] for g in gpu_util_samples])) if gpu_util_samples else 0

    return {
        "batch_size": batch_size,
        "n_samples": total,
        "correct": correct,
        "exact_match": accuracy,
        "avg_tps": avg_tps,
        "tps_std": tps_std,
        "avg_per_sample_latency_sec": avg_latency,
        "latency_std": latency_std,
        "avg_accept_rate": float(np.mean(accept_rates)) if accept_rates else 0,
        "avg_m1_chr": float(np.mean(chr_values)) if chr_values else 0,
        "avg_gpu_utilization_pct": avg_gpu_util,
        "avg_gpu_memory_used_mb": avg_gpu_mem,
        "sample_texts": sample_texts[:3],
    }


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

def main():
    write_pid()
    start_time = datetime.now()
    print(f"[batch_sensitivity] Starting at {start_time.isoformat()}")
    print(f"[batch_sensitivity] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[batch_sensitivity] Batch sizes: {BATCH_SIZES}")
    print(f"[batch_sensitivity] Configs: {[c['name'] for c in CONFIGS]}")

    # ── VRAM Profile ────────────────────────────────────────────────
    torch.cuda.empty_cache()
    gc.collect()
    vram_before = torch.cuda.memory_allocated() / 1024**2

    # ── Load Model ──────────────────────────────────────────────────
    from transformers import AutoTokenizer, AutoModel

    print(f"\n[batch_sensitivity] Loading LLaDA-8B-Instruct from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    vram_after_model = torch.cuda.memory_allocated() / 1024**2
    print(f"[batch_sensitivity] Model VRAM: {vram_after_model:.0f} MB")

    # Create generator
    generator = BatchedThreeWayGenerator(model, tokenizer, device="cuda")

    # ── Load Data ───────────────────────────────────────────────────
    set_seed(SEED)
    gsm8k_data = load_gsm8k(N_GSM8K, seed=SEED)
    print(f"[batch_sensitivity] Loaded {len(gsm8k_data)} GSM8K samples")

    # ── Run Experiments ─────────────────────────────────────────────
    all_results = {}
    total_configs = len(CONFIGS) * len(BATCH_SIZES)
    config_idx = 0

    for cfg in CONFIGS:
        config_name = cfg["name"]
        config_key = cfg["config_key"]
        print(f"\n{'='*70}")
        print(f"Config: {config_name} ({config_key})")
        print(f"{'='*70}")

        config_results = {}

        for bs in BATCH_SIZES:
            config_idx += 1
            print(f"\n--- Batch size = {bs} ({config_idx}/{total_configs}) ---")
            set_seed(SEED)

            torch.cuda.empty_cache()
            gc.collect()

            result = evaluate_batched(generator, gsm8k_data, cfg, bs)

            # Compute speedup and QAS
            baseline_tps = PILOT_BASELINE["gsm8k"]["avg_tps"]
            baseline_acc = PILOT_BASELINE["gsm8k"]["exact_match"]

            speedup = result["avg_tps"] / baseline_tps if baseline_tps > 0 else 0
            acc_ret = result["exact_match"] / baseline_acc if baseline_acc > 0 else 0
            qas = speedup * acc_ret

            # Ortho relative to best individual method
            ref_key = cfg["ref_best_qas_key"]
            ref_qas = INDIVIDUAL_QAS[ref_key]["gsm8k"]["qas"]
            ortho = qas / ref_qas if ref_qas > 0 else 0

            result["speedup"] = speedup
            result["acc_retention"] = acc_ret
            result["qas"] = qas
            result["ortho"] = ortho
            result["ref_individual_qas"] = ref_qas
            result["ref_individual_key"] = ref_key

            config_results[f"bs_{bs}"] = result

            print(f"  Results: acc={result['exact_match']:.3f}, "
                  f"tps={result['avg_tps']:.1f}, "
                  f"speedup={speedup:.2f}x, "
                  f"acc_ret={acc_ret:.3f}, "
                  f"qas={qas:.3f}, "
                  f"ortho={ortho:.3f}, "
                  f"latency={result['avg_per_sample_latency_sec']:.3f}s/sample, "
                  f"gpu_util={result['avg_gpu_utilization_pct']:.0f}%")

            report_progress(config_idx, total_configs, {
                "config": config_key,
                "batch_size": bs,
                "accuracy": result["exact_match"],
                "tps": result["avg_tps"],
                "speedup": speedup,
            })

        all_results[config_key] = config_results

    # ── GPU Profile ─────────────────────────────────────────────────
    vram_info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": int(torch.cuda.get_device_properties(0).total_memory / 1024**2),
        "vram_used_mb": int(torch.cuda.memory_allocated() / 1024**2),
        "vram_reserved_mb": int(torch.cuda.memory_reserved() / 1024**2),
    }

    # Write GPU profile
    gpu_profile = {
        "gpu_name": vram_info["gpu_name"],
        "vram_total_mb": vram_info["vram_total_mb"],
        "model_vram_mb": int(vram_after_model),
        "note": "LLaDA-8B bf16, batch_size varies",
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(json.dumps(gpu_profile, indent=2))

    # ── Compute Summary Statistics ──────────────────────────────────
    summary_table = []
    pass_criteria_met = True

    for cfg in CONFIGS:
        config_key = cfg["config_key"]
        config_results = all_results[config_key]

        bs1_result = config_results.get("bs_1", {})
        bs8_result = config_results.get("bs_8", {})

        # Check pass criteria: accuracy drop from bs=1 to bs=8 < 5%
        if bs1_result and bs8_result:
            acc_drop = bs1_result.get("exact_match", 0) - bs8_result.get("exact_match", 0)
            if abs(acc_drop) > 0.05:
                pass_criteria_met = False

        for bs in BATCH_SIZES:
            bs_key = f"bs_{bs}"
            r = config_results.get(bs_key, {})
            summary_table.append({
                "Config": cfg["name"],
                "BatchSize": bs,
                "Accuracy": r.get("exact_match", 0),
                "TPS": round(r.get("avg_tps", 0), 1),
                "Speedup": round(r.get("speedup", 0), 3),
                "AccRet": round(r.get("acc_retention", 0), 3),
                "QAS": round(r.get("qas", 0), 3),
                "Ortho": round(r.get("ortho", 0), 3),
                "Latency_s": round(r.get("avg_per_sample_latency_sec", 0), 3),
                "AcceptRate": round(r.get("avg_accept_rate", 0), 3),
                "M1_CHR": round(r.get("avg_m1_chr", 0), 3),
                "GPU_Util_pct": round(r.get("avg_gpu_utilization_pct", 0), 1),
                "GPU_Mem_MB": round(r.get("avg_gpu_memory_used_mb", 0), 0),
            })

    # Check all batch sizes have valid TPS
    for cfg in CONFIGS:
        config_key = cfg["config_key"]
        for bs in BATCH_SIZES:
            bs_key = f"bs_{bs}"
            r = all_results[config_key].get(bs_key, {})
            if r.get("avg_tps", 0) <= 0:
                pass_criteria_met = False

    # ── Batch Scaling Analysis ──────────────────────────────────────
    scaling_analysis = {}
    for cfg in CONFIGS:
        config_key = cfg["config_key"]
        config_results = all_results[config_key]

        bs1 = config_results.get("bs_1", {})
        bs4 = config_results.get("bs_4", {})
        bs8 = config_results.get("bs_8", {})

        if bs1.get("avg_tps", 0) > 0:
            scaling_analysis[config_key] = {
                "tps_scaling": {
                    "bs1_tps": round(bs1.get("avg_tps", 0), 1),
                    "bs4_tps": round(bs4.get("avg_tps", 0), 1),
                    "bs8_tps": round(bs8.get("avg_tps", 0), 1),
                    "bs4_over_bs1": round(bs4.get("avg_tps", 0) / bs1.get("avg_tps", 1), 2),
                    "bs8_over_bs1": round(bs8.get("avg_tps", 0) / bs1.get("avg_tps", 1), 2),
                },
                "accuracy_stability": {
                    "bs1_acc": bs1.get("exact_match", 0),
                    "bs4_acc": bs4.get("exact_match", 0),
                    "bs8_acc": bs8.get("exact_match", 0),
                    "max_acc_drop": round(
                        max(abs(bs1.get("exact_match", 0) - bs4.get("exact_match", 0)),
                            abs(bs1.get("exact_match", 0) - bs8.get("exact_match", 0))), 3
                    ),
                },
                "accept_rate_change": {
                    "bs1": round(bs1.get("avg_accept_rate", 0), 4),
                    "bs4": round(bs4.get("avg_accept_rate", 0), 4),
                    "bs8": round(bs8.get("avg_accept_rate", 0), 4),
                },
                "m1_chr_change": {
                    "bs1": round(bs1.get("avg_m1_chr", 0), 4),
                    "bs4": round(bs4.get("avg_m1_chr", 0), 4),
                    "bs8": round(bs8.get("avg_m1_chr", 0), 4),
                },
                "latency_change": {
                    "bs1": round(bs1.get("avg_per_sample_latency_sec", 0), 3),
                    "bs4": round(bs4.get("avg_per_sample_latency_sec", 0), 3),
                    "bs8": round(bs8.get("avg_per_sample_latency_sec", 0), 3),
                },
                "ortho_change": {
                    "bs1": round(bs1.get("ortho", 0), 3),
                    "bs4": round(bs4.get("ortho", 0), 3),
                    "bs8": round(bs8.get("ortho", 0), 3),
                },
            }

    # ── Build Output ────────────────────────────────────────────────
    elapsed_min = (datetime.now() - start_time).total_seconds() / 60

    output = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "iteration": 2,
        "model": "LLaDA-8B-Instruct",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(elapsed_min, 1),
        "seed": SEED,
        "pilot_config": {
            "gsm8k_samples": N_GSM8K,
            "gen_length": GEN_LENGTH,
            "steps": T_FULL,
        },
        "baseline_reference": PILOT_BASELINE,
        "qas_formula": "QAS = Speedup * Accuracy_Retention (corrected, no 0.5x penalty)",
        "ortho_formula": "Ortho = QAS(composition) / max(QAS(individual_methods))",
        "pass_criteria": {
            "requirement": "All batch sizes produce valid TPS AND accuracy drop <= 5% from bs=1 to bs=8",
            "met": pass_criteria_met,
        },
        "configs_tested": [cfg["config_key"] for cfg in CONFIGS],
        "batch_sizes": BATCH_SIZES,
        "all_results": all_results,
        "summary_table": summary_table,
        "scaling_analysis": scaling_analysis,
        "vram": vram_info,
    }

    # ── Write Results ───────────────────────────────────────────────
    out_path = RESULTS_DIR / "batch_sensitivity.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n[batch_sensitivity] Results saved to {out_path}")

    # ── Write Pilot Summary ─────────────────────────────────────────
    pilot_dir = WORKSPACE / "exp" / "results" / "pilots"
    pilot_dir.mkdir(parents=True, exist_ok=True)

    pilot_summary = {
        "task_id": TASK_ID,
        "go_no_go": "GO" if pass_criteria_met else "REFINE",
        "confidence": 0.80 if pass_criteria_met else 0.50,
        "pass_criteria_met": pass_criteria_met,
        "key_findings": [],
        "scaling_analysis": scaling_analysis,
    }

    # Summarize key findings
    for cfg in CONFIGS:
        config_key = cfg["config_key"]
        sa = scaling_analysis.get(config_key, {})
        tps_sc = sa.get("tps_scaling", {})
        acc_stab = sa.get("accuracy_stability", {})

        finding = (
            f"{cfg['name']}: bs1->bs8 TPS scaling "
            f"{tps_sc.get('bs8_over_bs1', 0):.1f}x, "
            f"max acc drop {acc_stab.get('max_acc_drop', 0):.3f}"
        )
        pilot_summary["key_findings"].append(finding)

    (pilot_dir / f"{TASK_ID}_pilot_summary.json").write_text(
        json.dumps(pilot_summary, indent=2, default=str)
    )

    # ── Print Summary ───────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BATCH SENSITIVITY SUMMARY")
    print(f"{'='*70}")
    print(f"Pass criteria met: {pass_criteria_met}")
    for row in summary_table:
        print(f"  {row['Config']:30s} bs={row['BatchSize']}: "
              f"acc={row['Accuracy']:.3f}, tps={row['TPS']:.0f}, "
              f"speedup={row['Speedup']:.2f}x, qas={row['QAS']:.3f}, "
              f"ortho={row['Ortho']:.3f}, lat={row['Latency_s']:.3f}s")

    for cfg in CONFIGS:
        config_key = cfg["config_key"]
        sa = scaling_analysis.get(config_key, {})
        tps_sc = sa.get("tps_scaling", {})
        print(f"\n  {cfg['name']}: TPS scaling: "
              f"bs1={tps_sc.get('bs1_tps', 0):.0f}, "
              f"bs4={tps_sc.get('bs4_tps', 0):.0f} ({tps_sc.get('bs4_over_bs1', 0):.1f}x), "
              f"bs8={tps_sc.get('bs8_tps', 0):.0f} ({tps_sc.get('bs8_over_bs1', 0):.1f}x)")

    mark_done("success", f"Completed in {elapsed_min:.1f} min")
    print(f"\n[batch_sensitivity] Completed in {elapsed_min:.1f} minutes.")


if __name__ == "__main__":
    main()
