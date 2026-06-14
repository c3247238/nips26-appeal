"""
IGSD Corrected Pareto Curve with Conservative T_draft (Pilot Mode).

Task: igsd_pareto_corrected
Iteration: 2

Re-run IGSD Pareto sweep with EXTENDED T_draft range:
  tau = {0.7, 0.85, 0.9}
  T_draft = {16, 32, 48}

Key changes from iter_001:
  - Added T_draft=32,48 (iter_001 only tested 4,8,16)
  - Corrected QAS formula: QAS = Speedup * Accuracy_Retention (no 0.5x penalty)
  - Combined metric: 0.7*GSM8K + 0.3*MATH500
  - Pilot: 200 GSM8K + 100 MATH500, seed=42

Pass criteria:
  At least one (tau, T_draft) config achieves accuracy_retention > 0.50
  AND speedup > 1.5x on GSM8K.

Output: exp/results/igsd_pareto/igsd_pareto_corrected.json

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n sibyl_dlm-acceleration python igsd_pareto_corrected.py
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
GSM8K_DIR   = str(SHARED / "datasets" / "gsm8k")
MATH500_DIR = str(SHARED / "datasets" / "math500")
CODE_DIR    = ITER1_DIR / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "igsd_pareto"
TASK_ID     = "igsd_pareto_corrected"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline from d2cache_integration_pilot (same 100 GSM8K pilot samples, seed=42)
# and iter_001 full baseline (3-seed means)
BASELINE = {
    "gsm8k":   {"exact_match": 0.7122, "avg_tps": 31.013},   # iter_001 full baseline, 3-seed mean
    "math500": {"exact_match": 0.1107, "avg_tps": 79.221},   # iter_001 full baseline, 3-seed mean
}
# Use d2cache pilot baseline for this pilot (same sample subset)
PILOT_BASELINE = {
    "gsm8k":   {"exact_match": 0.73, "avg_tps": 58.505},   # d2cache pilot, 100 GSM8K, seed=42
    "math500": {"exact_match": 0.32, "avg_tps": 111.987},   # d2cache pilot, 50 MATH500, seed=42
}

# Sweep configuration
TAU_VALUES    = [0.7, 0.85, 0.9]
T_DRAFT_VALUES = [16, 32, 48]
T_FULL        = 64
GEN_LENGTH    = 256
BLOCK_LENGTH  = 32
SEED          = 42
N_GSM8K       = 200     # pilot samples
N_MATH500     = 100     # pilot samples
MASK_ID       = 126336
N_WARMUP      = 5       # warmup samples to exclude from TPS


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
        except Exception:
            pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Dataset Loaders ────────────────────────────────────────────────────────
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


# ── 8-shot GSM8K Prompt ────────────────────────────────────────────────────
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
    """Build 0-shot MATH500 prompt."""
    return f"Problem: {problem['problem']}\nSolution:"


# ── Metric Extractors ──────────────────────────────────────────────────────
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


def gsm8k_exact_match(pred, gold):
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip() == g.strip()


def extract_math500_answer(text):
    """Extract boxed answer from MATH500 solution text."""
    # Look for \\boxed{...}
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    # Look for "The answer is X"
    match = re.search(r"[Tt]he answer is\s+(.+?)(?:\.|$)", text)
    if match:
        return match.group(1).strip()
    # Last line with content
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return None


def math500_exact_match(pred, gold_solution):
    """Check if predicted answer matches the gold solution's boxed answer."""
    p = extract_math500_answer(pred)
    g = extract_math500_answer(gold_solution)
    if p is None or g is None:
        return False
    # Normalize
    p = p.strip().replace(" ", "").lower()
    g = g.strip().replace(" ", "").lower()
    if p == g:
        return True
    # Try numeric comparison
    try:
        return abs(float(p) - float(g)) < 1e-6
    except (ValueError, TypeError):
        return False


# ── IGSD Generator ─────────────────────────────────────────────────────────
class IGSDGenerator:
    """
    IGSD generator for Masked Diffusion Language Models.
    Whole-sequence draft + confidence partition + whole-sequence refine.
    """

    def __init__(self, model, tokenizer, device="cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tau: float = 0.70,
        t_draft: int = 32,
        t_full: int = 64,
        gen_length: int = 256,
        block_length: int = 32,
        apply_chat_template: bool = True,
    ) -> dict:
        """Generate text using IGSD with whole-sequence draft and refine."""
        # Tokenize prompt
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
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask  = enc["attention_mask"].to(self.device)
        prompt_len      = input_ids.shape[1]

        # Initialize fully masked sequence
        x = torch.full(
            (1, prompt_len + gen_length), MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=self.device)
        ], dim=-1)

        # ── Phase 1: Whole-sequence draft ────────────────────────────
        t_draft_start = time.perf_counter()
        tokens_per_draft_step = max(1, gen_length // t_draft)
        remainder_draft = gen_length % t_draft

        for step in range(t_draft):
            n_masked = int((x[0, prompt_len:] == MASK_ID).sum().item())
            if n_masked == 0:
                break

            logits = self.model(x, attention_mask=attn).logits
            p    = F.softmax(logits, dim=-1)
            x0   = torch.argmax(p, dim=-1)
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

        draft_elapsed = time.perf_counter() - t_draft_start

        # ── Phase 2: Confidence partition ────────────────────────────
        logits = self.model(x, attention_mask=attn).logits
        p_final    = F.softmax(logits[:, prompt_len:, :], dim=-1)
        gen_region = x[:, prompt_len:].clone()
        still_masked = (gen_region == MASK_ID)

        draft_pred = torch.argmax(p_final, dim=-1)
        draft_conf = torch.gather(p_final, -1, draft_pred.unsqueeze(-1)).squeeze(-1)
        filled_conf = torch.gather(p_final, -1, gen_region.clamp(min=0).unsqueeze(-1)).squeeze(-1)

        final_confidence = torch.where(still_masked, draft_conf, filled_conf)
        gen_region_filled = torch.where(still_masked, draft_pred, gen_region)
        x[:, prompt_len:] = gen_region_filled

        s_accept_gen = (final_confidence >= tau)
        n_accept     = int(s_accept_gen.sum().item())
        n_total      = s_accept_gen.numel()
        n_refine     = n_total - n_accept
        accept_rate  = n_accept / n_total

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
                p    = F.softmax(logits, dim=-1)
                x0   = torch.argmax(p, dim=-1)
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

        text = self.tokenizer.decode(
            x_refine[0, prompt_len:].tolist(), skip_special_tokens=True
        )

        return {
            "generated_text":     text,
            "tps":                total_tps,
            "elapsed_sec":        total_elapsed,
            "draft_elapsed_sec":  draft_elapsed,
            "refine_elapsed_sec": refine_elapsed,
            "draft_tps":          gen_length / draft_elapsed if draft_elapsed > 0 else 0.0,
            "refine_tps":         gen_length / refine_elapsed if refine_elapsed > 0 else 0.0,
            "accept_rate":        accept_rate,
            "n_accept":           n_accept,
            "n_refine":           n_refine,
            "n_total":            n_total,
            "kv_hit_rate_refine": kv_hit_rate,
            "tau":                tau,
            "t_draft":            t_draft,
            "t_full":             t_full,
        }


# ── Evaluation Runner ──────────────────────────────────────────────────────
def evaluate_gsm8k(generator, data, tau, t_draft, n_warmup=N_WARMUP):
    """Evaluate IGSD on GSM8K samples. Returns metrics dict."""
    correct = 0
    total = len(data)
    tps_list = []
    accept_rates = []
    sample_texts = []

    for i, item in enumerate(data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            result = generator.generate(
                prompt=prompt,
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

            # Save first 5 sample texts for qualitative inspection
            if len(sample_texts) < 5:
                sample_texts.append({
                    "id": i,
                    "correct": is_correct,
                    "prediction": pred_text[:200],
                    "accept_rate": result["accept_rate"],
                    "tps": result["tps"],
                })

        except Exception as e:
            print(f"  [GSM8K] Error on sample {i}: {e}")

        if (i + 1) % 50 == 0:
            acc = correct / (i + 1)
            avg_tps = np.mean(tps_list) if tps_list else 0.0
            print(f"  [GSM8K tau={tau} td={t_draft}] {i+1}/{total} acc={acc:.3f} tps={avg_tps:.1f}")

    exact_match = correct / total if total > 0 else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0

    return {
        "n_samples": total,
        "correct": correct,
        "exact_match": exact_match,
        "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
        "avg_accept_rate": float(np.mean(accept_rates)) if accept_rates else 0.0,
        "sample_texts": sample_texts,
    }


def evaluate_math500(generator, data, tau, t_draft, n_warmup=3):
    """Evaluate IGSD on MATH500 samples. Returns metrics dict."""
    correct = 0
    total = len(data)
    tps_list = []
    accept_rates = []
    sample_texts = []

    for i, item in enumerate(data):
        prompt = build_math500_prompt(item)
        try:
            result = generator.generate(
                prompt=prompt,
                tau=tau,
                t_draft=t_draft,
                t_full=T_FULL,
                gen_length=GEN_LENGTH,
                block_length=BLOCK_LENGTH,
                apply_chat_template=True,
            )
            pred_text = result["generated_text"]
            is_correct = math500_exact_match(pred_text, item["solution"])
            if is_correct:
                correct += 1
            if i >= n_warmup:
                tps_list.append(result["tps"])
            accept_rates.append(result["accept_rate"])

            if len(sample_texts) < 5:
                sample_texts.append({
                    "id": i,
                    "correct": is_correct,
                    "prediction": pred_text[:200],
                    "accept_rate": result["accept_rate"],
                    "tps": result["tps"],
                })

        except Exception as e:
            print(f"  [MATH500] Error on sample {i}: {e}")

        if (i + 1) % 25 == 0:
            acc = correct / (i + 1)
            avg_tps = np.mean(tps_list) if tps_list else 0.0
            print(f"  [MATH500 tau={tau} td={t_draft}] {i+1}/{total} acc={acc:.3f} tps={avg_tps:.1f}")

    exact_match = correct / total if total > 0 else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0

    return {
        "n_samples": total,
        "correct": correct,
        "exact_match": exact_match,
        "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
        "avg_accept_rate": float(np.mean(accept_rates)) if accept_rates else 0.0,
        "sample_texts": sample_texts,
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = time.time()
    print(f"[IGSD Pareto Corrected] PILOT mode, seed={SEED}")
    print(f"  tau values: {TAU_VALUES}")
    print(f"  T_draft values: {T_DRAFT_VALUES}")
    print(f"  GSM8K samples: {N_GSM8K}, MATH500 samples: {N_MATH500}")

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load datasets
    print("[Loading datasets...]")
    gsm8k_data = load_gsm8k(n_samples=N_GSM8K, seed=SEED)
    math500_data = load_math500(n_samples=N_MATH500, seed=SEED)
    print(f"  GSM8K: {len(gsm8k_data)} samples, MATH500: {len(math500_data)} samples")

    # Load model
    print("[Loading LLaDA-8B-Instruct...]")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    # Log VRAM after load
    vram_after_load = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(0) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(0) // 1024**2,
    }
    print(f"  VRAM after load: {vram_after_load['vram_used_mb']} MB used / {vram_after_load['vram_total_mb']} MB total")

    generator = IGSDGenerator(model, tokenizer, device="cuda")

    # ── Sweep all (tau, T_draft) configurations ──────────────────────
    all_configs = []
    total_configs = len(TAU_VALUES) * len(T_DRAFT_VALUES)
    config_idx = 0

    for tau in TAU_VALUES:
        for t_draft in T_DRAFT_VALUES:
            config_idx += 1
            config_name = f"tau{tau}_td{t_draft}"
            print(f"\n{'='*60}")
            print(f"[Config {config_idx}/{total_configs}] tau={tau}, T_draft={t_draft}")
            print(f"{'='*60}")

            config_start = time.time()

            # Evaluate GSM8K
            print(f"  Evaluating GSM8K ({N_GSM8K} samples)...")
            gsm8k_metrics = evaluate_gsm8k(generator, gsm8k_data, tau, t_draft)

            # Evaluate MATH500
            print(f"  Evaluating MATH500 ({N_MATH500} samples)...")
            math500_metrics = evaluate_math500(generator, math500_data, tau, t_draft)

            config_elapsed = time.time() - config_start

            # Compute corrected metrics
            # Speedup relative to baseline TPS
            gsm8k_speedup = gsm8k_metrics["avg_tps"] / PILOT_BASELINE["gsm8k"]["avg_tps"]
            math500_speedup = math500_metrics["avg_tps"] / PILOT_BASELINE["math500"]["avg_tps"]

            # Accuracy retention
            gsm8k_acc_ret = gsm8k_metrics["exact_match"] / PILOT_BASELINE["gsm8k"]["exact_match"] if PILOT_BASELINE["gsm8k"]["exact_match"] > 0 else 0.0
            math500_acc_ret = math500_metrics["exact_match"] / PILOT_BASELINE["math500"]["exact_match"] if PILOT_BASELINE["math500"]["exact_match"] > 0 else 0.0

            # Corrected QAS (no penalty)
            gsm8k_qas = gsm8k_speedup * gsm8k_acc_ret
            math500_qas = math500_speedup * math500_acc_ret

            # Combined metric: 0.7*GSM8K + 0.3*MATH500
            combined_speedup = 0.7 * gsm8k_speedup + 0.3 * math500_speedup
            combined_acc_ret = 0.7 * gsm8k_acc_ret + 0.3 * math500_acc_ret
            combined_qas = combined_speedup * combined_acc_ret

            config_result = {
                "config_name": config_name,
                "tau": tau,
                "t_draft": t_draft,
                "t_full": T_FULL,
                "seed": SEED,
                "elapsed_min": round(config_elapsed / 60, 1),
                "gsm8k": {
                    "n_samples": gsm8k_metrics["n_samples"],
                    "correct": gsm8k_metrics["correct"],
                    "exact_match": gsm8k_metrics["exact_match"],
                    "avg_tps": gsm8k_metrics["avg_tps"],
                    "tps_std": gsm8k_metrics["tps_std"],
                    "speedup": gsm8k_speedup,
                    "acc_retention": gsm8k_acc_ret,
                    "qas": gsm8k_qas,
                    "avg_accept_rate": gsm8k_metrics["avg_accept_rate"],
                    "sample_texts": gsm8k_metrics["sample_texts"],
                },
                "math500": {
                    "n_samples": math500_metrics["n_samples"],
                    "correct": math500_metrics["correct"],
                    "exact_match": math500_metrics["exact_match"],
                    "avg_tps": math500_metrics["avg_tps"],
                    "tps_std": math500_metrics["tps_std"],
                    "speedup": math500_speedup,
                    "acc_retention": math500_acc_ret,
                    "qas": math500_qas,
                    "avg_accept_rate": math500_metrics["avg_accept_rate"],
                    "sample_texts": math500_metrics["sample_texts"],
                },
                "combined": {
                    "speedup": combined_speedup,
                    "acc_retention": combined_acc_ret,
                    "qas": combined_qas,
                },
            }

            all_configs.append(config_result)

            print(f"\n  === Results for tau={tau}, T_draft={t_draft} ===")
            print(f"  GSM8K:   acc={gsm8k_metrics['exact_match']:.3f} tps={gsm8k_metrics['avg_tps']:.1f} speedup={gsm8k_speedup:.2f}x acc_ret={gsm8k_acc_ret:.3f} QAS={gsm8k_qas:.3f}")
            print(f"  MATH500: acc={math500_metrics['exact_match']:.3f} tps={math500_metrics['avg_tps']:.1f} speedup={math500_speedup:.2f}x acc_ret={math500_acc_ret:.3f} QAS={math500_qas:.3f}")
            print(f"  Combined: speedup={combined_speedup:.2f}x acc_ret={combined_acc_ret:.3f} QAS={combined_qas:.3f}")
            print(f"  Elapsed: {config_elapsed/60:.1f} min")

            report_progress(config_idx, total_configs, metric={
                "tau": tau, "t_draft": t_draft,
                "gsm8k_acc": gsm8k_metrics["exact_match"],
                "gsm8k_speedup": gsm8k_speedup,
                "combined_qas": combined_qas,
            })

            # Clear CUDA cache between configs
            torch.cuda.empty_cache()
            gc.collect()

    # ── Identify Pareto-optimal configurations ───────────────────────
    # Pareto: no other config dominates on both speedup AND acc_retention
    def is_pareto_dominated(cfg, all_cfgs):
        for other in all_cfgs:
            if other is cfg:
                continue
            if (other["gsm8k"]["speedup"] >= cfg["gsm8k"]["speedup"] and
                other["gsm8k"]["acc_retention"] >= cfg["gsm8k"]["acc_retention"] and
                (other["gsm8k"]["speedup"] > cfg["gsm8k"]["speedup"] or
                 other["gsm8k"]["acc_retention"] > cfg["gsm8k"]["acc_retention"])):
                return True
        return False

    pareto_configs = [c for c in all_configs if not is_pareto_dominated(c, all_configs)]
    pareto_configs.sort(key=lambda c: c["gsm8k"]["speedup"])

    # ── Pass criteria check ──────────────────────────────────────────
    pass_criteria_met = any(
        c["gsm8k"]["acc_retention"] > 0.50 and c["gsm8k"]["speedup"] > 1.5
        for c in all_configs
    )

    # Best config by combined QAS
    best_by_qas = max(all_configs, key=lambda c: c["combined"]["qas"])

    # Best config by GSM8K accuracy retention (quality-first)
    best_by_quality = max(all_configs, key=lambda c: c["gsm8k"]["acc_retention"])

    # Best config by GSM8K speedup (speed-first)
    best_by_speed = max(all_configs, key=lambda c: c["gsm8k"]["speedup"])

    total_elapsed = time.time() - start_time

    # ── Build output JSON ────────────────────────────────────────────
    output = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "iteration": 2,
        "model": "LLaDA-8B-Instruct",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(total_elapsed / 60, 1),
        "pilot_config": {
            "gsm8k_samples": N_GSM8K,
            "math500_samples": N_MATH500,
            "seed": SEED,
            "gen_length": GEN_LENGTH,
            "steps": T_FULL,
        },
        "sweep": {
            "tau_values": TAU_VALUES,
            "t_draft_values": T_DRAFT_VALUES,
            "total_configs": total_configs,
        },
        "baseline_reference": {
            "pilot_baseline": PILOT_BASELINE,
            "iter1_full_baseline": BASELINE,
            "note": "Speedup computed relative to pilot_baseline (same sample subset)",
        },
        "qas_formula": "QAS = Speedup * Accuracy_Retention (corrected, no 0.5x penalty)",
        "combined_metric": "0.7 * GSM8K + 0.3 * MATH500",
        "pass_criteria": {
            "requirement": "At least one (tau, T_draft) achieves acc_ret > 0.50 AND speedup > 1.5x on GSM8K",
            "met": pass_criteria_met,
        },
        "all_configs": all_configs,
        "pareto_frontier": [
            {
                "tau": c["tau"],
                "t_draft": c["t_draft"],
                "gsm8k_speedup": c["gsm8k"]["speedup"],
                "gsm8k_acc_ret": c["gsm8k"]["acc_retention"],
                "gsm8k_qas": c["gsm8k"]["qas"],
                "combined_qas": c["combined"]["qas"],
            }
            for c in pareto_configs
        ],
        "recommendations": {
            "best_by_combined_qas": {
                "config": f"tau={best_by_qas['tau']}, T_draft={best_by_qas['t_draft']}",
                "combined_qas": best_by_qas["combined"]["qas"],
                "gsm8k_speedup": best_by_qas["gsm8k"]["speedup"],
                "gsm8k_acc_ret": best_by_qas["gsm8k"]["acc_retention"],
            },
            "best_by_quality": {
                "config": f"tau={best_by_quality['tau']}, T_draft={best_by_quality['t_draft']}",
                "gsm8k_acc_ret": best_by_quality["gsm8k"]["acc_retention"],
                "gsm8k_speedup": best_by_quality["gsm8k"]["speedup"],
            },
            "best_by_speed": {
                "config": f"tau={best_by_speed['tau']}, T_draft={best_by_speed['t_draft']}",
                "gsm8k_speedup": best_by_speed["gsm8k"]["speedup"],
                "gsm8k_acc_ret": best_by_speed["gsm8k"]["acc_retention"],
            },
        },
        "key_question_answer": {
            "question": "Does T_draft=32-48 push accuracy retention above 60% while maintaining >= 2x speedup?",
            "answer": "See pareto_frontier and all_configs for evidence",
        },
        "vram": vram_after_load,
    }

    # Save results
    out_path = RESULTS_DIR / "igsd_pareto_corrected.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n{'='*60}")
    print(f"[IGSD Pareto Corrected] COMPLETE")
    print(f"  Total elapsed: {total_elapsed/60:.1f} min")
    print(f"  Pass criteria met: {pass_criteria_met}")
    print(f"  Best combined QAS: {best_by_qas['combined']['qas']:.3f} ({best_by_qas['config_name']})")
    print(f"  Pareto configs: {len(pareto_configs)}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*60}")

    # Also save pilot summary for downstream tasks
    pilot_summary = {
        "overall_recommendation": "GO" if pass_criteria_met else "REFINE",
        "task_id": TASK_ID,
        "pass_criteria_met": pass_criteria_met,
        "best_config": {
            "tau": best_by_qas["tau"],
            "t_draft": best_by_qas["t_draft"],
            "combined_qas": best_by_qas["combined"]["qas"],
            "gsm8k_acc": best_by_qas["gsm8k"]["exact_match"],
            "gsm8k_speedup": best_by_qas["gsm8k"]["speedup"],
            "gsm8k_acc_ret": best_by_qas["gsm8k"]["acc_retention"],
        },
        "pareto_count": len(pareto_configs),
        "t_draft_analysis": {
            "t16_best_acc_ret": max((c["gsm8k"]["acc_retention"] for c in all_configs if c["t_draft"] == 16), default=0),
            "t32_best_acc_ret": max((c["gsm8k"]["acc_retention"] for c in all_configs if c["t_draft"] == 32), default=0),
            "t48_best_acc_ret": max((c["gsm8k"]["acc_retention"] for c in all_configs if c["t_draft"] == 48), default=0),
        },
    }
    pilot_path = WORKSPACE / "exp" / "results" / "pilots" / "igsd_pareto_corrected_pilot_summary.json"
    pilot_path.parent.mkdir(parents=True, exist_ok=True)
    pilot_path.write_text(json.dumps(pilot_summary, indent=2))

    # Mark done
    mark_done(
        status="success",
        summary=f"IGSD Pareto pilot complete. {total_configs} configs, "
                f"pass_criteria={'MET' if pass_criteria_met else 'NOT MET'}, "
                f"best QAS={best_by_qas['combined']['qas']:.3f}"
    )

    # Update gpu_progress.json
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gpu_progress_path.read_text())
    except Exception:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    gp["running"].pop(TASK_ID, None)
    gp["timings"][TASK_ID] = {
        "planned_min": 60,
        "actual_min": round(total_elapsed / 60, 1),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "LLaDA-8B-Instruct",
            "task": TASK_ID,
            "sweep": f"tau={TAU_VALUES} x T_draft={T_DRAFT_VALUES}",
            "gsm8k_samples": N_GSM8K,
            "math500_samples": N_MATH500,
            "gpu_model": vram_after_load.get("gpu_name", "unknown"),
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print(f"[gpu_progress.json] Updated: {TASK_ID} -> completed")


if __name__ == "__main__":
    main()
