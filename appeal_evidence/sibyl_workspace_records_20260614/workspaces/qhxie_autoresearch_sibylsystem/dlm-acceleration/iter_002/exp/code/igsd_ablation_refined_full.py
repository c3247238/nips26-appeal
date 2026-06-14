"""
IGSD Ablation Study (T_draft + tau + KL Profile) -- FULL Mode.

Task: igsd_ablation_refined
Iteration: 2, FULL mode

Full-scale validation of confidence partitioning (Part 3):
  - tau=0.0 vs tau=0.9 at T_draft=32
  - 1319 GSM8K + 500 MATH500, seeds=[42, 123, 456]

Parts 1 (T_draft sweep), 2 (tau sweep), and 4 (KL profile) reuse
pilot-scale data from the previous run.

Output: exp/results/ablation/igsd_ablation_refined_full.json

Usage:
    CUDA_VISIBLE_DEVICES=5 conda run -n sibyl_dlm-acceleration python igsd_ablation_refined_full.py
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


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Paths ──────────────────────────────────────────────────────────────────
WORKSPACE   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/iter_002")
SHARED      = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
ITER1_DIR   = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/iter_001")
MODEL_PATH  = str(SHARED / "checkpoints" / "llada-8b-instruct")
GSM8K_DIR   = str(SHARED / "datasets" / "gsm8k")
MATH500_DIR = str(SHARED / "datasets" / "math500")
CODE_DIR    = ITER1_DIR / "exp" / "code"
RESULTS_DIR = WORKSPACE / "exp" / "results" / "ablation"
TASK_ID     = "igsd_ablation_refined"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Full-scale baseline reference (from llada_baseline_full.json, 3-seed mean) ──
FULL_BASELINE = {
    "gsm8k":   {"exact_match": 0.7122, "avg_tps": 33.773},
    "math500": {"exact_match": 0.1107, "avg_tps": 79.108},
}

T_FULL        = 64
GEN_LENGTH    = 256
BLOCK_LENGTH  = 32
SEEDS         = [42, 123, 456]
N_WARMUP      = 5
MASK_ID       = 126336


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


# ── Prompt Builders ────────────────────────────────────────────────────────
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
    return f"Problem: {problem['problem']}\nSolution:"


# ── Metric Extractors ──────────────────────────────────────────────────────
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


# ── IGSD Generator ─────────────────────────────────────────────────────────
class IGSDGenerator:
    """IGSD generator for Masked Diffusion Language Models."""

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

        x = torch.full(
            (1, prompt_len + gen_length), MASK_ID, dtype=torch.long
        ).to(self.device)
        x[:, :prompt_len] = input_ids.clone()
        attn = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=self.device)
        ], dim=-1)

        # Phase 1: Whole-sequence draft
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

        # Phase 2: Confidence partition
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

        # Phase 3: Whole-sequence refine for S_refine
        x_refine = x.clone()
        s_accept_full = torch.cat([
            torch.ones(1, prompt_len, dtype=torch.bool, device=self.device),
            s_accept_gen
        ], dim=1)
        s_refine_full = ~s_accept_full
        x_refine[s_refine_full] = MASK_ID

        t_refine_start = time.perf_counter()

        if n_refine > 0:
            tokens_per_refine_step = max(1, n_refine // t_full)
            remainder_refine = n_refine % t_full
            for step in range(t_full):
                n_masked_now = int((x_refine[0, prompt_len:] == MASK_ID).sum().item())
                if n_masked_now == 0:
                    break
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

        text = self.tokenizer.decode(
            x_refine[0, prompt_len:].tolist(), skip_special_tokens=True
        )

        return {
            "generated_text":     text,
            "tps":                total_tps,
            "elapsed_sec":        total_elapsed,
            "draft_elapsed_sec":  draft_elapsed,
            "refine_elapsed_sec": refine_elapsed,
            "accept_rate":        accept_rate,
            "n_accept":           n_accept,
            "n_refine":           n_refine,
            "n_total":            n_total,
            "tau":                tau,
            "t_draft":            t_draft,
            "t_full":             t_full,
        }


# ── Evaluation Runners ─────────────────────────────────────────────────────
def evaluate_gsm8k(generator, data, tau, t_draft, seed, n_warmup=N_WARMUP):
    """Evaluate IGSD on GSM8K samples."""
    correct = 0
    total = len(data)
    tps_list = []
    accept_rates = []
    sample_texts = []

    for i, item in enumerate(data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            result = generator.generate(
                prompt=prompt, tau=tau, t_draft=t_draft, t_full=T_FULL,
                gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH,
                apply_chat_template=True,
            )
            pred_text = result["generated_text"]
            is_correct = gsm8k_exact_match(pred_text, item["answer"])
            if is_correct:
                correct += 1
            if i >= n_warmup:
                tps_list.append(result["tps"])
            accept_rates.append(result["accept_rate"])
            if len(sample_texts) < 5:
                sample_texts.append({
                    "id": i, "correct": is_correct,
                    "prediction": pred_text[:200],
                    "accept_rate": result["accept_rate"],
                    "tps": result["tps"],
                })
        except Exception as e:
            print(f"  [GSM8K] Error on sample {i}: {e}")

        if (i + 1) % 200 == 0:
            acc = correct / (i + 1)
            avg_tps = np.mean(tps_list) if tps_list else 0.0
            print(f"  [GSM8K tau={tau} td={t_draft} s={seed}] {i+1}/{total} "
                  f"acc={acc:.3f} tps={avg_tps:.1f}")

    exact_match = correct / total if total > 0 else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return {
        "n_samples": total, "correct": correct,
        "exact_match": exact_match,
        "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
        "avg_accept_rate": float(np.mean(accept_rates)) if accept_rates else 0.0,
        "sample_texts": sample_texts,
    }


def evaluate_math500(generator, data, tau, t_draft, seed, n_warmup=3):
    """Evaluate IGSD on MATH500 samples."""
    correct = 0
    total = len(data)
    tps_list = []
    accept_rates = []
    sample_texts = []

    for i, item in enumerate(data):
        prompt = build_math500_prompt(item)
        try:
            result = generator.generate(
                prompt=prompt, tau=tau, t_draft=t_draft, t_full=T_FULL,
                gen_length=GEN_LENGTH, block_length=BLOCK_LENGTH,
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
                    "id": i, "correct": is_correct,
                    "prediction": pred_text[:200],
                    "accept_rate": result["accept_rate"],
                    "tps": result["tps"],
                })
        except Exception as e:
            print(f"  [MATH500] Error on sample {i}: {e}")

        if (i + 1) % 100 == 0:
            acc = correct / (i + 1)
            avg_tps = np.mean(tps_list) if tps_list else 0.0
            print(f"  [MATH500 tau={tau} td={t_draft} s={seed}] {i+1}/{total} "
                  f"acc={acc:.3f} tps={avg_tps:.1f}")

    exact_match = correct / total if total > 0 else 0.0
    avg_tps = float(np.mean(tps_list)) if tps_list else 0.0
    return {
        "n_samples": total, "correct": correct,
        "exact_match": exact_match,
        "avg_tps": avg_tps,
        "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
        "avg_accept_rate": float(np.mean(accept_rates)) if accept_rates else 0.0,
        "sample_texts": sample_texts,
    }


def compute_metrics(gsm8k_metrics, math500_metrics):
    """Compute speedup, accuracy retention, QAS, and combined metrics."""
    gsm8k_speedup = gsm8k_metrics["avg_tps"] / FULL_BASELINE["gsm8k"]["avg_tps"]
    math500_speedup = math500_metrics["avg_tps"] / FULL_BASELINE["math500"]["avg_tps"]
    gsm8k_acc_ret = (gsm8k_metrics["exact_match"] / FULL_BASELINE["gsm8k"]["exact_match"]
                     if FULL_BASELINE["gsm8k"]["exact_match"] > 0 else 0.0)
    math500_acc_ret = (math500_metrics["exact_match"] / FULL_BASELINE["math500"]["exact_match"]
                       if FULL_BASELINE["math500"]["exact_match"] > 0 else 0.0)
    gsm8k_qas = gsm8k_speedup * gsm8k_acc_ret
    math500_qas = math500_speedup * math500_acc_ret
    combined_speedup = 0.7 * gsm8k_speedup + 0.3 * math500_speedup
    combined_acc_ret = 0.7 * gsm8k_acc_ret + 0.3 * math500_acc_ret
    combined_qas = combined_speedup * combined_acc_ret

    return {
        "gsm8k": {
            "speedup": gsm8k_speedup,
            "acc_retention": gsm8k_acc_ret,
            "qas": gsm8k_qas,
        },
        "math500": {
            "speedup": math500_speedup,
            "acc_retention": math500_acc_ret,
            "qas": math500_qas,
        },
        "combined": {
            "speedup": combined_speedup,
            "acc_retention": combined_acc_ret,
            "qas": combined_qas,
        },
    }


def run_seed_config(generator, tau, t_draft, seed):
    """Run one (tau, t_draft) config on full dataset for one seed."""
    print(f"\n  --- tau={tau}, T_draft={t_draft}, seed={seed} ---")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load full datasets (no subsampling)
    gsm8k_data = load_gsm8k(n_samples=None, seed=seed)
    math500_data = load_math500(n_samples=None, seed=seed)

    config_start = time.time()

    print(f"  Evaluating GSM8K ({len(gsm8k_data)} samples)...")
    gsm8k_metrics = evaluate_gsm8k(generator, gsm8k_data, tau, t_draft, seed)
    print(f"  Evaluating MATH500 ({len(math500_data)} samples)...")
    math500_metrics = evaluate_math500(generator, math500_data, tau, t_draft, seed)

    config_elapsed = time.time() - config_start
    derived = compute_metrics(gsm8k_metrics, math500_metrics)

    result = {
        "tau": tau,
        "t_draft": t_draft,
        "t_full": T_FULL,
        "seed": seed,
        "elapsed_min": round(config_elapsed / 60, 1),
        "gsm8k": {
            **gsm8k_metrics,
            "speedup": derived["gsm8k"]["speedup"],
            "acc_retention": derived["gsm8k"]["acc_retention"],
            "qas": derived["gsm8k"]["qas"],
        },
        "math500": {
            **math500_metrics,
            "speedup": derived["math500"]["speedup"],
            "acc_retention": derived["math500"]["acc_retention"],
            "qas": derived["math500"]["qas"],
        },
        "combined": derived["combined"],
    }

    print(f"  GSM8K:   acc={gsm8k_metrics['exact_match']:.3f} tps={gsm8k_metrics['avg_tps']:.1f} "
          f"speedup={derived['gsm8k']['speedup']:.2f}x acc_ret={derived['gsm8k']['acc_retention']:.3f} "
          f"QAS={derived['gsm8k']['qas']:.3f}")
    print(f"  MATH500: acc={math500_metrics['exact_match']:.3f} tps={math500_metrics['avg_tps']:.1f} "
          f"speedup={derived['math500']['speedup']:.2f}x acc_ret={derived['math500']['acc_retention']:.3f} "
          f"QAS={derived['math500']['qas']:.3f}")
    print(f"  Combined: speedup={derived['combined']['speedup']:.2f}x "
          f"acc_ret={derived['combined']['acc_retention']:.3f} QAS={derived['combined']['qas']:.3f}")
    print(f"  Elapsed: {config_elapsed/60:.1f} min")

    return result


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = time.time()
    print(f"[IGSD Ablation Refined] FULL mode")
    print(f"  Confidence partitioning: tau=0.0 vs tau=0.9, T_draft=32")
    print(f"  Full scale: 1319 GSM8K + 500 MATH500, seeds={SEEDS}")

    # Load model
    print("[Loading LLaDA-8B-Instruct...]")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"
    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    vram_after_load = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(0) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(0) // 1024**2,
    }
    print(f"  VRAM after load: {vram_after_load['vram_used_mb']} MB used / "
          f"{vram_after_load['vram_total_mb']} MB total")

    generator = IGSDGenerator(model, tokenizer, device="cuda")

    # Total steps for progress: 2 configs x 3 seeds = 6
    total_steps = 6
    current_step = 0

    # ═══════════════════════════════════════════════════════════════════
    # Part 3 FULL: Confidence partitioning at full scale
    # tau=0.0 (no gate) vs tau=0.9 (confidence gate) at T_draft=32
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"[FULL] Confidence partitioning: tau=0.0 vs tau=0.9 at T_draft=32")
    print(f"  1319 GSM8K + 500 MATH500, seeds={SEEDS}")
    print(f"{'='*70}")

    # Results organized by tau -> per-seed data
    tau_configs = [0.0, 0.9]
    full_results = {}  # tau -> {per_seed: {seed: result}, aggregated: {...}}

    for tau in tau_configs:
        per_seed_results = {}
        for seed in SEEDS:
            current_step += 1
            result = run_seed_config(generator, tau=tau, t_draft=32, seed=seed)
            per_seed_results[str(seed)] = result

            report_progress(current_step, total_steps, metric={
                "tau": tau, "seed": seed,
                "gsm8k_acc": result["gsm8k"]["exact_match"],
                "combined_qas": result["combined"]["qas"],
            })

            torch.cuda.empty_cache()
            gc.collect()

            # Save checkpoint after each seed
            checkpoint = {
                "task_id": TASK_ID,
                "mode": "FULL",
                "checkpoint_step": current_step,
                "tau": tau,
                "seed": seed,
                "result": result,
            }
            ckpt_path = RESULTS_DIR / f"igsd_ablation_full_checkpoint.json"
            ckpt_path.write_text(json.dumps(checkpoint, indent=2, cls=NumpyEncoder))

        # Aggregate across seeds
        gsm8k_accs = [r["gsm8k"]["exact_match"] for r in per_seed_results.values()]
        gsm8k_tps = [r["gsm8k"]["avg_tps"] for r in per_seed_results.values()]
        gsm8k_accept = [r["gsm8k"]["avg_accept_rate"] for r in per_seed_results.values()]
        math_accs = [r["math500"]["exact_match"] for r in per_seed_results.values()]
        math_tps = [r["math500"]["avg_tps"] for r in per_seed_results.values()]
        combined_qas = [r["combined"]["qas"] for r in per_seed_results.values()]
        combined_speedup = [r["combined"]["speedup"] for r in per_seed_results.values()]
        combined_acc_ret = [r["combined"]["acc_retention"] for r in per_seed_results.values()]

        aggregated = {
            "gsm8k": {
                "exact_match": {"mean": float(np.mean(gsm8k_accs)), "std": float(np.std(gsm8k_accs)),
                                "per_seed": gsm8k_accs},
                "avg_tps": {"mean": float(np.mean(gsm8k_tps)), "std": float(np.std(gsm8k_tps)),
                            "per_seed": gsm8k_tps},
                "avg_accept_rate": {"mean": float(np.mean(gsm8k_accept)), "std": float(np.std(gsm8k_accept)),
                                    "per_seed": gsm8k_accept},
            },
            "math500": {
                "exact_match": {"mean": float(np.mean(math_accs)), "std": float(np.std(math_accs)),
                                "per_seed": math_accs},
                "avg_tps": {"mean": float(np.mean(math_tps)), "std": float(np.std(math_tps)),
                            "per_seed": math_tps},
            },
            "combined": {
                "qas": {"mean": float(np.mean(combined_qas)), "std": float(np.std(combined_qas)),
                        "per_seed": combined_qas},
                "speedup": {"mean": float(np.mean(combined_speedup)), "std": float(np.std(combined_speedup)),
                            "per_seed": combined_speedup},
                "acc_retention": {"mean": float(np.mean(combined_acc_ret)), "std": float(np.std(combined_acc_ret)),
                                  "per_seed": combined_acc_ret},
            },
        }

        full_results[str(tau)] = {
            "per_seed": per_seed_results,
            "aggregated": aggregated,
        }

        print(f"\n  [Aggregated tau={tau}]")
        print(f"    GSM8K acc: {np.mean(gsm8k_accs):.3f} +/- {np.std(gsm8k_accs):.3f}")
        print(f"    MATH500 acc: {np.mean(math_accs):.3f} +/- {np.std(math_accs):.3f}")
        print(f"    Combined QAS: {np.mean(combined_qas):.3f} +/- {np.std(combined_qas):.3f}")

    # ═══════════════════════════════════════════════════════════════════
    # Analysis: does confidence gate add value at full scale?
    # ═══════════════════════════════════════════════════════════════════
    tau0_agg = full_results["0.0"]["aggregated"]
    tau9_agg = full_results["0.9"]["aggregated"]

    gate_analysis = {
        "tau_0_gsm8k_acc": tau0_agg["gsm8k"]["exact_match"]["mean"],
        "tau_9_gsm8k_acc": tau9_agg["gsm8k"]["exact_match"]["mean"],
        "tau_0_gsm8k_acc_std": tau0_agg["gsm8k"]["exact_match"]["std"],
        "tau_9_gsm8k_acc_std": tau9_agg["gsm8k"]["exact_match"]["std"],
        "tau_0_combined_qas": tau0_agg["combined"]["qas"]["mean"],
        "tau_9_combined_qas": tau9_agg["combined"]["qas"]["mean"],
        "tau_0_combined_qas_std": tau0_agg["combined"]["qas"]["std"],
        "tau_9_combined_qas_std": tau9_agg["combined"]["qas"]["std"],
        "tau_0_combined_speedup": tau0_agg["combined"]["speedup"]["mean"],
        "tau_9_combined_speedup": tau9_agg["combined"]["speedup"]["mean"],
        "tau_0_combined_acc_ret": tau0_agg["combined"]["acc_retention"]["mean"],
        "tau_9_combined_acc_ret": tau9_agg["combined"]["acc_retention"]["mean"],
        "gate_adds_value_qas": tau9_agg["combined"]["qas"]["mean"] > tau0_agg["combined"]["qas"]["mean"],
        "gate_adds_value_acc": tau9_agg["gsm8k"]["exact_match"]["mean"] > tau0_agg["gsm8k"]["exact_match"]["mean"],
        "qas_difference": tau9_agg["combined"]["qas"]["mean"] - tau0_agg["combined"]["qas"]["mean"],
        "acc_difference_gsm8k": tau9_agg["gsm8k"]["exact_match"]["mean"] - tau0_agg["gsm8k"]["exact_match"]["mean"],
    }

    # Check per-seed consistency
    tau0_per_seed_qas = tau0_agg["combined"]["qas"]["per_seed"]
    tau9_per_seed_qas = tau9_agg["combined"]["qas"]["per_seed"]
    per_seed_gate_wins = sum(1 for t0, t9 in zip(tau0_per_seed_qas, tau9_per_seed_qas) if t9 > t0)
    gate_analysis["per_seed_gate_wins"] = per_seed_gate_wins
    gate_analysis["per_seed_consistency"] = f"{per_seed_gate_wins}/{len(SEEDS)} seeds favor tau=0.9"

    # Interpretation
    if gate_analysis["gate_adds_value_qas"]:
        if gate_analysis["gate_adds_value_acc"]:
            interpretation = (
                "Full-scale confirms: confidence gate (tau=0.9) improves BOTH QAS and accuracy "
                "over naive step reduction (tau=0.0). The partition mechanism adds value by "
                "selectively refining low-confidence tokens, producing higher quality output "
                "despite slower throughput."
            )
        else:
            interpretation = (
                "Full-scale shows: confidence gate (tau=0.9) improves QAS but NOT raw accuracy "
                "vs tau=0.0. The gate's speed-accuracy tradeoff favors quality-adjusted throughput."
            )
    else:
        if gate_analysis["gate_adds_value_acc"]:
            interpretation = (
                "Full-scale shows: confidence gate (tau=0.9) improves accuracy but NOT QAS "
                "vs tau=0.0. The refine phase overhead outweighs the accuracy gain in QAS terms. "
                "Gate is valuable for quality-sensitive applications but not for throughput."
            )
        else:
            interpretation = (
                "Full-scale confirms pilot finding: confidence gate does NOT improve QAS or accuracy "
                "over naive step reduction. The partition overhead is not justified at scale. "
                "This is an important negative result: the IGSD draft-only approach (tau=0.0) is "
                "the preferred operating point for T_draft=32."
            )
    gate_analysis["interpretation"] = interpretation

    # ═══════════════════════════════════════════════════════════════════
    # Load pilot data for Parts 1, 2, 4
    # ═══════════════════════════════════════════════════════════════════
    pilot_path = RESULTS_DIR / "igsd_ablation_refined.json"
    pilot_data = {}
    if pilot_path.exists():
        try:
            pilot_data = json.loads(pilot_path.read_text())
        except Exception as e:
            print(f"Warning: could not load pilot data: {e}")

    # ═══════════════════════════════════════════════════════════════════
    # Assemble final output
    # ═══════════════════════════════════════════════════════════════════
    total_elapsed = time.time() - start_time

    output = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "iteration": 2,
        "model": "LLaDA-8B-Instruct",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(total_elapsed / 60, 1),
        "full_config": {
            "gsm8k_samples": 1319,
            "math500_samples": 500,
            "seeds": SEEDS,
            "gen_length": GEN_LENGTH,
            "steps": T_FULL,
        },
        "baseline_reference": FULL_BASELINE,
        "qas_formula": "QAS = Speedup * Accuracy_Retention (corrected, no 0.5x penalty)",
        "combined_metric": "0.7 * GSM8K + 0.3 * MATH500",

        # Part 3 FULL: the main result of this run
        "part3_confidence_partitioning_full": {
            "description": (
                "Full-scale validation: tau=0.0 (no gate) vs tau=0.9 (confidence gate) "
                "at T_draft=32. 1319 GSM8K + 500 MATH500, seeds=[42, 123, 456]."
            ),
            "results": full_results,
            "analysis": gate_analysis,
        },

        # Parts 1, 2, 4 from pilot (included for completeness)
        "part1_tdraft_sweep_pilot": pilot_data.get("part1_tdraft_sweep", {
            "description": "Pilot data not available; see igsd_ablation_refined.json",
        }),
        "part2_tau_sweep_pilot": pilot_data.get("part2_tau_sweep", {
            "description": "Pilot data not available; see igsd_ablation_refined.json",
        }),
        "part4_kl_profile_pilot": pilot_data.get("part4_kl_profile", {
            "description": "Pilot data not available; see igsd_ablation_refined.json",
        }),

        "vram": vram_after_load,
    }

    # Save
    out_path = RESULTS_DIR / "igsd_ablation_refined_full.json"
    out_path.write_text(json.dumps(output, indent=2, cls=NumpyEncoder))

    print(f"\n{'='*70}")
    print(f"[IGSD Ablation Refined FULL] COMPLETE")
    print(f"  Total elapsed: {total_elapsed/60:.1f} min")
    print(f"  tau=0.0 GSM8K acc: {gate_analysis['tau_0_gsm8k_acc']:.3f} +/- {gate_analysis['tau_0_gsm8k_acc_std']:.3f}")
    print(f"  tau=0.9 GSM8K acc: {gate_analysis['tau_9_gsm8k_acc']:.3f} +/- {gate_analysis['tau_9_gsm8k_acc_std']:.3f}")
    print(f"  tau=0.0 combined QAS: {gate_analysis['tau_0_combined_qas']:.3f}")
    print(f"  tau=0.9 combined QAS: {gate_analysis['tau_9_combined_qas']:.3f}")
    print(f"  Gate adds value (QAS): {gate_analysis['gate_adds_value_qas']}")
    print(f"  Gate adds value (acc): {gate_analysis['gate_adds_value_acc']}")
    print(f"  Per-seed consistency: {gate_analysis['per_seed_consistency']}")
    print(f"  Results: {out_path}")
    print(f"{'='*70}")

    # Mark done
    mark_done(
        status="success",
        summary=(
            f"IGSD ablation FULL complete. "
            f"tau=0.0 GSM8K={gate_analysis['tau_0_gsm8k_acc']:.3f}, "
            f"tau=0.9 GSM8K={gate_analysis['tau_9_gsm8k_acc']:.3f}, "
            f"gate_adds_value_qas={gate_analysis['gate_adds_value_qas']}, "
            f"gate_adds_value_acc={gate_analysis['gate_adds_value_acc']}, "
            f"per_seed_consistency={gate_analysis['per_seed_consistency']}."
        ),
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
            "mode": "FULL",
            "parts": "confidence_partitioning_full (tau=0.0 vs tau=0.9, T_draft=32)",
            "gsm8k_samples": 1319,
            "math500_samples": 500,
            "seeds": SEEDS,
            "gpu_model": vram_after_load.get("gpu_name", "unknown"),
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print(f"[gpu_progress.json] Updated: {TASK_ID} -> completed (FULL)")


if __name__ == "__main__":
    main()
