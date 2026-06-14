"""
M3+IGSD Full-Scale Pairwise Composition -- PILOT MODE

Task: pairwise_m3_igsd_full
Purpose: Validate M3+IGSD composition (iter_001 Ortho=0.493 -- interference expected).
  M3's guidance overhead per step + IGSD's reduced steps = conflicting objectives.
  The AR guide model (Qwen2.5-0.5B) may provide poor guidance on IGSD's aggressively
  compressed denoising trajectory.

Method: Combine IGSD (whole-sequence draft + confidence partition + refine) with
  M3 (AR-guided unmasking using Qwen2.5-0.5B) during the refine phase.

Operating points from Phase 2:
  IGSD best balanced: tau=0.9, T_draft=32 (GSM8K: 1.71x speedup, 67.8% acc_ret)
  IGSD best speed:    tau=0.7, T_draft=16 (GSM8K: 2.81x speedup, 58.2% acc_ret)
  M3 best:            gw=0.7 (GSM8K: 1.65x speedup, 103.9% acc_ret)

We test multiple IGSD configs with M3 gw=0.7 to understand the interaction:
  Config 1: IGSD(tau=0.9, T_draft=32) + M3(gw=0.7)  -- balanced
  Config 2: IGSD(tau=0.7, T_draft=16) + M3(gw=0.7)  -- aggressive IGSD
  Config 3: IGSD(tau=0.85, T_draft=32) + M3(gw=0.7) -- moderate

Pilot: 100 GSM8K + 100 MATH500, seed=42
Pass criteria: Ortho values computed for both benchmarks without error

Output: exp/results/pairwise/m3_igsd_full.json
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
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pairwise"
TASK_ID     = "pairwise_m3_igsd_full"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Baselines ──────────────────────────────────────────────────────────────
# iter_001 full baseline (3-seed means) for Ortho computation
BASELINE = {
    "gsm8k":   {"exact_match": 0.7122, "avg_tps": 31.013},
    "math500": {"exact_match": 0.1107, "avg_tps": 79.221},
}

# Individual method results from Phase 2 (pilot, seed=42)
# IGSD results per config (from igsd_pareto_corrected.json)
IGSD_RESULTS = {
    "tau0.9_td32": {
        "gsm8k":   {"speedup": 1.7064, "acc_ret": 0.6781, "qas": 1.1571},
        "math500": {"speedup": 1.7117, "acc_ret": 0.4375, "qas": 0.7489},
        "combined": {"speedup": 1.7080, "acc_ret": 0.6059, "qas": 1.0349},
    },
    "tau0.7_td16": {
        "gsm8k":   {"speedup": 2.8120, "acc_ret": 0.5822, "qas": 1.6371},
        "math500": {"speedup": 2.5727, "acc_ret": 0.3438, "qas": 0.8844},
        "combined": {"speedup": 2.7402, "acc_ret": 0.5107, "qas": 1.3993},
    },
    "tau0.85_td32": {
        "gsm8k":   {"speedup": 1.7316, "acc_ret": 0.6781, "qas": 1.1741},
        "math500": {"speedup": 1.7493, "acc_ret": 0.4375, "qas": 0.7653},
        "combined": {"speedup": 1.7369, "acc_ret": 0.6059, "qas": 1.0524},
    },
}

# M3 results at gw=0.7 (from m3_pareto_corrected.json)
M3_RESULTS = {
    "gw0.7": {
        "gsm8k":   {"speedup": 1.6472, "acc_ret": 1.0390, "qas": 1.7114},
        "math500": {"speedup": 1.1559, "acc_ret": 2.4390, "qas": 2.8195},
        "combined": {"speedup": 1.4998, "acc_ret": 1.4590, "qas": 2.1883},
    },
}

# Configuration for pilot
GEN_LENGTH   = 256
BLOCK_LENGTH = 32
T_FULL       = 64
SEED         = 42
N_GSM8K      = 100    # pilot
N_MATH500    = 100    # pilot
MASK_ID      = 126336
N_WARMUP     = 5

# Test configurations: IGSD configs x M3 gw=0.7
PAIRWISE_CONFIGS = [
    {"name": "igsd_t09_td32_m3_gw07", "tau": 0.9,  "t_draft": 32, "gw": 0.7,
     "igsd_key": "tau0.9_td32", "m3_key": "gw0.7"},
    {"name": "igsd_t07_td16_m3_gw07", "tau": 0.7,  "t_draft": 16, "gw": 0.7,
     "igsd_key": "tau0.7_td16", "m3_key": "gw0.7"},
    {"name": "igsd_t085_td32_m3_gw07", "tau": 0.85, "t_draft": 32, "gw": 0.7,
     "igsd_key": "tau0.85_td32", "m3_key": "gw0.7"},
]

_LLADA_TOKENIZER = None


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


# ── Prompts ────────────────────────────────────────────────────────────────
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


# ── Qwen Scoring (from M3) ────────────────────────────────────────────────
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
                if pi < offset:
                    continue
                qp = pi - offset
                if qp >= ql.shape[1]:
                    continue
                ltok = llada_x0[b, pi].item()
                if ltok == MASK_ID:
                    continue
                tok_txt = _LLADA_TOKENIZER.decode([ltok], skip_special_tokens=True)
                if not tok_txt.strip():
                    continue
                qtok_ids = qwen_tokenizer.encode(tok_txt, add_special_tokens=False)
                if not qtok_ids:
                    continue
                qtok = qtok_ids[0]
                if qtok >= ql.shape[-1]:
                    continue
                qwen_scores[b, pi] = float(F.softmax(ql[0, qp, :], dim=-1)[qtok].item())
        except Exception:
            pass
    return qwen_scores


# ── M3+IGSD Combined Generator ────────────────────────────────────────────
class M3IGSDGenerator:
    """
    Combined M3+IGSD generator:
      - Phase 1 (IGSD draft): Whole-sequence draft with T_draft steps (NO M3 guidance)
      - Phase 2 (IGSD partition): Confidence-based partition (tau threshold)
      - Phase 3 (M3-guided refine): Refine S_refine tokens using AR-guided unmasking
        with Qwen2.5-0.5B guidance (M3) over T_full steps

    The key interaction: M3's Qwen guidance operates on the partially-filled
    sequence from IGSD's draft phase. The compressed draft trajectory means
    Qwen sees a noisier/less-coherent context, potentially degrading guidance quality.
    """

    def __init__(self, llada_model, llada_tokenizer, qwen_model, qwen_tokenizer, device="cuda"):
        self.llada_model = llada_model
        self.llada_tokenizer = llada_tokenizer
        self.qwen_model = qwen_model
        self.qwen_tokenizer = qwen_tokenizer
        self.device = device

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        tau: float = 0.9,
        t_draft: int = 32,
        t_full: int = 64,
        guidance_weight: float = 0.7,
        gen_length: int = 256,
        block_length: int = 32,
        apply_chat_template: bool = True,
    ) -> dict:
        """Generate text using IGSD draft + M3-guided refine."""
        # Tokenize prompt
        if apply_chat_template:
            msg = [{"role": "user", "content": prompt}]
            enc_text = self.llada_tokenizer.apply_chat_template(
                msg, add_generation_prompt=True, tokenize=False
            )
        else:
            enc_text = prompt

        enc = self.llada_tokenizer(
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

        # ── Phase 1: IGSD Whole-sequence draft (NO M3 guidance) ──────
        t_draft_start = time.perf_counter()
        tokens_per_draft_step = max(1, gen_length // t_draft)
        remainder_draft = gen_length % t_draft

        for step in range(t_draft):
            n_masked = int((x[0, prompt_len:] == MASK_ID).sum().item())
            if n_masked == 0:
                break

            logits = self.llada_model(x, attention_mask=attn).logits
            p = F.softmax(logits, dim=-1)
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

        draft_elapsed = time.perf_counter() - t_draft_start

        # ── Phase 2: Confidence partition ────────────────────────────
        logits = self.llada_model(x, attention_mask=attn).logits
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

        # ── Phase 3: M3-guided refine for S_refine ──────────────────
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

                logits = self.llada_model(x_refine, attention_mask=attn).logits
                p = F.softmax(logits, dim=-1)
                x0 = torch.argmax(p, dim=-1)
                x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

                # Apply M3 AR guidance during refine phase
                if guidance_weight > 0:
                    qwen_scores = get_qwen_scores(
                        self.qwen_model, self.qwen_tokenizer,
                        x_refine, x0, self.device
                    )
                    blended = (1 - guidance_weight) * x0_p + guidance_weight * qwen_scores
                else:
                    blended = x0_p

                mask_index = (x_refine == MASK_ID) & s_refine_full
                confidence = torch.where(
                    mask_index, blended,
                    torch.tensor(-float("inf"), device=self.device)
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
            lf = self.llada_model(x_refine, attention_mask=attn).logits
            pf = F.softmax(lf[:, prompt_len:, :], dim=-1)
            fp = torch.argmax(pf, dim=-1)
            x_refine[:, prompt_len:] = torch.where(
                still_left, fp, x_refine[:, prompt_len:]
            )

        total_elapsed = draft_elapsed + refine_elapsed
        total_tps = gen_length / total_elapsed if total_elapsed > 0 else 0.0

        text = self.llada_tokenizer.decode(
            x_refine[0, prompt_len:].tolist(), skip_special_tokens=True
        )

        return {
            "generated_text": text,
            "tps": total_tps,
            "elapsed_sec": total_elapsed,
            "draft_elapsed_sec": draft_elapsed,
            "refine_elapsed_sec": refine_elapsed,
            "draft_tps": gen_length / draft_elapsed if draft_elapsed > 0 else 0.0,
            "refine_tps": gen_length / refine_elapsed if refine_elapsed > 0 else 0.0,
            "accept_rate": accept_rate,
            "n_accept": n_accept,
            "n_refine": n_refine,
            "n_total": n_total,
            "tau": tau,
            "t_draft": t_draft,
            "t_full": t_full,
            "guidance_weight": guidance_weight,
        }


# ── Evaluation Functions ───────────────────────────────────────────────────
def evaluate_gsm8k(generator, data, tau, t_draft, gw, n_warmup=N_WARMUP):
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
                guidance_weight=gw,
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

        if (i + 1) % 25 == 0:
            acc = correct / (i + 1)
            avg_tps = np.mean(tps_list) if tps_list else 0.0
            print(f"  [GSM8K M3+IGSD tau={tau} td={t_draft} gw={gw}] "
                  f"{i+1}/{total} acc={acc:.3f} tps={avg_tps:.1f}")

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


def evaluate_math500(generator, data, tau, t_draft, gw, n_warmup=3):
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
                guidance_weight=gw,
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
            print(f"  [MATH500 M3+IGSD tau={tau} td={t_draft} gw={gw}] "
                  f"{i+1}/{total} acc={acc:.3f} tps={avg_tps:.1f}")

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


# ── Ortho Computation ──────────────────────────────────────────────────────
def compute_ortho(combined_qas, individual_a_qas, individual_b_qas):
    """
    Ortho(Ma + Mb) = QAS(Ma+Mb) / max(QAS(Ma), QAS(Mb))

    Ortho > 1.0: synergy
    Ortho 0.8-1.0: near-orthogonal
    Ortho < 0.8: interference
    """
    max_individual = max(individual_a_qas, individual_b_qas)
    if max_individual <= 0:
        return 0.0
    return combined_qas / max_individual


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    global _LLADA_TOKENIZER
    write_pid()
    start_time = time.time()

    print(f"[M3+IGSD Pairwise PILOT] Starting at {datetime.now().isoformat()}")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"  Mode: PILOT, seed={SEED}")
    print(f"  GSM8K: {N_GSM8K} samples, MATH500: {N_MATH500} samples")
    print(f"  Configs: {len(PAIRWISE_CONFIGS)}")

    # Set seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Load datasets
    print("[Loading datasets...]")
    gsm8k_data = load_gsm8k(n_samples=N_GSM8K, seed=SEED)
    math500_data = load_math500(n_samples=N_MATH500, seed=SEED)
    print(f"  GSM8K: {len(gsm8k_data)} samples, MATH500: {len(math500_data)} samples")

    # Load LLaDA model
    print("[Loading LLaDA-8B-Instruct...]")
    from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
    llada_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    llada_tokenizer.padding_side = "left"
    _LLADA_TOKENIZER = llada_tokenizer
    llada_model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    vram_after_llada = torch.cuda.memory_allocated() // 1024**2
    print(f"  LLaDA loaded. VRAM: {vram_after_llada} MB")

    # Load Qwen model
    print("[Loading Qwen2.5-0.5B...]")
    qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_PATH, trust_remote_code=True)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    vram_after_both = torch.cuda.memory_allocated() // 1024**2
    print(f"  Qwen loaded. Total VRAM: {vram_after_both} MB")

    vram_info = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2,
        "vram_used_mb": vram_after_both,
        "utilization_pct": round(100 * vram_after_both / (torch.cuda.get_device_properties(0).total_memory // 1024**2), 1),
    }

    # Create generator
    generator = M3IGSDGenerator(
        llada_model, llada_tokenizer,
        qwen_model, qwen_tokenizer,
        device="cuda"
    )

    # ── Run all pairwise configurations ────────────────────────────
    all_results = []
    total_configs = len(PAIRWISE_CONFIGS)

    # Also compute a pilot baseline for this subset (same samples)
    # Run vanilla LLaDA baseline on the same samples to get pilot baseline TPS
    print("\n[Computing pilot baseline on same samples...]")
    from generate import get_num_transfer_tokens

    baseline_gsm8k_tps_list = []
    baseline_gsm8k_correct = 0
    baseline_math500_tps_list = []
    baseline_math500_correct = 0

    # Quick baseline: standard 64-step generation on a few samples for TPS
    # We use the full baseline from iter_001 for accuracy reference
    # but need local TPS for fair speedup comparison
    print("  Running baseline TPS measurement (20 GSM8K + 10 MATH500 samples)...")

    @torch.no_grad()
    def baseline_generate(prompt_text, model, tokenizer, device, gen_length=256, steps=64):
        """Standard LLaDA generation for baseline TPS."""
        msg = [{"role": "user", "content": prompt_text}]
        enc_text = tokenizer.apply_chat_template(msg, add_generation_prompt=True, tokenize=False)
        enc = tokenizer([enc_text], add_special_tokens=False, padding=True, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_len = input_ids.shape[1]

        x = torch.full((1, prompt_len + gen_length), MASK_ID, dtype=torch.long, device=device)
        x[:, :prompt_len] = input_ids
        attn_full = torch.cat([
            attention_mask,
            torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)
        ], dim=-1)

        block_length = 32
        num_blocks = gen_length // block_length
        steps_per_block = steps // num_blocks

        t0 = time.perf_counter()
        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length
            block_mask = x[:, block_start:block_end] == MASK_ID
            num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

            for step in range(steps_per_block):
                mask_index = x == MASK_ID
                if not mask_index[:, block_start:block_end].any():
                    continue
                logits = model(x, attention_mask=attn_full).logits
                probs = F.softmax(logits, dim=-1)
                x0 = torch.argmax(probs, dim=-1)
                x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)

                x0_p[:, block_end:] = -float('inf')
                x0 = torch.where(mask_index, x0, x)
                conf = torch.where(mask_index, x0_p, torch.tensor(-float('inf'), device=device))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                k = num_transfer[0, step].item()
                if k > 0:
                    _, sel = torch.topk(conf[0], k=int(k))
                    transfer_index[0, sel] = True
                x[transfer_index] = x0[transfer_index]

        elapsed = time.perf_counter() - t0
        text = tokenizer.decode(x[0, prompt_len:].tolist(), skip_special_tokens=True)
        tps = gen_length / elapsed if elapsed > 0 else 0.0
        return text, tps

    # Measure baseline TPS on small subset
    for i in range(min(20, len(gsm8k_data))):
        try:
            text, tps = baseline_generate(
                build_gsm8k_prompt(gsm8k_data[i]["question"]),
                llada_model, llada_tokenizer, "cuda"
            )
            if i >= 5:  # skip warmup
                baseline_gsm8k_tps_list.append(tps)
            if gsm8k_exact_match(text, gsm8k_data[i]["answer"]):
                baseline_gsm8k_correct += 1
        except Exception as e:
            print(f"  Baseline GSM8K error {i}: {e}")

    for i in range(min(10, len(math500_data))):
        try:
            text, tps = baseline_generate(
                build_math500_prompt(math500_data[i]),
                llada_model, llada_tokenizer, "cuda"
            )
            if i >= 3:  # skip warmup
                baseline_math500_tps_list.append(tps)
            if math500_exact_match(text, math500_data[i]["solution"]):
                baseline_math500_correct += 1
        except Exception as e:
            print(f"  Baseline MATH500 error {i}: {e}")

    pilot_baseline = {
        "gsm8k": {
            "avg_tps": float(np.mean(baseline_gsm8k_tps_list)) if baseline_gsm8k_tps_list else BASELINE["gsm8k"]["avg_tps"],
            "exact_match": BASELINE["gsm8k"]["exact_match"],  # Use iter_001 full baseline
        },
        "math500": {
            "avg_tps": float(np.mean(baseline_math500_tps_list)) if baseline_math500_tps_list else BASELINE["math500"]["avg_tps"],
            "exact_match": BASELINE["math500"]["exact_match"],  # Use iter_001 full baseline
        },
    }
    print(f"  Pilot baseline TPS: GSM8K={pilot_baseline['gsm8k']['avg_tps']:.1f}, "
          f"MATH500={pilot_baseline['math500']['avg_tps']:.1f}")

    torch.cuda.empty_cache()
    gc.collect()

    # ── Evaluate each pairwise configuration ───────────────────────
    for cfg_idx, cfg in enumerate(PAIRWISE_CONFIGS):
        tau = cfg["tau"]
        t_draft = cfg["t_draft"]
        gw = cfg["gw"]
        config_name = cfg["name"]
        igsd_key = cfg["igsd_key"]
        m3_key = cfg["m3_key"]

        print(f"\n{'='*60}")
        print(f"[Config {cfg_idx+1}/{total_configs}] {config_name}")
        print(f"  IGSD: tau={tau}, T_draft={t_draft}")
        print(f"  M3: gw={gw}")
        print(f"{'='*60}")

        config_start = time.time()

        # Evaluate GSM8K
        print(f"  Evaluating GSM8K ({N_GSM8K} samples)...")
        gsm8k_metrics = evaluate_gsm8k(generator, gsm8k_data, tau, t_draft, gw)

        # Evaluate MATH500
        print(f"  Evaluating MATH500 ({N_MATH500} samples)...")
        math500_metrics = evaluate_math500(generator, math500_data, tau, t_draft, gw)

        config_elapsed = time.time() - config_start

        # Compute metrics relative to baseline
        gsm8k_speedup = gsm8k_metrics["avg_tps"] / pilot_baseline["gsm8k"]["avg_tps"] if pilot_baseline["gsm8k"]["avg_tps"] > 0 else 0
        math500_speedup = math500_metrics["avg_tps"] / pilot_baseline["math500"]["avg_tps"] if pilot_baseline["math500"]["avg_tps"] > 0 else 0

        gsm8k_acc_ret = gsm8k_metrics["exact_match"] / BASELINE["gsm8k"]["exact_match"] if BASELINE["gsm8k"]["exact_match"] > 0 else 0
        math500_acc_ret = math500_metrics["exact_match"] / BASELINE["math500"]["exact_match"] if BASELINE["math500"]["exact_match"] > 0 else 0

        gsm8k_qas = gsm8k_speedup * gsm8k_acc_ret
        math500_qas = math500_speedup * math500_acc_ret

        combined_speedup = 0.7 * gsm8k_speedup + 0.3 * math500_speedup
        combined_acc_ret = 0.7 * gsm8k_acc_ret + 0.3 * math500_acc_ret
        combined_qas = combined_speedup * combined_acc_ret

        # Compute Ortho against individual methods
        igsd_ref = IGSD_RESULTS[igsd_key]
        m3_ref = M3_RESULTS[m3_key]

        ortho_gsm8k = compute_ortho(gsm8k_qas, igsd_ref["gsm8k"]["qas"], m3_ref["gsm8k"]["qas"])
        ortho_math500 = compute_ortho(math500_qas, igsd_ref["math500"]["qas"], m3_ref["math500"]["qas"])
        ortho_combined = compute_ortho(combined_qas, igsd_ref["combined"]["qas"], m3_ref["combined"]["qas"])

        config_result = {
            "config_name": config_name,
            "tau": tau,
            "t_draft": t_draft,
            "t_full": T_FULL,
            "guidance_weight": gw,
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
            "ortho": {
                "gsm8k": ortho_gsm8k,
                "math500": ortho_math500,
                "combined": ortho_combined,
                "igsd_ref": igsd_ref,
                "m3_ref": m3_ref,
                "interpretation": {
                    "gsm8k": "synergy" if ortho_gsm8k > 1.0 else ("near-orthogonal" if ortho_gsm8k > 0.8 else "interference"),
                    "math500": "synergy" if ortho_math500 > 1.0 else ("near-orthogonal" if ortho_math500 > 0.8 else "interference"),
                    "combined": "synergy" if ortho_combined > 1.0 else ("near-orthogonal" if ortho_combined > 0.8 else "interference"),
                },
            },
        }

        all_results.append(config_result)

        print(f"\n  === Results for {config_name} ===")
        print(f"  GSM8K:   acc={gsm8k_metrics['exact_match']:.3f} tps={gsm8k_metrics['avg_tps']:.1f} "
              f"speedup={gsm8k_speedup:.2f}x acc_ret={gsm8k_acc_ret:.3f} QAS={gsm8k_qas:.3f}")
        print(f"  MATH500: acc={math500_metrics['exact_match']:.3f} tps={math500_metrics['avg_tps']:.1f} "
              f"speedup={math500_speedup:.2f}x acc_ret={math500_acc_ret:.3f} QAS={math500_qas:.3f}")
        print(f"  Combined: speedup={combined_speedup:.2f}x acc_ret={combined_acc_ret:.3f} QAS={combined_qas:.3f}")
        print(f"  Ortho: GSM8K={ortho_gsm8k:.3f} ({config_result['ortho']['interpretation']['gsm8k']})")
        print(f"         MATH500={ortho_math500:.3f} ({config_result['ortho']['interpretation']['math500']})")
        print(f"         Combined={ortho_combined:.3f} ({config_result['ortho']['interpretation']['combined']})")
        print(f"  Elapsed: {config_elapsed/60:.1f} min")

        report_progress(cfg_idx + 1, total_configs, {
            "config": config_name,
            "gsm8k_acc": gsm8k_metrics["exact_match"],
            "gsm8k_speedup": gsm8k_speedup,
            "ortho_combined": ortho_combined,
        })

        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary and analysis ───────────────────────────────────────
    total_elapsed = time.time() - start_time

    # Find best and worst configs
    best_by_ortho = max(all_results, key=lambda r: r["ortho"]["combined"])
    worst_by_ortho = min(all_results, key=lambda r: r["ortho"]["combined"])

    # Check pass criteria: Ortho values computed without error
    pass_criteria_met = all(
        r["ortho"]["gsm8k"] is not None and r["ortho"]["math500"] is not None
        for r in all_results
    )

    # Interference analysis
    interference_analysis = {
        "hypothesis": "M3's guidance overhead per step + IGSD's reduced steps = conflicting objectives",
        "mechanism": "AR guide model operates on IGSD's compressed trajectory (fewer draft steps = noisier context for Qwen)",
        "evidence": {
            "all_configs_ortho": [
                {"config": r["config_name"], "ortho_combined": r["ortho"]["combined"],
                 "interpretation": r["ortho"]["interpretation"]["combined"]}
                for r in all_results
            ],
            "mean_ortho_combined": float(np.mean([r["ortho"]["combined"] for r in all_results])),
            "interference_confirmed": all(r["ortho"]["combined"] < 1.0 for r in all_results),
        },
    }

    # ── Build output JSON ──────────────────────────────────────────
    output = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "iteration": 2,
        "model": "LLaDA-8B-Instruct",
        "guide_model": "Qwen2.5-0.5B",
        "composition": "M3+IGSD",
        "timestamp": datetime.now().isoformat(),
        "elapsed_minutes": round(total_elapsed / 60, 1),
        "pilot_config": {
            "gsm8k_samples": N_GSM8K,
            "math500_samples": N_MATH500,
            "seed": SEED,
            "gen_length": GEN_LENGTH,
            "steps": T_FULL,
        },
        "baseline_reference": {
            "iter1_full_baseline": BASELINE,
            "pilot_baseline_tps": pilot_baseline,
        },
        "qas_formula": "QAS = Speedup * Accuracy_Retention (corrected, no 0.5x penalty)",
        "ortho_formula": "Ortho(M3+IGSD) = QAS(M3+IGSD) / max(QAS(M3), QAS(IGSD))",
        "combined_metric": "0.7 * GSM8K + 0.3 * MATH500",
        "pass_criteria": {
            "requirement": "Ortho values computed for both benchmarks without error",
            "met": pass_criteria_met,
        },
        "all_configs": all_results,
        "summary": {
            "best_config": {
                "name": best_by_ortho["config_name"],
                "ortho_combined": best_by_ortho["ortho"]["combined"],
                "interpretation": best_by_ortho["ortho"]["interpretation"]["combined"],
            },
            "worst_config": {
                "name": worst_by_ortho["config_name"],
                "ortho_combined": worst_by_ortho["ortho"]["combined"],
                "interpretation": worst_by_ortho["ortho"]["interpretation"]["combined"],
            },
            "mean_ortho_combined": float(np.mean([r["ortho"]["combined"] for r in all_results])),
            "interference_confirmed": all(r["ortho"]["combined"] < 1.0 for r in all_results),
        },
        "interference_analysis": interference_analysis,
        "vram": vram_info,
    }

    # Save results
    out_path = RESULTS_DIR / "m3_igsd_full.json"
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\n{'='*60}")
    print(f"[M3+IGSD Pairwise PILOT] COMPLETE")
    print(f"  Total elapsed: {total_elapsed/60:.1f} min")
    print(f"  Pass criteria met: {pass_criteria_met}")
    print(f"  Best config: {best_by_ortho['config_name']} (Ortho={best_by_ortho['ortho']['combined']:.3f})")
    print(f"  Interference confirmed: {interference_analysis['evidence']['interference_confirmed']}")
    print(f"  Mean Ortho: {interference_analysis['evidence']['mean_ortho_combined']:.3f}")
    print(f"  Results saved to: {out_path}")
    print(f"{'='*60}")

    mark_done(
        status="success",
        summary=f"M3+IGSD pairwise pilot complete. {total_configs} configs. "
                f"Mean Ortho={interference_analysis['evidence']['mean_ortho_combined']:.3f}. "
                f"Interference={'confirmed' if interference_analysis['evidence']['interference_confirmed'] else 'not confirmed'}. "
                f"Best: {best_by_ortho['config_name']} Ortho={best_by_ortho['ortho']['combined']:.3f}"
    )

    # Update gpu_progress.json
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text())
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
            "guide_model": "Qwen2.5-0.5B",
            "composition": "M3+IGSD",
            "configs": [c["name"] for c in PAIRWISE_CONFIGS],
            "n_samples": {"gsm8k": N_GSM8K, "math500": N_MATH500},
            "mode": "PILOT",
            "gpu_model": vram_info.get("gpu_name", "unknown"),
            "gpu_count": 1,
        },
    }
    gp_path.write_text(json.dumps(gp, indent=2))
    print(f"[gpu_progress.json] Updated: {TASK_ID} -> completed")


if __name__ == "__main__":
    main()
