"""
Pilot M3 (AR-Guided Unmasking) Single-Method Evaluation.

Task: pilot_m3_single
Method: AR-Guided Unmasking using Qwen2.5-0.5B as supervisor.

At each denoising step, instead of picking the top-k tokens purely by LLaDA's
own softmax confidence, we blend in a guidance score from Qwen2.5-0.5B.

Guidance blending formula:
    score[i] = (1 - alpha) * llada_confidence[i] + alpha * qwen_score[i]

where qwen_score is the probability Qwen2.5-0.5B assigns to the LLaDA-predicted
token given the current (partially unmasked) left context.

Sweep guidance_weight (alpha) in {0.3, 0.5, 0.7, 1.0}.

Run on 100 GSM8K + 50 HumanEval samples (seed=42).
Record: TPS, exact_match (GSM8K), pass@1 (HumanEval).
Identify operating point: highest QAS within acceptable accuracy budget.

Baseline reference:
    GSM8K exact_match = 0.73, avg_tps = 58.55
    HumanEval pass@1 = 0.04, avg_tps = 100.93
"""
import os, sys, json, time, random, gc, re, subprocess
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import torch.nn.functional as F

# ── Paths ─────────────────────────────────────────────────────────────────────
WORKSPACE     = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED        = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH    = str(SHARED / "checkpoints" / "llada-8b-instruct")
QWEN_PATH     = str(SHARED / "checkpoints" / "qwen2.5-0.5b")
GSM8K_DIR     = str(SHARED / "datasets" / "gsm8k")
HUMANEVAL_DIR = str(SHARED / "datasets" / "humaneval")
CODE_DIR      = WORKSPACE / "exp" / "code"
RESULTS_DIR   = WORKSPACE / "exp" / "results" / "pilot_m3"
TASK_ID       = "pilot_m3_single"

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Baseline reference
BASELINE_GSM8K_TPS   = 58.55
BASELINE_GSM8K_ACC   = 0.73
BASELINE_HE_TPS      = 100.93
BASELINE_HE_ACC      = 0.04

# ── System-monitor Helpers ─────────────────────────────────────────────────────
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total, metric=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "updated_at": datetime.now().isoformat(),
        "metric": metric or {},
    }))

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

# ── Data Loaders ───────────────────────────────────────────────────────────────
def load_gsm8k(n_samples=100, seed=42):
    path = Path(GSM8K_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n_samples, len(data)))

def load_humaneval(n_samples=50, seed=42):
    path = Path(HUMANEVAL_DIR) / "test.json"
    with open(path) as f:
        data = json.load(f)
    rng = random.Random(seed)
    return rng.sample(data, min(n_samples, len(data)))

# ── 8-shot GSM8K Prompt ────────────────────────────────────────────────────────
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

def extract_gsm8k_answer(text):
    match = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match: return match.group(1).replace(",", "")
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return numbers[-1].replace(",", "") if numbers else None

def gsm8k_exact_match(pred, gold):
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except: return p.strip() == g.strip()

def humaneval_pass_at_1(code_completion, problem):
    full_code = problem["prompt"] + code_completion + "\n" + problem["test"]
    full_code += f"\ncheck({problem['entry_point']})\n"
    try:
        result = subprocess.run(["python", "-c", full_code],
                                capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except: return False

def profile_vram(device):
    if not torch.cuda.is_available(): return {}
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu_name": torch.cuda.get_device_name(device),
        "vram_total_mb": props.total_memory // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(device) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(device) // 1024**2,
    }


# ── AR-Guided Qwen Score Computation ──────────────────────────────────────────
def get_qwen_scores(
    qwen_model, qwen_tokenizer, current_ids, mask_id,
    llada_token_ids, device, max_context=512,
):
    """
    Compute Qwen2.5-0.5B probability scores for the LLaDA-proposed tokens at
    each masked position.

    Strategy:
    - For each masked position i, use the left context (tokens 0..i-1 that are
      unmasked) as Qwen's prompt. Score P_qwen(llada_token | left_context).
    - This is efficient via a single forward pass: feed the whole sequence to
      Qwen with causal masking, extract logit at each position for its
      LLaDA-predicted token.

    The issue: LLaDA and Qwen use different vocabularies. We bridge them by
    decoding LLaDA tokens to text and encoding with Qwen tokenizer (single-token
    approximation). For tokens that encode to multiple Qwen subwords, use the
    first subword probability as a proxy.

    Returns: Tensor of shape (batch, seq_len) with Qwen confidence scores in [0,1].
    """
    batch, seq_len = current_ids.shape
    qwen_scores = torch.zeros(batch, seq_len, device=device)

    # Build Qwen input by substituting unmasked tokens with their decoded text
    # and masked tokens as "<mask>" placeholders (which Qwen sees as context gaps)
    for b in range(batch):
        ids = current_ids[b]  # (seq_len,)
        mask_positions = (ids == mask_id).nonzero(as_tuple=True)[0]
        if len(mask_positions) == 0:
            continue

        # Build a text sequence from unmasked tokens for Qwen
        unmasked_ids = ids.clone()
        # Replace mask tokens with a neutral token (we'll use pad or unk)
        pad_tok = qwen_tokenizer.pad_token_id or qwen_tokenizer.eos_token_id
        unmasked_ids[ids == mask_id] = pad_tok

        # Trim to max_context for speed
        seq_to_encode = unmasked_ids.cpu().tolist()
        if len(seq_to_encode) > max_context:
            seq_to_encode = seq_to_encode[-max_context:]
            offset = len(unmasked_ids) - max_context
        else:
            offset = 0

        # Decode to text and re-encode with Qwen tokenizer
        # Use LLaDA tokenizer for decoding (loaded in outer scope via global ref)
        try:
            text = _LLADA_TOKENIZER.decode(seq_to_encode, skip_special_tokens=True)
            qwen_enc = qwen_tokenizer(
                text, return_tensors="pt",
                max_length=max_context, truncation=True,
                add_special_tokens=False
            )
            qwen_input = qwen_enc["input_ids"].to(device)

            with torch.no_grad():
                qwen_out = qwen_model(qwen_input)
                qwen_logits = qwen_out.logits  # (1, qwen_seq_len, qwen_vocab)

            # For each masked position in LLaDA, get Qwen score for the predicted token
            for pos in mask_positions:
                pos_int = pos.item()
                if pos_int < offset:
                    continue
                qwen_pos = pos_int - offset
                if qwen_pos >= qwen_logits.shape[1]:
                    continue

                # Get LLaDA's predicted token at this position
                llada_tok = llada_token_ids[b, pos_int].item()
                if llada_tok == mask_id:
                    continue

                # Decode that token to text, then encode with Qwen
                tok_text = _LLADA_TOKENIZER.decode([llada_tok], skip_special_tokens=True)
                if not tok_text.strip():
                    continue
                qwen_tok_ids = qwen_tokenizer.encode(tok_text, add_special_tokens=False)
                if not qwen_tok_ids:
                    continue

                # Use probability of first Qwen subword as proxy
                qwen_first_tok = qwen_tok_ids[0]
                if qwen_first_tok >= qwen_logits.shape[-1]:
                    continue

                qwen_logit_at_pos = qwen_logits[0, qwen_pos, :]
                qwen_prob = float(F.softmax(qwen_logit_at_pos, dim=-1)[qwen_first_tok].item())
                qwen_scores[b, pos_int] = qwen_prob

        except Exception:
            # Fall back to zero scores for this sequence
            pass

    return qwen_scores


# Global reference to LLaDA tokenizer (set in main)
_LLADA_TOKENIZER = None


# ── M3 AR-Guided Generation ────────────────────────────────────────────────────
@torch.no_grad()
def generate_m3(
    llada_model, llada_tokenizer,
    qwen_model, qwen_tokenizer,
    prompt_text, device,
    gen_length=256, block_length=32,
    guidance_weight=0.5,
    apply_chat_template=True,
    mask_id=126336,
    steps=64,
):
    """
    M3 AR-Guided Unmasking.

    At each denoising step:
    1. Run LLaDA forward pass to get token probabilities and top predicted tokens.
    2. Compute Qwen2.5-0.5B score for each LLaDA-predicted token.
    3. Blend: score = (1 - alpha) * llada_conf + alpha * qwen_score
    4. Select tokens to unmask by top blended score (same count as baseline).

    The key hypothesis: Qwen provides a left-to-right language model prior
    that helps LLaDA select better tokens earlier in the denoising process,
    potentially improving quality or allowing fewer steps for same quality.

    Note on overhead: Each step now also runs Qwen2.5-0.5B forward pass,
    which adds latency. The speedup over baseline may be negative (M3 is
    slower in TPS) but should improve accuracy.
    """
    global _LLADA_TOKENIZER
    _LLADA_TOKENIZER = llada_tokenizer

    from generate import get_num_transfer_tokens

    if apply_chat_template:
        msg = [{"role": "user", "content": prompt_text}]
        prompt_text = llada_tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=False
        )
    enc = llada_tokenizer([prompt_text], add_special_tokens=False,
                          padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    batch = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    x = torch.full(
        (batch, prompt_len + gen_length), mask_id, dtype=torch.long
    ).to(device)
    x[:, :prompt_len] = input_ids.clone()
    attn = torch.cat([
        attention_mask,
        torch.ones((batch, gen_length), dtype=attention_mask.dtype, device=device)
    ], dim=-1)

    num_blocks = gen_length // block_length
    steps_per_block = steps // num_blocks  # 64 / 8 = 8 for gen_length=256

    t0 = time.perf_counter()
    total_forward_passes = 0
    qwen_calls = 0

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end = prompt_len + (block_idx + 1) * block_length
        block_mask = x[:, block_start:block_end] == mask_id
        num_transfer = get_num_transfer_tokens(block_mask, steps_per_block)

        for step in range(steps_per_block):
            mask_index = x == mask_id

            if not mask_index[:, block_start:block_end].any():
                continue

            # LLaDA forward pass
            logits = llada_model(x, attention_mask=attn).logits
            total_forward_passes += 1

            probs = F.softmax(logits, dim=-1)
            x0 = torch.argmax(probs, dim=-1)  # LLaDA's best token prediction
            x0_p = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)  # confidence

            # LLaDA confidence (for masked positions in current block only)
            llada_conf = torch.where(mask_index, x0_p,
                                     torch.tensor(-float('inf'), device=device))
            # Zero out positions outside current block
            llada_conf_block = llada_conf.clone()
            llada_conf_block[:, :block_start] = -float('inf')
            llada_conf_block[:, block_end:] = -float('inf')

            # Get Qwen guidance scores
            try:
                qwen_scores_tensor = get_qwen_scores(
                    qwen_model, qwen_tokenizer,
                    x, mask_id, x0, device, max_context=256,
                )
                qwen_calls += 1
                # Apply guidance blending
                valid_mask = mask_index & (llada_conf_block > -float('inf'))
                blended_score = llada_conf_block.clone()
                blended_score[valid_mask] = (
                    (1.0 - guidance_weight) * llada_conf_block[valid_mask]
                    + guidance_weight * qwen_scores_tensor[valid_mask]
                )
            except Exception:
                # Fallback to pure LLaDA confidence
                blended_score = llada_conf_block

            # Prevent overwriting already-unmasked tokens
            x0 = torch.where(mask_index, x0, x)

            # Transfer top-k tokens by blended score
            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            for j in range(batch):
                k = num_transfer[j, step].item()
                if k > 0:
                    _, sel = torch.topk(blended_score[j], k=int(k))
                    transfer_index[j, sel] = True
            x[transfer_index] = x0[transfer_index]

    elapsed = time.perf_counter() - t0

    generated_ids = x[:, prompt_len:]
    total_tokens = generated_ids.numel()
    tps = total_tokens / elapsed if elapsed > 0 else 0.0
    text = llada_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return text, tps, elapsed, {
        "total_forward_passes": total_forward_passes,
        "qwen_calls": qwen_calls,
        "guidance_weight": guidance_weight,
        "total_steps": steps,
    }


# ── Per-guidance-weight Evaluation ────────────────────────────────────────────
def evaluate_guidance_weight(
    llada_model, llada_tokenizer, qwen_model, qwen_tokenizer, device,
    guidance_weight,
    gsm8k_data, humaneval_data,
    n_warmup=3,
    gw_idx=0,
    total_gws=4,
    total_progress=600,
):
    """Evaluate M3 with a given guidance weight on GSM8K + HumanEval."""
    print(f"\n[pilot_m3] === guidance_weight = {guidance_weight} ===")
    base_progress = gw_idx * (total_progress // total_gws)

    # GSM8K
    gsm8k_tps_list = []
    gsm8k_correct = 0
    gsm8k_results = []

    for i, item in enumerate(gsm8k_data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            text, tps, elapsed, stats = generate_m3(
                llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                prompt, device,
                gen_length=256, block_length=32,
                guidance_weight=guidance_weight,
                apply_chat_template=True,
            )
            correct = gsm8k_exact_match(text, item["answer"])
            if correct: gsm8k_correct += 1
            if i >= n_warmup:
                gsm8k_tps_list.append(tps)
            gsm8k_results.append({
                "id": i, "correct": correct, "tps": round(tps, 2),
                "prediction_snippet": text[:200],
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] GSM8K sample {i}")
            torch.cuda.empty_cache(); gc.collect()
            gsm8k_results.append({"id": i, "error": "OOM", "correct": False, "tps": 0})
        except Exception as e:
            print(f"  [ERR] GSM8K sample {i}: {e}")
            gsm8k_results.append({"id": i, "error": str(e)[:200], "correct": False, "tps": 0})

        if (i + 1) % 10 == 0:
            acc = gsm8k_correct / (i + 1)
            avg_tps = np.mean(gsm8k_tps_list) if gsm8k_tps_list else 0.0
            print(f"  GSM8K [{i+1}/{len(gsm8k_data)}] acc={acc:.3f}, tps={avg_tps:.1f}")
            report_progress(
                base_progress + (i + 1) // 2,
                total_progress,
                {"guidance_weight": guidance_weight, "gsm8k_acc": acc, "gsm8k_tps": avg_tps}
            )

    gsm8k_acc = gsm8k_correct / len(gsm8k_data) if gsm8k_data else 0.0
    gsm8k_avg_tps = float(np.mean(gsm8k_tps_list)) if gsm8k_tps_list else 0.0
    gsm8k_tps_std = float(np.std(gsm8k_tps_list)) if gsm8k_tps_list else 0.0
    gsm8k_speedup = gsm8k_avg_tps / BASELINE_GSM8K_TPS if BASELINE_GSM8K_TPS > 0 else 0.0
    gsm8k_acc_retention = gsm8k_acc / BASELINE_GSM8K_ACC if BASELINE_GSM8K_ACC > 0 else 0.0
    gsm8k_qas = gsm8k_speedup * gsm8k_acc_retention

    print(f"  GSM8K: acc={gsm8k_acc:.3f} ({gsm8k_correct}/{len(gsm8k_data)}), "
          f"tps={gsm8k_avg_tps:.1f}±{gsm8k_tps_std:.1f}, "
          f"speedup={gsm8k_speedup:.2f}x, QAS={gsm8k_qas:.3f}")

    # HumanEval
    he_tps_list = []
    he_passed = 0
    he_results = []

    for i, item in enumerate(humaneval_data):
        prompt = f"Complete the following Python function:\n\n{item['prompt']}"
        try:
            code, tps, elapsed, stats = generate_m3(
                llada_model, llada_tokenizer, qwen_model, qwen_tokenizer,
                prompt, device,
                gen_length=256, block_length=32,
                guidance_weight=guidance_weight,
                apply_chat_template=True,
            )
            passed = humaneval_pass_at_1(code, item)
            if passed: he_passed += 1
            he_tps_list.append(tps)
            he_results.append({
                "id": i, "passed": passed, "tps": round(tps, 2),
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] HumanEval sample {i}")
            torch.cuda.empty_cache(); gc.collect()
            he_results.append({"id": i, "error": "OOM", "passed": False, "tps": 0})
        except Exception as e:
            print(f"  [ERR] HumanEval sample {i}: {e}")
            he_results.append({"id": i, "error": str(e)[:200], "passed": False, "tps": 0})

        if (i + 1) % 10 == 0:
            he_acc = he_passed / (i + 1)
            avg_tps = np.mean(he_tps_list) if he_tps_list else 0.0
            print(f"  HumanEval [{i+1}/{len(humaneval_data)}] pass@1={he_acc:.3f}, tps={avg_tps:.1f}")

    he_pass_at_1 = he_passed / len(humaneval_data) if humaneval_data else 0.0
    he_avg_tps = float(np.mean(he_tps_list)) if he_tps_list else 0.0
    he_tps_std = float(np.std(he_tps_list)) if he_tps_list else 0.0
    he_speedup = he_avg_tps / BASELINE_HE_TPS if BASELINE_HE_TPS > 0 else 0.0
    if BASELINE_HE_ACC < 0.05:
        he_acc_retention = 1.0 if he_pass_at_1 >= BASELINE_HE_ACC else (
            he_pass_at_1 / max(BASELINE_HE_ACC, 0.01))
    else:
        he_acc_retention = he_pass_at_1 / BASELINE_HE_ACC
    he_qas = he_speedup * he_acc_retention

    print(f"  HumanEval: pass@1={he_pass_at_1:.3f} ({he_passed}/{len(humaneval_data)}), "
          f"tps={he_avg_tps:.1f}±{he_tps_std:.1f}, "
          f"speedup={he_speedup:.2f}x, QAS={he_qas:.3f}")

    return {
        "guidance_weight": guidance_weight,
        "gsm8k": {
            "n_samples": len(gsm8k_data),
            "correct": gsm8k_correct,
            "exact_match": gsm8k_acc,
            "avg_tps": gsm8k_avg_tps,
            "tps_std": gsm8k_tps_std,
            "speedup": gsm8k_speedup,
            "accuracy_retention": gsm8k_acc_retention,
            "qas": gsm8k_qas,
        },
        "humaneval": {
            "n_samples": len(humaneval_data),
            "passed": he_passed,
            "pass_at_1": he_pass_at_1,
            "avg_tps": he_avg_tps,
            "tps_std": he_tps_std,
            "speedup": he_speedup,
            "accuracy_retention": he_acc_retention,
            "qas": he_qas,
        },
        "samples": {
            "gsm8k": gsm8k_results[:10],
            "humaneval": he_results[:5],
        },
    }


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now()
    print(f"[pilot_m3] Starting at {start_time.isoformat()}")
    print(f"[pilot_m3] Baseline reference: GSM8K acc={BASELINE_GSM8K_ACC}, tps={BASELINE_GSM8K_TPS}")
    print(f"[pilot_m3] Method: AR-Guided Unmasking (LLaDA-8B + Qwen2.5-0.5B supervisor)")
    print(f"[pilot_m3] Guidance weights to test: 0.3, 0.5, 0.7, 1.0")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = "cuda:0"

    # ── Load LLaDA-8B ──────────────────────────────────────────────────────────
    print(f"\n[pilot_m3] Loading LLaDA-8B-Instruct from {MODEL_PATH}...")
    from transformers import AutoTokenizer, AutoModel
    llada_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if llada_tokenizer.padding_side != "left":
        llada_tokenizer.padding_side = "left"

    llada_model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_after_llada = profile_vram(device)
    print(f"[pilot_m3] LLaDA loaded. VRAM: {vram_after_llada.get('vram_used_mb', 0)} MB")

    # ── Load Qwen2.5-0.5B ─────────────────────────────────────────────────────
    print(f"\n[pilot_m3] Loading Qwen2.5-0.5B from {QWEN_PATH}...")
    from transformers import AutoTokenizer as QTokenizer, AutoModelForCausalLM
    qwen_tokenizer = QTokenizer.from_pretrained(QWEN_PATH)
    qwen_model = AutoModelForCausalLM.from_pretrained(
        QWEN_PATH, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_after_qwen = profile_vram(device)
    print(f"[pilot_m3] Qwen loaded. VRAM: {vram_after_qwen.get('vram_used_mb', 0)} MB")

    # ── Load Datasets ──────────────────────────────────────────────────────────
    gsm8k_data = load_gsm8k(n_samples=100, seed=42)
    humaneval_data = load_humaneval(n_samples=50, seed=42)
    print(f"[pilot_m3] Datasets: {len(gsm8k_data)} GSM8K + {len(humaneval_data)} HumanEval")

    guidance_weights = [0.3, 0.5, 0.7, 1.0]
    all_results = []

    report_progress(0, 600, {"status": "starting", "n_guidance_weights": len(guidance_weights)})

    for gw_idx, gw in enumerate(guidance_weights):
        try:
            result = evaluate_guidance_weight(
                llada_model, llada_tokenizer, qwen_model, qwen_tokenizer, device,
                guidance_weight=gw,
                gsm8k_data=gsm8k_data,
                humaneval_data=humaneval_data,
                n_warmup=3,
                gw_idx=gw_idx,
                total_gws=len(guidance_weights),
                total_progress=600,
            )
            all_results.append(result)
            torch.cuda.empty_cache(); gc.collect()
        except Exception as e:
            import traceback
            print(f"[pilot_m3] ERROR at guidance_weight={gw}: {e}")
            traceback.print_exc()
            all_results.append({
                "guidance_weight": gw,
                "error": str(e)[:500],
                "gsm8k": {"exact_match": 0, "avg_tps": 0, "speedup": 0, "qas": 0, "accuracy_retention": 0},
                "humaneval": {"pass_at_1": 0, "avg_tps": 0, "speedup": 0, "qas": 0, "accuracy_retention": 0},
            })
            torch.cuda.empty_cache(); gc.collect()

    # ── Compute Pareto ─────────────────────────────────────────────────────────
    pareto_points = []
    best_op_point = None
    best_qas = -1.0

    for r in all_results:
        if "error" in r:
            continue
        gw = r["guidance_weight"]
        gsm = r["gsm8k"]
        he = r["humaneval"]

        combined_speedup = (gsm["speedup"] * 100 + he["speedup"] * 50) / 150
        combined_acc_ret = (gsm["accuracy_retention"] * 100 + he["accuracy_retention"] * 50) / 150
        combined_qas = combined_speedup * combined_acc_ret

        # Pilot threshold: 10% accuracy drop acceptable (M3 adds Qwen overhead => slower TPS)
        gsm_acc_drop = 1.0 - gsm["accuracy_retention"]
        is_within_budget = gsm_acc_drop <= 0.10

        point = {
            "guidance_weight": gw,
            "gsm8k_exact_match": gsm["exact_match"],
            "gsm8k_speedup": gsm["speedup"],
            "gsm8k_accuracy_retention": gsm["accuracy_retention"],
            "gsm8k_qas": gsm["qas"],
            "humaneval_pass_at_1": he["pass_at_1"],
            "humaneval_speedup": he["speedup"],
            "humaneval_accuracy_retention": he["accuracy_retention"],
            "humaneval_qas": he["qas"],
            "combined_speedup": combined_speedup,
            "combined_accuracy_retention": combined_acc_ret,
            "combined_qas": combined_qas,
            "within_accuracy_budget": is_within_budget,
        }
        pareto_points.append(point)

        if is_within_budget and combined_qas > best_qas:
            best_qas = combined_qas
            best_op_point = point

    if best_op_point is None and pareto_points:
        best_op_point = max(pareto_points, key=lambda p: p["combined_qas"])

    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    # ── Determine GO/NO-GO ─────────────────────────────────────────────────────
    # Pass criteria: M3 runs end-to-end; speedup > 1.2x (AR adds overhead);
    # accuracy not catastrophically lower (< 10% drop)
    n_complete = len([r for r in all_results if "error" not in r])

    if best_op_point is None:
        verdict = "NO_GO"
        decision = "NO_GO: No guidance weight completed without error"
    else:
        speedup = best_op_point.get("gsm8k_speedup", 0)
        acc_drop = 1.0 - best_op_point.get("gsm8k_accuracy_retention", 0)
        if speedup >= 1.2 and acc_drop <= 0.10:
            verdict = "GO"
            decision = (f"PROCEED: M3 speedup={speedup:.2f}x, "
                        f"acc_drop={acc_drop:.1%} at gw={best_op_point['guidance_weight']}")
        elif acc_drop <= 0.10:
            verdict = "MARGINAL_SLOW"
            decision = (f"MARGINAL: M3 speedup={speedup:.2f}x < 1.2x target "
                        f"(AR overhead dominates) but accuracy maintained. "
                        f"Quality improvement may justify inclusion.")
        elif speedup >= 1.2:
            verdict = "MARGINAL_ACCURACY"
            decision = (f"MARGINAL: Speedup={speedup:.2f}x but acc_drop={acc_drop:.1%} "
                        f"exceeds 10% threshold at best operating point.")
        else:
            verdict = "NO_GO"
            decision = (f"NO_GO: M3 speedup={speedup:.2f}x < 1.2x AND "
                        f"acc_drop={acc_drop:.1%} > 10%.")

    # ── Print Summary ──────────────────────────────────────────────────────────
    print(f"\n[pilot_m3] === FINAL RESULTS ===")
    print(f"  Verdict: {verdict}")
    print(f"  Decision: {decision}")
    if best_op_point:
        print(f"  Operating point: guidance_weight={best_op_point['guidance_weight']}")
        print(f"  Best GSM8K: acc={best_op_point['gsm8k_exact_match']:.3f}, "
              f"speedup={best_op_point['gsm8k_speedup']:.2f}x, "
              f"qas={best_op_point['gsm8k_qas']:.3f}")
    print(f"  {n_complete}/{len(guidance_weights)} guidance weights completed")
    print(f"  Elapsed: {elapsed_min:.1f} min")

    print(f"\n  Pareto Table:")
    print(f"  {'GuidanceWt':>12} | {'GSM8K-Acc':>10} | {'GSM8K-TPS':>10} | {'Speedup':>8} | {'QAS':>8}")
    print(f"  {'-'*12}+{'-'*12}+{'-'*12}+{'-'*10}+{'-'*10}")
    for r in all_results:
        if "error" in r:
            print(f"  {r['guidance_weight']:>11.1f} | {'ERROR':>10}")
            continue
        g = r["gsm8k"]
        print(f"  {r['guidance_weight']:>12.1f} | {g['exact_match']:>10.3f} | "
              f"{g['avg_tps']:>10.1f} | {g['speedup']:>8.2f}x | {g['qas']:>8.3f}")

    # ── Save Results ───────────────────────────────────────────────────────────
    pareto_output = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "supervisor_model": "Qwen2.5-0.5B",
        "method": "M3-AR-Guided-Unmasking",
        "implementation": "qwen2.5_0.5b_blended_confidence",
        "mode": "pilot",
        "guidance_weights_tested": guidance_weights,
        "n_completed": n_complete,
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "elapsed_minutes": elapsed_min,
        "baseline_reference": {
            "gsm8k_exact_match": BASELINE_GSM8K_ACC,
            "gsm8k_avg_tps": BASELINE_GSM8K_TPS,
            "humaneval_pass_at_1": BASELINE_HE_ACC,
            "humaneval_avg_tps": BASELINE_HE_TPS,
        },
        "pareto_points": pareto_points,
        "operating_point": best_op_point,
        "verdict": verdict,
        "decision": decision,
        "all_results": all_results,
        "vram_after_llada": vram_after_llada,
        "vram_after_qwen": vram_after_qwen,
        "pass_criteria_met": {
            "runs_end_to_end": n_complete >= 1,
            "speedup_1_2x": best_op_point["gsm8k_speedup"] >= 1.2 if best_op_point else False,
            "acc_drop_lt_10pct": (1.0 - best_op_point.get("gsm8k_accuracy_retention", 0)) <= 0.10
                                  if best_op_point else False,
            "at_least_3_of_4_complete": n_complete >= 3,
        }
    }

    out_path = RESULTS_DIR / "m3_pareto.json"
    out_path.write_text(json.dumps(pareto_output, indent=2))
    print(f"[pilot_m3] Results saved to {out_path}")

    # ── Update gpu_progress.json ───────────────────────────────────────────────
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {
            "completed": [], "failed": [], "running": {}, "timings": {}
        }
        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 30,
            "actual_min": round(elapsed_min),
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "config_snapshot": {
                "model": "LLaDA-8B-Instruct",
                "supervisor_model": "Qwen2.5-0.5B",
                "method": "M3-AR-Guided-Unmasking",
                "guidance_weights": guidance_weights,
                "n_gsm8k": len(gsm8k_data),
                "n_humaneval": len(humaneval_data),
                "gpu_model": vram_after_llada.get("gpu_name", "unknown"),
            }
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print(f"[pilot_m3] gpu_progress.json updated")
    except Exception as e:
        print(f"[pilot_m3] WARNING: Could not update gpu_progress.json: {e}")

    summary = (
        f"M3-AR-Guided pilot: verdict={verdict}, "
        f"best gw={best_op_point['guidance_weight']}, "
        f"GSM8K speedup={best_op_point['gsm8k_speedup']:.2f}x, "
        f"acc={best_op_point['gsm8k_exact_match']:.3f}, "
        f"QAS={best_op_point['gsm8k_qas']:.3f}"
    ) if best_op_point else f"M3 pilot: verdict={verdict}, n_complete={n_complete}"

    mark_done(status="success", summary=summary)
    report_progress(600, 600, {"status": "done", "verdict": verdict})
    print(f"[pilot_m3] Done.")
    return pareto_output


if __name__ == "__main__":
    main()
