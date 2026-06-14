"""
H6 Pilot: IGSD Token Acceptance Rate Feasibility Study.

Task: pilot_h6_igsd_feasibility
- 200 GSM8K + 50 HumanEval prompts, seed=42
- Draft phase: 8-step denoising (T_draft=8 out of T_full=64)
- Measure per-token confidence distribution
- Compute acceptance rate at tau in {0.70, 0.80, 0.85, 0.90}
- Measure draft accuracy (8-step GSM8K exact match)

Pass criteria: Script completes. acceptance_rate at tau=0.85 is recorded.
Decision rule:
  - accept_rate >= 0.50 at tau=0.85 → IGSD is primary contribution
  - accept_rate 0.40-0.50 at tau=0.85 → tune tau to 0.70, report as exploratory
  - accept_rate < 0.40 at tau=0.70 → drop IGSD, pivot to composability study

Usage:
    CUDA_VISIBLE_DEVICES=1 conda run -n sibyl_dlm-acceleration python pilot_h6_igsd_feasibility.py
"""
import os, sys, json, time, random, gc, re
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

# ── Paths ──────────────────────────────────────────────────────────────────────
WORKSPACE     = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/current")
SHARED        = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")
MODEL_PATH    = str(SHARED / "checkpoints" / "llada-8b-instruct")
GSM8K_DIR     = str(SHARED / "datasets" / "gsm8k")
HUMANEVAL_DIR = str(SHARED / "datasets" / "humaneval")
CODE_DIR      = WORKSPACE / "exp" / "code"
RESULTS_DIR   = WORKSPACE / "exp" / "results" / "pilots"
TASK_ID       = "pilot_h6_igsd_feasibility"
MASK_ID       = 126336

sys.path.insert(0, str(CODE_DIR))
sys.path.insert(0, str(CODE_DIR / "LLaDA"))

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── System Monitor Helpers ────────────────────────────────────────────────────
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

# ── Data Loaders ──────────────────────────────────────────────────────────────
def load_gsm8k(n_samples=200, seed=42):
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

# ── GSM8K 8-shot Prompt ───────────────────────────────────────────────────────
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

# ── Draft-phase confidence measurement ───────────────────────────────────────
def run_draft_and_measure_confidence(
    model, tokenizer, prompt_text, device,
    t_draft=8, gen_length=256, block_length=32,
    apply_chat_template=True
):
    """
    Run T_draft=8 denoising steps. Return:
    - draft_text: decoded output after draft
    - per_token_confidence: np.array of shape (gen_length,) ∈ [0,1]
    - acceptance_rates: dict {tau → float}
    - draft_elapsed: wall-clock seconds
    """
    from generate import get_num_transfer_tokens

    if apply_chat_template:
        msg = [{"role": "user", "content": prompt_text}]
        prompt_text = tokenizer.apply_chat_template(
            msg, add_generation_prompt=True, tokenize=False
        )

    enc = tokenizer([prompt_text], add_special_tokens=False,
                    padding=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    # Initialize fully masked generation
    x = torch.full(
        (1, prompt_len + gen_length), MASK_ID, dtype=torch.long
    ).to(device)
    x[:, :prompt_len] = input_ids.clone()
    attn = torch.cat(
        [attention_mask, torch.ones((1, gen_length), dtype=attention_mask.dtype, device=device)],
        dim=-1,
    )

    t0 = time.perf_counter()

    # Block-based draft denoising (T_draft steps total)
    num_blocks = gen_length // block_length
    draft_steps_per_block = max(1, t_draft // num_blocks)

    for block_idx in range(num_blocks):
        block_start = prompt_len + block_idx * block_length
        block_end   = prompt_len + (block_idx + 1) * block_length
        block_mask  = x[:, block_start:block_end] == MASK_ID
        num_transfer = get_num_transfer_tokens(block_mask, draft_steps_per_block)

        for step in range(draft_steps_per_block):
            mask_index = (x == MASK_ID)
            with torch.no_grad():
                logits = model(x, attention_mask=attn).logits
            p = F.softmax(logits, dim=-1)
            x0 = torch.argmax(p, dim=-1)
            x0_p = torch.gather(p, -1, x0.unsqueeze(-1)).squeeze(-1)

            # Prevent premature unmasking of future blocks
            x0_p[:, block_end:] = -float("inf")
            x0 = torch.where(mask_index, x0, x)
            confidence = torch.where(
                mask_index, x0_p,
                torch.tensor(-float("inf"), device=device)
            )

            transfer_index = torch.zeros_like(x0, dtype=torch.bool)
            k = num_transfer[0, step].item()
            if k > 0:
                _, sel = torch.topk(confidence[0], k=int(k))
                transfer_index[0, sel] = True
            x[transfer_index] = x0[transfer_index]

    draft_elapsed = time.perf_counter() - t0

    # Final confidence pass: run one more forward pass on draft output
    # Fill any remaining masked tokens with argmax prediction
    with torch.no_grad():
        logits = model(x, attention_mask=attn).logits
    p_final = F.softmax(logits[:, prompt_len:, :], dim=-1)  # (1, gen_len, vocab)

    gen_region = x[:, prompt_len:].clone()  # (1, gen_len)
    still_masked = (gen_region == MASK_ID)

    # For still-masked tokens, use argmax prediction
    draft_pred = torch.argmax(p_final, dim=-1)  # (1, gen_len)
    draft_conf = torch.gather(p_final, -1, draft_pred.unsqueeze(-1)).squeeze(-1)

    # Confidence for filled tokens: probability of the chosen token
    filled_conf = torch.gather(
        p_final, -1,
        gen_region.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)

    # Merge: filled = filled_conf, masked = draft_conf
    final_confidence = torch.where(still_masked, draft_conf, filled_conf).squeeze(0)  # (gen_len,)
    final_tokens = torch.where(still_masked, draft_pred, gen_region).squeeze(0)

    per_token_conf = final_confidence.float().cpu().numpy()

    # Acceptance rates at each tau
    tau_values = [0.70, 0.80, 0.85, 0.90]
    acceptance_rates = {tau: float((per_token_conf >= tau).mean()) for tau in tau_values}

    # Decode draft text
    draft_text = tokenizer.decode(final_tokens.tolist(), skip_special_tokens=True)

    # TPS = gen_length / elapsed
    tps = gen_length / draft_elapsed if draft_elapsed > 0 else 0.0

    return draft_text, per_token_conf, acceptance_rates, draft_elapsed, tps


def aggregate_confidence_stats(all_confidences):
    """Aggregate per-token confidence arrays into distribution statistics."""
    flat = np.concatenate(all_confidences) if all_confidences else np.array([])
    if len(flat) == 0:
        return {}
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "median": float(np.median(flat)),
        "p10": float(np.percentile(flat, 10)),
        "p25": float(np.percentile(flat, 25)),
        "p75": float(np.percentile(flat, 75)),
        "p90": float(np.percentile(flat, 90)),
        "histogram_counts": np.histogram(flat, bins=10, range=(0, 1))[0].tolist(),
        "histogram_edges": np.histogram(flat, bins=10, range=(0, 1))[1].tolist(),
        "n_tokens": int(len(flat)),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    write_pid()
    start_time = datetime.now().isoformat()
    print(f"[pilot_h6] Starting IGSD feasibility pilot at {start_time}")

    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device = "cuda:0"  # CUDA_VISIBLE_DEVICES=1 → this is GPU 1 physically

    print(f"[pilot_h6] Loading LLaDA-8B-Instruct from {MODEL_PATH}...")
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.padding_side != "left":
        tokenizer.padding_side = "left"

    model = AutoModel.from_pretrained(
        MODEL_PATH, trust_remote_code=True, torch_dtype=torch.bfloat16,
    ).to(device).eval()

    vram_used = torch.cuda.memory_allocated(device) // 1024**2
    print(f"[pilot_h6] Model loaded. VRAM used: {vram_used} MB")

    tau_values = [0.70, 0.80, 0.85, 0.90]

    # ── GSM8K: 200 samples ─────────────────────────────────────────────────────
    print("\n[pilot_h6] === GSM8K Draft Confidence (200 samples) ===")
    gsm8k_data = load_gsm8k(n_samples=200, seed=42)
    gsm8k_confidences = []
    gsm8k_acceptance = {tau: [] for tau in tau_values}
    gsm8k_draft_correct = 0
    gsm8k_results = []

    for i, item in enumerate(gsm8k_data):
        prompt = build_gsm8k_prompt(item["question"])
        try:
            draft_text, per_conf, acc_rates, elapsed, tps = run_draft_and_measure_confidence(
                model, tokenizer, prompt, device,
                t_draft=8, gen_length=256, block_length=32,
                apply_chat_template=True
            )
            gsm8k_confidences.append(per_conf)
            for tau in tau_values:
                gsm8k_acceptance[tau].append(acc_rates[tau])

            correct = gsm8k_exact_match(draft_text, item["answer"])
            if correct: gsm8k_draft_correct += 1

            gsm8k_results.append({
                "id": i,
                "draft_correct": correct,
                "acceptance_rates": acc_rates,
                "mean_confidence": float(np.mean(per_conf)),
                "elapsed_sec": round(elapsed, 3),
                "tps": round(tps, 1),
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Sample {i}")
            torch.cuda.empty_cache(); gc.collect()
            gsm8k_results.append({"id": i, "error": "OOM"})
        except Exception as e:
            print(f"  [ERR] Sample {i}: {e}")
            gsm8k_results.append({"id": i, "error": str(e)})

        if (i + 1) % 20 == 0:
            completed = len([r for r in gsm8k_results if "acceptance_rates" in r])
            if completed > 0:
                ar_085 = float(np.mean([r["acceptance_rates"][0.85] for r in gsm8k_results if "acceptance_rates" in r]))
                draft_acc = gsm8k_draft_correct / (i + 1)
                print(f"  [{i+1}/200] accept_rate@tau=0.85={ar_085:.3f}, draft_acc={draft_acc:.3f}")
            report_progress(i + 1, 250, {
                "gsm8k_accept_rate_085": float(np.mean(gsm8k_acceptance[0.85])) if gsm8k_acceptance[0.85] else 0,
                "gsm8k_draft_accuracy": gsm8k_draft_correct / (i + 1),
            })

    gsm8k_conf_stats = aggregate_confidence_stats(gsm8k_confidences)
    gsm8k_mean_acceptance = {tau: float(np.mean(gsm8k_acceptance[tau])) if gsm8k_acceptance[tau] else 0
                             for tau in tau_values}
    gsm8k_draft_accuracy = gsm8k_draft_correct / len(gsm8k_data) if gsm8k_data else 0.0

    print(f"\n[pilot_h6] GSM8K Summary:")
    print(f"  Confidence: mean={gsm8k_conf_stats.get('mean', 0):.3f}, std={gsm8k_conf_stats.get('std', 0):.3f}")
    for tau in tau_values:
        print(f"  accept_rate@tau={tau:.2f}: {gsm8k_mean_acceptance[tau]:.3f}")
    print(f"  draft_accuracy (8-step GSM8K exact match): {gsm8k_draft_accuracy:.3f}")

    # ── HumanEval: 50 samples ─────────────────────────────────────────────────
    print("\n[pilot_h6] === HumanEval Draft Confidence (50 samples) ===")
    humaneval_data = load_humaneval(n_samples=50, seed=42)
    he_confidences = []
    he_acceptance = {tau: [] for tau in tau_values}
    he_results = []

    for i, problem in enumerate(humaneval_data):
        prompt = problem["prompt"]
        try:
            draft_text, per_conf, acc_rates, elapsed, tps = run_draft_and_measure_confidence(
                model, tokenizer, prompt, device,
                t_draft=8, gen_length=256, block_length=32,
                apply_chat_template=False
            )
            he_confidences.append(per_conf)
            for tau in tau_values:
                he_acceptance[tau].append(acc_rates[tau])

            he_results.append({
                "task_id": problem["task_id"],
                "acceptance_rates": acc_rates,
                "mean_confidence": float(np.mean(per_conf)),
                "elapsed_sec": round(elapsed, 3),
                "tps": round(tps, 1),
            })
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Problem {i}")
            torch.cuda.empty_cache(); gc.collect()
            he_results.append({"task_id": problem.get("task_id", str(i)), "error": "OOM"})
        except Exception as e:
            print(f"  [ERR] Problem {i}: {e}")
            he_results.append({"task_id": problem.get("task_id", str(i)), "error": str(e)})

        if (i + 1) % 10 == 0:
            completed = len([r for r in he_results if "acceptance_rates" in r])
            if completed > 0:
                ar_085 = float(np.mean([r["acceptance_rates"][0.85] for r in he_results if "acceptance_rates" in r]))
                print(f"  [{i+1}/50] accept_rate@tau=0.85={ar_085:.3f}")
            report_progress(200 + i + 1, 250, {
                "humaneval_accept_rate_085": float(np.mean(he_acceptance[0.85])) if he_acceptance[0.85] else 0,
            })

    he_conf_stats = aggregate_confidence_stats(he_confidences)
    he_mean_acceptance = {tau: float(np.mean(he_acceptance[tau])) if he_acceptance[tau] else 0
                          for tau in tau_values}

    print(f"\n[pilot_h6] HumanEval Summary:")
    print(f"  Confidence: mean={he_conf_stats.get('mean', 0):.3f}, std={he_conf_stats.get('std', 0):.3f}")
    for tau in tau_values:
        print(f"  accept_rate@tau={tau:.2f}: {he_mean_acceptance[tau]:.3f}")

    # ── Combined Stats ─────────────────────────────────────────────────────────
    all_confidences = gsm8k_confidences + he_confidences
    overall_conf_stats = aggregate_confidence_stats(all_confidences)
    overall_mean_acceptance = {
        tau: float(np.mean(gsm8k_acceptance[tau] + he_acceptance[tau]))
        if (gsm8k_acceptance[tau] or he_acceptance[tau]) else 0
        for tau in tau_values
    }

    # ── Decision Logic ─────────────────────────────────────────────────────────
    ar_085 = overall_mean_acceptance.get(0.85, 0)
    ar_070 = overall_mean_acceptance.get(0.70, 0)
    if ar_085 >= 0.50:
        igsd_decision = "PROCEED: IGSD is primary contribution (accept_rate@0.85 >= 50%)"
        igsd_verdict = "go"
    elif ar_085 >= 0.40:
        igsd_decision = f"EXPLORATORY: Tune tau to 0.70 (accept_rate@0.85 = {ar_085:.3f}). Proceed with reduced expectations."
        igsd_verdict = "conditional"
    elif ar_070 < 0.40:
        igsd_decision = f"DROP IGSD: accept_rate@0.70 = {ar_070:.3f} < 40%. Pivot to composability study."
        igsd_verdict = "drop"
    else:
        igsd_decision = f"MARGINAL: accept_rate@0.85={ar_085:.3f}, @0.70={ar_070:.3f}. Review before proceeding."
        igsd_verdict = "marginal"

    print(f"\n[pilot_h6] ═══════════════════════════════════════════════")
    print(f"[pilot_h6] DECISION: {igsd_decision}")
    print(f"[pilot_h6] ═══════════════════════════════════════════════")

    # ── Save Results ───────────────────────────────────────────────────────────
    end_time = datetime.now().isoformat()
    metrics = {
        "task_id": TASK_ID,
        "model": "LLaDA-8B-Instruct",
        "t_draft": 8,
        "t_full": 64,
        "gen_length": 256,
        "block_length": 32,
        "seed": 42,
        "tau_values": tau_values,
        "start_time": start_time,
        "end_time": end_time,
        "igsd_verdict": igsd_verdict,
        "igsd_decision": igsd_decision,
        "acceptance_rate_by_tau": {
            "overall": overall_mean_acceptance,
            "gsm8k": gsm8k_mean_acceptance,
            "humaneval": he_mean_acceptance,
        },
        "confidence_distribution_stats": {
            "overall": overall_conf_stats,
            "gsm8k": gsm8k_conf_stats,
            "humaneval": he_conf_stats,
        },
        "draft_accuracy_vs_baseline": {
            "gsm8k_8step_exact_match": gsm8k_draft_accuracy,
            "gsm8k_n_samples": len(gsm8k_data),
            "gsm8k_n_correct": gsm8k_draft_correct,
            "note": "baseline (64-step) GSM8K accuracy measured in pilot_baseline"
        },
        "vram_used_mb": torch.cuda.memory_allocated(device) // 1024**2,
    }

    out_path = RESULTS_DIR / "h6_igsd_metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"\n[pilot_h6] Saved metrics to {out_path}")

    # Save per-sample results
    samples_path = RESULTS_DIR / "h6_igsd_samples.json"
    samples_path.write_text(json.dumps({
        "gsm8k_samples": gsm8k_results[:50],
        "humaneval_samples": he_results[:20],
    }, indent=2))

    summary = (
        f"accept_rate@tau=0.85={ar_085:.3f}, "
        f"@tau=0.70={ar_070:.3f}, "
        f"draft_acc(8step)={gsm8k_draft_accuracy:.3f}, "
        f"verdict={igsd_verdict}"
    )
    mark_done(status="success", summary=summary)
    print(f"\n[pilot_h6] DONE. {summary}")
    return metrics


if __name__ == "__main__":
    main()
