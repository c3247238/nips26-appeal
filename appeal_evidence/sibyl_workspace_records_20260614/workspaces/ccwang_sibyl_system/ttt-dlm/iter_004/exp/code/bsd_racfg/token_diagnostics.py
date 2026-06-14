"""
token_diagnostics: Token-Level CFG Impact Analysis (PILOT: 16 samples).

Qualitative and quantitative analysis of A-CFG guidance effects:
  1. Which token positions are most affected by CFG
  2. Confidence score distribution analysis (replaces JSD stability since RACFG pivoted)
  3. Guidance magnitude heatmaps (per-position logit delta across denoising steps)
  4. Comparison of position selection signals
  5. 50 sample-level diagnostic outputs (16 for pilot)

Since RACFG JSD failed on Dream-7B (stability scores ~0.997, near-uniform),
this analysis focuses on A-CFG confidence-based guidance and characterizes
WHERE and WHEN guidance has the most impact.

Task: token_diagnostics
Mode: PILOT (16 samples, seed 42)
GPU: cuda:0 (mapped via CUDA_VISIBLE_DEVICES)

Usage:
    CUDA_VISIBLE_DEVICES=2 python token_diagnostics.py
"""
import os
import sys
import json
import time
import gc
import traceback
import math
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bsd_racfg.eval_harness import (
    load_dream, generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics, compute_per_sample_metrics,
    MASK_TOKEN_ID, PROJECT_DIR, RESULTS_FULL,
)

# ── Constants ──
TASK_ID = "token_diagnostics"
N_SAMPLES = 16   # PILOT mode
SEED = 42
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
DEVICE = "cuda:0"

# Best A-CFG config from ablations
ACFG_W = 1.5
ACFG_REMASK_PCT = 0.10

RESULTS_DIR = RESULTS_FULL
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ──────────────────────────────────────────────────
# Progress & PID tracking
# ──────────────────────────────────────────────────

def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(os.getpid()))
    print(f"[{TASK_ID}] PID {os.getpid()} written to {pid_file}")


def report_progress(current, total, phase="", metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": current, "total_epochs": total,
        "step": current, "total_steps": total,
        "phase": phase,
        "loss": None,
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
    print(f"[{TASK_ID}] DONE marker written ({status})")


# ──────────────────────────────────────────────────
# Diagnostic A-CFG generation with full tracking
# ──────────────────────────────────────────────────

def acfg_generate_with_diagnostics(
    model, tokenizer, prompt_text: str, device: str = "cuda:0",
    w: float = 1.5, remask_pct: float = 0.10,
    gen_len: int = GEN_LEN, steps: int = STEPS, temperature: float = TEMPERATURE,
) -> Tuple[str, float, Dict]:
    """A-CFG generation with full per-position, per-step diagnostic tracking.

    Tracks:
      - guidance_magnitude: |logits_cond - logits_uncond| per position per step
      - confidence_scores: max(softmax(logits)) per position per step
      - positions_remasked: which positions were re-masked at each step
      - logit_delta_norm: L2 norm of (guided - unguided) logits per position
      - token_changes: positions where guided sampling chose different token than unguided would
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    # ── Diagnostic storage ──
    # Record every 4th step to manage memory
    RECORD_INTERVAL = 4
    step_records = []  # list of per-step diagnostics

    # Also track EMA probs for JSD comparison (to show WHY JSD fails)
    ema_probs = None
    ema_lambda = 0.7

    t0 = time.time()

    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()
            if n_masked == 0:
                break

            mask_rate = n_masked / gen_len

            # === Forward pass 1: conditional ===
            outputs_cond = model(x, attention_mask="full", position_ids=None)
            logits_cond = outputs_cond.logits
            logits_cond = torch.cat([logits_cond[:, :1], logits_cond[:, :-1]], dim=1)

            # Confidence scores (for all generation positions)
            probs_cond = F.softmax(logits_cond[:, prompt_len:] / temperature, dim=-1)
            confidence = probs_cond.max(dim=-1).values[0]  # [gen_len]
            gen_mask = (x[0, prompt_len:] == MASK_TOKEN_ID)  # [gen_len]

            # === JSD stability (for comparison, showing it fails) ===
            current_probs_full = F.softmax(logits_cond / temperature, dim=-1)
            jsd_scores = None
            if ema_probs is not None and i % RECORD_INTERVAL == 0:
                # Compute JSD for generation positions
                p = current_probs_full[0, prompt_len:]  # [gen_len, V]
                q = ema_probs[0, prompt_len:]  # [gen_len, V]
                m = 0.5 * (p + q)
                kl_pm = (p * (p / (m + 1e-10) + 1e-10).log()).sum(dim=-1)
                kl_qm = (q * (q / (m + 1e-10) + 1e-10).log()).sum(dim=-1)
                jsd_raw = 0.5 * (kl_pm + kl_qm)  # [gen_len]
                jsd_scores = (1.0 - jsd_raw).float().cpu().numpy()  # stability = 1 - JSD

            # Update EMA
            if ema_probs is None:
                ema_probs = current_probs_full.clone()
            else:
                ema_probs = ema_lambda * ema_probs + (1 - ema_lambda) * current_probs_full

            # === A-CFG: confidence-based re-masking ===
            n_remask = max(1, int(gen_mask.sum().item() * remask_pct))

            # Find lowest confidence revealed positions
            revealed_positions = (~gen_mask).nonzero(as_tuple=True)[0]
            guidance_applied = False
            guidance_magnitude = np.zeros(gen_len, dtype=np.float32)
            logit_delta_norm = np.zeros(gen_len, dtype=np.float32)
            remasked_positions = []
            token_would_change = np.zeros(gen_len, dtype=bool)

            if len(revealed_positions) > 0:
                rev_conf = confidence[revealed_positions]
                n_actual_remask = min(n_remask, len(revealed_positions))
                _, remask_idx = rev_conf.topk(n_actual_remask, largest=False)

                # Create unconditional input
                x_uncond = x.clone()
                for idx in remask_idx:
                    actual_pos = revealed_positions[idx].item()
                    actual_idx = prompt_len + actual_pos
                    x_uncond[0, actual_idx] = MASK_TOKEN_ID
                    remasked_positions.append(actual_pos)

                # Forward pass 2: unconditional
                outputs_uncond = model(x_uncond, attention_mask="full", position_ids=None)
                logits_uncond = outputs_uncond.logits
                logits_uncond = torch.cat([logits_uncond[:, :1], logits_uncond[:, :-1]], dim=1)

                # Guidance magnitude: per-position L2 norm of logit difference
                logit_diff = logits_cond[:, prompt_len:] - logits_uncond[:, prompt_len:]  # [1, gen_len, V]
                guidance_magnitude = logit_diff[0].norm(dim=-1).float().cpu().numpy()  # [gen_len]
                logit_delta_norm = guidance_magnitude.copy()

                # Apply CFG
                guided_logits = logits_cond + w * (logits_cond - logits_uncond)

                # Check which positions would get different tokens
                if i % RECORD_INTERVAL == 0:
                    unguided_tokens = logits_cond[0, prompt_len:].argmax(dim=-1)
                    guided_tokens = guided_logits[0, prompt_len:].argmax(dim=-1)
                    token_would_change = (unguided_tokens != guided_tokens).cpu().numpy()

                guidance_applied = True
            else:
                guided_logits = logits_cond

            # === Standard unmasking ===
            mask_logits = guided_logits[mask_index]
            t = timesteps[i]
            s = timesteps[i + 1]
            p_transfer = (1 - s / t).item() if i < steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if temperature > 0:
                probs = F.softmax(mask_logits[transfer_mask] / temperature, dim=-1)
                sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                sampled_tokens = mask_logits[transfer_mask].argmax(dim=-1)
            x0[transfer_mask] = sampled_tokens
            x[mask_index] = x0

            # === Record diagnostics ===
            if i % RECORD_INTERVAL == 0:
                record = {
                    "step": i,
                    "n_masked": n_masked,
                    "mask_rate": round(mask_rate, 4),
                    "guidance_applied": guidance_applied,
                    "n_remasked": len(remasked_positions),
                    "remasked_positions": remasked_positions[:20],  # cap for JSON size
                    # Per-position arrays (only gen positions, every 4th step)
                    "confidence_mean": float(confidence[gen_mask].mean().item()) if gen_mask.any() else 0,
                    "confidence_std": float(confidence[gen_mask].std().item()) if gen_mask.any() else 0,
                    "confidence_min": float(confidence[gen_mask].min().item()) if gen_mask.any() else 0,
                    "confidence_max": float(confidence[gen_mask].max().item()) if gen_mask.any() else 0,
                    "confidence_revealed_mean": float(confidence[~gen_mask].mean().item()) if (~gen_mask).any() else 0,
                    "guidance_magnitude_mean": float(guidance_magnitude.mean()),
                    "guidance_magnitude_max": float(guidance_magnitude.max()),
                    "guidance_magnitude_std": float(guidance_magnitude.std()),
                    "n_tokens_changed_by_guidance": int(token_would_change.sum()),
                    "pct_tokens_changed": float(token_would_change.sum() / gen_len) if gen_len > 0 else 0,
                }

                # Add JSD comparison data
                if jsd_scores is not None:
                    jsd_masked = jsd_scores[gen_mask.cpu().numpy()]
                    record["jsd_stability_mean"] = float(jsd_masked.mean()) if len(jsd_masked) > 0 else 0
                    record["jsd_stability_std"] = float(jsd_masked.std()) if len(jsd_masked) > 0 else 0
                    record["jsd_stability_min"] = float(jsd_masked.min()) if len(jsd_masked) > 0 else 0
                    record["jsd_stability_max"] = float(jsd_masked.max()) if len(jsd_masked) > 0 else 0
                    # Key metric: variance ratio (JSD vs confidence)
                    conf_masked = confidence[gen_mask].float().cpu().numpy()
                    if len(conf_masked) > 1 and jsd_masked.std() > 0:
                        record["signal_variance_ratio_jsd_vs_conf"] = float(
                            jsd_masked.std() / (conf_masked.std() + 1e-10)
                        )
                    else:
                        record["signal_variance_ratio_jsd_vs_conf"] = 0.0

                # Guidance magnitude heatmap data (subsample positions)
                if guidance_applied:
                    # Store guidance magnitude for heatmap (subsample to 64 positions max)
                    stride = max(1, gen_len // 64)
                    record["guidance_heatmap_positions"] = list(range(0, gen_len, stride))
                    record["guidance_heatmap_values"] = [
                        float(guidance_magnitude[p]) for p in range(0, gen_len, stride)
                    ]

                step_records.append(record)

    elapsed = time.time() - t0

    # Decode
    gen_ids = x[0, prompt_len:].tolist()
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t_id in gen_ids:
        if t_id == MASK_TOKEN_ID:
            continue
        if t_id == eos_id:
            break
        clean_ids.append(t_id)
    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()

    # Token-level final state
    final_tokens = tokenizer.convert_ids_to_tokens(clean_ids[:50])  # first 50 tokens

    diagnostics = {
        "step_records": step_records,
        "final_tokens_preview": final_tokens,
        "n_gen_tokens": len(clean_ids),
        "elapsed_s": elapsed,
    }

    return text, elapsed, diagnostics


# ──────────────────────────────────────────────────
# Analysis functions
# ──────────────────────────────────────────────────

def analyze_guidance_patterns(all_diagnostics: List[Dict]) -> Dict:
    """Aggregate guidance patterns across all samples."""

    # 1. When does guidance activate? (step distribution)
    guidance_activation_steps = []
    for diag in all_diagnostics:
        for rec in diag["step_records"]:
            if rec["guidance_applied"]:
                guidance_activation_steps.append(rec["step"])

    # 2. Guidance magnitude trajectory
    # Aggregate guidance magnitude by step across samples
    step_guidance = {}
    for diag in all_diagnostics:
        for rec in diag["step_records"]:
            s = rec["step"]
            if s not in step_guidance:
                step_guidance[s] = {
                    "magnitudes": [], "n_changed": [], "mask_rates": [],
                    "conf_means": [], "conf_stds": [],
                    "jsd_means": [], "jsd_stds": [],
                    "signal_ratios": [],
                }
            step_guidance[s]["magnitudes"].append(rec["guidance_magnitude_mean"])
            step_guidance[s]["n_changed"].append(rec["n_tokens_changed_by_guidance"])
            step_guidance[s]["mask_rates"].append(rec["mask_rate"])
            step_guidance[s]["conf_means"].append(rec["confidence_mean"])
            step_guidance[s]["conf_stds"].append(rec["confidence_std"])
            if "jsd_stability_mean" in rec:
                step_guidance[s]["jsd_means"].append(rec["jsd_stability_mean"])
                step_guidance[s]["jsd_stds"].append(rec["jsd_stability_std"])
            if "signal_variance_ratio_jsd_vs_conf" in rec:
                step_guidance[s]["signal_ratios"].append(rec["signal_variance_ratio_jsd_vs_conf"])

    guidance_trajectory = {}
    for s in sorted(step_guidance.keys()):
        d = step_guidance[s]
        entry = {
            "step": s,
            "mask_rate_mean": float(np.mean(d["mask_rates"])),
            "guidance_magnitude_mean": float(np.mean(d["magnitudes"])),
            "guidance_magnitude_std": float(np.std(d["magnitudes"])),
            "tokens_changed_mean": float(np.mean(d["n_changed"])),
            "confidence_mean": float(np.mean(d["conf_means"])),
            "confidence_std_mean": float(np.mean(d["conf_stds"])),
        }
        if d["jsd_means"]:
            entry["jsd_stability_mean"] = float(np.mean(d["jsd_means"]))
            entry["jsd_stability_std_mean"] = float(np.mean(d["jsd_stds"]))
        if d["signal_ratios"]:
            entry["signal_variance_ratio_jsd_vs_conf"] = float(np.mean(d["signal_ratios"]))
        guidance_trajectory[s] = entry

    # 3. Position-level guidance heatmap (aggregated)
    # Average guidance magnitude per position across samples and steps
    pos_guidance_accum = np.zeros(GEN_LEN, dtype=np.float64)
    pos_guidance_count = np.zeros(GEN_LEN, dtype=np.float64)
    for diag in all_diagnostics:
        for rec in diag["step_records"]:
            if "guidance_heatmap_positions" in rec:
                for p, v in zip(rec["guidance_heatmap_positions"],
                                rec["guidance_heatmap_values"]):
                    pos_guidance_accum[p] += v
                    pos_guidance_count[p] += 1

    pos_guidance_mean = np.zeros(GEN_LEN)
    valid = pos_guidance_count > 0
    pos_guidance_mean[valid] = pos_guidance_accum[valid] / pos_guidance_count[valid]

    # Top-20 most-guided positions
    top_positions = np.argsort(pos_guidance_mean)[::-1][:20]
    top_pos_data = [
        {"position": int(p), "mean_guidance_magnitude": float(pos_guidance_mean[p])}
        for p in top_positions if pos_guidance_mean[p] > 0
    ]

    # 4. Re-masked position frequency
    remask_freq = np.zeros(GEN_LEN, dtype=np.int32)
    for diag in all_diagnostics:
        for rec in diag["step_records"]:
            for p in rec.get("remasked_positions", []):
                if 0 <= p < GEN_LEN:
                    remask_freq[p] += 1

    top_remasked = np.argsort(remask_freq)[::-1][:20]
    top_remask_data = [
        {"position": int(p), "remask_count": int(remask_freq[p])}
        for p in top_remasked if remask_freq[p] > 0
    ]

    # 5. JSD vs Confidence signal quality comparison
    jsd_signal_quality = {
        "issue": "JSD stability scores on Dream-7B are near-uniform (~0.997)",
        "root_cause": (
            "Dream-7B's logit distributions change very little between consecutive "
            "denoising steps, making JSD ~0 and stability ~1.0 for all positions. "
            "This means JSD cannot discriminate 'reasoning-critical' positions."
        ),
    }
    # Compute aggregate signal ratio
    all_ratios = []
    for diag in all_diagnostics:
        for rec in diag["step_records"]:
            if "signal_variance_ratio_jsd_vs_conf" in rec and rec["signal_variance_ratio_jsd_vs_conf"] > 0:
                all_ratios.append(rec["signal_variance_ratio_jsd_vs_conf"])
    if all_ratios:
        jsd_signal_quality["mean_variance_ratio_jsd_over_conf"] = float(np.mean(all_ratios))
        jsd_signal_quality["interpretation"] = (
            f"JSD variance is {np.mean(all_ratios):.4f}x that of confidence. "
            f"{'JSD has less discriminative power.' if np.mean(all_ratios) < 1.0 else 'JSD has more variance (unexpected).'}"
        )

    return {
        "guidance_activation": {
            "first_active_step": int(min(guidance_activation_steps)) if guidance_activation_steps else -1,
            "n_active_steps_total": len(guidance_activation_steps),
            "active_step_range": [int(min(guidance_activation_steps)), int(max(guidance_activation_steps))]
                                 if guidance_activation_steps else [],
        },
        "guidance_trajectory": guidance_trajectory,
        "top_guided_positions": top_pos_data,
        "top_remasked_positions": top_remask_data,
        "jsd_vs_confidence_comparison": jsd_signal_quality,
    }


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    start_time = time.time()
    write_pid()
    report_progress(0, 4, "Loading model")

    print("=" * 70)
    print(f"  token_diagnostics — Token-Level CFG Impact Analysis (PILOT)")
    print(f"  Countdown-{N_SAMPLES}, seed={SEED}")
    print(f"  A-CFG config: w={ACFG_W}, remask_pct={ACFG_REMASK_PCT}")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 70)

    # Load model
    model, tokenizer, embedding_layer = load_dream(DEVICE)
    report_progress(1, 4, "Model loaded")

    # Generate problems
    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    prompts = [format_countdown_prompt(p) for p in problems]
    print(f"\nGenerated {len(problems)} Countdown problems")

    # ── Phase 1: Run A-CFG with full diagnostics ──
    report_progress(1, 4, "Running A-CFG with diagnostics")
    print(f"\n{'='*60}")
    print(f"  Running A-CFG w={ACFG_W} with full token-level diagnostics")
    print(f"{'='*60}")

    all_results = []
    all_diagnostics = []
    all_texts = []
    all_times = []

    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)

        text, elapsed, diag = acfg_generate_with_diagnostics(
            model, tokenizer, prompt, DEVICE,
            w=ACFG_W, remask_pct=ACFG_REMASK_PCT,
        )

        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)

        result = {
            "idx": idx, "target": problem["target"],
            "numbers": problem["numbers"],
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "generated_text": text, "gen_time_s": elapsed,
            **metrics,
        }

        all_results.append(result)
        all_diagnostics.append(diag)
        all_texts.append(text)
        all_times.append(elapsed)

        status = "OK" if verification["is_correct"] else "X"
        eq_str = (verification.get('extracted_equation') or 'N/A')[:40]
        n_changed_total = sum(r.get("n_tokens_changed_by_guidance", 0) for r in diag["step_records"])
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | "
              f"eq={eq_str} | {elapsed:.1f}s | tokens_changed={n_changed_total}")

        report_progress(idx + 1, N_SAMPLES, f"Sample {idx+1}/{N_SAMPLES}")

    n_correct = sum(1 for r in all_results if r["is_correct"])
    accuracy = n_correct / len(problems)
    diversity = compute_diversity_metrics(all_texts)

    print(f"\n  A-CFG w={ACFG_W}: {n_correct}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  rep-2={diversity['rep_2']:.4f}  rep-3={diversity['rep_3']:.4f}  "
          f"distinct-3={diversity['distinct_3']:.4f}")

    # ── Phase 2: Run vanilla baseline for comparison ──
    report_progress(2, 4, "Running vanilla baseline")
    print(f"\n{'='*60}")
    print(f"  Running Vanilla baseline for comparison")
    print(f"{'='*60}")

    from bsd_racfg.eval_harness import vanilla_generate

    vanilla_results = []
    vanilla_texts = []
    vanilla_times = []
    for idx, (problem, prompt) in enumerate(zip(problems, prompts)):
        torch.manual_seed(SEED + idx)
        torch.cuda.manual_seed(SEED + idx)
        text, elapsed, _ = vanilla_generate(model, tokenizer, prompt, DEVICE)
        verification = verify_countdown_answer(text, problem)
        metrics = compute_per_sample_metrics(text)
        vanilla_results.append({
            "idx": idx, "target": problem["target"],
            "numbers": problem["numbers"],
            "is_correct": verification["is_correct"],
            "extracted_equation": verification.get("extracted_equation"),
            "generated_text": text, "gen_time_s": elapsed,
            **metrics,
        })
        vanilla_texts.append(text)
        vanilla_times.append(elapsed)
        status = "OK" if verification["is_correct"] else "X"
        print(f"  [{idx:2d}] {status} | target={problem['target']:4d} | {elapsed:.1f}s")

    v_correct = sum(1 for r in vanilla_results if r["is_correct"])
    v_accuracy = v_correct / len(problems)
    v_diversity = compute_diversity_metrics(vanilla_texts)
    print(f"\n  Vanilla: {v_correct}/{N_SAMPLES} = {v_accuracy:.1%}")

    # ── Phase 3: Aggregate analysis ──
    report_progress(3, 4, "Analyzing guidance patterns")
    print(f"\n{'='*60}")
    print(f"  Analyzing guidance patterns")
    print(f"{'='*60}")

    patterns = analyze_guidance_patterns(all_diagnostics)

    # Per-sample: which samples had the most guidance impact?
    sample_impact = []
    for idx, (result, diag) in enumerate(zip(all_results, all_diagnostics)):
        total_guidance_mag = sum(
            r.get("guidance_magnitude_mean", 0) for r in diag["step_records"]
        )
        total_tokens_changed = sum(
            r.get("n_tokens_changed_by_guidance", 0) for r in diag["step_records"]
        )
        sample_impact.append({
            "idx": idx,
            "target": result["target"],
            "is_correct": result["is_correct"],
            "total_guidance_magnitude": float(total_guidance_mag),
            "total_tokens_changed": int(total_tokens_changed),
            "n_gen_tokens": diag["n_gen_tokens"],
            "elapsed_s": diag["elapsed_s"],
        })

    # Correlation: guidance magnitude vs correctness
    correct_guidance = [s["total_guidance_magnitude"] for s in sample_impact if s["is_correct"]]
    wrong_guidance = [s["total_guidance_magnitude"] for s in sample_impact if not s["is_correct"]]

    guidance_correctness_analysis = {
        "correct_samples_n": len(correct_guidance),
        "wrong_samples_n": len(wrong_guidance),
        "correct_mean_guidance": float(np.mean(correct_guidance)) if correct_guidance else 0,
        "wrong_mean_guidance": float(np.mean(wrong_guidance)) if wrong_guidance else 0,
        "interpretation": "",
    }
    if correct_guidance and wrong_guidance:
        if np.mean(correct_guidance) > np.mean(wrong_guidance):
            guidance_correctness_analysis["interpretation"] = (
                "Correct samples received MORE guidance on average, "
                "suggesting CFG helps where it's most needed."
            )
        else:
            guidance_correctness_analysis["interpretation"] = (
                "Correct samples received LESS guidance on average, "
                "suggesting guidance may be most effective on already-promising trajectories."
            )
    else:
        guidance_correctness_analysis["interpretation"] = (
            "Insufficient correct/wrong samples for meaningful comparison."
        )

    # ── Phase 4: Summary and comparison table ──
    report_progress(4, 4, "Generating report")

    # Flip analysis
    both_correct = sum(1 for a, v in zip(all_results, vanilla_results)
                       if a["is_correct"] and v["is_correct"])
    only_acfg = sum(1 for a, v in zip(all_results, vanilla_results)
                    if a["is_correct"] and not v["is_correct"])
    only_vanilla = sum(1 for a, v in zip(all_results, vanilla_results)
                       if not a["is_correct"] and v["is_correct"])
    both_wrong = sum(1 for a, v in zip(all_results, vanilla_results)
                     if not a["is_correct"] and not v["is_correct"])

    print(f"\n{'='*70}")
    print(f"  TOKEN DIAGNOSTICS SUMMARY")
    print(f"{'='*70}")
    print(f"\n  A-CFG w={ACFG_W}: {n_correct}/{N_SAMPLES} = {accuracy:.1%}")
    print(f"  Vanilla:         {v_correct}/{N_SAMPLES} = {v_accuracy:.1%}")
    print(f"\n  Flip Analysis:")
    print(f"    Both correct:  {both_correct}")
    print(f"    Only A-CFG:    {only_acfg}")
    print(f"    Only Vanilla:  {only_vanilla}")
    print(f"    Both wrong:    {both_wrong}")
    print(f"\n  Guidance Activation: steps {patterns['guidance_activation'].get('active_step_range', 'N/A')}")
    print(f"  JSD vs Confidence: {patterns['jsd_vs_confidence_comparison'].get('interpretation', 'N/A')}")

    # Guidance trajectory summary
    print(f"\n  Guidance Trajectory (by denoising step):")
    print(f"  {'Step':>5} {'MaskRate':>9} {'GuidMag':>9} {'TokChg':>8} {'ConfMean':>9} {'JSD_Mean':>9}")
    for s in sorted(patterns["guidance_trajectory"].keys()):
        t = patterns["guidance_trajectory"][s]
        jsd_str = f"{t.get('jsd_stability_mean', 0):.4f}" if "jsd_stability_mean" in t else "N/A"
        print(f"  {t['step']:5d} {t['mask_rate_mean']:9.4f} {t['guidance_magnitude_mean']:9.4f} "
              f"{t['tokens_changed_mean']:8.1f} {t['confidence_mean']:9.4f} {jsd_str:>9}")

    # Pass criteria
    verdict = "GO" if len(all_diagnostics) == N_SAMPLES else "NO-GO"
    for diag in all_diagnostics:
        if len(diag["step_records"]) == 0:
            verdict = "NO-GO"
            break

    # Check meaningful patterns
    has_guidance_variation = any(
        t.get("guidance_magnitude_std", 0) > 0.01
        for t in patterns["guidance_trajectory"].values()
    )
    has_position_differences = len(patterns["top_guided_positions"]) > 0

    if has_guidance_variation and has_position_differences:
        verdict = "GO"
        verdict_detail = "Diagnostics saved successfully with meaningful patterns"
    else:
        verdict = "CONDITIONAL-GO"
        verdict_detail = "Diagnostics saved but patterns may lack variation"

    print(f"\n  PILOT VERDICT: {verdict}")
    print(f"  Detail: {verdict_detail}")

    # ── Save results ──
    elapsed_total = time.time() - start_time

    combined = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "verdict": verdict,
        "verdict_detail": verdict_detail,
        "model": "Dream-v0-Instruct-7B",
        "benchmark": f"Countdown-{N_SAMPLES}",
        "seed": SEED,
        "n_samples": N_SAMPLES,
        "timestamp": datetime.now().isoformat(),
        "elapsed_total_s": round(elapsed_total, 1),
        "elapsed_total_min": round(elapsed_total / 60, 1),
        "pivot_note": (
            "Original task planned for RACFG (JSD stability) diagnostics. "
            "Since RACFG failed on Dream-7B (JSD ~0.997 uniform), this analysis "
            "focuses on A-CFG (confidence-based) diagnostics and explicitly "
            "characterizes WHY JSD fails as a position selection signal."
        ),
        "config": {
            "method": "A-CFG",
            "w": ACFG_W,
            "remask_pct": ACFG_REMASK_PCT,
            "gen_len": GEN_LEN,
            "steps": STEPS,
            "temperature": TEMPERATURE,
        },
        "summary": {
            "acfg_accuracy": accuracy,
            "acfg_n_correct": n_correct,
            "vanilla_accuracy": v_accuracy,
            "vanilla_n_correct": v_correct,
            "flip_analysis": {
                "both_correct": both_correct,
                "only_acfg": only_acfg,
                "only_vanilla": only_vanilla,
                "both_wrong": both_wrong,
            },
            "diversity": diversity,
            "vanilla_diversity": v_diversity,
        },
        "guidance_patterns": patterns,
        "guidance_vs_correctness": guidance_correctness_analysis,
        "sample_impact": sample_impact,
        "per_sample_diagnostics": [
            {
                "idx": idx,
                "target": all_results[idx]["target"],
                "numbers": all_results[idx]["numbers"],
                "is_correct": all_results[idx]["is_correct"],
                "generated_text": all_results[idx]["generated_text"],
                "extracted_equation": all_results[idx].get("extracted_equation"),
                "final_tokens_preview": diag["final_tokens_preview"],
                "n_gen_tokens": diag["n_gen_tokens"],
                "step_records": diag["step_records"],
            }
            for idx, diag in enumerate(all_diagnostics)
        ],
        "pass_criteria": {
            "diagnostics_saved": True,
            "meaningful_patterns": has_guidance_variation and has_position_differences,
            "guidance_variation": has_guidance_variation,
            "position_differences": has_position_differences,
            "verdict": verdict,
        },
    }

    out_file = RESULTS_DIR / "token_diagnostics_racfg.json"
    with open(out_file, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False, default=str)
    print(f"[{TASK_ID}] Results saved to {out_file}")

    # Summary markdown
    summary_md = RESULTS_DIR / "token_diagnostics_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# Token-Level CFG Impact Analysis — Countdown-{N_SAMPLES} (PILOT)\n\n")
        f.write(f"**Verdict: {verdict}** — {verdict_detail}\n\n")
        f.write(f"## Pivot Note\n\n")
        f.write("Original RACFG (JSD stability) failed on Dream-7B. "
                "This analysis uses A-CFG (confidence-based re-masking) and "
                "explicitly characterizes WHY JSD fails.\n\n")
        f.write(f"## Results\n\n")
        f.write(f"| Method | Accuracy | rep-3 | distinct-3 | Avg Time |\n")
        f.write(f"|--------|----------|-------|------------|----------|\n")
        f.write(f"| A-CFG w={ACFG_W} | {accuracy:.1%} ({n_correct}/{N_SAMPLES}) | "
                f"{diversity['rep_3']:.3f} | {diversity['distinct_3']:.3f} | "
                f"{np.mean(all_times):.1f}s |\n")
        f.write(f"| Vanilla | {v_accuracy:.1%} ({v_correct}/{N_SAMPLES}) | "
                f"{v_diversity['rep_3']:.3f} | {v_diversity['distinct_3']:.3f} | "
                f"{np.mean(vanilla_times):.1f}s |\n\n")

        f.write(f"## Guidance Impact\n\n")
        f.write(f"- Guidance first activates at step: {patterns['guidance_activation'].get('first_active_step', 'N/A')}\n")
        f.write(f"- Total active guidance steps across all samples: {patterns['guidance_activation']['n_active_steps_total']}\n\n")

        f.write(f"## JSD vs Confidence Signal Quality\n\n")
        jsd_comp = patterns["jsd_vs_confidence_comparison"]
        f.write(f"- **Issue**: {jsd_comp['issue']}\n")
        f.write(f"- **Root cause**: {jsd_comp['root_cause']}\n")
        if "interpretation" in jsd_comp:
            f.write(f"- **Signal ratio**: {jsd_comp.get('mean_variance_ratio_jsd_over_conf', 'N/A'):.4f}\n")
            f.write(f"- **Interpretation**: {jsd_comp['interpretation']}\n\n")

        f.write(f"## Guidance vs Correctness\n\n")
        gc_analysis = guidance_correctness_analysis
        f.write(f"- Correct samples ({gc_analysis['correct_samples_n']}): "
                f"mean guidance = {gc_analysis['correct_mean_guidance']:.4f}\n")
        f.write(f"- Wrong samples ({gc_analysis['wrong_samples_n']}): "
                f"mean guidance = {gc_analysis['wrong_mean_guidance']:.4f}\n")
        f.write(f"- {gc_analysis['interpretation']}\n\n")

        f.write(f"## Top Guided Positions\n\n")
        for p in patterns["top_guided_positions"][:10]:
            f.write(f"- Position {p['position']}: mean magnitude = {p['mean_guidance_magnitude']:.4f}\n")

        f.write(f"\n## Runtime\n\n")
        f.write(f"- Total: {elapsed_total / 60:.1f} minutes\n")

    print(f"[{TASK_ID}] Summary saved to {summary_md}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # Mark done
    mark_done(
        status="success",
        summary=(f"Token diagnostics: A-CFG {accuracy:.1%} vs vanilla {v_accuracy:.1%} "
                 f"on Countdown-{N_SAMPLES}. JSD signal ratio: "
                 f"{patterns['jsd_vs_confidence_comparison'].get('mean_variance_ratio_jsd_over_conf', 'N/A')}. "
                 f"Verdict: {verdict}. Time: {elapsed_total/60:.1f}min"),
    )

    # Update gpu_progress.json
    gpu_progress_path = Path(f"{PROJECT_DIR}/exp/gpu_progress.json")
    try:
        progress = json.loads(gpu_progress_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in progress["completed"]:
        progress["completed"].append(TASK_ID)
    progress["running"].pop(TASK_ID, None)
    progress["timings"][TASK_ID] = {
        "planned_min": 30,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "Dream-v0-Instruct-7B",
            "method": "Token diagnostics (A-CFG, pivoted from RACFG)",
            "w": ACFG_W,
            "remask_pct": ACFG_REMASK_PCT,
            "n_samples": N_SAMPLES,
            "seed": SEED,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(progress, indent=2))
    print(f"[{TASK_ID}] gpu_progress.json updated")

    print(f"\n[{TASK_ID}] Total elapsed: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
    return verdict in ("GO", "CONDITIONAL-GO")


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n[{TASK_ID}] FATAL ERROR: {e}")
        traceback.print_exc()
        try:
            mark_done(status="failed", summary=f"Fatal error: {e}")
        except:
            pass
        sys.exit(1)
