"""
Compute-Fair Comparison — Countdown-16 (PILOT mode).

Fair comparison at equal compute budgets:
  (1) BSD (~1.1x) vs vanilla with 1.1x more steps (141 steps)
  (2) A-CFG (~2x) vs vanilla with 2x steps (256 steps)
  (3) BSD+ACFG (~2.1x) vs vanilla 2.1x steps (269 steps)
  (4) ReMDM-conf (~1.5x) vs vanilla 1.5x steps (192 steps)

Hypothesis H11: Methods outperform vanilla at equal FLOPs.
PILOT: 16 samples, seed 42.
"""

import os
import sys
import json
import time
import math
import random
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F

# ── Setup paths ──
REMOTE_BASE = "/home/ccwang/sibyl_system"
PROJECT_DIR = f"{REMOTE_BASE}/projects/ttt-dlm"
sys.path.insert(0, f"{PROJECT_DIR}/exp/code")

from bsd_racfg.eval_harness import (
    MASK_TOKEN_ID, MODEL_DIR,
    generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics,
    compute_per_sample_metrics, load_dream,
    vanilla_generate, save_results,
    check_degeneration,
    RESULTS_PILOTS, RESULTS_FULL,
)

from bsd_racfg.bsd import BSDConfig, bsd_generate
from bsd_racfg.combination_bsd_racfg import (
    BSDACFGComboConfig, bsd_acfg_combo_generate,
    dmi_generate, acfg_standalone_generate,
)


def remdm_conf_generate(
    model, tokenizer, prompt_text: str,
    device: str = "cuda:0", gen_len: int = 256,
    steps: int = 128, temperature: float = 0.4,
    remask_pct: float = 0.1,
) -> Tuple[str, float, Dict]:
    """ReMDM-conf: confidence-based remasking baseline.

    At each step, re-mask the lowest-confidence revealed tokens (remask_pct fraction),
    then do standard unmasking. ~1.5x FLOPs (extra softmax + re-mask overhead).
    """
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)
    prompt_len = len(prompt_ids)
    max_length = prompt_len + gen_len
    eps = 1e-3

    x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=MASK_TOKEN_ID)
    timesteps = torch.linspace(1, eps, steps + 1, device=device)

    t0 = time.time()
    with torch.no_grad():
        for i in range(steps):
            mask_index = (x == MASK_TOKEN_ID)
            n_masked = mask_index.sum().item()
            if n_masked == 0:
                break

            # Forward pass
            outputs = model(x, attention_mask="full", position_ids=None)
            logits = outputs.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

            # Confidence-based remasking
            probs = F.softmax(logits / temperature, dim=-1)
            confidence = probs.max(dim=-1).values  # [1, total_len]

            gen_conf = confidence[:, prompt_len:]
            gen_mask = (x[:, prompt_len:] == MASK_TOKEN_ID)
            revealed = (~gen_mask[0]).nonzero(as_tuple=True)[0]

            if len(revealed) > 0 and i < steps - 1:
                n_remask = max(1, int(len(revealed) * remask_pct))
                rev_conf = gen_conf[0, revealed]
                _, remask_idx = rev_conf.topk(min(n_remask, len(revealed)), largest=False)
                for idx in remask_idx:
                    actual_idx = prompt_len + revealed[idx].item()
                    x[0, actual_idx] = MASK_TOKEN_ID
                # Recount after remasking
                mask_index = (x == MASK_TOKEN_ID)

            # Standard unmasking
            mask_logits = logits[mask_index]
            t_val = timesteps[i]
            s_val = timesteps[i + 1]
            p_transfer = (1 - s_val / t_val).item() if i < steps - 1 else 1.0

            x0 = torch.full_like(x[mask_index], MASK_TOKEN_ID, device=device)
            transfer_mask = torch.rand(x0.shape, device=device) < p_transfer
            if temperature > 0:
                p = F.softmax(mask_logits[transfer_mask] / temperature, dim=-1)
                sampled_tokens = torch.multinomial(p, num_samples=1).squeeze(-1)
            else:
                sampled_tokens = mask_logits[transfer_mask].argmax(dim=-1)
            x0[transfer_mask] = sampled_tokens
            x[mask_index] = x0

    elapsed = time.time() - t0

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

    return text, elapsed, {"remask_pct": remask_pct}


def run_method(name, gen_fn, problems, n_samples):
    """Run a method on all problems and collect results."""
    texts, times, correct_count = [], [], 0
    per_sample = []
    for j, prob in enumerate(problems):
        prompt = format_countdown_prompt(prob)
        result = gen_fn(prompt)
        text, elapsed = result[0], result[1]
        ver = verify_countdown_answer(text, prob)
        texts.append(text)
        times.append(elapsed)
        is_correct = ver["is_correct"]
        if is_correct:
            correct_count += 1
        per_sample.append({
            "sample_idx": j,
            "target": prob["target"],
            "numbers": prob["numbers"],
            "is_correct": is_correct,
            "extracted_equation": ver.get("extracted_equation"),
            "generated_text": text[:300],
        })
        if j < 3:
            print(f"  [{j}] {'OK' if is_correct else 'NO'} | {text[:100]}")

    acc = correct_count / n_samples
    div = compute_diversity_metrics(texts)
    avg_time = float(np.mean(times))
    print(f"  {name}: {acc:.1%} ({correct_count}/{n_samples}), avg_time={avg_time:.2f}s")
    return {
        "accuracy": acc,
        "n_correct": correct_count,
        "n_samples": n_samples,
        **div,
        "avg_gen_time_s": avg_time,
        "per_sample": per_sample,
    }


def main():
    task_id = "compute_fair"
    n_samples = 16  # PILOT
    seed = 42
    device = "cuda:0"

    print("=" * 70)
    print(f"Compute-Fair Comparison PILOT — Countdown-{n_samples}")
    print("=" * 70)

    # Write PID file
    results_dir = RESULTS_FULL
    results_dir.mkdir(parents=True, exist_ok=True)
    pid_file = results_dir / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[PID] Written to {pid_file}")

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Load model
    model, tokenizer, embedding_layer = load_dream(device)

    # Generate problems
    problems = generate_countdown_problems(n_samples, seed=seed)
    print(f"Generated {len(problems)} Countdown problems")

    # Warmup: run a single forward pass to initialize CUDA context properly
    print("Warming up CUDA...")
    warmup_prompt = format_countdown_prompt(problems[0])
    _warmup_text, _warmup_t, _ = vanilla_generate(
        model, tokenizer, warmup_prompt, device, gen_len=32, steps=4, temperature=0.4
    )
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    print(f"  Warmup done ({_warmup_t:.2f}s)")

    # Re-seed after warmup
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # ── Define all methods ──
    # Standard step counts for compute-fair comparisons
    BASE_STEPS = 128
    GEN_LEN = 256
    TEMP = 0.4

    # FLOPs ratios and corresponding vanilla step counts
    flops_configs = {
        "vanilla_1x":    {"steps": BASE_STEPS,                "flops_ratio": 1.0,  "label": "Vanilla (128 steps, 1.0x)"},
        "vanilla_1.1x":  {"steps": int(BASE_STEPS * 1.1),     "flops_ratio": 1.1,  "label": "Vanilla (141 steps, 1.1x)"},
        "vanilla_1.5x":  {"steps": int(BASE_STEPS * 1.5),     "flops_ratio": 1.5,  "label": "Vanilla (192 steps, 1.5x)"},
        "vanilla_2.0x":  {"steps": int(BASE_STEPS * 2.0),     "flops_ratio": 2.0,  "label": "Vanilla (256 steps, 2.0x)"},
        "vanilla_2.1x":  {"steps": int(BASE_STEPS * 2.1),     "flops_ratio": 2.1,  "label": "Vanilla (269 steps, 2.1x)"},
    }

    all_results = {}
    t_total_start = time.time()

    # ── 1. Vanilla baselines at different step counts ──
    for key, cfg in flops_configs.items():
        steps = cfg["steps"]
        print(f"\n--- {cfg['label']} ---")

        def gen_fn(prompt, _steps=steps):
            return vanilla_generate(model, tokenizer, prompt, device,
                                    gen_len=GEN_LEN, steps=_steps, temperature=TEMP)

        all_results[key] = run_method(cfg["label"], gen_fn, problems, n_samples)
        all_results[key]["flops_ratio"] = cfg["flops_ratio"]
        all_results[key]["steps"] = steps

        # Report progress
        progress_file = results_dir / f"{task_id}_PROGRESS.json"
        progress_file.write_text(json.dumps({
            "task_id": task_id,
            "epoch": list(flops_configs.keys()).index(key) + 1,
            "total_epochs": 10,  # approximate total methods
            "metric": {"completed": key},
            "updated_at": datetime.now().isoformat(),
        }))

    # ── 2. BSD (~1.1x FLOPs) ──
    print(f"\n--- BSD (k=0.75, linear alpha, ~1.1x) ---")
    bsd_cfg = BSDConfig(
        k_frac=0.75, alpha_schedule="linear",
        alpha_start=0.1, alpha_end=0.8,
        tau_start=1.0, tau_end=0.3,
    )

    def gen_bsd(prompt):
        return bsd_generate(model, tokenizer, prompt, embedding_layer,
                            bsd_cfg, device)

    all_results["bsd"] = run_method("BSD (~1.1x)", gen_bsd, problems, n_samples)
    all_results["bsd"]["flops_ratio"] = 1.1
    all_results["bsd"]["method"] = "BSD"

    # ── 3. A-CFG (~2.0x FLOPs) ──
    print(f"\n--- A-CFG (w=1.5, ~2.0x) ---")

    def gen_acfg(prompt):
        return acfg_standalone_generate(model, tokenizer, prompt, device,
                                         w=1.5, remask_pct=0.1,
                                         gen_len=GEN_LEN, steps=BASE_STEPS,
                                         temperature=TEMP)

    all_results["acfg"] = run_method("A-CFG (~2.0x)", gen_acfg, problems, n_samples)
    all_results["acfg"]["flops_ratio"] = 2.0
    all_results["acfg"]["method"] = "A-CFG w=1.5"

    # ── 4. ReMDM-conf (~1.5x FLOPs) ──
    print(f"\n--- ReMDM-conf (~1.5x) ---")

    def gen_remdm(prompt):
        return remdm_conf_generate(model, tokenizer, prompt, device,
                                    gen_len=GEN_LEN, steps=BASE_STEPS,
                                    temperature=TEMP, remask_pct=0.1)

    all_results["remdm_conf"] = run_method("ReMDM-conf (~1.5x)", gen_remdm, problems, n_samples)
    all_results["remdm_conf"]["flops_ratio"] = 1.5
    all_results["remdm_conf"]["method"] = "ReMDM-conf"

    # ── 5. BSD + A-CFG combination (~2.1x FLOPs) ──
    print(f"\n--- BSD+ACFG Combo (~2.1x) ---")
    combo_cfg = BSDACFGComboConfig()

    def gen_combo(prompt):
        return bsd_acfg_combo_generate(model, tokenizer, prompt, embedding_layer,
                                        combo_cfg, device, track_entropy=False)

    all_results["bsd_acfg_combo"] = run_method("BSD+ACFG (~2.1x)", gen_combo, problems, n_samples)
    all_results["bsd_acfg_combo"]["flops_ratio"] = 2.1
    all_results["bsd_acfg_combo"]["method"] = "BSD+ACFG"

    # ── 6. DMI (~1.05x FLOPs) ──
    print(f"\n--- DMI (alpha=0.3, ~1.05x) ---")

    def gen_dmi(prompt):
        return dmi_generate(model, tokenizer, prompt, embedding_layer, device)

    all_results["dmi"] = run_method("DMI (~1.05x)", gen_dmi, problems, n_samples)
    all_results["dmi"]["flops_ratio"] = 1.05
    all_results["dmi"]["method"] = "DMI alpha=0.3"

    t_total_elapsed = time.time() - t_total_start

    # ── Compute-Fair Comparisons ──
    print(f"\n{'=' * 70}")
    print(f"COMPUTE-FAIR COMPARISON RESULTS")
    print(f"{'=' * 70}")

    comparisons = [
        {
            "method": "BSD (~1.1x)",
            "method_key": "bsd",
            "fair_baseline_key": "vanilla_1.1x",
            "flops": 1.1,
        },
        {
            "method": "ReMDM-conf (~1.5x)",
            "method_key": "remdm_conf",
            "fair_baseline_key": "vanilla_1.5x",
            "flops": 1.5,
        },
        {
            "method": "A-CFG (~2.0x)",
            "method_key": "acfg",
            "fair_baseline_key": "vanilla_2.0x",
            "flops": 2.0,
        },
        {
            "method": "BSD+ACFG (~2.1x)",
            "method_key": "bsd_acfg_combo",
            "fair_baseline_key": "vanilla_2.1x",
            "flops": 2.1,
        },
        {
            "method": "DMI (~1.05x)",
            "method_key": "dmi",
            "fair_baseline_key": "vanilla_1.1x",  # closest
            "flops": 1.05,
        },
    ]

    comparison_results = []
    print(f"\n{'Method':<25} {'Acc':>8} {'Fair Vanilla Acc':>18} {'Delta':>8} {'Wins':>6}")
    print("-" * 70)

    for comp in comparisons:
        method_acc = all_results[comp["method_key"]]["accuracy"]
        baseline_acc = all_results[comp["fair_baseline_key"]]["accuracy"]
        delta = method_acc - baseline_acc

        # Flip analysis
        method_ps = all_results[comp["method_key"]]["per_sample"]
        baseline_ps = all_results[comp["fair_baseline_key"]]["per_sample"]
        method_wins = sum(1 for m, b in zip(method_ps, baseline_ps)
                          if m["is_correct"] and not b["is_correct"])
        baseline_wins = sum(1 for m, b in zip(method_ps, baseline_ps)
                            if not m["is_correct"] and b["is_correct"])

        print(f"{comp['method']:<25} {method_acc:>7.1%} {baseline_acc:>17.1%} "
              f"{delta:>+7.1%} {method_wins:>3}-{baseline_wins}")

        comparison_results.append({
            "method": comp["method"],
            "method_accuracy": method_acc,
            "fair_baseline": comp["fair_baseline_key"],
            "fair_baseline_accuracy": baseline_acc,
            "delta_pp": round(delta * 100, 1),
            "flops_budget": comp["flops"],
            "method_wins": method_wins,
            "baseline_wins": baseline_wins,
            "method_outperforms": method_acc > baseline_acc,
        })

    # ── Pareto analysis ──
    print(f"\n--- Pareto Frontier (Accuracy vs Compute) ---")
    pareto_points = []
    for key, res in all_results.items():
        pareto_points.append({
            "method": key,
            "flops_ratio": res["flops_ratio"],
            "accuracy": res["accuracy"],
        })
    pareto_points.sort(key=lambda p: p["flops_ratio"])

    print(f"{'Method':<25} {'FLOPs':>8} {'Accuracy':>10}")
    print("-" * 45)
    for p in pareto_points:
        print(f"{p['method']:<25} {p['flops_ratio']:>7.1f}x {p['accuracy']:>9.1%}")

    # Determine Pareto-optimal points
    best_acc = -1
    pareto_optimal = []
    for p in pareto_points:
        if p["accuracy"] > best_acc:
            pareto_optimal.append(p["method"])
            best_acc = p["accuracy"]

    print(f"\nPareto-optimal methods: {pareto_optimal}")

    # ── Determine verdict ──
    # Pass if at least one method outperforms vanilla at equal FLOPs
    any_method_wins = any(c["method_outperforms"] for c in comparison_results)
    verdict = "GO" if any_method_wins else "NO-GO"

    print(f"\n{'=' * 70}")
    print(f"VERDICT: {verdict}")
    print(f"  At least one method outperforms fair vanilla: {any_method_wins}")
    print(f"{'=' * 70}")

    # ── Save results ──
    # Remove per_sample from all_results for the main output (too verbose)
    all_results_clean = {}
    for key, res in all_results.items():
        res_clean = {k: v for k, v in res.items() if k != "per_sample"}
        all_results_clean[key] = res_clean

    result = {
        "task_id": task_id,
        "mode": "PILOT",
        "verdict": verdict,
        "model": "Dream-v0-Instruct-7B",
        "benchmark": f"Countdown-{n_samples}",
        "seed": seed,
        "n_samples": n_samples,
        "timestamp": datetime.now().isoformat(),
        "elapsed_total_s": t_total_elapsed,
        "elapsed_total_min": round(t_total_elapsed / 60, 1),
        "configs": {
            "base_steps": BASE_STEPS,
            "gen_len": GEN_LEN,
            "temperature": TEMP,
            "bsd": {
                "k_frac": 0.75, "alpha_schedule": "linear(0.1->0.8)",
                "tau_schedule": "linear(1.0->0.3)",
            },
            "acfg": {"w": 1.5, "remask_pct": 0.1},
            "remdm_conf": {"remask_pct": 0.1},
            "dmi": {"alpha": 0.3, "soft_tau": 0.5},
            "bsd_acfg_combo": {
                "bsd_k_frac": 0.75, "bsd_alpha_schedule": "linear(0.1->0.8)",
                "acfg_w": 1.5, "acfg_remask_pct": 0.1,
            },
        },
        "results": all_results_clean,
        "compute_fair_comparisons": comparison_results,
        "pareto_points": pareto_points,
        "pareto_optimal": pareto_optimal,
    }

    out_path = results_dir / "compute_fair_countdown500.json"
    save_results(result, str(out_path))
    print(f"\nResults saved to {out_path}")

    # ── Write summary ──
    summary_path = results_dir / "compute_fair_summary.md"
    with open(summary_path, "w") as f:
        f.write(f"# Compute-Fair Comparison — Countdown-{n_samples} (PILOT)\n\n")
        f.write(f"**Verdict: {verdict}**\n\n")
        f.write(f"## Compute-Fair Head-to-Head\n\n")
        f.write(f"| Method | FLOPs | Accuracy | Fair Vanilla | Delta | Wins |\n")
        f.write(f"|--------|-------|----------|-------------|-------|------|\n")
        for comp in comparison_results:
            f.write(f"| {comp['method']} | {comp['flops_budget']}x | "
                    f"{comp['method_accuracy']:.1%} | "
                    f"{comp['fair_baseline_accuracy']:.1%} | "
                    f"{comp['delta_pp']:+.1f}pp | "
                    f"{comp['method_wins']}-{comp['baseline_wins']} |\n")

        f.write(f"\n## Pareto Frontier (Accuracy vs Compute)\n\n")
        f.write(f"| Method | FLOPs | Accuracy |\n")
        f.write(f"|--------|-------|----------|\n")
        for p in pareto_points:
            is_pareto = "*" if p["method"] in pareto_optimal else ""
            f.write(f"| {p['method']}{is_pareto} | {p['flops_ratio']:.1f}x | "
                    f"{p['accuracy']:.1%} |\n")
        f.write(f"\nPareto-optimal: {', '.join(pareto_optimal)}\n\n")

        f.write(f"## All Methods Summary\n\n")
        f.write(f"| Method | Accuracy | rep-2 | rep-3 | distinct-3 | Avg Time | FLOPs |\n")
        f.write(f"|--------|----------|-------|-------|------------|----------|-------|\n")
        for key in sorted(all_results.keys(), key=lambda k: all_results[k]["flops_ratio"]):
            res = all_results[key]
            f.write(f"| {key} | {res['accuracy']:.1%} ({res['n_correct']}/{n_samples}) | "
                    f"{res.get('rep_2', 0):.3f} | {res.get('rep_3', 0):.3f} | "
                    f"{res.get('distinct_3', 0):.3f} | {res['avg_gen_time_s']:.1f}s | "
                    f"{res['flops_ratio']:.1f}x |\n")

        f.write(f"\n## Runtime\n\n")
        f.write(f"- Total: {t_total_elapsed / 60:.1f} minutes\n")
    print(f"Summary saved to {summary_path}")

    # ── Write DONE marker ──
    done_marker = results_dir / f"{task_id}_DONE"
    done_marker.write_text(json.dumps({
        "task_id": task_id,
        "status": "success",
        "summary": f"Compute-fair comparison: verdict={verdict}, "
                   f"{sum(1 for c in comparison_results if c['method_outperforms'])}/{len(comparison_results)} methods beat fair vanilla",
        "final_progress": {
            "verdict": verdict,
            "n_methods_tested": len(comparison_results),
            "n_methods_win": sum(1 for c in comparison_results if c["method_outperforms"]),
        },
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"DONE marker written to {done_marker}")

    # Clean up PID file
    if pid_file.exists():
        pid_file.unlink()

    return result


if __name__ == "__main__":
    main()
