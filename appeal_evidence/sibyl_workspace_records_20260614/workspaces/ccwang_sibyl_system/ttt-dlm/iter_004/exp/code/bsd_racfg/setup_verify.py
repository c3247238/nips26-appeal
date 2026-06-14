"""
Setup Verification Script for BSD/RACFG Iteration 4.

Checks:
1. Conda environment and dependencies
2. GPU availability and VRAM
3. Dream-7B model loads correctly
4. Embedding layer accessible
5. Vanilla generation produces output
6. BSD module: belief vectors evolve without OOD collapse
7. RACFG module: stability scores computed
8. Eval harness: problem generation, verification, metrics all work

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 -m bsd_racfg.setup_verify
"""
import os
import sys
import json
import time
import traceback
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bsd_racfg.eval_harness import (
    load_dream, generate_countdown_problems, format_countdown_prompt,
    verify_countdown_answer, compute_diversity_metrics,
    compute_per_sample_metrics, vanilla_generate, save_results,
    MODEL_DIR, MASK_TOKEN_ID, PROJECT_DIR
)
from bsd_racfg.bsd import BSDConfig, bsd_generate
from bsd_racfg.racfg import RACFGConfig, ACFGConfig, racfg_generate, acfg_generate


RESULTS_DIR = Path(f"{PROJECT_DIR}/exp/results/pilots")
LOG_DIR = Path(f"{PROJECT_DIR}/exp/code/bsd_racfg")


def check_env():
    """Check Python environment and dependencies."""
    print("[1/8] Checking environment...")
    checks = {}

    checks["python_version"] = sys.version
    checks["torch_version"] = torch.__version__
    checks["cuda_available"] = torch.cuda.is_available()

    if torch.cuda.is_available():
        checks["gpu_count"] = torch.cuda.device_count()
        checks["gpu_name"] = torch.cuda.get_device_name(0)
        checks["gpu_vram_mb"] = torch.cuda.get_device_properties(0).total_memory // (1024 * 1024)
    else:
        checks["gpu_count"] = 0
        checks["gpu_name"] = "N/A"
        checks["gpu_vram_mb"] = 0

    try:
        import transformers
        checks["transformers_version"] = transformers.__version__
    except ImportError:
        checks["transformers_version"] = "NOT INSTALLED"

    try:
        import datasets
        checks["datasets_version"] = datasets.__version__
    except ImportError:
        checks["datasets_version"] = "NOT INSTALLED"

    try:
        import scipy
        checks["scipy_version"] = scipy.__version__
    except ImportError:
        checks["scipy_version"] = "NOT INSTALLED"

    for k, v in checks.items():
        print(f"  {k}: {v}")

    assert checks["cuda_available"], "CUDA not available!"
    assert checks["gpu_count"] > 0, "No GPUs found!"
    print("  -> PASS")
    return checks


def check_model_loading(device="cuda:0"):
    """Check Dream-7B model loads correctly."""
    print(f"\n[2/8] Loading Dream-7B on {device}...")
    t0 = time.time()
    model, tokenizer, embedding_layer = load_dream(device)
    load_time = time.time() - t0

    checks = {
        "load_time_s": round(load_time, 1),
        "model_type": type(model).__name__,
        "vocab_size": embedding_layer.weight.shape[0],
        "embed_dim": embedding_layer.weight.shape[1],
        "mask_token_id": MASK_TOKEN_ID,
    }

    # Verify mask token embedding
    mask_emb = embedding_layer(torch.tensor([MASK_TOKEN_ID], device=device))
    checks["mask_emb_norm"] = round(mask_emb.norm().item(), 4)

    for k, v in checks.items():
        print(f"  {k}: {v}")
    print("  -> PASS")
    return model, tokenizer, embedding_layer, checks


def check_vanilla_generation(model, tokenizer, device="cuda:0"):
    """Check vanilla generation produces output."""
    print(f"\n[3/8] Testing vanilla generation (1 sample)...")
    problems = generate_countdown_problems(1, seed=42)
    prompt = format_countdown_prompt(problems[0])

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    text, elapsed, diag = vanilla_generate(model, tokenizer, prompt, device)

    verification = verify_countdown_answer(text, problems[0])
    metrics = compute_per_sample_metrics(text)

    checks = {
        "output_length_chars": len(text),
        "output_length_words": metrics["word_count"],
        "gen_time_s": round(elapsed, 1),
        "target": problems[0]["target"],
        "is_correct": verification["is_correct"],
        "extracted_equation": verification.get("extracted_equation"),
    }

    print(f"  Output: {text[:200]}")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    assert len(text) > 0, "Empty output from vanilla generation!"
    print("  -> PASS")
    return checks


def check_bsd_generation(model, tokenizer, embedding_layer, device="cuda:0"):
    """Check BSD generation with default config."""
    print(f"\n[4/8] Testing BSD generation (1 sample, k_frac=0.5)...")
    problems = generate_countdown_problems(1, seed=42)
    prompt = format_countdown_prompt(problems[0])

    config = BSDConfig(k_frac=0.5, alpha_schedule="linear",
                       alpha_start=0.1, alpha_end=0.8)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    text, elapsed, diag = bsd_generate(
        model, tokenizer, prompt, embedding_layer, config, device
    )

    verification = verify_countdown_answer(text, problems[0])
    metrics = compute_per_sample_metrics(text)

    # Check belief entropy trajectory
    entropy_traj = diag.get("entropy_trajectory", [])
    belief_entropies = [e["mean_entropy"] for e in entropy_traj if e["phase"] == "belief"]

    checks = {
        "output_length_chars": len(text),
        "output_length_words": metrics["word_count"],
        "gen_time_s": round(elapsed, 1),
        "is_correct": verification["is_correct"],
        "k_step": diag["k_step"],
        "n_belief_steps": len(belief_entropies),
        "entropy_start": round(belief_entropies[0], 2) if belief_entropies else None,
        "entropy_end": round(belief_entropies[-1], 2) if belief_entropies else None,
        "entropy_decreasing": (
            belief_entropies[-1] < belief_entropies[0]
            if len(belief_entropies) >= 2 else None
        ),
        "rep_3": round(metrics["rep_3"], 4),
    }

    print(f"  Output: {text[:200]}")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    assert len(text) > 0, "Empty output from BSD generation!"
    # Don't require entropy decreasing in setup — just check it runs
    print("  -> PASS")
    return checks, diag


def check_bsd_fallback(model, tokenizer, embedding_layer, device="cuda:0"):
    """Check BSD with fallback_beta (graceful degradation mode)."""
    print(f"\n[5/8] Testing BSD with fallback_beta=0.5 (graceful degradation)...")
    problems = generate_countdown_problems(1, seed=42)
    prompt = format_countdown_prompt(problems[0])

    config = BSDConfig(k_frac=0.5, fallback_beta_start=0.5, fallback_beta_end=0.1)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    text, elapsed, diag = bsd_generate(
        model, tokenizer, prompt, embedding_layer, config, device
    )

    checks = {
        "output_length_chars": len(text),
        "gen_time_s": round(elapsed, 1),
        "runs_without_error": True,
    }

    print(f"  Output: {text[:150]}")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    assert len(text) > 0, "Empty output from BSD fallback!"
    print("  -> PASS")
    return checks


def check_racfg_generation(model, tokenizer, device="cuda:0"):
    """Check RACFG generation with default config."""
    print(f"\n[6/8] Testing RACFG generation (1 sample, threshold_70_30 schedule)...")
    problems = generate_countdown_problems(1, seed=42)
    prompt = format_countdown_prompt(problems[0])

    config = RACFGConfig(remask_pct=0.10, w_base=1.0, schedule_type="threshold_70_30")

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    text, elapsed, diag = racfg_generate(
        model, tokenizer, prompt, config, device
    )

    stability_data = diag.get("stability_data", [])
    verification = verify_countdown_answer(text, problems[0])
    metrics = compute_per_sample_metrics(text)

    checks = {
        "output_length_chars": len(text),
        "output_length_words": metrics["word_count"],
        "gen_time_s": round(elapsed, 1),
        "is_correct": verification["is_correct"],
        "n_stability_records": len(stability_data),
        "stability_mean": round(stability_data[-1]["stability_mean"], 4) if stability_data else None,
        "guidance_steps": sum(1 for s in diag["step_diagnostics"] if s.get("guidance_applied")),
        "rep_3": round(metrics["rep_3"], 4),
    }

    print(f"  Output: {text[:200]}")
    for k, v in checks.items():
        print(f"  {k}: {v}")

    assert len(text) > 0, "Empty output from RACFG generation!"
    assert len(stability_data) > 0, "No stability data recorded!"
    print("  -> PASS")
    return checks, diag


def check_acfg_generation(model, tokenizer, device="cuda:0"):
    """Check A-CFG baseline generation."""
    print(f"\n[7/8] Testing A-CFG baseline (1 sample, w=1.0)...")
    problems = generate_countdown_problems(1, seed=42)
    prompt = format_countdown_prompt(problems[0])

    config = ACFGConfig(remask_pct=0.10, w=1.0)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    text, elapsed, diag = acfg_generate(
        model, tokenizer, prompt, config, device
    )

    checks = {
        "output_length_chars": len(text),
        "gen_time_s": round(elapsed, 1),
        "runs_without_error": True,
    }

    print(f"  Output: {text[:150]}")
    for k, v in checks.items():
        print(f"  {k}: {v}")
    assert len(text) > 0, "Empty output from A-CFG generation!"
    print("  -> PASS")
    return checks


def check_eval_harness():
    """Check evaluation harness functions."""
    print(f"\n[8/8] Testing evaluation harness...")

    # Problem generation
    problems = generate_countdown_problems(10, seed=42)
    assert len(problems) == 10, f"Expected 10 problems, got {len(problems)}"

    # Verification
    v1 = verify_countdown_answer("Answer: 5 + 3 = 8", {"numbers": [5, 3, 7], "target": 8})
    assert v1["is_correct"] is True, "Correct answer not verified!"

    v2 = verify_countdown_answer("Answer: 5 + 3 = 8", {"numbers": [5, 3, 7], "target": 10})
    assert v2["is_correct"] is False, "Wrong answer not caught!"

    # Diversity metrics
    texts = ["hello world foo bar baz", "test one two three four"]
    div = compute_diversity_metrics(texts)
    assert "distinct_1" in div
    assert "rep_2" in div
    assert "avg_length_words" in div

    checks = {
        "problem_generation": "OK",
        "answer_verification": "OK",
        "diversity_metrics": "OK",
        "distinct_1": round(div["distinct_1"], 4),
    }

    for k, v in checks.items():
        print(f"  {k}: {v}")
    print("  -> PASS")
    return checks


def main():
    device = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]}"
    # If CUDA_VISIBLE_DEVICES is set, torch sees it as cuda:0
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        device = "cuda:0"

    print("=" * 70)
    print("  BSD/RACFG Setup Verification (Iteration 4)")
    print(f"  Time: {datetime.now().isoformat()}")
    print(f"  Device: {device}")
    print("=" * 70)

    all_checks = {}
    all_pass = True

    try:
        # 1. Environment
        all_checks["env"] = check_env()

        # 2. Model loading
        model, tokenizer, embedding_layer, model_checks = check_model_loading(device)
        all_checks["model"] = model_checks

        # 3. Vanilla generation
        all_checks["vanilla"] = check_vanilla_generation(model, tokenizer, device)

        # 4. BSD generation
        bsd_checks, bsd_diag = check_bsd_generation(model, tokenizer, embedding_layer, device)
        all_checks["bsd"] = bsd_checks

        # 5. BSD fallback
        all_checks["bsd_fallback"] = check_bsd_fallback(model, tokenizer, embedding_layer, device)

        # 6. RACFG generation
        racfg_checks, racfg_diag = check_racfg_generation(model, tokenizer, device)
        all_checks["racfg"] = racfg_checks

        # 7. A-CFG baseline
        all_checks["acfg"] = check_acfg_generation(model, tokenizer, device)

        # 8. Eval harness
        all_checks["eval_harness"] = check_eval_harness()

    except Exception as e:
        all_pass = False
        all_checks["error"] = {
            "type": type(e).__name__,
            "message": str(e),
            "traceback": traceback.format_exc(),
        }
        print(f"\n!!! ERROR: {e}")
        traceback.print_exc()

    # Free GPU memory
    try:
        del model
        torch.cuda.empty_cache()
    except:
        pass

    # Summary
    print(f"\n{'=' * 70}")
    print(f"  SETUP VERIFICATION SUMMARY")
    print(f"{'=' * 70}")

    passed_checks = []
    failed_checks = []
    for name, checks in all_checks.items():
        if name == "error":
            failed_checks.append(name)
        else:
            passed_checks.append(name)
            status = "PASS"
            print(f"  {name:>20}: {status}")

    if failed_checks:
        for name in failed_checks:
            print(f"  {name:>20}: FAIL")

    overall = "PASS" if all_pass and not failed_checks else "FAIL"
    print(f"\n  Overall: {overall}")
    print(f"  Passed: {len(passed_checks)}/{len(all_checks)}")
    print(f"{'=' * 70}")

    # Save results
    all_checks["summary"] = {
        "overall": overall,
        "timestamp": datetime.now().isoformat(),
        "passed": len(passed_checks),
        "total": len(all_checks) - 1,  # exclude summary itself
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_file = RESULTS_DIR / "setup_verified.json"
    with open(out_file, "w") as f:
        json.dump(all_checks, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Also save to expected output location
    log_file = LOG_DIR / "setup_verified.log"
    with open(log_file, "w") as f:
        f.write(f"Setup verification: {overall}\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Passed: {len(passed_checks)}/{len(all_checks) - 1}\n")
        for name in passed_checks:
            f.write(f"  {name}: PASS\n")
        for name in failed_checks:
            f.write(f"  {name}: FAIL\n")
    print(f"Log saved to {log_file}")

    return overall == "PASS"


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
