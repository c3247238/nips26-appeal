#!/usr/bin/env python3
"""
G1 Saturation Experiment: Validate Arrhenius Kinetics in LLM Reasoning

Tests H1: P_k(correct) follows exponential saturation: P_k = P_∞(1-exp(-k/k₀))

Sample each MATH problem k=1,2,4,8,16 times and fit Arrhenius model.
"""

import json
import os
import gc
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots" / "g1_saturation"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DATASET_NAME = "HuggingFaceH4/MATH"
PILOT_SAMPLES = 30  # Reduced for pilot (100 * 5 k_values = 500 steps)
SEED = 42
K_VALUES = [1, 2, 4, 8, 16]
TEMPERATURE = 0.7
MAX_TOKENS = 1024
DEVICE = "cuda"

TASK_ID = "g1_saturation"


def setup_logging():
    """Setup results directory."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Write PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[PID] {os.getpid()}")


def report_progress(epoch=0, total=1, step=0, total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    """Write DONE marker file."""
    done_file = RESULTS_DIR / f"{TASK_ID}_DONE"

    # Read progress if exists
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass

    done_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

    # Cleanup PID file
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()


def load_model():
    """Load Qwen2.5-Math-7B model."""
    print(f"[MODEL] Loading {MODEL_NAME}...")
    start = time.time()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"[MODEL] Loaded in {time.time() - start:.1f}s")
    return model, tokenizer


def load_dataset_subset():
    """Load MATH test subset."""
    print(f"[DATA] Loading MATH dataset...")
    dataset = load_dataset(DATASET_NAME, split="test")

    # Set seed and take subset
    dataset = dataset.shuffle(seed=SEED).select(range(PILOT_SAMPLES))

    print(f"[DATA] Loaded {len(dataset)} problems")
    return dataset


def extract_answer(text: str) -> str:
    """Extract final answer from model output."""
    # Try to find boxed answer first
    if "\\boxed{" in text:
        start = text.rfind("\\boxed{") + len("\\boxed{")
        end = text.find("}", start)
        if end > start:
            return text[start:end].strip()

    # Try \final{ or \answer{
    for marker in ["\\final{", "\\answer{", "\\result{"]:
        if marker in text:
            start = text.rfind(marker) + len(marker)
            end = text.find("}", start)
            if end > start:
                return text[start:end].strip()

    # Fallback: take last line
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith("```"):
            return line

    return text.strip()[-100:]  # Last 100 chars


def normalize_answer(answer: str) -> str:
    """Normalize answer for comparison."""
    import re
    # Remove LaTeX commands
    answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', answer)
    answer = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', answer)
    answer = re.sub(r'\\cdot', '*', answer)
    answer = re.sub(r'\\times', '*', answer)
    answer = re.sub(r'[{}]', '', answer)
    answer = re.sub(r'\\', '', answer)
    # Normalize whitespace
    answer = ' '.join(answer.split())
    return answer.lower().strip()


def sample_once(problem: str, model, tokenizer, k: int) -> Tuple[bool, int, List[str]]:
    """Generate k samples for one problem. Returns (correct, tokens, answers)."""
    prompt = f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    all_answers = []
    total_tokens = 0

    for _ in range(k):
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response.split("assistant\n")[-1].strip()

        answer = extract_answer(response)
        all_answers.append(answer)
        total_tokens += len(outputs[0]) - len(inputs["input_ids"][0])

        # Reset input for next sample (use same prompt)
        inputs = {k: v[:, :inputs["input_ids"].shape[1]] for k, v in inputs.items()}

    return total_tokens, all_answers


def fit_arrhenius_model(k_values: List[int], p_values: List[float]) -> Dict[str, float]:
    """Fit Arrhenius model P_k = P_inf * (1 - exp(-k / k0))."""
    from scipy.optimize import curve_fit
    from scipy.stats import pearsonr

    def arrhenius(k, p_inf, k0):
        return p_inf * (1 - np.exp(-np.array(k) / k0))

    # Initial guesses
    p0 = [max(p_values) * 1.1, 5.0]

    try:
        popt, pcov = curve_fit(arrhenius, k_values, p_values, p0=p0, maxfev=5000)
        p_inf, k0 = popt

        # Calculate R²
        predicted = arrhenius(k_values, p_inf, k0)
        ss_res = np.sum((np.array(p_values) - predicted) ** 2)
        ss_tot = np.sum((np.array(p_values) - np.mean(p_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Calculate alternative fit scores (power law, logarithmic)
        # Power law: P_k = a * k^b
        try:
            def power_law(k, a, b):
                return a * np.power(np.array(k) + 1, b)
            popt_power, _ = curve_fit(power_law, k_values, p_values, p0=[0.5, 0.3], maxfev=5000)
            predicted_power = power_law(k_values, *popt_power)
            ss_res_power = np.sum((np.array(p_values) - predicted_power) ** 2)
            r_squared_power = 1 - (ss_res_power / ss_tot) if ss_tot > 0 else 0
        except:
            r_squared_power = 0

        # Logarithmic: P_k = a * log(k + 1) + b
        try:
            def log_func(k, a, b):
                return a * np.log(np.array(k) + 1) + b
            popt_log, _ = curve_fit(log_func, k_values, p_values, p0=[0.2, 0.3], maxfev=5000)
            predicted_log = log_func(k_values, *popt_log)
            ss_res_log = np.sum((np.array(p_values) - predicted_log) ** 2)
            r_squared_log = 1 - (ss_res_log / ss_tot) if ss_tot > 0 else 0
        except:
            r_squared_log = 0

        return {
            "p_inf": float(p_inf),
            "k0": float(k0),
            "r_squared": float(r_squared),
            "r_squared_power_law": float(r_squared_power),
            "r_squared_log": float(r_squared_log),
            "fit_success": True,
            "best_model": "arrhenius" if r_squared >= r_squared_power and r_squared >= r_squared_log else (
                "power_law" if r_squared_power > r_squared_log else "logarithmic"
            )
        }
    except Exception as e:
        print(f"[FIT] Arrhenius fit failed: {e}")
        return {
            "p_inf": np.mean(p_values),
            "k0": 5.0,
            "r_squared": 0,
            "r_squared_power_law": 0,
            "r_squared_log": 0,
            "fit_success": False,
            "error": str(e)
        }


def run_experiment(model, tokenizer, dataset):
    """Run saturation experiment."""
    results = {
        "task_id": TASK_ID,
        "k_values": K_VALUES,
        "problems": [],
        "aggregate": {},
        "per_problem_fits": [],
        "start_time": datetime.now().isoformat(),
    }

    total_problems = len(dataset)
    total_samples = total_problems * len(K_VALUES)

    print(f"[EXP] Running saturation experiment: {total_problems} problems × {len(K_VALUES)} k values = {total_samples} samples")

    # Track accuracy for each k
    k_correct = {k: 0 for k in K_VALUES}
    k_tokens = {k: 0 for k in K_VALUES}

    # Per-problem tracking
    problem_results = []

    for idx, item in enumerate(dataset):
        problem = item["problem"]
        level = item.get("level", "Unknown")
        type_ = item.get("type", "Unknown")

        # Ground truth
        answer = normalize_answer(item["solution"])

        problem_data = {
            "idx": idx,
            "problem": problem[:200] + "..." if len(problem) > 200 else problem,
            "level": level,
            "type": type_,
            "ground_truth": answer,
            "k_results": {},
            "saturation_fit": None,
        }

        tokens_for_problem = 0

        for k in K_VALUES:
            num_tokens, answers = sample_once(problem, model, tokenizer, k)

            # Check if any answer is correct
            correct_count = sum(1 for ans in answers if normalize_answer(ans) == answer)
            is_correct = correct_count > 0

            problem_data["k_results"][k] = {
                "correct": is_correct,
                "correct_count": correct_count,
                "total_samples": k,
                "answers": answers,
                "tokens": num_tokens,
            }

            k_correct[k] += 1 if is_correct else 0
            k_tokens[k] += num_tokens
            tokens_for_problem += num_tokens

            # Update progress
            step = idx * len(K_VALUES) + K_VALUES.index(k) + 1
            report_progress(step=step, total_steps=total_samples, metric={"accuracy": k_correct[k] / (idx + 1)})

        # Fit Arrhenius model for this problem
        p_values = [1 if problem_data["k_results"][k]["correct"] else 0 for k in K_VALUES]
        if sum(p_values) > 0 and sum(p_values) < len(K_VALUES):  # Only fit if non-trivial
            fit_result = fit_arrhenius_model(K_VALUES, p_values)
            problem_data["saturation_fit"] = fit_result

        problem_results.append(problem_data)

        # Print progress
        if (idx + 1) % 10 == 0:
            print(f"[PROGRESS] {idx + 1}/{total_problems} problems done, tokens: {tokens_for_problem}")

        # Occasional cleanup
        if (idx + 1) % 20 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    results["problems"] = problem_results

    # Calculate aggregate accuracy for each k
    aggregate = {}
    for k in K_VALUES:
        aggregate[k] = {
            "accuracy": k_correct[k] / total_problems,
            "total_correct": k_correct[k],
            "total_problems": total_problems,
            "avg_tokens": k_tokens[k] / total_problems,
        }
    results["aggregate"] = aggregate

    # Fit Arrhenius model on aggregate data
    p_agg = [aggregate[k]["accuracy"] for k in K_VALUES]
    agg_fit = fit_arrhenius_model(K_VALUES, p_agg)
    results["aggregate_fit"] = agg_fit

    # Aggregate per-problem fit statistics
    valid_fits = [p["saturation_fit"] for p in problem_results if p["saturation_fit"] and p["saturation_fit"].get("fit_success")]
    if valid_fits:
        results["fit_statistics"] = {
            "valid_fits": len(valid_fits),
            "mean_r_squared": np.mean([f["r_squared"] for f in valid_fits]),
            "median_r_squared": np.median([f["r_squared"] for f in valid_fits]),
            "mean_p_inf": np.mean([f["p_inf"] for f in valid_fits]),
            "mean_k0": np.mean([f["k0"] for f in valid_fits]),
            "r_squared_histogram": {
                "0.0-0.5": sum(1 for f in valid_fits if f["r_squared"] < 0.5),
                "0.5-0.7": sum(1 for f in valid_fits if 0.5 <= f["r_squared"] < 0.7),
                "0.7-0.85": sum(1 for f in valid_fits if 0.7 <= f["r_squared"] < 0.85),
                "0.85-0.95": sum(1 for f in valid_fits if 0.85 <= f["r_squared"] < 0.95),
                "0.95-1.0": sum(1 for f in valid_fits if f["r_squared"] >= 0.95),
            },
            "best_model_distribution": {
                "arrhenius": sum(1 for f in valid_fits if f.get("best_model") == "arrhenius"),
                "power_law": sum(1 for f in valid_fits if f.get("best_model") == "power_law"),
                "logarithmic": sum(1 for f in valid_fits if f.get("best_model") == "logarithmic"),
            }
        }

    results["end_time"] = datetime.now().isoformat()

    return results


def analyze_results(results: Dict) -> Dict:
    """Analyze and summarize experiment results."""
    analysis = {
        "task_id": TASK_ID,
        "h1_status": "UNKNOWN",
        "h1_pass": False,
        "r_squared_threshold": 0.85,
        "actual_r_squared": 0,
        "pass_criteria": "Exponential fit R² > 0.85",
        "summary": {},
        "recommendation": "",
    }

    # Check aggregate fit
    agg_fit = results.get("aggregate_fit", {})
    r_squared = agg_fit.get("r_squared", 0)
    p_inf = agg_fit.get("p_inf", 0)
    k0 = agg_fit.get("k0", 0)

    analysis["actual_r_squared"] = r_squared
    analysis["p_inf_estimate"] = p_inf
    analysis["k0_estimate"] = k0

    # H1 evaluation
    if r_squared >= 0.85:
        analysis["h1_status"] = "CONFIRMED"
        analysis["h1_pass"] = True
        analysis["h1_notes"] = f"Arrhenius kinetics confirmed with R²={r_squared:.3f}"
    elif r_squared >= 0.7:
        analysis["h1_status"] = "PARTIAL"
        analysis["h1_notes"] = f"Weak Arrhenius fit with R²={r_squared:.3f}, alternative models may fit better"
    else:
        analysis["h1_status"] = "REJECTED"
        analysis["h1_notes"] = f"Arrhenius kinetics rejected with R²={r_squared:.3f}"

    # Check alternative model comparison
    if "fit_statistics" in results:
        stats = results["fit_statistics"]
        dist = stats.get("best_model_distribution", {})
        total_valid = sum(dist.values()) if dist else 0
        if total_valid > 0:
            arrhenius_pct = dist.get("arrhenius", 0) / total_valid * 100
            analysis["arrhenius_adoption_rate"] = arrhenius_pct
            if arrhenius_pct > 50:
                analysis["h1_notes"] += f" ({arrhenius_pct:.1f}% of fits prefer Arrhenius)"

    # Aggregate accuracy table
    agg = results.get("aggregate", {})
    accuracy_table = {f"k={k}": f"{agg[k]['accuracy']:.3f}" for k in K_VALUES if k in agg}
    analysis["accuracy_by_k"] = accuracy_table

    # Summary
    if analysis["h1_pass"]:
        analysis["recommendation"] = "GO - H1 confirmed: Arrhenius kinetics validated"
    elif r_squared >= 0.7:
        analysis["recommendation"] = "REFINE - Weak Arrhenius fit, consider alternative models"
    else:
        analysis["recommendation"] = "NO_GO - Arrhenius form rejected"

    return analysis


def save_results(results: Dict, analysis: Dict):
    """Save results to files."""
    # Main results JSON
    results_file = RESULTS_DIR / f"{TASK_ID}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVE] Results saved to {results_file}")

    # Analysis JSON
    analysis_file = RESULTS_DIR / f"{TASK_ID}_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[SAVE] Analysis saved to {analysis_file}")

    # Summary markdown
    summary_md = f"""# G1 Saturation Experiment Results

## Task: {TASK_ID}
## Date: {datetime.now().isoformat()}

## H1 Evaluation

**Status**: {analysis['h1_status']}
**R² (threshold > 0.85)**: {analysis['actual_r_squared']:.4f}
**P_∞ estimate**: {analysis.get('p_inf_estimate', 'N/A'):.3f}
**k₀ estimate**: {analysis.get('k0_estimate', 'N/A'):.3f}

**Pass**: {'YES' if analysis['h1_pass'] else 'NO'}

### Notes
{analysis.get('h1_notes', 'N/A')}

## Accuracy by k

| k | Accuracy |
|---|----------|
{chr(10).join([f"| {k} | {acc} |" for k, acc in analysis.get('accuracy_by_k', {}).items()])}

## Recommendation
{analysis['recommendation']}

## Fit Statistics
"""
    if "fit_statistics" in results:
        stats = results["fit_statistics"]
        summary_md += f"""
- Valid fits: {stats.get('valid_fits', 0)}
- Mean R²: {stats.get('mean_r_squared', 0):.3f}
- Median R²: {stats.get('median_r_squared', 0):.3f}
- Mean k₀: {stats.get('mean_k0', 0):.3f}

### R² Distribution
"""
        for bin_, count in stats.get('r_squared_histogram', {}).items():
            summary_md += f"- {bin_}: {count}\n"

    summary_file = RESULTS_DIR / f"{TASK_ID}_summary.md"
    with open(summary_file, "w") as f:
        f.write(summary_md)
    print(f"[SAVE] Summary saved to {summary_file}")

    return results_file, analysis_file


def main():
    """Main entry point."""
    print("=" * 60)
    print(f"G1 SATURATION EXPERIMENT - Arrhenius Kinetics Validation")
    print("=" * 60)

    setup_logging()

    try:
        # Load model and data
        model, tokenizer = load_model()
        dataset = load_dataset_subset()

        # Run experiment
        print("\n[RUN] Starting saturation experiment...")
        results = run_experiment(model, tokenizer, dataset)

        # Analyze results
        print("\n[ANALYZE] Analyzing results...")
        analysis = analyze_results(results)

        # Save results
        print("\n[SAVE] Saving results...")
        save_results(results, analysis)

        # Mark done
        status = "success" if analysis["h1_pass"] else "partial"
        summary = f"R²={analysis['actual_r_squared']:.3f}, H1={analysis['h1_status']}"
        mark_done(status=status, summary=summary)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print(f"H1 Status: {analysis['h1_status']}")
        print(f"R²: {analysis['actual_r_squared']:.4f}")
        print(f"Recommendation: {analysis['recommendation']}")
        print("=" * 60)

    except Exception as e:
        print(f"[ERROR] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        raise


if __name__ == "__main__":
    main()
