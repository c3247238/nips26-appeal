#!/usr/bin/env python3
"""
G1 Saturation Experiment v2: Validate Arrhenius Kinetics in LLM Reasoning

Tests H1: P_k(correct) follows exponential saturation: P_k = P_∞(1-exp(-k/k₀))

Key fixes from v1:
- Improved answer extraction with multiple fallback strategies
- System prompt to encourage boxed answers
- Numeric comparison for numerical answers
- Larger max_tokens to avoid truncation
"""

import json
import os
import gc
import re
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from scipy.optimize import curve_fit

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots" / "g1_saturation_v2"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
DATASET_NAME = "HuggingFaceH4/MATH"
PILOT_SAMPLES = 30  # Reduced for pilot (30 * 5 k_values = 150 samples)
SEED = 42
K_VALUES = [1, 2, 4, 8, 16]
TEMPERATURE = 0.7
MAX_TOKENS = 2048  # Increased from 1024 to avoid truncation
DEVICE = "cuda"

TASK_ID = "g1_saturation_v2"


def setup_logging():
    """Setup results directory and write PID file."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"[PID] {os.getpid()}")


def report_progress(step=0, total_steps=0, metric=None):
    """Write progress file for system monitor."""
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    """Write DONE marker file."""
    done_file = RESULTS_DIR / f"{TASK_ID}_DONE"

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
    dataset = dataset.shuffle(seed=SEED).select(range(PILOT_SAMPLES))
    print(f"[DATA] Loaded {len(dataset)} problems")
    return dataset


def extract_boxed_answer(text: str) -> Optional[str]:
    """Extract answer from \\boxed{} LaTeX command."""
    if "\\boxed{" not in text:
        return None

    # Find the last \boxed{} in the text (model usually puts answer there)
    start = text.rfind("\\boxed{") + len("\\boxed{")
    end = text.find("}", start)

    if end > start:
        return text[start:end].strip()

    return None


def extract_final_answer(text: str) -> Optional[str]:
    """Extract answer using various markers."""
    # Try various answer markers
    markers = [
        ("final", r"\\final\{([^}]+)\}"),
        ("answer", r"\\answer\{([^}]+)\}"),
        ("result", r"\\result\{([^}]+)\}"),
        ("Final Answer:", r"Final Answer:\s*(.+?)(?:\n|$)"),
        ("Therefore, the answer is", r"Therefore, the answer is\s+(.+?)(?:\n|$)"),
        ("the answer is", r"the answer is\s+(.+?)(?:\n\.\s|$)"),
    ]

    for name, pattern in markers:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()

    return None


def extract_last_line(text: str) -> Optional[str]:
    """Extract the last non-empty line."""
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    return None


def normalize_latex(answer: str) -> str:
    """Normalize LaTeX answer for comparison."""
    # Remove common LaTeX commands
    answer = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'(\1)/(\2)', answer)
    answer = re.sub(r'\\sqrt\{([^}]*)\}', r'sqrt(\1)', answer)
    answer = re.sub(r'\^{([^}]*)\}', r'^\1', answer)
    answer = re.sub(r'\\cdot', '*', answer)
    answer = re.sub(r'\\times', '*', answer)
    answer = re.sub(r'\\div', '/', answer)
    answer = re.sub(r'\\pi', 'pi', answer)
    answer = re.sub(r'\\theta', 'theta', answer)
    answer = re.sub(r'\\alpha', 'alpha', answer)
    answer = re.sub(r'\\beta', 'beta', answer)
    answer = re.sub(r'\\gamma', 'gamma', answer)
    answer = re.sub(r'\\sin', 'sin', answer)
    answer = re.sub(r'\\cos', 'cos', answer)
    answer = re.sub(r'\\tan', 'tan', answer)
    answer = re.sub(r'\\log', 'log', answer)
    answer = re.sub(r'\\ln', 'ln', answer)
    answer = re.sub(r'\\int', 'int', answer)
    answer = re.sub(r'\\sum', 'sum', answer)
    answer = re.sub(r'\\prod', 'prod', answer)
    answer = re.sub(r'\\lim', 'lim', answer)
    answer = re.sub(r'\\infty', 'infinity', answer)
    answer = re.sub(r'\\pm', '+-', answer)
    answer = re.sub(r'\\mp', '-+', answer)
    answer = re.sub(r'\\leq', '<=', answer)
    answer = re.sub(r'\\geq', '>=', answer)
    answer = re.sub(r'\\neq', '!=', answer)
    answer = re.sub(r'\\approx', '~', answer)
    answer = re.sub(r'\\dots', '...', answer)
    answer = re.sub(r'\\cdots', '...', answer)
    answer = re.sub(r'\\vdots', '|', answer)
    answer = re.sub(r'\\ddots', 'D', answer)
    answer = re.sub(r'\\vec\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\hat\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\bar\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\dot\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\text\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathbf\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathbb\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\mathcal\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\\rm\{([^}]+)\}', r'\1', answer)
    answer = re.sub(r'\{([^{}]+)\}', r'\1', answer)  # Remove remaining braces
    answer = re.sub(r'\\', '', answer)  # Remove remaining backslashes
    answer = re.sub(r'\$', '', answer)  # Remove dollar signs

    # Normalize whitespace
    answer = ' '.join(answer.split())

    return answer.lower().strip()


def normalize_numeric(answer: str) -> str:
    """Normalize numeric answer for comparison."""
    # Extract numeric parts
    answer = normalize_latex(answer)

    # Try to extract a clean number
    # Handle things like "1/2", "3.14", "-5", "sqrt(2)", etc.
    answer = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1/\2', answer)  # Keep fractions
    answer = re.sub(r'sqrt\((\d+)\)', r'sqrt(\1)', answer)
    answer = re.sub(r'pi', '3.14159', answer)  # Replace pi with numeric value

    # Remove any remaining non-alphanumeric except basic math operators
    answer = re.sub(r'[^a-z0-9+\-*/().=<>^_\s]', '', answer, flags=re.IGNORECASE)
    answer = ' '.join(answer.split())

    return answer.lower().strip()


def try_numeric_compare(extracted: str, ground_truth: str) -> bool:
    """Try to compare numeric answers."""
    try:
        # Extract numbers from both answers
        extracted_nums = re.findall(r'-?\d+\.?\d*', extracted)
        gt_nums = re.findall(r'-?\d+\.?\d*', ground_truth)

        if not extracted_nums or not gt_nums:
            return False

        # Compare first significant numbers
        for en in extracted_nums:
            for gn in gt_nums:
                if abs(float(en) - float(gn)) < 1e-6:
                    return True

        return False
    except:
        return False


def check_text_match(extracted: str, ground_truth: str) -> bool:
    """Check if extracted answer matches ground truth."""
    if not extracted:
        return False

    # Direct string match
    if extracted.strip() == ground_truth.strip():
        return True

    # Normalized LaTeX comparison
    norm_extracted = normalize_latex(extracted)
    norm_gt = normalize_latex(ground_truth)

    if norm_extracted == norm_gt:
        return True

    # Numeric comparison
    if try_numeric_compare(extracted, ground_truth):
        return True

    # Check if extracted is contained in ground truth or vice versa
    if norm_gt in norm_extracted or norm_extracted in norm_gt:
        # But only if it's not trivially small
        if len(norm_extracted) >= 2 and len(norm_gt) >= 2:
            return True

    return False


def extract_answer(text: str) -> Tuple[str, str]:
    """
    Extract final answer from model output.
    Returns (extracted_answer, method_used)
    """
    # Method 1: \boxed{} (most reliable)
    boxed = extract_boxed_answer(text)
    if boxed:
        return boxed, "boxed"

    # Method 2: Other markers
    final = extract_final_answer(text)
    if final:
        return final, "marker"

    # Method 3: Try to find answer after "=" or "=>"
    lines = text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        # Skip empty lines and lines that are clearly part of reasoning
        if not line or line.startswith("```") or "step" in line.lower():
            continue
        # Look for lines with = at the end or containing the answer pattern
        if "=" in line and len(line) < 200:
            # Extract right side of =
            match = re.search(r'=\s*(.+?)(?:\s*$)', line)
            if match:
                answer = match.group(1).strip()
                if answer and len(answer) >= 1:
                    return answer, "equals"

    # Method 4: Last non-empty line
    last = extract_last_line(text)
    if last:
        return last, "last_line"

    # Method 5: Last 200 chars
    return text.strip()[-200:], "fallback"


def sample_once(problem: str, model, tokenizer, k: int) -> Tuple[int, List[str]]:
    """Generate k samples for one problem. Returns (total_tokens, list_of_answers)."""
    # Use a system prompt that encourages boxed answers
    messages = [
        {"role": "system", "content": "You are a math expert. Provide clear step-by-step solutions and put your final answer in \\boxed{}. Example: \\boxed{42}"},
        {"role": "user", "content": problem},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
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

        # Extract just the assistant's response
        if "assistant" in response:
            response = response.split("assistant")[-1].strip()
        elif "<|im_end|>" in response:
            response = response.split("<|im_end|>")[-1].strip()

        answer, method = extract_answer(response)
        all_answers.append(answer)
        total_tokens += len(outputs[0]) - len(inputs["input_ids"][0])

        # Reset input for next sample
        inputs = {key: val[:, :inputs["input_ids"].shape[1]] for key, val in inputs.items()}

    return total_tokens, all_answers


def fit_arrhenius_model(k_values: List[int], p_values: List[float]) -> Dict[str, Any]:
    """Fit Arrhenius model P_k = P_inf * (1 - exp(-k / k0))."""
    def arrhenius(k, p_inf, k0):
        return p_inf * (1 - np.exp(-np.array(k, dtype=float) / k0))

    # Initial guesses
    p0 = [max(p_values) * 1.1 if max(p_values) > 0 else 0.5, 5.0]

    result = {
        "fit_success": False,
        "p_inf": float(np.mean(p_values)),
        "k0": 5.0,
        "r_squared": 0.0,
        "r_squared_power_law": 0.0,
        "r_squared_log": 0.0,
        "best_model": "none",
    }

    try:
        popt, pcov = curve_fit(arrhenius, k_values, p_values, p0=p0, maxfev=5000)
        p_inf, k0 = popt

        predicted = arrhenius(k_values, p_inf, k0)
        ss_res = np.sum((np.array(p_values) - predicted) ** 2)
        ss_tot = np.sum((np.array(p_values) - np.mean(p_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        result["p_inf"] = float(p_inf)
        result["k0"] = float(k0)
        result["r_squared"] = float(r_squared)
        result["fit_success"] = True

    except Exception as e:
        print(f"[FIT] Arrhenius fit failed: {e}")

    # Try power law: P_k = a * k^b
    try:
        def power_law(k, a, b):
            return a * np.power(np.array(k, dtype=float), b)
        popt_power, _ = curve_fit(power_law, k_values, p_values, p0=[0.5, 0.3], maxfev=5000)
        predicted_power = power_law(k_values, *popt_power)
        ss_res_power = np.sum((np.array(p_values) - predicted_power) ** 2)
        r_squared_power = 1 - (ss_res_power / ss_tot) if ss_tot > 0 else 0
        result["r_squared_power_law"] = float(r_squared_power)
    except:
        pass

    # Try logarithmic: P_k = a * log(k + 1) + b
    try:
        def log_func(k, a, b):
            return a * np.log(np.array(k, dtype=float) + 1) + b
        popt_log, _ = curve_fit(log_func, k_values, p_values, p0=[0.2, 0.3], maxfev=5000)
        predicted_log = log_func(k_values, *popt_log)
        ss_res_log = np.sum((np.array(p_values) - predicted_log) ** 2)
        r_squared_log = 1 - (ss_res_log / ss_tot) if ss_tot > 0 else 0
        result["r_squared_log"] = float(r_squared_log)
    except:
        pass

    # Determine best model
    if result["r_squared"] >= result["r_squared_power_law"] and result["r_squared"] >= result["r_squared_log"]:
        result["best_model"] = "arrhenius"
    elif result["r_squared_power_law"] > result["r_squared_log"]:
        result["best_model"] = "power_law"
    else:
        result["best_model"] = "logarithmic"

    return result


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

    print(f"[EXP] Running saturation experiment v2: {total_problems} problems × {len(K_VALUES)} k values = {total_samples} samples")

    k_correct = {k: 0 for k in K_VALUES}
    k_tokens = {k: 0 for k in K_VALUES}

    problem_results = []

    for idx, item in enumerate(dataset):
        problem = item["problem"]
        level = item.get("level", "Unknown")
        type_ = item.get("type", "Unknown")
        ground_truth = item["solution"]

        problem_data = {
            "idx": idx,
            "problem": problem[:200] + "..." if len(problem) > 200 else problem,
            "level": level,
            "type": type_,
            "ground_truth": ground_truth,
            "ground_truth_normalized": normalize_latex(ground_truth),
            "k_results": {},
            "saturation_fit": None,
        }

        tokens_for_problem = 0

        for k in K_VALUES:
            num_tokens, answers = sample_once(problem, model, tokenizer, k)

            # Check if any answer is correct
            correct_count = sum(1 for ans in answers if check_text_match(ans, ground_truth))
            is_correct = correct_count > 0

            problem_data["k_results"][k] = {
                "correct": is_correct,
                "correct_count": correct_count,
                "total_samples": k,
                "answers": answers[:3],  # Save first 3 for debugging
                "extraction_methods": [],
                "tokens": num_tokens,
            }

            k_correct[k] += 1 if is_correct else 0
            k_tokens[k] += num_tokens
            tokens_for_problem += num_tokens

            step = idx * len(K_VALUES) + K_VALUES.index(k) + 1
            current_acc = k_correct[k] / (idx + 1)
            report_progress(step=step, total_steps=total_samples,
                          metric={"accuracy_k1": k_correct[1] / (idx + 1) if idx >= 0 else 0,
                                 "current_k": k,
                                 "problem": idx + 1})

        # Fit Arrhenius model for this problem
        p_values = [1 if problem_data["k_results"][k]["correct"] else 0 for k in K_VALUES]
        if sum(p_values) > 0 and sum(p_values) < len(K_VALUES):
            fit_result = fit_arrhenius_model(K_VALUES, p_values)
            problem_data["saturation_fit"] = fit_result

        problem_results.append(problem_data)

        if (idx + 1) % 5 == 0:
            acc_str = ", ".join([f"k{k}={k_correct[k]/(idx+1):.2f}" for k in K_VALUES])
            print(f"[PROGRESS] {idx + 1}/{total_problems} problems, {tokens_for_problem} tokens. Acc: {acc_str}")

        if (idx + 1) % 10 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    results["problems"] = problem_results

    # Calculate aggregate accuracy
    aggregate = {}
    for k in K_VALUES:
        aggregate[k] = {
            "accuracy": k_correct[k] / total_problems,
            "total_correct": k_correct[k],
            "total_problems": total_problems,
            "avg_tokens": k_tokens[k] / total_problems,
        }
    results["aggregate"] = aggregate

    # Fit on aggregate data
    p_agg = [aggregate[k]["accuracy"] for k in K_VALUES]
    agg_fit = fit_arrhenius_model(K_VALUES, p_agg)
    results["aggregate_fit"] = agg_fit

    # Fit statistics
    valid_fits = [p["saturation_fit"] for p in problem_results
                  if p["saturation_fit"] and p["saturation_fit"].get("fit_success")]
    if valid_fits:
        results["fit_statistics"] = {
            "valid_fits": len(valid_fits),
            "mean_r_squared": float(np.mean([f["r_squared"] for f in valid_fits])),
            "median_r_squared": float(np.median([f["r_squared"] for f in valid_fits])),
            "mean_p_inf": float(np.mean([f["p_inf"] for f in valid_fits])),
            "mean_k0": float(np.mean([f["k0"] for f in valid_fits])),
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
        analysis["h1_notes"] = f"Weak Arrhenius fit with R²={r_squared:.3f}"
    else:
        analysis["h1_status"] = "REJECTED"
        analysis["h1_notes"] = f"Arrhenius kinetics rejected with R²={r_squared:.3f}"

    # Accuracy by k
    agg = results.get("aggregate", {})
    accuracy_table = {}
    for k in K_VALUES:
        if k in agg:
            accuracy_table[f"k={k}"] = f"{agg[k]['accuracy']:.3f}"
    analysis["accuracy_by_k"] = accuracy_table

    # Summary
    if analysis["h1_pass"]:
        analysis["recommendation"] = "GO - H1 confirmed"
    elif r_squared >= 0.7:
        analysis["recommendation"] = "REFINE - Weak Arrhenius fit"
    else:
        analysis["recommendation"] = "NO_GO - Arrhenius form rejected"

    return analysis


def save_results(results: Dict, analysis: Dict):
    """Save results to files."""
    results_file = RESULTS_DIR / f"{TASK_ID}_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"[SAVE] Results saved to {results_file}")

    analysis_file = RESULTS_DIR / f"{TASK_ID}_analysis.json"
    with open(analysis_file, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"[SAVE] Analysis saved to {analysis_file}")

    # Summary markdown
    summary_md = f"""# G1 Saturation Experiment v2 Results

## Task: {TASK_ID}
## Date: {datetime.now().isoformat()}

## Key Improvements from v1
- System prompt to encourage \\boxed{{}} answers
- Improved answer extraction with multiple fallback strategies
- Numeric comparison for numerical answers
- Increased MAX_TOKENS to 2048

## H1 Evaluation

**Status**: {analysis['h1_status']}
**R² (threshold > 0.85)**: {analysis['actual_r_squared']:.4f}
**P_∞ estimate**: {analysis.get('p_inf_estimate', 0):.3f}
**k₀ estimate**: {analysis.get('k0_estimate', 0):.3f}

**Pass**: {'YES' if analysis['h1_pass'] else 'NO'}

### Notes
{analysis.get('h1_notes', 'N/A')}

## Accuracy by k

| k | Accuracy |
|---|----------|
"""
    for k, acc in analysis.get('accuracy_by_k', {}).items():
        summary_md += f"| {k} | {acc} |\n"

    summary_md += f"""
## Recommendation
{analysis['recommendation']}
"""
    if "fit_statistics" in results:
        stats = results["fit_statistics"]
        summary_md += f"""
## Fit Statistics
- Valid fits: {stats.get('valid_fits', 0)}
- Mean R²: {stats.get('mean_r_squared', 0):.3f}
- Median R²: {stats.get('median_r_squared', 0):.3f}
- Mean k₀: {stats.get('mean_k0', 0):.3f}
"""

    summary_file = RESULTS_DIR / f"{TASK_ID}_summary.md"
    with open(summary_file, "w") as f:
        f.write(summary_md)
    print(f"[SAVE] Summary saved to {summary_file}")

    return results_file, analysis_file


def main():
    """Main entry point."""
    print("=" * 60)
    print(f"G1 SATURATION EXPERIMENT v2 - Arrhenius Kinetics Validation")
    print("=" * 60)

    setup_logging()

    try:
        # Load model and data
        model, tokenizer = load_model()
        dataset = load_dataset_subset()

        # Run experiment
        print("\n[RUN] Starting saturation experiment v2...")
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
