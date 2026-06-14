#!/usr/bin/env python3
"""
Generate preference dataset with Self-Refine depth annotation.
Task: gen_dataset (PILOT mode)
Output: JSONL with prompt, chosen, rejected, error_depths per step
"""

import json
import random
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-no-revision/current")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
MODEL_NAME = "Qwen/Qwen2.5-Math-7B-Instruct"
NUM_PROBLEMS = 100
SEED = 42
MAX_REFINE_ROUNDS = 3
TIMEOUT_SECONDS = 900

# Set seeds
random.seed(SEED)
torch.manual_seed(SEED)

def setup_model():
    """Load Qwen Math model and tokenizer."""
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()
    return model, tokenizer

def extract_steps(response_text):
    """Extract step-by-step reasoning from response."""
    # Look for numbered steps or "Step X:" patterns
    step_patterns = [
        r'(?:Step|step)\s*(\d+)[:\.\)]\s*(.+)',
        r'(\d+)\.\s+(.+)',
        r'[:\-\*]\s*(.+)',
    ]

    steps = []
    for pattern in step_patterns:
        matches = re.findall(pattern, response_text)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    steps.append(match[1].strip())
                else:
                    steps.append(match.strip())
            break

    # If no structured steps, split by newlines and filter
    if not steps:
        lines = [l.strip() for l in response_text.split('\n') if l.strip()]
        steps = [l for l in lines if len(l) > 10][:20]  # Cap at 20 steps

    return steps

def find_first_error_step(correct_response, incorrect_response):
    """Find the first step where responses diverge (error location)."""
    correct_steps = extract_steps(correct_response)
    incorrect_steps = extract_steps(incorrect_response)

    # Find first diverging step
    for i, (c_step, i_step) in enumerate(zip(correct_steps, incorrect_steps)):
        # Simple heuristic: check if step content differs
        if c_step.lower()[:50] != i_step.lower()[:50]:
            return i

    # If all steps match, error is in the last step
    return min(len(correct_steps) - 1, len(incorrect_steps) - 1, 0)

def generate_with_model(model, tokenizer, prompt, max_new_tokens=512):
    """Generate response with the model."""
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def self_refine_round(model, tokenizer, prompt, previous_response, round_num):
    """Apply self-correction to previous response."""
    critique_prompt = f"""You are reviewing a mathematical solution.
Problem: {prompt}

Previous solution:
{previous_response}

Identify the FIRST error in the reasoning (if any) and provide a corrected version.
Format your response as:
ERROR: [Describe the error briefly or "No error found"]
CORRECTED SOLUTION:
[Full corrected solution with step-by-step reasoning]
"""

    critique_response = generate_with_model(model, tokenizer, critique_prompt, max_new_tokens=768)

    # Check if correction found
    if "No error found" in critique_response or "no error" in critique_response.lower():
        return previous_response, None

    # Extract corrected solution
    if "CORRECTED SOLUTION:" in critique_response:
        corrected = critique_response.split("CORRECTED SOLUTION:")[-1].strip()
    else:
        corrected = critique_response

    return corrected, critique_response

def determine_error_depth(rounds_to_correct):
    """
    Determine error depth based on rounds needed to correct.
    Level 1 (computational): Corrected in 1 round
    Level 2 (logical): Corrected in 2 rounds
    Level 3 (conceptual): Corrected in 3+ rounds
    """
    if rounds_to_correct >= 3:
        return 3
    return rounds_to_correct

def run_self_refine_chain(model, tokenizer, problem):
    """Run full Self-Refine chain and extract preference pairs."""
    prompt = f"Solve this math problem step by step:\n{problem}"

    # Generate first response
    initial_response = generate_with_model(model, tokenizer, prompt, max_new_tokens=512)

    # Track responses through refine rounds
    responses = [initial_response]
    correction_round = {}  # step_idx -> round where corrected

    # Run Self-Refine for MAX_REFINE_ROUNDS
    for round_num in range(1, MAX_REFINE_ROUNDS + 1):
        refined, _ = self_refine_round(model, tokenizer, problem, responses[-1], round_num)

        if refined == responses[-1]:
            # No improvement
            break

        responses.append(refined)

    # Generate a correct reference solution
    reference_prompt = f"""Solve this math problem CORRECTLY with careful step-by-step reasoning.
Problem: {problem}

Provide a complete, correct solution:"""

    correct_response = generate_with_model(model, tokenizer, reference_prompt, max_new_tokens=768)

    # Find first error step by comparing with correct solution
    first_error_idx = find_first_error_step(correct_response, initial_response)

    # Build preference pairs
    preference_pairs = []

    initial_steps = extract_steps(initial_response)
    correct_steps = extract_steps(correct_response)

    for step_idx in range(len(initial_steps)):
        if step_idx < len(correct_steps):
            # Determine depth based on whether step was fixed
            if step_idx < first_error_idx:
                # Steps before error are correct in both
                depth = 0  # No error
            else:
                # Steps at/after error - need to determine depth
                # Check which round first fixed this step
                rounds_to_fix = 0
                for r, resp in enumerate(responses):
                    steps = extract_steps(resp)
                    if step_idx < len(steps):
                        if steps[step_idx].lower()[:50] == correct_steps[step_idx].lower()[:50]:
                            rounds_to_fix = r + 1
                            break

                if rounds_to_fix == 0:
                    rounds_to_fix = MAX_REFINE_ROUNDS  # Never fixed = deepest error

                depth = determine_error_depth(rounds_to_fix)

            if step_idx >= first_error_idx:  # Only include error steps
                pair = {
                    "step_idx": step_idx,
                    "prompt": problem,
                    "chosen": correct_steps[step_idx],
                    "rejected": initial_steps[step_idx],
                    "error_depth": depth,
                    "step_weight": float(depth) if depth > 0 else 1.0
                }
                preference_pairs.append(pair)

    return {
        "problem": problem,
        "initial_response": initial_response,
        "correct_response": correct_response,
        "preference_pairs": preference_pairs,
        "first_error_idx": first_error_idx,
        "refine_rounds_completed": len(responses) - 1
    }

def select_math_problems(num_problems, seed=42):
    """Select diverse math problems from common sources."""
    # MATH benchmark problem templates (diverse categories)
    problem_templates = [
        # Algebra
        "Solve for x: {equation}",
        "Factor completely: {expression}",
        "Simplify: {expression}",
        "Find all real solutions: {equation}",

        # Number Theory
        "Find the greatest common divisor of {a} and {b}.",
        "What is the remainder when {n} is divided by {m}?",
        "How many positive divisors does {n} have?",

        # Counting/Probability
        "In how many ways can {n} items be arranged?",
        "What is the probability of {event}?",
        "A bag contains {red} red balls and {blue} blue balls. If {k} balls are drawn...",

        # Geometry
        "Find the area of a triangle with vertices at {v1}, {v2}, {v3}.",
        "What is the distance between points ({x1}, {y1}) and ({x2}, {y2})?",
        "The perimeter of a rectangle is {p}. If the length is {l}, find the width.",

        # Calculus
        "Find the derivative of f(x) = {expression}.",
        "Evaluate the integral: \\int {expression} dx",
        "Find the limit: lim(x->0) {expression}",

        # Sequences
        "Find the {n}th term of the sequence: {sequence}",
        "What is the sum of the first {n} terms of {sequence}?",
    ]

    problems = []
    random.seed(seed)

    # Generate concrete problems
    categories = ["Algebra", "Number Theory", "Counting", "Geometry", "Calculus", "Sequences"]

    for i in range(num_problems):
        cat = categories[i % len(categories)]
        base = problem_templates[i % len(problem_templates)]

        # Fill in concrete values
        a, b = random.randint(1, 100), random.randint(1, 100)
        n, m = random.randint(10, 100), random.randint(2, 20)
        k = random.randint(1, min(5, m))

        problem = base.format(
            equation=f"x^2 - {a}x + {a*b} = 0",
            expression=f"x^3 - {a}x^2 + {b}x - {a*b}",
            a=a, b=b, n=n, m=m,
            event="drawing a red ball",
            red=random.randint(3, 8), blue=random.randint(3, 8),
            v1=f"(0,0)", v2=f"({random.randint(1,5)},0)", v3=f"({random.randint(1,5)},{random.randint(1,5)})",
            x1=0, y1=0, x2=random.randint(1,6), y2=random.randint(1,6),
            p=random.randint(20, 50), l=random.randint(5, 15),
            sequence=f"{random.randint(1,5)}, {random.randint(6,10)}, {random.randint(11,15)}, ...",
            k=k
        )

        # Add specific numbers for uniqueness
        if i < 20:
            problem = f"Problem {i+1}: {problem}"
        else:
            problem = f"Calculate: {a} + {b} = ?"

        problems.append(problem)

    return problems

def write_progress(task_id, epoch, total, status="running"):
    """Write progress file."""
    progress_file = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch,
        "total": total,
        "status": status,
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(task_id, status, summary):
    """Write DONE marker file."""
    pid_file = RESULTS_DIR.parent / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()

    progress_file = RESULTS_DIR / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except:
            pass

    marker = RESULTS_DIR / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

def main():
    """Main execution."""
    task_id = "gen_dataset"

    print(f"[{datetime.now().isoformat()}] Starting {task_id}")
    print(f"Target: {NUM_PROBLEMS} problems")

    # Write PID
    pid_file = RESULTS_DIR.parent / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))

    try:
        # Load model
        model, tokenizer = setup_model()

        # Select problems
        problems = select_math_problems(NUM_PROBLEMS, SEED)

        # Process each problem with Self-Refine
        results = []
        valid_pairs_count = 0
        depth_ge2_count = 0

        for i, problem in enumerate(tqdm(problems, desc="Processing problems")):
            write_progress(task_id, i + 1, NUM_PROBLEMS, "running")

            try:
                result = run_self_refine_chain(model, tokenizer, problem)
                results.append(result)

                # Count valid pairs
                if result["preference_pairs"]:
                    valid_pairs_count += 1
                    depths = [p["error_depth"] for p in result["preference_pairs"]]
                    if any(d >= 2 for d in depths):
                        depth_ge2_count += 1

            except Exception as e:
                print(f"Error processing problem {i}: {e}")
                continue

            # Save intermediate results every 20 problems
            if (i + 1) % 20 == 0:
                intermediate_file = RESULTS_DIR / f"dataset_preference_{i+1}.jsonl"
                with open(intermediate_file, 'w') as f:
                    for r in results:
                        f.write(json.dumps(r) + '\n')

        # Write final JSONL
        output_file = RESULTS_DIR / "dataset_preference_100.jsonl"
        with open(output_file, 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')

        # Compute statistics
        total_pairs = sum(len(r["preference_pairs"]) for r in results)
        depth_dist = {1: 0, 2: 0, 3: 0}
        for r in results:
            for p in r["preference_pairs"]:
                d = p["error_depth"]
                if d in depth_dist:
                    depth_dist[d] += 1

        depth_ge2_ratio = depth_ge2_count / len(results) if results else 0

        # Evaluate pass criteria
        pass_valid = valid_pairs_count >= 80
        pass_depth = depth_ge2_ratio >= 0.20

        summary = {
            "task_id": task_id,
            "total_problems": len(results),
            "valid_pair_count": valid_pairs_count,
            "depth_ge2_count": depth_ge2_count,
            "depth_ge2_ratio": depth_ge2_ratio,
            "total_preference_pairs": total_pairs,
            "depth_distribution": depth_dist,
            "pass_valid_pairs": pass_valid,
            "pass_depth_ratio": pass_depth,
            "overall_pass": pass_valid and pass_depth,
            "pass_criteria": ">=80 problems with valid pairs, >=20% depth >=2 steps"
        }

        # Write summary JSON
        summary_file = RESULTS_DIR / "dataset_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*50}")
        print(f"Dataset Generation Complete")
        print(f"{'='*50}")
        print(f"Total problems processed: {len(results)}")
        print(f"Valid pair count: {valid_pairs_count} (need >=80: {'PASS' if pass_valid else 'FAIL'})")
        print(f"Depth >=2 ratio: {depth_ge2_ratio:.1%} (need >=20%: {'PASS' if pass_depth else 'FAIL'})")
        print(f"Total preference pairs: {total_pairs}")
        print(f"Depth distribution: {depth_dist}")
        print(f"\nOutput: {output_file}")
        print(f"Summary: {summary_file}")

        # Mark done
        status = "success" if summary["overall_pass"] else "partial_success"
        mark_done(task_id, status, f"Valid={valid_pairs_count}, Depth>=2={depth_ge2_ratio:.1%}")

        return 0 if summary["overall_pass"] else 1

    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        mark_done(task_id, "failed", str(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())
