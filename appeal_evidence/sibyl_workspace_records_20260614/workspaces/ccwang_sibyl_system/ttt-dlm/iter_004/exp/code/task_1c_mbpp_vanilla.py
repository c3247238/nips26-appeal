"""
Task 1c (PILOT): Vanilla Baseline on MBPP (16 samples).
Dream-7B vanilla (origin, 128 steps, temp=0.4) on MBPP sanitized test set.
Metrics: pass@1, code extraction rate, generation time, diversity.

Usage:
    CUDA_VISIBLE_DEVICES=1 python3 task_1c_mbpp_vanilla.py
"""
import os, sys, json, time, re, tempfile, subprocess, signal
from pathlib import Path
from collections import Counter

import numpy as np
import torch

# === Config ===
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 16
SEED = 42
GEN_LEN = 512       # Code can be longer than math answers
STEPS = 128
TEMPERATURE = 0.4
ALG = "origin"
MASK_TOKEN_ID = 151666
SANDBOX_TIMEOUT = 15  # seconds per test execution


# === MBPP Data Loading ===
def load_mbpp_sanitized(n=None):
    """Load MBPP sanitized test split."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    problems = []
    for i, item in enumerate(ds):
        if n is not None and i >= n:
            break
        problems.append({
            "id": i,
            "task_id": item.get("task_id", i),
            "text": item["prompt"],
            "test_list": item["test_list"],
            "code": item.get("code", ""),
        })
    return problems


def format_mbpp_prompt(problem):
    """Format an MBPP problem as a 0-shot prompt for Dream-7B."""
    test_example = problem["test_list"][0] if problem["test_list"] else ""
    return (
        f"Write a Python function to solve the following problem.\n\n"
        f"Problem: {problem['text']}\n\n"
        f"Test case: {test_example}\n\n"
        f"Provide only the Python function code, no explanations."
    )


# === Code Extraction ===
def extract_code_from_response(text):
    """Extract Python code from model response.

    Handles:
    - Code blocks (```python ... ```)
    - Raw code (def ...)
    - Mixed text and code
    """
    # Try to find code block first
    code_block = re.search(r'```(?:python)?\s*\n?(.*?)```', text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()

    # Try to find function definition
    func_match = re.search(r'((?:def |class |import |from ).*)', text, re.DOTALL)
    if func_match:
        return func_match.group(1).strip()

    # Return the full text as last resort
    return text.strip()


# === Sandbox Execution ===
def run_code_sandbox(code, test_code, timeout=SANDBOX_TIMEOUT):
    """Run code + tests in a sandboxed subprocess.

    Returns:
      - passed: bool
      - error: str or None
      - runtime: float (seconds)
    """
    full_code = code + "\n\n" + test_code

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_code)
        tmp_path = f.name

    try:
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        runtime = time.time() - t0

        if result.returncode == 0:
            return {"passed": True, "error": None, "runtime": runtime}
        else:
            return {"passed": False, "error": result.stderr[:500], "runtime": runtime}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": f"Timeout ({timeout}s)", "runtime": timeout}
    except Exception as e:
        return {"passed": False, "error": str(e)[:500], "runtime": 0}
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass


# === Text Quality Metrics ===
def distinct_n(text, n):
    tokens = text.split()
    if len(tokens) < n + 1:
        return 1.0
    ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams) if ngrams else 1.0


def rep_n(text, n):
    return 1.0 - distinct_n(text, n)


def compute_diversity_metrics(texts):
    """Compute corpus-level diversity metrics."""
    if not texts:
        return {}
    return {
        "distinct_1": float(np.mean([distinct_n(t, 1) for t in texts])),
        "distinct_2": float(np.mean([distinct_n(t, 2) for t in texts])),
        "distinct_3": float(np.mean([distinct_n(t, 3) for t in texts])),
        "rep_2": float(np.mean([rep_n(t, 2) for t in texts])),
        "rep_3": float(np.mean([rep_n(t, 3) for t in texts])),
        "avg_length_words": float(np.mean([len(t.split()) for t in texts])),
        "avg_length_chars": float(np.mean([len(t) for t in texts])),
    }


# === Model Loading & Generation ===
def load_dream(device="cuda:0"):
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"Dream-7B loaded on {device}, dtype=bfloat16")
    return model, tokenizer


def generate_vanilla(model, tokenizer, prompt_text, device="cuda:0"):
    """Generate with Dream-7B vanilla decoding."""
    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)

    t0 = time.time()
    output = model.diffusion_generate(
        input_ids,
        max_new_tokens=GEN_LEN,
        output_history=False,
        return_dict_in_generate=True,
        steps=STEPS,
        temperature=TEMPERATURE,
        alg=ALG,
    )
    elapsed = time.time() - t0

    seq = output.sequences
    prompt_len = len(prompt_ids)
    gen_ids = seq[0, prompt_len:].tolist()
    # Remove mask tokens and EOS
    eos_id = tokenizer.eos_token_id
    clean_ids = []
    for t in gen_ids:
        if t == MASK_TOKEN_ID:
            continue
        if t == eos_id:
            break
        clean_ids.append(t)
    text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()
    return text, elapsed, len(prompt_ids)


# === Main ===
def main():
    device = "cuda:0"
    print(f"=== Task 1c PILOT: Vanilla Baseline on MBPP ===")
    print(f"Samples: {N_SAMPLES}, Seed: {SEED}, Steps: {STEPS}, Temp: {TEMPERATURE}")
    print(f"Gen length: {GEN_LEN}, Device: {device}")

    # Load MBPP problems
    print(f"\nLoading MBPP sanitized test set...")
    problems = load_mbpp_sanitized(n=N_SAMPLES)
    print(f"Loaded {len(problems)} MBPP problems")
    for i, p in enumerate(problems[:3]):
        print(f"  [{i}] task_id={p['task_id']}: {p['text'][:80]}...")

    # Load model
    model, tokenizer = load_dream(device)

    # Set seeds
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)

    # Run generation + evaluation
    results = []
    pass_count = 0
    total_gen_time = 0
    n_nonempty = 0
    n_has_def = 0

    for i, problem in enumerate(problems):
        prompt_text = format_mbpp_prompt(problem)
        text, elapsed, prompt_len = generate_vanilla(model, tokenizer, prompt_text, device)
        total_gen_time += elapsed

        # Extract code
        code = extract_code_from_response(text)
        is_nonempty = len(code.strip()) > 0
        has_def = "def " in code
        if is_nonempty:
            n_nonempty += 1
        if has_def:
            n_has_def += 1

        # Run sandbox evaluation
        test_code = "\n".join(problem["test_list"])
        sandbox_result = run_code_sandbox(code, test_code)

        passed = sandbox_result["passed"]
        if passed:
            pass_count += 1

        # Diversity metrics for this sample
        d1 = distinct_n(text, 1)
        d2 = distinct_n(text, 2)
        word_count = len(text.split())

        result = {
            "idx": i,
            "task_id": problem["task_id"],
            "problem_text": problem["text"][:200],
            "generated_text": text,
            "extracted_code": code[:500],
            "passed": passed,
            "sandbox_error": sandbox_result["error"],
            "sandbox_runtime": sandbox_result["runtime"],
            "gen_time_s": elapsed,
            "word_count": word_count,
            "has_def": has_def,
            "is_nonempty": is_nonempty,
            "distinct_1": d1,
            "distinct_2": d2,
        }
        results.append(result)

        status = "PASS" if passed else "FAIL"
        print(f"  [{i+1}/{N_SAMPLES}] {status} | task={problem['task_id']} | "
              f"has_def={has_def} | time={elapsed:.1f}s | words={word_count}")
        print(f"    Problem: {problem['text'][:100]}")
        print(f"    Code: {code[:150]}")
        if sandbox_result["error"]:
            print(f"    Error: {sandbox_result['error'][:150]}")

    # Free model
    del model
    torch.cuda.empty_cache()

    # Aggregate metrics
    pass_at_1 = pass_count / N_SAMPLES
    avg_gen_time = total_gen_time / N_SAMPLES
    texts = [r["generated_text"] for r in results]
    diversity = compute_diversity_metrics(texts)

    summary = {
        "task": "task_1c",
        "mode": "pilot",
        "method": "vanilla",
        "model": "Dream-v0-Instruct-7B",
        "benchmark": "mbpp_sanitized",
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "steps": STEPS,
        "temperature": TEMPERATURE,
        "alg": ALG,
        "gen_len": GEN_LEN,
        "pass_at_1": pass_at_1,
        "pass_count": pass_count,
        "total_count": N_SAMPLES,
        "n_nonempty": n_nonempty,
        "n_has_def": n_has_def,
        "code_extraction_rate": n_nonempty / N_SAMPLES,
        "def_rate": n_has_def / N_SAMPLES,
        "avg_gen_time_s": avg_gen_time,
        "total_gen_time_s": total_gen_time,
        "diversity": diversity,
        "results": results,
        "torch_version": torch.__version__,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Print summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: Task 1c Pilot - Vanilla MBPP Baseline")
    print(f"{'='*70}")
    print(f"  Pass@1:             {pass_count}/{N_SAMPLES} = {pass_at_1:.1%}")
    print(f"  Non-empty outputs:  {n_nonempty}/{N_SAMPLES} = {n_nonempty/N_SAMPLES:.1%}")
    print(f"  Has 'def':          {n_has_def}/{N_SAMPLES} = {n_has_def/N_SAMPLES:.1%}")
    print(f"  Avg gen time:       {avg_gen_time:.1f}s per sample")
    print(f"  Total gen time:     {total_gen_time:.0f}s")
    if diversity:
        print(f"  Distinct-1:         {diversity['distinct_1']:.3f}")
        print(f"  Distinct-2:         {diversity['distinct_2']:.3f}")
        print(f"  Avg words:          {diversity['avg_length_words']:.0f}")

    # Pass criteria check
    print(f"\n--- Pass Criteria ---")
    gen_ok = n_nonempty == N_SAMPLES
    sandbox_ok = True  # If we reach here, sandbox didn't crash
    print(f"  16 non-empty code outputs: {'PASS' if gen_ok else 'FAIL'} ({n_nonempty}/{N_SAMPLES})")
    print(f"  Sandbox execution OK:      {'PASS' if sandbox_ok else 'FAIL'}")
    overall = "GO" if (gen_ok and sandbox_ok) else "CONDITIONAL-GO" if sandbox_ok else "NO-GO"
    print(f"  Overall: {overall}")

    # Save results
    out_file = RESULTS_DIR / "task_1c_mbpp_vanilla.json"
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
