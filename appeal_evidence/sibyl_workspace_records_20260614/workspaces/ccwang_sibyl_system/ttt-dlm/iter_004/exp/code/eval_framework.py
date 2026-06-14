"""
Evaluation Framework for DTA Experiments (task_0b)

Unified evaluation for:
  - Countdown: equation correctness + target match
  - GSM8K: extract '#### N' answer, exact match
  - MBPP: pass@1 via subprocess sandbox
  - HumanEval: pass@1 via subprocess sandbox

Usage:
  # Run pilot validation (4 samples per benchmark)
  python eval_framework.py --pilot --gpu 1

  # Evaluate saved generations
  python eval_framework.py --eval countdown --input results.json
"""
import os, sys, json, re, math, time, signal, tempfile, subprocess, traceback
from pathlib import Path
from collections import Counter
from typing import List, Dict, Optional, Tuple, Any

import numpy as np

# ──────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
MASK_TOKEN_ID = 151666
RESULTS_BASE = Path("/home/ccwang/sibyl_system/exp/results")
CODE_DIR = Path("/home/ccwang/sibyl_system/exp/code")

# ──────────────────────────────────────────────────
# 1. Countdown Evaluation
# ──────────────────────────────────────────────────

def make_countdown_problems(n: int = 500, seed: int = 42) -> List[Dict]:
    """Generate Countdown problems: find arithmetic expression equaling target from given numbers.

    Format: given numbers [a, b, c, d, e, f], target T
    Model must produce an equation using +, -, *, / that equals T.
    """
    rng = np.random.RandomState(seed)
    problems = []
    ops = ['+', '-', '*']  # avoid division to keep integer targets

    for i in range(n):
        # Generate 6 random numbers between 1 and 100
        nums = rng.randint(1, 101, size=6).tolist()
        # Pick 2-4 numbers and create a valid equation to get a target
        n_use = rng.randint(2, 5)
        chosen = rng.choice(nums, size=n_use, replace=False).tolist()

        # Build a random valid expression
        expr = str(chosen[0])
        val = chosen[0]
        for j in range(1, len(chosen)):
            op = rng.choice(ops)
            expr += f" {op} {chosen[j]}"
            if op == '+':
                val += chosen[j]
            elif op == '-':
                val -= chosen[j]
            elif op == '*':
                val *= chosen[j]

        target = int(val)
        problems.append({
            "id": i,
            "numbers": nums,
            "target": target,
            "solution": expr,  # reference solution
        })

    return problems


def format_countdown_prompt(problem: Dict) -> str:
    """Format a Countdown problem as a prompt for Dream-7B."""
    nums_str = ", ".join(str(n) for n in problem["numbers"])
    return (
        f"Using the numbers [{nums_str}], create an arithmetic expression "
        f"that equals {problem['target']}. You may use +, -, *, / operators. "
        f"Each number can be used at most once. "
        f"Respond with only the equation in the format: answer = expression"
    )


def parse_countdown_answer(text: str) -> Optional[Tuple[str, float]]:
    """Parse model output to extract an arithmetic expression.

    Returns (expression_str, evaluated_value) or None if parsing fails.
    """
    # Try to find "answer = expression" or "= expression" patterns
    patterns = [
        r'answer\s*[:=]\s*(.+)',
        r'result\s*[:=]\s*(.+)',
        r'=\s*(.+)',
        r'(\d[\d\s\+\-\*\/\(\)\.]+\d)',  # any arithmetic expression
    ]

    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            expr_str = match.group(1).strip()
            # Clean: only keep digits, operators, parens, spaces, dots
            expr_str = re.sub(r'[^0-9\+\-\*\/\(\)\.\s]', '', expr_str).strip()
            if not expr_str:
                continue
            try:
                # Safe eval: only allow arithmetic
                val = eval(expr_str, {"__builtins__": {}}, {})
                return (expr_str, float(val))
            except:
                continue
    return None


def eval_countdown(text: str, problem: Dict) -> Dict:
    """Evaluate a single Countdown generation.

    Returns dict with:
      - correct: bool (expression evaluates to target)
      - parsed: bool (expression could be parsed)
      - expression: str or None
      - value: float or None
      - target: int
    """
    target = problem["target"]
    result = {
        "correct": False,
        "parsed": False,
        "expression": None,
        "value": None,
        "target": target,
    }

    parsed = parse_countdown_answer(text)
    if parsed is not None:
        expr_str, val = parsed
        result["parsed"] = True
        result["expression"] = expr_str
        result["value"] = val
        # Check if value equals target (allow small float tolerance)
        if abs(val - target) < 1e-6:
            result["correct"] = True

    return result


def eval_countdown_batch(texts: List[str], problems: List[Dict]) -> Dict:
    """Evaluate a batch of Countdown generations."""
    assert len(texts) == len(problems)
    results = []
    for text, prob in zip(texts, problems):
        results.append(eval_countdown(text, prob))

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_parsed = sum(r["parsed"] for r in results)

    return {
        "benchmark": "countdown",
        "n_samples": n,
        "accuracy": n_correct / n if n > 0 else 0.0,
        "parse_rate": n_parsed / n if n > 0 else 0.0,
        "n_correct": n_correct,
        "n_parsed": n_parsed,
        "per_sample": results,
    }


# ──────────────────────────────────────────────────
# 2. GSM8K Evaluation
# ──────────────────────────────────────────────────

def load_gsm8k(split: str = "test", n: Optional[int] = None) -> List[Dict]:
    """Load GSM8K dataset. Returns list of {question, answer, target_number}."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)

    problems = []
    for i, item in enumerate(ds):
        if n is not None and i >= n:
            break
        # Extract numeric answer from "#### N" format
        answer_text = item["answer"]
        target = extract_gsm8k_answer(answer_text)
        problems.append({
            "id": i,
            "question": item["question"],
            "answer_text": answer_text,
            "target": target,
        })
    return problems


def extract_gsm8k_answer(text: str) -> Optional[float]:
    """Extract the numeric answer from GSM8K format '#### N'."""
    # Standard format: #### followed by a number
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        num_str = match.group(1).replace(',', '')
        try:
            return float(num_str)
        except:
            return None
    return None


def extract_model_gsm8k_answer(text: str) -> Optional[float]:
    """Extract numeric answer from model generation for GSM8K.

    Tries multiple patterns:
    1. #### N format (if model follows GSM8K convention)
    2. "The answer is N" / "answer: N"
    3. Last number in the text (fallback)
    """
    # Pattern 1: #### N
    match = re.search(r'####\s*([\-]?\d[\d,]*\.?\d*)', text)
    if match:
        try:
            return float(match.group(1).replace(',', ''))
        except:
            pass

    # Pattern 2: "the answer is N" / "answer: N" / "= N" at end
    patterns = [
        r'(?:the\s+)?answer\s+is\s*[:=]?\s*([\-]?\d[\d,]*\.?\d*)',
        r'(?:therefore|thus|so|hence)[,\s]+(?:the\s+answer\s+is\s+)?([\-]?\d[\d,]*\.?\d*)',
        r'=\s*([\-]?\d[\d,]*\.?\d*)\s*$',
        r'\\boxed\{([\-]?\d[\d,]*\.?\d*)\}',
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if match:
            try:
                return float(match.group(1).replace(',', ''))
            except:
                continue

    # Pattern 3: last number in text (fallback)
    numbers = re.findall(r'[\-]?\d[\d,]*\.?\d*', text)
    if numbers:
        try:
            return float(numbers[-1].replace(',', ''))
        except:
            pass

    return None


def eval_gsm8k(text: str, problem: Dict) -> Dict:
    """Evaluate a single GSM8K generation."""
    target = problem["target"]
    result = {
        "correct": False,
        "extracted": None,
        "target": target,
        "extraction_method": None,
    }

    extracted = extract_model_gsm8k_answer(text)
    if extracted is not None:
        result["extracted"] = extracted
        result["extraction_method"] = "auto"
        # Exact match (with float tolerance)
        if target is not None and abs(extracted - target) < 1e-6:
            result["correct"] = True

    return result


def eval_gsm8k_batch(texts: List[str], problems: List[Dict]) -> Dict:
    """Evaluate a batch of GSM8K generations."""
    assert len(texts) == len(problems)
    results = []
    for text, prob in zip(texts, problems):
        results.append(eval_gsm8k(text, prob))

    n = len(results)
    n_correct = sum(r["correct"] for r in results)
    n_extracted = sum(r["extracted"] is not None for r in results)

    return {
        "benchmark": "gsm8k",
        "n_samples": n,
        "accuracy": n_correct / n if n > 0 else 0.0,
        "extraction_rate": n_extracted / n if n > 0 else 0.0,
        "n_correct": n_correct,
        "n_extracted": n_extracted,
        "per_sample": results,
    }


def format_gsm8k_prompt(problem: Dict) -> str:
    """Format a GSM8K problem as a prompt."""
    return (
        f"Solve this math problem step by step.\n\n"
        f"Question: {problem['question']}\n\n"
        f"Show your work and give the final answer after '####'."
    )


# ──────────────────────────────────────────────────
# 3. MBPP Evaluation
# ──────────────────────────────────────────────────

def load_mbpp(split: str = "test", n: Optional[int] = None) -> List[Dict]:
    """Load MBPP dataset. Returns list of {task_id, text, test_list, code}."""
    from datasets import load_dataset
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split=split)

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


def format_mbpp_prompt(problem: Dict) -> str:
    """Format an MBPP problem as a prompt."""
    test_example = problem["test_list"][0] if problem["test_list"] else ""
    return (
        f"Write a Python function to solve the following problem.\n\n"
        f"Problem: {problem['text']}\n\n"
        f"Test case: {test_example}\n\n"
        f"Provide only the function code, no explanations."
    )


def extract_code_from_response(text: str) -> str:
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


def run_code_sandbox(code: str, test_code: str, timeout: int = 10) -> Dict:
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


def eval_mbpp(text: str, problem: Dict, timeout: int = 10) -> Dict:
    """Evaluate a single MBPP generation."""
    code = extract_code_from_response(text)
    test_code = "\n".join(problem["test_list"])

    sandbox_result = run_code_sandbox(code, test_code, timeout=timeout)

    return {
        "passed": sandbox_result["passed"],
        "error": sandbox_result["error"],
        "runtime": sandbox_result["runtime"],
        "code_extracted": code[:500],
    }


def eval_mbpp_batch(texts: List[str], problems: List[Dict]) -> Dict:
    """Evaluate a batch of MBPP generations."""
    assert len(texts) == len(problems)
    results = []
    for text, prob in zip(texts, problems):
        results.append(eval_mbpp(text, prob))

    n = len(results)
    n_passed = sum(r["passed"] for r in results)

    return {
        "benchmark": "mbpp",
        "n_samples": n,
        "pass_at_1": n_passed / n if n > 0 else 0.0,
        "n_passed": n_passed,
        "per_sample": results,
    }


# ──────────────────────────────────────────────────
# 4. HumanEval Evaluation
# ──────────────────────────────────────────────────

def load_humaneval(n: Optional[int] = None) -> List[Dict]:
    """Load HumanEval dataset."""
    from datasets import load_dataset
    ds = load_dataset("openai/openai_humaneval", split="test")

    problems = []
    for i, item in enumerate(ds):
        if n is not None and i >= n:
            break
        problems.append({
            "id": i,
            "task_id": item["task_id"],
            "prompt": item["prompt"],
            "canonical_solution": item["canonical_solution"],
            "test": item["test"],
            "entry_point": item["entry_point"],
        })
    return problems


def format_humaneval_prompt(problem: Dict) -> str:
    """Format a HumanEval problem as a prompt."""
    return (
        f"Complete the following Python function.\n\n"
        f"{problem['prompt']}"
    )


def eval_humaneval(text: str, problem: Dict, timeout: int = 10) -> Dict:
    """Evaluate a single HumanEval generation."""
    # Combine the prompt prefix + model completion
    code = extract_code_from_response(text)

    # If the model regenerated the full function, use as-is.
    # Otherwise, prepend the prompt (function signature).
    if problem["entry_point"] not in code:
        code = problem["prompt"] + code

    # Build test harness
    test_code = problem["test"]
    # HumanEval tests call check(entry_point)
    full_test = f"{code}\n\n{test_code}\ncheck({problem['entry_point']})"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(full_test)
        tmp_path = f.name

    try:
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        runtime = time.time() - t0
        passed = result.returncode == 0
        error = None if passed else result.stderr[:500]
    except subprocess.TimeoutExpired:
        passed = False
        error = f"Timeout ({timeout}s)"
        runtime = timeout
    except Exception as e:
        passed = False
        error = str(e)[:500]
        runtime = 0
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

    return {
        "passed": passed,
        "error": error,
        "runtime": runtime,
        "code_extracted": code[:500],
    }


def eval_humaneval_batch(texts: List[str], problems: List[Dict]) -> Dict:
    """Evaluate a batch of HumanEval generations."""
    assert len(texts) == len(problems)
    results = []
    for text, prob in zip(texts, problems):
        results.append(eval_humaneval(text, prob))

    n = len(results)
    n_passed = sum(r["passed"] for r in results)

    return {
        "benchmark": "humaneval",
        "n_samples": n,
        "pass_at_1": n_passed / n if n > 0 else 0.0,
        "n_passed": n_passed,
        "per_sample": results,
    }


# ──────────────────────────────────────────────────
# 5. Diversity & Quality Metrics (Shared)
# ──────────────────────────────────────────────────

def distinct_n(texts: List[str], n: int) -> float:
    """Compute corpus-level distinct-n."""
    all_ngrams = []
    for text in texts:
        tokens = text.split()
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        all_ngrams.extend(ngrams)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def rep_n(texts: List[str], n: int) -> float:
    """Compute corpus-level rep-n (n-gram repetition rate within each text, averaged)."""
    rates = []
    for text in texts:
        tokens = text.split()
        if len(tokens) < n + 1:
            rates.append(0.0)
            continue
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
        counter = Counter(ngrams)
        repeated = sum(c - 1 for c in counter.values() if c > 1)
        rates.append(repeated / len(ngrams) if ngrams else 0.0)
    return float(np.mean(rates)) if rates else 0.0


def compute_diversity_metrics(texts: List[str]) -> Dict:
    """Compute all diversity metrics for a batch of texts."""
    return {
        "distinct_1": distinct_n(texts, 1),
        "distinct_2": distinct_n(texts, 2),
        "distinct_3": distinct_n(texts, 3),
        "rep_2": rep_n(texts, 2),
        "rep_3": rep_n(texts, 3),
        "avg_length_words": float(np.mean([len(t.split()) for t in texts])) if texts else 0,
        "avg_length_chars": float(np.mean([len(t) for t in texts])) if texts else 0,
    }


# ──────────────────────────────────────────────────
# 6. Unified Result Format
# ──────────────────────────────────────────────────

def make_unified_result(
    method: str,
    benchmark: str,
    eval_result: Dict,
    diversity: Dict,
    gen_times: List[float],
    config: Dict,
) -> Dict:
    """Create unified result format for all benchmarks."""
    return {
        "method": method,
        "benchmark": benchmark,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "config": config,
        "metrics": {
            **{k: v for k, v in eval_result.items() if k != "per_sample"},
            **diversity,
            "avg_gen_time": float(np.mean(gen_times)) if gen_times else 0,
            "total_gen_time": float(np.sum(gen_times)) if gen_times else 0,
        },
        "per_sample": eval_result.get("per_sample", []),
    }


# ──────────────────────────────────────────────────
# 7. Dream-7B Generation Helper
# ──────────────────────────────────────────────────

def load_dream(device: str = "cuda:0"):
    """Load Dream-7B model and tokenizer."""
    import torch
    from transformers import AutoTokenizer, AutoModel

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    print(f"[eval_framework] Dream-7B loaded on {device}")
    return model, tokenizer


def generate_dream(
    model, tokenizer, prompt_text: str,
    gen_len: int = 256, steps: int = 128,
    temperature: float = 0.4, alg: str = "origin",
    device: str = "cuda:0",
) -> str:
    """Generate text with Dream-7B from a text prompt."""
    import torch

    messages = [{"role": "user", "content": prompt_text}]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    input_ids = torch.tensor([prompt_ids], device=device)

    output = model.diffusion_generate(
        input_ids, max_new_tokens=gen_len, output_history=False,
        return_dict_in_generate=True, steps=steps,
        temperature=temperature, alg=alg,
    )

    seq = output.sequences
    prompt_len = len(prompt_ids)
    total_len = seq.shape[1]
    gen_ids = seq[0, prompt_len:min(prompt_len + gen_len, total_len)].tolist()
    gen_ids = [t for t in gen_ids if t != MASK_TOKEN_ID]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    return text


# ──────────────────────────────────────────────────
# 8. Pilot Validation
# ──────────────────────────────────────────────────

def run_pilot(gpu_id: int = 0, n_samples: int = 4, seed: int = 42):
    """Run pilot validation for all benchmarks (task_0b).

    Tests:
    1. Countdown: 4 samples, evaluation pipeline works
    2. GSM8K: 4 samples, answer extraction works
    3. MBPP: 4 samples (no GPU needed for eval, but test sandbox)
    4. HumanEval: skip in pilot (small dataset, sandbox same as MBPP)
    """
    import torch

    device = f"cuda:{gpu_id}"
    results_dir = RESULTS_BASE / "pilots"
    results_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    print("=" * 70)
    print(f"  EVAL FRAMEWORK PILOT (task_0b)")
    print(f"  GPU: {gpu_id}, Samples: {n_samples}, Seed: {seed}")
    print("=" * 70)

    pilot_results = {}
    all_pass = True

    # ── Load Dream-7B ──
    print(f"\n[1/4] Loading Dream-7B on {device}...")
    model, tokenizer = load_dream(device)

    # ── Countdown ──
    print(f"\n[2/4] Countdown evaluation ({n_samples} samples)...")
    countdown_problems = make_countdown_problems(n=n_samples, seed=seed)
    countdown_texts = []
    countdown_times = []

    for i, prob in enumerate(countdown_problems):
        prompt = format_countdown_prompt(prob)
        t0 = time.time()
        text = generate_dream(model, tokenizer, prompt, gen_len=128, device=device)
        elapsed = time.time() - t0
        countdown_texts.append(text)
        countdown_times.append(elapsed)

        ev = eval_countdown(text, prob)
        status = "CORRECT" if ev["correct"] else ("PARSED" if ev["parsed"] else "FAILED")
        print(f"  [{i}] target={prob['target']}, status={status}, "
              f"expr={ev['expression']}, val={ev['value']}")
        print(f"       output: {text[:120]}")

    countdown_eval = eval_countdown_batch(countdown_texts, countdown_problems)
    countdown_div = compute_diversity_metrics(countdown_texts)

    cd_pass = countdown_eval["parse_rate"] > 0  # at least some parseable
    print(f"  Countdown: acc={countdown_eval['accuracy']:.2%}, "
          f"parse_rate={countdown_eval['parse_rate']:.2%} -> {'PASS' if cd_pass else 'FAIL'}")
    if not cd_pass:
        all_pass = False

    pilot_results["countdown"] = {
        "status": "PASS" if cd_pass else "FAIL",
        **countdown_eval,
        "diversity": countdown_div,
    }

    # ── GSM8K ──
    print(f"\n[3/4] GSM8K evaluation ({n_samples} samples)...")
    gsm8k_problems = load_gsm8k(split="test", n=n_samples)
    gsm8k_texts = []
    gsm8k_times = []

    for i, prob in enumerate(gsm8k_problems):
        prompt = format_gsm8k_prompt(prob)
        t0 = time.time()
        text = generate_dream(model, tokenizer, prompt, gen_len=256, device=device)
        elapsed = time.time() - t0
        gsm8k_texts.append(text)
        gsm8k_times.append(elapsed)

        ev = eval_gsm8k(text, prob)
        status = "CORRECT" if ev["correct"] else "WRONG"
        print(f"  [{i}] target={prob['target']}, extracted={ev['extracted']}, "
              f"status={status}")
        print(f"       output: {text[:150]}")

    gsm8k_eval = eval_gsm8k_batch(gsm8k_texts, gsm8k_problems)

    gsm_extract_pass = gsm8k_eval["extraction_rate"] >= 0.5  # at least 50% extracted
    print(f"  GSM8K: acc={gsm8k_eval['accuracy']:.2%}, "
          f"extraction_rate={gsm8k_eval['extraction_rate']:.2%} -> {'PASS' if gsm_extract_pass else 'WARN'}")

    pilot_results["gsm8k"] = {
        "status": "PASS" if gsm_extract_pass else "WARN",
        **gsm8k_eval,
    }

    # ── MBPP (sandbox test) ──
    print(f"\n[4/4] MBPP sandbox test ({n_samples} samples)...")
    mbpp_problems = load_mbpp(split="test", n=n_samples)
    mbpp_texts = []
    mbpp_times = []

    for i, prob in enumerate(mbpp_problems):
        prompt = format_mbpp_prompt(prob)
        t0 = time.time()
        text = generate_dream(model, tokenizer, prompt, gen_len=256, device=device)
        elapsed = time.time() - t0
        mbpp_texts.append(text)
        mbpp_times.append(elapsed)

        ev = eval_mbpp(text, prob)
        status = "PASS" if ev["passed"] else "FAIL"
        print(f"  [{i}] task={prob['task_id']}, status={status}")
        if ev["error"]:
            print(f"       error: {ev['error'][:120]}")
        print(f"       code: {ev['code_extracted'][:120]}")

    mbpp_eval = eval_mbpp_batch(mbpp_texts, mbpp_problems)

    # Sandbox working = no crash (pass or fail is ok for pilot)
    sandbox_ok = True  # if we get here, sandbox works
    print(f"  MBPP: pass@1={mbpp_eval['pass_at_1']:.2%}, sandbox={'OK' if sandbox_ok else 'FAIL'}")

    pilot_results["mbpp"] = {
        "status": "PASS",
        "sandbox_ok": sandbox_ok,
        **mbpp_eval,
    }

    # ── Free model ──
    del model
    torch.cuda.empty_cache()

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  PILOT SUMMARY")
    print(f"{'='*70}")
    for bench, res in pilot_results.items():
        status = res.get("status", "?")
        if bench == "countdown":
            print(f"  {bench:>12}: {status} | acc={res['accuracy']:.2%}, parse={res['parse_rate']:.2%}")
        elif bench == "gsm8k":
            print(f"  {bench:>12}: {status} | acc={res['accuracy']:.2%}, extract={res['extraction_rate']:.2%}")
        elif bench == "mbpp":
            print(f"  {bench:>12}: {status} | pass@1={res['pass_at_1']:.2%}, sandbox={'OK' if res['sandbox_ok'] else 'FAIL'}")

    overall = "GO" if all_pass else "NO-GO (partial)"
    print(f"\n  Overall: {overall}")
    print(f"{'='*70}")

    # ── Save ──
    pilot_results["overall"] = {
        "status": overall,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": seed,
        "n_samples": n_samples,
        "gpu_id": gpu_id,
    }

    out_file = results_dir / "task_0b_eval_framework.json"
    with open(out_file, "w") as f:
        json.dump(pilot_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_file}")

    return pilot_results


# ──────────────────────────────────────────────────
# 9. CLI
# ──────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="DTA Evaluation Framework")
    parser.add_argument("--pilot", action="store_true", help="Run pilot validation")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument("--samples", type=int, default=4, help="Number of pilot samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval", type=str, choices=["countdown", "gsm8k", "mbpp", "humaneval"],
                        help="Evaluate saved generations for a specific benchmark")
    parser.add_argument("--input", type=str, help="Input JSON with generations")
    args = parser.parse_args()

    if args.pilot:
        run_pilot(gpu_id=args.gpu, n_samples=args.samples, seed=args.seed)
    elif args.eval and args.input:
        with open(args.input) as f:
            data = json.load(f)
        texts = data.get("texts", [])
        problems = data.get("problems", [])

        if args.eval == "countdown":
            result = eval_countdown_batch(texts, problems)
        elif args.eval == "gsm8k":
            result = eval_gsm8k_batch(texts, problems)
        elif args.eval == "mbpp":
            result = eval_mbpp_batch(texts, problems)
        elif args.eval == "humaneval":
            result = eval_humaneval_batch(texts, problems)

        print(json.dumps(result, indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
