"""
Evaluation Harness for DLM Acceleration Study.

Supports evaluation on:
- GSM8K (math reasoning, exact match after 8-shot prompt)
- HumanEval (code generation, pass@1)
- MATH500 (hard math, exact match)
- MBPP (code, pass@1)

Usage:
    python eval_harness.py \
        --model_path /path/to/llada \
        --mode baseline \
        --dataset gsm8k \
        --n_samples 200 \
        --seed 42 \
        --output_dir exp/results/baseline
"""
import os
import sys
import re
import json
import time
import random
import argparse
import traceback
import subprocess
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple

import torch
import numpy as np

_CODE_DIR = Path(__file__).parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

# PID reporting for system monitor
def write_pid(task_id: str, results_dir: str):
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    (Path(results_dir) / f"{task_id}.pid").write_text(str(os.getpid()))

def report_progress(task_id, results_dir, epoch, total_epochs, step=0,
                    total_steps=0, loss=None, metric=None):
    from datetime import datetime
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(task_id, results_dir, status="success", summary=""):
    from datetime import datetime
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ── Dataset Loaders ─────────────────────────────────────────────────────────

def load_gsm8k(data_dir: str, split: str = "test", n_samples: Optional[int] = None,
               seed: int = 42) -> List[Dict]:
    path = Path(data_dir) / f"{split}.json"
    with open(path) as f:
        data = json.load(f)
    if n_samples and n_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, n_samples)
    return data


def load_humaneval(data_dir: str, n_samples: Optional[int] = None, seed: int = 42) -> List[Dict]:
    path = Path(data_dir) / "test.json"
    with open(path) as f:
        data = json.load(f)
    if n_samples and n_samples < len(data):
        rng = random.Random(seed)
        data = rng.sample(data, n_samples)
    return data


# ── Prompt Builders ──────────────────────────────────────────────────────────

# 8-shot GSM8K prompt examples (from official LLaDA eval)
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

def build_gsm8k_prompt(question: str) -> str:
    return GSM8K_8SHOT + f"Question: {question}\nAnswer:"


def build_humaneval_prompt(problem: Dict) -> str:
    """HumanEval 0-shot: provide function signature + docstring."""
    return problem["prompt"]


# ── Metric Extractors ────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    """Extract final numeric answer from GSM8K solution text."""
    # Look for "The answer is X" pattern
    match = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    # Look for #### X pattern
    match = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if match:
        return match.group(1).replace(",", "")
    # Last number in text
    numbers = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def gsm8k_exact_match(pred: str, gold: str) -> bool:
    """Compare predicted and gold answers (normalize numbers)."""
    p = extract_gsm8k_answer(pred)
    g = extract_gsm8k_answer(gold)
    if p is None or g is None:
        return False
    try:
        return abs(float(p) - float(g)) < 1e-6
    except ValueError:
        return p.strip() == g.strip()


def humaneval_pass_at_1(code_completion: str, problem: Dict) -> bool:
    """Check if code passes HumanEval test cases (run in subprocess)."""
    # Build complete function
    full_code = problem["prompt"] + code_completion + "\n" + problem["test"]
    # Add entry point call
    full_code += f"\ncheck({problem['entry_point']})\n"

    try:
        result = subprocess.run(
            ["python", "-c", full_code],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


# ── Evaluator ────────────────────────────────────────────────────────────────

class DLMEvaluator:
    """Unified evaluator for DLM acceleration experiments."""

    def __init__(
        self,
        model_path: str,
        mode: str = "baseline",
        mode_kwargs: Optional[Dict] = None,
        device: str = "cuda",
        seed: int = 42,
    ):
        self.model_path = model_path
        self.mode = mode
        self.mode_kwargs = mode_kwargs or {}
        self.device = device
        self.seed = seed
        self.wrapper = None

    def load_model(self):
        from inference_wrapper import build_wrapper
        self.wrapper = build_wrapper(
            model_path=self.model_path,
            mode=self.mode,
            device=self.device,
            **self.mode_kwargs,
        )

    def evaluate_gsm8k(
        self,
        data: List[Dict],
        steps: int = 64,
        gen_length: int = 256,
        block_length: int = 32,
        task_id: str = "eval",
        results_dir: str = "results",
        n_warmup: int = 5,
    ) -> Dict:
        """
        Evaluate on GSM8K. Returns metrics dict including TPS, exact_match.
        """
        write_pid(task_id, results_dir)

        correct = 0
        total = len(data)
        all_results = []
        tps_list = []

        print(f"[Evaluator] GSM8K: {total} samples, mode={self.mode}")

        for i, item in enumerate(data):
            prompt = build_gsm8k_prompt(item["question"])
            gold = item["answer"]

            try:
                result = self.wrapper.generate(
                    prompts=prompt,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    apply_chat_template=True,
                )
                pred_text = result.generated_text[0]
                is_correct = gsm8k_exact_match(pred_text, gold)
                if is_correct:
                    correct += 1

                # Skip warmup samples for TPS
                if i >= n_warmup:
                    tps_list.append(result.tps)

                all_results.append({
                    "id": i,
                    "question": item["question"],
                    "gold": gold,
                    "prediction": pred_text,
                    "correct": is_correct,
                    "tps": result.tps,
                    "metadata": result.metadata,
                })

            except Exception as e:
                print(f"[Evaluator] Error on sample {i}: {e}")
                all_results.append({
                    "id": i, "question": item["question"], "gold": gold,
                    "prediction": "", "correct": False, "tps": 0.0,
                    "error": str(e),
                })

            if (i + 1) % 10 == 0:
                acc = correct / (i + 1)
                avg_tps = np.mean(tps_list) if tps_list else 0.0
                print(f"  [{i+1}/{total}] acc={acc:.3f}, avg_tps={avg_tps:.1f}")
                report_progress(task_id, results_dir, epoch=i+1, total_epochs=total,
                                metric={"accuracy": acc, "tps": avg_tps})

        exact_match = correct / total if total > 0 else 0.0
        avg_tps = float(np.mean(tps_list)) if tps_list else 0.0

        metrics = {
            "dataset": "gsm8k",
            "mode": self.mode,
            "mode_kwargs": self.mode_kwargs,
            "n_samples": total,
            "exact_match": exact_match,
            "correct": correct,
            "avg_tps": avg_tps,
            "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
            "steps": steps,
            "gen_length": gen_length,
        }

        # Save results
        out_path = Path(results_dir) / f"{task_id}_gsm8k_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "metrics": metrics,
            "samples": all_results[:20],  # save first 20 as examples
        }, indent=2))

        mark_done(task_id, results_dir, status="success",
                  summary=f"GSM8K exact_match={exact_match:.3f} tps={avg_tps:.1f}")
        return metrics

    def evaluate_humaneval(
        self,
        data: List[Dict],
        steps: int = 64,
        gen_length: int = 256,
        block_length: int = 32,
        task_id: str = "eval",
        results_dir: str = "results",
        n_warmup: int = 3,
    ) -> Dict:
        """
        Evaluate on HumanEval. Returns pass@1, TPS.
        """
        write_pid(task_id, results_dir)

        passed = 0
        total = len(data)
        all_results = []
        tps_list = []

        print(f"[Evaluator] HumanEval: {total} problems, mode={self.mode}")

        for i, problem in enumerate(data):
            prompt = build_humaneval_prompt(problem)

            try:
                result = self.wrapper.generate(
                    prompts=prompt,
                    steps=steps,
                    gen_length=gen_length,
                    block_length=block_length,
                    apply_chat_template=False,  # HumanEval: no instruct wrapping
                )
                code_completion = result.generated_text[0]
                is_pass = humaneval_pass_at_1(code_completion, problem)
                if is_pass:
                    passed += 1

                if i >= n_warmup:
                    tps_list.append(result.tps)

                all_results.append({
                    "task_id": problem["task_id"],
                    "completion": code_completion[:500],
                    "passed": is_pass,
                    "tps": result.tps,
                })

            except Exception as e:
                print(f"[Evaluator] Error on problem {i}: {e}")
                all_results.append({
                    "task_id": problem.get("task_id", str(i)),
                    "completion": "", "passed": False, "tps": 0.0,
                    "error": str(e),
                })

            if (i + 1) % 10 == 0:
                p1 = passed / (i + 1)
                avg_tps = np.mean(tps_list) if tps_list else 0.0
                print(f"  [{i+1}/{total}] pass@1={p1:.3f}, avg_tps={avg_tps:.1f}")
                report_progress(task_id, results_dir, epoch=i+1, total_epochs=total,
                                metric={"pass_at_1": p1, "tps": avg_tps})

        pass_at_1 = passed / total if total > 0 else 0.0
        avg_tps = float(np.mean(tps_list)) if tps_list else 0.0

        metrics = {
            "dataset": "humaneval",
            "mode": self.mode,
            "mode_kwargs": self.mode_kwargs,
            "n_samples": total,
            "pass_at_1": pass_at_1,
            "passed": passed,
            "avg_tps": avg_tps,
            "tps_std": float(np.std(tps_list)) if tps_list else 0.0,
            "steps": steps,
            "gen_length": gen_length,
        }

        out_path = Path(results_dir) / f"{task_id}_humaneval_results.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "metrics": metrics,
            "samples": all_results[:20],
        }, indent=2))

        mark_done(task_id, results_dir, status="success",
                  summary=f"HumanEval pass@1={pass_at_1:.3f} tps={avg_tps:.1f}")
        return metrics


# ── Setup Verification ───────────────────────────────────────────────────────

def run_setup_verification(
    model_path: str,
    gsm8k_dir: str,
    humaneval_dir: str,
    output_path: str,
    device: str = "cuda",
    n_samples: int = 5,
) -> Dict:
    """
    Verify that model loads and produces valid output on 5 GSM8K examples.
    Returns verification dict saved to output_path.
    """
    from datetime import datetime
    results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": model_path,
        "device": device,
        "steps": [],
        "status": "unknown",
    }

    # Step 1: Check model files exist
    results["steps"].append({"step": "check_model_files"})
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        results["steps"][-1]["status"] = "FAIL"
        results["steps"][-1]["error"] = f"config.json not found at {model_path}"
        results["status"] = "FAIL"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, indent=2))
        return results
    results["steps"][-1]["status"] = "PASS"

    # Step 2: GPU available
    results["steps"].append({"step": "check_gpu"})
    if not torch.cuda.is_available():
        results["steps"][-1]["status"] = "FAIL"
        results["steps"][-1]["error"] = "CUDA not available"
        results["status"] = "FAIL"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, indent=2))
        return results
    gpu_info = {
        "count": torch.cuda.device_count(),
        "name": torch.cuda.get_device_name(0),
        "vram_total_mb": torch.cuda.get_device_properties(0).total_memory // 1024**2,
    }
    results["steps"][-1]["status"] = "PASS"
    results["steps"][-1]["gpu_info"] = gpu_info

    # Step 3: Load model
    results["steps"].append({"step": "load_model"})
    try:
        t0 = time.time()
        from inference_wrapper import build_wrapper
        wrapper = build_wrapper(model_path=model_path, mode="baseline", device=device)
        load_time = time.time() - t0
        results["steps"][-1]["status"] = "PASS"
        results["steps"][-1]["load_time_sec"] = round(load_time, 1)
    except Exception as e:
        results["steps"][-1]["status"] = "FAIL"
        results["steps"][-1]["error"] = str(e)
        results["status"] = "FAIL"
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(json.dumps(results, indent=2))
        return results

    # Step 4: Check VRAM
    results["steps"].append({"step": "check_vram"})
    try:
        vram_used = torch.cuda.memory_allocated(0) // 1024**2
        vram_reserved = torch.cuda.memory_reserved(0) // 1024**2
        results["steps"][-1].update({
            "status": "PASS",
            "vram_used_mb": vram_used,
            "vram_reserved_mb": vram_reserved,
        })
    except Exception as e:
        results["steps"][-1] = {"step": "check_vram", "status": "WARN", "error": str(e)}

    # Step 5: Run inference on 5 GSM8K examples
    results["steps"].append({"step": "inference_gsm8k"})
    try:
        gsm8k = load_gsm8k(gsm8k_dir, split="test", n_samples=n_samples, seed=42)
        prompts = [item["question"] for item in gsm8k]

        gen_results = []
        for i, (prompt, item) in enumerate(zip(prompts, gsm8k)):
            res = wrapper.generate(
                prompts=prompt,
                steps=32,        # fewer steps for quick verification
                gen_length=128,
                block_length=32,
                apply_chat_template=True,
            )
            pred = res.generated_text[0]
            correct = gsm8k_exact_match(pred, item["answer"])
            gen_results.append({
                "question": prompt[:100] + "...",
                "gold": item["answer"][-50:],
                "prediction": pred[:200],
                "correct": correct,
                "tps": round(res.tps, 2),
            })
            print(f"  Sample {i+1}: tps={res.tps:.1f}, correct={correct}")

        results["steps"][-1]["status"] = "PASS"
        results["steps"][-1]["samples"] = gen_results
        results["steps"][-1]["avg_tps"] = float(np.mean([r["tps"] for r in gen_results]))

    except Exception as e:
        results["steps"][-1]["status"] = "FAIL"
        results["steps"][-1]["error"] = str(e)
        results["steps"][-1]["traceback"] = traceback.format_exc()
        results["status"] = "FAIL"

    # Final status
    failed = [s for s in results["steps"] if s.get("status") == "FAIL"]
    results["status"] = "FAIL" if failed else "PASS"
    results["summary"] = f"{len(results['steps']) - len(failed)}/{len(results['steps'])} steps passed"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(results, indent=2))
    print(f"\n[Verification] Status: {results['status']}")
    print(f"[Verification] Summary: {results['summary']}")
    return results


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DLM Evaluation Harness")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--mode", default="baseline",
                        choices=["baseline", "m1_prefix", "m1_dllm", "m1_d2cache",
                                 "m2_sched", "m3_ar", "igsd"])
    parser.add_argument("--dataset", default="gsm8k",
                        choices=["gsm8k", "humaneval", "verify"])
    parser.add_argument("--gsm8k_dir", default=None)
    parser.add_argument("--humaneval_dir", default=None)
    parser.add_argument("--n_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_dir", default="exp/results")
    parser.add_argument("--task_id", default="eval")
    parser.add_argument("--device", default="cuda")
    # Mode kwargs
    parser.add_argument("--entropy_threshold", type=float, default=1.0)
    parser.add_argument("--step_jump", type=int, default=2)
    parser.add_argument("--guidance_weight", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.85)
    parser.add_argument("--t_draft", type=int, default=8)
    args = parser.parse_args()

    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    mode_kwargs = {}
    if args.mode in ("m1_prefix", "m1_dllm", "m1_d2cache"):
        mode_kwargs["entropy_threshold"] = args.entropy_threshold
    elif args.mode == "m2_sched":
        mode_kwargs["step_jump"] = args.step_jump
    elif args.mode == "m3_ar":
        mode_kwargs["guidance_weight"] = args.guidance_weight
    elif args.mode == "igsd":
        mode_kwargs["tau"] = args.tau
        mode_kwargs["t_draft"] = args.t_draft

    if args.dataset == "verify":
        run_setup_verification(
            model_path=args.model_path,
            gsm8k_dir=args.gsm8k_dir or "shared/datasets/gsm8k",
            humaneval_dir=args.humaneval_dir or "shared/datasets/humaneval",
            output_path=os.path.join(args.output_dir, "setup_verification.json"),
            device=args.device,
            n_samples=args.n_samples,
        )
        return

    evaluator = DLMEvaluator(
        model_path=args.model_path,
        mode=args.mode,
        mode_kwargs=mode_kwargs,
        device=args.device,
        seed=args.seed,
    )
    evaluator.load_model()

    if args.dataset == "gsm8k":
        gsm8k_dir = args.gsm8k_dir or "shared/datasets/gsm8k"
        data = load_gsm8k(gsm8k_dir, n_samples=args.n_samples, seed=args.seed)
        metrics = evaluator.evaluate_gsm8k(
            data,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            task_id=args.task_id,
            results_dir=args.output_dir,
        )
        print(json.dumps(metrics, indent=2))

    elif args.dataset == "humaneval":
        humaneval_dir = args.humaneval_dir or "shared/datasets/humaneval"
        data = load_humaneval(humaneval_dir, n_samples=args.n_samples, seed=args.seed)
        metrics = evaluator.evaluate_humaneval(
            data,
            steps=args.steps,
            gen_length=args.gen_length,
            block_length=args.block_length,
            task_id=args.task_id,
            results_dir=args.output_dir,
        )
        print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
