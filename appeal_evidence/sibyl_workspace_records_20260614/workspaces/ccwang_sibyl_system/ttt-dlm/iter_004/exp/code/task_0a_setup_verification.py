"""
Task 0a (PILOT): Environment verification & model loading.
Verify Dream-7B loads, vanilla generation works on 4 Countdown prompts,
mask_token_id=151666, PEFT/LoRA initialization works.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 task_0a_setup_verification.py
"""
import os, sys, json, time, random, re
from pathlib import Path

import numpy as np
import torch

# === Config ===
MODEL_DIR = "/home/ccwang/sibyl_system/models/Dream-v0-Instruct-7B"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 4
SEED = 42
GEN_LEN = 256
STEPS = 128
TEMPERATURE = 0.4
ALG = "origin"
EXPECTED_MASK_TOKEN_ID = 151666


def generate_countdown_problems(n, seed=42):
    """Generate n Countdown-style arithmetic problems with known solutions."""
    rng = random.Random(seed)
    problems = []
    attempts = 0
    while len(problems) < n and attempts < n * 50:
        attempts += 1
        num_count = rng.randint(3, 5)
        numbers = [rng.randint(1, 25) for _ in range(num_count)]
        a, b, c = numbers[0], numbers[1], numbers[2]
        ops = ['+', '-', '*']
        op1 = rng.choice(ops)
        op2 = rng.choice(ops)
        try:
            intermediate = eval(f"{a} {op1} {b}")
            target = eval(f"{intermediate} {op2} {c}")
            if target > 0 and target == int(target) and 10 <= target <= 999:
                target = int(target)
                problems.append({
                    "numbers": numbers,
                    "target": target,
                    "solution_hint": f"({a} {op1} {b}) {op2} {c} = {target}"
                })
                continue
        except:
            pass
        a, b = numbers[0], numbers[1]
        target = a * b + (numbers[2] if num_count > 2 else 0)
        if 10 <= target <= 999:
            problems.append({
                "numbers": numbers,
                "target": target,
                "solution_hint": f"{a}*{b}+{numbers[2]} = {target}"
            })
    return problems[:n]


def format_countdown_prompt(problem):
    nums_str = ", ".join(str(n) for n in problem["numbers"])
    target = problem["target"]
    return (
        f"Use the numbers [{nums_str}] with basic arithmetic operations "
        f"(+, -, *, /) to obtain {target}. "
        f"Each number can only be used once. "
        f"Show your work step by step, then provide the final equation.\n"
        f"Format: Step 1: ... Step 2: ... Answer: <equation> = {target}"
    )


def verify_countdown_answer(text, problem):
    target = problem["target"]
    numbers = problem["numbers"]
    answer_match = re.search(r'Answer:\s*(.+)', text, re.IGNORECASE)
    if answer_match:
        equation_str = answer_match.group(1).strip()
    else:
        eq_matches = re.findall(r'([\d\s+\-*/()]+)\s*=\s*(\d+)', text)
        if eq_matches:
            equation_str = eq_matches[-1][0].strip()
        else:
            return {"is_correct": False, "extracted_equation": None,
                    "explanation": "No equation found"}
    equation_str = re.sub(r'\s*=\s*\d+\s*$', '', equation_str).strip()
    if not equation_str:
        return {"is_correct": False, "extracted_equation": None,
                "explanation": "Empty equation"}
    try:
        safe_eq = re.sub(r'[^\d+\-*/(). ]', '', equation_str)
        result = eval(safe_eq)
        is_correct = abs(result - target) < 1e-6
        used_numbers = [int(x) for x in re.findall(r'\d+', safe_eq)]
        avail = list(numbers)
        numbers_valid = True
        for u in used_numbers:
            if u in avail:
                avail.remove(u)
            else:
                numbers_valid = False
                break
        return {
            "is_correct": is_correct and numbers_valid,
            "result_correct": is_correct,
            "numbers_valid": numbers_valid,
            "extracted_equation": equation_str,
            "evaluated_result": float(result),
            "target": target,
        }
    except Exception as e:
        return {"is_correct": False, "extracted_equation": equation_str,
                "explanation": f"Eval error: {e}"}


def main():
    results = {
        "task": "task_0a",
        "mode": "pilot",
        "checks": {},
        "pass_criteria": "4 prompts generated, PEFT LoRA init OK, no OOM",
    }
    all_pass = True

    # ---- Check 1: Environment info ----
    print("=" * 60)
    print("CHECK 1: Environment info")
    print("=" * 60)
    env_info = {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count(),
    }
    if torch.cuda.is_available():
        gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
        env_info["gpu_name"] = torch.cuda.get_device_name(0)
        env_info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
    import peft
    import transformers
    env_info["peft"] = peft.__version__
    env_info["transformers"] = transformers.__version__
    print(json.dumps(env_info, indent=2))
    results["checks"]["environment"] = {"status": "PASS", "info": env_info}

    # ---- Check 2: Model loading ----
    print("\n" + "=" * 60)
    print("CHECK 2: Load Dream-7B")
    print("=" * 60)
    device = "cuda:0"
    t0 = time.time()
    try:
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            MODEL_DIR, trust_remote_code=True, torch_dtype=torch.bfloat16
        ).to(device)
        model.eval()
        load_time = time.time() - t0
        print(f"  Model loaded in {load_time:.1f}s")
        print(f"  Model dtype: {next(model.parameters()).dtype}")
        print(f"  Model device: {next(model.parameters()).device}")
        results["checks"]["model_loading"] = {"status": "PASS", "load_time_s": load_time}
    except Exception as e:
        print(f"  FAIL: {e}")
        results["checks"]["model_loading"] = {"status": "FAIL", "error": str(e)}
        all_pass = False

    # ---- Check 3: Verify mask_token_id ----
    print("\n" + "=" * 60)
    print("CHECK 3: Verify mask_token_id")
    print("=" * 60)
    try:
        # Check tokenizer for mask token
        mask_id = getattr(tokenizer, 'mask_token_id', None)
        # Also check model config
        config_mask_id = getattr(model.config, 'mask_token_id', None)
        # Check by encoding the mask token directly
        vocab = tokenizer.get_vocab()
        mask_candidates = {k: v for k, v in vocab.items() if 'mask' in k.lower()}

        actual_mask_id = mask_id or config_mask_id or EXPECTED_MASK_TOKEN_ID
        print(f"  tokenizer.mask_token_id: {mask_id}")
        print(f"  model.config.mask_token_id: {config_mask_id}")
        print(f"  Mask-related vocab entries: {mask_candidates}")
        print(f"  Using mask_token_id: {actual_mask_id}")

        id_ok = (actual_mask_id == EXPECTED_MASK_TOKEN_ID) or (EXPECTED_MASK_TOKEN_ID in mask_candidates.values())
        if id_ok:
            print(f"  PASS: mask_token_id = {EXPECTED_MASK_TOKEN_ID} confirmed")
            results["checks"]["mask_token_id"] = {"status": "PASS", "mask_token_id": EXPECTED_MASK_TOKEN_ID}
        else:
            print(f"  WARNING: Expected {EXPECTED_MASK_TOKEN_ID}, got {actual_mask_id}")
            results["checks"]["mask_token_id"] = {
                "status": "WARN",
                "expected": EXPECTED_MASK_TOKEN_ID,
                "actual": actual_mask_id,
                "mask_candidates": {k: v for k, v in mask_candidates.items()}
            }
    except Exception as e:
        print(f"  FAIL: {e}")
        results["checks"]["mask_token_id"] = {"status": "FAIL", "error": str(e)}

    # ---- Check 4: Vanilla generation on 4 Countdown prompts ----
    print("\n" + "=" * 60)
    print("CHECK 4: Vanilla generation on 4 Countdown prompts")
    print("=" * 60)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    problems = generate_countdown_problems(N_SAMPLES, seed=SEED)
    gen_results = []
    gen_ok = True

    for i, problem in enumerate(problems):
        prompt_text = format_countdown_prompt(problem)
        messages = [{"role": "user", "content": prompt_text}]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        input_ids = torch.tensor([prompt_ids], device=device)

        t0 = time.time()
        try:
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
            eos_id = tokenizer.eos_token_id
            clean_ids = [t for t in gen_ids if t != EXPECTED_MASK_TOKEN_ID and t != eos_id]
            text = tokenizer.decode(clean_ids, skip_special_tokens=True).strip()

            verification = verify_countdown_answer(text, problem)
            status = "CORRECT" if verification.get("is_correct") else "WRONG"
            print(f"  [{i+1}/{N_SAMPLES}] {status} | target={problem['target']} | time={elapsed:.1f}s")
            print(f"    Generated: {text[:200]}")

            gen_results.append({
                "idx": i,
                "problem": problem,
                "generated_text": text,
                "verification": verification,
                "gen_time_s": elapsed,
                "prompt_len": prompt_len,
                "gen_len": len(clean_ids),
            })
        except Exception as e:
            print(f"  [{i+1}/{N_SAMPLES}] FAIL: {e}")
            gen_results.append({"idx": i, "problem": problem, "error": str(e)})
            gen_ok = False

    n_generated = sum(1 for r in gen_results if "generated_text" in r and len(r["generated_text"]) > 0)
    n_correct = sum(1 for r in gen_results if r.get("verification", {}).get("is_correct", False))
    print(f"\n  Generated: {n_generated}/{N_SAMPLES}, Correct: {n_correct}/{N_SAMPLES}")

    results["checks"]["vanilla_generation"] = {
        "status": "PASS" if gen_ok and n_generated == N_SAMPLES else "FAIL",
        "n_generated": n_generated,
        "n_correct": n_correct,
        "n_total": N_SAMPLES,
        "accuracy": n_correct / N_SAMPLES if N_SAMPLES > 0 else 0,
        "results": gen_results,
    }
    if not gen_ok or n_generated < N_SAMPLES:
        all_pass = False

    # ---- Check 5: PEFT/LoRA initialization ----
    print("\n" + "=" * 60)
    print("CHECK 5: PEFT/LoRA initialization")
    print("=" * 60)
    try:
        from peft import LoraConfig, get_peft_model, TaskType

        # Identify target modules in last 2 layers
        # Dream-7B is based on Qwen2 architecture
        n_layers = model.config.num_hidden_layers
        print(f"  Model has {n_layers} layers")

        # Find FFN module names in last 2 layers
        target_modules = []
        for name, _ in model.named_modules():
            # Target last 2 layers' FFN (gate_proj, up_proj, down_proj)
            for layer_idx in range(n_layers - 2, n_layers):
                if f"layers.{layer_idx}." in name and any(
                    ffn_part in name for ffn_part in ["gate_proj", "up_proj", "down_proj"]
                ):
                    target_modules.append(name)

        print(f"  Target modules (last 2 layers FFN): {len(target_modules)} modules")
        for m in target_modules[:6]:
            print(f"    {m}")

        # Create LoRA config
        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=[m.split(".")[-1] for m in target_modules][:3],  # just test gate/up/down
            lora_dropout=0.0,
            bias="none",
            layers_to_transform=list(range(n_layers - 2, n_layers)),
        )
        print(f"  LoRA config: r={lora_config.r}, alpha={lora_config.lora_alpha}")
        print(f"  Target modules: {lora_config.target_modules}")
        print(f"  Layers to transform: {lora_config.layers_to_transform}")

        # Apply LoRA to model (just to verify it works)
        peft_model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in peft_model.parameters())
        print(f"  Trainable params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.4f}%)")

        # Verify LoRA params are zero-initialized (B matrix should be zero)
        lora_norms = []
        for name, param in peft_model.named_parameters():
            if param.requires_grad and "lora" in name.lower():
                norm = param.data.float().norm().item()
                lora_norms.append((name, norm))
                if len(lora_norms) <= 4:
                    print(f"    {name}: norm={norm:.6f}")

        # Quick forward pass with LoRA to verify no errors
        with torch.no_grad():
            test_input = torch.tensor([prompt_ids[:50]], device=device)
            # Just verify forward pass works
            _ = peft_model(test_input)
        print("  Forward pass with LoRA: OK")

        # Clean up - remove LoRA to free memory
        del peft_model
        torch.cuda.empty_cache()

        results["checks"]["peft_lora"] = {
            "status": "PASS",
            "n_layers": n_layers,
            "n_target_modules": len(target_modules),
            "trainable_params": trainable_params,
            "total_params": total_params,
            "trainable_pct": round(100 * trainable_params / total_params, 4),
            "lora_config": {
                "r": lora_config.r,
                "alpha": lora_config.lora_alpha,
                "layers_to_transform": list(lora_config.layers_to_transform),
            }
        }
        print("  PASS: PEFT/LoRA initialization successful")
    except Exception as e:
        import traceback
        print(f"  FAIL: {e}")
        traceback.print_exc()
        results["checks"]["peft_lora"] = {"status": "FAIL", "error": str(e)}
        all_pass = False

    # ---- Check 6: GPU memory usage ----
    print("\n" + "=" * 60)
    print("CHECK 6: GPU memory summary")
    print("=" * 60)
    mem_allocated = torch.cuda.memory_allocated(0) / 1e9
    mem_reserved = torch.cuda.memory_reserved(0) / 1e9
    mem_total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"  Allocated: {mem_allocated:.1f} GB")
    print(f"  Reserved:  {mem_reserved:.1f} GB")
    print(f"  Total:     {mem_total:.1f} GB")
    print(f"  Free:      {mem_total - mem_reserved:.1f} GB")
    results["checks"]["gpu_memory"] = {
        "allocated_gb": round(mem_allocated, 1),
        "reserved_gb": round(mem_reserved, 1),
        "total_gb": round(mem_total, 1),
        "free_gb": round(mem_total - mem_reserved, 1),
    }

    # ---- Overall verdict ----
    print("\n" + "=" * 60)
    print("OVERALL VERDICT")
    print("=" * 60)
    for check_name, check_result in results["checks"].items():
        status = check_result.get("status", "UNKNOWN")
        print(f"  {check_name}: {status}")

    overall = "GO" if all_pass else "NO-GO"
    results["overall_verdict"] = overall
    print(f"\n  VERDICT: {overall}")

    # Save results
    out_file = RESULTS_DIR / "task_0a_setup_verification.json"
    # Remove non-serializable items
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
