"""
ACA-DLM v9: Multi-model validation.

Tests ReMask-Retry on a SECOND MDLM model (Qwen2.5-Coder-0.5B-diffusion-mdlm)
to demonstrate generalization beyond Qwen3-0.6B.

Uses 64 prompts (subset of 256) for efficiency.
"""
import os, sys, json, math, random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, "/home/ccwang/sibyl_system/src/dllm")
sys.path.insert(0, "/home/ccwang/sibyl_system/exp/code")
import dllm

RESULTS_DIR = Path("/home/ccwang/sibyl_system/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
N_PROMPTS = 64
SEED = 42

MODEL_NAME = "dllm-hub/Qwen2.5-Coder-0.5B-Instruct-diffusion-mdlm-v0.1"


def load_coder_model():
    model_args = SimpleNamespace(
        model_name_or_path=MODEL_NAME, dtype="bfloat16",
        load_in_4bit=False, attn_implementation=None, lora_name_or_path=None,
    )
    model = dllm.utils.get_model(model_args=model_args).eval()
    tokenizer = dllm.utils.get_tokenizer(model_args=model_args)
    return model, tokenizer


def make_coder_prompts(tokenizer, n=64):
    """Create diverse prompts suitable for a code-focused model."""
    questions = [
        # Code questions
        "Write a Python function to check if a number is prime.",
        "Explain the difference between a list and a tuple in Python.",
        "Write a function to reverse a linked list.",
        "What is the time complexity of binary search?",
        "Explain how hash tables work.",
        "Write a Python class for a binary search tree.",
        "What are decorators in Python?",
        "Explain the concept of recursion with an example.",
        "Write a function to find the longest common subsequence.",
        "What is the difference between BFS and DFS?",
        "Explain how garbage collection works in Python.",
        "Write a function to merge two sorted arrays.",
        "What are generators in Python and when should you use them?",
        "Explain the SOLID principles in software engineering.",
        "Write a function to detect a cycle in a linked list.",
        "What is dynamic programming? Give an example.",
        # General knowledge
        "What is machine learning?",
        "Explain how neural networks learn.",
        "What is the difference between supervised and unsupervised learning?",
        "Explain the concept of overfitting in machine learning.",
        "What is gradient descent?",
        "Explain the transformer architecture.",
        "What is attention mechanism in deep learning?",
        "Explain the difference between CNN and RNN.",
        "What is transfer learning?",
        "Explain the bias-variance tradeoff.",
        "What is regularization and why is it important?",
        "Explain how backpropagation works.",
        "What is the vanishing gradient problem?",
        "Explain the concept of embedding in NLP.",
        "What is tokenization in natural language processing?",
        "Explain the difference between precision and recall.",
        # Science and math
        "Explain the theory of general relativity.",
        "What is quantum computing?",
        "Explain how DNA replication works.",
        "What is the Pythagorean theorem?",
        "Explain the concept of entropy in thermodynamics.",
        "What is the Fourier transform?",
        "Explain how photosynthesis works.",
        "What is calculus used for?",
        "Explain the concept of probability distributions.",
        "What is the central limit theorem?",
        "Explain the difference between correlation and causation.",
        "What is Bayes' theorem?",
        "Explain the concept of eigenvalues and eigenvectors.",
        "What is the second law of thermodynamics?",
        "Explain how vaccines work.",
        "What is the Heisenberg uncertainty principle?",
        # Technology
        "What is cloud computing?",
        "Explain how blockchain technology works.",
        "What is containerization in software development?",
        "Explain the concept of microservices architecture.",
        "What is REST API?",
        "Explain how HTTPS encryption works.",
        "What is version control and why is it important?",
        "Explain the concept of continuous integration.",
        "What is database normalization?",
        "Explain the CAP theorem.",
        "What is the difference between SQL and NoSQL databases?",
        "Explain how load balancing works.",
        "What is serverless computing?",
        "Explain the concept of DevOps.",
        "What is infrastructure as code?",
        "Explain how DNS resolution works.",
    ]

    prompts = []
    for q in questions[:n]:
        messages = [{"role": "user", "content": q}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def adaptive_remask_retry(model, tokenizer, prompt, config,
                          n_retries=2, refine_steps=32, max_ratio=0.7, min_ratio=0.3):
    """ReMask-Retry with adaptive ratio."""
    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    with torch.no_grad():
        seq = sampler.sample([prompt], config=config)

    prompt_len = len(prompt)
    gen_len = config.max_new_tokens
    info = {"adaptive_ratio": 0, "avg_conf": 0}

    for retry in range(n_retries):
        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            confs = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)

        gen_confs = confs[0, prompt_len:prompt_len+gen_len].cpu().numpy()
        avg_conf = float(np.mean(gen_confs))

        adaptive_ratio = max_ratio - (max_ratio - min_ratio) * min(avg_conf / 0.5, 1.0)
        adaptive_ratio = max(min_ratio, min(max_ratio, adaptive_ratio))
        info = {"adaptive_ratio": adaptive_ratio, "avg_conf": avg_conf}

        n_remask = int(adaptive_ratio * gen_len)
        if n_remask < 10:
            break

        sorted_indices = np.argsort(gen_confs)
        remask_positions = sorted_indices[:n_remask]

        for mt in ["[MASK]", "<mask>", "<|mask|>"]:
            ids = tokenizer.encode(mt, add_special_tokens=False)
            if len(ids) == 1:
                mtid = ids[0]
                break
        else:
            mtid = 0

        for pos in remask_positions:
            seq[0, prompt_len + pos] = mtid

        refine_config = dllm.core.samplers.MDLMSamplerConfig(
            max_new_tokens=gen_len, steps=refine_steps, temperature=0.2,
            remasking="low_confidence", block_size=32)

        with torch.no_grad():
            seq = sampler.sample([prompt], config=refine_config, x_initial=seq)

    return seq, info


def run_method(method_name, batch_size=32):
    """Run vanilla or adaptive on the Coder model."""
    progress_file = RESULTS_DIR / f"v9_{method_name}_progress.json"

    if progress_file.exists():
        with open(progress_file) as f:
            progress = json.load(f)
    else:
        progress = {"texts": [], "confs": [], "ratios": [], "n_done": 0}

    if progress["n_done"] >= N_PROMPTS:
        print(f"{method_name}: All {N_PROMPTS} done")
        return progress

    model, tokenizer = load_coder_model()
    device = next(model.parameters()).device
    prompts = make_coder_prompts(tokenizer, N_PROMPTS)

    config = dllm.core.samplers.MDLMSamplerConfig(
        max_new_tokens=256, steps=128, temperature=0.2,
        remasking="low_confidence", block_size=32)

    start_idx = progress["n_done"]
    end_idx = min(start_idx + batch_size, N_PROMPTS)
    print(f"\n{method_name}: Prompts {start_idx}-{end_idx-1} of {N_PROMPTS}")

    torch.manual_seed(SEED + start_idx)
    torch.cuda.manual_seed(SEED + start_idx)

    sampler = dllm.core.samplers.MDLMSampler(
        model=model, tokenizer=tokenizer,
        scheduler=dllm.core.schedulers.LinearAlphaScheduler())

    for pi in range(start_idx, end_idx):
        prompt = prompts[pi]

        if method_name == "coder_vanilla":
            with torch.no_grad():
                seq = sampler.sample([prompt], config=config)
            progress["ratios"].append(0)
        elif method_name == "coder_adaptive":
            seq, info = adaptive_remask_retry(model, tokenizer, prompt, config)
            progress["ratios"].append(info["adaptive_ratio"])

        text = dllm.utils.sample_trim(tokenizer, seq.tolist(), [prompt])[0].strip()
        progress["texts"].append(text)

        attention_mask = torch.ones_like(seq)
        with torch.no_grad():
            logits = model(seq, attention_mask=attention_mask).logits
            probs = F.softmax(logits.float(), dim=-1)
            token_conf = torch.gather(probs, dim=-1, index=seq.unsqueeze(-1)).squeeze(-1)
        prompt_t = torch.as_tensor(prompt, dtype=torch.long, device=device)
        pl = prompt_t.shape[0]
        gen_conf = token_conf[0, pl:pl+config.max_new_tokens].mean().item()
        progress["confs"].append(gen_conf)

        if (pi - start_idx + 1) % 16 == 0:
            print(f"  {pi+1}/{N_PROMPTS} done")

    progress["n_done"] = end_idx
    with open(progress_file, "w") as f:
        json.dump(progress, f)
    print(f"  Batch done. Total: {end_idx}/{N_PROMPTS}")
    return progress


def run_ppl_eval(method_name):
    """PPL eval using GPT-2 (cross-family)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    progress_file = RESULTS_DIR / f"v9_{method_name}_progress.json"
    if not progress_file.exists():
        print(f"No progress for {method_name}")
        return None

    with open(progress_file) as f:
        progress = json.load(f)

    if progress["n_done"] < N_PROMPTS:
        print(f"{method_name}: Only {progress['n_done']}/{N_PROMPTS}")
        return None

    ppl_file = RESULTS_DIR / f"v9_{method_name}_ppls.json"
    if ppl_file.exists():
        with open(ppl_file) as f:
            return json.load(f)

    print(f"\nPPL eval for {method_name}")
    device = torch.device("cuda:0")

    # Use both Qwen3-0.6B and GPT-2
    evaluators = {
        "qwen3": ("Qwen/Qwen3-0.6B", {}),
        "gpt2": ("/home/ccwang/sibyl_system/models/gpt2", {}),
    }

    all_results = {}
    for eval_name, (model_path, kwargs) in evaluators.items():
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        eval_model = AutoModelForCausalLM.from_pretrained(
            model_path, dtype=torch.bfloat16, trust_remote_code=True).to(device)
        eval_model.eval()

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ppls = []
        for text in progress["texts"]:
            if len(text.strip()) < 10:
                ppls.append(None)
                continue
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
            with torch.no_grad():
                out = eval_model(**enc, labels=enc["input_ids"])
            ppls.append(math.exp(min(out.loss.item(), 20)))

        del eval_model
        torch.cuda.empty_cache()

        valid_ppls = [float(p) for p in ppls if p is not None]
        all_results[eval_name] = {
            "ppl_mean": float(np.mean(valid_ppls)),
            "ppl_median": float(np.median(valid_ppls)),
            "ppl_std": float(np.std(valid_ppls)),
            "n_valid": len(valid_ppls),
            "all_ppls": valid_ppls,
        }
        print(f"  {eval_name}: mean={all_results[eval_name]['ppl_mean']:.3f}, "
              f"median={all_results[eval_name]['ppl_median']:.3f}")

    result = {
        "method": method_name,
        "n_prompts": N_PROMPTS,
        "model": MODEL_NAME,
        "evaluators": all_results,
        "avg_ratio": float(np.mean([r for r in progress.get("ratios", []) if r > 0])) if any(r > 0 for r in progress.get("ratios", [])) else 0,
    }
    with open(ppl_file, "w") as f:
        json.dump(result, f, indent=2)
    return result


def print_summary():
    from scipy import stats as scipy_stats

    v_file = RESULTS_DIR / "v9_coder_vanilla_ppls.json"
    a_file = RESULTS_DIR / "v9_coder_adaptive_ppls.json"

    if not v_file.exists() or not a_file.exists():
        print("Missing results files")
        return

    v = json.load(open(v_file))
    a = json.load(open(a_file))

    print(f"\n{'='*70}")
    print(f"MULTI-MODEL VALIDATION: {MODEL_NAME}")
    print(f"{'='*70}")

    for eval_name in ["qwen3", "gpt2"]:
        vp = v["evaluators"][eval_name]["all_ppls"]
        ap = a["evaluators"][eval_name]["all_ppls"]

        n = min(len(vp), len(ap))
        stat, p = scipy_stats.wilcoxon(vp[:n], ap[:n])
        win = sum(1 for x, y in zip(vp, ap) if y < x)
        v_med = np.median(vp)
        a_med = np.median(ap)
        delta = (a_med - v_med) / v_med * 100

        v_cat = sum(1 for x in vp if x > 100)
        a_cat = sum(1 for x in ap if x > 100)

        print(f"\n  {eval_name} evaluator:")
        print(f"    Vanilla:  median={v_med:.3f}, mean={np.mean(vp):.3f}, cat={v_cat}")
        print(f"    Adaptive: median={a_med:.3f}, mean={np.mean(ap):.3f}, cat={a_cat}")
        print(f"    Delta(median): {delta:+.1f}%, Wilcoxon p={p:.2e}, win_rate={win/n*100:.1f}%")

    print(f"\n  Avg adaptive ratio: {a.get('avg_ratio', 0):.3f}")
    print(f"{'='*70}")


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "auto"

    if task == "auto":
        # Run vanilla first, then adaptive, then eval, then summary
        for method in ["coder_vanilla", "coder_adaptive"]:
            p_file = RESULTS_DIR / f"v9_{method}_progress.json"
            if p_file.exists():
                with open(p_file) as f:
                    p = json.load(f)
                if p["n_done"] < N_PROMPTS:
                    run_method(method)
                    return
            else:
                run_method(method)
                return

        # All generation done, run eval
        for method in ["coder_vanilla", "coder_adaptive"]:
            ppl_file = RESULTS_DIR / f"v9_{method}_ppls.json"
            if not ppl_file.exists():
                run_ppl_eval(method)
                return

        print_summary()

    elif task == "summary":
        print_summary()
    elif task.startswith("eval_"):
        run_ppl_eval(task[5:])
    else:
        run_method(task)
        p_file = RESULTS_DIR / f"v9_{task}_progress.json"
        with open(p_file) as f:
            p = json.load(f)
        if p["n_done"] >= N_PROMPTS:
            print(f"\n{task} complete! Running PPL eval...")
            run_ppl_eval(task)


if __name__ == "__main__":
    main()
