#!/usr/bin/env python3
"""
AR Baseline Honest Comparison: Qwen2.5-7B-Instruct.

Uses:
  - vLLM for greedy inference (batch=1, batch=8) -- best AR throughput
  - HuggingFace assisted generation for speculative decoding (batch=1 only)
    because Qwen2.5-7B (vocab=152064) and Qwen2.5-0.5B (vocab=151936) have
    different vocabulary sizes, which vLLM's speculative decoding rejects.
    HF's Universal Assisted Decoding handles cross-tokenizer generation.

Benchmarks: 200 GSM8K + 100 MATH500, seed=42

Usage:
    CUDA_VISIBLE_DEVICES=6 conda run -n sibyl_dlm-acceleration python ar_baseline_vllm_v2.py
"""

import os
import sys
import re
import gc
import json
import time
import random
import traceback
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime

import torch
import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
TASK_ID = "ar_baseline_comparison"
# Use resolved path instead of symlink to avoid race conditions
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/iter_002")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "ar_comparison"
SHARED = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/dlm-acceleration/shared")

MODEL_PATH = str(SHARED / "checkpoints" / "qwen2.5-7b-instruct")
DRAFT_MODEL_PATH = str(SHARED / "checkpoints" / "qwen2.5-0.5b")

GSM8K_PATH = str(SHARED / "datasets" / "gsm8k" / "test.json")
MATH500_PATH = str(SHARED / "datasets" / "math500" / "test.json")

LLADA_BASELINE = {
    "gsm8k": {"accuracy": 0.7122, "tps": 31.01},
    "math500": {"accuracy": 0.1107, "tps": 79.22},
}

SAMPLES_GSM8K = 200
SAMPLES_MATH500 = 100
SEED = 42
N_WARMUP = 5
MAX_NEW_TOKENS = 512


# ── Markers ────────────────────────────────────────────────────────────────────

def write_pid(task_id: str, results_dir: Path):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{task_id}.pid").write_text(str(os.getpid()))

def report_progress(task_id: str, results_dir: Path, step: int, total: int,
                    metric: dict = None):
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"{task_id}_PROGRESS.json").write_text(json.dumps({
        "task_id": task_id, "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total, "loss": None,
        "metric": metric or {}, "updated_at": datetime.now().isoformat(),
    }))

def mark_done(task_id: str, results_dir: Path, status="success", summary=""):
    results_dir.mkdir(parents=True, exist_ok=True)
    pid_file = results_dir / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    fp = {}
    pf = results_dir / f"{task_id}_PROGRESS.json"
    if pf.exists():
        try: fp = json.loads(pf.read_text())
        except: pass
    (results_dir / f"{task_id}_DONE").write_text(json.dumps({
        "task_id": task_id, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    }))


# ── Datasets ───────────────────────────────────────────────────────────────────

def load_gsm8k(n: int, seed: int = 42) -> List[Dict]:
    with open(GSM8K_PATH) as f: data = json.load(f)
    if n < len(data): data = random.Random(seed).sample(data, n)
    return data

def load_math500(n: int, seed: int = 42) -> List[Dict]:
    with open(MATH500_PATH) as f: data = json.load(f)
    if n < len(data): data = random.Random(seed).sample(data, n)
    return data

def build_gsm8k_chat(q: str) -> List[Dict]:
    return [
        {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step. End your answer with 'The answer is X.' where X is the final numerical answer."},
        {"role": "user", "content": q},
    ]

def build_math500_chat(p: str) -> List[Dict]:
    return [
        {"role": "system", "content": "You are a helpful math tutor. Solve the problem step by step. Put your final answer in \\boxed{}."},
        {"role": "user", "content": p},
    ]


# ── Answer Extraction ──────────────────────────────────────────────────────────

def extract_gsm8k_answer(text: str) -> Optional[str]:
    m = re.search(r"[Tt]he answer is\s+(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    m = re.search(r"####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)", text)
    if m: return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d+(?:,\d+)*(?:\.\d+)?", text)
    return nums[-1].replace(",", "") if nums else None

def gsm8k_exact_match(pred: str, gold: str) -> bool:
    p, g = extract_gsm8k_answer(pred), extract_gsm8k_answer(gold)
    if p is None or g is None: return False
    try: return abs(float(p) - float(g)) < 1e-6
    except ValueError: return p.strip() == g.strip()

def extract_math500_answer(text: str) -> Optional[str]:
    ms = re.findall(r"\\boxed\{([^}]*)\}", text)
    if ms: return ms[-1].strip()
    m = re.search(r"[Tt]he answer is\s+(.+?)(?:\.|$)", text)
    return m.group(1).strip() if m else None

def math500_exact_match(pred: str, gold: str) -> bool:
    p = extract_math500_answer(pred)
    g = gold.strip()
    if p is None: return False
    p = p.replace(" ", "").replace("\\,", "").strip()
    g = g.replace(" ", "").replace("\\,", "").strip()
    gm = re.match(r"\\boxed\{(.+)\}", g)
    if gm: g = gm.group(1).strip()
    if p == g: return True
    try: return abs(float(p) - float(g)) < 1e-6
    except: return False


# ── QAS / Speedup ──────────────────────────────────────────────────────────────

def compute_qas(speedup, acc_ret): return speedup * acc_ret
def compute_speedup(tps, baseline_tps): return tps / baseline_tps if baseline_tps > 0 else 0


# ── GPU Profile ────────────────────────────────────────────────────────────────

def get_gpu_profile() -> Dict:
    if not torch.cuda.is_available(): return {}
    props = torch.cuda.get_device_properties(0)
    total = getattr(props, 'total_memory', None) or getattr(props, 'total_mem', 0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": total // 1024**2,
        "vram_used_mb": torch.cuda.memory_allocated(0) // 1024**2,
        "vram_reserved_mb": torch.cuda.memory_reserved(0) // 1024**2,
    }


# ── vLLM Evaluation ───────────────────────────────────────────────────────────

def evaluate_vllm(engine, dataset, ds_name, batch_size, tokenizer):
    """Evaluate using vLLM."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    total = len(dataset)
    correct = 0
    tps_list = []
    samples_out = []
    total_tokens = 0
    total_time = 0.0

    print(f"\n[{ds_name.upper()}] vLLM eval: {total} samples, batch={batch_size}")

    # Build all prompts
    prompts = []
    for item in dataset:
        msgs = build_gsm8k_chat(item["question"]) if ds_name == "gsm8k" \
            else build_math500_chat(item["problem"])
        prompts.append(tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True))

    for bs in range(0, total, batch_size):
        be = min(bs + batch_size, total)
        bp = prompts[bs:be]
        bd = dataset[bs:be]

        try:
            t0 = time.time()
            outputs = engine.generate(bp, sp)
            elapsed = time.time() - t0

            batch_tok = sum(len(o.outputs[0].token_ids) for o in outputs)
            batch_tps = batch_tok / elapsed if elapsed > 0 else 0

            for i, (out, item) in enumerate(zip(outputs, bd)):
                idx = bs + i
                text = out.outputs[0].text
                n_tok = len(out.outputs[0].token_ids)

                if ds_name == "gsm8k":
                    ok = gsm8k_exact_match(text, item["answer"])
                    gold = extract_gsm8k_answer(item["answer"])
                    pred = extract_gsm8k_answer(text)
                    kf = ("question", item["question"][:100])
                else:
                    ok = math500_exact_match(text, item["answer"])
                    gold = item["answer"]
                    pred = extract_math500_answer(text)
                    kf = ("problem", item["problem"][:100])

                if ok: correct += 1
                total_tokens += n_tok
                total_time += elapsed / len(bp)
                if idx >= N_WARMUP: tps_list.append(batch_tps)

                if idx < 10:
                    samples_out.append({
                        "id": idx, kf[0]: kf[1],
                        "gold_answer": gold, "predicted_answer": pred,
                        "correct": ok, "tps": round(batch_tps, 2),
                        "n_tokens": n_tok,
                        "prediction_snippet": text[:200],
                    })
        except Exception as e:
            print(f"  [ERROR] {bs}-{be}: {e}")
            for i, item in enumerate(bd):
                k = "question" if ds_name == "gsm8k" else "problem"
                samples_out.append({"id": bs+i, k: "", "correct": False, "error": str(e)})

        if be % 20 == 0 or be == total:
            acc = correct / be if be > 0 else 0
            avg = np.mean(tps_list) if tps_list else 0
            print(f"  [{be}/{total}] acc={acc:.3f}, tps={avg:.1f}")
            report_progress(TASK_ID, RESULTS_DIR, be, total,
                            {f"{ds_name}_acc": acc, "tps": float(avg)})

    return {
        "dataset": ds_name, "n_samples": total, "correct": correct,
        "exact_match": correct / total if total > 0 else 0,
        "avg_tps": float(np.mean(tps_list)) if tps_list else 0,
        "tps_std": float(np.std(tps_list)) if len(tps_list) > 1 else 0,
        "total_tokens": total_tokens,
        "total_gen_time_s": round(total_time, 2),
        "batch_size": batch_size, "samples": samples_out[:10],
    }


# ── HF Speculative Evaluation ─────────────────────────────────────────────────

def evaluate_hf_speculative(dataset, ds_name, model, draft_model,
                            tokenizer, draft_tokenizer):
    """Evaluate using HF assisted generation (batch=1 only)."""
    total = len(dataset)
    correct = 0
    tps_list = []
    samples_out = []

    print(f"\n[{ds_name.upper()}] HF spec eval: {total} samples, batch=1")

    for idx, item in enumerate(dataset):
        msgs = build_gsm8k_chat(item["question"]) if ds_name == "gsm8k" \
            else build_math500_chat(item["problem"])
        prompt = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True)

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                           max_length=3584).to("cuda")
        input_len = inputs["input_ids"].shape[1]

        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(
                **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                assistant_model=draft_model,
                tokenizer=tokenizer,
                assistant_tokenizer=draft_tokenizer,
            )
        torch.cuda.synchronize()
        latency = time.time() - t0

        new_toks = out[0][input_len:]
        n_tok = len(new_toks)
        text = tokenizer.decode(new_toks, skip_special_tokens=True)
        tps = n_tok / latency if latency > 0 else 0

        if ds_name == "gsm8k":
            ok = gsm8k_exact_match(text, item["answer"])
            gold = extract_gsm8k_answer(item["answer"])
            pred = extract_gsm8k_answer(text)
            kf = ("question", item["question"][:100])
        else:
            ok = math500_exact_match(text, item["answer"])
            gold = item["answer"]
            pred = extract_math500_answer(text)
            kf = ("problem", item["problem"][:100])

        if ok: correct += 1
        if idx >= N_WARMUP: tps_list.append(tps)

        if idx < 10:
            samples_out.append({
                "id": idx, kf[0]: kf[1],
                "gold_answer": gold, "predicted_answer": pred,
                "correct": ok, "tps": round(tps, 2), "n_tokens": n_tok,
                "prediction_snippet": text[:200],
            })

        if (idx + 1) % 20 == 0 or (idx + 1) == total:
            acc = correct / (idx + 1)
            avg = np.mean(tps_list) if tps_list else 0
            print(f"  [{idx+1}/{total}] acc={acc:.3f}, tps={avg:.1f}")
            report_progress(TASK_ID, RESULTS_DIR, idx+1, total,
                            {f"{ds_name}_acc": acc, "tps": float(avg)})

    return {
        "dataset": ds_name, "n_samples": total, "correct": correct,
        "exact_match": correct / total if total > 0 else 0,
        "avg_tps": float(np.mean(tps_list)) if tps_list else 0,
        "tps_std": float(np.std(tps_list)) if len(tps_list) > 1 else 0,
        "total_tokens": 0,
        "total_gen_time_s": 0.0,
        "batch_size": 1, "samples": samples_out[:10],
    }


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    start_time = datetime.now()
    import vllm
    print("=" * 70)
    print("AR Baseline Honest Comparison (Qwen2.5-7B-Instruct)")
    print(f"  vLLM {vllm.__version__} (greedy) + HF (speculative)")
    print(f"  Task: {TASK_ID} | Seed: {SEED}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Benchmarks: {SAMPLES_GSM8K} GSM8K + {SAMPLES_MATH500} MATH500")
    print("=" * 70)

    write_pid(TASK_ID, RESULTS_DIR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

    gsm8k_data = load_gsm8k(SAMPLES_GSM8K, SEED)
    math500_data = load_math500(SAMPLES_MATH500, SEED)
    print(f"\nLoaded {len(gsm8k_data)} GSM8K + {len(math500_data)} MATH500")

    all_configs = []
    gpu_profiles = {}

    # ────────────────────────────────────────────────────────────────────
    # Phase 1: vLLM greedy (batch=1 and batch=8)
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Phase 1: vLLM greedy inference")
    print("=" * 50)

    from vllm import LLM
    engine = LLM(
        model=MODEL_PATH,
        gpu_memory_utilization=0.90,
        max_model_len=4096,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=SEED,
    )
    gpu_profiles["vllm_greedy"] = get_gpu_profile()
    print(f"GPU after vLLM load: {gpu_profiles['vllm_greedy']}")

    # Config 1: greedy batch=1
    print("\n--- Config 1: greedy batch=1 ---")
    g1 = evaluate_vllm(engine, gsm8k_data, "gsm8k", 1, tokenizer)
    m1 = evaluate_vllm(engine, math500_data, "math500", 1, tokenizer)
    all_configs.append({
        "config_name": "qwen7b_greedy_b1",
        "model": "Qwen2.5-7B-Instruct", "speculative": False,
        "batch_size": 1, "engine": "vllm",
        "gsm8k": g1, "math500": m1,
    })
    print(f"  GSM8K: {g1['exact_match']:.3f} @ {g1['avg_tps']:.1f} TPS")
    print(f"  MATH500: {m1['exact_match']:.3f} @ {m1['avg_tps']:.1f} TPS")

    # Config 2: greedy batch=8
    print("\n--- Config 2: greedy batch=8 ---")
    g8 = evaluate_vllm(engine, gsm8k_data, "gsm8k", 8, tokenizer)
    m8 = evaluate_vllm(engine, math500_data, "math500", 8, tokenizer)
    all_configs.append({
        "config_name": "qwen7b_greedy_b8",
        "model": "Qwen2.5-7B-Instruct", "speculative": False,
        "batch_size": 8, "engine": "vllm",
        "gsm8k": g8, "math500": m8,
    })
    print(f"  GSM8K: {g8['exact_match']:.3f} @ {g8['avg_tps']:.1f} TPS")
    print(f"  MATH500: {m8['exact_match']:.3f} @ {m8['avg_tps']:.1f} TPS")

    # Cleanup vLLM
    del engine
    gc.collect(); torch.cuda.empty_cache()
    try:
        import ray
        if ray.is_initialized(): ray.shutdown()
    except: pass
    time.sleep(5)

    # ────────────────────────────────────────────────────────────────────
    # Phase 2: HF speculative decoding (batch=1 only)
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 50)
    print("Phase 2: HF speculative decoding (batch=1)")
    print("  NOTE: vLLM speculative requires matching vocab sizes.")
    print("  Qwen2.5-7B (152064) != Qwen2.5-0.5B (151936)")
    print("  Using HF Universal Assisted Decoding (cross-tokenizer)")
    print("=" * 50)

    from transformers import AutoModelForCausalLM

    print("\nLoading target model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    model.eval()

    print("Loading draft model...")
    draft = AutoModelForCausalLM.from_pretrained(
        DRAFT_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="cuda",
        trust_remote_code=True)
    draft.eval()

    draft_tok = AutoTokenizer.from_pretrained(DRAFT_MODEL_PATH, trust_remote_code=True)
    if draft_tok.pad_token is None: draft_tok.pad_token = draft_tok.eos_token

    gpu_profiles["hf_speculative"] = get_gpu_profile()
    print(f"GPU after HF load: {gpu_profiles['hf_speculative']}")

    # Config 3: speculative batch=1
    print("\n--- Config 3: speculative batch=1 ---")
    gs1 = evaluate_hf_speculative(gsm8k_data, "gsm8k", model, draft,
                                   tokenizer, draft_tok)
    ms1 = evaluate_hf_speculative(math500_data, "math500", model, draft,
                                   tokenizer, draft_tok)
    all_configs.append({
        "config_name": "qwen7b_speculative_b1",
        "model": "Qwen2.5-7B-Instruct", "draft_model": "Qwen2.5-0.5B",
        "speculative": True, "batch_size": 1,
        "engine": "hf_assisted",
        "gsm8k": gs1, "math500": ms1,
    })
    print(f"  GSM8K: {gs1['exact_match']:.3f} @ {gs1['avg_tps']:.1f} TPS")
    print(f"  MATH500: {ms1['exact_match']:.3f} @ {ms1['avg_tps']:.1f} TPS")

    # Config 4: speculative batch=8
    # HF assisted generation does not support batch>1
    print("\n--- Config 4: speculative batch=8 ---")
    print("  SKIPPED: HF assisted generation only supports batch=1")
    print("  vLLM speculative also unavailable (vocab mismatch)")
    all_configs.append({
        "config_name": "qwen7b_speculative_b8",
        "model": "Qwen2.5-7B-Instruct", "draft_model": "Qwen2.5-0.5B",
        "speculative": True, "batch_size": 8,
        "engine": "none",
        "gsm8k": _empty("gsm8k", SAMPLES_GSM8K, 8,
                        "Speculative batch>1 not available: "
                        "vLLM requires matching vocab, HF only supports batch=1"),
        "math500": _empty("math500", SAMPLES_MATH500, 8,
                          "Speculative batch>1 not available"),
        "note": "Speculative decoding at batch=8 is not possible with "
                "mismatched vocabularies (Qwen2.5-7B: 152064 vs 0.5B: 151936). "
                "vLLM rejects vocab mismatch; HF only supports batch=1."
    })

    del model, draft
    gc.collect(); torch.cuda.empty_cache()

    # ────────────────────────────────────────────────────────────────────
    # Comparison Table
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARISON WITH LLaDA-8B-Instruct BASELINE")
    print("=" * 70)

    comparison_table = []
    for cfg in all_configs:
        g_tps = cfg["gsm8k"]["avg_tps"]
        m_tps = cfg["math500"]["avg_tps"]
        g_acc = cfg["gsm8k"]["exact_match"]
        m_acc = cfg["math500"]["exact_match"]

        g_sp = compute_speedup(g_tps, LLADA_BASELINE["gsm8k"]["tps"])
        g_ar = g_acc / LLADA_BASELINE["gsm8k"]["accuracy"] if LLADA_BASELINE["gsm8k"]["accuracy"] > 0 else 0
        g_qas = compute_qas(g_sp, g_ar)

        m_sp = compute_speedup(m_tps, LLADA_BASELINE["math500"]["tps"])
        m_ar = m_acc / LLADA_BASELINE["math500"]["accuracy"] if LLADA_BASELINE["math500"]["accuracy"] > 0 else 0
        m_qas = compute_qas(m_sp, m_ar)

        c_ar = 0.7 * g_ar + 0.3 * m_ar
        c_sp = 0.7 * g_sp + 0.3 * m_sp
        c_qas = compute_qas(c_sp, c_ar)

        row = {
            "config_name": cfg["config_name"],
            "system": f"{'AR+SpecDec' if cfg.get('speculative') else 'AR'} (Qwen7B)",
            "batch_size": cfg["batch_size"],
            "engine": cfg["engine"],
            "gsm8k_acc": g_acc,
            "gsm8k_tps": round(g_tps, 1),
            "gsm8k_speedup_vs_llada": round(g_sp, 2),
            "gsm8k_acc_retention": round(g_ar, 3),
            "gsm8k_qas": round(g_qas, 3),
            "math500_acc": m_acc,
            "math500_tps": round(m_tps, 1),
            "math500_speedup_vs_llada": round(m_sp, 2),
            "math500_acc_retention": round(m_ar, 3),
            "math500_qas": round(m_qas, 3),
            "combined_acc_retention": round(c_ar, 3),
            "combined_speedup": round(c_sp, 2),
            "combined_qas": round(c_qas, 3),
        }
        comparison_table.append(row)

        print(f"\n  {row['config_name']} ({row['engine']}):")
        print(f"    GSM8K:   acc={g_acc:.3f}, TPS={row['gsm8k_tps']:.1f}, "
              f"speedup={row['gsm8k_speedup_vs_llada']:.2f}x")
        print(f"    MATH500: acc={m_acc:.3f}, TPS={row['math500_tps']:.1f}, "
              f"speedup={row['math500_speedup_vs_llada']:.2f}x")
        print(f"    Combined QAS={row['combined_qas']:.3f}")

    llada_ref = {
        "config_name": "llada8b_baseline",
        "system": "DLM (LLaDA-8B)",
        "batch_size": 8, "engine": "custom",
        "gsm8k_acc": LLADA_BASELINE["gsm8k"]["accuracy"],
        "gsm8k_tps": LLADA_BASELINE["gsm8k"]["tps"],
        "gsm8k_speedup_vs_llada": 1.0, "gsm8k_acc_retention": 1.0, "gsm8k_qas": 1.0,
        "math500_acc": LLADA_BASELINE["math500"]["accuracy"],
        "math500_tps": LLADA_BASELINE["math500"]["tps"],
        "math500_speedup_vs_llada": 1.0, "math500_acc_retention": 1.0, "math500_qas": 1.0,
        "combined_acc_retention": 1.0, "combined_speedup": 1.0, "combined_qas": 1.0,
        "note": "From iter_001 full baseline (3 seeds mean, batch=8)",
    }

    # ── Save ──────────────────────────────────────────────────────────────
    end_time = datetime.now()
    dur = (end_time - start_time).total_seconds() / 60

    final = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "Qwen2.5-7B-Instruct",
        "draft_model": "Qwen2.5-0.5B",
        "engine_used": "vllm (greedy) + hf_assisted (speculative)",
        "vllm_version": vllm.__version__,
        "speculative_note": (
            "vLLM speculative decoding rejected due to vocab size mismatch "
            "(Qwen2.5-7B: 152064, Qwen2.5-0.5B: 151936). "
            "HF Universal Assisted Decoding used for batch=1 speculative. "
            "Batch=8 speculative not available."
        ),
        "benchmarks": {"gsm8k_n": SAMPLES_GSM8K, "math500_n": SAMPLES_MATH500},
        "llada_baseline_reference": LLADA_BASELINE,
        "configs": all_configs,
        "comparison_table": comparison_table,
        "llada_reference_row": llada_ref,
        "gpu_profile": gpu_profiles,
        "timing": {
            "start": start_time.isoformat(),
            "end": end_time.isoformat(),
            "duration_min": round(dur, 1),
        },
        "timestamp": end_time.isoformat(),
    }

    out_path = RESULTS_DIR / "ar_baseline.json"
    out_path.write_text(json.dumps(final, indent=2, default=str))
    print(f"\n[SAVED] {out_path}")

    # Summary markdown
    md = [
        "# AR Baseline Comparison (vLLM + HF)", "",
        f"**Date**: {end_time.strftime('%Y-%m-%d %H:%M')}",
        f"**Duration**: {dur:.1f} min",
        f"**Engine**: vLLM {vllm.__version__} (greedy) + HF (speculative)",
        f"**Benchmarks**: {SAMPLES_GSM8K} GSM8K + {SAMPLES_MATH500} MATH500",
        "",
        "## Results", "",
        "| System | Batch | Engine | GSM8K Acc | GSM8K TPS | vs LLaDA | MATH500 Acc | MATH500 TPS | Combined QAS |",
        "|--------|-------|--------|-----------|-----------|----------|-------------|-------------|--------------|",
    ]
    for r in comparison_table:
        md.append(
            f"| {r['system']} | {r['batch_size']} | {r['engine']} | "
            f"{r['gsm8k_acc']:.3f} | {r['gsm8k_tps']:.1f} | "
            f"{r['gsm8k_speedup_vs_llada']:.2f}x | "
            f"{r['math500_acc']:.3f} | {r['math500_tps']:.1f} | "
            f"{r['combined_qas']:.3f} |"
        )
    md.append(
        f"| DLM (LLaDA-8B) | 8 | custom | "
        f"{LLADA_BASELINE['gsm8k']['accuracy']:.3f} | "
        f"{LLADA_BASELINE['gsm8k']['tps']:.1f} | 1.00x | "
        f"{LLADA_BASELINE['math500']['accuracy']:.3f} | "
        f"{LLADA_BASELINE['math500']['tps']:.1f} | 1.000 |"
    )
    md.extend(["", "## Notes", "",
        "- LLaDA baseline: batch=8 (MDM parallel generation), 64 denoising steps",
        "- AR batch=1 is the fair comparison for interactive use",
        "- Speculative decoding: Qwen2.5-0.5B draft, HF Universal Assisted Decoding",
        "- Speculative batch=8 unavailable: vocab size mismatch prevents vLLM spec dec",
    ])
    (RESULTS_DIR / "ar_baseline_summary.md").write_text("\n".join(md))
    print(f"[SAVED] {RESULTS_DIR / 'ar_baseline_summary.md'}")

    # ── gpu_progress.json ────────────────────────────────────────────────
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try: gp = json.loads(gp_path.read_text())
    except: gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
    if TASK_ID not in gp.get("completed", []):
        gp.setdefault("completed", []).append(TASK_ID)
    gp.get("running", {}).pop(TASK_ID, None)
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(dur),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_snapshot": {
            "model": "Qwen2.5-7B-Instruct",
            "draft_model": "Qwen2.5-0.5B",
            "engine": "vllm+hf",
            "vllm_version": vllm.__version__,
            "benchmarks": ["gsm8k", "math500"],
            "samples": {"gsm8k": SAMPLES_GSM8K, "math500": SAMPLES_MATH500},
            "gpu_model": torch.cuda.get_device_name(0),
            "gpu_count": 1,
        },
    }
    gp_path.write_text(json.dumps(gp, indent=2))
    print(f"[UPDATED] {gp_path}")

    summary = (
        f"AR baseline (vLLM {vllm.__version__}+HF): "
        f"greedy_b1={comparison_table[0]['gsm8k_acc']:.3f}@{comparison_table[0]['gsm8k_tps']:.1f}TPS, "
        f"greedy_b8={comparison_table[1]['gsm8k_acc']:.3f}@{comparison_table[1]['gsm8k_tps']:.1f}TPS, "
        f"spec_b1={comparison_table[2]['gsm8k_acc']:.3f}@{comparison_table[2]['gsm8k_tps']:.1f}TPS. "
        f"Duration={dur:.1f}min"
    )
    mark_done(TASK_ID, RESULTS_DIR, "success", summary)

    print(f"\n{'='*70}")
    print(f"COMPLETED in {dur:.1f} minutes")
    print(f"{'='*70}")
    return final


def _empty(ds, n, bs, msg):
    return {
        "dataset": ds, "n_samples": n, "correct": 0, "exact_match": 0.0,
        "avg_tps": 0, "tps_std": 0, "total_tokens": 0, "total_gen_time_s": 0.0,
        "batch_size": bs, "samples": [{"error": msg}],
    }


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] {e}")
        traceback.print_exc()
        mark_done(TASK_ID, RESULTS_DIR, "failed", str(e))
        try:
            gp = json.loads((WORKSPACE / "exp" / "gpu_progress.json").read_text())
            if TASK_ID not in gp.get("failed", []):
                gp.setdefault("failed", []).append(TASK_ID)
            gp.get("running", {}).pop(TASK_ID, None)
            (WORKSPACE / "exp" / "gpu_progress.json").write_text(json.dumps(gp, indent=2))
        except: pass
        sys.exit(1)
