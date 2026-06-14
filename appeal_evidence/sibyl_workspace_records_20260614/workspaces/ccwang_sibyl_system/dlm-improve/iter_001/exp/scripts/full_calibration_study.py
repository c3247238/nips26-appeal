#!/usr/bin/env python3
"""
DLM Calibration Profiling (H1 + H2) — FULL mode
Task: calibration_study

Runs on the FULL GSM8K test set (~1319 samples) with batched inference.
Measures calibration of LLaDA-8B-Instruct during denoising with TWO methods:

METHOD A — "Self-consistency calibration":
  Run standard denoising. At each step, record confidence for each masked position.
  After full denoising completes, compare each step's top-1 prediction against the
  FINAL committed token at that position.

METHOD B — "Oracle calibration" (teacher-forced):
  Given the reference answer, mask a fraction of tokens (matching each denoising stage),
  and measure whether the model can recover the masked tokens.

Both methods compute ECE per stage band and entropy-error correlation.
Fits isotonic regression calibrators and saves them for downstream tasks.

Optimizations over pilot:
  - Batched inference (auto-detected max batch size)
  - Left-padded variable-length prompts
  - torch.compile for forward pass
  - Resumable: saves intermediate checkpoints

Usage:
    CUDA_VISIBLE_DEVICES=5 python full_calibration_study.py
"""

import os
import sys
import gc
import json
import time
import math
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from datasets import load_dataset

warnings.filterwarnings("ignore")

# Flush print output immediately
import functools
print = functools.partial(print, flush=True)

# ── Config ────────────────────────────────────────────────────────────
TASK_ID = "calibration_study"
MODEL_PATH = "/home/ccwang/sibyl_system/models/LLaDA-8B-Instruct"
RESULTS_DIR = Path("/home/ccwang/sibyl_system/projects/dlm-improve/exp/results")
FULL_RESULTS_DIR = RESULTS_DIR / "full"
SEED = 42
NUM_SAMPLES = None  # Full dataset
NUM_STEPS = 64
GEN_LENGTH = 256
MASK_TOKEN_ID = 126336
EOS_TOKEN_ID = 126081
DEVICE = "cuda"
CHECKPOINT_INTERVAL = 100  # Save intermediate results every N samples

# Stage bands: fraction of positions still masked
STAGE_BANDS = [(0.80, 1.00), (0.50, 0.80), (0.20, 0.50), (0.00, 0.20)]
STAGE_BAND_NAMES = ["early_80_100", "mid_50_80", "mid_20_50", "late_0_20"]
ECE_BINS = 10

# Oracle calibration: masking ratios to test
ORACLE_MASK_RATIOS = [0.9, 0.7, 0.5, 0.3, 0.1]

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class NumpyEncoder(json.JSONEncoder):
    """Handle numpy types in JSON serialization."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def write_pid():
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0,
                    loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }, cls=NumpyEncoder))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }, cls=NumpyEncoder))


def cosine_schedule(t, T):
    """Fraction unmasked after step t (0..T). t=0 -> 0, t=T -> 1."""
    return 1.0 - math.cos(math.pi * t / (2 * T))


def compute_ece(confidences, accuracies, n_bins=10):
    if len(confidences) == 0:
        return 0.0, [], [], []
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_ece = 0.0
    bin_confs, bin_accs, bin_counts = [], [], []
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences >= lo) & (confidences <= hi) if i == n_bins - 1 \
            else (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            bin_confs.append((lo + hi) / 2)
            bin_accs.append(0.0)
            bin_counts.append(0)
            continue
        bc = confidences[mask].mean()
        ba = accuracies[mask].mean()
        bn = mask.sum()
        bin_ece += (bn / len(confidences)) * abs(ba - bc)
        bin_confs.append(float(bc))
        bin_accs.append(float(ba))
        bin_counts.append(int(bn))
    return float(bin_ece), bin_confs, bin_accs, bin_counts


def assign_stage_band(masking_ratio):
    for i, (lo, hi) in enumerate(STAGE_BANDS):
        if lo <= masking_ratio <= hi:
            return i
    return 0 if masking_ratio > 1.0 else 3


def get_gsm8k_prompts(tokenizer, num_samples=None):
    dataset = load_dataset("gsm8k", "main", split="test")
    total = len(dataset) if num_samples is None else min(num_samples, len(dataset))
    samples = []
    for idx in range(total):
        question = dataset[idx]["question"]
        answer = dataset[idx]["answer"]
        final_answer = answer.split("####")[-1].strip() if "####" in answer else answer
        prompt = f"Solve the following math problem step by step.\n\nQuestion: {question}\n\nAnswer:"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(" " + answer, add_special_tokens=False)
        samples.append({
            "idx": idx,
            "prompt_ids": prompt_ids,
            "answer_ids": answer_ids[:GEN_LENGTH],
            "final_answer": final_answer,
            "question": question,
        })
    return samples


# ══════════════════════════════════════════════════════════════════════
# Batch size configuration
# ══════════════════════════════════════════════════════════════════════

# Skip auto-detection (CUBLAS errors corrupt CUDA state).
# Conservative estimate for 97GB VRAM, model ~16GB, seq_len ~330:
# Each sample uses ~(seq_len * hidden_dim * layers * dtype_size) for activations
# LLaDA-8B: hidden=4096, layers=32, bf16=2 bytes
# Per sample ~330 * 4096 * 32 * 2 * 3 (attn+ffn+grad) ~ 250MB
# Available: ~80GB -> ~320 samples. Use conservative 16 for safety.
ORACLE_BATCH_SIZE = 16


# ══════════════════════════════════════════════════════════════════════
# METHOD A: Self-Consistency Calibration (batched)
# ══════════════════════════════════════════════════════════════════════

def run_denoising_self_consistency_single(model, prompt_ids, gen_len, num_steps=64):
    """
    Run denoising on a single sample and compare each step's predictions
    to the FINAL tokens. Returns per-step records.
    """
    device = next(model.parameters()).device
    prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device)
    input_ids = torch.cat([
        prompt_tensor,
        torch.full((gen_len,), MASK_TOKEN_ID, dtype=torch.long, device=device)
    ]).unsqueeze(0)

    prompt_len = len(prompt_ids)
    gen_start = prompt_len
    gen_end = prompt_len + gen_len

    # Accumulate per-band data directly (avoid storing huge records list)
    band_data = {i: {"confidences": [], "accuracies": [], "entropies": [], "is_errors": []}
                 for i in range(len(STAGE_BANDS))}

    # Store predictions at each step for is_correct computation
    step_preds_by_pos = defaultdict(list)  # pos -> [(step, conf, token, entropy, masking_ratio)]

    for step in range(num_steps):
        frac_unmasked_prev = cosine_schedule(step, num_steps) if step > 0 else 0.0
        frac_unmasked_curr = cosine_schedule(step + 1, num_steps)

        gen_region = input_ids[0, gen_start:gen_end]
        is_masked = (gen_region == MASK_TOKEN_ID)
        num_masked = is_masked.sum().item()

        if num_masked == 0:
            break

        masking_ratio = num_masked / gen_len

        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[0]

        gen_logits = logits[gen_start:gen_end]
        masked_indices = torch.where(is_masked)[0]
        masked_logits = gen_logits[masked_indices]

        probs = F.softmax(masked_logits, dim=-1)
        top1_conf, top1_token = probs.max(dim=-1)
        log_probs = F.log_softmax(masked_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)

        # Store predictions for later is_correct computation
        confs_cpu = top1_conf.float().cpu().numpy()
        tokens_cpu = top1_token.cpu().numpy()
        ents_cpu = entropy.float().cpu().numpy()
        indices_cpu = masked_indices.cpu().numpy()

        for j in range(len(indices_cpu)):
            pos = indices_cpu[j]
            step_preds_by_pos[pos].append(
                (step, confs_cpu[j], tokens_cpu[j], ents_cpu[j], masking_ratio)
            )

        # Unmask highest-confidence
        num_to_unmask = max(1, int(round((frac_unmasked_curr - frac_unmasked_prev) * gen_len)))
        num_to_unmask = min(num_to_unmask, num_masked)
        _, topk_indices = top1_conf.topk(num_to_unmask)
        positions_to_unmask = masked_indices[topk_indices]
        tokens_to_place = top1_token[topk_indices]
        input_ids[0, gen_start + positions_to_unmask] = tokens_to_place

    # Final committed tokens
    final_tokens = input_ids[0, gen_start:gen_end].cpu().numpy()

    # Now compute is_correct for each prediction
    for pos, preds in step_preds_by_pos.items():
        final_tok = final_tokens[pos]
        for (step_idx, conf, pred_tok, ent, mr) in preds:
            is_correct = int(pred_tok == final_tok)
            band_idx = assign_stage_band(mr)
            band_data[band_idx]["confidences"].append(conf)
            band_data[band_idx]["accuracies"].append(is_correct)
            band_data[band_idx]["entropies"].append(ent)
            band_data[band_idx]["is_errors"].append(1 - is_correct)

    return band_data


# ══════════════════════════════════════════════════════════════════════
# METHOD B: Oracle (Teacher-Forced) Calibration (batched)
# ══════════════════════════════════════════════════════════════════════

def run_oracle_calibration_batched(model, batch_samples, mask_ratio, tokenizer,
                                    max_batch_size=16):
    """
    Process multiple samples at a given mask ratio using batched inference.
    Left-pads to uniform length.
    """
    device = next(model.parameters()).device
    all_records = defaultdict(lambda: {"confidences": [], "accuracies": [],
                                        "entropies": [], "is_errors": []})

    # Group into sub-batches
    for start_idx in range(0, len(batch_samples), max_batch_size):
        sub_batch = batch_samples[start_idx:start_idx + max_batch_size]
        bs = len(sub_batch)

        # Build full sequences (prompt + answer) and find max length
        full_seqs = []
        prompt_lens = []
        answer_lens = []
        for s in sub_batch:
            full = s["prompt_ids"] + s["answer_ids"]
            full_seqs.append(full)
            prompt_lens.append(len(s["prompt_ids"]))
            answer_lens.append(len(s["answer_ids"]))

        max_len = max(len(seq) for seq in full_seqs)

        # Left-pad with MASK_TOKEN_ID (will be ignored for loss)
        # Use pad_token_id if available, else use 0
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        padded = torch.full((bs, max_len), pad_id, dtype=torch.long, device=device)
        for i, seq in enumerate(full_seqs):
            offset = max_len - len(seq)
            padded[i, offset:] = torch.tensor(seq, dtype=torch.long, device=device)

        # Apply masking to answer regions
        rng = np.random.RandomState(SEED + int(mask_ratio * 1000))
        mask_positions_list = []  # per-sample list of masked positions (in padded coords)
        ref_tokens_list = []

        for i, s in enumerate(sub_batch):
            offset = max_len - len(full_seqs[i])
            gen_start = offset + prompt_lens[i]
            gen_len = answer_lens[i]
            num_to_mask = max(1, int(round(mask_ratio * gen_len)))

            positions_in_answer = rng.choice(gen_len, size=num_to_mask, replace=False)
            abs_positions = positions_in_answer + gen_start

            ref_toks = padded[i, abs_positions].clone()
            padded[i, abs_positions] = MASK_TOKEN_ID

            mask_positions_list.append(abs_positions)
            ref_tokens_list.append(ref_toks)

        # Forward pass with OOM fallback to batch_size=1
        try:
            with torch.no_grad():
                outputs = model(input_ids=padded)
                logits = outputs.logits  # [bs, max_len, vocab]
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            err_str = str(e).lower()
            if "out of memory" in err_str or "cublas" in err_str:
                print(f"    OOM at batch_size={bs}, falling back to batch_size=1")
                del padded
                torch.cuda.empty_cache()
                gc.collect()
                # Process one by one
                for i, s in enumerate(sub_batch):
                    full = s["prompt_ids"] + s["answer_ids"]
                    single = torch.tensor(full, dtype=torch.long, device=device).unsqueeze(0)
                    gen_start_i = len(s["prompt_ids"])
                    gen_len_i = len(s["answer_ids"])
                    num_to_mask_i = max(1, int(round(mask_ratio * gen_len_i)))
                    rng_i = np.random.RandomState(SEED + int(mask_ratio * 1000) + start_idx + i)
                    pos_i = rng_i.choice(gen_len_i, size=num_to_mask_i, replace=False)
                    ref_toks_i = single[0, gen_start_i + pos_i].clone()
                    single[0, gen_start_i + pos_i] = MASK_TOKEN_ID
                    with torch.no_grad():
                        out_i = model(input_ids=single)
                        logits_i = out_i.logits[0]
                    masked_logits_i = logits_i[gen_start_i + pos_i]
                    probs_i = F.softmax(masked_logits_i, dim=-1)
                    top1_conf_i, top1_token_i = probs_i.max(dim=-1)
                    log_probs_i = F.log_softmax(masked_logits_i, dim=-1)
                    entropy_i = -(probs_i * log_probs_i).sum(dim=-1)
                    is_correct_i = (top1_token_i == ref_toks_i)
                    key = f"mask_{int(mask_ratio*100)}pct"
                    all_records[key]["confidences"].extend(top1_conf_i.float().cpu().numpy().tolist())
                    all_records[key]["accuracies"].extend(is_correct_i.cpu().numpy().astype(int).tolist())
                    all_records[key]["entropies"].extend(entropy_i.float().cpu().numpy().tolist())
                    all_records[key]["is_errors"].extend((1 - is_correct_i.cpu().numpy()).astype(int).tolist())
                    del single, out_i, logits_i
                torch.cuda.empty_cache()
                continue  # Skip the normal extraction below
            else:
                raise

        # Extract predictions at masked positions
        for i in range(bs):
            abs_positions = mask_positions_list[i]
            ref_toks = ref_tokens_list[i]

            masked_logits = logits[i, abs_positions]  # [num_masked, vocab]
            probs = F.softmax(masked_logits, dim=-1)
            top1_conf, top1_token = probs.max(dim=-1)
            log_probs = F.log_softmax(masked_logits, dim=-1)
            entropy = -(probs * log_probs).sum(dim=-1)

            is_correct = (top1_token == ref_toks)

            confs = top1_conf.float().cpu().numpy()
            ents = entropy.float().cpu().numpy()
            corrects = is_correct.cpu().numpy()

            key = f"mask_{int(mask_ratio*100)}pct"
            all_records[key]["confidences"].extend(confs.tolist())
            all_records[key]["accuracies"].extend(corrects.astype(int).tolist())
            all_records[key]["entropies"].extend(ents.tolist())
            all_records[key]["is_errors"].extend((1 - corrects).astype(int).tolist())

        del padded, logits, outputs
        torch.cuda.empty_cache()

    return dict(all_records)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    write_pid()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{TASK_ID}] Starting FULL calibration study")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Samples: FULL GSM8K (~1319), Steps: {NUM_STEPS}")
    print(f"  Device: {DEVICE}, GPU: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load model
    print("\n[1/6] Loading model...")
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, dtype=torch.bfloat16, trust_remote_code=True,
    ).to(DEVICE).eval()
    vram_after_load = torch.cuda.memory_allocated() / 1e6
    print(f"  Model loaded. VRAM: {vram_after_load:.0f} MB")

    # Load GSM8K
    print("\n[2/6] Loading GSM8K samples...")
    samples = get_gsm8k_prompts(tokenizer, NUM_SAMPLES)
    total_samples = len(samples)
    print(f"  Loaded {total_samples} samples")
    prompt_lens = [len(s['prompt_ids']) for s in samples]
    answer_lens = [len(s['answer_ids']) for s in samples]
    print(f"  Prompt lengths: min={min(prompt_lens)}, max={max(prompt_lens)}, "
          f"mean={np.mean(prompt_lens):.0f}")
    print(f"  Answer lengths: min={min(answer_lens)}, max={max(answer_lens)}, "
          f"mean={np.mean(answer_lens):.0f}")

    # Batch size for oracle method (skip auto-detection to avoid CUBLAS state corruption)
    oracle_batch_size = ORACLE_BATCH_SIZE
    print(f"\n[3/6] Oracle batch size: {oracle_batch_size} (fixed, no auto-detection)")

    gpu_profile = {
        "gpu_name": torch.cuda.get_device_name(0),
        "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
        "used_oracle_batch_size": oracle_batch_size,
        "vram_after_model_mb": round(vram_after_load),
    }
    (RESULTS_DIR / f"{TASK_ID}_gpu_profile.json").write_text(
        json.dumps(gpu_profile, indent=2))

    # Check for checkpoint to resume
    checkpoint_path = FULL_RESULTS_DIR / "calibration_checkpoint.pkl"
    sc_band_data = {i: {"confidences": [], "accuracies": [], "entropies": [], "is_errors": []}
                    for i in range(len(STAGE_BANDS))}
    sc_start_idx = 0

    if checkpoint_path.exists():
        print(f"\n  Found checkpoint, loading...")
        try:
            with open(checkpoint_path, "rb") as f:
                ckpt = pickle.load(f)
            sc_band_data = ckpt["sc_band_data"]
            sc_start_idx = ckpt["sc_samples_done"]
            print(f"  Resuming from sample {sc_start_idx}")
        except Exception as e:
            print(f"  Failed to load checkpoint: {e}, starting fresh")
            sc_start_idx = 0

    # Verify CUDA is clean before starting inference
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    gc.collect()
    # Warm-up forward pass to catch any deferred errors
    print("\n  Warming up model with a short forward pass...")
    warmup_ids = torch.tensor([[1, 2, 3, MASK_TOKEN_ID, MASK_TOKEN_ID]],
                               dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        _ = model(input_ids=warmup_ids)
    del warmup_ids
    torch.cuda.synchronize()
    print("  Warm-up OK")

    # ── METHOD A: Self-Consistency ────────────────────────────────────
    # Self-consistency must be done sample-by-sample (sequential denoising)
    print(f"\n[4/6] Method A: Self-consistency calibration "
          f"({total_samples} samples, starting at {sc_start_idx})...")

    for si in range(sc_start_idx, total_samples):
        sample = samples[si]
        if si % 50 == 0:
            elapsed = time.time() - start_time
            rate = (si - sc_start_idx + 1) / max(elapsed, 1)
            remaining = (total_samples - si) / max(rate, 0.001)
            report_progress(
                epoch=si, total_epochs=total_samples * 2,
                metric={"phase": "self_consistency", "samples_done": si,
                        "total_samples": total_samples,
                        "elapsed_sec": round(elapsed, 1),
                        "eta_sec": round(remaining, 0),
                        "rate_samples_per_sec": round(rate, 3)})
            print(f"  [A] Sample {si}/{total_samples}, elapsed: {elapsed:.0f}s, "
                  f"ETA: {remaining/60:.1f}min, rate: {rate:.2f} samp/s")

        gen_len = min(len(sample["answer_ids"]), GEN_LENGTH)
        sample_band_data = run_denoising_self_consistency_single(
            model, sample["prompt_ids"], gen_len, num_steps=NUM_STEPS
        )

        # Merge into global band data
        for band_idx in range(len(STAGE_BANDS)):
            for key in ["confidences", "accuracies", "entropies", "is_errors"]:
                sc_band_data[band_idx][key].extend(sample_band_data[band_idx][key])

        # Periodic checkpoint
        if (si + 1) % CHECKPOINT_INTERVAL == 0:
            with open(checkpoint_path, "wb") as f:
                pickle.dump({"sc_band_data": sc_band_data, "sc_samples_done": si + 1}, f)
            print(f"  [A] Checkpoint saved at sample {si + 1}")

        if si % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    elapsed_a = time.time() - start_time
    print(f"  Method A complete: {elapsed_a:.0f}s ({elapsed_a/60:.1f} min)")

    # ── METHOD B: Oracle Calibration (batched) ────────────────────────
    print(f"\n[5/6] Method B: Oracle (teacher-forced) calibration "
          f"({total_samples} samples, batch_size={oracle_batch_size})...")

    oracle_data = {}
    for mr_idx, mask_ratio in enumerate(ORACLE_MASK_RATIOS):
        mr_start = time.time()
        print(f"  [B] mask_ratio={mask_ratio:.0%} ({mr_idx+1}/{len(ORACLE_MASK_RATIOS)})...")

        mr_records = run_oracle_calibration_batched(
            model, samples, mask_ratio, tokenizer, max_batch_size=oracle_batch_size
        )
        oracle_data.update(mr_records)

        mr_elapsed = time.time() - mr_start
        key = f"mask_{int(mask_ratio*100)}pct"
        n_points = len(oracle_data.get(key, {}).get("confidences", []))
        print(f"    Done: {n_points:,} data points in {mr_elapsed:.0f}s")

        elapsed = time.time() - start_time
        report_progress(
            epoch=total_samples + int((mr_idx + 1) / len(ORACLE_MASK_RATIOS) * total_samples),
            total_epochs=total_samples * 2,
            metric={"phase": "oracle", "mask_ratio": mask_ratio,
                    "elapsed_sec": round(elapsed, 1)})

    elapsed_b = time.time() - start_time
    print(f"  Method B complete: {elapsed_b:.0f}s ({elapsed_b/60:.1f} min)")

    # ── Compute metrics ───────────────────────────────────────────────
    print("\n[6/6] Computing calibration metrics and fitting calibrators...")

    # Method A metrics
    sc_profile = {}
    sc_calibrators = {}
    for band_idx in range(len(STAGE_BANDS)):
        band_name = STAGE_BAND_NAMES[band_idx]
        bd = sc_band_data[band_idx]
        confs = np.array(bd["confidences"], dtype=np.float64)
        accs = np.array(bd["accuracies"], dtype=np.float64)
        ents = np.array(bd["entropies"], dtype=np.float64)
        errs = np.array(bd["is_errors"], dtype=np.float64)

        n = len(confs)
        if n == 0:
            sc_profile[band_name] = {"n_points": 0}
            continue

        ece, bin_confs, bin_accs, bin_counts = compute_ece(confs, accs, ECE_BINS)
        pearson_r, pearson_p = (stats.pearsonr(ents, errs)
                                if np.std(ents) > 1e-8 and np.std(errs) > 1e-8
                                else (0.0, 1.0))

        # Fit calibrator
        if n >= 20:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(confs, accs)
            sc_calibrators[band_name] = iso
            cal_confs = iso.predict(confs)
            cal_ece, cal_bconfs, cal_baccs, cal_bcounts = compute_ece(cal_confs, accs, ECE_BINS)
        else:
            cal_ece = None
            cal_bconfs, cal_baccs, cal_bcounts = [], [], []

        sc_profile[band_name] = {
            "stage_band": list(STAGE_BANDS[band_idx]),
            "n_points": n,
            "ece": float(ece),
            "ece_calibrated": float(cal_ece) if cal_ece is not None else None,
            "pearson_entropy_error": {"r": float(pearson_r), "p_value": float(pearson_p)},
            "mean_confidence": float(confs.mean()),
            "mean_accuracy": float(accs.mean()),
            "mean_entropy": float(ents.mean()),
            "mean_error_rate": float(errs.mean()),
            "reliability_diagram": {
                "bin_confidences": bin_confs, "bin_accuracies": bin_accs, "bin_counts": bin_counts,
            },
            "calibrated_reliability_diagram": {
                "bin_confidences": cal_bconfs, "bin_accuracies": cal_baccs, "bin_counts": cal_bcounts,
            },
        }

        print(f"  [A] {band_name}: N={n:,}, ECE={ece:.4f}, "
              f"Cal-ECE={f'{cal_ece:.6f}' if cal_ece is not None else 'N/A'}")
        print(f"       Pearson(ent,err): r={pearson_r:.4f}, mean_acc={float(accs.mean()):.4f}")

    # Method B metrics
    oracle_profile = {}
    oracle_calibrators = {}
    for mr in ORACLE_MASK_RATIOS:
        key = f"mask_{int(mr*100)}pct"
        bd = oracle_data.get(key, {"confidences": [], "accuracies": [],
                                    "entropies": [], "is_errors": []})
        confs = np.array(bd["confidences"], dtype=np.float64)
        accs = np.array(bd["accuracies"], dtype=np.float64)
        ents = np.array(bd["entropies"], dtype=np.float64)
        errs = np.array(bd["is_errors"], dtype=np.float64)

        n = len(confs)
        if n == 0:
            oracle_profile[key] = {"n_points": 0}
            continue

        ece, bin_confs, bin_accs, bin_counts = compute_ece(confs, accs, ECE_BINS)
        pearson_r, pearson_p = (stats.pearsonr(ents, errs)
                                if np.std(ents) > 1e-8 and np.std(errs) > 1e-8
                                else (0.0, 1.0))

        if n >= 20:
            iso = IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip")
            iso.fit(confs, accs)
            oracle_calibrators[key] = iso
            cal_confs = iso.predict(confs)
            cal_ece, cal_bconfs, cal_baccs, cal_bcounts = compute_ece(cal_confs, accs, ECE_BINS)
        else:
            cal_ece = None
            cal_bconfs, cal_baccs, cal_bcounts = [], [], []

        oracle_profile[key] = {
            "mask_ratio": mr,
            "n_points": n,
            "ece": float(ece),
            "ece_calibrated": float(cal_ece) if cal_ece is not None else None,
            "pearson_entropy_error": {"r": float(pearson_r), "p_value": float(pearson_p)},
            "mean_confidence": float(confs.mean()),
            "mean_accuracy": float(accs.mean()),
            "mean_entropy": float(ents.mean()),
            "mean_error_rate": float(errs.mean()),
            "reliability_diagram": {
                "bin_confidences": bin_confs, "bin_accuracies": bin_accs, "bin_counts": bin_counts,
            },
            "calibrated_reliability_diagram": {
                "bin_confidences": cal_bconfs, "bin_accuracies": cal_baccs, "bin_counts": cal_bcounts,
            },
        }

        print(f"  [B] {key}: N={n:,}, ECE={ece:.4f}, "
              f"Cal-ECE={f'{cal_ece:.6f}' if cal_ece is not None else 'N/A'}")
        print(f"       Pearson(ent,err): r={pearson_r:.4f}, mean_acc={float(accs.mean()):.4f}")

    # ── Overall ECE ──────────────────────────────────────────────────
    all_c = np.concatenate([np.array(bd["confidences"])
                            for bd in sc_band_data.values() if bd["confidences"]])
    all_a = np.concatenate([np.array(bd["accuracies"])
                            for bd in sc_band_data.values() if bd["accuracies"]])
    overall_sc_ece, _, _, _ = compute_ece(all_c, all_a, ECE_BINS)

    all_oc = np.concatenate([np.array(bd["confidences"])
                             for bd in oracle_data.values() if bd["confidences"]])
    all_oa = np.concatenate([np.array(bd["accuracies"])
                             for bd in oracle_data.values() if bd["accuracies"]])
    overall_oracle_ece, _, _, _ = compute_ece(all_oc, all_oa, ECE_BINS)

    # Save calibrators
    all_calibrators = {
        "self_consistency": sc_calibrators,
        "oracle": oracle_calibrators,
    }
    # Save to both pilots (for backward compat) and full results
    for cal_dir in [RESULTS_DIR / "pilots", FULL_RESULTS_DIR]:
        cal_dir.mkdir(parents=True, exist_ok=True)
        cal_path = cal_dir / "calibrators.pkl"
        with open(cal_path, "wb") as f:
            pickle.dump(all_calibrators, f)
        print(f"  Saved calibrators to {cal_path}")

    # ── Hypothesis Testing ────────────────────────────────────────────
    early_sc_ece = sc_profile.get("early_80_100", {}).get("ece", 0)
    early_oracle_ece = oracle_profile.get("mask_90pct", {}).get("ece", 0)
    late_sc_pearson = sc_profile.get("late_0_20", {}).get(
        "pearson_entropy_error", {}).get("r", 0)
    late_oracle_pearson = oracle_profile.get("mask_10pct", {}).get(
        "pearson_entropy_error", {}).get("r", 0)

    h1_pass = (early_sc_ece > 0.10) or (early_oracle_ece > 0.10)
    h2_pass = (late_sc_pearson > 0.15) or (late_oracle_pearson > 0.15)

    if h1_pass and h2_pass:
        go_no_go = "GO"
        overall_rec = "PROCEED"
    elif not h1_pass and early_sc_ece < 0.05 and early_oracle_ece < 0.05:
        go_no_go = "NO_GO"
        overall_rec = "PIVOT"
    else:
        go_no_go = "CONDITIONAL_GO"
        overall_rec = "REFINE"

    summary_text = (
        f"FULL mode ({total_samples} samples). "
        f"H1 (ECE>0.10 early): {'PASS' if h1_pass else 'FAIL'} "
        f"(SC-ECE={early_sc_ece:.4f}, Oracle-ECE={early_oracle_ece:.4f}). "
        f"H2 (corr>0.15 late): {'PASS' if h2_pass else 'FAIL'} "
        f"(SC-r={late_sc_pearson:.4f}, Oracle-r={late_oracle_pearson:.4f}). "
        f"Overall ECE: SC={overall_sc_ece:.4f}, Oracle={overall_oracle_ece:.4f}."
    )

    elapsed_total = time.time() - start_time

    # Build result
    result = {
        "task_id": TASK_ID,
        "candidate_id": "cand_card",
        "mode": "FULL",
        "model": "LLaDA-8B-Instruct",
        "benchmark": "GSM8K",
        "num_samples": total_samples,
        "num_steps": NUM_STEPS,
        "gen_length": GEN_LENGTH,
        "seed": SEED,
        "elapsed_sec": round(elapsed_total, 1),
        "method_a_self_consistency": {
            "description": "Compare each step's prediction to final committed token",
            "overall_ece": float(overall_sc_ece),
            "calibration_profile": sc_profile,
        },
        "method_b_oracle": {
            "description": "Teacher-forced: mask fraction of reference, measure recovery",
            "overall_ece": float(overall_oracle_ece),
            "calibration_profile": oracle_profile,
        },
        "hypothesis_tests": {
            "H1_early_miscalibration": {
                "criterion": "ECE > 0.10 at early stages",
                "sc_value": float(early_sc_ece),
                "oracle_value": float(early_oracle_ece),
                "pass": h1_pass,
            },
            "H2_entropy_error_correlation": {
                "criterion": "Pearson corr(entropy, is_error) > 0.15 at late stages",
                "sc_value": float(late_sc_pearson),
                "oracle_value": float(late_oracle_pearson),
                "pass": h2_pass,
            },
        },
        "go_no_go": go_no_go,
        "recommendation": overall_rec,
        "summary": summary_text,
        "gpu_info": {
            "device": torch.cuda.get_device_name(0),
            "vram_total_mb": round(torch.cuda.get_device_properties(0).total_memory / 1e6),
            "vram_peak_mb": round(torch.cuda.max_memory_allocated() / 1e6),
            "oracle_batch_size": oracle_batch_size,
        },
        "timestamp": datetime.now().isoformat(),
    }

    # Save to full results dir
    result_path = FULL_RESULTS_DIR / "calibration_profile.json"
    result_path.write_text(json.dumps(result, indent=2, cls=NumpyEncoder))
    print(f"\n  Saved full results to {result_path}")

    # Also overwrite the main calibration_profile.json (used by downstream tasks)
    main_result_path = RESULTS_DIR / "calibration_profile.json"
    result_path_copy = result.copy()
    main_result_path.write_text(json.dumps(result_path_copy, indent=2, cls=NumpyEncoder))
    print(f"  Updated main calibration profile at {main_result_path}")

    # Clean up checkpoint
    if checkpoint_path.exists():
        checkpoint_path.unlink()
        print(f"  Removed checkpoint file")

    # Print summary
    print(f"\n{'='*70}")
    print(f"FULL CALIBRATION STUDY RESULTS ({total_samples} samples)")
    print(f"{'='*70}")
    print(f"  Self-Consistency ECE: {overall_sc_ece:.4f}")
    print(f"  Oracle ECE:           {overall_oracle_ece:.4f}")
    print()
    for band_name in STAGE_BAND_NAMES:
        bp = sc_profile.get(band_name, {})
        n = bp.get("n_points", 0)
        ece = bp.get("ece", 0)
        cal_ece = bp.get("ece_calibrated")
        pr = bp.get("pearson_entropy_error", {}).get("r", 0)
        print(f"  [SC] {band_name}: N={n:,}, ECE={ece:.4f}, "
              f"Cal-ECE={f'{cal_ece:.6f}' if cal_ece is not None else 'N/A'}, "
              f"Pearson-r={pr:.4f}")
    print()
    for mr in ORACLE_MASK_RATIOS:
        key = f"mask_{int(mr*100)}pct"
        bp = oracle_profile.get(key, {})
        n = bp.get("n_points", 0)
        ece = bp.get("ece", 0)
        cal_ece = bp.get("ece_calibrated")
        pr = bp.get("pearson_entropy_error", {}).get("r", 0)
        print(f"  [Oracle] {key}: N={n:,}, ECE={ece:.4f}, "
              f"Cal-ECE={f'{cal_ece:.6f}' if cal_ece is not None else 'N/A'}, "
              f"Pearson-r={pr:.4f}")
    print()
    print(f"  GO/NO-GO: {go_no_go}")
    print(f"  {summary_text}")
    print(f"  Elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")
    print(f"{'='*70}")

    report_progress(
        epoch=total_samples * 2, total_epochs=total_samples * 2,
        metric={"overall_sc_ece": float(overall_sc_ece),
                "overall_oracle_ece": float(overall_oracle_ece),
                "go_no_go": go_no_go,
                "elapsed_sec": round(elapsed_total, 1),
                "num_samples": total_samples})
    mark_done(status="success", summary=summary_text)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"\n[FATAL] {error_msg}")
        traceback.print_exc()
        mark_done(status="failed", summary=error_msg)
        sys.exit(1)
