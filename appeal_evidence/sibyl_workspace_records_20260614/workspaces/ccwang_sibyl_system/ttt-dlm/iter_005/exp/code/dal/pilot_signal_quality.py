#!/usr/bin/env python3
"""
P2: Self-Supervised Signal Quality Across Mask Ratios

On Dream-7B frozen backbone with randomly initialized TTT-MLP layer, measure:
  - Self-supervised loss (MLM on revealed tokens) at mask ratios {0.1..0.9}
  - Gradient magnitude at each mask ratio
  - Signal-to-noise ratio (SNR) of gradients

For 16 GSM8K prompts, seed 42.

Identifies the critical zone where signal quality is highest,
validating the phase-transition scheduling hypothesis.
"""

import os
import sys
import json
import time
import gc
import math
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModel

# Add code dir to path
CODE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, CODE_DIR)
from ttt_layer import TTTLayer

# =============================================================================
# Configuration
# =============================================================================

TASK_ID = "pilot_signal_quality"
SEED = 42
NUM_SAMPLES = 16
MAX_SEQ_LEN = 256
MASK_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

# Paths (set via env or defaults)
REMOTE_BASE = os.environ.get("REMOTE_BASE", "/home/ccwang/sibyl_system")
PROJECT = os.environ.get("PROJECT", "ttt-dlm")
PROJECT_DIR = f"{REMOTE_BASE}/projects/{PROJECT}"
RESULTS_DIR = f"{PROJECT_DIR}/exp/results/pilots"
CHECKPOINT_DIR = f"{REMOTE_BASE}/shared/checkpoints/Dream-v0-Instruct-7B"
DATASET_DIR = f"{REMOTE_BASE}/shared/datasets/gsm8k"

# When CUDA_VISIBLE_DEVICES is set externally, torch sees only cuda:0
GPU_ID = 0


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def write_pid():
    """Write PID file for system recovery."""
    pid_path = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    """Write progress file for system monitor."""
    progress_path = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    progress_path.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_task_done(status="success", summary=""):
    """Write DONE marker file."""
    pid_file = Path(RESULTS_DIR) / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(RESULTS_DIR) / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except (json.JSONDecodeError, ValueError):
            pass
    marker = Path(RESULTS_DIR) / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


def create_masked_input(target_ids, mask_token_id, mask_ratio, seed=None):
    """Create masked version of input at a specific mask ratio."""
    if seed is not None:
        torch.manual_seed(seed)
    B, S = target_ids.shape
    mask_probs = torch.rand(B, S, device=target_ids.device)
    is_masked = mask_probs < mask_ratio
    input_ids = target_ids.clone()
    input_ids[is_masked] = mask_token_id
    revealed_mask = (~is_masked).float()
    return input_ids, revealed_mask


def compute_gradient_stats(grads):
    """Compute gradient magnitude and per-parameter SNR."""
    all_grads = torch.cat([g.flatten() for g in grads])
    grad_mag = all_grads.norm().item()
    grad_mean = all_grads.mean().item()
    grad_std = all_grads.std().item()
    # SNR = |mean| / std  (higher = cleaner signal)
    snr = abs(grad_mean) / (grad_std + 1e-10)
    # Also compute per-element SNR using running stats
    return {
        "grad_magnitude": grad_mag,
        "grad_mean": grad_mean,
        "grad_std": grad_std,
        "grad_snr": snr,
        "grad_l2_per_param": grad_mag / math.sqrt(max(all_grads.numel(), 1)),
    }


def main():
    print(f"[{TASK_ID}] 开始执行 — P2: 自监督信号质量分析")
    print(f"[{TASK_ID}] GPU: {GPU_ID}, seed: {SEED}, samples: {NUM_SAMPLES}")
    print(f"[{TASK_ID}] mask ratios: {MASK_RATIOS}")

    start_time = time.time()
    set_seed(SEED)
    write_pid()

    # =========================================================================
    # 1. Load model and tokenizer
    # =========================================================================
    print(f"[{TASK_ID}] 加载 Dream-7B 模型...")
    device = torch.device(f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(
        CHECKPOINT_DIR, trust_remote_code=True
    )
    model = AutoModel.from_pretrained(
        CHECKPOINT_DIR,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        device_map={"": device},
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Get model config
    config = model.config
    d_model = config.hidden_size
    vocab_size = config.vocab_size
    mask_token_id = config.mask_token_id
    n_layers = config.num_hidden_layers
    insertion_layer = n_layers // 2

    print(f"[{TASK_ID}] 模型加载完成: d_model={d_model}, vocab={vocab_size}, "
          f"layers={n_layers}, mask_token_id={mask_token_id}")

    # =========================================================================
    # 2. Create TTT-MLP layer (randomly initialized)
    # =========================================================================
    print(f"[{TASK_ID}] 创建 TTT-MLP 层 (随机初始化)...")
    set_seed(SEED)  # Ensure deterministic initialization
    ttt_layer = TTTLayer(
        d_model=d_model,
        variant="mlp",
        vocab_size=vocab_size,
        ttt_lr=1e-3,
        precision_weighted=False,  # Uniform weighting for this diagnostic
    ).to(device).to(torch.bfloat16)

    fast_param_count = ttt_layer.get_fast_weight_param_count()
    total_param_count = ttt_layer.get_trainable_param_count()
    print(f"[{TASK_ID}] TTT-MLP: fast_params={fast_param_count:,}, "
          f"trainable_params={total_param_count:,}")

    # =========================================================================
    # 3. Load GSM8K data
    # =========================================================================
    print(f"[{TASK_ID}] 加载 GSM8K 数据集...")
    dataset = load_from_disk(DATASET_DIR)
    if "test" in dataset:
        data_split = dataset["test"]
    elif "train" in dataset:
        data_split = dataset["train"]
    else:
        data_split = dataset

    # Select samples
    indices = list(range(min(NUM_SAMPLES, len(data_split))))
    samples = [data_split[i] for i in indices]

    # Tokenize
    texts = []
    for s in samples:
        if "question" in s:
            texts.append(s["question"])
        elif "text" in s:
            texts.append(s["text"])
        else:
            texts.append(str(s))

    print(f"[{TASK_ID}] 对 {len(texts)} 个样本进行 tokenize...")
    tokenized = tokenizer(
        texts,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    target_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)

    print(f"[{TASK_ID}] 数据准备完成: shape={target_ids.shape}")
    report_progress(0, len(MASK_RATIOS), metric={"status": "data_loaded"})

    # =========================================================================
    # 4. Hook to capture hidden states at layer L/2
    # =========================================================================
    captured_hidden = {}

    def get_backbone_layers(m):
        """Get transformer layers from Dream model (DreamModel -> .model -> .layers)."""
        if hasattr(m, 'model') and hasattr(m.model, 'layers'):
            return m.model.layers
        if hasattr(m, 'layers'):
            return m.layers
        raise ValueError("Cannot find transformer layers in model")

    print(f"[{TASK_ID}] 使用 insertion_layer={insertion_layer} (总层数={n_layers})")

    layers = get_backbone_layers(model)
    target_layer = layers[insertion_layer]

    def capture_hook(module, input, output):
        if isinstance(output, tuple):
            captured_hidden["h"] = output[0].detach()
        else:
            captured_hidden["h"] = output.detach()
        return output

    # =========================================================================
    # 5. Main experiment: sweep mask ratios
    # =========================================================================
    results = {
        "task_id": TASK_ID,
        "model": "Dream-7B-Instruct",
        "variant": "mlp",
        "num_samples": NUM_SAMPLES,
        "max_seq_len": MAX_SEQ_LEN,
        "seed": SEED,
        "d_model": d_model,
        "insertion_layer": insertion_layer,
        "fast_param_count": fast_param_count,
        "mask_ratios": MASK_RATIOS,
        "per_ratio": {},
        "per_sample_per_ratio": {},
    }

    for ratio_idx, mask_ratio in enumerate(MASK_RATIOS):
        print(f"\n[{TASK_ID}] === mask_ratio={mask_ratio:.1f} ({ratio_idx+1}/{len(MASK_RATIOS)}) ===")

        # Reset TTT layer for each mask ratio (fresh random init, same seed)
        set_seed(SEED)
        ttt_layer = TTTLayer(
            d_model=d_model,
            variant="mlp",
            vocab_size=vocab_size,
            ttt_lr=1e-3,
            precision_weighted=False,
        ).to(device).to(torch.bfloat16)

        ratio_losses = []
        ratio_grad_mags = []
        ratio_grad_snrs = []
        ratio_grad_means = []
        ratio_grad_stds = []
        ratio_num_revealed = []
        per_sample_data = []

        # Process samples one at a time (or small batches) to get per-sample stats
        batch_size = 4  # Process 4 at a time for efficiency
        for batch_start in range(0, NUM_SAMPLES, batch_size):
            batch_end = min(batch_start + batch_size, NUM_SAMPLES)
            b_target = target_ids[batch_start:batch_end]
            b_attn = attention_mask[batch_start:batch_end]
            B = b_target.shape[0]

            # Create masked input at this ratio
            set_seed(SEED + ratio_idx * 1000 + batch_start)
            b_input, b_revealed = create_masked_input(b_target, mask_token_id, mask_ratio)

            # Mask out padding positions from revealed mask
            b_revealed = b_revealed * b_attn.float()

            # Reset TTT fast weights for this batch
            ttt_layer.reset_fast_weights(B)

            # Forward through backbone to get hidden states at insertion layer
            hook_handle = target_layer.register_forward_hook(capture_hook)
            try:
                with torch.no_grad():
                    outputs = model(input_ids=b_input, output_hidden_states=False, return_dict=True)
                    backbone_logits = outputs.logits
            finally:
                hook_handle.remove()

            hidden_states = captured_hidden["h"]  # (B, S, D)

            # Now compute TTT self-supervised loss and gradients
            # Enable grad on fast weights
            h_normed = ttt_layer.layer_norm(hidden_states)
            fast_params = ttt_layer.fast_weight.get_params_for_grad()
            for p in fast_params:
                p.requires_grad_(True)

            # Forward through fast weight
            fast_output = ttt_layer.fast_weight(h_normed)

            # Compute SSL loss
            ssl_logits = ttt_layer.ssl_head(fast_output)  # (B, S, V)
            B_curr, S, V = ssl_logits.shape
            per_token_loss = F.cross_entropy(
                ssl_logits.reshape(-1, V).float(),
                b_target.reshape(-1),
                reduction="none",
            ).reshape(B_curr, S)

            # Masked loss per sample
            for i in range(B_curr):
                sample_idx = batch_start + i
                sample_revealed = b_revealed[i]
                n_rev = sample_revealed.sum().item()
                if n_rev > 0:
                    sample_loss = (per_token_loss[i] * sample_revealed).sum() / n_rev
                else:
                    sample_loss = torch.tensor(0.0, device=device)

                # Compute gradient for this sample
                # We need per-sample gradient — use the batch loss but record per-sample
                per_sample_data.append({
                    "sample_idx": int(sample_idx),
                    "loss": sample_loss.item(),
                    "num_revealed": int(n_rev),
                    "num_total": int(sample_revealed.numel()),
                    "actual_mask_ratio": 1.0 - n_rev / sample_revealed.numel(),
                })
                ratio_losses.append(sample_loss.item())
                ratio_num_revealed.append(int(n_rev))

            # Compute batch-level gradient for SNR
            batch_revealed = b_revealed
            n_revealed_total = batch_revealed.sum().clamp(min=1.0)
            batch_loss = (per_token_loss * batch_revealed).sum() / n_revealed_total

            grads = torch.autograd.grad(
                batch_loss, fast_params, create_graph=False, allow_unused=True
            )
            grads = [g if g is not None else torch.zeros_like(p)
                     for g, p in zip(grads, fast_params)]

            grad_stats = compute_gradient_stats(grads)
            ratio_grad_mags.append(grad_stats["grad_magnitude"])
            ratio_grad_snrs.append(grad_stats["grad_snr"])
            ratio_grad_means.append(grad_stats["grad_mean"])
            ratio_grad_stds.append(grad_stats["grad_std"])

            # Clean up
            for p in fast_params:
                p.requires_grad_(False)
            del fast_output, ssl_logits, per_token_loss, batch_loss, grads
            torch.cuda.empty_cache()

        # Aggregate stats for this mask ratio
        avg_loss = np.mean(ratio_losses)
        std_loss = np.std(ratio_losses)
        avg_grad_mag = np.mean(ratio_grad_mags)
        avg_grad_snr = np.mean(ratio_grad_snrs)
        avg_num_revealed = np.mean(ratio_num_revealed)

        ratio_result = {
            "mask_ratio": mask_ratio,
            "avg_ssl_loss": float(avg_loss),
            "std_ssl_loss": float(std_loss),
            "avg_grad_magnitude": float(avg_grad_mag),
            "avg_grad_snr": float(avg_grad_snr),
            "avg_grad_mean": float(np.mean(ratio_grad_means)),
            "avg_grad_std": float(np.mean(ratio_grad_stds)),
            "avg_num_revealed": float(avg_num_revealed),
            "num_samples": len(ratio_losses),
        }

        results["per_ratio"][str(mask_ratio)] = ratio_result
        results["per_sample_per_ratio"][str(mask_ratio)] = per_sample_data

        print(f"[{TASK_ID}]   loss={avg_loss:.4f}±{std_loss:.4f}, "
              f"grad_mag={avg_grad_mag:.6f}, grad_snr={avg_grad_snr:.6f}, "
              f"avg_revealed={avg_num_revealed:.1f}")

        report_progress(
            ratio_idx + 1, len(MASK_RATIOS),
            loss=float(avg_loss),
            metric={
                "mask_ratio": mask_ratio,
                "grad_snr": float(avg_grad_snr),
            }
        )

    # =========================================================================
    # 6. Analysis: identify critical zone
    # =========================================================================
    print(f"\n[{TASK_ID}] === 分析信号质量趋势 ===")

    # Check if loss or gradient SNR monotonically improves at mask ratio < 0.6
    ratios_below_06 = [r for r in MASK_RATIOS if r < 0.6]
    losses_below_06 = [results["per_ratio"][str(r)]["avg_ssl_loss"] for r in ratios_below_06]
    snrs_below_06 = [results["per_ratio"][str(r)]["avg_grad_snr"] for r in ratios_below_06]
    grad_mags_below_06 = [results["per_ratio"][str(r)]["avg_grad_magnitude"] for r in ratios_below_06]

    # Check monotonic trends
    # For loss: should decrease as more tokens are revealed (lower mask ratio = more revealed)
    loss_decreasing = all(losses_below_06[i] >= losses_below_06[i+1]
                         for i in range(len(losses_below_06)-1))
    # For gradient magnitude: should increase with more revealed tokens
    grad_increasing = all(grad_mags_below_06[i] <= grad_mags_below_06[i+1]
                          for i in range(len(grad_mags_below_06)-1))

    # Find best mask ratio (highest SNR)
    all_snrs = {r: results["per_ratio"][str(r)]["avg_grad_snr"] for r in MASK_RATIOS}
    best_snr_ratio = max(all_snrs, key=all_snrs.get)
    best_snr_value = all_snrs[best_snr_ratio]

    # Find critical zone: contiguous range where SNR > 50% of peak
    snr_threshold = best_snr_value * 0.5
    critical_zone = [r for r in MASK_RATIOS if all_snrs[r] >= snr_threshold]
    critical_zone_range = (min(critical_zone), max(critical_zone)) if critical_zone else (0, 1)

    # Also analyze by gradient magnitude: find peak zone
    all_grad_mags = {r: results["per_ratio"][str(r)]["avg_grad_magnitude"] for r in MASK_RATIOS}
    best_grad_ratio = max(all_grad_mags, key=all_grad_mags.get)

    analysis = {
        "loss_monotonic_below_06": loss_decreasing,
        "grad_monotonic_below_06": grad_increasing,
        "best_snr_mask_ratio": best_snr_ratio,
        "best_snr_value": float(best_snr_value),
        "best_grad_magnitude_ratio": best_grad_ratio,
        "critical_zone": list(critical_zone),
        "critical_zone_range": list(critical_zone_range),
        "snr_threshold_used": float(snr_threshold),
    }

    # Pass criteria check
    # "Clear signal improvement at mask ratio < 0.6 AND identifiable critical zone"
    # We check: either monotonic loss decrease OR monotonic grad increase below 0.6
    signal_improves = loss_decreasing or grad_increasing
    has_critical_zone = len(critical_zone) >= 2  # At least 2 ratios in the zone
    pass_criteria = signal_improves and has_critical_zone

    analysis["signal_improves_below_06"] = signal_improves
    analysis["has_critical_zone"] = has_critical_zone
    analysis["pass_criteria"] = pass_criteria
    analysis["verdict"] = "GO" if pass_criteria else "NO-GO"

    results["analysis"] = analysis

    print(f"[{TASK_ID}] 损失随 mask ratio 下降单调递减 (< 0.6): {loss_decreasing}")
    print(f"[{TASK_ID}] 梯度幅度随 mask ratio 下降单调递增 (< 0.6): {grad_increasing}")
    print(f"[{TASK_ID}] 最佳 SNR mask ratio: {best_snr_ratio} (SNR={best_snr_value:.6f})")
    print(f"[{TASK_ID}] 最佳梯度幅度 mask ratio: {best_grad_ratio}")
    print(f"[{TASK_ID}] 关键区域: {critical_zone}")
    print(f"[{TASK_ID}] 判定: {'GO ✓' if pass_criteria else 'NO-GO ✗'}")

    # =========================================================================
    # 7. Save results
    # =========================================================================
    elapsed_min = (time.time() - start_time) / 60
    results["elapsed_minutes"] = round(elapsed_min, 1)
    results["timestamp"] = datetime.now().isoformat()

    results_path = Path(RESULTS_DIR) / "p2_signal_quality.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n[{TASK_ID}] 结果已保存: {results_path}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"{'Mask Ratio':>12} {'SSL Loss':>12} {'Grad Mag':>12} {'Grad SNR':>12} {'#Revealed':>12}")
    print(f"{'='*80}")
    for r in MASK_RATIOS:
        d = results["per_ratio"][str(r)]
        print(f"{r:>12.1f} {d['avg_ssl_loss']:>12.4f} {d['avg_grad_magnitude']:>12.6f} "
              f"{d['avg_grad_snr']:>12.6f} {d['avg_num_revealed']:>12.1f}")
    print(f"{'='*80}")
    print(f"\n[{TASK_ID}] 完成！耗时 {elapsed_min:.1f} 分钟")

    # Mark done
    mark_task_done(
        status="success" if pass_criteria else "completed_no_pass",
        summary=f"Signal quality sweep across 9 mask ratios. "
                f"Best SNR at r={best_snr_ratio}. "
                f"Critical zone: {critical_zone_range}. "
                f"Verdict: {'GO' if pass_criteria else 'NO-GO'}. "
                f"Elapsed: {elapsed_min:.1f}min.",
    )


if __name__ == "__main__":
    main()
