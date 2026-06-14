#!/usr/bin/env python3
"""E1 Full: Pareto evaluation on GPT-2 Small (20-30 checkpoints)."""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer
from scipy import stats

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
N_SAMPLES = 2048
BATCH_SIZE = 64
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = RESULTS_DIR / "e1_full_gpt2.pid"
PROGRESS_FILE = RESULTS_DIR / "e1_full_gpt2_PROGRESS.json"
DONE_FILE = RESULTS_DIR / "e1_full_gpt2_DONE"
GPU_PROFILE_FILE = RESULTS_DIR / "e1_full_gpt2_gpu_profile.json"

PID_FILE.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": "e1_full_gpt2",
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": "e1_full_gpt2",
        "status": status,
        "summary": summary,
        "final_progress": progress,
        "timestamp": datetime.now().isoformat(),
    }))


torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Checkpoints: diverse GPT-2 Small SAEs
# ---------------------------------------------------------------------------
CHECKPOINTS = [
    # Standard residual stream
    {"name": "res-jb-l8-resid_pre-24k", "release": "gpt2-small-res-jb", "sae_id": "blocks.8.hook_resid_pre", "architecture": "standard"},
    {"name": "res-jb-l0-resid_pre", "release": "gpt2-small-res-jb", "sae_id": "blocks.0.hook_resid_pre", "architecture": "standard"},
    {"name": "res-jb-l4-resid_pre", "release": "gpt2-small-res-jb", "sae_id": "blocks.4.hook_resid_pre", "architecture": "standard"},
    {"name": "resid-post-v5-32k-l8", "release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.8.hook_resid_post", "architecture": "standard"},
    {"name": "resid-post-v5-128k-l8", "release": "gpt2-small-resid-post-v5-128k", "sae_id": "blocks.8.hook_resid_post", "architecture": "standard"},
    {"name": "resid-post-v5-32k-l0", "release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.0.hook_resid_post", "architecture": "standard"},
    {"name": "resid-post-v5-32k-l4", "release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.4.hook_resid_post", "architecture": "standard"},
    {"name": "resid-mid-v5-32k-l8", "release": "gpt2-small-resid-mid-v5-32k", "sae_id": "blocks.8.hook_resid_mid", "architecture": "standard"},
    {"name": "resid-mid-v5-128k-l8", "release": "gpt2-small-resid-mid-v5-128k", "sae_id": "blocks.8.hook_resid_mid", "architecture": "standard"},
    # MLP / Attention
    {"name": "mlp-out-v5-32k-l8", "release": "gpt2-small-mlp-out-v5-32k", "sae_id": "blocks.8.hook_mlp_out", "architecture": "standard"},
    {"name": "mlp-out-v5-128k-l8", "release": "gpt2-small-mlp-out-v5-128k", "sae_id": "blocks.8.hook_mlp_out", "architecture": "standard"},
    {"name": "mlp-out-v5-32k-l0", "release": "gpt2-small-mlp-out-v5-32k", "sae_id": "blocks.0.hook_mlp_out", "architecture": "standard"},
    {"name": "attn-out-v5-32k-l8", "release": "gpt2-small-attn-out-v5-32k", "sae_id": "blocks.8.hook_attn_out", "architecture": "standard"},
    {"name": "attn-out-v5-128k-l8", "release": "gpt2-small-attn-out-v5-128k", "sae_id": "blocks.8.hook_attn_out", "architecture": "standard"},
    {"name": "attn-out-v5-32k-l0", "release": "gpt2-small-attn-out-v5-32k", "sae_id": "blocks.0.hook_attn_out", "architecture": "standard"},
    # Hook-Z
    {"name": "hook-z-kk-l8", "release": "gpt2-small-hook-z-kk", "sae_id": "blocks.8.hook_z", "architecture": "standard"},
    {"name": "hook-z-kk-l0", "release": "gpt2-small-hook-z-kk", "sae_id": "blocks.0.hook_z", "architecture": "standard"},
    # MLP-TM
    {"name": "mlp-tm-l8", "release": "gpt2-small-mlp-tm", "sae_id": "blocks.8.hook_mlp_out", "architecture": "standard"},
    {"name": "mlp-tm-l0", "release": "gpt2-small-mlp-tm", "sae_id": "blocks.0.hook_mlp_out", "architecture": "standard"},
    # Feature splitting variants (different width)
    {"name": "res-jb-fs-768", "release": "gpt2-small-res-jb-feature-splitting", "sae_id": "blocks.8.hook_resid_pre_768", "architecture": "feature_splitting"},
    {"name": "res-jb-fs-1536", "release": "gpt2-small-res-jb-feature-splitting", "sae_id": "blocks.8.hook_resid_pre_1536", "architecture": "feature_splitting"},
    {"name": "res-jb-fs-3072", "release": "gpt2-small-res-jb-feature-splitting", "sae_id": "blocks.8.hook_resid_pre_3072", "architecture": "feature_splitting"},
    {"name": "res-jb-fs-6144", "release": "gpt2-small-res-jb-feature-splitting", "sae_id": "blocks.8.hook_resid_pre_6144", "architecture": "feature_splitting"},
    # AJT variants (different regularization)
    {"name": "res-sce-ajt-l6", "release": "gpt2-small-res_sce-ajt", "sae_id": "blocks.6.hook_resid_pre", "architecture": "standard"},
    {"name": "res-scl-ajt-l6", "release": "gpt2-small-res_scl-ajt", "sae_id": "blocks.6.hook_resid_pre", "architecture": "standard"},
    {"name": "res-sle-ajt-l6", "release": "gpt2-small-res_sle-ajt", "sae_id": "blocks.6.hook_resid_pre", "architecture": "standard"},
    {"name": "res-sll-ajt-l6", "release": "gpt2-small-res_sll-ajt", "sae_id": "blocks.6.hook_resid_pre", "architecture": "standard"},
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_text_corpus(n_samples=N_SAMPLES, max_length=MAX_SEQ_LEN):
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds = ds.shuffle(seed=SEED)
    texts = []
    for item in ds:
        text = item["text"].strip()
        if len(text) > 50:
            texts.append(text)
        if len(texts) >= n_samples:
            break
    return texts


def get_hook_name(sae):
    if hasattr(sae.cfg, "hook_name"):
        return sae.cfg.hook_name
    return sae.cfg.metadata.get("hook_name", "")


def get_hook_layer(sae):
    hook_name = get_hook_name(sae)
    for p in hook_name.split("."):
        if p.isdigit():
            return int(p)
    return 0


@torch.no_grad()
def collect_activations(model, sae, texts, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LEN):
    all_acts = []
    all_l0 = []
    all_recon_mse = []
    all_ev = []
    total_tokens = 0
    hook_name = get_hook_name(sae)
    hook_layer = get_hook_layer(sae)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        _, cache = model.run_with_cache(tokens["input_ids"], stop_at_layer=hook_layer + 1, names_filter=[hook_name])
        acts = cache[hook_name]

        sae_acts = sae.encode(acts)
        recon = sae.decode(sae_acts)

        # For hook_z and similar 4D hooks, flatten for mask alignment
        if acts.ndim == 4:
            acts_flat = acts.reshape(acts.size(0), acts.size(1), -1)
            recon_flat = recon.reshape(recon.size(0), recon.size(1), -1)
        else:
            acts_flat = acts
            recon_flat = recon

        mask = tokens["attention_mask"].unsqueeze(-1).float()
        l0 = (sae_acts > 0).float().sum(dim=-1).mean().item()
        all_l0.append(l0)

        mse = ((recon_flat - acts_flat) ** 2 * mask).sum() / mask.sum()
        all_recon_mse.append(mse.item())

        var = ((acts_flat - acts_flat.mean(dim=-1, keepdim=True)) ** 2 * mask).sum() / mask.sum()
        ev = 1 - mse / (var + 1e-8)
        all_ev.append(ev.item())

        all_acts.append(sae_acts.cpu())
        total_tokens += mask.sum().item()

    all_acts = torch.cat(all_acts, dim=0)
    return {
        "acts": all_acts,
        "l0": float(np.mean(all_l0)),
        "recon_mse": float(np.mean(all_recon_mse)),
        "explained_variance": float(np.mean(all_ev)),
        "total_tokens": total_tokens,
    }


def compute_dead_neuron_rate(all_acts, threshold=1e-5):
    freq = (all_acts > 0).float().mean(dim=(0, 1))
    dead = (freq < threshold).float().mean().item()
    return dead


@torch.no_grad()
def compute_ce_loss_recovered(model, sae, texts, tokenizer, batch_size=BATCH_SIZE, max_length=MAX_SEQ_LEN):
    losses = {"base": [], "recon": [], "zero_ablation": []}
    hook_name = get_hook_name(sae)

    for i in range(0, min(len(texts), 256), batch_size):
        batch_texts = texts[i:i + batch_size]
        tokens = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        input_ids = tokens["input_ids"]
        attn_mask = tokens["attention_mask"]

        logits_base = model(input_ids, attention_mask=attn_mask)
        loss_base = F.cross_entropy(logits_base.reshape(-1, logits_base.size(-1)), input_ids.reshape(-1), reduction="none")
        losses["base"].append((loss_base * attn_mask.reshape(-1)).sum().item() / attn_mask.sum().item())

        def recon_hook(value, hook):
            return sae.decode(sae.encode(value))
        logits_recon = model.run_with_hooks(input_ids, attention_mask=attn_mask, fwd_hooks=[(hook_name, recon_hook)])
        loss_recon = F.cross_entropy(logits_recon.reshape(-1, logits_recon.size(-1)), input_ids.reshape(-1), reduction="none")
        losses["recon"].append((loss_recon * attn_mask.reshape(-1)).sum().item() / attn_mask.sum().item())

        def zero_hook(value, hook):
            return torch.zeros_like(value)
        logits_zero = model.run_with_hooks(input_ids, attention_mask=attn_mask, fwd_hooks=[(hook_name, zero_hook)])
        loss_zero = F.cross_entropy(logits_zero.reshape(-1, logits_zero.size(-1)), input_ids.reshape(-1), reduction="none")
        losses["zero_ablation"].append((loss_zero * attn_mask.reshape(-1)).sum().item() / attn_mask.sum().item())

    base = np.mean(losses["base"])
    recon = np.mean(losses["recon"])
    zero = np.mean(losses["zero_ablation"])
    recovered = (zero - recon) / (zero - base + 1e-8)
    return {
        "ce_loss_base": float(base),
        "ce_loss_recon": float(recon),
        "ce_loss_zero": float(zero),
        "ce_loss_recovered": float(recovered),
    }


# ---------------------------------------------------------------------------
# Fast torch-based probe for absorption
# ---------------------------------------------------------------------------
FIRST_LETTER_WORDS = [
    "apple", "banana", "cherry", "date", "elderberry",
    "fig", "grape", "honeydew", "kiwi", "lemon",
    "mango", "nectarine", "orange", "papaya", "quince",
    "raspberry", "strawberry", "tangerine", "ugli", "vanilla",
    "watermelon", "xigua", "yam", "zucchini",
    "america", "brazil", "canada", "denmark", "egypt",
]


def make_first_letter_labels(words):
    letters = [w[0].lower() for w in words]
    unique = sorted(set(letters))
    letter2idx = {l: i for i, l in enumerate(unique)}
    labels = [letter2idx[l] for l in letters]
    return labels, unique


def fast_linear_probe_accuracy(X, y, n_epochs=100, lr=0.1):
    X = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y = torch.tensor(y, dtype=torch.long, device=DEVICE)
    n_classes = int(y.max().item()) + 1
    probe = nn.Linear(X.size(1), n_classes, device=DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = probe(X)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        preds = probe(X).argmax(dim=-1)
        acc = (preds == y).float().mean().item()
    return acc


def compute_absorption_metric(model, sae, tokenizer, words=FIRST_LETTER_WORDS):
    labels, unique_letters = make_first_letter_labels(words)
    prompts = [f"{w} has the first letter:" for w in words]
    hook_name = get_hook_name(sae)
    hook_layer = get_hook_layer(sae)

    with torch.no_grad():
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

        _, cache = model.run_with_cache(tokens["input_ids"], stop_at_layer=hook_layer + 1, names_filter=[hook_name])
        acts = cache[hook_name]

        word_acts = []
        for i in range(len(prompts)):
            pos = 1
            word_acts.append(acts[i, pos, :].cpu().numpy())
        word_acts = np.stack(word_acts, axis=0)
        # For SAE encode, pass original shape (hook_z needs 3D/4D)
        word_acts_sae_in = word_acts
        # For probe, flatten to 2D
        if word_acts.ndim > 2 or (word_acts.ndim == 2 and word_acts.shape[1] != sae.cfg.d_in):
            word_acts = word_acts.reshape(word_acts.shape[0], -1)
        sae_acts = sae.encode(torch.tensor(word_acts_sae_in, dtype=torch.float32, device=DEVICE)).cpu().numpy()

    labels_arr = np.array(labels)
    resid_acc = fast_linear_probe_accuracy(word_acts, labels_arr, n_epochs=200, lr=0.05)
    sae_acc = fast_linear_probe_accuracy(sae_acts, labels_arr, n_epochs=200, lr=0.05)
    absorption_score = max(0.0, (resid_acc - sae_acc) / (resid_acc + 1e-8))

    return {
        "resid_probe_accuracy": float(resid_acc),
        "sae_probe_accuracy": float(sae_acc),
        "absorption_score": float(absorption_score),
        "n_words": len(words),
        "n_classes": len(unique_letters),
    }


# ---------------------------------------------------------------------------
# Hedging metric
# ---------------------------------------------------------------------------
HEDGING_PAIRS = [
    ("king", "queen"), ("man", "woman"), ("brother", "sister"),
    ("hot", "cold"), ("big", "small"), ("happy", "sad"),
    ("fast", "slow"), ("rich", "poor"), ("old", "young"),
    ("love", "hate"), ("war", "peace"), ("day", "night"),
]


@torch.no_grad()
def compute_hedging_metric(model, sae, tokenizer, pairs=HEDGING_PAIRS):
    prompts_a = [f"The word {p[0]} means" for p in pairs]
    prompts_b = [f"The word {p[1]} means" for p in pairs]
    hook_name = get_hook_name(sae)
    hook_layer = get_hook_layer(sae)

    def get_latents(prompts):
        tokens = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        _, cache = model.run_with_cache(tokens["input_ids"], stop_at_layer=hook_layer + 1, names_filter=[hook_name])
        acts = cache[hook_name]
        mask = tokens["attention_mask"]
        # For 4D acts (hook_z), pool over pos while keeping last dims
        if acts.ndim == 4:
            mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).float()  # [batch, pos, 1, 1]
            acts_pooled = (acts * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1)  # [batch, n_heads, d_head]
        else:
            mask = mask.unsqueeze(-1).float()
            acts_pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1)
        sae_acts = sae.encode(acts_pooled)
        topk = torch.topk(sae_acts, k=min(10, sae_acts.size(-1)), dim=-1)
        return topk.indices.cpu().numpy()

    topk_a = get_latents(prompts_a)
    topk_b = get_latents(prompts_b)

    jaccards = []
    for i in range(len(pairs)):
        set_a = set(topk_a[i])
        set_b = set(topk_b[i])
        inter = len(set_a & set_b)
        union = len(set_a | set_b)
        jaccards.append(inter / union if union > 0 else 0.0)

    return {
        "mean_topk_overlap": float(np.mean([len(set(topk_a[i]) & set(topk_b[i])) for i in range(len(pairs))])),
        "mean_jaccard": float(np.mean(jaccards)),
        "hedging_score": float(np.mean(jaccards)),
        "n_pairs": len(pairs),
    }


# ---------------------------------------------------------------------------
# Pareto / dominance helpers
# ---------------------------------------------------------------------------
def normalize_metrics(results):
    """Normalize each metric to [0,1] across all successful checkpoints."""
    successful = [r for r in results if r.get("status") == "success"]
    if not successful:
        return results

    keys = ["absorption_score", "hedging_score", "explained_variance", "ce_loss_recovered", "l0_inv", "dead_neuron_inv"]
    # Invert L0 and dead_neuron so higher is better (lower pathology)
    for r in successful:
        r["l0_inv"] = 1.0 / (r["l0"] + 1.0)
        r["dead_neuron_inv"] = 1.0 - r["dead_neuron_rate"]

    mins = {}
    maxs = {}
    for k in keys:
        vals = [r[k] for r in successful]
        mins[k] = min(vals)
        maxs[k] = max(vals)

    for r in successful:
        for k in keys:
            denom = maxs[k] - mins[k]
            r[f"norm_{k}"] = (r[k] - mins[k]) / (denom + 1e-8)
    return results


def is_pareto_dominated(point, others, objectives):
    """Check if point is dominated by any other point (maximization)."""
    for other in others:
        if other is point:
            continue
        strictly_better = False
        at_least_as_good = True
        for obj in objectives:
            if other.get(obj, 0) < point.get(obj, 0) - 1e-8:
                at_least_as_good = False
                break
            if other.get(obj, 0) > point.get(obj, 0) + 1e-8:
                strictly_better = True
        if at_least_as_good and strictly_better:
            return True
    return False


def compute_pareto_fronts(results):
    objectives = ["norm_absorption_score", "norm_hedging_score", "norm_explained_variance",
                  "norm_ce_loss_recovered", "norm_l0_inv", "norm_dead_neuron_inv"]
    successful = [r for r in results if r.get("status") == "success"]
    for r in successful:
        r["is_pareto"] = not is_pareto_dominated(r, successful, objectives)
    return results


def mann_whitney_dominance(group_a, group_b, metric_key):
    vals_a = [r[metric_key] for r in group_a if metric_key in r]
    vals_b = [r[metric_key] for r in group_b if metric_key in r]
    if len(vals_a) < 2 or len(vals_b) < 2:
        return {"U": None, "p": None, "mean_a": np.mean(vals_a) if vals_a else None, "mean_b": np.mean(vals_b) if vals_b else None}
    U, p = stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")
    return {"U": float(U), "p": float(p), "mean_a": float(np.mean(vals_a)), "mean_b": float(np.mean(vals_b))}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()
    report_progress(0, len(CHECKPOINTS), step=0, total_steps=len(CHECKPOINTS))

    print(f"Loading GPT-2 Small on {DEVICE} (GPU {GPU_ID})...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = model.tokenizer

    # GPU profile
    gpu_name = torch.cuda.get_device_name(DEVICE)
    vram_total = torch.cuda.get_device_properties(DEVICE).total_memory // (1024 * 1024)
    GPU_PROFILE_FILE.write_text(json.dumps({
        "gpu_name": gpu_name,
        "vram_total_mb": int(vram_total),
        "batch_size": BATCH_SIZE,
    }))

    print("Loading corpus...")
    texts = load_text_corpus(n_samples=N_SAMPLES)
    print(f"Loaded {len(texts)} texts")

    results = []

    for idx, ckpt in enumerate(CHECKPOINTS):
        print(f"\n{'='*60}")
        print(f"Evaluating {ckpt['name']} ({idx+1}/{len(CHECKPOINTS)})")
        print(f"{'='*60}")
        report_progress(idx, len(CHECKPOINTS), step=idx+1, total_steps=len(CHECKPOINTS),
                        metric={"current_checkpoint": ckpt["name"]})

        try:
            sae = SAE.from_pretrained(ckpt["release"], ckpt["sae_id"], device=DEVICE)
            sae.eval()
        except Exception as e:
            print(f"FAILED to load {ckpt['name']}: {e}")
            results.append({"checkpoint": ckpt["name"], "architecture": ckpt["architecture"],
                            "release": ckpt["release"], "sae_id": ckpt["sae_id"],
                            "status": "load_failed", "error": str(e)})
            continue

        t0 = time.time()
        basic = collect_activations(model, sae, texts, tokenizer)
        dead_rate = compute_dead_neuron_rate(basic["acts"])
        ce_metrics = compute_ce_loss_recovered(model, sae, texts, tokenizer)
        absorption = compute_absorption_metric(model, sae, tokenizer)
        hedging = compute_hedging_metric(model, sae, tokenizer)
        elapsed = time.time() - t0

        result = {
            "checkpoint": ckpt["name"],
            "architecture": ckpt["architecture"],
            "release": ckpt["release"],
            "sae_id": ckpt["sae_id"],
            "status": "success",
            "d_in": sae.cfg.d_in,
            "d_sae": sae.cfg.d_sae,
            "l0": basic["l0"],
            "explained_variance": basic["explained_variance"],
            "recon_mse": basic["recon_mse"],
            "dead_neuron_rate": dead_rate,
            "ce_loss_recovered": ce_metrics["ce_loss_recovered"],
            "absorption_score": absorption["absorption_score"],
            "resid_probe_accuracy": absorption["resid_probe_accuracy"],
            "sae_probe_accuracy": absorption["sae_probe_accuracy"],
            "hedging_score": hedging["hedging_score"],
            "mean_topk_overlap": hedging["mean_topk_overlap"],
            "eval_time_sec": elapsed,
        }
        results.append(result)
        print(json.dumps({k: v for k, v in result.items() if k not in {"release", "sae_id"}}, indent=2))

        del sae, basic["acts"]
        torch.cuda.empty_cache()

    total_time = time.time() - start_time

    # Normalization and Pareto
    results = normalize_metrics(results)
    results = compute_pareto_fronts(results)

    # Architecture family summaries
    successful = [r for r in results if r.get("status") == "success"]
    arch_families = {}
    for r in successful:
        arch = r["architecture"]
        arch_families.setdefault(arch, []).append(r)

    dominance_tests = {}
    archs = list(arch_families.keys())
    for i in range(len(archs)):
        for j in range(i + 1, len(archs)):
            a, b = archs[i], archs[j]
            key = f"{a}_vs_{b}"
            dominance_tests[key] = {
                "absorption": mann_whitney_dominance(arch_families[a], arch_families[b], "absorption_score"),
                "hedging": mann_whitney_dominance(arch_families[a], arch_families[b], "hedging_score"),
                "explained_variance": mann_whitney_dominance(arch_families[a], arch_families[b], "explained_variance"),
                "ce_loss_recovered": mann_whitney_dominance(arch_families[a], arch_families[b], "ce_loss_recovered"),
            }

    out = {
        "task_id": "e1_full_gpt2",
        "model": "gpt2-small",
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "n_samples": N_SAMPLES,
        "seed": SEED,
        "total_time_sec": total_time,
        "n_checkpoints_evaluated": len(CHECKPOINTS),
        "n_successful": len(successful),
        "n_pareto": sum(1 for r in successful if r.get("is_pareto")),
        "checkpoints": results,
        "architecture_summary": {
            arch: {
                "count": len(arch_families[arch]),
                "mean_absorption": float(np.mean([r["absorption_score"] for r in arch_families[arch]])),
                "mean_hedging": float(np.mean([r["hedging_score"] for r in arch_families[arch]])),
                "mean_explained_variance": float(np.mean([r["explained_variance"] for r in arch_families[arch]])),
                "mean_ce_loss_recovered": float(np.mean([r["ce_loss_recovered"] for r in arch_families[arch]])),
                "mean_l0": float(np.mean([r["l0"] for r in arch_families[arch]])),
                "mean_dead_neuron_rate": float(np.mean([r["dead_neuron_rate"] for r in arch_families[arch]])),
            }
            for arch in arch_families
        },
        "dominance_tests": dominance_tests,
    }

    out_dir = RESULTS_DIR / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e1_full_gpt2_pareto_results.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved to {out_path}")

    # Summary markdown
    md_path = out_dir / "e1_full_gpt2_summary.md"
    md_lines = [
        "# E1 Full: GPT-2 Small Pareto Evaluation",
        "",
        f"**Checkpoints Evaluated:** {len(successful)}/{len(CHECKPOINTS)}",
        f"**Pareto Points:** {out['n_pareto']}",
        f"**Total Time:** {total_time:.0f}s",
        "",
        "## Architecture Family Summary",
        "",
        "| Architecture | Count | Absorption | Hedging | Explained Var | CE Recovered | L0 | Dead Neurons |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arch, s in out["architecture_summary"].items():
        md_lines.append(
            f"| {arch} | {s['count']} | {s['mean_absorption']:.3f} | {s['mean_hedging']:.3f} | "
            f"{s['mean_explained_variance']:.3f} | {s['mean_ce_loss_recovered']:.3f} | {s['mean_l0']:.1f} | {s['mean_dead_neuron_rate']:.4f} |"
        )
    md_lines.append("")
    md_lines.append("## Per-Checkpoint Results")
    md_lines.append("")
    md_lines.append("| Checkpoint | Arch | L0 | ExpVar | Dead | CE Rec | Absorption | Hedging | Pareto |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|")
    for r in successful:
        md_lines.append(
            f"| {r['checkpoint']} | {r['architecture']} | {r['l0']:.1f} | {r['explained_variance']:.3f} | "
            f"{r['dead_neuron_rate']:.4f} | {r['ce_loss_recovered']:.3f} | {r['absorption_score']:.3f} | {r['hedging_score']:.3f} | {'Yes' if r['is_pareto'] else 'No'} |"
        )
    md_lines.append("")
    md_lines.append("## Dominance Tests (Mann-Whitney U)")
    md_lines.append("")
    for test_key, test_vals in dominance_tests.items():
        md_lines.append(f"### {test_key}")
        md_lines.append("| Metric | U | p | Mean A | Mean B |")
        md_lines.append("|---|---|---|---|---|")
        for metric, vals in test_vals.items():
            u = f"{vals['U']:.1f}" if vals['U'] is not None else "N/A"
            p = f"{vals['p']:.4f}" if vals['p'] is not None else "N/A"
            ma = f"{vals['mean_a']:.3f}" if vals['mean_a'] is not None else "N/A"
            mb = f"{vals['mean_b']:.3f}" if vals['mean_b'] is not None else "N/A"
            md_lines.append(f"| {metric} | {u} | {p} | {ma} | {mb} |")
        md_lines.append("")
    md_path.write_text("\n".join(md_lines))
    print(f"Summary saved to {md_path}")

    mark_done(status="success", summary=f"E1 full GPT-2 complete. {len(successful)}/{len(CHECKPOINTS)} checkpoints, {out['n_pareto']} Pareto points, {total_time:.0f}s.")
    return out


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        raise
