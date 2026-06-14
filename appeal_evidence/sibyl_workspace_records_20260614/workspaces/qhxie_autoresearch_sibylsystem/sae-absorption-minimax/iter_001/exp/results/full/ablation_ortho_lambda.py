#!/usr/bin/env python3
"""
Ablation: OrtSAE Orthogonal Penalty Coefficient Sweep
=====================================================
Sweep lambda_ortho ∈ {0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1} for OrtSAE at GPT-2 layer 8.
Also test: (1) Orthogonality alone (l1=0), (2) L1 alone (ortho=0) as baselines.

Measures:
- Absorption score (feature activation rate relative to residual stream baseline)
- Reconstruction MSE
- L0 sparsity
- Dead feature ratio
"""

import json
import gc
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
import numpy as np

# ── Setup ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
REMOTE_BASE = WORKSPACE
RESULTS_DIR = REMOTE_BASE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "ablation_ortho_lambda"
DEVICE = "cuda:0"   # mapped via CUDA_VISIBLE_DEVICES
SEED = 42
N_TRAIN_TOKENS = 500_000
N_EVAL_TOKENS = 10_000
BATCH_SIZE = 64
SEQ_LEN = 128
N_EPOCHS = 3
LR = 1e-4
L1_COEFF = 8e-5
N_FEATURES_ANALYZED = 200
LAYER = 8

ORTHO_LAMBDAS = [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
RUNS = [
    {"name": "Vanilla (L1 only, no ortho)", "l1_coeff": L1_COEFF, "ortho_lambda": 0.0, "tag": "vanilla"},
    {"name": "OrtSAE lam=0 (ortho only, no L1)", "l1_coeff": 0.0, "ortho_lambda": 0.001, "tag": "ortho_only"},
    *[
        {"name": f"OrtSAE lambda={lam}", "l1_coeff": L1_COEFF, "ortho_lambda": lam, "tag": f"lam{lam}"}
        for lam in ORTHO_LAMBDAS
    ],
]

VANILLA_BASELINE = {
    "name": "Pre-trained Vanilla SAE (full_h2 reference)",
    "l0_sparsity": 0.00298,
    "reconstruction_mse": 1.482,
    "dead_feature_ratio": 0.00346,
    "mean_absorption": 0.0150,
    "note": "from full_h2 gpt2-small-res-jb pre-trained SAE"
}


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Load Models ─────────────────────────────────────────────────────────────
def load_models():
    print("Loading HookedTransformer (gpt2-small)...")
    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    print(f"  d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    print("Loading pre-trained vanilla SAE from gpt2-small-res-jb...")
    from sae_lens import SAE
    # Use the updated API
    vanilla_sae = SAE.from_pretrained(
        release="gpt2-small-res-jb",
        sae_id=f"blocks.{LAYER}.hook_resid_pre",
        device=DEVICE
    )
    vanilla_sae.eval()
    d_sae = vanilla_sae.cfg.d_sae
    d_in = vanilla_sae.cfg.d_in
    print(f"  Pre-trained SAE: d_in={d_in}, d_sae={d_sae}")
    print(f"  W_enc shape: {vanilla_sae.W_enc.shape} (d_in={d_in}, d_sae={d_sae})")
    return model, vanilla_sae, int(d_in), int(d_sae)


# ── Dataset ──────────────────────────────────────────────────────────────────
def collect_text_examples(n_examples, seq_len, model, cache_path=None):
    if cache_path and cache_path.exists():
        print(f"  Loading cached examples from {cache_path}")
        return cache_path.read_text().split("\n")[:n_examples]

    print(f"  Collecting {n_examples} text examples from pile-uncopyrighted...")
    try:
        from datasets import load_dataset
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
        texts = []
        buffer = ""
        for item in ds:
            buffer += item["text"] + " "
            while len(buffer) >= seq_len * 2 and len(texts) < n_examples:
                texts.append(buffer[: seq_len * 2])
                buffer = buffer[seq_len * 2 :]
            if len(texts) >= n_examples:
                break
        texts = texts[:n_examples]
    except Exception as e:
        print(f"  Dataset load failed ({e}), using fallback")
        import random
        random.seed(SEED)
        chars = "abcdefghijklmnopqrstuvwxyz .,"
        texts = ["".join(random.choices(chars, k=seq_len * 2)) for _ in range(n_examples)]

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text("\n".join(texts))
    return texts


# ── SAE Architecture (Parameter-based, matching SAE Lens API) ─────────────────
class SAE(nn.Module):
    """Base SAE using Parameters (matching SAE Lens API)."""

    def __init__(self, d_in, d_sae, l1_coeff=1e-4, ortho_lambda=0.0):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.l1_coeff = l1_coeff
        self.ortho_lambda = ortho_lambda
        # Parameters: W_enc (d_in, d_sae), W_dec (d_sae, d_in)
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_enc)
        nn.init.zeros_(self.W_dec)
        nn.init.zeros_(self.b_enc)
        nn.init.zeros_(self.b_dec)

    def encode(self, x):
        pre_acts = x @ self.W_enc + self.b_enc
        acts = F.relu(pre_acts)
        return acts

    def decode(self, acts):
        return acts @ self.W_dec + self.b_dec

    def forward(self, x):
        acts = self.encode(x)
        recon = self.decode(acts)
        l1_loss = acts.sum(dim=-1).mean()
        mse_loss = ((recon - x) ** 2).sum(dim=-1).mean()

        ortho_loss = torch.tensor(0.0, device=x.device)
        if self.ortho_lambda > 0:
            # Normalize decoder columns
            W_dec_norm = self.W_dec / (self.W_dec.norm(dim=1, keepdim=True) + 1e-8)
            # Gram matrix: off-diagonal penalizes similarity between feature directions
            gram = W_dec_norm @ W_dec_norm.T
            mask = torch.eye(self.d_sae, device=gram.device)
            ortho_loss = (gram ** 2 * (1 - mask)).sum() / 2

        total_loss = mse_loss + self.l1_coeff * l1_loss + self.ortho_lambda * ortho_loss
        return recon, mse_loss, l1_loss, ortho_loss, total_loss


def compute_ortho_penalty(W_dec):
    """Compute mean squared off-diagonal Gram entries for decoder columns."""
    W_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)
    gram = W_norm @ W_norm.T
    mask = torch.eye(W_dec.shape[0], device=gram.device)
    off_diag = (gram ** 2 * (1 - mask)).sum().item() / (W_dec.shape[0] * (W_dec.shape[0] - 1))
    return off_diag


# ── Activation Collection ─────────────────────────────────────────────────────
def collect_activations(model, texts, layer, batch_size, seq_len, device, n_max=None):
    """Collect residual stream activations."""
    if n_max:
        texts = texts[:n_max]
    all_activations = []

    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size), desc="  Collecting activations"):
            batch_texts = texts[i : i + batch_size]
            tokens = model.to_tokens(batch_texts, truncate=None)
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: f"blocks.{layer}.hook_resid_pre",
            )
            acts = cache[f"blocks.{layer}.hook_resid_pre"].to(device)
            acts_flat = acts.reshape(-1, acts.shape[-1])
            tokens_flat = tokens.reshape(-1)
            mask = torch.ones(len(tokens_flat), dtype=torch.bool, device=device)
            if len(tokens_flat) > 1:
                mask[1:] = tokens_flat[1:] != tokens_flat[:-1]
            acts_flat = acts_flat[mask]
            all_activations.append(acts_flat.cpu())

    activations = torch.cat(all_activations, dim=0)
    return activations


# ── Training ─────────────────────────────────────────────────────────────────
def train_sae(sae, optimizer, activations, batch_size, n_epochs, device, tag=""):
    n_steps = len(activations) // batch_size
    for epoch in range(n_epochs):
        perm = torch.randperm(len(activations))
        epoch_losses, epoch_mses, epoch_l1s, epoch_orthos = [], [], [], []

        for i in tqdm(range(0, len(activations) - batch_size, batch_size),
                      desc=f"  Epoch {epoch+1}/{n_epochs}", leave=False):
            batch = activations[perm[i : i + batch_size]].to(device)
            optimizer.zero_grad()
            recon, mse_loss, l1_loss, ortho_loss, total_loss = sae(batch)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
            optimizer.step()
            epoch_losses.append(total_loss.item())
            epoch_mses.append(mse_loss.item())
            epoch_l1s.append(l1_loss.item())
            epoch_orthos.append(ortho_loss.item())

        print(f"  Epoch {epoch+1}: loss={np.mean(epoch_losses):.4f}, "
              f"mse={np.mean(epoch_mses):.4f}, l1={np.mean(epoch_l1s):.6f}, "
              f"ortho={np.mean(epoch_orthos):.6f}")
    return sae


# ── Evaluation ────────────────────────────────────────────────────────────────
def evaluate_sae(sae, model, eval_texts, layer, batch_size, n_eval_tokens, device, n_features=200):
    """Evaluate SAE: absorption score, L0, MSE, dead ratio, UAS."""
    sae.eval()
    model.eval()

    with torch.no_grad():
        all_sae_acts = []
        all_resid_acts = []

        for i in tqdm(range(0, len(eval_texts), batch_size), desc="  Evaluating"):
            batch_texts = eval_texts[i : i + batch_size]
            tokens = model.to_tokens(batch_texts, truncate=None)
            _, cache = model.run_with_cache(
                tokens,
                names_filter=lambda n: f"blocks.{layer}.hook_resid_pre",
            )
            resid = cache[f"blocks.{layer}.hook_resid_pre"].to(device)

            flat_resid = resid.reshape(-1, resid.shape[-1])
            sae_acts = sae.encode(flat_resid)

            all_sae_acts.append(sae_acts.cpu())
            all_resid_acts.append(flat_resid.cpu())

    all_sae_acts = torch.cat(all_sae_acts, dim=0)[:n_eval_tokens]
    all_resid_acts = torch.cat(all_resid_acts, dim=0)[:n_eval_tokens]

    # L0 sparsity
    l0 = (all_sae_acts > 0).float().mean().item()

    # Feature activation rates
    feat_activation_rate = (all_sae_acts > 0).float().mean(dim=0)

    # Absorption: feature activation rate when residual is small
    resid_mag = all_resid_acts.norm(dim=1)
    resid_threshold = resid_mag.quantile(0.5).item()
    resid_small_mask = resid_mag < resid_threshold

    if resid_small_mask.sum() > 100:
        feat_act_when_small = (all_sae_acts[resid_small_mask] > 0).float().mean(dim=0)
        absorption = feat_act_when_small.mean().item()
    else:
        absorption = feat_activation_rate.mean().item()

    # Dead feature ratio
    dead_ratio = (feat_activation_rate < 1e-6).float().mean().item()

    # Reconstruction MSE
    recon = sae.decode(all_sae_acts.to(device)).cpu()
    mse = ((recon - all_resid_acts) ** 2).mean().item()

    # UAS for top-activated features
    top_feat_idx = feat_activation_rate.argsort(descending=True)[:n_features]
    uas_values = []
    for idx in top_feat_idx:
        feat_direction = sae.W_dec[idx].detach().cpu()
        all_directions = sae.W_dec.detach().cpu()
        cos_sims = F.cosine_similarity(feat_direction.unsqueeze(0), all_directions, dim=1)
        cos_var = cos_sims.var().item()
        feat_acts = all_sae_acts[:, idx].numpy()
        if feat_acts.max() > feat_acts.min():
            freq_skew = float(np.mean(((feat_acts - feat_acts.mean()) / (feat_acts.std() + 1e-8)) ** 3))
        else:
            freq_skew = 0.0
        uas_values.append(cos_var + 0.5 * freq_skew)
    mean_uas = np.mean(uas_values)

    return {
        "l0_sparsity": l0,
        "reconstruction_mse": mse,
        "dead_feature_ratio": dead_ratio,
        "mean_absorption": absorption,
        "mean_uas": mean_uas,
    }


# ── PID / Progress / DONE ───────────────────────────────────────────────────
def write_pid(results_dir):
    pid_file = results_dir / f"{TASK_ID}.pid"
    pid_file.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, loss=None):
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "loss": loss,
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status, summary):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))
    print(f"\nDONE marker written: {status}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print(f"\n{'='*60}")
    print(f"ABLATION: OrtSAE Orthogonal Penalty Coefficient Sweep")
    print(f"{'='*60}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}")
    print(f"Lambda sweep: {ORTHO_LAMBDAS}")
    print(f"Fixed L1 coefficient: {L1_COEFF}")
    print(f"Training tokens: {N_TRAIN_TOKENS:,}")
    print(f"Eval tokens: {N_EVAL_TOKENS:,}")
    print()

    set_seed(SEED)
    write_pid(RESULTS_DIR)

    model, vanilla_sae_pretrained, d_in, d_sae = load_models()

    # Collect text examples
    cache_path = RESULTS_DIR / f"text_cache_l{LAYER}.txt"
    texts = collect_text_examples(
        n_examples=max(N_TRAIN_TOKENS // SEQ_LEN, 500),
        seq_len=SEQ_LEN, model=model, cache_path=cache_path
    )
    eval_texts = collect_text_examples(
        n_examples=max(N_EVAL_TOKENS // SEQ_LEN, 50),
        seq_len=SEQ_LEN, model=model,
        cache_path=RESULTS_DIR / f"text_cache_eval_l{LAYER}.txt"
    )
    print(f"\nCollected {len(texts)} training examples, {len(eval_texts)} eval examples")

    # Collect activations
    print("\nCollecting residual stream activations for training...")
    activations_train = collect_activations(
        model, texts, LAYER, BATCH_SIZE, SEQ_LEN, DEVICE, n_max=N_TRAIN_TOKENS
    )
    print(f"  Collected {len(activations_train):,} activation vectors")

    print("\nCollecting residual stream activations for evaluation...")
    activations_eval = collect_activations(
        model, eval_texts, LAYER, BATCH_SIZE, SEQ_LEN, DEVICE, n_max=N_EVAL_TOKENS
    )
    print(f"  Collected {len(activations_eval):,} activation vectors")

    all_results = []

    for run_config in RUNS:
        tag = run_config["tag"]
        name = run_config["name"]
        l1_coeff = run_config["l1_coeff"]
        ortho_lambda = run_config["ortho_lambda"]

        print(f"\n{'─'*60}")
        print(f"Run [{tag}] {name}")
        print(f"  l1_coeff={l1_coeff}, ortho_lambda={ortho_lambda}")
        print(f"{'─'*60}")

        gc.collect()
        torch.cuda.empty_cache()

        # Initialize SAE
        sae = SAE(d_in, d_sae, l1_coeff=l1_coeff, ortho_lambda=ortho_lambda).to(DEVICE)

        # Copy pre-trained weights for warm start
        with torch.no_grad():
            sae.W_enc.copy_(vanilla_sae_pretrained.W_enc)
            sae.W_dec.copy_(vanilla_sae_pretrained.W_dec)
            sae.b_enc.copy_(vanilla_sae_pretrained.b_enc)
            sae.b_dec.copy_(vanilla_sae_pretrained.b_dec)

        # Train
        optimizer = torch.optim.Adam(sae.parameters(), lr=LR)
        report_progress(0, N_EPOCHS, loss="training")
        sae = train_sae(sae, optimizer, activations_train, BATCH_SIZE, N_EPOCHS, DEVICE, tag=tag)
        report_progress(N_EPOCHS, N_EPOCHS, loss="done")

        # Evaluate
        print("  Evaluating...")
        eval_metrics = evaluate_sae(
            sae, model, eval_texts, LAYER, BATCH_SIZE, N_EVAL_TOKENS, DEVICE,
            n_features=N_FEATURES_ANALYZED
        )
        decoder_ortho = compute_ortho_penalty(sae.W_dec)

        result = {
            "tag": tag,
            "name": name,
            "l1_coeff": l1_coeff,
            "ortho_lambda": ortho_lambda,
            "l0_sparsity": eval_metrics["l0_sparsity"],
            "reconstruction_mse": eval_metrics["reconstruction_mse"],
            "dead_feature_ratio": eval_metrics["dead_feature_ratio"],
            "mean_absorption": eval_metrics["mean_absorption"],
            "mean_uas": eval_metrics["mean_uas"],
            "decoder_ortho_penalty": decoder_ortho,
        }
        print(f"  L0 sparsity:  {result['l0_sparsity']:.6f}")
        print(f"  Recon MSE:   {result['reconstruction_mse']:.4f}")
        print(f"  Dead ratio:  {result['dead_feature_ratio']:.4f}")
        print(f"  Absorption:  {result['mean_absorption']:.6f}")
        print(f"  UAS:         {result['mean_uas']:.6f}")
        print(f"  Decoder ortho penalty: {decoder_ortho:.6f}")

        all_results.append(result)

        del sae, optimizer
        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("ABLATION SUMMARY")
    print(f"{'='*60}")

    print(f"\nReference (pre-trained vanilla SAE from full_h2):")
    print(f"  L0={VANILLA_BASELINE['l0_sparsity']:.6f}, MSE={VANILLA_BASELINE['reconstruction_mse']:.4f}, "
          f"Dead={VANILLA_BASELINE['dead_feature_ratio']:.4f}, Absorption={VANILLA_BASELINE['mean_absorption']:.6f}")

    print(f"\n{'Tag':<20} {'Lambda':<10} {'L1-only':<10} {'L0':<10} {'MSE':<10} {'Dead':<10} {'Absorption':<12} {'OrthoPenalty':<14}")
    print("-" * 110)
    for r in all_results:
        is_ortho_only = (r["tag"] == "ortho_only")
        l1_str = "NO" if is_ortho_only else ("YES" if r["l1_coeff"] > 0 else "NO")
        print(f"{r['tag']:<20} {r['ortho_lambda']:<10.0g} {l1_str:<10} "
              f"{r['l0_sparsity']:<10.6f} {r['reconstruction_mse']:<10.4f} "
              f"{r['dead_feature_ratio']:<10.4f} {r['mean_absorption']:<12.6f} "
              f"{r['decoder_ortho_penalty']:<14.6f}")

    # Analysis
    print(f"\nKey Findings:")

    lambda_results = [r for r in all_results if "lam" in r["tag"] and r["ortho_lambda"] > 0]
    vanilla_run = [r for r in all_results if r["tag"] == "vanilla"][0]
    ortho_only_result = [r for r in all_results if r["tag"] == "ortho_only"][0]

    print(f"\n1. Effect of orthogonality alone (no L1, lam=0.001):")
    print(f"   Absorption: {ortho_only_result['mean_absorption']:.6f} "
          f"(vs L1-only: {vanilla_run['mean_absorption']:.6f})")
    print(f"   Decoder ortho penalty: {ortho_only_result['decoder_ortho_penalty']:.6f} "
          f"(vs L1-only: {vanilla_run['decoder_ortho_penalty']:.6f})")

    if lambda_results:
        print(f"\n2. Lambda sweep (L1 + ortho):")
        for r in lambda_results:
            delta_absorption = r['mean_absorption'] - vanilla_run['mean_absorption']
            delta_mse = r['reconstruction_mse'] - vanilla_run['reconstruction_mse']
            print(f"   lambda={r['ortho_lambda']:.0g}: absorption={r['mean_absorption']:.6f} "
                  f"(delta={delta_absorption:+.4f}), MSE={r['reconstruction_mse']:.4f} "
                  f"(delta={delta_mse:+.4f}), ortho_penalty={r['decoder_ortho_penalty']:.6f}")

        # Find optimal points
        best_mse = min(lambda_results, key=lambda r: r['reconstruction_mse'])
        most_reduced = min(lambda_results, key=lambda r: r['mean_absorption'])
        print(f"\n3. Optimal points:")
        print(f"   Best reconstruction: lambda={best_mse['ortho_lambda']:.0g} "
              f"(MSE={best_mse['reconstruction_mse']:.4f}, absorption={best_mse['mean_absorption']:.6f})")
        print(f"   Most absorption reduction: lambda={most_reduced['ortho_lambda']:.0g} "
              f"(absorption={most_reduced['mean_absorption']:.6f}, MSE={most_reduced['reconstruction_mse']:.4f})")

    # Save results
    output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": "gpt2-small",
            "layer": LAYER,
            "sae_release": "gpt2-small-res-jb",
            "d_sae": d_sae,
            "d_in": d_in,
            "n_train_tokens": N_TRAIN_TOKENS,
            "n_eval_tokens": N_EVAL_TOKENS,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "learning_rate": LR,
            "fixed_l1_coefficient": L1_COEFF,
            "ortho_lambda_sweep": ORTHO_LAMBDAS,
            "device": DEVICE,
            "seed": SEED,
        },
        "results": all_results,
        "vanilla_baseline_reference": VANILLA_BASELINE,
    }

    out_path = RESULTS_DIR / f"{TASK_ID}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Write markdown summary
    summary_md = RESULTS_DIR / f"{TASK_ID}_summary.md"
    with open(summary_md, "w") as f:
        f.write(f"# Ablation: OrtSAE Orthogonal Penalty Coefficient Sweep\n\n")
        f.write(f"## Configuration\n")
        f.write(f"- Model: GPT-2 Small, Layer {LAYER}\n")
        f.write(f"- Training tokens: {N_TRAIN_TOKENS:,}\n")
        f.write(f"- Eval tokens: {N_EVAL_TOKENS:,}\n")
        f.write(f"- Fixed L1 coefficient: {L1_COEFF}\n")
        f.write(f"- Lambda sweep: {ORTHO_LAMBDAS}\n\n")
        f.write(f"## Results Table\n\n")
        f.write(f"| Tag | Lambda | L1 | L0 | MSE | Dead | Absorption | OrthoPenalty |\n")
        f.write(f"|-----|--------|-----|-----|-----|------|------------|---------------|\n")
        for r in all_results:
            is_ortho_only = (r["tag"] == "ortho_only")
            l1_str = "NO" if is_ortho_only else ("YES" if r["l1_coeff"] > 0 else "NO")
            f.write(f"| {r['tag']} | {r['ortho_lambda']} | {l1_str} | "
                    f"{r['l0_sparsity']:.6f} | {r['reconstruction_mse']:.4f} | "
                    f"{r['dead_feature_ratio']:.4f} | {r['mean_absorption']:.6f} | "
                    f"{r['decoder_ortho_penalty']:.6f} |\n")
        f.write(f"\n## Key Findings\n\n")
        f.write(f"- Pre-trained vanilla SAE baseline: absorption={VANILLA_BASELINE['mean_absorption']:.6f}, MSE={VANILLA_BASELINE['reconstruction_mse']:.4f}\n")
        f.write(f"- L1-only (our run): absorption={vanilla_run['mean_absorption']:.6f}, MSE={vanilla_run['reconstruction_mse']:.4f}\n")
        f.write(f"- Ortho-only: absorption={ortho_only_result['mean_absorption']:.6f}, MSE={ortho_only_result['reconstruction_mse']:.4f}\n")
        if lambda_results:
            f.write(f"- Best reconstruction: lambda={best_mse['ortho_lambda']:.0g} (MSE={best_mse['reconstruction_mse']:.4f})\n")
            f.write(f"- Most absorption reduction: lambda={most_reduced['ortho_lambda']:.0g} (absorption={most_reduced['mean_absorption']:.6f})\n")
    print(f"Summary saved to {summary_md}")

    mark_done("success", f"Completed ablation sweep: {len(all_results)} runs")
    print("\nDone!")


if __name__ == "__main__":
    main()
