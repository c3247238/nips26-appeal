"""
Pilot H2: Mitigation Methods Comparison (Lightweight)
- Load pre-trained vanilla SAE (gpt2-small-res-jb, layer 8)
- Train lightweight TopK and JumpReLU variants (~50K tokens, 1 epoch)
- Evaluate absorption and reconstruction for each
- Compare vs vanilla baseline

Pilot: ~10 min target on 1 GPU (100 tokens sample, seed 42, 900s timeout)
"""

import json
import os
import sys
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, IterableDataset
from scipy.stats import skew

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GPU_ID = 6
# With CUDA_VISIBLE_DEVICES=6, only 1 GPU is visible so it maps to index 0 internally
DEVICE = "cuda:0"  # Always use 0 when CUDA_VISIBLE_DEVICES isolates a single GPU
SEED = 42
LAYER = 8
N_TOKENS_TRAIN = 50_000    # Pilot: lightweight training (~12 steps)
N_TOKENS_EVAL = 5_000      # Evaluation tokens
BATCH_SIZE = 64
SEQ_LEN = 128
N_EPOCHS = 1
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
TOPK_K = 50
JUMPRELU_THRESHOLD = 0.01
L1_COEFFICIENT = 8e-5
LEARNING_RATE = 1e-3
N_FEATURES = 200  # Top features for metric computation

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES isolates single GPU as index 0
    torch.cuda.manual_seed(SEED)


# ─── PID / Progress / Done tracking ─────────────────────────────────────────

def report_progress(task_id, results_dir, step=0, total_steps=0, loss=None, metric=None):
    progress = Path(results_dir) / f"pilot_h2_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_task_done(task_id, results_dir, status="success", summary=""):
    pid_file = Path(results_dir) / f"pilot_h2.pid"
    if pid_file.exists():
        pid_file.unlink()
    marker = Path(results_dir) / f"pilot_h2_DONE"
    marker.write_text(json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))


# ─── Dataset ─────────────────────────────────────────────────────────────────

def collect_text_examples(model, n_examples=50):
    """Pre-collect text examples from streaming dataset into memory."""
    from datasets import load_dataset
    print(f"  Collecting {n_examples} text examples...")
    t0 = datetime.now()
    texts = []
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    count = 0
    for ex in ds:
        texts.append(ex["text"])
        count += 1
        if count >= n_examples:
            break
    print(f"  Collected {len(texts)} examples in {(datetime.now() - t0).total_seconds():.1f}s")
    return texts


class TextDataset(IterableDataset):
    def __init__(self, model, text_examples, n_tokens, seq_len=SEQ_LEN, seed=SEED):
        self.model = model
        self.text_examples = text_examples
        self.n_tokens = n_tokens
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n_batches = self.n_tokens // (BATCH_SIZE * self.seq_len)
        for _ in range(n_batches):
            text = self.text_examples[self.rng.integers(0, len(self.text_examples))]
            try:
                tokens = self.model.to_tokens(text, truncate=True).flatten()
                n_chunks = len(tokens) // self.seq_len
                if n_chunks == 0:
                    continue
                i = self.rng.integers(0, n_chunks)
                chunk = tokens[i * self.seq_len:(i + 1) * self.seq_len]
                if len(chunk) == self.seq_len:
                    yield chunk
            except Exception:
                continue


# ─── SAE Variants ────────────────────────────────────────────────────────────

class StandardSAE(nn.Module):
    def __init__(self, d_in, d_sae, l1_coeff=L1_COEFFICIENT):
        super().__init__()
        self.W_enc = nn.Linear(d_in, d_sae, bias=False)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Linear(d_sae, d_in, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.act_fn = nn.ReLU()
        self.l1_coeff = l1_coeff
        self.d_in = d_in
        self.d_sae = d_sae

    def initialize(self, activations_sample):
        with torch.no_grad():
            flat = activations_sample.reshape(-1, self.d_in).float()
            flat = flat[torch.norm(flat, dim=1) > 1e-6]
            if flat.shape[0] >= self.d_sae:
                try:
                    from sklearn.decomposition import TruncatedSVD
                    svd = TruncatedSVD(n_components=self.d_sae, random_state=42)
                    W_dec_init = svd.fit_transform(flat.cpu().numpy())
                    W_dec_init = W_dec_init / (np.linalg.norm(W_dec_init, axis=1, keepdims=True) + 1e-8)
                    self.W_dec.weight.data = torch.tensor(W_dec_init, dtype=torch.float32, device=self.W_dec.weight.device)
                    self.W_enc.weight.data = self.W_dec.weight.data.T.clone()
                except Exception:
                    std = 1.0 / np.sqrt(self.d_in)
                    nn.init.uniform_(self.W_enc.weight.data, -std, std)
                    nn.init.uniform_(self.W_dec.weight.data, -std, std)

    def encode(self, x):
        if x.dim() == 3:
            batch_size, seq_len, d_in = x.shape
            x_flat = x.reshape(-1, d_in)
            h_flat = self.act_fn(self.W_enc(x_flat - self.b_dec) + self.b_enc)
            return h_flat.reshape(batch_size, seq_len, self.d_sae)
        return self.act_fn(self.W_enc(x - self.b_dec) + self.b_enc)

    def decode(self, h):
        if h.dim() == 3:
            batch_size, seq_len, d_sae = h.shape
            h_flat = h.reshape(-1, d_sae)
            recon_flat = self.W_dec(h_flat) + self.b_dec
            return recon_flat.reshape(batch_size, seq_len, self.d_in)
        return self.W_dec(h) + self.b_dec

    def forward(self, x):
        h = self.encode(x - self.b_dec)
        recon = self.decode(h)
        h_flat = h.reshape(-1, self.d_sae)
        recon_flat = recon.reshape(-1, self.d_in)
        x_flat = x.reshape(-1, self.d_in)
        l1_loss = h_flat.abs().sum() / h_flat.numel()
        mse_loss = ((x_flat - recon_flat) ** 2).mean()
        return mse_loss, l1_loss


class TopKSAE(StandardSAE):
    def __init__(self, d_in, d_sae, k=TOPK_K, l1_coeff=L1_COEFFICIENT):
        super().__init__(d_in, d_sae, l1_coeff)
        self.k = k

    def encode(self, x):
        if x.dim() == 3:
            batch_size, seq_len, d_in = x.shape
            x_flat = x.reshape(-1, d_in)
            h_flat = self.act_fn(self.W_enc(x_flat - self.b_dec) + self.b_enc)
            if h_flat.shape[1] > self.k:
                vals, idx = torch.topk(h_flat, k=self.k, dim=1)
                h_sparse = torch.zeros_like(h_flat)
                h_sparse.scatter_(1, idx, vals)
                h_sparse = h_sparse * (h_sparse > 0).float()
            else:
                h_sparse = h_flat
            return h_sparse.reshape(batch_size, seq_len, self.d_sae)
        h = self.act_fn(self.W_enc(x - self.b_dec) + self.b_enc)
        if h.shape[1] > self.k:
            vals, idx = torch.topk(h, k=self.k, dim=1)
            h_sparse = torch.zeros_like(h)
            h_sparse.scatter_(1, idx, vals)
            h_sparse = h_sparse * (h_sparse > 0).float()
            return h_sparse
        return h


class JumpReLUSAE(StandardSAE):
    def __init__(self, d_in, d_sae, threshold=JUMPRELU_THRESHOLD, l1_coeff=L1_COEFFICIENT):
        super().__init__(d_in, d_sae, l1_coeff)
        self.threshold = threshold

    def encode(self, x):
        if x.dim() == 3:
            batch_size, seq_len, d_in = x.shape
            x_flat = x.reshape(-1, d_in)
            h_flat = self.act_fn(self.W_enc(x_flat - self.b_dec) + self.b_enc)
            h_flat = h_flat * (h_flat > self.threshold).float()
            return h_flat.reshape(batch_size, seq_len, self.d_sae)
        h = self.act_fn(self.W_enc(x - self.b_dec) + self.b_enc)
        return h * (h > self.threshold).float()


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_vanilla_sae():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    print(f"  d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    print(f"Loading pre-trained vanilla SAE from {SAE_RELEASE}...")
    sae_id = f"blocks.{LAYER}.hook_resid_pre"
    vanilla_sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)
    print(f"  Vanilla SAE: d_sae={vanilla_sae.cfg.d_sae}")

    return model, vanilla_sae


def get_activations_batch(model, tokens_batch, layer):
    hook_name = f"blocks.{layer}.hook_resid_pre"
    _, cache = model.run_with_cache(
        tokens_batch,
        names_filter=[hook_name],
        return_type="loss",
    )
    return cache[hook_name]


# ─── Training ────────────────────────────────────────────────────────────────

def train_variant(variant_class, model, text_examples, variant_name, n_epochs=N_EPOCHS):
    print(f"\n{'='*60}")
    print(f"Training {variant_name}")
    print("=" * 60)

    d_in = model.cfg.d_model
    d_sae = 24576
    sae = variant_class(d_in, d_sae).to(DEVICE)
    print(f"  SAE: d_in={d_in}, d_sae={d_sae}, device={DEVICE}")

    # Initialize from activation sample
    print("  Collecting activation sample for initialization...")
    init_samples = []
    for text in text_examples[:3]:
        try:
            tokens = model.to_tokens(text, truncate=True)
            flat = tokens.flatten()
            for i in range(0, min(len(flat), 5 * SEQ_LEN), SEQ_LEN):
                chunk = flat[i:i + SEQ_LEN].unsqueeze(0)
                acts = get_activations_batch(model, chunk, LAYER)
                init_samples.append(acts.cpu())
        except Exception:
            continue
    if init_samples:
        init_acts = torch.cat(init_samples, dim=1).to(DEVICE)
        print(f"  Initializing with {init_acts.shape[1]} activation vectors...")
        sae.initialize(init_acts)
    del init_samples
    if init_acts is not None:
        del init_acts
    gc.collect()
    torch.cuda.empty_cache()

    # Training
    dataset = TextDataset(model, text_examples, N_TOKENS_TRAIN, SEQ_LEN, seed=SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)
    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)

    n_steps = N_TOKENS_TRAIN // (BATCH_SIZE * SEQ_LEN)
    print(f"  Training: {n_steps} steps, {n_epochs} epoch(s), batch={BATCH_SIZE}x{SEQ_LEN}")

    sae.train()
    for epoch in range(n_epochs):
        step = 0
        for batch_tokens in dataloader:
            if step >= n_steps:
                break
            tokens = batch_tokens.long().to(DEVICE)
            acts = get_activations_batch(model, tokens, LAYER)
            optimizer.zero_grad()
            mse_loss, l1_loss = sae(acts)
            loss = mse_loss + l1_loss * L1_COEFFICIENT
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            step += 1
            if step % 5 == 0:
                l0 = (sae.encode(acts - sae.b_dec) > 0).float().mean().item()
                print(f"  Step {step}/{n_steps}: loss={loss.item():.4f}, mse={mse_loss.item():.4f}, l0={l0:.4f}")
                report_progress("pilot_h2", str(RESULTS_DIR), step=step, total_steps=n_steps,
                              loss=float(loss.item()), metric={"mse": float(mse_loss.item()), "l0": l0})

    sae.eval()
    return sae


# ─── Evaluation ─────────────────────────────────────────────────────────────

def compute_gini_absorption(feature_acts, topk_indices):
    """Gini-based absorption: uniform firing = high absorption."""
    absorption_scores = {}
    for feat_idx in topk_indices:
        acts = np.abs(feature_acts[0, :, feat_idx].cpu().numpy())
        if np.sum(acts) == 0:
            absorption_scores[feat_idx] = 0.0
            continue
        sorted_acts = np.sort(acts)
        n = len(sorted_acts)
        gini = (2 * np.sum(np.arange(1, n + 1) * sorted_acts)) / (n * np.sum(sorted_acts)) - (n + 1) / n
        absorption_scores[feat_idx] = 1.0 - max(0.0, min(1.0, gini))
    return absorption_scores


def compute_uas(sae, feature_acts, topk_indices):
    if hasattr(sae.W_dec, 'weight'):
        W_dec = sae.W_dec.weight.data.cpu().numpy().T
    else:
        W_dec = sae.W_dec.data.cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)
    alpha, beta = 1.0, 0.5
    uas_scores = {}
    for feat_idx in topk_indices:
        other_indices = [i for i in topk_indices if i != feat_idx]
        if len(other_indices) < 2:
            uas_scores[feat_idx] = 0.0
            continue
        cos_sims = np.dot(W_dec_norm[feat_idx], W_dec_norm[other_indices].T)
        cos_sim_var = float(np.var(cos_sims))
        feat_act = feature_acts[0, :, feat_idx].cpu().numpy()
        non_zero = feat_act[feat_act > 0]
        freq_skew = float(abs(skew(non_zero))) if len(non_zero) > 2 else 0.0
        uas_scores[feat_idx] = alpha * cos_sim_var + beta * freq_skew
    return uas_scores


def evaluate_sae(sae, model, text_examples, variant_name, n_tokens=N_TOKENS_EVAL):
    print(f"\n  Evaluating {variant_name}...")

    all_acts = []
    all_feature_acts = []
    all_recons = []

    rng = np.random.default_rng(SEED + 1)
    total = 0
    max_attempts = n_tokens * 2 // SEQ_LEN + 100
    attempts = 0
    while total < n_tokens and attempts < max_attempts:
        attempts += 1
        text = text_examples[rng.integers(0, len(text_examples))]
        try:
            tokens = model.to_tokens(text, truncate=True).flatten()
            n_chunks = len(tokens) // SEQ_LEN
            if n_chunks == 0:
                continue
            i = rng.integers(0, n_chunks)
            chunk = tokens[i * SEQ_LEN:(i + 1) * SEQ_LEN]
            if len(chunk) != SEQ_LEN:
                continue
            chunk = chunk.unsqueeze(0)
            acts = get_activations_batch(model, chunk, LAYER)
            with torch.no_grad():
                fa = sae.encode(acts - sae.b_dec)
                rec = sae.decode(fa)
            all_acts.append(acts.float())
            all_feature_acts.append(fa.float())
            all_recons.append(rec.float())
            total += SEQ_LEN
        except Exception:
            continue

    if not all_acts:
        print(f"    ERROR: No activations collected!")
        return {"variant": variant_name, "error": "No activations collected"}

    eval_acts = torch.cat(all_acts, dim=1).float()
    eval_fa = torch.cat(all_feature_acts, dim=1)
    eval_rec = torch.cat(all_recons, dim=1).float()

    recon_mse = torch.nn.functional.mse_loss(eval_acts.to(DEVICE), eval_rec.to(DEVICE)).item()
    l0 = (eval_fa > 0).float().mean().item()

    # Top features by total activation
    total_act = eval_fa[0].sum(dim=0).cpu().numpy()
    topk_indices = np.argsort(total_act)[-N_FEATURES:][::-1]

    absorption_scores = compute_gini_absorption(eval_fa, topk_indices)
    uas_scores = compute_uas(sae, eval_fa, topk_indices)

    mean_absorption = float(np.mean([absorption_scores.get(i, 0.0) for i in topk_indices]))
    mean_uas = float(np.mean([uas_scores.get(i, 0.0) for i in topk_indices]))

    print(f"    L0 sparsity: {l0:.4f}")
    print(f"    Reconstruction MSE: {recon_mse:.6f}")
    print(f"    Mean absorption: {mean_absorption:.4f}")
    print(f"    Mean UAS: {mean_uas:.6f}")

    return {
        "variant": variant_name,
        "l0_sparsity": l0,
        "reconstruction_mse": recon_mse,
        "mean_absorption": mean_absorption,
        "mean_uas": mean_uas,
        "n_features_analyzed": N_FEATURES,
        "n_eval_tokens": eval_acts.shape[1],
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    task_id = "pilot_h2"
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"\nPID: {os.getpid()}")

    print("\n" + "=" * 60)
    print("PILOT H2: Mitigation Methods Comparison (Lightweight)")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}, GPU: {GPU_ID}")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"SAE Release: {SAE_RELEASE}")
    print(f"Train tokens: {N_TOKENS_TRAIN:,}, Eval tokens: {N_TOKENS_EVAL:,}")
    report_progress(task_id, str(RESULTS_DIR), step=0, total_steps=1)

    # Load model and vanilla SAE
    model, vanilla_sae = load_model_and_vanilla_sae()
    d_sae = vanilla_sae.cfg.d_sae

    # Collect text examples
    print("\nCollecting text examples...")
    text_examples = collect_text_examples(model, n_examples=50)

    # Evaluate pre-trained vanilla SAE
    print("\nEvaluating pre-trained vanilla SAE...")
    vanilla_result = evaluate_sae(vanilla_sae, model, text_examples, "Vanilla (pre-trained)")
    results = [vanilla_result]

    # Train and evaluate variants
    variants = [
        (TopKSAE, "TopK SAE"),
        (JumpReLUSAE, "JumpReLU SAE"),
    ]

    for variant_class, variant_name in variants:
        try:
            sae = train_variant(variant_class, model, text_examples, variant_name)
            result = evaluate_sae(sae, model, text_examples, variant_name)
            results.append(result)
            del sae
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({"variant": variant_name, "error": str(e)})
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Pilot H2 Mitigation Results")
    print("=" * 60)
    valid_results = [r for r in results if "error" not in r]
    print(f"\n{'Variant':<22} {'L0':>8} {'Recon MSE':>12} {'Absorption':>12} {'UAS':>10}")
    print("-" * 66)
    for r in valid_results:
        print(f"{r['variant']:<22} {r['l0_sparsity']:>8.4f} {r['reconstruction_mse']:>12.6f} "
              f"{r['mean_absorption']:>12.4f} {r['mean_uas']:>10.4f}")

    # Pass criteria check
    vanilla_mse = vanilla_result["reconstruction_mse"]
    trained_results = [r for r in valid_results if r["variant"] != "Vanilla (pre-trained)"]
    methods_passing_mse = sum(1 for r in trained_results
                              if abs(r["reconstruction_mse"] - vanilla_mse) / vanilla_mse < 0.30)

    print(f"\nPass Criteria Check:")
    print(f"  At least 2 methods with <30% CE loss degradation: {methods_passing_mse}/{len(trained_results)}")
    print(f"  Pilot_h2 criterion: At least 2 methods show promise (low MSE, reasonable absorption)")
    pilot_pass = len(trained_results) >= 2 and all("error" not in r for r in trained_results)
    print(f"  Overall pilot pass: {pilot_pass}")

    # Save results
    output = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "sae_release": SAE_RELEASE,
            "d_sae": d_sae,
            "n_train_tokens": N_TOKENS_TRAIN,
            "n_eval_tokens": N_TOKENS_EVAL,
            "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE,
            "seq_len": SEQ_LEN,
            "learning_rate": LEARNING_RATE,
            "l1_coefficient": L1_COEFFICIENT,
            "topk_k": TOPK_K,
            "jumprelu_threshold": JUMPRELU_THRESHOLD,
            "device": DEVICE,
        },
        "results": results,
        "pilot_assessment": {
            "vanilla_baseline": {
                "l0": vanilla_result["l0_sparsity"],
                "recon_mse": vanilla_result["reconstruction_mse"],
                "mean_absorption": vanilla_result["mean_absorption"],
                "mean_uas": vanilla_result["mean_uas"],
            },
            "pilot_pass": pilot_pass,
            "methods_evaluated": len(trained_results),
            "methods_with_acceptable_mse": methods_passing_mse,
        },
    }

    output_path = RESULTS_DIR / "h2_pilot.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    summary_str = f"variants={len(valid_results)}, vanilla_mse={vanilla_mse:.4f}, pilot_pass={pilot_pass}"
    mark_task_done(task_id, str(RESULTS_DIR), status="success", summary=summary_str)
    print(f"\nPilot H2 completed: {task_id}")
    return output


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback; traceback.print_exc()
        mark_task_done("pilot_h2", str(RESULTS_DIR), status="failed", summary=str(e))
        sys.exit(1)
