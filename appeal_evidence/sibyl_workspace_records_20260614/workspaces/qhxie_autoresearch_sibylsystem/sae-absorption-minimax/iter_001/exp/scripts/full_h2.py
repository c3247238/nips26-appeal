"""
Full H2: Mitigation Benchmark - Compare SAE variants on absorption reduction.
- Vanilla SAE (pre-trained from gpt2-small-res-jb)
- TopK SAE (trained)
- JumpReLU SAE (trained)
- OrtSAE (trained with orthogonal regularization)
- ATM (adaptive temporal masking)

Due to Gemma-2B being gated, this experiment uses GPT-2 Small at layer 8.
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
from scipy.stats import spearmanr, skew

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GPU_ID = 5
DEVICE = f"cuda:{GPU_ID}" if torch.cuda.is_available() else "cpu"
SEED = 42
LAYER = 8
N_TOKENS_TRAIN = 3_000_000  # Training tokens per variant (~366 steps/epoch, ~22 min/variant)
N_TOKENS_EVAL = 50_000    # Evaluation tokens
BATCH_SIZE = 64
SEQ_LEN = 128
LEARNING_RATE = 1e-4  # Lower LR for stability with 50M-param SAE
L1_COEFFICIENT = 8e-5
ORTHO_LAMBDA = 1e-3
JUMPRELU_THRESHOLD = 0.01
TOPK_K = 50
N_EPOCHS = 3
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(GPU_ID)
    torch.cuda.manual_seed(SEED)


# ─── PID / Progress / Done tracking ─────────────────────────────────────────

def report_progress(task_id, results_dir, epoch=1, total_epochs=1, step=0,
                    total_steps=0, loss=None, metric=None):
    import json as _json
    progress = Path(results_dir) / f"{task_id}_PROGRESS.json"
    progress.write_text(_json.dumps({
        "task_id": task_id,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_task_done(task_id, results_dir, status="success", summary=""):
    import json as _json
    pid_file = Path(results_dir) / f"{task_id}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = Path(results_dir) / f"{task_id}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = _json.loads(progress_file.read_text())
        except (_json.JSONDecodeError, ValueError):
            pass
    marker = Path(results_dir) / f"{task_id}_DONE"
    marker.write_text(_json.dumps({
        "task_id": task_id,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ─── Dataset ─────────────────────────────────────────────────────────────────

def collect_text_examples(model, n_examples=200):
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
        if count % 50 == 0:
            print(f"    {count}/{n_examples} examples...")
    print(f"  Collected {len(texts)} examples in {(datetime.now() - t0).total_seconds():.1f}s")
    return texts


class TextDataset(IterableDataset):
    """Yields text batches from pre-loaded text examples."""
    def __init__(self, model, text_examples, n_tokens, seq_len=SEQ_LEN, seed=SEED):
        self.model = model
        self.text_examples = text_examples
        self.n_tokens = n_tokens
        self.seq_len = seq_len
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        n_batches = self.n_tokens // (BATCH_SIZE * self.seq_len)
        for _ in range(n_batches):
            # Randomly sample a text example
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
        """Initialize encoder/decoder from activation sample."""
        with torch.no_grad():
            # Use k-means-like initialization: W_dec = first principal components of activations
            flat = activations_sample.reshape(-1, self.d_in).float()
            flat = flat[torch.norm(flat, dim=1) > 1e-6]
            if flat.shape[0] > self.d_sae:
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
        # x: [batch, d_in] or [batch, seq, d_in]
        if x.dim() == 3:
            batch_size, seq_len, d_in = x.shape
            x_flat = x.reshape(-1, d_in)
            h_flat = self.act_fn(self.W_enc(x_flat - self.b_dec) + self.b_enc)
            return h_flat.reshape(batch_size, seq_len, self.d_sae)
        return self.act_fn(self.W_enc(x - self.b_dec) + self.b_enc)

    def decode(self, h):
        # h: [batch, d_sae] or [batch, seq, d_sae]
        if h.dim() == 3:
            batch_size, seq_len, d_sae = h.shape
            h_flat = h.reshape(-1, d_sae)
            # DEBUG
            assert h_flat.shape[-1] == self.d_sae, f"decode h_flat dim mismatch: {h_flat.shape} vs d_sae={self.d_sae}, d_in={self.d_in}"
            assert self.W_dec.out_features == self.d_in, f"W_dec out_features mismatch: {self.W_dec.out_features} vs d_in={self.d_in}"
            recon_flat = self.W_dec(h_flat) + self.b_dec
            return recon_flat.reshape(batch_size, seq_len, self.d_in)
        return self.W_dec(h) + self.b_dec

    def forward(self, x):
        h = self.encode(x - self.b_dec)
        recon = self.decode(h)
        # Flatten for loss computation
        h_flat = h.reshape(-1, self.d_sae)
        recon_flat = recon.reshape(-1, self.d_in)
        x_flat = x.reshape(-1, self.d_in)
        l1_loss = h_flat.abs().sum() / h_flat.numel()
        mse_loss = ((x_flat - recon_flat) ** 2).mean()
        return mse_loss, l1_loss

    def compute_loss(self, x, recon, l1_loss):
        # Override: handles recon separately
        pass


class TopKSAE(StandardSAE):
    def __init__(self, d_in, d_sae, k=TOPK_K, l1_coeff=L1_COEFFICIENT):
        super().__init__(d_in, d_sae, l1_coeff)
        self.k = k

    def encode(self, x):
        if x.dim() == 3:
            batch_size, seq_len, d_in = x.shape
            x_flat = x.reshape(-1, d_in)
            h_flat = self.act_fn(self.W_enc(x_flat - self.b_dec) + self.b_enc)
            # TopK
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

    def forward(self, x):
        h = self.encode(x - self.b_dec)
        recon = self.decode(h)
        h_flat = h.reshape(-1, self.d_sae)
        recon_flat = recon.reshape(-1, self.d_in)
        x_flat = x.reshape(-1, self.d_in)
        l1_loss = h_flat.abs().sum() / h_flat.numel()
        mse_loss = ((x_flat - recon_flat) ** 2).mean()
        return mse_loss, l1_loss


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

    def forward(self, x):
        h = self.encode(x - self.b_dec)
        recon = self.decode(h)
        h_flat = h.reshape(-1, self.d_sae)
        recon_flat = recon.reshape(-1, self.d_in)
        x_flat = x.reshape(-1, self.d_in)
        l1_loss = h_flat.abs().sum() / h_flat.numel()
        mse_loss = ((x_flat - recon_flat) ** 2).mean()
        return mse_loss, l1_loss


class OrtSAE(StandardSAE):
    def __init__(self, d_in, d_sae, ortho_lambda=ORTHO_LAMBDA, l1_coeff=L1_COEFFICIENT):
        super().__init__(d_in, d_sae, l1_coeff)
        self.ortho_lambda = ortho_lambda

    def ortho_loss(self):
        # W_dec.weight: [d_in, d_sae] (Linear convention: out_features, in_features)
        # Decoder direction for feature j = column j = W[:, j]
        # Transpose to get [d_sae, d_in] where each row = one feature's decoder direction
        W = self.W_dec.weight.T  # [d_sae, d_in]
        W_norm = W / W.norm(dim=1, keepdim=True).clamp(min=1e-8)
        # cos_sim between all pairs of decoder directions
        cos_sim = W_norm @ W_norm.T  # [d_sae, d_sae]
        mask = ~torch.eye(self.d_sae, dtype=torch.bool, device=cos_sim.device)
        if mask.sum() > 0:
            return cos_sim[mask].pow(2).mean()
        return torch.tensor(0.0, device=W.device)

    def forward(self, x):
        h = self.encode(x - self.b_dec)
        recon = self.decode(h)
        h_flat = h.reshape(-1, self.d_sae)
        recon_flat = recon.reshape(-1, self.d_in)
        x_flat = x.reshape(-1, self.d_in)
        l1_loss = h_flat.abs().sum() / h_flat.numel()
        mse_loss = ((x_flat - recon_flat) ** 2).mean()
        ortho = self.ortho_loss()
        return mse_loss + l1_loss * self.l1_coeff + ortho * self.ortho_lambda, l1_loss


class ATMSAE(StandardSAE):
    def __init__(self, d_in, d_sae, l1_coeff=L1_COEFFICIENT, importance_warmup=200):
        super().__init__(d_in, d_sae, l1_coeff)
        self.importance_weight = nn.Parameter(torch.ones(d_sae) * 0.1)
        self.importance_warmup = importance_warmup
        self.step_count = 0
        self.running_act_sum = torch.zeros(d_sae, device=DEVICE)

    def encode(self, x):
        if x.dim() == 3:
            batch_size, seq_len, d_in = x.shape
            x_flat = x.reshape(-1, d_in)
            h_flat = self.act_fn(self.W_enc(x_flat - self.b_dec) + self.b_enc)
            # ATM masking
            self.step_count += 1
            act_freq = (h_flat > 0).float().sum(dim=0)
            self.running_act_sum = 0.99 * self.running_act_sum + 0.01 * act_freq
            if self.step_count > self.importance_warmup:
                importance = self.importance_weight.sigmoid()
                mask = importance.unsqueeze(0).clamp(0.3, 1.0)
                h_flat = h_flat * mask
            return h_flat.reshape(batch_size, seq_len, self.d_sae)
        h = self.act_fn(self.W_enc(x - self.b_dec) + self.b_enc)
        self.step_count += 1
        act_freq = (h > 0).float().sum(dim=0)
        self.running_act_sum = 0.99 * self.running_act_sum + 0.01 * act_freq
        if self.step_count > self.importance_warmup:
            importance = self.importance_weight.sigmoid()
            h = h * importance.clamp(0.3, 1.0).unsqueeze(0)
        return h

    def forward(self, x):
        h = self.encode(x - self.b_dec)
        recon = self.decode(h)
        h_flat = h.reshape(-1, self.d_sae)
        recon_flat = recon.reshape(-1, self.d_in)
        x_flat = x.reshape(-1, self.d_in)
        l1_loss = h_flat.abs().sum() / h_flat.numel()
        mse_loss = ((x_flat - recon_flat) ** 2).mean()
        return mse_loss, l1_loss


# ─── Model Loading ───────────────────────────────────────────────────────────

def load_model_and_vanilla_sae():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    compute_device = f"cuda:{GPU_ID}"
    print(f"Loading {MODEL_NAME} on {compute_device}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=compute_device)
    print(f"  d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    print(f"Loading pre-trained vanilla SAE from {SAE_RELEASE} on {compute_device}...")
    sae_id = f"blocks.{LAYER}.hook_resid_pre"
    vanilla_sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=compute_device)
    print(f"  Vanilla SAE: d_sae={vanilla_sae.cfg.d_sae}, architecture={vanilla_sae.cfg.architecture()}")

    return model, vanilla_sae


def get_activations_batch(model, tokens_batch, layer):
    """Get residual stream activations for a batch of tokens."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    _, cache = model.run_with_cache(
        tokens_batch,
        names_filter=[hook_name],
        return_type="loss",
    )
    return cache[hook_name]  # [batch, seq, d_model]


# ─── Training ────────────────────────────────────────────────────────────────

def train_variant(variant_class, model, text_examples, variant_name, n_epochs=N_EPOCHS,
                  n_train_tokens=N_TOKENS_TRAIN, ortho_lambda=ORTHO_LAMBDA, k=TOPK_K,
                  layer=LAYER):
    """Train an SAE variant by capturing activations on-the-fly."""
    print(f"\n{'='*60}")
    print(f"Training {variant_name}")
    print("=" * 60)

    d_in = model.cfg.d_model
    # Build kwargs
    kwargs = {"l1_coeff": L1_COEFFICIENT}
    if variant_class is TopKSAE:
        kwargs["k"] = k
    elif variant_class is JumpReLUSAE:
        kwargs["threshold"] = JUMPRELU_THRESHOLD
    elif variant_class is OrtSAE:
        kwargs["ortho_lambda"] = ortho_lambda

    sae = variant_class(d_in, 24576, **kwargs).to(DEVICE)
    print(f"  SAE: d_in={d_in}, d_sae=24576, device={DEVICE}")

    # Collect activation sample for initialization (use first few examples)
    print("  Collecting activation sample for initialization...")
    init_samples = []
    init_acts = None
    for text in text_examples[:5]:
        try:
            tokens = model.to_tokens(text, truncate=True)
            flat = tokens.flatten()
            for i in range(0, min(len(flat), 10 * SEQ_LEN), SEQ_LEN):
                chunk = flat[i:i + SEQ_LEN].unsqueeze(0)
                acts = get_activations_batch(model, chunk, layer)
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Training
    dataset = TextDataset(model, text_examples, n_train_tokens, SEQ_LEN, seed=SEED)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    n_steps = n_train_tokens // (BATCH_SIZE * SEQ_LEN)

    print(f"  Training: {n_steps} steps, {n_epochs} epochs, batch_size={BATCH_SIZE}x{SEQ_LEN}")
    report_progress("full_h2", str(RESULTS_DIR), epoch=0, total_epochs=n_epochs)

    sae.train()
    for epoch in range(n_epochs):
        epoch_loss, epoch_mse, epoch_l1 = 0.0, 0.0, 0.0
        steps = 0
        for batch_tokens in dataloader:
            if steps >= n_steps:
                break
            tokens = batch_tokens.long().to(DEVICE)
            # Get residual stream activations using hook
            acts = get_activations_batch(model, tokens, layer)  # [batch, seq, d_in]
            # Train SAE
            optimizer.zero_grad()
            mse_loss, l1_loss = sae(acts)
            if variant_class is StandardSAE:
                loss = mse_loss + l1_loss * L1_COEFFICIENT
            else:
                loss = mse_loss + l1_loss * L1_COEFFICIENT
            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_mse += mse_loss.item()
            epoch_l1 += l1_loss.item()
            steps += 1
            if steps % 200 == 0:
                l0 = (sae.encode(acts - sae.b_dec) > 0).float().mean().item()
                print(f"  E{epoch+1} S{steps}: loss={loss.item():.4f}, mse={mse_loss.item():.4f}, l0={l0:.4f}")
                report_progress("full_h2", str(RESULTS_DIR), epoch=epoch+1, total_epochs=n_epochs,
                              step=steps, total_steps=n_steps, loss=float(loss.item()),
                              metric={"mse": float(mse_loss.item()), "l0": l0})
        print(f"  Epoch {epoch+1} done: avg_loss={epoch_loss/max(steps,1):.4f}, "
              f"avg_mse={epoch_mse/max(steps,1):.4f}, avg_l1={epoch_l1/max(steps,1):.4f}")

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
    # SAELens SAE: W_dec is a Parameter [d_sae, d_in]
    # Custom SAE: W_dec is nn.Linear(d_sae, d_in) with .weight [d_in, d_sae]
    if hasattr(sae.W_dec, 'weight'):
        # Linear: weight is [d_in, d_sae], transpose to [d_sae, d_in]
        W_dec = sae.W_dec.weight.data.cpu().numpy().T
    else:
        W_dec = sae.W_dec.data.cpu().numpy()  # [d_sae, d_in]
    # Normalize each decoder direction (each row = one feature's decoder vector)
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


def compute_dead_feature_ratio(feature_acts):
    n_features = feature_acts.shape[-1]
    n_dead = (feature_acts.sum(dim=(0, 1)) == 0).sum().item()
    return n_dead / n_features


def evaluate_sae(sae, model, text_examples, variant_name, n_tokens=N_TOKENS_EVAL, n_features=200):
    """Evaluate SAE variant on absorption metrics using pre-cached text examples."""
    print(f"\n  Evaluating {variant_name}...")

    all_acts = []
    all_recons = []
    all_feature_acts = []

    # Sample from pre-cached text examples — no streaming, no network
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
                # SAELens SAE handles b_dec internally; custom SAE expects centered input
                if hasattr(sae, 'process_sae_in'):
                    fa = sae.encode(acts)
                    rec = sae.decode(fa)
                else:
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

    eval_acts = torch.cat(all_acts, dim=1).float()  # [1, n_tokens, d_in]
    eval_fa = torch.cat(all_feature_acts, dim=1)      # [1, n_tokens, d_sae]
    eval_rec = torch.cat(all_recons, dim=1)           # [1, n_tokens, d_in]

    recon_mse = torch.nn.functional.mse_loss(eval_acts.to(DEVICE), eval_rec.to(DEVICE)).item()
    l0 = (eval_fa > 0).float().mean().item()
    dead_ratio = compute_dead_feature_ratio(eval_fa)

    # Select top features by total activation
    total_act = eval_fa[0].sum(dim=0).cpu().numpy()
    topk_indices = np.argsort(total_act)[-n_features:][::-1]

    absorption_scores = compute_gini_absorption(eval_fa, topk_indices)
    uas_scores = compute_uas(sae, eval_fa, topk_indices)

    mean_absorption = float(np.mean([absorption_scores.get(i, 0.0) for i in topk_indices]))
    mean_uas = float(np.mean([uas_scores.get(i, 0.0) for i in topk_indices]))

    print(f"    L0 sparsity: {l0:.4f}")
    print(f"    Reconstruction MSE: {recon_mse:.6f}")
    print(f"    Dead feature ratio: {dead_ratio:.4f}")
    print(f"    Mean absorption: {mean_absorption:.4f}")
    print(f"    Mean UAS: {mean_uas:.6f}")

    return {
        "variant": variant_name,
        "l0_sparsity": l0,
        "reconstruction_mse": recon_mse,
        "dead_feature_ratio": dead_ratio,
        "mean_absorption": mean_absorption,
        "mean_uas": mean_uas,
        "n_features_analyzed": n_features,
        "n_eval_tokens": eval_acts.shape[1],
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    task_id = "full_h2"
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"\nPID: {os.getpid()}")

    print("\n" + "=" * 60)
    print("FULL H2: SAE Mitigation Benchmark")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}, GPU: {GPU_ID}")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    print(f"SAE Release: {SAE_RELEASE}")
    print(f"Training tokens: {N_TOKENS_TRAIN:,}, Eval tokens: {N_TOKENS_EVAL:,}")
    report_progress(task_id, str(RESULTS_DIR), epoch=0, total_epochs=N_EPOCHS)

    # Load model and vanilla SAE
    model, vanilla_sae = load_model_and_vanilla_sae()
    d_sae = vanilla_sae.cfg.d_sae

    # Collect text examples ONCE (avoids streaming shuffle blocking)
    print("\nCollecting text examples for training and evaluation...")
    text_examples = collect_text_examples(model, n_examples=200)

    # Evaluate pre-trained vanilla SAE
    vanilla_result = evaluate_sae(vanilla_sae, model, text_examples, "Vanilla (pre-trained)")
    results = [vanilla_result]

    # Train and evaluate variants
    variants = [
        (TopKSAE, "TopK SAE"),
        (JumpReLUSAE, "JumpReLU SAE"),
        (OrtSAE, "OrtSAE"),
        (ATMSAE, "ATM SAE"),
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
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY: Mitigation Benchmark Results")
    print("=" * 60)
    valid_results = [r for r in results if "error" not in r]
    print(f"\n{'Variant':<22} {'L0':>8} {'Recon MSE':>12} {'Dead%':>8} {'Absorption':>12}")
    print("-" * 65)
    for r in valid_results:
        dead_pct = r.get("dead_feature_ratio", 0) * 100
        print(f"{r['variant']:<22} {r['l0_sparsity']:>8.4f} {r['reconstruction_mse']:>12.6f} "
              f"{dead_pct:>7.1f}% {r['mean_absorption']:>12.4f}")

    # Absorption comparison: trained variants vs TopK baseline
    trained_variants = [r for r in valid_results if r["variant"] != "Vanilla (pre-trained)"]
    topk_result = next((r for r in trained_variants if "TopK" in r["variant"]), None)
    vanilla_absorption = vanilla_result["mean_absorption"]

    print(f"\nAbsorption Comparison (trained vs TopK SAE baseline):")
    print(f"{'Variant':<22} {'Absorption':>12} {'vs TopK':>12} {'Dead%':>8} {'Recon MSE':>12}")
    print("-" * 68)
    best_variant = None
    best_absorption = -1
    for r in trained_variants:
        if topk_result:
            delta = (r["mean_absorption"] - topk_result["mean_absorption"]) / max(topk_result["mean_absorption"], 1e-6) * 100
            delta_str = f"{delta:>+11.1f}%"
        else:
            delta_str = "N/A"
        dead_pct = r.get("dead_feature_ratio", 0) * 100
        print(f"{r['variant']:<22} {r['mean_absorption']:>12.4f} {delta_str:>12} {dead_pct:>7.1f}% {r['reconstruction_mse']:>12.4f}")
        if r["mean_absorption"] > best_absorption:
            best_absorption = r["mean_absorption"]
            best_variant = r["variant"]

    # H2 pass: at least one variant achieves absorption > 0.5 with dead_ratio < 5% and MSE < 50
    h2_candidates = [r for r in trained_variants
                     if r["mean_absorption"] > 0.5
                     and r.get("dead_feature_ratio", 1) < 0.05
                     and r["reconstruction_mse"] < 50]
    h2_pass = len(h2_candidates) > 0
    print(f"\nH2 Assessment:")
    print(f"  Baseline (Vanilla) absorption: {vanilla_absorption:.4f}")
    print(f"  Best trained: {best_variant} (absorption={best_absorption:.4f})")
    print(f"  H2 criteria (>0.5 absorption, <5% dead, MSE<50): {'PASS' if h2_pass else 'FAIL'}")
    if h2_candidates:
        for c in h2_candidates:
            print(f"    {c['variant']}: absorption={c['mean_absorption']:.4f}, dead={c.get('dead_feature_ratio',0)*100:.1f}%, MSE={c['reconstruction_mse']:.4f}")

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
            "ortho_lambda": ORTHO_LAMBDA,
            "topk_k": TOPK_K,
            "device": DEVICE,
        },
        "results": results,
        "h2_assessment": {
            "baseline_absorption": vanilla_absorption,
            "best_variant": best_variant,
            "best_variant": best_variant,
            "best_absorption": best_absorption,
            "h2_pass": h2_pass,
            "h2_pass_candidates": [{"variant": c["variant"], "absorption": c["mean_absorption"],
                                    "dead_ratio": c.get("dead_feature_ratio", 0),
                                    "mse": c["reconstruction_mse"]} for c in h2_candidates],
        },
    }

    output_path = RESULTS_DIR / "h2_mitigation_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    summary_str = f"variants={len(valid_results)}, best={best_variant}, best_abs={best_absorption:.4f}, h2={'PASS' if h2_pass else 'FAIL'}"
    mark_task_done(task_id, str(RESULTS_DIR), status="success", summary=summary_str)
    print(f"\nTask completed: {task_id}")
    return output


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback; traceback.print_exc()
        mark_task_done("full_h2", str(RESULTS_DIR), status="failed", summary=str(e))
        sys.exit(1)
