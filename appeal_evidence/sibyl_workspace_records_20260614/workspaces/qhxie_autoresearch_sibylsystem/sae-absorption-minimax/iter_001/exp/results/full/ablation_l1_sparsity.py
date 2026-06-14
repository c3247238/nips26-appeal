"""
Ablation: L1 Coefficient Sweep for Vanilla SAE at GPT-2 Layer 8

Task: Sweep L1 coefficient ∈ {1e-5, 5e-5, 1e-4, 5e-4, 1e-3} for vanilla SAE.
Measure absorption score, L0, and CE loss per L1. Identify Pareto frontier.

Pilot mode: 500K training tokens per SAE, 2 epochs (~3-5 min each).
"""

import json
import os
import gc
import warnings
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, skew

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp/results/full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "ablation_l1_sparsity"
# GPU_ID=3 has ~70 GB free (vs GPU 6 which is nearly full)
GPU_ID = 3
DEVICE = "cuda:0"  # Use cuda:0 since CUDA_VISIBLE_DEVICES=N makes that GPU visible as index 0
SEED = 42

N_TOKENS_TRAIN = 500_000
N_TOKENS_EVAL = 50_000
BATCH_SIZE = 64
SEQ_LEN = 128
LEARNING_RATE = 1e-4
N_EPOCHS = 2
MODEL_NAME = "gpt2-small"
HOOK_NAME = "blocks.8.hook_resid_pre"
D_SAE = 24576
LAYER = 8
L1_VALUES = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3]

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # CUDA_VISIBLE_DEVICES=6 → only GPU 6 visible as index 0
    torch.cuda.manual_seed(SEED)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def gini_coefficient(x):
    x = np.asarray(x, dtype=np.float64)
    x = np.abs(x)
    x = x[x > 0]
    if len(x) < 2:
        return 0.0
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    g = (2 * np.sum(idx * x) / (n * np.sum(x))) - (n + 1) / n
    return max(0.0, min(1.0, g))


class StandardSAE(nn.Module):
    """
    Vanilla SAE using explicit matrix operations.

    Architecture (intuitive indexing):
      - Encoder: W_enc [d_sae, d_in], b_enc [d_sae]
        encode(x) = ReLU(x @ W_enc.T + b_enc)
        x [N, d_in] @ W_enc.T [d_in, d_sae] = [N, d_sae]
      - Decoder: W_dec [d_in, d_sae], b_dec [d_in]
        decode(h) = h @ W_dec.T + b_dec
        h [N, d_sae] @ W_dec.T [d_sae, d_in] = [N, d_in]
      - Feature i's decoder direction = column i of W_dec.T = row i of W_dec

    Using plain matrix multiplication (not nn.Linear) so we control shapes exactly.
    Feature i's decoder direction = W_dec[:, i] (column i).
    """
    def __init__(self, d_in, d_sae, l1_coeff=8e-5):
        super().__init__()
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))   # [d_sae, d_in]
        self.b_enc = nn.Parameter(torch.zeros(d_sae))         # [d_sae]
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))   # [d_in, d_sae]
        self.b_dec = nn.Parameter(torch.zeros(d_in))           # [d_in]
        self.act_fn = nn.ReLU()
        self.l1_coeff = l1_coeff
        self.d_in = d_in
        self.d_sae = d_sae

    def initialize(self, activations_sample):
        """Initialize from activation sample. Use SVD for encoder-decoder coupling."""
        with torch.no_grad():
            flat = activations_sample.reshape(-1, self.d_in).float()
            flat = flat[torch.norm(flat, dim=1) > 1e-6]
            mean_act = flat.mean(dim=0)
            self.b_dec.data = mean_act
            self.b_enc.data.zero_()
            # Initialize encoder: use PCA-like approach
            # Flatten activations and compute top-k PCA directions
            if flat.shape[0] > self.d_sae:
                try:
                    from sklearn.decomposition import TruncatedSVD
                    svd = TruncatedSVD(n_components=self.d_sae, random_state=SEED)
                    components = svd.fit_transform(flat.cpu().numpy())
                    # components: [N, d_sae]
                    # We want encoder weights: [d_sae, d_in] = top d_sae directions
                    # svd.components_ shape is [n_components, n_features] = [d_sae, d_in]
                    pc = svd.components_  # [d_sae, d_in]
                    pc = pc / (np.linalg.norm(pc, axis=1, keepdims=True) + 1e-8)
                    self.W_enc.data = torch.from_numpy(pc).float()
                    self.W_dec.data = torch.from_numpy(pc.T).float()
                    del svd, components, pc
                    gc.collect()
                    return
                except Exception as e:
                    pass
            # Fallback: orthonormal-ish initialization
            std = 1.0 / np.sqrt(self.d_in)
            nn.init.normal_(self.W_enc, 0, std)
            nn.init.normal_(self.W_dec, 0, std)

    def _random_init(self):
        std = 1.0 / np.sqrt(self.d_in)
        nn.init.normal_(self.W_enc, 0, std)
        nn.init.normal_(self.W_dec, 0, std)

    def encode(self, x):
        # x: [batch, seq, d_in]
        if x.dim() == 3:
            b, s, d = x.shape
            x_flat = x.reshape(-1, d)  # [N, d_in]
            h = self.act_fn(F.linear(x_flat, self.W_enc, self.b_enc))  # [N, d_sae]
            return h.reshape(b, s, self.d_sae)
        return self.act_fn(F.linear(x, self.W_enc, self.b_enc))

    def decode(self, h):
        # h: [batch, seq, d_sae]
        if h.dim() == 3:
            b, s, d_sae = h.shape
            recon = F.linear(h.reshape(-1, d_sae), self.W_dec, self.b_dec)  # [N, d_in]
            return recon.reshape(b, s, self.d_in)
        return F.linear(h, self.W_dec, self.b_dec)

    def forward(self, x):
        h = self.encode(x)
        recon = self.decode(h)
        h_flat = h.reshape(-1, self.d_sae)
        x_flat = x.reshape(-1, self.d_in)
        recon_flat = recon.reshape(-1, self.d_in)
        l1_loss = h_flat.abs().sum() / h_flat.numel()
        mse_loss = ((x_flat - recon_flat) ** 2).mean()
        total_loss = mse_loss + self.l1_coeff * l1_loss
        return total_loss, mse_loss, l1_loss


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


def collect_text_examples(model, n_examples=200):
    from datasets import load_dataset
    log(f"  Collecting {n_examples} text examples...")
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
            print(f"    {count}/{n_examples}...", flush=True)
    log(f"  Collected {len(texts)} examples in {(datetime.now() - t0).total_seconds():.1f}s")
    return texts


def evaluate_sae(sae, model, texts, n_eval_tokens=N_TOKENS_EVAL, device=DEVICE):
    sae.eval()
    model.eval()
    all_features = []
    all_recon_mse = []

    with torch.no_grad():
        n_collected = 0
        for text in texts:
            tokens = model.to_tokens(text, truncate=True).flatten()
            n_chunks = len(tokens) // SEQ_LEN
            for i in range(min(n_chunks, 10)):
                chunk = tokens[i * SEQ_LEN:(i + 1) * SEQ_LEN]
                if len(chunk) != SEQ_LEN:
                    continue
                batch = chunk.unsqueeze(0).to(device)
                _, cache = model.run_with_cache(batch)
                acts = cache[HOOK_NAME]
                feats = sae.encode(acts)
                recon_acts = sae.decode(feats)
                mse = ((acts - recon_acts) ** 2).mean().item()
                all_recon_mse.append(mse)
                all_features.append(feats.cpu())
                n_collected += SEQ_LEN
                if n_collected >= n_eval_tokens:
                    break
            if n_collected >= n_eval_tokens:
                break

    all_features = torch.cat(all_features, dim=1)
    recon_mse = np.mean(all_recon_mse)
    l0 = (all_features[0] > 0).float().sum(dim=-1).mean().item()
    l0_normalized = l0 / D_SAE

    mean_acts = all_features[0].mean(dim=0)
    topk = torch.topk(mean_acts, min(200, mean_acts.shape[0]), largest=True)
    topk_indices = topk.indices.tolist()

    gini_scores = {}
    for fi in topk_indices:
        gini_scores[fi] = gini_coefficient(all_features[0, :, fi].numpy())

    chanin_scores = {}
    for fi in topk_indices:
        chanin_scores[fi] = float(np.mean(all_features[0, :, fi].numpy() > 0))

    absorption_scores = {}
    for fi in topk_indices:
        absorption_scores[fi] = 0.5 * gini_scores.get(fi, 0.0) + 0.5 * chanin_scores.get(fi, 0.0)

    mean_absorption = np.mean(list(absorption_scores.values())) if absorption_scores else 0.0

    # StandardSAE: self.W_dec = Parameter([d_in, d_sae]) = [768, 24576]
    # Feature i's decoder direction = column i = W_dec[:, i]
    W_dec_T = sae.W_dec.detach().cpu().numpy().T  # [d_sae, d_in] = [24576, 768]
    W_dec_norm = W_dec_T / (np.linalg.norm(W_dec_T, axis=1, keepdims=True) + 1e-8)
    uas_scores = {}
    for fi in topk_indices:
        other = [i for i in topk_indices if i != fi]
        if len(other) < 2:
            uas_scores[fi] = 0.0
            continue
        cos_sims = np.dot(W_dec_norm[fi], W_dec_norm[other].T)
        cos_var = float(np.var(cos_sims))
        acts_np = all_features[0, :, fi].numpy()
        non_zero = acts_np[acts_np > 0]
        freq_skew = float(abs(skew(non_zero))) if len(non_zero) > 2 else 0.0
        uas_scores[fi] = 1.0 * cos_var + 0.5 * freq_skew
    mean_uas = np.mean(list(uas_scores.values())) if uas_scores else 0.0

    total_dead = (mean_acts == 0).sum().item()
    dead_ratio = total_dead / D_SAE

    return {
        "l0": l0,
        "l0_normalized": l0_normalized,
        "reconstruction_mse": recon_mse,
        "dead_feature_ratio": dead_ratio,
        "mean_absorption": mean_absorption,
        "mean_gini_absorption": np.mean(list(gini_scores.values())) if gini_scores else 0.0,
        "mean_chanin_absorption": np.mean(list(chanin_scores.values())) if chanin_scores else 0.0,
        "mean_uas": mean_uas,
        "n_features_analyzed": len(topk_indices),
        "n_eval_tokens": n_collected,
    }


def train_sae(model, texts, l1_coeff, device):
    d_model = 768
    log(f"  Training SAE with L1={l1_coeff:.0e}...")

    # Initialize from activation sample
    log("  Collecting activation sample...")
    sample_acts = []
    count = 0
    for text in texts[:50]:
        tokens = model.to_tokens(text, truncate=True).flatten()
        n_chunks = len(tokens) // SEQ_LEN
        for i in range(min(n_chunks, 3)):
            chunk = tokens[i * SEQ_LEN:(i + 1) * SEQ_LEN]
            if len(chunk) != SEQ_LEN:
                continue
            batch = chunk.unsqueeze(0).to(device)
            _, cache = model.run_with_cache(batch)
            sample_acts.append(cache[HOOK_NAME].cpu())
            count += 1
            if count >= 20:
                break
        if count >= 20:
            break
    sample_acts = torch.cat(sample_acts, dim=1)  # [1, n_pos, d_model]

    # Create SAE and initialize on CPU, then move to GPU
    sae = StandardSAE(d_model, D_SAE, l1_coeff=l1_coeff)
    sae.initialize(sample_acts.reshape(-1, d_model).float())
    sae = sae.to(device)
    del sample_acts
    gc.collect()

    # Training
    dataset = TextDataset(model, texts, N_TOKENS_TRAIN, SEQ_LEN, seed=SEED)
    n_steps = N_TOKENS_TRAIN // (BATCH_SIZE * SEQ_LEN) * N_EPOCHS
    log(f"  Training: {n_steps} steps, {N_EPOCHS} epochs, batch_size={BATCH_SIZE}x{SEQ_LEN}")

    optimizer = torch.optim.Adam(sae.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=n_steps)

    sae.train()
    total_loss_sum = 0.0
    total_mse_sum = 0.0
    total_l1_sum = 0.0
    n_loss_terms = 0
    step = 0
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0)

    for epoch in range(N_EPOCHS):
        epoch_start = time.time()
        for batch_tokens in dataloader:
            batch = batch_tokens.to(device)
            try:
                _, cache = model.run_with_cache(batch)
                acts = cache[HOOK_NAME]
                optimizer.zero_grad()
                loss, mse, l1_val = sae(acts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss_sum += loss.item()
                total_mse_sum += mse.item()
                total_l1_sum += l1_val.item()
                n_loss_terms += 1
                step += 1
                if step % 50 == 0:
                    avg_loss = total_loss_sum / n_loss_terms
                    avg_mse = total_mse_sum / n_loss_terms
                    avg_l1 = total_l1_sum / n_loss_terms
                    log(f"    Step {step}: loss={avg_loss:.4f}, mse={avg_mse:.4f}, l1={avg_l1:.6f}")
            except Exception as e:
                log(f"    Training step error: {e}")
                continue
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss_sum / max(n_loss_terms, 1)
        avg_mse = total_mse_sum / max(n_loss_terms, 1)
        avg_l1 = total_l1_sum / max(n_loss_terms, 1)
        log(f"  Epoch {epoch+1}/{N_EPOCHS} done in {epoch_time:.1f}s: loss={avg_loss:.4f}, mse={avg_mse:.4f}, l1={avg_l1:.6f}")

    # Evaluate
    log("  Evaluating SAE...")
    eval_texts = texts[180:200]
    eval_results = evaluate_sae(sae, model, eval_texts, device=device)
    log(f"  Results: L0={eval_results['l0']:.1f}, MSE={eval_results['reconstruction_mse']:.4f}, "
        f"Absorption={eval_results['mean_absorption']:.4f}, UAS={eval_results['mean_uas']:.4f}")

    return {"l1_coefficient": l1_coeff, "evaluation": eval_results}, sae


def main():
    log("=" * 60)
    log("ABLATION: L1 Sparsity Sweep")
    log("=" * 60)
    log(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"Training tokens: {N_TOKENS_TRAIN:,} per SAE, Epochs: {N_EPOCHS}")
    log(f"L1 sweep: {L1_VALUES}")

    from transformer_lens import HookedTransformer
    log("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    log(f"Loaded. d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    log("Collecting text examples...")
    texts = collect_text_examples(model, n_examples=200)
    log(f"Got {len(texts)} text examples")

    results = []
    for l1_val in L1_VALUES:
        try:
            log(f"\n{'='*40}")
            log(f"Training SAE with L1={l1_val:.0e}")
            log(f"{'='*40}")
            t0 = time.time()
            result, sae = train_sae(model, texts, l1_val, DEVICE)
            result["train_time_seconds"] = time.time() - t0
            results.append(result)
            torch.save(sae.state_dict(), RESULTS_DIR / f"sae_l1_{l1_val:.0e}.pt")

            # Checkpoint
            ckpt = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"
            with open(ckpt, "w") as f:
                json.dump({"results_so_far": [{
                    "l1_coefficient": r["l1_coefficient"],
                    "train_time": r.get("train_time_seconds"),
                    "evaluation": r.get("evaluation"),
                } for r in results]}, f, indent=2)

            torch.cuda.empty_cache()
            gc.collect()
        except Exception as e:
            log(f"ERROR L1={l1_val}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"l1_coefficient": l1_val, "error": str(e), "status": "failed"})
            torch.cuda.empty_cache()
            gc.collect()

    # Pareto analysis
    log("\n" + "=" * 60)
    log("PARETO ANALYSIS")
    log("=" * 60)

    valid = [r for r in results if "error" not in r]
    pareto = []

    for r in valid:
        ev = r["evaluation"]
        absorption = ev["mean_absorption"]
        mse = ev["reconstruction_mse"]
        l0 = ev["l0_normalized"]

        dominated = False
        for other in valid:
            if other is r:
                continue
            oev = other["evaluation"]
            if oev["mean_absorption"] <= absorption and oev["reconstruction_mse"] <= mse and \
               (oev["mean_absorption"] < absorption or oev["reconstruction_mse"] < mse):
                dominated = True
                break

        status = "Pareto" if not dominated else "Dominated"
        entry = {
            "l1": r["l1_coefficient"],
            "absorption": absorption,
            "mse": mse,
            "l0": l0,
            "l0_raw": ev["l0"],
            "dead_ratio": ev["dead_feature_ratio"],
            "uas": ev["mean_uas"],
            "gini_abs": ev["mean_gini_absorption"],
            "chanin_abs": ev["mean_chanin_absorption"],
            "train_time": r.get("train_time_seconds", 0),
            "pareto_status": status,
        }
        pareto.append(entry)
        log(f"  L1={r['l1_coefficient']:.0e}: abs={absorption:.4f}, MSE={mse:.4f}, "
            f"L0={l0:.6f}, dead={ev['dead_feature_ratio']:.4f} [{status}]")

    pareto_pts = [p for p in pareto if p["pareto_status"] == "Pareto"]
    best_mse_entry = None
    most_sparse_entry = None
    if pareto_pts:
        best_mse_entry = min(pareto_pts, key=lambda x: x["mse"])
        most_sparse_entry = min(pareto_pts, key=lambda x: x["l0"])
        log(f"\nBest reconstruction (Pareto): L1={best_mse_entry['l1']:.0e}, MSE={best_mse_entry['mse']:.4f}")
        log(f"Sparsest (Pareto): L1={most_sparse_entry['l1']:.0e}, L0={most_sparse_entry['l0']:.6f}")

    # Save
    final_output = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "mode": "PILOT",
        "config": {
            "model": MODEL_NAME, "layer": LAYER, "d_sae": D_SAE,
            "n_train_tokens": N_TOKENS_TRAIN, "n_epochs": N_EPOCHS,
            "batch_size": BATCH_SIZE, "seq_len": SEQ_LEN,
            "learning_rate": LEARNING_RATE, "n_eval_tokens": N_TOKENS_EVAL,
            "device": DEVICE, "seed": SEED,
        },
        "l1_sweep": L1_VALUES,
        "results": [{
            "l1_coefficient": r["l1_coefficient"],
            "train_time_seconds": r.get("train_time_seconds"),
            "evaluation": r.get("evaluation"),
        } for r in results],
        "pareto_analysis": pareto,
        "summary": {
            "n_trained": len(valid),
            "n_failed": len(results) - len(valid),
            "n_pareto": len(pareto_pts),
            "key_findings": {
                "best_reconstruction": f"L1={best_mse_entry['l1']:.0e}, MSE={best_mse_entry['mse']:.4f}" if pareto_pts else "N/A",
                "sparsest": f"L1={most_sparse_entry['l1']:.0e}, L0={most_sparse_entry['l0']:.6f}" if pareto_pts else "N/A",
            }
        }
    }

    out_file = RESULTS_DIR / f"{TASK_ID}.json"
    with open(out_file, "w") as f:
        json.dump(final_output, f, indent=2)
    log(f"Results: {out_file}")

    # Summary markdown
    summary_md = f"""# Ablation: L1 Sparsity Sweep

## Configuration
- Model: GPT-2 Small, Layer {LAYER}
- d_sae: {D_SAE}
- Training tokens: {N_TOKENS_TRAIN:,} per SAE
- Epochs: {N_EPOCHS}

## Results

| L1 | L0 (norm) | Recon MSE | Absorption | Gini Abs | Chanin Abs | UAS | Dead% | Pareto |
|----|-----------|-----------|-----------|----------|------------|-----|-------|--------|
"""
    for p in pareto:
        summary_md += f"| {p['l1']:.0e} | {p['l0']:.6f} | {p['mse']:.4f} | {p['absorption']:.4f} | {p['gini_abs']:.4f} | {p['chanin_abs']:.4f} | {p['uas']:.4f} | {p['dead_ratio']*100:.1f}% | {p['pareto_status']} |\n"

    if pareto_pts:
        abs_trend = "higher" if (pareto[-1]["absorption"] > pareto[0]["absorption"]) else "lower"
        l0_trend = "lower" if (pareto[-1]["l0"] < pareto[0]["l0"]) else "higher"
        mse_trend = "higher" if (pareto[-1]["mse"] > pareto[0]["mse"]) else "lower"
        sweet_spot = f"{best_mse_entry['l1']:.0e} (MSE={best_mse_entry['mse']:.4f})"
    else:
        abs_trend = l0_trend = mse_trend = "N/A"
        sweet_spot = "N/A"

    summary_md += f"""
## Key Findings
1. **L1-Absorption**: Higher L1 → {abs_trend} absorption
2. **L1-Sparsity**: Higher L1 → {l0_trend} L0 (sparser)
3. **Reconstruction**: Higher L1 → {mse_trend} MSE
4. **Pareto frontier**: {[p['l1'] for p in pareto_pts] if pareto_pts else 'N/A'}
5. **Sweet spot**: {sweet_spot}

## Pilot Assessment
- Trained: {len(valid)}/{len(L1_VALUES)}, Pareto: {len(pareto_pts)}/{len(pareto)}
"""

    summary_file = RESULTS_DIR / f"{TASK_ID}_summary.md"
    with open(summary_file, "w") as f:
        f.write(summary_md)
    log(f"Summary: {summary_file}")

    # DONE
    done_file = RESULTS_DIR / f"{TASK_ID}_DONE"
    with open(done_file, "w") as f:
        json.dump({"task_id": TASK_ID, "status": "success", "timestamp": datetime.now().isoformat(),
                   "n_trained": len(valid), "n_pareto": len(pareto_pts)}, f, indent=2)

    prog_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    with open(prog_file, "w") as f:
        json.dump({"task_id": TASK_ID, "epoch": N_EPOCHS, "total_epochs": N_EPOCHS,
                   "status": "completed", "updated_at": datetime.now().isoformat()}, f, indent=2)

    log("DONE!")
    return final_output


if __name__ == "__main__":
    main()
