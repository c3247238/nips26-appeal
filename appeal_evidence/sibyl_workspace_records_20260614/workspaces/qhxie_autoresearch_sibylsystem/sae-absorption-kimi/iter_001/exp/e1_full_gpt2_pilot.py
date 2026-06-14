"""
E1 Full GPT-2 Pilot: Multi-objective metric pipeline on 10 GPT-2 Small checkpoints.
Computes L0, explained variance, dead-neuron rate, CE loss recovered, absorption, and hedging.
Pilot mode: 2048 tokens, seed 42.
"""
import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
SEED = 42
N_TOKENS = 2048
MAX_SEQ_LEN = 128
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path("exp/results/full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "e1_full_gpt2"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

CHECKPOINTS = [
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.0.hook_resid_pre", "family": "Standard"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.4.hook_resid_pre", "family": "Standard"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.8.hook_resid_pre", "family": "Standard"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.11.hook_resid_pre", "family": "Standard"},
    {"release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.4.hook_resid_post", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-32k", "sae_id": "blocks.8.hook_resid_post", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-128k", "sae_id": "blocks.4.hook_resid_post", "family": "TopK"},
    {"release": "gpt2-small-resid-post-v5-128k", "sae_id": "blocks.8.hook_resid_post", "family": "TopK"},
    {"release": "gpt2-small-mlp-out-v5-32k", "sae_id": "blocks.8.hook_mlp_out", "family": "TopK_MLP"},
    {"release": "gpt2-small-attn-out-v5-32k", "sae_id": "blocks.8.hook_attn_out", "family": "TopK_Attn"},
]

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def write_pid():
    PID_FILE.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_c4_tokens(n_tokens=N_TOKENS, max_seq_len=MAX_SEQ_LEN):
    print("Loading C4 subset...")
    ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
    ds = ds.shuffle(seed=SEED)
    texts = []
    total_chars = 0
    target_chars = n_tokens * 6
    for item in ds:
        text = item["text"]
        texts.append(text)
        total_chars += len(text)
        if total_chars >= target_chars:
            break
    full_text = "\n".join(texts)
    print(f"Collected ~{len(full_text)} chars from C4 ({len(texts)} snippets).")
    return full_text

@torch.no_grad()
def compute_metrics(model, sae, texts, tokenizer, hook_name, device):
    """Compute L0, explained variance, dead neurons, CE loss recovered."""
    sae.eval()
    model.eval()

    all_l0 = []
    all_ev = []
    all_ce_orig = []
    all_ce_rec = []
    activation_counts = torch.zeros(sae.cfg.d_sae, device=device)
    total_tokens = 0

    # Tokenize in chunks
    tokens_list = []
    for text in texts:
        toks = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, return_tensors="pt")["input_ids"]
        tokens_list.append(toks)

    for toks in tokens_list:
        toks = toks.to(device)
        if toks.numel() == 0:
            continue

        with torch.cuda.amp.autocast():
            logits_orig, cache = model.run_with_cache(toks, return_type="logits")
            acts = cache[hook_name]

            feats = sae.encode(acts)
            rec_acts = sae.decode(feats)

            # L0
            l0 = (feats > 0).float().sum(dim=-1).mean().item()
            all_l0.append(l0)

            # Explained variance
            var_orig = (acts - acts.mean(dim=-1, keepdim=True)).pow(2).sum(dim=-1)
            var_err = (acts - rec_acts).pow(2).sum(dim=-1)
            ev = (1 - var_err / (var_orig + 1e-8)).mean().item()
            all_ev.append(ev)

            # Dead neuron tracking
            active = (feats > 0).float().sum(dim=[0, 1])
            if active.dim() == 0:
                active = active.unsqueeze(0)
            activation_counts += active
            total_tokens += feats.shape[0] * feats.shape[1]

            # CE loss recovered via hook replacement
            def hook_fn(activation, hook):
                return rec_acts

            logits_rec = model.run_with_hooks(toks, fwd_hooks=[(hook_name, hook_fn)], return_type="logits")
            ce_orig = torch.nn.functional.cross_entropy(
                logits_orig[:, :-1, :].reshape(-1, logits_orig.shape[-1]),
                toks[:, 1:].reshape(-1),
                reduction="mean",
            ).item()
            ce_rec = torch.nn.functional.cross_entropy(
                logits_rec[:, :-1, :].reshape(-1, logits_rec.shape[-1]),
                toks[:, 1:].reshape(-1),
                reduction="mean",
            ).item()
            all_ce_orig.append(ce_orig)
            all_ce_rec.append(ce_rec)

    dead_frac = (activation_counts < 1e-5 * total_tokens).float().mean().item()
    mean_l0 = float(np.mean(all_l0))
    mean_ev = float(np.mean(all_ev))
    mean_ce_orig = float(np.mean(all_ce_orig)) if all_ce_orig else None
    mean_ce_rec = float(np.mean(all_ce_rec)) if all_ce_rec else None
    ce_loss_recovered = None
    if mean_ce_orig is not None and mean_ce_orig != 0:
        ce_loss_recovered = (mean_ce_orig - mean_ce_rec) / mean_ce_orig * 100.0

    return {
        "l0": mean_l0,
        "explained_variance": mean_ev,
        "dead_neuron_fraction": dead_frac,
        "ce_loss_orig": mean_ce_orig,
        "ce_loss_rec": mean_ce_rec,
        "ce_loss_recovered_pct": ce_loss_recovered,
    }

@torch.no_grad()
def compute_absorption_metric(model, sae, tokenizer, hook_name, device):
    """
    Simplified first-letter absorption metric.
    For each letter A-Z, form prompt 'The word <letter>' and check if top SAE feature
    for the letter token is also active for the full word token (absorption proxy).
    """
    sae.eval()
    model.eval()

    words = [
        "apple", "bear", "cat", "dog", "eel", "fox", "goat", "hat", "ice", "jug",
        "kite", "lion", "moth", "nest", "owl", "pig", "quail", "rat", "sun", "toy",
        "umbrella", "vase", "wolf", "xray", "yak", "zebra"
    ]

    absorption_scores = []
    for word in words:
        if len(word) < 2:
            continue
        first_letter = word[0].upper()
        prompt = f"The word {first_letter}"
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        _, cache = model.run_with_cache(tokens)
        acts = cache[hook_name]
        feats = sae.encode(acts)
        last_pos = feats.shape[1] - 1
        top_feat = feats[0, last_pos].argmax().item()

        word_tokens = tokenizer(word, return_tensors="pt")["input_ids"].to(device)
        _, cache2 = model.run_with_cache(word_tokens)
        acts2 = cache2[hook_name]
        feats2 = sae.encode(acts2)
        top3 = feats2[0].topk(3, dim=-1).indices
        absorbed = (top3 == top_feat).any().item()
        absorption_scores.append(float(absorbed))

    mean_abs = float(np.mean(absorption_scores)) if absorption_scores else None
    return {"absorption_rate": mean_abs, "n_samples": len(absorption_scores)}

@torch.no_grad()
def compute_hedging_metric(model, sae, tokenizer, hook_name, device, n_pairs=25):
    """
    Simplified hedging metric: correlated token pairs (antonyms, synonyms).
    For each pair, check if the same top feature activates for both tokens.
    Higher overlap = more hedging.
    """
    sae.eval()
    model.eval()

    pairs = [
        ("good", "bad"), ("happy", "sad"), ("big", "small"), ("hot", "cold"),
        ("fast", "slow"), ("rich", "poor"), ("love", "hate"), ("war", "peace"),
        ("light", "dark"), ("strong", "weak"), ("high", "low"), ("old", "young"),
        ("easy", "hard"), ("clean", "dirty"), ("safe", "dangerous"),
        ("begin", "end"), ("win", "lose"), ("accept", "reject"), ("build", "destroy"),
        ("buy", "sell"), ("create", "destroy"), ("enter", "exit"), ("find", "lose"),
        ("give", "take"), ("open", "close")
    ]
    pairs = pairs[:n_pairs]

    overlaps = []
    for w1, w2 in pairs:
        for w in (w1, w2):
            toks = tokenizer(w, return_tensors="pt")["input_ids"].to(device)
            _, cache = model.run_with_cache(toks)
            acts = cache[hook_name]
            feats = sae.encode(acts)
            top = feats[0, -1].argmax().item()
            if w == w1:
                top1 = top
            else:
                top2 = top
        overlaps.append(1.0 if top1 == top2 else 0.0)

    mean_hedge = float(np.mean(overlaps)) if overlaps else None
    return {"hedging_rate": mean_hedge, "n_pairs": len(pairs)}

# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    write_pid()
    set_seed(SEED)
    start_time = time.time()

    print(f"[{datetime.now().isoformat()}] Starting {TASK_ID} on device {DEVICE} (GPU {GPU_ID})")

    print("Loading GPT-2 Small model...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = model.tokenizer

    full_text = load_c4_tokens(N_TOKENS)
    text_chunks = [full_text[i:i+1024] for i in range(0, len(full_text), 1024)]
    print(f"Split into {len(text_chunks)} text chunks.")

    results = []
    total = len(CHECKPOINTS)

    for idx, ckpt in enumerate(CHECKPOINTS):
        release = ckpt["release"]
        sae_id = ckpt["sae_id"]
        print(f"\n[{idx+1}/{total}] Loading {release}/{sae_id} ...")
        try:
            sae = SAE.from_pretrained(
                release=release,
                sae_id=sae_id,
                device=DEVICE,
            )
            if isinstance(sae, tuple):
                sae = sae[0]
            hook_name = getattr(sae.cfg, 'hook_name', None) or sae.cfg.metadata.get('hook_name')
        except Exception as e:
            print(f"  ERROR loading {sae_id}: {e}")
            results.append({
                "release": release,
                "sae_id": sae_id,
                "family": ckpt["family"],
                "error": str(e),
            })
            report_progress(epoch=idx+1, total_epochs=total, step=idx+1, total_steps=total,
                            metric={"loaded": False, "sae_id": sae_id})
            continue

        print(f"  Hook: {hook_name}")
        print(f"  Computing metrics...")
        metrics = compute_metrics(model, sae, text_chunks, tokenizer, hook_name, DEVICE)
        print(f"    L0={metrics['l0']:.2f}, EV={metrics['explained_variance']:.4f}, Dead={metrics['dead_neuron_fraction']:.4f}, CE_Rec={metrics['ce_loss_recovered_pct']}")

        print(f"  Computing absorption...")
        absorption = compute_absorption_metric(model, sae, tokenizer, hook_name, DEVICE)
        print(f"    Absorption={absorption['absorption_rate']}")

        print(f"  Computing hedging...")
        hedging = compute_hedging_metric(model, sae, tokenizer, hook_name, DEVICE)
        print(f"    Hedging={hedging['hedging_rate']}")

        results.append({
            "release": release,
            "sae_id": sae_id,
            "family": ckpt["family"],
            "hook_name": hook_name,
            "metrics": metrics,
            "absorption": absorption,
            "hedging": hedging,
        })

        report_progress(epoch=idx+1, total_epochs=total, step=idx+1, total_steps=total,
                        metric={"loaded": True, "sae_id": sae_id, "l0": metrics["l0"]})

        del sae
        torch.cuda.empty_cache()
        gc.collect()

    out_path = RESULTS_DIR / f"{TASK_ID}_results.json"
    out_path.write_text(json.dumps({
        "task_id": TASK_ID,
        "model": "gpt2-small",
        "n_tokens": N_TOKENS,
        "seed": SEED,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "results": results,
        "runtime_seconds": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
    }, indent=2, default=str))

    print(f"\nResults saved to {out_path}")
    elapsed = time.time() - start_time
    mark_done(status="success", summary=f"Completed {len(results)} checkpoints in {elapsed/60:.1f} minutes.")
    print(f"[{datetime.now().isoformat()}] Done in {elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failed", summary=str(e))
        raise
