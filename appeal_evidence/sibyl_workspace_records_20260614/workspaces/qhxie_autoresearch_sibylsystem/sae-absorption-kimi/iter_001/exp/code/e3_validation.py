#!/usr/bin/env python3
"""E3 Validation: Correlate task-agnostic absorption metric with first-letter benchmark.

Uses GPT-2 Small checkpoints since Gemma-2-2B is gated.
Computes both metrics on the same set of SAEs and reports correlation.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sae_lens import SAE
from transformer_lens import HookedTransformer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current/exp/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PID_FILE = RESULTS_DIR / "e3_validation.pid"
PROGRESS_FILE = RESULTS_DIR / "e3_validation_PROGRESS.json"
DONE_FILE = RESULTS_DIR / "e3_validation_DONE"

PID_FILE.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": "e3_validation",
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
        "task_id": "e3_validation",
        "status": status,
        "summary": summary,
        "final_progress": progress,
        "timestamp": datetime.now().isoformat(),
    }))


torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------------------
# Geography hierarchy: parent (continent) -> child (country)
# ---------------------------------------------------------------------------
GEO_HIERARCHIES = [
    ("Europe", ["France", "Germany", "Italy", "Spain", "Polland"]),
    ("Asia", ["China", "Japan", "India", "Thailand", "Vietnam"]),
    ("Africa", ["Egypt", "Nigeria", "Kenya", "Ethiopia", "Morocco"]),
    ("North America", ["Canada", "Mexico", "Cuba", "Guatemala", "Honduras"]),
    ("South America", ["Brazil", "Argentina", "Chile", "Peru", "Colombia"]),
]

PROMPT_TEMPLATES = [
    "{concept} is a place with rich history.",
    "Many people visit {concept} every year.",
    "The culture of {concept} is fascinating.",
    "I have always wanted to travel to {concept}.",
    "{concept} has a unique geography and climate.",
]


def generate_sentences(concept, n_per_concept=20):
    """Generate simple sentences about a concept."""
    sentences = []
    for i in range(n_per_concept):
        template = PROMPT_TEMPLATES[i % len(PROMPT_TEMPLATES)]
        sentences.append(template.format(concept=concept))
    return sentences


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
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
def get_activations(model, sae, sentences, tokenizer, max_length=64):
    """Get residual-stream and SAE activations for sentences."""
    hook_name = get_hook_name(sae)
    hook_layer = get_hook_layer(sae)

    tokens = tokenizer(sentences, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    tokens = {k: v.to(DEVICE) for k, v in tokens.items()}

    _, cache = model.run_with_cache(tokens["input_ids"], stop_at_layer=hook_layer + 1, names_filter=[hook_name])
    acts = cache[hook_name]  # [batch, pos, d_model]

    # Mean-pool over non-padding positions
    mask = tokens["attention_mask"].unsqueeze(-1).float()
    acts_pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1)  # [batch, d_model]

    sae_acts = sae.encode(acts_pooled)  # [batch, d_sae]

    return acts_pooled.cpu().numpy(), sae_acts.cpu().numpy()


def train_probe(X, y, n_epochs=200, lr=0.05):
    """Train a logistic regression probe and return accuracy + model weights."""
    X_t = torch.tensor(X, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    n_classes = int(y_t.max().item()) + 1

    probe = nn.Linear(X_t.size(1), n_classes, device=DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = probe(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = probe(X_t).argmax(dim=-1)
        acc = (preds == y_t).float().mean().item()

    return acc


def probe_sae_latents(sae_acts, y, n_epochs=200, lr=0.05):
    """Train probe on SAE latents and return accuracy."""
    X_t = torch.tensor(sae_acts, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    n_classes = int(y_t.max().item()) + 1

    probe = nn.Linear(X_t.size(1), n_classes, device=DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    for _ in range(n_epochs):
        optimizer.zero_grad()
        logits = probe(X_t)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = probe(X_t).argmax(dim=-1)
        acc = (preds == y_t).float().mean().item()
    return acc


def k_sparse_probe_accuracy(sae_acts, y, k=10):
    """Use top-k SAE latents as features for a probe."""
    sae_acts_t = torch.tensor(sae_acts, dtype=torch.float32, device=DEVICE)
    topk_vals, topk_indices = torch.topk(sae_acts_t, k=k, dim=-1)

    sparse_acts = torch.zeros_like(sae_acts_t)
    sparse_acts.scatter_(-1, topk_indices, topk_vals)

    y_t = torch.tensor(y, dtype=torch.long, device=DEVICE)
    n_classes = int(y_t.max().item()) + 1

    probe = nn.Linear(sparse_acts.size(1), n_classes, device=DEVICE)
    optimizer = torch.optim.Adam(probe.parameters(), lr=0.05)

    for _ in range(200):
        optimizer.zero_grad()
        logits = probe(sparse_acts)
        loss = F.cross_entropy(logits, y_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = probe(sparse_acts).argmax(dim=-1)
        acc = (preds == y_t).float().mean().item()
    return acc


def compute_task_agnostic_absorption(model, sae, tokenizer):
    """Compute task-agnostic absorption score for a single SAE."""
    scores = []
    for parent, children in GEO_HIERARCHIES:
        parent_sentences = generate_sentences(parent, n_per_concept=20)
        child_sentences = []
        labels = []
        for child in children:
            child_sentences.extend(generate_sentences(child, n_per_concept=20))
            labels.extend([1] * 20)

        all_sentences = parent_sentences + child_sentences
        all_labels = [0] * len(parent_sentences) + labels

        resid_acts, sae_acts = get_activations(model, sae, all_sentences, tokenizer)

        resid_acc = train_probe(resid_acts, np.array(all_labels))
        if resid_acc < 0.6:
            continue
        sae_acc = probe_sae_latents(sae_acts, np.array(all_labels))
        k_sparse_acc = k_sparse_probe_accuracy(sae_acts, np.array(all_labels), k=10)

        sae_drop = max(0.0, resid_acc - sae_acc) / resid_acc
        k_sparse_drop = max(0.0, resid_acc - k_sparse_acc) / resid_acc
        scores.append(float(max(sae_drop, k_sparse_drop)))

    return float(np.mean(scores)) if scores else None


@torch.no_grad()
def compute_first_letter_absorption(model, sae, tokenizer, hook_name):
    """Simplified first-letter absorption metric."""
    sae.eval()
    model.eval()

    words = [
        "apple", "bear", "cat", "dog", "eel", "fox", "goat", "hat", "ice", "jug",
        "kite", "lion", "moth", "nest", "owl", "pig", "quail", "rat", "sun", "toy",
        "umbrella", "vase", "wolf", "xray", "yak", "zebra"
    ]

    hook_layer = get_hook_layer(sae)
    absorption_scores = []
    for word in words:
        if len(word) < 2:
            continue
        first_letter = word[0].upper()
        prompt = f"The word {first_letter}"
        tokens = tokenizer(prompt, return_tensors="pt")["input_ids"].to(DEVICE)
        _, cache = model.run_with_cache(tokens)
        if hook_layer is None:
            acts = model.embed(tokens)
        else:
            acts = cache[hook_name]
        feats = sae.encode(acts)
        last_pos = feats.shape[1] - 1
        top_feat = feats[0, last_pos].argmax().item()

        word_tokens = tokenizer(word, return_tensors="pt")["input_ids"].to(DEVICE)
        _, cache2 = model.run_with_cache(word_tokens)
        if hook_layer is None:
            acts2 = model.embed(word_tokens)
        else:
            acts2 = cache2[hook_name]
        feats2 = sae.encode(acts2)
        top3 = feats2[0].topk(3, dim=-1).indices
        absorbed = (top3 == top_feat).any().item()
        absorption_scores.append(float(absorbed))

    return float(np.mean(absorption_scores)) if absorption_scores else None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    start_time = time.time()

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

    report_progress(0, len(CHECKPOINTS), step=0, total_steps=len(CHECKPOINTS))

    print(f"Loading GPT-2 Small on {DEVICE} (GPU {GPU_ID})...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=DEVICE)
    tokenizer = model.tokenizer

    results = []
    total = len(CHECKPOINTS)

    for idx, ckpt in enumerate(CHECKPOINTS):
        release = ckpt["release"]
        sae_id = ckpt["sae_id"]
        print(f"\n[{idx+1}/{total}] Loading {release}/{sae_id} ...")
        try:
            sae = SAE.from_pretrained(release, sae_id, device=DEVICE)
            if isinstance(sae, tuple):
                sae = sae[0]
            sae.eval()
            hook_name = get_hook_name(sae)
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

        print(f"  Computing task-agnostic absorption...")
        task_agnostic = compute_task_agnostic_absorption(model, sae, tokenizer)
        print(f"    Task-agnostic={task_agnostic}")

        print(f"  Computing first-letter absorption...")
        first_letter = compute_first_letter_absorption(model, sae, tokenizer, hook_name)
        print(f"    First-letter={first_letter}")

        results.append({
            "release": release,
            "sae_id": sae_id,
            "family": ckpt["family"],
            "task_agnostic_absorption": task_agnostic,
            "first_letter_absorption": first_letter,
        })

        report_progress(epoch=idx+1, total_epochs=total, step=idx+1, total_steps=total,
                        metric={"loaded": True, "sae_id": sae_id, "task_agnostic": task_agnostic, "first_letter": first_letter})

        del sae
        torch.cuda.empty_cache()

    # Compute correlation
    valid = [r for r in results if r.get("task_agnostic_absorption") is not None and r.get("first_letter_absorption") is not None]
    if len(valid) >= 3:
        x = np.array([r["first_letter_absorption"] for r in valid])
        y = np.array([r["task_agnostic_absorption"] for r in valid])
        pearson_r = float(np.corrcoef(x, y)[0, 1])
        from scipy import stats
        spearman_r, p_value = stats.spearmanr(x, y)
    else:
        pearson_r = None
        spearman_r = None
        p_value = None

    out = {
        "task_id": "e3_validation",
        "model": "gpt2-small",
        "n_checkpoints": total,
        "valid_checkpoints": len(valid),
        "seed": SEED,
        "device": DEVICE,
        "gpu_id": GPU_ID,
        "correlation": {
            "pearson_r": pearson_r,
            "spearman_r": float(spearman_r) if spearman_r is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
        },
        "results": results,
        "runtime_seconds": time.time() - start_time,
        "timestamp": datetime.now().isoformat(),
    }

    out_path = RESULTS_DIR / "e3_validation_correlation_results.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nResults saved to {out_path}")

    elapsed = time.time() - start_time
    mark_done(status="success", summary=f"Completed {len(valid)}/{total} checkpoints in {elapsed/60:.1f} minutes. Pearson r={pearson_r}")
    print(f"[{datetime.now().isoformat()}] Done in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        raise
