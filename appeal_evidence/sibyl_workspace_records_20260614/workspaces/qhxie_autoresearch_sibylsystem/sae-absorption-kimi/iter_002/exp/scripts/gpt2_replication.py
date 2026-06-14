#!/usr/bin/env python3
"""Replication: Semantic-Hierarchy and Non-Hierarchy Absorption on GPT-2 Small.

Runs the custom semantic-hierarchy and non-hierarchy absorption pipelines on
GPT-2 small (layer 8, resid_pre / resid_post) using 2 representative SAEs
(Standard res-jb and TopK resid-post v5-32k). Tests whether the correlation
pattern replicates across models.
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformer_lens import HookedTransformer

from sae_lens import SAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "gpt2_replication"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_NAME = "gpt2-small"
LAYER = 8
SAMPLES_PER_CONCEPT = 100
K_SPARSE = 10

SAE_SPECS = [
    {
        "name": "Standard_resjb_blocks8_resid_pre",
        "family": "Standard",
        "release": "gpt2-small-res-jb",
        "sae_id": "blocks.8.hook_resid_pre",
        "hook_name": "blocks.8.hook_resid_pre",
        "is_random": False,
    },
    {
        "name": "TopK_residpost_v5_32k_blocks8_resid_post",
        "family": "TopK",
        "release": "gpt2-small-resid-post-v5-32k",
        "sae_id": "blocks.8.hook_resid_post",
        "hook_name": "blocks.8.hook_resid_post",
        "is_random": False,
    },
]

HIERARCHIES = [
    {"parent": "building", "children": ["house", "school", "library"]},
    {"parent": "container", "children": ["box", "bag", "cup"]},
    {"parent": "tool", "children": ["fork", "comb", "rake"]},
    {"parent": "room", "children": ["cell", "court", "hall"]},
    {"parent": "device", "children": ["fan", "key"]},
    {"parent": "book", "children": ["album", "journal"]},
    {"parent": "tree", "children": ["ash", "poon"]},
    {"parent": "food", "children": ["fish", "water"]},
    {"parent": "fruit", "children": ["berry", "seed"]},
    {"parent": "animal", "children": ["pet", "male"]},
]

PAIRS = [
    {"word_a": "big", "word_b": "large"},
    {"word_a": "fast", "word_b": "quick"},
    {"word_a": "begin", "word_b": "start"},
    {"word_a": "doctor", "word_b": "hospital"},
    {"word_a": "student", "word_b": "school"},
    {"word_a": "river", "word_b": "water"},
    {"word_a": "fire", "word_b": "heat"},
    {"word_a": "sun", "word_b": "light"},
    {"word_a": "tree", "word_b": "wood"},
    {"word_a": "stone", "word_b": "rock"},
]

# ---------------------------------------------------------------------------
# Process tracking
# ---------------------------------------------------------------------------
PID_FILE.write_text(str(os.getpid()))
start_time_iso = datetime.now().isoformat()


def report_progress(epoch, total_epochs, step=0, total_steps=0, message=""):
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "total_steps": total_steps,
                "message": message,
                "updated_at": datetime.now().isoformat(),
            }
        )
    )


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(
        json.dumps(
            {
                "task_id": TASK_ID,
                "status": status,
                "summary": summary,
                "final_progress": progress,
                "timestamp": datetime.now().isoformat(),
            }
        )
    )


def make_random_sae(base_sae):
    """Create a random-SAE control with completely random weights.

    Since the absorption metric is computed from latents (x @ W_enc + b_enc),
    randomizing W_dec alone has no effect. We randomize W_enc and b_enc
    (and W_dec for completeness) to ensure the latent representations are
    uninformative. Scales are matched to the original weight statistics.
    """
    import copy
    random_sae = copy.deepcopy(base_sae)
    with torch.no_grad():
        # Randomize encoder
        orig_enc = random_sae.W_enc
        new_enc = torch.randn_like(orig_enc)
        new_enc = new_enc * (orig_enc.std() / new_enc.std().clamp_min(1e-8))
        random_sae.W_enc = torch.nn.Parameter(new_enc)

        # Randomize encoder bias
        orig_b = random_sae.b_enc
        new_b = torch.randn_like(orig_b)
        new_b = new_b * (orig_b.std() / new_b.std().clamp_min(1e-8))
        random_sae.b_enc = torch.nn.Parameter(new_b)

        # Randomize decoder (completeness)
        orig_dec = random_sae.W_dec
        new_dec = torch.randn_like(orig_dec)
        new_dec = new_dec * (orig_dec.std() / new_dec.std().clamp_min(1e-8))
        random_sae.W_dec = torch.nn.Parameter(new_dec)
    return random_sae


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def make_hierarchy_sentences(parent, child, n):
    templates = [
        "The {parent}, which is a {child}, stands nearby.",
        "A {child} is a type of {parent}.",
        "We visited the {parent}; it was a beautiful {child}.",
        "The {child}, like every {parent}, has a purpose.",
        "I entered the {parent} and saw a {child} inside.",
        "A {child} belongs to the category of {parent}.",
        "The {parent} contained a small {child}.",
        "Every {child} is fundamentally a {parent}.",
        "The {parent} was designed as a {child}.",
        "A {child} is an example of a {parent}.",
        "They built the {parent} to be a {child}.",
        "The {child} serves the function of a {parent}.",
        "Look at that {parent}; specifically, it is a {child}.",
        "A {child} is one kind of {parent}.",
        "The {parent} and its {child} were both impressive.",
        "Inside the {parent}, there was a {child}.",
        "The {child} is classified as a {parent}.",
        "A {parent} such as a {child} is essential.",
        "The {child} represents a type of {parent}.",
        "We need a {parent}, preferably a {child}.",
    ]
    sentences = []
    for i in range(n):
        sentences.append(templates[i % len(templates)].format(parent=parent, child=child))
    return sentences


def generate_hierarchy_dataset(hierarchy, samples_per_concept):
    parent = hierarchy["parent"]
    children = hierarchy["children"]
    sentences = []
    labels = []
    for child in children:
        sents = make_hierarchy_sentences(parent, child, samples_per_concept)
        sentences.extend(sents)
        labels.extend([child] * len(sents))
    return sentences, labels


def make_pair_sentences(word_a, word_b, n):
    templates = [
        "The {word_a} and the {word_b} are related.",
        "A {word_a} is often associated with a {word_b}.",
        "We discussed the {word_a} and the {word_b}.",
        "The {word_b} reminds me of a {word_a}.",
        "Every {word_a} connects to a {word_b}.",
        "The {word_a} or {word_b} can be chosen.",
        "A {word_b} is similar to a {word_a}.",
        "The {word_a} versus {word_b} debate continues.",
        "We need a {word_a}, not a {word_b}.",
        "The {word_b} and {word_a} share meaning.",
        "A {word_a} may become a {word_b}.",
        "The {word_a} near the {word_b} stood out.",
        "Consider the {word_a} and the {word_b}.",
        "A {word_b} is like a {word_a} in context.",
        "The {word_a} or the {word_b} will suffice.",
        "We observed the {word_a} and the {word_b}.",
        "The {word_b} follows from the {word_a}.",
        "A {word_a} and a {word_b} were compared.",
        "The {word_a} relates closely to the {word_b}.",
        "Choose between the {word_a} and the {word_b}.",
    ]
    sentences = []
    for i in range(n):
        if i % 2 == 0:
            sentences.append(templates[i % len(templates)].format(word_a=word_a, word_b=word_b))
        else:
            sentences.append(templates[i % len(templates)].format(word_a=word_b, word_b=word_a))
    return sentences


def generate_pair_dataset(pair, samples_per_concept):
    word_a = pair["word_a"]
    word_b = pair["word_b"]
    sentences = []
    labels = []
    sents_a = make_pair_sentences(word_a, word_b, samples_per_concept)
    sents_b = make_pair_sentences(word_b, word_a, samples_per_concept)
    sentences.extend(sents_a)
    labels.extend([word_a] * len(sents_a))
    sentences.extend(sents_b)
    labels.extend([word_b] * len(sents_b))
    return sentences, labels


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_mean_pooled_activations(model, sentences, hook_name, batch_size=32):
    all_acts = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name]
        mask = (tokens != model.tokenizer.pad_token_id).unsqueeze(-1).float()
        pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_acts.append(pooled.cpu().float())
    return torch.cat(all_acts, dim=0).numpy()


def get_sae_latents(model, sae, sentences, hook_name, batch_size=32):
    all_latents = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            acts = cache[hook_name]
            latents = sae.encode(acts)
        mask = (tokens != model.tokenizer.pad_token_id).unsqueeze(-1).float()
        pooled = (latents * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_latents.append(pooled.cpu().float())
    return torch.cat(all_latents, dim=0).numpy()


def get_k_sparse_latents(sae_latents, k):
    sparse = np.zeros_like(sae_latents)
    for i in range(sae_latents.shape[0]):
        topk_idx = np.argsort(sae_latents[i])[-k:]
        sparse[i, topk_idx] = sae_latents[i, topk_idx]
    return sparse


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_multiclass_probe(X, y, classes):
    mask = np.isin(y, classes)
    X_mc = X[mask]
    y_mc = y[mask]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[lbl] for lbl in y_mc])

    if len(np.unique(y_idx)) < 2:
        raise ValueError(f"Only one class present among {classes}")

    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X_mc, y_idx)
    acc = clf.score(X_mc, y_idx)

    if len(classes) == 2:
        probs = clf.predict_proba(X_mc)[:, 1]
        auroc = roc_auc_score(y_idx, probs)
    else:
        probs = clf.predict_proba(X_mc)
        auroc = 0.0
        for i in range(len(classes)):
            y_binary = (y_idx == i).astype(int)
            if len(np.unique(y_binary)) == 2:
                auroc += roc_auc_score(y_binary, probs[:, i])
        auroc /= len(classes)

    return clf, acc, auroc


def train_binary_probe(X, y, classes):
    mask = np.isin(y, classes)
    X_bin = X[mask]
    y_bin = y[mask]

    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([class_to_idx[lbl] for lbl in y_bin])

    if len(np.unique(y_idx)) < 2:
        raise ValueError(f"Only one class present among {classes}")

    clf = LogisticRegression(max_iter=500, solver="lbfgs")
    clf.fit(X_bin, y_idx)
    acc = clf.score(X_bin, y_idx)

    probs = clf.predict_proba(X_bin)[:, 1]
    auroc = roc_auc_score(y_idx, probs)

    return clf, acc, auroc


# ---------------------------------------------------------------------------
# Absorption score (SAEBench formula)
# ---------------------------------------------------------------------------

def compute_absorption(resid_acc, sae_acc, k_sparse_acc):
    return max(
        0.0,
        (resid_acc - sae_acc) / resid_acc if resid_acc > 0 else 0.0,
        (resid_acc - k_sparse_acc) / resid_acc if resid_acc > 0 else 0.0,
    )


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main():
    total_steps = len(SAE_SPECS) * 2 + 3
    report_progress(1, total_steps, message="Starting gpt2_replication evaluation")

    total_start = time.time()

    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
    print("Model loaded.")

    all_sae_results = []
    per_hierarchy_aurocs = {h["parent"]: [] for h in HIERARCHIES}
    per_pair_aurocs = {f"{p['word_a']}-{p['word_b']}": [] for p in PAIRS}

    step = 2
    for idx, spec in enumerate(SAE_SPECS, start=1):
        name = spec["name"]
        family = spec["family"]
        release = spec["release"]
        sae_id = spec["sae_id"]
        hook_name = spec["hook_name"]
        is_random = spec["is_random"]

        report_progress(step, total_steps, message=f"Evaluating {name} - hierarchies")
        print(f"\n=== [{idx}/{len(SAE_SPECS)}] {name} ===")

        print(f"Loading SAE {release}/{sae_id}...")
        sae = SAE.from_pretrained(release=release, sae_id=sae_id, device=DEVICE)
        if isinstance(sae, tuple):
            sae = sae[0]

        if is_random:
            print("Permuting decoder directions for random control...")
            sae = make_random_sae(sae)

        sae_result = {
            "sae_name": name,
            "family": family,
            "release": release,
            "sae_id": sae_id,
            "hook_name": hook_name,
            "is_random": is_random,
            "hierarchies": [],
            "pairs": [],
        }

        # Semantic hierarchy absorption
        hierarchy_absorptions = []
        for h in HIERARCHIES:
            parent = h["parent"]
            children = h["children"]
            sentences, labels = generate_hierarchy_dataset(h, SAMPLES_PER_CONCEPT)
            labels = np.array(labels, dtype=object)

            resid_acts = get_mean_pooled_activations(model, sentences, hook_name)
            sae_latents = get_sae_latents(model, sae, sentences, hook_name)
            k_sparse = get_k_sparse_latents(sae_latents, K_SPARSE)

            _, resid_acc, resid_auroc = train_multiclass_probe(resid_acts, labels, children)
            _, sae_acc, sae_auroc = train_multiclass_probe(sae_latents, labels, children)
            _, k_acc, k_auroc = train_multiclass_probe(k_sparse, labels, children)

            absorption = compute_absorption(resid_acc, sae_acc, k_acc)
            hierarchy_absorptions.append(absorption)

            sae_result["hierarchies"].append({
                "parent": parent,
                "children": children,
                "resid_acc": float(resid_acc),
                "resid_auroc": float(resid_auroc),
                "sae_acc": float(sae_acc),
                "sae_auroc": float(sae_auroc),
                "k_sparse_acc": float(k_acc),
                "k_sparse_auroc": float(k_auroc),
                "absorption": float(absorption),
            })

            per_hierarchy_aurocs[parent].append(float(resid_auroc))

        sae_result["mean_hierarchy_absorption"] = float(np.mean(hierarchy_absorptions))
        print(f"  mean hierarchy absorption: {sae_result['mean_hierarchy_absorption']:.4f}")

        # Non-hierarchy control absorption
        report_progress(step + 1, total_steps, message=f"Evaluating {name} - control pairs")
        pair_absorptions = []
        for pair in PAIRS:
            word_a = pair["word_a"]
            word_b = pair["word_b"]
            sentences, labels = generate_pair_dataset(pair, SAMPLES_PER_CONCEPT)
            labels = np.array(labels, dtype=object)

            resid_acts = get_mean_pooled_activations(model, sentences, hook_name)
            sae_latents = get_sae_latents(model, sae, sentences, hook_name)
            k_sparse = get_k_sparse_latents(sae_latents, K_SPARSE)

            _, resid_acc, resid_auroc = train_binary_probe(resid_acts, labels, [word_a, word_b])
            _, sae_acc, sae_auroc = train_binary_probe(sae_latents, labels, [word_a, word_b])
            _, k_acc, k_auroc = train_binary_probe(k_sparse, labels, [word_a, word_b])

            absorption = compute_absorption(resid_acc, sae_acc, k_acc)
            pair_absorptions.append(absorption)

            sae_result["pairs"].append({
                "word_a": word_a,
                "word_b": word_b,
                "resid_acc": float(resid_acc),
                "resid_auroc": float(resid_auroc),
                "sae_acc": float(sae_acc),
                "sae_auroc": float(sae_auroc),
                "k_sparse_acc": float(k_acc),
                "k_sparse_auroc": float(k_auroc),
                "absorption": float(absorption),
            })

            per_pair_aurocs[f"{word_a}-{word_b}"].append(float(resid_auroc))

        sae_result["mean_pair_absorption"] = float(np.mean(pair_absorptions))
        print(f"  mean pair absorption: {sae_result['mean_pair_absorption']:.4f}")

        all_sae_results.append(sae_result)

        # Cleanup
        del sae
        gc.collect()
        torch.cuda.empty_cache()
        step += 2

    total_time = time.time() - total_start

    # Save results
    output = {
        "task_id": TASK_ID,
        "sae_results": all_sae_results,
        "hierarchy_probe_aurocs": per_hierarchy_aurocs,
        "pair_probe_aurocs": per_pair_aurocs,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / "gpt2_replication_results.json"
    results_path.write_text(json.dumps(output, indent=2))
    print(f"\nSaved metrics to {results_path}")

    # Update gpu_progress
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    if gpu_progress_path.exists():
        gpu_progress = json.loads(gpu_progress_path.read_text())
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID in gpu_progress.get("running", {}):
        del gpu_progress["running"][TASK_ID]
    if TASK_ID not in gpu_progress.get("completed", []):
        gpu_progress.setdefault("completed", []).append(TASK_ID)
    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 20,
        "actual_min": round(total_time / 60),
        "start_time": start_time_iso,
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "saes": len(SAE_SPECS),
            "hierarchies": len(HIERARCHIES),
            "pairs": len(PAIRS),
            "model": MODEL_NAME,
            "device": DEVICE,
            "gpu_model": "local",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

    # Summary markdown
    summary_md = RESULTS_DIR / "gpt2_replication_summary.md"
    md_lines = [
        f"# GPT-2 Replication Summary",
        f"",
        f"**Task:** {TASK_ID}",
        f"**Total Time:** {total_time/60:.1f} min",
        f"",
        "## Results",
        "",
        "| SAE | Mean Hierarchy Absorption | Mean Pair Absorption |",
        "|-----|---------------------------|----------------------|",
    ]
    for r in all_sae_results:
        md_lines.append(f"| {r['family']} | {r['mean_hierarchy_absorption']:.4f} | {r['mean_pair_absorption']:.4f} |")
    md_lines.append("")
    summary_md.write_text("\n".join(md_lines))

    mark_done(
        status="success",
        summary=f"gpt2_replication completed on {len(all_sae_results)}/{len(SAE_SPECS)} SAEs in {total_time/60:.1f} min",
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failure", summary=str(e))
        raise
