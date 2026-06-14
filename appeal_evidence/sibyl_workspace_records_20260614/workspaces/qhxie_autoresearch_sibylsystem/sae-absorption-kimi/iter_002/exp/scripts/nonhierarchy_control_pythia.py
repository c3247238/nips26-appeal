#!/usr/bin/env python3
"""Non-Hierarchy Control Absorption on Pythia-160M SAEs.

Runs the custom absorption pipeline on 10 non-hierarchical correlated pairs
for all 8 Pythia-160M SAE configurations. Computes mean control absorption
score per SAE.
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

import sae_bench.custom_saes.run_all_evals_dictionary_learning_saes as dl_saes

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TASK_ID = "nonhierarchy_control_pythia"
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

REPO_ID = "adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"
DOWNLOAD_LOCATION = "/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/downloaded_saes"

LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
SAMPLES_PER_CONCEPT = 100
K_SPARSE = 10

FAMILIES = [
    "BatchTopK", "GatedSAE", "JumpRelu", "MatryoshkaBatchTopK",
    "PAnneal", "Standard", "TopK",
]

SAE_SPECS = []
for family in FAMILIES:
    location = f"{family}_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0"
    SAE_SPECS.append({
        "name": f"{family}_trainer_0",
        "family": family,
        "location": location,
        "is_random": False,
    })

# Random-SAE control: permuted Standard trainer_0
SAE_SPECS.append({
    "name": "Random_trainer_0",
    "family": "Random",
    "location": "Standard_pythia-160m-deduped__0108/resid_post_layer_8/trainer_0",
    "is_random": True,
})

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

    Uses in-place mutation on a deep copy and verifies weights actually changed.
    """
    import copy
    random_sae = copy.deepcopy(base_sae)

    # Store original checksums for verification
    orig_enc_sum = random_sae.W_enc.data.sum().item()
    orig_benc_sum = random_sae.b_enc.data.sum().item()
    orig_dec_sum = random_sae.W_dec.data.sum().item()

    with torch.no_grad():
        # Randomize encoder in-place (more reliable than reassigning Parameter)
        new_enc = torch.randn_like(random_sae.W_enc.data)
        new_enc = new_enc * (random_sae.W_enc.data.std() / new_enc.std().clamp_min(1e-8))
        random_sae.W_enc.data.copy_(new_enc)

        # Randomize encoder bias in-place
        new_b = torch.randn_like(random_sae.b_enc.data)
        new_b = new_b * (random_sae.b_enc.data.std() / new_b.std().clamp_min(1e-8))
        random_sae.b_enc.data.copy_(new_b)

        # Randomize decoder in-place
        new_dec = torch.randn_like(random_sae.W_dec.data)
        new_dec = new_dec * (random_sae.W_dec.data.std() / new_dec.std().clamp_min(1e-8))
        random_sae.W_dec.data.copy_(new_dec)

    # Verify that weights actually changed
    new_enc_sum = random_sae.W_enc.data.sum().item()
    new_benc_sum = random_sae.b_enc.data.sum().item()
    new_dec_sum = random_sae.W_dec.data.sum().item()

    if abs(new_enc_sum - orig_enc_sum) < 1e-6:
        raise RuntimeError("Random SAE encoder weights did not change after randomization!")
    if abs(new_benc_sum - orig_benc_sum) < 1e-6:
        raise RuntimeError("Random SAE encoder bias did not change after randomization!")
    if abs(new_dec_sum - orig_dec_sum) < 1e-6:
        raise RuntimeError("Random SAE decoder weights did not change after randomization!")

    print(f"  Randomization verified: enc changed {abs(new_enc_sum - orig_enc_sum):.2f}, "
          f"b_enc changed {abs(new_benc_sum - orig_benc_sum):.2f}, "
          f"dec changed {abs(new_dec_sum - orig_dec_sum):.2f}")

    return random_sae


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

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
        # Alternate which word fills which slot to balance
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

def get_mean_pooled_activations(model, sentences, batch_size=32):
    all_acts = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            acts = cache[HOOK_NAME]
        mask = (tokens != model.tokenizer.pad_token_id).unsqueeze(-1).float()
        pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_acts.append(pooled.cpu().float())
    return torch.cat(all_acts, dim=0).numpy()


def get_sae_latents(model, sae, sentences, batch_size=32):
    all_latents = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            acts = cache[HOOK_NAME]
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
    report_progress(1, len(SAE_SPECS) + 2, message="Starting nonhierarchy_control_pythia evaluation")

    total_start = time.time()

    print(f"Loading model {MODEL_NAME}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
    print("Model loaded.")

    all_sae_results = []
    per_pair_aurocs = {f"{p['word_a']}-{p['word_b']}": [] for p in PAIRS}

    for idx, spec in enumerate(SAE_SPECS, start=3):
        name = spec["name"]
        family = spec["family"]
        location = spec["location"]
        is_random = spec["is_random"]

        report_progress(idx, len(SAE_SPECS) + 2, message=f"Evaluating {name}")
        print(f"\n=== [{idx-2}/{len(SAE_SPECS)}] {name} ===")

        print(f"Loading SAE {location}...")
        sae = dl_saes.load_dictionary_learning_sae(
            repo_id=REPO_ID,
            location=location,
            model_name=MODEL_NAME,
            device=DEVICE,
            dtype=torch.float32,
            download_location=DOWNLOAD_LOCATION,
        )

        if is_random:
            print("Permuting decoder directions for random control...")
            sae = make_random_sae(sae)

        sae_result = {
            "sae_name": name,
            "family": family,
            "is_random": is_random,
            "pairs": [],
        }

        pair_absorptions = []
        for pair in PAIRS:
            word_a = pair["word_a"]
            word_b = pair["word_b"]
            sentences, labels = generate_pair_dataset(pair, SAMPLES_PER_CONCEPT)
            labels = np.array(labels, dtype=object)

            resid_acts = get_mean_pooled_activations(model, sentences)
            sae_latents = get_sae_latents(model, sae, sentences)
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
        all_sae_results.append(sae_result)

        print(f"  mean pair absorption: {sae_result['mean_pair_absorption']:.4f}")

        # Cleanup
        del sae
        gc.collect()
        torch.cuda.empty_cache()

    total_time = time.time() - total_start

    # Save results
    output = {
        "task_id": TASK_ID,
        "sae_results": all_sae_results,
        "pair_probe_aurocs": per_pair_aurocs,
        "total_time_sec": total_time,
        "timestamp": datetime.now().isoformat(),
    }
    results_path = RESULTS_DIR / "nonhierarchy_control_pythia_results.json"
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
            "pairs": len(PAIRS),
            "model": MODEL_NAME,
            "device": DEVICE,
            "gpu_model": "local",
            "gpu_count": 1,
        },
    }
    gpu_progress_path.write_text(json.dumps(gpu_progress, indent=2))

    # Summary markdown
    summary_md = RESULTS_DIR / "nonhierarchy_control_pythia_summary.md"
    md_lines = [
        f"# Non-Hierarchy Control Absorption (Pythia-160M) Summary",
        f"",
        f"**Task:** {TASK_ID}",
        f"**Total Time:** {total_time/60:.1f} min",
        f"",
        "## Results",
        "",
        "| SAE | Mean Pair Absorption |",
        "|-----|----------------------|",
    ]
    for r in all_sae_results:
        md_lines.append(f"| {r['family']} | {r['mean_pair_absorption']:.4f} |")
    md_lines.append("")
    summary_md.write_text("\n".join(md_lines))

    mark_done(
        status="success",
        summary=f"nonhierarchy_control_pythia completed on {len(all_sae_results)}/{len(SAE_SPECS)} SAEs in {total_time/60:.1f} min",
    )
    print("Done.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failure", summary=str(e))
        raise
