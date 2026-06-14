"""Pilot: Semantic-Hierarchy Probe Pipeline.

Runs custom semantic-hierarchy absorption on 2 SAEs
(MatryoshkaBatchTopK_trainer_0 and TopK_trainer_0)
using 3 WordNet hierarchies and 2 control pairs.

Absorption metric: k-sparse probe accuracy degradation on co-occurrence sentences.
Sentences contain BOTH parent and child concepts, making the classification task
harder and requiring the model to actually process the hierarchical relationship.
"""

import json
import os
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformer_lens import HookedTransformer

# Custom SAE loading (bypass hf_hub_download issues)
from sae_bench.custom_saes.batch_topk_sae import BatchTopKSAE
from sae_bench.custom_saes.topk_sae import TopKSAE

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

SAMPLES_PER_CONCEPT = 100  # pilot sample budget
LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
MODEL_NAME = "EleutherAI/pythia-160m-deduped"

# 3 hierarchies for pilot
HIERARCHIES = [
    {"parent": "building", "children": ["house", "school", "library"]},
    {"parent": "container", "children": ["box", "bag", "cup"]},
    {"parent": "tool", "children": ["fork", "comb", "rake"]},
]

# 2 control pairs for pilot
CONTROL_PAIRS = [
    ["big", "large"],
    ["fast", "quick"],
]

SAE_SPECS = [
    {
        "name": "MatryoshkaBatchTopK_trainer_0",
        "repo_folder": "MatryoshkaBatchTopK_pythia-160m-deduped__0108",
        "cls": "matryoshka",
    },
    {
        "name": "TopK_trainer_0",
        "repo_folder": "TopK_pythia-160m-deduped__0108",
        "cls": "topk",
    },
]

RESULTS_DIR = Path("exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

PID_FILE = Path("exp/results/pilot_semantic_probe.pid")
PROGRESS_FILE = Path("exp/results/pilot_semantic_probe_PROGRESS.json")
DONE_FILE = Path("exp/results/pilot_semantic_probe_DONE")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def write_pid():
    PID_FILE.write_text(str(os.getpid()))


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    PROGRESS_FILE.write_text(
        json.dumps(
            {
                "task_id": "pilot_semantic_probe",
                "epoch": epoch,
                "total_epochs": total_epochs,
                "step": step,
                "total_steps": total_steps,
                "loss": loss,
                "metric": metric or {},
                "updated_at": datetime.now().isoformat(),
            }
        )
    )


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(
        json.dumps(
            {
                "task_id": "pilot_semantic_probe",
                "status": status,
                "summary": summary,
                "final_progress": final_progress,
                "timestamp": datetime.now().isoformat(),
            }
        )
    )


def load_sae_manual(spec):
    base = Path(f"/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-kimi/downloaded_saes/{spec['repo_folder']}/resid_post_layer_8/trainer_0")
    pt_params = torch.load(base / "ae.pt", map_location="cpu")
    config = json.loads((base / "config.json").read_text())
    layer = config["trainer"]["layer"]
    k = config["trainer"]["k"]

    if spec["cls"] == "matryoshka":
        del pt_params["group_sizes"]
        sae = BatchTopKSAE(
            d_in=pt_params["b_dec"].shape[0],
            d_sae=pt_params["b_enc"].shape[0],
            k=k,
            model_name=MODEL_NAME,
            hook_layer=layer,
            device=DEVICE,
            dtype=torch.float32,
        )
        sae.load_state_dict(pt_params)
    elif spec["cls"] == "topk":
        key_mapping = {
            "encoder.weight": "W_enc",
            "decoder.weight": "W_dec",
            "encoder.bias": "b_enc",
            "bias": "b_dec",
            "k": "k",
        }
        if "threshold" in pt_params:
            del pt_params["threshold"]
        renamed = {key_mapping.get(k, k): v for k, v in pt_params.items()}
        renamed["W_enc"] = renamed["W_enc"].T
        renamed["W_dec"] = renamed["W_dec"].T
        sae = TopKSAE(
            d_in=renamed["b_dec"].shape[0],
            d_sae=renamed["b_enc"].shape[0],
            k=k,
            model_name=MODEL_NAME,
            hook_layer=layer,
            device=DEVICE,
            dtype=torch.float32,
        )
        sae.load_state_dict(renamed)
    else:
        raise ValueError(f"Unknown SAE class: {spec['cls']}")

    sae.to(DEVICE, torch.float32)
    if not sae.check_decoder_norms():
        raise ValueError("Decoder vectors are not normalized")
    return sae, k


# ---------------------------------------------------------------------------
# Dataset generation with co-occurrence
# ---------------------------------------------------------------------------

def make_hierarchy_sentences(parent, child, n):
    """Generate sentences where both parent and child co-occur."""
    templates = [
        "The {parent}, which is a {child}, stands nearby.",
        "A {child} is a type of {building}.",
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
    # Fix template 2 which uses {building} instead of {parent}
    templates[1] = "A {child} is a type of {parent}."
    sentences = []
    for i in range(n):
        sentences.append(templates[i % len(templates)].format(parent=parent, child=child))
    return sentences


def make_control_sentences(a, b, n):
    """Generate sentences where both control words co-occur."""
    templates = [
        "The word {a} is similar to {b}.",
        "{a} and {b} mean almost the same thing.",
        "People use {a} when they could say {b}.",
        "The concept of {a} relates to {b}.",
        "{a} is synonymous with {b}.",
        "In this context, {a} and {b} are interchangeable.",
        "The dictionary lists {a} and {b} together.",
        "{a} reminds me of {b}.",
        "Both {a} and {b} describe the same idea.",
        "The teacher said {a} and {b} are equivalent.",
        "Writers often choose {a} over {b}.",
        "{a} has a meaning close to {b}.",
        "The sentence used {a} but meant {b}.",
        "{a} and {b} share a common definition.",
        "In many languages, {a} and {b} translate similarly.",
        "The article discussed {a} and mentioned {b}.",
        "{a} can be replaced by {b} without changing meaning.",
        "The difference between {a} and {b} is subtle.",
        "Most people consider {a} and {b} to be alike.",
        "The book uses {a} while the movie uses {b}.",
    ]
    sentences = []
    for i in range(n):
        sentences.append(templates[i % len(templates)].format(a=a, b=b))
    return sentences


def generate_hierarchy_dataset(hierarchy, samples_per_concept):
    parent = hierarchy["parent"]
    children = hierarchy["children"]
    sentences = []
    labels = []
    # For each child, generate co-occurrence sentences with the parent
    # Label is the CHILD concept (we classify which child is mentioned)
    for child in children:
        sents = make_hierarchy_sentences(parent, child, samples_per_concept)
        sentences.extend(sents)
        labels.extend([child] * len(sents))
    return sentences, labels


def generate_control_dataset(pair, samples_per_concept):
    a, b = pair
    sentences = []
    labels = []
    # Generate sentences with both words co-occurring
    # Label is word A (we classify which word is the 'target')
    sents = make_control_sentences(a, b, samples_per_concept)
    sentences.extend(sents)
    labels.extend([a] * len(sents))
    # Also generate with reversed roles
    sents = make_control_sentences(b, a, samples_per_concept)
    sentences.extend(sents)
    labels.extend([b] * len(sents))
    return sentences, labels


# ---------------------------------------------------------------------------
# Activation extraction
# ---------------------------------------------------------------------------

def get_mean_pooled_activations(model, sentences, batch_size=32):
    """Return mean-pooled residual activations over non-padding positions."""
    all_acts = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)  # [B, L]
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            acts = cache[HOOK_NAME]  # [B, L, d_model]
        mask = (tokens != model.tokenizer.pad_token_id).unsqueeze(-1).float()
        pooled = (acts * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_acts.append(pooled.cpu().float())
    return torch.cat(all_acts, dim=0).numpy()


def get_sae_latents(model, sae, sentences, batch_size=32):
    """Return SAE latent activations (mean-pooled)."""
    all_latents = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=[HOOK_NAME])
            acts = cache[HOOK_NAME]
            latents = sae.encode(acts)  # [B, L, d_sae]
        mask = (tokens != model.tokenizer.pad_token_id).unsqueeze(-1).float()
        pooled = (latents * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        all_latents.append(pooled.cpu().float())
    return torch.cat(all_latents, dim=0).numpy()


def get_k_sparse_latents(sae_latents, k):
    """Zero out all but top-k latents per sample."""
    sparse = np.zeros_like(sae_latents)
    for i in range(sae_latents.shape[0]):
        topk_idx = np.argsort(sae_latents[i])[-k:]
        sparse[i, topk_idx] = sae_latents[i, topk_idx]
    return sparse


# ---------------------------------------------------------------------------
# Probe training
# ---------------------------------------------------------------------------

def train_multiclass_probe(X, y, classes):
    """Train a multi-class logistic regression probe."""
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
# Main
# ---------------------------------------------------------------------------

def main():
    write_pid()
    report_progress(0, 1, metric={"status": "starting"})

    print("Loading model...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype="float32")
    print("Model loaded.")

    results = []
    summary = {
        "sae_results": [],
        "hierarchy_probe_aurocs": {},
        "control_probe_aurocs": {},
    }

    total_steps = len(SAE_SPECS)
    for step_idx, spec in enumerate(SAE_SPECS):
        print(f"\n=== SAE {spec['name']} ({step_idx+1}/{total_steps}) ===")
        report_progress(step_idx, total_steps, step=step_idx, total_steps=total_steps, metric={"sae": spec["name"]})

        sae, k = load_sae_manual(spec)

        sae_result = {
            "sae_name": spec["name"],
            "hierarchies": [],
            "controls": [],
        }

        # --- Hierarchies ---
        hierarchy_absorptions = []
        for h in HIERARCHIES:
            parent = h["parent"]
            children = h["children"]
            sentences, labels = generate_hierarchy_dataset(h, SAMPLES_PER_CONCEPT)
            labels = np.array(labels, dtype=object)

            resid_acts = get_mean_pooled_activations(model, sentences)
            sae_latents = get_sae_latents(model, sae, sentences)
            k_sparse = get_k_sparse_latents(sae_latents, min(10, k))

            _, resid_acc, resid_auroc = train_multiclass_probe(resid_acts, labels, children)
            _, sae_acc, sae_auroc = train_multiclass_probe(sae_latents, labels, children)
            _, k_acc, k_auroc = train_multiclass_probe(k_sparse, labels, children)

            absorption = compute_absorption(resid_acc, sae_acc, k_acc)
            hierarchy_absorptions.append(absorption)

            sae_result["hierarchies"].append(
                {
                    "parent": parent,
                    "children": children,
                    "resid_acc": float(resid_acc),
                    "resid_auroc": float(resid_auroc),
                    "sae_acc": float(sae_acc),
                    "sae_auroc": float(sae_auroc),
                    "k_sparse_acc": float(k_acc),
                    "k_sparse_auroc": float(k_auroc),
                    "absorption": float(absorption),
                }
            )

            key = f"{parent}"
            if key not in summary["hierarchy_probe_aurocs"]:
                summary["hierarchy_probe_aurocs"][key] = []
            summary["hierarchy_probe_aurocs"][key].append(float(resid_auroc))

        # --- Controls ---
        control_absorptions = []
        for pair in CONTROL_PAIRS:
            a, b = pair
            sentences, labels = generate_control_dataset(pair, SAMPLES_PER_CONCEPT)
            labels = np.array(labels, dtype=object)

            resid_acts = get_mean_pooled_activations(model, sentences)
            sae_latents = get_sae_latents(model, sae, sentences)
            k_sparse = get_k_sparse_latents(sae_latents, min(10, k))

            _, resid_acc, resid_auroc = train_multiclass_probe(resid_acts, labels, [a, b])
            _, sae_acc, sae_auroc = train_multiclass_probe(sae_latents, labels, [a, b])
            _, k_acc, k_auroc = train_multiclass_probe(k_sparse, labels, [a, b])

            absorption = compute_absorption(resid_acc, sae_acc, k_acc)
            control_absorptions.append(absorption)

            sae_result["controls"].append(
                {
                    "pair": f"{a}-{b}",
                    "resid_acc": float(resid_acc),
                    "resid_auroc": float(resid_auroc),
                    "sae_acc": float(sae_acc),
                    "sae_auroc": float(sae_auroc),
                    "k_sparse_acc": float(k_acc),
                    "k_sparse_auroc": float(k_auroc),
                    "absorption": float(absorption),
                }
            )

            key = f"{a}-{b}"
            if key not in summary["control_probe_aurocs"]:
                summary["control_probe_aurocs"][key] = []
            summary["control_probe_aurocs"][key].append(float(resid_auroc))

        sae_result["mean_hierarchy_absorption"] = float(np.mean(hierarchy_absorptions))
        sae_result["mean_control_absorption"] = float(np.mean(control_absorptions))
        results.append(sae_result)
        summary["sae_results"].append(sae_result)

        print(f"  mean hierarchy absorption: {sae_result['mean_hierarchy_absorption']:.4f}")
        print(f"  mean control absorption:   {sae_result['mean_control_absorption']:.4f}")

    # -----------------------------------------------------------------------
    # Pass criteria evaluation
    # -----------------------------------------------------------------------
    all_finite = all(
        np.isfinite(r["mean_hierarchy_absorption"]) and np.isfinite(r["mean_control_absorption"])
        for r in results
    )

    all_aurocs_ok = all(
        h["resid_auroc"] > 0.6 for r in results for h in r["hierarchies"]
    )

    matryoshka_sem = next(r["mean_hierarchy_absorption"] for r in results if r["sae_name"].startswith("Matryoshka"))
    topk_sem = next(r["mean_hierarchy_absorption"] for r in results if r["sae_name"].startswith("TopK_"))
    ordering_ok = matryoshka_sem < topk_sem

    specificity_ok = any(
        r["mean_hierarchy_absorption"] > r["mean_control_absorption"]
        for r in results
    )

    passed = all_finite and all_aurocs_ok and ordering_ok and specificity_ok

    summary["pass_criteria"] = {
        "all_finite": all_finite,
        "all_aurocs_ok": all_aurocs_ok,
        "ordering_ok": ordering_ok,
        "specificity_ok": specificity_ok,
        "overall_passed": passed,
    }

    # Save JSON results
    out_json = RESULTS_DIR / "pilot_semantic_probe_results.json"
    out_json.write_text(json.dumps(summary, indent=2))

    # Save pilot_summary.json (machine-readable for orchestrator)
    pilot_summary = {
        "overall_recommendation": "GO" if passed else "REFINE",
        "selected_candidate_id": "cand_a",
        "candidates": [
            {
                "candidate_id": "cand_a",
                "go_no_go": "GO" if passed else "NO_GO",
                "confidence": 0.75 if passed else 0.35,
                "supported_hypotheses": ["H1", "H2"] if passed else [],
                "failed_assumptions": [] if passed else ["pilot ordering or specificity failed"],
                "key_metrics": {
                    "matryoshka_mean_hierarchy_absorption": float(matryoshka_sem),
                    "topk_mean_hierarchy_absorption": float(topk_sem),
                },
                "notes": "Pilot semantic probe pipeline completed successfully."
                if passed
                else "Pilot failed one or more pass criteria.",
            }
        ],
    }
    (RESULTS_DIR / "pilot_summary.json").write_text(json.dumps(pilot_summary, indent=2))

    # Save pilot_summary.md
    md_lines = [
        "# Pilot Summary: Semantic-Hierarchy Probe",
        "",
        f"**Overall:** {'GO' if passed else 'REFINE'}",
        "",
        "## Results",
        "",
        "| SAE | Mean Hierarchy Absorption | Mean Control Absorption |",
        "|-----|---------------------------|-------------------------|",
    ]
    for r in results:
        md_lines.append(
            f"| {r['sae_name']} | {r['mean_hierarchy_absorption']:.4f} | {r['mean_control_absorption']:.4f} |"
        )
    md_lines.extend([
        "",
        "## Pass Criteria",
        "",
        f"- All finite: {all_finite}",
        f"- All probe AUROCs > 0.6: {all_aurocs_ok}",
        f"- Matryoshka < TopK ordering: {ordering_ok}",
        f"- Hierarchy > control for at least 1 SAE: {specificity_ok}",
        "",
    ])
    (RESULTS_DIR / "pilot_summary.md").write_text("\n".join(md_lines))

    print("\nPilot complete. Passed:", passed)
    mark_done(status="success" if passed else "failure", summary=f"Pilot passed={passed}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        mark_done(status="failure", summary=str(e))
        raise
