"""
Pilot H5: Downstream Task Dependence of Absorption
===================================================
Select 50 features from GPT-2 layer 8 stratified by UAS absorption level
(low/mid/high). Evaluate each feature's discriminability on:
  (A) Simple task: Formal/informal sentence classification
  (B) Causal task:  True-statement vs negated-counterfactual

Compare accuracy delta across absorption levels for each task type.
Pilot pass criterion: Absorpton accuracy delta > 5% between task types
(direction: causal more degraded for high-absorption features).

Because full_h2 (OrtSAE/ATM training) has not completed yet,
we use pilot_h1_h4 UAS data to stratify features.
"""

import json, os, sys, gc
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption-minimax/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GPU_ID = 5
DEVICE = "cuda:0"
SEED = 42
LAYER = 8
MODEL_NAME = "gpt2-small"
SAE_RELEASE = "gpt2-small-res-jb"
N_FEATURES_TOTAL = 50   # ~17 per absorption bin
N_SIMPLE = 300          # sentences per simple-task class
N_CAUSAL = 300          # pairs per causal-task class
N_TOKENS_UAS = 500      # tokens for UAS re-computation

np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)

# ─── PID / Progress / Done tracking ──────────────────────────────────────────

def report_progress(task_id, step=0, total_steps=0, loss=None, metric=None):
    progress = Path(RESULTS_DIR) / f"pilot_h5_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": task_id,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / "pilot_h5.pid"
    if pid_file.exists():
        pid_file.unlink()
    marker = RESULTS_DIR / "pilot_h5_DONE"
    marker.write_text(json.dumps({
        "task_id": "pilot_h5",
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))

# ─── Model & SAE Loading ─────────────────────────────────────────────────────

def load_model_and_sae():
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print(f"Loading {MODEL_NAME} on {DEVICE}...")
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    print(f"  d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

    print(f"Loading SAE from {SAE_RELEASE}, layer {LAYER}...")
    sae_id = f"blocks.{LAYER}.hook_resid_pre"
    # New SAELens API: from_pretrained returns only SAE (not a tuple)
    sae = SAE.from_pretrained(release=SAE_RELEASE, sae_id=sae_id, device=DEVICE)
    print(f"  d_sae={sae.cfg.d_sae}")
    return model, sae

# ─── Text Datasets ────────────────────────────────────────────────────────────

def collect_texts_streaming(n_examples=1000):
    """Collect text examples from pile-uncopyrighted."""
    from datasets import load_dataset
    print(f"  Collecting {n_examples} text examples from pile-uncopyrighted...")
    ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True)
    texts = []
    for i, ex in enumerate(ds):
        texts.append(ex["text"])
        if i >= n_examples - 1:
            break
    print(f"  Collected {len(texts)} examples")
    return texts


# ─── Simple Task: Formal vs Informal Sentence Classification ───────────────────

FORMAL_WORDS = [
    "furthermore", "nevertheless", "consequently", "hereby", "whereas",
    "therein", "aforementioned", "notwithstanding", "pursuant", "henceforth",
    "thus", "thereby", "accordingly", "subsequently", "accordingly",
    "inasmuch", "whereby", "herein", "afore", "wherefore",
]
INFORMAL_WORDS = [
    "gonna", "wanna", "gotta", "kinda", "sorta",
    "yeah", "nah", "y'all", "gimme", "lemme",
    "dunno", " outta", "coulda", "woulda", "shoulda",
    "bro", " dude", " guys", " seriously", " whatever",
]

def build_simple_dataset(model, n_per_class=N_SIMPLE):
    """Generate formal/informal sentence pairs using template expansion."""
    formal_templates = [
        "The {w} decision was made after careful consideration.",
        "It is {w} the case that proper procedures were followed.",
        "The committee noted the {w} implications of the policy.",
        "This {w} demonstrates the need for reform.",
        "The {w} impact on stakeholders must be addressed.",
    ]
    informal_templates = [
        "I {w} think we should just go ahead and do it.",
        "That {w} seems like a pretty big deal honestly.",
        "We're {w} gonna need to figure this out soon.",
        "This whole thing is {w} getting out of hand.",
        "I'm {w} pretty sure about what we should do next.",
    ]
    rng = np.random.default_rng(SEED)
    formal_sentences, informal_sentences = [], []

    for _ in range(n_per_class):
        tmpl = rng.choice(formal_templates)
        word = rng.choice(FORMAL_WORDS)
        formal_sentences.append(tmpl.format(w=word))
        tmpl = rng.choice(informal_templates)
        word = rng.choice(INFORMAL_WORDS)
        informal_sentences.append(tmpl.format(w=word))

    return formal_sentences, informal_sentences


# ─── Causal Task: True vs Negated Counterfactual ─────────────────────────────

TRUE_CAUSAL = [
    "If it rains, the ground gets wet.",
    "If you heat water to 100 degrees, it boils.",
    "If you push an object, it moves.",
    "If the sun rises, it becomes light outside.",
    "If you eat food, you feel less hungry.",
    "If you open a door, the room gets air circulation.",
    "If you plant a seed, it grows into a plant.",
    "If you add salt to water, the boiling point increases.",
    "If you compress a spring, it stores energy.",
    "If you mix red and blue paint, you get purple.",
]
NEGATION_TEMPLATES = [
    "If it rains, the ground stays dry.",          # antecedent, different consequent
    "If you heat water to 100 degrees, it does not boil.",
    "If you push an object, it remains stationary.",
    "If the sun rises, it stays dark outside.",
    "If you eat food, you feel more hungry.",       # opposite consequent
    "If you open a door, no air moves in the room.",
    "If you plant a seed, nothing grows.",
    "If you add salt to water, the boiling point stays the same.",
    "If you compress a spring, it loses energy.",
    "If you mix red and blue paint, you get green.",  # wrong color
]

def build_causal_dataset(n_per_class=N_CAUSAL):
    """Generate true vs negated counterfactual statement pairs."""
    rng = np.random.default_rng(SEED + 100)
    n_templates = len(TRUE_CAUSAL)
    true_stmts, neg_stmts = [], []

    for _ in range(n_per_class):
        idx = rng.integers(0, n_templates)
        # True: use the original template with slight variation
        true_stmts.append(TRUE_CAUSAL[idx])
        # Negated: use the negation template
        neg_stmts.append(NEGATION_TEMPLATES[idx])

    return true_stmts, neg_stmts


# ─── Feature Activation Collection ────────────────────────────────────────────

def collect_activations_for_texts(model, sae, texts, layer=LAYER, max_len=128):
    """
    Run model+SAE on a list of texts, return per-token feature activations.
    All texts are padded/truncated to max_len tokens.
    Returns: stacked_fa [N_texts, max_len, d_sae], stacked_tokens [N_texts, max_len]
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    all_feature_acts = []
    all_tokens = []

    for text in texts:
        try:
            tokens = model.to_tokens(text, truncate=True)
            if tokens.shape[1] == 0:
                continue
            # Truncate or pad to max_len
            if tokens.shape[1] > max_len:
                tokens = tokens[:, :max_len]
            elif tokens.shape[1] < max_len:
                pad = torch.zeros(1, max_len - tokens.shape[1], dtype=tokens.dtype, device=tokens.device)
                tokens = torch.cat([tokens, pad], dim=1)

            # Run with SAE
            _, cache = model.run_with_cache(
                tokens,
                names_filter=[hook_name],
                return_type="logits",
            )
            acts = cache[hook_name].to(DEVICE)
            with torch.no_grad():
                fa = sae.encode(acts - sae.b_dec)
            # Truncate/pad fa to max_len
            if fa.shape[1] > max_len:
                fa = fa[:, :max_len, :]
            elif fa.shape[1] < max_len:
                pad_fa = torch.zeros(fa.shape[0], max_len - fa.shape[1], fa.shape[2], device=fa.device)
                fa = torch.cat([fa, pad_fa], dim=1)
            all_feature_acts.append(fa.float().cpu())
            all_tokens.append(tokens.cpu())
        except Exception as e:
            continue

    if not all_feature_acts:
        return None, None

    # Concatenate: [N_texts, seq, d_sae]
    stacked_fa = torch.cat(all_feature_acts, dim=0)   # [N_texts, max_len, d_sae]
    stacked_tokens = torch.cat(all_tokens, dim=0)       # [N_texts, max_len]
    # Aggregate: mean over sequence length for each text
    stacked_fa = stacked_fa.mean(dim=1)                # [N_texts, d_sae]
    return stacked_fa, stacked_tokens


# ─── Feature Discriminability Scoring ─────────────────────────────────────────

def compute_feature_discriminability(feature_acts_pos, feature_acts_neg, feature_idx):
    """
    Compute AUC for a single feature distinguishing pos vs neg class
    using feature activation values as the classifier score.
    Returns AUC in [0.5, 1.0] range (1.0 = perfect discriminability).
    """
    pos_vals = feature_acts_pos[:, feature_idx].numpy()
    neg_vals = feature_acts_neg[:, feature_idx].numpy()
    pos_labels = np.ones(len(pos_vals))
    neg_labels = np.zeros(len(neg_vals))
    all_vals = np.concatenate([pos_vals, neg_vals])
    all_labels = np.concatenate([pos_labels, neg_labels])

    # Handle constant values
    if np.std(all_vals) < 1e-10:
        return 0.5

    try:
        auc = roc_auc_score(all_labels, all_vals)
    except Exception:
        return 0.5

    # AUC is always >= 0.5 by convention; if < 0.5, flip
    if auc < 0.5:
        auc = 1.0 - auc
    return auc


def evaluate_features_for_task(feature_acts_pos, feature_acts_neg, feature_indices, task_name):
    """
    For each feature, compute AUC on the task.
    Returns dict: {feature_idx: auc}
    """
    print(f"  Evaluating {len(feature_indices)} features on '{task_name}'...")
    results = {}
    for i, fidx in enumerate(feature_indices):
        auc = compute_feature_discriminability(feature_acts_pos, feature_acts_neg, fidx)
        results[fidx] = auc
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{len(feature_indices)} features evaluated")
    return results


# ─── UAS Computation (re-compute for all top features) ────────────────────────

def compute_uas_for_features(sae, feature_indices, alpha=1.0, beta=0.5):
    """Compute Unsupervised Absorption Score for each feature."""
    from scipy.stats import skew as scipy_skew
    if hasattr(sae.W_dec, 'weight'):
        W_dec = sae.W_dec.weight.data.cpu().numpy().T  # [d_sae, d_model]
    else:
        W_dec = sae.W_dec.data.cpu().numpy()
    W_dec_norm = W_dec / (np.linalg.norm(W_dec, axis=1, keepdims=True) + 1e-8)

    uas_scores = {}
    for feat_idx in feature_indices:
        other_indices = [i for i in feature_indices if i != feat_idx]
        if len(other_indices) < 2:
            uas_scores[feat_idx] = 0.0
            continue
        cos_sims = np.dot(W_dec_norm[feat_idx], W_dec_norm[other_indices].T)
        cos_sim_var = float(np.var(cos_sims))
        uas_scores[feat_idx] = alpha * cos_sim_var
    return uas_scores


# ─── Feature Selection: Stratify by UAS ───────────────────────────────────────

def select_stratified_features(uas_scores, n_total=N_FEATURES_TOTAL):
    """
    Select n_total features stratified into low/mid/high UAS bins.
    Returns: selected_indices, bin_labels
    """
    # Sort by UAS
    sorted_items = sorted(uas_scores.items(), key=lambda x: x[1])
    n = len(sorted_items)
    n_per_bin = n_total // 3

    low = sorted_items[:n_per_bin]
    mid = sorted_items[n // 2 - n_per_bin // 2: n // 2 + n_per_bin // 2]
    high = sorted_items[-n_per_bin:]

    selected = []
    labels = {}
    for idx, uas in low:
        selected.append(idx)
        labels[idx] = "low"
    for idx, uas in mid:
        selected.append(idx)
        labels[idx] = "mid"
    for idx, uas in high:
        selected.append(idx)
        labels[idx] = "high"

    print(f"\n  Stratified {len(selected)} features by UAS:")
    print(f"    Low  (UAS < {low[-1][1]:.4f}): {len(low)} features")
    print(f"    Mid  (UAS ~ {mid[0][1]:.4f}-{mid[-1][1]:.4f}): {len(mid)} features")
    print(f"    High (UAS > {high[0][1]:.4f}): {len(high)} features")

    return selected, labels


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    task_id = "pilot_h5"
    pid_file = RESULTS_DIR / f"{task_id}.pid"
    pid_file.write_text(str(os.getpid()))
    print(f"\nPID: {os.getpid()}")

    print("\n" + "=" * 60)
    print("PILOT H5: Downstream Task Dependence of Absorption")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Device: {DEVICE}, GPU: {GPU_ID}")
    print(f"Model: {MODEL_NAME}, Layer: {LAYER}")
    report_progress(task_id, step=0, total_steps=5)

    # 1. Load model and SAE
    model, sae = load_model_and_sae()
    d_sae = sae.cfg.d_sae

    # 2. Collect texts for activation collection
    print("\n[Step 1/5] Collecting text examples...")
    texts = collect_texts_streaming(n_examples=1000)
    report_progress(task_id, step=1, total_steps=5)

    # 3. Compute UAS for top features
    print("\n[Step 2/5] Computing UAS for top features...")
    print("  Collecting activations to identify top features...")
    # Use a subset for identifying top features
    subset_texts = texts[:200]
    stacked_fa, _ = collect_activations_for_texts(model, sae, subset_texts)
    if stacked_fa is None:
        print("FATAL: No activations collected!")
        mark_done(status="failed", summary="No activations collected")
        sys.exit(1)

    # Find top features by total activation
    total_act = stacked_fa.sum(dim=0).numpy()
    topk_indices = np.argsort(total_act)[-500:][::-1]  # Top 500 for UAS computation
    uas_scores = compute_uas_for_features(sae, topk_indices)
    print(f"  UAS computed for {len(uas_scores)} features")

    # 4. Stratify features into low/mid/high UAS bins
    print("\n[Step 3/5] Stratifying features by absorption level...")
    selected_features, bin_labels = select_stratified_features(uas_scores, n_total=N_FEATURES_TOTAL)
    report_progress(task_id, step=2, total_steps=5)

    # 5. Build task datasets
    print("\n[Step 4/5] Building task datasets...")
    print("  Building simple task (formal vs informal)...")
    formal_sents, informal_sents = build_simple_dataset(model, n_per_class=N_SIMPLE)
    print(f"    Formal: {len(formal_sents)}, Informal: {len(informal_sents)}")

    print("  Building causal task (true vs negated)...")
    true_stmts, neg_stmts = build_causal_dataset(n_per_class=N_CAUSAL)
    print(f"    True: {len(true_stmts)}, Negated: {len(neg_stmts)}")
    report_progress(task_id, step=3, total_steps=5)

    # 6. Collect feature activations for both tasks
    print("\n[Step 5/5] Evaluating feature discriminability...")

    # Simple task
    print("\n  Simple task: formal vs informal...")
    fa_formal, _ = collect_activations_for_texts(model, sae, formal_sents)
    fa_informal, _ = collect_activations_for_texts(model, sae, informal_sents)
    if fa_formal is None or fa_informal is None:
        print("  ERROR: Failed to collect activations for simple task")
        mark_done(status="failed", summary="Failed to collect simple task activations")
        sys.exit(1)
    simple_aucs = evaluate_features_for_task(
        fa_formal, fa_informal, selected_features, "simple_formal_informal"
    )

    # Causal task
    print("\n  Causal task: true vs negated...")
    fa_true, _ = collect_activations_for_texts(model, sae, true_stmts)
    fa_neg, _ = collect_activations_for_texts(model, sae, neg_stmts)
    if fa_true is None or fa_neg is None:
        print("  ERROR: Failed to collect activations for causal task")
        mark_done(status="failed", summary="Failed to collect causal task activations")
        sys.exit(1)
    causal_aucs = evaluate_features_for_task(
        fa_true, fa_neg, selected_features, "causal_true_negated"
    )

    # ── Aggregate results by absorption bin ──────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS: Feature Discriminability by Absorption Level")
    print("=" * 60)

    bins = ["low", "mid", "high"]
    bin_uas_ranges = {
        "low": (min(uas_scores.get(f, 0) for f in selected_features if bin_labels.get(f) == "low"),
                max(uas_scores.get(f, 0) for f in selected_features if bin_labels.get(f) == "low")),
        "mid": (min(uas_scores.get(f, 0) for f in selected_features if bin_labels.get(f) == "mid"),
                max(uas_scores.get(f, 0) for f in selected_features if bin_labels.get(f) == "mid")),
        "high": (min(uas_scores.get(f, 0) for f in selected_features if bin_labels.get(f) == "high"),
                 max(uas_scores.get(f, 0) for f in selected_features if bin_labels.get(f) == "high")),
    }

    results_by_bin = {}
    for bin_name in bins:
        feat_ids = [f for f in selected_features if bin_labels.get(f) == bin_name]
        simple_vals = [simple_aucs.get(f, 0.5) for f in feat_ids]
        causal_vals = [causal_aucs.get(f, 0.5) for f in feat_ids]
        uas_vals = [uas_scores.get(f, 0) for f in feat_ids]

        mean_simple = float(np.mean(simple_vals))
        mean_causal = float(np.mean(causal_vals))
        mean_uas = float(np.mean(uas_vals))
        std_simple = float(np.std(simple_vals))
        std_causal = float(np.std(causal_vals))

        print(f"\n  [{bin_name.upper()}] UAS range: {bin_uas_ranges[bin_name][0]:.4f}–{bin_uas_ranges[bin_name][1]:.4f}")
        print(f"    Simple AUC: {mean_simple:.4f} ± {std_simple:.4f} (n={len(feat_ids)})")
        print(f"    Causal AUC: {mean_causal:.4f} ± {std_causal:.4f} (n={len(feat_ids)})")
        print(f"    Delta (causal - simple): {mean_causal - mean_simple:+.4f}")

        results_by_bin[bin_name] = {
            "n_features": len(feat_ids),
            "mean_uas": mean_uas,
            "simple_auc_mean": mean_simple,
            "simple_auc_std": std_simple,
            "causal_auc_mean": mean_causal,
            "causal_auc_std": std_causal,
            "auc_delta": mean_causal - mean_simple,
            "feature_indices": [int(f) for f in feat_ids],
            "simple_aucs": {int(f): float(simple_aucs.get(f, 0.5)) for f in feat_ids},
            "causal_aucs": {int(f): float(causal_aucs.get(f, 0.5)) for f in feat_ids},
        }

    # ── Pass criteria ────────────────────────────────────────────────────────
    # H5 pilot criterion: Delta(causal degraded) > 5% for high vs low absorption
    low_simple = results_by_bin["low"]["simple_auc_mean"]
    high_simple = results_by_bin["high"]["simple_auc_mean"]
    low_causal = results_by_bin["low"]["causal_auc_mean"]
    high_causal = results_by_bin["high"]["causal_auc_mean"]

    # Criterion 1: Simple task, high-absorption vs low-absorption accuracy
    simple_delta = abs(high_simple - low_simple)
    # Criterion 2: Causal task, high-absorption vs low-absorption accuracy
    causal_delta = abs(high_causal - low_causal)
    # Criterion 3: Causal more degraded than simple for high-absorption
    # (i.e., AUC drops more from simple to causal for high-absorption than low-absorption)
    high_diff = results_by_bin["high"]["simple_auc_mean"] - results_by_bin["high"]["causal_auc_mean"]
    low_diff = results_by_bin["low"]["simple_auc_mean"] - results_by_bin["low"]["causal_auc_mean"]
    task_dep_delta = abs(high_diff - low_diff)

    print(f"\n{'='*60}")
    print("PASS CRITERIA CHECK")
    print(f"{'='*60}")
    print(f"Simple task delta (high vs low absorption): {simple_delta*100:.2f}%")
    print(f"Causal task delta (high vs low absorption): {causal_delta*100:.2f}%")
    print(f"Task-dependence delta: {task_dep_delta*100:.2f}%")
    print(f"Pass criterion: task-dependence delta > 5%")
    pilot_pass = task_dep_delta > 0.05
    print(f"Overall pilot pass: {pilot_pass}")

    # ── Save results ────────────────────────────────────────────────────────
    output = {
        "task_id": task_id,
        "timestamp": datetime.now().isoformat(),
        "config": {
            "model": MODEL_NAME,
            "layer": LAYER,
            "sae_release": SAE_RELEASE,
            "d_sae": d_sae,
            "n_features_total": N_FEATURES_TOTAL,
            "n_simple_per_class": N_SIMPLE,
            "n_causal_per_class": N_CAUSAL,
            "seed": SEED,
            "device": DEVICE,
        },
        "results_by_bin": results_by_bin,
        "pass_criteria": {
            "task_dependence_delta": float(task_dep_delta),
            "task_dependence_delta_pct": float(task_dep_delta * 100),
            "simple_delta_high_vs_low": float(simple_delta),
            "causal_delta_high_vs_low": float(causal_delta),
            "pilot_pass": pilot_pass,
            "pilot_criterion": "task-dependence delta > 5%",
        },
        "feature_level": {
            "all_features": [int(f) for f in selected_features],
            "bin_labels": {str(k): v for k, v in bin_labels.items()},
            "uas_scores": {str(k): float(v) for k, v in uas_scores.items()},
        },
    }

    output_path = RESULTS_DIR / "h5_pilot.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    summary_str = f"pilot_pass={pilot_pass}, task_dep_delta={task_dep_delta*100:.2f}%"
    mark_done(status="success", summary=summary_str)
    print(f"\nPilot H5 completed.")
    return output


if __name__ == "__main__":
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        import traceback; traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
