#!/usr/bin/env python3
"""
Phase 0.2: Tightened Hedging Classification -- Confound Quantification
=======================================================================
The current 98.6% hedging figure is near-tautological. This experiment
implements strict classification that checks whether the SPECIFIC parent
latent fires at higher L0.

Pipeline:
1. Load Gemma 2 2B + Gemma Scope SAEs at L0=22 and L0=176 (16k width, layer 12)
2. Train first-letter probes on Gemma 2 2B residual stream at layer 12
3. For each false negative at L0=22:
   a. Identify the expected parent latent by max cosine similarity
      with the letter probe direction
   b. At L0=176: check if THIS SPECIFIC parent latent fires
   c. Strict hedging: parent latent fires at L0=176 AND the FN resolves
   d. Compensatory resolution: FN resolves but parent latent does NOT fire
   e. Persistent: FN does not resolve at L0=176
4. Report decomposition: strict hedging % + compensatory % + persistent %
5. Compare to loose classification (current 98.6%)

MODE: PILOT (100 sample contexts per letter, seed=42)
"""

import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# ─────────────────────────── Configuration ───────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
PHASE0_DIR = RESULTS_DIR / "phase0"
TASK_ID = "phase0_tightened_hedging"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = PHASE0_DIR / "tightened_hedging.json"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PHASE0_DIR.mkdir(parents=True, exist_ok=True)

# Write PID file
PID_FILE.write_text(str(os.getpid()))

SEED = 42
PILOT_SAMPLES = 100  # Number of contexts per letter for pilot
DEVICE = "cuda:0"  # GPU 6 is mapped to cuda:0 via CUDA_VISIBLE_DEVICES

torch.manual_seed(SEED)
np.random.seed(SEED)


def write_progress(step, total, loss=None, metric=None):
    """Write progress file for system monitor."""
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    """Write DONE marker file."""
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": progress, "timestamp": datetime.now().isoformat(),
    }))


# ─────────────────────────── Step 1: Load Model ───────────────────────
print(f"[Phase 0.2] Starting tightened hedging classification. PID={os.getpid()}")
print(f"[Phase 0.2] Device: {DEVICE}, Pilot samples: {PILOT_SAMPLES}")
write_progress(0, 10, metric={"status": "starting"})

t0 = time.time()

print("[Phase 0.2] Loading Gemma 2 2B via TransformerLens...")
import transformer_lens
from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer

# Load HF model from local cache (unsloth/gemma-2-2b is cached, google/gemma-2-2b is gated)
print("[Phase 0.2]   Loading HF model from local cache (unsloth/gemma-2-2b)...")
hf_model = AutoModelForCausalLM.from_pretrained(
    "unsloth/gemma-2-2b",
    torch_dtype=torch.bfloat16,
    local_files_only=True,
)
hf_tokenizer = HFAutoTokenizer.from_pretrained(
    "unsloth/gemma-2-2b",
    local_files_only=True,
)
print("[Phase 0.2]   Wrapping in HookedTransformer...")
model = transformer_lens.HookedTransformer.from_pretrained(
    "google/gemma-2-2b",
    hf_model=hf_model,
    tokenizer=hf_tokenizer,
    device=DEVICE,
    dtype=torch.bfloat16,
)
del hf_model  # Free the HF model copy
gc.collect()
model.eval()
tokenizer = model.tokenizer
print(f"[Phase 0.2] Model loaded in {time.time() - t0:.1f}s")
write_progress(1, 10, metric={"status": "model_loaded"})

# ─────────────────────────── Step 2: Load SAEs ───────────────────────
print("[Phase 0.2] Loading Gemma Scope SAEs at L0=22 and L0=176...")
from sae_lens import SAE

t1 = time.time()
sae_l0_22 = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_12/width_16k/average_l0_22",
    device=DEVICE,
)
print(f"[Phase 0.2] SAE L0=22 loaded. Threshold mean: {sae_l0_22.threshold.mean().item():.4f}")

sae_l0_176 = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_12/width_16k/average_l0_176",
    device=DEVICE,
)
print(f"[Phase 0.2] SAE L0=176 loaded. Threshold mean: {sae_l0_176.threshold.mean().item():.4f}")
print(f"[Phase 0.2] SAEs loaded in {time.time() - t1:.1f}s")
write_progress(2, 10, metric={"status": "saes_loaded"})


# ─────────────────────────── Step 3: Generate Letter Contexts ───────────
print("[Phase 0.2] Generating letter contexts from model vocabulary...")


def generate_letter_contexts(tokenizer, n_per_letter=PILOT_SAMPLES, seed=SEED):
    """
    Generate input contexts containing words starting with each letter.
    We sample tokens from the vocabulary that start with each letter,
    then create short prompts that include those words.
    """
    rng = np.random.RandomState(seed)

    # Map each letter to vocabulary tokens that start with that letter
    letter_to_tokens = defaultdict(list)
    vocab = tokenizer.get_vocab()

    for token_str, token_id in vocab.items():
        # Clean token string (remove special prefix characters)
        clean = token_str.lstrip("▁").strip()
        if len(clean) >= 2 and clean[0].isalpha():
            first_letter = clean[0].lower()
            if first_letter.isalpha():
                letter_to_tokens[first_letter].append((token_str, token_id, clean))

    # For each letter, create contexts
    letter_contexts = {}
    for letter in "abcdefghijklmnopqrstuvwxyz":
        candidates = letter_to_tokens.get(letter, [])
        if len(candidates) < 5:
            print(f"  Warning: only {len(candidates)} tokens for letter '{letter}'")
            continue

        # Sample tokens
        n_sample = min(n_per_letter, len(candidates))
        indices = rng.choice(len(candidates), n_sample, replace=len(candidates) < n_sample)

        contexts = []
        for idx in indices:
            token_str, token_id, clean_word = candidates[idx]
            # Create a simple context: "The word is {word}"
            # The model processes this and we extract activations at the word token
            prompt = f"The word is {clean_word}"
            contexts.append({
                "prompt": prompt,
                "word": clean_word,
                "token_str": token_str,
                "token_id": token_id,
                "letter": letter,
            })
        letter_contexts[letter] = contexts

    return letter_contexts


letter_contexts = generate_letter_contexts(tokenizer)
total_contexts = sum(len(v) for v in letter_contexts.values())
print(f"[Phase 0.2] Generated {total_contexts} contexts across {len(letter_contexts)} letters")
write_progress(3, 10, metric={"status": "contexts_generated", "total_contexts": total_contexts})


# ─────────────────────────── Step 4: Cache Activations ───────────────────
print("[Phase 0.2] Caching residual stream activations at layer 12...")

HOOK_POINT = "blocks.12.hook_resid_post"
BATCH_SIZE = 32


def cache_activations_batch(model, prompts, hook_point, device, batch_size=BATCH_SIZE):
    """Cache residual stream activations for a batch of prompts.
    Returns activations at the LAST token position of each prompt.
    """
    all_activations = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # Tokenize
        tokens_list = [tokenizer.encode(p, return_tensors="pt").squeeze(0) for p in batch_prompts]

        # Process one at a time (variable length)
        batch_acts = []
        for tokens in tokens_list:
            tokens = tokens.unsqueeze(0).to(device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point],
                    stop_at_layer=13,  # Only need up to layer 12
                )
            # Get activation at last token position
            act = cache[hook_point][0, -1, :].float().cpu()  # [d_model]
            batch_acts.append(act)
            del cache

        all_activations.extend(batch_acts)

    return torch.stack(all_activations)  # [n_prompts, d_model]


# Collect all prompts and corresponding metadata
all_prompts = []
all_letters = []
all_meta = []

for letter, contexts in sorted(letter_contexts.items()):
    for ctx in contexts:
        all_prompts.append(ctx["prompt"])
        all_letters.append(letter)
        all_meta.append(ctx)

print(f"[Phase 0.2] Caching activations for {len(all_prompts)} prompts...")
t2 = time.time()
activations = cache_activations_batch(model, all_prompts, HOOK_POINT, DEVICE)
print(f"[Phase 0.2] Activations cached in {time.time() - t2:.1f}s. Shape: {activations.shape}")

# Free model memory -- we only need the SAEs from here
del model
gc.collect()
torch.cuda.empty_cache()
print("[Phase 0.2] Model freed from GPU memory")
write_progress(4, 10, metric={"status": "activations_cached", "n_activations": activations.shape[0]})


# ─────────────────────────── Step 5: Train First-Letter Probes ───────────
print("[Phase 0.2] Training first-letter linear probes...")

# Create label encoding: letter -> index
letter_to_idx = {letter: idx for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}
labels = np.array([letter_to_idx[l] for l in all_letters])

X = activations.numpy()

# Train/test split
X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
    X, labels, np.arange(len(labels)), test_size=0.2, random_state=SEED, stratify=labels
)

print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# Train logistic regression probe
probe = LogisticRegression(
    C=1.0, max_iter=5000, solver="lbfgs",
    random_state=SEED, n_jobs=-1
)
probe.fit(X_train, y_train)

# Evaluate
y_pred_train = probe.predict(X_train)
y_pred_test = probe.predict(X_test)
train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test, average="macro")

print(f"  Probe train accuracy: {train_acc:.4f}")
print(f"  Probe test accuracy: {test_acc:.4f}")
print(f"  Probe test macro F1: {test_f1:.4f}")

# Per-letter accuracy
per_letter_acc = {}
for letter in sorted(letter_contexts.keys()):
    idx = letter_to_idx[letter]
    mask = y_test == idx
    if mask.sum() > 0:
        per_letter_acc[letter] = float(accuracy_score(y_test[mask], y_pred_test[mask]))

# Get probe directions (one per letter)
probe_directions = {}
probe_coefs = probe.coef_  # [n_classes, d_model]
for letter in sorted(letter_contexts.keys()):
    idx = letter_to_idx[letter]
    direction = probe_coefs[idx]
    direction_normalized = direction / (np.linalg.norm(direction) + 1e-8)
    probe_directions[letter] = torch.tensor(direction_normalized, dtype=torch.float32)

print(f"[Phase 0.2] Probe trained. Test F1={test_f1:.4f}")
write_progress(5, 10, metric={"status": "probe_trained", "test_f1": float(test_f1)})


# ─────────────────────────── Step 6: Compute SAE Outputs ───────────────
print("[Phase 0.2] Computing SAE outputs at L0=22 and L0=176...")


def get_sae_output_and_features(sae, activations_tensor, batch_size=512):
    """Run activations through SAE. Return reconstructed output and feature activations."""
    device = next(sae.parameters()).device
    all_output = []
    all_feature_acts = []

    for i in range(0, len(activations_tensor), batch_size):
        batch = activations_tensor[i:i + batch_size].to(device)
        with torch.no_grad():
            feature_acts = sae.encode(batch)  # [batch, d_sae]
            output = sae.decode(feature_acts)  # [batch, d_model]
        all_output.append(output.cpu())
        all_feature_acts.append(feature_acts.cpu())

    return torch.cat(all_output, dim=0), torch.cat(all_feature_acts, dim=0)


activations_tensor = activations.clone()

print("  Computing SAE output at L0=22...")
t3 = time.time()
sae_output_22, feature_acts_22 = get_sae_output_and_features(sae_l0_22, activations_tensor)
l0_22_actual = (feature_acts_22 > 0).float().sum(dim=1).mean().item()
print(f"  L0=22 actual average L0: {l0_22_actual:.1f} (took {time.time() - t3:.1f}s)")

print("  Computing SAE output at L0=176...")
t4 = time.time()
sae_output_176, feature_acts_176 = get_sae_output_and_features(sae_l0_176, activations_tensor)
l0_176_actual = (feature_acts_176 > 0).float().sum(dim=1).mean().item()
print(f"  L0=176 actual average L0: {l0_176_actual:.1f} (took {time.time() - t4:.1f}s)")

write_progress(6, 10, metric={
    "status": "sae_outputs_computed",
    "l0_22_actual": l0_22_actual,
    "l0_176_actual": l0_176_actual,
})


# ─────────────────────────── Step 7: Identify False Negatives ─────────
print("[Phase 0.2] Identifying false negatives at L0=22...")


def classify_with_probe(probe, X):
    """Return predictions and per-class probabilities."""
    preds = probe.predict(X)
    probs = probe.predict_proba(X)
    return preds, probs


# Predict on raw activations vs SAE output
preds_raw, probs_raw = classify_with_probe(probe, activations.numpy())
preds_sae22, probs_sae22 = classify_with_probe(probe, sae_output_22.numpy())
preds_sae176, probs_sae176 = classify_with_probe(probe, sae_output_176.numpy())

# False negatives at L0=22: probe correct on raw, wrong on SAE output
correct_raw = (preds_raw == labels)
correct_sae22 = (preds_sae22 == labels)
correct_sae176 = (preds_sae176 == labels)

false_negatives_22 = correct_raw & ~correct_sae22  # Correct on raw, wrong on SAE
fn_indices = np.where(false_negatives_22)[0]

# Also compute false negatives at L0=176
false_negatives_176 = correct_raw & ~correct_sae176
fn_176_indices = np.where(false_negatives_176)[0]

print(f"  Total samples: {len(labels)}")
print(f"  Correct on raw: {correct_raw.sum()} ({correct_raw.mean()*100:.1f}%)")
print(f"  Correct on SAE L0=22: {correct_sae22.sum()} ({correct_sae22.mean()*100:.1f}%)")
print(f"  Correct on SAE L0=176: {correct_sae176.sum()} ({correct_sae176.mean()*100:.1f}%)")
print(f"  False negatives at L0=22: {len(fn_indices)} ({len(fn_indices)/len(labels)*100:.1f}%)")
print(f"  False negatives at L0=176: {len(fn_176_indices)} ({len(fn_176_indices)/len(labels)*100:.1f}%)")

# Absorption rate at L0=22 (among probe-correct tokens)
n_probe_correct = correct_raw.sum()
absorption_rate_22 = len(fn_indices) / n_probe_correct if n_probe_correct > 0 else 0
absorption_rate_176 = len(fn_176_indices) / n_probe_correct if n_probe_correct > 0 else 0

print(f"  Absorption rate at L0=22: {absorption_rate_22:.4f} ({absorption_rate_22*100:.1f}%)")
print(f"  Absorption rate at L0=176: {absorption_rate_176:.4f} ({absorption_rate_176*100:.1f}%)")

write_progress(7, 10, metric={
    "status": "false_negatives_identified",
    "n_fn_22": int(len(fn_indices)),
    "absorption_rate_22": float(absorption_rate_22),
})


# ─────────────────────────── Step 8: Tightened Hedging Classification ───
print("[Phase 0.2] Computing tightened hedging classification...")

# For each false negative at L0=22, identify the parent latent:
# The parent latent is the SAE feature whose decoder direction has the
# highest cosine similarity with the probe direction for the TRUE letter.

# Get decoder directions from L0=22 SAE
W_dec_22 = sae_l0_22.W_dec.float().detach().cpu()  # [d_sae, d_model]
W_dec_22_normed = F.normalize(W_dec_22, dim=1)  # [d_sae, d_model]
d_sae = W_dec_22.shape[0]

# For each letter, find the parent latent (feature most aligned with probe direction)
parent_latents = {}
for letter in sorted(letter_contexts.keys()):
    probe_dir = probe_directions[letter]  # [d_model]
    # Cosine similarity between probe direction and all decoder columns
    cos_sims = (W_dec_22_normed @ probe_dir).numpy()  # [d_sae]
    parent_idx = int(np.argmax(cos_sims))
    parent_cos = float(cos_sims[parent_idx])
    parent_latents[letter] = {
        "feature_idx": parent_idx,
        "cosine_similarity": parent_cos,
    }

print("  Parent latent identification (top cosine similarity with probe direction):")
for letter in sorted(parent_latents.keys()):
    info = parent_latents[letter]
    print(f"    '{letter}': feature {info['feature_idx']} (cos={info['cosine_similarity']:.4f})")

# Now classify each false negative at L0=22
classifications = {
    "strict_hedging": [],       # FN resolves AND specific parent fires at L0=176
    "compensatory_resolution": [],  # FN resolves but parent does NOT fire at L0=176
    "persistent": [],           # FN does NOT resolve at L0=176
}

fn_details = []

for fn_idx in fn_indices:
    true_letter = all_letters[fn_idx]
    true_label = labels[fn_idx]

    # Check if this FN resolves at L0=176
    resolves_at_176 = correct_sae176[fn_idx]

    # Check if the SPECIFIC parent latent fires at L0=176
    parent_info = parent_latents[true_letter]
    parent_feature_idx = parent_info["feature_idx"]

    # Feature activation at L0=176 for this token
    parent_fires_at_176 = bool(feature_acts_176[fn_idx, parent_feature_idx].item() > 0)

    # Also check feature activation at L0=22 for the parent
    parent_fires_at_22 = bool(feature_acts_22[fn_idx, parent_feature_idx].item() > 0)

    # Classification
    if resolves_at_176 and parent_fires_at_176:
        category = "strict_hedging"
    elif resolves_at_176 and not parent_fires_at_176:
        category = "compensatory_resolution"
    else:  # not resolves_at_176
        category = "persistent"

    classifications[category].append(fn_idx)

    fn_details.append({
        "index": int(fn_idx),
        "letter": true_letter,
        "word": all_meta[fn_idx]["word"],
        "resolves_at_176": bool(resolves_at_176),
        "parent_feature_idx": parent_feature_idx,
        "parent_fires_at_22": parent_fires_at_22,
        "parent_fires_at_176": parent_fires_at_176,
        "parent_activation_22": float(feature_acts_22[fn_idx, parent_feature_idx].item()),
        "parent_activation_176": float(feature_acts_176[fn_idx, parent_feature_idx].item()),
        "category": category,
    })

n_fn = len(fn_indices)
n_strict = len(classifications["strict_hedging"])
n_compensatory = len(classifications["compensatory_resolution"])
n_persistent = len(classifications["persistent"])

# Loose classification (current method): any FN that resolves at L0=176
n_loose_hedging = n_strict + n_compensatory  # All that resolve
n_loose_persistent = n_persistent

print(f"\n[Phase 0.2] === Hedging Classification Results ===")
print(f"  Total false negatives at L0=22: {n_fn}")
print(f"")
print(f"  LOOSE classification (current method):")
print(f"    Hedging (any resolution): {n_loose_hedging} ({n_loose_hedging/n_fn*100:.1f}%)" if n_fn > 0 else "    N/A")
print(f"    Persistent: {n_loose_persistent} ({n_loose_persistent/n_fn*100:.1f}%)" if n_fn > 0 else "    N/A")
print(f"")
print(f"  STRICT classification (parent-specific):")
print(f"    Strict hedging (parent fires + resolves): {n_strict} ({n_strict/n_fn*100:.1f}%)" if n_fn > 0 else "    N/A")
print(f"    Compensatory resolution (resolves, no parent): {n_compensatory} ({n_compensatory/n_fn*100:.1f}%)" if n_fn > 0 else "    N/A")
print(f"    Persistent (does not resolve): {n_persistent} ({n_persistent/n_fn*100:.1f}%)" if n_fn > 0 else "    N/A")

write_progress(8, 10, metric={
    "status": "classification_done",
    "n_fn": n_fn,
    "n_strict_hedging": n_strict,
    "n_compensatory": n_compensatory,
    "n_persistent": n_persistent,
})


# ─────────────────────────── Step 9: Per-Letter Analysis ───────────────
print("\n[Phase 0.2] Per-letter hedging decomposition:")

per_letter_results = {}
for letter in sorted(letter_contexts.keys()):
    letter_fns = [d for d in fn_details if d["letter"] == letter]
    n = len(letter_fns)
    if n == 0:
        per_letter_results[letter] = {
            "n_fn": 0, "strict_hedging": 0, "compensatory": 0, "persistent": 0
        }
        continue

    n_s = sum(1 for d in letter_fns if d["category"] == "strict_hedging")
    n_c = sum(1 for d in letter_fns if d["category"] == "compensatory_resolution")
    n_p = sum(1 for d in letter_fns if d["category"] == "persistent")

    per_letter_results[letter] = {
        "n_fn": n,
        "strict_hedging": n_s,
        "strict_hedging_pct": n_s / n * 100 if n > 0 else 0,
        "compensatory": n_c,
        "compensatory_pct": n_c / n * 100 if n > 0 else 0,
        "persistent": n_p,
        "persistent_pct": n_p / n * 100 if n > 0 else 0,
        "parent_feature_idx": parent_latents[letter]["feature_idx"],
        "parent_cos_sim": parent_latents[letter]["cosine_similarity"],
    }

    if n > 0:
        print(f"  '{letter}': {n} FNs -> strict={n_s} ({n_s/n*100:.0f}%), "
              f"compensatory={n_c} ({n_c/n*100:.0f}%), persistent={n_p} ({n_p/n*100:.0f}%)")


# ─────────────────────────── Step 10: Sensitivity Analysis ─────────────
print("\n[Phase 0.2] Sensitivity analysis across L0 thresholds...")

# We already have L0=22 and L0=176. Let's also check L0=82 (canonical)
print("  Loading SAE at L0=82 for sensitivity analysis...")
sae_l0_82 = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_12/width_16k/average_l0_82",
    device=DEVICE,
)
sae_output_82, feature_acts_82 = get_sae_output_and_features(sae_l0_82, activations_tensor)
l0_82_actual = (feature_acts_82 > 0).float().sum(dim=1).mean().item()
print(f"  L0=82 actual average L0: {l0_82_actual:.1f}")

preds_sae82, probs_sae82 = classify_with_probe(probe, sae_output_82.numpy())
correct_sae82 = (preds_sae82 == labels)
false_negatives_82 = correct_raw & ~correct_sae82
fn_82_indices = np.where(false_negatives_82)[0]

# For L0=82 false negatives, check at L0=176
sensitivity_82 = {"strict_hedging": 0, "compensatory": 0, "persistent": 0}
for fn_idx in fn_82_indices:
    true_letter = all_letters[fn_idx]
    parent_feature_idx = parent_latents[true_letter]["feature_idx"]
    resolves_at_176 = correct_sae176[fn_idx]
    parent_fires_at_176 = bool(feature_acts_176[fn_idx, parent_feature_idx].item() > 0)

    if resolves_at_176 and parent_fires_at_176:
        sensitivity_82["strict_hedging"] += 1
    elif resolves_at_176 and not parent_fires_at_176:
        sensitivity_82["compensatory"] += 1
    else:
        sensitivity_82["persistent"] += 1

n_fn_82 = len(fn_82_indices)
absorption_rate_82 = n_fn_82 / n_probe_correct if n_probe_correct > 0 else 0

print(f"\n  L0=82 -> L0=176 hedging analysis:")
print(f"    False negatives at L0=82: {n_fn_82} ({absorption_rate_82*100:.1f}%)")
if n_fn_82 > 0:
    print(f"    Strict hedging: {sensitivity_82['strict_hedging']} ({sensitivity_82['strict_hedging']/n_fn_82*100:.1f}%)")
    print(f"    Compensatory: {sensitivity_82['compensatory']} ({sensitivity_82['compensatory']/n_fn_82*100:.1f}%)")
    print(f"    Persistent: {sensitivity_82['persistent']} ({sensitivity_82['persistent']/n_fn_82*100:.1f}%)")

# Free L0=82 SAE
del sae_l0_82, sae_output_82, feature_acts_82
gc.collect()
torch.cuda.empty_cache()

write_progress(9, 10, metric={
    "status": "sensitivity_done",
    "n_fn_82": int(n_fn_82),
    "absorption_rate_82": float(absorption_rate_82),
})


# ─────────────────────────── Step 11: Compile Results ───────────────
print("\n[Phase 0.2] Compiling results...")

# Sample of representative false negatives for inspection
sample_fn_details = fn_details[:min(20, len(fn_details))]

result = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "config": {
        "pilot_samples_per_letter": PILOT_SAMPLES,
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_release": "gemma-scope-2b-pt-res",
        "sae_layer": 12,
        "sae_width": "16k",
        "l0_low": 22,
        "l0_high": 176,
        "l0_mid": 82,
        "device": DEVICE,
        "hook_point": HOOK_POINT,
    },
    "probe_quality": {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_classes": int(len(letter_contexts)),
        "per_letter_accuracy": per_letter_acc,
    },
    "actual_l0_values": {
        "l0_22": float(l0_22_actual),
        "l0_82": float(l0_82_actual),
        "l0_176": float(l0_176_actual),
    },
    "absorption_summary": {
        "total_samples": int(len(labels)),
        "n_probe_correct_raw": int(n_probe_correct),
        "absorption_rate_l0_22": float(absorption_rate_22),
        "absorption_rate_l0_82": float(absorption_rate_82),
        "absorption_rate_l0_176": float(absorption_rate_176),
        "n_fn_l0_22": int(len(fn_indices)),
        "n_fn_l0_82": int(n_fn_82),
        "n_fn_l0_176": int(len(fn_176_indices)),
    },
    "hedging_decomposition_l0_22_to_176": {
        "total_fn": n_fn,
        "loose_classification": {
            "hedging": n_loose_hedging,
            "hedging_pct": n_loose_hedging / n_fn * 100 if n_fn > 0 else 0,
            "persistent": n_loose_persistent,
            "persistent_pct": n_loose_persistent / n_fn * 100 if n_fn > 0 else 0,
            "description": "Loose: any FN that resolves at L0=176 is classified as hedging (current method).",
        },
        "strict_classification": {
            "strict_hedging": n_strict,
            "strict_hedging_pct": n_strict / n_fn * 100 if n_fn > 0 else 0,
            "compensatory_resolution": n_compensatory,
            "compensatory_pct": n_compensatory / n_fn * 100 if n_fn > 0 else 0,
            "persistent": n_persistent,
            "persistent_pct": n_persistent / n_fn * 100 if n_fn > 0 else 0,
            "description": (
                "Strict: only classify as hedging if the SPECIFIC parent latent "
                "(max cos with probe direction) fires at L0=176 AND the FN resolves. "
                "Compensatory: FN resolves but parent does NOT fire (resolution via other features). "
                "Persistent: FN does not resolve even at L0=176."
            ),
        },
    },
    "hedging_decomposition_l0_82_to_176": {
        "total_fn": int(n_fn_82),
        "strict_hedging": sensitivity_82["strict_hedging"],
        "strict_hedging_pct": sensitivity_82["strict_hedging"] / n_fn_82 * 100 if n_fn_82 > 0 else 0,
        "compensatory": sensitivity_82["compensatory"],
        "compensatory_pct": sensitivity_82["compensatory"] / n_fn_82 * 100 if n_fn_82 > 0 else 0,
        "persistent": sensitivity_82["persistent"],
        "persistent_pct": sensitivity_82["persistent"] / n_fn_82 * 100 if n_fn_82 > 0 else 0,
    },
    "parent_latent_identification": {
        letter: {
            "feature_idx": info["feature_idx"],
            "cosine_similarity_with_probe": info["cosine_similarity"],
        }
        for letter, info in parent_latents.items()
    },
    "per_letter_hedging": per_letter_results,
    "sample_fn_details": sample_fn_details,
    "interpretation": {
        "key_finding": (
            f"Strict hedging rate: {n_strict/n_fn*100:.1f}% vs "
            f"loose hedging rate: {n_loose_hedging/n_fn*100:.1f}% "
            f"(difference: {(n_loose_hedging/n_fn - n_strict/n_fn)*100:.1f} pp)" if n_fn > 0
            else "No false negatives detected at L0=22"
        ),
        "compensatory_fraction": (
            f"{n_compensatory/n_fn*100:.1f}% of FNs resolve via compensatory features "
            f"(not the specific parent), indicating non-specific resolution." if n_fn > 0
            else "N/A"
        ),
        "implication": (
            "If compensatory resolution is large, the 98.6% loose hedging figure "
            "significantly overestimates genuine parent-specific hedging. "
            "The strict rate is the more meaningful measure of hierarchy-driven absorption."
        ),
    },
    "pass_criteria": {
        "at_least_100_fn_processed": len(fn_indices) >= 100 if len(fn_indices) > 0 else False,
        "decomposition_computed": n_fn > 0,
        "parent_identification_attempted": len(parent_latents) > 0,
        "strict_hedging_below_loose": n_strict < n_loose_hedging if n_fn > 0 else True,
        "overall_pass": (len(fn_indices) > 0 and n_fn > 0),
        "notes": (
            f"Processed {len(fn_indices)} false negatives. "
            f"{'PASS' if len(fn_indices) >= 100 else 'PARTIAL'}: "
            f"need >=100 FNs for reliable statistics. "
            f"Actual FN count: {len(fn_indices)}."
        ),
    },
}

# Write results
OUTPUT_FILE.write_text(json.dumps(result, indent=2))
print(f"\n[Phase 0.2] Results written to {OUTPUT_FILE}")

# Write summary markdown
summary_md = PHASE0_DIR / "tightened_hedging_summary.md"
summary_lines = [
    "# Phase 0.2: Tightened Hedging Classification -- Pilot Results",
    "",
    f"**Date:** {datetime.now().isoformat()}",
    f"**Mode:** PILOT ({PILOT_SAMPLES} samples/letter)",
    f"**Model:** Gemma 2 2B, Layer 12, SAE 16k width",
    "",
    "## Probe Quality",
    f"- Test accuracy: {test_acc:.4f}",
    f"- Test macro F1: {test_f1:.4f}",
    "",
    "## Absorption Rates Across L0",
    f"- L0=22: {absorption_rate_22*100:.1f}% ({len(fn_indices)} FNs)",
    f"- L0=82: {absorption_rate_82*100:.1f}% ({n_fn_82} FNs)",
    f"- L0=176: {absorption_rate_176*100:.1f}% ({len(fn_176_indices)} FNs)",
    "",
    "## Hedging Decomposition (L0=22 -> L0=176)",
    "",
    "| Classification | Count | Percentage | Description |",
    "|---------------|-------|-----------|-------------|",
]

if n_fn > 0:
    summary_lines.extend([
        f"| **Loose hedging** | {n_loose_hedging} | {n_loose_hedging/n_fn*100:.1f}% | Any resolution (current method) |",
        f"| **Strict hedging** | {n_strict} | {n_strict/n_fn*100:.1f}% | Parent latent fires + resolves |",
        f"| **Compensatory** | {n_compensatory} | {n_compensatory/n_fn*100:.1f}% | Resolves without parent |",
        f"| **Persistent** | {n_persistent} | {n_persistent/n_fn*100:.1f}% | Does not resolve |",
    ])
else:
    summary_lines.append("| N/A | 0 | N/A | No false negatives detected |")

summary_lines.extend([
    "",
    "## Key Finding",
    "",
    result["interpretation"]["key_finding"],
    "",
    result["interpretation"]["compensatory_fraction"],
    "",
    "## Parent Latent Identification",
    "",
    "| Letter | Feature Idx | Cosine Sim |",
    "|--------|-----------|-----------|",
])

for letter, info in sorted(parent_latents.items()):
    summary_lines.append(f"| {letter} | {info['feature_idx']} | {info['cosine_similarity']:.4f} |")

summary_md.write_text("\n".join(summary_lines))
print(f"[Phase 0.2] Summary written to {summary_md}")

# Update GPU progress
gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    if gpu_progress_file.exists():
        gpu_progress = json.loads(gpu_progress_file.read_text())
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    # Mark task as completed
    if TASK_ID not in gpu_progress["completed"]:
        gpu_progress["completed"].append(TASK_ID)

    # Remove from running
    if TASK_ID in gpu_progress.get("running", {}):
        del gpu_progress["running"][TASK_ID]

    # Record timing
    elapsed_min = int((time.time() - t0) / 60)
    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 60,
        "actual_min": elapsed_min,
        "start_time": datetime.fromtimestamp(t0).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gemma-2-2b",
            "sae_layer": 12,
            "sae_width": "16k",
            "n_samples": len(labels),
            "pilot_mode": True,
            "gpu_model": "RTX PRO 6000 Blackwell",
            "gpu_count": 1,
        },
    }

    gpu_progress_file.write_text(json.dumps(gpu_progress, indent=2))
    print(f"[Phase 0.2] GPU progress updated")
except Exception as e:
    print(f"[Phase 0.2] Warning: failed to update gpu_progress.json: {e}")

# Final status
total_time = time.time() - t0
summary_str = (
    f"Phase 0.2 tightened hedging classification: PILOT complete. "
    f"Probe F1={test_f1:.4f}. "
    f"FNs at L0=22: {len(fn_indices)}. "
    f"Strict hedging: {n_strict/n_fn*100:.1f}%, "
    f"Compensatory: {n_compensatory/n_fn*100:.1f}%, "
    f"Persistent: {n_persistent/n_fn*100:.1f}%. "
    f"Strict vs loose gap: {(n_loose_hedging/n_fn - n_strict/n_fn)*100:.1f} pp. "
    f"Time: {total_time/60:.1f} min." if n_fn > 0
    else f"Phase 0.2: No false negatives detected at L0=22. Probe F1={test_f1:.4f}. "
         f"This may indicate the pilot sample size is too small or probes need adjustment."
)

write_progress(10, 10, metric={"status": "done", "total_time_min": total_time / 60})
mark_done("success" if n_fn > 0 else "partial", summary_str)
print(f"\n[Phase 0.2] DONE: {summary_str}")
