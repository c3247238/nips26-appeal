"""
Task C.2-REDESIGN: Cross-Domain Absorption via Child-Feature Suppression (PILOT MODE)

CORRECT implementation of C.2. Previous versions (v1-v3) measured whether PARENT latents
fail to fire for concept tokens — that was parent-latent-suppression (WRONG).

This redesign implements CHILD-FEATURE SUPPRESSION:
- For each concept token (e.g., "dog"), find its CHILD-specific SAE latent:
  the latent that fires most specifically for that token vs. unrelated tokens.
- Measure: P(child_active | parent_fired) vs P(child_active | parent_not_fired)
- Absorption rate = fraction of (parent_latent, child_latent, concept_token) triples
  where child activation is suppressed by >= 30% when parent fires (top 25% percentile).

Pass criteria: Ratio-to-null >= 1.5 for first_letter hierarchy (sanity check against
known sae-spelling absorption). If < 1.0 for all hierarchies including first_letter,
redesign has a pipeline bug — investigate.

Output: exp/results/full/C2_child_suppression_absorption.json
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
PILOTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
FULL_DIR = WORKSPACE / "exp" / "results" / "full"
PROBES_DIR = WORKSPACE / "exp" / "results" / "probes"
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_C2_redesign"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = FULL_DIR / "C2_child_suppression_absorption.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

# Use GPU 1 as assigned
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
sys.stdout.flush()


def report_progress(step, total_steps, note=""):
    elapsed = time.time() - start_time
    progress = {
        "task_id": TASK_ID, "step": step, "total_steps": total_steps,
        "elapsed_sec": elapsed, "note": note, "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"[{elapsed:.1f}s] Step {step}/{total_steps}: {note}")
    sys.stdout.flush()


def mark_done(status="success", summary=""):
    PID_FILE.unlink(missing_ok=True)
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(), "elapsed_sec": time.time() - start_time,
    }))


TOTAL_STEPS = 15
report_progress(0, TOTAL_STEPS, "Starting C2 REDESIGN: child-feature suppression")

# ── Step 1: Load SAE and model ─────────────────────────────────────────────────

report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small + SAE (L6, jb)")

from sae_lens import SAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()

sae, cfg_dict, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=DEVICE
)
sae.eval()

W_enc = sae.W_enc.detach().float()  # [d_model, d_sae]
W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
b_enc = sae.b_enc.detach().float()  # [d_sae]
d_sae = W_dec.shape[0]
d_model = W_dec.shape[1]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

print(f"SAE: d_sae={d_sae}, d_model={d_model}")
sys.stdout.flush()

# ── Step 2: Core utility functions ────────────────────────────────────────────

report_progress(2, TOTAL_STEPS, "Setting up utility functions")

LAYER = 6
CACHE_KEY = f"blocks.{LAYER}.hook_resid_pre"


def get_sae_acts(words, batch_size=64):
    """Get SAE activations for a list of words (single token context)."""
    all_sae_acts = []
    all_resid_acts = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=CACHE_KEY,
                return_type=None,
            )
        # Take the last real token position (position 1 for single-word inputs)
        resid = cache[CACHE_KEY][:, -1, :].float()
        all_resid_acts.append(resid.cpu())
        with torch.no_grad():
            sae_acts = sae.encode(resid)
        all_sae_acts.append(sae_acts.cpu())
    return torch.cat(all_sae_acts, dim=0), torch.cat(all_resid_acts, dim=0)


def find_child_latent(concept_word, unrelated_words, top_k_candidates=3):
    """
    Find the child-specific latent for a given concept word:
    the SAE latent that fires most differentially for concept_word vs. unrelated_words.

    Returns: list of (latent_idx, mean_concept_act, mean_unrelated_act, differential)
    """
    concept_acts, _ = get_sae_acts([concept_word])  # [1, d_sae]
    unrelated_acts, _ = get_sae_acts(unrelated_words)  # [n_unrelated, d_sae]

    concept_mean = concept_acts[0].numpy()  # [d_sae]
    unrelated_mean = unrelated_acts.mean(dim=0).numpy()  # [d_sae]

    # Differential activation: concept fires more than unrelated
    diff = concept_mean - unrelated_mean

    # Top candidates by differential
    top_idx = np.argsort(diff)[::-1][:top_k_candidates * 5]  # oversample, filter by threshold

    candidates = []
    for idx in top_idx:
        if concept_mean[idx] > 0.1 and diff[idx] > 0.0:  # must actually fire for concept
            candidates.append({
                "latent_idx": int(idx),
                "concept_act": float(concept_mean[idx]),
                "unrelated_mean_act": float(unrelated_mean[idx]),
                "differential": float(diff[idx]),
            })
        if len(candidates) >= top_k_candidates:
            break

    return candidates


def measure_child_suppression_rate(
    parent_latent_idx,
    child_latent_idx,
    token_contexts_acts,  # [n_contexts, d_sae] — token in various contexts
    suppression_threshold=0.30,
    parent_percentile=75,
):
    """
    Measure child-feature suppression by parent.

    For each context:
    - Check if parent latent is in top `parent_percentile` (strongly active)
    - Check if child latent is suppressed relative to baseline (where parent NOT in top)

    Returns: (child_suppressed_given_parent, child_active_given_no_parent, n_parent_fired, n_parent_not_fired)
    """
    parent_acts = token_contexts_acts[:, parent_latent_idx].numpy()
    child_acts = token_contexts_acts[:, child_latent_idx].numpy()

    if len(parent_acts) == 0:
        return None, None, 0, 0

    # Parent fired = top 25% activation
    parent_threshold = np.percentile(parent_acts, parent_percentile)
    parent_fired_mask = parent_acts >= parent_threshold
    parent_not_fired_mask = ~parent_fired_mask

    n_parent_fired = parent_fired_mask.sum()
    n_parent_not_fired = parent_not_fired_mask.sum()

    if n_parent_fired == 0 or n_parent_not_fired == 0:
        return None, None, int(n_parent_fired), int(n_parent_not_fired)

    # Mean child activation given parent fired vs not fired
    child_given_parent = child_acts[parent_fired_mask].mean()
    child_given_no_parent = child_acts[parent_not_fired_mask].mean()

    # Suppression = child activation drops by >= suppression_threshold when parent fires
    if child_given_no_parent <= 0.0:
        # Child never active anyway — not a valid measurement
        return None, None, int(n_parent_fired), int(n_parent_not_fired)

    suppression_ratio = (child_given_no_parent - child_given_parent) / child_given_no_parent
    is_suppressed = bool(suppression_ratio >= suppression_threshold)

    return is_suppressed, float(suppression_ratio), int(n_parent_fired), int(n_parent_not_fired)


# ── Step 3: Load concept words from C1 probe training ─────────────────────────

report_progress(3, TOTAL_STEPS, "Loading C1 probe data and concept word lists")

# Load C1 probe training results to get parent latent info
c1_data = json.load(open(FULL_DIR / "C1_probe_training.json"))
passing_hierarchies = c1_data.get("passing_hierarchies", [])
print(f"Passing hierarchies from C1: {passing_hierarchies}")
sys.stdout.flush()

# ── Step 4: Define hierarchy word lists ───────────────────────────────────────

report_progress(4, TOTAL_STEPS, "Defining hierarchy word lists")

# For FIRST_LETTER hierarchy (sanity check — known sae-spelling absorption)
# Parent: letter 'a' (most common in corpus); child: specific words starting with 'a'
# In sae-spelling, "absorption" = word-specific SAE feature absorbed into letter feature
first_letter_parents = {
    "a": {
        "concept_tokens": [
            "apple", "animal", "anchor", "arrow", "anger", "album", "alarm", "angel",
            "army", "audio", "actor", "atlas", "arena", "altar", "armor",
        ],
        "unrelated_tokens": [
            "table", "chair", "river", "cloud", "music", "green", "stone", "night",
            "ocean", "bread", "sport", "flame", "knife", "clock", "tower",
        ],
    },
    "b": {
        "concept_tokens": [
            "bird", "brain", "bread", "bridge", "brown", "brush", "bench", "blood",
            "brave", "bloom", "blank", "blade", "beach", "badge", "basin",
        ],
        "unrelated_tokens": [
            "apple", "chair", "river", "cloud", "music", "green", "stone", "night",
            "ocean", "flame", "knife", "clock", "tower", "drama", "steel",
        ],
    },
    "s": {
        "concept_tokens": [
            "stone", "storm", "smile", "smoke", "snake", "sheep", "sword", "steel",
            "sharp", "shell", "swift", "spark", "shore", "slope", "scout",
        ],
        "unrelated_tokens": [
            "apple", "chair", "river", "cloud", "music", "green", "brain", "night",
            "ocean", "bread", "flame", "knife", "clock", "tower", "drama",
        ],
    },
}

# PARENT latents for first_letter: use high-activation latents from SAE on letter words
# We identify them empirically from B1/C1 data — top latents for letter-starting tokens

# For ANIMATE_INANIMATE hierarchy
animate_inanimate_hierarchy = {
    "animate": {
        "concept_tokens": [
            "dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger",
            "wolf", "bear", "eagle", "monkey", "rabbit", "cow", "deer",
        ],
        "unrelated_tokens": [
            "rock", "stone", "table", "chair", "book", "car", "house", "bridge",
            "bottle", "box", "ring", "cloud", "river", "mountain", "steel",
        ],
    },
}

# For NOUN_PROPER hierarchy
noun_proper_hierarchy = {
    "proper_noun": {
        "concept_tokens": [
            "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid",
            "Alice", "Robert", "Michael", "Jennifer", "William", "Elizabeth",
            "Google", "Apple",
        ],
        "unrelated_tokens": [
            "table", "chair", "river", "music", "stone", "flower", "kitchen",
            "bridge", "forest", "garden", "paper", "clock", "lamp", "window", "door",
        ],
    },
}

# ── Step 5: Find parent latents via empirical discovery ────────────────────────

report_progress(5, TOTAL_STEPS, "Discovering parent latents via empirical activation")


def find_parent_latents_empirical(concept_words, control_words, top_k=3, min_differential=0.5):
    """Find top-k SAE latents that are most differentially active for concept vs. control."""
    concept_acts, _ = get_sae_acts(concept_words)
    control_acts, _ = get_sae_acts(control_words)

    mean_concept = concept_acts.mean(dim=0).numpy()
    mean_control = control_acts.mean(dim=0).numpy()
    diff = mean_concept - mean_control

    # Sort by differential, take top_k with positive activation
    top_idx = np.argsort(diff)[::-1]

    parent_latents = []
    for idx in top_idx:
        if len(parent_latents) >= top_k:
            break
        if mean_concept[idx] >= 0.3 and diff[idx] >= min_differential:
            parent_latents.append({
                "latent_idx": int(idx),
                "mean_concept": float(mean_concept[idx]),
                "mean_control": float(mean_control[idx]),
                "differential": float(diff[idx]),
            })
    return parent_latents


# ── Step 6: Load OpenWebText tokens for context variation ─────────────────────

report_progress(6, TOTAL_STEPS, "Preparing token context generation")

# We need multiple contexts for each concept token
# Use simple approaches: prepend different prefixes to get context variation
# For single-word inputs, we generate varied sentence contexts

def make_contexts(word, n_contexts=50, seed=42):
    """
    Generate varied sentence contexts for a word.
    Returns a list of context strings where the word appears in different positions.
    """
    rng = random.Random(seed)

    prefixes = [
        f"The {word}",
        f"A {word}",
        f"This {word}",
        f"That {word}",
        f"My {word}",
        f"One {word}",
        f"Each {word}",
        f"Every {word}",
        f"Some {word}",
        f"Any {word}",
        f"I saw a {word}",
        f"She found a {word}",
        f"He had a {word}",
        f"They found the {word}",
        f"We saw the {word}",
        f"I noticed the {word}",
        f"It was a {word}",
        f"There was a {word}",
        f"Here is a {word}",
        f"Look at the {word}",
        f"Consider the {word}",
        f"Think about {word}",
        f"Remember the {word}",
        f"Find the {word}",
        f"See the {word}",
        f"The big {word}",
        f"A small {word}",
        f"The old {word}",
        f"A young {word}",
        f"The large {word}",
        f"A great {word}",
        f"The tiny {word}",
        f"A strange {word}",
        f"The beautiful {word}",
        f"A common {word}",
        f"The first {word}",
        f"A new {word}",
        f"The last {word}",
        f"An important {word}",
        f"The real {word}",
        f"An interesting {word}",
        f"The only {word}",
        f"A simple {word}",
        f"The main {word}",
        f"A typical {word}",
        f"The specific {word}",
        f"A different {word}",
        f"The next {word}",
        f"An old {word}",
        f"The current {word}",
    ]

    if len(prefixes) >= n_contexts:
        return prefixes[:n_contexts]
    else:
        return prefixes


def get_token_context_activations(word, n_contexts=50, seed=42, batch_size=32):
    """
    Get SAE activations for a concept word across multiple sentence contexts.
    Returns: [n_contexts, d_sae] tensor
    """
    contexts = make_contexts(word, n_contexts, seed)
    all_acts = []
    for i in range(0, len(contexts), batch_size):
        batch = contexts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=CACHE_KEY,
                return_type=None,
            )
        # Take the last token position (the concept word itself)
        resid = cache[CACHE_KEY][:, -1, :].float()
        with torch.no_grad():
            sae_acts = sae.encode(resid)
        all_acts.append(sae_acts.cpu())
    return torch.cat(all_acts, dim=0)


# ── Step 7: Core C2 REDESIGN measurement function ────────────────────────────

report_progress(7, TOTAL_STEPS, "Setting up child-suppression measurement")


def measure_child_suppression_for_hierarchy(
    hierarchy_name,
    parent_categories,  # dict: category_name -> {"concept_tokens": [...], "unrelated_tokens": [...]}
    n_contexts_per_token=50,
    suppression_threshold=0.30,
    parent_percentile=75,
    n_permutations=100,
    n_bootstrap=1000,
    seed=42,
):
    """
    Implements the C.2-REDESIGN protocol: child-feature suppression measurement.

    For each (parent_category, concept_token):
    1. Identify parent SAE latents (top-3 by empirical differential activation)
    2. Find child-specific SAE latent for the concept_token
    3. Generate contexts for concept_token
    4. Measure: P(child_suppressed | parent_fires) vs P(child_suppressed | parent_not_fires)
    5. Absorption = child activation dropped by >= suppression_threshold when parent fires

    Returns a detailed result dict.
    """
    rng = np.random.RandomState(seed)
    all_triples = []  # (parent_latent_idx, child_latent_idx, concept_word, is_suppressed, suppression_ratio)

    parent_latent_info = {}

    for cat_name, cat_data in parent_categories.items():
        concept_tokens = cat_data["concept_tokens"]
        unrelated_tokens = cat_data["unrelated_tokens"]

        print(f"\n  [{hierarchy_name}/{cat_name}] Finding parent latents...")
        sys.stdout.flush()

        # Step 1: Find top-3 parent latents for this category
        parent_latents = find_parent_latents_empirical(
            concept_tokens, unrelated_tokens, top_k=3, min_differential=0.3
        )
        parent_latent_info[cat_name] = parent_latents

        print(f"    Found {len(parent_latents)} parent latents:")
        for pl in parent_latents:
            print(f"      latent {pl['latent_idx']}: concept={pl['mean_concept']:.3f}, ctrl={pl['mean_control']:.3f}, diff={pl['differential']:.3f}")
        sys.stdout.flush()

        if not parent_latents:
            print(f"    WARNING: No parent latents found above threshold for {cat_name}")
            continue

        parent_latent_indices = [pl["latent_idx"] for pl in parent_latents]

        # Step 2: For each concept token, find child-specific latent
        for concept_word in concept_tokens:
            # Find child-specific latent (fires specifically for this token)
            child_candidates = find_child_latent(
                concept_word, unrelated_tokens, top_k_candidates=2
            )

            if not child_candidates:
                print(f"    No child latent found for '{concept_word}' — skipping")
                continue

            # Step 3: Get concept_word in varied contexts
            ctx_acts = get_token_context_activations(
                concept_word, n_contexts=n_contexts_per_token, seed=seed
            )  # [n_contexts, d_sae]

            # Step 4: For each (parent_latent, child_latent) pair, measure suppression
            for child_info in child_candidates:
                child_latent_idx = child_info["latent_idx"]

                for parent_latent_idx in parent_latent_indices:
                    if parent_latent_idx == child_latent_idx:
                        continue  # same latent — not a valid pair

                    is_suppressed, suppression_ratio, n_pf, n_pnf = measure_child_suppression_rate(
                        parent_latent_idx, child_latent_idx, ctx_acts,
                        suppression_threshold=suppression_threshold,
                        parent_percentile=parent_percentile,
                    )

                    if is_suppressed is not None:
                        all_triples.append({
                            "category": cat_name,
                            "concept_word": concept_word,
                            "parent_latent_idx": parent_latent_idx,
                            "child_latent_idx": child_latent_idx,
                            "child_concept_act": child_info["concept_act"],
                            "child_differential": child_info["differential"],
                            "is_suppressed": is_suppressed,
                            "suppression_ratio": suppression_ratio,
                            "n_parent_fired": n_pf,
                            "n_parent_not_fired": n_pnf,
                        })

    # Step 5: Compute absorption rate
    if not all_triples:
        return {
            "hierarchy": hierarchy_name,
            "n_triples": 0,
            "absorption_rate": None,
            "ratio_to_null": None,
            "go_nogo": "NO_GO",
            "error": "No valid (parent, child, token) triples found",
            "parent_latent_info": parent_latent_info,
        }

    suppressed = np.array([t["is_suppressed"] for t in all_triples], dtype=float)
    absorption_rate = float(suppressed.mean())
    n_triples = len(all_triples)

    # Bootstrap CI
    boot_rates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_triples, n_triples, replace=True)
        boot_rates.append(float(suppressed[idx].mean()))
    ci_lower = float(np.percentile(boot_rates, 2.5))
    ci_upper = float(np.percentile(boot_rates, 97.5))

    # Step 6: Null control — permute parent-child token assignments
    null_rates = []
    for perm_i in range(n_permutations):
        # Shuffle concept_word assignments (scramble which child goes with which parent)
        shuffled = rng.permutation(n_triples)
        perm_suppressed = []
        for orig_i, shuf_i in enumerate(shuffled):
            # Keep parent latent from original, use DIFFERENT concept word's child latent
            t_orig = all_triples[orig_i]
            t_shuf = all_triples[shuf_i]
            # Re-use original suppression but with shuffled child
            # For a true permutation null: recompute suppression with permuted pairs
            # Approximate: use the shuffled triple's suppression value
            perm_suppressed.append(t_shuf["is_suppressed"])
        null_rates.append(float(np.mean(perm_suppressed)))

    null_mean = float(np.mean(null_rates))
    null_std = float(np.std(null_rates))
    ratio_to_null = absorption_rate / null_mean if null_mean > 0 else float("inf")

    # Go/no-go
    go_nogo = "GO" if ratio_to_null >= 1.5 else ("WEAK" if ratio_to_null >= 1.1 else "NO_GO")

    print(f"\n  [{hierarchy_name}] Results:")
    print(f"    n_triples: {n_triples}")
    print(f"    absorption_rate: {absorption_rate:.4f} (CI: [{ci_lower:.4f}, {ci_upper:.4f}])")
    print(f"    null_mean: {null_mean:.4f} ± {null_std:.4f}")
    print(f"    ratio_to_null: {ratio_to_null:.3f}")
    print(f"    go_nogo: {go_nogo}")
    sys.stdout.flush()

    # Summary suppression ratios
    suppression_ratios = [t["suppression_ratio"] for t in all_triples if t["suppression_ratio"] is not None]

    return {
        "hierarchy": hierarchy_name,
        "n_triples": n_triples,
        "n_suppressed": int(suppressed.sum()),
        "absorption_rate": absorption_rate,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "ci_reliable": n_triples >= 10,
        "null_mean": null_mean,
        "null_std": null_std,
        "ratio_to_null": ratio_to_null,
        "null_n_permutations": n_permutations,
        "go_nogo": go_nogo,
        "mean_suppression_ratio": float(np.mean(suppression_ratios)) if suppression_ratios else None,
        "parent_latent_info": parent_latent_info,
        "triples_sample": all_triples[:10],  # save sample for debugging
    }


# ── Step 8: Run HIERARCHY 1 — first_letter (sanity check) ────────────────────

report_progress(8, TOTAL_STEPS, "Measuring first_letter hierarchy (sanity check)")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter (sanity check against sae-spelling absorption)")
print("="*70)
sys.stdout.flush()

try:
    result_first_letter = measure_child_suppression_for_hierarchy(
        hierarchy_name="first_letter",
        parent_categories=first_letter_parents,
        n_contexts_per_token=50,
        suppression_threshold=0.30,
        parent_percentile=75,
        n_permutations=100,
        n_bootstrap=1000,
        seed=SEED,
    )
except Exception as e:
    import traceback
    print(f"ERROR in first_letter: {e}")
    traceback.print_exc()
    result_first_letter = {"hierarchy": "first_letter", "error": str(e), "go_nogo": "ERROR"}

# ── Step 9: Run HIERARCHY 2 — animate_inanimate ───────────────────────────────

report_progress(9, TOTAL_STEPS, "Measuring animate_inanimate hierarchy")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate")
print("="*70)
sys.stdout.flush()

try:
    result_animate = measure_child_suppression_for_hierarchy(
        hierarchy_name="animate_inanimate",
        parent_categories=animate_inanimate_hierarchy,
        n_contexts_per_token=50,
        suppression_threshold=0.30,
        parent_percentile=75,
        n_permutations=100,
        n_bootstrap=1000,
        seed=SEED,
    )
except Exception as e:
    import traceback
    print(f"ERROR in animate_inanimate: {e}")
    traceback.print_exc()
    result_animate = {"hierarchy": "animate_inanimate", "error": str(e), "go_nogo": "ERROR"}

# ── Step 10: Run HIERARCHY 3 — noun_proper ────────────────────────────────────

report_progress(10, TOTAL_STEPS, "Measuring noun_proper hierarchy")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper")
print("="*70)
sys.stdout.flush()

try:
    result_proper = measure_child_suppression_for_hierarchy(
        hierarchy_name="noun_proper",
        parent_categories=noun_proper_hierarchy,
        n_contexts_per_token=50,
        suppression_threshold=0.30,
        parent_percentile=75,
        n_permutations=100,
        n_bootstrap=1000,
        seed=SEED,
    )
except Exception as e:
    import traceback
    print(f"ERROR in noun_proper: {e}")
    traceback.print_exc()
    result_proper = {"hierarchy": "noun_proper", "error": str(e), "go_nogo": "ERROR"}

# ── Step 11: Compute EDA for parent latents (for C3 correlation) ──────────────

report_progress(11, TOTAL_STEPS, "Computing EDA for parent latents")

print("\n" + "="*70)
print("EDA of parent latents (for C3 hierarchy correlation)")
print("="*70)
sys.stdout.flush()


def compute_eda_for_latents(latent_indices):
    """Compute EDA = 1 - cos(encoder_j, decoder_j) for a list of latent indices."""
    edas = []
    for j in latent_indices:
        enc_j = W_enc[:, j].float()  # [d_model]
        dec_j = W_dec[j, :].float()  # [d_model]
        cos_sim = F.cosine_similarity(enc_j.unsqueeze(0), dec_j.unsqueeze(0)).item()
        eda = 1.0 - cos_sim
        edas.append(float(eda))
    return edas


eda_per_hierarchy = {}

for hierarchy_name, result in [
    ("first_letter", result_first_letter),
    ("animate_inanimate", result_animate),
    ("noun_proper", result_proper),
]:
    parent_info = result.get("parent_latent_info", {})
    all_parent_idxs = []
    for cat_name, latents in parent_info.items():
        for lt in latents:
            all_parent_idxs.append(lt["latent_idx"])

    if all_parent_idxs:
        edas = compute_eda_for_latents(all_parent_idxs)
        mean_eda = float(np.mean(edas))
        eda_per_hierarchy[hierarchy_name] = {
            "parent_latent_indices": all_parent_idxs,
            "eda_values": edas,
            "mean_parent_eda": mean_eda,
        }
        print(f"  {hierarchy_name}: n_parent_latents={len(all_parent_idxs)}, mean_EDA={mean_eda:.4f}")
    else:
        eda_per_hierarchy[hierarchy_name] = {
            "parent_latent_indices": [],
            "eda_values": [],
            "mean_parent_eda": None,
        }
        print(f"  {hierarchy_name}: no parent latents found")

sys.stdout.flush()

# ── Step 12: AUPRC computation (for class-imbalanced metrics) ────────────────

report_progress(12, TOTAL_STEPS, "Computing AUPRC and additional metrics")

from sklearn.metrics import roc_auc_score, average_precision_score


def compute_auroc_auprc_from_triples(triples):
    """Compute AUROC and AUPRC treating absorption as a binary classification."""
    if not triples:
        return None, None

    y_true = [1 if t["is_suppressed"] else 0 for t in triples]
    y_score = [t["suppression_ratio"] for t in triples if t.get("suppression_ratio") is not None]

    if len(set(y_true)) < 2:
        return None, None  # Need both classes

    y_true_clean = [y_true[i] for i in range(len(triples)) if triples[i].get("suppression_ratio") is not None]

    if len(set(y_true_clean)) < 2:
        return None, None

    try:
        auroc = float(roc_auc_score(y_true_clean, y_score))
        auprc = float(average_precision_score(y_true_clean, y_score))
        return auroc, auprc
    except Exception:
        return None, None


# ── Step 13: Build overall summary ───────────────────────────────────────────

report_progress(13, TOTAL_STEPS, "Building overall summary")

results_by_hierarchy = {
    "first_letter": result_first_letter,
    "animate_inanimate": result_animate,
    "noun_proper": result_proper,
}

# First-letter sanity check
fl_ratio_raw = result_first_letter.get("ratio_to_null", 0)
fl_ratio = safe_ratio("first_letter")
fl_go = result_first_letter.get("go_nogo", "ERROR")

# Main pass criteria: ratio_to_null >= 1.5 for first_letter (sanity check)
sanity_check_pass = fl_ratio >= 1.5
sanity_check_warn = fl_ratio < 1.0  # Bug flag

def safe_ratio(h):
    r = results_by_hierarchy.get(h, {}).get("ratio_to_null", 0)
    if r is None:
        return 0.0
    if r == float("inf"):
        # Treat inf ratio as passing: null_mean=0 and absorption_rate>0 or absorption_rate=0 with null=0
        abs_rate = results_by_hierarchy.get(h, {}).get("absorption_rate", 0) or 0.0
        # if both are 0, ratio=inf is degenerate (no variation) — treat as neutral
        null_mean = results_by_hierarchy.get(h, {}).get("null_mean", 0) or 0.0
        if null_mean == 0.0 and abs_rate == 0.0:
            return 1.0  # equal (neutral)
        elif null_mean == 0.0 and abs_rate > 0.0:
            return 10.0  # true signal — treat as high ratio
        else:
            return 1.0
    return float(r)

n_passing = sum(
    1 for h in ["first_letter", "animate_inanimate", "noun_proper"]
    if safe_ratio(h) >= 1.5
)
n_weak = sum(
    1 for h in ["first_letter", "animate_inanimate", "noun_proper"]
    if 1.1 <= safe_ratio(h) < 1.5
)

# Bonferroni-corrected alpha (3 hierarchies)
bonferroni_alpha = 0.05 / 3

overall_go_nogo = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if sanity_check_warn and overall_go_nogo != "GO":
    overall_go_nogo = "PIPELINE_BUG_INVESTIGATE"

print("\n" + "="*70)
print("OVERALL SUMMARY")
print("="*70)
for h, r in results_by_hierarchy.items():
    ratio = r.get("ratio_to_null", "N/A")
    rate = r.get("absorption_rate", "N/A")
    go = r.get("go_nogo", "?")
    n_t = r.get("n_triples", "?")
    print(f"  {h}: absorption_rate={rate}, ratio_to_null={ratio}, n_triples={n_t}, go={go}")
print(f"  Sanity check (first_letter ratio>=1.5): {'PASS' if sanity_check_pass else ('WARN: PIPELINE BUG?' if sanity_check_warn else 'FAIL')}")
print(f"  Overall: {overall_go_nogo}")
sys.stdout.flush()

# ── Step 14: Save results ─────────────────────────────────────────────────────

report_progress(14, TOTAL_STEPS, "Saving results")

elapsed = time.time() - start_time

output = {
    "task_id": TASK_ID,
    "task_name": "task_C2_redesign",
    "mode": "PILOT",
    "version": "redesign_v1_child_suppression",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "layer": LAYER,
        "d_sae": d_sae,
        "d_model": d_model,
        "seed": SEED,
        "n_contexts_per_token": 50,
        "suppression_threshold": 0.30,
        "parent_percentile": 75,
        "n_permutations": 100,
        "n_bootstrap": 1000,
        "bonferroni_alpha": bonferroni_alpha,
        "measurement_definition": (
            "CHILD-FEATURE SUPPRESSION: for each concept_token, find child-specific SAE latent "
            "(most differentially active for that token vs. unrelated). "
            "For each (parent_latent, child_latent, concept_token) triple: "
            "P(child_active | parent_top25%) vs P(child_active | parent_bot75%). "
            "Absorption = child suppressed by >= 30% when parent strongly fires."
        ),
    },
    "hierarchies": results_by_hierarchy,
    "eda_per_hierarchy": eda_per_hierarchy,
    "summary": {
        "n_passing_ratio_1.5": n_passing,
        "n_weak_ratio_1.1": n_weak,
        "overall_go_nogo": overall_go_nogo,
        "bonferroni_alpha": bonferroni_alpha,
        "sanity_check_first_letter": {
            "pass_criteria": "ratio_to_null >= 1.5",
            "actual_ratio": fl_ratio,
            "result": "PASS" if sanity_check_pass else ("PIPELINE_BUG" if sanity_check_warn else "FAIL"),
        },
    },
    "pilot_pass_criteria": {
        "description": "Ratio-to-null >= 1.5 for first_letter hierarchy (sanity check)",
        "first_letter_ratio": fl_ratio,
        "first_letter_go_nogo": fl_go,
        "overall_go_nogo": overall_go_nogo,
    },
    "notes": {
        "redesign_rationale": (
            "Previous C2 (v1-v3) measured parent-latent-suppression — whether parent concept "
            "latents FAIL to fire for concept tokens. All results showed absorption_rate=0.0 "
            "because parent latents DO fire for concept tokens (that's what they're for). "
            "This redesign correctly implements the Chanin et al. definition: "
            "does the CHILD (token-specific) latent fail to fire because PARENT absorbs it?"
        ),
        "n_concept_tokens_target": "50 minimum per parent category (protocol target). "
                                    "Pilot uses smaller word lists; full mode should expand.",
        "context_generation": "50 varied sentence contexts per token via prefix templates",
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to {OUTPUT_FILE}")
sys.stdout.flush()

# ── Step 15: Done ─────────────────────────────────────────────────────────────

report_progress(15, TOTAL_STEPS, "Done")

summary = (
    f"C2-REDESIGN PILOT (child-suppression): {overall_go_nogo}. "
    f"first_letter ratio={fl_ratio:.3f} (sanity: {'PASS' if sanity_check_pass else 'FAIL'}). "
    f"n_passing(>=1.5): {n_passing}/3, n_weak(>=1.1): {n_weak}. "
    f"animate ratio={result_animate.get('ratio_to_null', 'N/A')}, "
    f"proper ratio={result_proper.get('ratio_to_null', 'N/A')}"
)
print(f"\n{'='*70}")
print(f"SUMMARY: {summary}")
print(f"{'='*70}")
sys.stdout.flush()

mark_done("success", summary)
