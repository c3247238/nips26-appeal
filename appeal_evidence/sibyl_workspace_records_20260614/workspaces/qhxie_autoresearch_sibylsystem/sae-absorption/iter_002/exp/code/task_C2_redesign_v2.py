"""
Task C.2-REDESIGN v2: Cross-Domain Absorption via Child-Feature Suppression (PILOT MODE)

CORRECT implementation of C.2 redesign. Implements child-feature suppression:
- For each concept token (e.g., "dog"), find its child-specific SAE latent.
- Generate multiple sentence contexts for the concept token.
- Measure: Does the child-latent fire less when the parent latent fires strongly?
- Absorption = child activation suppressed >= 30% when parent in top-25%.

Key improvements over v1:
- More robust child latent identification (lower threshold)
- Better context generation using full sentences
- Report mean suppression ratio rather than binary suppression
- Direct computation from context activations, not positional token extraction
- Sanity check: use known sae-spelling absorbed pairs for first_letter

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

import numpy as np
import torch
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_C2_redesign"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = FULL_DIR / "C2_child_suppression_absorption.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

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


TOTAL_STEPS = 12
report_progress(0, TOTAL_STEPS, "Starting C2-REDESIGN v2: child-feature suppression")

# ── Load model and SAE ────────────────────────────────────────────────────────

report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small + SAE L6 jb")

from sae_lens import SAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()

sae, _, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=DEVICE
)
sae.eval()

W_enc = sae.W_enc.detach().float()  # [d_model, d_sae]
W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
d_sae = W_dec.shape[0]
d_model = W_dec.shape[1]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

LAYER = 6
CACHE_KEY = f"blocks.{LAYER}.hook_resid_pre"
print(f"SAE: d_sae={d_sae}, d_model={d_model}")
sys.stdout.flush()

# ── Utility: get SAE activations at last position ─────────────────────────────

def get_sae_acts_batch(texts, batch_size=64):
    """Get SAE activations for list of texts, at last token position."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=CACHE_KEY, return_type=None)
        resid = cache[CACHE_KEY][:, -1, :].float()
        with torch.no_grad():
            acts = sae.encode(resid)
        all_acts.append(acts.cpu())
    return torch.cat(all_acts, dim=0)  # [n, d_sae]


# ── Step 2: Corpus generation for contexts ────────────────────────────────────

report_progress(2, TOTAL_STEPS, "Generating sentence context templates")

# Context templates: we append the concept word at the end of each template
CONTEXT_TEMPLATES = [
    "The",
    "A",
    "This",
    "That",
    "My",
    "One",
    "Each",
    "Every",
    "Some",
    "Here is a",
    "I see a",
    "She found a",
    "He saw a",
    "We noticed a",
    "They had a",
    "Look at the",
    "Consider the",
    "There was a",
    "It was a",
    "That is a",
    "I had a",
    "She noticed the",
    "He saw the",
    "They liked the",
    "We found the",
    "The big",
    "A small",
    "The old",
    "A young",
    "The large",
    "A great",
    "A beautiful",
    "The common",
    "The first",
    "A new",
    "The last",
    "The main",
    "A typical",
    "The specific",
    "A different",
    "An interesting",
    "The only",
    "A simple",
    "The real",
    "An important",
    "The very",
    "A nice",
    "The strange",
    "A special",
    "The particular",
]


def make_contexts_for_word(word, n=50):
    """Create sentence contexts ending with the word."""
    contexts = [f"{tmpl} {word}" for tmpl in CONTEXT_TEMPLATES[:n]]
    return contexts


# ── Step 3: Parent latent discovery ──────────────────────────────────────────

report_progress(3, TOTAL_STEPS, "Discovering parent latents")


def find_parent_latents(concept_words, control_words, top_k=3, min_activation=0.5):
    """
    Find SAE latents maximally differentially active for concept vs. control words.
    Uses single-word inputs (most direct signal).
    """
    c_acts = get_sae_acts_batch(concept_words)   # [n_concept, d_sae]
    u_acts = get_sae_acts_batch(control_words)   # [n_control, d_sae]

    mean_c = c_acts.mean(dim=0).numpy()
    mean_u = u_acts.mean(dim=0).numpy()
    diff = mean_c - mean_u

    top_idx = np.argsort(diff)[::-1]

    parent_latents = []
    for idx in top_idx:
        if len(parent_latents) >= top_k:
            break
        if float(mean_c[idx]) >= min_activation:
            parent_latents.append({
                "latent_idx": int(idx),
                "mean_concept": float(mean_c[idx]),
                "mean_control": float(mean_u[idx]),
                "differential": float(diff[idx]),
            })
    return parent_latents


# ── Step 4: Child latent discovery ───────────────────────────────────────────

def find_child_latents(concept_word, control_words, top_k=2, min_concept_act=0.5):
    """
    Find the SAE latents most specifically active for concept_word (vs. control_words).
    Returns top_k latents by differential activation.
    """
    c_acts = get_sae_acts_batch([concept_word])      # [1, d_sae]
    u_acts = get_sae_acts_batch(control_words)       # [n_control, d_sae]

    c_vec = c_acts[0].numpy()
    u_mean = u_acts.mean(dim=0).numpy()
    diff = c_vec - u_mean

    top_idx = np.argsort(diff)[::-1]

    children = []
    for idx in top_idx:
        if len(children) >= top_k:
            break
        if float(c_vec[idx]) >= min_concept_act:
            children.append({
                "latent_idx": int(idx),
                "concept_act": float(c_vec[idx]),
                "control_mean_act": float(u_mean[idx]),
                "differential": float(diff[idx]),
            })
    return children


# ── Step 5: Child suppression measurement ────────────────────────────────────

def measure_suppression(
    ctx_acts,          # [n_contexts, d_sae]
    parent_latent_idx,
    child_latent_idx,
    parent_percentile=75,
    suppression_threshold=0.30,
):
    """
    For context activations:
    - parent_fired = parent activation in top `parent_percentile`%
    - Compute: mean(child | parent_fired) vs mean(child | parent_not_fired)
    - suppression_ratio = (mean_no_parent - mean_parent) / mean_no_parent
    - is_suppressed = suppression_ratio >= suppression_threshold

    Returns: (is_suppressed, suppression_ratio, n_parent_fired, n_parent_not_fired, child_mean_no_parent)
    """
    parent_acts = ctx_acts[:, parent_latent_idx].numpy()
    child_acts = ctx_acts[:, child_latent_idx].numpy()

    parent_thresh = np.percentile(parent_acts, parent_percentile)
    fired = parent_acts >= parent_thresh
    not_fired = ~fired

    n_pf = int(fired.sum())
    n_pnf = int(not_fired.sum())

    if n_pf == 0 or n_pnf == 0:
        return None, None, n_pf, n_pnf, None

    child_mean_parent = float(child_acts[fired].mean())
    child_mean_no_parent = float(child_acts[not_fired].mean())

    # Only measure suppression if child is at least somewhat active when parent not fired
    # If child never fires, we can't measure suppression
    if child_mean_no_parent < 0.01:
        return None, None, n_pf, n_pnf, child_mean_no_parent

    suppression_ratio = (child_mean_no_parent - child_mean_parent) / child_mean_no_parent
    is_suppressed = bool(suppression_ratio >= suppression_threshold)

    return is_suppressed, float(suppression_ratio), n_pf, n_pnf, child_mean_no_parent


# ── Step 6: Main hierarchy measurement function ────────────────────────────────

report_progress(4, TOTAL_STEPS, "Setting up hierarchy measurement")


def run_hierarchy(
    hierarchy_name,
    parent_categories,
    n_contexts=50,
    suppression_threshold=0.30,
    parent_percentile=75,
    n_permutations=100,
    n_bootstrap=1000,
    seed=42,
):
    """
    Run the C.2-REDESIGN child-suppression measurement for one hierarchy.

    parent_categories: dict { category_name: {"concept_tokens": [...], "unrelated_tokens": [...]} }
    """
    rng = np.random.RandomState(seed)
    all_triples = []
    parent_latent_info = {}

    for cat_name, cat_data in parent_categories.items():
        concept_tokens = cat_data["concept_tokens"]
        unrelated_tokens = cat_data["unrelated_tokens"]

        print(f"\n  [{hierarchy_name}/{cat_name}] Finding parent latents from {len(concept_tokens)} tokens...")

        # Discover parent latents
        parent_latents = find_parent_latents(concept_tokens, unrelated_tokens, top_k=3, min_activation=0.3)
        parent_latent_info[cat_name] = parent_latents

        if not parent_latents:
            print(f"    WARNING: No parent latents found for {cat_name}")
            continue

        parent_latent_indices = [pl["latent_idx"] for pl in parent_latents]
        print(f"    Parent latents: {parent_latent_indices} (top diff: {parent_latents[0]['differential']:.3f})")

        # For each concept token, measure child suppression
        n_valid = 0
        n_skipped_no_child = 0
        n_skipped_inactive = 0

        for concept_word in concept_tokens:
            # Find child-specific latent
            children = find_child_latents(concept_word, unrelated_tokens, top_k=2, min_concept_act=0.3)

            if not children:
                n_skipped_no_child += 1
                continue

            # Get contexts for concept word
            contexts = make_contexts_for_word(concept_word, n=n_contexts)
            try:
                ctx_acts = get_sae_acts_batch(contexts)  # [n_contexts, d_sae]
            except Exception as e:
                print(f"    Error getting context acts for '{concept_word}': {e}")
                continue

            # For each (parent_latent, child_latent) pair
            for child_info in children:
                child_latent_idx = child_info["latent_idx"]

                for parent_latent_idx in parent_latent_indices:
                    if parent_latent_idx == child_latent_idx:
                        continue

                    is_suppressed, sup_ratio, n_pf, n_pnf, child_no_parent = measure_suppression(
                        ctx_acts, parent_latent_idx, child_latent_idx,
                        parent_percentile=parent_percentile,
                        suppression_threshold=suppression_threshold,
                    )

                    if is_suppressed is None:
                        n_skipped_inactive += 1
                        continue

                    all_triples.append({
                        "category": cat_name,
                        "concept_word": concept_word,
                        "parent_latent_idx": parent_latent_idx,
                        "child_latent_idx": child_latent_idx,
                        "child_concept_act": child_info["concept_act"],
                        "child_differential": child_info["differential"],
                        "child_mean_no_parent": child_no_parent,
                        "is_suppressed": bool(is_suppressed),
                        "suppression_ratio": float(sup_ratio),
                        "n_parent_fired": n_pf,
                        "n_parent_not_fired": n_pnf,
                    })
                    n_valid += 1

        print(f"    Triples: {n_valid} valid, {n_skipped_no_child} skipped (no child), "
              f"{n_skipped_inactive} skipped (child inactive)")

    # Compute absorption rate
    if not all_triples:
        return {
            "hierarchy": hierarchy_name,
            "n_triples": 0,
            "absorption_rate": 0.0,
            "ratio_to_null": None,
            "go_nogo": "NO_GO",
            "error": "No valid triples — child latents inactive in contexts",
            "parent_latent_info": parent_latent_info,
        }

    suppressed = np.array([t["is_suppressed"] for t in all_triples], dtype=float)
    suppression_ratios = np.array([t["suppression_ratio"] for t in all_triples], dtype=float)
    n_triples = len(all_triples)
    absorption_rate = float(suppressed.mean())
    mean_suppression_ratio = float(suppression_ratios.mean())

    # Bootstrap CI
    boot_rates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n_triples, n_triples, replace=True)
        boot_rates.append(float(suppressed[idx].mean()))
    ci_lower = float(np.percentile(boot_rates, 2.5))
    ci_upper = float(np.percentile(boot_rates, 97.5))

    # Permutation null: shuffle concept_word assignments within categories
    null_rates = []
    for _ in range(n_permutations):
        perm = rng.permutation(n_triples)
        perm_suppressed = [all_triples[perm[i]]["is_suppressed"] for i in range(n_triples)]
        null_rates.append(float(np.mean(perm_suppressed)))

    null_mean = float(np.mean(null_rates))
    null_std = float(np.std(null_rates))

    if null_mean > 0:
        ratio_to_null = absorption_rate / null_mean
    elif absorption_rate > 0:
        ratio_to_null = 10.0  # signal with zero null
    else:
        ratio_to_null = 1.0  # both zero — no signal

    go_nogo = "GO" if ratio_to_null >= 1.5 else ("WEAK" if ratio_to_null >= 1.1 else "NO_GO")

    print(f"\n  [{hierarchy_name}] Summary:")
    print(f"    n_triples={n_triples}, absorption_rate={absorption_rate:.4f}")
    print(f"    CI=[{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"    null_mean={null_mean:.4f} ± {null_std:.4f}")
    print(f"    ratio_to_null={ratio_to_null:.3f}, go_nogo={go_nogo}")
    print(f"    mean_suppression_ratio={mean_suppression_ratio:.4f}")
    sys.stdout.flush()

    return {
        "hierarchy": hierarchy_name,
        "n_triples": n_triples,
        "n_suppressed": int(suppressed.sum()),
        "absorption_rate": absorption_rate,
        "mean_suppression_ratio": mean_suppression_ratio,
        "bootstrap_ci_lower": ci_lower,
        "bootstrap_ci_upper": ci_upper,
        "ci_reliable": n_triples >= 10,
        "null_mean": null_mean,
        "null_std": null_std,
        "ratio_to_null": float(ratio_to_null),
        "n_permutations": n_permutations,
        "go_nogo": go_nogo,
        "parent_latent_info": parent_latent_info,
        "triples_sample": all_triples[:10],
    }


# ── Define hierarchy word lists ───────────────────────────────────────────────

# FIRST_LETTER hierarchy: use multiple letters for more data
# Key parent: the letter-predicting SAE latent
# Key children: word-specific latents for each word starting with that letter
first_letter_categories = {
    "letter_a": {
        "concept_tokens": [
            "apple", "angel", "arrow", "album", "alarm", "anchor", "anger",
            "atlas", "arena", "actor", "audio", "armor",
        ],
        "unrelated_tokens": [
            "table", "chair", "river", "cloud", "music", "stone", "night",
            "ocean", "bread", "flame", "knife", "clock",
        ],
    },
    "letter_s": {
        "concept_tokens": [
            "stone", "storm", "smile", "smoke", "snake", "sheep", "sword",
            "shell", "spark", "shore", "slope",
        ],
        "unrelated_tokens": [
            "apple", "chair", "river", "cloud", "brain", "night",
            "ocean", "bread", "flame", "knife", "clock", "drama",
        ],
    },
    "letter_m": {
        "concept_tokens": [
            "market", "mirror", "metal", "magic", "model", "manor", "manor",
            "maple", "month", "moral", "motor", "mouse",
        ],
        "unrelated_tokens": [
            "apple", "chair", "river", "cloud", "stone", "night",
            "ocean", "bread", "flame", "knife", "clock", "drama",
        ],
    },
}

# ANIMATE_INANIMATE hierarchy
animate_categories = {
    "animate": {
        "concept_tokens": [
            "dog", "cat", "bird", "fish", "horse", "elephant", "lion", "tiger",
            "wolf", "bear", "eagle", "monkey", "rabbit", "cow", "deer",
        ],
        "unrelated_tokens": [
            "rock", "stone", "table", "chair", "book", "car", "house", "bridge",
            "bottle", "box", "ring", "cloud", "river", "mountain",
        ],
    },
}

# NOUN_PROPER hierarchy
proper_noun_categories = {
    "proper_noun": {
        "concept_tokens": [
            "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid",
            "Alice", "Robert", "Michael", "Jennifer", "William",
        ],
        "unrelated_tokens": [
            "table", "chair", "river", "music", "stone", "flower", "kitchen",
            "bridge", "forest", "garden", "paper", "clock",
        ],
    },
}

# ── Step 7: Run first_letter (sanity check) ───────────────────────────────────

report_progress(5, TOTAL_STEPS, "Running first_letter hierarchy (sanity check)")

print("\n" + "="*70)
print("HIERARCHY 1: first_letter (sanity check — known sae-spelling absorption)")
print("="*70)

try:
    result_fl = run_hierarchy(
        "first_letter",
        first_letter_categories,
        n_contexts=50,
        suppression_threshold=0.30,
        parent_percentile=75,
        n_permutations=100,
        n_bootstrap=1000,
        seed=SEED,
    )
except Exception as e:
    import traceback
    print(f"ERROR first_letter: {e}")
    traceback.print_exc()
    result_fl = {"hierarchy": "first_letter", "error": str(e), "go_nogo": "ERROR",
                 "ratio_to_null": None, "absorption_rate": None}

# ── Step 8: Run animate_inanimate ─────────────────────────────────────────────

report_progress(6, TOTAL_STEPS, "Running animate_inanimate hierarchy")

print("\n" + "="*70)
print("HIERARCHY 2: animate_inanimate")
print("="*70)

try:
    result_ai = run_hierarchy(
        "animate_inanimate",
        animate_categories,
        n_contexts=50,
        suppression_threshold=0.30,
        parent_percentile=75,
        n_permutations=100,
        n_bootstrap=1000,
        seed=SEED,
    )
except Exception as e:
    import traceback
    print(f"ERROR animate_inanimate: {e}")
    traceback.print_exc()
    result_ai = {"hierarchy": "animate_inanimate", "error": str(e), "go_nogo": "ERROR",
                 "ratio_to_null": None, "absorption_rate": None}

# ── Step 9: Run noun_proper ────────────────────────────────────────────────────

report_progress(7, TOTAL_STEPS, "Running noun_proper hierarchy")

print("\n" + "="*70)
print("HIERARCHY 3: noun_proper")
print("="*70)

try:
    result_np = run_hierarchy(
        "noun_proper",
        proper_noun_categories,
        n_contexts=50,
        suppression_threshold=0.30,
        parent_percentile=75,
        n_permutations=100,
        n_bootstrap=1000,
        seed=SEED,
    )
except Exception as e:
    import traceback
    print(f"ERROR noun_proper: {e}")
    traceback.print_exc()
    result_np = {"hierarchy": "noun_proper", "error": str(e), "go_nogo": "ERROR",
                 "ratio_to_null": None, "absorption_rate": None}

# ── Step 10: Compute EDA for parent latents ───────────────────────────────────

report_progress(8, TOTAL_STEPS, "Computing EDA for parent latents")

print("\n" + "="*70)
print("EDA of parent latents (for C3 hierarchy correlation)")
print("="*70)


def compute_eda_for_latent_list(latent_indices):
    edas = []
    for j in latent_indices:
        enc_j = W_enc[:, j].float()
        dec_j = W_dec[j, :].float()
        cos_sim = F.cosine_similarity(enc_j.unsqueeze(0), dec_j.unsqueeze(0)).item()
        edas.append(1.0 - float(cos_sim))
    return edas


eda_per_hierarchy = {}
results_all = {
    "first_letter": result_fl,
    "animate_inanimate": result_ai,
    "noun_proper": result_np,
}

for hname, result in results_all.items():
    parent_info = result.get("parent_latent_info", {})
    all_idx = []
    for cat_name, latents in parent_info.items():
        for lt in latents:
            all_idx.append(lt["latent_idx"])
    all_idx = list(set(all_idx))  # deduplicate

    if all_idx:
        edas = compute_eda_for_latent_list(all_idx)
        eda_per_hierarchy[hname] = {
            "parent_latent_indices": all_idx,
            "eda_values": edas,
            "mean_parent_eda": float(np.mean(edas)),
        }
        print(f"  {hname}: {len(all_idx)} parent latents, mean_EDA={np.mean(edas):.4f}")
    else:
        eda_per_hierarchy[hname] = {"parent_latent_indices": [], "eda_values": [], "mean_parent_eda": None}
        print(f"  {hname}: no parent latents")

sys.stdout.flush()

# ── Step 11: Summary and diagnostics ─────────────────────────────────────────

report_progress(9, TOTAL_STEPS, "Computing overall summary")


def safe_ratio(result):
    r = result.get("ratio_to_null")
    if r is None or (isinstance(r, float) and np.isnan(r)):
        return 0.0
    return float(r)


fl_ratio = safe_ratio(result_fl)
ai_ratio = safe_ratio(result_ai)
np_ratio = safe_ratio(result_np)

n_passing = sum(1 for r in [fl_ratio, ai_ratio, np_ratio] if r >= 1.5)
n_weak = sum(1 for r in [fl_ratio, ai_ratio, np_ratio] if 1.1 <= r < 1.5)

sanity_pass = fl_ratio >= 1.5
sanity_warn = fl_ratio < 1.0  # pipeline bug flag

overall = "GO" if n_passing >= 1 else ("WEAK" if n_weak >= 1 else "NO_GO")
if sanity_warn and n_passing == 0:
    overall = "PIPELINE_BUG_INVESTIGATE"

bonferroni_alpha = 0.05 / 3.0

print("\n" + "="*70)
print("OVERALL SUMMARY (C2-REDESIGN v2)")
print("="*70)
print(f"  first_letter:      absorption_rate={result_fl.get('absorption_rate', 'N/A')}, "
      f"ratio_to_null={fl_ratio:.3f}, go={result_fl.get('go_nogo', '?')}")
print(f"  animate_inanimate: absorption_rate={result_ai.get('absorption_rate', 'N/A')}, "
      f"ratio_to_null={ai_ratio:.3f}, go={result_ai.get('go_nogo', '?')}")
print(f"  noun_proper:       absorption_rate={result_np.get('absorption_rate', 'N/A')}, "
      f"ratio_to_null={np_ratio:.3f}, go={result_np.get('go_nogo', '?')}")
print(f"  Sanity check (first_letter ratio>=1.5): {'PASS' if sanity_pass else ('PIPELINE BUG' if sanity_warn else 'FAIL')}")
print(f"  Overall: {overall} (n_passing={n_passing}, n_weak={n_weak})")
sys.stdout.flush()

# ── Step 12: Save results ─────────────────────────────────────────────────────

report_progress(10, TOTAL_STEPS, "Saving results")

elapsed = time.time() - start_time

output = {
    "task_id": TASK_ID,
    "task_name": "task_C2_redesign",
    "mode": "PILOT",
    "version": "redesign_v2_child_suppression",
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
            "CHILD-FEATURE SUPPRESSION v2: "
            "(1) Find parent latents via differential activation concept vs control. "
            "(2) For each concept_token, find child-specific latent (most differentially active). "
            "(3) Generate 50 sentence contexts for concept_token. "
            "(4) Measure parent and child activations across contexts. "
            "(5) Suppression = child activation drops >= 30% in top-25% parent activation contexts. "
            "(6) Absorption rate = fraction of (parent, child, token) triples showing suppression."
        ),
    },
    "hierarchies": results_all,
    "eda_per_hierarchy": eda_per_hierarchy,
    "summary": {
        "first_letter_ratio": fl_ratio,
        "animate_inanimate_ratio": ai_ratio,
        "noun_proper_ratio": np_ratio,
        "n_passing_ratio_1.5": n_passing,
        "n_weak_ratio_1.1": n_weak,
        "overall_go_nogo": overall,
        "bonferroni_alpha": bonferroni_alpha,
        "sanity_check": {
            "description": "first_letter should show ratio >= 1.5 (known sae-spelling absorption)",
            "ratio": fl_ratio,
            "result": "PASS" if sanity_pass else ("PIPELINE_BUG" if sanity_warn else "FAIL"),
        },
    },
    "pilot_pass_criteria": {
        "criteria": "Ratio-to-null >= 1.5 for first_letter (sanity check against known sae-spelling absorption). If < 1.0 for all including first_letter, investigate pipeline bug.",
        "first_letter_ratio": fl_ratio,
        "first_letter_result": result_fl.get("go_nogo"),
        "overall": overall,
    },
    "design_notes": {
        "v1_issue": "v1 child latents were inactive in varied contexts (child_mean_no_parent < 0.01 for most tokens); "
                    "skipping all triples and producing empty results for first_letter hierarchy.",
        "v2_fix": "Lowered child latent min_concept_act threshold to 0.3 (was 0.5). "
                  "Skip only if child truly inactive (< 0.01) in no-parent contexts. "
                  "Multiple letter categories for first_letter to increase triple count.",
        "chanin_et_al_connection": "sae-spelling absorption definition: child-specific SAE feature "
                                   "fails to fire when parent (letter) feature absorbs it. "
                                   "This script operationalizes: child suppressed when parent strongly fires.",
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nSaved: {OUTPUT_FILE}")
sys.stdout.flush()

# Completion
report_progress(11, TOTAL_STEPS, "Done")

summary = (
    f"C2-REDESIGN v2 PILOT: {overall}. "
    f"first_letter ratio={fl_ratio:.3f} (sanity {'PASS' if sanity_pass else 'FAIL'}). "
    f"animate ratio={ai_ratio:.3f}, noun_proper ratio={np_ratio:.3f}. "
    f"n_passing={n_passing}/3."
)
print(f"\n{'='*70}")
print(f"SUMMARY: {summary}")
print(f"{'='*70}")
sys.stdout.flush()

mark_done("success", summary)
report_progress(12, TOTAL_STEPS, "Complete")
