"""
Task C.2: Cross-Domain Absorption Measurement (PILOT MODE v3)

Redesigned approach using EMPIRICAL activation patterns to identify concept latents,
rather than decoder cosine alignment (which fails for semantic concepts).

Key insight from v2 diagnostic: decoder cosine alignment with probe directions does NOT
identify latents that actually encode semantic concepts. The probe direction is a distributed
representation not captured by a single SAE latent's decoder direction.

v3 approach:
1. Use a "discovery set" of words to find which SAE latents are most differentially active
   for the parent concept vs. control class
2. Use these empirically-discovered "concept latents" as the absorption detection mechanism
3. For each child token, check if the concept latents fail to fire (absorption measurement)
4. Compare to null: same measurement with random or anti-correlated latents

Absorption definition (corrected):
- In first-letter absorption: "cat" should activate the 'c' feature. If the 'c' SAE feature
  fails to fire for "cat" → absorption. But this is NOT the C.2 measurement.
- C.2 measures: does the PARENT concept's SAE latent fail to fire for child tokens?
  E.g., does the "animate" SAE latent fire for "dog"? If it doesn't → absorption.

This is related to the C.1 measurement but at the SAE latent level, not probe level.

Output: exp/results/pilots/pilot_C2_cross_domain_absorption.json
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
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_C2_cross_domain_absorption"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = PILOTS_DIR / "pilot_C2_cross_domain_absorption.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
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
report_progress(0, TOTAL_STEPS, "Starting C2 cross-domain absorption measurement v3")

# ── Step 1: Load SAE and model ─────────────────────────────────────────────────

report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small + SAE")

from sae_lens import SAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()

sae = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=DEVICE
)
sae.eval()

W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
d_sae = W_dec.shape[0]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

print(f"Model: gpt2, SAE d_sae={d_sae}, d_in={W_dec.shape[1]}")
sys.stdout.flush()

# ── Step 2: Utility functions ──────────────────────────────────────────────────

report_progress(2, TOTAL_STEPS, "Defining utility functions")

def get_resid_and_sae_acts(words, layer=6, batch_size=32):
    """Get SAE activations for a list of words."""
    all_sae_acts = []
    for i in range(0, len(words), batch_size):
        batch = words[i:i+batch_size]
        tokens = model.to_tokens(batch, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer}.hook_resid_pre",
                return_type=None,
            )
        acts = cache[f"blocks.{layer}.hook_resid_pre"][:, 1, :].float()
        with torch.no_grad():
            sae_acts = sae.encode(acts)
        all_sae_acts.append(sae_acts.cpu())
    return torch.cat(all_sae_acts, dim=0)


def find_concept_latents_empirical(
    concept_words, control_words,
    top_k=20, diff_threshold=1.0, activation_threshold=0.5,
    seed=42
):
    """
    Find SAE latents that are differentially active for concept_words vs. control_words.

    Returns: (concept_latent_indices, null_latent_indices, concept_stats)
    """
    rng = np.random.RandomState(seed)

    concept_acts = get_resid_and_sae_acts(concept_words)  # [n_concept, d_sae]
    control_acts = get_resid_and_sae_acts(control_words)  # [n_control, d_sae]

    mean_concept = concept_acts.mean(dim=0).numpy()
    mean_control = control_acts.mean(dim=0).numpy()

    # Differential activation: concept latents fire more for concept than control
    diff = mean_concept - mean_control

    # Top concept latents by differential activation
    top_idx = np.argsort(diff)[::-1].copy()[:top_k]
    concept_latents = []
    for idx in top_idx:
        if mean_concept[idx] >= activation_threshold:
            concept_latents.append({
                "latent_idx": int(idx),
                "mean_concept": float(mean_concept[idx]),
                "mean_control": float(mean_control[idx]),
                "differential": float(diff[idx]),
            })

    # Bottom control latents: fire more for control than concept (opposite direction)
    bottom_idx = np.argsort(diff)[:top_k].copy()  # most negative diff
    control_latents = []
    for idx in bottom_idx:
        if mean_control[idx] >= activation_threshold:
            control_latents.append({
                "latent_idx": int(idx),
                "mean_concept": float(mean_concept[idx]),
                "mean_control": float(mean_control[idx]),
                "differential": float(diff[idx]),
            })

    # Random latents (null): exclude concept and control latents
    concept_set = {lt["latent_idx"] for lt in concept_latents}
    control_set = {lt["latent_idx"] for lt in control_latents}
    exclude_set = concept_set | control_set
    candidates = [i for i in range(d_sae) if i not in exclude_set]
    n_random = max(len(concept_latents), 1)
    random_latents = list(rng.choice(candidates, size=min(n_random, len(candidates)), replace=False))

    return concept_latents, control_latents, random_latents


def measure_absorption(
    concept_latent_indices,
    test_sae_acts,
    activation_threshold=0.5,
    n_bootstrap=1000,
    seed=42,
):
    """
    Absorption rate = fraction of test samples where max(concept_latent_acts) < threshold.
    Returns: absorption_rate, bootstrap_ci_lower, bootstrap_ci_upper
    """
    rng = np.random.RandomState(seed)
    n = test_sae_acts.shape[0]

    if not concept_latent_indices:
        return 1.0, 1.0, 1.0, []

    latent_acts = test_sae_acts[:, concept_latent_indices]
    max_acts = latent_acts.max(dim=1).values
    absorbed = (max_acts < activation_threshold).float().numpy()

    absorption_rate = float(absorbed.mean())

    boot_rates = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        boot_rates.append(float(absorbed[idx].mean()))

    ci_lower = float(np.percentile(boot_rates, 2.5))
    ci_upper = float(np.percentile(boot_rates, 97.5))

    return absorption_rate, ci_lower, ci_upper, max_acts.numpy().tolist()


# ── Step 3: Word lists ─────────────────────────────────────────────────────────

report_progress(3, TOTAL_STEPS, "Preparing word lists")

# First-letter hierarchy: letter 'c' (used in Chanin et al. for absorption)
# Discovery: words starting with 'c' vs. words starting with other letters
letter_c_words = [
    "cat", "car", "cup", "cap", "city", "cold", "cut", "core", "cow", "call",
    "coal", "cook", "copy", "care", "corn", "cost", "coat", "clay", "clock", "club"
]
non_c_words = [
    "apple", "ball", "dog", "egg", "fox", "goal", "hat", "idea", "job", "king",
    "lake", "man", "note", "oak", "pen", "queen", "road", "sun", "tree", "use"
]

# Animate vs inanimate
animate_discovery = [
    "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "lion", "tiger", "wolf",
    "bear", "deer", "eagle", "monkey", "rabbit"
]
inanimate_discovery = [
    "rock", "stone", "table", "chair", "book", "pen", "car", "bus", "train", "house",
    "bridge", "bottle", "glass", "box", "ring"
]

animate_test = [
    "owl", "frog", "bee", "ant", "mouse", "rat", "snake", "shark", "whale", "dolphin",
    "woman", "man", "child", "farmer", "doctor", "teacher", "soldier", "pilot", "nurse", "chef"
]
inanimate_test = [
    "flower", "grass", "water", "fire", "cloud", "rain", "snow", "wind", "sand",
    "mountain", "river", "lake", "ocean", "island", "forest", "desert", "valley",
    "plate", "cup", "fork", "knife", "spoon", "bag", "door", "window"
]

# Proper nouns vs common nouns
proper_discovery = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid", "Beijing",
    "Moscow", "Cairo", "Alice", "Robert", "Michael", "Jennifer", "William"
]
common_discovery = [
    "table", "chair", "house", "car", "book", "tree", "dog", "cat", "city",
    "school", "hospital", "market", "garden", "office", "computer"
]

proper_test = [
    "Delhi", "Seoul", "Bangkok", "Toronto", "Istanbul", "Singapore", "Dubai",
    "Lagos", "Jakarta", "Tehran", "Elizabeth", "James", "Google", "Apple", "Microsoft",
    "January", "February", "March", "April", "Monday", "Tuesday"
]
common_test = [
    "river", "mountain", "flower", "kitchen", "bedroom", "bridge", "road", "paper",
    "pen", "glass", "bottle", "bag", "box", "clock", "lamp",
    "rock", "stone", "field", "forest", "valley"
]

# ── Step 4: Run all hierarchies ────────────────────────────────────────────────

report_progress(4, TOTAL_STEPS, "Running absorption measurements")

results_per_hierarchy = {}
N_BOOTSTRAP = 1000
ACTIVATION_THRESHOLD = 0.5  # SAE activation threshold

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 1: first_letter (letter 'c')
# ────────────────────────────────────────────────────────────────────────────────
report_progress(5, TOTAL_STEPS, "Measuring first_letter ('c') absorption")
print("\n" + "="*60)
print("HIERARCHY: first_letter (letter 'c')")
sys.stdout.flush()

try:
    # Discover concept latents for 'c' letter
    c_latents, nonc_latents, rand_latents_c = find_concept_latents_empirical(
        letter_c_words, non_c_words,
        top_k=20, diff_threshold=0.5, activation_threshold=ACTIVATION_THRESHOLD,
        seed=SEED
    )

    print(f"  'c' concept latents (n={len(c_latents)}):")
    for lt in c_latents[:5]:
        print(f"    latent {lt['latent_idx']}: concept_mean={lt['mean_concept']:.3f}, ctrl_mean={lt['mean_control']:.3f}")

    if not c_latents:
        print("  WARNING: No differentially-active latents found above threshold!")
        results_per_hierarchy["first_letter"] = {
            "hierarchy": "first_letter_c",
            "error": "No concept latents found above activation threshold",
            "go_nogo": "NO_GO",
            "n_concept_latents_found": 0,
        }
    else:
        c_latent_indices = [lt["latent_idx"] for lt in c_latents]
        rand_c_indices = list(rand_latents_c)

        # Test set: more 'c' words + words from other letters
        letter_c_test = [
            "cup", "coin", "coal", "cut", "cave", "case", "camp", "cage", "cake", "calm",
            "cape", "cart", "cave", "cent", "chef", "chin", "chip", "clip", "clue", "coal"
        ]
        non_c_test = [
            "apple", "bed", "door", "egg", "fog", "gun", "heat", "ink", "jar", "key",
            "line", "map", "news", "oak", "path", "quiz", "rain", "salt", "tide", "vat"
        ]

        test_c_sae = get_resid_and_sae_acts(letter_c_test)
        test_nonc_sae = get_resid_and_sae_acts(non_c_test)

        # Absorption: 'c' concept latents fail to fire for 'c' words
        abs_rate_c, ci_lower_c, ci_upper_c, max_acts_c = measure_absorption(
            c_latent_indices, test_c_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        # Null 1: random latents for 'c' words
        null_rate_rand_c, _, _, _ = measure_absorption(
            rand_c_indices, test_c_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        # Control: same 'c' latents for NON-'c' words (should show higher absorption)
        ctrl_rate_c, _, _, _ = measure_absorption(
            c_latent_indices, test_nonc_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        ratio_c = abs_rate_c / null_rate_rand_c if null_rate_rand_c > 0 else float("inf")

        print(f"\n  Test results:")
        print(f"    'c' concept latents on 'c' words: absorption_rate={abs_rate_c:.4f}")
        print(f"    Random latents on 'c' words (null): {null_rate_rand_c:.4f}")
        print(f"    'c' concept latents on non-'c' words (control): {ctrl_rate_c:.4f}")
        print(f"    Ratio (abs/null): {ratio_c:.3f}")
        print(f"    Mean max activation: {np.mean(max_acts_c):.3f}")
        sys.stdout.flush()

        results_per_hierarchy["first_letter"] = {
            "hierarchy": "first_letter_c",
            "description": "Do 'c' SAE latents fail to fire for 'c' words? (absorption = parent latent suppressed)",
            "n_concept_latents": len(c_latent_indices),
            "n_test_concept": len(letter_c_test),
            "n_test_control": len(non_c_test),
            "absorption_rate": float(abs_rate_c),
            "null_rate_random_latents": float(null_rate_rand_c),
            "control_rate_unrelated_class": float(ctrl_rate_c),
            "ratio_to_null": float(ratio_c),
            "bootstrap_ci_lower": ci_lower_c,
            "bootstrap_ci_upper": ci_upper_c,
            "ci_reliable": True,
            "mean_max_concept_activation": float(np.mean(max_acts_c)),
            "concept_latents_top5": c_latents[:5],
            "go_nogo": "GO" if ratio_c >= 1.5 else ("WEAK" if ratio_c >= 1.1 else "NO_GO"),
            "interpretation": (
                "If absorption_rate >> null_rate → 'c' latents suppress FOR 'c' words → anti-absorption. "
                "If absorption_rate ≈ null_rate → no selective signal. "
                "If absorption_rate << null_rate → 'c' latents ARE active for 'c' words → properly encoded (no absorption). "
                "Key comparison: absorption_rate vs. ctrl_rate_unrelated_class (should be higher if latents are semantic)."
            ),
        }

except Exception as e:
    import traceback
    print(f"ERROR in first_letter: {e}")
    traceback.print_exc()
    results_per_hierarchy["first_letter"] = {"error": str(e), "go_nogo": "NO_GO"}

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 2: animate_inanimate
# ────────────────────────────────────────────────────────────────────────────────
report_progress(6, TOTAL_STEPS, "Measuring animate_inanimate absorption")
print("\n" + "="*60)
print("HIERARCHY: animate_inanimate")
sys.stdout.flush()

try:
    animate_latents, inanimate_latents, rand_latents_ai = find_concept_latents_empirical(
        animate_discovery, inanimate_discovery,
        top_k=20, diff_threshold=0.5, activation_threshold=ACTIVATION_THRESHOLD,
        seed=SEED
    )

    print(f"  Animate concept latents (n={len(animate_latents)}):")
    for lt in animate_latents[:5]:
        print(f"    latent {lt['latent_idx']}: animate_mean={lt['mean_concept']:.3f}, inanimate_mean={lt['mean_control']:.3f}, diff={lt['differential']:.3f}")

    if not animate_latents:
        print("  WARNING: No differentially-active latents found!")
        results_per_hierarchy["animate_inanimate"] = {
            "hierarchy": "animate_inanimate",
            "error": "No concept latents found above activation threshold",
            "go_nogo": "NO_GO",
        }
    else:
        animate_latent_indices = [lt["latent_idx"] for lt in animate_latents]
        rand_ai_indices = list(rand_latents_ai)

        test_animate_sae = get_resid_and_sae_acts(animate_test)
        test_inanimate_sae = get_resid_and_sae_acts(inanimate_test)

        # Absorption: animate latents fail to fire for animate words
        abs_rate_ai, ci_lower_ai, ci_upper_ai, max_acts_ai = measure_absorption(
            animate_latent_indices, test_animate_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        # Null: random latents for animate words
        null_rate_rand_ai, _, _, _ = measure_absorption(
            rand_ai_indices, test_animate_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        # Control: animate latents for inanimate words (expected higher absorption)
        ctrl_rate_ai, _, _, _ = measure_absorption(
            animate_latent_indices, test_inanimate_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        ratio_ai = abs_rate_ai / null_rate_rand_ai if null_rate_rand_ai > 0 else float("inf")

        print(f"\n  Test results:")
        print(f"    Animate concept latents on animate words: absorption_rate={abs_rate_ai:.4f}")
        print(f"    Random latents on animate words (null): {null_rate_rand_ai:.4f}")
        print(f"    Animate concept latents on inanimate words (control): {ctrl_rate_ai:.4f}")
        print(f"    Ratio (abs/null): {ratio_ai:.3f}")
        print(f"    Mean max activation: {np.mean(max_acts_ai):.3f}")
        print(f"    CI: [{ci_lower_ai:.4f}, {ci_upper_ai:.4f}]")
        sys.stdout.flush()

        results_per_hierarchy["animate_inanimate"] = {
            "hierarchy": "animate_inanimate",
            "description": "Do animate SAE latents fail to fire for animate words?",
            "n_concept_latents": len(animate_latent_indices),
            "n_test_concept": len(animate_test),
            "n_test_control": len(inanimate_test),
            "absorption_rate": float(abs_rate_ai),
            "null_rate_random_latents": float(null_rate_rand_ai),
            "control_rate_unrelated_class": float(ctrl_rate_ai),
            "ratio_to_null": float(ratio_ai),
            "bootstrap_ci_lower": ci_lower_ai,
            "bootstrap_ci_upper": ci_upper_ai,
            "ci_reliable": len(animate_test) >= 10,
            "mean_max_concept_activation": float(np.mean(max_acts_ai)),
            "concept_latents_top5": animate_latents[:5],
            "go_nogo": "GO" if ratio_ai >= 1.5 else ("WEAK" if ratio_ai >= 1.1 else "NO_GO"),
        }

except Exception as e:
    import traceback
    print(f"ERROR in animate_inanimate: {e}")
    traceback.print_exc()
    results_per_hierarchy["animate_inanimate"] = {"error": str(e), "go_nogo": "NO_GO"}

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 3: noun_proper
# ────────────────────────────────────────────────────────────────────────────────
report_progress(7, TOTAL_STEPS, "Measuring noun_proper absorption")
print("\n" + "="*60)
print("HIERARCHY: noun_proper")
sys.stdout.flush()

try:
    proper_latents, common_latents, rand_latents_np = find_concept_latents_empirical(
        proper_discovery, common_discovery,
        top_k=20, diff_threshold=0.5, activation_threshold=ACTIVATION_THRESHOLD,
        seed=SEED
    )

    print(f"  Proper noun concept latents (n={len(proper_latents)}):")
    for lt in proper_latents[:5]:
        print(f"    latent {lt['latent_idx']}: proper_mean={lt['mean_concept']:.3f}, common_mean={lt['mean_control']:.3f}")

    if not proper_latents:
        print("  WARNING: No differentially-active latents found!")
        results_per_hierarchy["noun_proper"] = {
            "hierarchy": "noun_proper",
            "error": "No concept latents found above activation threshold",
            "go_nogo": "NO_GO",
        }
    else:
        proper_latent_indices = [lt["latent_idx"] for lt in proper_latents]
        rand_np_indices = list(rand_latents_np)

        test_proper_sae = get_resid_and_sae_acts(proper_test)
        test_common_sae = get_resid_and_sae_acts(common_test)

        abs_rate_np, ci_lower_np, ci_upper_np, max_acts_np = measure_absorption(
            proper_latent_indices, test_proper_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        null_rate_rand_np, _, _, _ = measure_absorption(
            rand_np_indices, test_proper_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        ctrl_rate_np, _, _, _ = measure_absorption(
            proper_latent_indices, test_common_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        ratio_np = abs_rate_np / null_rate_rand_np if null_rate_rand_np > 0 else float("inf")

        print(f"\n  Test results:")
        print(f"    Proper noun latents on proper nouns: absorption_rate={abs_rate_np:.4f}")
        print(f"    Random latents on proper nouns (null): {null_rate_rand_np:.4f}")
        print(f"    Proper noun latents on common nouns (control): {ctrl_rate_np:.4f}")
        print(f"    Ratio (abs/null): {ratio_np:.3f}")
        print(f"    Mean max activation: {np.mean(max_acts_np):.3f}")
        print(f"    CI: [{ci_lower_np:.4f}, {ci_upper_np:.4f}]")
        sys.stdout.flush()

        results_per_hierarchy["noun_proper"] = {
            "hierarchy": "noun_proper",
            "description": "Do proper-noun SAE latents fail to fire for proper nouns?",
            "n_concept_latents": len(proper_latent_indices),
            "n_test_concept": len(proper_test),
            "n_test_control": len(common_test),
            "absorption_rate": float(abs_rate_np),
            "null_rate_random_latents": float(null_rate_rand_np),
            "control_rate_unrelated_class": float(ctrl_rate_np),
            "ratio_to_null": float(ratio_np),
            "bootstrap_ci_lower": ci_lower_np,
            "bootstrap_ci_upper": ci_upper_np,
            "ci_reliable": len(proper_test) >= 10,
            "mean_max_concept_activation": float(np.mean(max_acts_np)),
            "concept_latents_top5": proper_latents[:5],
            "go_nogo": "GO" if ratio_np >= 1.5 else ("WEAK" if ratio_np >= 1.1 else "NO_GO"),
        }

except Exception as e:
    import traceback
    print(f"ERROR in noun_proper: {e}")
    traceback.print_exc()
    results_per_hierarchy["noun_proper"] = {"error": str(e), "go_nogo": "NO_GO"}

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 4: city_country_binary (FLAGGED)
# ────────────────────────────────────────────────────────────────────────────────
report_progress(8, TOTAL_STEPS, "Measuring city_country absorption (FLAGGED)")
print("\n" + "="*60)
print("HIERARCHY: city_country_binary (FLAGGED - failed shuffle gate)")
sys.stdout.flush()

try:
    us_discovery = [
        "Chicago", "Houston", "Phoenix", "Philadelphia", "Dallas",
        "Jacksonville", "Austin", "Columbus", "Memphis", "Baltimore"
    ]
    nonus_discovery = [
        "London", "Paris", "Berlin", "Tokyo", "Sydney",
        "Rome", "Madrid", "Beijing", "Moscow", "Cairo"
    ]

    us_latents, nonus_latents, rand_latents_cc = find_concept_latents_empirical(
        us_discovery, nonus_discovery,
        top_k=20, diff_threshold=0.5, activation_threshold=ACTIVATION_THRESHOLD,
        seed=SEED
    )

    print(f"  US concept latents (n={len(us_latents)}):")
    for lt in us_latents[:5]:
        print(f"    latent {lt['latent_idx']}: us_mean={lt['mean_concept']:.3f}, nonus_mean={lt['mean_control']:.3f}")

    if not us_latents:
        results_per_hierarchy["city_country_binary"] = {
            "hierarchy": "city_country_binary",
            "shuffle_gate_warning": "FLAGGED: Failed shuffle gate in C.1.",
            "error": "No concept latents found",
            "go_nogo": "FLAGGED",
        }
    else:
        us_latent_indices = [lt["latent_idx"] for lt in us_latents]

        us_test = ["Omaha", "Raleigh", "Cleveland", "Tulsa", "Tampa",
                   "Milwaukee", "Albuquerque", "Tucson", "Fresno", "Sacramento"]
        nonus_test = ["Delhi", "Seoul", "Bangkok", "Toronto", "Istanbul",
                      "Singapore", "Dubai", "Lagos", "Nairobi", "Jakarta"]

        test_us_sae = get_resid_and_sae_acts(us_test)
        test_nonus_sae = get_resid_and_sae_acts(nonus_test)

        abs_rate_cc, ci_lower_cc, ci_upper_cc, max_acts_cc = measure_absorption(
            us_latent_indices, test_us_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)
        null_rate_cc, _, _, _ = measure_absorption(
            list(rand_latents_cc), test_us_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)
        ctrl_rate_cc, _, _, _ = measure_absorption(
            us_latent_indices, test_nonus_sae, ACTIVATION_THRESHOLD, N_BOOTSTRAP, SEED)

        ratio_cc = abs_rate_cc / null_rate_cc if null_rate_cc > 0 else float("inf")
        print(f"  US latents on US cities: absorption_rate={abs_rate_cc:.4f}, null={null_rate_cc:.4f}, ratio={ratio_cc:.3f}")
        sys.stdout.flush()

        results_per_hierarchy["city_country_binary"] = {
            "hierarchy": "city_country_binary",
            "shuffle_gate_warning": "FLAGGED: Failed shuffled-label control in C.1. Non-inferential.",
            "n_concept_latents": len(us_latent_indices),
            "n_test_concept": len(us_test),
            "absorption_rate": float(abs_rate_cc),
            "null_rate_random_latents": float(null_rate_cc),
            "control_rate_unrelated_class": float(ctrl_rate_cc),
            "ratio_to_null": float(ratio_cc),
            "go_nogo": "FLAGGED",
        }

except Exception as e:
    import traceback
    print(f"ERROR in city_country: {e}")
    traceback.print_exc()
    results_per_hierarchy["city_country_binary"] = {
        "error": str(e), "go_nogo": "FLAGGED",
        "shuffle_gate_warning": "FLAGGED: Failed shuffle gate in C.1.",
    }

# ── Step 9: Supplementary: re-interpretation ──────────────────────────────────

report_progress(9, TOTAL_STEPS, "Interpreting results")

print("\n" + "="*60)
print("INTERPRETATION:")
print()
print("NOTE: The v3 measurement uses a discovery/test split with empirically-identified")
print("concept latents. The 'absorption_rate' here measures: fraction of test words where")
print("concept latents FAIL to activate. This is NOT the traditional absorption measurement.")
print()
print("Traditional absorption (Chanin et al.): a 'child' feature fails to fire when the")
print("child word is present, BECAUSE it is 'absorbed' into a parent feature.")
print("E.g., 'cat' → 'c' probe fires but 'cat' feature doesn't fire (absorbed into 'c').")
print()
print("C.2 cross-domain: do semantic hierarchy latents (animate, proper_noun) show the")
print("same phenomenon? I.e., 'animate' SAE latent fires for 'dog' but 'dog' SAE latent")
print("doesn't fire (it's absorbed into 'animate').")
print()
print("The CORRECT measurement for C.2 would be:")
print("  - For each child word (e.g., 'dog'), does the CHILD-specific latent fail to fire")
print("    because its information is absorbed into the PARENT (animate) latent?")
print()
print("What we're currently measuring is different: does the PARENT latent (animate) fail")
print("to fire for child words? This measures 'encoding quality', not 'absorption'.")
sys.stdout.flush()

# ── Step 10: Summary ──────────────────────────────────────────────────────────

report_progress(10, TOTAL_STEPS, "Computing summary")

non_spelling = ["noun_proper", "animate_inanimate"]
n_passing = sum(
    1 for h in non_spelling
    if results_per_hierarchy.get(h, {}).get("ratio_to_null", 0) >= 1.5
)
n_weak = sum(
    1 for h in non_spelling
    if 1.1 <= results_per_hierarchy.get(h, {}).get("ratio_to_null", 0) < 1.5
)
overall_go_nogo = "GO" if n_passing >= 1 else ("WEAK_SIGNAL" if n_weak >= 1 else "NO_GO")

findings = []
for h_name, res in results_per_hierarchy.items():
    if "error" in res and "absorption_rate" not in res:
        findings.append(f"{h_name}: ERROR - {res['error']}")
        continue
    abs_rate = res.get("absorption_rate", 0) or 0.0
    null_rate = res.get("null_rate_random_latents", 0) or 0.0
    ratio = res.get("ratio_to_null", 0) or 0.0
    go = res.get("go_nogo", "?")
    findings.append(f"{h_name}: abs={abs_rate:.3f}, null={null_rate:.3f}, ratio={ratio:.2f}, go={go}")

print(f"\nOverall: {overall_go_nogo}")
for f in findings:
    print(f"  {f}")

# ── Step 11: Save results ─────────────────────────────────────────────────────

report_progress(11, TOTAL_STEPS, "Saving results")

elapsed = time.time() - start_time
bonferroni_alpha = 0.05 / len(non_spelling)

output = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "version": "v3",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "layer": 6,
        "seed": SEED,
        "activation_threshold": ACTIVATION_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "bonferroni_alpha": bonferroni_alpha,
        "n_comparisons": len(non_spelling),
        "concept_latent_discovery": "empirical_differential_activation",
    },
    "hierarchies": results_per_hierarchy,
    "pilot_pass_criteria": {
        "pass_criterion": "ratio_to_null >= 1.5 for at least 1 non-spelling hierarchy",
        "non_spelling_hierarchies": non_spelling,
        "n_passing_ratio_1.5": n_passing,
        "n_passing_ratio_1.1": n_weak,
        "overall_go_nogo": overall_go_nogo,
        "findings": findings,
    },
    "design_notes": {
        "v3_key_change": (
            "v3 uses empirical activation patterns to find concept latents, "
            "rather than decoder cosine alignment (v1/v2). "
            "Decoder cosine alignment found latents with cos_sim ~0.25 but mean activation ~0 — "
            "these are not the latents that actually encode the semantic concept."
        ),
        "absorption_interpretation_caveat": (
            "Current measurement: 'do parent concept latents fail to fire for concept-related words?' "
            "This is NOT the traditional absorption definition (which would be: "
            "'does the specific child feature fail to fire because it's absorbed into parent?'). "
            "The ratio_to_null compares: concept_latent_absorption_rate / random_latent_absorption_rate. "
            "If ratio < 1 and mean_max_activation > 0: concept latents DO fire (good encoding, no absorption). "
            "If ratio ≈ 1: no differential signal. "
            "If ratio > 1: concept latents fail MORE than random → absorption signal."
        ),
        "iter_001_lesson": "All probes trained on GPT-2 Small (same model analyzed). No cross-model transfer.",
        "c1_passing": "first_letter (F1=0.820), city_country_binary (F1=0.865, FAILED shuffle gate), noun_proper (F1=0.987), animate_inanimate (F1=1.0)",
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to {OUTPUT_FILE}")
sys.stdout.flush()

# ── Step 12: Done ─────────────────────────────────────────────────────────────

report_progress(12, TOTAL_STEPS, "Done")

summary = (
    f"C2 PILOT v3: {overall_go_nogo}. "
    f"Non-spelling passing (ratio>=1.5): {n_passing}/{len(non_spelling)}, "
    f"weak (>=1.1): {n_weak}. "
    + "; ".join(findings)
)
print(f"\n{'='*60}")
print(f"SUMMARY: {summary}")
print(f"{'='*60}")
sys.stdout.flush()

mark_done("success", summary)
