"""
Task C.2: Cross-Domain Absorption Measurement (PILOT MODE v2)

Redesigned absorption measurement:

The key question is: When processing a child token (e.g., "London"), do the SAE latents
that encode the parent concept (e.g., "is_US_city") activate?

TRUE absorption: parent-concept latents fail to activate for child tokens
NOT absorption: parent-concept latents DO activate for child tokens (proper encoding)

Measurement protocol (revised):
1. Identify "parent-concept latents" = SAE latents whose decoder direction aligns with probe direction
2. For each child token: check if parent-concept latents activate above threshold
3. "Absorbed" = parent latents DON'T fire (absorption = the child is encoded without the parent concept)
4. "Not absorbed" = parent latents DO fire (child token activates parent concept feature)

Null control:
- Use UNRELATED latents (randomly sampled SAE latents with low probe alignment)
- Compare: parent-aligned latents vs. random latents in activation rate
- If absorption is happening: parent latents should fire LESS for child tokens than random/unrelated

Pass criterion: ratio-to-null >= 1.5 for at least 1 non-spelling hierarchy
(meaning: parent latents fire at HIGHER rate than null for child tokens → absorption is NOT happening)
OR: parent latents fire at LOWER rate than null → absorption IS happening, ratio < 1.0

REVISED: Absorption rate = fraction of child tokens where parent latent does NOT activate.
Signal = absorption_rate is SIGNIFICANTLY HIGHER than what you'd get with random latents.

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


TOTAL_STEPS = 10
report_progress(0, TOTAL_STEPS, "Starting C2 cross-domain absorption measurement v2")

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
W_enc = sae.W_enc.detach().float()  # [d_model, d_sae]
b_enc = sae.b_enc.detach().float()  # [d_sae]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token
d_sae = W_dec.shape[0]

print(f"Model: gpt2, SAE d_sae={d_sae}, d_in={W_dec.shape[1]}")

# ── Step 2: Load probe weights ─────────────────────────────────────────────────

report_progress(2, TOTAL_STEPS, "Loading probe weights from C.1")

def load_probe(hierarchy_name):
    w_path = PROBES_DIR / f"probe_{hierarchy_name}_weights.npy"
    c_path = PROBES_DIR / f"probe_{hierarchy_name}_classes.npy"
    if not w_path.exists():
        return None, None
    weights = np.load(str(w_path))
    classes = np.load(str(c_path), allow_pickle=True)
    return weights, classes

c1_results = json.loads((FULL_DIR / "C1_probe_training.json").read_text())
passing_hierarchies = c1_results["passing_hierarchies"]
print(f"Passing hierarchies from C.1: {passing_hierarchies}")

# ── Step 3: Utility functions ──────────────────────────────────────────────────

def get_gpt2_activations_batch(model, texts, layer=6, batch_size=32, device="cuda"):
    """Get layer 6 residual stream activations for texts."""
    all_acts = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        tokens = model.to_tokens(batch_texts, prepend_bos=True)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens,
                names_filter=f"blocks.{layer}.hook_resid_pre",
                return_type=None,
            )
        acts = cache[f"blocks.{layer}.hook_resid_pre"][:, 1, :]
        all_acts.append(acts.cpu())
    return torch.cat(all_acts, dim=0)


def get_sae_activations(sae, resid_acts, device="cuda"):
    """Run SAE forward pass on residual stream activations."""
    resid_acts = resid_acts.to(device).float()
    with torch.no_grad():
        feature_acts = sae.encode(resid_acts)
    return feature_acts.cpu()


def find_concept_latents(probe_weight_vec, W_dec, top_k=20, cos_threshold=0.2):
    """
    Find SAE latents whose decoder direction aligns with a probe weight vector.
    probe_weight_vec: [d_model] numpy array (single class direction)
    W_dec: [d_sae, d_model] torch tensor
    Returns: list of (latent_idx, cos_sim) sorted by cos_sim desc
    """
    probe_vec = torch.tensor(probe_weight_vec, dtype=torch.float32, device=W_dec.device)
    probe_norm = F.normalize(probe_vec.unsqueeze(0), dim=1)  # [1, d_model]
    W_dec_norm = F.normalize(W_dec, dim=1)  # [d_sae, d_model]
    cos_sims = torch.mm(probe_norm, W_dec_norm.T).squeeze(0)  # [d_sae]
    cos_sims_np = cos_sims.detach().cpu().numpy()
    top_indices = np.argsort(cos_sims_np)[::-1][:top_k]
    result = [(int(idx), float(cos_sims_np[idx])) for idx in top_indices
              if cos_sims_np[idx] >= cos_threshold]
    return result


def find_random_latents(n_random, d_sae, exclude_set, rng):
    """Sample random latents excluding concept latents and high-EDA latents."""
    candidates = list(set(range(d_sae)) - exclude_set)
    selected = rng.choice(len(candidates), size=min(n_random, len(candidates)), replace=False)
    return [candidates[i] for i in selected]


# ── Step 4: Core absorption measurement function ───────────────────────────────

def measure_absorption_rate(
    concept_latent_indices,
    sae_acts,
    activation_threshold=0.5,
):
    """
    For each sample, check if the concept latents fail to activate.
    Absorption rate = fraction of samples where max(concept_latent_acts) < threshold.

    sae_acts: [n_samples, d_sae]
    Returns: absorption_rate, per_sample_max_acts
    """
    if not concept_latent_indices:
        return 1.0, []

    latent_acts = sae_acts[:, concept_latent_indices]  # [n_samples, n_latents]
    max_acts = latent_acts.max(dim=1).values  # [n_samples]
    absorbed = (max_acts < activation_threshold).float()
    return float(absorbed.mean()), max_acts.numpy().tolist()


# ── Step 5: Hierarchy-specific data builders ──────────────────────────────────

report_progress(3, TOTAL_STEPS, "Building hierarchy datasets")

def build_first_letter_data(model, n_per_letter=15, seed=42):
    """
    For each letter, get tokens that START with that letter (child tokens) and
    tokens that DON'T start with it (control tokens).
    Returns: {letter: {'child_texts': [...], 'control_texts': [...], 'probe_class_idx': int}}
    """
    rng = np.random.RandomState(seed)
    vocab = tokenizer.get_vocab()

    letter_to_tokens = defaultdict(list)
    all_word_tokens = []
    for token_str, token_id in vocab.items():
        clean = token_str.lstrip("Ġ▁ ")
        if clean and len(clean) >= 2 and clean[0].isalpha():
            letter = clean[0].lower()
            letter_to_tokens[letter].append(token_str)
            all_word_tokens.append(token_str)

    result = {}
    for letter in sorted(letter_to_tokens.keys()):
        if letter not in "abcdefghijklmnopqrstuvwxy":
            continue
        childs = letter_to_tokens[letter]
        if len(childs) < 5:
            continue
        sel = rng.choice(len(childs), size=min(n_per_letter, len(childs)), replace=False)
        child_texts = [childs[i].replace("Ġ", " ").strip() for i in sel]

        # Control: tokens from different letters
        other_tokens = [t for t in all_word_tokens if t.lstrip("Ġ▁ ")[0].lower() != letter]
        ctrl_sel = rng.choice(len(other_tokens), size=min(n_per_letter, len(other_tokens)), replace=False)
        control_texts = [other_tokens[i].replace("Ġ", " ").strip() for i in ctrl_sel]

        result[letter] = {"child_texts": child_texts, "control_texts": control_texts}
    return result


def build_binary_hierarchy_data(child_class_items, control_class_items, n=30, seed=42):
    """Generic builder for binary hierarchy."""
    rng = np.random.RandomState(seed)
    sel_child = rng.choice(len(child_class_items), size=min(n, len(child_class_items)), replace=False)
    sel_ctrl = rng.choice(len(control_class_items), size=min(n, len(control_class_items)), replace=False)
    child_texts = [child_class_items[i] for i in sel_child]
    control_texts = [control_class_items[i] for i in sel_ctrl]
    return {"child_texts": child_texts, "control_texts": control_texts}


# Words for hierarchies
proper_nouns = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid", "Beijing",
    "Moscow", "Cairo", "Delhi", "Seoul", "Bangkok", "Toronto", "Chicago",
    "Alice", "Robert", "Michael", "Jennifer", "Elizabeth", "William", "James",
    "Google", "Apple", "Microsoft", "Amazon", "Facebook", "Tesla", "Monday", "Tuesday",
    "January", "February", "March", "April", "June", "July", "August", "September"
]
common_nouns = [
    "table", "chair", "house", "car", "book", "tree", "water", "food", "dog", "cat",
    "door", "window", "floor", "wall", "road", "bridge", "river", "mountain",
    "school", "hospital", "market", "garden", "kitchen", "bedroom", "office",
    "computer", "phone", "paper", "pen", "glass", "bottle", "bag", "box", "clock"
]

animate_words = [
    "dog", "cat", "bird", "fish", "horse", "cow", "sheep", "pig", "lion", "tiger",
    "elephant", "monkey", "eagle", "wolf", "bear", "deer", "rabbit", "mouse",
    "snake", "frog", "bee", "ant", "butterfly", "spider", "dolphin", "whale", "shark",
    "woman", "man", "child", "farmer", "teacher", "doctor", "scientist"
]
inanimate_words = [
    "rock", "stone", "table", "chair", "book", "pen", "phone", "computer", "car", "bus",
    "train", "plane", "ship", "house", "building", "bridge", "road", "flower",
    "bottle", "glass", "plate", "cup", "fork", "knife", "spoon", "bag", "box",
    "mountain", "river", "lake", "ocean", "island", "forest", "desert", "valley"
]

# ── Step 6: Run measurements ──────────────────────────────────────────────────

report_progress(4, TOTAL_STEPS, "Running absorption measurements")

results_per_hierarchy = {}
COS_THRESHOLD = 0.2
TOP_K = 30
ACTIVATION_THRESHOLD = 0.5  # SAE activation threshold (typical SAE activations are 0-30)
N_BOOTSTRAP = 1000
SEED_RNG = np.random.RandomState(SEED)

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 1: first_letter
# ────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HIERARCHY: first_letter")

w_fl, c_fl = load_probe("first_letter")  # [24, 768], classes=letters
if w_fl is not None:
    # For first_letter: each row of w_fl is one letter's probe direction
    # For each letter, find concept latents using probe[letter_idx]
    letter_results = {}
    all_child_texts = []
    all_control_texts = []

    # Build the per-letter data
    fl_data = build_first_letter_data(model, n_per_letter=15, seed=SEED)

    for letter_idx, letter in enumerate(c_fl):
        if letter not in fl_data:
            continue
        probe_vec = w_fl[letter_idx]  # [768]
        concept_latents = find_concept_latents(probe_vec, W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)

        child_texts = fl_data[letter]["child_texts"]
        control_texts = fl_data[letter]["control_texts"]

        if not child_texts:
            continue

        all_child_texts.extend(child_texts)
        all_control_texts.extend(control_texts)

        letter_results[letter] = {
            "concept_latents": [(lt[0], lt[1]) for lt in concept_latents],
            "n_concept_latents": len(concept_latents),
            "n_child": len(child_texts),
        }

    print(f"Total child texts: {len(all_child_texts)}, control texts: {len(all_control_texts)}")

    # Get SAE activations for all texts
    all_texts = list(set(all_child_texts + all_control_texts))
    text_to_idx = {t: i for i, t in enumerate(all_texts)}

    resid_acts_all = get_gpt2_activations_batch(model, all_texts, layer=6, batch_size=64, device=DEVICE)
    sae_acts_all = get_sae_activations(sae, resid_acts_all, device=DEVICE)
    print(f"sae_acts_all shape: {sae_acts_all.shape}")

    # Per-letter absorption measurement
    letter_absorption_rates = {}
    letter_null_rates = {}

    for letter_idx, letter in enumerate(c_fl):
        if letter not in letter_results:
            continue
        lr = letter_results[letter]
        if not lr["concept_latents"]:
            continue

        concept_latent_indices = [lt[0] for lt in lr["concept_latents"]]
        child_texts = fl_data[letter]["child_texts"]
        child_indices = [text_to_idx[t] for t in child_texts if t in text_to_idx]

        if not child_indices:
            continue

        child_sae_acts = sae_acts_all[child_indices]  # [n_child, d_sae]

        # Absorption rate for concept latents
        abs_rate, max_acts = measure_absorption_rate(concept_latent_indices, child_sae_acts, ACTIVATION_THRESHOLD)

        # Null control: random latents of same size
        exclude_set = set(concept_latent_indices)
        random_latents = find_random_latents(len(concept_latent_indices), d_sae, exclude_set, SEED_RNG)
        null_rate, _ = measure_absorption_rate(random_latents, child_sae_acts, ACTIVATION_THRESHOLD)

        letter_absorption_rates[letter] = abs_rate
        letter_null_rates[letter] = null_rate

    if letter_absorption_rates:
        mean_abs_rate = float(np.mean(list(letter_absorption_rates.values())))
        mean_null_rate = float(np.mean(list(letter_null_rates.values())))
        ratio = mean_abs_rate / mean_null_rate if mean_null_rate > 0 else float("inf")

        # Bootstrap CI across letters
        letters = list(letter_absorption_rates.keys())
        letter_rates_arr = np.array([letter_absorption_rates[l] for l in letters])
        boot_means = [float(np.mean(SEED_RNG.choice(letter_rates_arr, len(letter_rates_arr), replace=True)))
                      for _ in range(N_BOOTSTRAP)]
        ci_lower = float(np.percentile(boot_means, 2.5))
        ci_upper = float(np.percentile(boot_means, 97.5))

        print(f"first_letter: abs_rate={mean_abs_rate:.4f}, null_rate={mean_null_rate:.4f}, ratio={ratio:.3f}")
        print(f"Per-letter absorption rates: {letter_absorption_rates}")

        results_per_hierarchy["first_letter"] = {
            "hierarchy": "first_letter",
            "n_total": sum(len(fl_data[l]["child_texts"]) for l in letters if l in fl_data),
            "n_letters": len(letters),
            "absorption_rate": mean_abs_rate,
            "null_rate_random_latents": mean_null_rate,
            "ratio_to_null": ratio,
            "bootstrap_ci_lower": ci_lower,
            "bootstrap_ci_upper": ci_upper,
            "ci_reliable": True,
            "per_letter_absorption": letter_absorption_rates,
            "per_letter_null": letter_null_rates,
            "interpretation": (
                "Absorption rate = fraction of child tokens where concept latents fail to activate. "
                "Null = same metric for random (unrelated) latents. "
                "If ratio < 1.0: concept latents fire LESS than random → absorption is happening. "
                "If ratio ≈ 1.0: no differential signal."
            ),
        }

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 2: noun_proper
# ────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HIERARCHY: noun_proper")

w_np, c_np = load_probe("noun_proper")  # [1, 768], classes=['common_noun', 'proper_noun']
if w_np is not None:
    print(f"noun_proper probe shape: {w_np.shape}, classes: {c_np}")

    # The single weight vector discriminates common_noun from proper_noun
    # For binary probe: w[0] is the discriminative direction
    # Positive direction → more "common_noun" (class 0)
    # Negative direction → more "proper_noun" (class 1)

    probe_vec_common = w_np[0]  # [768] - direction for common_noun vs proper_noun
    probe_vec_proper = -w_np[0]  # reverse for proper_noun direction

    # Find latents for each class direction
    concept_latents_common = find_concept_latents(probe_vec_common, W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)
    concept_latents_proper = find_concept_latents(probe_vec_proper, W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)

    print(f"  common_noun latents: {len(concept_latents_common)}")
    print(f"  proper_noun latents: {len(concept_latents_proper)}")

    # Build test data
    np_data = build_binary_hierarchy_data(proper_nouns, common_nouns, n=35, seed=SEED)
    child_texts = np_data["child_texts"]    # proper nouns = child tokens
    control_texts = np_data["control_texts"]  # common nouns = control

    all_texts_np = list(set(child_texts + control_texts))
    text_to_idx_np = {t: i for i, t in enumerate(all_texts_np)}

    resid_acts_np = get_gpt2_activations_batch(model, all_texts_np, layer=6, batch_size=64, device=DEVICE)
    sae_acts_np = get_sae_activations(sae, resid_acts_np, device=DEVICE)

    child_indices_np = [text_to_idx_np[t] for t in child_texts if t in text_to_idx_np]
    ctrl_indices_np = [text_to_idx_np[t] for t in control_texts if t in text_to_idx_np]

    child_sae_np = sae_acts_np[child_indices_np]
    ctrl_sae_np = sae_acts_np[ctrl_indices_np]

    # Absorption measurement:
    # Proper nouns are "children" - they should activate proper_noun latents (not be absorbed)
    # If proper_noun latents fail to fire for proper nouns → absorption
    proper_latent_indices = [lt[0] for lt in concept_latents_proper]
    common_latent_indices = [lt[0] for lt in concept_latents_common]

    if proper_latent_indices:
        # How often do proper_noun latents fail to fire for proper noun tokens?
        abs_rate_proper, max_acts_proper = measure_absorption_rate(
            proper_latent_indices, child_sae_np, ACTIVATION_THRESHOLD)
        # Random null
        exclude_np = set(proper_latent_indices)
        rand_latents_np = find_random_latents(len(proper_latent_indices), d_sae, exclude_np, SEED_RNG)
        null_rate_np, _ = measure_absorption_rate(rand_latents_np, child_sae_np, ACTIVATION_THRESHOLD)

        # For common nouns - how often do proper_noun latents fail to fire? (should be high - control)
        abs_rate_common_vs_proper, _ = measure_absorption_rate(
            proper_latent_indices, ctrl_sae_np, ACTIVATION_THRESHOLD)

        ratio_np = abs_rate_proper / null_rate_np if null_rate_np > 0 else float("inf")
        print(f"noun_proper: proper_noun latents, proper_noun_tokens absorption_rate={abs_rate_proper:.4f}")
        print(f"           null_rate={null_rate_np:.4f}, ratio={ratio_np:.3f}")
        print(f"           common_noun_tokens vs proper_latents: {abs_rate_common_vs_proper:.4f} (control)")
        print(f"  Max activation of proper_noun latents: mean={np.mean(max_acts_proper):.3f}")

        # Bootstrap CI
        n_child = len(child_indices_np)
        child_absorbed = (child_sae_np[:, proper_latent_indices].max(dim=1).values < ACTIVATION_THRESHOLD).float()
        boot_rates = [float(child_absorbed[SEED_RNG.choice(n_child, n_child, replace=True)].mean())
                      for _ in range(N_BOOTSTRAP)]
        ci_lower_np = float(np.percentile(boot_rates, 2.5))
        ci_upper_np = float(np.percentile(boot_rates, 97.5))

        results_per_hierarchy["noun_proper"] = {
            "hierarchy": "noun_proper",
            "n_child": len(child_indices_np),
            "n_control": len(ctrl_indices_np),
            "n_concept_latents": len(proper_latent_indices),
            "absorption_rate": float(abs_rate_proper),
            "null_rate_random_latents": float(null_rate_np),
            "ratio_to_null": float(ratio_np),
            "control_rate_unrelated_class": float(abs_rate_common_vs_proper),
            "bootstrap_ci_lower": ci_lower_np,
            "bootstrap_ci_upper": ci_upper_np,
            "ci_reliable": len(child_indices_np) >= 10,
            "mean_max_concept_activation": float(np.mean(max_acts_proper)),
            "concept_latents_top5": [(lt[0], float(lt[1])) for lt in concept_latents_proper[:5]],
            "go_nogo": "GO" if ratio_np >= 1.5 else ("WEAK" if ratio_np >= 1.1 else "NO_GO"),
        }
    else:
        print("  No proper_noun latents found at threshold!")
        results_per_hierarchy["noun_proper"] = {
            "error": "No proper_noun latents found at cos_threshold",
            "go_nogo": "NO_GO",
        }

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 3: animate_inanimate
# ────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HIERARCHY: animate_inanimate")

w_ai, c_ai = load_probe("animate_inanimate")
if w_ai is not None:
    print(f"animate_inanimate probe shape: {w_ai.shape}, classes: {c_ai}")

    # Binary probe: w[0] discriminates animate from inanimate
    probe_vec_animate = w_ai[0]
    probe_vec_inanimate = -w_ai[0]

    concept_latents_animate = find_concept_latents(probe_vec_animate, W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)
    concept_latents_inanimate = find_concept_latents(probe_vec_inanimate, W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)

    print(f"  animate latents: {len(concept_latents_animate)}")
    print(f"  inanimate latents: {len(concept_latents_inanimate)}")

    ai_data = build_binary_hierarchy_data(animate_words, inanimate_words, n=35, seed=SEED)
    animate_texts = ai_data["child_texts"]
    inanimate_texts = ai_data["control_texts"]

    all_texts_ai = list(set(animate_texts + inanimate_texts))
    text_to_idx_ai = {t: i for i, t in enumerate(all_texts_ai)}

    resid_acts_ai = get_gpt2_activations_batch(model, all_texts_ai, layer=6, batch_size=64, device=DEVICE)
    sae_acts_ai = get_sae_activations(sae, resid_acts_ai, device=DEVICE)

    animate_indices = [text_to_idx_ai[t] for t in animate_texts if t in text_to_idx_ai]
    inanimate_indices = [text_to_idx_ai[t] for t in inanimate_texts if t in text_to_idx_ai]

    animate_sae = sae_acts_ai[animate_indices]
    inanimate_sae = sae_acts_ai[inanimate_indices]

    if concept_latents_animate:
        animate_latent_indices = [lt[0] for lt in concept_latents_animate]

        # Absorption: animate latents fail to fire for animate words
        abs_rate_ai, max_acts_ai = measure_absorption_rate(
            animate_latent_indices, animate_sae, ACTIVATION_THRESHOLD)
        exclude_ai = set(animate_latent_indices)
        rand_latents_ai = find_random_latents(len(animate_latent_indices), d_sae, exclude_ai, SEED_RNG)
        null_rate_ai, _ = measure_absorption_rate(rand_latents_ai, animate_sae, ACTIVATION_THRESHOLD)

        # Control: animate latents fail to fire for inanimate words (expected to be high)
        ctrl_rate_ai, _ = measure_absorption_rate(animate_latent_indices, inanimate_sae, ACTIVATION_THRESHOLD)

        ratio_ai = abs_rate_ai / null_rate_ai if null_rate_ai > 0 else float("inf")
        print(f"animate_inanimate: animate_latents, animate_tokens absorption_rate={abs_rate_ai:.4f}")
        print(f"                   null_rate={null_rate_ai:.4f}, ratio={ratio_ai:.3f}")
        print(f"                   inanimate_tokens vs animate_latents: {ctrl_rate_ai:.4f} (control)")
        print(f"  Mean max animate activation: {np.mean(max_acts_ai):.3f}")

        n_animate = len(animate_indices)
        animate_absorbed = (animate_sae[:, animate_latent_indices].max(dim=1).values < ACTIVATION_THRESHOLD).float()
        boot_rates_ai = [float(animate_absorbed[SEED_RNG.choice(n_animate, n_animate, replace=True)].mean())
                         for _ in range(N_BOOTSTRAP)]
        ci_lower_ai = float(np.percentile(boot_rates_ai, 2.5))
        ci_upper_ai = float(np.percentile(boot_rates_ai, 97.5))

        results_per_hierarchy["animate_inanimate"] = {
            "hierarchy": "animate_inanimate",
            "n_child": len(animate_indices),
            "n_control": len(inanimate_indices),
            "n_concept_latents": len(animate_latent_indices),
            "absorption_rate": float(abs_rate_ai),
            "null_rate_random_latents": float(null_rate_ai),
            "ratio_to_null": float(ratio_ai),
            "control_rate_unrelated_class": float(ctrl_rate_ai),
            "bootstrap_ci_lower": ci_lower_ai,
            "bootstrap_ci_upper": ci_upper_ai,
            "ci_reliable": n_animate >= 10,
            "mean_max_concept_activation": float(np.mean(max_acts_ai)),
            "concept_latents_top5": [(lt[0], float(lt[1])) for lt in concept_latents_animate[:5]],
            "go_nogo": "GO" if ratio_ai >= 1.5 else ("WEAK" if ratio_ai >= 1.1 else "NO_GO"),
        }
    else:
        results_per_hierarchy["animate_inanimate"] = {"error": "No animate latents found", "go_nogo": "NO_GO"}

# ────────────────────────────────────────────────────────────────────────────────
# HIERARCHY 4: city_country_binary (FLAGGED)
# ────────────────────────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("HIERARCHY: city_country_binary (FLAGGED - failed shuffle gate)")

us_cities = [
    "Chicago", "Houston", "Phoenix", "Philadelphia", "Dallas", "Jacksonville",
    "Austin", "Columbus", "Memphis", "Baltimore", "Milwaukee", "Albuquerque",
    "Tucson", "Fresno", "Sacramento", "Omaha", "Raleigh", "Cleveland", "Tulsa", "Tampa"
]
non_us_cities = [
    "London", "Paris", "Berlin", "Tokyo", "Sydney", "Rome", "Madrid", "Beijing",
    "Moscow", "Cairo", "Delhi", "Seoul", "Bangkok", "Toronto", "Istanbul",
    "Singapore", "Dubai", "Lagos", "Nairobi", "Jakarta"
]

w_cc, c_cc = load_probe("city_country_binary")
if w_cc is not None:
    print(f"city_country_binary probe shape: {w_cc.shape}, classes: {c_cc}")

    # US direction (w[0] points toward US class)
    probe_vec_us = w_cc[0]
    probe_vec_nonus = -w_cc[0]

    concept_latents_us = find_concept_latents(probe_vec_us, W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)
    concept_latents_nonus = find_concept_latents(probe_vec_nonus, W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)
    print(f"  US latents: {len(concept_latents_us)}, non-US latents: {len(concept_latents_nonus)}")

    cc_data = build_binary_hierarchy_data(us_cities, non_us_cities, n=20, seed=SEED)
    us_texts = cc_data["child_texts"]
    nonus_texts = cc_data["control_texts"]

    all_texts_cc = list(set(us_texts + nonus_texts))
    text_to_idx_cc = {t: i for i, t in enumerate(all_texts_cc)}

    resid_acts_cc = get_gpt2_activations_batch(model, all_texts_cc, layer=6, batch_size=64, device=DEVICE)
    sae_acts_cc = get_sae_activations(sae, resid_acts_cc, device=DEVICE)

    us_indices = [text_to_idx_cc[t] for t in us_texts if t in text_to_idx_cc]
    nonus_indices = [text_to_idx_cc[t] for t in nonus_texts if t in text_to_idx_cc]

    us_sae = sae_acts_cc[us_indices]

    if concept_latents_us:
        us_latent_indices = [lt[0] for lt in concept_latents_us]
        abs_rate_cc, max_acts_cc = measure_absorption_rate(us_latent_indices, us_sae, ACTIVATION_THRESHOLD)
        rand_latents_cc = find_random_latents(len(us_latent_indices), d_sae, set(us_latent_indices), SEED_RNG)
        null_rate_cc, _ = measure_absorption_rate(rand_latents_cc, us_sae, ACTIVATION_THRESHOLD)
        ratio_cc = abs_rate_cc / null_rate_cc if null_rate_cc > 0 else float("inf")
        print(f"city_country_binary: US_latents, US_cities absorption_rate={abs_rate_cc:.4f}, null={null_rate_cc:.4f}, ratio={ratio_cc:.3f}")

        results_per_hierarchy["city_country_binary"] = {
            "hierarchy": "city_country_binary",
            "shuffle_gate_warning": "FLAGGED: Failed shuffled-label control in C.1. Results non-inferential.",
            "n_child": len(us_indices),
            "absorption_rate": float(abs_rate_cc),
            "null_rate_random_latents": float(null_rate_cc),
            "ratio_to_null": float(ratio_cc),
            "go_nogo": "FLAGGED",
        }
    else:
        results_per_hierarchy["city_country_binary"] = {
            "error": "No US latents found", "go_nogo": "FLAGGED",
            "shuffle_gate_warning": "FLAGGED: Failed shuffled-label control in C.1.",
        }

# ── Step 7: Summary statistics ────────────────────────────────────────────────

report_progress(5, TOTAL_STEPS, "Computing summary statistics")

# Also compute additional diagnostic: mean max activation for concept latents
# This tells us if the concept latents actually encode anything useful

print("\n" + "="*60)
print("SUMMARY OF RESULTS:")
for h, r in results_per_hierarchy.items():
    if "error" in r:
        print(f"  {h}: ERROR - {r['error']}")
        continue
    abs_rate = r.get("absorption_rate", 0) or 0.0
    null_rate = r.get("null_rate_random_latents", 0) or 0.0
    ratio = r.get("ratio_to_null", 0) or 0.0
    print(f"  {h}: abs_rate={abs_rate:.4f}, null_rate={null_rate:.4f}, ratio={ratio:.3f}, go={r.get('go_nogo')}")

# Pass criterion for pilot
non_spelling = ["noun_proper", "animate_inanimate"]
n_passing = sum(
    1 for h in non_spelling
    if results_per_hierarchy.get(h, {}).get("ratio_to_null", 0) >= 1.5
)
# Also check weaker signal
n_weak = sum(
    1 for h in non_spelling
    if 1.1 <= results_per_hierarchy.get(h, {}).get("ratio_to_null", 0) < 1.5
)

overall_go_nogo = "GO" if n_passing >= 1 else ("WEAK_SIGNAL" if n_weak >= 1 else "NO_GO")

# Compute Bonferroni correction
from scipy import stats as scipy_stats
bonferroni_alpha = 0.05 / len(non_spelling)

# ── Step 8: Diagnostic analysis ───────────────────────────────────────────────

report_progress(6, TOTAL_STEPS, "Diagnostic analysis")

# Key diagnostic: check if concept latents actually respond to semantically relevant words
# by computing mean activation for concept-related vs. unrelated words

print("\nDIAGNOSTIC: Mean SAE activation for concept latents:")
diagnostic_results = {}

# Animate latents: do they fire for animate vs inanimate words?
if "animate_inanimate" in results_per_hierarchy and "error" not in results_per_hierarchy["animate_inanimate"]:
    w_ai, c_ai = load_probe("animate_inanimate")
    animate_latent_indices_diag = [lt[0] for lt in find_concept_latents(w_ai[0], W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)]
    inanimate_latent_indices_diag = [lt[0] for lt in find_concept_latents(-w_ai[0], W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)]

    animate_texts_diag = animate_words[:20]
    inanimate_texts_diag = inanimate_words[:20]
    all_diag_texts = list(set(animate_texts_diag + inanimate_texts_diag))
    diag_idx = {t: i for i, t in enumerate(all_diag_texts)}

    resid_diag = get_gpt2_activations_batch(model, all_diag_texts, layer=6, batch_size=32, device=DEVICE)
    sae_diag = get_sae_activations(sae, resid_diag, device=DEVICE)

    animate_diag_idx = [diag_idx[t] for t in animate_texts_diag if t in diag_idx]
    inanimate_diag_idx = [diag_idx[t] for t in inanimate_texts_diag if t in diag_idx]

    if animate_latent_indices_diag:
        # Mean activation of animate latents for animate vs inanimate words
        mean_animate_for_animate = float(sae_diag[animate_diag_idx][:, animate_latent_indices_diag].mean())
        mean_animate_for_inanimate = float(sae_diag[inanimate_diag_idx][:, animate_latent_indices_diag].mean())
        print(f"  animate latents | animate words: {mean_animate_for_animate:.4f}")
        print(f"  animate latents | inanimate words: {mean_animate_for_inanimate:.4f}")
        print(f"  Selectivity: {mean_animate_for_animate / max(mean_animate_for_inanimate, 1e-6):.2f}x")

        diagnostic_results["animate_inanimate"] = {
            "n_animate_latents": len(animate_latent_indices_diag),
            "mean_animate_latents_for_animate_words": mean_animate_for_animate,
            "mean_animate_latents_for_inanimate_words": mean_animate_for_inanimate,
            "selectivity_ratio": float(mean_animate_for_animate / max(mean_animate_for_inanimate, 1e-6)),
            "animate_latent_top5": animate_latent_indices_diag[:5],
        }

        # Update animate_inanimate result with diagnostic
        results_per_hierarchy["animate_inanimate"]["diagnostic"] = diagnostic_results["animate_inanimate"]

# Noun/proper diagnostic
if "noun_proper" in results_per_hierarchy and "error" not in results_per_hierarchy["noun_proper"]:
    w_np2, c_np2 = load_probe("noun_proper")
    proper_latents_diag = [lt[0] for lt in find_concept_latents(-w_np2[0], W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)]
    common_latents_diag = [lt[0] for lt in find_concept_latents(w_np2[0], W_dec, top_k=TOP_K, cos_threshold=COS_THRESHOLD)]

    pn_texts = proper_nouns[:15]
    cn_texts = common_nouns[:15]
    all_nc_texts = list(set(pn_texts + cn_texts))
    nc_idx = {t: i for i, t in enumerate(all_nc_texts)}

    resid_nc = get_gpt2_activations_batch(model, all_nc_texts, layer=6, batch_size=32, device=DEVICE)
    sae_nc = get_sae_activations(sae, resid_nc, device=DEVICE)

    pn_idx = [nc_idx[t] for t in pn_texts if t in nc_idx]
    cn_idx_list = [nc_idx[t] for t in cn_texts if t in nc_idx]

    if proper_latents_diag:
        mean_proper_for_pn = float(sae_nc[pn_idx][:, proper_latents_diag].mean())
        mean_proper_for_cn = float(sae_nc[cn_idx_list][:, proper_latents_diag].mean())
        print(f"  proper_noun latents | proper nouns: {mean_proper_for_pn:.4f}")
        print(f"  proper_noun latents | common nouns: {mean_proper_for_cn:.4f}")
        print(f"  Selectivity: {mean_proper_for_pn / max(mean_proper_for_cn, 1e-6):.2f}x")

        diagnostic_results["noun_proper"] = {
            "n_proper_latents": len(proper_latents_diag),
            "mean_proper_latents_for_proper_nouns": mean_proper_for_pn,
            "mean_proper_latents_for_common_nouns": mean_proper_for_cn,
            "selectivity_ratio": float(mean_proper_for_pn / max(mean_proper_for_cn, 1e-6)),
            "proper_latent_top5": proper_latents_diag[:5],
        }
        results_per_hierarchy["noun_proper"]["diagnostic"] = diagnostic_results["noun_proper"]

# ── Step 9: Save results ──────────────────────────────────────────────────────

report_progress(7, TOTAL_STEPS, "Saving results")

elapsed = time.time() - start_time

# Build summary
findings = []
for h_name, res in results_per_hierarchy.items():
    if "error" in res:
        findings.append(f"{h_name}: ERROR - {res['error']}")
    else:
        abs_rate = res.get("absorption_rate", 0) or 0.0
        null_rate = res.get("null_rate_random_latents", 0) or 0.0
        ratio = res.get("ratio_to_null", 0) or 0.0
        findings.append(f"{h_name}: absorption_rate={abs_rate:.3f}, null_rate={null_rate:.3f}, ratio={ratio:.2f}, go={res.get('go_nogo')}")

output = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "layer": 6,
        "seed": SEED,
        "cos_threshold": COS_THRESHOLD,
        "top_k_latents": TOP_K,
        "activation_threshold": ACTIVATION_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "bonferroni_alpha": bonferroni_alpha,
        "n_comparisons": len(non_spelling),
    },
    "hierarchies": results_per_hierarchy,
    "diagnostic_analysis": diagnostic_results,
    "pilot_pass_criteria": {
        "pass_criterion": "ratio_to_null >= 1.5 for at least 1 non-spelling hierarchy",
        "non_spelling_hierarchies": non_spelling,
        "n_passing_ratio_1.5": n_passing,
        "n_passing_ratio_1.1": n_weak,
        "overall_go_nogo": overall_go_nogo,
        "findings": findings,
        "interpretation": (
            "ratio_to_null = absorption_rate_concept_latents / absorption_rate_random_latents. "
            "ratio >> 1: concept latents are MORE likely to fail to fire than random latents "
            "→ absorption is happening. "
            "ratio ≈ 1: no differential signal (concept latents not selective). "
            "ratio < 1: concept latents fire MORE reliably than random latents (strong encoding)."
        ),
    },
    "notes": {
        "methodology_v2": (
            "v2 uses random-latent null control instead of parent-class permutation. "
            "Absorption rate = fraction of child tokens where concept latents fail to activate (< threshold). "
            "Null = same metric for randomly sampled SAE latents of equal count. "
            "If absorption is real, concept latents should fail to fire MORE than random latents."
        ),
        "city_country_binary": "FLAGGED: Failed shuffle gate in C.1. Non-inferential.",
        "iter_001_lesson": "All probes trained on GPT-2 Small (same model analyzed). No cross-model transfer.",
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to {OUTPUT_FILE}")

# ── Step 10: Done ─────────────────────────────────────────────────────────────

report_progress(8, TOTAL_STEPS, "Completing")

summary = (
    f"C2 PILOT: {overall_go_nogo}. "
    f"Non-spelling passing (ratio>=1.5): {n_passing}/{len(non_spelling)}, "
    f"weak (ratio>=1.1): {n_weak}. "
    + "; ".join(findings)
)
print(f"\n{'='*60}")
print(f"SUMMARY: {summary}")
print(f"{'='*60}")

report_progress(9, TOTAL_STEPS, "Done")
mark_done("success", summary)
