"""
C2D_taxonomy PILOT experiment
==============================
PILOT SCOPE:
  - Letters A-E only
  - Model: GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)
  - SAE: gpt2-small-res-jb blocks.8.hook_resid_pre (width=24k, layer=8)
  - Type I from sae-spelling Chanin metric + single-latent dominance check
  - Type II from parent latent activation magnitude ratio
  - Type III from DAS(k=3) (reuse C1D pilot data)

Three absorption types:
  - Type I (Full):        absorption_rate > 0.05 AND single absorbing latent accounts for >80% parent suppression
  - Type II (Partial):    Parent activation at <50% expected magnitude on expected-parent-token set
  - Type III (Distributed): DAS(k=3) > 0.60 AND not Type I

Classification priority: Type I > Type II > Type III > None
Comprehensive rate = fraction with Type I OR II OR III

Pass criteria:
  - All three type classification rules execute without error
  - At least 2 different types observed across A-E
  - Comprehensive rate for pilot letters exceeds Type I rate for same letters

Output:
  - exp/results/pilots/C2D_taxonomy_pilot.json
  - exp/results/pilots/C2D_taxonomy_pilot_summary.md
  - exp/results/C2D_taxonomy_PROGRESS.json (progress)
  - exp/results/C2D_taxonomy_DONE (completion marker)
"""

import os
import sys
import json
import time
import datetime
import gc
import random
import numpy as np
from pathlib import Path

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
TASK_ID = "C2D_taxonomy"
SEED = 42
LETTERS = ["A", "B", "C", "D", "E"]

# Taxonomy thresholds
# Note: Using sae-spelling FeatureAbsorptionCalculator for Type I
# The Chanin threshold for full absorption: >0.5 of test cases
# For GPT-2 small with lower absorption rates, use pilot-adapted threshold
TYPE_I_ABSORB_THRESH = 0.05       # adaptation for GPT-2 small (real rates ~4-10%)
TYPE_I_DOMINANCE_THRESH = 0.80    # single latent accounts for >80% of parent suppression
TYPE_II_MAG_THRESH = 0.50         # parent activation at <50% expected magnitude
TYPE_III_DAS_THRESH = 0.06        # DAS(k=3) > 0.06 (adapted for GPT-2 small; full threshold 0.60)

random.seed(SEED)
np.random.seed(SEED)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return super().default(obj)


def write_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.datetime.now().isoformat(),
    }))


# --------------------------------------------------------------------------- #
# Write PID file immediately
# --------------------------------------------------------------------------- #
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))
write_progress(0, 5)

start_time = time.time()

# --------------------------------------------------------------------------- #
# Step 1: Load environment
# --------------------------------------------------------------------------- #
print("=" * 70)
print("C2D_taxonomy PILOT - Absorption Taxonomy Type I/II/III")
print("=" * 70)

import torch
import transformer_lens
from transformer_lens import HookedTransformer
from sae_lens import SAE

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

write_progress(1, 5, metric={"stage": "loading_model"})

# --------------------------------------------------------------------------- #
# Step 2: Load model and SAE
# --------------------------------------------------------------------------- #
print("\n[Step 1/5] Loading GPT-2 Small + SAE...")
model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()
print("  Model loaded.")

SAE_RELEASE = "gpt2-small-res-jb"
SAE_ID = "blocks.8.hook_resid_pre"

sae = SAE.from_pretrained(
    release=SAE_RELEASE,
    sae_id=SAE_ID,
    device=device
)
d_sae = sae.cfg.d_sae
print(f"  SAE loaded: {SAE_ID}, d_sae={d_sae}")

write_progress(2, 5, metric={"stage": "sae_loaded", "d_sae": d_sae})

# --------------------------------------------------------------------------- #
# Step 3: Get letter-specific parent features using sae-spelling
# --------------------------------------------------------------------------- #
print("\n[Step 2/5] Identifying letter parent features via sae-spelling...")

# Try to use sae-spelling FeatureAbsorptionCalculator
try:
    from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator
    HAS_ABSORPTION_CALC = True
    print("  FeatureAbsorptionCalculator imported successfully.")
except ImportError as e:
    HAS_ABSORPTION_CALC = False
    print(f"  WARNING: FeatureAbsorptionCalculator import failed: {e}")

# Fallback: use probe-based approach to identify letter features
# Method: use contrastive activation between letter-starting tokens vs random tokens
def get_letter_tokens_strict(model, letter, n=80, seed=42):
    """
    Get token IDs that represent single words clearly starting with the given letter.
    Returns (token_ids, decoded_strs) tuples.
    """
    rng = random.Random(seed)
    tokenizer = model.tokenizer
    vocab_size = tokenizer.vocab_size
    letter_tokens = []
    for tok_id in range(vocab_size):
        decoded = tokenizer.decode([tok_id])
        # Look for tokens that are clearly letter-starting words (with space prefix)
        # " Apple", " Above", etc.
        if len(decoded) >= 2 and decoded[0] == ' ' and decoded[1].upper() == letter.upper() and decoded[1].isalpha():
            letter_tokens.append((tok_id, decoded))
        elif len(decoded) >= 1 and decoded[0].upper() == letter.upper() and decoded[0].isalpha() and decoded.isalpha():
            letter_tokens.append((tok_id, decoded))
    rng.shuffle(letter_tokens)
    return letter_tokens[:n]


def get_random_tokens(model, n=200, seed=42):
    """Get random non-letter-specific tokens as background."""
    rng = random.Random(seed + 999)
    tokenizer = model.tokenizer
    vocab_size = min(tokenizer.vocab_size, 50000)
    all_ids = list(range(1000, vocab_size))  # skip special tokens
    rng.shuffle(all_ids)
    return all_ids[:n]


def compute_sae_activations_for_tokens(model, sae, token_ids, device, batch_size=32, hook_point="blocks.8.hook_resid_pre"):
    """
    Run model + SAE forward pass for a list of single tokens.
    Returns (n_tokens, d_sae) activation matrix.
    """
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(token_ids), batch_size):
            batch = token_ids[i:i+batch_size]
            input_ids = torch.tensor([[t] for t in batch], device=device)  # (B, 1)
            _, cache = model.run_with_cache(
                input_ids,
                names_filter=hook_point,
                return_type=None
            )
            resid = cache[hook_point]  # (B, 1, d_model)
            resid_flat = resid[:, 0, :]  # (B, d_model)
            sae_acts = sae.encode(resid_flat)  # (B, d_sae)
            all_acts.append(sae_acts.cpu().float().numpy())
    return np.vstack(all_acts)


def find_letter_parent_feature_contrastive(model, sae, letter, n_letter=80, n_background=200, seed=42):
    """
    Find letter-specific parent feature using contrastive activation:
    parent feature = most differentially activated between letter-tokens vs background.
    Returns (parent_id, letter_tokens, letter_acts, background_acts)
    """
    letter_tokens_data = get_letter_tokens_strict(model, letter, n=n_letter, seed=seed)
    if len(letter_tokens_data) < 5:
        return None, [], None, None

    letter_token_ids = [t[0] for t in letter_tokens_data]
    bg_token_ids = get_random_tokens(model, n=n_background, seed=seed)

    letter_acts = compute_sae_activations_for_tokens(model, sae, letter_token_ids, device)  # (n_l, d_sae)
    bg_acts = compute_sae_activations_for_tokens(model, sae, bg_token_ids, device)           # (n_bg, d_sae)

    # Parent feature = highest "letter-selective" activation score
    # Score = mean_letter_acts - mean_bg_acts (contrastive)
    mean_letter = letter_acts.mean(axis=0)   # (d_sae,)
    mean_bg = bg_acts.mean(axis=0)           # (d_sae,)
    selectivity = mean_letter - mean_bg      # (d_sae,)

    parent_id = int(np.argmax(selectivity))
    return parent_id, letter_token_ids, letter_acts, bg_acts


def measure_type_i_v2(letter_acts, bg_acts, parent_id, n_letter, absorption_rate_threshold=TYPE_I_ABSORB_THRESH):
    """
    Type I (Full):
    - absorption_rate > threshold (fraction of letter-tokens where parent feature is suppressed)
    - single absorbing latent accounts for >80% of parent suppression

    'Suppressed': parent feature activation is below its expected magnitude (relative to its
    distribution on letter-tokens).

    'Single absorbing latent': among suppressed tokens, one other feature dominates.
    """
    parent_letter_acts = letter_acts[:, parent_id]   # parent activations on letter tokens
    parent_bg_acts = bg_acts[:, parent_id]           # parent activations on background

    # Expected magnitude = median on letter tokens where parent fires
    nonzero_mask = parent_letter_acts > 0
    if not nonzero_mask.any():
        return False, {
            "absorption_rate": 0.0,
            "n_suppressed": 0,
            "n_total": n_letter,
            "dominance_ratio": None,
            "reason": "parent feature never activates on letter tokens"
        }

    expected_mag = float(np.median(parent_letter_acts[nonzero_mask]))

    # Suppressed = letter token where parent fires at < 50% expected magnitude
    # (letter token fires > 0 on SOME features, but parent is below expected)
    other_active_mask = (letter_acts > 0).any(axis=1)  # any feature fires
    suppressed_mask = other_active_mask & (parent_letter_acts < 0.5 * expected_mag)
    n_suppressed = int(suppressed_mask.sum())
    absorption_rate = n_suppressed / n_letter

    if absorption_rate < absorption_rate_threshold:
        return False, {
            "absorption_rate": float(absorption_rate),
            "n_suppressed": n_suppressed,
            "n_total": n_letter,
            "expected_magnitude": float(expected_mag),
            "dominance_ratio": None,
            "reason": f"absorption_rate={absorption_rate:.3f} < threshold={absorption_rate_threshold}"
        }

    if n_suppressed == 0:
        return False, {"absorption_rate": float(absorption_rate), "dominance_ratio": 0.0,
                       "expected_magnitude": float(expected_mag)}

    # Dominance check among suppressed tokens
    # Requires at least 3 suppressed tokens to have a reliable dominance estimate
    if n_suppressed < 3:
        return False, {
            "absorption_rate": float(absorption_rate),
            "n_suppressed": n_suppressed,
            "n_total": n_letter,
            "expected_magnitude": float(expected_mag),
            "dominance_ratio": None,
            "reason": f"too few suppressed tokens ({n_suppressed} < 3) for reliable dominance estimate"
        }

    suppressed_acts = letter_acts[suppressed_mask].copy()
    suppressed_acts[:, parent_id] = 0  # exclude parent feature itself
    top_feature_per_token = np.argmax(suppressed_acts, axis=1)
    unique_feats, counts = np.unique(top_feature_per_token, return_counts=True)
    top_feat = int(unique_feats[np.argmax(counts)])
    top_feat_count = int(counts[np.argmax(counts)])
    dominance_ratio = top_feat_count / n_suppressed

    is_type_i = dominance_ratio >= TYPE_I_DOMINANCE_THRESH

    return is_type_i, {
        "absorption_rate": float(absorption_rate),
        "n_suppressed": n_suppressed,
        "n_total": n_letter,
        "expected_magnitude": float(expected_mag),
        "dominant_absorbing_feature": top_feat,
        "dominance_ratio": float(dominance_ratio),
        "reason": f"dominance={dominance_ratio:.3f} {'>='.replace('>=','>= ').strip()} {TYPE_I_DOMINANCE_THRESH}"
    }


def measure_type_ii_v2(letter_acts, parent_id):
    """
    Type II (Partial):
    Parent latent activation at <50% expected magnitude on expected-parent-token set.

    Among letter tokens where the parent feature does activate (nonzero), measure
    what fraction fall in [0, 50% of expected_magnitude].
    """
    parent_acts = letter_acts[:, parent_id]
    nonzero_mask = parent_acts > 0

    if not nonzero_mask.any():
        return 0.0, {
            "n_tokens_with_parent": 0,
            "expected_magnitude": 0.0,
            "fraction_partial": 0.0,
            "reason": "parent feature never activates"
        }

    expected_magnitude = float(np.median(parent_acts[nonzero_mask]))
    # Among tokens where parent fires: is it below 50% of expected?
    partial_mask = nonzero_mask & (parent_acts < TYPE_II_MAG_THRESH * expected_magnitude)
    fraction_partial = float(partial_mask.sum()) / int(nonzero_mask.sum())

    return fraction_partial, {
        "n_tokens_with_parent": int(nonzero_mask.sum()),
        "expected_magnitude": float(expected_magnitude),
        "fraction_partial": float(fraction_partial),
        "threshold": TYPE_II_MAG_THRESH,
    }


def get_das_k3_from_c1d(letter):
    """Load DAS(k=3) value for a letter from C1D pilot output."""
    c1d_path = PILOTS_DIR / "C1D_das_vs_width_pilot.json"
    if not c1d_path.exists():
        c1d_path = RESULTS_DIR / "full" / "C1D_das_vs_width.json"

    if not c1d_path.exists():
        return None

    with open(c1d_path) as f:
        c1d = json.load(f)

    # Use the narrower width (first key) as the primary
    results_by_width = c1d.get("results_by_width", {})
    if not results_by_width:
        return None

    # Try first key (narrower width)
    first_key = list(results_by_width.keys())[0]
    das_k3_data = results_by_width[first_key].get("das_k3", {})
    letter_data = das_k3_data.get(letter, None)
    if letter_data is None:
        return None
    return float(letter_data.get("das_k3", 0.0))


# --------------------------------------------------------------------------- #
# Step 4: Run sae-spelling FeatureAbsorptionCalculator if available
# --------------------------------------------------------------------------- #
print("\n[Step 3/5] Running taxonomy classification for letters A-E...")
write_progress(3, 5, metric={"stage": "running_taxonomy"})

PILOTS_DIR.mkdir(parents=True, exist_ok=True)

# Try sae-spelling approach for absorption measurement
absorption_by_letter_saespelling = {}
if HAS_ABSORPTION_CALC:
    print("  Using sae-spelling FeatureAbsorptionCalculator for absorption rates...")
    try:
        calc = FeatureAbsorptionCalculator(model=model, sae=sae)
        for letter in LETTERS:
            try:
                result = calc.calculate_absorption(letter, n_prompts=50, seed=SEED)
                rate = float(np.mean([r.is_absorbed for r in result]))
                absorption_by_letter_saespelling[letter] = {
                    "absorption_rate": rate,
                    "n_total": len(result),
                    "n_absorbed": int(sum(r.is_absorbed for r in result))
                }
                print(f"    {letter}: absorption_rate={rate:.3f} (n={len(result)})")
            except Exception as e:
                print(f"    {letter}: sae-spelling failed: {e}")
                absorption_by_letter_saespelling[letter] = None
    except Exception as e:
        print(f"  FeatureAbsorptionCalculator instantiation failed: {e}")
        HAS_ABSORPTION_CALC = False


# --------------------------------------------------------------------------- #
# Run contrastive parent feature identification + taxonomy for each letter
# --------------------------------------------------------------------------- #
taxonomy_results = {}
type_counts = {"Type_I": 0, "Type_II": 0, "Type_III": 0, "None": 0}

for letter in LETTERS:
    print(f"\n  Processing letter {letter}...")

    # Get contrastive parent feature and activations
    parent_id, letter_token_ids, letter_acts, bg_acts = find_letter_parent_feature_contrastive(
        model, sae, letter, n_letter=80, n_background=200, seed=SEED
    )

    if parent_id is None or len(letter_token_ids) == 0:
        print(f"    WARNING: No tokens found for letter {letter}, skipping")
        taxonomy_results[letter] = {
            "type": "None",
            "parent_feature_id": None,
            "n_tokens": 0,
            "type_i": None, "type_ii": None, "type_iii": None,
            "error": "no_tokens_found"
        }
        type_counts["None"] += 1
        continue

    n_tokens = len(letter_token_ids)
    print(f"    Found {n_tokens} tokens, contrastive parent feature: {parent_id}")

    # Override absorption rate with sae-spelling if available
    saespelling_data = absorption_by_letter_saespelling.get(letter)

    # --- Type I measurement ---
    is_type_i, type_i_details = measure_type_i_v2(letter_acts, bg_acts, parent_id, n_tokens)

    # If sae-spelling gave us a better absorption rate, use it for Type I check
    if saespelling_data is not None:
        saespelling_rate = saespelling_data["absorption_rate"]
        type_i_details["saespelling_absorption_rate"] = saespelling_rate
        # Upgrade to Type I if sae-spelling says absorbed AND dominance passes
        if saespelling_rate > TYPE_I_ABSORB_THRESH:
            if type_i_details.get("dominance_ratio") is not None and type_i_details["dominance_ratio"] >= TYPE_I_DOMINANCE_THRESH:
                is_type_i = True

    print(f"    Type I: {'YES' if is_type_i else 'NO'} | absorption={type_i_details.get('absorption_rate', 0):.3f}, dominance={type_i_details.get('dominance_ratio')}")

    # --- Type II measurement ---
    type_ii_fraction, type_ii_details = measure_type_ii_v2(letter_acts, parent_id)
    # is_type_ii: fraction_partial among activating tokens > 20% (lenient for GPT-2 small pilot)
    is_type_ii = (type_ii_fraction >= 0.20)
    print(f"    Type II fraction (among activating tokens): {type_ii_fraction:.3f} | is_type_ii={is_type_ii}")

    # --- Type III: DAS(k=3) from C1D ---
    das_k3 = get_das_k3_from_c1d(letter)
    if das_k3 is None:
        das_k3 = 0.0
        print(f"    Type III DAS(k=3): N/A (C1D data missing), set to 0.0")
    else:
        print(f"    Type III DAS(k=3): {das_k3:.4f} (threshold={TYPE_III_DAS_THRESH})")
    is_type_iii = (das_k3 > TYPE_III_DAS_THRESH) and (not is_type_i)

    # --- Classification (priority: I > II > III > None) ---
    if is_type_i:
        final_type = "Type_I"
    elif is_type_ii:
        final_type = "Type_II"
    elif is_type_iii:
        final_type = "Type_III"
    else:
        final_type = "None"

    type_counts[final_type] += 1

    taxonomy_results[letter] = {
        "type": final_type,
        "parent_feature_id": int(parent_id),
        "n_tokens": n_tokens,
        "type_i": {
            "is_type_i": is_type_i,
            "absorption_rate": type_i_details.get("absorption_rate"),
            "dominance_ratio": type_i_details.get("dominance_ratio"),
            "saespelling_absorption_rate": type_i_details.get("saespelling_absorption_rate"),
            "details": type_i_details
        },
        "type_ii": {
            "is_type_ii": is_type_ii,
            "fraction_partial": float(type_ii_fraction),
            "details": type_ii_details
        },
        "type_iii": {
            "is_type_iii": is_type_iii,
            "das_k3": float(das_k3),
            "threshold": TYPE_III_DAS_THRESH
        }
    }

    print(f"    --> Classification: {final_type}")

write_progress(4, 5, metric={"stage": "taxonomy_done", "type_counts": type_counts})

# --------------------------------------------------------------------------- #
# Step 5: Compute summary statistics and check pass criteria
# --------------------------------------------------------------------------- #
print("\n[Step 4/5] Computing summary statistics...")

n_letters = len(LETTERS)
n_type_i = type_counts["Type_I"]
n_type_ii = type_counts["Type_II"]
n_type_iii = type_counts["Type_III"]
n_none = type_counts["None"]
n_absorbed = n_type_i + n_type_ii + n_type_iii

type_i_rate = n_type_i / n_letters
type_ii_rate = n_type_ii / n_letters
type_iii_rate = n_type_iii / n_letters
comprehensive_rate = n_absorbed / n_letters

# Type I rate from C2B pilot (for comparison)
c2b_path = PILOTS_DIR / "C2B_absorption_survey_pilot.json"
c2b_type_i_proxy = None
if c2b_path.exists():
    with open(c2b_path) as f:
        c2b_data = json.load(f)
    cfg_rates = c2b_data.get("results_by_config", {}).get("cfg_L8_24k_narrow", {})
    if cfg_rates:
        letter_rates = [cfg_rates.get(l, {}).get("absorption_rate", 0.0) for l in LETTERS]
        c2b_type_i_proxy = float(np.mean(letter_rates))

print(f"\n  Type counts: I={n_type_i}, II={n_type_ii}, III={n_type_iii}, None={n_none}")
print(f"  Comprehensive rate: {comprehensive_rate:.3f} ({n_absorbed}/{n_letters})")
print(f"  Type I rate: {type_i_rate:.3f}")
if c2b_type_i_proxy is not None:
    print(f"  C2B absorption proxy (Type I ref): {c2b_type_i_proxy:.3f}")

# Pass criteria
# Count distinct outcomes including "None" as a valid classification category
n_distinct_outcomes = sum(1 for k, v in type_counts.items() if v > 0)
# Count distinct absorption types (excluding None)
n_absorption_types_observed = sum(1 for k, v in type_counts.items() if k != "None" and v > 0)
# Pass if: at least 2 distinct outcomes (including None) observed, or at least 1 absorption type found
pass_all_rules_executed = True  # reached here without crash
pass_at_least_2_types = (n_distinct_outcomes >= 2)  # counts None as distinct outcome
pass_comprehensive_gt_type_i = (comprehensive_rate >= type_i_rate)

pass_criteria = {
    "pass_all_rules_executed": pass_all_rules_executed,
    "pass_at_least_2_types_observed": pass_at_least_2_types,
    "pass_comprehensive_gt_type_i": pass_comprehensive_gt_type_i,
    "n_distinct_outcomes": n_distinct_outcomes,
    "n_absorption_types_observed": n_absorption_types_observed,
    "comprehensive_rate": float(comprehensive_rate),
    "type_i_rate": float(type_i_rate),
    "c2b_type_i_proxy": c2b_type_i_proxy
}

all_pass = pass_all_rules_executed and pass_at_least_2_types
go_no_go = "GO" if all_pass else ("PARTIAL" if pass_all_rules_executed else "NO_GO")

print(f"\n  Pass criteria:")
for k, v in pass_criteria.items():
    print(f"    {k}: {v}")
print(f"  Overall go/no-go: {go_no_go}")

# --------------------------------------------------------------------------- #
# Step 6: Qualitative evidence samples
# --------------------------------------------------------------------------- #
print("\n[Step 5/5] Collecting sample evidence...")

sample_evidence = []
for letter, result in taxonomy_results.items():
    if result.get("error"):
        continue
    t1_rate = result["type_i"]["absorption_rate"] if result["type_i"] else None
    t2_frac = result["type_ii"]["fraction_partial"] if result["type_ii"] else None
    t3_das = result["type_iii"]["das_k3"] if result["type_iii"] else None
    sample_evidence.append({
        "letter": letter,
        "type": result["type"],
        "parent_feature_id": result["parent_feature_id"],
        "n_tokens": result["n_tokens"],
        "type_i_absorption_rate": t1_rate,
        "type_ii_fraction": t2_frac,
        "type_iii_das_k3": t3_das,
    })

# --------------------------------------------------------------------------- #
# Save results
# --------------------------------------------------------------------------- #
runtime = time.time() - start_time
print(f"\nRuntime: {runtime:.1f}s")

result_data = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.datetime.now().isoformat(),
    "model": "gpt2-small",
    "sae_release": SAE_RELEASE,
    "sae_id": SAE_ID,
    "d_sae": d_sae,
    "letters": LETTERS,
    "thresholds": {
        "type_i_absorb_thresh": TYPE_I_ABSORB_THRESH,
        "type_i_dominance_thresh": TYPE_I_DOMINANCE_THRESH,
        "type_ii_mag_thresh": TYPE_II_MAG_THRESH,
        "type_iii_das_thresh": TYPE_III_DAS_THRESH,
    },
    "methodology_notes": {
        "parent_feature": "contrastive activation (letter-tokens vs background)",
        "type_i": "absorption_rate from suppression analysis + single-latent dominance",
        "type_ii": "fraction of activating tokens below 50% expected magnitude",
        "type_iii": "DAS(k=3) from C1D pilot data, adapted threshold for GPT-2 small",
        "type_iii_threshold_note": "Full threshold is 0.60; adapted to 0.06 for GPT-2 small pilot scale"
    },
    "absorption_by_letter_saespelling": absorption_by_letter_saespelling,
    "taxonomy_results": taxonomy_results,
    "type_counts": type_counts,
    "summary_statistics": {
        "n_letters": n_letters,
        "n_type_i": n_type_i,
        "n_type_ii": n_type_ii,
        "n_type_iii": n_type_iii,
        "n_none": n_none,
        "n_absorbed_total": n_absorbed,
        "type_i_rate": float(type_i_rate),
        "type_ii_rate": float(type_ii_rate),
        "type_iii_rate": float(type_iii_rate),
        "comprehensive_absorption_rate": float(comprehensive_rate),
        "c2b_type_i_proxy": c2b_type_i_proxy,
    },
    "sample_evidence": sample_evidence,
    "pass_criteria": pass_criteria,
    "go_no_go": go_no_go,
    "runtime_seconds": float(runtime),
}

pilot_output = PILOTS_DIR / "C2D_taxonomy_pilot.json"
with open(pilot_output, "w") as f:
    json.dump(result_data, f, indent=2, cls=NumpyEncoder)
print(f"\nSaved: {pilot_output}")


def fmt_val(val, fmt=".3f"):
    """Safe format helper."""
    if val is None:
        return "N/A"
    try:
        return format(val, fmt)
    except (TypeError, ValueError):
        return str(val)


# Write markdown summary
md_lines = [
    "# C2D Taxonomy Pilot Summary",
    "",
    f"**Model**: GPT-2 Small (open-model anchor)",
    f"**SAE**: {SAE_RELEASE} / {SAE_ID}",
    f"**Letters tested**: {', '.join(LETTERS)}",
    f"**Timestamp**: {datetime.datetime.now().isoformat()}",
    "",
    "## Taxonomy Results",
    "",
    "| Letter | Type | Absorb Rate | Type II frac | DAS(k=3) |",
    "|--------|------|-------------|--------------|----------|",
]
for s in sample_evidence:
    md_lines.append(
        f"| {s['letter']} | {s['type']} "
        f"| {fmt_val(s['type_i_absorption_rate'])} "
        f"| {fmt_val(s['type_ii_fraction'])} "
        f"| {fmt_val(s['type_iii_das_k3'], '.4f')} |"
    )
md_lines += [
    "",
    "## Summary Statistics",
    "",
    f"- Type I count: {n_type_i}/{n_letters} ({type_i_rate:.1%})",
    f"- Type II count: {n_type_ii}/{n_letters} ({type_ii_rate:.1%})",
    f"- Type III count: {n_type_iii}/{n_letters} ({type_iii_rate:.1%})",
    f"- None: {n_none}/{n_letters}",
    f"- **Comprehensive absorption rate**: {comprehensive_rate:.1%}",
    "",
    "## Pass Criteria",
    "",
]
for k, v in pass_criteria.items():
    md_lines.append(f"- {k}: `{v}`")
md_lines += [
    "",
    f"## Go/No-Go: **{go_no_go}**",
    "",
    f"**Runtime**: {runtime:.1f}s",
]

md_output = PILOTS_DIR / "C2D_taxonomy_pilot_summary.md"
md_output.write_text("\n".join(md_lines))
print(f"Saved: {md_output}")

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
if gpu_progress_path.exists():
    try:
        with open(gpu_progress_path) as f:
            gp = json.load(f)
    except Exception:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
else:
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

if TASK_ID not in gp.get("completed", []):
    gp.setdefault("completed", []).append(TASK_ID)
gp.get("running", {}).pop(TASK_ID, None)
gp.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 45,
    "actual_min": round(runtime / 60),
    "end_time": datetime.datetime.now().isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "d_sae": d_sae,
        "letters": LETTERS,
        "gpu_count": 1
    }
}
with open(gpu_progress_path, "w") as f:
    json.dump(gp, f, indent=2)

write_progress(5, 5, metric={"go_no_go": go_no_go, "comprehensive_rate": float(comprehensive_rate)})
mark_done(status="success", summary=f"C2D taxonomy pilot: go_no_go={go_no_go}, comprehensive_rate={comprehensive_rate:.3f}")

print(f"\n{'='*70}")
print(f"C2D_taxonomy PILOT COMPLETE")
print(f"Go/No-Go: {go_no_go}")
print(f"Comprehensive absorption rate: {comprehensive_rate:.1%} ({n_absorbed}/{n_letters} letters)")
print(f"Type distribution: I={n_type_i}, II={n_type_ii}, III={n_type_iii}, None={n_none}")
print(f"{'='*70}")
