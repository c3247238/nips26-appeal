"""
Task B2 (PILOT): Sparsity-Absorption Scaling Curve

For all available GPT-2 Small SAEs at different L0 values (layer 6, from SAELens suite),
measure absorption rate on first-letter task (100 tokens/letter minimum, seed 42).
Also supplement with feature-splitting release (different widths at L8).

Plot absorption rate vs. 1/L0. Fit three functional forms:
  - linear y=a*x+b
  - power-law y=a*x^b
  - sigmoid y=L/(1+exp(-k*(x-x0)))
Compute BIC, AIC, R^2 for each. LRT: sigmoid vs. linear.
Report inflection point L0_c with 95% bootstrap CI (1000 resamples).

PILOT MODE: Use 100 tokens per letter (fast sampling).
Pass criteria: At least 3 different L0 settings measured.

Output: exp/results/pilots/pilot_B2_scaling_curve.json
         exp/results/full/B2_scaling_curve.json (if pilot passes)
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
from scipy import stats, optimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_B2_scaling_curve"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE_PILOT = RESULTS_DIR / "pilot_B2_scaling_curve.json"
OUTPUT_FILE_FULL = FULL_RESULTS_DIR / "B2_scaling_curve.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


def report_progress(step, total_steps, note=""):
    elapsed = time.time() - start_time
    progress = {
        "task_id": TASK_ID, "step": step, "total_steps": total_steps,
        "elapsed_sec": elapsed, "note": note, "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"[{elapsed:.1f}s] Step {step}/{total_steps}: {note}")
    sys.stdout.flush()


def mark_done(status="success", summary="", result=None):
    PID_FILE.unlink(missing_ok=True)
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "result": result, "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
    }))


TOTAL_STEPS = 12
report_progress(0, TOTAL_STEPS, "Starting B2 Sparsity-Absorption Scaling Curve")

# ─── Step 1: Load GPT-2 Small model ────────────────────────────────────────
report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small transformer model")

from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2", center_writing_weights=True)
model.eval()
model.to(DEVICE)
tokenizer = model.tokenizer
print(f"Model loaded: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

# ─── Step 2: Load OpenWebText tokens for co-activation frequency ────────────
report_progress(2, TOTAL_STEPS, "Loading OpenWebText tokens for frequency measurement")

from datasets import load_dataset

# Load small sample - 100 tokens per letter is the pilot requirement
# We use ~2000 tokens from OpenWebText to get enough letter occurrences
print("Loading OpenWebText sample...")

# Try cached version first
OWT_CACHE = WORKSPACE / "exp" / "results" / "owt_tokens_cache.pt"
if OWT_CACHE.exists():
    print("Using cached OWT tokens")
    owt_tokens = torch.load(OWT_CACHE)
    print(f"Cached tokens: {owt_tokens.shape}")
else:
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    all_text = ""
    for item in dataset:
        all_text += item["text"] + " "
        if len(all_text) > 50000:
            break

    tokens = model.to_tokens(all_text[:50000], prepend_bos=True)
    owt_tokens = tokens[:, :5000]  # Take first 5000 tokens
    torch.save(owt_tokens, OWT_CACHE)
    print(f"Saved OWT tokens: {owt_tokens.shape}")

print(f"OWT tokens shape: {owt_tokens.shape}")

# ─── Step 3: Define SAE configurations to test ─────────────────────────────
report_progress(3, TOTAL_STEPS, "Defining SAE configurations for scaling curve")

# Strategy: Use gpt2-small-res-jb at different layers
# Each layer has a different empirical L0 (measured from sparsity tensor)
# This gives us multiple (L0, absorption_rate) data points

# Primary: gpt2-small-res-jb layers 0, 2, 4, 6, 8, 10 (width 24576)
# Secondary: gpt2-small-res-jb-feature-splitting layer 8 (varying widths)
# Focus on layers where we can identify letter features

SAE_CONFIGS_TO_TEST = [
    # (release, sae_id, layer, width, use_for_primary)
    ("gpt2-small-res-jb", "blocks.6.hook_resid_pre", 6, 24576, True),
    ("gpt2-small-res-jb", "blocks.10.hook_resid_pre", 10, 24576, True),
    ("gpt2-small-res-jb", "blocks.8.hook_resid_pre", 8, 24576, True),
    ("gpt2-small-res-jb", "blocks.4.hook_resid_pre", 4, 24576, True),
    ("gpt2-small-res-jb", "blocks.2.hook_resid_pre", 2, 24576, True),
    ("gpt2-small-res-jb", "blocks.0.hook_resid_pre", 0, 24576, False),  # Layer 0 may not have letter features
    # Feature splitting at layer 8 (varying widths - for width analysis)
    ("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_12288", 8, 12288, False),
    ("gpt2-small-res-jb-feature-splitting", "blocks.8.hook_resid_pre_49152", 8, 49152, False),
]

PILOT_TOKENS_PER_LETTER = 100  # Pilot mode
MIN_LETTERS_FOR_RATE = 5  # Need at least this many letters with enough tokens


# ─── Step 4: Letter-feature identification helper ───────────────────────────
report_progress(4, TOTAL_STEPS, "Defining letter-feature identification function")

def identify_letter_features(sae_w_dec, d_model, device, method="probe_decoder"):
    """
    Identify which SAE features are 'letter features' (first-letter features).
    Uses the decoder-cosine method from the pilot:
    features whose decoder vector is highly aligned with first-PCA direction of letter tokens.

    Returns dict: {letter: [feature_ids]}
    """
    if method == "probe_decoder":
        # We use the simple approach: for each letter, find features with high
        # decoder cosine similarity to the mean activation direction
        # This is a lightweight version; full version would train LR probes
        # For now, use threshold-based: cos > 0.30 with common letter embeddings

        # Get embedding vectors for letter tokens (single uppercase/lowercase)
        letter_feature_dict = {}
        sae_w_dec_norm = F.normalize(sae_w_dec, dim=-1)  # (d_sae, d_model)

        for letter in "abcdefghijklmnopqrstuvwxyz":
            letter_tokens = []
            for prefix in [" " + letter.upper(), " " + letter.lower(),
                           " " + letter.upper() + " ", letter.upper()]:
                try:
                    t = tokenizer.encode(prefix)
                    if len(t) == 1 or (len(t) == 2 and t[0] == tokenizer.bos_token_id):
                        letter_tokens.extend(t[-1:])
                except:
                    pass

            if not letter_tokens:
                continue

            # Get embedding vectors for these tokens
            embed_vecs = model.embed.W_E[letter_tokens].to(device)  # (n_toks, d_model)
            mean_embed = embed_vecs.mean(0)
            mean_embed_norm = F.normalize(mean_embed.unsqueeze(0), dim=-1)

            # Cosine similarity with all decoder directions
            cos_sims = (sae_w_dec_norm @ mean_embed_norm.T).squeeze()  # (d_sae,)

            # Features with cos > threshold are considered letter features
            threshold = 0.25
            letter_feats = torch.where(cos_sims > threshold)[0].cpu().numpy().tolist()

            if letter_feats:
                letter_feature_dict[letter] = letter_feats

        return letter_feature_dict

    return {}


def get_letter_tokens(model, tokenizer, letter, n_tokens=100):
    """Get token IDs that start with a given letter."""
    vocab = tokenizer.get_vocab()
    result = []
    for tok_str, tok_id in vocab.items():
        # Check if token starts with the letter (case insensitive)
        clean = tok_str.lstrip("Ġ▁ ")  # Remove BPE prefix chars
        if clean and clean[0].lower() == letter.lower():
            result.append(tok_id)
    return result[:n_tokens]


def measure_absorption_rate(model, sae, layer, owt_tokens, device,
                            n_tokens_per_letter=100, seed=42,
                            max_letters=20, max_seq_len=256):
    """
    Measure absorption rate on first-letter task.

    Absorption rate = fraction of letter tokens where the letter-associated SAE feature
    fires LESS than expected (i.e., the feature is 'absorbed' by a parent).

    Method:
    1. For each letter, identify SAE features associated with that letter
       (features with high decoder alignment to letter token embeddings)
    2. Run the model on OWT text, extract residual stream activations at `layer`
    3. Pass through SAE, get feature activations
    4. For each letter token occurrence: check if any associated letter feature fires
    5. Absorption rate = fraction of letter tokens with NO letter feature firing

    This is the standard sae-spelling approach applied directly.
    """
    rng = np.random.RandomState(seed)

    w_dec = sae.W_dec.data.to(device)  # (d_sae, d_model)
    d_sae = w_dec.shape[0]

    # Identify letter features for this SAE
    letter_features = identify_letter_features(w_dec, model.cfg.d_model, device)

    if len(letter_features) < 3:
        return {
            "absorption_rate": None,
            "n_letters_tested": len(letter_features),
            "error": f"Only {len(letter_features)} letters have identified features"
        }

    # Limit to max_letters letters for speed
    letters_to_test = list(letter_features.keys())[:max_letters]

    # For each letter, get the letter tokens
    letter_token_ids = {}
    for letter in letters_to_test:
        tids = get_letter_tokens(model, tokenizer, letter, n_tokens=n_tokens_per_letter * 2)
        if tids:
            letter_token_ids[letter] = set(tids[:n_tokens_per_letter])

    # Run model on OWT tokens to get activations
    # Flatten tokens to find occurrences of letter tokens
    tokens_flat = owt_tokens.flatten()[:max_seq_len * 8]  # Limit for speed

    # Run in batches through the model
    batch_size = max_seq_len
    all_absorbed = []
    all_total = []
    per_letter_stats = {}

    # Prepare batches from owt_tokens
    n_seqs = min(owt_tokens.shape[0], 4)  # Use up to 4 sequences in pilot
    seqs = owt_tokens[:n_seqs, :max_seq_len].to(device)

    with torch.no_grad():
        # Get residual stream activations at the specified layer
        hook_name = f"blocks.{layer}.hook_resid_pre"
        _, cache = model.run_with_cache(seqs, names_filter=hook_name)
        resid = cache[hook_name]  # (batch, seq_len, d_model)

        # Pass through SAE
        b, s, d = resid.shape
        resid_flat = resid.reshape(b * s, d)

        # SAE encode
        sae_out = sae.encode(resid_flat)  # (b*s, d_sae)
        if isinstance(sae_out, tuple):
            feature_acts = sae_out[1] if len(sae_out) > 1 else sae_out[0]
        else:
            feature_acts = sae_out

        # Get token IDs for each position
        token_ids = seqs.reshape(-1).cpu().numpy()  # (b*s,)

        # For each letter, find positions where letter tokens occur
        for letter in letters_to_test:
            if letter not in letter_token_ids or letter not in letter_features:
                continue

            letter_tids = letter_token_ids[letter]
            letter_feat_ids = letter_features[letter]

            # Find positions with this letter's tokens
            positions = [i for i, t in enumerate(token_ids) if t in letter_tids]

            if len(positions) < 10:  # Need at least 10 occurrences
                continue

            # For each position, check if letter feature fires
            n_fires = 0
            n_absorbed = 0

            for pos in positions[:n_tokens_per_letter]:
                acts_at_pos = feature_acts[pos]  # (d_sae,)

                # Check if any letter feature fires (activation > threshold)
                max_letter_act = acts_at_pos[letter_feat_ids].max().item() if letter_feat_ids else 0

                if max_letter_act > 0.1:  # Feature fires
                    n_fires += 1
                else:
                    n_absorbed += 1  # Feature doesn't fire = absorbed

            total = n_fires + n_absorbed
            if total > 0:
                rate = n_absorbed / total
                per_letter_stats[letter] = {
                    "n_positions": len(positions),
                    "n_tested": total,
                    "n_fires": n_fires,
                    "n_absorbed": n_absorbed,
                    "absorption_rate": rate,
                    "n_features": len(letter_feat_ids)
                }

    if not per_letter_stats:
        return {
            "absorption_rate": None,
            "n_letters_tested": 0,
            "error": "No letters had sufficient token occurrences"
        }

    # Aggregate absorption rate across letters
    rates = [v["absorption_rate"] for v in per_letter_stats.values()]
    mean_rate = float(np.mean(rates))

    return {
        "absorption_rate": mean_rate,
        "absorption_rate_std": float(np.std(rates)),
        "n_letters_tested": len(per_letter_stats),
        "per_letter_stats": per_letter_stats,
        "n_features_per_letter_mean": float(np.mean([v["n_features"] for v in per_letter_stats.values()]))
    }


def get_empirical_l0(sae, owt_tokens, model, layer, device, n_tokens=500):
    """Measure empirical L0 (mean number of active features per token)."""
    n_seqs = min(owt_tokens.shape[0], 4)
    seqs = owt_tokens[:n_seqs, :128].to(device)

    with torch.no_grad():
        hook_name = f"blocks.{layer}.hook_resid_pre"
        _, cache = model.run_with_cache(seqs, names_filter=hook_name)
        resid = cache[hook_name]

        b, s, d = resid.shape
        resid_flat = resid.reshape(b * s, d)

        sae_out = sae.encode(resid_flat)
        if isinstance(sae_out, tuple):
            feature_acts = sae_out[1] if len(sae_out) > 1 else sae_out[0]
        else:
            feature_acts = sae_out

        # L0 = mean number of non-zero features per token
        l0 = (feature_acts > 0).float().sum(-1).mean().item()

    return l0


# ─── Step 5: Load SAEs and measure L0 + absorption rate ────────────────────
report_progress(5, TOTAL_STEPS, "Loading SAEs and measuring L0 and absorption rates")

scaling_curve_data = []
n_loaded = 0

for release, sae_id, layer, width, is_primary in SAE_CONFIGS_TO_TEST:
    print(f"\n  Testing: {release} | {sae_id} | layer={layer}, width={width}")

    try:
        sae = SAE.from_pretrained(release, sae_id)
        sae.eval()
        sae.to(DEVICE)

        # Measure empirical L0
        l0 = get_empirical_l0(sae, owt_tokens, model, layer, DEVICE)
        print(f"    Empirical L0: {l0:.2f}")

        # Measure absorption rate
        result = measure_absorption_rate(
            model, sae, layer, owt_tokens, DEVICE,
            n_tokens_per_letter=PILOT_TOKENS_PER_LETTER,
            seed=SEED
        )

        if result["absorption_rate"] is not None:
            print(f"    Absorption rate: {result['absorption_rate']:.4f} "
                  f"(n_letters={result['n_letters_tested']})")

            scaling_curve_data.append({
                "release": release,
                "sae_id": sae_id,
                "layer": layer,
                "width": width,
                "is_primary": is_primary,
                "l0": l0,
                "inv_l0": 1.0 / l0 if l0 > 0 else None,
                "absorption_rate": result["absorption_rate"],
                "absorption_rate_std": result.get("absorption_rate_std"),
                "n_letters_tested": result["n_letters_tested"],
                "per_letter_stats": result.get("per_letter_stats", {}),
                "n_features_per_letter": result.get("n_features_per_letter_mean"),
                "status": "success"
            })
            n_loaded += 1
        else:
            print(f"    SKIP: {result.get('error', 'Unknown error')}")
            scaling_curve_data.append({
                "release": release,
                "sae_id": sae_id,
                "layer": layer,
                "width": width,
                "l0": l0,
                "absorption_rate": None,
                "error": result.get("error"),
                "status": "failed"
            })

        # Free VRAM
        sae.cpu()
        del sae
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"    ERROR: {e}")
        scaling_curve_data.append({
            "release": release,
            "sae_id": sae_id,
            "layer": layer,
            "width": width,
            "status": "error",
            "error": str(e)
        })
        import traceback
        traceback.print_exc()

report_progress(6, TOTAL_STEPS, f"Loaded {n_loaded} SAE configs with absorption rates")

# ─── Step 6: Filter valid data points ───────────────────────────────────────
report_progress(6, TOTAL_STEPS, "Filtering valid data points")

valid_points = [d for d in scaling_curve_data
                if d.get("absorption_rate") is not None and d.get("l0") is not None]
print(f"\nValid data points: {len(valid_points)}")
for p in valid_points:
    print(f"  L0={p['l0']:.2f}, 1/L0={p['inv_l0']:.4f}, "
          f"abs_rate={p['absorption_rate']:.4f}, "
          f"layer={p['layer']}, width={p['width']}")

# Check pilot pass criteria
pass_criteria_met = len(valid_points) >= 3
print(f"\nPilot pass criteria (>=3 data points): {'PASS' if pass_criteria_met else 'FAIL'}")

# ─── Step 7: Curve fitting ──────────────────────────────────────────────────
report_progress(7, TOTAL_STEPS, "Fitting curve models (linear, power-law, sigmoid)")

curve_fit_results = {}

if len(valid_points) >= 3:
    # Extract data
    primary_points = [p for p in valid_points if p.get("is_primary", True) and
                      p["width"] == 24576]  # Same-width for primary analysis

    if len(primary_points) < 3:
        primary_points = valid_points  # Fall back to all points

    inv_l0_arr = np.array([p["inv_l0"] for p in primary_points])
    abs_rate_arr = np.array([p["absorption_rate"] for p in primary_points])

    # Sort by inv_l0
    sort_idx = np.argsort(inv_l0_arr)
    inv_l0_arr = inv_l0_arr[sort_idx]
    abs_rate_arr = abs_rate_arr[sort_idx]

    n_pts = len(inv_l0_arr)
    print(f"\nFitting curves to {n_pts} primary data points")

    # Model 1: Linear y = a*x + b
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(inv_l0_arr, abs_rate_arr)
        y_pred_linear = slope * inv_l0_arr + intercept
        ss_res = np.sum((abs_rate_arr - y_pred_linear)**2)
        ss_tot = np.sum((abs_rate_arr - abs_rate_arr.mean())**2)
        r2_linear = 1 - ss_res / ss_tot if ss_tot > 0 else 0

        # Log-likelihood for BIC/AIC
        sigma2 = ss_res / n_pts
        ll_linear = -n_pts/2 * np.log(2*np.pi*sigma2) - ss_res/(2*sigma2)
        k_linear = 2  # 2 params
        bic_linear = k_linear * np.log(n_pts) - 2 * ll_linear
        aic_linear = 2 * k_linear - 2 * ll_linear

        curve_fit_results["linear"] = {
            "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2_linear), "r_value": float(r_value),
            "p_value": float(p_value),
            "bic": float(bic_linear), "aic": float(aic_linear),
            "log_likelihood": float(ll_linear),
            "n_params": k_linear
        }
        print(f"  Linear: slope={slope:.4f}, R2={r2_linear:.4f}, BIC={bic_linear:.2f}")
    except Exception as e:
        print(f"  Linear fit error: {e}")
        curve_fit_results["linear"] = {"error": str(e)}

    # Model 2: Power-law y = a*x^b
    if len(inv_l0_arr) >= 3 and np.all(inv_l0_arr > 0) and np.all(abs_rate_arr > 0):
        try:
            from scipy.optimize import curve_fit as scipy_curve_fit
            def powerlaw(x, a, b):
                return a * np.power(x, b)

            popt, pcov = scipy_curve_fit(powerlaw, inv_l0_arr, abs_rate_arr,
                                          p0=[1.0, 1.0], maxfev=5000)
            y_pred_power = powerlaw(inv_l0_arr, *popt)
            ss_res = np.sum((abs_rate_arr - y_pred_power)**2)
            r2_power = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            sigma2 = ss_res / n_pts
            ll_power = -n_pts/2 * np.log(2*np.pi*sigma2) - ss_res/(2*sigma2) if sigma2 > 0 else -1e10
            k_power = 2
            bic_power = k_power * np.log(n_pts) - 2 * ll_power
            aic_power = 2 * k_power - 2 * ll_power

            curve_fit_results["power_law"] = {
                "a": float(popt[0]), "b": float(popt[1]),
                "r2": float(r2_power),
                "bic": float(bic_power), "aic": float(aic_power),
                "log_likelihood": float(ll_power),
                "n_params": k_power
            }
            print(f"  Power-law: a={popt[0]:.4f}, b={popt[1]:.4f}, R2={r2_power:.4f}, BIC={bic_power:.2f}")
        except Exception as e:
            print(f"  Power-law fit error: {e}")
            curve_fit_results["power_law"] = {"error": str(e)}
    else:
        print("  Power-law: skipped (insufficient valid points or non-positive values)")
        curve_fit_results["power_law"] = {"error": "insufficient data or non-positive values"}

    # Model 3: Sigmoid y = L/(1+exp(-k*(x-x0)))
    if len(inv_l0_arr) >= 4:
        try:
            from scipy.optimize import curve_fit as scipy_curve_fit
            def sigmoid(x, L, k, x0):
                return L / (1 + np.exp(-k * (x - x0)))

            # Initial guess
            L0_init = abs_rate_arr.max()
            k_init = 50.0
            x0_init = np.median(inv_l0_arr)

            popt, pcov = scipy_curve_fit(sigmoid, inv_l0_arr, abs_rate_arr,
                                          p0=[L0_init, k_init, x0_init],
                                          maxfev=10000,
                                          bounds=([0, 0, 0], [1, 1000, 1]))
            y_pred_sigmoid = sigmoid(inv_l0_arr, *popt)
            ss_res = np.sum((abs_rate_arr - y_pred_sigmoid)**2)
            r2_sigmoid = 1 - ss_res / ss_tot if ss_tot > 0 else 0

            sigma2 = ss_res / n_pts
            ll_sigmoid = -n_pts/2 * np.log(2*np.pi*sigma2) - ss_res/(2*sigma2) if sigma2 > 0 else -1e10
            k_sigmoid = 3
            bic_sigmoid = k_sigmoid * np.log(n_pts) - 2 * ll_sigmoid
            aic_sigmoid = 2 * k_sigmoid - 2 * ll_sigmoid

            # LRT: sigmoid vs linear
            if "log_likelihood" in curve_fit_results.get("linear", {}):
                lrt_stat = 2 * (ll_sigmoid - curve_fit_results["linear"]["log_likelihood"])
                lrt_df = k_sigmoid - k_linear  # extra params in sigmoid
                lrt_pvalue = float(1 - stats.chi2.cdf(lrt_stat, df=lrt_df)) if lrt_stat > 0 else 1.0
            else:
                lrt_stat = None
                lrt_pvalue = None

            # Inflection point = x0 (midpoint of sigmoid)
            inflection_l0_c = 1.0 / popt[2] if popt[2] > 0 else None

            curve_fit_results["sigmoid"] = {
                "L": float(popt[0]), "k": float(popt[1]), "x0": float(popt[2]),
                "r2": float(r2_sigmoid),
                "bic": float(bic_sigmoid), "aic": float(aic_sigmoid),
                "log_likelihood": float(ll_sigmoid),
                "n_params": k_sigmoid,
                "inflection_inv_l0": float(popt[2]),
                "inflection_l0_c": float(inflection_l0_c) if inflection_l0_c else None,
                "lrt_stat": float(lrt_stat) if lrt_stat is not None else None,
                "lrt_pvalue": float(lrt_pvalue) if lrt_pvalue is not None else None,
                "lrt_df": lrt_df if "log_likelihood" in curve_fit_results.get("linear", {}) else None
            }
            print(f"  Sigmoid: L={popt[0]:.4f}, k={popt[1]:.2f}, x0={popt[2]:.4f}, "
                  f"R2={r2_sigmoid:.4f}, BIC={bic_sigmoid:.2f}")
            if lrt_pvalue is not None:
                print(f"  LRT sigmoid vs linear: stat={lrt_stat:.3f}, p={lrt_pvalue:.4f}")
        except Exception as e:
            print(f"  Sigmoid fit error: {e}")
            curve_fit_results["sigmoid"] = {"error": str(e)}
    else:
        print(f"  Sigmoid: skipped (need >=4 points, have {len(inv_l0_arr)})")
        curve_fit_results["sigmoid"] = {
            "error": f"Need >=4 data points, have {len(inv_l0_arr)}"
        }

# ─── Step 8: Bootstrap CI for inflection point ─────────────────────────────
report_progress(8, TOTAL_STEPS, "Computing 95% bootstrap CI for inflection point")

bootstrap_results = {}
n_bootstrap = 1000  # Full 1000 resamples as specified

if (len(valid_points) >= 4 and
    "sigmoid" in curve_fit_results and
    "error" not in curve_fit_results.get("sigmoid", {})):

    from scipy.optimize import curve_fit as scipy_curve_fit

    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    primary_pts = [p for p in valid_points if p.get("is_primary", True) and p["width"] == 24576]
    if len(primary_pts) < 4:
        primary_pts = valid_points

    inv_l0_arr = np.array([p["inv_l0"] for p in primary_pts])
    abs_rate_arr = np.array([p["absorption_rate"] for p in primary_pts])

    boot_inflections = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        idx = np.random.choice(len(primary_pts), len(primary_pts), replace=True)
        x_boot = inv_l0_arr[idx]
        y_boot = abs_rate_arr[idx]

        if len(np.unique(x_boot)) < 3:
            continue

        try:
            popt, _ = scipy_curve_fit(
                sigmoid, x_boot, y_boot,
                p0=[y_boot.max(), 50.0, np.median(x_boot)],
                maxfev=5000,
                bounds=([0, 0, 0], [1, 1000, 1])
            )
            if popt[2] > 0:
                boot_inflections.append(1.0 / popt[2])
        except:
            pass

    if len(boot_inflections) >= 50:
        ci_low = float(np.percentile(boot_inflections, 2.5))
        ci_high = float(np.percentile(boot_inflections, 97.5))
        ci_mean = float(np.mean(boot_inflections))
        bootstrap_results = {
            "n_successful": len(boot_inflections),
            "inflection_l0c_mean": ci_mean,
            "inflection_l0c_ci_low": ci_low,
            "inflection_l0c_ci_high": ci_high,
            "nominal_inflection_l0c": curve_fit_results["sigmoid"].get("inflection_l0_c")
        }
        print(f"Bootstrap CI for L0_c: {ci_mean:.1f} [{ci_low:.1f}, {ci_high:.1f}] "
              f"(n_boot={len(boot_inflections)})")
    else:
        bootstrap_results = {"error": f"Only {len(boot_inflections)} successful bootstrap fits"}
        print(f"Bootstrap failed: only {len(boot_inflections)} successful fits")
else:
    reason = ("insufficient data points" if len(valid_points) < 4
              else "sigmoid fit failed")
    bootstrap_results = {"skipped": f"Bootstrap skipped: {reason}"}
    print(f"Bootstrap skipped: {reason}")

# ─── Step 9: Width analysis (feature-splitting) ─────────────────────────────
report_progress(9, TOTAL_STEPS, "Analyzing width effect (feature-splitting SAEs)")

width_analysis = {}
width_points = [p for p in valid_points if p["release"] == "gpt2-small-res-jb-feature-splitting"]
same_layer_points = [p for p in valid_points if p["layer"] == 8 and p["width"] == 24576]

all_l8_points = width_points + same_layer_points
if len(all_l8_points) >= 2:
    print(f"\nWidth analysis ({len(all_l8_points)} points at layer 8):")
    for p in sorted(all_l8_points, key=lambda x: x["width"]):
        print(f"  Width={p['width']}, L0={p['l0']:.2f}, "
              f"abs_rate={p['absorption_rate']:.4f}")

    width_analysis = {
        "n_points": len(all_l8_points),
        "points": [{"width": p["width"], "l0": p["l0"],
                    "inv_l0": p["inv_l0"], "absorption_rate": p["absorption_rate"]}
                   for p in sorted(all_l8_points, key=lambda x: x["width"])],
        "note": "Wider SAEs tend to have lower L0 (sparser) and potentially lower absorption"
    }
else:
    width_analysis = {"n_points": len(all_l8_points), "note": "Insufficient data for width analysis"}

# ─── Step 10: Assess H4 prediction ─────────────────────────────────────────
report_progress(10, TOTAL_STEPS, "Assessing H1 quantitative prediction (lambda* = sin^2(theta))")

# From B1 results: mean cos^2(theta) for absorbed features was measured
# H1 predicts: absorption onset at lambda* = sin^2(theta_mean_absorbed) = 1 - cos^2(theta_mean_absorbed)
# Load B1 results for this
b1_results_path = FULL_RESULTS_DIR / "B1_decoder_geometry.json"
h1_prediction_check = {}

if b1_results_path.exists():
    with open(b1_results_path) as f:
        b1_data = json.load(f)

    # Get mean cos^2 for absorbed (letter) features at L6
    l6_data = b1_data.get("layer6", {})
    l6_eda = l6_data.get("eda_analysis", {})
    l6_pair = l6_data.get("pair_level_analysis", {})

    # Use max_cos2 for absorbed features
    pos_cos2_mean = l6_pair.get("max_cos2", {}).get("pos_mean", None)
    if pos_cos2_mean is not None:
        lambda_star = 1 - pos_cos2_mean  # sin^2(theta) = 1 - cos^2(theta)
        predicted_l0c = 1.0 / lambda_star if lambda_star > 0 else None

        h1_prediction_check = {
            "pos_cos2_mean_l6": float(pos_cos2_mean),
            "lambda_star": float(lambda_star),
            "predicted_l0c": float(predicted_l0c) if predicted_l0c else None,
            "observed_l0c": curve_fit_results.get("sigmoid", {}).get("inflection_l0_c"),
            "note": "H1 predicts absorption onset at lambda* = sin^2(theta_mean_absorbed)"
        }
        print(f"\nH1 prediction: lambda*={lambda_star:.4f}, predicted L0_c={predicted_l0c:.1f}")
        if curve_fit_results.get("sigmoid", {}).get("inflection_l0_c"):
            obs = curve_fit_results["sigmoid"]["inflection_l0_c"]
            print(f"Observed L0_c={obs:.1f}, ratio={obs/predicted_l0c:.2f}")
    else:
        h1_prediction_check = {"error": "No pos_cos2_mean in B1 results"}
else:
    h1_prediction_check = {"error": "B1 results not found"}

# ─── Step 11: Compile results ────────────────────────────────────────────────
report_progress(11, TOTAL_STEPS, "Compiling and saving results")

elapsed_total = time.time() - start_time

# Determine overall pass/fail
n_valid = len(valid_points)
pilot_pass = n_valid >= 3

# Determine primary finding
if curve_fit_results.get("sigmoid", {}).get("r2", 0) > curve_fit_results.get("linear", {}).get("r2", 0):
    primary_finding = f"Sigmoid fit (R2={curve_fit_results['sigmoid']['r2']:.3f}) > Linear fit (R2={curve_fit_results['linear'].get('r2', 0):.3f})"
elif "r2" in curve_fit_results.get("linear", {}):
    primary_finding = f"Linear fit best available (R2={curve_fit_results['linear']['r2']:.3f})"
else:
    primary_finding = "Insufficient data for curve fitting"

results = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed_total,
    "config": {
        "model": "gpt2-small",
        "sae_release_primary": "gpt2-small-res-jb",
        "seed": SEED,
        "pilot_tokens_per_letter": PILOT_TOKENS_PER_LETTER,
        "n_bootstrap": n_bootstrap,
        "device": DEVICE
    },
    "pilot_pass": pilot_pass,
    "pass_criteria": ">=3 different L0 settings measured",
    "n_valid_points": n_valid,
    "scaling_curve_data": scaling_curve_data,
    "valid_data_points": valid_points,
    "curve_fit_results": curve_fit_results,
    "bootstrap_ci": bootstrap_results,
    "width_analysis": width_analysis,
    "h1_prediction_check": h1_prediction_check,
    "primary_finding": primary_finding,
    "summary": {
        "n_configs_tested": len(SAE_CONFIGS_TO_TEST),
        "n_valid": n_valid,
        "l0_range": [min(p["l0"] for p in valid_points) if valid_points else None,
                      max(p["l0"] for p in valid_points) if valid_points else None],
        "abs_rate_range": [min(p["absorption_rate"] for p in valid_points) if valid_points else None,
                            max(p["absorption_rate"] for p in valid_points) if valid_points else None],
        "pilot_pass": pilot_pass
    }
}

# Save pilot results
OUTPUT_FILE_PILOT.write_text(json.dumps(results, indent=2))
print(f"\nPilot results saved: {OUTPUT_FILE_PILOT}")

# Also save as full results if pilot passes
if pilot_pass:
    OUTPUT_FILE_FULL.write_text(json.dumps(results, indent=2))
    print(f"Full results saved: {OUTPUT_FILE_FULL}")

# ─── Step 12: Mark done ──────────────────────────────────────────────────────
report_progress(12, TOTAL_STEPS, f"Done! Pilot {'PASS' if pilot_pass else 'FAIL'}: {primary_finding}")

status = "success" if pilot_pass else "partial"
summary = (f"B2 Scaling Curve: {n_valid} valid configs, "
           f"L0 range=[{results['summary']['l0_range'][0]:.1f}, {results['summary']['l0_range'][1]:.1f}], "
           f"abs_rate range=[{results['summary']['abs_rate_range'][0]:.3f}, {results['summary']['abs_rate_range'][1]:.3f}]"
           if valid_points else "B2 Scaling Curve: No valid data points")

mark_done(status=status, summary=summary, result=results["summary"])

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
if gpu_progress_path.exists():
    with open(gpu_progress_path) as f:
        gp = json.load(f)
else:
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

end_time = datetime.now().isoformat()
if pilot_pass:
    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
else:
    if TASK_ID not in gp["failed"]:
        gp["failed"].append(TASK_ID)

gp["running"].pop(TASK_ID, None)
gp["timings"][TASK_ID] = {
    "planned_min": 40,
    "actual_min": int(elapsed_total / 60),
    "start_time": datetime.fromtimestamp(start_time).isoformat(),
    "end_time": end_time,
    "config_snapshot": {
        "model": "gpt2-small",
        "n_sae_configs": len(SAE_CONFIGS_TO_TEST),
        "n_valid_points": n_valid,
        "pilot_pass": pilot_pass,
        "primary_finding": primary_finding[:100],
        "gpu_model": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    }
}

with open(gpu_progress_path, "w") as f:
    json.dump(gp, f, indent=2)

print(f"\n{'='*60}")
print(f"Task B2 Scaling Curve COMPLETE")
print(f"Pilot pass: {pilot_pass}")
print(f"Valid data points: {n_valid}/{len(SAE_CONFIGS_TO_TEST)}")
if valid_points:
    print(f"L0 range: {results['summary']['l0_range']}")
    print(f"Absorption rate range: {results['summary']['abs_rate_range']}")
print(f"Primary finding: {primary_finding}")
print(f"Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f}min)")
print(f"{'='*60}")
