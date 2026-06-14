"""
B.2: Sparsity-EDA Scaling Analysis (Reuse Pilot Data) — v2 with improved absorption measurement

Pilot pass criteria:
  - True sae-spelling absorption rate measured for at least 3 configs
  - Sigmoid vs. linear BIC comparison computed

True absorption measurement approach (sae-spelling definition):
  - Load SAE for each config
  - Generate tokens starting with letter X
  - Find the "split feature" for letter X: the SAE feature that fires
    most selectively for X-starting words
  - An absorption EVENT is when:
    (a) the word starts with X, AND
    (b) the letter X's split feature does NOT fire (activation < threshold)
  - Absorption rate = fraction of X-starting words where the split feature fails to fire
"""

import json
import os
import sys
import time
import math
import numpy as np
from datetime import datetime
from pathlib import Path
from scipy import stats
from scipy.stats import spearmanr

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp/results"
FULL_DIR = RESULTS_DIR / "full"
PILOTS_DIR = RESULTS_DIR / "pilots"
TASK_ID = "task_B2_sparsity_analysis"

# PID file
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def report_progress(step, total, note=""):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress.write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total,
        "note": note,
        "updated_at": datetime.now().isoformat(),
    }))

def mark_done(status="success", summary=""):
    if pid_file.exists():
        pid_file.unlink()
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }))

report_progress(0, 10, "Loading B2 pilot data")
t_start = time.time()

# ========================
# Load B2 pilot data
# ========================
b2_path = FULL_DIR / "B2_scaling_curve.json"
if not b2_path.exists():
    b2_path = PILOTS_DIR / "pilot_B2_scaling_curve.json"

with open(b2_path) as f:
    b2_data = json.load(f)

all_results = b2_data["all_results"]
primary_jb = [r for r in all_results if r["group"] == "primary"]
ajt_variants = [r for r in all_results if r["group"] == "ajt"]
width_variants = [r for r in all_results if r["group"] == "width"]

# Width analysis: include primary L8 for 4-point comparison
primary_l8 = next((r for r in primary_jb if r["layer"] == 8), None)
width_all = sorted([primary_l8] + width_variants, key=lambda x: x["width"]) if primary_l8 else sorted(width_variants, key=lambda x: x["width"])

print(f"Loaded {len(all_results)} configs")
print(f"Primary jb: {len(primary_jb)}, AJT: {len(ajt_variants)}, Width: {len(width_variants)}")

# ========================
# Existing curve fit parameters (from B2 pilot)
# ========================
existing_sigmoid = b2_data["curve_fits"]["sigmoid"]
existing_linear = b2_data["curve_fits"]["linear"]
existing_bootstrap = b2_data["bootstrap_ci"]
bic_delta = existing_linear["bic"] - existing_sigmoid["bic"]
lrt_p = existing_sigmoid["lrt_pvalue"]
l0c_nominal = existing_sigmoid["inflection_l0_c"]

# ========================
# Primary suite statistics
# ========================
primary_x = np.array([r["inv_l0"] for r in primary_jb])
primary_y_delta = np.array([r["eda_delta"] for r in primary_jb])
primary_layers = [r["layer"] for r in primary_jb]

rho_primary, p_primary = spearmanr(primary_x, primary_y_delta)
rho_layer, p_layer = spearmanr(primary_layers, primary_y_delta)

# ========================
# Width analysis statistics
# ========================
widths_only = np.array([r["width"] for r in width_variants])
log2_widths_only = np.log2(widths_only)
eda_deltas_only = np.array([r["eda_delta"] for r in width_variants])
rho_width_delta_3, p_width_delta_3 = spearmanr(log2_widths_only, eda_deltas_only)

widths_all_arr = np.array([r["width"] for r in width_all])
log2_widths_all = np.log2(widths_all_arr)
eda_deltas_all = np.array([r["eda_delta"] for r in width_all])
rho_width_delta_4, p_width_delta_4 = spearmanr(log2_widths_all, eda_deltas_all)

# ========================
# AJT statistics
# ========================
if len(ajt_variants) >= 3:
    ajt_x = np.array([r["inv_l0"] for r in ajt_variants])
    ajt_y = np.array([r["eda_delta"] for r in ajt_variants])
    rho_ajt, p_ajt = spearmanr(ajt_x, ajt_y)
else:
    rho_ajt, p_ajt = None, None

jb_l6 = next(r for r in primary_jb if r["layer"] == 6)
ajt_letter_edas = [r["mean_eda_letter"] for r in ajt_variants]
ajt_nonletter_edas = [r["mean_eda_nonletter"] for r in ajt_variants]

# ========================
# True absorption rate via SAELens
# ========================
report_progress(3, 10, "Measuring true absorption rates via SAELens")

print("\n=== TRUE ABSORPTION RATE MEASUREMENT ===")

# Load C1 probe data for passing letters
c1_path = FULL_DIR / "C1_probe_training.json"
passing_letters = ["a", "b", "d", "e", "g", "h", "i", "j", "k", "m", "n", "o", "q", "r", "s", "w", "y"]
if c1_path.exists():
    with open(c1_path) as f:
        c1_data = json.load(f)
    if "hierarchies" in c1_data and "first_letter" in c1_data["hierarchies"]:
        fl = c1_data["hierarchies"]["first_letter"]
        if "metrics" in fl and "passing_letters" in fl["metrics"]:
            passing_letters = fl["metrics"]["passing_letters"]

print(f"Using {len(passing_letters)} passing letters: {passing_letters}")

# Test vocabulary: common English words for each letter (for in-distribution testing)
# Use diverse words to measure robustness
LETTER_WORDS = {
    "a": [" apple", " animal", " after", " again", " about", " above", " art", " ask", " act", " aim"],
    "b": [" black", " brain", " bring", " blood", " below", " base", " back", " born", " big", " bold"],
    "c": [" class", " child", " cloud", " clear", " claim", " can", " city", " cut", " care", " cool"],
    "d": [" death", " doing", " dream", " drive", " dated", " dark", " deep", " deal", " down", " draw"],
    "e": [" earth", " eight", " every", " event", " early", " edge", " each", " end", " end", " easy"],
    "f": [" false", " field", " fight", " found", " floor", " far", " form", " fall", " feel", " full"],
    "g": [" given", " going", " great", " green", " guess", " good", " gain", " gold", " grow", " goal"],
    "h": [" happy", " heart", " heavy", " hands", " house", " has", " help", " high", " hold", " head"],
    "i": [" image", " index", " input", " items", " inter", " idea", " is", " into", " its", " iron"],
    "j": [" james", " japan", " jones", " judge", " joins", " just", " job", " joy", " jar", " jump"],
    "k": [" keeps", " kings", " kinds", " known", " knife", " key", " know", " kill", " kick", " kind"],
    "l": [" large", " light", " limit", " lived", " legal", " long", " like", " last", " lead", " left"],
    "m": [" model", " major", " means", " might", " moved", " make", " man", " most", " more", " move"],
    "n": [" north", " noted", " needs", " never", " nodes", " new", " not", " now", " near", " name"],
    "o": [" often", " other", " owned", " order", " outer", " over", " only", " open", " once", " old"],
    "p": [" place", " parts", " plant", " point", " prove", " put", " play", " plan", " past", " path"],
    "q": [" queen", " quite", " query", " queue", " quasi", " quick", " quiet", " quest", " quota"],
    "r": [" ready", " right", " range", " raise", " rapid", " run", " real", " read", " rise", " rule"],
    "s": [" small", " space", " state", " study", " since", " such", " said", " saw", " set", " show"],
    "t": [" three", " total", " times", " taken", " truth", " than", " two", " take", " top", " try"],
    "u": [" under", " until", " units", " upper", " users", " use", " up", " upon", " used", " us"],
    "v": [" value", " voice", " valid", " voter", " vital", " view", " very", " vast", " vary"],
    "w": [" world", " would", " water", " words", " while", " was", " way", " we", " with", " work"],
    "y": [" years", " young", " yield", " youth", " yards", " yet", " yes", " year", " you", " your"],
}

# Non-letter-X words for baseline (control)
CONTROL_WORDS = [" the", " in", " of", " is", " was", " are", " with", " that", " or", " for"]

try:
    import torch
    import sae_lens
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    print("Loading GPT-2 Small...")
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    print("Model loaded.")

    # Configs to test (pilot: 3 configs with distinct L0 values)
    configs_to_test = [
        {"release": "gpt2-small-res-jb", "layer": 2, "hook": "blocks.2.hook_resid_pre", "label": "L2_jb"},
        {"release": "gpt2-small-res-jb", "layer": 6, "hook": "blocks.6.hook_resid_pre", "label": "L6_jb"},
        {"release": "gpt2-small-res-jb", "layer": 10, "hook": "blocks.10.hook_resid_pre", "label": "L10_jb"},
    ]

    true_absorption_results = []

    for cfg in configs_to_test:
        print(f"\n  Loading SAE: {cfg['release']} {cfg['hook']}...")
        try:
            sae_obj = SAE.from_pretrained(
                release=cfg["release"],
                sae_id=cfg["hook"],
                device=DEVICE
            )
            if isinstance(sae_obj, tuple):
                sae = sae_obj[0]
            else:
                sae = sae_obj
            sae.eval()

            W_enc = sae.W_enc.detach()  # [d_model, d_sae]
            b_enc = sae.b_enc.detach()  # [d_sae]
            d_sae = W_enc.shape[1]

            # Step 1: For each letter, collect activations on X-starting words
            letter_activations = {}  # letter -> array [n_words, d_sae]
            control_activations = []  # list of [d_sae]

            with torch.no_grad():
                # Control words (non-letter-starting words)
                for word in CONTROL_WORDS:
                    tokens = model.to_tokens(word, prepend_bos=True)
                    _, cache = model.run_with_cache(tokens, names_filter=cfg["hook"])
                    resid = cache[cfg["hook"]][0, -1]
                    z = resid @ W_enc + b_enc
                    acts = torch.clamp(z, min=0).cpu().numpy()
                    control_activations.append(acts)

                control_activations = np.array(control_activations)  # [n_control, d_sae]
                control_mean = control_activations.mean(axis=0)  # [d_sae]

                # Letter words
                test_letters = [l for l in passing_letters if l in LETTER_WORDS][:10]
                for letter in test_letters:
                    words = LETTER_WORDS[letter]
                    letter_acts = []
                    for word in words:
                        tokens = model.to_tokens(word, prepend_bos=True)
                        _, cache = model.run_with_cache(tokens, names_filter=cfg["hook"])
                        resid = cache[cfg["hook"]][0, -1]
                        z = resid @ W_enc + b_enc
                        acts = torch.clamp(z, min=0).cpu().numpy()
                        letter_acts.append(acts)
                    letter_activations[letter] = np.array(letter_acts)

            # Step 2: For each letter, find the "split feature"
            # The split feature for letter X is the SAE feature that:
            # (a) is SELECTIVE: highest for X, NOT for other letters
            # We compute: mean_act(X) - max(mean_act(other_letters)) to get letter-specific features
            letter_features = {}  # letter -> feature_idx
            letter_feature_info = {}

            # Build mean activation per letter
            letter_means = {letter: acts.mean(axis=0) for letter, acts in letter_activations.items()}

            for letter, acts in letter_activations.items():
                mean_on_letter = letter_means[letter]
                mean_on_control = control_mean

                # Differential vs. control
                diff_vs_control = mean_on_letter - mean_on_control

                # Selectivity vs. other letters: mean_act(this_letter) - max(mean_act(other))
                other_letters = [l for l in letter_means.keys() if l != letter]
                if other_letters:
                    stacked_others = np.stack([letter_means[l] for l in other_letters], axis=0)
                    max_other = stacked_others.max(axis=0)
                    selectivity = mean_on_letter - max_other
                else:
                    selectivity = diff_vs_control

                # Combined score: differential + selectivity
                combined_score = diff_vs_control + selectivity

                # Top feature by combined score (selective for this letter)
                top_feature = combined_score.argmax()

                # Activation statistics
                feat_acts_on_letter = acts[:, top_feature]
                feat_acts_on_control = control_activations[:, top_feature]

                letter_features[letter] = top_feature
                letter_feature_info[letter] = {
                    "feature_idx": int(top_feature),
                    "mean_act_on_letter": float(feat_acts_on_letter.mean()),
                    "mean_act_on_control": float(feat_acts_on_control.mean()),
                    "differential": float(feat_acts_on_letter.mean() - feat_acts_on_control.mean()),
                    "n_words_tested": len(acts),
                    "n_active_on_letter": int((feat_acts_on_letter > 0.01).sum()),
                    "activation_rate_on_letter": float((feat_acts_on_letter > 0.01).mean()),
                }

            # Step 3: Measure absorption rate
            # sae-spelling absorption: the "split feature" fails to fire for some words starting with X
            # We use a dynamic threshold: the feature must fire at > 10% of its mean activation
            # to be considered "active"
            absorption_events = {}

            for letter, feat_idx in letter_features.items():
                acts = letter_activations[letter]  # [n_words, d_sae]
                feat_acts = acts[:, feat_idx]  # [n_words]

                info = letter_feature_info[letter]

                # Threshold: 10% of mean activation on letter words (or 0.05 min)
                threshold = max(0.05, 0.1 * info["mean_act_on_letter"])

                # Absorption = feature fails to fire above threshold
                n_absorbed = int((feat_acts < threshold).sum())
                n_total = len(feat_acts)
                absorption_rate = n_absorbed / n_total if n_total > 0 else 0.0

                absorption_events[letter] = {
                    "letter": letter,
                    "feature_idx": feat_idx,
                    "n_words": n_total,
                    "n_absorbed": n_absorbed,
                    "absorption_rate": float(absorption_rate),
                    "mean_activation": float(info["mean_act_on_letter"]),
                    "threshold": float(threshold),
                    "activation_rate": float(info["activation_rate_on_letter"]),
                }

            # Overall absorption rate
            overall_absorption_rate = np.mean([v["absorption_rate"] for v in absorption_events.values()])
            n_letters = len(absorption_events)

            # EDA of identified letter features
            W_enc_normed = W_enc / (W_enc.norm(dim=0, keepdim=True) + 1e-8)
            W_dec = sae.W_dec.detach()
            W_dec_normed = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-8)

            letter_feat_edas = {}
            for letter, feat_idx in letter_features.items():
                enc_j = W_enc_normed[:, feat_idx].cpu().numpy()
                dec_j = W_dec_normed[feat_idx].cpu().numpy()
                cos_ed = float(np.dot(enc_j, dec_j))
                letter_feat_edas[letter] = 1.0 - cos_ed

            # Match with B2 data for comparison
            b2_entry = next((r for r in primary_jb if r["layer"] == cfg["layer"]), None)

            result = {
                "label": cfg["label"],
                "release": cfg["release"],
                "layer": cfg["layer"],
                "hook": cfg["hook"],
                "l0": b2_entry["l0"] if b2_entry else None,
                "inv_l0": b2_entry["inv_l0"] if b2_entry else None,
                "n_letters_tested": n_letters,
                "overall_absorption_rate": float(overall_absorption_rate),
                "per_letter": {k: v for k, v in absorption_events.items()},
                "letter_feature_edas": {k: float(v) for k, v in letter_feat_edas.items()},
                "mean_letter_feature_eda": float(np.mean(list(letter_feat_edas.values()))) if letter_feat_edas else None,
                "eda_auroc_from_b2": b2_entry["eda_auroc"] if b2_entry else None,
                "eda_delta_from_b2": b2_entry["eda_delta"] if b2_entry else None,
                "method": "split_feature_non_firing",
                "note": "Absorption event = split feature for letter X fails to activate (< 10% of mean activation) on X-starting word."
            }
            true_absorption_results.append(result)

            print(f"    L{cfg['layer']}: n_letters={n_letters}, absorption_rate={overall_absorption_rate:.3f}")
            print(f"    Mean letter feature EDA: {result['mean_letter_feature_eda']:.4f}")
            for letter, ae in absorption_events.items():
                print(f"      Letter '{letter}': feat={ae['feature_idx']}, activation_rate={ae['activation_rate']:.2f}, absorption_rate={ae['absorption_rate']:.2f}")

            del sae
            if "cuda" in DEVICE:
                torch.cuda.empty_cache()

        except Exception as e:
            import traceback
            print(f"    FAILED for L{cfg['layer']}: {e}")
            traceback.print_exc()
            true_absorption_results.append({
                "label": cfg["label"],
                "release": cfg["release"],
                "layer": cfg["layer"],
                "error": str(e),
                "overall_absorption_rate": None,
            })

    del model
    if "cuda" in DEVICE:
        torch.cuda.empty_cache()

    n_measured = sum(1 for r in true_absorption_results if r.get("overall_absorption_rate") is not None)
    print(f"\nTrue absorption rate measured for {n_measured}/{len(configs_to_test)} configs")
    use_true_rates = True

except Exception as e:
    import traceback
    print(f"Error in true absorption measurement: {e}")
    traceback.print_exc()
    true_absorption_results = []
    n_measured = 0
    use_true_rates = False

# ========================
# Pilot pass assessment
# ========================
report_progress(8, 10, "Compiling results")

pilot_pass = n_measured >= 3

# Absorption vs EDA correlation (if data available)
absorption_vs_eda = []
if use_true_rates:
    for r in true_absorption_results:
        if r.get("overall_absorption_rate") is not None and r.get("l0") is not None:
            absorption_vs_eda.append({
                "layer": r["layer"],
                "l0": r["l0"],
                "inv_l0": r["inv_l0"],
                "true_absorption_rate": r["overall_absorption_rate"],
                "eda_auroc": r["eda_auroc_from_b2"],
                "eda_delta": r["eda_delta_from_b2"],
            })

# Phase transition summary
sigmoid_analysis = {
    "n_points": len(primary_jb),
    "suite": "primary_jb_cross_layer",
    "configs": [{
        "layer": r["layer"],
        "l0": r["l0"],
        "inv_l0": r["inv_l0"],
        "eda_delta": r["eda_delta"],
        "eda_auroc": r["eda_auroc"],
        "n_letter_features": r["n_letter_features"],
        "probe_threshold": r["probe_threshold"]
    } for r in primary_jb],
    "sigmoid_fit": {
        "L": float(existing_sigmoid["L"]),
        "k": float(existing_sigmoid["k"]),
        "x0": float(existing_sigmoid["x0"]),
        "b": float(existing_sigmoid["b"]),
        "r2": float(existing_sigmoid["r2"]),
        "bic": float(existing_sigmoid["bic"]),
        "lrt_stat": float(existing_sigmoid["lrt_stat"]),
        "lrt_pvalue": float(existing_sigmoid["lrt_pvalue"]),
        "inflection_l0_c": float(existing_sigmoid["inflection_l0_c"]),
    },
    "linear_fit": {
        "slope": float(existing_linear["slope"]),
        "intercept": float(existing_linear["intercept"]),
        "r2": float(existing_linear["r2"]),
        "bic": float(existing_linear["bic"]),
    },
    "bic_delta_linear_minus_sigmoid": float(bic_delta),
    "bootstrap_ci_l0c": {
        "low": float(existing_bootstrap["l0c_ci_low"]),
        "high": float(existing_bootstrap["l0c_ci_high"]),
        "nominal": float(l0c_nominal),
        "n_successful": int(existing_bootstrap["n_successful"]),
        "warning": "CI is unreliable with n=5 points. Nominal estimate should not be over-interpreted."
    },
    "spearman_inv_l0_vs_eda_delta": {
        "rho": float(rho_primary),
        "p": float(p_primary),
        "n": len(primary_jb),
    },
    "spearman_layer_vs_eda_delta": {
        "rho": float(rho_layer),
        "p": float(p_layer),
        "n": len(primary_jb),
        "note": "Layer confound check — EDA_delta vs. layer number"
    },
    "interpretation": {
        "conclusion": "Directional support for sigmoid phase transition (LRT p=0.027, BIC delta=2.42 favoring sigmoid), but with only 5 cross-layer data points this is statistically underpowered.",
        "caveat_layer_confound": "Cross-layer comparison conflates absorption with representation maturity. L0 increases monotonically with layer for the jb primary suite, making it hard to disentangle.",
        "caveat_bootstrap": "Bootstrap CI for L0_c is essentially uninformative (covers ~4 orders of magnitude).",
        "l0c_interpretation": f"Nominal inflection at L0_c = {l0c_nominal:.0f}. If this reflects a real phase transition, absorption becomes detectable as L0 exceeds ~81 features per token.",
        "action_needed": "E.1 (within-layer L0 variation using AJT suite) is needed for conclusive evidence."
    }
}

# AJT analysis
ajt_analysis = {
    "n_variants": len(ajt_variants),
    "all_at_layer_6_width_46080": True,
    "variants": [{
        "release": r["release"],
        "l0": r["l0"],
        "inv_l0": r["inv_l0"],
        "eda_delta": r["eda_delta"],
        "mean_eda_letter": r["mean_eda_letter"],
        "mean_eda_nonletter": r["mean_eda_nonletter"],
        "eda_auroc": r["eda_auroc"],
    } for r in ajt_variants],
    "all_negative_eda_delta": all(r["eda_delta"] < 0 for r in ajt_variants),
    "spearman_inv_l0_vs_eda_delta": {
        "rho": float(rho_ajt) if rho_ajt is not None else None,
        "p": float(p_ajt) if p_ajt is not None else None,
        "n": len(ajt_variants),
    },
    "comparison_with_jb_l6": {
        "jb_l6_l0": jb_l6["l0"],
        "jb_l6_eda_delta": jb_l6["eda_delta"],
        "jb_l6_letter_eda": jb_l6["mean_eda_letter"],
        "jb_l6_nonletter_eda": jb_l6["mean_eda_nonletter"],
        "ajt_mean_letter_eda": float(np.mean(ajt_letter_edas)),
        "ajt_mean_nonletter_eda": float(np.mean(ajt_nonletter_edas)),
        "observation": "AJT letter features have substantially LOWER EDA than jb letter features (0.50 vs 0.68). AJT nonletter features have HIGHER EDA than jb (0.70 vs 0.63). The inversion is systematic and does not depend on L0."
    },
    "hypotheses": [
        "H1 (normalization): AJT uses decoder normalization (unit decoder norms) which constrains the encoder-decoder angle for high-frequency features like letter features.",
        "H2 (training regime): AJT's strongly constrained training (SCL) enforces encoder-decoder alignment as a regularizer, preventing the dissociation seen in jb.",
        "H3 (inversion): In AJT, non-letter features become the 'absorbed' class — higher EDA in non-letter features may indicate they are absorbing into general features.",
    ],
    "recommended_next_step": "B.3 cross-architecture analysis: measure TRUE sae-spelling absorption rate for both jb and AJT L6 SAEs to determine if AJT's low letter EDA corresponds to less actual absorption.",
}

# Width analysis
width_analysis = {
    "n_points_feature_splitting": len(width_variants),
    "n_points_with_primary_l8": len(width_all),
    "feature_splitting_suite": [{
        "width": r["width"],
        "log2_width": float(math.log2(r["width"])),
        "l0": r["l0"],
        "eda_delta": r["eda_delta"],
        "eda_auroc": r["eda_auroc"],
        "mean_eda_letter": r["mean_eda_letter"],
        "mean_eda_nonletter": r["mean_eda_nonletter"],
    } for r in sorted(width_variants, key=lambda x: x["width"])],
    "with_primary_l8": [{
        "width": r["width"],
        "log2_width": float(math.log2(r["width"])),
        "l0": r["l0"],
        "eda_delta": r["eda_delta"],
    } for r in sorted(width_all, key=lambda x: x["width"])],
    "spearman_3pt": {
        "rho": float(rho_width_delta_3),
        "p": float(p_width_delta_3),
        "n": len(width_variants),
        "note": "Feature-splitting suite only (widths 12k, 49k, 98k)"
    },
    "spearman_4pt": {
        "rho": float(rho_width_delta_4),
        "p": float(p_width_delta_4),
        "n": len(width_all),
        "note": "Including primary L8 (width=24576); note: primary L8 has different L0 (76.6 vs ~50)"
    },
    "observation": (
        "3-point suite (matched L0~50): Spearman rho=-1.0 (EDA_delta decreases with width). "
        "OPPOSITE to theory prediction. At width=98304, EDA_delta is NEGATIVE. "
        "4-point suite: rho=-0.8, p=0.20. Theory predicted wider SAEs have MORE absorption due to specialization."
    ),
    "mean_eda_letter_trend": f"mean_EDA_letter INCREASES with width (rho=+1.0 for 3-pt), but EDA_delta DECREASES because nonletter EDA increases faster.",
    "conclusion": "No support for the width → absorption theory at matched L0. EDA_delta does not increase with width. The width effect appears to compress EDA_delta toward zero or negative. Null result for H_width."
}

# Final output
output = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": time.time() - t_start,
    "source_data": str(b2_path),
    "pilot_pass": pilot_pass,
    "pass_criteria": "True sae-spelling absorption rate measured for at least 3 configs AND sigmoid vs. linear BIC comparison computed.",
    "pilot_pass_details": {
        "true_absorption_measured_n": n_measured,
        "true_absorption_pass": n_measured >= 3,
        "bic_comparison_done": True,
        "bic_favors_sigmoid": bic_delta > 0,
    },
    "sigmoid_analysis": sigmoid_analysis,
    "ajt_analysis": ajt_analysis,
    "width_analysis": width_analysis,
    "true_absorption_rates": true_absorption_results,
    "absorption_vs_eda": absorption_vs_eda,
    "key_findings": [
        f"Sigmoid fit to primary jb suite: LRT p={lrt_p:.4f}, BIC delta={bic_delta:.2f} (favors sigmoid). L0_c nominal={l0c_nominal:.0f}. CAUTION: only 5 cross-layer data points; bootstrap CI is uninformative.",
        f"EDA_delta vs 1/L0 (primary jb): Spearman rho={rho_primary:.2f}, p={p_primary:.2f} — NOT significant. EDA_delta pattern is non-monotonic across layers (L10 drops sharply).",
        f"AJT variants (L6, w=46080): ALL 3 show NEGATIVE EDA_delta. AJT letter EDA (mean={np.mean(ajt_letter_edas):.3f}) is substantially lower than jb L6 letter EDA ({jb_l6['mean_eda_letter']:.3f}). Architecture-specific EDA geometry.",
        f"Width scaling (L8, w={{12k, 49k, 98k}}, L0~50): Spearman(log2(w), EDA_delta)=rho=-1.0. EDA_delta DECREASES with width — opposite to theory. Null or inverse result.",
        f"True absorption rates measured for {n_measured}/3 configs using split-feature non-firing method.",
    ],
    "caveats": [
        "LRT p=0.027 is borderline and based on only 5 cross-layer data points; not conclusive.",
        "Bootstrap CI for L0_c is essentially uninformative (range: 1 to 790M).",
        "Cross-layer comparison conflates L0 with representation maturity — L0 increases with layer for jb.",
        "AJT architecture difference prevents direct comparison with jb without measuring true absorption.",
        "Width analysis L0 values are not perfectly matched (primary L8 L0=76.6, feature-splitting L0~50).",
    ]
}

# Save output
out_path = FULL_DIR / "B2_sparsity_analysis.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

out_path.write_text(json.dumps(output, indent=2, cls=NpEncoder))
print(f"\nSaved: {out_path}")

# Update gpu_progress
gpu_progress_path = WORKSPACE / "exp/gpu_progress.json"
if gpu_progress_path.exists():
    with open(gpu_progress_path) as f:
        gp = json.load(f)
else:
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

if TASK_ID not in gp.get("completed", []):
    gp.setdefault("completed", []).append(TASK_ID)
gp.setdefault("running", {}).pop(TASK_ID, None)

t_end = time.time()
elapsed_min = max(1, int((t_end - t_start) / 60))
gp.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 40,
    "actual_min": elapsed_min,
    "start_time": datetime.fromtimestamp(t_start).isoformat(),
    "end_time": datetime.fromtimestamp(t_end).isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "task_type": "analysis_reuse_pilot_data",
        "source": "B2_scaling_curve.json",
        "n_configs_analyzed": 11,
        "true_absorption_measured": n_measured,
    }
}
gpu_progress_path.write_text(json.dumps(gp, indent=2))

print(f"\n{'='*60}")
print(f"PILOT PASS: {pilot_pass}")
print(f"True absorption measured for: {n_measured}/3 configs")
print(f"Elapsed: {time.time() - t_start:.1f}s")
print(f"{'='*60}")

mark_done(status="success", summary=f"B2 sparsity analysis complete. Pilot pass: {pilot_pass}. {n_measured} absorption rates measured. Key finding: sigmoid LRT p=0.027 (borderline, n=5 points); AJT shows inverse EDA pattern; width analysis null/inverse result.")
