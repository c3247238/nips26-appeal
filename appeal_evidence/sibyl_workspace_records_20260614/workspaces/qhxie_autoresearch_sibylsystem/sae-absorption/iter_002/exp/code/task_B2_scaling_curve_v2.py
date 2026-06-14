"""
Task B2 (PILOT v2): Sparsity-Absorption Scaling Curve

For all available GPT-2 Small SAEs at different L0 values, measure absorption-related
metrics on the first-letter task. Uses the probe-based letter feature identification
approach from pilot_A_pipeline.py.

Key design:
- Letter features are identified via probe-decoder cosine alignment (same method as Pilot A)
- EDA (encoder-decoder asymmetry = 1 - cos(enc_j, dec_j)) is the working absorption metric
- Absorption rate proxy: fraction of letter features with EDA above threshold (0.5)
- Multiple SAE configs give us different L0 points

Data points from gpt2-small-res-jb across layers:
  L0 varies from ~13 (layer 0) to ~65 (layer 8)
  But letter features only emerge in mid layers (6, 10)
  Use layers 6, 10 as primary anchors; supplement with ajt variants at same layers

Additional data from gpt2-small-res-jb-feature-splitting (layer 8):
  Varying widths (768 to 98304) at roughly same L0 window

PILOT MODE: Use fast probe training on small word vocab.
Pass criteria: At least 3 different L0 settings measured with letter features identified.

Output: exp/results/pilots/pilot_B2_scaling_curve.json
         exp/results/full/B2_scaling_curve.json (if pilot passes)
"""

import os
import sys
import json
import time
import random
import string
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats, optimize
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.linear_model as sklm

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


TOTAL_STEPS = 13
report_progress(0, TOTAL_STEPS, "Starting B2 Sparsity-Absorption Scaling Curve v2")

# ─── Step 1: Load GPT-2 Small model ─────────────────────────────────────────
report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small transformer model")

from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained(
    "gpt2", center_writing_weights=True, center_unembed=True,
    fold_ln=True, refactor_factored_attn_matrices=True
)
model.eval()
model.to(DEVICE)
tokenizer = model.tokenizer
print(f"Model loaded: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

# ─── Step 2: Build probe training word vocabulary ────────────────────────────
report_progress(2, TOTAL_STEPS, "Building probe training vocabulary")

SIMPLE_WORDS = [
    "able", "above", "act", "add", "age", "air", "all", "also", "area", "back",
    "bad", "bag", "ball", "base", "bear", "bed", "big", "bird", "blow", "blue",
    "boat", "body", "book", "born", "box", "boy", "break", "bring", "burn", "busy",
    "call", "camp", "card", "care", "cat", "city", "clean", "clear", "close", "coat",
    "cold", "come", "cook", "cool", "cut", "dark", "data", "date", "deal", "deep",
    "desk", "die", "dirt", "dish", "door", "down", "draw", "drop", "duck", "dust",
    "earn", "east", "edge", "end", "even", "ever", "face", "fact", "fail", "fall",
    "far", "farm", "fast", "feel", "feet", "file", "fill", "film", "find", "fire",
    "fish", "five", "flag", "flat", "flow", "food", "foot", "form", "four", "free",
    "fuel", "full", "gain", "game", "gave", "girl", "give", "glad", "goal", "gold",
    "good", "gray", "grew", "grow", "hack", "hair", "half", "hall", "hand", "hang",
    "hard", "harm", "hate", "have", "head", "heal", "hear", "heat", "help", "here",
    "high", "hill", "hint", "hire", "hold", "hole", "home", "hook", "hope", "horn",
    "host", "hour", "hunt", "hurt", "idea", "inch", "iron", "jail", "join", "joke",
    "just", "keen", "keep", "kick", "kill", "kind", "king", "knew", "know", "lack",
    "lake", "land", "lane", "last", "late", "lead", "leaf", "lean", "left", "less",
    "lick", "lift", "like", "lime", "line", "link", "lion", "list", "load", "lock",
    "long", "look", "loop", "lose", "loss", "lost", "loud", "love", "luck", "made",
    "mail", "main", "make", "mark", "mass", "mate", "mean", "meat", "meet", "melt",
    "mild", "milk", "mill", "mine", "miss", "mode", "moon", "more", "most", "move",
    "much", "must", "name", "near", "need", "nest", "news", "next", "nice", "nine",
    "node", "none", "noon", "norm", "nose", "note", "noun", "once", "only", "open",
    "over", "pace", "pack", "page", "pain", "pair", "palm", "park", "part", "pass",
    "past", "path", "peak", "pick", "pill", "pine", "pink", "pipe", "plan", "play",
    "plot", "plug", "plus", "poem", "poll", "pond", "pool", "poor", "port", "pose",
    "post", "pour", "pull", "pump", "pure", "push", "race", "rack", "rain", "rank",
    "rate", "read", "real", "rent", "rest", "rice", "rich", "ride", "ring", "rise",
    "risk", "road", "rock", "role", "roll", "roof", "room", "root", "rope", "rose",
    "ruin", "rule", "rush", "rust", "safe", "sail", "salt", "same", "sand", "save",
    "seal", "seek", "sell", "send", "ship", "shoe", "shot", "show", "sick", "side",
    "silk", "sing", "sink", "site", "size", "skip", "slim", "slip", "slow", "snow",
    "sock", "soft", "soil", "sole", "some", "song", "soon", "soul", "spot", "stem",
    "step", "stop", "suit", "swim", "tail", "take", "talk", "tall", "tank", "task",
    "team", "tear", "tell", "test", "text", "thin", "tick", "tile", "time", "tire",
    "told", "tone", "tool", "toss", "tour", "town", "trap", "tree", "trim", "trip",
    "true", "tube", "tune", "turn", "type", "unit", "upon", "used", "vary", "vast",
    "very", "vest", "view", "vine", "vote", "wade", "wage", "wake", "walk", "wall",
    "warm", "warn", "wash", "wave", "wear", "weed", "week", "went", "west", "wide",
    "wild", "will", "wind", "wine", "wing", "wire", "wish", "wood", "word", "work",
]

valid_words = []
for word in SIMPLE_WORDS:
    word = word.strip().lower()
    if not word.isalpha() or len(word) < 2:
        continue
    try:
        if len(tokenizer.encode(" " + word)) == 1 and len(tokenizer.encode(word)) == 1:
            valid_words.append(word)
    except:
        pass

vocab_by_letter = {lt: [] for lt in string.ascii_lowercase}
for word in valid_words:
    vocab_by_letter[word[0]].append(word)
good_letters = {lt: ws for lt, ws in vocab_by_letter.items() if len(ws) >= 5}
print(f"Single-token words: {len(valid_words)}, letters with >=5 words: {len(good_letters)}")

rng_random = random.Random(SEED)
probe_train_words = []
for lt in sorted(good_letters.keys()):
    ws = good_letters[lt]
    probe_train_words.extend(rng_random.sample(ws, min(len(ws), 30)))
print(f"Probe training words: {len(probe_train_words)}")


def train_letter_probes(model, probe_train_words, hook_name, device, seed=42):
    """Train logistic regression probes for each letter at given layer."""
    all_acts_list = []
    all_word_list = []

    with torch.no_grad():
        for word in probe_train_words:
            prompt = f" {word}:"
            try:
                tok = model.to_tokens(prompt)
                _, cache = model.run_with_cache(tok, names_filter=hook_name)
                act = cache[hook_name][0, -2, :].cpu().float().numpy()
                all_acts_list.append(act)
                all_word_list.append(word)
                del cache
            except:
                pass

    if len(all_acts_list) < 10:
        return {}, []

    all_acts = np.stack(all_acts_list)
    first_letters_arr = np.array([w[0] for w in all_word_list])

    letter_probe_dirs = {}
    letters_with_probes = []
    for letter in sorted(good_letters.keys()):
        y = (first_letters_arr == letter).astype(int)
        if y.sum() < 3 or (1-y).sum() < 3:
            continue
        try:
            clf = sklm.LogisticRegression(C=1.0, max_iter=300, random_state=seed,
                                           solver='lbfgs')
            clf.fit(all_acts, y)
            probe_dir = clf.coef_[0]
            probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
            letter_probe_dirs[letter] = probe_dir
            letters_with_probes.append(letter)
        except:
            pass

    return letter_probe_dirs, letters_with_probes


def identify_letter_features_from_probes(sae_w_dec, letter_probe_dirs, n_pos_target=67):
    """
    Identify letter features via probe-decoder cosine similarity.
    Returns: letter_feature_ids, threshold, n_pos, max_probe_cos
    """
    letters = sorted(letter_probe_dirs.keys())
    probe_dirs = np.stack([letter_probe_dirs[lt] for lt in letters])  # (n_letters, d_model)

    W_dec_norm = F.normalize(sae_w_dec, dim=1).cpu().float().numpy()  # (d_sae, d_model)
    cos_probe_dec = probe_dirs @ W_dec_norm.T  # (n_letters, d_sae)
    max_probe_cos = cos_probe_dec.max(axis=0)  # (d_sae,)

    # Find threshold closest to n_pos_target
    best_thr = 0.3
    for thr in np.arange(0.20, 0.55, 0.01):
        n = (max_probe_cos >= thr).sum()
        if abs(n - n_pos_target) < abs((max_probe_cos >= best_thr).sum() - n_pos_target):
            best_thr = thr

    letter_feature_ids = np.where(max_probe_cos >= best_thr)[0].tolist()
    n_pos = len(letter_feature_ids)

    return letter_feature_ids, float(best_thr), n_pos, max_probe_cos


def compute_eda_scores(sae):
    """Compute EDA = 1 - cos(enc_j, dec_j) for each feature."""
    with torch.no_grad():
        W_enc = sae.W_enc.detach().float()  # (d_in, d_sae)
        w_enc = W_enc.T  # (d_sae, d_in)
        w_dec = sae.W_dec.detach().float()  # (d_sae, d_in)
        enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)
        cos_ed = ((w_enc * w_dec).sum(dim=1) / (enc_norms * dec_norms)).cpu().numpy()
        eda_scores = 1.0 - cos_ed
    return eda_scores


def compute_empirical_l0(sae, model, layer, device, n_tokens=500):
    """Compute empirical L0 from feature activation frequency."""
    from sae_lens import SAE as SAEClass
    # Use sparsity tensor if available
    try:
        _, _, sparsity = SAEClass.from_pretrained_with_cfg_and_sparsity(
            sae.cfg.metadata.get("model_name", ""), sae.cfg.metadata.get("hook_name", "")
        )
        if sparsity is not None:
            log_freqs = sparsity.numpy()
            freqs = np.exp(log_freqs)
            l0 = float(freqs.sum())
            return l0
    except:
        pass

    # Fall back to empirical measurement from small text batch
    hook_name = f"blocks.{layer}.hook_resid_pre"
    # Use simple text
    texts = ["The quick brown fox jumps over the lazy dog. " * 5]
    n_measured = 0
    total_active = 0

    with torch.no_grad():
        for text in texts:
            try:
                tokens = model.to_tokens(text)[:, :64]
                _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                resid = cache[hook_name][0]  # (seq, d_model)

                # Encode through SAE
                sae_out = sae.encode(resid)
                if isinstance(sae_out, tuple):
                    acts = sae_out[1] if len(sae_out) > 1 else sae_out[0]
                else:
                    acts = sae_out

                total_active += (acts > 0).float().sum().item()
                n_measured += resid.shape[0]
                del cache
            except Exception as e:
                pass

    if n_measured > 0:
        return total_active / n_measured
    return None


# ─── Step 3: Define SAE configurations ──────────────────────────────────────
report_progress(3, TOTAL_STEPS, "Defining SAE configurations for scaling curve")

# Primary strategy: use gpt2-small-res-jb at layers where letter features emerge
# Secondary: use ajt variants (different architecture/training) at L6 and L10
# Tertiary: feature-splitting at L8 for width variation

SAE_CONFIGS = [
    # Primary: gpt2-small-res-jb at multiple layers (same width=24576, different L0)
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.10.hook_resid_pre",
     "layer": 10, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.8.hook_resid_pre",
     "layer": 8, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.4.hook_resid_pre",
     "layer": 4, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.2.hook_resid_pre",
     "layer": 2, "width": 24576, "group": "primary"},
    # Secondary: ajt variants at L6 (varying L0 via different architectures)
    {"release": "gpt2-small-res_sce-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": None, "group": "ajt"},
    {"release": "gpt2-small-res_scl-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": None, "group": "ajt"},
    {"release": "gpt2-small-res_sle-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": None, "group": "ajt"},
    # Tertiary: feature-splitting at L8 (varying width)
    {"release": "gpt2-small-res-jb-feature-splitting", "sae_id": "blocks.8.hook_resid_pre_12288",
     "layer": 8, "width": 12288, "group": "width"},
    {"release": "gpt2-small-res-jb-feature-splitting", "sae_id": "blocks.8.hook_resid_pre_49152",
     "layer": 8, "width": 49152, "group": "width"},
    {"release": "gpt2-small-res-jb-feature-splitting", "sae_id": "blocks.8.hook_resid_pre_98304",
     "layer": 8, "width": 98304, "group": "width"},
]

# ─── Step 4: Pre-compute probes for layers we'll test ───────────────────────
report_progress(4, TOTAL_STEPS, "Pre-computing letter probes for target layers")

# We need probes at each target layer
layer_probes = {}
for layer in [2, 4, 6, 8, 10]:
    hook_name = f"blocks.{layer}.hook_resid_pre"
    print(f"  Training probes for layer {layer}...")
    probe_dirs, letters_with_probes = train_letter_probes(
        model, probe_train_words, hook_name, DEVICE, seed=SEED
    )
    layer_probes[layer] = {"dirs": probe_dirs, "letters": letters_with_probes}
    print(f"    Layer {layer}: {len(letters_with_probes)} letters with probes")

# ─── Step 5: Process each SAE config ─────────────────────────────────────────
report_progress(5, TOTAL_STEPS, "Processing SAE configs - identifying letter features and EDA")

scaling_curve_data = []

for cfg in SAE_CONFIGS:
    release = cfg["release"]
    sae_id = cfg["sae_id"]
    layer = cfg["layer"]
    width = cfg["width"]
    group = cfg["group"]

    print(f"\n  [{group}] {release} | {sae_id} | layer={layer}")

    try:
        sae_obj = SAE.from_pretrained(release, sae_id)
        sae_obj.eval()
        sae_obj.to(DEVICE)

        d_sae = sae_obj.cfg.d_sae
        if width is None:
            width = d_sae
        print(f"    d_sae={d_sae}")

        # Get letter probes for this layer
        probe_data = layer_probes.get(layer, {})
        probe_dirs = probe_data.get("dirs", {})
        letters_with_probes = probe_data.get("letters", [])

        if len(probe_dirs) < 5:
            print(f"    SKIP: Only {len(probe_dirs)} letters have probes at layer {layer}")
            scaling_curve_data.append({
                "release": release, "sae_id": sae_id, "layer": layer, "width": width,
                "group": group, "status": "skipped",
                "reason": f"Only {len(probe_dirs)} probes at layer {layer}"
            })
            sae_obj.cpu()
            del sae_obj
            torch.cuda.empty_cache()
            continue

        # Identify letter features
        letter_feature_ids, threshold, n_pos, max_probe_cos = identify_letter_features_from_probes(
            sae_obj.W_dec.data.to(DEVICE), probe_dirs, n_pos_target=min(67, d_sae // 200)
        )
        print(f"    Letter features: n_pos={n_pos} at threshold={threshold:.2f}")

        if n_pos < 10:
            print(f"    SKIP: Only {n_pos} letter features identified")
            scaling_curve_data.append({
                "release": release, "sae_id": sae_id, "layer": layer, "width": width,
                "group": group, "status": "skipped",
                "reason": f"Only {n_pos} letter features (need >= 10)"
            })
            sae_obj.cpu()
            del sae_obj
            torch.cuda.empty_cache()
            continue

        # Compute EDA scores
        eda_scores = compute_eda_scores(sae_obj)
        letter_eda_scores = eda_scores[letter_feature_ids]
        non_letter_eda = eda_scores[[i for i in range(d_sae) if i not in set(letter_feature_ids)]]

        # Absorption rate proxy: mean EDA of letter features
        mean_eda_letter = float(letter_eda_scores.mean())
        mean_eda_non_letter = float(non_letter_eda.mean())

        # Fraction of letter features with EDA > 0.5 (above-median threshold)
        eda_threshold = 0.5
        absorption_rate_eda = float((letter_eda_scores > eda_threshold).mean())

        # AUROC of EDA for binary classification (letter vs non-letter)
        binary_labels = np.zeros(d_sae)
        binary_labels[letter_feature_ids] = 1
        try:
            eda_auroc = float(roc_auc_score(binary_labels, eda_scores))
        except:
            eda_auroc = None

        # Empirical L0 from frequency measurement
        hook_name = f"blocks.{layer}.hook_resid_pre"
        l0 = compute_empirical_l0(sae_obj, model, layer, DEVICE)
        l0_str = f"{l0:.2f}" if l0 else "N/A"
        eda_auroc_str = f"{eda_auroc:.4f}" if eda_auroc else "N/A"
        print(f"    L0={l0_str}, "
              f"mean_eda_letter={mean_eda_letter:.4f}, "
              f"absorption_rate(EDA>0.5)={absorption_rate_eda:.4f}, "
              f"EDA_AUROC={eda_auroc_str}")

        scaling_curve_data.append({
            "release": release,
            "sae_id": sae_id,
            "layer": layer,
            "width": width,
            "group": group,
            "status": "success",
            "d_sae": d_sae,
            "n_pos_letter_features": n_pos,
            "probe_threshold": threshold,
            "l0": l0,
            "inv_l0": 1.0 / l0 if l0 and l0 > 0 else None,
            "lambda_empirical": 1.0 / l0 if l0 and l0 > 0 else None,
            "mean_eda_letter": mean_eda_letter,
            "mean_eda_non_letter": mean_eda_non_letter,
            "eda_delta": mean_eda_letter - mean_eda_non_letter,
            "absorption_rate_eda": absorption_rate_eda,  # Fraction of letter features with high EDA
            "eda_auroc": eda_auroc,
            "letter_eda_stats": {
                "mean": float(letter_eda_scores.mean()),
                "std": float(letter_eda_scores.std()),
                "median": float(np.median(letter_eda_scores)),
                "q75": float(np.percentile(letter_eda_scores, 75))
            }
        })

        sae_obj.cpu()
        del sae_obj
        torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        traceback.print_exc()
        scaling_curve_data.append({
            "release": release, "sae_id": sae_id, "layer": layer, "width": width,
            "group": group, "status": "error", "error": str(e)
        })

# ─── Step 6: Filter valid points ────────────────────────────────────────────
report_progress(6, TOTAL_STEPS, "Filtering valid data points")

valid_points = [d for d in scaling_curve_data
                if d.get("status") == "success" and d.get("l0") is not None
                and d.get("absorption_rate_eda") is not None]

print(f"\nValid data points: {len(valid_points)}")
for p in sorted(valid_points, key=lambda x: x["l0"]):
    print(f"  [{p['group']}] layer={p['layer']}, width={p['width']}, "
          f"L0={p['l0']:.2f}, 1/L0={p['inv_l0']:.4f}, "
          f"abs_rate={p['absorption_rate_eda']:.4f}, "
          f"mean_EDA={p['mean_eda_letter']:.4f}")

pilot_pass = len(valid_points) >= 3
print(f"\nPilot pass (>=3 valid points): {'PASS' if pilot_pass else 'FAIL'}")

# ─── Step 7: Fit curves if enough data ──────────────────────────────────────
report_progress(7, TOTAL_STEPS, "Fitting curve models")

curve_fit_results = {}

if len(valid_points) >= 3:
    # Primary analysis: same-width points (gpt2-small-res-jb, width=24576)
    primary_pts = [p for p in valid_points if p["width"] == 24576 and p["group"] == "primary"]
    if len(primary_pts) < 3:
        primary_pts = valid_points  # Use all

    # Use absorption_rate_eda as the primary metric
    inv_l0_arr = np.array([p["inv_l0"] for p in primary_pts])
    abs_rate_arr = np.array([p["absorption_rate_eda"] for p in primary_pts])

    # Sort
    sort_idx = np.argsort(inv_l0_arr)
    inv_l0_arr = inv_l0_arr[sort_idx]
    abs_rate_arr = abs_rate_arr[sort_idx]

    n_pts = len(inv_l0_arr)
    print(f"\nFitting curves to {n_pts} primary data points")

    # Compute SS_tot for R^2
    ss_tot = np.sum((abs_rate_arr - abs_rate_arr.mean())**2)

    # Model 1: Linear
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(inv_l0_arr, abs_rate_arr)
        y_pred = slope * inv_l0_arr + intercept
        ss_res = np.sum((abs_rate_arr - y_pred)**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        sigma2 = ss_res / n_pts if n_pts > 0 else 1e-10
        ll = -n_pts/2 * np.log(2*np.pi*max(sigma2, 1e-10)) - ss_res/(2*max(sigma2, 1e-10))
        k = 2
        bic = k * np.log(n_pts) - 2 * ll
        aic = 2 * k - 2 * ll
        curve_fit_results["linear"] = {
            "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2), "r_value": float(r_value), "p_value": float(p_value),
            "bic": float(bic), "aic": float(aic), "log_likelihood": float(ll), "n_params": k
        }
        print(f"  Linear: slope={slope:.4f}, R2={r2:.4f}, p={p_value:.4f}")
    except Exception as e:
        curve_fit_results["linear"] = {"error": str(e)}
        print(f"  Linear error: {e}")

    # Model 2: Power-law
    if n_pts >= 3 and np.all(inv_l0_arr > 0):
        try:
            from scipy.optimize import curve_fit as scipy_curve_fit
            def powerlaw(x, a, b):
                return a * np.power(np.abs(x), b)
            popt, _ = scipy_curve_fit(powerlaw, inv_l0_arr, abs_rate_arr,
                                       p0=[1.0, 0.5], maxfev=5000)
            y_pred = powerlaw(inv_l0_arr, *popt)
            ss_res = np.sum((abs_rate_arr - y_pred)**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            sigma2 = ss_res / n_pts
            ll = (-n_pts/2 * np.log(2*np.pi*max(sigma2, 1e-10))
                  - ss_res/(2*max(sigma2, 1e-10)))
            k = 2
            bic = k * np.log(n_pts) - 2 * ll
            aic = 2 * k - 2 * ll
            curve_fit_results["power_law"] = {
                "a": float(popt[0]), "b": float(popt[1]),
                "r2": float(r2), "bic": float(bic), "aic": float(aic),
                "log_likelihood": float(ll), "n_params": k
            }
            print(f"  Power-law: a={popt[0]:.4f}, b={popt[1]:.4f}, R2={r2:.4f}")
        except Exception as e:
            curve_fit_results["power_law"] = {"error": str(e)}

    # Model 3: Sigmoid (needs >=4 points for meaningful fit)
    if n_pts >= 4:
        try:
            from scipy.optimize import curve_fit as scipy_curve_fit
            def sigmoid(x, L, k, x0):
                return L / (1 + np.exp(-k * (x - x0)))
            L_init = abs_rate_arr.max()
            k_init = 100.0
            x0_init = np.median(inv_l0_arr)
            popt, _ = scipy_curve_fit(
                sigmoid, inv_l0_arr, abs_rate_arr,
                p0=[L_init, k_init, x0_init], maxfev=10000,
                bounds=([0, 0, 0], [1, 10000, 1])
            )
            y_pred = sigmoid(inv_l0_arr, *popt)
            ss_res = np.sum((abs_rate_arr - y_pred)**2)
            r2 = 1 - ss_res / (ss_tot + 1e-10)
            sigma2 = ss_res / n_pts
            ll_sig = (-n_pts/2 * np.log(2*np.pi*max(sigma2, 1e-10))
                      - ss_res/(2*max(sigma2, 1e-10)))
            k_sig = 3
            bic_sig = k_sig * np.log(n_pts) - 2 * ll_sig
            aic_sig = 2 * k_sig - 2 * ll_sig

            # LRT
            ll_lin = curve_fit_results.get("linear", {}).get("log_likelihood", None)
            lrt_stat = 2 * (ll_sig - ll_lin) if ll_lin is not None else None
            lrt_p = float(1 - stats.chi2.cdf(lrt_stat, df=1)) if lrt_stat and lrt_stat > 0 else None

            inflection_l0c = 1.0 / popt[2] if popt[2] > 0 else None
            curve_fit_results["sigmoid"] = {
                "L": float(popt[0]), "k": float(popt[1]), "x0": float(popt[2]),
                "r2": float(r2), "bic": float(bic_sig), "aic": float(aic_sig),
                "log_likelihood": float(ll_sig), "n_params": k_sig,
                "inflection_inv_l0": float(popt[2]),
                "inflection_l0_c": float(inflection_l0c) if inflection_l0c else None,
                "lrt_stat": float(lrt_stat) if lrt_stat else None,
                "lrt_pvalue": float(lrt_p) if lrt_p else None
            }
            print(f"  Sigmoid: L={popt[0]:.4f}, k={popt[1]:.2f}, x0={popt[2]:.4f}, R2={r2:.4f}")
        except Exception as e:
            curve_fit_results["sigmoid"] = {"error": str(e)}
            print(f"  Sigmoid error: {e}")
    else:
        curve_fit_results["sigmoid"] = {"error": f"Need >=4 data points, have {n_pts}"}
        print(f"  Sigmoid: skipped (need >=4, have {n_pts})")

    # Also fit using mean_eda_letter as alternative metric
    print("\n  Alternative metric: mean_EDA_letter")
    eda_arr = np.array([p["mean_eda_letter"] for p in primary_pts])
    eda_arr = eda_arr[sort_idx]  # Same sort order
    if n_pts >= 2:
        try:
            slope_e, intercept_e, r_e, p_e, _ = linregress(inv_l0_arr, eda_arr)
            r2_e = r_e**2
            curve_fit_results["linear_mean_eda"] = {
                "slope": float(slope_e), "intercept": float(intercept_e),
                "r2": float(r2_e), "r_value": float(r_e), "p_value": float(p_e)
            }
            print(f"    Linear (mean EDA): slope={slope_e:.4f}, R2={r2_e:.4f}, p={p_e:.4f}")
        except Exception as e:
            print(f"    Linear (mean EDA) error: {e}")

# ─── Step 8: Bootstrap CI ───────────────────────────────────────────────────
report_progress(8, TOTAL_STEPS, "Bootstrap CI for inflection point")

bootstrap_results = {"skipped": "Need >=4 valid points or sigmoid fit for bootstrap"}

if (len(valid_points) >= 4 and "sigmoid" in curve_fit_results and
        "error" not in curve_fit_results.get("sigmoid", {})):
    from scipy.optimize import curve_fit as scipy_curve_fit

    def sigmoid(x, L, k, x0):
        return L / (1 + np.exp(-k * (x - x0)))

    primary_pts = [p for p in valid_points if p["width"] == 24576 and p["group"] == "primary"]
    if len(primary_pts) < 4:
        primary_pts = valid_points

    inv_l0_arr = np.array([p["inv_l0"] for p in primary_pts])
    abs_rate_arr = np.array([p["absorption_rate_eda"] for p in primary_pts])

    boot_inflections = []
    rng_boot = np.random.RandomState(SEED)
    for _ in range(1000):
        idx = rng_boot.choice(len(primary_pts), len(primary_pts), replace=True)
        x_b, y_b = inv_l0_arr[idx], abs_rate_arr[idx]
        if len(np.unique(x_b)) < 3:
            continue
        try:
            popt, _ = scipy_curve_fit(
                sigmoid, x_b, y_b,
                p0=[y_b.max(), 100.0, np.median(x_b)],
                maxfev=5000, bounds=([0, 0, 0], [1, 10000, 1])
            )
            if popt[2] > 0:
                boot_inflections.append(1.0 / popt[2])
        except:
            pass

    if len(boot_inflections) >= 50:
        bootstrap_results = {
            "n_successful": len(boot_inflections),
            "inflection_l0c_mean": float(np.mean(boot_inflections)),
            "ci_low": float(np.percentile(boot_inflections, 2.5)),
            "ci_high": float(np.percentile(boot_inflections, 97.5)),
            "nominal": curve_fit_results["sigmoid"].get("inflection_l0_c")
        }
        print(f"Bootstrap CI: {bootstrap_results['inflection_l0c_mean']:.1f} "
              f"[{bootstrap_results['ci_low']:.1f}, {bootstrap_results['ci_high']:.1f}]")
    else:
        bootstrap_results = {"error": f"Only {len(boot_inflections)} successful bootstrap fits"}

# ─── Step 9: Width analysis ──────────────────────────────────────────────────
report_progress(9, TOTAL_STEPS, "Width analysis (feature-splitting)")

width_pts = [p for p in valid_points if p["group"] == "width"]
same_layer_pts = [p for p in valid_points if p["layer"] == 8 and p["group"] == "primary"]
all_width_pts = width_pts + same_layer_pts
width_analysis = {
    "n_points": len(all_width_pts),
    "points": [{"width": p["width"], "l0": p["l0"], "inv_l0": p["inv_l0"],
                 "absorption_rate": p["absorption_rate_eda"],
                 "mean_eda": p["mean_eda_letter"]}
                for p in sorted(all_width_pts, key=lambda x: x["width"])]
}
print(f"Width analysis: {len(all_width_pts)} data points")

# ─── Step 10: H1 prediction check ───────────────────────────────────────────
report_progress(10, TOTAL_STEPS, "H1 quantitative prediction check")

b1_path = FULL_RESULTS_DIR / "B1_decoder_geometry.json"
h1_check = {}
if b1_path.exists():
    with open(b1_path) as f:
        b1 = json.load(f)
    l6 = b1.get("layer6", {})
    pa = l6.get("pair_level_analysis", {})
    pos_cos2_mean = pa.get("max_cos2", {}).get("pos_mean")
    if pos_cos2_mean:
        lambda_star = 1 - pos_cos2_mean
        l0c_pred = 1.0 / lambda_star if lambda_star > 0 else None
        l0c_obs = curve_fit_results.get("sigmoid", {}).get("inflection_l0_c")
        h1_check = {
            "pos_cos2_mean": float(pos_cos2_mean),
            "lambda_star": float(lambda_star),
            "predicted_l0c": float(l0c_pred) if l0c_pred else None,
            "observed_l0c": float(l0c_obs) if l0c_obs else None,
            "ratio": float(l0c_obs / l0c_pred) if (l0c_obs and l0c_pred and l0c_pred > 0) else None
        }
        l0c_pred_str = f"{l0c_pred:.1f}" if l0c_pred else "N/A"
        print(f"H1 prediction: lambda*={lambda_star:.4f}, predicted L0_c={l0c_pred_str}")
        if l0c_obs:
            ratio_str = f"{h1_check['ratio']:.2f}" if h1_check['ratio'] else "N/A"
            print(f"Observed L0_c={l0c_obs:.1f}, ratio={ratio_str}")

# ─── Step 11: Wilcoxon test across all valid configs ────────────────────────
report_progress(11, TOTAL_STEPS, "Wilcoxon test: EDA vs L0 correlation")

# Test if mean EDA increases with 1/L0 (higher 1/L0 = lower L0 = more sparsity = more absorption)
wilcoxon_results = {}
if len(valid_points) >= 3:
    inv_l0s = np.array([p["inv_l0"] for p in valid_points])
    mean_edas = np.array([p["mean_eda_letter"] for p in valid_points])
    abs_rates = np.array([p["absorption_rate_eda"] for p in valid_points])

    from scipy.stats import spearmanr
    spearman_eda, spearman_p_eda = spearmanr(inv_l0s, mean_edas)
    spearman_abs, spearman_p_abs = spearmanr(inv_l0s, abs_rates)

    wilcoxon_results = {
        "spearman_inv_l0_vs_mean_eda": {"rho": float(spearman_eda), "p": float(spearman_p_eda)},
        "spearman_inv_l0_vs_abs_rate": {"rho": float(spearman_abs), "p": float(spearman_p_abs)},
        "n_points": len(valid_points)
    }
    print(f"Spearman(1/L0, mean EDA): rho={spearman_eda:.4f}, p={spearman_p_eda:.4f}")
    print(f"Spearman(1/L0, abs_rate): rho={spearman_abs:.4f}, p={spearman_p_abs:.4f}")

# ─── Step 12: Compile results ────────────────────────────────────────────────
report_progress(12, TOTAL_STEPS, "Compiling results")

elapsed = time.time() - start_time

primary_finding = "No valid data points"
if valid_points:
    if wilcoxon_results.get("spearman_inv_l0_vs_abs_rate", {}).get("rho", 0) > 0.3:
        direction = "positive"
    elif wilcoxon_results.get("spearman_inv_l0_vs_abs_rate", {}).get("rho", 0) < -0.3:
        direction = "negative (unexpected)"
    else:
        direction = "weak/no"
    primary_finding = (f"{len(valid_points)} configs measured, "
                       f"Spearman(1/L0, abs_rate)="
                       f"{wilcoxon_results.get('spearman_inv_l0_vs_abs_rate', {}).get('rho', 0):.3f}, "
                       f"{direction} correlation with sparsity")

results = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "sae_release_primary": "gpt2-small-res-jb",
        "seed": SEED,
        "n_configs_tested": len(SAE_CONFIGS),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    },
    "pilot_pass": pilot_pass,
    "pass_criteria": ">=3 different L0 settings measured",
    "n_valid_points": len(valid_points),
    "valid_data_points": valid_points,
    "scaling_curve_data": scaling_curve_data,
    "curve_fit_results": curve_fit_results,
    "bootstrap_ci": bootstrap_results,
    "width_analysis": width_analysis,
    "h1_prediction_check": h1_check,
    "wilcoxon_results": wilcoxon_results,
    "primary_finding": primary_finding,
    "summary": {
        "n_configs_tested": len(SAE_CONFIGS),
        "n_valid": len(valid_points),
        "l0_range": ([min(p["l0"] for p in valid_points),
                      max(p["l0"] for p in valid_points)] if valid_points else None),
        "abs_rate_range": ([min(p["absorption_rate_eda"] for p in valid_points),
                            max(p["absorption_rate_eda"] for p in valid_points)]
                           if valid_points else None),
        "pilot_pass": pilot_pass
    }
}

OUTPUT_FILE_PILOT.write_text(json.dumps(results, indent=2))
print(f"Pilot results saved: {OUTPUT_FILE_PILOT}")

if pilot_pass:
    OUTPUT_FILE_FULL.write_text(json.dumps(results, indent=2))
    print(f"Full results saved: {OUTPUT_FILE_FULL}")

# ─── Step 13: Mark done ──────────────────────────────────────────────────────
report_progress(13, TOTAL_STEPS, f"Done! Pilot {'PASS' if pilot_pass else 'FAIL'}")

mark_done(
    status="success" if pilot_pass else "partial",
    summary=(f"B2: {len(valid_points)} valid configs, "
             f"L0=[{results['summary']['l0_range']}], "
             f"abs_rate=[{results['summary']['abs_rate_range']}]"
             if valid_points else "B2: No valid data points"),
    result=results["summary"]
)

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
if gpu_progress_path.exists():
    with open(gpu_progress_path) as f:
        gp = json.load(f)
else:
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

if pilot_pass:
    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
else:
    if TASK_ID not in gp["failed"]:
        gp["failed"].append(TASK_ID)

gp["running"].pop(TASK_ID, None)
gp["timings"][TASK_ID] = {
    "planned_min": 40, "actual_min": int(elapsed / 60),
    "start_time": datetime.fromtimestamp(start_time).isoformat(),
    "end_time": datetime.now().isoformat(),
    "config_snapshot": {
        "model": "gpt2-small", "n_sae_configs": len(SAE_CONFIGS),
        "n_valid_points": len(valid_points), "pilot_pass": pilot_pass,
        "gpu_model": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    }
}

with open(gpu_progress_path, "w") as f:
    json.dump(gp, f, indent=2)

print(f"\n{'='*60}")
print(f"Task B2 Scaling Curve COMPLETE")
print(f"Pilot pass: {pilot_pass}")
print(f"Valid configs: {len(valid_points)}/{len(SAE_CONFIGS)}")
if valid_points:
    print(f"L0 range: {results['summary']['l0_range']}")
    print(f"Abs rate range: {results['summary']['abs_rate_range']}")
print(f"Primary finding: {primary_finding}")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"{'='*60}")
