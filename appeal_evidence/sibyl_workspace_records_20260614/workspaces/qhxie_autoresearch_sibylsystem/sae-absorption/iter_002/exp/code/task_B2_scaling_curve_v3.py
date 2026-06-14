"""
Task B2 (PILOT v3): Sparsity-Absorption Scaling Curve — Fixed version

Uses mean_eda_letter as primary absorption proxy (continuous metric).
Also computes EDA_delta = mean_eda_letter - mean_eda_non_letter.

Key fixes from v2:
- Fixed f-string format specifier issues
- Use mean_eda_letter as primary metric (not binary threshold which saturates)
- EDA_delta = letter - non-letter mean EDA gap (more diagnostic)
- Proper AJT release L0 measurement
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
from scipy import stats
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


TOTAL_STEPS = 10
report_progress(0, TOTAL_STEPS, "Starting B2 Sparsity-Absorption Scaling Curve v3")

# ─── Step 1: Load model ──────────────────────────────────────────────────────
report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small")

from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained(
    "gpt2", center_writing_weights=True, center_unembed=True,
    fold_ln=True, refactor_factored_attn_matrices=True
)
model.eval().to(DEVICE)
tokenizer = model.tokenizer
print(f"d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

# ─── Step 2: Build word vocab and probes ────────────────────────────────────
report_progress(2, TOTAL_STEPS, "Building vocabulary and training probes")

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
    if not word.isalpha():
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
print(f"Vocab: {len(valid_words)} words, {len(good_letters)} letters with >=5 words")

rng_random = random.Random(SEED)
probe_train_words = []
for lt in sorted(good_letters.keys()):
    ws = good_letters[lt]
    probe_train_words.extend(rng_random.sample(ws, min(len(ws), 30)))


def train_probes_at_layer(model, words, layer, device, seed=42):
    """Train letter probes at given layer; return probe_dirs dict."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    acts_list, word_list = [], []
    with torch.no_grad():
        for word in words:
            try:
                tok = model.to_tokens(f" {word}:")
                _, cache = model.run_with_cache(tok, names_filter=hook_name)
                acts_list.append(cache[hook_name][0, -2, :].cpu().float().numpy())
                word_list.append(word)
                del cache
            except:
                pass
    if len(acts_list) < 10:
        return {}, []
    acts = np.stack(acts_list)
    first_letters = np.array([w[0] for w in word_list])
    probe_dirs, letters = {}, []
    for lt in sorted(good_letters.keys()):
        y = (first_letters == lt).astype(int)
        if y.sum() < 3 or (1-y).sum() < 3:
            continue
        try:
            clf = sklm.LogisticRegression(C=1.0, max_iter=300, random_state=seed)
            clf.fit(acts, y)
            d = clf.coef_[0]
            probe_dirs[lt] = d / (np.linalg.norm(d) + 1e-8)
            letters.append(lt)
        except:
            pass
    return probe_dirs, letters


# Pre-compute probes for key layers
print("Training probes for layers 2, 4, 6, 8, 10...")
layer_probes = {}
for layer in [2, 4, 6, 8, 10]:
    dirs, letters = train_probes_at_layer(model, probe_train_words, layer, DEVICE)
    layer_probes[layer] = {"dirs": dirs, "letters": letters}
    print(f"  Layer {layer}: {len(letters)} letters with probes")


def identify_letter_features(sae_w_dec, probe_dirs, n_pos_target=67):
    """Find SAE features whose decoder aligns with letter probes."""
    letters = sorted(probe_dirs.keys())
    probe_mat = np.stack([probe_dirs[lt] for lt in letters])  # (n_letters, d_model)
    W_dec_np = F.normalize(sae_w_dec.cpu().float(), dim=1).numpy()  # (d_sae, d_model)
    cos = probe_mat @ W_dec_np.T  # (n_letters, d_sae)
    max_cos = cos.max(axis=0)  # (d_sae,)

    best_thr = 0.30
    for thr in np.arange(0.20, 0.55, 0.01):
        n = int((max_cos >= thr).sum())
        if abs(n - n_pos_target) < abs(int((max_cos >= best_thr).sum()) - n_pos_target):
            best_thr = thr

    ids = np.where(max_cos >= best_thr)[0].tolist()
    return ids, float(best_thr), len(ids), max_cos


def compute_eda(sae):
    """EDA = 1 - cos(enc_j, dec_j) per feature."""
    with torch.no_grad():
        we = sae.W_enc.detach().float().T  # (d_sae, d_in)
        wd = sae.W_dec.detach().float()    # (d_sae, d_in)
        cos_ed = F.cosine_similarity(we, wd, dim=1).cpu().numpy()
    return 1.0 - cos_ed


def measure_empirical_l0(sae, model, layer, device):
    """Measure L0 from a short text passage."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    text = ("The quick brown fox jumps over the lazy dog. "
            "Scientists discovered new species in deep ocean. "
            "Technology companies announced partnerships. " * 3)
    try:
        tokens = model.to_tokens(text)[:, :128].to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            resid = cache[hook_name][0]  # (seq, d_model)
            out = sae.encode(resid)
            acts = out[1] if isinstance(out, tuple) and len(out) > 1 else out
            l0 = float((acts > 0).float().sum(-1).mean().item())
        del cache
        return l0
    except Exception as e:
        print(f"    L0 measurement error: {e}")
        return None


# ─── Step 3: SAE configs ─────────────────────────────────────────────────────
report_progress(3, TOTAL_STEPS, "Defining SAE configurations")

SAE_CONFIGS = [
    # Primary: gpt2-small-res-jb at multiple layers (fixed width 24576, varying L0)
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.2.hook_resid_pre",
     "layer": 2, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.4.hook_resid_pre",
     "layer": 4, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.8.hook_resid_pre",
     "layer": 8, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.10.hook_resid_pre",
     "layer": 10, "group": "primary"},
    # AJT variants at L6 (different L0 due to different training)
    {"release": "gpt2-small-res_sce-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "group": "ajt"},
    {"release": "gpt2-small-res_scl-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "group": "ajt"},
    {"release": "gpt2-small-res_sle-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "group": "ajt"},
    # Feature-splitting: varying width at L8 (within-width L0 analysis)
    {"release": "gpt2-small-res-jb-feature-splitting",
     "sae_id": "blocks.8.hook_resid_pre_12288", "layer": 8, "group": "width"},
    {"release": "gpt2-small-res-jb-feature-splitting",
     "sae_id": "blocks.8.hook_resid_pre_49152", "layer": 8, "group": "width"},
    {"release": "gpt2-small-res-jb-feature-splitting",
     "sae_id": "blocks.8.hook_resid_pre_98304", "layer": 8, "group": "width"},
]

# ─── Step 4: Process SAEs ────────────────────────────────────────────────────
report_progress(4, TOTAL_STEPS, "Processing each SAE: identify letter features, compute EDA")

results_data = []

for cfg in SAE_CONFIGS:
    release = cfg["release"]
    sae_id = cfg["sae_id"]
    layer = cfg["layer"]
    group = cfg["group"]
    print(f"\n  [{group}] {release} | {sae_id}")

    try:
        sae_obj = SAE.from_pretrained(release, sae_id)
        sae_obj.eval().to(DEVICE)
        d_sae = sae_obj.cfg.d_sae
        width = d_sae

        probe_data = layer_probes.get(layer, {"dirs": {}, "letters": []})
        probe_dirs = probe_data["dirs"]

        if len(probe_dirs) < 5:
            print(f"    SKIP: only {len(probe_dirs)} probes at layer {layer}")
            sae_obj.cpu(); del sae_obj; torch.cuda.empty_cache()
            results_data.append({"release": release, "sae_id": sae_id, "layer": layer,
                                  "width": width, "group": group, "status": "skipped",
                                  "reason": "insufficient probes"})
            continue

        # Identify letter features
        n_target = min(67, d_sae // 200)
        letter_ids, thr, n_pos, max_cos_arr = identify_letter_features(
            sae_obj.W_dec.data, probe_dirs, n_pos_target=n_target
        )
        print(f"    d_sae={d_sae}, n_letter_features={n_pos} at threshold={thr:.2f}")

        if n_pos < 10:
            print(f"    SKIP: n_pos={n_pos} < 10")
            sae_obj.cpu(); del sae_obj; torch.cuda.empty_cache()
            results_data.append({"release": release, "sae_id": sae_id, "layer": layer,
                                  "width": width, "group": group, "status": "skipped",
                                  "reason": f"n_pos={n_pos} too small"})
            continue

        # Compute EDA
        eda_scores = compute_eda(sae_obj)
        letter_eda = eda_scores[letter_ids]
        non_letter_ids = [i for i in range(d_sae) if i not in set(letter_ids)]
        non_letter_eda = eda_scores[non_letter_ids]

        # Key metrics
        mean_eda_letter = float(letter_eda.mean())
        mean_eda_nonletter = float(non_letter_eda.mean())
        eda_delta = mean_eda_letter - mean_eda_nonletter  # >0 means letter features have higher EDA

        # Fraction above various thresholds
        frac_eda_gt50 = float((letter_eda > 0.50).mean())
        frac_eda_gt60 = float((letter_eda > 0.60).mean())
        frac_eda_gt70 = float((letter_eda > 0.70).mean())

        # AUROC (EDA as absorption detector vs all features)
        binary_labels = np.zeros(d_sae)
        binary_labels[letter_ids] = 1
        try:
            eda_auroc = float(roc_auc_score(binary_labels, eda_scores))
        except:
            eda_auroc = None

        # Empirical L0
        l0 = measure_empirical_l0(sae_obj, model, layer, DEVICE)
        l0_str = f"{l0:.2f}" if l0 is not None else "N/A"
        eda_auroc_str = f"{eda_auroc:.4f}" if eda_auroc is not None else "N/A"
        print(f"    L0={l0_str}, mean_EDA(letter)={mean_eda_letter:.4f}, "
              f"EDA_delta={eda_delta:.4f}, EDA_AUROC={eda_auroc_str}")

        results_data.append({
            "release": release,
            "sae_id": sae_id,
            "layer": layer,
            "width": width,
            "group": group,
            "status": "success",
            "d_sae": d_sae,
            "n_letter_features": n_pos,
            "probe_threshold": thr,
            "l0": l0,
            "inv_l0": float(1.0 / l0) if l0 and l0 > 0 else None,
            "mean_eda_letter": mean_eda_letter,
            "mean_eda_nonletter": mean_eda_nonletter,
            "eda_delta": float(eda_delta),
            "frac_eda_gt50": frac_eda_gt50,
            "frac_eda_gt60": frac_eda_gt60,
            "frac_eda_gt70": frac_eda_gt70,
            "eda_auroc": eda_auroc,
            "letter_eda_percentiles": {
                "p25": float(np.percentile(letter_eda, 25)),
                "p50": float(np.percentile(letter_eda, 50)),
                "p75": float(np.percentile(letter_eda, 75)),
                "p90": float(np.percentile(letter_eda, 90))
            },
        })

        sae_obj.cpu(); del sae_obj; torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        traceback.print_exc()
        results_data.append({"release": release, "sae_id": sae_id, "layer": layer,
                              "group": group, "status": "error", "error": str(e)})

# ─── Step 5: Filter and report valid points ──────────────────────────────────
report_progress(5, TOTAL_STEPS, "Filtering valid points")

valid_pts = [d for d in results_data
             if d.get("status") == "success" and d.get("l0") is not None]

print(f"\nValid points: {len(valid_pts)}")
for p in sorted(valid_pts, key=lambda x: x["l0"]):
    print(f"  [{p['group']}] layer={p['layer']}, width={p['width']}, "
          f"L0={p['l0']:.2f}, 1/L0={p['inv_l0']:.4f}, "
          f"mean_EDA={p['mean_eda_letter']:.4f}, EDA_delta={p['eda_delta']:.4f}")

pilot_pass = len(valid_pts) >= 3
print(f"\nPilot pass (>=3 valid): {'PASS' if pilot_pass else 'FAIL'}")

# ─── Step 6: Correlation analysis ────────────────────────────────────────────
report_progress(6, TOTAL_STEPS, "Spearman correlation: 1/L0 vs EDA metrics")

spearman_results = {}
if len(valid_pts) >= 3:
    from scipy.stats import spearmanr, pearsonr

    inv_l0s = np.array([p["inv_l0"] for p in valid_pts])
    mean_edas = np.array([p["mean_eda_letter"] for p in valid_pts])
    eda_deltas = np.array([p["eda_delta"] for p in valid_pts])
    eda_aurocs = np.array([p["eda_auroc"] for p in valid_pts if p["eda_auroc"] is not None])

    rho_eda, p_eda = spearmanr(inv_l0s, mean_edas)
    rho_delta, p_delta = spearmanr(inv_l0s, eda_deltas)

    spearman_results = {
        "inv_l0_vs_mean_eda": {"rho": float(rho_eda), "p": float(p_eda), "n": len(valid_pts)},
        "inv_l0_vs_eda_delta": {"rho": float(rho_delta), "p": float(p_delta), "n": len(valid_pts)},
    }
    print(f"Spearman(1/L0, mean_EDA): rho={rho_eda:.4f}, p={p_eda:.4f}")
    print(f"Spearman(1/L0, EDA_delta): rho={rho_delta:.4f}, p={p_delta:.4f}")

    # Primary-only (same width, gpt2-small-res-jb)
    primary_only = [p for p in valid_pts if p["group"] == "primary"]
    if len(primary_only) >= 3:
        inv_l0_p = np.array([p["inv_l0"] for p in primary_only])
        eda_p = np.array([p["mean_eda_letter"] for p in primary_only])
        rho_p, p_p = spearmanr(inv_l0_p, eda_p)
        spearman_results["primary_only"] = {
            "inv_l0_vs_mean_eda": {"rho": float(rho_p), "p": float(p_p), "n": len(primary_only)}
        }
        print(f"Primary only: Spearman(1/L0, mean_EDA): rho={rho_p:.4f}, p={p_p:.4f}")

# ─── Step 7: Curve fitting ────────────────────────────────────────────────────
report_progress(7, TOTAL_STEPS, "Fitting curves")

curve_fits = {}

if len(valid_pts) >= 3:
    from scipy.stats import linregress
    from scipy.optimize import curve_fit as scipy_curve_fit

    # Use mean_eda_letter as target (continuous, not saturated)
    inv_l0_all = np.array([p["inv_l0"] for p in sorted(valid_pts, key=lambda x: x["l0"])])
    eda_all = np.array([p["mean_eda_letter"] for p in sorted(valid_pts, key=lambda x: x["l0"])])

    n = len(inv_l0_all)
    ss_tot = np.sum((eda_all - eda_all.mean())**2)

    # Linear fit
    try:
        slope, intercept, r_val, p_val, _ = linregress(inv_l0_all, eda_all)
        y_pred = slope * inv_l0_all + intercept
        ss_res = np.sum((eda_all - y_pred)**2)
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        sigma2 = max(ss_res / n, 1e-12)
        ll = -n/2 * np.log(2*np.pi*sigma2) - ss_res/(2*sigma2)
        bic = 2 * np.log(n) - 2 * ll
        aic = 4 - 2 * ll
        curve_fits["linear"] = {
            "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2), "r_value": float(r_val), "p_value": float(p_val),
            "bic": float(bic), "aic": float(aic), "ll": float(ll)
        }
        print(f"  Linear: slope={slope:.5f}, R2={r2:.4f}, p={p_val:.4f}, BIC={bic:.2f}")
    except Exception as e:
        curve_fits["linear"] = {"error": str(e)}

    # Sigmoid (need >=4 points)
    if n >= 4:
        try:
            def sigmoid(x, L, k, x0, b):
                return b + L / (1 + np.exp(-k * (x - x0)))
            L_init = float(eda_all.max() - eda_all.min())
            k_init = 100.0
            x0_init = float(np.median(inv_l0_all))
            b_init = float(eda_all.min())
            popt, pcov = scipy_curve_fit(
                sigmoid, inv_l0_all, eda_all,
                p0=[L_init, k_init, x0_init, b_init],
                maxfev=10000,
                bounds=([0, 0, 0, 0], [1, 10000, 1, 1])
            )
            y_pred_s = sigmoid(inv_l0_all, *popt)
            ss_res_s = np.sum((eda_all - y_pred_s)**2)
            r2_s = 1 - ss_res_s / (ss_tot + 1e-12)
            sigma2_s = max(ss_res_s / n, 1e-12)
            ll_s = -n/2 * np.log(2*np.pi*sigma2_s) - ss_res_s/(2*sigma2_s)
            bic_s = 4 * np.log(n) - 2 * ll_s
            aic_s = 8 - 2 * ll_s
            ll_lin = curve_fits.get("linear", {}).get("ll", None)
            lrt_stat = float(2*(ll_s - ll_lin)) if ll_lin else None
            lrt_p = float(1 - stats.chi2.cdf(lrt_stat, df=2)) if (lrt_stat and lrt_stat > 0) else None
            inflection_l0c = float(1.0 / popt[2]) if popt[2] > 0 else None
            curve_fits["sigmoid"] = {
                "L": float(popt[0]), "k": float(popt[1]),
                "x0": float(popt[2]), "b": float(popt[3]),
                "r2": float(r2_s), "bic": float(bic_s), "aic": float(aic_s), "ll": float(ll_s),
                "inflection_inv_l0": float(popt[2]),
                "inflection_l0_c": inflection_l0c,
                "lrt_stat": lrt_stat, "lrt_pvalue": lrt_p
            }
            print(f"  Sigmoid: L={popt[0]:.4f}, k={popt[1]:.1f}, x0={popt[2]:.4f}, "
                  f"R2={r2_s:.4f}, BIC={bic_s:.2f}")
            if lrt_p is not None:
                print(f"  LRT sigmoid vs linear: stat={lrt_stat:.3f}, p={lrt_p:.4f}")
        except Exception as e:
            curve_fits["sigmoid"] = {"error": str(e)}
            print(f"  Sigmoid error: {e}")

# ─── Step 8: Bootstrap CI ────────────────────────────────────────────────────
report_progress(8, TOTAL_STEPS, "Bootstrap CI (1000 resamples)")

bootstrap_ci = {}
if (len(valid_pts) >= 4 and "sigmoid" in curve_fits
        and "error" not in curve_fits.get("sigmoid", {})):
    from scipy.optimize import curve_fit as scipy_curve_fit

    def sigmoid(x, L, k, x0, b):
        return b + L / (1 + np.exp(-k * (x - x0)))

    inv_l0_arr = np.array([p["inv_l0"] for p in valid_pts])
    eda_arr = np.array([p["mean_eda_letter"] for p in valid_pts])
    rng_b = np.random.RandomState(SEED)
    boot_x0s = []

    for _ in range(1000):
        idx = rng_b.choice(len(valid_pts), len(valid_pts), replace=True)
        xb, yb = inv_l0_arr[idx], eda_arr[idx]
        if len(np.unique(xb)) < 3:
            continue
        try:
            popt, _ = scipy_curve_fit(
                sigmoid, xb, yb,
                p0=[0.2, 100.0, np.median(xb), yb.min()],
                maxfev=5000,
                bounds=([0, 0, 0, 0], [1, 10000, 1, 1])
            )
            if popt[2] > 0:
                boot_x0s.append(float(1.0 / popt[2]))
        except:
            pass

    if len(boot_x0s) >= 50:
        bootstrap_ci = {
            "n_successful": len(boot_x0s),
            "l0c_mean": float(np.mean(boot_x0s)),
            "l0c_ci_low": float(np.percentile(boot_x0s, 2.5)),
            "l0c_ci_high": float(np.percentile(boot_x0s, 97.5)),
            "nominal_l0c": curve_fits.get("sigmoid", {}).get("inflection_l0_c")
        }
        print(f"Bootstrap CI L0_c: {bootstrap_ci['l0c_mean']:.1f} "
              f"[{bootstrap_ci['l0c_ci_low']:.1f}, {bootstrap_ci['l0c_ci_high']:.1f}]")
    else:
        bootstrap_ci = {"error": f"Only {len(boot_x0s)} successful bootstrap fits"}
        print(f"Bootstrap: only {len(boot_x0s)} successful fits")
else:
    reason = ("need >=4 valid points" if len(valid_pts) < 4
              else "sigmoid fit unavailable")
    bootstrap_ci = {"skipped": reason}
    print(f"Bootstrap skipped: {reason}")

# ─── Step 9: H1 prediction check ─────────────────────────────────────────────
report_progress(9, TOTAL_STEPS, "H1 quantitative prediction check")

h1_check = {}
b1_path = FULL_RESULTS_DIR / "B1_decoder_geometry.json"
if b1_path.exists():
    with open(b1_path) as f:
        b1 = json.load(f)
    l6 = b1.get("layer6", {})
    pa = l6.get("pair_level_analysis", {})
    pos_cos2 = pa.get("max_cos2", {}).get("pos_mean")
    if pos_cos2:
        lambda_star = 1.0 - pos_cos2
        l0c_pred = 1.0 / lambda_star if lambda_star > 0 else None
        l0c_obs = curve_fits.get("sigmoid", {}).get("inflection_l0_c")
        ratio = (l0c_obs / l0c_pred) if (l0c_obs and l0c_pred and l0c_pred > 0) else None
        h1_check = {
            "pos_cos2_mean_l6": float(pos_cos2),
            "lambda_star": float(lambda_star),
            "predicted_l0c": float(l0c_pred) if l0c_pred else None,
            "observed_l0c": float(l0c_obs) if l0c_obs else None,
            "ratio": float(ratio) if ratio else None,
        }
        l0c_pred_str = f"{l0c_pred:.1f}" if l0c_pred else "N/A"
        l0c_obs_str = f"{l0c_obs:.1f}" if l0c_obs else "N/A"
        ratio_str = f"{ratio:.2f}" if ratio else "N/A"
        print(f"H1: lambda*={lambda_star:.4f}, pred_L0c={l0c_pred_str}, "
              f"obs_L0c={l0c_obs_str}, ratio={ratio_str}")

# ─── Step 10: Compile and save ────────────────────────────────────────────────
report_progress(10, TOTAL_STEPS, "Compiling and saving results")

elapsed = time.time() - start_time

# Primary finding
if spearman_results:
    rho = spearman_results.get("inv_l0_vs_mean_eda", {}).get("rho", 0)
    p_val = spearman_results.get("inv_l0_vs_mean_eda", {}).get("p", 1)
    primary_finding = (f"Spearman(1/L0, mean_EDA)=rho={rho:.3f}, p={p_val:.3f}, "
                       f"n={len(valid_pts)} configs. "
                       f"{'Positive' if rho > 0.3 else 'Weak/No'} correlation.")
else:
    primary_finding = f"{len(valid_pts)} configs measured, insufficient for correlation"

# Width analysis data
width_pts = [p for p in valid_pts if p["group"] in ("width", "primary") and p["layer"] == 8]
width_analysis = {
    "n_points": len(width_pts),
    "points": [{"width": p["width"], "l0": p["l0"], "inv_l0": p["inv_l0"],
                 "mean_eda": p["mean_eda_letter"], "eda_delta": p["eda_delta"]}
                for p in sorted(width_pts, key=lambda x: x["width"])]
}

final_results = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "seed": SEED,
        "n_configs_tested": len(SAE_CONFIGS),
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    },
    "pilot_pass": pilot_pass,
    "pass_criteria": ">=3 different L0 settings measured",
    "n_valid_points": len(valid_pts),
    "all_results": results_data,
    "valid_data_points": valid_pts,
    "spearman_results": spearman_results,
    "curve_fits": curve_fits,
    "bootstrap_ci": bootstrap_ci,
    "width_analysis": width_analysis,
    "h1_prediction_check": h1_check,
    "primary_finding": primary_finding,
    "summary": {
        "n_configs_tested": len(SAE_CONFIGS),
        "n_valid": len(valid_pts),
        "l0_range": ([float(min(p["l0"] for p in valid_pts)),
                      float(max(p["l0"] for p in valid_pts))] if valid_pts else None),
        "mean_eda_range": ([float(min(p["mean_eda_letter"] for p in valid_pts)),
                            float(max(p["mean_eda_letter"] for p in valid_pts))]
                           if valid_pts else None),
        "pilot_pass": pilot_pass
    }
}

OUTPUT_FILE_PILOT.write_text(json.dumps(final_results, indent=2))
print(f"Saved: {OUTPUT_FILE_PILOT}")

if pilot_pass:
    OUTPUT_FILE_FULL.write_text(json.dumps(final_results, indent=2))
    print(f"Saved: {OUTPUT_FILE_FULL}")

mark_done(
    status="success" if pilot_pass else "partial",
    summary=(f"B2 complete: {len(valid_pts)} configs, "
             f"L0={final_results['summary']['l0_range']}, "
             f"EDA range={final_results['summary']['mean_eda_range']}, "
             + primary_finding[:100]),
    result=final_results["summary"]
)

# Update gpu_progress.json
gp_path = WORKSPACE / "exp" / "gpu_progress.json"
if gp_path.exists():
    with open(gp_path) as f:
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
    "planned_min": 40,
    "actual_min": max(1, int(elapsed / 60)),
    "start_time": datetime.fromtimestamp(start_time).isoformat(),
    "end_time": datetime.now().isoformat(),
    "config_snapshot": {
        "model": "gpt2-small",
        "n_sae_configs": len(SAE_CONFIGS),
        "n_valid_points": len(valid_pts),
        "pilot_pass": pilot_pass,
        "primary_metric": "mean_eda_letter",
        "gpu_model": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu"
    }
}
with open(gp_path, "w") as f:
    json.dump(gp, f, indent=2)

print(f"\n{'='*60}")
print(f"B2 Scaling Curve DONE | pilot={'PASS' if pilot_pass else 'FAIL'}")
print(f"Valid configs: {len(valid_pts)}/{len(SAE_CONFIGS)}")
if valid_pts:
    print(f"L0 range: {final_results['summary']['l0_range']}")
    print(f"Mean EDA range: {final_results['summary']['mean_eda_range']}")
print(f"Primary finding: {primary_finding}")
print(f"Time: {elapsed:.1f}s ({elapsed/60:.1f}min)")
print(f"{'='*60}")
