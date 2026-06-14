"""
Task E1 (PILOT): Phase Transition Detection in Sparsity

Measures absorption rate vs 1/L0 across multiple SAE configs and fits:
  1) linear:     y = a*x + b
  2) power-law:  y = a * x^b
  3) sigmoid:    y = L / (1 + exp(-k*(x - x0))) + b

Runs LRT: sigmoid vs linear (H4a).
Identifies inflection point L0_c with 95% bootstrap CI.
Also checks if critical L0 shifts with SAE width.

Reuses B2 data where available (EDA proxy for absorption).
Also computes direct absorption_rate via first-letter task for L6 configs.

Mode: PILOT (seed 42, timeout=900s)
"""

import os
import sys
import json
import time
import warnings
import random
import string
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats, optimize
from scipy.special import expit

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_E1_phase_transition"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "E1_phase_transition.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


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
report_progress(0, TOTAL_STEPS, "Starting E1 Phase Transition Detection (PILOT)")

# ─── Step 1: Load B2 data ────────────────────────────────────────────────────
report_progress(1, TOTAL_STEPS, "Loading B2 scaling curve data")

B2_FILE = WORKSPACE / "exp" / "results" / "full" / "B2_scaling_curve.json"
b2_data = None
if B2_FILE.exists():
    with open(B2_FILE) as f:
        b2_data = json.load(f)
    print(f"  Loaded B2 data: {b2_data.get('n_valid_points', 0)} configs")
else:
    print("  WARNING: B2 data not found, will measure from scratch")


# ─── Step 2: Load model and train probes ─────────────────────────────────────
report_progress(2, TOTAL_STEPS, "Loading GPT-2 Small and training probes")

from transformer_lens import HookedTransformer
from sae_lens import SAE
import sklearn.linear_model as sklm

model = HookedTransformer.from_pretrained(
    "gpt2", center_writing_weights=True, center_unembed=True,
    fold_ln=True, refactor_factored_attn_matrices=True
)
model.eval().to(DEVICE)
tokenizer = model.tokenizer
print(f"d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

# Build word vocab for first-letter task
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


# Pre-compute probes for layers 2, 4, 6, 8, 10
print("Training probes for layers 2, 4, 6, 8, 10...")
layer_probes = {}
for layer in [2, 4, 6, 8, 10]:
    dirs, letters = train_probes_at_layer(model, probe_train_words, layer, DEVICE)
    layer_probes[layer] = {"dirs": dirs, "letters": letters}
    print(f"  Layer {layer}: {len(letters)} letters with probes")


def identify_letter_features(sae_w_dec, probe_dirs, n_pos_target=67):
    """Find SAE features whose decoder aligns with letter probes."""
    letters = sorted(probe_dirs.keys())
    probe_mat = np.stack([probe_dirs[lt] for lt in letters])
    W_dec_np = F.normalize(sae_w_dec.cpu().float(), dim=1).numpy()
    cos = probe_mat @ W_dec_np.T
    max_cos = cos.max(axis=0)
    best_thr = 0.30
    for thr in np.arange(0.20, 0.55, 0.01):
        n = int((max_cos >= thr).sum())
        if abs(n - n_pos_target) < abs(int((max_cos >= best_thr).sum()) - n_pos_target):
            best_thr = thr
    ids = np.where(max_cos >= best_thr)[0].tolist()
    return ids, float(best_thr), len(ids), max_cos


def measure_empirical_l0(sae, model, layer, device):
    """Measure empirical L0 from a short text passage."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    text = ("The quick brown fox jumps over the lazy dog. "
            "Scientists discovered new species in deep ocean. "
            "Technology companies announced partnerships. " * 3)
    try:
        tokens = model.to_tokens(text)[:, :128].to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            resid = cache[hook_name][0]
            out = sae.encode(resid)
            if isinstance(out, tuple) and len(out) > 1:
                acts = out[1]
            elif isinstance(out, tuple):
                acts = out[0]
            else:
                acts = out
            l0 = float((acts > 0).float().sum(-1).mean().item())
        del cache
        return l0
    except Exception as e:
        print(f"    L0 measurement error: {e}")
        return None


def measure_absorption_rate_first_letter(sae, model, probe_dirs, letter_ids, layer, device,
                                         n_per_letter=30, seed=42):
    """
    Measure absorption rate using first-letter task.

    For each letter L with a probe direction d_L:
      - Collect tokens that start with L (child tokens)
      - Run model, get SAE activations
      - For each letter feature j (in letter_ids for letter L):
          absorption = fraction of times j does NOT activate (act_j < threshold)
          even though d_L · resid > letter_probe_threshold (parent should be active)

    absorption_rate = mean over all (letter, feature) pairs of absorption fraction
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    rng = random.Random(seed)

    # Build letter-to-words mapping from vocab
    letters = sorted(probe_dirs.keys())
    absorbed_count = 0
    total_count = 0
    per_letter_rates = {}

    # Identify which letter feature belongs to which letter
    if not letter_ids:
        return 0.0, {}

    # For each letter, find which feature IDs correspond to it
    sae_w_dec = sae.W_dec.detach().cpu().float()
    W_dec_np = F.normalize(sae_w_dec, dim=1).numpy()

    # Map: for each letter_id in letter_ids, find best matching letter
    probe_mat = np.stack([probe_dirs[lt] for lt in letters])  # (n_letters, d_model)
    cos_letter_features = probe_mat @ W_dec_np[letter_ids].T  # (n_letters, n_letter_ids)
    # Each column: which letter does this feature best match?
    best_letter_idx = cos_letter_features.argmax(axis=0)  # (n_letter_ids,)

    feature_to_letter = {}
    letter_to_features = {lt: [] for lt in letters}
    for feat_pos, feat_id in enumerate(letter_ids):
        lt = letters[best_letter_idx[feat_pos]]
        feature_to_letter[feat_id] = lt
        letter_to_features[lt].append(feat_id)

    # For each letter, collect word samples and measure absorption
    for lt in letters:
        feat_ids_for_lt = letter_to_features[lt]
        if not feat_ids_for_lt:
            continue
        # Get words starting with this letter
        words_for_lt = good_letters.get(lt, [])
        if len(words_for_lt) < 3:
            continue

        sampled_words = rng.sample(words_for_lt, min(n_per_letter, len(words_for_lt)))

        letter_absorbed = 0
        letter_total = 0

        for word in sampled_words:
            try:
                # Context that should activate the letter concept
                tok = model.to_tokens(f" {word}:")
                tok = tok.to(device)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tok, names_filter=hook_name)
                    resid = cache[hook_name][0, -2, :]  # (d_model,)

                    # Check probe activation: is the letter concept present?
                    probe_dir = torch.tensor(probe_dirs[lt], dtype=torch.float32).to(device)
                    probe_act = torch.dot(resid, probe_dir).item()

                    if probe_act < 0:  # Probe not firing, skip
                        del cache
                        continue

                    # Encode through SAE
                    sae_input = resid.unsqueeze(0)
                    out = sae.encode(sae_input)
                    if isinstance(out, tuple) and len(out) > 1:
                        sae_acts = out[1][0]  # (d_sae,) - hidden acts
                    elif isinstance(out, tuple):
                        sae_acts = out[0][0]  # (d_sae,)
                    else:
                        sae_acts = out[0]  # (d_sae,) - direct tensor output

                    # For each letter feature: is it absorbed (not firing)?
                    for feat_id in feat_ids_for_lt:
                        feat_act = sae_acts[feat_id].item()
                        is_absorbed = int(feat_act < 0.1)  # Near-zero activation
                        letter_absorbed += is_absorbed
                        letter_total += 1

                    del cache
            except Exception as e:
                continue

        if letter_total > 0:
            rate = letter_absorbed / letter_total
            per_letter_rates[lt] = {
                "n_features": len(feat_ids_for_lt),
                "n_samples": letter_total,
                "absorption_rate": rate
            }
            absorbed_count += letter_absorbed
            total_count += letter_total

    overall_rate = absorbed_count / total_count if total_count > 0 else 0.0
    return overall_rate, per_letter_rates


# ─── Step 3: Define SAE configs to measure ──────────────────────────────────
report_progress(3, TOTAL_STEPS, "Defining SAE configurations for phase transition analysis")

SAE_CONFIGS = [
    # Primary: gpt2-small-res-jb at multiple layers (fixed width 24576, varying L0)
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.2.hook_resid_pre",
     "layer": 2, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.4.hook_resid_pre",
     "layer": 4, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.8.hook_resid_pre",
     "layer": 8, "width": 24576, "group": "primary"},
    {"release": "gpt2-small-res-jb", "sae_id": "blocks.10.hook_resid_pre",
     "layer": 10, "width": 24576, "group": "primary"},
    # AJT variants at L6 (same layer, different L0 via different training)
    {"release": "gpt2-small-res_sce-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": 24576, "group": "ajt"},
    {"release": "gpt2-small-res_scl-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": 24576, "group": "ajt"},
    {"release": "gpt2-small-res_sle-ajt", "sae_id": "blocks.6.hook_resid_pre",
     "layer": 6, "width": 24576, "group": "ajt"},
    # Feature-splitting: varying width at L8
    {"release": "gpt2-small-res-jb-feature-splitting",
     "sae_id": "blocks.8.hook_resid_pre_12288", "layer": 8, "width": 12288, "group": "width"},
    {"release": "gpt2-small-res-jb-feature-splitting",
     "sae_id": "blocks.8.hook_resid_pre_49152", "layer": 8, "width": 49152, "group": "width"},
    {"release": "gpt2-small-res-jb-feature-splitting",
     "sae_id": "blocks.8.hook_resid_pre_98304", "layer": 8, "width": 98304, "group": "width"},
]


# ─── Step 4: Measure absorption rates ───────────────────────────────────────
report_progress(4, TOTAL_STEPS, "Measuring absorption rates for each SAE config")

results_data = []

for i, cfg in enumerate(SAE_CONFIGS):
    release = cfg["release"]
    sae_id = cfg["sae_id"]
    layer = cfg["layer"]
    width = cfg.get("width", 24576)
    group = cfg["group"]
    print(f"\n  [{i+1}/{len(SAE_CONFIGS)}] [{group}] {release} | {sae_id}")

    try:
        sae_obj = SAE.from_pretrained(release, sae_id)
        if isinstance(sae_obj, tuple):
            sae_obj = sae_obj[0]
        sae_obj.eval().to(DEVICE)
        d_sae = sae_obj.cfg.d_sae
        actual_width = d_sae

        probe_data = layer_probes.get(layer, {"dirs": {}, "letters": []})
        probe_dirs = probe_data["dirs"]

        if len(probe_dirs) < 5:
            print(f"    SKIP: only {len(probe_dirs)} probes at layer {layer}")
            sae_obj.cpu(); del sae_obj; torch.cuda.empty_cache()
            results_data.append({"release": release, "sae_id": sae_id, "layer": layer,
                                  "width": actual_width, "group": group, "status": "skipped",
                                  "reason": "insufficient probes"})
            continue

        # Identify letter features
        n_target = min(67, max(10, d_sae // 200))
        letter_ids, thr, n_pos, max_cos_arr = identify_letter_features(
            sae_obj.W_dec.data, probe_dirs, n_pos_target=n_target
        )
        print(f"    d_sae={d_sae}, n_letter_features={n_pos} at threshold={thr:.2f}")

        if n_pos < 5:
            print(f"    SKIP: n_pos={n_pos} < 5")
            sae_obj.cpu(); del sae_obj; torch.cuda.empty_cache()
            results_data.append({"release": release, "sae_id": sae_id, "layer": layer,
                                  "width": actual_width, "group": group, "status": "skipped",
                                  "reason": f"n_pos={n_pos} too small"})
            continue

        # Measure empirical L0
        l0 = measure_empirical_l0(sae_obj, model, layer, DEVICE)
        print(f"    L0={l0:.2f}" if l0 else "    L0=N/A")

        # Measure absorption rate via first-letter task
        absorption_rate, per_letter = measure_absorption_rate_first_letter(
            sae_obj, model, probe_dirs, letter_ids, layer, DEVICE,
            n_per_letter=20,  # 20 samples per letter (pilot budget)
            seed=SEED
        )
        n_letters_measured = len(per_letter)
        print(f"    absorption_rate={absorption_rate:.4f}, n_letters={n_letters_measured}")

        # Also compute EDA proxy (from B2 methodology for cross-validation)
        with torch.no_grad():
            we = sae_obj.W_enc.detach().float().T  # (d_sae, d_in)
            wd = sae_obj.W_dec.detach().float()    # (d_sae, d_in)
            cos_ed = F.cosine_similarity(we, wd, dim=1).cpu().numpy()
        eda_scores = 1.0 - cos_ed
        if letter_ids:
            letter_eda = eda_scores[np.array(letter_ids)]
            mean_eda_letter = float(letter_eda.mean())
            eda_delta = mean_eda_letter - float(eda_scores.mean())
        else:
            mean_eda_letter = float(eda_scores.mean())
            eda_delta = 0.0

        results_data.append({
            "release": release,
            "sae_id": sae_id,
            "layer": layer,
            "width": actual_width,
            "group": group,
            "status": "success",
            "d_sae": d_sae,
            "n_letter_features": n_pos,
            "probe_threshold": thr,
            "l0": l0,
            "inv_l0": float(1.0 / l0) if l0 and l0 > 0 else None,
            "absorption_rate": float(absorption_rate),
            "per_letter_rates": per_letter,
            "n_letters_measured": n_letters_measured,
            "mean_eda_letter": float(mean_eda_letter),
            "eda_delta": float(eda_delta),
        })

        sae_obj.cpu(); del sae_obj; torch.cuda.empty_cache()

    except Exception as e:
        import traceback
        print(f"    ERROR: {e}")
        traceback.print_exc()
        results_data.append({"release": release, "sae_id": sae_id, "layer": layer,
                              "width": width, "group": group, "status": "error",
                              "error": str(e)})

# Filter successful results with valid L0 and absorption_rate
valid_results = [r for r in results_data
                 if r.get("status") == "success"
                 and r.get("l0") is not None and r.get("l0", 0) > 0
                 and r.get("absorption_rate") is not None]

print(f"\n  Valid results: {len(valid_results)} configs")
n_valid = len(valid_results)

# ─── Step 5: Curve fitting ───────────────────────────────────────────────────
report_progress(5, TOTAL_STEPS, "Fitting curve models (linear, power-law, sigmoid)")

# Extract X (inv_L0) and Y (absorption_rate)
inv_l0_all = np.array([r["inv_l0"] for r in valid_results])
absorption_all = np.array([r["absorption_rate"] for r in valid_results])
l0_all = np.array([r["l0"] for r in valid_results])
width_all = np.array([r["width"] for r in valid_results])
layer_all = np.array([r["layer"] for r in valid_results])

print(f"\n  Data points: n={n_valid}")
for r in valid_results:
    print(f"    layer={r['layer']}, w={r['width']}, L0={r['l0']:.1f}, "
          f"inv_L0={r['inv_l0']:.4f}, absorption={r['absorption_rate']:.4f}")

curve_fits = {}
lrt_result = {}

if n_valid >= 3:
    # --- Linear fit ---
    try:
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, se = linregress(inv_l0_all, absorption_all)
        y_pred_lin = slope * inv_l0_all + intercept
        ss_res = np.sum((absorption_all - y_pred_lin) ** 2)
        ss_tot = np.sum((absorption_all - absorption_all.mean()) ** 2)
        r2_lin = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(inv_l0_all)
        k_lin = 2  # slope + intercept
        ll_lin = -0.5 * n * np.log(ss_res / n + 1e-12) if ss_res > 0 else 0
        bic_lin = k_lin * np.log(n) - 2 * ll_lin
        aic_lin = 2 * k_lin - 2 * ll_lin
        curve_fits["linear"] = {
            "slope": float(slope), "intercept": float(intercept),
            "r2": float(r2_lin), "p_value": float(p_value),
            "bic": float(bic_lin), "aic": float(aic_lin), "ll": float(ll_lin),
            "y_pred": y_pred_lin.tolist()
        }
        print(f"  Linear: R2={r2_lin:.4f}, p={p_value:.4f}, BIC={bic_lin:.3f}")
    except Exception as e:
        print(f"  Linear fit error: {e}")
        curve_fits["linear"] = {"error": str(e)}

    # --- Power-law fit ---
    try:
        def powerlaw_func(x, a, b):
            return a * np.power(np.maximum(x, 1e-10), b)

        popt, _ = optimize.curve_fit(powerlaw_func, inv_l0_all, absorption_all,
                                     p0=[0.5, 0.5], maxfev=5000,
                                     bounds=([0, -5], [10, 10]))
        y_pred_pw = powerlaw_func(inv_l0_all, *popt)
        ss_res_pw = np.sum((absorption_all - y_pred_pw) ** 2)
        r2_pw = 1 - ss_res_pw / max(ss_tot, 1e-12)
        n = len(inv_l0_all)
        k_pw = 2
        ll_pw = -0.5 * n * np.log(ss_res_pw / n + 1e-12) if ss_res_pw > 0 else 0
        bic_pw = k_pw * np.log(n) - 2 * ll_pw
        aic_pw = 2 * k_pw - 2 * ll_pw
        curve_fits["power_law"] = {
            "a": float(popt[0]), "b": float(popt[1]),
            "r2": float(r2_pw), "bic": float(bic_pw), "aic": float(aic_pw),
            "ll": float(ll_pw), "y_pred": y_pred_pw.tolist()
        }
        print(f"  Power-law (a={popt[0]:.3f}, b={popt[1]:.3f}): R2={r2_pw:.4f}, BIC={bic_pw:.3f}")
    except Exception as e:
        print(f"  Power-law fit error: {e}")
        curve_fits["power_law"] = {"error": str(e)}

    # --- Sigmoid fit ---
    try:
        def sigmoid_func(x, L, k, x0, b):
            return L / (1 + np.exp(-k * (x - x0))) + b

        # Multiple initial guesses for sigmoid
        best_sigmoid = None
        best_r2 = -np.inf
        x0_candidates = [np.median(inv_l0_all), np.mean(inv_l0_all),
                         np.percentile(inv_l0_all, 25), np.percentile(inv_l0_all, 75)]
        for x0_init in x0_candidates:
            try:
                popt_s, _ = optimize.curve_fit(
                    sigmoid_func, inv_l0_all, absorption_all,
                    p0=[absorption_all.max() - absorption_all.min(), 100.0, x0_init, absorption_all.min()],
                    maxfev=10000,
                    bounds=([-2, 0.1, 1e-6, -1], [2, 1e6, 1.0, 1])
                )
                y_pred_s = sigmoid_func(inv_l0_all, *popt_s)
                ss_res_s = np.sum((absorption_all - y_pred_s) ** 2)
                r2_s = 1 - ss_res_s / max(ss_tot, 1e-12)
                if r2_s > best_r2:
                    best_r2 = r2_s
                    best_sigmoid = (popt_s, y_pred_s, ss_res_s, r2_s)
            except:
                pass

        if best_sigmoid is not None:
            popt_s, y_pred_s, ss_res_s, r2_s = best_sigmoid
            n = len(inv_l0_all)
            k_sig = 4
            ll_sig = -0.5 * n * np.log(ss_res_s / n + 1e-12) if ss_res_s > 0 else 0
            bic_sig = k_sig * np.log(n) - 2 * ll_sig
            aic_sig = 2 * k_sig - 2 * ll_sig
            inflection_inv_l0 = float(popt_s[2])  # x0
            inflection_l0_c = float(1.0 / inflection_inv_l0) if inflection_inv_l0 > 0 else None

            # LRT: sigmoid vs linear
            ll_lin_val = curve_fits.get("linear", {}).get("ll", 0)
            lrt_stat = 2 * (ll_sig - ll_lin_val)
            # Degrees of freedom: sigmoid has 4 params, linear has 2 → df=2
            from scipy.stats import chi2
            lrt_pvalue = float(1 - chi2.cdf(max(lrt_stat, 0), df=2))

            curve_fits["sigmoid"] = {
                "L": float(popt_s[0]), "k": float(popt_s[1]),
                "x0": float(popt_s[2]), "b": float(popt_s[3]),
                "r2": float(r2_s), "bic": float(bic_sig), "aic": float(aic_sig),
                "ll": float(ll_sig),
                "inflection_inv_l0": inflection_inv_l0,
                "inflection_l0_c": inflection_l0_c,
                "y_pred": y_pred_s.tolist()
            }
            lrt_result = {
                "lrt_stat": float(lrt_stat),
                "lrt_pvalue": float(lrt_pvalue),
                "df": 2,
                "sigmoid_bic": float(bic_sig),
                "linear_bic": curve_fits.get("linear", {}).get("bic", 0),
                "bic_diff": curve_fits.get("linear", {}).get("bic", 0) - float(bic_sig),
                "sigmoid_wins": bool(float(bic_sig) < curve_fits.get("linear", {}).get("bic", 0)),
                "h4a_supported": bool(float(bic_sig) < curve_fits.get("linear", {}).get("bic", 0) - 10 and lrt_pvalue < 0.05)
            }
            print(f"  Sigmoid (L={popt_s[0]:.3f}, k={popt_s[1]:.1f}, x0={popt_s[2]:.4f}): "
                  f"R2={r2_s:.4f}, BIC={bic_sig:.3f}")
            print(f"  LRT: stat={lrt_stat:.3f}, p={lrt_pvalue:.4f}, "
                  f"BIC diff={lrt_result['bic_diff']:.2f}, H4a={lrt_result['h4a_supported']}")
        else:
            print("  Sigmoid fit failed for all initial guesses")
            curve_fits["sigmoid"] = {"error": "optimization failed"}
    except Exception as e:
        print(f"  Sigmoid fit error: {e}")
        curve_fits["sigmoid"] = {"error": str(e)}
else:
    print(f"  INSUFFICIENT DATA: only {n_valid} valid points (need >= 3 for curve fitting)")
    curve_fits = {"insufficient_data": True, "n_valid": n_valid}

# ─── Step 6: Bootstrap CI for inflection point ──────────────────────────────
report_progress(6, TOTAL_STEPS, "Bootstrap CI for inflection point L0_c")

bootstrap_ci = {}
if n_valid >= 4 and "sigmoid" in curve_fits and "error" not in curve_fits.get("sigmoid", {}):
    try:
        n_boot = 500  # Pilot: 500 resamples (vs 1000 for full)
        l0c_boots = []
        np.random.seed(SEED)
        for _ in range(n_boot):
            idx = np.random.choice(n_valid, n_valid, replace=True)
            x_b = inv_l0_all[idx]
            y_b = absorption_all[idx]
            if len(np.unique(x_b)) < 3:
                continue
            try:
                popt_b, _ = optimize.curve_fit(
                    sigmoid_func, x_b, y_b,
                    p0=[0.3, 100.0, np.median(inv_l0_all), 0.2],
                    maxfev=3000,
                    bounds=([-2, 0.1, 1e-6, -1], [2, 1e6, 1.0, 1])
                )
                x0_b = popt_b[2]
                if 1e-6 < x0_b < 1.0:
                    l0c_boots.append(1.0 / x0_b)
            except:
                pass

        if len(l0c_boots) > 50:
            l0c_boots = np.array(l0c_boots)
            l0c_boots = l0c_boots[np.isfinite(l0c_boots)]
            if len(l0c_boots) > 10:
                l0c_boots_clipped = np.clip(l0c_boots, 1, np.percentile(l0c_boots, 99))
                bootstrap_ci = {
                    "n_successful": int(len(l0c_boots)),
                    "l0c_mean": float(np.mean(l0c_boots_clipped)),
                    "l0c_median": float(np.median(l0c_boots_clipped)),
                    "l0c_ci_low": float(np.percentile(l0c_boots_clipped, 2.5)),
                    "l0c_ci_high": float(np.percentile(l0c_boots_clipped, 97.5)),
                    "nominal_l0c": curve_fits.get("sigmoid", {}).get("inflection_l0_c")
                }
                print(f"  Bootstrap CI: L0_c={bootstrap_ci['l0c_mean']:.1f} "
                      f"[{bootstrap_ci['l0c_ci_low']:.1f}, {bootstrap_ci['l0c_ci_high']:.1f}]")
            else:
                bootstrap_ci = {"error": "too few successful bootstrap samples"}
        else:
            bootstrap_ci = {"error": f"only {len(l0c_boots)} successful bootstrap samples"}
    except Exception as e:
        bootstrap_ci = {"error": str(e)}
else:
    bootstrap_ci = {"skipped": "need >= 4 data points and successful sigmoid fit"}


# ─── Step 7: Width analysis — does L0_c shift with width? ───────────────────
report_progress(7, TOTAL_STEPS, "Width analysis: critical L0 vs SAE width")

width_analysis = {}
try:
    # Group by width for configs with the same layer
    width_groups = {}
    for r in valid_results:
        w = r["width"]
        if w not in width_groups:
            width_groups[w] = []
        width_groups[w].append(r)

    # Also look at absorption rate vs width at matched L0
    width_points = []
    for w, group_rs in sorted(width_groups.items()):
        # Take mean absorption rate within each width group
        rates = [r["absorption_rate"] for r in group_rs]
        l0s = [r["l0"] for r in group_rs]
        width_points.append({
            "width": w,
            "log2_width": float(np.log2(w)),
            "mean_absorption_rate": float(np.mean(rates)),
            "mean_l0": float(np.mean(l0s)),
            "n_configs": len(group_rs)
        })

    # Spearman correlation: width vs absorption_rate
    if len(width_all) >= 3:
        rho, pval = stats.spearmanr(np.log2(width_all), absorption_all)
        width_analysis = {
            "n_widths": len(width_groups),
            "width_points": width_points,
            "spearman_log2width_vs_absorption": {
                "rho": float(rho), "p": float(pval), "n": int(len(width_all))
            }
        }
        print(f"  Width analysis: {len(width_groups)} widths, "
              f"Spearman(log2_width, absorption)=rho={rho:.3f}, p={pval:.3f}")
    else:
        width_analysis = {"n_widths": len(width_groups), "width_points": width_points}
except Exception as e:
    width_analysis = {"error": str(e)}


# ─── Step 8: Spearman correlation (1/L0 vs absorption) ──────────────────────
report_progress(8, TOTAL_STEPS, "Spearman correlation analysis")

spearman_results = {}
if n_valid >= 3:
    try:
        rho_all, p_all = stats.spearmanr(inv_l0_all, absorption_all)
        spearman_results["all"] = {
            "rho": float(rho_all), "p": float(p_all), "n": int(n_valid)
        }
        print(f"  Spearman(1/L0, absorption_rate): rho={rho_all:.3f}, p={p_all:.4f}, n={n_valid}")

        # Primary only (layer 2-10 from gpt2-small-res-jb)
        primary = [r for r in valid_results if r.get("group") == "primary"]
        if len(primary) >= 3:
            inv_l0_primary = np.array([r["inv_l0"] for r in primary])
            absorption_primary = np.array([r["absorption_rate"] for r in primary])
            rho_p, p_p = stats.spearmanr(inv_l0_primary, absorption_primary)
            spearman_results["primary_only"] = {
                "rho": float(rho_p), "p": float(p_p), "n": len(primary)
            }
            print(f"  Spearman (primary only): rho={rho_p:.3f}, p={p_p:.4f}, n={len(primary)}")
    except Exception as e:
        spearman_results = {"error": str(e)}


# ─── Step 9: Determine pilot pass/fail ──────────────────────────────────────
report_progress(9, TOTAL_STEPS, "Evaluating pilot pass criteria")

# Pass criteria: at least 4 distinct L0 values measured
distinct_l0_vals = sorted(set(r["l0"] for r in valid_results if r.get("l0")))
n_distinct_l0 = len(distinct_l0_vals)
pilot_pass = n_distinct_l0 >= 4

print(f"\n  Pilot pass criteria: >=4 distinct L0 values")
print(f"  Distinct L0 values: {n_distinct_l0} -> {'PASS' if pilot_pass else 'FAIL (insufficient data)'}")
if distinct_l0_vals:
    print(f"  L0 range: [{min(distinct_l0_vals):.1f}, {max(distinct_l0_vals):.1f}]")

# Check H4a: sigmoid BIC lower than linear BIC by >= 10
h4a_result = "untested"
if "sigmoid" in curve_fits and "error" not in curve_fits.get("sigmoid", {}):
    bic_diff = lrt_result.get("bic_diff", 0)
    h4a_supported = lrt_result.get("h4a_supported", False)
    h4a_result = f"{'SUPPORTED' if h4a_supported else 'NOT SUPPORTED'} (BIC diff={bic_diff:.2f}, LRT p={lrt_result.get('lrt_pvalue', 'N/A'):.4f})"
print(f"  H4a (phase transition): {h4a_result}")


# ─── Step 10: Save results ──────────────────────────────────────────────────
report_progress(10, TOTAL_STEPS, "Saving results")

# Compose summary
summary_str = (
    f"E1 PILOT: {n_valid}/{len(SAE_CONFIGS)} configs valid, "
    f"{n_distinct_l0} distinct L0 values. "
    f"pilot_pass={pilot_pass}. H4a: {h4a_result}. "
    f"Spearman(1/L0, absorption): rho={spearman_results.get('all', {}).get('rho', 'N/A'):.3f}."
)

output = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": time.time() - start_time,
    "config": {
        "model": "gpt2-small",
        "seed": SEED,
        "n_configs_tested": len(SAE_CONFIGS),
        "n_valid": n_valid,
        "n_distinct_l0": n_distinct_l0,
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    },
    "pilot_pass": pilot_pass,
    "pass_criteria": ">=4 distinct L0 values measured to fit a curve",
    "h4a_result": h4a_result,
    "n_valid_points": n_valid,
    "all_results": results_data,
    "valid_results_summary": [
        {
            "release": r["release"], "sae_id": r["sae_id"],
            "layer": r["layer"], "width": r["width"],
            "l0": r["l0"], "inv_l0": r["inv_l0"],
            "absorption_rate": r["absorption_rate"],
        }
        for r in valid_results
    ],
    "curve_fits": curve_fits,
    "lrt_result": lrt_result,
    "bootstrap_ci": bootstrap_ci,
    "spearman": spearman_results,
    "width_analysis": width_analysis,
    "x_values_inv_l0": inv_l0_all.tolist() if n_valid > 0 else [],
    "y_values_absorption": absorption_all.tolist() if n_valid > 0 else [],
    "primary_finding": summary_str,
    "summary": {
        "n_configs_tested": len(SAE_CONFIGS),
        "n_valid": n_valid,
        "n_distinct_l0": n_distinct_l0,
        "l0_range": [float(min(l0_all)) if n_valid > 0 else None,
                     float(max(l0_all)) if n_valid > 0 else None],
        "absorption_rate_range": [float(absorption_all.min()) if n_valid > 0 else None,
                                   float(absorption_all.max()) if n_valid > 0 else None],
        "pilot_pass": pilot_pass,
    }
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to {OUTPUT_FILE}")
print(f"Summary: {summary_str}")

# Also write gpu_progress
gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    if gpu_progress_file.exists():
        with open(gpu_progress_file) as f:
            gpu_progress = json.load(f)
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    elapsed_min = int((time.time() - start_time) / 60 + 0.5)
    if pilot_pass:
        if TASK_ID not in gpu_progress["completed"]:
            gpu_progress["completed"].append(TASK_ID)
    else:
        if TASK_ID not in gpu_progress["failed"]:
            gpu_progress["failed"].append(TASK_ID)

    gpu_progress["running"].pop(TASK_ID, None)
    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 15,
        "actual_min": elapsed_min,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "n_sae_configs": len(SAE_CONFIGS),
            "n_valid": n_valid,
            "gpu_model": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        }
    }
    gpu_progress_file.write_text(json.dumps(gpu_progress, indent=2))
    print(f"Updated gpu_progress.json (elapsed={elapsed_min}min)")
except Exception as e:
    print(f"Warning: could not update gpu_progress.json: {e}")

mark_done(
    status="success" if pilot_pass else "warning",
    summary=summary_str,
    result={
        "n_valid": n_valid,
        "n_distinct_l0": n_distinct_l0,
        "pilot_pass": pilot_pass,
        "h4a_result": h4a_result,
    }
)

print(f"\n=== E1 PILOT COMPLETE === {summary_str}")
