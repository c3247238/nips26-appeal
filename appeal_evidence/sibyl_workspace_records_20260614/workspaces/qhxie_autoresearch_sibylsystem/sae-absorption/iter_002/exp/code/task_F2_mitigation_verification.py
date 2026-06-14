"""
Task F2 (PILOT): Empirical Verification — Matryoshka vs. TopK Decoder Angles

Primary plan: Compare decoder cosine similarity between known absorbed pairs (first-letter task)
for Matryoshka SAE vs. TopK SAE on Gemma 2 2B, layer 12.

Fallback (if Gemma inaccessible): Use GPT-2 Small Standard/ReLU (L1) vs TopK SAEs from B3
as a proxy architecture comparison, and reuse B3 result data.

Theory prediction: SAE with hierarchical codebook (Matryoshka) or exact-k constraint (TopK)
should show different decoder angle distributions for absorbed pairs compared to
unconstrained architectures.

Output: exp/results/full/F2_mitigation_verification.json
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

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_F2_mitigation_verification"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "F2_mitigation_verification.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

TOTAL_STEPS = 12


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
    marker = {
        "task_id": TASK_ID, "status": status, "summary": summary,
        "timestamp": datetime.now().isoformat(), "elapsed_sec": time.time() - start_time,
    }
    if result is not None:
        marker["result"] = result
    DONE_FILE.write_text(json.dumps(marker))


report_progress(0, TOTAL_STEPS, "Starting F2 Mitigation Verification")

SIMPLE_WORDS = [
    "able", "above", "act", "add", "age", "ago", "air", "all", "also", "any",
    "area", "back", "bad", "bag", "ball", "base", "bath", "bear", "bed", "big",
    "bird", "blow", "blue", "boat", "body", "book", "born", "box", "boy", "break",
    "bring", "burn", "busy", "call", "camp", "card", "care", "cat", "city", "clean",
    "clear", "close", "coat", "cold", "come", "cook", "cool", "corn", "cut",
    "dark", "data", "date", "dead", "deal", "deep", "desk", "die", "dirt", "dish",
    "door", "down", "draw", "drop", "duck", "dust", "each", "earn", "east",
    "edge", "end", "even", "ever", "face", "fact", "fail", "fall", "far", "farm",
    "fast", "feel", "feet", "file", "fill", "film", "find", "fire", "fish", "five",
    "flag", "flat", "flow", "food", "foot", "form", "four", "free", "fuel", "full",
    "gain", "game", "gave", "girl", "give", "glad", "goal", "gold", "golf", "gone",
    "good", "gray", "grew", "grin", "grip", "grow", "hack", "hair", "half",
    "hall", "hand", "hang", "hard", "harm", "hate", "have", "head", "heal", "hear",
    "heat", "help", "here", "high", "hill", "hint", "hire", "hold", "hole", "home",
    "hook", "hope", "horn", "host", "hour", "hunt", "hurt", "idea", "inch", "iron",
    "jail", "join", "joke", "just", "keen", "keep", "kick", "kill", "kind",
    "king", "knew", "know", "lack", "lake", "land", "lane", "last", "late", "lead",
    "leaf", "lean", "leap", "left", "less", "lick", "lift", "like", "lime", "line",
    "link", "lion", "list", "load", "lock", "long", "look", "loop", "lose", "loss",
    "lost", "loud", "love", "luck", "made", "mail", "main", "make", "mark",
    "mass", "mate", "mean", "meat", "meet", "melt", "mild", "milk",
    "mill", "mine", "miss", "mode", "moon", "more", "most", "move", "much", "must",
    "name", "near", "need", "nest", "news", "next", "nice", "nine", "node", "none",
    "noon", "norm", "nose", "note", "noun", "once", "only", "open", "over",
    "pace", "pack", "page", "pain", "pair", "palm", "park", "part",
    "pass", "past", "path", "peak", "pear", "peel", "peer", "pick", "pill", "pine",
    "pink", "pipe", "plan", "play", "plot", "plug", "plus", "poem", "poll",
    "pond", "pool", "poor", "port", "pose", "post", "pour", "pull", "pump", "pure",
    "push", "race", "rack", "rage", "raid", "rain", "rank", "rate", "read", "real",
    "rely", "rent", "rest", "rice", "rich", "ride", "ring", "riot", "rise", "risk",
    "road", "rock", "role", "roll", "roof", "room", "root", "rope",
    "rose", "ruin", "rule", "rush", "rust", "safe", "sage", "sail", "salt", "same",
    "sand", "save", "seal", "seek", "self", "sell", "send", "ship", "shoe",
    "shot", "show", "sick", "side", "silk", "sing", "sink", "site", "size", "skip",
    "slim", "slip", "slow", "snow", "soak", "sock", "soft", "soil", "sole", "some",
    "song", "soon", "soul", "spin", "spot", "stem", "step", "stop", "suit", "swim",
    "tail", "take", "talk", "tall", "tank", "task", "team", "tear", "tell", "tend",
    "test", "text", "thin", "tick", "tile", "time", "tire", "told",
    "tone", "took", "tool", "toss", "tour", "town", "trap", "tree", "trim",
    "trip", "true", "tube", "tune", "turn", "type", "unit", "upon", "used",
    "vary", "vast", "very", "vest", "view", "vine", "vote", "wade", "wage", "wake",
    "walk", "wall", "warm", "warn", "wash", "wave", "wear", "weed", "week", "went",
    "west", "wide", "wild", "will", "wind", "wine", "wing", "wire", "wish",
    "wood", "word", "work", "wrap", "yard", "year", "zinc", "zone",
    "apple", "ant", "axe", "bat", "brain", "cup", "dog", "day", "dream",
    "egg", "earth", "fox", "fan", "frog", "gun", "gate", "hat", "hub",
    "inn", "jar", "joy", "key", "kid", "lab", "map", "mob", "nun", "net",
    "oil", "owl", "pen", "pin", "rag", "sum", "sun", "tan", "tip", "van",
    "web", "yam", "zip",
]


def identify_letter_features_gpt2(model, sae, hook_name, good_letters, probe_train_words,
                                    target_n_pos=67, arch_desc=""):
    """
    Identify SAE features that respond to first-letter of words.
    Uses probe-decoder cosine alignment.
    """
    import sklearn.linear_model as sklm

    print(f"  Collecting activations for {arch_desc}...")
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
            except Exception:
                pass

    if len(all_acts_list) < 10:
        print(f"  WARNING: Only {len(all_acts_list)} activations for {arch_desc}")
        return None

    all_acts = np.stack(all_acts_list)
    all_word_arr = np.array(all_word_list)
    first_letters_arr = np.array([w[0] for w in all_word_arr])
    letter_list = sorted(good_letters.keys())

    letter_probe_dirs = {}
    letters_with_probes = []
    for letter in letter_list:
        y = (first_letters_arr == letter).astype(int)
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        try:
            clf = sklm.LogisticRegression(C=1.0, max_iter=300, random_state=SEED, solver='lbfgs')
            clf.fit(all_acts, y)
            probe_dir = clf.coef_[0]
            probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
            letter_probe_dirs[letter] = probe_dir
            letters_with_probes.append(letter)
        except Exception:
            pass

    print(f"  Probes trained for {len(letters_with_probes)} letters")
    if not letter_probe_dirs:
        return None

    with torch.no_grad():
        W_dec = sae.W_dec.detach().float()  # (d_sae, d_in)
        W_dec_norm = F.normalize(W_dec, dim=1).cpu().numpy()
        d_sae = W_dec.shape[0]
        # EDA = 1 - cos(encoder_j, decoder_j)
        try:
            W_enc = sae.W_enc.detach().float()  # (d_in, d_sae)
            w_enc = W_enc.T  # (d_sae, d_in)
            enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
            dec_norms = W_dec.norm(dim=1).clamp(min=1e-8)
            cos_ed = ((w_enc * W_dec).sum(dim=1) / (enc_norms * dec_norms)).cpu().numpy()
            eda_scores = 1.0 - cos_ed
        except AttributeError:
            print(f"  WARNING: Cannot compute EDA for {arch_desc}")
            eda_scores = np.zeros(d_sae)

    probe_dirs = np.stack([letter_probe_dirs[lt] for lt in letters_with_probes])
    cos_probe_dec = probe_dirs @ W_dec_norm.T  # (n_letters, d_sae)
    max_probe_cos = cos_probe_dec.max(axis=0)  # (d_sae,)

    best_thr = 0.3
    for thr in np.arange(0.20, 0.55, 0.01):
        n = (max_probe_cos >= thr).sum()
        if abs(n - target_n_pos) < abs((max_probe_cos >= best_thr).sum() - target_n_pos):
            best_thr = thr
    if (max_probe_cos >= best_thr).sum() < 20:
        best_thr = 0.25

    THRESHOLD = float(best_thr)
    letter_feature_mask = (max_probe_cos >= THRESHOLD)
    letter_feature_ids = np.where(letter_feature_mask)[0]
    n_pos = len(letter_feature_ids)
    print(f"  Letter features: n_pos={n_pos} (threshold={THRESHOLD:.2f}, d_sae={d_sae})")

    letter_eda = eda_scores[letter_feature_ids]
    nonletter_mask = ~letter_feature_mask
    nonletter_eda = eda_scores[nonletter_mask]

    mean_eda_letter = float(np.mean(letter_eda)) if len(letter_eda) > 0 else float('nan')
    mean_eda_nonletter = float(np.mean(nonletter_eda)) if len(nonletter_eda) > 0 else float('nan')
    eda_delta = float(mean_eda_letter - mean_eda_nonletter) if not np.isnan(mean_eda_letter) else float('nan')
    frac_eda_gt50 = float((letter_eda > 0.5).mean()) if len(letter_eda) > 0 else float('nan')

    if len(letter_eda) > 1 and len(nonletter_eda) > 1:
        n_neg_sample = min(len(nonletter_eda), 500)
        neg_idx = np.random.choice(len(nonletter_eda), n_neg_sample, replace=False)
        neg_sample = nonletter_eda[neg_idx]
        try:
            w_stat, w_pval = stats.ranksums(letter_eda, neg_sample)
        except Exception:
            w_stat, w_pval = float('nan'), float('nan')
    else:
        w_stat, w_pval = float('nan'), float('nan')

    # Pairwise cos^2 between letter features
    if n_pos >= 2:
        sample_size = min(n_pos, 50)
        np.random.seed(SEED)
        sample_ids_idx = np.random.choice(len(letter_feature_ids), sample_size, replace=False)
        sample_ids = letter_feature_ids[sample_ids_idx]
        letter_dec = W_dec_norm[sample_ids]  # (sample_size, d_in)
        pair_cos = letter_dec @ letter_dec.T  # (sample_size, sample_size)
        upper_tri_idx = np.triu_indices(sample_size, k=1)
        pair_cos_values = pair_cos[upper_tri_idx]
        pair_cos_sq_values = pair_cos_values ** 2
        mean_pair_cos_sq = float(np.mean(pair_cos_sq_values))
        median_pair_cos_sq = float(np.median(pair_cos_sq_values))
        n_pairs = len(pair_cos_sq_values)
    else:
        pair_cos_sq_values = np.array([])
        mean_pair_cos_sq = float('nan')
        median_pair_cos_sq = float('nan')
        n_pairs = 0

    return {
        "n_pos": n_pos,
        "d_sae": int(d_sae),
        "threshold": THRESHOLD,
        "letters_with_probes": letters_with_probes,
        "letter_feature_ids": letter_feature_ids.tolist(),
        "eda_scores_letter_mean": mean_eda_letter,
        "eda_scores_nonletter_mean": mean_eda_nonletter,
        "eda_delta": eda_delta,
        "frac_eda_gt50": frac_eda_gt50,
        "letter_eda_list": letter_eda.tolist(),
        "letter_eda_percentiles": {
            "p25": float(np.percentile(letter_eda, 25)) if len(letter_eda) > 0 else float('nan'),
            "p50": float(np.percentile(letter_eda, 50)) if len(letter_eda) > 0 else float('nan'),
            "p75": float(np.percentile(letter_eda, 75)) if len(letter_eda) > 0 else float('nan'),
        },
        "wilcoxon_letter_vs_nonletter": {"statistic": float(w_stat), "p_value": float(w_pval)},
        "pair_cos_sq_between_letter_features": {
            "n_pairs_sampled": n_pairs,
            "mean": mean_pair_cos_sq,
            "median": median_pair_cos_sq,
        },
    }


# ─── Step 1: Try to access Gemma 2 2B (primary plan) ─────────────────────────
report_progress(1, TOTAL_STEPS, "Checking Gemma 2 2B accessibility")

from transformer_lens import HookedTransformer
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

gemma_accessible = False
try:
    # Quick test: try loading tokenizer only
    from transformers import AutoTokenizer
    tok_test = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    gemma_accessible = True
    print("Gemma 2 2B tokenizer accessible")
except Exception as e:
    gemma_accessible = False
    print(f"Gemma 2 2B NOT accessible: {e}")

# ─── Step 2: Fallback to GPT-2 Small ─────────────────────────────────────────
if not gemma_accessible:
    report_progress(2, TOTAL_STEPS, "Gemma inaccessible, falling back to GPT-2 Small comparison")
    print("FALLBACK: Using GPT-2 Small Standard/ReLU vs TopK comparison")
    print("Note: No Matryoshka SAEs available for GPT-2. Using Standard/ReLU vs TopK as architecture proxy.")

    USE_GEMMA = False
    MODEL_NAME = "gpt2-small"
    LAYER = 6
    HOOK_NAME_STD = "blocks.6.hook_resid_pre"
    HOOK_NAME_TOPK = "blocks.6.hook_resid_post"

    # Architectures to compare:
    # Arch1: Standard/ReLU SAE (L1 penalty) — proxy for "unconstrained" architecture
    # Arch2: TopK SAE (exact k) — proxy for "constrained" architecture
    # Note: Matryoshka not available for GPT-2; B3 data reused + deeper analysis
    ARCH1_RELEASE = "gpt2-small-res-jb"
    ARCH1_SAE_ID = "blocks.6.hook_resid_pre"
    ARCH2_RELEASE = "gpt2-small-resid-post-v5-32k"
    ARCH2_SAE_ID = "blocks.6.hook_resid_post"

    try:
        model = HookedTransformer.from_pretrained(
            "gpt2",
            center_unembed=True, center_writing_weights=True,
            fold_ln=True, refactor_factored_attn_matrices=True,
        )
        model = model.to(device)
        model.eval()
        print(f"GPT-2 Small: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
        model_loaded = True
        D_MODEL = model.cfg.d_model
    except Exception as e:
        print(f"ERROR loading GPT-2: {e}")
        model_loaded = False
else:
    report_progress(2, TOTAL_STEPS, "Gemma 2 2B accessible, proceeding with primary plan")
    USE_GEMMA = True
    MODEL_NAME = "gemma-2-2b"
    LAYER = 12
    HOOK_NAME_STD = f"blocks.{LAYER}.hook_resid_post"
    HOOK_NAME_TOPK = f"blocks.{LAYER}.hook_resid_post"
    ARCH1_RELEASE = "gemma-2-2b-res-matryoshka-dc"
    ARCH1_SAE_ID = f"blocks.{LAYER}.hook_resid_post"
    ARCH2_RELEASE = "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109"
    ARCH2_SAE_ID = f"blocks.{LAYER}.hook_resid_post__trainer_0"

    try:
        model = HookedTransformer.from_pretrained(
            "gemma-2-2b",
            center_unembed=True, center_writing_weights=True,
            fold_ln=True, refactor_factored_attn_matrices=True,
        )
        model = model.to(device)
        model.eval()
        print(f"Gemma 2 2B: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
        model_loaded = True
        D_MODEL = model.cfg.d_model
    except Exception as e:
        print(f"ERROR loading Gemma 2 2B: {e}")
        model_loaded = False

if not model_loaded:
    summary = "Model load failed. Task F2 cannot proceed."
    result_dict = {
        "task_id": TASK_ID, "mode": "PILOT", "timestamp": datetime.now().isoformat(),
        "pilot_pass": False, "skip_reason": summary,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result_dict, f, indent=2)
    mark_done("failed", summary, {"pilot_pass": False})
    sys.exit(0)

# ─── Step 3: Build vocabulary ─────────────────────────────────────────────────
report_progress(3, TOTAL_STEPS, "Building vocabulary for first-letter task")

tokenizer = model.tokenizer
valid_words = []
for word in SIMPLE_WORDS:
    word = word.strip().lower()
    if not word.isalpha() or len(word) < 2:
        continue
    try:
        toks = tokenizer.encode(" " + word)
        if len(toks) <= (2 if USE_GEMMA else 1):
            valid_words.append(word)
    except Exception:
        pass

vocab_by_letter = {lt: [] for lt in string.ascii_lowercase}
for word in valid_words:
    vocab_by_letter[word[0]].append(word)
good_letters = {lt: ws for lt, ws in vocab_by_letter.items() if len(ws) >= 5}
print(f"Vocabulary: {len(valid_words)} words, {len(good_letters)} letters with >=5 words")

rng_random = random.Random(SEED)
probe_train_words = []
for lt in sorted(good_letters.keys()):
    ws = good_letters[lt]
    probe_train_words.extend(rng_random.sample(ws, min(len(ws), 50)))
print(f"Probe training words: {len(probe_train_words)}")

# ─── Step 4: Load Arch1 SAE ───────────────────────────────────────────────────
arch1_name = "Matryoshka" if USE_GEMMA else "Standard/ReLU (L1)"
arch2_name = "TopK (gemma-2b, w=2^14)" if USE_GEMMA else "TopK (k=32)"

report_progress(4, TOTAL_STEPS, f"Loading Arch1 SAE ({arch1_name})")

arch1_loaded = False
sae_arch1 = None
arch1_info = {}

try:
    sae_arch1, cfg1, _ = SAE.from_pretrained(release=ARCH1_RELEASE, sae_id=ARCH1_SAE_ID)
    sae_arch1 = sae_arch1.to(device)
    sae_arch1.eval()
    d_sae_arch1 = sae_arch1.cfg.d_sae
    print(f"Arch1 SAE loaded: class={type(sae_arch1).__name__}, d_sae={d_sae_arch1}")
    arch1_info = {
        "name": arch1_name, "release": ARCH1_RELEASE, "sae_id": ARCH1_SAE_ID,
        "class": type(sae_arch1).__name__, "d_sae": d_sae_arch1,
    }
    arch1_loaded = True
except Exception as e:
    print(f"ERROR loading Arch1 SAE: {e}")
    arch1_info = {"name": arch1_name, "error": str(e)}

# ─── Step 5: Load Arch2 SAE ───────────────────────────────────────────────────
report_progress(5, TOTAL_STEPS, f"Loading Arch2 SAE ({arch2_name})")

arch2_loaded = False
sae_arch2 = None
arch2_info = {}

try:
    sae_arch2, cfg2, _ = SAE.from_pretrained(release=ARCH2_RELEASE, sae_id=ARCH2_SAE_ID)
    sae_arch2 = sae_arch2.to(device)
    sae_arch2.eval()
    d_sae_arch2 = sae_arch2.cfg.d_sae
    k_val = getattr(sae_arch2.cfg, 'k', 'N/A')
    print(f"Arch2 SAE loaded: class={type(sae_arch2).__name__}, d_sae={d_sae_arch2}, k={k_val}")
    arch2_info = {
        "name": arch2_name, "release": ARCH2_RELEASE, "sae_id": ARCH2_SAE_ID,
        "class": type(sae_arch2).__name__, "d_sae": d_sae_arch2, "k": k_val,
    }
    arch2_loaded = True
except Exception as e:
    print(f"ERROR loading Arch2 SAE: {e}")
    arch2_info = {"name": arch2_name, "error": str(e)}

if not arch1_loaded and not arch2_loaded:
    summary = "Both SAEs failed to load. Task F2 skipped."
    result_dict = {
        "task_id": TASK_ID, "mode": "PILOT", "timestamp": datetime.now().isoformat(),
        "pilot_pass": False, "skip_reason": summary,
        "arch1_info": arch1_info, "arch2_info": arch2_info,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(result_dict, f, indent=2)
    mark_done("failed", summary, {"pilot_pass": False})
    sys.exit(0)

# ─── Step 6: Identify letter features in Arch1 ───────────────────────────────
report_progress(6, TOTAL_STEPS, f"Identifying letter features in Arch1 ({arch1_name})")

arch1_result = None
if arch1_loaded:
    hook1 = HOOK_NAME_STD
    arch1_result = identify_letter_features_gpt2(
        model, sae_arch1, hook1, good_letters, probe_train_words,
        target_n_pos=67, arch_desc=f"{arch1_name} L{LAYER}"
    )
    if arch1_result is None:
        print(f"WARNING: Failed to identify letter features for {arch1_name}")
        arch1_loaded = False
    else:
        arch1_result["hook_name"] = hook1
        print(f"{arch1_name}: n_pos={arch1_result['n_pos']}, "
              f"mean_EDA={arch1_result['eda_scores_letter_mean']:.4f}, "
              f"frac_gt50={arch1_result['frac_eda_gt50']:.3f}")

# ─── Step 7: Identify letter features in Arch2 ───────────────────────────────
report_progress(7, TOTAL_STEPS, f"Identifying letter features in Arch2 ({arch2_name})")

arch2_result = None
if arch2_loaded:
    hook2 = HOOK_NAME_TOPK
    arch2_result = identify_letter_features_gpt2(
        model, sae_arch2, hook2, good_letters, probe_train_words,
        target_n_pos=67, arch_desc=f"{arch2_name} L{LAYER}"
    )
    if arch2_result is None:
        print(f"WARNING: Failed to identify letter features for {arch2_name}")
        arch2_loaded = False
    else:
        arch2_result["hook_name"] = hook2
        print(f"{arch2_name}: n_pos={arch2_result['n_pos']}, "
              f"mean_EDA={arch2_result['eda_scores_letter_mean']:.4f}, "
              f"frac_gt50={arch2_result['frac_eda_gt50']:.3f}")

# ─── Step 8: Statistical comparison ──────────────────────────────────────────
report_progress(8, TOTAL_STEPS, "Statistical comparison of EDA and decoder geometry")

comparison = None
decoder_geometry_comparison = None

if arch1_result and arch2_result:
    a1_eda = arch1_result["eda_scores_letter_mean"]
    a2_eda = arch2_result["eda_scores_letter_mean"]
    eda_diff = a2_eda - a1_eda
    rel_diff = eda_diff / a1_eda if a1_eda > 0 else float('nan')

    a1_eda_list = arch1_result.get("letter_eda_list", [])
    a2_eda_list = arch2_result.get("letter_eda_list", [])

    w_stat, w_pval = float('nan'), float('nan')
    mwu, mwu_p, rr = float('nan'), float('nan'), float('nan')

    if len(a1_eda_list) >= 3 and len(a2_eda_list) >= 3:
        try:
            w_stat, w_pval = stats.ranksums(a1_eda_list, a2_eda_list)
            mwu, mwu_p = stats.mannwhitneyu(a1_eda_list, a2_eda_list, alternative='two-sided')
            n1, n2 = len(a1_eda_list), len(a2_eda_list)
            rr = float(1 - (2 * mwu) / (n1 * n2))
        except Exception as e:
            print(f"WARNING: Statistical test error: {e}")

    # Theory prediction:
    # - If Gemma Matryoshka: Matryoshka should have LOWER EDA than TopK
    # - If GPT-2 fallback: Standard/ReLU should have HIGHER EDA than TopK (confirmed in B3)
    if USE_GEMMA:
        theory_confirmed = bool(a1_eda < a2_eda)  # Matryoshka lower
        theory_text = "Matryoshka should have lower EDA (less absorbed) than TopK"
    else:
        theory_confirmed = bool(a1_eda > a2_eda)  # Standard/ReLU higher (already confirmed in B3)
        theory_text = "Standard/ReLU (L1) should have higher EDA (more absorbed) than TopK"

    comparison = {
        "arch1_name": arch1_name,
        "arch2_name": arch2_name,
        "arch1_n_pos": arch1_result["n_pos"],
        "arch2_n_pos": arch2_result["n_pos"],
        "arch1_d_sae": arch1_result["d_sae"],
        "arch2_d_sae": arch2_result["d_sae"],
        "mean_eda_arch1": a1_eda,
        "mean_eda_arch2": a2_eda,
        "eda_diff_arch2_minus_arch1": eda_diff,
        "eda_relative_diff": float(rel_diff),
        "frac_eda_gt50_arch1": arch1_result["frac_eda_gt50"],
        "frac_eda_gt50_arch2": arch2_result["frac_eda_gt50"],
        "theory_prediction": theory_text,
        "theory_prediction_confirmed": theory_confirmed,
        "wilcoxon_test": {"statistic": float(w_stat), "p_value": float(w_pval)},
        "mannwhitney_test": {
            "u_statistic": float(mwu), "p_value": float(mwu_p),
            "rank_biserial_r": float(rr),
        },
        "interpretation": (
            f"{arch1_name} EDA={a1_eda:.4f}, {arch2_name} EDA={a2_eda:.4f}. "
            f"Diff={eda_diff:.4f}. "
            f"Theory ({theory_text}) {'CONFIRMED' if theory_confirmed else 'NOT CONFIRMED'}. "
            f"Wilcoxon p={w_pval:.4f}."
        ),
    }

    print(f"\n=== EDA Comparison ===")
    print(f"{arch1_name}: EDA={a1_eda:.4f}, frac_gt50={arch1_result['frac_eda_gt50']:.3f}")
    print(f"{arch2_name}: EDA={a2_eda:.4f}, frac_gt50={arch2_result['frac_eda_gt50']:.3f}")
    print(f"Theory confirmed: {theory_confirmed}")
    print(f"Wilcoxon p: {w_pval:.4f}")

    # Decoder geometry comparison
    a1_cos2_mean = arch1_result["pair_cos_sq_between_letter_features"]["mean"]
    a2_cos2_mean = arch2_result["pair_cos_sq_between_letter_features"]["mean"]

    if USE_GEMMA:
        cos2_theory_confirmed = bool(a1_cos2_mean < a2_cos2_mean)  # Matryoshka lower cos2
    else:
        cos2_theory_confirmed = bool(a1_cos2_mean > a2_cos2_mean)  # Standard higher cos2

    decoder_geometry_comparison = {
        "arch1_mean_pair_cos2": a1_cos2_mean,
        "arch2_mean_pair_cos2": a2_cos2_mean,
        "arch1_median_pair_cos2": arch1_result["pair_cos_sq_between_letter_features"]["median"],
        "arch2_median_pair_cos2": arch2_result["pair_cos_sq_between_letter_features"]["median"],
        "cos2_diff_arch2_minus_arch1": float(a2_cos2_mean - a1_cos2_mean),
        "theory_prediction_confirmed": cos2_theory_confirmed,
        "interpretation": (
            f"{arch1_name} letter pair cos^2: {a1_cos2_mean:.6f}. "
            f"{arch2_name} letter pair cos^2: {a2_cos2_mean:.6f}. "
            f"Cos^2 diff = {a2_cos2_mean - a1_cos2_mean:.6f}."
        ),
    }

    print(f"\n=== Decoder Geometry ===")
    print(f"{arch1_name} mean pair cos^2: {a1_cos2_mean:.6f}")
    print(f"{arch2_name} mean pair cos^2: {a2_cos2_mean:.6f}")

# ─── Step 9: Load and incorporate B3 data for GPT-2 fallback ─────────────────
report_progress(9, TOTAL_STEPS, "Loading B3 cross-architecture data for reference")

b3_data = None
if not USE_GEMMA:
    b3_path = RESULTS_DIR / "B3_cross_arch.json"
    if b3_path.exists():
        try:
            with open(b3_path) as f:
                b3_data = json.load(f)
            print(f"Loaded B3 data: {b3_data.get('finding', '')[:100]}")
        except Exception as e:
            print(f"Could not load B3: {e}")

# ─── Step 10: Null distribution ───────────────────────────────────────────────
report_progress(10, TOTAL_STEPS, "Computing shuffle-label null distributions")

null_eda_diff_mean = float('nan')
null_eda_diff_std = float('nan')

if arch1_result and arch2_result:
    a1_arr = np.array(arch1_result.get("letter_eda_list", []))
    a2_arr = np.array(arch2_result.get("letter_eda_list", []))
    if len(a1_arr) >= 3 and len(a2_arr) >= 3:
        all_eda = np.concatenate([a1_arr, a2_arr])
        n1 = len(a1_arr)
        null_diffs = []
        rng_null = np.random.RandomState(SEED)
        for _ in range(100):
            rng_null.shuffle(all_eda)
            null_a1 = all_eda[:n1]
            null_a2 = all_eda[n1:]
            null_diffs.append(float(np.mean(null_a2) - np.mean(null_a1)))
        null_eda_diff_mean = float(np.mean(null_diffs))
        null_eda_diff_std = float(np.std(null_diffs))
        print(f"Null EDA diff: mean={null_eda_diff_mean:.4f}, std={null_eda_diff_std:.4f}")
        if comparison:
            obs_diff = comparison["eda_diff_arch2_minus_arch1"]
            z = (obs_diff - null_eda_diff_mean) / null_eda_diff_std if null_eda_diff_std > 0 else float('nan')
            print(f"Observed diff: {obs_diff:.4f}, z-score: {z:.2f}")
            comparison["null_eda_diff_mean"] = null_eda_diff_mean
            comparison["null_eda_diff_std"] = null_eda_diff_std
            comparison["z_score_vs_null"] = float(z)

# ─── Step 11: Pilot pass assessment ──────────────────────────────────────────
report_progress(11, TOTAL_STEPS, "Assessing pilot pass criteria")

n_pos_a1 = arch1_result["n_pos"] if arch1_result else 0
n_pos_a2 = arch2_result["n_pos"] if arch2_result else 0
architectures_available = sum([
    1 if (arch1_loaded and arch1_result) else 0,
    1 if (arch2_loaded and arch2_result) else 0,
])
n_pos_min = min(n_pos_a1, n_pos_a2) if architectures_available >= 2 else max(n_pos_a1, n_pos_a2)
pilot_pass = architectures_available >= 2 and n_pos_min >= 10

print(f"Pilot pass: {pilot_pass}")
print(f"Architectures: {architectures_available}/2, min n_pos: {n_pos_min}")

# ─── Step 12: Save results and finalize ───────────────────────────────────────
report_progress(12, TOTAL_STEPS, "Saving results and writing completion markers")

elapsed_sec = time.time() - start_time

# Determine skip/limitation note
if not USE_GEMMA:
    scope_note = (
        "PRIMARY PLAN SKIPPED: Gemma 2 2B is gated (HuggingFace access required). "
        "No Matryoshka SAEs exist for GPT-2 Small. "
        "FALLBACK: Standard/ReLU (L1) vs TopK comparison on GPT-2 Small layer 6. "
        "This does not test Matryoshka specifically. "
        "Analysis limited to architectural comparison (L1-trained vs TopK-trained SAEs). "
        "The theoretical account of Matryoshka mitigation is documented in F1_theory_analysis.md."
    )
else:
    scope_note = (
        f"Primary plan: Matryoshka vs TopK on {MODEL_NAME} layer {LAYER}. "
        "EDA (1 - cos(encoder_j, decoder_j)) used as absorption proxy (no exact absorption labels for Gemma)."
    )

results = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": elapsed_sec,
    "config": {
        "model": MODEL_NAME,
        "layer": LAYER,
        "used_gemma": USE_GEMMA,
        "gemma_accessible": gemma_accessible,
        "seed": SEED,
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    },
    "pilot_pass": pilot_pass,
    "pass_criteria": "Both architectures loaded with >= 10 letter features each",
    "architectures_available": architectures_available,
    "arch1_info": arch1_info,
    "arch2_info": arch2_info,
    "arch1_result": arch1_result,
    "arch2_result": arch2_result,
    "comparison": comparison,
    "decoder_geometry_comparison": decoder_geometry_comparison,
    "b3_reference_data_summary": {
        "finding": b3_data.get("finding", "") if b3_data else None,
        "arch1_mean_eda": b3_data["arch1"]["result"]["eda_scores_letter_mean"] if b3_data else None,
        "arch2_mean_eda": b3_data["arch2"]["result"]["eda_scores_letter_mean"] if b3_data else None,
        "note": "B3 used Standard/ReLU and TopK SAEs on GPT-2 Small layer 6 with same comparison setup",
    } if not USE_GEMMA else None,
    "null_distribution": {
        "null_eda_diff_mean": null_eda_diff_mean,
        "null_eda_diff_std": null_eda_diff_std,
        "n_permutations": 100,
    },
    "scope_note": scope_note,
    "key_findings": {},
    "methods_note": (
        "Task F2 empirical verification: "
        "Matryoshka SAEs unavailable for GPT-2 (only Gemma 2 2B, which is gated). "
        "Using Standard/ReLU vs TopK as architectural proxy comparison. "
        "Full Matryoshka verification requires Gemma 2 2B access. "
        "See F1_theory_analysis.md for theoretical account of how Matryoshka should reduce absorption."
    ) if not USE_GEMMA else None,
}

if comparison:
    results["key_findings"]["eda_comparison"] = comparison["interpretation"]
    results["key_findings"]["theory_status"] = (
        f"Theory prediction {'CONFIRMED' if comparison['theory_prediction_confirmed'] else 'NOT CONFIRMED'}: "
        f"{comparison['theory_prediction']}"
    )
    print(f"\n=== KEY FINDINGS ===")
    print(results["key_findings"]["eda_comparison"])
    print(results["key_findings"]["theory_status"])

if decoder_geometry_comparison:
    results["key_findings"]["decoder_geometry"] = decoder_geometry_comparison["interpretation"]

with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {OUTPUT_FILE}")

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    with open(gpu_progress_path) as f:
        gp = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

if TASK_ID not in gp.get("completed", []):
    gp.setdefault("completed", []).append(TASK_ID)
if TASK_ID in gp.get("failed", []):
    gp["failed"].remove(TASK_ID)
if TASK_ID in gp.get("running", {}):
    del gp["running"][TASK_ID]

gp.setdefault("timings", {})[TASK_ID] = {
    "planned_min": 30,
    "actual_min": int(elapsed_sec / 60),
    "start_time": datetime.fromtimestamp(start_time).isoformat(),
    "end_time": datetime.now().isoformat(),
    "config_snapshot": {
        "model": MODEL_NAME,
        "layer": LAYER,
        "used_gemma": USE_GEMMA,
        "architectures_available": architectures_available,
        "pilot_pass": pilot_pass,
        "gpu_model": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
    }
}

with open(gpu_progress_path, 'w') as f:
    json.dump(gp, f, indent=2)

summary = (
    f"F2 Mitigation Verification pilot complete. "
    f"Used {'Gemma 2 2B Matryoshka vs TopK' if USE_GEMMA else 'GPT-2 Standard/ReLU vs TopK (Gemma gated)'}. "
    f"Architectures: {architectures_available}/2. "
    f"Pilot pass: {pilot_pass}. "
    f"n_pos: arch1={n_pos_a1}, arch2={n_pos_a2}."
)

mark_done(
    status="success" if pilot_pass else "partial",
    summary=summary,
    result={
        "pilot_pass": pilot_pass,
        "architectures_available": architectures_available,
        "used_gemma": USE_GEMMA,
        "n_pos_arch1": n_pos_a1,
        "n_pos_arch2": n_pos_a2,
        "theory_confirmed": comparison["theory_prediction_confirmed"] if comparison else None,
    }
)

print(f"\n=== F2 PILOT COMPLETE ===")
print(f"Elapsed: {elapsed_sec:.1f}s")
print(f"Pilot pass: {pilot_pass}")
print(f"Output: {OUTPUT_FILE}")
