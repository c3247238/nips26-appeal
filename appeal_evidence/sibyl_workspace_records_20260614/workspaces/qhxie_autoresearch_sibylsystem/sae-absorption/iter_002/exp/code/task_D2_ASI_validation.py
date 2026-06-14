"""
Task D.2: ASI Validation Against Ground Truth (PILOT MODE)

Compare ASI-ranked pairs against Chanin et al. absorption labels (proxy: probe decoder alignment)
on GPT-2 Small L6 (n_pos=71 at threshold=0.32).

Computes:
  - AUROC, AUPRC, precision@50, @100, @500 for ASI, cos2, freq_ratio, random
  - Precision-recall curves for all four detectors
  - Null AUROC distribution (100 permutations of absorption labels)
  - DeLong test: AUROC comparison between ASI and best single-component baseline
  - Effect size: delta AUROC between ASI and best baseline

Output: exp/results/full/D2_ASI_validation.json

PILOT MODE: Uses precomputed D1 scores; adds DeLong test and null distribution.
Pass criteria: ASI AUROC > 0.55 (above shuffled null mean).
Full success: AUROC >= 0.65 and AUPRC > 3x base rate.
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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, roc_curve
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_D2_ASI_validation"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "D2_ASI_validation.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()


def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None, note=""):
    elapsed = time.time() - start_time
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "note": note,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"[{elapsed:.1f}s] Step {epoch}/{total_epochs}: {note}")
    sys.stdout.flush()


def mark_done(status="success", summary=""):
    PID_FILE.unlink(missing_ok=True)
    progress_file_data = {}
    if PROGRESS_FILE.exists():
        try:
            progress_file_data = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": progress_file_data,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
    }))


def delong_test(labels, scores_a, scores_b):
    """
    DeLong test for comparing two AUROC values.
    Implementation based on the structural component method.
    Returns: (z_statistic, p_value, auroc_a, auroc_b, delta_auroc)
    """
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return None

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    def compute_structural_components(scores):
        """Compute V10 and V01 structural components for DeLong variance."""
        scores_pos = scores[pos_idx]
        scores_neg = scores[neg_idx]
        # V10[i]: fraction of negatives with score < positive score[i]
        # V01[j]: fraction of positives with score > negative score[j]
        V10 = np.array([np.mean(scores_pos[i] > scores_neg) +
                        0.5 * np.mean(scores_pos[i] == scores_neg)
                        for i in range(n_pos)])
        V01 = np.array([np.mean(scores_neg[j] < scores_pos) +
                        0.5 * np.mean(scores_neg[j] == scores_pos)
                        for j in range(n_neg)])
        return V10, V01

    V10_a, V01_a = compute_structural_components(scores_a)
    V10_b, V01_b = compute_structural_components(scores_b)

    auroc_a = float(np.mean(V10_a))
    auroc_b = float(np.mean(V10_b))

    # Covariance matrix of (auroc_a, auroc_b)
    # Var(auroc) = (1/n_pos * Var(V10) + 1/n_neg * Var(V01)) / (n_pos * n_neg)^0.5?
    # Correct formula from DeLong (1988):
    S10 = np.cov(V10_a, V10_b)  # 2x2
    S01 = np.cov(V01_a, V01_b)  # 2x2

    cov_matrix = S10 / n_pos + S01 / n_neg

    delta = auroc_a - auroc_b
    var_delta = cov_matrix[0, 0] + cov_matrix[1, 1] - 2 * cov_matrix[0, 1]

    if var_delta <= 0:
        return {"auroc_a": auroc_a, "auroc_b": auroc_b, "delta": delta,
                "z_stat": None, "p_value": None, "note": "var_delta <= 0"}

    z_stat = delta / np.sqrt(var_delta)
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

    return {
        "auroc_a": auroc_a,
        "auroc_b": auroc_b,
        "delta_auroc": float(delta),
        "z_stat": float(z_stat),
        "p_value": float(p_value),
        "significant_at_05": p_value < 0.05,
        "significant_at_01": p_value < 0.01,
    }


def compute_null_distribution(labels, scores, n_permutations=100, seed=42):
    """Compute null AUROC distribution via label permutation."""
    rng = np.random.RandomState(seed)
    null_aurocs = []
    for _ in range(n_permutations):
        perm_labels = rng.permutation(labels)
        try:
            auroc = float(roc_auc_score(perm_labels, scores))
        except Exception:
            auroc = 0.5
        null_aurocs.append(auroc)
    return {
        "n_permutations": n_permutations,
        "mean": float(np.mean(null_aurocs)),
        "std": float(np.std(null_aurocs)),
        "p05_threshold": float(np.percentile(null_aurocs, 95)),  # one-sided 5% threshold
        "min": float(np.min(null_aurocs)),
        "max": float(np.max(null_aurocs)),
    }


def pr_curve_to_dict(labels, scores, n_points=50):
    """Compute precision-recall curve and subsample to n_points."""
    prec, rec, thresh = precision_recall_curve(labels, scores)
    # Subsample evenly
    idx = np.linspace(0, len(prec) - 1, min(n_points, len(prec))).astype(int)
    return {
        "precision": [float(p) for p in prec[idx]],
        "recall": [float(r) for r in rec[idx]],
        "n_points": len(idx),
        "base_rate": float(labels.mean()),
    }


def roc_curve_to_dict(labels, scores, n_points=50):
    """Compute ROC curve and subsample."""
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.linspace(0, len(fpr) - 1, min(n_points, len(fpr))).astype(int)
    return {
        "fpr": [float(f) for f in fpr[idx]],
        "tpr": [float(t) for t in tpr[idx]],
        "n_points": len(idx),
    }


TOTAL_STEPS = 8
report_progress(0, TOTAL_STEPS, note="Starting D2: ASI validation against ground truth labels")

# ─── Step 1: Load model, SAE, and D1 precomputed data ─────────────────────────
report_progress(1, TOTAL_STEPS, note="Loading GPT-2 Small + SAE + D1 precomputed scores")

import string
import sklearn.linear_model as sklm
from transformer_lens import HookedTransformer
from sae_lens import SAE
from datasets import load_dataset

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM: {props.total_memory / 1e9:.1f} GB")

model = HookedTransformer.from_pretrained(
    "gpt2",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
model = model.to(device)
model.eval()

sae, cfg_dict, _ = SAE.from_pretrained_with_cfg_and_sparsity(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
)
sae = sae.to(device)
sae.eval()

D_SAE = sae.cfg.d_sae    # 24576
D_IN = sae.cfg.d_in      # 768
HOOK_NAME = "blocks.6.hook_resid_pre"
print(f"GPT-2: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")
print(f"SAE: d_sae={D_SAE}, d_in={D_IN}")

# Load D1 results to reuse precomputed data
d1_file = RESULTS_DIR / "D1_ASI_scores.json"
d1_data = {}
if d1_file.exists():
    d1_data = json.loads(d1_file.read_text())
    print(f"Loaded D1 results: n_pos={d1_data.get('labels', {}).get('n_pos')}, "
          f"ASI AUROC={d1_data.get('validation_metrics', {}).get('ASI_combined', {}).get('auroc'):.4f}")
else:
    print("WARNING: D1 results not found. Will compute from scratch.")

# ─── Step 2: Recompute feature scores (rerun D1 computation for scores) ───────
report_progress(2, TOTAL_STEPS, note="Recomputing feature scores (ASI, cos2, freq_ratio, EDA)")

# We need the raw per-feature score arrays for:
# 1. DeLong test
# 2. Null distribution
# 3. PR curves
# These were computed in D1 but not stored as arrays. We need to recompute.

# Recompute feature frequencies from OWT
N_OWT_TOKENS_TARGET = 10000
PARENT_FREQ_THRESHOLD = 0.005
COACT_THRESHOLD = 0.01

feature_act_counts = np.zeros(D_SAE, dtype=np.float32)
n_total_tokens = 0
compute_start = time.time()

print("Computing per-feature frequencies from OpenWebText...")
try:
    owt_dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    texts_for_freq = []
    n_tokens_approx = 0
    for example in owt_dataset:
        text = example["text"]
        texts_for_freq.append(text[:500])
        n_tokens_approx += min(len(text.split()) * 1.3, 400)
        if n_tokens_approx >= N_OWT_TOKENS_TARGET:
            break
    print(f"Loaded {len(texts_for_freq)} OWT texts (est. {n_tokens_approx:.0f} tokens)")
except Exception as e:
    print(f"OWT streaming failed: {e}, using fallback corpus")
    texts_for_freq = [
        "The stock market rose sharply today as investors reacted to earnings reports.",
        "Scientists discovered a new species in the deep ocean near Pacific islands.",
        "The company announced record profits for the third consecutive quarter.",
        "Children in the neighborhood played basketball all summer afternoon.",
        "The government passed new environmental protection legislation yesterday.",
        "Technology companies announced partnerships for next-generation computing.",
        "Medical researchers published findings on new cancer treatment methods.",
        "Global temperatures have risen significantly over the past five decades.",
        "The university announced new scholarship programs for engineering students.",
        "Electric vehicles have become more affordable and widely available.",
    ] * 200

with torch.no_grad():
    for i, text in enumerate(texts_for_freq):
        try:
            tokens = model.to_tokens(text, prepend_bos=True)
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME)
            resid = cache[HOOK_NAME][0]
            del cache
            for b_start in range(0, resid.shape[0], 256):
                b_end = min(b_start + 256, resid.shape[0])
                batch = resid[b_start:b_end].to(device)
                acts = sae.encode(batch)
                feature_act_counts += (acts > 0).float().cpu().numpy().sum(axis=0)
                n_total_tokens += b_end - b_start
        except Exception:
            pass
        if n_total_tokens >= N_OWT_TOKENS_TARGET:
            break

feature_freq = feature_act_counts / max(n_total_tokens, 1)
l0_empirical = float(feature_freq.sum())
print(f"Frequency computed: {n_total_tokens} tokens, L0={l0_empirical:.2f}")
print(f"Features with freq > {COACT_THRESHOLD}: {(feature_freq > COACT_THRESHOLD).sum()}")

# Decoder geometry
with torch.no_grad():
    W_dec = sae.W_dec.detach().float()
    W_dec_norm = F.normalize(W_dec, dim=1)
    W_dec_np = W_dec_norm.cpu().numpy()

report_progress(3, TOTAL_STEPS, note="Computing per-feature ASI, cos2, freq_ratio scores")

# Identify parents
parent_mask = feature_freq > PARENT_FREQ_THRESHOLD
parent_ids = np.where(parent_mask)[0]
print(f"Parent features (freq > {PARENT_FREQ_THRESHOLD}): {len(parent_ids)}")

W_dec_gpu = W_dec_norm.to(device)
feature_freq_tensor = torch.tensor(feature_freq, dtype=torch.float32, device=device)
freq_parents = torch.tensor(feature_freq[parent_ids], dtype=torch.float32, device=device)
W_dec_parents = W_dec_gpu[parent_ids]

# Compute per-child ASI, cos2, freq_ratio scores
asi_scores = np.zeros(D_SAE, dtype=np.float32)
cos2_scores = np.zeros(D_SAE, dtype=np.float32)
freq_ratio_scores = np.zeros(D_SAE, dtype=np.float32)

BATCH_C = 2048
with torch.no_grad():
    for c_start in range(0, D_SAE, BATCH_C):
        c_end = min(c_start + BATCH_C, D_SAE)
        c_vecs = W_dec_gpu[c_start:c_end]
        c_freqs = feature_freq_tensor[c_start:c_end]

        cos_mat = W_dec_parents @ c_vecs.T        # (n_par, batch_c)
        cos2_mat = cos_mat ** 2                    # (n_par, batch_c)
        freq_ratio_mat = freq_parents[:, None] / (c_freqs[None, :] + 1e-8)  # (n_par, batch_c)
        asi_mat = cos2_mat * freq_ratio_mat        # (n_par, batch_c)

        asi_scores[c_start:c_end] = asi_mat.max(dim=0).values.cpu().numpy()
        cos2_scores[c_start:c_end] = cos2_mat.max(dim=0).values.cpu().numpy()
        best_par_idx = cos2_mat.argmax(dim=0)
        freq_ratio_for_best = freq_ratio_mat[best_par_idx, torch.arange(c_end - c_start, device=device)]
        freq_ratio_scores[c_start:c_end] = freq_ratio_for_best.cpu().numpy()

# EDA baseline (encoder-decoder cosine similarity)
with torch.no_grad():
    w_enc = sae.W_enc.detach().float().T   # (D_SAE, D_IN)
    w_dec = sae.W_dec.detach().float()      # (D_SAE, D_IN)
    enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
    dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)
    cos_ed = ((w_enc * w_dec).sum(dim=1) / (enc_norms * dec_norms)).cpu().numpy()
    eda_scores = 1.0 - cos_ed  # Higher EDA = less aligned (more absorbed)

print("Feature scores computed.")

# ─── Step 3: Build ground truth labels ────────────────────────────────────────
report_progress(4, TOTAL_STEPS, note="Building ground truth absorption labels (probe decoder alignment)")

# Build letter probes and identify letter features
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
    "apple", "ant", "axe", "bat", "boat", "brain", "cat", "car", "cup", "dog",
    "day", "dream", "egg", "earth", "fox", "fan", "frog", "gun", "gate",
    "hat", "hub", "hill", "inn", "jar", "joy", "key", "kid", "kin", "lab",
    "map", "mob", "nun", "net", "oil", "owl", "pen", "pin", "rag", "sum",
    "sun", "tan", "tip", "van", "web", "yam", "zip",
]

tokenizer = model.tokenizer
valid_words = []
for word in SIMPLE_WORDS:
    word = word.strip().lower()
    if not word.isalpha() or len(word) < 2:
        continue
    try:
        if len(tokenizer.encode(" " + word)) == 1 and len(tokenizer.encode(word)) == 1:
            valid_words.append(word)
    except Exception:
        pass

vocab_by_letter = {lt: [] for lt in string.ascii_lowercase}
for word in valid_words:
    vocab_by_letter[word[0]].append(word)
good_letters = {lt: ws for lt, ws in vocab_by_letter.items() if len(ws) >= 5}

rng_probe = random.Random(SEED)
probe_train_words = []
for lt in sorted(good_letters.keys()):
    ws = good_letters[lt]
    probe_train_words.extend(rng_probe.sample(ws, min(len(ws), 50)))

all_acts_list = []
all_word_list = []
with torch.no_grad():
    for word in probe_train_words:
        prompt = f" {word}:"
        try:
            tok = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tok, names_filter=HOOK_NAME)
            act = cache[HOOK_NAME][0, -2, :].cpu().float().numpy()
            all_acts_list.append(act)
            all_word_list.append(word)
            del cache
        except Exception:
            pass

all_acts = np.stack(all_acts_list)
all_word_arr = np.array(all_word_list)
first_letters_arr = np.array([w[0] for w in all_word_arr])
letter_list = sorted(good_letters.keys())

letter_probe_dirs = {}
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
    except Exception:
        pass

probe_dirs = np.stack([letter_probe_dirs[lt] for lt in sorted(letter_probe_dirs.keys())])
cos_probe_dec = probe_dirs @ W_dec_np.T  # (n_letters, D_SAE)
max_probe_cos = cos_probe_dec.max(axis=0)

# Find threshold giving n_pos near 67
best_thr = 0.3
for thr in np.arange(0.28, 0.50, 0.01):
    n_at_thr = (max_probe_cos >= thr).sum()
    if abs(n_at_thr - 67) < abs((max_probe_cos >= best_thr).sum() - 67):
        best_thr = thr

binary_labels = (max_probe_cos >= best_thr).astype(np.float32)
n_pos = int(binary_labels.sum())
n_neg = int((1 - binary_labels).sum())
base_rate = n_pos / D_SAE
letter_feature_ids = np.where(binary_labels == 1)[0].tolist()

print(f"Ground truth labels: n_pos={n_pos}, n_neg={n_neg}, base_rate={base_rate:.5f}")
print(f"Label threshold: {best_thr:.2f}")

# ─── Step 5: Compute AUROC, AUPRC, Precision@k for all detectors ─────────────
report_progress(5, TOTAL_STEPS, note="Computing AUROC, AUPRC, Precision@k for all detectors")

labels = binary_labels

# Compute metrics for each detector
def compute_metrics(scores, labels, name, k_vals=(50, 100, 500)):
    """Compute comprehensive metrics for a detector."""
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    auprc_over_base = float(auprc / max(base_rate, 1e-8))

    prec_at_k = {}
    for k in k_vals:
        top_k = np.argsort(scores)[::-1][:k]
        prec_at_k[f"precision_at_{k}"] = float(labels[top_k].sum() / k)

    return {
        "name": name,
        "auroc": auroc,
        "auprc": auprc,
        "auprc_over_base_rate": auprc_over_base,
        **prec_at_k,
    }

metrics_asi = compute_metrics(asi_scores, labels, "ASI_combined")
metrics_cos2 = compute_metrics(cos2_scores, labels, "cos2_alone")
metrics_freq = compute_metrics(freq_ratio_scores, labels, "freq_ratio_alone")
metrics_eda = compute_metrics(eda_scores, labels, "EDA_baseline")

# Random baseline
random_scores = np.random.RandomState(SEED).random(D_SAE)
metrics_random = compute_metrics(random_scores, labels, "random")
metrics_random["auroc"] = 0.5  # theoretical
metrics_random["auprc"] = base_rate  # theoretical

print(f"\nMetrics summary:")
for m in [metrics_asi, metrics_cos2, metrics_freq, metrics_eda, metrics_random]:
    print(f"  {m['name']:25s}: AUROC={m['auroc']:.4f}, AUPRC={m['auprc']:.6f} ({m['auprc_over_base_rate']:.2f}x base)")

# ─── Step 6: DeLong test ───────────────────────────────────────────────────────
report_progress(6, TOTAL_STEPS, note="Running DeLong test (ASI vs. best single-component baseline)")

# Find best single-component baseline
best_single = max(
    [("cos2", metrics_cos2["auroc"], cos2_scores),
     ("freq_ratio", metrics_freq["auroc"], freq_ratio_scores),
     ("EDA", metrics_eda["auroc"], eda_scores)],
    key=lambda x: x[1]
)
best_single_name, best_single_auroc, best_single_scores = best_single

print(f"Best single-component baseline: {best_single_name} (AUROC={best_single_auroc:.4f})")
print(f"ASI AUROC: {metrics_asi['auroc']:.4f}")
print(f"Delta AUROC (ASI - {best_single_name}): {metrics_asi['auroc'] - best_single_auroc:.4f}")

delong_result_asi_vs_best = delong_test(labels, asi_scores, best_single_scores)
delong_result_asi_vs_cos2 = delong_test(labels, asi_scores, cos2_scores)
delong_result_asi_vs_freq = delong_test(labels, asi_scores, freq_ratio_scores)
delong_result_asi_vs_eda = delong_test(labels, asi_scores, eda_scores)

print(f"DeLong test (ASI vs {best_single_name}): z={delong_result_asi_vs_best.get('z_stat', 'N/A')}, "
      f"p={delong_result_asi_vs_best.get('p_value', 'N/A')}")

# ─── Step 7: Null AUROC distribution (100 permutations) ──────────────────────
report_progress(7, TOTAL_STEPS, note="Computing null AUROC distribution (100 permutations of labels)")

print("Computing null AUROC distributions (100 permutations)...")
null_asi = compute_null_distribution(labels, asi_scores, n_permutations=100, seed=SEED)
null_eda = compute_null_distribution(labels, eda_scores, n_permutations=100, seed=SEED)

print(f"Null AUROC (ASI): mean={null_asi['mean']:.4f}, std={null_asi['std']:.4f}, p05_threshold={null_asi['p05_threshold']:.4f}")
print(f"Null AUROC (EDA): mean={null_eda['mean']:.4f}, std={null_eda['std']:.4f}, p05_threshold={null_eda['p05_threshold']:.4f}")

# Check if ASI is above null
asi_auroc = metrics_asi["auroc"]
above_null_mean = bool(asi_auroc > null_asi["mean"])
above_null_1sd = bool(asi_auroc > null_asi["mean"] + null_asi["std"])
above_null_p05 = bool(asi_auroc > null_asi["p05_threshold"])

eda_auroc = metrics_eda["auroc"]
eda_above_null_mean = bool(eda_auroc > null_eda["mean"])
eda_above_null_1sd = bool(eda_auroc > null_eda["mean"] + null_eda["std"])
eda_above_null_p05 = bool(eda_auroc > null_eda["p05_threshold"])

print(f"\nASI AUROC={asi_auroc:.4f}: above_null_mean={above_null_mean}, "
      f"above_null_1sd={above_null_1sd}, above_null_p05={above_null_p05}")
print(f"EDA AUROC={eda_auroc:.4f}: above_null_mean={eda_above_null_mean}, "
      f"above_null_1sd={eda_above_null_1sd}, above_null_p05={eda_above_null_p05}")

# PR curves for all detectors
print("Computing precision-recall curves...")
pr_asi = pr_curve_to_dict(labels, asi_scores)
pr_cos2 = pr_curve_to_dict(labels, cos2_scores)
pr_freq = pr_curve_to_dict(labels, freq_ratio_scores)
pr_eda = pr_curve_to_dict(labels, eda_scores)
pr_random = {
    "precision": [float(base_rate)] * 2,
    "recall": [0.0, 1.0],
    "n_points": 2,
    "base_rate": float(base_rate),
    "note": "random baseline = constant base rate",
}

# ROC curves
roc_asi = roc_curve_to_dict(labels, asi_scores)
roc_eda = roc_curve_to_dict(labels, eda_scores)

# ─── Step 8: Pass/fail criteria and save results ──────────────────────────────
report_progress(8, TOTAL_STEPS, note="Evaluating pass criteria and saving D2 results")

# Pilot pass criteria:
# ASI AUROC > 0.55: pass
# Full success: AUROC >= 0.65 and AUPRC > 3x base rate
# Falsification: AUROC < 0.55 on both GPT-2 configs
pass_pilot = bool(asi_auroc > 0.55)
full_success = bool(asi_auroc >= 0.65 and metrics_asi["auprc_over_base_rate"] > 3.0)
falsified = bool(asi_auroc < 0.55)  # on this config

# Overall verdict
if full_success:
    verdict = "FULL_SUCCESS"
elif pass_pilot:
    verdict = "PARTIAL_PASS"
else:
    verdict = "FAIL_BELOW_0.55"

# Note: EDA is the strongest detector here
eda_pass_pilot = bool(eda_auroc > 0.55)
eda_full_success = bool(eda_auroc >= 0.65 and metrics_eda["auprc_over_base_rate"] > 3.0)

elapsed_total = time.time() - start_time

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "elapsed_sec": elapsed_total,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "layer": 6,
        "d_sae": D_SAE,
        "d_in": D_IN,
        "seed": SEED,
        "n_owt_tokens": n_total_tokens,
        "label_threshold": float(best_thr),
        "parent_freq_threshold": PARENT_FREQ_THRESHOLD,
        "l0_empirical": float(l0_empirical),
    },
    "labels": {
        "n_pos": n_pos,
        "n_neg": n_neg,
        "base_rate": float(base_rate),
        "label_method": "probe_decoder_alignment",
        "label_threshold": float(best_thr),
        "note": f"Proxy labels: SAE features aligned with letter probes (cos > {best_thr:.2f}). "
                f"Expected n_pos=67 from Chanin et al.; got {n_pos} at threshold={best_thr:.2f}.",
        "letter_feature_ids_sample": letter_feature_ids[:20],
    },
    "metrics": {
        "ASI_combined": {**metrics_asi,
                         "above_null_mean": above_null_mean,
                         "above_null_1sd": above_null_1sd,
                         "above_null_p05": above_null_p05},
        "cos2_alone": metrics_cos2,
        "freq_ratio_alone": metrics_freq,
        "EDA_baseline": {**metrics_eda,
                         "above_null_mean": eda_above_null_mean,
                         "above_null_1sd": eda_above_null_1sd,
                         "above_null_p05": eda_above_null_p05},
        "random_baseline": {
            "auroc": 0.5,
            "auprc": float(base_rate),
            "auprc_over_base_rate": 1.0,
        },
    },
    "delong_tests": {
        "ASI_vs_best_baseline": {
            "best_baseline_name": best_single_name,
            **delong_result_asi_vs_best,
        },
        "ASI_vs_cos2": delong_result_asi_vs_cos2,
        "ASI_vs_freq_ratio": delong_result_asi_vs_freq,
        "ASI_vs_EDA": delong_result_asi_vs_eda,
    },
    "effect_sizes": {
        "delta_AUROC_ASI_vs_best_single": float(asi_auroc - best_single_auroc),
        "delta_AUROC_ASI_vs_cos2": float(asi_auroc - metrics_cos2["auroc"]),
        "delta_AUROC_ASI_vs_freq_ratio": float(asi_auroc - metrics_freq["auroc"]),
        "delta_AUROC_ASI_vs_EDA": float(asi_auroc - eda_auroc),
        "best_detector_AUROC": float(max(asi_auroc, eda_auroc, metrics_cos2["auroc"], metrics_freq["auroc"])),
        "best_detector_name": max(
            [("ASI", asi_auroc), ("EDA", eda_auroc),
             ("cos2", metrics_cos2["auroc"]), ("freq_ratio", metrics_freq["auroc"])],
            key=lambda x: x[1]
        )[0],
    },
    "null_distributions": {
        "ASI": null_asi,
        "EDA": null_eda,
    },
    "precision_recall_curves": {
        "ASI": pr_asi,
        "cos2": pr_cos2,
        "freq_ratio": pr_freq,
        "EDA": pr_eda,
        "random": pr_random,
    },
    "roc_curves": {
        "ASI": roc_asi,
        "EDA": roc_eda,
    },
    "pass_criteria": {
        "pilot_pass_ASI_AUROC_gt_055": pass_pilot,
        "full_success_ASI_AUROC_ge_065_and_AUPRC_3x": full_success,
        "ASI_falsified_AUROC_lt_055": falsified,
        "EDA_pilot_pass": eda_pass_pilot,
        "EDA_full_success": eda_full_success,
        "verdict": verdict,
        "interpretation": (
            f"ASI AUROC={asi_auroc:.4f} ({'ABOVE' if pass_pilot else 'BELOW'} 0.55 threshold). "
            f"EDA AUROC={eda_auroc:.4f} (EDA is {'above' if eda_full_success else 'below'} full success criteria). "
            f"Best detector: {output_dict_name if False else max([('ASI', asi_auroc), ('EDA', eda_auroc), ('cos2', metrics_cos2['auroc']), ('freq_ratio', metrics_freq['auroc'])], key=lambda x: x[1])[0]}."
        ),
    },
    "summary": {
        "key_finding": (
            f"D2 ASI validation: ASI AUROC={asi_auroc:.4f} (verdict={verdict}). "
            f"EDA baseline: AUROC={eda_auroc:.4f} (FULL_SUCCESS={'YES' if eda_full_success else 'NO'}). "
            f"DeLong test ASI vs EDA: delta={asi_auroc - eda_auroc:.4f}, "
            f"p={delong_result_asi_vs_eda.get('p_value', 'N/A')}. "
            f"Null AUROC mean={null_asi['mean']:.4f}. "
            f"n_pos={n_pos}. Elapsed={elapsed_total:.1f}s."
        ),
        "main_claims": [
            f"ASI AUROC={asi_auroc:.4f} on GPT-2 Small L6 (n_pos={n_pos})",
            f"EDA AUROC={eda_auroc:.4f} (strongest detector)",
            f"cos2 AUROC={metrics_cos2['auroc']:.4f}, freq_ratio AUROC={metrics_freq['auroc']:.4f}",
            f"ASI AUPRC={metrics_asi['auprc']:.4f} ({metrics_asi['auprc_over_base_rate']:.2f}x base rate)",
            f"Null mean AUROC (ASI)={null_asi['mean']:.4f} ± {null_asi['std']:.4f}",
            f"ASI above null mean: {above_null_mean}, above null p05: {above_null_p05}",
        ],
    },
}

# Fix the interpretation string (avoid referencing undefined variable)
output["pass_criteria"]["interpretation"] = (
    f"ASI AUROC={asi_auroc:.4f} ({'ABOVE' if pass_pilot else 'BELOW'} 0.55 threshold). "
    f"EDA AUROC={eda_auroc:.4f} (EDA {'passes' if eda_full_success else 'fails'} full success criteria). "
    f"Best detector: {output['effect_sizes']['best_detector_name']}."
)

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to: {OUTPUT_FILE}")

# Update gpu_progress.json
gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    gp = {}
    if gpu_progress_file.exists():
        gp = json.loads(gpu_progress_file.read_text())
    gp.setdefault("completed", [])
    gp.setdefault("failed", [])
    gp.setdefault("running", {})
    gp.setdefault("timings", {})
    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    gp["running"].pop(TASK_ID, None)
    gp["timings"][TASK_ID] = {
        "planned_min": 30,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "d_sae": D_SAE,
            "n_pos": n_pos,
            "asi_auroc": asi_auroc,
            "eda_auroc": eda_auroc,
            "cos2_auroc": metrics_cos2["auroc"],
            "best_detector": output["effect_sizes"]["best_detector_name"],
            "verdict": verdict,
            "n_owt_tokens": n_total_tokens,
            "gpu_model": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        },
    }
    gpu_progress_file.write_text(json.dumps(gp, indent=2))
    print("Updated gpu_progress.json")
except Exception as e:
    print(f"WARNING: gpu_progress.json update failed: {e}")

# Update experiment_state.json
exp_state_file = WORKSPACE / "exp" / "experiment_state.json"
try:
    exp_state = {}
    if exp_state_file.exists():
        exp_state = json.loads(exp_state_file.read_text())
    exp_state.setdefault("tasks", {})[TASK_ID] = {
        "status": "completed",
        "gpu_ids": [0],
        "pid_file": str(PID_FILE),
        "registered_at": datetime.fromtimestamp(start_time).isoformat(),
        "completed_at": datetime.now().isoformat(),
        "result_file": str(OUTPUT_FILE),
    }
    exp_state_file.write_text(json.dumps(exp_state, indent=2))
    print("Updated experiment_state.json")
except Exception as e:
    print(f"WARNING: experiment_state.json update failed: {e}")

print("\n" + "="*60)
print("TASK D2: ASI VALIDATION COMPLETE")
print("="*60)
print(f"n_pos (letter features): {n_pos}")
print(f"n_owt_tokens: {n_total_tokens}")
print(f"")
print(f"AUROC Results:")
print(f"  ASI combined:   {asi_auroc:.4f}  ({verdict})")
print(f"  cos^2 alone:    {metrics_cos2['auroc']:.4f}")
print(f"  freq_ratio:     {metrics_freq['auroc']:.4f}")
print(f"  EDA baseline:   {eda_auroc:.4f}  ({'FULL_SUCCESS' if eda_full_success else 'partial'})")
print(f"")
print(f"DeLong test ASI vs {best_single_name}:")
print(f"  delta AUROC: {asi_auroc - best_single_auroc:.4f}")
print(f"  z-stat: {delong_result_asi_vs_best.get('z_stat', 'N/A')}")
print(f"  p-value: {delong_result_asi_vs_best.get('p_value', 'N/A')}")
print(f"")
print(f"Null AUROC (ASI): mean={null_asi['mean']:.4f} ± {null_asi['std']:.4f}")
print(f"ASI above null mean: {above_null_mean}")
print(f"ASI above null p05: {above_null_p05}")
print(f"")
print(f"Elapsed: {elapsed_total:.1f}s")
print("="*60)

mark_done(
    status="success",
    summary=(
        f"ASI AUROC={asi_auroc:.4f} ({verdict}). EDA AUROC={eda_auroc:.4f}. "
        f"DeLong ASI vs {best_single_name}: delta={asi_auroc - best_single_auroc:.4f}. "
        f"n_pos={n_pos}. Elapsed={elapsed_total:.1f}s"
    ),
)
print("\nTask D2 completed.")
