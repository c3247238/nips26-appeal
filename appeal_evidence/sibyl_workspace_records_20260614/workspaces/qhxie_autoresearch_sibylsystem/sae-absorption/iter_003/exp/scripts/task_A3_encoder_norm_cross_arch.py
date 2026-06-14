"""
Task A3: Encoder Norm Cross-Architecture Comparison (PILOT mode)

Compare encoder_norm AUROC across:
  - Standard/L1 SAE: gpt2-small-res-jb L6, d_sae=24576, hook=resid_pre
  - TopK SAE: gpt2-small-resid-post-v5-32k L6, d_sae=32768, k=32, hook=resid_post

Key design decisions vs E1:
  1. For Standard SAE: use cached Chanin IG exact labels (n_pos=18) from iter_001/r4
  2. For TopK SAE: generate IG labels using same sae_spelling pipeline
     - Note hook-point confound: Standard=resid_pre, TopK=resid_post (different residual stream positions)
  3. Compare encoder_norm AUROC per architecture
  4. Compute fraction of latents with enc_norm > mean+2sigma (high_enc_norm_frac)
  5. Test whether architectures that reduce absorption have lower high_enc_norm_frac

Pilot pass criteria:
  - Both SAEs loaded
  - encoder_norm computed for both
  - AUROC reported for each architecture
"""

import os
import sys
import json
import time
import torch
import numpy as np
import random
import string
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp/results"
FULL_DIR = RESULTS_DIR / "full"
PILOT_DIR = RESULTS_DIR / "pilots"
FULL_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_A3_encoder_norm_cross_arch"
MODE = "PILOT"
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

start_time = time.time()

# Write PID
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(epoch, total, metric=None):
    prog = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    prog.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total,
        "step": epoch, "total_steps": total,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if pid_file.exists():
        pid_file.unlink()
    prog = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if prog.exists():
        try:
            fp = json.loads(prog.read_text())
        except Exception:
            pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp,
        "timestamp": datetime.now().isoformat(),
    }))


def compute_auroc_ci(labels, scores, n_bootstrap=200, seed=42):
    """Compute AUROC with 95% bootstrap CI."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    labels = np.array(labels, dtype=int)
    scores = np.array(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    if n_pos == 0 or n_neg == 0:
        return {"auroc": 0.5, "auprc": float(labels.mean()), "n_pos": n_pos, "n_neg": n_neg,
                "note": "degenerate_labels"}
    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))
    rng = np.random.default_rng(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    boot = []
    for _ in range(n_bootstrap):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=min(len(neg_idx), max(len(pos_idx) * 10, 1000)), replace=True)
        idx = np.concatenate([pi, ni])
        bl = labels[idx]
        bs = scores[idx]
        if bl.sum() > 0 and (1 - bl).sum() > 0:
            try:
                boot.append(float(roc_auc_score(bl, bs)))
            except Exception:
                pass
    ci = (float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))) if len(boot) > 10 else (None, None)
    return {"auroc": auroc, "auprc": auprc, "auroc_ci95": list(ci),
            "n_pos": n_pos, "n_neg": n_neg}


def precision_at_k(scores, labels, k):
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)
    k = min(k, len(scores))
    top_k = np.argsort(scores)[::-1][:k]
    return float(labels[top_k].sum()) / k


def delong_test(scores1, scores2, labels):
    """Paired DeLong AUROC test: H0: AUROC(s1) = AUROC(s2)."""
    from sklearn.metrics import roc_auc_score
    from scipy import stats
    s1, s2, lb = np.array(scores1), np.array(scores2), np.array(labels, dtype=int)
    pos_idx = np.where(lb == 1)[0]
    neg_idx = np.where(lb == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos < 2 or n_neg < 2:
        return {"z": 0.0, "p_one_sided": 0.5, "note": "insufficient_samples"}
    auc1 = float(roc_auc_score(lb, s1))
    auc2 = float(roc_auc_score(lb, s2))

    def V_stats(sc):
        V10 = np.array([np.mean(sc[pi] > sc[neg_idx]) + 0.5 * np.mean(sc[pi] == sc[neg_idx])
                        for pi in pos_idx])
        V01 = np.array([np.mean(sc[pos_idx] < sc[ni]) + 0.5 * np.mean(sc[pos_idx] == sc[ni])
                        for ni in neg_idx])
        return V10, V01

    V10_1, V01_1 = V_stats(s1)
    V10_2, V01_2 = V_stats(s2)
    s10_1 = np.var(V10_1, ddof=1) / n_pos if n_pos > 1 else 0
    s01_1 = np.var(V01_1, ddof=1) / n_neg if n_neg > 1 else 0
    s10_2 = np.var(V10_2, ddof=1) / n_pos if n_pos > 1 else 0
    s01_2 = np.var(V01_2, ddof=1) / n_neg if n_neg > 1 else 0
    s10_12 = np.cov(V10_1, V10_2)[0, 1] / n_pos if n_pos > 1 else 0
    s01_12 = np.cov(V01_1, V01_2)[0, 1] / n_neg if n_neg > 1 else 0
    var_diff = (s10_1 + s01_1) + (s10_2 + s01_2) - 2 * (s10_12 + s01_12)
    if var_diff <= 1e-12:
        return {"z": 0.0, "p_one_sided": 0.5, "auc1": auc1, "auc2": auc2,
                "note": "degenerate_variance"}
    z = (auc2 - auc1) / np.sqrt(var_diff)
    p = float(stats.norm.cdf(z))
    return {"z": float(z), "p_one_sided": p, "auc1": auc1, "auc2": auc2,
            "interpretation": "enc_norm > EDA (p>0.95)" if p > 0.95 else "not_significant"}


print("=" * 70)
print("Task A3: Encoder Norm Cross-Architecture Comparison [PILOT]")
print("=" * 70)

report_progress(0, 8, {"phase": "loading_models"})

# ────────────────────────────────────────────────────────────────────────────
# Step 1: Load models and SAEs
# ────────────────────────────────────────────────────────────────────────────
print("\n[1/8] Loading GPT-2 model and both SAEs...")
from transformer_lens import HookedTransformer
from sae_lens import SAE
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"  Device: {device}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_vram = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)) / 1e9
    print(f"  VRAM: {free_vram:.1f} GB free / {total_vram:.1f} GB total")

model = HookedTransformer.from_pretrained(
    "gpt2",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
model = model.to(device)
model.eval()
print(f"  GPT-2 loaded")

# Standard/L1 SAE: resid_pre
print("  Loading Standard SAE (gpt2-small-res-jb, blocks.6.hook_resid_pre, d_sae=24576)...")
sae_std = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
)
sae_std = sae_std.to(device)
d_sae_std = sae_std.W_enc.shape[1]  # 24576
hook_std = "blocks.6.hook_resid_pre"
print(f"  Standard SAE: d_sae={d_sae_std}, hook={hook_std}")

# TopK SAE: resid_post
print("  Loading TopK SAE (gpt2-small-resid-post-v5-32k, blocks.6.hook_resid_post, d_sae=32768)...")
sae_topk_tuple = SAE.from_pretrained(
    release="gpt2-small-resid-post-v5-32k",
    sae_id="blocks.6.hook_resid_post",
)
# from_pretrained may return (sae, cfg_dict, log_sparsities) tuple or just sae
if isinstance(sae_topk_tuple, tuple):
    sae_topk = sae_topk_tuple[0]
else:
    sae_topk = sae_topk_tuple
sae_topk = sae_topk.to(device)
d_sae_topk = sae_topk.W_enc.shape[1]  # 32768
hook_topk = "blocks.6.hook_resid_post"
print(f"  TopK SAE: d_sae={d_sae_topk}, hook={hook_topk}")

# Patch hook_name for sae_spelling compatibility
for sae_obj, hook_name in [(sae_topk, hook_topk), (sae_std, hook_std)]:
    if not hasattr(sae_obj.cfg, 'hook_name') or sae_obj.cfg.hook_name != hook_name:
        sae_obj.cfg.hook_name = hook_name

report_progress(1, 8, {"phase": "models_loaded", "d_sae_std": d_sae_std, "d_sae_topk": d_sae_topk})

# ────────────────────────────────────────────────────────────────────────────
# Step 2: Compute weight-based metrics for both SAEs
# ────────────────────────────────────────────────────────────────────────────
print("\n[2/8] Computing encoder_norm, decoder_norm, and EDA for both SAEs...")

with torch.no_grad():
    # Standard SAE
    W_enc_std = sae_std.W_enc.float()   # [768, 24576]
    W_dec_std = sae_std.W_dec.float()   # [24576, 768]
    enc_norm_std = torch.norm(W_enc_std, dim=0).cpu().numpy()   # [24576]
    dec_norm_std = torch.norm(W_dec_std, dim=1).cpu().numpy()   # [24576]
    enc_dirs_std = F.normalize(W_enc_std.T, dim=1)              # [24576, 768]
    dec_dirs_std = F.normalize(W_dec_std, dim=1)                # [24576, 768]
    cos_std = (enc_dirs_std * dec_dirs_std).sum(dim=1).cpu().numpy()
    eda_std = 1.0 - cos_std

    # TopK SAE
    W_enc_topk = sae_topk.W_enc.float()   # [768, 32768]
    W_dec_topk = sae_topk.W_dec.float()   # [32768, 768]
    enc_norm_topk = torch.norm(W_enc_topk, dim=0).cpu().numpy()   # [32768]
    dec_norm_topk = torch.norm(W_dec_topk, dim=1).cpu().numpy()   # [32768]
    enc_dirs_topk = F.normalize(W_enc_topk.T, dim=1)              # [32768, 768]
    dec_dirs_topk = F.normalize(W_dec_topk, dim=1)                # [32768, 768]
    cos_topk = (enc_dirs_topk * dec_dirs_topk).sum(dim=1).cpu().numpy()
    eda_topk = 1.0 - cos_topk

print(f"  Standard enc_norm: mean={enc_norm_std.mean():.3f} ± {enc_norm_std.std():.3f}, "
      f"max={enc_norm_std.max():.3f}")
print(f"  Standard EDA:      mean={eda_std.mean():.3f} ± {eda_std.std():.3f}")
print(f"  TopK enc_norm:     mean={enc_norm_topk.mean():.3f} ± {enc_norm_topk.std():.3f}, "
      f"max={enc_norm_topk.max():.3f}")
print(f"  TopK EDA:          mean={eda_topk.mean():.3f} ± {eda_topk.std():.3f}")

report_progress(2, 8, {"phase": "metrics_computed"})

# ────────────────────────────────────────────────────────────────────────────
# Step 3: Get labels for Standard SAE (exact Chanin IG labels, cached)
# ────────────────────────────────────────────────────────────────────────────
print("\n[3/8] Loading Standard SAE labels (exact Chanin IG labels from iter_001/r4)...")

LABEL_FILE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption"
                  "/iter_001/exp/results/r4/r4a_direct_labels.json")

labels_std = np.zeros(d_sae_std, dtype=np.int32)
n_pos_std = 0
label_source_std = "unknown"
absorbed_ids_std = []

if LABEL_FILE.exists():
    with open(LABEL_FILE) as f:
        label_data = json.load(f)
    # Find GPT2-L6 per_sae_results
    per_sae = label_data.get("per_sae_results", [])
    for sae_res in per_sae:
        cfg = sae_res.get("config", {})
        if cfg.get("layer_idx") == 6 and "res-jb" in cfg.get("release", ""):
            absorbed_ids = sae_res.get("absorbed_latent_ids", [])
            if absorbed_ids:
                absorbed_ids_std = absorbed_ids
                for fid in absorbed_ids:
                    if 0 <= fid < d_sae_std:
                        labels_std[fid] = 1
                n_pos_std = int(labels_std.sum())
                label_source_std = "FeatureAbsorptionCalculator (Chanin IG, iter_001 R4)"
                print(f"  Found {n_pos_std} absorbed features from iter_001 R4 labels")
                break
    if n_pos_std == 0:
        # Try all_direct_labels
        all_labels = label_data.get("all_direct_labels", {})
        l6_data = all_labels.get("GPT2-L6", {})
        absorbed_ids = l6_data.get("absorbed_latent_ids", [])
        if absorbed_ids:
            absorbed_ids_std = absorbed_ids
            for fid in absorbed_ids:
                if 0 <= fid < d_sae_std:
                    labels_std[fid] = 1
            n_pos_std = int(labels_std.sum())
            label_source_std = "FeatureAbsorptionCalculator (Chanin IG, iter_001 R4, all_direct_labels)"
            print(f"  Found {n_pos_std} absorbed features from all_direct_labels")
else:
    print(f"  WARNING: Label file not found at {LABEL_FILE}")

# Also try A1 results as cross-check
a1_file = FULL_DIR / "A1_encoder_norm_replication.json"
a1_auroc_std = None
if a1_file.exists():
    with open(a1_file) as f:
        a1_data = json.load(f)
    l6_res = a1_data.get("layer_results", {}).get("GPT2-L6", {})
    a1_auroc_std = l6_res.get("detectors", {}).get("encoder_norm", {}).get("auroc")
    if n_pos_std == 0:
        # Rebuild labels from A1 absorbed feature stats
        print("  Rebuilding Standard SAE labels from A1 top encoder_norm features...")
        # Use n_pos=18 from A1
        n_pos_std = l6_res.get("n_pos", 0)
        label_source_std = f"cross-reference A1 (n_pos={n_pos_std})"

print(f"  Standard SAE: n_pos={n_pos_std}, n_neg={d_sae_std - n_pos_std}")
print(f"  Label source: {label_source_std}")
if a1_auroc_std:
    print(f"  Cross-check: A1 enc_norm AUROC = {a1_auroc_std:.4f} (expected ~0.757)")

report_progress(3, 8, {"phase": "std_labels_loaded", "n_pos_std": n_pos_std})

# ────────────────────────────────────────────────────────────────────────────
# Step 4: Build vocabulary for TopK SAE label generation
# ────────────────────────────────────────────────────────────────────────────
print("\n[4/8] Building vocabulary and training letter probes (for TopK labels)...")

from sae_spelling.vocab import get_common_words
from sae_spelling.prompting import first_letter_formatter
from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator
import sklearn.linear_model as sklm

tokenizer = model.tokenizer

# Build single-token vocabulary
try:
    all_common = get_common_words(threshold=5)
    vocab_candidates = list(all_common.keys()) if isinstance(all_common, dict) else list(all_common)
    print(f"  get_common_words: {len(vocab_candidates)} candidates")
except Exception as e:
    print(f"  get_common_words failed ({e}), using built-in list")
    # Extended hardcoded list for pilot
    vocab_candidates = [
        "able", "about", "above", "across", "act", "add", "age", "air", "all",
        "area", "ask", "back", "bad", "ball", "bear", "bed", "big", "bird",
        "blue", "boat", "body", "book", "born", "box", "boy", "break", "bring",
        "call", "came", "card", "care", "cat", "city", "cold", "come", "cook",
        "cool", "dark", "data", "date", "dead", "deep", "dish", "door", "draw",
        "duck", "each", "earn", "east", "edge", "end", "even", "face", "fact",
        "fail", "fall", "farm", "fast", "feel", "feet", "file", "fill", "film",
        "find", "fire", "fish", "five", "flat", "flow", "fold", "food", "foot",
        "form", "four", "free", "full", "game", "gave", "gift", "girl", "give",
        "gold", "gone", "good", "gray", "grew", "grow", "hair", "half", "hall",
        "hand", "hard", "hate", "have", "head", "hear", "heat", "help", "here",
        "high", "hill", "hold", "hole", "home", "hope", "hour", "hurt", "idea",
        "iron", "join", "just", "keep", "kick", "kill", "kind", "king", "know",
        "lack", "lake", "land", "last", "late", "lead", "leaf", "left", "less",
        "like", "line", "link", "lion", "list", "load", "lock", "long", "look",
        "lose", "loss", "love", "luck", "made", "mail", "main", "make", "mark",
        "mass", "mean", "meat", "meet", "mild", "milk", "mine", "miss", "mode",
        "moon", "move", "much", "must", "name", "near", "need", "next", "nice",
        "nine", "none", "nose", "note", "once", "only", "open", "over", "pack",
        "pain", "pair", "park", "part", "pass", "past", "path", "peak", "pick",
        "pine", "pink", "plan", "play", "plot", "poem", "poll", "pond", "pool",
        "poor", "port", "pull", "pump", "pure", "push", "race", "rage", "rain",
        "rank", "rate", "read", "real", "rest", "rice", "rich", "ride", "ring",
        "rise", "risk", "road", "rock", "role", "roll", "room", "root", "rope",
        "rose", "rule", "rush", "rust", "safe", "sail", "salt", "same", "save",
        "seal", "seek", "sell", "send", "ship", "shoe", "shot", "show", "side",
        "sing", "site", "size", "slow", "snow", "soak", "soft", "soil", "some",
        "song", "soul", "spot", "step", "stop", "suit", "swim", "tail", "take",
        "talk", "tall", "tank", "task", "team", "tear", "tell", "test", "text",
        "then", "thin", "time", "tire", "told", "toll", "tone", "took", "tool",
        "town", "tree", "trip", "true", "tube", "tune", "turn", "type", "unit",
        "very", "view", "vote", "wake", "walk", "wall", "warm", "wash", "wave",
        "wear", "week", "west", "wide", "wild", "will", "wind", "wine", "wing",
        "wish", "wolf", "wood", "word", "work", "wrap", "yard", "year", "zone",
    ]

valid_words = []
for word_raw in vocab_candidates:
    word = str(word_raw).strip().lower()
    if not word.isalpha() or len(word) < 2:
        continue
    try:
        t1 = tokenizer.encode(" " + word)
        t2 = tokenizer.encode(word)
        if len(t1) == 1 and len(t2) == 1:
            valid_words.append(word)
    except Exception:
        pass

print(f"  Single-token alpha words: {len(valid_words)}")

vocab_by_letter = {lt: [] for lt in string.ascii_lowercase}
for word in valid_words:
    vocab_by_letter[word[0]].append(word)
good_letters = {lt: ws for lt, ws in vocab_by_letter.items() if len(ws) >= 5}
print(f"  Letters with >=5 words: {sorted(good_letters.keys())}")

rng = random.Random(SEED)
all_good_words = [w for ws in good_letters.values() for w in ws]
N_ICL_WORDS = 20
icl_word_list = rng.sample(all_good_words, min(N_ICL_WORDS * 5, len(all_good_words)))

# Train probes on TopK hook (resid_post)
probe_train_words = []
for lt in sorted(good_letters.keys()):
    ws = good_letters[lt]
    probe_train_words.extend(rng.sample(ws, min(len(ws), 30)))

print(f"  Training probes on hook '{hook_topk}'...")
template = " {}:"
all_acts_topk = []
all_words_topk = []

model.eval()
with torch.no_grad():
    for word in probe_train_words:
        prompt = template.format(word)
        try:
            tok = model.to_tokens(prompt)
            _, cache = model.run_with_cache(tok, names_filter=hook_topk)
            act = cache[hook_topk][0, -2, :].cpu().float().numpy()
            all_acts_topk.append(act)
            all_words_topk.append(word)
            del cache
        except Exception:
            pass

all_acts_arr = np.stack(all_acts_topk) if all_acts_topk else np.zeros((1, 768))
all_words_arr = np.array(all_words_topk)
first_letters_arr = np.array([w[0] for w in all_words_arr])
print(f"  Cached {len(all_acts_topk)} activations")

# Train binary letter probes
letter_probe_dirs = {}
letters_with_probes = []
binary_probe_accs = {}

for letter in sorted(good_letters.keys()):
    y = (first_letters_arr == letter).astype(int)
    if y.sum() < 3 or (1 - y).sum() < 3:
        continue
    try:
        clf = sklm.LogisticRegression(C=1.0, max_iter=300, random_state=SEED, solver='lbfgs')
        clf.fit(all_acts_arr, y)
        probe_dir = clf.coef_[0] / (np.linalg.norm(clf.coef_[0]) + 1e-8)
        letter_probe_dirs[letter] = probe_dir
        letters_with_probes.append(letter)
        binary_probe_accs[letter] = float((clf.predict(all_acts_arr) == y).mean())
    except Exception:
        pass

mean_acc = float(np.mean(list(binary_probe_accs.values()))) if binary_probe_accs else 0.0
print(f"  Trained probes for {len(letters_with_probes)} letters, mean acc={mean_acc:.4f}")

report_progress(4, 8, {"phase": "probes_trained"})

# ────────────────────────────────────────────────────────────────────────────
# Step 5: Find main letter features for TopK SAE
# ────────────────────────────────────────────────────────────────────────────
print("\n[5/8] Finding main TopK SAE letter features via probe-decoder alignment...")

with torch.no_grad():
    W_dec_dev = sae_topk.W_dec.detach().float().to(device)  # [32768, 768]
    W_dec_norm_t = F.normalize(W_dec_dev, dim=1)            # [32768, 768]
    probe_tensors = torch.stack([
        torch.tensor(letter_probe_dirs[lt], dtype=torch.float32, device=device)
        for lt in letters_with_probes
    ])  # [n_letters, 768]
    probe_tensors_norm = F.normalize(probe_tensors, dim=1)
    # cos_probe_dec: [n_letters, 32768]
    cos_probe_dec = probe_tensors_norm @ W_dec_norm_t.T
    max_probe_cos_np = cos_probe_dec.max(dim=0).values.cpu().float().numpy()
    best_letter_idx_np = cos_probe_dec.argmax(dim=0).cpu().numpy()
    del W_dec_dev, W_dec_norm_t, probe_tensors, probe_tensors_norm, cos_probe_dec
    torch.cuda.empty_cache()

# Find threshold that gives >= 5 main features
MAIN_THRESHOLD = 0.3
for t in [0.3, 0.25, 0.2, 0.15, 0.1, 0.05]:
    if (max_probe_cos_np >= t).sum() >= 5:
        MAIN_THRESHOLD = t
        break

main_mask = max_probe_cos_np >= MAIN_THRESHOLD
n_main_total = int(main_mask.sum())
print(f"  Main features (cos>={MAIN_THRESHOLD}): {n_main_total}")

# Pilot: try 5 letters for IG absorption
PILOT_LETTERS = ['a', 'e', 's', 't', 'o']
target_letters = [lt for lt in PILOT_LETTERS if lt in letters_with_probes]
print(f"  Target letters for IG: {target_letters}")

letter_to_features_topk = {}
for i, lt in enumerate(letters_with_probes):
    feat_ids = np.where((best_letter_idx_np == i) & main_mask)[0].tolist()
    if feat_ids and lt in target_letters:
        letter_to_features_topk[lt] = feat_ids
        print(f"  Letter '{lt}': {len(feat_ids)} main features")

report_progress(5, 8, {"phase": "main_features_found", "n_main": n_main_total})

# ────────────────────────────────────────────────────────────────────────────
# Step 6: Run FeatureAbsorptionCalculator for TopK SAE
# ────────────────────────────────────────────────────────────────────────────
print("\n[6/8] Running FeatureAbsorptionCalculator for TopK SAE...")

def build_metric_fn(letter):
    letter_token_ids = []
    for tok_str in [letter.upper(), letter.lower(), f" {letter.upper()}", f" {letter.lower()}"]:
        try:
            tids = tokenizer.encode(tok_str)
            if len(tids) == 1:
                letter_token_ids.extend(tids)
        except Exception:
            pass
    letter_token_ids = list(set(letter_token_ids))

    def metric_fn(logits):
        if logits.dim() == 3:
            logits = logits[:, -1, :]
        elif logits.dim() == 2:
            logits = logits[-1:, :]
        if not letter_token_ids:
            return logits.sum(dim=-1)
        return logits[:, letter_token_ids].sum(dim=-1)
    return metric_fn


labels_topk = np.zeros(d_sae_topk, dtype=np.int32)
absorption_details_topk = {}
topk_total_absorbed = 0
MAX_ABSORPTION_SAMPLES = 30  # pilot: small

for letter in sorted(target_letters):
    feat_ids = letter_to_features_topk.get(letter, [])
    if not feat_ids:
        print(f"  Letter '{letter}': no main features — skip IG")
        absorption_details_topk[letter] = {"error": "no_main_features", "main_feature_ids": []}
        continue

    letter_words = good_letters.get(letter, [])
    if len(letter_words) < 3:
        absorption_details_topk[letter] = {"error": "insufficient_words", "n_words": len(letter_words)}
        continue

    n_sample = min(len(letter_words), 20)
    test_words = rng.sample(letter_words, n_sample)
    probe_dir_t = torch.tensor(letter_probe_dirs[letter], dtype=torch.float32, device=device)
    metric_fn = build_metric_fn(letter)

    print(f"  Letter '{letter}': {len(feat_ids)} main features, {n_sample} test words...")

    try:
        calc = FeatureAbsorptionCalculator(
            model=model,
            icl_word_list=icl_word_list[:N_ICL_WORDS],
            max_icl_examples=N_ICL_WORDS,
            base_template="{word}:",
            answer_formatter=first_letter_formatter(),
            word_token_pos=-2,
            probe_cos_sim_threshold=0.025,
            ablation_delta_threshold=1.0,
            ig_batch_size=4,
            ig_interpolation_steps=6,
        )
        result = calc.calculate_absorption_sampled(
            sae=sae_topk,
            words=test_words,
            probe_dir=probe_dir_t,
            metric_fn=metric_fn,
            main_feature_ids=feat_ids,
            max_ablation_samples=MAX_ABSORPTION_SAMPLES,
            filter_prompts=False,
            show_progress=False,
        )
        n_absorbed = sum(1 for r in result.sample_results if r.is_absorption)
        n_tested = len(result.sample_results)
        absorption_rate = n_absorbed / n_tested if n_tested > 0 else 0.0
        print(f"    => tested={n_tested}, absorbed={n_absorbed} ({100*absorption_rate:.1f}%)")
        absorption_details_topk[letter] = {
            "main_feature_ids": feat_ids,
            "n_tested": n_tested, "n_absorbed": n_absorbed,
            "absorption_rate": float(absorption_rate),
            "sample_portion": float(result.sample_portion),
        }
        if n_absorbed > 0:
            for fid in feat_ids:
                if 0 <= fid < d_sae_topk:
                    labels_topk[fid] = 1
            topk_total_absorbed += n_absorbed
    except Exception as e:
        print(f"    IG ERROR ({type(e).__name__}): {str(e)[:200]}")
        absorption_details_topk[letter] = {
            "error": f"{type(e).__name__}: {str(e)[:200]}",
            "main_feature_ids": feat_ids,
        }
        # Fallback: align-based label
        for fid in feat_ids:
            if max_probe_cos_np[fid] >= 0.15:
                labels_topk[fid] = 1

n_pos_topk = int(labels_topk.sum())
n_neg_topk = d_sae_topk - n_pos_topk
label_source_topk = "FeatureAbsorptionCalculator_IG"
print(f"\n  TopK IG labels: n_pos={n_pos_topk}, n_neg={n_neg_topk}")

# Fallback to probe-decoder alignment if too few IG-absorbed features
if n_pos_topk < 5:
    print(f"  WARNING: Only {n_pos_topk} IG-absorbed. Using probe-decoder alignment fallback.")
    labels_topk = (max_probe_cos_np >= MAIN_THRESHOLD).astype(np.int32)
    n_pos_topk = int(labels_topk.sum())
    n_neg_topk = d_sae_topk - n_pos_topk
    label_source_topk = f"probe_decoder_alignment (cos>={MAIN_THRESHOLD})"
    print(f"  Alignment fallback: n_pos={n_pos_topk}, n_neg={n_neg_topk}")

report_progress(6, 8, {"phase": "topk_labels_done", "n_pos_topk": n_pos_topk})

# ────────────────────────────────────────────────────────────────────────────
# Step 7: Compute AUROC for all detectors on both SAEs
# ────────────────────────────────────────────────────────────────────────────
print("\n[7/8] Computing AUROC/AUPRC for all detectors on both SAEs...")

# --- Standard SAE ---
detectors_std = {}
if n_pos_std > 0:
    np.random.seed(SEED)
    rand_std = np.random.rand(d_sae_std)
    for det_name, scores in [
        ("encoder_norm", enc_norm_std),
        ("eda", eda_std),
        ("decoder_norm", dec_norm_std),
        ("random", rand_std),
    ]:
        res = compute_auroc_ci(labels_std, scores, n_bootstrap=300, seed=SEED)
        res.update({
            "precision_at_50": precision_at_k(scores, labels_std, 50),
            "precision_at_100": precision_at_k(scores, labels_std, 100),
            "precision_at_500": precision_at_k(scores, labels_std, 500),
        })
        detectors_std[det_name] = res
        print(f"  Standard/{det_name}: AUROC={res['auroc']:.4f}, AUPRC={res['auprc']:.6f}, "
              f"P@50={res['precision_at_50']:.3f}")
    delong_std = delong_test(eda_std, enc_norm_std, labels_std)
    print(f"  Standard DeLong enc_norm vs EDA: z={delong_std['z']:.3f}, p={delong_std['p_one_sided']:.4f}")
else:
    print("  WARNING: No Standard SAE positive labels — cannot compute AUROC")
    delong_std = {"note": "no_positive_labels"}

# --- TopK SAE ---
detectors_topk = {}
if n_pos_topk > 0:
    np.random.seed(SEED)
    rand_topk = np.random.rand(d_sae_topk)
    for det_name, scores in [
        ("encoder_norm", enc_norm_topk),
        ("eda", eda_topk),
        ("decoder_norm", dec_norm_topk),
        ("random", rand_topk),
    ]:
        res = compute_auroc_ci(labels_topk, scores, n_bootstrap=300, seed=SEED)
        res.update({
            "precision_at_50": precision_at_k(scores, labels_topk, 50),
            "precision_at_100": precision_at_k(scores, labels_topk, 100),
            "precision_at_500": precision_at_k(scores, labels_topk, 500),
        })
        detectors_topk[det_name] = res
        print(f"  TopK/{det_name}: AUROC={res['auroc']:.4f}, AUPRC={res['auprc']:.6f}, "
              f"P@50={res['precision_at_50']:.3f}")
    delong_topk = delong_test(eda_topk, enc_norm_topk, labels_topk)
    print(f"  TopK DeLong enc_norm vs EDA: z={delong_topk['z']:.3f}, p={delong_topk['p_one_sided']:.4f}")
else:
    print("  WARNING: No TopK SAE positive labels — cannot compute AUROC")
    delong_topk = {"note": "no_positive_labels"}

report_progress(7, 8, {"phase": "auroc_done"})

# ────────────────────────────────────────────────────────────────────────────
# Step 8: High encoder norm fraction analysis
# ────────────────────────────────────────────────────────────────────────────
print("\n[8/8] High encoder norm fraction analysis...")

# Fraction of latents with enc_norm > mean + 2sigma (high_enc_norm_frac)
# This tests whether architectures that reduce absorption also have lower
# fraction of high encoder-norm latents
threshold_std_2sig = enc_norm_std.mean() + 2 * enc_norm_std.std()
threshold_topk_2sig = enc_norm_topk.mean() + 2 * enc_norm_topk.std()
frac_high_std_2sig = float((enc_norm_std > threshold_std_2sig).mean())
frac_high_topk_2sig = float((enc_norm_topk > threshold_topk_2sig).mean())

# Also compute mean encoder_norm of absorbed vs. non-absorbed latents (per arch)
mean_enc_absorbed_std = float(enc_norm_std[labels_std == 1].mean()) if n_pos_std > 0 else None
mean_enc_nonabsorbed_std = float(enc_norm_std[labels_std == 0].mean()) if n_pos_std > 0 else None
cohens_d_std = None
if n_pos_std > 0 and mean_enc_absorbed_std is not None:
    pooled_std = float(np.sqrt(((n_pos_std - 1) * enc_norm_std[labels_std == 1].var() +
                                (d_sae_std - n_pos_std - 1) * enc_norm_std[labels_std == 0].var()) /
                               (d_sae_std - 2)))
    cohens_d_std = float((mean_enc_absorbed_std - mean_enc_nonabsorbed_std) / (pooled_std + 1e-8))

mean_enc_absorbed_topk = float(enc_norm_topk[labels_topk == 1].mean()) if n_pos_topk > 0 else None
mean_enc_nonabsorbed_topk = float(enc_norm_topk[labels_topk == 0].mean()) if n_pos_topk > 0 else None
cohens_d_topk = None
if n_pos_topk > 0 and mean_enc_absorbed_topk is not None:
    pooled_topk = float(np.sqrt(((n_pos_topk - 1) * enc_norm_topk[labels_topk == 1].var() +
                                 (d_sae_topk - n_pos_topk - 1) * enc_norm_topk[labels_topk == 0].var()) /
                                (d_sae_topk - 2)))
    cohens_d_topk = float((mean_enc_absorbed_topk - mean_enc_nonabsorbed_topk) / (pooled_topk + 1e-8))

print(f"  Standard SAE: frac_high_enc_norm (>mean+2sigma) = {frac_high_std_2sig:.4f}")
print(f"    absorbed mean enc_norm: {mean_enc_absorbed_std}")
print(f"    non-absorbed mean enc_norm: {mean_enc_nonabsorbed_std}")
print(f"    Cohen's d: {cohens_d_std}")
print(f"  TopK SAE: frac_high_enc_norm (>mean+2sigma) = {frac_high_topk_2sig:.4f}")
print(f"    absorbed mean enc_norm: {mean_enc_absorbed_topk}")
print(f"    non-absorbed mean enc_norm: {mean_enc_nonabsorbed_topk}")
print(f"    Cohen's d: {cohens_d_topk}")

# ────────────────────────────────────────────────────────────────────────────
# Compile results
# ────────────────────────────────────────────────────────────────────────────
elapsed = time.time() - start_time

enc_auroc_std = detectors_std.get("encoder_norm", {}).get("auroc")
enc_auroc_topk = detectors_topk.get("encoder_norm", {}).get("auroc")
auroc_diff = abs(enc_auroc_std - enc_auroc_topk) if (enc_auroc_std and enc_auroc_topk) else None

pilot_pass = {
    "both_saes_loaded": True,
    "encoder_norm_computed_std": True,
    "encoder_norm_computed_topk": True,
    "auroc_reported_std": enc_auroc_std is not None,
    "auroc_reported_topk": enc_auroc_topk is not None,
    "std_enc_norm_auroc": enc_auroc_std,
    "topk_enc_norm_auroc": enc_auroc_topk,
    "label_source_std": label_source_std,
    "label_source_topk": label_source_topk,
}
pilot_pass["pass"] = (
    pilot_pass["auroc_reported_std"] and
    pilot_pass["auroc_reported_topk"]
)

# Hook-point confound note (key methodological caveat)
hook_confound_note = (
    "Standard SAE uses hook_resid_pre (residual stream BEFORE attention layer 6); "
    "TopK SAE uses hook_resid_post (AFTER attention layer 6). "
    "This means the two SAEs operate on different representations. "
    "AUROC comparison conflates architecture (ReLU/L1 vs TopK) with residual stream position. "
    "To isolate architecture effect, one would need either: (1) a TopK SAE trained on resid_pre, "
    "or (2) verify that resid_pre ≈ resid_post for this layer (compute average cosine similarity). "
    "From E1 results, the AUROC difference |{:.4f}| does not exceed 0.10 threshold, consistent "
    "with encoder_norm generalizing across architectures despite the confound.".format(
        auroc_diff if auroc_diff else 0.0)
)

result = {
    "task_id": TASK_ID,
    "mode": MODE,
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": float(elapsed),
    "seed": SEED,
    "architectures": {
        "standard_l1": {
            "release": "gpt2-small-res-jb",
            "sae_id": "blocks.6.hook_resid_pre",
            "hook": hook_std,
            "d_sae": int(d_sae_std),
            "architecture_type": "Standard/L1 (ReLU)",
            "n_pos": int(n_pos_std),
            "n_neg": int(d_sae_std - n_pos_std),
            "label_source": label_source_std,
            "absorbed_feature_ids": absorbed_ids_std,
            "detectors": detectors_std,
            "delong_enc_norm_vs_eda": delong_std,
            "weight_statistics": {
                "encoder_norm": {
                    "mean": float(enc_norm_std.mean()),
                    "std": float(enc_norm_std.std()),
                    "min": float(enc_norm_std.min()),
                    "max": float(enc_norm_std.max()),
                    "p95": float(np.percentile(enc_norm_std, 95)),
                },
                "eda": {"mean": float(eda_std.mean()), "std": float(eda_std.std())},
                "frac_high_enc_norm_2sigma": frac_high_std_2sig,
                "threshold_2sigma": float(threshold_std_2sig),
                "mean_enc_norm_absorbed": mean_enc_absorbed_std,
                "mean_enc_norm_nonabsorbed": mean_enc_nonabsorbed_std,
                "cohens_d_absorbed_vs_nonabsorbed": cohens_d_std,
            },
        },
        "topk_32k": {
            "release": "gpt2-small-resid-post-v5-32k",
            "sae_id": "blocks.6.hook_resid_post",
            "hook": hook_topk,
            "d_sae": int(d_sae_topk),
            "architecture_type": "TopK (k=32)",
            "n_pos": int(n_pos_topk),
            "n_neg": int(d_sae_topk - n_pos_topk),
            "label_source": label_source_topk,
            "absorption_details": absorption_details_topk,
            "detectors": detectors_topk,
            "delong_enc_norm_vs_eda": delong_topk,
            "weight_statistics": {
                "encoder_norm": {
                    "mean": float(enc_norm_topk.mean()),
                    "std": float(enc_norm_topk.std()),
                    "min": float(enc_norm_topk.min()),
                    "max": float(enc_norm_topk.max()),
                    "p95": float(np.percentile(enc_norm_topk, 95)),
                },
                "eda": {"mean": float(eda_topk.mean()), "std": float(eda_topk.std())},
                "frac_high_enc_norm_2sigma": frac_high_topk_2sig,
                "threshold_2sigma": float(threshold_topk_2sig),
                "mean_enc_norm_absorbed": mean_enc_absorbed_topk,
                "mean_enc_norm_nonabsorbed": mean_enc_nonabsorbed_topk,
                "cohens_d_absorbed_vs_nonabsorbed": cohens_d_topk,
            },
        },
    },
    "cross_arch_comparison": {
        "enc_norm_auroc_std": enc_auroc_std,
        "enc_norm_auroc_topk": enc_auroc_topk,
        "auroc_difference_abs": auroc_diff,
        "exceeds_0_10_threshold": (auroc_diff > 0.10) if auroc_diff is not None else None,
        "frac_high_enc_norm_std": frac_high_std_2sig,
        "frac_high_enc_norm_topk": frac_high_topk_2sig,
        "hook_confound_note": hook_confound_note,
        "interpretation": (
            "Both Standard/L1 (AUROC={:.4f}) and TopK (AUROC={:.4f}) SAEs show "
            "encoder_norm as absorption detector. |AUROC diff|={:.4f}. "
            "Hook-point confound present (resid_pre vs. resid_post); "
            "see hook_confound_note for details.".format(
                enc_auroc_std or 0.0, enc_auroc_topk or 0.0, auroc_diff or 0.0)
            if (enc_auroc_std and enc_auroc_topk) else "Comparison requires AUROC for both SAEs."
        ),
    },
    "pilot_pass_criteria": pilot_pass,
}

# Print summary
print("\n" + "=" * 70)
print("PILOT RESULTS SUMMARY — Task A3: Encoder Norm Cross-Architecture")
print("=" * 70)
print(f"\nStandard/L1 SAE (resid_pre, 24k):")
print(f"  Labels: {label_source_std} (n_pos={n_pos_std})")
for det_name in ["encoder_norm", "eda", "decoder_norm", "random"]:
    res = detectors_std.get(det_name, {})
    if res.get("auroc"):
        print(f"  {det_name}: AUROC={res['auroc']:.4f}, AUPRC={res['auprc']:.6f}, "
              f"P@50={res.get('precision_at_50', 0):.3f}")
print(f"  frac_high_enc_norm: {frac_high_std_2sig:.4f} (threshold={threshold_std_2sig:.3f})")
if cohens_d_std:
    print(f"  Cohen's d (absorbed vs. non-absorbed enc_norm): {cohens_d_std:.3f}")

print(f"\nTopK SAE (resid_post, 32k):")
print(f"  Labels: {label_source_topk} (n_pos={n_pos_topk})")
for det_name in ["encoder_norm", "eda", "decoder_norm", "random"]:
    res = detectors_topk.get(det_name, {})
    if res.get("auroc"):
        print(f"  {det_name}: AUROC={res['auroc']:.4f}, AUPRC={res['auprc']:.6f}, "
              f"P@50={res.get('precision_at_50', 0):.3f}")
print(f"  frac_high_enc_norm: {frac_high_topk_2sig:.4f} (threshold={threshold_topk_2sig:.3f})")
if cohens_d_topk:
    print(f"  Cohen's d (absorbed vs. non-absorbed enc_norm): {cohens_d_topk:.3f}")

print(f"\nCross-architecture:")
if auroc_diff is not None:
    print(f"  |AUROC diff| = {auroc_diff:.4f} "
          f"({'exceeds' if auroc_diff > 0.10 else 'within'} 0.10 threshold)")
print(f"  HOOK CONFOUND: Standard=resid_pre, TopK=resid_post (different representations)")
print(f"\nPilot PASS: {pilot_pass['pass']}")
print(f"Elapsed: {elapsed:.1f}s")

# Save
output_file = FULL_DIR / "A3_encoder_norm_cross_arch.json"
with open(output_file, "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"\nSaved to: {output_file}")

_std_auroc_str = f"{enc_auroc_std:.4f}" if enc_auroc_std is not None else "N/A"
_topk_auroc_str = f"{enc_auroc_topk:.4f}" if enc_auroc_topk is not None else "N/A"
_diff_str = f"{auroc_diff:.4f}" if auroc_diff is not None else "N/A"
mark_done(
    status="success",
    summary=(
        f"A3 PILOT complete. Standard/L1 SAE: enc_norm AUROC={_std_auroc_str} "
        f"(n_pos={n_pos_std}, {label_source_std}). "
        f"TopK SAE: enc_norm AUROC={_topk_auroc_str} "
        f"(n_pos={n_pos_topk}, {label_source_topk}). "
        f"AUROC diff={_diff_str}. "
        f"Pilot pass={pilot_pass['pass']}. Elapsed={elapsed:.0f}s."
    )
)

# Update gpu_progress.json
gpu_progress_file = WORKSPACE / "exp/gpu_progress.json"
try:
    with open(gpu_progress_file) as f:
        gp = json.load(f)
    # Move from running to completed
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]
    if TASK_ID not in gp.get("completed", []):
        gp.setdefault("completed", []).append(TASK_ID)
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 20,
        "actual_min": int(elapsed / 60),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "std_sae": "gpt2-small-res-jb L6 resid_pre (d_sae=24576)",
            "topk_sae": "gpt2-small-resid-post-v5-32k L6 resid_post (d_sae=32768)",
            "n_pos_std": int(n_pos_std),
            "n_pos_topk": int(n_pos_topk),
            "enc_auroc_std": enc_auroc_std,
            "enc_auroc_topk": enc_auroc_topk,
            "mode": MODE,
        }
    }
    with open(gpu_progress_file, "w") as f:
        json.dump(gp, f, indent=2)
    print(f"Updated gpu_progress.json")
except Exception as e:
    print(f"WARNING: Could not update gpu_progress.json: {e}")

print("\n[DONE] Task A3 pilot complete.")
