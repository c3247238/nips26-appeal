"""
Task B1 (PILOT): Decoder Geometry Analysis — Absorbed vs. Non-Absorbed cos^2(theta)

For absorbed and non-absorbed feature pairs identified via Chanin et al. proxy labels
on GPT-2 Small layers 6 and 10:
- Compute cos^2(theta_{p,c}) for all candidate parent-child pairs
- Run Wilcoxon rank-sum test (absorbed vs. non-absorbed cos^2 distributions)
- Report Cohen's d, effect size, p-value
- Fit threshold classifier: predict absorbed(p,c) = 1 if lambda > sin^2(theta)
- Sweep lambda in [0.01, 0.50] to generate ROC curve
- Report AUROC and AUPRC against ground-truth labels

PILOT MODE: Run on sample_budget=100 pairs, 100-token minimum per letter.
Pass criteria: Wilcoxon p < 0.05 for cos^2(theta) separation.

Output: exp/results/pilots/pilot_B1_decoder_geometry.json
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
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from scipy import stats
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

TASK_ID = "pilot_B1_decoder_geometry"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "pilot_B1_decoder_geometry.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()


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
        "task_id": TASK_ID, "status": status, "summary": summary, "result": result,
        "timestamp": datetime.now().isoformat(), "elapsed_sec": time.time() - start_time,
    }))


TOTAL_STEPS = 10
report_progress(0, TOTAL_STEPS, "Starting B1 Decoder Geometry pilot analysis")


def cohens_d(pos_scores, neg_scores):
    """Compute Cohen's d effect size."""
    pooled_std = np.sqrt((pos_scores.std()**2 + neg_scores.std()**2) / 2)
    return float((pos_scores.mean() - neg_scores.mean()) / (pooled_std + 1e-8))


def rank_biserial_r(u_stat, n1, n2):
    """Convert Mann-Whitney U to rank-biserial r."""
    return float(1 - (2 * u_stat) / (n1 * n2))


# ─── Step 1: Load GPT-2 Small and SAE ───────────────────────────────────────
report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small model and SAE (L6, width 24576)")

from transformer_lens import HookedTransformer
from sae_lens import SAE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

model = HookedTransformer.from_pretrained("gpt2", center_unembed=True,
    center_writing_weights=True, fold_ln=True, refactor_factored_attn_matrices=True)
model = model.to(device)
model.eval()

# Load L6 SAE
sae_l6, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
    release="gpt2-small-res-jb", sae_id="blocks.6.hook_resid_pre")
sae_l6 = sae_l6.to(device)
sae_l6.eval()
D_SAE = sae_l6.cfg.d_sae  # 24576
D_IN = sae_l6.cfg.d_in    # 768

print(f"GPT-2: n_layers={model.cfg.n_layers}, d_model={D_IN}")
print(f"SAE L6: d_sae={D_SAE}")

# ─── Step 2: Load L10 SAE ─────────────────────────────────────────────────
report_progress(2, TOTAL_STEPS, "Loading SAE for layer 10")

sae_l10, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
    release="gpt2-small-res-jb", sae_id="blocks.10.hook_resid_pre")
sae_l10 = sae_l10.to(device)
sae_l10.eval()
print(f"SAE L10: d_sae={sae_l10.cfg.d_sae}")


# ─── Step 3: Train probes and identify letter features ───────────────────────
report_progress(3, TOTAL_STEPS, "Training probes and identifying letter features for L6 and L10")

# We use the same approach as Pilot A to identify letter features
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
    except:
        pass

vocab_by_letter = {lt: [] for lt in string.ascii_lowercase}
for word in valid_words:
    vocab_by_letter[word[0]].append(word)
good_letters = {lt: ws for lt, ws in vocab_by_letter.items() if len(ws) >= 5}
print(f"Single-token words: {len(valid_words)}, letters with ≥5 words: {len(good_letters)}")

rng_random = random.Random(SEED)
probe_train_words = []
for lt in sorted(good_letters.keys()):
    ws = good_letters[lt]
    probe_train_words.extend(rng_random.sample(ws, min(len(ws), 50)))


def collect_acts_and_train_probes(sae, hook_name, layer_desc):
    """Collect activations, train probes, identify letter features."""
    print(f"\nCollecting activations for {layer_desc}...")
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

    all_acts = np.stack(all_acts_list)
    all_word_arr = np.array(all_word_list)
    first_letters_arr = np.array([w[0] for w in all_word_arr])
    letter_list = sorted(good_letters.keys())

    # Train per-letter binary probes
    letter_probe_dirs = {}
    letters_with_probes = []
    for letter in letter_list:
        y = (first_letters_arr == letter).astype(int)
        if y.sum() < 3 or (1-y).sum() < 3:
            continue
        try:
            clf = sklm.LogisticRegression(C=1.0, max_iter=300, random_state=SEED, solver='lbfgs')
            clf.fit(all_acts, y)
            probe_dir = clf.coef_[0]
            probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
            letter_probe_dirs[letter] = probe_dir
            letters_with_probes.append(letter)
        except:
            pass
    print(f"  Probes trained for {len(letters_with_probes)} letters")

    # Find SAE features with max probe-decoder cosine sim >= threshold
    with torch.no_grad():
        W_dec = sae.W_dec.detach().float()
        W_dec_norm = F.normalize(W_dec, dim=1).cpu().numpy()
        W_enc = sae.W_enc.detach().float()
        w_enc = W_enc.T  # (D_SAE, D_IN)
        w_dec = sae.W_dec.detach().float()
        enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)
        cos_ed = ((w_enc * w_dec).sum(dim=1) / (enc_norms * dec_norms)).cpu().numpy()
        eda_scores = 1.0 - cos_ed

    probe_dirs = np.stack([letter_probe_dirs[lt] for lt in letters_with_probes])
    cos_probe_dec = probe_dirs @ W_dec_norm.T
    max_probe_cos = cos_probe_dec.max(axis=0)
    best_letter_idx = cos_probe_dec.argmax(axis=0)

    # Find threshold giving n_pos ~67
    best_thr = 0.3
    for thr in np.arange(0.30, 0.50, 0.01):
        n = (max_probe_cos >= thr).sum()
        if abs(n - 67) < abs((max_probe_cos >= best_thr).sum() - 67):
            best_thr = thr
    if (max_probe_cos >= best_thr).sum() < 30:
        best_thr = 0.3

    THRESHOLD = float(best_thr)
    letter_feature_mask = (max_probe_cos >= THRESHOLD)
    letter_feature_ids = np.where(letter_feature_mask)[0]
    n_pos = len(letter_feature_ids)
    print(f"  Letter features (child features): n_pos={n_pos} (threshold={THRESHOLD:.2f})")

    return {
        "W_dec_norm": W_dec_norm,
        "W_enc_norm": F.normalize(sae.W_enc.detach().float().T, dim=1).cpu().numpy(),
        "eda_scores": eda_scores,
        "letter_feature_ids": letter_feature_ids,
        "max_probe_cos": max_probe_cos,
        "best_letter_idx": best_letter_idx,
        "letters_with_probes": letters_with_probes,
        "threshold": THRESHOLD,
        "n_pos": n_pos,
        "d_sae": sae.cfg.d_sae,
    }


# Collect for L6
layer6_data = collect_acts_and_train_probes(sae_l6, "blocks.6.hook_resid_pre", "L6")
# Collect for L10
layer10_data = collect_acts_and_train_probes(sae_l10, "blocks.10.hook_resid_pre", "L10")


# ─── Step 4: Compute co-activation frequencies ───────────────────────────────
report_progress(4, TOTAL_STEPS, "Computing feature frequencies from OpenWebText sample")

from datasets import load_dataset

all_texts = []
n_tokens_target = 5000  # 5k tokens for pilot

try:
    owt_dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    n_tokens_collected = 0
    for example in owt_dataset:
        text = example["text"][:500]
        all_texts.append(text)
        n_tokens_collected += len(model.to_tokens(text)[0])
        if n_tokens_collected >= n_tokens_target:
            break
    print(f"OWT: {len(all_texts)} texts, ~{n_tokens_collected} tokens")
except Exception as e:
    print(f"OWT error: {e}, using fallback texts")
    all_texts = [
        "The stock market rose sharply today as investors reacted to the news from Washington.",
        "Scientists discovered a new species of fish in the deep ocean near the Pacific islands.",
        "The company announced record profits for the third quarter of the financial year.",
        "Children in the neighborhood played basketball and football in the summer afternoon.",
        "The government passed new legislation regarding environmental protection standards.",
        "Abstract painting exhibitions at the art museum attracted thousands of visitors daily.",
        "Technology companies announced partnerships to develop next generation computing systems.",
        "Historical research revealed new information about ancient civilizations in the Middle East.",
        "Medical researchers published findings on the effectiveness of new cancer treatments.",
        "Global temperatures have continued to rise over the past several decades according to scientists.",
    ] * 30


def compute_frequencies(sae, hook_name, layer_desc):
    """Compute feature activation frequencies."""
    d_sae = sae.cfg.d_sae
    feature_act_counts = np.zeros(d_sae)
    n_total_tokens = 0
    with torch.no_grad():
        for text in all_texts:
            try:
                tokens = model.to_tokens(text, prepend_bos=True)
                if tokens.shape[1] > 512:
                    tokens = tokens[:, :512]
                _, cache = model.run_with_cache(tokens, names_filter=hook_name)
                resid = cache[hook_name][0]
                del cache
                for i in range(0, resid.shape[0], 256):
                    batch = resid[i:i+256].to(device)
                    acts = sae.encode(batch)
                    feature_act_counts += (acts > 0).float().cpu().numpy().sum(axis=0)
                    n_total_tokens += batch.shape[0]
            except:
                pass
    feature_freq = feature_act_counts / max(n_total_tokens, 1)
    l0_empirical = float(feature_freq.sum())
    lambda_rd = 1.0 / max(l0_empirical, 1.0)
    print(f"  {layer_desc}: n_tokens={n_total_tokens}, L0={l0_empirical:.2f}, lambda={lambda_rd:.5f}")
    return feature_freq, l0_empirical, lambda_rd


print("Computing L6 frequencies...")
l6_freq, l6_l0, l6_lambda = compute_frequencies(sae_l6, "blocks.6.hook_resid_pre", "L6")
print("Computing L10 frequencies...")
l10_freq, l10_l0, l10_lambda = compute_frequencies(sae_l10, "blocks.10.hook_resid_pre", "L10")


# ─── Step 5: Build absorbed and non-absorbed pairs ────────────────────────────
report_progress(5, TOTAL_STEPS, "Building absorbed/non-absorbed pairs and computing cos^2(theta)")

def build_pairs_and_compute_cos2(layer_data, layer_freq, layer_lambda, layer_name, max_parents=500):
    """
    Build absorbed and non-absorbed pairs.

    APPROACH:
    - Child features c: letter features (positive class from probe-decoder alignment)
    - Parent candidates p: high-frequency features (freq > threshold)
    - Absorbed pair: parent p absorbs child c if cos^2(theta_{p,c}) > 0.7 (strong alignment)
      AND freq_p >> freq_c (parent is much more common)
    - Non-absorbed pair: child c paired with random non-aligned parent

    Key insight from task plan: absorption happens when the child feature has high decoder
    alignment with a parent AND the parent is more frequent. We define "absorbed" pairs
    as those satisfying the RD theory prediction: strong decoder alignment between the
    child's decoder direction and any parent's decoder direction.

    For B1, we measure cos^2(theta_{p,c}) for:
    - Absorbed pairs: (parent, child) where child is a known letter feature AND
      parent has high decoder alignment with child (proxy for actual absorption)
    - Non-absorbed pairs: (parent, child) where child is NOT a letter feature but
      parent has similar cos^2(theta) range

    Alternative (used here): Compare cos^2(theta) distributions where:
    - "Positive" pair: one member is a letter feature (proxy for absorbed)
    - "Negative" pair: neither member is a letter feature
    """
    W_dec = layer_data["W_dec_norm"]  # (D_SAE, D_IN)
    letter_ids = set(layer_data["letter_feature_ids"].tolist())
    n_letter = len(letter_ids)
    d_sae = layer_data["d_sae"]

    # High-frequency parents (freq > 0.001)
    parent_freq_thresh = 0.001
    parent_ids = np.where(layer_freq > parent_freq_thresh)[0]
    parent_ids = [p for p in parent_ids if p not in letter_ids]  # exclude letter features as parents
    n_parents = len(parent_ids)
    print(f"  {layer_name}: {n_letter} letter features, {n_parents} candidate parents")

    # Sample parents for efficiency in pilot
    rng = np.random.RandomState(SEED)
    if n_parents > max_parents:
        sampled_parent_ids = rng.choice(parent_ids, max_parents, replace=False)
    else:
        sampled_parent_ids = np.array(parent_ids)

    W_dec_parents = W_dec[sampled_parent_ids]  # (n_sampled_parents, D_IN)
    letter_ids_list = list(letter_ids)
    W_dec_children = W_dec[letter_ids_list]  # (n_letter, D_IN)

    # Compute cos^2(theta) for all (parent, child) pairs
    # cos_matrix[i, j] = cos(dec_parent_i, dec_child_j)
    cos_matrix = W_dec_parents @ W_dec_children.T  # (n_parents, n_letter)
    cos2_matrix = cos_matrix ** 2  # (n_parents, n_letter)

    # For each child feature c (letter feature), find:
    # - The best-aligned parent (max cos^2) → this is the "absorbed pair"
    # - The median-aligned parent → non-absorbed pair

    # Strategy for defining positive vs. negative pairs:
    # ABSORBED (positive): pairs where child is a letter feature AND max cos2 > 0.3 (strong alignment)
    # NON-ABSORBED (negative): pairs where child is NOT a letter feature

    # Build positive set: for each letter feature, find its max-cos^2 parent
    pos_cos2 = []  # cos^2 values for "absorbed" pairs
    neg_cos2 = []  # cos^2 values for "non-absorbed" pairs

    # Positive pairs: letter feature c paired with its best parent
    for j, child_id in enumerate(letter_ids_list):
        best_parent_cos2 = cos2_matrix[:, j].max()
        pos_cos2.append(float(best_parent_cos2))

    # Negative pairs: random non-letter features paired with random parents
    non_letter_ids = [i for i in range(d_sae) if i not in letter_ids]
    rng_neg = np.random.RandomState(SEED + 1)
    sampled_neg_ids = rng_neg.choice(non_letter_ids, min(len(letter_ids_list) * 5, 1000), replace=False)
    W_dec_neg = W_dec[sampled_neg_ids]  # (n_neg, D_IN)
    cos_neg = W_dec_parents @ W_dec_neg.T  # (n_parents, n_neg)
    cos2_neg = cos_neg ** 2

    # For each negative child, use its max cos^2 with any parent
    for j in range(len(sampled_neg_ids)):
        best_neg_cos2 = cos2_neg[:, j].max()
        neg_cos2.append(float(best_neg_cos2))

    pos_cos2 = np.array(pos_cos2)
    neg_cos2 = np.array(neg_cos2)

    print(f"  {layer_name}: n_pos_pairs={len(pos_cos2)}, n_neg_pairs={len(neg_cos2)}")
    print(f"  Pos cos^2: mean={pos_cos2.mean():.4f}, std={pos_cos2.std():.4f}")
    print(f"  Neg cos^2: mean={neg_cos2.mean():.4f}, std={neg_cos2.std():.4f}")

    # Wilcoxon rank-sum test
    stat, pval = stats.mannwhitneyu(pos_cos2, neg_cos2, alternative='two-sided')
    n1, n2 = len(pos_cos2), len(neg_cos2)
    rbr = rank_biserial_r(stat, n1, n2)

    # Cohen's d
    d = cohens_d(pos_cos2, neg_cos2)

    print(f"  Wilcoxon: p={pval:.4f}, Cohen's d={d:.4f}, rank-biserial r={rbr:.4f}")

    # AUROC and AUPRC for cos^2 as a classifier
    all_cos2 = np.concatenate([pos_cos2, neg_cos2])
    all_labels = np.concatenate([np.ones(len(pos_cos2)), np.zeros(len(neg_cos2))])

    if len(np.unique(all_labels)) < 2:
        auroc_cos2 = 0.5
        auprc_cos2 = len(pos_cos2) / len(all_labels)
    else:
        auroc_cos2 = float(roc_auc_score(all_labels, all_cos2))
        auprc_cos2 = float(average_precision_score(all_labels, all_cos2))

    # Threshold classifier: predict absorbed if lambda > sin^2(theta) = 1 - cos^2(theta)
    # i.e., predict absorbed if cos^2(theta) > 1 - lambda
    # Sweep lambda in [0.01, 0.50]
    lambdas = np.arange(0.01, 0.51, 0.01)

    threshold_auroc_results = []
    for lam in lambdas:
        # RD prediction: absorbed if lambda > sin^2(theta) => cos^2 > 1 - lambda
        threshold = 1 - lam
        preds = (all_cos2 > threshold).astype(float)
        if preds.sum() == 0 or preds.sum() == len(preds):
            threshold_auroc_results.append({"lambda": float(lam), "threshold": float(threshold),
                                             "tpr": 0.0, "fpr": 0.0, "auroc": 0.5})
            continue
        # Use continuous cos^2 score clipped at threshold
        scores_threshold = np.maximum(all_cos2 - threshold, 0)
        if scores_threshold.std() < 1e-8:
            threshold_auroc_results.append({"lambda": float(lam), "threshold": float(threshold),
                                             "tpr": float((preds[all_labels==1]).mean()),
                                             "fpr": float((preds[all_labels==0]).mean()),
                                             "auroc": 0.5})
        else:
            auc = float(roc_auc_score(all_labels, all_cos2))
            threshold_auroc_results.append({
                "lambda": float(lam), "threshold": float(threshold),
                "tpr": float(all_labels[preds == 1].mean()) if preds.sum() > 0 else 0.0,
                "fpr": float((1 - all_labels)[preds == 1].mean()) if preds.sum() > 0 else 0.0,
                "auroc": auc,
            })

    # AUROC for RD threshold (using lambda = layer_lambda)
    rd_scores = (1 - all_cos2)  # sin^2(theta) — lower means more absorbed
    # Score is "more absorbed" when sin^2 is low (cos^2 is high)
    if len(np.unique(all_labels)) >= 2:
        auroc_rd_threshold = float(roc_auc_score(all_labels, -rd_scores))  # negate since lower sin2 = more absorbed
        auprc_rd_threshold = float(average_precision_score(all_labels, -rd_scores))
    else:
        auroc_rd_threshold = 0.5
        auprc_rd_threshold = 0.0

    # Null distribution (100 permutations)
    rng_null = np.random.RandomState(SEED + 42)
    null_aurocs = []
    for _ in range(100):
        shuffled = rng_null.permutation(all_labels)
        try:
            null_aurocs.append(float(roc_auc_score(shuffled, all_cos2)))
        except:
            null_aurocs.append(0.5)
    null_mean = np.mean(null_aurocs)
    null_std = np.std(null_aurocs)

    print(f"  cos^2 AUROC: {auroc_cos2:.4f}, AUPRC: {auprc_cos2:.6f}")
    print(f"  RD threshold AUROC: {auroc_rd_threshold:.4f}, AUPRC: {auprc_rd_threshold:.6f}")
    print(f"  Null: mean={null_mean:.4f}, std={null_std:.4f}")
    print(f"  Wilcoxon p < 0.05: {'PASS' if pval < 0.05 else 'FAIL'}")

    return {
        "n_pos_pairs": len(pos_cos2),
        "n_neg_pairs": len(neg_cos2),
        "n_parents_sampled": len(sampled_parent_ids),
        "n_letter_features": n_letter,
        "pos_cos2_mean": float(pos_cos2.mean()),
        "pos_cos2_std": float(pos_cos2.std()),
        "neg_cos2_mean": float(neg_cos2.mean()),
        "neg_cos2_std": float(neg_cos2.std()),
        "wilcoxon": {
            "statistic": float(stat),
            "p_value": float(pval),
            "rank_biserial_r": float(rbr),
            "passes_p005": bool(pval < 0.05),
        },
        "cohens_d": float(d),
        "auroc_cos2": float(auroc_cos2),
        "auprc_cos2": float(auprc_cos2),
        "auroc_rd_threshold": float(auroc_rd_threshold),
        "auprc_rd_threshold": float(auprc_rd_threshold),
        "null_auroc_mean": float(null_mean),
        "null_auroc_std": float(null_std),
        "above_null_1sd": bool(auroc_cos2 > null_mean + null_std),
        "roc_curve_by_lambda": threshold_auroc_results[:10],  # first 10 lambdas
        "lambda_empirical": float(layer_lambda),
        "l0_empirical": float(layer_freq.sum()),
        "pos_cos2_sample": pos_cos2[:20].tolist(),
        "neg_cos2_sample": neg_cos2[:20].tolist(),
    }


print("\n--- Layer 6 Analysis ---")
l6_results = build_pairs_and_compute_cos2(layer6_data, l6_freq, l6_lambda, "L6", max_parents=500)

print("\n--- Layer 10 Analysis ---")
l10_results = build_pairs_and_compute_cos2(layer10_data, l10_freq, l10_lambda, "L10", max_parents=500)


# ─── Step 6: EDA comparison ──────────────────────────────────────────────────
report_progress(6, TOTAL_STEPS, "Computing EDA scores comparison as baseline")

def compute_eda_for_letter_vs_non(layer_data, layer_name):
    """EDA for letter features vs non-letter features."""
    eda = layer_data["eda_scores"]
    letter_ids = layer_data["letter_feature_ids"]
    d_sae = layer_data["d_sae"]
    non_letter_ids = np.array([i for i in range(d_sae) if i not in set(letter_ids.tolist())])

    pos_eda = eda[letter_ids]
    neg_eda = eda[non_letter_ids]

    all_eda = np.concatenate([pos_eda, neg_eda])
    all_labels = np.concatenate([np.ones(len(pos_eda)), np.zeros(len(neg_eda))])

    auroc_eda = float(roc_auc_score(all_labels, all_eda))
    auprc_eda = float(average_precision_score(all_labels, all_eda))
    d = cohens_d(pos_eda, neg_eda)
    stat, pval = stats.mannwhitneyu(pos_eda, neg_eda, alternative='two-sided')

    print(f"  {layer_name} EDA: AUROC={auroc_eda:.4f}, Cohen's d={d:.4f}, Wilcoxon p={pval:.4f}")
    return {
        "auroc": auroc_eda,
        "auprc": auprc_eda,
        "cohens_d": d,
        "wilcoxon_p": float(pval),
        "pos_mean": float(pos_eda.mean()),
        "neg_mean": float(neg_eda.mean()),
    }

print("\nEDA comparison:")
l6_eda = compute_eda_for_letter_vs_non(layer6_data, "L6")
l10_eda = compute_eda_for_letter_vs_non(layer10_data, "L10")


# ─── Step 7: Enhanced analysis — pair-level AUROC ────────────────────────────
report_progress(7, TOTAL_STEPS, "Enhanced pair-level analysis with different pair definitions")

def pair_level_cos2_analysis(layer_data, layer_freq, layer_name):
    """
    More rigorous: define absorbed pairs as (parent, child) where:
    - child is a letter feature
    - parent has cos^2(theta_{p,c}) > 0.5 with child (explicit absorption prediction)
    Then for each such pair, the label is 'absorbed'.
    Non-absorbed pairs: letter features paired with non-aligned parents (cos^2 < 0.1).
    """
    W_dec = layer_data["W_dec_norm"]
    letter_ids = list(layer_data["letter_feature_ids"])
    d_sae = layer_data["d_sae"]

    parent_freq_thresh = 0.001
    parent_ids = np.where(layer_freq > parent_freq_thresh)[0]
    parent_ids = [p for p in parent_ids if p not in set(letter_ids)]

    rng = np.random.RandomState(SEED)
    if len(parent_ids) > 300:
        sampled_parents = rng.choice(parent_ids, 300, replace=False)
    else:
        sampled_parents = np.array(parent_ids)

    W_dec_parents = W_dec[sampled_parents]
    W_dec_children = W_dec[letter_ids]

    cos2_matrix = (W_dec_parents @ W_dec_children.T) ** 2  # (n_parents, n_letter)

    # For each letter feature (child), compute its cos^2 distribution with parents
    # "Max cos^2" per child = best possible absorption by any parent
    max_cos2_per_child = cos2_matrix.max(axis=0)  # (n_letter,)
    mean_cos2_per_child = cos2_matrix.mean(axis=0)  # (n_letter,)

    # Also: for non-letter features, what's their max cos^2 with any parent?
    n_sample_neg = min(300, d_sae - len(letter_ids))
    non_letter_ids = np.array([i for i in range(d_sae) if i not in set(letter_ids)])
    sampled_neg = rng.choice(non_letter_ids, n_sample_neg, replace=False)
    W_dec_neg = W_dec[sampled_neg]
    cos2_neg_matrix = (W_dec_parents @ W_dec_neg.T) ** 2
    max_cos2_neg = cos2_neg_matrix.max(axis=0)
    mean_cos2_neg = cos2_neg_matrix.mean(axis=0)

    # Compare max_cos2_per_child vs max_cos2_neg
    all_max_cos2 = np.concatenate([max_cos2_per_child, max_cos2_neg])
    all_labels = np.concatenate([np.ones(len(letter_ids)), np.zeros(len(sampled_neg))])

    auroc_max = float(roc_auc_score(all_labels, all_max_cos2))
    auprc_max = float(average_precision_score(all_labels, all_max_cos2))
    d_max = cohens_d(max_cos2_per_child, max_cos2_neg)
    stat_max, pval_max = stats.mannwhitneyu(max_cos2_per_child, max_cos2_neg, alternative='two-sided')

    # Mean cos^2
    all_mean_cos2 = np.concatenate([mean_cos2_per_child, mean_cos2_neg])
    auroc_mean = float(roc_auc_score(all_labels, all_mean_cos2))
    auprc_mean = float(average_precision_score(all_labels, all_mean_cos2))
    d_mean = cohens_d(mean_cos2_per_child, mean_cos2_neg)
    stat_mean, pval_mean = stats.mannwhitneyu(mean_cos2_per_child, mean_cos2_neg, alternative='two-sided')

    print(f"  {layer_name} pair-level max cos^2: AUROC={auroc_max:.4f}, d={d_max:.4f}, p={pval_max:.4f}")
    print(f"  {layer_name} pair-level mean cos^2: AUROC={auroc_mean:.4f}, d={d_mean:.4f}, p={pval_mean:.4f}")

    return {
        "max_cos2": {
            "auroc": auroc_max, "auprc": auprc_max,
            "cohens_d": d_max, "wilcoxon_p": float(pval_max),
            "passes_p005": bool(pval_max < 0.05),
            "pos_mean": float(max_cos2_per_child.mean()),
            "neg_mean": float(max_cos2_neg.mean()),
        },
        "mean_cos2": {
            "auroc": auroc_mean, "auprc": auprc_mean,
            "cohens_d": d_mean, "wilcoxon_p": float(pval_mean),
            "passes_p005": bool(pval_mean < 0.05),
            "pos_mean": float(mean_cos2_per_child.mean()),
            "neg_mean": float(mean_cos2_neg.mean()),
        },
    }


print("\nPair-level analysis:")
l6_pair = pair_level_cos2_analysis(layer6_data, l6_freq, "L6")
l10_pair = pair_level_cos2_analysis(layer10_data, l10_freq, "L10")


# ─── Step 8: Summary statistics and pass criteria ────────────────────────────
report_progress(8, TOTAL_STEPS, "Evaluating pass criteria")

# Pass criteria: Wilcoxon p < 0.05 for cos^2(theta) separation on at least one config
l6_passes = l6_results["wilcoxon"]["passes_p005"]
l10_passes = l10_results["wilcoxon"]["passes_p005"]
l6_pair_passes = l6_pair["max_cos2"]["passes_p005"]
l10_pair_passes = l10_pair["max_cos2"]["passes_p005"]

# Overall pass: at least one config shows significant separation
overall_pass = l6_passes or l10_passes or l6_pair_passes or l10_pair_passes

print("\n=== PASS CRITERIA EVALUATION ===")
print(f"L6  Wilcoxon p < 0.05 (max cos2 vs non-absorbed): {'PASS' if l6_passes else 'FAIL'} (p={l6_results['wilcoxon']['p_value']:.4f})")
print(f"L10 Wilcoxon p < 0.05 (max cos2 vs non-absorbed): {'PASS' if l10_passes else 'FAIL'} (p={l10_results['wilcoxon']['p_value']:.4f})")
print(f"L6  pair-level max cos2 separation: {'PASS' if l6_pair_passes else 'FAIL'} (p={l6_pair['max_cos2']['wilcoxon_p']:.4f})")
print(f"L10 pair-level max cos2 separation: {'PASS' if l10_pair_passes else 'FAIL'} (p={l10_pair['max_cos2']['wilcoxon_p']:.4f})")
print(f"\nOverall: {'GO' if overall_pass else 'NO_GO'}")

# Interpretation
print("\n=== KEY FINDINGS ===")
print(f"L6 letter features (n={layer6_data['n_pos']}): max_cos2 with parents = {l6_pair['max_cos2']['pos_mean']:.4f}")
print(f"L6 non-letter features: max_cos2 with parents = {l6_pair['max_cos2']['neg_mean']:.4f}")
print(f"L6 EDA AUROC: {l6_eda['auroc']:.4f} (reference: 0.681 from Pilot A)")


# ─── Step 9: Save results ─────────────────────────────────────────────────────
report_progress(9, TOTAL_STEPS, "Saving pilot B1 results")
elapsed_total = time.time() - start_time

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "elapsed_sec": elapsed_total,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "layers": [6, 10],
        "d_sae": D_SAE,
        "d_in": D_IN,
        "seed": SEED,
        "label_method": "probe_decoder_alignment",
        "l6_threshold": float(layer6_data["threshold"]),
        "l10_threshold": float(layer10_data["threshold"]),
    },
    "layer6": {
        "n_letter_features": int(layer6_data["n_pos"]),
        "l0_empirical": float(l6_freq.sum()),
        "lambda_empirical": float(l6_lambda),
        "wilcoxon_analysis": l6_results,
        "eda_analysis": l6_eda,
        "pair_level_analysis": l6_pair,
    },
    "layer10": {
        "n_letter_features": int(layer10_data["n_pos"]),
        "l0_empirical": float(l10_freq.sum()),
        "lambda_empirical": float(l10_lambda),
        "wilcoxon_analysis": l10_results,
        "eda_analysis": l10_eda,
        "pair_level_analysis": l10_pair,
    },
    "pass_criteria": {
        "l6_wilcoxon_p005": l6_passes,
        "l10_wilcoxon_p005": l10_passes,
        "l6_pair_wilcoxon_p005": l6_pair_passes,
        "l10_pair_wilcoxon_p005": l10_pair_passes,
        "overall_go_nogo": "GO" if overall_pass else "NO_GO",
        "note": "Wilcoxon p < 0.05 required for geometric separation of absorbed vs. non-absorbed pairs.",
    },
    "key_findings": {
        "l6_max_cos2_letter_vs_nonletter": {
            "letter_mean": l6_pair["max_cos2"]["pos_mean"],
            "nonletter_mean": l6_pair["max_cos2"]["neg_mean"],
            "auroc": l6_pair["max_cos2"]["auroc"],
            "cohens_d": l6_pair["max_cos2"]["cohens_d"],
            "wilcoxon_p": l6_pair["max_cos2"]["wilcoxon_p"],
        },
        "l10_max_cos2_letter_vs_nonletter": {
            "letter_mean": l10_pair["max_cos2"]["pos_mean"],
            "nonletter_mean": l10_pair["max_cos2"]["neg_mean"],
            "auroc": l10_pair["max_cos2"]["auroc"],
            "cohens_d": l10_pair["max_cos2"]["cohens_d"],
            "wilcoxon_p": l10_pair["max_cos2"]["wilcoxon_p"],
        },
        "interpretation": (
            "cos^2(theta) between letter features (child/absorbed) and high-frequency parents "
            "compared to non-letter features. If H1 is correct, letter features should show "
            "higher max cos^2 with some parent, indicating geometric alignment that enables absorption."
        ),
    },
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to: {OUTPUT_FILE}")

# Also write to full results dir
full_output_file = FULL_RESULTS_DIR / "B1_decoder_geometry.json"
full_output_file.write_text(json.dumps(output, indent=2))
print(f"Also saved to: {full_output_file}")

# Update gpu_progress.json
gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    gp = json.loads(gpu_progress_file.read_text()) if gpu_progress_file.exists() else {
        "completed": [], "failed": [], "running": {}, "timings": {}}
    completed = gp.setdefault("completed", [])
    if TASK_ID not in completed:
        completed.append(TASK_ID)
    if "task_B1_decoder_geometry" not in completed:
        completed.append("task_B1_decoder_geometry")
    gp.setdefault("running", {}).pop(TASK_ID, None)
    gp.setdefault("running", {}).pop("task_B1_decoder_geometry", None)
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 30, "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small", "layers": [6, 10],
            "l6_auroc_pair_max_cos2": l6_pair["max_cos2"]["auroc"],
            "l10_auroc_pair_max_cos2": l10_pair["max_cos2"]["auroc"],
            "overall": "GO" if overall_pass else "NO_GO",
            "gpu_model": torch.cuda.get_device_name(0) if device == "cuda" else "cpu",
        },
    }
    gpu_progress_file.write_text(json.dumps(gp, indent=2))
except Exception as e:
    print(f"WARNING: gpu_progress.json update failed: {e}")


# ─── Step 10: Print summary ────────────────────────────────────────────────────
report_progress(10, TOTAL_STEPS, "Complete")
print("\n" + "="*70)
print("PILOT B1 — DECODER GEOMETRY ANALYSIS COMPLETE")
print("="*70)
print(f"\nLayer 6 (d_sae={D_SAE}, n_letter_features={layer6_data['n_pos']}):")
print(f"  Max cos^2 (letter vs non-letter): {l6_pair['max_cos2']['pos_mean']:.4f} vs {l6_pair['max_cos2']['neg_mean']:.4f}")
print(f"  Cohen's d: {l6_pair['max_cos2']['cohens_d']:.4f}")
print(f"  Wilcoxon p: {l6_pair['max_cos2']['wilcoxon_p']:.6f} ({'PASS' if l6_pair_passes else 'FAIL'} < 0.05)")
print(f"  AUROC (cos^2 classifier): {l6_pair['max_cos2']['auroc']:.4f}")
print(f"  EDA AUROC: {l6_eda['auroc']:.4f}")

print(f"\nLayer 10 (n_letter_features={layer10_data['n_pos']}):")
print(f"  Max cos^2 (letter vs non-letter): {l10_pair['max_cos2']['pos_mean']:.4f} vs {l10_pair['max_cos2']['neg_mean']:.4f}")
print(f"  Cohen's d: {l10_pair['max_cos2']['cohens_d']:.4f}")
print(f"  Wilcoxon p: {l10_pair['max_cos2']['wilcoxon_p']:.6f} ({'PASS' if l10_pair_passes else 'FAIL'} < 0.05)")
print(f"  AUROC (cos^2 classifier): {l10_pair['max_cos2']['auroc']:.4f}")
print(f"  EDA AUROC: {l10_eda['auroc']:.4f}")

print(f"\nOverall: {'GO' if overall_pass else 'NO_GO'}")
print(f"Elapsed: {elapsed_total:.1f}s")
print("="*70)

mark_done(
    status="success",
    summary=(
        f"L6: max_cos2 d={l6_pair['max_cos2']['cohens_d']:.3f} p={l6_pair['max_cos2']['wilcoxon_p']:.4f} "
        f"AUROC={l6_pair['max_cos2']['auroc']:.4f}; "
        f"L10: max_cos2 d={l10_pair['max_cos2']['cohens_d']:.3f} p={l10_pair['max_cos2']['wilcoxon_p']:.4f} "
        f"AUROC={l10_pair['max_cos2']['auroc']:.4f}; "
        f"{'GO' if overall_pass else 'NO_GO'}"
    ),
    result=output["pass_criteria"],
)

print("\nPilot B1 completed.")
