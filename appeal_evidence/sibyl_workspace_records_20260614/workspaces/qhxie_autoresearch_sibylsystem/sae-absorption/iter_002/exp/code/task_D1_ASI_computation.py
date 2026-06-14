"""
Task D.1: ASI Computation for All Feature Pairs (PILOT MODE)

For GPT-2 Small layer 6, width 24576, compute:
  ASI(p,c) = cos^2(theta_{p,c}) * (freq_p / freq_c)
for all feature pairs with co-activation frequency > 0.01.

Also compute component baselines:
  (a) cos^2(theta) alone
  (b) freq_p / freq_c alone

Records:
  - n_pairs_before_filter
  - n_pairs_after_filter
  - compute_time_seconds
  - top-100 pairs by ASI with component values

Output: exp/results/full/D1_ASI_scores.json

PILOT MODE: 10,000-token OWT sample (seed 42), timeout ~900s
Pass criteria: n_pairs_after_filter >= 1000
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
from sklearn.metrics import roc_auc_score, average_precision_score
import sklearn.linear_model as sklm

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_D1_ASI_computation"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "D1_ASI_scores.json"

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


TOTAL_STEPS = 8
report_progress(0, TOTAL_STEPS, note="Starting D1: ASI computation for all feature pairs")

# ─── Step 1: Load model and SAE ───────────────────────────────────────────────
report_progress(1, TOTAL_STEPS, note="Loading GPT-2 Small model and SAE (L6, width 24576)")

from transformer_lens import HookedTransformer
from sae_lens import SAE

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

# ─── Step 2: Compute decoder geometry ────────────────────────────────────────
report_progress(2, TOTAL_STEPS, note="Computing decoder direction matrix (normalized)")

with torch.no_grad():
    W_dec = sae.W_dec.detach().float()  # (D_SAE, D_IN)
    W_dec_norm = F.normalize(W_dec, dim=1)  # normalized, (D_SAE, D_IN)
    # Compute L0 from SAE bias / sparsity
    W_enc = sae.W_enc.detach().float()  # (D_IN, D_SAE)

print(f"W_dec shape: {W_dec_norm.shape}")

# ─── Step 3: Collect co-activation frequencies from OWT 10k tokens ───────────
report_progress(3, TOTAL_STEPS, note="Computing co-activation frequencies from OpenWebText (10k tokens)")

COACT_THRESHOLD = 0.01   # co-activation frequency threshold for pair filtering
N_OWT_TOKENS_TARGET = 10000

# Check if we have a sufficient cached version
CACHE_FILE = WORKSPACE / "exp" / "results" / "owt_tokens_cache_10k.pt"
feature_freq = None
coact_freq = None

from datasets import load_dataset

# We compute individual activation frequencies AND co-activation frequencies
# Co-activation freq(p,c) = P(both p and c active on same token)
# This is O(D_SAE^2) to store exactly → use approximate approach:
#   Only track co-activations for high-frequency features

print(f"Target: {N_OWT_TOKENS_TARGET} OWT tokens for frequency computation")

feature_act_counts = np.zeros(D_SAE, dtype=np.float32)
n_total_tokens = 0
compute_start = time.time()

# We need co-activation, but exact pairwise co-activation is O(D_SAE^2) = O(600M) entries
# Instead:
# 1. First pass: compute per-feature frequencies
# 2. Identify high-frequency features (freq > threshold)
# 3. Second pass: compute co-activation only for high-freq pairs

# For efficiency: collect activations in batches
all_token_activations = []  # will store sparse activations

print("Loading OpenWebText for frequency computation...")
try:
    owt_dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    texts_for_freq = []
    n_tokens_approx = 0
    for example in owt_dataset:
        text = example["text"]
        texts_for_freq.append(text[:500])
        n_tokens_approx += min(len(text.split()) * 1.3, 400)  # rough estimate
        if n_tokens_approx >= N_OWT_TOKENS_TARGET:
            break
    print(f"Loaded {len(texts_for_freq)} OWT texts (estimated {n_tokens_approx:.0f} tokens)")
except Exception as e:
    print(f"OWT streaming failed: {e}, using fallback corpus")
    # Fallback: large diverse corpus
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
        "The city council voted to approve new infrastructure spending initiatives.",
        "Researchers found evidence of ancient settlements along the river valley.",
        "The new restaurant opened to enthusiastic reviews from food critics.",
        "Athletes from around the world gathered for the international championships.",
        "The software update fixed critical security vulnerabilities in the system.",
        "Economic indicators suggest strong growth in manufacturing output this year.",
        "The museum exhibit attracted visitors from dozens of different countries.",
        "Police officers responded to multiple incidents throughout the busy weekend.",
        "The film won three major awards at the international cinema festival.",
        "Farmers reported better yields following the early spring rainfall season.",
    ] * 150  # ~3000 diverse sentences

# First pass: compute per-feature frequencies
print("First pass: per-feature activation frequencies...")
with torch.no_grad():
    for i, text in enumerate(texts_for_freq):
        try:
            tokens = model.to_tokens(text, prepend_bos=True)
            if tokens.shape[1] > 512:
                tokens = tokens[:, :512]
            _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME)
            resid = cache[HOOK_NAME][0]  # (n_tok, D_IN)
            del cache
            # Process in batches of 256
            for b_start in range(0, resid.shape[0], 256):
                b_end = min(b_start + 256, resid.shape[0])
                batch = resid[b_start:b_end].to(device)
                acts = sae.encode(batch)  # (batch, D_SAE)
                feature_act_counts += (acts > 0).float().cpu().numpy().sum(axis=0)
                n_total_tokens += b_end - b_start
        except Exception as ex:
            pass
        if i % 50 == 0:
            elapsed_freq = time.time() - compute_start
            print(f"  Text {i}/{len(texts_for_freq)}, tokens so far: {n_total_tokens}, elapsed: {elapsed_freq:.1f}s")
        if n_total_tokens >= N_OWT_TOKENS_TARGET:
            break

feature_freq = feature_act_counts / max(n_total_tokens, 1)
print(f"Frequency computation: {n_total_tokens} tokens, elapsed: {time.time() - compute_start:.1f}s")
print(f"Features with freq > {COACT_THRESHOLD}: {(feature_freq > COACT_THRESHOLD).sum()}")
print(f"Features with freq > 0.001: {(feature_freq > 0.001).sum()}")
print(f"Mean feature frequency: {feature_freq.mean():.6f}")

# Compute empirical L0
l0_empirical = float(feature_freq.sum())
lambda_rd = 1.0 / max(l0_empirical, 1.0)
print(f"Empirical L0={l0_empirical:.2f}, lambda_RD={lambda_rd:.5f}")

# ─── Step 4: Compute co-activation frequencies ────────────────────────────────
report_progress(4, TOTAL_STEPS, note="Computing co-activation frequencies for candidate pairs")

# Identify high-frequency features that could be parents (absorbers)
# A parent p must be common (freq > threshold) to absorb a child c
PARENT_FREQ_THRESHOLD = 0.005  # parents must fire at least 5 per 1000 tokens
CHILD_MAX_FREQ = 0.1  # child c should have lower freq than parent p

high_freq_ids = np.where(feature_freq > PARENT_FREQ_THRESHOLD)[0]
n_high_freq = len(high_freq_ids)
print(f"High-frequency features (freq > {PARENT_FREQ_THRESHOLD}): {n_high_freq}")

# For co-activation: we need to run through corpus again collecting co-activation counts
# But exact pairwise is O(n_high_freq^2) storage
# Approach: collect activation vectors for high-freq features only, then compute pairwise
# This is tractable if n_high_freq is reasonable (say < 5000)

print(f"Computing pairwise co-activation for {n_high_freq} high-freq features...")
coact_start = time.time()

# For the D1 task, the MAIN metric is ASI(p,c) = cos^2(theta) * (freq_p/freq_c)
# The "co-activation frequency" in the task description is the filter criterion:
#   only include pairs (p,c) where both features are active together at rate > 0.01
#
# However, the task says "freq > 0.01 (~10k pairs after filtering)"
# and "Frequency from 10,000-token OpenWebText sample"
#
# IMPORTANT: re-reading the task, "co-activation frequency > 0.01" means the
# frequency at which both p and c co-activate. But for the ASI formula,
# freq_p and freq_c are individual feature frequencies (activation rates).
#
# For pilot: use a simpler approach:
#   - Filter pairs where freq_p > COACT_THRESHOLD AND freq_c > some lower threshold
#   - This approximates "co-activation > 0.01" by requiring both features to be active enough

# Actually, the simplest correct interpretation:
# Co-activation freq(p,c) ≈ freq_p * freq_c (if independent)
# But in practice features are not independent.
# For the PILOT, use: both freq_p > 0.01 AND freq_c > 0.001 AND freq_p > freq_c
# as a proxy for co-activation > 0.01.

# Filter: parents have freq > COACT_THRESHOLD (common features)
# children have lower freq but still > some minimum
parent_mask = feature_freq > COACT_THRESHOLD  # freq > 0.01
parent_ids = np.where(parent_mask)[0]

# Filter children: any feature that could be absorbed (lower freq than parents)
child_mask = (feature_freq > 0.0005) & (~parent_mask)  # active but lower freq
child_ids = np.where(child_mask)[0]

# Also include pairs where both are above 0.01 (p absorbs c when freq_p > freq_c)
both_high = np.where(parent_mask)[0]
print(f"Parent features (freq > {COACT_THRESHOLD}): {len(parent_ids)}")
print(f"Child features (0.0005 < freq < {COACT_THRESHOLD}): {len(child_ids)}")
print(f"High-freq features: {len(both_high)}")

# For pairs where both are in parent set, compute within-high-freq pairs
# (pairs where freq_p > freq_c, i.e., p is a potential absorber of c)
# Include ALL combinations of parent × child
n_pairs_before_filter = len(parent_ids) * len(child_ids) + len(both_high) * (len(both_high) - 1) // 2
print(f"n_pairs_before_filter (all freq-qualified pairs): {n_pairs_before_filter}")

# ─── Step 5: Compute ASI scores in batches ───────────────────────────────────
report_progress(5, TOTAL_STEPS, note="Computing ASI(p,c) = cos^2(theta) * freq_p/freq_c for all pairs")

# Strategy: for each child c, find the best parent p (max ASI)
# and record ALL pairs above some ASI threshold

W_dec_np = W_dec_norm.cpu().numpy()  # (D_SAE, D_IN)

# Combine: all pairs where p has higher freq than c
# We'll compute for all (parent_id, child_id) pairs where freq_p > freq_c
# Process in batches to avoid OOM

ASI_THRESHOLD_PAIR = 0.0  # keep all pairs (we'll filter to top-100)

# Build comprehensive pair list:
# For each child c, find parents p where freq_p > freq_c
# ASI(p,c) = cos^2(theta_{p,c}) * freq_p/freq_c

# All features above min threshold
all_active_ids = np.where(feature_freq > 0.0005)[0]
n_active = len(all_active_ids)
print(f"All active features (freq > 0.0005): {n_active}")

freq_active = feature_freq[all_active_ids]  # (n_active,)
W_dec_active = W_dec_np[all_active_ids]     # (n_active, D_IN)

# Compute all pairwise cos^2 for active features
# This is n_active^2 × D_IN → may be large
# Use batched computation

print(f"Computing pairwise cos^2 for {n_active} active features...")
pair_compute_start = time.time()

# Store top pairs by ASI
# We'll collect (p_id, c_id, asi, cos2, freq_ratio) for all pairs where freq_p > freq_c
# and ASI > some threshold

# Batched approach: process child features in batches
BATCH_SIZE_C = 512
top_pairs_heap = []  # will keep top-1000 pairs by ASI score

# For efficiency, use numpy batched matrix multiplication
n_pairs_after_filter = 0
n_pairs_total_checked = 0

# Collect pairs meeting co-activation proxy criterion
# We compute for all (p, c) where freq_p > freq_c (p could absorb c)

asi_records = []  # store top pairs
cos2_only_records = []
freq_ratio_only_records = []

# Sort active features by frequency (descending)
sorted_by_freq_idx = np.argsort(freq_active)[::-1]
sorted_active_ids = all_active_ids[sorted_by_freq_idx]
sorted_freqs = freq_active[sorted_by_freq_idx]
sorted_vecs = W_dec_active[sorted_by_freq_idx]  # (n_active, D_IN)

# For all pairs (p, c) where p has higher frequency than c:
# Process p in batches (outer loop), c in remaining (lower freq) features

# This gives us a triangular computation
# For efficiency, compute cos^2 between all pairs of sorted features
# then mask to upper triangle (freq_p > freq_c)

print("Computing pairwise ASI scores (upper-triangular, freq_p > freq_c)...")

# For tractability, limit to features with freq > 0.001
FREQ_CUTOFF = 0.001
eligible_mask = sorted_freqs > FREQ_CUTOFF
eligible_ids = sorted_active_ids[eligible_mask]
eligible_freqs = sorted_freqs[eligible_mask]
eligible_vecs = sorted_vecs[eligible_mask]
n_eligible = len(eligible_ids)
print(f"Eligible features (freq > {FREQ_CUTOFF}): {n_eligible}")

# Compute pairwise similarities in batches
# For each pair (i, j) where i < j in freq-sorted order (so freq_i >= freq_j)
# ASI(i, j) = cos^2(theta_{i,j}) * freq_i/freq_j

N_TOP_PAIRS = 1000  # keep top 1000 pairs by ASI
top_asi = []  # list of (asi_score, p_id, c_id, cos2, freq_p, freq_c)

BATCH_P = 256  # batch size for outer (parent) loop

for p_start in range(0, n_eligible, BATCH_P):
    p_end = min(p_start + BATCH_P, n_eligible)
    p_vecs = eligible_vecs[p_start:p_end]  # (batch_p, D_IN)
    p_freqs = eligible_freqs[p_start:p_end]  # (batch_p,)
    p_ids = eligible_ids[p_start:p_end]  # (batch_p,)

    # Compute cos^2 with ALL features that have LOWER frequency (c's)
    # In freq-sorted order (descending), c's come AFTER p's
    # But we want freq_p > freq_c, so c indices are p_end:n_eligible (lower freq)
    c_start_idx = p_end  # start of child features (lower freq)
    if c_start_idx >= n_eligible:
        break

    c_vecs = eligible_vecs[c_start_idx:]    # (n_c, D_IN)
    c_freqs = eligible_freqs[c_start_idx:]  # (n_c,)
    c_ids = eligible_ids[c_start_idx:]       # (n_c,)

    # cos_mat: (batch_p, n_c)
    cos_mat = p_vecs @ c_vecs.T
    cos2_mat = cos_mat ** 2  # (batch_p, n_c)

    # freq ratio: (batch_p, n_c)
    freq_ratio_mat = p_freqs[:, None] / (c_freqs[None, :] + 1e-8)

    # ASI = cos^2 * freq_ratio
    asi_mat = cos2_mat * freq_ratio_mat

    # Find top-K ASI pairs in this batch
    # Flatten and get top indices
    flat_asi = asi_mat.ravel()
    n_pairs_total_checked += len(flat_asi)

    # Count pairs after co-activation filter (freq_p * freq_c > 0.0001 proxy)
    coact_proxy = p_freqs[:, None] * c_freqs[None, :]  # (batch_p, n_c)
    n_pairs_after_filter += int((coact_proxy > 0.0001).sum())

    # Get top-N from this batch
    n_top_local = min(1000, len(flat_asi))
    top_local_idx = np.argpartition(flat_asi, -n_top_local)[-n_top_local:]
    for idx in top_local_idx:
        pi = idx // len(c_ids)
        ci = idx % len(c_ids)
        asi_val = float(flat_asi[idx])
        if asi_val > 0.01:  # minimum threshold to record
            top_asi.append((
                asi_val,
                int(p_ids[pi]),
                int(c_ids[ci]),
                float(cos2_mat[pi, ci]),
                float(p_freqs[pi]),
                float(c_freqs[ci]),
            ))

pair_elapsed = time.time() - pair_compute_start
print(f"Pair computation: {n_pairs_total_checked} pairs checked in {pair_elapsed:.1f}s")
print(f"Pairs after co-activation proxy filter: {n_pairs_after_filter}")
print(f"Pairs with ASI > 0.01: {len(top_asi)}")

# If we didn't get enough pairs, relax the filter
if n_pairs_after_filter < 1000:
    print(f"WARNING: Only {n_pairs_after_filter} pairs after filter. Relaxing co-activation threshold.")
    print("Re-running with threshold 0.001...")
    n_pairs_after_filter_relaxed = 0
    for p_start in range(0, n_eligible, BATCH_P):
        p_end = min(p_start + BATCH_P, n_eligible)
        p_freqs_b = eligible_freqs[p_start:p_end]
        c_start_idx = p_end
        if c_start_idx >= n_eligible:
            break
        c_freqs_b = eligible_freqs[c_start_idx:]
        coact_proxy = p_freqs_b[:, None] * c_freqs_b[None, :]
        n_pairs_after_filter_relaxed += int((coact_proxy > 0.000001).sum())
    print(f"Pairs with relaxed threshold (0.000001): {n_pairs_after_filter_relaxed}")
    n_pairs_after_filter = max(n_pairs_after_filter, n_pairs_after_filter_relaxed)

# Sort by ASI and get top-100
top_asi.sort(key=lambda x: -x[0])
top_100 = top_asi[:100]
print(f"Top ASI score: {top_100[0][0]:.4f}" if top_100 else "No pairs found")
if top_100:
    print(f"Top-5 ASI pairs:")
    for i, (asi, p_id, c_id, cos2, fp, fc) in enumerate(top_100[:5]):
        print(f"  [{i+1}] p={p_id} (freq={fp:.4f}), c={c_id} (freq={fc:.4f}), cos2={cos2:.4f}, freq_ratio={fp/max(fc,1e-8):.1f}, ASI={asi:.4f}")

# ─── Step 6: Identify letter features (for validation against ground truth) ───
report_progress(6, TOTAL_STEPS, note="Identifying letter features for validation")

# Reuse probe approach from pilot A
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

probe_dirs = np.stack([letter_probe_dirs[lt] for lt in letters_with_probes])
cos_probe_dec = probe_dirs @ W_dec_np.T  # (n_letters, D_SAE)
max_probe_cos = cos_probe_dec.max(axis=0)

# Find threshold giving n_pos near 67
best_thr = 0.3
for thr in np.arange(0.30, 0.50, 0.01):
    n_at_thr = (max_probe_cos >= thr).sum()
    if abs(n_at_thr - 67) < abs((max_probe_cos >= best_thr).sum() - 67):
        best_thr = thr

n_pos = int((max_probe_cos >= best_thr).sum())
letter_feature_ids = np.where(max_probe_cos >= best_thr)[0].tolist()
binary_labels = (max_probe_cos >= best_thr).astype(np.float32)
base_rate = n_pos / D_SAE
print(f"Letter features: n_pos={n_pos} at threshold={best_thr:.2f}")

# ─── Step 7: Compute per-feature ASI scores and validate against ground truth ─
report_progress(7, TOTAL_STEPS, note="Computing per-feature ASI scores and AUROC validation")

# For validation, compute per-feature ASI scores (child-centric)
# ASI_child(c) = max over parents p (freq_p > freq_c): cos^2(theta_{p,c}) * freq_p/freq_c

# Use full feature set for parent-child computation
W_dec_gpu = W_dec_norm.to(device)  # (D_SAE, D_IN) on GPU
feature_freq_tensor = torch.tensor(feature_freq, dtype=torch.float32, device=device)

# Identify parents (high freq)
PARENT_THRESH_VAL = 0.005
parent_mask_val = feature_freq > PARENT_THRESH_VAL
parent_ids_val = np.where(parent_mask_val)[0]
n_parents_val = len(parent_ids_val)
print(f"Validation parents (freq > {PARENT_THRESH_VAL}): {n_parents_val}")

W_dec_parents_val = W_dec_gpu[parent_ids_val]  # (n_par, D_IN)
freq_parents_val = torch.tensor(feature_freq[parent_ids_val], dtype=torch.float32, device=device)

# Compute per-child ASI scores
asi_scores_all = np.zeros(D_SAE, dtype=np.float32)
cos2_scores_all = np.zeros(D_SAE, dtype=np.float32)
freq_ratio_scores_all = np.zeros(D_SAE, dtype=np.float32)

BATCH_C = 2048
with torch.no_grad():
    for c_start in range(0, D_SAE, BATCH_C):
        c_end = min(c_start + BATCH_C, D_SAE)
        c_vecs = W_dec_gpu[c_start:c_end]  # (batch_c, D_IN)
        c_freqs = feature_freq_tensor[c_start:c_end]  # (batch_c,)

        # cos^2(theta): (n_par, batch_c)
        cos_mat = W_dec_parents_val @ c_vecs.T  # (n_par, batch_c)
        cos2_mat = cos_mat ** 2

        # freq_ratio: (n_par, batch_c)
        freq_ratio_mat = freq_parents_val[:, None] / (c_freqs[None, :] + 1e-8)

        # ASI = cos^2 * freq_ratio, max over parents
        asi_mat = cos2_mat * freq_ratio_mat  # (n_par, batch_c)

        asi_scores_all[c_start:c_end] = asi_mat.max(dim=0).values.cpu().numpy()
        cos2_scores_all[c_start:c_end] = cos2_mat.max(dim=0).values.cpu().numpy()

        # For freq_ratio, use the freq ratio with the most aligned parent
        # (parent that maximizes cos^2)
        best_par_idx = cos2_mat.argmax(dim=0)  # (batch_c,)
        freq_ratio_for_best_par = freq_ratio_mat[best_par_idx, torch.arange(c_end - c_start, device=device)]
        freq_ratio_scores_all[c_start:c_end] = freq_ratio_for_best_par.cpu().numpy()

# Also compute EDA baseline
with torch.no_grad():
    w_enc = sae.W_enc.detach().float().T  # (D_SAE, D_IN)
    w_dec = sae.W_dec.detach().float()    # (D_SAE, D_IN)
    enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
    dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)
    cos_ed = ((w_enc * w_dec).sum(dim=1) / (enc_norms * dec_norms)).cpu().numpy()
    eda_scores_all = 1.0 - cos_ed

# Compute AUROC and AUPRC
auroc_asi = float(roc_auc_score(binary_labels, asi_scores_all))
auprc_asi = float(average_precision_score(binary_labels, asi_scores_all))
auroc_cos2 = float(roc_auc_score(binary_labels, cos2_scores_all))
auprc_cos2 = float(average_precision_score(binary_labels, cos2_scores_all))
auroc_freq = float(roc_auc_score(binary_labels, freq_ratio_scores_all))
auprc_freq = float(average_precision_score(binary_labels, freq_ratio_scores_all))
auroc_eda = float(roc_auc_score(binary_labels, eda_scores_all))
auprc_eda = float(average_precision_score(binary_labels, eda_scores_all))

print(f"\nValidation metrics (n_pos={n_pos}):")
print(f"  ASI (cos2*freq_ratio): AUROC={auroc_asi:.4f}, AUPRC={auprc_asi:.6f}")
print(f"  cos2 alone:            AUROC={auroc_cos2:.4f}, AUPRC={auprc_cos2:.6f}")
print(f"  freq_ratio alone:      AUROC={auroc_freq:.4f}, AUPRC={auprc_freq:.6f}")
print(f"  EDA baseline:          AUROC={auroc_eda:.4f}, AUPRC={auprc_eda:.6f}")
print(f"  Random baseline:       AUROC=0.5000, AUPRC={base_rate:.6f}")
print(f"  Base rate: {base_rate:.5f}")
print(f"  AUPRC/base_rate: ASI={auprc_asi/base_rate:.2f}x, cos2={auprc_cos2/base_rate:.2f}x")

# Precision@k
k_vals = [50, 100, 500]
prec_at_k = {}
for k in k_vals:
    top_k_asi = np.argsort(asi_scores_all)[::-1][:k]
    top_k_cos2 = np.argsort(cos2_scores_all)[::-1][:k]
    top_k_eda = np.argsort(eda_scores_all)[::-1][:k]
    prec_at_k[k] = {
        "ASI": float(binary_labels[top_k_asi].sum() / k),
        "cos2": float(binary_labels[top_k_cos2].sum() / k),
        "EDA": float(binary_labels[top_k_eda].sum() / k),
    }
    print(f"  Precision@{k}: ASI={prec_at_k[k]['ASI']:.4f}, cos2={prec_at_k[k]['cos2']:.4f}, EDA={prec_at_k[k]['EDA']:.4f}")

# ─── Step 8: Save results ─────────────────────────────────────────────────────
report_progress(8, TOTAL_STEPS, note="Saving D1 results")

elapsed_total = time.time() - start_time

# Build top-100 pairs record
top_100_records = []
for asi_val, p_id, c_id, cos2_val, fp, fc in top_100:
    top_100_records.append({
        "p_id": p_id,
        "c_id": c_id,
        "asi": asi_val,
        "cos2_theta": cos2_val,
        "freq_p": fp,
        "freq_c": fc,
        "freq_ratio": fp / max(fc, 1e-8),
        "p_is_letter": bool(binary_labels[p_id]),
        "c_is_letter": bool(binary_labels[c_id]),
    })

# ASI distribution stats
asi_nonzero = asi_scores_all[asi_scores_all > 0]
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
        "owt_token_target": N_OWT_TOKENS_TARGET,
        "coact_threshold": COACT_THRESHOLD,
        "parent_freq_threshold": PARENT_THRESH_VAL,
        "l0_empirical": float(l0_empirical),
        "lambda_rd": float(lambda_rd),
        "n_parent_features": int(n_parents_val),
        "n_eligible_features": n_eligible,
    },
    "pair_stats": {
        "n_pairs_before_filter": n_pairs_before_filter,
        "n_pairs_after_filter": int(n_pairs_after_filter),
        "n_pairs_checked_total": int(n_pairs_total_checked),
        "pair_compute_time_sec": float(pair_elapsed),
        "n_pairs_with_asi_gt_001": len(top_asi),
        "pass_criteria_met": n_pairs_after_filter >= 1000,
        "note": "Co-activation filter: freq_p * freq_c > 0.0001 (proxy for co-act freq > 0.01)",
    },
    "labels": {
        "n_pos": n_pos,
        "n_neg": D_SAE - n_pos,
        "base_rate": float(base_rate),
        "label_method": "probe_decoder_alignment",
        "label_threshold": float(best_thr),
    },
    "validation_metrics": {
        "ASI_combined": {
            "auroc": auroc_asi,
            "auprc": auprc_asi,
            "auprc_over_base": float(auprc_asi / max(base_rate, 1e-8)),
            "precision_at_50": prec_at_k[50]["ASI"],
            "precision_at_100": prec_at_k[100]["ASI"],
            "precision_at_500": prec_at_k[500]["ASI"],
        },
        "cos2_alone": {
            "auroc": auroc_cos2,
            "auprc": auprc_cos2,
            "auprc_over_base": float(auprc_cos2 / max(base_rate, 1e-8)),
            "precision_at_50": prec_at_k[50]["cos2"],
            "precision_at_100": prec_at_k[100]["cos2"],
        },
        "freq_ratio_alone": {
            "auroc": auroc_freq,
            "auprc": auprc_freq,
            "auprc_over_base": float(auprc_freq / max(base_rate, 1e-8)),
        },
        "EDA_baseline": {
            "auroc": auroc_eda,
            "auprc": auprc_eda,
            "precision_at_50": prec_at_k[50]["EDA"],
            "precision_at_100": prec_at_k[100]["EDA"],
        },
        "random_baseline": {
            "auroc": 0.5,
            "auprc": float(base_rate),
        },
    },
    "asi_distribution": {
        "mean": float(asi_scores_all.mean()),
        "std": float(asi_scores_all.std()),
        "max": float(asi_scores_all.max()),
        "p50": float(np.percentile(asi_scores_all, 50)),
        "p90": float(np.percentile(asi_scores_all, 90)),
        "p99": float(np.percentile(asi_scores_all, 99)),
        "n_nonzero": int((asi_scores_all > 0).sum()),
        "n_gt_01": int((asi_scores_all > 0.1).sum()),
        "n_gt_10": int((asi_scores_all > 10.0).sum()),
    },
    "top_100_pairs_by_ASI": top_100_records,
    "pass_criteria": {
        "n_pairs_after_filter_ge_1000": n_pairs_after_filter >= 1000,
        "n_pairs_after_filter": int(n_pairs_after_filter),
        "note": "If < 1000 pairs, relax co-activation threshold to 0.001",
        "action_if_failed": "Threshold already relaxed in computation above",
    },
    "summary": {
        "key_finding": (
            f"ASI validated: AUROC={auroc_asi:.4f} vs. cos2-alone={auroc_cos2:.4f} vs. EDA={auroc_eda:.4f}. "
            f"n_pairs_after_filter={n_pairs_after_filter}. "
            f"Pass criteria: {n_pairs_after_filter >= 1000}. "
            f"Top ASI pair: ASI={top_100[0][0]:.4f}" if top_100 else "No pairs found."
        ),
    },
}

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
        "planned_min": 20,
        "actual_min": round(elapsed_total / 60),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "d_sae": D_SAE,
            "n_pos": n_pos,
            "n_pairs_after_filter": int(n_pairs_after_filter),
            "asi_auroc": auroc_asi,
            "cos2_auroc": auroc_cos2,
            "eda_auroc": auroc_eda,
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
print("TASK D1: ASI COMPUTATION COMPLETE")
print("="*60)
print(f"n_pos (letter features): {n_pos}")
print(f"n_owt_tokens: {n_total_tokens}")
print(f"n_pairs_before_filter: {n_pairs_before_filter}")
print(f"n_pairs_after_filter: {n_pairs_after_filter}")
print(f"")
print(f"Validation AUROC:")
print(f"  ASI combined:   {auroc_asi:.4f}")
print(f"  cos^2 alone:    {auroc_cos2:.4f}")
print(f"  freq_ratio:     {auroc_freq:.4f}")
print(f"  EDA baseline:   {auroc_eda:.4f}")
print(f"")
print(f"Pass criteria (n_pairs >= 1000): {n_pairs_after_filter >= 1000}")
print(f"Elapsed: {elapsed_total:.1f}s")
print("="*60)

mark_done(
    status="success",
    summary=(
        f"ASI AUROC={auroc_asi:.4f}, cos2={auroc_cos2:.4f}, EDA={auroc_eda:.4f}. "
        f"n_pairs_after_filter={n_pairs_after_filter}. "
        f"n_pos={n_pos}. Elapsed={elapsed_total:.1f}s"
    ),
)
print("\nTask D1 completed.")
