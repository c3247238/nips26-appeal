#!/usr/bin/env python3
"""
successive_refinement.py — Stage 5: CMI Estimation and Markov Chain Test

For 26 first-letter features on Gemma 2 2B L12 16k, compute
I(X; f_parent | f_child) using k-NN MI estimator after projecting onto
decoder-direction subspace (d'~10-50).

Loads absorption rates from first_letter_validation.json (dependency).
Partitions letters into absorbed (rate>10%) and non-absorbed (rate<5%).
Mann-Whitney U test for CMI difference. Report Cohen's d.

Theoretical framing: Successive refinement (Equitz & Cover, 1991) says that
if f_child carries all info about X that f_parent carries, then
I(X; f_parent | f_child) ≈ 0 and absorption is "lossless". If >> 0,
absorption destroys information that f_parent uniquely carried.
"""

import json
import os
import sys
import time
import gc
import random
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.special import digamma

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "successive_refinement"
SEED = 42
GPU_ID = 1

K_SPARSE = 5
COSINE_THRESHOLD = 0.1
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 25

# CMI estimation parameters
KNN_K = 5  # k for k-NN MI estimator
SUBSPACE_DIM_CANDIDATES = [10, 20, 30, 50]  # d' candidates
ABSORBED_THRESHOLD = 0.10  # letters with rate > 10% are "absorbed"
NON_ABSORBED_THRESHOLD = 0.05  # letters with rate < 5% are "non-absorbed"

# When CUDA_VISIBLE_DEVICES is set, the visible GPU always appears as cuda:0
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID / Progress / Done ──────────────────────────────────────────────────
pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(step, total_steps, description, extra=None):
    progress = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID, "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "description": description, "metric": extra or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress.write_text(json.dumps(data, indent=2))


def mark_done(status="success", summary="", results=None):
    if pid_file.exists():
        pid_file.unlink()
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if progress_file.exists():
        try: fp = json.loads(progress_file.read_text())
        except: pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "results": results or {}, "final_progress": fp,
        "timestamp": datetime.now().isoformat(),
    }, indent=2))


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Common English words (same as first_letter_validation) ─────────────────
COMMON_WORDS_BY_LETTER = {
    'A': ['apple', 'animal', 'arrow', 'answer', 'album', 'actor', 'anger', 'above', 'alive', 'audio',
          'aunt', 'award', 'avoid', 'aside', 'adult', 'agree', 'alien', 'armor', 'arena', 'angle',
          'angel', 'alert', 'adopt', 'angry', 'alone', 'after', 'allow', 'argue', 'alert', 'awful',
          'accent', 'access', 'active', 'advice', 'affect', 'afford'],
    'B': ['brain', 'brave', 'bread', 'bring', 'brown', 'build', 'buyer', 'badge', 'basic', 'beach',
          'begin', 'bench', 'birth', 'blade', 'blank', 'blast', 'blaze', 'blend', 'blind', 'block',
          'bloom', 'board', 'bonus', 'bound', 'brand', 'break', 'brief', 'broad', 'brush', 'burst',
          'bottle', 'bridge', 'broken', 'bright', 'banner', 'basket'],
    'C': ['chain', 'chair', 'chalk', 'chase', 'cheap', 'check', 'chief', 'child', 'chunk', 'claim',
          'clash', 'class', 'clean', 'clear', 'click', 'climb', 'clock', 'close', 'cloud', 'coach',
          'coast', 'color', 'comic', 'count', 'cover', 'crack', 'craft', 'crash', 'crazy', 'cream',
          'crown', 'cruel', 'crush', 'curve', 'cycle', 'castle'],
    'D': ['dance', 'death', 'debug', 'delay', 'dense', 'depth', 'devil', 'dirty', 'dozen', 'draft',
          'drain', 'drama', 'drawn', 'dream', 'dress', 'dried', 'drift', 'drink', 'drive', 'drone',
          'drugs', 'drunk', 'dying', 'daily', 'dairy', 'danger', 'dealer', 'debate', 'defeat', 'demand',
          'desert', 'device', 'dinner', 'direct', 'double', 'dragon'],
    'E': ['eagle', 'early', 'earth', 'elect', 'elite', 'empty', 'enemy', 'enjoy', 'enter', 'equal',
          'error', 'essay', 'event', 'exact', 'exist', 'extra', 'eager', 'elder', 'email', 'ember',
          'every', 'eight', 'evening', 'escape', 'ethnic', 'expert', 'export', 'extent', 'engine', 'emerge',
          'enable', 'ensure', 'entire', 'entity', 'evolve', 'exceed'],
    'F': ['faced', 'faith', 'false', 'feast', 'fence', 'fiber', 'field', 'fight', 'final', 'flame',
          'flash', 'flesh', 'float', 'flood', 'floor', 'focus', 'force', 'forge', 'forth', 'forum',
          'found', 'frame', 'frank', 'fraud', 'fresh', 'front', 'frost', 'fruit', 'funny', 'fully',
          'factor', 'family', 'father', 'female', 'finger', 'flower'],
    'G': ['giant', 'given', 'glass', 'globe', 'glory', 'grace', 'grade', 'grain', 'grand', 'grant',
          'graph', 'grasp', 'grass', 'grave', 'great', 'green', 'greet', 'grief', 'grill', 'grind',
          'groan', 'groom', 'gross', 'group', 'grown', 'guard', 'guess', 'guest', 'guide', 'guilt',
          'gather', 'gentle', 'global', 'golden', 'govern', 'guitar'],
    'H': ['happy', 'harsh', 'heart', 'heavy', 'hence', 'honor', 'horse', 'hotel', 'house', 'human',
          'humor', 'hurry', 'haven', 'heard', 'hello', 'herbs', 'honey', 'hover', 'habit', 'hedge',
          'height', 'hidden', 'hockey', 'hollow', 'honest', 'horror', 'hunger', 'hunter', 'handle', 'happen',
          'harbor', 'health', 'heaven', 'hereby', 'highly', 'holder'],
    'I': ['ideal', 'image', 'imply', 'index', 'inner', 'input', 'intro', 'irony', 'ivory', 'inbox',
          'issue', 'inter', 'infer', 'imply', 'incur', 'inert', 'input', 'irate', 'ivory', 'inert',
          'ignore', 'immune', 'import', 'impose', 'income', 'indeed', 'inform', 'inject', 'insect', 'insert',
          'inside', 'insist', 'intact', 'intend', 'invent', 'invest'],
    'J': ['jewel', 'joint', 'judge', 'juice', 'jolly', 'joker', 'jumbo', 'jumps', 'jelly', 'jimmy',
          'japan', 'jetty', 'jiffy', 'jimmy', 'juice', 'judge', 'jimmy', 'jumps', 'jewel', 'joint',
          'jacket', 'jungle', 'junior', 'jersey', 'jockey', 'journal'],
    'K': ['kayak', 'kebab', 'knack', 'kneel', 'knife', 'knock', 'known', 'karma', 'kitty', 'knight',
          'kernel', 'keeper', 'kidney', 'killer', 'kindle', 'kitten', 'knight', 'kettle', 'kicker', 'kindly',
          'kingdom', 'kitchen', 'keynote', 'kinetic', 'knowing', 'kindred'],
    'L': ['label', 'laser', 'later', 'layer', 'legal', 'level', 'light', 'limit', 'local', 'logic',
          'loose', 'lover', 'lucky', 'lunch', 'lynch', 'learn', 'leave', 'lemon', 'lever', 'liner',
          'launch', 'leader', 'league', 'legacy', 'length', 'lesson', 'letter', 'likely', 'listen', 'living',
          'longer', 'lovely', 'lowest', 'luxury', 'layout', 'legend'],
    'M': ['magic', 'major', 'maker', 'march', 'match', 'mayor', 'media', 'mercy', 'metal', 'meter',
          'might', 'minor', 'mixed', 'model', 'money', 'moral', 'mount', 'mouse', 'mouth', 'movie',
          'maker', 'manor', 'maple', 'marsh', 'medal', 'merge', 'method', 'middle', 'mighty', 'minute',
          'mirror', 'mobile', 'modern', 'monkey', 'mother', 'murder'],
    'N': ['naked', 'nasty', 'nerve', 'never', 'night', 'noble', 'noise', 'north', 'noted', 'novel',
          'nurse', 'nylon', 'naive', 'naval', 'needs', 'newer', 'nexus', 'ninth', 'noise', 'north',
          'narrow', 'nation', 'nature', 'nearby', 'needle', 'nested', 'neural', 'nickel', 'nobody', 'normal',
          'notice', 'notion', 'number', 'naming', 'native', 'nimble'],
    'O': ['ocean', 'offer', 'often', 'olive', 'onset', 'opera', 'orbit', 'order', 'organ', 'other',
          'outer', 'omega', 'onset', 'opted', 'orbit', 'oxide', 'ozone', 'occur', 'owing', 'overt',
          'object', 'obtain', 'office', 'online', 'oppose', 'orange', 'orphan', 'outfit', 'output', 'oxygen',
          'option', 'origin', 'outage', 'occupy', 'offend', 'onward'],
    'P': ['panel', 'paper', 'paste', 'patch', 'pause', 'peace', 'pearl', 'phase', 'phone', 'photo',
          'piano', 'piece', 'pilot', 'pitch', 'pixel', 'pizza', 'place', 'plain', 'plane', 'plant',
          'plate', 'plaza', 'plead', 'point', 'polar', 'pound', 'power', 'press', 'price', 'pride',
          'prime', 'print', 'prior', 'probe', 'proof', 'pulse'],
    'Q': ['queen', 'query', 'quest', 'queue', 'quick', 'quiet', 'quilt', 'quirk', 'quota', 'quote',
          'quake', 'qualm', 'quart', 'quasi', 'queen', 'query', 'quest', 'queue', 'quick', 'quiet',
          'quarter', 'quality', 'quantum', 'quickly', 'qualify', 'quarter'],
    'R': ['radar', 'raise', 'range', 'rapid', 'ratio', 'reach', 'ready', 'realm', 'rebel', 'refer',
          'reign', 'relax', 'rider', 'rifle', 'rigid', 'rival', 'river', 'robot', 'rocky', 'Roman',
          'royal', 'rugby', 'rural', 'rogue', 'route', 'ruler', 'rabbit', 'racing', 'random', 'rather',
          'reason', 'record', 'reduce', 'reform', 'region', 'relief'],
    'S': ['saint', 'scale', 'scene', 'scope', 'score', 'scout', 'sense', 'serve', 'shade', 'shake',
          'shame', 'shape', 'share', 'sharp', 'shell', 'shift', 'shine', 'shirt', 'shock', 'shoot',
          'short', 'shout', 'sight', 'since', 'skill', 'sleep', 'slice', 'slide', 'smart', 'smile',
          'smoke', 'snake', 'solar', 'solid', 'solve', 'space'],
    'T': ['table', 'taste', 'teach', 'teens', 'theft', 'theme', 'thick', 'thing', 'think', 'those',
          'three', 'throw', 'thumb', 'tiger', 'tight', 'timer', 'tired', 'title', 'toast', 'token',
          'total', 'touch', 'tough', 'tower', 'toxic', 'trace', 'track', 'trade', 'trail', 'train',
          'trait', 'trash', 'treat', 'trend', 'trial', 'trick'],
    'U': ['ultra', 'under', 'union', 'unite', 'unity', 'upper', 'upset', 'urban', 'usage', 'usual',
          'utter', 'uncle', 'unify', 'unlit', 'until', 'usher', 'using', 'unity', 'ultra', 'under',
          'unfair', 'unique', 'united', 'unless', 'unlike', 'update', 'upload', 'upside', 'urgent', 'useful',
          'utmost', 'uphold', 'upbeat', 'unfold', 'unlock', 'unveil'],
    'V': ['valid', 'value', 'vapor', 'vault', 'venue', 'verse', 'video', 'vigor', 'vinyl', 'viola',
          'viral', 'virus', 'visit', 'vital', 'vivid', 'vocal', 'vodka', 'voice', 'voter', 'vowel',
          'victim', 'viewer', 'violin', 'virtue', 'vision', 'visual', 'voyage', 'vacuum', 'valley', 'vanish',
          'varied', 'velvet', 'vendor', 'verbal', 'verify', 'vortex'],
    'W': ['watch', 'water', 'wheat', 'wheel', 'where', 'while', 'white', 'whole', 'width', 'witch',
          'woman', 'world', 'worry', 'worst', 'worth', 'wound', 'wrist', 'write', 'wrong', 'wages',
          'wallet', 'wander', 'warmth', 'weapon', 'wealth', 'weekly', 'weight', 'wicked', 'window', 'winner',
          'winter', 'wisdom', 'wizard', 'wonder', 'worker', 'worthy'],
    'X': ['xenon', 'xerox', 'xenon', 'xenon', 'xerox', 'xerox'],
    'Y': ['yacht', 'yield', 'young', 'youth', 'yearly', 'yellow', 'yogurt', 'yonder'],
    'Z': ['zebra', 'zeros', 'zippy', 'zones', 'zombie', 'zenith', 'zigzag', 'zodiac'],
}


# ── k-NN Mutual Information Estimator (KSG, Method 1) ─────────────────────

def knn_mi_estimate(X, Y, k=5):
    """
    Estimate mutual information I(X; Y) using the KSG estimator (Kraskov et al. 2004).
    X: (n, d_x) array, Y: (n, d_y) array, k: number of nearest neighbors
    Returns: estimated MI in nats.
    """
    n = X.shape[0]
    if n < k + 2:
        return float('nan')

    XY = np.hstack([X, Y])

    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev', algorithm='ball_tree')
    nn_joint.fit(XY)
    dists_joint, _ = nn_joint.kneighbors(XY)
    eps = dists_joint[:, k]  # k-th NN distance

    nn_x = NearestNeighbors(metric='chebyshev', algorithm='ball_tree')
    nn_x.fit(X)
    nn_y = NearestNeighbors(metric='chebyshev', algorithm='ball_tree')
    nn_y.fit(Y)

    n_x = np.zeros(n)
    n_y = np.zeros(n)

    for i in range(n):
        # Add small eps to avoid zero-radius queries
        r = max(eps[i], 1e-10)
        idx_x = nn_x.radius_neighbors([X[i]], radius=r, return_distance=False)[0]
        n_x[i] = max(len(idx_x) - 1, 0)  # subtract self
        idx_y = nn_y.radius_neighbors([Y[i]], radius=r, return_distance=False)[0]
        n_y[i] = max(len(idx_y) - 1, 0)

    # KSG estimator (Method 1): I(X;Y) = psi(k) - <psi(nx+1) + psi(ny+1)> + psi(n)
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return float(mi)


def knn_cmi_estimate(X, Y, Z, k=5):
    """
    Estimate conditional mutual information I(X; Y | Z) using the chain rule:
    I(X; Y | Z) = I(X; Y, Z) - I(X; Z)
    """
    YZ = np.hstack([Y, Z])
    mi_x_yz = knn_mi_estimate(X, YZ, k=k)
    mi_x_z = knn_mi_estimate(X, Z, k=k)
    if np.isnan(mi_x_yz) or np.isnan(mi_x_z):
        return float('nan')
    return float(mi_x_yz - mi_x_z)


def get_single_token_english_words(tokenizer, max_per_letter=25):
    """Get common English words that tokenize as single tokens in Gemma 2."""
    letter_words = {}
    for letter, candidates in COMMON_WORDS_BY_LETTER.items():
        valid = []
        seen = set()
        for word in candidates:
            if word.lower() in seen:
                continue
            seen.add(word.lower())
            for w in [word.lower(), word]:
                encoded = tokenizer.encode(f" {w}", add_special_tokens=False)
                if len(encoded) == 1:
                    valid.append({"word": w, "token_id": encoded[0]})
                    break
            if len(valid) >= max_per_letter:
                break
        letter_words[letter] = valid
    return letter_words


def collect_activations_for_words(model, sae, tokenizer, letter_words, hook_point, device):
    """
    Collect SAE latent activations AND raw model activations for each word.
    Returns: sae_acts [N, d_sae], raw_acts [N, d_model], labels, words
    """
    all_sae_acts = []
    all_raw_acts = []
    all_labels = []
    all_words = []

    for letter in sorted(letter_words.keys()):
        for w_info in letter_words[letter]:
            word = w_info["word"]
            prompt = f" {word}"
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_point],
                    return_type=None, prepend_bos=True,
                )
            raw_act = cache[hook_point][0, -1, :].detach()  # [d_model]

            with torch.no_grad():
                sae_out = sae.encode(raw_act.unsqueeze(0))
            sae_act = sae_out[0].detach()

            all_sae_acts.append(sae_act.cpu())
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(letter)
            all_words.append(word)

            del cache
            torch.cuda.empty_cache()

    sae_acts = torch.stack(all_sae_acts).numpy()
    raw_acts = torch.stack(all_raw_acts).numpy()
    return sae_acts, raw_acts, all_labels, all_words


def compute_cmi_for_letter(letter, split_features, absorption_rate,
                           sae_acts, raw_acts, labels, W_dec_np,
                           subspace_dim=20, k=5, seed=42):
    """
    Compute I(X; f_parent | f_child) for a single letter.

    X: token identity — represented as raw model activation projected into subspace
    f_parent: projection onto the aggregate "parent direction" in model space
              (weighted sum of split feature decoder directions by probe weight)
    f_child: the top-k child latent activations (the SAE's split features)

    Project everything into the decoder-direction subspace for reliable MI estimation.
    """
    labels_arr = np.array(labels)
    pos_mask = labels_arr == letter
    n_pos = pos_mask.sum()
    if n_pos < 5:
        return {"status": "skipped", "reason": f"n_pos={n_pos}<5"}

    # Build decoder-direction subspace using relevant features
    relevant_decoder_dirs = []

    # Add split features' decoder directions
    for sf in split_features:
        relevant_decoder_dirs.append(W_dec_np[sf])

    # Add decoder directions of features most active on this letter's words
    pos_indices = np.where(pos_mask)[0]
    pos_sae_acts = sae_acts[pos_indices]
    mean_acts = pos_sae_acts.mean(axis=0)
    n_extra = max(subspace_dim - len(split_features), 10)
    top_active = np.argsort(mean_acts)[::-1][:n_extra]
    for fi in top_active:
        if mean_acts[fi] > 0:
            relevant_decoder_dirs.append(W_dec_np[fi])

    if len(relevant_decoder_dirs) < 3:
        return {"status": "skipped", "reason": "insufficient decoder directions"}

    # Build subspace via SVD
    D = np.stack(relevant_decoder_dirs)
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    actual_dim = min(subspace_dim, len(S), Vt.shape[0])
    V_sub = Vt[:actual_dim]  # [actual_dim, d_model]

    # X: token identity — project raw activations into subspace
    X_proj = raw_acts @ V_sub.T  # [N, actual_dim]

    # f_parent: the "parent concept direction" in model space
    # Use the first (dominant) split feature's decoder direction as the parent direction
    # This is the feature that would fire if absorption didn't occur
    parent_dir = W_dec_np[split_features[0]].copy()
    parent_norm = np.linalg.norm(parent_dir)
    if parent_norm < 1e-10:
        return {"status": "skipped", "reason": "zero parent direction"}
    parent_dir /= parent_norm

    # Y_parent: scalar projection of raw activations onto parent direction
    Y_parent = (raw_acts @ parent_dir).reshape(-1, 1)  # [N, 1]

    # Z_child: activations of the split features (child latents)
    Z_child = sae_acts[:, split_features]  # [N, k_sparse]

    # Standardize for numerical stability
    n_samples = len(labels)
    X_std = (X_proj - X_proj.mean(axis=0)) / (X_proj.std(axis=0) + 1e-10)
    Y_std = (Y_parent - Y_parent.mean(axis=0)) / (Y_parent.std(axis=0) + 1e-10)
    Z_std = (Z_child - Z_child.mean(axis=0)) / (Z_child.std(axis=0) + 1e-10)

    try:
        cmi = knn_cmi_estimate(X_std, Y_std, Z_std, k=k)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    # Reference: I(X; f_parent) and I(X; f_child)
    try:
        mi_parent = knn_mi_estimate(X_std, Y_std, k=k)
    except:
        mi_parent = float('nan')
    try:
        mi_child = knn_mi_estimate(X_std, Z_std, k=k)
    except:
        mi_child = float('nan')

    return {
        "status": "ok",
        "cmi": float(cmi),
        "mi_parent": float(mi_parent),
        "mi_child": float(mi_child),
        "n_samples": n_samples,
        "n_pos": int(n_pos),
        "subspace_dim_actual": int(actual_dim),
        "n_decoder_dirs": len(relevant_decoder_dirs),
        "absorption_rate": float(absorption_rate),
        "is_finite": bool(np.isfinite(cmi)),
    }


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return float('nan')
    return float((m1 - m2) / pooled_std)


def main():
    set_seeds(SEED)
    start_time = time.time()
    device = DEVICE
    total_steps = 9

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_config": "L12-16k",
        "timestamp_start": datetime.now().isoformat(),
        "parameters": {
            "k_sparse": K_SPARSE,
            "cosine_threshold": COSINE_THRESHOLD,
            "magnitude_gap": MAGNITUDE_GAP,
            "knn_k": KNN_K,
            "subspace_dim_candidates": SUBSPACE_DIM_CANDIDATES,
            "absorbed_threshold": ABSORBED_THRESHOLD,
            "non_absorbed_threshold": NON_ABSORBED_THRESHOLD,
        },
    }

    # ── Step 1: Load absorption data from first_letter_validation ───────
    report_progress(1, total_steps, "Loading first_letter_validation results")
    print("\n[Step 1/9] Loading first_letter_validation results...")

    fl_path = RESULTS_DIR / "first_letter_validation.json"
    if not fl_path.exists():
        raise FileNotFoundError(f"Required dependency not found: {fl_path}")

    fl_data = json.loads(fl_path.read_text())
    fl_l12_16k = fl_data["l12_16k"]

    # Extract per-letter absorption rates and split features from dependency
    per_letter_absorption = {}
    per_letter_split_features = {}
    for letter, info in fl_l12_16k["per_letter"].items():
        if info.get("status") == "ok":
            per_letter_absorption[letter] = info["absorption_rate"]
            per_letter_split_features[letter] = info["split_features"]
            print(f"  {letter}: absorption_rate={info['absorption_rate']:.2f}, "
                  f"probe_f1={info['probe_f1']:.4f}, split_features={info['split_features'][:3]}...")

    # Partition into absorbed / non-absorbed using the VALIDATED absorption rates
    absorbed_letters = sorted([l for l, r in per_letter_absorption.items() if r > ABSORBED_THRESHOLD])
    non_absorbed_letters = sorted([l for l, r in per_letter_absorption.items() if r < NON_ABSORBED_THRESHOLD])
    ambiguous_letters = sorted([l for l, r in per_letter_absorption.items()
                                if NON_ABSORBED_THRESHOLD <= r <= ABSORBED_THRESHOLD])

    print(f"\n  Absorbed (rate>{ABSORBED_THRESHOLD}): {absorbed_letters} ({len(absorbed_letters)})")
    print(f"  Non-absorbed (rate<{NON_ABSORBED_THRESHOLD}): {non_absorbed_letters} ({len(non_absorbed_letters)})")
    print(f"  Ambiguous: {ambiguous_letters} ({len(ambiguous_letters)})")

    results["first_letter_source"] = str(fl_path)
    results["partition"] = {
        "absorbed_letters": absorbed_letters,
        "non_absorbed_letters": non_absorbed_letters,
        "ambiguous_letters": ambiguous_letters,
        "absorbed_rates": {l: per_letter_absorption[l] for l in absorbed_letters},
        "non_absorbed_rates": {l: per_letter_absorption[l] for l in non_absorbed_letters},
        "absorbed_threshold": ABSORBED_THRESHOLD,
        "non_absorbed_threshold": NON_ABSORBED_THRESHOLD,
    }

    # ── Step 2: Load model ──────────────────────────────────────────────
    report_progress(2, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 2/9] Loading Gemma 2 2B model...")
    t0 = time.time()

    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model_name = "unsloth/gemma-2-2b"
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float16)
    model = HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model,
        tokenizer=tokenizer,
        device=device,
        dtype=torch.float16,
    )
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ── Step 3: Load SAE L12-16k ────────────────────────────────────────
    report_progress(3, total_steps, "Loading SAE L12-16k")
    print("\n[Step 3/9] Loading SAE L12-16k...")
    t0 = time.time()

    from sae_lens import SAE
    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device=device,
    )
    print(f"  SAE loaded in {time.time()-t0:.1f}s, features={sae.cfg.d_sae}")

    hook_point = "blocks.12.hook_resid_post"
    W_dec = sae.W_dec.data.cpu()  # [n_features, d_model]
    W_dec_np = W_dec.numpy()

    # ── Step 4: Build vocabulary and collect activations ─────────────────
    report_progress(4, total_steps, "Building vocabulary and collecting activations")
    print("\n[Step 4/9] Building vocabulary and collecting activations...")
    t0 = time.time()

    letter_words = get_single_token_english_words(tokenizer, max_per_letter=MAX_PER_LETTER)
    if 'X' in letter_words and len(letter_words['X']) < 3:
        del letter_words['X']

    total_words = sum(len(v) for v in letter_words.values())
    print(f"  {len(letter_words)} letters, {total_words} words")

    sae_acts, raw_acts, all_labels, all_words = collect_activations_for_words(
        model, sae, tokenizer, letter_words, hook_point, device
    )
    print(f"  Collected {len(all_labels)} activations in {time.time()-t0:.1f}s")
    print(f"  SAE shape: {sae_acts.shape}, Raw shape: {raw_acts.shape}")

    results["vocabulary"] = {
        "n_letters": len(letter_words),
        "total_words": total_words,
        "per_letter_counts": {k: len(v) for k, v in sorted(letter_words.items())},
    }

    # Free GPU memory - we only need CPU numpy arrays from here
    del model, sae
    gc.collect(); torch.cuda.empty_cache()
    print("  GPU memory freed")

    # ── Step 5: Compute CMI for each letter at multiple subspace dims ───
    report_progress(5, total_steps, "Computing CMI for each letter")
    print("\n[Step 5/9] Computing CMI for each letter at multiple subspace dimensions...")

    # Only compute for letters that have both absorption data and sufficient samples
    valid_letters = sorted([l for l in per_letter_absorption.keys()
                            if l in set(all_labels) and l != 'X'])

    cmi_results_by_dim = {}
    best_dim = None
    best_n_valid = 0

    for subspace_dim in SUBSPACE_DIM_CANDIDATES:
        print(f"\n  === Subspace dim d'={subspace_dim} ===")
        t0 = time.time()

        cmi_per_letter = {}
        for letter in valid_letters:
            if letter not in per_letter_split_features:
                cmi_per_letter[letter] = {"status": "skipped", "reason": "no split features"}
                continue

            cmi_result = compute_cmi_for_letter(
                letter=letter,
                split_features=per_letter_split_features[letter],
                absorption_rate=per_letter_absorption[letter],
                sae_acts=sae_acts,
                raw_acts=raw_acts,
                labels=all_labels,
                W_dec_np=W_dec_np,
                subspace_dim=subspace_dim,
                k=KNN_K,
                seed=SEED,
            )
            cmi_per_letter[letter] = cmi_result
            if cmi_result["status"] == "ok":
                print(f"    {letter}: CMI={cmi_result['cmi']:.4f}, "
                      f"MI_parent={cmi_result['mi_parent']:.4f}, "
                      f"MI_child={cmi_result['mi_child']:.4f}, "
                      f"abs_rate={per_letter_absorption[letter]:.2f}")

        n_valid = sum(1 for v in cmi_per_letter.values()
                      if v.get("status") == "ok" and v.get("is_finite", False))
        print(f"  d'={subspace_dim}: {n_valid} valid CMI estimates in {time.time()-t0:.1f}s")

        cmi_results_by_dim[subspace_dim] = cmi_per_letter
        if n_valid > best_n_valid:
            best_n_valid = n_valid
            best_dim = subspace_dim

    results["cmi_by_subspace_dim"] = cmi_results_by_dim
    print(f"\n  Selected best subspace dim: d'={best_dim} ({best_n_valid} valid)")
    results["best_subspace_dim"] = best_dim

    # ── Step 6: Mann-Whitney U test ─────────────────────────────────────
    report_progress(6, total_steps, "Mann-Whitney U test: absorbed vs non-absorbed")
    print("\n[Step 6/9] Mann-Whitney U test...")

    primary_cmi = cmi_results_by_dim[best_dim]

    cmi_absorbed = []
    cmi_non_absorbed = []
    cmi_all = {}

    for letter in valid_letters:
        info = primary_cmi.get(letter, {})
        if info.get("status") != "ok" or not info.get("is_finite", False):
            continue
        cmi_val = info["cmi"]
        cmi_all[letter] = cmi_val
        if letter in absorbed_letters:
            cmi_absorbed.append(cmi_val)
        elif letter in non_absorbed_letters:
            cmi_non_absorbed.append(cmi_val)

    print(f"  Absorbed ({len(cmi_absorbed)}): {[f'{v:.4f}' for v in cmi_absorbed]}")
    print(f"  Non-absorbed ({len(cmi_non_absorbed)}): {[f'{v:.4f}' for v in cmi_non_absorbed]}")

    mann_whitney_result = {}
    if len(cmi_absorbed) >= 2 and len(cmi_non_absorbed) >= 2:
        U_stat, p_value = stats.mannwhitneyu(
            cmi_absorbed, cmi_non_absorbed, alternative='greater'
        )
        d = cohens_d(cmi_absorbed, cmi_non_absorbed)

        print(f"\n  Mann-Whitney U = {U_stat:.2f}, p = {p_value:.6f}")
        print(f"  Cohen's d = {d:.4f}")
        print(f"  Absorbed mean CMI = {np.mean(cmi_absorbed):.4f} +/- {np.std(cmi_absorbed):.4f}")
        print(f"  Non-absorbed mean CMI = {np.mean(cmi_non_absorbed):.4f} +/- {np.std(cmi_non_absorbed):.4f}")

        # Bootstrap CI for difference in means
        rng = np.random.RandomState(SEED)
        abs_arr = np.array(cmi_absorbed)
        na_arr = np.array(cmi_non_absorbed)
        diffs = []
        for _ in range(10000):
            a_sample = rng.choice(abs_arr, size=len(abs_arr), replace=True)
            na_sample = rng.choice(na_arr, size=len(na_arr), replace=True)
            diffs.append(np.mean(a_sample) - np.mean(na_sample))
        ci_low = float(np.percentile(diffs, 2.5))
        ci_high = float(np.percentile(diffs, 97.5))

        mann_whitney_result = {
            "U_statistic": float(U_stat),
            "p_value_one_sided": float(p_value),
            "cohens_d": d,
            "absorbed_mean": float(np.mean(cmi_absorbed)),
            "absorbed_std": float(np.std(cmi_absorbed)),
            "absorbed_median": float(np.median(cmi_absorbed)),
            "non_absorbed_mean": float(np.mean(cmi_non_absorbed)),
            "non_absorbed_std": float(np.std(cmi_non_absorbed)),
            "non_absorbed_median": float(np.median(cmi_non_absorbed)),
            "n_absorbed": len(cmi_absorbed),
            "n_non_absorbed": len(cmi_non_absorbed),
            "significant_at_0_05": bool(p_value < 0.05),
            "large_effect": bool(abs(d) > 0.8 if np.isfinite(d) else False),
            "mean_diff_bootstrap_ci_95": [ci_low, ci_high],
        }
        print(f"  Mean diff CI (95%): [{ci_low:.4f}, {ci_high:.4f}]")
    else:
        print(f"  INSUFFICIENT DATA: absorbed={len(cmi_absorbed)}, non_absorbed={len(cmi_non_absorbed)}")
        mann_whitney_result = {
            "status": "insufficient_data",
            "n_absorbed": len(cmi_absorbed),
            "n_non_absorbed": len(cmi_non_absorbed),
            "absorbed_values": [float(v) for v in cmi_absorbed],
            "non_absorbed_values": [float(v) for v in cmi_non_absorbed],
        }

    results["mann_whitney_test"] = mann_whitney_result

    # ── Step 7: Spearman correlation: CMI vs absorption rate ────────────
    report_progress(7, total_steps, "Spearman correlation: CMI vs absorption rate")
    print("\n[Step 7/9] Spearman correlation: CMI vs absorption rate...")

    corr_letters = []
    corr_cmi = []
    corr_rates = []
    for letter in sorted(cmi_all.keys()):
        if letter in per_letter_absorption:
            corr_letters.append(letter)
            corr_cmi.append(cmi_all[letter])
            corr_rates.append(per_letter_absorption[letter])

    correlation_result = {}
    if len(corr_letters) >= 5:
        rho, p_val = stats.spearmanr(corr_rates, corr_cmi)
        pearson_r, pearson_p = stats.pearsonr(corr_rates, corr_cmi)

        print(f"  N = {len(corr_letters)} letters")
        print(f"  Spearman rho = {rho:.4f}, p = {p_val:.6f}")
        print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")

        correlation_result = {
            "n_letters": len(corr_letters),
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "letters": corr_letters,
            "cmi_values": [float(x) for x in corr_cmi],
            "absorption_rates": [float(x) for x in corr_rates],
        }
    else:
        correlation_result = {"status": "insufficient_data", "n_letters": len(corr_letters)}

    results["correlation_cmi_vs_absorption"] = correlation_result

    # ── Step 8: Summary table and subspace sensitivity ──────────────────
    report_progress(8, total_steps, "Building summary and sensitivity analysis")
    print("\n[Step 8/9] Summary table...")

    summary_table = []
    for letter in sorted(per_letter_absorption.keys()):
        row = {
            "letter": letter,
            "absorption_rate": per_letter_absorption[letter],
            "group": "absorbed" if letter in absorbed_letters else
                     ("non_absorbed" if letter in non_absorbed_letters else "ambiguous"),
        }
        if letter in cmi_all:
            row["cmi"] = cmi_all[letter]
            info = primary_cmi.get(letter, {})
            row["mi_parent"] = info.get("mi_parent")
            row["mi_child"] = info.get("mi_child")
        else:
            row["cmi"] = None
        summary_table.append(row)
        if row.get("cmi") is not None:
            print(f"  {letter}: rate={row['absorption_rate']:.2f}, CMI={row['cmi']:.4f}, "
                  f"group={row['group']}")

    results["summary_table"] = summary_table

    # Subspace dimension sensitivity
    dim_sensitivity = {}
    for dim, cmi_per_letter in cmi_results_by_dim.items():
        abs_vals = [cmi_per_letter[l]["cmi"] for l in absorbed_letters
                    if l in cmi_per_letter and cmi_per_letter[l].get("status") == "ok"
                    and cmi_per_letter[l].get("is_finite", False)]
        non_abs_vals = [cmi_per_letter[l]["cmi"] for l in non_absorbed_letters
                        if l in cmi_per_letter and cmi_per_letter[l].get("status") == "ok"
                        and cmi_per_letter[l].get("is_finite", False)]

        # Also compute correlation at this dim
        dim_letters = []
        dim_cmi_vals = []
        dim_rates = []
        for l in valid_letters:
            if l in cmi_per_letter and cmi_per_letter[l].get("status") == "ok" and \
               cmi_per_letter[l].get("is_finite", False) and l in per_letter_absorption:
                dim_letters.append(l)
                dim_cmi_vals.append(cmi_per_letter[l]["cmi"])
                dim_rates.append(per_letter_absorption[l])

        rho_dim, p_dim = (float('nan'), float('nan'))
        if len(dim_letters) >= 5:
            rho_dim, p_dim = stats.spearmanr(dim_rates, dim_cmi_vals)

        if len(abs_vals) >= 2 and len(non_abs_vals) >= 2:
            u, p = stats.mannwhitneyu(abs_vals, non_abs_vals, alternative='greater')
            d = cohens_d(abs_vals, non_abs_vals)
        else:
            u, p, d = float('nan'), float('nan'), float('nan')

        dim_sensitivity[dim] = {
            "n_absorbed_valid": len(abs_vals),
            "n_non_absorbed_valid": len(non_abs_vals),
            "U_statistic": float(u) if np.isfinite(u) else None,
            "p_value": float(p) if np.isfinite(p) else None,
            "cohens_d": float(d) if np.isfinite(d) else None,
            "absorbed_mean": float(np.mean(abs_vals)) if abs_vals else None,
            "non_absorbed_mean": float(np.mean(non_abs_vals)) if non_abs_vals else None,
            "spearman_rho": float(rho_dim) if np.isfinite(rho_dim) else None,
            "spearman_p": float(p_dim) if np.isfinite(p_dim) else None,
        }

    results["subspace_dim_sensitivity"] = dim_sensitivity
    print("\n  Subspace dimension sensitivity:")
    for dim, info in dim_sensitivity.items():
        u_str = f"{info['U_statistic']:.1f}" if info['U_statistic'] is not None else "N/A"
        p_str = f"{info['p_value']:.4f}" if info['p_value'] is not None else "N/A"
        d_str = f"{info['cohens_d']:.3f}" if info['cohens_d'] is not None else "N/A"
        rho_str = f"{info['spearman_rho']:.3f}" if info['spearman_rho'] is not None else "N/A"
        print(f"    d'={dim}: U={u_str}, p={p_str}, d={d_str}, rho={rho_str}")

    # ── Step 9: Cross-domain extension check ────────────────────────────
    report_progress(9, total_steps, "Cross-domain extension check")
    print("\n[Step 9/9] Cross-domain extension check...")

    cross_domain_files = {
        "city_country": RESULTS_DIR / "cross_domain_city_country.json",
        "city_continent": RESULTS_DIR / "cross_domain_city_continent.json",
        "city_language": RESULTS_DIR / "cross_domain_city_language.json",
        "animal_class": RESULTS_DIR / "cross_domain_animal_class.json",
    }
    cross_domain_extension = {}
    for domain, filepath in cross_domain_files.items():
        if filepath.exists():
            try:
                cd_data = json.loads(filepath.read_text())
                agg_key = None
                if "l12_16k" in cd_data and "aggregate" in cd_data["l12_16k"]:
                    agg_key = "l12_16k"
                elif "aggregate" in cd_data:
                    agg_key = None
                if agg_key:
                    agg = cd_data[agg_key]["aggregate"]
                    cross_domain_extension[domain] = {
                        "available": True,
                        "aggregate_absorption_rate": agg.get("aggregate_absorption_rate"),
                        "note": "CMI extension deferred; first-letter validates methodology first"
                    }
                else:
                    cross_domain_extension[domain] = {"available": True, "note": "results present but format differs"}
            except Exception as e:
                cross_domain_extension[domain] = {"available": False, "error": str(e)}
        else:
            cross_domain_extension[domain] = {"available": False}
    results["cross_domain_extension"] = cross_domain_extension

    # ── Pass criteria ───────────────────────────────────────────────────
    n_valid_cmi = sum(1 for row in summary_table if row.get("cmi") is not None
                      and np.isfinite(row["cmi"]))
    cmi_values = [row["cmi"] for row in summary_table
                  if row.get("cmi") is not None and np.isfinite(row["cmi"])]
    all_finite = all(np.isfinite(v) for v in cmi_values) if cmi_values else False

    mw_p = mann_whitney_result.get("p_value_one_sided", 1.0)
    mw_d = mann_whitney_result.get("cohens_d", 0.0)
    if not isinstance(mw_p, (int, float)) or not np.isfinite(mw_p):
        mw_p = 1.0
    if not isinstance(mw_d, (int, float)) or not np.isfinite(mw_d):
        mw_d = 0.0

    spearman_rho = correlation_result.get("spearman_rho", 0.0)
    if not isinstance(spearman_rho, (int, float)) or not np.isfinite(spearman_rho):
        spearman_rho = 0.0

    pass_criteria = {
        "cmi_at_least_20_letters": n_valid_cmi >= 20,
        "n_valid_cmi": n_valid_cmi,
        "mann_whitney_p_lt_0_05": bool(mw_p < 0.05),
        "p_value": float(mw_p),
        "cohens_d_gt_0_5": bool(abs(mw_d) > 0.5),
        "cohens_d": float(mw_d),
        "cmi_finite": all_finite,
        "spearman_rho_with_absorption": float(spearman_rho),
        "overall_pass": (n_valid_cmi >= 20 and mw_p < 0.05 and abs(mw_d) > 0.5 and all_finite),
    }
    results["pass_criteria"] = pass_criteria

    print(f"\n{'='*60}")
    print(f"PASS CRITERIA:")
    print(f"  CMI >= 20 letters: {'PASS' if pass_criteria['cmi_at_least_20_letters'] else 'FAIL'} "
          f"({n_valid_cmi}/20)")
    print(f"  Mann-Whitney p < 0.05: {'PASS' if pass_criteria['mann_whitney_p_lt_0_05'] else 'FAIL'} "
          f"(p={mw_p:.6f})")
    print(f"  Cohen's d > 0.5: {'PASS' if pass_criteria['cohens_d_gt_0_5'] else 'FAIL'} "
          f"(d={mw_d:.4f})")
    print(f"  CMI finite: {'PASS' if pass_criteria['cmi_finite'] else 'FAIL'}")
    print(f"  Spearman rho(CMI, absorption): {spearman_rho:.4f}")
    print(f"  OVERALL: {'PASS' if pass_criteria['overall_pass'] else 'FAIL'}")
    print(f"{'='*60}")

    # ── Finalize ────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 2)
    results["timestamp_end"] = datetime.now().isoformat()

    output_path = RESULTS_DIR / "successive_refinement.json"
    clean_results = json.loads(json.dumps(results, default=str))
    output_path.write_text(json.dumps(clean_results, indent=2))
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {elapsed:.1f}s")

    # Update gpu_progress.json
    try:
        gp_path = WORKSPACE / "exp" / "gpu_progress.json"
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {
            "completed": [], "failed": [], "running": {}, "timings": {}
        }
        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        gp["timings"][TASK_ID] = {
            "planned_min": 30,
            "actual_min": int(round(elapsed / 60)),
            "start_time": results["timestamp_start"],
            "end_time": results["timestamp_end"],
            "config_snapshot": {
                "model": "gemma-2-2b",
                "sae_config": "L12-16k",
                "n_letters": len(per_letter_absorption),
                "knn_k": KNN_K,
                "best_subspace_dim": best_dim,
                "n_valid_cmi": n_valid_cmi,
                "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                "gpu_count": 1,
            },
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print("  gpu_progress.json updated")
    except Exception as e:
        print(f"  Warning: could not update gpu_progress.json: {e}")

    # Mark done
    summary_msg = (
        f"CMI computed for {n_valid_cmi}/25 letters. "
        f"Mann-Whitney p={mw_p:.4f}, Cohen's d={mw_d:.3f}, "
        f"Spearman rho(CMI,absorption)={spearman_rho:.3f}. "
        f"Absorbed letters: {absorbed_letters}. "
        f"{'PASS' if pass_criteria['overall_pass'] else 'FAIL'}"
    )
    mark_done(
        status="success" if pass_criteria["overall_pass"] else "completed_with_caveats",
        summary=summary_msg,
        results={
            "n_valid_cmi": n_valid_cmi,
            "mann_whitney_p": float(mw_p),
            "cohens_d": float(mw_d),
            "spearman_rho": float(spearman_rho),
            "overall_pass": pass_criteria["overall_pass"],
            "absorbed_letters": absorbed_letters,
            "non_absorbed_letters": non_absorbed_letters,
        },
    )

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
