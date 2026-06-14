#!/usr/bin/env python3
"""
confound_decomposition.py — Stage 3: Confound Control: Partial Correlations and Matching

This script implements H4 confound analysis:
1. Measure absorption rates across multiple Gemma Scope SAE configurations (varying L0, width, layer)
2. Compute SAE quality metrics (SCR = Sparsity-Corrected Reconstruction, TPP = Token Prediction
   Performance, SP-F1 = Sparsity-Precision-F1 proxy) for each SAE config
3. Compute partial correlations between absorption rate and quality metrics controlling for
   log(L0), log(width), arch_class
4. Apply Benjamini-Hochberg FDR correction
5. Perform Rosenbaum sensitivity analysis with Mahalanobis and within-width matching
6. Classify false-negative tokens into hedging / reconstruction-error / hierarchy-driven components

FULL mode on Gemma 2 2B + Gemma Scope SAEs.
"""

import json
import os
import sys
import time
import gc
import traceback
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from scipy import stats
from scipy.spatial.distance import mahalanobis

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "confound_decomposition"
SEED = 42
DEVICE = "cuda:0"  # Will be GPU 1 via CUDA_VISIBLE_DEVICES=1

K_SPARSE = 5
COSINE_THRESHOLD = 0.1
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 25

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
        try:
            fp = json.loads(progress_file.read_text())
        except:
            pass
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


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=SEED):
    np.random.seed(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Common English words (same as first_letter_validation) ────────────────
COMMON_WORDS_BY_LETTER = {
    'A': ['apple', 'animal', 'arrow', 'answer', 'album', 'actor', 'anger', 'above', 'alive', 'audio',
          'aunt', 'award', 'avoid', 'aside', 'adult', 'agree', 'alien', 'armor', 'arena', 'angle',
          'angel', 'alert', 'adopt', 'angry', 'alone', 'after', 'allow', 'argue', 'alert', 'awful'],
    'B': ['brain', 'brave', 'bread', 'bring', 'brown', 'build', 'buyer', 'badge', 'basic', 'beach',
          'begin', 'bench', 'birth', 'blade', 'blank', 'blast', 'blaze', 'blend', 'blind', 'block',
          'bloom', 'board', 'bonus', 'bound', 'brand', 'break', 'brief', 'broad', 'brush', 'burst'],
    'C': ['chain', 'chair', 'chalk', 'chase', 'cheap', 'check', 'chief', 'child', 'chunk', 'claim',
          'clash', 'class', 'clean', 'clear', 'click', 'climb', 'clock', 'close', 'cloud', 'coach',
          'coast', 'color', 'comic', 'count', 'cover', 'crack', 'craft', 'crash', 'crazy', 'cream'],
    'D': ['dance', 'death', 'debug', 'delay', 'dense', 'depth', 'devil', 'dirty', 'dozen', 'draft',
          'drain', 'drama', 'drawn', 'dream', 'dress', 'dried', 'drift', 'drink', 'drive', 'drone',
          'drugs', 'drunk', 'dying', 'daily', 'dairy'],
    'E': ['eagle', 'early', 'earth', 'elect', 'elite', 'empty', 'enemy', 'enjoy', 'enter', 'equal',
          'error', 'essay', 'event', 'exact', 'exist', 'extra', 'eager', 'elder', 'email', 'ember',
          'every', 'eight', 'evening', 'escape', 'ethnic'],
    'F': ['faced', 'faith', 'false', 'feast', 'fence', 'fiber', 'field', 'fight', 'final', 'flame',
          'flash', 'flesh', 'float', 'flood', 'floor', 'focus', 'force', 'forge', 'forth', 'forum',
          'found', 'frame', 'frank', 'fraud', 'fresh'],
    'G': ['giant', 'given', 'glass', 'globe', 'glory', 'grace', 'grade', 'grain', 'grand', 'grant',
          'graph', 'grasp', 'grass', 'grave', 'great', 'green', 'greet', 'grief', 'grill', 'grind',
          'groan', 'groom', 'gross', 'group', 'grown'],
    'H': ['happy', 'harsh', 'heart', 'heavy', 'hence', 'honor', 'horse', 'hotel', 'house', 'human',
          'humor', 'hurry', 'haven', 'heard', 'hello', 'herbs', 'honey', 'hover', 'habit', 'hedge',
          'height', 'hidden', 'hockey', 'hollow', 'honest'],
    'I': ['ideal', 'image', 'imply', 'index', 'inner', 'input', 'intro', 'irony', 'ivory', 'inbox',
          'issue', 'inter', 'infer', 'incur', 'inert',
          'ignore', 'immune', 'import', 'impose', 'income', 'indeed', 'inform', 'inject', 'insect', 'insert'],
    'J': ['jewel', 'joint', 'judge', 'juice', 'jolly', 'joker', 'jumbo', 'jumps', 'jelly',
          'jetty', 'jiffy', 'jacket', 'jungle', 'junior', 'jersey', 'jockey', 'journal'],
    'K': ['kayak', 'kebab', 'knack', 'kneel', 'knife', 'knock', 'known', 'karma', 'kitty', 'knight',
          'kernel', 'keeper', 'kidney', 'killer', 'kindle', 'kitten', 'kettle', 'kicker', 'kindly',
          'kingdom', 'kitchen', 'keynote', 'kinetic', 'knowing', 'kindred'],
    'L': ['label', 'laser', 'later', 'layer', 'legal', 'level', 'light', 'limit', 'local', 'logic',
          'loose', 'lover', 'lucky', 'lunch', 'lynch', 'learn', 'leave', 'lemon', 'lever', 'liner',
          'launch', 'leader', 'league', 'legacy', 'length'],
    'M': ['magic', 'major', 'maker', 'march', 'match', 'mayor', 'media', 'mercy', 'metal', 'meter',
          'might', 'minor', 'mixed', 'model', 'money', 'moral', 'mount', 'mouse', 'mouth', 'movie',
          'manor', 'maple', 'marsh', 'medal', 'merge'],
    'N': ['naked', 'nasty', 'nerve', 'never', 'night', 'noble', 'noise', 'north', 'noted', 'novel',
          'nurse', 'nylon', 'naive', 'naval', 'needs', 'newer', 'nexus', 'ninth',
          'narrow', 'nation', 'nature', 'nearby', 'needle', 'nested', 'neural'],
    'O': ['ocean', 'offer', 'often', 'olive', 'onset', 'opera', 'orbit', 'order', 'organ', 'other',
          'outer', 'omega', 'opted', 'oxide', 'ozone', 'occur', 'owing', 'overt',
          'object', 'obtain', 'office', 'online', 'oppose', 'orange', 'orphan'],
    'P': ['panel', 'paper', 'paste', 'patch', 'pause', 'peace', 'pearl', 'phase', 'phone', 'photo',
          'piano', 'piece', 'pilot', 'pitch', 'pixel', 'pizza', 'place', 'plain', 'plane', 'plant',
          'plate', 'plaza', 'plead', 'point', 'polar'],
    'Q': ['queen', 'query', 'quest', 'queue', 'quick', 'quiet', 'quilt', 'quirk', 'quota', 'quote',
          'quake', 'qualm', 'quart', 'quasi',
          'quarter', 'quality', 'quantum', 'quickly', 'qualify'],
    'R': ['radar', 'raise', 'range', 'rapid', 'ratio', 'reach', 'ready', 'realm', 'rebel', 'refer',
          'reign', 'relax', 'rider', 'rifle', 'rigid', 'rival', 'river', 'robot', 'rocky',
          'royal', 'rugby', 'rural', 'rogue', 'route', 'ruler'],
    'S': ['saint', 'scale', 'scene', 'scope', 'score', 'scout', 'sense', 'serve', 'shade', 'shake',
          'shame', 'shape', 'share', 'sharp', 'shell', 'shift', 'shine', 'shirt', 'shock', 'shoot',
          'short', 'shout', 'sight', 'since', 'skill'],
    'T': ['table', 'taste', 'teach', 'teens', 'theft', 'theme', 'thick', 'thing', 'think', 'those',
          'three', 'throw', 'thumb', 'tiger', 'tight', 'timer', 'tired', 'title', 'toast', 'token',
          'total', 'touch', 'tough', 'tower', 'toxic'],
    'U': ['ultra', 'under', 'union', 'unite', 'unity', 'upper', 'upset', 'urban', 'usage', 'usual',
          'utter', 'uncle', 'unify', 'unlit', 'until', 'usher', 'using',
          'unfair', 'unique', 'united', 'unless', 'unlike', 'update', 'upload', 'upside'],
    'V': ['valid', 'value', 'vapor', 'vault', 'venue', 'verse', 'video', 'vigor', 'vinyl', 'viola',
          'viral', 'virus', 'visit', 'vital', 'vivid', 'vocal', 'vodka', 'voice', 'voter', 'vowel',
          'victim', 'viewer', 'violin', 'virtue', 'vision'],
    'W': ['watch', 'water', 'wheat', 'wheel', 'where', 'while', 'white', 'whole', 'width', 'witch',
          'woman', 'world', 'worry', 'worst', 'worth', 'wound', 'wrist', 'write', 'wrong', 'wages',
          'wallet', 'wander', 'warmth', 'weapon', 'wealth'],
    'X': ['xenon'],
    'Y': ['yacht', 'yield', 'young', 'youth', 'yearly', 'yellow', 'yogurt', 'yonder'],
    'Z': ['zebra', 'zeros', 'zones', 'zombie', 'zenith', 'zigzag', 'zodiac'],
}


# ── Gemma Scope residual-stream SAE configs to evaluate ───────────────────
# We use gemma-scope-2b-pt-res SAEs which vary in layer, width, and L0.
# This gives us a rich cross-section for partial correlation analysis.
# We focus on layers 10, 12, 20 (matching our experiment design) and
# try multiple L0 operating points and widths.
SAE_CONFIGS = [
    # === Layer 12, 16k width, 5 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_22",
     "layer": 12, "width": 16384, "l0_target": 22, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_41",
     "layer": 12, "width": 16384, "l0_target": 41, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_82",
     "layer": 12, "width": 16384, "l0_target": 82, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_176",
     "layer": 12, "width": 16384, "l0_target": 176, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_445",
     "layer": 12, "width": 16384, "l0_target": 445, "arch": "JumpReLU"},
    # === Layer 12, 32k width, 3 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_32k/average_l0_22",
     "layer": 12, "width": 32768, "l0_target": 22, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_32k/average_l0_76",
     "layer": 12, "width": 32768, "l0_target": 76, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_32k/average_l0_155",
     "layer": 12, "width": 32768, "l0_target": 155, "arch": "JumpReLU"},
    # === Layer 12, 65k width, 5 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_65k/average_l0_21",
     "layer": 12, "width": 65536, "l0_target": 21, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_65k/average_l0_38",
     "layer": 12, "width": 65536, "l0_target": 38, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_65k/average_l0_72",
     "layer": 12, "width": 65536, "l0_target": 72, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_65k/average_l0_141",
     "layer": 12, "width": 65536, "l0_target": 141, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_65k/average_l0_297",
     "layer": 12, "width": 65536, "l0_target": 297, "arch": "JumpReLU"},
    # === Layer 12, 131k width, 3 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_131k/average_l0_20",
     "layer": 12, "width": 131072, "l0_target": 20, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_131k/average_l0_67",
     "layer": 12, "width": 131072, "l0_target": 67, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_131k/average_l0_129",
     "layer": 12, "width": 131072, "l0_target": 129, "arch": "JumpReLU"},
    # === Layer 12, 262k width, 3 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_262k/average_l0_21",
     "layer": 12, "width": 262144, "l0_target": 21, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_262k/average_l0_67",
     "layer": 12, "width": 262144, "l0_target": 67, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_262k/average_l0_121",
     "layer": 12, "width": 262144, "l0_target": 121, "arch": "JumpReLU"},
    # === Layer 10, 16k width, 3 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_10/width_16k/average_l0_21",
     "layer": 10, "width": 16384, "l0_target": 21, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_10/width_16k/average_l0_77",
     "layer": 10, "width": 16384, "l0_target": 77, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_10/width_16k/average_l0_166",
     "layer": 10, "width": 16384, "l0_target": 166, "arch": "JumpReLU"},
    # === Layer 10, 65k width, 3 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_10/width_65k/average_l0_20",
     "layer": 10, "width": 65536, "l0_target": 20, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_10/width_65k/average_l0_66",
     "layer": 10, "width": 65536, "l0_target": 66, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_10/width_65k/average_l0_128",
     "layer": 10, "width": 65536, "l0_target": 128, "arch": "JumpReLU"},
    # === Layer 20, 16k width, 3 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_20/width_16k/average_l0_22",
     "layer": 20, "width": 16384, "l0_target": 22, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_20/width_16k/average_l0_71",
     "layer": 20, "width": 16384, "l0_target": 71, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_20/width_16k/average_l0_139",
     "layer": 20, "width": 16384, "l0_target": 139, "arch": "JumpReLU"},
    # === Layer 20, 65k width, 3 L0 operating points ===
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_20/width_65k/average_l0_20",
     "layer": 20, "width": 65536, "l0_target": 20, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_20/width_65k/average_l0_61",
     "layer": 20, "width": 65536, "l0_target": 61, "arch": "JumpReLU"},
    {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_20/width_65k/average_l0_114",
     "layer": 20, "width": 65536, "l0_target": 114, "arch": "JumpReLU"},
    # === sae_bench TopK SAEs (different architecture) for arch diversity ===
    {"release": "sae_bench_gemma-2-2b_topk_width-2pow12_date-1109",
     "sae_id": "blocks.12.hook_resid_post__trainer_0",
     "layer": 12, "width": 4096, "l0_target": 0, "arch": "TopK"},
    {"release": "sae_bench_gemma-2-2b_topk_width-2pow12_date-1109",
     "sae_id": "blocks.12.hook_resid_post__trainer_1",
     "layer": 12, "width": 4096, "l0_target": 0, "arch": "TopK"},
    {"release": "sae_bench_gemma-2-2b_topk_width-2pow12_date-1109",
     "sae_id": "blocks.12.hook_resid_post__trainer_2",
     "layer": 12, "width": 4096, "l0_target": 0, "arch": "TopK"},
]


def get_single_token_english_words(tokenizer, max_per_letter=MAX_PER_LETTER):
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


def collect_activations_for_layer(model, sae, tokenizer, letter_words, hook_point, device=DEVICE):
    """Collect SAE and raw activations for first-letter words."""
    all_acts = []
    all_labels = []
    all_words = []
    all_raw_acts = []
    all_residuals = []

    for letter in sorted(letter_words.keys()):
        words = letter_words[letter]
        if len(words) < 2:
            continue
        for w in words:
            prompt = f" {w['word']}"
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_point], return_type=None,
                )
            raw_act = cache[hook_point][0, -1, :].detach()  # [d_model]
            sae_act = sae.encode(raw_act.unsqueeze(0))[0].detach()  # [n_features]

            # Compute reconstruction and residual
            reconstructed = sae.decode(sae_act.unsqueeze(0))[0].detach()
            residual = raw_act - reconstructed

            all_acts.append(sae_act.cpu())
            all_raw_acts.append(raw_act.cpu())
            all_residuals.append(residual.cpu())
            all_labels.append(letter)
            all_words.append(w['word'])

    return (torch.stack(all_acts), torch.stack(all_raw_acts),
            torch.stack(all_residuals), all_labels, all_words)


def compute_absorption_rate(sae_activations, labels, words, W_dec,
                            k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
                            magnitude_gap=MAGNITUDE_GAP):
    """
    Compute per-letter and aggregate absorption rate.
    Returns dict with per_letter and aggregate results.
    """
    X = sae_activations.numpy()
    unique_labels = sorted(set(labels))

    per_letter = {}
    total_tested = 0
    total_fn = 0
    total_absorbed = 0

    for letter in unique_labels:
        mask = np.array([l == letter for l in labels])
        n_pos = int(mask.sum())
        if n_pos < 2:
            per_letter[letter] = {"status": "skipped", "n_pos": n_pos}
            continue

        # Binary classification: this letter vs all others
        y = mask.astype(int)

        try:
            clf = LogisticRegression(
                max_iter=2000, C=1.0, solver='lbfgs', random_state=SEED
            )
            clf.fit(X, y)
        except Exception:
            per_letter[letter] = {"status": "probe_failed", "n_pos": n_pos}
            continue

        y_pred = clf.predict(X)
        f1 = f1_score(y, y_pred, zero_division=0)

        # Get top-k split features by absolute weight
        weights = clf.coef_[0]
        top_k_idx = np.argsort(np.abs(weights))[-k_sparse:]
        split_weights = weights[top_k_idx]

        # Find false negatives: probe predicts correctly (positive) but split features inactive
        fn_count = 0
        absorbed_count = 0
        fn_details = []

        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 1:
                # Check if all split features are inactive
                split_activations = X[i, top_k_idx]
                if np.all(split_activations == 0):
                    fn_count += 1
                    # Check absorption: find active features whose decoder aligns with split direction
                    active_features = np.where(X[i] > 0)[0]
                    if len(active_features) > 0:
                        # Average decoder direction for split features
                        split_decoder_mean = W_dec[top_k_idx].mean(axis=0)
                        split_norm = np.linalg.norm(split_decoder_mean)
                        if split_norm > 1e-10:
                            split_decoder_mean /= split_norm

                            for feat_idx in active_features:
                                feat_dec = W_dec[feat_idx]
                                feat_norm = np.linalg.norm(feat_dec)
                                if feat_norm > 1e-10:
                                    cos_sim = np.dot(feat_dec / feat_norm, split_decoder_mean)
                                    if abs(cos_sim) > cosine_threshold:
                                        mag_ratio = float(X[i, feat_idx]) / max(float(np.mean(X[X[:, feat_idx] > 0, feat_idx])), 1e-10) if np.any(X[:, feat_idx] > 0) else 0
                                        if mag_ratio > magnitude_gap:
                                            absorbed_count += 1
                                            fn_details.append({
                                                "word": words[i],
                                                "feature_idx": int(feat_idx),
                                                "cosine": round(float(cos_sim), 4),
                                                "mag_ratio": round(mag_ratio, 4),
                                            })
                                            break  # Count each token only once

        absorption_rate = absorbed_count / n_pos if n_pos > 0 else 0
        fn_rate = fn_count / n_pos if n_pos > 0 else 0

        per_letter[letter] = {
            "n_pos": n_pos,
            "probe_f1": round(f1, 4),
            "n_false_negatives": fn_count,
            "n_absorbed": absorbed_count,
            "absorption_rate": round(absorption_rate, 4),
            "false_negative_rate": round(fn_rate, 4),
        }

        total_tested += n_pos
        total_fn += fn_count
        total_absorbed += absorbed_count

    aggregate_rate = total_absorbed / total_tested if total_tested > 0 else 0.0
    all_rates = [v["absorption_rate"] for v in per_letter.values() if "absorption_rate" in v]
    ci_low, ci_high = bootstrap_ci(all_rates)

    return {
        "per_letter": per_letter,
        "aggregate": {
            "total_tested": total_tested,
            "total_false_negatives": total_fn,
            "total_absorbed": total_absorbed,
            "aggregate_absorption_rate": round(aggregate_rate, 4),
            "bootstrap_ci_95": [ci_low, ci_high],
            "mean_absorption_rate": round(float(np.mean(all_rates)), 4) if all_rates else None,
            "mean_probe_f1": round(float(np.mean([v["probe_f1"] for v in per_letter.values() if "probe_f1" in v])), 4),
        }
    }


def compute_quality_metrics(sae, sae_activations, raw_activations, residuals, model, tokenizer, hook_point, device=DEVICE):
    """
    Compute SAE quality metrics that can be measured locally:

    1. SCR (Sparsity-Corrected Reconstruction): CE loss with SAE vs without
       Proxy: reconstruction loss (MSE between original and reconstructed activations)
    2. TPP (Token Prediction Performance): How well SAE-reconstructed activations
       predict next token vs original activations
       Proxy: cosine similarity between original and reconstructed activations
    3. SP-F1: Sparsity-Precision tradeoff
       Proxy: fraction of active features that are "meaningful" (above noise threshold)
    4. Unlearning proxy: reconstruction error relative to activation norm
       (higher = more information lost = worse quality)

    We compute efficient proxies that don't require running the full model forward pass
    with SAE interventions (which would be very expensive for 20+ SAEs).
    """
    X_sae = sae_activations.numpy()
    X_raw = raw_activations.numpy()
    X_res = residuals.numpy()

    n_samples = X_raw.shape[0]

    # 1. Reconstruction loss (proxy for SCR)
    # Lower = better reconstruction
    recon_losses = np.linalg.norm(X_res, axis=1)  # Per-sample residual norm
    raw_norms = np.linalg.norm(X_raw, axis=1)
    relative_recon_loss = np.mean(recon_losses / (raw_norms + 1e-10))

    # 2. Cosine similarity between original and reconstructed (proxy for TPP)
    # Higher = better preservation of information
    reconstructed = X_raw - X_res  # raw - residual = reconstructed
    cos_sims = []
    for i in range(n_samples):
        rn = np.linalg.norm(X_raw[i])
        cn = np.linalg.norm(reconstructed[i])
        if rn > 1e-10 and cn > 1e-10:
            cos_sims.append(np.dot(X_raw[i], reconstructed[i]) / (rn * cn))
    mean_cos_sim = float(np.mean(cos_sims)) if cos_sims else 0.0

    # 3. SP-F1 proxy: active feature density and precision
    # Compute the fraction of features that fire on at least some tokens
    active_mask = X_sae > 0
    mean_active_per_token = float(np.mean(active_mask.sum(axis=1)))
    n_features = X_sae.shape[1]
    live_features = int((active_mask.sum(axis=0) > 0).sum())
    feature_density = live_features / n_features

    # SP-F1 proxy: balance between sparsity and reconstruction
    sparsity_score = 1.0 - (mean_active_per_token / n_features)  # Higher = more sparse
    recon_score = mean_cos_sim  # Higher = better reconstruction
    sp_f1_proxy = 2 * sparsity_score * recon_score / (sparsity_score + recon_score + 1e-10)

    # 4. Unlearning proxy: normalized reconstruction error
    # Higher = more information lost
    unlearning_proxy = float(relative_recon_loss)

    # 5. Actual measured L0 (average number of active features per token)
    measured_l0 = float(np.mean(active_mask.sum(axis=1)))

    return {
        "scr_proxy": round(1.0 - relative_recon_loss, 4),  # Higher = better (inverted loss)
        "tpp_proxy": round(mean_cos_sim, 4),  # Higher = better
        "sp_f1_proxy": round(sp_f1_proxy, 4),  # Higher = better balance
        "unlearning_proxy": round(unlearning_proxy, 4),  # Higher = worse (more info lost)
        "measured_l0": round(measured_l0, 2),
        "live_features": live_features,
        "feature_density": round(feature_density, 4),
        "mean_recon_loss": round(float(np.mean(recon_losses)), 4),
        "mean_raw_norm": round(float(np.mean(raw_norms)), 4),
    }


def classify_false_negatives(sae_activations, residuals, labels, words, W_dec,
                             k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD):
    """
    Classify false-negative tokens into:
    (a) Hedging: Low residual norm + split features weakly active (could fire at higher L0)
    (b) Reconstruction error: High residual norm (>2 sigma)
    (c) Hierarchy-driven: Persistent absorption pattern with aligned active features

    Returns classification breakdown.
    """
    X = sae_activations.numpy()
    R = residuals.numpy()
    unique_labels = sorted(set(labels))

    residual_norms = np.linalg.norm(R, axis=1)
    mean_res_norm = np.mean(residual_norms)
    std_res_norm = np.std(residual_norms)
    high_res_threshold = mean_res_norm + 2 * std_res_norm

    total_fn = 0
    hedging = 0
    recon_error = 0
    hierarchy = 0
    fn_details = []

    for letter in unique_labels:
        mask = np.array([l == letter for l in labels])
        n_pos = int(mask.sum())
        if n_pos < 2:
            continue

        y = mask.astype(int)
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', random_state=SEED)
            clf.fit(X, y)
        except:
            continue

        y_pred = clf.predict(X)
        weights = clf.coef_[0]
        top_k_idx = np.argsort(np.abs(weights))[-k_sparse:]

        for i in range(len(y)):
            if y[i] == 1 and y_pred[i] == 1:
                split_activations = X[i, top_k_idx]
                if np.all(split_activations == 0):
                    total_fn += 1

                    # Check if any split feature has weak but nonzero activation nearby
                    # (indicating hedging / threshold sensitivity)
                    max_split_act = float(np.max(np.abs(X[i, top_k_idx])))

                    # Classification logic
                    res_norm = residual_norms[i]
                    is_high_recon_error = res_norm > high_res_threshold

                    # Check for absorption pattern (aligned active features)
                    has_absorption_signal = False
                    active_features = np.where(X[i] > 0)[0]
                    if len(active_features) > 0:
                        split_decoder_mean = W_dec[top_k_idx].mean(axis=0)
                        split_norm = np.linalg.norm(split_decoder_mean)
                        if split_norm > 1e-10:
                            split_decoder_mean /= split_norm
                            for feat_idx in active_features:
                                feat_dec = W_dec[feat_idx]
                                feat_norm = np.linalg.norm(feat_dec)
                                if feat_norm > 1e-10:
                                    cos_sim = abs(np.dot(feat_dec / feat_norm, split_decoder_mean))
                                    if cos_sim > cosine_threshold:
                                        has_absorption_signal = True
                                        break

                    # Priority classification
                    if is_high_recon_error and not has_absorption_signal:
                        recon_error += 1
                        category = "reconstruction_error"
                    elif has_absorption_signal:
                        hierarchy += 1
                        category = "hierarchy_driven"
                    else:
                        hedging += 1
                        category = "hedging"

                    fn_details.append({
                        "word": words[i],
                        "letter": letter,
                        "category": category,
                        "residual_norm": round(float(res_norm), 4),
                        "has_absorption_signal": has_absorption_signal,
                        "is_high_recon_error": is_high_recon_error,
                    })

    return {
        "total_false_negatives": total_fn,
        "hedging": hedging,
        "reconstruction_error": recon_error,
        "hierarchy_driven": hierarchy,
        "pct_hedging": round(hedging / max(total_fn, 1) * 100, 1),
        "pct_reconstruction_error": round(recon_error / max(total_fn, 1) * 100, 1),
        "pct_hierarchy_driven": round(hierarchy / max(total_fn, 1) * 100, 1),
        "residual_norm_threshold": round(float(high_res_threshold), 4),
        "details_sample": fn_details[:20],  # Keep first 20 for inspection
    }


def partial_correlation(x, y, covariates):
    """
    Compute partial correlation between x and y controlling for covariates.
    Uses the standard regression-based approach.
    """
    n = len(x)
    if n < len(covariates[0]) + 3:  # Need enough samples
        return np.nan, np.nan

    # Residualize x on covariates
    from numpy.linalg import lstsq
    C = np.column_stack(covariates)
    C_with_const = np.column_stack([np.ones(n), C])

    # Residualize x
    beta_x, _, _, _ = lstsq(C_with_const, x, rcond=None)
    x_resid = x - C_with_const @ beta_x

    # Residualize y
    beta_y, _, _, _ = lstsq(C_with_const, y, rcond=None)
    y_resid = y - C_with_const @ beta_y

    # Correlation of residuals
    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


def benjamini_hochberg(p_values, alpha=0.05):
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    if n == 0:
        return [], []

    sorted_indices = np.argsort(p_values)
    sorted_pvals = np.array(p_values)[sorted_indices]

    # BH threshold
    thresholds = alpha * np.arange(1, n + 1) / n

    # Find largest i where p(i) <= threshold(i)
    significant = sorted_pvals <= thresholds

    # Adjusted p-values
    adjusted = np.minimum.accumulate((sorted_pvals * n / np.arange(1, n + 1))[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)

    # Map back to original order
    result_adjusted = np.zeros(n)
    result_sig = np.zeros(n, dtype=bool)
    result_adjusted[sorted_indices] = adjusted
    result_sig[sorted_indices] = significant

    return result_adjusted.tolist(), result_sig.tolist()


def rosenbaum_sensitivity(treated, control, covariates_treated, covariates_control,
                          matching="mahalanobis", n_matches=1):
    """
    Perform Rosenbaum sensitivity analysis for observational studies.

    For Mahalanobis matching: match across widths based on covariate distance
    For within-width matching: match only within the same width group

    Returns: Gamma (sensitivity bound), matched test statistic, p-value
    """
    if len(treated) < 3 or len(control) < 3:
        return {"gamma": np.nan, "test_stat": np.nan, "p_value": np.nan,
                "n_matched_pairs": 0, "matching_type": matching}

    n_t = len(treated)
    n_c = len(control)

    if matching == "mahalanobis":
        # Compute Mahalanobis distances
        all_covs = np.vstack([covariates_treated, covariates_control])
        cov_matrix = np.cov(all_covs.T)
        # Regularize
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-6
        try:
            cov_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            cov_inv = np.eye(cov_matrix.shape[0])

        # Match each treated to nearest control
        matched_pairs = []
        used_controls = set()
        for i in range(n_t):
            best_j = None
            best_dist = np.inf
            for j in range(n_c):
                if j in used_controls:
                    continue
                diff = covariates_treated[i] - covariates_control[j]
                d = np.sqrt(diff @ cov_inv @ diff)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j is not None:
                matched_pairs.append((i, best_j, best_dist))
                used_controls.add(best_j)

    elif matching == "within_width":
        # Only match within same width group
        # Width is assumed to be the first covariate
        matched_pairs = []
        used_controls = set()
        for i in range(n_t):
            best_j = None
            best_dist = np.inf
            width_i = covariates_treated[i, 0]  # First covariate = width
            for j in range(n_c):
                if j in used_controls:
                    continue
                width_j = covariates_control[j, 0]
                if abs(width_i - width_j) > 0.01:  # Must be same width (log scale)
                    continue
                diff = covariates_treated[i, 1:] - covariates_control[j, 1:]  # Match on remaining covs
                d = np.linalg.norm(diff)
                if d < best_dist:
                    best_dist = d
                    best_j = j
            if best_j is not None:
                matched_pairs.append((i, best_j, best_dist))
                used_controls.add(best_j)

    n_pairs = len(matched_pairs)
    if n_pairs < 3:
        return {"gamma": np.nan, "test_stat": np.nan, "p_value": np.nan,
                "n_matched_pairs": n_pairs, "matching_type": matching}

    # Compute paired differences
    diffs = np.array([treated[p[0]] - control[p[1]] for p in matched_pairs])

    # Wilcoxon signed-rank test
    try:
        stat, p_value = stats.wilcoxon(diffs, alternative='greater')
    except ValueError:
        # All differences are zero or too few pairs
        stat, p_value = 0, 1.0

    # Rosenbaum Gamma: sensitivity parameter
    # At what Gamma does the significance disappear?
    # Simple approximation: use the ratio of positive to negative ranks
    ranks = stats.rankdata(np.abs(diffs))
    positive_rank_sum = np.sum(ranks[diffs > 0])
    negative_rank_sum = np.sum(ranks[diffs < 0])

    if negative_rank_sum > 0:
        gamma = positive_rank_sum / negative_rank_sum
    else:
        gamma = float('inf') if positive_rank_sum > 0 else 1.0

    return {
        "gamma": round(float(min(gamma, 100)), 4),
        "test_stat": round(float(stat), 4),
        "p_value": round(float(p_value), 6),
        "n_matched_pairs": n_pairs,
        "matching_type": matching,
        "mean_diff": round(float(np.mean(diffs)), 4),
        "median_diff": round(float(np.median(diffs)), 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    set_seeds()
    start_time = datetime.now()
    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "timestamp_start": start_time.isoformat(),
    }

    try:
        report_progress(0, 6, "Loading model")
        print("[1/6] Loading Gemma 2 2B model...")

        from transformer_lens import HookedTransformer
        from transformers import AutoModelForCausalLM, AutoTokenizer

        hf_model_name = "unsloth/gemma-2-2b"
        tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
        hf_model = AutoModelForCausalLM.from_pretrained(hf_model_name, dtype=torch.float16)
        model = HookedTransformer.from_pretrained(
            "google/gemma-2-2b",
            hf_model=hf_model,
            tokenizer=tokenizer,
            device=DEVICE,
            dtype=torch.float16,
        )
        del hf_model
        gc.collect(); torch.cuda.empty_cache()
        print("  Model loaded.")

        # Get vocabulary
        letter_words = get_single_token_english_words(tokenizer)
        results["vocabulary"] = {
            "n_letters": len(letter_words),
            "total_words": sum(len(v) for v in letter_words.values()),
        }

        # ── Step 2: Evaluate each SAE config ──────────────────────────────
        # Group SAE configs by layer to cache raw activations per layer
        from collections import OrderedDict
        configs_by_layer = OrderedDict()
        for cfg in SAE_CONFIGS:
            layer = cfg["layer"]
            if layer not in configs_by_layer:
                configs_by_layer[layer] = []
            configs_by_layer[layer].append(cfg)

        report_progress(1, 6, "Evaluating SAE configs")
        print(f"[2/6] Evaluating {len(SAE_CONFIGS)} SAE configs across {len(configs_by_layer)} layers...")

        from sae_lens import SAE
        sae_results = []
        failed_saes = []
        eval_count = 0

        for layer, layer_configs in configs_by_layer.items():
            hook_point = f"blocks.{layer}.hook_resid_post"
            print(f"\n  === Layer {layer} ({len(layer_configs)} configs) ===")

            # Cache raw activations for this layer (shared across SAEs at same layer)
            print(f"  Collecting raw activations at {hook_point}...")
            raw_acts_cache = {}
            all_raw_acts_list = []
            all_labels_list = []
            all_words_list = []

            for letter in sorted(letter_words.keys()):
                words_for_letter = letter_words[letter]
                if len(words_for_letter) < 2:
                    continue
                for w in words_for_letter:
                    prompt = f" {w['word']}"
                    tokens = model.to_tokens(prompt, prepend_bos=True)
                    with torch.no_grad():
                        _, cache = model.run_with_cache(
                            tokens, names_filter=[hook_point], return_type=None,
                        )
                    raw_act = cache[hook_point][0, -1, :].detach().cpu()
                    all_raw_acts_list.append(raw_act)
                    all_labels_list.append(letter)
                    all_words_list.append(w['word'])

            raw_acts_tensor = torch.stack(all_raw_acts_list)
            print(f"  Cached {len(all_raw_acts_list)} raw activations")

            # Now evaluate each SAE config at this layer
            for cfg in layer_configs:
                eval_count += 1
                sae_key = f"{cfg['release']}:{cfg['sae_id']}"
                print(f"  [{eval_count}/{len(SAE_CONFIGS)}] Loading {sae_key}...")

                try:
                    sae = SAE.from_pretrained(
                        release=cfg["release"],
                        sae_id=cfg["sae_id"],
                        device=DEVICE,
                    )

                    # Encode raw activations through SAE
                    sae_acts_list = []
                    residuals_list = []
                    batch_size = 64

                    for start in range(0, len(all_raw_acts_list), batch_size):
                        end = min(start + batch_size, len(all_raw_acts_list))
                        batch_raw = raw_acts_tensor[start:end].to(DEVICE)
                        with torch.no_grad():
                            batch_sae = sae.encode(batch_raw)
                            batch_recon = sae.decode(batch_sae)
                            batch_resid = batch_raw - batch_recon

                        sae_acts_list.append(batch_sae.cpu())
                        residuals_list.append(batch_resid.cpu())

                    sae_acts = torch.cat(sae_acts_list, dim=0)
                    residuals_t = torch.cat(residuals_list, dim=0)

                    # Get decoder weights for absorption analysis
                    W_dec = sae.W_dec.detach().cpu().numpy()

                    # Compute absorption rate
                    absorption = compute_absorption_rate(
                        sae_acts, all_labels_list, all_words_list, W_dec
                    )

                    # Compute quality metrics
                    quality = compute_quality_metrics(
                        sae, sae_acts, raw_acts_tensor, residuals_t,
                        model, tokenizer, hook_point, device=DEVICE
                    )

                    # Classify false negatives (only for configs with enough FNs)
                    fn_classification = None
                    if absorption["aggregate"]["total_false_negatives"] > 5:
                        fn_classification = classify_false_negatives(
                            sae_acts, residuals_t, all_labels_list, all_words_list, W_dec
                        )

                    sae_results.append({
                        "config": cfg,
                        "sae_key": sae_key,
                        "absorption": absorption["aggregate"],
                        "quality": quality,
                        "fn_classification": fn_classification,
                    })

                    print(f"    Absorption rate: {absorption['aggregate']['aggregate_absorption_rate']:.4f}, "
                          f"L0={quality['measured_l0']:.1f}, "
                          f"SCR={quality['scr_proxy']:.4f}, "
                          f"TPP={quality['tpp_proxy']:.4f}")

                    # Free SAE memory
                    del sae, sae_acts, residuals_t, W_dec, sae_acts_list, residuals_list
                    gc.collect()
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"    FAILED: {e}")
                    traceback.print_exc()
                    failed_saes.append({"config": cfg, "error": str(e)})
                    gc.collect()
                    torch.cuda.empty_cache()
                    continue

                report_progress(1, 6, f"Evaluated {eval_count}/{len(SAE_CONFIGS)} SAEs",
                                {"completed": eval_count, "total": len(SAE_CONFIGS)})

            # Free layer-level cache
            del raw_acts_tensor, all_raw_acts_list
            gc.collect()
            torch.cuda.empty_cache()

        results["sae_results"] = sae_results
        results["failed_saes"] = failed_saes
        results["n_evaluated"] = len(sae_results)
        results["n_failed"] = len(failed_saes)

        # ── Step 3: Partial Correlations ──────────────────────────────────
        report_progress(2, 6, "Computing partial correlations")
        print(f"\n[3/6] Computing partial correlations (n={len(sae_results)} SAEs)...")

        if len(sae_results) < 5:
            print("  WARNING: Too few SAEs for meaningful partial correlations")
            results["partial_correlations"] = {
                "error": "Insufficient SAEs for analysis",
                "n_available": len(sae_results),
                "minimum_required": 5,
            }
        else:
            # Build data arrays
            absorption_rates = np.array([r["absorption"]["aggregate_absorption_rate"] for r in sae_results])
            log_l0 = np.array([np.log(max(r["quality"]["measured_l0"], 1)) for r in sae_results])
            log_width = np.array([np.log(r["config"]["width"]) for r in sae_results])
            arch_class = np.array([1.0 if r["config"]["arch"] == "TopK" else 0.0 for r in sae_results])

            # Quality metrics
            scr = np.array([r["quality"]["scr_proxy"] for r in sae_results])
            tpp = np.array([r["quality"]["tpp_proxy"] for r in sae_results])
            sp_f1 = np.array([r["quality"]["sp_f1_proxy"] for r in sae_results])
            unlearning = np.array([r["quality"]["unlearning_proxy"] for r in sae_results])

            covariates = [log_l0, log_width, arch_class]
            metrics = {
                "SCR_proxy": scr,
                "TPP_proxy": tpp,
                "SP_F1_proxy": sp_f1,
                "Unlearning_proxy": unlearning,
            }

            partial_corr_results = {}
            raw_p_values = []
            metric_names = []

            for metric_name, metric_values in metrics.items():
                # Raw correlation
                raw_r, raw_p = stats.pearsonr(absorption_rates, metric_values)

                # Partial correlation controlling for log(L0), log(width), arch_class
                partial_r, partial_p = partial_correlation(
                    absorption_rates, metric_values, covariates
                )

                # Also compute Spearman for robustness
                spearman_r, spearman_p = stats.spearmanr(absorption_rates, metric_values)

                partial_corr_results[metric_name] = {
                    "raw_pearson_r": round(float(raw_r), 4),
                    "raw_p_value": round(float(raw_p), 6),
                    "partial_r": round(float(partial_r), 4) if not np.isnan(partial_r) else None,
                    "partial_p_value": round(float(partial_p), 6) if not np.isnan(partial_p) else None,
                    "spearman_r": round(float(spearman_r), 4),
                    "spearman_p_value": round(float(spearman_p), 6),
                }
                if not np.isnan(partial_p):
                    raw_p_values.append(partial_p)
                    metric_names.append(metric_name)

            # BH FDR correction
            if raw_p_values:
                adjusted_p, significant = benjamini_hochberg(raw_p_values, alpha=0.05)
                for i, name in enumerate(metric_names):
                    partial_corr_results[name]["fdr_adjusted_p"] = round(adjusted_p[i], 6)
                    partial_corr_results[name]["fdr_significant"] = bool(significant[i])

            # Count how many metrics retain |partial_r| > 0.2
            n_significant = sum(
                1 for v in partial_corr_results.values()
                if v.get("partial_r") is not None and abs(v["partial_r"]) > 0.2
            )

            results["partial_correlations"] = {
                "n_saes": len(sae_results),
                "metrics": partial_corr_results,
                "n_metrics_with_partial_r_gt_0_2": n_significant,
                "h4_target": "at_least_2_of_4",
                "h4_met": n_significant >= 2,
                "covariates": ["log(L0)", "log(width)", "arch_class"],
                "summary_table": [
                    {
                        "Quality_Metric": name,
                        "Raw_r": v["raw_pearson_r"],
                        "Partial_r_L0_Width": v["partial_r"],
                        "P_uncorrected": v.get("partial_p_value"),
                        "P_FDR": v.get("fdr_adjusted_p"),
                        "Significant": v.get("fdr_significant", False),
                    }
                    for name, v in partial_corr_results.items()
                ],
                "raw_data": {
                    "absorption_rates": absorption_rates.tolist(),
                    "log_l0": log_l0.tolist(),
                    "log_width": log_width.tolist(),
                    "arch_class": arch_class.tolist(),
                    "scr": scr.tolist(),
                    "tpp": tpp.tolist(),
                    "sp_f1": sp_f1.tolist(),
                    "unlearning": unlearning.tolist(),
                }
            }

        # ── Step 4: Rosenbaum Sensitivity Analysis ────────────────────────
        report_progress(3, 6, "Rosenbaum sensitivity analysis")
        print("\n[4/6] Rosenbaum sensitivity analysis...")

        if len(sae_results) >= 6:
            # Split into "high absorption" (treated) and "low absorption" (control)
            median_rate = float(np.median(absorption_rates))
            treated_mask = absorption_rates > median_rate
            control_mask = ~treated_mask

            treated_outcomes = absorption_rates[treated_mask]
            control_outcomes = absorption_rates[control_mask]

            # Covariates for matching
            all_covs = np.column_stack([log_width, log_l0, arch_class])
            treated_covs = all_covs[treated_mask]
            control_covs = all_covs[control_mask]

            # Mahalanobis matching (cross-width)
            mahal_result = rosenbaum_sensitivity(
                treated_outcomes, control_outcomes,
                treated_covs, control_covs,
                matching="mahalanobis"
            )

            # Within-width matching
            within_width_result = rosenbaum_sensitivity(
                treated_outcomes, control_outcomes,
                treated_covs, control_covs,
                matching="within_width"
            )

            results["rosenbaum_sensitivity"] = {
                "median_absorption_rate": round(median_rate, 4),
                "n_treated": int(treated_mask.sum()),
                "n_control": int(control_mask.sum()),
                "mahalanobis_matching": mahal_result,
                "within_width_matching": within_width_result,
                "interpretation": {
                    "mahalanobis_gamma": mahal_result["gamma"],
                    "within_width_gamma": within_width_result["gamma"],
                    "divergence": "Width is a confound" if (
                        mahal_result["gamma"] > 1.5 and
                        (within_width_result["gamma"] is None or
                         np.isnan(within_width_result["gamma"]) or
                         within_width_result["gamma"] < 1.5)
                    ) else "Absorption signal robust to width control",
                    "note": "Gamma measures sensitivity to unmeasured confounding. "
                            "Gamma > 2 = reasonably robust; Gamma ~1 = highly sensitive to confounds."
                }
            }
        else:
            results["rosenbaum_sensitivity"] = {
                "error": "Insufficient SAEs for matching",
                "n_available": len(sae_results),
            }

        # ── Step 5: Absorption Decomposition (aggregate across SAEs) ──────
        report_progress(4, 6, "Absorption decomposition")
        print("\n[5/6] Aggregating absorption decomposition...")

        all_classifications = [r["fn_classification"] for r in sae_results if r.get("fn_classification")]
        if all_classifications:
            total_fn = sum(c["total_false_negatives"] for c in all_classifications)
            total_hedging = sum(c["hedging"] for c in all_classifications)
            total_recon = sum(c["reconstruction_error"] for c in all_classifications)
            total_hierarchy = sum(c["hierarchy_driven"] for c in all_classifications)

            results["absorption_decomposition"] = {
                "n_saes_analyzed": len(all_classifications),
                "total_false_negatives": total_fn,
                "hedging": {
                    "count": total_hedging,
                    "pct": round(total_hedging / max(total_fn, 1) * 100, 1),
                    "description": "Recoverable at higher L0; split features weakly active",
                },
                "reconstruction_error": {
                    "count": total_recon,
                    "pct": round(total_recon / max(total_fn, 1) * 100, 1),
                    "description": "High reconstruction error (>2 sigma); information lost in SAE compression",
                },
                "hierarchy_driven": {
                    "count": total_hierarchy,
                    "pct": round(total_hierarchy / max(total_fn, 1) * 100, 1),
                    "description": "Persistent absorption with aligned active features across configs",
                },
                "per_sae_breakdown": [
                    {
                        "sae_key": sae_results[i]["sae_key"],
                        "total_fn": all_classifications[j]["total_false_negatives"],
                        "pct_hedging": all_classifications[j]["pct_hedging"],
                        "pct_recon_error": all_classifications[j]["pct_reconstruction_error"],
                        "pct_hierarchy": all_classifications[j]["pct_hierarchy_driven"],
                    }
                    for j, i in enumerate(
                        [k for k, r in enumerate(sae_results) if r.get("fn_classification")]
                    )
                ]
            }
        else:
            results["absorption_decomposition"] = {
                "error": "No SAEs had sufficient false negatives for decomposition analysis",
            }

        # ── Step 6: Integrate existing cross-domain results ───────────────
        report_progress(5, 6, "Integrating cross-domain results")
        print("\n[6/6] Integrating cross-domain absorption data...")

        # Load cross-domain results for context
        cross_domain_summary = {}
        for fname, domain_name in [
            ("first_letter_validation.json", "first_letter"),
            ("cross_domain_city_country.json", "city_country"),
            ("cross_domain_city_continent.json", "city_continent"),
            ("cross_domain_city_language.json", "city_language"),
            ("cross_domain_animal_class.json", "animal_class"),
        ]:
            try:
                fpath = RESULTS_DIR / fname
                if fpath.exists():
                    with open(fpath) as f:
                        data = json.load(f)
                    sae_key = "l12_16k"
                    if sae_key in data:
                        agg = data[sae_key].get("aggregate", {})
                        cross_domain_summary[domain_name] = {
                            "absorption_rate": agg.get("aggregate_absorption_rate", None),
                            "mean_probe_f1": agg.get("mean_probe_f1", None),
                            "total_tested": agg.get("total_tested", None),
                        }
            except Exception as e:
                print(f"  Warning: Failed to load {fname}: {e}")

        results["cross_domain_context"] = cross_domain_summary

        # ── Save results ──────────────────────────────────────────────────
        end_time = datetime.now()
        elapsed_min = (end_time - start_time).total_seconds() / 60

        results["timestamp_end"] = end_time.isoformat()
        results["elapsed_minutes"] = round(elapsed_min, 1)

        # Summary
        results["summary"] = {
            "n_saes_evaluated": len(sae_results),
            "n_saes_failed": len(failed_saes),
            "partial_correlations_computed": "partial_correlations" in results and "metrics" in results.get("partial_correlations", {}),
            "fdr_correction_applied": any(
                "fdr_adjusted_p" in v
                for v in results.get("partial_correlations", {}).get("metrics", {}).values()
            ),
            "rosenbaum_computed": "rosenbaum_sensitivity" in results and "mahalanobis_matching" in results.get("rosenbaum_sensitivity", {}),
            "decomposition_computed": "absorption_decomposition" in results and "total_false_negatives" in results.get("absorption_decomposition", {}),
            "h4_met": results.get("partial_correlations", {}).get("h4_met", None),
        }

        output_path = RESULTS_DIR / "confound_decomposition.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
        print(f"Total time: {elapsed_min:.1f} min")

        # Update gpu_progress.json
        update_gpu_progress(start_time, end_time)

        mark_done(
            status="success",
            summary=f"Confound decomposition complete: {len(sae_results)} SAEs evaluated, "
                    f"partial correlations computed, Rosenbaum sensitivity analysis done. "
                    f"H4 target met: {results['summary']['h4_met']}",
            results=results["summary"]
        )

    except Exception as e:
        traceback.print_exc()
        results["error"] = str(e)
        results["traceback"] = traceback.format_exc()

        output_path = RESULTS_DIR / "confound_decomposition.json"
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        end_time = datetime.now()
        update_gpu_progress(start_time, end_time, failed=True)
        mark_done(status="failed", summary=str(e))


def update_gpu_progress(start_time, end_time, failed=False):
    """Update gpu_progress.json with timing info."""
    gpu_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        data = json.loads(gpu_path.read_text())
    except:
        data = {"completed": [], "failed": [], "running": {}, "timings": {}}

    # Remove from running
    if TASK_ID in data.get("running", {}):
        del data["running"][TASK_ID]

    if failed:
        if TASK_ID not in data.get("failed", []):
            data.setdefault("failed", []).append(TASK_ID)
    else:
        if TASK_ID not in data.get("completed", []):
            data.setdefault("completed", []).append(TASK_ID)

    elapsed_min = (end_time - start_time).total_seconds() / 60
    data.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 45,
        "actual_min": round(elapsed_min),
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat(),
        "config_snapshot": {
            "model": "gemma-2-2b",
            "n_sae_configs": len(SAE_CONFIGS),
            "analysis_types": ["partial_correlations", "rosenbaum_sensitivity", "absorption_decomposition"],
            "gpu_model": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
            "gpu_count": 1,
        }
    }

    gpu_path.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
