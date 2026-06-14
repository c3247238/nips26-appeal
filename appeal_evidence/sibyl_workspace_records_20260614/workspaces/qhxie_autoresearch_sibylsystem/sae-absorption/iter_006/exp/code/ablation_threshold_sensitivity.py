#!/usr/bin/env python3
"""
ablation_threshold_sensitivity.py — Absorption Threshold Sensitivity Ablation (FULL)

Full 5x4 threshold grid on first-letter task, Gemma 2 2B L12 16k:
  cosine thresholds: {0.01, 0.02, 0.025, 0.03, 0.05}
  magnitude gaps:    {0.5, 1.0, 1.5, 2.0}

For each grid cell, measures absorption rate using the Chanin et al. protocol.
Reports:
  - Per-cell absorption rate with bootstrap 95% CI
  - CV across all 20 threshold combinations
  - Monotonicity check (higher cosine threshold -> fewer detections)
  - Per-letter breakdown for the default threshold
  - Heatmap data for visualization

This addresses the iter5 finding that tau sensitivity was a critical weakness.

Depends on: first_letter_validation (for establishing the probe infrastructure)
"""

import json
import os
import sys
import time
import gc
import traceback
import string
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
FULL_RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "ablation_threshold_sensitivity"
SEED = 42

K_SPARSE = 5
N_PRESELECT = 200  # Feature pre-selection for probe training

# Threshold grid
COSINE_THRESHOLDS = [0.01, 0.02, 0.025, 0.03, 0.05]
MAGNITUDE_GAPS = [0.5, 1.0, 1.5, 2.0]

MAX_PER_LETTER = 25  # Same as first_letter_validation

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID / Progress / Done ──────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(step, total_steps, description, extra=None):
    progress = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
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
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if progress_file.exists():
        try: fp = json.loads(progress_file.read_text())
        except: pass
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
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
    rng = np.random.RandomState(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Vocabulary: common English words (identical to first_letter_validation) ──
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


def collect_activations(model, sae, tokenizer, letter_words, hook_point, device="cuda:0"):
    """Collect SAE activations for each word using single-word prompt."""
    all_acts = []
    all_labels = []
    all_words = []

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
            raw_act = cache[hook_point][0, -1, :].detach()
            sae_act = sae.encode(raw_act.unsqueeze(0))[0].detach()
            all_acts.append(sae_act.cpu())
            all_labels.append(letter)
            all_words.append(w['word'])

    all_acts_tensor = torch.stack(all_acts)
    return all_acts_tensor, all_labels, all_words


def train_probes_and_get_splits(sae_activations, labels, W_dec, k_sparse=K_SPARSE):
    """
    Train probes for each letter and return probe info needed for absorption detection.
    This is done ONCE, independent of thresholds -- probes don't change with threshold.

    Returns dict[letter] -> {
        probe_f1, split_indices, split_weights, probe_direction,
        baseline_act, pos_indices, tested_indices, fn_indices, fn_active_features
    }
    """
    X = sae_activations.numpy()
    letters = sorted(set(labels))
    probe_info = {}

    for letter in letters:
        y = np.array([1 if l == letter else 0 for l in labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if n_pos < 3 or n_neg < 3:
            continue
        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            continue

        # Feature pre-selection
        mean_pos = X[y == 1].mean(axis=0)
        mean_neg = X[y == 0].mean(axis=0)
        feat_score = np.abs(mean_pos - mean_neg)
        preselect_idx = np.argsort(-feat_score)[:N_PRESELECT]
        X_pre = X[:, preselect_idx]

        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            fold_f1s = []
            for train_idx, test_idx in skf.split(X_pre, y):
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, solver='lbfgs',
                    random_state=SEED, class_weight='balanced',
                )
                clf.fit(X_pre[train_idx], y[train_idx])
                preds = clf.predict(X_pre[test_idx])
                fold_f1s.append(f1_score(y[test_idx], preds, zero_division=0))
            probe_f1 = float(np.mean(fold_f1s))
        except Exception:
            continue

        # Retrain on full data
        clf_full = LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=SEED, class_weight='balanced',
        )
        clf_full.fit(X_pre, y)
        full_preds = clf_full.predict(X_pre)

        # Split features
        weights = clf_full.coef_[0]
        top_k_in_pre = np.argsort(-np.abs(weights))[:k_sparse]
        split_indices = preselect_idx[top_k_in_pre]
        split_weights = weights[top_k_in_pre]

        # Probe direction
        probe_direction = torch.zeros(W_dec.shape[1])
        for si, sw in zip(split_indices, split_weights):
            probe_direction += sw * W_dec[si]
        probe_direction = probe_direction / (probe_direction.norm() + 1e-8)

        # Baseline activation
        pos_indices = np.where(y == 1)[0]
        active_vals = []
        for pi in pos_indices:
            for si in split_indices:
                v = sae_activations[pi, si].item()
                if v > 0:
                    active_vals.append(v)
        baseline_act = np.mean(active_vals) if active_vals else 1.0

        # Find false negatives (threshold-independent: probe says yes, split features don't fire)
        tested_indices = []
        fn_indices = []
        fn_active_features = {}  # pi -> list of (fi, cosine, mag_ratio, activation)

        for pi in pos_indices:
            if full_preds[pi] != 1:
                continue
            tested_indices.append(pi)

            split_acts = sae_activations[pi, split_indices]
            if (split_acts > 0).any().item():
                continue

            fn_indices.append(pi)

            # Pre-compute feature info for all active features on this FN token
            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]
            feat_info = []
            for fi in active_feat_indices:
                fi_int = fi.item()
                if fi_int in split_indices:
                    continue
                feat_dec = W_dec[fi_int]
                feat_norm = feat_dec.norm().item()
                if feat_norm < 1e-8:
                    continue
                cos_sim = torch.dot(feat_dec / feat_norm, probe_direction).item()
                feat_act = sae_activations[pi, fi_int].item()
                mag_ratio = feat_act / (baseline_act + 1e-8)
                feat_info.append((fi_int, cos_sim, mag_ratio, feat_act))
            fn_active_features[pi] = feat_info

        probe_info[letter] = {
            "probe_f1": probe_f1,
            "n_pos": n_pos,
            "split_indices": split_indices,
            "split_weights": split_weights,
            "probe_direction": probe_direction,
            "baseline_act": baseline_act,
            "tested_indices": tested_indices,
            "fn_indices": fn_indices,
            "fn_active_features": fn_active_features,
        }

    return probe_info


def measure_absorption_at_threshold(probe_info, cosine_threshold, magnitude_gap, all_words):
    """
    Given pre-computed probe info, measure absorption at a specific threshold.
    This is fast because all the expensive work (model inference, probe training) is done.
    """
    per_letter = {}
    total_tested = 0
    total_fn = 0
    total_absorbed = 0
    absorption_rates = []

    for letter, info in sorted(probe_info.items()):
        n_tested = len(info["tested_indices"])
        n_fn = len(info["fn_indices"])
        n_abs = 0

        for pi in info["fn_indices"]:
            feat_infos = info["fn_active_features"].get(pi, [])
            absorbed = False
            for fi_int, cos_sim, mag_ratio, feat_act in feat_infos:
                if abs(cos_sim) > cosine_threshold and mag_ratio >= magnitude_gap:
                    absorbed = True
                    break
            if absorbed:
                n_abs += 1

        abs_rate = n_abs / n_tested if n_tested > 0 else float('nan')
        fn_rate = n_fn / n_tested if n_tested > 0 else float('nan')

        total_tested += n_tested
        total_fn += n_fn
        total_absorbed += n_abs

        if not np.isnan(abs_rate):
            absorption_rates.append(abs_rate)

        per_letter[letter] = {
            "n_pos": info["n_pos"],
            "n_tested": n_tested,
            "probe_f1": round(info["probe_f1"], 4),
            "n_false_negatives": n_fn,
            "n_absorbed": n_abs,
            "absorption_rate": round(abs_rate, 4) if not np.isnan(abs_rate) else None,
            "false_negative_rate": round(fn_rate, 4) if not np.isnan(fn_rate) else None,
        }

    agg_abs_rate = total_absorbed / total_tested if total_tested > 0 else None
    ci_lower, ci_upper = bootstrap_ci(absorption_rates) if absorption_rates else (None, None)

    return {
        "per_letter": per_letter,
        "aggregate": {
            "total_tested": total_tested,
            "total_false_negatives": total_fn,
            "total_absorbed": total_absorbed,
            "aggregate_absorption_rate": round(agg_abs_rate, 4) if agg_abs_rate is not None else None,
            "aggregate_fn_rate": round(total_fn / total_tested, 4) if total_tested > 0 else None,
            "bootstrap_ci_95": [ci_lower, ci_upper],
            "n_letters_measured": len(absorption_rates),
        },
        "per_letter_absorption_rates": absorption_rates,
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    set_seeds()
    total_steps = 7

    print("=" * 70)
    print("Ablation: Absorption Threshold Sensitivity — Gemma 2 2B L12 16k")
    print(f"Task ID: {TASK_ID}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Seed: {SEED}")
    print(f"Cosine thresholds: {COSINE_THRESHOLDS}")
    print(f"Magnitude gaps: {MAGNITUDE_GAPS}")
    print(f"Grid size: {len(COSINE_THRESHOLDS)} x {len(MAGNITUDE_GAPS)} = {len(COSINE_THRESHOLDS)*len(MAGNITUDE_GAPS)} cells")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    device = "cuda:0"

    # ── Step 1: Load model ──────────────────────────────────────────────
    report_progress(1, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 1/7] Loading Gemma 2 2B model...")
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

    # ── Step 2: Load SAE L12-16k ────────────────────────────────────────
    report_progress(2, total_steps, "Loading SAE L12-16k")
    print("\n[Step 2/7] Loading SAE L12-16k...")
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

    # ── Step 3: Build vocabulary ────────────────────────────────────────
    report_progress(3, total_steps, "Building vocabulary")
    print("\n[Step 3/7] Building single-token English vocabulary per letter...")

    letter_words = get_single_token_english_words(tokenizer, max_per_letter=MAX_PER_LETTER)
    total_words = sum(len(v) for v in letter_words.values())
    print(f"  Total words: {total_words}, Letters: {len(letter_words)}")

    # ── Step 4: Collect activations ─────────────────────────────────────
    report_progress(4, total_steps, "Collecting SAE activations")
    print("\n[Step 4/7] Collecting activations...")
    t0 = time.time()

    sae_acts, all_labels, all_words = collect_activations(
        model, sae, tokenizer, letter_words, hook_point, device
    )
    print(f"  Collected {len(all_labels)} activations in {time.time()-t0:.1f}s")
    print(f"  SAE shape: {sae_acts.shape}")

    # Free model memory -- we only need SAE activations from here
    del model, sae
    gc.collect(); torch.cuda.empty_cache()

    # ── Step 5: Train probes (threshold-independent) ────────────────────
    report_progress(5, total_steps, "Training probes for all letters")
    print("\n[Step 5/7] Training probes and computing split features...")
    t0 = time.time()

    probe_info = train_probes_and_get_splits(sae_acts, all_labels, W_dec, k_sparse=K_SPARSE)

    n_letters_with_probes = len(probe_info)
    total_fn_tokens = sum(len(v["fn_indices"]) for v in probe_info.values())
    total_tested_tokens = sum(len(v["tested_indices"]) for v in probe_info.values())
    print(f"  Probes trained for {n_letters_with_probes} letters in {time.time()-t0:.1f}s")
    print(f"  Total tested tokens: {total_tested_tokens}")
    print(f"  Total false negative tokens: {total_fn_tokens}")

    probe_summary = {}
    for letter, info in sorted(probe_info.items()):
        probe_summary[letter] = {
            "probe_f1": round(info["probe_f1"], 4),
            "n_pos": info["n_pos"],
            "n_tested": len(info["tested_indices"]),
            "n_fn": len(info["fn_indices"]),
        }
        print(f"    {letter}: F1={info['probe_f1']:.3f}, tested={len(info['tested_indices'])}, "
              f"FN={len(info['fn_indices'])}")

    # ── Step 6: Sweep threshold grid ────────────────────────────────────
    report_progress(6, total_steps, "Sweeping 5x4 threshold grid")
    print("\n[Step 6/7] Running threshold sensitivity sweep...")
    t0 = time.time()

    grid_results = {}
    all_aggregate_rates = []

    for i, cos_thr in enumerate(COSINE_THRESHOLDS):
        for j, mag_gap in enumerate(MAGNITUDE_GAPS):
            cell_key = f"cos_{cos_thr}_gap_{mag_gap}"
            result = measure_absorption_at_threshold(probe_info, cos_thr, mag_gap, all_words)
            grid_results[cell_key] = {
                "cosine_threshold": cos_thr,
                "magnitude_gap": mag_gap,
                **result,
            }

            agg_rate = result["aggregate"]["aggregate_absorption_rate"]
            if agg_rate is not None:
                all_aggregate_rates.append(agg_rate)

            ci = result["aggregate"]["bootstrap_ci_95"]
            ci_str = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci[0] is not None else "N/A"
            rate_str = f"{agg_rate:.4f}" if agg_rate is not None else "N/A"
            print(f"    cos={cos_thr:.3f}, gap={mag_gap:.1f}: "
                  f"rate={rate_str}, "
                  f"CI={ci_str}, "
                  f"absorbed={result['aggregate']['total_absorbed']}/{result['aggregate']['total_tested']}")

    sweep_time = time.time() - t0
    print(f"  Grid sweep completed in {sweep_time:.1f}s")

    # ── Step 7: Compute statistics ──────────────────────────────────────
    report_progress(7, total_steps, "Computing stability metrics")
    print("\n[Step 7/7] Computing stability statistics...")

    # CV across all 20 cells
    if len(all_aggregate_rates) > 1:
        mean_rate = np.mean(all_aggregate_rates)
        std_rate = np.std(all_aggregate_rates, ddof=1)
        cv = std_rate / mean_rate if mean_rate > 0 else float('inf')
    else:
        mean_rate = all_aggregate_rates[0] if all_aggregate_rates else 0
        std_rate = 0
        cv = 0

    # Monotonicity checks
    # (1) Higher cosine threshold -> fewer detections (expected)
    cos_monotonic_count = 0
    cos_monotonic_total = 0
    for mag_gap in MAGNITUDE_GAPS:
        rates_at_gap = []
        for cos_thr in COSINE_THRESHOLDS:
            key = f"cos_{cos_thr}_gap_{mag_gap}"
            r = grid_results[key]["aggregate"]["aggregate_absorption_rate"]
            if r is not None:
                rates_at_gap.append(r)
        if len(rates_at_gap) >= 2:
            cos_monotonic_total += 1
            # Check if rates are non-increasing (allowing ties)
            is_mono = all(rates_at_gap[i] >= rates_at_gap[i+1] - 0.001
                         for i in range(len(rates_at_gap)-1))
            if is_mono:
                cos_monotonic_count += 1

    # (2) Higher magnitude gap -> fewer detections (expected)
    gap_monotonic_count = 0
    gap_monotonic_total = 0
    for cos_thr in COSINE_THRESHOLDS:
        rates_at_cos = []
        for mag_gap in MAGNITUDE_GAPS:
            key = f"cos_{cos_thr}_gap_{mag_gap}"
            r = grid_results[key]["aggregate"]["aggregate_absorption_rate"]
            if r is not None:
                rates_at_cos.append(r)
        if len(rates_at_cos) >= 2:
            gap_monotonic_total += 1
            is_mono = all(rates_at_cos[i] >= rates_at_cos[i+1] - 0.001
                         for i in range(len(rates_at_cos)-1))
            if is_mono:
                gap_monotonic_count += 1

    # Build heatmap matrix (for visualization)
    heatmap_rates = []
    for mag_gap in MAGNITUDE_GAPS:
        row = []
        for cos_thr in COSINE_THRESHOLDS:
            key = f"cos_{cos_thr}_gap_{mag_gap}"
            r = grid_results[key]["aggregate"]["aggregate_absorption_rate"]
            row.append(r)
        heatmap_rates.append(row)

    # Rate range
    rate_min = min(all_aggregate_rates) if all_aggregate_rates else None
    rate_max = max(all_aggregate_rates) if all_aggregate_rates else None
    rate_range = rate_max - rate_min if rate_min is not None else None

    # Stability assessment
    if cv < 0.3:
        stability = "STABLE"
        stability_detail = f"CV={cv:.3f} < 0.3: metric is robust to threshold choice"
    elif cv < 0.5:
        stability = "MODERATE"
        stability_detail = f"CV={cv:.3f}: moderate sensitivity to threshold choice"
    else:
        stability = "BRITTLE"
        stability_detail = f"CV={cv:.3f} > 0.5: metric is highly sensitive to threshold choice"

    # Per-cosine-threshold aggregation (averaged over magnitude gaps)
    per_cosine_summary = {}
    for cos_thr in COSINE_THRESHOLDS:
        rates = []
        for mag_gap in MAGNITUDE_GAPS:
            key = f"cos_{cos_thr}_gap_{mag_gap}"
            r = grid_results[key]["aggregate"]["aggregate_absorption_rate"]
            if r is not None:
                rates.append(r)
        if rates:
            per_cosine_summary[str(cos_thr)] = {
                "mean_rate": round(float(np.mean(rates)), 4),
                "std_rate": round(float(np.std(rates, ddof=1)), 4) if len(rates) > 1 else 0,
                "min_rate": round(float(min(rates)), 4),
                "max_rate": round(float(max(rates)), 4),
                "n_gaps": len(rates),
            }

    # Per-gap-threshold aggregation (averaged over cosine thresholds)
    per_gap_summary = {}
    for mag_gap in MAGNITUDE_GAPS:
        rates = []
        for cos_thr in COSINE_THRESHOLDS:
            key = f"cos_{cos_thr}_gap_{mag_gap}"
            r = grid_results[key]["aggregate"]["aggregate_absorption_rate"]
            if r is not None:
                rates.append(r)
        if rates:
            per_gap_summary[str(mag_gap)] = {
                "mean_rate": round(float(np.mean(rates)), 4),
                "std_rate": round(float(np.std(rates, ddof=1)), 4) if len(rates) > 1 else 0,
                "min_rate": round(float(min(rates)), 4),
                "max_rate": round(float(max(rates)), 4),
                "n_thresholds": len(rates),
            }

    elapsed = time.time() - start_time

    # ── Compile output ──────────────────────────────────────────────────
    output = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "config": {
            "model": "gemma-2-2b",
            "sae_id": "layer_12/width_16k/average_l0_82",
            "k_sparse": K_SPARSE,
            "cosine_thresholds": COSINE_THRESHOLDS,
            "magnitude_gaps": MAGNITUDE_GAPS,
            "n_grid_cells": len(COSINE_THRESHOLDS) * len(MAGNITUDE_GAPS),
            "seed": SEED,
            "n_words": total_words,
            "n_letters_with_probes": n_letters_with_probes,
        },
        "probe_summary": probe_summary,
        "grid_results": grid_results,
        "heatmap": {
            "rows_label": "magnitude_gap",
            "cols_label": "cosine_threshold",
            "row_values": MAGNITUDE_GAPS,
            "col_values": COSINE_THRESHOLDS,
            "rates": heatmap_rates,
        },
        "per_cosine_summary": per_cosine_summary,
        "per_gap_summary": per_gap_summary,
        "stability_metrics": {
            "mean_absorption_rate": round(float(mean_rate), 4),
            "std_absorption_rate": round(float(std_rate), 4),
            "cv": round(float(cv), 4),
            "stability_assessment": stability,
            "stability_detail": stability_detail,
            "rate_min": round(float(rate_min), 4) if rate_min is not None else None,
            "rate_max": round(float(rate_max), 4) if rate_max is not None else None,
            "rate_range": round(float(rate_range), 4) if rate_range is not None else None,
            "n_grid_cells_computed": len(all_aggregate_rates),
        },
        "monotonicity": {
            "cosine_monotonic": {
                "description": "Higher cosine threshold -> fewer or equal detections (expected)",
                "n_monotonic": cos_monotonic_count,
                "n_total": cos_monotonic_total,
                "fraction": round(cos_monotonic_count / cos_monotonic_total, 3) if cos_monotonic_total > 0 else None,
            },
            "gap_monotonic": {
                "description": "Higher magnitude gap -> fewer or equal detections (expected)",
                "n_monotonic": gap_monotonic_count,
                "n_total": gap_monotonic_total,
                "fraction": round(gap_monotonic_count / gap_monotonic_total, 3) if gap_monotonic_total > 0 else None,
            },
        },
        "pass_criteria": {
            "all_20_cells_computed": len(all_aggregate_rates) == 20,
            "cv_reported": True,
            "cv_value": round(float(cv), 4),
            "cv_stable": cv < 0.3,
            "cv_moderate": cv < 0.5,
            "cosine_monotonicity_expected": cos_monotonic_count == cos_monotonic_total if cos_monotonic_total > 0 else None,
            "overall_assessment": stability,
        },
    }

    # ── Print summary ───────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("THRESHOLD SENSITIVITY RESULTS")
    print("=" * 70)
    print(f"  Grid size: {len(COSINE_THRESHOLDS)}x{len(MAGNITUDE_GAPS)} = {len(all_aggregate_rates)} cells")
    print(f"  Mean absorption rate: {mean_rate:.4f}")
    print(f"  Std:  {std_rate:.4f}")
    print(f"  CV:   {cv:.4f}")
    print(f"  Range: [{rate_min:.4f}, {rate_max:.4f}] (span={rate_range:.4f})")
    print(f"  Stability: {stability}")
    print(f"  Cosine monotonicity: {cos_monotonic_count}/{cos_monotonic_total}")
    print(f"  Gap monotonicity: {gap_monotonic_count}/{gap_monotonic_total}")
    print(f"\n  Heatmap (rows=gap, cols=cosine):")
    print(f"  {'':>8s}", end="")
    for cos_thr in COSINE_THRESHOLDS:
        print(f"  {cos_thr:>7.3f}", end="")
    print()
    for i, mag_gap in enumerate(MAGNITUDE_GAPS):
        print(f"  {mag_gap:>6.1f}:", end="")
        for j, cos_thr in enumerate(COSINE_THRESHOLDS):
            r = heatmap_rates[i][j]
            print(f"  {r:>7.4f}" if r is not None else "    N/A ", end="")
        print()

    print(f"\n  Per-cosine summary:")
    for cos_thr, summ in per_cosine_summary.items():
        print(f"    cos={cos_thr}: mean={summ['mean_rate']:.4f}, range=[{summ['min_rate']:.4f}, {summ['max_rate']:.4f}]")

    print(f"\n  Per-gap summary:")
    for gap, summ in per_gap_summary.items():
        print(f"    gap={gap}: mean={summ['mean_rate']:.4f}, range=[{summ['min_rate']:.4f}, {summ['max_rate']:.4f}]")

    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    # ── Save ────────────────────────────────────────────────────────────
    output_path = FULL_RESULTS_DIR / "ablation_threshold_sensitivity.json"
    output_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\n  Result written to {output_path}")

    mark_done(
        "success",
        f"Threshold sensitivity: CV={cv:.3f} ({stability}), "
        f"mean_rate={mean_rate:.4f}, range=[{rate_min:.4f},{rate_max:.4f}], "
        f"cos_mono={cos_monotonic_count}/{cos_monotonic_total}, "
        f"gap_mono={gap_monotonic_count}/{gap_monotonic_total}, "
        f"{len(all_aggregate_rates)}/20 cells computed",
    )

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        mark_done("failed", f"Unhandled exception: {e}")
        sys.exit(1)
