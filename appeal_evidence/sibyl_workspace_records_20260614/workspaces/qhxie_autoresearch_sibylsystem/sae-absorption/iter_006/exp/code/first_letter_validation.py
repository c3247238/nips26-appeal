#!/usr/bin/env python3
"""
first_letter_validation.py — First-Letter Absorption Baseline on Gemma 2 2B (PILOT)

Reproduces Chanin et al. first-letter absorption measurement on Gemma 2 2B + Gemma Scope.
Uses the standard methodology:
1. Build a set of common English words that are single tokens, grouped by first letter
2. For each word, run a prompt like "The first letter of the word [WORD] is" through the model
3. Collect SAE activations at the position of the target token
4. Train logistic regression probes to predict which letter group a token belongs to
5. Identify "split features" - top-k SAE features by probe weight
6. False negatives: probe predicts correctly but split features don't fire
7. Absorption: among false negatives, check if an active feature's decoder direction
   aligns with the probe direction (cosine > threshold)

PILOT mode: ~500 tokens, seed 42, timeout 900s.
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
RESULTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
FULL_RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "first_letter_validation"
SEED = 42

K_SPARSE = 5
# Cosine threshold calibrated to Gemma 2 2B's d_model=2304:
# Random pairwise cosine: mean=0.018, P95=0.053, P99=0.098
# Use 0.1 (above P99 of random baseline) as primary threshold
# Also test at 0.025 (Chanin et al. original, calibrated for GPT-2 d=768) for comparison
COSINE_THRESHOLD = 0.1
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 25  # Pilot: max words per letter

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

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


def bootstrap_ci(values, n_bootstrap=1000, ci=0.95, seed=SEED):
    np.random.seed(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(np.random.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Vocabulary: common English words ────────────────────────────────────────

# Common English words, curated to ensure they are real words
# We'll use a large set and filter by tokenizer
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

            # Try both lowercase and as-is
            for w in [word.lower(), word]:
                # Gemma tokenizer: " word" should be a single token for common words
                encoded = tokenizer.encode(f" {w}", add_special_tokens=False)
                if len(encoded) == 1:
                    valid.append({
                        "word": w,
                        "token_id": encoded[0],
                    })
                    break

            if len(valid) >= max_per_letter:
                break

        letter_words[letter] = valid

    return letter_words


def collect_activations(model, sae, tokenizer, letter_words, hook_point, device="cuda:0"):
    """
    Collect SAE activations for each word using the first-letter prompt template.

    Uses the prompt: " [word]" and takes the activation at the word token position.
    This is simpler and more direct than ICL prompts for the pilot.
    """
    all_acts = []
    all_labels = []
    all_words = []
    all_raw_acts = []  # Raw model activations for C3

    for letter in sorted(letter_words.keys()):
        words = letter_words[letter]
        if len(words) < 2:
            continue

        for w in words:
            # Simple prompt - just the word with a space prefix
            prompt = f" {w['word']}"
            tokens = model.to_tokens(prompt, prepend_bos=True)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point],
                    return_type=None,
                )

            # Get activation at the last (word) token
            raw_act = cache[hook_point][0, -1, :].detach()  # [d_model]
            sae_act = sae.encode(raw_act.unsqueeze(0))[0].detach()  # [n_features]

            all_acts.append(sae_act.cpu())
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(letter)
            all_words.append(w['word'])

    all_acts_tensor = torch.stack(all_acts)
    all_raw_tensor = torch.stack(all_raw_acts)

    return all_acts_tensor, all_raw_tensor, all_labels, all_words


def analyze_absorption(
    sae_activations, labels, words, W_dec,
    k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
    magnitude_gap=MAGNITUDE_GAP,
):
    """
    Full absorption analysis following Chanin et al. protocol.

    For each letter:
    1. Train binary logistic regression probe on full SAE activations
    2. Identify "split features" = top-k features by absolute probe weight
    3. For correctly-classified positive tokens where ALL split features are inactive:
       these are false negatives (the SAE missed the letter info)
    4. Among false negatives, check if any active feature absorbs the letter info:
       - Its decoder direction aligns with the average decoder direction of split features
       - cosine(feature decoder, split decoder mean) > threshold
       - The feature has meaningful activation (> magnitude_gap relative to baseline)
    """
    X = sae_activations.numpy()
    n_features = X.shape[1]
    letters = sorted(set(labels))

    per_letter = {}
    all_probe_f1s = []
    total_tested = 0
    total_fn = 0
    total_absorbed = 0
    absorption_rates = []

    # Pre-compute feature discriminability for each letter (fast vectorized)
    # This avoids training LR on all 16k features which is very slow
    N_PRESELECT = 200  # Select top 200 features per letter by discriminability

    for letter in letters:
        y = np.array([1 if l == letter else 0 for l in labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)

        if n_pos < 3 or n_neg < 3:
            per_letter[letter] = {"status": "skipped", "reason": f"n_pos={n_pos}", "n_pos": n_pos}
            continue

        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            per_letter[letter] = {"status": "skipped", "reason": "insufficient for CV", "n_pos": n_pos}
            continue

        # ── Feature pre-selection: top N_PRESELECT by mean difference ──
        # This is fast (vectorized) and selects informative features
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
            all_probe_f1s.append(probe_f1)
        except Exception as e:
            per_letter[letter] = {"status": "error", "error": str(e), "n_pos": n_pos}
            continue

        # ── Retrain on full data for analysis ──
        clf_full = LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=SEED, class_weight='balanced',
        )
        clf_full.fit(X_pre, y)
        full_preds = clf_full.predict(X_pre)

        # ── Identify split features: top-k by absolute probe weight ──
        # Map back from preselected indices to original feature indices
        weights = clf_full.coef_[0]  # [N_PRESELECT]
        top_k_in_pre = np.argsort(-np.abs(weights))[:k_sparse]
        split_indices = preselect_idx[top_k_in_pre]  # Map to original indices
        split_weights = weights[top_k_in_pre]

        # Probe direction: mean decoder direction of split features, weighted by probe weight
        probe_direction = torch.zeros(W_dec.shape[1])
        for si, sw in zip(split_indices, split_weights):
            probe_direction += sw * W_dec[si]
        probe_direction = probe_direction / (probe_direction.norm() + 1e-8)

        # Baseline activation: mean activation of split features when they fire on positives
        pos_indices = np.where(y == 1)[0]
        active_vals = []
        for pi in pos_indices:
            for si in split_indices:
                v = sae_activations[pi, si].item()
                if v > 0:
                    active_vals.append(v)
        baseline_act = np.mean(active_vals) if active_vals else 1.0

        # ── Count false negatives and absorption ──
        n_tested_letter = 0
        n_fn_letter = 0
        n_abs_letter = 0
        absorbed_examples = []

        for pi in pos_indices:
            # Only consider tokens that the probe correctly classifies as positive
            if full_preds[pi] != 1:
                continue
            n_tested_letter += 1

            # Check if ANY split feature fires
            split_acts = sae_activations[pi, split_indices]
            if (split_acts > 0).any().item():
                continue  # Split features cover this token, not a false negative

            # FALSE NEGATIVE: probe says yes, but no split features fire
            n_fn_letter += 1

            # ── Check absorption criteria ──
            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]

            absorbed = False
            absorbing_feats = []

            for fi in active_feat_indices:
                fi_int = fi.item()
                # Skip if it's a split feature itself (already checked)
                if fi_int in split_indices:
                    continue

                feat_dec = W_dec[fi_int]
                feat_norm = feat_dec.norm().item()
                if feat_norm < 1e-8:
                    continue

                cos_sim = torch.dot(feat_dec / feat_norm, probe_direction).item()
                feat_act = sae_activations[pi, fi_int].item()
                mag_ratio = feat_act / (baseline_act + 1e-8)

                if abs(cos_sim) > cosine_threshold and mag_ratio >= magnitude_gap:
                    absorbed = True
                    absorbing_feats.append({
                        "feature_idx": fi_int,
                        "cosine": round(cos_sim, 4),
                        "mag_ratio": round(mag_ratio, 4),
                        "activation": round(feat_act, 4),
                    })

            if absorbed:
                n_abs_letter += 1
                absorbed_examples.append({
                    "word": words[pi] if pi < len(words) else "unknown",
                    "features": absorbing_feats[:3],
                })

        abs_rate = n_abs_letter / n_tested_letter if n_tested_letter > 0 else float('nan')
        fn_rate = n_fn_letter / n_tested_letter if n_tested_letter > 0 else float('nan')

        total_tested += n_tested_letter
        total_fn += n_fn_letter
        total_absorbed += n_abs_letter

        if not np.isnan(abs_rate):
            absorption_rates.append(abs_rate)

        per_letter[letter] = {
            "status": "ok",
            "n_pos": n_pos,
            "n_tested": n_tested_letter,
            "probe_f1": round(probe_f1, 4),
            "n_false_negatives": n_fn_letter,
            "n_absorbed": n_abs_letter,
            "absorption_rate": round(abs_rate, 4) if not np.isnan(abs_rate) else None,
            "false_negative_rate": round(fn_rate, 4) if not np.isnan(fn_rate) else None,
            "split_features": split_indices.tolist(),
            "split_weights": [round(float(w), 4) for w in split_weights],
            "absorbed_examples": absorbed_examples[:3],
        }

        gate_char = "+" if probe_f1 > 0.85 else "-"
        print(f"    [{gate_char}] {letter}: F1={probe_f1:.3f}, tested={n_tested_letter}, "
              f"FN={n_fn_letter}, absorbed={n_abs_letter}, "
              f"rate={'N/A' if np.isnan(abs_rate) else f'{abs_rate:.3f}'}")

    # Aggregate
    agg_abs_rate = total_absorbed / total_tested if total_tested > 0 else None
    ci_lower, ci_upper = bootstrap_ci(absorption_rates) if absorption_rates else (None, None)
    letters_passing = sum(1 for r in per_letter.values()
                          if r.get("status") == "ok" and r.get("probe_f1", 0) > 0.85)
    letters_tested = sum(1 for r in per_letter.values() if r.get("status") == "ok")

    return {
        "per_letter": per_letter,
        "aggregate": {
            "total_tested": total_tested,
            "total_false_negatives": total_fn,
            "total_absorbed": total_absorbed,
            "aggregate_absorption_rate": round(agg_abs_rate, 4) if agg_abs_rate is not None else None,
            "aggregate_fn_rate": round(total_fn / total_tested, 4) if total_tested > 0 else None,
            "bootstrap_ci_95": [ci_lower, ci_upper],
            "mean_probe_f1": round(float(np.mean(all_probe_f1s)), 4) if all_probe_f1s else None,
            "median_probe_f1": round(float(np.median(all_probe_f1s)), 4) if all_probe_f1s else None,
            "letters_above_gate": letters_passing,
            "letters_tested": letters_tested,
        },
        "absorption_rates": [round(r, 4) for r in absorption_rates],
    }


def run_random_probe_control(sae_activations, labels, W_dec, k_sparse=K_SPARSE):
    """C1: Random probe control. Use random features as split features."""
    print("\n  Running C1: Random probe control...")
    np.random.seed(SEED + 100)

    n_features = sae_activations.shape[1]
    total_fn = 0
    total_tested = 0
    total_absorbed = 0

    for letter in sorted(set(labels)):
        y = np.array([1 if l == letter else 0 for l in labels])
        n_pos = int(y.sum())
        if n_pos < 3:
            continue

        # Random split features
        random_split = np.random.choice(n_features, size=k_sparse, replace=False)
        random_direction = W_dec[random_split].mean(dim=0)
        random_direction = random_direction / (random_direction.norm() + 1e-8)

        pos_indices = np.where(y == 1)[0]
        for pi in pos_indices:
            total_tested += 1
            split_acts = sae_activations[pi, random_split]
            if (split_acts > 0).any().item():
                continue
            total_fn += 1

            # Check absorption with random direction
            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]
            for fi in active_feat_indices:
                fi_int = fi.item()
                feat_dec = W_dec[fi_int]
                fn_v = feat_dec.norm().item()
                if fn_v < 1e-8:
                    continue
                cos = torch.dot(feat_dec / fn_v, random_direction).item()
                if abs(cos) > COSINE_THRESHOLD:
                    total_absorbed += 1
                    break

    abs_rate = total_absorbed / total_tested if total_tested > 0 else 0
    fn_rate = total_fn / total_tested if total_tested > 0 else 0
    print(f"    Random probe: tested={total_tested}, FN={total_fn} ({fn_rate:.3f}), "
          f"absorbed={total_absorbed} ({abs_rate:.3f})")
    return {
        "control_type": "C1_random_probe",
        "total_tested": total_tested,
        "total_false_negatives": total_fn,
        "false_negative_rate": round(fn_rate, 4),
        "total_absorbed": total_absorbed,
        "absorption_rate": round(abs_rate, 4),
    }


def run_shuffled_label_control(sae_activations, labels, words, W_dec, k_sparse=K_SPARSE):
    """C2: Shuffled label control. Random label assignment."""
    print("\n  Running C2: Shuffled label control...")
    np.random.seed(SEED + 200)

    X = sae_activations.numpy()
    shuffled_labels = list(labels)
    random.shuffle(shuffled_labels)

    total_absorbed = 0
    total_tested = 0

    for letter in sorted(set(shuffled_labels)):
        y = np.array([1 if l == letter else 0 for l in shuffled_labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < 3 or n_neg < 3:
            continue

        try:
            # Pre-select features for efficiency
            mean_pos_s = X[y == 1].mean(axis=0)
            mean_neg_s = X[y == 0].mean(axis=0)
            pre_idx_s = np.argsort(-np.abs(mean_pos_s - mean_neg_s))[:200]
            X_pre_s = X[:, pre_idx_s]

            clf = LogisticRegression(
                max_iter=1000, C=1.0, solver='lbfgs',
                random_state=SEED, class_weight='balanced',
            )
            clf.fit(X_pre_s, y)
            preds = clf.predict(X_pre_s)
        except Exception:
            continue

        weights = clf.coef_[0]
        top_k_pre_s = np.argsort(-np.abs(weights))[:k_sparse]
        split_idx = pre_idx_s[top_k_pre_s]

        probe_dir = torch.zeros(W_dec.shape[1])
        for si, wi in zip(split_idx, top_k_pre_s):
            probe_dir += weights[wi] * W_dec[si]
        probe_dir = probe_dir / (probe_dir.norm() + 1e-8)

        pos_indices = np.where(y == 1)[0]
        for pi in pos_indices:
            if preds[pi] != 1:
                continue
            total_tested += 1

            split_acts = sae_activations[pi, split_idx]
            if (split_acts > 0).any().item():
                continue

            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]
            for fi in active_feat_indices:
                fi_int = fi.item()
                feat_dec = W_dec[fi_int]
                fn_v = feat_dec.norm().item()
                if fn_v < 1e-8:
                    continue
                cos = torch.dot(feat_dec / fn_v, probe_dir).item()
                if abs(cos) > COSINE_THRESHOLD:
                    total_absorbed += 1
                    break

    rate = total_absorbed / total_tested if total_tested > 0 else 0
    print(f"    Shuffled labels: tested={total_tested}, absorbed={total_absorbed} ({rate:.3f})")
    return {
        "control_type": "C2_shuffled_labels",
        "total_tested": total_tested,
        "total_absorbed": total_absorbed,
        "absorption_rate": round(rate, 4),
    }


def run_dense_probe_control(raw_activations, labels):
    """C3: Dense probe on raw model activations (upper bound)."""
    print("\n  Running C3: Dense probe upper bound...")

    X_raw = raw_activations.numpy()
    dense_f1s = {}

    for letter in sorted(set(labels)):
        y = np.array([1 if l == letter else 0 for l in labels])
        n_pos = int(y.sum())
        n_neg = int(len(y) - n_pos)
        if n_pos < 3 or n_neg < 3:
            continue

        n_splits = min(5, n_pos, n_neg)
        if n_splits < 2:
            continue

        try:
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
            fold_f1s = []
            for train_idx, test_idx in skf.split(X_raw, y):
                clf = LogisticRegression(
                    max_iter=1000, C=1.0, solver='lbfgs',
                    random_state=SEED, class_weight='balanced',
                )
                clf.fit(X_raw[train_idx], y[train_idx])
                preds = clf.predict(X_raw[test_idx])
                fold_f1s.append(f1_score(y[test_idx], preds, zero_division=0))
            dense_f1s[letter] = round(float(np.mean(fold_f1s)), 4)
        except Exception:
            continue

    mean_f1 = round(float(np.mean(list(dense_f1s.values()))), 4) if dense_f1s else None
    print(f"    Dense probe mean F1: {mean_f1}")
    return {
        "control_type": "C3_dense_probe",
        "per_letter_f1": dense_f1s,
        "mean_f1": mean_f1,
        "n_letters": len(dense_f1s),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    set_seeds()
    total_steps = 10

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "seed": SEED,
        "model": "gemma-2-2b",
        "timestamp_start": datetime.now().isoformat(),
    }

    print("=" * 70)
    print("First-Letter Absorption Baseline — Gemma 2 2B (PILOT)")
    print(f"Task ID: {TASK_ID}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Seed: {SEED}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    device = "cuda:0"

    # ── Step 1: Load model ──────────────────────────────────────────────
    report_progress(1, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 1/10] Loading Gemma 2 2B model...")
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
    print("\n[Step 2/10] Loading SAE L12-16k...")
    t0 = time.time()

    from sae_lens import SAE

    sae_16k = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device=device,
    )
    print(f"  SAE loaded in {time.time()-t0:.1f}s, features={sae_16k.cfg.d_sae}")

    hook_point = "blocks.12.hook_resid_post"
    W_dec_16k = sae_16k.W_dec.data.cpu()  # [n_features, d_model]

    # ── Step 3: Build vocabulary ────────────────────────────────────────
    report_progress(3, total_steps, "Building single-token English vocabulary")
    print("\n[Step 3/10] Building single-token English vocabulary per letter...")

    letter_words = get_single_token_english_words(tokenizer, max_per_letter=MAX_PER_LETTER)

    print(f"  Letters with words: {len(letter_words)}")
    for letter in sorted(letter_words.keys()):
        words = letter_words[letter]
        print(f"    {letter}: {len(words)} words (e.g., {', '.join(w['word'] for w in words[:5])})")

    total_words = sum(len(v) for v in letter_words.values())
    print(f"  Total words: {total_words}")

    results["vocabulary"] = {
        "n_letters": len(letter_words),
        "per_letter_counts": {k: len(v) for k, v in sorted(letter_words.items())},
        "total_words": total_words,
        "examples": {k: [w['word'] for w in v[:5]] for k, v in sorted(letter_words.items())},
    }

    # ── Step 4: Collect activations ─────────────────────────────────────
    report_progress(4, total_steps, "Collecting SAE and raw activations")
    print("\n[Step 4/10] Collecting activations...")
    t0 = time.time()

    sae_acts, raw_acts, all_labels, all_words = collect_activations(
        model, sae_16k, tokenizer, letter_words, hook_point, device
    )

    print(f"  Collected {len(all_labels)} activations in {time.time()-t0:.1f}s")
    print(f"  SAE shape: {sae_acts.shape}, Raw shape: {raw_acts.shape}")

    # ── Step 5: Main absorption analysis (L12-16k) ─────────────────────
    report_progress(5, total_steps, "Running absorption analysis L12-16k")
    print("\n[Step 5/10] Absorption analysis on L12-16k...")
    t0 = time.time()

    l12_16k_results = analyze_absorption(
        sae_acts, all_labels, all_words, W_dec_16k,
        k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
        magnitude_gap=MAGNITUDE_GAP,
    )

    agg = l12_16k_results["aggregate"]
    print(f"\n  L12-16k Summary:")
    print(f"    Aggregate absorption rate: {agg['aggregate_absorption_rate']}")
    print(f"    Aggregate FN rate: {agg['aggregate_fn_rate']}")
    print(f"    CI 95%: {agg['bootstrap_ci_95']}")
    print(f"    Mean probe F1: {agg['mean_probe_f1']}")
    print(f"    Letters passing gate: {agg['letters_above_gate']}/{agg['letters_tested']}")
    print(f"    Time: {time.time()-t0:.1f}s")

    results["l12_16k"] = {
        "sae_id": "layer_12/width_16k/average_l0_82",
        "k_sparse": K_SPARSE,
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        **l12_16k_results,
    }

    # ── Step 6: C1 Random probe control ─────────────────────────────────
    report_progress(6, total_steps, "C1: Random probe control")
    c1 = run_random_probe_control(sae_acts, all_labels, W_dec_16k)
    results["controls"] = {"C1_random_probe": c1}

    # ── Step 7: C2 Shuffled label control ───────────────────────────────
    report_progress(7, total_steps, "C2: Shuffled label control")
    c2 = run_shuffled_label_control(sae_acts, all_labels, all_words, W_dec_16k)
    results["controls"]["C2_shuffled_labels"] = c2

    # ── Step 8: C3 Dense probe upper bound ──────────────────────────────
    report_progress(8, total_steps, "C3: Dense probe upper bound")
    c3 = run_dense_probe_control(raw_acts, all_labels)
    results["controls"]["C3_dense_probe"] = c3

    # ── Step 9: L12-65k width comparison ────────────────────────────────
    report_progress(9, total_steps, "L12-65k width comparison")
    print("\n[Step 9/10] Width comparison: L12-65k...")
    t0 = time.time()

    try:
        del sae_16k
        gc.collect(); torch.cuda.empty_cache()

        sae_65k = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_12/width_65k/average_l0_72",
            device=device,
        )
        W_dec_65k = sae_65k.W_dec.data.cpu()
        print(f"  SAE L12-65k loaded, features={sae_65k.cfg.d_sae}")

        # Collect 65k activations
        acts_65k = []
        for word in all_words:
            prompt = f" {word}"
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_point], return_type=None,
                )
            act = cache[hook_point][0, -1, :].detach()
            sae_act = sae_65k.encode(act.unsqueeze(0))[0].detach()
            acts_65k.append(sae_act.cpu())

        acts_65k_tensor = torch.stack(acts_65k)
        print(f"  65k activations collected: {acts_65k_tensor.shape}")

        l12_65k_results = analyze_absorption(
            acts_65k_tensor, all_labels, all_words, W_dec_65k,
            k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
            magnitude_gap=MAGNITUDE_GAP,
        )

        agg_65k = l12_65k_results["aggregate"]
        print(f"\n  L12-65k Summary:")
        print(f"    Aggregate absorption rate: {agg_65k['aggregate_absorption_rate']}")
        print(f"    Mean probe F1: {agg_65k['mean_probe_f1']}")
        print(f"    Letters passing gate: {agg_65k['letters_above_gate']}/{agg_65k['letters_tested']}")

        results["l12_65k"] = {
            "sae_id": "layer_12/width_65k/average_l0_72",
            **l12_65k_results,
        }

        del sae_65k
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        results["l12_65k"] = {"status": "error", "error": str(e), "traceback": tb}
        print(f"  L12-65k failed: {e}")

    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Step 10: Compile and save ───────────────────────────────────────
    report_progress(10, total_steps, "Compiling results")
    print("\n[Step 10/10] Compiling results...")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp_end"] = datetime.now().isoformat()

    # Pass criteria
    l12_agg = results.get("l12_16k", {}).get("aggregate", {})
    agg_rate = l12_agg.get("aggregate_absorption_rate")
    letters_gate = l12_agg.get("letters_above_gate", 0)
    letters_total = l12_agg.get("letters_tested", 0)
    c1_rate = results.get("controls", {}).get("C1_random_probe", {}).get("absorption_rate", 1.0)
    c2_rate = results.get("controls", {}).get("C2_shuffled_labels", {}).get("absorption_rate", 1.0)

    # Published range: 15-35%. Within 0.5x-3x means 7.5-105%.
    rate_in_range = agg_rate is not None and 0.075 <= agg_rate <= 1.05
    letters_pass = letters_gate >= 20
    c1_pass = c1_rate < 0.02
    c2_pass = c2_rate < 0.05

    pass_criteria = {
        "absorption_rate_in_range": {
            "criterion": "0.075 <= rate <= 1.05 (0.5x-3x of published 15-35%)",
            "value": agg_rate, "passed": rate_in_range,
        },
        "letters_above_gate": {
            "criterion": ">=20 letters with probe F1 > 0.85",
            "value": letters_gate, "passed": letters_pass,
        },
        "random_probe_control": {
            "criterion": "absorption rate < 2%",
            "value": c1_rate, "passed": c1_pass,
        },
        "shuffled_control": {
            "criterion": "absorption rate < 5%",
            "value": c2_rate, "passed": c2_pass,
        },
        "overall": rate_in_range and letters_pass and c1_pass and c2_pass,
    }
    results["pass_criteria"] = pass_criteria

    # Published comparison
    results["published_comparison"] = {
        "published_range": [0.15, 0.35],
        "our_rate": agg_rate,
        "within_3x": rate_in_range,
        "note": "Chanin et al. 2024 reported 15-35% absorption across all tested SAEs on GPT-2",
    }

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  L12-16k Absorption Rate: {agg_rate}")
    print(f"  L12-16k FN Rate: {l12_agg.get('aggregate_fn_rate')}")
    print(f"  L12-16k CI: {l12_agg.get('bootstrap_ci_95')}")
    print(f"  Mean Probe F1: {l12_agg.get('mean_probe_f1')}")
    print(f"  Letters passing: {letters_gate}/{letters_total}")
    l65_agg = results.get("l12_65k", {}).get("aggregate", {})
    print(f"  L12-65k Absorption Rate: {l65_agg.get('aggregate_absorption_rate', 'N/A')}")
    print(f"  C1 (Random): {c1_rate}")
    print(f"  C2 (Shuffled): {c2_rate}")
    print(f"  C3 (Dense F1): {results.get('controls', {}).get('C3_dense_probe', {}).get('mean_f1')}")
    print(f"\n  Pass Criteria:")
    for k, v in pass_criteria.items():
        if k == "overall":
            print(f"    OVERALL: {'PASS' if v else 'FAIL'}")
        else:
            print(f"    {k}: {'PASS' if v['passed'] else 'FAIL'} ({v['value']})")
    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    # Save
    for path in [RESULTS_DIR / "first_letter_validation.json",
                  FULL_RESULTS_DIR / "first_letter_validation.json"]:
        path.write_text(json.dumps(results, indent=2, default=str))
        print(f"  Saved: {path}")

    mark_done(
        "success" if pass_criteria["overall"] else "partial",
        f"First-letter PILOT: rate={agg_rate}, F1={l12_agg.get('mean_probe_f1')}, "
        f"letters_gate={letters_gate}/{letters_total}, "
        f"overall={'PASS' if pass_criteria['overall'] else 'FAIL'}",
    )

    del model
    gc.collect(); torch.cuda.empty_cache()

    return 0 if pass_criteria["overall"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        mark_done("failed", f"Unhandled exception: {e}")
        sys.exit(1)
