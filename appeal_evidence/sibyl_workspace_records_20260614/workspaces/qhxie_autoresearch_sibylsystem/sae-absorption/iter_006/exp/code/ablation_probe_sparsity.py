#!/usr/bin/env python3
"""
ablation_probe_sparsity.py — Ablation: Probe Sparsity (k=1,3,5,10)

Tests how probe granularity (number of "split features" k) affects absorption
measurement on the first-letter task using Gemma 2 2B + Gemma Scope L12 16k.

For each k value:
1. Train k-sparse logistic regression probes for each letter
2. Use top-k features by probe weight as "split features"
3. Measure false negatives (probe predicts correctly but no split features fire)
4. Measure absorption rate among false negatives
5. Compare rates across k values to assess robustness

Pass criteria: All k values produce valid absorption rates;
rates vary by <50% across k values (indicating robustness).
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
from sklearn.model_selection import StratifiedKFold
from scipy import stats

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "ablation_probe_sparsity"
SEED = 42

# Sparsity values to test
K_VALUES = [1, 3, 5, 10]

# Absorption thresholds (same as first_letter_validation)
COSINE_THRESHOLD = 0.1
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 25
N_PRESELECT = 200  # Feature pre-selection for probe training

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
    rng = np.random.RandomState(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Vocabulary: same as first_letter_validation ────────────────────────────
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
    """Collect SAE activations for each word using simple prompt."""
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

    return torch.stack(all_acts), all_labels, all_words


def analyze_absorption_for_k(sae_activations, labels, words, W_dec, k_sparse,
                              cosine_threshold=COSINE_THRESHOLD, magnitude_gap=MAGNITUDE_GAP):
    """
    Absorption analysis for a specific k-sparse probe setting.

    Returns per-letter and aggregate statistics.
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

        # Feature pre-selection: top N_PRESELECT by mean difference
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

        # Retrain on full data
        clf_full = LogisticRegression(
            max_iter=1000, C=1.0, solver='lbfgs',
            random_state=SEED, class_weight='balanced',
        )
        clf_full.fit(X_pre, y)
        full_preds = clf_full.predict(X_pre)

        # Top-k split features
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

        # Count false negatives and absorption
        n_tested_letter = 0
        n_fn_letter = 0
        n_abs_letter = 0

        for pi in pos_indices:
            if full_preds[pi] != 1:
                continue
            n_tested_letter += 1

            split_acts = sae_activations[pi, split_indices]
            if (split_acts > 0).any().item():
                continue

            n_fn_letter += 1

            # Check absorption
            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]
            absorbed = False

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

                if abs(cos_sim) > cosine_threshold and mag_ratio >= magnitude_gap:
                    absorbed = True
                    break

            if absorbed:
                n_abs_letter += 1

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
        }

    # Aggregate
    agg_abs_rate = total_absorbed / total_tested if total_tested > 0 else None
    ci_lower, ci_upper = bootstrap_ci(absorption_rates) if absorption_rates else (None, None)
    letters_passing = sum(1 for r in per_letter.values()
                          if r.get("status") == "ok" and r.get("probe_f1", 0) > 0.85)
    letters_tested = sum(1 for r in per_letter.values() if r.get("status") == "ok")

    return {
        "k_sparse": k_sparse,
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
        "absorption_rates_per_letter": [round(r, 4) for r in absorption_rates],
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    set_seeds()

    total_steps = 4 + len(K_VALUES)  # load model, load SAE, build vocab, collect acts, N k-values

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_config": "L12-16k",
        "k_values_tested": K_VALUES,
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        "timestamp_start": datetime.now().isoformat(),
    }

    print("=" * 70)
    print("Ablation: Probe Sparsity (k=1,3,5,10) — Gemma 2 2B")
    print(f"Task ID: {TASK_ID}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"K values: {K_VALUES}")
    print(f"Seed: {SEED}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    device = "cuda:0"

    # ── Step 1: Load model ──────────────────────────────────────────────
    report_progress(1, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 1] Loading Gemma 2 2B model...")
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
    gc.collect()
    torch.cuda.empty_cache()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # ── Step 2: Load SAE L12-16k ────────────────────────────────────────
    report_progress(2, total_steps, "Loading SAE L12-16k")
    print("\n[Step 2] Loading SAE L12-16k...")
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
    print("\n[Step 3] Building single-token English vocabulary...")

    letter_words = get_single_token_english_words(tokenizer, max_per_letter=MAX_PER_LETTER)
    total_words = sum(len(v) for v in letter_words.values())
    n_letters = sum(1 for v in letter_words.values() if len(v) >= 2)
    print(f"  Letters: {n_letters}, Total words: {total_words}")

    results["vocabulary"] = {
        "n_letters": n_letters,
        "per_letter_counts": {k: len(v) for k, v in sorted(letter_words.items())},
        "total_words": total_words,
    }

    # ── Step 4: Collect activations (once, reuse for all k) ─────────────
    report_progress(4, total_steps, "Collecting SAE activations")
    print("\n[Step 4] Collecting SAE activations...")
    t0 = time.time()

    sae_acts, all_labels, all_words = collect_activations(
        model, sae, tokenizer, letter_words, hook_point, device
    )
    print(f"  Collected {len(all_labels)} activations in {time.time()-t0:.1f}s")
    print(f"  SAE shape: {sae_acts.shape}")

    # Free model memory - only need SAE activations and decoder now
    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    # ── Steps 5+: Run absorption for each k ─────────────────────────────
    k_results = {}
    for i, k in enumerate(K_VALUES):
        step = 5 + i
        report_progress(step, total_steps, f"Absorption analysis k={k}")
        print(f"\n{'='*50}")
        print(f"[Step {step}] Absorption analysis with k={k}...")
        print(f"{'='*50}")
        t0 = time.time()

        result_k = analyze_absorption_for_k(
            sae_acts, all_labels, all_words, W_dec,
            k_sparse=k,
            cosine_threshold=COSINE_THRESHOLD,
            magnitude_gap=MAGNITUDE_GAP,
        )

        agg = result_k["aggregate"]
        print(f"\n  k={k} Summary:")
        print(f"    Aggregate absorption rate: {agg['aggregate_absorption_rate']}")
        print(f"    Aggregate FN rate: {agg['aggregate_fn_rate']}")
        print(f"    CI 95%: {agg['bootstrap_ci_95']}")
        print(f"    Mean probe F1: {agg['mean_probe_f1']}")
        print(f"    Letters passing gate (F1>0.85): {agg['letters_above_gate']}/{agg['letters_tested']}")
        print(f"    Time: {time.time()-t0:.1f}s")

        k_results[f"k_{k}"] = result_k

    results["k_results"] = k_results

    # ── Comparative analysis ────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("COMPARATIVE ANALYSIS")
    print("=" * 70)

    comparison = {}
    abs_rates = []
    fn_rates = []
    valid_k = []

    for k in K_VALUES:
        key = f"k_{k}"
        agg = k_results[key]["aggregate"]
        rate = agg["aggregate_absorption_rate"]
        fn_rate = agg["aggregate_fn_rate"]

        print(f"  k={k}: abs_rate={rate}, fn_rate={fn_rate}, "
              f"mean_F1={agg['mean_probe_f1']}, "
              f"CI={agg['bootstrap_ci_95']}")

        comparison[key] = {
            "k": k,
            "absorption_rate": rate,
            "fn_rate": fn_rate,
            "mean_probe_f1": agg["mean_probe_f1"],
            "ci_95": agg["bootstrap_ci_95"],
            "letters_passing_gate": agg["letters_above_gate"],
            "total_tested": agg["total_tested"],
        }

        if rate is not None:
            abs_rates.append(rate)
            fn_rates.append(fn_rate if fn_rate is not None else 0)
            valid_k.append(k)

    # Compute coefficient of variation (CV)
    abs_rates_arr = np.array(abs_rates)
    fn_rates_arr = np.array(fn_rates)

    if len(abs_rates_arr) > 1 and np.mean(abs_rates_arr) > 0:
        abs_cv = float(np.std(abs_rates_arr) / np.mean(abs_rates_arr))
    else:
        abs_cv = float('nan')

    if len(fn_rates_arr) > 1 and np.mean(fn_rates_arr) > 0:
        fn_cv = float(np.std(fn_rates_arr) / np.mean(fn_rates_arr))
    else:
        fn_cv = float('nan')

    # Max variation ratio: max_rate / min_rate
    if len(abs_rates_arr) > 0 and min(abs_rates_arr) > 0:
        max_variation_ratio = float(max(abs_rates_arr) / min(abs_rates_arr))
    elif len(abs_rates_arr) > 0:
        max_variation_ratio = float('inf') if max(abs_rates_arr) > 0 else 1.0
    else:
        max_variation_ratio = float('nan')

    # Spearman correlation between k and absorption rate
    if len(valid_k) >= 3:
        spearman_r, spearman_p = stats.spearmanr(valid_k, abs_rates)
    else:
        spearman_r, spearman_p = float('nan'), float('nan')

    # Per-letter stability: for each letter, compute CV of absorption rate across k values
    per_letter_stability = {}
    letters_all = sorted(set(all_labels))
    for letter in letters_all:
        letter_rates = []
        for k in K_VALUES:
            key = f"k_{k}"
            pl = k_results[key].get("per_letter", {}).get(letter, {})
            r = pl.get("absorption_rate")
            if r is not None and not (isinstance(r, float) and np.isnan(r)):
                letter_rates.append(r)
        if len(letter_rates) >= 2 and np.mean(letter_rates) > 0:
            letter_cv = float(np.std(letter_rates) / np.mean(letter_rates))
        elif len(letter_rates) >= 2:
            letter_cv = 0.0  # All zero rates
        else:
            letter_cv = None
        per_letter_stability[letter] = {
            "rates_by_k": {str(k): k_results[f"k_{k}"].get("per_letter", {}).get(letter, {}).get("absorption_rate") for k in K_VALUES},
            "cv": round(letter_cv, 4) if letter_cv is not None else None,
            "n_valid": len(letter_rates),
        }

    comparison["summary"] = {
        "absorption_rates": {str(k): r for k, r in zip(valid_k, abs_rates)},
        "fn_rates": {str(k): r for k, r in zip(valid_k, fn_rates)},
        "absorption_rate_cv": round(abs_cv, 4) if not np.isnan(abs_cv) else None,
        "fn_rate_cv": round(fn_cv, 4) if not np.isnan(fn_cv) else None,
        "max_variation_ratio": round(max_variation_ratio, 4) if not (np.isnan(max_variation_ratio) or np.isinf(max_variation_ratio)) else None,
        "mean_absorption_rate": round(float(np.mean(abs_rates_arr)), 4) if len(abs_rates_arr) > 0 else None,
        "std_absorption_rate": round(float(np.std(abs_rates_arr)), 4) if len(abs_rates_arr) > 0 else None,
        "spearman_k_vs_rate": {
            "rho": round(float(spearman_r), 4) if not np.isnan(spearman_r) else None,
            "p_value": round(float(spearman_p), 4) if not np.isnan(spearman_p) else None,
        },
    }
    comparison["per_letter_stability"] = per_letter_stability

    results["comparison"] = comparison

    # ── Pass criteria ───────────────────────────────────────────────────
    all_k_valid = all(
        k_results[f"k_{k}"]["aggregate"]["aggregate_absorption_rate"] is not None
        for k in K_VALUES
    )

    # "rates vary by <50% across k values" -- interpret as CV < 0.5
    # or max/min ratio < 1.5
    cv_robust = abs_cv < 0.5 if not np.isnan(abs_cv) else False

    pass_criteria = {
        "all_k_valid": {
            "criterion": "All k values produce valid (non-NaN) absorption rates",
            "value": all_k_valid,
            "passed": all_k_valid,
        },
        "cv_below_50pct": {
            "criterion": "CV of absorption rates across k < 0.5",
            "value": round(abs_cv, 4) if not np.isnan(abs_cv) else None,
            "passed": cv_robust,
        },
        "overall": all_k_valid and cv_robust,
    }
    results["pass_criteria"] = pass_criteria

    # ── Print summary ───────────────────────────────────────────────────
    print(f"\n  Absorption rate CV: {abs_cv:.4f}" if not np.isnan(abs_cv) else "\n  Absorption rate CV: N/A")
    print(f"  FN rate CV: {fn_cv:.4f}" if not np.isnan(fn_cv) else "  FN rate CV: N/A")
    print(f"  Max variation ratio: {max_variation_ratio:.4f}" if not (np.isnan(max_variation_ratio) or np.isinf(max_variation_ratio)) else "  Max variation ratio: N/A")
    print(f"  Spearman(k, abs_rate): rho={spearman_r:.4f}, p={spearman_p:.4f}" if not np.isnan(spearman_r) else "  Spearman: N/A")
    print(f"\n  Pass Criteria:")
    for k, v in pass_criteria.items():
        if k == "overall":
            print(f"    OVERALL: {'PASS' if v else 'FAIL'}")
        else:
            print(f"    {k}: {'PASS' if v['passed'] else 'FAIL'} ({v['value']})")

    # ── Timing ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp_end"] = datetime.now().isoformat()

    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    # ── Save results ────────────────────────────────────────────────────
    output_path = RESULTS_DIR / "ablation_probe_sparsity.json"
    output_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {output_path}")

    mark_done(
        "success" if pass_criteria["overall"] else "partial",
        f"Probe sparsity ablation: k={K_VALUES}, "
        f"abs_rates={[round(r,4) for r in abs_rates]}, "
        f"CV={round(abs_cv,4) if not np.isnan(abs_cv) else 'N/A'}, "
        f"overall={'PASS' if pass_criteria['overall'] else 'FAIL'}",
        results={"absorption_rate_cv": round(abs_cv, 4) if not np.isnan(abs_cv) else None},
    )

    return 0 if pass_criteria["overall"] else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:
        tb = traceback.format_exc()
        print(f"\nFATAL ERROR: {e}\n{tb}")
        mark_done("failed", f"Unhandled exception: {e}")
        sys.exit(1)
