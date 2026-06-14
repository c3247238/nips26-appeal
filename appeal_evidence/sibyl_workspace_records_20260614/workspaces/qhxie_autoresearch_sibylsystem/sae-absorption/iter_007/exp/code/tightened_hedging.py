#!/usr/bin/env python3
"""
tightened_hedging.py — Tightened Hedging Classification (GATE 1B)

For each of 657 false negatives at L0=22 (from confound_decomposition_multi_l0):
1. Identify the k=5 parent-associated latents (probe top-k features at L0=22).
2. Load Gemma Scope L12-16k at L0=176.
3. For each FN token, check whether ANY of the 5 parent latents has activation > 0 at L0=176.
4. Classify as 'strict hedging' ONLY if at least one parent latent fires at L0=176.
5. Compare strict_hedging_rate vs permissive_hedging_rate (98.6%).
6. CONTROL: Apply same classification to shuffled-label false negatives.
7. Report per-letter breakdown.

PILOT MODE: Uses first 200 FN tokens (--pilot flag or default).

Generates: exp/results/full/tightened_hedging.json (or pilots/ for pilot mode)
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

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
ITER006_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_006")
TASK_ID = "tightened_hedging"
SEED = 42
DEVICE = "cuda:0"

K_SPARSE = 5
COSINE_THRESHOLD = 0.025
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 70
N_BOOTSTRAP = 10000
N_SHUFFLE_CONTROLS = 10  # Number of shuffled control replicates

# Pilot mode: limit to first 200 FN tokens for fast validation
PILOT_MODE = "--pilot" in sys.argv or len(sys.argv) == 1  # Default to pilot
PILOT_MAX_TOKENS = 200

# SAE configurations
SAE_CONFIGS = {
    22: {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_22",
         "layer": 12, "width": 16384, "l0_target": 22, "arch": "JumpReLU"},
    176: {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_176",
          "layer": 12, "width": 16384, "l0_target": 176, "arch": "JumpReLU"},
}

MODE = "PILOT" if PILOT_MODE else "FULL"
RESULTS_DIR = WORKSPACE / "exp" / "results" / ("pilots" if PILOT_MODE else "full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID / Progress / Done ─────────────────────────────────────────────────
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
        except Exception:
            pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "results": results or {}, "final_progress": fp,
        "timestamp": datetime.now().isoformat(),
    }, indent=2, default=str))


def set_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bootstrap_ci_binary(successes, total, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Bootstrap CI for a binary proportion."""
    rng = np.random.RandomState(seed)
    if total == 0:
        return None, None
    vals = np.array([1] * successes + [0] * (total - successes))
    bs = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Extended Vocabulary (identical to confound_decomposition_multi_l0.py) ──
EXTENDED_WORDS = {
    'A': [
        'apple', 'animal', 'arrow', 'answer', 'album', 'actor', 'anger', 'above', 'alive', 'audio',
        'award', 'avoid', 'aside', 'adult', 'agree', 'alien', 'armor', 'arena', 'angle', 'angel',
        'alert', 'adopt', 'angry', 'alone', 'after', 'allow', 'argue', 'awful', 'arise', 'amaze',
        'ankle', 'antic', 'ample', 'atlas', 'attic', 'await', 'awake', 'array', 'asset', 'anime',
        'amber', 'amino', 'annoy', 'apart', 'apply', 'areas', 'armed', 'asked', 'audit',
        'aunts', 'axial', 'azure', 'abbey', 'adapt', 'adore', 'aging', 'aided', 'aimed',
        'aired', 'aisle', 'alarm', 'algae', 'align', 'alloy', 'alter',
    ],
    'B': [
        'brain', 'brave', 'bread', 'bring', 'brown', 'build', 'badge', 'basic', 'beach', 'begin',
        'bench', 'birth', 'blade', 'blank', 'blast', 'blaze', 'blend', 'blind', 'block', 'bloom',
        'board', 'bonus', 'bound', 'brand', 'break', 'brief', 'broad', 'brush', 'burst', 'buyer',
        'bliss', 'blown', 'blunt', 'blush', 'bobby', 'bogus', 'bolts', 'bones', 'boost', 'booth',
        'boxer', 'brass', 'bravo', 'breed', 'bride', 'brook', 'broth', 'bunch', 'burns', 'bushy',
        'baker', 'banks', 'baron', 'basin', 'beans', 'beard', 'beats', 'berry', 'bikes',
        'birds', 'bleed', 'bless', 'bluff', 'bombs', 'bonds', 'boots',
    ],
    'C': [
        'chain', 'chair', 'chalk', 'chase', 'cheap', 'check', 'chief', 'child', 'chunk', 'claim',
        'clash', 'class', 'clean', 'clear', 'click', 'climb', 'clock', 'close', 'cloud', 'coach',
        'coast', 'color', 'comic', 'count', 'cover', 'crack', 'craft', 'crash', 'crazy', 'cream',
        'crown', 'cruel', 'crush', 'curve', 'cycle', 'cabin', 'cable', 'camel', 'candy', 'cargo',
        'carol', 'catch', 'cedar', 'cents', 'charm', 'chart', 'chess', 'chest', 'chips', 'choir',
        'chose', 'circa', 'cited', 'civic', 'civil', 'clone', 'cloth', 'clubs', 'codes', 'coins',
        'coral', 'corps', 'costs', 'couch', 'creek', 'crews', 'crime', 'crops', 'crude',
    ],
    'D': [
        'dance', 'death', 'debug', 'delay', 'dense', 'depth', 'devil', 'dirty', 'dozen', 'draft',
        'drain', 'drama', 'drawn', 'dream', 'dress', 'dried', 'drift', 'drink', 'drive', 'drone',
        'drugs', 'drunk', 'dying', 'daily', 'dairy', 'deals', 'debut', 'decks', 'decor', 'deity',
        'delta', 'demon', 'denim', 'depot', 'derby', 'digit', 'disco', 'ditch', 'dizzy', 'dodge',
        'donor', 'doors', 'doses', 'doubt', 'dough', 'downs', 'drags', 'drape', 'draws', 'dread',
        'drips', 'drove', 'drums', 'ducks', 'dumps', 'dunks', 'dusty', 'dwarf', 'dwell',
        'dated', 'diary',
    ],
    'E': [
        'eagle', 'early', 'earth', 'elect', 'elite', 'empty', 'enemy', 'enjoy', 'enter', 'equal',
        'error', 'essay', 'event', 'exact', 'exist', 'extra', 'eager', 'elder', 'email', 'ember',
        'every', 'eight', 'eased', 'eaten', 'edges', 'edits', 'elbow', 'ended', 'epoch', 'equip',
        'erase', 'evade', 'evoke', 'exile', 'expel', 'easel', 'ebony', 'eclat', 'edict',
        'eerie', 'elfin', 'embed', 'emits', 'emoji', 'enact', 'endow', 'envoy', 'ethic',
        'evict', 'exalt', 'exams', 'exert', 'exits', 'expat', 'exude', 'earns',
    ],
    'F': [
        'faced', 'faith', 'false', 'feast', 'fence', 'fiber', 'field', 'fight', 'final', 'flame',
        'flash', 'flesh', 'float', 'flood', 'floor', 'focus', 'force', 'forge', 'forth', 'forum',
        'found', 'frame', 'frank', 'fraud', 'fresh', 'front', 'frost', 'fruit', 'funny', 'fully',
        'fable', 'faded', 'fairy', 'falls', 'fancy', 'fatal', 'fauna', 'favor', 'fears', 'feeds',
        'feels', 'felon', 'femur', 'ferry', 'fetch', 'fetus', 'fever', 'films', 'finds', 'fired',
        'firms', 'fixed', 'flags', 'flank', 'flare', 'flaws', 'flies', 'flock', 'floss',
        'flour', 'flows', 'fluid', 'flush', 'flute', 'focal', 'foggy', 'foils', 'folks', 'fonts',
    ],
    'G': [
        'giant', 'given', 'glass', 'globe', 'glory', 'grace', 'grade', 'grain', 'grand', 'grant',
        'graph', 'grasp', 'grass', 'grave', 'great', 'green', 'greet', 'grief', 'grill', 'grind',
        'groan', 'groom', 'gross', 'group', 'grown', 'guard', 'guess', 'guest', 'guide', 'guilt',
        'gains', 'gamma', 'gangs', 'gates', 'gauge', 'gazer', 'genes', 'genre', 'gifts', 'girls',
        'gives', 'gland', 'glare', 'gleam', 'glide', 'gloom', 'gloss', 'glove', 'glued', 'goats',
        'going', 'golds', 'goods', 'goose', 'grabs', 'graft', 'grams',
        'grape', 'grays', 'graze', 'greed', 'grids', 'grips', 'grits', 'grove', 'grubs',
    ],
    'H': [
        'happy', 'harsh', 'heart', 'heavy', 'hence', 'honor', 'horse', 'hotel', 'house', 'human',
        'humor', 'hurry', 'haven', 'heard', 'hello', 'herbs', 'honey', 'hover', 'habit', 'hedge',
        'heath', 'heels', 'heirs', 'helps', 'heron', 'hinge', 'hippo', 'hired', 'hobby', 'holds',
        'holes', 'holly', 'homes', 'hooks', 'hoped', 'horns', 'hosts', 'hours', 'humid', 'hunts',
        'haste', 'hatch', 'hated', 'haunt', 'hawks', 'heads', 'heals', 'heaps', 'hefty',
        'herds', 'highs', 'hills', 'hints', 'hitch', 'hoist', 'homer',
        'hopes', 'howls', 'hulks', 'humps',
    ],
    'I': [
        'ideal', 'image', 'imply', 'index', 'inner', 'input', 'intro', 'irony', 'ivory', 'inbox',
        'issue', 'infer', 'incur', 'inert', 'irate', 'icing', 'icons', 'idiot', 'idols', 'igloo',
        'inked', 'inter', 'ionic', 'irked', 'items', 'indie', 'intel',
        'irons', 'isles', 'ideas', 'idled', 'inane', 'inept', 'inlet',
    ],
    'J': [
        'jewel', 'joint', 'judge', 'juice', 'jolly', 'joker', 'jumbo', 'jumps', 'jelly', 'jimmy',
        'japan', 'jetty', 'jiffy', 'jolts', 'joust', 'juicy', 'jazzy', 'jeans', 'jerks',
        'jibes', 'joked', 'jones', 'juror', 'jails', 'jaunt', 'jenny', 'jerky', 'joins',
        'judas', 'jumpy', 'junky', 'jaded',
    ],
    'K': [
        'kayak', 'kebab', 'knack', 'kneel', 'knife', 'knock', 'known', 'karma', 'kitty', 'knees',
        'knelt', 'knobs', 'knots', 'knows', 'kazoo', 'keels', 'keeps', 'kelly', 'kicks', 'kills',
        'kinds', 'kings', 'kites', 'knead', 'koala', 'kraft', 'kudos', 'keyed',
    ],
    'L': [
        'label', 'laser', 'later', 'layer', 'legal', 'level', 'light', 'limit', 'local', 'logic',
        'loose', 'lover', 'lucky', 'lunch', 'learn', 'leave', 'lemon', 'lever', 'liner', 'links',
        'lions', 'lists', 'lived', 'loads', 'loans', 'lobby', 'locks', 'lofty', 'login', 'looks',
        'loops', 'lords', 'lotus', 'loved', 'lower', 'loyal', 'lucid', 'lumps', 'lunar', 'lures',
        'lurks', 'lusty', 'lynch', 'lyric', 'laced', 'laden', 'lakes', 'lambs', 'lamps', 'lance',
        'lanes', 'lapse', 'large', 'latch', 'latex', 'lauds', 'lawns', 'leads', 'leafy', 'leaks',
        'leaps', 'lease', 'ledge', 'lefty', 'lends', 'lifts', 'liked', 'limbs', 'linen',
    ],
    'M': [
        'magic', 'major', 'maker', 'march', 'match', 'mayor', 'media', 'mercy', 'metal', 'meter',
        'might', 'minor', 'mixed', 'model', 'money', 'moral', 'mount', 'mouse', 'mouth', 'movie',
        'manor', 'maple', 'marsh', 'medal', 'merge', 'meals', 'means', 'meets', 'melon', 'metro',
        'midst', 'mills', 'minds', 'mines', 'minus', 'mists', 'moans', 'mocks', 'modes', 'moist',
        'molds', 'monks', 'moods', 'moons', 'moose', 'mover', 'mulch', 'mules', 'mummy', 'music',
        'mango', 'mania', 'masks', 'mates', 'mazes', 'menus', 'messy', 'micro',
        'miner', 'mints', 'mirth', 'mocha', 'modal', 'mogul', 'mossy', 'moths',
    ],
    'N': [
        'naked', 'nasty', 'nerve', 'never', 'night', 'noble', 'noise', 'north', 'noted', 'novel',
        'nurse', 'nylon', 'naive', 'naval', 'needs', 'newer', 'nexus', 'ninth', 'nails', 'named',
        'nanny', 'navel', 'necks', 'nests', 'newly', 'niche', 'nifty', 'ninja',
        'nodes', 'norms', 'notch', 'notes', 'nouns', 'nudge', 'nutty', 'nadir', 'names',
        'nappy', 'nerds',
    ],
    'O': [
        'ocean', 'offer', 'often', 'olive', 'onset', 'opera', 'orbit', 'order', 'organ', 'other',
        'outer', 'omega', 'opted', 'oxide', 'ozone', 'occur', 'owing', 'overt', 'oasis',
        'oddly', 'olden', 'onion', 'opens', 'optic', 'otter',
        'ought', 'ounce', 'ovals', 'ovens', 'owned', 'oaken',
    ],
    'P': [
        'panel', 'paper', 'paste', 'patch', 'pause', 'peace', 'pearl', 'phase', 'phone', 'photo',
        'piano', 'piece', 'pilot', 'pitch', 'pixel', 'pizza', 'place', 'plain', 'plane', 'plant',
        'plate', 'plaza', 'plead', 'point', 'polar', 'pound', 'power', 'press', 'price', 'pride',
        'prime', 'print', 'prior', 'probe', 'proof', 'pulse', 'panic', 'parks', 'party', 'paths',
        'paved', 'peaks', 'peers', 'penny', 'perks', 'picks', 'piles', 'pills', 'pines', 'pipes',
        'pivot', 'plaid', 'plans', 'plows', 'pluck', 'plugs', 'plumb', 'plume', 'plump',
        'plush', 'poems', 'poets', 'polls', 'ponds', 'pools', 'porch', 'ports', 'posed', 'posts',
    ],
    'Q': [
        'queen', 'query', 'quest', 'queue', 'quick', 'quiet', 'quilt', 'quirk', 'quota', 'quote',
        'quake', 'qualm', 'quart', 'quasi', 'quail', 'quark', 'quash',
    ],
    'R': [
        'radar', 'raise', 'range', 'rapid', 'ratio', 'reach', 'ready', 'realm', 'rebel', 'refer',
        'reign', 'relax', 'rider', 'rifle', 'rigid', 'rival', 'river', 'robot', 'rocky', 'roman',
        'royal', 'rugby', 'rural', 'rogue', 'route', 'ruler', 'raids', 'rails', 'rains', 'rally',
        'ramps', 'ranch', 'ranks', 'rated', 'raven', 'reads', 'reaps', 'reefs', 'reels',
        'reins', 'remit', 'renal', 'rents', 'repay', 'reply', 'ridge', 'rings',
        'riots', 'risen', 'risks', 'rites', 'roams', 'roast', 'robes', 'rocks', 'roles', 'rolls',
        'rooms', 'roots', 'ropes', 'roses', 'rouge', 'round', 'rowdy', 'rowed', 'ruins', 'ruled',
    ],
    'S': [
        'saint', 'scale', 'scene', 'scope', 'score', 'scout', 'sense', 'serve', 'shade', 'shake',
        'shame', 'shape', 'share', 'sharp', 'shell', 'shift', 'shine', 'shirt', 'shock', 'shoot',
        'short', 'shout', 'sight', 'since', 'skill', 'sleep', 'slice', 'slide', 'smart', 'smile',
        'smoke', 'snake', 'solar', 'solid', 'solve', 'space', 'spark', 'speak', 'speed', 'spell',
        'spend', 'spike', 'spine', 'split', 'spoke', 'sport', 'spray', 'staff', 'stage', 'stain',
        'stake', 'stale', 'stall', 'stamp', 'stand', 'stare', 'stark', 'stars', 'start', 'state',
        'stays', 'steak', 'steal', 'steam', 'steel', 'steep', 'stems', 'steps', 'stern', 'stick',
    ],
    'T': [
        'table', 'taste', 'teach', 'teens', 'theft', 'theme', 'thick', 'thing', 'think', 'those',
        'three', 'throw', 'thumb', 'tiger', 'tight', 'timer', 'tired', 'title', 'toast', 'token',
        'total', 'touch', 'tough', 'tower', 'toxic', 'trace', 'track', 'trade', 'trail', 'train',
        'trait', 'trash', 'treat', 'trend', 'trial', 'trick', 'trips', 'troop', 'truck', 'truly',
        'trump', 'trunk', 'trust', 'truth', 'tubes', 'tulip', 'tumor', 'tunes', 'turns', 'tutor',
        'taken', 'tales', 'talks', 'tanks', 'tapes', 'tasks', 'taxes', 'teams', 'tears', 'tells',
        'tempo', 'tends', 'tense', 'tenth', 'terms', 'tests', 'texts', 'thorn', 'tiles', 'timed',
    ],
    'U': [
        'ultra', 'under', 'union', 'unite', 'unity', 'upper', 'upset', 'urban', 'usage', 'usual',
        'utter', 'uncle', 'unify', 'until', 'using', 'udder', 'ulcer', 'umbra', 'uncut', 'undo',
        'undid', 'undue', 'unfit', 'units', 'unlit', 'unmet', 'unpin', 'unset',
        'untie', 'unwed', 'usher', 'upped', 'urged', 'usurp',
    ],
    'V': [
        'valid', 'value', 'vapor', 'vault', 'venue', 'verse', 'video', 'vigor', 'vinyl', 'viola',
        'viral', 'virus', 'visit', 'vital', 'vivid', 'vocal', 'vodka', 'voice', 'voter', 'vowel',
        'vague', 'valet', 'valve', 'vamps', 'vanes', 'veins', 'venom', 'verge', 'vibes', 'views',
        'vines', 'visor', 'vista', 'votes', 'vowed', 'vying',
    ],
    'W': [
        'watch', 'water', 'wheat', 'wheel', 'where', 'while', 'white', 'whole', 'width', 'witch',
        'woman', 'world', 'worry', 'worst', 'worth', 'wound', 'wrist', 'write', 'wrong', 'wages',
        'walks', 'walls', 'wands', 'wards', 'warns', 'warts', 'waste', 'waves', 'waxes', 'weary',
        'weave', 'weeds', 'weeks', 'weird', 'wells', 'whale', 'whack', 'wharf',
        'which', 'whiff', 'whine', 'whips', 'whirl', 'wider', 'wield',
        'wilds', 'wills', 'winds', 'wines', 'wings', 'winks', 'wiper', 'wired', 'wires',
        'wives', 'woken', 'women', 'woods', 'words', 'works', 'worms', 'worse', 'wraps',
    ],
    'X': ['xenon', 'xerox', 'xeric', 'xylem'],
    'Y': ['yacht', 'yield', 'young', 'youth', 'yards', 'yarns', 'years', 'yeast', 'yells', 'yearn',
          'yoked', 'yolks', 'yours'],
    'Z': ['zebra', 'zeros', 'zones', 'zombie', 'zenith', 'zigzag', 'zodiac', 'zoned', 'zesty'],
}


def get_single_token_words(tokenizer, max_per_letter=MAX_PER_LETTER):
    """Get common English words that tokenize as single tokens in Gemma 2."""
    letter_words = {}
    for letter, candidates in EXTENDED_WORDS.items():
        valid = []
        seen = set()
        for word in candidates:
            wl = word.lower()
            if wl in seen:
                continue
            seen.add(wl)
            for w in [wl, word]:
                encoded = tokenizer.encode(f" {w}", add_special_tokens=False)
                if len(encoded) == 1:
                    valid.append({"word": w, "token_id": encoded[0]})
                    break
            if len(valid) >= max_per_letter:
                break
        letter_words[letter] = valid
    return letter_words


def collect_raw_activations(model, tokenizer, letter_words, hook_point, device=DEVICE):
    """Collect raw model activations for first-letter words (run model ONCE)."""
    all_raw_acts = []
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
            raw_act = cache[hook_point][0, -1, :].detach()  # [d_model]
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(letter)
            all_words.append(w['word'])

    return torch.stack(all_raw_acts), all_labels, all_words


def encode_through_sae(sae, raw_acts, device=DEVICE, batch_size=256):
    """Encode raw activations through SAE, returning features and residuals."""
    all_sae_acts = []
    all_residuals = []

    for i in range(0, len(raw_acts), batch_size):
        batch = raw_acts[i:i+batch_size].to(device)
        with torch.no_grad():
            sae_act = sae.encode(batch)
            reconstructed = sae.decode(sae_act)
            residual = batch - reconstructed
        all_sae_acts.append(sae_act.cpu())
        all_residuals.append(residual.cpu())
        del batch, sae_act, reconstructed, residual
        torch.cuda.empty_cache()

    return torch.cat(all_sae_acts), torch.cat(all_residuals)


def train_probes(sae_activations, labels, k_sparse=K_SPARSE):
    """Train k-sparse logistic regression probes per letter. Returns dict of probes and metrics."""
    X = sae_activations.numpy()
    unique_labels = sorted(set(labels))
    probes = {}

    for letter in unique_labels:
        mask = np.array([l == letter for l in labels])
        n_pos = int(mask.sum())
        if n_pos < 2:
            continue
        y = mask.astype(int)
        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs', random_state=SEED)
            clf.fit(X, y)
        except Exception:
            continue

        y_pred = clf.predict(X)
        f1 = f1_score(y, y_pred, zero_division=0)
        # Get top-k features by absolute weight
        weights = clf.coef_[0]
        top_k_idx = np.argsort(np.abs(weights))[-k_sparse:]

        probes[letter] = {
            "clf": clf,
            "f1": float(f1),
            "n_pos": n_pos,
            "top_k_idx": top_k_idx,
            "weights": weights[top_k_idx],
        }
    return probes


def identify_false_negatives(sae_activations, labels, words, probes, W_dec,
                             cosine_threshold=COSINE_THRESHOLD, magnitude_gap=MAGNITUDE_GAP):
    """
    Identify false-negative tokens: probe predicts positive but all k-split features are inactive.
    Returns list of FN records with word, letter, index, absorbing_features info.
    """
    X = sae_activations.numpy()
    fn_records = []

    for i, (label, word) in enumerate(zip(labels, words)):
        if label not in probes:
            continue
        probe = probes[label]
        y_pred = probe["clf"].predict(X[i:i+1])[0]

        if y_pred != 1:
            continue  # Probe doesn't predict positive for this word

        # Check if all split features are inactive
        top_k_idx = probe["top_k_idx"]
        split_activations = X[i, top_k_idx]
        all_split_inactive = np.all(split_activations == 0)

        if not all_split_inactive:
            continue  # Not a false negative

        # This IS a false negative
        # Check absorption: find active features whose decoder aligns with split direction
        active_features = np.where(X[i] > 0)[0]
        absorbing_features = []

        if len(active_features) > 0:
            split_decoder_mean = W_dec[top_k_idx].mean(axis=0)
            split_norm = np.linalg.norm(split_decoder_mean)
            if split_norm > 1e-10:
                split_decoder_mean /= split_norm

                for feat_idx in active_features:
                    feat_dec = W_dec[feat_idx]
                    feat_norm = np.linalg.norm(feat_dec)
                    if feat_norm > 1e-10:
                        cos_sim = float(np.dot(feat_dec / feat_norm, split_decoder_mean))
                        if abs(cos_sim) > cosine_threshold:
                            feat_activations = X[:, feat_idx]
                            active_vals = feat_activations[feat_activations > 0]
                            mean_active = float(np.mean(active_vals)) if len(active_vals) > 0 else 1e-10
                            mag_ratio = float(X[i, feat_idx]) / max(mean_active, 1e-10)
                            if mag_ratio > magnitude_gap:
                                absorbing_features.append({
                                    "feature_idx": int(feat_idx),
                                    "cosine": round(cos_sim, 4),
                                    "mag_ratio": round(mag_ratio, 4),
                                    "activation": round(float(X[i, feat_idx]), 4),
                                })

        is_absorbed = len(absorbing_features) > 0

        fn_records.append({
            "word": word,
            "letter": label,
            "idx_in_dataset": i,
            "is_absorbed": is_absorbed,
            "absorbing_features": absorbing_features,
        })

    return fn_records


def tightened_hedging_check(fn_records_l0_22, sae_acts_176, probes_22, probes_176):
    """
    For each FN at L0=22, check whether ANY of the k=5 parent-associated latents
    (from L0=22 probe) has activation > 0 at L0=176.

    'strict hedging' = at least one parent latent fires at L0=176
    'non-hedging' = NONE of the parent latents fire at L0=176 (persistent absence)

    Returns classification for each FN.
    """
    X_176 = sae_acts_176.numpy()
    classified = []

    for fn in fn_records_l0_22:
        letter = fn["letter"]
        idx = fn["idx_in_dataset"]

        if letter not in probes_22:
            classified.append({
                **fn,
                "strict_hedging": None,
                "parent_latents_at_176": [],
                "reason": "no_probe_at_l0_22",
            })
            continue

        # Get the k=5 parent-associated latents from L0=22 probe
        parent_latent_indices = probes_22[letter]["top_k_idx"]  # shape (5,)

        # Check activations of these parent latents at L0=176
        parent_acts_at_176 = X_176[idx, parent_latent_indices]
        any_parent_fires = bool(np.any(parent_acts_at_176 > 0))

        # Also record which specific parent latents fire and their magnitudes
        parent_details = []
        for j, pidx in enumerate(parent_latent_indices):
            act = float(X_176[idx, pidx])
            parent_details.append({
                "latent_idx": int(pidx),
                "activation_at_176": round(act, 6),
                "fires": act > 0,
            })

        # ALSO check probes at L0=176: does the probe predict positive at L0=176?
        l0_176_still_fn = True  # Default: still a false negative
        if letter in probes_176:
            y_pred_176 = probes_176[letter]["clf"].predict(X_176[idx:idx+1])[0]
            l0_176_still_fn = (y_pred_176 == 1) and np.all(X_176[idx, probes_176[letter]["top_k_idx"]] == 0)
        else:
            l0_176_still_fn = None  # Can't determine

        classified.append({
            "word": fn["word"],
            "letter": fn["letter"],
            "idx_in_dataset": idx,
            "is_absorbed_at_l0_22": fn["is_absorbed"],
            "strict_hedging": any_parent_fires,
            "n_parent_latents_firing": int(np.sum(parent_acts_at_176 > 0)),
            "parent_latents_at_176": parent_details,
            "still_fn_at_l0_176": l0_176_still_fn,
        })

    return classified


def shuffled_control(fn_records_l0_22, sae_acts_176, probes_22, labels, n_replicates=N_SHUFFLE_CONTROLS):
    """
    Control: shuffle letter labels of the FN tokens and re-apply the tightened hedging check.
    This tests whether the strict hedging rate is specific to the true letter assignment.
    """
    rng = np.random.RandomState(SEED)
    X_176 = sae_acts_176.numpy()
    control_rates = []

    all_letters = sorted(set(labels))

    for rep in range(n_replicates):
        strict_count = 0
        total = 0

        for fn in fn_records_l0_22:
            idx = fn["idx_in_dataset"]
            # Assign a random letter (different from true)
            available = [l for l in all_letters if l != fn["letter"] and l in probes_22]
            if not available:
                continue
            shuffled_letter = rng.choice(available)

            parent_latent_indices = probes_22[shuffled_letter]["top_k_idx"]
            parent_acts_at_176 = X_176[idx, parent_latent_indices]
            any_parent_fires = bool(np.any(parent_acts_at_176 > 0))

            if any_parent_fires:
                strict_count += 1
            total += 1

        rate = strict_count / max(total, 1)
        control_rates.append(round(rate, 4))

    return {
        "n_replicates": n_replicates,
        "mean_strict_rate": round(float(np.mean(control_rates)), 4),
        "std_strict_rate": round(float(np.std(control_rates)), 4),
        "min_strict_rate": round(float(np.min(control_rates)), 4),
        "max_strict_rate": round(float(np.max(control_rates)), 4),
        "per_replicate_rates": control_rates,
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    set_seeds()
    start_time = datetime.now()

    print(f"{'='*80}")
    print(f"TIGHTENED HEDGING CLASSIFICATION — {MODE} MODE")
    print(f"{'='*80}")
    if PILOT_MODE:
        print(f"  Pilot: limiting to first {PILOT_MAX_TOKENS} FN tokens")
    print()

    report_progress(0, 10, f"Starting tightened hedging ({MODE})")

    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "seed": SEED,
        "model": "gemma-2-2b",
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        "k_sparse": K_SPARSE,
        "pilot_max_tokens": PILOT_MAX_TOKENS if PILOT_MODE else None,
        "n_shuffle_controls": N_SHUFFLE_CONTROLS,
        "timestamp_start": start_time.isoformat(),
    }

    try:
        # ── Step 1: Load model ──────────────────────────────────────────
        report_progress(1, 10, "Loading Gemma 2 2B model")
        print("[1/10] Loading Gemma 2 2B model...")

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
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Model loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # ── Step 2: Build vocabulary ──────────────────────────────────────
        report_progress(2, 10, "Building vocabulary")
        print("[2/10] Building single-token vocabulary...")

        letter_words = get_single_token_words(tokenizer)
        total_words = sum(len(v) for v in letter_words.values())
        per_letter_counts = {k: len(v) for k, v in letter_words.items()}
        results["vocabulary"] = {
            "n_letters": len(letter_words),
            "total_words": total_words,
            "per_letter_counts": per_letter_counts,
        }
        print(f"  {total_words} total words across {len(letter_words)} letters")

        # ── Step 3: Collect raw model activations ONCE ─────────────────────
        report_progress(3, 10, "Collecting raw model activations (one-time)")
        print("[3/10] Collecting raw model activations...")
        hook_point = "blocks.12.hook_resid_post"
        raw_acts, labels, words = collect_raw_activations(
            model, tokenizer, letter_words, hook_point, device=DEVICE
        )
        print(f"  Collected {len(labels)} activation vectors, shape={raw_acts.shape}")

        # Free model memory (we only need raw activations from here)
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Model freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # ── Step 4: Load SAE at L0=22, encode, train probes, find FNs ──────
        report_progress(4, 10, "Processing L0=22: encode + probes + FN detection")
        print("[4/10] Loading SAE at L0=22...")

        from sae_lens import SAE
        cfg_22 = SAE_CONFIGS[22]
        sae_22 = SAE.from_pretrained(
            release=cfg_22["release"], sae_id=cfg_22["sae_id"], device=DEVICE,
        )
        print(f"  SAE L0=22 loaded: d_sae={sae_22.cfg.d_sae}")

        print("  Encoding through SAE L0=22...")
        sae_acts_22, residuals_22 = encode_through_sae(sae_22, raw_acts, device=DEVICE)
        W_dec_22 = sae_22.W_dec.detach().cpu().numpy()
        print(f"  SAE acts shape: {sae_acts_22.shape}")

        print("  Training probes at L0=22...")
        probes_22 = train_probes(sae_acts_22, labels)
        probe_f1s_22 = {letter: probes_22[letter]["f1"] for letter in probes_22}
        mean_f1_22 = float(np.mean(list(probe_f1s_22.values())))
        print(f"  Mean probe F1 at L0=22: {mean_f1_22:.3f}")

        # Record probe top-k indices for each letter (these are the "parent latents")
        parent_latent_map = {}
        for letter, probe_data in probes_22.items():
            parent_latent_map[letter] = {
                "top_k_indices": [int(x) for x in probe_data["top_k_idx"]],
                "top_k_weights": [round(float(x), 6) for x in probe_data["weights"]],
                "probe_f1": probe_data["f1"],
            }
        results["parent_latent_map_l0_22"] = parent_latent_map

        print("  Finding false negatives at L0=22...")
        fn_records = identify_false_negatives(
            sae_acts_22, labels, words, probes_22, W_dec_22,
        )
        total_fn = len(fn_records)
        total_absorbed = sum(1 for r in fn_records if r["is_absorbed"])
        print(f"  Found {total_fn} false negatives ({total_absorbed} absorbed)")

        # Per-letter FN breakdown
        fn_per_letter = defaultdict(int)
        for r in fn_records:
            fn_per_letter[r["letter"]] += 1
        results["fn_at_l0_22"] = {
            "total": total_fn,
            "total_absorbed": total_absorbed,
            "per_letter": dict(fn_per_letter),
        }

        # Pilot mode: limit FN records
        if PILOT_MODE and len(fn_records) > PILOT_MAX_TOKENS:
            print(f"  PILOT: Limiting from {len(fn_records)} to {PILOT_MAX_TOKENS} FN tokens")
            fn_records = fn_records[:PILOT_MAX_TOKENS]

        # Free SAE L0=22
        del sae_22, residuals_22
        gc.collect()
        torch.cuda.empty_cache()

        # ── Step 5: Load SAE at L0=176, encode ──────────────────────────
        report_progress(5, 10, "Processing L0=176: encode + probes")
        print("[5/10] Loading SAE at L0=176...")

        cfg_176 = SAE_CONFIGS[176]
        sae_176 = SAE.from_pretrained(
            release=cfg_176["release"], sae_id=cfg_176["sae_id"], device=DEVICE,
        )
        print(f"  SAE L0=176 loaded: d_sae={sae_176.cfg.d_sae}")

        print("  Encoding through SAE L0=176...")
        sae_acts_176, _ = encode_through_sae(sae_176, raw_acts, device=DEVICE)
        print(f"  SAE acts shape: {sae_acts_176.shape}")

        # Train probes at L0=176 (for additional validation)
        print("  Training probes at L0=176...")
        probes_176 = train_probes(sae_acts_176, labels)
        probe_f1s_176 = {letter: probes_176[letter]["f1"] for letter in probes_176}
        mean_f1_176 = float(np.mean(list(probe_f1s_176.values())))
        print(f"  Mean probe F1 at L0=176: {mean_f1_176:.3f}")

        results["probe_f1_l0_176"] = {
            "mean": round(mean_f1_176, 4),
            "per_letter": {l: round(v, 4) for l, v in probe_f1s_176.items()},
        }

        # Free SAE L0=176 (keep acts)
        del sae_176
        gc.collect()
        torch.cuda.empty_cache()

        # ── Step 6: Tightened hedging check ──────────────────────────────
        report_progress(6, 10, "Tightened hedging classification")
        print(f"\n[6/10] Tightened hedging classification on {len(fn_records)} FN tokens...")

        classified = tightened_hedging_check(
            fn_records, sae_acts_176, probes_22, probes_176,
        )

        # Compute aggregate rates
        n_classified = len(classified)
        n_strict = sum(1 for c in classified if c["strict_hedging"] is True)
        n_non_hedging = sum(1 for c in classified if c["strict_hedging"] is False)
        n_unknown = sum(1 for c in classified if c["strict_hedging"] is None)

        strict_rate = n_strict / max(n_classified - n_unknown, 1)
        non_hedging_rate = n_non_hedging / max(n_classified - n_unknown, 1)

        # Also check: how many strict-hedging tokens are ALSO no longer FN at L0=176?
        n_strict_and_not_fn_176 = sum(
            1 for c in classified
            if c["strict_hedging"] is True and c["still_fn_at_l0_176"] is False
        )
        n_strict_and_still_fn_176 = sum(
            1 for c in classified
            if c["strict_hedging"] is True and c["still_fn_at_l0_176"] is True
        )

        print(f"  Results ({n_classified} tokens, {n_unknown} unknown):")
        print(f"    Strict hedging (parent latent fires at L0=176): {n_strict} ({strict_rate:.1%})")
        print(f"    Non-hedging (NO parent latent fires at L0=176): {n_non_hedging} ({non_hedging_rate:.1%})")
        print(f"    Of strict hedging: {n_strict_and_not_fn_176} are NOT FN at L0=176, "
              f"{n_strict_and_still_fn_176} are still FN at L0=176")

        # Bootstrap CIs
        ci_strict = bootstrap_ci_binary(n_strict, n_classified - n_unknown)
        ci_non = bootstrap_ci_binary(n_non_hedging, n_classified - n_unknown)

        results["tightened_classification"] = {
            "n_fn_tokens": n_classified,
            "n_unknown": n_unknown,
            "n_valid": n_classified - n_unknown,
            "strict_hedging": {
                "count": n_strict,
                "rate": round(strict_rate, 4),
                "ci_95_lower": ci_strict[0],
                "ci_95_upper": ci_strict[1],
                "description": "At least 1 of 5 parent-associated latents fires at L0=176",
            },
            "non_hedging": {
                "count": n_non_hedging,
                "rate": round(non_hedging_rate, 4),
                "ci_95_lower": ci_non[0],
                "ci_95_upper": ci_non[1],
                "description": "NONE of the 5 parent-associated latents fire at L0=176",
            },
            "strict_and_not_fn_at_176": n_strict_and_not_fn_176,
            "strict_and_still_fn_at_176": n_strict_and_still_fn_176,
            "comparison_with_permissive": {
                "permissive_rate": 0.986,
                "strict_rate": round(strict_rate, 4),
                "difference": round(0.986 - strict_rate, 4),
                "interpretation": (
                    "Strict rate is close to permissive rate, validating headline claim"
                    if strict_rate > 0.8 else
                    "MAJOR DISCREPANCY: strict rate << permissive, headline 98.6% is an upper bound"
                ),
            },
        }

        # ── Step 7: Per-letter breakdown ──────────────────────────────────
        report_progress(7, 10, "Computing per-letter breakdown")
        print("\n[7/10] Per-letter breakdown...")

        per_letter_results = {}
        for letter in sorted(set(c["letter"] for c in classified)):
            letter_classified = [c for c in classified if c["letter"] == letter]
            n_l = len(letter_classified)
            n_strict_l = sum(1 for c in letter_classified if c["strict_hedging"] is True)
            n_non_l = sum(1 for c in letter_classified if c["strict_hedging"] is False)
            n_unk_l = sum(1 for c in letter_classified if c["strict_hedging"] is None)
            valid_l = n_l - n_unk_l

            strict_rate_l = n_strict_l / max(valid_l, 1)
            non_rate_l = n_non_l / max(valid_l, 1)

            per_letter_results[letter] = {
                "n_fn": n_l,
                "n_strict_hedging": n_strict_l,
                "n_non_hedging": n_non_l,
                "strict_hedging_rate": round(strict_rate_l, 4),
                "non_hedging_rate": round(non_rate_l, 4),
            }
            print(f"  {letter}: {n_l} FNs, strict={n_strict_l} ({strict_rate_l:.1%}), "
                  f"non-hedging={n_non_l} ({non_rate_l:.1%})")

        results["per_letter_breakdown"] = per_letter_results

        # ── Step 8: Shuffled label control ────────────────────────────────
        report_progress(8, 10, "Running shuffled-label control")
        print(f"\n[8/10] Running shuffled-label control ({N_SHUFFLE_CONTROLS} replicates)...")

        control_results = shuffled_control(
            fn_records, sae_acts_176, probes_22, labels, n_replicates=N_SHUFFLE_CONTROLS,
        )
        results["shuffled_control"] = control_results

        print(f"  Control strict hedging rate: {control_results['mean_strict_rate']:.4f} "
              f"(+/- {control_results['std_strict_rate']:.4f})")
        print(f"  True strict hedging rate: {strict_rate:.4f}")
        print(f"  Difference: {strict_rate - control_results['mean_strict_rate']:.4f}")

        # Statistical comparison
        # One-sample z-test: is true rate different from control mean?
        if control_results['std_strict_rate'] > 0:
            z_score = (strict_rate - control_results['mean_strict_rate']) / control_results['std_strict_rate']
            p_val = 2 * (1 - stats.norm.cdf(abs(z_score)))
        else:
            z_score = float('inf') if strict_rate != control_results['mean_strict_rate'] else 0.0
            p_val = 0.0

        results["control_comparison"] = {
            "true_strict_rate": round(strict_rate, 4),
            "control_mean_strict_rate": control_results['mean_strict_rate'],
            "z_score": round(float(z_score), 4),
            "p_value": round(float(p_val), 6),
            "significant_at_005": p_val < 0.05,
            "interpretation": (
                "True strict rate significantly different from shuffled control"
                if p_val < 0.05 else
                "True strict rate NOT significantly different from shuffled control"
            ),
        }

        # ── Step 9: Detailed analysis of parent latent firing patterns ────
        report_progress(9, 10, "Analyzing parent latent firing patterns")
        print("\n[9/10] Analyzing parent latent firing patterns...")

        # How many of the 5 parent latents fire at L0=176 (distribution)?
        firing_counts = [c["n_parent_latents_firing"] for c in classified if c["strict_hedging"] is not None]
        firing_dist = {}
        for n in range(K_SPARSE + 1):
            count = sum(1 for fc in firing_counts if fc == n)
            firing_dist[str(n)] = {
                "count": count,
                "fraction": round(count / max(len(firing_counts), 1), 4),
            }
        print(f"  Distribution of parent latents firing at L0=176:")
        for n in range(K_SPARSE + 1):
            print(f"    {n}/5 fire: {firing_dist[str(n)]['count']} ({firing_dist[str(n)]['fraction']:.1%})")

        results["parent_latent_firing_distribution"] = firing_dist

        # Mean activations of parent latents when they fire
        all_parent_acts = []
        for c in classified:
            for p in c.get("parent_latents_at_176", []):
                if p["fires"]:
                    all_parent_acts.append(p["activation_at_176"])

        if all_parent_acts:
            results["parent_activation_when_firing"] = {
                "mean": round(float(np.mean(all_parent_acts)), 6),
                "median": round(float(np.median(all_parent_acts)), 6),
                "std": round(float(np.std(all_parent_acts)), 6),
                "min": round(float(np.min(all_parent_acts)), 6),
                "max": round(float(np.max(all_parent_acts)), 6),
                "n_total": len(all_parent_acts),
            }
            print(f"  Parent latent activation when firing: "
                  f"mean={np.mean(all_parent_acts):.4f}, "
                  f"median={np.median(all_parent_acts):.4f}")

        # ── Step 10: Per-word detail for non-hedging (persistent) cases ────
        report_progress(10, 10, "Collecting persistent (non-hedging) word details")
        print("\n[10/10] Non-hedging words (parent latents DON'T fire at L0=176):")

        persistent_words = [
            c for c in classified if c["strict_hedging"] is False
        ]
        persistent_sample = []
        for pw in persistent_words[:20]:
            entry = {
                "word": pw["word"],
                "letter": pw["letter"],
                "is_absorbed_at_l0_22": pw["is_absorbed_at_l0_22"],
                "still_fn_at_l0_176": pw["still_fn_at_l0_176"],
                "parent_latent_activations_at_176": [
                    round(p["activation_at_176"], 6) for p in pw["parent_latents_at_176"]
                ],
            }
            persistent_sample.append(entry)
            print(f"  {pw['word']:12s} letter={pw['letter']} absorbed_l22={pw['is_absorbed_at_l0_22']} "
                  f"still_fn_176={pw['still_fn_at_l0_176']}")

        results["persistent_non_hedging_sample"] = persistent_sample

        # ── Summary ──────────────────────────────────────────────────────
        summary_text = (
            f"Tightened hedging ({MODE}): {n_classified} FN tokens from L0=22. "
            f"Strict hedging rate: {strict_rate:.1%} (at least 1 of 5 parent latents fires at L0=176). "
            f"Non-hedging rate: {non_hedging_rate:.1%}. "
            f"Permissive rate (from iter_006): 98.6%. "
            f"Control (shuffled): {control_results['mean_strict_rate']:.1%}. "
            f"Difference (true-control): {strict_rate - control_results['mean_strict_rate']:.4f}."
        )

        results["summary"] = {
            "key_finding": summary_text,
            "pass_criteria": {
                "strict_and_permissive_computed": True,
                "per_letter_breakdown_available": len(per_letter_results) > 0,
                "control_completed": True,
                "clear_comparison_table": True,
            },
        }

        # Decision gate information
        if strict_rate < 0.5:
            gate_recommendation = (
                "MAJOR NARRATIVE REVISION NEEDED. Strict hedging rate < 50%: "
                "the 98.6% headline is a permissive upper bound. "
                "Strict rate becomes the primary reported number."
            )
        elif strict_rate > 0.8:
            gate_recommendation = (
                "Headline 98.6% validated by strict check (strict rate > 80%). "
                "Minor revision: add strict rate as confirmation."
            )
        else:
            gate_recommendation = (
                f"Moderate discrepancy: strict rate {strict_rate:.1%} vs permissive 98.6%. "
                "Report both rates, discuss the gap as informative about hedging mechanism."
            )
        results["decision_gate"] = gate_recommendation

        results["timestamp_end"] = datetime.now().isoformat()
        results["duration_seconds"] = round((datetime.now() - start_time).total_seconds(), 1)

        # ── Save ─────────────────────────────────────────────────────────
        output_path = RESULTS_DIR / "tightened_hedging.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n{'='*80}")
        print(f"Results saved to {output_path}")
        print(f"Duration: {results['duration_seconds']}s")
        print(f"{'='*80}")

        mark_done(
            status="success",
            summary=summary_text,
            results={
                "strict_hedging_rate": round(strict_rate, 4),
                "non_hedging_rate": round(non_hedging_rate, 4),
                "permissive_rate": 0.986,
                "control_rate": control_results['mean_strict_rate'],
                "n_fn_tokens": n_classified,
                "decision_gate": gate_recommendation,
            }
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {e}\n{tb}")
        results["error"] = str(e)
        results["traceback"] = tb
        results["timestamp_end"] = datetime.now().isoformat()

        output_path = RESULTS_DIR / "tightened_hedging.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        mark_done(status="failed", summary=str(e))
        sys.exit(1)

    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
