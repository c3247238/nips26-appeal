#!/usr/bin/env python3
"""
activation_patching_core_words.py — Activation Patching on Persistent Core Words (GATE 1A)

HIGHEST PRIORITY EXPERIMENT. For each of the 9 persistent core words (false negatives at
ALL four L0 values {22, 41, 82, 176}):

1. Load Gemma 2 2B + Gemma Scope L12-16k at L0=82
2. Identify child feature idx (highest-activation absorbing feature)
3. Run the word through the SAE encoder to get feature activations
4. Zero the child feature activation
5. Re-decode through SAE
6. Check if parent feature (first-letter feature from k=5 probe) now activates
7. CONTROL: Zero a random non-child feature and check parent — expect no recovery

Report per-word: word, letter, child_feature_idx, parent_recovered (bool),
                  recovery_magnitude, control_recovered (bool)

PILOT MODE: Runs on all 9 core words (no sampling needed — small number of targets)
but uses a streamlined pipeline for faster iteration.

Generates: exp/results/full/activation_patching_core_words.json
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
TASK_ID = "activation_patching_core_words"
SEED = 42
DEVICE = "cuda:0"  # GPU 1 via CUDA_VISIBLE_DEVICES=1

K_SPARSE = 5
COSINE_THRESHOLD = 0.025
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 70
N_BOOTSTRAP = 10000
N_CONTROL_FEATURES = 10  # Number of random control features to test per word

# Pilot vs Full — for this task pilot = full since we only have 9 words
PILOT_MODE = "--pilot" in sys.argv
MODE = "PILOT" if PILOT_MODE else "FULL"
RESULTS_DIR = WORKSPACE / "exp" / "results" / ("pilots" if PILOT_MODE else "full")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# SAE configurations for all 4 L0 operating points (needed to identify persistent core words)
SAE_CONFIGS = {
    22: {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_22",
         "layer": 12, "width": 16384, "l0_target": 22, "arch": "JumpReLU"},
    41: {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_41",
         "layer": 12, "width": 16384, "l0_target": 41, "arch": "JumpReLU"},
    82: {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_82",
         "layer": 12, "width": 16384, "l0_target": 82, "arch": "JumpReLU"},
    176: {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_176",
          "layer": 12, "width": 16384, "l0_target": 176, "arch": "JumpReLU"},
}

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
    rng = np.random.RandomState(seed)
    if total == 0:
        return None, None
    vals = np.array([1] * successes + [0] * (total - successes))
    bs = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Extended Vocabulary (identical to tightened_hedging.py) ──────────────
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
    """Collect raw model activations for first-letter words."""
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
            raw_act = cache[hook_point][0, -1, :].detach()
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(letter)
            all_words.append(w['word'])

    return torch.stack(all_raw_acts), all_labels, all_words


def encode_through_sae(sae, raw_acts, device=DEVICE, batch_size=256):
    """Encode raw activations through SAE."""
    all_sae_acts = []
    for i in range(0, len(raw_acts), batch_size):
        batch = raw_acts[i:i+batch_size].to(device)
        with torch.no_grad():
            sae_act = sae.encode(batch)
        all_sae_acts.append(sae_act.cpu())
        del batch, sae_act
        torch.cuda.empty_cache()
    return torch.cat(all_sae_acts)


def train_probes(sae_activations, labels, k_sparse=K_SPARSE):
    """Train k-sparse logistic regression probes per letter."""
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
    """Identify false-negative tokens."""
    X = sae_activations.numpy()
    fn_records = []

    for i, (label, word) in enumerate(zip(labels, words)):
        if label not in probes:
            continue
        probe = probes[label]
        y_pred = probe["clf"].predict(X[i:i+1])[0]
        if y_pred != 1:
            continue

        top_k_idx = probe["top_k_idx"]
        split_activations = X[i, top_k_idx]
        all_split_inactive = np.all(split_activations == 0)

        if not all_split_inactive:
            continue

        # Find absorbing features
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


def identify_persistent_core_words(fn_per_l0, words, labels):
    """
    Find words that are false negatives at ALL four L0 values.
    These are the 'persistent core words' — candidates for genuine competitive exclusion.
    """
    # Build word -> set of L0s where it's a false negative
    word_fn_l0s = defaultdict(set)
    for l0_val, fn_records in fn_per_l0.items():
        for fn in fn_records:
            word_fn_l0s[fn["word"]].add(l0_val)

    # Find words that are FN at all 4 L0 values
    all_l0s = set(fn_per_l0.keys())
    persistent = []
    for word, fn_l0s in word_fn_l0s.items():
        if fn_l0s == all_l0s:
            # Find the letter for this word
            idx = words.index(word)
            letter = labels[idx]
            persistent.append({
                "word": word,
                "letter": letter,
                "idx_in_dataset": idx,
                "fn_at_l0s": sorted(fn_l0s),
            })

    persistent.sort(key=lambda x: (x["letter"], x["word"]))
    return persistent


def activation_patching_experiment(sae, raw_acts, sae_acts, probes, W_dec,
                                   core_word_info, all_labels, all_words,
                                   device=DEVICE, n_control=N_CONTROL_FEATURES):
    """
    Core activation patching experiment for a single word.

    Protocol:
    1. Get SAE encoding for the word
    2. Identify child feature (highest-activation absorbing feature)
    3. Zero the child feature
    4. Re-decode through SAE
    5. Re-encode through SAE (to get new feature activations after patching)
    6. Check if parent features (probe top-k) now activate
    7. CONTROL: Zero a random non-child feature, repeat checks

    Returns detailed results dict.
    """
    word = core_word_info["word"]
    letter = core_word_info["letter"]
    idx = core_word_info["idx_in_dataset"]

    X = sae_acts.numpy()

    # Get the original SAE activations for this word
    original_sae_act = sae_acts[idx].clone()  # [d_sae]
    original_raw_act = raw_acts[idx].clone()  # [d_model]

    # Parent features (from probe top-k)
    probe = probes[letter]
    parent_feature_indices = probe["top_k_idx"]  # k=5 indices
    original_parent_acts = original_sae_act.numpy()[parent_feature_indices]

    # Verify: parent features should be INACTIVE (this is a false negative)
    parent_all_zero = bool(np.all(original_parent_acts == 0))

    # Identify child features (absorbing features for this word)
    active_features = np.where(X[idx] > 0)[0]
    split_decoder_mean = W_dec[parent_feature_indices].mean(axis=0)
    split_norm = np.linalg.norm(split_decoder_mean)
    if split_norm > 1e-10:
        split_decoder_mean /= split_norm

    child_candidates = []
    for feat_idx in active_features:
        feat_dec = W_dec[feat_idx]
        feat_norm = np.linalg.norm(feat_dec)
        if feat_norm > 1e-10:
            cos_sim = float(np.dot(feat_dec / feat_norm, split_decoder_mean))
            if abs(cos_sim) > COSINE_THRESHOLD:
                child_candidates.append({
                    "feature_idx": int(feat_idx),
                    "cosine": round(cos_sim, 4),
                    "activation": round(float(X[idx, feat_idx]), 4),
                })

    # Sort by activation magnitude (strongest child first)
    child_candidates.sort(key=lambda x: abs(x["activation"]), reverse=True)

    if not child_candidates:
        return {
            "word": word,
            "letter": letter,
            "status": "no_child_found",
            "parent_all_zero": parent_all_zero,
            "n_active_features": int(len(active_features)),
            "child_candidates": [],
            "patching_result": None,
            "control_results": [],
        }

    # Select the primary child feature (highest activation)
    primary_child = child_candidates[0]
    child_idx = primary_child["feature_idx"]

    # ── PATCHING EXPERIMENT ──────────────────────────────────────────────
    # Method: Zero the child feature in the SAE activation space,
    # then decode back to model activation space, then re-encode through SAE
    # to see if parent features now activate.

    # Step 1: Zero the child feature
    patched_sae_act = original_sae_act.clone()
    patched_sae_act[child_idx] = 0.0

    # Step 2: Decode through SAE to get patched model-space activation
    with torch.no_grad():
        patched_model_act = sae.decode(patched_sae_act.unsqueeze(0).to(device))  # [1, d_model]

    # Step 3: Re-encode patched model-space activation through SAE
    with torch.no_grad():
        re_encoded = sae.encode(patched_model_act)  # [1, d_sae]

    re_encoded_np = re_encoded[0].cpu().numpy()

    # Step 4: Check parent feature activations after patching
    patched_parent_acts = re_encoded_np[parent_feature_indices]
    any_parent_recovered = bool(np.any(patched_parent_acts > 0))
    n_parents_recovered = int(np.sum(patched_parent_acts > 0))
    max_parent_recovery = float(np.max(patched_parent_acts))
    mean_parent_recovery = float(np.mean(patched_parent_acts[patched_parent_acts > 0])) if any_parent_recovered else 0.0

    parent_recovery_details = []
    for j, pidx in enumerate(parent_feature_indices):
        parent_recovery_details.append({
            "parent_feature_idx": int(pidx),
            "original_activation": round(float(original_parent_acts[j]), 6),
            "patched_activation": round(float(patched_parent_acts[j]), 6),
            "recovered": bool(patched_parent_acts[j] > 0),
            "probe_weight": round(float(probe["weights"][j]), 6),
        })

    # Also check: what's the decoder cosine between child and each parent feature?
    child_parent_cosines = []
    child_dec = W_dec[child_idx]
    child_dec_norm = child_dec / (np.linalg.norm(child_dec) + 1e-10)
    for pidx in parent_feature_indices:
        parent_dec = W_dec[pidx]
        parent_dec_norm = parent_dec / (np.linalg.norm(parent_dec) + 1e-10)
        cos = float(np.dot(child_dec_norm, parent_dec_norm))
        child_parent_cosines.append(round(cos, 6))

    # Also try: direct projection method
    # Measure how much of the probe direction the child feature's decoder column captures
    probe_direction = split_decoder_mean  # Already normalized
    child_proj = float(np.dot(child_dec_norm, probe_direction))

    # Alternative patching: instead of decode-reencode, directly check what the
    # child feature's removal does to the probe's predicted direction
    # The child feature contributes: child_activation * W_dec[child_idx] to the reconstruction
    # Removing it changes reconstruction by: -child_activation * W_dec[child_idx]
    child_contribution = float(original_sae_act[child_idx]) * W_dec[child_idx]
    child_contribution_proj_on_probe = float(np.dot(child_contribution / (np.linalg.norm(child_contribution) + 1e-10),
                                                     probe_direction))

    patching_result = {
        "child_feature_idx": child_idx,
        "child_activation": primary_child["activation"],
        "child_cosine_with_probe": primary_child["cosine"],
        "child_projection_on_probe_direction": round(child_proj, 6),
        "child_contribution_magnitude": round(float(np.linalg.norm(child_contribution)), 4),
        "parent_recovered": any_parent_recovered,
        "n_parents_recovered": n_parents_recovered,
        "max_parent_recovery_activation": round(max_parent_recovery, 6),
        "mean_parent_recovery_activation": round(mean_parent_recovery, 6),
        "parent_recovery_details": parent_recovery_details,
        "child_parent_decoder_cosines": child_parent_cosines,
    }

    # ── CONTROL EXPERIMENTS ──────────────────────────────────────────────
    # Zero a random non-child, non-parent feature and check parent recovery
    rng = np.random.RandomState(SEED)
    non_child_non_parent = [
        f for f in active_features
        if f != child_idx and f not in parent_feature_indices
    ]

    control_results = []
    control_features = rng.choice(
        non_child_non_parent,
        size=min(n_control, len(non_child_non_parent)),
        replace=False
    ) if len(non_child_non_parent) >= 1 else []

    for ctrl_idx in control_features:
        ctrl_idx = int(ctrl_idx)
        ctrl_sae_act = original_sae_act.clone()
        ctrl_sae_act[ctrl_idx] = 0.0

        with torch.no_grad():
            ctrl_model_act = sae.decode(ctrl_sae_act.unsqueeze(0).to(device))
            ctrl_re_encoded = sae.encode(ctrl_model_act)

        ctrl_re_np = ctrl_re_encoded[0].cpu().numpy()
        ctrl_parent_acts = ctrl_re_np[parent_feature_indices]
        ctrl_any_recovered = bool(np.any(ctrl_parent_acts > 0))

        control_results.append({
            "control_feature_idx": ctrl_idx,
            "control_feature_activation": round(float(original_sae_act[ctrl_idx]), 4),
            "parent_recovered": ctrl_any_recovered,
            "n_parents_recovered": int(np.sum(ctrl_parent_acts > 0)),
            "max_parent_activation": round(float(np.max(ctrl_parent_acts)), 6),
        })

        del ctrl_sae_act, ctrl_model_act, ctrl_re_encoded
        torch.cuda.empty_cache()

    n_control_with_recovery = sum(1 for c in control_results if c["parent_recovered"])

    # ── ADDITIONAL ANALYSIS: Zero ALL child features simultaneously ──────
    all_child_result = None
    if len(child_candidates) > 1:
        all_child_sae_act = original_sae_act.clone()
        for cc in child_candidates:
            all_child_sae_act[cc["feature_idx"]] = 0.0

        with torch.no_grad():
            all_child_model_act = sae.decode(all_child_sae_act.unsqueeze(0).to(device))
            all_child_re_encoded = sae.encode(all_child_model_act)

        all_child_re_np = all_child_re_encoded[0].cpu().numpy()
        all_child_parent_acts = all_child_re_np[parent_feature_indices]
        all_child_any_recovered = bool(np.any(all_child_parent_acts > 0))

        all_child_result = {
            "n_children_zeroed": len(child_candidates),
            "parent_recovered": all_child_any_recovered,
            "n_parents_recovered": int(np.sum(all_child_parent_acts > 0)),
            "max_parent_activation": round(float(np.max(all_child_parent_acts)), 6),
            "parent_activations": [round(float(a), 6) for a in all_child_parent_acts],
        }

        del all_child_sae_act, all_child_model_act, all_child_re_encoded
        torch.cuda.empty_cache()

    # ── ADDITIONAL ANALYSIS: Residual-based patching ─────────────────────
    # Instead of decode+reencode, directly modify the raw activation:
    # Remove child's contribution from the residual stream, then encode
    residual_patching = None
    raw_act_patched = original_raw_act.clone().to(device)
    child_dec_vec = torch.from_numpy(W_dec[child_idx]).to(device).to(raw_act_patched.dtype)
    child_act_val = original_sae_act[child_idx].item()
    # Remove child feature's contribution: raw_act -= child_act * decoder_column
    raw_act_patched = raw_act_patched - child_act_val * child_dec_vec

    with torch.no_grad():
        resid_re_encoded = sae.encode(raw_act_patched.unsqueeze(0))

    resid_re_np = resid_re_encoded[0].cpu().numpy()
    resid_parent_acts = resid_re_np[parent_feature_indices]
    resid_any_recovered = bool(np.any(resid_parent_acts > 0))

    residual_patching = {
        "method": "subtract_child_decoder_from_raw_activation",
        "parent_recovered": resid_any_recovered,
        "n_parents_recovered": int(np.sum(resid_parent_acts > 0)),
        "max_parent_activation": round(float(np.max(resid_parent_acts)), 6),
        "parent_activations": [round(float(a), 6) for a in resid_parent_acts],
    }

    del raw_act_patched, resid_re_encoded
    torch.cuda.empty_cache()

    return {
        "word": word,
        "letter": letter,
        "status": "completed",
        "parent_all_zero_confirmed": parent_all_zero,
        "n_active_features": int(len(active_features)),
        "n_child_candidates": len(child_candidates),
        "child_candidates": child_candidates[:5],  # Top 5
        "patching_result": patching_result,
        "all_children_zeroed_result": all_child_result,
        "residual_patching_result": residual_patching,
        "control_results": control_results,
        "n_control_with_recovery": n_control_with_recovery,
        "n_control_total": len(control_results),
    }


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    set_seeds()
    start_time = datetime.now()

    print(f"{'='*80}")
    print(f"ACTIVATION PATCHING ON PERSISTENT CORE WORDS — {MODE} MODE")
    print(f"{'='*80}")
    print()

    report_progress(0, 12, f"Starting activation patching ({MODE})")

    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_primary_l0": 82,
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        "k_sparse": K_SPARSE,
        "n_control_features": N_CONTROL_FEATURES,
        "timestamp_start": start_time.isoformat(),
    }

    try:
        # ── Step 1: Load model ──────────────────────────────────────────
        report_progress(1, 12, "Loading Gemma 2 2B model")
        print("[1/12] Loading Gemma 2 2B model...")

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
        report_progress(2, 12, "Building vocabulary")
        print("[2/12] Building single-token vocabulary...")

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
        report_progress(3, 12, "Collecting raw model activations (one-time)")
        print("[3/12] Collecting raw model activations...")
        hook_point = "blocks.12.hook_resid_post"
        raw_acts, labels, words = collect_raw_activations(
            model, tokenizer, letter_words, hook_point, device=DEVICE
        )
        print(f"  Collected {len(labels)} activation vectors, shape={raw_acts.shape}")

        # Free model memory
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(f"  Model freed. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

        # ── Step 4: Identify persistent core words across all 4 L0 values ──
        report_progress(4, 12, "Identifying persistent core words across L0={22,41,82,176}")
        print("[4/12] Identifying persistent core words...")

        from sae_lens import SAE

        fn_per_l0 = {}  # l0_value -> list of FN records

        for l0_val in [22, 41, 82, 176]:
            print(f"  Loading SAE at L0={l0_val}...")
            cfg = SAE_CONFIGS[l0_val]
            sae_temp = SAE.from_pretrained(
                release=cfg["release"], sae_id=cfg["sae_id"], device=DEVICE,
            )

            sae_acts_temp = encode_through_sae(sae_temp, raw_acts, device=DEVICE)
            W_dec_temp = sae_temp.W_dec.detach().cpu().numpy()

            probes_temp = train_probes(sae_acts_temp, labels)
            fn_records_temp = identify_false_negatives(
                sae_acts_temp, labels, words, probes_temp, W_dec_temp,
            )
            fn_per_l0[l0_val] = fn_records_temp

            print(f"    L0={l0_val}: {len(fn_records_temp)} false negatives")

            # Keep L0=82 data for the patching experiment
            if l0_val == 82:
                sae_82 = sae_temp
                sae_acts_82 = sae_acts_temp
                W_dec_82 = W_dec_temp
                probes_82 = probes_temp
            else:
                del sae_temp
            del sae_acts_temp
            gc.collect()
            torch.cuda.empty_cache()

        # Find persistent core words
        persistent_words = identify_persistent_core_words(fn_per_l0, words, labels)
        n_persistent = len(persistent_words)
        print(f"\n  Found {n_persistent} persistent core words (FN at ALL 4 L0 values):")
        for pw in persistent_words:
            print(f"    {pw['word']:12s} letter={pw['letter']} idx={pw['idx_in_dataset']}")

        results["persistent_core_words"] = {
            "n_found": n_persistent,
            "expected": 9,
            "words": persistent_words,
            "fn_counts_per_l0": {str(l0): len(fns) for l0, fns in fn_per_l0.items()},
        }

        if n_persistent == 0:
            msg = "No persistent core words found across all 4 L0 values. Cannot proceed with patching."
            print(f"\n  WARNING: {msg}")
            results["error"] = msg
            results["timestamp_end"] = datetime.now().isoformat()
            output_path = RESULTS_DIR / "activation_patching_core_words.json"
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2, default=str)
            mark_done(status="failed", summary=msg)
            sys.exit(1)

        # ── Step 5: Get child features for each persistent word at L0=82 ──
        report_progress(5, 12, "Identifying child features at L0=82")
        print("\n[5/12] Identifying child features at L0=82...")

        # For each persistent word, find its absorbing features from the L0=82 FN records
        fn_82_by_word = {fn["word"]: fn for fn in fn_per_l0[82]}
        for pw in persistent_words:
            if pw["word"] in fn_82_by_word:
                fn_data = fn_82_by_word[pw["word"]]
                pw["absorbing_features_l0_82"] = fn_data.get("absorbing_features", [])
                pw["is_absorbed_l0_82"] = fn_data.get("is_absorbed", False)
            else:
                pw["absorbing_features_l0_82"] = []
                pw["is_absorbed_l0_82"] = False
            print(f"  {pw['word']:12s}: absorbed={pw['is_absorbed_l0_82']}, "
                  f"n_absorbing_features={len(pw['absorbing_features_l0_82'])}")

        # ── Step 6: Record probe info for each letter ──────────────────────
        report_progress(6, 12, "Recording probe info (parent features)")
        print("\n[6/12] Recording probe info...")

        probe_info = {}
        for letter in sorted(probes_82.keys()):
            p = probes_82[letter]
            probe_info[letter] = {
                "f1": round(p["f1"], 4),
                "top_k_idx": [int(x) for x in p["top_k_idx"]],
                "top_k_weights": [round(float(x), 6) for x in p["weights"]],
            }
        results["probes_l0_82"] = probe_info

        # ── Step 7-8: Run activation patching for each persistent word ─────
        report_progress(7, 12, "Running activation patching experiments")
        print(f"\n[7/12] Running activation patching on {n_persistent} persistent core words...")

        patching_results = []
        for i, pw in enumerate(persistent_words):
            print(f"\n  [{i+1}/{n_persistent}] Patching word '{pw['word']}' (letter {pw['letter']})...")
            report_progress(
                7 + (i / n_persistent),
                12,
                f"Patching word {i+1}/{n_persistent}: {pw['word']}",
            )

            result = activation_patching_experiment(
                sae=sae_82,
                raw_acts=raw_acts,
                sae_acts=sae_acts_82,
                probes=probes_82,
                W_dec=W_dec_82,
                core_word_info=pw,
                all_labels=labels,
                all_words=words,
                device=DEVICE,
            )
            patching_results.append(result)

            # Print summary
            if result["patching_result"]:
                pr = result["patching_result"]
                ctrl_rate = (result["n_control_with_recovery"] / max(result["n_control_total"], 1))
                print(f"    Child feature: {pr['child_feature_idx']} (act={pr['child_activation']:.2f})")
                print(f"    Parent recovered: {pr['parent_recovered']} "
                      f"({pr['n_parents_recovered']}/{K_SPARSE} parents)")
                print(f"    Max parent recovery: {pr['max_parent_recovery_activation']:.6f}")
                print(f"    Control recovery: {result['n_control_with_recovery']}/{result['n_control_total']}")
                if result.get("residual_patching_result"):
                    rp = result["residual_patching_result"]
                    print(f"    Residual patching parent recovered: {rp['parent_recovered']} "
                          f"({rp['n_parents_recovered']}/{K_SPARSE})")
                if result.get("all_children_zeroed_result"):
                    ac = result["all_children_zeroed_result"]
                    print(f"    All children zeroed parent recovered: {ac['parent_recovered']} "
                          f"({ac['n_parents_recovered']}/{K_SPARSE})")
            else:
                print(f"    Status: {result['status']}")

        results["patching_experiments"] = patching_results

        # ── Step 9: Aggregate results ────────────────────────────────────
        report_progress(9, 12, "Computing aggregate statistics")
        print(f"\n[9/12] Aggregate statistics...")

        completed = [r for r in patching_results if r["status"] == "completed"]
        n_completed = len(completed)

        # Primary outcome: how many words show parent recovery after child zeroing?
        n_parent_recovered_primary = sum(
            1 for r in completed
            if r["patching_result"]["parent_recovered"]
        )
        # Residual patching variant
        n_parent_recovered_residual = sum(
            1 for r in completed
            if r.get("residual_patching_result", {}).get("parent_recovered", False)
        )
        # All-children-zeroed variant
        n_parent_recovered_all_children = sum(
            1 for r in completed
            if r.get("all_children_zeroed_result", {}).get("parent_recovered", False)
        )

        # Control statistics
        total_control_tests = sum(r["n_control_total"] for r in completed)
        total_control_recovery = sum(r["n_control_with_recovery"] for r in completed)
        control_recovery_rate = total_control_recovery / max(total_control_tests, 1)

        print(f"  Completed: {n_completed}/{n_persistent}")
        print(f"  Primary patching (decode-reencode): {n_parent_recovered_primary}/{n_completed} show parent recovery")
        print(f"  Residual patching (subtract decoder): {n_parent_recovered_residual}/{n_completed} show parent recovery")
        print(f"  All-children zeroed: {n_parent_recovered_all_children}/{n_completed} show parent recovery")
        print(f"  Control (random feature zeroed): {total_control_recovery}/{total_control_tests} "
              f"({control_recovery_rate:.1%})")

        # Bootstrap CIs
        ci_primary = bootstrap_ci_binary(n_parent_recovered_primary, n_completed)
        ci_residual = bootstrap_ci_binary(n_parent_recovered_residual, n_completed)
        ci_control = bootstrap_ci_binary(total_control_recovery, total_control_tests)

        aggregate = {
            "n_persistent_words": n_persistent,
            "n_completed": n_completed,
            "n_no_child_found": sum(1 for r in patching_results if r["status"] == "no_child_found"),
            "primary_patching": {
                "method": "zero_child_in_SAE_space_then_decode_reencode",
                "n_parent_recovered": n_parent_recovered_primary,
                "recovery_rate": round(n_parent_recovered_primary / max(n_completed, 1), 4),
                "ci_95_lower": ci_primary[0],
                "ci_95_upper": ci_primary[1],
            },
            "residual_patching": {
                "method": "subtract_child_decoder_from_raw_activation_then_encode",
                "n_parent_recovered": n_parent_recovered_residual,
                "recovery_rate": round(n_parent_recovered_residual / max(n_completed, 1), 4),
                "ci_95_lower": ci_residual[0],
                "ci_95_upper": ci_residual[1],
            },
            "all_children_zeroed": {
                "method": "zero_all_absorbing_features_then_decode_reencode",
                "n_parent_recovered": n_parent_recovered_all_children,
                "recovery_rate": round(n_parent_recovered_all_children / max(n_completed, 1), 4),
            },
            "control": {
                "n_control_tests": total_control_tests,
                "n_control_recovery": total_control_recovery,
                "control_recovery_rate": round(control_recovery_rate, 4),
                "ci_95_lower": ci_control[0],
                "ci_95_upper": ci_control[1],
            },
        }
        results["aggregate"] = aggregate

        # ── Step 10: Decision gate ────────────────────────────────────────
        report_progress(10, 12, "Evaluating decision gate")
        print("\n[10/12] Decision gate evaluation...")

        if n_parent_recovered_residual >= 5:
            gate_decision = (
                f"COMPETITIVE EXCLUSION CONFIRMED: {n_parent_recovered_residual}/9 words show "
                f"parent recovery after child feature removal (residual patching). "
                f"Control rate: {control_recovery_rate:.1%}. "
                f"This is causal evidence for competitive exclusion."
            )
        elif n_parent_recovered_residual >= 3:
            gate_decision = (
                f"PARTIAL EVIDENCE: {n_parent_recovered_residual}/9 words show parent recovery. "
                f"Competitive exclusion exists but is not dominant. "
                f"Some persistent FNs may be due to reconstruction limitations."
            )
        else:
            gate_decision = (
                f"ALL-HEDGING NARRATIVE STRENGTHENED: Only {n_parent_recovered_residual}/9 words "
                f"show parent recovery. The 9 persistent core words are persistent FN due to "
                f"reconstruction failure or metric miscalibration, not competitive exclusion."
            )

        results["decision_gate"] = {
            "recommendation": gate_decision,
            "primary_recovery_count": n_parent_recovered_primary,
            "residual_recovery_count": n_parent_recovered_residual,
            "all_children_recovery_count": n_parent_recovered_all_children,
            "control_rate": round(control_recovery_rate, 4),
            "threshold_for_confirmation": ">=5/9",
            "threshold_for_partial": ">=3/9",
        }

        print(f"  GATE DECISION: {gate_decision}")

        # ── Step 11: Per-word summary table ──────────────────────────────
        report_progress(11, 12, "Building summary table")
        print("\n[11/12] Per-word summary table:")
        print(f"  {'Word':12s} {'Letter':6s} {'Child':6s} {'Primary':8s} {'Residual':9s} "
              f"{'AllChild':9s} {'Control':8s}")
        print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*8} {'-'*9} {'-'*9} {'-'*8}")

        summary_table = []
        for r in completed:
            pr = r["patching_result"]
            rp = r.get("residual_patching_result", {})
            ac = r.get("all_children_zeroed_result", {})

            row = {
                "word": r["word"],
                "letter": r["letter"],
                "child_feature_idx": pr["child_feature_idx"],
                "child_activation": pr["child_activation"],
                "child_cosine": pr["child_cosine_with_probe"],
                "primary_parent_recovered": pr["parent_recovered"],
                "primary_n_parents": pr["n_parents_recovered"],
                "primary_max_activation": pr["max_parent_recovery_activation"],
                "residual_parent_recovered": rp.get("parent_recovered", None),
                "residual_n_parents": rp.get("n_parents_recovered", None),
                "residual_max_activation": rp.get("max_parent_activation", None),
                "all_children_parent_recovered": ac.get("parent_recovered", None) if ac else None,
                "control_recovery_rate": round(
                    r["n_control_with_recovery"] / max(r["n_control_total"], 1), 4
                ),
            }
            summary_table.append(row)

            primary_str = "YES" if pr["parent_recovered"] else "no"
            residual_str = "YES" if rp.get("parent_recovered") else "no"
            all_child_str = "YES" if ac and ac.get("parent_recovered") else ("no" if ac else "n/a")
            ctrl_str = f"{r['n_control_with_recovery']}/{r['n_control_total']}"

            print(f"  {r['word']:12s} {r['letter']:6s} {pr['child_feature_idx']:6d} "
                  f"{primary_str:8s} {residual_str:9s} {all_child_str:9s} {ctrl_str:8s}")

        results["summary_table"] = summary_table

        # ── Step 12: Summary ─────────────────────────────────────────────
        report_progress(12, 12, "Finalizing results")

        summary_text = (
            f"Activation patching ({MODE}): {n_persistent} persistent core words identified. "
            f"{n_completed} completed. "
            f"Primary (decode-reencode): {n_parent_recovered_primary}/{n_completed} parent recovery. "
            f"Residual (subtract decoder): {n_parent_recovered_residual}/{n_completed} parent recovery. "
            f"All-children zeroed: {n_parent_recovered_all_children}/{n_completed} parent recovery. "
            f"Control: {total_control_recovery}/{total_control_tests} ({control_recovery_rate:.1%}). "
            f"Gate: {gate_decision[:100]}..."
        )

        results["summary"] = {
            "key_finding": summary_text,
            "pass_criteria": {
                "patching_completed_7_of_9": n_completed >= 7,
                "control_shows_no_recovery": control_recovery_rate < 0.1,
                "per_word_results_table": True,
            },
        }

        results["timestamp_end"] = datetime.now().isoformat()
        results["duration_seconds"] = round((datetime.now() - start_time).total_seconds(), 1)

        # ── Save ─────────────────────────────────────────────────────────
        output_path = RESULTS_DIR / "activation_patching_core_words.json"
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
                "n_persistent": n_persistent,
                "n_completed": n_completed,
                "primary_recovery": n_parent_recovered_primary,
                "residual_recovery": n_parent_recovered_residual,
                "all_children_recovery": n_parent_recovered_all_children,
                "control_rate": round(control_recovery_rate, 4),
                "decision_gate": gate_decision[:200],
            }
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {e}\n{tb}")
        results["error"] = str(e)
        results["traceback"] = tb
        results["timestamp_end"] = datetime.now().isoformat()

        output_path = RESULTS_DIR / "activation_patching_core_words.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        mark_done(status="failed", summary=str(e))
        sys.exit(1)

    finally:
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
