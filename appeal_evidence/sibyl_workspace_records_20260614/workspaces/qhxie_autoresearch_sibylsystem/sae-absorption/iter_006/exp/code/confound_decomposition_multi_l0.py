#!/usr/bin/env python3
"""
confound_decomposition_multi_l0.py — Multi-L0 Confound Decomposition (H2 Primary)

PRIMARY CONTRIBUTION: Decompose false negatives at L0={22, 41, 82, 176} on the first-letter
task into three components:
  (a) Hedging: recoverable at a different L0 (parent latent fires at another L0 operating point)
  (b) Reconstruction error: co-occurring with high reconstruction error (residual norm > 2 sigma)
  (c) Hierarchy-driven: persistent across ALL L0 values AND low reconstruction error — genuine absorption

NOVEL: profiles how the decomposition changes across L0, testing prediction that:
  - hierarchy-driven fraction peaks at intermediate L0 (~41-82)
  - hedging dominates at low L0 (22)
  - reconstruction error grows at high L0 (176)

Uses the improved vocabulary from first_letter_improved.py (50+ words per letter).
Runs on Gemma 2 2B + Gemma Scope L12 16k at 4 L0 operating points.

Generates: exp/results/full/confound_decomposition_multi_l0.json
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
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "confound_decomposition_multi_l0"
SEED = 42
DEVICE = "cuda:0"  # GPU 4 via CUDA_VISIBLE_DEVICES=4

K_SPARSE = 5
COSINE_THRESHOLD = 0.025  # Chanin et al. standard
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 70
N_BOOTSTRAP = 10000

# Four L0 operating points spanning the available range for L12 16k
L0_VALUES = [22, 41, 82, 176]
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


# ── Extended Vocabulary (same as first_letter_improved.py) ──────────────────
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


def encode_through_sae(sae, raw_acts, device=DEVICE, batch_size=128):
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


def find_false_negatives(sae_activations, labels, words, probes, W_dec,
                         residual_norms, cosine_threshold=COSINE_THRESHOLD,
                         magnitude_gap=MAGNITUDE_GAP):
    """
    Find false-negative tokens per letter.
    A false negative: probe predicts positive but all k split features are inactive.
    Returns per-word info: {word, letter, is_fn, is_absorbed, residual_norm, absorbing_features}.
    """
    X = sae_activations.numpy()
    results = []

    for i, (label, word) in enumerate(zip(labels, words)):
        if label not in probes:
            continue
        probe = probes[label]
        y_true = 1
        y_pred = probe["clf"].predict(X[i:i+1])[0]

        if y_pred != 1:
            continue  # Probe doesn't predict positive for this word

        # Check if all split features are inactive
        top_k_idx = probe["top_k_idx"]
        split_activations = X[i, top_k_idx]
        all_split_inactive = np.all(split_activations == 0)

        if not all_split_inactive:
            results.append({
                "word": word, "letter": label,
                "is_fn": False, "is_absorbed": False,
                "residual_norm": float(residual_norms[i]),
                "absorbing_features": [],
            })
            continue

        # This is a false negative
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

        results.append({
            "word": word, "letter": label,
            "is_fn": True, "is_absorbed": is_absorbed,
            "residual_norm": float(residual_norms[i]),
            "absorbing_features": absorbing_features,
        })

    return results


def classify_false_negatives(fn_results_by_l0, residual_stats_by_l0):
    """
    Classify each false negative at each L0 into hedging / reconstruction_error / hierarchy_driven.

    For a given L0 value:
    - Hedging: FN at this L0 but NOT FN at at least one other L0
    - Reconstruction error: FN at this L0, residual norm > 2 sigma above mean
    - Hierarchy-driven: FN at ALL L0 values AND low reconstruction error
    """
    l0_values = sorted(fn_results_by_l0.keys())

    # Build per-word FN status across all L0 values
    all_words = set()
    word_fn_status = {}  # word -> {l0 -> is_fn}
    word_info = {}  # word -> {l0 -> result_dict}

    for l0 in l0_values:
        for r in fn_results_by_l0[l0]:
            word = r["word"]
            all_words.add(word)
            if word not in word_fn_status:
                word_fn_status[word] = {}
                word_info[word] = {}
            word_fn_status[word][l0] = r["is_fn"]
            word_info[word][l0] = r

    # Classify each FN at each L0
    classification = {}
    for l0 in l0_values:
        mean_res = residual_stats_by_l0[l0]["mean"]
        std_res = residual_stats_by_l0[l0]["std"]
        threshold_2sigma = mean_res + 2 * std_res

        hedging = []
        recon_error = []
        hierarchy_driven = []

        for r in fn_results_by_l0[l0]:
            if not r["is_fn"]:
                continue
            word = r["word"]
            letter = r["letter"]

            # Check if FN at ALL L0 values
            fn_at_all = all(
                word_fn_status.get(word, {}).get(other_l0, False)
                for other_l0 in l0_values
            )

            # Check reconstruction error
            high_recon = r["residual_norm"] > threshold_2sigma

            if not fn_at_all:
                # Recoverable at another L0 -> hedging
                hedging.append({
                    "word": word, "letter": letter,
                    "residual_norm": round(r["residual_norm"], 4),
                    "fn_at_l0s": [
                        other_l0 for other_l0 in l0_values
                        if word_fn_status.get(word, {}).get(other_l0, False)
                    ],
                })
            elif high_recon:
                # Persistent FN with high reconstruction error
                recon_error.append({
                    "word": word, "letter": letter,
                    "residual_norm": round(r["residual_norm"], 4),
                    "threshold": round(threshold_2sigma, 4),
                })
            else:
                # Persistent across ALL L0 and low reconstruction error -> genuine absorption
                hierarchy_driven.append({
                    "word": word, "letter": letter,
                    "residual_norm": round(r["residual_norm"], 4),
                    "is_absorbed": r.get("is_absorbed", False),
                    "absorbing_features": r.get("absorbing_features", [])[:3],  # sample
                })

        total_fn = len(hedging) + len(recon_error) + len(hierarchy_driven)
        classification[l0] = {
            "total_false_negatives": total_fn,
            "hedging": len(hedging),
            "reconstruction_error": len(recon_error),
            "hierarchy_driven": len(hierarchy_driven),
            "pct_hedging": round(100 * len(hedging) / max(total_fn, 1), 1),
            "pct_reconstruction_error": round(100 * len(recon_error) / max(total_fn, 1), 1),
            "pct_hierarchy_driven": round(100 * len(hierarchy_driven) / max(total_fn, 1), 1),
            "residual_threshold_2sigma": round(threshold_2sigma, 4),
            "hedging_details": hedging[:5],  # sample
            "recon_error_details": recon_error[:5],
            "hierarchy_details": hierarchy_driven[:5],
        }

    return classification


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    set_seeds()
    start_time = datetime.now()
    report_progress(0, 8, "Starting multi-L0 confound decomposition")

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        "k_sparse": K_SPARSE,
        "l0_values": L0_VALUES,
        "timestamp_start": start_time.isoformat(),
    }

    try:
        # ── Step 1: Load model ──────────────────────────────────────────
        report_progress(1, 8, "Loading Gemma 2 2B model")
        print("[1/8] Loading Gemma 2 2B model...")

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
        report_progress(2, 8, "Building vocabulary")
        print("[2/8] Building single-token vocabulary...")

        letter_words = get_single_token_words(tokenizer)
        total_words = sum(len(v) for v in letter_words.values())
        per_letter_counts = {k: len(v) for k, v in letter_words.items()}
        results["vocabulary"] = {
            "n_letters": len(letter_words),
            "total_words": total_words,
            "per_letter_counts": per_letter_counts,
            "letters_with_50_plus": sum(1 for c in per_letter_counts.values() if c >= 50),
        }
        print(f"  {total_words} total words across {len(letter_words)} letters")

        # ── Step 3: Collect raw model activations ONCE ─────────────────────
        report_progress(3, 9, "Collecting raw model activations (one-time)")
        print("\n[3/9] Collecting raw model activations (one-time for all L0 configs)...")
        hook_point = "blocks.12.hook_resid_post"
        raw_acts, labels, words = collect_raw_activations(
            model, tokenizer, letter_words, hook_point, device=DEVICE
        )
        print(f"  Collected {len(labels)} activation vectors, shape={raw_acts.shape}")

        fn_results_by_l0 = {}
        residual_stats_by_l0 = {}
        per_l0_results = {}

        # ── Step 4-7: Process each L0 value (SAE encoding only) ────────────
        for idx, l0 in enumerate(L0_VALUES):
            step = 4 + idx
            report_progress(step, 9, f"Processing L0={l0} ({idx+1}/{len(L0_VALUES)})")
            print(f"\n[{step}/9] Processing L0={l0}...")

            # Load SAE
            from sae_lens import SAE
            cfg = SAE_CONFIGS[l0]
            sae = SAE.from_pretrained(
                release=cfg["release"],
                sae_id=cfg["sae_id"],
                device=DEVICE,
            )
            print(f"  SAE loaded: {cfg['sae_id']}, d_sae={sae.cfg.d_sae}")

            # Encode through SAE (much faster than re-running model)
            print(f"  Encoding {len(labels)} tokens through SAE...")
            sae_acts, residuals = encode_through_sae(sae, raw_acts, device=DEVICE)
            print(f"  SAE features shape: {sae_acts.shape}")

            # Get decoder weights
            W_dec = sae.W_dec.detach().cpu().numpy()

            # Compute residual statistics
            residual_norms = torch.norm(residuals, dim=1).numpy()
            mean_res = float(np.mean(residual_norms))
            std_res = float(np.std(residual_norms))
            residual_stats_by_l0[l0] = {"mean": mean_res, "std": std_res}

            # Train probes
            print(f"  Training probes...")
            probes = train_probes(sae_acts, labels)
            probe_f1s = {letter: probes[letter]["f1"] for letter in probes}
            mean_f1 = float(np.mean(list(probe_f1s.values())))
            letters_above_085 = sum(1 for f in probe_f1s.values() if f > 0.85)
            print(f"  Mean probe F1: {mean_f1:.3f}, {letters_above_085}/{len(probes)} above 0.85")

            # Find false negatives
            print(f"  Finding false negatives...")
            fn_results = find_false_negatives(
                sae_acts, labels, words, probes, W_dec, residual_norms,
            )
            fn_results_by_l0[l0] = fn_results

            # Per-L0 absorption stats
            fn_words = [r for r in fn_results if r["is_fn"]]
            absorbed_words = [r for r in fn_results if r["is_fn"] and r["is_absorbed"]]
            n_tested = len([r for r in fn_results])  # words with probe prediction = positive
            n_fn = len(fn_words)
            n_absorbed = len(absorbed_words)

            # Per-letter breakdown
            per_letter_stats = {}
            for letter in sorted(probes.keys()):
                letter_fn = [r for r in fn_results if r["letter"] == letter and r["is_fn"]]
                letter_absorbed = [r for r in fn_results if r["letter"] == letter and r["is_fn"] and r["is_absorbed"]]
                letter_tested = [r for r in fn_results if r["letter"] == letter]
                n_t = len(letter_tested)
                n_a = len(letter_absorbed)
                ci_lo, ci_hi = bootstrap_ci_binary(n_a, n_t) if n_t > 0 else (None, None)
                per_letter_stats[letter] = {
                    "probe_f1": round(probes[letter]["f1"], 4),
                    "n_tested": n_t,
                    "n_false_negatives": len(letter_fn),
                    "n_absorbed": n_a,
                    "absorption_rate": round(n_a / max(n_t, 1), 4),
                    "ci_lower": ci_lo,
                    "ci_upper": ci_hi,
                }

            # Aggregate absorption
            agg_rate = n_absorbed / max(n_tested, 1)
            agg_ci_lo, agg_ci_hi = bootstrap_ci_binary(n_absorbed, n_tested)

            per_l0_results[l0] = {
                "sae_config": cfg,
                "n_tested": n_tested,
                "n_false_negatives": n_fn,
                "n_absorbed": n_absorbed,
                "absorption_rate": round(agg_rate, 4),
                "fn_rate": round(n_fn / max(n_tested, 1), 4),
                "bootstrap_ci_95": [agg_ci_lo, agg_ci_hi],
                "mean_probe_f1": round(mean_f1, 4),
                "letters_above_085": letters_above_085,
                "residual_norm_mean": round(mean_res, 4),
                "residual_norm_std": round(std_res, 4),
                "per_letter": per_letter_stats,
            }

            print(f"  L0={l0}: {n_fn} FNs, {n_absorbed} absorbed ({agg_rate:.1%}), "
                  f"mean F1={mean_f1:.3f}")

            # Free SAE memory (keep raw_acts for next L0 iteration)
            del sae, W_dec, sae_acts, residuals
            gc.collect()
            torch.cuda.empty_cache()

        # ── Step 8: Cross-L0 classification ───────────────────────────────
        report_progress(8, 9, "Classifying false negatives across L0 values")
        print("\n[8/9] Cross-L0 false negative classification...")

        classification = classify_false_negatives(fn_results_by_l0, residual_stats_by_l0)

        # Print summary table
        print("\n" + "=" * 80)
        print("MULTI-L0 CONFOUND DECOMPOSITION RESULTS")
        print("=" * 80)
        print(f"{'L0':>6} | {'Total FN':>10} | {'Hedging':>10} | {'Recon Err':>10} | {'Hierarchy':>10} | "
              f"{'%Hedge':>7} | {'%Recon':>7} | {'%Hier':>7} | {'Abs Rate':>9}")
        print("-" * 100)

        for l0 in L0_VALUES:
            c = classification[l0]
            r = per_l0_results[l0]
            print(f"{l0:>6} | {c['total_false_negatives']:>10} | {c['hedging']:>10} | "
                  f"{c['reconstruction_error']:>10} | {c['hierarchy_driven']:>10} | "
                  f"{c['pct_hedging']:>6.1f}% | {c['pct_reconstruction_error']:>6.1f}% | "
                  f"{c['pct_hierarchy_driven']:>6.1f}% | {r['absorption_rate']:>8.1%}")
        print("=" * 100)

        # ── Step 9: Compute trends and statistical tests ──────────────────
        report_progress(9, 9, "Computing trends and statistics")
        print("\n[9/9] Computing trends and statistical tests...")

        # Trend analysis: does hierarchy-driven fraction change with L0?
        l0_arr = np.array(L0_VALUES, dtype=float)
        hier_pct_arr = np.array([classification[l0]["pct_hierarchy_driven"] for l0 in L0_VALUES])
        hedge_pct_arr = np.array([classification[l0]["pct_hedging"] for l0 in L0_VALUES])
        recon_pct_arr = np.array([classification[l0]["pct_reconstruction_error"] for l0 in L0_VALUES])
        abs_rate_arr = np.array([per_l0_results[l0]["absorption_rate"] for l0 in L0_VALUES])
        fn_count_arr = np.array([classification[l0]["total_false_negatives"] for l0 in L0_VALUES])

        # Spearman correlations
        if len(l0_arr) >= 3:
            rho_hier_l0, p_hier_l0 = stats.spearmanr(l0_arr, hier_pct_arr)
            rho_hedge_l0, p_hedge_l0 = stats.spearmanr(l0_arr, hedge_pct_arr)
            rho_recon_l0, p_recon_l0 = stats.spearmanr(l0_arr, recon_pct_arr)
            rho_abs_l0, p_abs_l0 = stats.spearmanr(l0_arr, abs_rate_arr)
        else:
            rho_hier_l0 = p_hier_l0 = rho_hedge_l0 = p_hedge_l0 = None
            rho_recon_l0 = p_recon_l0 = rho_abs_l0 = p_abs_l0 = None

        trend_analysis = {
            "hierarchy_driven_vs_l0": {
                "spearman_rho": round(float(rho_hier_l0), 4) if rho_hier_l0 is not None else None,
                "p_value": round(float(p_hier_l0), 4) if p_hier_l0 is not None else None,
                "values": {l0: round(classification[l0]["pct_hierarchy_driven"], 2) for l0 in L0_VALUES},
            },
            "hedging_vs_l0": {
                "spearman_rho": round(float(rho_hedge_l0), 4) if rho_hedge_l0 is not None else None,
                "p_value": round(float(p_hedge_l0), 4) if p_hedge_l0 is not None else None,
                "values": {l0: round(classification[l0]["pct_hedging"], 2) for l0 in L0_VALUES},
            },
            "recon_error_vs_l0": {
                "spearman_rho": round(float(rho_recon_l0), 4) if rho_recon_l0 is not None else None,
                "p_value": round(float(p_recon_l0), 4) if p_recon_l0 is not None else None,
                "values": {l0: round(classification[l0]["pct_reconstruction_error"], 2) for l0 in L0_VALUES},
            },
            "absorption_rate_vs_l0": {
                "spearman_rho": round(float(rho_abs_l0), 4) if rho_abs_l0 is not None else None,
                "p_value": round(float(p_abs_l0), 4) if p_abs_l0 is not None else None,
                "values": {l0: round(per_l0_results[l0]["absorption_rate"], 4) for l0 in L0_VALUES},
            },
        }

        # Check pass criteria: hierarchy-driven > 80% at L0=22 (replicating pilot)
        hier_at_22 = classification[22]["pct_hierarchy_driven"]
        pass_criteria = {
            "hierarchy_driven_gt_80_at_l0_22": hier_at_22 > 80,
            "hierarchy_driven_pct_at_22": hier_at_22,
            "all_4_l0_computed": len(classification) == 4,
            "min_fn_count": int(min(fn_count_arr)),
            "fn_gt_50_all_l0": all(c >= 50 for c in fn_count_arr),
            "clear_trend": abs(rho_hier_l0) > 0.5 if rho_hier_l0 is not None else False,
        }

        # Overall summary
        abs_rates_str = ", ".join(
            "L0={}: {:.1%}".format(l0, per_l0_results[l0]["absorption_rate"])
            for l0 in L0_VALUES
        )
        if rho_hier_l0 is not None:
            key_finding = (
                "At L0=22 (lowest sparsity): {:.1f}% hierarchy-driven FNs. "
                "Absorption rates: {}. "
                "Hierarchy-driven trend with L0: rho={:.3f}"
            ).format(hier_at_22, abs_rates_str, rho_hier_l0)
        else:
            key_finding = "Insufficient data for trend analysis"

        summary = {
            "pass_criteria": pass_criteria,
            "key_finding": key_finding,
        }

        # ── Assemble final results ────────────────────────────────────────
        results["per_l0_results"] = {str(l0): per_l0_results[l0] for l0 in L0_VALUES}
        results["cross_l0_classification"] = {str(l0): classification[l0] for l0 in L0_VALUES}
        results["trend_analysis"] = trend_analysis
        results["summary"] = summary
        results["timestamp_end"] = datetime.now().isoformat()
        results["duration_seconds"] = round((datetime.now() - start_time).total_seconds(), 1)

        # Save results
        output_path = RESULTS_DIR / "confound_decomposition_multi_l0.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")
        print(f"Duration: {results['duration_seconds']}s")

        mark_done(
            status="success",
            summary=summary["key_finding"],
            results={
                "absorption_rates": {l0: per_l0_results[l0]["absorption_rate"] for l0 in L0_VALUES},
                "hierarchy_pcts": {l0: classification[l0]["pct_hierarchy_driven"] for l0 in L0_VALUES},
                "pass_criteria": pass_criteria,
            }
        )

    except Exception as e:
        tb = traceback.format_exc()
        print(f"\n[ERROR] {e}\n{tb}")
        results["error"] = str(e)
        results["traceback"] = tb
        results["timestamp_end"] = datetime.now().isoformat()

        output_path = RESULTS_DIR / "confound_decomposition_multi_l0.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        mark_done(status="failed", summary=str(e))
        sys.exit(1)

    finally:
        # Clean up
        try:
            del model
        except NameError:
            pass
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
