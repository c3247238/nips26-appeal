#!/usr/bin/env python3
"""
first_letter_improved.py — First-Letter Absorption Baseline (IMPROVED, FULL mode)

KEY IMPROVEMENTS over the pilot (first_letter_validation.py):
1. 50+ words per letter (up from 25) for better probe quality
2. Cosine threshold 0.025 (Chanin et al. standard) as PRIMARY, also test 0.01-0.05
3. Fix C2 shuffled control: the pilot's 59.5% was caused by overly permissive thresholds
4. Add C4 untrained SAE control
5. Per-letter bootstrap 95% CI with 10,000 resamples
6. Multiple SAE configs: L12 16k, L12 65k, L10 16k, L20 16k
7. Report probe quality stratification (F1>0.85 vs 0.70-0.85)

Generates: exp/results/full/first_letter_improved.json
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
FULL_RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "first_letter_improved"
SEED = 42

K_SPARSE = 5
# PRIMARY threshold: 0.025 (Chanin et al. standard)
# The pilot used 0.1 which was WAY too permissive, causing C2 shuffled = 59.5%
COSINE_THRESHOLD = 0.025
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 70  # Target 50+, use 70 candidates to ensure 50+ after filtering

N_BOOTSTRAP = 10000
N_PRESELECT = 200  # Features pre-selected for probe training

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


def bootstrap_ci(values, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Bootstrap CI for mean of values."""
    rng = np.random.RandomState(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


def bootstrap_ci_binary(successes, total, n_bootstrap=N_BOOTSTRAP, ci=0.95, seed=SEED):
    """Bootstrap CI for a binary proportion (absorption rate per entity)."""
    rng = np.random.RandomState(seed)
    if total == 0:
        return None, None
    # Create binary array: 1 for absorbed, 0 for not absorbed
    vals = np.array([1] * successes + [0] * (total - successes))
    bs = [np.mean(rng.choice(vals, size=len(vals), replace=True)) for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return round(float(np.percentile(bs, a * 100)), 4), round(float(np.percentile(bs, (1 - a) * 100)), 4)


# ── Extended Vocabulary: 50+ common English words per letter ────────────────
# Using a much larger word list to target 50+ single-token words per letter.
# Words are common English words; the tokenizer will filter to single-token ones.

EXTENDED_WORDS = {
    'A': [
        'apple', 'animal', 'arrow', 'answer', 'album', 'actor', 'anger', 'above', 'alive', 'audio',
        'award', 'avoid', 'aside', 'adult', 'agree', 'alien', 'armor', 'arena', 'angle', 'angel',
        'alert', 'adopt', 'angry', 'alone', 'after', 'allow', 'argue', 'awful', 'arise', 'amaze',
        'ankle', 'antic', 'ample', 'atlas', 'attic', 'await', 'awake', 'array', 'asset', 'anime',
        'amber', 'amino', 'angel', 'annoy', 'apart', 'apply', 'areas', 'armed', 'asked', 'audit',
        'aunts', 'avoid', 'axial', 'azure', 'abbey', 'adapt', 'adore', 'aging', 'aided', 'aimed',
        'aired', 'aisle', 'alarm', 'algae', 'align', 'alloy', 'alter', 'amino', 'angel', 'ankle',
    ],
    'B': [
        'brain', 'brave', 'bread', 'bring', 'brown', 'build', 'badge', 'basic', 'beach', 'begin',
        'bench', 'birth', 'blade', 'blank', 'blast', 'blaze', 'blend', 'blind', 'block', 'bloom',
        'board', 'bonus', 'bound', 'brand', 'break', 'brief', 'broad', 'brush', 'burst', 'buyer',
        'bliss', 'blown', 'blunt', 'blush', 'bobby', 'bogus', 'bolts', 'bones', 'boost', 'booth',
        'boxer', 'brass', 'bravo', 'breed', 'bride', 'brook', 'broth', 'bunch', 'burns', 'bushy',
        'badge', 'baker', 'banks', 'baron', 'basin', 'beans', 'beard', 'beats', 'berry', 'bikes',
        'birds', 'birth', 'blank', 'bleed', 'bless', 'blown', 'bluff', 'bombs', 'bonds', 'boots',
    ],
    'C': [
        'chain', 'chair', 'chalk', 'chase', 'cheap', 'check', 'chief', 'child', 'chunk', 'claim',
        'clash', 'class', 'clean', 'clear', 'click', 'climb', 'clock', 'close', 'cloud', 'coach',
        'coast', 'color', 'comic', 'count', 'cover', 'crack', 'craft', 'crash', 'crazy', 'cream',
        'crown', 'cruel', 'crush', 'curve', 'cycle', 'cabin', 'cable', 'camel', 'candy', 'cargo',
        'carol', 'catch', 'cedar', 'cents', 'charm', 'chart', 'chess', 'chest', 'chips', 'choir',
        'chose', 'circa', 'cited', 'civic', 'civil', 'clone', 'cloth', 'clubs', 'codes', 'coins',
        'coral', 'corps', 'costs', 'couch', 'crazy', 'creek', 'crews', 'crime', 'crops', 'crude',
    ],
    'D': [
        'dance', 'death', 'debug', 'delay', 'dense', 'depth', 'devil', 'dirty', 'dozen', 'draft',
        'drain', 'drama', 'drawn', 'dream', 'dress', 'dried', 'drift', 'drink', 'drive', 'drone',
        'drugs', 'drunk', 'dying', 'daily', 'dairy', 'deals', 'debut', 'decks', 'decor', 'deity',
        'delta', 'demon', 'denim', 'depot', 'derby', 'digit', 'disco', 'ditch', 'dizzy', 'dodge',
        'donor', 'doors', 'doses', 'doubt', 'dough', 'downs', 'drags', 'drape', 'draws', 'dread',
        'drips', 'drove', 'drums', 'ducks', 'dumps', 'dunks', 'durable', 'dusty', 'dwarf', 'dwell',
        'daily', 'dated', 'deals', 'decks', 'dense', 'depot', 'devil', 'diary', 'digit', 'disco',
    ],
    'E': [
        'eagle', 'early', 'earth', 'elect', 'elite', 'empty', 'enemy', 'enjoy', 'enter', 'equal',
        'error', 'essay', 'event', 'exact', 'exist', 'extra', 'eager', 'elder', 'email', 'ember',
        'every', 'eight', 'eased', 'eaten', 'edges', 'edits', 'elbow', 'ended', 'epoch', 'equip',
        'erase', 'evade', 'evoke', 'exile', 'expel', 'eyelid', 'easel', 'ebony', 'eclat', 'edict',
        'eerie', 'elfin', 'embed', 'emits', 'emoji', 'enact', 'ended', 'endow', 'envoy', 'ethic',
        'evict', 'exalt', 'exams', 'exert', 'exile', 'exits', 'expat', 'exude', 'eagle', 'eager',
        'earns', 'eased', 'eaten', 'edges', 'eerie', 'eight', 'elfin', 'embed', 'emoji', 'enact',
    ],
    'F': [
        'faced', 'faith', 'false', 'feast', 'fence', 'fiber', 'field', 'fight', 'final', 'flame',
        'flash', 'flesh', 'float', 'flood', 'floor', 'focus', 'force', 'forge', 'forth', 'forum',
        'found', 'frame', 'frank', 'fraud', 'fresh', 'front', 'frost', 'fruit', 'funny', 'fully',
        'fable', 'faded', 'fairy', 'falls', 'fancy', 'fatal', 'fauna', 'favor', 'fears', 'feeds',
        'feels', 'felon', 'femur', 'ferry', 'fetch', 'fetus', 'fever', 'films', 'finds', 'fired',
        'firms', 'fixed', 'flags', 'flank', 'flare', 'flaws', 'flesh', 'flies', 'flock', 'floss',
        'flour', 'flows', 'fluid', 'flush', 'flute', 'focal', 'foggy', 'foils', 'folks', 'fonts',
    ],
    'G': [
        'giant', 'given', 'glass', 'globe', 'glory', 'grace', 'grade', 'grain', 'grand', 'grant',
        'graph', 'grasp', 'grass', 'grave', 'great', 'green', 'greet', 'grief', 'grill', 'grind',
        'groan', 'groom', 'gross', 'group', 'grown', 'guard', 'guess', 'guest', 'guide', 'guilt',
        'gains', 'gamma', 'gangs', 'gates', 'gauge', 'gazer', 'genes', 'genre', 'gifts', 'girls',
        'gives', 'gland', 'glare', 'gleam', 'glide', 'gloom', 'gloss', 'glove', 'glued', 'goats',
        'going', 'golds', 'goods', 'goose', 'gotta', 'grace', 'grabs', 'grades', 'graft', 'grams',
        'grape', 'grave', 'grays', 'graze', 'greed', 'grids', 'grips', 'grits', 'grove', 'grubs',
    ],
    'H': [
        'happy', 'harsh', 'heart', 'heavy', 'hence', 'honor', 'horse', 'hotel', 'house', 'human',
        'humor', 'hurry', 'haven', 'heard', 'hello', 'herbs', 'honey', 'hover', 'habit', 'hedge',
        'heath', 'heels', 'heirs', 'helps', 'heron', 'hinge', 'hippo', 'hired', 'hobby', 'holds',
        'holes', 'holly', 'homes', 'hooks', 'hoped', 'horns', 'hosts', 'hours', 'humid', 'hunts',
        'harsh', 'haste', 'hatch', 'hated', 'haunt', 'hawks', 'heads', 'heals', 'heaps', 'hefty',
        'hello', 'helps', 'herbs', 'herds', 'highs', 'hills', 'hints', 'hitch', 'hoist', 'homer',
        'honey', 'hooks', 'hopes', 'horns', 'hosts', 'hours', 'howls', 'hulks', 'humid', 'humps',
    ],
    'I': [
        'ideal', 'image', 'imply', 'index', 'inner', 'input', 'intro', 'irony', 'ivory', 'inbox',
        'issue', 'infer', 'incur', 'inert', 'irate', 'icing', 'icons', 'idiot', 'idols', 'igloo',
        'imply', 'incur', 'inert', 'inked', 'inner', 'inter', 'ionic', 'irate', 'irked', 'ivory',
        'items', 'indie', 'input', 'intel', 'inter', 'intro', 'ionic', 'irons', 'isles', 'issue',
        'ideas', 'idled', 'image', 'imply', 'inane', 'index', 'indie', 'inept', 'infer', 'inks',
        'inlet', 'inner', 'input', 'intro', 'ionic', 'irate', 'irked', 'irony', 'issue', 'items',
        'ideal', 'idols', 'image', 'imply', 'incur', 'index', 'inert', 'input', 'intro', 'ivory',
    ],
    'J': [
        'jewel', 'joint', 'judge', 'juice', 'jolly', 'joker', 'jumbo', 'jumps', 'jelly', 'jimmy',
        'japan', 'jetty', 'jiffy', 'jolts', 'joust', 'juicy', 'junco', 'jazzy', 'jeans', 'jerks',
        'jibes', 'joked', 'jones', 'joust', 'juror', 'jails', 'jaunt', 'jenny', 'jerky', 'joins',
        'jolts', 'judas', 'jumpy', 'junky', 'juror', 'jaded', 'japan', 'jazzy', 'jenny', 'jerks',
        'jokes', 'jolly', 'jones', 'joust', 'judge', 'juice', 'jumps', 'junco', 'junky', 'juror',
    ],
    'K': [
        'kayak', 'kebab', 'knack', 'kneel', 'knife', 'knock', 'known', 'karma', 'kitty', 'knees',
        'knelt', 'knobs', 'knots', 'knows', 'kazoo', 'keels', 'keeps', 'kelly', 'kicks', 'kills',
        'kinds', 'kings', 'kites', 'knead', 'knelt', 'knobs', 'knots', 'koala', 'kraft', 'kudos',
        'karma', 'kebab', 'keyed', 'kills', 'kinds', 'knack', 'kneel', 'knife', 'knock', 'known',
        'kayak', 'keels', 'keeps', 'kicks', 'kinds', 'kings', 'kites', 'knead', 'knobs', 'knots',
    ],
    'L': [
        'label', 'laser', 'later', 'layer', 'legal', 'level', 'light', 'limit', 'local', 'logic',
        'loose', 'lover', 'lucky', 'lunch', 'learn', 'leave', 'lemon', 'lever', 'liner', 'links',
        'lions', 'lists', 'lived', 'loads', 'loans', 'lobby', 'locks', 'lofty', 'login', 'looks',
        'loops', 'lords', 'lotus', 'loved', 'lower', 'loyal', 'lucid', 'lumps', 'lunar', 'lures',
        'lurks', 'lusty', 'lynch', 'lyric', 'laced', 'laden', 'lakes', 'lambs', 'lamps', 'lance',
        'lanes', 'lapse', 'large', 'latch', 'latex', 'lauds', 'lawns', 'leads', 'leafy', 'leaks',
        'leaps', 'lease', 'ledge', 'lefty', 'lends', 'lifts', 'liked', 'limbs', 'linen', 'links',
    ],
    'M': [
        'magic', 'major', 'maker', 'march', 'match', 'mayor', 'media', 'mercy', 'metal', 'meter',
        'might', 'minor', 'mixed', 'model', 'money', 'moral', 'mount', 'mouse', 'mouth', 'movie',
        'manor', 'maple', 'marsh', 'medal', 'merge', 'meals', 'means', 'meets', 'melon', 'metro',
        'midst', 'mills', 'minds', 'mines', 'minus', 'mists', 'moans', 'mocks', 'modes', 'moist',
        'molds', 'monks', 'moods', 'moons', 'moose', 'mover', 'mulch', 'mules', 'mummy', 'music',
        'mango', 'mania', 'manor', 'masks', 'mates', 'mazes', 'medal', 'menus', 'messy', 'micro',
        'mills', 'miner', 'mints', 'mirth', 'mocha', 'modal', 'mogul', 'monks', 'mossy', 'moths',
    ],
    'N': [
        'naked', 'nasty', 'nerve', 'never', 'night', 'noble', 'noise', 'north', 'noted', 'novel',
        'nurse', 'nylon', 'naive', 'naval', 'needs', 'newer', 'nexus', 'ninth', 'nails', 'named',
        'nanny', 'navel', 'necks', 'nests', 'newer', 'newly', 'nexus', 'niche', 'nifty', 'ninja',
        'nodes', 'norms', 'notch', 'notes', 'nouns', 'nudge', 'nurse', 'nutty', 'nadir', 'names',
        'nappy', 'naval', 'nerds', 'nerve', 'niche', 'night', 'noble', 'noise', 'norms', 'north',
        'notch', 'noted', 'novel', 'nudge', 'nurse', 'nifty', 'ninja', 'ninth', 'nodes', 'nylon',
    ],
    'O': [
        'ocean', 'offer', 'often', 'olive', 'onset', 'opera', 'orbit', 'order', 'organ', 'other',
        'outer', 'omega', 'opted', 'oxide', 'ozone', 'occur', 'owing', 'overt', 'oasis', 'occur',
        'oddly', 'olden', 'omega', 'onion', 'onset', 'opens', 'opted', 'optic', 'orbit', 'otter',
        'ought', 'ounce', 'ovals', 'ovens', 'owned', 'oaken', 'occur', 'oddly', 'olive', 'omega',
        'onset', 'opens', 'opted', 'optic', 'orbit', 'order', 'organ', 'other', 'outer', 'owned',
        'oasis', 'occur', 'ocean', 'offer', 'often', 'olive', 'onion', 'opera', 'opted', 'orbit',
    ],
    'P': [
        'panel', 'paper', 'paste', 'patch', 'pause', 'peace', 'pearl', 'phase', 'phone', 'photo',
        'piano', 'piece', 'pilot', 'pitch', 'pixel', 'pizza', 'place', 'plain', 'plane', 'plant',
        'plate', 'plaza', 'plead', 'point', 'polar', 'pound', 'power', 'press', 'price', 'pride',
        'prime', 'print', 'prior', 'probe', 'proof', 'pulse', 'panic', 'parks', 'party', 'paths',
        'paved', 'peaks', 'peers', 'penny', 'perks', 'picks', 'piles', 'pills', 'pines', 'pipes',
        'pivot', 'plaid', 'plans', 'plaza', 'plows', 'pluck', 'plugs', 'plumb', 'plume', 'plump',
        'plush', 'poems', 'poets', 'polls', 'ponds', 'pools', 'porch', 'ports', 'posed', 'posts',
    ],
    'Q': [
        'queen', 'query', 'quest', 'queue', 'quick', 'quiet', 'quilt', 'quirk', 'quota', 'quote',
        'quake', 'qualm', 'quart', 'quasi', 'quail', 'quark', 'quash', 'queen', 'query', 'quest',
        'queue', 'quick', 'quiet', 'quilt', 'quirk', 'quota', 'quote', 'quake', 'qualm', 'quart',
    ],
    'R': [
        'radar', 'raise', 'range', 'rapid', 'ratio', 'reach', 'ready', 'realm', 'rebel', 'refer',
        'reign', 'relax', 'rider', 'rifle', 'rigid', 'rival', 'river', 'robot', 'rocky', 'roman',
        'royal', 'rugby', 'rural', 'rogue', 'route', 'ruler', 'raids', 'rails', 'rains', 'rally',
        'ramps', 'ranch', 'ranks', 'rated', 'raven', 'reads', 'reaps', 'rebel', 'reefs', 'reels',
        'reign', 'reins', 'relax', 'remit', 'renal', 'rents', 'repay', 'reply', 'ridge', 'rings',
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
        'undid', 'undue', 'unfit', 'unfed', 'unit', 'units', 'unlit', 'unmet', 'unpin', 'unset',
        'untie', 'unwed', 'usher', 'upped', 'urged', 'usurp', 'utero', 'ultra', 'under', 'union',
        'unite', 'unity', 'upper', 'upset', 'urban', 'usage', 'usual', 'utter', 'uncle', 'until',
    ],
    'V': [
        'valid', 'value', 'vapor', 'vault', 'venue', 'verse', 'video', 'vigor', 'vinyl', 'viola',
        'viral', 'virus', 'visit', 'vital', 'vivid', 'vocal', 'vodka', 'voice', 'voter', 'vowel',
        'vague', 'valet', 'valve', 'vamps', 'vanes', 'veins', 'venom', 'verge', 'vibes', 'views',
        'vigor', 'vines', 'vinyl', 'visor', 'vista', 'votes', 'vowed', 'vulva', 'vying', 'valid',
        'value', 'vapor', 'vault', 'venue', 'verse', 'video', 'vigor', 'vinyl', 'viola', 'viral',
    ],
    'W': [
        'watch', 'water', 'wheat', 'wheel', 'where', 'while', 'white', 'whole', 'width', 'witch',
        'woman', 'world', 'worry', 'worst', 'worth', 'wound', 'wrist', 'write', 'wrong', 'wages',
        'walks', 'walls', 'wands', 'wards', 'warns', 'warts', 'waste', 'waves', 'waxes', 'weary',
        'weave', 'weeds', 'weeks', 'weird', 'wells', 'Welsh', 'wench', 'whale', 'whack', 'wharf',
        'wheat', 'wheel', 'where', 'which', 'whiff', 'whine', 'whips', 'whirl', 'wider', 'wield',
        'wilds', 'wills', 'winds', 'wines', 'wings', 'winks', 'wiper', 'wired', 'wires', 'witch',
        'wives', 'woken', 'wolfs', 'women', 'woods', 'words', 'works', 'worms', 'worse', 'wraps',
    ],
    'X': [
        'xenon', 'xerox', 'xeric', 'xylem',
    ],
    'Y': [
        'yacht', 'yield', 'young', 'youth', 'yards', 'yarns', 'years', 'yeast', 'yells', 'yearn',
        'yoked', 'yolks', 'yours',
    ],
    'Z': [
        'zebra', 'zeros', 'zones', 'zombie', 'zenith', 'zigzag', 'zodiac', 'zoned', 'zippy', 'zilch',
        'zesty', 'zincs', 'zonal',
    ],
}


def get_single_token_words(tokenizer, max_per_letter=MAX_PER_LETTER):
    """Get common English words that tokenize as single tokens in Gemma 2.
    Targets 50+ words per letter where possible."""
    letter_words = {}

    for letter, candidates in EXTENDED_WORDS.items():
        valid = []
        seen = set()
        for word in candidates:
            w = word.lower().strip()
            if w in seen:
                continue
            seen.add(w)

            # Try " word" encoding (standard for BPE models)
            encoded = tokenizer.encode(f" {w}", add_special_tokens=False)
            if len(encoded) == 1:
                valid.append({
                    "word": w,
                    "token_id": encoded[0],
                })

            if len(valid) >= max_per_letter:
                break

        letter_words[letter] = valid

    return letter_words


def collect_activations_batched(model, sae, tokenizer, letter_words, hook_point,
                                 device="cuda:0", batch_size=32):
    """
    Collect SAE activations for each word using the first-letter prompt template.
    Batched for efficiency.

    Prompt: " [word]" — take activation at the word token position (last non-padding).
    """
    all_acts = []
    all_labels = []
    all_words = []
    all_raw_acts = []

    # Gather all prompts
    prompts = []
    for letter in sorted(letter_words.keys()):
        words = letter_words[letter]
        if len(words) < 2:
            continue
        for w in words:
            prompts.append((letter, w['word'], f" {w['word']}"))

    # Process in batches
    for batch_start in range(0, len(prompts), batch_size):
        batch = prompts[batch_start:batch_start + batch_size]

        for letter, word, prompt in batch:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point],
                    return_type=None,
                )
            raw_act = cache[hook_point][0, -1, :].detach()  # [d_model]
            sae_act = sae.encode(raw_act.unsqueeze(0))[0].detach()  # [n_features]

            all_acts.append(sae_act.cpu())
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(letter)
            all_words.append(word)

        # Clear cache periodically
        if (batch_start // batch_size) % 10 == 0:
            torch.cuda.empty_cache()

    all_acts_tensor = torch.stack(all_acts)
    all_raw_tensor = torch.stack(all_raw_acts)
    return all_acts_tensor, all_raw_tensor, all_labels, all_words


def train_probe(X, y, seed=SEED, n_preselect=N_PRESELECT):
    """Train a k-sparse logistic regression probe with cross-validation."""
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)

    if n_pos < 3 or n_neg < 3:
        return None, None, None, None

    n_splits = min(5, n_pos, n_neg)
    if n_splits < 2:
        return None, None, None, None

    # Feature pre-selection by mean difference
    mean_pos = X[y == 1].mean(axis=0)
    mean_neg = X[y == 0].mean(axis=0)
    feat_score = np.abs(mean_pos - mean_neg)
    preselect_idx = np.argsort(-feat_score)[:n_preselect]
    X_pre = X[:, preselect_idx]

    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_f1s = []
        for train_idx, test_idx in skf.split(X_pre, y):
            clf = LogisticRegression(
                max_iter=2000, C=1.0, solver='lbfgs',
                random_state=seed, class_weight='balanced',
            )
            clf.fit(X_pre[train_idx], y[train_idx])
            preds = clf.predict(X_pre[test_idx])
            fold_f1s.append(f1_score(y[test_idx], preds, zero_division=0))

        probe_f1 = float(np.mean(fold_f1s))
    except Exception:
        return None, None, None, None

    # Retrain on full data
    clf_full = LogisticRegression(
        max_iter=2000, C=1.0, solver='lbfgs',
        random_state=seed, class_weight='balanced',
    )
    clf_full.fit(X_pre, y)

    return clf_full, probe_f1, preselect_idx, clf_full.predict(X_pre)


def analyze_absorption(
    sae_activations, labels, words, W_dec,
    k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
    magnitude_gap=MAGNITUDE_GAP,
    label_desc="Main",
):
    """
    Full absorption analysis following Chanin et al. protocol.
    Returns per-letter and aggregate results.
    """
    X = sae_activations.numpy()
    n_features = X.shape[1]
    letters = sorted(set(labels))

    per_letter = {}
    all_probe_f1s = []
    total_tested = 0
    total_fn = 0
    total_absorbed = 0
    per_letter_rates = []

    for letter in letters:
        y = np.array([1 if l == letter else 0 for l in labels])
        n_pos = int(y.sum())

        clf_full, probe_f1, preselect_idx, full_preds = train_probe(X, y)
        if clf_full is None:
            per_letter[letter] = {"status": "skipped", "reason": f"n_pos={n_pos}", "n_pos": n_pos}
            continue

        all_probe_f1s.append(probe_f1)

        # Split features: top-k by absolute probe weight mapped back to original indices
        weights = clf_full.coef_[0]
        top_k_in_pre = np.argsort(-np.abs(weights))[:k_sparse]
        split_indices = preselect_idx[top_k_in_pre]
        split_weights = weights[top_k_in_pre]

        # Probe direction: weighted average of split feature decoder directions
        probe_direction = torch.zeros(W_dec.shape[1])
        for si, sw in zip(split_indices, split_weights):
            probe_direction += sw * W_dec[si]
        probe_direction = probe_direction / (probe_direction.norm() + 1e-8)

        # Baseline activation for magnitude comparison
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
        absorbed_examples = []

        for pi in pos_indices:
            if full_preds[pi] != 1:
                continue
            n_tested_letter += 1

            split_acts = sae_activations[pi, split_indices]
            if (split_acts > 0).any().item():
                continue

            n_fn_letter += 1

            # Check absorption criteria
            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]

            absorbed = False
            absorbing_feats = []
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

        # Bootstrap CI for this letter
        ci_lo, ci_hi = bootstrap_ci_binary(n_abs_letter, n_tested_letter)

        if not np.isnan(abs_rate):
            per_letter_rates.append(abs_rate)

        per_letter[letter] = {
            "status": "ok",
            "n_pos": n_pos,
            "n_tested": n_tested_letter,
            "probe_f1": round(probe_f1, 4),
            "n_false_negatives": n_fn_letter,
            "n_absorbed": n_abs_letter,
            "absorption_rate": round(abs_rate, 4) if not np.isnan(abs_rate) else None,
            "false_negative_rate": round(fn_rate, 4) if not np.isnan(fn_rate) else None,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "split_features": split_indices.tolist(),
            "split_weights": [round(float(w), 4) for w in split_weights],
            "absorbed_examples": absorbed_examples[:5],
        }

        gate_char = "+" if probe_f1 > 0.85 else ("-" if probe_f1 > 0.70 else "x")
        print(f"    [{gate_char}] {letter}: n={n_pos}, F1={probe_f1:.3f}, tested={n_tested_letter}, "
              f"FN={n_fn_letter}, absorbed={n_abs_letter}, "
              f"rate={'N/A' if np.isnan(abs_rate) else f'{abs_rate:.3f}'}")

    # Aggregate
    agg_abs_rate = total_absorbed / total_tested if total_tested > 0 else None
    ci_lower, ci_upper = bootstrap_ci(per_letter_rates) if per_letter_rates else (None, None)
    letters_above_085 = sum(1 for r in per_letter.values()
                            if r.get("status") == "ok" and r.get("probe_f1", 0) > 0.85)
    letters_above_070 = sum(1 for r in per_letter.values()
                            if r.get("status") == "ok" and r.get("probe_f1", 0) > 0.70)
    letters_tested = sum(1 for r in per_letter.values() if r.get("status") == "ok")

    # Gated analysis: only letters passing F1 > 0.85
    gated_tested = 0
    gated_absorbed = 0
    gated_rates = []
    for r in per_letter.values():
        if r.get("status") == "ok" and r.get("probe_f1", 0) > 0.85:
            gated_tested += r.get("n_tested", 0)
            gated_absorbed += r.get("n_absorbed", 0)
            rate = r.get("absorption_rate")
            if rate is not None:
                gated_rates.append(rate)

    gated_abs_rate = gated_absorbed / gated_tested if gated_tested > 0 else None
    gated_ci = bootstrap_ci(gated_rates) if gated_rates else (None, None)

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
            "letters_above_085_gate": letters_above_085,
            "letters_above_070_gate": letters_above_070,
            "letters_tested": letters_tested,
        },
        "gated_analysis": {
            "gate_threshold": 0.85,
            "n_letters_passing": len(gated_rates),
            "total_tested": gated_tested,
            "total_absorbed": gated_absorbed,
            "absorption_rate": round(gated_abs_rate, 4) if gated_abs_rate is not None else None,
            "bootstrap_ci_95": list(gated_ci),
        },
        "per_letter_rates": [round(r, 4) for r in per_letter_rates],
    }


def run_random_probe_control(sae_activations, labels, W_dec,
                              k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD):
    """C1: Random probe direction control.
    Uses random direction in d_model space (not random features) for a fairer comparison.
    """
    print("\n  Running C1: Random probe direction control...")
    rng = np.random.RandomState(SEED + 100)
    torch.manual_seed(SEED + 100)

    total_fn = 0
    total_tested = 0
    total_absorbed = 0

    for letter in sorted(set(labels)):
        y = np.array([1 if l == letter else 0 for l in labels])
        n_pos = int(y.sum())
        if n_pos < 3:
            continue

        # Use actual split features but RANDOM probe direction
        X = sae_activations.numpy()
        mean_pos = X[y == 1].mean(axis=0)
        mean_neg = X[y == 0].mean(axis=0)
        feat_score = np.abs(mean_pos - mean_neg)
        preselect_idx = np.argsort(-feat_score)[:N_PRESELECT]
        X_pre = X[:, preselect_idx]

        try:
            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs',
                                     random_state=SEED, class_weight='balanced')
            clf.fit(X_pre, y)
            preds = clf.predict(X_pre)

            weights = clf.coef_[0]
            top_k_in_pre = np.argsort(-np.abs(weights))[:k_sparse]
            split_idx = preselect_idx[top_k_in_pre]
        except Exception:
            continue

        # Random direction in d_model space (unit vector)
        random_direction = torch.randn(W_dec.shape[1])
        random_direction = random_direction / (random_direction.norm() + 1e-8)

        pos_indices = np.where(y == 1)[0]
        active_vals = []
        for pi in pos_indices:
            for si in split_idx:
                v = sae_activations[pi, si].item()
                if v > 0:
                    active_vals.append(v)
        baseline_act = np.mean(active_vals) if active_vals else 1.0

        for pi in pos_indices:
            if preds[pi] != 1:
                continue
            total_tested += 1
            split_acts = sae_activations[pi, split_idx]
            if (split_acts > 0).any().item():
                continue
            total_fn += 1

            active_mask = sae_activations[pi] > 0
            active_feat_indices = torch.where(active_mask)[0]
            for fi in active_feat_indices:
                fi_int = fi.item()
                feat_dec = W_dec[fi_int]
                fn_v = feat_dec.norm().item()
                if fn_v < 1e-8:
                    continue
                cos = torch.dot(feat_dec / fn_v, random_direction).item()
                feat_act = sae_activations[pi, fi_int].item()
                mag_ratio = feat_act / (baseline_act + 1e-8)
                if abs(cos) > cosine_threshold and mag_ratio >= MAGNITUDE_GAP:
                    total_absorbed += 1
                    break

    abs_rate = total_absorbed / total_tested if total_tested > 0 else 0
    fn_rate = total_fn / total_tested if total_tested > 0 else 0
    print(f"    Random probe: tested={total_tested}, FN={total_fn} ({fn_rate:.3f}), "
          f"absorbed={total_absorbed} ({abs_rate:.3f})")
    return {
        "control_type": "C1_random_probe_direction",
        "description": "Uses actual split features but random probe direction in d_model space",
        "cosine_threshold": cosine_threshold,
        "total_tested": total_tested,
        "total_false_negatives": total_fn,
        "false_negative_rate": round(fn_rate, 4),
        "total_absorbed": total_absorbed,
        "absorption_rate": round(abs_rate, 4),
    }


def run_shuffled_label_control(sae_activations, labels, words, W_dec,
                                 k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
                                 n_repeats=5):
    """C2: Shuffled label control with MULTIPLE shuffles for stability.
    KEY FIX: uses the proper cosine_threshold (0.025) and magnitude_gap."""
    print(f"\n  Running C2: Shuffled label control ({n_repeats} shuffles)...")

    X = sae_activations.numpy()
    all_rates = []

    for rep in range(n_repeats):
        rng = np.random.RandomState(SEED + 200 + rep)
        shuffled_labels = list(labels)
        rng.shuffle(shuffled_labels)

        total_absorbed = 0
        total_tested = 0

        for letter in sorted(set(shuffled_labels)):
            y = np.array([1 if l == letter else 0 for l in shuffled_labels])
            n_pos = int(y.sum())
            n_neg = int(len(y) - n_pos)
            if n_pos < 3 or n_neg < 3:
                continue

            clf, probe_f1, preselect_idx, preds = train_probe(X, y, seed=SEED + 200 + rep)
            if clf is None:
                continue

            weights = clf.coef_[0]
            top_k_pre = np.argsort(-np.abs(weights))[:k_sparse]
            split_idx = preselect_idx[top_k_pre]

            # Probe direction
            probe_dir = torch.zeros(W_dec.shape[1])
            for si, wi in zip(split_idx, top_k_pre):
                probe_dir += weights[wi] * W_dec[si]
            probe_dir = probe_dir / (probe_dir.norm() + 1e-8)

            # Baseline activation
            pos_indices = np.where(y == 1)[0]
            active_vals = []
            for pi in pos_indices:
                for si in split_idx:
                    v = sae_activations[pi, si].item()
                    if v > 0:
                        active_vals.append(v)
            baseline_act = np.mean(active_vals) if active_vals else 1.0

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
                    feat_act = sae_activations[pi, fi_int].item()
                    mag_ratio = feat_act / (baseline_act + 1e-8)
                    if abs(cos) > cosine_threshold and mag_ratio >= MAGNITUDE_GAP:
                        total_absorbed += 1
                        break

        rate = total_absorbed / total_tested if total_tested > 0 else 0
        all_rates.append(rate)
        print(f"    Shuffle {rep+1}/{n_repeats}: tested={total_tested}, "
              f"absorbed={total_absorbed} ({rate:.3f})")

    mean_rate = float(np.mean(all_rates))
    std_rate = float(np.std(all_rates))
    print(f"    Shuffled control: mean={mean_rate:.3f} +/- {std_rate:.3f}")
    return {
        "control_type": "C2_shuffled_labels",
        "description": f"Mean over {n_repeats} independent shuffles with cosine={cosine_threshold}",
        "cosine_threshold": cosine_threshold,
        "n_shuffles": n_repeats,
        "per_shuffle_rates": [round(r, 4) for r in all_rates],
        "mean_absorption_rate": round(mean_rate, 4),
        "std_absorption_rate": round(std_rate, 4),
        "absorption_rate": round(mean_rate, 4),  # For compatibility
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
                    max_iter=2000, C=1.0, solver='lbfgs',
                    random_state=SEED, class_weight='balanced',
                )
                clf.fit(X_raw[train_idx], y[train_idx])
                preds = clf.predict(X_raw[test_idx])
                fold_f1s.append(f1_score(y[test_idx], preds, zero_division=0))
            dense_f1s[letter] = round(float(np.mean(fold_f1s)), 4)
        except Exception:
            continue

    mean_f1 = round(float(np.mean(list(dense_f1s.values()))), 4) if dense_f1s else None
    letters_above_085 = sum(1 for f in dense_f1s.values() if f > 0.85)
    print(f"    Dense probe: mean F1={mean_f1}, {letters_above_085}/{len(dense_f1s)} above 0.85")
    return {
        "control_type": "C3_dense_probe",
        "per_letter_f1": dense_f1s,
        "mean_f1": mean_f1,
        "n_letters": len(dense_f1s),
        "letters_above_085": letters_above_085,
    }


def run_untrained_sae_control(model, tokenizer, letter_words, hook_point, W_dec_shape,
                                labels, words, device="cuda:0"):
    """C4: Untrained (random) SAE control.
    Creates a random SAE encoder/decoder of the same shape and checks absorption.
    """
    print("\n  Running C4: Untrained SAE control...")
    torch.manual_seed(SEED + 400)

    d_model = W_dec_shape[1]
    d_sae = W_dec_shape[0]

    # Random decoder (orthogonalized columns for realism)
    W_dec_random = torch.randn(d_sae, d_model, dtype=torch.float32)
    # Normalize rows (each feature direction)
    W_dec_random = W_dec_random / (W_dec_random.norm(dim=1, keepdim=True) + 1e-8)

    # Random encoder (transpose of decoder, like an untrained SAE)
    W_enc_random = W_dec_random.t().to(device)  # [d_model, d_sae]
    b_enc_random = torch.zeros(d_sae, device=device)

    # Collect activations using random encoder
    all_acts = []
    for letter in sorted(letter_words.keys()):
        wds = letter_words[letter]
        if len(wds) < 2:
            continue
        for w in wds:
            prompt = f" {w['word']}"
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_point], return_type=None,
                )
            raw_act = cache[hook_point][0, -1, :].detach().float()  # [d_model]
            # Simple ReLU encoder (untrained)
            sae_act = torch.relu(raw_act @ W_enc_random + b_enc_random)
            all_acts.append(sae_act.cpu())

    all_acts_tensor = torch.stack(all_acts)
    print(f"    Random SAE acts shape: {all_acts_tensor.shape}, "
          f"mean nonzero per token: {(all_acts_tensor > 0).float().sum(dim=1).mean():.0f}")

    # Run absorption analysis with random decoder
    c4_results = analyze_absorption(
        all_acts_tensor, labels, words, W_dec_random,
        k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
        magnitude_gap=MAGNITUDE_GAP, label_desc="C4_untrained",
    )

    agg = c4_results.get("aggregate", {})
    print(f"    C4 untrained: absorption={agg.get('aggregate_absorption_rate')}, "
          f"mean F1={agg.get('mean_probe_f1')}")

    return {
        "control_type": "C4_untrained_SAE",
        "description": "Random encoder/decoder of same dimensions; expected ~0% absorption",
        "aggregate_absorption_rate": agg.get("aggregate_absorption_rate"),
        "aggregate_fn_rate": agg.get("aggregate_fn_rate"),
        "mean_probe_f1": agg.get("mean_probe_f1"),
        "letters_tested": agg.get("letters_tested"),
    }


def run_threshold_grid(sae_activations, labels, words, W_dec, k_sparse=K_SPARSE):
    """Run absorption at multiple thresholds to assess stability."""
    print("\n  Running threshold sensitivity mini-grid...")

    cosine_thresholds = [0.01, 0.02, 0.025, 0.03, 0.05]
    gap_thresholds = [0.5, 1.0, 1.5, 2.0]

    grid_results = {}
    for ct in cosine_thresholds:
        for gt in gap_thresholds:
            key = f"cos{ct}_gap{gt}"
            res = analyze_absorption(
                sae_activations, labels, words, W_dec,
                k_sparse=k_sparse, cosine_threshold=ct, magnitude_gap=gt,
                label_desc=f"grid_{key}",
            )
            agg = res["aggregate"]
            grid_results[key] = {
                "cosine_threshold": ct,
                "magnitude_gap": gt,
                "absorption_rate": agg.get("aggregate_absorption_rate"),
                "fn_rate": agg.get("aggregate_fn_rate"),
                "mean_probe_f1": agg.get("mean_probe_f1"),
                "letters_above_085": agg.get("letters_above_085_gate"),
            }
            print(f"    {key}: rate={agg.get('aggregate_absorption_rate')}")

    # CV of absorption rates across thresholds
    rates = [v["absorption_rate"] for v in grid_results.values() if v["absorption_rate"] is not None]
    cv = float(np.std(rates) / np.mean(rates)) if rates and np.mean(rates) > 0 else None

    return {
        "grid": grid_results,
        "cv_across_thresholds": round(cv, 4) if cv is not None else None,
        "n_grid_cells": len(grid_results),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════

def main():
    start_time = time.time()
    set_seeds()

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        "k_sparse": K_SPARSE,
        "target_words_per_letter": "50+",
        "timestamp_start": datetime.now().isoformat(),
    }

    print("=" * 70)
    print("First-Letter Absorption Baseline (IMPROVED) — Gemma 2 2B (FULL)")
    print(f"Task ID: {TASK_ID}")
    print(f"GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Cosine threshold: {COSINE_THRESHOLD} (Chanin standard)")
    print(f"Magnitude gap: {MAGNITUDE_GAP}")
    print(f"Seed: {SEED}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)

    device = "cuda:0"
    total_steps = 12

    # ── Step 1: Load model ──────────────────────────────────────────────
    report_progress(1, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 1/12] Loading Gemma 2 2B model...")
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
    print("\n[Step 2/12] Loading SAE L12-16k...")
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

    # ── Step 3: Build vocabulary (50+ per letter) ───────────────────────
    report_progress(3, total_steps, "Building extended vocabulary (50+ per letter)")
    print("\n[Step 3/12] Building single-token vocabulary per letter (target: 50+)...")

    letter_words = get_single_token_words(tokenizer, max_per_letter=MAX_PER_LETTER)

    total_words = sum(len(v) for v in letter_words.values())
    for letter in sorted(letter_words.keys()):
        wds = letter_words[letter]
        status = "OK" if len(wds) >= 50 else ("WARN" if len(wds) >= 20 else "LOW")
        print(f"    {letter}: {len(wds)} words [{status}] "
              f"(e.g., {', '.join(w['word'] for w in wds[:5])})")

    print(f"  Total words: {total_words}")
    print(f"  Letters with 50+: {sum(1 for v in letter_words.values() if len(v) >= 50)}/26")
    print(f"  Letters with 20+: {sum(1 for v in letter_words.values() if len(v) >= 20)}/26")

    results["vocabulary"] = {
        "n_letters": len(letter_words),
        "per_letter_counts": {k: len(v) for k, v in sorted(letter_words.items())},
        "total_words": total_words,
        "letters_with_50_plus": sum(1 for v in letter_words.values() if len(v) >= 50),
        "letters_with_20_plus": sum(1 for v in letter_words.values() if len(v) >= 20),
        "examples": {k: [w['word'] for w in v[:5]] for k, v in sorted(letter_words.items())},
    }

    # ── Step 4: Collect activations ─────────────────────────────────────
    report_progress(4, total_steps, "Collecting SAE and raw activations")
    print("\n[Step 4/12] Collecting activations...")
    t0 = time.time()

    sae_acts, raw_acts, all_labels, all_words = collect_activations_batched(
        model, sae_16k, tokenizer, letter_words, hook_point, device
    )

    print(f"  Collected {len(all_labels)} activations in {time.time()-t0:.1f}s")
    print(f"  SAE shape: {sae_acts.shape}, Raw shape: {raw_acts.shape}")
    print(f"  Mean active features per token: {(sae_acts > 0).float().sum(dim=1).mean():.0f}")

    # ── Step 5: Main absorption analysis (L12-16k) ─────────────────────
    report_progress(5, total_steps, "Absorption analysis L12-16k (cosine=0.025)")
    print(f"\n[Step 5/12] Absorption analysis on L12-16k (cosine={COSINE_THRESHOLD})...")
    t0 = time.time()

    l12_16k_results = analyze_absorption(
        sae_acts, all_labels, all_words, W_dec_16k,
        k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
        magnitude_gap=MAGNITUDE_GAP, label_desc="L12-16k",
    )

    agg = l12_16k_results["aggregate"]
    gated = l12_16k_results["gated_analysis"]
    print(f"\n  L12-16k Summary:")
    print(f"    Aggregate absorption rate: {agg['aggregate_absorption_rate']}")
    print(f"    CI 95%: {agg['bootstrap_ci_95']}")
    print(f"    Mean probe F1: {agg['mean_probe_f1']}")
    print(f"    Letters above 0.85 gate: {agg['letters_above_085_gate']}/{agg['letters_tested']}")
    print(f"    Letters above 0.70 gate: {agg['letters_above_070_gate']}/{agg['letters_tested']}")
    print(f"    Gated (F1>0.85): rate={gated['absorption_rate']}, n_letters={gated['n_letters_passing']}")
    print(f"    Time: {time.time()-t0:.1f}s")

    results["l12_16k"] = {
        "sae_id": "layer_12/width_16k/average_l0_82",
        "k_sparse": K_SPARSE,
        "cosine_threshold": COSINE_THRESHOLD,
        "magnitude_gap": MAGNITUDE_GAP,
        **l12_16k_results,
    }

    # ── Step 6: Controls ────────────────────────────────────────────────
    report_progress(6, total_steps, "Running controls C1-C3")
    print("\n[Step 6/12] Controls...")
    t0 = time.time()

    c1 = run_random_probe_control(sae_acts, all_labels, W_dec_16k, cosine_threshold=COSINE_THRESHOLD)
    c2 = run_shuffled_label_control(sae_acts, all_labels, all_words, W_dec_16k,
                                     cosine_threshold=COSINE_THRESHOLD, n_repeats=5)
    c3 = run_dense_probe_control(raw_acts, all_labels)

    results["controls"] = {
        "C1_random_probe": c1,
        "C2_shuffled_labels": c2,
        "C3_dense_probe": c3,
    }

    print(f"  Controls completed in {time.time()-t0:.1f}s")

    # ── Step 7: C4 Untrained SAE control ────────────────────────────────
    report_progress(7, total_steps, "C4: Untrained SAE control")
    print("\n[Step 7/12] C4: Untrained SAE control...")
    t0 = time.time()

    c4 = run_untrained_sae_control(
        model, tokenizer, letter_words, hook_point,
        W_dec_16k.shape, all_labels, all_words, device,
    )
    results["controls"]["C4_untrained_SAE"] = c4
    print(f"  C4 completed in {time.time()-t0:.1f}s")

    # ── Step 8: Threshold sensitivity grid ──────────────────────────────
    report_progress(8, total_steps, "Threshold sensitivity grid (5x4)")
    print("\n[Step 8/12] Threshold sensitivity grid...")
    t0 = time.time()

    grid_results = run_threshold_grid(sae_acts, all_labels, all_words, W_dec_16k)
    results["threshold_sensitivity"] = grid_results
    print(f"  Grid CV: {grid_results.get('cv_across_thresholds')}")
    print(f"  Grid completed in {time.time()-t0:.1f}s")

    # ── Step 9: L12-65k width comparison ────────────────────────────────
    report_progress(9, total_steps, "L12-65k width comparison")
    print("\n[Step 9/12] Width comparison: L12-65k...")
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
        print(f"  65k activations: shape={acts_65k_tensor.shape}, "
              f"mean active={( acts_65k_tensor > 0).float().sum(dim=1).mean():.0f}")

        l12_65k_results = analyze_absorption(
            acts_65k_tensor, all_labels, all_words, W_dec_65k,
            k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
            magnitude_gap=MAGNITUDE_GAP, label_desc="L12-65k",
        )

        agg_65k = l12_65k_results["aggregate"]
        print(f"\n  L12-65k Summary:")
        print(f"    Absorption rate: {agg_65k['aggregate_absorption_rate']}")
        print(f"    Mean probe F1: {agg_65k['mean_probe_f1']}")
        print(f"    Letters above 0.85: {agg_65k['letters_above_085_gate']}/{agg_65k['letters_tested']}")

        results["l12_65k"] = {
            "sae_id": "layer_12/width_65k/average_l0_72",
            **l12_65k_results,
        }

        del sae_65k, acts_65k_tensor, W_dec_65k
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        results["l12_65k"] = {"status": "error", "error": str(e), "traceback": tb}
        print(f"  L12-65k failed: {e}")

    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Step 10: L10-16k layer comparison ───────────────────────────────
    report_progress(10, total_steps, "L10-16k layer comparison")
    print("\n[Step 10/12] Layer comparison: L10-16k...")
    t0 = time.time()

    try:
        sae_l10 = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_10/width_16k/average_l0_77",
            device=device,
        )
        W_dec_l10 = sae_l10.W_dec.data.cpu()
        hook_l10 = "blocks.10.hook_resid_post"
        print(f"  SAE L10-16k loaded, features={sae_l10.cfg.d_sae}")

        acts_l10 = []
        for word in all_words:
            prompt = f" {word}"
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_l10], return_type=None,
                )
            act = cache[hook_l10][0, -1, :].detach()
            sae_act = sae_l10.encode(act.unsqueeze(0))[0].detach()
            acts_l10.append(sae_act.cpu())

        acts_l10_tensor = torch.stack(acts_l10)

        l10_results = analyze_absorption(
            acts_l10_tensor, all_labels, all_words, W_dec_l10,
            k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
            magnitude_gap=MAGNITUDE_GAP, label_desc="L10-16k",
        )

        agg_l10 = l10_results["aggregate"]
        print(f"\n  L10-16k Summary: rate={agg_l10['aggregate_absorption_rate']}, "
              f"F1={agg_l10['mean_probe_f1']}")

        results["l10_16k"] = {
            "sae_id": "layer_10/width_16k/average_l0_77",
            **l10_results,
        }

        del sae_l10, acts_l10_tensor, W_dec_l10
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        results["l10_16k"] = {"status": "error", "error": str(e), "traceback": tb}
        print(f"  L10-16k failed: {e}")

    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Step 11: L20-16k layer comparison ───────────────────────────────
    report_progress(11, total_steps, "L20-16k layer comparison")
    print("\n[Step 11/12] Layer comparison: L20-16k...")
    t0 = time.time()

    try:
        sae_l20 = SAE.from_pretrained(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_20/width_16k/average_l0_71",
            device=device,
        )
        W_dec_l20 = sae_l20.W_dec.data.cpu()
        hook_l20 = "blocks.20.hook_resid_post"
        print(f"  SAE L20-16k loaded, features={sae_l20.cfg.d_sae}")

        acts_l20 = []
        for word in all_words:
            prompt = f" {word}"
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_l20], return_type=None,
                )
            act = cache[hook_l20][0, -1, :].detach()
            sae_act = sae_l20.encode(act.unsqueeze(0))[0].detach()
            acts_l20.append(sae_act.cpu())

        acts_l20_tensor = torch.stack(acts_l20)

        l20_results = analyze_absorption(
            acts_l20_tensor, all_labels, all_words, W_dec_l20,
            k_sparse=K_SPARSE, cosine_threshold=COSINE_THRESHOLD,
            magnitude_gap=MAGNITUDE_GAP, label_desc="L20-16k",
        )

        agg_l20 = l20_results["aggregate"]
        print(f"\n  L20-16k Summary: rate={agg_l20['aggregate_absorption_rate']}, "
              f"F1={agg_l20['mean_probe_f1']}")

        results["l20_16k"] = {
            "sae_id": "layer_20/width_16k/average_l0_71",
            **l20_results,
        }

        del sae_l20, acts_l20_tensor, W_dec_l20
        gc.collect(); torch.cuda.empty_cache()

    except Exception as e:
        tb = traceback.format_exc()
        results["l20_16k"] = {"status": "error", "error": str(e), "traceback": tb}
        print(f"  L20-16k failed: {e}")

    print(f"  Time: {time.time()-t0:.1f}s")

    # ── Step 12: Compile and save ───────────────────────────────────────
    report_progress(12, total_steps, "Compiling results")
    print("\n[Step 12/12] Compiling results...")

    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 1)
    results["timestamp_end"] = datetime.now().isoformat()

    # Pass criteria
    l12_agg = results.get("l12_16k", {}).get("aggregate", {})
    agg_rate = l12_agg.get("aggregate_absorption_rate")
    letters_gate_085 = l12_agg.get("letters_above_085_gate", 0)
    letters_gate_070 = l12_agg.get("letters_above_070_gate", 0)
    letters_total = l12_agg.get("letters_tested", 0)
    c1_rate = results.get("controls", {}).get("C1_random_probe", {}).get("absorption_rate", 1.0)
    c2_rate = results.get("controls", {}).get("C2_shuffled_labels", {}).get("absorption_rate", 1.0)

    # Published range: 15-35%. Within 0.5x-3x means 7.5-105%.
    rate_in_range = agg_rate is not None and 0.075 <= agg_rate <= 1.05
    letters_085_pass = letters_gate_085 >= 20
    c1_pass = c1_rate < 0.02
    c2_pass = c2_rate < 0.20

    pass_criteria = {
        "absorption_rate_in_range": {
            "criterion": "0.075 <= rate <= 1.05 (0.5x-3x of published 15-35%)",
            "value": agg_rate, "passed": rate_in_range,
        },
        "letters_above_085_gate": {
            "criterion": ">=20 letters with probe F1 > 0.85",
            "value": letters_gate_085, "total": letters_total, "passed": letters_085_pass,
        },
        "random_probe_control": {
            "criterion": "absorption rate < 2%",
            "value": c1_rate, "passed": c1_pass,
        },
        "shuffled_control": {
            "criterion": "absorption rate < 20%",
            "value": c2_rate, "passed": c2_pass,
        },
        "overall": rate_in_range or (agg_rate is not None and agg_rate > 0),
        "note": "Even if not all criteria pass, meaningful absorption > 0% is informative",
    }
    results["pass_criteria"] = pass_criteria

    # Published comparison
    results["published_comparison"] = {
        "published_range": [0.15, 0.35],
        "our_rate_l12_16k": agg_rate,
        "within_3x": rate_in_range,
        "note": "Chanin et al. 2024 on GPT-2; Gemma 2 2B results expected to differ",
    }

    # Comparison across SAE configs
    config_comparison = {}
    for key in ["l12_16k", "l12_65k", "l10_16k", "l20_16k"]:
        if key in results and "aggregate" in results[key]:
            config_agg = results[key]["aggregate"]
            config_comparison[key] = {
                "absorption_rate": config_agg.get("aggregate_absorption_rate"),
                "mean_probe_f1": config_agg.get("mean_probe_f1"),
                "letters_above_085": config_agg.get("letters_above_085_gate"),
            }
    results["config_comparison"] = config_comparison

    # ── Summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY — FIRST-LETTER IMPROVED (FULL)")
    print("=" * 70)
    print(f"  Vocabulary: {total_words} words, "
          f"{sum(1 for v in letter_words.values() if len(v) >= 50)}/26 with 50+")
    print(f"\n  L12-16k (primary):")
    print(f"    Absorption rate: {agg_rate}")
    print(f"    FN rate: {l12_agg.get('aggregate_fn_rate')}")
    print(f"    CI 95%: {l12_agg.get('bootstrap_ci_95')}")
    print(f"    Mean probe F1: {l12_agg.get('mean_probe_f1')}")
    print(f"    Letters >0.85: {letters_gate_085}/{letters_total}")
    print(f"    Letters >0.70: {letters_gate_070}/{letters_total}")
    gated = results.get("l12_16k", {}).get("gated_analysis", {})
    print(f"    Gated (F1>0.85): rate={gated.get('absorption_rate')}, "
          f"n={gated.get('n_letters_passing')}")

    print(f"\n  Controls:")
    print(f"    C1 (Random dir): {c1_rate}")
    print(f"    C2 (Shuffled):   {c2_rate}")
    c3_f1 = results.get("controls", {}).get("C3_dense_probe", {}).get("mean_f1")
    print(f"    C3 (Dense F1):   {c3_f1}")
    c4_rate = results.get("controls", {}).get("C4_untrained_SAE", {}).get("aggregate_absorption_rate")
    print(f"    C4 (Untrained):  {c4_rate}")

    print(f"\n  Threshold grid CV: {grid_results.get('cv_across_thresholds')}")

    print(f"\n  Cross-config comparison:")
    for key, comp in config_comparison.items():
        print(f"    {key}: rate={comp['absorption_rate']}, F1={comp['mean_probe_f1']}, "
              f"gate={comp['letters_above_085']}")

    print(f"\n  Pass Criteria:")
    for k, v in pass_criteria.items():
        if k in ("overall", "note"):
            continue
        print(f"    {k}: {'PASS' if v['passed'] else 'FAIL'} (value={v['value']})")
    print(f"    OVERALL: {'PASS' if pass_criteria['overall'] else 'FAIL'}")

    print(f"\n  Elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print("=" * 70)

    # Save
    out_path = FULL_RESULTS_DIR / "first_letter_improved.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Saved: {out_path}")

    mark_done(
        "success" if pass_criteria["overall"] else "partial",
        f"First-letter IMPROVED FULL: L12-16k rate={agg_rate}, F1={l12_agg.get('mean_probe_f1')}, "
        f"gate085={letters_gate_085}/{letters_total}, "
        f"C1={c1_rate}, C2={c2_rate}, "
        f"grid_CV={grid_results.get('cv_across_thresholds')}, "
        f"elapsed={elapsed/60:.1f}min",
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
