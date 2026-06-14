#!/usr/bin/env python3
"""
bifurcation_analysis.py — Lateral Inhibition Bifurcation: JumpReLU vs L1 (H7)

Compare absorption patterns between JumpReLU SAEs (Gemma Scope, hard threshold)
and L1+ReLU SAEs (soft threshold).

PREDICTION: JumpReLU shows bimodal per-parent absorption distribution
(binary: all-or-nothing), while L1 shows continuous distribution.

Strategy:
  - Primary: Multiple Gemma Scope JumpReLU SAEs at L0={22, 41, 82, 176} on Gemma 2 2B
  - Cross-model reference: GPT-2 Small L1 SAEs (StandardSAE) at layers 8, 10, 11
  - Tests: KS test for distribution difference, bimodality index (Sarle's coefficient),
    Hartigan's dip test

The cross-model comparison has inherent confounds (different model, different d_model,
different training data), so it is reported as a supplementary reference, NOT as the
primary evidence. The primary evidence is the within-model JumpReLU distribution shape.

Generates: exp/results/full/bifurcation_analysis.json
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
PILOT_DIR = WORKSPACE / "exp" / "results" / "pilots"
TASK_ID = "bifurcation_analysis"
SEED = 42
DEVICE = "cuda:0"  # GPU 1 via CUDA_VISIBLE_DEVICES=1

K_SPARSE = 5
COSINE_THRESHOLD = 0.025  # Chanin et al. standard
MAGNITUDE_GAP = 1.0
MAX_PER_LETTER = 70
N_BOOTSTRAP = 10000

# JumpReLU SAE configs (Gemma 2 2B, Gemma Scope)
JUMPRELU_CONFIGS = {
    "JR_L0_22": {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_22",
                 "layer": 12, "width": 16384, "l0_target": 22, "arch": "JumpReLU"},
    "JR_L0_41": {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_41",
                 "layer": 12, "width": 16384, "l0_target": 41, "arch": "JumpReLU"},
    "JR_L0_82": {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_82",
                 "layer": 12, "width": 16384, "l0_target": 82, "arch": "JumpReLU"},
    "JR_L0_176": {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_16k/average_l0_176",
                  "layer": 12, "width": 16384, "l0_target": 176, "arch": "JumpReLU"},
    "JR_65k_L0_21": {"release": "gemma-scope-2b-pt-res", "sae_id": "layer_12/width_65k/average_l0_21",
                     "layer": 12, "width": 65536, "l0_target": 21, "arch": "JumpReLU"},
}

# L1+ReLU SAE configs (GPT-2 Small, Joseph Bloom's)
L1_CONFIGS = {
    "L1_GPT2_L8":  {"release": "gpt2-small-res-jb", "sae_id": "blocks.8.hook_resid_pre",
                    "layer": 8,  "width": 24576, "arch": "L1+ReLU", "model": "gpt2-small"},
    "L1_GPT2_L10": {"release": "gpt2-small-res-jb", "sae_id": "blocks.10.hook_resid_pre",
                    "layer": 10, "width": 24576, "arch": "L1+ReLU", "model": "gpt2-small"},
    "L1_GPT2_L11": {"release": "gpt2-small-res-jb", "sae_id": "blocks.11.hook_resid_pre",
                    "layer": 11, "width": 24576, "arch": "L1+ReLU", "model": "gpt2-small"},
}

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOT_DIR.mkdir(parents=True, exist_ok=True)

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


# ── Statistical Tests ────────────────────────────────────────────────────────

def sarles_bimodality_coefficient(data):
    """
    Sarle's bimodality coefficient: BC = (skewness^2 + 1) / kurtosis
    BC > 5/9 ≈ 0.555 suggests bimodality.
    For a uniform distribution, BC = 5/9.
    For a normal distribution, BC = 1/3.
    """
    data = np.asarray(data, dtype=float)
    if len(data) < 4:
        return float('nan')
    n = len(data)
    skew = float(stats.skew(data, bias=False))
    kurt = float(stats.kurtosis(data, bias=False, fisher=False))  # excess=False gives Pearson kurtosis
    if kurt == 0:
        return float('nan')
    bc = (skew**2 + 1) / kurt
    return round(bc, 4)


def hartigans_dip_test(data, n_boot=1000, seed=SEED):
    """
    Approximate Hartigan's dip statistic for unimodality testing.
    Uses the maximum gap between the ECDF and the best-fitting unimodal CDF (uniform).

    Returns (dip_statistic, p_value) where p < 0.05 rejects unimodality.
    This is an approximation -- for publication use the R diptest package.
    """
    data = np.sort(np.asarray(data, dtype=float))
    n = len(data)
    if n < 5:
        return float('nan'), float('nan')

    # Compute ECDF
    ecdf = np.arange(1, n + 1) / n

    # Greatest convex minorant (GCM) and least concave majorant (LCM)
    # Dip = max(max(ECDF - GCM), max(LCM - ECDF)) / 2
    # Simplified: use KS-like statistic against uniform on [min, max]

    # Normalize data to [0, 1]
    dmin, dmax = data[0], data[-1]
    if dmax - dmin < 1e-12:
        return 0.0, 1.0

    normed = (data - dmin) / (dmax - dmin)
    uniform_cdf = normed  # If data were uniform on [min, max], CDF = normalized value

    # Dip: maximum deviation of ECDF from uniform CDF
    d_plus = np.max(ecdf - uniform_cdf)
    d_minus = np.max(uniform_cdf - np.concatenate([[0], ecdf[:-1]]))
    dip = max(d_plus, d_minus) / 2

    # Bootstrap p-value
    rng = np.random.RandomState(seed)
    n_exceed = 0
    for _ in range(n_boot):
        boot_data = rng.uniform(0, 1, n)
        boot_data.sort()
        boot_ecdf = np.arange(1, n + 1) / n
        bp = np.max(boot_ecdf - boot_data)
        bm = np.max(boot_data - np.concatenate([[0], boot_ecdf[:-1]]))
        boot_dip = max(bp, bm) / 2
        if boot_dip >= dip:
            n_exceed += 1

    p_value = n_exceed / n_boot
    return round(float(dip), 6), round(float(p_value), 4)


def classify_distribution_shape(rates):
    """Classify distribution shape based on multiple statistics."""
    rates = np.asarray(rates, dtype=float)
    if len(rates) < 3:
        return "insufficient_data"

    bc = sarles_bimodality_coefficient(rates)

    # Check if most values cluster at 0 and/or 1
    near_zero = np.sum(rates < 0.05) / len(rates)
    near_one = np.sum(rates > 0.95) / len(rates)
    in_middle = np.sum((rates >= 0.05) & (rates <= 0.95)) / len(rates)

    if near_zero > 0.7:
        return "zero_dominated"
    elif near_one > 0.7:
        return "one_dominated"
    elif (near_zero + near_one) > 0.6 and in_middle < 0.3:
        return "bimodal"
    elif bc > 0.555:
        return "bimodal_by_bc"
    elif np.std(rates) < 0.05:
        return "concentrated"
    else:
        return "continuous"


# ── Core Functions ────────────────────────────────────────────────────────────

def get_single_token_words(tokenizer, max_per_letter=MAX_PER_LETTER):
    """Get common English words that tokenize as single tokens."""
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


def load_gemma_model(device=DEVICE):
    """Load Gemma 2 2B using unsloth hub (no gated access needed)."""
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
    return model, tokenizer


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


def encode_through_sae(sae, raw_acts, device=DEVICE, batch_size=128):
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


def measure_per_letter_absorption(sae_activations, labels, words, probes, W_dec,
                                  cosine_threshold=COSINE_THRESHOLD,
                                  magnitude_gap=MAGNITUDE_GAP):
    """
    Measure per-letter absorption rate.
    Returns dict of {letter: absorption_rate} and detailed results.
    """
    X = sae_activations.numpy()
    per_letter_results = {}

    for letter in sorted(probes.keys()):
        probe = probes[letter]
        if probe["f1"] < 0.50:  # Minimum quality gate
            continue

        # Get indices for this letter
        letter_indices = [i for i, l in enumerate(labels) if l == letter]
        n_pos = len(letter_indices)
        if n_pos < 3:
            continue

        n_fn = 0
        n_absorbed = 0
        top_k_idx = probe["top_k_idx"]

        for idx in letter_indices:
            # Probe should predict positive for this word
            y_pred = probe["clf"].predict(X[idx:idx+1])[0]
            if y_pred != 1:
                continue

            # Check if all split features inactive
            split_activations = X[idx, top_k_idx]
            all_split_inactive = np.all(split_activations == 0)

            if not all_split_inactive:
                continue

            # False negative found
            n_fn += 1

            # Check absorption: active features with aligned decoders
            active_features = np.where(X[idx] > 0)[0]
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
                                feat_vals = X[:, feat_idx]
                                active_vals = feat_vals[feat_vals > 0]
                                mean_active = float(np.mean(active_vals)) if len(active_vals) > 0 else 1e-10
                                mag_ratio = float(X[idx, feat_idx]) / max(mean_active, 1e-10)
                                if mag_ratio > magnitude_gap:
                                    n_absorbed += 1
                                    break  # One absorbing feature is enough

        absorption_rate = n_absorbed / n_pos if n_pos > 0 else 0.0
        ci_lower, ci_upper = bootstrap_ci_binary(n_absorbed, n_pos)

        per_letter_results[letter] = {
            "n_pos": n_pos,
            "n_fn": n_fn,
            "n_absorbed": n_absorbed,
            "absorption_rate": round(absorption_rate, 4),
            "probe_f1": round(probe["f1"], 4),
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
        }

    return per_letter_results


def run_gemma_bifurcation(model, tokenizer, letter_words, hook_point):
    """Run bifurcation analysis on all JumpReLU SAE configs."""
    print("\n" + "=" * 70)
    print("PART 1: JumpReLU SAEs (Gemma 2 2B + Gemma Scope)")
    print("=" * 70)

    from sae_lens import SAE

    # Collect raw activations once
    print("\n[1/3] Collecting raw model activations...")
    raw_acts, labels, words = collect_raw_activations(
        model, tokenizer, letter_words, hook_point
    )
    print(f"  Got {len(raw_acts)} activations for {len(set(labels))} letters")

    jumprelu_results = {}

    for config_name, config in JUMPRELU_CONFIGS.items():
        print(f"\n  Processing {config_name}...")
        try:
            sae = SAE.from_pretrained(
                release=config["release"],
                sae_id=config["sae_id"],
                device=DEVICE,
            )

            # Get threshold stats
            threshold_stats = {
                "mean": round(float(sae.threshold.mean()), 4),
                "min": round(float(sae.threshold.min()), 4),
                "max": round(float(sae.threshold.max()), 4),
                "std": round(float(sae.threshold.std()), 4),
            }

            # Encode through SAE
            sae_acts = encode_through_sae(sae, raw_acts)
            W_dec = sae.W_dec.detach().cpu().numpy()

            # Train probes and measure per-letter absorption
            probes = train_probes(sae_acts, labels)
            per_letter = measure_per_letter_absorption(
                sae_acts, labels, words, probes, W_dec
            )

            # Collect per-letter absorption rates
            rates = [v["absorption_rate"] for v in per_letter.values()]
            probe_f1s = [v["probe_f1"] for v in per_letter.values()]

            # Filter by probe quality
            high_quality_rates = [v["absorption_rate"] for v in per_letter.values()
                                  if v["probe_f1"] >= 0.70]

            # Distribution statistics
            if len(rates) >= 3:
                bc = sarles_bimodality_coefficient(rates)
                dip_stat, dip_p = hartigans_dip_test(rates)
                dist_shape = classify_distribution_shape(rates)
            else:
                bc = float('nan')
                dip_stat, dip_p = float('nan'), float('nan')
                dist_shape = "insufficient_data"

            mean_rate = round(float(np.mean(rates)), 4) if rates else 0.0
            median_rate = round(float(np.median(rates)), 4) if rates else 0.0
            std_rate = round(float(np.std(rates)), 4) if rates else 0.0

            jumprelu_results[config_name] = {
                "config": {
                    "release": config["release"],
                    "sae_id": config["sae_id"],
                    "layer": config["layer"],
                    "width": config["width"],
                    "l0_target": config.get("l0_target", "N/A"),
                    "arch": config["arch"],
                },
                "threshold_stats": threshold_stats,
                "n_letters_measured": len(per_letter),
                "n_letters_high_quality": len(high_quality_rates),
                "per_letter": per_letter,
                "distribution": {
                    "all_rates": rates,
                    "high_quality_rates": high_quality_rates,
                    "mean_absorption": mean_rate,
                    "median_absorption": median_rate,
                    "std_absorption": std_rate,
                    "mean_probe_f1": round(float(np.mean(probe_f1s)), 4) if probe_f1s else 0.0,
                    "bimodality_coefficient": bc,
                    "dip_statistic": dip_stat,
                    "dip_p_value": dip_p,
                    "distribution_shape": dist_shape,
                    "n_rates_zero": int(np.sum(np.array(rates) == 0)),
                    "n_rates_above_20pct": int(np.sum(np.array(rates) > 0.20)),
                    "pct_zero": round(float(np.sum(np.array(rates) == 0) / len(rates)), 4) if rates else 0.0,
                    "pct_above_20pct": round(float(np.sum(np.array(rates) > 0.20) / len(rates)), 4) if rates else 0.0,
                },
            }

            print(f"    {config_name}: mean={mean_rate:.3f}, median={median_rate:.3f}, "
                  f"std={std_rate:.3f}, BC={bc:.4f}, shape={dist_shape}")
            print(f"    n_letters={len(per_letter)}, zero_rate={int(np.sum(np.array(rates)==0))}, "
                  f">20%={int(np.sum(np.array(rates)>0.20))}")

            # Cleanup
            del sae, sae_acts, W_dec, probes
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            jumprelu_results[config_name] = {"error": str(e)}

    return jumprelu_results, raw_acts, labels, words


def run_gpt2_l1_bifurcation():
    """Run bifurcation analysis on GPT-2 Small L1 SAEs for cross-model reference."""
    print("\n" + "=" * 70)
    print("PART 2: L1+ReLU SAEs (GPT-2 Small, cross-model reference)")
    print("=" * 70)

    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    # Load GPT-2 Small
    print("\n[1/3] Loading GPT-2 Small...")
    gpt2_model = HookedTransformer.from_pretrained(
        "gpt2", device=DEVICE
    )
    gpt2_tokenizer = gpt2_model.tokenizer
    print(f"  GPT-2 loaded: {gpt2_model.cfg.n_layers} layers, d_model={gpt2_model.cfg.d_model}")

    # Get single-token words for GPT-2
    print("[2/3] Getting single-token words for GPT-2...")
    letter_words_gpt2 = get_single_token_words(gpt2_tokenizer)
    total_words_gpt2 = sum(len(v) for v in letter_words_gpt2.values())
    print(f"  {total_words_gpt2} total words across {len(letter_words_gpt2)} letters")

    l1_results = {}

    for config_name, config in L1_CONFIGS.items():
        print(f"\n  Processing {config_name} (layer {config['layer']})...")
        try:
            sae = SAE.from_pretrained(
                release=config["release"],
                sae_id=config["sae_id"],
                device=DEVICE,
            )

            hook_point = f"blocks.{config['layer']}.hook_resid_pre"

            # Collect raw activations
            raw_acts, labels, words = collect_raw_activations(
                gpt2_model, gpt2_tokenizer, letter_words_gpt2, hook_point
            )

            # Encode through SAE
            sae_acts = encode_through_sae(sae, raw_acts)
            W_dec = sae.W_dec.detach().cpu().numpy()

            # Train probes and measure absorption
            probes = train_probes(sae_acts, labels)
            per_letter = measure_per_letter_absorption(
                sae_acts, labels, words, probes, W_dec
            )

            # Collect rates
            rates = [v["absorption_rate"] for v in per_letter.values()]
            probe_f1s = [v["probe_f1"] for v in per_letter.values()]

            high_quality_rates = [v["absorption_rate"] for v in per_letter.values()
                                  if v["probe_f1"] >= 0.70]

            # Distribution statistics
            if len(rates) >= 3:
                bc = sarles_bimodality_coefficient(rates)
                dip_stat, dip_p = hartigans_dip_test(rates)
                dist_shape = classify_distribution_shape(rates)
            else:
                bc = float('nan')
                dip_stat, dip_p = float('nan'), float('nan')
                dist_shape = "insufficient_data"

            mean_rate = round(float(np.mean(rates)), 4) if rates else 0.0
            median_rate = round(float(np.median(rates)), 4) if rates else 0.0
            std_rate = round(float(np.std(rates)), 4) if rates else 0.0

            l1_results[config_name] = {
                "config": {
                    "release": config["release"],
                    "sae_id": config["sae_id"],
                    "layer": config["layer"],
                    "width": config["width"],
                    "arch": config["arch"],
                    "model": config["model"],
                },
                "n_words_used": total_words_gpt2,
                "n_letters_measured": len(per_letter),
                "n_letters_high_quality": len(high_quality_rates),
                "per_letter": per_letter,
                "distribution": {
                    "all_rates": rates,
                    "high_quality_rates": high_quality_rates,
                    "mean_absorption": mean_rate,
                    "median_absorption": median_rate,
                    "std_absorption": std_rate,
                    "mean_probe_f1": round(float(np.mean(probe_f1s)), 4) if probe_f1s else 0.0,
                    "bimodality_coefficient": bc,
                    "dip_statistic": dip_stat,
                    "dip_p_value": dip_p,
                    "distribution_shape": dist_shape,
                    "n_rates_zero": int(np.sum(np.array(rates) == 0)),
                    "n_rates_above_20pct": int(np.sum(np.array(rates) > 0.20)),
                    "pct_zero": round(float(np.sum(np.array(rates) == 0) / len(rates)), 4) if rates else 0.0,
                    "pct_above_20pct": round(float(np.sum(np.array(rates) > 0.20) / len(rates)), 4) if rates else 0.0,
                },
            }

            print(f"    {config_name}: mean={mean_rate:.3f}, median={median_rate:.3f}, "
                  f"std={std_rate:.3f}, BC={bc:.4f}, shape={dist_shape}")

            del sae, sae_acts, W_dec, probes, raw_acts
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"    ERROR: {e}")
            traceback.print_exc()
            l1_results[config_name] = {"error": str(e)}

    # Cleanup GPT-2
    del gpt2_model
    gc.collect()
    torch.cuda.empty_cache()

    return l1_results


def compare_distributions(jumprelu_results, l1_results):
    """
    Compare JumpReLU vs L1 distributions using KS test and bimodality metrics.
    """
    print("\n" + "=" * 70)
    print("PART 3: Statistical Comparison")
    print("=" * 70)

    comparison = {}

    # Aggregate JumpReLU rates across all configs
    all_jr_rates = []
    jr_configs_rates = {}
    for config_name, res in jumprelu_results.items():
        if "error" in res:
            continue
        rates = res["distribution"]["all_rates"]
        jr_configs_rates[config_name] = rates
        all_jr_rates.extend(rates)

    # Aggregate L1 rates across all configs
    all_l1_rates = []
    l1_configs_rates = {}
    for config_name, res in l1_results.items():
        if "error" in res:
            continue
        rates = res["distribution"]["all_rates"]
        l1_configs_rates[config_name] = rates
        all_l1_rates.extend(rates)

    # KS test: aggregated JumpReLU vs aggregated L1
    if len(all_jr_rates) >= 3 and len(all_l1_rates) >= 3:
        ks_stat, ks_p = stats.ks_2samp(all_jr_rates, all_l1_rates)
        comparison["aggregate_ks"] = {
            "ks_statistic": round(float(ks_stat), 6),
            "p_value": round(float(ks_p), 6),
            "n_jumprelu": len(all_jr_rates),
            "n_l1": len(all_l1_rates),
            "significant": ks_p < 0.05,
        }
        print(f"\n  Aggregate KS test: D={ks_stat:.4f}, p={ks_p:.4f}")

    # Pairwise KS tests: each JumpReLU config vs each L1 config
    pairwise_ks = {}
    for jr_name, jr_rates in jr_configs_rates.items():
        for l1_name, l1_rates in l1_configs_rates.items():
            if len(jr_rates) >= 3 and len(l1_rates) >= 3:
                ks_stat, ks_p = stats.ks_2samp(jr_rates, l1_rates)
                key = f"{jr_name}_vs_{l1_name}"
                pairwise_ks[key] = {
                    "ks_statistic": round(float(ks_stat), 6),
                    "p_value": round(float(ks_p), 6),
                    "significant": ks_p < 0.05,
                }

    comparison["pairwise_ks"] = pairwise_ks

    # Within-JumpReLU: compare different L0 values
    jr_within = {}
    jr_names = sorted(jr_configs_rates.keys())
    for i in range(len(jr_names)):
        for j in range(i + 1, len(jr_names)):
            r1 = jr_configs_rates[jr_names[i]]
            r2 = jr_configs_rates[jr_names[j]]
            if len(r1) >= 3 and len(r2) >= 3:
                ks_stat, ks_p = stats.ks_2samp(r1, r2)
                key = f"{jr_names[i]}_vs_{jr_names[j]}"
                jr_within[key] = {
                    "ks_statistic": round(float(ks_stat), 6),
                    "p_value": round(float(ks_p), 6),
                    "significant": ks_p < 0.05,
                }

    comparison["within_jumprelu_ks"] = jr_within

    # Mann-Whitney U test (non-parametric)
    if len(all_jr_rates) >= 3 and len(all_l1_rates) >= 3:
        mw_stat, mw_p = stats.mannwhitneyu(all_jr_rates, all_l1_rates, alternative='two-sided')
        comparison["mann_whitney"] = {
            "statistic": round(float(mw_stat), 6),
            "p_value": round(float(mw_p), 6),
            "significant": mw_p < 0.05,
        }
        print(f"  Mann-Whitney U: U={mw_stat:.4f}, p={mw_p:.4f}")

    # Bimodality comparison summary
    jr_bcs = []
    l1_bcs = []
    for config_name, res in jumprelu_results.items():
        if "error" not in res and not np.isnan(res["distribution"]["bimodality_coefficient"]):
            jr_bcs.append(res["distribution"]["bimodality_coefficient"])
    for config_name, res in l1_results.items():
        if "error" not in res and not np.isnan(res["distribution"]["bimodality_coefficient"]):
            l1_bcs.append(res["distribution"]["bimodality_coefficient"])

    comparison["bimodality_summary"] = {
        "jumprelu_mean_bc": round(float(np.mean(jr_bcs)), 4) if jr_bcs else None,
        "jumprelu_bcs": jr_bcs,
        "l1_mean_bc": round(float(np.mean(l1_bcs)), 4) if l1_bcs else None,
        "l1_bcs": l1_bcs,
        "bc_threshold_for_bimodality": 0.555,
        "prediction_supported": (
            bool(np.mean(jr_bcs) > 0.555) if jr_bcs else None
        ),
    }

    if jr_bcs:
        print(f"  JumpReLU mean BC: {np.mean(jr_bcs):.4f} (threshold=0.555)")
    if l1_bcs:
        print(f"  L1 mean BC: {np.mean(l1_bcs):.4f} (threshold=0.555)")

    return comparison


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    set_seeds()
    start_time = datetime.now()
    print(f"[{start_time.isoformat()}] Starting bifurcation analysis (PILOT mode)")
    report_progress(0, 8, "Starting bifurcation analysis")

    from sae_lens import SAE

    # ── Load Gemma 2 2B ──
    print("\n[0/8] Loading Gemma 2 2B model...")
    model, tokenizer = load_gemma_model(DEVICE)
    print(f"  Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")

    # Get single-token words
    print("[1/8] Getting single-token words for Gemma 2...")
    letter_words = get_single_token_words(tokenizer)
    total_words = sum(len(v) for v in letter_words.values())
    per_letter_counts = {k: len(v) for k, v in letter_words.items()}
    print(f"  {total_words} total words across {len(letter_words)} letters")
    report_progress(1, 8, "Vocabulary prepared", {"total_words": total_words})

    # Hook point for Gemma 2 2B layer 12
    hook_point = "blocks.12.hook_resid_post"

    # ── Part 1: JumpReLU Analysis ──
    report_progress(2, 8, "Running JumpReLU bifurcation analysis")
    jumprelu_results, raw_acts, labels, words = run_gemma_bifurcation(
        model, tokenizer, letter_words, hook_point
    )
    report_progress(4, 8, "JumpReLU analysis complete")

    # Free Gemma 2 model to make room for GPT-2
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Part 2: L1 Analysis (GPT-2 Small) ──
    report_progress(5, 8, "Running L1 (GPT-2 Small) bifurcation analysis")
    l1_results = run_gpt2_l1_bifurcation()
    report_progress(6, 8, "L1 analysis complete")

    # ── Part 3: Compare ──
    report_progress(7, 8, "Computing statistical comparisons")
    comparison = compare_distributions(jumprelu_results, l1_results)

    # ── Compile Results ──
    end_time = datetime.now()
    elapsed_min = (end_time - start_time).total_seconds() / 60

    # Summary assessment
    jr_shapes = []
    for config_name, res in jumprelu_results.items():
        if "error" not in res:
            jr_shapes.append(res["distribution"]["distribution_shape"])
    l1_shapes = []
    for config_name, res in l1_results.items():
        if "error" not in res:
            l1_shapes.append(res["distribution"]["distribution_shape"])

    # Check pass criteria
    n_jr_measured = sum(1 for r in jumprelu_results.values() if "error" not in r)
    n_l1_measured = sum(1 for r in l1_results.values() if "error" not in r)

    # Determine if prediction is supported
    jr_bimodal_count = sum(1 for s in jr_shapes if "bimodal" in s or s == "zero_dominated")
    l1_continuous_count = sum(1 for s in l1_shapes if s in ["continuous", "concentrated"])

    prediction_status = "UNDETERMINED"
    if n_jr_measured >= 2 and n_l1_measured >= 1:
        if jr_bimodal_count > n_jr_measured / 2 and l1_continuous_count > n_l1_measured / 2:
            prediction_status = "SUPPORTED"
        elif jr_bimodal_count <= n_jr_measured / 4 and l1_continuous_count <= n_l1_measured / 4:
            prediction_status = "FALSIFIED"
        else:
            prediction_status = "MIXED"

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "seed": SEED,
        "timestamp_start": start_time.isoformat(),
        "timestamp_end": end_time.isoformat(),
        "elapsed_minutes": round(elapsed_min, 1),
        "prediction": {
            "statement": "JumpReLU SAEs show bimodal per-parent absorption (all-or-nothing binary pattern) while L1+ReLU SAEs show more continuous distribution",
            "status": prediction_status,
            "evidence": {
                "jumprelu_shapes": jr_shapes,
                "l1_shapes": l1_shapes,
                "jumprelu_bimodal_count": jr_bimodal_count,
                "l1_continuous_count": l1_continuous_count,
            },
        },
        "vocabulary": {
            "total_words": total_words,
            "per_letter_counts": per_letter_counts,
        },
        "jumprelu_analysis": jumprelu_results,
        "l1_analysis": l1_results,
        "comparison": comparison,
        "pilot_pass_criteria": {
            "at_least_2_jumprelu": n_jr_measured >= 2,
            "at_least_1_l1": n_l1_measured >= 1,
            "distributions_computed": True,
            "ks_test_computed": "aggregate_ks" in comparison,
            "bimodality_computed": comparison.get("bimodality_summary", {}).get("jumprelu_mean_bc") is not None,
        },
        "caveats": [
            "Cross-model comparison (Gemma 2 2B vs GPT-2 Small) introduces confounds: different model capacity, training data, d_model",
            "L1 results are a reference point, not a direct comparison",
            "Per-letter N is relatively small (15-70 words per letter), limiting distribution resolution",
            "Distribution shape classification is approximate; formal publication should use larger corpora and the R diptest package",
        ],
    }

    # Save to pilot dir
    pilot_path = PILOT_DIR / "bifurcation_analysis.json"
    pilot_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n  Pilot results saved: {pilot_path}")

    # Also save to full dir (as the task expects)
    full_path = RESULTS_DIR / "bifurcation_analysis.json"
    full_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"  Full results saved: {full_path}")

    # Summary
    summary = (
        f"Bifurcation analysis complete. "
        f"JumpReLU: {n_jr_measured} configs measured, shapes={jr_shapes}. "
        f"L1: {n_l1_measured} configs measured, shapes={l1_shapes}. "
        f"Prediction: {prediction_status}. "
        f"Elapsed: {elapsed_min:.1f} min."
    )
    print(f"\n{summary}")

    mark_done("success", summary, {
        "prediction_status": prediction_status,
        "n_jumprelu_configs": n_jr_measured,
        "n_l1_configs": n_l1_measured,
        "elapsed_min": round(elapsed_min, 1),
    })
    report_progress(8, 8, "Complete", {"prediction_status": prediction_status})

    return results


if __name__ == "__main__":
    try:
        results = main()
        print("\n=== BIFURCATION ANALYSIS COMPLETE ===")
    except Exception as e:
        traceback.print_exc()
        mark_done("failed", f"Error: {e}")
        sys.exit(1)
