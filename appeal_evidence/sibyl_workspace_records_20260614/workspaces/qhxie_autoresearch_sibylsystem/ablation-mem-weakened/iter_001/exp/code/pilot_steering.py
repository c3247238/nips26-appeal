#!/usr/bin/env python3
"""
Pilot: Feature Steering Experiment for First-Letter Features (A-Z)
Model: GPT-2 Small, Layer 8, 32K dictionary (jbloom SAEs)

Approach:
1. Load absorption results to identify first-letter features
2. Steer each feature at strengths [1.0, 2.0, 5.0, 10.0]
3. Measure target-letter token probability increase
4. Include random feature baseline and null steering control
5. Compare HIGH vs LOW absorption features
"""

import os
import sys
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from safetensors.torch import load_file

# Configuration
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
RESULTS_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/ablation-mem-weakened/current/exp/results/pilots")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Task identification
TASK_ID = "pilot_steering"
PID_FILE = RESULTS_DIR.parent / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR.parent / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR.parent / f"{TASK_ID}_DONE"

# Write PID file immediately
PID_FILE.write_text(str(os.getpid()))

def report_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress = {
        "task_id": TASK_ID,
        "epoch": epoch, "total_epochs": total_epochs,
        "step": step, "total_steps": total_steps,
        "loss": loss, "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress))

def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except:
            pass
    marker = {
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }
    DONE_FILE.write_text(json.dumps(marker))

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)

# Set random seed
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print(f"[{TASK_ID}] Starting feature steering pilot")
print(f"  Device: {DEVICE} (GPU {GPU_ID})")
print(f"  Seed: {SEED}")

from transformer_lens import HookedTransformer

MODEL_NAME = "gpt2"
SAE_LAYER = 8
SAE_DICT_SIZE = 32768

# Steering strengths to test
STEERING_STRENGTHS = [0.0, 1.0, 2.0, 5.0, 10.0]

# Number of test prompts per letter
N_PROMPTS = 100

print(f"\nLoading model: {MODEL_NAME}")
report_progress(0, 6, metric={"stage": "loading_model"})

model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype=torch.float32)
print(f"  Model loaded: {model.cfg.n_layers} layers, {model.cfg.d_model} d_model")

print(f"\nLoading SAE for layer {SAE_LAYER}")
report_progress(1, 6, metric={"stage": "loading_sae"})

sae_cache_dir = Path.home() / ".cache/huggingface/hub/models--jbloom--GPT2-Small-OAI-v5-32k-resid-post-SAEs/snapshots"
sae_dirs = list(sae_cache_dir.glob(f"*/v5_32k_layer_{SAE_LAYER}.pt"))
if not sae_dirs:
    print(f"ERROR: No cached SAE found for layer {SAE_LAYER}")
    mark_done("failed", f"No cached SAE for layer {SAE_LAYER}")
    sys.exit(1)

sae_dir = sae_dirs[0]
sae_weights_file = sae_dir / "sae_weights.safetensors"
sae_cfg_file = sae_dir / "cfg.json"

with open(sae_cfg_file) as f:
    sae_cfg = json.load(f)

sae_weights = load_file(sae_weights_file, device=DEVICE)
W_enc = sae_weights["W_enc"]
W_dec = sae_weights["W_dec"]
b_enc = sae_weights["b_enc"]
b_dec = sae_weights["b_dec"]

d_model = sae_cfg["d_in"]
d_sae = sae_cfg["d_sae"]

print(f"  d_model: {d_model}, d_sae: {d_sae}")

# Load absorption results
absorption_file = RESULTS_DIR / "absorption_layer8_16k.json"
if not absorption_file.exists():
    print(f"ERROR: Absorption results not found at {absorption_file}")
    mark_done("failed", "Absorption results not found")
    sys.exit(1)

with open(absorption_file) as f:
    absorption_data = json.load(f)

# Extract feature mapping: letter -> (feature_id, absorption_rate)
letter_features = {}
for letter, info in absorption_data["results"].items():
    letter_features[letter] = {
        "feature_id": info["feature_id"],
        "absorption_rate": info["absorption_rate"],
        "cosine_similarity": info.get("cosine_similarity", 0.0),
    }

# Get unique features (some letters may share features)
unique_features = {}
for letter, info in letter_features.items():
    fid = info["feature_id"]
    if fid not in unique_features:
        unique_features[fid] = {
            "letters": [letter],
            "absorption_rate": info["absorption_rate"],
        }
    else:
        unique_features[fid]["letters"].append(letter)

print(f"\nLoaded {len(letter_features)} letter features ({len(unique_features)} unique)")
print(f"  HIGH absorption: {absorption_data['summary']['high_count']}")
print(f"  LOW absorption: {absorption_data['summary']['low_count']}")

# Generate test prompts
print(f"\nGenerating {N_PROMPTS} test prompts per letter...")
report_progress(2, 6, metric={"stage": "generating_prompts"})

# Use a diverse set of sentence templates with blanks
sentence_templates = [
    "The word starts with the letter _",
    "I am thinking of a word that begins with _",
    "The answer starts with _",
    "A common word beginning with _ is",
    "The letter _ is the first letter of",
    "Many words start with _ such as",
    "The alphabet letter _ comes before",
    "Words like _ are common",
    "The first letter is _ in the word",
    "_ is a popular starting letter",
    "Think of a word starting with _",
    "The word beginning with _ means",
    "A _ word is one that starts with",
    "The letter _ appears at the start",
    "Many names begin with _ like",
]

# Word lists for each letter
word_lists = {
    'A': ['apple', 'ant', 'arrow', 'anchor', 'april', 'art', 'atom', 'ace', 'arm', 'axe', 'area', 'ask', 'add', 'age', 'ago', 'aid', 'aim', 'air', 'all', 'alone', 'along', 'also', 'always', 'am', 'among', 'an', 'and', 'anger', 'animal', 'another', 'answer', 'any', 'appear', 'apple', 'are', 'arrive', 'art', 'as', 'ask', 'at', 'away'],
    'B': ['banana', 'bird', 'boat', 'book', 'blue', 'bread', 'ball', 'bear', 'bell', 'bone', 'baby', 'back', 'bad', 'bag', 'bake', 'balance', 'ball', 'band', 'bank', 'bar', 'base', 'basket', 'bat', 'bath', 'be', 'beach', 'bean', 'bear', 'beat', 'beautiful', 'because', 'become', 'bed', 'been', 'before', 'began', 'begin', 'behind', 'believe', 'bell'],
    'C': ['cat', 'car', 'cake', 'cold', 'cloud', 'city', 'coin', 'corn', 'cup', 'cow', 'call', 'came', 'can', 'cap', 'car', 'card', 'care', 'careful', 'carefully', 'carry', 'case', 'cat', 'catch', 'caught', 'cause', 'cell', 'center', 'century', 'certain', 'chance', 'change', 'chart', 'check', 'chief', 'child', 'children', 'choose', 'church', 'circle', 'city'],
    'D': ['dog', 'door', 'desk', 'dance', 'dark', 'dream', 'doll', 'duck', 'dust', 'dig', 'daily', 'dance', 'danger', 'dark', 'date', 'daughter', 'day', 'dead', 'deal', 'dear', 'death', 'decide', 'deep', 'degree', 'depend', 'describe', 'desert', 'design', 'destroy', 'detail', 'determine', 'develop', 'dictionary', 'did', 'died', 'difference', 'different', 'difficult', 'direct', 'direction'],
    'E': ['elephant', 'egg', 'earth', 'easy', 'energy', 'eagle', 'edge', 'exit', 'end', 'ear', 'each', 'early', 'earth', 'east', 'easy', 'eat', 'edge', 'education', 'effect', 'effort', 'egg', 'eight', 'either', 'electric', 'element', 'else', 'end', 'enemy', 'energy', 'engine', 'enjoy', 'enough', 'enter', 'entire', 'equal', 'especially', 'even', 'evening', 'event', 'ever'],
    'F': ['fish', 'fire', 'flower', 'fast', 'food', 'forest', 'frog', 'flag', 'fan', 'foot', 'face', 'fact', 'factory', 'failed', 'fair', 'fall', 'family', 'famous', 'far', 'farm', 'fast', 'fat', 'father', 'fear', 'feel', 'feet', 'fell', 'felt', 'few', 'field', 'fight', 'figure', 'fill', 'film', 'final', 'finally', 'find', 'fine', 'finger', 'finish'],
    'G': ['grape', 'green', 'grass', 'game', 'gold', 'glass', 'goat', 'gift', 'gate', 'girl', 'game', 'garden', 'gas', 'gather', 'gave', 'general', 'gentle', 'get', 'girl', 'give', 'glad', 'glass', 'go', 'gold', 'gone', 'good', 'got', 'government', 'grass', 'gray', 'great', 'green', 'grew', 'ground', 'group', 'grow', 'guess', 'guide', 'gun', 'guy'],
    'H': ['house', 'horse', 'happy', 'hot', 'heart', 'hill', 'hat', 'hand', 'hen', 'hook', 'had', 'hair', 'half', 'hall', 'hand', 'happen', 'happy', 'hard', 'has', 'hat', 'have', 'he', 'head', 'hear', 'heard', 'heart', 'heat', 'heavy', 'held', 'help', 'her', 'here', 'high', 'hill', 'him', 'himself', 'his', 'history', 'hit', 'hold'],
    'I': ['ice', 'igloo', 'iron', 'island', 'idea', 'image', 'inch', 'ink', 'iron', 'ice', 'ice', 'idea', 'if', 'imagine', 'important', 'in', 'inch', 'include', 'increase', 'indeed', 'indicate', 'industry', 'information', 'inside', 'instead', 'instrument', 'interest', 'interesting', 'into', 'iron', 'is', 'island', 'it', 'its', 'itself', 'improve', 'include', 'income', 'increase', 'individual'],
    'J': ['jump', 'juice', 'jelly', 'job', 'join', 'joke', 'jar', 'jet', 'jam', 'jaw', 'job', 'join', 'joy', 'judge', 'jump', 'just', 'jack', 'jane', 'january', 'japan', 'jar', 'jazz', 'jeans', 'jeep', 'jet', 'jewel', 'jill', 'job', 'joe', 'john', 'join', 'joke', 'joy', 'judge', 'juice', 'july', 'jump', 'june', 'jury', 'just'],
    'K': ['kite', 'king', 'key', 'kick', 'kind', 'knife', 'kitten', 'kangaroo', 'kite', 'knee', 'keen', 'keep', 'kept', 'key', 'kid', 'kill', 'kind', 'king', 'kiss', 'kitchen', 'knew', 'knife', 'know', 'knowledge', 'known', 'kept', 'key', 'kick', 'kid', 'kill', 'kind', 'king', 'kiss', 'kitchen', 'kite', 'knew', 'knife', 'knock', 'know', 'known'],
    'L': ['lion', 'leaf', 'light', 'long', 'love', 'lake', 'lamp', 'lady', 'leg', 'log', 'labor', 'lack', 'ladder', 'lady', 'lake', 'land', 'language', 'large', 'last', 'late', 'laugh', 'law', 'lay', 'lead', 'learn', 'least', 'leave', 'led', 'left', 'leg', 'length', 'less', 'let', 'letter', 'level', 'library', 'lie', 'life', 'lift', 'light'],
    'M': ['monkey', 'moon', 'mountain', 'music', 'magic', 'mirror', 'mouse', 'milk', 'map', 'man', 'machine', 'made', 'magazine', 'main', 'major', 'make', 'man', 'many', 'map', 'march', 'mark', 'market', 'marry', 'mass', 'match', 'material', 'matter', 'may', 'maybe', 'me', 'mean', 'measure', 'meat', 'meet', 'member', 'memory', 'men', 'metal', 'method', 'middle'],
    'N': ['nest', 'night', 'name', 'new', 'nine', 'noise', 'nose', 'neck', 'net', 'nut', 'nail', 'name', 'nation', 'natural', 'nature', 'near', 'nearly', 'necessary', 'neck', 'need', 'needle', 'neighbor', 'neither', 'nerve', 'net', 'never', 'new', 'news', 'next', 'nice', 'night', 'nine', 'no', 'nobody', 'nod', 'noise', 'none', 'noon', 'nor', 'north'],
    'O': ['orange', 'ocean', 'octopus', 'old', 'open', 'oval', 'oven', 'owl', 'oil', 'oak', 'object', 'observe', 'occur', 'ocean', 'october', 'off', 'offer', 'office', 'often', 'oil', 'old', 'omit', 'once', 'one', 'only', 'open', 'operate', 'opinion', 'opportunity', 'opposite', 'orange', 'order', 'organ', 'original', 'other', 'otherwise', 'ought', 'our', 'out', 'outside'],
    'P': ['pig', 'pen', 'paper', 'purple', 'people', 'piano', 'pear', 'park', 'pot', 'pan', 'page', 'pain', 'paint', 'pair', 'paper', 'parent', 'park', 'part', 'party', 'pass', 'past', 'path', 'pattern', 'pay', 'peace', 'people', 'per', 'perfect', 'perhaps', 'period', 'person', 'phrase', 'pick', 'picture', 'piece', 'place', 'plain', 'plan', 'plane', 'plant'],
    'Q': ['queen', 'quiet', 'quick', 'question', 'quilt', 'quiz', 'quack', 'quote', 'quit', 'quest', 'quality', 'quarter', 'queen', 'question', 'quick', 'quickly', 'quiet', 'quite', 'quiz', 'quote', 'quality', 'quantity', 'quarter', 'queen', 'question', 'quick', 'quiet', 'quite', 'quiz', 'quotation', 'quote', 'quality', 'quantity', 'quarter', 'queen', 'question', 'quick', 'quiet', 'quite', 'quiz'],
    'R': ['rabbit', 'red', 'rain', 'river', 'round', 'road', 'rose', 'ring', 'rat', 'rock', 'race', 'radio', 'rail', 'rain', 'raise', 'ran', 'range', 'rapid', 'rare', 'rate', 'rather', 'raw', 'ray', 'reach', 'read', 'ready', 'real', 'realize', 'really', 'reason', 'receive', 'record', 'red', 'remember', 'repeat', 'reply', 'report', 'represent', 'require', 'rest'],
    'S': ['sun', 'star', 'snake', 'school', 'small', 'sweet', 'sea', 'sand', 'sock', 'ship', 'sad', 'safe', 'said', 'sail', 'salt', 'same', 'sand', 'sat', 'save', 'saw', 'say', 'scale', 'school', 'science', 'score', 'sea', 'season', 'seat', 'second', 'section', 'see', 'seed', 'seem', 'seen', 'select', 'self', 'sell', 'send', 'sense', 'sent'],
    'T': ['tiger', 'tree', 'table', 'time', 'tall', 'train', 'tent', 'toy', 'top', 'toe', 'table', 'tail', 'take', 'talk', 'tall', 'teacher', 'team', 'tear', 'tell', 'temperature', 'ten', 'term', 'test', 'than', 'thank', 'that', 'the', 'their', 'them', 'themselves', 'then', 'there', 'these', 'they', 'thick', 'thin', 'thing', 'think', 'third', 'this'],
    'U': ['umbrella', 'under', 'up', 'use', 'unit', 'uncle', 'uniform', 'urn', 'ugly', 'urge', 'ugly', 'uncle', 'under', 'understand', 'unit', 'until', 'up', 'upon', 'us', 'use', 'usual', 'usually', 'umbrella', 'unable', 'uncle', 'under', 'understand', 'uniform', 'union', 'unique', 'unit', 'university', 'unless', 'until', 'unusual', 'up', 'upon', 'us', 'use', 'using'],
    'V': ['violin', 'violet', 'voice', 'visit', 'very', 'village', 'valley', 'vest', 'van', 'vase', 'valley', 'value', 'van', 'vary', 'verb', 'very', 'village', 'visit', 'voice', 'volume', 'vote', 'vowel', 'vacation', 'valley', 'value', 'van', 'vase', 'vegetable', 'verb', 'very', 'victory', 'view', 'village', 'visit', 'voice', 'volume', 'vote', 'voyage', 'valley', 'value'],
    'W': ['water', 'wolf', 'window', 'white', 'warm', 'world', 'wind', 'wall', 'web', 'wing', 'wait', 'wake', 'walk', 'wall', 'want', 'war', 'warm', 'was', 'wash', 'watch', 'water', 'wave', 'way', 'we', 'weak', 'wear', 'weather', 'week', 'weight', 'welcome', 'well', 'went', 'were', 'west', 'wet', 'what', 'wheel', 'when', 'where', 'whether'],
    'X': ['xylophone', 'xray', 'box', 'fox', 'six', 'mix', 'fix', 'next', 'axe', 'fox', 'xray', 'xylophone', 'exact', 'example', 'except', 'excite', 'exercise', 'expect', 'expense', 'experience', 'explain', 'express', 'extend', 'extra', 'eye', 'xray', 'xylophone', 'exact', 'example', 'except', 'excite', 'exercise', 'expect', 'expense', 'experience', 'explain', 'express', 'extend', 'extra', 'eye'],
    'Y': ['yellow', 'yes', 'young', 'year', 'yard', 'yesterday', 'yawn', 'yacht', 'yam', 'yolk', 'yard', 'year', 'yellow', 'yes', 'yet', 'you', 'young', 'your', 'yard', 'year', 'yellow', 'yes', 'yesterday', 'yet', 'you', 'young', 'your', 'yard', 'year', 'yellow', 'yes', 'yesterday', 'yet', 'you', 'young', 'your', 'yard', 'year', 'yellow', 'yes'],
    'Z': ['zebra', 'zoo', 'zero', 'zone', 'zip', 'zigzag', 'zest', 'zoom', 'zoo', 'zinc', 'zoo', 'zero', 'zone', 'zip', 'zinc', 'zebra', 'zigzag', 'zest', 'zoom', 'zone', 'zoo', 'zero', 'zip', 'zinc', 'zebra', 'zigzag', 'zest', 'zoom', 'zone', 'zoo', 'zero', 'zip', 'zinc', 'zebra', 'zigzag', 'zest', 'zoom', 'zone', 'zoo', 'zero'],
}

def generate_prompts(letter, n=100):
    """Generate diverse test prompts for a letter."""
    words = word_lists.get(letter, [])
    prompts = []
    for i in range(n):
        template = sentence_templates[i % len(sentence_templates)]
        word = words[i % len(words)]
        # Create prompt: "The word starts with the letter A: apple"
        prompt = f"{template.replace('_', letter)}: {word}"
        prompts.append(prompt)
    return prompts

# Generate all prompts
all_prompts = {}
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    all_prompts[letter] = generate_prompts(letter, N_PROMPTS)

print(f"  Generated {sum(len(v) for v in all_prompts.values())} total prompts")

# Step 3: Define steering function
print(f"\nSetting up steering hooks...")
report_progress(3, 6, metric={"stage": "setting_up_steering"})

def steer_feature_hook(direction, strength):
    """Create a hook function that adds direction * strength to residual stream."""
    def hook_fn(activations, hook):
        # activations shape: (batch, seq_len, d_model)
        # Add steering vector to all positions
        steering = direction * strength
        return activations + steering
    return hook_fn

def measure_target_probability(model, prompts, target_letter, hook_fn=None, hook_point=None, batch_size=8):
    """
    Measure the probability of tokens starting with target_letter in model outputs.
    Returns average probability increase and success rate.
    """
    target_tokens = set()
    target_letter_lower = target_letter.lower()

    # Find all tokens that start with target letter
    for token_id in range(model.cfg.d_vocab):
        token_str = model.to_string(token_id).strip().lower()
        if token_str.startswith(target_letter_lower) and len(token_str) >= 2:
            target_tokens.add(token_id)

    if len(target_tokens) == 0:
        return {"mean_prob": 0.0, "success_rate": 0.0, "n_target_tokens": 0}

    probs = []
    successes = []

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        tokens = model.to_tokens(batch, padding_side="right")

        with torch.no_grad():
            if hook_fn and hook_point:
                logits = model.run_with_hooks(
                    tokens,
                    fwd_hooks=[(hook_point, hook_fn)],
                    return_type="logits"
                )
            else:
                logits = model(tokens, return_type="logits")

            # Get probabilities for next token prediction
            # logits shape: (batch, seq_len, vocab)
            next_token_logits = logits[:, -1, :]  # Last position
            probs_dist = torch.softmax(next_token_logits, dim=-1)

            # Sum probability mass on target tokens
            for b in range(probs_dist.shape[0]):
                target_prob = sum(probs_dist[b, tid].item() for tid in target_tokens if tid < probs_dist.shape[1])
                probs.append(target_prob)
                successes.append(1 if target_prob > 0.1 else 0)  # Success if >10% prob mass

    return {
        "mean_prob": float(np.mean(probs)),
        "success_rate": float(np.mean(successes)),
        "n_target_tokens": len(target_tokens),
        "probs": probs,
    }

# Step 4: Run steering experiments
print(f"\nRunning steering experiments...")
report_progress(4, 6, metric={"stage": "running_steering"})

hook_point = f"blocks.{SAE_LAYER}.hook_resid_post"

# Results storage
steering_results = {}

# Select random features for baseline
np.random.seed(SEED)
random_feature_ids = np.random.choice(d_sae, size=26, replace=False).tolist()

# Track which letters have unique features (not shared)
letters_with_unique_features = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    fid = letter_features[letter]["feature_id"]
    # Check if this feature is shared
    count = sum(1 for v in letter_features.values() if v["feature_id"] == fid)
    if count == 1:
        letters_with_unique_features.append(letter)

print(f"  Letters with unique features: {len(letters_with_unique_features)}")
print(f"  Letters: {letters_with_unique_features}")

# For each letter, test steering at different strengths
for idx, letter in enumerate(tqdm("ABCDEFGHIJKLMNOPQRSTUVWXYZ", desc="Steering letters")):
    info = letter_features[letter]
    feature_id = info["feature_id"]
    absorption_rate = info["absorption_rate"]

    # Get decoder direction for this feature
    direction = W_dec[feature_id]  # (d_model,)
    direction = direction / (direction.norm() + 1e-8)  # Normalize

    prompts = all_prompts[letter][:20]  # Use 20 prompts per letter for speed (pilot)

    letter_results = {
        "letter": letter,
        "feature_id": feature_id,
        "absorption_rate": absorption_rate,
        "direction_norm": float(direction.norm().cpu()),
        "strength_results": {},
    }

    # Test each steering strength
    for strength in STEERING_STRENGTHS:
        if strength == 0.0:
            # Null steering (no hook)
            result = measure_target_probability(model, prompts, letter, hook_fn=None, hook_point=None)
        else:
            hook_fn = steer_feature_hook(direction, strength)
            result = measure_target_probability(model, prompts, letter, hook_fn=hook_fn, hook_point=hook_point)

        letter_results["strength_results"][str(strength)] = {
            "mean_prob": result["mean_prob"],
            "success_rate": result["success_rate"],
            "n_target_tokens": result["n_target_tokens"],
        }

    # Random feature baseline (only at strength=5.0 for efficiency)
    random_fid = random_feature_ids[idx]
    random_direction = W_dec[random_fid]
    random_direction = random_direction / (random_direction.norm() + 1e-8)
    random_hook = steer_feature_hook(random_direction, 5.0)
    random_result = measure_target_probability(model, prompts, letter, hook_fn=random_hook, hook_point=hook_point)

    letter_results["random_baseline"] = {
        "feature_id": random_fid,
        "mean_prob": random_result["mean_prob"],
        "success_rate": random_result["success_rate"],
    }

    steering_results[letter] = letter_results

    # Report progress periodically
    if idx % 5 == 0:
        report_progress(4, 6, step=idx, total_steps=26, metric={
            "stage": "running_steering",
            "letters_complete": idx,
            "current_letter": letter,
        })

# Step 5: Compute summary statistics
print(f"\nComputing summary statistics...")
report_progress(5, 6, metric={"stage": "computing_summary"})

# Aggregate by absorption class
high_absorption_letters = [l for l, r in steering_results.items() if r["absorption_rate"] > 0.5]
low_absorption_letters = [l for l, r in steering_results.items() if r["absorption_rate"] <= 0.1]

summary = {
    "n_letters": len(steering_results),
    "n_high_absorption": len(high_absorption_letters),
    "n_low_absorption": len(low_absorption_letters),
    "steering_strengths": STEERING_STRENGTHS,
}

# Compute average success rates by absorption class for each strength
for strength in STEERING_STRENGTHS:
    strength_str = str(strength)

    high_probs = [steering_results[l]["strength_results"][strength_str]["mean_prob"]
                  for l in high_absorption_letters if strength_str in steering_results[l]["strength_results"]]
    low_probs = [steering_results[l]["strength_results"][strength_str]["mean_prob"]
                 for l in low_absorption_letters if strength_str in steering_results[l]["strength_results"]]

    high_success = [steering_results[l]["strength_results"][strength_str]["success_rate"]
                    for l in high_absorption_letters if strength_str in steering_results[l]["strength_results"]]
    low_success = [steering_results[l]["strength_results"][strength_str]["success_rate"]
                   for l in low_absorption_letters if strength_str in steering_results[l]["strength_results"]]

    summary[f"strength_{strength_str}"] = {
        "high_absorption": {
            "mean_prob": float(np.mean(high_probs)) if high_probs else 0.0,
            "mean_success_rate": float(np.mean(high_success)) if high_success else 0.0,
            "n": len(high_probs),
        },
        "low_absorption": {
            "mean_prob": float(np.mean(low_probs)) if low_probs else 0.0,
            "mean_success_rate": float(np.mean(low_success)) if low_success else 0.0,
            "n": len(low_probs),
        },
    }

# Random baseline
random_probs = [steering_results[l]["random_baseline"]["mean_prob"] for l in steering_results]
random_success = [steering_results[l]["random_baseline"]["success_rate"] for l in steering_results]
summary["random_baseline"] = {
    "mean_prob": float(np.mean(random_probs)),
    "mean_success_rate": float(np.mean(random_success)),
}

# Compute steering success at strength=5.0 (primary metric)
steering_success_5 = {}
for letter in steering_results:
    baseline_prob = steering_results[letter]["strength_results"]["0.0"]["mean_prob"]
    steered_prob = steering_results[letter]["strength_results"]["5.0"]["mean_prob"]
    improvement = steered_prob - baseline_prob
    steering_success_5[letter] = {
        "baseline_prob": baseline_prob,
        "steered_prob": steered_prob,
        "improvement": improvement,
        "absorption_rate": steering_results[letter]["absorption_rate"],
    }

# Correlation between absorption and steering improvement
absorption_rates = [steering_success_5[l]["absorption_rate"] for l in sorted(steering_success_5.keys())]
improvements = [steering_success_5[l]["improvement"] for l in sorted(steering_success_5.keys())]

from scipy import stats
pearson_r, pearson_p = stats.pearsonr(absorption_rates, improvements) if len(absorption_rates) > 2 else (0.0, 1.0)
spearman_r, spearman_p = stats.spearmanr(absorption_rates, improvements) if len(absorption_rates) > 2 else (0.0, 1.0)

summary["correlation"] = {
    "pearson_r": float(pearson_r),
    "pearson_p": float(pearson_p),
    "spearman_r": float(spearman_r),
    "spearman_p": float(spearman_p),
    "n": len(absorption_rates),
}

# Print results
print(f"\n{'='*60}")
print(f"STEERING EXPERIMENT RESULTS")
print(f"{'='*60}")
print(f"\nSuccess rates by absorption class (strength=5.0):")
print(f"  HIGH absorption: {summary['strength_5.0']['high_absorption']['mean_success_rate']:.3f}")
print(f"  LOW absorption:  {summary['strength_5.0']['low_absorption']['mean_success_rate']:.3f}")
print(f"  Random baseline: {summary['random_baseline']['mean_success_rate']:.3f}")

print(f"\nMean target probability by absorption class (strength=5.0):")
print(f"  HIGH absorption: {summary['strength_5.0']['high_absorption']['mean_prob']:.4f}")
print(f"  LOW absorption:  {summary['strength_5.0']['low_absorption']['mean_prob']:.4f}")
print(f"  Random baseline: {summary['random_baseline']['mean_prob']:.4f}")

print(f"\nCorrelation (absorption vs steering improvement):")
print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})")
print(f"  Spearman rho = {spearman_r:.3f} (p = {spearman_p:.3f})")

# Per-letter results table
print(f"\nPer-letter steering results (strength=5.0):")
print(f"{'Letter':<8} {'Absorp':<8} {'Baseline':<10} {'Steered':<10} {'Improve':<10} {'Random':<10}")
print(f"{'-'*60}")
for letter in sorted(steering_success_5.keys()):
    r = steering_success_5[letter]
    random_prob = steering_results[letter]["random_baseline"]["mean_prob"]
    print(f"{letter:<8} {r['absorption_rate']:<8.2f} {r['baseline_prob']:<10.4f} {r['steered_prob']:<10.4f} {r['improvement']:<10.4f} {random_prob:<10.4f}")

# Step 6: Save results
print(f"\nSaving results...")
report_progress(6, 6, metric={"stage": "saving_results"})

output = {
    "task_id": TASK_ID,
    "model": MODEL_NAME,
    "sae_source": "jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs",
    "layer": SAE_LAYER,
    "dictionary_size": SAE_DICT_SIZE,
    "seed": SEED,
    "device": DEVICE,
    "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    "timestamp": datetime.now().isoformat(),
    "steering_strengths": STEERING_STRENGTHS,
    "n_prompts_per_letter": 20,
    "summary": summary,
    "steering_results": steering_results,
    "steering_success_5": steering_success_5,
}

output_file = RESULTS_DIR / "steering_layer8_16k.json"
with open(output_file, "w") as f:
    json.dump(output, f, cls=NumpyEncoder, indent=2)

print(f"  Saved: {output_file}")

# Determine GO/NO-GO
random_success_rate = summary["random_baseline"]["mean_success_rate"]
low_success_rate = summary["strength_5.0"]["low_absorption"]["mean_success_rate"]

# Pass criteria from task_plan:
# - Random baseline steering success < 15%
# - At least 5 first-letter features show >30% steering success
n_above_30 = sum(1 for l, r in steering_success_5.items() if r["improvement"] > 0.30)

go_criteria = {
    "random_below_15pct": random_success_rate < 0.15,
    "at_least_5_above_30pct": n_above_30 >= 5,
    "completes_within_timeout": True,
}

overall_go = all(go_criteria.values())

print(f"\n{'='*60}")
print(f"PILOT GO/NO-GO ASSESSMENT")
print(f"{'='*60}")
print(f"Random baseline success rate: {random_success_rate:.3f} (target < 0.15): {'PASS' if go_criteria['random_below_15pct'] else 'FAIL'}")
print(f"Features with >30% improvement: {n_above_30} (target >= 5): {'PASS' if go_criteria['at_least_5_above_30pct'] else 'FAIL'}")
print(f"\nOverall: {'GO' if overall_go else 'NO-GO'}")

# Save summary
summary_text = f"""# Pilot Steering Results

## Configuration
- Model: {MODEL_NAME}
- Layer: {SAE_LAYER}
- Dictionary size: {SAE_DICT_SIZE}
- Steering strengths: {STEERING_STRENGTHS}
- Prompts per letter: 20

## Summary
- Letters tested: {len(steering_results)}
- HIGH absorption features: {len(high_absorption_letters)}
- LOW absorption features: {len(low_absorption_letters)}

## Results (strength=5.0)
| Metric | HIGH Absorption | LOW Absorption | Random Baseline |
|--------|-----------------|----------------|-----------------|
| Mean success rate | {summary['strength_5.0']['high_absorption']['mean_success_rate']:.3f} | {summary['strength_5.0']['low_absorption']['mean_success_rate']:.3f} | {summary['random_baseline']['mean_success_rate']:.3f} |
| Mean target prob | {summary['strength_5.0']['high_absorption']['mean_prob']:.4f} | {summary['strength_5.0']['low_absorption']['mean_prob']:.4f} | {summary['random_baseline']['mean_prob']:.4f} |

## Correlation
- Pearson r = {pearson_r:.3f} (p = {pearson_p:.3f})
- Spearman rho = {spearman_r:.3f} (p = {spearman_p:.3f})

## GO/NO-GO Criteria
- Random baseline < 15%: {'PASS' if go_criteria['random_below_15pct'] else 'FAIL'} ({random_success_rate:.3f})
- At least 5 features >30% improvement: {'PASS' if go_criteria['at_least_5_above_30pct'] else 'FAIL'} ({n_above_30})

## Overall: {'GO' if overall_go else 'NO-GO'}

## Notes
- Steering adds decoder direction directly to residual stream
- This bypasses the SAE encoder, so encoder-side absorption may not affect steering
- Random baseline is critical for interpreting steering success rates
"""

summary_file = RESULTS_DIR / "steering_layer8_16k_summary.md"
summary_file.write_text(summary_text)
print(f"  Saved: {summary_file}")

mark_done("success", f"Steering pilot complete. GO={overall_go}, n_above_30={n_above_30}, random_success={random_success_rate:.3f}")
print(f"\n[{TASK_ID}] Complete!")
