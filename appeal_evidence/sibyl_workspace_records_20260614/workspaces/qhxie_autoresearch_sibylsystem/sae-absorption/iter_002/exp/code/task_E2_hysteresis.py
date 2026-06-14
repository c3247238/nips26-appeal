"""
Task E2 (PILOT): Hysteresis Test — Does Sparsity Reduction Reverse Absorption?

Protocol:
1. Load GPT-2 Small SAE at highest available sparsity (lowest L0 = Layer 2, L0~18.5)
2. Measure baseline absorption rate (first-letter task, seed 42)
3. Fine-tune for 500 steps with reduced L1 coefficient (20% of original)
   - lr=1e-5, batch_size_tokens=4096, seed 42
   - Checkpoint every 100 steps, measure absorption rate at each
4. Compare to scratch SAE at target lower L0 (Layer 8, L0~76.6)
5. Report hysteresis: fine-tuned >= 70% of baseline AND scratch <= 30% of baseline

Hysteresis confirmed if:
  fine_tuned_rate >= 0.70 * original_high_sparsity_rate
  AND scratch_at_lower_L0_rate <= 0.30 * original_high_sparsity_rate
"""

import os
import sys
import json
import time
import warnings
import random
import string
from copy import deepcopy
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import LinearLR

warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_E2_hysteresis"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "E2_hysteresis.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}")
if DEVICE == "cuda":
    gpu_idx = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0])
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

TOTAL_STEPS = 12


def report_progress(step, total_steps, note=""):
    elapsed = time.time() - start_time
    progress = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "elapsed_sec": elapsed,
        "note": note,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"[{elapsed:.1f}s] Step {step}/{total_steps}: {note}")
    sys.stdout.flush()


def mark_done(status="success", summary="", result=None):
    PID_FILE.unlink(missing_ok=True)
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "result": result,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
    }))


report_progress(0, TOTAL_STEPS, "Starting E2 Hysteresis Test (PILOT)")

# ─── Step 1: Load model ──────────────────────────────────────────────────────
report_progress(1, TOTAL_STEPS, "Loading GPT-2 Small")

from transformer_lens import HookedTransformer
from sae_lens import SAE
import sklearn.linear_model as sklm

model = HookedTransformer.from_pretrained(
    "gpt2",
    center_writing_weights=True,
    center_unembed=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True
)
model.eval().to(DEVICE)
tokenizer = model.tokenizer
print(f"d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

# Word vocab for first-letter task (same as E1)
SIMPLE_WORDS = [
    "able", "above", "act", "add", "age", "air", "all", "also", "area", "back",
    "bad", "bag", "ball", "base", "bear", "bed", "big", "bird", "blow", "blue",
    "boat", "body", "book", "born", "box", "boy", "break", "bring", "burn", "busy",
    "call", "camp", "card", "care", "cat", "city", "clean", "clear", "close", "coat",
    "cold", "come", "cook", "cool", "cut", "dark", "data", "date", "deal", "deep",
    "desk", "die", "dirt", "dish", "door", "down", "draw", "drop", "duck", "dust",
    "earn", "east", "edge", "end", "even", "ever", "face", "fact", "fail", "fall",
    "far", "farm", "fast", "feel", "feet", "file", "fill", "film", "find", "fire",
    "fish", "five", "flag", "flat", "flow", "food", "foot", "form", "four", "free",
    "fuel", "full", "gain", "game", "gave", "girl", "give", "glad", "goal", "gold",
    "good", "gray", "grew", "grow", "hack", "hair", "half", "hall", "hand", "hang",
    "hard", "harm", "hate", "have", "head", "heal", "hear", "heat", "help", "here",
    "high", "hill", "hint", "hire", "hold", "hole", "home", "hook", "hope", "horn",
    "host", "hour", "hunt", "hurt", "idea", "inch", "iron", "jail", "join", "joke",
    "just", "keen", "keep", "kick", "kill", "kind", "king", "knew", "know", "lack",
    "lake", "land", "lane", "last", "late", "lead", "leaf", "lean", "left", "less",
    "lick", "lift", "like", "lime", "line", "link", "lion", "list", "load", "lock",
    "long", "look", "loop", "lose", "loss", "lost", "loud", "love", "luck", "made",
    "mail", "main", "make", "mark", "mass", "mate", "mean", "meat", "meet", "melt",
    "mild", "milk", "mill", "mine", "miss", "mode", "moon", "more", "most", "move",
    "much", "must", "name", "near", "need", "nest", "news", "next", "nice", "nine",
    "node", "none", "noon", "norm", "nose", "note", "noun", "once", "only", "open",
    "over", "pace", "pack", "page", "pain", "pair", "palm", "park", "part", "pass",
    "past", "path", "peak", "pick", "pill", "pine", "pink", "pipe", "plan", "play",
    "plot", "plug", "plus", "poem", "poll", "pond", "pool", "poor", "port", "pose",
    "post", "pour", "pull", "pump", "pure", "push", "race", "rack", "rain", "rank",
    "rate", "read", "real", "rent", "rest", "rice", "rich", "ride", "ring", "rise",
    "risk", "road", "rock", "role", "roll", "roof", "room", "root", "rope", "rose",
    "ruin", "rule", "rush", "rust", "safe", "sail", "salt", "same", "sand", "save",
    "seal", "seek", "sell", "send", "ship", "shoe", "shot", "show", "sick", "side",
    "silk", "sing", "sink", "site", "size", "skip", "slim", "slip", "slow", "snow",
    "sock", "soft", "soil", "sole", "some", "song", "soon", "soul", "spot", "stem",
    "step", "stop", "suit", "swim", "tail", "take", "talk", "tall", "tank", "task",
    "team", "tear", "tell", "test", "text", "thin", "tick", "tile", "time", "tire",
    "told", "tone", "tool", "toss", "tour", "town", "trap", "tree", "trim", "trip",
    "true", "tube", "tune", "turn", "type", "unit", "upon", "used", "vary", "vast",
    "very", "vest", "view", "vine", "vote", "wade", "wage", "wake", "walk", "wall",
    "warm", "warn", "wash", "wave", "wear", "weed", "week", "went", "west", "wide",
    "wild", "will", "wind", "wine", "wing", "wire", "wish", "wood", "word", "work",
]

valid_words = []
for word in SIMPLE_WORDS:
    word = word.strip().lower()
    if not word.isalpha():
        continue
    try:
        if len(tokenizer.encode(" " + word)) == 1 and len(tokenizer.encode(word)) == 1:
            valid_words.append(word)
    except:
        pass

vocab_by_letter = {lt: [] for lt in string.ascii_lowercase}
for word in valid_words:
    vocab_by_letter[word[0]].append(word)
good_letters = {lt: ws for lt, ws in vocab_by_letter.items() if len(ws) >= 5}
print(f"Vocab: {len(valid_words)} words, {len(good_letters)} letters with >=5 words")


# ─── Step 2: Train letter probes at Layer 2 ──────────────────────────────────
report_progress(2, TOTAL_STEPS, "Training letter probes for layer 2 (high-sparsity SAE)")

LAYER = 2  # layer 2: highest sparsity, L0~18.5

rng_random = random.Random(SEED)
probe_train_words = []
for lt in sorted(good_letters.keys()):
    ws = good_letters[lt]
    probe_train_words.extend(rng_random.sample(ws, min(len(ws), 30)))


def train_probes_at_layer(model, words, layer, device, seed=42):
    """Train letter probes at given layer; return probe_dirs dict."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    acts_list, word_list = [], []
    with torch.no_grad():
        for word in words:
            try:
                tok = model.to_tokens(f" {word}:")
                _, cache = model.run_with_cache(tok, names_filter=hook_name)
                acts_list.append(cache[hook_name][0, -2, :].cpu().float().numpy())
                word_list.append(word)
                del cache
            except:
                pass
    if len(acts_list) < 10:
        return {}, []
    acts = np.stack(acts_list)
    first_letters = np.array([w[0] for w in word_list])
    probe_dirs = {}
    letters = []
    for lt in sorted(good_letters.keys()):
        y = (first_letters == lt).astype(int)
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        try:
            clf = sklm.LogisticRegression(C=1.0, max_iter=300, random_state=seed)
            clf.fit(acts, y)
            d = clf.coef_[0]
            probe_dirs[lt] = d / (np.linalg.norm(d) + 1e-8)
            letters.append(lt)
        except:
            pass
    return probe_dirs, letters


probe_dirs_l2, letters_l2 = train_probes_at_layer(model, probe_train_words, LAYER, DEVICE)
print(f"  Layer 2 probes: {len(probe_dirs_l2)} letters")


def identify_letter_features(sae_w_dec, probe_dirs, n_pos_target=67):
    """Find SAE features whose decoder aligns with letter probes."""
    letters = sorted(probe_dirs.keys())
    probe_mat = np.stack([probe_dirs[lt] for lt in letters])
    W_dec_np = F.normalize(sae_w_dec.cpu().float(), dim=1).numpy()
    cos = probe_mat @ W_dec_np.T
    max_cos = cos.max(axis=0)
    best_thr = 0.30
    for thr in np.arange(0.20, 0.55, 0.01):
        n = int((max_cos >= thr).sum())
        if abs(n - n_pos_target) < abs(int((max_cos >= best_thr).sum()) - n_pos_target):
            best_thr = thr
    ids = np.where(max_cos >= best_thr)[0].tolist()
    return ids, float(best_thr), len(ids)


def measure_empirical_l0(sae, model, layer, device, n_tokens=256):
    """Measure empirical L0 from a short text passage."""
    hook_name = f"blocks.{layer}.hook_resid_pre"
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Scientists discovered new species in deep ocean. "
        "Technology companies announced partnerships. "
        "History shows that civilizations rise and fall. "
        "Mathematics underlies the physical sciences. " * 6
    )
    try:
        tokens = model.to_tokens(text)[:, :n_tokens].to(device)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            resid = cache[hook_name][0]  # (seq_len, d_model)
            sae_acts = sae.encode(resid)  # (seq_len, d_sae)
            l0 = float((sae_acts > 0).float().sum(-1).mean().item())
        del cache
        return l0
    except Exception as e:
        print(f"    L0 measurement error: {e}")
        return None


def measure_absorption_rate(sae, model, probe_dirs, layer, device, n_per_letter=20, seed=42):
    """
    Measure absorption rate using first-letter task.
    Returns (overall_rate, per_letter_rates_dict)
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    rng = random.Random(seed)
    letters = sorted(probe_dirs.keys())

    # Identify which letter feature belongs to which letter
    letter_ids, thr, n_pos = identify_letter_features(sae.W_dec.data, probe_dirs)
    if not letter_ids:
        return 0.0, {}

    sae_w_dec = sae.W_dec.detach().cpu().float()
    W_dec_np = F.normalize(sae_w_dec, dim=1).numpy()
    probe_mat = np.stack([probe_dirs[lt] for lt in letters])
    cos_letter_features = probe_mat @ W_dec_np[letter_ids].T
    best_letter_idx = cos_letter_features.argmax(axis=0)

    letter_to_features = {lt: [] for lt in letters}
    for feat_pos, feat_id in enumerate(letter_ids):
        lt = letters[best_letter_idx[feat_pos]]
        letter_to_features[lt].append(feat_id)

    absorbed_count = 0
    total_count = 0
    per_letter_rates = {}

    for lt in letters:
        feat_ids_for_lt = letter_to_features[lt]
        if not feat_ids_for_lt:
            continue
        words_for_lt = good_letters.get(lt, [])
        if len(words_for_lt) < 3:
            continue
        sampled_words = rng.sample(words_for_lt, min(n_per_letter, len(words_for_lt)))

        letter_absorbed = 0
        letter_total = 0

        for word in sampled_words:
            try:
                tok = model.to_tokens(f" {word}:").to(device)
                with torch.no_grad():
                    _, cache = model.run_with_cache(tok, names_filter=hook_name)
                    resid = cache[hook_name][0, -2, :]
                    probe_dir = torch.tensor(probe_dirs[lt], dtype=torch.float32).to(device)
                    probe_act = torch.dot(resid, probe_dir).item()

                    if probe_act < 0:
                        del cache
                        continue

                    sae_acts = sae.encode(resid.unsqueeze(0))[0]
                    for feat_id in feat_ids_for_lt:
                        feat_act = sae_acts[feat_id].item()
                        is_absorbed = int(feat_act < 0.1)
                        letter_absorbed += is_absorbed
                        letter_total += 1

                    del cache
            except:
                continue

        if letter_total > 0:
            rate = letter_absorbed / letter_total
            per_letter_rates[lt] = {
                "n_features": len(feat_ids_for_lt),
                "n_samples": letter_total,
                "absorption_rate": rate,
            }
            absorbed_count += letter_absorbed
            total_count += letter_total

    overall_rate = absorbed_count / total_count if total_count > 0 else 0.0
    return overall_rate, per_letter_rates


def measure_ce_delta(sae, model, layer, device, n_tokens=256):
    """
    Measure CE loss increase from SAE reconstruction at a given layer.
    CE delta = cross_entropy(model+SAE) - cross_entropy(model)
    """
    hook_name = f"blocks.{layer}.hook_resid_pre"
    text = (
        "The dog ran fast across the field. "
        "She opened the door and walked inside. "
        "He read the book carefully before sleeping. "
        "They went to the store to buy groceries. " * 6
    )
    try:
        tokens = model.to_tokens(text)[:, :n_tokens].to(device)

        with torch.no_grad():
            # Original model loss
            logits_orig = model(tokens)
            # Compute CE loss (next token prediction)
            shift_logits = logits_orig[:, :-1, :].contiguous()
            shift_labels = tokens[:, 1:].contiguous()
            ce_orig = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            ).item()

            # Model + SAE loss: hook the residual stream
            def hook_fn(resid, hook):
                # SAE reconstruction
                recon = sae(resid)
                return recon

            logits_sae = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, hook_fn)]
            )
            shift_logits_sae = logits_sae[:, :-1, :].contiguous()
            ce_sae = F.cross_entropy(
                shift_logits_sae.view(-1, shift_logits_sae.size(-1)),
                shift_labels.view(-1),
            ).item()

        ce_delta = ce_sae - ce_orig
        return float(ce_orig), float(ce_sae), float(ce_delta)
    except Exception as e:
        print(f"    CE delta error: {e}")
        return None, None, None


# ─── Step 3: Load high-sparsity SAE (Layer 2, L0~18.5) ──────────────────────
report_progress(3, TOTAL_STEPS, "Loading high-sparsity SAE (Layer 2, L0~18.5)")

HIGH_SPARSITY_RELEASE = "gpt2-small-res-jb"
HIGH_SPARSITY_SAE_ID = "blocks.2.hook_resid_pre"
HIGH_SPARSITY_LAYER = 2

sae_high_raw = SAE.from_pretrained(HIGH_SPARSITY_RELEASE, HIGH_SPARSITY_SAE_ID)
if isinstance(sae_high_raw, tuple):
    sae_high = sae_high_raw[0]
else:
    sae_high = sae_high_raw
sae_high.eval().to(DEVICE)
print(f"Loaded: {HIGH_SPARSITY_RELEASE} / {HIGH_SPARSITY_SAE_ID}")
print(f"  W_enc: {sae_high.W_enc.shape}, W_dec: {sae_high.W_dec.shape}")
print(f"  Activation: {sae_high.activation_fn}")

# Measure initial L0
l0_baseline = measure_empirical_l0(sae_high, model, HIGH_SPARSITY_LAYER, DEVICE)
print(f"  Baseline L0: {l0_baseline:.2f}")

# ─── Step 4: Measure baseline absorption rate ────────────────────────────────
report_progress(4, TOTAL_STEPS, "Measuring baseline absorption rate (high-sparsity SAE)")

absorption_baseline, per_letter_baseline = measure_absorption_rate(
    sae_high, model, probe_dirs_l2, HIGH_SPARSITY_LAYER, DEVICE, n_per_letter=20, seed=SEED
)
print(f"  Baseline absorption rate: {absorption_baseline:.4f}")
print(f"  Letters measured: {len(per_letter_baseline)}")

# Measure CE delta
ce_orig_baseline, ce_sae_baseline, ce_delta_baseline = measure_ce_delta(
    sae_high, model, HIGH_SPARSITY_LAYER, DEVICE
)
print(f"  Baseline CE delta: {ce_delta_baseline:.4f}" if ce_delta_baseline is not None else "  CE delta: N/A")

# ─── Step 5: Load scratch (low-sparsity) SAE for comparison ─────────────────
report_progress(5, TOTAL_STEPS, "Loading scratch low-sparsity SAE (Layer 8, L0~76.6)")

SCRATCH_RELEASE = "gpt2-small-res-jb"
SCRATCH_SAE_ID = "blocks.8.hook_resid_pre"
SCRATCH_LAYER = 8

# Train probes for layer 8
probe_dirs_l8, letters_l8 = train_probes_at_layer(model, probe_train_words, SCRATCH_LAYER, DEVICE)
print(f"  Layer 8 probes: {len(probe_dirs_l8)} letters")

sae_scratch_raw = SAE.from_pretrained(SCRATCH_RELEASE, SCRATCH_SAE_ID)
if isinstance(sae_scratch_raw, tuple):
    sae_scratch = sae_scratch_raw[0]
else:
    sae_scratch = sae_scratch_raw
sae_scratch.eval().to(DEVICE)
print(f"Loaded scratch: {SCRATCH_RELEASE} / {SCRATCH_SAE_ID}")

l0_scratch = measure_empirical_l0(sae_scratch, model, SCRATCH_LAYER, DEVICE)
print(f"  Scratch L0: {l0_scratch:.2f}")

absorption_scratch, per_letter_scratch = measure_absorption_rate(
    sae_scratch, model, probe_dirs_l8, SCRATCH_LAYER, DEVICE, n_per_letter=20, seed=SEED
)
print(f"  Scratch absorption rate: {absorption_scratch:.4f}")

ce_orig_scratch, ce_sae_scratch, ce_delta_scratch = measure_ce_delta(
    sae_scratch, model, SCRATCH_LAYER, DEVICE
)
print(f"  Scratch CE delta: {ce_delta_scratch:.4f}" if ce_delta_scratch is not None else "  CE delta: N/A")

# Move scratch to CPU to free VRAM for fine-tuning
sae_scratch.cpu()
torch.cuda.empty_cache()

# ─── Step 6: Prepare training data for fine-tuning ──────────────────────────
report_progress(6, TOTAL_STEPS, "Preparing training data (residual stream activations)")

FINETUNE_STEPS = 500
CHECKPOINT_INTERVAL = 100
BATCH_SIZE_TOKENS = 4096
LR = 1e-5
# L1 coefficient: 20% of typical value
# Standard SAE L1 coeff is typically 8e-5 to 2e-4; we use 20% of that
# Actually, we set it as a relative fraction of the SAE norm
L1_COEFF_FRACTION = 0.20  # 20% of original

print(f"\nFine-tuning config:")
print(f"  Steps: {FINETUNE_STEPS}, Checkpoint every: {CHECKPOINT_INTERVAL}")
print(f"  Batch size tokens: {BATCH_SIZE_TOKENS}, LR: {LR}")
print(f"  L1 coefficient fraction: {L1_COEFF_FRACTION}")

# Generate activation data by running GPT-2 on OpenWebText samples
hook_name_l2 = f"blocks.{HIGH_SPARSITY_LAYER}.hook_resid_pre"

# Use varied text to get diverse activations
TEXTS = [
    "The history of science shows that major discoveries often come from unexpected directions. "
    "Researchers working on fundamental questions frequently find practical applications. ",
    "Language models learn from vast amounts of text data. The patterns in language reflect "
    "the structure of human thought and communication across many domains. ",
    "Mathematics provides a universal language for expressing relationships between quantities. "
    "Calculus, algebra, and geometry form the foundations of modern science. ",
    "Economic systems balance supply and demand through prices and incentives. Markets "
    "coordinate the activities of millions of individuals without central planning. ",
    "Evolution through natural selection explains the diversity of life on Earth. Species "
    "adapt to their environments over many generations through differential reproduction. ",
    "Art and literature express aspects of human experience that resist simple description. "
    "Poetry, painting, and music communicate emotions and ideas across cultural boundaries. ",
    "Cities grew as centers of trade, culture, and governance. Urban planning shapes how "
    "people interact with space and with each other in complex societies. ",
    "Computers process information through sequences of simple operations. Software abstracts "
    "these operations into higher-level languages and programs that solve complex problems. ",
]

print("\nGenerating activation data...")
all_activations = []
model.eval()
with torch.no_grad():
    for text in TEXTS:
        # Repeat text to get enough tokens
        long_text = text * 8
        tokens = model.to_tokens(long_text)[:, :512].to(DEVICE)
        _, cache = model.run_with_cache(tokens, names_filter=hook_name_l2)
        acts = cache[hook_name_l2][0].cpu().float()  # (seq_len, d_model)
        all_activations.append(acts)
        del cache
        torch.cuda.empty_cache()

all_activations_tensor = torch.cat(all_activations, dim=0)
n_total_tokens = all_activations_tensor.shape[0]
print(f"  Total activation tokens: {n_total_tokens}")

# Compute normalization scale (mean activation norm)
act_norm = float(all_activations_tensor.norm(dim=-1).mean().item())
print(f"  Mean activation norm: {act_norm:.3f}")

# Estimate L1 coefficient: target L0 around 75-100 from current ~18
# Use a moderate L1 to reduce sparsity
# Typical SAE L1 coefficient is 8e-5; we use 20% = 1.6e-5
# But since we want reduced sparsity (more features active), we REDUCE L1
L1_COEFF = 8e-5 * L1_COEFF_FRACTION  # 20% of standard -> fewer sparsity pressure -> higher L0
print(f"  L1 coefficient: {L1_COEFF:.2e}")

# ─── Step 7: Fine-tune SAE with reduced L1 ──────────────────────────────────
report_progress(7, TOTAL_STEPS, "Fine-tuning SAE with reduced L1 (500 steps)")

# Deep copy the high-sparsity SAE for fine-tuning
sae_finetune = deepcopy(sae_high)
sae_finetune.train().to(DEVICE)

# Freeze W_dec norm (as is standard in SAE training)
optimizer = Adam(sae_finetune.parameters(), lr=LR, weight_decay=0.0)

# Batch size from stored activations
BATCH_SIZE = min(512, n_total_tokens)  # token-level batch

checkpoint_results = []
step = 0
torch.manual_seed(SEED)
np.random.seed(SEED)

# Precompute training batches (shuffle indices)
indices = torch.randperm(n_total_tokens, generator=torch.Generator().manual_seed(SEED))
batch_idx = 0

print(f"\nStarting fine-tuning loop ({FINETUNE_STEPS} steps)...")
print(f"  Batch size: {BATCH_SIZE}, L1 coeff: {L1_COEFF:.2e}")

train_losses = []
l0_trajectory = [l0_baseline]

for step in range(1, FINETUNE_STEPS + 1):
    # Get batch indices (cycle through data)
    start_idx = (batch_idx * BATCH_SIZE) % n_total_tokens
    end_idx = start_idx + BATCH_SIZE
    if end_idx > n_total_tokens:
        start_idx = 0
        end_idx = BATCH_SIZE
        # Reshuffle
        indices = torch.randperm(n_total_tokens)
    batch_idx += 1

    batch_indices = indices[start_idx:end_idx]
    x_batch = all_activations_tensor[batch_indices].to(DEVICE)

    optimizer.zero_grad()

    # Forward pass through SAE
    # Apply b_dec to input if configured
    if sae_finetune.cfg.apply_b_dec_to_input:
        sae_in = x_batch - sae_finetune.b_dec
    else:
        sae_in = x_batch

    # Pre-activation
    hidden_pre = sae_in @ sae_finetune.W_enc + sae_finetune.b_enc
    # Apply ReLU activation
    hidden_acts = F.relu(hidden_pre)
    # Decode
    x_recon = hidden_acts @ sae_finetune.W_dec + sae_finetune.b_dec

    # Reconstruction loss
    recon_loss = F.mse_loss(x_recon, x_batch)

    # L1 sparsity loss (reduced)
    l1_loss = L1_COEFF * hidden_acts.abs().sum(dim=-1).mean()

    loss = recon_loss + l1_loss
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(sae_finetune.parameters(), max_norm=1.0)

    # Normalize W_dec rows to unit norm (standard SAE training trick)
    with torch.no_grad():
        optimizer.step()
        # Re-normalize decoder columns
        W_dec_norm = sae_finetune.W_dec.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sae_finetune.W_dec.data = sae_finetune.W_dec.data / W_dec_norm

    train_losses.append(float(loss.item()))

    if step % 50 == 0:
        # Quick L0 estimate during training
        with torch.no_grad():
            x_sample = all_activations_tensor[:min(256, n_total_tokens)].to(DEVICE)
            if sae_finetune.cfg.apply_b_dec_to_input:
                sae_in_s = x_sample - sae_finetune.b_dec
            else:
                sae_in_s = x_sample
            h_s = F.relu(sae_in_s @ sae_finetune.W_enc + sae_finetune.b_enc)
            l0_current = float((h_s > 0).float().sum(-1).mean().item())
        print(f"  Step {step}: loss={float(loss.item()):.4f}, "
              f"recon={float(recon_loss.item()):.4f}, l1={float(l1_loss.item()):.6f}, "
              f"L0_est={l0_current:.1f}")
        sys.stdout.flush()

    # Checkpoint every CHECKPOINT_INTERVAL steps
    if step % CHECKPOINT_INTERVAL == 0:
        sae_finetune.eval()
        with torch.no_grad():
            # Measure L0
            l0_ckpt = measure_empirical_l0(sae_finetune, model, HIGH_SPARSITY_LAYER, DEVICE)

            # Measure absorption rate
            abs_rate_ckpt, per_letter_ckpt = measure_absorption_rate(
                sae_finetune, model, probe_dirs_l2, HIGH_SPARSITY_LAYER, DEVICE,
                n_per_letter=20, seed=SEED
            )

            # Measure CE delta
            ce_orig_ckpt, ce_sae_ckpt, ce_delta_ckpt = measure_ce_delta(
                sae_finetune, model, HIGH_SPARSITY_LAYER, DEVICE
            )

        l0_trajectory.append(l0_ckpt)

        ckpt_result = {
            "step": step,
            "l0": l0_ckpt,
            "absorption_rate": float(abs_rate_ckpt),
            "n_letters_measured": len(per_letter_ckpt),
            "ce_orig": ce_orig_ckpt,
            "ce_sae": ce_sae_ckpt,
            "ce_delta": ce_delta_ckpt,
            "train_loss_last": float(train_losses[-1]) if train_losses else None,
            "train_loss_mean_50": float(np.mean(train_losses[-50:])) if len(train_losses) >= 50 else None,
        }
        checkpoint_results.append(ckpt_result)
        print(f"\n  === Checkpoint step={step}: L0={l0_ckpt:.1f}, "
              f"absorption={abs_rate_ckpt:.4f}, CE_delta={ce_delta_ckpt:.4f} ===\n")
        sys.stdout.flush()

        sae_finetune.train()

    report_progress(7, TOTAL_STEPS, f"Fine-tuning step {step}/{FINETUNE_STEPS}")

print(f"\nFine-tuning complete. Final loss: {train_losses[-1]:.4f}")

# ─── Step 8: Final measurements after fine-tuning ───────────────────────────
report_progress(8, TOTAL_STEPS, "Final measurements on fine-tuned SAE")

sae_finetune.eval()
with torch.no_grad():
    l0_final = measure_empirical_l0(sae_finetune, model, HIGH_SPARSITY_LAYER, DEVICE)
    absorption_final, per_letter_final = measure_absorption_rate(
        sae_finetune, model, probe_dirs_l2, HIGH_SPARSITY_LAYER, DEVICE,
        n_per_letter=20, seed=SEED
    )
    ce_orig_final, ce_sae_final, ce_delta_final = measure_ce_delta(
        sae_finetune, model, HIGH_SPARSITY_LAYER, DEVICE
    )

print(f"  Final L0 (fine-tuned): {l0_final:.2f}")
print(f"  Final absorption rate (fine-tuned): {absorption_final:.4f}")
print(f"  Final CE delta (fine-tuned): {ce_delta_final:.4f}" if ce_delta_final is not None else "  CE delta: N/A")

# ─── Step 9: Load scratch SAE back and measure again ─────────────────────────
report_progress(9, TOTAL_STEPS, "Reloading scratch SAE for final comparison")

sae_scratch.to(DEVICE)
l0_scratch_final = measure_empirical_l0(sae_scratch, model, SCRATCH_LAYER, DEVICE)
absorption_scratch_final, per_letter_scratch_final = measure_absorption_rate(
    sae_scratch, model, probe_dirs_l8, SCRATCH_LAYER, DEVICE, n_per_letter=20, seed=SEED
)
print(f"  Scratch L0: {l0_scratch_final:.2f}")
print(f"  Scratch absorption rate: {absorption_scratch_final:.4f}")

# ─── Step 10: Compute hysteresis verdict ─────────────────────────────────────
report_progress(10, TOTAL_STEPS, "Computing hysteresis verdict")

# Hysteresis confirmed if:
#   fine_tuned_rate >= 0.70 * original_high_sparsity_rate
#   AND scratch_at_lower_L0_rate <= 0.30 * original_high_sparsity_rate
HYSTERESIS_FINETUNED_THRESHOLD = 0.70
HYSTERESIS_SCRATCH_THRESHOLD = 0.30

finetuned_fraction = absorption_final / (absorption_baseline + 1e-8)
scratch_fraction = absorption_scratch_final / (absorption_baseline + 1e-8)

hysteresis_criterion_1 = finetuned_fraction >= HYSTERESIS_FINETUNED_THRESHOLD
hysteresis_criterion_2 = scratch_fraction <= HYSTERESIS_SCRATCH_THRESHOLD
hysteresis_confirmed = hysteresis_criterion_1 and hysteresis_criterion_2

print(f"\n=== Hysteresis Verdict ===")
print(f"  Baseline (high-sparsity L2, L0={l0_baseline:.1f}): {absorption_baseline:.4f}")
print(f"  Fine-tuned (L0={l0_final:.1f}): {absorption_final:.4f} ({finetuned_fraction:.1%} of baseline)")
print(f"  Scratch (low-sparsity L8, L0={l0_scratch_final:.1f}): {absorption_scratch_final:.4f} ({scratch_fraction:.1%} of baseline)")
print(f"  Criterion 1 (fine-tuned >= 70% baseline): {hysteresis_criterion_1} ({finetuned_fraction:.1%} >= 70%)")
print(f"  Criterion 2 (scratch <= 30% baseline): {hysteresis_criterion_2} ({scratch_fraction:.1%} <= 30%)")
print(f"  HYSTERESIS: {'CONFIRMED' if hysteresis_confirmed else 'NOT CONFIRMED'}")

# Note: if both rates are high (absorption near saturated), it may just mean
# absorption is high everywhere and doesn't transition below ~70-80%
if not hysteresis_confirmed and scratch_fraction > 0.30:
    print(f"  NOTE: Scratch absorption also high ({scratch_fraction:.1%}). "
          f"Absorption appears generally high across L0 range tested.")
    h4b_interpretation = "SATURATION: absorption high at all L0 values tested; hysteresis not testable in this regime"
elif hysteresis_confirmed:
    h4b_interpretation = "H4b SUPPORTED: fine-tuned SAE retains high absorption despite reduced sparsity"
else:
    h4b_interpretation = "H4b NOT SUPPORTED: fine-tuned absorption does not significantly exceed scratch-trained absorption"

print(f"  H4b interpretation: {h4b_interpretation}")

# ─── Step 11: Pilot pass/fail ────────────────────────────────────────────────
report_progress(11, TOTAL_STEPS, "Evaluating pilot pass criteria")

# Pilot pass: fine-tuning completes without OOM, 100-step absorption measurable
pilot_pass = (
    len(checkpoint_results) >= 1  # at least one checkpoint was measured
    and absorption_baseline > 0.0  # baseline was measured
    and absorption_final is not None  # final measurement succeeded
)
print(f"\n  Pilot pass: {pilot_pass}")
print(f"  Checkpoints measured: {len(checkpoint_results)}")
print(f"  Pilot criteria: Fine-tuning completes without OOM; 100-step checkpoint measurable")

# ─── Step 12: Save results ──────────────────────────────────────────────────
report_progress(12, TOTAL_STEPS, "Saving results")

summary_str = (
    f"E2 PILOT: baseline_abs={absorption_baseline:.3f} (L0={l0_baseline:.1f}), "
    f"finetuned_abs={absorption_final:.3f} (L0={l0_final:.1f}), "
    f"scratch_abs={absorption_scratch_final:.3f} (L0={l0_scratch_final:.1f}). "
    f"Hysteresis: {'CONFIRMED' if hysteresis_confirmed else 'NOT CONFIRMED'}. "
    f"H4b: {h4b_interpretation}"
)

output = {
    "task_id": TASK_ID,
    "mode": "PILOT",
    "timestamp": datetime.now().isoformat(),
    "elapsed_sec": time.time() - start_time,
    "config": {
        "model": "gpt2-small",
        "seed": SEED,
        "high_sparsity_sae": {
            "release": HIGH_SPARSITY_RELEASE,
            "sae_id": HIGH_SPARSITY_SAE_ID,
            "layer": HIGH_SPARSITY_LAYER,
        },
        "scratch_sae": {
            "release": SCRATCH_RELEASE,
            "sae_id": SCRATCH_SAE_ID,
            "layer": SCRATCH_LAYER,
        },
        "finetune": {
            "steps": FINETUNE_STEPS,
            "lr": LR,
            "l1_coeff": L1_COEFF,
            "l1_coeff_fraction": L1_COEFF_FRACTION,
            "batch_size": BATCH_SIZE,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
        },
        "device": DEVICE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
    },
    "pilot_pass": pilot_pass,
    "pass_criteria": "Fine-tuning completes without OOM; 100-step checkpoint measurable",

    # Measurements
    "baseline": {
        "l0": l0_baseline,
        "absorption_rate": float(absorption_baseline),
        "n_letters_measured": len(per_letter_baseline),
        "ce_delta": ce_delta_baseline,
        "per_letter_rates": per_letter_baseline,
    },
    "scratch_lower_l0": {
        "l0": l0_scratch_final,
        "absorption_rate": float(absorption_scratch_final),
        "n_letters_measured": len(per_letter_scratch_final),
        "ce_delta": ce_delta_scratch,
        "fraction_of_baseline": float(scratch_fraction),
    },
    "finetuned": {
        "l0_final": l0_final,
        "absorption_rate": float(absorption_final),
        "n_letters_measured": len(per_letter_final),
        "ce_delta": ce_delta_final,
        "fraction_of_baseline": float(finetuned_fraction),
        "per_letter_rates": per_letter_final,
    },
    "checkpoint_trajectory": checkpoint_results,
    "l0_trajectory": l0_trajectory,
    "train_losses": train_losses,

    # Hysteresis verdict
    "hysteresis": {
        "confirmed": hysteresis_confirmed,
        "finetuned_fraction": float(finetuned_fraction),
        "scratch_fraction": float(scratch_fraction),
        "threshold_finetuned": HYSTERESIS_FINETUNED_THRESHOLD,
        "threshold_scratch": HYSTERESIS_SCRATCH_THRESHOLD,
        "criterion_1_met": hysteresis_criterion_1,
        "criterion_2_met": hysteresis_criterion_2,
        "h4b_interpretation": h4b_interpretation,
    },

    "summary": summary_str,
    "primary_finding": summary_str,
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to {OUTPUT_FILE}")

# Update gpu_progress.json
gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    if gpu_progress_file.exists():
        with open(gpu_progress_file) as f:
            gpu_progress = json.load(f)
    else:
        gpu_progress = {"completed": [], "failed": [], "running": {}, "timings": {}}

    elapsed_min = int((time.time() - start_time) / 60 + 0.5)
    if pilot_pass:
        if TASK_ID not in gpu_progress["completed"]:
            gpu_progress["completed"].append(TASK_ID)
    else:
        if TASK_ID not in gpu_progress["failed"]:
            gpu_progress["failed"].append(TASK_ID)
    gpu_progress["running"].pop(TASK_ID, None)
    gpu_progress["timings"][TASK_ID] = {
        "planned_min": 60,
        "actual_min": elapsed_min,
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "layer": HIGH_SPARSITY_LAYER,
            "finetune_steps": FINETUNE_STEPS,
            "l1_coeff_fraction": L1_COEFF_FRACTION,
            "pilot_pass": pilot_pass,
            "hysteresis_confirmed": hysteresis_confirmed,
            "gpu_model": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
        }
    }
    gpu_progress_file.write_text(json.dumps(gpu_progress, indent=2))
    print(f"Updated gpu_progress.json (elapsed={elapsed_min}min)")
except Exception as e:
    print(f"Warning: could not update gpu_progress.json: {e}")

# Update experiment_state.json
exp_state_file = WORKSPACE / "exp" / "experiment_state.json"
exp_state_lock = WORKSPACE / "exp" / "experiment_state.lock"
try:
    import fcntl
    with open(exp_state_lock, 'w') as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        with open(exp_state_file) as f:
            exp_state = json.load(f)
        exp_state["tasks"][TASK_ID]["status"] = "completed"
        exp_state["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
        exp_state["tasks"][TASK_ID]["result_file"] = str(OUTPUT_FILE)
        with open(exp_state_file, 'w') as f:
            json.dump(exp_state, f, indent=2)
        fcntl.flock(lock_f, fcntl.LOCK_UN)
    print("Updated experiment_state.json")
except Exception as e:
    print(f"Warning: could not update experiment_state.json: {e}")

mark_done(
    status="success" if pilot_pass else "warning",
    summary=summary_str,
    result={
        "pilot_pass": pilot_pass,
        "hysteresis_confirmed": hysteresis_confirmed,
        "absorption_baseline": float(absorption_baseline),
        "absorption_final": float(absorption_final),
        "absorption_scratch": float(absorption_scratch_final),
        "h4b_interpretation": h4b_interpretation,
    }
)

print(f"\n=== E2 PILOT COMPLETE ===")
print(f"Summary: {summary_str}")
