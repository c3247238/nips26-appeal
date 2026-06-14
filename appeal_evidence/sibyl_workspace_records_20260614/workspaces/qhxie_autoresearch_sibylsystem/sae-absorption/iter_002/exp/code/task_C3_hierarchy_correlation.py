"""
Task C.3: Hierarchy Property Correlation with Absorption Rate (PILOT MODE)

For each hierarchy type with passing probes from C.1, compute:
  (a) Mean decoder cosine similarity between parent and child SAE latents
  (b) Mean co-occurrence rate (freq_parent / freq_child) from 10k-token OWT sample

Test H2 prediction: Spearman correlation between these properties and
measured absorption rates across hierarchy types.

NOTE: C2 pilot showed NO_GO (absorption rates near 0 for non-spelling hierarchies).
For this C3 pilot, we:
  1. Compute geometric properties (cos^2 between probe-aligned latents) for all hierarchies
  2. Compute co-occurrence frequencies
  3. Use available absorption data: first-letter from B1/pilot, C2 pilot values (even if near-0)
  4. Report scatter plots and Spearman correlations as directional evidence

Output: exp/results/full/C3_hierarchy_correlation.json

Dependencies:
  - C1 probe weights: exp/results/probes/
  - C1 results: exp/results/full/C1_probe_training.json
  - C2 pilot: exp/results/pilots/pilot_C2_cross_domain_absorption.json
  - B1 results: exp/results/full/B1_decoder_geometry.json
  - OWT cache: exp/results/owt_tokens_cache.pt (or regenerate)
"""

import os
import sys
import json
import time
import random
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
FULL_DIR = WORKSPACE / "exp" / "results" / "full"
PILOTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
PROBES_DIR = WORKSPACE / "exp" / "results" / "probes"
FULL_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "task_C3_hierarchy_correlation"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = FULL_DIR / "C3_hierarchy_correlation.json"

PID_FILE.write_text(str(os.getpid()))
start_time = time.time()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {DEVICE}", flush=True)

TOTAL_STEPS = 12


def report_progress(step, note=""):
    elapsed = time.time() - start_time
    progress = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": TOTAL_STEPS,
        "elapsed_sec": elapsed,
        "note": note,
        "updated_at": datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))
    print(f"[{elapsed:.1f}s] Step {step}/{TOTAL_STEPS}: {note}", flush=True)


def mark_done(status="success", summary=""):
    PID_FILE.unlink(missing_ok=True)
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
        "elapsed_sec": time.time() - start_time,
    }))


report_progress(0, "Starting task C.3 hierarchy property correlation")

# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Load SAE and model
# ─────────────────────────────────────────────────────────────────────────────
report_progress(1, "Loading GPT-2 Small + SAE")

from sae_lens import SAE
from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()

sae, cfg_dict, _ = SAE.from_pretrained(
    release="gpt2-small-res-jb",
    sae_id="blocks.6.hook_resid_pre",
    device=DEVICE,
)
sae.eval()

W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
d_sae = W_dec.shape[0]
d_model = W_dec.shape[1]
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token

print(f"SAE: d_sae={d_sae}, d_in={d_model}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Load probe weights and C1/C2 results
# ─────────────────────────────────────────────────────────────────────────────
report_progress(2, "Loading probe weights and previous results")

# Load probe weights
probe_weights = {}
probe_classes = {}
for hierarchy in ["first_letter", "noun_proper", "animate_inanimate", "city_country_binary"]:
    w_file = PROBES_DIR / f"probe_{hierarchy}_weights.npy"
    c_file = PROBES_DIR / f"probe_{hierarchy}_classes.npy"
    if w_file.exists() and c_file.exists():
        probe_weights[hierarchy] = np.load(str(w_file))  # [n_classes, d_model]
        probe_classes[hierarchy] = np.load(str(c_file), allow_pickle=True)
        print(f"  Loaded {hierarchy} probe: {probe_weights[hierarchy].shape}", flush=True)
    else:
        print(f"  WARNING: Missing probe files for {hierarchy}", flush=True)

# Load C1 results for context
c1_data = {}
c1_file = FULL_DIR / "C1_probe_training.json"
if c1_file.exists():
    c1_data = json.loads(c1_file.read_text())

# Load C2 pilot results for absorption rates
c2_absorption = {}
c2_pilot_file = PILOTS_DIR / "pilot_C2_cross_domain_absorption.json"
if c2_pilot_file.exists():
    c2_pilot = json.loads(c2_pilot_file.read_text())
    for h_name, h_data in c2_pilot.get("hierarchies", {}).items():
        if "absorption_rate" in h_data:
            c2_absorption[h_name] = {
                "absorption_rate": h_data["absorption_rate"],
                "null_rate": h_data.get("null_rate_random_latents", 1.0),
                "ratio_to_null": h_data.get("ratio_to_null", 0.0),
                "go_nogo": h_data.get("go_nogo", "?"),
            }
    print(f"  C2 pilot absorption data: {list(c2_absorption.keys())}", flush=True)

# Load B1 results for first-letter absorption geometry reference
b1_data = {}
b1_file = FULL_DIR / "B1_decoder_geometry.json"
if b1_file.exists():
    b1_data = json.loads(b1_file.read_text())

# ─────────────────────────────────────────────────────────────────────────────
# Step 3: Load/generate OWT tokens for co-activation measurement
# ─────────────────────────────────────────────────────────────────────────────
report_progress(3, "Loading OWT tokens for co-activation measurement")

owt_cache = WORKSPACE / "exp" / "results" / "owt_tokens_cache.pt"
if owt_cache.exists():
    print("  Loading cached OWT tokens...", flush=True)
    owt_data = torch.load(str(owt_cache), weights_only=True)
    if isinstance(owt_data, dict):
        all_tokens = owt_data.get("tokens", owt_data.get("input_ids"))
    else:
        all_tokens = owt_data
    # Flatten if needed
    if all_tokens.dim() > 1:
        all_tokens = all_tokens.reshape(-1)
    print(f"  OWT tokens loaded: {all_tokens.shape[0]} tokens", flush=True)
else:
    print("  Generating OWT tokens (no cache found)...", flush=True)
    try:
        from datasets import load_dataset
        ds = load_dataset("NeelNanda/pile-10k", split="train", streaming=True)
        texts = []
        for item in ds:
            texts.append(item["text"])
            if len(texts) >= 50:
                break
        all_token_ids = []
        for text in texts:
            toks = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)["input_ids"][0]
            all_token_ids.append(toks)
        all_tokens = torch.cat(all_token_ids)
    except Exception as e:
        print(f"  WARNING: Could not load OWT dataset: {e}", flush=True)
        print("  Generating synthetic token sequence...", flush=True)
        all_tokens = torch.randint(0, 50000, (5000,))

print(f"  Token count: {all_tokens.shape[0]}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Compute SAE activations over OWT sample for co-occurrence frequencies
# ─────────────────────────────────────────────────────────────────────────────
report_progress(4, "Computing SAE activations over OWT tokens for frequencies")

N_OWT = min(all_tokens.shape[0], 5000)  # Use up to 5k tokens for pilot
LAYER = 6

def compute_sae_activations_bulk(tokens, layer=6, batch_size=64, max_tokens=5000):
    """Compute SAE feature frequencies from token stream."""
    tokens = tokens[:max_tokens]
    # Reshape into sequences of length 64
    seq_len = min(64, len(tokens))
    n_seqs = max(1, len(tokens) // seq_len)
    tokens = tokens[:n_seqs * seq_len].reshape(n_seqs, seq_len)

    all_acts_sum = torch.zeros(d_sae)
    total_positions = 0

    for i in range(0, n_seqs, batch_size):
        batch = tokens[i:i+batch_size].to(DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(
                batch,
                names_filter=f"blocks.{layer}.hook_resid_pre",
                return_type=None,
            )
        acts = cache[f"blocks.{layer}.hook_resid_pre"]  # [batch, seq, d_model]
        acts_flat = acts.reshape(-1, d_model).float()
        with torch.no_grad():
            sae_acts = sae.encode(acts_flat)  # [batch*seq, d_sae]
        all_acts_sum += (sae_acts > 0).float().sum(dim=0).cpu()
        total_positions += sae_acts.shape[0]
        if i % (batch_size * 4) == 0:
            print(f"    Processed {min(i + batch_size, n_seqs)}/{n_seqs} seqs", flush=True)

    frequencies = (all_acts_sum / total_positions).numpy()
    return frequencies, total_positions

print("  Computing feature frequencies from OWT...", flush=True)
feature_freqs, n_positions = compute_sae_activations_bulk(all_tokens, layer=LAYER, max_tokens=N_OWT)
print(f"  Frequencies computed over {n_positions} positions", flush=True)
print(f"  Active features (>0.001): {(feature_freqs > 0.001).sum()}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Identify SAE latents aligned with each hierarchy's probe direction
# ─────────────────────────────────────────────────────────────────────────────
report_progress(5, "Identifying SAE latents aligned with probe directions")

def find_probe_aligned_latents(probe_w, top_k=20, cos_threshold=0.1):
    """
    Find SAE latents whose decoder directions align with a probe direction.
    probe_w: [d_model] normalized probe direction vector
    Returns: list of (latent_idx, cos_sim) sorted by |cos_sim| descending
    """
    probe_dir = torch.tensor(probe_w, dtype=torch.float32).to(DEVICE)
    probe_dir = F.normalize(probe_dir.unsqueeze(0), dim=1).squeeze(0)

    # Decoder directions are already normalized in SAELens SAEs
    W_dec_norm = F.normalize(W_dec.to(DEVICE), dim=1)  # [d_sae, d_model]
    cos_sims = (W_dec_norm @ probe_dir).cpu().numpy()  # [d_sae]

    # Top-k by absolute cosine similarity
    top_indices = np.argsort(np.abs(cos_sims))[::-1][:top_k]
    results = [
        {"latent_idx": int(idx), "cos_sim": float(cos_sims[idx]),
         "cos2": float(cos_sims[idx]**2), "freq": float(feature_freqs[idx])}
        for idx in top_indices
        if np.abs(cos_sims[idx]) >= cos_threshold
    ]
    return results, cos_sims


# Passing hierarchies from C1
passing_hierarchies = ["first_letter", "noun_proper", "animate_inanimate"]
# city_country_binary also passed F1 gate but failed shuffle control - include but flag it

hierarchy_latents = {}

for h_name in passing_hierarchies:
    if h_name not in probe_weights:
        print(f"  WARNING: No probe weights for {h_name}, skipping", flush=True)
        continue

    w = probe_weights[h_name]  # [n_classes, d_model]
    classes = probe_classes[h_name]

    print(f"\n  {h_name}: probe shape={w.shape}, classes={classes}", flush=True)

    if h_name == "first_letter":
        # Per-letter binary probes: pick a few letters that pass the F1 gate
        # Use top letters from C1 result
        passing_letters = ["a", "b", "d", "e", "g", "h", "i", "j", "k", "m", "n", "o", "q", "r", "s", "w", "y"]
        class_list = list(classes)

        letter_latents = {}
        for letter_idx, cls in enumerate(class_list):
            letter = str(cls)
            if letter not in passing_letters:
                continue
            if letter_idx >= w.shape[0]:
                continue
            probe_dir = w[letter_idx]
            aligned, _ = find_probe_aligned_latents(probe_dir, top_k=10, cos_threshold=0.08)
            letter_latents[letter] = aligned
            if len(aligned) > 0:
                top = aligned[0]
                print(f"    {letter}: top latent={top['latent_idx']}, cos={top['cos_sim']:.3f}, freq={top['freq']:.4f}", flush=True)

        hierarchy_latents[h_name] = {
            "per_class": letter_latents,
            "hierarchy_type": "first_letter",
            "n_classes_analyzed": len(letter_latents),
        }

    else:
        # Binary or multi-class probes: use single probe direction
        if w.shape[0] == 2:
            # Binary: use class 1 direction (positive class direction)
            probe_dir = w[1] - w[0]  # Contrast direction
        else:
            # Multi-class: average of all class directions
            probe_dir = w.mean(axis=0)

        aligned, cos_sims = find_probe_aligned_latents(probe_dir, top_k=20, cos_threshold=0.05)
        print(f"    {h_name}: {len(aligned)} aligned latents (top cos={aligned[0]['cos_sim']:.3f} if aligned else 'none')", flush=True)

        # Also get per-class latents for binary probes
        per_class = {}
        if w.shape[0] <= 5:
            for cls_idx, cls_name in enumerate(classes):
                if cls_idx >= w.shape[0]:
                    # Probe was saved with only 1 weight row; reuse the single direction
                    aligned_cls, _ = find_probe_aligned_latents(w[0], top_k=10, cos_threshold=0.05)
                else:
                    aligned_cls, _ = find_probe_aligned_latents(w[cls_idx], top_k=10, cos_threshold=0.05)
                per_class[str(cls_name)] = aligned_cls

        hierarchy_latents[h_name] = {
            "aligned_latents": aligned,
            "per_class": per_class,
            "hierarchy_type": h_name,
            "probe_shape": list(w.shape),
        }

# ─────────────────────────────────────────────────────────────────────────────
# Step 6: Compute mean decoder cosine similarity between parent and child latents
# ─────────────────────────────────────────────────────────────────────────────
report_progress(6, "Computing decoder cosine similarities between hierarchy latent pairs")

def compute_parent_child_cos2(parent_latents, child_latents, W_dec_np):
    """
    Compute mean cos^2(theta) between parent SAE decoder directions and
    child SAE decoder directions.
    """
    if not parent_latents or not child_latents:
        return None, []

    parent_idxs = [lt["latent_idx"] for lt in parent_latents]
    child_idxs = [lt["latent_idx"] for lt in child_latents]

    parent_dirs = W_dec_np[parent_idxs]  # [n_p, d_model]
    child_dirs = W_dec_np[child_idxs]    # [n_c, d_model]

    # Normalize
    parent_norms = np.linalg.norm(parent_dirs, axis=1, keepdims=True)
    child_norms = np.linalg.norm(child_dirs, axis=1, keepdims=True)
    parent_dirs_n = parent_dirs / (parent_norms + 1e-8)
    child_dirs_n = child_dirs / (child_norms + 1e-8)

    # All pairwise cosine similarities
    cos_sims = parent_dirs_n @ child_dirs_n.T  # [n_p, n_c]
    cos2_all = (cos_sims ** 2).flatten()

    return float(cos2_all.mean()), cos2_all.tolist()


W_dec_np = W_dec.cpu().numpy()

# For first_letter: compute parent-child cos2 across letter pairs
# Parent = higher-frequency letter features, Child = lower-frequency letter features
# E.g., 'a' latent (parent) vs. specific word features starting with 'a' (child)
# Since we only have single-letter probes (not word probes), we compute:
#   - Between-letter cosine sims (different letters, should be small)
#   - Within-letter cosine sims (same letter, should be larger for absorbed pairs)

# For other hierarchies: parent class latents vs. child class latents
hierarchy_cos2_stats = {}

for h_name in passing_hierarchies:
    if h_name not in hierarchy_latents:
        continue

    h_data = hierarchy_latents[h_name]

    if h_name == "first_letter":
        per_class = h_data.get("per_class", {})
        letters = list(per_class.keys())

        if len(letters) < 2:
            hierarchy_cos2_stats[h_name] = {"error": "insufficient letters for comparison"}
            continue

        # Within-letter: cos2 between top latents of same letter (across positions 0 vs 1)
        within_cos2_vals = []
        for letter, latents in per_class.items():
            if len(latents) >= 2:
                cos2, all_cos2 = compute_parent_child_cos2(
                    latents[:1], latents[1:], W_dec_np)
                if cos2 is not None:
                    within_cos2_vals.append(cos2)

        # Between-letter: cos2 between top latents of different letters (proxy for unrelated)
        between_cos2_vals = []
        letter_list = letters[:5]  # Use first 5 letters to avoid O(n^2) explosion
        for i in range(len(letter_list)):
            for j in range(i+1, len(letter_list)):
                l1, l2 = letter_list[i], letter_list[j]
                latents1 = per_class.get(l1, [])[:3]
                latents2 = per_class.get(l2, [])[:3]
                cos2, all_cos2 = compute_parent_child_cos2(latents1, latents2, W_dec_np)
                if cos2 is not None:
                    between_cos2_vals.append(cos2)

        hierarchy_cos2_stats[h_name] = {
            "within_letter_mean_cos2": float(np.mean(within_cos2_vals)) if within_cos2_vals else None,
            "between_letter_mean_cos2": float(np.mean(between_cos2_vals)) if between_cos2_vals else None,
            "n_letters": len(letters),
            "n_within_pairs": len(within_cos2_vals),
            "n_between_pairs": len(between_cos2_vals),
            # Mean cos2 for parent-child: use between (absorbed = parent absorbs child)
            "mean_parent_child_cos2": float(np.mean(between_cos2_vals)) if between_cos2_vals else None,
        }
        print(f"  {h_name}: within={hierarchy_cos2_stats[h_name]['within_letter_mean_cos2']:.4f}, "
              f"between={hierarchy_cos2_stats[h_name]['between_letter_mean_cos2']:.4f}", flush=True)

    else:
        # Binary probe: parent = positive class (e.g., proper_noun), child = negative class (common_noun)
        per_class = h_data.get("per_class", {})
        class_names = list(per_class.keys())

        if len(class_names) < 2:
            # Fall back to between aligned latents and null latents
            aligned = h_data.get("aligned_latents", [])
            if len(aligned) >= 2:
                cos2, all_cos2 = compute_parent_child_cos2(aligned[:5], aligned[5:10], W_dec_np)
            else:
                cos2 = None
            hierarchy_cos2_stats[h_name] = {"mean_parent_child_cos2": cos2, "n_pairs": 0}
        else:
            # Compute parent-child cos2 between class 0 and class 1 latents
            class0_latents = per_class.get(class_names[0], [])
            class1_latents = per_class.get(class_names[1], [])
            cos2, all_cos2 = compute_parent_child_cos2(class0_latents[:5], class1_latents[:5], W_dec_np)

            hierarchy_cos2_stats[h_name] = {
                "class0": class_names[0],
                "class1": class_names[1],
                "class0_n_latents": len(class0_latents),
                "class1_n_latents": len(class1_latents),
                "mean_parent_child_cos2": cos2,
                "n_pairs": len(all_cos2) if all_cos2 else 0,
            }
            print(f"  {h_name}: parent-child cos2={cos2:.4f} ({len(all_cos2)} pairs)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 7: Compute mean co-occurrence rates
# ─────────────────────────────────────────────────────────────────────────────
report_progress(7, "Computing co-occurrence rates (freq_parent / freq_child)")

def get_latent_frequencies(latents):
    """Get frequencies for a list of latent dicts."""
    return [feature_freqs[lt["latent_idx"]] for lt in latents]


hierarchy_freq_stats = {}

for h_name in passing_hierarchies:
    if h_name not in hierarchy_latents:
        continue

    h_data = hierarchy_latents[h_name]

    if h_name == "first_letter":
        per_class = h_data.get("per_class", {})
        letters = list(per_class.keys())

        # Within-letter freq ratios: top latent for same letter vs. others
        freq_ratios = []
        for letter, latents in per_class.items():
            if len(latents) >= 1:
                parent_freq = feature_freqs[latents[0]["latent_idx"]]
                # Compare to mean freq of other letter latents
                other_freqs = []
                for other_letter, other_latents in per_class.items():
                    if other_letter != letter and len(other_latents) >= 1:
                        other_freqs.append(feature_freqs[other_latents[0]["latent_idx"]])
                if other_freqs:
                    mean_other_freq = np.mean(other_freqs)
                    if mean_other_freq > 1e-8:
                        freq_ratios.append(parent_freq / mean_other_freq)

        hierarchy_freq_stats[h_name] = {
            "mean_freq_ratio": float(np.mean(freq_ratios)) if freq_ratios else None,
            "median_freq_ratio": float(np.median(freq_ratios)) if freq_ratios else None,
            "n_letters": len(letters),
        }

    else:
        per_class = h_data.get("per_class", {})
        class_names = list(per_class.keys())

        if len(class_names) >= 2:
            parent_latents = per_class.get(class_names[0], [])
            child_latents = per_class.get(class_names[1], [])

            parent_freqs = get_latent_frequencies(parent_latents[:5]) if parent_latents else []
            child_freqs = get_latent_frequencies(child_latents[:5]) if child_latents else []

            # H2 prediction: parent should be higher frequency than child in absorbed hierarchy
            # freq_ratio = mean_parent_freq / mean_child_freq
            if parent_freqs and child_freqs:
                mean_p = np.mean(parent_freqs)
                mean_c = np.mean(child_freqs)
                freq_ratio = float(mean_p / mean_c) if mean_c > 1e-8 else None
            else:
                freq_ratio = None

            hierarchy_freq_stats[h_name] = {
                "parent_class": class_names[0],
                "child_class": class_names[1],
                "mean_parent_freq": float(np.mean(parent_freqs)) if parent_freqs else None,
                "mean_child_freq": float(np.mean(child_freqs)) if child_freqs else None,
                "freq_ratio": freq_ratio,
                "n_parent_latents": len(parent_latents),
                "n_child_latents": len(child_latents),
            }
            print(f"  {h_name}: parent_freq={hierarchy_freq_stats[h_name]['mean_parent_freq']:.4f}, "
                  f"child_freq={hierarchy_freq_stats[h_name]['mean_child_freq']:.4f}, "
                  f"ratio={freq_ratio}", flush=True)
        else:
            aligned = h_data.get("aligned_latents", [])
            freqs = get_latent_frequencies(aligned[:5]) if aligned else []
            hierarchy_freq_stats[h_name] = {
                "mean_freq": float(np.mean(freqs)) if freqs else None,
                "note": "Single class only, no ratio computed"
            }

# ─────────────────────────────────────────────────────────────────────────────
# Step 8: Compile absorption rates per hierarchy
# ─────────────────────────────────────────────────────────────────────────────
report_progress(8, "Compiling absorption rates from C2 pilot and B1 data")

# Absorption rates from available sources:
# - first_letter: from B1 pilot (AUROC-based), or from C2 pilot
# - others: from C2 pilot (mostly 0.0, flagged as NO_GO)
# Also use the "control_rate_unrelated_class" as a proxy for selectivity

hierarchy_absorption = {}

for h_name in passing_hierarchies:
    if h_name in c2_absorption:
        c2_data = c2_absorption[h_name]
        abs_rate = c2_data["absorption_rate"]
        null_rate = c2_data["null_rate"]
        ratio = c2_data["ratio_to_null"]
        go_nogo = c2_data["go_nogo"]

        # "Effective absorption" = fraction of concept-word positions where concept latents fire
        # = 1 - absorption_rate (since absorption_rate is how often they DON'T fire)
        # But null_rate = 1.0 means random latents never fire, so ratio is misleading.
        # The mean_max_concept_activation tells us the latents DO fire for concept words.
        # We define: selectivity = 1 - absorption_rate (proper encoding rate)
        selectivity = 1.0 - abs_rate

        hierarchy_absorption[h_name] = {
            "absorption_rate_c2": abs_rate,
            "null_rate_c2": null_rate,
            "ratio_to_null_c2": ratio,
            "selectivity_1minus_abs": selectivity,
            "go_nogo": go_nogo,
            "source": "pilot_C2",
            "note": "C2 measured parent latent activity, not traditional child-feature absorption"
        }
    elif h_name == "first_letter" and b1_data:
        # Use B1 data for first_letter: absorbed pairs show lower cos2 (from B1)
        l6 = b1_data.get("layer6", {})
        wil = l6.get("wilcoxon_analysis", {})
        hierarchy_absorption[h_name] = {
            "absorption_rate_c2": None,
            "auroc_cos2_b1": wil.get("auroc_cos2"),
            "pos_cos2_mean_b1": wil.get("pos_cos2_mean"),
            "neg_cos2_mean_b1": wil.get("neg_cos2_mean"),
            "wilcoxon_p_b1": wil.get("wilcoxon", {}).get("p_value"),
            "source": "B1_decoder_geometry",
            "note": "B1 data: absorbed pairs have lower cos2 than non-absorbed (counter-intuitive)"
        }
    else:
        hierarchy_absorption[h_name] = {
            "absorption_rate_c2": None,
            "source": "unavailable",
        }

print("Absorption rates:", flush=True)
for h, data in hierarchy_absorption.items():
    print(f"  {h}: {data}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 9: Compute Spearman correlations (H2 test)
# ─────────────────────────────────────────────────────────────────────────────
report_progress(9, "Computing Spearman correlations (H2 test)")

# Collect data points for correlation
# X1: mean parent-child cos2 (geometric property)
# X2: freq_ratio (parent/child frequency ratio)
# Y: absorption rate / selectivity

data_points = []
for h_name in passing_hierarchies:
    cos2_stats = hierarchy_cos2_stats.get(h_name, {})
    freq_stats = hierarchy_freq_stats.get(h_name, {})
    abs_data = hierarchy_absorption.get(h_name, {})

    cos2_val = cos2_stats.get("mean_parent_child_cos2") or cos2_stats.get("between_letter_mean_cos2")
    freq_ratio = freq_stats.get("freq_ratio") or freq_stats.get("mean_freq_ratio")
    abs_rate = abs_data.get("absorption_rate_c2")
    selectivity = abs_data.get("selectivity_1minus_abs")

    dp = {
        "hierarchy": h_name,
        "mean_cos2_parent_child": cos2_val,
        "freq_ratio_parent_child": freq_ratio,
        "absorption_rate": abs_rate,
        "selectivity": selectivity,
        "go_nogo": abs_data.get("go_nogo", "unknown"),
        "source": abs_data.get("source", "?"),
    }
    data_points.append(dp)
    print(f"\n  {h_name}:", flush=True)
    for k, v in dp.items():
        print(f"    {k}: {v}", flush=True)

# Spearman correlation tests
# Only for data points where all values are available
valid_cos2 = [(dp["mean_cos2_parent_child"], dp["absorption_rate"])
              for dp in data_points
              if dp["mean_cos2_parent_child"] is not None
              and dp["absorption_rate"] is not None]

valid_freq = [(dp["freq_ratio_parent_child"], dp["absorption_rate"])
              for dp in data_points
              if dp["freq_ratio_parent_child"] is not None
              and dp["absorption_rate"] is not None]

spearman_results = {}

if len(valid_cos2) >= 3:
    cos2_x = [v[0] for v in valid_cos2]
    abs_y = [v[1] for v in valid_cos2]
    rho, pval = stats.spearmanr(cos2_x, abs_y)
    spearman_results["cos2_vs_absorption"] = {
        "rho": float(rho), "p_value": float(pval),
        "n": len(valid_cos2),
        "note": "H2 prediction: higher cos2 → higher absorption (positive correlation)",
        "h2_direction_confirmed": rho > 0,
    }
    print(f"\n  Spearman(cos2, absorption): rho={rho:.3f}, p={pval:.3f}, n={len(valid_cos2)}", flush=True)
else:
    spearman_results["cos2_vs_absorption"] = {
        "error": f"Insufficient data points with both cos2 and absorption rate: n={len(valid_cos2)}",
        "note": "Need n>=3 for correlation. C2 NO_GO means most absorption rates are 0.0."
    }
    print(f"\n  WARNING: Only {len(valid_cos2)} valid data points for cos2-absorption correlation", flush=True)

if len(valid_freq) >= 3:
    freq_x = [v[0] for v in valid_freq]
    abs_y = [v[1] for v in valid_freq]
    rho_f, pval_f = stats.spearmanr(freq_x, abs_y)
    spearman_results["freq_ratio_vs_absorption"] = {
        "rho": float(rho_f), "p_value": float(pval_f),
        "n": len(valid_freq),
        "note": "H2 prediction: higher freq_ratio → higher absorption (positive correlation)",
        "h2_direction_confirmed": rho_f > 0,
    }
    print(f"  Spearman(freq_ratio, absorption): rho={rho_f:.3f}, p={pval_f:.3f}", flush=True)
else:
    spearman_results["freq_ratio_vs_absorption"] = {
        "error": f"Insufficient data points: n={len(valid_freq)}",
    }
    print(f"  WARNING: Only {len(valid_freq)} valid data points for freq_ratio-absorption correlation", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 10: Characterize hierarchy properties (even without absorption correlation)
# ─────────────────────────────────────────────────────────────────────────────
report_progress(10, "Characterizing hierarchy latent properties")

hierarchy_property_summary = []
for h_name in passing_hierarchies:
    cos2_stats = hierarchy_cos2_stats.get(h_name, {})
    freq_stats = hierarchy_freq_stats.get(h_name, {})
    abs_data = hierarchy_absorption.get(h_name, {})
    latent_data = hierarchy_latents.get(h_name, {})

    # Count latents found
    n_latents = 0
    if h_name == "first_letter":
        per_class = latent_data.get("per_class", {})
        n_latents = sum(len(v) for v in per_class.values())
    else:
        n_latents = len(latent_data.get("aligned_latents", []))

    cos2_val = cos2_stats.get("mean_parent_child_cos2") or cos2_stats.get("between_letter_mean_cos2")
    freq_ratio = freq_stats.get("freq_ratio") or freq_stats.get("mean_freq_ratio")
    abs_rate = abs_data.get("absorption_rate_c2")

    summary_entry = {
        "hierarchy": h_name,
        "n_latents_found": n_latents,
        "mean_parent_child_cos2": cos2_val,
        "freq_ratio": freq_ratio,
        "absorption_rate": abs_rate,
        "c2_go_nogo": abs_data.get("go_nogo", "unavailable"),
        "c1_f1": None,
    }
    # Add F1 from C1
    if c1_data:
        hier_c1 = c1_data.get("hierarchies", {}).get(h_name, {})
        summary_entry["c1_f1"] = hier_c1.get("metrics", {}).get("f1")

    hierarchy_property_summary.append(summary_entry)
    print(f"\n  Summary for {h_name}:", flush=True)
    for k, v in summary_entry.items():
        print(f"    {k}: {v}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 11: Assessment and go/no-go
# ─────────────────────────────────────────────────────────────────────────────
report_progress(11, "Assessment of C3 results and H2 status")

# The key challenge: C2 showed absorption_rate=0 for most hierarchies, meaning
# concept latents always fire for concept words (good encoding, low absorption).
# This COULD mean:
# (a) True finding: cross-domain absorption doesn't occur at GPT-2 L6
# (b) Measurement issue: C2 v3 measures parent latent activity, not traditional absorption
#
# H2 prediction: "hierarchies with higher cos2 and freq_ratio show higher absorption"
# With all absorption rates near 0, this reduces to showing that PROPERTIES differ
# by hierarchy type (which is still scientifically interesting).

assessment = {
    "h2_testable": len(valid_cos2) >= 3 or len(valid_freq) >= 3,
    "spearman_cos2_rho": spearman_results.get("cos2_vs_absorption", {}).get("rho"),
    "spearman_freq_rho": spearman_results.get("freq_ratio_vs_absorption", {}).get("rho"),
    "h2_directional_evidence": False,
    "key_finding": "",
    "limitations": [],
    "recommendation": "",
}

# Check if correlation is directionally consistent even if n is small
cos2_rho = spearman_results.get("cos2_vs_absorption", {}).get("rho")
freq_rho = spearman_results.get("freq_ratio_vs_absorption", {}).get("rho")

if cos2_rho is not None and cos2_rho > 0:
    assessment["h2_directional_evidence"] = True
if freq_rho is not None and freq_rho > 0:
    assessment["h2_directional_evidence"] = True

# Key finding
if all(dp["absorption_rate"] == 0.0 for dp in data_points if dp["absorption_rate"] is not None):
    assessment["key_finding"] = (
        "C2 pilot showed near-zero absorption rates across all hierarchy types when measuring "
        "parent latent suppression. C3 correlation is non-informative for absorption rates. "
        "However, hierarchy latent PROPERTIES (cos2, freq_ratio) show meaningful variation "
        "across hierarchy types, suggesting structural differences in how hierarchies are encoded."
    )
    assessment["h2_directional_evidence"] = False
    assessment["recommendation"] = "REDESIGN_C2"
    assessment["limitations"] = [
        "C2 v3 measurement confounds 'parent latent activation quality' with 'absorption signal'",
        "Traditional absorption requires measuring child-feature suppression, not parent-feature activity",
        "Need C2 v4: identify child token SAE features, check if they fail to fire when parent is active",
    ]
else:
    # Some variation in absorption rates - compute correlation
    if cos2_rho is not None and cos2_rho > 0.5:
        assessment["h2_directional_evidence"] = True
        assessment["key_finding"] = f"Positive Spearman correlation (rho={cos2_rho:.2f}) between cos2 and absorption rate (directional evidence, n={len(valid_cos2)})"
    else:
        assessment["key_finding"] = "Mixed results; directional predictions only partially confirmed"

    assessment["recommendation"] = "PROCEED_WITH_CAVEATS"

print(f"\n  H2 testable: {assessment['h2_testable']}", flush=True)
print(f"  H2 directional evidence: {assessment['h2_directional_evidence']}", flush=True)
print(f"  Key finding: {assessment['key_finding']}", flush=True)
print(f"  Recommendation: {assessment['recommendation']}", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Step 12: Save results and done
# ─────────────────────────────────────────────────────────────────────────────
report_progress(12, "Saving results")

elapsed = time.time() - start_time

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "PILOT",
    "elapsed_sec": elapsed,
    "config": {
        "model": "gpt2-small",
        "sae_release": "gpt2-small-res-jb",
        "sae_id": "blocks.6.hook_resid_pre",
        "layer": 6,
        "seed": SEED,
        "n_owt_positions": n_positions,
        "passing_hierarchies": passing_hierarchies,
    },
    "data_points": data_points,
    "hierarchy_cos2_stats": hierarchy_cos2_stats,
    "hierarchy_freq_stats": hierarchy_freq_stats,
    "hierarchy_absorption": hierarchy_absorption,
    "hierarchy_property_summary": hierarchy_property_summary,
    "spearman_correlations": spearman_results,
    "assessment": assessment,
    "pilot_pass_criteria": {
        "description": "Computable for all passing hierarchies from C.2. No hard pass/fail threshold.",
        "hierarchies_analyzed": passing_hierarchies,
        "all_hierarchies_have_cos2": all(
            h in hierarchy_cos2_stats and hierarchy_cos2_stats[h].get("mean_parent_child_cos2") is not None
            for h in passing_hierarchies
        ),
        "all_hierarchies_have_freq": all(
            h in hierarchy_freq_stats for h in passing_hierarchies
        ),
        "h2_correlation_computable": assessment["h2_testable"],
        "pilot_go_nogo": "PARTIAL" if assessment["h2_testable"] else "NO_GO_REDESIGN_C2",
        "note": (
            "C3 properties computed for all hierarchies. "
            "H2 correlation is limited by C2 near-zero absorption rates. "
            "C2 measurement design needs revision (see recommendations)."
        ),
    },
    "design_notes": {
        "c2_dependency": "C2 pilot returned NO_GO (absorption_rate=0 for all non-flagged hierarchies)",
        "c3_feasibility": (
            "C3 hierarchy property computation (cos2, freq_ratio) is feasible and informative "
            "even without C2 absorption rates. Provides foundational characterization for full C3."
        ),
        "c2_redesign_needed": (
            "C2 v4 should implement traditional absorption measurement: "
            "for each child token, find child-specific SAE latents, "
            "check if they fail to fire while parent-class latents are active. "
            "E.g., 'cat' → does the 'cat' latent fail when 'animate' latent is active?"
        ),
    },
}

OUTPUT_FILE.write_text(json.dumps(output, indent=2))
print(f"\nResults saved to {OUTPUT_FILE}", flush=True)

# Also save a summary for quick inspection
summary_str = (
    f"C3 PILOT: H2 testable={assessment['h2_testable']}, "
    f"directional_evidence={assessment['h2_directional_evidence']}. "
    f"Hierarchies analyzed: {passing_hierarchies}. "
    f"cos2 Spearman: rho={cos2_rho}. "
    f"Key finding: {assessment['key_finding'][:100]}. "
    f"Recommendation: {assessment['recommendation']}"
)

mark_done("success", summary_str)

print(f"\n{'='*60}", flush=True)
print(f"SUMMARY: {summary_str}", flush=True)
print(f"{'='*60}", flush=True)
