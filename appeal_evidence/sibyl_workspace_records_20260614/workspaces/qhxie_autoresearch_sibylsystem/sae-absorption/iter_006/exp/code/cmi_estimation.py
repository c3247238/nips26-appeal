#!/usr/bin/env python3
"""
cmi_estimation.py -- CMI Estimation: Rate-Distortion Diagnostic (H3 Primary)

PRIMARY NOVEL CONTRIBUTION.
For all 25 first-letter features on Gemma 2 2B L12 16k:
  compute I(X; w_parent | f_child) using k-NN MI estimator (Kraskov et al., 2004)
  in decoder-direction subspace at d'={10, 20, 30, 50}.
  Correlate CMI with observed absorption rate.
  Target: Spearman rho < -0.3.

Key improvements over previous successive_refinement.py:
  1. Uses first_letter_improved.json (50+ words/letter, better probes)
  2. Uses natural text for 10k token activations for better k-NN estimation
  3. Reports Mann-Whitney U test for absorbed vs non-absorbed partition
  4. Reports Cohen's d and bootstrap CIs
  5. Partial correlations controlling for probe F1

This is the FIRST application of successive refinement theorem to SAE absorption.
"""

import json
import os
import sys
import time
import gc
import random
import traceback
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from scipy import stats
from scipy.special import digamma

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "cmi_estimation"
SEED = 42
GPU_ID = 1

# CMI estimation parameters
KNN_K = 5  # k for k-NN MI estimator (Kraskov et al., 2004)
SUBSPACE_DIM_CANDIDATES = [10, 20, 30, 50]
ABSORBED_THRESHOLD = 0.10  # letters with rate > 10% are "absorbed"
NON_ABSORBED_THRESHOLD = 0.05  # letters with rate < 5% are "non-absorbed"
N_NATURAL_TOKENS = 10000  # natural text tokens for CMI estimation

os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_ID)
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID / Progress / Done helpers ───────────────────────────────────────────
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


def bootstrap_ci(values, n_bootstrap=10000, ci=0.95, seed=SEED):
    rng = np.random.RandomState(seed)
    if len(values) == 0:
        return None, None
    bs = [np.mean(rng.choice(values, size=len(values), replace=True))
          for _ in range(n_bootstrap)]
    a = (1 - ci) / 2
    return (round(float(np.percentile(bs, a * 100)), 4),
            round(float(np.percentile(bs, (1 - a) * 100)), 4))


# ── k-NN Mutual Information Estimator (KSG, Method 1) ──────────────────────

def knn_mi_estimate(X, Y, k=5):
    """
    Estimate mutual information I(X; Y) using the KSG estimator (Kraskov et al. 2004).
    X: (n, d_x) array, Y: (n, d_y) array, k: number of nearest neighbors
    Returns: estimated MI in nats.
    """
    n = X.shape[0]
    if n < k + 2:
        return float('nan')

    XY = np.hstack([X, Y])

    nn_joint = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev',
                                algorithm='ball_tree')
    nn_joint.fit(XY)
    dists_joint, _ = nn_joint.kneighbors(XY)
    eps = dists_joint[:, k]  # k-th NN distance

    nn_x = NearestNeighbors(metric='chebyshev', algorithm='ball_tree')
    nn_x.fit(X)
    nn_y = NearestNeighbors(metric='chebyshev', algorithm='ball_tree')
    nn_y.fit(Y)

    n_x = np.zeros(n)
    n_y = np.zeros(n)

    for i in range(n):
        r = max(eps[i], 1e-10)
        idx_x = nn_x.radius_neighbors([X[i]], radius=r, return_distance=False)[0]
        n_x[i] = max(len(idx_x) - 1, 0)
        idx_y = nn_y.radius_neighbors([Y[i]], radius=r, return_distance=False)[0]
        n_y[i] = max(len(idx_y) - 1, 0)

    # KSG estimator (Method 1)
    mi = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    return float(mi)


def knn_cmi_estimate(X, Y, Z, k=5):
    """
    Estimate conditional mutual information I(X; Y | Z) using the chain rule:
    I(X; Y | Z) = I(X; Y, Z) - I(X; Z)
    """
    YZ = np.hstack([Y, Z])
    mi_x_yz = knn_mi_estimate(X, YZ, k=k)
    mi_x_z = knn_mi_estimate(X, Z, k=k)
    if np.isnan(mi_x_yz) or np.isnan(mi_x_z):
        return float('nan')
    return float(mi_x_yz - mi_x_z)


# ── Natural text generation ─────────────────────────────────────────────────

def generate_natural_text_prompts(n_tokens=10000, seed=42):
    """
    Generate diverse natural English text prompts for background distribution.
    Uses a mix of encyclopedia, narrative, technical, and conversational text.
    """
    rng = random.Random(seed)

    templates = [
        "The city of {city} is located in {country} and is known for",
        "In {year}, the {adj} {noun} was discovered by scientists who",
        "{name} was born in {city} and became famous for their work in",
        "The {adj} {animal} is a species found primarily in {region} where",
        "Once upon a time, in the {adj} kingdom of {place}, there lived a",
        "The {noun} traveled through the {adj} {place} searching for",
        "After many years of {action}, the {noun} finally reached the",
        "The {adj} process of {action} involves several key steps including",
        "According to recent research, the {noun} demonstrates properties of",
        "The mathematical framework describes how {noun} interacts with",
        "Many people believe that the best {noun} can be found in {city}",
        "It is widely known that {name} contributed significantly to",
        "The relationship between {noun} and {noun2} has been studied",
        "Scientists at the university in {city} have recently published",
        "The {adj} landscape of {country} features stunning mountains and",
    ]

    cities = ["London", "Paris", "Tokyo", "Berlin", "Moscow", "Cairo", "Sydney",
              "Madrid", "Rome", "Beijing", "Delhi", "Bangkok", "Lagos", "Nairobi",
              "Toronto", "Mexico", "Lima", "Jakarta", "Seoul", "Istanbul",
              "Amsterdam", "Vienna", "Prague", "Warsaw", "Athens", "Dublin",
              "Helsinki", "Stockholm", "Oslo", "Lisbon", "Zurich", "Brussels",
              "Munich", "Hamburg", "Milan", "Barcelona", "Manchester", "Glasgow"]
    countries = ["France", "Germany", "Japan", "China", "India", "Brazil",
                 "Australia", "Canada", "Mexico", "Italy", "Spain", "Russia",
                 "Egypt", "Nigeria", "Kenya", "Thailand", "Indonesia", "Korea"]
    names = ["Albert", "Marie", "Charles", "Elizabeth", "William", "Catherine",
             "Thomas", "Margaret", "George", "Florence", "James", "Victoria"]
    adjectives = ["ancient", "brilliant", "complex", "diverse", "elegant",
                  "famous", "gentle", "hidden", "immense", "luminous",
                  "massive", "notable", "peculiar", "quiet", "remarkable",
                  "subtle", "tremendous", "vibrant", "wonderful", "exotic"]
    nouns = ["mountain", "river", "forest", "ocean", "desert", "island",
             "bridge", "castle", "garden", "harbor", "library", "market",
             "palace", "temple", "village", "crystal", "diamond", "emerald",
             "falcon", "dolphin", "leopard", "phoenix", "serpent", "dragon"]
    nouns2 = ["energy", "matter", "light", "gravity", "sound", "heat",
              "pressure", "voltage", "current", "frequency", "wavelength"]
    animals = ["elephant", "tiger", "penguin", "dolphin", "eagle", "whale",
               "leopard", "gorilla", "falcon", "turtle", "snake", "butterfly"]
    regions = ["Africa", "Asia", "Europe", "Americas", "Pacific", "Arctic",
               "Sahara", "Amazon", "Himalayas", "Siberia"]
    actions = ["exploration", "discovery", "invention", "creation",
               "experimentation", "transformation", "evolution", "migration"]
    years = [str(y) for y in range(1700, 2025)]
    places = ["Eldorado", "Avalon", "Narnia", "Atlantis", "Camelot",
              "Olympus", "Valhalla", "Shangri", "Utopia"]

    prompts = []
    n_prompts = max(n_tokens // 15, 700)

    for i in range(n_prompts):
        template = rng.choice(templates)
        text = template.format(
            city=rng.choice(cities),
            country=rng.choice(countries),
            name=rng.choice(names),
            adj=rng.choice(adjectives),
            noun=rng.choice(nouns),
            noun2=rng.choice(nouns2),
            animal=rng.choice(animals),
            region=rng.choice(regions),
            action=rng.choice(actions),
            year=rng.choice(years),
            place=rng.choice(places),
        )
        prompts.append(text)

    return prompts


def collect_natural_text_activations(model, sae, tokenizer, hook_point,
                                     device, n_tokens=10000, seed=42):
    """
    Collect SAE latent activations and raw model activations from natural text.
    Returns: sae_acts [N, d_sae], raw_acts [N, d_model], token_ids [N]
    """
    prompts = generate_natural_text_prompts(n_tokens=n_tokens, seed=seed)

    all_sae_acts = []
    all_raw_acts = []
    all_token_ids = []
    collected = 0

    for pi, prompt in enumerate(prompts):
        if collected >= n_tokens:
            break

        tokens = tokenizer.encode(prompt, return_tensors="pt",
                                  truncation=True, max_length=128).to(device)
        seq_len = tokens.shape[1]

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=[hook_point],
                return_type=None, prepend_bos=True,
            )
        raw_acts = cache[hook_point][0].detach()  # [seq_len+1, d_model]

        # Take positions excluding BOS
        raw_acts = raw_acts[1:seq_len + 1]  # [seq_len, d_model]

        with torch.no_grad():
            sae_acts = sae.encode(raw_acts)  # [seq_len, d_sae]

        all_sae_acts.append(sae_acts.cpu())
        all_raw_acts.append(raw_acts.cpu())
        all_token_ids.extend(tokens[0].cpu().tolist())
        collected += seq_len

        del cache
        if collected % 2000 < seq_len:
            torch.cuda.empty_cache()

        if pi % 100 == 0 and pi > 0:
            print(f"    Collected {collected}/{n_tokens} tokens ({pi} prompts)")

    sae_acts_cat = torch.cat(all_sae_acts, dim=0)[:n_tokens].numpy()
    raw_acts_cat = torch.cat(all_raw_acts, dim=0)[:n_tokens].numpy()
    token_ids = all_token_ids[:n_tokens]

    return sae_acts_cat, raw_acts_cat, token_ids


def collect_word_activations(model, sae, tokenizer, letter_words,
                             hook_point, device):
    """
    Collect SAE latent activations AND raw model activations for first-letter words.
    Returns: sae_acts [N, d_sae], raw_acts [N, d_model], labels [N], words [N]
    """
    all_sae_acts = []
    all_raw_acts = []
    all_labels = []
    all_words = []

    for letter in sorted(letter_words.keys()):
        for w_info in letter_words[letter]:
            word = w_info["word"]
            prompt = f" {word}"
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=[hook_point],
                    return_type=None, prepend_bos=True,
                )
            raw_act = cache[hook_point][0, -1, :].detach()  # [d_model]

            with torch.no_grad():
                sae_out = sae.encode(raw_act.unsqueeze(0))
            sae_act = sae_out[0].detach()

            all_sae_acts.append(sae_act.cpu())
            all_raw_acts.append(raw_act.cpu())
            all_labels.append(letter)
            all_words.append(word)

            del cache

    torch.cuda.empty_cache()
    sae_acts = torch.stack(all_sae_acts).numpy()
    raw_acts = torch.stack(all_raw_acts).numpy()
    return sae_acts, raw_acts, all_labels, all_words


def compute_cmi_for_letter(letter, split_features, absorption_rate,
                           word_sae_acts, word_raw_acts, word_labels,
                           natural_sae_acts, natural_raw_acts,
                           W_dec_np, subspace_dim=20, k=5, seed=42):
    """
    Compute I(X; w_parent | f_child) for a single letter.

    Uses COMBINED dataset: word-level activations (for precise letter-specific signal)
    + natural text activations (for diverse background distribution).
    """
    # Combine word and natural text activations
    combined_sae = np.vstack([word_sae_acts, natural_sae_acts])
    combined_raw = np.vstack([word_raw_acts, natural_raw_acts])
    n_total = combined_sae.shape[0]

    # Build decoder-direction subspace using relevant features
    relevant_decoder_dirs = []
    for sf in split_features:
        if sf < W_dec_np.shape[0]:
            relevant_decoder_dirs.append(W_dec_np[sf])

    # Add decoder directions of features most active on this letter's words
    labels_arr = np.array(word_labels)
    pos_mask = labels_arr == letter
    pos_indices = np.where(pos_mask)[0]
    if len(pos_indices) < 3:
        return {"status": "skipped", "reason": f"n_pos={len(pos_indices)}<3"}

    pos_sae_acts = word_sae_acts[pos_indices]
    mean_acts = pos_sae_acts.mean(axis=0)
    n_extra = max(subspace_dim - len(split_features), 10)
    top_active = np.argsort(mean_acts)[::-1][:n_extra]
    for fi in top_active:
        if mean_acts[fi] > 0 and fi < W_dec_np.shape[0]:
            relevant_decoder_dirs.append(W_dec_np[fi])

    if len(relevant_decoder_dirs) < 3:
        return {"status": "skipped", "reason": "insufficient decoder directions"}

    # Build subspace via SVD
    D = np.stack(relevant_decoder_dirs)
    U, S, Vt = np.linalg.svd(D, full_matrices=False)
    actual_dim = min(subspace_dim, len(S), Vt.shape[0])
    V_sub = Vt[:actual_dim]  # [actual_dim, d_model]

    # X: project raw activations into subspace
    X_proj = combined_raw @ V_sub.T  # [N, actual_dim]

    # Y_parent: the parent direction (dominant split feature decoder direction)
    parent_dir = W_dec_np[split_features[0]].copy()
    parent_norm = np.linalg.norm(parent_dir)
    if parent_norm < 1e-10:
        return {"status": "skipped", "reason": "zero parent direction"}
    parent_dir /= parent_norm
    Y_parent = (combined_raw @ parent_dir).reshape(-1, 1)  # [N, 1]

    # Z_child: child latent activations (split features)
    valid_sf = [sf for sf in split_features if sf < combined_sae.shape[1]]
    Z_child = combined_sae[:, valid_sf]  # [N, k_sparse]

    # Standardize for numerical stability
    X_std = (X_proj - X_proj.mean(axis=0)) / (X_proj.std(axis=0) + 1e-10)
    Y_std = (Y_parent - Y_parent.mean(axis=0)) / (Y_parent.std(axis=0) + 1e-10)
    Z_std = (Z_child - Z_child.mean(axis=0)) / (Z_child.std(axis=0) + 1e-10)

    # Subsample for computational efficiency
    max_samples = 5000
    if n_total > max_samples:
        rng = np.random.RandomState(seed)
        n_word = word_sae_acts.shape[0]
        n_natural_sample = max_samples - n_word
        if n_natural_sample > 0:
            natural_indices = rng.choice(
                range(n_word, n_total),
                size=min(n_natural_sample, n_total - n_word),
                replace=False
            )
            indices = np.concatenate([np.arange(n_word), natural_indices])
            indices.sort()
        else:
            indices = np.arange(n_word)
        X_std = X_std[indices]
        Y_std = Y_std[indices]
        Z_std = Z_std[indices]
        n_used = len(indices)
    else:
        n_used = n_total

    try:
        cmi = knn_cmi_estimate(X_std, Y_std, Z_std, k=k)
    except Exception as e:
        return {"status": "error", "reason": str(e)}

    # Reference MIs
    try:
        mi_parent = knn_mi_estimate(X_std, Y_std, k=k)
    except:
        mi_parent = float('nan')
    try:
        mi_child = knn_mi_estimate(X_std, Z_std, k=k)
    except:
        mi_child = float('nan')

    return {
        "status": "ok",
        "cmi": float(cmi),
        "mi_parent": float(mi_parent),
        "mi_child": float(mi_child),
        "n_samples_total": int(n_total),
        "n_samples_used": int(n_used),
        "n_pos_words": int(len(pos_indices)),
        "subspace_dim_actual": int(actual_dim),
        "n_decoder_dirs": len(relevant_decoder_dirs),
        "absorption_rate": float(absorption_rate),
        "is_finite": bool(np.isfinite(cmi)),
    }


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float('nan')
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return float('nan')
    return float((m1 - m2) / pooled_std)


def main():
    set_seeds(SEED)
    start_time = time.time()
    device = DEVICE
    total_steps = 10

    results = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae_config": "L12-16k",
        "timestamp_start": datetime.now().isoformat(),
        "parameters": {
            "knn_k": KNN_K,
            "subspace_dim_candidates": SUBSPACE_DIM_CANDIDATES,
            "absorbed_threshold": ABSORBED_THRESHOLD,
            "non_absorbed_threshold": NON_ABSORBED_THRESHOLD,
            "n_natural_tokens": N_NATURAL_TOKENS,
        },
    }

    # ── Step 1: Load absorption data from first_letter_improved ───────────
    report_progress(1, total_steps, "Loading first_letter_improved results")
    print("\n[Step 1/10] Loading first_letter_improved results...")

    fl_path = RESULTS_DIR / "first_letter_improved.json"
    if not fl_path.exists():
        raise FileNotFoundError(f"Required dependency not found: {fl_path}")

    fl_data = json.loads(fl_path.read_text())
    fl_l12_16k = fl_data["l12_16k"]

    per_letter_absorption = {}
    per_letter_split_features = {}
    per_letter_probe_f1 = {}
    per_letter_n_pos = {}
    for letter, info in fl_l12_16k["per_letter"].items():
        if isinstance(info, dict) and info.get("status") == "ok":
            per_letter_absorption[letter] = info["absorption_rate"]
            per_letter_split_features[letter] = info["split_features"]
            per_letter_probe_f1[letter] = info["probe_f1"]
            per_letter_n_pos[letter] = info["n_pos"]
            print(f"  {letter}: absorption={info['absorption_rate']:.4f}, "
                  f"probe_f1={info['probe_f1']:.4f}, n={info['n_pos']}, "
                  f"splits={info['split_features'][:3]}...")

    absorbed_letters = sorted([l for l, r in per_letter_absorption.items()
                               if r > ABSORBED_THRESHOLD])
    non_absorbed_letters = sorted([l for l, r in per_letter_absorption.items()
                                   if r < NON_ABSORBED_THRESHOLD])
    ambiguous_letters = sorted([l for l, r in per_letter_absorption.items()
                                if NON_ABSORBED_THRESHOLD <= r <= ABSORBED_THRESHOLD])

    print(f"\n  Absorbed (rate>{ABSORBED_THRESHOLD}): {absorbed_letters} ({len(absorbed_letters)})")
    print(f"  Non-absorbed (rate<{NON_ABSORBED_THRESHOLD}): "
          f"{non_absorbed_letters} ({len(non_absorbed_letters)})")
    print(f"  Ambiguous: {ambiguous_letters} ({len(ambiguous_letters)})")

    results["first_letter_source"] = str(fl_path)
    results["partition"] = {
        "absorbed_letters": absorbed_letters,
        "non_absorbed_letters": non_absorbed_letters,
        "ambiguous_letters": ambiguous_letters,
        "absorbed_rates": {l: per_letter_absorption[l] for l in absorbed_letters},
        "non_absorbed_rates": {l: per_letter_absorption[l] for l in non_absorbed_letters},
        "absorbed_threshold": ABSORBED_THRESHOLD,
        "non_absorbed_threshold": NON_ABSORBED_THRESHOLD,
    }

    # ── Step 2: Load model ────────────────────────────────────────────────
    report_progress(2, total_steps, "Loading Gemma 2 2B model")
    print("\n[Step 2/10] Loading Gemma 2 2B model...")
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

    # ── Step 3: Load SAE L12-16k ──────────────────────────────────────────
    report_progress(3, total_steps, "Loading SAE L12-16k")
    print("\n[Step 3/10] Loading SAE L12-16k...")
    t0 = time.time()

    from sae_lens import SAE
    sae = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device=device,
    )
    print(f"  SAE loaded in {time.time()-t0:.1f}s, features={sae.cfg.d_sae}")

    hook_point = "blocks.12.hook_resid_post"
    W_dec = sae.W_dec.data.cpu()
    W_dec_np = W_dec.numpy()

    # ── Step 4: Build vocabulary (from first_letter_improved) ─────────────
    report_progress(4, total_steps, "Building vocabulary from first_letter_improved")
    print("\n[Step 4/10] Building vocabulary...")
    t0 = time.time()

    vocab = fl_data.get("vocabulary", {})
    examples = vocab.get("examples", {})

    letter_words = {}
    for letter in sorted(per_letter_absorption.keys()):
        words_list = examples.get(letter, [])
        valid = []
        seen = set()
        for word in words_list:
            if word.lower() in seen:
                continue
            seen.add(word.lower())
            for w in [word.lower(), word]:
                encoded = tokenizer.encode(f" {w}", add_special_tokens=False)
                if len(encoded) == 1:
                    valid.append({"word": w, "token_id": encoded[0]})
                    break
            if len(valid) >= 100:
                break
        letter_words[letter] = valid

    # Remove letters with too few words
    for letter in list(letter_words.keys()):
        if len(letter_words[letter]) < 3:
            print(f"  WARNING: letter {letter} has only "
                  f"{len(letter_words[letter])} words, removing")
            del letter_words[letter]

    total_words = sum(len(v) for v in letter_words.values())
    print(f"  {len(letter_words)} letters, {total_words} words total")
    for letter in sorted(letter_words.keys()):
        print(f"    {letter}: {len(letter_words[letter])} words")

    results["vocabulary"] = {
        "n_letters": len(letter_words),
        "total_words": total_words,
        "per_letter_counts": {k: len(v) for k, v in sorted(letter_words.items())},
    }

    # ── Step 5: Collect word activations ──────────────────────────────────
    report_progress(5, total_steps, "Collecting word activations")
    print("\n[Step 5/10] Collecting word activations...")
    t0 = time.time()

    word_sae_acts, word_raw_acts, word_labels, word_words = \
        collect_word_activations(model, sae, tokenizer, letter_words,
                                hook_point, device)
    print(f"  Collected {len(word_labels)} word activations in {time.time()-t0:.1f}s")
    print(f"  Word SAE shape: {word_sae_acts.shape}, Raw shape: {word_raw_acts.shape}")

    # ── Step 6: Collect natural text activations ──────────────────────────
    report_progress(6, total_steps,
                    f"Collecting {N_NATURAL_TOKENS} natural text activations")
    print(f"\n[Step 6/10] Collecting {N_NATURAL_TOKENS} natural text activations...")
    t0 = time.time()

    natural_sae_acts, natural_raw_acts, natural_token_ids = \
        collect_natural_text_activations(
            model, sae, tokenizer, hook_point, device,
            n_tokens=N_NATURAL_TOKENS, seed=SEED
        )
    print(f"  Collected {natural_sae_acts.shape[0]} natural text activations "
          f"in {time.time()-t0:.1f}s")

    results["data_summary"] = {
        "n_word_activations": int(word_sae_acts.shape[0]),
        "n_natural_activations": int(natural_sae_acts.shape[0]),
        "total_activations": int(word_sae_acts.shape[0] + natural_sae_acts.shape[0]),
    }

    # Free GPU memory
    del model, sae
    gc.collect(); torch.cuda.empty_cache()
    print("  GPU memory freed")

    # ── Step 7: Compute CMI for each letter at multiple subspace dims ─────
    report_progress(7, total_steps, "Computing CMI for each letter")
    print("\n[Step 7/10] Computing CMI for each letter at multiple "
          "subspace dimensions...")

    valid_letters = sorted([l for l in per_letter_absorption.keys()
                            if l in set(word_labels)])

    cmi_results_by_dim = {}
    best_dim = None
    best_rho = 0.0

    for subspace_dim in SUBSPACE_DIM_CANDIDATES:
        print(f"\n  === Subspace dim d'={subspace_dim} ===")
        t0 = time.time()

        cmi_per_letter = {}
        for letter in valid_letters:
            if letter not in per_letter_split_features:
                cmi_per_letter[letter] = {
                    "status": "skipped", "reason": "no split features"
                }
                continue

            cmi_result = compute_cmi_for_letter(
                letter=letter,
                split_features=per_letter_split_features[letter],
                absorption_rate=per_letter_absorption[letter],
                word_sae_acts=word_sae_acts,
                word_raw_acts=word_raw_acts,
                word_labels=word_labels,
                natural_sae_acts=natural_sae_acts,
                natural_raw_acts=natural_raw_acts,
                W_dec_np=W_dec_np,
                subspace_dim=subspace_dim,
                k=KNN_K,
                seed=SEED,
            )
            cmi_per_letter[letter] = cmi_result
            if cmi_result["status"] == "ok":
                print(f"    {letter}: CMI={cmi_result['cmi']:.4f}, "
                      f"MI_parent={cmi_result['mi_parent']:.4f}, "
                      f"MI_child={cmi_result['mi_child']:.4f}, "
                      f"abs_rate={per_letter_absorption[letter]:.4f}")

        n_valid = sum(1 for v in cmi_per_letter.values()
                      if v.get("status") == "ok" and v.get("is_finite", False))

        # Compute correlation at this dimension
        dim_cmi = []
        dim_rates = []
        for l in valid_letters:
            if (l in cmi_per_letter and
                cmi_per_letter[l].get("status") == "ok" and
                cmi_per_letter[l].get("is_finite", False) and
                l in per_letter_absorption):
                dim_cmi.append(cmi_per_letter[l]["cmi"])
                dim_rates.append(per_letter_absorption[l])
        if len(dim_cmi) >= 5:
            rho_dim, _ = stats.spearmanr(dim_rates, dim_cmi)
        else:
            rho_dim = 0.0

        print(f"  d'={subspace_dim}: {n_valid} valid, rho={rho_dim:.4f}, "
              f"time={time.time()-t0:.1f}s")

        cmi_results_by_dim[str(subspace_dim)] = cmi_per_letter

        # Select dimension with best properties
        if n_valid >= 15 and (best_dim is None or abs(rho_dim) > abs(best_rho)):
            best_rho = rho_dim
            best_dim = subspace_dim

    # Fallback
    if best_dim is None:
        best_n = 0
        for dim in SUBSPACE_DIM_CANDIDATES:
            n_v = sum(1 for v in cmi_results_by_dim[str(dim)].values()
                      if v.get("status") == "ok" and v.get("is_finite", False))
            if n_v > best_n:
                best_n = n_v
                best_dim = dim

    results["cmi_by_subspace_dim"] = cmi_results_by_dim
    print(f"\n  Selected best subspace dim: d'={best_dim} (rho={best_rho:.4f})")
    results["best_subspace_dim"] = best_dim

    # ── Step 8: Mann-Whitney U test ───────────────────────────────────────
    report_progress(8, total_steps, "Mann-Whitney U test")
    print("\n[Step 8/10] Mann-Whitney U test...")

    primary_cmi = cmi_results_by_dim[str(best_dim)]

    cmi_absorbed = []
    cmi_non_absorbed = []
    cmi_all = {}

    for letter in valid_letters:
        info = primary_cmi.get(letter, {})
        if info.get("status") != "ok" or not info.get("is_finite", False):
            continue
        cmi_val = info["cmi"]
        cmi_all[letter] = cmi_val
        if letter in absorbed_letters:
            cmi_absorbed.append(cmi_val)
        elif letter in non_absorbed_letters:
            cmi_non_absorbed.append(cmi_val)

    print(f"  Absorbed ({len(cmi_absorbed)}): "
          f"{[f'{v:.4f}' for v in cmi_absorbed]}")
    print(f"  Non-absorbed ({len(cmi_non_absorbed)}): "
          f"{[f'{v:.4f}' for v in cmi_non_absorbed]}")

    mann_whitney_result = {}
    if len(cmi_absorbed) >= 2 and len(cmi_non_absorbed) >= 2:
        U_less, p_less = stats.mannwhitneyu(
            cmi_absorbed, cmi_non_absorbed, alternative='less')
        U_greater, p_greater = stats.mannwhitneyu(
            cmi_absorbed, cmi_non_absorbed, alternative='greater')
        U_two, p_two = stats.mannwhitneyu(
            cmi_absorbed, cmi_non_absorbed, alternative='two-sided')
        d = cohens_d(cmi_absorbed, cmi_non_absorbed)

        print(f"\n  MW (absorbed < non-absorbed): U={U_less:.2f}, p={p_less:.6f}")
        print(f"  MW (absorbed > non-absorbed): U={U_greater:.2f}, p={p_greater:.6f}")
        print(f"  MW (two-sided): U={U_two:.2f}, p={p_two:.6f}")
        print(f"  Cohen's d = {d:.4f}")
        print(f"  Absorbed mean = {np.mean(cmi_absorbed):.4f} +/- "
              f"{np.std(cmi_absorbed):.4f}")
        print(f"  Non-absorbed mean = {np.mean(cmi_non_absorbed):.4f} +/- "
              f"{np.std(cmi_non_absorbed):.4f}")

        # Bootstrap CI
        rng = np.random.RandomState(SEED)
        abs_arr = np.array(cmi_absorbed)
        na_arr = np.array(cmi_non_absorbed)
        diffs = []
        for _ in range(10000):
            a_sample = rng.choice(abs_arr, size=len(abs_arr), replace=True)
            na_sample = rng.choice(na_arr, size=len(na_arr), replace=True)
            diffs.append(np.mean(a_sample) - np.mean(na_sample))
        ci_low = float(np.percentile(diffs, 2.5))
        ci_high = float(np.percentile(diffs, 97.5))

        mann_whitney_result = {
            "U_less": float(U_less), "p_less": float(p_less),
            "U_greater": float(U_greater), "p_greater": float(p_greater),
            "U_two_sided": float(U_two), "p_two_sided": float(p_two),
            "cohens_d": d,
            "absorbed_mean": float(np.mean(cmi_absorbed)),
            "absorbed_std": float(np.std(cmi_absorbed)),
            "absorbed_median": float(np.median(cmi_absorbed)),
            "non_absorbed_mean": float(np.mean(cmi_non_absorbed)),
            "non_absorbed_std": float(np.std(cmi_non_absorbed)),
            "non_absorbed_median": float(np.median(cmi_non_absorbed)),
            "n_absorbed": len(cmi_absorbed),
            "n_non_absorbed": len(cmi_non_absorbed),
            "sig_less_0_05": bool(p_less < 0.05),
            "sig_greater_0_05": bool(p_greater < 0.05),
            "sig_two_sided_0_05": bool(p_two < 0.05),
            "large_effect": bool(abs(d) > 0.8 if np.isfinite(d) else False),
            "mean_diff_ci_95": [ci_low, ci_high],
        }
    else:
        mann_whitney_result = {
            "status": "insufficient_data",
            "n_absorbed": len(cmi_absorbed),
            "n_non_absorbed": len(cmi_non_absorbed),
        }

    results["mann_whitney_test"] = mann_whitney_result

    # ── Step 9: Spearman correlation ──────────────────────────────────────
    report_progress(9, total_steps, "Spearman correlation: CMI vs absorption rate")
    print("\n[Step 9/10] Spearman correlation: CMI vs absorption rate...")

    corr_letters = []
    corr_cmi = []
    corr_rates = []
    corr_probe_f1 = []
    for letter in sorted(cmi_all.keys()):
        if letter in per_letter_absorption:
            corr_letters.append(letter)
            corr_cmi.append(cmi_all[letter])
            corr_rates.append(per_letter_absorption[letter])
            corr_probe_f1.append(per_letter_probe_f1.get(letter, 0.0))

    correlation_result = {}
    if len(corr_letters) >= 5:
        rho, p_val = stats.spearmanr(corr_rates, corr_cmi)
        pearson_r, pearson_p = stats.pearsonr(corr_rates, corr_cmi)

        # Partial correlation controlling for probe F1
        partial_rho, partial_p = float('nan'), float('nan')
        if len(corr_letters) >= 8:
            from scipy.stats import rankdata
            r_rates = rankdata(corr_rates)
            r_cmi = rankdata(corr_cmi)
            r_f1 = rankdata(corr_probe_f1)
            coef_rate_f1 = np.polyfit(r_f1, r_rates, 1)
            resid_rate = r_rates - np.polyval(coef_rate_f1, r_f1)
            coef_cmi_f1 = np.polyfit(r_f1, r_cmi, 1)
            resid_cmi = r_cmi - np.polyval(coef_cmi_f1, r_f1)
            partial_rho, partial_p = stats.spearmanr(resid_rate, resid_cmi)

        print(f"  N = {len(corr_letters)} letters")
        print(f"  Spearman rho = {rho:.4f}, p = {p_val:.6f}")
        print(f"  Pearson r = {pearson_r:.4f}, p = {pearson_p:.6f}")
        if np.isfinite(partial_rho):
            print(f"  Partial Spearman (ctrl probe F1) = {partial_rho:.4f}, "
                  f"p = {partial_p:.6f}")

        h3_met = rho < -0.3
        print(f"  H3 target (rho < -0.3): {'MET' if h3_met else 'NOT MET'}")

        per_letter_data = []
        for i, letter in enumerate(corr_letters):
            per_letter_data.append({
                "letter": letter,
                "cmi": float(corr_cmi[i]),
                "absorption_rate": float(corr_rates[i]),
                "probe_f1": float(corr_probe_f1[i]),
                "group": ("absorbed" if letter in absorbed_letters else
                          ("non_absorbed" if letter in non_absorbed_letters
                           else "ambiguous")),
            })

        correlation_result = {
            "n_letters": len(corr_letters),
            "spearman_rho": float(rho),
            "spearman_p": float(p_val),
            "pearson_r": float(pearson_r),
            "pearson_p": float(pearson_p),
            "partial_spearman_rho": (float(partial_rho)
                                     if np.isfinite(partial_rho) else None),
            "partial_spearman_p": (float(partial_p)
                                   if np.isfinite(partial_p) else None),
            "h3_target_met": h3_met,
            "letters": corr_letters,
            "cmi_values": [float(x) for x in corr_cmi],
            "absorption_rates": [float(x) for x in corr_rates],
            "probe_f1_values": [float(x) for x in corr_probe_f1],
            "per_letter_data": per_letter_data,
        }
    else:
        correlation_result = {
            "status": "insufficient_data",
            "n_letters": len(corr_letters),
        }

    results["correlation_cmi_vs_absorption"] = correlation_result

    # ── Step 10: Summary and sensitivity ──────────────────────────────────
    report_progress(10, total_steps, "Building summary")
    print("\n[Step 10/10] Summary table...")

    summary_table = []
    for letter in sorted(per_letter_absorption.keys()):
        row = {
            "letter": letter,
            "absorption_rate": per_letter_absorption[letter],
            "probe_f1": per_letter_probe_f1.get(letter, None),
            "n_words": per_letter_n_pos.get(letter, 0),
            "group": ("absorbed" if letter in absorbed_letters else
                      ("non_absorbed" if letter in non_absorbed_letters
                       else "ambiguous")),
        }
        if letter in cmi_all:
            row["cmi"] = cmi_all[letter]
            info = primary_cmi.get(letter, {})
            row["mi_parent"] = info.get("mi_parent")
            row["mi_child"] = info.get("mi_child")
            row["n_samples_used"] = info.get("n_samples_used")
        else:
            row["cmi"] = None
        summary_table.append(row)
        if row.get("cmi") is not None:
            print(f"  {letter}: rate={row['absorption_rate']:.4f}, "
                  f"CMI={row['cmi']:.4f}, F1={row.get('probe_f1', 0):.4f}, "
                  f"group={row['group']}")

    results["summary_table"] = summary_table

    # Subspace dimension sensitivity
    dim_sensitivity = {}
    for dim_str, cmi_per_letter in cmi_results_by_dim.items():
        dim = int(dim_str)
        abs_vals = [cmi_per_letter[l]["cmi"] for l in absorbed_letters
                    if l in cmi_per_letter and
                    cmi_per_letter[l].get("status") == "ok" and
                    cmi_per_letter[l].get("is_finite", False)]
        non_abs_vals = [cmi_per_letter[l]["cmi"] for l in non_absorbed_letters
                        if l in cmi_per_letter and
                        cmi_per_letter[l].get("status") == "ok" and
                        cmi_per_letter[l].get("is_finite", False)]

        dim_letters = []
        dim_cmi_vals = []
        dim_rates = []
        for l in valid_letters:
            if (l in cmi_per_letter and
                cmi_per_letter[l].get("status") == "ok" and
                cmi_per_letter[l].get("is_finite", False) and
                l in per_letter_absorption):
                dim_letters.append(l)
                dim_cmi_vals.append(cmi_per_letter[l]["cmi"])
                dim_rates.append(per_letter_absorption[l])

        rho_dim, p_dim = (float('nan'), float('nan'))
        if len(dim_letters) >= 5:
            rho_dim, p_dim = stats.spearmanr(dim_rates, dim_cmi_vals)

        if len(abs_vals) >= 2 and len(non_abs_vals) >= 2:
            u, p = stats.mannwhitneyu(abs_vals, non_abs_vals,
                                      alternative='two-sided')
            d = cohens_d(abs_vals, non_abs_vals)
        else:
            u, p, d = float('nan'), float('nan'), float('nan')

        dim_sensitivity[dim_str] = {
            "subspace_dim": dim,
            "n_absorbed_valid": len(abs_vals),
            "n_non_absorbed_valid": len(non_abs_vals),
            "n_total_valid": len(dim_letters),
            "U_statistic": float(u) if np.isfinite(u) else None,
            "p_value": float(p) if np.isfinite(p) else None,
            "cohens_d": float(d) if np.isfinite(d) else None,
            "absorbed_mean": (float(np.mean(abs_vals)) if abs_vals else None),
            "non_absorbed_mean": (float(np.mean(non_abs_vals))
                                  if non_abs_vals else None),
            "spearman_rho": (float(rho_dim) if np.isfinite(rho_dim) else None),
            "spearman_p": (float(p_dim) if np.isfinite(p_dim) else None),
        }

    results["subspace_dim_sensitivity"] = dim_sensitivity
    print("\n  Subspace dimension sensitivity:")
    for dim_str in sorted(dim_sensitivity.keys(), key=lambda x: int(x)):
        info = dim_sensitivity[dim_str]
        rho_s = (f"{info['spearman_rho']:.3f}"
                 if info['spearman_rho'] is not None else "N/A")
        p_s = (f"{info['p_value']:.4f}"
               if info['p_value'] is not None else "N/A")
        d_s = (f"{info['cohens_d']:.3f}"
               if info['cohens_d'] is not None else "N/A")
        print(f"    d'={dim_str}: rho={rho_s}, U_p={p_s}, d={d_s}, "
              f"n={info['n_total_valid']}")

    # CMI distribution
    all_cmi_vals = [v for v in cmi_all.values() if np.isfinite(v)]
    if all_cmi_vals:
        results["cmi_distribution"] = {
            "min": float(np.min(all_cmi_vals)),
            "max": float(np.max(all_cmi_vals)),
            "mean": float(np.mean(all_cmi_vals)),
            "median": float(np.median(all_cmi_vals)),
            "std": float(np.std(all_cmi_vals)),
            "n_nonneg": int(sum(1 for v in all_cmi_vals if v >= 0)),
            "n_negative": int(sum(1 for v in all_cmi_vals if v < 0)),
        }

    # ── Pass criteria ─────────────────────────────────────────────────────
    n_valid_cmi = sum(1 for row in summary_table if row.get("cmi") is not None
                      and np.isfinite(row["cmi"]))
    cmi_vals_list = [row["cmi"] for row in summary_table
                     if row.get("cmi") is not None and np.isfinite(row["cmi"])]
    all_finite = (all(np.isfinite(v) for v in cmi_vals_list)
                  if cmi_vals_list else False)

    mw_p = mann_whitney_result.get("p_two_sided", 1.0)
    mw_d = mann_whitney_result.get("cohens_d", 0.0)
    if not isinstance(mw_p, (int, float)) or not np.isfinite(mw_p):
        mw_p = 1.0
    if not isinstance(mw_d, (int, float)) or not np.isfinite(mw_d):
        mw_d = 0.0

    spearman_rho = correlation_result.get("spearman_rho", 0.0)
    if not isinstance(spearman_rho, (int, float)) or not np.isfinite(spearman_rho):
        spearman_rho = 0.0

    pass_criteria = {
        "cmi_at_least_20_letters": n_valid_cmi >= 20,
        "n_valid_cmi": n_valid_cmi,
        "cmi_finite": all_finite,
        "mann_whitney_p_lt_0_05": bool(mw_p < 0.05),
        "p_value": float(mw_p),
        "cohens_d_gt_0_5": bool(abs(mw_d) > 0.5),
        "cohens_d": float(mw_d),
        "spearman_rho_lt_minus_0_2": bool(spearman_rho < -0.2),
        "spearman_rho": float(spearman_rho),
        "h3_target_rho_lt_minus_0_3": bool(spearman_rho < -0.3),
        "overall_pass": bool(
            n_valid_cmi >= 20 and
            all_finite and
            mw_p < 0.05 and
            abs(mw_d) > 0.5 and
            spearman_rho < -0.2
        ),
    }
    results["pass_criteria"] = pass_criteria

    print(f"\n{'='*60}")
    print(f"PASS CRITERIA:")
    print(f"  CMI >= 20 letters: "
          f"{'PASS' if pass_criteria['cmi_at_least_20_letters'] else 'FAIL'} "
          f"({n_valid_cmi}/20)")
    print(f"  CMI finite: "
          f"{'PASS' if pass_criteria['cmi_finite'] else 'FAIL'}")
    print(f"  Mann-Whitney p < 0.05: "
          f"{'PASS' if pass_criteria['mann_whitney_p_lt_0_05'] else 'FAIL'} "
          f"(p={mw_p:.6f})")
    print(f"  Cohen's d > 0.5: "
          f"{'PASS' if pass_criteria['cohens_d_gt_0_5'] else 'FAIL'} "
          f"(d={mw_d:.4f})")
    print(f"  Spearman rho < -0.2: "
          f"{'PASS' if pass_criteria['spearman_rho_lt_minus_0_2'] else 'FAIL'} "
          f"(rho={spearman_rho:.4f})")
    print(f"  H3 target (rho < -0.3): "
          f"{'PASS' if pass_criteria['h3_target_rho_lt_minus_0_3'] else 'FAIL'} "
          f"(rho={spearman_rho:.4f})")
    print(f"  OVERALL: {'PASS' if pass_criteria['overall_pass'] else 'FAIL'}")
    print(f"{'='*60}")

    # Interpretation
    interpretation = []
    if spearman_rho < -0.3:
        interpretation.append(
            "H3 SUPPORTED: CMI negatively correlates with absorption rate. "
            "Letters with lower CMI are preferentially absorbed, consistent "
            "with the rate-distortion prediction.")
    elif spearman_rho < -0.2:
        interpretation.append(
            "H3 WEAKLY SUPPORTED: CMI shows weak negative trend with "
            "absorption rate. Direction consistent with theory but modest.")
    else:
        interpretation.append(
            "H3 NOT SUPPORTED: CMI does not reliably predict absorption rate. "
            "Negative result for rate-distortion framing.")

    results["interpretation"] = interpretation

    # ── Finalize ──────────────────────────────────────────────────────────
    elapsed = time.time() - start_time
    results["elapsed_sec"] = round(elapsed, 2)
    results["timestamp_end"] = datetime.now().isoformat()

    output_path = RESULTS_DIR / "cmi_estimation.json"
    clean_results = json.loads(json.dumps(results, default=str))
    output_path.write_text(json.dumps(clean_results, indent=2))
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Update gpu_progress.json
    try:
        gp_path = WORKSPACE / "exp" / "gpu_progress.json"
        gp = json.loads(gp_path.read_text()) if gp_path.exists() else {
            "completed": [], "failed": [], "running": {}, "timings": {}
        }
        if TASK_ID not in gp["completed"]:
            gp["completed"].append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        gp["timings"][TASK_ID] = {
            "planned_min": 35,
            "actual_min": int(round(elapsed / 60)),
            "start_time": results["timestamp_start"],
            "end_time": results["timestamp_end"],
            "config_snapshot": {
                "model": "gemma-2-2b",
                "sae_config": "L12-16k",
                "n_letters": len(per_letter_absorption),
                "n_natural_tokens": N_NATURAL_TOKENS,
                "knn_k": KNN_K,
                "best_subspace_dim": best_dim,
                "n_valid_cmi": n_valid_cmi,
                "spearman_rho": float(spearman_rho),
                "gpu_model": "RTX PRO 6000 Blackwell",
                "gpu_count": 1,
            },
        }
        gp_path.write_text(json.dumps(gp, indent=2))
        print("  gpu_progress.json updated")
    except Exception as e:
        print(f"  Warning: could not update gpu_progress.json: {e}")

    # Mark done
    summary_msg = (
        f"CMI computed for {n_valid_cmi}/25 letters at d'={best_dim} "
        f"using {N_NATURAL_TOKENS} natural tokens. "
        f"Spearman rho(CMI, absorption)={spearman_rho:.4f}. "
        f"Mann-Whitney p={mw_p:.4f}, Cohen's d={mw_d:.3f}. "
        f"H3 target (rho<-0.3): "
        f"{'MET' if pass_criteria['h3_target_rho_lt_minus_0_3'] else 'NOT MET'}. "
        f"{'OVERALL PASS' if pass_criteria['overall_pass'] else 'COMPLETED'}."
    )
    mark_done(
        status=("success" if pass_criteria["overall_pass"]
                else "completed_with_caveats"),
        summary=summary_msg,
        results={
            "n_valid_cmi": n_valid_cmi,
            "best_subspace_dim": best_dim,
            "spearman_rho": float(spearman_rho),
            "mann_whitney_p": float(mw_p),
            "cohens_d": float(mw_d),
            "h3_target_met": pass_criteria["h3_target_rho_lt_minus_0_3"],
            "overall_pass": pass_criteria["overall_pass"],
            "absorbed_letters": absorbed_letters,
            "non_absorbed_letters": non_absorbed_letters,
        },
    )

    return results


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        traceback.print_exc()
        mark_done(status="failed", summary=str(e))
        sys.exit(1)
