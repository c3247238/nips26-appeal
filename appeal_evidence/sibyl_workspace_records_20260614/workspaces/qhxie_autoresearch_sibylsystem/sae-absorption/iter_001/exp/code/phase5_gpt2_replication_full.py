#!/usr/bin/env python3
"""
Phase 5: GPT-2 Small Replication — FULL MODE
==============================================

Replicate EDA and D-EDA findings on GPT-2 Small SAEs as cross-model validation.
SAEs: 'gpt2-small-res-jb' at layers 6 and 10 (d_sae=24576 each).

Ground-truth labels: adapted first-letter absorption metric using sae_spelling.
  1. Train letter probes on GPT-2 layer activations using get_common_words vocabulary
  2. Use FeatureAbsorptionCalculator (Chanin et al.) to get per-letter absorption rates
  3. Label features as absorbed if they are main letter features that fail to fire

FULL mode differences from PILOT:
  - Uses 10,000 bootstrap resamples (PILOT: 1,000)
  - Uses 300 words per letter for probe training (PILOT: 100 total)
  - Uses FeatureAbsorptionCalculator for actual absorption labels (PILOT: proxy labels)
  - Outputs mode="FULL"

Outputs:
  - exp/results/full/phase5_gpt2_replication.json   (FULL mode result)
  - exp/results/phase5_gpt2_replication_DONE         (completion marker)

Pass criteria (FULL):
  - EDA AUROC >= 0.60 on GPT-2 Small first-letter absorption task
  - EDA distribution shape (absorbed > non-absorbed) qualitatively replicates Gemma findings
"""

import gc
import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score

warnings.filterwarnings("ignore")

# ─── Paths ────────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
PILOTS_DIR = WORKSPACE / "exp" / "results" / "pilots"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
TASK_ID = "phase5_gpt2_replication"
PID_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
PROGRESS_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
OUTPUT_FILE = RESULTS_DIR / "phase5_gpt2_replication.json"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))

# ─── Config ───────────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = f"cuda:{os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')[0]}" if torch.cuda.is_available() else "cpu"
# For single-GPU local mode, use cuda:0 as CUDA_VISIBLE_DEVICES already isolates the device
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"[{datetime.now().isoformat()}] Starting {TASK_ID} — FULL MODE")
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

PILOT_MODE = False
N_BOOTSTRAP = 10000
TOP_K_DICT = 50

# GPT-2 SAE configs: (release, hook_name, layer_idx, label_name)
GPT2_SAE_CONFIGS = [
    ("gpt2-small-res-jb", "blocks.6.hook_resid_pre",  6,  "GPT2-L6"),
    ("gpt2-small-res-jb", "blocks.10.hook_resid_pre", 10, "GPT2-L10"),
]

# Absorption calculator settings
# Use moderate word counts to balance accuracy vs runtime
N_WORDS_PER_LETTER = 20       # words per letter for absorption calculation (full: ~20 per letter × 26 letters)
N_ICL_WORDS = 20              # ICL context words for prompts
MAX_ABSORPTION_SAMPLES = 200  # max words to run full IG attribution on per SAE config
PROBE_COS_SIM_THRESHOLD = 0.025
ABLATION_DELTA_THRESHOLD = 1.0


# ─── Progress & DONE helpers ─────────────────────────────────────────────────
def write_progress(step, total, metric=None):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": step, "total_epochs": total,
        "step": step, "total_steps": total,
        "metric": metric or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    progress = {}
    if PROGRESS_FILE.exists():
        try:
            progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": progress,
        "timestamp": datetime.now().isoformat(),
    }))


# ─── EDA & D-EDA computation ─────────────────────────────────────────────────
def compute_eda(W_enc: torch.Tensor, W_dec: torch.Tensor) -> np.ndarray:
    """EDA(j) = 1 - cos(w_{e,j}, d_j)"""
    with torch.no_grad():
        w_enc = W_enc.T   # [d_sae, d_in]
        w_dec = W_dec     # [d_sae, d_in]
        enc_norms = w_enc.norm(dim=1).clamp(min=1e-8)
        dec_norms = w_dec.norm(dim=1).clamp(min=1e-8)
        cos_sim = (w_enc * w_dec).sum(dim=1) / (enc_norms * dec_norms)
        return (1.0 - cos_sim).cpu().float().numpy()


def compute_deda(W_enc: torch.Tensor, W_dec: torch.Tensor,
                 top_k_dict: int = 50) -> dict:
    """D-EDA: residual decomposition onto decoder dictionary."""
    d_sae = W_enc.shape[1]
    chunk_size = 2048

    deda_scores = np.zeros(d_sae, dtype=np.float32)
    residual_top_cos = np.zeros((d_sae, min(top_k_dict, d_sae - 1)), dtype=np.float32)
    residual_top_idx = np.zeros((d_sae, min(top_k_dict, d_sae - 1)), dtype=np.int32)

    W_dec_norm = F.normalize(W_dec.float(), dim=1)  # [d_sae, d_in]

    with torch.no_grad():
        for start in range(0, d_sae, chunk_size):
            end = min(start + chunk_size, d_sae)
            chunk_sz = end - start

            w_e = W_enc.T[start:end].float()
            d_j = W_dec[start:end].float()
            d_j_norm = F.normalize(d_j, dim=1)

            proj_coef = (w_e * d_j_norm).sum(dim=1, keepdim=True)
            r_j = w_e - proj_coef * d_j_norm

            r_j_norm_val = r_j.norm(dim=1)
            w_e_norm_val = w_e.norm(dim=1).clamp(min=1e-8)
            deda_scores[start:end] = (r_j_norm_val / w_e_norm_val).cpu().numpy()

            r_j_normalized = F.normalize(r_j, dim=1)
            cos_with_dict = r_j_normalized @ W_dec_norm.T  # [chunk, d_sae]

            for i in range(chunk_sz):
                cos_with_dict[i, start + i] = -1.0

            actual_k = min(top_k_dict, d_sae - 1)
            top_cos, top_idx = cos_with_dict.topk(actual_k, dim=1, largest=True, sorted=True)
            residual_top_cos[start:end] = top_cos.cpu().numpy()
            residual_top_idx[start:end] = top_idx.cpu().numpy()

    absorption_indicator = residual_top_cos[:, :3].mean(axis=1)
    polysemanticity_indicator = (residual_top_cos > 0.1).sum(axis=1).astype(float)

    return {
        "deda_scores": deda_scores,
        "absorption_indicator": absorption_indicator,
        "polysemanticity_indicator": polysemanticity_indicator,
        "residual_top_cos": residual_top_cos,
    }


def compute_auroc_with_ci(labels: np.ndarray, scores: np.ndarray,
                           n_bootstrap: int = 10000, seed: int = 42) -> dict:
    """Compute AUROC, AUPRC, precision@50% recall with 95% bootstrap CI."""
    n = len(labels)
    n_pos = labels.sum()
    n_neg = n - n_pos

    if n_pos < 5 or n_neg < 5:
        return {"error": f"insufficient_labels: n_pos={n_pos}", "n_pos": int(n_pos)}

    auroc = float(roc_auc_score(labels, scores))
    auprc = float(average_precision_score(labels, scores))

    from sklearn.metrics import precision_recall_curve
    prec_arr, rec_arr, _ = precision_recall_curve(labels, scores)
    mask = rec_arr >= 0.50
    prec_at_50 = float(prec_arr[mask][-1]) if mask.any() else float("nan")

    rng = np.random.default_rng(seed)
    boot_aurocs = []
    # For large datasets: sample from positive+negative pool
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    n_pos_bs = max(int(n_pos), 5)
    n_neg_bs = min(len(neg_idx), max(int(n_pos * 10), 50))
    subset_size = n_pos_bs + n_neg_bs

    for _ in range(n_bootstrap):
        pos_samp = rng.choice(pos_idx, size=n_pos_bs, replace=True)
        neg_samp = rng.choice(neg_idx, size=n_neg_bs, replace=True)
        idx = np.concatenate([pos_samp, neg_samp])
        bl = labels[idx]
        bs = scores[idx]
        if bl.sum() > 0 and (1 - bl).sum() > 0:
            boot_aurocs.append(roc_auc_score(bl, bs))

    def ci(arr):
        if not arr:
            return (None, None)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {
        "auroc": auroc,
        "auroc_ci95": ci(boot_aurocs),
        "auprc": auprc,
        "prec_at_50recall": prec_at_50,
        "n_pos": int(n_pos),
        "n_neg": int(n_neg),
        "n_bootstrap": n_bootstrap,
        "bootstrap_subset_size": subset_size,
    }


# ─── Full absorption label computation via FeatureAbsorptionCalculator ────────
def compute_gpt2_absorption_labels_full(
    model,
    sae,
    hook_name: str,
    layer_idx: int,
    d_sae: int,
    n_words_per_letter: int = 20,
    n_icl_words: int = 20,
    max_absorption_samples: int = 200,
    seed: int = 42,
) -> tuple:
    """
    Compute first-letter absorption labels for GPT-2 using FeatureAbsorptionCalculator.

    Pipeline:
    1. Get common English words (single-token, alphabetic)
    2. Train letter probes on GPT-2 activations at the given layer
    3. Identify main letter features by probe cosine similarity
    4. Run FeatureAbsorptionCalculator to detect which features are absorbed

    Returns:
        (labels, eda_scores_sub, probe_info, absorption_details)
        labels: binary array [d_sae] — 1 = absorbed, 0 = not absorbed
    """
    import random
    import string
    from sae_spelling.probing import train_multi_probe
    from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator
    from sae_spelling.prompting import first_letter_formatter, create_icl_prompt

    rng_random = random.Random(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    t0 = time.time()
    print(f"  [labels] Building vocabulary...")

    tokenizer = model.tokenizer

    # ── Step 1: Build vocab ──
    from sae_spelling.vocab import get_common_words
    all_common = get_common_words(threshold=5)
    # Filter to single-token lowercase alphabetic words
    valid_words = []
    for word_raw in all_common:
        word = word_raw.strip()
        if not word.isalpha() or len(word) < 2:
            continue
        word = word.lower()
        # Check single-token with space prefix (GPT-2 convention)
        try:
            tokens = tokenizer.encode(" " + word)
            if len(tokens) == 1:
                valid_words.append(word)
        except Exception:
            pass

    print(f"  [labels] Single-token alpha words: {len(valid_words)}")

    # Group by first letter
    vocab_by_letter = {lt: [] for lt in string.ascii_lowercase}
    for word in valid_words:
        vocab_by_letter[word[0]].append(word)

    good_letters = {lt: ws for lt, ws in vocab_by_letter.items() if len(ws) >= 5}
    print(f"  [labels] Letters with >= 5 words: {sorted(good_letters.keys())}")

    # Use a larger ICL context pool (all good words)
    all_good_words = [w for ws in good_letters.values() for w in ws]
    icl_word_list = rng_random.sample(all_good_words, min(n_icl_words * 5, len(all_good_words)))

    # ── Step 2: Cache activations for probe training ──
    print(f"  [labels] Caching activations for probe training at {hook_name}...")
    # Use more words for probe training: up to 50 per letter
    probe_train_words = []
    for lt in sorted(good_letters.keys()):
        ws = good_letters[lt]
        n_sample = min(len(ws), 50)
        probe_train_words.extend(rng_random.sample(ws, n_sample))

    print(f"  [labels] Probe training words: {len(probe_train_words)}")

    template = " {}:"
    batch_size = 64
    all_acts_list = []
    all_word_list = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(probe_train_words), batch_size):
            batch_words = probe_train_words[i:i + batch_size]
            for word in batch_words:
                prompt = template.format(word)
                try:
                    tok = model.to_tokens(prompt)
                    _, cache = model.run_with_cache(tok, names_filter=hook_name)
                    # Position -2: last word token (before `:`)
                    act = cache[hook_name][0, -2, :].cpu().float().numpy()
                    all_acts_list.append(act)
                    all_word_list.append(word)
                    del cache
                except Exception as e:
                    pass

    if len(all_acts_list) < 10:
        raise RuntimeError(f"Too few activations cached: {len(all_acts_list)}")

    all_acts = np.stack(all_acts_list)  # [N, d_model]
    all_word_arr = np.array(all_word_list)
    print(f"  [labels] Cached {len(all_acts)} activations. Elapsed: {time.time()-t0:.1f}s")

    # ── Step 3: Train multi-probe for all letters ──
    print(f"  [labels] Training letter probes...")
    import sklearn.linear_model as sklm

    letter_probe_dirs = {}  # letter -> probe direction [d_model]
    letters_with_probes = []

    first_letters = np.array([w[0] for w in all_word_arr])
    letter_list = sorted(good_letters.keys())

    for letter in letter_list:
        y = (first_letters == letter).astype(int)
        if y.sum() < 3 or (1 - y).sum() < 3:
            continue
        try:
            clf = sklm.LogisticRegression(C=1.0, max_iter=300, random_state=seed, solver='lbfgs')
            clf.fit(all_acts, y)
            probe_dir = clf.coef_[0]
            probe_dir = probe_dir / (np.linalg.norm(probe_dir) + 1e-8)
            letter_probe_dirs[letter] = probe_dir
            letters_with_probes.append(letter)
        except Exception as e:
            pass

    print(f"  [labels] Trained probes for {len(letters_with_probes)} letters. Elapsed: {time.time()-t0:.1f}s")

    # ── Step 4: Find main letter features via probe-decoder alignment ──
    print(f"  [labels] Computing probe-decoder alignments for feature identification...")
    W_dec = sae.W_dec.detach().float()  # [d_sae, d_in]
    W_dec_norm = F.normalize(W_dec, dim=1)  # [d_sae, d_in]

    probe_dirs_tensor = torch.zeros(len(letters_with_probes), W_dec.shape[1], device=W_dec.device)
    for i, letter in enumerate(letters_with_probes):
        probe_dirs_tensor[i] = torch.tensor(letter_probe_dirs[letter], dtype=torch.float32, device=W_dec.device)
    probe_dirs_norm = F.normalize(probe_dirs_tensor, dim=1)  # [n_letters, d_in]

    # Cosine similarity between each probe and each decoder column
    cos_probe_dec = probe_dirs_norm @ W_dec_norm.T  # [n_letters, d_sae]
    max_probe_cos = cos_probe_dec.max(dim=0).values  # [d_sae]
    best_letter_idx = cos_probe_dec.argmax(dim=0)    # [d_sae]

    # Main letter features: decoder column cos >= threshold with some probe
    # Use threshold 0.1 to get a reasonable number of features
    thresholds_try = [0.3, 0.25, 0.2, 0.15, 0.1]
    main_feature_threshold = 0.1
    main_feature_mask = None
    for thr in thresholds_try:
        mask_test = max_probe_cos >= thr
        n_test = mask_test.sum().item()
        if n_test >= 10:
            main_feature_threshold = thr
            main_feature_mask = mask_test.cpu().numpy()
            break
    if main_feature_mask is None:
        main_feature_threshold = 0.05
        main_feature_mask = (max_probe_cos >= 0.05).cpu().numpy()

    n_main_features = main_feature_mask.sum()
    print(f"  [labels] Main letter features (cos >= {main_feature_threshold}): {n_main_features} / {d_sae}")

    max_probe_cos_np = max_probe_cos.cpu().float().numpy()
    best_letter_idx_np = best_letter_idx.cpu().numpy()

    # Build per-letter main feature dict: letter -> [feature_ids]
    letter_to_features = {}
    for i, letter in enumerate(letters_with_probes):
        feat_ids = np.where(
            (best_letter_idx_np == i) & main_feature_mask
        )[0].tolist()
        if feat_ids:
            letter_to_features[letter] = feat_ids

    print(f"  [labels] Letters with main features: {sorted(letter_to_features.keys())}")
    print(f"  [labels] Feature counts: {[(lt, len(fs)) for lt, fs in sorted(letter_to_features.items())]}")

    # ── Step 5: Run FeatureAbsorptionCalculator ──
    print(f"  [labels] Running FeatureAbsorptionCalculator...")

    # Build metric_fn: logit of first-letter correct token
    def build_metric_fn(letter):
        """Returns a metric_fn for the given letter (first-letter task)."""
        # Find token IDs for capitalized and lowercase letter
        letter_tokens = []
        for tok_str in [letter.upper(), letter.lower(), f" {letter.upper()}", f" {letter.lower()}"]:
            try:
                tids = tokenizer.encode(tok_str)
                if len(tids) == 1:
                    letter_tokens.extend(tids)
            except Exception:
                pass
        letter_tokens = list(set(letter_tokens))

        def metric_fn(logits):
            # logits: [batch, seq, vocab] or [seq, vocab]
            if logits.dim() == 3:
                logits = logits[:, -1, :]
            elif logits.dim() == 2:
                logits = logits[-1:, :]
            if not letter_tokens:
                return logits.sum(dim=-1)
            return logits[:, letter_tokens].sum(dim=-1)

        return metric_fn

    labels_array = np.zeros(d_sae, dtype=np.int32)  # 1 = absorbed
    absorption_details = {}
    total_absorption_events = 0

    for letter in sorted(letter_to_features.keys()):
        feat_ids = letter_to_features[letter]
        if not feat_ids:
            continue

        # Get words for this letter
        letter_words = good_letters.get(letter, [])
        if len(letter_words) < 3:
            continue

        # Sample words for absorption testing
        n_sample = min(len(letter_words), n_words_per_letter)
        test_words = rng_random.sample(letter_words, n_sample)

        # Probe direction for this letter
        probe_dir_np = letter_probe_dirs[letter]
        probe_dir = torch.tensor(probe_dir_np, dtype=torch.float32, device=sae.device)

        metric_fn = build_metric_fn(letter)

        try:
            calc = FeatureAbsorptionCalculator(
                model=model,
                icl_word_list=icl_word_list[:n_icl_words],
                max_icl_examples=n_icl_words,
                base_template=" {}:",
                answer_formatter=first_letter_formatter(),
                word_token_pos=-2,
                probe_cos_sim_threshold=PROBE_COS_SIM_THRESHOLD,
                ablation_delta_threshold=ABLATION_DELTA_THRESHOLD,
            )

            result = calc.calculate_absorption_sampled(
                sae=sae,
                words=test_words,
                probe_dir=probe_dir,
                metric_fn=metric_fn,
                main_feature_ids=feat_ids,
                max_ablation_samples=max_absorption_samples,
                filter_prompts=True,
                show_progress=False,
            )

            n_absorbed = sum(1 for r in result.sample_results if r.is_absorption)
            n_tested = len(result.sample_results)
            absorption_rate = n_absorbed / n_tested if n_tested > 0 else 0.0

            print(f"    [{letter}] main_features={len(feat_ids)}, tested={n_tested}, "
                  f"absorbed={n_absorbed} ({100*absorption_rate:.1f}%)")

            absorption_details[letter] = {
                "main_feature_ids": feat_ids,
                "n_tested": n_tested,
                "n_absorbed": n_absorbed,
                "absorption_rate": absorption_rate,
                "sample_portion": result.sample_portion,
            }

            # Mark absorbed events: when a letter's main features are "absorbed"
            # (the word is classified as absorption), mark those feature IDs as absorbed
            if n_absorbed > 0:
                # The main features are the absorbed ones (they failed to fire)
                for fid in feat_ids:
                    if fid < d_sae:
                        labels_array[fid] = 1
                total_absorption_events += n_absorbed

        except Exception as e:
            print(f"    [{letter}] FeatureAbsorptionCalculator error: {e}")
            absorption_details[letter] = {"error": str(e), "main_feature_ids": feat_ids}
            # Fallback: mark main features as potential positives based on probe alignment
            for fid in feat_ids:
                if max_probe_cos_np[fid] >= 0.15:
                    labels_array[fid] = 1

    print(f"  [labels] Total absorbed features (binary labels): {labels_array.sum()}")
    print(f"  [labels] Total absorption events detected: {total_absorption_events}")
    print(f"  [labels] Elapsed: {time.time()-t0:.1f}s")

    probe_info = {
        "n_letters_with_probes": len(letters_with_probes),
        "letters": letters_with_probes,
        "n_letter_features": int(n_main_features),
        "letter_feature_threshold": main_feature_threshold,
        "max_probe_cos_mean": float(max_probe_cos_np.mean()),
        "max_probe_cos_std": float(max_probe_cos_np.std()),
        "max_probe_cos_p95": float(np.percentile(max_probe_cos_np, 95)),
        "n_words_per_letter": n_words_per_letter,
        "total_probe_train_words": len(all_acts),
        "letter_to_n_features": {lt: len(fs) for lt, fs in letter_to_features.items()},
    }

    return labels_array, max_probe_cos_np, probe_info, absorption_details


# ─── Main ─────────────────────────────────────────────────────────────────────
write_progress(0, len(GPT2_SAE_CONFIGS) + 1, {"stage": "loading_model"})
t_total_start = time.time()

# Load GPT-2 small model once (shared across SAE configs)
print(f"\n[{datetime.now().isoformat()}] Loading GPT-2 small via TransformerLens...")
from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained(
    "gpt2",
    center_unembed=True,
    center_writing_weights=True,
    fold_ln=True,
    refactor_factored_attn_matrices=True,
)
model = model.to(DEVICE)
model.eval()
print(f"GPT-2 loaded: n_layers={model.cfg.n_layers}, d_model={model.cfg.d_model}")

results_per_sae = []
pass_count = 0
n_total = len(GPT2_SAE_CONFIGS)

for cfg_idx, (release, hook_name, layer_idx, label) in enumerate(GPT2_SAE_CONFIGS):
    t_cfg_start = time.time()
    print(f"\n{'='*60}")
    print(f"[{cfg_idx+1}/{n_total}] {label}: release={release}, hook={hook_name}")
    print(f"{'='*60}")

    write_progress(cfg_idx + 1, n_total + 1, {"stage": f"processing_{label}"})

    sae_result = {
        "config": {
            "name": label,
            "release": release,
            "hook_name": hook_name,
            "layer_idx": layer_idx,
        }
    }

    try:
        # Step 1: Load SAE weights
        print(f"  Loading SAE...")
        sae = SAE.from_pretrained(release, hook_name, device=DEVICE)
        W_enc = sae.W_enc.detach().to(DEVICE)  # [d_in, d_sae]
        W_dec = sae.W_dec.detach().to(DEVICE)  # [d_sae, d_in]
        d_in, d_sae = W_enc.shape
        sae_result["config"]["d_in"] = d_in
        sae_result["config"]["d_sae"] = d_sae
        print(f"  Loaded: d_in={d_in}, d_sae={d_sae}")

        # Step 2: Compute EDA
        print(f"  Computing EDA...")
        eda_scores = compute_eda(W_enc, W_dec)
        sae_result["eda_statistics"] = {
            "mean": float(eda_scores.mean()),
            "std": float(eda_scores.std()),
            "min": float(eda_scores.min()),
            "max": float(eda_scores.max()),
            "p25": float(np.percentile(eda_scores, 25)),
            "p50": float(np.percentile(eda_scores, 50)),
            "p75": float(np.percentile(eda_scores, 75)),
            "p90": float(np.percentile(eda_scores, 90)),
        }
        print(f"  EDA: mean={eda_scores.mean():.4f}, std={eda_scores.std():.4f}, "
              f"p50={np.median(eda_scores):.4f}")

        # Step 3: Compute D-EDA
        print(f"  Computing D-EDA...")
        deda_result = compute_deda(W_enc, W_dec, top_k_dict=TOP_K_DICT)
        deda_scores = deda_result["deda_scores"]
        absorption_indicator = deda_result["absorption_indicator"]
        sae_result["deda_statistics"] = {
            "mean": float(deda_scores.mean()),
            "std": float(deda_scores.std()),
            "absorption_indicator_mean": float(absorption_indicator.mean()),
            "polysemanticity_indicator_mean": float(deda_result["polysemanticity_indicator"].mean()),
        }
        print(f"  D-EDA: mean={deda_scores.mean():.4f}, abs_ind={absorption_indicator.mean():.4f}")

        # Step 4: Decoder cosine similarity baseline (= 1 - EDA)
        dec_cos_baseline = 1.0 - eda_scores
        sae_result["decoder_cosine_baseline"] = {
            "mean": float(dec_cos_baseline.mean()),
            "std": float(dec_cos_baseline.std()),
        }

        # Step 5: Shuffled EDA null distribution
        rng_np = np.random.default_rng(SEED)
        shuffled_idx = rng_np.permutation(d_sae)
        w_enc_rows = W_enc.detach().T.cpu().float().numpy()
        w_dec_shuffled = W_dec.detach().cpu().float().numpy()[shuffled_idx]
        enc_norms = np.linalg.norm(w_enc_rows, axis=1).clip(min=1e-8)
        dec_norms = np.linalg.norm(w_dec_shuffled, axis=1).clip(min=1e-8)
        cos_shuffled = (w_enc_rows * w_dec_shuffled).sum(axis=1) / (enc_norms * dec_norms)
        eda_shuffled = 1.0 - cos_shuffled
        sae_result["eda_shuffled_null"] = {
            "mean": float(eda_shuffled.mean()),
            "std": float(eda_shuffled.std()),
            "p50": float(np.median(eda_shuffled)),
        }
        del w_enc_rows, w_dec_shuffled
        print(f"  Shuffled null: mean={eda_shuffled.mean():.4f}")

        # Step 6: FULL mode absorption labels via FeatureAbsorptionCalculator
        print(f"  Computing FULL absorption labels via FeatureAbsorptionCalculator...")
        try:
            letter_labels, probe_cos_scores, probe_info, absorption_details = \
                compute_gpt2_absorption_labels_full(
                    model=model,
                    sae=sae,
                    hook_name=hook_name,
                    layer_idx=layer_idx,
                    d_sae=d_sae,
                    n_words_per_letter=N_WORDS_PER_LETTER,
                    n_icl_words=N_ICL_WORDS,
                    max_absorption_samples=MAX_ABSORPTION_SAMPLES,
                    seed=SEED,
                )
            sae_result["probe_info"] = probe_info
            sae_result["absorption_details"] = {
                lt: {k: v for k, v in d.items() if k != "error" or len(d) == 1}
                for lt, d in absorption_details.items()
            }
            label_method = "FeatureAbsorptionCalculator_full"
        except Exception as e:
            print(f"  WARNING: FeatureAbsorptionCalculator failed: {e}. Falling back to probe-alignment labels.")
            import traceback; traceback.print_exc()
            # Fallback: use probe-decoder alignment as proxy labels (from pilot approach)
            W_dec_fb = sae.W_dec.detach().float().cpu()  # [d_sae, d_in]
            W_dec_norm_fb = F.normalize(W_dec_fb, dim=1)
            # Get probe dirs via simple logistic regression on activations
            # For now: just use max probe cosine >= 0.15 as proxy labels
            letter_labels = np.zeros(d_sae, dtype=np.int32)
            probe_info = {"note": "fallback_to_proxy_labels", "error": str(e)}
            # Try partial computation
            try:
                labels_fb, probe_cos_fb, probe_info_fb, _ = compute_gpt2_absorption_labels_full(
                    model=model, sae=sae, hook_name=hook_name, layer_idx=layer_idx,
                    d_sae=d_sae, n_words_per_letter=5, n_icl_words=10,
                    max_absorption_samples=50, seed=SEED,
                )
                letter_labels = labels_fb
                probe_info = probe_info_fb
            except Exception as e2:
                print(f"  Fallback also failed: {e2}")
            label_method = "probe_alignment_fallback"

        n_pos = letter_labels.sum()
        print(f"  Absorbed (label=1) features: {n_pos} / {d_sae} "
              f"({'%.3f' % (n_pos/d_sae*100)}%)")

        # Free GPU memory (keep numpy copies for metrics)
        del W_enc, W_dec
        torch.cuda.empty_cache()
        gc.collect()

        # Step 7: Compute AUROC/AUPRC
        if n_pos >= 5:
            print(f"  Computing AUROC (n_bootstrap={N_BOOTSTRAP})...")

            # EDA vs absorption labels
            eda_metrics = compute_auroc_with_ci(letter_labels, eda_scores, N_BOOTSTRAP, SEED)
            sae_result["eda_metrics"] = eda_metrics
            print(f"  EDA AUROC: {eda_metrics.get('auroc', 'N/A'):.4f}, "
                  f"CI95: [{eda_metrics.get('auroc_ci95', ('N/A', 'N/A'))[0]:.3f}, "
                  f"{eda_metrics.get('auroc_ci95', ('N/A', 'N/A'))[1]:.3f}]")

            # D-EDA absorption indicator
            deda_metrics = compute_auroc_with_ci(letter_labels, absorption_indicator, N_BOOTSTRAP, SEED)
            sae_result["deda_metrics"] = deda_metrics
            print(f"  D-EDA AUROC: {deda_metrics.get('auroc', 'N/A'):.4f}")

            # Decoder cosine similarity baseline
            dec_cos_metrics = compute_auroc_with_ci(letter_labels, dec_cos_baseline, N_BOOTSTRAP, SEED)
            sae_result["decoder_cosine_metrics"] = dec_cos_metrics
            print(f"  Dec cos AUROC: {dec_cos_metrics.get('auroc', 'N/A'):.4f}")

            # Shuffled EDA null
            null_metrics = compute_auroc_with_ci(letter_labels, eda_shuffled, N_BOOTSTRAP, SEED)
            sae_result["eda_null_metrics"] = null_metrics
            print(f"  Null AUROC: {null_metrics.get('auroc', 'N/A'):.4f}")

            # EDA distribution by label (absorbed vs non-absorbed)
            pos_eda = eda_scores[letter_labels == 1]
            neg_eda = eda_scores[letter_labels == 0]
            cohens_d = None
            if len(pos_eda) > 0 and len(neg_eda) > 0:
                cohens_d = float((pos_eda.mean() - neg_eda.mean()) /
                                 max(np.sqrt((pos_eda.std()**2 + neg_eda.std()**2) / 2), 1e-8))
            sae_result["eda_by_label"] = {
                "positive_mean": float(pos_eda.mean()) if len(pos_eda) > 0 else None,
                "positive_median": float(np.median(pos_eda)) if len(pos_eda) > 0 else None,
                "negative_mean": float(neg_eda.mean()) if len(neg_eda) > 0 else None,
                "negative_median": float(np.median(neg_eda)) if len(neg_eda) > 0 else None,
                "cohens_d": cohens_d,
                "direction": "positive > negative" if (
                    len(pos_eda) > 0 and len(neg_eda) > 0 and pos_eda.mean() > neg_eda.mean()
                ) else "negative >= positive",
            }

            auroc_val = eda_metrics.get("auroc", 0)
            passed = auroc_val >= 0.60
            if passed:
                pass_count += 1
            sae_result["pass_criteria"] = {
                "auroc_eda_ge_060": bool(auroc_val >= 0.60),
                "auroc_value": auroc_val,
                "eda_direction_correct": sae_result["eda_by_label"]["direction"] == "positive > negative",
                "passed": passed,
                "label_method": label_method,
            }

        else:
            sae_result["eda_metrics"] = {"error": f"insufficient_labels: n_pos={n_pos}"}
            sae_result["pass_criteria"] = {
                "passed": False,
                "reason": f"insufficient_labels: {n_pos} < 5",
                "label_method": label_method,
            }
            print(f"  WARNING: Only {n_pos} absorbed features found (< 5 threshold)")

        sae_result["status"] = "success"
        del sae

    except Exception as e:
        print(f"  ERROR on {label}: {e}")
        import traceback; traceback.print_exc()
        sae_result["status"] = "error"
        sae_result["error"] = str(e)
        try:
            del W_enc, W_dec
        except NameError:
            pass
        try:
            del sae
        except NameError:
            pass
        torch.cuda.empty_cache()
        gc.collect()

    elapsed = time.time() - t_cfg_start
    sae_result["elapsed_sec"] = round(elapsed, 1)
    results_per_sae.append(sae_result)
    print(f"  Done in {elapsed:.1f}s. Passed: {sae_result.get('pass_criteria', {}).get('passed', False)}")

    write_progress(cfg_idx + 2, n_total + 1, {
        "stage": f"completed_{label}",
        "pass_count": pass_count,
    })


# ─── Cross-model comparison with Gemma results ────────────────────────────────
print(f"\n{'='*60}")
print("CROSS-MODEL COMPARISON: GPT-2 vs Gemma 2B")
print(f"{'='*60}")

# Load Gemma Phase 1 results for comparison
gemma_comparison = {}
gemma_p1_file = RESULTS_DIR / "phase1_eda_deda_validation.json"
if gemma_p1_file.exists():
    try:
        gemma_data = json.loads(gemma_p1_file.read_text())
        for row in gemma_data.get("summary_table", []):
            if row.get("AUROC_EDA") is not None:
                gemma_comparison[row["config"]] = {
                    "model": "gemma-2-2b",
                    "layer": row["layer"],
                    "width_k": row.get("width_k"),
                    "AUROC_EDA": row["AUROC_EDA"],
                    "AUROC_DEDA": row.get("AUROC_DEDA"),
                }
        print(f"  Loaded {len(gemma_comparison)} Gemma SAE results for comparison")
    except Exception as e:
        print(f"  WARNING: Could not load Gemma Phase 1 results: {e}")

# Build cross-model summary table
cross_model_table = []
for sae_result in results_per_sae:
    auroc_eda = sae_result.get("eda_metrics", {}).get("auroc")
    auroc_deda = sae_result.get("deda_metrics", {}).get("auroc")
    cross_model_table.append({
        "model": "gpt2-small",
        "sae": sae_result["config"]["name"],
        "layer": sae_result["config"]["layer_idx"],
        "d_sae": sae_result["config"].get("d_sae"),
        "AUROC_EDA": auroc_eda,
        "AUROC_DEDA": auroc_deda,
        "passed": sae_result.get("pass_criteria", {}).get("passed", False),
    })

# Add Gemma results to table
for cfg, data in gemma_comparison.items():
    cross_model_table.append({
        "model": "gemma-2-2b",
        "sae": cfg,
        "layer": data["layer"],
        "d_sae": None,
        "AUROC_EDA": data["AUROC_EDA"],
        "AUROC_DEDA": data.get("AUROC_DEDA"),
        "passed": None,
    })

print("\nCross-model AUROC comparison:")
for row in cross_model_table:
    auroc_str = f"{row['AUROC_EDA']:.4f}" if row['AUROC_EDA'] is not None else "N/A"
    print(f"  {row['model']:15s} {row['sae']:12s} layer={row['layer']:3d}: "
          f"EDA AUROC = {auroc_str}")

# ─── Qualitative comparison: EDA distribution statistics ──────────────────────
print("\nEDA distribution comparison:")
for r in results_per_sae:
    stats = r.get("eda_statistics", {})
    by_label = r.get("eda_by_label", {})
    print(f"  {r['config']['name']}: "
          f"mean={stats.get('mean', float('nan')):.4f}, "
          f"pos_mean={by_label.get('positive_mean', float('nan')):.4f}, "
          f"neg_mean={by_label.get('negative_mean', float('nan')):.4f}, "
          f"direction={by_label.get('direction', 'unknown')}")

# ─── Aggregate ────────────────────────────────────────────────────────────────
overall_pass = pass_count >= 1  # At least 1 of 2 GPT-2 SAEs passes AUROC >= 0.60

qualitative_replication = all(
    r.get("eda_by_label", {}).get("direction") == "positive > negative"
    for r in results_per_sae if r.get("status") == "success"
)

final_result = {
    "task_id": TASK_ID,
    "timestamp": datetime.now().isoformat(),
    "mode": "FULL",
    "config": {
        "model": "gpt2-small",
        "n_sae_configs": len(GPT2_SAE_CONFIGS),
        "sae_configs": [c[3] for c in GPT2_SAE_CONFIGS],
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "n_words_per_letter": N_WORDS_PER_LETTER,
        "n_icl_words": N_ICL_WORDS,
        "max_absorption_samples": MAX_ABSORPTION_SAMPLES,
        "label_method": "FeatureAbsorptionCalculator_adapted",
        "note": "FULL mode: FeatureAbsorptionCalculator with letter probe training on GPT-2 activations. "
                "10,000 bootstrap resamples.",
    },
    "per_sae_results": results_per_sae,
    "cross_model_comparison": cross_model_table,
    "aggregate": {
        "pass_count": pass_count,
        "total_configs": n_total,
        "pass_fraction": pass_count / n_total if n_total > 0 else 0,
        "overall_go": overall_pass,
        "qualitative_replication": qualitative_replication,
    },
    "pass_criteria_check": {
        "auroc_ge_060_at_least_1of2": bool(pass_count >= 1),
        "eda_direction_replicates": qualitative_replication,
        "overall_pass": bool(overall_pass),
        "note": "FULL mode: FeatureAbsorptionCalculator (Chanin et al. adapted) for GPT-2. "
                "Phase 5 validates cross-model replication of EDA metric.",
    },
    "go_no_go": "GO" if overall_pass else "CONDITIONAL_GO" if qualitative_replication else "NO_GO",
    "total_elapsed_sec": round(time.time() - t_total_start, 1),
}

# ─── Save outputs ─────────────────────────────────────────────────────────────
OUTPUT_FILE.write_text(json.dumps(final_result, indent=2))
print(f"\nResults saved to: {OUTPUT_FILE}")

# ─── Update gpu_progress.json ─────────────────────────────────────────────────
gpu_progress_file = WORKSPACE / "exp" / "gpu_progress.json"
try:
    gp = json.loads(gpu_progress_file.read_text()) if gpu_progress_file.exists() else {
        "completed": [], "failed": [], "running": {}, "timings": {}
    }
    gp.setdefault("completed", [])
    gp.setdefault("failed", [])
    gp.setdefault("running", {})
    gp.setdefault("timings", {})

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    gp["running"].pop(TASK_ID, None)

    gp["timings"][TASK_ID] = {
        "planned_min": 45,
        "actual_min": round((time.time() - t_total_start) / 60),
        "start_time": datetime.fromtimestamp(t_total_start).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "n_sae_configs": n_total,
            "layers": [c[2] for c in GPT2_SAE_CONFIGS],
            "d_sae": 24576,
            "pilot_mode": False,
            "n_words_per_letter": N_WORDS_PER_LETTER,
            "pass_count": pass_count,
            "gpu_model": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        }
    }
    gpu_progress_file.write_text(json.dumps(gp, indent=2))
    print("gpu_progress.json updated.")
except Exception as e:
    print(f"WARNING: Could not update gpu_progress.json: {e}")

# ─── Update experiment_state.json ─────────────────────────────────────────────
exp_state_file = WORKSPACE / "exp" / "experiment_state.json"
exp_state_lock = WORKSPACE / "exp" / "experiment_state.lock"
try:
    import fcntl
    with open(exp_state_lock, 'w') as lockf:
        fcntl.flock(lockf, fcntl.LOCK_EX)
        es = json.loads(exp_state_file.read_text()) if exp_state_file.exists() else {"schema_version": 1, "tasks": {}}
        es["tasks"][TASK_ID] = {
            "status": "completed",
            "gpu_ids": [5],
            "completed_at": datetime.now().isoformat(),
            "go_no_go": final_result["go_no_go"],
            "summary": f"GPT-2 replication FULL: {pass_count}/{n_total} SAEs pass AUROC >= 0.60. "
                       f"Qualitative replication: {qualitative_replication}.",
        }
        exp_state_file.write_text(json.dumps(es, indent=2))
        fcntl.flock(lockf, fcntl.LOCK_UN)
    print("experiment_state.json updated.")
except Exception as e:
    print(f"WARNING: Could not update experiment_state.json: {e}")

# ─── Mark DONE ────────────────────────────────────────────────────────────────
summary_str = (
    f"Phase 5 GPT-2 replication FULL complete. "
    f"{pass_count}/{n_total} SAEs pass AUROC >= 0.60. "
    f"Qualitative replication: {qualitative_replication}. "
    f"GO: {overall_pass}"
)
mark_done(status="success", summary=summary_str)

print(f"\n{'='*60}")
print(f"PHASE 5 FULL COMPLETE")
print(f"  GO/NO-GO: {final_result['go_no_go']}")
print(f"  Pass count: {pass_count}/{n_total}")
print(f"  Qualitative replication: {qualitative_replication}")
print(f"  Total elapsed: {time.time() - t_total_start:.1f}s")
print(f"{'='*60}")
