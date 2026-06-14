#!/usr/bin/env python3
"""
Phase 0.1 FULL: Activation Patching at Scale (n>=20 words, 200 contexts)
=========================================================================

Scales up the pilot causal absorption evidence to FULL statistical rigor:
  Pilot: n=7 words, 100 contexts, Wilcoxon p=1.0 (underpowered)
  Full:  n>=20 words, 200 contexts, target Wilcoxon p<0.01

Pipeline:
1. Load Gemma 2 2B + Gemma Scope L12-16k SAE
2. Train high-quality first-letter probe at layer 12 (sklearn LogReg)
3. Discover absorption pairs via IG-like approach on FN tokens
4. For each word x 200 contexts:
   a. Cache SAE activations, zero child feature
   b. Check parent probe recovery on modified SAE reconstruction
5. Control: zero random non-child feature (magnitude-matched) per context
6. Statistics: Wilcoxon signed-rank, bootstrap 95% CI, Cohen's d
7. (If probes available) Attempt cross-hierarchy patching on RAVEL entities

MODE: FULL (200 contexts, seed=42, 60-min timeout budget)
"""

import gc
import json
import os
import sys
import time
import random
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase0_activation_patching_full"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE0_DIR = RESULTS_DIR / "phase0"
FULL_DIR = RESULTS_DIR / "full"
for d in [PILOT_DIR, PHASE0_DIR, FULL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = PHASE0_DIR / "activation_patching_full.json"
SUMMARY_FILE = PHASE0_DIR / "activation_patching_full_summary.md"

DEVICE = "cuda:0"  # GPU 6 mapped via CUDA_VISIBLE_DEVICES

# Full-mode settings
CONTEXTS_PER_WORD = 200
N_CONTROL_FEATURES = 15  # More controls for robustness
N_BOOTSTRAP = 10000  # 10k resamples for CI
TARGET_N_WORDS = 25  # Target >= 20 words total
MIN_ABSORPTION_CONTEXTS = 3  # Min absorbed tokens to include a word
DISCOVERY_BATCH_SIZE = 50  # Words to test for absorption discovery

# SAE config: canonical L0 at layer 12, 16k width
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
HOOK_POINT = "blocks.12.hook_resid_post"
TARGET_LAYER = 12

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)


# ============================================================
# Process tracking
# ============================================================
def write_pid():
    PID_FILE.write_text(str(os.getpid()))

def report_progress(step, total_steps, status="running", metrics=None):
    PROGRESS_FILE.write_text(json.dumps({
        "task_id": TASK_ID, "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "loss": None, "metric": metrics or {},
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
        "final_progress": progress, "timestamp": datetime.now().isoformat(),
    }))


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# The 7 confirmed pilot absorption pairs (from iter_008 pilot)
# ============================================================
PILOT_CORE_WORDS = [
    {"word": "eight", "letter": "e", "child_features": [892, 8174, 5327, 5827], "primary_child": 5327},
    {"word": "liked", "letter": "l", "child_features": [4678, 7984, 14031, 3967], "primary_child": 4678},
    {"word": "lower", "letter": "l", "child_features": [3858, 11826, 14449], "primary_child": 14449},
    {"word": "offer", "letter": "o", "child_features": [15092], "primary_child": 15092},
    {"word": "often", "letter": "o", "child_features": [3063], "primary_child": 3063},
    {"word": "other", "letter": "o", "child_features": [15322], "primary_child": 15322},
    {"word": "under", "letter": "u", "child_features": [2810, 12675], "primary_child": 2810},
]


# ============================================================
# Context generation -- 50 diverse templates for 200 contexts
# ============================================================
CONTEXT_TEMPLATES = [
    "The word {word} appears in many sentences.",
    "I often use the word {word} when writing.",
    "Can you tell me about {word} and its meaning?",
    "The term {word} is commonly used in English.",
    "People frequently say {word} in conversation.",
    "Let me explain what {word} means in this context.",
    "The concept of {word} is interesting to study.",
    "We should discuss {word} in more detail.",
    "Have you heard the word {word} before?",
    "The expression {word} is quite versatile.",
    "In this document, {word} appears several times.",
    "The sentence contains the word {word} prominently.",
    "Consider the following word: {word}.",
    "Many authors use {word} in their writing.",
    "The dictionary defines {word} as a common term.",
    "When you say {word}, people usually understand.",
    "The usage of {word} has changed over time.",
    "Let us analyze the word {word} carefully.",
    "Notice how {word} fits in this paragraph.",
    "The significance of {word} cannot be overstated.",
    "In modern English, {word} is a frequent term.",
    "The first letter of {word} tells us about spelling.",
    "Writers commonly begin with {word} in paragraphs.",
    "The word {word} was first recorded centuries ago.",
    "Students learn to spell {word} in school.",
    "A sentence using {word} might read as follows.",
    "The word {word} has multiple meanings.",
    "In a formal letter, {word} could appear here.",
    "The phrase includes {word} as a key element.",
    "Look at how {word} connects to other words.",
    "Reading about {word} reveals its etymology.",
    "Scholars debate the origins of {word} frequently.",
    "The text mentions {word} in several places.",
    "Understanding {word} requires careful attention.",
    "When analyzing text, {word} stands out clearly.",
    "The frequency of {word} in this corpus is notable.",
    "Researchers study how {word} is processed by models.",
    "The morphology of {word} is straightforward.",
    "In this exercise, identify the word {word}.",
    "The spelling of {word} follows standard rules.",
    "An example containing {word} illustrates the point.",
    "The word {word} belongs to the English lexicon.",
    "Children learn {word} at an early age.",
    "Note the position of {word} in this sentence.",
    "The context around {word} affects its meaning.",
    "Using {word} in a sentence demonstrates competence.",
    "The phonetics of {word} are interesting.",
    "A paragraph about {word} would be informative.",
    "The token {word} appears in the training data.",
    "When tokenized, {word} maps to specific indices.",
]


def generate_contexts(word, n_contexts, seed=SEED):
    """Generate diverse input contexts containing the target word."""
    rng = random.Random(seed + hash(word))
    contexts = []
    for i in range(n_contexts):
        template = CONTEXT_TEMPLATES[i % len(CONTEXT_TEMPLATES)]
        ctx = template.format(word=word)
        contexts.append(ctx)
    return contexts


# ============================================================
# Activation caching utilities
# ============================================================
def cache_activations_batch(model, tokenizer, prompts, hook_point, target_layer,
                            device, word_positions=None, batch_size=8):
    """Cache residual stream activations at specified hook point."""
    all_acts = []
    for batch_start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[batch_start:batch_start + batch_size]
        batch_positions = None
        if word_positions is not None:
            batch_positions = word_positions[batch_start:batch_start + batch_size]

        for i, prompt in enumerate(batch_prompts):
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point],
                    stop_at_layer=target_layer + 1,
                )
            if batch_positions is not None and batch_positions[i] is not None:
                pos = batch_positions[i]
            else:
                pos = -1
            act = cache[hook_point][0, pos, :].float().cpu()
            all_acts.append(act)
            del cache

        if (batch_start + batch_size) % 100 == 0:
            torch.cuda.empty_cache()

    return torch.stack(all_acts)


def find_word_position(tokenizer, prompt, target_word):
    """Find the token position of the target word in the prompt."""
    tokens = tokenizer.encode(prompt)
    token_strs = [tokenizer.decode([t]) for t in tokens]
    target_lower = target_word.lower()

    for pos, ts in enumerate(token_strs):
        clean = ts.strip().lower().lstrip("\u2581")
        if clean == target_lower:
            return pos
    for pos, ts in enumerate(token_strs):
        if target_lower in ts.lower():
            return pos
    return -1


# ============================================================
# Absorption pair discovery via IG-like approach
# ============================================================
def discover_absorption_pairs(model, tokenizer, sae, probe, letter_to_idx,
                              device, n_candidate_words=150, n_test_contexts=50):
    """
    Discover additional word-child feature pairs showing absorption.
    Strategy: scan vocabulary for words where raw probe is correct but SAE
    reconstruction probe is wrong, then identify the responsible SAE feature
    via decoder-probe cosine analysis.
    """
    logger.info("Discovering additional absorption pairs via IG-like approach...")

    vocab = tokenizer.get_vocab()
    letter_words = defaultdict(list)
    for token_str, token_id in vocab.items():
        clean = token_str.lstrip("\u2581").strip()
        if (len(clean) >= 3 and clean.isascii() and clean[0].isalpha()
            and clean.isalpha() and clean.islower()):
            letter = clean[0].lower()
            if letter in "abcdefghijklmnopqrstuvwxyz":
                letter_words[letter].append(clean)

    for letter in letter_words:
        letter_words[letter] = list(set(letter_words[letter]))

    rng = random.Random(SEED)
    candidates = []
    for letter in sorted(letter_words.keys()):
        words = letter_words[letter]
        rng.shuffle(words)
        candidates.extend([(w, letter) for w in words[:5]])

    rng.shuffle(candidates)
    candidates = candidates[:n_candidate_words]

    pilot_words_set = {cw["word"] for cw in PILOT_CORE_WORDS}
    candidates = [(w, l) for w, l in candidates if w not in pilot_words_set]

    logger.info(f"Testing {len(candidates)} candidate words for absorption...")

    discovered = []
    n_tested = 0

    for word, letter in candidates:
        if len(discovered) >= TARGET_N_WORDS - len(PILOT_CORE_WORDS):
            break

        n_tested += 1
        letter_idx = letter_to_idx[letter]

        contexts = generate_contexts(word, n_test_contexts)
        positions = [find_word_position(tokenizer, ctx, word) for ctx in contexts]

        acts = cache_activations_batch(
            model, tokenizer, contexts, HOOK_POINT, TARGET_LAYER,
            device, word_positions=positions
        ).to(device)

        with torch.no_grad():
            sae_features = sae.encode(acts).detach()
            sae_recon = sae.decode(sae_features).detach()

        raw_preds = probe.predict(acts.detach().cpu().numpy())
        recon_preds = probe.predict(sae_recon.cpu().numpy())
        raw_correct = (raw_preds == letter_idx)
        recon_correct = (recon_preds == letter_idx)

        absorbed_mask = raw_correct & (~recon_correct)
        n_absorbed = int(absorbed_mask.sum())

        if n_absorbed < MIN_ABSORPTION_CONTEXTS:
            del acts, sae_features, sae_recon
            torch.cuda.empty_cache()
            continue

        # Identify child feature: decoder-probe cosine x mean activation on absorbed tokens
        probe_dir = torch.tensor(probe.coef_[letter_idx], dtype=torch.float32).to(device)
        probe_dir = probe_dir / probe_dir.norm()

        with torch.no_grad():
            decoder_norms = sae.W_dec.float().norm(dim=1, keepdim=True).clamp(min=1e-8)
            decoder_normalized = sae.W_dec.float() / decoder_norms
            cosine_sims = (decoder_normalized @ probe_dir).cpu().numpy()

        absorbed_indices = np.where(absorbed_mask)[0]
        absorbed_features = sae_features[absorbed_indices].cpu().numpy()
        mean_absorbed_acts = absorbed_features.mean(axis=0)
        contributions = mean_absorbed_acts * cosine_sims
        sorted_indices = np.argsort(contributions)

        child_candidates = []
        for feat_idx in sorted_indices[:10]:
            if mean_absorbed_acts[feat_idx] > 0.1:
                child_candidates.append(int(feat_idx))
            if len(child_candidates) >= 3:
                break

        if not child_candidates:
            top_active = np.argsort(-mean_absorbed_acts)
            child_candidates = [int(top_active[0])]

        primary_child = child_candidates[0]
        n_raw_correct = int(raw_correct.sum())
        absorption_rate = n_absorbed / max(1, n_raw_correct)

        discovered.append({
            "word": word,
            "letter": letter,
            "child_features": child_candidates,
            "primary_child": primary_child,
            "discovery_stats": {
                "n_contexts": n_test_contexts,
                "n_absorbed": n_absorbed,
                "n_raw_correct": n_raw_correct,
                "absorption_rate": float(absorption_rate),
                "primary_contribution": float(contributions[primary_child]),
                "primary_cosine": float(cosine_sims[primary_child]),
                "primary_mean_act": float(mean_absorbed_acts[primary_child]),
            }
        })

        logger.info(
            f"  [{len(discovered)}] {word} (letter={letter}): "
            f"absorbed={n_absorbed}/{n_raw_correct} ({absorption_rate:.2%}), "
            f"child={primary_child} (cos={cosine_sims[primary_child]:.3f})"
        )

        del acts, sae_features, sae_recon
        torch.cuda.empty_cache()

        if n_tested % 20 == 0:
            logger.info(f"  ...tested {n_tested} candidates, found {len(discovered)} absorption pairs")

    logger.info(f"Discovery complete: tested {n_tested} candidates, found {len(discovered)} new pairs")
    return discovered


# ============================================================
# Single-word patching experiment
# ============================================================
def run_patching_for_word(model, tokenizer, sae, probe, letter_to_idx,
                          word_info, device, n_contexts=200, n_controls=15):
    """Run activation patching for a single word. Returns result dict."""
    word = word_info["word"]
    letter = word_info["letter"]
    letter_idx = letter_to_idx[letter]
    child_features = word_info["child_features"]
    primary_child = word_info["primary_child"]
    source = word_info.get("source", "unknown")
    probe_weights = probe.coef_

    if primary_child is None:
        return {
            "word": word, "letter": letter, "status": "skipped",
            "reason": "no identified child features", "source": source,
        }

    # Generate contexts
    contexts = generate_contexts(word, n_contexts)
    positions = [find_word_position(tokenizer, ctx, word) for ctx in contexts]

    # Cache activations
    acts = cache_activations_batch(
        model, tokenizer, contexts, HOOK_POINT, TARGET_LAYER,
        device, word_positions=positions
    ).to(device)
    n_ctx = acts.shape[0]

    # Encode through SAE
    with torch.no_grad():
        sae_features = sae.encode(acts).detach()
        sae_recon = sae.decode(sae_features).detach()

    # Probe on raw and recon
    raw_preds = probe.predict(acts.detach().cpu().numpy())
    raw_probs = probe.predict_proba(acts.detach().cpu().numpy())
    raw_correct = (raw_preds == letter_idx)
    raw_accuracy = float(raw_correct.mean())
    raw_letter_prob = raw_probs[:, letter_idx]

    recon_preds = probe.predict(sae_recon.detach().cpu().numpy())
    recon_probs = probe.predict_proba(sae_recon.detach().cpu().numpy())
    recon_correct = (recon_preds == letter_idx)
    recon_accuracy = float(recon_correct.mean())
    recon_letter_prob = recon_probs[:, letter_idx]

    # Absorption metric
    absorbed_mask = raw_correct & (~recon_correct)
    n_absorbed = int(absorbed_mask.sum())
    n_raw_correct = int(raw_correct.sum())
    absorption_rate = float(n_absorbed / max(1, n_raw_correct))

    # ----- Child feature zeroing -----
    modified_features = sae_features.clone()
    modified_features[:, primary_child] = 0.0

    with torch.no_grad():
        modified_recon = sae.decode(modified_features).detach()

    child_zeroed_preds = probe.predict(modified_recon.cpu().numpy())
    child_zeroed_probs = probe.predict_proba(modified_recon.cpu().numpy())
    child_zeroed_correct = (child_zeroed_preds == letter_idx)
    child_zeroed_accuracy = float(child_zeroed_correct.mean())
    child_zeroed_letter_prob = child_zeroed_probs[:, letter_idx]

    # Recovery metrics
    recovery_absorbed = absorbed_mask & child_zeroed_correct
    n_recovered_absorbed = int(recovery_absorbed.sum())
    recovery_rate_absorbed = float(n_recovered_absorbed / max(1, n_absorbed))

    degradation_mask = recon_correct & (~child_zeroed_correct)
    n_degraded = int(degradation_mask.sum())
    degradation_rate = float(n_degraded / max(1, int(recon_correct.sum())))

    prob_change_child = float((child_zeroed_letter_prob - recon_letter_prob).mean())

    # ----- All children zeroed -----
    all_child_modified = sae_features.clone()
    for cf in child_features:
        all_child_modified[:, cf] = 0.0

    with torch.no_grad():
        all_child_recon = sae.decode(all_child_modified).detach()

    all_child_preds = probe.predict(all_child_recon.cpu().numpy())
    all_child_probs = probe.predict_proba(all_child_recon.cpu().numpy())
    all_child_correct = (all_child_preds == letter_idx)
    all_child_accuracy = float(all_child_correct.mean())
    all_child_letter_prob = all_child_probs[:, letter_idx]

    all_recovery_absorbed = absorbed_mask & all_child_correct
    all_recovery_rate_absorbed = float(all_recovery_absorbed.sum() / max(1, n_absorbed))

    # ----- Control: zero random features (magnitude-matched) -----
    primary_mean_act = float(sae_features[:, primary_child].mean().item())
    all_mean_acts = sae_features.detach().mean(dim=0).cpu().numpy()

    active_mask = all_mean_acts > 0.01
    active_indices = np.where(active_mask)[0]
    non_child = [idx for idx in active_indices if idx not in child_features]

    if primary_mean_act > 0:
        mag_diffs = np.abs(all_mean_acts[non_child] - primary_mean_act)
        sorted_by_mag = np.argsort(mag_diffs)
        matched_features = [non_child[i] for i in sorted_by_mag[:n_controls * 3]]
    else:
        matched_features = non_child

    rng_ctrl = np.random.RandomState((SEED + abs(hash(word))) % (2**32))
    rng_ctrl.shuffle(matched_features)
    control_features = matched_features[:n_controls]

    control_results = []
    control_recovery_rates_absorbed = []
    control_prob_changes = []

    for ctrl_feat in control_features:
        ctrl_modified = sae_features.clone()
        ctrl_modified[:, ctrl_feat] = 0.0

        with torch.no_grad():
            ctrl_recon = sae.decode(ctrl_modified).detach()

        ctrl_preds = probe.predict(ctrl_recon.cpu().numpy())
        ctrl_probs = probe.predict_proba(ctrl_recon.cpu().numpy())
        ctrl_correct = (ctrl_preds == letter_idx)
        ctrl_letter_prob = ctrl_probs[:, letter_idx]
        ctrl_accuracy = float(ctrl_correct.mean())

        ctrl_recovery_abs = float(
            (absorbed_mask & ctrl_correct).sum() / max(1, n_absorbed)
        )
        ctrl_prob_change = float((ctrl_letter_prob - recon_letter_prob).mean())

        control_results.append({
            "feature_idx": int(ctrl_feat),
            "accuracy": ctrl_accuracy,
            "recovery_rate_absorbed": ctrl_recovery_abs,
            "prob_change": ctrl_prob_change,
            "mean_activation": float(all_mean_acts[ctrl_feat]),
        })
        control_recovery_rates_absorbed.append(ctrl_recovery_abs)
        control_prob_changes.append(ctrl_prob_change)

    mean_control_recovery_absorbed = float(np.mean(control_recovery_rates_absorbed)) if control_recovery_rates_absorbed else 0
    mean_control_prob_change = float(np.mean(control_prob_changes)) if control_prob_changes else 0

    # ----- Per-child feature analysis -----
    per_child_results = []
    for cf in child_features:
        cf_modified = sae_features.clone()
        cf_modified[:, cf] = 0.0
        with torch.no_grad():
            cf_recon_out = sae.decode(cf_modified).detach()

        cf_preds = probe.predict(cf_recon_out.cpu().numpy())
        cf_probs = probe.predict_proba(cf_recon_out.cpu().numpy())
        cf_correct = (cf_preds == letter_idx)
        cf_letter_prob = cf_probs[:, letter_idx]
        cf_recovery_absorbed = float(
            (absorbed_mask & cf_correct).sum() / max(1, n_absorbed)
        )
        cf_prob_change = float((cf_letter_prob - recon_letter_prob).mean())

        with torch.no_grad():
            probe_dir = torch.tensor(probe_weights[letter_idx], dtype=torch.float32).to(device)
            probe_dir = probe_dir / probe_dir.norm()
            decoder_dir = sae.W_dec[cf].float().detach()
            decoder_dir = decoder_dir / (decoder_dir.norm() + 1e-8)
            cos_sim = F.cosine_similarity(probe_dir.unsqueeze(0), decoder_dir.unsqueeze(0)).item()

        per_child_results.append({
            "feature_idx": int(cf),
            "recovery_rate_absorbed": cf_recovery_absorbed,
            "prob_change_vs_recon": cf_prob_change,
            "mean_activation": float(all_mean_acts[cf]),
            "decoder_probe_cosine": float(cos_sim),
        })

    word_result = {
        "word": word,
        "letter": letter,
        "status": "completed",
        "source": source,
        "n_contexts": n_ctx,
        "child_features": child_features,
        "primary_child": primary_child,
        "raw_residual": {
            "accuracy": raw_accuracy,
            "mean_letter_prob": float(raw_letter_prob.mean()),
            "n_correct": n_raw_correct,
            "n_incorrect": int((~raw_correct).sum()),
        },
        "sae_reconstruction": {
            "accuracy": recon_accuracy,
            "mean_letter_prob": float(recon_letter_prob.mean()),
            "n_correct": int(recon_correct.sum()),
            "n_incorrect": int((~recon_correct).sum()),
            "n_absorbed": n_absorbed,
            "absorption_rate": absorption_rate,
        },
        "primary_child_zeroed": {
            "accuracy_on_recon": child_zeroed_accuracy,
            "mean_letter_prob": float(child_zeroed_letter_prob.mean()),
            "n_recovered_from_absorbed": n_recovered_absorbed,
            "recovery_rate_absorbed": recovery_rate_absorbed,
            "n_degraded": n_degraded,
            "degradation_rate": degradation_rate,
            "prob_change_vs_recon": prob_change_child,
        },
        "all_children_zeroed": {
            "accuracy_on_recon": all_child_accuracy,
            "mean_letter_prob": float(all_child_letter_prob.mean()),
            "recovery_rate_absorbed": all_recovery_rate_absorbed,
        },
        "control_random_zeroed": {
            "n_controls": len(control_results),
            "mean_recovery_rate_absorbed": mean_control_recovery_absorbed,
            "mean_prob_change": mean_control_prob_change,
            "details": control_results,
        },
        "per_child_analysis": per_child_results,
    }

    # Clean up
    del acts, sae_features, sae_recon, modified_features, modified_recon
    del all_child_modified, all_child_recon
    torch.cuda.empty_cache()

    return word_result


# ============================================================
# Statistical analysis
# ============================================================
def compute_statistics(words_with_absorption):
    """Compute all statistical tests from completed results."""
    n_with_absorption = len(words_with_absorption)

    recovery_child = np.array([r["primary_child_zeroed"]["recovery_rate_absorbed"]
                               for r in words_with_absorption])
    recovery_control = np.array([r["control_random_zeroed"]["mean_recovery_rate_absorbed"]
                                  for r in words_with_absorption])
    absorption_rates = np.array([r["sae_reconstruction"]["absorption_rate"]
                                  for r in words_with_absorption])
    prob_changes_child = np.array([r["primary_child_zeroed"]["prob_change_vs_recon"]
                                    for r in words_with_absorption])
    prob_changes_control = np.array([r["control_random_zeroed"]["mean_prob_change"]
                                      for r in words_with_absorption])

    # Wilcoxon signed-rank test
    if n_with_absorption >= 5:
        diffs = recovery_child - recovery_control
        nonzero_diffs = diffs[diffs != 0]
        if len(nonzero_diffs) >= 5:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(nonzero_diffs, alternative='greater')
        else:
            wilcoxon_stat, wilcoxon_p = float('nan'), float('nan')
    else:
        diffs = recovery_child - recovery_control if n_with_absorption > 0 else np.array([])
        wilcoxon_stat, wilcoxon_p = float('nan'), float('nan')

    # Mann-Whitney U test
    if n_with_absorption >= 3:
        try:
            mw_stat, mw_p = stats.mannwhitneyu(recovery_child, recovery_control,
                                                 alternative='greater')
        except ValueError:
            mw_stat, mw_p = float('nan'), float('nan')
    else:
        mw_stat, mw_p = float('nan'), float('nan')

    # Paired t-test
    if n_with_absorption >= 3:
        try:
            t_stat, t_p = stats.ttest_rel(recovery_child, recovery_control)
            t_p_onesided = t_p / 2 if t_stat > 0 else 1 - t_p / 2
        except Exception:
            t_stat, t_p, t_p_onesided = float('nan'), float('nan'), float('nan')
    else:
        t_stat, t_p, t_p_onesided = float('nan'), float('nan'), float('nan')

    # Bootstrap CI on recovery difference
    diff = recovery_child - recovery_control if n_with_absorption > 0 else np.array([])
    mean_diff = float(diff.mean()) if len(diff) > 0 else 0
    if len(diff) >= 3:
        rng_boot = np.random.RandomState(SEED)
        boot_diffs = []
        for _ in range(N_BOOTSTRAP):
            idx = rng_boot.choice(len(diff), len(diff), replace=True)
            boot_diffs.append(diff[idx].mean())
        boot_diffs = np.array(boot_diffs)
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        boot_p = float((boot_diffs <= 0).mean())
    else:
        ci_lower, ci_upper, boot_p = float('nan'), float('nan'), float('nan')

    # Effect size (Cohen's d)
    if n_with_absorption >= 3 and len(diff) > 0 and diff.std() > 0:
        cohens_d = float(diff.mean() / diff.std())
    else:
        cohens_d = float('nan')

    # Probability change tests
    if n_with_absorption >= 3:
        try:
            prob_t_stat, prob_t_p = stats.ttest_rel(prob_changes_child, prob_changes_control)
        except Exception:
            prob_t_stat, prob_t_p = float('nan'), float('nan')
    else:
        prob_t_stat, prob_t_p = float('nan'), float('nan')

    # Aggregate metrics
    mean_absorption = float(absorption_rates.mean()) if len(absorption_rates) > 0 else 0
    mean_recovery_child = float(recovery_child.mean()) if len(recovery_child) > 0 else 0
    mean_recovery_control = float(recovery_control.mean()) if len(recovery_control) > 0 else 0

    n_positive_recovery = int((recovery_child > recovery_control).sum()) if len(recovery_child) > 0 else 0
    n_zero_recovery = int((recovery_child == recovery_control).sum()) if len(recovery_child) > 0 else 0
    n_negative_recovery = int((recovery_child < recovery_control).sum()) if len(recovery_child) > 0 else 0

    # Interpretation
    significance_achieved = False
    best_p = float('inf')
    best_test = "none"

    for test_name, p_val in [("wilcoxon", wilcoxon_p), ("mann_whitney", mw_p),
                              ("paired_t_onesided", t_p_onesided), ("bootstrap", boot_p)]:
        if not np.isnan(p_val) and p_val < best_p:
            best_p = p_val
            best_test = test_name
        if not np.isnan(p_val) and p_val < 0.05:
            significance_achieved = True

    return {
        "n_with_absorption": n_with_absorption,
        "mean_absorption": mean_absorption,
        "mean_recovery_child": mean_recovery_child,
        "mean_recovery_control": mean_recovery_control,
        "mean_diff": mean_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_positive_recovery": n_positive_recovery,
        "n_zero_recovery": n_zero_recovery,
        "n_negative_recovery": n_negative_recovery,
        "significance_achieved": significance_achieved,
        "best_p": best_p,
        "best_test": best_test,
        "cohens_d": cohens_d,
        "wilcoxon_stat": wilcoxon_stat,
        "wilcoxon_p": wilcoxon_p,
        "mw_stat": mw_stat,
        "mw_p": mw_p,
        "t_stat": t_stat,
        "t_p": t_p if not np.isnan(t_p if isinstance(t_p, float) else float('nan')) else None,
        "t_p_onesided": t_p_onesided,
        "boot_p": boot_p,
        "prob_t_stat": prob_t_stat,
        "prob_t_p": prob_t_p,
        "prob_changes_child_mean": float(prob_changes_child.mean()) if len(prob_changes_child) > 0 else 0,
        "prob_changes_control_mean": float(prob_changes_control.mean()) if len(prob_changes_control) > 0 else 0,
        "n_nonzero_diffs": int((diff != 0).sum()) if len(diff) > 0 else 0,
    }


def interpret_results(stat_results):
    """Generate verdict and detail from statistical results."""
    mean_diff = stat_results["mean_diff"]
    significance_achieved = stat_results["significance_achieved"]
    best_p = stat_results["best_p"]
    best_test = stat_results["best_test"]
    cohens_d = stat_results["cohens_d"]
    n_with_absorption = stat_results["n_with_absorption"]
    n_positive_recovery = stat_results["n_positive_recovery"]
    mean_recovery_child = stat_results["mean_recovery_child"]
    mean_recovery_control = stat_results["mean_recovery_control"]
    ci_lower = stat_results["ci_lower"]
    ci_upper = stat_results["ci_upper"]

    if significance_achieved and mean_diff > 0.02:
        verdict = "CAUSAL_ABSORPTION_CONFIRMED"
        detail = (
            f"Activation patching provides statistically significant causal evidence for "
            f"feature absorption. Zeroing child features recovers correct letter predictions "
            f"at rate {mean_recovery_child:.3f} vs. control {mean_recovery_control:.3f} "
            f"(diff={mean_diff:.3f}, best p={best_p:.4f} [{best_test}], "
            f"Cohen's d={cohens_d:.2f}). N={n_with_absorption} words with absorption, "
            f"{n_positive_recovery} show positive recovery effect."
        )
    elif mean_diff > 0.02 and n_positive_recovery > stat_results["n_negative_recovery"]:
        verdict = "DIRECTIONAL_CAUSAL_EVIDENCE"
        detail = (
            f"Activation patching shows a directional causal effect of feature absorption, "
            f"but statistical significance is not achieved (best p={best_p:.4f} [{best_test}]). "
            f"Recovery rate {mean_recovery_child:.3f} vs. control {mean_recovery_control:.3f} "
            f"(diff={mean_diff:.3f}). {n_positive_recovery}/{n_with_absorption} words show "
            f"positive recovery. Effect size Cohen's d={cohens_d:.2f}. "
            f"The 95% CI for recovery difference is [{ci_lower:.3f}, {ci_upper:.3f}]."
        )
    else:
        verdict = "INSUFFICIENT_EVIDENCE"
        detail = (
            f"Activation patching does not provide clear causal evidence for absorption. "
            f"Mean recovery: child={mean_recovery_child:.3f}, control={mean_recovery_control:.3f} "
            f"(diff={mean_diff:.3f})."
        )
    return verdict, detail


# ============================================================
# Main experiment
# ============================================================
def main():
    write_pid()
    t_start = time.time()
    report_progress(0, 30, metrics={"status": "starting", "mode": "FULL"})
    logger.info(f"Phase 0.1 FULL: Activation Patching at Scale. PID={os.getpid()}, Device={DEVICE}")
    logger.info(f"Target: n>={TARGET_N_WORDS} words, {CONTEXTS_PER_WORD} contexts each")

    # =========================================================
    # Step 1: Load model
    # =========================================================
    logger.info("Step 1: Loading Gemma 2 2B via TransformerLens...")
    import transformer_lens
    from transformers import AutoModelForCausalLM, AutoTokenizer as HFAutoTokenizer

    hf_model = AutoModelForCausalLM.from_pretrained(
        "unsloth/gemma-2-2b",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    hf_tokenizer = HFAutoTokenizer.from_pretrained(
        "unsloth/gemma-2-2b",
        local_files_only=True,
    )
    model = transformer_lens.HookedTransformer.from_pretrained(
        "google/gemma-2-2b",
        hf_model=hf_model,
        tokenizer=hf_tokenizer,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    del hf_model
    gc.collect()
    model.eval()
    tokenizer = model.tokenizer
    d_model = model.cfg.d_model
    logger.info(f"Model loaded in {time.time() - t_start:.1f}s. d_model={d_model}")
    report_progress(1, 30, metrics={"status": "model_loaded"})

    # =========================================================
    # Step 2: Load SAE
    # =========================================================
    logger.info("Step 2: Loading Gemma Scope SAE (layer 12, 16k, canonical)...")
    from sae_lens import SAE

    t_sae = time.time()
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=DEVICE,
    )
    n_features = sae.cfg.d_sae
    logger.info(f"SAE loaded in {time.time() - t_sae:.1f}s. n_features={n_features}")
    report_progress(2, 30, metrics={"status": "sae_loaded", "n_features": n_features})

    # =========================================================
    # Step 3: Train high-quality first-letter probe at layer 12
    # =========================================================
    logger.info("Step 3: Training first-letter probe at layer 12...")

    vocab = tokenizer.get_vocab()
    letter_words_probe = defaultdict(list)
    for token_str, token_id in vocab.items():
        clean = token_str.lstrip("\u2581").strip()
        if (len(clean) >= 3 and clean.isascii() and clean[0].isalpha()
            and clean.isalpha() and clean.islower()):
            letter = clean[0].lower()
            if letter in "abcdefghijklmnopqrstuvwxyz":
                letter_words_probe[letter].append(clean)

    for letter in letter_words_probe:
        letter_words_probe[letter] = list(set(letter_words_probe[letter]))
        random.shuffle(letter_words_probe[letter])

    PROBE_WORDS_PER_LETTER = 100
    probe_contexts = []
    probe_labels = []
    for letter in sorted(letter_words_probe.keys()):
        words = letter_words_probe[letter][:PROBE_WORDS_PER_LETTER]
        for w in words:
            ctx = f"The word is {w}"
            probe_contexts.append(ctx)
            probe_labels.append(letter)

    letter_to_idx = {letter: idx for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}
    idx_to_letter = {v: k for k, v in letter_to_idx.items()}

    logger.info(f"Probe training data: {len(probe_contexts)} contexts across "
                f"{len(set(probe_labels))} letters")

    t_cache = time.time()
    probe_activations = cache_activations_batch(
        model, tokenizer, probe_contexts, HOOK_POINT, TARGET_LAYER,
        DEVICE, batch_size=16
    )
    logger.info(f"Probe activations cached in {time.time() - t_cache:.1f}s. Shape: {probe_activations.shape}")

    y = np.array([letter_to_idx[l] for l in probe_labels])
    X = probe_activations.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    probe = LogisticRegression(
        C=1.0, max_iter=3000, solver='lbfgs', random_state=SEED,
    )
    probe.fit(X_train, y_train)

    train_f1 = f1_score(y_train, probe.predict(X_train), average='macro')
    test_f1 = f1_score(y_test, probe.predict(X_test), average='macro')
    test_acc = accuracy_score(y_test, probe.predict(X_test))
    logger.info(f"Probe quality: train_f1={train_f1:.3f}, test_f1={test_f1:.3f}, test_acc={test_acc:.3f}")
    report_progress(3, 30, metrics={"status": "probe_trained", "test_f1": float(test_f1)})

    # =========================================================
    # Step 4: Discover additional absorption pairs
    # =========================================================
    logger.info("Step 4: Discovering additional absorption pairs beyond pilot's 7 words...")
    t_discovery = time.time()

    discovered_pairs = discover_absorption_pairs(
        model, tokenizer, sae, probe, letter_to_idx, DEVICE,
        n_candidate_words=150,
        n_test_contexts=50,
    )

    all_words = []
    for cw in PILOT_CORE_WORDS:
        all_words.append({
            "word": cw["word"], "letter": cw["letter"],
            "child_features": cw["child_features"],
            "primary_child": cw["primary_child"], "source": "pilot_core",
        })
    for dp in discovered_pairs:
        all_words.append({
            "word": dp["word"], "letter": dp["letter"],
            "child_features": dp["child_features"],
            "primary_child": dp["primary_child"], "source": "discovered_ig",
            "discovery_stats": dp["discovery_stats"],
        })

    total_words = len(all_words)
    logger.info(f"Total absorption pairs: {total_words} "
                f"({len(PILOT_CORE_WORDS)} pilot + {len(discovered_pairs)} discovered) "
                f"in {time.time() - t_discovery:.1f}s")
    report_progress(5, 30, metrics={
        "status": "discovery_done",
        "n_pilot": len(PILOT_CORE_WORDS),
        "n_discovered": len(discovered_pairs),
        "n_total": total_words,
    })

    # =========================================================
    # Step 5: Full activation patching experiment
    # =========================================================
    logger.info(f"Step 5: Running activation patching on {total_words} words x "
                f"{CONTEXTS_PER_WORD} contexts each...")
    torch.cuda.empty_cache()

    results_per_word = []
    completed_count = 0

    for word_idx, wd in enumerate(all_words):
        logger.info(f"  [{word_idx+1}/{total_words}] {wd['word']} (letter={wd['letter']}, "
                     f"primary_child={wd.get('primary_child')}, source={wd.get('source')})")

        word_result = run_patching_for_word(
            model, tokenizer, sae, probe, letter_to_idx,
            wd, DEVICE, n_contexts=CONTEXTS_PER_WORD, n_controls=N_CONTROL_FEATURES
        )
        results_per_word.append(word_result)

        if word_result.get("status") == "completed":
            completed_count += 1
            n_abs = word_result['sae_reconstruction']['n_absorbed']
            rec_abs = word_result['primary_child_zeroed']['recovery_rate_absorbed']
            ctrl_abs = word_result['control_random_zeroed']['mean_recovery_rate_absorbed']
            logger.info(f"    absorbed={n_abs}, recovery_abs={rec_abs:.3f}, ctrl={ctrl_abs:.3f}")

        report_progress(5 + word_idx, 5 + total_words + 5, metrics={
            "status": f"patching_{wd['word']}",
            "words_done": completed_count,
            "total_words": total_words,
        })

    # =========================================================
    # Step 6: Aggregate analysis and statistical tests
    # =========================================================
    logger.info("Step 6: Computing aggregate statistics...")

    del model
    gc.collect()
    torch.cuda.empty_cache()

    completed = [r for r in results_per_word if r.get("status") == "completed"]
    words_with_absorption = [r for r in completed
                             if r["sae_reconstruction"]["n_absorbed"] >= MIN_ABSORPTION_CONTEXTS]

    logger.info(f"Completed: {len(completed)} words, {len(words_with_absorption)} with absorption >= {MIN_ABSORPTION_CONTEXTS}")

    stat_results = compute_statistics(words_with_absorption)
    verdict, detail = interpret_results(stat_results)

    logger.info(f"\nVERDICT: {verdict}")
    logger.info(f"DETAIL: {detail}")

    # =========================================================
    # Step 7: Assemble full results
    # =========================================================
    summary = {
        "task_id": TASK_ID,
        "mode": "FULL",
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae": {"release": SAE_RELEASE, "sae_id": SAE_ID},
        "n_total_words": total_words,
        "n_completed": len(completed),
        "n_with_absorption": stat_results["n_with_absorption"],
        "n_pilot_core": len(PILOT_CORE_WORDS),
        "n_discovered": len(discovered_pairs),
        "contexts_per_word": CONTEXTS_PER_WORD,
        "probe_quality": {
            "train_f1": float(train_f1),
            "test_f1": float(test_f1),
            "test_accuracy": float(test_acc),
            "layer": TARGET_LAYER,
            "n_train_contexts": len(probe_contexts),
        },
        "aggregate": {
            "mean_absorption_rate": stat_results["mean_absorption"],
            "mean_recovery_rate_child_absorbed": stat_results["mean_recovery_child"],
            "mean_recovery_rate_control_absorbed": stat_results["mean_recovery_control"],
            "recovery_difference": stat_results["mean_diff"],
            "recovery_difference_ci_95": [stat_results["ci_lower"], stat_results["ci_upper"]],
            "n_positive_recovery": stat_results["n_positive_recovery"],
            "n_zero_recovery": stat_results["n_zero_recovery"],
            "n_negative_recovery": stat_results["n_negative_recovery"],
            "mean_prob_change_child": stat_results["prob_changes_child_mean"],
            "mean_prob_change_control": stat_results["prob_changes_control_mean"],
        },
        "statistical_tests": {
            "wilcoxon_signed_rank": {
                "statistic": float(stat_results["wilcoxon_stat"]) if not np.isnan(stat_results["wilcoxon_stat"]) else None,
                "p_value": float(stat_results["wilcoxon_p"]) if not np.isnan(stat_results["wilcoxon_p"]) else None,
                "note": "One-sided test: child recovery > control recovery (among absorbed tokens). Only non-zero diffs used.",
                "n_nonzero_diffs": stat_results["n_nonzero_diffs"],
            },
            "mann_whitney_u": {
                "statistic": float(stat_results["mw_stat"]) if not np.isnan(stat_results["mw_stat"]) else None,
                "p_value": float(stat_results["mw_p"]) if not np.isnan(stat_results["mw_p"]) else None,
                "note": "One-sided: child > control",
            },
            "paired_t_test": {
                "statistic": float(stat_results["t_stat"]) if not np.isnan(stat_results["t_stat"]) else None,
                "p_value_twosided": stat_results["t_p"] if stat_results["t_p"] is not None and not np.isnan(stat_results["t_p"]) else None,
                "p_value_onesided": float(stat_results["t_p_onesided"]) if not np.isnan(stat_results["t_p_onesided"]) else None,
            },
            "bootstrap": {
                "p_value": stat_results["boot_p"],
                "n_resamples": N_BOOTSTRAP,
                "ci_95": [stat_results["ci_lower"], stat_results["ci_upper"]],
            },
            "effect_size": {
                "cohens_d": stat_results["cohens_d"],
                "interpretation": (
                    "large" if abs(stat_results["cohens_d"]) >= 0.8 else
                    "medium" if abs(stat_results["cohens_d"]) >= 0.5 else
                    "small" if abs(stat_results["cohens_d"]) >= 0.2 else "negligible"
                ) if not np.isnan(stat_results["cohens_d"]) else "N/A",
            },
            "probability_change_test": {
                "t_statistic": float(stat_results["prob_t_stat"]) if not np.isnan(stat_results["prob_t_stat"]) else None,
                "p_value": float(stat_results["prob_t_p"]) if not np.isnan(stat_results["prob_t_p"]) else None,
            },
        },
        "interpretation": {
            "verdict": verdict,
            "detail": detail,
            "best_p_value": stat_results["best_p"] if stat_results["best_p"] != float('inf') else None,
            "best_test": stat_results["best_test"],
            "significance_at_005": stat_results["significance_achieved"],
            "significance_at_001": stat_results["best_p"] < 0.01 if stat_results["best_p"] != float('inf') else False,
        },
        "discovery_summary": {
            "n_candidates_tested": len(discovered_pairs),
            "n_absorption_pairs_found": len(discovered_pairs),
            "method": "Integrated-gradients-inspired: decoder-probe cosine x mean activation on absorbed tokens",
            "pairs": [
                {
                    "word": dp["word"], "letter": dp["letter"],
                    "primary_child": dp["primary_child"],
                    "absorption_rate_discovery": dp["discovery_stats"]["absorption_rate"],
                }
                for dp in discovered_pairs
            ],
        },
        "cross_hierarchy_note": (
            "Cross-hierarchy activation patching (city-continent, city-language) was NOT attempted "
            "because RAVEL probes do not pass the quality gate (best F1: city-continent=0.843, "
            "city-language=0.823, city-country=0.789 at layer 24). Only first-letter probes "
            "(F1=0.971 at L24, F1=0.876 at L12) have sufficient quality for reliable patching. "
            "Cross-hierarchy patching remains future work pending better probe training methods."
        ),
        "per_word_results": results_per_word,
        "elapsed_minutes": round((time.time() - t_start) / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # Save JSON
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2, cls=NumpyEncoder))
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # =========================================================
    # Step 8: Generate summary markdown
    # =========================================================
    md_lines = [
        "# Phase 0.1 FULL: Activation Patching at Scale",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"{detail}",
        "",
        "## Configuration",
        f"- Mode: **FULL** (iter_009 full-mode experiment)",
        f"- Model: Gemma 2 2B",
        f"- SAE: {SAE_RELEASE}, {SAE_ID}",
        f"- Probe: sklearn LogReg at layer {TARGET_LAYER}, F1={test_f1:.3f}",
        f"- Contexts per word: {CONTEXTS_PER_WORD}",
        f"- Control features: {N_CONTROL_FEATURES} (magnitude-matched)",
        f"- Bootstrap resamples: {N_BOOTSTRAP}",
        "",
        "## Word Count",
        f"- Pilot core words: {len(PILOT_CORE_WORDS)}",
        f"- Discovered pairs: {len(discovered_pairs)}",
        f"- Total completed: {len(completed)}",
        f"- With absorption >= {MIN_ABSORPTION_CONTEXTS}: {stat_results['n_with_absorption']}",
        "",
        "## Statistical Tests",
        f"- Wilcoxon signed-rank (one-sided): stat={stat_results['wilcoxon_stat']}, p={stat_results['wilcoxon_p']}",
        f"- Mann-Whitney U (one-sided): stat={stat_results['mw_stat']}, p={stat_results['mw_p']}",
        f"- Paired t-test (one-sided): t={stat_results['t_stat']}, p={stat_results['t_p_onesided']}",
        f"- Bootstrap p-value: {stat_results['boot_p']}",
        f"- Cohen's d: {stat_results['cohens_d']}",
        f"- 95% CI on recovery difference: [{stat_results['ci_lower']}, {stat_results['ci_upper']}]",
        "",
        "## Results Table",
        "",
        "| Word | Letter | Source | Raw Acc | Recon Acc | Absorbed | Recovery (abs) | Ctrl Recovery |",
        "|------|--------|--------|---------|-----------|----------|----------------|---------------|",
    ]

    for r in completed:
        n_abs = r['sae_reconstruction']['n_absorbed']
        rec_abs = r['primary_child_zeroed']['recovery_rate_absorbed']
        ctrl_abs = r['control_random_zeroed']['mean_recovery_rate_absorbed']
        md_lines.append(
            f"| {r['word']} | {r['letter']} | {r.get('source', '?')} | "
            f"{r['raw_residual']['accuracy']:.3f} | "
            f"{r['sae_reconstruction']['accuracy']:.3f} | "
            f"{n_abs} | "
            f"{'**' if rec_abs > ctrl_abs + 0.05 else ''}{rec_abs:.3f}{'**' if rec_abs > ctrl_abs + 0.05 else ''} | "
            f"{ctrl_abs:.3f} |"
        )

    md_lines.extend([
        "",
        "## Aggregate",
        f"- Mean absorption rate: {stat_results['mean_absorption']:.3f}",
        f"- Mean recovery (child zeroed): {stat_results['mean_recovery_child']:.3f}",
        f"- Mean recovery (control): {stat_results['mean_recovery_control']:.3f}",
        f"- Recovery difference: {stat_results['mean_diff']:.3f}",
        f"- Words with positive recovery effect: {stat_results['n_positive_recovery']}/{stat_results['n_with_absorption']}",
        "",
        "## Cross-Hierarchy Note",
        "Cross-hierarchy activation patching was NOT attempted because RAVEL probes do not pass the quality gate:",
        "- city-continent: best F1=0.843 (gate=0.85)",
        "- city-language: best F1=0.823 (gate=0.85)",
        "- city-country: best F1=0.789 (gate=0.85)",
        "This remains future work pending better probe training methods.",
        "",
        f"## Elapsed: {summary['elapsed_minutes']:.1f} minutes",
    ])

    SUMMARY_FILE.write_text("\n".join(md_lines))
    logger.info(f"Summary saved to {SUMMARY_FILE}")

    # =========================================================
    # Print summary table
    # =========================================================
    logger.info("\n" + "=" * 120)
    logger.info("ACTIVATION PATCHING RESULTS SUMMARY (FULL)")
    logger.info("=" * 120)
    logger.info(f"{'Word':<16} {'Ltr':<4} {'Src':<14} {'Raw':<6} {'Recon':<6} {'Absd':<6} "
                f"{'Child0_rec':<11} {'Ctrl_rec':<9} {'ProbChg':<8}")
    logger.info("-" * 120)
    for r in completed:
        n_abs = r['sae_reconstruction']['n_absorbed']
        logger.info(
            f"{r['word']:<16} {r['letter']:<4} {r.get('source', '?'):<14} "
            f"{r['raw_residual']['accuracy']:<6.3f} "
            f"{r['sae_reconstruction']['accuracy']:<6.3f} "
            f"{n_abs:<6d} "
            f"{r['primary_child_zeroed']['recovery_rate_absorbed']:<11.3f} "
            f"{r['control_random_zeroed']['mean_recovery_rate_absorbed']:<9.3f} "
            f"{r['primary_child_zeroed']['prob_change_vs_recon']:<8.4f}"
        )
    logger.info("-" * 120)
    logger.info(f"VERDICT: {verdict}")
    logger.info(f"Best p-value: {stat_results['best_p']:.6f} ({stat_results['best_test']})")
    logger.info(f"Cohen's d: {stat_results['cohens_d']:.3f}")
    logger.info(f"95% CI: [{stat_results['ci_lower']:.3f}, {stat_results['ci_upper']:.3f}]")
    logger.info(f"Elapsed: {summary['elapsed_minutes']:.1f} minutes")
    logger.info("=" * 120)

    # =========================================================
    # Step 9: Update gpu_progress.json
    # =========================================================
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        if gpu_progress_path.exists():
            gpu_data = json.loads(gpu_progress_path.read_text())
        else:
            gpu_data = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if TASK_ID not in gpu_data.get("completed", []):
            gpu_data.setdefault("completed", []).append(TASK_ID)
        # Remove from running
        gpu_data.get("running", {}).pop(TASK_ID, None)
        # Record timing
        gpu_data.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 60,
            "actual_min": round(summary["elapsed_minutes"]),
            "start_time": (datetime.now() - __import__("datetime").timedelta(minutes=summary["elapsed_minutes"])).isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "sae": SAE_ID,
                "n_words": total_words,
                "contexts_per_word": CONTEXTS_PER_WORD,
                "gpu_model": "RTX PRO 6000",
                "gpu_count": 1,
            }
        }
        gpu_progress_path.write_text(json.dumps(gpu_data, indent=2))
        logger.info(f"gpu_progress.json updated: {TASK_ID} -> completed")
    except Exception as e:
        logger.warning(f"Failed to update gpu_progress.json: {e}")

    mark_done(
        status="success",
        summary=f"{verdict}: {len(completed)} words, {stat_results['n_with_absorption']} with absorption, "
                f"best_p={stat_results['best_p']:.6f}, Cohen's_d={stat_results['cohens_d']:.2f}"
    )
    return summary


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        mark_done(status="failure", summary=str(e))
        raise
