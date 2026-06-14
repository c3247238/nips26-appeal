#!/usr/bin/env python3
"""
Phase 0.1: Activation Patching on Persistent Core Words -- Causal Absorption Evidence
======================================================================================

This experiment provides metric-independent causal evidence for competitive exclusion
(feature absorption). It resolves the two-interpretation ambiguity:
  (a) JumpReLU genuinely has minimal absorption (metric correct, near-zero true absorption)
  (b) Metric thresholds miscalibrated for JumpReLU activation geometry (metric wrong)

Pipeline:
1. Load Gemma 2 2B + Gemma Scope 16k SAE at layer 12 (JumpReLU, canonical L0)
2. For each persistent core word identified as absorption candidate:
   a. Collect 100+ input contexts containing the word
   b. Run forward pass, cache residual stream activations at layer 12
   c. Encode through SAE to get feature activations
   d. Zero the identified child feature in SAE features
   e. Reconstruct modified residual: SAE.decode(modified_features) + error_term
   f. Apply first-letter linear probe to modified residual
   g. Check if probe prediction recovers (from wrong to correct)
3. Control: zero random non-child features (same procedure)
4. Statistical test: paired comparison child-zeroed vs random-zeroed

KEY INSIGHT: The iter_007 approach checked if SAE parent FEATURES recovered when child
features were zeroed. This is wrong because the SAE is a fixed encoding -- parent features
won't spontaneously activate. The correct approach patches the RESIDUAL STREAM through
the SAE and checks if a LINEAR PROBE on the modified residual recovers.

MODE: PILOT (100 contexts per word, seed=42)
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
TASK_ID = "phase0_activation_patching"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE0_DIR = PILOT_DIR / "phase0"
for d in [PILOT_DIR, PHASE0_DIR]:
    d.mkdir(parents=True, exist_ok=True)

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"
OUTPUT_FILE = PHASE0_DIR / "activation_patching.json"

DEVICE = "cuda:0"  # GPU 6 mapped via CUDA_VISIBLE_DEVICES

# Pilot settings
PILOT_CONTEXTS_PER_WORD = 100
N_CONTROL_FEATURES = 10
N_BOOTSTRAP = 1000

# SAE config: canonical L0 at layer 12, 16k width
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_12/width_16k/canonical"
HOOK_POINT = "blocks.12.hook_resid_post"
TARGET_LAYER = 12

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
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


# ============================================================
# The 8 persistent core words from iter_007 analysis
# These remain false negatives across ALL L0 levels (22, 41, 82, 176)
# Each has identified absorbing child features at L0=82
# ============================================================
CORE_WORDS = [
    {
        "word": "eight",
        "letter": "e",
        "child_features": [892, 8174, 5327, 5827],  # from iter_007 analysis
        "primary_child": 5327,  # highest activation
    },
    {
        "word": "liked",
        "letter": "l",
        "child_features": [4678, 7984, 14031, 3967],
        "primary_child": 4678,
    },
    {
        "word": "lower",
        "letter": "l",
        "child_features": [3858, 11826, 14449],
        "primary_child": 14449,  # highest activation
    },
    {
        "word": "offer",
        "letter": "o",
        "child_features": [15092],
        "primary_child": 15092,
    },
    {
        "word": "often",
        "letter": "o",
        "child_features": [3063],
        "primary_child": 3063,
    },
    {
        "word": "other",
        "letter": "o",
        "child_features": [15322],
        "primary_child": 15322,
    },
    {
        "word": "under",
        "letter": "u",
        "child_features": [2810, 12675],
        "primary_child": 2810,
    },
    {
        "word": "until",
        "letter": "u",
        "child_features": [],  # No identified absorbing features at L0=82
        "primary_child": None,
    },
]


# ============================================================
# Context generation templates
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
# Main experiment
# ============================================================
def main():
    write_pid()
    t_start = time.time()
    report_progress(0, 10, metrics={"status": "starting"})
    logger.info(f"Phase 0.1: Activation Patching. PID={os.getpid()}, Device={DEVICE}")

    # ----- Step 1: Load model -----
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
    report_progress(1, 10, metrics={"status": "model_loaded"})

    # ----- Step 2: Load SAE -----
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
    report_progress(2, 10, metrics={"status": "sae_loaded", "n_features": n_features})

    # ----- Step 3: Generate contexts and cache activations -----
    logger.info("Step 3: Generating contexts and caching residual stream activations...")

    # Generate contexts for ALL core words + additional vocabulary for probe training
    all_word_contexts = {}
    for cw in CORE_WORDS:
        word = cw["word"]
        contexts = generate_contexts(word, PILOT_CONTEXTS_PER_WORD)
        all_word_contexts[word] = contexts

    # Also generate contexts for probe training: sample words from each letter
    vocab = tokenizer.get_vocab()
    letter_words = defaultdict(list)
    for token_str, token_id in vocab.items():
        clean = token_str.lstrip("▁").strip()
        if len(clean) >= 3 and clean.isascii() and clean[0].isalpha() and clean.isalpha():
            letter = clean[0].lower()
            if letter in "abcdefghijklmnopqrstuvwxyz":
                letter_words[letter].append(clean.lower())

    # Deduplicate and sample
    for letter in letter_words:
        letter_words[letter] = list(set(letter_words[letter]))
        random.shuffle(letter_words[letter])

    # Generate probe training data: 50 words per letter for diverse probe training
    probe_contexts = []
    probe_labels = []
    probe_words = []
    PROBE_WORDS_PER_LETTER = 50

    for letter in sorted(letter_words.keys()):
        words = letter_words[letter][:PROBE_WORDS_PER_LETTER]
        for w in words:
            ctx = f"The word is {w}"
            probe_contexts.append(ctx)
            probe_labels.append(letter)
            probe_words.append(w)

    logger.info(f"Probe training data: {len(probe_contexts)} contexts across {len(set(probe_labels))} unique letters")

    # Cache activations for probe training data
    def cache_activations(model, prompts, hook_point, device, word_positions=None):
        """Cache residual stream activations at specified hook point.

        If word_positions is None, uses last token position.
        If word_positions is provided, uses those specific positions.
        """
        all_acts = []
        for i, prompt in enumerate(prompts):
            tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens,
                    names_filter=[hook_point],
                    stop_at_layer=TARGET_LAYER + 1,
                )
            if word_positions is not None and word_positions[i] is not None:
                pos = word_positions[i]
            else:
                pos = -1  # last token
            act = cache[hook_point][0, pos, :].float().cpu()
            all_acts.append(act)
            del cache
            if (i + 1) % 200 == 0:
                torch.cuda.empty_cache()
        return torch.stack(all_acts)

    t_cache = time.time()

    # Find word token positions in each prompt
    def find_word_position(tokenizer, prompt, target_word):
        """Find the token position of the target word in the prompt.
        Returns the position of the first token of the target word.
        For 'The word is {word}', the word token is typically at position 4 or 5.
        """
        tokens = tokenizer.encode(prompt)
        token_strs = [tokenizer.decode([t]) for t in tokens]

        # Find token that contains the target word (or starts with it)
        for pos, ts in enumerate(token_strs):
            clean = ts.strip().lower()
            if clean == target_word.lower() or clean == f"▁{target_word.lower()}" or \
               clean.lstrip("▁") == target_word.lower():
                return pos
        # Fallback: search for the word in decoded tokens
        for pos, ts in enumerate(token_strs):
            if target_word.lower() in ts.lower():
                return pos
        return -1  # last token as fallback

    # Cache probe training activations (at last token -- simpler)
    logger.info(f"Caching {len(probe_contexts)} probe training activations...")
    probe_activations = cache_activations(model, probe_contexts, HOOK_POINT, DEVICE)
    logger.info(f"Probe activations cached in {time.time() - t_cache:.1f}s. Shape: {probe_activations.shape}")
    report_progress(3, 10, metrics={"status": "probe_activations_cached"})

    # Cache core word activations (at word position for more accurate patching)
    logger.info("Caching core word activations...")
    core_word_acts = {}
    core_word_positions = {}
    for cw in CORE_WORDS:
        word = cw["word"]
        contexts = all_word_contexts[word]

        # Find word positions
        positions = []
        for ctx in contexts:
            pos = find_word_position(tokenizer, ctx, word)
            positions.append(pos)
        core_word_positions[word] = positions

        acts = cache_activations(model, contexts, HOOK_POINT, DEVICE, word_positions=positions)
        core_word_acts[word] = acts
        logger.info(f"  {word}: {acts.shape[0]} contexts cached")

    logger.info(f"All core word activations cached in {time.time() - t_cache:.1f}s")
    report_progress(4, 10, metrics={"status": "core_activations_cached"})

    # Free model memory -- probe training + SAE patching only from here
    del model
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Model freed from GPU memory")

    # ----- Step 4: Train first-letter probe -----
    logger.info("Step 4: Training first-letter linear probe...")

    letter_to_idx = {letter: idx for idx, letter in enumerate("abcdefghijklmnopqrstuvwxyz")}
    idx_to_letter = {v: k for k, v in letter_to_idx.items()}
    y = np.array([letter_to_idx[l] for l in probe_labels])
    X = probe_activations.numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    probe = LogisticRegression(
        C=1.0, max_iter=2000, solver='lbfgs', random_state=SEED
    )
    probe.fit(X_train, y_train)

    train_f1 = f1_score(y_train, probe.predict(X_train), average='macro')
    test_f1 = f1_score(y_test, probe.predict(X_test), average='macro')
    test_acc = accuracy_score(y_test, probe.predict(X_test))
    logger.info(f"Probe quality: train_f1={train_f1:.3f}, test_f1={test_f1:.3f}, test_acc={test_acc:.3f}")
    report_progress(5, 10, metrics={"status": "probe_trained", "test_f1": test_f1})

    # Get probe weight vectors per letter (for projection analysis)
    probe_weights = probe.coef_  # [26, d_model]

    # ----- Step 5: Activation patching experiment -----
    logger.info("Step 5: Running activation patching experiment...")

    results_per_word = []
    total_words = len([cw for cw in CORE_WORDS if cw["primary_child"] is not None])
    word_idx = 0

    for cw in CORE_WORDS:
        word = cw["word"]
        letter = cw["letter"]
        letter_idx = letter_to_idx[letter]
        child_features = cw["child_features"]
        primary_child = cw["primary_child"]

        if primary_child is None:
            logger.info(f"  Skipping {word}: no identified child features")
            results_per_word.append({
                "word": word,
                "letter": letter,
                "status": "skipped",
                "reason": "no identified child features",
            })
            continue

        word_idx += 1
        logger.info(f"  Processing {word} ({word_idx}/{total_words}), letter={letter}, "
                     f"child_features={child_features}, primary_child={primary_child}")

        acts = core_word_acts[word].to(DEVICE)  # [N, d_model]
        n_contexts = acts.shape[0]

        # Step 5a: Encode through SAE
        with torch.no_grad():
            sae_features = sae.encode(acts)  # [N, n_features]
            sae_recon = sae.decode(sae_features)  # [N, d_model]
            error = acts - sae_recon  # [N, d_model]

        # Measure probe on THREE representations:
        # (A) Raw residual stream (ground truth -- should be correct)
        # (B) SAE reconstruction only (no error term -- absorption manifests here)
        # (C) Full residual = SAE recon + error (should equal raw)

        raw_preds = probe.predict(acts.cpu().numpy())
        raw_probs = probe.predict_proba(acts.cpu().numpy())
        raw_correct = (raw_preds == letter_idx)
        raw_accuracy = float(raw_correct.mean())
        raw_letter_prob = raw_probs[:, letter_idx]

        recon_preds = probe.predict(sae_recon.cpu().numpy())
        recon_probs = probe.predict_proba(sae_recon.cpu().numpy())
        recon_correct = (recon_preds == letter_idx)
        recon_accuracy = float(recon_correct.mean())
        recon_letter_prob = recon_probs[:, letter_idx]

        # Absorption metric: tokens where raw is correct but SAE recon is wrong
        absorbed_mask = raw_correct & (~recon_correct)
        n_absorbed = int(absorbed_mask.sum())
        absorption_rate = float(n_absorbed / max(1, int(raw_correct.sum())))

        logger.info(f"    Raw accuracy: {raw_accuracy:.3f}, SAE recon accuracy: {recon_accuracy:.3f}, "
                     f"absorbed: {n_absorbed}/{int(raw_correct.sum())} ({absorption_rate:.3f})")

        # Step 5b: Zero PRIMARY child feature -- check if probe RECOVERS on SAE recon
        # Key test: among absorbed tokens, does zeroing child restore correct prediction?
        modified_features = sae_features.clone()
        child_original_activations = modified_features[:, primary_child].clone()
        modified_features[:, primary_child] = 0.0

        with torch.no_grad():
            modified_recon = sae.decode(modified_features)

        child_zeroed_preds = probe.predict(modified_recon.cpu().numpy())
        child_zeroed_probs = probe.predict_proba(modified_recon.cpu().numpy())
        child_zeroed_correct = (child_zeroed_preds == letter_idx)
        child_zeroed_accuracy = float(child_zeroed_correct.mean())
        child_zeroed_letter_prob = child_zeroed_probs[:, letter_idx]

        # Recovery on SAE recon: tokens that were WRONG on SAE recon but CORRECT after child zeroing
        recovery_from_recon = (~recon_correct) & child_zeroed_correct
        n_recovered_recon = int(recovery_from_recon.sum())
        recovery_rate_recon = float(n_recovered_recon / max(1, int((~recon_correct).sum())))

        # Recovery specifically among ABSORBED tokens (raw correct, recon wrong)
        recovery_absorbed = absorbed_mask & child_zeroed_correct
        n_recovered_absorbed = int(recovery_absorbed.sum())
        recovery_rate_absorbed = float(n_recovered_absorbed / max(1, n_absorbed))

        # Degradation: tokens that were correct on SAE recon and become wrong
        degradation_mask = recon_correct & (~child_zeroed_correct)
        n_degraded = int(degradation_mask.sum())
        degradation_rate = float(n_degraded / max(1, int(recon_correct.sum())))

        # Probability change for the correct letter on SAE reconstruction
        prob_change = float((child_zeroed_letter_prob - recon_letter_prob).mean())

        logger.info(f"    Primary child zeroed: recon_acc {recon_accuracy:.3f} -> {child_zeroed_accuracy:.3f}, "
                     f"recovery_rate_recon={recovery_rate_recon:.3f} ({n_recovered_recon}), "
                     f"recovery_rate_absorbed={recovery_rate_absorbed:.3f} ({n_recovered_absorbed}/{n_absorbed}), "
                     f"prob_change={prob_change:.4f}")

        # Step 5c: Also test with full residual (recon + error) for comparison
        modified_full = modified_recon + error
        full_preds = probe.predict(modified_full.cpu().numpy())
        full_correct = (full_preds == letter_idx)
        full_accuracy = float(full_correct.mean())

        # Step 5d: Zero ALL child features
        all_child_modified = sae_features.clone()
        for cf in child_features:
            all_child_modified[:, cf] = 0.0

        with torch.no_grad():
            all_child_recon = sae.decode(all_child_modified)

        all_child_preds = probe.predict(all_child_recon.cpu().numpy())
        all_child_correct = (all_child_preds == letter_idx)
        all_child_accuracy = float(all_child_correct.mean())
        all_child_probs = probe.predict_proba(all_child_recon.cpu().numpy())
        all_child_letter_prob = all_child_probs[:, letter_idx]

        all_recovery_recon = (~recon_correct) & all_child_correct
        all_recovery_rate = float(all_recovery_recon.sum() / max(1, int((~recon_correct).sum())))
        all_recovery_absorbed = absorbed_mask & all_child_correct
        all_recovery_rate_absorbed = float(all_recovery_absorbed.sum() / max(1, n_absorbed))

        logger.info(f"    All children zeroed: recon_acc {recon_accuracy:.3f} -> {all_child_accuracy:.3f}, "
                     f"recovery_rate_absorbed={all_recovery_rate_absorbed:.3f}")

        # Step 5e: Control -- zero random features
        rng = np.random.RandomState((SEED + abs(hash(word))) % (2**32))
        # Select random features that are NOT child features and ARE active
        active_mask = (sae_features.abs().mean(dim=0) > 0).cpu().numpy()
        active_indices = np.where(active_mask)[0]
        non_child_active = [idx for idx in active_indices if idx not in child_features]

        control_results = []
        control_accuracies = []
        control_recovery_rates = []
        control_recovery_rates_absorbed = []
        control_prob_changes = []

        for ctrl_i in range(min(N_CONTROL_FEATURES, len(non_child_active))):
            ctrl_feat = int(rng.choice(non_child_active))
            non_child_active = [x for x in non_child_active if x != ctrl_feat]

            ctrl_modified = sae_features.clone()
            ctrl_modified[:, ctrl_feat] = 0.0

            with torch.no_grad():
                ctrl_recon = sae.decode(ctrl_modified)

            # Probe on SAE reconstruction (no error) for fair comparison
            ctrl_preds = probe.predict(ctrl_recon.cpu().numpy())
            ctrl_probs = probe.predict_proba(ctrl_recon.cpu().numpy())
            ctrl_correct = (ctrl_preds == letter_idx)
            ctrl_letter_prob = ctrl_probs[:, letter_idx]
            ctrl_accuracy = float(ctrl_correct.mean())

            # Recovery from SAE recon baseline
            ctrl_recovery = float(((~recon_correct) & ctrl_correct).sum() / max(1, int((~recon_correct).sum())))
            # Recovery specifically among absorbed tokens
            ctrl_recovery_abs = float((absorbed_mask & ctrl_correct).sum() / max(1, n_absorbed))
            ctrl_prob_change = float((ctrl_letter_prob - recon_letter_prob).mean())

            control_results.append({
                "feature_idx": ctrl_feat,
                "accuracy": ctrl_accuracy,
                "recovery_rate": ctrl_recovery,
                "recovery_rate_absorbed": ctrl_recovery_abs,
                "prob_change": ctrl_prob_change,
                "mean_activation": float(sae_features[:, ctrl_feat].mean().item()),
            })
            control_accuracies.append(ctrl_accuracy)
            control_recovery_rates.append(ctrl_recovery)
            control_recovery_rates_absorbed.append(ctrl_recovery_abs)
            control_prob_changes.append(ctrl_prob_change)

        mean_control_accuracy = float(np.mean(control_accuracies)) if control_accuracies else 0
        mean_control_recovery = float(np.mean(control_recovery_rates)) if control_recovery_rates else 0
        mean_control_recovery_absorbed = float(np.mean(control_recovery_rates_absorbed)) if control_recovery_rates_absorbed else 0
        mean_control_prob_change = float(np.mean(control_prob_changes)) if control_prob_changes else 0

        logger.info(f"    Control (random feature): mean_acc={mean_control_accuracy:.3f}, "
                     f"mean_recovery_recon={mean_control_recovery:.3f}, "
                     f"mean_recovery_absorbed={mean_control_recovery_absorbed:.3f}")

        # Step 5f: Statistical test (paired comparison)
        # Compare child-zeroed recovery among absorbed tokens vs control recovery
        if len(control_recovery_rates_absorbed) >= 2 and n_absorbed > 0:
            t_stat, p_value = stats.ttest_1samp(control_recovery_rates_absorbed, recovery_rate_absorbed)
            p_value = float(p_value)
        else:
            t_stat, p_value = float('nan'), float('nan')

        # Step 5g: Per-child feature analysis
        per_child_results = []
        for cf in child_features:
            cf_modified = sae_features.clone()
            cf_modified[:, cf] = 0.0
            with torch.no_grad():
                cf_recon_out = sae.decode(cf_modified)

            # Probe on SAE reconstruction (no error)
            cf_preds = probe.predict(cf_recon_out.cpu().numpy())
            cf_probs = probe.predict_proba(cf_recon_out.cpu().numpy())
            cf_correct = (cf_preds == letter_idx)
            cf_letter_prob = cf_probs[:, letter_idx]
            cf_accuracy = float(cf_correct.mean())
            cf_recovery_recon = float(((~recon_correct) & cf_correct).sum() / max(1, int((~recon_correct).sum())))
            cf_recovery_absorbed = float((absorbed_mask & cf_correct).sum() / max(1, n_absorbed))
            cf_prob_change = float((cf_letter_prob - recon_letter_prob).mean())

            # Compute decoder cosine similarity with probe direction
            probe_dir = torch.tensor(probe_weights[letter_idx], dtype=torch.float32).to(DEVICE)
            probe_dir = probe_dir / probe_dir.norm()
            decoder_dir = sae.W_dec[cf].float()
            decoder_dir = decoder_dir / decoder_dir.norm()
            cos_sim = F.cosine_similarity(probe_dir.unsqueeze(0), decoder_dir.unsqueeze(0)).item()

            per_child_results.append({
                "feature_idx": int(cf),
                "accuracy_on_recon_after_zeroing": cf_accuracy,
                "recovery_rate_recon": cf_recovery_recon,
                "recovery_rate_absorbed": cf_recovery_absorbed,
                "prob_change_vs_recon": cf_prob_change,
                "mean_activation": float(sae_features[:, cf].mean().item()),
                "decoder_probe_cosine": float(cos_sim),
            })

        # Assemble results for this word
        word_result = {
            "word": word,
            "letter": letter,
            "status": "completed",
            "n_contexts": n_contexts,
            "child_features": child_features,
            "primary_child": primary_child,
            "raw_residual": {
                "accuracy": raw_accuracy,
                "mean_letter_prob": float(raw_letter_prob.mean()),
                "n_correct": int(raw_correct.sum()),
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
                "n_recovered_from_recon": n_recovered_recon,
                "recovery_rate_recon": recovery_rate_recon,
                "n_recovered_from_absorbed": n_recovered_absorbed,
                "recovery_rate_absorbed": recovery_rate_absorbed,
                "n_degraded": n_degraded,
                "degradation_rate": degradation_rate,
                "prob_change_vs_recon": prob_change,
                "accuracy_on_full_residual": full_accuracy,
            },
            "all_children_zeroed": {
                "accuracy_on_recon": all_child_accuracy,
                "mean_letter_prob": float(all_child_letter_prob.mean()),
                "recovery_rate_recon": all_recovery_rate,
                "recovery_rate_absorbed": all_recovery_rate_absorbed,
            },
            "control_random_zeroed": {
                "n_controls": len(control_results),
                "mean_accuracy": mean_control_accuracy,
                "mean_recovery_rate_recon": mean_control_recovery,
                "mean_recovery_rate_absorbed": mean_control_recovery_absorbed,
                "mean_prob_change": mean_control_prob_change,
                "details": control_results,
            },
            "per_child_analysis": per_child_results,
            "statistical_test": {
                "test": "one-sample t-test (control recovery_absorbed vs child recovery_absorbed)",
                "t_statistic": float(t_stat) if not np.isnan(t_stat) else None,
                "p_value": float(p_value) if not np.isnan(p_value) else None,
            },
        }
        results_per_word.append(word_result)

        report_progress(5 + word_idx, 5 + total_words, metrics={
            "status": f"patching_{word}",
            "words_done": word_idx,
            "total_words": total_words,
        })

        # Free GPU memory for this word
        del acts, sae_features, sae_recon, error, modified_features, modified_recon, modified_full
        del all_child_modified, all_child_recon
        torch.cuda.empty_cache()

    # ----- Step 6: Aggregate analysis -----
    logger.info("Step 6: Aggregating results and computing summary statistics...")

    completed = [r for r in results_per_word if r.get("status") == "completed"]

    # Summary statistics -- focus on recovery among absorbed tokens
    recovery_rates_child = []
    recovery_rates_child_absorbed = []
    recovery_rates_all = []
    recovery_rates_all_absorbed = []
    recovery_rates_control = []
    recovery_rates_control_absorbed = []
    absorption_rates = []
    prob_changes_child = []
    prob_changes_control = []

    for r in completed:
        recovery_rates_child.append(r["primary_child_zeroed"]["recovery_rate_recon"])
        recovery_rates_child_absorbed.append(r["primary_child_zeroed"]["recovery_rate_absorbed"])
        recovery_rates_all.append(r["all_children_zeroed"]["recovery_rate_recon"])
        recovery_rates_all_absorbed.append(r["all_children_zeroed"]["recovery_rate_absorbed"])
        recovery_rates_control.append(r["control_random_zeroed"]["mean_recovery_rate_recon"])
        recovery_rates_control_absorbed.append(r["control_random_zeroed"]["mean_recovery_rate_absorbed"])
        absorption_rates.append(r["sae_reconstruction"]["absorption_rate"])
        prob_changes_child.append(r["primary_child_zeroed"]["prob_change_vs_recon"])
        prob_changes_control.append(r["control_random_zeroed"]["mean_prob_change"])

    mean_recovery_child = float(np.mean(recovery_rates_child)) if recovery_rates_child else 0
    mean_recovery_child_absorbed = float(np.mean(recovery_rates_child_absorbed)) if recovery_rates_child_absorbed else 0
    mean_recovery_control = float(np.mean(recovery_rates_control)) if recovery_rates_control else 0
    mean_recovery_control_absorbed = float(np.mean(recovery_rates_control_absorbed)) if recovery_rates_control_absorbed else 0
    mean_absorption = float(np.mean(absorption_rates)) if absorption_rates else 0

    # Paired test across words: child recovery vs control recovery (among absorbed tokens)
    if len(recovery_rates_child_absorbed) >= 3:
        try:
            wilcoxon_stat, wilcoxon_p = stats.wilcoxon(
                recovery_rates_child_absorbed, recovery_rates_control_absorbed
            )
        except ValueError:
            # All differences are zero
            wilcoxon_stat, wilcoxon_p = 0.0, 1.0
    else:
        wilcoxon_stat, wilcoxon_p = float('nan'), float('nan')

    # Bootstrap confidence interval on mean recovery rate difference (absorbed tokens)
    diff = np.array(recovery_rates_child_absorbed) - np.array(recovery_rates_control_absorbed)
    if len(diff) >= 2:
        boot_diffs = []
        rng_boot = np.random.RandomState(SEED)
        for _ in range(N_BOOTSTRAP):
            idx = rng_boot.choice(len(diff), len(diff), replace=True)
            boot_diffs.append(diff[idx].mean())
        ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    else:
        ci_lower, ci_upper = float('nan'), float('nan')

    # Determine causal absorption evidence
    # Key question: does zeroing child features cause RECOVERY of correct letter prediction
    # specifically among absorbed tokens (where the SAE reconstruction lost the letter info)?
    any_significant_recovery = any(
        r["primary_child_zeroed"]["recovery_rate_absorbed"] > 0.05 for r in completed
        if r["sae_reconstruction"]["n_absorbed"] > 0
    )
    mean_recovery_above_control = mean_recovery_child_absorbed > mean_recovery_control_absorbed + 0.02

    if any_significant_recovery and mean_recovery_above_control:
        interpretation = "CAUSAL_ABSORPTION_CONFIRMED"
        interpretation_detail = (
            "Zeroing child features in the SAE causally recovers correct letter predictions "
            "among absorbed tokens (where probe failed on SAE reconstruction but succeeded on raw "
            "residual stream). This confirms that child features were suppressing parent (letter) "
            "information in the SAE's representation. True competitive exclusion / absorption."
        )
    elif any_significant_recovery:
        interpretation = "WEAK_CAUSAL_EVIDENCE"
        interpretation_detail = (
            "Some recovery observed when zeroing child features, but effect is small "
            "relative to control. Partial evidence for causal absorption."
        )
    else:
        interpretation = "NO_CAUSAL_ABSORPTION_ON_SAE_RECON"
        interpretation_detail = (
            "Zeroing child features in the SAE does NOT recover correct letter predictions "
            "among absorbed tokens. This could mean: (a) absorption operates through a different "
            "mechanism (e.g., the SAE's encoder actively suppresses letter features rather than "
            "decoder competition), or (b) the identified child features are not the actual "
            "absorbers for this SAE config, or (c) the probe trained on raw residual stream "
            "does not transfer perfectly to SAE reconstruction space."
        )

    summary = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae": {"release": SAE_RELEASE, "sae_id": SAE_ID},
        "n_core_words": len(CORE_WORDS),
        "n_completed": len(completed),
        "n_skipped": len(results_per_word) - len(completed),
        "probe_quality": {
            "train_f1": float(train_f1),
            "test_f1": float(test_f1),
            "test_accuracy": float(test_acc),
        },
        "aggregate": {
            "mean_absorption_rate": mean_absorption,
            "mean_recovery_rate_child_zeroed_recon": mean_recovery_child,
            "mean_recovery_rate_child_zeroed_absorbed": mean_recovery_child_absorbed,
            "mean_recovery_rate_control_recon": mean_recovery_control,
            "mean_recovery_rate_control_absorbed": mean_recovery_control_absorbed,
            "recovery_difference_absorbed": float(mean_recovery_child_absorbed - mean_recovery_control_absorbed),
            "recovery_difference_ci_95": [float(ci_lower), float(ci_upper)],
            "mean_prob_change_child": float(np.mean(prob_changes_child)) if prob_changes_child else 0,
            "mean_prob_change_control": float(np.mean(prob_changes_control)) if prob_changes_control else 0,
        },
        "statistical_tests": {
            "wilcoxon_signed_rank": {
                "statistic": float(wilcoxon_stat) if not np.isnan(wilcoxon_stat) else None,
                "p_value": float(wilcoxon_p) if not np.isnan(wilcoxon_p) else None,
                "note": "Paired test: child recovery rate vs control recovery rate across words",
            },
        },
        "interpretation": {
            "verdict": interpretation,
            "detail": interpretation_detail,
            "evidence_for_absorption": any_significant_recovery,
            "recovery_above_control": mean_recovery_above_control,
        },
        "per_word_results": results_per_word,
        "elapsed_minutes": round((time.time() - t_start) / 60, 1),
        "timestamp": datetime.now().isoformat(),
    }

    # Custom JSON encoder for numpy types
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

    # Save results
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2, cls=NumpyEncoder))
    logger.info(f"Results saved to {OUTPUT_FILE}")

    # Print summary table
    logger.info("\n" + "=" * 100)
    logger.info("ACTIVATION PATCHING RESULTS SUMMARY")
    logger.info("=" * 100)
    logger.info(f"{'Word':<10} {'Ltr':<5} {'Raw':<6} {'Recon':<6} {'Absd':<6} "
                f"{'Child0':<7} {'Rec_abs':<8} {'Ctrl_abs':<9} {'ProbChg':<8}")
    logger.info("-" * 100)
    for r in completed:
        logger.info(
            f"{r['word']:<10} {r['letter']:<5} "
            f"{r['raw_residual']['accuracy']:<6.3f} "
            f"{r['sae_reconstruction']['accuracy']:<6.3f} "
            f"{r['sae_reconstruction']['n_absorbed']:<6d} "
            f"{r['primary_child_zeroed']['accuracy_on_recon']:<7.3f} "
            f"{r['primary_child_zeroed']['recovery_rate_absorbed']:<8.3f} "
            f"{r['control_random_zeroed']['mean_recovery_rate_absorbed']:<9.3f} "
            f"{r['primary_child_zeroed']['prob_change_vs_recon']:<8.4f}"
        )
    logger.info("-" * 100)
    logger.info(f"Mean absorption rate: {mean_absorption:.3f}")
    logger.info(f"Mean recovery (child zeroed, absorbed): {mean_recovery_child_absorbed:.3f}")
    logger.info(f"Mean recovery (control zeroed, absorbed): {mean_recovery_control_absorbed:.3f}")
    logger.info(f"INTERPRETATION: {interpretation}")
    logger.info(f"DETAIL: {interpretation_detail}")
    logger.info(f"Elapsed: {summary['elapsed_minutes']:.1f} minutes")
    logger.info("=" * 100)

    mark_done(status="success", summary=f"{interpretation}: {len(completed)} words analyzed")
    return summary


if __name__ == "__main__":
    try:
        result = main()
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        mark_done(status="failure", summary=str(e))
        raise
