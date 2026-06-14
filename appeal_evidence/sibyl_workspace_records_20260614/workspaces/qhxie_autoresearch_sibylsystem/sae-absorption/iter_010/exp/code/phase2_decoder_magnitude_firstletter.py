"""
Phase 2.1: Decoder Direction Magnitude on First-Letter (H8 Cross-Domain Consistency)
Iteration 10, FULL mode.

Replicates the decoder direction ablation diagnostic from city-continent (iter_009)
on FIRST-LETTER at L24 16k. The iter_009 city-continent result was:
  - Mean logit change: -3.98 nats
  - 0% benign at all thresholds (0.05, 0.1, 0.2)
  - Random direction control: ~0.12 nats

This experiment tests whether the same "pathological" severity appears for
first-letter absorption, which has a different mechanism (F1=1.0 probes,
concentrated via main letter features) vs. city-continent (F1=0.84, distributed).

Method for each FN word:
  1. Load model + SAE L24 16k
  2. Build first-letter ICL prompt, get activations at token_pos=-6
  3. Apply sklearn probe to raw activation -> correct letter? (clean)
  4. SAE encode/decode -> apply probe -> correct letter? (SAE prediction)
  5. False negative = correct on raw, wrong on SAE (= absorption instance)
  6. For each FN instance:
     a. Identify child features (top-5 decoder-aligned with parent probe direction)
     b. Ablate parent direction: d_child_mod = d_child - proj_parent(d_child)
     c. Modified reconstruction = SAE_output - parent_contribution
     d. Measure probe logit change for true letter class
  7. Control: ablate random direction of same norm
  8. Classify benign/pathological at thresholds 0.05, 0.1, 0.2

Comparison with city-continent (iter_009):
  - Is mean |logit_change| comparable to 3.98 nats?
  - Is benign% similarly near-zero?
  - Does direction specificity hold (parent >> random)?
"""

import os
import sys
import json
import time
import gc
import random
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from scipy import stats

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase2_decoder_magnitude_firstletter"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE2_DIR = RESULTS_DIR / "phase2"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# SAE config -- L24 16k only (same as city-continent benchmark)
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_24/width_16k/canonical"
SAE_KEY = "L24_16k"
SAE_LAYER = 24

# Token position -- first-letter uses -6 (sae_spelling convention)
TOKEN_POS = -6

LETTERS = "abcdefghijklmnopqrstuvwxyz"

# Number of ICL prompts per word
# Need enough prompts to capture FN instances (absorption rate ~27%, so ~1 in 4 prompts are FN)
N_PROMPTS_PER_WORD = 10 if MODE == "PILOT" else 15

# Max words -- need ~200+ to ensure >= 20 FN instances given ~27% absorption rate
MAX_WORDS = 300 if MODE == "PILOT" else 500

# Benign/pathological thresholds
THRESHOLDS = [0.05, 0.1, 0.2]

# Number of control random directions per FN instance
N_CONTROL_DIRECTIONS = 5

# Bootstrap
N_BOOTSTRAP = 2000 if MODE == "PILOT" else 10000

# Iter 009 benign/pathological city-continent baseline for comparison
CITY_CONTINENT_BASELINE = {
    "mean_logit_change": -3.979,
    "abs_mean": 3.979,
    "control_abs_mean": 0.12,
    "benign_pct_01": 0.0,
    "n_instances": 1471,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Process tracking
# ============================================================
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(step, total_steps, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "status": status,
        "metric": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }))


def mark_done(status="success", summary=""):
    pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid_file.exists():
        pid_file.unlink()
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    final_progress = {}
    if progress_file.exists():
        try:
            final_progress = json.loads(progress_file.read_text())
        except Exception:
            pass
    done_data = json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    })
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(done_data)
    logger.info(f"DONE (status={status}): {summary}")


# ============================================================
# Model / SAE loading
# ============================================================
def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16
    )
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    return model


def load_sae(release, sae_id, device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    return sae


# ============================================================
# Load or train first-letter probe
# ============================================================
def load_probe(layer=24):
    """Load saved first-letter sklearn probe from iter_009."""
    # Try current iter first, then iter_009
    candidates = [
        PHASE1_DIR / f"probe_first-letter_L{layer}_sklearn.npz",
        WORKSPACE.parent / "iter_009" / "exp" / "results" / "phase1" / f"probe_first-letter_L{layer}_sklearn.npz",
    ]

    for probe_path in candidates:
        if probe_path.exists():
            data = np.load(probe_path, allow_pickle=True)
            coef = data["coef"]
            intercept = data["intercept"]
            classes = data["classes"]

            probe = LogisticRegression(max_iter=1)
            probe.classes_ = classes
            probe.coef_ = coef
            probe.intercept_ = intercept

            logger.info(f"  Loaded probe from {probe_path.name}: "
                        f"coef={coef.shape}, classes={len(classes)}")
            return probe, classes

    raise FileNotFoundError(f"First-letter probe not found at L{layer}")


# ============================================================
# Word list
# ============================================================
def get_word_list(tokenizer, max_words=200):
    """Get balanced word list from tokenizer vocab (same as absorption measurement)."""
    try:
        from sae_spelling.vocab import get_common_word_tokens
        common_tokens = get_common_word_tokens(tokenizer)
        words = []
        for tok in common_tokens:
            w = tok.strip()
            if w.startswith(" "):
                w = w[1:]
            if len(w) >= 3 and w.isascii() and w.isalpha():
                words.append(w.lower())
        words = sorted(set(words))
        logger.info(f"Common words from vocab: {len(words)}")
    except Exception as e:
        logger.warning(f"Common words failed ({e}), using alpha tokens")
        from sae_spelling.vocab import get_alpha_tokens
        alpha_tokens = get_alpha_tokens(tokenizer)
        words = []
        for tok in alpha_tokens:
            w = tok.strip()
            if w.startswith(" "):
                w = w[1:]
            if len(w) >= 3 and w.isascii() and w.isalpha():
                words.append(w.lower())
        words = sorted(set(words))

    # Balance across letters
    letter_words = defaultdict(list)
    for w in words:
        letter_words[w[0].lower()].append(w)

    rng = np.random.RandomState(SEED)
    per_letter = max(5, max_words // 26)
    balanced = []
    for letter in LETTERS:
        available = letter_words.get(letter, [])
        n = min(len(available), per_letter)
        if n > 0:
            chosen = rng.choice(available, size=n, replace=False).tolist()
            balanced.extend(chosen)

    rng.shuffle(balanced)
    if len(balanced) > max_words:
        balanced = list(balanced[:max_words])

    letter_coverage = len(set(w[0] for w in balanced))
    logger.info(f"Word list: {len(balanced)} words, {letter_coverage}/26 letters")
    return balanced


# ============================================================
# ICL prompt building (sae_spelling style)
# ============================================================
def build_prompts_for_word(word, word_list, n_prompts=3):
    """Build ICL prompts for one word using sae_spelling convention."""
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    prompts = []

    for prompt_idx in range(n_prompts):
        try:
            sp = create_icl_prompt(
                word=word,
                examples=word_list,
                base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                answer_formatter=formatter,
                max_icl_examples=10,
                shuffle_examples=True,
            )
            prompts.append(sp.base)
        except Exception:
            pass

    return prompts


# ============================================================
# Bootstrap CI
# ============================================================
def bootstrap_ci(values, n_bootstrap=2000, ci=0.95, seed=42):
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0, "n": 0}
    values = np.array(values, dtype=float)
    rng = np.random.RandomState(seed)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(values, size=len(values), replace=True)
        boot_means.append(float(np.mean(sample)))
    boot_means = sorted(boot_means)
    alpha = (1 - ci) / 2
    lo = boot_means[int(alpha * n_bootstrap)]
    hi = boot_means[min(int((1 - alpha) * n_bootstrap), len(boot_means) - 1)]
    return {
        "mean": float(np.mean(values)),
        "ci_lower": float(lo),
        "ci_upper": float(hi),
        "std": float(np.std(values)),
        "n": len(values),
    }


# ============================================================
# Core: decoder direction magnitude ablation for one word
# ============================================================
def decoder_magnitude_for_word(
    model, sae, probe, probe_classes,
    word, word_list,
    n_prompts=3,
    thresholds=None,
    n_control_dirs=5,
    layer=24,
    token_pos=-6,
    device="cuda:0",
):
    """
    Decoder direction magnitude measurement for one word.

    For each prompt context:
    1. Get residual stream at token_pos
    2. Probe on raw activation -> correct letter?
    3. SAE encode/decode -> probe on SAE output -> correct letter?
    4. If FN (absorption instance):
       a. Identify child features (top-5 aligned with parent probe direction)
       b. Ablate parent direction from child decoders
       c. Measure probe logit change
    5. Control: ablate random direction of same norm
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    tl_hook = f"blocks.{layer}.hook_resid_post"
    letter = word[0].lower()
    letter_idx = LETTERS.index(letter)

    # Map letter_idx to probe class index
    cls_list = probe_classes.tolist() if hasattr(probe_classes, 'tolist') else list(probe_classes)
    if letter_idx not in cls_list:
        return {"status": "letter_not_in_probe", "word": word, "letter": letter}
    probe_true_idx = cls_list.index(letter_idx)

    # Parent probe direction for the true letter
    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)
    d_parent = probe_coefs[probe_true_idx].clone()
    d_parent_norm_sq = (d_parent @ d_parent).item()
    if d_parent_norm_sq < 1e-12:
        return {"status": "degenerate_probe_direction", "word": word}

    d_parent_normalized = d_parent / (d_parent.norm() + 1e-8)

    # SAE decoder weights
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]

    # Build prompts
    prompts = build_prompts_for_word(word, word_list, n_prompts=n_prompts)
    if not prompts:
        return {"status": "no_prompts", "word": word}

    # Track per-context results
    logit_changes = []
    abs_logit_changes = []
    control_logit_changes = []
    context_details = []

    n_probe_correct_raw = 0
    n_fn = 0
    n_processed = 0

    for ctx_idx, prompt in enumerate(prompts):
        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)

            with torch.no_grad():
                logits_full, cache = model.run_with_cache(
                    tokens, names_filter=[tl_hook]
                )

            raw_act = cache[tl_hook][0, token_pos, :].detach().float()
            del cache, logits_full

            # Probe on raw activation
            raw_np = raw_act.cpu().numpy().reshape(1, -1)
            raw_pred = probe.predict(raw_np)[0]
            probe_correct_raw = (raw_pred == letter_idx)

            if not probe_correct_raw:
                # Probe wrong on raw -- skip (not an absorption case)
                continue

            n_probe_correct_raw += 1

            # SAE encode/decode
            raw_act_device = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_device.unsqueeze(0))
                sae_out = sae.decode(sae_features)

            sae_np = sae_out[0].detach().float().cpu().numpy().reshape(1, -1)
            sae_pred = probe.predict(sae_np)[0]
            probe_correct_sae = (sae_pred == letter_idx)

            if probe_correct_sae:
                # Not a false negative -- skip
                continue

            # ---- FALSE NEGATIVE (absorption instance) ----
            n_fn += 1
            sae_features_cpu = sae_features[0].detach().float().cpu()

            # Identify active features
            active_mask = sae_features_cpu.abs() > 1e-6
            active_indices = torch.where(active_mask)[0]

            if len(active_indices) == 0:
                continue

            # Cosine similarity between active features' decoders and parent direction
            active_decoder = W_dec[active_indices]  # [n_active, d_model]
            cos_sims = F.cosine_similarity(
                d_parent_normalized.unsqueeze(0),
                active_decoder,
                dim=-1
            )

            # Child features: top-5 most aligned with parent direction
            n_child = min(5, len(active_indices))
            top_cos_vals, top_cos_idx = cos_sims.topk(n_child)
            child_feature_indices = active_indices[top_cos_idx].numpy()
            child_cos_sims = top_cos_vals.numpy()

            if len(child_feature_indices) == 0:
                continue

            n_processed += 1

            # ===== ABLATION: Remove parent direction component from child decoders =====
            sae_out_cpu = sae_out[0].detach().float().cpu()

            # Compute the contribution of parent direction through child features
            parent_contribution = torch.zeros(W_dec.shape[1])
            for fid in child_feature_indices:
                act_val = sae_features_cpu[fid].item()
                d_child = W_dec[fid]
                proj_coef = (d_child @ d_parent) / d_parent_norm_sq
                parent_component = proj_coef * d_parent
                parent_contribution += act_val * parent_component

            # Modified reconstruction: subtract parent direction contribution
            modified_reconstruction = sae_out_cpu - parent_contribution

            # ===== Measure: Probe logit change =====
            mod_np = modified_reconstruction.numpy().reshape(1, -1)
            orig_probe_logits = probe.decision_function(sae_np)[0]
            mod_probe_logits = probe.decision_function(mod_np)[0]

            # Logit for true letter class
            orig_probe_logit_true = orig_probe_logits[probe_true_idx]
            mod_probe_logit_true = mod_probe_logits[probe_true_idx]
            probe_logit_change = float(mod_probe_logit_true - orig_probe_logit_true)

            # Check if probe recovers after ablation
            mod_pred = probe.predict(mod_np)[0]
            probe_recovers = (mod_pred == letter_idx)

            # ===== Control: ablate random direction of same norm =====
            parent_dir_norm = d_parent.norm().item()
            control_changes = []
            rng_ctrl = np.random.RandomState(SEED + hash(word) % (2**31) + ctx_idx)

            for ctrl_i in range(n_control_dirs):
                random_dir = torch.randn_like(d_parent)
                random_dir = random_dir / (random_dir.norm() + 1e-8) * parent_dir_norm

                ctrl_contribution = torch.zeros(W_dec.shape[1])
                random_norm_sq = (random_dir @ random_dir).item()
                for fid in child_feature_indices:
                    act_val = sae_features_cpu[fid].item()
                    d_child = W_dec[fid]
                    proj_coef = (d_child @ random_dir) / random_norm_sq
                    ctrl_component = proj_coef * random_dir
                    ctrl_contribution += act_val * ctrl_component

                ctrl_reconstruction = sae_out_cpu - ctrl_contribution
                ctrl_np = ctrl_reconstruction.numpy().reshape(1, -1)
                ctrl_probe_logits = probe.decision_function(ctrl_np)[0]
                ctrl_change = float(ctrl_probe_logits[probe_true_idx] - orig_probe_logit_true)
                control_changes.append(ctrl_change)

            mean_control_change = float(np.mean(control_changes)) if control_changes else 0.0

            # Record results
            logit_changes.append(probe_logit_change)
            abs_logit_changes.append(abs(probe_logit_change))
            control_logit_changes.append(mean_control_change)

            if len(context_details) < 10:
                context_details.append({
                    "ctx_idx": ctx_idx,
                    "word": word,
                    "letter": letter,
                    "probe_logit_change": probe_logit_change,
                    "control_logit_change": mean_control_change,
                    "probe_recovers_after_ablation": bool(probe_recovers),
                    "n_child_features": len(child_feature_indices),
                    "top_child_cos_sims": child_cos_sims[:3].tolist(),
                    "parent_contribution_norm": float(parent_contribution.norm().item()),
                })

            del sae_out, sae_features
            torch.cuda.empty_cache()

        except Exception as e:
            if ctx_idx < 2:
                logger.warning(f"  Context {ctx_idx} error: {e}")
            continue

    if n_fn == 0 or n_processed == 0:
        return {
            "status": "no_fn_found",
            "word": word,
            "letter": letter,
            "n_probe_correct_raw": n_probe_correct_raw,
            "n_fn": n_fn,
            "n_processed": n_processed,
        }

    # Classify benign vs pathological at each threshold
    abs_changes = np.array(abs_logit_changes)
    classifications = {}
    for thr in thresholds:
        n_benign = int(np.sum(abs_changes <= thr))
        n_pathological = int(np.sum(abs_changes > thr))
        classifications[f"threshold_{thr}"] = {
            "threshold": thr,
            "n_benign": n_benign,
            "n_pathological": n_pathological,
            "benign_pct": float(n_benign / len(abs_changes)) if len(abs_changes) > 0 else 0.0,
            "pathological_pct": float(n_pathological / len(abs_changes)) if len(abs_changes) > 0 else 0.0,
        }

    return {
        "status": "completed",
        "word": word,
        "letter": letter,
        "n_probe_correct_raw": n_probe_correct_raw,
        "n_fn": n_fn,
        "n_processed": n_processed,
        "classifications": classifications,
        "logit_change_stats": {
            "mean": float(np.mean(logit_changes)) if logit_changes else 0.0,
            "std": float(np.std(logit_changes)) if logit_changes else 0.0,
            "median": float(np.median(logit_changes)) if logit_changes else 0.0,
            "min": float(np.min(logit_changes)) if logit_changes else 0.0,
            "max": float(np.max(logit_changes)) if logit_changes else 0.0,
            "abs_mean": float(np.mean(abs_logit_changes)) if abs_logit_changes else 0.0,
        },
        "control_logit_change_stats": {
            "mean": float(np.mean(control_logit_changes)) if control_logit_changes else 0.0,
            "std": float(np.std(control_logit_changes)) if control_logit_changes else 0.0,
            "abs_mean": float(np.mean([abs(c) for c in control_logit_changes])) if control_logit_changes else 0.0,
        },
        "logit_changes": logit_changes[:50],
        "control_changes": control_logit_changes[:50],
        "context_details": context_details,
    }


# ============================================================
# GPU progress / experiment state tracking
# ============================================================
_GLOBAL_START_TIME = None

def update_gpu_progress(elapsed_seconds, status="completed"):
    import filelock
    progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / "gpu_progress.lock"
    end_time = datetime.now()
    start_time_str = (_GLOBAL_START_TIME or end_time).isoformat()
    end_time_str = end_time.isoformat()
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            data.setdefault("timings", {})[TASK_ID] = {
                "planned_min": 30,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": start_time_str,
                "end_time": end_time_str,
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "sae": SAE_KEY,
                    "layer": SAE_LAYER,
                    "token_pos": TOKEN_POS,
                    "n_prompts_per_word": N_PROMPTS_PER_WORD,
                    "max_words": MAX_WORDS,
                    "thresholds": THRESHOLDS,
                    "gpu_model": "NVIDIA RTX PRO 6000 Blackwell",
                    "gpu_count": 1,
                },
            }
            progress_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"gpu_progress update failed: {e}")
        try:
            data = json.loads(progress_path.read_text()) if progress_path.exists() else {
                "completed": [], "failed": [], "running": {}, "timings": {}
            }
            if status == "completed":
                if TASK_ID not in data.get("completed", []):
                    data.setdefault("completed", []).append(TASK_ID)
            else:
                if TASK_ID not in data.get("failed", []):
                    data.setdefault("failed", []).append(TASK_ID)
            data.get("running", {}).pop(TASK_ID, None)
            progress_path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass


def update_experiment_state(status, error_summary=None):
    import filelock
    state_path = WORKSPACE / "exp" / "experiment_state.json"
    lock_path = WORKSPACE / "exp" / "experiment_state.lock"
    try:
        lock = filelock.FileLock(str(lock_path), timeout=10)
        with lock:
            data = json.loads(state_path.read_text()) if state_path.exists() else {
                "schema_version": 1, "tasks": {}
            }
            if TASK_ID in data.get("tasks", {}):
                data["tasks"][TASK_ID]["status"] = status
                if status == "completed":
                    data["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()
                if error_summary:
                    data["tasks"][TASK_ID]["error_summary"] = error_summary
            state_path.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logger.warning(f"experiment_state update failed: {e}")


# ============================================================
# Main
# ============================================================
def main():
    global _GLOBAL_START_TIME
    start_time = time.time()
    _GLOBAL_START_TIME = datetime.now()
    write_pid()
    report_progress(0, 10, "starting")

    logger.info("=" * 70)
    logger.info("Phase 2.1: Decoder Direction Magnitude on First-Letter (H8)")
    logger.info(f"Mode: {MODE}, Layer: {SAE_LAYER}, Token pos: {TOKEN_POS}")
    logger.info(f"Thresholds: {THRESHOLDS}")
    logger.info(f"N prompts per word: {N_PROMPTS_PER_WORD}")
    logger.info(f"Max words: {MAX_WORDS}")
    logger.info("=" * 70)

    device = "cuda:0"

    # Step 1: Load model
    report_progress(1, 10, "loading_model")
    model = load_model(device=device)

    # Step 2: Load SAE
    report_progress(2, 10, "loading_sae")
    sae = load_sae(SAE_RELEASE, SAE_ID, device=device)

    # Step 3: Load first-letter probe
    report_progress(3, 10, "loading_probe")
    probe, probe_classes = load_probe(layer=SAE_LAYER)
    logger.info(f"  Probe classes: {len(probe_classes)}, shape: {probe.coef_.shape}")

    # Step 4: Get word list
    report_progress(4, 10, "building_word_list")
    word_list = get_word_list(model.tokenizer, max_words=MAX_WORDS)

    # Step 5: Run decoder magnitude ablation for each word
    report_progress(5, 10, "running_ablation",
                    {"n_words": len(word_list)})

    all_word_results = []
    all_logit_changes = []
    all_control_changes = []
    all_abs_changes = []
    per_letter_results = defaultdict(lambda: {
        "n_words": 0, "n_fn": 0, "n_processed": 0,
        "logit_changes": [], "control_changes": [],
        "classifications": {f"threshold_{t}": {"n_benign": 0, "n_pathological": 0}
                           for t in THRESHOLDS}
    })

    n_completed_words = 0
    n_fn_words = 0

    for i, word in enumerate(word_list):
        letter = word[0].lower()
        if (i + 1) % 20 == 0 or i < 5:
            logger.info(f"\n  [{i+1}/{len(word_list)}] Processing '{word}' (letter: {letter})...")

        report_progress(5, 10, f"word_{i+1}/{len(word_list)}",
                        {"word": word, "letter": letter,
                         "fn_words_so_far": n_fn_words})

        result = decoder_magnitude_for_word(
            model=model,
            sae=sae,
            probe=probe,
            probe_classes=probe_classes,
            word=word,
            word_list=word_list,
            n_prompts=N_PROMPTS_PER_WORD,
            thresholds=THRESHOLDS,
            n_control_dirs=N_CONTROL_DIRECTIONS,
            layer=SAE_LAYER,
            token_pos=TOKEN_POS,
            device=device,
        )

        all_word_results.append(result)

        if result["status"] == "completed":
            n_completed_words += 1
            n_fn_words += 1

            if (i + 1) % 20 == 0 or i < 5:
                logger.info(f"    FN: {result['n_fn']}, processed: {result['n_processed']}")
                logger.info(f"    Mean |logit_change|: {result['logit_change_stats']['abs_mean']:.4f}")

            all_logit_changes.extend(result.get("logit_changes", []))
            all_control_changes.extend(result.get("control_changes", []))
            all_abs_changes.extend([abs(x) for x in result.get("logit_changes", [])])

            # Per-letter aggregation
            pld = per_letter_results[letter]
            pld["n_words"] += 1
            pld["n_fn"] += result["n_fn"]
            pld["n_processed"] += result["n_processed"]
            pld["logit_changes"].extend(result.get("logit_changes", []))
            pld["control_changes"].extend(result.get("control_changes", []))
            for thr in THRESHOLDS:
                thr_key = f"threshold_{thr}"
                if thr_key in result.get("classifications", {}):
                    pld["classifications"][thr_key]["n_benign"] += result["classifications"][thr_key]["n_benign"]
                    pld["classifications"][thr_key]["n_pathological"] += result["classifications"][thr_key]["n_pathological"]

        if (i + 1) % 50 == 0:
            gc.collect()
            torch.cuda.empty_cache()
            # Intermediate checkpoint
            elapsed_so_far = time.time() - start_time
            logger.info(f"  --- Checkpoint {i+1}/{len(word_list)}: "
                        f"FN words={n_fn_words}, "
                        f"total_instances={len(all_logit_changes)}, "
                        f"elapsed={elapsed_so_far/60:.1f}min ---")

    # Step 6: Aggregate results
    report_progress(6, 10, "aggregating")
    logger.info(f"\n{'='*70}")
    logger.info("Aggregate Results -- First-Letter Decoder Magnitude")
    logger.info(f"{'='*70}")

    completed = [r for r in all_word_results if r["status"] == "completed"]
    n_completed = len(completed)
    total_fn = sum(r["n_fn"] for r in completed)
    total_processed = sum(r["n_processed"] for r in completed)

    logger.info(f"  Completed words: {n_completed}/{len(word_list)}")
    logger.info(f"  Words with FN: {n_fn_words}")
    logger.info(f"  Total FN instances: {total_fn}")
    logger.info(f"  Total processed FN instances: {total_processed}")

    # Aggregate classifications at each threshold
    aggregate_classifications = {}
    for thr in THRESHOLDS:
        thr_key = f"threshold_{thr}"
        total_benign = sum(r["classifications"][thr_key]["n_benign"]
                          for r in completed if thr_key in r.get("classifications", {}))
        total_pathological = sum(r["classifications"][thr_key]["n_pathological"]
                                for r in completed if thr_key in r.get("classifications", {}))
        total_n = total_benign + total_pathological
        aggregate_classifications[thr_key] = {
            "threshold": thr,
            "n_benign": total_benign,
            "n_pathological": total_pathological,
            "n_total": total_n,
            "benign_pct": float(total_benign / max(total_n, 1)),
            "pathological_pct": float(total_pathological / max(total_n, 1)),
        }
        logger.info(f"  {thr_key}: benign={total_benign}/{total_n} "
                    f"({aggregate_classifications[thr_key]['benign_pct']:.1%}), "
                    f"pathological={total_pathological}/{total_n}")

    # Control aggregate
    ctrl_abs = [abs(c) for c in all_control_changes]
    ctrl_benign_01 = sum(1 for c in ctrl_abs if c <= 0.1) / max(len(ctrl_abs), 1)
    logger.info(f"  Control benign% at 0.1: {ctrl_benign_01:.1%}")

    # Logit change distribution statistics
    if all_logit_changes:
        logit_dist = {
            "mean": float(np.mean(all_logit_changes)),
            "std": float(np.std(all_logit_changes)),
            "median": float(np.median(all_logit_changes)),
            "min": float(np.min(all_logit_changes)),
            "max": float(np.max(all_logit_changes)),
            "abs_mean": float(np.mean(all_abs_changes)),
            "abs_median": float(np.median(all_abs_changes)),
            "n": len(all_logit_changes),
            "percentiles": {
                "5": float(np.percentile(all_logit_changes, 5)),
                "25": float(np.percentile(all_logit_changes, 25)),
                "50": float(np.percentile(all_logit_changes, 50)),
                "75": float(np.percentile(all_logit_changes, 75)),
                "95": float(np.percentile(all_logit_changes, 95)),
            },
        }
    else:
        logit_dist = {"n": 0}

    if all_control_changes:
        control_dist = {
            "mean": float(np.mean(all_control_changes)),
            "std": float(np.std(all_control_changes)),
            "abs_mean": float(np.mean([abs(c) for c in all_control_changes])),
            "benign_pct_at_01": ctrl_benign_01,
            "n": len(all_control_changes),
        }
    else:
        control_dist = {"n": 0}

    # Step 7: Statistical tests
    report_progress(7, 10, "statistical_tests")
    stat_tests = {}

    # Test 1: Are logit changes significantly different from zero?
    if len(all_logit_changes) >= 3:
        t_stat, t_p = stats.ttest_1samp(all_logit_changes, 0)
        stat_tests["ttest_vs_zero"] = {
            "t_statistic": float(t_stat),
            "p_value": float(t_p),
            "significant_005": bool(t_p < 0.05),
            "mean": float(np.mean(all_logit_changes)),
        }
        logger.info(f"\n  t-test vs zero: t={t_stat:.4f}, p={t_p:.6f}")

    # Test 2: Parent-direction vs random direction
    if len(all_logit_changes) >= 3 and len(all_control_changes) >= 3:
        parent_abs = np.array([abs(c) for c in all_logit_changes])
        control_abs = np.array([abs(c) for c in all_control_changes])

        min_len = min(len(parent_abs), len(control_abs))
        if min_len >= 3:
            try:
                ws, wp = stats.wilcoxon(
                    parent_abs[:min_len], control_abs[:min_len],
                    alternative='greater'
                )
                stat_tests["wilcoxon_parent_vs_control"] = {
                    "statistic": float(ws),
                    "p_value": float(wp),
                    "n": min_len,
                    "significant_005": bool(wp < 0.05),
                    "mean_parent_abs": float(np.mean(parent_abs)),
                    "mean_control_abs": float(np.mean(control_abs)),
                }
                logger.info(f"  Wilcoxon parent vs control: p={wp:.6f}")
            except Exception as e:
                stat_tests["wilcoxon_parent_vs_control"] = {"error": str(e)}

        u_stat, u_p = stats.mannwhitneyu(
            parent_abs, control_abs, alternative='greater'
        )
        stat_tests["mannwhitney_parent_vs_control"] = {
            "u_statistic": float(u_stat),
            "p_value": float(u_p),
            "n_parent": len(parent_abs),
            "n_control": len(control_abs),
            "significant_005": bool(u_p < 0.05),
        }
        logger.info(f"  Mann-Whitney U: U={u_stat:.1f}, p={u_p:.6f}")

    # Test 3: Bootstrap CI on benign percentage
    if all_abs_changes:
        for thr in THRESHOLDS:
            benign_indicators = [1 if c <= thr else 0 for c in all_abs_changes]
            ci = bootstrap_ci(benign_indicators, n_bootstrap=N_BOOTSTRAP, seed=SEED)
            stat_tests[f"benign_pct_ci_threshold_{thr}"] = ci
            logger.info(f"  Benign% CI at threshold {thr}: "
                       f"[{ci['ci_lower']:.3f}, {ci['ci_upper']:.3f}] (mean={ci['mean']:.3f})")

    # Test 4: Compare with city-continent baseline (two-sample test)
    if len(all_logit_changes) >= 10:
        # Compare mean |logit_change| with city-continent baseline
        stat_tests["cross_hierarchy_comparison"] = {
            "first_letter_abs_mean": logit_dist.get("abs_mean", 0),
            "city_continent_abs_mean": CITY_CONTINENT_BASELINE["abs_mean"],
            "ratio": logit_dist.get("abs_mean", 0) / max(CITY_CONTINENT_BASELINE["abs_mean"], 1e-8),
            "first_letter_n": len(all_logit_changes),
            "city_continent_n": CITY_CONTINENT_BASELINE["n_instances"],
            "first_letter_benign_pct_01": aggregate_classifications.get("threshold_0.1", {}).get("benign_pct", 0),
            "city_continent_benign_pct_01": CITY_CONTINENT_BASELINE["benign_pct_01"],
        }

    # Step 8: Per-letter summary
    report_progress(8, 10, "per_letter_analysis")
    per_letter_summary = {}
    for letter, pld in sorted(per_letter_results.items()):
        total_at_01 = (pld["classifications"]["threshold_0.1"]["n_benign"] +
                       pld["classifications"]["threshold_0.1"]["n_pathological"])
        per_letter_summary[letter] = {
            "n_words": pld["n_words"],
            "n_fn": pld["n_fn"],
            "n_processed": pld["n_processed"],
            "mean_abs_logit_change": float(np.mean([abs(c) for c in pld["logit_changes"]])) if pld["logit_changes"] else 0.0,
            "mean_control_abs": float(np.mean([abs(c) for c in pld["control_changes"]])) if pld["control_changes"] else 0.0,
            "benign_pct_01": float(pld["classifications"]["threshold_0.1"]["n_benign"] / max(total_at_01, 1)),
            "n_at_01": total_at_01,
        }
        if pld["n_processed"] > 0:
            logger.info(f"  {letter}: words={pld['n_words']}, processed={pld['n_processed']}, "
                        f"mean|delta|={per_letter_summary[letter]['mean_abs_logit_change']:.4f}")

    # Step 9: H8 cross-domain consistency verdict
    report_progress(9, 10, "hypothesis_verdict")

    parent_mean_abs = logit_dist.get("abs_mean", 0) if logit_dist.get("n", 0) > 0 else 0
    control_mean_abs = control_dist.get("abs_mean", 0) if control_dist.get("n", 0) > 0 else 0
    benign_pct_01 = aggregate_classifications.get("threshold_0.1", {}).get("benign_pct", 0)
    direction_specificity = parent_mean_abs > control_mean_abs * 1.5

    # Cross-hierarchy consistency check
    cc_baseline_abs = CITY_CONTINENT_BASELINE["abs_mean"]
    ratio_to_cc = parent_mean_abs / max(cc_baseline_abs, 1e-8)
    consistent = 0.3 <= ratio_to_cc <= 3.0  # Within order of magnitude

    if consistent and direction_specificity:
        consistency_verdict = "CONSISTENT"
        consistency_detail = (
            f"First-letter mean |logit_change|={parent_mean_abs:.3f} nats "
            f"vs city-continent {cc_baseline_abs:.3f} nats (ratio={ratio_to_cc:.2f}). "
            f"Both show strong direction specificity. "
            f"Pathological severity is consistent across hierarchy types."
        )
    elif direction_specificity and not consistent:
        consistency_verdict = "MAGNITUDE_DIFFERS"
        consistency_detail = (
            f"First-letter mean |logit_change|={parent_mean_abs:.3f} nats "
            f"vs city-continent {cc_baseline_abs:.3f} nats (ratio={ratio_to_cc:.2f}). "
            f"Direction specificity holds, but severity differs between hierarchies. "
            f"This may reflect different absorption mechanisms."
        )
    else:
        consistency_verdict = "INCONSISTENT"
        consistency_detail = (
            f"First-letter mean |logit_change|={parent_mean_abs:.3f} nats "
            f"vs city-continent {cc_baseline_abs:.3f} nats. "
            f"Direction specificity: {'YES' if direction_specificity else 'NO'}. "
            f"Cross-domain consistency not established."
        )

    elapsed = time.time() - start_time

    # Pilot pass criteria
    pilot_pass = (
        n_completed >= 10 and
        total_processed >= 20 and
        logit_dist.get("n", 0) >= 20 and
        len(THRESHOLDS) >= 3
    )

    # Step 10: Compile output
    report_progress(10, 10, "compiling_output")

    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae": {"release": SAE_RELEASE, "sae_id": SAE_ID, "key": SAE_KEY},
        "layer": SAE_LAYER,
        "token_pos": TOKEN_POS,
        "n_prompts_per_word": N_PROMPTS_PER_WORD,
        "thresholds": THRESHOLDS,
        "n_control_directions": N_CONTROL_DIRECTIONS,
        "hierarchy": "first-letter",
        "word_counts": {
            "n_words_input": len(word_list),
            "n_completed": n_completed,
            "n_fn_words": n_fn_words,
            "total_fn_instances": total_fn,
            "total_processed": total_processed,
        },
        "aggregate_classifications": aggregate_classifications,
        "logit_change_distribution": logit_dist,
        "control_distribution": control_dist,
        "statistical_tests": stat_tests,
        "per_letter_summary": per_letter_summary,
        "cross_hierarchy_comparison": {
            "first_letter": {
                "abs_mean": parent_mean_abs,
                "control_abs_mean": control_mean_abs,
                "benign_pct_01": benign_pct_01,
                "n_instances": logit_dist.get("n", 0),
                "direction_specificity": direction_specificity,
            },
            "city_continent_baseline": CITY_CONTINENT_BASELINE,
            "ratio": ratio_to_cc,
            "consistency_verdict": consistency_verdict,
            "consistency_detail": consistency_detail,
        },
        "pilot_criteria": {
            "met": pilot_pass,
            "details": {
                "enough_words_10": n_completed >= 10,
                "enough_processed_20": total_processed >= 20,
                "enough_measurements_20": logit_dist.get("n", 0) >= 20,
                "three_thresholds": len(THRESHOLDS) >= 3,
                "control_computed": control_dist.get("n", 0) > 0,
            },
        },
        "per_word_results": [
            {k: v for k, v in r.items()
             if k not in ("logit_changes", "control_changes", "context_details")}
            for r in all_word_results
        ],
        "sample_context_details": [
            d for r in completed[:10]
            for d in r.get("context_details", [])[:2]
        ],
        "logit_change_histogram": {
            "bins": list(np.arange(-10.0, 2.05, 0.2)),
            "counts": list(np.histogram(all_logit_changes,
                                         bins=np.arange(-10.0, 2.05, 0.2))[0].tolist())
            if all_logit_changes else [],
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save to phase2 directory (primary output)
    out_path = PHASE2_DIR / "decoder_magnitude_firstletter.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    # Also save to pilots
    pilot_path = PILOT_DIR / f"{TASK_ID}.json"
    pilot_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Also saved: {pilot_path}")

    # Generate summary markdown
    summary_md = generate_summary_md(output)
    md_path = PHASE2_DIR / f"{TASK_ID}_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    summary_text = (
        f"Phase 2.1 decoder magnitude first-letter ({MODE}). "
        f"Words: {n_completed}, FN processed: {total_processed}. "
        f"Mean |logit_change|: {parent_mean_abs:.3f} nats "
        f"(city-continent baseline: {cc_baseline_abs:.3f}). "
        f"Benign%: {benign_pct_01:.1%} at threshold 0.1. "
        f"Cross-hierarchy: {consistency_verdict}. "
        f"Pilot: {'PASS' if pilot_pass else 'FAIL'}. "
        f"Time: {elapsed/60:.1f}min."
    )

    mark_done("success" if pilot_pass else "partial", summary_text)
    update_gpu_progress(elapsed, "completed")
    update_experiment_state("completed")

    logger.info(f"\n{'='*70}")
    logger.info(f"COMPLETED: {summary_text}")
    logger.info(f"{'='*70}")

    return output


def generate_summary_md(results):
    """Generate human-readable markdown summary."""
    lines = [
        "# Phase 2.1: Decoder Direction Magnitude on First-Letter (H8 Cross-Domain)",
        "",
        f"**Cross-Hierarchy Verdict**: {results.get('cross_hierarchy_comparison', {}).get('consistency_verdict', 'UNKNOWN')}",
        f"**Pilot**: {'PASS' if results.get('pilot_criteria', {}).get('met') else 'FAIL'}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        "",
        "## Design",
        "",
        f"- **Hierarchy**: first-letter at L{results.get('layer', '?')} (token_pos={results.get('token_pos', '?')})",
        f"- **SAE**: {results.get('sae', {}).get('key', '?')}",
        f"- **Thresholds**: {results.get('thresholds', [])}",
        f"- **Prompts per word**: {results.get('n_prompts_per_word', '?')}",
        f"- **Control directions**: {results.get('n_control_directions', '?')}",
        "",
        "## Cross-Hierarchy Comparison",
        "",
        f"{results.get('cross_hierarchy_comparison', {}).get('consistency_detail', '')}",
        "",
        "| Metric | First-Letter | City-Continent |",
        "|--------|-------------|----------------|",
    ]

    fl = results.get("cross_hierarchy_comparison", {}).get("first_letter", {})
    cc = results.get("cross_hierarchy_comparison", {}).get("city_continent_baseline", {})
    lines.extend([
        f"| Mean |logit change| | {fl.get('abs_mean', 0):.3f} | {cc.get('abs_mean', 0):.3f} |",
        f"| Control |logit change| | {fl.get('control_abs_mean', 0):.3f} | {cc.get('control_abs_mean', 0):.3f} |",
        f"| Benign % (0.1) | {fl.get('benign_pct_01', 0):.1%} | {cc.get('benign_pct_01', 0):.1%} |",
        f"| N instances | {fl.get('n_instances', 0)} | {cc.get('n_instances', 0)} |",
        f"| Direction specificity | {'YES' if fl.get('direction_specificity') else 'NO'} | YES |",
    ])

    lines.extend([
        "",
        "## Classification at Multiple Thresholds",
        "",
        "| Threshold | Benign | Pathological | Benign % |",
        "|-----------|--------|--------------|----------|",
    ])

    for thr_key, cls in results.get("aggregate_classifications", {}).items():
        lines.append(
            f"| {cls.get('threshold', '?')} | {cls.get('n_benign', 0)} | "
            f"{cls.get('n_pathological', 0)} | {cls.get('benign_pct', 0):.1%} |"
        )

    # Logit change distribution
    dist = results.get("logit_change_distribution", {})
    if dist.get("n", 0) > 0:
        lines.extend([
            "",
            "## Logit Change Distribution",
            "",
            f"- **Mean**: {dist.get('mean', 0):.4f} +/- {dist.get('std', 0):.4f}",
            f"- **Median**: {dist.get('median', 0):.4f}",
            f"- **|Mean|**: {dist.get('abs_mean', 0):.4f}",
            f"- **Range**: [{dist.get('min', 0):.4f}, {dist.get('max', 0):.4f}]",
            f"- **N**: {dist.get('n', 0)}",
        ])
        percs = dist.get("percentiles", {})
        if percs:
            lines.append(f"- **Percentiles**: 5th={percs.get('5', 0):.3f}, "
                         f"25th={percs.get('25', 0):.3f}, "
                         f"50th={percs.get('50', 0):.3f}, "
                         f"75th={percs.get('75', 0):.3f}, "
                         f"95th={percs.get('95', 0):.3f}")

    # Statistical tests
    st = results.get("statistical_tests", {})
    if st:
        lines.extend(["", "## Statistical Tests", ""])
        if "ttest_vs_zero" in st:
            tt = st["ttest_vs_zero"]
            lines.append(f"- **t-test vs zero**: t={tt.get('t_statistic', 0):.4f}, "
                        f"p={tt.get('p_value', 1):.6f}, "
                        f"significant: {tt.get('significant_005', False)}")
        if "mannwhitney_parent_vs_control" in st:
            mw = st["mannwhitney_parent_vs_control"]
            lines.append(f"- **Mann-Whitney (parent vs control)**: "
                        f"U={mw.get('u_statistic', 0):.1f}, "
                        f"p={mw.get('p_value', 1):.6f}, "
                        f"significant: {mw.get('significant_005', False)}")

    # Per-letter breakdown (top 10 by n_processed)
    pls = results.get("per_letter_summary", {})
    if pls:
        sorted_letters = sorted(pls.items(),
                                key=lambda x: x[1].get("n_processed", 0),
                                reverse=True)
        lines.extend([
            "",
            "## Per-Letter Results (top by n_processed)",
            "",
            "| Letter | Words | FN Processed | Mean |delta| | Control |delta| | Benign% @0.1 |",
            "|--------|-------|-------------|------------|---------------|--------------|",
        ])
        for letter, d in sorted_letters[:15]:
            if d.get("n_processed", 0) > 0:
                lines.append(
                    f"| {letter} | {d.get('n_words', 0)} | {d.get('n_processed', 0)} | "
                    f"{d.get('mean_abs_logit_change', 0):.4f} | "
                    f"{d.get('mean_control_abs', 0):.4f} | "
                    f"{d.get('benign_pct_01', 0):.1%} |"
                )

    # Word counts
    wc = results.get("word_counts", {})
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Words tested: {wc.get('n_words_input', 0)}",
        f"- Words with absorption: {wc.get('n_fn_words', 0)}",
        f"- Total FN instances: {wc.get('total_fn_instances', 0)}",
        f"- Total processed: {wc.get('total_processed', 0)}",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        try:
            update_gpu_progress(0, "failed")
            update_experiment_state("failed", str(e))
        except Exception:
            pass
        sys.exit(1)
