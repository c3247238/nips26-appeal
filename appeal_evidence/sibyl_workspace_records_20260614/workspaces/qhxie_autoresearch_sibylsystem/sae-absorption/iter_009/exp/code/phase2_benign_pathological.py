"""
Phase 2.2: Benign vs. Pathological Absorption Diagnostic (H8) -- Iteration 9 Pilot

First-ever classification of absorption instances by downstream causal impact.

For each confirmed absorption instance (entity with FN in Phase 1):
  1. Load model + SAE at L24.
  2. For each absorbed token context:
     a. Get SAE reconstruction and identify active child features (probe-aligned absorbers).
     b. Compute parent direction d_parent from probe weights.
     c. Ablate parent direction component from child latent's decoder vector:
        d_child_modified = d_child - (d_child . d_parent / ||d_parent||^2) * d_parent
     d. Run model with modified SAE decoder (replace decoder temporarily), get new reconstruction.
     e. Compute original model logits and modified logits at the token position.
     f. Measure logit change for parent-relevant tokens.
  3. Classify: benign (|logit_change| <= threshold) vs pathological (|logit_change| > threshold).
  4. Test at thresholds 0.05, 0.1, 0.2 for robustness.
  5. Control: ablate random direction of same norm from decoder.

PILOT configuration:
- Hierarchy: city-continent at L24 (primary), first-letter at L24 (if time permits)
- SAE: L24-16k JumpReLU
- Sample up to 50 entities with FN from Phase 1/Phase 2 results
- Target: >= 30% benign (H8 hypothesis)
- Falsification: < 10% benign across all hierarchies

Dependencies:
- phase2_activation_patching_crossdomain (COMPLETED): confirmed absorption pairs
- phase1_absorption_crossdomain (COMPLETED): FN entities + main features
- phase1_absorption_firstletter (COMPLETED): first-letter FN entities
- phase1_probe_training (COMPLETED): trained probes

Prior evidence (iter_008):
- Activation patching confirmed competitive exclusion (p=0.000218 for first-letter)
- Cross-domain patching showed city-continent child recovery ~0.05% (very low),
  but absorption is real (93 entities have FN)
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
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from scipy import stats

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase2_benign_pathological"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
PHASE2_DIR = RESULTS_DIR / "phase2"
for d in [PILOT_DIR, PHASE2_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# SAE config
SAE_RELEASE = "gemma-scope-2b-pt-res-canonical"
SAE_ID = "layer_24/width_16k/canonical"
SAE_KEY = "L24_16k"
SAE_LAYER = 24

# Token position -- must match probe training
TOKEN_POS = -2

# ICL configuration
N_ICL = 5

# Number of input contexts per entity
N_CONTEXTS = 30 if MODE == "PILOT" else 100

# Max entities to process per hierarchy in pilot
MAX_ENTITIES_PILOT = 50
MAX_ENTITIES_FULL = 200

# Benign/pathological thresholds
THRESHOLDS = [0.05, 0.1, 0.2]

# Number of control random directions per entity
N_CONTROL_DIRECTIONS = 5

# Bootstrap
N_BOOTSTRAP = 2000 if MODE == "PILOT" else 10000

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
# Load pre-trained probe
# ============================================================
def load_probe(hierarchy_name, layer=24):
    """Load saved sklearn LogisticRegression probe."""
    probe_path = PHASE1_DIR / f"probe_{hierarchy_name}_L{layer}.npz"
    if not probe_path.exists():
        raise FileNotFoundError(f"Probe not found: {probe_path}")

    data = np.load(probe_path, allow_pickle=True)
    coef = data["coef"]
    intercept = data["intercept"]
    classes = data["classes"]

    probe = LogisticRegression(max_iter=1)
    probe.classes_ = np.arange(len(classes))
    probe.coef_ = coef
    probe.intercept_ = intercept

    logger.info(f"  Loaded probe {hierarchy_name}_L{layer}: "
                f"coef={coef.shape}, classes={list(classes)}")
    return probe, classes


# ============================================================
# RAVEL data loading
# ============================================================
def prepare_ravel_data():
    """Load RAVEL dataset for city-continent."""
    from datasets import load_dataset

    logger.info("Loading RAVEL dataset (hij/ravel, city_entity)...")
    ds = load_dataset("hij/ravel", "city_entity", split="train")

    cities = list(ds["City"])
    continents = list(ds["Continent"])

    valid = [
        (c, l) for c, l in zip(cities, continents)
        if c and l and str(l).strip() and str(c).strip()
    ]
    cities_valid = [x[0] for x in valid]
    labels_valid = ["Oceania" if x[1] == "Australia" else x[1] for x in valid]

    label_counts = Counter(labels_valid)
    valid_labels = {l for l, n in label_counts.items() if n >= 3}
    keep = [(c, l) for c, l in zip(cities_valid, labels_valid) if l in valid_labels]

    return {
        "cities": [x[0] for x in keep],
        "labels": [x[1] for x in keep],
    }


# ============================================================
# ICL prompt construction
# ============================================================
def build_icl_prompts(city, label, all_cities, all_labels, n_contexts, n_icl=5):
    """Build varied ICL prompts for one city."""
    base_template = "The city of {entity} is on the continent of"
    answer_template = " {label}"

    examples = [(c, l) for c, l in zip(all_cities, all_labels) if c != city]

    prompts = []
    for ctx_idx in range(n_contexts):
        rng_ctx = random.Random(SEED + hash(city) + ctx_idx * 7919)
        rng_ctx.shuffle(examples)
        icl_selected = examples[:n_icl]

        icl_parts = []
        for ex_city, ex_label in icl_selected:
            ex_text = base_template.format(entity=ex_city) + answer_template.format(label=ex_label)
            icl_parts.append(ex_text)

        full_prompt = "\n".join(icl_parts) + "\n" + base_template.format(entity=city)
        prompts.append(full_prompt)

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
# Load FN entities from Phase 1 / Phase 2 results
# ============================================================
def load_fn_entities_crossdomain():
    """Load FN entity list from phase1/phase2 results for city-continent."""
    # Try phase2 results first (has detailed per-entity data including patching)
    phase2_path = PHASE2_DIR / "activation_patching_crossdomain.json"
    pilot_path = PILOT_DIR / "phase1_absorption_crossdomain.json"

    fn_entities = {}  # city -> {label, n_fn, source}

    # From phase2 activation patching results (most detailed)
    if phase2_path.exists():
        logger.info(f"Loading FN entities from {phase2_path.name}")
        data = json.loads(phase2_path.read_text())
        for entity in data.get("per_entity_results", []):
            if entity.get("status") == "completed" and entity.get("n_false_negatives", 0) > 0:
                city = entity["city"]
                fn_entities[city] = {
                    "label": entity["true_label"],
                    "n_fn": entity["n_false_negatives"],
                    "source": "phase2_patching",
                }

    # Also check pilot absorption results
    if pilot_path.exists():
        logger.info(f"Loading additional FN from {pilot_path.name}")
        data = json.loads(pilot_path.read_text())
        fn_examples = data.get("absorption_results", {}).get("L24_16k", {}).get("fn_examples", [])
        for fn in fn_examples:
            city = fn.get("city", "")
            label = fn.get("true_label", "")
            if city and label and city not in fn_entities:
                fn_entities[city] = {
                    "label": label,
                    "n_fn": fn.get("n_fn", 1),
                    "source": "phase1_pilot",
                }

    logger.info(f"  Total FN entities (city-continent): {len(fn_entities)}")
    return fn_entities


def load_fn_entities_firstletter():
    """Load FN entities from first-letter absorption results."""
    fl_path = PILOT_DIR / "phase1_absorption_firstletter.json"
    fn_entities = {}

    if not fl_path.exists():
        logger.warning(f"First-letter results not found: {fl_path}")
        return fn_entities

    data = json.loads(fl_path.read_text())
    l24_data = data.get("absorption_results", {}).get("L24_16k", {})

    if "per_letter" in l24_data:
        for letter, letter_data in l24_data["per_letter"].items():
            if letter_data.get("false_negatives", 0) > 0:
                fn_entities[letter] = {
                    "label": letter,
                    "n_fn": letter_data["false_negatives"],
                    "source": "phase1_firstletter",
                }

    # Also check per_word_results if available
    if "per_word_results" in l24_data:
        for word_result in l24_data["per_word_results"]:
            if word_result.get("false_negative", False):
                word = word_result.get("word", "")
                letter = word_result.get("true_letter", "")
                if word and letter:
                    fn_entities[word] = {
                        "label": letter,
                        "n_fn": 1,
                        "word": word,
                        "source": "phase1_firstletter_word",
                    }

    logger.info(f"  Total FN entities (first-letter): {len(fn_entities)}")
    return fn_entities


# ============================================================
# Core: Benign vs Pathological diagnostic for one entity
# ============================================================
def benign_pathological_for_entity(
    model, sae, probe, class_labels,
    entity_name, true_label,
    all_entities, all_labels,
    hierarchy_type,  # "city-continent" or "first-letter"
    n_contexts=30,
    thresholds=None,
    n_control_dirs=5,
    layer=24,
    token_pos=-2,
    device="cuda:0",
):
    """
    Benign vs Pathological classification for one entity.

    For each FN context:
    1. Get SAE activations, identify child features (probe-aligned absorbers).
    2. Compute parent probe direction d_parent.
    3. For each child feature i:
       - Original decoder: d_i
       - Project out parent component: d_i_mod = d_i - (d_i . d_parent / ||d_parent||^2) * d_parent
       - Modified reconstruction: x_mod = SAE_decode(acts, with d_i replaced by d_i_mod)
    4. Feed x_mod back through remaining model layers, get logit change for parent-relevant tokens.
    5. Classify based on magnitude of logit change.

    Simplified approach (feasible in pilot):
    - Instead of modifying the full forward pass through remaining layers,
      we measure the change in the probe's output logits (linear model) when the
      parent component is ablated from child decoder vectors.
    - This directly measures how much the absorbed parent information in child features
      contributes to the probe's prediction.
    - Additionally, we measure the change in the model's next-token logits for
      parent-relevant tokens (e.g., continent names) after ablating parent direction
      from the SAE reconstruction.
    """
    if thresholds is None:
        thresholds = THRESHOLDS

    tl_hook = f"blocks.{layer}.hook_resid_post"
    cls_list = class_labels.tolist() if hasattr(class_labels, 'tolist') else list(class_labels)

    if true_label not in cls_list:
        return {"status": "label_not_in_probe", "entity": entity_name}

    probe_true_idx = cls_list.index(true_label)

    # Probe direction for the true class (parent direction)
    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)
    d_parent = probe_coefs[probe_true_idx].clone()
    d_parent_norm_sq = (d_parent @ d_parent).item()
    if d_parent_norm_sq < 1e-12:
        return {"status": "degenerate_probe_direction", "entity": entity_name}

    d_parent_normalized = d_parent / (d_parent.norm() + 1e-8)

    # W_dec for accessing decoder vectors
    W_dec = sae.W_dec.detach().float().cpu()  # [d_sae, d_model]

    # Build prompts
    prompts = build_icl_prompts(
        entity_name, true_label, all_entities, all_labels,
        n_contexts=n_contexts, n_icl=N_ICL,
    )

    # Parent-relevant token IDs for model logit measurement
    # For city-continent: continent names
    # For first-letter: letter tokens
    tokenizer = model.tokenizer
    parent_token_ids = []
    for cls_name in cls_list:
        toks = tokenizer.encode(f" {cls_name}", add_special_tokens=False)
        if toks:
            parent_token_ids.append(toks[0])
        else:
            parent_token_ids.append(-1)

    # Track results per context
    logit_changes = []           # float: signed logit change for true class
    abs_logit_changes = []       # float: absolute logit change
    probe_logit_changes = []     # float: probe logit change
    control_logit_changes = []   # float: logit change from random direction ablation
    context_details = []         # per-context detail dicts

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
            # Get original logits at this position
            original_logits = logits_full[0, token_pos, :].detach().float().cpu()
            del cache, logits_full

            # Probe on raw activation
            raw_np = raw_act.cpu().numpy().reshape(1, -1)
            raw_pred = probe.predict(raw_np)[0]
            probe_correct_raw = (raw_pred == probe_true_idx)

            if not probe_correct_raw:
                continue

            n_probe_correct_raw += 1

            # SAE encode/decode
            raw_act_device = raw_act.to(device)
            with torch.no_grad():
                sae_features = sae.encode(raw_act_device.unsqueeze(0))
                sae_out = sae.decode(sae_features)

            sae_np = sae_out[0].detach().float().cpu().numpy().reshape(1, -1)
            sae_pred = probe.predict(sae_np)[0]
            probe_correct_sae = (sae_pred == probe_true_idx)

            if probe_correct_sae:
                continue  # Not a false negative

            # ---- FALSE NEGATIVE (absorption instance) ----
            n_fn += 1
            sae_features_cpu = sae_features[0].detach().float().cpu()

            # Identify child features: active features aligned with parent probe direction
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

            # ===== ABLATION: Remove parent direction component from child decoder =====
            # For each child feature, compute modified decoder:
            #   d_i_mod = d_i - (d_i . d_parent / ||d_parent||^2) * d_parent
            # Then recompute SAE output with modified decoders

            # Original SAE output (as reference)
            sae_out_cpu = sae_out[0].detach().float().cpu()

            # Compute the contribution of parent direction through child features
            parent_contribution = torch.zeros(W_dec.shape[1])
            for fid in child_feature_indices:
                act_val = sae_features_cpu[fid].item()
                d_child = W_dec[fid]
                # Project child decoder onto parent direction
                proj_coef = (d_child @ d_parent) / d_parent_norm_sq
                parent_component = proj_coef * d_parent
                parent_contribution += act_val * parent_component

            # Modified reconstruction: subtract parent direction contribution from child features
            modified_reconstruction = sae_out_cpu - parent_contribution

            # ===== Measure 1: Probe logit change =====
            mod_np = modified_reconstruction.numpy().reshape(1, -1)
            # Get probe logits (decision function)
            orig_probe_logits = probe.decision_function(sae_np)[0]
            mod_probe_logits = probe.decision_function(mod_np)[0]

            # Logit for true class
            orig_probe_logit_true = orig_probe_logits[probe_true_idx]
            mod_probe_logit_true = mod_probe_logits[probe_true_idx]
            probe_logit_change = float(mod_probe_logit_true - orig_probe_logit_true)

            # Check if probe now predicts correctly after ablation
            mod_pred = probe.predict(mod_np)[0]
            probe_recovers = (mod_pred == probe_true_idx)

            # ===== Measure 2: Model logit change for parent-relevant tokens =====
            # Feed modified reconstruction through remaining layers
            # We approximate by measuring the change in residual stream and projecting
            # onto the unembedding matrix
            model_logit_change = float('nan')
            try:
                # Use the model's unembedding to measure logit impact
                # W_U shape: [d_model, d_vocab]
                W_U = model.W_U.detach().float().cpu()  # [d_model, d_vocab]

                # Change in residual stream at this position
                delta_resid = modified_reconstruction - sae_out_cpu

                # Project onto vocabulary space
                delta_logits = delta_resid @ W_U  # [d_vocab]

                # Get logit change for true class token
                true_token_id = parent_token_ids[probe_true_idx]
                if true_token_id >= 0 and true_token_id < delta_logits.shape[0]:
                    model_logit_change = float(delta_logits[true_token_id].item())
                else:
                    # Use max across parent tokens
                    valid_parent_ids = [pid for pid in parent_token_ids if 0 <= pid < delta_logits.shape[0]]
                    if valid_parent_ids:
                        parent_delta_logits = delta_logits[valid_parent_ids]
                        model_logit_change = float(parent_delta_logits[probe_true_idx % len(valid_parent_ids)].item())
            except Exception as e:
                logger.debug(f"Model logit measurement failed: {e}")

            # ===== Measure 3: Control -- ablate random direction of same norm =====
            parent_dir_norm = d_parent.norm().item()
            control_changes = []
            rng_ctrl = np.random.RandomState(SEED + ctx_idx)

            for ctrl_i in range(n_control_dirs):
                # Generate random direction of same norm as parent direction
                random_dir = torch.randn_like(d_parent)
                random_dir = random_dir / (random_dir.norm() + 1e-8) * parent_dir_norm

                # Compute parent-like contribution through child features using random direction
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

            # ===== Record results =====
            logit_changes.append(probe_logit_change)
            abs_logit_changes.append(abs(probe_logit_change))
            probe_logit_changes.append(probe_logit_change)
            control_logit_changes.append(mean_control_change)

            if len(context_details) < 20:
                context_details.append({
                    "ctx_idx": ctx_idx,
                    "entity": entity_name,
                    "true_label": true_label,
                    "probe_logit_change": probe_logit_change,
                    "model_logit_change": model_logit_change,
                    "control_logit_change": mean_control_change,
                    "probe_recovers_after_ablation": bool(probe_recovers),
                    "n_child_features": len(child_feature_indices),
                    "top_child_cos_sims": child_cos_sims[:3].tolist(),
                    "parent_contribution_norm": float(parent_contribution.norm().item()),
                })

            del sae_out, sae_features
            torch.cuda.empty_cache()

        except Exception as e:
            if ctx_idx < 3:
                logger.warning(f"  Context {ctx_idx} error: {e}")
            continue

    if n_fn == 0 or n_processed == 0:
        return {
            "status": "no_fn_found",
            "entity": entity_name,
            "true_label": true_label,
            "n_probe_correct_raw": n_probe_correct_raw,
            "n_fn": n_fn,
            "n_processed": n_processed,
        }

    # ===== Classify benign vs pathological at each threshold =====
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

    # Control classification at default threshold (0.1)
    ctrl_abs_changes = np.array([abs(c) for c in control_logit_changes])
    n_ctrl_benign = int(np.sum(ctrl_abs_changes <= 0.1))
    control_benign_pct = float(n_ctrl_benign / len(ctrl_abs_changes)) if len(ctrl_abs_changes) > 0 else 0.0

    return {
        "status": "completed",
        "entity": entity_name,
        "true_label": true_label,
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
            "benign_pct_at_01": control_benign_pct,
        },
        "logit_changes": logit_changes[:100],  # Save raw values (up to 100)
        "control_changes": control_logit_changes[:100],
        "context_details": context_details,
    }


# ============================================================
# GPU progress tracking
# ============================================================
def update_gpu_progress(elapsed_seconds, status="completed"):
    import filelock
    progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    lock_path = WORKSPACE / "exp" / "gpu_progress.lock"
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
                "planned_min": 45,
                "actual_min": round(elapsed_seconds / 60, 1),
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "config_snapshot": {
                    "model": "gemma-2-2b",
                    "mode": MODE,
                    "sae": SAE_KEY,
                    "layer": SAE_LAYER,
                    "token_pos": TOKEN_POS,
                    "n_contexts": N_CONTEXTS,
                    "thresholds": THRESHOLDS,
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
    start_time = time.time()
    write_pid()
    report_progress(0, 10, "starting")

    logger.info("=" * 70)
    logger.info("Phase 2.2: Benign vs. Pathological Absorption Diagnostic (H8)")
    logger.info(f"Mode: {MODE}, Layer: {SAE_LAYER}, Token pos: {TOKEN_POS}")
    logger.info(f"Thresholds: {THRESHOLDS}")
    logger.info(f"N_CONTEXTS per entity: {N_CONTEXTS}")
    logger.info("=" * 70)

    device = "cuda:0"
    max_entities = MAX_ENTITIES_PILOT if MODE == "PILOT" else MAX_ENTITIES_FULL

    # Step 1: Load model
    report_progress(1, 10, "loading_model")
    model = load_model(device=device)

    # Step 2: Load SAE
    report_progress(2, 10, "loading_sae")
    sae = load_sae(SAE_RELEASE, SAE_ID, device=device)

    # Step 3: Load probes
    report_progress(3, 10, "loading_probes")
    probe_cc, class_labels_cc = load_probe("city-continent", layer=SAE_LAYER)
    cls_list_cc = class_labels_cc.tolist() if hasattr(class_labels_cc, 'tolist') else list(class_labels_cc)
    logger.info(f"  City-continent probe classes: {cls_list_cc}")

    # Step 4: Load RAVEL data
    report_progress(4, 10, "loading_data")
    ravel_data = prepare_ravel_data()
    all_cities = ravel_data["cities"]
    all_labels = ravel_data["labels"]

    # Step 5: Load FN entities
    report_progress(5, 10, "loading_fn_entities")
    fn_entities_cc = load_fn_entities_crossdomain()

    # Limit for pilot
    fn_list = sorted(fn_entities_cc.items(), key=lambda x: x[1].get("n_fn", 0), reverse=True)
    fn_list = fn_list[:max_entities]
    logger.info(f"  Processing {len(fn_list)} city-continent FN entities (max={max_entities})")

    # Step 6: Run benign/pathological diagnostic for city-continent
    report_progress(6, 10, "benign_pathological_city_continent",
                    {"n_entities": len(fn_list)})

    all_entity_results = []
    all_logit_changes = []
    all_control_changes = []
    all_abs_changes = []

    for i, (entity_name, entity_info) in enumerate(fn_list):
        true_label = entity_info["label"]
        logger.info(f"\n  [{i+1}/{len(fn_list)}] Processing '{entity_name}' (label: {true_label}, "
                     f"expected FN: {entity_info.get('n_fn', '?')})...")

        report_progress(6, 10, f"entity_{i+1}/{len(fn_list)}",
                        {"entity": entity_name, "label": true_label})

        result = benign_pathological_for_entity(
            model=model,
            sae=sae,
            probe=probe_cc,
            class_labels=class_labels_cc,
            entity_name=entity_name,
            true_label=true_label,
            all_entities=all_cities,
            all_labels=all_labels,
            hierarchy_type="city-continent",
            n_contexts=N_CONTEXTS,
            thresholds=THRESHOLDS,
            n_control_dirs=N_CONTROL_DIRECTIONS,
            layer=SAE_LAYER,
            token_pos=TOKEN_POS,
            device=device,
        )

        all_entity_results.append(result)

        if result["status"] == "completed":
            logger.info(f"    FN: {result['n_fn']}, processed: {result['n_processed']}")
            logger.info(f"    Mean |logit_change|: {result['logit_change_stats']['abs_mean']:.4f}")
            logger.info(f"    Control |logit_change|: {result['control_logit_change_stats']['abs_mean']:.4f}")
            for thr_key, cls_data in result["classifications"].items():
                logger.info(f"    {thr_key}: benign={cls_data['benign_pct']:.1%}, "
                            f"pathological={cls_data['pathological_pct']:.1%}")

            all_logit_changes.extend(result.get("logit_changes", []))
            all_control_changes.extend(result.get("control_changes", []))
            all_abs_changes.extend([abs(x) for x in result.get("logit_changes", [])])
        else:
            logger.info(f"    Status: {result['status']}")

        if (i + 1) % 10 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    # Step 7: Aggregate results
    report_progress(7, 10, "aggregating")
    logger.info(f"\n{'='*70}")
    logger.info("Aggregate Results -- City-Continent")
    logger.info(f"{'='*70}")

    completed = [r for r in all_entity_results if r["status"] == "completed"]
    n_completed = len(completed)
    total_fn = sum(r["n_fn"] for r in completed)
    total_processed = sum(r["n_processed"] for r in completed)

    logger.info(f"  Completed: {n_completed}/{len(fn_list)}")
    logger.info(f"  Total FN: {total_fn}")
    logger.info(f"  Total processed: {total_processed}")

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

    # Control aggregate at threshold 0.1
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

    # Step 8: Statistical tests
    report_progress(8, 10, "statistical_tests")

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

    # Test 2: Are parent-direction ablation changes different from random direction?
    if len(all_logit_changes) >= 3 and len(all_control_changes) >= 3:
        # Compare absolute changes
        parent_abs = np.array([abs(c) for c in all_logit_changes])
        control_abs = np.array([abs(c) for c in all_control_changes])

        # Truncate to same length for paired test
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

        # Mann-Whitney U test (unpaired, allows different lengths)
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

    # Step 9: Per-class breakdown
    report_progress(9, 10, "per_class_analysis")

    per_class_results = defaultdict(lambda: {
        "n_entities": 0, "n_fn": 0, "n_processed": 0,
        "logit_changes": [], "control_changes": [],
        "classifications": {f"threshold_{t}": {"n_benign": 0, "n_pathological": 0}
                           for t in THRESHOLDS}
    })

    for r in completed:
        label = r["true_label"]
        pcd = per_class_results[label]
        pcd["n_entities"] += 1
        pcd["n_fn"] += r["n_fn"]
        pcd["n_processed"] += r["n_processed"]
        pcd["logit_changes"].extend(r.get("logit_changes", []))
        pcd["control_changes"].extend(r.get("control_changes", []))
        for thr in THRESHOLDS:
            thr_key = f"threshold_{thr}"
            if thr_key in r.get("classifications", {}):
                pcd["classifications"][thr_key]["n_benign"] += r["classifications"][thr_key]["n_benign"]
                pcd["classifications"][thr_key]["n_pathological"] += r["classifications"][thr_key]["n_pathological"]

    per_class_summary = {}
    for label, pcd in per_class_results.items():
        total_at_01 = (pcd["classifications"]["threshold_0.1"]["n_benign"] +
                       pcd["classifications"]["threshold_0.1"]["n_pathological"])
        per_class_summary[label] = {
            "n_entities": pcd["n_entities"],
            "n_fn": pcd["n_fn"],
            "n_processed": pcd["n_processed"],
            "mean_abs_logit_change": float(np.mean([abs(c) for c in pcd["logit_changes"]])) if pcd["logit_changes"] else 0.0,
            "benign_pct_01": float(pcd["classifications"]["threshold_0.1"]["n_benign"] / max(total_at_01, 1)),
            "pathological_pct_01": float(pcd["classifications"]["threshold_0.1"]["n_pathological"] / max(total_at_01, 1)),
        }
        logger.info(f"\n  {label}: entities={pcd['n_entities']}, processed={pcd['n_processed']}, "
                    f"benign@0.1={per_class_summary[label]['benign_pct_01']:.1%}")

    # Step 10: H8 hypothesis verdict
    report_progress(10, 10, "compiling_output")

    # H8: >= 30% benign at threshold 0.1
    benign_pct_01 = aggregate_classifications.get("threshold_0.1", {}).get("benign_pct", 0)
    h8_supported = benign_pct_01 >= 0.30
    h8_falsified = benign_pct_01 < 0.10

    if h8_supported:
        h8_verdict = "SUPPORTED"
        h8_detail = (f"{benign_pct_01:.1%} of absorption instances are benign "
                     f"(|logit_change| <= 0.1), exceeding 30% threshold.")
    elif h8_falsified:
        h8_verdict = "FALSIFIED"
        h8_detail = (f"Only {benign_pct_01:.1%} of absorption instances are benign "
                     f"(|logit_change| <= 0.1), below 10% falsification criterion.")
    else:
        h8_verdict = "INCONCLUSIVE"
        h8_detail = (f"{benign_pct_01:.1%} of absorption instances are benign "
                     f"(|logit_change| <= 0.1), between 10-30% -- neither clearly supported nor falsified.")

    # Control check: is parent direction ablation different from random?
    parent_mean_abs = logit_dist.get("abs_mean", 0) if logit_dist.get("n", 0) > 0 else 0
    control_mean_abs = control_dist.get("abs_mean", 0) if control_dist.get("n", 0) > 0 else 0
    direction_specificity = parent_mean_abs > control_mean_abs * 1.5  # Parent ablation has 50% more impact

    elapsed = time.time() - start_time

    # Pilot pass criteria
    pilot_pass = (
        n_completed >= 10 and  # Enough entities
        total_processed >= 20 and  # Enough FN instances processed
        logit_dist.get("n", 0) >= 20 and  # Enough logit change measurements
        len(THRESHOLDS) >= 3  # Classification at 3 thresholds
    )

    # Compile output
    output = {
        "task_id": TASK_ID,
        "mode": MODE,
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "sae": {"release": SAE_RELEASE, "sae_id": SAE_ID, "key": SAE_KEY},
        "layer": SAE_LAYER,
        "token_pos": TOKEN_POS,
        "n_contexts_per_entity": N_CONTEXTS,
        "thresholds": THRESHOLDS,
        "n_control_directions": N_CONTROL_DIRECTIONS,
        "hierarchy": "city-continent",
        "entity_counts": {
            "n_fn_entities_input": len(fn_list),
            "n_completed": n_completed,
            "n_with_fn": sum(1 for r in completed if r.get("n_fn", 0) > 0),
            "total_fn_instances": total_fn,
            "total_processed": total_processed,
        },
        "aggregate_classifications": aggregate_classifications,
        "logit_change_distribution": logit_dist,
        "control_distribution": control_dist,
        "statistical_tests": stat_tests,
        "per_class_summary": per_class_summary,
        "h8_hypothesis": {
            "verdict": h8_verdict,
            "detail": h8_detail,
            "benign_pct_at_01": benign_pct_01,
            "target_benign_pct": 0.30,
            "falsification_threshold": 0.10,
            "direction_specificity": direction_specificity,
            "parent_mean_abs_change": parent_mean_abs,
            "control_mean_abs_change": control_mean_abs,
        },
        "pilot_criteria": {
            "met": pilot_pass,
            "details": {
                "enough_entities_10": n_completed >= 10,
                "enough_processed_20": total_processed >= 20,
                "enough_measurements_20": logit_dist.get("n", 0) >= 20,
                "three_thresholds": len(THRESHOLDS) >= 3,
                "control_computed": control_dist.get("n", 0) > 0,
            },
        },
        "per_entity_results": [
            {k: v for k, v in r.items()
             if k not in ("logit_changes", "control_changes", "context_details")}
            for r in all_entity_results
        ],
        "sample_context_details": [
            d for r in completed[:10]
            for d in r.get("context_details", [])[:3]
        ],
        "logit_change_histogram": {
            "bins": list(np.arange(-2.0, 2.05, 0.1)),
            "counts": list(np.histogram(all_logit_changes,
                                         bins=np.arange(-2.0, 2.05, 0.1))[0].tolist())
            if all_logit_changes else [],
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Save
    out_path = PILOT_DIR / f"{TASK_ID}.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"\nSaved: {out_path}")

    phase2_path = PHASE2_DIR / "benign_pathological.json"
    phase2_path.write_text(json.dumps(output, indent=2, default=str))
    logger.info(f"Also saved: {phase2_path}")

    # Generate summary markdown
    summary_md = generate_summary_md(output)
    md_path = PILOT_DIR / f"{TASK_ID}_summary.md"
    md_path.write_text(summary_md)
    logger.info(f"Summary: {md_path}")

    del model, sae
    gc.collect()
    torch.cuda.empty_cache()

    summary_text = (
        f"Phase 2.2 benign/pathological diagnostic ({MODE}). "
        f"Entities: {n_completed}, FN processed: {total_processed}. "
        f"Benign%: {benign_pct_01:.1%} at threshold 0.1. "
        f"H8 verdict: {h8_verdict}. "
        f"Direction specificity: {'YES' if direction_specificity else 'NO'} "
        f"(parent={parent_mean_abs:.4f}, control={control_mean_abs:.4f}). "
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
        "# Phase 2.2: Benign vs. Pathological Absorption Diagnostic (H8)",
        "",
        f"**H8 Verdict**: {results.get('h8_hypothesis', {}).get('verdict', 'UNKNOWN')}",
        f"**Pilot**: {'PASS' if results.get('pilot_criteria', {}).get('met') else 'FAIL'}",
        f"**Time**: {results.get('elapsed_minutes', 0):.1f} minutes",
        "",
        "## Design",
        "",
        f"- **Hierarchy**: city-continent at L{results.get('layer', '?')}",
        f"- **SAE**: {results.get('sae', {}).get('key', '?')}",
        f"- **Thresholds**: {results.get('thresholds', [])}",
        f"- **Contexts per entity**: {results.get('n_contexts_per_entity', '?')}",
        f"- **Control directions**: {results.get('n_control_directions', '?')}",
        "",
        "## H8 Hypothesis (Benign Absorption)",
        "",
        f"{results.get('h8_hypothesis', {}).get('detail', '')}",
        "",
        f"- **Direction specificity**: {'YES' if results.get('h8_hypothesis', {}).get('direction_specificity') else 'NO'}",
        f"- **Parent ablation mean |change|**: {results.get('h8_hypothesis', {}).get('parent_mean_abs_change', 0):.4f}",
        f"- **Random direction mean |change|**: {results.get('h8_hypothesis', {}).get('control_mean_abs_change', 0):.4f}",
        "",
        "## Classification at Multiple Thresholds",
        "",
        "| Threshold | Benign | Pathological | Benign % |",
        "|-----------|--------|--------------|----------|",
    ]

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

    # Per-class breakdown
    pcs = results.get("per_class_summary", {})
    if pcs:
        lines.extend([
            "",
            "## Per-Class Results",
            "",
            "| Class | Entities | Processed | Mean |delta| | Benign% @0.1 |",
            "|-------|----------|-----------|------------|--------------|",
        ])
        for cls in sorted(pcs.keys()):
            d = pcs[cls]
            lines.append(
                f"| {cls} | {d.get('n_entities', 0)} | {d.get('n_processed', 0)} | "
                f"{d.get('mean_abs_logit_change', 0):.4f} | "
                f"{d.get('benign_pct_01', 0):.1%} |"
            )

    # Entity counts
    ec = results.get("entity_counts", {})
    lines.extend([
        "",
        "## Summary",
        "",
        f"- Entities completed: {ec.get('n_completed', 0)}/{ec.get('n_fn_entities_input', 0)}",
        f"- Total FN instances: {ec.get('total_fn_instances', 0)}",
        f"- Total processed: {ec.get('total_processed', 0)}",
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
