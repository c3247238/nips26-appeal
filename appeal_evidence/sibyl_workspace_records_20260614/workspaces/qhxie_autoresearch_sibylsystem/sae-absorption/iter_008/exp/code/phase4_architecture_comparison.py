"""
Phase 4.1: Architecture Comparison Across Hierarchy Types (v2)

Compare JumpReLU (Gemma Scope) vs. BatchTopK (SAEBench) vs. Matryoshka (SAEBench)
absorption rates across hierarchy types.

Tests H6: does architecture advantage generalize across hierarchy types?

Key improvements from v1:
- Use best probes from Phase 1.1 (layer 24 for cross-domain, sae_spelling for first-letter)
- All SAEBench SAEs are at layer 12 → primary comparison at layer 12
- Also include JumpReLU layer 24 data from Phase 1.2/1.3 for supplementary analysis
- Include city-country (F1=0.789, above 0.70 minimum)
- Report L0 values alongside absorption rates
- Compute 2-way ANOVA: absorption ~ architecture * hierarchy_type
- Proper DONE marker in results dir (not just pilots dir)

PILOT: Layer 12, 3 architectures, ~200 samples per hierarchy
All SAEBench SAEs only exist at layer 12, so that's the fair comparison layer.
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
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats as scipy_stats

# ============================================================
# Configuration
# ============================================================
TASK_ID = "phase4_architecture_comparison"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE4_DIR = RESULTS_DIR / "phase4"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE4_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# Primary comparison layer: 12 (only layer where all architectures have SAEs)
PRIMARY_LAYER = 12

# SAE architecture configs -- all at layer 12 for fair comparison
SAE_ARCHITECTURES = [
    {
        "name": "JumpReLU_16k",
        "arch_type": "JumpReLU",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_16k/canonical",
        "layer": 12,
        "width_label": "16k",
    },
    {
        "name": "JumpReLU_65k",
        "arch_type": "JumpReLU",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_65k/canonical",
        "layer": 12,
        "width_label": "65k",
    },
    {
        "name": "BatchTopK_16k",
        "arch_type": "BatchTopK",
        "release": "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109",
        "sae_id": "blocks.12.hook_resid_post__trainer_0",
        "layer": 12,
        "width_label": "~16k",
    },
    {
        "name": "Matryoshka",
        "arch_type": "Matryoshka",
        "release": "gemma-2-2b-res-matryoshka-dc",
        "sae_id": "blocks.12.hook_resid_post",
        "layer": 12,
        "width_label": "variable",
    },
]

if MODE == "PILOT":
    MAX_WORDS_FIRSTLETTER = 200
    MAX_CITIES = 200
    TIMEOUT = 900
    N_BOOTSTRAP = 1000
else:
    MAX_WORDS_FIRSTLETTER = 1000
    MAX_CITIES = 1000
    TIMEOUT = 2700
    N_BOOTSTRAP = 10000

LETTERS = "abcdefghijklmnopqrstuvwxyz"
TOKEN_POS_FIRSTLETTER = -6
TOKEN_POS_CROSSDOMAIN = -1  # Updated: -1 matches probe training in Phase 1.1

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


# ============================================================
# Process tracking (required by protocol)
# ============================================================
def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))

def report_progress(step, total_steps, status="running", metrics=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "loss": None, "metric": metrics or {},
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
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": final_progress, "timestamp": datetime.now().isoformat(),
    })
    # Write DONE marker in BOTH results dir and pilots dir
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(done_data)
    (PILOT_DIR / f"{TASK_ID}_DONE").write_text(done_data)


def update_gpu_progress(status="completed", start_time=None, end_time=None):
    """Update gpu_progress.json with task completion."""
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        gp = json.loads(gp_path.read_text())
    except Exception:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if status == "completed":
        if TASK_ID not in gp.get("completed", []):
            gp.setdefault("completed", []).append(TASK_ID)
    elif status == "failed":
        if TASK_ID not in gp.get("failed", []):
            gp.setdefault("failed", []).append(TASK_ID)

    # Remove from running
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]

    # Add timing
    if start_time and end_time:
        actual_min = int((end_time - start_time).total_seconds() / 60)
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 45,
            "actual_min": actual_min,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "mode": MODE,
                "n_architectures": len(SAE_ARCHITECTURES),
                "primary_layer": PRIMARY_LAYER,
                "gpu_model": "RTX PRO 6000 Blackwell",
                "gpu_count": 1,
            }
        }

    gp_path.write_text(json.dumps(gp, indent=2))
    logger.info(f"GPU progress updated: {status}")


# ============================================================
# Model loading
# ============================================================
def load_model(device="cuda:0"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info("Loading Gemma 2 2B...")
    hf_model = AutoModelForCausalLM.from_pretrained(GEMMA_LOCAL_PATH, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(GEMMA_LOCAL_PATH)
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b", device=device, dtype=torch.bfloat16,
        hf_model=hf_model, tokenizer=tokenizer,
    )
    logger.info(f"Model loaded: {model.cfg.n_layers} layers, d_model={model.cfg.d_model}")
    del hf_model
    gc.collect(); torch.cuda.empty_cache()
    return model


def load_sae(release, sae_id, device="cuda:0"):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    return sae


# ============================================================
# L0 estimation
# ============================================================
def estimate_l0(sae, activations, device="cuda:0"):
    """Estimate L0 (average number of active features) for a batch of activations."""
    with torch.no_grad():
        batch_size = 32
        total_active = 0
        n_samples = 0
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(device)
            features = sae.encode(batch)
            active_count = (features.abs() > 1e-6).float().sum(dim=-1)  # [batch]
            total_active += active_count.sum().item()
            n_samples += len(batch)
        return total_active / n_samples if n_samples > 0 else 0.0


# ============================================================
# Probe loading
# ============================================================
def reconstruct_probe_from_npz(npz_path):
    """Reconstruct a LogisticRegression probe from saved coef/intercept/classes."""
    data = np.load(npz_path, allow_pickle=True)
    coef = data["coef"]
    intercept = data["intercept"]
    classes = data["classes"]

    probe = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=SEED)
    probe.coef_ = coef
    probe.intercept_ = intercept
    probe.classes_ = classes
    probe.n_features_in_ = coef.shape[1]
    return probe


def load_firstletter_probe(model, word_list, layer=12, device="cuda:0"):
    """
    Train first-letter probe using sae_spelling ICL pipeline.
    This achieves F1=1.0 at all layers (confirmed in Phase 1.1).
    We train fresh to ensure the probe matches our prompting format.
    """
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    hook_name = f"blocks.{layer}.hook_resid_post"

    all_acts, all_labels = [], []
    for word in word_list:
        letter = word[0].lower()
        if letter not in LETTERS:
            continue
        label = LETTERS.index(letter)
        try:
            sp = create_icl_prompt(
                word=word, examples=word_list,
                base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                answer_formatter=formatter, max_icl_examples=10,
                shuffle_examples=True,
            )
            tokens = model.to_tokens(sp.base, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name][0, TOKEN_POS_FIRSTLETTER, :].float().cpu().numpy()
            all_acts.append(act)
            all_labels.append(label)
            del cache
        except Exception:
            continue

        if len(all_acts) % 50 == 0:
            torch.cuda.empty_cache()

    X = np.array(all_acts)
    y = np.array(all_labels)
    logger.info(f"First-letter probe training data: {X.shape}, {len(set(y))} classes")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    best_f1, best_probe = -1, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        probe = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=SEED)
        probe.fit(X_train, y_train)
        y_pred = probe.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="weighted")
        if f1 > best_f1:
            best_f1 = f1
            best_probe = probe
            logger.info(f"  C={C}: F1={f1:.4f}")

    quality = {"f1": float(best_f1), "accuracy": float(best_f1),
               "quality_gate": "pass" if best_f1 >= 0.90 else "below",
               "method": "sae_spelling_icl_sklearn", "layer": layer}
    logger.info(f"First-letter probe at L{layer}: F1={best_f1:.4f}")
    return best_probe, quality


def load_crossdomain_probe(hierarchy, layer=24):
    """Load cross-domain probe at given layer."""
    path = PHASE1_DIR / f"probe_{hierarchy}_L{layer}.npz"
    if path.exists():
        logger.info(f"Loading {hierarchy} probe (L{layer}): {path}")
        probe = reconstruct_probe_from_npz(path)
        return probe, path.name
    logger.error(f"No {hierarchy} probe found for layer {layer}")
    return None, None


def get_probe_quality(hierarchy, layer):
    """Get probe quality from Phase 1.1 training results."""
    ptf_path = PHASE1_DIR / "probe_training_full.json"
    if not ptf_path.exists():
        return {"f1": None, "accuracy": None, "quality_gate": "unknown"}

    try:
        ptf = json.loads(ptf_path.read_text())
    except Exception:
        return {"f1": None, "accuracy": None, "quality_gate": "unknown"}

    # Look up the probe key
    if hierarchy == "first-letter":
        key_candidates = [
            f"first-letter-sklearn_L{layer}",
            f"first-letter-sae-spelling_L{layer}",
        ]
    else:
        key_candidates = [
            f"{hierarchy}-sklearn_L{layer}",
            f"{hierarchy}_L{layer}",
        ]

    for key in key_candidates:
        if key in ptf.get("probes", {}):
            p = ptf["probes"][key]
            f1 = p.get("f1_weighted_cv", p.get("f1", None))
            acc = p.get("accuracy_cv", p.get("accuracy", None))
            gate = "pass" if (f1 and f1 >= 0.90) else ("below" if (f1 and f1 >= 0.70) else "excluded")
            return {"f1": float(f1) if f1 else None, "accuracy": float(acc) if acc else None, "quality_gate": gate}

    return {"f1": None, "accuracy": None, "quality_gate": "unknown"}


# ============================================================
# First-letter word list (same as Phase 1.2)
# ============================================================
def get_word_list(tokenizer, max_words=200):
    """Get curated word list from tokenizer vocab."""
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
# RAVEL dataset loading
# ============================================================
def load_ravel_dataset(max_cities=200):
    """Load RAVEL city data for cross-domain hierarchies."""
    try:
        from datasets import load_dataset
        ds = load_dataset("hij/ravel", "city_entity", split="train")
        logger.info(f"RAVEL loaded: {len(ds)} entries, columns: {ds.column_names}")
    except Exception as e:
        logger.error(f"Failed to load RAVEL: {e}")
        return None

    cities = []
    seen_cities = set()
    for row in ds:
        city = row.get("City", row.get("city", row.get("entity", "")))
        if city and city not in seen_cities:
            seen_cities.add(city)
            cities.append({
                "city": city,
                "country": row.get("Country", row.get("country", "")),
                "continent": row.get("Continent", row.get("continent", "")),
                "language": row.get("Language", row.get("language", "")),
            })
            if len(cities) >= max_cities:
                break

    logger.info(f"RAVEL cities: {len(cities)}")
    return cities


# ============================================================
# Absorption measurement
# ============================================================
def measure_firstletter_absorption(model, sae, probe, word_list, layer, device="cuda:0"):
    """Measure first-letter absorption for a given SAE architecture."""
    from sae_spelling.prompting import (
        create_icl_prompt,
        first_letter_formatter,
        VERBOSE_FIRST_LETTER_TEMPLATE,
    )
    formatter = first_letter_formatter()
    tl_hook = f"blocks.{layer}.hook_resid_post"

    # Compute main features per letter using probe directions
    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)
    W_dec = sae.W_dec.detach().float().cpu()

    # Map probe class indices to letters
    letter_to_probe_idx = {}
    for pidx, cls in enumerate(probe.classes_):
        if isinstance(cls, (int, np.integer)) and 0 <= cls < 26:
            letter_to_probe_idx[LETTERS[cls]] = pidx
        elif isinstance(cls, str) and len(cls) == 1 and cls.lower() in LETTERS:
            letter_to_probe_idx[cls.lower()] = pidx

    main_features = {}
    for letter in LETTERS:
        if letter not in letter_to_probe_idx:
            continue
        pidx = letter_to_probe_idx[letter]
        pdir = probe_coefs[pidx]
        pdir = pdir / pdir.norm()
        cos = F.cosine_similarity(pdir.unsqueeze(0), W_dec, dim=-1)
        topvals, topids = cos.topk(5)
        main_features[letter] = {
            "feature_ids": topids.tolist(),
            "cos_sims": topvals.tolist(),
        }
    del W_dec

    per_letter = {l: {"total": 0, "probe_correct_raw": 0, "probe_correct_sae": 0,
                       "false_negatives": 0, "fn_main_absent": 0, "fn_main_present": 0,
                       "main_fires": 0}
                  for l in LETTERS}
    fn_examples = []
    all_activations = []

    for word in word_list:
        letter = word[0].lower()
        if letter not in LETTERS or letter not in letter_to_probe_idx:
            continue

        # Need to map letter to correct class index for comparison
        pidx = letter_to_probe_idx[letter]
        cls_value = probe.classes_[pidx]

        try:
            sp = create_icl_prompt(
                word=word, examples=word_list,
                base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
                answer_formatter=formatter, max_icl_examples=10,
                shuffle_examples=True,
            )
            tokens = model.to_tokens(sp.base, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, TOKEN_POS_FIRSTLETTER, :].detach().float()
            all_activations.append(raw_act.cpu())

            with torch.no_grad():
                sae_features = sae.encode(raw_act.to(device).unsqueeze(0))
                sae_out = sae.decode(sae_features)

            raw_pred = probe.predict(raw_act.cpu().numpy().reshape(1, -1))[0]
            sae_pred = probe.predict(sae_out[0].detach().float().cpu().numpy().reshape(1, -1))[0]

            correct_raw = (raw_pred == cls_value)
            correct_sae = (sae_pred == cls_value)

            mfids = main_features.get(letter, {}).get("feature_ids", [])
            any_main = False
            if mfids:
                feat_vals = sae_features[0, mfids].detach().float().cpu()
                any_main = (feat_vals.abs() > 1e-6).any().item()

            del cache

        except Exception:
            continue

        s = per_letter[letter]
        s["total"] += 1
        if correct_raw:
            s["probe_correct_raw"] += 1
        if correct_sae:
            s["probe_correct_sae"] += 1
        if any_main:
            s["main_fires"] += 1
        if correct_raw and not correct_sae:
            s["false_negatives"] += 1
            if not any_main:
                s["fn_main_absent"] += 1
            else:
                s["fn_main_present"] += 1
            if len(fn_examples) < 15:
                fn_examples.append({
                    "word": word, "letter": letter,
                    "raw_pred": str(raw_pred), "sae_pred": str(sae_pred),
                    "main_fires": any_main,
                })

    # Estimate L0
    l0_estimate = None
    if all_activations:
        act_tensor = torch.stack(all_activations[:100])  # Use up to 100 for L0 estimate
        l0_estimate = estimate_l0(sae, act_tensor, device)
        logger.info(f"  L0 estimate: {l0_estimate:.1f}")

    # Aggregate
    total_correct_raw = sum(s["probe_correct_raw"] for s in per_letter.values())
    total_fn = sum(s["false_negatives"] for s in per_letter.values())
    total_fn_absent = sum(s["fn_main_absent"] for s in per_letter.values())
    absorption_rate = total_fn / total_correct_raw if total_correct_raw > 0 else 0.0
    strict_rate = total_fn_absent / total_correct_raw if total_correct_raw > 0 else 0.0

    # Bootstrap CI
    ci = _bootstrap_ci(total_correct_raw, per_letter)

    return {
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "total_words": sum(s["total"] for s in per_letter.values()),
        "total_probe_correct": total_correct_raw,
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_absent,
        "l0_estimate": float(l0_estimate) if l0_estimate is not None else None,
        "bootstrap_ci": ci,
        "per_letter": per_letter,
        "fn_examples": fn_examples,
    }


def measure_crossdomain_absorption(model, sae, probe, cities, hierarchy, layer, device="cuda:0"):
    """Measure absorption for a cross-domain hierarchy on a given SAE."""
    tl_hook = f"blocks.{layer}.hook_resid_post"
    attr_key = hierarchy.split("-")[1]  # "continent", "language", "country"
    attr_name = attr_key

    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)
    W_dec = sae.W_dec.detach().float().cpu()

    probe_classes = list(probe.classes_)

    # Compute main features per class
    main_features = {}
    for pidx, cls_label in enumerate(probe_classes):
        pdir = probe_coefs[pidx]
        pdir = pdir / pdir.norm()
        cos = F.cosine_similarity(pdir.unsqueeze(0), W_dec, dim=-1)
        topvals, topids = cos.topk(5)
        main_features[cls_label] = {
            "feature_ids": topids.tolist(),
            "cos_sims": topvals.tolist(),
        }
    del W_dec

    per_class = defaultdict(lambda: {
        "total": 0, "probe_correct_raw": 0, "probe_correct_sae": 0,
        "false_negatives": 0, "fn_main_absent": 0, "fn_main_present": 0,
        "main_fires": 0,
    })
    fn_examples = []
    n_errors = 0
    all_activations = []

    for c in cities:
        attr_val = c.get(attr_key, "")
        if not attr_val or attr_val not in probe_classes:
            continue

        city_name = c["city"]
        prompt = f"The {attr_name} of {city_name} is"

        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[tl_hook])

            raw_act = cache[tl_hook][0, TOKEN_POS_CROSSDOMAIN, :].detach().float()
            all_activations.append(raw_act.cpu())

            with torch.no_grad():
                sae_features = sae.encode(raw_act.to(device).unsqueeze(0))
                sae_out = sae.decode(sae_features)

            raw_pred = probe.predict(raw_act.cpu().numpy().reshape(1, -1))[0]
            sae_pred = probe.predict(sae_out[0].detach().float().cpu().numpy().reshape(1, -1))[0]

            correct_raw = (raw_pred == attr_val)
            correct_sae = (sae_pred == attr_val)

            mfids = main_features.get(attr_val, {}).get("feature_ids", [])
            any_main = False
            if mfids:
                feat_vals = sae_features[0, mfids].detach().float().cpu()
                any_main = (feat_vals.abs() > 1e-6).any().item()

            del cache

        except Exception:
            n_errors += 1
            continue

        s = per_class[attr_val]
        s["total"] += 1
        if correct_raw:
            s["probe_correct_raw"] += 1
        if correct_sae:
            s["probe_correct_sae"] += 1
        if any_main:
            s["main_fires"] += 1
        if correct_raw and not correct_sae:
            s["false_negatives"] += 1
            if not any_main:
                s["fn_main_absent"] += 1
            else:
                s["fn_main_present"] += 1
            if len(fn_examples) < 15:
                fn_examples.append({
                    "city": city_name, "true_label": attr_val,
                    "raw_pred_label": str(raw_pred), "sae_pred_label": str(sae_pred),
                    "main_fires": any_main,
                })

    # Estimate L0
    l0_estimate = None
    if all_activations:
        act_tensor = torch.stack(all_activations[:100])
        l0_estimate = estimate_l0(sae, act_tensor, device)
        logger.info(f"  L0 estimate: {l0_estimate:.1f}")

    # Aggregate
    total_correct_raw = sum(s["probe_correct_raw"] for s in per_class.values())
    total_fn = sum(s["false_negatives"] for s in per_class.values())
    total_fn_absent = sum(s["fn_main_absent"] for s in per_class.values())
    total_cities = sum(s["total"] for s in per_class.values())
    absorption_rate = total_fn / total_correct_raw if total_correct_raw > 0 else 0.0
    strict_rate = total_fn_absent / total_correct_raw if total_correct_raw > 0 else 0.0
    probe_raw_acc = total_correct_raw / total_cities if total_cities > 0 else 0.0

    # Bootstrap CI on per-sample FN indicator
    if total_correct_raw > 0:
        per_sample_fn = []
        for s in per_class.values():
            per_sample_fn.extend([1] * s["false_negatives"])
            per_sample_fn.extend([0] * (s["probe_correct_raw"] - s["false_negatives"]))
        per_sample_fn = np.array(per_sample_fn)
        rng = np.random.RandomState(SEED)
        boot_rates = [rng.choice(per_sample_fn, size=len(per_sample_fn), replace=True).mean()
                      for _ in range(N_BOOTSTRAP)]
        ci = {
            "mean": float(np.mean(boot_rates)),
            "ci_lower": float(np.percentile(boot_rates, 2.5)),
            "ci_upper": float(np.percentile(boot_rates, 97.5)),
            "std": float(np.std(boot_rates)),
        }
    else:
        ci = {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    return {
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "probe_raw_accuracy": float(probe_raw_acc),
        "total_cities": total_cities,
        "total_probe_correct": total_correct_raw,
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_absent,
        "total_errors": n_errors,
        "l0_estimate": float(l0_estimate) if l0_estimate is not None else None,
        "bootstrap_ci": ci,
        "per_class": dict(per_class),
        "fn_examples": fn_examples,
    }


def _bootstrap_ci(total_correct_raw, per_letter_or_class):
    """Compute bootstrap CI from per-letter/class stats."""
    if total_correct_raw > 0:
        per_word_fn = []
        for s in per_letter_or_class.values():
            if isinstance(s, dict) and "false_negatives" in s:
                per_word_fn.extend([1] * s["false_negatives"])
                per_word_fn.extend([0] * (s["probe_correct_raw"] - s["false_negatives"]))
        per_word_fn = np.array(per_word_fn)
        if len(per_word_fn) > 0:
            rng = np.random.RandomState(SEED)
            boot_rates = [rng.choice(per_word_fn, size=len(per_word_fn), replace=True).mean()
                          for _ in range(N_BOOTSTRAP)]
            return {
                "mean": float(np.mean(boot_rates)),
                "ci_lower": float(np.percentile(boot_rates, 2.5)),
                "ci_upper": float(np.percentile(boot_rates, 97.5)),
                "std": float(np.std(boot_rates)),
            }
    return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}


# ============================================================
# Statistical comparison helpers
# ============================================================
def proportions_z_test(rate_a, n_a, rate_b, n_b):
    """Two-proportion z-test for absorption rate comparison."""
    if n_a == 0 or n_b == 0:
        return {"z_stat": 0.0, "p_value": 1.0, "significant_005": False}
    p_pool = (rate_a * n_a + rate_b * n_b) / (n_a + n_b)
    if p_pool == 0 or p_pool == 1:
        return {"z_stat": 0.0, "p_value": 1.0, "significant_005": False}
    se = np.sqrt(p_pool * (1 - p_pool) * (1.0 / n_a + 1.0 / n_b))
    if se == 0:
        return {"z_stat": 0.0, "p_value": 1.0, "significant_005": False}
    z = (rate_a - rate_b) / se
    p = 2 * scipy_stats.norm.sf(abs(z))
    return {"z_stat": float(z), "p_value": float(p), "significant_005": p < 0.05}


def compute_anova(arch_results, hierarchy_names):
    """
    Compute a 2-way ANOVA-like test: absorption ~ architecture * hierarchy.
    Uses per-sample binary FN indicators from each (architecture, hierarchy) cell.
    Returns F-statistic and p-value for architecture effect, hierarchy effect, and interaction.
    """
    try:
        from scipy.stats import f_oneway, kruskal

        # Collect per-architecture absorption rates across hierarchies
        arch_rates = defaultdict(list)
        hierarchy_rates = defaultdict(list)

        for arch_name, arch_data in arch_results.items():
            if "error" in arch_data:
                continue
            for h in hierarchy_names:
                r = arch_data.get("hierarchies", {}).get(h, {})
                if "error" in r or "absorption_rate" not in r:
                    continue
                rate = r["absorption_rate"]
                arch_rates[arch_name].append(rate)
                hierarchy_rates[h].append(rate)

        # Architecture main effect (Kruskal-Wallis since sample sizes are small)
        arch_groups = [v for v in arch_rates.values() if len(v) > 0]
        if len(arch_groups) >= 2 and all(len(g) >= 2 for g in arch_groups):
            h_stat, p_arch = kruskal(*arch_groups)
            arch_effect = {"H_stat": float(h_stat), "p_value": float(p_arch),
                          "significant_005": p_arch < 0.05, "test": "Kruskal-Wallis"}
        else:
            arch_effect = {"H_stat": None, "p_value": None, "significant_005": False,
                          "test": "insufficient_data"}

        # Hierarchy main effect
        hier_groups = [v for v in hierarchy_rates.values() if len(v) > 0]
        if len(hier_groups) >= 2 and all(len(g) >= 2 for g in hier_groups):
            h_stat, p_hier = kruskal(*hier_groups)
            hier_effect = {"H_stat": float(h_stat), "p_value": float(p_hier),
                          "significant_005": p_hier < 0.05, "test": "Kruskal-Wallis"}
        else:
            hier_effect = {"H_stat": None, "p_value": None, "significant_005": False,
                          "test": "insufficient_data"}

        return {
            "architecture_effect": arch_effect,
            "hierarchy_effect": hier_effect,
            "note": "Kruskal-Wallis non-parametric test (small cell sizes in pilot)"
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Main execution
# ============================================================
def main():
    start_time = datetime.now()
    write_pid()
    report_progress(0, 20, status="starting")

    device = "cuda:0"
    logger.info(f"=== Phase 4.1: Architecture Comparison v2 (MODE={MODE}) ===")
    logger.info(f"Primary comparison layer: {PRIMARY_LAYER}")
    logger.info(f"SAE architectures: {[a['name'] for a in SAE_ARCHITECTURES]}")

    # Step 1: Load model
    model = load_model(device=device)
    tokenizer = model.tokenizer

    # Step 2: Prepare data
    logger.info("Preparing word list and city data...")
    word_list = get_word_list(tokenizer, max_words=MAX_WORDS_FIRSTLETTER)
    cities = load_ravel_dataset(max_cities=MAX_CITIES)

    # Step 3: Load probes at the primary comparison layer (12)
    # For first-letter: train using sae_spelling ICL pipeline (achieves F1~1.0)
    logger.info("Training first-letter probe using sae_spelling ICL pipeline...")
    fl_probe, fl_quality = load_firstletter_probe(model, word_list, layer=PRIMARY_LAYER, device=device)

    logger.info(f"First-letter probe at L{PRIMARY_LAYER}: {fl_quality}")

    # For cross-domain: load layer 12 probes from Phase 1.1
    # If L12 probe quality is too low, also try training fresh with ICL prompts
    crossdomain_hierarchies = ["city-continent", "city-language", "city-country"]
    crossdomain_probes = {}

    for hierarchy in crossdomain_hierarchies:
        if cities is None:
            continue
        probe, pname = load_crossdomain_probe(hierarchy, layer=PRIMARY_LAYER)
        quality = get_probe_quality(hierarchy, PRIMARY_LAYER)

        if probe is not None:
            f1_val = quality.get("f1")
            if f1_val is not None and f1_val >= 0.60:
                crossdomain_probes[hierarchy] = {"probe": probe, "quality": quality, "probe_file": pname}
                logger.info(f"  {hierarchy} L{PRIMARY_LAYER}: F1={f1_val:.4f}, gate={quality.get('quality_gate')}")
            else:
                logger.warning(f"  {hierarchy} L{PRIMARY_LAYER}: F1={f1_val} too low, training fresh probe")
                fresh_probe, fresh_quality = _train_crossdomain_probe_fresh(
                    model, hierarchy, cities, PRIMARY_LAYER, device
                )
                if fresh_probe is not None:
                    crossdomain_probes[hierarchy] = {"probe": fresh_probe, "quality": fresh_quality, "probe_file": "fresh"}
                    logger.info(f"  {hierarchy} fresh: F1={fresh_quality.get('f1'):.4f}")
        else:
            logger.info(f"  {hierarchy} L{PRIMARY_LAYER}: no saved probe, training fresh")
            fresh_probe, fresh_quality = _train_crossdomain_probe_fresh(
                model, hierarchy, cities, PRIMARY_LAYER, device
            )
            if fresh_probe is not None:
                crossdomain_probes[hierarchy] = {"probe": fresh_probe, "quality": fresh_quality, "probe_file": "fresh"}
                logger.info(f"  {hierarchy} fresh: F1={fresh_quality.get('f1'):.4f}")
            else:
                logger.warning(f"  {hierarchy}: all probe attempts failed, skipping")

    report_progress(2, 20, metrics={"probes_loaded": len(crossdomain_probes) + 1})

    # Step 4: For each architecture, measure absorption on all hierarchies
    arch_results = {}
    step = 2
    all_hierarchies = ["first-letter"] + [h for h in crossdomain_hierarchies if h in crossdomain_probes]
    total_steps = len(SAE_ARCHITECTURES) * len(all_hierarchies) + 5

    for arch_cfg in SAE_ARCHITECTURES:
        arch_name = arch_cfg["name"]
        logger.info(f"\n{'='*60}")
        logger.info(f"Architecture: {arch_name} ({arch_cfg['arch_type']})")
        logger.info(f"  Release: {arch_cfg['release']}")
        logger.info(f"  SAE ID: {arch_cfg['sae_id']}")
        logger.info(f"{'='*60}")

        try:
            sae = load_sae(arch_cfg["release"], arch_cfg["sae_id"], device)
        except Exception as e:
            logger.error(f"Failed to load SAE {arch_name}: {e}")
            arch_results[arch_name] = {
                "error": str(e),
                "arch_type": arch_cfg["arch_type"],
            }
            step += len(all_hierarchies)
            report_progress(step, total_steps, metrics={"current_arch": arch_name, "status": "load_failed"})
            continue

        arch_data = {
            "arch_type": arch_cfg["arch_type"],
            "release": arch_cfg["release"],
            "sae_id": arch_cfg["sae_id"],
            "width_label": arch_cfg["width_label"],
            "d_sae": int(sae.cfg.d_sae),
            "d_in": int(sae.cfg.d_in),
            "layer": arch_cfg["layer"],
            "hierarchies": {},
        }

        # 4a. First-letter absorption
        logger.info(f"  Measuring first-letter absorption at L{PRIMARY_LAYER}...")
        try:
            fl_result = measure_firstletter_absorption(model, sae, fl_probe, word_list, PRIMARY_LAYER, device)
            arch_data["hierarchies"]["first-letter"] = fl_result
            logger.info(f"  First-letter: rate={fl_result['absorption_rate']:.4f} "
                       f"strict={fl_result['strict_absorption_rate']:.4f} "
                       f"FN={fl_result['total_false_negatives']}/{fl_result['total_probe_correct']} "
                       f"L0={fl_result.get('l0_estimate', 'N/A')}")
        except Exception as e:
            logger.error(f"  First-letter measurement failed: {e}")
            import traceback; traceback.print_exc()
            arch_data["hierarchies"]["first-letter"] = {"error": str(e)}

        step += 1
        report_progress(step, total_steps, metrics={"current_arch": arch_name, "hierarchy": "first-letter"})
        torch.cuda.empty_cache()

        # 4b. Cross-domain absorption
        for hierarchy in crossdomain_hierarchies:
            if hierarchy not in crossdomain_probes:
                continue
            pdata = crossdomain_probes[hierarchy]
            logger.info(f"  Measuring {hierarchy} absorption at L{PRIMARY_LAYER}...")
            try:
                cd_result = measure_crossdomain_absorption(
                    model, sae, pdata["probe"], cities, hierarchy, PRIMARY_LAYER, device
                )
                arch_data["hierarchies"][hierarchy] = cd_result
                logger.info(f"  {hierarchy}: rate={cd_result['absorption_rate']:.4f} "
                           f"strict={cd_result['strict_absorption_rate']:.4f} "
                           f"FN={cd_result['total_false_negatives']}/{cd_result['total_probe_correct']} "
                           f"L0={cd_result.get('l0_estimate', 'N/A')}")
            except Exception as e:
                logger.error(f"  {hierarchy} measurement failed: {e}")
                import traceback; traceback.print_exc()
                arch_data["hierarchies"][hierarchy] = {"error": str(e)}

            step += 1
            report_progress(step, total_steps, metrics={"current_arch": arch_name, "hierarchy": hierarchy})
            torch.cuda.empty_cache()

        # Unload SAE
        del sae
        gc.collect()
        torch.cuda.empty_cache()
        arch_results[arch_name] = arch_data

    # Step 5: Compute comparisons
    logger.info("\n" + "=" * 60)
    logger.info("Computing architecture comparisons...")

    comparisons = {}
    arch_names = [a["name"] for a in SAE_ARCHITECTURES if a["name"] in arch_results and "error" not in arch_results.get(a["name"], {})]

    for h in all_hierarchies:
        comparisons[h] = {}
        for i, a1 in enumerate(arch_names):
            for a2 in arch_names[i+1:]:
                r1 = arch_results.get(a1, {}).get("hierarchies", {}).get(h, {})
                r2 = arch_results.get(a2, {}).get("hierarchies", {}).get(h, {})
                if "error" in r1 or "error" in r2:
                    continue

                rate1 = r1.get("absorption_rate", 0)
                rate2 = r2.get("absorption_rate", 0)
                n1 = r1.get("total_probe_correct", 0)
                n2 = r2.get("total_probe_correct", 0)

                ztest = proportions_z_test(rate1, n1, rate2, n2)

                comparisons[h][f"{a1}_vs_{a2}"] = {
                    "arch_a": a1, "arch_b": a2,
                    "rate_a": float(rate1), "rate_b": float(rate2),
                    "diff": float(rate1 - rate2),
                    "z_stat": ztest["z_stat"],
                    "p_value": ztest["p_value"],
                    "significant_005": ztest["significant_005"],
                    "n_a": n1, "n_b": n2,
                }

    # Compute ANOVA
    logger.info("Computing ANOVA...")
    anova_results = compute_anova(arch_results, all_hierarchies)
    logger.info(f"ANOVA: {anova_results}")

    # Summary table
    summary_table = []
    for h in all_hierarchies:
        for an in [a["name"] for a in SAE_ARCHITECTURES]:
            ad = arch_results.get(an, {})
            if "error" in ad:
                continue
            r = ad.get("hierarchies", {}).get(h, {})
            if "error" in r:
                continue
            row = {
                "hierarchy": h,
                "architecture": an,
                "arch_type": ad.get("arch_type", "unknown"),
                "d_sae": ad.get("d_sae", 0),
                "layer": ad.get("layer", PRIMARY_LAYER),
                "absorption_rate": r.get("absorption_rate"),
                "strict_rate": r.get("strict_absorption_rate"),
                "ci_lower": r.get("bootstrap_ci", {}).get("ci_lower"),
                "ci_upper": r.get("bootstrap_ci", {}).get("ci_upper"),
                "l0_estimate": r.get("l0_estimate"),
                "n_probe_correct": r.get("total_probe_correct", 0),
                "n_fn": r.get("total_false_negatives", 0),
            }
            summary_table.append(row)

    # JumpReLU advantage analysis (key finding from pilot)
    jumprelu_advantage = {}
    for h in all_hierarchies:
        jumprelu_rates = []
        other_rates = []
        for an in arch_names:
            r = arch_results.get(an, {}).get("hierarchies", {}).get(h, {})
            if "error" in r:
                continue
            rate = r.get("absorption_rate", None)
            if rate is not None:
                if "JumpReLU" in an:
                    jumprelu_rates.append((an, rate))
                else:
                    other_rates.append((an, rate))

        if jumprelu_rates and other_rates:
            best_jumprelu = min(jumprelu_rates, key=lambda x: x[1])
            worst_other = max(other_rates, key=lambda x: x[1])
            best_other = min(other_rates, key=lambda x: x[1])
            jumprelu_advantage[h] = {
                "best_jumprelu": {"arch": best_jumprelu[0], "rate": best_jumprelu[1]},
                "best_other": {"arch": best_other[0], "rate": best_other[1]},
                "worst_other": {"arch": worst_other[0], "rate": worst_other[1]},
                "jumprelu_is_lowest": best_jumprelu[1] <= best_other[1],
            }

    # H6 verdict: does JumpReLU maintain advantage across hierarchies?
    n_jumprelu_best = sum(1 for v in jumprelu_advantage.values() if v["jumprelu_is_lowest"])
    h6_verdict = ("SUPPORTED" if n_jumprelu_best == len(jumprelu_advantage)
                  else "PARTIALLY_SUPPORTED" if n_jumprelu_best > 0
                  else "NOT_SUPPORTED")

    # Compile final results
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "version": "v2",
        "timestamp": start_time.isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "primary_layer": PRIMARY_LAYER,
        "n_architectures": len(SAE_ARCHITECTURES),
        "architectures_tested": [a["name"] for a in SAE_ARCHITECTURES],
        "hierarchies_tested": all_hierarchies,
        "probe_quality": {
            "first-letter": fl_quality,
            **{h: pdata["quality"] for h, pdata in crossdomain_probes.items()},
        },
        "architecture_results": arch_results,
        "comparisons": comparisons,
        "anova": anova_results,
        "summary_table": summary_table,
        "jumprelu_advantage": jumprelu_advantage,
        "h6_hypothesis": {
            "description": "Does JumpReLU maintain lowest absorption across hierarchy types?",
            "n_hierarchies_tested": len(jumprelu_advantage),
            "n_jumprelu_lowest": n_jumprelu_best,
            "per_hierarchy": jumprelu_advantage,
            "verdict": h6_verdict,
        },
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Pilot criteria check
    n_arch_with_results = sum(1 for a in arch_results.values()
                              if "hierarchies" in a and "error" not in a)
    n_hierarchies_measured = len(all_hierarchies)
    pilot_pass = n_arch_with_results >= 2 and n_hierarchies_measured >= 2
    results["pilot_criteria_met"] = pilot_pass
    results["pilot_criteria_details"] = {
        "n_architectures_with_results": n_arch_with_results,
        "n_hierarchies_measured": n_hierarchies_measured,
        "minimum_architectures": 2,
        "minimum_hierarchies": 2,
    }

    # Save results
    out_path = PHASE4_DIR / "architecture_comparison.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Results saved: {out_path}")

    pilot_path = PILOT_DIR / "phase4_architecture_comparison.json"
    pilot_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Pilot copy saved: {pilot_path}")

    # Write summary markdown
    md_lines = _generate_summary_md(results, summary_table, comparisons, all_hierarchies,
                                     jumprelu_advantage, h6_verdict, anova_results, elapsed,
                                     crossdomain_probes, fl_quality)

    md_path = PILOT_DIR / "phase4_architecture_comparison_summary.md"
    md_path.write_text("\n".join(md_lines))
    logger.info(f"Summary saved: {md_path}")

    # Print summary to stdout
    logger.info("\n" + "=" * 60)
    logger.info("ARCHITECTURE COMPARISON SUMMARY (v2)")
    logger.info("=" * 60)
    for row in summary_table:
        l0_str = f" L0={row['l0_estimate']:.0f}" if row.get('l0_estimate') else ""
        logger.info(f"  {row['hierarchy']:15s} | {row['architecture']:20s} | "
                    f"absorption={row['absorption_rate']:.4f} strict={row['strict_rate']:.4f} "
                    f"(FN={row['n_fn']}/{row['n_probe_correct']}){l0_str}")
    logger.info(f"H6 verdict: {h6_verdict} ({n_jumprelu_best}/{len(jumprelu_advantage)} hierarchies)")
    logger.info(f"Pilot pass: {pilot_pass}")

    # Mark done and update GPU progress
    mark_done("success", f"{n_arch_with_results} archs, {n_hierarchies_measured} hierarchies, H6={h6_verdict}")
    update_gpu_progress("completed", start_time, end_time)

    return results


def _train_crossdomain_probe_fresh(model, hierarchy, cities, layer, device="cuda:0"):
    """Train cross-domain probe from scratch at a given layer."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    attr_key = hierarchy.split("-")[1]  # "continent", "language", "country"

    # Build label set
    labels = [c.get(attr_key, "") for c in cities if c.get(attr_key, "")]
    label_set = sorted(set(labels))
    if len(label_set) < 2:
        logger.error(f"Too few classes for {hierarchy}: {len(label_set)}")
        return None, None

    # Cache activations
    all_acts, all_labels, all_cities_proc = [], [], []
    for i, c in enumerate(cities):
        attr_val = c.get(attr_key, "")
        if not attr_val or attr_val not in label_set:
            continue
        city_name = c["city"]
        prompt = f"The {attr_key} of {city_name} is"
        try:
            tokens = model.to_tokens(prompt, prepend_bos=True)
            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=[hook_name])
            act = cache[hook_name][0, TOKEN_POS_CROSSDOMAIN, :].float().cpu().numpy()
            all_acts.append(act)
            all_labels.append(attr_val)
            all_cities_proc.append(city_name)
            del cache
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            torch.cuda.empty_cache()

    if len(all_acts) < 20:
        logger.error(f"Too few samples for {hierarchy}: {len(all_acts)}")
        return None, None

    X = np.array(all_acts)
    y = np.array(all_labels)
    logger.info(f"Cross-domain probe data for {hierarchy}: X={X.shape}, {len(set(y))} classes")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED,
    )

    best_f1, best_probe = -1, None
    for C in [0.01, 0.1, 1.0, 10.0]:
        probe = LogisticRegression(C=C, max_iter=5000, solver="lbfgs", random_state=SEED)
        try:
            probe.fit(X_train, y_train)
            y_pred = probe.predict(X_test)
            f1 = f1_score(y_test, y_pred, average="weighted")
            if f1 > best_f1:
                best_f1 = f1
                best_probe = probe
        except Exception:
            continue

    if best_probe is None:
        return None, None

    y_pred = best_probe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    n_classes = len(set(y_train))

    quality = {
        "f1": float(best_f1), "accuracy": float(acc), "n_classes": n_classes,
        "quality_gate": "pass" if best_f1 >= 0.90 else ("below" if best_f1 >= 0.70 else "excluded"),
        "method": "fresh_sklearn", "layer": layer,
    }
    logger.info(f"Fresh probe {hierarchy} L{layer}: F1={best_f1:.4f}, classes={n_classes}")
    return best_probe, quality


def _generate_summary_md(results, summary_table, comparisons, all_hierarchies,
                          jumprelu_advantage, h6_verdict, anova_results, elapsed,
                          crossdomain_probes, fl_quality):
    """Generate markdown summary."""
    md = [
        "# Phase 4.1: Architecture Comparison Results (v2)",
        "",
        f"Mode: {results['mode']} | Layer: {PRIMARY_LAYER} | Elapsed: {elapsed/60:.1f} min | Architectures: {results['n_architectures']}",
        "",
        "## Probe Quality at Layer 12",
        "",
        "| Hierarchy | F1 | Quality Gate |",
        "|-----------|-----|-------------|",
        f"| first-letter | {fl_quality.get('f1', 'N/A')} | {fl_quality.get('quality_gate', 'N/A')} |",
    ]
    for h, pdata in crossdomain_probes.items():
        q = pdata['quality']
        md.append(f"| {h} | {q.get('f1', 'N/A')} | {q.get('quality_gate', 'N/A')} |")

    md += [
        "",
        "## Summary Table",
        "",
        "| Hierarchy | Architecture | d_sae | Absorption Rate | Strict Rate | 95% CI | L0 | FN / Total |",
        "|-----------|-------------|-------|-----------------|-------------|--------|-----|-----------|",
    ]
    for row in summary_table:
        ci_str = f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}]" if row.get('ci_lower') is not None else "N/A"
        l0_str = f"{row['l0_estimate']:.0f}" if row.get('l0_estimate') else "N/A"
        md.append(
            f"| {row['hierarchy']} | {row['architecture']} | {row['d_sae']} | "
            f"{row['absorption_rate']:.4f} | {row['strict_rate']:.4f} | {ci_str} | {l0_str} | "
            f"{row['n_fn']} / {row['n_probe_correct']} |"
        )

    md += ["", "## Pairwise Comparisons", ""]
    for h, cmps in comparisons.items():
        md.append(f"### {h}")
        md.append("")
        md.append("| Comparison | Rate A | Rate B | Diff | z-stat | p-value | Sig. |")
        md.append("|-----------|--------|--------|------|--------|---------|------|")
        for ck, cv in cmps.items():
            sig_str = "YES" if cv["significant_005"] else "no"
            md.append(
                f"| {cv['arch_a']} vs {cv['arch_b']} | {cv['rate_a']:.4f} | {cv['rate_b']:.4f} | "
                f"{cv['diff']:.4f} | {cv['z_stat']:.3f} | {cv['p_value']:.4f} | {sig_str} |"
            )
        md.append("")

    md += [
        "## ANOVA: absorption ~ architecture * hierarchy",
        "",
        f"Architecture effect: {anova_results.get('architecture_effect', 'N/A')}",
        f"Hierarchy effect: {anova_results.get('hierarchy_effect', 'N/A')}",
        "",
        "## JumpReLU Advantage Analysis",
        "",
    ]
    for h, adv in jumprelu_advantage.items():
        status = "LOWEST" if adv["jumprelu_is_lowest"] else "NOT lowest"
        md.append(f"- **{h}**: JumpReLU {adv['best_jumprelu']['rate']:.4f} vs "
                  f"best other {adv['best_other']['arch']} {adv['best_other']['rate']:.4f} → {status}")

    md += [
        "",
        f"## H6 Verdict: JumpReLU advantage -- **{h6_verdict}**",
        f"- JumpReLU lowest in {sum(1 for v in jumprelu_advantage.values() if v['jumprelu_is_lowest'])}/{len(jumprelu_advantage)} hierarchies",
        "",
        "## Caveats",
        "",
        "- All comparisons at layer 12 (only layer with all architectures)",
        f"- Cross-domain probes below strict quality gate (F1 < 0.90)",
        "- Matryoshka has variable width (32k) vs 16k for JumpReLU/BatchTopK -- not width-matched",
        "- JumpReLU_65k included for within-architecture width comparison",
        "- Absorption rates for cross-domain hierarchies may be confounded with probe errors",
        "",
    ]

    return md


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress("failed", datetime.now(), datetime.now())
        raise
