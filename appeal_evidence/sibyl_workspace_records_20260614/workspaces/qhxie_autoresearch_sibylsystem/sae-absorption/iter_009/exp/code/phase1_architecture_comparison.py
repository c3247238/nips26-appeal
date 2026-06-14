"""
Phase 1.4: Architecture Comparison Across Hierarchies (Iter 9 Pilot)

Compare JumpReLU (Gemma Scope) vs BatchTopK (SAEBench) vs Matryoshka (SAEBench)
absorption rates across hierarchy types.

PRIMARY: Layer 12 (only layer where all 3 architectures have SAEs)
SUPPLEMENTARY: Layer 24 (JumpReLU + Matryoshka only; BatchTopK unavailable at L24)

Tests H6: does architecture advantage generalize across hierarchy types?
Iter_008 found: architecture ANOVA p=0.87 (non-significant), hierarchy p=0.005.

PILOT: ~200 samples per hierarchy, 4 architectures at L12 + 3 at L24.
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
TASK_ID = "phase1_architecture_comparison"
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
PHASE1_DIR = RESULTS_DIR / "phase1"
for d in [PILOT_DIR, PHASE1_DIR]:
    d.mkdir(parents=True, exist_ok=True)

GEMMA_LOCAL_PATH = "/home/qhxie/.cache/huggingface/hub/models--unsloth--gemma-2-2b/snapshots/25319945f7fd83b8b903e12081777b7eef2ba993"

MODE = os.environ.get("PILOT_MODE", "PILOT").upper()
if "--full" in sys.argv:
    MODE = "FULL"

# GPU assignment: when CUDA_VISIBLE_DEVICES is set, use cuda:0 (the masked device)
# Otherwise use cuda:6 as specified
if os.environ.get("CUDA_VISIBLE_DEVICES"):
    DEVICE = "cuda:0"
else:
    DEVICE = "cuda:6"

# Layer 12 = primary comparison (all architectures available)
# Layer 24 = supplementary (JumpReLU + Matryoshka only)
PRIMARY_LAYER = 12
SECONDARY_LAYER = 24

# SAE architecture configs at Layer 12
SAE_ARCHITECTURES_L12 = [
    {
        "name": "JumpReLU_16k_L12",
        "arch_type": "JumpReLU",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_16k/canonical",
        "layer": 12,
        "width_label": "16k",
    },
    {
        "name": "JumpReLU_65k_L12",
        "arch_type": "JumpReLU",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_12/width_65k/canonical",
        "layer": 12,
        "width_label": "65k",
    },
    {
        "name": "BatchTopK_16k_L12",
        "arch_type": "BatchTopK",
        "release": "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109",
        "sae_id": "blocks.12.hook_resid_post__trainer_0",
        "layer": 12,
        "width_label": "~16k",
    },
    {
        "name": "Matryoshka_L12",
        "arch_type": "Matryoshka",
        "release": "gemma-2-2b-res-matryoshka-dc",
        "sae_id": "blocks.12.hook_resid_post",
        "layer": 12,
        "width_label": "32k",
    },
]

# SAE architecture configs at Layer 24 (no BatchTopK available)
SAE_ARCHITECTURES_L24 = [
    {
        "name": "JumpReLU_16k_L24",
        "arch_type": "JumpReLU",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_24/width_16k/canonical",
        "layer": 24,
        "width_label": "16k",
    },
    {
        "name": "JumpReLU_65k_L24",
        "arch_type": "JumpReLU",
        "release": "gemma-scope-2b-pt-res-canonical",
        "sae_id": "layer_24/width_65k/canonical",
        "layer": 24,
        "width_label": "65k",
    },
    {
        "name": "Matryoshka_L24",
        "arch_type": "Matryoshka",
        "release": "gemma-2-2b-res-matryoshka-dc",
        "sae_id": "blocks.24.hook_resid_post",
        "layer": 24,
        "width_label": "32k",
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
TOKEN_POS_CROSSDOMAIN = -1

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
        actual_min = round((end_time - start_time).total_seconds() / 60, 1)
        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 15,
            "actual_min": actual_min,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "config_snapshot": {
                "model": "gemma-2-2b",
                "mode": MODE,
                "n_architectures_l12": len(SAE_ARCHITECTURES_L12),
                "n_architectures_l24": len(SAE_ARCHITECTURES_L24),
                "primary_layer": PRIMARY_LAYER,
                "secondary_layer": SECONDARY_LAYER,
                "gpu": DEVICE,
            }
        }

    gp_path.write_text(json.dumps(gp, indent=2))
    logger.info(f"GPU progress updated: {status}")


def update_experiment_state(status="completed"):
    """Update experiment_state.json."""
    es_path = WORKSPACE / "exp" / "experiment_state.json"
    try:
        es = json.loads(es_path.read_text())
    except Exception:
        es = {"schema_version": 1, "tasks": {}}

    if TASK_ID in es.get("tasks", {}):
        es["tasks"][TASK_ID]["status"] = status
        if status == "completed":
            es["tasks"][TASK_ID]["completed_at"] = datetime.now().isoformat()

    es_path.write_text(json.dumps(es, indent=2))


# ============================================================
# Model loading
# ============================================================
def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformer_lens import HookedTransformer

    logger.info(f"Loading Gemma 2 2B on {device}...")
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


def load_sae(release, sae_id, device):
    from sae_lens import SAE
    logger.info(f"Loading SAE: {release} / {sae_id} on {device}")
    sae = SAE.from_pretrained(release, sae_id, device=device)
    logger.info(f"  d_sae={sae.cfg.d_sae}, d_in={sae.cfg.d_in}")
    return sae


# ============================================================
# L0 estimation
# ============================================================
def estimate_l0(sae, activations, device):
    """Estimate L0 (average number of active features) for a batch of activations."""
    with torch.no_grad():
        batch_size = 32
        total_active = 0
        n_samples = 0
        for i in range(0, len(activations), batch_size):
            batch = activations[i:i+batch_size].to(device)
            features = sae.encode(batch)
            active_count = (features.abs() > 1e-6).float().sum(dim=-1)
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


def load_firstletter_probe(model, word_list, layer, device):
    """
    Train first-letter probe using sae_spelling ICL pipeline.
    Achieves F1 ~1.0 at L24, ~0.28 at L12 (limited because L12 hasn't learned spelling yet).
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
    logger.info(f"First-letter probe training data (L{layer}): {X.shape}, {len(set(y))} classes")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y if len(set(y)) > 1 else None
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

    quality = {
        "f1": float(best_f1), "accuracy": float(accuracy_score(y_test, best_probe.predict(X_test))),
        "quality_gate": "pass" if best_f1 >= 0.90 else ("below" if best_f1 >= 0.70 else "low"),
        "method": "sae_spelling_icl_sklearn", "layer": layer,
        "n_train": len(X_train), "n_test": len(X_test), "n_classes": len(set(y)),
    }
    logger.info(f"First-letter probe at L{layer}: F1={best_f1:.4f}, gate={quality['quality_gate']}")
    return best_probe, quality


def load_crossdomain_probe(hierarchy, layer):
    """Load cross-domain probe at given layer from saved npz files."""
    path = PHASE1_DIR / f"probe_{hierarchy}_L{layer}.npz"
    if path.exists():
        logger.info(f"Loading {hierarchy} probe (L{layer}): {path}")
        probe = reconstruct_probe_from_npz(path)
        return probe, path.name
    # Also check iter_008 probes
    iter008_path = WORKSPACE.parent / "iter_008" / "exp" / "results" / "phase1" / f"probe_{hierarchy}_L{layer}.npz"
    if iter008_path.exists():
        logger.info(f"Loading {hierarchy} probe from iter_008 (L{layer}): {iter008_path}")
        probe = reconstruct_probe_from_npz(iter008_path)
        return probe, f"iter008_{iter008_path.name}"
    logger.warning(f"No {hierarchy} probe found for layer {layer}")
    return None, None


# ============================================================
# Word list
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
def measure_firstletter_absorption(model, sae, probe, word_list, layer, device):
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
        act_tensor = torch.stack(all_activations[:100])
        l0_estimate = estimate_l0(sae, act_tensor, device)
        logger.info(f"  L0 estimate: {l0_estimate:.1f}")

    # Aggregate
    total_correct_raw = sum(s["probe_correct_raw"] for s in per_letter.values())
    total_fn = sum(s["false_negatives"] for s in per_letter.values())
    total_fn_absent = sum(s["fn_main_absent"] for s in per_letter.values())
    absorption_rate = total_fn / total_correct_raw if total_correct_raw > 0 else 0.0
    strict_rate = total_fn_absent / total_correct_raw if total_correct_raw > 0 else 0.0

    ci = _bootstrap_ci(per_letter)

    return {
        "absorption_rate": float(absorption_rate),
        "strict_absorption_rate": float(strict_rate),
        "total_words": sum(s["total"] for s in per_letter.values()),
        "total_probe_correct": total_correct_raw,
        "total_false_negatives": total_fn,
        "total_fn_main_absent": total_fn_absent,
        "l0_estimate": float(l0_estimate) if l0_estimate is not None else None,
        "bootstrap_ci": ci,
        "fn_examples": fn_examples[:10],
    }


def measure_crossdomain_absorption(model, sae, probe, cities, hierarchy, layer, device):
    """Measure absorption for a cross-domain hierarchy on a given SAE."""
    tl_hook = f"blocks.{layer}.hook_resid_post"
    attr_key = hierarchy.split("-")[1]  # "continent", "language", "country"

    probe_coefs = torch.tensor(probe.coef_, dtype=torch.float32)
    W_dec = sae.W_dec.detach().float().cpu()
    probe_classes = list(probe.classes_)

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
        prompt = f"The {attr_key} of {city_name} is"

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

    ci = _bootstrap_ci(dict(per_class))

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
        "fn_examples": fn_examples[:10],
    }


def _bootstrap_ci(per_unit_stats):
    """Compute bootstrap CI from per-letter/class stats."""
    per_sample_fn = []
    for s in per_unit_stats.values():
        if isinstance(s, dict) and "false_negatives" in s:
            per_sample_fn.extend([1] * s["false_negatives"])
            per_sample_fn.extend([0] * (s["probe_correct_raw"] - s["false_negatives"]))
    per_sample_fn = np.array(per_sample_fn)
    if len(per_sample_fn) > 0:
        rng = np.random.RandomState(SEED)
        boot_rates = [rng.choice(per_sample_fn, size=len(per_sample_fn), replace=True).mean()
                      for _ in range(N_BOOTSTRAP)]
        return {
            "mean": float(np.mean(boot_rates)),
            "ci_lower": float(np.percentile(boot_rates, 2.5)),
            "ci_upper": float(np.percentile(boot_rates, 97.5)),
            "std": float(np.std(boot_rates)),
        }
    return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}


# ============================================================
# Statistical helpers
# ============================================================
def proportions_z_test(rate_a, n_a, rate_b, n_b):
    """Two-proportion z-test."""
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
    return {"z_stat": float(z), "p_value": float(p), "significant_005": bool(p < 0.05)}


def compute_anova(arch_results, hierarchy_names):
    """Compute Kruskal-Wallis test for architecture and hierarchy effects."""
    try:
        from scipy.stats import kruskal

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
                arch_type = arch_data.get("arch_type", arch_name)
                arch_rates[arch_type].append(rate)
                hierarchy_rates[h].append(rate)

        # Architecture main effect
        arch_groups = [v for v in arch_rates.values() if len(v) > 0]
        if len(arch_groups) >= 2 and all(len(g) >= 2 for g in arch_groups):
            h_stat, p_arch = kruskal(*arch_groups)
            arch_effect = {"H_stat": float(h_stat), "p_value": float(p_arch),
                          "significant_005": bool(p_arch < 0.05), "test": "Kruskal-Wallis",
                          "groups": {k: len(v) for k, v in arch_rates.items()}}
        else:
            arch_effect = {"H_stat": None, "p_value": None, "significant_005": False,
                          "test": "insufficient_data",
                          "groups": {k: len(v) for k, v in arch_rates.items()}}

        # Hierarchy main effect
        hier_groups = [v for v in hierarchy_rates.values() if len(v) > 0]
        if len(hier_groups) >= 2 and all(len(g) >= 2 for g in hier_groups):
            h_stat, p_hier = kruskal(*hier_groups)
            hier_effect = {"H_stat": float(h_stat), "p_value": float(p_hier),
                          "significant_005": bool(p_hier < 0.05), "test": "Kruskal-Wallis",
                          "groups": {k: len(v) for k, v in hierarchy_rates.items()}}
        else:
            hier_effect = {"H_stat": None, "p_value": None, "significant_005": False,
                          "test": "insufficient_data",
                          "groups": {k: len(v) for k, v in hierarchy_rates.items()}}

        return {
            "architecture_effect": arch_effect,
            "hierarchy_effect": hier_effect,
            "note": "Kruskal-Wallis non-parametric test (small cell sizes in pilot)"
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================
# Fresh cross-domain probe training
# ============================================================
def train_crossdomain_probe_fresh(model, hierarchy, cities, layer, device):
    """Train cross-domain probe from scratch at a given layer."""
    hook_name = f"blocks.{layer}.hook_resid_post"
    attr_key = hierarchy.split("-")[1]

    labels = [c.get(attr_key, "") for c in cities if c.get(attr_key, "")]
    label_set = sorted(set(labels))
    if len(label_set) < 2:
        logger.error(f"Too few classes for {hierarchy}: {len(label_set)}")
        return None, None

    all_acts, all_labels = [], []
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
    logger.info(f"Cross-domain probe data for {hierarchy} at L{layer}: X={X.shape}, {len(set(y))} classes")

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
        "quality_gate": "pass" if best_f1 >= 0.90 else ("below" if best_f1 >= 0.70 else "low"),
        "method": "fresh_sklearn", "layer": layer,
        "n_train": len(X_train), "n_test": len(X_test),
    }
    logger.info(f"Fresh probe {hierarchy} L{layer}: F1={best_f1:.4f}, classes={n_classes}")
    return best_probe, quality


# ============================================================
# Main execution
# ============================================================
def main():
    start_time = datetime.now()
    write_pid()
    report_progress(0, 30, status="starting")

    device = DEVICE
    logger.info(f"=== Phase 1.4: Architecture Comparison (MODE={MODE}, device={device}) ===")
    logger.info(f"Primary layer: {PRIMARY_LAYER} (all architectures)")
    logger.info(f"Secondary layer: {SECONDARY_LAYER} (JumpReLU + Matryoshka only)")

    # Step 1: Load model
    model = load_model(device=device)
    tokenizer = model.tokenizer

    # Step 2: Prepare data
    logger.info("Preparing word list and city data...")
    word_list = get_word_list(tokenizer, max_words=MAX_WORDS_FIRSTLETTER)
    cities = load_ravel_dataset(max_cities=MAX_CITIES)

    report_progress(1, 30, metrics={"step": "data_prepared"})

    # ============================================================
    # Phase A: Layer 12 (all architectures)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE A: Layer 12 comparison (4 architectures)")
    logger.info("=" * 60)

    # Train first-letter probe at L12
    fl_probe_l12, fl_quality_l12 = load_firstletter_probe(model, word_list, layer=PRIMARY_LAYER, device=device)

    # Load/train crossdomain probes at L12
    crossdomain_hierarchies = ["city-continent"]  # Only continent has saved probe at L12
    crossdomain_probes_l12 = {}

    for hierarchy in crossdomain_hierarchies:
        if cities is None:
            continue
        probe, pname = load_crossdomain_probe(hierarchy, PRIMARY_LAYER)
        if probe is not None:
            crossdomain_probes_l12[hierarchy] = {"probe": probe, "quality": {"source": pname}, "probe_file": pname}
            logger.info(f"  {hierarchy} L{PRIMARY_LAYER}: loaded from {pname}")
        else:
            logger.info(f"  {hierarchy} L{PRIMARY_LAYER}: training fresh probe")
            fresh_probe, fresh_quality = train_crossdomain_probe_fresh(
                model, hierarchy, cities, PRIMARY_LAYER, device
            )
            if fresh_probe is not None:
                crossdomain_probes_l12[hierarchy] = {"probe": fresh_probe, "quality": fresh_quality, "probe_file": "fresh"}

    # Also try city-language at L12
    for hierarchy in ["city-language", "city-country"]:
        if cities is None:
            continue
        probe, pname = load_crossdomain_probe(hierarchy, PRIMARY_LAYER)
        if probe is not None:
            crossdomain_probes_l12[hierarchy] = {"probe": probe, "quality": {"source": pname}, "probe_file": pname}
        else:
            fresh_probe, fresh_quality = train_crossdomain_probe_fresh(
                model, hierarchy, cities, PRIMARY_LAYER, device
            )
            if fresh_probe is not None:
                crossdomain_probes_l12[hierarchy] = {"probe": fresh_probe, "quality": fresh_quality, "probe_file": "fresh"}

    report_progress(3, 30, metrics={"probes_l12": len(crossdomain_probes_l12) + 1})

    all_hierarchies_l12 = ["first-letter"] + [h for h in ["city-continent", "city-language", "city-country"]
                                                if h in crossdomain_probes_l12]

    # Run L12 architecture sweep
    arch_results_l12 = {}
    step = 3
    for arch_cfg in SAE_ARCHITECTURES_L12:
        arch_name = arch_cfg["name"]
        logger.info(f"\n--- {arch_name} ({arch_cfg['arch_type']}) ---")

        try:
            sae = load_sae(arch_cfg["release"], arch_cfg["sae_id"], device)
        except Exception as e:
            logger.error(f"Failed to load SAE {arch_name}: {e}")
            arch_results_l12[arch_name] = {"error": str(e), "arch_type": arch_cfg["arch_type"]}
            step += len(all_hierarchies_l12)
            report_progress(step, 30, metrics={"current_arch": arch_name, "status": "load_failed"})
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

        # First-letter
        logger.info(f"  Measuring first-letter absorption at L{PRIMARY_LAYER}...")
        try:
            fl_result = measure_firstletter_absorption(model, sae, fl_probe_l12, word_list, PRIMARY_LAYER, device)
            arch_data["hierarchies"]["first-letter"] = fl_result
            logger.info(f"  first-letter: rate={fl_result['absorption_rate']:.4f} "
                       f"strict={fl_result['strict_absorption_rate']:.4f} "
                       f"FN={fl_result['total_false_negatives']}/{fl_result['total_probe_correct']} "
                       f"L0={fl_result.get('l0_estimate', 'N/A')}")
        except Exception as e:
            logger.error(f"  first-letter failed: {e}")
            arch_data["hierarchies"]["first-letter"] = {"error": str(e)}
        step += 1
        report_progress(step, 30, metrics={"arch": arch_name, "hierarchy": "first-letter"})
        torch.cuda.empty_cache()

        # Cross-domain
        for hierarchy in ["city-continent", "city-language", "city-country"]:
            if hierarchy not in crossdomain_probes_l12:
                continue
            pdata = crossdomain_probes_l12[hierarchy]
            logger.info(f"  Measuring {hierarchy} at L{PRIMARY_LAYER}...")
            try:
                cd_result = measure_crossdomain_absorption(
                    model, sae, pdata["probe"], cities, hierarchy, PRIMARY_LAYER, device
                )
                arch_data["hierarchies"][hierarchy] = cd_result
                logger.info(f"  {hierarchy}: rate={cd_result['absorption_rate']:.4f} "
                           f"FN={cd_result['total_false_negatives']}/{cd_result['total_probe_correct']} "
                           f"L0={cd_result.get('l0_estimate', 'N/A')}")
            except Exception as e:
                logger.error(f"  {hierarchy} failed: {e}")
                arch_data["hierarchies"][hierarchy] = {"error": str(e)}
            step += 1
            report_progress(step, 30, metrics={"arch": arch_name, "hierarchy": hierarchy})
            torch.cuda.empty_cache()

        del sae; gc.collect(); torch.cuda.empty_cache()
        arch_results_l12[arch_name] = arch_data

    # ============================================================
    # Phase B: Layer 24 (JumpReLU + Matryoshka)
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE B: Layer 24 comparison (JumpReLU + Matryoshka)")
    logger.info("=" * 60)

    # Train first-letter probe at L24
    fl_probe_l24, fl_quality_l24 = load_firstletter_probe(model, word_list, layer=SECONDARY_LAYER, device=device)

    # Load crossdomain probes at L24
    crossdomain_probes_l24 = {}
    for hierarchy in ["city-continent", "city-language", "city-country"]:
        if cities is None:
            continue
        probe, pname = load_crossdomain_probe(hierarchy, SECONDARY_LAYER)
        if probe is not None:
            crossdomain_probes_l24[hierarchy] = {"probe": probe, "quality": {"source": pname}, "probe_file": pname}
        else:
            fresh_probe, fresh_quality = train_crossdomain_probe_fresh(
                model, hierarchy, cities, SECONDARY_LAYER, device
            )
            if fresh_probe is not None:
                crossdomain_probes_l24[hierarchy] = {"probe": fresh_probe, "quality": fresh_quality, "probe_file": "fresh"}

    all_hierarchies_l24 = ["first-letter"] + [h for h in ["city-continent", "city-language", "city-country"]
                                                if h in crossdomain_probes_l24]

    # Run L24 architecture sweep
    arch_results_l24 = {}
    for arch_cfg in SAE_ARCHITECTURES_L24:
        arch_name = arch_cfg["name"]
        logger.info(f"\n--- {arch_name} ({arch_cfg['arch_type']}) ---")

        try:
            sae = load_sae(arch_cfg["release"], arch_cfg["sae_id"], device)
        except Exception as e:
            logger.error(f"Failed to load SAE {arch_name}: {e}")
            arch_results_l24[arch_name] = {"error": str(e), "arch_type": arch_cfg["arch_type"]}
            step += len(all_hierarchies_l24)
            report_progress(step, 30, metrics={"current_arch": arch_name, "status": "load_failed"})
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

        # First-letter
        logger.info(f"  Measuring first-letter absorption at L{SECONDARY_LAYER}...")
        try:
            fl_result = measure_firstletter_absorption(model, sae, fl_probe_l24, word_list, SECONDARY_LAYER, device)
            arch_data["hierarchies"]["first-letter"] = fl_result
            logger.info(f"  first-letter: rate={fl_result['absorption_rate']:.4f} "
                       f"strict={fl_result['strict_absorption_rate']:.4f} "
                       f"FN={fl_result['total_false_negatives']}/{fl_result['total_probe_correct']} "
                       f"L0={fl_result.get('l0_estimate', 'N/A')}")
        except Exception as e:
            logger.error(f"  first-letter failed: {e}")
            arch_data["hierarchies"]["first-letter"] = {"error": str(e)}
        step += 1
        report_progress(step, 30, metrics={"arch": arch_name, "hierarchy": "first-letter", "layer": 24})
        torch.cuda.empty_cache()

        # Cross-domain at L24
        for hierarchy in ["city-continent", "city-language", "city-country"]:
            if hierarchy not in crossdomain_probes_l24:
                continue
            pdata = crossdomain_probes_l24[hierarchy]
            logger.info(f"  Measuring {hierarchy} at L{SECONDARY_LAYER}...")
            try:
                cd_result = measure_crossdomain_absorption(
                    model, sae, pdata["probe"], cities, hierarchy, SECONDARY_LAYER, device
                )
                arch_data["hierarchies"][hierarchy] = cd_result
                logger.info(f"  {hierarchy}: rate={cd_result['absorption_rate']:.4f} "
                           f"FN={cd_result['total_false_negatives']}/{cd_result['total_probe_correct']} "
                           f"L0={cd_result.get('l0_estimate', 'N/A')}")
            except Exception as e:
                logger.error(f"  {hierarchy} failed: {e}")
                arch_data["hierarchies"][hierarchy] = {"error": str(e)}
            step += 1
            report_progress(step, 30, metrics={"arch": arch_name, "hierarchy": hierarchy, "layer": 24})
            torch.cuda.empty_cache()

        del sae; gc.collect(); torch.cuda.empty_cache()
        arch_results_l24[arch_name] = arch_data

    # ============================================================
    # Compile results and statistics
    # ============================================================
    logger.info("\n" + "=" * 60)
    logger.info("Computing architecture comparisons...")
    logger.info("=" * 60)

    # Merge all results
    all_arch_results = {}
    all_arch_results.update(arch_results_l12)
    all_arch_results.update(arch_results_l24)

    # Pairwise comparisons (within same layer)
    comparisons_l12 = _compute_pairwise(arch_results_l12, all_hierarchies_l12)
    comparisons_l24 = _compute_pairwise(arch_results_l24, all_hierarchies_l24)

    # ANOVA at L12 (primary)
    anova_l12 = compute_anova(arch_results_l12, all_hierarchies_l12)
    logger.info(f"ANOVA L12: {json.dumps(anova_l12, indent=2)}")

    # ANOVA at L24 (supplementary)
    anova_l24 = compute_anova(arch_results_l24, all_hierarchies_l24)
    logger.info(f"ANOVA L24: {json.dumps(anova_l24, indent=2)}")

    # Summary tables
    summary_l12 = _build_summary_table(arch_results_l12, all_hierarchies_l12, SAE_ARCHITECTURES_L12)
    summary_l24 = _build_summary_table(arch_results_l24, all_hierarchies_l24, SAE_ARCHITECTURES_L24)

    # JumpReLU advantage analysis (per layer)
    jumprelu_adv_l12 = _compute_jumprelu_advantage(arch_results_l12, all_hierarchies_l12)
    jumprelu_adv_l24 = _compute_jumprelu_advantage(arch_results_l24, all_hierarchies_l24)

    # H6 verdict
    all_jumprelu_adv = {**{f"L12_{k}": v for k, v in jumprelu_adv_l12.items()},
                        **{f"L24_{k}": v for k, v in jumprelu_adv_l24.items()}}
    n_jumprelu_best = sum(1 for v in all_jumprelu_adv.values() if v.get("jumprelu_is_lowest", False))
    n_total = len(all_jumprelu_adv)
    h6_verdict = ("SUPPORTED" if n_total > 0 and n_jumprelu_best == n_total
                  else "PARTIALLY_SUPPORTED" if n_jumprelu_best > 0
                  else "NOT_SUPPORTED")

    # Compile final results
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()

    # Iter_008 baselines for comparison
    iter008_baselines = {
        "architecture_anova_p": 0.87,
        "hierarchy_anova_p": 0.005,
        "h6_verdict": "PARTIALLY_SUPPORTED",
    }

    results = {
        "task_id": TASK_ID,
        "mode": MODE,
        "version": "iter009_v1",
        "timestamp": start_time.isoformat(),
        "seed": SEED,
        "model": "gemma-2-2b",
        "gpu": DEVICE,
        "layers_tested": [PRIMARY_LAYER, SECONDARY_LAYER],
        "primary_layer": PRIMARY_LAYER,
        "secondary_layer": SECONDARY_LAYER,

        # Layer 12 results (primary -- all 3 architectures)
        "l12": {
            "architectures_tested": [a["name"] for a in SAE_ARCHITECTURES_L12],
            "hierarchies_tested": all_hierarchies_l12,
            "architecture_results": arch_results_l12,
            "comparisons": comparisons_l12,
            "anova": anova_l12,
            "summary_table": summary_l12,
            "jumprelu_advantage": jumprelu_adv_l12,
        },

        # Layer 24 results (supplementary -- JumpReLU + Matryoshka)
        "l24": {
            "architectures_tested": [a["name"] for a in SAE_ARCHITECTURES_L24],
            "hierarchies_tested": all_hierarchies_l24,
            "architecture_results": arch_results_l24,
            "comparisons": comparisons_l24,
            "anova": anova_l24,
            "summary_table": summary_l24,
            "jumprelu_advantage": jumprelu_adv_l24,
            "note": "BatchTopK SAE not available at L24; only JumpReLU and Matryoshka compared",
        },

        # Combined analysis
        "h6_hypothesis": {
            "description": "Does architecture advantage generalize across hierarchy types?",
            "verdict": h6_verdict,
            "n_total_comparisons": n_total,
            "n_jumprelu_lowest": n_jumprelu_best,
            "per_layer_verdict": {
                "l12": jumprelu_adv_l12,
                "l24": jumprelu_adv_l24,
            },
        },

        "probe_quality": {
            "first-letter_L12": fl_quality_l12,
            "first-letter_L24": fl_quality_l24,
            "crossdomain_L12": {h: pdata["quality"] for h, pdata in crossdomain_probes_l12.items()},
            "crossdomain_L24": {h: pdata["quality"] for h, pdata in crossdomain_probes_l24.items()},
        },

        "iter008_baselines": iter008_baselines,
        "elapsed_seconds": elapsed,
        "elapsed_minutes": elapsed / 60,
    }

    # Pilot criteria check
    n_arch_l12 = sum(1 for a in arch_results_l12.values()
                     if "hierarchies" in a and "error" not in a)
    n_arch_l24 = sum(1 for a in arch_results_l24.values()
                     if "hierarchies" in a and "error" not in a)
    pilot_pass = (n_arch_l12 >= 3 and len(all_hierarchies_l12) >= 1 and
                  n_arch_l24 >= 2 and len(all_hierarchies_l24) >= 1)
    results["pilot_criteria_met"] = pilot_pass
    results["pilot_criteria_details"] = {
        "l12_architectures_with_results": n_arch_l12,
        "l12_hierarchies_measured": len(all_hierarchies_l12),
        "l24_architectures_with_results": n_arch_l24,
        "l24_hierarchies_measured": len(all_hierarchies_l24),
        "minimum_l12_architectures": 3,
        "minimum_l24_architectures": 2,
    }

    # Save results
    out_path = PHASE1_DIR / "architecture_comparison.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Results saved: {out_path}")

    pilot_path = PILOT_DIR / "phase1_architecture_comparison.json"
    pilot_path.write_text(json.dumps(results, indent=2, default=str))
    logger.info(f"Pilot copy saved: {pilot_path}")

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("ARCHITECTURE COMPARISON SUMMARY")
    logger.info("=" * 60)
    logger.info(f"\n--- Layer 12 (PRIMARY: all architectures) ---")
    for row in summary_l12:
        l0_str = f" L0={row['l0_estimate']:.0f}" if row.get('l0_estimate') else ""
        logger.info(f"  {row['hierarchy']:15s} | {row['architecture']:25s} | "
                    f"rate={row['absorption_rate']:.4f}{l0_str} "
                    f"FN={row['n_fn']}/{row['n_probe_correct']}")
    logger.info(f"\n--- Layer 24 (SUPPLEMENTARY: JumpReLU + Matryoshka) ---")
    for row in summary_l24:
        l0_str = f" L0={row['l0_estimate']:.0f}" if row.get('l0_estimate') else ""
        logger.info(f"  {row['hierarchy']:15s} | {row['architecture']:25s} | "
                    f"rate={row['absorption_rate']:.4f}{l0_str} "
                    f"FN={row['n_fn']}/{row['n_probe_correct']}")

    logger.info(f"\nANOVA L12: arch_p={anova_l12.get('architecture_effect',{}).get('p_value','N/A')}, "
                f"hier_p={anova_l12.get('hierarchy_effect',{}).get('p_value','N/A')}")
    logger.info(f"ANOVA L24: arch_p={anova_l24.get('architecture_effect',{}).get('p_value','N/A')}, "
                f"hier_p={anova_l24.get('hierarchy_effect',{}).get('p_value','N/A')}")
    logger.info(f"H6 verdict: {h6_verdict}")
    logger.info(f"Pilot pass: {pilot_pass}")
    logger.info(f"Elapsed: {elapsed/60:.1f} min")

    # Mark done and update progress
    mark_done("success", f"L12: {n_arch_l12} archs, L24: {n_arch_l24} archs, H6={h6_verdict}")
    update_gpu_progress("completed", start_time, end_time)
    update_experiment_state("completed")

    return results


def _compute_pairwise(arch_results, hierarchy_names):
    """Compute pairwise z-tests between architectures."""
    comparisons = {}
    arch_names = [an for an, ad in arch_results.items() if "error" not in ad]

    for h in hierarchy_names:
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

    return comparisons


def _build_summary_table(arch_results, hierarchy_names, arch_configs):
    """Build summary table rows."""
    table = []
    for h in hierarchy_names:
        for arch_cfg in arch_configs:
            an = arch_cfg["name"]
            ad = arch_results.get(an, {})
            if "error" in ad:
                continue
            r = ad.get("hierarchies", {}).get(h, {})
            if "error" in r or "absorption_rate" not in r:
                continue
            table.append({
                "hierarchy": h,
                "architecture": an,
                "arch_type": ad.get("arch_type", "unknown"),
                "d_sae": ad.get("d_sae", 0),
                "layer": ad.get("layer", 0),
                "absorption_rate": r.get("absorption_rate"),
                "strict_rate": r.get("strict_absorption_rate"),
                "ci_lower": r.get("bootstrap_ci", {}).get("ci_lower"),
                "ci_upper": r.get("bootstrap_ci", {}).get("ci_upper"),
                "l0_estimate": r.get("l0_estimate"),
                "n_probe_correct": r.get("total_probe_correct", 0),
                "n_fn": r.get("total_false_negatives", 0),
            })
    return table


def _compute_jumprelu_advantage(arch_results, hierarchy_names):
    """Analyze whether JumpReLU has lowest absorption across hierarchies."""
    arch_names = [an for an, ad in arch_results.items() if "error" not in ad]
    advantage = {}

    for h in hierarchy_names:
        jumprelu_rates = []
        other_rates = []
        for an in arch_names:
            r = arch_results.get(an, {}).get("hierarchies", {}).get(h, {})
            if "error" in r or "absorption_rate" not in r:
                continue
            rate = r["absorption_rate"]
            arch_type = arch_results[an].get("arch_type", "")
            if arch_type == "JumpReLU":
                jumprelu_rates.append((an, rate))
            else:
                other_rates.append((an, rate))

        if jumprelu_rates and other_rates:
            best_jumprelu = min(jumprelu_rates, key=lambda x: x[1])
            best_other = min(other_rates, key=lambda x: x[1])
            advantage[h] = {
                "best_jumprelu": {"arch": best_jumprelu[0], "rate": float(best_jumprelu[1])},
                "best_other": {"arch": best_other[0], "rate": float(best_other[1])},
                "jumprelu_is_lowest": best_jumprelu[1] <= best_other[1],
            }

    return advantage


if __name__ == "__main__":
    try:
        results = main()
    except Exception as e:
        logger.error(f"FATAL: {e}", exc_info=True)
        mark_done("failed", str(e))
        update_gpu_progress("failed", datetime.now(), datetime.now())
        update_experiment_state("failed")
        raise
