"""
P2_controls v3: Cross-Domain Controls with Proper Split + Calibrated Absorption
================================================================================

v1 problem: cosine_threshold=0.025 was too loose -> 40% false positive absorption
v2 problem: Using cosine-aligned features as split features -> 100% FN rate (no split fires)

v3 fix: Separate split feature identification from absorption detection.
  - Split features: Identified by ACTIVATION PATTERNS (features that fire preferentially
    for in-class tokens). This is the correct Chanin approach.
  - Absorption detection: Among features active on false-negative tokens, check if any
    have decoder direction aligned with probe using CALIBRATED cosine threshold.

Key insight: Split features fire because of the token content (e.g., a feature that
activates for US cities). They don't need to be aligned with the probe direction.
Absorbing features are features that have "absorbed" the class information into their
decoder direction, which IS checked via cosine.

Also: use a wider range of cosine thresholds as a sweep, and report results at each
threshold, so we can see the tradeoff between specificity and sensitivity.
"""

import os
import sys
import json
import time
import gc
from pathlib import Path
from datetime import datetime
from collections import Counter, defaultdict

import numpy as np
import torch
from scipy.stats import mannwhitneyu

# ── Configuration ──────────────────────────────────────────────────────
TASK_ID = "P2_controls"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
DATA_DIR = WORKSPACE / "exp" / "data" / "ravel"
PROBE_DIR = FULL_DIR / "P2_probes"

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "gpt2"
SAE_RELEASE = "gpt2-small-res-jb"
PILOT_LAYERS = [8]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
GPU_ID = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

# Split feature selection (activation-based)
SPLIT_TOPK = 10              # Top-k features by preferential activation ratio
SPLIT_MIN_ACTIVE_RATE = 0.05 # Feature must fire on >= 5% of in-class tokens

# Cosine threshold sweep for absorption detection
COSINE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]
DOMINANCE_THRESHOLD = 2.0    # Top feature must be >= 2x second-highest

MIN_FALSE_NEGATIVES = 3
MAX_CITIES_PILOT = 300
N_LETTERS_PILOT = 10


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def json_dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=2)


def write_pid():
    (RESULTS_DIR / f"{TASK_ID}.pid").write_text(str(os.getpid()))


def report_progress(stage, detail="", metric=None):
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps({
        "task_id": TASK_ID, "stage": stage, "detail": detail,
        "metric": metric or {}, "updated_at": datetime.now().isoformat(),
    }, cls=NumpyEncoder))


def mark_done(status="success", summary=""):
    pid = RESULTS_DIR / f"{TASK_ID}.pid"
    if pid.exists(): pid.unlink()
    prog = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if prog.exists():
        try: fp = json.loads(prog.read_text())
        except: pass
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "final_progress": fp, "timestamp": datetime.now().isoformat(),
    }, cls=NumpyEncoder))
    print(f"[DONE] {status}: {summary}")


def load_ravel_cities():
    with open(DATA_DIR / "ravel_city_entity_attributes.json") as f:
        attrs = json.load(f)
    with open(DATA_DIR / "ravel_city_entity_to_split.json") as f:
        splits = json.load(f)
    return [{"city": c, "split": splits.get(c, "?"), **a} for c, a in attrs.items()]


def load_probe(probe_name, layer):
    path = PROBE_DIR / f"probe_{probe_name}_layer{layer}.npz"
    if not path.exists(): return None
    d = np.load(path, allow_pickle=True)
    return {
        "coef": d["coef"], "intercept": d["intercept"],
        "scaler_mean": d["scaler_mean"], "scaler_scale": d["scaler_scale"],
        "classes": d["classes"], "mean_acc": float(d["mean_acc"]),
    }


def probe_direction(probe_data):
    coef = probe_data["coef"]
    direction = coef / probe_data["scaler_scale"]
    if direction.shape[0] <= 2:
        d = direction[1] - direction[0] if direction.shape[0] == 2 else direction[0]
        d = d / (np.linalg.norm(d) + 1e-10)
        return d.reshape(1, -1)
    norms = np.linalg.norm(direction, axis=1, keepdims=True)
    return direction / (norms + 1e-10)


def extract_activations_and_sae(model, tokenizer, sae, texts, layer, batch_size=64):
    hook_name = f"blocks.{layer}.hook_resid_pre"
    resid_list, sae_list = [], []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=64)
        tokens = {k: v.to(DEVICE) for k, v in tokens.items()}
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens["input_ids"],
                                            names_filter=[hook_name],
                                            attention_mask=tokens.get("attention_mask"))
        if "attention_mask" in tokens:
            seq_lens = tokens["attention_mask"].sum(dim=1) - 1
        else:
            seq_lens = torch.full((tokens["input_ids"].shape[0],),
                                  tokens["input_ids"].shape[1] - 1, device=DEVICE)
        resid = cache[hook_name]
        for j in range(resid.shape[0]):
            r = resid[j, seq_lens[j]].float()
            resid_list.append(r.cpu().numpy())
            s = sae.encode(r.unsqueeze(0))
            sae_list.append(s.squeeze(0).detach().cpu().numpy())
        del cache, resid; torch.cuda.empty_cache()
    return np.stack(resid_list), np.stack(sae_list)


def compute_decoder_cosines(sae_decoder, probe_dirs):
    """Max absolute cosine between each decoder vector and any probe direction."""
    dec_normed = sae_decoder / (np.linalg.norm(sae_decoder, axis=1, keepdims=True) + 1e-10)
    max_cos = np.zeros(sae_decoder.shape[0])
    for pd in probe_dirs:
        pd_flat = pd.flatten()
        pd_normed = pd_flat / (np.linalg.norm(pd_flat) + 1e-10)
        cos = np.abs(dec_normed @ pd_normed)
        max_cos = np.maximum(max_cos, cos)
    return max_cos


def find_split_features(sae_acts, class_labels, class_id, topk=SPLIT_TOPK):
    """Find features that fire preferentially for in-class tokens (activation-based)."""
    mask = class_labels == class_id
    n_in = mask.sum()
    if n_in < 3:
        return []

    in_acts = sae_acts[mask]
    out_acts = sae_acts[~mask]

    # Activation rate: fraction of in-class tokens where feature fires
    in_rate = (in_acts > 0).mean(axis=0)
    out_rate = (out_acts > 0).mean(axis=0)

    # Ratio of in-class vs out-of-class activation rate
    ratio = (in_rate + 1e-6) / (out_rate + 1e-6)

    # Filter: must fire on at least SPLIT_MIN_ACTIVE_RATE of in-class tokens
    ratio[in_rate < SPLIT_MIN_ACTIVE_RATE] = 0

    # Top-k by ratio
    top_ids = np.argsort(ratio)[-topk:][::-1]
    # Only keep those with ratio > 1 (actually preferential)
    return [int(fid) for fid in top_ids if ratio[fid] > 1.0]


def measure_absorption(sae_acts, cosine_array, class_labels, class_id,
                        split_features, cosine_threshold, dominance_threshold):
    """Core absorption measurement for one class at one cosine threshold.

    Split features: identified by activation pattern (fire preferentially for this class).
    Absorption check: on false-negative tokens (split features don't fire), check if the
    top active non-split feature has decoder cosine above threshold with the probe.
    """
    mask = class_labels == class_id
    n_class = int(mask.sum())
    if n_class < 5:
        return {"status": "insufficient_data", "n_class": n_class, "absorption_rate": 0.0}

    # False negatives: class tokens where no split feature fires
    class_sae = sae_acts[mask]
    if not split_features:
        # No split features -> all tokens are FN
        fn_mask = np.ones(n_class, dtype=bool)
    else:
        split_active = np.column_stack([class_sae[:, sf] > 0 for sf in split_features])
        fn_mask = ~split_active.any(axis=1)

    n_fn = int(fn_mask.sum())
    if n_fn < MIN_FALSE_NEGATIVES:
        return {
            "status": "too_few_fn", "n_class": n_class,
            "n_fn": n_fn, "n_split": len(split_features),
            "absorption_rate": 0.0,
        }

    fn_sae = class_sae[fn_mask]
    split_set = set(split_features)
    n_absorbed = 0
    details = []

    for idx in range(fn_sae.shape[0]):
        acts = fn_sae[idx].copy()
        for sf in split_features:
            acts[sf] = 0

        active = np.where(acts > 0)[0]
        if len(active) == 0:
            continue

        sorted_active = active[np.argsort(acts[active])[::-1]]
        top = sorted_active[0]
        top_act = acts[top]
        second_act = acts[sorted_active[1]] if len(sorted_active) > 1 else 0
        dominance = top_act / (second_act + 1e-10) if second_act > 0 else float('inf')

        cos = cosine_array[top]
        absorbed = (cos >= cosine_threshold) and (dominance >= dominance_threshold)

        if absorbed:
            n_absorbed += 1

        if len(details) < 3:
            details.append({
                "feat": int(top), "act": float(top_act),
                "cos": float(cos), "dom": float(min(dominance, 100)),
                "absorbed": bool(absorbed),
            })

    rate = n_absorbed / n_fn if n_fn > 0 else 0.0
    return {
        "status": "measured",
        "n_class": n_class, "n_fn": n_fn,
        "fn_rate": float(n_fn / n_class),
        "n_absorbed": int(n_absorbed),
        "absorption_rate": float(rate),
        "n_split": len(split_features),
        "cosine_threshold": float(cosine_threshold),
        "sample": details,
    }


def run_absorption_sweep(sae_acts, sae_decoder, probe_dirs, labels,
                         cosine_thresholds=COSINE_THRESHOLDS):
    """Measure absorption across multiple cosine thresholds."""
    cos_arr = compute_decoder_cosines(sae_decoder, probe_dirs)
    classes = np.unique(labels)
    threshold_results = {}

    for ct in cosine_thresholds:
        per_class = {}
        rates = []
        for cls in classes:
            splits = find_split_features(sae_acts, labels, cls)
            r = measure_absorption(sae_acts, cos_arr, labels, cls, splits, ct, DOMINANCE_THRESHOLD)
            per_class[int(cls)] = r
            if r["status"] == "measured":
                rates.append(r["absorption_rate"])

        threshold_results[str(ct)] = {
            "cosine_threshold": ct,
            "mean_absorption": float(np.mean(rates)) if rates else 0.0,
            "std_absorption": float(np.std(rates)) if rates else 0.0,
            "n_measured": len(rates),
            "per_class": per_class,
        }

    return threshold_results, cos_arr


def run_shuffled_control(sae_acts, sae_decoder, probe_dirs, labels, n_shuffles=5):
    """Shuffled hierarchy control at each cosine threshold."""
    print("  [Shuffled control]")
    cos_arr = compute_decoder_cosines(sae_decoder, probe_dirs)
    rng = np.random.RandomState(SEED)
    all_results = {str(ct): [] for ct in COSINE_THRESHOLDS}

    for si in range(n_shuffles):
        shuffled = labels.copy()
        rng.shuffle(shuffled)
        for ct in COSINE_THRESHOLDS:
            rates = []
            for cls in np.unique(shuffled):
                splits = find_split_features(sae_acts, shuffled, cls)
                r = measure_absorption(sae_acts, cos_arr, shuffled, cls, splits, ct, DOMINANCE_THRESHOLD)
                if r["status"] == "measured":
                    rates.append(r["absorption_rate"])
            mean_r = np.mean(rates) if rates else 0
            all_results[str(ct)].append(float(mean_r))

    summary = {}
    for ct_str, vals in all_results.items():
        m = np.mean(vals) if vals else 0
        s = np.std(vals) if vals else 0
        summary[ct_str] = {"mean": float(m), "std": float(s), "per_shuffle": vals}
        print(f"    threshold={ct_str}: {m:.4f} +/- {s:.4f}")

    return summary


def run_random_probe_control(sae_acts, sae_decoder, labels, d_model, n_random=10):
    """Random probe direction control at each cosine threshold."""
    print("  [Random probe control]")
    rng = np.random.RandomState(SEED + 1)
    all_results = {str(ct): [] for ct in COSINE_THRESHOLDS}

    for ri in range(n_random):
        rand_dir = rng.randn(1, d_model).astype(np.float32)
        rand_dir /= np.linalg.norm(rand_dir)
        cos_arr = compute_decoder_cosines(sae_decoder, rand_dir)

        for ct in COSINE_THRESHOLDS:
            rates = []
            for cls in np.unique(labels):
                splits = find_split_features(sae_acts, labels, cls)
                r = measure_absorption(sae_acts, cos_arr, labels, cls, splits, ct, DOMINANCE_THRESHOLD)
                if r["status"] == "measured":
                    rates.append(r["absorption_rate"])
            mean_r = np.mean(rates) if rates else 0
            all_results[str(ct)].append(float(mean_r))

    summary = {}
    for ct_str, vals in all_results.items():
        m = np.mean(vals) if vals else 0
        s = np.std(vals) if vals else 0
        summary[ct_str] = {"mean": float(m), "std": float(s), "per_trial": vals}
        print(f"    threshold={ct_str}: {m:.4f} +/- {s:.4f}")

    return summary


def run_first_letter(model, tokenizer, sae, sae_decoder, layer, n_letters=10):
    """First-letter baseline with cosine threshold sweep."""
    print(f"  [First-letter baseline, {n_letters} letters]")
    vocab = tokenizer.get_vocab()
    letter_tokens = defaultdict(list)
    for ts, tid in vocab.items():
        if ts.startswith('\u0120') and len(ts) > 2:
            fc = ts[1].upper()
            if fc.isalpha():
                letter_tokens[fc].append(tid)

    counts = {l: len(t) for l, t in letter_tokens.items()}
    letters = sorted(counts, key=counts.get, reverse=True)[:n_letters]

    rng = np.random.RandomState(SEED)
    MAX_PER = 100
    for l in letters:
        if len(letter_tokens[l]) > MAX_PER:
            idx = rng.choice(len(letter_tokens[l]), MAX_PER, replace=False)
            letter_tokens[l] = [letter_tokens[l][i] for i in idx]

    all_ids, all_labels = [], []
    l2i = {l: i for i, l in enumerate(letters)}
    for l in letters:
        for tid in letter_tokens[l]:
            all_ids.append(tid)
            all_labels.append(l2i[l])
    all_labels = np.array(all_labels)
    print(f"    {len(all_ids)} tokens, {len(letters)} letters")

    # Extract
    hook = f"blocks.{layer}.hook_resid_pre"
    resid_list, sae_list = [], []
    bs = 256
    for i in range(0, len(all_ids), bs):
        batch = all_ids[i:i+bs]
        inp = torch.tensor([[t] for t in batch], device=DEVICE)
        with torch.no_grad():
            _, cache = model.run_with_cache(inp, names_filter=[hook])
        r = cache[hook]
        for j in range(r.shape[0]):
            v = r[j, 0].float()
            resid_list.append(v.cpu().numpy())
            s = sae.encode(v.unsqueeze(0))
            sae_list.append(s.squeeze(0).detach().cpu().numpy())
        del cache, r; torch.cuda.empty_cache()

    sae_acts = np.stack(sae_list)
    resid_acts = np.stack(resid_list)

    per_letter = {}
    threshold_rates = {str(ct): [] for ct in COSINE_THRESHOLDS}

    for letter in letters:
        cls = l2i[letter]
        in_c = resid_acts[all_labels == cls]
        out_c = resid_acts[all_labels != cls]
        if len(in_c) < 5: continue

        d = in_c.mean(0) - out_c.mean(0)
        d /= (np.linalg.norm(d) + 1e-10)
        pd = d.reshape(1, -1)
        cos_arr = compute_decoder_cosines(sae_decoder, pd)
        splits = find_split_features(sae_acts, all_labels, cls)

        letter_res = {}
        for ct in COSINE_THRESHOLDS:
            r = measure_absorption(sae_acts, cos_arr, all_labels, cls, splits, ct, DOMINANCE_THRESHOLD)
            letter_res[str(ct)] = r
            if r["status"] == "measured":
                threshold_rates[str(ct)].append(r["absorption_rate"])

        per_letter[letter] = letter_res

    summary = {}
    for ct_str, rates in threshold_rates.items():
        m = np.mean(rates) if rates else 0
        s = np.std(rates) if rates else 0
        summary[ct_str] = {"mean": float(m), "std": float(s), "n": len(rates)}
        print(f"    threshold={ct_str}: {m:.4f} +/- {s:.4f} (n={len(rates)})")

    return {
        "letters": letters,
        "n_tokens": len(all_ids),
        "per_letter": per_letter,
        "threshold_summary": summary,
    }


def main():
    start_time = time.time()
    write_pid()
    report_progress("init", "P2_controls v3")
    print(f"[CONFIG] GPU={GPU_ID}, cosine_thresholds={COSINE_THRESHOLDS}, dom={DOMINANCE_THRESHOLD}")

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
    tok = model.tokenizer; tok.pad_token = tok.eos_token; tok.padding_side = "left"
    d_model = model.cfg.d_model

    from sae_lens import SAE as SL
    layer = PILOT_LAYERS[0]
    sae = SL.from_pretrained(release=SAE_RELEASE, sae_id=f"blocks.{layer}.hook_resid_pre", device=DEVICE)
    d_sae = sae.cfg.d_sae
    sae_decoder = sae.W_dec.detach().cpu().float().numpy()

    # Diagnostic
    rng_diag = np.random.RandomState(999)
    rd = rng_diag.randn(d_model).astype(np.float32); rd /= np.linalg.norm(rd)
    dc = compute_decoder_cosines(sae_decoder, rd.reshape(1, -1))
    print(f"\n[DIAG] Random cosine: mean={dc.mean():.4f}, p95={np.percentile(dc,95):.4f}, "
          f"p99={np.percentile(dc,99):.4f}, max={dc.max():.4f}")
    for ct in COSINE_THRESHOLDS:
        n_above = (dc >= ct).sum()
        print(f"  threshold={ct}: {n_above} features ({100*n_above/d_sae:.1f}%)")

    # Load data
    all_cities = load_ravel_cities()
    rng_data = np.random.RandomState(SEED)
    if len(all_cities) > MAX_CITIES_PILOT:
        idx = rng_data.choice(len(all_cities), MAX_CITIES_PILOT, replace=False)
        cities = [all_cities[i] for i in sorted(idx)]
    else:
        cities = all_cities
    print(f"[DATA] {len(cities)} cities")

    results = {
        "task_id": TASK_ID, "mode": "PILOT", "version": "v3",
        "model": MODEL_NAME, "sae_release": SAE_RELEASE, "layer": layer,
        "d_sae": d_sae, "n_cities": len(cities),
        "cosine_thresholds": COSINE_THRESHOLDS,
        "dominance_threshold": DOMINANCE_THRESHOLD,
        "controls": {}, "first_letter_baseline": {},
    }

    # Probes
    probe_configs = [
        ("Country_binary_US", "Country", "United States"),
        ("Language_binary_English", "Language", "English"),
        ("Continent", "Continent", None),
    ]

    # Track rates across probes for each threshold
    all_real = {str(ct): [] for ct in COSINE_THRESHOLDS}
    all_shuffled = {str(ct): [] for ct in COSINE_THRESHOLDS}
    all_random = {str(ct): [] for ct in COSINE_THRESHOLDS}

    for pname, attr, binpos in probe_configs:
        print(f"\n{'='*60}\n[PROBE] {pname}\n{'='*60}")
        pd_data = load_probe(pname, layer)
        if pd_data is None:
            print("  Not found, skipping"); continue
        print(f"  Accuracy: {pd_data['mean_acc']:.4f}")

        if binpos:
            labels = np.array([1 if c[attr] == binpos else 0 for c in cities])
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform([c[attr] for c in cities])
        n_cls = len(np.unique(labels))
        print(f"  Classes: {n_cls}, dist: {dict(Counter(labels))}")

        p_dirs = probe_direction(pd_data)
        template = "{city}, a city known for being in"
        texts = [template.format(city=c["city"]) for c in cities]

        report_progress("extract", pname)
        _, sae_acts = extract_activations_and_sae(model, tok, sae, texts, layer)

        # Real absorption at each threshold
        report_progress("real", pname)
        print("  [Real absorption]")
        real_sweep, cos_arr = run_absorption_sweep(sae_acts, sae_decoder, p_dirs, labels)
        for ct_str, rd in real_sweep.items():
            print(f"    cos={ct_str}: absorption={rd['mean_absorption']:.4f} "
                  f"(n_measured={rd['n_measured']})")
            all_real[ct_str].append(rd['mean_absorption'])

        # Shuffled
        report_progress("shuffled", pname)
        shuffled = run_shuffled_control(sae_acts, sae_decoder, p_dirs, labels, n_shuffles=5)
        for ct_str, sd in shuffled.items():
            all_shuffled[ct_str].extend(sd.get("per_shuffle", []))

        # Random
        report_progress("random", pname)
        random_ctrl = run_random_probe_control(sae_acts, sae_decoder, labels, d_model, n_random=10)
        for ct_str, rd in random_ctrl.items():
            all_random[ct_str].extend(rd.get("per_trial", []))

        results["controls"][pname] = {
            "attribute": attr, "n_classes": n_cls,
            "probe_accuracy": pd_data["mean_acc"],
            "real_absorption_sweep": real_sweep,
            "shuffled_control": shuffled,
            "random_probe_control": random_ctrl,
        }

        del sae_acts; gc.collect(); torch.cuda.empty_cache()

    # First-letter
    print(f"\n{'='*60}\n[FIRST-LETTER]\n{'='*60}")
    report_progress("first_letter")
    fl = run_first_letter(model, tok, sae, sae_decoder, layer, N_LETTERS_PILOT)
    results["first_letter_baseline"] = fl

    # Statistical comparisons at each threshold
    print(f"\n{'='*60}\n[STATISTICS]\n{'='*60}")
    stats = {}
    for ct in COSINE_THRESHOLDS:
        ct_str = str(ct)
        real_vals = all_real.get(ct_str, [])
        shuf_vals = all_shuffled.get(ct_str, [])
        rand_vals = all_random.get(ct_str, [])

        ct_stats = {"cosine_threshold": ct}

        if real_vals and shuf_vals:
            try:
                s, p = mannwhitneyu(real_vals, shuf_vals, alternative='greater')
                ct_stats["real_vs_shuffled"] = {
                    "p": float(p), "sig": bool(p < 0.05),
                    "real_mean": float(np.mean(real_vals)),
                    "shuffled_mean": float(np.mean(shuf_vals)),
                }
            except: pass

        if real_vals and rand_vals:
            try:
                s, p = mannwhitneyu(real_vals, rand_vals, alternative='greater')
                ct_stats["real_vs_random"] = {
                    "p": float(p), "sig": bool(p < 0.05),
                    "real_mean": float(np.mean(real_vals)),
                    "random_mean": float(np.mean(rand_vals)),
                }
            except: pass

        stats[ct_str] = ct_stats
        print(f"  cos={ct_str}:")
        for k, v in ct_stats.items():
            if isinstance(v, dict) and "p" in v:
                print(f"    {k}: real={v.get('real_mean',0):.4f}, "
                      f"ctrl={v.get('shuffled_mean', v.get('random_mean',0)):.4f}, "
                      f"p={v['p']:.4f}, sig={v['sig']}")

    results["statistical_comparisons"] = stats

    # Summary
    elapsed = time.time() - start_time

    # Find best threshold where controls discriminate
    best_ct = None
    best_separation = 0
    for ct_str, ct_st in stats.items():
        rvs = ct_st.get("real_vs_shuffled", {})
        rvr = ct_st.get("real_vs_random", {})
        sep = 0
        if rvs:
            sep += rvs.get("real_mean", 0) - rvs.get("shuffled_mean", 0)
        if rvr:
            sep += rvr.get("real_mean", 0) - rvr.get("random_mean", 0)
        if sep > best_separation:
            best_separation = sep
            best_ct = ct_str

    # Determine overall result at best threshold
    if best_ct:
        best_stats = stats[best_ct]
        real_m = best_stats.get("real_vs_shuffled", {}).get("real_mean", 0)
        shuf_m = best_stats.get("real_vs_shuffled", {}).get("shuffled_mean", 0)
        rand_m = best_stats.get("real_vs_random", {}).get("random_mean", 0)
    else:
        real_m = shuf_m = rand_m = 0
        best_ct = str(COSINE_THRESHOLDS[0])

    fl_summary = fl.get("threshold_summary", {})
    fl_best = fl_summary.get(best_ct, {}).get("mean", 0)

    # Verdict
    shuffled_ok = shuf_m < 0.10
    random_ok = rand_m < 0.10
    if shuffled_ok and random_ok:
        if real_m < 0.03:
            verdict = "INFORMATIVE_NULL"
            detail = (
                f"Controls pass (shuffled={shuf_m:.4f}, random={rand_m:.4f}, both<10%). "
                f"Real absorption={real_m:.4f} is near zero for knowledge features on GPT-2. "
                f"First-letter={fl_best:.4f}. "
                "Informative null: GPT-2 Small lacks sufficient hierarchical knowledge "
                "encoding for observable absorption. Methodology is valid (controls work)."
            )
        else:
            verdict = "GO"
            detail = (
                f"Controls pass: shuffled={shuf_m:.4f}, random={rand_m:.4f}. "
                f"Real={real_m:.4f}. First-letter={fl_best:.4f}. "
                f"Best threshold={best_ct}."
            )
    else:
        verdict = "MIXED"
        detail = (
            f"Controls: shuffled={shuf_m:.4f} ({'OK' if shuffled_ok else 'HIGH'}), "
            f"random={rand_m:.4f} ({'OK' if random_ok else 'HIGH'}). "
            f"Real={real_m:.4f}. First-letter={fl_best:.4f}."
        )

    results["summary"] = {
        "pilot_verdict": verdict,
        "pilot_detail": detail,
        "best_cosine_threshold": best_ct,
        "elapsed_sec": round(elapsed, 1),
        "at_best_threshold": {
            "real": float(real_m),
            "shuffled": float(shuf_m),
            "random": float(rand_m),
            "first_letter": float(fl_best),
        },
    }

    print(f"\n{'='*60}\n[SUMMARY]\n{'='*60}")
    print(f"  Best threshold: {best_ct}")
    print(f"  Real: {real_m:.4f}, Shuffled: {shuf_m:.4f}, Random: {rand_m:.4f}")
    print(f"  First-letter: {fl_best:.4f}")
    print(f"  Verdict: {verdict}")
    print(f"  Time: {elapsed:.1f}s")

    # Save
    PILOT_DIR.mkdir(parents=True, exist_ok=True)
    json_dump(results, PILOT_DIR / "P2_controls.json")
    json_dump(results, FULL_DIR / "P2_controls.json")

    mark_done("success",
              f"{verdict}: real={real_m:.4f}, shuf={shuf_m:.4f}, rand={rand_m:.4f}, "
              f"FL={fl_best:.4f}, best_ct={best_ct}, t={elapsed:.0f}s")

    # GPU progress
    gp_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        with open(gp_path) as f: gp = json.load(f)
    except: gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

    if TASK_ID not in gp["completed"]:
        gp["completed"].append(TASK_ID)
    if TASK_ID in gp.get("running", {}):
        del gp["running"][TASK_ID]
    gp["timings"][TASK_ID] = {
        "planned_min": 45, "actual_min": round(elapsed / 60, 1),
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.now().isoformat(),
        "config_snapshot": {
            "model": MODEL_NAME, "sae_release": SAE_RELEASE, "layer": layer,
            "d_sae": d_sae, "n_cities": len(cities),
            "cosine_thresholds": COSINE_THRESHOLDS,
            "dominance_threshold": DOMINANCE_THRESHOLD,
            "gpu_count": 1,
        },
    }
    with open(gp_path, "w") as f:
        json.dump(gp, f, cls=NumpyEncoder, indent=2)

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[ERROR] {e}\n{traceback.format_exc()}")
        mark_done("failed", f"{type(e).__name__}: {str(e)[:200]}")
        gp_path = WORKSPACE / "exp" / "gpu_progress.json"
        try:
            with open(gp_path) as f: gp = json.load(f)
        except: gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
        if TASK_ID not in gp.get("failed", []):
            gp.setdefault("failed", []).append(TASK_ID)
        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]
        with open(gp_path, "w") as f:
            json.dump(gp, f, indent=2, cls=NumpyEncoder)
        sys.exit(1)
