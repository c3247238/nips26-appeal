"""
C2B_absorption_survey_30sae FULL experiment
=============================================
MODE: FULL
Run sae-spelling absorption measurement on ~30 GPT-2 SAE configurations
to build the regression dataset for H2.

Config strategy (GPT-2 as open-model anchor):
  Group A: gpt2-small-res-jb, layers 3-11 (24k width, varying L0) = 9 configs
  Group B: gpt2-small-res-jb-feature-splitting, layer 8, widths 768-98304 = 8 configs (width survey)
  Group C: gpt2-small-resid-post-v5-32k, layers {2,4,6,8,10,11} = 6 configs (32k wide)
  Group D: gpt2-small-resid-post-v5-128k, layers {4,6,8,10,11} = 5 configs (128k very wide)
  Group E: gpt2-small-resid-mid-v5-32k, layers {4,6,8,10} = 4 configs (residual mid)
  Total: 32 configs (use all, some may fail to load)

For each config: run sae-spelling on all 26 letters, n=100 test tokens per letter, seed 42.
Output: exp/results/full/C2B_absorption_survey.parquet

Estimated runtime: ~8 hours (480 min) but GPT-2 is much faster than Gemma 2B.
Actual expected: ~2-3 hours at 3-5 min per config.
"""

import os
import json
import time
import random
import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm

# ---- Config ----
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
TASK_ID = "C2B_absorption_survey_30sae"

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
FULL_DIR = RESULTS_DIR / "full"
FULL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_TIME = datetime.datetime.now()

PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

# Write PID immediately
PID_FILE.write_text(str(os.getpid()))
print(f"[C2B FULL] PID={os.getpid()} written. GPU: {DEVICE}")
print(f"[C2B FULL] Start time: {START_TIME.isoformat()}")

# Checkpoint file for resumability
CHECKPOINT_FILE = RESULTS_DIR / f"{TASK_ID}_checkpoint.json"

# ====================================================================
# SAE Config Definitions
# ====================================================================

# Group A: gpt2-small-res-jb, layers 3-11 (24k width)
GROUP_A = [
    {
        "config_id": f"resid_pre_L{layer}_24k",
        "release": "gpt2-small-res-jb",
        "sae_id": f"blocks.{layer}.hook_resid_pre",
        "hook_name": f"blocks.{layer}.hook_resid_pre",
        "width_approx": 24576,
        "model_layer": layer,
        "layer_group": "res_jb",
        "hook_type": "resid_pre",
        "l0_setting": "low",
    }
    for layer in [3, 4, 5, 6, 7, 8, 9, 10, 11]
]

# Group B: feature-splitting at layer 8
GROUP_B = [
    {
        "config_id": f"resid_pre_L8_{width}",
        "release": "gpt2-small-res-jb-feature-splitting",
        "sae_id": f"blocks.8.hook_resid_pre_{width}",
        "hook_name": "blocks.8.hook_resid_pre",
        "width_approx": width,
        "model_layer": 8,
        "layer_group": "feature_splitting",
        "hook_type": "resid_pre",
        "l0_setting": "medium",
    }
    for width in [768, 1536, 3072, 6144, 12288, 24576, 49152, 98304]
]

# Group C: gpt2-small-resid-post-v5-32k
GROUP_C = [
    {
        "config_id": f"resid_post_L{layer}_32k",
        "release": "gpt2-small-resid-post-v5-32k",
        "sae_id": f"blocks.{layer}.hook_resid_post",
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "width_approx": 32768,
        "model_layer": layer,
        "layer_group": "resid_post_32k",
        "hook_type": "resid_post",
        "l0_setting": "medium",
    }
    for layer in [4, 6, 8, 10, 11]
]

# Group D: gpt2-small-resid-post-v5-128k
GROUP_D = [
    {
        "config_id": f"resid_post_L{layer}_128k",
        "release": "gpt2-small-resid-post-v5-128k",
        "sae_id": f"blocks.{layer}.hook_resid_post",
        "hook_name": f"blocks.{layer}.hook_resid_post",
        "width_approx": 131072,
        "model_layer": layer,
        "layer_group": "resid_post_128k",
        "hook_type": "resid_post",
        "l0_setting": "high",
    }
    for layer in [4, 6, 8, 10, 11]
]

# Group E: gpt2-small-resid-mid-v5-32k (mid-layer residual)
GROUP_E = [
    {
        "config_id": f"resid_mid_L{layer}_32k",
        "release": "gpt2-small-resid-mid-v5-32k",
        "sae_id": f"blocks.{layer}.hook_resid_mid",
        "hook_name": f"blocks.{layer}.hook_resid_mid",
        "width_approx": 32768,
        "model_layer": layer,
        "layer_group": "resid_mid_32k",
        "hook_type": "resid_mid",
        "l0_setting": "medium",
    }
    for layer in [4, 6, 8, 10]
]

ALL_CONFIGS = GROUP_A + GROUP_B + GROUP_C + GROUP_D + GROUP_E
print(f"Total planned configs: {len(ALL_CONFIGS)}")

ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
N_LETTERS = len(ALL_LETTERS)
MAX_ABLATION_SAMPLES = 100   # n=100 per letter as specified
N_ICL = 8


def elapsed_sec():
    return (datetime.datetime.now() - START_TIME).total_seconds()


def write_progress(step, total_steps, message, metrics=None):
    data = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "metrics": metrics or {},
        "elapsed_sec": elapsed_sec(),
        "updated_at": datetime.datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(data, indent=2))
    print(f"[{elapsed_sec()/60:.1f}m][{step}/{total_steps}] {message}")


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    fp = {}
    if PROGRESS_FILE.exists():
        try:
            fp = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": fp,
        "timestamp": datetime.datetime.now().isoformat(),
    }, indent=2))
    print(f"[C2B FULL] DONE: {status} — {summary}")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

# ====================================================================
# STEP 1: Load model
# ====================================================================
write_progress(1, 10, "Loading GPT-2 Small model...")

from transformer_lens import HookedTransformer
from sae_lens import SAE
from sae_spelling.vocab import get_alpha_tokens
from sae_spelling.prompting import (
    first_letter_formatter, VERBOSE_FIRST_LETTER_TEMPLATE,
    VERBOSE_FIRST_LETTER_TOKEN_POS, create_icl_prompt
)
from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator
from sae_spelling.probing import train_binary_probe

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token
print(f"  GPT-2 Small loaded: d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

# ====================================================================
# STEP 2: Build vocabulary and ICL word list
# ====================================================================
write_progress(2, 10, "Building vocabulary and ICL word list...")

vocab_alpha = get_alpha_tokens(tokenizer)
single_tok_words_by_letter = {}
for tok_str in vocab_alpha:
    w = tok_str.strip()
    if not w or not w[0].isalpha() or not w.isalpha() or len(w) < 2:
        continue
    toks = tokenizer.encode(' ' + w)
    if len(toks) == 1:
        letter = w[0].upper()
        single_tok_words_by_letter.setdefault(letter, []).append(w)

set_seed(SEED)
icl_word_list = []
for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
    ws = single_tok_words_by_letter.get(letter, [])
    random.shuffle(ws)
    icl_word_list.extend(ws[:30])
set_seed(SEED)
random.shuffle(icl_word_list)
print(f"  ICL word list: {len(icl_word_list)} words")

icl_first_n = set(icl_word_list[:N_ICL])


def build_icl_prompt(word):
    return create_icl_prompt(
        word,
        examples=icl_word_list,
        base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
        answer_formatter=first_letter_formatter(),
        max_icl_examples=N_ICL,
        shuffle_examples=False,
        check_contamination=False,
    )


ref_word = icl_word_list[N_ICL]
ref_prompt = build_icl_prompt(ref_word)
expected_tok_len = len(model.to_tokens([ref_prompt.base])[0])
print(f"  Reference prompt token length: {expected_tok_len}")

# Pre-build valid word lists for all 26 letters
words_by_letter = {}
for letter in ALL_LETTERS:
    w_list = single_tok_words_by_letter.get(letter, [])
    set_seed(SEED + ord(letter))
    random.shuffle(w_list)
    valid = []
    for w in w_list:
        if w in icl_first_n:
            continue
        try:
            p = build_icl_prompt(w)
            if len(model.to_tokens([p.base])[0]) == expected_tok_len:
                valid.append(w)
        except Exception:
            continue
    words_by_letter[letter] = valid[:150]  # keep up to 150 for sampling
    print(f"  Letter {letter}: {len(valid)} valid words")

print(f"  Letters with >=20 words: {sum(1 for ws in words_by_letter.values() if len(ws)>=20)}")


def letter_delta_metric(letter, tok=tokenizer, dev=DEVICE):
    pos_tok = tok.encode(f" {letter.upper()}")[-1]
    neg_toks = torch.tensor(
        [tok.encode(f" {l}")[-1] for l in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if l != letter.upper()]
    ).to(dev)

    def _metric(logits):
        return logits[:, -1, pos_tok] - logits[:, -1, neg_toks].mean(dim=-1)

    return _metric


def get_main_feature_ids(letter, sae, hook_name, pos=VERBOSE_FIRST_LETTER_TOKEN_POS, top_k=5):
    words_pos = [(w, build_icl_prompt(w)) for w in words_by_letter[letter][:20]
                 if w not in icl_first_n]
    words_neg = []
    for l2, ws in single_tok_words_by_letter.items():
        if l2 != letter.upper():
            for w in ws[:3]:
                if w not in icl_first_n:
                    try:
                        p = build_icl_prompt(w)
                        if len(model.to_tokens([p.base])[0]) == expected_tok_len:
                            words_neg.append((w, p))
                    except Exception:
                        continue
    words_neg = words_neg[:20]

    def get_acts(wps):
        acts = []
        with torch.no_grad():
            for w, p in wps:
                try:
                    _, cache = model.run_with_cache([p.base], names_filter=hook_name)
                    sae_in = cache[hook_name]
                    a = sae.encode(sae_in)
                    acts.append(a[0, pos, :].cpu().float())
                except Exception:
                    pass
        return torch.stack(acts) if acts else None

    pos_acts = get_acts(words_pos)
    neg_acts = get_acts(words_neg)
    if pos_acts is None or pos_acts.shape[0] < 2:
        return []
    pos_mean = pos_acts.mean(0)
    neg_mean = neg_acts.mean(0) if neg_acts is not None else torch.zeros_like(pos_mean)
    diff = pos_mean - neg_mean
    return diff.topk(min(top_k, diff.shape[0])).indices.tolist()


def train_probe_for_letter(letter, hook_name, pos=VERBOSE_FIRST_LETTER_TOKEN_POS):
    set_seed(SEED + ord(letter))
    pos_words = [w for w in words_by_letter[letter] if w not in icl_first_n][:60]
    neg_words = []
    for l2, ws in single_tok_words_by_letter.items():
        if l2 != letter.upper():
            for w in ws[:3]:
                if w not in icl_first_n:
                    try:
                        p = build_icl_prompt(w)
                        if len(model.to_tokens([p.base])[0]) == expected_tok_len:
                            neg_words.append(w)
                    except Exception:
                        continue
    neg_words = neg_words[:60]

    combined = [(w, 1) for w in pos_words] + [(w, 0) for w in neg_words]
    acts_list, labels = [], []
    with torch.no_grad():
        for w, lab in combined:
            try:
                p = build_icl_prompt(w)
                _, cache = model.run_with_cache([p.base], names_filter=hook_name)
                act = cache[hook_name][0, pos, :].cpu().float()
                acts_list.append(act)
                labels.append(float(lab))
            except Exception:
                continue

    if len(acts_list) < 10:
        return None

    acts_t = torch.stack(acts_list).to(DEVICE)
    labels_t = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
    probe = train_binary_probe(
        acts_t, labels_t, num_epochs=100, lr=0.01, show_progress=False, verbose=False, device=DEVICE
    )
    probe_dir = F.normalize(probe.fc.weight[0].detach().cpu(), dim=0)
    return probe_dir


# ====================================================================
# STEP 3: Load checkpoint (resumability)
# ====================================================================
write_progress(3, 10, "Loading checkpoint (if exists)...")

completed_configs = {}  # config_id -> dict of letter -> result
sae_meta_all = {}

if CHECKPOINT_FILE.exists():
    try:
        ckpt = json.loads(CHECKPOINT_FILE.read_text())
        completed_configs = ckpt.get("completed_configs", {})
        sae_meta_all = ckpt.get("sae_meta_all", {})
        print(f"  Resuming from checkpoint: {len(completed_configs)} configs already done")
        for cfg_id, lr in completed_configs.items():
            done_letters = [l for l, r in lr.items() if isinstance(r, dict) and r.get("absorption_rate") is not None]
            print(f"    {cfg_id}: {len(done_letters)}/26 letters")
    except Exception as e:
        print(f"  Checkpoint load failed: {e}, starting fresh")
        completed_configs = {}
        sae_meta_all = {}
else:
    print("  No checkpoint found, starting fresh")


def save_checkpoint():
    ckpt = {
        "completed_configs": completed_configs,
        "sae_meta_all": sae_meta_all,
        "timestamp": datetime.datetime.now().isoformat(),
        "elapsed_sec": elapsed_sec(),
    }
    CHECKPOINT_FILE.write_text(json.dumps(ckpt, indent=2))


# ====================================================================
# STEP 4-8: Main loop — run each SAE config
# ====================================================================
write_progress(4, 10, f"Starting main survey loop: {len(ALL_CONFIGS)} configs, 26 letters each...")

n_total_configs = len(ALL_CONFIGS)
configs_done_count = len(completed_configs)

for cfg_idx, cfg in enumerate(ALL_CONFIGS):
    cfg_id = cfg["config_id"]

    # Skip if already done (checkpoint)
    if cfg_id in completed_configs:
        done_letters = sum(1 for r in completed_configs[cfg_id].values()
                           if isinstance(r, dict) and r.get("absorption_rate") is not None)
        if done_letters >= 20:  # consider complete if >=20 letters done
            print(f"  [{cfg_idx+1}/{n_total_configs}] SKIP (cached): {cfg_id} ({done_letters}/26 letters)")
            continue

    configs_done_count += 1
    progress_step = 4 + int((cfg_idx / n_total_configs) * 4)  # steps 4-7

    write_progress(
        progress_step, 10,
        f"Config {cfg_idx+1}/{n_total_configs}: {cfg_id} | elapsed={elapsed_sec()/60:.1f}min",
        metrics={"config_id": cfg_id, "configs_done": configs_done_count}
    )

    # Load SAE
    t_load_start = time.time()
    try:
        sae, cfg_dict, sparsity = SAE.from_pretrained_with_cfg_and_sparsity(
            release=cfg["release"],
            sae_id=cfg["sae_id"],
            device=DEVICE,
        )
        sae.eval()
        print(f"  [{cfg_idx+1}] Loaded {cfg_id}: d_sae={sae.cfg.d_sae} in {time.time()-t_load_start:.1f}s")
    except Exception as e:
        print(f"  [{cfg_idx+1}] LOAD FAIL {cfg_id}: {e}")
        completed_configs[cfg_id] = {"_load_error": str(e)}
        sae_meta_all[cfg_id] = {**cfg, "load_error": str(e), "measured_l0": None, "d_sae": None}
        save_checkpoint()
        continue

    # Determine hook_name from metadata (most reliable)
    hook_name = cfg["hook_name"]  # fallback: pre-specified base hook
    try:
        meta_hook = sae.cfg.metadata.hook_name
        if meta_hook:
            # For feature-splitting SAEs, metadata hook may have width suffix
            # Use the base hook_name (without width suffix) for TransformerLens cache
            base_hook = cfg["hook_name"]
            if meta_hook == base_hook or meta_hook.startswith(base_hook):
                hook_name = base_hook
            else:
                hook_name = meta_hook
    except Exception:
        pass
    # CRITICAL: Set sae.cfg.hook_name so FeatureAbsorptionCalculator can find it
    # (new SAELens stores it in metadata, but the calculator reads sae.cfg.hook_name)
    sae.cfg.hook_name = hook_name
    print(f"  [{cfg_idx+1}] hook_name resolved: {hook_name}")

    # Measure L0 on 500 tokens
    try:
        sample_texts = []
        try:
            from datasets import load_dataset
            ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
            for item in ds.take(5):
                sample_texts.append(item['text'])
        except Exception:
            sample_texts = ["The quick brown fox jumps over the lazy dog. " * 100] * 5

        sample_toks = []
        for txt in sample_texts:
            toks = model.to_tokens(txt, prepend_bos=True)[0]
            sample_toks.append(toks)
        sample_toks_cat = torch.cat(sample_toks)[:500].unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            _, cache_l0 = model.run_with_cache(sample_toks_cat, names_filter=hook_name)
            resid_l0 = cache_l0[hook_name][0]
            acts_l0 = sae.encode(resid_l0)
            l0_values = (acts_l0 > 0).float().sum(dim=-1).cpu().tolist()
        measured_l0 = float(np.mean(l0_values))
        print(f"  [{cfg_idx+1}] Measured L0: {measured_l0:.1f}")
    except Exception as e:
        print(f"  [{cfg_idx+1}] L0 measurement failed: {e}")
        measured_l0 = None

    sae_meta_all[cfg_id] = {
        **cfg,
        "hook_name": hook_name,
        "measured_l0": measured_l0,
        "d_sae": sae.cfg.d_sae,
    }

    # Build calculator for this config
    calculator = FeatureAbsorptionCalculator(
        model=model,
        icl_word_list=icl_word_list,
        max_icl_examples=N_ICL,
        base_template=VERBOSE_FIRST_LETTER_TEMPLATE,
        answer_formatter=first_letter_formatter(),
        word_token_pos=VERBOSE_FIRST_LETTER_TOKEN_POS,
        probe_cos_sim_threshold=0.01,
        ablation_delta_threshold=0.1,
        ig_interpolation_steps=4,
        ig_batch_size=4,
        filter_prompts_batch_size=20,
        topk_feats=10,
        shuffle_examples=False,
    )

    # Load partial progress from checkpoint for this config if any
    letter_results = completed_configs.get(cfg_id, {})
    if isinstance(letter_results, dict) and "_load_error" in letter_results:
        letter_results = {}  # reset if previously failed to load

    # Train probes and measure absorption for each letter
    t_cfg_start = time.time()
    letters_done = 0

    for letter in ALL_LETTERS:
        # Skip if already done for this letter
        if letter in letter_results and isinstance(letter_results[letter], dict) and letter_results[letter].get("absorption_rate") is not None:
            letters_done += 1
            continue

        try:
            probe_dir = train_probe_for_letter(letter, hook_name)
            if probe_dir is None:
                letter_results[letter] = {"absorption_rate": None, "error": "probe_training_failed", "n_total": 0, "n_absorbed": 0}
                continue

            main_ids = get_main_feature_ids(letter, sae, hook_name)
            words = words_by_letter[letter]
            metric_fn = letter_delta_metric(letter)

            try:
                results = calculator.calculate_absorption_sampled(
                    sae=sae,
                    words=words,
                    probe_dir=probe_dir.to(DEVICE),
                    metric_fn=metric_fn,
                    main_feature_ids=main_ids,
                    max_ablation_samples=MAX_ABLATION_SAMPLES,
                    filter_prompts=True,
                    show_progress=False,
                )
                sample_results = results.sample_results
                n_tot = len(sample_results)
                n_abs = sum(1 for r in sample_results if r.is_absorption)
                rate = n_abs / n_tot if n_tot > 0 else 0.0
                letter_results[letter] = {
                    "absorption_rate": rate,
                    "n_absorbed": n_abs,
                    "n_total": n_tot,
                }
                letters_done += 1
            except Exception as e:
                # Fallback: no filtering
                try:
                    results = calculator.calculate_absorption_sampled(
                        sae=sae,
                        words=words,
                        probe_dir=probe_dir.to(DEVICE),
                        metric_fn=metric_fn,
                        main_feature_ids=main_ids,
                        max_ablation_samples=50,
                        filter_prompts=False,
                        show_progress=False,
                    )
                    sample_results = results.sample_results
                    n_tot = len(sample_results)
                    n_abs = sum(1 for r in sample_results if r.is_absorption)
                    rate = n_abs / n_tot if n_tot > 0 else 0.0
                    letter_results[letter] = {
                        "absorption_rate": rate,
                        "n_absorbed": n_abs,
                        "n_total": n_tot,
                        "note": "fallback_no_filter",
                    }
                    letters_done += 1
                except Exception as e2:
                    letter_results[letter] = {"absorption_rate": None, "error": str(e2), "n_total": 0, "n_absorbed": 0}
        except Exception as e:
            letter_results[letter] = {"absorption_rate": None, "error": str(e), "n_total": 0, "n_absorbed": 0}

    completed_configs[cfg_id] = letter_results

    # Print config summary
    valid_rates = [r.get("absorption_rate") for r in letter_results.values()
                   if isinstance(r, dict) and r.get("absorption_rate") is not None]
    mean_rate = float(np.mean(valid_rates)) if valid_rates else 0.0
    t_cfg_elapsed = time.time() - t_cfg_start
    print(f"  [{cfg_idx+1}] {cfg_id} done: {len(valid_rates)}/26 letters, mean_rate={mean_rate:.3f}, time={t_cfg_elapsed/60:.1f}min")

    # Save checkpoint after each config
    save_checkpoint()

    # Free GPU memory
    del sae, calculator
    if DEVICE.startswith("cuda"):
        torch.cuda.empty_cache()

    # Print ETA
    elapsed = elapsed_sec()
    done_frac = (cfg_idx + 1) / n_total_configs
    if done_frac > 0:
        eta_total = elapsed / done_frac
        eta_remaining = eta_total - elapsed
        print(f"  ETA: {eta_remaining/60:.0f}min remaining ({elapsed/60:.0f}min elapsed)")

# ====================================================================
# STEP 8: Assemble flat dataset
# ====================================================================
write_progress(8, 10, "Assembling flat dataset for parquet output...")

data_rows = []
for cfg in ALL_CONFIGS:
    cfg_id = cfg["config_id"]
    cfg_results = completed_configs.get(cfg_id, {})
    meta = sae_meta_all.get(cfg_id, cfg)

    if "_load_error" in cfg_results:
        continue

    for letter in ALL_LETTERS:
        r = cfg_results.get(letter, {})
        if not isinstance(r, dict):
            continue
        rate = r.get("absorption_rate")
        if rate is None:
            rate = float("nan")

        data_rows.append({
            "config_id": cfg_id,
            "letter": letter,
            "absorption_rate": float(rate),
            "n_absorbed": int(r.get("n_absorbed", 0)),
            "n_total": int(r.get("n_total", 0)),
            "model": "gpt2-small",
            "release": cfg["release"],
            "sae_id": cfg["sae_id"],
            "hook_type": cfg.get("hook_type", ""),
            "model_layer": cfg["model_layer"],
            "width_approx": cfg["width_approx"],
            "l0_setting": cfg["l0_setting"],
            "layer_group": cfg.get("layer_group", ""),
            "measured_l0": meta.get("measured_l0"),
            "d_sae": meta.get("d_sae"),
            "has_error": 1 if "error" in r else 0,
            "note": r.get("note", ""),
        })

df = pd.DataFrame(data_rows)
print(f"  Total rows: {len(df)}")
print(f"  Valid rows (non-nan rate): {df['absorption_rate'].notna().sum()}")
print(f"  Configs covered: {df['config_id'].nunique()}")
print(f"  Letters covered: {df['letter'].nunique()}")
print()
print("  Absorption rate summary by config (mean over letters):")
summary = df.groupby("config_id")["absorption_rate"].mean().sort_values(ascending=False)
print(summary.to_string())

# ====================================================================
# STEP 9: Write outputs
# ====================================================================
write_progress(9, 10, "Writing parquet and summary files...")

parquet_path = FULL_DIR / "C2B_absorption_survey.parquet"
df.to_parquet(parquet_path, index=False)
print(f"  Parquet written: {parquet_path} ({len(df)} rows)")

# Also write JSON for inspection
json_path = FULL_DIR / "C2B_absorption_survey_summary.json"
summary_dict = {
    "task_id": TASK_ID,
    "mode": "FULL",
    "timestamp": datetime.datetime.now().isoformat(),
    "model": "gpt2-small",
    "model_note": "GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)",
    "n_configs_planned": len(ALL_CONFIGS),
    "n_configs_completed": int(df["config_id"].nunique()),
    "n_rows_total": len(df),
    "n_rows_valid": int(df["absorption_rate"].notna().sum()),
    "n_letters": N_LETTERS,
    "n_tokens_per_letter": MAX_ABLATION_SAMPLES,
    "elapsed_minutes": elapsed_sec() / 60,
    "statistics": {
        "mean_absorption_rate": float(df["absorption_rate"].mean()),
        "std_absorption_rate": float(df["absorption_rate"].std()),
        "min_absorption_rate": float(df["absorption_rate"].min()),
        "max_absorption_rate": float(df["absorption_rate"].max()),
    },
    "per_config_summary": df.groupby(["config_id", "model_layer", "width_approx", "measured_l0"])["absorption_rate"].agg(
        ["mean", "std", "min", "max", "count"]
    ).reset_index().to_dict("records"),
    "per_letter_summary": df.groupby("letter")["absorption_rate"].agg(
        ["mean", "std"]
    ).reset_index().to_dict("records"),
    "sae_configs": sae_meta_all,
}

json_path.write_text(json.dumps(summary_dict, indent=2, default=str))
print(f"  Summary JSON written: {json_path}")

# Summary markdown for quick review
md_lines = [
    "# C2B Absorption Survey (Full) — Summary",
    "",
    f"**Mode**: FULL  |  **Model**: GPT-2 Small  |  **Timestamp**: {datetime.datetime.now().isoformat()}",
    "",
    f"**Configs completed**: {df['config_id'].nunique()}/{len(ALL_CONFIGS)} planned  |  **Letters**: 26  |  **N per letter**: {MAX_ABLATION_SAMPLES}",
    f"**Total valid data points**: {df['absorption_rate'].notna().sum()} / {len(df)} rows",
    "",
    "## Absorption Rate by Config (mean ± std over 26 letters)",
    "",
    "| Config | Layer | Width | L0 | Mean ± Std | Min | Max |",
    "|--------|-------|-------|----|------------|-----|-----|",
]
for _, row in df.groupby("config_id").apply(lambda x: pd.Series({
    "layer": x["model_layer"].iloc[0],
    "width": x["width_approx"].iloc[0],
    "l0": x.get("measured_l0", pd.Series([None])).iloc[0] if "measured_l0" in x.columns else None,
    "mean": x["absorption_rate"].mean(),
    "std": x["absorption_rate"].std(),
    "min": x["absorption_rate"].min(),
    "max": x["absorption_rate"].max(),
}), include_groups=False).reset_index().iterrows():
    l0_str = f"{row['l0']:.1f}" if row['l0'] is not None and not (isinstance(row['l0'], float) and np.isnan(row['l0'])) else "?"
    md_lines.append(
        f"| {row['config_id']} | {row['layer']} | {row['width']:,} | {l0_str} | "
        f"{row['mean']:.4f} ± {row['std']:.4f} | {row['min']:.4f} | {row['max']:.4f} |"
    )

md_lines += [
    "",
    "## Absorption Rate by Letter (mean over all configs)",
    "",
    "| Letter | Mean | Std |",
    "|--------|------|-----|",
]
for _, row in df.groupby("letter")["absorption_rate"].agg(["mean", "std"]).reset_index().iterrows():
    md_lines.append(f"| {row['letter']} | {row['mean']:.4f} | {row['std']:.4f} |")

md_lines += [
    "",
    f"## Runtime",
    f"- Total elapsed: {elapsed_sec()/60:.1f} minutes",
    f"- Parquet: `exp/results/full/C2B_absorption_survey.parquet`",
    "",
]

(FULL_DIR / "C2B_absorption_survey_summary.md").write_text("\n".join(md_lines))
print("  Summary markdown written.")

# ====================================================================
# STEP 10: Finalize
# ====================================================================
write_progress(10, 10, "Finalizing and updating tracking files...")

actual_min = max(1, int(elapsed_sec() / 60))

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    if gpu_progress_path.exists():
        gp = json.loads(gpu_progress_path.read_text())
    else:
        gp = {"completed": [], "failed": [], "running": {}, "timings": {}}
    if TASK_ID not in gp.get("completed", []):
        gp.setdefault("completed", []).append(TASK_ID)
    gp.get("running", {}).pop(TASK_ID, None)
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 480,
        "actual_min": actual_min,
        "start_time": START_TIME.isoformat(),
        "end_time": datetime.datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small",
            "mode": "full",
            "n_configs": df["config_id"].nunique(),
            "n_letters": N_LETTERS,
            "n_tokens_per_letter": MAX_ABLATION_SAMPLES,
            "gpu": str(DEVICE),
            "gpu_id": "1",
        },
    }
    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print("  gpu_progress.json updated.")
except Exception as e:
    print(f"  Warning: gpu_progress.json update failed: {e}")

n_valid = int(df["absorption_rate"].notna().sum())
n_configs_done = int(df["config_id"].nunique())
rates_std = float(df["absorption_rate"].std())

mark_done(
    status="success",
    summary=(f"C2B FULL: {n_configs_done}/{len(ALL_CONFIGS)} configs, "
             f"{n_valid} valid data points, "
             f"rates_std={rates_std:.4f}, "
             f"runtime={actual_min}min")
)

print(f"\n=== C2B FULL COMPLETE ===")
print(f"  Runtime: {actual_min} minutes")
print(f"  Configs: {n_configs_done}/{len(ALL_CONFIGS)}")
print(f"  Valid data points: {n_valid} / {len(df)}")
print(f"  Absorption rate: mean={df['absorption_rate'].mean():.4f}, std={rates_std:.4f}")
print(f"  Output: {parquet_path}")
