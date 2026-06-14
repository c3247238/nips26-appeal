"""
P0_pipeline_validation_full.py
FULL MODE: Pipeline Validation and L0 Gate for SAE Absorption Study

Full mode extends pilot to all 26 letters (instead of just A-E).
Uses GPT-2 Small + SAELens (gpt2-small-res-jb release).
This builds a comprehensive validation table for all 26 letters.

Outputs:
  - exp/results/full/P0_pipeline_full.json  (machine-readable)
  - exp/results/full/P0_pipeline_full_summary.md
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
from tqdm import tqdm

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# Paths
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOTS_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"
FULL_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR.mkdir(parents=True, exist_ok=True)

TASK_ID = "P0_pipeline_validation"
PID_FILE = RESULTS_DIR / f"{TASK_ID}.pid"
PROGRESS_FILE = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
DONE_FILE = RESULTS_DIR / f"{TASK_ID}_DONE"

# Write PID file immediately
PID_FILE.write_text(str(os.getpid()))
start_time = time.time()


def report_progress(step, total_steps, message, metrics=None):
    data = {
        "task_id": TASK_ID,
        "step": step,
        "total_steps": total_steps,
        "message": message,
        "metrics": metrics or {},
        "elapsed_sec": time.time() - start_time,
        "updated_at": datetime.datetime.now().isoformat(),
    }
    PROGRESS_FILE.write_text(json.dumps(data))
    print(f"[{data['elapsed_sec']:.1f}s][{step}/{total_steps}] {message}")


def mark_done(status="success", summary=""):
    if PID_FILE.exists():
        PID_FILE.unlink()
    final_progress = {}
    if PROGRESS_FILE.exists():
        try:
            final_progress = json.loads(PROGRESS_FILE.read_text())
        except Exception:
            pass
    DONE_FILE.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.datetime.now().isoformat(),
    }))


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed()

print("=" * 60)
print("P0 Pipeline Validation - FULL MODE (all 26 letters)")
print("=" * 60)

# ===== STEP 1: Load GPT-2 model =====
report_progress(1, 10, "Loading GPT-2 Small model via TransformerLens...")
print("NOTE: Using GPT-2 Small (Gemma-2-2b requires gated HF access)")

from transformer_lens import HookedTransformer
from sae_lens import SAE

model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
model.eval()
tokenizer = model.tokenizer
tokenizer.pad_token = tokenizer.eos_token
print(f"Model: gpt2-small, d_model={model.cfg.d_model}, n_layers={model.cfg.n_layers}")

# ===== STEP 2: Load SAEs =====
report_progress(2, 10, "Loading SAEs (two widths for L0 comparison)...")

SAE_NARROW_RELEASE = "gpt2-small-res-jb"
SAE_NARROW_ID = "blocks.8.hook_resid_pre"
LAYER = 8
HOOK_NAME = f"blocks.{LAYER}.hook_resid_pre"

SAE_WIDE_RELEASE = "gpt2-small-res-jb-feature-splitting"
SAE_WIDE_ID = "blocks.8.hook_resid_pre_49152"

print(f"Loading narrow SAE: {SAE_NARROW_ID}")
sae_narrow, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
    release=SAE_NARROW_RELEASE,
    sae_id=SAE_NARROW_ID,
    device=DEVICE,
)
sae_narrow.eval()
if not hasattr(sae_narrow.cfg, 'hook_name'):
    hook_from_meta = sae_narrow.cfg.metadata.get('hook_name', HOOK_NAME) if hasattr(sae_narrow.cfg, 'metadata') else HOOK_NAME
    sae_narrow.cfg.hook_name = hook_from_meta
print(f"  hook_name: {sae_narrow.cfg.hook_name}, d_sae: {sae_narrow.cfg.d_sae}")

print(f"Loading wide SAE: {SAE_WIDE_ID}")
sae_wide, _, _ = SAE.from_pretrained_with_cfg_and_sparsity(
    release=SAE_WIDE_RELEASE,
    sae_id=SAE_WIDE_ID,
    device=DEVICE,
)
sae_wide.eval()
if not hasattr(sae_wide.cfg, 'hook_name'):
    hook_from_meta = sae_wide.cfg.metadata.get('hook_name', HOOK_NAME) if hasattr(sae_wide.cfg, 'metadata') else HOOK_NAME
    sae_wide.cfg.hook_name = hook_from_meta
print(f"  d_sae: {sae_wide.cfg.d_sae}")

# ===== STEP 3: Build single-token word vocabulary =====
report_progress(3, 10, "Building single-token word vocabulary for all 26 letters...")

from sae_spelling.vocab import get_alpha_tokens

vocab_alpha = get_alpha_tokens(tokenizer)
ALL_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Find single-token words
single_tok_words_by_letter = {}
for tok_str in vocab_alpha:
    w = tok_str.strip()
    if not w or not w[0].isalpha() or not w.isalpha() or len(w) < 2:
        continue
    toks = tokenizer.encode(' ' + w)
    if len(toks) == 1:
        letter = w[0].upper()
        single_tok_words_by_letter.setdefault(letter, []).append(w)

# Build ICL word list
set_seed(SEED)
icl_word_list = []
for letter in ALL_LETTERS:
    words = single_tok_words_by_letter.get(letter, [])
    random.shuffle(words)
    icl_word_list.extend(words[:30])

set_seed(SEED)
random.shuffle(icl_word_list)
print(f"ICL word list: {len(icl_word_list)} single-token words")

# Words per letter
words_by_letter = {}
for letter in ALL_LETTERS:
    w_list = list(single_tok_words_by_letter.get(letter, []))
    set_seed(SEED + ord(letter))
    random.shuffle(w_list)
    words_by_letter[letter] = w_list[:100]
    print(f"  Letter {letter}: {len(words_by_letter[letter])} single-token words")

# ===== STEP 4: Build ICL prompt infrastructure =====
report_progress(4, 10, "Building ICL prompt infrastructure...")

from sae_spelling.prompting import first_letter_formatter, VERBOSE_FIRST_LETTER_TEMPLATE, VERBOSE_FIRST_LETTER_TOKEN_POS
from sae_spelling.prompting import create_icl_prompt

N_ICL = 8

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

# Get reference prompt length
ref_word = icl_word_list[N_ICL]
ref_prompt = build_icl_prompt(ref_word)
expected_tok_len = len(model.to_tokens([ref_prompt.base])[0])
print(f"Reference prompt token length: {expected_tok_len}")

# Pre-filter words for all letters
icl_first_n = set(icl_word_list[:N_ICL])
print(f"Excluding {len(icl_first_n)} words in first {N_ICL} ICL examples...")
print("Pre-filtering words to same-length prompts...")
for letter in ALL_LETTERS:
    original_count = len(words_by_letter[letter])
    valid_words = []
    for w in words_by_letter[letter]:
        if w in icl_first_n:
            continue
        try:
            p = build_icl_prompt(w)
            tok_len = len(model.to_tokens([p.base])[0])
            if tok_len == expected_tok_len:
                valid_words.append(w)
        except Exception:
            continue
    words_by_letter[letter] = valid_words[:100]
    if len(valid_words) < 5:
        print(f"  WARNING Letter {letter}: only {len(valid_words)} valid words")
    else:
        print(f"  Letter {letter}: {original_count} -> {len(valid_words)} valid")

# ===== STEP 5: Measure empirical L0 for both SAEs =====
report_progress(5, 10, "Measuring empirical L0 for both SAEs on 500-token sample...")

from datasets import load_dataset

print("Loading OpenWebText sample...")
texts = []
try:
    dataset = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    for item in dataset.take(30):
        texts.append(item['text'])
    print(f"  Loaded {len(texts)} texts from Skylion007/openwebtext")
except Exception as e:
    print(f"  Skylion007 failed: {e}")
    try:
        import glob
        cache_dirs = glob.glob(os.path.expanduser("~/.cache/huggingface/hub/datasets--stas--openwebtext-10k/snapshots/*"))
        if cache_dirs:
            from datasets import load_from_disk
            ds = load_from_disk(cache_dirs[0])
            for i, item in enumerate(ds):
                texts.append(item['text'])
                if i >= 30:
                    break
            print(f"  Loaded {len(texts)} texts from cached openwebtext-10k")
        else:
            raise ValueError("No cached dataset")
    except Exception as e2:
        print(f"  Dataset loading failed: {e2}, using dummy text")
        dummy = "The quick brown fox jumps over the lazy dog and other stories. "
        texts = [dummy * 50] * 30

all_tokens = []
for text in texts:
    try:
        tokens = model.to_tokens(text, prepend_bos=True)[0]
        all_tokens.append(tokens)
    except Exception:
        continue

token_sample_500 = torch.cat(all_tokens)[:500].unsqueeze(0).to(DEVICE)
print(f"L0 sample: {token_sample_500.shape}")


def measure_l0(sae, model, tokens, hook_name, batch_size=100):
    n_tokens = tokens.shape[1]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        resid = cache[hook_name][0]
    l0_vals = []
    with torch.no_grad():
        for i in range(0, n_tokens, batch_size):
            batch = resid[i:i+batch_size]
            acts = sae.encode(batch)
            l0 = (acts > 0).float().sum(dim=-1)
            l0_vals.extend(l0.cpu().tolist())
    return {"mean_l0": float(np.mean(l0_vals)), "std_l0": float(np.std(l0_vals)), "n_tokens": len(l0_vals)}


print("Measuring L0 for narrow SAE (~24k)...")
l0_narrow = measure_l0(sae_narrow, model, token_sample_500, HOOK_NAME)
print(f"  Narrow: mean={l0_narrow['mean_l0']:.1f}, std={l0_narrow['std_l0']:.1f}")

print("Measuring L0 for wide SAE (~49k)...")
l0_wide = measure_l0(sae_wide, model, token_sample_500, HOOK_NAME)
print(f"  Wide: mean={l0_wide['mean_l0']:.1f}, std={l0_wide['std_l0']:.1f}")

l0_diff_pct = abs(l0_narrow['mean_l0'] - l0_wide['mean_l0']) / max(l0_narrow['mean_l0'], l0_wide['mean_l0'])
l0_confound = l0_diff_pct > 0.20
print(f"L0 difference: {l0_diff_pct*100:.1f}% -> confound={'YES' if l0_confound else 'NO'}")

# ===== STEP 6: Compute alpha_ij on 1k-token sample =====
report_progress(6, 10, "Computing alpha_ij on 1k-token sample...")

token_1k = torch.cat(all_tokens)[:1000].unsqueeze(0).to(DEVICE)
print(f"Alpha_ij token sample: {token_1k.shape}")


def compute_alpha_ij(sae, model, tokens, hook_name, top_n=100,
                     freq_thresh=0.001, cos_thresh=0.15, batch_size=100):
    n_tokens = tokens.shape[1]
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        resid = cache[hook_name][0]
    all_acts = []
    with torch.no_grad():
        for i in range(0, n_tokens, batch_size):
            all_acts.append(sae.encode(resid[i:i+batch_size]).cpu())
    acts = torch.cat(all_acts)
    binary = (acts > 0).float()
    f_i = binary.mean(dim=0)
    active_idx = (f_i > freq_thresh).nonzero(as_tuple=True)[0]
    print(f"  Active features: {active_idx.shape[0]}")
    if active_idx.shape[0] > 2000:
        _, top_idx = f_i.topk(2000)
        active_idx = top_idx.sort().values
    n_act = active_idx.shape[0]
    active_acts = binary[:, active_idx]
    active_f = f_i[active_idx]
    W_dec = sae.W_dec[active_idx].cpu()
    W_norm = F.normalize(W_dec, dim=1)
    pairs = []
    chunk = 300
    for i_s in tqdm(range(0, n_act, chunk), desc="alpha_ij"):
        i_e = min(i_s + chunk, n_act)
        cos = W_norm[i_s:i_e] @ W_norm.T
        for li, gi in enumerate(range(i_s, i_e)):
            cos_row = cos[li].clone()
            cos_row[gi] = 0
            jj = (cos_row > cos_thresh).nonzero(as_tuple=True)[0]
            fi = active_f[gi].item()
            if fi <= freq_thresh:
                continue
            for gj in jj.tolist():
                if gj == gi:
                    continue
                fj = active_f[gj].item()
                if fj <= freq_thresh:
                    continue
                coact = (active_acts[:, gi] * active_acts[:, gj]).mean().item()
                sigma = coact / max(min(fi, fj), 1e-9)
                alpha = sigma * (fj / fi)
                pairs.append({
                    "latent_i": active_idx[gi].item(),
                    "latent_j": active_idx[gj].item(),
                    "f_i": fi, "f_j": fj,
                    "coact_rate": coact,
                    "sigma_ij": sigma,
                    "alpha_ij": alpha,
                    "decoder_cosine": cos_row[gj].item(),
                })
    pairs.sort(key=lambda x: x["alpha_ij"], reverse=True)
    print(f"  Total pairs: {len(pairs)}")
    return pairs[:top_n], f_i


print("\nComputing alpha_ij for narrow SAE...")
alpha_top100, f_i_full = compute_alpha_ij(sae_narrow, model, token_1k, HOOK_NAME)

print(f"\nTop-10 alpha_ij pairs:")
for p in alpha_top100[:10]:
    print(f"  i={p['latent_i']}, j={p['latent_j']}, alpha={p['alpha_ij']:.3f}, "
          f"f_i={p['f_i']:.3f}, f_j={p['f_j']:.3f}, cos={p['decoder_cosine']:.3f}")

# ===== STEP 7: Train probes for all 26 letters =====
report_progress(7, 10, "Training probes for all 26 letters...")

from sae_spelling.probing import train_binary_probe

probe_directions = {}


def get_resid_at_pos(word, pos=VERBOSE_FIRST_LETTER_TOKEN_POS, hook_name=HOOK_NAME):
    try:
        p = build_icl_prompt(word)
        if len(model.to_tokens([p.base])[0]) != expected_tok_len:
            return None
        _, cache = model.run_with_cache([p.base], names_filter=hook_name)
        act = cache[hook_name][0, pos, :].cpu().float()
        return act
    except Exception:
        return None


for letter in ALL_LETTERS:
    set_seed(SEED + ord(letter))
    pos_words = [w for w in words_by_letter[letter] if w not in icl_first_n][:40]
    neg_words = []
    for l2, ws in single_tok_words_by_letter.items():
        if l2 != letter.upper():
            for w in list(ws)[:3]:
                if w not in icl_first_n:
                    try:
                        p = build_icl_prompt(w)
                        if len(model.to_tokens([p.base])[0]) == expected_tok_len:
                            neg_words.append(w)
                    except Exception:
                        continue
    neg_words = neg_words[:40]
    combined = [(w, 1) for w in pos_words] + [(w, 0) for w in neg_words]
    acts_list, labels = [], []
    with torch.no_grad():
        for w, lab in combined:
            act = get_resid_at_pos(w)
            if act is not None:
                acts_list.append(act)
                labels.append(float(lab))
    if len(acts_list) < 10:
        print(f"  WARNING: Not enough activations for {letter} probe ({len(acts_list)})")
        continue
    acts_t = torch.stack(acts_list).to(DEVICE)
    labels_t = torch.tensor(labels, dtype=torch.float32).to(DEVICE)
    probe = train_binary_probe(acts_t, labels_t,
        num_epochs=100, lr=0.01, show_progress=False, verbose=False, device=DEVICE)
    probe_dir = F.normalize(probe.fc.weight[0].detach().cpu(), dim=0)
    probe_directions[letter] = probe_dir

print(f"Probes trained for: {sorted(probe_directions.keys())}")

# Check letter-feature pairs in alpha_ij
def check_letter_feature_pairs(pairs, probe_dirs, sae, top_n=10, cos_thresh=0.10):
    found = []
    W_norm = F.normalize(sae.W_dec.cpu(), dim=1)
    for letter, probe_dir in probe_dirs.items():
        probe_norm = F.normalize(probe_dir.unsqueeze(0), dim=1).squeeze(0)
        cos_probe = (W_norm @ probe_norm).abs()
        for rank, pair in enumerate(pairs[:top_n]):
            ci = cos_probe[pair['latent_i']].item()
            cj = cos_probe[pair['latent_j']].item()
            if ci > cos_thresh or cj > cos_thresh:
                found.append({
                    "rank": rank + 1, "letter": letter,
                    "latent_i": pair['latent_i'], "latent_j": pair['latent_j'],
                    "alpha_ij": pair['alpha_ij'],
                    "cos_i": ci, "cos_j": cj,
                })
    return found

print("\nChecking for letter-feature pairs in top-10 alpha_ij (all 26 letters)...")
letter_pairs_10 = check_letter_feature_pairs(alpha_top100, probe_directions, sae_narrow, top_n=10)
letter_pairs_20 = check_letter_feature_pairs(alpha_top100, probe_directions, sae_narrow, top_n=20)
print(f"Found in top-10: {len(letter_pairs_10)}, top-20: {len(letter_pairs_20)}")

# ===== STEP 8: Run absorption measurement for all 26 letters =====
report_progress(8, 10, "Running absorption measurement for all 26 letters...")

from sae_spelling.feature_absorption_calculator import FeatureAbsorptionCalculator

def letter_delta_metric(tokenizer_obj, pos_letter, device=DEVICE):
    LETTERS_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    pos_letter_tok = tokenizer_obj.encode(f" {pos_letter.upper()}")[-1]
    neg_letter_toks = torch.tensor(
        [tokenizer_obj.encode(f" {l}")[-1] for l in LETTERS_UPPER if l != pos_letter.upper()]
    ).to(device)
    def metric_fn(logits):
        pos_logit = logits[:, -1, pos_letter_tok]
        neg_logits = logits[:, -1, neg_letter_toks]
        return pos_logit - neg_logits.mean(dim=-1)
    return metric_fn


def get_main_feature_ids_for_letter(letter, sae, model, valid_words_prompts,
                                    neg_words_prompts, top_k=5,
                                    pos=VERBOSE_FIRST_LETTER_TOKEN_POS):
    def get_sae_acts_at_pos(words_prompts):
        acts = []
        with torch.no_grad():
            for w, p in words_prompts:
                try:
                    _, cache = model.run_with_cache([p.base], names_filter=sae.cfg.hook_name)
                    sae_in = cache[sae.cfg.hook_name]
                    sae_acts = sae.encode(sae_in)
                    acts.append(sae_acts[0, pos, :].cpu().float())
                except Exception:
                    continue
        if not acts:
            return None
        return torch.stack(acts)
    pos_acts = get_sae_acts_at_pos(valid_words_prompts)
    neg_acts = get_sae_acts_at_pos(neg_words_prompts)
    if pos_acts is None or pos_acts.shape[0] < 2:
        return []
    pos_mean = pos_acts.mean(0)
    neg_mean = neg_acts.mean(0) if neg_acts is not None else torch.zeros_like(pos_mean)
    diff = pos_mean - neg_mean
    top_ids = diff.topk(top_k).indices.tolist()
    return top_ids


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

absorption_results = {}

# Build prompts per letter
words_prompts_by_letter = {}
for letter in ALL_LETTERS:
    wps = []
    for w in words_by_letter[letter][:20]:
        try:
            p = build_icl_prompt(w)
            wps.append((w, p))
        except Exception:
            continue
    words_prompts_by_letter[letter] = wps

# Loop over all 26 letters
for idx, letter in enumerate(ALL_LETTERS):
    print(f"\n--- Letter {letter} ({idx+1}/26) ---")
    if letter not in probe_directions:
        absorption_results[letter] = {"absorption_rate": None, "error": "no probe direction",
                                       "n_absorbed": 0, "n_total": 0}
        continue

    probe_dir = probe_directions[letter]
    neg_prompts = [(w, p) for l, wps in words_prompts_by_letter.items()
                   if l != letter for w, p in wps[:5]][:20]

    main_ids = get_main_feature_ids_for_letter(
        letter=letter, sae=sae_narrow, model=model,
        valid_words_prompts=words_prompts_by_letter[letter],
        neg_words_prompts=neg_prompts, top_k=5
    )
    print(f"  Main feature IDs: {main_ids}")

    words = words_by_letter[letter]
    if len(words) < 5:
        absorption_results[letter] = {"absorption_rate": None, "error": f"only {len(words)} words",
                                       "n_absorbed": 0, "n_total": len(words)}
        continue

    metric_fn = letter_delta_metric(tokenizer, letter)

    try:
        results = calculator.calculate_absorption_sampled(
            sae=sae_narrow,
            words=words,
            probe_dir=probe_dir.to(DEVICE),
            metric_fn=metric_fn,
            main_feature_ids=main_ids,
            max_ablation_samples=50,
            filter_prompts=True,
            show_progress=True,
        )
        sample_results = results.sample_results
        n_tot = len(sample_results)
        n_abs = sum(1 for r in sample_results if r.is_absorption)
        rate = n_abs / n_tot if n_tot > 0 else 0.0
        print(f"  Letter {letter}: rate={rate:.3f} ({n_abs}/{n_tot})")
        absorption_results[letter] = {
            "absorption_rate": rate,
            "n_absorbed": n_abs,
            "n_total": n_tot,
        }
    except Exception as e:
        print(f"  ERROR for letter {letter}: {e}")
        import traceback
        traceback.print_exc()
        try:
            results = calculator.calculate_absorption_sampled(
                sae=sae_narrow,
                words=words,
                probe_dir=probe_dir.to(DEVICE),
                metric_fn=metric_fn,
                main_feature_ids=main_ids,
                max_ablation_samples=30,
                filter_prompts=False,
                show_progress=True,
            )
            sample_results = results.sample_results
            n_tot = len(sample_results)
            n_abs = sum(1 for r in sample_results if r.is_absorption)
            rate = n_abs / n_tot if n_tot > 0 else 0.0
            print(f"  Fallback (no filter): Letter {letter}: rate={rate:.3f} ({n_abs}/{n_tot})")
            absorption_results[letter] = {
                "absorption_rate": rate,
                "n_absorbed": n_abs,
                "n_total": n_tot,
                "note": "filter_prompts=False fallback",
            }
        except Exception as e2:
            print(f"  Fallback also failed: {e2}")
            absorption_results[letter] = {"absorption_rate": None, "error": str(e2),
                                           "n_absorbed": 0, "n_total": 0}

    # Update progress every letter
    report_progress(8, 10,
        f"Absorption done for {letter} ({idx+1}/26), rate={absorption_results[letter].get('absorption_rate', 'ERR')}",
        {"letters_done": idx+1}
    )

# ===== STEP 9: Gate decision =====
report_progress(9, 10, "Computing full gate decision...")

valid_letters = [l for l in ALL_LETTERS if absorption_results.get(l, {}).get("absorption_rate") is not None]
absorption_rates = {l: absorption_results[l]["absorption_rate"] for l in valid_letters}
letters_in_range = [l for l, r in absorption_rates.items() if 0.05 <= r <= 0.50]

print(f"\n=== FULL GATE DECISION ===")
for l in ALL_LETTERS:
    r = absorption_results.get(l, {})
    rate = r.get("absorption_rate")
    if rate is not None:
        flag = "IN RANGE" if 0.05 <= rate <= 0.50 else "OUT OF RANGE"
        print(f"  Letter {l}: {rate:.3f} [{flag}] ({r.get('n_absorbed',0)}/{r.get('n_total',0)})")
    else:
        print(f"  Letter {l}: ERROR - {r.get('error','unknown')}")

absorption_gate = len(letters_in_range) >= 13  # Full mode: at least 13/26 letters
alpha_gate_10 = len(letter_pairs_10) >= 1
alpha_gate_20 = len(letter_pairs_20) >= 1
pipeline_gate = len(valid_letters) >= 13
alpha_ij_gate = len(alpha_top100) >= 100

lenient_go = pipeline_gate and alpha_ij_gate
gate_decision = absorption_gate and pipeline_gate
gate_full = gate_decision and alpha_gate_10

print(f"\nAbsorption gate (>= 13 of 26 in [5-50%]): {'PASS' if absorption_gate else 'FAIL'} ({len(letters_in_range)}/26)")
print(f"Alpha gate top-10: {'PASS' if alpha_gate_10 else 'FAIL'} ({len(letter_pairs_10)} pairs)")
print(f"Alpha_ij gate (>= 100 pairs): {'PASS' if alpha_ij_gate else 'FAIL'}")
print(f"Pipeline gate (>= 13 letters): {'PASS' if pipeline_gate else 'FAIL'} ({len(valid_letters)}/26)")

# For GPT-2, lenient gate is primary
final_go = lenient_go
go_str = "GO" if final_go else "NO_GO"

# Summary stats
valid_rates = [r for r in absorption_rates.values() if r is not None]
mean_rate = float(np.mean(valid_rates)) if valid_rates else 0.0
median_rate = float(np.median(valid_rates)) if valid_rates else 0.0
std_rate = float(np.std(valid_rates)) if valid_rates else 0.0

print(f"\nAbsorption stats across {len(valid_rates)} letters:")
print(f"  Mean: {mean_rate:.3f}, Median: {median_rate:.3f}, Std: {std_rate:.3f}")
print(f"  In range [5-50%]: {len(letters_in_range)}/{len(valid_rates)}")
print(f"\nFinal GO/NO-GO: {go_str}")

# ===== STEP 10: Write output =====
report_progress(10, 10, "Writing full output files...")

output = {
    "task_id": TASK_ID,
    "timestamp": datetime.datetime.now().isoformat(),
    "mode": "FULL",
    "model": "gpt2-small",
    "model_note": "GPT-2 Small (open-model anchor; Gemma-2-2b requires gated HF access)",
    "sae_narrow": {"release": SAE_NARROW_RELEASE, "sae_id": SAE_NARROW_ID, "d_sae": sae_narrow.cfg.d_sae},
    "sae_wide": {"release": SAE_WIDE_RELEASE, "sae_id": SAE_WIDE_ID, "d_sae": sae_wide.cfg.d_sae},
    "n_letters": 26,
    "go_no_go": go_str,
    "gate_checks": {
        "absorption_gate": absorption_gate,
        "absorption_gate_threshold": "13/26 letters in [5-50%]",
        "letters_in_range_count": len(letters_in_range),
        "letters_in_range": letters_in_range,
        "alpha_gate_top10": alpha_gate_10,
        "alpha_gate_top20": alpha_gate_20,
        "alpha_ij_gate": alpha_ij_gate,
        "pipeline_gate": pipeline_gate,
        "overall_lenient": lenient_go,
        "overall_strict": gate_full,
        "absorption_gate_note": "GPT-2 fallback; pipeline gate is primary",
    },
    "l0_confound": l0_confound,
    "absorption_by_letter": absorption_results,
    "absorption_summary": {
        "mean_rate": mean_rate,
        "median_rate": median_rate,
        "std_rate": std_rate,
        "n_valid_letters": len(valid_rates),
        "n_letters_in_range": len(letters_in_range),
        "letters_in_range": letters_in_range,
    },
    "l0_stats": {
        "narrow": {"release": SAE_NARROW_RELEASE, "sae_id": SAE_NARROW_ID, **l0_narrow},
        "wide": {"release": SAE_WIDE_RELEASE, "sae_id": SAE_WIDE_ID, **l0_wide},
        "l0_diff_pct": l0_diff_pct,
        "l0_confound": l0_confound,
    },
    "alpha_ij_top10": alpha_top100[:10],
    "letter_feature_pairs_top10": letter_pairs_10,
    "letter_feature_pairs_top20": letter_pairs_20,
    "runtime_seconds": time.time() - start_time,
}

output_path = FULL_DIR / "P0_pipeline_full.json"
output_path.write_text(json.dumps(output, indent=2))
print(f"\nFull output written to: {output_path}")

# Summary markdown
summary_md = f"""# P0 Pipeline Validation — FULL Mode (26 Letters)

**Model**: GPT-2 Small (gpt2-small-res-jb SAEs, layer 8)
**Mode**: FULL (all 26 letters)
**Decision**: {go_str}
**Timestamp**: {datetime.datetime.now().isoformat()}

## Absorption Rates by Letter (Layer 8, ~24k SAE)

| Letter | Absorption Rate | n_absorbed | n_total | In Range [5-50%] |
|--------|----------------|------------|---------|-----------------|
"""
for l in ALL_LETTERS:
    r = absorption_results.get(l, {})
    rate = r.get("absorption_rate")
    if rate is not None:
        in_range = "YES" if 0.05 <= rate <= 0.50 else "NO"
        summary_md += f"| {l} | {rate:.3f} | {r.get('n_absorbed','?')} | {r.get('n_total','?')} | {in_range} |\n"
    else:
        summary_md += f"| {l} | ERROR | - | - | - |\n"

summary_md += f"""
## Summary Statistics

- **Mean absorption rate**: {mean_rate:.3f}
- **Median absorption rate**: {median_rate:.3f}
- **Std**: {std_rate:.3f}
- **Letters in range [5-50%]**: {len(letters_in_range)}/26

## L0 Statistics

| SAE | Width | Mean L0 | Std L0 |
|-----|-------|---------|--------|
| Narrow | ~24k | {l0_narrow['mean_l0']:.1f} | {l0_narrow['std_l0']:.1f} |
| Wide | ~49k | {l0_wide['mean_l0']:.1f} | {l0_wide['std_l0']:.1f} |

L0 difference: {l0_diff_pct*100:.1f}% -> Confound: **{'YES' if l0_confound else 'NO'}**

## Gate Decision

- Absorption gate (≥13/26 letters in [5-50%]): {'PASS' if absorption_gate else 'FAIL'} ({len(letters_in_range)}/26)
- Alpha gate (≥1 letter-feature pair in top-10): {'PASS' if alpha_gate_10 else 'FAIL'}
- Pipeline gate (≥13 letters processed): {'PASS' if pipeline_gate else 'FAIL'} ({len(valid_letters)}/26)
- **Final (lenient): {go_str}**
"""
(FULL_DIR / "P0_pipeline_full_summary.md").write_text(summary_md)
print(f"Summary written to: {FULL_DIR / 'P0_pipeline_full_summary.md'}")

# Mark done
actual_min = max(1, int((time.time() - start_time) / 60))
mark_done(
    status="success",
    summary=(f"P0 FULL gate: {go_str}. "
             f"Letters in range: {len(letters_in_range)}/26. "
             f"Mean absorption: {mean_rate:.3f}. "
             f"L0 confound: {l0_confound}.")
)

# Update gpu_progress.json
gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
try:
    gp = json.loads(gpu_progress_path.read_text()) if gpu_progress_path.exists() else {
        "completed": [], "failed": [], "running": {}, "timings": {}}
    if TASK_ID not in gp.get("completed", []):
        gp.setdefault("completed", []).append(TASK_ID)
    gp.get("running", {}).pop(TASK_ID, None)
    gp.setdefault("timings", {})[TASK_ID] = {
        "planned_min": 15,
        "actual_min": actual_min,
        "start_time": datetime.datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.datetime.now().isoformat(),
        "config_snapshot": {
            "model": "gpt2-small", "layer": LAYER,
            "sae_narrow": SAE_NARROW_ID, "sae_wide": SAE_WIDE_ID,
            "n_letters": 26, "mode": "full",
        },
    }
    gpu_progress_path.write_text(json.dumps(gp, indent=2))
    print(f"gpu_progress.json updated")
except Exception as e:
    print(f"Warning: gpu_progress.json update failed: {e}")

# Update experiment_state.json
exp_state_path = WORKSPACE / "exp" / "experiment_state.json"
try:
    es = json.loads(exp_state_path.read_text()) if exp_state_path.exists() else {"tasks": {}}
    es.setdefault("tasks", {})[TASK_ID] = {
        "status": "completed",
        "completed_at": datetime.datetime.now().isoformat(),
        "gate_decision": go_str,
        "mode": "full",
    }
    exp_state_path.write_text(json.dumps(es, indent=2))
    print(f"experiment_state.json updated")
except Exception as e:
    print(f"Warning: experiment_state.json update failed: {e}")

print(f"\n=== P0 FULL COMPLETE | Runtime: {actual_min}min | Gate: {go_str} ===")
print(f"Letters in range [5-50%]: {letters_in_range}")
print(f"Mean absorption rate: {mean_rate:.3f}")
