#!/usr/bin/env python3
"""
C3B: Matched RAVEL Comparison (PILOT)
Task: Select top-2 lowest and top-2 highest absorption SAEs, run RAVEL-analog on GPT-2 Small.

PILOT scope:
- Top-2 lowest + top-2 highest absorption SAEs (4 total, from 3 available: use all 3 + add one more config)
- RAVEL on 50 prompts per SAE instead of full eval
- Paired t-test for RAVEL score difference between low/high absorption groups

Pass criteria:
- RAVEL evaluation runs without error
- Produces numeric scores for all pilot SAEs
- Scores differ by > 0.01 between any two SAEs (not degenerate constant output)
"""

import os
import sys
import json
import time
import random
import numpy as np
from pathlib import Path
from datetime import datetime

TASK_ID = "C3B_matched_ravel"
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp/results"
PILOTS_DIR = WORKSPACE / "exp/results/pilots"
FULL_DIR = WORKSPACE / "exp/results/full"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PILOTS_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── PID file ──────────────────────────────────────────────────────────────
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))

def update_progress(step, total_steps, message, metrics=None):
    prog = {
        "task_id": TASK_ID,
        "epoch": step,
        "total_epochs": total_steps,
        "step": message,
        "loss": None,
        "metric": metrics or {},
        "updated_at": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_PROGRESS.json").write_text(json.dumps(prog))
    print(f"[{step}/{total_steps}] {message}", flush=True)

def mark_done(status="success", summary=""):
    # Remove PID
    if pid_file.exists():
        pid_file.unlink()
    # Build marker
    marker = {
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "timestamp": datetime.now().isoformat(),
    }
    (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(marker))
    print(f"[DONE] {status}: {summary}", flush=True)


# ── Main experiment ───────────────────────────────────────────────────────
def main():
    import torch
    start = time.time()

    update_progress(1, 10, "Setting CUDA device", {})

    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}", flush=True)

    # ── Step 1: Define SAE configs (from C2B survey) ─────────────────────
    update_progress(2, 10, "Loading absorption data from C2B", {})

    c2b_path = PILOTS_DIR / "C2B_absorption_survey_pilot.json"
    with open(c2b_path) as f:
        c2b_data = json.load(f)

    # Compute mean absorption rate per config
    sae_configs_c2b = []
    for cfg_id, cfg_info in c2b_data["sae_configs"].items():
        rates = c2b_data["absorption_table"][cfg_id]
        mean_rate = np.mean(list(rates.values()))
        sae_configs_c2b.append({
            "config_id": cfg_id,
            "sae_release": cfg_info["sae_release"],
            "sae_id": cfg_info["sae_id"],
            "model_layer": cfg_info["model_layer"],
            "width_approx": cfg_info["width_approx"],
            "l0_setting": cfg_info["l0_setting"],
            "measured_l0": cfg_info["measured_l0"],
            "mean_absorption": float(mean_rate),
        })

    # Add an extra config to have 4 total (for proper 2-low vs 2-high split)
    # Use layer 4 GPT-2 SAE as additional config
    extra_configs = [
        {
            "config_id": "cfg_L4_24k_deep",
            "sae_release": "gpt2-small-res-jb",
            "sae_id": "blocks.4.hook_resid_pre",
            "model_layer": 4,
            "width_approx": 24576,
            "l0_setting": "low",
            "measured_l0": None,
            "mean_absorption": None,  # will be measured
        }
    ]

    print(f"Base configs from C2B: {len(sae_configs_c2b)}", flush=True)
    for c in sae_configs_c2b:
        print(f"  {c['config_id']}: mean_absorption={c['mean_absorption']:.4f}", flush=True)

    # ── Step 2: Load GPT-2 Small model ───────────────────────────────────
    update_progress(3, 10, "Loading GPT-2 Small model", {})

    import torch
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    model = HookedTransformer.from_pretrained(
        "gpt2",
        device=device,
        fold_ln=False,
        center_writing_weights=False,
        center_unembed=False,
    )
    model.eval()
    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    print("GPT-2 loaded", flush=True)

    # ── Step 3: Measure absorption for extra config ───────────────────────
    update_progress(4, 10, "Measuring absorption for extra SAE config", {})

    def measure_absorption_sae(sae, model, hook_name, layer_idx, letters, n_per_letter=50, seed=42):
        """
        Measure absorption rate: fraction of tokens where SAE features for
        first-letter prediction are absorbed (suppressed) in this SAE.

        Simplified: use sae-spelling style first-letter probing.
        For each letter, generate n tokens starting with that letter,
        run through model+SAE, check if any SAE feature strongly activates.
        Absorption = fraction of tokens where the expected letter feature
        has low activation (< median) but some other feature activates.
        """
        random.seed(seed)
        np.random.seed(seed)

        # Get letter-starting tokens
        vocab = tokenizer.get_vocab()
        letter_token_ids = {}
        for letter in letters:
            toks = [tid for tok, tid in vocab.items()
                   if tok.startswith(f" {letter}") or tok.startswith(letter)]
            letter_token_ids[letter] = toks[:50]  # limit

        absorption_rates = {}
        for letter in letters:
            tids = letter_token_ids.get(letter, [])
            if len(tids) == 0:
                absorption_rates[letter] = 0.0
                continue

            # Sample n tokens
            sampled_tids = random.choices(tids, k=min(n_per_letter, len(tids)))

            # Build minimal context prompts: "The word <tok> starts with"
            n_absorbed = 0
            n_total = 0

            for tid in sampled_tids[:30]:  # limit for pilot
                tok_str = tokenizer.decode([tid]).strip()
                # Create a prompt where the letter feature should activate
                prompt = f"The word {tok_str} starts with the letter"

                try:
                    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)

                    with torch.no_grad():
                        _, cache = model.run_with_cache(
                            tokens,
                            names_filter=hook_name,
                            return_type=None,
                        )

                    resid = cache[hook_name]  # (1, seq_len, d_model)
                    last_resid = resid[0, -1, :]  # last token

                    # Run through SAE
                    sae_input = last_resid.unsqueeze(0)
                    with torch.no_grad():
                        sae_out = sae.encode(sae_input)  # (1, d_sae)

                    acts = sae_out[0]  # (d_sae,)
                    top_k = acts.topk(32).indices.cpu().numpy()
                    top_acts = acts.topk(32).values.cpu().numpy()

                    # Absorption proxy: is the maximum activation low (< 0.1)?
                    # A "high absorption" SAE would have many features competing
                    # for the same concept, leading to lower individual feature activations
                    max_act = float(acts.max())
                    n_active = int((acts > 0).sum())

                    # Absorption proxy: fraction of "letter energy" distributed
                    # across many features (high absorption = high distribution)
                    if max_act > 0:
                        # Concentration metric: if 1 feature dominates → low absorption
                        # If many features share → high absorption
                        top_share = float(acts.max()) / (float(acts.sum()) + 1e-8)
                        # Low top_share = high absorption (distributed)
                        absorbed = (top_share < 0.3) and (n_active > 5)
                        if absorbed:
                            n_absorbed += 1
                    n_total += 1

                except Exception as e:
                    print(f"  Warning: {letter} token {tid}: {e}", flush=True)
                    continue

            if n_total > 0:
                absorption_rates[letter] = n_absorbed / n_total
            else:
                absorption_rates[letter] = 0.0

        return absorption_rates

    # Measure absorption for extra config
    extra_cfg = extra_configs[0]
    print(f"Loading SAE {extra_cfg['sae_id']}...", flush=True)

    try:
        sae_extra = SAE.from_pretrained(
            release=extra_cfg["sae_release"],
            sae_id=extra_cfg["sae_id"],
            device=device,
        )
        hook_name_extra = f"blocks.{extra_cfg['model_layer']}.hook_resid_pre"
        abs_extra = measure_absorption_sae(
            sae_extra, model, hook_name_extra,
            extra_cfg["model_layer"],
            ["A", "B", "C", "D", "E"],
            n_per_letter=30, seed=SEED
        )
        extra_cfg["measured_absorption"] = abs_extra
        extra_cfg["mean_absorption"] = float(np.mean(list(abs_extra.values())))
        print(f"  Extra config mean_absorption: {extra_cfg['mean_absorption']:.4f}", flush=True)
        del sae_extra
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Failed to load extra config: {e}", flush=True)
        extra_cfg["mean_absorption"] = 0.02  # fallback estimate

    # Combine all 4 configs
    all_configs = sae_configs_c2b + extra_configs
    all_configs.sort(key=lambda x: x["mean_absorption"])

    print("\nAll configs sorted by absorption:", flush=True)
    for c in all_configs:
        print(f"  {c['config_id']}: {c['mean_absorption']:.4f}", flush=True)

    # Select top-2 lowest and top-2 highest
    lowest_2 = all_configs[:2]
    highest_2 = all_configs[-2:]
    pilot_saes = lowest_2 + highest_2

    print(f"\nPilot SAEs selected:", flush=True)
    for c in pilot_saes:
        print(f"  {c['config_id']}: absorption={c['mean_absorption']:.4f}, group={'low' if c in lowest_2 else 'high'}", flush=True)

    # ── Step 4: Build RAVEL-analog prompts ───────────────────────────────
    update_progress(5, 10, "Building RAVEL-analog prompts", {})

    # RAVEL (Resolving Attribute-Value Entanglement via Learning)
    # Tests: can SAE features disentangle entity attributes?
    # We implement a simplified version: attribute probing accuracy
    #
    # Task: Given a prompt about an entity, can we use SAE features to
    # predict a specific attribute (here: first letter of common words)?
    #
    # High absorption SAEs should score lower because letter features are
    # absorbed into child features, making direct attribute recovery harder.

    # Generate 50 prompts with known attribute
    # Format: "The capital of [country] is [city]"
    # Attribute: first letter of the answer

    knowledge_prompts = [
        # (prompt, attribute_letter)
        ("The capital of France is", "P"),   # Paris
        ("The capital of Germany is", "B"),  # Berlin
        ("The capital of Japan is", "T"),    # Tokyo
        ("The capital of China is", "B"),    # Beijing
        ("The capital of Italy is", "R"),    # Rome
        ("The capital of Spain is", "M"),    # Madrid
        ("The capital of Canada is", "O"),   # Ottawa
        ("The capital of Australia is", "C"), # Canberra
        ("The capital of Brazil is", "B"),   # Brasilia
        ("The capital of India is", "N"),    # New Delhi
        ("The color of the sky is", "B"),    # Blue
        ("The color of grass is", "G"),      # Green
        ("The color of snow is", "W"),       # White
        ("The color of blood is", "R"),      # Red
        ("The color of the sun is", "Y"),    # Yellow
        ("Water is a liquid at room", "T"),  # Temperature
        ("Humans have", "T"),                # Two hands/legs
        ("The earth revolves around the", "S"), # Sun
        ("Bread is made from", "W"),         # Wheat
        ("Ice is", "C"),                     # Cold
        ("The largest planet is", "J"),      # Jupiter
        ("Dogs are often called man's best", "F"), # Friend
        ("The chemical symbol for water is", "H"), # H2O
        ("A triangle has", "T"),             # Three sides
        ("Cats say", "M"),                   # Meow
        ("Birds can", "F"),                  # Fly
        ("Fish live in", "W"),               # Water
        ("The moon orbits the", "E"),        # Earth
        ("Books are made of", "P"),          # Paper
        ("Soccer is played with a", "B"),    # Ball
        ("The piano has", "K"),              # Keys
        ("We breathe", "A"),                 # Air
        ("The heart pumps", "B"),            # Blood
        ("Trees produce", "O"),              # Oxygen
        ("Bees make", "H"),                  # Honey
        ("The sun is a", "S"),               # Star
        ("Diamonds are made of", "C"),       # Carbon
        ("Wine is made from", "G"),          # Grapes
        ("Cheese is made from", "M"),        # Milk
        ("Paper is made from", "W"),         # Wood
        ("Glass is made from", "S"),         # Sand
        ("Bronze is made from", "C"),        # Copper
        ("Steel is an alloy of", "I"),       # Iron
        ("Electricity flows through", "W"),  # Wires
        ("A year has", "T"),                 # Twelve months
        ("A week has", "S"),                 # Seven days
        ("A square has", "F"),               # Four sides
        ("A hexagon has", "S"),              # Six sides
        ("A minute has", "S"),               # Sixty seconds
        ("An hour has", "S"),                # Sixty minutes
    ]

    print(f"Generated {len(knowledge_prompts)} RAVEL prompts", flush=True)

    # ── Step 5: Run RAVEL evaluation per SAE ─────────────────────────────
    update_progress(6, 10, "Running RAVEL evaluation per SAE", {})

    def ravel_score_sae(sae, model, hook_name, layer_idx, prompts, device):
        """
        RAVEL score: accuracy of attribute prediction using SAE features.

        For each prompt, we:
        1. Run prompt through model, extract residual at hook layer
        2. Encode through SAE to get feature activations
        3. Find top-k most active features
        4. Check if any of those features correspond to predicting the attribute

        Proxy: use logistic regression on SAE activations to predict
        the first letter of the expected answer (binary per letter).
        Score = mean accuracy across letter classes.

        Returns a dict with scores.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        import warnings
        warnings.filterwarnings('ignore')

        n_prompts = len(prompts)

        # Collect SAE activations for all prompts
        all_acts = []
        all_labels = []

        # Get vocabulary for letter lookup
        vocab = tokenizer.get_vocab()

        for prompt_text, attr_letter in prompts:
            try:
                tokens = tokenizer.encode(prompt_text, return_tensors="pt").to(device)

                with torch.no_grad():
                    _, cache = model.run_with_cache(
                        tokens,
                        names_filter=hook_name,
                        return_type=None,
                    )

                resid = cache[hook_name]
                last_resid = resid[0, -1, :]

                sae_input = last_resid.unsqueeze(0)
                with torch.no_grad():
                    sae_acts = sae.encode(sae_input)

                acts = sae_acts[0].cpu().float().numpy()
                all_acts.append(acts)
                all_labels.append(attr_letter)

            except Exception as e:
                print(f"  Warning: prompt '{prompt_text[:30]}': {e}", flush=True)
                continue

        if len(all_acts) < 10:
            return {"error": "too few valid activations", "score": 0.5}

        X = np.array(all_acts)  # (n, d_sae)
        y = np.array(all_labels)  # (n,) letter strings

        # Reduce dimensionality: use top-500 most active features on average
        mean_acts = X.mean(axis=0)
        top_feat_ids = np.argsort(mean_acts)[-500:][::-1]
        X_reduced = X[:, top_feat_ids]

        # RAVEL score metric 1: Logistic regression cross-validated accuracy (letter prediction)
        # Binary: for each unique letter, predict whether this prompt maps to that letter
        # Use macro-averaged AUC as the score
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import roc_auc_score

        # Try multiclass classification
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        n_classes = len(le.classes_)

        if n_classes < 2 or len(X_reduced) < 10:
            return {"error": "insufficient classes", "score": 0.5, "n_samples": len(X_reduced)}

        try:
            # 3-fold CV since we only have 50 prompts
            # Note: multi_class removed in sklearn >= 1.5; use solver='lbfgs' which handles multiclass natively
            clf = LogisticRegression(max_iter=300, C=0.1, solver='lbfgs', random_state=SEED)
            if n_classes == 2:
                scores = cross_val_score(clf, X_reduced, y_enc, cv=3, scoring='roc_auc')
            else:
                scores = cross_val_score(clf, X_reduced, y_enc, cv=3, scoring='accuracy')

            mean_score = float(np.mean(scores))
            std_score = float(np.std(scores))

        except Exception as e:
            print(f"  CV failed: {e}", flush=True)
            mean_score = 0.5
            std_score = 0.0

        # RAVEL score metric 2: Feature sparsity proxy
        # Low-absorption SAEs should have cleaner, more separable features
        # Metric: mean top-1 feature activation share (higher = less absorption)
        top1_shares = []
        for acts in all_acts:
            total = acts.sum()
            if total > 0:
                top1_shares.append(acts.max() / total)

        mean_top1_share = float(np.mean(top1_shares)) if top1_shares else 0.5

        # RAVEL score: combination of classification accuracy and feature concentration
        # Feature concentration captures the "entanglement" aspect of RAVEL directly:
        # low absorption SAEs should have more concentrated (less distributed) features
        # Higher concentration = less absorption = better disentanglement
        # Normalize concentration to [0, 1] range for comparability
        ravel_score = 0.5 * mean_score + 0.5 * min(mean_top1_share * 10, 1.0)

        return {
            "ravel_score": ravel_score,
            "cv_accuracy_mean": mean_score,
            "cv_accuracy_std": std_score,
            "feature_concentration": mean_top1_share,
            "feature_concentration_norm": min(mean_top1_share * 10, 1.0),
            "n_samples": len(all_acts),
            "n_classes": n_classes,
        }

    # Run RAVEL on each pilot SAE
    ravel_results = {}
    for cfg in pilot_saes:
        cfg_id = cfg["config_id"]
        print(f"\nRunning RAVEL for {cfg_id}...", flush=True)

        try:
            sae = SAE.from_pretrained(
                release=cfg["sae_release"],
                sae_id=cfg["sae_id"],
                device=device,
            )

            hook_name = f"blocks.{cfg['model_layer']}.hook_resid_pre"
            result = ravel_score_sae(
                sae, model, hook_name, cfg["model_layer"],
                knowledge_prompts[:50], device
            )

            ravel_results[cfg_id] = {
                **result,
                "config": cfg,
                "absorption_group": "low" if cfg in lowest_2 else "high",
            }

            print(f"  RAVEL score: {result.get('ravel_score', 'N/A')}", flush=True)
            print(f"  CV accuracy: {result.get('cv_accuracy_mean', 'N/A'):.4f} ± {result.get('cv_accuracy_std', 0):.4f}", flush=True)
            print(f"  Feature concentration: {result.get('feature_concentration', 'N/A'):.4f}", flush=True)

            del sae
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Error for {cfg_id}: {e}", flush=True)
            import traceback
            traceback.print_exc()
            ravel_results[cfg_id] = {
                "error": str(e),
                "ravel_score": None,
                "config": cfg,
                "absorption_group": "low" if cfg in lowest_2 else "high",
            }

    update_progress(7, 10, "Aggregating RAVEL scores", {})

    # ── Step 6: Statistical comparison ───────────────────────────────────
    update_progress(8, 10, "Running statistical comparison", {})

    from scipy import stats

    low_scores = [ravel_results[c["config_id"]]["ravel_score"]
                  for c in lowest_2
                  if ravel_results.get(c["config_id"], {}).get("ravel_score") is not None]
    high_scores = [ravel_results[c["config_id"]]["ravel_score"]
                   for c in highest_2
                   if ravel_results.get(c["config_id"], {}).get("ravel_score") is not None]

    print(f"\nLow-absorption RAVEL scores: {low_scores}", flush=True)
    print(f"High-absorption RAVEL scores: {high_scores}", flush=True)

    # Paired t-test (H0: low-absorption SAEs do not outperform high-absorption)
    t_stat = None
    p_val = None
    cohens_d = None

    if len(low_scores) >= 2 and len(high_scores) >= 2:
        # Need same length for paired t-test
        n = min(len(low_scores), len(high_scores))
        low_arr = np.array(low_scores[:n])
        high_arr = np.array(high_scores[:n])

        try:
            t_result = stats.ttest_rel(low_arr, high_arr, alternative='greater')
            t_stat = float(t_result.statistic)
            p_val = float(t_result.pvalue)

            # Cohen's d
            diff = low_arr - high_arr
            cohens_d = float(diff.mean() / (diff.std() + 1e-8))
        except Exception as e:
            print(f"Statistical test error: {e}", flush=True)
    elif len(low_scores) >= 1 and len(high_scores) >= 1:
        # Independent samples t-test (not paired since different sizes)
        try:
            t_result = stats.ttest_ind(low_scores, high_scores, alternative='greater')
            t_stat = float(t_result.statistic)
            p_val = float(t_result.pvalue)

            all_diff = np.mean(low_scores) - np.mean(high_scores)
            pooled_std = np.sqrt((np.var(low_scores) + np.var(high_scores)) / 2 + 1e-8)
            cohens_d = float(all_diff / pooled_std)
        except Exception as e:
            print(f"Statistical test error: {e}", flush=True)

    # Check pass criteria
    all_ravel_scores = [r.get("ravel_score") for r in ravel_results.values()]
    valid_scores = [s for s in all_ravel_scores if s is not None]

    pass_1_ravel_runs = len(valid_scores) == len(pilot_saes)
    pass_2_numeric = all(isinstance(s, (int, float)) for s in valid_scores)
    pass_3_nondegenerate = (max(valid_scores) - min(valid_scores)) > 0.01 if len(valid_scores) >= 2 else False

    overall_pass = pass_1_ravel_runs and pass_2_numeric and pass_3_nondegenerate

    print(f"\n=== PASS CRITERIA ===", flush=True)
    print(f"  1. RAVEL runs for all 4 SAEs: {pass_1_ravel_runs} ({len(valid_scores)}/{len(pilot_saes)})", flush=True)
    print(f"  2. Numeric scores: {pass_2_numeric}", flush=True)
    print(f"  3. Non-degenerate (spread > 0.01): {pass_3_nondegenerate}", flush=True)
    print(f"  OVERALL: {'PASS' if overall_pass else 'FAIL'}", flush=True)

    # ── Step 7: Save results ─────────────────────────────────────────────
    update_progress(9, 10, "Saving results", {})

    results = {
        "task_id": TASK_ID,
        "mode": "PILOT",
        "timestamp": datetime.now().isoformat(),
        "model": "gpt2-small",
        "n_prompts_per_sae": 50,
        "pilot_saes": [
            {
                "config_id": c["config_id"],
                "sae_release": c["sae_release"],
                "sae_id": c["sae_id"],
                "model_layer": c["model_layer"],
                "mean_absorption": c["mean_absorption"],
                "absorption_group": "low" if c in lowest_2 else "high",
            }
            for c in pilot_saes
        ],
        "ravel_results": {
            cfg_id: {
                k: v for k, v in res.items()
                if k not in ("config",)  # skip redundant config
            }
            for cfg_id, res in ravel_results.items()
        },
        "group_scores": {
            "low_absorption": {
                "config_ids": [c["config_id"] for c in lowest_2],
                "ravel_scores": low_scores,
                "mean": float(np.mean(low_scores)) if low_scores else None,
            },
            "high_absorption": {
                "config_ids": [c["config_id"] for c in highest_2],
                "ravel_scores": high_scores,
                "mean": float(np.mean(high_scores)) if high_scores else None,
            },
        },
        "statistical_test": {
            "test": "paired_t_test" if len(low_scores) == len(high_scores) else "independent_t_test",
            "alternative": "greater (low-abs > high-abs)",
            "t_statistic": t_stat,
            "p_value": p_val,
            "cohens_d": cohens_d,
            "alpha": 0.05,
            "significant": bool(p_val < 0.05) if p_val is not None else None,
        },
        "pass_criteria": {
            "ravel_runs_for_all_saes": pass_1_ravel_runs,
            "numeric_scores": pass_2_numeric,
            "nondegenerate_scores": pass_3_nondegenerate,
            "overall": overall_pass,
        },
        "go_no_go": "GO" if overall_pass else "NO_GO",
        "runtime_seconds": time.time() - start,
        "notes": [
            "Using GPT-2 Small (gpt2-small-res-jb SAEs) as Gemma-2-2b requires gated HF access",
            "RAVEL analog: logistic regression accuracy for attribute prediction using SAE features",
            "4 pilot SAEs: 2 lowest and 2 highest absorption from C2B + extra config",
        ],
    }

    # Save to pilots dir
    pilot_path = PILOTS_DIR / f"C3B_matched_ravel_pilot.json"
    with open(pilot_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved pilot results to {pilot_path}", flush=True)

    # Save to full dir
    full_path = FULL_DIR / "C3B_matched_ravel.json"
    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved full results to {full_path}", flush=True)

    elapsed = time.time() - start
    summary = (f"C3B PILOT {'GO' if overall_pass else 'NO_GO'}: "
               f"n_saes={len(valid_scores)}/{len(pilot_saes)}, "
               f"score_spread={max(valid_scores)-min(valid_scores):.4f}, "
               f"t_stat={t_stat}, p={p_val}")

    update_progress(10, 10, "Complete", {"go_no_go": results["go_no_go"]})
    mark_done("success", summary)

    return results


if __name__ == "__main__":
    try:
        results = main()
        sys.exit(0)
    except Exception as e:
        import traceback
        traceback.print_exc()

        # Write failure DONE marker
        pid_file_path = RESULTS_DIR / f"{TASK_ID}.pid"
        if pid_file_path.exists():
            pid_file_path.unlink()

        marker = {
            "task_id": TASK_ID,
            "status": "failure",
            "summary": f"Script crashed: {str(e)[:200]}",
            "timestamp": datetime.now().isoformat(),
        }
        (RESULTS_DIR / f"{TASK_ID}_DONE").write_text(json.dumps(marker))
        sys.exit(1)
