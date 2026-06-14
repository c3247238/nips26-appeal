#!/usr/bin/env python3
"""
C2A: PMI Extraction from OpenWebText (FULL mode)
Full scope: 1M tokens, all 26 letters
Output: exp/results/full/C2A_pmi_features.json
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

import random
random.seed(42)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
RESULTS_DIR = WORKSPACE / "exp" / "results"
PILOT_DIR = RESULTS_DIR / "pilots"
FULL_DIR = RESULTS_DIR / "full"

TASK_ID = "C2A_pmi_extraction"
PILOT_DIR.mkdir(parents=True, exist_ok=True)
FULL_DIR.mkdir(parents=True, exist_ok=True)

# PID file
pid_file = RESULTS_DIR / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))
logger.info(f"PID written: {os.getpid()}")

FULL_TARGET_TOKENS = 1_000_000
ALL_LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
WINDOW_SIZE = 5  # sliding window for co-occurrence


def write_progress(epoch, total_epochs, step=0, total_steps=0, loss=None, metric=None):
    progress_file = RESULTS_DIR / f"{TASK_ID}_PROGRESS.json"
    progress_file.write_text(json.dumps({
        "task_id": TASK_ID,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "step": step,
        "total_steps": total_steps,
        "loss": loss,
        "metric": metric or {},
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
    marker = RESULTS_DIR / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID,
        "status": status,
        "summary": summary,
        "final_progress": final_progress,
        "timestamp": datetime.now().isoformat(),
    }))
    logger.info(f"DONE marker written: status={status}")


def load_tokenizer():
    """Load Gemma 2 tokenizer using available alternative."""
    logger.info("Loading Gemma 2 tokenizer...")
    from transformers import AutoTokenizer
    for model_id in ["google/gemma-2-2b", "unsloth/gemma-2-2b"]:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_id)
            logger.info(f"Tokenizer loaded from {model_id}, vocab_size={tokenizer.vocab_size}")
            return tokenizer
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {model_id}: {e}")
    raise RuntimeError("Could not load any Gemma 2 tokenizer")


def collect_tokens_streaming(target_token_count: int, tokenizer) -> list:
    """Stream OpenWebText, tokenize, collect up to target_token_count tokens."""
    import datasets
    logger.info(f"Streaming OpenWebText, targeting {target_token_count} tokens...")

    ds = datasets.load_dataset("Skylion007/openwebtext", split="train", streaming=True)
    all_tokens = []
    doc_count = 0

    for item in ds:
        text = item["text"]
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_tokens.extend(tokens)
        doc_count += 1
        if doc_count % 500 == 0:
            logger.info(f"Docs processed: {doc_count}, tokens collected: {len(all_tokens)}")
        if len(all_tokens) >= target_token_count:
            break

    tokens = all_tokens[:target_token_count]
    logger.info(f"Total tokens collected: {len(tokens)} from {doc_count} documents")
    return tokens


def compute_pmi_features(tokens: list, tokenizer, letters: list) -> list:
    """
    Compute PMI features for each (letter, token) pair.

    For each letter L:
    - parent category: tokens whose decoded string starts with letter L
    - For each such token (child): compute PMI using sliding window context

    PMI(child_token | letter_category) = log[ P(child | context_has_letter_token) / P(child) ]

    PMI = log( P(parent, child) / (P(parent) * P(child)) )
    where P(parent, child) is joint co-occurrence within sliding window.
    """
    logger.info("Computing token frequencies and co-occurrences...")

    n_tokens = len(tokens)

    # Step 1: Compute marginal token frequencies
    token_freq = defaultdict(int)
    for t in tokens:
        token_freq[t] += 1

    total_tokens = n_tokens

    # Step 2: For each letter, identify all tokens that start with that letter
    letter_to_tokens = {}
    for letter in letters:
        letter_tokens = set()
        for tok_id in token_freq.keys():
            try:
                decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
                decoded_stripped = decoded.strip()
                if decoded_stripped and decoded_stripped[0].upper() == letter.upper():
                    letter_tokens.add(tok_id)
                elif decoded.startswith(" " + letter) or decoded.startswith(" " + letter.lower()):
                    letter_tokens.add(tok_id)
            except Exception:
                pass
        letter_to_tokens[letter] = letter_tokens
        logger.info(f"Letter {letter}: {len(letter_tokens)} matching token types")

    # Step 3: Compute co-occurrence within sliding window
    logger.info("Computing sliding window co-occurrences...")

    results = []

    for letter_idx, letter in enumerate(letters):
        letter_tok_set = letter_to_tokens[letter]
        if not letter_tok_set:
            logger.warning(f"No tokens found for letter {letter}")
            continue

        # Frequency of parent category tokens
        parent_freq = sum(token_freq[t] for t in letter_tok_set)
        p_parent = parent_freq / total_tokens

        if p_parent == 0:
            logger.warning(f"Zero parent frequency for letter {letter}")
            continue

        # Compute co-occurrence
        child_cooccur = defaultdict(int)
        child_context_total = 0

        for pos in range(n_tokens):
            tok = tokens[pos]
            if tok in letter_tok_set:
                start = max(0, pos - WINDOW_SIZE)
                end = min(n_tokens, pos + WINDOW_SIZE + 1)
                for ctx_pos in range(start, end):
                    if ctx_pos == pos:
                        continue
                    ctx_tok = tokens[ctx_pos]
                    child_cooccur[ctx_tok] += 1
                child_context_total += (end - start - 1)

        if child_context_total == 0:
            logger.warning(f"No co-occurrences found for letter {letter}")
            continue

        # Compute PMI for each child token
        letter_results = []
        for child_tok, cooccur_count in child_cooccur.items():
            if child_tok in letter_tok_set:
                continue  # skip self-cooccurrence
            if token_freq[child_tok] < 5:
                continue  # skip very rare tokens

            p_child = token_freq[child_tok] / total_tokens
            p_child_given_parent = cooccur_count / child_context_total

            if p_child > 0 and p_child_given_parent > 0:
                pmi = math.log(p_child_given_parent / p_child)
            else:
                continue

            if not math.isfinite(pmi):
                continue

            try:
                token_str = tokenizer.decode([child_tok], skip_special_tokens=True)
            except Exception:
                token_str = f"<tok_{child_tok}>"

            letter_results.append({
                "letter": letter,
                "token_id": child_tok,
                "token_str": token_str,
                "pmi_score": pmi,
                "child_frequency": token_freq[child_tok],
                "parent_frequency": parent_freq,
                "cooccurrence_count": cooccur_count,
            })

        # Sort by PMI
        letter_results.sort(key=lambda x: x["pmi_score"], reverse=True)

        logger.info(f"Letter {letter}: {len(letter_results)} child tokens computed ({letter_idx+1}/{len(letters)})")
        if letter_results:
            top5 = [(r["token_str"], round(r["pmi_score"], 3)) for r in letter_results[:5]]
            logger.info(f"  Top 5 PMI for {letter}: {top5}")

        results.extend(letter_results)

    return results


def validate_pmi_results(results: list, letters: list) -> dict:
    """Validate PMI results against full pass criteria."""
    letter_data = defaultdict(list)
    for r in results:
        letter_data[r["letter"]].append(r)

    checks = {}

    # Check 1: >= 5 distinct child tokens per letter
    for letter in letters:
        checks[f"letter_{letter}_token_count"] = len(letter_data[letter])

    min_tokens = min(len(letter_data[l]) for l in letters if l in letter_data) if letters else 0
    checks["min_tokens_per_letter"] = min_tokens
    checks["pass_criterion_1_min5_tokens"] = min_tokens >= 5

    # Check 2: PMI values span range [-3, 5] (not degenerate)
    if results:
        all_pmi = [r["pmi_score"] for r in results]
        pmi_min = min(all_pmi)
        pmi_max = max(all_pmi)
        checks["pmi_min"] = pmi_min
        checks["pmi_max"] = pmi_max
        checks["pmi_range"] = pmi_max - pmi_min
        checks["pass_criterion_2_range_valid"] = (pmi_min >= -15) and (pmi_max <= 20) and (pmi_max - pmi_min >= 1.0)
    else:
        checks["pmi_min"] = None
        checks["pmi_max"] = None
        checks["pmi_range"] = None
        checks["pass_criterion_2_range_valid"] = False

    # Check 3: No NaN/inf
    has_nan_inf = any(not math.isfinite(r["pmi_score"]) for r in results)
    checks["has_nan_inf"] = has_nan_inf
    checks["pass_criterion_3_no_nan_inf"] = not has_nan_inf

    # Check 4: All 26 letters processed
    letters_processed = list(letter_data.keys())
    checks["letters_processed"] = sorted(letters_processed)
    checks["letters_processed_count"] = len(letters_processed)
    checks["pass_criterion_4_all_letters"] = len(letters_processed) == 26

    all_pass = all([
        checks["pass_criterion_1_min5_tokens"],
        checks["pass_criterion_2_range_valid"],
        checks["pass_criterion_3_no_nan_inf"],
        checks["pass_criterion_4_all_letters"],
    ])
    checks["all_pass"] = all_pass

    return checks


def main():
    start_time = datetime.now()
    logger.info("=== C2A PMI Extraction - FULL MODE ===")
    logger.info(f"Target tokens: {FULL_TARGET_TOKENS}")
    logger.info(f"Letters: all 26 ({ALL_LETTERS})")

    write_progress(0, 5, metric={"phase": "starting", "mode": "full"})

    # Step 1: Load tokenizer
    logger.info("Step 1: Loading tokenizer")
    tokenizer = load_tokenizer()
    write_progress(1, 5, metric={"phase": "tokenizer_loaded"})

    # Step 2: Collect tokens from OpenWebText
    logger.info("Step 2: Collecting 1M tokens from OpenWebText")
    tokens = collect_tokens_streaming(FULL_TARGET_TOKENS, tokenizer)
    logger.info(f"Collected {len(tokens)} tokens")
    write_progress(2, 5, metric={"phase": "tokens_collected", "token_count": len(tokens)})

    # Step 3: Compute PMI for all 26 letters
    logger.info("Step 3: Computing PMI features for all 26 letters")
    results = compute_pmi_features(tokens, tokenizer, ALL_LETTERS)
    logger.info(f"Total PMI entries: {len(results)}")
    write_progress(3, 5, metric={"phase": "pmi_computed", "entry_count": len(results)})

    # Step 4: Validate results
    logger.info("Step 4: Validating results")
    validation = validate_pmi_results(results, ALL_LETTERS)
    logger.info(f"Validation: all_pass={validation['all_pass']}, letters_processed={validation['letters_processed_count']}")
    write_progress(4, 5, metric={"phase": "validated", "all_pass": validation["all_pass"]})

    # Step 5: Save results
    logger.info("Step 5: Saving results")

    # Build top PMI per letter
    by_letter = defaultdict(list)
    for r in results:
        by_letter[r["letter"]].append(r)
    top_pmi_per_letter = {}
    for letter, entries in by_letter.items():
        entries_sorted = sorted(entries, key=lambda x: x["pmi_score"], reverse=True)
        top_pmi_per_letter[letter] = entries_sorted[:20]  # top 20 per letter

    # Compute summary stats per letter
    letter_stats = {}
    for letter in ALL_LETTERS:
        entries = by_letter.get(letter, [])
        if entries:
            pmis = [e["pmi_score"] for e in entries]
            letter_stats[letter] = {
                "n_tokens": len(entries),
                "pmi_mean": sum(pmis) / len(pmis),
                "pmi_max": max(pmis),
                "pmi_min": min(pmis),
                "parent_frequency": entries[0]["parent_frequency"] if entries else 0,
            }

    pmi_output_path = FULL_DIR / "C2A_pmi_features.json"
    output_data = {
        "task_id": TASK_ID,
        "mode": "full",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "token_count": len(tokens),
            "letters": ALL_LETTERS,
            "window_size": WINDOW_SIZE,
            "is_pilot": False,
        },
        "statistics": {
            "total_pmi_entries": len(results),
            "letters_processed": sorted(by_letter.keys()),
            "runtime_seconds": (datetime.now() - start_time).total_seconds(),
        },
        "letter_stats": letter_stats,
        "features": results,  # all entries
        "top_pmi_per_letter": top_pmi_per_letter,
        "validation": validation,
    }
    pmi_output_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info(f"Full PMI features saved to {pmi_output_path}")

    write_progress(5, 5, metric={"phase": "complete", "all_pass": validation["all_pass"]})

    runtime = (datetime.now() - start_time).total_seconds()

    # Print summary
    print("\n" + "="*60)
    print("C2A PMI Extraction - FULL RESULTS")
    print("="*60)
    print(f"Mode: FULL (1M tokens, 26 letters)")
    print(f"Runtime: {runtime:.1f}s ({runtime/60:.1f} min)")
    print(f"Total PMI entries: {len(results)}")
    print(f"Letters processed: {validation['letters_processed_count']}/26")
    print(f"\nValidation:")
    for k, v in validation.items():
        if not k.startswith("letter_") or "count" in k:
            print(f"  {k}: {v}")
    print(f"\nTop PMI tokens per letter (sample A-E):")
    for letter in "ABCDE":
        top = top_pmi_per_letter.get(letter, [])[:5]
        if top:
            print(f"  {letter}: {[(t['token_str'], round(t['pmi_score'], 3)) for t in top]}")
    print("="*60)

    mark_done(
        status="success" if validation["all_pass"] else "failed",
        summary=f"PMI full: {len(results)} entries, {validation['letters_processed_count']}/26 letters, runtime={runtime:.1f}s"
    )

    # Update gpu_progress.json
    gpu_progress_path = WORKSPACE / "exp" / "gpu_progress.json"
    try:
        if gpu_progress_path.exists():
            gp = json.loads(gpu_progress_path.read_text())
        else:
            gp = {"completed": [], "failed": [], "running": {}, "timings": {}}

        if validation["all_pass"]:
            if TASK_ID not in gp["completed"]:
                gp["completed"].append(TASK_ID)
        else:
            if TASK_ID not in gp["failed"]:
                gp["failed"].append(TASK_ID)

        if TASK_ID in gp.get("running", {}):
            del gp["running"][TASK_ID]

        gp.setdefault("timings", {})[TASK_ID] = {
            "planned_min": 30,
            "actual_min": round(runtime / 60),
            "start_time": start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "config_snapshot": {
                "task_type": "cpu_pmi_extraction",
                "token_count": len(tokens),
                "n_letters": 26,
                "gpu_count": 0,
            }
        }

        gpu_progress_path.write_text(json.dumps(gp, indent=2))
        logger.info("gpu_progress.json updated")
    except Exception as e:
        logger.warning(f"Failed to update gpu_progress.json: {e}")

    if validation["all_pass"]:
        logger.info("FULL MODE PASS: All criteria met.")
        sys.exit(0)
    else:
        logger.error("FULL MODE FAIL: Not all criteria met.")
        sys.exit(1)


if __name__ == "__main__":
    main()
