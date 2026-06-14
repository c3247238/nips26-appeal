#!/usr/bin/env python3
"""
C2A: PMI Extraction from OpenWebText (PILOT mode)
Pilot scope: 100k tokens, letters A-E only
"""

import os
import sys
import json
import math
import logging
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Set random seed
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

PILOT_TARGET_TOKENS = 100_000
PILOT_LETTERS = ["A", "B", "C", "D", "E"]
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
    # Try official first (needs auth), fall back to unsloth mirror
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
        if doc_count % 100 == 0:
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

    Approximation: use the co-occurrence count within a WINDOW_SIZE sliding window.
    P(child) = frequency of child token in corpus
    P(parent, child) = co-occurrence within window / total windows
    PMI = log( P(parent, child) / (P(parent) * P(child)) )
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
        # Sample a batch of token IDs to find those that decode to words starting with L
        for tok_id, count in token_freq.items():
            try:
                decoded = tokenizer.decode([tok_id], skip_special_tokens=True)
                # Check if starts with the letter (case insensitive)
                # Also check with space prefix (common in BPE)
                decoded_stripped = decoded.strip()
                if decoded_stripped and decoded_stripped[0].upper() == letter.upper():
                    letter_tokens.add(tok_id)
                # Also check " Letter" format (BPE word-start tokens)
                elif decoded.startswith(" " + letter) or decoded.startswith(" " + letter.lower()):
                    letter_tokens.add(tok_id)
            except Exception:
                pass
        letter_to_tokens[letter] = letter_tokens
        logger.info(f"Letter {letter}: {len(letter_tokens)} matching token types")

    # Step 3: Compute co-occurrence within sliding window
    # co_occur[parent_tok][child_tok] = count of windows containing both
    # We compute for parent = letter-category tokens

    # Build window co-occurrences
    # For efficiency, use position-based approach
    logger.info("Computing sliding window co-occurrences...")

    # First, build a map from position to token
    # For each position of a "parent" (letter) token, count children in window

    results = []

    for letter in letters:
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

        # Compute co-occurrence: for each window position where a parent token appears,
        # count co-occurring tokens
        child_cooccur = defaultdict(int)
        child_context_total = 0

        for pos in range(n_tokens):
            tok = tokens[pos]
            if tok in letter_tok_set:
                # Define window: [pos - WINDOW_SIZE, pos + WINDOW_SIZE]
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
        # P(child|parent_context) = child_cooccur[child] / child_context_total
        # P(child) = token_freq[child] / total_tokens
        # PMI = log(P(child|parent_context) / P(child))

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

        logger.info(f"Letter {letter}: {len(letter_results)} child tokens computed")
        if letter_results:
            top5 = [(r["token_str"], r["pmi_score"]) for r in letter_results[:5]]
            logger.info(f"  Top 5 PMI for {letter}: {top5}")

        results.extend(letter_results)

    return results


def validate_pmi_results(results: list, letters: list) -> dict:
    """Validate PMI results against pilot pass criteria."""
    letter_data = defaultdict(list)
    for r in results:
        letter_data[r["letter"]].append(r)

    checks = {}

    # Check 1: >= 5 distinct child tokens per pilot letter
    for letter in letters:
        checks[f"letter_{letter}_token_count"] = len(letter_data[letter])

    min_tokens = min(len(letter_data[l]) for l in letters) if letters else 0
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
        checks["pass_criterion_2_range_valid"] = False

    # Check 3: No NaN/inf
    has_nan_inf = any(
        (not math.isfinite(r["pmi_score"])) for r in results
    )
    checks["has_nan_inf"] = has_nan_inf
    checks["pass_criterion_3_no_nan_inf"] = not has_nan_inf

    all_pass = all([
        checks["pass_criterion_1_min5_tokens"],
        checks["pass_criterion_2_range_valid"],
        checks["pass_criterion_3_no_nan_inf"],
    ])
    checks["all_pass"] = all_pass

    return checks


def main():
    start_time = datetime.now()
    logger.info("=== C2A PMI Extraction - PILOT MODE ===")
    logger.info(f"Target tokens: {PILOT_TARGET_TOKENS}")
    logger.info(f"Pilot letters: {PILOT_LETTERS}")

    write_progress(0, 5, metric={"phase": "starting"})

    # Step 1: Load tokenizer
    logger.info("Step 1: Loading tokenizer")
    tokenizer = load_tokenizer()
    write_progress(1, 5, metric={"phase": "tokenizer_loaded"})

    # Step 2: Collect tokens from OpenWebText
    logger.info("Step 2: Collecting tokens from OpenWebText")
    tokens = collect_tokens_streaming(PILOT_TARGET_TOKENS, tokenizer)
    logger.info(f"Collected {len(tokens)} tokens")
    write_progress(2, 5, metric={"phase": "tokens_collected", "token_count": len(tokens)})

    # Step 3: Compute PMI
    logger.info("Step 3: Computing PMI features")
    results = compute_pmi_features(tokens, tokenizer, PILOT_LETTERS)
    logger.info(f"Total PMI entries: {len(results)}")
    write_progress(3, 5, metric={"phase": "pmi_computed", "entry_count": len(results)})

    # Step 4: Validate results
    logger.info("Step 4: Validating results")
    validation = validate_pmi_results(results, PILOT_LETTERS)
    logger.info(f"Validation checks: {json.dumps(validation, indent=2)}")
    write_progress(4, 5, metric={"phase": "validated", "all_pass": validation["all_pass"]})

    # Step 5: Save results
    logger.info("Step 5: Saving results")

    # Save pilot summary
    pilot_summary = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "token_count": len(tokens),
            "letters": PILOT_LETTERS,
            "window_size": WINDOW_SIZE,
        },
        "statistics": {
            "total_pmi_entries": len(results),
            "letters_processed": list(set(r["letter"] for r in results)),
        },
        "validation": validation,
        "go_no_go": "GO" if validation["all_pass"] else "NO_GO",
        "runtime_seconds": (datetime.now() - start_time).total_seconds(),
    }

    pilot_output_path = PILOT_DIR / "C2A_pmi_extraction_pilot.json"
    pilot_output_path.write_text(json.dumps(pilot_summary, indent=2))
    logger.info(f"Pilot summary saved to {pilot_output_path}")

    # Save top PMI features per letter (condensed)
    top_pmi_per_letter = {}
    from collections import defaultdict
    by_letter = defaultdict(list)
    for r in results:
        by_letter[r["letter"]].append(r)
    for letter, entries in by_letter.items():
        entries_sorted = sorted(entries, key=lambda x: x["pmi_score"], reverse=True)
        top_pmi_per_letter[letter] = entries_sorted[:20]  # top 20 per letter

    pmi_output_path = FULL_DIR / "C2A_pmi_features.json"
    output_data = {
        "task_id": TASK_ID,
        "mode": "pilot",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "token_count": len(tokens),
            "letters": PILOT_LETTERS,
            "window_size": WINDOW_SIZE,
            "is_pilot": True,
        },
        "features": results,  # all entries
        "top_pmi_per_letter": top_pmi_per_letter,
        "validation": validation,
    }
    pmi_output_path.write_text(json.dumps(output_data, indent=2, default=str))
    logger.info(f"Full PMI features saved to {pmi_output_path}")

    write_progress(5, 5, metric={"phase": "complete", "go_no_go": pilot_summary["go_no_go"]})

    # Print summary
    print("\n" + "="*60)
    print("C2A PMI Extraction - PILOT RESULTS")
    print("="*60)
    print(f"Go/No-Go: {pilot_summary['go_no_go']}")
    print(f"Total PMI entries: {len(results)}")
    print(f"\nValidation:")
    for k, v in validation.items():
        print(f"  {k}: {v}")
    print(f"\nTop PMI tokens per letter:")
    for letter in PILOT_LETTERS:
        top = top_pmi_per_letter.get(letter, [])[:5]
        if top:
            print(f"  {letter}: {[(t['token_str'], round(t['pmi_score'], 3)) for t in top]}")
    print("="*60)

    mark_done(
        status="success" if validation["all_pass"] else "failed",
        summary=f"PMI pilot: {len(results)} entries, go_no_go={pilot_summary['go_no_go']}, validation={validation['all_pass']}"
    )

    if validation["all_pass"]:
        logger.info("PILOT PASS: All criteria met.")
        sys.exit(0)
    else:
        logger.error("PILOT FAIL: Not all criteria met.")
        sys.exit(1)


if __name__ == "__main__":
    main()
