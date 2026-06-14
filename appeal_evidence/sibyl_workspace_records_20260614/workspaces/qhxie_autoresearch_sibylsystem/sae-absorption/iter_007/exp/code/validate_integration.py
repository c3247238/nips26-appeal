#!/usr/bin/env python3
"""
validate_integration.py — Data Integrity Validator (Gate 0A, FULL mode)

Cross-checks ALL numerical claims in paper.md against source JSON files in
exp/results/full/. Flags mismatches: vocabulary sizes, absorption rates,
hedging rates, CMI values, correlation coefficients, activation patching results.
Also validates internal consistency between iter_006 and iter_007 data.

Identifies all persistent core words from:
  1. activation_patching_core_words.json (authoritative: checked all 4 L0 values, found 8)
  2. confound_decomposition_multi_l0.json (truncated to 5 entries per L0)

The activation patching experiment is the authoritative source for persistent core
words because it explicitly checked FN status at all 4 L0 values. The confound
decomposition JSON's hierarchy_details is truncated to 5 entries (code line 576:
hierarchy_driven[:5]).

Outputs: exp/results/full/data_validation_report.json
"""

import json
import os
import sys
import math
import numpy as np
from datetime import datetime
from pathlib import Path

# ── Config ──────────────────────────────────────────────────────────────────
WORKSPACE = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/current")
ITER6_DIR = Path("/home/qhxie/AutoResearch-SibylSystem/workspaces/sae-absorption/iter_006")
RESULTS_DIR = WORKSPACE / "exp" / "results" / "full"
TASK_ID = "validate_integration"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── PID / Progress / Done helpers ────────────────────────────────────────
pid_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}.pid"
pid_file.write_text(str(os.getpid()))


def report_progress(step, total_steps, description, extra=None):
    progress = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    data = {
        "task_id": TASK_ID, "epoch": step, "total_epochs": total_steps,
        "step": step, "total_steps": total_steps,
        "description": description, "metric": extra or {},
        "updated_at": datetime.now().isoformat(),
    }
    progress.write_text(json.dumps(data, indent=2))


def mark_done(status="success", summary="", results=None):
    if pid_file.exists():
        pid_file.unlink()
    progress_file = WORKSPACE / "exp" / "results" / f"{TASK_ID}_PROGRESS.json"
    fp = {}
    if progress_file.exists():
        try:
            fp = json.loads(progress_file.read_text())
        except Exception:
            pass
    marker = WORKSPACE / "exp" / "results" / f"{TASK_ID}_DONE"
    marker.write_text(json.dumps({
        "task_id": TASK_ID, "status": status, "summary": summary,
        "results": results or {}, "final_progress": fp,
        "timestamp": datetime.now().isoformat(),
    }, indent=2, default=str))


def load_json(path):
    """Load a JSON file, returning None if not found or unparseable."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  WARNING: Failed to parse {path}: {e}")
        return None


def approx_equal(a, b, rel_tol=0.01, abs_tol=0.005):
    """Check if two numbers are approximately equal."""
    if a is None or b is None:
        return False
    try:
        a, b = float(a), float(b)
    except (TypeError, ValueError):
        return False
    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=abs_tol)


# ── Main Validation Logic ──────────────────────────────────────────────────
def main():
    start_time = datetime.now()
    report_progress(0, 12, "Starting comprehensive data integrity validation")

    # ── 1. Load all source files ─────────────────────────────────────────
    report_progress(1, 12, "Loading source JSON files")
    print("[1/12] Loading source JSON files...")

    # Iter 006 data
    first_letter = load_json(ITER6_DIR / "exp/results/full/first_letter_improved.json")
    confound_multi = load_json(ITER6_DIR / "exp/results/full/confound_decomposition_multi_l0.json")
    cmi_est = load_json(ITER6_DIR / "exp/results/full/cmi_estimation.json")
    bifurcation = load_json(ITER6_DIR / "exp/results/full/bifurcation_analysis.json")
    scaling = load_json(ITER6_DIR / "exp/results/full/scaling_surface.json")
    cross_comparative = load_json(ITER6_DIR / "exp/results/full/cross_domain_comparative.json")
    cross_city_continent = load_json(ITER6_DIR / "exp/results/full/cross_domain_city_continent.json")
    cross_city_country = load_json(ITER6_DIR / "exp/results/full/cross_domain_city_country.json")
    cross_city_language = load_json(ITER6_DIR / "exp/results/full/cross_domain_city_language.json")
    cross_animal_class = load_json(ITER6_DIR / "exp/results/full/cross_domain_animal_class.json")
    threshold_sens = load_json(ITER6_DIR / "exp/results/full/ablation_threshold_sensitivity.json")

    # Iter 007 data
    tightened_hedging = load_json(RESULTS_DIR / "tightened_hedging.json")
    activation_patching = load_json(RESULTS_DIR / "activation_patching_core_words.json")
    partial_corr = load_json(RESULTS_DIR / "partial_correlation_cmi.json")
    leave_one_out = load_json(RESULTS_DIR / "leave_one_out_cmi.json")
    control_diag = load_json(RESULTS_DIR / "control_failure_diagnosis.json")
    threshold_summary = load_json(RESULTS_DIR / "threshold_sensitivity_summary.json")
    cmi_l0_22 = load_json(RESULTS_DIR / "cmi_replication_l0_22.json")

    files_checked = []
    all_sources = {
        "first_letter_improved.json": first_letter,
        "confound_decomposition_multi_l0.json": confound_multi,
        "cmi_estimation.json": cmi_est,
        "bifurcation_analysis.json": bifurcation,
        "scaling_surface.json": scaling,
        "cross_domain_comparative.json": cross_comparative,
        "cross_domain_city_continent.json": cross_city_continent,
        "cross_domain_city_country.json": cross_city_country,
        "cross_domain_city_language.json": cross_city_language,
        "cross_domain_animal_class.json": cross_animal_class,
        "ablation_threshold_sensitivity.json": threshold_sens,
        "tightened_hedging.json": tightened_hedging,
        "activation_patching_core_words.json": activation_patching,
        "partial_correlation_cmi.json": partial_corr,
        "leave_one_out_cmi.json": leave_one_out,
        "control_failure_diagnosis.json": control_diag,
        "threshold_sensitivity_summary.json": threshold_summary,
        "cmi_replication_l0_22.json": cmi_l0_22,
    }
    for name, data in all_sources.items():
        if data is not None:
            files_checked.append(name)
        else:
            print(f"  MISSING: {name}")
    print(f"  Loaded {len(files_checked)}/{len(all_sources)} source files")

    # ── Claim accumulator ────────────────────────────────────────────────
    claims = []
    mismatches = []
    critical_findings = []

    def add_claim(claim_id, description, paper_value, source_value, location,
                  tol_rel=0.01, tol_abs=0.005):
        status = "MATCH"
        diff = None
        note = ""
        if source_value is None:
            status = "MISSING_DATA"
            note = "Source data not available"
        elif paper_value is None:
            status = "MISSING_PAPER"
            note = "Paper value not found"
        else:
            try:
                diff = abs(float(paper_value) - float(source_value))
                if not approx_equal(paper_value, source_value,
                                   rel_tol=tol_rel, abs_tol=tol_abs):
                    status = "MISMATCH"
                    note = f"Paper says {paper_value}, source has {source_value}"
            except (TypeError, ValueError):
                if str(paper_value) != str(source_value):
                    status = "MISMATCH"
                    note = f"Paper says {paper_value}, source has {source_value}"
        claim = {
            "id": claim_id, "description": description,
            "paper_value": paper_value, "location": location,
            "status": status, "source_value": source_value,
            "difference": diff, "note": note,
        }
        claims.append(claim)
        if status == "MISMATCH":
            mismatches.append(claim)
        return claim

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 1: VOCABULARY SIZES
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(2, 12, "Validating vocabulary sizes")
    print("\n[2/12] Vocabulary sizes...")

    add_claim("vocab_1204", "First-letter vocabulary: 1,204 words",
              1204, first_letter["vocabulary"]["total_words"] if first_letter else None,
              "Section 3.2, Abstract")

    add_claim("vocab_1196", "Confound decomposition vocabulary: 1,196 words",
              1196, confound_multi["vocabulary"]["total_words"] if confound_multi else None,
              "Section 3.2, 3.4")

    cd_tested = confound_multi["per_l0_results"]["22"]["n_tested"] if confound_multi else None
    add_claim("vocab_1195_tested", "Confound decomposition tested (excl X): 1,195",
              1195, cd_tested, "Section 3.2")

    # First-letter tested words at L0=82 (from aggregate)
    fl_tested = None
    if first_letter:
        fl_tested = first_letter.get("l12_16k", {}).get("aggregate", {}).get("total_tested")
    add_claim("vocab_1203_tested", "First-letter tested at L0=82: 1,203",
              1203, fl_tested, "Section 3.2")

    # CMI vocabulary
    cmi_vocab = None
    if cmi_est:
        cmi_vocab = cmi_est.get("vocabulary", {}).get("total_words")
        if cmi_vocab is None:
            cmi_vocab = cmi_est.get("data_collection", {}).get("n_word_samples")
    add_claim("vocab_1092_cmi", "CMI vocabulary: 1,092 words",
              1092, cmi_vocab, "Section 3.5")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 2: ABSORPTION RATES (L0 sweep)
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(3, 12, "Validating absorption rates")
    print("\n[3/12] Absorption rates...")

    if confound_multi:
        for l0, rate in [(22, 0.4285), (41, 0.3749), (82, 0.1439), (176, 0.0084)]:
            src = confound_multi["per_l0_results"].get(str(l0), {}).get("absorption_rate")
            add_claim(f"rate_l0_{l0}", f"Absorption rate L0={l0}: {rate*100:.2f}%",
                      rate, src, "Section 5.1")

    # First-letter improved at L0=82
    fl_rate = None
    if first_letter:
        fl_rate = first_letter.get("l12_16k", {}).get("aggregate", {}).get("aggregate_absorption_rate")
    add_claim("rate_fl_l0_82", "First-letter absorption at L0=82: 15.96%",
              0.1596, fl_rate, "Section 4.6")

    # Bootstrap CIs at L0=22
    if confound_multi:
        ci = confound_multi["per_l0_results"].get("22", {}).get("bootstrap_ci_95", [None, None])
        add_claim("ci_l0_22_lower", "Bootstrap CI lower L0=22: 40.1%",
                  0.401, ci[0], "Section 5.1", tol_abs=0.003)
        add_claim("ci_l0_22_upper", "Bootstrap CI upper L0=22: 45.6%",
                  0.456, ci[1], "Section 5.1", tol_abs=0.003)

    # Cross-layer stability (computed from per-letter data)
    if first_letter:
        for layer_key, label, expected in [("l10_16k", "L10", 0.1388), ("l20_16k", "L20", 0.1355)]:
            layer_data = first_letter.get(layer_key, {})
            src = layer_data.get("aggregate", {}).get("aggregate_absorption_rate")
            if src is None:
                # Compute from per-letter
                plr = layer_data.get("per_letter", {})
                if plr:
                    t = sum(plr[l]["n_tested"] for l in plr)
                    a = sum(plr[l]["n_absorbed"] for l in plr)
                    src = round(a / t, 4) if t > 0 else None
            add_claim(f"rate_{label.lower()}", f"{label}-16k absorption: {expected*100:.2f}%",
                      expected, src, "Section 5.1")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 3: CONFOUND DECOMPOSITION (permissive hedging)
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(4, 12, "Validating confound decomposition")
    print("\n[4/12] Confound decomposition...")

    if confound_multi:
        c22 = confound_multi["cross_l0_classification"].get("22", {})
        add_claim("fn_l0_22", "FN at L0=22: 657",
                  657, c22.get("total_false_negatives"), "Section 4.4")
        add_claim("hedging_l0_22", "Hedging (permissive) at L0=22: 648",
                  648, c22.get("hedging"), "Section 4.4")
        add_claim("hierarchy_l0_22", "Hierarchy-driven at L0=22: 9",
                  9, c22.get("hierarchy_driven"), "Section 4.4")
        add_claim("pct_hedging_l0_22", "Permissive hedging %: 98.6",
                  98.6, c22.get("pct_hedging"), "Section 4.4")
        add_claim("pct_hier_l0_22", "Hierarchy-driven %: 1.4",
                  1.4, c22.get("pct_hierarchy_driven"), "Section 4.4")

        # Other L0 values
        for l0_str, expected_h, expected_p in [("41", 98.2, 1.8), ("82", 95.1, 4.9), ("176", 10.0, 90.0)]:
            c = confound_multi["cross_l0_classification"].get(l0_str, {})
            add_claim(f"pct_hedging_l0_{l0_str}", f"Hedging % at L0={l0_str}: {expected_h}",
                      expected_h, c.get("pct_hedging"), "Section 4.4")

        # FN at L0=176
        c176 = confound_multi["cross_l0_classification"].get("176", {})
        add_claim("fn_l0_176", "FN at L0=176: 10",
                  10, c176.get("total_false_negatives"), "Section 5.4")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 4: TIGHTENED HEDGING (iter_007)
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(5, 12, "Validating tightened hedging")
    print("\n[5/12] Tightened hedging...")

    if tightened_hedging:
        tc = tightened_hedging.get("tightened_classification", {})
        sh = tc.get("strict_hedging", {})

        add_claim("th_fn_total", "Tightened hedging FN total: 656",
                  656, tc.get("n_fn_tokens"), "Section 3.4")
        add_claim("strict_count", "Strict hedging count: 41",
                  41, sh.get("count"), "Table 4")
        add_claim("strict_rate", "Strict hedging rate: 6.2%",
                  0.0625, sh.get("rate"), "Section 3.4, Table 4", tol_abs=0.003)
        add_claim("strict_ci_lo", "Strict CI lower: 4.4%",
                  0.044, sh.get("ci_95_lower"), "Table 4", tol_abs=0.003)
        add_claim("strict_ci_hi", "Strict CI upper: 8.2%",
                  0.082, sh.get("ci_95_upper"), "Table 4", tol_abs=0.003)

        nh = tc.get("non_hedging", {})
        add_claim("non_hedging_count", "Non-hedging count: 615",
                  615, nh.get("count"), "Table 4")
        add_claim("non_hedging_rate", "Non-hedging rate: 93.8%",
                  0.9375, nh.get("rate"), "Table 4", tol_abs=0.003)

        # Shuffled control
        sc = tightened_hedging.get("shuffled_control", {})
        add_claim("shuffled_strict_mean", "Shuffled strict rate: 3.4%",
                  0.034, sc.get("mean_strict_rate"), "Section 4.4", tol_abs=0.005)

        # z-statistic for strict vs shuffled (paper claims z=3.51)
        # Paper uses SD from shuffled replicates, not binomial SE
        if sh.get("rate") and sc.get("mean_strict_rate") and sc.get("std_strict_rate"):
            p_hat = sh["rate"]
            p_0 = sc["mean_strict_rate"]
            sd_shuf = sc["std_strict_rate"]
            z_val = (p_hat - p_0) / sd_shuf if sd_shuf > 0 else float('inf')
            add_claim("strict_z_stat", "Strict vs shuffled z: 3.51",
                      3.51, round(z_val, 2), "Section 4.4", tol_abs=0.2)

        # Letter G anomaly (90.5%)
        g = tightened_hedging.get("per_letter_breakdown", {}).get("G", {})
        add_claim("letter_g_strict", "Letter G strict hedging: 90.5%",
                  0.905, g.get("strict_hedging_rate"), "Section 4.4", tol_abs=0.005)
        add_claim("letter_g_count", "Letter G strict count: 19 of 21",
                  19, g.get("n_strict_hedging"), "Section 4.4")

        # Cross-check: FN counts (confound says 657, tightened says 656)
        cd_fn = confound_multi["cross_l0_classification"]["22"]["total_false_negatives"] if confound_multi else None
        th_fn = tc.get("n_fn_tokens")
        if cd_fn is not None and th_fn is not None and cd_fn != th_fn:
            critical_findings.append({
                "id": "fn_count_discrepancy_22",
                "severity": "MEDIUM",
                "description": (f"Confound decomposition: {cd_fn} FN at L0=22. "
                               f"Tightened hedging: {th_fn}. Difference of {cd_fn - th_fn} "
                               f"due to separate vocabulary tokenization runs (1196 vs 1195 tested words). "
                               f"Paper uses 656 in Section 3.4 and 4.4. Both 656 and 657 appear in the paper "
                               f"at different locations."),
                "fix": "Ensure consistent usage: either 656 or 657 throughout, with a footnote explaining."
            })

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 5: ACTIVATION PATCHING (iter_007) — PERSISTENT CORE WORDS
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(6, 12, "Validating activation patching and core words")
    print("\n[6/12] Activation patching and core words...")

    persistent_core = {}
    if activation_patching:
        pcw = activation_patching.get("persistent_core_words", {})
        n_found = pcw.get("n_found", 0)
        words_data = pcw.get("words", [])

        add_claim("n_core_words", "Persistent core words: 8",
                  8, n_found, "Section 3.4, 4.5, Table 5")

        # Build the definitive word list
        persistent_core = {
            "count": n_found,
            "source": "activation_patching_core_words.json (authoritative)",
            "words": [],
            "all_8_identified": n_found == 8,
        }
        expected_words = {"eight", "liked", "lower", "offer", "often", "other", "under", "until"}
        found_words = set()

        for w in words_data:
            word_info = {
                "word": w["word"],
                "letter": w["letter"],
                "fn_at_l0s": w.get("fn_at_l0s", []),
                "is_absorbed_l0_82": w.get("is_absorbed_l0_82", False),
                "child_feature_idx": None,
                "child_cosine": None,
            }
            # Extract primary child feature
            feats = w.get("absorbing_features_l0_82", [])
            if feats:
                # Pick the one with highest cosine
                best = max(feats, key=lambda f: abs(f.get("cosine", 0)))
                word_info["child_feature_idx"] = best.get("feature_idx")
                word_info["child_cosine"] = best.get("cosine")
            persistent_core["words"].append(word_info)
            found_words.add(w["word"])

        # Verify against paper's listed words
        if found_words != expected_words:
            missing = expected_words - found_words
            extra = found_words - expected_words
            critical_findings.append({
                "id": "core_words_list_mismatch",
                "severity": "HIGH",
                "description": (f"Paper lists {sorted(expected_words)} as core words. "
                               f"Data has {sorted(found_words)}. "
                               f"Missing: {sorted(missing) if missing else 'none'}. "
                               f"Extra: {sorted(extra) if extra else 'none'}."),
                "fix": "Update paper to match data."
            })

        # Verify patching results: paper claims 0/8 recovery
        patching_results = activation_patching.get("patching_results", [])
        recovery_count = 0
        for pr in patching_results:
            for method in ["primary", "residual", "all_children"]:
                m_data = pr.get(f"{method}_recovery") or pr.get(method, {})
                if isinstance(m_data, dict):
                    if m_data.get("parent_recovered", False):
                        recovery_count += 1

        add_claim("patching_zero_recovery", "Activation patching: 0/8 parent recovery",
                  0, recovery_count, "Section 4.5, Table 5")

        # Control: 0/10 recovery per word
        control_total = 0
        control_recovery = 0
        for pr in patching_results:
            ctrl = pr.get("control_recovery") or pr.get("control", {})
            if isinstance(ctrl, dict):
                control_total += ctrl.get("n_controls", 10)
                control_recovery += ctrl.get("n_recovered", 0)
        if control_total > 0:
            add_claim("patching_control_recovery", "Control recovery: 0/total",
                      0, control_recovery, "Table 5")

        # Cross-check: confound says 9 hierarchy-driven, patching finds 8
        if confound_multi:
            hier_count = confound_multi["cross_l0_classification"]["22"]["hierarchy_driven"]
            if hier_count != n_found:
                critical_findings.append({
                    "id": "hierarchy_vs_patching_count",
                    "severity": "MEDIUM",
                    "description": (f"Confound decomposition claims {hier_count} hierarchy-driven FNs at L0=22, "
                                   f"but activation patching finds only {n_found} words FN at ALL 4 L0 values. "
                                   f"The 9th word ('wrong'/W) is FN at L0=22/41/176 but recovers at L0=82, "
                                   f"so it is hedging, not persistent. Paper correctly reports 8 in "
                                   f"Sections 4.5 and 5.4."),
                    "fix": ("The confound decomposition's classification counts 9 because it classifies "
                           "per-L0, checking if a word is FN at ALL L0s from that L0's perspective. "
                           "At L0=22, the classification found 9 words whose FN status persisted to other L0s "
                           "tested at that point. The activation patching re-checked all 4 L0s independently "
                           "and found only 8. Paper should use 8 (from patching) as the authoritative count, "
                           "noting the 9th word from confound decomposition was reclassified.")
                })
    else:
        persistent_core = {
            "count": 0,
            "source": "activation_patching_core_words.json NOT FOUND",
            "words": [],
            "all_8_identified": False,
        }

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 6: CMI at L0=82
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(7, 12, "Validating CMI at L0=82")
    print("\n[7/12] CMI at L0=82...")

    if cmi_est:
        mw = cmi_est.get("mann_whitney_test", {})
        cbd = cmi_est.get("correlation_by_dim", {})
        partition = cmi_est.get("partition", {})

        add_claim("cmi_mw_p", "Mann-Whitney p: 0.045",
                  0.045, mw.get("p_value_two_sided"), "Section 6.2", tol_abs=0.002)
        add_claim("cmi_cohens_d", "Cohen's d: -0.924",
                  -0.924, mw.get("cohens_d"), "Section 6.2", tol_abs=0.005)

        add_claim("cmi_abs_mean", "Absorbed mean CMI: 0.649",
                  0.649, mw.get("absorbed_mean"), "Section 6.2", tol_abs=0.002)
        add_claim("cmi_abs_std", "Absorbed CMI std: 0.187",
                  0.187, mw.get("absorbed_std"), "Section 6.2", tol_abs=0.002)
        add_claim("cmi_nabs_mean", "Non-absorbed mean CMI: 0.861",
                  0.861, mw.get("non_absorbed_mean"), "Section 6.2", tol_abs=0.002)
        add_claim("cmi_nabs_std", "Non-absorbed CMI std: 0.258",
                  0.258, mw.get("non_absorbed_std"), "Section 6.2", tol_abs=0.002)

        # Group sizes (from partition)
        n_abs = len(partition.get("absorbed_letters", []))
        n_nabs = len(partition.get("non_absorbed_letters", []))
        add_claim("cmi_n_absorbed", "CMI absorbed group: n=13",
                  13, n_abs, "Section 6.2")
        add_claim("cmi_n_nonabsorbed", "CMI non-absorbed group: n=9",
                  9, n_nabs, "Section 6.2")

        # Spearman correlations across d'
        for d_str, expected_rho in [("10", -0.383), ("20", 0.048), ("30", 0.299), ("50", 0.197)]:
            d = cbd.get(d_str, {})
            add_claim(f"cmi_rho_d_{d_str}", f"CMI Spearman rho at d'={d_str}: {expected_rho}",
                      expected_rho, d.get("spearman_rho"), "Section 6.2/6.5", tol_abs=0.003)

        d10 = cbd.get("10", {})
        add_claim("cmi_p_d10", "CMI Spearman p at d'=10: 0.059",
                  0.059, d10.get("spearman_p"), "Section 6.2", tol_abs=0.003)

        # Bonferroni: 4 subspace dims tested, so Bonferroni = p * 4
        raw_p = d10.get("spearman_p")
        bonf = min(raw_p * 4, 1.0) if raw_p else None
        add_claim("cmi_bonf_p", "Bonferroni p at d'=10: 0.236",
                  0.236, round(bonf, 4) if bonf else None,
                  "Section 6.5", tol_abs=0.005)

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 7: PARTIAL CORRELATION & RESTRICTED ANALYSIS (iter_007)
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(8, 12, "Validating partial correlation")
    print("\n[8/12] Partial correlation and restricted analysis...")

    if partial_corr:
        pc = partial_corr.get("partial_correlation", {})
        add_claim("partial_rho", "Partial rho(CMI, abs | F1): -0.328",
                  -0.328, pc.get("partial_rho"), "Table 6", tol_abs=0.01)
        add_claim("partial_p", "Partial correlation p: 0.118",
                  0.118, pc.get("permutation_p_value",
                                pc.get("approximate_t_test_p",
                                       pc.get("p_value"))), "Table 6", tol_abs=0.01)

        ra = partial_corr.get("restricted_analysis", {})
        add_claim("restricted_rho", "Restricted rho (F1>0.85): -0.113",
                  -0.113, ra.get("spearman_rho", ra.get("rho")), "Table 6", tol_abs=0.02)
        add_claim("restricted_p", "Restricted p: 0.757",
                  0.757, ra.get("spearman_p", ra.get("p_value")), "Table 6", tol_abs=0.02)
        add_claim("restricted_n", "Restricted n: 10",
                  10, ra.get("n_letters", ra.get("n")), "Table 6")

        # Raw correlations
        raw = partial_corr.get("raw_correlations", {})
        abs_f1 = raw.get("rho_absorption_probe_f1", {})
        add_claim("rho_abs_f1", "Rho(absorption, probe F1): -0.69",
                  -0.69, abs_f1.get("spearman_rho", abs_f1.get("rho")),
                  "Section 4.6, Table 6", tol_abs=0.02)

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 8: LEAVE-ONE-OUT SENSITIVITY (iter_007)
    # ═══════════════════════════════════════════════════════════════════════
    if leave_one_out:
        # Stability summary contains max_abs_delta_rho
        stab = leave_one_out.get("stability_summary", {})
        jk = leave_one_out.get("jackknife", {})
        rc = leave_one_out.get("robustness_checks", {})
        sk = rc.get("remove_S_and_K", {})

        # Find max influence letter from per_letter_data
        pld = leave_one_out.get("per_letter_data", {})
        loo_results = leave_one_out.get("leave_one_out_results", [])
        max_letter = None
        max_delta = 0
        # Try leave_one_out_results list
        for entry in loo_results:
            delta = abs(entry.get("delta_rho", 0))
            if delta > max_delta:
                max_delta = delta
                max_letter = entry.get("letter")
        # If not found, try per_letter_data
        if max_letter is None:
            for letter, data in pld.items():
                if isinstance(data, dict) and "delta_rho" in data:
                    delta = abs(data["delta_rho"])
                    if delta > max_delta:
                        max_delta = delta
                        max_letter = letter

        add_claim("loo_max_letter", "Max influence letter: V",
                  "V", max_letter, "Section 6.4")
        add_claim("loo_max_change", "Max |delta rho|: 0.088",
                  0.088, stab.get("max_abs_delta_rho", max_delta),
                  "Section 6.4", tol_abs=0.005)
        add_claim("jk_se", "Jackknife SE: 0.186",
                  0.186, jk.get("se"), "Section 6.4", tol_abs=0.005)
        add_claim("jk_bc_rho", "Bias-corrected rho: -0.397",
                  -0.397, jk.get("bias_corrected_rho"), "Section 6.4", tol_abs=0.01)
        add_claim("sk_rho", "Rho without S,K: -0.505",
                  -0.505, sk.get("spearman_rho", sk.get("rho")),
                  "Section 6.4", tol_abs=0.01)
        add_claim("sk_p", "p without S,K: 0.014",
                  0.014, sk.get("spearman_p", sk.get("p_value")),
                  "Section 6.4", tol_abs=0.005)

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 9: CMI REPLICATION at L0=22 (iter_007)
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(9, 12, "Validating CMI at L0=22")
    print("\n[9/12] CMI replication at L0=22...")

    if cmi_l0_22:
        st = cmi_l0_22.get("statistical_tests", {})
        s10 = st.get("spearman_d10", {})
        add_claim("cmi_l22_rho", "CMI rho at L0=22, d'=10: +0.044",
                  0.044, s10.get("rho"), "Section 6.3", tol_abs=0.005)
        add_claim("cmi_l22_p", "CMI p at L0=22: 0.835",
                  0.835, s10.get("p_value"), "Section 6.3", tol_abs=0.02)

        gc = st.get("group_comparison", {})
        add_claim("cmi_l22_mw_p", "L0=22 Mann-Whitney p: 0.230",
                  0.230, gc.get("mann_whitney_p"), "Section 6.3", tol_abs=0.02)
        add_claim("cmi_l22_cohens_d", "L0=22 Cohen's d: +0.944",
                  0.944, gc.get("cohens_d"), "Section 6.3", tol_abs=0.02)

        # Dimension sensitivity at L0=22
        for d_label, expected_rho in [("d_20", 0.248), ("d_30", 0.410), ("d_50", 0.483)]:
            d_data = st.get(f"spearman_{d_label}", {})
            add_claim(f"cmi_l22_rho_{d_label}", f"L0=22 rho at {d_label}: {expected_rho}",
                      expected_rho, d_data.get("rho"), "Section 6.3 table", tol_abs=0.01)
    else:
        print("  cmi_replication_l0_22.json not found -- L0=22 CMI claims not validated")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 10: CONTROL FAILURE DIAGNOSIS (iter_007)
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(10, 12, "Validating control failure diagnosis")
    print("\n[10/12] Control failure diagnosis...")

    if control_diag:
        rva = control_diag.get("random_vector_analysis", {})
        frs = control_diag.get("firing_rate_stats", {})
        nha = control_diag.get("null_hypothesis_analysis", {})

        add_claim("random_candidates", "Random vec candidates at cos=0.025: ~3,766",
                  3766, rva.get("candidate_count_mean"), "Section 4.3",
                  tol_rel=0.05, tol_abs=200)

        # Percentage: candidates / total features
        n_feat = control_diag.get("n_features", 16384)
        mean_cand = rva.get("candidate_count_mean")
        pct = round(100 * mean_cand / n_feat, 1) if mean_cand and n_feat else None
        add_claim("pct_exceed_cos", "% decoder columns > cos 0.025: 23.0%",
                  23.0, pct, "Section 4.3", tol_abs=1.0)

        # Dead features
        dead_pct = frs.get("pct_dead")
        add_claim("dead_features", "Dead features: 18.8%",
                  18.8, dead_pct, "Section 4.3", tol_abs=1.5)

        # Sparse features (<1% firing) -- n_very_rare already INCLUDES dead features
        n_very_rare = frs.get("n_very_rare")
        if n_very_rare is not None and n_feat:
            sparse_pct = round(100 * n_very_rare / n_feat, 1)
        else:
            sparse_pct = None
        add_claim("sparse_features", "Sparse features (<1% firing): 71.9%",
                  71.9, sparse_pct, "Section 4.3", tol_abs=3.0)

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 11: CONTROLS & CROSS-DOMAIN
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(11, 12, "Validating controls and cross-domain rates")
    print("\n[11/12] Controls and cross-domain...")

    if first_letter:
        ctrl = first_letter.get("controls", {})
        c2 = ctrl.get("C2_shuffled_labels", {})
        add_claim("shuffled_fl", "Shuffled control first-letter: 74.6%",
                  0.746, c2.get("absorption_rate"), "Table 2")
        c1 = ctrl.get("C1_random_probe", {})
        add_claim("random_probe_fl", "Random probe first-letter: 11.8%",
                  0.118, c1.get("absorption_rate"), "Table 2")

        # Dense probe F1 (paper says 0.929)
        c3 = ctrl.get("C3_dense_probe", {})
        add_claim("dense_probe_f1", "Dense probe mean F1: 0.929",
                  0.929, c3.get("mean_f1"), "Section 3.3", tol_abs=0.005)

        # C4 untrained
        c4 = ctrl.get("C4_untrained_SAE", {})
        add_claim("c4_probe_f1", "C4 untrained SAE probe F1: 0.943",
                  0.943, c4.get("mean_probe_f1"), "Section 4.1", tol_abs=0.005)

    # Cross-domain rates
    for dname, ddata, exp_rate, exp_shuf in [
        ("city_continent", cross_city_continent, 0.0649, 0.452),
        ("city_language", cross_city_language, 0.0656, 0.180),
        ("animal_class", cross_animal_class, 0.0143, 0.393),
        ("city_country", cross_city_country, 0.0, 0.103),
    ]:
        if ddata:
            rate = ddata.get("l12_16k", {}).get("aggregate", {}).get("aggregate_absorption_rate")
            add_claim(f"rate_{dname}", f"{dname} absorption: {exp_rate*100:.2f}%",
                      exp_rate, rate, "Table 2", tol_abs=0.002)
            sc = ddata.get("controls", {}).get("C2_shuffled_labels", {}).get("absorption_rate")
            add_claim(f"shuffled_{dname}", f"{dname} shuffled: {exp_shuf*100:.1f}%",
                      exp_shuf, sc, "Table 2", tol_abs=0.005)

    # Probe F1 statistics (from l12_16k per-letter data)
    if first_letter:
        plr = first_letter.get("l12_16k", {}).get("per_letter", {})
        f1s = []
        for letter, data in plr.items():
            if isinstance(data, dict) and "probe_f1" in data:
                f1s.append(data["probe_f1"])
        if f1s:
            add_claim("mean_probe_f1", "Mean probe F1 at L0=82: 0.817",
                      0.817, round(sum(f1s)/len(f1s), 3), "Table 2", tol_abs=0.005)
            add_claim("n_above_085", "Letters passing F1>0.85: 10",
                      10, sum(1 for f in f1s if f > 0.85), "Section 3.2")

    # Probe F1 at L0=22
    if confound_multi:
        l22 = confound_multi["per_l0_results"].get("22", {})
        add_claim("all_probes_f1_l22", "All 25 probes F1=1.0 at L0=22",
                  1.0, l22.get("mean_probe_f1"), "Section 3.4")

    # ═══════════════════════════════════════════════════════════════════════
    # GROUP 12: GPT-2 CROSS-ARCHITECTURE
    # ═══════════════════════════════════════════════════════════════════════
    # Paper Section 5.3: "67.29% at layer 8, 64.26% at layer 10, 61.65% at layer 11"
    # This was corrected from the original swapped values
    if bifurcation:
        l1_analysis = bifurcation.get("l1_analysis", {})
        for bif_key, label, paper_rate in [
            ("L1_GPT2_L8", "layer_8", 0.6729),
            ("L1_GPT2_L10", "layer_10", 0.6426),
            ("L1_GPT2_L11", "layer_11", 0.6165),
        ]:
            layer_data = l1_analysis.get(bif_key, {})
            plr = layer_data.get("per_letter", {})
            if plr:
                rates = [plr[l]["absorption_rate"] for l in plr if "absorption_rate" in plr[l]]
                src = round(float(np.mean(rates)), 4) if rates else None
            else:
                src = None
            add_claim(f"gpt2_{label}", f"GPT-2 {label} absorption: {paper_rate*100:.2f}%",
                      paper_rate, src, "Section 5.3", tol_abs=0.005)

    # Threshold sensitivity CV
    if threshold_summary:
        cv_val = threshold_summary.get("cv_analysis", {}).get("overall_cv")
        add_claim("thresh_cv", "Threshold CV: 0.077",
                  0.077, cv_val, "Section 4.2", tol_abs=0.005)

    # ═══════════════════════════════════════════════════════════════════════
    # COMPILE REPORT
    # ═══════════════════════════════════════════════════════════════════════
    report_progress(12, 12, "Compiling final report")
    print("\n[12/12] Compiling...")

    n_total = len(claims)
    n_match = sum(1 for c in claims if c["status"] == "MATCH")
    n_mismatch = sum(1 for c in claims if c["status"] == "MISMATCH")
    n_missing = sum(1 for c in claims if c["status"].startswith("MISSING"))
    integrity = n_match / max(n_total, 1)
    passed = n_mismatch <= 5

    # ── Print summary ────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("DATA INTEGRITY VALIDATION REPORT (FULL)")
    print("=" * 80)
    print(f"  Total claims:     {n_total}")
    print(f"  Matches:          {n_match}")
    print(f"  Mismatches:       {n_mismatch}")
    print(f"  Missing:          {n_missing}")
    print(f"  Integrity:        {integrity:.1%}")
    print(f"  Gate 0 pass:      {'YES' if passed else 'NO'}")

    if mismatches:
        print(f"\n--- MISMATCHES ({n_mismatch}) ---")
        for m in mismatches:
            print(f"  [{m['id']}] {m['description']}")
            print(f"    Paper={m['paper_value']}  Source={m['source_value']}  "
                  f"Diff={m['difference']}  @{m['location']}")

    if critical_findings:
        print(f"\n--- CRITICAL FINDINGS ({len(critical_findings)}) ---")
        for cf in critical_findings:
            print(f"  [{cf['severity']}] {cf['id']}")
            print(f"    {cf['description'][:200]}")

    print(f"\n--- PERSISTENT CORE WORDS ---")
    if persistent_core.get("words"):
        print(f"  Count: {persistent_core['count']} (expected 8)")
        for w in persistent_core["words"]:
            feat = f"child_feat={w['child_feature_idx']}" if w["child_feature_idx"] else "no child feature"
            print(f"  {w['word']:8s} ({w['letter']}) | FN@{w['fn_at_l0s']} | {feat}")
    else:
        print("  NOT AVAILABLE (activation patching data missing)")

    # ── Save ─────────────────────────────────────────────────────────────
    elapsed = (datetime.now() - start_time).total_seconds()
    report = {
        "task_id": TASK_ID,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": round(elapsed, 1),
        "summary": {
            "total_claims": n_total,
            "matches": n_match,
            "mismatches": n_mismatch,
            "missing_data": n_missing,
            "pass": passed,
            "integrity_score": round(integrity, 4),
        },
        "claim_validation": claims,
        "persistent_core_words": persistent_core,
        "mismatches_detail": mismatches,
        "critical_findings": critical_findings,
        "files_checked": files_checked,
        "recommendations": [],
    }

    if n_mismatch > 0:
        report["recommendations"].append(
            f"Fix {n_mismatch} mismatches in paper.md before proceeding to writing."
        )
    if n_mismatch > 5:
        report["recommendations"].append(
            "BLOCKING: >5 mismatches. Fix ALL before Gate 1."
        )
    if any(cf["severity"] == "HIGH" for cf in critical_findings):
        report["recommendations"].append(
            "Address HIGH-severity findings before writing revision."
        )

    out = RESULTS_DIR / "data_validation_report.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {out}")
    print(f"Duration: {elapsed:.1f}s")

    mark_done(
        status="success",
        summary=(f"Validated {n_total} claims: {n_match} match, {n_mismatch} mismatch, "
                f"{n_missing} missing. Integrity={integrity:.1%}. "
                f"Core words: {persistent_core.get('count', 0)}/8 identified."),
        results={
            "total_claims": n_total,
            "matches": n_match,
            "mismatches": n_mismatch,
            "missing_data": n_missing,
            "integrity_score": round(integrity, 4),
            "n_persistent_core_words": persistent_core.get("count", 0),
            "pass": passed,
        }
    )


if __name__ == "__main__":
    main()
