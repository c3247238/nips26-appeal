"""
Final Analysis: Aggregate all Tier 0-2 results for the paper.

This script consolidates all experimental findings into a comprehensive
analysis document suitable for writing the paper.
"""

import json
from pathlib import Path
import sys

RESULTS_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("../results")


def load_summary(path):
    with open(path) as f:
        return json.load(f)


def load_epoch_metrics(path):
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line.strip()))
    return records


def main():
    output_dir = RESULTS_DIR / "final_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("FINAL ANALYSIS: Alignment-Aware Dynamic Weight Decay")
    print("Complete Experimental Results")
    print("=" * 80)

    # ═══════════════════════════════════════════════════════════════
    # TABLE 1: Main Results (ResNet20/CIFAR-10)
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE 1: Main Results — ResNet20/CIFAR-10 (200 epochs, seed=42)")
    print("=" * 80)

    grid_dir = RESULTS_DIR / "tier1_fixed_wd_grid"
    sc_dir = RESULTS_DIR / "tier1_stagewise_cwd"
    aadwd_dir = RESULTS_DIR / "tier1_aadwd_variants"

    main_results = {}

    # Load all tier1 results
    for sub in grid_dir.iterdir():
        if sub.is_dir() and (sub / "summary.json").exists():
            s = load_summary(sub / "summary.json")
            name = s.get("config", {}).get("name")
            if name:
                main_results[name] = s

    for sub in sc_dir.iterdir():
        if sub.is_dir() and (sub / "summary.json").exists():
            s = load_summary(sub / "summary.json")
            name = s.get("config", {}).get("name")
            if name:
                main_results[name] = s

    for sub in aadwd_dir.iterdir():
        if sub.is_dir() and (sub / "summary.json").exists():
            s = load_summary(sub / "summary.json")
            main_results[f"aadwd_{sub.name}"] = s

    order = [
        "no_wd", "fixed_wd_0.0001", "fixed_wd_0.0003", "fixed_wd_0.0005",
        "fixed_wd_0.001", "fixed_wd_0.003",
        "stagewise_wd", "cwd",
        "aadwd_conservative", "aadwd_aggressive", "aadwd_square"
    ]

    print(f"\n{'Method':<25} {'Best%':>8} {'Final%':>8} {'Gap%':>7} {'WN':>8}")
    print("-" * 60)
    table1_rows = []
    for name in order:
        if name not in main_results:
            continue
        s = main_results[name]
        row = {
            "method": name,
            "best": s["best_test_acc"],
            "final": s["final_test_acc"],
            "gap": s["final_gen_gap"],
            "wn": s["final_weight_norm"],
        }
        table1_rows.append(row)
        marker = " ★" if name == "fixed_wd_0.0005" else ""
        print(f"  {name:<23} {row['best']:>7.2f}% {row['final']:>7.2f}% "
              f"{row['gap']:>6.2f}% {row['wn']:>7.2f}{marker}")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 2: Cross-Architecture Validation
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE 2: Cross-Architecture Validation (200 epochs)")
    print("=" * 80)

    cross_dir = RESULTS_DIR / "tier2_cross_arch"
    if cross_dir.exists():
        agg = load_summary(cross_dir / "aggregate_summary.json")

        # CIFAR-100
        print("\n  ResNet20/CIFAR-100:")
        print(f"  {'Method':<30} {'Best%':>8} {'Final%':>8} {'Gap%':>7} {'WN':>8}")
        print("  " + "-" * 65)
        for name, exp in agg["experiments"].items():
            if "cifar100" in name:
                short = name.replace("resnet20_cifar100_", "")
                print(f"    {short:<28} {exp['best_test_acc']:>7.2f}% "
                      f"{exp['final_test_acc']:>7.2f}% "
                      f"{exp['final_gen_gap']:>6.2f}% {exp['final_weight_norm']:>7.2f}")

        # VGG16/CIFAR-10
        print("\n  VGG16-BN/CIFAR-10:")
        print(f"  {'Method':<30} {'Best%':>8} {'Final%':>8} {'Gap%':>7} {'WN':>8}")
        print("  " + "-" * 65)
        for name, exp in agg["experiments"].items():
            if "vgg16" in name:
                short = name.replace("vgg16_bn_cifar10_", "")
                print(f"    {short:<28} {exp['best_test_acc']:>7.2f}% "
                      f"{exp['final_test_acc']:>7.2f}% "
                      f"{exp['final_gen_gap']:>6.2f}% {exp['final_weight_norm']:>7.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 3: Ablation Study
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE 3: Ablation Study — Is the Alignment Signal Useful? (H5)")
    print("=" * 80)

    abl_dir = RESULTS_DIR / "tier2_ablations"
    if abl_dir.exists():
        agg = load_summary(abl_dir / "aggregate_summary.json")
        print(f"\n  {'Method':<30} {'Best%':>8} {'Final%':>8} {'WN':>8}")
        print("  " + "-" * 55)
        for name, exp in agg["experiments"].items():
            print(f"    {name:<28} {exp['best_test_acc']:>7.2f}% "
                  f"{exp['final_test_acc']:>7.2f}% {exp['final_weight_norm']:>7.2f}")

        # H5 test
        agg_ref = agg["experiments"].get("aadwd_aggressive_ref", {})
        random_wd = agg["experiments"].get("random_dynamic_wd", {})
        if agg_ref and random_wd:
            delta = agg_ref.get("best_test_acc", 0) - random_wd.get("best_test_acc", 0)
            print(f"\n  H5 Test: AADWD Aggressive - Random Dynamic = {delta:+.2f}%")
            print(f"  H5 Criterion: >= +0.3%")
            print(f"  H5 Result: {'PASS' if delta >= 0.3 else 'FAIL'}")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 4: Hyperparameter Sensitivity
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE 4: Hyperparameter Sensitivity (AADWD Aggressive)")
    print("=" * 80)

    hp_dir = RESULTS_DIR / "tier2_hyperparam_sensitivity"
    if hp_dir.exists():
        agg = load_summary(hp_dir / "aggregate_summary.json")

        print("\n  c sweep (beta=0.999):")
        print(f"  {'c':>6} {'Best%':>8} {'Final%':>8} {'WN':>8}")
        print("  " + "-" * 35)
        for name, exp in agg["experiments"].items():
            if name.startswith("aadwd_agg_c"):
                c = exp["config"]["c"]
                print(f"  {c:>6} {exp['best_test_acc']:>7.2f}% "
                      f"{exp['final_test_acc']:>7.2f}% {exp['final_weight_norm']:>7.2f}")

        print("\n  beta sweep (c=2.5):")
        print(f"  {'beta':>8} {'Best%':>8} {'Final%':>8} {'WN':>8}")
        print("  " + "-" * 37)
        for name, exp in agg["experiments"].items():
            if name.startswith("aadwd_agg_beta"):
                beta = exp["config"]["beta"]
                print(f"  {beta:>8} {exp['best_test_acc']:>7.2f}% "
                      f"{exp['final_test_acc']:>7.2f}% {exp['final_weight_norm']:>7.2f}")

    # ═══════════════════════════════════════════════════════════════
    # TABLE 5: LR-Decoupled AADWD
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("TABLE 5: LR-Decoupled AADWD Variants")
    print("=" * 80)

    dec_dir = RESULTS_DIR / "tier2_decoupled"
    if dec_dir.exists():
        agg = load_summary(dec_dir / "aggregate_summary.json")
        print(f"\n  {'Method':<35} {'Best%':>8} {'Final%':>8} {'WN':>8}")
        print("  " + "-" * 65)
        for name, exp in agg["experiments"].items():
            print(f"    {name:<33} {exp['best_test_acc']:>7.2f}% "
                  f"{exp['final_test_acc']:>7.2f}% {exp['final_weight_norm']:>7.2f}")

    # ═══════════════════════════════════════════════════════════════
    # KEY FINDINGS FOR PAPER
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("KEY FINDINGS FOR PAPER")
    print("=" * 80)

    findings = [
        "F1. ALIGNMENT SIGNAL IS NOT ACTIONABLE: Random dynamic WD (92.06%) matches "
        "AADWD aggressive (92.05%). The alignment proxy δ̂_t does not provide "
        "useful information for WD scheduling. H5 hypothesis REJECTED.",

        "F2. CONSERVATIVE ≈ FIXED WD: Because δ ≈ O(10⁻³), the conservative formula "
        "λ = c·γ·(1-δ) ≈ c·γ degenerates to a deterministic function of LR. "
        "Best acc matches fixed WD within 0.17% on all settings.",

        "F3. γ_t COUPLING DOMINATES DYNAMICS: All AADWD variants' λ_t trajectories "
        "are dominated by the LR multiplier γ_t. After milestone LR drops (0.1→0.01→0.001), "
        "λ drops by 10-100×, effectively eliminating WD in late training.",

        "F4. LR-DECOUPLING IS CATASTROPHIC: Removing γ_t (decoupled mode) causes "
        "aggressive to collapse (positive feedback: high δ → high λ → weight death → "
        "higher δ → repeat) and conservative to over-regularize (constant λ=5e-4 is "
        "10× too strong when LR=0.01, 100× when LR=0.001).",

        "F5. γ_t SERVES AS STABILIZER: The LR coupling is not merely theoretical "
        "convenience — it's essential for stability. It replicates the natural "
        "LR-WD coupling that L2 regularization provides for free "
        "(effective_decay = lr × wd scales automatically with LR).",

        "F6. CWD CONSISTENTLY COLLAPSES: Cautious Weight Decay shows late-training "
        "instability across all settings: CIFAR-10/ResNet20 (91.79%→86.95%), "
        "CIFAR-100/ResNet20 (66.84%→54.27%), CIFAR-10/VGG16 (92.95%→86.47%). "
        "Coordinate-wise decay creates systematic weight shrinkage.",

        "F7. FIXED WD IS OPTIMAL: Simple fixed WD (5e-4) achieves the best or "
        "near-best performance across all settings: 92.54% (CIFAR-10/ResNet20), "
        "68.45% (CIFAR-100/ResNet20), 93.86% (CIFAR-10/VGG16). "
        "No dynamic method consistently outperforms it.",

        "F8. EQUIV CUMULATIVE = FIXED WD: When dynamic WD's average matches "
        "fixed WD, performance is identical (92.54% vs 92.54%). This proves "
        "the mean WD magnitude matters, not temporal dynamics.",

        "F9. SQUARE VARIANT LR² COUPLING: λ = c·γ²·(1-δ) causes WD to drop "
        "100× at first milestone and 10000× overall. Final WN=38.75 (highest "
        "among WD methods), effectively no regularization in late training.",

        "F10. PAPER PIVOT OPPORTUNITY: Results constitute a strong negative-result "
        "paper on dynamic WD. Key message: theoretically motivated alignment-based "
        "WD adaptation fails because (a) alignment signal is too weak, "
        "(b) LR coupling dominates dynamics, (c) decoupling is catastrophic. "
        "Simple fixed WD remains optimal for nonconvex SGD."
    ]

    for f in findings:
        print(f"\n{f}")

    # ═══════════════════════════════════════════════════════════════
    # SUGGESTED PAPER STRUCTURE
    # ═══════════════════════════════════════════════════════════════
    print("\n\n" + "=" * 80)
    print("SUGGESTED PAPER STRUCTURE")
    print("=" * 80)
    structure = """
    Title: "When Theory Meets Practice: Why Alignment-Aware Dynamic Weight Decay
           Fails to Improve on Fixed Regularization"
    or: "The Alignment Mirage: A Negative-Result Study of Dynamic Weight Decay
         for Nonconvex SGD"

    1. Introduction
       - WD is ubiquitous but underexplored theoretically
       - Alignment between gradient and parameters suggests adaptive WD
       - We propose AADWD, motivated by convergence theory
       - Extensive experiments reveal alignment signal is not actionable

    2. Background & Related Work
       - Weight decay: L2 vs decoupled
       - Adaptive regularization: AdamW, CWD, etc.
       - Gradient-parameter alignment in optimization

    3. Theoretical Framework
       - Convergence bound involving alignment proxy δ̂_t
       - Three AADWD variants: conservative, aggressive, square
       - Stability analysis: role of γ_t coupling

    4. Experimental Setup
       - ResNet20/VGG16 on CIFAR-10/100
       - 10+ methods: fixed WD, stagewise, CWD, 3 AADWD variants, ablations
       - 200 epochs, MultiStepLR [80,120], seed=42

    5. Results
       5.1 Main results (Table 1): Fixed WD wins
       5.2 Ablation (Table 3): Alignment signal = random noise (H5 FAIL)
       5.3 Why conservative ≈ fixed WD (δ is O(10⁻³))
       5.4 Why aggressive underperforms (LR coupling dominates)
       5.5 LR-decoupled variants catastrophically fail (Table 5)
       5.6 Cross-architecture validation (Table 2)
       5.7 CWD instability analysis

    6. Analysis: The Alignment Mirage
       - δ̂_t is detectable but not actionable
       - γ_t coupling: stabilizer or crutch?
       - L2 regularization's hidden advantage
       - Why mean(λ_t) = fixed WD (ergodic argument)

    7. Recommendations & Conclusion
       - Use fixed WD for nonconvex SGD
       - Alignment-based adaptation needs stronger signals
       - Theoretical WD schedules may not translate to practice
    """
    print(structure)

    # Save comprehensive analysis
    analysis = {
        "table1_main_results": table1_rows,
        "findings": findings,
        "h5_result": "FAIL",
        "best_method_overall": "fixed_wd_0.0005",
    }
    with open(output_dir / "final_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)

    # DONE marker
    with open(RESULTS_DIR / "final_analysis_DONE", "w") as f:
        json.dump({"task_id": "final_analysis", "success": True}, f)

    print(f"\nFinal analysis saved to {output_dir}/final_analysis.json")


if __name__ == "__main__":
    main()
