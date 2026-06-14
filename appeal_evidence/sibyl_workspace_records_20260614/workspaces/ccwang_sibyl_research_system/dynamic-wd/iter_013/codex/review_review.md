# Codex Independent Review - review

**Review Time**: 2026-03-24T22:30:41Z
**Model**: Claude Opus 4.6 (Codex MCP unavailable; fallback independent review)
**Note**: Codex MCP was not registered in this environment. This review was conducted by an independent Claude agent operating under the Codex reviewer protocol, providing a third-party assessment distinct from the internal review process.

## Review Summary

The paper introduces Equilibrium-Driven Weight Decay (EqWD), which modulates per-layer weight decay based on the deviation of the gradient-to-weight ratio from its EMA-tracked equilibrium. The core idea is clean and well-motivated by Defazio (2025)'s ratio equilibrium theory. On ImageNet ResNet-50 (45 epochs, 3 seeds), EqWD achieves 72.27 +/- 0.20% vs. FixedWD 71.89 +/- 0.24%. On CIFAR-100 ResNet-20, EqWD (65.05%) does not beat FixedWD (65.19%). The paper has undergone two revision rounds and now presents statistical claims with appropriate caution.

## Strengths

### S1. Clean, Minimal Algorithm
EqWD adds exactly two hyperparameters (beta, alpha) to standard SGDW, computes two norms per parameter group per step, and can be implemented in ~10 lines. The 2% wall-clock overhead claim is credible for these operations. This is a genuinely practical method.

### S2. Honest Statistical Reporting (Post-Revision)
The revised paper properly qualifies its claims: Cohen's d = 1.72 for EqWD vs. FixedWD (large effect), d = 0.72 for EqWD vs. SWD (medium effect, bootstrap CI includes zero). The introduction uses "tends to achieve" rather than definitive ranking claims. This level of honesty at n=3 is appropriate.

### S3. Informative Negative Results
Two negative results are genuinely valuable:
- CAWD (continuous alignment modulation) underperforms FixedWD (71.44% vs. 71.89%), demonstrating that cosine alignment is not a useful modulation signal in isolation.
- The AIS diagnostic showing MI(alignment; test_acc | norms) near zero across all convolutional layers is a clean empirical result that informs the broader alignment-aware WD literature.

### S4. Coherent Theoretical Narrative (Post-Revision)
The Proposition 2 + AIS verification structure now forms a coherent argument: ratio is sufficient *if* alignment depends only on norms (conditional claim), and AIS empirically verifies this condition. This resolves the internal contradiction present in the first draft.

### S5. Per-Layer Design is Justified
The paper's claim about layer heterogeneity in ratio dynamics is supported by the alignment diagnostic data: fc.weight in ResNet-20 has residual variance ratio = 0.62 (alignment carries information beyond norms), while all conv layers have ratios > 0.95. This motivates per-layer treatment and honestly reveals a limitation (ratio sufficiency may not hold for FC layers).

## Weaknesses

### W1. Effective Weight Decay Inflation Confound (Major)
This is the single most important unresolved issue. EqWD's modulation factor phi >= 1 always, meaning the effective average lambda over training is systematically higher than lambda_base. The paper acknowledges this clearly (Section 3.2, 5.6) but does not run the critical control experiment: FixedWD with lambda = 6e-4 or 7e-4 on ImageNet. Without this, it is impossible to determine whether EqWD's +0.38% improvement comes from *better-timed* modulation or simply *stronger average* regularization.

The CAWD partial control is acknowledged as imperfect: CAWD also modulates upward but uses a different signal (alignment), so its underperformance could reflect alignment being a bad signal rather than timing being irrelevant. A single-seed FixedWD run at higher lambda would cost ~2.5 GPU-hours and substantially resolve this ambiguity.

**Severity**: Major. This confound undermines the core claim that equilibrium-driven *timing* of weight decay is the source of improvement.

### W2. CIFAR-100 Results Undermine Generality Claims (Moderate)
EqWD at default beta=1.0 ranks 3rd on CIFAR-100 (65.05%), behind FixedWD (65.19%) and CPR (65.19%). The paper argues this is because CIFAR-100 has small ratio deviations that need higher beta, pointing to the single-seed beta=5.0 result (66.07%). However:
- A method that requires task-dependent beta tuning loses its "robust default" advantage.
- The beta=5.0 result is single-seed and could be an outlier.
- If the practitioner needs to tune beta per task, EqWD's search space is not obviously smaller than tuning lambda directly.

The paper handles this honestly, but it weakens the contribution narrative: EqWD is primarily an ImageNet-scale method with default parameters.

### W3. No 90-Epoch ImageNet Validation (Moderate)
45 epochs is half the canonical schedule. The paper's concern (Section 5.5) is correct: EqWD exploits transitional phases (warmup, cosine knee), and with 90 epochs these transitions become proportionally smaller. The improvement could shrink or vanish at standard training length. A single-seed 90-epoch run (~5 GPU-hours) would substantially strengthen the paper.

### W4. SGDW-Only, No AdamW/Transformer (Moderate)
The dominant modern training paradigm is AdamW + Transformer. The paper explicitly scopes itself to SGDW + CNNs and acknowledges this limitation. However, the ratio dynamics under Adam (where the effective gradient is divided by sqrt(v_t)) may differ fundamentally. Without at least a small-scale AdamW validation (e.g., ViT-Tiny on CIFAR-100), the method's relevance to current practice is uncertain.

### W5. Figures Missing from Paper (Moderate)
The paper references Figure 1 (ratio trajectories) and Figure 2 (WD heatmap) in Section 4.4, but the writing/figures/ directory is empty. The figures exist in exp/results/full/figures/ (ratio_trajectories.png, wd_heatmap.png, etc.) but have not been integrated into the manuscript. For a paper whose central narrative is about ratio dynamics, these visualizations are essential. The LaTeX directory is also empty, suggesting the paper is not yet compiled.

### W6. AIS Diagnostic Has a Blind Spot at FC Layers (Minor-to-Moderate)
The alignment diagnostic data reveals that fc.weight in ResNet-20 has residual_variance_ratio = 0.62 and MI = 0.077 -- meaning alignment *does* carry meaningful information beyond norms for fully-connected layers. Similarly, VGG-16-BN's classifier layers (classifier.3.weight: 0.81, classifier.6.weight: 0.86) show the same pattern. The paper's claim that "ratio is sufficient for alignment-relevant information" is thus only true for convolutional layers, not the full network. This exception is important because FC layers have a disproportionate impact on the final classification decision.

The paper mentions this for Transformers (Section 3.3, "Scope and caveats") but does not discuss the FC layer exception in the current CNN experiments. This is an omission that reviewers will catch.

### W7. Ablation Tables Use Single-Seed Results (Minor)
Beta ablation (Table 3) and EMA ablation (Table 4) are single-seed on CIFAR-100. The non-monotonic pattern (beta=0.1: 65.21, beta=0.5: 65.07, beta=1.0: 65.39) is within single-seed noise (~0.3%). The paper appropriately caveats this, but it means the "robustness across beta" claim is weakly supported. The multi-seed main experiments partially compensate (beta=1.0 achieves 65.05 +/- 0.36 in the 3-seed run, consistent with the ablation).

### W8. Budget Equivalence Analysis Incomplete (Minor)
The proposal mentions Budget Equivalence Metric (BEM) and Coupling Stability Index (CSI) as standardized metrics, but the paper does not report these. The budget_equivalence results directory exists but these metrics are not integrated into the main results table. This is a gap between the proposal's ambition (unified evaluation framework) and the paper's delivery.

## Specific Issues

1. **Table 1 (method comparison)**: The "Overhead" column lists EqWD as "~2%" but does not quantify this relative to what. 2% of total training time? 2% of forward+backward time? The measurement methodology is not described.

2. **Proposition 1 "Proof"**: This is a restatement of the definition, not a proof. Calling it a "proof sketch" for a trivially true statement may invite reviewer skepticism about theoretical rigor.

3. **Proposition 2 conditional sufficiency**: The condition "if alignment deviation is also a function of norms" is strong and non-trivial. The paper correctly treats this as an empirical question, but the proposition's statement could be tightened to make the empirical nature more prominent.

4. **Section 4.1 asymmetric tuning**: All baselines get 50 Bayesian optimization trials; EqWD uses defaults. This is a strength if EqWD wins, but the paper should report what the optimal baseline hyperparameters are. If FixedWD's optimal lambda from Optuna is already 5e-4 (the standard default), the 50-trial search did not help and the comparison is symmetric after all.

5. **Forward-dated references**: CWD is cited as "ICLR 2026." For a paper written in 2026, this is plausible but reviewers will verify the reference exists.

## Missing Experiments (Priority-Ordered)

1. **FixedWD at higher lambda on ImageNet** (highest priority, ~2.5 GPU-hours): Resolves the WD inflation confound.
2. **90-epoch ImageNet single-seed** (high priority, ~5 GPU-hours): Validates method ranking at canonical training length.
3. **Multi-seed beta=5.0 on CIFAR-100** (medium priority, ~1 GPU-hour): Validates the most promising CIFAR finding.
4. **Report average phi(t) over training** (zero cost): Quantifies effective WD inflation.
5. **AdamW + ViT-Tiny on CIFAR-100** (medium priority, ~2 GPU-hours): Tests optimizer/architecture generalization.

## Scoring

| Criterion | Score (1-10) | Weight | Notes |
|-----------|:---:|:---:|-------|
| Novelty | 6 | 20% | Clean application of ratio equilibrium to WD scheduling; incremental rather than surprising |
| Theoretical Soundness | 6 | 15% | Prop 1 is trivial; Prop 2 is conditional on an empirical assumption; AIS verification is clean |
| Experimental Rigor | 5 | 25% | WD inflation confound unresolved; n=3 limits power; 45-epoch regime untested at 90 |
| Writing Quality | 8 | 15% | Honest framing, appropriate caveats, well-structured |
| Practical Impact | 5 | 15% | SGDW+CNN only; no AdamW/Transformer; limited adoption pathway |
| Reproducibility | 7 | 10% | Algorithm is simple and fully specified; code overhead is minimal |

**Weighted Score**: 0.2*6 + 0.15*6 + 0.25*5 + 0.15*8 + 0.15*5 + 0.10*7 = 1.2 + 0.9 + 1.25 + 1.2 + 0.75 + 0.7 = **6.0/10**

## Score: 6/10

## Recommendation

**WEAK ACCEPT** with the following conditions for camera-ready:

1. Run FixedWD at lambda = {6e-4, 7e-4} on ImageNet (single seed minimum) to address the WD inflation confound.
2. Report the average modulation factor phi(t) over training for EqWD (this is zero-cost and directly quantifies the inflation).
3. Discuss the FC-layer AIS exception in Section 3.3 (the data already exists in alignment_diagnostic.json).
4. Integrate the existing figures (ratio_trajectories.png, wd_heatmap.png) into the manuscript.

The paper presents a clean, principled method with an honest evaluation. The core contribution -- translating Defazio's ratio equilibrium diagnostic into a practical WD scheduling algorithm -- is incremental but sound. The negative results (CAWD, AIS) are genuinely informative for the community. The main risk is that the improvement comes from effective WD inflation rather than better timing, which one additional experiment could resolve. The narrow scope (SGDW + CNNs) limits impact but is honestly acknowledged.

VERDICT: APPROVE
