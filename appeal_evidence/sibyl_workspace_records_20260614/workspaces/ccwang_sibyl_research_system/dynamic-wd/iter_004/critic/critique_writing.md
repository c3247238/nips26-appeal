# Writing Critique

**Date**: 2026-03-18
**Critic**: sibyl-heavy (critic role)
**Iteration**: 7 (updated)

---

## Overall Assessment: Good Writing, Multiple Unresolved Inconsistencies

The paper's prose quality is above average. Claims are specific and backed by numbers. The limitations section is admirably comprehensive. Statistical honesty statements in Sections 5.3, 6.1, and 6.2 are exemplary. However, the paper contains several unresolved factual inconsistencies that a reviewer will catch, and one critical scientific integrity issue (Figure 4).

---

## Critical Issue: Figure 4 Uses Reconstructed Trajectories as Mechanistic Evidence

The Figure 4 footer states: "Note: Trajectories are illustrative; final norms are documented experimental values."

Section 6.4 uses these trajectories as evidence for the l_inf constraint mechanism: "all seven methods converge to weight norms in the narrow range 95.89–97.04 (only 1.2% variation), despite 10x variation in effective BEM. This empirically confirms the l_inf constraint mechanism: AdamW's implicit constraint absorbs weight decay perturbations, driving all trajectories to a common neighborhood."

The visual argument — convergence trajectories smoothly approaching a common basin — rests entirely on the reconstructed curves, not on measured data. The "final norms" that are documented real values provide the endpoint claim (1.2% variation) but do not support the trajectory argument. A reviewer reading this will immediately ask: what are the actual measured trajectories? Do they converge smoothly or erratically? Are the trajectories from the paper monotone because the interpolation forced it?

The epoch_metrics.jsonl files exist in the workspace and contain measured weight norms per epoch. There is no justification for using reconstructed trajectories when actual data is available.

**Fix**: Regenerate Figure 4 from epoch_metrics.jsonl. If the actual trajectories are noisy or non-monotone, report them honestly. If the paper cannot access the actual training trajectory data, replace Figure 4 with a bar chart of final weight norms (the documented real values), and remove any claim about the convergence path.

---

## Critical Issue: Cohen's d Formula Mislabeled

Section 5.3 states: "Cohen's d is computed as the paired formula d = delta_bar / s_delta where delta_bar is the mean difference and s_delta is the standard deviation of differences."

Actual values in Tables 2 and 3 match the UNPAIRED POOLED formula: d = (mu1 - mu2) / sqrt((s1^2 + s2^2) / 2).

Verification (from actual seed data and workspace analysis JSON):
- SGD SWD vs. constant: paired d = 2.04, unpaired pooled d = 3.478 → paper reports 3.48 (unpaired pooled)
- SGD no_wd vs. constant: paired d = 12.17, unpaired pooled d = 10.287 → paper reports 10.29 (unpaired pooled)
- Both tables use the same (unpaired pooled) formula throughout

The hypotheses.md file itself distinguishes both values: "Cohen's d (paired) = -12.17; pooled d = -10.29." The paper reports the pooled value but labels it as "paired." This matters for cross-study comparability: researchers comparing Cohen's d values between papers need to know which formula was used.

**Fix**: Change Section 5.3 to: "Cohen's d uses the standard unpaired pooled formula d = (mu1-mu2)/sqrt((s1^2+s2^2)/2), consistent with conventional effect-size reporting for cross-study comparability. Statistical tests use the paired t-test to exploit the matched-seed design." Also correct Table 3 cwd_hard d from 1.08 to 1.13 (JSON: 1.1327).

---

## Major Issue: SGD SWD Significance Is Contradictory Across Sections

**experiments.md, Section 6.2, Table 3**: Lists SWD p_adj=0.054 (not significant), states "only one pairwise comparison achieves significance: constant vs. no_wd."

**paper.md, Abstract and Section 1.4**: States "two of six comparisons achieve significance after Holm correction (constant vs. no_wd: d=10.29, p_adj=0.002; constant vs. swd: d=3.48, p_adj=0.004)."

These irreconcilable statements exist in the same document bundle and reflect two different Holm family sizes (6 within-dataset vs. 11 cross-dataset comparisons). A reviewer comparing the abstract to the experiment section will flag this immediately.

**Fix**: Choose ONE statistical framing (either 6 within-dataset or 11 cross-dataset), document which is primary, and apply consistently throughout abstract, contributions, experiments, and conclusion.

---

## Major Issue: Figure 3 CIFAR-100 Panel Title Is Factually Wrong

The CIFAR-100 panel of Figure 3 shows Pearson r=0.48 with a visible upward trend line, but the subtitle reads "No correlation in Regime I." r=0.48 is a moderate positive correlation, not "no correlation." The "no correlation" characterization applies only to the CIFAR-10 panel (r=-0.05).

**Fix**: Update CIFAR-100 panel title to "CIFAR-100: moderate trend (r=0.48, non-significant at n=3)."

---

## Major Issue: Figure/Section Cross-Reference Errors

Section 6.4 cites "Figure 5" for weight norm convergence data. Figure 5 is the AIS distribution; Figure 4 is the weight norm trajectories. This is a stale cross-reference.

The Figures section of paper.md (if present) may contain further stale filenames (fig2_accuracy_comparison.png, fig4_diagnostic_heatmap.png, etc.) that do not match actual generated files (figure1_adamw_distributions.png through figure5_ais_distribution.png). The order also appears swapped: generated Figure 4 = weight norms, Figure 5 = AIS, while stale references may have these reversed.

**Fix**: Search paper.md for all "Figure N" citations and verify against actual figure content. Update the Figures section with correct filenames.

---

## Minor Notation Issues

**phi argument inconsistency**: Abstract and Section 1.4 write phi(t, theta, g) using raw gradient g; formal definition in Section 3.1 uses phi(t, theta_t, s_t) where s_t is optimizer state. Table 1 entries mix g (for SWD) and u_t (for CWD). The general symbol s_t is correct; the abstract should use s_t with a parenthetical clarification.

**"Remark 1" vs "Observation 1" mismatch**: Section 1.3 forward-references "Remark 1, Section 4" for the dual characterization; Section 4.1 labels the same item "Observation 1." One must be corrected.

**rho definition uses lambda vs. lambda-bar inconsistently**: Section 4.1 correctly defines rho = lambda-bar / eta (with lambda-bar = lambda * E[phi]). Sections 5.2 and the abstract use rho = lambda / eta. At standard settings phi=1, these are equal; but the inconsistency will confuse readers applying the framework to non-standard settings.

**Method naming**: "cosine_schedule," "cosine schedule," "Cosine WD," and "cosine annealing" all refer to the same method across the paper. Similarly "no_wd," "no-WD," "no weight decay," and "no-weight-decay." Standardize to "cosine-WD" (prose) / "cosine_schedule" (table headers); "no-WD" (prose) / "no_wd" (table headers).

**Section 1.3 AdamW/SGD setting conflation**: The parenthetical "(AdamW: ..., rho=0.5; SGD: ..., rho=0.05)" appears within a sentence claiming "at AdamW settings, seven strategies are statistically indistinguishable," making rho_SGD=0.05 appear to describe the AdamW operating point.

---

## What Works Well

1. **Abstract is dense and accurate**. Every sentence conveys information. The abstract correctly scopes claims to "batch-normalized architectures at CIFAR scale" and explicitly flags the BN ablation as blocking.

2. **Limitations section is thorough and prioritized** (Section 7.3). Ordering limitations by blocking priority (P0 > P1 > P2) with explicit cross-references to planned experiments is rare and reviewer-friendly.

3. **Falsification criteria are explicit** (Section 4.2, Predictions 1-3). Stating "if any dynamic WD strategy shows >1% improvement over constant at rho=0.5 with n>=7 seeds, Regime I is falsified" is pre-registration-style thinking.

4. **Figures 1 and 3 together make the core claim visually compelling**. Figure 1's tightly overlapping error bars for all 7 AdamW methods, combined with Figure 3's CIFAR-10 panel (r=-0.05, completely flat BEM vs. accuracy), are stronger evidence than any table.

5. **Statistical honesty statements** (Sections 5.3, 6.1, 6.2) correctly distinguish non-significance from proven equivalence. The TOST power ~15-20% disclosure and MDE ~0.77% quantification are examples of scientific rigor that should be standard but rarely appear.

---

## Priority Fix Order

1. Figure 4: regenerate from actual epoch_metrics.jsonl or replace with bar chart of final norms
2. Cohen's d: fix Section 5.3 label and correct cwd_hard d=1.08→1.13
3. SGD swd significance: choose one statistical framing and apply consistently
4. Figure 3 CIFAR-100 panel title: "No correlation" → "moderate trend (r=0.48)"
5. Figure cross-references: fix Figure 4/5 swap in Section 6.4
6. Phi argument notation: standardize to s_t with parenthetical in abstract
7. Remark 1 / Observation 1 naming: pick one
8. Method naming: standardize cosine-WD and no-WD
