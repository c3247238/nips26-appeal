# Codex 独立评审 - review

**评审时间**: 2026-04-02T06:33:37Z
**模型**: sibyl-light (Sonnet 4.6) — Codex MCP unavailable; independent review conducted by sibyl-codex-reviewer agent as documented fallback

> NOTE: The `mcp__codex__codex` tool is not registered in this session (confirmed by `claude mcp list` — only context7 and serena are connected). This review is conducted by the sibyl-light reviewer agent as an independent third-party assessment. The review provides a differentiated perspective from the existing internal `writing/review.md`.

---

## Review Scope

This review covers the final paper draft at `writing/paper.md` (505 lines, 7 sections + abstract) and cross-checks against available experimental artifacts in `exp/results/full/`. The focus is on issues not fully addressed in the existing internal review: scientific validity, theoretical claims, reproducibility, and framing relative to the literature.

---

## 评审意见

### 1. Core Scientific Claim: Is the PID Framing Falsifiable?

**Central concern**: The paper's main claim — that four WD sub-traditions are special cases of a single PID control law — is presented as a unifying framework, but it functions more as a post-hoc descriptive taxonomy than a predictive theory.

Specific issues:

**1a. Unfalsifiable mapping construction.** The mapping in Table 1 assigns each method to PID gains by inspection of their mathematical form, not by fitting to observed trajectories. CWD is assigned $K_d > 0$ because it uses an alignment term; CPR is assigned $K_i > 0$ because it accumulates constraint violations. These assignments are definitionally true — any method that uses a weighted sum of error, integrated error, and derivative-of-error terms will fit the PID form. The framework cannot be falsified by any new WD method, because any method can be assigned to some $(K_p, K_i, K_d)$ configuration.

The fitting results in Table 2 (CWD 4.71% error, CPR 9.57% error) do validate that the *per-layer trajectory* of these methods can be captured by the extended control law. But SWD (45.81%) and DefazioCorrective (37.56%) failing the 20% threshold is not a falsification of the framework — the paper reinterprets this as "scheduling-based methods require a separate global-feedback extension," which is unfalsifiable retrofitting.

**Recommendation**: Reframe the contribution. The PID parameterization is a useful *taxonomy* and *design language*, not a falsifiable unified theory. The Introduction and abstract should say so explicitly: "We propose a PID-inspired taxonomy..." rather than "We prove that all four traditions are special cases of..."

---

### 2. UDWDC: The "Proposed Method" Does Not Work

The paper is unusually honest about UDWDC's failure, but the framing still creates a misleading impression.

**2a. A proposed algorithm that underperforms NoWD is not a contribution in the standard ML paper sense.** UDWDC v1 achieves 90.15% on CIFAR-10, below the no-regularization baseline (90.25%). The paper frames this as "UDWDC's contribution is conceptual and methodological" (Section 7.6) and "the instability finding has broader implications" (Section 6.4). This is a valid research framing, but reviewers at NeurIPS/ICML will ask: why propose UDWDC at all if it fails? The paper needs a clearer statement of what was learned from the failure and why the failure is itself a contribution worth reporting.

**2b. UDWDC-v2's WD budget anomaly is underexplained.** Table 3 reports UDWDC-v2's WD budget as 98,599 — approximately 205,000× FixedWD's 0.48. Section 6.4 explains this as "floor clipping applies $\lambda_{\min}$ to all 65 layers including BN layers at every step." This is a design error, not a fundamental finding. The fix (restrict control loop to weight layers only) is mentioned in Section 6.4 but never implemented or tested. Including UDWDC-v2 in Table 3 with an anomalous 98,599 WD budget invites confusion — a reviewer may assume the result is a bug rather than a documented design artifact.

---

### 3. Theoretical Claims: Theorem 1 Scope Is Overstated

**3a. Theorem 1 extends Sun et al. (CVPR 2025) to "time-varying WD with alignment modulation" but does not prove a novel bound.** The statement in Section 3.4 says: "The generalization bound depends on $A_T$ rather than the worst-case alignment. When alignment variance $\text{Var}_t[\phi(\delta_t)] > 0$, alignment-modulated WD achieves a strictly tighter generalization bound per unit WD budget than fixed WD." This is a corollary of Sun et al.'s framework applied to a variable function $\phi$, not a new theorem. The proof structure ("intuition" paragraph) does not constitute a rigorous proof. If this is a proposition/corollary, it should be labeled as such, not "Theorem 1."

**3b. Proposition 2 is a prediction without experimental support.** "Our experiments use SGD for CNNs, so this proposition is a prediction for future Adam-based evaluations." A proposition that is explicitly unvalidated should not be presented as a contribution in the current paper. It can appear as a remark or future work direction, but calling it "Proposition 2" in the main contributions list (Section 1, item 4) overstates the empirical support.

---

### 4. ImageNet Results: Statistical Validity Issues

**4a. CWD has only 1 completed seed on ImageNet (Table 5).** The paper reports CWD's result (70.66%) without error bars. This is flagged in Section 7.6 but the table itself does not visually distinguish this from results with 3–5 seeds. A single-seed result cannot be used to conclude anything about CWD's ImageNet performance. The table should either (a) add a footnote marking CWD as single-seed with statistical uncertainty, or (b) remove CWD from Table 5 entirely and note it as "insufficient seeds."

**4b. FixedWD uses 5 seeds on ImageNet while dynamic methods use 3 seeds.** This asymmetry means the FixedWD baseline has a tighter confidence interval than comparators, giving it an unfair appearance of precision. All ImageNet comparisons should use the same number of seeds.

**4c. 71.72% for FixedWD ResNet-50 is substantially below the standard 76–77%.** The paper attributes this to "minimal augmentation protocol." This is a legitimate experimental design choice — isolating WD effects — but it means all results are from a non-standard training regime. Claims about which method "achieves the best performance" apply only in this specific reduced-augmentation setting. At least one experiment with standard augmentation (mixup + RandAugment) would strengthen generalizability claims.

---

### 5. Standardized Metrics: Implementation Issues

**5a. BEM denominator choice is unusual.** $\text{BEM} = (\text{acc} - \text{acc}_{\text{NoWD}}) / \text{TotalWDBudget}$ divides by cumulative $\sum_t \sum_l \lambda_t^l \|w_t^l\|^2$. This means methods that apply WD to more layers (more parameters) will have higher denominators. UDWDC-v2's denominator of 98,599 effectively zeroes its BEM regardless of accuracy. A more interpretable denominator would be the time-averaged effective WD coefficient, which is scale-invariant to the number of layers.

**5b. CSI's negative values need clearer interpretation.** UDWDC's $\text{CSI}_{\text{temporal}} = -5.75$ is described as indicating the controller "amplifies rather than damps perturbations." But the CSI formula is $1 / \text{Var}_t[\lambda_{\text{eff}}^l]$ — this cannot be negative unless there is an implementation detail not described in Section 4.2. A ratio of $1/\text{Var}$ is always non-negative. The paper must explicitly state how negative CSI values arise (presumably from the normalization step relative to FixedWD's near-infinite CSI when lambda is constant, creating a signed ratio).

---

### 6. Related Work: Missing Key References

**6a. Nesterov momentum and weight decay interaction.** The paper omits Ilyas et al. (2022, "Datamodels") and related work on feature learning vs. regularization tradeoffs. More critically, the paper does not cite Kosson et al. (2024) on spectral analysis of weight decay's role in stabilizing training across widths — this is directly relevant to the norm-matching tradition and the fixed-point analysis.

**6b. The paper cites "Kosson et al. (2025)" in Section 2.3 but provides no full citation in the bibliography.** Similarly, "Chou (2025)" and "GWA (Holzl et al., NeurIPS 2025)" appear in the text without complete citation information. At least three references in the paper are incomplete or lack arxiv/venue identifiers.

---

### 7. What the Paper Does Well

1. **Honest negative result reporting.** Sections 3.3, 7.6, and the abstract all pre-register UDWDC's failure. This is rare in ML papers and valuable. The "engineering patches, not principled solutions" language in Section 3.3 is precise and creditable.

2. **BEM captures a real research gap.** The CPR/FixedWD BEM comparison (0.39 vs. 0.89) reveals that CPR's accuracy gains come partly from a 9.3× higher WD budget, not just better WD allocation. This is a genuine contribution to evaluation methodology.

3. **The control-theoretic vocabulary is useful.** Even if the framework is descriptive rather than predictive, having a shared language for comparing "derivative-only" (CWD) vs. "integral-dominant" (CPR) vs. "open-loop" (FixedWD) controllers is valuable for future method development.

4. **Per-layer diagnostic tracking.** Logging $\rho_t^l$, $\alpha_t^l$, $\|w_t^l\|$, and $\lambda_t^l$ at every epoch for all 65 layers across 3 seeds is a thorough empirical commitment. The temporal predictability analysis (Section 5.6) and CSI stability comparison (Section 5.7) are enabled by this infrastructure.

---

## 总结评分

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Scientific originality | 6/10 | PID taxonomy is useful but descriptive, not falsifiable; UDWDC fails empirically |
| Theoretical rigor | 5/10 | Theorem 1 is a corollary; Proposition 2 unvalidated |
| Empirical quality | 7/10 | CIFAR results solid; ImageNet has seed/augmentation issues |
| Evaluation methodology | 7/10 | BEM/CSI/AIS genuine contributions; CSI negative-value issue |
| Writing quality | 8/10 | Evidence-first, honest, no banned patterns; internal contradiction in Section 6.4 |
| Reproducibility | 6/10 | Missing figure sources; incomplete bibliography; AIS operationally undefined |

**综合评分: 6.5/10**

The paper's strongest claim is the PID taxonomy and the standardized metrics. Its weakest claim is UDWDC as a practical algorithm. For top-venue submission (NeurIPS/ICML), the key revisions needed are: (1) reframe PID as taxonomy rather than falsifiable theory, (2) validate or demote Proposition 2 to future work, (3) fix ImageNet seed imbalance and CWD single-seed issue, (4) clarify how CSI takes negative values, and (5) complete the figure pipeline. The honest negative result reporting is a genuine differentiator that should be preserved and made more prominent in the abstract.
