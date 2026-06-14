# Result Debate Synthesis

**Synthesizer**: sibyl-result-synthesizer (Opus 4.6)
**Date**: 2026-04-14
**Iteration**: 6/7
**Perspectives Synthesized**: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist

---

## 1. Consensus Map

The following conclusions are endorsed by all six perspectives with high confidence:

### 1.1 The Multi-L0 Confound Decomposition Is the Strongest Finding

**Universal agreement.** All perspectives identify the multi-L0 confound decomposition as the study's most robust and novel contribution. At L0=22 with perfect probes (F1=1.0 for all 25 letters), 98.6% of the 657 false negatives classified as "absorbed" by the Chanin metric are actually hedging -- recoverable by increasing L0 -- with only 1.4% (9 words) being genuinely hierarchy-driven. The trend is perfectly monotonic (Spearman rho=1.0): hierarchy-driven fraction increases from 1.4% at L0=22 to 90.0% at L0=176 as hedging false negatives resolve.

- **Optimist**: Calls the perfect monotonic trend and the L0=22 perfect-probe experiment (42.85% absorption) a breakthrough.
- **Skeptic**: Acknowledges this is informative despite flagging the 69x pilot-to-full discrepancy.
- **Strategist**: Ranks this as "VERY HIGH" signal strength and anchor for the paper.
- **Methodologist**: Rates the L0 profile as "genuinely novel" but insists on correcting the 96.9% pilot headline.
- **Comparativist**: Identifies this as the ONE thing the work does that no prior work does.
- **Revisionist**: Calls it "the single most consequential finding" requiring a massive belief update.

**Synthesizer judgment**: This is the paper's anchor contribution. It directly extends and challenges the interpretation of the most-cited absorption work (Chanin et al., NeurIPS 2025 Oral). The finding that published 15-35% absorption rates likely conflate hedging (~98.6% at low L0) with genuine hierarchy-driven absorption (~0.75%) is important for the mechanistic interpretability community.

### 1.2 The Control Failure Is Real, Not a Bug

**Universal agreement.** Shuffled-label controls exceed measured absorption rates in ALL five domains. The improved first-letter experiment worsened this: shuffled went from 59.6% to 74.6%, random probe from 9.2% to 11.8%. No domain achieves `control_credible: true`.

- **Optimist**: Acknowledges controls are not calibrated, treats as an honest caveat.
- **Skeptic**: Labels this Fatal Flaw #1, invalidating all absorption rate claims.
- **Strategist**: Agrees this is a blocking issue but notes it is also a publishable finding.
- **Methodologist**: Identifies this as THE critical validity threat and recommends diagnosing root cause.
- **Comparativist**: Classifies cross-domain claims as "UNRELIABLE" but notes the control failure itself is a contribution.
- **Revisionist**: Attributes the failure to JumpReLU activation statistics (dead features, unit normalization, binary activations).

**Synthesizer judgment**: The control failure is a genuine structural problem of the Chanin metric on JumpReLU SAEs, not an implementation error. This means absolute absorption rates cannot be taken at face value. However, this is itself a publishable methodological finding: "The standard probe-based absorption metric does not transfer from GPT-2/L1 to Gemma 2 2B/JumpReLU SAEs without recalibration." The Skeptic is correct that this invalidates naive cross-domain rate claims, but the Strategist and Comparativist are correct that it can be reframed as a primary contribution rather than a fatal embarrassment.

### 1.3 The L0-Absorption Monotonic Relationship Is Robust

All perspectives agree that the monotonic decline in absorption with L0 (42.9% at L0=22, 37.5% at L0=41, 14.4% at L0=82, 0.8% at L0=176) is a clean empirical finding. This is informative about SAE sparsity dynamics regardless of whether individual absorption rates are well-calibrated.

### 1.4 Unsupervised Detection (H4) Is Definitively Dead

All perspectives agree ITAC-based unsupervised absorption detection failed completely (AUROC below chance, precision@50 = 0, best rho = -0.104 wrong direction). This should be reported as a pre-registered negative result.

### 1.5 Hierarchy Predictors (H6) Are Fatally Underpowered

With n=5 domains and bootstrap CIs spanning [-1, 1], no statistical inference is possible. All perspectives agree this should be reframed as purely descriptive or dropped entirely.

### 1.6 The Pilot-to-Full Decomposition Discrepancy Must Be Explained

The 96.9% (pilot) vs 1.4% (full) hierarchy-driven fraction at L0=22 -- a 69x gap -- is universally flagged. The explanation is structural: the pilot tested only one L0, making hedging classification impossible, and used ~25 words per letter vs 1,196 in the full experiment. This must be transparently discussed and the 96.9% headline must be retired.

---

## 2. Conflict Resolution

### Conflict 1: Is the CMI-Absorption Correlation a Real Finding or a Cherry-Picked Artifact?

**Optimist position**: rho=-0.383 with large effect size (Cohen's d=-0.924) and significant group separation (Mann-Whitney p=0.042) constitutes the first evidence that information theory predicts absorption susceptibility. The d'=10 subspace captures the absorption-relevant information geometry.

**Skeptic position**: This is Fatal Flaw #2. The correlation exists ONLY at d'=10 and reverses at d'=20/30/50. With Bonferroni correction p=0.236. The sign flip is a red flag for noise. Probe quality confound may drive the entire effect.

**Methodologist position**: Post-hoc d' selection is researcher degrees of freedom. The sign reversal is concerning. But the effect size is genuinely large (Cohen's d=-0.924), and d'=10 might correspond to the effective dimensionality of the decoder subspace.

**Comparativist position**: Novel theoretical contribution (no prior work), but fragile empirical support. Rates it 1-5% contribution margin.

**Revisionist position**: Confirmed at d'=10 only; dimension-dependent. Confidence 0.45.

**Synthesizer judgment**: The CMI result is **promising but not yet established**. The large effect size and correct direction are real positive signals that cannot be dismissed as pure noise. However, the dimension dependence is a genuine fragility that must be resolved. I side with the Methodologist's recommendation: the next critical experiment should be a pre-registered CMI replication at d'=10 with L0=22 data (where all probes have F1=1.0, eliminating probe quality confound). If that replication confirms rho < -0.3, p < 0.05, the theoretical contribution is secured. If not, CMI should be reported as "directionally consistent, statistically inconclusive."

**Resolution**: CMI is a conditional pillar. Include it in the paper framing but with honest caveats about dimension dependence. The L0=22 replication is the decisive test. Score this contribution as **4/10 currently, potentially 7/10 after replication**.

### Conflict 2: Can ANY Absorption Rate Claims Survive the Control Failure?

**Skeptic position**: No. All absorption rate claims are invalidated. H1 is dead.

**Optimist position**: The first-letter replication at 15.96% within Chanin et al.'s 15-35% range is meaningful despite the control anomaly. Cross-layer consistency (13.88%, 15.96%, 13.55%) provides additional confidence.

**Strategist position**: The raw rates are unreliable, but the L0-absorption profile survives because it measures *relative* changes, not absolute levels.

**Synthesizer judgment**: I side primarily with the Skeptic and Strategist here, against the Optimist. The absolute absorption rates (15.96%, 6.5%, etc.) are not credible as standalone claims when shuffled controls exceed them by 4.7x. The Optimist's "replication" argument is weakened by the fact that Chanin et al. worked on a different model with different SAE architecture -- replicating the same number on a structurally different system may be coincidental.

However, two things survive the control failure: (1) the multi-L0 profile, which measures relative changes as L0 varies (the control issue likely affects the absolute level, not the trend), and (2) the 9 genuinely absorbed words identified by multi-L0 consistency, which are independent of the metric's noise floor. The paper must clearly distinguish between these robust findings and the invalidated absolute rate claims.

**Resolution**: Absolute cross-domain absorption rates are unreliable. Report them honestly with controls. The L0 profile and 9-word ground truth set are the credible empirical contributions.

### Conflict 3: What Is the Right Paper Framing?

**Optimist**: Three-pillar story (cross-domain characterization, rate-distortion diagnostic, multi-L0 decomposition).

**Skeptic**: Methodological audit paper ("Challenges in Cross-Architecture Absorption Measurement").

**Strategist**: Three-pillar with reframing (metric validation + L0 profile + rate-distortion theory).

**Comparativist**: Metric audit + L0 phase transition + rate-distortion diagnostic.

**Revisionist**: Confound decomposition as primary, metric validation as secondary, rate-distortion as tertiary.

**Synthesizer judgment**: The convergence across Strategist, Comparativist, and Revisionist is clear. The paper should be framed around **three pillars**, ordered by evidence strength:

1. **Primary (Empirical)**: Multi-L0 confound decomposition revealing that 98.6% of measured absorption is hedging, not hierarchy-driven.
2. **Secondary (Methodological)**: The Chanin absorption metric does not transfer to JumpReLU SAEs -- universal control failure with diagnostic investigation.
3. **Tertiary (Theoretical)**: Rate-distortion CMI diagnostic for predicting absorption susceptibility -- directionally correct, conditional on replication.

The cross-domain rates become supporting evidence for the methodological finding (demonstrating that the metric fails across multiple hierarchy types), not a primary contribution.

### Conflict 4: Paper Venue Tier

**Optimist**: Publishable at main conference with the current three pillars.

**Skeptic**: Requires fundamental fixes; possible downgrade to methodology-focused paper.

**Strategist**: 55% main conference, 75% workshop, 98% any venue.

**Comparativist**: Workshop current state, mid-tier after blocking experiments, top-tier unlikely.

**Synthesizer judgment**: The Strategist's probability assessment is the most calibrated. The Comparativist is too pessimistic (underweights the confound decomposition's impact), and the Optimist is too optimistic (ignores the control failure's severity). After the three priority experiments, a realistic venue range is:

- **Best case** (controls diagnosable + CMI replicates + patching confirms): NeurIPS/ICML main -- competitive submission.
- **Expected case** (1-2 of 3 pass): AAAI/EMNLP main or NeurIPS/ICML workshop.
- **Worst case** (all fail): TMLR or workshop -- still publishable, but lower tier.

**Resolution**: Current state supports a workshop submission. The three priority experiments (3.5-5 GPU hours total) determine whether it rises to main conference level.

---

## 3. Result Quality Score: 5.5 / 10

**Justification**:

| Dimension | Score | Weight | Contribution |
|-----------|-------|--------|-------------|
| Study design quality | 8/10 | 15% | 1.20 |
| Execution rigor | 5/10 | 20% | 1.00 |
| Statistical strength | 4/10 | 20% | 0.80 |
| Novelty of findings | 7/10 | 15% | 1.05 |
| Credibility of main claims | 4/10 | 20% | 0.80 |
| Negative result handling | 8/10 | 10% | 0.80 |
| **Weighted total** | | 100% | **5.65 -> 5.5** |

The study has an excellent structural design (5 domains, multi-L0 sweeps, pre-registered gates, honest negative result reporting) with a genuinely novel finding (confound decomposition). But the execution-level problems (universal control failure, dimension-dependent CMI, pilot-full discrepancy) drag down the overall quality. The gap between design ambition (8/10) and claim credibility (4/10) reflects a study that asked the right questions but discovered along the way that its primary measurement instrument has serious limitations.

---

## 4. Key Findings

### Finding 1: The absorption metric conflates hedging with genuine hierarchy-driven absorption
At L0=22 with perfect probes (F1=1.0), the Chanin absorption metric detects a 42.9% false-negative rate, but 98.6% of these are hedging artifacts (tokens that recover at higher L0), not hierarchy-driven competitive exclusion. Only 9 out of 1,195 tested words (0.75%) exhibit genuine hierarchy-driven absorption that persists across all sparsity levels. This challenges the widely-cited 15-35% absorption rate from Chanin et al.

### Finding 2: The probe-based absorption metric does not transfer to JumpReLU SAEs
Shuffled-label controls exceed measured absorption in all 5 tested domains (ratio 1.7x to 4.7x). The random-probe control rate of 11.8% on the improved first-letter experiment exceeds the 2% target by 6x. This is not an implementation error but a structural property of the metric's interaction with JumpReLU activation statistics (88% dead features, unit-normalized decoders, hard thresholds).

### Finding 3: Absorption severity follows a monotonic L0 phase transition
Absorption rates decline monotonically with L0: 42.9% (L0=22) to 37.5% (L0=41) to 14.4% (L0=82) to 0.8% (L0=176). Cross-layer consistency is strong (CV < 10% across layers 10, 12, 20). JumpReLU SAEs show a dramatic L0-dependent phase transition while L1/GPT-2 SAEs show uniformly high absorption (61-67%), though the cross-model confound prevents definitive architecture attribution.

### Finding 4: Rate-distortion theory shows correct qualitative direction but fragile quantitative support
CMI negatively correlates with absorption at d'=10 (rho=-0.383, Cohen's d=-0.924, Mann-Whitney p=0.042), consistent with the prediction that features with lower conditional mutual information are preferentially absorbed. However, the correlation reverses at higher subspace dimensions (d'=20/30/50), making the result dimension-dependent. The phase transition scale prediction (L0_crit=24.7 vs empirical 22.4, 10.2% error) is partially circular since lambda is fit from empirical data. The geometric constant c degenerates for unit-normalized SAEs (CV=2.16%), simplifying the rate-distortion threshold to approximately lambda/CMI.

### Finding 5: Where genuine absorption occurs, it is concentrated in specific dominant child subgroups
City-continent absorption (6.5% aggregate) is driven almost entirely by Chinese cities absorbing the "Asia" parent (21.6%). City-language absorption (6.6% aggregate) is driven entirely by English (25.5%). This selectivity is consistent with rate-distortion theory (low CMI for dominant child clusters) but means absorption is a feature-pair-specific phenomenon, not a domain-level one.

---

## 5. Methodology Gaps (From Skeptic + Methodologist)

### Critical Gaps (Must Fix Before Submission)

1. **Diagnose the control failure root cause.** The shuffled control exceeding measured absorption is an existential threat to all empirical claims. Three diagnostic experiments are needed: (a) analytical computation of expected shuffled-rate from SAE feature cosine distribution, (b) null-domain benchmark on non-hierarchical tasks, (c) threshold recalibration study. Estimated effort: 2-3 GPU-hours.

2. **Resolve CMI dimension dependence.** Either provide a principled argument for d'=10 (e.g., effective dimensionality of decoder-direction subspace via eigenspectrum analysis), cross-validate d' across letter subsets, or use a dimension-free MI estimator. The current post-hoc d' selection is indefensible under peer review. Estimated effort: 1 GPU-hour.

3. **Validate the 9 genuinely absorbed words via activation patching.** This provides the only metric-independent causal evidence for genuine absorption. Zero the identified child features and check if parent features recover. Estimated effort: 0.5-1 GPU-hour.

### Important Gaps (Should Fix)

4. **Missing CMI ablations**: k-NN k parameter (fixed at k=5), corpus size convergence test, absorbed/non-absorbed partition threshold sensitivity.

5. **Cross-domain threshold sensitivity grid**: The 5x4 threshold grid was only run on first-letter. Running on city-continent and city-language would assess whether those rates are threshold-robust.

6. **Vocabulary sensitivity**: No bootstrap vocabulary test. Letters with few words (X=1, Z=7-9) have unreliable per-letter rates.

7. **Prompt format sensitivity**: No test of prompt format variation.

### Design Limitations (Report Honestly)

8. **n=5 domains for cross-domain correlations**: Bootstrap CIs span [-1, 1]. Report H6 as descriptive only.

9. **Cross-model confound in bifurcation analysis**: L1 results from GPT-2 Small vs JumpReLU from Gemma 2 2B. Cannot separate architecture from model effects.

10. **Single model-SAE family**: All measurements on Gemma 2 2B + Gemma Scope. Generalization claims require replication on at least one other model.

---

## 6. Competitive Position (From Comparativist)

### Positioning vs SOTA

The work occupies a **genuine but narrow niche** in the SAE absorption literature:

| Competitor | Relationship | Our Edge |
|---|---|---|
| Chanin et al. (NeurIPS 2025 Oral) | We audit their metric; they created it | Confound decomposition, control analysis |
| "Sparse but Wrong" (Chanin & Garriga-Alonso, 2025) | Our hedging finding is an empirical quantification of their theoretical point | Multi-L0 perfect-probe decomposition |
| SAEBench (ICML 2025) | Different scope (benchmark vs diagnostic) | Complementary |
| ATM-SAE (Li et al., 2025) | They mitigate absorption; we diagnose it | Understanding "why" vs "how to fix" |
| Matryoshka SAE (ICML 2025) | They mitigate via architecture; we provide diagnostic framework | Complementary |
| Unified SDL Theory (2024) | More complete theory (WHY); our CMI asks WHEN | Complementary, narrower |

**Unique contribution**: First quantitative decomposition of measured absorption into hedging vs hierarchy-driven components. No published work has done this.

**Threat**: The strongest competitor is "Sparse but Wrong," which establishes the theoretical foundation that low L0 causes wrong features. Reviewers may view our finding as an obvious corollary. We must clearly articulate the incremental value: (1) explicit quantification within the absorption metric's output, (2) perfect-probe isolation of the effect, (3) monotonic L0 profile with 4 sparsity levels.

### Venue Assessment

| Venue Tier | Probability | Conditions |
|---|---|---|
| NeurIPS/ICML/ICLR main | 30-35% | Controls diagnosable + CMI replicates + patching validates |
| AAAI/EMNLP or NeurIPS workshop | 40-45% | 1-2 of 3 blocking experiments pass |
| TMLR or workshop | 20-25% | Current state, floor outcome |
| Not publishable | <2% | Even worst case has publishable findings |

---

## 7. Hypothesis Update (From Revisionist)

### Survived (With Modifications)

| Hypothesis | Status | Modification |
|---|---|---|
| H3: CMI predicts absorption | **Alive, conditional** | Only at d'=10; needs pre-registered replication at L0=22 |
| H5: L0-absorption phase transition | **Confirmed** | Observed transition near L0~29, not L0~7-14 as predicted; relationship is cleanly monotonic |

### Revised

| Hypothesis | Original | Revised |
|---|---|---|
| H2: >80% hierarchy-driven | Most absorption is hierarchy-driven competition | ~99% of what the metric detects at low L0 is hedging; genuine hierarchy-driven absorption is ~0.75% |
| H7: JumpReLU bimodal vs L1 continuous | Architecture determines distribution shape | Both architectures bimodal; JumpReLU shows L0-dependent severity phase transition, L1 uniformly high; but cross-model confound prevents definitive attribution |

### Falsified

| Hypothesis | Evidence | Confidence |
|---|---|---|
| H1: Cross-domain >= 5% in 2+ domains | Controls fail across all domains; net signal negative everywhere | 0.80 (leaning refuted) |
| H4: Unsupervised detection rho > 0.3 | AUROC below chance, precision@50 = 0, best rho = -0.104 | 0.98 (definitively falsified) |
| H6: Hierarchy predictors | n=5, all CIs span [-1, 1] | Untestable by design |

### New Hypotheses Generated

1. **NH1**: Hedging fraction follows pct_hedging ~ 1 - k/L0 (predictable parametric function). Testable with 4 additional L0 values.
2. **NH2**: Activation patching on the 9 genuinely absorbed words will show parent recovery in >= 7/9 cases. Testable in 0.5-1 GPU-hour.
3. **NH3**: The Chanin metric's noise floor correlates positively with dead feature fraction across SAE architectures. Testable cross-architecture with 8+ configurations.

---

## 8. Action Plan

### Verdict: PROCEED (Conditional on Three Priority Experiments)

The data supports a publishable paper, but the paper's framing and tier depend critically on three blocking experiments. The strategic position has improved substantially since the prior PIVOT decision: the CMI signal recovered (rho from +0.14 to -0.383), the confound decomposition emerged as a genuinely novel contribution, and the control failure was reframed as a methodological finding rather than a pure liability.

### Priority Experiments (Ordered by Information Gain per GPU-Hour)

| Priority | Experiment | GPU-Hours | Decision Gate | Impact |
|---|---|---|---|---|
| **P1** | Control failure diagnosis (analytical expected rate + null-domain benchmark + threshold recalibration) | 2-3h | If recalibration yields positive net signal in >=2 domains, cross-domain claims recover. If not, control failure becomes primary methodological finding. | Determines paper framing for Pillar 2 |
| **P2** | CMI replication at L0=22 (pre-registered d'=10, F1=1.0 probes, maximum absorption variance) | 1h | If rho < -0.3, p < 0.05: theoretical contribution secured. If rho > -0.2: drop CMI from primary contributions. | Determines whether paper has 2 or 3 pillars |
| **P3** | Activation patching on 9 core absorbed words (zero child features, check parent recovery) | 0.5-1h | If parent recovery in >= 7/9: genuine absorption confirmed independent of all metric concerns. If < 4/9: even "robust" absorbed words may be artifacts. | Provides the only metric-independent ground truth |

**Total**: 3.5-5 GPU-hours. This is an extremely efficient investment that determines the paper's tier.

### Writing Plan (After Priority Experiments)

The paper should adopt the three-pillar structure regardless of experiment outcomes:

**Title direction**: "Decomposing Feature Absorption: Hedging Artifacts, Metric Portability, and Rate-Distortion Diagnostics for Sparse Autoencoders"

**Structure**:
1. Introduction: Absorption is widely cited (15-35%) but its measurement and interpretation are less examined.
2. Background: Chanin metric, successive refinement theorem, SAE architecture.
3. Multi-L0 confound decomposition (Pillar 1): 98.6% hedging at L0=22, monotonic L0 profile.
4. Metric portability analysis (Pillar 2): Universal control failure on JumpReLU SAEs + diagnostic.
5. Rate-distortion diagnostic (Pillar 3, conditional): CMI-absorption correlation, geometric constant degeneration.
6. Cross-domain results: Presented as supporting evidence for metric portability, not as primary claims.
7. Discussion: 9 genuinely absorbed words as ground truth, activation patching validation.
8. Negative results: H4 (unsupervised), H6 (underpowered).

### Scenario-Dependent Outcomes

| Scenario | P(outcome) | Paper Tier | Title Emphasis |
|---|---|---|---|
| All 3 pass | 25% | NeurIPS/ICML main (competitive) | Rate-distortion + decomposition + metric audit |
| P1 + P2 pass, P3 partial | 20% | NeurIPS/ICML main (marginal) | Rate-distortion + decomposition |
| P1 pass, P2 fails | 15% | AAAI/EMNLP or workshop (strong) | Decomposition + metric audit |
| P2 pass, P1 fails | 15% | AAAI/EMNLP or workshop (strong) | Rate-distortion + decomposition |
| P1 fails, P2 fails, P3 pass | 10% | Workshop or TMLR | Decomposition + negative methodology |
| All fail | 15% | Workshop or TMLR | Methodology + negative results (floor) |

### Resource Allocation

| Resource | Allocation | Rationale |
|---|---|---|
| GPU time (next 5h) | 100% on P1-P3 | These three experiments determine the paper's tier |
| Writing effort | 0% until P1-P2 complete | Paper framing is conditional on results |
| Theoretical refinement | Hold until P2 | CMI theory lives or dies with replication |
| Cross-domain expansion | **Deprioritize** | Blocked by control failure |
| Unsupervised detection | **Drop** | H4 definitively negative |
| Hierarchy predictors (H6) | **Drop** | Fatally underpowered |

### What NOT To Do

1. **Do not sink GPU time into additional cross-domain experiments** before resolving the control issue. Every hour spent on this is wasted if the metric is uncalibrated.
2. **Do not abandon the CMI line** without the L0=22 replication that uses perfect probes and maximum absorption variance. The d'=10 result has a large effect size -- it deserves one clean replication attempt.
3. **Do not smooth over the control failure** in the paper. Transparent reporting of the failure, its diagnosis, and its implications is a strength, not a weakness.
4. **Do not present the L0=22 42.85% absorption rate as a "true rate"** without noting that the vast majority (98.6%) is hedging. The "true hierarchy-driven absorption rate" is 0.75%, not 42.85%.
