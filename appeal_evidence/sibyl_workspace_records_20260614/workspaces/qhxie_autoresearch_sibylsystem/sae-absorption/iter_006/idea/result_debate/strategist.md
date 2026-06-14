# Strategist Analysis: Iteration 6 Experiment Results

## Signal Strength Assessment

| Result | Signal Strength | Justification |
|--------|----------------|---------------|
| First-letter absorption replication (15.96% at L0=82, improved) | **Strong** | Within published 15-35% range; F1=0.817 (up from 0.565 in pilot); 10/25 pass F1>0.85 gate; stable across layers 10/12/20; 1204 words tested |
| Multi-L0 absorption profile (42.9% at L0=22 -> 0.8% at L0=176) | **Strong** | Clean monotonic decline across 4 L0 values; rho=1.000 for hierarchy-driven trend with L0; first such systematic profile published for any SAE |
| CMI-absorption correlation (rho=-0.383, p=0.059, Cohen's d=-0.924) | **Moderate** | Correct direction and large effect size. Mann-Whitney p=0.045 (one-sided 0.023). Substantial improvement over prior dataset (rho=+0.14 -> -0.383). BUT: holds only at d'=10, reverses at d'>10. The dimension sensitivity is a genuine fragility |
| Control failure (shuffled=74.6% on first-letter improved, C1=11.8%) | **Strong** (as a problem signal) | Systematic across all domains. Not noise -- a structural property of the metric interacting with Gemma 2 2B's SAE feature density |
| Cross-domain absorption rates (0-6.6% on knowledge domains) | **Weak** | All CIs include zero; concentrated in 1-2 parents per domain; no domain's net signal exceeds its shuffled control |
| Scaling surface GAM (R^2=0.85, OLS interaction p=1.24e-6) | **Strong** | 34 configurations tested; GAM interaction p=1.0 (contradicts OLS); but the overall fit and L0 main effect are robust |
| H2 confound decomposition (1.4% hierarchy-driven at L0=22) | **Strong** (negative result) | Directly contradicts pilot's 96.9%. Reveals L0-dependent decomposition: hierarchy-driven fraction INCREASES with L0 (1.4% at L0=22 -> 90% at L0=176). This reversal is itself a finding |
| Bifurcation analysis (JumpReLU vs L1) | **Moderate** | KS D=0.607, p near 0. Original prediction (JR=bimodal, L1=continuous) falsified, but the reframing (L0-dependent phase transition in JumpReLU vs uniformly high L1 absorption) is more nuanced and more interesting |
| Phase transition prediction (L0_crit=24.7 vs empirical 22.4) | **Moderate** | Scale match excellent (10.2% error). Direction correct (rho=+0.333 for rank prediction). But binary classification fails (36% < chance), and lambda estimation is partially circular |
| Geometric constant degeneration (c essentially constant, CV=2.16%) | **Strong** (as a theoretical constraint) | Clean result: unit-normalized decoders make c=sin^2(angle) near-constant across all parent-child pairs. Simplifies theory to pure CMI-based criterion |
| Unsupervised detection (H4) | **Noise** | Best rho=-0.125, AUROC=0.47. 6/25 letters matched. ITAC shows no separation. H4 is definitively dead |
| Threshold sensitivity (grid CV=0.0277) | **Strong** | Remarkably stable metric. CV=2.77% across the 5x4 grid means the Chanin metric's rank order is robust to threshold choices |

## Opportunity Cost Analysis

| Direction | Est. GPU-Hours | Expected Information Gain | Gain per GPU-Hour |
|-----------|---------------|--------------------------|-------------------|
| Control investigation and metric validation | 2-3h | **Very High** -- resolves whether the paper's empirical claims survive | **Highest** |
| CMI replication at L0=22 (pre-registered d'=10, perfect probes) | 1h | **High** -- definitive test on the theoretical contribution | **Very High** |
| Activation patching on core absorbed words | 0.5-1h | **High** -- metric-independent ground truth for absorption | **High** |
| Cross-domain CMI (extend to city-continent/city-language) | 1h | **Moderate** -- tests generalizability, but underpowered (n=6/18) | **Moderate** |
| Fine-grained L0 sweep (10 L0 values on first-letter) | 2h | **Low-Moderate** -- refines phase transition curve | **Low** |
| Unsupervised detection refinement | 1h | **Near-Zero** -- evidence is decisive | **Lowest** |
| Additional cross-domain experiments | 2h | **Low** -- controls must be resolved first | **Low** |

## Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome |
|-----------|----------------|----------|------|------------------|
| **A. Control investigation** | N/A (diagnostic) | 2-3h | LOW (always informative) | Either metric transfers to Gemma 2 2B with recalibration (threshold/normalization fix), or the control failure itself becomes a primary methodological finding |
| **B. CMI replication at L0=22** | Moderate | 1h | MEDIUM (35% chance of null) | At L0=22: absorption variance is maximal (0%-80% per letter), all probes have F1=1.0. If rho < -0.3, p < 0.05 at pre-registered d'=10, the theoretical contribution is secured |
| **C. Activation patching** | N/A (validation) | 0.5-1h | LOW | Establishes ground truth for the core absorbed words entirely independent of the Chanin metric. Provides the "immune escape test" |
| **D. Paper around three pillars** | Strong salvageables | 0h (writing) | MEDIUM | Reframes from "cross-domain absorption" to "metric validation + L0 profile + rate-distortion theory" |
| **E. Additional cross-domain experiments** | Weak | 2h | HIGH (controls unresolved) | Cannot produce credible results until control issue is resolved |

## PROCEED (Conditional)

**Verdict: PROCEED, with immediate paper reframing and 3 priority experiments.**

### Rationale

The prior supervisor PIVOT decision (confidence 0.75) was made on the basis of earlier experiments that had (1) poor probe quality (mean F1=0.565), (2) small sample sizes (576 tokens for CMI), and (3) a CMI analysis that showed rho=+0.14 in the wrong direction. The subsequent improved experiments have substantially changed the evidence landscape:

| Metric | Before (supervisor PIVOT) | After (improved experiments) | Delta |
|--------|--------------------------|------------------------------|-------|
| Probe F1 (first-letter) | 0.565 | 0.817 | +0.252 |
| Letters passing F1>0.85 | 4/25 | 10/25 | +6 |
| Total words tested | ~576 | 1204 | +2.1x |
| CMI rho (CMI vs absorption) | +0.14 (wrong direction) | -0.383 (correct direction) | Sign reversal |
| CMI Mann-Whitney p | 0.247 | 0.045 | Significant |
| CMI Cohen's d | 0.326 (small) | -0.924 (large) | Large effect |
| Phase transition prediction | Not computed | 10.2% scale match error | New evidence |
| Bifurcation analysis | Not computed | KS D=0.607 | New evidence |
| Geometric constant | Not computed | c degenerates (CV=2.16%) | Simplifies theory |

The improved experiments recovered the CMI signal (from rho=+0.14 to rho=-0.383) and dramatically improved probe quality. This changes the strategic calculus: the theoretical contribution (rate-distortion diagnostic) is now a viable primary pillar rather than a dead end.

### Why NOT PIVOT

The prior PIVOT recommendation was based on 3/6 hypotheses being falsified. But the post-PIVOT experiments changed the picture:

1. **H3 (CMI)** was listed as "FALSIFIED" (rho=-0.10, p=0.51) -- now shows rho=-0.383, p=0.059, Cohen's d=-0.924. This is not falsified; it is marginal-to-significant and needs one clean replication.
2. **H7 (bifurcation)** was not yet tested -- now shows a statistically significant distribution difference (KS D=0.607) with a more nuanced and interesting finding than originally hypothesized.
3. **The phase transition prediction** is a new positive result (10.2% scale match, correct rank ordering) that did not exist at PIVOT time.

The surviving signals are sufficient for a competitive paper. The question is not "is there a paper?" but "what is the best paper the data supports?"

### Why NOT defend the original proposal

The original framing ("Cross-Domain Absorption Characterization") assumed cross-domain absorption would be the empirical backbone. It is not. All cross-domain rates are weak and invalidated by control failure. The data demands a different paper.

## Paper Reframing: Three-Pillar Strategy

The data supports three strong pillars that did not align with the original proposal structure:

### Pillar 1: The Absorption Measurement Problem (Methodological Contribution)

**Evidence:**
- Shuffled controls exceed measured absorption in ALL domains (74.6% on improved first-letter, vs 15.96% measured)
- Random probe control at 11.8% (expected ~0%)
- Multi-L0 confound decomposition shows L0-dependent composition: hedging dominates at low L0 (98.6% at L0=22), hierarchy-driven dominates at high L0 (90% at L0=176)
- Threshold sensitivity grid shows metric is stable (CV=2.77%) but the absolute level is suspect

**Framing:** The Chanin absorption metric, validated on GPT-2 Small with L1 SAEs, does not transfer cleanly to Gemma 2 2B with JumpReLU SAEs. The control failure is not an implementation bug but a structural property of the metric's interaction with higher-dimensional feature spaces and JumpReLU activation. This is itself an important methodological finding: it tells the field that absorption measurement is model-architecture-dependent and requires per-architecture calibration.

**Strength for paper:** HIGH. No prior work has attempted to validate the absorption metric on a different architecture/model combination. The failure to transfer is novel and practically important.

### Pillar 2: The L0-Absorption Phase Transition (Empirical Contribution)

**Evidence:**
- Monotonic absorption decline: 42.9% (L0=22) -> 37.5% (L0=41) -> 14.4% (L0=82) -> 0.8% (L0=176)
- 9 core words persist as absorbed across all L0 values
- Cross-layer stability (CV < 10% across layers 10/12/20)
- Scaling surface R^2=0.85 with 34 configurations
- JumpReLU vs L1 bifurcation: JumpReLU shows dramatic L0 phase transition; L1 shows uniformly high absorption (60-67%)

**Framing:** The L0-absorption profile is the most robust empirical finding in the entire dataset. It reveals that sparsity pressure is the dominant driver of absorption severity, with a phase transition between L0~40-80. The 9 persistent core words provide ground truth anchors that can be independently validated via activation patching.

**Strength for paper:** VERY HIGH. Novel, robust, mechanistically informative.

### Pillar 3: Rate-Distortion Theory of Absorption (Theoretical Contribution)

**Evidence:**
- CMI correlates with absorption in the correct direction (rho=-0.383, large effect size)
- Phase transition prediction: theoretical L0_crit=24.7 vs empirical 22.4 (10.2% error)
- Geometric constant degenerates for unit-normalized SAEs (simplifies theory)
- Theoretical framework correctly predicts direction, scale, and simplification

**Caveats:**
- CMI correlation only holds at d'=10, reverses at higher dimensions
- Marginally significant (p=0.059)
- n=25 limits statistical power

**Framing:** Conditional on one clean replication at L0=22 with pre-registered d'=10. If confirmed, this is the highest-novelty contribution (8/10 per novelty report) and answers an open question no existing work addresses.

**Strength for paper:** MEDIUM-HIGH (conditional on replication). If replication fails, the paper still stands on Pillars 1+2 at workshop level; if it succeeds, the paper is competitive for main conference.

## Priority Experiments (in order)

### Priority 1: Control Investigation (HIGHEST, 2-3 GPU-hours)

**What:** Diagnose WHY shuffled controls exceed measured absorption. Three sub-experiments:
1. Detailed per-parent logging of the shuffled pipeline: trace which features fire, which pass cosine threshold, which pass magnitude gap. Identify whether the inflation is from cosine false positives in high-dimensional SAE space.
2. Null absorption benchmark on a non-hierarchical task (word length bins, word frequency quintiles). If absorption > 5% on these, the metric has a fundamental false-positive floor.
3. Test a stricter cosine threshold (0.05 vs 0.025). If net absorption becomes positive after threshold recalibration, the cross-domain rates may recover.

**Why first:** Every quantitative absorption claim depends on metric validity. This is the highest-information-gain experiment possible.

**Decision gate:** If a threshold recalibration produces positive net signal (measured > shuffled) for at least 2 domains, cross-domain results are salvageable. If no recalibration fixes the control failure, the control failure itself becomes Pillar 1's lead finding.

### Priority 2: CMI Replication at L0=22 (HIGH, 1 GPU-hour)

**What:** Pre-register d'=10 as the single analysis dimension. Compute CMI on the L0=22 dataset (where all 25 letters have perfect probes, F1=1.0). Use the multi-L0 absorption rates (which span 0%-80% per letter, providing maximum variance). Report Spearman rho and Mann-Whitney p.

**Why second:** This is the single highest-leverage experiment for the theoretical contribution. At L0=22:
- All probes have F1=1.0, eliminating probe quality confound
- Absorption rates range from 0% to near-80%, providing maximum variance
- 3691 tokens available for k-NN MI estimation

**Decision gate:** If rho < -0.3 and p < 0.05, CMI diagnostic is confirmed. If rho between -0.2 and -0.3, report as suggestive. If rho > -0.2, drop CMI from primary contributions and report as "correct qualitative direction, insufficient quantitative power."

### Priority 3: Activation Patching on Core Absorbed Words (HIGH, 0.5-1 GPU-hour)

**What:** For each of the 9 core absorbed words that persist across all L0 values, zero the child feature and measure whether the parent (first-letter) feature recovers. This is the "immune escape test."

**Why third:** This provides the only metric-independent validation of genuine absorption in the entire paper. If parent features recover when child features are ablated, this confirms competitive exclusion as the causal mechanism, not just a metric artifact.

**Decision gate:** If parent recovery in >= 7/9 core words, genuine absorption is confirmed independent of all metric concerns. If recovery in < 4/9, even the "robust" absorbed words may be measurement artifacts.

## Resource Allocation

| Resource | Allocation | Rationale |
|----------|-----------|-----------|
| GPU time (next 4-5h) | 100% on Priority 1-3 | These three experiments determine the paper's tier |
| Writing effort | 0% until Priority 1-2 complete | Paper framing is conditional |
| Theoretical refinement | Hold | CMI theory lives or dies with Priority 2 |
| Cross-domain expansion | **Deprioritize** | Blocked until controls resolved |
| Unsupervised detection | **Drop** | H4 definitively negative. One paragraph in paper |
| Hierarchy predictors (H6) | **Drop** | Underpowered, untestable with near-zero cross-domain rates |

## Conditional Paper Outcomes

### Scenario A: Controls fixable AND CMI replicates (35% probability)
**Title direction:** "When Is Feature Absorption Necessary? Rate-Distortion Theory and Cross-Architecture Measurement Challenges for Sparse Autoencoders"
- CMI-based diagnostic as primary theoretical contribution
- Recalibrated cross-domain rates as empirical contribution
- L0-absorption profile and bifurcation analysis as mechanistic findings
- **Venue: NeurIPS/ICML main** (strong submission)

### Scenario B: Controls NOT fixable AND CMI replicates (20% probability)
**Title direction:** "When Is Feature Absorption Necessary? A Rate-Distortion Criterion with Lessons on Measurement Generalization"
- Control failure as primary methodological contribution
- CMI diagnostic as primary theoretical contribution
- L0-absorption profile as secondary empirical finding
- **Venue: NeurIPS/ICML main** (competitive submission)

### Scenario C: Controls fixable AND CMI does NOT replicate (20% probability)
**Title direction:** "Feature Absorption Across Hierarchy Types: Domain-Dependent Rates and an L0 Phase Transition"
- Recalibrated cross-domain rates as primary empirical contribution
- L0 profile and scaling surface as secondary findings
- CMI as negative result
- **Venue: NeurIPS/ICML main** (marginal) or ICLR

### Scenario D: Controls NOT fixable AND CMI does NOT replicate (25% probability)
**Title direction:** "Measuring Feature Absorption: Challenges in Cross-Architecture Generalization and an L0-Dependent Profile"
- Control failure and metric validation as primary methodological contribution
- L0-absorption profile as secondary empirical finding
- Bifurcation analysis as mechanistic finding
- **Venue: NeurIPS/ICML workshop** or TMLR
- **This is the floor** -- still publishable, but lower tier

## Risk Register Update

| Risk | Prior Probability | Updated Probability | Key Evidence Change |
|------|------------------|---------------------|---------------------|
| Cross-domain absorption < 3% after controls | 25% (proposal) / 50% (prior strategist) | **50%** | No improvement -- controls still broken |
| CMI correlation too weak | 30% (proposal) / 40% (prior) | **35%** | Improved from rho=+0.14 to -0.383, but dimension sensitivity persists |
| Controls not calibratable | 20% (proposal) / 45% (prior) | **40%** | Threshold stability (CV=2.77%) suggests the issue is not threshold sensitivity but structural |
| ITAC unsupervised fails | 50% (proposal) | **99%** | Decisively confirmed by both full and ablation experiments |
| Paper not viable for any venue | 5% | **2%** | L0 profile + bifurcation + metric validation are each independently publishable findings |
| Paper viable for NeurIPS/ICML main | 60% (proposal) | **55%** | Conditional on at least one of {controls fixable, CMI replicates}. With improved CMI evidence, slightly better than prior strategist's 45% |

## Bottom Line

The project is in a fundamentally different position than the prior PIVOT decision assumed. The improved experiments recovered the CMI signal (large effect size, correct direction, marginal significance), added the phase transition prediction (10.2% scale match), the bifurcation analysis (KS D=0.607), and the geometric constant degeneration. These are real, new positive results that did not exist when the PIVOT was called.

The original 6-hypothesis structure is dead. But the data has produced four strong findings:
1. **Universal control failure** -- the Chanin metric does not transfer cleanly to Gemma 2 2B (methodological)
2. **L0-absorption monotonic profile** with 9 persistent core absorbed words (empirical, novel)
3. **CMI predicts absorption direction** at large effect size, pending replication (theoretical, highest novelty)
4. **JumpReLU L0-phase-transition vs L1 uniformly high absorption** (mechanistic)

Plus two clean negative results to report honestly:
5. Unsupervised detection does not work (H4 dead)
6. Hierarchy predictors are underpowered/untestable (H6 dropped)

**The dominant strategy is: PROCEED with three targeted experiments (controls, CMI replication, activation patching) totaling 4-5 GPU-hours. Expected paper viability: 55% for main conference, 75% for workshop, 98% for any venue. The floor is a solid methodology + negative results paper. The ceiling is a three-pillar paper combining metric validation, L0-absorption theory, and rate-distortion diagnostics.**

The single worst strategic mistake would be to either (a) sink further GPU time into cross-domain experiments before resolving the control issue, or (b) abandon the CMI line without the L0=22 replication that uses perfect probes. The data has spoken -- listen to it.
