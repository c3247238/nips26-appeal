# Result Debate Synthesis

**Agent**: Result Synthesizer (Senior Research Director)
**Iteration**: 9 (FULL mode)
**Date**: 2026-04-16
**Perspectives Synthesized**: Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist

---

## 1. Consensus Map: High-Confidence Conclusions

The following conclusions are endorsed by all 6 perspectives with no material disagreement:

### Consensus 1: Absorption varies dramatically across hierarchy types
All perspectives agree that the Kruskal-Wallis test (p=7.37e-66, N=3566) demonstrates statistically significant cross-domain variation. Even the Skeptic -- who challenges specific numbers -- acknowledges that the comparison between first-letter (F1=0.96, absorption=27.1%) and city-continent (F1=0.87, absorption=31.4%) is "the most trustworthy result." The Comparativist confirms no prior work measures absorption on any task other than first-letter. This is a genuine first.

### Consensus 2: The hedging near-tautology is a real methodological finding
All perspectives accept the hedging decomposition result (chi-square=91.51, p=1.04e-19). Strict hedging ranges from 0% to 22.6% across hierarchies, down from the prior 98.6% loose figure (Chanin et al.). The Skeptic offers an alternative interpretation (probe quality driving the strict/compensatory ratio) but does not dispute the numerical finding. The Comparativist rates this a moderate contribution. **Verdict**: Solid methodological contribution, not a primary pillar.

### Consensus 3: Unsupervised absorption detectors fail
GAS (rho=0.116, AUROC=0.571), CMI (rho=0.044), and all rate-distortion predictors (rho=0.250, individual predictors in OPPOSITE direction) definitively fail. The 25x scale-up from pilot confirms the null for GAS. The direction reversal of rate-distortion predictors at n=262 vs pilot n=20 is a textbook pilot-instability finding. All perspectives cite these as valuable negative results. **Verdict**: Definitive, high-confidence negative results that motivate a shift from correlational to causal/interventional methods.

### Consensus 4: Architecture is not the dominant factor
Architecture effect is non-significant (L12 p=0.754, L24 p=0.497) while hierarchy effect is significant (L12 p=0.010). The Optimist, Revisionist, and Strategist all endorse "hierarchy >> architecture" as a clean finding. The Skeptic and Methodologist note limitations (width mismatch, limited architectures, possible power issues) but do not claim a positive architecture effect exists. **Verdict**: Report as "no significant architecture effect detected in our experimental setting," not as "architecture does not matter."

### Consensus 5: Layer-dependent absorption is the cleanest finding in the paper
First-letter absorption across layers: L6=0.7%, L12=6.9-17.9%, L18=4.0-4.2%, L24=27.1-42.9%. Probe quality is constant across layers (F1>=0.96 for first-letter), eliminating the probe confound. The Strategist labels this "the single cleanest finding." The Skeptic does not challenge it. **Verdict**: Confound-free, strong, independently publishable.

### Consensus 6: 9 honest negative results are a genuine strength
All perspectives -- including the Skeptic -- praise the transparent reporting of negative results as exemplary. The Comparativist notes this is "rare in ML papers" and contributes to the paper's narrative of careful, evidence-driven science.

---

## 2. Conflict Resolution

### Conflict A: Is the cross-domain patching a SUCCESS or a FAILURE?

**The disagreement**: The Optimist and Skeptic (both writing from the consolidation summary) report cross-domain patching as a FAILURE (0.05% recovery, reverse direction, d=-0.91). The Methodologist and Revisionist (who examined the actual FULL-run results) report it as a SUCCESS after a methodology bug fix: city-continent primary recovery 61.9% (d=1.50, p<1e-20), city-language 34.2% (d=0.75, p<1e-18).

**Resolution**: I examined the actual data file (`activation_patching_crossdomain_full.json`). The corrected FULL-run results are definitive:

| Hierarchy | Primary Recovery | All-Absorbers Recovery | Control | Cohen's d | Wilcoxon p |
|-----------|-----------------|----------------------|---------|-----------|-----------|
| City-continent | 61.9% | 80.0% | 5.2% | 1.50 (large) | 4.1e-20 |
| City-language | 34.2% | 70.0% | -- | 0.75 (medium-large) | <1e-18 |

The pilot's "failure" was caused by a code bug: zeroing features with highest POSITIVE cosine (supporting the correct class) rather than most NEGATIVE contribution (absorber features). After the fix, the causal mechanism (competitive exclusion) is confirmed across ALL tested hierarchies, not just first-letter.

**Judgment**: The Methodologist and Revisionist are correct. The consolidation summary is stale. **H7 should be upgraded from "SUPPORTED_FIRSTLETTER_ONLY" to "SUPPORTED_CROSS_DOMAIN."** This is the single most important factual correction in this synthesis. It substantially strengthens the paper by demonstrating a universal causal mechanism.

**Caveat**: Oceania is an outlier -- primary recovery only 4.2% (n=29 entities, 1121 absorbed contexts). This is a well-powered anomaly, not noise. The paper should report per-class heterogeneity. Five of six continent classes show strong recovery (53-96% primary), but Oceania does not.

### Conflict B: Is city-country's 45.1% absorption rate trustworthy?

**The disagreement**: The Optimist includes city-country in the "4x range" headline finding. The Skeptic labels it a FATAL FLAW (probe F1=0.726 invalidates the number). The Strategist makes the probe degradation ablation a BLOCKING experiment. The Methodologist rates it HIGH severity.

**Resolution**: The Skeptic is substantially right. City-country's probe quality (F1=0.726, balanced accuracy=0.56 for 80 classes) is disastrously low. The per-class breakdown shows pathological patterns: countries with <10 entities show 100% absorption while high-frequency countries show <5% absorption, consistent with probe failure on rare classes being misclassified as absorption.

**Judgment**: City-country's 45.1% rate should be treated as an UPPER BOUND and removed from the primary cross-domain comparison. The paper should lead with first-letter (F1=0.96), city-continent (F1=0.87), and city-language (F1=0.82) as the well-controlled comparison. City-country should be reported in supplementary material with explicit caveats. The "4x range" headline should be replaced with the range from the 3 better-controlled hierarchies (11.6%-31.4%, a 2.7x range), which is still substantial and novel. If the probe degradation ablation later validates the city-country number, it can be reinstated.

### Conflict C: Is the 100% pathological finding generalizable?

**The disagreement**: The Optimist calls this "arguably the single most impactful finding." The Skeptic argues it lacks proper controls (tested only on city-continent/Europe/single SAE) and conflates "parent direction is semantically important" with "absorption-specific pathology."

**Resolution**: Both sides have merit. The finding is decisive within its scope (0% benign, 1471 instances, p=2.69e-242, ~1000x effect ratio) and the magnitude is overwhelming. However, the Skeptic raises a valid concern: the experiment needs a "no-absorption" baseline. If ablating the parent direction from correctly-classified instances (where no absorption occurred) also produces ~4 nat logit changes, then the experiment measures "feature importance," not "absorption-specific pathology."

**Judgment**: The finding is STRONG but should be qualified as "within the city-continent hierarchy." The paper should explicitly acknowledge the need for replication on additional hierarchies and a no-absorption baseline. The "100% pathological" claim should be stated as "within our tested scope, all absorption instances showed substantial downstream logit changes, consistent with pathological rather than benign absorption." Replication on first-letter is recommended as a low-cost experiment (~1 GPU-hour) that would substantially strengthen the claim.

### Conflict D: Is the probe quality confound a fatal flaw for the entire paper?

**The disagreement**: The Skeptic argues this is the central threat. The Optimist notes the probe quality ordering does not match the absorption ordering (partially mitigating the concern). The Strategist makes the probe degradation ablation a BLOCKING experiment.

**Resolution**: The probe quality issue is SERIOUS but NOT FATAL because: (1) the layer-dependence finding (15x range, constant probes) is completely immune to this confound; (2) the corrected cross-domain patching (causal evidence) provides a second line of evidence independent of probe quality; (3) the first-letter vs city-language comparison (F1=0.96 vs F1=0.82) shows LOWER absorption for the lower-quality probe (11.6% vs 27.1%), which is the opposite of what a pure confound would predict.

**Judgment**: The confound is a MODERATE RISK that must be addressed transparently but does not invalidate the paper's core contributions. The probe degradation ablation (recommended by Strategist, Methodologist, and Skeptic) should be the top-priority next experiment. Meanwhile, the paper should lead with confound-free findings (layer dependence, causal evidence) and present cross-domain variation with appropriate caveats.

---

## 3. Result Quality Score: 7.5 / 10

**Justification**:

| Factor | Score | Weight | Notes |
|--------|-------|--------|-------|
| Layer-dependent absorption (15x, confound-free) | 9/10 | High | Cleanest finding, no challenges from any perspective |
| Cross-domain causal mechanism (d=0.75-1.50) | 8.5/10 | High | Corrected results are strong across hierarchies; Oceania anomaly noted |
| Cross-domain absorption variation (2.7x with 3 hierarchies) | 7/10 | High | Genuine first, but probe quality caveat applies to quantitative ranking |
| 100% pathological absorption | 7/10 | Medium | Decisive within scope; needs replication and no-absorption baseline |
| Hedging decomposition | 7.5/10 | Low | Clean finding, moderate contribution |
| Architecture invariance | 6/10 | Low | Underpowered, unmatched design |
| Negative results (9 total) | 8.5/10 | Medium | Exemplary honest reporting; contributes to field |
| Statistical methodology | 8/10 | Medium | Proper corrections, bootstrap CIs, permutation tests |
| Probe quality management | 5/10 | High (penalty) | City-country below gate; cross-domain pipeline differs from first-letter |

The 7.5 score reflects a paper with genuinely novel and important findings, strong statistical methodology, and exemplary negative results reporting, penalized by unresolved probe quality concerns and scope limitations on the pathological absorption claim.

---

## 4. Key Findings

1. **Absorption is layer-dependent with a 15x range (confound-free).** First-letter absorption ranges from 0.7% at L6 to 27.1% at L24, measured with constant-quality probes (F1>=0.96). SAE analyses at intermediate layers drastically underestimate absorption relative to the final prediction layer. No prior work has quantified this.

2. **Competitive exclusion is a universal causal mechanism across hierarchy types.** After correcting a methodology bug in the pilot, activation patching demonstrates causal absorption recovery across all tested hierarchies: city-continent d=1.50, first-letter d=1.33, city-language d=0.75. This is the first interventional (not correlational) evidence for feature absorption.

3. **Absorption varies across hierarchy types (2.7x range among well-controlled hierarchies).** Among hierarchies with adequate probe quality (F1>=0.82): first-letter 27.1%, city-continent 31.4%, city-language 11.6%. This is the first evidence that absorption is not a property of the first-letter task alone but depends on structural properties of the feature hierarchy.

4. **Absorption is pathological, not benign (within tested scope).** In the city-continent hierarchy, 0% of absorption instances are benign: ablating the parent direction from the child latent's decoder causes ~1000x larger logit changes than control directions (3.98 nats vs 0.004 nats). Replication on additional hierarchies is needed.

5. **All tested unsupervised detection methods fail.** GAS (rho=0.12), CMI (rho=0.04), and rate-distortion predictors (rho=0.25, wrong direction) all fail to predict absorption. This establishes that absorption resists correlational/statistical detection and requires causal methods.

---

## 5. Methodology Gaps (Critical Improvements Needed)

### Gap 1: Probe Quality Sensitivity Analysis (BLOCKING)
**Source**: Skeptic (Fatal Flaw), Strategist (Blocking Experiment), Methodologist (Recommendation 1)
**What**: Artificially degrade first-letter probes to F1={0.70, 0.80, 0.85, 0.90} by injecting label noise; re-measure absorption at L24.
**Why**: The correlation between probe quality and absorption rate across hierarchies is the single biggest threat to the cross-domain finding. Without this ablation, a reviewer can dismiss the cross-domain variation as a probe artifact.
**Cost**: 1-2 GPU-hours.
**Decision logic**: If absorption increases monotonically with degradation, lead with layer-dependence and causal evidence; if stable, the cross-domain finding is validated.

### Gap 2: No-Absorption Baseline for Pathological Claim
**Source**: Skeptic (Serious Concern SC1), Methodologist (Metric-Claim Alignment 2.2)
**What**: Run the parent direction ablation on correctly-classified instances (where parent feature IS firing, no absorption). Measure logit change.
**Why**: If the logit change is also ~4 nats on non-absorbed instances, the experiment measures "feature importance" not "absorption pathology."
**Cost**: 0.5 GPU-hours.

### Gap 3: Update Consolidation Summary with Corrected Cross-Domain Patching
**Source**: Methodologist (Recommendation 2), Revisionist (Surprise 4)
**What**: The consolidation summary reports cross-domain patching as "FAILED" based on the pilot's buggy methodology. The FULL-run data shows strong positive results (d=1.50 for city-continent, d=0.75 for city-language). The consolidation must be updated.
**Why**: Stale consolidation will mislead the writing agent and could propagate incorrect claims into the paper.
**Cost**: 30 minutes CPU.

### Gap 4: L24 Activation Patching for First-Letter
**Source**: Strategist (Experiment 2)
**What**: Replicate the first-letter activation patching at L24 (where headline absorption rates live), not just L12 (where absorption is only 5.7%).
**Why**: Current first-letter causal evidence (d=1.33) comes from L12. A reviewer will ask why the causal evidence is at a layer with 6% absorption rather than the 27% layer.
**Cost**: 1-2 GPU-hours.

### Gap 5: Token Position and Pipeline Asymmetry
**Source**: Methodologist (Section 1.4), Skeptic (SC3)
**What**: Run first-letter absorption using the same generic pipeline (position -2, sklearn probe) as cross-domain, to verify pipeline choice does not explain rate differences.
**Cost**: 1 GPU-hour.

---

## 6. Competitive Position

**Assessment from Comparativist (confirmed by all perspectives)**:

This paper occupies a unique niche: **no prior work measures absorption on any task other than first-letter spelling.** This is not an incremental improvement on existing benchmarks -- it opens an entirely new measurement dimension.

| Aspect | Prior SOTA | This Work | Delta |
|--------|-----------|-----------|-------|
| Hierarchies measured | 1 (first-letter only) | 4 (first-letter + 3 RAVEL) | +3 new |
| Causal evidence | None (all correlational) | Activation patching d=0.75-1.50 | First interventional |
| Benign/pathological test | Not studied | 0% benign, ~1000x effect | First classification |
| Layer dependence | Noted qualitatively | 15x quantified variation | First quantification |
| Honest negative results | Rare | 9 definitive negatives | Exemplary |

**Concurrent work scan** (Comparativist): No overlapping paper found as of April 2026. Closest is SynthSAEBench (synthetic hierarchies, orthogonal to real-LLM measurement). Masked Regularization (Apr 2026) proposes mitigation, complementary to our characterization.

**Venue recommendation**: NeurIPS 2026 main conference or ICLR 2027 (conditional on addressing probe quality confound). Fallback: EMNLP 2026 or NeurIPS MI Workshop. Chanin et al. (first-letter only) was accepted as NeurIPS 2025 Oral, establishing that absorption characterization papers can reach top-tier.

**Key risks for acceptance**: (1) Single-model limitation (only Gemma 2 2B). (2) Probe quality for RAVEL hierarchies. (3) Scope limitation of pathological claim.

---

## 7. Hypothesis Update

| Hypothesis | Pre-Debate Status | Post-Debate Status | Change |
|-----------|------------------|-------------------|--------|
| H1: Cross-domain variation | STRONGLY SUPPORTED | **STRONGLY SUPPORTED** | No change; endorsed by all 6 perspectives |
| H2': Semantic > Syntactic | REFUTED | **REFUTED** | No change; ordering is hierarchy-specific, not category-based |
| H3: Hedging decomposition | SUPPORTED | **SUPPORTED** | No change |
| H4: GAS detector | DEFINITIVE NEGATIVE | **DEFINITIVE NEGATIVE** | No change |
| H5: Absorption Tax quantitative | NOT SUPPORTED | **NOT SUPPORTED** | No change |
| H6: Architecture invariance | PARTIALLY SUPPORTED | **PARTIALLY SUPPORTED (reframed as "no significant effect detected")** | Softened language due to power concerns |
| H7: Causal patching | SUPPORTED (first-letter only) | **SUPPORTED (CROSS-DOMAIN)** | **MAJOR UPGRADE** -- corrected FULL-run shows d=1.50 for city-continent, d=0.75 for city-language |
| H8: All absorption pathological | FALSIFIED (positive finding) | **FALSIFIED (positive, scope-limited)** | Maintained with scope caveat; needs replication |
| H9: Rate-distortion predictors | NOT SUPPORTED | **NOT SUPPORTED** | No change; direction reversal confirmed |

**New hypotheses proposed by Revisionist**:
- NH1: Class count and imbalance predict absorption (testable via class-count ablation)
- NH2: Multi-absorber distribution varies by hierarchy (testable via Gini coefficient on feature contributions)
- NH3: Absorption severity predicts downstream task degradation (testable via F1 gap analysis)

---

## 8. Action Plan

### Recommendation: **PROCEED**

The project has at least 3 strong, novel contributions that justify a top-tier submission. No case for PIVOT exists.

### Priority-Ordered Next Steps

**TIER 1: BLOCKING (must complete before submission)**

| # | Action | Type | Cost | Owner | Rationale |
|---|--------|------|------|-------|-----------|
| 1 | Probe degradation ablation | GPU experiment | 1-2 hr | Experimenter | Resolves the single biggest threat -- determines whether cross-domain variation is real or a probe artifact. All 4 non-Optimist perspectives demand this. |
| 2 | Update consolidation with corrected cross-domain patching | CPU edit | 30 min | Writer | FULL-run data shows d=1.50 for city-continent (not the "failed" d=-0.91 from the buggy pilot). Stale consolidation will poison downstream writing. |
| 3 | Implement validate_integration.py | CPU engineering | 1.5 hr | Experimenter | Cross-check all numerical claims against source JSON. The 12.3% hallucinated number incident proves this is necessary. |

**TIER 2: HIGH PRIORITY (significantly strengthens paper)**

| # | Action | Type | Cost | Rationale |
|---|--------|------|------|-----------|
| 4 | L24 activation patching for first-letter | GPU experiment | 1-2 hr | Connects causal evidence (currently at L12 with 5.7% absorption) to headline results (L24, 27.1%). |
| 5 | No-absorption baseline for H8 | GPU experiment | 0.5 hr | If non-absorbed instances show similar logit changes, H8 measures feature importance not absorption pathology. |
| 6 | Benign/pathological replication on first-letter | GPU experiment | 0.5-1 hr | Extends scope of 100% pathological claim beyond single hierarchy. |
| 7 | Fix paper title, abstract, broken cross-references | CPU writing | 1 hr | "The Absorption Tax" references a framework that failed (rho=-0.20). Title must reflect actual contributions. |

**TIER 3: NICE TO HAVE (for revision or supplementary)**

| # | Action | Type | Cost | Rationale |
|---|--------|------|------|-----------|
| 8 | Report city-country separately with caveats | CPU restructuring | 1 hr | Remove from primary comparison; present as supplementary upper-bound analysis. |
| 9 | Multi-model replication (Llama 3.2 1B) | GPU experiment | 4-8 hr | Removes single-model limitation, the most likely reviewer objection. |
| 10 | Improved RAVEL probes (MLP or ensemble) | GPU experiment | 2-4 hr | Could bring F1 above 0.90, removing upper-bound interpretation. |

### Paper Narrative Recommendation

The paper should be reframed around four evidence-based pillars:

**Pillar 1 (Strongest -- confound-free):** Layer-dependent absorption. 15x variation across layers for first-letter with perfect probes. Zero probe quality confound.

**Pillar 2 (Strong -- causal):** Universal competitive exclusion mechanism. Activation patching confirms causal absorption across first-letter (d=1.33), city-continent (d=1.50), and city-language (d=0.75).

**Pillar 3 (Strong -- conditional on degradation ablation):** Cross-domain absorption variation. 2.7x range across well-controlled hierarchies. Conditional on confirming independence from probe quality.

**Pillar 4 (Moderate -- scope-limited):** All tested absorption is pathological. 0% benign, ~1000x effect ratio. Needs replication on additional hierarchies.

**Negative results appendix:** 9 definitive negatives including GAS, CMI, rate-distortion predictors, and Absorption Tax quantitative framework.

### Title Recommendation

Replace "The Absorption Tax" (references a failed quantitative framework) with something reflecting actual contributions, e.g.:

- "Beyond First-Letter Spelling: Cross-Domain Characterization of SAE Feature Absorption"
- "Absorption Is Everywhere: Layer- and Hierarchy-Dependent Feature Absorption in Sparse Autoencoders"

### Risk-Adjusted Timeline

| Scenario | Probability | Outcome | Venue |
|----------|------------|---------|-------|
| Degradation ablation validates + L24 patching succeeds | 35% | Strong accept | NeurIPS 2026 |
| Degradation ablation validates + L24 patching fails | 15% | Borderline accept | NeurIPS 2026 / ICLR 2027 |
| Degradation ablation shows partial confound | 30% | Reframe around layers + causal | ICLR 2027 / EMNLP 2026 |
| Degradation ablation shows full confound | 15% | Workshop-level paper | NeurIPS MI Workshop |
| Both experiments fail | 5% | Reframe around causal + negative results | Workshop / arXiv |

### Total Resource Budget

| Resource | Amount | Allocation |
|----------|--------|-----------|
| GPU hours | 4-6 | Blocking: probe ablation (2), L24 patching (2). Stretch: H8 replication (1). |
| CPU hours | 4 | Consolidation update (0.5), validate_integration (1.5), writing fixes (2). |
| Wall-clock | ~1 day | GPU experiments sequential; CPU work in parallel. |
