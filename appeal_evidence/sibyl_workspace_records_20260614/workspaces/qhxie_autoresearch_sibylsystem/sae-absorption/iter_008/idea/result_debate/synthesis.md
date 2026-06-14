# Result Debate Synthesis: SAE Feature Absorption Cross-Domain Characterization

**Synthesized by:** Result Debate Synthesizer (Opus 4.6)
**Date:** 2026-04-15
**Iteration:** 9/10 (Full-Mode Consolidation)
**Perspectives synthesized:** Optimist, Skeptic, Methodologist, Strategist, Comparativist, Revisionist

---

## 1. Consensus Map: Where All Six Perspectives Agree

The following conclusions are endorsed by all six perspectives as high-confidence findings:

### 1.1 Cross-Domain Absorption Variation Is Real (H1: Confirmed)
All perspectives accept that the ANOVA result (hierarchy effect p=0.005, 4/6 pairwise comparisons significant) demonstrates genuine statistical variation in measured absorption rates across hierarchy types. The disagreement is about the *cause* of that variation (see Section 2.1).

### 1.2 Activation Patching Is the Strongest Result (H7: Confirmed)
All perspectives agree that the activation patching result (32.5% recovery vs 1.5% control, Wilcoxon p=0.000218, Cohen's d=1.33) is the most statistically robust finding. Even the Skeptic, who raises valid concerns about word selection, acknowledges the effect size is large and the within-word paired design is sound.

### 1.3 GAS Fails as a Detector (H4: Definitively Refuted)
All six perspectives concur: rho=0.116, AUROC=0.571, confirmed at 25x scale-up. This is a clean, unambiguous negative result. The Skeptic makes an important distinction (the hypothesis fails, not the measure itself), but the practical conclusion is unanimous.

### 1.4 H2' Is Refuted -- First-Letter Is NOT the Least Absorbed
Every perspective agrees that at L24 (best probe quality), first-letter shows the highest absorption (34.5%), not the lowest as the pilot at L12 suggested. The "semantic hierarchies absorb more" narrative from the proposal must be abandoned.

### 1.5 Layer Dependence of Absorption Is Novel and Striking
All perspectives highlight the 15x variation in first-letter absorption across layers (2.2% at L18 to 34.5% at L24) as a genuinely surprising discovery with independent contribution value. The Methodologist specifically notes this is the "most cleanly measured finding" because first-letter probe quality is uniformly excellent across layers.

### 1.6 Tightened Hedging Exposes Near-Tautological Metric
All perspectives agree that the strict/loose hedging decomposition (7.9% strict vs 94.1% loose for first-letter, 86.2% compensatory resolution) is a valuable methodological contribution. The Skeptic offers the alternative framing ("higher L0 trivially adds more features"), but even this framing supports the conclusion that the loose hedging metric is uninformative.

### 1.7 Honest Negative Results Are a Strength
GAS (rho=0.12), CMI (rho=0.044), Absorption Tax (rho=-0.20 ranking, ~50% concordance), and the H2' refutation are unanimously recognized as strengthening the paper's scientific credibility.

---

## 2. Conflict Resolution: Where Perspectives Disagree

### 2.1 CRITICAL: Is the Cross-Domain Variation Genuine or a Probe Quality Artifact?

**The conflict:** The Optimist treats cross-domain variation as a strong finding ("significant and prescient design choice"). The Skeptic and Methodologist classify probe quality confounding as a **fatal flaw** that undermines the primary contribution. The Comparativist calls it a "strong contribution" for novelty of measurement but flags the narrative inconsistency. The Revisionist identifies probe quality as the second-strongest predictor of measured absorption (after layer position).

**Evidence weighed:**
- Probe quality vs hierarchy type is perfectly confounded: first-letter F1=0.971, RAVEL F1=0.789-0.843
- rho(probe_quality, false_negative_rate) = -0.756 (p<0.001) -- the Skeptic's most compelling datapoint
- No probe-quality-controlled analysis exists (no ANCOVA, no degraded-probe ablation)
- The direction at L24 (first-letter highest, not lowest) contradicts the original narrative AND could be explained either by genuine hierarchy effects or by the fact that near-perfect probes detect more true positives in the raw condition, inflating the denominator

**My judgment:** The Skeptic and Methodologist are correct that this is a severe confound, but calling it "fatal" overstates the case. The finding is better characterized as **ambiguous rather than invalidated**. Three considerations moderate the concern:

1. **The variation itself is informative regardless of mechanism.** Whether absorption rates differ because of hierarchy structure or because of probe-hierarchy interactions, the practical implication is the same: single-task absorption benchmarks (like SAEBench first-letter) do not generalize. This is worth publishing.

2. **The probe-quality explanation is not completely clean.** City-continent (F1=0.843, second-best probe) shows absorption comparable to first-letter (35.8% vs 34.5%), while city-country (F1=0.789, worst probe) shows lower absorption (18.5%). If probe quality simply inflated apparent absorption, worse probes should show *higher* rates -- but city-country's worse probe shows *lower* absorption. The Skeptic's S4 point acknowledges this complication.

3. **The remediation is feasible.** The degraded-probe experiment (degrade first-letter probe to F1~0.84) is a 1-2 GPU-hour investment that would resolve the ambiguity. This should be recommended but does not block writing.

**Verdict:** The cross-domain variation finding should be presented as **"measured absorption rates differ significantly across hierarchy types (p=0.005), but this comparison is confounded by differential probe quality (rho=-0.756)"** -- not as a settled conclusion. The paper should (a) report probe-only FN baselines for each hierarchy (Methodologist's Recommendation 2, near-zero cost), and (b) ideally include the degraded-probe ablation. The narrative should frame this as an important open question rather than a proven result.

### 2.2 SERIOUS: How Strong Is the Activation Patching Evidence?

**The conflict:** The Optimist celebrates the dual-role finding and context-dependent dynamics. The Skeptic raises serious concerns about word selection (18/25 are non-English fragments with low raw accuracy). The Methodologist flags the selection bias in the discovery procedure. The Comparativist calls it "moderate" contribution. The Strategist considers it the strongest causal evidence to date.

**Evidence weighed:**
- Full-sample: 32.5% recovery vs 1.5% control, p=0.000218, d=1.33 -- undeniably significant
- Word composition: 7 pilot-core English words + 18 discovered via IG-inspired search
- Many discovered words have very low raw accuracy (xfa=0.10, udy=0.12, uzu=0.06)
- But high-confidence subset (backward=0.50, yaitu=0.455, conmigo=1.0, zorgt=0.50, menjadikan, antigos, recoge) shows substantial recovery
- The discovery procedure is biased toward finding absorption, inflating the 58.8% absorption rate among tested words

**My judgment:** The Skeptic raises a valid concern that must be addressed, but the evidence is stronger than the Comparativist's "moderate" assessment suggests. The correct resolution is:

1. The aggregate p=0.000218 is robust. Even restricting to words with raw accuracy >= 0.50, the high-confidence subset (approximately 8-10 words) likely sustains significance given the large recovery effects (several at 45-100%).

2. The paper should **report both full-sample and high-confidence-restricted analyses** as the Skeptic recommends. This is a low-cost analysis fix.

3. The discovery bias should be transparently disclosed. The 58.8% absorption rate among tested words reflects the selection procedure, not the corpus-wide rate.

4. The claim should be "first interventional evidence that child features suppress parent representations in SAEs, demonstrated on first-letter features with large effect size (d=1.33)" -- not "first causal proof of competitive exclusion" (which implies a specific mechanism the patching does not uniquely identify, as the Methodologist notes).

**Verdict:** The activation patching result is **strong** -- the strongest quantitative finding in the paper -- but requires (a) restricted analysis on high-confidence words, (b) transparent reporting of discovery bias, and (c) careful mechanistic language (absorption/suppression, not exclusively "competitive exclusion").

### 2.3 MODERATE: Venue Tier

**The conflict:** The Strategist recommends ICLR 2027 as primary target (score estimate 7.0-8.0). The Comparativist recommends NeurIPS 2026 MI Workshop as the realistic target, with EMNLP/AAAI as upgrade targets. The gap is significant: top-tier main conference vs workshop.

**My judgment:** The truth lies between these positions, closer to the Strategist's assessment. The reasoning:

1. The Comparativist's analysis is based on the pilot-era evidence (references n=7 for activation patching when the full result is n=25; references n=1 success when 16/19 show positive recovery). The full-mode results substantially strengthen the paper.

2. However, the probe quality confound remains unresolved, and the narrative has been fundamentally restructured. The paper needs the degraded-probe ablation to claim "cross-domain variation is genuine" at a top venue.

3. The combination of (a) novel cross-domain measurement (first of its kind), (b) strong causal evidence (d=1.33), (c) layer-dependence discovery (15x variation), (d) tightened hedging methodology, and (e) transparent negative results is genuinely competitive at ICLR/NeurIPS main, IF the probe confound is adequately addressed.

**Verdict:** Target **ICLR 2027 main conference** with the probe-quality ablation completed. Without it, the paper is competitive for **EMNLP 2026 or COLM 2027**. The NeurIPS MI Workshop assessment undervalues the work's scope.

### 2.4 MINOR: Architecture Comparison (H6)

**All perspectives agree** the architecture comparison is underpowered (p=0.87). The conflict is only about how much space to devote in the paper. The Revisionist says "kill it." The Strategist says "compress to 2-3 paragraphs."

**Verdict:** The Strategist is correct. Report the null result with a power analysis (Skeptic's S2 recommendation), frame the hierarchy effect (p=0.005) as dominating the architecture effect, and limit to one paragraph in the main text. This is an honest negative result that costs minimal space.

### 2.5 MINOR: Cross-Domain Hedging (City-Language n=3)

**All perspectives agree** the city-language hedging decomposition (66.7% strict) is based on n=3 and is statistically meaningless. The Methodologist and Skeptic are most emphatic.

**Verdict:** Do not present the city-language vs first-letter hedging comparison as a finding. Mention it in supplementary material with explicit n=3 caveat. The first-letter hedging decomposition (n=304, bootstrap CI [82.6%, 90.1%]) is the only reliable hedging result.

---

## 3. Result Quality Score: 7.0 / 10

**Justification:**

| Dimension | Score | Weight | Rationale |
|-----------|-------|--------|-----------|
| Statistical rigor | 7/10 | 25% | Strong activation patching stats (d=1.33, p<0.001). Cross-domain ANOVA significant but confounded. Proper corrections applied. |
| Novelty | 8/10 | 25% | First cross-domain absorption measurement. First layer-dependence characterization. First interventional evidence. Verified no competing work. |
| Evidence completeness | 6/10 | 20% | Probe quality ablation missing. Cross-domain causal evidence absent. Architecture comparison underpowered. Hedging cross-domain n=3. |
| Negative results quality | 9/10 | 15% | GAS, CMI, Absorption Tax all cleanly refuted with appropriate statistics. H2' refutation honestly reported. |
| Practical impact | 6/10 | 15% | Shows SAEBench single-task evaluation is insufficient. No concrete mitigation proposed. Activation patching primarily on rare tokens. |

**Overall: 7.0** -- Strong novelty and honest methodology, but the critical probe-quality confound prevents full confidence in the primary cross-domain comparison. The activation patching and layer-dependence findings independently justify publication.

---

## 4. Key Findings: What We Actually Learned

1. **Absorption is layer-dependent with extreme variation.** First-letter absorption varies 15x across layers (2.2% at L18 to 34.5% at L24 for Gemma 2 2B). This is a genuinely novel discovery: no prior work characterized absorption across layers. The implication is that absorption measured at any single layer is unrepresentative, invalidating existing benchmarks that use a fixed layer.

2. **Measured absorption rates differ significantly across hierarchy types, but the cause is ambiguous.** The ANOVA hierarchy effect (p=0.005) is statistically robust, with city-country and city-language showing lower measured absorption than first-letter at L24. However, probe quality (rho=-0.756 with FN rate) is confounded with hierarchy type. The practical conclusion -- that single-task benchmarks are insufficient -- holds regardless of mechanism.

3. **Activation patching provides the first interventional evidence for feature suppression in SAEs.** Zeroing high-IG child features produces 32.5% mean recovery (vs 1.5% control, d=1.33, p<0.001) on first-letter features. This demonstrates that specific child features causally suppress parent representations during SAE encoding. The evidence is primarily from uncommon/non-English tokens; generalization to common English tokens requires further testing.

4. **Loose hedging classification is near-tautological.** 86.2% of false negatives at L0=22 resolve through compensatory (non-parent) features at L0=176, not through the parent feature itself (only 7.9% strict hedging). This means the widely-cited ~98% hedging rate reflects the combinatorial inevitability of higher-L0 SAEs containing more information, not a meaningful absorption-hedging mechanism.

5. **Three proposed unsupervised detectors fail definitively.** GAS (rho=0.12), CMI (rho=0.044), and Absorption Tax ranking predictions (rho=-0.20, concordance ~50%) all fail at scale. Decoder geometry alone does not predict which features will experience absorption. These are valuable negative results that redirect community effort.

---

## 5. Methodology Gaps: Critical Experimental Improvements

### Gap 1: Probe-Quality-Controlled Cross-Domain Comparison (CRITICAL)
**What is missing:** No analysis controls for the differential probe quality across hierarchies (F1 ranges from 0.789 to 0.971). The cross-domain comparison is currently uninterpretable as a hierarchy effect vs a probe-quality effect.

**Remediation (ordered by effort):**
1. **(Low effort, high impact)** Report probe-only FN rate on raw residual stream for each hierarchy x layer. This separates "probe baseline error" from "SAE-induced absorption."
2. **(Medium effort, decisive)** Degrade first-letter probe to F1~0.84 and remeasure absorption. If absorption remains at ~34.5%, hierarchy effect is genuine. If it rises substantially, it is a probe artifact.
3. **(Medium effort, alternative)** Train MLP probes for RAVEL hierarchies to push F1 above 0.90 and remeasure.

### Gap 2: Activation Patching on Representative Tokens
**What is missing:** 18/25 tested words are non-English fragments with low raw accuracy. The causal evidence applies to the long tail of rare tokens, not to the common tokens where absorption matters most for safety and interpretability.

**Remediation:** Report restricted Wilcoxon analysis for words with raw accuracy >= 0.50. If significance holds, add a sentence confirming generalization. If not, scope the causal claim to rare/non-standard tokens explicitly.

### Gap 3: Cross-Domain Causal Evidence
**What is missing:** Activation patching is demonstrated for first-letter only. The paper claims absorption varies across hierarchies but has causal evidence for only one hierarchy type.

**Remediation:** This is a genuine limitation that should be stated explicitly. Cross-domain activation patching requires RAVEL probes above F1=0.90, which is currently not achieved. Frame as future work.

### Gap 4: SAE Reconstruction Quality as Baseline
**What is missing:** No per-layer, per-SAE reconstruction quality metric (MSE, loss recovered) is reported. The layer-dependence finding could reflect differential SAE reconstruction fidelity rather than a genuine absorption mechanism.

**Remediation:** Report MSE or fraction of loss recovered for each SAE configuration. If L24 SAEs have substantially worse reconstruction, the layer-dependence finding needs reinterpretation.

---

## 6. Competitive Position: Where We Stand vs SOTA

**Unique contributions verified against the literature (as of April 2026):**

| Contribution | Prior Art | Our Advance | Competitive Moat |
|---|---|---|---|
| Cross-domain absorption measurement | None -- all prior work uses first-letter only | First systematic measurement on 4 hierarchy types | Strong: no competing work found |
| Layer-dependence characterization | None -- all prior work uses single layer | First measurement showing 15x variation across layers | Strong: simple but nobody has done it |
| Interventional causal evidence | Chanin et al.: correlational only (IG ablation) | Activation patching with proper controls (d=1.33) | Moderate: methodology is standard, application is novel |
| Tightened hedging decomposition | Chanin et al.: binary hedging classification | Strict/compensatory/persistent decomposition showing loose metric is near-tautological | Moderate: methodological refinement |
| GAS/CMI/Tax negative results | No prior tests of these metrics | Definitive negative results at scale | Moderate: prevents others from pursuing dead ends |

**Nearest competitors:**
- **SynthSAEBench (arXiv:2602.14687):** Studies absorption on synthetic hierarchies. Complementary to our real-world measurement, not competitive.
- **Masked Regularization (arXiv:2604.06495, April 2026):** Proposes training-time fix. Orthogonal (mitigation vs measurement).
- **Hierarchical Semantics in SAEs (arXiv:2506.01197):** Models hierarchy in architecture. Does not measure absorption cross-domain.

**Window of opportunity:** No concurrent work measures absorption cross-domain on real LLM hierarchies. The window is open but may not remain so indefinitely. An arXiv preprint after writing is strategically important.

---

## 7. Hypothesis Update

| Hypothesis | Status | Implication for Paper |
|---|---|---|
| H1 (Cross-Domain Variation) | **Confirmed with caveat** | Core finding, but must acknowledge probe quality confound |
| H2' (Semantic > First-Letter) | **Refuted** | Must completely restructure narrative away from "semantic worse" |
| H3 (Hedging Decomposition) | **Confirmed for first-letter; inconclusive cross-domain** | Strong methodological contribution on first-letter (n=304). Drop cross-domain hedging comparison (n=3) |
| H4 (GAS Detector) | **Definitively refuted** | Clean negative result for appendix or supplementary |
| H5 (Absorption Tax) | **Definitively refuted** | Clean negative result; qualitative framework may have pedagogical value only |
| H6 (Architecture Effect) | **Inconclusive (underpowered)** | Report null result with power analysis; one paragraph in main text |
| H7 (Causal Absorption) | **Confirmed** | Feature as primary methodological contribution |
| NEW: Layer Dependence | **Discovered** | Elevate to co-primary contribution alongside cross-domain variation |
| NEW: H8 (Layer-Position Mechanism) | **Proposed** | Testable hypothesis for discussion section |
| NEW: H9 (Probe Quality as Causal Confounder) | **Proposed** | Critical to address; drives Methodology Gap 1 |

---

## 8. Action Plan

### VERDICT: PROCEED -- Enter Writing Stage

The evidence base supports a publishable paper with multiple novel contributions. The probe quality confound is a real limitation but does not invalidate the work -- it constrains the strength of specific claims while the overall contribution (first systematic cross-domain + cross-layer characterization, causal evidence, hedging methodology, negative results) remains substantial.

### Priority Actions (Ordered)

**P0 -- Must Do Before Submission (0-2 GPU hours):**

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | **Report probe-only FN baselines** for each hierarchy x layer (raw residual stream, no SAE) | 0 GPU-hr (data exists) | Resolves whether cross-domain FN differences pre-exist SAE encoding |
| 2 | **Run restricted Wilcoxon** on activation patching words with raw accuracy >= 0.50 | 0 GPU-hr (reanalysis) | Validates causal evidence on representative tokens |
| 3 | **Restructure narrative** from "semantic > syntactic" to "layer-dependent x hierarchy-dependent interaction" | 0 GPU-hr (writing) | Prevents reviewer rejection on narrative inconsistency |
| 4 | **Report SAE reconstruction quality** (MSE or loss recovered) per layer per config | 0 GPU-hr (data likely exists) | Controls for differential reconstruction as confound for layer dependence |

**P1 -- Strongly Recommended (1-2 GPU hours):**

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 5 | **Degraded-probe ablation**: degrade first-letter probe to F1~0.84 and remeasure absorption | 1-2 GPU-hr | Resolves the probe quality vs hierarchy debate decisively |
| 6 | **Power analysis** for architecture comparison: report minimum detectable effect at N=16 | 0 GPU-hr | Converts non-significant p=0.87 from "no effect" to "uninformative" |

**P2 -- Optional Enhancements:**

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 7 | Scale activation patching to n>50 words across hierarchies | 1-2 GPU-hr | Strengthens causal claim but current evidence is already significant |
| 8 | Ecological Phase Transition analysis (Backup B) | 0 GPU-hr (existing data) | Theoretical depth for discussion section |
| 9 | ATM/OrtSAE comparison on cross-domain hierarchies | 2-4 GPU-hr | Enriches architecture comparison but low priority given null H6 |

### Writing Strategy

The paper should be organized around three pillars:

1. **Layer dependence** (primary, strongest evidence): The 15x variation finding with clean probe quality. No confound issues. Novel and striking.

2. **Cross-domain variation** (primary, with caveat): The ANOVA p=0.005 result, presented with probe-quality limitations front and center. Novel measurement regardless of mechanism.

3. **Causal mechanism** (secondary): Activation patching d=1.33 result as the first interventional evidence, with honest sample composition and restricted analysis.

Supporting contributions: tightened hedging (methodology), negative results (GAS, CMI, Tax), architecture null result.

### Target Venue

**Primary:** ICLR 2027 (deadline ~September-October 2026) -- sufficient time for probe ablation and polished writing.
**Backup:** COLM 2027 or EMNLP 2026 -- if probe ablation results weaken the cross-domain claim.
**Immediate:** arXiv preprint upon completion of writing to establish priority on layer-dependence finding.
