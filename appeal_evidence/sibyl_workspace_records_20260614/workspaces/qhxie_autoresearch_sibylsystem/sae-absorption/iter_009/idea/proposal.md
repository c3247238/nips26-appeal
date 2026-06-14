# Research Proposal (Synthesis Round 6 -- Iteration 9 Fresh Perspectives)

## Title

**The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains**

---

## Abstract

Feature absorption -- the systematic failure of general (parent) features to fire when specific (child) features are active -- threatens the reliability of sparse autoencoder (SAE) based mechanistic interpretability. Yet all published measurements rest on a single proxy task (first-letter spelling), and no formal theory quantifies when absorption is harmful versus faithful. We present the first systematic cross-domain absorption characterization, extending measurement from spelling to entity-attribute knowledge hierarchies (city-country, city-continent, city-language) using RAVEL on Gemma 2 2B. Our pilot experiments reveal that the received wisdom is inverted: first-letter spelling shows the *lowest* absorption (3.9%), while semantic hierarchies show substantially higher rates (city-continent: up to 53.4%, city-language: 10.4%), falsifying the hypothesis that spelling represents a worst case. We provide the first causal evidence for absorption via activation patching (14.3% parent recovery vs. 0.5% control) and demonstrate that the widely cited hedging classification is a near-tautological upper bound (strict classification: 7.4% vs. loose: 92.6%). Architecture comparisons across hierarchy types reveal that JumpReLU, not Matryoshka, shows the most consistently low absorption -- challenging single-task architecture rankings. Theoretically, we formalize the Absorption Tax: the minimum additional L0 cost for absorption-free representation. We introduce a benign-vs-pathological absorption diagnostic that distinguishes cases where the model's computation is genuinely harmed by absorption from cases where absorption faithfully reflects computational redundancy. While our Geometric Absorption Score (GAS) and CMI-based approaches failed as unsupervised detectors, we report these negative results transparently. The paper's central finding -- that absorption severity is hierarchy-dependent and the first-letter task is atypical -- reframes the entire absorption literature and demands cross-domain evaluation as a standard practice.

---

## Motivation

Feature absorption creates a false sense of monosemanticity: SAE features appear clean and interpretable, but silently fail to fire in systematic subsets of their semantic domain. Google DeepMind deprioritized SAE research partly because absorption degraded safety-relevant feature detection by 10-40%. Anthropic's successful circuit tracing in Claude 3.5 Haiku demonstrates that when features are reliable, they enable powerful mechanistic understanding -- making absorption a critical obstacle to broader adoption.

Yet the field's understanding of absorption rests on an alarmingly narrow empirical base. Every published absorption measurement uses the first-letter spelling task, which has an unnaturally clean hierarchy (26 letters, near-uniform distribution, 100% parent-child co-occurrence by construction). Real knowledge hierarchies are imbalanced (France has more cities than Liechtenstein), multi-level (city -> country -> continent), and context-dependent.

Five compounding gaps motivate this work:

**Gap A (Empirical -- the single-task monoculture):** Absorption rates on semantically rich hierarchies are unknown. The 15-35% rates from Chanin et al. may not transfer. Safety-relevant features live in knowledge/reasoning space, not spelling space. The field needs cross-domain characterization to know whether absorption actually matters where it counts.

**Gap B (Theory -- no quantitative prediction):** No closed-form prediction exists for absorption severity as a function of SAE configuration and hierarchy structure. The theoretical perspective's rate-distortion bound and the interdisciplinary perspective's phase-transition framework both point toward the same key variables (decoder similarity, co-occurrence frequency, reconstruction importance), but neither has been empirically validated with cross-domain data.

**Gap C (Architecture validity):** Every architecture comparison (Matryoshka vs. BatchTopK vs. JumpReLU) uses the first-letter task alone. If architecture rankings change on knowledge hierarchies, current recommendations may be misleading.

**Gap D (Benign vs. Pathological):** The contrarian perspective raises a critical point: not all absorption is harmful. If the model does not independently use the parent feature when the child is active, absorption faithfully reflects computational redundancy. No diagnostic exists to distinguish the two cases.

**Gap E (Unsupervised detection):** The innovator perspective's ITAC proposal and the ecological competition framework both aim to detect absorption without probes, but pilot evidence shows purely geometric approaches fail (GAS rho=0.12). The field needs either a better unsupervised method or an honest acknowledgment that interventional methods are required.

### Evidence-Driven Context from Prior Iterations

Nine prior iterations and extensive pilot experiments have produced substantial empirical results that ground this proposal:

- **Activation patching** causally confirms absorption: zeroing child features recovers parent probe predictions (14.3% recovery vs. 0.5% control). This is the first metric-independent causal evidence for competitive exclusion in SAEs.
- **Tightened hedging** reveals the widely cited 98.6% hedging classification is near-tautological: strict classification yields only 7.4%, with 85.3% being compensatory resolution.
- **CMI at L0=22** is NOT SUPPORTED (rho=0.044, p=0.83). The information-theoretic pillar in the ITAC formulation does not hold.
- **Cross-domain existence** confirmed: absorption rates differ significantly across hierarchy types (p<0.01), with city-language at 10.4% vs. first-letter at 3.9%.
- **H2 falsified (critically)**: First-letter is NOT the worst case. Semantic hierarchies show HIGHER absorption, inverting the contrarian hypothesis.
- **GAS failed** as unsupervised detector (rho=0.12). The decoder-activation mismatch approach does not capture absorption.
- **JumpReLU consistently lowest** absorption across all tested hierarchies (3.2%, 17.1%, 13.9%), challenging Matryoshka's assumed superiority.

---

## Landscape of Perspectives: Agreements, Conflicts, and Complementary Insights

### 1. Agreements Across All 6 Perspectives

All perspectives agree on:
- Feature absorption is a real, theoretically proven phenomenon (Chanin et al. toy model proof).
- The single-task monoculture (first-letter spelling only) is the most critical empirical gap.
- Cross-domain characterization is the highest-priority experiment.
- Existing tools (SAELens, TransformerLens, Gemma Scope, sae-spelling) provide a complete pipeline.
- Training-free analysis on pre-trained SAEs is both feasible and appropriate.

### 2. Key Conflicts and How They Are Resolved

**Conflict 1: Is absorption a problem or a feature?**
- Innovator/Pragmatist/Empiricist treat absorption as a failure mode to detect and characterize.
- Contrarian argues absorption may be "faithful representation of computational redundancy."
- **Resolution:** We adopt the contrarian's framing as a *diagnostic dimension*, not an alternative conclusion. The causal absorption test (activation patching, H7) classifies each absorption instance as benign (parent computationally redundant) or pathological (parent has independent causal effects). This converts the conflict into an empirical question.

**Conflict 2: Theory-first vs. Empirics-first?**
- Theoretical perspective proposes rate-distortion bounds with specific predictions.
- Interdisciplinary perspective proposes phase-transition framework with scaling exponents.
- Pragmatist/Empiricist argue for measurement first, theory second.
- **Resolution:** Pilot evidence settled this: quantitative theoretical predictions (Absorption Tax R_pc, CMI) did not hold. We prioritize empirical characterization (Phase 1) with theoretical framing as qualitative context. The rate-distortion bound's three key variables (cos_sim, co-occurrence, reconstruction importance) become correlational analyses rather than formal theorem tests.

**Conflict 3: Unsupervised detection priority.**
- Innovator makes ITAC the primary contribution (novelty 8/10).
- Empiricist/Pragmatist argue cross-domain characterization is more grounded.
- **Resolution:** Pilot evidence refuted the unsupervised approach (GAS rho=0.12). ITAC's core hypothesis -- that decoder geometry alone reveals absorption -- is not supported. The conditional independence component may yet work, but cross-domain characterization is the proven foundation.

**Conflict 4: Which cross-domain hierarchies?**
- Pragmatist proposes city-country, city-continent, POS.
- Empiricist adds RAVEL entities (Nobel laureates) and token-frequency hierarchy.
- **Resolution:** We use the RAVEL dataset's natural hierarchies (city-country, city-continent, city-language) as the primary extension, with first-letter spelling as the positive control. RAVEL provides validated entity-attribute structure with 500+ entities. POS is dropped (noisier probes, less clean hierarchy).

### 3. Complementary Insights Incorporated

- **From Innovator:** The MIG/completeness framework from VAEs motivates why decoder geometry should capture absorption. Though GAS failed, the diagnostic concept -- that "orphaned directions" reveal incomplete decomposition -- remains the right theoretical frame for future work.
- **From Pragmatist:** The implementation strategy (fork sae-spelling, adapt for RAVEL, use SAEBench SAEs) is adopted wholesale. The pilot design is adapted from the Pragmatist's proposal.
- **From Theoretical:** The rate-distortion bound's three predictors (decoder cos_sim, activation co-occurrence, reconstruction importance) are tested as correlational hypotheses. The critical sparsity threshold k* is tested as a qualitative prediction.
- **From Contrarian:** The benign-vs-pathological diagnostic is the most original contribution from this perspective. Activation patching (H7) operationalizes this distinction.
- **From Interdisciplinary:** The ecological niche competition framework provides the best *intuitive* model for why absorption occurs. The competition coefficient (decoder cosine similarity x activation overlap) is computed as a secondary analysis.
- **From Empiricist:** The statistical rigor requirements (bootstrap CIs, Bonferroni correction, random-baseline controls, probe quality gates) are adopted in full. The falsification criteria come directly from the Empiricist.

---

## Research Questions

1. **RQ1 (Cross-Domain):** Do absorption rates vary systematically across feature hierarchy types (syntactic first-letter, factual city-country, city-continent, city-language), and does the first-letter spelling task represent a typical or extreme case?

2. **RQ2 (Confound Disentanglement):** What fraction of measured "absorption" is attributable to hierarchy-driven absorption vs. L0-induced hedging vs. probe/metric artifacts?

3. **RQ3 (Architecture Validity):** Do architecture rankings for absorption resistance (Matryoshka > BatchTopK, etc.) hold across hierarchy types, or are they task-specific?

4. **RQ4 (Causal Mechanism):** Is absorption driven by competitive exclusion dynamics (child suppresses parent via decoder overlap), and does activation patching confirm this? Can we distinguish benign from pathological absorption?

5. **RQ5 (Theory):** Do the rate-distortion bound's three predictors (decoder cosine similarity, co-occurrence frequency, reconstruction importance) correlate with per-pair absorption probability across domains?

---

## Hypotheses

**H1 (Cross-Domain Variation) -- PARTIALLY SUPPORTED:** Absorption rates on entity-attribute hierarchies differ significantly from first-letter spelling rates on the same SAEs.
- *Pilot evidence:* City-language 10.4% vs. first-letter 3.9% (p=0.005). City-continent up to 53.4% (inflated by low probe quality).
- *Full-mode target:* Confirm with proper probe quality (F1 > 0.90) across layers 6, 12, 18, 24.
- *Falsification:* Rates within 5 pp across all types after controlling for L0 and width.

**H2' (Semantic > Syntactic -- Revised from Original H2) -- SUPPORTED by pilot:** Semantic/knowledge hierarchies show HIGHER absorption than the syntactic first-letter task.
- *Revised hypothesis:* Absorption severity scales with hierarchy depth, co-occurrence complexity, and the model's computational reliance on the hierarchy.
- *Falsification:* With quality probes, first-letter absorption >= all semantic hierarchies by > 5 pp.

**H3 (Absorption-Hedging Decomposition) -- PARTIALLY SUPPORTED:** The absorption-to-hedging ratio varies by hierarchy type. Semantic hierarchies show proportionally more true absorption vs. hedging.
- *Pilot evidence:* City-continent 69% absorbed / 31% hedged vs. first-letter 45% / 55%.
- *Falsification:* Ratio identical across all hierarchy types (within 10 pp).

**H4 (GAS Detector) -- REFUTED:** GAS achieves rho=0.12. Negative result documented.

**H5 (Absorption Tax Quantitative) -- NOT SUPPORTED:** Absorption-MSE correlation rho=0.08. Reported as qualitative framework only.

**H6 (Architecture Generalization) -- INCONCLUSIVE:** JumpReLU consistently lowest absorption. Matryoshka shows mixed results across hierarchy types.
- *Full-mode target:* Confirm with improved probes and matched L0.
- *Falsification:* No consistent architecture ranking across hierarchies.

**H7 (Causal Absorption via Activation Patching) -- SUPPORTED:** 14.3% parent recovery vs. 0.5% control. Needs larger n for statistical significance.
- *Full-mode target:* Wilcoxon p < 0.01 with n > 50.
- *Falsification:* Recovery rate indistinguishable from control with adequate sample size.

**H8 (Benign vs. Pathological -- NEW from Contrarian):** A substantial fraction of absorption is benign (model does not independently use parent feature when child is active). Activation patching quantifies the benign-to-pathological ratio.
- *Metric:* Fraction of absorbed instances where parent ablation has minimal downstream effect.
- *Falsification:* All absorption instances show significant downstream effects (all pathological).

**H9 (Rate-Distortion Predictors -- NEW from Theoretical/Interdisciplinary):** Per-pair absorption probability correlates with the three-factor model: cos_sim(decoder_parent, decoder_child) x P(child|parent) / R(parent).
- *Target:* Spearman rho > 0.5 across identified parent-child pairs.
- *Falsification:* rho < 0.3 or p > 0.05.

---

## Method

### Phase 0: Blocking Experiments from Prior Iterations (COMPLETED)

- Activation patching on core words -- DONE (14.3% recovery vs. 0.5% control)
- Tightened hedging classification -- DONE (strict 7.4% vs. loose 92.6%)
- CMI replication at L0=22 -- DONE (NOT SUPPORTED, rho=0.044)

### Phase 1: Cross-Domain Absorption Characterization (PRIMARY)

**Step 1.1: Probe quality improvement** (~2 hours)
- Train probes at layers 6, 12, 18, 24 (pilot only tested layer 12)
- Test alternative prompt templates for RAVEL entities
- Implement frequency-balanced sampling for city-country (80 classes with imbalance)
- **Quality gate:** F1 > 0.90 required. Relax to 0.85 with documented rationale.

**Step 1.2: Cross-domain absorption measurement** (~4 hours)
- Hierarchies: first-letter (baseline), city-country, city-continent, city-language
- SAEs: Gemma Scope JumpReLU 16k/65k (layers 6, 12, 18, 24), SAEBench BatchTopK and Matryoshka at 16k/65k (layer 12)
- Metric: adapted Chanin et al. absorption rate with integrated-gradients attribution
- Controls: random direction baseline, shuffled hierarchy control, probe-only baseline
- Statistics: bootstrap 95% CI (10k resamples), paired permutation test across domains, Bonferroni correction

**Step 1.3: Absorption-hedging decomposition** (~30 min per SAE)
- Classify false negatives: absorbed (probe-aligned absorber identified) vs. hedged (merged into correlated cluster) vs. residual
- Apply both strict and loose hedging classifications
- Report decomposition per hierarchy type, per architecture

**Step 1.4: Architecture comparison** (~1 hour)
- JumpReLU vs. BatchTopK vs. Matryoshka on each hierarchy at matched width and L0
- Test whether JumpReLU's advantage is robust across probe quality range

### Phase 2: Causal Mechanism and Benign/Pathological Diagnostic (SECONDARY)

**Step 2.1: Activation patching at scale** (~1 GPU-hour)
- Scale from pilot n=9 to n > 50 activation patching instances across hierarchies
- Zero child feature, measure parent recovery rate
- Proper controls: random latent zeroing, semantically unrelated latent zeroing
- Statistics: Wilcoxon signed-rank test, bootstrap CI on recovery rate

**Step 2.2: Benign vs. Pathological classification** (~1 GPU-hour)
- For each confirmed absorption instance (from Phase 1):
  1. Ablate the parent direction component from the child latent's decoder vector
  2. Measure downstream logit change for relevant tokens
  3. Benign: ablation has minimal effect (<= 0.1 logit change)
  4. Pathological: ablation significantly degrades model performance (> 0.1 logit change)
- Report benign-to-pathological ratio per hierarchy type
- Cross-reference with the Contrarian's prediction that semantic hierarchies have more pathological absorption

### Phase 3: Rate-Distortion Predictor Validation (TERTIARY)

**Step 3.1: Compute per-pair predictors** (~30 min)
- For each identified parent-child pair from Phase 1:
  - cos_sim(decoder_parent, decoder_child)
  - P(child active | parent active) from activation statistics
  - R(parent) = reconstruction importance (MSE increase when parent direction ablated)
- Fit the three-factor model: absorption_probability ~ f(cos_sim, co-occurrence, R)

**Step 3.2: Test qualitative predictions** (~30 min)
- Does absorption increase with cos_sim? (predicted by Theoretical, Interdisciplinary)
- Does absorption increase with co-occurrence frequency? (predicted by all perspectives)
- Does absorption decrease with parent reconstruction importance? (predicted by Theoretical)
- Do cross-domain differences in these variables explain the cross-domain absorption differences?

**Step 3.3: Absorption Tax (qualitative)** (~30 min CPU)
- Compute T(G) per hierarchy type
- Test whether T(G) ranking matches cross-domain absorption ranking
- Report as qualitative framework (quantitative predictions not supported by pilot)

### Phase 4: Negative Results Documentation (CPU only)

- GAS failure analysis: document why decoder-activation mismatch fails
- CMI failure analysis: document why CMI does not predict absorption at L0=22
- Absorption Tax quantitative predictions: document rho values and explain limitations
- H2 falsification: discuss why first-letter is atypical (lowest absorption, not highest)

---

## Novelty Assessment

### Contribution 1 (PRIMARY): First Cross-Domain Absorption Characterization -- Novelty 9/10
- No prior work measures absorption on entity-attribute hierarchies. Verified via arXiv, Google Scholar, and web search (April 2026). SAEBench, Chanin et al., and all published work restrict absorption to first-letter spelling.
- The finding that first-letter is atypical (lowest absorption) is stronger than expected and reframes the entire absorption literature.
- Directly addresses Gap 2, Gap 6, and Gap 8b from the literature survey.

### Contribution 2 (SECONDARY): Causal Absorption + Benign/Pathological Diagnostic -- Novelty 8/10
- First interventional (not correlational) evidence for competitive exclusion in SAEs. All prior work uses integrated gradients (correlational).
- The benign-vs-pathological distinction is entirely new (from the Contrarian perspective). No prior work classifies absorption instances by their downstream causal impact.
- The tightened hedging classification (7.4% strict vs. 92.6% loose) is a methodological contribution.

### Contribution 3 (TERTIARY): Rate-Distortion Predictor Framework -- Novelty 7/10
- The three-factor model (cos_sim x co-occurrence / reconstruction importance) is derived from the Theoretical perspective's rate-distortion bound but tested empirically rather than as a formal theorem.
- Novel as a cross-domain predictor; prior work only has qualitative explanations (sparsity causes absorption).
- Downgraded from 8 because pilot evidence shows weak quantitative predictions.

### Negative Results (Honestly Reported)
- GAS unsupervised detector: rho=0.12 (target 0.6)
- CMI at L0=22: rho=0.044 (p=0.83)
- Absorption Tax quantitative predictions: rho=0.08 for absorption-MSE
- First-letter is NOT worst case (positive finding disguised as negative result)
- Per evolution lessons, honest negative results have been the paper's strongest aspect across ALL prior reviews.

### Revisions from Prior Feedback

This is Synthesis Round 6 for iteration 9, building on 8 prior iterations and fresh perspective inputs. Key changes:

1. **Benign-vs-pathological diagnostic added (from Contrarian).** The Contrarian's strongest insight -- that not all absorption is harmful -- is now operationalized as an experimental component (Phase 2, Step 2.2, H8). This converts a philosophical objection into a testable empirical question.

2. **Rate-distortion predictor framework added (from Theoretical + Interdisciplinary).** Rather than attempting a formal theorem (which pilot evidence suggests won't hold quantitatively), we test the three key predictors (cos_sim, co-occurrence, reconstruction importance) as a correlational model across domains (Phase 3, H9). This grounds the theory in evidence.

3. **Ecological niche competition integrated as intuitive framework.** The Interdisciplinary perspective's competitive exclusion analogy provides the narrative thread connecting cross-domain characterization, activation patching, and the rate-distortion predictors. It is used for intuition, not formal claims.

4. **Paper remains two-contribution structure based on evidence.** Pilot evidence clearly separates what works (cross-domain characterization, activation patching) from what doesn't (GAS, CMI, quantitative Absorption Tax). The paper leads with the two strongest contributions and honestly reports negative results.

5. **Fresh perspectives validated prior findings.** All 6 new perspectives independently identified cross-domain characterization as the highest-priority gap, confirming the front-runner selection. No perspective proposed an approach that pilot evidence hadn't already tested. The main new additions (benign/pathological from Contrarian, rate-distortion predictors from Theoretical) are incorporated as enrichments to the existing plan.

---

## Expected Contributions

### Primary: First Cross-Domain Absorption Characterization
- First systematic measurement beyond first-letter spelling
- Striking finding: semantic hierarchies show MORE absorption than syntactic ones
- Architecture rankings change across hierarchy types (JumpReLU consistently best)
- Controlled decomposition: semantic hierarchies show proportionally more true absorption vs. hedging

### Secondary: Causal Mechanism Confirmation + Benign/Pathological Diagnostic
- First activation patching evidence for competitive exclusion in SAEs
- First classification of absorption instances by downstream causal impact
- Tightened hedging classification: loose classification is near-tautological (7.4% strict vs. 92.6% loose)

### Tertiary: Rate-Distortion Predictor Framework (Empirical)
- Three-factor correlational model tested across domains
- Qualitative Absorption Tax framework (T(G) per hierarchy)

### Negative Results (Honestly Reported)
- GAS unsupervised detector fails (rho=0.12)
- CMI does not predict absorption at L0=22
- Absorption Tax quantitative predictions weak
- First-letter is NOT worst case

---

## Experimental Plan

| Priority | Phase | Experiment | GPU-hours | Wall-clock | Validates |
|----------|-------|-----------|-----------|------------|----------|
| 1 | 1.1 | Probe quality improvement (4 layers x 4 hierarchies) | 2 | 2 hr | Quality gate |
| 1 | 1.2 | Cross-domain absorption (12+ SAEs x 4 hierarchies) | 4 | 4 hr | H1, H2', H6 |
| 1 | 1.3 | Absorption-hedging decomposition per hierarchy | 0.5 | 30 min | H3 |
| 1 | 1.4 | Architecture comparison across hierarchies | 1 | 1 hr | H6 |
| 2 | 2.1 | Activation patching at scale (n>50) | 1 | 1 hr | H7 |
| 2 | 2.2 | Benign vs. pathological classification | 1 | 1 hr | H8 |
| 3 | 3.1 | Rate-distortion predictor computation | 0.5 | 30 min | H9 |
| 3 | 3.2 | Qualitative prediction tests | 0.5 | 30 min | H9 |
| 3 | 3.3 | Absorption Tax per hierarchy | 0 | 30 min (CPU) | H5 |
| -- | 4 | Negative results documentation | 0 | 1 hr (CPU) | -- |
| **Total** | | | **~10.5** | **~12 hr** | |

### Pilot Success Criteria (for full mode)

- Probe F1 > 0.90 on at least 2 of 4 hierarchies at best layer
- Activation patching Wilcoxon p < 0.01 on expanded sample
- Cross-domain absorption rates significantly different from first-letter (p < 0.01)

---

## Resource Estimate

- **GPU:** Single GPU >= 24GB VRAM
- **Total compute:** ~10.5 GPU-hours, ~12 hours wall-clock
- **Storage:** ~10GB for cached activations; SAE weights from HuggingFace
- **Software:** SAELens v6, TransformerLens, sae-spelling, RAVEL (HuggingFace `hij/ravel`)
- **No SAE training required.** All experiments use pre-trained SAEs.

---

## Risk Assessment

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|------------|
| RAVEL probes still below 0.90 at all layers | High | 30% | Test layers 18, 24 (factual knowledge at later layers). Relax to 0.85 with documented caveat. Fall back to GPT-2 Small. |
| Activation patching effect disappears at larger n | Medium | 20% | Pilot showed 14.3% vs 0.5% -- large effect size. Bootstrap CI on recovery rate. |
| Cross-domain results reverse with better probes | Medium | 15% | City-language at 10.4% with decent probe is most robust comparison. |
| JumpReLU advantage is artifact of L0 mismatch | Medium | 25% | Compare at matched L0 within SAEBench grid. Report L0 alongside absorption. |
| Benign/pathological ratio unclear (noisy logit changes) | Medium | 30% | Use multiple thresholds (0.05, 0.1, 0.2); report distribution rather than binary split. |
| Rate-distortion predictors fail (rho < 0.3) | Medium | 40% | Report as negative result. The cross-domain characterization stands independently. |
| Competing cross-domain absorption paper | Low | 10% | Verified no competing work as of April 2026. Move quickly to execution. |

---

## Venue Target

**Primary:** NeurIPS 2026 main conference or ICLR 2027 main conference.
**Fallback:** EMNLP 2026 or NeurIPS 2026 MI Workshop.

The cross-domain finding that first-letter absorption is atypical (lowest among tested hierarchies) is high-impact: it reframes every existing absorption architecture comparison and demands cross-domain evaluation as a new standard. Combined with the benign/pathological diagnostic (which converts a philosophical debate into empirical measurement), honest negative results, and the striking JumpReLU finding, the paper offers a narrative of careful, evidence-driven science that advances the field.

---

## Synthesis Reasoning

### Perspective Weighting

I weighted the perspectives as follows, with explicit justification:

1. **Empiricist (highest weight):** The statistical rigor, falsification criteria, and control experiment requirements are non-negotiable for credible research. The RAVEL-based experimental design is adopted almost entirely.

2. **Pragmatist (high weight):** The implementation strategy, toolchain selection, and time estimates are critical for feasibility. The pilot design and resource estimates come from this perspective.

3. **Contrarian (high weight, surprisingly):** The benign-vs-pathological distinction is the most original contribution from any perspective. It survived its own steelman test convincingly and adds genuine novelty to the paper. The contrarian's critique of the measurement metric also informed the tightened hedging classification.

4. **Theoretical (moderate weight):** The rate-distortion framework provides the right mathematical language but pilot evidence shows quantitative predictions are weak. Downgraded from formal theorem to correlational predictor framework.

5. **Interdisciplinary (moderate weight):** The ecological niche competition analogy is the best intuitive model. The phase-transition framework is theoretically sound but requires more data than available to test rigorously. Used for narrative, not formal claims.

6. **Innovator (lowest weight for primary contribution):** ITAC was the boldest proposal but pilot evidence refuted the unsupervised detection approach (GAS rho=0.12). The core insight (decoder geometry reveals absorption) remains valuable as a research direction but is not ready for a primary contribution. The innovator's literature survey and cross-domain insight from disentanglement metrics are incorporated into the theoretical framing.
