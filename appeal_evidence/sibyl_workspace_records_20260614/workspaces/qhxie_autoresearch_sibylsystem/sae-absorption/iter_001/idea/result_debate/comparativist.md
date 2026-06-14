# Comparativist Analysis: Positioning Against SOTA and Related Work (Post-R4 Update)

**Agent**: sibyl-comparativist
**Date**: 2026-04-13
**Workspace**: sae-absorption / iter_001
**Round**: 4 (post-experimental, all blocking experiments complete)

---

## 1. Baseline Landscape: Comparison Table of Existing Methods

### 1a. Absorption Detection Methods

No prior work addresses unsupervised, weight-only absorption detection. The landscape is as follows:

| Method | Year | Venue | Approach | Requires Probes? | Training-Free? | Cross-Model? | Reported Performance |
|--------|------|-------|----------|-------------------|----------------|--------------|---------------------|
| **Chanin et al. supervised metric** | 2024 | NeurIPS 2025 | Integrated-gradients ablation + probe-labeled FN detection | Yes (requires known probe directions + labeled tokens) | Yes (inference only) | Yes (tested on Gemma, Llama, Qwen) | 15-35% absorption rate; ground-truth standard |
| **SAEBench absorption score** | 2025 | arXiv | Absorption fraction from feature-split latents | Yes (uses probe directions) | Yes (inference only) | Yes (200+ SAEs) | Standardized metric; ranks architectures |
| **KronSAE 3-metric suite** | 2025 | arXiv | Mean absorption fraction + full-absorption score + feature splits | Yes | Yes | Limited (KronSAE configs) | Complementary to Chanin metric |
| **This work: EDA** | 2026 | (submitted) | 1 - cos(w_enc, d_dec), pure weight geometry | **No** (weight-only) | Yes | Gemma 2B, GPT-2 Small, Llama-3.1-8B (weight-only) | AUROC 0.650-0.776 in favorable regime (3/8 configs pass) |

**Key gap filled**: EDA is the only weight-only metric that correlates with absorption labels without requiring any probe directions, labeled tokens, or model activations. This gap was confirmed via web search -- no competing unsupervised weight-based detector was found.

**Critical caveat**: EDA is mathematically equivalent to 1 - decoder_cosine_similarity (validated: r = -1.0 across Gemma 2B, GPT-2, and Llama). Decoder cosine similarity already exists as part of SAEBench's evaluation toolkit. The novelty is NOT the formula itself but rather: (a) the formal lower-bound theorem connecting EDA to absorption degree via the Tang et al. (2025) biconvex optimization framework, and (b) the systematic regime characterization showing where it works and where it fails.

### 1b. Absorption Mitigation / Architectural Solutions

| Method | Year | Venue | Approach | Absorption Reduction | SAEBench Ranking |
|--------|------|-------|----------|---------------------|-----------------|
| **Matryoshka SAE** (Bussmann et al.) | 2025 | ICML 2025 | Nested training losses at increasing dictionary widths | Best on SAEBench absorption; positive scaling | #1 on absorption |
| **OrtSAE** (Korznikov et al.) | 2025 | arXiv | Chunked orthogonality penalty on decoder features | 65% absorption reduction; 9% more distinct features | #2 on absorption |
| **KronSAE** (Kurochkin et al.) | 2025 | arXiv | Kronecker factorization of latent space | ~15% reduction via structured hierarchy | Below Matryoshka |
| **Masked Regularization** (Narayanaswamy et al.) | 2026 | arXiv:2604.06495 | Token masking during training disrupts co-occurrence | Reduces absorption across architectures | Not yet on SAEBench |
| **MP-SAE** (Costa et al.) | 2025 | arXiv:2506.03093 | Iterative residual-guided matching pursuit encoder | Better hierarchical feature recovery in theory | WORSE than standard SAEs on SAEBench absorption |
| **Select-and-Project** (Barbulau et al.) | 2025 | arXiv:2509.10809 | Encoder-centric projection framework for steering | Not primarily absorption-focused; encoder-only correction | N/A |
| **AlignSAE** | 2026 | arXiv:2512.02004 | Post-training concept alignment of SAE features | Implicit absorption reduction via concept supervision | Not yet evaluated |
| **Feature Anchoring** (Tang et al.) | 2025 | arXiv:2512.05534 | Identifiability restoration via decoder anchoring | Validated on synthetic only | Not on SAEBench |
| **This work: ITAC** | 2026 | (submitted) | Training-free post-hoc encoder correction for late-absorbed latents | 2.69% mean FN reduction (target was 20%) | N/A (negative result) |

### 1c. Absorption Taxonomy / Characterization

| Method | Year | What it characterizes | Subtypes identified | Sample |
|--------|------|-----------------------|---------------------|--------|
| **Chanin et al.** | 2024 | Absorption rate per letter; single category | None (all absorption treated uniformly) | First-letter spelling, Gemma/Llama/Qwen |
| **Feature Hedging** (Chanin & Dulka) | 2025 | Complementary failure mode (hedging vs absorption) | Hedging vs absorption as opposite poles | Narrow SAEs |
| **This work: Three-subtype taxonomy** | 2026 | Geometric subtype classification via decoder dictionary lookup | Early (~75%), Late (~13%), Diffuse/Partial (~13%) | L12-16k (n=16), L12-65k (n=65) |

**Key gap filled**: No prior work classifies absorbed latents by mechanistic subtype. The taxonomy is novel.

---

## 2. Contribution Margin Analysis

### Contribution 1: EDA as Regime-Specific Absorption Detector

**Our best results**: Gemma L12-16k AUROC = 0.776 [0.700, 0.863]; Gemma L5-16k AUROC = 0.698 [0.637, 0.779]; GPT-2 L6 AUROC = 0.650 [0.531, 0.761] (direct labels).

**Failing configs**: Gemma L5-65k (0.617), L12-65k (0.468), L19-16k (0.458), L19-65k (0.562); GPT-2 L10 (0.344, reversed direction). 5/8 configs fail AUROC >= 0.65.

**Delta vs. SOTA**:
- vs. Chanin et al. supervised metric: EDA is strictly weaker in discriminative power but operates without ANY labeled data. The trade-off (unsupervised but regime-limited) is a genuine contribution.
- vs. Random baseline: Cohen's d = 1.02 at L12-16k, 0.53 at GPT-2 L6. The signal is real but moderate.
- vs. SAEBench decoder cosine: EDA = 1 - dec_cos (mathematically identical). The metric formula adds no novelty. The novelty is the theoretical grounding (Theorem 1: EDA >= delta^2 * sin^2(theta) / (2 + delta^2)) and the systematic regime characterization.

**Margin classification**: **MODERATE**. The AUROC of 0.776 is meaningful, but 5/8 failing configs and the EDA = dec_cos equivalence significantly weaken the contribution. The theoretical lower bound from biconvex optimization is the strongest differentiator, but it requires careful framing to avoid the impression that the paper simply re-derives an existing metric.

**Honest concern**: A reviewer who knows SAEBench will immediately notice that EDA = 1 - decoder_cosine_similarity. The paper must front-load this equivalence and argue that the theoretical derivation + regime characterization provides value beyond the existing ad-hoc metric.

### Contribution 2: Three-Subtype Taxonomy + Early-Dominance Insight

**Our result**: Early absorption dominates at 72-75% (both configs). Late absorption at ~13%. Diffuse/partial at ~13%. KW test significant at L12-65k (p = 0.0002). EDA ordering (late > early) stable across all 5 threshold variants.

**Delta vs. SOTA**:
- vs. Chanin et al.: Single-category treatment. Our three-way split is the first.
- vs. All mitigation literature: The finding that 75% of absorbed latents are "early" (decoder-absent -- the SAE never learned the parent feature) directly challenges the implicit assumption of ITAC/Select-and-Project/MP-SAE that absorption is primarily an encoder alignment issue. This is the paper's most actionable finding.
- vs. Matryoshka SAE: The early-dominance finding provides a theoretical explanation for WHY Matryoshka SAEs excel at absorption -- nested training ensures parent features are allocated dictionary capacity, directly attacking the 75% early-absorption problem. This connection should be made explicit.

**Margin classification**: **MODERATE-TO-STRONG**. The taxonomy itself is a clean novel contribution. The early-dominance insight (75% dictionary-coverage failure) is the most practically impactful finding in the paper. However, the evidence base is thin:
- Only 2 SAE configs (L12-16k: n=16, L12-65k: n=65)
- Threshold sensitivity: at tau=0.2, early proportion drops to ~32% at L12-65k
- The classification criterion (max decoder cosine with proxy parent direction >= 0.3) is somewhat arbitrary

**Honest concern**: The 75% figure is load-bearing for the paper's central narrative. It rests on tau=0.3 at 2 configs. Reviewers will challenge this.

### Contribution 3 (DROPPED): Cross-Domain Generalization

**R4 finding**: H3 FALSIFIED. RAVEL bridge-model probes (GPT-2 Medium to Gemma 2B projection) achieve max 59.5% accuracy (gate: 85%). Shuffled hierarchy control shows real absorption rates statistically indistinguishable from shuffled null (0/9 domain-SAE pairs exceed shuffled p95). Gemma 2B and Llama-3.1-8B models remain HF-gated.

**Delta vs. SOTA**: Cannot claim cross-domain generalization of absorption. The R3 intra-RAVEL rho = 0.924 was based on wrong-model probes and does not survive the shuffled control. This contribution is removed from primary claims.

**Margin classification**: **NOT PUBLISHABLE** as a primary contribution. Retain as a limitation/future-work discussion.

### Contribution 4 (NEGATIVE RESULT): ITAC

**Our result**: 2.69% mean FN reduction on synthetic activations (target: 20%). H5 falsified.

**Delta vs. SOTA**:
- vs. Matryoshka SAE: Matryoshka is #1 on SAEBench absorption with training-time intervention. ITAC's 2.69% is negligible by comparison.
- vs. OrtSAE: 65% absorption reduction at training time. Not competitive.
- vs. MP-SAE: Even MP-SAE, which performs worse than standard SAEs on SAEBench absorption, shows more structural change to absorption patterns than ITAC's 2.69%.

**Margin classification**: **MARGINAL** (<1% effective improvement). ITAC is a negative result. It has value as confirmatory evidence for early-dominance (if 75% of absorption is decoder-absent, any encoder-side correction is inherently limited to ~25% of cases). This reframing is legitimate.

### Contribution 5 (NEGATIVE RESULT): Scaling Analysis

H6 falsified. Partial rho(width|L0) = +0.37 (predicted negative). Limited by L0 variation (65-72 across 6 configs). **Not publishable as a claim.**

---

## 3. Concurrent Work Scan (Updated April 2026)

### Direct competitors (last 6 months, same problem space):

| Paper | Date | Core Contribution | Overlap with This Work | Threat Level |
|-------|------|-------------------|----------------------|--------------|
| **Masked Regularization** (Narayanaswamy et al.) | Apr 2026 | Token masking during training reduces absorption + OOD robustness | Training-time mitigation. Does NOT do detection or taxonomy. | LOW |
| **HSAE** (Luo et al.) | Feb 2026 | Hierarchical SAE with structural parent-child constraints | Learns hierarchy; does not characterize absorption by subtype | LOW |
| **SynthSAEBench** (Feb 2026) | Feb 2026 | Synthetic benchmark with ground-truth hierarchies | Evaluation framework; does not propose detection metric or taxonomy | LOW |
| **Sanity Checks for SAEs** (Korznikov et al.) | Feb 2026 | SAEs recover only 9% of synthetic features; random baselines match | Meta-criticism of SAE utility. Could undermine the paper's motivation. | MEDIUM |
| **Domain-Specific SAEs** (Aug 2025) | Aug 2025 | Domain-constrained training reduces absorption | Mitigation approach, not characterization | LOW |
| **AlignSAE** (Dec 2025) | Dec 2025 | Concept-aligned SAE post-training | Could reduce absorption via supervised alignment; does not characterize | LOW |
| **Select-and-Project** (Barbulau et al.) | Sep 2025 | Encoder-centric steering framework (S&P Top-K) | Targets encoder features for model steering, not absorption detection | LOW |
| **CB-SAE** (Kulkarni, Narayanaswamy et al.) | Dec 2025 | Concept Bottleneck SAEs with pruning + augmentation | SAE feature quality improvement, not absorption-specific | LOW |
| **Feature Sensitivity** (Sep 2025) | Sep 2025 | Measures SAE feature sensitivity to input perturbations | Related evaluation metric; does not address absorption | LOW |

**Assessment**: No concurrent work directly competes with either (a) weight-only absorption detection + regime characterization, or (b) absorption subtype taxonomy. The novelty space is clear. The main reputational risk is the "Sanity Checks" paper (Korznikov et al., 2026): if reviewers adopt the view that SAEs are fundamentally broken, characterizing an SAE failure mode has diminished impact. Counter-argument: the "Use SAEs for Discovery, Not Action" paper (arXiv:2506.23845) argues SAEs retain value for concept discovery, which preserves the motivation for understanding their failure modes.

---

## 4. Novelty Verdict

**The ONE thing this work does that no prior work does:**

"This paper provides the first mechanistic subtype taxonomy of SAE feature absorption, revealing that ~75% of absorbed latents suffer from dictionary coverage failure rather than encoder misalignment -- reframing the dominant absorption mitigation strategy from encoder fixes to dictionary design."

**Assessment**: This is articulable in one sentence. It is genuinely novel. No concurrent work proposes a taxonomy or characterizes the early/late split. However, the supporting evidence is limited (2 SAE configs, threshold-sensitive proportions).

The EDA detection contribution is secondary: it re-derives an existing metric (decoder cosine similarity) with theoretical grounding and regime characterization. Novel in the theoretical connection, but the formula itself is not new.

**Cross-domain contribution is dropped** after R4 shuffled control falsified H3.

---

## 5. Venue Recommendation

| Venue Tier | Assessment | Justification |
|------------|------------|---------------|
| **Top-tier (NeurIPS/ICML/ICLR main)** | **NOT RECOMMENDED** | 4/6 hypotheses falsified (H2 D-EDA, H3, H5, H6). EDA passes on only 3/8 configs. Cross-domain contribution collapsed. Taxonomy rests on 2 configs with n=16 and n=65. The evidence base is too thin for the impact claim. |
| **NeurIPS 2026 MI Workshop** | **RECOMMENDED** (strong fit) | The MI workshop values nuanced, honest characterization work. The taxonomy + early-dominance finding would generate productive discussion. Negative results (ITAC, H6) are explicitly valued. The EDA = dec_cos equivalence finding has practical implications for SAEBench users. |
| **EMNLP 2026 / AISTATS 2027** | **POSSIBLE** with strong revision | If the paper is framed as primarily a characterization + negative-results paper (what we learned about what does NOT work for absorption), the honest reporting of 4/6 falsified hypotheses could be positioned as methodological contribution. Requires substantially expanded config coverage. |
| **Alignment Forum / LessWrong** | **ALSO RECOMMENDED** | The early-dominance finding and EDA regime characterization would generate high engagement in the MI community. The negative ITAC result is informative for practitioners. |

**Comparable published work**:
- Chanin et al. (NeurIPS 2025): Defined and characterized absorption with clean empirical study. Their scope was narrower (first-letter only) but execution was cleaner (100% probe quality, hundreds of SAEs). Our work attempts broader scope but has weaker execution on multiple fronts.
- Feature Hedging (Chanin & Dulka, 2025): Characterized a complementary failure mode with clear theoretical framework + empirical validation. Clean, focused contribution. Our taxonomy contribution has similar character but thinner evidence.
- SAEBench (2025): Comprehensive benchmark paper with 200+ SAEs, 8 metrics. The scale of validation dwarfs our 8-config study.

**Honest assessment**: The paper currently sits at the MI workshop level. The taxonomy + early-dominance finding is the strongest card. The EDA contribution is weakened by the dec_cos equivalence. The cross-domain contribution is gone. ITAC and scaling are negative results. To reach EMNLP/AISTATS, the paper needs: (a) expanded taxonomy validation to >= 5 SAE configs, (b) threshold sensitivity analysis elevated from ablation to primary methodological concern, and (c) extremely honest framing about what worked and what did not.

---

## 6. Strengthening Plan

### Priority 1 (HIGHEST IMPACT): Expand Taxonomy Evidence Base

The 75% early-dominance claim is the paper's crown jewel but rests on 2 configs. Adding taxonomy analysis to:
- GPT-2 L6 (already has direct absorption labels, n_pos = 18)
- GPT-2 L10 (n_pos = 39, though EDA reversed)
- Additional Gemma configs at different layers

...would transform the claim from "interesting observation at L12" to "robust cross-model finding."

**Estimated effort**: Low (reuse existing code, no new data collection needed for GPT-2).

### Priority 2 (HIGH IMPACT): Matryoshka SAE Comparison on Taxonomy

Run the three-subtype classification on Matryoshka SAE latents (if available on SAEBench/Neuronpedia). If Matryoshka SAEs show dramatically lower early-absorption percentage, this:
- Validates the taxonomy's predictive utility
- Explains WHY Matryoshka works (nested training allocates dictionary capacity to parent features)
- Creates a compelling "diagnosis + explanation" narrative

### Priority 3 (MEDIUM IMPACT): Front-Load EDA = dec_cos Equivalence

Rather than hiding this equivalence, make it the opening move:
- "We derive from first principles that EDA = 1 - decoder_cosine_similarity, connecting the Tang et al. (2025) theoretical framework to an existing but atheoretical metric in SAEBench."
- "This theoretical grounding enables us to predict WHERE the metric should work (narrow SAEs, mid-layers) and fail (wide SAEs, extreme class imbalance)."
- This reframing turns a potential weakness (formula is not new) into a strength (we understand WHY it works).

### Additional Baselines That Would Maximally Strengthen Positioning

1. **Random direction baseline for taxonomy**: Classify random (non-absorbed) latents using the same early/late/diffuse criteria. If the 75% early proportion is specific to absorbed latents (and not a generic property of all latents), the taxonomy gains credibility.

2. **Layer-wise taxonomy variation**: Run taxonomy at L5 and L19 (not just L12). If early-dominance is layer-dependent, this is informative for the dictionary-coverage interpretation.

3. **Width-controlled comparison**: Compare early-absorption proportion at 16k vs. 65k at the same layer. If wider SAEs have lower early-absorption percentage (because the wider dictionary captures more parent features), this directly supports the dictionary-coverage narrative.

---

## 7. Risk Flags

### Risk 1: "EDA = dec_cos, so where is the novelty?"

Probability: HIGH if any reviewer checks SAEBench.
Mitigation: Front-load the equivalence. Emphasize theoretical derivation + regime characterization as the contribution, not the formula.

### Risk 2: "75% early dominance on n=16 + n=65 with threshold tau=0.3"

Probability: HIGH for any quantitatively rigorous reviewer.
Mitigation: Report threshold sensitivity prominently. At tau=0.2, early% drops to ~32% at L12-65k. Frame the claim carefully: "At the standard threshold (tau=0.3), early absorption dominates; the proportion is threshold-sensitive and we report full sweep."

### Risk 3: "SAEs are broken anyway" (Sanity Checks paper framing)

Probability: MEDIUM if reviewer is familiar with Korznikov et al. (2026).
Mitigation: Cite the "Use SAEs for Discovery" counter-paper. Frame absorption characterization as important PRECISELY BECAUSE understanding failure modes guides architectural improvement (Matryoshka, OrtSAE).

### Risk 4: Cross-domain collapse weakens the paper significantly

Probability: Already realized. H3 is FALSIFIED.
Mitigation: Do not claim cross-domain generalization. Report the failed experiment honestly. Retain intra-RAVEL coherence (rho = 0.924 from R3) as suggestive future-work evidence, clearly flagged as based on wrong-model probes.

### Risk 5: "Why not just use Matryoshka SAEs?"

Probability: MEDIUM. Matryoshka is #1 on SAEBench absorption and addresses the problem architecturally.
Mitigation: Position this work as complementary diagnostic, not competing mitigation. EDA + taxonomy can evaluate WHETHER Matryoshka succeeds and WHY. The taxonomy provides mechanistic insight that Matryoshka alone does not offer.

---

## 8. Summary Verdict

| Contribution | Novelty | Effect Size | Evidence Quality | Overall Assessment |
|-------------|---------|-------------|-----------------|-------------------|
| EDA regime-specific detector | MODERATE (formula = dec_cos; theorem + regime characterization are new) | MODERATE (AUROC 0.65-0.78 in favorable regime) | MODERATE (3/8 pass; direct + proxy labels; 2 model families) | MODERATE |
| Three-subtype taxonomy + early dominance | HIGH (no prior taxonomy; reframes mitigation literature) | STRONG (75% early reframes the field) | WEAK (2 configs, threshold-sensitive) | MODERATE |
| Cross-domain generalization | DROPPED (H3 falsified by shuffled control) | N/A | N/A | NOT CLAIMABLE |
| ITAC correction | MODERATE (first training-free attempt) | MARGINAL (2.69% vs 20% target) | WEAK (synthetic activations only) | NEGATIVE RESULT |
| Scaling analysis | LOW | MARGINAL (H6 falsified) | WEAK (underpowered) | NEGATIVE RESULT |
| EDA = dec_cos equivalence | MODERATE (connects theory to practice) | N/A (identity, not improvement) | STRONG (r = -1.0 on 3 architectures) | USEFUL FINDING |

**Paper Score (comparativist's honest assessment)**: 4.5/10 for top-tier venue. 6.5/10 for MI workshop. The taxonomy finding is genuinely valuable but under-evidenced. The EDA contribution needs careful positioning to survive the dec_cos equivalence concern. The paper has honest, well-characterized negative results (ITAC, scaling, cross-domain) that have value in the right venue.

**Recommended framing**: "Characterizing SAE Absorption: A Regime-Specific Detector and Three-Subtype Taxonomy" -- a characterization paper that tells the field WHAT absorption looks like geometrically, WHERE weight-only detection works, and WHY most absorbed latents cannot be fixed post-hoc.

---

## Sources

- [Chanin et al. (2024) -- A is for Absorption (NeurIPS 2025)](https://arxiv.org/abs/2409.14507)
- [SAEBench (Karvonen et al., 2025)](https://arxiv.org/abs/2503.09532)
- [Matryoshka SAE (Bussmann et al., 2025, ICML 2025)](https://arxiv.org/abs/2503.17547)
- [OrtSAE (Korznikov et al., 2025)](https://arxiv.org/abs/2509.22033)
- [KronSAE (Kurochkin et al., 2025)](https://arxiv.org/abs/2505.22255)
- [Masked Regularization (Narayanaswamy et al., 2026)](https://arxiv.org/abs/2604.06495)
- [MP-SAE (Costa et al., 2025)](https://arxiv.org/abs/2506.03093)
- [Select-and-Project (Barbulau et al., 2025)](https://arxiv.org/abs/2509.10809)
- [AlignSAE (2025)](https://arxiv.org/abs/2512.02004)
- [Unified SDL Theory + Feature Anchoring (Tang et al., 2025)](https://arxiv.org/abs/2512.05534)
- [Feature Hedging (Chanin & Dulka, 2025)](https://arxiv.org/abs/2505.11756)
- [Sanity Checks for SAEs (Korznikov et al., 2026)](https://arxiv.org/abs/2602.14111)
- [HSAE (Luo et al., 2026)](https://arxiv.org/abs/2602.11881)
- [SynthSAEBench (2026)](https://arxiv.org/abs/2602.14687)
- [Use SAEs for Discovery (2025)](https://arxiv.org/abs/2506.23845)
- [CB-SAE (Kulkarni et al., 2025)](https://arxiv.org/abs/2512.10805)
- [Domain-Specific SAEs (2025)](https://arxiv.org/abs/2508.09363)
- [SAE Survey (Shu et al., 2025)](https://arxiv.org/abs/2503.05613)
- [On the Limits of SAEs (Cui et al., ICLR 2026)](https://arxiv.org/abs/2506.15963)
- [RAVEL (Huang et al., 2024)](https://arxiv.org/abs/2402.17700)
