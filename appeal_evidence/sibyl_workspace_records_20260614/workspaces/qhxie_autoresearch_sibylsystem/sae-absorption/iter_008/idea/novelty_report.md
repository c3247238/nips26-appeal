# Novelty Report (Updated: April 2026, Synthesis Round 5)

**Assessment Date:** 2026-04-15
**Assessor:** Novelty Checker Agent
**Proposal:** "The Absorption Tax: How Feature Hierarchy Structure Governs SAE Failure Modes Across Semantic Domains"
**Candidate Pool:** 4 candidates (1 front-runner, 3 backups)

---

## Executive Summary

The front-runner candidate (`cand_crossdomain_causal_tax`) maintains **high overall novelty** after extensive prior-art search across arXiv, Google Scholar, and the web. The primary contribution -- first systematic cross-domain absorption characterization -- remains the cleanest novelty claim in the proposal with no competing work found as of April 2026. The restructured secondary contributions (causal activation patching, tightened hedging classification) are also novel. Several contributions from the previous proposal version have been correctly downgraded or dropped based on pilot evidence (GAS refuted, CMI not supported, Absorption Tax quantitative predictions weak), which strengthens rather than weakens the paper's credibility.

**Key finding from this updated assessment:** No new competing work on cross-domain absorption characterization has appeared since the last novelty check. The field velocity remains high for architectural *mitigations* (Masked Regularization, arXiv:2604.06495, April 2026), but the *characterization* niche is uncontested.

**Changes from previous novelty report:**
- Candidate ID updated: `cand_crossdomain_gas_tax` -> `cand_crossdomain_causal_tax`
- GAS contribution removed (REFUTED by pilot, rho=0.12)
- CMI contribution removed (NOT SUPPORTED, rho=0.044)
- NEW: Causal absorption via activation patching (Contribution B)
- NEW: Tightened hedging classification (Contribution C)
- Absorption Tax downgraded from 8/10 to 7/10 (quantitative predictions weak)
- Architecture comparison elevated (Contribution E)

---

## Candidate 1: `cand_crossdomain_causal_tax` (Front-Runner)

### Overall Novelty Score: 8/10

---

### Contribution A: First Cross-Domain Absorption Characterization
**Novelty: 9/10 -- Genuinely novel; no close prior work found.**

**Core claim:** Absorption rates measured on entity-attribute knowledge hierarchies (city-country, city-continent, city-language via RAVEL) differ significantly from first-letter spelling rates, with the striking finding that first-letter shows the LOWEST absorption (3.9%) while semantic hierarchies show substantially higher rates (city-language 10.4%, city-continent up to 53.4%).

**Prior art searched:**

| Paper | Overlap | Severity |
|-------|---------|----------|
| Chanin et al., 2024. "A is for Absorption." arXiv:2409.14507, NeurIPS 2025 | Defines absorption metric and measures it. **Only on first-letter spelling task.** Does not extend to entity-attribute hierarchies. | related_work |
| Karvonen et al., 2025. "SAEBench." arXiv:2503.09532, ICML 2025 | Includes absorption as one of 8 benchmark metrics. **Only first-letter task.** | related_work |
| Chaudhary & Geiger, 2024. "Evaluating Open-Source SAEs on Disentangling Factual Knowledge." arXiv:2409.04478 | Uses RAVEL for SAE evaluation on city-country/continent. **Measures disentanglement (causal intervention success), NOT absorption (parent feature false negative rate).** Different metric, different question. | partial_overlap |
| SynthSAEBench, 2026. arXiv:2602.14687 | Evaluates SAEs on synthetic hierarchical features including absorption. **Synthetic data, not real LLM knowledge hierarchies.** | related_work |
| Muchane et al., 2025. "Incorporating Hierarchical Semantics in SAE Architectures." arXiv:2506.01197 | Models semantic hierarchy in SAE architecture. **Focused on architecture, not absorption measurement.** | related_work |
| Luo et al., 2026. "HSAE." arXiv:2602.11881 | Jointly learns SAE series and parent-child relationships. **Architecture proposal, not absorption characterization.** | related_work |
| Narayanaswamy et al., 2026. "Masked Regularization." arXiv:2604.06495 | Proposes masking-based regularization to disrupt co-occurrence patterns. **Mitigation method, not characterization.** | related_work |
| "Resurrecting the Salmon." 2025. arXiv:2508.09363 | Notes SAEBench first-letter metric is insufficient for domain-specific evaluation. **Identifies the gap but does not fill it.** | related_work |

**Recommendation: PROCEED.** This is the cleanest novelty claim in the proposal. RAVEL has been used for disentanglement evaluation but never for absorption measurement. The absorption metric (Chanin et al.) has only been applied to first-letter spelling. No competing cross-domain absorption characterization exists as of April 2026. The pilot finding that first-letter is atypical (LOWEST absorption) adds unexpected value.

**Differentiation notes:** The key distinction from Chaudhary & Geiger (2024) is the metric: they measure *disentanglement* (can you surgically change country without changing continent?), while this proposal measures *absorption* (does the parent feature fail to fire when the child is active?). These are fundamentally different questions about SAE failure modes.

---

### Contribution B: Causal Absorption Confirmation via Activation Patching (NEW in Round 5)
**Novelty: 8/10 -- Novel; first interventional evidence for competitive exclusion.**

**Core claim:** Zeroing child SAE latents recovers parent probe predictions (14.3% recovery vs. 0.5% control), providing the first metric-independent causal evidence for absorption as competitive exclusion.

**Prior art searched:**

| Paper | Overlap | Severity |
|-------|---------|----------|
| Chanin et al., 2024. "A is for Absorption." arXiv:2409.14507 | Uses integrated gradients for attribution (correlational). Also uses ablation to measure causal effect of latents on downstream logit difference. **Does NOT zero child to measure parent recovery.** Their ablation measures the downstream effect of the absorbing latent, not whether removing the child *restores* the parent. | partial_overlap |
| Marks et al., 2024. "Sparse Feature Circuits." arXiv:2403.19647 | Uses activation patching on SAE features for circuit discovery. **General circuit tracing, not absorption-specific parent-child recovery.** | related_work |
| Dunefsky et al., 2024. "Transcoders Find Interpretable LLM Feature Circuits." NeurIPS 2024 | Uses causal patching between SAE features. **Transcoder context, not absorption-specific.** | related_work |
| VisualScratchpad, 2026. arXiv:2603.07335 | Notes hierarchical/correlated SAE latents complicate ablation -- concept can reappear through interconnected latent. **Identifies the issue but does not use it to study absorption mechanism.** | related_work |
| Concept-SAE, 2025. arXiv:2509.22015 | Notes standard SAE paradigm is "fundamentally correlational." Proposes active causal probing. **Vision models, not LLM absorption.** | related_work |

**Recommendation: PROCEED.** The specific experimental design -- zero the child feature latent, measure whether the parent feature's probe prediction recovers -- is novel. Prior work uses ablation to measure *effects of* absorbing latents, not to test whether removing the child *reverses* absorption. This is a qualitatively different (interventional vs. correlational) form of evidence.

**Differentiation notes:** Chanin et al. use integrated gradients (a gradient-based attribution method -- correlational). This proposal uses activation patching (an interventional method). The distinction is analogous to the observational vs. experimental distinction in causal inference -- patching provides stronger evidence for the competitive exclusion mechanism. The pilot result (14.3% vs. 0.5%) is promising but needs n>50 for statistical significance.

---

### Contribution C: Tightened Hedging Classification (NEW in Round 5)
**Novelty: 8/10 -- Novel methodological contribution.**

**Core claim:** The widely cited 98.6% hedging classification from Chanin et al. (2025) is near-tautological. Strict classification yields only 7.4%, with 85.3% being "compensatory resolution" -- a new category that the loose metric conflates with hedging.

**Prior art searched:**

| Paper | Overlap | Severity |
|-------|---------|----------|
| Chanin et al., 2025. "Feature Hedging." arXiv:2505.11756 | Defines hedging and provides classification criteria. **Does not critique or tighten the classification methodology.** | partial_overlap |
| Chanin & Garriga-Alonso, 2025. "Sparse but Wrong." arXiv:2508.16560 | Shows L0 confound in SAE features. Related to hedging vs. absorption distinction. **Does not analyze hedging classification threshold.** | related_work |
| Tian et al., 2025. "Feature Sensitivity." arXiv:2509.23717 | Frames absorption as special case of poor sensitivity. **Complementary but does not address hedging classification.** | related_work |

**Recommendation: PROCEED.** No prior work critiques the loose/strict hedging classification distinction. The 98.6% figure is cited without qualification in the SAE literature. Demonstrating that this figure is near-tautological (and that strict classification yields 7.4%) is a meaningful methodological contribution independent of the cross-domain work.

---

### Contribution D: Absorption Tax Theorem (Qualitative Framework)
**Novelty: 7/10 -- Downgraded from 8/10 due to weak quantitative predictions.**

**Core claim:** T(G) = sum of p_c * R_pc gives the minimum L0 overhead for absorption-free representation, connecting to rate-distortion theory.

**Prior art searched:**

| Paper | Overlap | Severity |
|-------|---------|----------|
| Chanin et al., 2024. "A is for Absorption." arXiv:2409.14507 | Notes informally that absorption "saves +1 L0 per parent-child pair." Qualitative observation only. | partial_overlap |
| Tang et al., 2025/2026. "Unified Theory of SDL." arXiv:2512.05534 | Characterizes absorption as spurious partial minima. Does NOT quantify cost of elimination or derive impossibility bounds. | partial_overlap |
| Ayonrinde et al., 2024. "MDL-SAEs." arXiv:2410.11179 | Rate-distortion framework for SAEs. Does not analyze absorption-elimination cost specifically. | related_work |
| Tilde Research. "Rate Distortion Dance of SAEs." Blog post. | Rate-distortion analysis of SAE sparsity. General framework, not absorption-specific. | related_work |
| Cui et al., 2025. "On the Limits of SAEs." arXiv:2506.15963 | Shows SAEs fail to recover features unless extremely sparse. Related impossibility result but different formulation. | related_work |

**Recommendation: PROCEED with caveat.** The formal impossibility result is novel but pilot evidence shows quantitative predictions are weak (absorption-MSE rho=0.08, R_pc rho=0.16). Report as qualitative framework, not as a predictive theory. The Wyner-Ziv connection is interesting but under-validated.

---

### Contribution E: Architecture Rankings Across Hierarchy Types
**Novelty: 7/10 -- Moderately novel.**

**Core claim:** Architecture rankings for absorption resistance (JumpReLU vs. BatchTopK vs. Matryoshka) change across hierarchy types. JumpReLU is consistently lowest, challenging Matryoshka's assumed superiority from single-task (first-letter) benchmarks.

**Prior art searched:**

| Paper | Overlap | Severity |
|-------|---------|----------|
| Karvonen et al., 2025. "SAEBench." arXiv:2503.09532 | Compares architectures on absorption. **Only first-letter task.** | partial_overlap |
| Bussmann et al., 2025. "Matryoshka SAEs." arXiv:2503.17547, ICML 2025 | Shows Matryoshka beats others on absorption. **Single task.** | partial_overlap |
| SynthSAEBench, 2026. arXiv:2602.14687 | Compares architectures on synthetic data. Not real knowledge hierarchies. | related_work |

**Recommendation: PROCEED.** The cross-task architecture comparison is a direct consequence of the cross-domain characterization. No prior work compares absorption architecture rankings across multiple real knowledge hierarchy types.

---

### Negative Results (Honestly Reported)
**Novelty: N/A (negative results do not require novelty claims)**

These negative results strengthen the paper's scientific integrity:
- **GAS unsupervised detector: REFUTED** (rho=0.12). Decoder-activation mismatch does not capture absorption.
- **CMI at L0=22: NOT SUPPORTED** (rho=0.044, p=0.83). Information-theoretic pillar does not hold.
- **Absorption Tax quantitative predictions: WEAK** (absorption-MSE rho=0.08, R_pc rho=0.16).
- **H2 falsified: First-letter is NOT worst case.** (This is actually a positive finding -- the inversion is more interesting than the original hypothesis.)

---

## Candidate 2: `cand_controlled_dictionary` (Backup)

### Novelty Score: 9/10

**Core claim:** Hold decoder dictionary constant from a pre-trained SAE, vary only the encoder (feedforward vs. OMP vs. 2-pass). Tests whether absorption is encoder failure or dictionary failure.

| Paper | Overlap | Severity |
|-------|---------|----------|
| Costa et al., 2025. "Evaluating SAEs: From Shallow Design to Matching Pursuit." arXiv:2506.05239 | Uses matching pursuit as alternative encoder. MP-SAE has no explicit encoder. **Jointly trains encoder and dictionary, does not hold dictionary constant.** | partial_overlap |
| "Stop Probing, Start Coding." 2026. arXiv:2603.28744 | Compares FISTA, LISTA, and Matching Pursuit inference strategies. **Different inference algorithms but not a controlled dictionary experiment.** | partial_overlap |
| SynthSAEBench, 2026. arXiv:2602.14687 | "No explicit encoder [for MP-SAE]; instead, at each iteration the latent with the highest projection onto the residual is selected." **Does not control the dictionary variable.** | related_work |

**Recommendation: PROCEED.** The controlled experiment design (hold pre-trained dictionary constant, vary only the encoding algorithm) has not been done. This is a clean causal experiment about the locus of absorption. Strengthened by pilot activation patching results confirming encoder-level competitive exclusion.

---

## Candidate 3: `cand_ecological_theory` (Backup)

### Novelty Score: 9/10

**Core claim:** Formalize absorption through Lotka-Volterra competitive exclusion. Predict phase transition at critical cosine similarity theta*.

| Paper | Overlap | Severity |
|-------|---------|----------|
| No direct prior work found. | No paper applies competitive exclusion or Lotka-Volterra dynamics to SAE feature absorption. | N/A |
| Tonolo et al., 2025. "Generalized Lotka-Volterra with Sparse Interactions." arXiv:2503.20887 | Studies LV models with sparse interactions and phase transitions. **Ecology paper, no SAE connection.** | related_work |

**Recommendation: PROCEED.** Genuinely novel ecological framing. The analogy is mathematically productive: both systems involve competition for shared resources under sparsity constraints leading to exclusion. Pilot data supports the framework (different hierarchies -> different absorption rates = different niche overlap distributions).

**Risk:** May be more evocative than productive if phase transition prediction does not quantitatively match data.

---

## Candidate 4: `cand_absorption_aware_correction` (Backup)

### Novelty Score: 8/10

**Core claim:** Given known absorption patterns, construct corrected feature activations by re-activating absorbed parent features when child fires. Tests whether absorption can be handled post-hoc at inference time.

| Paper | Overlap | Severity |
|-------|---------|----------|
| Wright & Sharkey, 2024. Per-latent scaling correction. | Proposes per-latent scaling to correct L1 shrinkage bias. **Corrects magnitude, not absorption-specific false negatives.** | related_work |
| Rajamanoharan et al., 2024. Gated SAE. arXiv:2404.16014 | Decouples detection from magnitude estimation. **Training-time architecture, not post-hoc correction.** | related_work |
| Kulkarni et al., 2026. CB-SAE. arXiv:2512.10805 | Prunes low-utility neurons and augments with concept bottleneck. **Different approach, not absorption-specific.** | related_work |

**Recommendation: PROCEED as supplementary.** No prior work attempts post-hoc absorption correction at inference time.

---

## Dropped Candidates (Confirmed Correct)

| Candidate | Status | Reason |
|-----------|--------|--------|
| `cand_gas_primary` | REFUTED | GAS rho=0.12, well below 0.3 threshold. Correctly dropped. |
| `cand_cmi_taxonomy` | REFUTED | CMI rho=0.044, p=0.83. Correctly dropped. |
| `cand_absorption_tax_quantitative` | DOWNGRADED | Quantitative predictions weak. Correctly downgraded to qualitative. |
| `cand_eda_universal` | MERGED | Merged into negative results. |
| `cand_itac_primary` | MERGED | ITAC achieves 3% vs 20% target. Merged into negative results. |
| `cand_hierarchy_coherent_loss` | DROPPED | Crowded by Muchane et al., Luo et al. HSAE, KronSAE. |
| `cand_scaling_laws` | LOW_PRIORITY | Novelty 5/10. Qualitative trends known from SAEBench. |

---

## Field Velocity Assessment

**High velocity for SAE architectures, low velocity for absorption characterization.**

Recent papers (within last 6 months) relevant to this proposal:

| Paper | Date | Relevance to Proposal |
|-------|------|-----------------------|
| Masked Regularization (arXiv:2604.06495) | April 2026 | Mitigation method, not characterization. No overlap. |
| SynthSAEBench (arXiv:2602.14687) | Feb 2026 | Synthetic absorption evaluation. Different domain. |
| HSAE (arXiv:2602.11881) | Feb 2026 | Hierarchical SAE architecture. Not absorption measurement. |
| Stop Probing, Start Coding (arXiv:2603.28744) | Mar 2026 | Critiques SAEs/probes for compositionality. Tangential. |
| Stable and Steerable SAEs (arXiv:2603.04198) | Mar 2026 | Weight regularization. Tangential. |
| Beyond Semantics: SAEs for CLIP (arXiv:2604.05724) | Apr 2026 | CLIP domain. Tangential. |

**Conclusion:** The field is actively producing architectural mitigations and alternative evaluation methods, but NO group has published cross-domain absorption characterization. The proposal's primary niche remains uncontested. Execution speed matters: if another group performs cross-domain absorption measurement using RAVEL before submission, the primary contribution loses its edge. However, the specific combination (absorption measurement + activation patching + hedging tightening + architecture comparison across hierarchy types) would remain novel even in that scenario.

---

## Required Citations

1. Chanin et al., 2024. "A is for Absorption." arXiv:2409.14507 (NeurIPS 2025)
2. Chanin et al., 2025. "Feature Hedging." arXiv:2505.11756
3. Chanin & Garriga-Alonso, 2025. "Sparse but Wrong." arXiv:2508.16560
4. Karvonen et al., 2025. "SAEBench." arXiv:2503.09532 (ICML 2025)
5. Bussmann et al., 2025. "Matryoshka SAEs." arXiv:2503.17547 (ICML 2025)
6. Chaudhary & Geiger, 2024. "Evaluating Open-Source SAEs." arXiv:2409.04478
7. Tang et al., 2025/2026. "Unified SDL Theory." arXiv:2512.05534
8. Korznikov et al., 2025. "OrtSAE." arXiv:2509.22033
9. SynthSAEBench, 2026. arXiv:2602.14687
10. Narayanaswamy et al., 2026. "Masked Regularization." arXiv:2604.06495
11. Costa et al., 2025. "MP-SAE." arXiv:2506.05239
12. Muchane et al., 2025. "Hierarchical Semantics SAE." arXiv:2506.01197
13. Luo et al., 2026. "HSAE." arXiv:2602.11881
14. Marks et al., 2024. "Sparse Feature Circuits." arXiv:2403.19647
15. Cui et al., 2025. "On the Limits of SAEs." arXiv:2506.15963
16. Ayonrinde et al., 2024. "MDL-SAEs." arXiv:2410.11179
17. Huang et al., 2024. "RAVEL." (dataset paper)
18. Lieberum et al., 2024. "Gemma Scope." arXiv:2408.05147
