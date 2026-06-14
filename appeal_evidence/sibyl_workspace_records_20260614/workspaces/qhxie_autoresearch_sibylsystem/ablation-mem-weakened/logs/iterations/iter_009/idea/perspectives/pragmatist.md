# Pragmatist Perspective

## Phase 1: Literature Survey
### Key Resources Found

1. **sae-spelling** (lasr-spelling/sae-spelling) — Official implementation of "A is for Absorption" (Chanin et al.). MIT license. **Code exists and reusable.** Contains the differential correlation absorption metric used in this project.

2. **SAEBench** (adamkarkarvonen/SAEBench) — ICML 2025 benchmark with 8 metrics including absorption. MIT license. **Code is mature and pip-installable.** Provides standardized evaluation protocol.

3. **SAELens** (decoderesearch/SAELens) — Training and loading library for SAEs. MIT license. **Production-ready.** Integrates with TransformerLens and provides GemmaScope/LlamaScope loaders.

4. **GemmaScope** (google/gemma-scope) — Comprehensive JumpReLU SAEs for Gemma 2 (2B/9B/27B). Apache-2.0. **Best available pretrained SAEs.** All layers covered.

5. **feature-hedging-paper** (chanind/feature-hedging-paper) — Code for Feature Hedging paper (Chanin et al., 2025). **Reference for Matryoshka implementation.** Limited documentation.

6. **sae_vis** (callummcdougall/sae_vis) — Feature/prompt visualizations. MIT license. **Useful for paper figures.**

7. **SynthSAEBench** (decoderesearch/synth-sae-bench-experiments) — ICML 2026 synthetic benchmark with 16K features. **Reproduces LLM SAE phenomena.**

### Landscape Summary

The SAE feature absorption landscape has matured significantly from 2024-2026:

- **What works**: JumpReLU SAEs (GemmaScope), SAEBench evaluation protocol, SAELens tooling
- **What doesn't**: Architectural fixes (Matryoshka, OrtSAE) are complex and show diminishing returns; they reduce absorption metrics but downstream utility gains are unclear
- **Practical gaps**:
  1. No systematic cross-model absorption quantification exists
  2. The "Sanity Checks" paper (arXiv:2602.14111) raises fundamental questions about whether SAE metrics are meaningful vs. random baselines
  3. Weak interpretability-utility correlation (~0.3) reported by Wang et al. (ICLR 2026) suggests absorption studies need utility validation

**Key pragmatic concern for this project**: The Sanity Checks challenge must be addressed. Any absorption study published today will be asked: "did you compare to random baselines?" This project already did this (H7: trained < random, p<0.001), which is a genuine strength.

---

## Phase 2: Initial Candidates

### Candidate A: Feature Absorption as Optimal Compression (cand_g)
- **Core hypothesis**: Absorption is rate-distortion optimal compression behavior, not a failure mode
- **Implementation sketch**: Use existing H1-H7 data; no new experiments needed
- **Simplest version**: Null-result paper with metric validation insight
- **Time estimate**: 0 GPU hours (all experiments completed)
- **Reusable components**: H7 random baseline comparison is the novel contribution
- **Novelty score**: 7/10 (confirmed by novelty-checker)

### Candidate B: Cross-Model Absorption Quantification Study
- **Core hypothesis**: Absorption rates vary systematically across model families and layers
- **Implementation sketch**: Apply SAEBench absorption metric to Gemma 2, Llama 3.1, Pythia at multiple layers
- **Simplest version**: Measure absorption across 3 models, 5 layers each, report systematic patterns
- **Time estimate**: ~4-6 GPU hours (using pretrained SAEs from SAELens)
- **Reusable components**: SAELens GemmaScope/LlamaScope loaders, SAEBench metric
- **Novelty score**: Unknown; Gap 1 in literature but straightforward execution

### Candidate C: Absorption-Aware SAE Training-Free Diagnostic
- **Core hypothesis**: Post-hoc detection of absorbed features via decoder weight analysis
- **Implementation sketch**: Analyze decoder weight structure to identify likely absorbed features without retraining
- **Simplest version**: Cluster decoder rows; measure cluster coherence vs. absorption rate
- **Time estimate**: ~2-3 GPU hours
- **Reusable components**: SAELens, TransformerLens hook system
- **Novelty score**: Unknown; training-free approaches are underexplored

---

## Phase 3: Self-Critique

### Against Candidate A (cand_g: Optimal Compression)

**Implementation reality check**: The project already executed this. H1-H4 show null results, H7 shows trained < random. **Status: No risk.** The data exists and is consistent.

**Reproducibility attack**: The random baseline comparison (H7) is the key claim. Is it reproducible? Yes—random SAE construction is deterministic (frozen orthonormal decoder, random encoder). The 8x difference (0.034 vs 0.278) is highly significant (p<0.001). **Low risk.**

**Baseline sanity check**: The Sanity Checks paper (arXiv:2602.14111) shows frozen/random SAEs match trained on standard metrics. This project shows trained < random specifically on absorption—**stronger than Sanity Checks**, which looked at general metrics not absorption specifically. **The claim is defensible.**

**Scope attack**: Single model (GPT-2 Small) limits generalization. Gemma 2 results mentioned in full_absorption_gemma.py but not integrated into main H1-H7 analysis. Reviewers will ask: "Does this hold for other models?" **Moderate concern.**

**Verdict: STRONG** with minor scope limitation.

---

### Against Candidate B (Cross-Model Quantification)

**Implementation reality check**: SAELens + GemmaScope + LlamaScope covers 3 model families. Layer coverage is good. **Feasible in ~6 hours GPU time.** Code exists and is reusable.

**Reproducibility attack**: Cross-model study would be highly reproducible. SAELens provides standardized loaders. **Low risk.**

**Baseline sanity check**: Without random baseline comparison, this study would face the same Sanity Checks challenge as other SAE papers. Must include random baseline at least for one model to be credible. **Design must include random baseline.**

**Scope attack**: The study would produce a reference table of absorption rates—useful but incremental. No downstream task validation. **Risk: could be dismissed as "survey paper."**

**Verdict: MODERATE** — useful reference data but incremental without utility validation.

---

### Against Candidate C (Training-Free Diagnostic)

**Implementation reality check**: Post-hoc decoder analysis is novel but risky. H6 (decoder correlation graph) was falsified—decoder geometry does not predict absorption pairs. Why would cluster analysis succeed where graph analysis failed? **Uncertain.**

**Reproducibility attack**: Clustering is stochastic; results may not replicate across random seeds. **Moderate risk.**

**Baseline sanity check**: If decoder clusters show absorption correlation, need to validate against ground truth (which we don't have for real LLMs). **Circular dependency.**

**Scope attack**: This is a new method with uncertain validity. Could take months to develop and validate. **High risk of no publishable result within project timeline.**

**Verdict: WEAK** — interesting but high implementation risk and uncertain payoff.

---

## Phase 4: Refinement

### Dropped Ideas
- **Candidate C**: High implementation risk. Decoder-based approaches already falsified in H6.

### Strengthened Ideas
- **Candidate A**: Already executed with strong results. H7 is a genuine contribution that directly addresses the Sanity Checks challenge. The null results are honest and methodologically rigorous.
- **Candidate B**: Could be a follow-up study, but not within current project scope.

### Selected Front-Runner

**cand_g (Optimal Compression)** is the clear choice:
1. All experiments completed
2. Novelty score 7/10 confirmed
3. H7 directly addresses Sanity Checks challenge (must-acknowledge for any SAE paper today)
4. Zero additional GPU cost
5. Paper writing phase already identified as next step

**Key validation needed before paper writing**:
- Cross-model absorption data (full_absorption_gemma.py) should be integrated to address scope limitation
- Gemma 2B results should be incorporated as secondary validation

---

## Phase 5: Final Proposal

### Title
**"Feature Absorption as Optimal Compression: Evidence that Training Reduces Structural Artifacts"**

### Hypothesis
Feature absorption in SAEs is rate-distortion optimal compression behavior rather than a failure mode. Specifically:
1. Absorption does not significantly degrade downstream steering or probing tasks (null result)
2. Trained SAEs exhibit significantly lower absorption than random baselines, indicating absorption is partially a structural artifact that training reduces

### Motivation
The field has treated absorption as a failure mode requiring architectural mitigation (Matryoshka, OrtSAE). However:
- The Sanity Checks paper (arXiv:2602.14111) challenges whether SAE metrics are meaningful vs. random baselines
- Wang et al. (ICLR 2026) show weak correlation (~0.3) between interpretability and utility
- No prior work has compared trained vs. random SAEs specifically on absorption metrics

This paper provides the first such comparison and reframes absorption as a metric artifact rather than learned pathology.

### Method
**Phase 1**: Differential correlation absorption metric (Chanin et al.) on 26 first-letter features, GPT-2 Small, layers 0/4/8/10

**Phase 2**: Feature steering and sparse probing evaluation; correlation with absorption rate; multiple comparison correction (Bonferroni + BH-FDR)

**Phase 3**: Random SAE baseline comparison (frozen orthonormal decoder, random encoder)

**Phase 4**: Rate-distortion interpretation connecting H5 (precision invariant, recall variable) to optimal compression theory

### Simplest Version
Report H1-H7 findings with full statistical rigor. No new experiments needed.

### Baselines
- **Random steering baseline**: Mean success = 0.344 (L4), 0.379 (L8)
- **Random SAE baseline**: mean = 0.278 (8x higher than trained SAE, p < 0.001)
- **Multiple comparison correction**: 12 tests, Bonferroni alpha = 0.00417, BH-FDR q < 0.05

### Experimental Plan
| Experiment | Status | Result |
|---|---|---|
| Absorption detection | Completed | Mean 2.1-3.9%, max 24.2% |
| Feature steering | Completed | No significant correlation |
| Sparse probing | Completed | No significant correlation |
| EC50 analysis | Completed | No significant correlation |
| Precision-recall | Completed | Precision=1.0, recall varies |
| Decoder graph | Completed | Falsified: precision@20=0.0 |
| Random baseline | Completed | trained < random (p<0.001) |

### Resource Estimate
**All experiments completed.** Remaining:
- Paper writing: ~1-2 days
- Figure generation: ~0.5 day
- Cross-model validation (Gemma 2 integration): ~2-3 hours GPU time (optional but recommended)

### Risk Assessment
| Risk | Severity | Mitigation |
|---|---|---|
| Single model (GPT-2) limits generalization | Moderate | Integrate Gemma 2 data from existing full_absorption_gemma.py |
| Null results may be seen as "nothing found" | Moderate | Strong framing: metric validation + methodological contributions |
| Field skepticism about SAE utility | Ongoing | Focus on utility validation (steering/probing) not just metrics |

### Novelty Claim
First systematic comparison of trained vs. random SAEs on absorption metrics, demonstrating that trained SAEs exhibit significantly lower absorption than random baselines. This reframes absorption from "learned failure" to "structural artifact that training reduces"—a metric validation contribution that directly addresses the Sanity Checks challenge.

### Practical Implementation Path
1. **Week 1**: Integrate Gemma 2 absorption data; generate figures
2. **Week 2**: Complete paper draft
3. **Week 3**: Revisions and submission

**Recommendation**: Proceed with paper writing using cand_g framing. Integrate existing Gemma 2 data to address scope limitation. The H7 random baseline comparison is the key contribution that addresses the Sanity Checks challenge and provides genuine novelty.