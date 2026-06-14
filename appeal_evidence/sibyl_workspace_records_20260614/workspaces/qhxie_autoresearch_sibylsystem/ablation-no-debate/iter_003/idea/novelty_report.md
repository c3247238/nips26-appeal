# Novelty Report: Encoder-Driven Feature Absorption in SAEs

**Date**: 2026-04-30
**Workspace**: ablation-no-debate/current
**Assessor**: sibyl-novelty-checker (sibyl-standard)

---

## Executive Summary

| Candidate | Novelty Score | Recommendation | Key Collision |
|-----------|---------------|----------------|---------------|
| cand_p1 | 7/10 | PROCEED | Partial overlap with Chanin 2024, Tang 2512.05534 |
| cand_safe | 8/10 | PROCEED | No prior work found on safety-critical feature absorption |
| cand_encreg | 7/10 | PROCEED WITH MODIFICATION | OrtSAE (Korznikov 2025) overlaps |
| cand_geom | 4/10 | REVISE OR DROP | Tang 2512.05534 covers theoretical foundation |
| cand_ens | 5/10 | MODIFY TO DIFFERENTIATE | Gadgil et al. 2025 on SAE ensembles |
| cand_eco | 2/10 | DROP | Barlow 1961 is classical, not novel contribution |

**Overall Novelty**: MEDIUM (candidates range from 4/10 to 8/10)

---

## Detailed Analysis by Candidate

---

### cand_p1: Encoder-Driven Feature Absorption: Mechanism and Safety Implications

**Novelty Score: 7/10** (Novel with minor overlap)

#### Core Contribution Claims
1. First factorial decomposition showing absorption is entirely encoder-driven (not decoder)
2. Sensitivity-absorption Pareto frontier quantification
3. Safety-critical feature absorption analysis on real Gemma Scope SAEs

#### Prior Work Assessment

**Chanin et al. (2024) - arXiv:2409.14507 "A is for Absorption"**
- **Overlap**: Documents the absorption phenomenon in SAEs
- **Severity**: partial_overlap
- **Differentiation**: Chanin et al. assume decoder geometry drives absorption. Our H_Mech factorial directly refutes this by showing decoder contributes nothing. This is a genuine theoretical correction, not just confirmation.

**Tang et al. (2025) - arXiv:2512.05534 "Theoretical Foundation of SDL in MI"**
- **Overlap**: Provides theoretical grounding for encoder-driven local minima in SAEs
- **Severity**: related_work
- **Differentiation**: Tang provides theory; we provide experimental factorial validation + safety implications. The combination of mechanism validation + safety testing is novel.

**Korznikov et al. (2026) - arXiv:2602.14111 "Sanity Checks for SAEs"**
- **Overlap**: Baseline comparison methodology
- **Severity**: related_work
- **Differentiation**: Their comparison is across SAEs; ours is within-SAE encoder vs decoder decomposition. Different scope.

**Basu et al. (2026) - arXiv:2603.18353 "Interpretability without Actionability"**
- **Overlap**: Safety-critical interpretability context
- **Severity**: related_work
- **Differentiation**: Basu raises the concern about safety features; we provide the methodology to actually test it. Strong differentiation.

**Hu et al. (2025) - arXiv:2509.23717 "Measuring SAE Feature Sensitivity"**
- **Overlap**: Feature sensitivity measurement methodology
- **Severity**: related_work
- **Differentiation**: We combine sensitivity with absorption analysis; their work is standalone measurement.

#### Collision Summary

| Paper | Overlap | Severity | Differentiation |
|-------|---------|----------|-----------------|
| Chanin 2024 | Absorption phenomenon documented | partial_overlap | We show encoder NOT decoder drives it |
| Tang 2512.05534 | Encoder local minima theory | related_work | We validate experimentally |
| Korznikov 2026 | Sanity checks methodology | related_work | Our encoder/decoder decomposition is new |
| Basu 2603.18353 | Safety interpretability framing | related_work | We test safety features empirically |

#### Differentiation Notes

The core novel contribution is the **2x2 factorial decomposition** showing:
- Condition B (trained encoder, random decoder) = 0.490
- Condition D (trained encoder, trained decoder) = 0.484
- Condition C (random encoder, trained decoder) = 0.299 (same as random)

This is genuinely novel -- no prior work has decomposed encoder vs decoder contributions to absorption this way.

#### Recommendation: PROCEED

The claims are defensible. The H_Mech factorial is genuinely novel as a methodology. Safety analysis (H_Safe) is the highest-novelty component.

---

### cand_safe: Safety-Critical Features Are Disproportionately Absorbed

**Novelty Score: 8/10** (Highly novel, minor related work)

#### Core Contribution Claims
Safety-critical features (deception, jailbreak, harm) show higher absorption rates than matched non-safety features in real Gemma Scope SAEs.

#### Prior Work Assessment

**Basu et al. (2026) - arXiv:2603.18353 "Interpretability without Actionability"**
- **Overlap**: Raises concern that interpretability tools may fail for safety-critical cases
- **Severity**: related_work
- **Differentiation**: Basu argues the problem exists; we test whether absorption specifically is the mechanism. The actual empirical test of absorption in safety features is absent in Basu.

**Bhargav & Zhu (2511.00029)**
- **Overlap**: Mentioned in proposal as related safety work
- **Overlap severity**: related_work (no specific absorption claims found)

#### Collision Summary

| Paper | Overlap | Severity |
|-------|---------|----------|
| Basu 2603.18353 | Safety-critical interpretability concern | related_work |
| Bhargav 2511.00029 | Safety-relevant features | related_work |

**No exact_match or partial_overlap found** for safety-critical feature absorption in SAEs.

#### Key Risk Flag

The synthetic pilot failed (p=0.665) because synthetic features lack semantic content. This is documented honestly in the proposal, which is good scientific practice. The real Gemma Scope test is the valid experiment.

#### Differentiation Notes

**Genuinely novel**: No prior work empirically tests whether safety-critical features are disproportionately absorbed. The combination of:
1. Neuronpedia-annotated safety features
2. Multi-child proportional absorption measurement
3. Mann-Whitney U test comparing safety vs non-safety

is unique to this work.

#### Recommendation: PROCEED (highest priority)

Novelty is high. The negative result from synthetic pilot is properly documented as not reportable. Real SAE test is the valid experiment.

---

### cand_encreg: Encoder Regularization to Reduce Absorption

**Novelty Score: 7/10** (Novel with some overlap)

#### Core Contribution Claims
Encoder-targeted regularization (penalizing parent-child activation correlation) reduces absorption by >30% with <5% reconstruction degradation.

#### Prior Work Assessment

**Korznikov et al. (2025) - OrtSAE**
- **Overlap**: Orthogonal SAE training to reduce absorption
- **Severity**: partial_overlap
- **Differentiation**: OrtSAE modifies both encoder and decoder. Our approach is **encoder-only** modification, which is a cleaner test of the encoder-driven mechanism. If H_Mech is correct, encoder-only intervention should suffice.

**Tang et al. (2512.05534)**
- **Overlap**: Encoder local minima theory
- **Severity**: related_work
- **Differentiation**: Tang provides theory; we provide constructive intervention.

#### Collision Summary

| Paper | Overlap | Severity | Differentiation |
|-------|---------|----------|-----------------|
| OrtSAE (Korznikov 2025) | Orthogonal SAE to reduce absorption | partial_overlap | We target encoder only, not both |
| Tang 2512.05534 | Encoder theory | related_work | We provide constructive intervention |

#### Differentiation Notes

The key differentiation is **encoder-only** vs **encoder+decoder** modification. If absorption is truly encoder-driven (as H_Mech suggests), encoder-only regularization should be sufficient and may preserve decoder reconstruction quality better.

However, the novelty is somewhat reduced because:
1. OrtSAE already demonstrates regularization reduces absorption
2. The specific encoder-only approach is a variation, not a fundamentally new idea

#### Recommendation: PROCEED WITH MODIFICATION

Modify to clearly differentiate from OrtSAE:
- Focus on encoder-only intervention (vs both encoder+decoder in OrtSAE)
- Emphasize the theoretical basis from H_Mech (encoder drives absorption)
- Target <5% reconstruction degradation (conservative threshold)

---

### cand_geom: Encoder Geometry Diagnostic for Training-Free Absorption Prediction

**Novelty Score: 4/10** (Substantial overlap, needs repositioning)

#### Core Contribution Claims
Training-free proxy for absorption using encoder direction containment ratio (AUC > 0.75).

#### Prior Work Assessment

**Tang et al. (2512.05534)**
- **Overlap**: Provides theoretical framework for encoder direction geometry and absorption
- **Severity**: partial_overlap
- **Assessment**: This paper covers the theoretical basis for encoder geometry being related to absorption. Our proposed diagnostic would be an applied version of their theory.

**Blumenthal & Mehta (2023)**
- **Overlap**: Direction geometry in representation learning
- **Severity**: related_work

#### Problem: DEPRIORITIZED in proposal

The proposal states: "DEPRIORITIZED because H_Mech shows decoder geometry is irrelevant to absorption -- geometry-based prediction would need to operate on encoder directions instead."

This is correct. The cand_geom was originally about decoder directions, which H_Mech shows are irrelevant.

#### Collision Summary

| Paper | Overlap | Severity | Assessment |
|-------|---------|----------|------------|
| Tang 2512.05534 | Encoder geometry theory | partial_overlap | Covers theoretical foundation |
| Blumenthal 2023 | Direction geometry | related_work | General framework |

#### Key Issue

The candidate claims AUC > 0.75 but the theoretical foundation (Tang) already covers this ground. The novelty as a "diagnostic tool" is low because Tang provides the theory.

#### Recommendation: REVISE OR DROP

If proceeding, reposition as:
- Practical implementation of Tang's theory
- Training-free validation study
- Focus on empirical validation rather than claiming novelty of the concept

Novelty score 4/10 is justified because Tang's paper is recent and covers the theoretical basis.

---

### cand_ens: Multi-Resolution SAE Ensemble for Hierarchical Feature Recovery

**Novelty Score: 5/10** (Partial overlap, needs repositioning)

#### Core Contribution Claims
Train SAEs with varying L0 targets (16, 64, 256) to collectively recover hierarchical features -- high-sparsity captures coarse parent, low-sparsity captures fine child.

#### Prior Work Assessment

**Gadgil et al. (2025)**
- **Overlap**: SAE ensemble concept
- **Severity**: partial_overlap
- **Differentiation**: The proposal explicitly acknowledges this overlap. The L0 diversity mechanism is the claimed differentiator, but the core ensemble concept overlaps with Gadgil.

**Muchane et al. (2025)**
- **Overlap**: Mentioned as related ensemble work
- **Severity**: related_work

#### Collision Summary

| Paper | Overlap | Severity | Differentiation |
|-------|---------|----------|-----------------|
| Gadgil 2025 | SAE ensemble | partial_overlap | L0 diversity mechanism vs general ensemble |
| Muchane 2025 | Ensemble concept | related_work | General ensemble methodology |

#### Key Issue

The differentiation (L0 diversity) is a mechanism difference, not a conceptual difference. The ensemble idea is covered by Gadgil.

#### Recommendation: MODIFY TO DIFFERENTIATE

If proceeding, must:
1. Emphasize the specific L0 diversity mechanism more clearly
2. Compare explicitly against Gadgil's approach in the methodology
3. Focus on the hierarchical feature recovery aspect (which Gadgil may not address)

Novelty score 5/10 is appropriate given the partial overlap with Gadgil.

---

### cand_eco: Efficient Coding Framing (Replaces Competitive Exclusion)

**Novelty Score: 2/10** (Already done / reframing only)

#### Core Contribution Claims
Absorption is efficient coding (Barlow 1961), not competitive exclusion. H2 failure (positive frequency correlation) motivated this reframing.

#### Prior Work Assessment

**Barlow 1961 "Possible Principles Governing the Relearning of Sensory Data"**
- **Overlap**: Efficient coding hypothesis
- **Severity**: exact_match (classical)
- **Assessment**: This is a 60-year-old foundational paper. Claiming efficient coding as a "novelty" is not defensible.

**Tang et al. (2512.05534)**
- **Overlap**: Mentioned in proposal as supporting efficient coding interpretation
- **Severity**: related_work

#### Key Issue

This is a **reframing** of the mechanism, not a new empirical contribution. The proposal acknowledges:
- "Reinterpretation of mechanism, not new empirical contribution."
- "Novelty: 3/10 - Reframing, not new empirical contribution"

The proposal correctly assesses this as low novelty.

#### Collision Summary

| Paper | Overlap | Severity | Assessment |
|-------|---------|----------|------------|
| Barlow 1961 | Efficient coding hypothesis | exact_match (classical) | 60-year-old work |
| Tang 2512.05534 | Efficient coding interpretation | related_work | Supports but doesn't originate |

#### Recommendation: DROP

Novelty score 2/10 is generous. This is a theoretical reframing grounded in classical work from 1961. It may be valuable as a discussion section or framing device in the paper, but it cannot be a standalone contribution.

If used at all, it should be as a framing device within cand_p1, not as an independent candidate.

---

## Summary and Recommendations

### Novelty Rankings

| Rank | Candidate | Score | Verdict |
|------|-----------|-------|---------|
| 1 | cand_safe | 8/10 | PROCEED (highest priority) |
| 2 | cand_p1 | 7/10 | PROCEED |
| 3 | cand_encreg | 7/10 | PROCEED WITH MODIFICATION |
| 4 | cand_ens | 5/10 | MODIFY TO DIFFERENTIATE |
| 5 | cand_geom | 4/10 | REVISE OR DROP |
| 6 | cand_eco | 2/10 | DROP |

### Key Findings

1. **cand_safe (safety-critical absorption)** is genuinely novel -- no prior work tests whether safety features are disproportionately absorbed. This should be the highest-priority experiment.

2. **cand_p1 (encoder-driven mechanism)** has a genuinely novel methodology (2x2 factorial decomposition) even though it builds on prior work. The decoder-irrelevance finding is novel.

3. **cand_encreg (encoder regularization)** differentiates from OrtSAE by targeting encoder only, but the general approach is covered.

4. **cand_geom and cand_eco** have fundamental novelty problems -- Tang 2512.05534 covers the theoretical ground for geometry, and Barlow 1961 is classical for efficient coding.

5. **No "exact_match" collisions** found -- none of the candidates reproduce prior work exactly. The overlaps are either partial (addressable) or related work (acceptable).

### Recommended Actions

1. **Focus on cand_safe as the highest-novelty experiment** -- Gemma Scope test with Neuronpedia-annotated safety features
2. **Proceed with cand_p1's H_Mech factorial** -- the encoder vs decoder decomposition is methodologically novel
3. **cand_encreg as backup** -- encoder-only regularization differentiated from OrtSAE
4. **Drop cand_eco** -- use efficient coding framing within cand_p1 discussion if desired
5. **Deprioritize cand_geom** -- Tang 2512.05534 covers theoretical basis

---

## References

- Chanin et al. (2024). A is for Absorption. arXiv:2409.14507
- Korznikov et al. (2026). Sanity Checks for SAEs. arXiv:2602.14111
- Tang et al. (2025). Theoretical Foundation of SDL in MI. arXiv:2512.05534
- Basu et al. (2026). Interpretability without Actionability. arXiv:2603.18353
- Hu et al. (2025). Measuring SAE Feature Sensitivity. arXiv:2509.23717
- Barlow, H. (1961). Possible Principles Governing the Relearning of Sensory Data. In Sensory Communication.
- Gadgil et al. (2025). [SAE ensemble work - cited in proposal]
- Muchane et al. (2025). [Ensemble concept - cited in proposal]