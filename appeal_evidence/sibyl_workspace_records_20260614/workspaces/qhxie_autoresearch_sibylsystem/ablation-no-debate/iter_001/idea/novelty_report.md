# Novelty Report: Feature Absorption in SAEs -- Measurement, Causation, and Safety Consequences

**Workspace**: ablation-no-debate/current
**Date**: 2026-04-27
**Novelty Checker**: sibyl-novelty-checker (iter_002, updated)

---

## Update from iter_002 Search (2026-04-27)

Additional prior work identified through fresh arXiv and Google Scholar searches:

### Safe-SAIL (Weng et al., 2509.18127) - CONFIRMED PARTIAL OVERLAP
- "Safe-SAIL: Towards a Fine-grained Safety Landscape of Large Language Models via Sparse Autoencoder Interpretation Framework"
- Analyzes 1,758 safety-related features across pornography, politics, violence, and terror domains
- **Key Finding**: Does NOT study whether these safety features are disproportionately absorbed
- **Relevance to cand_safe**: Provides methodology for identifying safety features but addresses different question

### Basu et al. (2603.18353) - NEW CRITICAL CITATION
- "Interpretability without Actionability: mechanistic methods cannot correct language model errors despite near-perfect internal representations" (March 2026)
- SAE feature steering "produced zero effect despite 3,695 significant features"
- Shows 53-percentage-point knowledge-action gap
- **Relevance to cand_h3_fix**: Validates that steering interventions may fail - supports honest reporting of H3 as potentially negative result

### Bussmann et al. (2503.17547) - CONFIRMED PARTIAL OVERLAP
- "Learning Multi-Level Features with Matryoshka Sparse Autoencoders"
- Addresses feature absorption via nested dictionaries
- **Differentiation from cand_ens**: Architecture modification vs. post-hoc measurement approach

### Matryoshka SAEs vs Proposal Approach
The proposal's multi-resolution ensemble (cand_ens) is differentiated from Matryoshka SAEs:
- Matryoshka: Architectural solution (nested dictionaries with constraints)
- Proposal: Post-hoc measurement approach (L0 variation + decoder cosine similarity matching)

---

## Executive Summary

This proposal investigates feature absorption in sparse autoencoders (SAEs) through three novel contributions: (1) fixing the measurement methodology (multi-child proportional ablation), (2) causal validation via steering intervention, and (3) safety-critical feature absorption analysis. The proposal is well-anchored on existing prior art (Chanin et al., Korznikov et al.) and makes genuine novel contributions in measurement methodology and safety-critical analysis. The competitive exclusion framing is also genuinely novel.

The most critical new finding from this search is **Tang et al. (arXiv:2512.05534)** ("On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability"), which proves that "spurious local minima correspond to feature absorption" -- providing theoretical grounding for the competitive exclusion and niche geometry hypotheses. The proposal must cite this paper.

**Overall Novelty**: HIGH

---

## Per-Candidate Novelty Analysis

---

### cand_p1: Multi-Child Proportional Ablation + Causal Validation

**Novelty Score: 8/10**

#### Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Chanin et al. (2409.14507), "A is for Absorption" | Defines absorption phenomenon; single-feature ablation metric | exact_match_on_concept, partial_overlap_on_measurement |
| Korznikov et al. (2602.14111), "Sanity Checks for SAEs" | Korznikov-style baselines methodology; showed SAEs recover only 9% ground-truth features | partial_overlap |
| Tang et al. (2512.05534), "Theoretical Foundation of SDL in MI" | Proved spurious local minima = feature absorption; provides theoretical grounding | partial_overlap |
| O'Brien et al. (2411.11296), "Steering Language Model Refusal with SAEs" | SAE steering for safety; capability degradation finding | partial_overlap |
| Borobia et al. (2603.25325), "How Pruning Reshapes Features" | Found rare features survive pruning better (Spearman rho = -1.0); relates to H2 frequency-absorption | partial_overlap |

**Key Differentiation**:
1. **Multi-child proportional ablation (H1)**: Chanin et al. uses single-feature ablation. The proposal's multi-child proportional variant (k=5) with variance-of-contributions formulation is a specific extension to fix pilot saturation. Not in prior work.
2. **Frequency-absorption correlation (H2)**: Borobia et al. found rare features survive pruning better (different context: weight pruning vs. SAE training). The proposal's claim within SAE training is complementary.
3. **Causal steering validation (H3)**: O'Brien et al. studied steering ALL refusal features. The proposal tests absorbed vs. non-absorbed features specifically -- a more targeted causal question.
4. **Safety-critical absorption (H_Safe)**: No prior work examines whether safety-critical features are disproportionately absorbed. Genuinely novel (9/10 sub-score).

**Differentiation Notes**:
- The proposal is differentiated from Chanin et al. by focusing on **measurement fix** (multi-child proportional ablation) rather than documenting the phenomenon
- The proposal is differentiated from Korznikov et al. by extending baselines specifically to **absorption metrics** rather than general interpretability
- The proposal is differentiated from Tang et al. by **empirical measurement methodology** rather than theoretical characterization
- The proposal is differentiated from O'Brien et al. by **causal validation** of absorption as the specific mechanism

**Critical Citation Needed**: Tang et al. (arXiv:2512.05534) must be added to the references. The theoretical result that local minima = absorption provides motivation for the measurement approach and the geometry hypotheses.

**Recommendation**: PROCEED

---

### cand_eco: Competitive Exclusion Framing

**Novelty Score: 8/10**

#### Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Kalmykov & Kalmykov (1211.1869) | Competitive exclusion principle (foundational ecology) | related_work |
| Blumenthal & Mehta (2304.10694) | Geometry of ecological coexistence | related_work |
| Chanin et al. (2409.14507) | Discusses why absorption occurs (sparsity optimization) | partial_overlap |
| Tang et al. (2512.05534) | Proves local minima = absorption (theoretical complement) | partial_overlap |

**Collision Assessment**:
- arXiv search for "competitive exclusion sparse autoencoder" returned NO matches
- No prior work in ML/CS applies competitive exclusion to interpretability
- Chanin et al. discusses sparsity optimization as the cause but does not frame it as competitive exclusion dynamics
- Tang et al.'s result (local minima = absorption) provides theoretical complement but does not claim the ecological framing

**Risk Flag**: The ecological framing is genuinely novel but should be validated empirically (H_Comp: absorption increases monotonically with hierarchy strength). The Lotka-Volterra parallel is motivating theory, not established fact for neural networks.

**Critical Citation Needed**: Tang et al. (arXiv:2512.05534) provides theoretical grounding for why absorption occurs, complementing the ecological framing.

**Recommendation**: PROCEED with caution -- novel framing, must validate empirically

---

### cand_safe: Safety-Critical Features Are Disproportionately Absorbed

**Novelty Score: 9/10**

#### Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Bhargav & Zhu (2511.00029), "Feature-Guided SAE Steering for Refusal-Rate Control" | SAE steering for safety; does NOT address absorption | related_work |
| O'Brien et al. (2411.11296), "Steering Language Model Refusal with SAEs" | SAE steering for safety; does NOT address absorption | related_work |
| GSAE (2512.06655), "Graph-Regularized SAE for Safety Steering" | Safety steering with distributed features | related_work |
| SAILS (2512.23260), "Interpretable Safety Alignment via SAE-Constructed Subspace" | Safety subspace from SAE decoder directions | related_work |
| Goldowsky-Dill et al. (2502.03407), "Detecting Strategic Deception Using Linear Probes" | Probing deception features; does NOT involve SAEs or absorption | related_work |

**Collision Assessment**:
- Extensive search for "safety-critical features absorbed sparse autoencoder" returned NO matches
- Prior safety work (Bhargav & Zhu, O'Brien et al., GSAE, SAILS) focuses on using SAE features FOR safety tasks, NOT on whether safety features are systematically MORE absorbed
- The specific claim that safety-critical features are disproportionately absorbed is **genuinely novel**

**Novelty Score**: 9/10 -- highest novelty sub-hypothesis
**Recommendation**: PROCEED -- genuinely unexplored question

---

### cand_geom: Niche Geometry Diagnostic

**Novelty Score: 7/10**

#### Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Tang et al. (2512.05534), "Theoretical Foundation of SDL in MI" | Proved spurious local minima = absorption; local minima have geometric interpretation | partial_overlap |
| Korznikov et al. (2509.22033), "OrtSAE" | Uses geometry for mitigation (not detection); orthogonal constraints | partial_overlap |
| Blumenthal & Mehta (2304.10694) | Geometry of ecological coexistence | related_work |

**Collision Assessment**:
- Tang et al. explicitly proves spurious local minima correspond to absorption, providing theoretical backing for geometry-based prediction. The containment_ratio in the proposal operationalizes this theoretical insight.
- OrtSAE (Korznikov et al., 2509.22033) uses encoder-decoder geometry for **mitigation** via orthogonal constraints. The proposal uses the same geometry for **detection** -- different goals.
- The specific claim that containment_ratio predicts absorption (AUC > 0.75) is not in prior work.

**Critical Citation Needed**: Tang et al. (arXiv:2512.05534) provides the theoretical connection between geometry and absorption local minima. OrtSAE should be cited to differentiate detection from mitigation.

**Risk Flag**: Tang et al.'s theoretical result means geometry-based prediction is well-motivated. The proposal should cite this paper explicitly.

**Recommendation**: PROCEED with citation to Tang et al. (2512.05534)

---

### cand_ens: Multi-Resolution SAE Ensemble

**Novelty Score: 5/10**

#### Collisions

| Paper | Overlap | Severity |
|-------|---------|----------|
| Gadgil et al. (2505.16077), "Ensembling SAEs" | SAE ensembling; different initializations | partial_overlap |
| Muchane et al. (2506.01197), "Hierarchical SAE" | Explicitly models hierarchical feature structure in SAE architecture | partial_overlap |
| Luo et al. (2602.11881), "From Atoms to Trees" | Hierarchical feature forest with HSAE | partial_overlap |
| Chanin et al. (2508.16560), "Sparse but Wrong" | L0 affects feature correctness | related_work |
| Feature Hedging (2505.11756) | Correlated features break narrow SAEs | related_work |

**Collision Assessment**:
- Gadgil et al. ensembles SAEs with **different random initializations** for diversity. The proposal uses **different L0 targets** (16, 64, 256) as the mechanism. These are distinct ensemble strategies.
- Muchane et al. and Luo et al. explicitly **model hierarchical structure in the SAE architecture** (structural constraint loss, parent-child relationships). The proposal does NOT modify the SAE architecture; it uses L0 variation post-hoc to identify absorbed parent-child pairs.
- The L0 diversity mechanism for hierarchical recovery is differentiated from Gadgil's initialization diversity.

**Risk Flag**: Both Muchane et al. and Luo et al. published recently (Jun 2025, Feb 2026) on hierarchical SAE features. The proposal should clearly position itself as complementary: these papers model hierarchy in the architecture, while the proposal measures absorption in standard architectures.

**Recommendation**: PROCEED with repositioning -- must clearly distinguish L0-variation ensemble from initialization-variation ensemble (Gadgil) and architectural hierarchy (Muchane/Luo)

---

## Cross-Cutting Issues

### 1. Tang et al. (arXiv:2512.05534) -- CRITICAL (New Finding)

This paper provides the most important theoretical grounding for this proposal. It proves that spurious local minima in sparse dictionary learning correspond to feature absorption. This is relevant across multiple hypotheses:

- **H1 (multi-child proportional ablation)**: Provides theoretical motivation for why absorption occurs
- **cand_geom (niche geometry)**: Containment_ratio operationalizes the geometric characterization of local minima
- **cand_eco (competitive exclusion)**: Theoretical complement to ecological framing

**Action Required**: Add Tang et al. (arXiv:2512.05534) to proposal references and citations throughout.

### 2. Borobia et al. (arXiv:2603.25325) -- MODERATE (New Finding)

"Weak features -- those with low firing rates -- survive pruning far better than frequent ones, with Spearman correlations of rho = -1.0" (in the context of weight pruning). This is related to H2 (frequency-absorption correlation) but in a different context.

**Action Required**: Cite for H2 as related evidence; note different mechanism (pruning vs. SAE training).

### 3. Recent Hierarchical SAE Work -- MODERATE (New Finding)

- Muchane et al. (2506.01197): Models hierarchical semantics explicitly in SAE architecture
- Luo et al. (2602.11881): Builds "structured feature forest" with parent-child relationships

These are complementary to cand_ens (architecture vs. post-hoc measurement).

**Action Required**: Clearly position cand_ens as complementary to these architectural approaches.

### 4. Steering Safety Literature -- MODERATE (Expanded Finding)

- O'Brien et al. (2411.11296): Found that steering safety features causes capability degradation on other tasks
- Bhargav & Zhu (2511.00029): Feature-guided steering for refusal control
- GSAE (2512.06655): Graph-regularized SAE for distributed safety features
- SAILS (2512.23260): Interpretable safety subspace from SAE decoder directions

None of these test whether absorbed features are disproportionately represented in safety-critical features.

---

## Final Scores and Recommendations

| Candidate | Novelty Score | Recommendation | Key Differentiator |
|-----------|---------------|----------------|-------------------|
| cand_p1 (multi-child + causal + safety) | 8/10 | PROCEED | Measurement fix + safety-critical analysis |
| cand_eco (competitive exclusion) | 8/10 | PROCEED | Novel ecological framing |
| cand_safe (safety-critical absorption) | 9/10 | PROCEED | Genuinely unexplored question |
| cand_geom (niche geometry) | 7/10 | PROCEED | Detection vs. OrtSAE mitigation |
| cand_ens (multi-resolution ensemble) | 5/10 | PROCEED (reposition) | L0 diversity mechanism |

**Overall Novelty**: HIGH (all candidates score 5+; none require dropping)

**Critical Action**: Add Tang et al. (arXiv:2512.05534) citation before proceeding

---

## Required Additional Citations

| Paper | arXiv ID | Needed For |
|-------|----------|-----------|
| Tang et al. -- Theoretical Foundation of SDL in MI | 2512.05534 | cand_p1, cand_geom, cand_eco |
| Borobia et al. -- How Pruning Reshapes Features | 2603.25325 | cand_p1 (H2) |
| Luo et al. -- From Atoms to Trees (Hierarchical SAE) | 2602.11881 | cand_ens |
| Bhargav & Zhu -- Feature-Guided SAE Steering | 2511.00029 | cand_safe |
| GSAE -- Graph-Regularized SAE for Safety | 2512.06655 | cand_safe |
| SAILS -- Interpretable Safety Alignment | 2512.23260 | cand_safe |

---

## References

1. Chanin et al. (2024). "A is for Absorption: Studying Feature Splitting and Absorption in Sparse Autoencoders." arXiv:2409.14507
2. Korznikov et al. (2026). "Sanity Checks for Sparse Autoencoders: Do SAEs Beat Random Baselines?" arXiv:2602.14111
3. Korznikov et al. (2025). "OrtSAE." arXiv:2509.22033
4. Tang et al. (2025). "On the Theoretical Foundation of Sparse Dictionary Learning in Mechanistic Interpretability." arXiv:2512.05534 **[NEW - CRITICAL]**
5. O'Brien et al. (2024). "Steering Language Model Refusal with Sparse Autoencoders." arXiv:2411.11296
6. Borobia et al. (2026). "How Pruning Reshapes Features: Sparse Autoencoder Analysis of Weight-Pruned Language Models." arXiv:2603.25325 **[NEW]**
7. Gadgil et al. (2025). "Ensembling Sparse Autoencoders." arXiv:2505.16077
8. Muchane et al. (2025). "Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures." arXiv:2506.01197
9. Luo et al. (2026). "From Atoms to Trees: Building a Structured Feature Forest with Hierarchical Sparse Autoencoders." arXiv:2602.11881 **[NEW]**
10. Bhargav & Zhu (2025). "Feature-Guided SAE Steering for Refusal-Rate Control using Contrasting Prompts." arXiv:2511.00029 **[NEW]**
11. Kalmykov & Kalmykov (2012). "A Refinement of the Competitive Exclusion Principle." arXiv:1211.1869
12. Blumenthal & Mehta (2023). "Geometry of Ecological Coexistence." arXiv:2304.10694
