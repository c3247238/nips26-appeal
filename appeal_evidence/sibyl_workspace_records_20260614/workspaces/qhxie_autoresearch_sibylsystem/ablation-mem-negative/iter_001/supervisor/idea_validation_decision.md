# Idea Validation Decision

## Pilot Evidence Summary

### Candidate A: UAD + DFDA (Front-runner)
- **UAD F1**: 0.704 (threshold: >= 0.6) -- EXCEEDS threshold by 17%
- **UAD Recall**: 1.0 (perfect) -- no false negatives
- **UAD Precision**: 0.543
- **DFDA Mean Improvement**: 11.14% (threshold: > 10%) -- EXCEEDS threshold
- **DFDA Positive Pairs**: 3/4 (75%)
- **DFDA Parameters**: 388 total (< 0.01% of SAE params)
- **Pilot Runtime**: 27.8 seconds (well under 15-minute budget)
- **Collision Rate**: 30.77% on TopK SAE (8/26 letters share features)

### Candidate B: Co-Occurrence as General SAE Analysis Tool
- No direct pilot evidence
- Co-occurrence clustering works on first-letter features (from UAD pilot)
- Generalization to semantic hierarchies entirely untested

### Candidate C: Absorption as Nonidentifiability Artifact
- No pilot evidence
- Would require training multiple SAEs with different seeds
- High risk: if absorption IS stable, entire direction falsified

### Candidate D: Absorption as SAE Quality Diagnostic
- No pilot evidence
- Would require training SAEs with varying hyperparameters
- Dead feature ratios in trained SAEs were extremely high (94-99%)

## Decision Matrix

### Candidate A: UAD + DFDA

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 5 | UAD F1=0.704 exceeds 0.6 threshold; DFDA 11.14% improvement exceeds 10%; perfect recall |
| Hypothesis survival | 0.25 | 5 | H1 strongly supported (F1=0.704 > 0.6); H3 supported (11.14% > 10%); gates passed |
| Path to full result | 0.20 | 4 | Clear pipeline: cross-model validation, multi-seed, DFDA scaling, end-to-end |
| Novelty | 0.15 | 5 | First unsupervised absorption detection; first training-free compensation; no prior work |
| Resource efficiency | 0.10 | 4 | Pilot in 27.8s; full validation ~80 min; DFDA uses 388 params per pair |
| **Weighted Score** | **1.0** | **4.7** | |

### Candidate B: Co-Occurrence General Tool

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 2 | No direct pilot; clustering works on first-letter but semantic hierarchies untested |
| Hypothesis survival | 0.25 | 2 | All hypotheses untested |
| Path to full result | 0.20 | 2 | No clear pipeline; would require significant new WordNet experiments |
| Novelty | 0.15 | 3 | Extends "Geometry of Concepts" but less distinctive than UAD |
| Resource efficiency | 0.10 | 3 | Substantial new compute required |
| **Weighted Score** | **1.0** | **2.25** | |

### Candidate C: Nonidentifiability Artifact

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot evidence; seed stability untested |
| Hypothesis survival | 0.25 | 1 | All hypotheses untested; high falsification risk |
| Path to full result | 0.20 | 2 | Requires 3-5 SAE training runs; high risk of negative result |
| Novelty | 0.15 | 4 | High novelty if supported (challenges field's framing) |
| Resource efficiency | 0.10 | 3 | Multiple SAE training runs required |
| **Weighted Score** | **1.0** | **1.85** | |

### Candidate D: SAE Quality Diagnostic

| Criterion | Weight | Score (1-5) | Evidence |
|-----------|--------|-------------|----------|
| Pilot signal strength | 0.30 | 1 | No pilot evidence; correlations untested |
| Hypothesis survival | 0.25 | 1 | All correlations untested; dead feature ratios were 94-99% |
| Path to full result | 0.20 | 2 | Requires multiple SAE configs; confounding risks high |
| Novelty | 0.15 | 3 | Moderate novelty (reframing absorption as signal) |
| Resource efficiency | 0.10 | 3 | Multiple SAE training configurations required |
| **Weighted Score** | **1.0** | **1.7** | |

## Decision Rationale

**ADVANCE Candidate A (UAD + DFDA)** with high confidence (0.88).

The evidence is decisive:

1. **H1 (UAD feasibility)**: F1=0.704 exceeds the 0.6 threshold by a substantial margin. Perfect recall (1.0) means no absorbed pairs are missed. Precision=0.543 is moderate but acceptable for an unsupervised method -- the false positives can be filtered in downstream steps.

2. **H3 (DFDA efficacy)**: 11.14% mean MSE improvement exceeds the 10% threshold. 3/4 pairs show positive improvement. The method uses only 388 parameters total, confirming the <0.01% parameter budget claim.

3. **Novelty is genuine and verified**: Comprehensive prior art search found no unsupervised absorption detection method and no training-free compensation method. UAD and DFDA each fill a clear gap in the SAE literature.

4. **Clear path to publishable result**: The validation pipeline is well-defined -- cross-model test on Gemma-2B and Pythia-2.8B, multi-seed replication (3 seeds), DFDA scaling to 8+ pairs, and end-to-end pipeline validation. Each step has explicit success criteria.

5. **Go/No-Go gates are sensible**: G1 (H1: F1 >= 0.6) is already passed. G2 (cross-model F1 >= 0.55) is the critical remaining gate. If G2 fails (F1 < 0.5), the pivot criteria in candidates.json specify fallback to cand_b, cand_c, or cand_d.

**Why not REFINE?** The pilot did not expose specific methodological problems that require refinement. The pre-trained SAE loading issue is a technical fix, not a conceptual problem. The precision of 0.543 is acceptable for an unsupervised method.

**Why not PIVOT?** Only Candidate A has empirical support, and its support is strong. Pivoting would mean abandoning the only direction with positive evidence. The backup candidates (B, C, D) are escape routes if cross-model validation fails, not alternatives at this stage.

## Next Actions

1. **Execute Phase 1 (Cross-model UAD validation)**:
   - Run UAD on Gemma-2B layer 10 SAE (16K features, 10K tokens)
   - Run UAD on Pythia-2.8B
   - Run UAD on GPT-2 Small with 3 seeds (42, 123, 456)
   - Implement Chanin et al. true absorption protocol for ground-truth labels
   - Target: F1 >= 0.55 on each model, mean F1 >= 0.6 across seeds

2. **Execute Phase 2 (DFDA scaling + end-to-end)**:
   - Scale DFDA to >= 8 absorbed pairs on 2 models
   - Measure probe accuracy improvement post-compensation
   - Run ablation: UAD without clustering; DFDA without residual

3. **Critical technical fix**:
   - Fix pre-trained SAE loading compatibility with SAELens v6.39.0
   - Alternative: use SAEBench probe-projection as backup metric

4. **Go/No-Go Gate G2 check**:
   - If UAD F1 >= 0.55 on >= 2 larger models: proceed to full paper
   - If UAD F1 0.5-0.55: proceed with caveat, add cand_b as secondary
   - If UAD F1 < 0.5 on any model: PIVOT to backup candidate

SELECTED_CANDIDATE: cand_a
CONFIDENCE: 0.88
DECISION: ADVANCE
