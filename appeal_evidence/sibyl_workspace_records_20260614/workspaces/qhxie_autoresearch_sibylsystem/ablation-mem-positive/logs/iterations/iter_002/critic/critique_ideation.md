# Ideation Critique

## Overall Assessment

The research has evolved through three iterations to center on a quasi-critical phase transition framework with SAE feature absorption. The idea progressed from initial ablation metric limitations to cross-architecture validation, then to phase transition phenomenology with finite-size scaling. The CV reversal discovery (absorbed features have 733x higher CV) and its connection to steering actionability represent genuine intellectual progress. The system correctly identified and validated genuine absorption via activation patching (67.3% mean recovery). However, the HARKing problem in H2 framing persists despite prior feedback, and several hypotheses remain inadequately validated.

**Score: 5/10**

---

## Critical Issues

### 1. H2 Framing Still Contains HARKing (Unresolved)

**Location**: `writing/integrated_paper.md`, Section 4.2

The text says: "layer 8 achieves rho = 0.705 > 0.6, satisfying the threshold." This frames a post-hoc observation as meeting a pre-registered threshold. The original H2 threshold (rho > 0.6) applied to the mean across layers, not a single cherry-picked layer. With 3 layers tested, the probability of one reaching p < 0.05 by chance is approximately 14.3%. This was flagged in prior critique but persists.

**Fix**: Replace with: "Layer 8 shows rho = 0.705, but with only 3 layers tested, probability of one reaching p < 0.05 by chance is approximately 14.3%. This observation requires validation on additional layers."

### 2. Chi_ratio Below Threshold Undermines Phase Transition Framing

**Location**: `idea/proposal.md` vs `writing/integrated_paper.md`

The proposal correctly uses "quasi-critical behavior" (chi_ratio=1.88) but the integrated paper still uses "phase transition" language that implies sharpness. The chi_ratio=1.88 is below the "sharp transition" threshold of 3.0. The susceptibility peak (χ=11.19) confirms critical-like phenomenon, but the gradual transition should be emphasized consistently.

**Fix**: Use "quasi-critical threshold behavior" or "critical-like phenomenon" throughout the paper. Explicitly acknowledge chi_ratio < 3.0 as a limitation of the phase transition framing.

### 3. Lambda_c Instability Not Prospectively Validated

**Location**: `idea/proposal.md`

The proposal acknowledges "10x instability" in lambda_c but does not execute prospective validation. The value λ_c=5e-5 was derived from specific experiments and could be an artifact of the SAE configuration. Without prospective validation, the critical sparsity value is questionable.

**Fix**: Test lambda_c=5e-5 on held-out data (different layers, different SAE configs). Report stability bounds on the critical sparsity value.

---

## Major Issues

### 4. Full_cross_layer_critical Results Not Integrated

**Location**: `exp/experiment_state.json` shows task completed, but `writing/integrated_paper.md` does not reflect results

The proposal mentions measuring absorption at λ_c=5e-5 across layers (0, 3, 6, 9, 11), but the paper only covers layers 5, 8, 10 on GPT-2 and 5, 10, 15 on Gemma. If the full experiment was completed, its results are not visible in the current paper. This is a significant omission if the results show negative findings.

**Fix**: Either integrate full_cross_layer_critical results or document them as a negative result. If uniform saturation was found at λ_c, this contradicts the layer-dependence hypothesis and should be prominently reported.

### 5. CV-Steering Connection Is Correlational, Not Causal

**Location**: `exp/results/pilot_summary.json` (pilot_steering_cv)

High-CV features show 2x larger steering effect (0.153 vs 0.075), but this is pilot-stage (n=30 per group) with no mechanism established. Why high-CV features are more steerable is unexplained. The connection to the actionability paradox (Basu et al., 2026) remains theoretical.

**Fix**: Acknowledge explicitly that the CV-steering relationship is correlational. Run full validation with 1000 samples. Establish the mechanism: is it because high-CV features have more steering headroom? Because they're less redundant? Because of different decoder structures?

### 6. Contribution Claims Need Restructuring

**Location**: `writing/integrated_paper.md`, Abstract and Section 1.3

The five contributions listed in Section 1.3 are a mix of validated findings (Contribution 1, 2) and exploratory observations (Contribution 3, 4, 5). Contribution 3 ("Unexpected layer-dependent A_j correlation pattern") is presented as a contribution but explicitly requires validation. Contribution 4 ("decoder norm constraints persist") overgeneralizes from two architectures to "training dynamics rather than architectural design."

**Fix**: Restructure into "Validated Contributions" and "Exploratory Findings." Make clear which are confirmatory vs hypothesis-generating.

---

## Minor Issues

### 7. Finite-Size Scaling Exponent ν=3 Needs Replication

The claim of ν=3 with R²=0.951 is potentially the paper's most novel contribution, but it was derived from only 3 dictionary sizes (6144, 12288, 24576). Without independent replication, this could be an artifact.

### 8. No Negative Controls

No random latent selection, no nonsense category probes, no shuffled labels. This limits confidence that high projection absorption is specific to genuine semantic features.

---

## What Works Well

1. **Honest reporting of negative results**: H3 falsified (all layers saturate at λ=0.001), H6 falsified (graph topology), H2 falsified (mean rho = -0.194, not > 0.6). The paper correctly reports these as failed hypotheses.

2. **Activation patching validation**: 67.3% mean recovery confirms genuine absorption for persistent core words, not metric artifact. This is the paper's most rigorous validation.

3. **CV-based actionability prediction**: The discovery that high-CV absorbed features remain steerable (2x effect) is genuinely novel and addresses the actionability paradox directly.

4. **Adaptive framing**: The system correctly pivoted from "sharp phase transition" to "quasi-critical behavior" after discovering chi_ratio < 3.0. This shows intellectual honesty.

5. **Backup candidates well documented**: The alternatives.md provides clear decision logic for when to pivot to backup candidates.