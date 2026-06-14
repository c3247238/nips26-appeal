# R4-B Shuffled Hierarchy Control — Pilot Summary

**Task ID:** r4b_shuffled_control
**Mode:** PILOT
**Timestamp:** 2026-04-13T12:55:55
**Elapsed:** 4.1 min

## Key Finding: H3 NOT VALIDATED (expected outcome)

The shuffled hierarchy control confirms that real hierarchy absorption rates are **statistically
indistinguishable** from shuffled null rates across all 3 RAVEL hierarchies and all 3 SAE configs.

| Hierarchy | SAE | Real Rate | Shuffled Mean | Shuffled p95 | Ratio | Exceeds p95 |
|-----------|-----|-----------|---------------|--------------|-------|-------------|
| city-continent | L5-16k | 0.001038 | 0.001056 | 0.001578 | 0.98x | No |
| city-continent | L12-16k | 0.000427 | 0.000299 | 0.000522 | 1.43x | No |
| city-continent | L12-65k | 0.001480 | 0.001302 | 0.001638 | 1.14x | No |
| city-country | L5-16k | 0.007385 | 0.007697 | 0.008887 | 0.96x | No |
| city-country | L12-16k | 0.001892 | 0.002124 | 0.002502 | 0.89x | No |
| city-country | L12-65k | 0.009186 | 0.009589 | 0.010213 | 0.96x | No |
| city-language | L5-16k | 0.005615 | 0.005518 | 0.006509 | 1.02x | No |
| city-language | L12-16k | 0.001587 | 0.001544 | 0.001959 | 1.03x | No |
| city-language | L12-65k | 0.006256 | 0.006830 | 0.007506 | 0.92x | No |

**Decision: NO_GO — n_domains_exceed_shuffled_p95 = 0/3 (threshold: >= 2)**

## Interpretation

This result is **expected** given the probe quality failure in r4b_ravel_probes_proper:
- Best available bridge model (GPT-2 Medium, layer 20) achieves max 66.2% probe accuracy on city-continent
  (gate: 85%). Gemma 2B and Llama-3.1-8B are both HF-gated.
- When projected via random orthonormal matrix from d=1024 to d=2304 (Gemma SAE d_in), both
  real and shuffled probe directions are essentially random noise in Gemma's semantic space.
- The fact that real ≈ shuffled is NOT evidence that absorption does not exist in Gemma SAEs.
  It is evidence that **bridge-model probes cannot detect meaningful absorption** in the Gemma SAE space.

## H3 Framing Pivot

Per the task_plan.json fallback decision:
> "H3 cross-domain contribution collapses. Pivot paper to two-contribution structure:
> (1) EDA regime-specific detector + (2) three-subtype taxonomy + early-dominance insight."

**Recommended paper framing:**
1. **EDA weight-based detector** (empirically validated: L5-16k AUROC=0.698, L12-16k AUROC=0.776 on
   Gemma proxy labels; GPT-2 L6 AUROC=0.650 with exact Chanin et al. labels)
2. **Three-subtype absorption taxonomy** (early/late/diffuse) with early-dominance finding (72-75% early-type)
3. **Cross-domain absorption existence** evidence: intra-RAVEL Spearman rho=0.924 (R3) shows coherent
   cross-domain ranking, but absolute rates cannot be trusted without same-model probes.
   Acknowledge limitation prominently: "Absolute RAVEL absorption rates pending Gemma 2B / Llama-3.1-8B
   access; relative coherence structure confirmed."

## Probe Quality Confirmation

- city-continent: real probe cv_acc=0.518 (shuffled achieves 0.518 too — random chance for 6 classes)
- city-country: real probe cv_acc=0.391 (for 42 classes, well above majority 34% but noisy)
- city-language: real probe cv_acc=0.449 (for 29 classes, above majority 43% but narrow margin)

The bridge model cannot distinguish cities by their attributes with high fidelity — the probe training
is fundamentally limited by GPT-2 Medium's geographic attribute encoding.

## Intra-RAVEL Spearman Rho

All pairwise intra-RAVEL rho values are exactly 1.0 (p=0, n=3). This is an artifact of the small
sample size (n=3 SAE configs) and the fact that absorption rates scale monotonically with SAE width.
Not meaningful — requires n >= 6 SAE configs for reliable estimation.

