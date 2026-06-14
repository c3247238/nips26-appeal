# Experimental Methodology (Round 3 — Post-Experiment Update)

## Overview

This document records the experimental methodology for iteration 3 of the SAE absorption project. It updates the pre-experiment methodology with post-experiment findings and synthesizes what must be done before the writing phase.

**Research framing (current state):** The paper is converging toward a two-to-three contribution structure:

1. **encoder_norm as the best-performing training-free absorption detector** — AUROC 0.757 (Standard L1, GPT-2 L6, n_pos=18) and 0.837 (TopK 32k, L6, n_pos=77). Outperforms EDA significantly (DeLong p=0.0012). O_jaccard (co-occurrence Jaccard) provides complementary signal (AUROC 0.721) but multiplicative ARS product does not outperform encoder_norm alone.

2. **Sparsity landscape dominance confirmed** — OMP encoding with frozen decoder shows 0% absorption reduction ratio (mean_AR_feedforward = mean_AR_OMP = 0.978). The amortization gap is NOT the dominant driver. Encoder improvements (iterative encoding, LCA) are insufficient; hierarchically-aware training objectives and dictionary coverage are the necessary lever. Consistent with iter_001 findings (75% early absorption).

3. **TopK architecture shows drastically reduced absorption** — GPT-2 TopK (k=32) 32k SAE shows 0% absorption rate on letters A/E/O/S at 20 tokens per letter (proxy labels). Standard L1 24k SAE shows 97-100% absorption on the same tokens. This is a potentially major finding but needs validation with proper IG labels (proxy labels were used for the TopK result).

**Pending before writing:**
- Validate TopK 0% absorption with proper IG (Chanin et al.) labels — most critical open question
- Generate proper L10 IG labels for cross-layer encoder_norm comparison
- Test encoder_norm + O_jaccard combination with non-multiplicative combination (max, sum)
- Confirm OMP ceiling is not a methodological artifact (design proper per-IG-label OMP experiment)

---

## Candidate

**Primary candidate:** `cand_ars_amortization_crosshierarchy` — encoder_norm detector, sparsity landscape dominance, and TopK architecture comparison.

---

## Environment and Dependencies

```bash
pip install sae-lens transformer-lens datasets scikit-learn scipy statsmodels sae-spelling
```

**Models:**
- GPT-2 Small (gpt2-small, publicly available) — primary model throughout
- Gemma-2-2B (gemma-scope-2b-pt-res) — still HF-gated; SAE weights accessible but no activation caching
- Llama-3.1-8B — still HF-gated; SAE weights accessible

**SAEs used:**
- `gpt2-small-res-jb`, `blocks.6.hook_resid_pre` (L6, Standard/L1, d_sae=24576) — main SAE
- `gpt2-small-res-jb`, `blocks.10.hook_resid_pre` (L10, Standard/L1, d_sae=24576) — cross-layer
- `gpt2-small-resid-post-v5-32k`, `blocks.6.hook_resid_post` (L6, TopK k=32, d_sae=32768) — architecture comparison

---

## Completed Experiments (Iter 003 Round)

### Task A1: encoder_norm Replication
- **Result:** AUROC 0.757 at GPT-2 L6 (n_pos=18, exact IG labels). Replicated iter_002 D1 within rounding.
- **Comparison:** encoder_norm (0.757) > EDA (0.650) > activation_freq_inverted (0.615) > decoder_norm (0.515)
- **DeLong test:** encoder_norm significantly outperforms EDA (z=3.04, p=0.0012)
- **Cross-layer:** L6 AUROC=0.757, L10 AUROC=0.645 (proxy labels — confounded)

### Task A2: Encoder Norm Theory
- **Spearman(encoder_norm, EDA):** r=0.712 — highly correlated but not redundant
- **Early vs. late encoder_norm:** late mean=3.65, early mean=2.95 (late absorbed have HIGHER encoder_norm; Mann-Whitney p=0.055)
- **Layer peak:** encoder_norm ratio (absorbed/non-absorbed) peaks at L6 (1.267), drops at L8 (0.891) and L10 (0.933)
- **Interpretation:** Higher encoder_norm in late-absorbed (decoder-present) latents suggests encoder over-reach correlates with encoder-decoder misalignment, consistent with the sparsity landscape interpretation. Early absorption (decoder-absent) shows lower encoder_norm, suggesting the encoder has not adapted.

### Task A3: Cross-Architecture Comparison
- **Standard L1 (24k):** encoder_norm AUROC=0.757, EDA AUROC=0.650 (n_pos=18, exact IG labels)
- **TopK 32k (k=32):** encoder_norm AUROC=0.837, EDA AUROC=0.644 (n_pos=77, proxy labels)
- **Key finding:** TopK SAE shows 0% absorption rate on letters A/E/O/S (proxy check, 20 tokens each). Standard SAE shows 97-100%.
- **Caveat:** Cross-architecture AUROC comparison is confounded by different label methods and sample sizes.

### Task B1: Co-occurrence Graph
- Directed co-occurrence graph built from 10k tokens, saved as NumPy arrays.
- Key arrays: activation_freq.npy, A_cooccur_all.npy, O_jaccard_letter.npy

### Task B2: ARS_v2 Validation
- encoder_norm alone: AUROC=0.757 (best single detector)
- O_jaccard alone: AUROC=0.721 (strong, independent signal)
- A_cooccur alone: AUROC=0.584 (weak)
- ARS_v2 = encoder_norm × A_cooccur: AUROC=0.586 (worse than encoder_norm alone)
- ARS_original = O × A × cos^2: AUROC=0.528 (below both components)
- ARS_full = encoder_norm × A × O: AUROC=0.528 (same as ARS_original)
- **Conclusion:** Multiplicative product degrades performance. Non-multiplicative combinations not yet tested.

### Task C1/C2: OMP Amortization Gap
- mean_AR_feedforward=0.978, mean_AR_OMP=0.978, reduction_ratio=0.0
- **Interpretation:** sparsity_landscape_dominant — OMP with frozen decoder does not reduce absorption
- **Caveat:** Absorption measured as "do main letter features fire on letter tokens" — may have ceiling effects. Proper IG-based measurement pending.

### Task D1/D2: Cross-Hierarchy
- Entity-type probe best layer = L4; probe quality gate passed.
- Entity-type absorption = 0.0%, negative control = 5.0% at 100 tokens.
- H3 falsified at pilot scale (entity_type < negative_control).
- Spearman ρ = 0.387 (p=0.101) — does not meet >0.40 threshold.
- **Limitation:** 100 tokens may be too few; animal words may not appear often enough in random OpenWebText tokens.

### Task E1: Gemma Scope Fallback to TopK
- Gemma Scope still HF-gated; TopK 32k SAE showed 0% absorption.
- Strong suggestion that TopK architecture eliminates letter feature absorption.

### Task F1: Decoder Recovery (Successive Refinement)
- 67% of 18 absorbed features in 24k L1 SAE recovered (cos>0.8) in 32k TopK SAE.
- Mean best cosine sim: 0.791 across all 18.
- Consistent with early-absorption-as-dictionary-coverage: wider dicts represent more features.

---

## Key Remaining Experiments (Before Writing)

### Phase R5-A (CRITICAL): TopK Zero Absorption Validation with IG Labels
Run sae_spelling FeatureAbsorptionCalculator on TopK 32k SAE. Compare IG-labeled absorption to proxy-labeled result (currently 0%). This validates or refutes the TopK zero-absorption claim with the same methodology used for Standard SAE. If IG confirms <5% absorption on TopK vs ~18 features on Standard L1, this is a primary contribution.

### Phase R5-B (HIGH): Non-Multiplicative encoder_norm + O_jaccard Combination
Test `max(norm_enc_norm, norm_O_jaccard)` and linear combination `a*enc_norm + b*O_jaccard` where a,b optimized on 5-fold CV. Target AUROC > 0.78. Both components have AUROC > 0.70 and partial independence (Spearman < 0.50 expected between them).

### Phase R5-C (MEDIUM): Proper OMP IG Absorption Measurement
For 5 specific absorbed features (identified by IG method), generate test tokens where the standard encoder fires on the letter feature. Measure whether OMP assignment of the same K codes fires on the same feature. This directly tests the amortization gap hypothesis with a measurement not subject to ceiling effects.

### Phase R5-D (SUPPLEMENTARY): Cross-Hierarchy Scale-Up
Scale D2 to 500 tokens; also test animate/inanimate hierarchy (simpler, higher-frequency). If entity-type absorption remains 0% at 500 tokens, report H3 as conclusively falsified for entity-type hierarchies.

---

## Expected Visualizations

- **Table 1 (main detector results):** encoder_norm, O_jaccard, combination, EDA, dec_cosine, activation_freq, random — AUROC, AUPRC, P@50/100/500 at GPT-2 L6.
- **Figure 2 (architecture comparison):** Absorption rates for Standard L1 24k vs. TopK 32k — both IG and proxy labels shown; encoder_norm AUROC per architecture.
- **Figure 3 (encoder norm distributions):** Violin plots of encoder_norm for absorbed vs. non-absorbed latents at L6; stratified by early (n=10) and late (n=8) subtype.
- **Figure 4 (layer progression):** encoder_norm ratio (absorbed/non-absorbed) across layers L2, L4, L6, L8, L10. Peak at L6.
- **Table 2 (OMP results):** Absorption rate matrix: feedforward vs. OMP vs. 2-pass per letter.
- **Appendix (decoder recovery):** Fraction of 24k absorbed features recovered in 32k TopK SAE.
