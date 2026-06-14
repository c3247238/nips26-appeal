# Ideation Critique (Updated by Critic — Iter 003)

## Critical Issues

### 1. Core Mechanistic Claim Rests on a Weak Inference

The paper's primary novelty claim is that the OMP oracle experiment "decisively falsifies the amortization gap hypothesis." But the logical structure of the experiment conflates two distinct claims:

(a) **Weak amortization gap hypothesis:** A better inference-time encoder with the same trained decoder will reduce absorption. [Tested by OMP with fixed decoder]

(b) **Strong amortization gap hypothesis:** Jointly training encoder and decoder with an iterative encoder architecture reduces absorption. [Tested by MP-SAE, Costa et al.]

The OMP experiment tests (a) and finds null result. But O'Neill et al.'s original amortization gap argument is about (b) — the claim that the feedforward encoder is a compressed approximation that could be improved by jointly training the system with an iterative encoder. Fixing the pre-trained SAE decoder and applying OMP is testing whether an inference-time fix can overcome a training-time problem — it is not surprising that it cannot.

The paper correctly notes this limitation in Section 5.2, but then proceeds to claim practical guidance ("Do not expect OMP to help") that is correct only for the weak version. The implication that "no inference-time encoder change will help" is overstated. If MP-SAE (which retrains both encoder and decoder with iterative pursuit) reduces absorption, the amortization gap hypothesis remains viable in its strong form.

### 2. Encoder Norm Mechanism is Theoretically Inconsistent

The sparsity landscape account (Tang et al.) describes the gradient competition mechanism as follows: when child latent c fires on tokens where parent j should also fire, the L1 sparsity penalty pushes j's encoder direction away from its decoder direction, creating a stable partial minimum. This mechanism predicts that j's encoder direction drifts — but whether the encoder *norm* increases or decreases is not directly predicted by this account.

The paper argues (Section 3.1): "when child c fires while parent j should also fire, the sparsity penalty creates a gradient that suppresses z_j. A larger encoder weight increases the dot product... partially counteracting the suppressive gradient from c." This predicts norm increases — but this is the amortization gap account (encoder tries to compensate), not the sparsity landscape account. The paper mixes two mechanistic accounts in its explanation of encoder norm.

Under the sparsity landscape account, the encoder is trapped in a training-time minimum regardless of its norm. There is no reason to expect norms to increase specifically; the gradient competition could produce norms that increase or decrease depending on the architecture, initialization, and learning rate.

### 3. The Research Question Was Not Actually Tested

The iter_003 proposal identified these primary questions:
- H_ENC: Encoder weight norm predicts absorbed latents with AUROC > 0.70 across multiple GPT-2 layers and Gemma Scope. [Partially tested — L6 only, inverts at other layers]
- H_OMP: OMP reduces absorption by ≥50% if amortization gap dominates. [Tested but at insufficient scale]
- H_CROSSHIER: Entity-type hierarchy shows absorption > negative control with proper probes. [Failed — probe artifact]
- H_ARS (original): ARS co-occurrence achieves AUROC > 0.70. [Failed — AUROC = 0.527]

The primary novel contributions from the proposal (ARS, amortization gap controlled experiment, cross-hierarchy with proper negative controls) all either failed outright or were tested at pilot scale. The paper pivoted to encoder norm (a surprising empirical finding from iter_002) as the primary contribution, which is scientifically legitimate, but the paper still presents a narrative suggesting all three contributions were originally planned.

## Major Issues

### 4. The "Surprising" Encoder Norm Finding Was Not Pre-Registered

The encoder norm finding came from iter_002 as an unexpected result (described as "unexpected and unexplained" in the methodology). The iter_003 study was designed around this finding. This is a form of HARKing (Hypothesizing After Results are Known) — the hypothesis was generated post-hoc from the data it is being tested on. The validation on the same label set (n_pos=18 from the same iter_001 R4 labels) does not constitute independent validation; it is merely a replication of the finding within the same dataset.

To make the encoder norm finding credible as a novel discovery rather than overfitting to the iter_001 labels, it needs to be validated on independently labeled absorption data, preferably on a different model.

### 5. ARS/Co-occurrence Approach Was Abandoned Without Adequate Testing

The original proposal's primary novel contribution (ARS: Jaccard overlap × activation asymmetry × cosine similarity) failed, with AUROC = 0.527 (essentially random). The paper's explanation — "product formulations dilute both signals" — is offered after the fact. A more thorough investigation would:

1. Test the Jaccard component against a proper activation-frequency-matched null baseline
2. Investigate why 9 of 18 absorbed features have O_jaccard = 0 (they are not in the letter feature set)
3. Test whether the absorption asymmetry signal (A_cooccur) could work with more tokens (10k may be too few for sparse features)

The co-occurrence approach is theoretically well-motivated — absorbed features should appear whenever their absorbing child appears. The failure may be a data-scale issue rather than a conceptual failure. Abandoning it entirely without testing at 200k tokens (as specified in the proposal) is premature.

### 6. Safety Attribution (H5) Was Never Tested

H5 — that absorption explains ≥20% of the SAE probe vs. dense linear probe performance gap on harmful intent detection — was proposed as a high-impact downstream application but was never run. This was the proposal's strongest motivation for why absorption matters for safety. Without it, the paper's safety implications rest entirely on a literature reference to DeepMind's deprioritization of SAE research.

---

## Additional Critique (Fresh Analysis)

### 7. O_jaccard Coverage Is Structurally Limited — 9 of 18 Absorbed Features Have Score=0 by Construction

The B2 notes explicitly state: "O_jaccard is only non-zero for features in the 71 letter feature set. Absorbed features NOT in letter set (9 of 18) will have O_jaccard=0." This means O_jaccard cannot detect half the absorbed features regardless of how many tokens are used or how the threshold is tuned. The AUROC=0.721 is inflated relative to a general-purpose detector because those 9 features with O_jaccard=0 are still counted as hard negatives that O_jaccard correctly "avoids" (by giving them low scores). The effective AUROC for features O_jaccard can potentially detect (those in the letter set) may be substantially different.

The paper should compute AUROC separately for: (a) the 9 absorbed features with non-zero letter-set membership, and (b) the 9 with zero membership. This would clarify whether O_jaccard is a usable detector or whether it merely inherits information from the letter feature set structure.

### 8. The Spearman r=0.712 Between EncNorm and EDA Is a Red Flag, Not a Feature

The iter_003 paper presents EncNorm-EDA Spearman r=0.712 as evidence that they "capture related but distinct geometric properties." But with r=0.712, these two metrics share 51% of variance. The DeLong test improvement (0.757 vs 0.650) may be entirely attributable to EncNorm's distribution shape being more favorable for classification at extreme imbalance (18/24576), not to it capturing independent information.

The paper should run the partial correlation / logistic regression analysis (EncNorm predicting absorption after controlling for EDA) before claiming EncNorm as a "distinct" contribution. Without this, EncNorm may simply be EDA with a distribution transformation that happens to be better calibrated for AUROC computation at high class imbalance.

### 9. The Iter_001 Labels Are Used Across Three Iterations — Risk of In-Sample Optimization

The same n_pos=18 label set (iter_001, R4, FeatureAbsorptionCalculator at GPT-2 L6) has been used to evaluate EDA in iter_001, EDA+ARS in iter_002, and EncNorm+ARS+OMP in iter_003. Any metric that happened to correlate with these 18 specific absorbed features would appear validated. The "discovery" of EncNorm in iter_002 was made on these same 18 labels. The "validation" in iter_003 is on the same 18 labels. This is circular: EncNorm was discovered as surprising because it worked on these 18 labels, and is now being validated on the same 18 labels. This is not independent validation; it is replication within the same dataset.

True validation requires: (a) new absorption labels on a held-out set of letters, (b) replication on a different model (Gemma-2-2B), or (c) both. Without this, the AUROC=0.757 result is vulnerable to the criticism that it reflects properties of this specific 18-feature subset, not a general absorption detection signal.
