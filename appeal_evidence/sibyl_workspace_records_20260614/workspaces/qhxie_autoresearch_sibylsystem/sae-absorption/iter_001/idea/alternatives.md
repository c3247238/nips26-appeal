# Backup Ideas for Pivot (Round 3 — Post-Experimental Update)

## Current Status

The front-runner (cand_eda_crossdomain) remains viable with scope restructuring to 3 primary contributions. Two contributions (D-EDA improvement, ITAC as primary) have been demoted. The blocking dependency is Gemma 2B model access. Backup options are ordered by activation condition and implementation readiness.

---

## Backup A: Amortization Gap Controlled Dictionary Experiment (NEW — Promoted from Contrarian Round 3)

**Status:** High-priority backup AND potential supplement to main paper. Fast to run, clean hypothesis, high novelty.

**Activation condition:** Always run as a supplement; activate as primary if EDA fails across the board and cross-domain results are null.

**Core idea:** Take the SAME pre-trained decoder dictionary D from a Gemma Scope SAE and compare absorption rates under different encoding methods. This definitively separates "absorption is a sparsity landscape property of the dictionary" (Tang et al. view) from "absorption is primarily an amortization gap property of the encoder" (O'Neill et al. view).

**Key experiment (1-2 GPU-hours):**
1. Load Gemma Scope L12-16k decoder weights (pre-trained, frozen).
2. Run three encoding methods on 10,000 tokens from OpenWebText:
   - Standard feedforward encoder (original SAE)
   - Orthogonal Matching Pursuit on D (no learned encoder, same K)
   - 2-pass encoder: standard encoder + one residual correction pass
3. Compute absorption rates using adapted Chanin et al. metric on all three encoders with the SAME dictionary D.
4. Statistical test: paired Wilcoxon signed-rank on per-letter absorption rates.

**Interpretation:**
- If absorption drops > 50% with OMP vs. feedforward at matched L0: amortization gap is the dominant cause. Immediate implication: practitioners should use iterative encoding for deployed SAEs in absorption-sensitive applications.
- If absorption rates are similar across encoding methods: loss landscape (Tang et al.) is the dominant cause. The dictionary itself encodes absorption, and encoder changes alone cannot fix it.
- Either outcome is a clean, publishable result that the field needs.

**Why not already done:** No prior paper has controlled the dictionary while varying encoding method. This is the most direct experimental test of the encoding vs. landscape debate.

**Novelty:** 9/10. Not done before. Directly adjudicates the field's central mechanistic debate.

**Compute cost:** 1-2 GPU-hours. Very fast — no new training needed, only encoding method substitution.

**Connection to main paper:** The 75% early-absorption finding from H4 is consistent with either the amortization gap thesis or the Tang et al. thesis. Backup A resolves this ambiguity and would add a fourth contribution to the main paper at minimal cost.

---

## Backup B: LCA-SAE — Locally Competitive Encoder for Absorption Reduction

**Status:** Strong backup if main paper's primary contributions both fail. Requires more compute than Backup A.

**Activation condition:** EDA AUROC < 0.60 on all configs with direct labels AND cross-domain results null (all hierarchies within 3pp of random baseline with proper probes).

**Core idea:** Replace the feedforward SAE encoder with an unrolled Locally Competitive Algorithm (LCA) where features compete through lateral inhibition derived from the decoder Gram matrix. Iterative competition allows parent and child features to coexist at graded activation levels.

**Architecture:**
```
a^0 = ReLU(W_enc @ x + b_enc)           # standard encoder as initialization
G = max(0, W_dec.T @ W_dec - tau * I)   # sparse lateral inhibition matrix
For t in 1..T:
    a^(t+1) = S_lambda(W_enc @ x - G @ a^t + a^t)  # soft-threshold with lateral inhibition
x_hat = W_dec @ a^T + b_dec             # standard decoder, unchanged
```

**Key differentiator from MP-SAE (Costa et al. 2025):** MP-SAE uses sequential greedy selection; LCA uses parallel lateral competition. LCA's simultaneous competition preserves graded activity for hierarchically related features; MP-SAE's greedy approach cannot prevent a high-activation child from blocking a lower-activation parent in the first iteration.

**Pilot validation (10 min CPU):**
- Chanin et al. 2-feature hierarchical toy model
- Standard feedforward SAE: parent absorbed
- LCA encoder (5 iterations): parent and child coexist
- Pilot passes: proceed with real SAE implementation

**Compute cost:** 3-4 GPU-hours (fine-tuning alpha + SAEBench evaluation on Gemma Scope 16k).

---

## Backup C: Recoverability Analysis — Is Absorption Lossless or Lossy Compression?

**Activation condition:** Main contributions confirmed, add this as a 4th contribution. Also useful if Backup A raises the encoding question.

**Core idea:** If absorbed parent information is recoverable from child feature activations via a linear readout, absorption is informationally lossless (the information exists in the SAE, just in the wrong latent). If not recoverable, absorption represents genuine information loss.

**Key experiment (1-2 GPU-hours):**
1. For each late-absorbed parent feature (the ~13% minority with decoder direction present):
   - Extract child feature activations when parent should fire but is absorbed
   - Train a 1-layer linear probe: parent_concept ~ child_activations
   - Report recovery accuracy vs. non-absorbed case (parent probe accuracy)
2. Compute I(parent; child_activations) / I(parent; parent_activations) as compression efficiency ratio
3. If ratio > 0.9 for majority: absorption is approximately lossless for late-type cases.

**Why restricted to late-absorbed latents:** Early absorption (75% of cases) has no decoder direction — the parent feature was never learned, so there is nothing to recover. Recoverability is only a coherent question for late-absorbed latents.

**Novelty:** 7/10. The recoverability ratio is a new metric. The lossless vs. lossy framing is novel for SAE absorption.

**Practical implication:** If absorption is lossless, the safety concern is "information misaddressing" rather than "information loss" — absorbed information can still be decoded via child features, but 1-sparse probing misses it. This reframes the urgency of absorption mitigation.

---

## Backup D: Successive Refinement Test — Is Increasing Width a Valid Fix for Absorption?

**Activation condition:** After H6 falsification, add as a quick supplementary experiment.

**Core idea (from Interdisciplinary perspective, Equitz-Cover 1991):** If SAE representations are NOT successively refinable, then increasing SAE width will NOT monotonically recover absorbed features — wider SAEs find different, not more, features.

**Key experiment (< 1 GPU-hour):**
- For all Gemma Scope L12 SAEs (widths: 1k, 4k, 16k, 65k, 262k, 1M):
- For each absorbed letter feature in 16k: check if a matching feature exists in 65k (cosine similarity > 0.8)
- Count fraction of absorbed 16k features recovered in 65k, 262k, 1M
- If < 50% of absorbed 16k features are present in any wider SAE: NOT successively refinable

**Why this matters:** The H6 non-result showed wider SAEs have MORE absorption. If they also fail to recover previously absorbed features, then "just use wider SAEs" is not the right prescription. The correct fix is hierarchically-aware training objectives (Matryoshka SAE, KronSAE) or iterative encoding (MP-SAE, LCA-SAE).

**Novelty:** 8/10. The successive refinement framing (Equitz-Cover) has never been applied to SAE absorption. The test is simple but the theoretical framing is novel.

**Compute cost:** < 1 GPU-hour (pure weight analysis, no activation data needed).

---

## Dropped Candidates (Confirmed from Prior Rounds)

- **cand_hierarchy_coherent_loss:** Crowded by Muchane et al. (2506.01197), Luo et al. HSAE (2602.11881), KronSAE (2505.22255). Not activating.
- **cand_fisher_information:** Computationally infeasible for large SAEs. Not activating.
- **Competitive Exclusion Theory (Innovator):** Reframed as interpretive vocabulary in related work, not a standalone contribution.
- **Divisive Normalization SAE (Interdisciplinary):** Requires SAE training. Not activating without pilot validation.
- **KronSAE Extension (old Backup D):** KronSAE already covers the architecture angle. Would need cross-domain angle to differentiate. Low priority.

---

## Pivot Decision Tree (Updated Round 3)

```
CURRENT STATE:
- EDA: 2/6 pass (proxy labels block definitive test)
- Cross-domain: existence confirmed, probe quality blocks definitive validation
- Taxonomy: confirmed (late > early, 75% early-dominance)
- ITAC: falsified (3% vs. 20%)
- H6: falsified (no sign reversal)

Primary action: Obtain Gemma 2B access → re-run with direct labels

├── Gemma 2B obtained (or Llama-3.1-3B fallback)
│   ├── >= 3/6 EDA configs pass AUROC >= 0.65 with direct labels
│   │   ├── Shuffled control confirms cross-domain (proper probes)
│   │   │   → PROCEED: 3-contribution paper (EDA regime-detection, cross-domain, taxonomy)
│   │   │     + consider adding Backup A (amortization gap) as 4th supplementary result
│   │   └── Shuffled control invalidates cross-domain
│   │       → PIVOT: 2-contribution paper (EDA + taxonomy)
│   │         + add Backup A (amortization gap) as 3rd contribution
│   └── < 3/6 configs pass even with direct labels
│       ├── Taxonomy + early-dominance finding is robust and compelling
│       │   → PIVOT: 2-contribution paper (taxonomy + EDA theory)
│       │     + Backup A (amortization gap controlled dict experiment) as 3rd
│       └── EDA group separation also weak (Cohen's d < 0.3 everywhere)
│           → ACTIVATE Backup B (LCA-SAE) as primary. Taxonomy as characterization.
│             Framing: "Characterization and Correction of Feature Absorption"
│
└── Gemma 2B/Llama both blocked
    ├── Taxonomy finding (75% early) is robust with available data
    │   → Proceed with taxonomy as primary + Backup A as secondary
    │     Target: MI Workshop or EMNLP (characterization paper)
    └── EDA theory also strong (SynthSAEBench F1 = 0.974)
        → Proceed with theory (EDA theorem) + taxonomy as primary
          Add Backup D (successive refinement) as quick supplementary result
```

---

## Quick Addition for Any Paper Version: Backup A Recommendation

Regardless of which pivot path is taken, Backup A (amortization gap controlled dictionary experiment) should be run immediately because:
1. It takes 1-2 GPU-hours and requires no new model downloads beyond already-loaded Gemma Scope SAEs.
2. Its result is interpretable and high-impact regardless of direction (either confirms the amortization gap thesis or refutes it, both outcomes are publishable).
3. It directly connects to the 75% early-absorption finding: if OMP on the same dictionary shows dramatically less early absorption, the dictionary-learning process (joint encoder-decoder training) is the driver of early absorption, not the sparsity objective alone.
4. It differentiates this work from all existing absorption papers, which focus on either the loss function or the training procedure, not the encoding method given a fixed dictionary.
