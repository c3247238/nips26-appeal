# Glossary of Key Terms

Unified terminology for all paper sections. Use the preferred phrasing listed here; do not introduce synonyms mid-paper.

---

## Core SAE Concepts

**Sparse Autoencoder (SAE)**
A two-layer autoencoder trained to reconstruct language model residual stream activations via a sparse bottleneck. Preferred: "Sparse Autoencoder" on first use, then "SAE". Do not use "dictionary network" or "overcomplete dictionary" without referencing SAE.

**SAE latent** (preferred over "SAE feature", "SAE neuron", "SAE unit")
A single index j in the SAE dictionary, corresponding to encoder row W_enc_j and decoder column d_j. Preferred: "latent" or "SAE latent". In contexts requiring emphasis on the geometric direction, use "decoder direction d_j". Avoid "feature" alone (ambiguous with model features).

**Residual stream**
The transformer internal representation that SAEs are trained on. Preferred: "residual stream activation" or "residual stream". Do not use "hidden state" (ambiguous with LSTM terminology).

**L0 sparsity** (preferred over "number of active features", "sparsity level")
The count of non-zero entries in the SAE activation vector z. `L0(z) = #{j : z_j > 0}`. Report as an integer count, not a fraction.

**Dictionary atom** (preferred when discussing the geometric/decoder role)
A single column d_j of the decoder matrix W_dec. Used when emphasizing the geometric role in reconstruction. Synonym with "decoder direction d_j". Use "SAE latent" for the index; "dictionary atom" or "decoder direction" for the geometric vector.

---

## Absorption Terminology

**Feature absorption** (preferred over "suppression", "occlusion", "hiding")
The systematic failure of a parent SAE latent to activate on tokens where a child latent is present and absorbs the parent's activation budget. Formally: on tokens where the parent concept is present (as measured by an IG probe), the parent latent z_i fails to fire while a semantically narrower child latent z_j activates instead.

**Absorbed latent** (preferred over "absorbed feature", "suppressed feature", "missing feature")
A SAE latent j that systematically fails to fire on tokens where its associated concept is present, due to a more specific latent i absorbing the activation. In notation: j ∈ 𝒜.

**Absorbing latent** (preferred over "child feature", "specific feature")
The latent i that fires instead of the absorbed latent j on target tokens. Typically has higher activation frequency (f_i >> f_j in the hierarchical sense; note: for first-letter tasks, the letter feature is child and fires more often while parent spelling feature is absorbed — see "parent/child" clarification below).

**Parent feature / parent concept**
The broader, more general concept in a hierarchy (e.g., "starts with letter A" is the parent). In first-letter spelling task: the parent SAE latent is the one that should encode "word starts with A"; it is the absorbed latent (j ∈ 𝒜).

**Child feature / child concept**
The narrower, more specific concept (e.g., the full spelling of a specific word). In first-letter spelling task: child-letter features are the absorbing latents (i) that fire instead of the parent.

**CAUTION — frequency direction**: In the first-letter task, the "absorbing" (child) latent has HIGHER activation frequency than the absorbed (parent) latent. This is the opposite of the naive frequency expectation. Always verify the direction when discussing absorption in semantic hierarchies.

**Absorption rate (AR)**
The fraction of tokens (among those where the probe confirms the parent concept is present) where the absorbed latent j fails to activate. AR ∈ [0, 1]; higher AR means more absorption. Report as a decimal fraction (e.g., AR=0.978), not as a percentage, unless the context is explicitly comparative.

**Early absorption** (preferred over "Type I absorption", "decoder-absent absorption")
Absorbed latents where the decoder direction d_j has low cosine similarity with any other letter latent's decoder direction (max cos_sim < 0.3). Interpreted as: the dictionary has no representation for the parent concept at all. Approximately 10/18 absorbed latents in iter_002/003 GPT-2 L6 data.

**Late absorption** (preferred over "Type II absorption", "decoder-convergent absorption")
Absorbed latents where the decoder direction d_j is highly similar to another (non-absorbed) letter latent's decoder direction (max cos_sim ≥ 0.3). Interpreted as: the dictionary has a direction for the concept, but the encoder fails to use it. Approximately 8/18 absorbed latents in iter_002/003 data.

---

## Mechanistic Explanations

**Amortization gap**
The provable gap between the quality of sparse codes produced by feedforward encoders and those produced by optimal sparse inference algorithms (e.g., OMP, ISTA). Formalized by O'Neill et al. (2411.13117). The amortization gap hypothesis claims this gap is the primary cause of feature absorption.

**Sparsity landscape (partial minimum)**
The theoretical account (Tang et al., 2512.05534) that absorption arises because the combined encoder+decoder optimization of SAE training has multiple stable local minima, some of which correspond to absorption configurations. The partial minimum theory claims absorption is embedded in the dictionary geometry at training time, not in encoder inference quality.

**OMP oracle**
Orthogonal Matching Pursuit applied with a fixed pre-trained decoder dictionary at matched L0 (K = mean feedforward L0). Used as an idealized encoder that has zero amortization gap by construction (it finds the globally optimal K-sparse code for the given dictionary). If OMP does not reduce absorption, the amortization gap hypothesis is falsified.

---

## Detectors and Scores

**Encoder-Decoder Angle (EDA)**
`EDA(j) = 1 - cos(W_enc_j / ‖W_enc_j‖, d_j)`. Measures divergence between encoder direction and decoder direction for the same latent. High EDA = encoder points away from decoder; prior work finds AUROC=0.650 on GPT-2 L6.

**Encoder weight norm** (preferred over "encoder norm", "weight norm")
`enc_norm(j) = ‖W_enc_j‖₂`. The L2 norm of the encoder weight vector for latent j. The primary detection signal in this paper. Refer to as "encoder weight norm" on first use per section, then "encoder norm" is acceptable for brevity.

**Jaccard co-occurrence (O_jaccard)**
The Jaccard overlap between the activation sets of two latents i and j. `O_jaccard(j, i) = |𝒮_j ∩ 𝒮_i| / |𝒮_j ∪ 𝒮_i|`. Per-latent score is the maximum over candidate parent latents. Secondary detection signal in this paper; AUROC=0.721, independent of encoder norm (Spearman ρ=0.044).

**Absorption Risk Score (ARS)**
`ARS(j) = O_jaccard(j) × A_cooccur(j) × cos²(d_i, d_j)`. Original ARS from proposal; AUROC=0.528 (near-chance). Do not present as a contribution; report as a negative result. Refer to by its full name on first mention, then "ARS (original formulation)".

**ARS_v2**
`ARS_v2(j) = enc_norm(j) × A_cooccur(j)`. Revised score tested in B2; AUROC=0.586 — significantly worse than enc_norm alone (DeLong z=-2.455, p=0.993). Do not present as a contribution.

---

## Statistical Terms

**DeLong test**
A non-parametric test for comparing two AUROC values from the same set of predictions. Produces a z-statistic and one- or two-sided p-value. Used throughout to compare enc_norm vs. EDA and ARS variants vs. enc_norm.

**Bootstrap confidence interval (CI)**
95% CI for AUROC values computed by bootstrap resampling over positive and negative examples. Report as [lower, upper] with 2 decimal places.

**Cohen's d**
Effect size for the difference in mean enc_norm between absorbed and non-absorbed features. `d = (μ_absorbed - μ_all) / σ_all`. Values: 0.971 (Standard/L1), 1.235 (TopK-32k).

**Precision@k**
Among the top-k latents ranked by a detector score, the fraction that are truly absorbed. Report at k=50, 100, 500. Note: at 18/24,576 positive prevalence, P@50=0.0 is expected even for AUROC=0.757.

---

## Spelling and Style Conventions

| Preferred | Avoid |
|---|---|
| "SAE latent" | "feature" (alone), "neuron", "unit" |
| "encoder weight norm" / "encoder norm" | "weight norm" (ambiguous), "norm" |
| "feature absorption" | "feature suppression", "feature hiding", "feature occlusion" |
| "Orthogonal Matching Pursuit" (OMP) | "pursuit algorithm", "OMP oracle" on first use |
| "sparsity landscape" | "loss landscape", "optimization landscape" |
| "amortization gap" | "encoder gap", "approximation gap" |
| "decoder direction" | "dictionary atom" (use sparingly, only geometric context) |
| "activation frequency" | "firing rate" (acceptable), "base rate" |
| "gold-standard labels" / "IG labels" | "exact labels", "true labels" |
| "first-letter spelling task" | "spelling task", "letter task" |
| "fine-tuning" | "finetuning", "fine tuning" |
| "few-shot" | "few shot", "fewshot" |
| "pre-trained" | "pretrained", "pre trained" |
| AUROC (uppercase) | AUC, ROCAUC |
| AUPRC (uppercase) | AUPR, PRAUC |
| "absorption rate" | "absorption score", "absorption metric" |

---

## Citations Shorthand

| Shorthand | Full Reference |
|---|---|
| Chanin et al. (2024) | "Absorption, Superposition, and SAE Features" — FeatureAbsorptionCalculator and sae_spelling library |
| Tang et al. (2025) | arXiv:2512.05534 — sparsity landscape partial minimum theory |
| O'Neill et al. (2024) | arXiv:2411.13117 — amortization gap formalization |
| SAEBench | Karvonen et al. (2025) — aggregate absorption rate benchmark |
| iter_001 EDA finding | Internal result; cite as "prior iteration (EDA, AUROC=0.650)" or "(this work, iter_001)" |
