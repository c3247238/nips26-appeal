# Glossary

Unified terminology for the paper. All section writers, critics, and the editor reference this file. Use the **preferred form** consistently; avoid the listed alternatives.

---

## Core Concepts

| Term | Preferred Form | Definition | Avoid |
|------|---------------|------------|-------|
| Feature absorption | feature absorption | A parent SAE latent fails to fire on parent-positive inputs when a more specific child latent is active. The parent has high precision but systematic, invisible recall holes. | absorption (alone, without "feature" on first use in each section); feature suppression |
| Hedging | hedging | Information spreading across many SAE latents due to insufficient capacity. Produces false negatives that mimic absorption but resolve when sparsity is relaxed (higher L0). | feature hedging (acceptable but less concise); information diffusion |
| Competitive exclusion | competitive exclusion | The hypothesized mechanism where a child latent actively suppresses a parent latent under sparsity pressure. Implies a causal relationship: removing the child would restore the parent. | feature competition; mutual suppression |
| Confound decomposition | confound decomposition | Our method of separating hedging-driven FNs from hierarchy-driven FNs by comparing behavior across multiple L0 operating points. | decomposition analysis |
| L0 operating point | $L_0$ operating point | The configured number of non-zero SAE latents per forward pass. Controls the sparsity-fidelity tradeoff. Always typeset as $L_0$, not L0, in the manuscript body. | sparsity level (too vague); activation count |
| Phase transition | phase transition | The sharp change in absorption rate in the $L_0 \approx 40$--80 range, where absorption drops from >35% to <15%. | regime change; critical threshold |

## SAE Architecture

| Term | Preferred Form | Definition | Avoid |
|------|---------------|------------|-------|
| Sparse Autoencoder | SAE | An autoencoder with sparsity-inducing objective that maps polysemantic activations to an overcomplete sparse basis. Always expand on first use in each section. | sparse auto-encoder; Sparse Auto-Encoder |
| L1-ReLU SAE | L1-ReLU SAE | SAE variant using continuous L1 penalty on latent activations. Produces graded suppression. | L1 SAE; ReLU SAE (ambiguous) |
| JumpReLU SAE | JumpReLU SAE | SAE variant using hard per-latent thresholds; activations either exceed the threshold or equal zero. Trained via straight-through estimator. | Jump-ReLU; jump ReLU |
| Gemma Scope | Gemma Scope | Suite of 400+ open JumpReLU SAEs on Gemma 2 2B/9B/27B released by DeepMind. | GemmaScope; Gemma-Scope |
| SAE latent | SAE latent | A single feature direction in the SAE dictionary, with associated encoder/decoder weights and activation value. | SAE feature (acceptable but we prefer "latent" to distinguish from ground-truth features); SAE neuron |
| Decoder column | decoder column | The $j$-th column of the decoder matrix $W_d$, unit-normalized. Represents the direction in residual stream space associated with latent $j$. | decoder direction; feature direction |
| Parent latent | parent latent | An SAE latent encoding a general concept (e.g., "starts-with-A"). | parent feature (acceptable in informal contexts) |
| Child latent | child latent | An SAE latent encoding a more specific concept within the parent's scope (e.g., a word-specific token). | child feature (acceptable in informal contexts) |

## Measurement Protocol

| Term | Preferred Form | Definition | Avoid |
|------|---------------|------------|-------|
| k-sparse probe | $k$-sparse probe | Logistic regression using only $k$ SAE latents (selected by decoder cosine similarity to probe direction). $k = 5$ throughout this paper. | sparse logistic probe; k-sparse logistic regression (acceptable on first use) |
| Probe direction | probe direction | The learned weight vector $v_p$ of the $k$-sparse logistic regression. | probe vector; probe weight |
| Candidate feature | candidate feature | SAE latent with $\cos(d_j, v_p) \geq 0.025$ to the probe direction. | aligned feature; matching feature |
| False negative (FN) | false negative | A parent-positive token where all $k$ probe-associated latents are inactive but the probe correctly classifies it. | absorption candidate; missed token |
| Shuffled-label control | shuffled-label control | Control condition using permuted class assignments. Expected to produce absorption $\leq$ measured; INVERTED in all 5 domains. | shuffled control; random-label control |
| Permissive hedging | permissive hedging | Classification where any token ceasing to be FN at a higher L0 is labeled as hedging. Upper bound. 98.6% at L0=22. | loose hedging; broad hedging |
| Strict hedging | strict hedging | Classification requiring at least 1 of $k$ parent-associated latents to fire at L0=176. More conservative. 6.2% at L0=22. | tight hedging; narrow hedging |
| Persistent core word | persistent core word | A token that is FN at all 4 tested $L_0$ values {22, 41, 82, 176}. 8 such words identified. | persistent false negative; cross-L0 FN |
| Activation patching | activation patching | Causal intervention: zero a specific latent's activation and observe whether other latents' activations change. | ablation (acceptable but less precise); causal intervention |

## Statistical Terms

| Term | Preferred Form | Definition | Avoid |
|------|---------------|------------|-------|
| Bootstrap CI | bootstrap CI | 95% confidence interval from 10,000 bootstrap resamples (seed=42). | confidence interval (always specify bootstrap and resamples) |
| Bonferroni correction | Bonferroni correction | Multiple comparison correction: multiply p-values by the number of comparisons. | Bonferroni adjustment; FWER correction |
| Coefficient of variation | CV | Standard deviation divided by mean. Used for threshold sensitivity (0.077) and cross-layer stability (<10%). | variability coefficient |
| Spearman correlation | Spearman $\rho_s$ | Rank-based correlation. Typeset as $\rho_s$. | Spearman rho (in body text, use the symbol) |
| Partial correlation | partial correlation | Correlation between two variables after controlling for a third. Used to control for probe F1 in CMI analysis. | conditional correlation |
| Leave-one-out | leave-one-out | Jackknife sensitivity analysis: remove each letter in turn, recompute statistic. | LOO (acceptable in tables and figures) |

## Information Theory

| Term | Preferred Form | Definition | Avoid |
|------|---------------|------------|-------|
| Conditional mutual information | CMI | $I(X; f_p \mid f_c)$: information the parent encodes about input $X$ beyond what the child encodes. Estimated via $k$-NN in a $d'$-dimensional decoder subspace. | conditional MI; mutual information (too vague) |
| Successive refinement | successive refinement | Information-theoretic framework (Equitz & Cover, 1991): lossless staged encoding iff descriptions form a Markov chain. | multi-resolution coding |
| Rate-distortion | rate-distortion | Theoretical framework connecting compression quality to information rate. CMI diagnostic is motivated by rate-distortion theory. | R-D theory |

## Abbreviations

| Abbreviation | Expansion | Notes |
|-------------|-----------|-------|
| SAE | Sparse Autoencoder | Expand on first use per section |
| LLM | Large Language Model | Expand on first use per section |
| CMI | Conditional Mutual Information | Expand on first use per section |
| FN | False Negative | Expand on first use per section |
| CI | Confidence Interval | Always specify "bootstrap" and number of resamples |
| CV | Coefficient of Variation | |
| LOO | Leave-One-Out | Use in tables only; spell out in text |
| RAVEL | Relation Attribute Verification and Evaluation of Language Models | Dataset for cross-domain hierarchies |
| STE | Straight-Through Estimator | JumpReLU training method |

## Models and Datasets

| Name | Preferred Form | Notes |
|------|---------------|-------|
| Gemma 2 2B | Gemma 2 2B | Primary model. Not "Gemma-2-2B" or "Gemma 2B". |
| GPT-2 Small | GPT-2 Small | Secondary model for cross-architecture comparison. Not "GPT2 Small". |
| SAEBench | SAEBench | 8-metric evaluation suite. Not "SAE Bench" or "SAE-Bench". |
| SAELens | SAELens | SAE training/loading library. Not "SAE Lens" or "SAE-Lens". |
| TransformerLens | TransformerLens | Activation extraction library. Not "Transformer Lens". |
| WordNet | WordNet | Source for animal-class hierarchy. |
