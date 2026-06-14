# Glossary: Beyond Competitive Exclusion

Unified terminology definitions for all paper writers, critics, and editors.
Use the preferred phrasing listed here. Do not introduce synonyms without adding them here first.

---

## Core Concepts

**Sparse Autoencoder (SAE)**
A neural network trained to reconstruct model activations via a sparse code; used in mechanistic interpretability to decompose residual stream activations into interpretable learned features.
Preferred: "Sparse Autoencoder (SAE)" on first use, "SAE" thereafter.
Do NOT use: "sparse auto-encoder", "dictionary learning network".

**feature absorption** (lower-case noun phrase)
The failure mode where a parent latent's activation is suppressed to zero on inputs where a more specific child latent is active, even though the parent concept is present in the input. Originally defined by Chanin et al. (2024).
Preferred: "feature absorption" or just "absorption" after first definition.
Do NOT use: "feature suppression" (different meaning), "absorption failure", "absorption artifact".

**parent latent / parent feature**
A latent representing a more general concept that subsumes the child concept (e.g., "starts-with-A words" is parent to "starts-with-A proper nouns"; "European cities" is parent to "French cities").
Preferred: "parent latent" or "parent feature". Use consistently within a paragraph; do not alternate freely.

**child latent / child feature**
A latent representing a more specific concept whose activation suppresses the parent latent.
Preferred: "child latent" or "child feature".

**absorption rate**
Fraction of parent-positive inputs on which all probe-associated latents fail to activate despite correct probe classification. Report as percentage (e.g., 15.96%).
Preferred: "absorption rate". Always specify the SAE config and L0 when reporting.

---

## Metric Audit Terminology

**Chanin metric / Chanin et al. metric**
The supervised absorption measurement protocol from Chanin et al. (2024, NeurIPS 2025 Oral): train linear probes to identify parent features, find SAE latents whose decoder directions align with the probe, measure the false-negative rate of those latents on parent-positive inputs.
Preferred: "Chanin metric" after first definition as "Chanin et al. absorption metric".
Do NOT use: "canonical metric" (too vague), "ground-truth metric" (it is not ground truth).

**shuffled control / shuffled-label control**
A control where parent-class labels are randomly permuted before training probes and measuring absorption. Expected outcome: absorption rate lower than measured rate with true labels. The universal failure of this control is a primary finding.
Preferred: "shuffled control" or "shuffled-label control".

**random probe control**
A control using a random unit-sphere direction as the probe instead of a trained probe. Expected outcome: absorption rate < 2%.
Preferred: "random probe control".

**dense probe**
A logistic regression probe trained on raw model activations (not SAE latent activations). Provides an upper bound on probe quality and a baseline for SAE probe comparison.
Preferred: "dense probe".

**control failure**
The empirical finding that shuffled-label controls produce higher "absorption" rates than true labels across all tested domains. Indicates the metric is not measuring hierarchy-driven competitive exclusion in the tested configuration.
Preferred: "control failure" or "universal control failure" for the cross-domain finding.

---

## Confound Decomposition

**hedging**
The mechanism where a parent feature's information is distributed across many SAE latents (none clearing the activation threshold) rather than being concentrated in a single parent latent. Hedging produces false negatives that look like absorption but result from information spreading, not competitive suppression. Distinguished from hierarchy-driven absorption by resolving at higher L0 values as sparsity relaxes.
Preferred: "hedging". Do NOT use: "information diffusion" (too vague), "feature bleeding" (not established).

**hierarchy-driven absorption / competitive exclusion**
Genuine absorption where the child latent actively suppresses the parent latent under sparsity pressure. Distinguished from hedging by persisting across all tested L0 values regardless of sparsity pressure.
Preferred: "hierarchy-driven absorption" or "competitive exclusion" (interchangeable).
Do NOT use: "true absorption" (implies value judgment).

**confound decomposition**
The analysis that classifies each parent false-negative into hedging, hierarchy-driven, or reconstruction error categories by examining persistence across multiple L0 operating points.
Preferred: "confound decomposition".

**hedging dominance**
The empirical finding that 98.6% of false negatives at L0=22 are hedging, with only 1.4% hierarchy-driven. This is the paper's most important empirical finding.
Preferred: "hedging dominance" as a noun phrase.

---

## L0 and Sparsity

**L0 operating point**
The configured number of non-zero SAE latents per forward pass. For JumpReLU SAEs, this is determined by the hard threshold parameter. Gemma Scope provides multiple L0 settings (22, 41, 82, 176) per layer-width combination.
Preferred: "L0 operating point" or just "L0" after first definition.
Do NOT use: "sparsity level" without specifying L0 numerically.

**L0-absorption phase transition**
The empirical finding that absorption declines monotonically from 42.9% (L0=22) to 0.8% (L0=176), with the steepest decline in the L0~40-80 range.
Preferred: "L0-absorption phase transition" or "L0 phase transition".

**persistent core** / **core absorbed words**
The 9 words that show non-zero absorption at all tested L0 values (22, 41, 82, 176). Strongest candidates for genuine competitive exclusion.
Preferred: "persistent core" or "core absorbed words".

---

## Rate-Distortion Theory

**conditional mutual information (CMI)**
$I(X; f_{\text{parent}} \mid f_{\text{child}})$: the unique information the parent feature carries about the input beyond what the child feature already captures. Low CMI means the parent is nearly redundant given the child.
Preferred: "conditional mutual information (CMI)" on first use; "CMI" thereafter.
Do NOT use: "mutual information" without the "conditional" qualifier (different quantity).

**successive refinement**
A rate-distortion theory result (Equitz and Cover, 1991): a source is successively refinable (can be encoded in stages without loss) iff the descriptions form a Markov chain. Applied to SAEs: if X -- f_child -- f_parent is Markov (CMI = 0), absorbing the parent is information-theoretically lossless.
Preferred: "successive refinement theorem".

**geometric constant**
$c(w_P, w_C) = \|w_P\|^2 (1 - \cos^2(w_P, w_C))$: the geometric factor in the rate-distortion absorption threshold. For unit-normalized SAEs, this degenerates to $\sin^2(\text{angle}(w_P, w_C)) \approx 0.97$ (negligible variability).
Preferred: "geometric constant $c$".

**critical L0 / L0_crit**
The predicted L0 value from rate-distortion theory above which absorption becomes information-theoretically suboptimal for a given parent-child pair: $L_{0,\text{crit}} = \lambda / (\text{CMI} \cdot c)$.
Preferred: "$L_{0,\text{crit}}$" or "critical L0".

---

## SAE Architecture

**JumpReLU SAE**
An SAE with hard threshold activation: the pre-activation must exceed a per-latent threshold $\theta_j$ to produce non-zero output. Used in Gemma Scope (Rajamanoharan et al., 2024). Produces exact L0 control.
Preferred: "JumpReLU SAE". Do NOT use: "JumpReLU" alone as a noun (ambiguous).

**L1-ReLU SAE**
An SAE trained with an L1 sparsity penalty on activations and ReLU nonlinearity. Standard architecture in SAELens GPT-2 suite. Does not provide exact L0 control.
Preferred: "L1-ReLU SAE" or "L1 SAE" after first definition.

**Gemma Scope**
The publicly released collection of Sparse Autoencoders for Gemma 2 2B, from Google DeepMind. JumpReLU architecture with multiple L0 settings.
Preferred: "Gemma Scope". Full citation on first use.

**SAEBench**
The benchmark suite for evaluating SAE quality across multiple metrics including absorption, published by Karvonen et al. (2025).
Preferred: "SAEBench". Do NOT use "SAE Bench" (two words).

---

## Cross-Domain Hierarchies

**hierarchy domain**
A parent-child feature pair type used for absorption measurement. This paper tests 5 domains: first-letter (26 letter classes), city-country (29 countries), city-continent (6 continents), city-language (18 languages), animal-class (6 classes).
Preferred: "hierarchy domain" or "domain".

**probe quality gate**
The minimum probe F1 threshold (0.85) that a parent class must pass before its absorption rate is included in analysis. Parents failing this gate are reported separately.
Preferred: "probe quality gate (F1 > 0.85)".

---

## Statistical Terms

**bootstrap CI / bootstrap confidence interval**
Confidence interval estimated by resampling the data (10,000 resamples, seed = 42 throughout).
Preferred: "95% bootstrap CI" or "95% CI" after the method is defined.

**Spearman rho / Spearman rank correlation**
Non-parametric rank correlation; reported as $\rho_s$.
Preferred: "Spearman $\rho$" on first use; "$\rho$" thereafter in context.

**Mann-Whitney U test**
Non-parametric test for comparing distributions between two groups.
Preferred: "Mann-Whitney U test".

**Kruskal-Wallis test**
Non-parametric test for comparing distributions across three or more groups.
Preferred: "Kruskal-Wallis test".

**Cohen's d**
Standardized effect size: $d = (\mu_1 - \mu_2) / \sigma_{\text{pooled}}$.
Preferred: "Cohen's $d$" on first use; "$d$" in context.

**Hartigan's dip test**
Statistical test for unimodality; dip > 0 with p < 0.05 indicates non-uniform (potentially bimodal) distribution.
Preferred: "Hartigan's dip test".

**Bonferroni correction**
Multiple comparison correction: multiply p-values by the number of tests. Used for CMI dimension sensitivity analysis.
Preferred: "Bonferroni correction".

---

## Writing Conventions

- Numbers: use Arabic numerals for all numeric results (0.383, not "zero point three eight three").
- Percentages: "15.96%" not "15.96 percent" or "16%".
- Decimal precision: report absorption rates to 2 decimal places (15.96%); correlations to 3 decimal places (0.383); p-values as exact values ("p = 0.059") not inequality ranges ("p < 0.10").
- Equations: number all displayed equations; refer as "Equation (1)" in text.
- Figure references: always cite figure before it appears: "As shown in Figure 3, ..."
- Table references: always capitalize: "Table 1", "Table 2".
- Model names: "Gemma 2 2B" (not "Gemma-2-2B"); "GPT-2 Small" (not "GPT2 Small" or "GPT-2").
- SAE notation: "16k SAE" means $d_{\text{SAE}} = 16384$; "65k SAE" means $d_{\text{SAE}} = 65536$.
- L0 notation: "L0=22" (no spaces around equals sign in inline text); "$L_0 = 22$" in equations.
- Conditional claims: Claims dependent on pending experiments (CMI replication, activation patching) must be marked with [conditional] in the outline and with explicit hedging language ("if replicated", "pending validation") in the text.
