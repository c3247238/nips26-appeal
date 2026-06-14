# Critique of Experiments: Quantifying Feature Absorption in Sparse Autoencoders

## Summary

The pilot experiments (H1, H3, H4, H5) are well-designed with appropriate controls and rigorous metrics. The random dictionary control validation is exemplary. However, three issues reduce the experiments' informativeness: (1) the cumulative subsampling method for H5 may not generalize to independently trained SAEs; (2) H4's design flaw (task-agnostic latent selection) produced uninformative results; (3) the pilot scale (100 sequences) may miss rare absorption events for H1, though the 100x gap is too large to bridge with 10x scaling.

---

## What Works

### Random dictionary control validation
The use of random Gaussian decoders to establish the null distribution is excellent scientific practice. By construction, random decoders yield $A_f = 0$, confirming the metric detects learned structure rather than numerical artifacts.

### Clear falsification criteria
Each hypothesis has a pre-registered falsification criterion, and results are reported against those criteria without softening. The H1 falsification (0.19% vs >20% predicted, 100x gap) is clearly stated.

### Proper sensitivity analysis
Table C1 in the appendix confirms that absorption rarity is robust across RVE thresholds (0.70, 0.80, 0.90) and co-firer counts (top-3, top-5, top-10). Rankings are stable, confirming the conclusion is not an artifact of the specific parameter choices.

### Layer-wise design for H3
Auditing 6 layers (0, 2, 4, 6, 8, 10) provides sufficient resolution to detect the inverted-U pattern. The Spearman correlation (r = 0.086, p = 0.872) correctly rules out a monotonic relationship.

---

## Critical Issues

### 1. H5 Cumulative Subsampling May Not Generalize

The H5 experiment simulates dictionary sizes 2K and 8K by cumulatively subsampling latents from the full 24K SAE, prioritizing "absorbable" latents for inclusion. This gives an upper bound on absorption for each subsample size. However, independently trained SAEs with dictionary sizes 2K may not have the same latent structure as a 24K SAE with those latents subsampled out. The training process may produce qualitatively different dictionaries at different sizes, not just a subset of a larger dictionary.

**Recommendation**: Acknowledge this as a limitation. The finding (larger dict → lower absorption) is directionally valid but may not generalize to independently trained SAEs. The paper already mentions this in the Discussion but should be more explicit in the Methodology section.

### 2. H4 Design Flaw Produces Uninformative Results

Both low-absorption and high-absorption latent subsets yield 0.000 faithfulness on the France/Paris circuit, while the full SAE achieves 0.289. The difference is 0.000, making the hypothesized comparison impossible. The problem is that corpus-wide absorption scores do not predict circuit-level causal importance — the France/Paris circuit may be driven by latents that are neither high-absorption nor low-absorption by corpus standards.

**Root cause**: The two-step selection (select by corpus-wide absorption, then test on a specific circuit) is methodologically flawed. The circuit is not neutral to absorption selection; it has its own latent importance structure that is uncorrelated with corpus-wide absorption scores.

**Better design**: Compare full SAEs at different layers (layer 4 vs. layer 8) rather than subsets of a single SAE. This would hold dictionary size and training run constant while varying absorption level.

### 3. H1 Pilot Scale May Miss Rare Events

The H1 pilot uses 100 sequences (12,800 tokens). The result (0.19% at layer 8) is far below the hypothesis (>20%), so the conclusion is robust regardless of scale. However, if absorption were more rare than hypothesized (e.g., 2% instead of 20%), the pilot might miss it. The paper correctly notes this as a limitation, but the conclusion (absorption is rare) is well-supported even at pilot scale given the 100x gap.

---

## Minor Issues

### H3 L0 as proxy for lambda

L0 is used as a proxy for the L1 sparsity penalty lambda, but L0 varies across layers even for SAEs trained with the same lambda. This is acknowledged in the paper but should be stated more explicitly in the Methodology.

### 8 perfect-score latents

Eight latents achieve $A_f = 1.0$, each firing on exactly 100 tokens (one per sequence). This suspicious pattern (100 = number of sequences) suggests a position-embedding artifact rather than genuine semantic absorption. The paper correctly flags this as an open question but does not investigate it further. This is acceptable for a pilot study but should be addressed in a follow-up.

### H2 skipped

H2 was not run because H1/H3/H4 failures led to early termination. The correct justification (insufficient absorption variance at layer 8) is somewhat inconsistent with Section 5.5's editorializing about the decision not being data-driven. Layer 4 had 49.3% absorption (~12,000 latents), which would have provided sufficient variance for H2. The decision to skip H2 should be framed as "early termination after multiple falsifications" rather than "insufficient variance at the chosen layer."

---

## Assessment

The pilot experiments are methodologically sound with appropriate controls. The main limitations are: (1) H5 subsampling may not generalize to independently trained SAEs; (2) H4 design flaw produced uninformative results; (3) H2 was skipped due to early termination but the justification is inconsistent. The random dictionary control validation and sensitivity analysis are exemplary.