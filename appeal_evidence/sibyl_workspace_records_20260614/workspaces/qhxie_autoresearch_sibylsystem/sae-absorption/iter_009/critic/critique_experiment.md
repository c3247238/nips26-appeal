# Experiment Critique

## Overall Assessment: 6/10

The experimental execution covers an impressive scope -- four hierarchies, multiple layers, multiple SAE architectures, activation patching, benign/pathological diagnostics, and five correlational predictors. The breadth is commendable. However, six experimental issues undermine the reliability of the headline claims.

## Critical Issues

### 1. Benign/Pathological Diagnostic is Partially Circular

The diagnostic ablates the parent direction from the child feature's decoder vector and measures the logit change. This is presented as testing whether the model "independently uses the parent feature when the child is active." But it actually tests whether the child feature's decoder carries parent-direction information -- which is the definition of absorption. If a child feature absorbs the parent, its decoder vector necessarily has a large component along the parent direction (that is how the SAE reconstructs both parent and child from a single latent). Ablating that component will always produce large logit changes, regardless of whether the model's computation independently relied on the parent.

A proper test of computational redundancy would:
- Ablate the parent feature's activation directly (not the parent direction from the child decoder)
- Use path patching to determine if parent information flows through independent circuits
- Compare the model's behavior when the parent feature fires vs. when it does not, holding the child constant

The current test conflates "the child decoder carries parent information" (geometric) with "the model computationally depends on the parent" (causal). The 3.98 nats effect establishes that absorption has large magnitude. It does not establish that absorption is computationally pathological.

### 2. Single-Hierarchy Pathological Testing Generalized to Universal Claim

The benign/pathological diagnostic runs on city-continent only (F1=0.87). The paper headline ("absorption is always pathological," "100% pathological") treats this as a universal result. City-continent's distributed absorption mechanism (Section 5.2) may behave differently from first-letter's concentrated absorption. In the concentrated regime, the child feature explicitly captures a single parent, so the pathological question is more directly answerable. Testing only the distributed regime and generalizing to the concentrated regime is not warranted.

### 3. Cross-Domain Patching Child Identification May Be Broken

For first-letter, child feature identification is well-defined: a word-specific SAE feature that fires for "Saturday" but not for other words. For city-continent, what is the "child feature" for the parent "Europe"? The integrated-gradients attribution identifies the active feature with highest attribution to the probe's changed prediction. But with distributed absorption, no single feature dominates, and the identified "child" may be a random feature. The 0.05% recovery rate (vs. 14.5% for random zeroing) is consistent with zeroing the wrong feature entirely. The "distributed vs. concentrated" interpretation is one possibility; "broken child identification" is another, equally consistent with the data.

Test: does the identified child's decoder have above-chance cosine similarity with the parent probe direction? Does zeroing the top-5 attributed features improve recovery? These controls would distinguish the two interpretations.

### 4. Probe Quality Confound Unquantified

The paper acknowledges that RAVEL absorption rates are "upper bounds" due to probe imperfection, but never quantifies the bias. The probe-only baseline (control 3 in Section 3.3) -- what is the false negative rate on raw activations alone? -- is defined in the methodology but never reported. Without this number, we cannot bound the probe-induced inflation.

Rough estimate: if city-country probe has F1=0.73, approximately 27% of instances are misclassified. The "probe-correct" denominator filters out some of these, but probe boundary effects (instances near the decision boundary that flip prediction with minor perturbation) remain. A sensitivity analysis subtracting estimated probe-induced FNs from the measured FN count would bound the confound.

### 5. First-Letter Absorption Rate Aggregation Undocumented

The full-mode data file contains two first-letter L24_16k absorption rates: 0.2707 (per-unique-word) and 0.3448 (per-instance, computed as `iter008_baseline`). The paper reports 27.1% without explaining which aggregation is used. Per-unique-word averages across the 3 prompt templates per word, giving each word equal weight. Per-instance weights by number of prompts where the probe is correct. These can differ substantially (here by 28%).

Additionally, the raw probe accuracy at L24 is only 75.9% despite F1=1.0 on the test set. This means many prompt instances fail the probe even before SAE encoding. The relationship between probe test-set performance and prompt-specific performance is unclear.

### 6. Token Position Asymmetry

First-letter uses position -6 (sae-spelling convention); RAVEL uses position -2. Different token positions may encode information differently, and SAE encoding may disrupt these positions differently. This is a confound that is never mentioned.

## Positive Aspects

- **Activation patching for first-letter** (n=25, d=1.33, p=0.000218) is a genuine contribution. The effect size is large, the control is appropriate, and 16/19 words with absorption show positive recovery.
- **Threshold sensitivity analysis** (CV=0.077 across 20 configurations) convincingly demonstrates that absorption is structural, not threshold-dependent.
- **Shuffled hierarchy control** (near-zero rates) confirms hierarchy-specificity.
- **Per-class analysis** reveals fascinating within-hierarchy variation (Europe 90% vs. Africa 4% for city-continent) that motivates future work.
- **Comprehensive negative results** (5 correlational approaches all fail) with exact statistics. This is the paper's strongest methodological contribution.
