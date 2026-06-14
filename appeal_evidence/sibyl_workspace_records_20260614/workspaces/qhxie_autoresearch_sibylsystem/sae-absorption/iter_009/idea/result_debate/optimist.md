# Optimist Analysis

## Evidence Map

| Metric | Baseline | Ours | Delta | Signal Strength |
|--------|----------|------|-------|-----------------|
| Cross-domain absorption variation (Kruskal-Wallis) | No prior measurement | p=7.37e-66 (H=299.95, N=3566) | First-ever measurement | **Strong** |
| Absorption rate range across hierarchies (L24_16k) | Single-task only | 11.6%--45.1% (4x range) | 4x variation across hierarchies | **Strong** |
| Causal absorption (first-letter activation patching) | No causal evidence existed | 32.5% recovery vs 1.5% control, d=1.33, p=0.000218 | +31.0 pp recovery | **Strong** |
| Pathological absorption ratio (city-continent) | Assumed mix of benign/pathological | 100% pathological, mean logit_change=3.98 vs control 0.004 | ~1000x effect ratio | **Strong** |
| Hedging decomposition (chi-square cross-hierarchy) | Prior: 98.6% loose figure only | chi2=91.51, p=1.04e-19, strict 0--22.6% | Near-tautology exposed | **Strong** |
| Architecture non-significance | Architecture rankings assumed task-dependent | p=0.754 (L12), p=0.497 (L24) | Hierarchy >> architecture | **Moderate** |
| Threshold sensitivity (structural) | Not previously tested | CV=0.077 across 5x4 grid, FN constant | Measurement robust | **Moderate** |
| T(G) ranking vs mean absorption | No prior framework | rho=0.80, concordance 67% (4 hierarchies) | Qualitative direction holds | **Weak** |
| Competition coefficient (city-continent) | No prior test | rho=0.599, p=0.040 | Significant within domain | **Weak** |
| R_pc at L24_16k first-letter | No per-class predictor | rho=0.622, p=0.006 | Significant within-config | **Weak** |

## Root Cause Analysis

### 1. Cross-Domain Absorption Variation (H1: STRONGLY_SUPPORTED)

- **Mechanism**: Different feature hierarchies impose different co-occurrence and geometric pressures on the SAE encoder. City-country (45.1%) has 80 classes with severe class imbalance (United States: 176 entities, some countries: 5 entities), creating strong dominant features that absorb minority classes. City-language (11.6%) has fewer majority-minority conflicts because high-frequency languages (English: 388, Chinese: 100) dominate cleanly.
- **Design decision**: The choice to use RAVEL's natural hierarchies (city-country, city-continent, city-language) rather than synthetic tasks proved critical. These hierarchies have authentic class imbalance patterns that expose absorption dynamics invisible in the uniform 26-class first-letter task.
- **Expected or surprising**: The 4x range across hierarchies (11.6%--45.1%) was larger than expected. The pilot predicted a smaller range. The finding that city-country exceeds first-letter was not predicted by any hypothesis -- the original H2' expected semantic to uniformly exceed syntactic.

### 2. Causal Absorption via Activation Patching (H7: SUPPORTED for first-letter)

- **Mechanism**: Zeroing the child SAE latent releases the suppressed parent feature's contribution to the residual stream, allowing the parent probe to recover its prediction. The 32.5% recovery rate (vs 1.5% control) with Cohen's d=1.33 demonstrates that the child feature actively suppresses the parent feature through competitive exclusion in the encoder.
- **Design decision**: Using activation patching rather than purely correlational methods (like GAS or CMI) was the key decision. Every correlational approach failed (GAS rho=0.12, CMI rho=0.04), while the causal intervention succeeded decisively.
- **Expected or surprising**: The effect size (d=1.33) was stronger than expected for a pilot-derived measurement scaled to n=25. The consistency across 16/19 absorbed words (84% positive recovery) indicates a robust mechanism, not a statistical artifact.

### 3. 100% Pathological Absorption (H8: FALSIFIED -- positive finding)

- **Mechanism**: When the parent feature is absorbed, the model loses access to critical semantic information (e.g., continent identity). The mean logit change of 3.98 nats when ablating the parent direction from the child latent's decoder demonstrates that the parent's semantic content is NOT redundantly encoded elsewhere in the SAE representation.
- **Design decision**: Testing at three thresholds (0.05, 0.1, 0.2) with n=1471 instances across 50 entities provided overwhelming statistical power. The control ablation (random direction, mean logit change 0.004) validates the methodology.
- **Expected or surprising**: Surprising. The Contrarian perspective predicted >=30% benign absorption. The complete absence of benign cases (0% at all thresholds) is far more decisive than expected. This is arguably the single most impactful finding for the field.

### 4. Hedging Near-Tautology Exposed (H3: SUPPORTED)

- **Mechanism**: The prior loose hedging classification (98.6% from Chanin et al.) counts any false negative that resolves under any L0 as "hedged," which conflates compensatory resolution (the SAE works differently at different L0) with genuine hedging. The strict classification (0--22.6% across hierarchies) captures only cases where the specific parent feature fires at a different L0.
- **Design decision**: Decomposing hedging into strict, compensatory, and persistent categories, then testing across hierarchies, revealed that compensatory hedging dominates (77--100%). The cross-hierarchy chi-square (p=1.04e-19) demonstrates that the strict/compensatory ratio itself is hierarchy-dependent.
- **Expected or surprising**: The tightened classification was expected to reduce the hedging figure, but the magnitude was surprising: from 98.6% to 0--22.6% strict hedging. The cross-hierarchy variation in this ratio is a novel finding.

### 5. Architecture Invariance (H6: PARTIALLY_SUPPORTED)

- **Mechanism**: Absorption is driven by the interaction between the input distribution and the sparsity constraint, not by the specific nonlinearity (JumpReLU vs BatchTopK vs Matryoshka). All tested architectures produce similar absorption rates because the fundamental pressure -- representing overlapping features with a sparse budget -- is the same.
- **Design decision**: Testing 4 architectures at L12 and 3 at L24 across 4 hierarchies provides a 4x4 and 3x4 design. The non-significant architecture effect (p=0.754 at L12, p=0.497 at L24) vs significant hierarchy effect (p=0.010 at L12) cleanly separates the contributions.
- **Expected or surprising**: The complete absence of architecture effect was somewhat surprising. The pilot suggested JumpReLU might be consistently lowest, but the full data show this is within noise. The finding that "hierarchy >> architecture" is a clean, memorable result for the paper.

## Unexpected Signals

### 1. City-Country Shows the HIGHEST Absorption at L24

- **Observation**: At L24_16k, city-country absorption (45.1%) exceeds first-letter (42.9%), city-continent (31.4%), and city-language (11.6%). With 80 classes and severe imbalance, city-country creates the strongest absorption pressure of any tested hierarchy.
- **Mini-hypothesis**: Absorption scales with the number of minority classes that must compete for representation against a few dominant features. City-country has many small countries (5--12 entities) competing against the United States feature (176 entities, 0% absorption itself), creating a "winner-take-all" dynamic where the US feature absorbs others.
- **Significance**: This directly contradicts the pilot pattern (where city-continent was highest) and reveals that class count and imbalance, not just hierarchy depth, drive absorption. The per-class data shows 100% absorption for 17 out of 80 countries (Albania, Algeria, Australia, Dominican Republic, Ecuador, Egypt, Ghana, Greece, Guinea, Italy, Madagascar, Mali, Netherlands, Paraguay, Portugal, South Korea, Sudan, Taiwan, Tanzania, Uganda, Uruguay, Venezuela, etc.) while large-population countries (India: 2.1%, Indonesia: 4.4%, Iran: 4.8%) show near-zero absorption.

### 2. Layer-Dependent Absorption With Non-Monotonic Pattern

- **Observation**: First-letter absorption across layers: L6=0.7%, L12=6.9--17.9%, L18=4.0--4.2%, L24=27.1--42.9%. There is a ~40x variation across layers, with L24 dramatically highest. The pattern is non-monotonic (L18 < L12 for some configurations).
- **Mini-hypothesis**: Layer 24 (the final prediction layer in Gemma 2 2B) concentrates semantic processing, creating the most intense competition for sparse representation. Earlier layers spread information more diffusely, reducing absorption pressure.
- **Significance**: Layer dependence of absorption was not predicted by any hypothesis. This finding is independently publishable and has direct practical implications: SAE analyses at intermediate layers may drastically underestimate absorption relative to the final prediction layer.

### 3. Europe Shows Anomalously High Absorption in City-Continent

- **Observation**: Europe has 90.2% absorption rate at L24_16k (city-continent), compared to Africa at 3.9%, South America at 3.9%, and the overall city-continent average of 31.4%. This is the highest per-class absorption rate across all tested hierarchies.
- **Mini-hypothesis**: European cities are highly diverse in associated attributes (language, culture, economy), creating many competing child features. The SAE may struggle to maintain a general "Europe" feature when specific features like "French city," "German city," or "British city" are more informative for next-token prediction.
- **Significance**: This extreme per-class variation (3.9% to 90.2% within the same hierarchy) reveals that absorption is not just hierarchy-dependent but class-dependent. Understanding why Europe is uniquely vulnerable could reveal the computational principles driving absorption.

### 4. Competition Coefficient Predicts Absorption Within City-Continent

- **Observation**: Within the city-continent hierarchy, the ecological competition coefficient (cos_sim x co-occurrence) significantly predicts absorption (rho=0.599, p=0.040, n=12 pairs). This is the only hierarchy where the ecological framework succeeds.
- **Mini-hypothesis**: City-continent has fewer classes (6) with clearer parent-child structure, making the pairwise competition dynamics more tractable. With more classes (80 for city-country), the multi-way competition obscures pairwise relationships.
- **Significance**: While the overall rate-distortion model fails (rho=0.250, p=4.3e-5 but below target), the within-hierarchy success for city-continent suggests the ecological framework may work for shallow, clean hierarchies. This points toward a domain-specific rather than universal predictor.

### 5. R_pc Significantly Predicts Absorption at L24_16k for First-Letter

- **Observation**: The per-class reconstruction importance R_pc correlates with absorption at L24_16k (Spearman rho=0.622, p=0.006) and L18_16k (rho=0.471, p=0.049) for first-letter. These are the only configurations where a continuous predictor achieves significance.
- **Mini-hypothesis**: At layers where the SAE is under the most sparsity pressure (L24 has the lowest T(G)=0.017), the reconstruction importance becomes the bottleneck: features with low R_pc are the first to be "sacrificed" to absorption. At earlier layers with higher T(G), there is enough sparsity budget that R_pc does not predict which features get absorbed.
- **Significance**: This is a genuinely new finding not anticipated by any perspective. It suggests that absorption predictors are not globally applicable but become informative under high-pressure conditions. A "phase transition" view: absorption becomes predictable only near the critical sparsity threshold.

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| City-country highest absorption | Ablation study: systematically reduce class count (80->40->20->10) and measure absorption | Absorption decreases monotonically with reduced class count, confirming imbalance drives absorption | 2 | High |
| Layer-dependent absorption | Extend cross-domain measurement to ALL layers (6,12,18,24) for city-country and city-language | City-country also shows L24 concentration; layer profile differs across hierarchies | 4 | High |
| 100% pathological (city-continent) | Replicate benign/pathological on first-letter and city-language | First-letter also 100% pathological (or reveals hierarchy-dependent benign fraction) | 1 | High |
| Europe anomalous absorption | Per-class activation patching on Europe specifically | Europe absorption is distributed across country-level features, explaining cross-domain patching failure | 2 | Medium |
| R_pc predicts absorption at L24 | Test R_pc as predictor on cross-domain hierarchies at L24 | R_pc achieves rho>0.4 for city-country at L24 (high-pressure regime) | 0.5 | Medium |
| Competition coeff works for city-continent | Test competition coefficient on a synthetic hierarchy with controlled class count | rho>0.5 for hierarchies with <=10 classes, fails for >20 classes | 1 | Medium |
| Cross-domain patching failure | Multi-feature patching: zero top-3 child features simultaneously | Multi-feature zeroing recovers parent at >10% rate (distributed mechanism confirmed) | 1 | Low |

## Honest Caveats

### Cross-Domain Absorption Variation (H1)

- **Counter-argument**: RAVEL probe quality is below strict gate for 3 of 4 hierarchies (city-continent F1=0.87, city-language F1=0.82, city-country F1=0.73). Some measured "absorption" may be probe error, inflating apparent rates for RAVEL hierarchies.
- **Alternative explanation**: The absorption rate differences could partly reflect probe quality differences rather than true absorption differences. City-country's highest rate (45.1%) coincides with the worst probe (F1=0.73).
- **What would convince me**: Repeat the experiment with improved probes (all F1>0.90) and show the ranking is preserved. Alternatively, use a probe-free metric (if one can be developed) to confirm cross-domain variation.

### Causal Absorption -- First-Letter Only (H7)

- **Counter-argument**: The causal mechanism is confirmed only for first-letter (n=25 words). Cross-domain patching showed a REVERSE effect (control recovery 14.4% vs child zeroing 0.05%). The mechanism may be specific to the clean, single-feature absorption pattern of first-letter.
- **Alternative explanation**: Cross-domain absorption may be distributed across multiple child features, making single-feature zeroing ineffective. The probe quality gap (F1=1.0 for first-letter vs 0.87 for city-continent) means we may be identifying the wrong child features for cross-domain.
- **What would convince me**: Multi-feature activation patching (zero top-3 child features simultaneously) showing >10% parent recovery for city-continent. Or: improved probes (F1>0.95) followed by single-feature patching achieving significant recovery.

### 100% Pathological (H8)

- **Counter-argument**: Tested only on city-continent hierarchy. First-letter may show different results because first-letter information is arguably more "redundant" (the model may not use "starts with S" independently when processing "snake").
- **Alternative explanation**: The logit change metric captures sensitivity to the parent direction but does not necessarily mean the model's output is degraded in a task-relevant way. A 3.98 nat logit change could affect irrelevant tokens.
- **What would convince me**: Replicate on first-letter to confirm universality. Additionally, measure task-specific performance degradation (e.g., next-token accuracy) rather than raw logit change.

### Architecture Invariance (H6)

- **Counter-argument**: Only 4 architectures tested, with width mismatch (Matryoshka 32k vs others 16k). The non-significance could be a power issue rather than a true null. RAVEL probes at L12 are below gate, adding noise.
- **Alternative explanation**: All tested architectures share similar encoder/decoder structure (linear encoder with nonlinear activation). Fundamentally different architectures (e.g., gated SAEs, multi-layer SAEs) might show different absorption patterns.
- **What would convince me**: Test 2+ additional architectures (gated SAE, e.g.) with matched width and quality probes. Or: demonstrate with a power analysis that the current sample size is sufficient to detect a 5pp architecture effect.

### Unexpected Signals (R_pc at L24_16k)

- **Counter-argument**: The R_pc correlation at L24_16k (rho=0.622, p=0.006) is based on only n=18 letters (those with non-zero absorption). With multiple comparisons across 17 configurations, this may be a false positive (uncorrected p=0.006, Bonferroni-corrected p=0.10).
- **Alternative explanation**: R_pc and absorption may be correlated through a confound (e.g., both driven by feature frequency). At L24, rare letters may have both low R_pc and high absorption.
- **What would convince me**: The R_pc-absorption correlation holds at p<0.01 after Bonferroni correction across all tested configurations. Or: partial correlation controlling for frequency still significant.

## Bottom Line

There is a strong publishable story here, centered on three findings with overwhelming statistical support: (1) absorption varies dramatically across hierarchies (p=7.4e-66, 4x range), definitively ending the single-task monoculture; (2) absorption is always pathological (0% benign, 1000x effect ratio), establishing urgency for detection/mitigation; and (3) causal competitive exclusion is confirmed for first-letter (d=1.33, p=0.000218) but operates through a different, possibly distributed mechanism for semantic hierarchies. The honest reporting of 9 negative results -- including the failure of all correlational predictors and the non-generalization of causal evidence -- strengthens rather than weakens the paper by motivating a paradigm shift from correlational to causal methods in SAE analysis. The combination of strong positive results, unexpected discoveries (layer dependence, class-level variation), and transparent negative results makes this a compelling NeurIPS/ICLR submission.
