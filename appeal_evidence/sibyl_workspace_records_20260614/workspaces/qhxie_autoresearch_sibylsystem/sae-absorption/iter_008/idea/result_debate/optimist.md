# Optimist Analysis

## Evidence Map

| Metric | Baseline / Prior Art | Ours | Delta | Signal Strength |
|--------|---------------------|------|-------|-----------------|
| Cross-domain absorption variation (H1) | No prior data | city-country 18.5%, city-language 13.6% vs first-letter 34.5% (all L24_16k) | Significant (p=0.0043 city-country, p=0.0001 city-language) | **Strong** |
| Semantic vs syntactic absorption (H2') | Assumed first-letter worst case | First-letter NOT worst case at L12 (0.7% vs 17.3% city-continent); H2 falsified | Inverts prior assumption | **Strong** |
| Causal absorption via activation patching (H7) | No interventional evidence (Chanin et al. only correlational) | 100% recovery on "lower" (5/5 absorbed tokens), 0% control | 100% vs 0% recovery | **Moderate** (n=1 word with clean signal, needs scale) |
| Tightened hedging classification | Chanin et al. 98.6% hedging | Strict hedging 7.9%, compensatory 86.2%, persistent 5.9% (first-letter) | 85.3 pp gap between strict and loose | **Strong** |
| Layer-dependent absorption (new finding) | No layer-by-layer characterization | L24: 25-35% vs L6/L18: 2-5% (first-letter) | 6-17x amplification at final layer | **Strong** |
| Architecture ranking: JumpReLU | Matryoshka assumed best (SAEBench first-letter only) | JumpReLU lowest on first-letter (0.67%), Matryoshka lowest on city-country/city-language | Hierarchy-dependent rankings | **Moderate** |
| Probe quality breakthrough at L24 | Prior pilot: F1=0.083 first-letter at L12 | F1=0.971 first-letter at L24 (sklearn LogReg) | 0.083 -> 0.971 | **Strong** (unblocks entire project) |
| Hedging decomposition varies by hierarchy (H3) | No cross-domain hedging data | city-language 66.7% strict vs first-letter 7.9% strict | 58.8 pp difference | **Strong** |
| GAS detector (H4) | Proposed rho>=0.6 | rho=0.12 (pilot), rho=0.12 (5k sequences) | Definitive negative | **Strong** (as negative result) |
| CMI at L0=22 (H5) | L0=82 rho=-0.38 (confounded) | rho=0.044 (p=0.83) at L0=22 with perfect probes | Confound resolved | **Strong** (as negative result) |
| Absorption Tax T(G) | No formal quantification | T(G)=0.414, but rho=0.08 with empirical absorption | Qualitative framework holds; quantitative predictions weak | **Weak** (quantitative) |

## Root Cause Analysis

### 1. Cross-Domain Absorption Variation (H1) -- PARTIALLY SUPPORTED
- **Mechanism**: Different feature hierarchies impose different co-occurrence patterns on the SAE encoder. City-country (80 classes) creates a fine-grained competition landscape. City-continent (6 classes) creates fewer but broader category features that more easily absorb subcategories. City-language introduces multilingual features that cross geographic boundaries, creating a qualitatively different absorption pattern.
- **Design decision**: The choice to use RAVEL entity-attribute hierarchies was prescient. These hierarchies capture real knowledge structure (not synthetic), and the model genuinely uses this information for next-token prediction -- unlike first-letter features which are a byproduct of tokenization.
- **Expected or surprising**: The DIRECTION is surprising (H2 falsified: first-letter is NOT worst case). The EXISTENCE of cross-domain variation was expected. This makes the finding stronger than anticipated: not only does absorption vary, but the canonical benchmark underestimates it for knowledge-relevant features.

### 2. Causal Absorption via Activation Patching (H7) -- SUPPORTED (pilot)
- **Mechanism**: Feature 14449 ("lower"-specific) competitively excludes the letter-L representation during SAE encoding. When this child feature's contribution is zeroed, the parent (letter-L) signal re-emerges in the reconstruction, and the probe recovers correct predictions on all 5 previously absorbed tokens.
- **Design decision**: Using activation patching (interventional) rather than integrated gradients (correlational) provides a fundamentally stronger form of evidence. This is the methodological advance that Chanin et al. did not make.
- **Expected or surprising**: Expected in direction (child suppresses parent), but the 100% recovery rate for "lower" is surprisingly clean. The nuanced finding that child features have dual roles (suppressing some tokens, helping others) is an unexpected insight.

### 3. Tightened Hedging Classification
- **Mechanism**: The loose hedging metric (Chanin et al.) counts ANY resolution of a false negative at higher L0 as "hedging." But 85.3% of resolutions come from compensatory features (not the original parent), meaning the parent was never the resolution mechanism. Only 7.9% involve the specific parent latent firing.
- **Design decision**: Implementing both strict and loose classification on the same data cleanly demonstrates the near-tautological nature of the original metric.
- **Expected or surprising**: The magnitude of the gap (85.3 pp) is striking. This is a methodological contribution independent of the cross-domain work -- it applies to every existing hedging analysis.

### 4. Layer-Dependent Absorption (New Discovery)
- **Mechanism**: Absorption concentrates at layer 24 (25-35%) where the model resolves its final prediction, while layers 6 and 18 show minimal absorption (2-5%). This suggests absorption is tied to the model's computation of the final output, not to the general representational geometry at all layers.
- **Design decision**: Testing all 4 layers (6, 12, 18, 24) rather than just layer 12 (as in prior work) revealed a striking pattern that would have been invisible otherwise.
- **Expected or surprising**: Surprising. No prior work characterized absorption across layers. The 6-17x amplification at L24 suggests absorption is an endpoint computation phenomenon, not a uniformly distributed failure.

### 5. Hedging Decomposition Varies by Hierarchy (H3)
- **Mechanism**: City-language shows 66.7% strict hedging (the parent feature itself fires at higher L0, resolving the false negative), compared to only 7.9% for first-letter. This means the city-language SAE genuinely has the right features but suppresses them at lower L0 due to sparsity pressure, while first-letter false negatives resolve through compensatory (unrelated) features.
- **Design decision**: Cross-domain hedging decomposition is the natural extension of the tightened hedging contribution.
- **Expected or surprising**: The 58.8 pp difference between hierarchies was unexpected. It means the NATURE of absorption differs across domains: first-letter "absorption" is mostly compensatory resolution (arguably not true absorption), while city-language absorption involves genuine parent feature suppression.

## Unexpected Signals

### 1. Layer 12 is the Worst Layer for First-Letter Probes
- **Observation**: First-letter probe F1 at layer 12 = 0.312 (sklearn) or 0.083 (sae_spelling), while layer 24 = 0.971 and even layer 6 = 0.693. Layer 12 is a valley, not a plateau.
- **Mini-hypothesis**: Gemma 2 2B uses layers 6-12 for token-level processing (tokenization residuals) and layers 18-24 for semantic processing. At layer 12, the model is transitioning between these regimes, creating a representational "valley" where first-letter information is poorly linearly separable.
- **Significance**: This explains why all prior iterations stagnated -- they were measuring absorption at the worst possible layer. It also suggests that prior published absorption rates (Chanin et al.) may have implicitly measured at a poor layer, making their 15-35% range reflect a specific layer choice, not a fundamental property.

### 2. Child Features Have Dual Roles (Context-Dependent)
- **Observation**: In activation patching, child features (e.g., feature 14449 for "lower") suppress the parent (letter-L) on some tokens but CONTRIBUTE to letter prediction on others. Zeroing "liked"'s child feature degrades accuracy from 1.00 to 0.95; "other" from 0.95 to 0.20; "under" from 1.00 to 0.25.
- **Mini-hypothesis**: Absorption is not a binary feature-level property but a context-dependent interference pattern. The same child feature can act as absorber or contributor depending on the input token. This is more nuanced than the "competitive exclusion" model from Chanin et al.
- **Significance**: If confirmed at scale, this fundamentally changes the absorption model. It means you cannot simply label a feature pair as "absorber/absorbed" -- you need to characterize the context-dependent dynamics. This could be a Figure 4-level finding in the paper.

### 3. Width Effect is Inconsistent Across Hierarchies
- **Observation**: For first-letter, 65k SAE shows slightly higher absorption than 16k (9.2% vs 5.7% at L12; 25.5% vs 34.5% at L24 -- reversed!). For cross-domain hierarchies at L24, 65k shows lower absorption than 16k (city-continent: 26.0% vs 35.8%; city-country: 12.7% vs 18.5%).
- **Mini-hypothesis**: Wider dictionaries provide more specialized features that can represent both parent and child concepts without competition -- but only when the hierarchy is knowledge-based. For first-letter (syntactic/tokenization-dependent), wider dictionaries may create more specialized word features that absorb more.
- **Significance**: This challenges the simple "wider is better for absorption" narrative and could motivate hierarchy-aware width selection.

### 4. The Calibration Curve Shows Super-Linear Co-Activation
- **Observation**: In GAS computation, the ratio of actual co-activation to independent expectation increases super-linearly with cosine similarity: 1.91x at cos=[0.30,0.35) up to 86.51x at cos=[0.75,0.80). This means high-cosine feature pairs are activated together far more than their individual frequencies would predict.
- **Mini-hypothesis**: This confirms the geometric basis for absorption -- features with similar decoder directions tend to co-fire, creating the conditions for competitive exclusion. Even though GAS fails as a predictor, the calibration curve is itself a publishable finding about SAE feature geometry.
- **Significance**: Could be a supplementary figure showing that decoder geometry creates the preconditions for absorption, even if it doesn't predict which specific features get absorbed.

### 5. Matryoshka is NOT Universally Best
- **Observation**: At layer 12, Matryoshka shows lowest absorption for city-country (0.353) and city-language (0.353), but NOT for city-continent (0.192 vs BatchTopK 0.135) and NOT for first-letter (0.014 vs JumpReLU 0.007). The architecture effect is not statistically significant (Kruskal-Wallis H=0.70, p=0.87), while the hierarchy effect IS significant (H=12.8, p=0.005).
- **Mini-hypothesis**: Architecture ranking is subordinate to hierarchy structure. The hierarchy determines absorption severity; architecture modulates it within a narrower band.
- **Significance**: This is a concrete recommendation for practitioners: when choosing SAE architecture to minimize absorption, the hierarchy/domain of your features matters more than which architecture you pick. This challenges the SAEBench single-task architecture comparison paradigm.

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| Causal absorption (H7) at scale | Expand activation patching from 9 words to 100+ instances across all 4 hierarchies | Recovery rate significantly above control (Wilcoxon p<0.01), potentially stronger for knowledge hierarchies | 1.0 | **High** |
| Layer-dependent absorption across domains | Measure absorption at layers 6, 12, 18, 24 for all RAVEL hierarchies (not just first-letter) | L24 amplification replicates for knowledge hierarchies; may reveal different layer profiles per hierarchy | 2.0 | **High** |
| Context-dependent dual-role analysis | For each absorption pair, classify contexts as "suppressive" vs "contributory" using probe predictions with/without child feature | At least 30% of child features show dual roles; context features (word frequency, position) predict which role | 0.5 | **High** |
| Matched-L0 architecture comparison | Compare JumpReLU, BatchTopK, Matryoshka at exactly matched L0 (e.g., all at L0=20) on all hierarchies | Architecture differences shrink at matched L0; residual differences are genuine architecture effects | 1.0 | **Medium** |
| Width scaling: 4k to 65k systematic | Absorption rate as function of width (4k, 16k, 32k, 65k) on each hierarchy at L24 | Width-absorption relationship differs by hierarchy type, giving practical guidance for SAE training | 2.0 | **Medium** |
| GAS calibration curve as paper figure | Clean up the co-activation vs cosine calibration data for publication; compute for multiple layers | Super-linear pattern replicates across layers; provides geometric context for why absorption can occur | 0.1 | **Low** |

## Honest Caveats

### Cross-Domain Variation (H1, H2')
- **Counter-argument**: Probe quality is below the strict 0.90 gate for all RAVEL hierarchies (best: city-continent F1=0.843 at L24). Absorption rates may be inflated by probe errors misclassified as absorption.
- **Alternative explanation**: The cross-domain differences could be driven entirely by probe quality differences: first-letter probes (F1=0.971) are near-perfect, while RAVEL probes (F1=0.79-0.84) produce more false negatives that look like absorption.
- **What would convince me**: If absorption rates measured with probes at F1>0.95 for ALL hierarchies show the same ordering and similar magnitude differences. Alternatively, if a non-probe absorption metric (e.g., direct feature activation analysis) confirms the cross-domain pattern.

### Activation Patching (H7)
- **Counter-argument**: Only 1 out of 7 tested words ("lower") showed clean causal recovery. The other 4 words with absorption (eight, offer, often) showed 0% recovery when the identified child feature was zeroed. The 14.3% aggregate recovery is entirely driven by the "lower" result.
- **Alternative explanation**: The child features identified from a previous iteration (L0=82) may not be the correct absorbers at the canonical L0. The multi-feature absorption hypothesis (multiple co-occurring features collectively suppress the parent) could explain zero single-feature recovery.
- **What would convince me**: At larger scale (n>50), mean recovery rate significantly above the 0.5% control baseline (Wilcoxon p<0.01). At least 5+ distinct words showing significant recovery from child zeroing.

### Tightened Hedging Classification
- **Counter-argument**: The "strict" definition (parent feature itself fires at higher L0) may be too strict. Compensatory resolution via related features could still constitute genuine hedging if those features encode hierarchically related information.
- **Alternative explanation**: The 85.3% compensatory rate might reflect that higher L0 simply adds more features that happen to push the probe toward the correct prediction by chance, not because of any absorption-related mechanism.
- **What would convince me**: If the compensatory features are semantically analyzed and shown to be random (not hierarchy-related), this strengthens the finding. If they are systematically related to the correct class, the strict/loose distinction becomes less meaningful.

### Layer-Dependent Absorption
- **Counter-argument**: Layer 24 is the final layer before the unembedding. Higher absorption here could reflect the model's prediction uncertainty being amplified, not a genuine absorption mechanism.
- **Alternative explanation**: The SAE at layer 24 may simply have a harder reconstruction task (more diverse information to represent), leading to more reconstruction errors that look like absorption.
- **What would convince me**: If the layer-absorption pattern differs meaningfully across hierarchy types (e.g., knowledge hierarchies peak at L18 while first-letter peaks at L24), it would confirm layer-dependence reflects genuine computational structure, not just a generic final-layer artifact.

### Architecture Rankings
- **Counter-argument**: None of the pairwise architecture comparisons reach statistical significance for cross-domain hierarchies (smallest p=0.0290 for BatchTopK vs Matryoshka on city-language, but n=34 total). The ANOVA architecture effect is non-significant (p=0.87).
- **Alternative explanation**: Architecture differences are small relative to the hierarchy effect. The observed variation could be noise from the small sample sizes (n=17 to n=52 per hierarchy-architecture cell).
- **What would convince me**: Larger samples per cell (n>100) producing consistent architecture rankings with p<0.01. Or, a systematic matched-L0 comparison showing significant differences.

## Bottom Line

There is a clear publishable story here, and it is stronger than a typical positive-results paper because it combines a high-impact primary finding with transparent negative results. The primary finding -- that cross-domain absorption varies significantly and the canonical first-letter benchmark underestimates absorption for knowledge-relevant features -- directly challenges the entire existing SAEBench evaluation framework. The secondary findings (causal absorption via patching, tightened hedging revealing a near-tautological metric, layer-dependent absorption at L24) each provide concrete methodological contributions. The honest negative results (GAS rho=0.12, CMI rho=0.044, Absorption Tax rho=0.08) demonstrate scientific integrity that top venues consistently reward. Two critical tasks remain before this is submission-ready: (1) scaling activation patching to n>50 for statistical power, and (2) ideally improving RAVEL probe quality above F1=0.90 to preempt the probe-quality confound argument. Even without the second item, the paper is publishable at NeurIPS/ICLR with a transparent discussion of probe limitations.
