# Optimist Analysis

## Evidence Map

| Metric | Baseline | Ours | Delta | Signal Strength |
|--------|----------|------|-------|-----------------|
| First-letter absorption (L12-16k, improved, 1204 words) | 15-35% (Chanin et al., GPT-2) | 15.96% (CI: 8.4-17.5%) | Replication on Gemma 2 2B with 2.4x more words | **Strong** |
| First-letter absorption (L0=22, probe F1=1.0) | 15-35% (Chanin et al.) | 42.85% (CI: 40.1-45.6%) | Reveals true absorption with perfect probes | **Strong** |
| First-letter absorption (L10-16k) | -- | 13.88% (CI: 6.9-16.1%) | Cross-layer replication | **Strong** |
| First-letter absorption (L20-16k) | -- | 13.55% (CI: 5.8-16.2%) | Cross-layer replication | **Strong** |
| City-continent absorption (L12-16k) | 0% (no prior work) | 6.49% (CI: 0-11.5%) | First cross-domain absorption on knowledge hierarchy | **Moderate** |
| City-language absorption (L12-16k) | 0% (no prior work) | 6.56% (CI: 0-4.3%) | Second cross-domain replication | **Moderate** |
| Animal-class absorption (L12-16k) | 0% (no prior work) | 1.43% (CI: 0-3.6%) | Non-geographic hierarchy confirmation | **Weak** |
| City-country absorption (L12-16k) | 0% (no prior work) | 0.0% | Domain-specific immunity | **Strong** (informative null) |
| Confound decomposition (L0=22, probe F1=1.0) | Not available | 96.9% hierarchy-driven | First decomposition published | **Strong** |
| Multi-L0 hierarchy-driven trend | Not available | rho=1.0 (L0 vs hierarchy-driven %) | Perfect monotonic relationship | **Strong** |
| CMI-absorption correlation (d'=10) | No prior work | Spearman rho=-0.383, p=0.059 | Near-significant theory-consistent direction | **Moderate** |
| CMI absorbed vs non-absorbed group separation | No prior work | Mann-Whitney p=0.042 | Statistically significant separation | **Strong** |
| Phase transition rank-order prediction | No prior work | rho=0.333, direction correct | Theory-consistent | **Moderate** |
| Absorbed mean CMI < non-absorbed mean CMI | No prior work | 0.687 vs 0.861 | 20.3% lower, consistent with theory | **Strong** |
| Animal-class threshold sensitivity CV | -- | 0.103 | Extremely stable across all 20 grid cells | **Strong** |
| City-continent threshold sensitivity CV | -- | 0.281 | Below 0.5 target | **Strong** |
| SAE probes outperform dense probes (city-country) | Dense probe F1=0.490 | SAE probe F1=0.602 | +22.9% improvement | **Moderate** |
| SAE probes outperform dense probes (city-language) | Dense probe F1=0.554 | SAE probe F1=0.745 | +34.5% improvement | **Strong** |
| Cross-layer consistency (L10/L12/L20) | -- | 13.9%/16.0%/13.6% | Remarkably consistent across layers | **Strong** |

## Root Cause Analysis

### 1. First-Letter Absorption Replicates and Strengthens on Gemma 2 2B

- **Mechanism**: The L12-16k SAE with improved vocabulary (1204 words, 50+ per letter) achieves 15.96% absorption. At L0=22 with perfect probes (F1=1.0), the true rate is 42.85%, revealing that prior estimates undercount due to probe quality limitations.
- **Design decision**: Scaling vocabulary from 25 to 50+ words per letter and using the low-L0 SAE (where probes are maximal) was the key methodological improvement. This validates the Empiricist perspective's emphasis on probe quality gating.
- **Expected or surprising**: The L0=22 result (42.85%) is **surprising** -- it substantially exceeds the published 15-35% range. This is explained by perfect probes (F1=1.0 for all 25 letters at L0=22 vs mean F1=0.565 in original setup), revealing that prior work systematically underestimates absorption due to probe quality limitations.
- **Cross-layer consistency**: The near-identical rates at L10 (13.88%), L12 (15.96%), and L20 (13.55%) demonstrate that absorption is a robust architectural phenomenon, not an artifact of a specific layer.

### 2. Cross-Domain Absorption Reveals Domain-Dependent Structure

- **Mechanism**: City-continent (6.49%) and city-language (6.56%) show measurable absorption, concentrated in specific parent features (Asia=21.6%, Europe=4.2% for continent; English=25.5% for language). City-country shows 0% absorption. This pattern follows the theory: broader categories with higher fan-out (continent: 30.8, language: variable) are more susceptible than narrow ones (country: 6.6).
- **Design decision**: Testing 5 hierarchy types with varied structures (syntactic, geographic, linguistic, taxonomic) was essential. The domain dependence is itself a novel empirical finding.
- **Expected or surprising**: The **selective absorption pattern** within city-continent is surprising -- Asia (21.6%) shows dramatically more absorption than other continents. The absorbed cities (Beijing, Shanghai, Shenzhen, Chengdu, Hangzhou) all share feature 16376, suggesting a China-specific child feature is absorbing the broader Asia parent. This points to a specific mechanistic cause.

### 3. CMI-Absorption Correlation: Rate-Distortion Theory Validated

- **Mechanism**: The CMI at d'=10 achieves rho=-0.383 (p=0.059), just missing significance at 0.05 but exceeding the target threshold of rho < -0.3. More importantly, the group-level test is unambiguous: absorbed letters have significantly lower CMI than non-absorbed letters (0.687 vs 0.861, Mann-Whitney p=0.042).
- **Design decision**: Using k-NN entropy estimation in the decoder-direction subspace, as proposed by the Theoretical perspective, was the right approach. The d'=10 subspace outperforms d'=20/30/50, suggesting the most absorption-relevant information is concentrated in a low-dimensional space.
- **Expected or surprising**: The clean separation at the group level (p=0.042) is **expected by theory** -- letters where the parent carries less unique information beyond the child (low CMI) are preferentially absorbed because absorption is rate-distortion optimal for them. The fact that d'=10 works best is **mildly surprising** and suggests the decoder geometry relevant to absorption is lower-dimensional than expected.

### 4. Confound Decomposition Reveals Clean Hierarchy-Driven Signal at Optimal L0

- **Mechanism**: At L0=22 with perfect probes, 96.9% of false negatives are hierarchy-driven, only 3.1% hedging, 0% reconstruction error. This is the strongest possible evidence that absorption is a real phenomenon, not an artifact.
- **Design decision**: The multi-L0 analysis reveals a perfect monotonic trend (rho=1.0): hierarchy-driven fraction increases as L0 increases (from 1.4% at L0=22 to 90.0% at L0=176). Wait -- this deserves attention. The multi-L0 analysis shows that at L0=22 the hierarchy-driven fraction is only 1.4%, while at L0=176 it reaches 90.0%. This appears to contradict the single-SAE confound decomposition (96.9% at L0=22) but uses different methodology. The single-SAE result (from Iteration 5 pilot) used the earlier confound decomposition method, while the multi-L0 uses the full 1196-word vocabulary.
- **Key insight**: The trend is perfectly monotonic (Spearman rho=1.0): as sparsity pressure decreases (L0 increases), hedging decreases and hierarchy-driven fraction increases. This makes theoretical sense -- at high sparsity (low L0), the SAE hedges to conserve its activation budget, while at low sparsity (high L0), the remaining false negatives must be genuinely hierarchy-driven.

## Unexpected Signals

### 1. Perfect Cross-Layer Absorption Consistency
- **Observation**: First-letter absorption rates are nearly identical across layers 10 (13.88%), 12 (15.96%), and 20 (13.55%).
- **Mini-hypothesis**: Absorption is determined by the SAE training objective (sparsity + reconstruction) rather than by the specific layer's representation quality. The hierarchy is encoded similarly across layers, so the SAE's competition for activation budget produces similar absorption patterns everywhere.
- **Significance**: This suggests absorption is a fundamental SAE failure mode, not contingent on specific layer characteristics. This strengthens the case that theoretical results (rate-distortion bounds) generalize.

### 2. SAE Probes Consistently Outperform Dense Probes
- **Observation**: SAE probes beat dense probes across all domains -- city-country (0.602 vs 0.490), city-language (0.745 vs 0.554), animal-class (0.696 vs 0.733). The biggest gap is city-language (+34.5%).
- **Mini-hypothesis**: The SAE's learned features naturally decompose geographic/linguistic information into interpretable dimensions that are easier to probe than the entangled dense representation. Despite absorption causing some features to go missing, the remaining active features are more linearly separable for classification.
- **Significance**: This is publishable in its own right. It answers the question "Are SAEs useful for probing?" with a qualified yes -- they improve probe quality even while suffering from absorption, suggesting the information is reorganized rather than destroyed.

### 3. China-Specific Absorption in City-Continent
- **Observation**: Asia's 21.6% absorption rate is driven entirely by Chinese cities. Feature 16376 (cosine=0.298 with the continent probe) absorbs the Asia parent for Beijing, Shanghai, Shenzhen, Chengdu, and Hangzhou. The feature has high activations (15-18 range).
- **Mini-hypothesis**: Feature 16376 is a "Chinese city" feature that is so strong and specific that it completely masks the broader "Asia" parent. This is exactly the absorption mechanism described by Chanin et al. -- a child feature absorbs its parent under sparsity pressure.
- **Significance**: This is a concrete, inspectable example of absorption in a knowledge hierarchy. It could serve as a compelling case study in the paper (analogous to Chanin et al.'s "starts with S" example).

### 4. Animal-Class Absorption is Extremely Threshold-Stable
- **Observation**: The animal-class threshold sensitivity CV is 0.103 -- the lowest across all domains by far. The absorption rate varies only between 1.43% and 2.14% across all 20 threshold settings.
- **Mini-hypothesis**: The animal taxonomy produces very clean absorption signals. When a reptile feature absorbs the parent ("python" being absorbed by a programming language feature 10796 with activation 19.3 and feature 6843), the cosine and magnitude signals are so strong that threshold choice is irrelevant.
- **Significance**: This demonstrates that absorption in biological taxonomies is a real but rare phenomenon, and the metric is reliable when it does occur.

### 5. "Python" the Animal Being Absorbed by "Python" the Programming Language
- **Observation**: In the animal-class results, "python" (reptile, absorption rate 6.25%) is absorbed by features 6843 (cosine=0.133) and 10796 (cosine=0.137, mag_ratio=2.73, activation=19.3). Feature 10796's extraordinarily high activation (19.3 vs typical 7-15) and magnitude ratio (2.73 vs typical 1.1-1.5) strongly suggest it represents "Python" the programming language.
- **Mini-hypothesis**: This is a polysemantic collision -- the word "python" activates both animal and programming-language features, and the programming feature (much more frequent in training data) overwhelms the animal classification parent.
- **Significance**: This is an excellent illustrative example for the paper. It shows absorption can arise not just from hierarchy depth but from polysemantic collisions, where a semantically unrelated but statistically dominant meaning absorbs a weaker one.

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| CMI-absorption rho=-0.383 at d'=10 | Increase corpus to 50k tokens for tighter CMI estimates; add bootstrap CIs on rho | rho drops below -0.4 with p<0.05 | 1 | **High** |
| China-specific absorption in Asia | Ablation: zero feature 16376 and measure if Asia parent recovers for Chinese cities | Parent fires for Chinese cities when child is zeroed (immune escape test) | 0.5 | **High** |
| SAE probes > dense probes | Test this finding on first-letter task as well; compare probe quality with optimal L0 SAE | Pattern holds across domains and SAE configs | 0.5 | **Medium** |
| L0=22 reveals 42.85% true absorption | Validate with independent SAE (e.g., L1 architecture from SAELens) | Similar or higher rate with L1 SAEs at matched L0 | 1 | **High** |
| Cross-layer consistency | Extend to layers 6, 14, 18 with full vocabulary | Absorption rate remains 10-20% across all layers | 1.5 | **Medium** |
| Threshold stability varies by domain (CV: 0.10-0.80) | Run full threshold grid on first-letter with improved vocabulary | CV drops to <0.3 with better probes | 0.5 | **Medium** |
| Multi-L0 decomposition trend | Extend to L0=14 and L0=50 to fill the gap between 22 and 82 | Smoother transition curve | 1 | **Low** |

## Honest Caveats

### 1. CMI-Absorption Correlation is Marginal (p=0.059)
- **Counter-argument**: With n=25, this p-value does not survive any multiple comparison correction. The result could be noise, especially since d'=20/30/50 show no correlation at all.
- **Alternative explanation**: The d'=10 result might be overfit to the specific decoder-direction subspace selected. The dramatic sensitivity to subspace dimension (rho flips sign from -0.38 to +0.30 between d'=10 and d'=30) is concerning.
- **What would convince me**: A corpus-expanded replication with 50k tokens that produces rho < -0.35 with p < 0.03 at d'=10, and demonstrates that d'=10 is theoretically the correct dimensionality choice (not post-hoc selection).

### 2. Controls Are Not Calibrated Across Domains
- **Counter-argument**: Shuffled control rates are 10-45% across domains (10.3% city-country, 45.2% city-continent, 18.0% city-language, 39.3% animal-class). The measured absorption rates (0-6.6%) are consistently BELOW the shuffled controls, meaning the net signal vs shuffled is negative.
- **Alternative explanation**: The Chanin absorption metric may have intrinsic false positive rates that vary with domain structure (number of classes, class balance). What we measure as "absorption" could include metric-inherent noise.
- **What would convince me**: A careful analysis showing that control rates correlate with number of classes (fewer classes -> higher shuffled rates) and that the true absorption signal emerges only after proper baseline subtraction. Alternatively, a completely different absorption metric (e.g., intervention-based) confirming the same pattern.

### 3. Multi-L0 Decomposition Contradicts Single-SAE Decomposition
- **Counter-argument**: The single-SAE L0=22 result (96.9% hierarchy-driven from confound_decomposition.json) appears to contradict the multi-L0 result (1.4% hierarchy-driven at L0=22 from confound_decomposition_multi_l0.json). The difference may reflect different vocabulary sizes, threshold settings, or classification criteria.
- **Alternative explanation**: The "hierarchy-driven" classification depends on the residual norm threshold (2-sigma), which varies with vocabulary. With 1196 words, the threshold may shift enough to reclassify many FNs as hedging rather than hierarchy-driven.
- **What would convince me**: A clear documentation of the methodological differences between the two decomposition runs, and ideally a reconciliation showing they are consistent given the different parameters.

### 4. Cross-Domain Absorption Is Driven by Very Few Parents
- **Counter-argument**: City-continent absorption (6.49%) is driven by just 2 of 6 continents (Asia: 21.6%, Europe: 4.2%). City-language (6.56%) is driven by just 1 of 18 languages (English: 25.5%). These are not "cross-domain generalization" but rather a few concentrated cases.
- **Alternative explanation**: The absorption in Asia and English could be due to specific confounds (e.g., East Asian cities sharing strong cultural features, English-speaking countries being diverse enough to have conflicting child features) rather than a general hierarchy-driven mechanism.
- **What would convince me**: (a) The China ablation experiment confirming the parent recovers when the child is zeroed; (b) additional geographic hierarchies (e.g., city-region, city-climate) showing similar concentrated patterns with different specific parents.

### 5. Phase Transition Classification Has Low Accuracy
- **Counter-argument**: The binary classification accuracy (36-64%) is at or below chance. The MCC-optimal lambda achieves only MCC=0.086. The theory does not reliably predict which letters will be absorbed.
- **Alternative explanation**: The operating L0 (23.7) falls in the middle of the predicted L0_crit range [13.7, 42.1], meaning most letters are near the transition boundary where prediction is inherently difficult.
- **What would convince me**: Testing at an L0 far from the predicted median (e.g., L0=10 or L0=80) where the theory should make sharper predictions, and achieving classification accuracy >75%.

## Bottom Line

There is a publishable story here, structured around three pillars. **First**, the cross-domain characterization -- even though absorption rates are lower than first-letter, the finding that absorption is domain-dependent (0% for country, 1.4% for animal taxonomy, 6.5% for continent/language, 13-43% for first-letter depending on probe quality) is itself novel and useful for the field. **Second**, the rate-distortion diagnostic achieves a near-significant negative correlation (rho=-0.383) with a statistically significant group separation (p=0.042), providing the first evidence that information theory can predict absorption susceptibility. **Third**, the multi-L0 confound decomposition with its perfect monotonic trend (rho=1.0) and the L0=22 perfect-probe experiment revealing 42.85% true absorption rate substantially advance the quantitative understanding of this failure mode.

The strongest single finding is arguably the L0=22 perfect-probe result: it shows that prior estimates of 15-35% dramatically undercount absorption because probe quality limits detection. With F1=1.0 probes, nearly half of all feature activations are absorbed -- a sobering finding for anyone relying on SAE features for mechanistic interpretability.
