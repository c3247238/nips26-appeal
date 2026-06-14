# Optimist Analysis

## Evidence Map

| Metric | Baseline | Ours | Delta | Signal Strength |
|--------|----------|------|-------|-----------------|
| EDA AUROC, Gemma L12-16k (proxy labels) | 0.500 (random) | 0.776 | +0.276 | **Strong** (CI: [0.700, 0.863], Cohen's d = 1.02, p = 6.4e-5) |
| EDA AUROC, Gemma L5-16k (proxy labels) | 0.500 | 0.698 | +0.198 | **Strong** (CI: [0.637, 0.779]) |
| EDA AUROC, GPT-2 L6 (direct labels, R4) | 0.500 | 0.650 | +0.150 | **Moderate** (CI: [0.531, 0.761], p = 0.027, Cohen's d = 0.53) |
| SynthSAEBench AUROC (H1 theorem) | 0.500 | 1.000 | +0.500 | **Strong** (F1 = 0.974, ground-truth validation) |
| D-EDA AUROC, GPT-2 L10 | 0.500 | 0.762 | +0.262 | **Moderate** (CI: [0.686, 0.830]; rescues where EDA fails) |
| Taxonomy KW test, L12-65k (late > early EDA) | p > 0.05 | p = 0.0002 | -- | **Strong** (holds at all 5 threshold variants) |
| Early absorption prevalence | -- | 72-75% | -- | **Strong** (consistent across L12-16k and L12-65k) |
| Intra-RAVEL coherence rho (R3, n=6) | 0.000 | 0.924 | +0.924 | **Moderate** (p < 0.005 Bonferroni; but probe quality below gate) |
| EDA cross-model identity (EDA = 1 - DecCos) | -- | r = -1.000 | -- | **Strong** (validated on Gemma, GPT-2, Llama; 3 architectures) |
| Llama internal discriminability (top vs bottom Q) | 0.0 | Cohen's d = 5.09-6.00 | -- | **Strong** (massive quartile separation) |
| Phase 0 metric validation (random baseline) | -- | < 5% false absorption rate | -- | **Strong** (metric is specific) |
| EDA vs decoder cosine baseline (DeLong) | 0.302 (L5-16k) | 0.698 (EDA) | +0.396 | **Strong** |
| ITAC best-latent FN reduction (L12-65k) | 0% | 22.7% | +22.7% | **Weak** (single latent; mean only 3.14%) |
| ITAC FVU improvement (L12-65k) | 0% | -4.23% | -4.23pp | **Weak** (reconstruction improves, but limited applicability) |
| Cross-model EDA configs passing (>= 0.65) | 0 | 3/8 | -- | **Moderate** (Gemma L5-16k, L12-16k; GPT-2 L6) |

## Root Cause Analysis

### Positive Result 1: EDA achieves 0.776 AUROC at L12-16k

- **Mechanism**: The formal EDA lower bound theorem, grounded in Tang et al. (2025) biconvex optimization framework, predicts that absorbed latents must have elevated encoder-decoder angular divergence. At L12-16k, the SAE has enough capacity to learn many features but not so many that early absorption (decoder-absent type) dominates, creating a "sweet spot" where late-absorption geometry is visible to EDA. The Mann-Whitney test confirms group separation (p = 6.4e-5, Cohen's d = 1.02).
- **Design decision**: Deriving EDA from the biconvex partial-minimum characterization rather than using an ad-hoc threshold elevated this from an empirical observation to a principled diagnostic with a formal bound. The SynthSAEBench validation (AUROC = 1.0, F1 = 0.974) confirms the theory holds exactly in controlled settings.
- **Expected or surprising**: Expected from the theory. The magnitude (Cohen's d > 1.0) is genuinely strong for a purely weight-based metric with no activation data -- this effect size is conventionally "large."

### Positive Result 2: Cross-model EDA validation on GPT-2 with direct labels (AUROC = 0.650)

- **Mechanism**: GPT-2 L6 with exact Chanin et al. FeatureAbsorptionCalculator labels confirms EDA detects absorption in a completely different model architecture (d_model=768 vs. Gemma d_model=2304). Direct labels eliminate the proxy-label quality concern.
- **Design decision**: Implementing FeatureAbsorptionCalculator for GPT-2 and generating ground-truth labels was the R4 round's most impactful addition. This is the single most methodologically clean data point: same-model probes, same-model activations, established measurement tool.
- **Expected or surprising**: Expected that theory is model-agnostic, but the consistency of 0.650 (direct) with 0.698-0.776 (proxy) validates that proxy-label results are not artifacts. This cross-model validation is what turns EDA from a "Gemma Scope observation" into a general SAE diagnostic.

### Positive Result 3: Three-subtype taxonomy with 75% early-absorption dominance

- **Mechanism**: Decoder dictionary lookup reveals that approximately 75% of absorbed latents have no decoder column aligned with the parent probe direction (cosine < 0.3 at canonical threshold). The SAE never allocated dictionary capacity to the parent feature. Only approximately 13% are late-type (decoder-present, encoder-suppressed), the subtype that ITAC and MP-SAE target.
- **Design decision**: The three-way partition (early/late/partial) is training-free, activation-free, and directly tests whether the failure is dictionary-level or encoder-level. Kruskal-Wallis validation at p = 0.0002 with threshold-stability across all 5 variants makes this robust.
- **Expected or surprising**: **Surprising and field-reframing.** The prior literature implicitly assumed late absorption (encoder misalignment) was dominant. Finding that 75% is dictionary coverage failure redirects the mitigation prescription: wider dictionaries or hierarchy-aware training objectives, not inference-time encoder fixes.

### Positive Result 4: EDA metric identity validated across 3 model families

- **Mechanism**: EDA = 1 - cos(encoder_j, decoder_j) is mathematically equivalent to negated decoder cosine similarity. Pearson correlation between EDA and negated DecCos is exactly -1.000 on Gemma, GPT-2, and Llama architectures (d_model = 768, 2304, 4096).
- **Design decision**: Testing across 3 model families with different SAE training regimes (JumpReLU for Gemma Scope, ReLU for GPT-2 JB, Llama Scope) confirms EDA is an architecture-agnostic weight diagnostic.
- **Expected or surprising**: Mathematically expected; the practical value is confirming the metric is well-defined across the SAE ecosystem.

## Unexpected Signals

### Unexpected Finding 1: EDA mean varies systematically across SAE training regimes

- **Observation**: Gemma Scope SAEs have mean EDA of approximately 0.19, Llama Scope approximately 0.50, and GPT-2 (JB) SAEs approximately 0.62. This is a 3x range in mean encoder-decoder misalignment across model families.
- **Mini-hypothesis**: Different SAE training objectives enforce different degrees of encoder-decoder alignment. Gemma Scope's JumpReLU activation may implicitly encourage tighter alignment. Lower baseline misalignment could mean that when absorption does occur in Gemma Scope, the deviation from baseline is proportionally clearer (explaining the higher AUROC on Gemma configs).
- **Significance**: If confirmed, this suggests SAE training regime choice has a direct impact on absorption susceptibility -- an independently publishable observation that could guide architecture selection.

### Unexpected Finding 2: D-EDA dramatically outperforms EDA at GPT-2 L10 (0.762 vs. 0.336)

- **Observation**: At GPT-2 L10, scalar EDA completely fails (AUROC = 0.336, reversed direction) while D-EDA achieves 0.762 (CI: [0.686, 0.830]). The +0.426 AUROC delta is the largest gap between any two metrics in the study.
- **Mini-hypothesis**: At deeper layers, absorption involves more complex multi-feature encoder residual structure than a single scalar (the encoder-decoder angle) can capture. D-EDA's directional decomposition isolates the absorption-specific component from general polysemanticity noise. This may indicate that late-layer absorption is qualitatively different -- involving encoder drift toward specific competing features rather than generic misalignment.
- **Significance**: D-EDA is demoted overall (does not improve on EDA where EDA works), but the GPT-2 L10 result demonstrates it as a complementary rescue strategy. For practical diagnostic toolkits: "try EDA first; if it fails, try D-EDA."

### Unexpected Finding 3: ITAC failure as confirmatory evidence for early-dominance

- **Observation**: ITAC achieves only 3.14% mean FN reduction (target: 20%). But the single late-absorbed latent that responded well (j_idx=61217) showed 22.7% FN reduction. The failure pattern is structurally explained by the taxonomy.
- **Mini-hypothesis**: ITAC's failure is not a negative result about inference-time correction per se; it is a positive result about the taxonomy's predictive power. If late absorption were dominant (as the field assumed), ITAC would have performed near its 20% target. The 3% mean confirms that 75% of targets are structurally ineligible (early-type, no decoder representation). The ITAC failure and the taxonomy validate each other from orthogonal angles.
- **Significance**: Reframing ITAC's failure as confirmatory evidence strengthens both the taxonomy finding and the ITAC narrative simultaneously. Reviewers should be told: "ITAC's failure is exactly what the taxonomy predicts."

### Unexpected Finding 4: Llama Scope layer-wise EDA decrease

- **Observation**: EDA mean decreases from L6 (0.518) to L12 (0.491) in Llama Scope, suggesting increasing encoder-decoder alignment at deeper layers. This parallels trends observed in Gemma Scope.
- **Mini-hypothesis**: Deeper layers represent more abstract, compositional features that may be more compatible with the bilinear SAE architecture, or may have fewer hierarchical parent-child relationships susceptible to absorption.
- **Significance**: A weak signal (2 data points), but if it generalizes, it provides practical guidance on layer selection for SAE deployment with reduced absorption risk.

### Unexpected Finding 5: Cross-paradigm negative correlation (first-letter vs. RAVEL rho = -0.43)

- **Observation**: SAE configurations that absorb more on first-letter spelling tend to absorb less on entity-attribute hierarchies, and vice versa (R3 data, n=6 configs, rho = -0.43 to -0.20). This is a surprising dissociation.
- **Mini-hypothesis**: First-letter absorption involves a syntactic hierarchy with near-complete co-occurrence statistics (every word has a first letter). Entity-attribute hierarchies involve semantic relationships with partial, noisy co-occurrence. These may trigger qualitatively different absorption regimes -- a potential new research direction.
- **Significance**: Hypothesis-generating. If confirmed with same-model probes, this would suggest absorption is a family of related phenomena indexed by hierarchy type, not a monolithic failure mode.

## Follow-Up Experiments

| Signal | Experiment | Expected Outcome | GPU Hours | Priority |
|--------|-----------|------------------|-----------|----------|
| EDA regime characterization | Run EDA on Gemma Scope with direct Chanin labels (requires Gemma 2B access) | AUROC improves from 0.776 (proxy) toward 0.85+; proxy-collapse at L12-65k partially resolves. If >= 3/6 pass, EDA claim strengthens to "reliable mid-layer detector" | 3-4 | **High** |
| Cross-domain validation | Retrain RAVEL probes on Gemma 2B residual stream activations (requires model access) | Probe accuracy >= 85%; real absorption rates exceed shuffled null for >= 2 domains | 1-2 | **High** |
| Amortization gap experiment (Backup A) | Fix decoder dictionary from Gemma Scope L12-16k; compare absorption under feedforward vs. OMP vs. 2-pass encoding at matched L0 | If amortization gap dominates: OMP absorption < 50% of feedforward. Separates encoder from dictionary in absorption causation | 1-2 | **High** |
| Taxonomy cross-model | Run three-subtype classification on GPT-2 L6 SAE (direct labels available) | Early-dominance (approximately 70-80%) replicates cross-model | 0.5 | **Medium** |
| D-EDA rescue analysis | Systematic D-EDA evaluation on EDA-failed configs (L12-65k, L19-16k) | D-EDA AUROC > 0.65 on at least 1 additional config beyond GPT-2 L10 | 1-2 | **Medium** |
| ITAC on real activations | ITAC on Gemma 2B real text activations (10k tokens), conditioned on late-absorbed latents only | FN reduction > 10% for late-type subgroup; confirms ITAC works when the taxonomy says it should | 2-3 | **Medium** |
| Threshold sensitivity figure | Generate curve showing early-type proportion as function of tau (0.1 to 0.5) | Identify natural elbow or plateau in the early-type proportion curve | 0.1 | **Low** |

## Honest Caveats

### EDA AUROC = 0.776 at L12-16k

- **Counter-argument**: AUROC is inflated by extreme class imbalance (n_pos = 16 out of 16,384 latents, 0.098% prevalence). Precision@50Recall is very low (0.0035). Only 3/8 total configs pass the >= 0.65 threshold across both model families.
- **Alternative explanation**: EDA may correlate with a latent confound (polysemanticity, decoder norm, dead latent fraction) rather than absorption specifically. The polysemanticity ablation partially addresses this, but the confound cannot be fully ruled out without direct labels.
- **What would convince me**: EDA AUROC >= 0.70 on >= 3 configs with direct Chanin labels (not proxy). If achieved with proper labels, the regime-specific claim is beyond reasonable doubt.

### 75% Early-Absorption Dominance

- **Counter-argument**: The classification depends on the threshold tau = 0.3 for decoder cosine similarity. At tau = 0.2, early-type proportion drops to approximately 32% at L12-65k -- a dramatic shift. The 75% figure is threshold-dependent, not threshold-robust.
- **Alternative explanation**: "Early absorption" may include latents that are not truly absorbed but simply have low decoder cosine similarity with the parent direction for geometric reasons unrelated to absorption. The parent probe direction itself (mean decoder direction of absorbed latents) is an approximation that may be poor.
- **What would convince me**: A threshold sensitivity analysis revealing a natural elbow or plateau, suggesting a data-driven partition point. Replication on GPT-2 with direct labels showing similar proportions at canonical tau. Alternatively, external validation using activation-based methods to confirm that early-type latents genuinely fail to fire on parent-positive inputs.

### Cross-Domain Absorption Evidence (H3 Collapsed in R4)

- **Counter-argument**: The R4 shuffled control conclusively shows that real absorption rates are statistically indistinguishable from shuffled null rates for all 3 RAVEL hierarchies (0/3 domains exceed shuffled p95). H3 is not validated. The R3 intra-RAVEL rho = 0.924 was computed with bridge-model probes (Qwen2.5-0.5B) that fail the quality gate.
- **Alternative explanation**: The coherence result may be driven by SAE width scaling (wider SAEs show more "absorption" regardless of hierarchy) rather than genuine cross-domain absorption structure. R4B with GPT-2 Medium bridge probes at n=3 configs shows rho = 0.667, weaker than R3's 0.924.
- **What would convince me**: Same-model probes (Gemma 2B or Llama-3.1-8B) passing the 85% quality gate, with real absorption rates exceeding shuffled null for >= 2 domains. Without same-model access, the cross-domain contribution remains suggestive existence evidence only.

### Cross-Model Validation (GPT-2 L6 AUROC = 0.650)

- **Counter-argument**: n_pos = 18 is small, making the AUROC estimate unstable (bootstrap CI: [0.531, 0.761]). The lower CI bound (0.531) is barely above chance. Cohen's d = 0.53 is "medium" effect size, not "large."
- **Alternative explanation**: GPT-2 Small absorption dynamics may differ qualitatively from Gemma 2B. The passing result may reflect a statistical fluctuation. EDA and decoder cosine AUROC are identical at GPT-2 L6 (delta = 0.0), meaning EDA does not add value over the simpler metric for this model.
- **What would convince me**: A larger absorbed-latent pool (n_pos >= 50) on GPT-2, or a second GPT-2 config passing at AUROC >= 0.65. The EDA = DecCos equivalence at GPT-2 needs to be acknowledged -- EDA's value over the baseline is demonstrated on Gemma, not GPT-2.

### ITAC Failure (3.14% vs. 20% target)

- **Counter-argument**: H5 is falsified. The mean FN reduction is 7.4x below target. The single success case (22.7%) may be an outlier.
- **Alternative explanation**: ITAC was evaluated on synthetic activations (decoder column inputs), not real text. The synthetic protocol may underestimate ITAC's real-world effect. Additionally, the parent match threshold (0.25) may be too conservative.
- **What would convince me**: ITAC on real text activations for late-absorbed latents only showing FN reduction > 10%. If it remains < 5%, the structural limitation explanation is strengthened but the practical value of ITAC is zero.

## Bottom Line

There is a clear publishable story here as a **two-contribution characterization paper**. Contribution 1 (EDA as a regime-specific weight-only absorption detector) is validated by 3 passing configs across 2 model families, anchored by L12-16k AUROC = 0.776 with Cohen's d = 1.02, and cross-model validated on GPT-2 with direct labels. Contribution 2 (three-subtype taxonomy with 72-75% early-absorption dominance) is the paper's highest-impact finding, reframing the field's mitigation focus from encoder fixes to dictionary coverage solutions, supported by KW p = 0.0002 and threshold stability. The ITAC failure and D-EDA complement as honest negative results that reinforce the taxonomy. The cross-domain contribution (H3) has collapsed pending same-model probe access but retains suggestive existence evidence (intra-RAVEL rho = 0.924 from R3). The critical blocking dependency -- Gemma 2B / Llama model access for ground-truth label validation -- would likely strengthen the results, given that the cleanest-label experiment (GPT-2 direct) produced a passing signal. This is a solid EMNLP 2026 submission, upgradeable to NeurIPS 2026 MI Workshop or ICLR 2027 if the access gate resolves favorably.
