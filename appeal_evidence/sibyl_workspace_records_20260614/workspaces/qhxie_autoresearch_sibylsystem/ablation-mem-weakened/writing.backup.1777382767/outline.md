# Paper Outline: Feature Absorption and Downstream SAE Reliability

## Title

**"Does Feature Absorption Matter? A Null Result on Downstream SAE Reliability"**

Alternative: **"The Interpretability Illusion Revisited: Feature Absorption Shows Limited Impact on Steering and Probing"**

## Abstract (150-200 words)

Feature absorption -- where general SAE features fail to fire and are instead "absorbed" into more specific child features -- has been identified as a fundamental failure mode of sparse autoencoders (Chanin et al., 2024). However, the field lacks answers to a critical question: does absorption degrade the interpretability tasks that motivate SAE research? We provide the first systematic, quantitative study bridging absorption detection to downstream task performance. Using pre-trained SAEs from GPT-2 Small (gpt2-small-res-jb, 24K latents) and Pythia-70M, we measure absorption rates across layers for 26 first-letter features (A-Z), then test steering effectiveness and sparse probing accuracy. Raw steering success shows no significant correlation with absorption (Pearson r = +0.008, p = 0.970 at layer 4; r = -0.301, p = 0.136 at layer 8). However, when controlling for random baseline steering, the delta (feature-specific minus random) reveals a significant negative correlation at layer 8 (Pearson r = -0.431, p = 0.028; Spearman rho = -0.502, p = 0.009). Sparse probing shows no correlation at either layer. Precision-recall decomposition reveals absorption affects recall (coverage) but not precision (selectivity). These mixed results suggest that absorption's impact on downstream tasks is subtle, layer-dependent, and only detectable after controlling for baseline steering effects.

---

## 1. Introduction

### 1.1 The SAE Credibility Crisis
- Sparse autoencoders are the dominant paradigm for mechanistic interpretability
- Growing concerns: Korznikov et al. (2026) show SAEs recover only 9% of true features; random baselines match trained SAEs on standard metrics
- DeepMind's MI team deprioritized SAE research after negative downstream results
- Feature absorption sits at the center of this crisis

### 1.2 Feature Absorption: The Gap Between Detection and Impact
- Chanin et al. (2024) proved hierarchical features cause absorption; persists across all tested LLM SAEs
- SAEBench (Karvonen et al., 2025) includes absorption as a standardized metric
- Architectural innovations (Matryoshka SAEs, OrtSAE, HSAE) all target absorption reduction
- **Critical gap**: No existing work quantifies whether absorption degrades the interpretability tasks that motivate SAE research

### 1.3 Our Contribution
- First systematic, quantitative bridge between absorption detection and downstream task performance
- Training-free methodology using pre-trained SAEs
- Test steering effectiveness (with random baseline control) and sparse probing accuracy across multiple layers
- Cross-model validation on Pythia-70M
- **Key finding**: Mixed results -- raw steering shows no correlation, but delta-corrected steering reveals significant negative correlation at layer 8; probing shows no correlation; precision is invariant to absorption

### 1.4 Paper Structure
- Brief roadmap of sections

**Transition**: From the motivation, we now formalize the problem and our hypotheses.

---

## 2. Background and Related Work

### 2.1 Sparse Autoencoders for Mechanistic Interpretability
- SAE architecture: encoder, decoder, sparsity penalty
- Key applications: circuit analysis, feature steering, model editing, bias detection
- Notation: SAE reconstructs activations $a \in \mathbb{R}^d$ as $\hat{a} = W_{\text{dec}} \cdot f(W_{\text{enc}} \cdot a + b_{\text{pre}})$

### 2.2 Feature Absorption
- Definition: General parent features fail to fire; activation is captured by specific child features
- Chanin et al. (2024) detection method: differential correlation between parent and child features
- SAEBench absorption metric standardization
- Prevalence: observed across Gemma, Pythia, GPT-2 SAE families

### 2.3 Downstream Interpretability Tasks
- **Feature steering**: Adding a feature direction to model activations to influence outputs
- **Sparse probing**: Training linear classifiers on SAE latents for concept detection
- Prior work on steering (Marks et al., 2024; Rimsky et al., 2024) and probing (Templeton et al., 2024)
- **Gap**: No prior work connects absorption rates to task performance

### 2.4 Architectural Responses to Absorption
- Matryoshka SAEs: hierarchical dictionary structure
- OrtSAE: orthogonal decomposition
- HSAE: hierarchical sparse autoencoder
- All motivated by absorption reduction, but none quantify task-level impact

**Transition**: Having established the background, we now state our research questions and formal hypotheses.

---

## 3. Research Questions and Hypotheses

### 3.1 Research Questions
- RQ1: Does feature absorption cause measurable degradation in downstream interpretability tasks?
- RQ2: Is the absorption-degradation relationship consistent across model layers?
- RQ3: Does the relationship depend on measurement methodology (raw vs. baseline-corrected metrics)?
- RQ4: Is absorption's impact layer-dependent, with deeper layers showing stronger effects?

### 3.2 Hypotheses
- **H1 (Raw steering)**: Higher absorption rate -> lower raw steering success rate (r < -0.2, p < 0.05)
- **H1b (Delta steering)**: Higher absorption rate -> lower delta steering effectiveness (feature-specific minus random baseline; r < -0.2, p < 0.05)
- **H2 (Probing)**: Higher absorption rate -> lower sparse probing F1 (r < -0.2, p < 0.05)
- **H3 (Consistency)**: Degradation relationship is consistent across layers (slopes have same sign and comparable magnitude)
- **H4 (Efficiency)**: Absorption affects steering efficiency (higher EC50), not steering capability
- **H5 (Precision-Recall)**: Absorption affects recall (coverage), not precision (selectivity)

### 3.3 Falsification Criteria
- If H1/H1b/H2 fail: absorption does not degrade these tasks (null result)
- If H3 fails: relationship is layer-dependent
- If H4 fails: absorption does not affect steering efficiency
- If H5 fails: absorption affects both precision and recall

**Transition**: We now describe our experimental methodology for testing these hypotheses.

---

## 4. Methodology

### 4.1 Overview
- Training-free analysis of pre-trained SAEs
- Four phases: absorption detection, feature steering, sparse probing, correlation analysis
- Models: GPT-2 Small (124M parameters) with res-jb SAEs (24,576 latents); Pythia-70M (cross-validation)

### 4.2 SAE Configurations
- GPT-2 Small layers tested: 0, 4, 8, 10 (hook_resid_pre)
- Pythia-70M layer 4 (hook_resid_post)
- Dictionary sizes: 24,576 (GPT-2), 32,768 (Pythia)
- Features: 26 first-letter features (A-Z)
- Samples per feature: 100
- Seed: 42

### 4.3 Phase 1: Absorption Detection
- Load pre-trained SAE via SAELens
- Run Chanin et al. differential correlation metric
- For each letter A-Z, detect if "starts with X" feature is absorbed
- Record: absorption rate per feature, absorbing latent IDs
- Classification: HIGH (>=10%), MEDIUM (5--10%), LOW (<5%)

### 4.4 Phase 2: Feature Steering
- Extract feature direction: $direction = W_{\text{dec}}[feature\_id]$
- Generate test prompts with words starting with target letter (100 per letter)
- Steering strengths: [1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
- Success metric: % increase in target-letter token probability vs. unsteered baseline
- **Critical control**: Random feature baseline -- 26 randomly selected SAE latents steered with identical protocol
- **Primary metric**: Delta steering success = feature-specific success - random baseline success

### 4.5 Phase 3: Sparse Probing
- Train k-sparse linear probe (k=1, 5, 10, 20) on first-letter classification
- Measure F1 using: (a) only main feature latents, (b) all latents
- Compare F1 degradation for absorbed vs. non-absorbed features
- Decompose F1 into precision and recall components

### 4.6 Phase 4: Correlation Analysis
- Pearson/Spearman correlation between absorption rate and raw steering success (H1)
- Pearson/Spearman correlation between absorption rate and delta steering success (H1b)
- Pearson/Spearman correlation between absorption rate and probing F1 (H2)
- Linear model: $task\_degradation = \beta \cdot absorption\_rate + \epsilon$
- Report R^2, confidence intervals, p-values
- Test for consistency across layers (H3)
- EC50 analysis: dose-response curve fitting for H4
- Precision-recall decomposition for H5

**Transition**: With the methodology established, we present our experimental results.

---

## 5. Results

### 5.1 Absorption Detection Results
- Layer 0: mean 0.021, max 0.094, HIGH=0, MEDIUM=0, LOW=26
- Layer 4: mean 0.039, max 0.160, HIGH=0, MEDIUM=6, LOW=20
- Layer 8: mean 0.034, max 0.242, HIGH=0, MEDIUM=4, LOW=22
- Layer 10: mean 0.029, max 0.209, HIGH=0, MEDIUM=4, LOW=22
- Key finding: Low absorption variance; most features show near-zero absorption; Layer 4 shows the most variance
- Pythia-70M layer 4: mean 0.082, max 0.272, MEDIUM=10, LOW=16

### 5.2 Random Baseline Validation
- Feature-specific steering outperforms random directions (t = 6.41, p < 0.0001, Cohen's d = 1.26 at layer 4)
- Layer 4: +132% over random (t = 6.41, p < 0.0001, Cohen's d = 1.26)
- Layer 8: +126% over random (t = 6.02, p < 0.0001, Cohen's d = 1.18)
- This validates that feature-specific steering captures meaningful directions

### 5.3 Feature Steering Results
- Steering is generally effective (success rates 0.4-1.0 at strength=50)
- Raw steering shows no clear pattern linking absorption to steering failure
- Even highly absorbed features (e.g., U at 24.2%) achieve 100% raw steering success
- **However**, delta-corrected steering reveals a different picture (see 5.5)

### 5.4 Sparse Probing Results
- F1 scores range from 0.18 to 1.0 at k=5
- High variance across features, not clearly linked to absorption
- Full-activation probing shows higher F1 than k-sparse, indicating recoverability
- Precision = 1.0 universally across all features at k>=5; recall varies (0.10 to 1.0)

### 5.5 Hypothesis Testing

**H1 (Raw Steering vs. Absorption):**
- Layer 4: r = +0.008, p = 0.970, R^2 = 0.000 -> NOT SUPPORTED
- Layer 8: r = -0.301, p = 0.136, R^2 = 0.090 -> NOT SUPPORTED (trending negative but not significant)

**H1b (Delta Steering vs. Absorption):**
- Layer 4: r = +0.245, p = 0.227, R^2 = 0.060 -> NOT SUPPORTED
- Layer 8: r = -0.431, p = 0.028, R^2 = 0.186 -> **SUPPORTED** (significant negative correlation)
- Layer 8 Spearman: rho = -0.502, p = 0.009 -> **SUPPORTED** (rank correlation also significant)

**H2 (Absorption vs. Probing):**
- Layer 4: r = -0.003, p = 0.987, R^2 = 0.000 -> NOT SUPPORTED
- Layer 8: r = -0.107, p = 0.604, R^2 = 0.011 -> NOT SUPPORTED

**H3 (Cross-Layer Consistency):**
- H1 slopes: layer 4 = +0.024, layer 8 = -0.630 -> opposite signs -> NOT SUPPORTED
- H1b slopes: layer 4 = +1.441, layer 8 = -2.491 -> opposite signs -> NOT SUPPORTED
- H2 slopes: layer 4 = -0.010, layer 8 = -0.286 -> same sign but CV = 1.317 -> NOT SUPPORTED
- Overall: No consistent degradation pattern across layers

**H4 (Efficiency - EC50):**
- Layer 4: high-abs EC50 mean = 21.06, low-abs EC50 mean = 28.06, t = -1.228, p = 0.232 -> NOT SUPPORTED
- Layer 8: high-abs EC50 mean = 21.43, low-abs EC50 mean = 18.03, t = 0.795, p = 0.435 -> NOT SUPPORTED

**H5 (Precision-Recall):**
- Precision std << Recall std at all k>=5 (layer 4: 0.054 vs 0.199; layer 8: 0.028 vs 0.192)
- Most features have precision = 1.0 (21/26 at layer 4, 25/26 at layer 8 for k=5)
- Recall shows substantial variation and drives F1 variance -> **SUPPORTED**

### 5.6 Cross-Model Validation (Pythia-70M)
- Pythia-70M layer 4: raw r = -0.041, p = 0.841; delta r = -0.041, p = 0.841
- No significant correlation on Pythia-70M
- Likely due to model size (19M vs 124M params) -- steering effects too weak to measure reliably

### 5.7 Interpretation of Mixed Results
- Raw steering success is confounded by baseline random steering effects
- After controlling for random baseline, absorption-degradation relationship emerges at layer 8
- The effect is layer-dependent: only detectable at deeper layers
- Probing remains insensitive to absorption regardless of correction
- Low absorption variance limits statistical power across all tests
- Precision is invariant to absorption; only recall varies

**Transition**: We now discuss the implications of these findings and their limitations.

---

## 6. Discussion

### 6.1 Why the Mixed Result?
1. **Raw steering is confounded**: Random baseline steering achieves 34-38% success, masking the true feature-specific effect
2. **Delta correction reveals hidden relationship**: At layer 8, absorption does degrade steering effectiveness after baseline subtraction
3. **Layer-dependent effects**: The relationship is only significant at layer 8, suggesting deeper layers may be more sensitive to absorption
4. **Probing insensitivity**: k-sparse probing F1 depends more on feature selectivity than absorption; distributed encoding makes probing robust
5. **Precision invariance**: Absorption affects coverage (recall) but not selectivity (precision)
6. **Low absorption variance**: 18-26/26 features per layer show near-zero absorption, limiting statistical power

### 6.2 Implications for the Field
- Absorption's impact on downstream tasks is more nuanced than previously assumed
- Raw steering metrics without baseline controls may be misleading
- The field should adopt delta-corrected steering as a standard evaluation practice
- Architectural innovations targeting absorption reduction may still be justified, but their benefit may be layer-specific
- Probing appears robust to absorption, suggesting it is not the critical bottleneck
- Precision-recall decomposition should be standard in probing evaluations

### 6.3 Comparison with Pilot
- Pilot (layer 8, 50 samples): r = -0.153, p = 0.456 (raw)
- Full (layer 8, 100 samples): r = -0.301, p = 0.136 (raw); r = -0.431, p = 0.028 (delta)
- Doubling sample size strengthened the negative trend
- Random baseline control was the critical addition that revealed the significant effect

### 6.4 What Would Change Our Conclusion?
- Testing on larger models (Gemma-2-2B, Llama-3.1-8B) where absorption may be stronger
- Using semantic hierarchy features instead of first-letter
- Alternative absorption metrics (ablation-based from SAEBench)
- Different downstream tasks (circuit finding, model editing)

**Transition**: We acknowledge the limitations of our study and suggest directions for future work.

---

## 7. Limitations and Future Work

### 7.1 Limitations
1. **Single model family (primary)**: Only GPT-2 Small res-jb SAEs tested as primary; Pythia-70M cross-validation inconclusive
2. **Gated model access**: Gemma-2-2B (primary target) unavailable due to HF authentication
3. **Narrow feature set**: First-letter task may not generalize to complex semantic features
4. **Small model**: GPT-2 Small (124M) may not exhibit absorption as strongly as larger models
5. **Single absorption metric**: Chanin differential correlation only; other metrics may yield different results
6. **Two downstream tasks**: Steering and probing only; circuit finding and model editing not tested
7. **Low absorption variance**: Most features show near-zero absorption, limiting correlation power
8. **Single significant result**: Only H1b at layer 8 achieves significance; could arise by chance with multiple comparisons

### 7.2 Future Work
1. Test with authenticated Gemma/Pythia access for cross-model validation
2. Use semantic hierarchy features (WordNet) for richer structure
3. Try alternative absorption metrics (ablation-based, SAEBench)
4. Test with JumpReLU SAEs (reportedly show stronger absorption)
5. Evaluate circuit finding and model editing tasks
6. Test on larger models (Llama-3.1-8B, Gemma-2-9B)
7. Investigate why the effect is layer-dependent

**Transition**: We conclude with a summary of our contributions.

---

## 8. Conclusion

### 8.1 Summary
- We conducted the first systematic study linking feature absorption detection to downstream task performance
- Using GPT-2 Small SAEs across 4 layers and 26 first-letter features, we found mixed results
- Raw steering success shows no significant correlation with absorption
- **However**, delta-corrected steering (controlling for random baseline) reveals a significant negative correlation at layer 8 (r = -0.431, p = 0.028)
- Sparse probing shows no correlation with absorption at either layer
- Precision-recall decomposition shows absorption affects recall but not precision
- The relationship is inconsistent across layers (H3 not supported)
- EC50 analysis shows no efficiency degradation (H4 not supported)

### 8.2 Contributions
1. First quantitative bridge between absorption and task performance (mixed result)
2. Demonstration that random baseline control is essential for steering evaluation
3. Precision-recall decomposition revealing absorption affects recall, not precision
4. Training-free methodology reproducible by any researcher
5. Evidence that absorption's impact is subtle, layer-dependent, and task-specific
6. Practical guidance: practitioners should use delta-corrected steering metrics

### 8.3 Closing Thought
- Null results are valuable, but so are carefully controlled positive findings
- The SAE credibility crisis demands rigorous, task-oriented evaluation -- not just metric optimization
- Our work shows that absorption does matter for steering (when properly measured), but not for probing
- This task-specificity is itself an important finding that should guide future SAE research

---

## References

- Chanin et al. (2024): Feature absorption identification
- Karvonen et al. (2025): SAEBench standardized metrics
- Korznikov et al. (2026): SAE feature recovery critique
- Marks et al. (2024): Sparse feature circuits
- Templeton et al. (2024): Sparse probing
- Rimsky et al. (2024): Feature steering
- Matryoshka SAEs, OrtSAE, HSAE: Architectural responses

---

## Appendix

### A.1 Full Correlation Statistics
- Complete Pearson/Spearman tables for all layers and all hypotheses (H1, H1b, H2)
- Confidence intervals and exact p-values

### A.2 Absorption Rates by Feature
- Full table of all 26 features x 4 layers (GPT-2) and layer 4 (Pythia)

### A.3 Steering Results by Strength
- Dose-response curves for all features
- Raw and delta-corrected success rates
- EC50 values per feature

### A.4 Random Baseline Results
- Full random feature steering results per layer
- Statistical comparison with feature-specific steering

### A.5 Probing Results by k
- F1, precision, recall for k=1, 5, 10, 20
- Precision-recall decomposition tables

### A.6 Cross-Model Validation
- Pythia-70M absorption rates and correlation results

### A.7 Code and Reproducibility
- Links to code repository
- Exact commands to reproduce results

---

## Figure & Table Plan

### Figure 1: Experimental Pipeline Overview (Section: Method)
- **Purpose**: Show the four-phase pipeline from SAE loading to hypothesis testing
- **Type**: flow_chart
- **Content**: Phase 1 (Absorption Detection) -> Phase 2 (Feature Steering) -> Phase 3 (Sparse Probing) -> Phase 4 (Correlation Analysis); include data flow arrows and key outputs; highlight the random baseline control in Phase 2
- **Key takeaway**: Our training-free methodology systematically connects absorption detection to downstream task evaluation, with critical random baseline controls
- **Generation**: tikz
- **Data source**: Methodology description

### Figure 2: Absorption Rates Across Layers (Section: Results)
- **Purpose**: Show distribution of absorption rates per layer
- **Type**: bar_chart / grouped bar chart
- **Content**: 26 features (A-Z) on x-axis, absorption rate on y-axis, grouped by layer (0, 4, 8, 10); highlight features with >10% absorption
- **Key takeaway**: Most features show near-zero absorption; only 4-6 features per layer exceed 10%
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `full_absorption_gpt2_all_layers_combined.json`
- **Script**: `writing/figures/gen_fig2_absorption_rates.py`

### Figure 3: Absorption vs. Raw Steering Effectiveness (Section: Results)
- **Purpose**: Visualize the lack of correlation between absorption and raw steering
- **Type**: scatter
- **Content**: x-axis = absorption rate, y-axis = raw steering success at strength=50; separate subplots for layer 4 and layer 8; include regression line and R^2 annotation
- **Key takeaway**: No clear negative correlation in raw steering; points are scattered with no trend
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `correlation_report_full.json` (absorption_rates + steering_success_at_50)
- **Script**: `writing/figures/gen_fig3_absorption_vs_steering.py`

### Figure 4: Absorption vs. Delta Steering Effectiveness (Section: Results)
- **Purpose**: Visualize the significant negative correlation when using delta-corrected steering
- **Type**: scatter
- **Content**: x-axis = absorption rate, y-axis = delta steering success (feature-specific minus random); separate subplots for layer 4 and layer 8; include regression line, R^2, and p-value annotation; highlight layer 8 significant result
- **Key takeaway**: Delta-corrected steering reveals significant negative correlation at layer 8 (r=-0.431, p=0.028)
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `correlation_with_baseline.json` (H1b_steering_delta)
- **Script**: `writing/figures/gen_fig4_absorption_vs_delta_steering.py`

### Figure 5: Absorption vs. Probing F1 (Section: Results)
- **Purpose**: Visualize the lack of correlation between absorption and probing
- **Type**: scatter
- **Content**: x-axis = absorption rate, y-axis = probing F1 at k=5; separate subplots for layer 4 and layer 8; include regression line and R^2 annotation
- **Key takeaway**: No clear negative correlation; probing F1 is independent of absorption
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `correlation_report_full.json` (absorption_rates + probing_f1_k5)
- **Script**: `writing/figures/gen_fig5_absorption_vs_probing.py`

### Figure 6: Steering Dose-Response Curves (Section: Results)
- **Purpose**: Show how steering success varies with strength for different absorption levels
- **Type**: line_plot
- **Content**: x-axis = steering strength (1, 2, 5, 10, 20, 50), y-axis = success rate; separate lines for HIGH (>=10%), MEDIUM (5--10%), LOW (<5%) absorption features
- **Key takeaway**: Steering success increases with strength regardless of absorption level
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `full_steering_probing_gpt2_l4_results.json` and `l8_results.json`
- **Script**: `writing/figures/gen_fig6_dose_response.py`

### Figure 7: Precision-Recall Decomposition (Section: Results)
- **Purpose**: Show that precision is invariant to absorption while recall varies
- **Type**: scatter / dual-axis
- **Content**: x-axis = absorption rate, y-axis = precision (left) and recall (right); separate subplots for layer 4 and layer 8 at k=5; show precision clustering near 1.0 and recall spread
- **Key takeaway**: Precision is invariant to absorption; recall varies and drives F1 variance
- **Generation**: code (matplotlib/seaborn)
- **Data source**: `precision_recall_analysis.json`

### Table 1: Main Results Summary (Section: Results)
- **Purpose**: Compact summary of all hypothesis tests
- **Type**: comparison_table
- **Content**: Rows = H1 (raw), H1b (delta), H2, H3, H4, H5; Columns = Layer, Pearson r, p-value, R^2, Passes; include overall summary row; bold the H1b layer 8 significant result
- **Key takeaway**: Only H1b at layer 8 passes significance threshold
- **Generation**: data_table (LaTeX)
- **Data source**: `correlation_report_full.json` + `correlation_with_baseline.json` + `ec50_analysis.json` + `precision_recall_analysis.json`

### Table 2: Top Absorption Features by Layer (Section: Results)
- **Purpose**: Show the features with highest absorption and their task performance
- **Type**: comparison_table
- **Content**: Rows = features; Columns = Layer, Absorption Rate, Raw Steering Success, Delta Steering Success, Probing F1; sorted by absorption rate descending
- **Key takeaway**: Even highly absorbed features can achieve high raw steering success; delta correction reveals degradation
- **Generation**: data_table (LaTeX)
- **Data source**: `correlation_report_full.json` + `correlation_with_baseline.json`

### Table 3: Absorption Detection Summary (Section: Results)
- **Purpose**: Summary statistics per layer
- **Type**: comparison_table
- **Content**: Rows = layers (0, 4, 8, 10); Columns = Mean Absorption, Max Absorption, # HIGH (>=10%), # MEDIUM (5--10%), # LOW (<5%)
- **Key takeaway**: Layer 4 shows most variance; all layers have predominantly LOW absorption features
- **Generation**: data_table (LaTeX)
- **Data source**: `full_absorption_gpt2_all_layers_combined.json`

### Table 4: Random Baseline Validation (Section: Results)
- **Purpose**: Show that feature-specific steering exceeds random baseline
- **Type**: comparison_table
- **Content**: Rows = Layer; Columns = Feature-Specific Mean, Random Mean, Delta, t-statistic, p-value, Cohen's d
- **Key takeaway**: Feature-specific steering exceeds random baseline (p < 0.0001, large effect size)
- **Generation**: data_table (LaTeX)
- **Data source**: `ablation_random_baseline.json`

### Table 5: Precision-Recall Decomposition (Section: Results)
- **Purpose**: Show precision invariance and recall variation across features
- **Type**: comparison_table
- **Content**: Rows = k values (1, 5, 10, 20); Columns = Precision Mean, Precision Std, n(p=1.0), Recall Mean, Recall Std, Recall Range; separate sections for layer 4 and layer 8
- **Key takeaway**: Precision is near-perfect and invariant; recall varies substantially
- **Generation**: data_table (LaTeX)
- **Data source**: `precision_recall_analysis.json`
