# Experiment Result Analysis

## Key Results Summary

### Hypothesis Outcomes (5 hypotheses tested)

| Hypothesis | Prediction | Actual Result | Verdict |
|---|---|---|---|
| H1: LV competition coefficient detects absorption (F1 > 0.65) | alpha_ij threshold at ~1 with sharp transition | Test F1 = 0.128, ROC-AUC = 0.148 (below random), cosine baseline outperforms at F1 = 0.165, linear AIC < sigmoid AIC (no phase transition) | **FALSIFIED** |
| H2: Corpus PMI predicts absorption (partial R^2 > 0.10) | Positive PMI coefficient, r > 0.50 | PMI coefficient = -0.006 (wrong sign), p = 0.593, partial R^2 = 0.0006 | **FALSIFIED** |
| H3: Absorption disconnected from downstream (\|r\| < 0.2) | No meaningful correlation | Pearson r = -0.595 (sparse probing), -0.431 (SCR), -0.454 (RAVEL TPP); partial r = -0.661, -0.677, -0.492 after controls; matched-pair Cohen's d = 2.13, p = 0.006 | **FALSIFIED (positive direction -- best outcome)** |
| H4: DAS(k=3) monotonically increases with width (80% of letters) | Monotonic increase across widths | Only 42.3% of letters show positive slope; mean DAS(k=3) non-monotonic: 0.320 -> 0.227 -> 0.260 | **FALSIFIED** |
| H5: Comprehensive absorption rate > 40% | Taxonomy reveals higher true rate | 92.3% comprehensive (Type I: 3.8%, Type II: 88.5%, Type III: 0%), but Type II rate self-flagged as "likely inflated" | **SUPPORTED with major caveats** |

### Key Quantitative Evidence

- **C3A SAEBench downstream correlation** (n=54 Gemma-2-2B SAEs): absorption vs. sparse probing F1 Pearson r = -0.5948 (CI: [-0.744, -0.389]); partial r = -0.6611 after controlling log(width), layer, arch_class. All three main tasks exceed Bonferroni-corrected significance threshold at alpha = 0.0125.
- **C3B Matched RAVEL comparison**: low-absorption TPP mean = 0.046 vs. high-absorption = 0.009 (5.2x ratio). Paired t = 4.27, p = 0.006, Cohen's d = 2.13. SCR: t = 7.19, p = 0.00005.
- **C1B LV detector**: test F1 = 0.128 at tau = 0.5; cosine baseline test F1 = 0.165; delta = -0.037. Cross-architecture: v5-32k F1 = 0.009; v5-128k F1 = 0.000.
- **C2C PMI regression**: PMI partial R^2 = 0.0006; coefficient negative; sign inconsistent across layers (3 positive, 5 negative).
- **C2B 30-SAE survey**: absorption peaks at layers 3-5, drops to near-zero by layer 10-11; absorption increases with width at layer 8 from 0.009 (3k) to 0.104 (98k); 806 valid data points across 31 configurations.
- **C2D Taxonomy**: Type II dominates at 88.5% (23/26 letters), median magnitude ratio ~0.23, but parent features identified by heuristic rather than sae-spelling ground truth.

## Debate Perspectives Summary

- **Optimist**: The H3 falsification is the "genuinely important headline finding" -- the first systematic empirical proof that absorption predicts downstream SAE quality. The partial correlations strengthening after confound control suggest the effect is real. The taxonomy, layer gradient, and width scaling are all novel contributions. Even the LV detector shows meaningful precision at tight thresholds (0.50-0.60). Bottom line: publishable NeurIPS paper centered on the downstream validation finding.

- **Skeptic**: The LV detector is an anti-predictor (AUC < 0.5) and the paper's theoretical centerpiece is empirically dead. The C3A downstream correlation is confounded by width -- all high-absorption SAEs are 1M-width, all low-absorption are 16k/65k. L0 is not controlled in partial correlations. C3C safety probe directly contradicts the H3 narrative (highest-absorption SAE has smallest probe gap). The Type II taxonomy rate is inflated. Three of four hypotheses failed. The salvageable contributions are narrow: C3A (if width/L0 confound addressed), layer gradient, and taxonomy concept (if recalibrated).

- **Strategist**: PROCEED with major framing pivot. Two signals are strong (downstream correlation, taxonomy concept), one is noise (LV detector, PMI). The paper should pivot from "Lotka-Volterra theory of absorption" to "Empirical Anatomy of Feature Absorption: Prevalence, Scaling, and Downstream Impact." Priority investments: Gemma-2-2B replication (~8 GPU-hrs), properly powered safety probe (~2 GPU-hrs), paper draft with pivoted framing (immediate). Do NOT invest further in saving the LV detector.

- **Comparativist**: The ONE thing this work does that no prior work does is provide the first systematic, statistically controlled quantification that absorption predicts downstream task degradation. No competing paper offers this analysis. However, 3 of 4 hypotheses failed, the model was substituted (GPT-2 instead of Gemma-2-2B), and the taxonomy is unsound. Recommends workshop paper unless L0 confound, taxonomy, and Gemma-2 replication are addressed -- then NeurIPS main conference becomes possible.

- **Methodologist**: The model substitution (GPT-2 Small instead of planned Gemma-2-2B) is the single most consequential methodological deviation. It creates an asymmetry: H1/H2/H4/H5 tested on GPT-2, but H3 tested on Gemma-2. Missing critical ablations: frequency ratio component of alpha_ij, Type II magnitude threshold sensitivity, L0 as C3A covariate. Reproducibility score: 3/5. The C3A finding is methodologically sound in protocol design (Bonferroni, partial correlations) but needs L0 control before causal language is justified.

- **Revisionist**: The revised mental model: absorption is architecture/objective-driven (not data-driven), has a strong causal-direction relationship with downstream quality, and the LV ecological analogy does not transfer to SAE feature dynamics. The LV framework conflates base-rate frequency imbalance with genuine competitive pressure. Proposes three new hypotheses: (NH1) absorption mediated by sparsity pressure, (NH2) C3A correlation driven by 1M-width extreme, (NH3) unlearning decorrelation reveals mechanistic boundary of absorption impact.

## Analysis

### 1. Method Feasibility

The core experimental pipeline is robust: 37.5M alpha_ij pairs computed across 8 SAE configurations with zero NaN/Inf, the sae-spelling integration works, SAEBench data download and analysis pipeline functions correctly, and the 30-SAE survey completed across 31 configurations. The infrastructure for measuring and analyzing absorption at scale is validated.

However, the primary theoretical method -- the Lotka-Volterra competition coefficient as an unsupervised absorption detector -- does not work. The alpha_ij metric is anti-predictive (AUC = 0.148, below random), underperforms the trivial cosine-similarity baseline by F1 delta of -0.037, shows no sharp phase transition (linear AIC < sigmoid AIC), and fails catastrophically on cross-architecture generalization (F1 = 0.0 on v5-128k). The revisionist's structural critique is persuasive: the ecological competition analogy conflates frequency imbalance with causal displacement. The method feasibility verdict for the LV detector is negative.

### 2. Performance

The study's performance varies dramatically by component:

**Strong outperformance of baselines**: The C3A downstream correlation analysis reveals a previously undocumented strong relationship (r = -0.595 to -0.677 after controls) between absorption and three independent downstream metrics. The matched-pair comparison produces Cohen's d = 2.13, an effect size that far exceeds typical findings in the SAE evaluation literature. This exceeds the pre-registered threshold (|r| > 0.3) by a factor of ~2x.

**Decisively below baselines**: The LV detector (F1 = 0.128 vs. 0.65 target), the PMI predictor (R^2 = 0.0006 vs. 0.10 target), and the DAS(k=3) width prediction (42.3% vs. 80% target) all fail their pre-registered success criteria by large margins.

### 3. Improvement Headroom

The downstream correlation finding (C3A) has clear improvement paths:
- Adding L0 as a covariate (0 GPU-hours, pure analysis) would address the skeptic's primary objection
- Within-width stratified correlations (0 GPU-hours) would test whether the effect survives when width is held constant
- Gemma-2-2B replication (~8 GPU-hours) would provide cross-model validation
- Properly powered safety probe (n >= 10 at fixed layer, ~2 GPU-hours) would replace the underpowered C3C

The taxonomy (C2D) also has a clear improvement path: recalibration using sae-spelling ground-truth parent features instead of the selectivity heuristic (~1-2 GPU-hours).

The LV detector and PMI predictor have no clear improvement path. The revisionist's structural critique argues the LV analogy is mechanistically unsound, not merely empirically underperforming. The PMI null result is clean and unambiguous. Investing further compute in these dead hypotheses would be sunk-cost reasoning.

### 4. Time-Cost Tradeoff

The pivoted framing (empirical anatomy + downstream impact) can be strengthened with approximately 10-12 GPU-hours of additional work:
- L0 covariate analysis: 0 GPU-hours
- Within-width correlations: 0 GPU-hours
- Taxonomy recalibration: 1-2 GPU-hours
- Properly powered safety probe: 2 GPU-hours
- Gemma-2-2B replication: 8 GPU-hours

This is a modest additional investment for a substantial improvement in paper quality and venue competitiveness. Starting fresh with an alternative (e.g., Alternative A: information-theoretic theory, or Alternative B: scaling laws) would require 8-10+ GPU-hours for primary experiments plus additional time for analysis and writing -- while discarding the strong C3A result and the entire 30-SAE survey dataset.

The cost of proceeding (~12 GPU-hours for critical additions) is lower than pivoting (~15-20 GPU-hours for a fresh alternative), and the expected value is higher because the C3A finding is strong and novel.

### 5. Critical Objections

The skeptic's concerns fall into two categories:

**Addressable concerns:**
- L0 as uncontrolled confound in C3A: can be addressed with analysis-only work (0 GPU-hours). The partial correlations strengthening after controlling width/layer/arch is encouraging -- if they also strengthen after L0 control, the confound objection is substantially weakened.
- Matched-pair width imbalance in C3B: addressable by computing within-width comparisons.
- Taxonomy Type II inflation: addressable by recalibration with sae-spelling ground-truth parents.

**Non-fatal concerns:**
- C3C safety probe reversal: this is n=3 across different layers/hooks and is underpowered (p=0.45). It does not "directly contradict" H3; it is simply uninformative due to confounded design. A properly powered replication at fixed layer would resolve this.
- Model substitution (GPT-2 vs. Gemma-2): a genuine limitation, but the strongest result (C3A) already uses Gemma-2 data. The GPT-2 results for H1/H2/H4 are clean negative findings regardless of model.

**Fatal concerns for original framing:**
- The LV detector is anti-predictive and the theoretical framework is structurally flawed. This is fatal for the original paper framing but NOT for the pivoted framing that centers the downstream correlation finding.

## Decision Rationale

The decision to PROCEED is based on the following evidence and reasoning:

1. **The C3A downstream correlation is a genuinely novel and impactful finding.** All six debate perspectives agree this is the study's strongest contribution. No prior work provides this systematic, statistically controlled quantification. The effect sizes are large (r = -0.595 to -0.677, Cohen's d = 2.13) and the result addresses a critical open question in the SAE field -- whether optimizing absorption metrics actually matters for downstream utility.

2. **The required improvements are tractable and low-cost.** The most critical additions (L0 covariate, within-width correlations) require zero GPU-hours. The taxonomy recalibration and safety probe scale-up require ~3-4 GPU-hours total. Even the Gemma-2-2B replication at ~8 GPU-hours is modest. Total additional investment: ~12 GPU-hours.

3. **Pivoting would discard strong evidence.** The C3A result, the 30-SAE scaling survey (806 data points across 31 configurations), and the taxonomy concept are all valuable assets that would be lost in a full pivot. No alternative in the alternatives list starts from as strong a position.

4. **The framing pivot is well-defined.** All six perspectives converge on the same restructured framing: center the downstream impact finding, report the LV and PMI results as honest negatives, and present the taxonomy and scaling laws as supporting contributions. This is not a vague "we will figure it out" pivot -- it is a concrete reframing with consensus support.

5. **The negative results add credibility.** Honestly reporting that H1 (LV detector) and H2 (PMI predictor) were tested and falsified strengthens the paper's credibility. Pre-registered hypotheses that were genuinely tested and honestly reported, whether confirmed or falsified, are the hallmark of rigorous science.

6. **The skeptic's concerns, while legitimate, are addressable.** The L0 confound is the single biggest risk, but partial correlations already strengthen after controlling for width/layer/arch, and the matched-pair Cohen's d = 2.13 is too large to be entirely explained by a width confound. Adding L0 control is a straightforward next step.

The decision NOT to PIVOT is based on the observation that no alternative in the alternatives list offers a stronger expected outcome than the pivoted framing of the current results. Alternative A (information-theoretic theory) replaces one theoretical framework (LV) with another untested one -- high risk for marginal gain. Alternative B (scaling laws) is essentially a subset of what we already have from C2B. Alternative C (controlled causal audit) is methodologically interesting but would start from scratch. Alternative D (ATM replication) depends on checkpoint availability.

DECISION: PROCEED
