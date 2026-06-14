# Methodology: Activation Energy Theory Pilot

## Research Goal
Validate Activation Energy Theory on LLM reasoning by testing whether answer accuracy follows Arrhenius kinetics (exponential saturation) and whether activation energy (Ea) correlates with problem difficulty.

## Model & Dataset
- **Model**: Qwen/Qwen2.5-Math-7B-Instruct
- **Dataset**: MATH test set subset (200 problems)
- **Hardware**: RTX PRO 6000 Blackwell (sm_120) via PyTorch 2.11.0

## Experiment Design

### Group G0: Baseline (k=1)
- Single-pass evaluation on MATH 200
- Establish P_∞ baseline for the model
- Measure token counts for cost estimation

### Group G1: Saturation Curve (k=1,2,4,8,16)
- For each problem, sample k times with temperature=0.7
- Track: accuracy@1, accuracy@k, answer consistency, token counts
- Fit Arrhenius model: P_k = P_∞ × (1 - exp(-k/k₀))
- Extract per-problem k₀ (characteristic samples)

### Group G2: Consistency Analysis
- Track answer consistency trajectory across k samples
- Estimate Ea from consistency convergence rate
- Compare consistency-based Ea to saturation-derived k₀

### Group G3: Ea-Based Routing (Validation)
- Use Ea threshold to classify problems as "easy" vs "hard"
- Validate single-pass accuracy on easy problems >75%

## Hypotheses & Metrics

| Hypothesis | Metric | Threshold |
|------------|--------|-----------|
| H1: Arrhenius kinetics | Exponential fit R² | >0.85 |
| H2: Ea = difficulty | Spearman(Ea, level) | >0.4 |
| H3: Single-pass threshold | Accuracy (low Ea) | >75% |
| H5: Consistency = Ea | Spearman(consistency Ea, actual Ea) | >0.5 |

## Falsification Criteria
- If power-law fits better (lower AIC/BIC), H1 falsified
- If Ea-level correlation <0.2, H2 falsified
- If uniform accuracy across Ea levels, H3 falsified

## Pilot Configuration
- **Samples**: 100 (config), 200 problems × 16 k values = 3200 samples
- **Timeout**: 900s per group
- **Seed**: 42
- **Estimated time**: 60-90 minutes total

## Inference Configuration
- Temperature: 0.7 (balanced diversity/repeatability)
- Max tokens: 1024 (sufficient for math reasoning)
- Batch size: auto (VRAM-optimized)

## Expected Visualizations
1. Saturation curves (P_k vs k) with exponential fit
2. Ea distribution by MATH difficulty level (box plots)
3. ROC curve for single-pass threshold
4. Correlation scatter plots (Ea vs level, consistency vs actual Ea)

## Comparison Baselines
- Majority voting (k samples, pick most common answer)
- Self-consistency (Wang et al. 2022)
- RASC (Li et al. 2026)
