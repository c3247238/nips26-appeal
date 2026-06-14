# Paper Outline: L0-Matched or Misleading? A Systematic Re-evaluation of Architecture Claims for Feature Absorption in Sparse Autoencoders

## Abstract
Re-evaluates SAE architecture claims under L0-matched conditions. Finds that apparent absorption differences may be confounded by sparsity. Falsifies causal link between absorption and downstream interpretability.

## 1. Introduction
- Feature absorption as recognized pathology
- Explosion of mitigation architectures (Matryoshka, OrtSAE, etc.)
- Gap: no study controls for sparsity (L0) when comparing architectures
- Our contribution: L0-matched comparison + causal link test

## 2. Related Work
- Chanin et al. (2024): absorption identification
- Matryoshka SAE, OrtSAE, GBA: mitigation claims
- SAEBench: standardized evaluation
- Gap in existing work: L0 confound

## 3. Method
### 3.1 Experimental Framework
- SynthSAEBench synthetic data with known hierarchies
- Ground-truth absorption measurement
- 5 seeds per variant

### 3.2 Variants
- Baseline L1, TopK, Matryoshka, OrtSAE, Gated, Random

### 3.3 L0-Matching Protocol
- Target L0 = 50 and 200
- Baseline lambda sweep to match variant L0

### 3.4 Dose-Response Design
- Fix architecture, vary lambda
- Measure absorption and downstream MCC

### 3.5 Data Integrity Pipeline
- Cross-file duplicate detection
- Feature count verification
- Output file checks

## 4. Results
### 4.1 Cross-Architecture Comparison (Unmatched L0)
- Table: absorption rates across 7 variants
- Observation: TopK/Matryoshka low, Baseline/Gated high

### 4.2 L0-Matched Comparison
- Table: absorption at matched L0=50 and L0=200
- Key finding: architecture differences [persist / vanish]

### 4.3 Dose-Response Causality
- Figure: absorption vs MCC
- Finding: MCC flat (~0.22) regardless of absorption

### 4.4 Ablation Studies
- Matryoshka flat vs nested: no difference
- OrtSAE with/without penalty: no difference

## 5. Discussion
### 5.1 L0 Confound
- Importance of controlling sparsity
- Implications for existing mitigation claims

### 5.2 Null Causal Result
- Absorption does not predict downstream performance
- Redirects community effort

### 5.3 Limitations
- Synthetic data only
- 1024 features (not 16k)
- MCC may be insensitive

## 6. Conclusion
- Methodological contribution: L0-matching necessity
- Empirical contribution: absorption not causally linked to interpretability
- Practical guidance: control L0 before comparing architectures
