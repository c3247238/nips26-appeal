# Result Debate: Strategist Perspective

## Publication Target Assessment

### Primary Target: NeurIPS 2025 / ICLR 2026 (Interpretability Track)
**Fit: MODERATE-HIGH**

The result_debate package — encoder_norm as detection metric, H2 falsification, co-occurrence analysis — fits the interpretability track well. NeurIPS reviewers in this area will care about:
1. Novelty vs. Chanin et al. (2024) and Karvonen et al. (2025)
2. Whether the theoretical contribution is rigorous enough
3. Breadth of evaluation (single task: first-letter spelling)

### Secondary Target: ICML 2026 Mechanistic Interpretability Workshop
**Fit: HIGH**

Workshop format is better suited to the current experimental scale. Lower bar for n_pos, more tolerance for methodology notes. Strategic path: workshop submission now → venue submission after expanding to more features and tasks.

## Positioning Strategy

### Against Chanin et al. (2024)
Chanin et al. introduced the first-letter task and the IG-based absorption detection pipeline. Our contribution: **weight-only detection** (encoder_norm requires no activation data) vs. their **activation-based detection**. This is a genuine improvement in practical applicability — useful for SAE auditing at scale where running IG probes on millions of features is expensive.

### Against Karvonen et al. (2025 SAEBench)
SAEBench provides absorption rates as a benchmark metric. Our contribution: **mechanistic explanation** of why absorbed features are detectable (encoder competition pressure), not just detection. We explain SAEBench's metric rather than just replicating it.

### Against iter_001 EDA Paper
The current paper supersedes iter_001 by showing encoder_norm > EDA and providing the H2 falsification. The paper should be positioned as an extension/improvement, not a parallel contribution.

## Recommended Paper Title Options

1. **"Encoder Norms Detect Feature Absorption in Sparse Autoencoders"** — direct, specific
2. **"When Sparsity Eats Its Own Features: Detecting and Explaining Absorption in SAEs"** — narrative, broader
3. **"Absorption in Sparse Autoencoders is Primarily a Sparsity Landscape Problem, Not an Amortization Gap"** — H2 negative as primary contribution

**Recommendation: Option 3.** The H2 falsification is the most novel contribution and distinguishes the paper from all prior work. Detection (encoder_norm) supports this narrative but the mechanistic finding is the primary contribution.

## Risk Assessment

| Risk | Probability | Mitigation |
|------|-------------|------------|
| n=18 AUROC instability | HIGH | Add power analysis; present CI prominently |
| Hook confound in A3 | MODERATE | Add limitation; run matched hook experiment |
| H3 probe methodology rejection | MODERATE | Frame as "preliminary" with explicit limitation |
| EDA vs encoder_norm theoretical gap | MODERATE | Write formal bound for encoder_norm under absorption model |
| Reviewer unfamiliar with SAE absorption | LOW | Include background section with iter_001 citation |

## Strategic Decision

**ADVANCE with focused writing that leads with H2 falsification.** The paper narrative should be:
1. Background: What is absorption? Why does it matter?
2. EDA as baseline (iter_001, briefed here for context)
3. Encoder norm as improved detector (A1-A3)
4. Co-occurrence as independent signal (B1-B2)  
5. **H2 falsification as primary mechanistic finding (C1-C2): sparsity landscape, not amortization**
6. F1 as practical implication: wider SAEs partially help but don't solve the root cause
7. Discussion: what training-time interventions follow from this?

This narrative is clear, complete, and publishable at ICLR 2026 interpretability track.
