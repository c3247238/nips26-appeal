# Backup Ideas for Potential Pivot

## Alternative 1: Rethinking Feature Absorption -- Decomposing Artifact from Signal

### Summary
If cross-domain absorption rates are uniformly near-zero across all knowledge domains (H1 falsified), pivot to a focused metric validation and decomposition study. Systematically decompose measured absorption into hierarchy-driven, hedging, reconstruction error, threshold sensitivity, and frequency artifact components. Test whether some "absorption" reflects the model's own computational structure via activation patching (the "immune escape test" from the Interdisciplinary perspective: zero the child feature, check if the parent recovers).

### Motivation
Pilot evidence strengthens this alternative: city-country shows 0% absorption, shuffled controls show 59.5% on first-letter, and random controls show 9.2%. These unexpectedly high control rates suggest the metric may be capturing more noise than previously appreciated. The Contrarian perspective's strongest insight -- that absorption measurement needs validation before absorption rates are trusted -- is directly supported by pilot data.

### Key Hypotheses
- H_alt1: After threshold sensitivity sweep, the Chanin metric's absorption rate changes by > 30% (relative) across reasonable threshold ranges (CV > 0.3 across 5x4 grid)
- H_alt2: The immune escape test (zeroing child feature) causes parent recovery in > 60% of identified absorption pairs
- H_alt3: Hierarchy-driven absorption accounts for < 70% of total false negatives at the default L0=82 configuration

### Experiments
1. Threshold sensitivity: 5x4 grid on first-letter + city-continent (the two domains with absorption signal)
2. Immune escape test: For 50+ absorption pairs, zero child activation and measure parent recovery
3. Multi-L0 decomposition profile: Decompose at L0 = {22, 42, 82, 163}
4. Random baseline absorption: Absorption rate for 100 random decoder directions
5. Cross-architecture decomposition: Compare JumpReLU vs Matryoshka (if available)

### Pivot Trigger
Cross-domain absorption < 3% everywhere AND/OR controls not calibratable (shuffled > 30% after all refinements)

### Risk
If decomposition shows hierarchy-driven absorption dominates (> 80%) at all L0 values, H_alt3 is falsified. But the threshold sensitivity and immune escape test remain independently valuable.

---

## Alternative 2: Rate-Distortion Theory of Absorption (Theory-First)

### Summary
If empirical cross-domain experiments produce mostly null results but CMI analysis shows promise, pivot to a theory-focused paper. The core contribution becomes the rate-distortion bound on absorption (Theorem 1 from the Theoretical perspective) plus the lateral inhibition bifurcation analysis (Proposition 2), validated on first-letter where we have ground truth.

### Motivation
The novelty report gives the successive refinement framing 8/10 novelty -- the highest of any individual contribution. The theoretical landscape for rate-distortion applied to SAE absorption is genuinely empty, unlike the crowded impossibility theorem space. A clean theoretical paper with validation on one domain (first-letter) may be stronger than an empirical paper with weak cross-domain results.

### Key Hypotheses
- H_rd1: CMI negatively correlates with absorption rate (rho < -0.3) on first-letter
- H_rd2: The geometric constant c(w_P, w_C) modulates the absorption threshold beyond CMI alone
- H_rd3: Inhibition ratio q > 1 predicts per-token false negatives (precision@50 > 0.3)
- H_rd4: JumpReLU SAEs show more binary absorption patterns than L1 SAEs (bifurcation type prediction)
- H_rd5: CMI threshold predicts the L0 ~ 7-14 phase transition within a factor of 3

### Experiments
1. CMI estimation (25 first-letter features, multiple subspace dimensions)
2. Geometric constant computation from decoder weights
3. Per-token inhibition ratio vs false-negative labels (100k tokens)
4. Architecture bifurcation comparison (JumpReLU vs any available L1)
5. Phase transition prediction validation

### Pivot Trigger
H3 (CMI-absorption correlation) shows promise (rho < -0.3) but cross-domain empirical results are weak (all < 3%). Paper story shifts from "empirical measurement" to "theoretical understanding."

### Risk
CMI estimation may be too noisy in high dimensions even after projection (the Theoretical perspective acknowledges this). The lateral inhibition analysis maps inference dynamics, not training dynamics -- the gap must be argued carefully. But even approximate predictions are informative, and the theoretical framework stands independently of quantitative precision.

---

## Alternative 3: Information-Theoretic Absorption Diagnostic (Focused Paper)

### Summary
A focused, single-idea paper: the Markov chain violation I(X; f_parent | f_child) as a fully unsupervised, information-theoretically grounded diagnostic for absorption susceptibility. Derived from the successive refinement theorem, validated on first-letter, and proposed as a training regularizer.

### Motivation
This is the distilled essence of the highest-novelty contribution. If the full proposal proves too ambitious (6+ hypotheses, 14+ experiments), this alternative provides a clean, focused story: one theoretical idea, one validation, one design principle.

### Key Hypotheses
- CMI discriminates absorbed from non-absorbed features
- CMI correlates with downstream performance degradation
- Matryoshka SAEs have lower Markov chain violation than flat SAEs at matched L0
- Regularizing CMI during training reduces absorption (proposed, not tested -- future work)

### Pivot Trigger
Both cross-domain empirical results and unsupervised detection fail, but the CMI diagnostic shows clear signal on first-letter. Paper becomes "Feature Absorption as Failure of Successive Refinement."

### Risk
Single-domain validation (first-letter only) limits generalizability. But the theoretical contribution (the first connection between successive refinement and SAE absorption) is domain-independent.
