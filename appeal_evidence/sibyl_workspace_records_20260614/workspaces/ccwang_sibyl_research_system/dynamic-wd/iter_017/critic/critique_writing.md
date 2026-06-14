# Writing Critique -- Iteration 017

## Overall Assessment

The writing quality is substantially improved from pre-pivot iterations. The paper reads as a focused, honest method paper with commendable statistical calibration. The pivot from the "unified framework" to a focused EqWD method paper was the right decision and is well-executed.

## Strengths

1. **Statistical calibration is exceptional.** The paper consistently uses "directional trend" rather than "statistically significant" when bootstrap CIs include zero. Cohen's d effect sizes are reported alongside point estimates. The n=3 limitation is explicitly acknowledged in every relevant context. This is above community norm and will be appreciated by reviewers.

2. **Self-aware limitations section.** Section 5.6 honestly lists four major limitations including the effective WD inflation confound, optimizer scope (SGDW only), architecture scope (CNNs only), and training duration. This is uncommon and strengthens credibility.

3. **CAWD negative result is well-leveraged.** Section 5.4 turns CAWD's failure into a contribution: cosine alignment is redundant given gradient and weight norms. This is a genuinely useful finding presented with appropriate nuance.

4. **Clean algorithm presentation.** Algorithm 1 is concise, correct, and well-annotated with design rationale for each component.

## Issues

### Critical

1. **ImageNet epoch count not stated in main text.** The cross-dataset figure title says "45 epochs" but Section 4.1 never states the number. Readers scanning the text (not the figures) will assume 90 epochs. One-line fix: add "trained for 45 epochs" to Section 4.1.

2. **Abstract says "seven weight decay methods" ambiguously.** Does this count NoWD? Is EqWD included in the seven? The counting is unclear and will confuse reviewers on first read.

### Major

3. **Beta=5.0 narrative creates internal contradiction.** The paper simultaneously claims (a) beta=1.0 is a "robust default" and (b) beta=5.0 achieves 1.02% higher accuracy on CIFAR-100. These cannot both be true. The paper should either present beta as task-dependent (abandoning the "robust default" claim) or validate beta=5.0 with multiple seeds.

4. **"Empirical Observation" framing is improved but still slightly misleading.** The former Proposition 2 is now labeled "Empirical Observation (Ratio informativeness)." The rationale still reads: "If alignment deviation [...] is approximately determined by the gradient and weight norms [...] then the ratio captures the same information." This remains a conditional tautology. The improvement is that it is no longer labeled a formal proposition, but the text still presents it as a theoretical contribution rather than simply stating: "We conjecture that norms are sufficient for alignment information and test this empirically via AIS."

5. **CIFAR-10 section (Section 5.7) is a liability.** The table contains "DefazioCorrective" (undefined), lacks EqWD, and the analysis ("confirms our theoretical analysis") overreaches -- this section should either be completed with EqWD results or removed entirely.

### Minor

6. **WD heatmap caption overstates.** "Concentrates stronger regularization" suggests dramatic modulation, but the actual range is 0.00052-0.00062, a ~20% variation. Rewrite to match the visual evidence.

7. **CAWD should appear in Table 1.** Currently only in Table 2, making its first appearance confusing.

8. **Several bibliography entries use "and others" instead of et al.** This is sloppy and will be noticed. The Defazio arXiv ID is a placeholder (2501.00001).

9. **The "scope of equilibrium analysis" paragraph (Section 3.1) mentions "cosine learning rate schedules" but the actual experiments use step decay.** This creates a minor inconsistency -- the scope discussion should mention step decay as the practical deviation from the theoretical constant-LR assumption.
