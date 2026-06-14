# Skeptic Analysis: Iteration 9 Full-Mode Consolidation

## 1. Statistical Risk Inventory

### Risk 1: Cross-Domain Absorption Rates Are Confounded by Differential Probe Quality

The paper's primary contribution -- that absorption is hierarchy-dependent (ANOVA p=0.005) -- rests on comparing absorption rates across four hierarchy types with dramatically different probe quality. First-letter probes achieve F1=0.971 at L24, while RAVEL probes range from 0.789 (city-country) to 0.843 (city-continent). The consolidation summary itself acknowledges rho(probe_quality, false_negative_rate) = -0.756 (p<0.001).

This creates a direct confound: lower probe quality inflates false negatives, which the metric counts as absorption. Consider the specific numbers at L24_16k:
- First-letter: F1=0.971, absorption=34.5%, FN=30/87
- City-continent: F1=0.843, absorption=35.8%, FN=62/200
- City-country: F1=0.789, absorption=18.5%, FN=32/189
- City-language: F1=0.823, absorption=13.6%, FN=24/191

The 4/6 significant pairwise comparisons show first-letter *higher* than city-country and city-language. But first-letter has F1=0.971 while city-country/city-language have F1<0.83. The direction of the difference (first-letter higher at L24) could be a genuine finding, or it could reflect that perfect probes produce more true positives in the raw (non-SAE) condition, creating a larger denominator of "correct predictions that could be lost," mechanically inflating the rate. The paper has no partial correlation controlling for probe F1 across hierarchies because the confound is not within-hierarchy -- it IS the between-hierarchy comparison.

**Bottom line:** The ANOVA tests absorption_rate ~ hierarchy, but it should test absorption_rate ~ hierarchy + probe_F1. Without this control, the "cross-domain variation" finding is ambiguous.

### Risk 2: Activation Patching Relies on Non-Standard Word Samples with Low Base Accuracy

The activation patching experiment (n=25 words, 19 with absorption) is the strongest statistical result (Wilcoxon p=0.000218, d=1.33). However, inspecting the word list raises concerns:

- Of 25 tested words, only 7 are "pilot core" English words. The remaining 18 are "discovered_ig" tokens, many of which are non-English fragments: `xfa`, `udy`, `recoge`, `washingtonpost`, `uzu`, `menjadikan`, `wner`, `uestions`, `wikk`, `uki`, `antigos`, `zorgt`, `conmigo`, `unton`, `yaitu`, `membuka`.
- These tokens have extremely low raw accuracy (e.g., `xfa`=0.10, `udy`=0.12, `uzu`=0.06, `wner`=0.06, `uki`=0.08). The model barely knows these tokens' first letters at all. "Absorption" in this context means the SAE disrupts an already fragile representation.
- The highest-recovery words (`conmigo`=100% recovery, `wikk`=66.7%) are extreme outliers from tiny absorbed counts (40 and 12 contexts respectively).
- Probe F1=0.883, below the strict 0.90 gate. The probe itself makes errors, meaning some "recoveries" may be probe fluctuations rather than genuine feature recovery.

The paper claims "first metric-independent causal evidence for competitive exclusion in SAEs." But competitive exclusion should manifest on words where the model confidently knows the first letter (high raw accuracy) and the SAE disrupts this. Testing on `xfa` (10% raw accuracy) and `uzu` (6% raw accuracy) is testing whether removing noisy features from a near-random representation produces stochastic fluctuations that happen to align with probe predictions. The signal may be real for words like `backward` (94% raw, 50% recovery) and `yaitu` (100% raw, 45.5% recovery), but these are a minority.

**Specific concern:** If the analysis is restricted to words with raw accuracy >= 0.50 (where the model actually knows the letter), the sample drops substantially. The paper should report this restricted analysis.

### Risk 3: The Layer-Dependence Finding Involves Comparing Fundamentally Different Regimes

The "novel finding" that absorption is layer-dependent (2.2% at L18 to 34.5% at L24, a "15x variation") compares absorption across layers where probe quality and SAE behavior are qualitatively different. At L6/L12/L18, first-letter probes achieve F1=0.69-0.94 (sklearn), while at L24 they achieve F1=0.97. More critically, the number of "correctly predicted by probe" tokens (the denominator of absorption rate) varies across layers:
- L18_16k: FN=4/183 (absorption=2.2%)
- L24_16k: FN=30/87 (absorption=34.5%)

Note the denominator drops from 183 to 87, meaning far fewer tokens are correctly predicted after SAE encoding at L24. This could reflect genuine absorption or it could reflect that L24 SAEs have worse reconstruction fidelity for first-letter information specifically, producing more errors of all kinds (not just competitive exclusion). The paper does not report overall reconstruction quality (e.g., MSE, loss recovered) per SAE to disentangle these.

## 2. Alternative Explanations

### For the cross-domain absorption variation (H1):

**Alternative: Probe quality determines apparent absorption rates.** First-letter probes (F1=0.97) detect parent features with high precision. RAVEL probes (F1=0.79-0.84) make systematic errors. A RAVEL probe that misclassifies 16% of cities in the raw condition will produce a different false-negative pattern after SAE encoding that looks like "absorption" but is actually "probe error interacting with SAE noise." The ANOVA p=0.005 may be testing probe quality differences, not absorption differences.

### For the activation patching recovery (H7):

**Alternative: Decoder vector arithmetic without competitive exclusion.** When a child feature is zeroed, its decoder vector is subtracted from the reconstruction. If the child's decoder vector is partially anti-aligned with the probe's weight vector for the true letter, removing it mechanically shifts the residual toward the correct probe prediction. This would produce "recovery" without any competitive exclusion mechanism -- it is a geometric artifact of decoder-probe alignment in high-dimensional space. The control (random features) would show less recovery because random decoder vectors are not systematically aligned with letter probes. The paper should report the cosine similarity between child decoder vectors and probe letter vectors to test this.

### For the hedging near-tautology (H3):

**Alternative: Higher-L0 SAEs trivially fix more false negatives.** Going from L0=22 to L0=176 means 8x more features are active. The probability that some activated feature carries the letter information increases combinatorially. The "compensatory resolution" (86.2%) simply reflects that with enough active features, the information is almost certainly somewhere. This is not evidence about the mechanism of absorption at L0=22 -- it is evidence that higher-L0 SAEs are more complete representations. The strict hedging rate of 7.9% compared to shuffled control 3.4% (from prior iteration data) shows a real but small effect.

## 3. Proxy Metric Audit

| Claimed | Actually Measured | Gap |
|---------|------------------|-----|
| "Absorption rates differ across hierarchy types" | False-negative rates after SAE encoding differ across tasks with different probe quality | Cannot separate hierarchy-intrinsic absorption from probe-quality-driven FN variation. The two are aliased because probe quality IS hierarchy-specific. |
| "Causal evidence for competitive exclusion via activation patching" | Zeroing high-IG features changes probe predictions on mostly non-English token fragments | Tests decoder geometry effects on low-confidence predictions, not competitive exclusion on well-understood features. 18/25 words are non-English fragments. |
| "Layer-dependent absorption (novel finding)" | FN rates vary across layers | Could reflect differential SAE reconstruction quality across layers rather than a mechanistic absorption phenomenon. No MSE or loss-recovered data to control for this. |
| "Architecture effect non-significant (p=0.87)" | Kruskal-Wallis on 4x4=16 rates | Severely underpowered: 4 data points per architecture. The non-significant p-value is expected from insufficient sample size, not from a genuine null effect. Cannot claim "hierarchy matters more than architecture" from an underpowered test. |
| "Tightened hedging reveals near-tautology" | 86.2% of FNs at L0=22 are resolved at L0=176 by non-parent features | Measures L0-dependent information availability, not hedging mechanism. Calling this "compensatory" presupposes a mechanism; "resolved by more features at higher L0" is equally valid. |
| "GAS fails as detector" (rho=0.116) | Decoder geometry does not predict FN rate | GAS may correctly measure decoder geometry but decoder geometry may not cause absorption. This is failure of the hypothesis, not failure of the measure. The paper conflates them. |

## 4. Severity Classification

### Fatal Flaws

**F1: No probe-quality control in the cross-domain comparison.** The paper's primary contribution (ANOVA p=0.005 for hierarchy effect) compares tasks where probe quality differs by 0.18 F1 points (0.97 vs. 0.79). The known correlation rho(probe_quality, FN_rate) = -0.756 makes this confound severe. Without demonstrating that the cross-domain absorption differences survive after adjusting for probe quality, the main claim is not established. This is not remediable by post-hoc statistics because probe quality and hierarchy are perfectly confounded (each hierarchy has exactly one probe quality).

**Severity: Fatal** -- undermines the primary contribution. The paper cannot assert cross-domain variation in absorption without ruling out that it is measuring cross-domain variation in probe quality.

### Serious Concerns

**S1: Activation patching word selection bias.** 18/25 words are non-English fragments with <50% raw accuracy. The recovery statistic is dominated by noisy tokens where small perturbations can flip near-random probe predictions. A restricted analysis on words with raw accuracy >= 0.50 is essential. If the effect survives on high-confidence tokens, it is strong evidence. If it vanishes, the d=1.33 is an artifact of noise amplification.

**Severity: Serious** -- weakens but does not necessarily invalidate H7. The Wilcoxon test may survive restriction if the high-confidence subset (backward, yaitu, menjadikan, zorgt, antigos, recoge) shows consistent recovery.

**S2: Architecture comparison is underpowered and confounded by width mismatch.** The Kruskal-Wallis test on 16 data points (4 architectures x 4 hierarchies) has negligible power to detect a moderate effect. Matryoshka width is 32k vs. 16k for JumpReLU/BatchTopK -- not width-matched. The claim "hierarchy type matters more than architecture" is not supported by a non-significant p-value from an underpowered test. At minimum, the paper should compute the effect size and the minimum detectable effect given the sample size, to distinguish "no effect" from "insufficient data."

**Severity: Serious** -- the paper should either drop the architecture comparison claim or present it explicitly as "inconclusive due to insufficient power."

**S3: Cross-domain activation patching was not attempted.** The consolidation acknowledges that no RAVEL probe passes the quality gate for activation patching. This means the causal mechanism (H7) is only demonstrated for first-letter, the simplest and cleanest hierarchy. The paper's central argument is that absorption varies across hierarchies, but the causal evidence applies to only one hierarchy. There is no evidence that the mechanism is the same across hierarchy types.

**Severity: Serious** -- creates an asymmetry: descriptive evidence spans 4 hierarchies, causal evidence covers only 1. The paper should be explicit that the causal mechanism is demonstrated for first-letter only and that cross-domain causal evidence is future work.

**S4: City-continent absorption rates are statistically indistinguishable from first-letter.** Two of the six pairwise comparisons are non-significant (city-continent vs. first-letter: p=0.83 and p=0.93). The paper claims "absorption is hierarchy-dependent" but city-continent is not different from first-letter at L24. Only city-country and city-language are lower. This weakens the "cross-domain" narrative because the strongest semantic hierarchy (city-continent: fewer classes, higher probe F1) shows the same rate as first-letter, while the weaker-probe hierarchies show lower rates -- which is the opposite of what the probe-confound explanation would predict only if you assume lower probe quality should inflate rates. But the absorption rate is computed as FN/(probe-correct-raw), and lower probe quality could also reduce the denominator in complex ways.

**Severity: Serious** -- the finding is really "2 of 3 semantic hierarchies show lower absorption than first-letter at L24," which is a weaker claim than "absorption varies systematically across hierarchy types."

### Minor Caveats

**M1: Hedging decomposition for city-language has N=3.** The city-language hedging analysis has only 3 false negatives, making the strict hedging rate (66.7%) meaninglessly noisy. One additional FN classified differently would swing this to 50% or 75%. The paper should not draw conclusions from N=3.

**M2: The pilot-to-full framing reversal is underexplained.** The proposal (Synthesis Round 5) states "H2 falsified: first-letter is NOT worst case. Semantic hierarchies show HIGHER absorption." The full results (L24) show the exact opposite: first-letter IS highest. The paper pivoted from "semantic > first-letter" to "significant variation exists" without adequately explaining why the L12 pilot finding reversed at L24. The explanation is likely that L12 RAVEL probes were poor (F1=0.69-0.79) while L12 first-letter probes were also poor (F1=0.31), creating unreliable measurements at L12. This further strengthens the probe-quality confound concern.

**M3: "Total wall clock 5 minutes" for 4 completed tasks is suspiciously fast.** The timing data shows phase0_activation_patching_full took 2 minutes (planned 60 minutes). Either the actual computation was done in a prior run and this consolidation is reading cached results, or the "full mode" experiment is doing less work than planned. The paper should verify that activation patching actually ran 200 contexts x 25 words = 5000 forward passes, not that it loaded cached pilot results.

## 5. Concrete Remediation

### For F1 (Probe quality confound in cross-domain comparison):

**Experiment:** Within the first-letter hierarchy (where probe F1=0.97-1.0 across all layers), artificially degrade the probe by adding calibrated noise to probe weights such that effective F1 matches the RAVEL probe quality (e.g., F1=0.84). Measure the "absorption rate" with degraded probes on first-letter data. If absorption rate at F1=0.84 matches the RAVEL cross-domain rates, the probe quality explanation is confirmed. If first-letter absorption remains higher even with degraded probes, the cross-domain variation is genuine.

**Alternative:** Train RAVEL probes using data augmentation, ensemble methods, or larger training sets to push F1 above 0.90 for at least city-continent. If the cross-domain absorption rates change substantially with better probes, the current rates are unreliable.

**Expected outcome:** Given rho=-0.756, I expect degraded probes will substantially alter apparent absorption rates, partially explaining the cross-domain pattern.

### For S1 (Activation patching word selection):

**Analysis:** Re-run the Wilcoxon test restricted to words with raw accuracy >= 0.50 (N approximately equals 10-12 based on the results table: transitive, recoge, menjadikan, backward, antigos, zorgt, conmigo, yaitu, and possibly others). Report effect size and p-value separately for this high-confidence subset.

**Expected outcome:** The subset of 8-10 high-confidence words includes several with large recovery effects (backward=0.50, yaitu=0.455, conmigo=1.0, zorgt=0.50). If Wilcoxon p<0.05 on this restricted set, the causal finding is robust. If p>0.05, the full-sample significance was driven by noisy low-confidence tokens.

### For S2 (Architecture power analysis):

**Analysis:** Compute the minimum detectable effect size given N=16, alpha=0.05. Report this alongside the architecture p=0.87 to contextualize the null result. For example: "With N=16 data points, the Kruskal-Wallis test has 80% power to detect only effect sizes above H=X, corresponding to absorption rate differences of Y percentage points."

**Expected outcome:** The minimum detectable effect will be large (probably > 20pp difference), confirming that the architecture comparison is uninformative rather than providing evidence for the null.

### For S3 (Cross-domain causal evidence):

**Specific action:** In the paper, add an explicit limitation paragraph: "Activation patching confirms competitive exclusion for first-letter only. Whether the same mechanism operates in semantic hierarchies remains untested due to insufficient probe quality. The descriptive cross-domain results demonstrate that absorption rates differ, but the causal mechanism may differ across hierarchy types."

### For M3 (Timing verification):

**Check:** Verify that `phase0_activation_patching_full.json` contains per-word, per-context results for 200 contexts (not just aggregate statistics from a cached pilot). The file is 144KB, which at ~200 bytes per word-context result would accommodate 5000 entries, so the size is plausible. But the "2 minutes for 5000 forward passes on Gemma 2 2B" timing (400 forward passes per second on a single GPU) seems aggressive. Verify that the RTX PRO 6000 Blackwell throughput supports this.
