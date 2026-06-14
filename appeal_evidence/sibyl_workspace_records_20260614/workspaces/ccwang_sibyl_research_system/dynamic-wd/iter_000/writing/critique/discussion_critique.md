# Discussion & Conclusion Critique

## Score: 7/10

## Strengths

- The four practical recommendations are concrete and actionable, directly tied to empirical numbers from the paper (e.g., the smooth optimum landscape across three lambda values, the CWD degradation percentages). This is unusually good for a discussion section.
- The framing of limitations is honest and specific. Limitation 3 (single seed) explicitly confronts the weakest methodological point and provides a thoughtful partial mitigation argument. Very few papers acknowledge this so directly.
- The conclusion paragraph successfully synthesizes the three core findings into a coherent mechanistic narrative, and the "negative result yields three positive insights" rhetorical frame is effective and appropriate for NeurIPS.
- The "Implications for Weight Decay Theory" subsection adds genuine intellectual value by articulating the gap between the theory's explanatory power and its prescriptive power---this is a nuanced and honest characterization that elevates the paper above a simple negative result.
- The budget-matching diagnostic (Recommendation 4) is a genuinely useful contribution that practitioners can immediately apply without reading the theory.
- Numerical consistency across sections is excellent: the figures cited in the conclusion ($92.54\% = 92.54\%$, $10.00\%$ collapse, $92.06\%$ vs $92.05\%$) match the abstract and intro exactly.

---

## Issues (by priority)

### Critical

- **The abstract and intro describe the LR--WD decoupling collapse as "from $84.49\%$ to $10.00\%$" (abstract) and "from $92.05\%$ to $10.00\%$" (intro), but the conclusion cites "from $92.05\%$ to $10.00\%$."** The abstract's "$84.49\%$" is inconsistent with the intro and conclusion values of "$92.05\%$." This numerical inconsistency in a headline result is a critical credibility issue that reviewers will immediately flag. One of these numbers is wrong, and it must be resolved before submission. If $84.49\%$ refers to some intermediate or averaged quantity, that distinction must be made explicit. If it is a typo, it should be corrected throughout.

- **The "Implications for Weight Decay Theory" subsection introduces a claim not properly established in the analysis section:** the statement that "the budget equivalence principle suggests that weight decay theory should shift focus from instantaneous rates to cumulative effects $\sum_t \lambda_t$" is a significant theoretical repositioning. However, the analysis section's formal argument for budget equivalence relies on a small-$\lambda$ approximation and a hand-waving claim that "trajectory-dependent terms are also approximately matched." This is not a tight theoretical result. Stating in the discussion that theory "should shift focus" based on an empirical finding supported only by a heuristic approximation overstates the theoretical grounding. Either the formal budget equivalence needs a tighter proof (acknowledged as a future direction---which is good---but this makes the claim in the discussion premature), or the language must be softened to "suggests" and the empirical basis made more explicit.

### Major

- **Recommendation 2 in the practical guidelines is framed too broadly.** It warns against "any adaptive weight decay scheme that does not scale $\lambda_t$ with $\gamma_t$." However, AdamW explicitly decouples weight decay from the learning rate and is widely successful. The paper itself acknowledges (in the limitations and in the analysis section) that AdamW's design works precisely because $\lambda$ is constant and tuned jointly with the LR schedule. The recommendation as written contradicts the AdamW literature if read literally. It needs a qualifier: "in SGD with milestone LR schedules" or "when $\lambda_t$ is time-varying." Failing to add this qualifier will confuse practitioners and invite reviewer objections.

- **The future directions section is underdeveloped relative to the intellectual ambitions of the paper.** Direction 3 (formal proof of budget equivalence) is the most important theoretical gap but receives only two sentences. At NeurIPS, reviewers expect future directions to sketch at minimum the mathematical challenge or a candidate proof strategy. A statement like "establishing tight conditions under which $\sum_t \lambda_t$ is a sufficient statistic" is generic---it does not hint at what tools (e.g., stability analysis, PAC-Bayes bounds, uniform stability) would be appropriate. Direction 1 (when alignment becomes actionable) similarly reads as a restatement of the limitation rather than a genuine research direction.

- **The conclusion paragraph contains a subtle over-claim.** The phrase "constant weight decay remains the optimal strategy for nonconvex SGD" is too strong. The experiments establish that constant WD is sufficient and that the tested dynamic alternatives provide no benefit. Optimality (in the formal sense) has not been proven. The paper's own analysis section is explicit that the budget equivalence is empirical, not proven. The claim should be "remains an empirically undominated strategy" or "provides no identified disadvantage under the tested conditions."

- **Limitation 1 (scale) does not discuss transformers substantively.** The paper title includes "nonconvex SGD" generically, and the intro claims relevance to "a broad range of architectures." Yet transformers now dominate the NeurIPS audience's practical concerns, and the limitation section dispenses with them in one sentence. The specific concern for transformers is worth stating: batch normalization plays a key role in the LR--WD coupling analysis (BN's weight norms have different dynamics), and transformers typically use AdamW with layer norm---so the entire coupling analysis may simply not apply. Acknowledging this more precisely would strengthen credibility.

- **The alignment signal analysis in the "Implications" subsection introduces a quantitative claim ("$\delta_t$ varies by only $1.6\times$") that is cited as a reason the time-varying bound improvement is "negligible," but no calculation is provided.** How much does the $\sup_t \delta_t$ vs. $\bar{\delta}_T$ difference translate into in terms of convergence bound improvement? Without at least an order-of-magnitude estimate, "negligible" is an assertion rather than a demonstrated result. Readers interested in theory will want to see the arithmetic.

### Minor

- The phrasing "This negative result yields three positive mechanistic insights" in the conclusion repeats verbatim the framing from the introduction ("three negative but informative results"). Slight rephrasing in the conclusion would avoid the impression that the authors are simply re-summarizing the abstract. The conclusion should advance the narrative, not just restate it.

- The transition from the "Implications for Weight Decay Theory" subsection to "Limitations" is abrupt. There is no bridging sentence that acknowledges the scope conditions under which the implications hold. A single sentence connecting the theoretical implications to the experimental scope (CIFAR-scale, SGD, milestone schedule) would improve flow and reduce the risk that readers over-generalize the theoretical takeaways.

- The CWD instability finding appears in the practical recommendations (item 3) but is absent from the "Limitations" section and the "Conclusion" paragraph. Because CWD is a published method (cited as Liu et al. 2025), the paper's claim of "consistent late-training instability" is a secondary contribution that carries some responsibility for characterization. The conclusion's silence on CWD after the intro explicitly lists it as an "additional contribution" creates an inconsistency in what the paper claims to have established.

- Recommendation 1 cites "$\lambda = 5 \times 10^{-4}$ (for SGD with momentum on CIFAR-scale tasks) is sufficient." The parenthetical scope qualifier is buried inside the recommendation. Given that the paper is about nonconvex SGD generally, this qualifier should be made more prominent to avoid practitioners misapplying the advice to, e.g., ResNet-50 on ImageNet where different values are standard.

---

## Specific Suggestions

1. **Fix the abstract/intro numerical inconsistency immediately.** Audit all instances of the LR--WD collapse numbers ($84.49\%$ vs. $92.05\%$ as the starting accuracy). Determine which is correct and propagate it uniformly. If both are correct but refer to different experimental variants, add an explicit distinguishing label.

2. **Soften the theoretical repositioning claim.** Change "the budget equivalence principle suggests that weight decay theory *should shift focus*" to "the budget equivalence principle *provides empirical motivation* for weight decay theory to examine cumulative effects $\sum_t \lambda_t$, pending formal characterization." This appropriately hedges the claim given its empirical rather than formal basis.

3. **Add scope qualifier to Recommendation 2.** Revise to: "Avoid decoupling a *time-varying* weight decay schedule from the learning rate in custom schedules, particularly under milestone or step LR schedules with SGD. Note that AdamW's constant decoupled weight decay is not subject to this concern, as the coupling interaction only arises when $\lambda_t$ itself varies."

4. **Expand Future Direction 3.** Add 3--4 sentences outlining a candidate proof strategy. For example: "A natural approach is to show that when $\lambda_t \ll 1$, the weight norm trajectory $\|w_t\|$ depends on $\{\lambda_t\}$ only through $\sum_t \lambda_t$ up to $O(\lambda_{\max}^2 T)$ corrections, using the stability analysis of \citet{hardt2016train} or PAC-Bayes bounds that track weight norm explicitly."

5. **Add a brief quantitative estimate in the "Implications" discussion.** For the claim that the $\sup_t \delta_t$ vs. $\bar{\delta}_T$ improvement is "negligible," add a calculation: if $\delta_t$ ranges $[0.0028, 0.0045]$ and the convergence bound scales as $O(\delta_T^2)$ (check the exact dependence in Xie et al.), then the improvement is at most $(0.0045/0.0028)^2 - 1 \approx 60\%$---or state the actual formula and plug in numbers.

6. **Mention CWD briefly in the conclusion.** Add one sentence acknowledging the CWD instability as a secondary finding: "Additionally, we document systematic late-training instability in Cautious Weight Decay across all tested settings, suggesting that coordinate-wise adaptive regularization introduces optimization risks under standard CIFAR benchmarks."

7. **Revise the over-claiming conclusion statement.** Change "constant weight decay remains the optimal strategy" to "constant weight decay remains an empirically undominated strategy under the tested conditions." This is more defensible and no less impactful.

8. **Add one sentence bridging the transformer gap.** In Limitation 1, add: "In particular, transformer architectures trained with AdamW and layer normalization may exhibit qualitatively different weight norm dynamics, and the LR--WD coupling analysis developed here for SGD with batch normalization may not transfer without modification."
