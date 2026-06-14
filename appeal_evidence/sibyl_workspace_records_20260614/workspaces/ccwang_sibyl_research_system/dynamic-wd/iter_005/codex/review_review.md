# Codex 独立评审 - review

**评审时间**: 2026-03-19 01:07:48
**模型**: Codex (GPT-5.4, reasoning effort: xhigh)

## 评审意见

**Summary**

This manuscript asks a worthwhile question: when do dynamic weight decay (WD) schemes actually outperform a constant decay coefficient, and when are they mostly unnecessary? The paper is clearly written, the empirical motivation is understandable, and I appreciate that the authors are willing to report largely negative results rather than forcing a positive story. The attempt to unify several WD variants under a common "Phi modulator" view is also conceptually tidy.

However, from the perspective of an independent third-party reviewer for a top ML venue, I do not think the paper in its current form meets the bar for NeurIPS/ICML. The central theoretical claims are not yet rigorous enough to support the paper's strongest conclusions, and the empirical evidence is too narrow and too incomplete for the level of generality claimed in the abstract and conclusion. In particular, the paper repeatedly converts "we did not detect a significant improvement over constant WD on small CIFAR-scale BN models" into "constant WD is optimal," which is a much stronger statement than the data justify.

At a high level, the submission currently reads less like a theorem-backed optimization paper and more like a polished, hypothesis-generating position paper with partial experimental support. That is not worthless; in fact, the negative-result framing could be useful. But the paper needs either much stronger theory, or much broader and cleaner evidence, or ideally both.

**Strengths**

First, the paper tackles a real question rather than a synthetic benchmark game. There is genuine value in scrutinizing whether dynamic WD methods add practical benefit beyond constant WD, especially given the proliferation of schedule variants.

Second, the manuscript is unusually transparent about incomplete experiments. Table 6 is honest about the fact that high-rho, matched-rho SGD, NoBN, and low-rho coverage are incomplete. That transparency is better than hiding missing evidence.

Third, some parts of the framing are useful even if not yet publication-ready. Section 3.1's phi(t,theta,g) abstraction is a reasonable taxonomy, and the emphasis on optimizer-state coupling rather than treating WD as just "regularization strength" is directionally sensible.

Fourth, the controlled comparison in Table 1 is at least internally consistent in one respect: all methods share the same base LR and base WD coefficient. That is not sufficient to prove optimality, but it does test a clear plug-and-play question.

**Major Concerns**

1. **The theoretical core is not rigorous enough for the claims being made.**

Theorem 1 in Section 3.3 is presented as an "if and only if" result, but the proof sketch is not remotely sufficient for that level of claim. The sketch says the test loss difference decomposes into an "alignment benefit" proportional to AIS*lambda_bar and a "stability cost" proportional to (C*sigma^2/n)*Delta_CSI, and then concludes the result by comparing the two. That is essentially a verbal decomposition, not a proof. There is no formal stochastic process model, no clearly stated assumptions under which the decomposition holds, no derivation of the constants, and no justification for the claimed necessity and sufficiency.

This problem is amplified by the fact that the key quantities in Theorem 1 are themselves partly heuristic. In Section 3.2, CSI is defined as a weighted sum of three terms with manually chosen weights (0.4,0.3,0.3), and the text admits those weights were chosen because one component seemed "most predictive" in preliminary analysis. If a theorem's threshold depends on a hand-tuned diagnostic metric, then the theorem is not a clean mathematical statement about optimization dynamics.

Likewise, AIS is defined as a Spearman correlation between cos(theta_i,g_i) and Delta_loss_i, but the indexing and measurement protocol for Delta_loss_i are not adequately specified. Is i indexing layers, parameters, batches, or interventions?

Theorem 2 in Section 3.4 is also problematic. The bound is extremely loose by the paper's own admission, but the bigger issue is that some of the mathematical interpretation appears incorrect. The theorem defines CSI_param = max_i CV(lambda_{i,.}), yet the text argues that methods with lambda_min=0 incur "unbounded" per-parameter CSI during off-steps. For a binary {0,lambda} process with nonzero mean, the coefficient of variation is finite, not unbounded. A Bernoulli mask at rate 0.5 has a perfectly finite CV.

Theorem 3 in Section 3.5 is also much weaker than advertised. Even if I grant the PMP derivation, the result is only "optimal" for a surrogate control objective: keeping rho_t close to a target rho*. The manuscript does not prove that minimizing integral(rho_t - rho*)^2 dt is equivalent, or even tightly related, to minimizing test error. So the phrase "PMP-optimal WD" is stronger than the actual derivation warrants.

The additional "independent confirmation" by RG analysis in Remark 3.1 feels especially speculative. The mapping from delta_hat_t to rho_hat_t is hidden inside an undefined function f, and the statement that the two derivations "agree within 15%" is not backed by a figure or derivation in the main paper.

Finally, Proposition 1 in Section 3.6 is essentially intuition dressed as theorem language. Gradients and weights during training are not independent random vectors. There is no precise probabilistic model, no dependence on dimension, and no actual bound yielding "k >= 10" as a design constraint.

2. **The "Phi modulator" framework is only mildly useful and is currently too close to notation wrapping.**

Almost any multiplicative WD schedule can be written as lambda*phi(t,theta,g)*theta; that observation is formally correct, but by itself it is mostly a notation container.

There is also a troubling internal inconsistency: BEM is defined in Section 3.2 as the relative change in effective WD budget compared with constant WD. Under that definition, the "half-lambda" method should have BEM 0.5, not 0.0. Yet Table 4 and Section 5.7 list half-lambda as BEM 0.0. That is not a cosmetic typo; it directly undermines confidence in the diagnostic framework.

More broadly, Section 5.7 admits that within a fixed architecture, CSI and AIS do not predict accuracy. That substantially weakens the claim that the new metrics explain method behavior.

3. **The empirical evidence does not support the paper's strongest claims.**

The abstract and conclusion repeatedly say "constant WD is optimal." But the actual evidence is much narrower. In Table 1, constant WD is not the best method on CIFAR-100 with AdamW; cosine is numerically better. In Table 3, constant WD is not the best method on VGG-16-BN; half-lambda is numerically better.

The incomplete experiments are not peripheral; they are central to the paper's mechanistic narrative. Table 6 shows that the high-rho regime, matched-rho SGD, NoBN, and low-rho AdamW are incomplete or pilot-only -- precisely the settings needed to test whether rho, BN, and optimizer identity drive the claimed phase transition.

Most importantly, PMP-WD is never evaluated. For a top-tier ML paper, introducing a new algorithm and then explicitly deferring all empirical validation is a very serious weakness, close to fatal given that the theory is also not airtight.

4. **The optimizer-vs-rho story is internally inconsistent and empirically confounded.**

The introduction says the "inhibition" regime corresponds to rho < 0.1 with spread < 0.1%. But the paper's own SGD result is at rho ~= 0.005 with spread 0.91%, which directly contradicts that description. The regime narrative needs to be reconsidered.

The SGD-vs-AdamW comparison is heavily confounded. Moving from AdamW to SGD changes the optimizer, effective preconditioning, momentum dynamics, implicit bias, and the realized rho. The matched-rho SGD experiment is not complete.

5. **The statistical methodology is too weak for the claims being made.**

Three seeds are not enough for equivalence-style claims. Failure to reject is not equivalence. SWD's Cohen's d = 0.88 in Table 2 is not a negligible effect size.

The use of **best test accuracy** introduces optimistic bias. The paper should use a validation-selected checkpoint or final checkpoint.

The fairness argument in Section 4.4 is only partially persuasive. Using the same base LR and WD does not test whether constant WD is better than properly tuned dynamic methods.

**Answers to Specific Questions**

a) The theorems are closer to hand-wavy than rigorous. Theorem 1's proof sketch is not sufficient for an iff claim. Theorem 2 contains an apparent mismatch regarding "unbounded" CSI for binary masks. Theorem 3 is optimal for a surrogate objective, not test error. Proposition 1 is intuition rather than theorem-grade analysis.

b) The Phi modulator framework is mostly notation wrapping. It would become genuinely useful if it enabled reliable prediction or a nontrivial design theorem.

c) Yes, "constant WD is optimal" is an overclaim. The data support: on these small BN convolutional models, under untuned settings, dynamic WD does not show a robust advantage over constant WD.

d) PMP-WD being untested is a major weakness, close to fatal. The reader cannot tell whether it is useful, stable, or implementationally reasonable.

e) The SGD vs AdamW rho comparison is seriously confounded and the incomplete matched-rho SGD run means the confound remains unresolved.

f) Three seeds are not sufficient for the statistical claims. At minimum the paper needs more seeds or confidence-interval/equivalence analysis.

g) Realistic venue acceptance probability: ~10% for NeurIPS/ICML main track.

## 评分

4/10

The paper has a real motivating question, some useful negative evidence, and a coherent narrative. But the main theorems are not yet rigorous enough, the key proposed method is not evaluated, several experiments central to the causal story are incomplete, the optimizer-vs-rho interpretation is confounded, the statistical evidence is underpowered, and there are concrete internal inconsistencies (Figure 1's regime story and the BEM accounting for half-lambda). My recommendation would be to substantially revise the paper by narrowing the claims, fixing the theoretical issues, completing the missing experiments, and validating PMP-WD before resubmission.

VERDICT: REVISE
