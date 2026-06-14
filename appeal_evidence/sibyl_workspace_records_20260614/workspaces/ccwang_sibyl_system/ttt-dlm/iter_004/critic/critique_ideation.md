# Ideation Critique — Iteration 4

## Overall Assessment: MIXED — Strong Diagnosis, Weak Follow-Through

The proposal correctly identifies the information island problem and makes a compelling case for representation-space interventions based on DMI's success. However, the proposed solutions (BSD, RACFG) were insufficiently validated against known risks before commitment.

---

## 1. Correct Directional Bets

**Credit where due:**

- The diagnosis that "parameter-space updates fail because MLM loss is near-converged" is well-supported by DTA's failure (6.2% < vanilla 12.5%). This is a genuine insight.
- The pivot from DTA to BSD/RACFG after iteration 3's data is rational and well-reasoned.
- DMI's success (9.3% vs 4.7%) as the anchor insight is correctly identified.
- A-CFG as a direction was well-chosen — the published LLaDA results (GSM8K 73.5) provided strong prior evidence.

## 2. BSD: Over-Engineered DMI with Insufficient Theoretical Grounding

**Severity: HIGH**

The proposal frames BSD as a "generalization" of DMI, replacing mask embeddings entirely with belief vectors. The fundamental problem: Dream-7B was trained to process mask embeddings, not arbitrary continuous mixtures. The proposal acknowledges this (OOD concern, 30% failure probability) but proceeds with BSD as the primary contribution.

Specific issues:
- **The "generalization" is regressive**: DMI (12.5% pilot) > BSD (6.2% pilot). Generalizing a method to include a failing special case is not progress.
- **The fallback mechanism should have been the default**: The proposal mentions a DMI-style fallback (β·mask_emb + (1-β)·belief) but treats it as a degradation path. In hindsight, the sweet spot is clearly closer to DMI than to full replacement.
- **Missing theoretical analysis before coding**: Why should replacing mask_emb with probability-weighted embeddings help? The embeddings e_v were trained for discrete tokens, not distributional mixtures. The probability-weighted sum Σ p(v)·e_v has no reason to land in a semantically meaningful region of embedding space. This should have been analyzed (e.g., probing what the probability-weighted embedding looks like relative to the convex hull of token embeddings) before committing to full-scale experiments.
- **k_frac sensitivity is a red flag**: Only k_frac=0.75 (the *shortest* belief phase) works. This means 75% of BSD's mechanism (the belief phase) is essentially unused, and the 25% that works is just a slightly improved initialization for standard unmasking.

## 3. RACFG: Predictable Failure

**Severity: HIGH**

RACFG hypothesized that cross-step JSD stability would identify "reasoning-critical positions" for targeted re-masking. This failed completely (0% accuracy everywhere).

Why this was predictable:
- **No prior evidence** that JSD stability is informative for MDLMs. The proposal assumed "low stability = model hesitating = decision point" — but this is intuition, not evidence. A well-trained model *should* be stable across steps.
- **The pilot should have tested JSD distributions first**: Before building RACFG, a 30-minute analysis of cross-step JSD distributions on Dream-7B would have revealed the near-degenerate distribution (JSD ~0.997 everywhere). This would have killed RACFG before any implementation effort.
- **Overweighting the innovator perspective**: The proposal gave 30% weight to the "innovator" who championed BSD+RACFG+DGD. The "experimenter" (20%) and "contrarian" (20%) raised valid concerns about JSD feasibility and DLM-native issues that were insufficiently heeded.

## 4. BSD+RACFG Combination: Foreseeable Non-Composability

**Severity: MEDIUM**

The proposal's H7 (BSD+RACFG synergy ≥18%) assumed orthogonal composability. But:
- BSD modifies the input distribution to the model (belief vectors vs mask embeddings)
- A-CFG's confidence scores are calibrated for the model's original input distribution
- These obviously interact — one method changes what the other method uses as signal

This should have been flagged as high-risk in the proposal, not given 50% success probability.

## 5. Research Question Scope

**Severity: MEDIUM**

RQ1-RQ4 are well-formulated but overly ambitious given the 6-day timeline:
- RQ4 (comparison with MetaState, CORE, Self-Rewarding SMC) is completely unaddressed in the experiments
- RQ3 (BSD+RACFG orthogonality) was quickly falsified, making the paper's contribution structure weaker
- The proposal should have included a "minimum viable paper" definition upfront

## 6. Missed Opportunity: DMI Deep Analysis

**Severity: HIGH**

DMI is the single most validated finding across 4 iterations (9.3% vs 4.7%, 3-seed, p<0.05). Yet the proposal treats it as prior work and immediately moves to "generalizations." A much stronger paper would:
- Deep-dive into *why* DMI works (which token types benefit most, at which denoising steps, what the logit distributions look like before/after injection)
- Scale DMI to Countdown-1000, GSM8K-full, MATH-500
- Test DMI on LLaDA-8B for cross-model validation
- Vary DMI α across the full range [0.0, 1.0] with fine granularity
- This would have been a solid EMNLP/ACL paper with far less risk

## 7. Alternatives Assessment

The pivot decision tree in alternatives.md is well-structured. However:
- Alternative 3 (empirical study paper) should have been the *primary* plan given 18 iterations of accumulated data
- The 70% success probability for Alternative 3 is likely too conservative — with existing data, this is more like 85%+
- Alternative 1 (Tolerator+BSD) and Alternative 2 (HEX+RACFG) both depend on methods (BSD, RACFG) that have now shown weaknesses
