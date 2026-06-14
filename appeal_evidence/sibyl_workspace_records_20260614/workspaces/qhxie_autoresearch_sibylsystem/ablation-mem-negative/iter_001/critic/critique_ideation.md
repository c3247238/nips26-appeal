# Ideation Critique: SAE Feature Absorption Research Proposal

## Overall Assessment

The research proposal is well-structured with clear RQs and a feasible plan. However, the core research design has fundamental flaws that were not adequately addressed during ideation: the architecture comparison is confounded, the collision rate metric is a weak proxy for true absorption, and the exploratory contributions (UAD, DFDA) lack validation plans.

---

## Critical Issues

### 1. Confounded Architecture Comparison Was Not Flagged During Ideation

The proposal planned to compare "5 SAE architectures" but did not anticipate that pretrained SAEs (JumpReLU from GemmaScope) would confound architecture with training data, dictionary size, and training procedure.

**Evidence**: The full E1 results show JumpReLU (pretrained, d_SAE=24,576, Gemma data) vs TopK (trained, d_SAE=16,384, OpenWebText). The 4x collision rate difference cannot be attributed to architecture.

**Why this matters**: The CAAB contribution---the paper's primary claim---rests on this comparison. If the comparison is confounded, the contribution evaporates.

**What should have been done**: The ideation phase should have required matched training conditions for all architectures, or explicitly scoped the comparison as "pretrained vs trained" rather than "JumpReLU vs TopK."

---

### 2. Collision Rate as a Proxy Was Not Critically Examined

The proposal defines "absorption rate" following Chanin et al. but the actual experiments measure "collision rate" (multiple concepts sharing a feature). These are not the same phenomenon.

**Evidence**: The f1_caab_results.json shows feature 18486 is shared by letters c, i, o, p, u. This could be:
- True absorption (parent "starts with vowel" suppressing child "starts with 'i'")
- Polysemanticity (the feature responds to multiple unrelated concepts)
- A training artifact (dead features forcing concept overlap)

The proposal does not discuss how to distinguish these cases.

**What should have been done**: Include a validation step correlating collision rate with true Chanin et al. absorption rate on a subset of features with known hierarchies.

---

### 3. Exploratory Contributions Lack Clear Success Criteria

UAD and DFDA are labeled "exploratory" but their success criteria are vague:
- UAD: "Precision > 40% and Recall > 30%" (pilot) --- these thresholds are arbitrarily low
- DFDA: "Probe accuracy improvement > 5% OR reconstruction error increase < 10%" --- the OR makes this trivial to pass

**Evidence**: UAD achieved 54.3% precision at 100% recall, which meets the threshold but is still poor (nearly half false positives). DFDA achieved 11.1% per-pair residual MSE improvement on 4 pairs, but one pair degraded by 21.4%.

**What should have been done**: Set stricter success criteria (e.g., UAD precision > 70%, DFDA improvement on > 50% of pairs with no degradation > 20%).

---

### 4. No Discussion of Negative Result Framing

The proposal acknowledges that "absorption may be benign compression" (contrarian view) but does not plan for how to frame a paper where most hypotheses are falsified.

**Evidence**: H2, H3, H4 are all falsified. The paper's key finding is essentially a negative result.

**What should have been done**: Plan for negative result publication venues or framing strategies. A paper where 3 of 4 primary hypotheses fail needs a strong methodological or theoretical contribution to compensate.

---

## Strengths

1. **Clear hypothesis definitions** with falsification criteria (Section 3.2 of hypotheses.md)
2. **Risk assessment table** with mitigation strategies
3. **Pilot/Full phase structure** allows early termination of failing directions
4. **Honest contrarian hypothesis** (H-C: absorption is benign) is tested, not assumed

---

## Recommendation

The ideation phase should have:
1. Required matched training conditions for architecture comparison
2. Defined a validation protocol for the collision-absorption proxy relationship
3. Set stricter success criteria for exploratory methods
4. Planned for negative result framing
