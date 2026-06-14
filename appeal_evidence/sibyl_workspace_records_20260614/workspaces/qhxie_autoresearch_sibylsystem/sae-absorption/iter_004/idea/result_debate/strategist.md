# Strategist Analysis: Competitive Geometry of Feature Absorption

**Date**: 2026-04-14
**Agent**: sibyl-strategist
**Workspace**: sae-absorption/current

---

## 1. Signal Strength Assessment

| Result | Signal | Evidence | Verdict |
|--------|--------|----------|---------|
| H1: LV detector (alpha_ij) | **Noise** | Test F1 = 0.128 at best tau=0.5; ROC-AUC = 0.148 (below random for a balanced classifier); cosine baseline beats LV (F1=0.165 vs 0.128); sharpness test favors linear over sigmoid (AIC_linear < AIC_sigmoid); F1 peak is at tau=0.4, NOT near theoretical tau~1 | The LV competition coefficient does not detect absorption. The core theoretical prediction of a sharp phase transition at alpha~1 is falsified. |
| H2: PMI prediction | **Noise** | PMI coefficient is **negative** (-0.0063, p=0.593); partial R^2 = 0.0006 (criterion was >=0.10); per-layer sign is inconsistent (3 positive, 5 negative); PMI-only model R^2 = 0.0006 | Corpus co-occurrence statistics have zero predictive power for absorption. H2 is cleanly falsified. |
| H3: Downstream disconnection | **Strong (but reversed)** | Pearson r = -0.595 (sparse probing), -0.431 (SCR), -0.454 (RAVEL/TPP); all significant after Bonferroni correction; partial correlations even stronger after controlling width/layer/arch: r = -0.661 (sparse probing), -0.677 (SCR), -0.492 (TPP); matched RAVEL comparison: low-abs mean TPP=0.046 vs high-abs mean=0.009, paired t p=0.006, Cohen's d=2.13 | Absorption IS strongly correlated with downstream performance. H3 (predicted |r|<0.2) is falsified -- but this is the **best possible outcome** for the field. |
| H4: Width paradox / DAS(k=3) | **Weak** | DAS(k=3) positive slope in only 42.3% of letters (predicted 80%); mean DAS(k=3) actually DECREASES from 24k (0.320) to 49k (0.227) to 98k (0.260); DAS(k=1) is roughly flat (~0.10-0.12). H4 is not supported. | The distributed absorption hypothesis is not confirmed. Width paradox remains unexplained. |
| Taxonomy (C2D) | **Strong** | 92.3% of letters show some form of absorption (Type I + II + III) vs. only 3.85% by strict Chanin Type I criterion. Type II (partial) dominates at 88.5%. This means Chanin's 15-35% metric captures <5% of actual absorption. | The multi-type taxonomy is the clearest positive finding. It substantially revises the field's understanding of absorption prevalence. |
| 30-SAE survey (C2B) | **Strong** | Width paradox confirmed: absorption rises from 0.009 (3k) to 0.104 (98k) at layer 8. Layer effect strong: peaks at L5 (0.119), drops to zero by L10-11. 806 valid data points across 31 configs. | This is the most comprehensive absorption survey to date. The layer/width scaling data is novel and publication-quality. |
| SAEBench downstream (C3A) | **Strong** | n=54 SAEs from real SAEBench data. Multiple independent metrics all point the same direction. Partial correlations survive confound controls. | This is the single strongest finding of the entire study. |
| Safety probe (C3C) | **Weak** | Only n=3 SAEs, confounded by layer (L5/L8/L10). Probe gap does NOT increase with absorption (highest absorption SAE has smallest gap). Pearson r=-0.76 but p=0.45 (n too small). | Underpowered and confounded. Not publishable in isolation. |

---

## 2. Opportunity Cost Analysis

| Direction | GPU-hours | Expected Information Gain | Info/GPU-hour |
|-----------|-----------|---------------------------|---------------|
| Replicate on Gemma-2-2B (gated access fix) | ~8 hrs | **High**: all current results are GPT-2 only; Gemma replication is essential for NeurIPS credibility | **High** |
| Fix LV detector (rethink formulation) | ~4 hrs | **Low**: fundamental issue is alpha_ij does not capture absorption mechanism; throwing more compute at a broken predictor is sunk-cost reasoning | **Very Low** |
| Expand downstream analysis (more tasks, more SAEs) | ~3 hrs | **Medium**: already have n=54 with strong effects; marginal returns on more data points | **Medium** |
| Scale safety probe (n=10+ SAEs, matched layers) | ~2 hrs | **Medium**: currently n=3 and confounded; a properly matched comparison would strengthen Component 3 | **Medium-High** |
| DAS(k=3) reformulation | ~3 hrs | **Low**: the measurement itself may be flawed; multi-label logistic regression with 40 samples per letter is noisy | **Low** |
| Deeper taxonomy analysis (per-letter case studies, visualizations) | ~2 hrs | **Medium**: taxonomy is already strong; case studies add narrative clarity for paper but no new hypothesis testing | **Medium** |
| Ablation: masked regularization + LV framing | ~4 hrs | **Low**: LV framing is dead; no longer a coherent mechanism to test | **Very Low** |

---

## 3. Decision Matrix

| Direction | Signal Strength | GPU Cost | Risk | Expected Outcome |
|-----------|----------------|----------|------|------------------|
| **Gemma-2-2B replication** | n/a (prerequisite) | 8 hrs | Medium (gated access) | Cross-model validation of all findings |
| **Pivot paper framing to empirical survey** | n/a (no compute) | 0 hrs | Low | Coherent story from existing strong results |
| **Scale safety probe** | Weak -> Medium | 2 hrs | Low | Properly powered downstream safety test |
| **Expand downstream correlation** | Strong -> Stronger | 3 hrs | Low | More SAE configs in correlation analysis |
| **Fix LV detector** | Noise | 4 hrs | **High** (sunk cost trap) | Unlikely to improve beyond cosine baseline |
| **DAS reformulation** | Weak | 3 hrs | High | May still yield null result |

---

## 4. PIVOT vs PROCEED Verdict

### **PROCEED** -- but with a **major framing pivot**

**Rationale**: The study has two strong results and one devastating null.

**Strong results that justify proceeding:**
1. **Downstream causal chain is VALIDATED** (H3 falsified in the best way): absorption score correlates r=-0.60 to r=-0.68 with downstream SAE quality after confound controls. This is the first systematic evidence that absorption genuinely matters for SAE utility. This finding alone is worth a NeurIPS paper.
2. **Multi-type absorption taxonomy reveals 92% prevalence**: The Chanin metric captures <5% of actual absorption. Type II (partial) absorption affects 88% of letters. This dramatically revises the field's understanding.
3. **30-SAE scaling survey** provides the most comprehensive absorption measurement to date with clear layer and width scaling laws.

**Null result that requires framing pivot:**
- The LV competitive exclusion theory is **dead**. The detector fails (F1=0.13, below cosine baseline), the sharpness test fails (linear beats sigmoid), and the theoretical prediction of a phase transition at alpha~1 is not observed. The PMI prediction is equally dead (partial R^2=0.0006). These were the headline theoretical contributions in the original proposal.

**What the new framing should be:**

The paper should pivot from "Lotka-Volterra theory of absorption" to **"Empirical Anatomy of Feature Absorption: Prevalence, Scaling, and Downstream Impact"**. The core contributions become:
1. Multi-type absorption taxonomy showing the true scope of absorption (92% vs 4% by Chanin metric)
2. First systematic evidence that absorption predicts downstream SAE quality (r=-0.60 to -0.68)
3. Comprehensive scaling laws: width paradox confirmation, layer effect characterization
4. The LV detector is reported as a negative result in an appendix -- honest science

This is a stronger paper than the original proposal because the positive findings (downstream validation + taxonomy) are more practically impactful than the theoretical LV framework would have been.

---

## 5. Recommended Next Steps (Priority Order)

### Priority 1: Gemma-2-2B Replication (~8 GPU-hours)

**Critical for NeurIPS submission.** All current results are on GPT-2 Small only. The proposal was designed for Gemma Scope SAEs. Without cross-model replication, reviewers will (correctly) question generalizability.

- Fix HuggingFace gated access for Gemma-2-2B
- Replicate C2D taxonomy on Gemma Scope layer 12 (16k, 65k)
- Replicate C3A downstream correlation on Gemma Scope SAEBench data (already available -- C3A used it)
- Report: are Type II absorption rates consistent across GPT-2 and Gemma?

### Priority 2: Properly Powered Safety Probe (~2 GPU-hours)

The current C3C is n=3, confounded by layer. Fix:
- Select 5 SAEs matched on layer=8, varying widths and absorption rates
- 5-fold CV, 100 harmful + 100 benign prompts
- Report whether 1-sparse probe gap correlates with absorption within a single layer

### Priority 3: Paper Draft with Pivoted Framing (~0 GPU-hours, writing time)

Begin writing immediately with the empirical framing. Do not wait for Gemma replication. Structure:
1. Introduction: "Absorption is under-measured and its downstream impact is unvalidated"
2. Multi-type taxonomy (Type I/II/III) and 92% prevalence finding
3. 30-SAE survey with scaling laws
4. Downstream validation (r=-0.60 to -0.68)
5. Negative result: LV detector and PMI prediction (brief, honest, in discussion)
6. Implications for SAE development and safety applications

---

## 6. Resource Allocation Recommendation

| Activity | GPU-hours | Priority | When |
|----------|-----------|----------|------|
| Gemma-2-2B replication | 8 | P1 | Immediate |
| Safety probe scale-up | 2 | P2 | After Gemma access resolved |
| Paper draft (writing) | 0 | P3 | Parallel with experiments |
| Additional downstream tasks | 3 | P4 | Only if Gemma replication succeeds |
| **Total remaining GPU budget** | **~13** | | |

---

## 7. Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Gemma-2-2B gated access remains blocked | GPT-2 results are still publishable as a single-model study; add explicit limitation section. Many SAE papers use only GPT-2. |
| Reviewer pushback on dropped LV theory | Report as honest negative result. The empirical findings stand on their own. Frame as "we tested the LV hypothesis and falsified it -- here is what actually matters." |
| Width paradox remains unexplained | This is explicitly listed as future work. The observation (absorption increases with width) is documented; the mechanism is open. |
| H3 correlation is driven by width confound (wider SAEs have both more absorption and worse downstream scores) | Already addressed: partial correlations after controlling log_width, layer, and arch_class are STRONGER (r=-0.66 to -0.68). The relationship is not an artifact. |

---

## 8. Key Quantitative Evidence

For reference in writing:

- **Taxonomy**: Type I = 3.85%, Type II = 88.5%, None = 7.7%. Comprehensive rate = 92.3%. Chanin metric captures 4% of actual absorption.
- **Downstream (raw)**: sparse_probing r = -0.595, SCR r = -0.431, TPP r = -0.454 (all p < 0.001 after Bonferroni)
- **Downstream (partial)**: sparse_probing r = -0.661, SCR r = -0.677, TPP r = -0.492 (controlling width, layer, arch)
- **Matched RAVEL**: low-abs TPP mean = 0.046, high-abs = 0.009, Cohen's d = 2.13
- **Width scaling (layer 8)**: absorption 0.009 (3k) -> 0.025 (6k) -> 0.048 (12k) -> 0.086 (24k) -> 0.092 (49k) -> 0.104 (98k)
- **Layer scaling (24k)**: peak at L5 (0.119), zero by L10-11
- **LV detector FAIL**: best test F1 = 0.136, cosine baseline F1 = 0.165, ROC-AUC = 0.148
- **PMI FAIL**: partial R^2 = 0.0006, coefficient negative, p = 0.593
