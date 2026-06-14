# 3. Research Questions and Hypotheses

## 3.1 Research Questions

We formalize five research questions:

- **RQ1 (Primary):** Does the local inhibition graph predict known absorption pairs?
- **RQ2 (Secondary):** Does inhibition explain the precision--recall asymmetry?
- **RQ3 (Secondary):** Can the graph predict at-risk features before running absorption metrics?
- **RQ4 (Exploratory):** Does graph structure vary across layers?
- **RQ5 (Exploratory):** Can homeostatic rebalancing restore parent firing?

## 3.2 Hypotheses

We retain H1--H3 from our prior correlation study (Section 5.1) and introduce H6--H10 as validation hypotheses for the inhibition framework. H4 (EC50 efficiency) and H5 (precision--recall asymmetry) from prior work are discussed in Section 5.1.

### Prior Hypotheses (H1--H3, tested in Section 5.1)

- **H1 (Raw steering):** Higher absorption rate leads to lower raw steering success rate. Directional prediction: negative Pearson correlation between $A(f)$ and $S(f, 50)$, with $p < 0.05$.
- **H1b (Delta steering):** Higher absorption rate leads to lower delta steering effectiveness, where $\Delta S(f) = S(f, 50) - S_{\text{rand}}(50)$. Directional prediction: negative Pearson correlation between $A(f)$ and $\Delta S(f)$, with $p < 0.05$.
- **H2 (Probing):** Higher absorption rate leads to lower sparse probing F1. Directional prediction: negative Pearson correlation between $A(f)$ and $\text{F1}(f, 5)$ with $p < 0.05$.
- **H3 (Consistency):** The degradation relationship is consistent across layers. Prediction: regression slopes $\beta$ have the same sign and similar magnitude across layers, with $\text{CV} = \sigma / |\mu| < 0.5$.

The H1b hypothesis is critical because random feature steering achieves non-negligible success (34--38% in our data), indicating that raw steering metrics conflate feature-specific contribution with generic directional bias. H1b isolates the true feature-specific effect by subtracting the random baseline.

### Supplementary Exploratory Analyses (H4--H5, reported in Section 5.1)

- **H4 (EC50 efficiency):** Absorbed features require higher steering strength to achieve the same effect. We compare EC50 between HIGH- and LOW-absorption features.
- **H5 (Precision invariance):** Precision (selectivity) is invariant across absorption levels, while recall (coverage) varies. We decompose probing F1 into precision and recall.

H4 and H5 were not pre-registered; they are reported as supplementary exploratory analyses without multiple comparison correction.

### Proposed Validation Hypotheses (H6--H10, protocols in Section 5.2)

- **H6 (Graph predicts absorption):** Edges in the local inhibition graph predict known absorption pairs with enrichment over chance. Prediction: precision@20 $\geq 0.10$ (123$\times$ enrichment over random baseline).
- **H7 (Inhibition explains precision--recall asymmetry):** Total incoming inhibition correlates negatively with recall but not with precision. Prediction: $r(\text{inh}, \text{recall}) < 0$ with $p < 0.05$; $r(\text{inh}, \text{precision})$ non-significant ($p > 0.05$).
- **H8 (Graph predicts at-risk features):** Graph statistics correlate positively with absorption rate. Prediction: $r(\text{total\_inhibition}, \text{absorption\_rate}) > 0.3$ with $p < 0.05$.
- **H9 (Layer-dependent structure):** Mean edge weight increases with layer depth. Prediction: $r(\bar{G}, l) > 0.3$.
- **H10 (Homeostatic rebalancing):** Parent firing increases by $>20\%$ with $<5\%$ reconstruction error increase. Prediction: $\Delta_{\text{fire}} > 0.20$ and $\Delta_{\text{recon}} < 0.05$ at some $\alpha$.

## 3.3 Falsification Criteria

**H1, H1b, H2:** Not supported if the Pearson correlation is non-negative ($r \geq 0$) or fails to reach significance ($p \geq 0.05$). A negative but non-significant trend does not support the hypothesis. We perform 12 statistical tests (H1, H1b, H2, each with Pearson and Spearman across two layers) and apply both Bonferroni and Benjamini--Hochberg FDR corrections. A negative but uncorrected trend does not support the hypothesis.

**H3:** Not supported if slopes have opposite signs or differ substantially in magnitude ($\text{CV} \geq 0.5$).

**H6--H10:** Not supported if the predicted effect does not reach the stated threshold or significance level. These are proposed validation experiments; falsification criteria are specified with each protocol in Section 5.2.

<!-- FIGURES
- None
-->
