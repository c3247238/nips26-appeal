# Result Debate Verdict

## Project: ablation-mem-weakened | Iteration: 4 | Date: 2026-04-30

---

## Score: 5/10

Up from 3/10 (iteration 3). The improvement comes from H9/H10 adding methodological clarity about the absorption metric's limitations, and the precision-recall finding (H5) remaining robust.

---

## Key Conclusion

**The field's implicit assumption that reducing feature absorption improves SAE downstream utility is not supported in GPT-2 Small.** Across 12 statistical tests with multiple comparison correction, no significant correlation exists between absorption rate and steering success or probing F1. The one robust finding is that absorption affects recall (coverage) but not precision (selectivity) --- a coverage problem, not a selectivity problem.

---

## What Works

1. **H5 is rock-solid:** Precision = 1.0 universally at k >= 5; recall varies (0.05-1.0). This is the anchor finding.
2. **Honest null results:** H1-H4 are null after correction. This challenges an implicit industry assumption.
3. **H6 falsification is valuable:** Decoder correlations do not predict absorption (precision@20 = 0.0). This rules out a class of explanations.
4. **H10 reveals metric issues:** Random SAE shows 8x higher absorption than trained. The Chanin metric conflates learned and structural artifacts.
5. **Methodological tools are reusable:** Baseline correction, precision-recall decomposition, EC50 framework.

---

## What Doesn't Work

1. **No significant positive effects:** Zero hypotheses survive multiple comparison correction. The uncorrected H1b (r=-0.431, p=0.028) is a trend, not a confirmed finding.
2. **Small sample size:** n=26 features, power ~25% for medium effects. Underpowered for subtle relationships.
3. **Single model:** GPT-2 Small only; Pythia-70M cross-validation inconclusive.
4. **First-letter features may not generalize:** Shallow features may not represent semantic hierarchies where absorption matters most.
5. **Inhibition graph is falsified:** The LCA-SAE structural correspondence does not translate into predictive power.

---

## Action Plan

### PROCEED with revised framing

**New title direction:** "The Absorption-Utility Paradox: Does Reducing Feature Absorption Improve Sparse Autoencoder Downstream Performance?"

**Core argument:**
1. The field assumes absorption reduction improves quality (Matryoshka, OrtSAE, ATM all make this implicit claim).
2. We test this assumption systematically for the first time.
3. We find no evidence for it in GPT-2 Small (12 tests, zero significant after MCP).
4. We do find that absorption is a coverage problem (recall loss), not a selectivity problem (precision intact).
5. We falsify a plausible mechanistic hypothesis (decoder correlations predict absorption: they don't).
6. We show the standard absorption metric is not specific to learned structure (random > trained).

**Immediate actions:**
- Update abstract to reflect null results + H5 + methodological contributions
- Add H9/H10 results to results section
- Strengthen limitations discussion (power, single model, feature type)
- Frame H6 falsification as a valuable negative result
- Frame H10 as a methodological caution about metric validity

**Publication target:** Workshop or arXiv preprint is achievable. Conference acceptance (NeurIPS/ICLR/ICML) depends on reviewer receptiveness to null results and the strength of the methodological contribution framing.

**Future work:** Test on larger models (Gemma-2-2B with auth), semantic hierarchies (WordNet), and alternative absorption metrics.
