# Result Debate Synthesis: DTA Full-Scale Experiment Mid-Term Assessment

**Synthesizer**: Senior Research Director
**Date**: 2026-03-09
**Input**: 6 perspectives (Optimist, Skeptic, Strategist, Methodologist, Comparativist, Revisionist)

---

## 1. Consensus Map: High-Confidence Conclusions

All six perspectives agree on the following points, making these our highest-confidence findings:

### 1.1 DMI is the standout positive result
- DMI (Diffusion Memory Injection) achieves **9.3% vs 4.7% vanilla** on Countdown-500 (3 seeds), approximately 2x improvement
- Near-zero computational overhead (~1.05x)
- Consistent across seeds: 7.8% / 9.6% / 10.6% (std = 1.4pp)
- **All six perspectives acknowledge this as the strongest empirical finding to date**

### 1.2 Pure remasking methods fail on reasoning tasks
- ReMDM-conf: 4.4% (below vanilla 4.7%)
- RCR: 5.7% (marginal, likely not statistically significant)
- This supports the "Information Island" hypothesis: token-space remasking without cross-step information transfer cannot improve reasoning

### 1.3 DTA full-scale results are the critical missing piece
- DTA has **no full-scale data yet** — all pilot results are 16-sample and statistically unreliable
- DTA pilot signals are consistently negative (Countdown: 6.2% vs 12.5%, LLaDA GSM8K: 18.8% vs 43.8%)
- **However**, DMI's pilot-to-full reversal (0% → 9.3%) proves that pilot results can be completely misleading
- No strong conclusion about DTA can be drawn until full-scale experiments complete

### 1.4 Pilot-to-Full-Scale inconsistency is a methodological warning
- DMI: 0% (pilot) → 9.3% (full-scale) — complete direction reversal
- PPL metrics: 20+ pp pilot improvements vanished at full scale
- 16-sample pilots are insufficient for reliable conclusions in either direction
- Future pilots should use ≥50 samples × 2 seeds

### 1.5 The paper has a viable path regardless of DTA outcome
- All perspectives agree that DMI + systematic remasking failures + information hierarchy constitute publishable material
- The question is venue tier, not publishability

---

## 2. Conflict Resolution: Where Perspectives Disagree

### 2.1 DMI's evidence strength

| Perspective | Rating | Rationale |
|-------------|--------|-----------|
| Optimist | Strong | 2x improvement, consistent across seeds |
| Skeptic | Weak | No McNemar test, alternative explanations not excluded |
| Methodologist | Medium | Sound direction but pilot inconsistency raises concerns |
| Comparativist | Medium | Meaningful but small absolute gain; novelty threatened by Soft-Masked DLMs |

**My judgment**: DMI's evidence is **medium-strong**. The 3-seed consistency on 500 samples is statistically meaningful (effect size ~4.6pp on 500 samples should survive McNemar test), but the skeptic is correct that we lack formal paired statistical tests. The alternative explanations (implicit temperature smoothing, increased input information) are plausible but secondary — even if DMI works partly through input smoothing, the practical result stands. **Action**: Run McNemar + Bootstrap CI immediately; run random-logit DMI ablation to exclude the null hypothesis.

### 2.2 Vanilla baseline discrepancy (4.7% vs Dream paper's 16.0%)

| Perspective | Concern Level | Assessment |
|-------------|---------------|------------|
| Skeptic | Critical | If baseline is wrong, all comparisons are invalid |
| Optimist | Not mentioned | — |
| Methodologist | Noted | Different sample sets and evaluation settings likely explain gap |
| Comparativist | Significant | Documented as evaluation difference (8-shot vs our setting) |

**My judgment**: This is **important but likely explainable**. Dream paper uses 8-shot prompting while our evaluation likely uses zero-shot or different prompt format. The GSM8K baseline (~29.6%) matching Dream paper's report confirms our evaluation framework is not fundamentally broken. **Action**: Document the exact evaluation differences (shot count, prompt format, sample source) in the paper. This is a disclosure issue, not a validity issue.

### 2.3 DTA's prospects

| Perspective | Probability DTA works at full-scale | Rationale |
|-------------|-------------------------------------|-----------|
| Optimist | ~50% | Strong engineering foundation, MBPP positive signal |
| Skeptic | <10% | No evidence, theory may not hold |
| Strategist | ~35% (Scenario A) | DMI proves information transfer works; DTA is deeper version |
| Comparativist | 25-30% | Pilot signals consistently negative across 3 benchmarks |
| Revisionist | ~15% | Structural mismatch between gradient updates and denoising dynamics |
| Methodologist | Cannot assess | Hyperparameter search on 16 samples is unreliable |

**My judgment**: **~25% probability** that DTA significantly outperforms vanilla at full-scale. The revisionist's analysis is the most convincing: the structural mismatch between parameter updates and denoising scheduling (each step's mask pattern partially invalidates previous parameter adaptations) is a fundamental concern, not just a hyperparameter issue. However, the DMI pilot reversal (0% → 9.3%) prevents us from ruling out DTA entirely. **Action**: Complete DTA full-scale and treat it as hypothesis testing, not as an expected positive result.

### 2.4 Paper narrative and contribution positioning

| Perspective | Recommended core contribution |
|-------------|-------------------------------|
| Optimist | "Cross-step information transfer is critical for DLM reasoning" |
| Strategist | Scenario-dependent: DTA method paper OR DMI discovery paper |
| Comparativist | DMI as practical contribution; DTA status determines venue tier |
| Revisionist | "Why Deeper Isn't Better: shallow intervention advantage in DLM" |

**My judgment**: The revisionist's framing is the most intellectually honest and original. The finding that **embedding-level intervention outperforms parameter-level adaptation** is counterintuitive and publishable regardless of DTA's full-scale outcome. The paper should be framed around the information augmentation hierarchy with DMI as the practical anchor, not around DTA as the core method.

---

## 3. Result Quality Score: **5.5 / 10**

**Justification**:

**Strengths** (+):
- DMI's 2x improvement is real, consistent, and practically meaningful (near-zero overhead)
- Systematic negative results for remasking methods provide strong contrastive evidence
- Multi-seed, large-sample evaluation framework is sound
- SCP's Correction Precision (76.9% vs ReMDM's 31.3%) is a clean diagnostic finding
- Cross-model data (LLaDA-8B) adds breadth

**Weaknesses** (−):
- Core method (DTA) has zero full-scale data
- No formal statistical tests reported (McNemar, Bootstrap CI, Bonferroni correction)
- Pilot-to-full-scale reversals undermine confidence in any pilot-based conclusions
- Missing baselines (CORE, Soft-Masked, Best-of-N compute-matched)
- DMI novelty threatened by Soft-Masked DLMs (NeurIPS 2025)
- Absolute accuracy remains very low (<10%), limiting practical impact narrative

**Score breakdown**: Data completeness 4/10, Statistical rigor 3/10, Experimental design 6/10, Finding significance 7/10, Reproducibility readiness 5/10

---

## 4. Key Findings: What We Actually Learned

1. **Cross-step information transfer works, but simpler is better.** DMI's embedding-level soft signal injection (~1.05x overhead) achieves the same or better improvement than more complex mechanisms (SCP ~1.5x, DTA ~2.5x). The marginal value of deeper intervention is zero or negative.

2. **Pure remasking is insufficient for reasoning tasks.** ReMDM-conf (4.4%) and RCR (5.7%) provide no meaningful improvement over vanilla (4.7%) on Countdown. ReMDM-conf's Correction Precision of only 31.3% explains why: it remaskes ~70% correct tokens, destroying rather than improving the reasoning chain.

3. **Pilot results are unreliable at 16 samples.** DMI's complete reversal from 0% (pilot) to 9.3% (full-scale) is a methodological warning. PPL metrics also showed 20+ pp reversals. No pilot conclusion should be treated as definitive.

4. **DLM denoising dynamics may resist parameter-level perturbation.** DTA's LoRA delta norm of ~1e-5 suggests the updates are numerically negligible under bfloat16. Combined with consistently negative pilot signals, this points to a possible structural incompatibility between online parameter updates and the denoising process.

5. **The information augmentation hierarchy is inverted from expectations.** The predicted ordering DTA > SCP > DMI > Vanilla is empirically DMI > SCP ≈ RCR > Vanilla > ReMDM-conf. This inversion is itself a significant finding about DLM inference-time enhancement.

---

## 5. Methodology Gaps: Critical Improvements Needed

### 5.1 Blocking (must complete before paper submission)

1. **Formal statistical tests**: McNemar paired test + Bootstrap 95% CI + Bonferroni correction for all pairwise comparisons. Currently we have only means and standard deviations.

2. **DTA + DTA+ReMDM full-scale Countdown**: The entire paper's positioning depends on these results. No writing should be finalized until these are in hand.

3. **DMI ablation — random logit injection**: Must exclude the null hypothesis that DMI works through input smoothing rather than cross-step memory. Use random logits from a different sequence as control.

4. **Baseline calibration documentation**: Explain the 4.7% vs 16.0% (Dream paper) Countdown discrepancy. Specify prompt format, shot count, sample source, and denoising steps.

### 5.2 High priority (significantly improves paper quality)

5. **Compute-matched comparison**: Best-of-N (2-3 samples) at equivalent compute budget to DTA's ~2.5x overhead. If Best-of-3 vanilla > DTA, the parameter adaptation story collapses further.

6. **Soft-Masked DLM comparison**: DMI's main novelty competitor. Must either (a) run Soft-Masked in our setup, or (b) clearly differentiate (training-free vs continued pretraining).

7. **Per-sample difficulty analysis**: Stratify DMI improvement by Countdown problem difficulty. If DMI only helps on easy problems, the practical significance diminishes.

8. **Full-scale ablations**: LoRA rank, gamma, learning rate on ≥200 samples. Current 16-sample ablations contain zero statistical information.

### 5.3 Medium priority (strengthens narrative)

9. **Token-level Correction Precision/Recall for all methods** (currently only pilot-scale for ReMDM and SCP)
10. **GSM8K full method comparison** (currently only 1 seed, 27% progress for ReMDM-conf)
11. **MBPP full-scale for DTA** (pilot +12.5pp is the only positive DTA signal; needs validation)
12. **ReMDM-conf hyperparameter search** on Countdown (current config may be suboptimal for this task)

---

## 6. Competitive Position

### 6.1 Direct threats

| Concurrent Work | Threat Level | Why |
|-----------------|-------------|-----|
| **MetaState** (arXiv 2603.01331, 2026-03-02) | **High** | Same "Information Island" framing, trainable cross-step memory, verified on Dream-7B and LLaDA-8B |
| **Soft-Masked DLMs** (NeurIPS 2025) | **High** | DMI's core idea (前步预测信息传递) overlaps conceptually |
| **CORE** (arXiv 2602.04096) | **Medium** | Training-free inference correction, similar positioning |

### 6.2 Differentiation strategy

- **vs MetaState**: Our approach is training-free (MetaState requires K-step unrolling training). If DTA works, we offer stronger improvement without training. If DTA fails, DMI still provides a simpler training-free alternative to MetaState's complex GRU+CrossAttn architecture.
- **vs Soft-Masked**: DMI is training-free (no continued pretraining). Must cite Soft-Masked explicitly and position DMI as the zero-training variant.
- **vs CORE**: Different mechanism entirely (information injection vs remasking correction). Include CORE as baseline if feasible.

### 6.3 Honest competitive assessment

With DTA data missing, our strongest contribution is DMI (+4.6pp, training-free, near-zero overhead). This is a **useful but incremental** contribution. The information augmentation hierarchy (systematic ablation across 4 levels × 7+ methods × 2 models × 3 benchmarks) adds significant experimental value. Together, this positions the paper at **EMNLP/ICLR level** without DTA, and **NeurIPS/ICML level** with a successful DTA.

---

## 7. Hypothesis Update

### Survived (supported by full-scale data)
- **Information Island hypothesis**: Cross-step information loss is a real bottleneck. DMI's success + remasking's failure provides direct evidence.
- **Correction Precision limitation (H9)**: ReMDM's 31.3% precision confirms that confidence-based remasking cannot accurately identify error tokens.
- **DTA safety (H10)**: DTA does not introduce text degradation (rep-3 actually lower than vanilla).

### Need revision
- **Information augmentation hierarchy (H4)**: Predicted DTA > SCP > DMI > Vanilla. Actual: DMI > SCP ≈ Vanilla. **The hierarchy is inverted** — simpler, shallower interventions outperform deeper ones.
- **DTA as implicit TTT**: The revisionist correctly identifies a structural mismatch — DLM denoising steps remask/unmask tokens, partially invalidating previous parameter updates. DTA is NOT equivalent to AR-style TTT.
- **DTA + ReMDM complementarity (H2)**: Pilot data shows interference, not complementarity. DTA's parameter updates alter confidence distributions, disrupting remasking thresholds.

### Need new data (cannot assess)
- **DTA effectiveness (H1)**: Awaiting full-scale. Probability of positive outcome: ~25%.
- **Inference-time scaling (H3)**: 16-sample scaling curves are in noise.
- **Gamma optimal range (H8)**: All pilot DTA configs scored 0%, providing no discrimination.

### New hypotheses generated
- **NH1 (Shallow Intervention Principle)**: In DLM denoising, intervention effectiveness is inversely proportional to intervention depth. Embedding > Token > Parameter.
- **NH2 (Decision Delay)**: DMI works not through "cross-step memory" but through "delayed hard commitment" — soft embeddings prevent premature token decisions.
- **NH3 (Numerical Precision Floor)**: DTA's bfloat16 LoRA updates (~1e-5 delta norm) may be below effective precision, making DTA a costly identity map.
- **NH4 (Baseline Capability Floor)**: Inference-time enhancement has negligible absolute gains when model baseline accuracy is <10%.

---

## 8. Action Plan

### Verdict: **PROCEED** (with narrative pivot preparation)

The project should complete all planned full-scale experiments. DMI has eliminated the risk of total failure. However, the paper narrative should be prepared for the most likely outcome (DTA ≤ DMI) rather than the originally hoped outcome (DTA >> DMI).

### Immediate actions (next 24h)

| Priority | Action | Rationale | GPU/Time |
|----------|--------|-----------|----------|
| **P0** | Complete DTA full-scale Countdown (500 × 3 seeds) | Core hypothesis test | 3 GPU, ~10h |
| **P0** | Complete SCP full-scale Countdown | Finalize Level 2 positioning | Running |
| **P0** | Run McNemar + Bootstrap CI on all completed pairs | Statistical rigor is blocking | 0 GPU, ~2h |
| **P1** | DMI random-logit ablation (500 × 1 seed) | Exclude null hypothesis | 1 GPU, ~3h |
| **P1** | DTA+ReMDM full-scale Countdown | H2 complementarity test | 3 GPU, ~10h |

### Short-term actions (24-48h)

| Priority | Action | Rationale |
|----------|--------|-----------|
| **P1** | GSM8K: DTA + DMI (3 seeds each) | Cross-task generalization |
| **P2** | Best-of-3 vanilla compute-matched baseline | Fair comparison at DTA's compute budget |
| **P2** | Document Countdown baseline discrepancy | Required for paper honesty |
| **P2** | DMI alpha sweep (0.1, 0.2, 0.3, 0.5) at full-scale | Optimize the winning method |

### Narrative contingency plans

**Scenario A (DTA > 12%, prob ~25%)**: "Denoising-Time Adaptation" method paper. DTA as core, DMI/SCP as ablation hierarchy. Target: NeurIPS/ICML.

**Scenario B (DTA ≈ DMI 8-12%, prob ~30%)**: "Information Augmentation Hierarchy" paper. Finding that parameter-level isn't better than embedding-level. Target: ICLR/NeurIPS.

**Scenario C (DTA < 8% or < vanilla, prob ~40%)**: "Why Deeper Isn't Better" analysis + DMI discovery paper. Systematic ablation + diagnostic framework + practical DMI method. Target: NeurIPS/EMNLP.

**Scenario D (Total failure, prob ~5%)**: DMI as standalone finding + comprehensive negative results analysis. Target: ACL Findings/EMNLP.

### Risk mitigation

- **MetaState competition**: Differentiate explicitly as training-free. Cite and compare.
- **Soft-Masked overlap**: Must acknowledge and differentiate DMI from Soft-Masked DLMs.
- **Statistical weakness**: Complete all formal tests before any paper draft.
- **DTA hyperparameter concern**: If DTA fails at full-scale, run one round with lr=1e-3 + float32 LoRA to exclude the numerical precision floor hypothesis (NH3).

---

## Appendix: Evidence Confidence Matrix

| Claim | Evidence Level | Data Source | Action Needed |
|-------|---------------|-------------|---------------|
| DMI provides ~2x improvement | **Medium-High** | 500 × 3 seeds | McNemar test + ablation |
| Pure remasking fails on reasoning | **Medium** | 500 × 3 seeds | Hyperparameter search for ReMDM |
| Correction Precision explains remasking failure | **Medium** | 16 samples (but ratio metric) | Full-scale validation |
| DTA is ineffective | **Low** (pilot only) | 16 × 1 seed | Full-scale required |
| Information hierarchy is inverted | **Medium-Low** | Mixed pilot + partial full-scale | DTA + SCP full-scale to confirm |
| Paper is publishable | **High** | All perspectives agree | — |
