# Comparativist Analysis: Cross-Domain Absorption Tax

**Agent**: Comparativist
**Date**: 2026-04-15
**Iteration**: 8 (result_debate)

---

## 1. Baseline Landscape: Top Existing Methods

The following table summarizes the state of the art on feature absorption measurement and mitigation, with published numbers where available. All existing absorption measurements use the **first-letter spelling task** on Gemma 2 2B layer 12 unless noted.

| Method / Paper | Absorption Rate (first-letter) | Benchmark | Model / Layer | Key Innovation | Year |
|---|---|---|---|---|---|
| Chanin et al. ("A is for Absorption") | 15--35% (varies by L0/width) | Custom metric, NeurIPS 2025 | Gemma 2 2B, L0--17; also Llama 3.2, Qwen2 | Defines absorption metric; toy-model proof | 2024 |
| Matryoshka SAE (Bussmann et al.) | ~0.03 absorption score | SAEBench | Gemma 2 2B, L12, 65k | Nested prefix losses; hierarchical features | 2025, ICML |
| BatchTopK SAE (Bussmann) | ~0.29 absorption score | SAEBench | Gemma 2 2B, L12 | Direct sparsity control | 2024, NeurIPS Workshop |
| OrtSAE (Korznikov et al.) | ~65% reduction vs BatchTopK | SAEBench (SCR +6%) | Gemma 2 2B, L12, 65k | Orthogonality penalty on decoder | 2025 |
| ATM SAE (Li et al.) | 0.0068 mean score | Custom eval, XAI4Science ICLR 2025 | Gemma 2 2B, L12 | Adaptive temporal masking | 2025 |
| KronSAE | Reduced absorption fraction | Custom metric | Gemma 2 2B | Kronecker factorization | 2025 |
| Masked Regularization (Narayanaswamy et al.) | Claims reduction | OOD robustness | Gemma 2 2B | Token masking during training | April 2026 |

**Critical observation**: Every single entry in this table measures absorption exclusively on the first-letter spelling task. No published work measures absorption on entity-attribute knowledge hierarchies (city-country, city-continent, city-language). This is the central gap this paper fills.

---

## 2. Contribution Margin: How Our Results Compare

### 2.1 Cross-Domain Absorption Rates (Primary Contribution)

Our experimental results on Gemma 2 2B at Layer 24 (best probe quality):

| Hierarchy | SAE Config | Our Absorption Rate | First-Letter Rate (Same Config) | Delta | Cohen's d | p-value |
|---|---|---|---|---|---|---|
| city-continent | L24_16k | 35.8% | 34.5% | +1.4 pp | 0.306 | 0.829 |
| city-continent | L24_65k | 26.0% | 25.5% | +0.5 pp | 0.121 | 0.932 |
| city-country | L24_16k | 18.5% | 34.5% | -16.0 pp | -3.839 | 0.004** |
| city-country | L24_65k | 12.7% | 25.5% | -12.8 pp | -3.511 | 0.008** |
| city-language | L24_16k | 13.6% | 34.5% | -20.9 pp | -5.159 | 0.0001** |
| city-language | L24_65k | 13.6% | 25.5% | -11.9 pp | -3.242 | 0.015* |

**Contribution margin assessment**: The cross-domain variation is **statistically significant** (Kruskal-Wallis hierarchy effect p=0.005). The finding that absorption rates differ by 20+ percentage points across hierarchy types is a **strong contribution** -- no prior work has this data.

However, an important caveat: the v2 results at L24 **invert** the narrative from the earlier pilots. The pilot at L12 showed city-continent at 53.4% (much higher than first-letter at 3.9%), but the v2 results at L24 with better probes show city-country and city-language are **lower** than first-letter, while city-continent is roughly comparable. The "first-letter is atypical (lowest)" narrative from the proposal is **no longer clearly supported** at L24. Instead, the finding is more nuanced: **absorption rates are hierarchy-dependent and layer-dependent, with no single hierarchy universally highest or lowest.**

**Versus published baselines:**
- Chanin et al. report 15--35% on first-letter across layers and configs. Our L24 first-letter rates (25--35%) align with this range, confirming measurement validity.
- Our cross-domain rates (13--36%) fall within and below this range, suggesting knowledge hierarchies do not uniformly show *more* absorption than first-letter (contradicting H2' from the proposal).
- No published work has these cross-domain numbers at all, so any measured rate is novel.

**Verdict: >5% = STRONG contribution** for the cross-domain characterization itself (novelty of measurement), but the specific claim that "semantic hierarchies show MORE absorption" is **not robustly supported** in the v2 data.

### 2.2 Architecture Comparison Across Hierarchies

Our Phase 4 results (Layer 12, 4 architectures):

| Hierarchy | JumpReLU_16k | JumpReLU_65k | BatchTopK_16k | Matryoshka_32k |
|---|---|---|---|---|
| first-letter | 0.7% | 1.3% | 3.4% | 1.4% |
| city-continent | 17.3% | 23.1% | 13.5% | 19.2% |
| city-language | 41.2% | 38.2% | 61.8% | 35.3% |
| city-country | 47.1% | 47.1% | 52.9% | 35.3% |

Published SAEBench comparison (layer 12, first-letter only):
- Matryoshka: ~0.03 absorption score (SAEBench metric, not directly comparable to our rate metric)
- BatchTopK: ~0.29 absorption score
- ATM: 0.0068 mean score (different metric again)

**Contribution margin**: The finding that architecture rankings are **not consistent** across hierarchy types is a **moderate-to-strong contribution** (1--5% category in some comparisons, but the pattern itself is novel). However:
- JumpReLU is NOT consistently lowest (only in 1/4 hierarchies), contradicting the proposal's claim.
- Matryoshka tends to be lowest or near-lowest on cross-domain hierarchies (city-language, city-country), consistent with its SAEBench advantage on first-letter.
- BatchTopK is consistently worst, which aligns with published SAEBench findings.
- No pairwise comparisons reach statistical significance except BatchTopK vs Matryoshka on city-language (p=0.029).

**Verdict: MODERATE contribution.** The hierarchy-dependent ranking finding is novel but power-limited. The JumpReLU advantage claim needs to be retracted or heavily qualified.

### 2.3 Activation Patching (Causal Evidence)

Our result: 100% recovery rate for "lower" (feature 14449), 0% control. But only 1/7 words show this clean causal evidence (n=7 words total, n=1 showing recovery).

**Versus prior art:**
- Chanin et al. use integrated-gradients ablation (correlational, not interventional). Our activation patching is the first interventional evidence for competitive exclusion.
- However, the signal is extremely sparse: only 1 word out of 7 shows clean causal recovery. The other words either show no absorption or show absorption that does not resolve via single-feature patching.
- Sample size is a critical limitation: n=7 words is far below the n>50 recommended in the proposal.

**Versus the broader activation patching literature:** Activation patching is standard methodology in mechanistic interpretability (Conmy et al., 2023; Wang et al., 2023). Applying it to SAE feature absorption is novel but the execution is underpowered.

**Verdict: MODERATE contribution (methodological novelty), but MARGINAL in current evidence strength.** The 1/7 success rate needs honest framing -- this is suggestive, not conclusive.

### 2.4 Tightened Hedging Classification

Our result: Strict hedging 7.9% vs. loose 94.1% for first-letter. For city-language: strict 66.7% vs loose 100%.

**Versus prior art:**
- Chanin et al. (2025) define hedging but report it as a binary classification. Our strict/compensatory/persistent decomposition is novel.
- The 98.6% hedging rate from the broad literature is shown to be near-tautological under strict criteria.
- The cross-domain hedging comparison (first-letter 7.9% strict vs city-language 66.7% strict) is novel.

**Caveat**: The city-language decomposition is based on only n=3 false negatives, making it statistically unreliable.

**Verdict: MODERATE contribution.** The methodological critique of loose hedging classification is valuable. The cross-domain comparison is underpowered.

### 2.5 Negative Results

| Claim | Our Result | Published Baseline | Assessment |
|---|---|---|---|
| GAS as unsupervised detector | rho=0.12, AUROC=0.57 | No published GAS-absorption correlation | Novel negative result |
| CMI predicts absorption | rho=0.044, p=0.83 | No published CMI-absorption test | Novel negative result |
| Absorption Tax predictions | rho=0.08 (MSE), rho=0.16 (R_pc) | No published quantitative theory | Novel negative result |

**Verdict**: These are **valuable negative results** (1--5% contribution each), especially GAS, which was a reasonable hypothesis that definitively fails. The honest reporting aligns with community standards.

---

## 3. Concurrent Work Scan (Last 6 Months)

### Direct competitors (searching for cross-domain absorption):
- **No paper found** (as of April 2026) that measures SAE absorption on entity-attribute knowledge hierarchies. This remains a genuine gap.

### Closely adjacent work:
| Paper | Date | Relevance | Threat Level |
|---|---|---|---|
| Masked Regularization (arXiv:2604.06495) | April 2026 | Proposes training-time fix for absorption; orthogonal to our measurement contribution | LOW -- mitigation, not measurement |
| Stable & Steerable SAEs (arXiv:2603.04198) | March 2026 | Weight regularization for SAEs; not absorption-focused | LOW |
| SynthSAEBench (arXiv:2602.14687) | February 2026 | Synthetic benchmark with known hierarchies; studies absorption in synthetic setting | MEDIUM -- synthetic hierarchies overlap with our real-world knowledge hierarchy angle |
| Hierarchical SAE (arXiv:2602.11881) | February 2026 | Jointly learns parent-child relationships; not directly measuring absorption | LOW -- architectural, not measurement |
| Sanity Checks for SAEs (arXiv:2602.14111) | February 2026 | SAEs recover only 9% of true features; broader negative result | LOW -- different scope |
| Hierarchical Semantics in SAEs (arXiv:2506.01197) | June 2025 | Explicitly models hierarchy in SAE architecture | MEDIUM -- addresses hierarchy but does not measure absorption cross-domain |

**Assessment**: No concurrent paper measures absorption cross-domain on real LLM knowledge hierarchies. The closest threat is SynthSAEBench, but it uses synthetic data with known ground truth, which is complementary rather than competitive. Our real-world RAVEL-based measurement fills a distinct niche.

---

## 4. Novelty Verdict

**What is the ONE thing this work does that no prior work does?**

This paper provides the first systematic measurement of feature absorption across semantically distinct hierarchy types (syntactic first-letter, factual city-country, city-continent, city-language) on real pre-trained SAEs, showing that absorption rates are hierarchy-dependent.

This is articulable in one sentence and verified against concurrent work. The novelty is genuine.

**Secondary novelty claims:**
- First interventional (activation patching) evidence for absorption -- genuine but underpowered.
- Tightened hedging decomposition showing loose classification is near-tautological -- genuine.
- GAS definitive negative result -- genuine and cleanly established.

**Novelty risks:**
- The dramatic narrative ("first-letter is atypical, semantic hierarchies show MORE absorption") that was the paper's selling point in the proposal is **not cleanly supported** by the v2 data. The v2 data at L24 shows city-country and city-language have **lower** absorption than first-letter, while city-continent is comparable. This weakens the "reframes the entire literature" claim.
- The architecture comparison (JumpReLU advantage) is NOT supported by the data -- JumpReLU is lowest in only 1/4 hierarchies.

---

## 5. Venue Recommendation

### Evidence-based assessment:

**Strengths for top-tier venue:**
- Genuinely novel empirical contribution (cross-domain absorption measurement)
- Important negative results (GAS, CMI) reported honestly
- Timely topic (absorption is a key obstacle to SAE-based interpretability)
- Methodological contribution (tightened hedging, activation patching)

**Weaknesses for top-tier venue:**
- Results are pilot-scale, with probe quality below the 0.85 gate for all RAVEL hierarchies
- Architecture comparison lacks statistical power (no significant pairwise differences in most cases)
- Activation patching evidence is n=1 (out of 7 words)
- The flagship narrative claim (semantic > syntactic absorption) is not robustly supported in v2 data
- Absorption Tax theoretical framework has weak quantitative predictions (rho=0.08, 0.16)
- Cross-domain hedging decomposition based on n=3 false negatives

### Venue tier analysis:

| Venue Tier | Assessment | Justification |
|---|---|---|
| NeurIPS/ICML/ICLR main | NOT YET READY | Probe quality insufficient; key narrative claim not robustly supported; activation patching underpowered. Chanin et al. "A is for Absorption" was accepted as NeurIPS 2025 Oral, but it established the entire absorption phenomenon. Our contribution is incremental (extending to new domains) and the evidence is weaker. |
| NeurIPS MI Workshop / ICML Workshop | STRONG FIT | Cross-domain measurement novelty + honest negative results is ideal workshop material. Comparable to early-stage but novel empirical contributions at MI workshops. |
| EMNLP 2026 / AAAI 2027 | POSSIBLE | If probe quality is improved to >0.85 and activation patching is scaled to n>50 with significance. The cross-domain story needs cleaner support. |

**Recommendation: NeurIPS 2026 MI Workshop** as the most realistic target given current evidence quality. Upgrade to EMNLP/AAAI if full-mode experiments resolve probe quality and sample size issues.

---

## 6. Strengthening Plan

### Priority 1 (Critical, blocks venue upgrade):
1. **Improve RAVEL probe quality above 0.85 F1**: Currently best is city-continent at 0.843. This is the single biggest threat to the paper's credibility. Without reliable probes, all cross-domain absorption rates are unreliable. Explore: additional layers (L20, L22), alternative prompt templates, ensemble probes, larger RAVEL subsets, or restricting to high-confidence cities.

2. **Resolve the narrative inconsistency**: The v1 pilots at L12 showed city-continent at 53.4% >> first-letter at 3.9%, supporting "semantic hierarchies have MORE absorption." The v2 results at L24 show city-country/city-language LOWER than first-letter. The paper must either (a) present a coherent layer-dependent story, or (b) abandon the "semantic > syntactic" claim and pivot to "absorption is hierarchy-dependent with complex layer interactions." Option (b) is more honest and still novel.

3. **Scale activation patching to n>50**: The current n=7 with only 1 showing clean causal evidence is suggestive but not publishable as a primary contribution. Need at least n=50 instances across multiple hierarchies, with Wilcoxon p<0.01.

### Priority 2 (Desirable, strengthens positioning):
4. **Add comparison against ATM SAE**: ATM claims absorption score of 0.0068 (vs TopK 0.1402). Running ATM on cross-domain hierarchies would be a strong addition -- if ATM's advantage holds across hierarchy types, it strengthens both our characterization and their architecture. If it does not, that is an important finding.

5. **Add comparison against OrtSAE**: OrtSAE claims 65% absorption reduction. Testing whether this holds on knowledge hierarchies (not just first-letter) would strengthen the architecture comparison narrative and provide a more complete landscape.

6. **Report effect sizes alongside p-values for all comparisons**: Several results border on significance (e.g., BatchTopK vs Matryoshka on city-language, p=0.029). With Bonferroni correction for multiple comparisons, this would not survive. Being transparent about multiple testing is essential.

---

## 7. Honest Assessment Summary

| Dimension | Rating | Evidence |
|---|---|---|
| Novelty of measurement | HIGH | First cross-domain absorption characterization. Verified no competing work. |
| Strength of primary finding | MODERATE | Absorption differs across hierarchies (p=0.005), but "semantic > syntactic" narrative is not robustly supported at best probe layer. |
| Evidence quality | MODERATE-LOW | Probes below quality gate (best 0.843 vs 0.85 target). Activation patching n=1 effective. Hedging decomposition n=3. |
| Negative results quality | HIGH | GAS definitive negative (25x data, rho unchanged). CMI cleanly falsified. |
| Architecture comparison | LOW-MODERATE | No significant pairwise differences in most cases. JumpReLU advantage claim unsupported. |
| Theoretical contribution | LOW | Absorption Tax fails quantitative predictions (rho=0.08). Useful as qualitative framework only. |
| Overall venue readiness | WORKSHOP-TIER | Novel measurement with honest negatives is workshop-appropriate. Full mode must resolve probes and sample sizes for mid-tier venue. |

---

## Sources

- [Chanin et al. "A is for Absorption" (NeurIPS 2025)](https://arxiv.org/abs/2409.14507)
- [Bussmann et al. "Matryoshka SAEs" (ICML 2025)](https://arxiv.org/abs/2503.17547)
- [Karvonen et al. "SAEBench" (ICML 2025)](https://arxiv.org/abs/2503.09532)
- [Korznikov et al. "OrtSAE" (2025)](https://arxiv.org/abs/2509.22033)
- [Li et al. "ATM SAE" (ICLR 2025 XAI4Science)](https://arxiv.org/abs/2510.08855)
- [Chanin et al. "Feature Hedging" (2025)](https://arxiv.org/abs/2505.11756)
- [SynthSAEBench (2026)](https://arxiv.org/abs/2602.14687)
- [RAVEL dataset](https://huggingface.co/papers/2402.17700)
- [Gemma Scope SAEs](https://deepmind.google/models/gemma/gemma-scope/)
- [SAEBench interactive](https://www.neuronpedia.org/sae-bench/info)
- [Hierarchical Semantics in SAEs (2025)](https://arxiv.org/abs/2506.01197)
- [Unified SDL Theory (2024)](https://arxiv.org/abs/2512.05534)
