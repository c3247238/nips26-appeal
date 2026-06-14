# Backup Ideas for Pivot (Iteration 7)

## Alternative A: Metric Validation Study (Promoted from H10)

**Status:** Strong backup, supported by H10 evidence.

**Core claim:** "Feature Absorption Metrics Measure Dictionary Structure, Not Learned Pathology: A Metric Validation Study"

**Why it works:**
- H10 evidence shows trained SAEs have significantly lower absorption than random baselines (mean 0.034 vs 0.278)
- This reframes absorption as a structural artifact rather than a learned failure
- Either outcome advances the field: if metrics distinguish trained from random, we have validated the metric; if not, the field must develop better metrics
- Clean, falsifiable experimental design

**Content:**
- Report trained vs. random SAE absorption comparison
- Discuss metric sensitivity to dictionary structure
- Propose guidelines for interpreting absorption metrics
- Call for development of structure-insensitive absorption metrics

**Evidence updated:**
- H7 (trained < random): trained=0.034, random=0.278, p<0.001
- H9 found to be tautological (definitional relationship)
- Feature U (24.2% abs, 100% steering) shows absorption is benign when decoder alignment preserved

**Venue:** Workshop (ICML MiF, NeurIPS XAI) or arXiv preprint.

**Risk:** May be seen as undermining the absorption subfield. Needs careful framing.

**When to pivot:** If optimal compression framing is rejected by reviewers.

---

## Alternative B: Encoder-Correlation-Based Absorption Prediction

**Status:** Exploratory, follows from H6 falsification.

**Core claim:** Since decoder correlations do NOT predict absorption (H6 falsified: precision@20 = 0.0), perhaps encoder correlations (W_enc^T W_enc) or encoder-decoder cross-correlations do.

**Why it works:**
- H6 falsification implies absorption dynamics are in the encoder, not the decoder
- Encoder correlations capture competitive dynamics during inference
- Testable with existing data

**Experiments:**
1. Compute encoder correlation matrix for GPT-2 Small SAE
2. Test precision@k against absorption pairs (same protocol as H6)
3. Compare encoder vs. decoder prediction power

**Risk:** May also fail; the mechanism may be more complex than simple correlations

**When to pivot:** If reviewers ask "what does predict absorption?" after H6 falsification

---

## Alternative C: The Absorption-Pathology Trade-off as a Pareto Frontier (Theoretical)

**Status:** Moderate backup, theoretically ambitious.

**Core claim:** For any SAE architecture, there exists a Pareto frontier in (reconstruction, sparsity, absorption) space. No SAE can simultaneously achieve zero reconstruction, minimal sparsity, and zero absorption.

**Why it works:**
- Generalizes sparsity-reconstruction trade-off to include absorption as third axis
- Provides unifying framework for comparing SAE architectures
- Project data (absorption coexists with functional steering) consistent with frontier permitting functional performance

**Experiments:**
1. Measure (R, S, A) for 5+ SAE architectures on GPT-2 Small layer 8 (~1 hr)
2. Test if points lie on 2D manifold (PCA dimensionality test) (~15 min)
3. Show absorption-reducing interventions increase other pathologies (~30 min)

**Risk:** Pareto frontier may not be smooth; outlier architectures may break manifold claim. Theory elegant but may not be empirically tight.

**When to pivot:** If optimal compression claim needs stronger empirical support.

---

## Alternative D: Scaling Laws for Feature Absorption Rate

**Status:** Weak backup, requires more data.

**Core claim:** Feature absorption rate follows scaling law: A(N) = alpha_0 * N^{beta_N} * exp(-gamma * N / N^*(L)), where N is dictionary size and N^*(L) is critical dictionary size depending on layer depth.

**Why it works:**
- First scaling law for absorption rate
- Predicts non-monotonic inverted-U curve

**Experiments:**
1. Measure absorption on GemmaScope SAEs at 16K and 65K (layer 12) (~30 min)
2. Fit preliminary scaling curve (~15 min)
3. Validate on additional dictionary sizes if available (~30 min)

**Risk:** Power-law form is phenomenological, not derived from first principles. Measurement noise may bias estimates.

**When to pivot:** If scaling patterns are interesting but other directions fail.

---

## Alternative E: Precision-Recall Asymmetry as Diagnostic Tool

**Status:** Strong backup, follows from H5.

**Core claim:** The precision-recall asymmetry (precision = 1.0 universally, recall varies) can serve as a diagnostic tool for SAE quality. Features with low recall may indicate absorption or other pathologies.

**Why it works:**
- H5 is the one robust finding across all iterations
- Precision-recall decomposition is a reusable methodological contribution
- Directly applicable to SAE evaluation workflow

**Experiments:**
1. Apply precision-recall decomposition to multiple SAE architectures
2. Test whether PR asymmetry predicts downstream task performance
3. Develop guidelines for using PR asymmetry as SAE diagnostic

**Risk:** May be too narrow for a standalone paper. Better as a methodological contribution within another paper.

**When to pivot:** If reviewers want more "positive" contributions beyond null results.

---

## Alternative F: Null Results Meta-Analysis

**Status:** Weak backup, requires broader literature survey.

**Core claim:** Across multiple studies (this project + literature), absorption does not consistently degrade downstream tasks. Null results are the norm, not the exception.

**Why it works:**
- Provides context for the project's null results
- Shows the field has been expecting a result that doesn't exist
- Encourages more rigorous evaluation standards

**Experiments:**
1. Survey published SAE papers with downstream task evaluations
2. Tabulate absorption presence vs. downstream effect size
3. Test for publication bias (positive results more likely published)

**Risk:** May be seen as "attacking the field." Requires careful framing.

**When to pivot:** If the field reacts negatively to the project's individual null results.