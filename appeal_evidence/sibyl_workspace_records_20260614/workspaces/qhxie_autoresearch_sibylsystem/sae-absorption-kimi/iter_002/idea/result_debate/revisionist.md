# Revisionist Analysis: From Data Back to Theory

## 1. Hypothesis Verdict Table

| Hypothesis | Verdict | Key Evidence | Confidence |
|---|---|---|---|
| **H1 (Construct Validity)** | **Inconclusive** | Pearson r = 0.463 (7 SAEs), 95% bootstrap CI = [-0.389, 0.981]. The point estimate is below the 0.6 threshold, but the CI is so wide that it neither excludes 0.6 (support) nor falls entirely below 0.3 (rejection). | Low |
| **H2 (Hierarchy Specificity)** | **Refuted** | Paired t-test: t = -4.748, p = 0.0032. Mean semantic-hierarchy absorption (0.235) is *significantly lower* than non-hierarchy control absorption (0.331). The effect is in the opposite direction of the hypothesis. | High |
| **H3 (Robustness across tau_fs)** | **Inconclusive** | r values are stable (0.463, 0.468, 0.471 across tau_fs = 0.01, 0.03, 0.05), but all CIs span from strongly negative to near-perfect correlation. Stability of an imprecise estimate does not imply robust validity. | Low |
| **H4 (Model Generalization)** | **Inconclusive / Weakly negative** | GPT-2 replication with only 2 SAEs shows near-zero hierarchy absorption (Standard: 0.000, TopK: 0.003). The pattern is qualitatively different from Pythia-160M, but the sample is too small to draw firm conclusions. | Very Low |
| **H5 (Architecture Ordering)** | **Partially refuted** | MatryoshkaBatchTopK is indeed low on both tasks, but Standard SAE jumps from near-zero first-letter absorption (0.026) to the *highest* semantic-hierarchy absorption (0.352). The ordering is not preserved. | Moderate |

---

## 2. Surprise Analysis

### Surprise 1: The Random-SAE control shows substantial "absorption" on semantic hierarchies (0.352) and non-hierarchy controls (0.416)
**Deviation from expectation:** We expected near-zero absorption for a random decoder. Instead, random directions produce semantic-hierarchy absorption comparable to Standard SAE (0.352) and higher than Matryoshka (0.203).

**Wrong assumption:** We assumed the absorption formula was specific to *learned* SAE structure. The data suggest the metric captures geometric properties of the residual stream itself—or artifacts of the probe projection method—rather than SAE-specific pathology. A random decoder can "absorb" features if the base model activations already encode hierarchical information in directions that happen to correlate with arbitrary decoder axes.

**Implication:** This is the most damaging result for the metric's construct validity. If random directions score this high, the metric may measure representational geometry in the base model more than SAE architecture quality.

---

### Surprise 2: Non-hierarchy correlated features show *higher* absorption than semantic hierarchies
**Deviation from expectation:** We predicted semantic-hierarchy absorption > non-hierarchy control absorption (H2). The opposite is true and statistically significant (p = 0.0032).

**Wrong assumption:** We assumed the SAEBench absorption formula was particularly sensitive to hierarchical (parent-child) structure. Instead, it appears more sensitive to general correlation or co-occurrence structure. Synonyms and co-occurring attributes (doctor/hospital, happy/joyful) may produce stronger probe projections than hypernym-hyponym pairs, perhaps because their representations are more collinear in activation space.

**Implication:** The metric is *not* hierarchy-specific. It detects correlated-feature co-occurrence more strongly than true semantic hierarchies. This undermines one of the core theoretical justifications for using absorption as a benchmark for hierarchical features.

---

### Surprise 3: Standard SAE and Random SAE are near-identical on semantic and non-hierarchy tasks
**Deviation from expectation:** Standard SAE is a deliberately simple baseline; we expected it to perform poorly on first-letter absorption (true: 0.026) but did not expect it to match a *random* decoder on semantic tasks (both 0.352 for hierarchy, both 0.416 for non-hierarchy).

**Wrong assumption:** We assumed architectural differences would manifest across all task types. Instead, first-letter absorption appears to isolate a very specific failure mode (perhaps related to how ReLU SAEs split character-level features) that does not generalize to semantic structure. Standard SAE may be "bad" at first-letter tasks for reasons unrelated to general absorption behavior.

**Implication:** The first-letter task may be measuring an architectural quirk (e.g., how ReLU handles character bigrams) rather than a general tendency toward feature absorption.

---

### Surprise 4: GatedSAE shows near-zero first-letter absorption (0.008) but moderate semantic-hierarchy absorption (0.188)
**Deviation from expectation:** If the metric were valid, the lowest first-letter absorber should also be among the lowest semantic absorbers. GatedSAE breaks this pattern—it is the best first-letter performer but only middle-of-the-pack on semantic hierarchies.

**Wrong assumption:** We assumed a monotonic relationship: low first-letter absorption → low absorption everywhere. The data suggest architectures can solve the first-letter task through mechanisms that do not transfer to semantic hierarchies (e.g., gating may specifically suppress character-level splitting without affecting how abstract concepts are distributed).

---

## 3. Mental Model Revision

**We assumed** that first-letter absorption was a canary for a general SAE pathology: the tendency of sparse autoencoders to split hierarchical features across multiple latents. **The data suggest** that first-letter absorption is better understood as a *task-specific* metric that measures how well an SAE avoids splitting *character-level* features, which may be only weakly related to how it handles *semantic-level* hierarchies. The high scores on random decoders and the inversion of hierarchy vs. non-hierarchy specificity indicate that the metric is confounded by base-model representational geometry and general correlation structure, not just SAE architecture quality.

In short: **First-letter absorption is not a general measure of "feature absorption" as a construct; it is a measure of performance on a specific benchmark task that happens to be named after a hypothesized general phenomenon.**

---

## 4. Reframing Test

**Original research question:** "Do SAEs with low first-letter absorption rates also exhibit low absorption rates on matched-frequency semantic hierarchies?"

**Would we frame it the same way today?** No. The question presupposes that "absorption" is a unified construct that should generalize across task domains. The data challenge this presupposition.

**Revised research question:** "What specific representational properties does the SAEBench first-letter absorption metric actually capture, and why do these properties fail to generalize to semantic hierarchies?"

This reframing shifts the study from a construct-validity *confirmation* exercise to a *diagnostic* investigation. The interesting finding is not that the correlation is "inconclusive" but that the metric behaves in systematically unexpected ways: random decoders score highly, non-hierarchies outrank hierarchies, and architectures with excellent first-letter scores show only mediocre semantic performance.

---

## 5. New Hypotheses Emerging from Surprising Results

### NH1: The absorption metric primarily measures base-model representational collinearity, not SAE architecture quality
**Statement:** If we compute the absorption formula on randomly sampled directions in the residual-stream activation space (without any SAE training), the scores will correlate strongly with the corresponding trained-SAE scores.

**Falsification criterion:** Pearson r < 0.5 between random-direction absorption and trained-SAE absorption across architectures.

**Concrete experiment:** Sample 100 random orthonormal directions in the residual stream. Compute absorption scores for first-letter, semantic-hierarchy, and non-hierarchy tasks. Correlate with the trained-SAE scores from this study.

---

### NH2: First-letter absorption is low for gating architectures because gating suppresses character-level polysemy, not because it reduces general absorption
**Statement:** If we test SAEs on a non-hierarchical but character-level task (e.g., detecting words that contain the substring "ing"), gating architectures will show near-zero absorption, while non-gated architectures will show high absorption—mirroring the first-letter pattern more closely than semantic hierarchies do.

**Falsification criterion:** Gated architectures show high absorption (>0.3) on the character-level substring task.

**Concrete experiment:** Construct a substring-detection benchmark (e.g., "ends with -tion" vs. specific -tion words). Compute absorption scores across the same SAE cohort. Compare correlation with first-letter absorption (expected: high) vs. semantic-hierarchy absorption (expected: low).

---

### NH3: The semantic-hierarchy task underestimates absorption because WordNet hypernyms are not represented as nested linear classifiers in the residual stream
**Statement:** If we select parent-child pairs where the child is a *prototypical exemplar* of the parent (e.g., "robin" → "bird") rather than a broad category member (e.g., "dog" → "animal"), probe AUROCs will remain high but absorption scores will drop further—suggesting the task is measuring something other than the theoretical hierarchy structure.

**Falsification criterion:** Prototypical exemplars show *higher* absorption than broad category members.

**Concrete experiment:** Split the semantic-hierarchy dataset into "prototypical" and "non-prototypical" pairs based on human ratings or WordNet depth. Compare absorption scores. If prototypical pairs show lower absorption, it suggests the metric is sensitive to category breadth or representational overlap rather than true hierarchical containment.

---

## Summary of Belief Updates

| Prior Belief | Update |
|---|---|
| First-letter absorption generalizes to semantic hierarchies | **Downgraded strongly.** The correlation is weak (r = 0.46) and imprecise; the random-SAE control undermines the interpretation. |
| The metric is specific to hierarchical features | **Rejected.** Non-hierarchy correlations produce higher absorption scores. |
| Low first-letter absorption means "better SAE" | **Revised.** It means "better at first-letter tasks," which may reflect gating-specific character-level suppression rather than general representational quality. |
| Random-SAE is a clean null baseline | **Rejected.** Random decoders score as high as Standard SAE on semantic tasks, suggesting the metric is not SAE-specific. |
| Architecture ranking is preserved across tasks | **Partially rejected.** Standard SAE and Random SAE collapse together on semantic tasks, breaking the first-letter ordering. |
