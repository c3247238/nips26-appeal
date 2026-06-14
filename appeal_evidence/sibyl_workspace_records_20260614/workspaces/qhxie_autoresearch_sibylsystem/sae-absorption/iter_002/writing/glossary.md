# Glossary

Preferred terminology for all writing, critique, and editing agents. Enforce consistency across
all sections.

---

## Core Concepts

**Sparse Autoencoder (SAE)**
A neural network with a bottleneck encoder and over-complete decoder trained to reconstruct
residual stream activations from a sparse intermediate representation. Abbreviation: SAE.
Preferred phrasing: "sparse autoencoder" (lowercase, spelled out on first use), then "SAE".
Do NOT write: "Sparse Autoencoder" as a proper noun mid-sentence.

**Feature absorption**
The failure of a child SAE feature to activate when its corresponding concept is present in the
input, because a parent feature in the hierarchy has "absorbed" the activation signal. Defined
operationally via FeatureAbsorptionCalculator (Chanin et al. 2024): child feature activation
is suppressed relative to when the parent is absent.
Preferred phrasing: "feature absorption", "absorption". NOT "feature stealing", NOT "feature suppression"
(suppression is a separate phenomenon).

**Parent feature**
The SAE feature corresponding to a broader, more general concept (e.g., letter-class feature for
"words starting with A"). Abbreviated: parent.

**Child feature**
The SAE feature corresponding to a specific, narrow concept (e.g., token-specific feature for
"apple"). Abbreviated: child. In an absorbed pair, the child fails to activate.

**Encoder-decoder dissociation / EDA**
The misalignment between the encoder direction ($\hat{e}_j$) and decoder direction ($d_j$) of
a single SAE feature. Formally: $\text{EDA}(j) = 1 - \cos(\hat{e}_j, d_j)$.
Preferred phrasing: "encoder-decoder dissociation" (conceptual), "EDA" (when referring to the
metric). Do NOT write "encoder-decoder misalignment" (reserved for a general description);
use "EDA" when precision is needed.

**Rate-distortion theory (in SAE context)**
The framework of analyzing SAE training as a trade-off between reconstruction accuracy
(distortion) and activation sparsity (rate). The absorption preference (Proposition 1) is
derived within this framework.
Preferred phrasing: "rate-distortion framework". NOT "information-theoretic" (too broad).

---

## Experimental Setup

**First-letter task**
The hierarchical relationship where a parent feature detects "words beginning with letter X"
and child features detect individual word tokens (e.g., "apple", "arm"). Defined in Chanin et al.
(2024) sae-spelling benchmark.
Preferred phrasing: "first-letter task", "first-letter hierarchy". NOT "spelling task"
(too ambiguous).

**FeatureAbsorptionCalculator**
The tool from Chanin et al. (2024) that computes absorption rates via integrated gradients (IG)
ablation. Produces exact ground-truth absorption labels for GPT-2 Small layer 6.
Preferred phrasing: "FeatureAbsorptionCalculator (Chanin et al. 2024)". On subsequent uses:
"Chanin et al. labels" or "exact absorption labels".

**Ground-truth absorption labels**
Labels produced by FeatureAbsorptionCalculator (n_pos = 18 for GPT-2 Small L6, 24,576-width SAE).
Contrasted with proxy labels (n_pos = 50; letter features above decoder-probe threshold).
Preferred phrasing when distinguishing: "exact labels" vs. "proxy labels".

**Probe**
A linear classifier trained on residual stream activations to detect a concept (e.g., "does this
token start with the letter A?"). Probes are used to identify parent features and to define
the first-letter hierarchy.
Preferred phrasing: "linear probe". NOT "classifier head" (too neural-network-generic).

**Phase stability**
The empirical finding that absorption rates remain uniformly high (0.919–0.968) across all tested
SAE configurations (layers 2–10, widths 12k–98k). Contrasted with a "phase transition" model
where absorption would appear or disappear at a threshold sparsity.
Preferred phrasing: "absorption phase stability", "phase-stable phenomenon".

---

## Metrics and Statistics

**AUROC**
Area under the receiver operating characteristic curve. Range [0, 1]; 0.5 = random.
Preferred phrasing: "AUROC". Do NOT write "AUC" alone (ambiguous with AUPRC).

**AUPRC**
Area under the precision-recall curve. More informative than AUROC for extremely imbalanced
label sets (base rate ≈ 0.073%). Always report alongside AUROC when class imbalance is severe.
Preferred phrasing: "AUPRC", "AUPRC/base rate" (the ratio to the baseline precision).

**Cohen's d**
Standardized mean difference effect size. Computed as $(\mu_+ - \mu_-) / \sigma_\text{pooled}$.
Preferred phrasing: "Cohen's $d$". Report to 2–3 significant figures.

**Null distribution**
In this paper, the permutation null for AUROC: obtained by permuting absorption labels 100 times
and computing AUROC on each permutation. The z-score above the null mean confirms significance.
Preferred phrasing: "permutation null", "null distribution". NOT "bootstrap null" (different
resampling).

**LRT p-value**
Likelihood-ratio test p-value comparing sigmoid vs. linear fit to the (1/L0, EDA) relationship.
LRT p = 0.456 means sigmoid is NOT preferred.
Preferred phrasing: "LRT p-value = 0.456". NOT just "not significant" (report the number).

---

## Architecture Terms

**Standard/ReLU (L1 SAE)**
SAE trained with L1 sparsity penalty on activations. Primary suite in this paper: gpt2-small-res-jb.
Preferred phrasing: "L1-penalized SAE", "standard SAE". NOT "vanilla SAE".

**TopK SAE**
SAE variant that enforces exactly $k$ active features per input (no L1 penalty).
Preferred phrasing: "TopK SAE" with capitalized K. Note: "TopK" not "top-k" in compound modifier.

**AJT SAE**
Alternative training regime for GPT-2 Small (gpt2-small-res_*-ajt). Shows reversed EDA polarity
(negative EDA_delta for letter features) compared to L1-penalized SAEs.
Preferred phrasing: "AJT-trained SAE", "AJT regime". Do NOT call "alternative SAE" (too vague).

**OrtSAE**
SAE architecture with orthogonality penalty on decoder columns; forces decoders apart.
Preferred phrasing: "OrtSAE". NOT "orthogonal SAE".

**Matryoshka SAE**
SAE with hierarchical inner/outer codebook structure (Bussmann et al. 2025).
Preferred phrasing: "Matryoshka SAE".

---

## Claims and Evidence

**Mechanistic conjecture**
A claim that is algebraically derivable given stated assumptions, but whose assumptions have not
been fully verified empirically. Proposition 2 (encoder pulled toward parent decoder under
absorption) is labeled a mechanistic conjecture in this paper.
Preferred phrasing: "mechanistic conjecture". When using: always add "(Proposition 2, mechanistic
conjecture)" or similar qualifier. Do NOT present as a proven theorem.

**Proved / Proven**
Reserved for claims established via formal mathematical proof (Proposition 1 and its corollaries).
Do NOT use "proved" for empirically validated claims.

**Empirically supported / validated**
Used for claims supported by statistically significant experimental evidence.
Preferred phrasing: "empirically supported", "validated at AUROC = X".

**Unresolved tension**
An inconsistency between a theoretical prediction and observed data that we cannot currently
resolve. The EDA magnitude tension (large observed EDA for letter features vs. small predicted
from Proposition 1 for small $\theta_{p,c}$) is acknowledged as unresolved.
Preferred phrasing: "unresolved tension". Do NOT smooth over; report with data.

---

## Abbreviations

| Abbreviation | Full Form |
|---|---|
| SAE | Sparse Autoencoder |
| EDA | Encoder-Decoder Alignment (dissociation metric) |
| LLM | Large Language Model |
| MI | Mechanistic Interpretability |
| AUROC | Area Under the Receiver Operating Characteristic Curve |
| AUPRC | Area Under the Precision-Recall Curve |
| IG | Integrated Gradients |
| LRT | Likelihood-Ratio Test |
| L0 | Mean number of active features per forward pass (sparsity proxy) |
| CI | Confidence Interval |
| GPT-2 | GPT-2 Small (117M parameters), the primary model used in this paper |
| AJT | Alternative training regime for GPT-2 Small SAEs (specific SAELens release) |

---

## Banned Phrases (Do Not Use)

| Banned | Preferred Alternative |
|---|---|
| "novel contribution" (without quantification) | State the specific AUROC/effect size |
| "significantly outperforms" (without number) | "outperforms by AUROC = 0.080 (DeLong p = 0.15)" |
| "to the best of our knowledge" | Just state the claim; cite literature |
| "feature stealing" | "feature absorption" |
| "vanilla SAE" | "L1-penalized SAE" or "standard SAE" |
| "interesting finding" (without substance) | State what the finding shows numerically |
| "in recent years" | Start with the specific problem/finding |
| "Moreover", "Furthermore" | Use only when logically required; prefer direct transitions |
| "groundbreaking", "game-changing" | Not used unless backed by specific comparison |
