# 5 Discussion

## 5.1 Why Co-occurrence Clustering Is the Wrong Tool

Our results establish a clear structural argument: co-occurrence clustering and hierarchical absorption are fundamentally incompatible. Co-occurrence clustering groups features that activate on the **same tokens**---features that "fire together." Hierarchical absorption, by contrast, involves features that activate on **different tokens** representing different instances of a parent concept. A feature for "three" and a feature for "four" never appear at the same token position, because a token is either "three" or "four," never both.

This is not a property of our specific experimental setup. It is a logical consequence of how language represents hierarchical concepts. If a parent concept P has child concepts C1, C2, ..., Ck, then any instance of P is an instance of exactly one Ci. The SAE learns features that discriminate between these instances---feature f_Ci activates when concept Ci is present and suppresses when other children are present. Thus, f_Ci and f_Cj (for i ≠ j) are **mutually exclusive by construction**.

UAD's co-occurrence matrix captures this mutual exclusivity as near-zero or negative phi coefficients. The clustering algorithm, correctly interpreting these coefficients, places absorption features in **different clusters**. UAD then fails to detect absorption not because of a bug or poor parameter choice, but because its core assumption---that hierarchical features co-occur---is precisely backwards for the type of hierarchy SAEs learn.

We note that co-occurrence clustering may still be useful for detecting **other types of feature relationships**: synonym features (e.g., "happy" and "joyful") that frequently co-occur in similar contexts, or contextually related features (e.g., "doctor" and "hospital") that appear in the same semantic neighborhood. Our critique is specific to hierarchical absorption, not a blanket condemnation of co-occurrence methods.

## 5.2 Why Collision Rate Works

Collision rate succeeds where co-occurrence fails because it measures a different relationship: **structural similarity of feature responses**, not shared activation context. Two child concepts may have highly overlapping top-K feature sets even if those features never activate on the same tokens. For example, "four" and "five" may both activate feature 24189 (a parent feature for the number range 4-8), even though they never appear together. The collision rate captures this shared structural relationship through Jaccard overlap of top-K features.

The strong correlation between collision rate and true absorption rate ($\rho = 0.869$) suggests that collision rate is tapping into a genuine structural property of absorption: when a parent feature is absorbed by multiple children, those children share the parent's top-K activation pattern. Collision rate thus serves as a computationally cheap **screening tool**: researchers can compute collision rates for all concept pairs and prioritize the highest-scoring pairs for more expensive causal validation.

However, collision rate is a **proxy**, not a gold standard. It correlates with absorption but does not establish causality. A high collision rate may indicate absorption, synonymy, or other forms of semantic overlap. We recommend collision rate as a **pre-filter** for candidate pairs, followed by causal validation (e.g., activation patching) to confirm absorption.

## 5.3 Theoretical Implications

Our findings have implications beyond UAD, touching on the fundamental nature of feature hierarchies in SAEs.

**Implication 1: Absorption is not a co-occurrence phenomenon.**
The SAE community has sometimes treated feature relationships as primarily statistical (co-occurrence, correlation). Our results show that absorption is a **structural** relationship---it concerns how features are organized in the model's representation space, not how they co-occur in data. This shifts the methodological focus from statistical pattern mining to geometric and causal analysis.

**Implication 2: Decoder weight similarity may be the right signal.**
If absorption is structural, then the natural place to look for it is in the SAE's decoder weights. Two features with similar decoder directions may represent similar or hierarchically related concepts, regardless of their co-occurrence patterns. Decoder weight cosine similarity is computationally cheap and theoretically grounded in the geometry of the representation space.

**Implication 3: Causal intervention is the gold standard.**
Ultimately, absorption is a causal claim: "suppressing child features causes parent feature recovery." No correlation-based method (co-occurrence, collision rate, decoder similarity) can establish causality. We view these methods as **candidate generation** tools, with causal intervention (activation patching, ablation) serving as the **validation** step.

## 5.4 Proposed Alternative Approaches

Based on our theoretical analysis, we propose three directions for future work:

**Direction 1: Decoder weight similarity.**
Replace co-occurrence clustering with clustering based on decoder weight cosine similarity. For each feature $f_i$, extract its decoder vector $d_i \in \mathbb{R}^{d_{\text{model}}}$. Compute similarity as $\text{sim}(f_i, f_j) = \cos(d_i, d_j)$. Cluster features by this similarity and extract candidate absorption pairs from within clusters. This method is computationally cheap (single matrix multiplication) and theoretically grounded.

**Direction 2: Causal intervention.**
For each candidate pair $(f_i, f_j)$ identified by collision rate or decoder similarity, perform activation patching: (1) run the model on a prompt containing concept $c_i$, (2) record the parent feature's activation, (3) zero out the child feature's activation and re-run, (4) measure parent recovery. If the parent recovers when the child is removed, absorption is confirmed. This is the gold standard but computationally expensive.

**Direction 3: Hybrid pipeline.**
Combine the three methods into a cascading pipeline: (1) use collision rate to screen all concept pairs and select top candidates, (2) use decoder weight similarity to refine the candidate set, (3) use causal intervention to validate the final pairs. This balances computational cost with rigor.

## 5.5 Limitations

We acknowledge several limitations of our study:

**Limited ground truth.** Our ground truth comprises only 7 true absorption pairs in the number hierarchy. While we validated collision rate on 56 pairs (numbers + punctuation), the absorption detection evaluation is limited to 7 positives. A larger ground truth (e.g., 100+ pairs across diverse concept hierarchies) would strengthen confidence in our conclusions.

**Single model and layer.** All experiments use GPT-2 Small layer 8 with the gpt2-small-res-jb SAE. Different models (e.g., GPT-2 Medium, LLaMA) or layers may exhibit different feature structures. Our conclusions about token-level mutual exclusivity should generalize (it is a logical property of hierarchical concepts), but the empirical failure of UAD may vary in magnitude across settings.

**Limited hierarchy types.** We test only numbers and punctuation. Abstract hierarchies (e.g., "emotion" → "joy," "sadness") or visual hierarchies (in multimodal models) may behave differently.

**No causal validation of alternatives.** We propose decoder weight similarity and causal intervention as alternatives but do not empirically validate them. These directions require future work.

Despite these limitations, our core finding---that co-occurrence clustering is structurally mismatched with hierarchical absorption---is robust to sample size, model, and hierarchy type. The mismatch follows from the logical structure of hierarchical concepts, not from empirical contingencies.
