# Glossary

Unified terminology for all paper sections. Section writers, critics, and editors reference this file.

## Preferred Terms

| Term | Definition | Notes |
|------|-----------|-------|
| augmentation ordering | The sequence in which augmentation operations are composed within a single training image's transform pipeline. | Preferred over "augmentation order", "operation sequence", "transform permutation". |
| augmentation pipeline | A fixed-length sequence of augmentation operations composed via `transforms.Compose()` or equivalent. | Preferred over "augmentation chain", "transform chain". |
| non-commutativity | The property that composing two transforms in different orders produces different output distributions. | Preferred over "non-commutativity" with hyphen. Always pair with "of transforms" or "between transforms". |
| reversibility | The degree to which an augmentation operation can be approximately inverted, preserving the original image information. | Preferred over "invertibility". Classified as high, medium, or low. |
| contraction coefficient | The factor by which a stochastic channel reduces mutual information between input and output. | Preferred over "information loss rate". |
| pushforward | The distribution obtained by applying a transform to samples from a source distribution. | Notation: $t \# \mu$. Preferred over "image distribution" or "transformed distribution". |

## Augmentation Operations

| Term | Definition | Notes |
|------|-----------|-------|
| RandomCrop | Randomly crops the image to target size after zero-padding. | Always specify padding: "RandomCrop(32, padding=4)". Abbreviated as "Crop" in ordering labels. |
| RandomHorizontalFlip | Flips the image horizontally with probability p. | Always specify probability: "RandomHorizontalFlip(p=0.5)". Abbreviated as "Flip". |
| ColorJitter | Randomly adjusts brightness, contrast, saturation, and hue. | Always specify parameters: "ColorJitter(0.4, 0.4, 0.4, 0)". Abbreviated as "CJ". |
| RandomRotation | Rotates the image by a random angle within a range. | Always specify range: "RandomRotation(15)". |
| GaussianBlur | Applies Gaussian blur with a specified kernel size. | Always specify kernel: "GaussianBlur(kernel_size=3)". |
| Grayscale | Converts image to grayscale with probability p. | "Grayscale(p=0.1)". |
| random erasing | Randomly masks a rectangular region with random values. | Two words, lowercase. Preferred over "cutout" (which is a specific implementation). |

## Transform Categories

| Term | Definition | Examples |
|------|-----------|----------|
| geometric transform | Transforms that modify the spatial arrangement of pixels. | RandomCrop, RandomHorizontalFlip, RandomRotation. |
| photometric transform | Transforms that modify pixel intensity or color values without changing spatial layout. | ColorJitter, Grayscale, GaussianBlur. |
| block ordering | An ordering where all transforms of one category are applied before all transforms of the other category. | "all-geometric-first", "all-photometric-first". |
| interleaved ordering | An ordering that alternates between geometric and photometric transforms. | "G-P-G-P-G-P", "P-G-P-G-P-G". |

## Theoretical Terms

| Term | Definition | Notes |
|------|-----------|-------|
| Wasserstein distance | A distance metric between probability distributions based on optimal transport. | Preferred over "earth mover's distance". Specify order: "$W_2$" for 2-Wasserstein. |
| Sliced Wasserstein Distance (SWD) | A computationally tractable approximation of the Wasserstein distance via random 1D projections. | Abbreviation: SWD. Always identify as "proxy" when used in place of $W_2$. |
| Data Processing Inequality (DPI) | The principle that processing data through a channel cannot increase mutual information. | Abbreviation: DPI. Always expand on first use. |
| InfoNCE | A contrastive lower bound on mutual information. | Preferred over "info-NCE" or "Info-NCE". |
| generalization bound | An upper bound on the gap between population risk and empirical risk. | Preferred over "generalization gap bound". |

## Architecture Terms

| Term | Definition | Notes |
|------|-----------|-------|
| ResNet-18 | A residual network with 18 layers and 11M parameters. | Hyphenated. Preferred over "ResNet18" or "resnet-18". |
| ViT-Small | A Vision Transformer with Small configuration. | Hyphenated. When specifying patch size, write "ViT-S/4" (patch size 4). |
| patchification | The process of dividing an image into non-overlapping patches for input to a Vision Transformer. | Preferred over "patch embedding" when referring to the spatial division step specifically. |

## Statistical Terms

| Term | Definition | Notes |
|------|-----------|-------|
| paired t-test | A t-test comparing two conditions matched by seed. | Always specify "paired" to distinguish from unpaired. |
| Bonferroni correction | A multiple comparison correction that divides the significance threshold by the number of tests. | Preferred over "Bonferroni adjustment". |
| Spearman rank correlation | A non-parametric measure of monotonic association between two ranked variables. | Preferred over "Spearman's rho" in body text; use $\rho_s$ in equations. |
| Cohen's $d$ | A standardized measure of effect size. | Use "$d$" inline; threshold: $|d| > 0.2$ for practical relevance. |
| partial eta-squared | The proportion of variance explained by a factor in ANOVA, controlling for other factors. | Notation: $\eta^2_p$. Preferred over "eta-squared" (which does not partial out other factors). |

## Experiment Design Terms

| Term | Definition | Notes |
|------|-----------|-------|
| paired seed design | An experimental design where the same random seed is used across all experimental conditions, enabling paired statistical comparisons. | Preferred over "matched seed design". |
| factorial experiment | An experiment testing all combinations of multiple factors. | Preferred over "full factorial" unless emphasizing completeness. |
| magnitude level | The strength of augmentation parameters, scaled on a 1-15 integer range following the RandAugment convention. | Use $M$ in notation. Always specify the RandAugment convention for context. |

## Abbreviations

| Abbreviation | Expansion |
|-------------|-----------|
| CJ | ColorJitter |
| DPI | Data Processing Inequality |
| MI | Mutual Information |
| NC | Non-Commutativity |
| SWD | Sliced Wasserstein Distance |
| ViT | Vision Transformer |
| CNN | Convolutional Neural Network |
| SGD | Stochastic Gradient Descent |
| AdamW | Adam optimizer with decoupled weight decay |
| ANOVA | Analysis of Variance |
| RL | Reinforcement Learning |
