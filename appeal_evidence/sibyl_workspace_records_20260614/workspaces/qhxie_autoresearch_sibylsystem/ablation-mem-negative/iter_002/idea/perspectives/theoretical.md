# Theoretical Perspective (Iteration 2)

## Theoretical Framing
Dead features are not merely a training artifact---they are a symptom of the fundamental tension between sparsity and completeness. When 95% of dictionary features are dead, the effective dictionary size shrinks from 16K to ~800, making collisions inevitable.

## Hypothesis
H1: The apparent 4x collision rate difference between JumpReLU and TopK (15.4% vs 3.8%) is largely explained by dead feature ratio, not architecture.
H2: When dead features are eliminated, collision rates converge across architectures.
H3: Dead features create a "compression bias" that systematically underestimates true collision rates in trained SAEs.

## Theoretical Contribution
First formal analysis of how dictionary health (alive feature ratio) confounds absorption metrics.
