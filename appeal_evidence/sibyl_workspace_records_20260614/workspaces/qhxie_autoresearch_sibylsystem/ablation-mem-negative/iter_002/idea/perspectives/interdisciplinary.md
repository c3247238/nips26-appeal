# Interdisciplinary Perspective (Iteration 2)

## Analogy: Species Diversity in Ecology
Dead features in SAEs are analogous to species extinction in ecosystems. A dictionary with 95% dead features is like an ecosystem with 95% extinct species---the remaining 5% must handle all functional roles, leading to competition (collisions).

## Cross-Domain Insight
From ecology: species-area curves suggest that dictionary size should scale with "concept diversity" (task complexity). GPT-2 Small's first-letter task may need far fewer than 16K features, explaining the dead features.

## Proposed Extension
Test whether smaller dictionaries (d_SAE=4K or 8K) with AuxK achieve lower collision rates than larger dictionaries (16K) without AuxK. This tests the "species-area" hypothesis for SAEs.
