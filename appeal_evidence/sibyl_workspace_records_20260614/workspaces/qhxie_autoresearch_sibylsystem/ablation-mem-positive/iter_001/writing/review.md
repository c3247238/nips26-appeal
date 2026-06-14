# Paper Review

## Overall Assessment

This paper presents a systematic, training-free analysis of feature absorption in Sparse Autoencoders (SAEs) across model layers and architectures. The work is methodologically sound and provides novel empirical findings, but several areas need improvement to reach publication quality.

## Strengths

1. **Novel cross-layer quantification**: The 23x variation in absorption rates across layers (L0 to L6) is a genuinely new finding that advances beyond Chanin et al.'s early-layer-only analysis.
2. **Cross-model validation**: Replication on Pythia-70m-deduped strengthens the generalizability claims.
3. **Multiple validation approaches**: Statistical, causal (activation patching design), and human semantic validation provide converging evidence.
4. **Reproducible framework**: The three-condition detection framework is clearly specified and could be adopted by the community.

## Weaknesses

1. **Limited causal evidence**: The causal validation section presents only a design and simulated results. Without actual patching experiments, the causal claims are speculative.
2. **Narrow model scope**: Only GPT-2 Small and Pythia-70m are analyzed. Modern models (Gemma-2, Llama, Claude-scale) are not included.
3. **Methodological limitations**: The threshold choices (cosine similarity > 0.3, frequency ratio > 5) are not justified with ablation studies.
4. **Writing clarity**: Some sections are dense with statistics but lack intuitive explanations for non-specialist readers.

## Specific Issues

- Section 3.7 (Causal Validation) should clearly distinguish between designed experiments and simulated results.
- The Discussion section is underdeveloped relative to the Results section.
- Missing comparison with architectural variants (Matryoshka SAE, OrtSAE) that claim to reduce absorption.

## Recommendation

**Major Revision**. The core findings are valuable, but the causal claims need experimental validation, the model scope needs expansion, and the writing needs polishing.

**SCORE: 5.0/10**
