# 8 Conclusion

Shuffled-label controls exceed measured absorption rates by 4.7$\times$ across all five hierarchy domains tested on Gemma 2 2B JumpReLU SAEs (Table 2). The Chanin absorption metric, developed and validated on GPT-2 Small with L1-ReLU SAEs, does not transfer to JumpReLU architectures without recalibration. Confound decomposition at $L_0$=22---where all 25 probes achieve F1=1.0---classifies 648 of 657 false negatives (98.6%) as hedging: the parent feature's information is spread across many latents, none clearing the JumpReLU activation threshold. Hierarchy-driven competitive exclusion accounts for 9 false negatives (1.4%). These 9 words persist as absorbed across all four tested $L_0$ values, representing 0.75% of the 1,196-word vocabulary.

Absorption declines monotonically from 42.85% ($L_0$=22) to 0.84% ($L_0$=176) on L12-16k, with the steepest decline in the $L_0 \approx 40$--$80$ range (Spearman $\rho_s = -1.0$). This profile is stable across layers 10, 12, and 20 (CV $< 10\%$ at $L_0$=82) and constitutes the most robust empirical finding: the $L_0$ operating point---a training-time hyperparameter requiring no architectural change---is the primary control parameter for absorption severity on JumpReLU SAEs.

Conditional mutual information separates absorbed from non-absorbed letters at $d' = 10$: absorbed letters have mean CMI = 0.649 $\pm$ 0.187 versus 0.861 $\pm$ 0.258 for non-absorbed (Mann-Whitney $p = 0.045$, Cohen's $d = -0.924$). Rate-distortion theory predicts this pattern: low-CMI parent features carry little unique information beyond the child and are information-theoretically cheap to absorb. For unit-normalized Gemma Scope decoders, the geometric constant degenerates ($c \approx 0.960$, CV = 2.16%), simplifying the theoretical threshold to $L_{0,\text{crit}} \approx \lambda / \text{CMI}$. The correlation holds only at $d' = 10$ and reverses at higher dimensions (Bonferroni-corrected $p = 0.236$)---a limitation that motivates dimension-agnostic MI estimation.

Three practical recommendations follow. First, validate absorption metrics on each new SAE architecture before building mitigations: the shuffled-label control check (shuffled rate $<$ measured rate in $\geq 3$ domains) should be standard practice. Second, report shuffled-label and random-probe controls alongside measured absorption rates. Third, consider the $L_0$ operating point as a first-order intervention for absorption severity before pursuing encoder modifications---on JumpReLU SAEs, increasing $L_0$ from 22 to 176 reduces measured absorption by 98%.

Code and data are released as an SAEBench extension.

<!-- FIGURES
- None
-->
