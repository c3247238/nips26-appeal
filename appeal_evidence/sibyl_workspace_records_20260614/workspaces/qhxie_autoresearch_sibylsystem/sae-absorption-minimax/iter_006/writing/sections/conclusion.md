# 6. Conclusion

We provided the first joint analysis of feature absorption and feature sensitivity as failure modes in sparse autoencoders. Our findings, while primarily negative in terms of hypotheses, paint a coherent picture of the SAE failure landscape.

## 6.1 Key Findings

1. **Absorption does not degrade steering effectiveness**: High-absorption and low-absorption features show equivalent steering sensitivity across all tested steering magnitudes ($p = 0.299$). Feature-based steering produces effects significantly above random baseline ($p < 10^{-12}$), confirming that SAE feature directions contain meaningful steering information regardless of absorption status.

2. **Absorption and sensitivity are positively correlated**: We found $r = 0.59$ between the Chanin absorption score and Tian sensitivity score, contradicting our hypothesis of independence ($r < 0.3$). This suggests these failure modes share a common underlying cause rather than being independent.

3. **The best-case quadrant (Q4) is empty**: No features fell into the low-absorption + high-sensitivity quadrant, indicating that most features experience at least one failure mode. This may partially explain the Sanity Check finding.

4. **Mutual coherence protective effect did not replicate**: The earlier pilot finding ($r = -0.786$) was not replicated ($r = +0.36$), suggesting instability in the coherence-absorption relationship.

## 6.2 Implications

These findings have important implications for SAE research:

- **For steering practitioners**: Absorption level should not be used as a criterion for steering target selection. Other factors (activation frequency, task relevance) may be more important.

- **For SAE evaluation**: The Sanity Check crisis remains partially unexplained. Absorption alone does not degrade steering, so other factors (perhaps sensitivity) may better explain why random baselines match SAEs.

- **For SAE training**: The positive correlation between absorption and sensitivity suggests that reducing one may also reduce the other, or that both share a common cause that could be addressed simultaneously.

## 6.3 Limitations

Our study is limited by sample size (43 features in pilot), single model and layer (GPT-2 Small layer 8), and pilot-scale experiments (full-scale steering-by-quadrant was not conducted due to pilot failures).

## 6.4 Future Work

Future work should:
- Investigate the common cause of absorption and sensitivity correlation
- Conduct larger-scale experiments to confirm Q4 emptiness
- Develop new theoretical frameworks for understanding SAE failure modes
- Explore whether addressing sensitivity also reduces absorption

## 6.5 Summary

We found that feature absorption does not predict steering effectiveness, that absorption and sensitivity are positively correlated (not independent), and that the predicted best-case quadrant is empty. These findings constrain but do not eliminate the compound failure hypothesis for explaining the Sanity Check crisis.

<!-- FIGURES
- None
-->