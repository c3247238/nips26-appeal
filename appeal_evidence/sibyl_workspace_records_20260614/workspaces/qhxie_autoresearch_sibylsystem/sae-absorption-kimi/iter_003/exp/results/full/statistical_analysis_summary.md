# Statistical Analysis Summary

## H1: Construct Validity (First-Letter vs Semantic-Hierarchy Absorption)

- **Pearson r** (excluding Random SAE): 0.463
- **Bootstrap 95% CI**: [-0.389, 0.981]
- **Assessment**: INCONCLUSIVE

With Random SAE included:
- Pearson r = 0.497, CI = [-0.206, 0.958]

## H2: Hierarchy Specificity (Semantic vs Non-Hierarchy Control)

- **Paired t-test**: t = -4.748, p = 0.0032
- Mean semantic-hierarchy absorption = 0.235
- Mean non-hierarchy control absorption = 0.331
- **Assessment**: REJECTED

## H3: Robustness Across tau_fs

| tau_fs | Pearson r | 95% CI Lower | 95% CI Upper |
|--------|-----------|--------------|--------------|
| 0.03 | 0.463 | -0.389 | 0.981 |
| 0.01 | 0.468 | -0.394 | 0.981 |
| 0.05 | 0.471 | -0.379 | 0.979 |

- **Assessment**: INCONCLUSIVE

## Per-Architecture Scores

| Architecture | First-Letter | Semantic-Hierarchy | Non-Hierarchy Control |
|--------------|-------------:|-------------------:|----------------------:|
| BatchTopK | 0.547 | 0.359 | 0.398 |
| GatedSAE | 0.008 | 0.188 | 0.379 |
| JumpRelu | 0.281 | 0.230 | 0.348 |
| MatryoshkaBatchTopK | 0.204 | 0.203 | 0.333 |
| PAnneal | 0.012 | 0.064 | 0.131 |
| Standard | 0.026 | 0.352 | 0.416 |
| TopK | 0.576 | 0.250 | 0.311 |
| Random | 0.030 | 0.175 | 0.233 |

## Random-SAE Control

- First-letter: 0.030
- Semantic-hierarchy: 0.175
- Non-hierarchy control: 0.233

## Figures Generated

- `fig1_architecture_ranking.png`
- `fig2_firstletter_vs_semantic_scatter.png`
- `fig3_semantic_vs_nonhierarchy.png`
- `fig4_tau_fs_robustness.png`
- `fig5_gpt2_replication.png`
