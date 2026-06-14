# Critique: Experiments / Results

**Score: 6/10**

**Strengths:**
- Clear table structure
- Good coverage of L0-matched comparison
- Dose-response design is well-presented

**Issues:**
1. **Critical**: Table 1 TopK/Matryoshka dead latent percentages show as "N/A*"—this is a display bug that must be fixed
2. Missing statistical tests (p-values, confidence intervals)
3. No effect sizes (Cohen's d) reported
4. Missing raw data availability statement
5. L0=200 matching data missing for TopK/Matryoshka (they are trained at fixed L0=50)

**Suggestions:**
- Fix dead latent display bug
- Add Welch's t-test p-values for key comparisons
- Report Cohen's d with pooled std
- Add supplementary data link
