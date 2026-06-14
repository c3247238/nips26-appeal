# Notation

- $x$: an input problem instance.
- $y^\*$: the reference answer for $x$.
- $d(x)$: the shared 64-step draft trajectory for instance $x$.
- $H_i(x)$: normalized token entropy at draft position $i$ for instance $x$.
- $F(x)$: the active frontier selected for instance $x$ after draft-time entropy scoring.
- $\rho$: the frontier retention ratio; in the current full-scale study, $\rho \approx 0.1211$.
- $T_{\max}$: the maximum number of frontier-only revision steps after the shared draft; in the current study, $T_{\max}=3$.
- $\tau$: the stopping threshold applied to the masked-frontier entropy ratio; in the current study, $\tau=0.85$.
- $\text{Acc}$: benchmark accuracy.
- $\text{NFE}$: number of function evaluations / denoising evaluations.
- $R$: number of repaired examples in a paired comparison.
- $H$: number of harmed examples in a paired comparison.
- $\Delta_{\text{repair}} = R - H$: net repaired count in a paired comparison.
- $S_{\text{eqQ}}$: speed measured at an equal-quality band.
- $Q_{\text{eqC}}$: quality measured at equal compute.
