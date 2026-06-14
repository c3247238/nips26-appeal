# Notation Table

| Symbol | Description |
|--------|-------------|
| $\mathbf{x}$ | Token sequence under denoising |
| $L$ | Sequence length |
| $t \in \{1,\dots,T\}$ | Denoising step index |
| $T$ | Nominal number of denoising steps |
| $\mathbf{x}^{(t)}$ | Sequence state at denoising step $t$ |
| $p_\theta(\cdot \mid \mathbf{x}^{(t)})$ | Model distribution at step $t$ |
| $H_i^{(t)}$ | Token entropy at position $i$ and step $t$ |
| $\hat{x}_i^{(T)}$ | Final committed token at position $i$ |
| $\mathrm{NFE}_{\mathrm{nom}}$ | Nominal number of forward evaluations implied by the method label |
| $\mathrm{NFE}_{\mathrm{act}}$ | Actual number of forward evaluations measured during execution |
| $\Delta_{\mathrm{gap}}$ | Relative gap between nominal and actual compute |
| $\ell$ | End-to-end latency in seconds |
| $\tau$ | Throughput in tokens per second |
| $b$ | Batch size |
| $a_m$ | Task accuracy of method $m$ |
| $d(s)$ | Diagnostic score of signal $s$ |
| $g(s)$ | Control effectiveness of signal $s$ under the tested policy |
| $\mathrm{SC\mbox{-}ECE}$ | Self-consistency expected calibration error |
| $\rho$ | Correlation coefficient |
| $\mathcal{B}_{\mathrm{reason}}$ | Reasoning benchmark family |
| $\mathcal{B}_{\mathrm{code}}$ | Code benchmark family |

The paper uses $\mathrm{NFE}_{\mathrm{act}}$, latency, and throughput as first-class comparison variables. We reserve “observer” for signals evaluated through $d(s)$ and “controller” for intervention policies evaluated through $g(s)$.
