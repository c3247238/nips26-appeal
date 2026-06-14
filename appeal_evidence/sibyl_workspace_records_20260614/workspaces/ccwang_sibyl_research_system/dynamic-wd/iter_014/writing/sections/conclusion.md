# 8. Conclusion

The per-layer gradient-to-weight ratio $\rho_t^l = \|g_t^l\| / \|w_t^l\|$ is the shared control variable across four fragmented WD sub-traditions.  We formalize this observation into a PID-style control law parameterized by $(K_p, K_i, K_d)$ that maps FixedWD to open-loop control, CWD to derivative/alignment feedback, SWD to proportional scheduling, and CPR to integral constraint accumulation.  The framework provides a clean taxonomy for understanding *why* each method works: CWD exploits alignment geometry, CPR accumulates constraint violations, and scheduling methods operate on the training trajectory without per-layer feedback.

The empirical validation confirms the taxonomy's value: the control law fits CWD's effective-WD trajectories to 4.71% error and CPR's to 9.57% on 200-epoch CIFAR-10/ResNet-20 data (3 seeds, 72 per-layer traces).  Scheduling-based methods (SWD at 45.81%, DefazioCorrective at 37.56%) resist fitting, honestly delineating the framework's boundary.

UDWDC, our proportional controller closing the $\rho_t$ feedback loop, does not achieve the highest accuracy --- CPR outperforms it on both CIFAR-10 (91.74% vs. 90.15%) and ImageNet (74.74% vs. 69.93%).  UDWDC's contribution is conceptual and methodological: it demonstrates that the control-theoretic formulation is implementable, identifies proportional-only control's instability (CSI = $-$2.41), and motivates integral-augmented designs.

The three standardized metrics --- BEM, CSI, and AIS --- expose method differences invisible to accuracy-only evaluation.  BEM reveals that UDWDC achieves rank-1 budget efficiency despite rank-2 accuracy.  CSI quantitatively identifies instability in UDWDC that accuracy alone does not flag.  AIS confirms that the alignment signal carries information beyond time-polynomial trends ($R^2 < 0.85$ for 81.9% of layer-method combinations).

This work raises a clear question for the field: the unification identifies proportional, integral, and derivative channels for WD control, but no existing method jointly optimizes all three.  CPR's strong results suggest that integral control is the most impactful channel.  Whether a jointly-tuned PID controller, or an adaptive gain schedule, can systematically outperform fixed-gain designs remains open.

<!-- FIGURES
- None
-->
