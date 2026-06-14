# 2. Related Work

Work on diffusion language models has rapidly expanded from model construction to training-free inference control. Existing methods vary widely in what they modify: some keep the denoising schedule fixed, some reallocate computation across positions, and some introduce auxiliary signals to decide which tokens should be revisited. This diversity has produced strong empirical results, but it has also made comparisons difficult because nominal method labels often obscure runtime asymmetries and heterogeneous intervention scopes.

A second thread studies confidence and calibration signals for language generation. These signals are useful as observers because they help identify where a draft may be unreliable. However, the literature often moves too quickly from diagnostic usefulness to intervention usefulness. The distinction matters: a signal can predict error well without yielding a strong controller when converted into a revision policy. Our paper does not reject observer-side diagnostics; instead, it argues that observer quality and controller gain should be reported as separate empirical objects.

The paper is also closely related to work on evaluation protocol and compute accounting. In many generation settings, realized runtime depends on batchability, implementation path, backend support, and compilation status in addition to nominal step count. Reporting only nominal compute is therefore insufficient when the goal is to compare intervention policies as systems. Our runtime-lineage bundle is a DLM-specific response to that problem: it maps claims to concrete assets and makes execution assumptions auditable.

Finally, our bucket analysis connects to broader work on failure decomposition and recoverability. Aggregate metrics answer whether one method wins on average, but they do not explain whether that gain comes from repairing previously wrong drafts, preserving already correct drafts, or merely leaving most cases unchanged. We treat that decomposition as a first-class research object. Under this view, the contribution of the paper is not another revision heuristic, but a reporting contract for understanding when a reported gain is mechanistically interpretable.

<!-- FIGURES
- None
-->
