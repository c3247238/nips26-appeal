# Related Work

## Diffusion language models and the move from model validity to inference-time engineering

Early discrete diffusion work, including *A Reparameterized Discrete Diffusion Model for Text Generation* (2023), established that non-autoregressive denoising can support competitive text generation, but it did not settle how DLMs should be evaluated once attention shifts from model existence to inference behavior. Recent DLM papers increasingly emphasize test-time engineering rather than basic feasibility. Representative examples include LLaDA-family work such as *LLaDA 1.5* (2025), acceleration-oriented methods such as *DPad* (2025), early-stopping or early-commit approaches such as *Diffusion Language Models Know the Answer Before Decoding* (Prophet, 2025), and more aggressive efficiency lines such as *DLM-One* (2025). This literature establishes an important backdrop for our study: the field now has many ways to alter denoising behavior without retraining, but much less consensus on how to audit small downstream gains once test-time choices become the main source of movement.

## Uncertainty-guided control is not the same as validated intervention

The most relevant neighboring line comes from uncertainty and calibration. Recent work outside the DLM setting uses confidence-like signals to steer decoding, critique, or compute allocation, including *ActCab + CoDec* (2024), *Don’t Think Twice! Over-Reasoning Impairs Confidence Calibration* (2025), *Confidence over Time* (2026), and *COREA* (2026). These papers are useful because they separate observer quality from intervention quality: a signal may be predictive without licensing a strong controller claim. That distinction is central to our paper. We borrow the caution rather than the method. In the current audited slice, precomputed entropy helps identify vulnerable samples and therefore functions as a risk marker, but the stronger sham-control comparison blocks any stronger statement that entropy-guided targeting is itself validated.

## Sampler-centric and audit-centric evaluation

A second neighboring line asks whether apparent quality gains should be attributed to the denoiser, the sampler, or the surrounding runtime. The most direct example is *Is Your Diffusion Sampler Actually Correct?* (2026), which argues that aggregate performance can mask sampler-induced error and that controlled evaluation must separate those effects. Our contribution is aligned with that diagnosis but narrower in scope. We do not provide a new general-purpose sampler evaluation framework. Instead, we show how a reviewer-facing evidence bundle can keep a plausible small-gain story from being written up too aggressively. The important object is not a new theorem or a new controller family, but a claim-to-asset lineage that exposes where the story stops.

## Surveys and protocol gap

Recent surveys on diffusion language models catalog model families, inference optimizations, and open problems, but they do not yet resolve the protocol question raised by our workspace: what is the minimum evidence structure needed before a localized gain deserves method-forward language? The available surveys are strong on taxonomy and trend mapping, especially around efficient inference, multimodal expansion, and trustworthy DLM issues, yet the gap between runtime engineering and reviewer-safe reporting remains noticeable. Our paper occupies that gap. Relative to the mainline DLM literature, the novelty is not a stronger revision policy. Relative to uncertainty-guided decoding work, the novelty is not a new observer. Relative to sampler-evaluation work, the novelty is not broader formalism. The novelty is a documented negative case in which a stronger sham control materially changes the interpretation of an otherwise publishable-looking result.

## Position of this paper

Taken together, the surrounding literature suggests two lessons. First, test-time DLM gains are now easy enough to produce that stronger control logic matters as much as the intervention itself. Second, predictive uncertainty signals should not be conflated with validated intervention rules. Our paper follows both lessons. It positions `CARD-84` as an audited intervention arm inside a narrow evidentiary frame, not as a candidate winner to be scaled up. The resulting contribution margin is deliberately modest: a minimal audit template for small-gain DLM revision claims, demonstrated through an audited negative case rather than a new positive decoding method.

<!-- FIGURES
- None
-->
