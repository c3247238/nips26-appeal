# Iteration 4 Literature Note

## Focus

本轮 literature_search 不做泛化型综述，而是围绕 iteration 3 reflection 中确定的 reviewer-facing gap 做定向补强：

1. 给 training-free / test-time DLM revision 找到更扎实的领域锚点。
2. 给 “signal is informative but does not validate the intervention” 这一论点补 calibration / failure-prediction 文献。
3. 给 runtime-lineage / sampler attribution 补 reviewer-friendly 的 related-work 支撑。
4. 给 manuscript 的最小 bibliography enrichment 提供一组可直接落到 `references.bib` 的候选。

## arXiv track: DLM / revision / test-time scaling

### 1. Foundational discrete diffusion language modeling

- **Zheng et al., 2023 — _A Reparameterized Discrete Diffusion Model for Text Generation_**
  - arXiv: https://arxiv.org/abs/2302.05737
  - Why it matters:
    - 适合作为早期 discrete diffusion text generation 的基础锚点。
    - 可以支撑引言里“DLM 已从可行性走向 inference-time engineering”的前史。

- **Lou et al., 2023 — _Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution_ (SEDD)**
  - arXiv: https://arxiv.org/abs/2310.16834
  - Why it matters:
    - 是离散 diffusion LM 的另一条强基线主线。
    - 若 Related Work 要更像 reviewer-ready package，SEDD 比只列 RADD 更完整。

### 2. Current DLM inference / test-time scaling landscape

- **Bai et al., 2026 — _Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models_**
  - arXiv: https://arxiv.org/abs/2602.01842
  - Web confirmation: arXiv listing surfaced directly in search.
  - Why it matters:
    - 明确说明 discrete diffusion LM 也在走 test-time scaling 路线，而不是只有 autoregressive LLM 在做。
    - 可用于支撑本文“small-gain + inference-time compute reallocation 已经成为真实 reviewer 背景”的论点。

- **Lu et al., 2026 — _Advancing Block Diffusion Language Models for Test-Time Scaling_**
  - arXiv: https://arxiv.org/abs/2602.09555
  - Why it matters:
    - 展示更广义的 block diffusion / adaptive decoding test-time scaling 方向。
    - 有助于把本文放在“test-time intervention landscape”里，而不是只像单一 paper-specific case note。

- **Xia et al., 2026 — _MetaState: Persistent Working Memory for Discrete Diffusion Language Models_**
  - arXiv: https://arxiv.org/abs/2603.01331
  - Web confirmation: arXiv listing surfaced directly in search.
  - Why it matters:
    - 这是 reviewer 很可能会提的 direct neighboring line：通过持久化跨步 memory 改善 dLLM。
    - 即使本文不做 direct experiment，也应在 discussion 中交代：MetaState 属于带训练/微调的 architecture augmentation，与本文 training-free audit template 的约束不同。

### 3. Guidance / control in discrete diffusion

- **Schiff et al., 2024 — _Simple Guidance Mechanisms for Discrete Diffusion Models_**
  - arXiv: https://arxiv.org/abs/2412.10193
  - Why it matters:
    - 直接覆盖“guidance / classifier guidance 如何迁移到 discrete diffusion”。
    - 可用来说明本文不是在提出新的 guidance mechanism，而是在审计一个 training-free revision claim 的 attribution boundary。

### 4. Attribution / sampler-centric caution

- **_Is Your Diffusion Sampler Actually Correct? A Sampler-Centric Evaluation of Discrete Diffusion Language Models_**, 2026
  - arXiv: https://arxiv.org/abs/2602.19619
  - Web confirmation: arXiv listing surfaced directly in search.
  - Why it matters:
    - 这是目前最贴近本文 “runtime / sampler / method attribution 要分开” 的相关工作。
    - 非常适合支持本文的 reviewer-facing runtime-lineage note：headline 结果不能只看 accuracy，还要看 sampler / backend / execution contract。

## arXiv track: calibration / uncertainty / failure prediction

### 1. Calibration methods need not improve downstream trust decisions

- **Zhu et al., 2023 — _Rethinking Confidence Calibration for Failure Prediction_**
  - arXiv: https://arxiv.org/abs/2303.02970
  - Why it matters:
    - 论文核心发现就是：常见 calibration method 对 failure prediction 可能无效甚至有害。
    - 这和本文当前的结论高度同构：observer-side entropy / confidence signal 可以 informative，但不自动转化成 successful controller。
    - 建议在 discussion 里直接用来支撑 “risk marker != validated targeting rule”。

- **Deng et al., 2023 — _Towards A Unified View of Answer Calibration for Multi-Step Reasoning_**
  - arXiv: https://arxiv.org/abs/2311.09101
  - Why it matters:
    - 给 multi-step reasoning 中 answer calibration 的 taxonomy 和 unified view。
    - 可以支持本文在 related work 中把 calibration 放回 “post-processing / answer calibration family” 的正确位置，而不是把它写成已验证的 control principle。

### 2. Training-free guided decoding via confidence-like signals

- **_Enhancing Language Model Factuality via Activation-Based Confidence Calibration and Guided Decoding_**, 2024
  - arXiv: https://arxiv.org/abs/2406.13230
  - Why it matters:
    - 可用作“confidence-like observer signal 被用于 decoding / guidance”的近邻工作。
    - 但本文应明确区分：相关工作说明 observer signal 可被消费，不说明该 signal 在当前 DLM revision setup 中已经被验证为 causal controller。

## Web confirmations and reviewer-facing anchors

通过 Web 搜索补到的最有用锚点不是“更多二手综述”，而是以下几篇当前 reviewer 很可能认识或主动搜索的 arXiv 页面：

- Prism — https://arxiv.org/abs/2602.01842
- Sampler-centric evaluation — https://arxiv.org/abs/2602.19619
- MetaState — https://arxiv.org/abs/2603.01331

这些页面足够作为最小 primary-source anchors，便于后续把 `references.bib` 从“可编译”提升到“reviewer 看到不会觉得太薄”。

## What to cite next in the manuscript

如果下一轮只允许补最小引用集，优先级建议如下：

1. **RADD (2302.05737)**  
   用于 foundational discrete diffusion LM background。

2. **SEDD (2310.16834)**  
   用于补齐离散 diffusion LM 的核心主线，避免只引用自家相邻方法。

3. **Prism (2602.01842)**  
   用于 test-time scaling on dLLMs。

4. **Sampler-Centric Evaluation (2602.19619)**  
   用于 runtime / sampler attribution caution。

5. **MetaState (2603.01331)**  
   用于最直接 neighboring work / reviewer comparison point。

6. **Rethinking Confidence Calibration for Failure Prediction (2303.02970)**  
   用于支撑 “calibration can hurt or fail at downstream trust decisions”。

7. **Towards A Unified View of Answer Calibration for Multi-Step Reasoning (2311.09101)**  
   用于把 calibration/answer post-processing 放回统一 related-work 框架。

8. **ActCab+CoDec (2406.13230)**  
   用于 observer-signal-guided decoding 的近邻支撑。

## Direct writing implications for iteration 4

### 1. Abstract / introduction

- 加一句 scoped caveat：本文证据是 `n=100 audited slice` 的 bounded audit，而不是 benchmark-level population estimate。
- 加一句 literature-facing justification：在 dLLM test-time scaling / revision 已经进入 small-gain regime 的背景下，sham-control-driven reinterpretation 本身就是有价值的结果。

### 2. Related work

建议重写成三段，而不是平铺罗列：

1. **Discrete diffusion LM foundations and inference engineering**  
   RADD, SEDD, LLaDA 1.5, DPad, Prophet, Prism.

2. **Guidance / calibration / uncertainty as observer-side signals**  
   ActCab+CoDec, Unified View of Answer Calibration, Rethinking Confidence Calibration for Failure Prediction.

3. **Attribution and execution-envelope caution**  
   Sampler-Centric Evaluation, plus brief mention of MetaState as a trained neighboring alternative.

### 3. Discussion / limitations

- 明确写：本文不是提出新的 controller family，也不主张 entropy-guided revision 已被验证。
- 明确写：stronger sham control 改写 active-control gain 的解释，这一负面结果本身对 small-gain dLLM evaluation 有价值。

## Bottom line

这轮 literature_search 的结论很明确：下一轮不缺“更多想法”，缺的是**更像 reviewer 认识的 related-work scaffold**。  
最值得补的不是新实验，而是把本文放进下面这条清晰链条里：

`discrete diffusion LM -> test-time scaling / guidance -> calibration / failure prediction caution -> sampler/runtime attribution -> bounded negative-case contribution`

只要这条链写顺，配合 runtime-lineage note 与 artifact-release statement，当前 submission package 就有机会从 `7.2` 继续往 `8+` 推。
