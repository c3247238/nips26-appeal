# Literature Survey Report

**Research Topic**: Cross-domain compositional generalization of LeWorldModel: evaluate whether physical concepts learned in training environments transfer to unseen combinations via zero-shot probing and LoRA adaptation, identify failure modes and generalization boundaries.
**Survey Date**: 2026-04-10
**arXiv Search Keywords**: ["LeWorldModel JEPA world model generalization", "compositional generalization world model physical concepts transfer", "JEPA joint embedding predictive architecture zero-shot transfer", "LoRA fine-tuning world model adaptation new environment", "zero-shot probing world model latent space physical properties", "V-JEPA video prediction physical understanding generalization probing", "representation learning latent disentanglement physical properties generalization unseen combinations", "compositional generalization domain transfer physical concepts reinforcement learning"]
**Web Search Keywords**: ["LeWorldModel arXiv 2603.19312 JEPA SIGReg generalization", "DINO-WM world model probing latent space physical concepts 2024", "V-JEPA 2 world model zero-shot planning physical understanding 2025", "PLDM world model probing physical state prediction planning", "world model compositional generalization zero-shot probing physical concepts latent space ViT 2024 2025"]

---

## 1. Field Overview

World models — systems that learn internal representations of environment dynamics to support planning and decision-making — have seen rapid advancement following the JEPA (Joint Embedding Predictive Architecture) paradigm introduced by LeCun and collaborators. Unlike generative reconstructive models (VAEs, diffusion models), JEPAs predict future states in abstract latent space, avoiding the computational overhead of pixel-level generation while capturing task-relevant structure. This family of approaches has scaled from image-based I-JEPA to video-based V-JEPA and V-JEPA 2, and now to physically grounded world models like LeWorldModel (LeWM) and DINO-WM.

A core open question in this space is whether physical concepts acquired by a world model in a set of training environments genuinely transfer to unseen configurations — i.e., compositional generalization of physical knowledge. Compositional generalization is the ability to recognize and apply familiar component concepts (e.g., gravity, friction, collision physics) in novel, previously unseen combinations of environments or conditions. While the NLP community has long studied this question, it remains relatively underexplored in the context of physically grounded video world models. The dominant evaluation methodology has been either downstream task success rates or point-by-point probing of latent spaces via linear classifiers, but systematic cross-domain transfer studies across *combinations* of physical factors (e.g., novel object shape + novel gravity + novel friction coefficient) do not yet exist in the JEPA world model literature.

Recent work demonstrates that world models can encode diverse physical properties in their latent spaces (LeWM, DINO-WM, V-JEPA 2), yet several failure modes have been identified: representation collapse in low-diversity environments (LeWM's Two-Room failure), poor generalization when latent actions are scene-specific rather than universal (Olaf-World), and degraded performance when visual context shifts beyond the training distribution. The intersection of compositional generalization theory (which demands linear, orthogonal per-concept representations) with the Gaussian-distributed latent space constraint of LeWM's SIGReg regularizer is theoretically interesting and under-explored: the Gaussian prior may implicitly encourage the linear factorization required for composition, but this has not been empirically confirmed.

---

## 2. Core References

| # | Title | Source | Year | Key Contribution | Limitations |
|---|-------|--------|------|-----------------|-------------|
| 1 | LeWorldModel: Stable End-to-End JEPA from Pixels | arXiv:2603.19312 | 2026 | First end-to-end JEPA from raw pixels; SIGReg enforces Gaussian latent distribution preventing collapse; 15M params, 48x faster than foundation-model-based WMs; linear and non-linear probes on physical state | Fails in low-diversity Two-Room env; latent Gaussian prior may hurt low-complexity tasks; no compositional cross-domain evaluation |
| 2 | DINO-WM: World Models on Pre-trained Visual Features Enable Zero-shot Planning | arXiv:2411.04983 | 2024 | Uses frozen DINOv2 patch embeddings to model environment dynamics; enables zero-shot behavioral planning without reward, demonstrations, or task-specific training; physical state probing as evaluation | Frozen encoder cannot adapt end-to-end; relies on large pre-trained backbone; probing only within training-distribution environments |
| 3 | V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning | arXiv:2506.09985 | 2025 | Large-scale (1M+ hour) video pre-training; action-conditioned fine-tuning (V-JEPA 2-AC) with <62h unlabeled robot video; zero-shot planning on Franka arms in unseen labs | Requires large-scale video data; AC fine-tuning still needed for robotics; generalization boundary not systematically studied |
| 4 | V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning | arXiv:2603.14482 | 2026 | Dense predictive loss, hierarchical self-supervision, multi-modal tokenizers; 20-point improvement in robot grasping over V-JEPA 2 AC | Very recent; limited cross-domain compositional analysis |
| 5 | Probing the Latent World: Emergent Discrete Symbols and Physical Structure in Latent Representations | arXiv:2603.20327 | 2026 | AIM framework: passive quantization probe for V-JEPA 2 latent space; shows physical dimensions (grasp angle, object geometry, motion structure) are statistically separable (chi^2 p<1e-4); action categories share a representational core | Limited to Kinetics-mini; only 3 physical dimensions probed; no cross-domain transfer |
| 6 | Compositional Generalization Requires Linear, Orthogonal Representations in Vision Embedding Models | arXiv:2602.24264 | 2026 | Formalizes geometric conditions for compositional generalization (divisibility, transferability, stability); shows representations must decompose linearly with orthogonal per-concept components; validates on CLIP, SigLIP, DINO | Only evaluates pretrained vision encoders, not world model predictors |
| 7 | Does Data Scaling Lead to Visual Compositional Generalization? | arXiv:2507.07102 | 2025 | Shows compositional generalization driven by data diversity not scale; increased combinatorial coverage forces linear factored representation; pretrained DINO/CLIP show partial this structure | Not evaluated in physical control / world model setting |
| 8 | AdaWorld: Learning Adaptable World Models with Latent Actions | arXiv:2503.18938 | 2025 | Self-supervised latent action extraction from videos; enables efficient adaptation to novel action spaces with limited interactions; strong simulation quality and planning results | Focused on action adaptability, not physical concept transfer |
| 9 | Olaf-World: Orienting Latent Actions for Video World Modeling | arXiv:2602.10104 | 2026 | Sequence-level control-effect alignment (SeqDelta-REPA) for structured latent action space; stronger zero-shot action transfer and data-efficient adaptation | Addresses action space transfer, not physical property compositional generalization |
| 10 | Swing-by Dynamics in Concept Learning and Compositional Generalization | arXiv:2410.08309 | 2024 | Theoretical analysis via structured identity mapping (SIM); non-monotonic test loss during training; concept-centric data structure affects learning speed | Focused on diffusion models, not world models / control |

---

## 3. SOTA Methods and Benchmarks

### Current Best Methods (World Models for Planning)

**Latent JEPA approaches (pixel-based, end-to-end):**
- **LeWorldModel (LeWM)**: ViT-Tiny encoder (~5M) + transformer predictor (~10M); only 2 loss terms (prediction + SIGReg); achieves 18% higher success rate than PLDM on PushT; trains on single GPU in hours
- **PLDM**: End-to-end latent world model, 7-term objective (variance-invariance-covariance), strong physical state probing results but complex to tune
- **V-JEPA 2-AC**: 300M-parameter action-conditioned world model post-trained on 62h robot video; zero-shot on Franka Emika Panda, outperforms Cosmos by 30x in speed

**Pre-trained encoder approaches:**
- **DINO-WM**: Frozen DINOv2 spatial patch embeddings + learned transformer predictor; zero-shot planning via CEM in latent space; SR=0.82 on unseen maze walls; outperforms DreamerV3/IRIS substantially

### Mainstream Evaluation Environments

| Environment | Physical Concepts | Tasks |
|-------------|------------------|-------|
| PushT | Object-agent contact, friction, orientation | Push a T-shaped block to goal pose |
| TwoRoom | Agent locomotion, wall collision, navigation | Navigate from room A to B |
| OGBench-Cube | 6-DOF manipulation, arm kinematics | Pick and place cube |
| Maze (DINO-WM) | Wall collision, spatial navigation | Goal-conditioned maze traversal |
| MetaWorld / DMControl | Robotic arm kinematics, gravity, contact | Various manipulation/locomotion |

### Physical State Probing Evaluation Protocol

Physical probing evaluates latent representations by training linear / non-linear regression heads to predict ground-truth physical state variables (agent position, block position/velocity, end-effector pose) from frozen latent embeddings. Both LeWM and PLDM substantially outperform DINO-WM on linear probing, indicating the end-to-end training better encodes low-level physics. The AIM framework (arXiv:2603.20327) extends this to symbolic discrete probing in V-JEPA 2.

### LoRA Adaptation for World Models

LoRA (Low-Rank Adaptation) has been applied in the 1X World Model Challenge to post-train Wan-2.2 I2V-5B for robot state-conditioned prediction (arXiv:2510.07092), achieving 23.0 dB PSNR. In V-JEPA 2, action-conditioned fine-tuning of the predictor (a form of targeted adaptation) uses <62h data to enable zero-shot robotic planning. No work has yet applied LoRA specifically to probe or adapt the LeWM encoder/predictor for cross-domain physical concept generalization.

---

## 4. Identified Research Gaps

- **Gap 1 — Cross-domain compositional evaluation for JEPA world models**: No prior work systematically evaluates whether combinations of physical factors unseen during training (e.g., new friction × new object shape × new goal configuration) are handled correctly by LeWM or similar JEPA models. DINO-WM evaluates generalization to unseen object shapes or maze walls, but not structured combinations of multiple factors simultaneously.

- **Gap 2 — Link between SIGReg and compositional generalization**: LeWM enforces an isotropic Gaussian latent distribution via SIGReg. Theory (arXiv:2602.24264) shows compositional generalization requires linear, orthogonal per-concept representations. Whether the Gaussian prior implicitly encourages this structure — or competes with it — is unknown. This is a key mechanistic question.

- **Gap 3 — LoRA as an adaptation probe for world models**: Using LoRA to adapt specific parts of the LeWM encoder or predictor to unseen physical domains, and measuring the adaptation efficiency (data efficiency, generalization gap reduction), has not been done. This is a natural experiment: if LoRA can recover generalization from a small set of target-domain samples, it reveals what components encode domain-specific vs. transferable knowledge.

- **Gap 4 — Failure mode taxonomy for JEPA world models**: LeWM fails in the Two-Room low-diversity environment. The mechanism is hypothesized (low-diversity latent makes Gaussian constraint hard to satisfy), but a systematic study of failure conditions across a spectrum of environments — varying diversity, physical complexity, and compositional structure — is absent.

- **Gap 5 — Generalization boundary identification**: There is no framework for characterizing where JEPA world model generalization breaks down in terms of physical concept combinations — i.e., a "generalization boundary" in factor space analogous to work done in vision models (arXiv:2507.07102).

- **Gap 6 — Zero-shot probing of predictor vs. encoder contributions**: Current probing work (arXiv:2603.20327) probes encoder representations only. Whether the predictor's action-conditioning maintains physical plausibility when combined with unseen physical contexts is unexamined.

---

## 5. Available Resources

**Open-source code:**
- [le-wm GitHub](https://github.com/lucas-maes/le-wm): Official LeWorldModel codebase (Lucas Maes et al.); ViT-Tiny encoder + transformer predictor; PyTorch; MIT-compatible
- [DINO-WM project page](https://dino-wm.github.io/): Code and models for DINO-WM (DINOv2-based world model)
- [V-JEPA 2 (Meta AI)](https://ai.meta.com/research/vjepa/): Open-source code and model checkpoints for commercial and research use
- [IBM Scalable Compositional Generalization](https://github.com/IBM/scalable-compositional-generalization): Attribute Invariant Networks for compositional generalization evaluation
- [Necessary Compositionality](https://github.com/oshapio/necessary-compositionality): Code for geometric conditions analysis (CLIP/SigLIP/DINO)
- [Visual Compositional Generalization](https://github.com/oshapio/visual-compositional-generalization): Data diversity experiments

**Datasets:**
- **PushT** (LeWM, DINO-WM, PLDM evaluation): 2D push manipulation; physical state includes agent position + block pose
- **OGBench** (manipulation benchmark): 6-DOF robot with cube; diverse configurations
- **DROID dataset** (V-JEPA 2 fine-tuning): 62h+ of unlabeled robot videos across diverse manipulation
- **DMControl suite**: Continuous control environments with diverse physical configurations
- **MetaWorld**: 50 robotic manipulation tasks; systematic task variation

**Pretrained models:**
- V-JEPA 2 (ViT-H/16, 8B VQA variant): [HuggingFace/Meta AI](https://ai.meta.com/research/vjepa/)
- DINOv2 (frozen encoder in DINO-WM): Standard HuggingFace `facebookresearch/dinov2`
- LeWM (ViT-Tiny, ~15M params): [le-wm GitHub](https://github.com/lucas-maes/le-wm)

---

## 6. Implications for Idea Generation

**High-value directions:**

1. **Cross-domain compositional probing benchmark for LeWM**: Design a systematic benchmark with environments that vary 2-3 physical factors independently (friction, gravity, object shape, mass), train LeWM on a subset of factor combinations, and probe zero-shot transfer to held-out combinations. This directly tests the research topic and fills Gap 1.

2. **LoRA adaptation efficiency as a generalization diagnostic**: Apply LoRA to either the LeWM encoder or predictor (or both) on a small sample of target-domain data. Measure: (a) adaptation efficiency (number of samples needed to recover zero-shot performance), (b) which components adapt (encoder vs. predictor), (c) whether adapting only one component is sufficient. This fills Gap 3 and provides a principled generalization diagnostic.

3. **SIGReg geometry and compositionality**: Measure the linear factorization structure (principal angle analysis, CKA) of LeWM latent space organized by physical factor, and compare to a baseline model without SIGReg. Tests the mechanistic hypothesis linking Gaussian prior to compositional structure (Gap 2).

4. **Failure mode boundary mapping**: Systematically vary environment diversity and physical complexity, measuring probing accuracy and planning success as a function of these variables. Identifies where LeWM breaks down and why (Gap 4, Gap 5).

**Saturated directions (avoid):**
- Generic LoRA fine-tuning of LLMs (not world models) — very crowded
- New JEPA training objectives without compositional evaluation — saturated
- Pixel-reconstruction world models — superseded by latent prediction approaches

**Promising cross-domain analogies:**
- The linear representation hypothesis from vision (Uselis et al., 2026) maps directly to JEPA world models — physical concept dimensions should be linearly orthogonal for compositional generalization
- Meta-RL / few-shot adaptation literature (task context inference in DALI, Meta-DT) provides frameworks for measuring how much domain-specific data is needed for adaptation

---

## 7. Implementation Strategy Recommendations

| Existing Implementation | Match | License | Strategy | Rationale |
|------------------------|-------|---------|----------|-----------|
| [le-wm (LeWorldModel)](https://github.com/lucas-maes/le-wm) | High | MIT-compatible | **Adopt** | Official codebase; directly trains LeWM from pixels; PushT and other environments included; well-maintained |
| [DINO-WM](https://dino-wm.github.io/) | High | Research code | **Adopt** | Key baseline; provides physical state probing protocol and evaluation suite that can be re-used for LeWM probing experiments |
| V-JEPA 2 / V-JEPA 2-AC | Medium | Apache 2.0 | **Extend** | Can serve as a strong baseline comparison; V-JEPA 2-AC action-conditioning mechanism is relevant for cross-domain adaptation experiments |
| [Necessary Compositionality (Uselis et al.)](https://github.com/oshapio/necessary-compositionality) | Medium | MIT | **Adopt** | Evaluation tools for linear factorization (per-concept orthogonality, divisibility score) can be applied directly to LeWM latent space to test compositional structure |
| [IBM Scalable Compositional Generalization](https://github.com/IBM/scalable-compositional-generalization) | Medium | Apache 2.0 | **Compose** | Compositional generalization evaluation framework for supervised vision models; can be adapted to provide standardized holdout split methodology for physical factor combinations |
| LoRA (via `peft` library) | High | Apache 2.0 | **Adopt** | Standard LoRA implementation; apply directly to LeWM ViT encoder attention layers and/or transformer predictor; minimal integration effort |
| DMControl / MetaWorld | High | MIT / BSD | **Adopt** | Standard physical control benchmarks; support parameterization of physical factors (gravity, friction, mass); existing integration with world model codebases |

**Highlight — Reusable evaluation components:**
- **Physical state probing protocol** (from LeWM paper and DINO-WM): Linear and MLP regression heads on frozen latent embeddings → predict ground-truth state variables. Directly reusable with scikit-learn or PyTorch; takes <1h to run.
- **CEM-based planning evaluation** (from DINO-WM): Cross-Entropy Method planning in latent space; goal-conditioned success rate as primary metric. Reusable across environments.
- **AIM symbolic probing** (arXiv:2603.20327): Passive quantization probe for JEPA latents; reusable to probe LeWM physical structure across domains without modifying the encoder.
- **Linear factorization metric** (Uselis et al., arXiv:2602.24264, arXiv:2507.07102): Measures degree of linear orthogonal decomposition per concept; principal angle analysis; ~50 lines of numpy code.
