# Experiment Critique

## Overall Assessment

The 105 experiments that WERE executed are well-controlled: identical hyperparameters, unified codebase, proper seeds, fair comparison protocol. However, the experiment plan was only partially executed (Phases 1 and partial Phase 3 of 6 planned phases), and the missing experiments are precisely the ones needed to support the paper's main claims.

## Critical Issues

### 1. ImageNet Experiments Missing (Phase 4)

This is the single most damaging gap. The project constraints explicitly require ImageNet. The methodology allocated 75 GPU-hours for 15 runs. The paper's conclusions generalize beyond CIFAR without any large-scale evidence. Every reviewer will flag this.

With 8x RTX PRO 6000 Blackwell GPUs available, ImageNet/ResNet-50 experiments (90 epochs each) would take ~10-12 hours wall-clock time for 15 runs parallelized across 8 GPUs. This is well within the project's compute budget.

### 2. Non-BN Ablation Missing (Phase 2)

The paper's mechanistic claim (BN enables Phi Invariance even under SGD) REQUIRES the non-BN control. Without it, the BN hypothesis is speculation, not evidence. This is 9 runs (~4 GPU-hours)—trivial to execute.

### 3. PMP-WD and Hill Function Missing (Phase 3)

These experiments test the proposal's core theoretical prediction: the PMP-derived schedule should be optimal. Without testing it, the proposal's main contribution claim is unsupported. 18 runs (~9 GPU-hours).

### 4. Steady-State Formula Validation Missing (Phase 5)

The Phi framework's only mathematical prediction (r* = gamma * E[||g||cos(theta)] / (lambda * E[Phi_t])) is not validated. This would have been the framework's strongest evidence of analytical value. 3 runs (~2 GPU-hours).

### 5. Only 3 Seeds

The proposal planned 9 seeds for "pre-registered falsification criteria." The paper uses 3 seeds, giving 80% power to detect only effects >= 0.7%. For a null-result paper, this is too weak. The equivalence claim would be much stronger with 5-9 seeds allowing detection of 0.3-0.5% effects.

## Major Issues

### 6. VGG-16-BN Coverage Incomplete

VGG-16-BN was tested only with SGD on CIFAR-10. To properly attribute invariance to BN vs. optimizer, the 2x2 design (VGG-16-BN with AdamW, VGG-16 without BN with SGD) is needed. At minimum, VGG-16-BN + AdamW should be tested.

### 7. No Non-AdamW Adaptive Optimizers

CWD reports improvements with Lion and Muon. The paper claims Phi Invariance is due to "adaptive per-parameter scaling," which is a property shared by all adaptive optimizers. Testing Lion or Muon would strengthen the mechanism claim. Without it, the conjecture is AdamW-specific, not "adaptive optimizer"-general.

## Strengths

- Hyperparameter fairness protocol is excellent—no per-method tuning
- Unified codebase with pluggable Phi modulator ensures implementation consistency
- BEM normalization for budget-controlled comparison is a genuine methodological contribution
- The decision to test both AdamW and SGD provides the key contrast that makes the paper interesting
- 200-epoch training with cosine LR schedule is standard and appropriate
