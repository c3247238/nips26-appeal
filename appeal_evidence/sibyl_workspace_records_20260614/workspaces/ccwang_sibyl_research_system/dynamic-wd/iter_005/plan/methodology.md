# Iteration 5 Experiment Methodology
## When Does Dynamic Weight Decay Help? — Unified Framework Analysis Under AdamW

**Prepared by**: Sibyl Planner Agent (Iteration 5)
**Date**: 2026-03-18
**Context**: Iteration 4 established ResNet-20/CIFAR baseline (42 runs, 3 seeds), VGG-16-BN pilots
(10 epochs), and SGD comparisons. This iteration addresses the core deficiency: **experimental
scope limited to ResNet-20, no ImageNet, no NoBN ablation, no ρ sweep, VGG full runs missing**.

---

## 1. Prior Evidence Summary

### Completed Data (Iterations 1–4)
- **ResNet-20 / CIFAR-10 / AdamW**: 7 methods × 3 seeds = 21 runs, 200 epochs each. DONE.
- **ResNet-20 / CIFAR-100 / AdamW**: 7 methods × 3 seeds = 21 runs, 200 epochs. DONE.
- **ResNet-20 / CIFAR-10+100 / SGD**: 7 methods × 3 seeds × 2 datasets = 42 runs. DONE.
  - Key finding: SGD shows significantly larger spread (~0.9%) vs AdamW (~0.05–0.1%) on CIFAR-10.
  - 18.3× effect size ratio (SGD/AdamW) — but confounded by ρ mismatch (SGD ρ≈0.005 vs AdamW ρ≈0.5).
- **VGG-16-BN / CIFAR-10 / AdamW pilots**: 3 methods × 1 seed × 10 epochs. DONE (feasibility confirmed).
  - CWD_hard is 2.3× slower than constant (504s vs 219s per 10 epochs).

### Evidence-Based Decisions
- VGG pilot confirmed code is ready; full 200-epoch runs needed.
- ResNet-20 ρ≈0.5 (default AdamW wd=5e-4): falls in "Regime I" of the trichotomy hypothesis.
- NoBN architecture (replacing BN with Identity) is not yet implemented — required to test ℓ∞ path.
- ImageNet data not present locally; must use HuggingFace streaming or torchvision download.

---

## 2. Research Hypotheses Being Tested

| Hypothesis | Experiment | Falsification Criterion |
|-----------|-----------|------------------------|
| H5-1: Phi-invariance holds at ImageNet scale | P0-ALPHA: ResNet-50/ImageNet | AdamW spread < 0.3% across 4 WD methods |
| H5-2: BN is NOT required for WD invariance (ℓ∞ path sufficient) | P0-1: ResNet-20-NoBN | NoBN spread ≤ 0.5% (strong result: ℓ∞ sufficient) |
| H5-3: WD spread is monotone in ρ = ‖g‖/‖w‖ | P0-2: ρ sweep (0.05, 0.5, 5.0) | Spread(ρ=5.0) > Spread(ρ=0.5) > Spread(ρ=0.05) |
| H5-4: SGD-AdamW effect ratio collapses at matched ρ | P0-3: matched-ρ SGD | Ratio drops from 18.3× to < 3× when ρ matched |
| H5-5: VGG-16-BN shows same invariance as ResNet-20 | P0-4: VGG-16-BN full | VGG spread < 0.5% under AdamW |

---

## 3. Experimental Setup

### 3.1 Hardware
- 8× RTX PRO 6000 Blackwell Server Edition (98 GB VRAM each, local).
- Currently 6/8 GPUs partially occupied; GPUs 1, 7 are most free.
- All experiments run with `compute_backend: local`, `max_gpus: 4` default (expandable to 8).

### 3.2 Shared Training Config (CIFAR experiments)
- **Optimizer**: AdamW (lr=1e-3, betas=[0.9, 0.999], ε=1e-8) unless noted
- **LR schedule**: Cosine annealing from lr to 0 over total epochs
- **Batch size**: 128 (CIFAR); 256/GPU (ImageNet)
- **Epochs**: 200 (CIFAR), 90 (ImageNet/ResNet-50)
- **Seeds**: 42, 123, 456 (all full experiments)
- **WD base**: λ = 5×10⁻⁴ (CIFAR), λ = 1×10⁻⁴ (ImageNet, targeting ρ≈0.1)
- **Code base**: `iter_004/exp/code/` — reuse directly, extend for new architectures/datasets
- **Diagnostics**: weight_norm, gradient_norm, ρ_per_layer, alignment cosine, CSI, AIS, BEM, sign_flip_rate

### 3.3 WD Methods
| Method | Description | Phi(t, w, g) |
|--------|-------------|-------------|
| constant | Fixed WD λ | 1.0 |
| cosine_schedule | WD decays cosine from λ to 0 | cos schedule |
| cwd_hard | Binary sign-alignment mask | sign(g·w)⁺ |
| swd | Gradient-norm scaled WD | ‖g‖/‖g‖_mean |
| half_lambda | WD at λ/2 (CWD ablation) | 0.5 |
| random_mask | Random binary mask (CWD ablation) | Bernoulli(0.5) |
| no_wd | Zero WD | 0.0 |

### 3.4 Key Metric: Phi Spread
```
phi_spread = max(best_test_acc_{methods}) − min(best_test_acc_{methods})
```
**Interpretation threshold**: spread < 0.5% → WD method-invariant; > 1% → method matters.

---

## 4. Experiment Tasks

### P0-ALPHA: ImageNet / ResNet-50 — Highest Priority

**Goal**: Large-scale external validation; determines paper's final quality tier.

**Setup**:
- Model: torchvision ResNet-50 (25.6M params)
- Dataset: ImageNet-1K via torchvision (download required, ~150 GB)
- Methods: constant, cwd_hard, cosine_schedule, no_wd (4-method minimal set)
- Seeds: 42, 123, 456 → 12 runs total
- Epochs: 90; LR schedule: cosine + 5-epoch warmup; bf16 mixed precision
- WD: λ = 1×10⁻⁴; batch 256/GPU; multi-GPU DataParallel on 2 GPUs

**Phased strategy**:
1. Phase 1: seed=42, 4 methods, GPUs 0–3 (4 GPU DP per run) → ~2–3h → infra validation
2. Phase 2: seed=123, 4 methods, GPUs 4–7 → ~2–3h
3. Phase 3: seed=456, 4 methods → ~2–3h (if resource allows)

**New code required**: `train_imagenet.py`, `data_imagenet.py` (torchvision ImageNet loader), models via torchvision.

**Diagnostics**: Record per-layer ρ heatmap (50 layers × 90 epochs × 4 methods) — this is the novel visualization promised in the paper.

---

### P0-1: NoBN Ablation — ResNet-20 without BatchNorm

**Goal**: Test ℓ∞ path independent of BN scale-invariance confound.

**Setup**:
- Model: ResNet-20-NoBN (all BatchNorm replaced with `nn.Identity`)
- Dataset: CIFAR-10
- Optimizer: AdamW lr=5×10⁻⁴ (reduced; NoBN needs smaller lr for stability)
- Methods: constant, cwd_hard, no_wd (3-method core set)
- Seeds: 42, 123, 456 → 9 runs
- Epochs: 200; gradient clipping (max_norm=1.0) for stability; warmup 10 epochs

**New code required**: Add `resnet20_nobn` to `models.py`.

**Theory predictions**:
- ℓ∞ path: NoBN spread < 0.5% (BN not needed for invariance)
- BN necessary: NoBN spread > 1% (BN is required mechanism)

---

### P0-2: ρ Sweep — Trichotomy Regime Validation

**Goal**: Directly test whether ρ = ‖g‖/‖w‖ governs WD method sensitivity.

**Setup**:
- Model: ResNet-20
- Dataset: CIFAR-10
- Optimizer: AdamW
- **New ρ values** (ρ ≈ λ·(step_scale)):
  - ρ=0.05: achieved by λ=5×10⁻⁵ (10× smaller than default)
  - ρ=0.5: existing data from iter_003 (λ=5×10⁻⁴, default)
  - ρ=5.0: achieved by λ=5×10⁻³ (10× larger, risk of divergence → fallback ρ=2.0 at λ=2×10⁻³)
- Methods: constant, cwd_hard, half_lambda, no_wd (4 methods)
- Seeds: 42, 123, 456; Epochs: 200
- **Stability**: gradient clipping (max_norm=5.0) for ρ=5.0 case; early stopping if loss > 5.0

**Existing data reuse**: ρ=0.5 is already in iter_003 (21 runs, AdamW). Only add ρ=0.05 and ρ=5.0.

**Theory predictions**:
- Spread(ρ=0.05) < 0.1% — Regime I, deep inhibition
- Spread(ρ=0.5) ≈ 0.05–0.1% — Regime I/II boundary (existing data)
- Spread(ρ=5.0) > 1.0% — Regime II/III, methods matter

---

### P0-3: Matched-ρ SGD Comparison

**Goal**: Eliminate ρ confound in the 18.3× SGD/AdamW effect ratio.

**Problem**: Existing SGD used lr=0.1, λ=5×10⁻⁴ → ρ_SGD ≈ 0.005. AdamW ρ ≈ 0.5. 100× mismatch.

**Setup**:
- Model: ResNet-20
- Dataset: CIFAR-10 and CIFAR-100
- **New SGD config**: lr=0.01, λ=5×10⁻³, momentum=0.9 → ρ_SGD ≈ 0.5 (matched to AdamW)
  - Validation: check ρ_matched_sgd ≈ ρ_adamw ± 20% in first 10 epochs
- Methods: constant, cwd_hard, no_wd (3-method set matching the key comparison)
- Seeds: 42, 123, 456; Epochs: 200

**Existing data preserved**: iter_003/04 SGD (ρ=0.005) still valid for historical comparison.

---

### P0-4: VGG-16-BN Full Experiments

**Goal**: Second architecture validation (270K→15M params, 55× scale); break single-arch limitation.

**Setup**:
- Model: VGG-16-BN (existing code, confirmed working)
- Dataset: CIFAR-10
- Methods: constant, cwd_hard, half_lambda, cosine_schedule, swd, random_mask, no_wd (7 methods)
- Seeds: 42, 123, 456 → 21 runs
- Epochs: 200; Batch: 128; AdamW lr=1e-3, λ=5×10⁻⁴
- **CWD_hard timing**: 2.3× slower per epoch → plan ~2.8h per run; dispatch with 6–7 GPU parallelism

---

### P1: Seed Extension for Statistical Robustness

**Goal**: Add seeds 123, 456 for critical ResNet-20 configs where iter_003 only has limited data.

Currently ResNet-20/CIFAR iter_003 has all 7 methods × 3 seeds = 21 runs complete. This is sufficient.
**P1 is deferred** unless additional seeds are needed for TOST equivalence testing.

---

## 5. Execution Timeline (8× GPU Parallelism)

```
Wave 1 [T+0h → T+2h]:  P0-1 + P0-2 + P0-3 (small CIFAR tasks, 3-way parallelism)
  GPU 0: NoBN-constant  seed_42,123,456  (sequential, ~30 min each)
  GPU 1: NoBN-cwd_hard  seed_42,123,456  (~50 min each)
  GPU 2: NoBN-no_wd     seed_42,123,456  (~20 min each)
  GPU 3: rho0.05-constant, cwd_hard, half_lambda, no_wd  seed_42  (~35 min each)
  GPU 4: rho5.0-constant, cwd_hard, half_lambda, no_wd  seed_42  (~45 min each; watch divergence)
  GPU 5: SGD-matched_rho CIFAR-10 seed_42,123,456 × 3 methods (~25 min each)
  GPU 6: SGD-matched_rho CIFAR-100 seed_42,123,456 × 3 methods (~30 min each)
  GPU 7: rho0.05 seeds 123,456; rho5.0 seeds 123,456

Wave 2 [T+2h → T+8h]:  P0-4 VGG-16-BN full (21 runs, GPU 0-6 dispatch)
  GPU 0-6: VGG constant/cosine_schedule/half_lambda/swd/random_mask/no_wd (faster methods)
  GPU 7 + extras: VGG cwd_hard (2.8h per run; last priority)

Wave 3 [T+8h → T+14h]:  P0-ALPHA ImageNet Phase 1+2
  GPU 0-3: ImageNet seed=42, 4 methods (DataParallel, 2 GPU per run)
  GPU 4-7: ImageNet seed=123, 4 methods

Wave 4 [T+14h → T+17h]:  P0-ALPHA ImageNet Phase 3 (if GPUs freed)
  GPU 0-3: ImageNet seed=456, 4 methods

Total: ~17h wall-clock; ~100-120 GPU-hours
```

---

## 6. Gate Decisions

| Gate | Trigger | Decision |
|------|---------|---------|
| Gate 1 | NoBN + ρ sweep + matched-ρ SGD complete | Determine trichotomy claim strength; update Section 5 narrative |
| Gate 2 | VGG full complete | Confirm/deny cross-architecture generalization; begin data analysis and figure generation |
| Gate 3 | ImageNet Phase 1 (seed=42) complete | Validate infrastructure; set final paper claim scope (scenarios A/B/C/D) |
| Final | All P0 done | Enter writing/revision phase |

---

## 7. Expected Visualizations

| Figure | Type | Source Data | Paper Section |
|--------|------|------------|--------------|
| Fig 1: Method comparison table (3 architectures) | Table | All experiments | Experiments |
| Fig 2: Phi Spread vs ρ curve | Line plot | ρ sweep (P0-2) + existing ρ=0.5 | Core Analysis |
| Fig 3: NoBN vs BN Cohen's d comparison | Bar chart | P0-1 + iter_003 ResNet-20 | Ablation |
| Fig 4: Matched-ρ SGD vs AdamW effect ratio | Bar chart | P0-3 + iter_003 SGD | Core Analysis |
| Fig 5: ImageNet per-layer ρ heatmap | Heatmap | P0-ALPHA (50 layers × 90 epochs) | Novel Contribution |
| Fig 6: Cross-scale Phi Spread comparison | Multi-panel | ResNet-20 + VGG + ResNet-50 | Generalization |
| Fig 7: Training dynamics curves | Multi-line | All architectures, constant vs no_wd | Appendix |

---

## 8. Code Infrastructure Needed (New in Iter 5)

| File | Purpose | Based On |
|------|---------|---------|
| `train_imagenet.py` | ResNet-50 ImageNet training with bf16, per-layer diagnostics | `iter_004/exp/code/train_unified.py` |
| `data_imagenet.py` | ImageNet torchvision DataLoader with standard augmentation | `iter_004/exp/code/data.py` |
| `models_extended.py` | Add: `resnet20_nobn`, torchvision `resnet50` wrapper | `iter_004/exp/code/models.py` |
| `run_wave1.sh` | Batch launcher for Wave 1 (NoBN, ρ sweep, matched-ρ SGD) | New |
| `run_wave2.sh` | Batch launcher for VGG full experiments | New |
| `run_wave3.sh` | Batch launcher for ImageNet experiments | New |
| `analyze_iter5.py` | Cross-architecture analysis, figure generation | `iter_004/exp/code/analyze_sgd_baseline.py` |
| `verify_figures.py` | Data-paper consistency checker (zero-tolerance) | New |

---

## 9. Risk Mitigation

| Risk | Probability | Mitigation |
|------|------------|-----------|
| NoBN training unstable | High | Use lr=5×10⁻⁴, clip gradients at 1.0, 10-epoch warmup |
| ρ=5.0 divergence | High | Early stopping at loss>5; fallback to ρ=2.0 (λ=2×10⁻³) |
| ImageNet data unavailable locally | Medium | Use `torchvision.datasets.ImageNet` with download=True; ~150 GB |
| VGG CWD_hard very slow (2.8h/run) | Confirmed | Use GPU 7 for long-running tasks; accept 15h total wall clock |
| GPU contention from other workloads | Medium | Respect `gpu_free_threshold_mb=2000`; use GPUs 1 and 7 first |
| All P0 experiments failing again | Low (hard gate) | This iteration: orchestrator blocks writing stage until P0 gates pass |
