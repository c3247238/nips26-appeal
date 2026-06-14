# Experimental Methodology

## Title
Geometric Factorization Meets Physical Reality: A Systematic Compositional Generalization Benchmark for JEPA World Models

---

## 1. Setup and Environment

### Software Stack
- **Primary model**: LeWorldModel (`le-wm`, GitHub: lucas-maes/le-wm) — ViT-Tiny encoder (~5M params) + transformer predictor (~10M params), SIGReg loss
- **Environments**: DMControl Walker-walk (MuJoCo 3.x) with XML parameter variation; PushT (stable-world-model) as secondary
- **Dependencies**: `dm-control`, `mujoco`, `peft` (LoRA), `necessary-compositionality` (principal angle analysis), `scikit-learn` (probing heads), `torch>=2.0`, `torchvision`
- **Compute**: Local GPU cluster — 8× NVIDIA RTX PRO 6000 Blackwell (96GB VRAM each); experiments limited to `max_gpus=2` per config
- **All experiments**: single GPU unless specified (LeWM ~15M params fits on any GPU); multi-GPU used for parallel runs, not DDP

### Environment Installation (Task: setup)
```bash
# le-wm install
git clone https://github.com/lucas-maes/le-wm && cd le-wm && pip install -e .
# stable-worldmodel for environments
pip install stable-world-model
# dm-control + mujoco
pip install dm-control mujoco
# peft, analysis
pip install peft scikit-learn
```

---

## 2. Benchmark Design: ComPhys-LeWM

### Factor Space
- **Factor 1**: Gravity magnitude — {0.5g, 1.0g, 2.0g} (g = 9.81)
- **Factor 2**: Joint friction coefficient — {0.5x, 1.0x, 2.0x} relative to MuJoCo default
- **Factor 3**: Body density/mass — {0.5x, 1.0x, 2.0x} relative to MuJoCo default
- **Total**: 3×3×3 = 27 unique factor combinations

### Holdout Design (CoGenT-style)
- **Training split**: 18/27 combinations — each factor level appears in ≥6 training combos
- **Holdout split**: 9 combinations — balanced mix of interpolation-type and extrapolation-type
- **Splits**: 3 random CoGenT-style splits to measure sensitivity

### Data Collection
- **Trajectories**: 200 per combination using scripted random walk policy
- **Total**: 200 × 27 = 5,400 trajectories
- **Format**: HDF5, storing: RGB frames (64×64), joint angles, CoM velocity, contact forces, physics parameter labels
- **Storage**: ~0.5GB

---

## 3. Models and Baselines

| Model | Description | Training |
|-------|-------------|----------|
| **LeWM-SIGReg** (primary) | ViT-Tiny + transformer predictor, default SIGReg λ | 18/27 training combos |
| **LeWM-VICReg** (ablation) | Same architecture, VICReg regularizer instead of SIGReg | Same split |
| **LeWM-NoReg** (control) | Same architecture, no regularizer (collapse baseline) | Same split |
| **Oracle** | LeWM-SIGReg trained on all 27 combos (ceiling) | All 27 combos |
| **Matched-volume control** | LeWM-SIGReg trained on 18 random combos (no holdout structure) | 18 random combos |
| **DINO-WM** (architecture comparison) | Frozen DINOv2 encoder + learned transformer predictor | 18/27 training combos |
| **Random encoder** (floor) | Untrained ViT-Tiny + zero-init predictor | — |

**Training hyperparameters**: All models trained for 200 epochs on 18/27 combos, batch size 64, AdamW lr=1e-4, seed 42. Full study: 7 seeds for primary LeWM-SIGReg; 3 seeds for ablations.

---

## 4. Multi-Level Evaluation Protocol

### Level 1 — Probing (Standard)
- **Features**: Frozen encoder embeddings on held-out combinations
- **Probe heads**: Linear regression (main metric) and 2-layer MLP regression
- **Targets**: agent joint angles (DoF), CoM velocity (2D), gravity factor label, friction factor label, body mass label
- **Metric**: R² in-distribution vs. held-out combinations
- **Statistics**: Paired t-test across seeds; bootstrap 95% CI; Cohen's d

### Level 2 — Geometric Analysis
- **Per-factor latent subspaces**: PCA on embeddings grouped by each factor level; extract top-k principal components
- **Principal angle analysis**: `necessary-compositionality` codebase (Uselis et al. 2026); compute principal angles between per-factor subspaces; report mean cosine similarity
- **Displacement vector consistency (parallelogram test)**: For each factor, compute delta-vector (centroid of level-A embeddings → centroid of level-B embeddings) at ≥3 reference points in factor space; consistency = cosine similarity of delta vectors
- **CKA**: Cross-factor CKA as complementary metric

### Level 3 — Interventional Independence Score (IIS)
For each physical concept pair (C, D):
1. Encode trajectory → latent sequence z₁..z_T
2. Identify concept-C direction via trained linear probe
3. Shift z_t by δ (corresponding to known param change, e.g., gravity 1.0g → 2.0g)
4. Roll out modified z_t through predictor
5. Measure: (a) predicted dynamics change correctly for C; (b) predictions for D remain unchanged
6. IIS(C,D) = 1 − |correlation(intervention_on_C, change_in_D_prediction)|
- **Baseline**: random direction shift (should yield IIS ≈ 0.5 for independent concepts)

### Level 4 — Behavioral (Planning)
- **CEM (Cross-Entropy Method) planning** in latent space on held-out combos
- **Tasks**: Walk-reach (target joint configuration); 50 trajectories per holdout combo
- **Metric**: Success rate (%) and latent prediction MSE
- **Comparison**: In-distribution vs. holdout success rate; relative drop

### Level 5 — Physical Regime Analysis
- **Coupling strength estimation**: For each factor pair (f_i, f_j), compute finite-difference approximation of |∂²S/∂f_i∂f_j| from trajectory statistics S (velocity variance, contact frequency, energy dissipation) across training combos
- **Regime classification**: Binary threshold c* — "weakly coupled" vs. "strongly coupled"
- **Test**: Does coupling strength predict holdout failure better than probing R² alone? (partial regression)
- **Cross-model check**: Do strongly-coupled failures persist across LeWM-SIGReg, LeWM-VICReg, DINO-WM?

---

## 5. LoRA Adaptation Diagnostic (H3, H3a)

Applied to held-out combinations only, starting from trained LeWM:
- **Adaptation targets**: (a) encoder Q/V only, (b) predictor Q/V only, (c) both
- **LoRA ranks**: {2, 4, 8, 16}
- **Adaptation data**: {10, 25, 50, 100, 200} trajectories from target held-out domain
- **Metrics**: Probing R² recovery; CEM success rate; samples-to-90%-recovery
- **Baselines**: Zero-shot (no adaptation); head-only fine-tuning; full fine-tuning (ceiling); random LoRA (untrained adapters)
- **Controls**: In-distribution LoRA (verify it doesn't trivially improve already-good in-dist performance)
- **Risk handling**: If LoRA underperforms on ViT-Tiny (expected, known limitation), report as negative result; use full fine-tuning as ceiling; increase rank range to {32, 64} as fallback

---

## 6. SIGReg vs. VICReg Ablation (H2)

- Both trained on same 18/27 split, same seed
- Geometric analysis on both (principal angles, displacement consistency)
- Expected: SIGReg mean cosine similarity < 0.15; VICReg > 0.25
- Correlation analysis: orthogonality vs. holdout accuracy (Pearson r)
- SIGReg lambda sweep (secondary H2a): λ ∈ {0.01, 0.05, 0.1, 0.2} → separate in-distribution and holdout optima

---

## 7. Statistical Analysis

- **Primary test**: Paired t-test (in-distribution vs. holdout R²) across 7 seeds; significance threshold p < 0.05
- **Effect size**: Cohen's d > 0.8 required for H1 confirmation
- **Multiple comparisons**: Bonferroni correction across factor combinations
- **Regression for H4**: OLS regression; incremental R² when adding displacement consistency beyond principal angles
- **No multi-seed cross-validation** (per project constraints); report seed-level variation as bootstrap CI

---

## 8. Expected Visualizations

- **Table 1**: Main results — probing R² and CEM success rate (in-distribution vs. holdout) for all model variants. Columns: Model × Metric; rows: factor combination types.
- **Figure 1**: Heatmap of compositional generalization gap across 9 holdout combinations (3×3 factor grid); color = R² drop (%)
- **Figure 2**: Geometric analysis panel — principal angle matrix (SIGReg vs. VICReg); displacement vector consistency scatter
- **Figure 3**: LoRA adaptation curves — R² recovery vs. number of adaptation trajectories, by adaptation target (encoder / predictor / both) and LoRA rank
- **Figure 4**: Physical regime analysis — coupling strength vs. holdout failure rate; scatter with regression line; model-agnostic concordance across variants
- **Figure 5** (architecture diagram): Overall method pipeline — factor space → data collection → training split → multi-level evaluation → regime analysis
- **Table 2**: IIS matrix (interventional independence score) for all concept pairs; comparison to random-direction baseline

---

## 9. Controls and Mitigations

| Risk | Control |
|------|---------|
| Compositional gap too small on interpolation | Use extrapolation-type holdouts; extend factor range |
| Visual distribution shift confounds compositional failure | Visual oracle control: model with shuffled physics labels |
| Noisy coupling estimation | Binary expert-based regime classification as fallback |
| LoRA underperforms on ViT-Tiny | Report as negative; full fine-tuning as ceiling |
| Correlated concept probe directions | Gram-Schmidt orthogonalization of probes |
| PCA subspace instability at small n | Bootstrapped subspace stability check |
