# Continuous Generalized Preference Optimization for Plasma Fusion Foundation Models

[![License: OpenMDW](https://img.shields.io/badge/License-OpenMDW--1.0-blue.svg)](LICENSE.md)
[![Python: 3.11+](https://img.shields.io/badge/Python-3.11%2B-brightgreen.svg)](https://www.python.org/)
[![Status: Double-Blind Review](https://img.shields.io/badge/Status-Double--Blind%20Review-purple.svg)]()

This repository provides the reference implementation of **Continuous Generalized Preference Optimization (CGPO)** for continuous physical foundation models, evaluated on the **TokaMark** benchmark across 14 plasma dynamics tasks on the Mega Ampere Spherical Tokamak (MAST).

---

## Overview

Foundation models applied to complex physical dynamical systems, such as magnetic confinement fusion in tokamaks, frequently encounter an empirical performance plateau under standard Supervised Fine-Tuning (SFT). Mean Squared Error (MSE) objectives minimize conditional expectations E[y | x], which smooths over multi-scale dynamics, localized steep gradients, and transient physical perturbations.

While offline preference optimization frameworks (such as Direct Preference Optimization [DPO] and Identity Preference Optimization [IPO]) have proven effective in discrete domains, adapting them to continuous physical foundation models presents a fundamental mathematical barrier:

> **The Continuous Alignment Challenge:** Canonical preference formulations require evaluating token log-probabilities log π(y | x). While tractable for categorical distributions over discrete vocabularies, sequence log-probabilities are mathematically undefined for continuous deterministic regression heads producing continuous output vectors in ℝᴰ without an associated predictive probability density.

**Continuous Generalized Preference Optimization (CGPO)** resolves this gap by formulating reference-anchored preference margins directly within the foundation model's orthonormal spectral latent space (3D Discrete Cosine Transform, DCT3D). Experimental discharge telemetry serves as the objective ground-truth attractor (y_w), contrasted against model failure modes (y_l).

---

## Theoretical Framework

- **Continuous Preference Margin:** Replaces discrete token log-likelihood ratios with reference-anchored Euclidean distance metrics in spectral embedding space:
  ```text
  M(ŷ; x) = [d(ŷ, y_l) - d(ŷ_ref, y_l)] - [d(ŷ, y_w) - d(ŷ_ref, y_w)]
  ```
  where `d(u, v) = (1/D) ‖u - v‖₂²`. Differentiating the margin yields a constant contrastive vector field `∇_ŷ M = (2/D)(y_w - y_l)` directed from the historical failure anchor `y_l` toward ground truth `y_w`.
- **Continuous IPO Gradient Dynamics:** Continuous DPO exhibits asymptotic gradient saturation (σ(−β·M) → 0 as M > 0), permitting policy drift along unconstrained dimensions in unbounded vector spaces. In contrast, **Continuous IPO** optimizes a regularized quadratic penalty against a finite target margin `M* = 1 / (2β*)`, exerting an active restorative vector field that prevents gradient saturation and reward hacking.
- **Dual-Objective Dynamic Normalization:** Combines the contrastive preference objective on informative error pairs with an isotropic latent reconstruction anchor to maintain global calibration and stabilize adapter updates across sparse preference batches.
- **Error-Slice Pair Filtering:** Restricts pairwise optimization to the [25%, 95%] native-space NRMSE percentile band, filtering out uninformative quiescent intervals at the lower quartile and non-physical diagnostic acquisition artifacts at the extreme upper tail.

---

## Repository Structure

```text
.
├── src/mmt/                           # Dataset-agnostic core library
│   ├── data/                          # Codecs (DCT3D), signal specifications, streaming loaders
│   ├── models/                        # Transformer backbone, modality heads, and output adapters
│   ├── train/
│   │   ├── losses/
│   │   │   └── continuous_gpo.py      # Core CGPO loss functions (Continuous IPO, DPO, margin metrics)
│   │   └── loop_utils.py              # Dual-objective dynamic loss aggregator and training steps
│   ├── eval/                          # Physical space metrics (NRMSE, NMAE, RMSE, MAE)
│   └── examples/
│       └── toy_train.py               # Self-contained toy example (runnable without external data)
├── scripts_mast/                      # Benchmark integration and execution pipelines
│   ├── calibrate_gpo_task.py          # Scale-setting calibration heuristic (beta*)
│   ├── run_finetune.py                # Base supervised fine-tuning entrypoint (Stage 1)
│   ├── run_gpo_finetune.py            # Continuous GPO fine-tuning entrypoint (Stage 2)
│   ├── run_eval.py                    # Benchmark evaluation pipeline
│   ├── run_local_smoke_test.py        # Local workstation smoke test
│   └── configs/mmt/tasks/
│       ├── finetune_tasks.yaml        # Supervised fine-tuning task configurations
│       └── gpo_tasks.yaml             # Calibrated CGPO hyperparameters across all 14 tasks
├── docs/                              # Technical documentation
│   ├── gpo_local_runbook.md           # Guide for local execution and data streaming
│   ├── cgpo_shot_protocol.md          # Shot splitting and data provenance protocol
│   └── tuning_dct3d.md                # Orthonormal 3D DCT spectral representation tuning
├── tests/                             # Unit, protocol, and pipeline test suite
├── pyproject.toml                     # Package dependencies and installation specification
├── LICENSE.md                         # License agreement
└── README.md                          # Repository overview and execution guide
```

---

## Installation

### 1. Environment Setup

```bash
# Recommended Python: 3.11+
conda create -n cgpo-env python=3.11 -y
conda activate cgpo-env
```

### 2. Package Installation

```bash
# Install core package in editable mode
pip install -e .

# (Optional) Install testing and development tools
pip install -e ".[dev]"
```

Verify installation using the self-contained toy training loop (runs immediately without external data):
```bash
python src/mmt/examples/toy_train.py
```

---

## Verification and Smoke Test

The complete forward-backward pass, reference policy anchoring, Continuous IPO margin computation, and adapter updates can be verified locally on a workstation or personal computer in under 2 minutes:

```bash
python scripts_mast/run_local_smoke_test.py
```

This verification procedure tests:
- Loading model output adapters and reference policy checkpoints.
- Continuous IPO quadratic margin evaluation: `M(ŷ; x) = [d(ŷ, y_l) - d(ŷ_ref, y_l)] - [d(ŷ, y_w) - d(ŷ_ref, y_w)]`.
- Dual-objective dynamic normalization: `w_pref · L_pref + w_mse · L_embed_mse`.
- Gradient updates restricted to output adapters while preserving frozen backbone weights.

To execute the unit and integration test suite:
```bash
pytest tests/ -v
```

---

## Experimental Pipeline

The experimental protocol implements the two-stage training procedure described in the paper:

### Stage 1: Base Supervised Fine-Tuning (Baseline Plateau)

Train the model under standard supervised embedding MSE to establish the converged supervised baseline:

```bash
python scripts_mast/run_finetune.py \
  --task task_4-4 \
  --init scratch \
  --model_profile mmt \
  --emb_profile dct3d \
  --tag baseline
```

### Stage 2: Continuous GPO Fine-Tuning (Preference Alignment)

Fine-tune output adapters from the frozen base model using Continuous IPO:

```bash
python scripts_mast/run_gpo_finetune.py \
  --task task_4-4 \
  --loss ipo \
  --model_profile mmt \
  --emb_profile dct3d
```

Key optimization configurations (specified in `scripts_mast/configs/mmt/tasks/gpo_tasks.yaml`):
- **Frozen Parameters:** Transformer backbone and token encoders remain frozen.
- **Trained Parameters:** Modality heads and output adapters (learning rate: `1e-4`).
- **Temperature Scale Heuristic:** `β* = 1 / median(d(y_w, y_l))`.
- **Reconstruction Anchor Weight:** `w_mse = 0.2` on all windows.
- **Pair Filtering Band:** Native NRMSE percentile interval `[25%, 95%]` with per-shot window caps at the 90th percentile.

### Stage 3: Benchmark Evaluation

Evaluate checkpoints against the official held-out test split:

```bash
python scripts_mast/run_eval.py \
  --task task_4-4 \
  --model_source <checkpoint_path_or_run_id>
```

This evaluates physical space metrics (NRMSE, NMAE, RMSE, MAE) across evaluated discharges.

---

## License

This research codebase is distributed under the [OpenMDW License](LICENSE.md).
