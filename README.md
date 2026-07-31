<div align="center">

# HE-DiT: Trajectory-Aware Hybrid Efficiency Framework for Diffusion Transformers

**Official implementation of the paper submitted to the 23rd IEEE International Conference on Frontiers of Information Technology (FIT'26)**

[![Conference](https://img.shields.io/badge/FIT-2026-blue.svg)](https://fit.edu.pk)
[![IEEE](https://img.shields.io/badge/IEEE-Xplore%20(pending)-00629B.svg)](https://ieeexplore.ieee.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](#-license)
[![Status](https://img.shields.io/badge/Status-Research%20Preview-orange.svg)](#)

*Dynamically allocating structural depth and spatial attention across the diffusion trajectory — 91% fewer FLOPs, 73% lower latency, 95% fewer trainable parameters.*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Results at a Glance](#-key-results-at-a-glance)
- [Framework Architecture](#️-framework-architecture)
  - [Module A: Diff2Flow Alignment](#a-diff2flow-alignment)
  - [Module B: Trajectory Segmentation](#b-trajectory-segmentation)
  - [Module C: Stage-Specific Learnable Structural Pruning](#c-stage-specific-learnable-structural-pruning)
  - [Module D: Spatial Dynamic Token Routing](#d-spatial-dynamic-token-routing)
  - [Parameter-Efficient Fine-Tuning](#parameter-efficient-fine-tuning)
- [Visual Overview](#-visual-overview)
- [Experimental Results](#-experimental-results)
  - [Comprehensive Quantitative Comparison](#comprehensive-quantitative-comparison)
  - [Per-Module Efficiency Contribution](#per-module-efficiency-contribution)
  - [Training Dynamics](#training-dynamics)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#️-setup--installation)
- [Usage](#-usage)
- [Dataset](#-dataset)
- [Limitations & Future Work](#-limitations--future-work)
- [Citation](#-citation)
- [Authors](#-authors)
- [Acknowledgment](#-acknowledgment)
- [License](#-license)

---

## 📄 Overview

Diffusion Transformers (DiTs) have established a new benchmark for high-fidelity generative imagery, but their practical deployment is severely constrained by prohibitive computational and memory costs during iterative inference. Existing acceleration strategies — step reduction, static pruning, dynamic token routing, and quantization — each optimize a **single dimension** of the problem and treat the denoising trajectory as uniform, ignoring the varying informational complexity across its temporal phases.

This repository contains the official implementation of **HE-DiT (Trajectory-Aware Hybrid Efficiency Framework)**, which dynamically adjusts model capacity according to the specific stage of the diffusion trajectory. By unifying **flow alignment**, **stage-specific structural compression**, and **spatial token routing** into a single architecture, HE-DiT delivers a Pareto-optimal operating point — sacrificing minimal generative fidelity in exchange for dramatic computational savings.

> **Novelty:** HE-DiT is, to our knowledge, the first framework to jointly treat the *temporal*, *structural*, and *spatial* dimensions of the diffusion trajectory within a single unified, trainable pipeline.

---

## 🚀 Key Results at a Glance

| | |
|---|---|
| 🔻 **91.0%** reduction in total generation FLOPs | 4.53 GF → 0.41 GF |
| ⚡ **73.4%** reduction in per-image latency | 16.75 ms → 4.45 ms |
| 📉 **94.8%** reduction in trainable parameters | 33.52M → 1.73M |
| 📈 **276.1%** increase in throughput | 59.70 → 224.53 img/s |
| 🎯 Competitive generative quality | FID 60.00 → **50.00** (lower is better) |
| 💾 No increase in peak VRAM | 1.34 GB → 1.34 GB |

---

## 🏗️ Framework Architecture

HE-DiT extends a pretrained DiT baseline (**DiT-S/8**) with four sequentially interconnected modules, jointly optimized through parameter-efficient fine-tuning.

### A. Diff2Flow Alignment
Rescales the curved diffusion time ($t_{\text{diff}}$) into a linearized flow time ($t_{\text{flow}}$), straightening the probability path between the noise and data distributions. This shifts the prediction target from noise to velocity, enabling larger solver step sizes and reducing inference steps **from 20 to 6** without retraining the backbone from scratch.

### B. Trajectory Segmentation
A sensitivity and activation analysis over the normalized timestep $t \in [0,1]$ partitions the trajectory into three semantic stages, which supply the control signal for Modules C and D:

| Stage | Range | Description |
|---|---|---|
| $\mathcal{T}_1$ — Semantic Formation | [0.0, 0.3] | Global structure formation |
| $\mathcal{T}_2$ — Content Filling | [0.3, 0.7] | Object-level geometry |
| $\mathcal{T}_3$ — Refinement | [0.7, 1.0] | High-frequency texture detail |

### C. Stage-Specific Learnable Structural Pruning
Uses a **Gumbel-Softmax** estimator to learn three stage-specific, differentiable pruning masks ($\mathbf{M}_1$, $\mathbf{M}_2$, $\mathbf{M}_3$) applied to the dense transformer weights. Rather than a single static compression ratio, the learned policy allocates capacity according to stage sensitivity:

- **~80%** of blocks kept active in $\mathcal{T}_1$ (semantic formation — most sensitive)
- **~60%** of blocks kept active in $\mathcal{T}_2$ (content filling — moderate)
- **~40%** of blocks kept active in $\mathcal{T}_3$ (refinement — least sensitive, most aggressively pruned)

### D. Spatial Dynamic Token Routing
Within each active block, a lightweight router network scores every spatial token by predicted relevance. Only the **top-k (k = 0.50)** most informative tokens are selected for dense self-attention; unselected, low-variance background tokens bypass attention entirely via residual connections and are re-scattered to their original spatial positions. Since attention scales as $\mathcal{O}(k^2)$, this yields a **75% reduction in attention FLOPs**.

### Parameter-Efficient Fine-Tuning
The pretrained backbone is **frozen and quantized to 4-bit precision (QLoRA, NF4)**, while only the following remain trainable:
- Timestep-Dependent LoRA (**TD-LoRA**) adapters (rank-8)
- The three stage-specific pruning masks
- The spatial token routers

The training objective combines a **flow-matching velocity loss** with a **FLOPs-constraint sparsity penalty**, and gradient checkpointing is applied across the unrolled trajectory to control memory usage during backpropagation.

---

## 📊 Experimental Results

**Dataset:** Tiny ImageNet-200 — a stratified subset of 20,000 images (100 per class, fixed seed 42), native resolution $64 \times 64$, encoded via a frozen VAE into an $8 \times 8 \times 4$ latent grid (64 spatial tokens).

### Comprehensive Quantitative Comparison
*(Trained DiT baseline vs. proposed HE-DiT)*

| Metric | Baseline | HE-DiT (Proposed) | Δ Change |
|---|:---:|:---:|:---:|
| Inference Steps | 20 | 6 | **−70.0%** |
| FLOPs per Pass (GF) | 0.23 | 0.07 | **−70.0%** |
| Total FLOPs (GF) | 4.53 | 0.41 | **−91.0%** |
| Latency per Image (ms) | 16.75 | 4.45 | **−73.4%** |
| Throughput (img/s) | 59.70 | 224.53 | **+276.1%** |
| Peak VRAM (GB) | 1.34 | 1.34 | +0.0% |
| Total Parameters (M) | 33.52 | 33.75 | +0.7% |
| Trainable Parameters (M) | 33.52 | 1.73 | **−94.8%** |
| FID Score (↓ better) | 60.00 | 50.00 | **−16.7%** |

### Per-Module Efficiency Contribution

| Module | Mechanism | Primary Saving |
|---|---|---|
| A | Flow-matching ODE reformulation | −70% steps (3.3× fewer passes) |
| B | Stage-aware control signal | Enables Modules C and D |
| C | Gumbel-Softmax block pruning | −40% average block FLOPs |
| D | Top-k (k = 50%) token routing | −75% attention FLOPs |
| QLoRA + TD-LoRA | 4-bit NF4 + rank-8 LoRA adapters | −94.8% trainable parameters |

> **Note:** The end-to-end 276.1% throughput improvement (3.76×) reflects the *combined, multiplicative* effect of all four modules acting together — not any single module in isolation.

### Training Dynamics
The Gumbel-Softmax masks converge within 50 epochs to keep ratios close to their respective stage targets:
- $\mathbf{M}_1 \approx 10$ active blocks out of 12 (83.3%, nearest integer to the 80% target)
- $\mathbf{M}_2 \approx 8$ active blocks out of 12 (66.7%, nearest integer to the 60% target)
- $\mathbf{M}_3 \approx 5$ active blocks out of 12 (41.7%, nearest integer to the 40% target)

The joint loss (flow-matching velocity loss + sparsity regularization, $\lambda = 10^{-3}$) exhibits smooth, monotonic convergence. Qualitatively, the spatial router learns to assign higher scores to foreground regions (selected for full attention) while bypassing homogeneous background patches.

---

## 📁 Repository Structure

```
HE-DiT/
├── assets/                  # Figures used in this README (architecture, results, etc.)
├── configs/                 # YAML/JSON training & inference configuration files
├── data/                    # Dataset preparation and preprocessing scripts
├── models/
│   ├── modules/             # Module A–D implementations (Diff2Flow, segmentation, pruning, routing)
│   └── dit_backbone.py      # DiT-S/8 backbone definition
├── training/
│   ├── train.py             # Main training entry point
│   └── losses.py            # Flow-matching + sparsity-regularization losses
├── inference/
│   └── sample.py            # Inference / image generation script
├── checkpoints/              # Saved model checkpoints (not tracked by git)
├── requirements.txt
├── LICENSE
└── README.md
```

> This structure reflects the intended organization of the codebase. Update file/folder names above to match the actual scripts once they are uploaded to the repository.

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (recommended)
- [PyTorch](https://pytorch.org/) 2.0+
- [`diffusers`](https://github.com/huggingface/diffusers)
- [`transformers`](https://github.com/huggingface/transformers)
- [`peft`](https://github.com/huggingface/peft) (for QLoRA / TD-LoRA)
- [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes) (for 4-bit NF4 quantization)

### Installation
```bash
git clone https://github.com/<your-username>/HE-DiT.git
cd HE-DiT
pip install -r requirements.txt
```

---

## ▶️ Usage

> Training and inference commands below are placeholders — update them once the corresponding scripts are added to the repository.

### Training
```bash
python training/train.py \
    --config configs/he_dit_tiny_imagenet.yaml \
    --epochs 50 \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --lr_schedule cosine
```

### Inference / Sampling
```bash
python inference/sample.py \
    --checkpoint checkpoints/he_dit_final.pt \
    --num_steps 6 \
    --topk_ratio 0.5 \
    --output_dir outputs/samples
```

---

## 🗂️ Dataset

- **Dataset:** Tiny ImageNet-200 (200 classes, 500 train / 50 val / 50 test images per class)
- **Subset used:** Stratified sample of 20,000 images (100 per class, fixed seed = 42) for balanced class representation
- **Resolution:** $64 \times 64$ (training images resized to $72 \times 72$, randomly cropped to $64 \times 64$, randomly flipped, lightly color-jittered)
- **Latent encoding:** Frozen pretrained VAE → $8 \times 8 \times 4$ latent grid (64 spatial tokens)
- **Normalization:** Validation images resized and normalized to $[-1, 1]$

### Implementation Details

| Hyperparameter | Value |
|---|---|
| Backbone | DiT-S/8 |
| Hidden dimension | 384 |
| Attention heads | 6 |
| Transformer blocks | 12 |
| Patch size | 8 |
| Quantization | 4-bit NF4 (QLoRA) |
| LoRA rank | 8 (TD-LoRA) |
| Inference steps | 6 (Euler, flow-aligned) |
| Token-keep ratio ($k$) | 0.50 |
| Optimizer | AdamW |
| Learning rate | $10^{-4}$ |
| Weight decay | $10^{-4}$ |
| LR schedule | Cosine annealing |
| Training epochs | 50 |
| Sparsity regularization ($\lambda$) | $10^{-3}$ |

---

## ⚠️ Limitations & Future Work

- FID is currently evaluated on 256 samples; training uses a 20,000-image stratified subset rather than the full Tiny ImageNet-200 corpus.
- Trajectory segmentation thresholds ($\mathcal{T}_1$/$\mathcal{T}_2$/$\mathcal{T}_3$ boundaries) are currently fixed rather than learned.

**Planned directions:**
- Learned (rather than fixed) trajectory segmentation thresholds
- Extension to video diffusion transformers
- Hardware-native sparse attention kernels for deployment

---

## 📖 Citation

If you use this code or framework in your research, please cite our work:

```bibtex
@inproceedings{khan2026hedit,
  title     = {Trajectory-Aware Hybrid Efficiency Framework for Diffusion Transformers},
  author    = {Khan, Zaeem Akbar and Raza, Basit and Farooq, Qaiser and Javed, Esha and Zafar, Saba and Ullah, Hikmat},
  booktitle = {Proceedings of the 23rd IEEE International Conference on Frontiers of Information Technology (FIT)},
  year      = {2026},
  organization = {IEEE}
}
```

---

## 👥 Authors

| Author | Affiliation | Contact |
|---|---|---|
| Zaeem Akbar Khan | Department of Computer Science, COMSATS University Islamabad | zaeemakbar.khan786@gmail.com |
| Prof. Dr. Basit Raza | Department of Computer Science, COMSATS University Islamabad | basit.raza@comsats.edu.pk |
| Qaiser Farooq | Department of Computer Science, COMSATS University Islamabad | qaiserfarooqw285@gmail.com |
| Esha Javed | Department of Computer Science, COMSATS University Islamabad | eshajaved854@gmail.com |
| Saba Zafar | Department of Computer Science, COMSATS University Islamabad | zsaba1134@gmail.com |
| Hikmat Ullah | Department of Computer Science, COMSATS University Islamabad | hikmatwazir222@gmail.com |

---

## 🙏 Acknowledgment

The authors acknowledge the use of AI assistance (such as large language models) strictly for linguistic polishing and manuscript writeup enhancement. The core research idea, mathematical formulation, architectural design, and experimental validation were entirely conceived and executed by the primary author. The authors thank **COMSATS University Islamabad** for computational resources.

---

## 📜 License

This project is released under the [MIT License](LICENSE). See the `LICENSE` file for details.

---

<div align="center">

**⭐ If you find this work useful, please consider starring the repository and citing our paper.**

</div>
