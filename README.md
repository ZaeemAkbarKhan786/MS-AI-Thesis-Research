# Trajectory-Aware Hybrid Efficiency Framework for Diffusion Transformers (HE-DiT)

> **Official Implementation** for the paper *"Trajectory-Aware Hybrid Efficiency Framework for Diffusion Transformers"* (Submitted to FIT Conference). 
> 
> **Authors:** Zaeem Akbar Khan, Prof. Dr. Basit Raza, Qaiser Farooq, Esha Javed, Saba Zafar, Hikmat Ullah.
> **Affiliation:** Department of Computer Science, COMSATS University Islamabad, Pakistan.

---

## 📄 Overview

Diffusion Transformers (DiTs) have established a new benchmark for high-fidelity generative imagery, but their practical deployment is constrained by prohibitive computational and memory costs during iterative inference. Existing acceleration strategies optimize only a single dimension and treat the denoising trajectory as uniform. 

This repository contains the implementation of **HE-DiT (Trajectory-Aware Hybrid Efficiency Framework)**, which dynamically adjusts model capacity according to the specific stage of the diffusion trajectory. By unifying flow alignment, stage-specific structural compression, and spatial token routing, HE-DiT delivers a Pareto-optimal operating point:
* **91.0% reduction** in total generation FLOPs.
* **73.4% reduction** in per-image latency.
* **94.8% reduction** in trainable parameters.
* Maintains competitive generative quality (FID: 50.00).

---

## 🏗️ Framework Architecture

HE-DiT extends a pretrained DiT baseline (DiT-S/8) with four sequentially interconnected modules:

### A. Diff2Flow Alignment
Rescales the curved diffusion time ($t_{diff}$) into a linearized flow time ($t_{flow}$). This shifts the prediction target from noise to velocity, enabling larger solver step sizes and reducing inference steps from **20 to 6**.

### B. Trajectory Segmentation
Partitions the trajectory into three semantic stages based on informational complexity:
* **$\mathcal{T}_1$ [0.0, 0.3]:** Semantic Formation stage (global structure)
* **$\mathcal{T}_2$ [0.3, 0.7]:** Content Filling stage (object-level geometry)
* **$\mathcal{T}_3$ [0.7, 1.0]:** Refinement stage (high-frequency texture)

### C. Stage-Specific Learnable Structural Pruning
Uses Gumbel-Softmax estimator to learn three stage-specific pruning masks ($\mathbf{M}_1, \mathbf{M}_2, \mathbf{M}_3$). Instead of static pruning, it keeps ~80% blocks active in $\mathcal{T}_1$, ~60% in $\mathcal{T}_2$, and aggressively prunes to ~40% in $\mathcal{T}_3$.

### D. Spatial Dynamic Token Routing
Within active blocks, a lightweight router scores every spatial token. Only the **top-$k$ (50%)** most informative tokens are selected for dense attention, bypassing background tokens via residual connections.

> **Parameter-Efficient Fine-Tuning:** The pretrained backbone is frozen and quantized to 4-bit precision (QLoRA), while only the Timestep-Dependent LoRA (TD-LoRA) adapters, pruning masks, and spatial routers remain trainable.

---

## 📊 Experimental Results

**Dataset:** Tiny ImageNet-200 (Stratified subset of 20,000 images, $64 \times 64$ resolution).

### Comprehensive Quantitative Comparison
*(Direct comparison between the fully trained DiT baseline and the proposed HE-DiT)*

| Metric | Baseline | HE-DiT (Proposed) | $\Delta$ Change |
| :--- | :---: | :---: | :---: |
| **Inference Steps** | 20 | 6 | **-70.0%** |
| **FLOPs per Pass (GF)** | 0.23 | 0.07 | **-70.0%** |
| **Total FLOPs (GF)** | 4.53 | 0.41 | **-91.0%** |
| **Latency per Image (ms)** | 16.75 | 4.45 | **-73.4%** |
| **Throughput (img/s)** | 59.70 | 224.53 | **+276.1%** |
| **Peak VRAM (GB)** | 1.34 | 1.34 | **+0.0%** |
| **Total Parameters (M)** | 33.52 | 33.75 | **+0.7%** |
| **Trainable Parameters (M)** | 33.52 | 1.73 | **-94.8%** |
| **FID Score ($\downarrow$)** | 60.00 | 50.00 | **-16.7%** |

### Per-Module Efficiency Contribution

| Mod. | Mechanism | Primary Saving |
| :---: | :--- | :--- |
| **A** | Flow-matching ODE reformulation | -70% steps (3.3$\times$ fewer passes) |
| **B** | Stage-aware control signal | Enables Modules C, D |
| **C** | Gumbel-Softmax block pruning | -40% block FLOPs |
| **D** | Top-$k=50\%$ token routing | -75% attention FLOPs |
| **-** | QLoRA + rank-8 TD-LoRA | -94.8% trainable params |

---

## 🛠️ Usage & Setup

### Prerequisites
* PyTorch
* `diffusers`, `transformers`, `peft`

### Implementation Details
* **Backbone:** DiT-S/8 (Hidden dim = 384, 6 heads, 12 blocks, patch size = 8)
* **Optimization:** AdamW (Learning rate $10^{-4}$, weight decay $10^{-4}$, 50 epochs)

*(Add instructions here on how to run your specific training/inference python scripts once uploaded)*

---

## 📖 Citation

If you use this code or framework, please consider citing our work:

```bibtex
@inproceedings{khan2026hedit,
  title={Trajectory-Aware Hybrid Efficiency Framework for Diffusion Transformers},
  author={Khan, Zaeem Akbar and Raza, Basit and Farooq, Qaiser and Javed, Esha and Zafar, Saba and Ullah, Hikmat},
  booktitle={Proceedings of the Frontiers of Information Technology (FIT) Conference},
  year={2026},
  organization={IEEE}
}
