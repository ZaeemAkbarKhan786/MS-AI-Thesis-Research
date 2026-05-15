# Trajectory-Aware Hybrid Efficiency Framework for Diffusion Transformers

[![University](https://img.shields.io/badge/COMSATS%20University-Islamabad-002060?style=flat-square)](https://comsats.edu.pk)
[![Degree](https://img.shields.io/badge/Degree-MS%20Artificial%20Intelligence-4472C4?style=flat-square)]()
[![Status](https://img.shields.io/badge/Status-MS%20Thesis%20%7C%20Spring%202026-228B22?style=flat-square)]()

---

## 📄 Overview

This repository contains the full implementation of my MS thesis research on making large-scale **Diffusion Transformer (DiT)** models significantly more efficient at inference — without sacrificing image generation quality.

Diffusion Transformers have achieved state-of-the-art results in generative image synthesis, but their repeated execution of deep transformer blocks across dozens of denoising timesteps makes them prohibitively expensive: high floating-point operations (FLOPs), excessive GPU memory (VRAM) consumption, and slow inference latency. Existing optimization efforts address these problems in isolation — step reduction, static pruning, or token routing — and all of them treat the highly dynamic diffusion trajectory uniformly, applying identical model capacity regardless of what each stage of generation actually requires.

This work proposes **HE-DiT (Hybrid Efficient Diffusion Transformer)** — a trajectory-aware framework that fundamentally rethinks how computation is allocated across the generative process, achieving a **~89% reduction in total FLOPs** and a **16.7% improvement in FID** over the unoptimized baseline, while using only **5.1% trainable parameters**.

---

## 🔬 Research Contribution & Novelty

The core novelty of this work is the observation that **not all timesteps in a diffusion trajectory are computationally equivalent**. Early timesteps form global semantic structure and require high model capacity; middle timesteps build content and object form; late timesteps merely refine fine-grained texture and need far fewer active layers and tokens. Existing methods ignore this fundamental asymmetry.

This framework introduces a **"morphing architecture"** — a DiT that dynamically expands and contracts its own depth and width at runtime based on the active semantic stage of the denoising trajectory. This is achieved through four tightly integrated modules:

---

## 🏗️ Proposed Framework: Four Modules

### Module A — Diff2Flow Alignment
Converts the baseline DDPM noise-prediction objective into a **flow-matching velocity-prediction** formulation, linearizing the generative ODE. This reduces the required inference steps from **20 DDPM steps down to 6 Euler steps** — a 3.3× throughput gain that directly multiplies the benefit of every per-step saving downstream.

### Module B — Trajectory Segmentation
Analytically partitions the flow time interval `t ∈ [0, 1]` into three semantically distinct stages using gradient sensitivity analysis of transformer block activations:
- **Stage 1 — Semantic** (`t ∈ [0.0, 0.3)`): Global structure formation → High capacity needed
- **Stage 2 — Content** (`t ∈ [0.3, 0.7)`): Object and content formation → Moderate capacity
- **Stage 3 — Refinement** (`t ∈ [0.7, 1.0]`): Texture refinement → Minimal capacity needed

### Module C — Stage-Specific Structural Pruning (Depth Optimization)
Introduces three **learnable binary pruning masks** `M = {M₁, M₂, M₃}`, one per trajectory stage. Each mask decides which transformer blocks remain active during that stage. Masks are learned end-to-end via **Gumbel-Softmax continuous relaxation**, enabling differentiable discrete decisions without manual block selection. The result: aggressively pruned late-stage architecture (keep ~40% of blocks) and a near-complete early-stage architecture (keep ~80%) — reducing VRAM footprint dynamically during inference.

### Module D — Spatial Dynamic Token Routing (Width Optimization)
Inspired by DyDiT, a lightweight **spatial router network** scores each image patch token and selects only the Top-K most informative tokens for full attention computation. Background and already-converged uniform regions are skipped entirely. With a **50% token retention ratio**, attention FLOPs drop by 75% per block (since attention cost scales as O(k²) vs O(L²)). Combined with Module C, this yields **~85% reduction in per-pass compute**.

### Efficient Fine-Tuning: QLoRA + TD-LoRA
The entire framework is fine-tuned without from-scratch pretraining using **QLoRA** (4-bit NF4 quantization of the frozen DiT backbone) and **Timestep-Dependent LoRA (TD-LoRA)** adapters — making the approach accessible without datacenter-scale GPU resources. Only **5.1% of total parameters** are trainable.

---

## 📊 Experimental Results

All experiments were conducted on **Tiny ImageNet-200** (20,000-sample stratified subset, 64×64 resolution) on a single **NVIDIA Tesla T4 GPU (15.6 GB VRAM)** for 50 training epochs.

### Image Quality (FID — Lower is Better)

| Model | Training | Inference Steps | CFG Scale | FID ↓ |
|---|---|---|---|---|
| Baseline DiT | Untrained | 20 (DDPM) | 4.0 | 60.0 |
| **HE-DiT (Proposed)** | **50 epochs** | **6 (Flow)** | **4.0** | **50.0** |
| **Improvement** | — | — | — | **−16.7%** |

### Full Efficiency Comparison: Baseline DiT vs. HE-DiT

| Metric | Baseline | HE-DiT (Proposed) | Δ Change |
|---|---|---|---|
| **Inference Steps** | 20 | 6 | −70.0% ✅ |
| **Latency per Image (ms)** | 16.75 ms | 4.45 ms | −73.4% ✅ |
| **Throughput (img/s)** | 59.70 | 224.53 | +276.1% ✅ |
| **FLOPs per Forward Pass (GFLOPs)** | 0.23 | 0.07 | −70.0% ✅ |
| **Total FLOPs — Full Generation (GFLOPs)** | 4.53 | 0.41 | −91.0% ✅ |
| **Composite Cost Score** | 35.12 | 3.17 | −91.0% ✅ |
| **Peak VRAM (GB)** | 1.34 | 1.34 | ≈ Parity |
| **Trainable Parameters** | 33.52 M (100%) | 1.73 M (5.1%) | −94.8% ✅ |
| **FID Score (↓ better)** | 60.0 | 50.0 | −16.7% ✅ |

### Per-Module Efficiency Contribution

| Module | Mechanism | Primary Saving | Cascade Effect |
|---|---|---|---|
| **A — Diff2Flow** | DDPM → Flow ODE conversion | −70% inference steps (20→6) | 3.3× throughput; multiplies all per-step savings |
| **B — Segmentation** | Stage-aware resource allocation | Enables C & D | Prerequisite for dynamic modules |
| **C — Structural Pruning** | Gumbel-Softmax block masks (ρ₁=0.20, ρ₂=0.40, ρ₃=0.60) | −40% block FLOPs (κ̄=0.60) | Reduces per-pass GFLOPs by ~40% |
| **D — Token Routing** | Top-k=50% spatial token selection per stage | −75% attention FLOPs (k²=0.25) | Combined with C: −85% per-pass FLOPs |
| **QLoRA + TD-LoRA** | 4-bit NF4 + rank-8 LoRA + timestep gating | ~85% fewer trainable params | 4× weight memory reduction; T4-compatible |
| **Total (A+B+C+D)** | Multi-dimensional hybrid | **~89% total FLOP reduction** | FID: 60.0 → 50.0 (−16.7%) |

---

## 🛠️ Technical Stack

| Component | Technology |
|---|---|
| Base Model | DiT-S/8 (Peebles & Xie, 2023) |
| Dataset | Tiny ImageNet-200 (20K subset, 64×64) |
| Fine-Tuning | QLoRA (4-bit NF4) + TD-LoRA (rank-8) |
| Flow Alignment | Rectified Flow / Diff2Flow |
| Pruning | Gumbel-Softmax learnable binary masks |
| Token Routing | Top-K spatial router (DyDiT-inspired) |
| Training Objective | Flow MSE loss + FLOPs sparsity regularization |
| Memory Optimization | Gradient checkpointing across unrolled trajectory |
| Hardware | NVIDIA Tesla T4 (15.6 GB VRAM) |
| Framework | PyTorch 2.10 + CUDA 12.8 |
| Quantization | bitsandbytes (NF4 4-bit) |
| Adapters | PEFT / LoRA |

---

## 📁 Repository Structure

```
├── efficient_dit_hybrid_final.ipynb   # Full implementation (Google Colab)
├── README.md                          # This file
├── outputs/
│   ├── samples/                       # Generated image comparisons
│   │   ├── baseline_vs_proposed.png   # Side-by-side generation comparison
│   │   ├── per_class_comparison.png   # Per-class image results
│   │   └── final_samples.png          # Final HE-DiT sample grid
│   ├── he_dit_checkpoint.pt           # Trained model checkpoint
│   └── diagnostics/                   # Training curves & efficiency plots
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install torch torchvision timm einops diffusers accelerate
pip install transformers bitsandbytes peft pytorch-fid thop
```

### Run in Google Colab

1. Open `efficient_dit_hybrid_final.ipynb` in Google Colab.
2. Set the runtime to **GPU (T4 recommended)** via `Runtime → Change runtime type`.
3. Run cells sequentially — each cell is self-contained and clearly commented.
4. All outputs are saved to `/content/outputs/`.

### Key Cells

| Cell | Purpose |
|---|---|
| Cell 0 | Environment check (GPU/CUDA verification) |
| Cell 2 | Global configuration (all hyperparameters) |
| Cell 16 | Module A — Diff2Flow alignment |
| Cell 18 | Module B — Trajectory segmentation analysis |
| Cell 20 | Module C — Structural pruning masks |
| Cell 22 | Module D — Spatial dynamic token routing |
| Cell 26 | QLoRA + TD-LoRA fine-tuning setup |
| Cell 31 | Image generation comparison (Baseline vs HE-DiT) |
| Cell 33–34 | Quantitative evaluation: FID, FLOPs, VRAM, Throughput |
| Cell 35 | Cost reduction dashboard visualization |
| Cell 40 | Full diagnostics: training curves + efficiency charts |

---

## 📚 Key References

1. Peebles, W., & Xie, S. (2023). Scalable diffusion models with transformers. *ICCV 2023*, 4199–4209.
2. Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models. *NeurIPS*, vol. 33.
3. Schusterbauer, J., et al. (2025). Diff2Flow: Training flow matching models via diffusion model alignment. *CVPR 2025*.
4. Fang, Y., et al. (2025). TinyFusion: Diffusion transformers learned shallow. *CVPR 2025*.
5. Zhang, Q., et al. (2026). DyDiT: Dynamic token routing for efficient diffusion transformers. *CVPR 2026*.
6. Liu, S., et al. (2024). ShortDF: Shortened diffusion processes for efficient image generation. *ICML*, vol. 235.
7. Dettmers, T., et al. (2023). QLoRA: Efficient finetuning of quantized LLMs. *NeurIPS*, vol. 36.
8. Lipman, Y., et al. (2023). Flow matching for generative modeling. *ICLR 2023*.

---

## 👤 Author & Academic Information

| | |
|---|---|
| **Name** | Zaeem Akbar Khan |
| **Registration Number** | CIIT/FA24-RAI-019/ISB |
| **Degree** | MS Artificial Intelligence |
| **University** | COMSATS University Islamabad, Islamabad Campus |
| **Department** | Department of Computer Science |
| **Supervisor** | Prof. Dr. Basit Raza |

---

## 📜 License

This repository is submitted in partial fulfillment of the requirements for the degree of **Master of Science in Artificial Intelligence** at COMSATS University Islamabad. All rights reserved © 2026 Zaeem Akbar Khan.
