# 🧠 AnatoMaskGAN
**GNN-Driven Slice Feature Fusion and Noise Augmentation for Medical Semantic Image Synthesis**

[📂 Repository](https://github.com/noheadwuzonglin/AnatoMaskGAN/tree/main) · [📄 Paper (arXiv:2508.11375)](https://arxiv.org/abs/2508.11375)

---

## 📑 Table of Contents
- [1. Introduction](#1-introduction)
- [2. Features](#2-features)
- [3. Installation & Environment](#3-installation--environment)
  - [3.1 Requirements](#31-requirements)
  - [3.2 Clone the Repository](#32-clone-the-repository)
  - [3.3 Install dependencies](#33-Install-dependencies)
- [4. Usage](#4-usage)
  - [4.1 Training](#41-training)
  - [4.2 Testing / Generation](#42-testing--generation)
  - [4.3 Inference (Deployment)](#43-inference-deployment)
- [5. Model Architecture](#5-model-architecture)
- [6. Experimental Results & Metrics](#6-experimental-results--metrics)
- [7. Quick Start Demo](#7-quick-start-demo)
- [8. Project Structure](#8-project-structure)
- [9. FAQ](#9-faq)
- [10. Contribution Guide](#10-contribution-guide)
- [11. Acknowledgments](#11-acknowledgments)
- [12. License](#12-license)
- [13. References](#13-references)

---

## 1. Introduction
**AnatoMaskGAN** implements the ideas presented in the paper  
*“AnatoMaskGAN: GNN-Driven Slice Feature Fusion and Noise Augmentation for Medical Semantic Image Synthesis”* (arXiv:2508.11375).

This project focuses on medical semantic image synthesis using Generative Adversarial Networks (GANs).  
It introduces novel strategies to improve anatomical realism and inter-slice consistency.

**Key Contributions:**
- 💠 **GNN-Based Slice Feature Fusion:** Captures inter-slice anatomical relationships using Graph Neural Networks (GNNs).
- 🌀 **3D Spatial Noise Injection:** Enhances structure diversity by injecting spatial noise during generation.
- 🧩 **Grayscale–Texture Classifier:** Refines intensity distribution and texture fidelity.
- 🧠 **Cross-Dataset Generalization:** Demonstrated on L2R-OASIS and L2R-Abdomen CT datasets with significant performance gains over SOTA.

---

## 2. Features
- Supports both *Mask → Image* and *Image → Mask* generation tasks.  
- Incorporates **multi-slice GNN fusion** and **3D noise augmentation**.  
- Optional **texture classifier** for perceptual quality improvement.  
- Unified training, testing, and inference pipelines.  
- YAML-based configuration for flexible experiment setup.  
- Open-source under the **MIT License**.

---

## 3. Installation & Environment

### 3.1 Requirements
```bash
Python >= 3.8
PyTorch >= 1.12
CUDA >= 10.2  (recommended)

```

### 3.2 Clone the Repository
```bash
git clone https://github.com/noheadwuzonglin/AnatoMaskGAN.git
cd AnatoMaskGAN

```

### 3.3 Install dependencies
```bash
pip install -r requirements.txt
# or manually install
pip install numpy scipy matplotlib tqdm scikit-image networkx

```

## 4. Usage

### 4.1 Training
```bash
python train.py --opt options/train.yml
```
Common arguments:
| Argument               | Description                        |
| ---------------------- | ---------------------------------- |
| `--dataset`            | Dataset name (e.g., L2R-OASIS)     |
| `--use_noise3d`        | Enable 3D noise injection          |
| `--use_gnn_fusion`     | Enable slice fusion via GNN        |
| `--texture_classifier` | Enable texture classifier          |
| `--save_dir`           | Directory for checkpoints and logs |




### 4.2 Testing / Generation
```bash
python test.py --opt options/test.yml --model_path checkpoints/latest_net_G.pth
```
-Results (images, masks, metrics) will be saved under the results/ directory.


### 4.3 Inference (Deployment)
```bash
python create.py --opt options/create.yml --input_path my_masks/ --output_path my_images/
```
-Generates new medical images from given semantic masks.

## 5. Model Architecture
AnatoMaskGAN consists of the following core modules:
-Generator (G): Takes a mask and neighboring slice features to synthesize the target image.
-Discriminator (D): Distinguishes real vs. generated images.
-GNN-Fusion Module: Models inter-slice dependency using a graph convolutional structure.
-3D Noise Injection: Adds spatial perturbations to enhance structure variation.
-Grayscale–Texture Classifier: Improves texture and intensity realism.
Loss functions include adversarial loss, reconstruction loss (L1/L2), texture loss, and perceptual structure loss (e.g., LPIPS).

---

## 6. Experimental Results & Metrics
As reported in the paper:
| Dataset       | PSNR ↑       | SSIM ↑     | Improvement |
| ------------- | ------------ | ---------- | ----------- |
| L2R-OASIS     | **26.50 dB** | 0.857      | +0.43 dB    |
| L2R-AbdomenCT | 25.47 dB     | **0.8602** | +0.48 %     |
Ablation studies confirm that removing any module (GNN, Noise, Classifier) degrades performance across all metrics.

---
