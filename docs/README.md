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
  - [3.3 Dataset Preparation](#33-dataset-preparation)
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

