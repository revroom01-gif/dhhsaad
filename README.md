<div align="center">

# 🌕 The Pareidolia Paradox — Monocular Lunar Topography Classification

[![Competition](https://img.shields.io/badge/IEEE%20SIES%20GST-The%20Pareidolia%20Paradox-blue?style=for-the-badge&logo=ieee)](https://ieee.org)
[![Balanced Accuracy](https://img.shields.io/badge/Balanced%20Accuracy-86.40%25-brightgreen?style=for-the-badge&logo=target)](submission/submission.csv)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20%7C%20CUDA%2012.x-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue?style=for-the-badge&logo=python)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![GitHub Stars](https://img.shields.io/badge/⭐%20Leave%20a%20Star-If%20Helpful!-gold?style=for-the-badge)](#-support--acknowledgements)

**Official Submission for "The Pareidolia Paradox" (IEEE SIES GST)**  
*Binary classification of monocular lunar surface patches (crater/depth vs. mound/rise) under changing solar illumination.*

### 🎯 **Balanced Accuracy on 2,000-Image Evaluation Set: 86.4%**

[**The Problem**](#-the-problem) • [**Executive Summary**](#-executive-summary) • [**Key Innovations**](#-key-engineering-innovations) • [**Benchmark Progression**](#-benchmark-progression) • [**Checkpoints**](#-16-model-checkpoints-portfolio-731-mb) • [**Interactive Demo**](#-interactive-demo-cli) • [**Quickstart**](#-quickstart-guide) • [**Team**](#-team)

</div>

---

## 📌 The Problem

The Moon has no atmosphere to diffuse incoming light, so surface shading dominates monocular optical appearance. The exact same topographic patch can read as a crater (depth) or flip into a mound (rise) depending on where the sun is located—the optical illusion of pareidolia after which this competition is named. 

Every image is paired with its precise `sun_azimuth_angle`. The model must utilize this angle; otherwise, it is forced into blind guessing about the direction from which cast shadows originate.

<div align="center">
  <img src="assets/pareidolia_illusion.png" width="700" alt="Crater/mound shading illusion" />
  <p><em>Figure 1: The same topographic feature under opposite solar illumination. Depending on which way the light falls, the crater reads as a depression or flips optically into a dome.</em></p>
</div>

---

## 📌 Executive Summary

On the lunar surface, the absence of atmospheric diffusion causes extreme monocular shading pareidolia: optical inversions where craters (depths) appear as mounds (rises) and vice versa depending on the solar azimuth angle. 

This repository presents an **86.40% Balanced Accuracy** deep learning pipeline that mathematically resolves monocular pareidolia across **2,000 official evaluation images**. Moving beyond standard unpadded CNNs (**58.20% baseline**), this solution introduces:
1. **The 55-Pixel Reflection-Padding Theorem:** Mathematically eliminates 100% of black triangular rotation shortcuts.
2. **3-Channel Differential Physics Tensors:** Extracts real-time Sobel shadow-fall and rim-curvature gradients on GPU.
3. **5 Distinct Architectural Families (16 Checkpoints):** ResNet-34, EfficientNet-B0, EfficientNet-V2-S, ConvNeXt-Tiny, and DenseNet-121.
4. **Target-Domain Semi-Supervised Pseudo-Labeling:** Adapts feature representations to the test set's $158.9^\circ$ solar azimuth shift.
5. **SLSQP Non-Linear Weight Solver:** Directly maximizes Balanced Accuracy over out-of-fold probability matrices with optimal boundary calibration ($\tau = 0.527$).

```text
 58.20%  ──►  74.08%  ──►  78.43%  ──►  80.26%  ──►  81.20%  ──►  84.80%  ──►  85.10%  ──►  86.40%
 Baseline    55px Pad     Focal Loss   EffNet-B0    Full Phase 1   Phase 2 PL     Opt 1       Opt 2 (FINAL)
