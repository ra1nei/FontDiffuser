<div align="center">

# FontDiffuser-CL: Cross-Lingual Font Generation (Latin ↔ Hanzi)

**Enhanced FontDiffuser with Cross-Lingual Style Contrastive Refinement (CL-SCR)**

</div>

<p align="center">
  <img src="figures/logo.png" width="400" alt="FontDiffuser Logo">
</p>

<div align="center">

[![Thesis Project](https://img.shields.io/badge/Project-Graduation%20Thesis-purple.svg)](https://github.com/yourusername/your-repo)
[![Method](https://img.shields.io/badge/Method-CL--SCR-blue)](https://arxiv.org/abs/2312.12142)

</div>

## 📖 Introduction

This repository implements the core contribution of my graduation thesis: **Cross-Lingual Style Contrastive Refinement (CL-SCR)**.

While the original **FontDiffuser** (AAAI 2024) achieves state-of-the-art results in monolingual generation (Chinese $\rightarrow$ Chinese), it struggles significantly with **Cross-Lingual** tasks due to the large topological gap between script systems.

**Our Goal:**
To bridge the gap between **Latin scripts** and **Hanzi (Chinese characters)**. We propose **CL-SCR**, which aligns style features across these distinct script systems, enabling the model to transfer complex styles from Hanzi to Latin and vice versa without losing legibility.

## 🚀 Key Improvements

| Feature | Original FontDiffuser | **Ours (FontDiffuser-CL)** |
| :--- | :--- | :--- |
| **Core Module** | SCR (Style Contrastive Refinement) | **CL-SCR (Cross-Lingual SCR)** |
| **Scope** | Monolingual (Chinese Only) | **Cross-Lingual (Latin $\leftrightarrow$ Hanzi)** |
| **Challenge** | Handling complex styles | **Handling topological gaps between scripts** |
| **Data** | Paired data preferred | **Unpaired / Mixed-script data** |

## 🖼️ Method: CL-SCR

The **CL-SCR** module introduces a novel projection mechanism during the style feature extraction phase. This allows the model to:
1.  Decouple the "content structure" (e.g., the letter 'A' vs. character '永').
2.  Project the "style features" into a shared latent space.
3.  Apply the style of a Hanzi character to a Latin letter (and vice versa) accurately.

## 📊 Main Results (Latin ↔ Hanzi)

### 1. Hanzi Style $\rightarrow$ Latin Content
*(Place an image here: Input is a Chinese Calligraphy character, Output is Latin alphabet in that style)*
![Hanzi to Latin](figures/vis_hanzi_to_latin.png)

### 2. Latin Style $\rightarrow$ Hanzi Content
*(Place an image here: Input is a stylized Latin letter, Output is Chinese character in that style)*
![Latin to Hanzi](figures/vis_latin_to_hanzi.png)

## 🛠️ Installation

```bash
# Clone the repository
git clone [https://github.com/yourusername/FontDiffuser-CL.git](https://github.com/yourusername/FontDiffuser-CL.git)
cd FontDiffuser-CL

# Create environment
conda create -n fontdiffuser-cl python=3.9 -y
conda activate fontdiffuser-cl

# Install dependencies
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url [https://download.pytorch.org/whl/cu117](https://download.pytorch.org/whl/cu117)
pip install -r requirements.txt