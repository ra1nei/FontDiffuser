<div align="center">

# FontDiffuser-CL: Cross-Lingual Font Generation

**Enhanced FontDiffuser with Cross-Lingual Style Contrastive Refinement (CL-SCR)**

</div>

## Introduction

This repository implements the core contribution of my graduation thesis: **Cross-Lingual Style Contrastive Refinement (CL-SCR)**.

While the original **FontDiffuser** (AAAI 2024) achieves state-of-the-art results in monolingual generation (Chinese ↔ Chinese), it struggles significantly with **Cross-Lingual** tasks due to the large topological gap between script systems.

**Our Goal:** To bridge the gap between **Latin scripts** and **Hanzi (Chinese characters)**. We propose **CL-SCR**, which aligns style features across these distinct script systems, enabling the model to transfer complex styles from Hanzi to Latin and vice versa without losing legibility.

---

## Key Improvements

| Feature | Original FontDiffuser | **Ours (FontDiffuser-CL)** |
| :--- | :--- | :--- |
| **Core Module** | SCR (Style Contrastive Refinement) | **CL-SCR (Cross-Lingual SCR)** |
| **Scope** | Monolingual (Chinese Only) | **Cross-Lingual (Latin ↔ Hanzi)** |
| **Challenge** | Handling complex styles | **Handling topological gaps between scripts** |
| **Data** | Paired data preferred | **Unpaired / Mixed-script data** |

---

## Method: CL-SCR

The **CL-SCR** module introduces a novel projection mechanism during the style feature extraction phase. This allows the model to:
1. Decouple the "content structure" (e.g., the letter 'A' vs. character '永').
2. Project the "style features" into a shared latent space.
3. Apply the style of a Hanzi character to a Latin letter (and vice versa) accurately.

---