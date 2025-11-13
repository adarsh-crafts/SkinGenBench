# SkinGenBench

[![License](https://img.shields.io/badge/License-MIT-green)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-preprint-red)](#add-link-here)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2B-orange)](https://pytorch.org/)


<p align="center">
  <img src="images\lesion_types.svg" alt="SkinGenBench Teaser" width="400"/>
</p>

> 📢 Official PyTorch implementation of the **SkinGenBench**:  
> **SkinGenBench: Toward Reproducible and Clinically Aligned Synthetic Data Generation for Dermatology**  
> N. A. Adarsh Pritam, Jeba Shiney, Sanyam Jain  
[[Project]](https://github.com/adarsh-crafts/SkinGenBench) | [[Code]](https://github.com/adarsh-crafts/SkinGenBench)
---

## 🌟 Highlights
- **Systematic Evaluation:** First comprehensive comparison of preprocessing complexity impact on GANs (StyleGAN2-ADA) and Diffusion Models (DDPM) for melanoma synthesis.
- **Dual Pipeline Design:** Two distinct preprocessing approaches - basic and advanced (with DullRazor artifact removal) - evaluated across generative architectures.
- **Multi-Metric Assessment:** Quantitative evaluation using FID, Inception Score, and KID, combined with downstream classifier performance analysis.
- **Clinical Focus:** Addresses the critical melanoma class imbalance (11.03% of dataset) with synthetic augmentation achieving 12-20% F1-score improvements.
- **Comprehensive Benchmarking:** Five state-of-the-art classifiers evaluated (ResNet18, ResNet50, VGG16, ViT-B/16, EfficientNet-B0) with interpretability analysis via Grad-CAM.


## 📊 Dataset

<p align="center">
  <img src="images/reals.jpg" alt="Experimental Design" width="800"/>
</p>

Overall experimental design showing dual preprocessing pipelines, generative model training, synthetic data augmentation, and downstream classifier evaluation for melanoma diagnosis.

**Table**: Overview of curated dermatology dataset used in our study. The dataset combines ISIC 2025 (MLK10k) and HAM10000 sources.

| Class | Abbr. | Images | Percentage |
|---|---|---|---|
| Nevus | NV | 7,424 | 52.60% |
| Basal Cell Carcinoma | BCC | 3,026 | 21.43% |
| Benign Keratosis-like | BKL | 1,637 | 11.60% |
| **Melanoma** | **MEL** | **1,563** | **11.03%** |
| Squamous Cell Carcinoma | SCC | 466 | 3.34% |
| **Total** | | **14,116** | **100%** |

**Dataset Sources:**
- [ISIC 2025 (MLK10k)](https://api.isic-archive.com/doi/milk10k/)
- [HAM10000](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

---

## 🧠 Overview

<p align="center">
  <img src="images/overall.png" alt="Methodology Overview" width="600"/>
  <br>
  <em>Figure: General framework of SkinGenBench showing the two preprocessing pipelines (Basic and Advanced), generative model training (StyleGAN2-ADA and DDPM), and evaluation through image quality metrics and downstream classification tasks.</em>
</p>

**Preprocessing Pipelines:**
- **Pipeline A (Basic):** Standard augmentations including random rotations, horizontal/vertical flips, and resizing to 256×256 resolution.
- **Pipeline B (Advanced):** All transformations from Pipeline A plus DullRazor artifact removal algorithm to eliminate hair and ruler marks through morphological black-hat transformation and texture reconstruction.

**Generative Models:**
- **StyleGAN2-ADA:** Adaptive discriminator augmentation, 8-layer mapping network, progressive synthesis from 4×4×256 to 256×256. Trained for 240,000 images with transfer learning from FFHQ dataset.
- **DDPM:** U-Net architecture with self-attention and residual blocks, 500-step diffusion process. Trained for 120 epochs with transfer learning from CELEBA-HQ dataset.

**Image Subset Nomenclature:**

| Source | Basic Preprocessing (BS) | Advanced Preprocessing (AD) |
|--------|--------------------------|------------------------------|
| Ground Truth | BS_GT | AD_GT |
| StyleGAN2-ADA | BS_GN | AD_GN |
| DDPM | BS_DF | AD_DF |

---

## 📄 Publication
**SkinGenBench: Toward Reproducible and Clinically Aligned Synthetic Data Generation for Dermatology**  
N. A. Adarsh Pritam, Jeba Shiney  
[[GitHub]](#add-link-here) | [[PDF]](#add-link-here) | [[Project]](#add-link-here) | [[Results]](#add-link-here)

---

## 📑 Table of Contents
- [Installation](#installation)
- [Model Zoo](#model-zoo)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Citation](#citation)
- [Downloads](#downloads)
- [Acknowledgments](#acknowledgments)
- [License](#license)

---

## Installation

1. **Clone the repository:**
```bash
   git clone https://github.com/adarsh-crafts/SkinGenBench.git
   cd SkinGenBench
```

2. **Create a virtual environment:**
```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
```

3. **Install PyTorch:**
   Install PyTorch following instructions from [PyTorch official site](https://pytorch.org/).

**requirements.txt**
```markdown
torch>=2.0.0
torchvision>=0.15.0
numpy
opencv-python
matplotlib
scikit-learn
scipy
tqdm
h5py
pandas
Pillow
```

---

## Model Zoo
Pretrained Models for StyleGAN2-ADA and DDPM are available here in table below: 

We release pretrained models for StyleGAN2-ADA and DDPM trained on both preprocessing pipelines, you can load these models and use them for independent purposes.

### StyleGAN2-ADA Models
| Configuration      | File          |
|--------------------|------------------|
| Basic Preprocessing (BS_GN)               | [Link](#add-link-here)    |
| Advanced Preprocessing (AD_GN)      | [Link](#add-link-here) | 


### DDPM Models
| Configuration      | File           |
|--------------------|------------------|
| Basic Preprocessing (BS_DF)               | [Link](#add-link-here)    |
| Advanced Preprocessing (AD_DF)               | [Link](#add-link-here)    |


### Classifier Models
| Configuration      | File           |
|--------------------|------------------|
| ResNet-18 (A3 Pipeline)               | [Link](#add-link-here)    |
| ResNet-50 (A3 Pipeline)               | [Link](#add-link-here)    |
| VGG-16 (A3 Pipeline)               | [Link](#add-link-here)    |
| ViT-B/16 (A3 Pipeline)               | [Link](#add-link-here)    |
| EfficientNet-B0 (A3 Pipeline)               | [Link](#add-link-here)    |

---

## Training

Train StyleGAN2-ADA and DDPM with the provided configuration files in each nested directory.


**Training Details:**

| Configuration      | Minimum            | Maximum          |
|--------------------|-------------------|------------------|
| GPU               | NVIDIA RTX 4060 8GB × 1 | NVIDIA L4 22GB × 1    |
| RAM               | 8 GB            | 22 GB           |
| Train Batch (GAN/DDPM)       | 8                 | 16               |
| Train Batch (Classifier)  | 32                | 64               |
| Input Resolution  | 256×256×3         | 256×256×3        |

---

## Evaluation

(Coming Soon!)

---

## Results

<p align="center">
  <img src="images\t-SNE.svg" alt="t-SNE Visualization" width="600"/>
  <br>
  <em>Figure: t-SNE embeddings in 2D visualization for basic-pipeline (BS, left) and advanced-pipeline (AD, right) data showing ground truth (GT), StyleGAN2-ADA (GN), and DDPM (DF) distributions. Euclidean distances between centroids: BS_GT-BS_GN: 2.72, BS_GT-BS_DF: 58.01, AD_GT-AD_GN: 6.76, AD_GT-AD_DF: 69.97.</em>
</p>



**Fréchet Inception Distance (FID)** - Lower is better

| Comparison | Basic Pipeline (BS) | Advanced Pipeline (AD) |
|------------|---------------------|------------------------|
| Real vs StyleGAN2-ADA | 82.75 | 82.59 |
| Real vs DDPM | 191.75 | 187.36 |
| StyleGAN2-ADA vs DDPM | 126.00 | 138.39 |

**Inception Score (IS)** - Higher is better

| Type | Basic Pipeline (BS) | Advanced Pipeline (AD) |
|------|---------------------|------------------------|
| Real Images | 3.82 ± 0.14 | 3.77 ± 0.13 |
| StyleGAN2-ADA | 3.08 ± 0.09 | 2.93 ± 0.10 |
| DDPM | 2.17 ± 0.05 | 2.29 ± 0.06 |

<p align="center">
  <img src="images\model-acc-macf1_mel-f1-roc.png" alt="Classifier Performance" width="600"/>
  <br>
  <em>Figure: Classifier performance metrics across A1-A3 (BS) vs B1-B3 (AD). Top-left: Mean accuracy per model. Top-right: Average Macro-F1 scores. Bottom-left: Per-class F1 scores for melanoma (MEL). Bottom-right: ROC curves showing improved detectability with synthetic augmentation.</em>
</p>

**Mean Accuracy Across Pipelines**

| Model | Pipeline A (Basic) | Pipeline B (Advanced) | Δ |
|-------|-------------------|----------------------|-----|
| ViT-B/16 | **0.8942** | 0.8872 | +0.70% |
| ResNet-50 | **0.8958** | 0.8886 | +0.72% |
| VGG-16 | **0.8748** | 0.8636 | +1.12% |
| EfficientNet-B0 | **0.8596** | 0.8494 | +1.02% |
| ResNet-18 | **0.8358** | 0.8282 | +0.76% |

**Average Macro-F1 Scores**

| Model | Pipeline A (Basic) | Pipeline B (Advanced) | Δ |
|-------|-------------------|----------------------|-----|
| ViT-B/16 | **0.8282** | 0.8196 | +0.86% |
| ResNet-50 | **0.8307** | 0.8185 | +1.22% |
| VGG-16 | **0.8072** | 0.7881 | +1.91% |
| EfficientNet-B0 | **0.7832** | 0.7644 | +1.88% |
| ResNet-18 | **0.7362** | 0.7251 | +1.11% |

**Melanoma (MEL) F1-Score Improvements:** Synthetic augmentation improved melanoma detection by **12-20%** across all classifiers, with Pipeline A3 (DDPM + Basic preprocessing) achieving the best results.

<p align="center">
  <img src="images/gradcam.png" alt="Grad-CAM Visualization" width="600"/>
  <br>
  <em>Figure: Grad-CAM visualization of saliency maps produced by ViT-B/16 classifier. Real images (first row), DDPM-generated images (second row), and StyleGAN2-ADA-generated images (third row). ResNet-50 shows more spatially coherent activations compared to ViT-B/16's dispersed attention patterns.</em>
</p>

---

## Citation

If you find this work useful, please cite our paper:
```bibtex
@article{pritam2025skingenbench,
  title={SkinGenBench: Toward Reproducible and Clinically Aligned Synthetic Data Generation for Dermatology},
  author={Pritam, N. A. Adarsh, Shiney Jeba, and Sanyam Jain},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025},
  institution={Alliance University, Bangalore}
}
```

---

## Downloads

We release several offline materials for reproducing the experiments and results (but not limited to). These downloads will also be useful if you want to build on top of this work. Please consider using the citation code above for future work.

[StyleGAN2-ADA Models📥](#add-link-here)

[DDPM Models📥](#add-link-here)

[Classifier Models📥](#add-link-here)

[Preprocessed Datasets📥](#add-link-here)

---

## License

(Coming Soon!)