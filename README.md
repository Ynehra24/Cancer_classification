# A Lightweight Deep Learning Pipeline for Multiclass Cancer Classification Using TCGA RNA-Seq Data

This repository contains the implementation of a lightweight and biologically consistent deep learning pipeline for multiclass cancer classification using RNA-Seq gene expression data from The Cancer Genome Atlas (TCGA).

The work demonstrates that strong predictive performance can be achieved on high-dimensional transcriptomic data using minimal preprocessing and a compact convolutional neural network, without relying on complex or resource-intensive architectures.

---

## Overview

RNA-Seq–based cancer classification is challenged by:
- High dimensionality (p >> n)
- Technical noise and heteroscedasticity
- Class imbalance across cancer types

This project addresses these challenges using:
- Log-based variance stabilization
- Standard feature scaling
- Statistical feature selection with FDR control
- A lightweight 1D Convolutional Neural Network (CNN)

The pipeline is evaluated on **14 TCGA cancer types** and achieves stable, high accuracy with strong generalization.

---

## Dataset

- **Source:** The Cancer Genome Atlas (TCGA)
- **Input format:** CSV files
- **Number of classes:** 14 cancer types
- **Features:** Gene expression values
- **Identifiers:** Ensembl gene IDs and gene symbols
- **Setting:** High-dimensional, low-sample-size (p >> n)

Each sample corresponds to one patient’s tumor tissue transcriptome.

---

## Preprocessing Pipeline

1. **Log Normalization**
   - Applied as: `log2(x + 1)`
   - Reduces right-skewness and heteroscedasticity
   - Improves numerical stability during optimization

2. **Standard Scaling**
   - Z-score normalization per gene
   - Scaling parameters learned from training data only

3. **Feature Selection**
   - ANOVA F-test–based ranking
   - False Discovery Rate (FDR) control using the Benjamini–Hochberg procedure
   - Top **K = 2000** genes retained
   - Final gene list saved as a `.txt` file for reproducibility

4. **Class Imbalance Handling**
   - Class-weighted categorical cross-entropy loss
   - Weights computed inversely proportional to class frequencies
   - No oversampling applied to validation or test sets

---

## Model Architecture

### 1D Convolutional Neural Network (CNN)

- **Input shape:** `(1 × 2000)`
- **Conv1D:** 32 filters, kernel size 5, stride 1, padding 2
- **Batch Normalization**
- **ReLU activation**
- **MaxPooling:** pool size 2
- **Dropout:** p = 0.3
- **Fully Connected Layer:** 128 neurons + ReLU
- **Dropout:** p = 0.3
- **Output Layer:** 14 neurons (softmax)

Although gene features are unordered, 1D convolutions act as nonlinear feature extractors capturing short-range statistical co-activation patterns.

---

## Training Configuration

- **Train/Test split:** 80/20 (stratified)
- **Optimizer:** Adam
- **Learning rate:** 0.0005
- **Loss function:** Weighted CrossEntropyLoss
- **Epochs:** 25
- **Model selection:** Best validation loss

---

## Results

- **Validation accuracy:** ~96.5–97.0%
- **Training accuracy:** ~98.9%
- **Rapid convergence:** >95% validation accuracy within early epochs
- **Minimal overfitting:** Small train–validation gap
- **Balanced per-class performance** despite strong class imbalance

Evaluation includes:
- Accuracy curves
- Loss curves
- Confusion matrix
- Per-class precision, recall, and F1-score

---

## Explainability

To support interpretability and biological relevance:
- SHAP and LIME are used to identify influential genes
- Feature importance is analyzed post-training
- Selected genes can be used for downstream enrichment analysis or biomarker validation

---

## Why CNNs Instead of Transformers?

- Gene expression data are **unordered tabular features**
- Transformers impose inductive biases suited for sequences
- Self-attention introduces unnecessary computational overhead (O(N²))
- CNNs provide:
  - Lower parameter count
  - Faster training and inference
  - Better generalization on medium-sized omics datasets

Empirically, CNNs achieve comparable accuracy with significantly lower computational cost.

---

## Reproducibility

The pipeline emphasizes reproducibility:
- Fixed preprocessing order
- Feature selection with explicit FDR control
- Exported top-K gene list
- Stratified evaluation
- Deterministic training setup (where applicable)

---

## References

All references used in this project correspond exactly to those cited in the accompanying paper and are listed in the manuscript’s bibliography section.

---

## Authors

- **Yatharth Nehra** — Plaksha University  
- **Shubham Goel** — Plaksha University  
- **Sahil Dhiman** — Plaksha University  
- **Anandita Garg** — Plaksha University

---

## License

This project is intended for academic and research use.
