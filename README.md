# 🧠 Uncertainty Quantification in Machine Learning for Breast Cancer Diagnosis

This repository contains code and analysis for applying **Uncertainty Quantification (UQ)** methods in deep learning models to the **Breast Cancer Wisconsin Diagnostic Dataset**. The goal is to build reliable models that not only predict accurately but also express how confident they are in those predictions — which is crucial in healthcare applications.

## 📁 Repository Structure


## 📊 Dataset

- **Dataset**: Breast Cancer Wisconsin Diagnostic Dataset  
- **Source**: `sklearn.datasets.load_breast_cancer()`  
- **Features**: 30 numeric features  
- **Target**: Binary classification (Malignant / Benign)

---

## 🔍 Overview of Techniques

### 🟦 1. Monte Carlo (MC) Dropout
Implemented in `MC_Dropout.ipynb`

- Dropout layers are used during inference to simulate Bayesian model behavior.
- Utilizes **5-Fold Cross Validation** to ensure generalization.
- Multiple stochastic forward passes are made to estimate:
  - **Predictive mean**
  - **Uncertainty (variance/entropy)**

### 🟨 2. Deep Ensemble
Implemented in `Ensemble.ipynb`

- Multiple independent models are trained with different random seeds.
- Ensemble predictions are averaged.
- Captures both:
  - **Epistemic uncertainty** (model uncertainty)
  - **Aleatoric uncertainty** (data noise)

---

## 📈 Evaluation Metrics

- **Accuracy**
- **Cross-Entropy Loss**
- **Prediction Variance**
- **Standard Deviation / Entropy of Output Probabilities**
- (Optional) **Calibration Curves**, **Confidence Intervals**

---

## 🚀 How to Run

### 📦 Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn torch
