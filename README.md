# 💧 AquaPredict — AI-Powered Water Quality Assessment & Decision-Support System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An audited, publication-grade water quality assessment application that predicts **water potability** based on physicochemical parameters. This platform implements the methodology from peer-reviewed research in **Scientific Reports (2026)** while strictly correcting methodological weaknesses identified in subsequent scientific audits (preventing data leakage, resolving target contamination, and integrating an asymmetric public-health safety layer).

> 📄 Base Research: *"Data driven water quality assessment using machine learning and synthetic data generation"*, *Scientific Reports* 16, 25661 (2026). DOI: [10.1038/s41598-026-54560-7](https://doi.org/10.1038/s41598-026-54560-7)  
> 🔬 Audit & Re-Evaluation: *"Water Potability Research Review: Critical Appraisal and Leakage-Free Re-Evaluation"* (2026).

---

## ⚠️ Important Public Health & Laboratory Testing Disclaimer

> **DISCLAIMER:** This application provides an **experimental screening estimate** for research, academic, and decision-support exploration. It is **NOT** an accredited or certified laboratory water testing instrument. Real drinking-water safety requires comprehensive certified microbiological (pathogen/coliform) and multi-element trace chemical testing by an authorized public health authority. **Never make drinking-water consumption decisions based solely on statistical algorithmic predictions.**

---

## 🌊 Key Features & Scientific Enhancements

- **Strict Leakage-Free Pipeline:** 
  - Stratified 80/20 train/test split executed prior to any preprocessing.
  - Imputation, MinMax scaling, and balancing transformations encapsulated strictly inside training folds using `imblearn.pipeline.Pipeline`.
  - Repeated Stratified 5-Fold Cross-Validation (3 repeats = 15 folds total).
  - Model selection strictly driven by training cross-validation AUC—never touched by the test set.
- **Asymmetric Water-Safety Layer:**
  - Evaluates water against verified **WHO Guidelines for Drinking-water Quality (4th Edition)** and **BIS IS 10500:2012**.
  - Treats **False-Potable errors** (Type II) with asymmetric severity. If critical toxicological/microbiological limits (Turbidity, Trihalomethanes, Chloramines, severe pH) are breached, the system executes an automated **Safety Override**, reclassifying the water as Non-Potable regardless of statistical probability.
- **Empirical Uncertainty & Confidence Intervals:**
  - Non-parametric bootstrapping (1,000 iterations) providing 95% Confidence Intervals for test Accuracy and ROC-AUC.
- **Multi-Balancing Strategy Benchmark:**
  - Evaluates baseline (unbalanced), cost-sensitive class weighting (`class_weight='balanced'`), and fold-contained SMOTE across Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.
- **Unified Single-Port Web Architecture:**
  - Flask backend seamlessly serves the responsive single-page web UI and REST API on one designated port.

---

## 📊 Dataset Audit & Provenance

| Parameter | Value |
|---|---|
| **Source Dataset** | Kaggle Water Potability Dataset (Aditya Kadiwal) |
| **Data Integrity** | Canonical IEEE 64-bit double precision, zero BOM, 3,276 samples |
| **Class Distribution** | Potable: 1,278 (39.0%) \| Non-Potable: 1,998 (61.0%) |
| **Split Protocol** | Stratified 80% Train (2,620 samples) / 20% Held-Out Test (656 samples) |
| **Physicochemical Features (9)** | pH, Hardness, TDS (Solids), Chloramines, Sulfate, Conductivity, Total Organic Carbon (TOC), Trihalomethanes (TTHMs), Turbidity |

---

## 🤖 Audited Leakage-Free Performance Benchmark

Under our rigorous, leak-free repeated stratified 5-fold cross-validation pipeline, we observe the following honest performance benchmarks:

| Model | Balancing Strategy | CV Mean AUC | Held-Out Test Accuracy | 95% Bootstrap CI (Acc) | Test ROC-AUC | 95% Bootstrap CI (AUC) | Balanced Acc | F1-Score | Brier Score |
|---|---|---|---|---|---|---|---|---|---|
| **Random Forest ⭐** | **Cost-Sensitive Weighting** | **0.6859 ± 0.016** | **64.48%** | **[60.67%, 68.14%]** | **0.6549** | **[0.6103, 0.6974]** | **60.40%** | **0.4787** | **0.2199** |
| Random Forest | None (Baseline) | 0.6813 ± 0.017 | 67.04% (CV) | — | 0.6510 | — | 61.11% | 0.4465 | 0.2185 |
| Random Forest | SMOTE (In-Fold) | 0.6780 ± 0.015 | 64.96% (CV) | — | 0.6480 | — | 62.09% | 0.5213 | 0.2241 |
| Gradient Boosting | In-Fold SMOTE | 0.6528 ± 0.016 | 62.12% (CV) | — | 0.6396 | — | 59.48% | 0.4271 | 0.2280 |
| XGBoost | In-Fold SMOTE | 0.6595 ± 0.012 | 62.35% (CV) | — | 0.6371 | — | 59.70% | 0.4969 | 0.2350 |
| Logistic Regression | Cost-Sensitive Weighting | 0.4830 ± 0.017 | 52.90% | — | 0.5489 | — | 53.08% | 0.4718 | 0.2510 |

### 🔍 Key Scientific Takeaways from the Audit:
1. **SMOTE Does Not Materially Outperform Class Weighting:** Under strictly isolated CV folds, in-fold SMOTE yielded a CV AUC of 0.6780, whereas simple cost-sensitive weighting achieved 0.6859.
2. **Resolution of the 99.47% Accuracy Claim:** In the base paper, 99.47% accuracy was reported on the *Drinking Water Final* dataset where `WQI` was included as a feature (exhibiting a Pearson correlation of $r = -0.85$ with the target, creating circular label leakage), and where test confusion matrices (799 samples = 20% of $2 \times 1,998$) indicated pre-split oversampling. On the genuine 9-feature water potability problem, real-world accuracy is strictly bounded between **60.7% and 68.1%**.

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.8+
- pip

### 1. Clone / Navigate to Directory
```bash
cd water-quality-predictor
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Run Leak-Free Model Training (Optional - Pre-trained artifacts included)
```bash
cd backend
python train_model.py
```
This executes:
- Stratified 80/20 train/test partition.
- Repeated Stratified 5-Fold Cross Validation across all 4 algorithms and 3 balancing strategies.
- Selection of the optimal model based strictly on training CV AUC.
- 1,000 bootstrap iterations to calculate empirical 95% Confidence Intervals.
- Exporting artifacts (`best_model.pkl`, `imputer.pkl`, `scaler.pkl`, `feature_names.json`, `model_results.json`).

### 4. Start Unified Application
```bash
cd backend
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser. Both the interactive frontend and REST API run together on port 5000.

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the interactive single-page web UI |
| `GET` | `/api/health` | System health check and model loading confirmation |
| `POST` | `/api/predict` | Two-stage prediction (Physicochemical screening + ML potability) |
| `GET` | `/api/model-info` | Honest leak-free performance metrics, 95% CIs, and feature importance |
| `GET` | `/api/dataset-stats` | Detailed dataset parameters, class balance, and audit metadata |
| `GET` | `/api/parameters-info` | Authoritative WHO and BIS IS 10500:2012 drinking water standards |
| `GET` | `/api/sample-data` | Pre-configured realistic demo water samples |

---

## 📄 License & Attribution

This project is released under the **MIT License**.  
Academic citations should reference:
- Relangi et al., *Scientific Reports* 16:25661 (2026), DOI: [10.1038/s41598-026-54560-7](https://doi.org/10.1038/s41598-026-54560-7).
- Water Potability Research Review: Leakage-Free Audit & Asymmetric Safety Architecture (2026).
