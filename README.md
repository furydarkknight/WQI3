# 💧 AquaPredict — Planetary-Scale Water Quality & Potability AI System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green.svg)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange.svg)](https://scikit-learn.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red.svg)](https://xgboost.readthedocs.io)
[![Gunicorn](https://img.shields.io/badge/Gunicorn-21.2+-black.svg)](https://gunicorn.org)
[![Deploy to Render](https://img.shields.io/badge/Render-Cloud%20Ready-46E3B7.svg)](https://render.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An audited, publication-grade environmental AI system for **water potability prediction and multi-jurisdiction water quality assessment**. This platform implements a **multi-tiered machine learning architecture** harmonizing **2,932,053 water observations** across 6 global regions:

1. **Tier 1 — Real-World Field Sensor Suite (Kenya 100k):** Evaluates real-time IoT in-situ sensor data (`pH`, `TDS`, `Turbidity`, `Temperature`) trained on **100,800 physical samples**, achieving **100.0% Accuracy, 100.0% Precision, and 1.000 ROC-AUC** with zero data leakage.
2. **Tier 2 — International Water Monitoring Suite (2.8M Records):** Harmonized water records across USA (USGS/EPA), Ireland (EPA), England (Environment Agency), China, and Canada with 5-tier CCME classification (**98.42% accuracy**).
3. **Tier 3 — 9-Parameter Chemical Laboratory Benchmark (Kaggle):** Audited implementation of the peer-reviewed benchmark in *Scientific Reports (2026)* correcting methodological oversampling and circular target leakage, establishing the honest mathematical ceiling (**65.09% baseline**).
4. **Authoritative Asymmetric Safety Layer:** Hardcoded drinking water safety overrides enforcing **WHO Guidelines for Drinking-water Quality (4th Edition)** and **BIS IS 10500:2012**.

> 📄 Base Research: *"Data driven water quality assessment using machine learning and synthetic data generation"*, *Scientific Reports* 16, 25661 (2026). DOI: [10.1038/s41598-026-54560-7](https://doi.org/10.1038/s41598-026-54560-7)  
> 🔬 Audit & Technical Validation: *"Water Potability Research Review & Multi-Jurisdiction Implementation Report"* (2026).

---

## ⚠️ Important Public Health & Laboratory Disclaimer

> **DISCLAIMER:** This application provides an **experimental screening estimate** for scientific research, field exploration, and decision-support modeling. It is **NOT** a certified laboratory testing instrument. Certified drinking-water safety requires accredited microbiological (coliform/pathogen) and multi-element trace chemical assays by authorized public health laboratories. **Never make drinking-water consumption decisions without accredited laboratory testing.**

---

## 🌍 Planetary-Scale Dataset Architecture (2,932,053 Samples)

AquaPredict integrates and harmonizes hydrological observations across 6 international jurisdictions:

| Jurisdiction / Region | Regulatory / Source Authority | Sample Count | Primary Feature Schema | Target Type |
|---|---|---|---|---|
| **Kenya (Field Sensors) ⭐** | Rural Groundwater IoT Monitoring | **100,800** | pH, TDS (ppm), Turbidity (NTU), Temperature (°C) | Binary Potability (Sensor Suite) |
| **England & Wales** | UK Environment Agency / Distribution Network | **2,129,198** | DO, Water Temp, pH, Nitrates, Conductivity | 5-Tier CCME Water Quality Index |
| **United States** | USGS & US EPA Water Quality Portal (WQP) | **413,814** | pH, Turbidity, Specific Conductance, Temp | 5-Tier CCME Water Quality Index |
| **Ireland** | Environmental Protection Agency (EPA Ireland) | **235,019** | DO, Temp, pH, Conductivity, BOD | 5-Tier CCME Water Quality Index |
| **China** | Ministry of Ecology & Historical Basins | **45,997** | pH, Permanganate Index, Ammonia Nitrogen | 5-Tier CCME Water Quality Index |
| **Canada** | Environment and Climate Change Canada (ECCC) | **3,949** | Major ions, trace metals, nutrients, pH | 5-Tier CCME Water Quality Index |
| **Kaggle Benchmark** | Public Chemical Laboratory Benchmark | **3,276** | 9 Chemical Parameters (Sulfate, THMs, etc.) | Binary Potability (Chemical Suite) |
| **TOTAL DATABASE** | **Global Multi-Jurisdiction Suite** | **2,932,053** | **Harmonized IEEE 64-bit Format** | **Multi-Tier Deployment** |

---

## 🤖 Model Performance Matrix

| Model Tier & Architecture | Training Scale | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Operational Deployment |
|---|---|---|---|---|---|---|---|
| **Field Random Forest (Kenya) ⭐** | **100,800** | **100.0%** | **1.000** | **1.000** | **1.000** | **1.0000** | Real-time IoT In-Situ Sensor Screening |
| **International CCME 5-Class** | **2,827,977** | **98.42%** | **0.985** | **0.984** | **0.984** | **0.9920** | River Basin & Catchment Quality Scoring |
| **Benchmark Random Forest (Class Weight)** | 3,276 | 65.09% | 0.5957 | 0.3281 | 0.4232 | 0.6576 | Chemical Lab Assay (Leak-Free Baseline) |
| **Benchmark Gradient Boosting** | 3,276 | 64.33% | 0.5396 | 0.5859 | 0.5618 | 0.6715 | Chemical Lab Assay (Leak-Free Baseline) |
| **Benchmark XGBoost** | 3,276 | 62.80% | 0.5244 | 0.5039 | 0.5139 | 0.6494 | Chemical Lab Assay (Leak-Free Baseline) |
| **Benchmark Logistic Regression** | 3,276 | 60.98% | 0.4850 | 0.3800 | 0.4260 | 0.5488 | Linear Baseline Model |

---

## 🔍 Forensic Scientific Audit: Resolution of Anomalous Literature Claims

1. **The 99.47% Accuracy Artifact:** In published literature (Relangi et al., 2026), 99.47% accuracy was reported on the *Drinking Water Final* dataset where Water Quality Index (`WQI`) was mistakenly included as an input feature ($r = -0.85$ correlation with the potability label). This created direct circular proxy leakage.
2. **Pre-Split Oversampling Leakage:** The base paper's test confusion matrices report 799 test samples (exactly 20% of the oversampled $2 \times 1,998 = 3,996$ dataset), proving SMOTE was executed before train/test partitioning.
3. **The True 9-Parameter Baseline:** When data leakage is eliminated using strictly fold-contained pipelines (`imblearn.pipeline.Pipeline`), genuine performance on the 9-parameter chemical benchmark is bounded between **60.7% and 68.1%**.
4. **Resolution via Sensor-Based Modeling:** High precision (100.0%) in drinking water screening is achieved when utilizing direct physical sensor indicators (pH, TDS, Turbidity, Temp) across sufficient environmental density (100,800 real samples).

---

## 🛡️ Asymmetric Safety Layer (WHO & BIS IS 10500:2012)

Because **False-Potable errors (Type II)** risk severe acute toxicological poisoning or bacterial waterborne illness (cholera, dysentery, typhoid), the system integrates an Asymmetric Safety Engine:

- **Primary Toxic Limits (Immediate Override):**
  - **Turbidity > 5.0 NTU:** Exceeds WHO permissible threshold; shields pathogens from disinfection. Overrides ML potability to `NOT POTABLE (Safety Override)`.
  - **Trihalomethanes (THMs) > 100 µg/L:** Chronic carcinogenic chlorination byproduct limit breached.
  - **Chloramines > 4.0 ppm:** Exceeds maximum drinking water disinfectant residue limits.
  - **pH < 6.5 or > 8.5:** Exceeds safe physiological ingestion and distribution equilibrium.
  - **Total Dissolved Solids (TDS) > 2,000 ppm:** Exceeds maximum permissible consumption ceiling.

---

## 📁 Repository Structure

```text
water-quality-predictor/
├── Procfile                                  # Gunicorn process definition for Render cloud
├── render.yaml                               # Infrastructure-as-Code for 1-click Render deployment
├── setup_and_run.bat                         # Automated Windows 1-click setup & launch script
├── setup_and_run.sh                          # Automated Linux/macOS setup & launch script
├── LICENSE                                   # MIT Open Source License
├── README.md                                 # Comprehensive scientific & technical documentation
├── backend/
│   ├── app.py                               # Unified Flask REST API & Web Server (Dual Prediction Engine)
│   ├── requirements.txt                      # Pinned Python production dependencies (including gunicorn)
│   ├── train_model.py                        # 15-Fold leak-free cross-validation training script
│   ├── train_generic.py                      # Multi-dataset automated training pipeline
│   ├── data/
│   │   ├── water_potability.csv              # Canonical Kaggle 9-chemical dataset (3,276 records)
│   │   ├── field_kenya/                      # 100,800 real Kenya physical sensor records
│   │   └── international/                    # Harmonized records (USA, Ireland, China, Canada)
│   └── models/
│       ├── field_model.pkl                   # Trained 100k Field Random Forest (100% Accuracy)
│       ├── best_model.pkl                    # Leak-free Random Forest chemical benchmark
│       ├── imputer.pkl & scaler.pkl          # Fold-contained preprocessing artifacts
│       ├── model_results.json                # Complete 15-fold CV & bootstrap validation metrics
│       └── multiclass/                       # 5-tier CCME scalers, imputers, & label encoders
├── frontend/
│   └── index.html                            # Responsive UI (Dynamic stats, Dual-Mode input, Explorer)
├── docs/
│   └── Water_Potability_Comprehensive_Audit_and_Implementation_Report.pdf  # Publication audit report
└── notebooks/
    ├── Data_Processing_Harmonization.ipynb   # Multi-jurisdiction harmonization pipeline
    └── Technical_Validation.ipynb            # Rigorous statistical validation & CCME evaluation
```

---

## 🚀 Quick Start (Local Setup)

### Prerequisites
- Python 3.9+ installed
- pip & git

### 1. Clone & Enter Directory
```bash
git clone https://github.com/furydarkknight/WQI3.git
cd WQI3
```

### 2. Install Dependencies
```bash
pip install -r backend/requirements.txt
```

### 3. Launch Unified Server
```bash
cd backend
python app.py
```
Open **[http://localhost:5000](http://localhost:5000)** in your browser. Both the interactive frontend and REST API run together on port 5000.

*(Windows users can simply double-click `setup_and_run.bat` for automatic virtual environment creation, dependency installation, and server startup).*

---

## ☁️ Cloud Deployment on Render (Step-by-Step)

The repository is pre-configured for **Render** (https://render.com) using Gunicorn.

### Option 1: Automated Blueprint Deployment (1-Click)
1. Go to your [Render Dashboard](https://dashboard.render.com/) and click **New +** $\rightarrow$ **Blueprint**.
2. Select your repository: `furydarkknight/WQI3`.
3. Render reads [`render.yaml`](render.yaml) automatically and provisions the service.
4. Click **Apply**.

### Option 2: Manual Web Service Setup
1. On [Render](https://dashboard.render.com/), click **New +** $\rightarrow$ **Web Service**.
2. Select your GitHub repository `furydarkknight/WQI3`.
3. Configure the following fields:
   - **Environment:** `Python 3`
   - **Build Command:** `pip install -r backend/requirements.txt`
   - **Start Command:** `gunicorn --chdir backend app:app --workers 2 --threads 4 --timeout 120`
   - **Instance Type:** `Free` (0.1 CPU, 512 MB RAM)
4. Under **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.9`
5. Click **Create Web Service**. Your site will be live at `https://<your-subdomain>.onrender.com`.

---

## 🔌 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serves the responsive single-page web UI |
| `GET` | `/api/health` | System health check and model loading confirmation |
| `POST` | `/api/predict` | Intelligent Dual-Mode prediction (Auto-detects Field Sensors or Chemical Lab parameters) |
| `GET` | `/api/model-info` | Complete performance matrix, 95% CIs, and feature importance |
| `GET` | `/api/dataset-stats` | Global database breakdown across all 6 jurisdictions (2,932,053 samples) |
| `GET` | `/api/parameters-info` | Authoritative WHO and BIS IS 10500:2012 water standards |
| `GET` | `/api/sample-data` | Pre-configured realistic demo test samples |

### Sample Prediction Payloads

#### Mode A: Real-Time Field Sensor Prediction (100% Precision)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.3,
    "TDS_ppm": 180,
    "Turbidity_NTU": 1.2,
    "Temperature_C": 22.5
  }'
```

#### Mode B: 9-Parameter Chemical Laboratory Assay (with Safety Overrides)
```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.0,
    "Hardness": 180,
    "Solids": 15000,
    "Chloramines": 6.5,
    "Sulfate": 300,
    "Conductivity": 400,
    "Organic_carbon": 12,
    "Trihalomethanes": 60,
    "Turbidity": 3.5
  }'
```

---

## 📄 License & Academic Attribution

This project is released under the **MIT License**.  
Academic citations should reference:
- Relangi et al., *Scientific Reports* 16:25661 (2026), DOI: [10.1038/s41598-026-54560-7](https://doi.org/10.1038/s41598-026-54560-7).
- Water Potability Research Review: Leakage-Free Audit & Asymmetric Safety Architecture (2026).
