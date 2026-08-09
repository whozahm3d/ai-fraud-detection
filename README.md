<div align="center">

# AI Fraud Detection

### AI-Powered Fraud Detection for Digital Banking

#### Explainable AI · RAG-Based Policy Assistance · Live Streamlit Dashboard

<br>

<!-- Runtime & Language -->
[![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
<!-- Framework -->
[![Streamlit 1.45](https://img.shields.io/badge/Streamlit-1.45-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
<!-- ML Stack -->
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1-006400?style=flat-square)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/XAI-SHAP-FF6F00?style=flat-square)](https://shap.readthedocs.io)
[![RAG](https://img.shields.io/badge/RAG-ChromaDB%20+%20Cohere-7B68EE?style=flat-square)](https://www.trychroma.com)
<!-- Status & License -->
[![Status](https://img.shields.io/badge/Status-Hackathon%20Project-blueviolet?style=flat-square)](.)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/whozahm3d/ai-fraud-detection/ci.yml?style=flat-square&label=CI&logo=github)](https://github.com/whozahm3d/ai-fraud-detection/actions/workflows/ci.yml)

<br>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)]([NEW_DEPLOYMENT_URL])
&nbsp;&nbsp;
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/whozahm3d/ai-fraud-detection)

<br>

<!-- 🎬 DEMO GIF — record a 30s walkthrough of the dashboard and replace this block -->
<!-- Tip: use Kap (Mac), ScreenToGif (Windows), or peek (Linux) -->
<!-- Drag the .gif into any GitHub Issue to get a raw URL, then embed below -->
<!--
![AI Fraud Detection Demo](https://your-gif-url-here.gif)
-->

> An end-to-end AI system that detects financial fraud on severely imbalanced data (0.13% fraud rate, 6.3M transactions), explains every prediction with SHAP, and grounds risk reports in State Bank of Pakistan regulations via a RAG pipeline — all accessible through a live Streamlit dashboard.

<br>

</div>

---

## ⚡ Key Results at a Glance

<div align="center">

| Metric | XGBoost (Deployed) |
| :--- | :---: |
| **AUC-ROC** | **0.9995** |
| **Recall** | **0.9976** |
| **Avg Precision (AP)** | **0.9358** |
| **F1 Score** | **0.5691** |
| **Test Accuracy** | **99.80%** |
| **Inference Speed** | **< 50ms / transaction (CPU)** |
| RAG Hallucinations | **0 / 4** |
| Fraud cases caught (6.3M test set) | **8,190 / 8,213** |

</div>

---

## 🧭 How It Works — 60-Second Overview

```
① INPUT          ② PREDICT            ③ EXPLAIN           ④ RETRIEVE           ⑤ REPORT
─────────        ─────────────         ─────────────        ─────────────        ──────────────
Transaction  →   XGBoost model    →   SHAP waterfall   →   ChromaDB + BM25  →   Cohere
attributes        (12 features)        per-feature          hybrid retrieval      regulatory
(amount,          AUC-ROC=0.9995      contribution         from SBP corpus       risk report
 type, bal.)      < 50ms latency       plot + tier          CrossEncoder          grounded,
                                       (LOW/MED/HIGH/       reranking             0 hallucinations
                                        CRITICAL)
```

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Live Dashboard](#-live-dashboard)
- [Quick Start](#-quick-start)
- [Full Setup Guide](#-full-setup-guide)
- [Project Structure](#-project-structure)
- [Known Limitations](#-known-limitations)
- [Troubleshooting](#-troubleshooting)
- [Hackathon Context](#-hackathon-context)
- [How to Cite](#-how-to-cite)
- [Contributing](#-contributing)
- [Author](#-author)
- [References](#-references)

---

## 🎯 Project Overview

Financial fraud causes trillions of dollars in annual losses globally. Detecting it is hard — not because the algorithms are weak, but because **fraud is statistically invisible**: only 0.13% of transactions in the PaySim dataset are fraudulent. A naïve model predicting "legitimate" on everything achieves 99.87% accuracy while catching zero fraud.

**AI Fraud Detection** solves this through three integrated components:

1. **Fraud Detection Engine** — A two-stage imbalance pipeline (Fraud Simulation → SMOTE) trains four ML models under identical conditions. XGBoost is deployed with AUC-ROC = 0.9995 and Recall = 0.9976.

2. **Explainable AI (XAI)** — SHAP TreeExplainer generates per-transaction waterfall explanations, showing exactly which features drove each prediction and by how much.

3. **RAG Policy Assistant** — A hybrid retrieval system (BM25 + dense vectors + CrossEncoder reranking) retrieves relevant SBP regulatory provisions and feeds them to Cohere, generating grounded regulatory risk reports with zero hallucinations.

---

## 🏗️ System Architecture

```
Raw PaySim CSV (6.36M rows)
        │
        ▼
┌─────────────────────┐
│   Data Cleaning &   │  Drop nameOrig, nameDest, isFlaggedFraud
│  Feature Engineering│  Add balanceDiff, amount_ratio, one-hot type
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Fraud Simulation   │  Inject synthetic full-drain transactions
│     Engine          │  0.13% → 1.26% fraud rate
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  80/20 Stratified   │     │  SBP Regulatory Corpus  │
│    Train/Test Split │     │  (5 PDFs → ~100 chunks)  │
└─────────┬───────────┘     └────────────┬─────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  ImbPipeline × 5    │     │  ChromaDB Vector Store   │
│  (SMOTE 0.3 →       │     │  all-MiniLM-L6-v2        │
│  StandardScaler →   │     │  BM25 + Dense Retrieval  │
│  Classifier)        │     │  CrossEncoder Reranker   │
└─────────┬───────────┘     └────────────┬─────────────┘
          │                              │
          ▼                              ▼
┌─────────────────────┐     ┌──────────────────────────┐
│  XGBoost (deployed) │────▶│   Cohere Generator     │
│  + SHAP TreeExplain │     │   Regulatory Risk Report │
└─────────┬───────────┘     └────────────┬─────────────┘
          │                              │
          └──────────────┬───────────────┘
                         ▼
              ┌─────────────────────┐
              │  Streamlit Dashboard│
              │  6 interactive pages│
              └─────────┬───────────┘
```

---

## ✨ Features

### 🔍 Fraud Detection

- Trains **4 models** (Logistic Regression, Random Forest, Neural Network, XGBoost) under identical pipeline conditions
- Two-stage imbalance handling raises training fraud rate from **0.13% → 23.07%** without data leakage
- 5-fold stratified cross-validation with SMOTE applied strictly inside each fold
- Full ablation study across 7 conditions isolating the contribution of each pipeline component

### 🧠 Explainable AI

- **SHAP TreeExplainer** for per-transaction waterfall plots (< 50ms, CPU)
- Feature importance comparison across all 4 models
- Bias & fairness check across transaction type subgroups
- Threshold optimisation curve showing precision/recall trade-offs

### 📜 RAG Policy Assistant

- **Hybrid retrieval**: BM25 (lexical) + sentence-transformers (semantic) combined
- **CrossEncoder reranking** (`ms-marco-MiniLM-L-6-v2`) for passage quality scoring
- **Cohere** generation grounded in SBP regulatory documents
- Zero hallucinations across all evaluated transactions
- Average retrieval Precision@5 = 0.855 across 10 regulatory queries

### 📊 Interactive Dashboard

- Single transaction prediction with real-time SHAP explanation
- Batch CSV upload and scoring (up to 200MB)
- Full EDA visualisation suite
- Model comparison, confusion matrices, ROC/PR curves
- Ablation study explorer
- Regulatory report generation (requires Cohere API key)

---

## 📊 Dataset

**PaySim Synthetic Financial Dataset** — [Kaggle Link](https://www.kaggle.com/datasets/ealaxi/paysim1)

A multi-agent simulation calibrated against real mobile money transaction data from an African financial institution.

| Property | Value |
| :--- | :--- |
| Total transactions | 6,362,620 |
| Fraudulent transactions | 8,213 (**0.13%**) |
| Legitimate transactions | 6,354,407 |
| Simulation period | 30 days (743 hourly steps) |
| Transaction types | CASH_OUT, TRANSFER, PAYMENT, DEBIT, CASH_IN |
| Fraud types | CASH_OUT and TRANSFER **only** |
| Missing values | None |

> [!NOTE]
> The raw dataset (~500MB) is not included in this repository. Download from Kaggle and place it at `Data/original_dataset/`. The pre-trained model in `outputs/deployment/` works without it.

---

## 🔬 Methodology

### Class Imbalance Handling

The 0.13% fraud rate is too severe for SMOTE alone — minority folds in 5-fold CV can contain fewer than 10 fraud samples. This system uses a **two-stage strategy**:

```
Stage 1 — Fraud Simulation Engine (global, before split)
  • Sample 5% of legitimate TRANSFER & CASH_OUT transactions
  • Set amount = oldbalanceOrg (full drain), newbalanceOrig = 0
  • Recompute balanceDiff and amount_ratio
  • Label injected rows as fraud
  Result: 0.13% → 1.26% fraud rate

Stage 2 — SMOTE inside ImbPipeline (per CV fold, train only)
  • sampling_strategy = 0.3
  • Applied AFTER train/test split, INSIDE each fold
  • Validation fold never sees synthetic samples
  Result: 1.26% → 23.07% fraud rate in training
```

| Stage | Fraud Cases | Total Rows | Fraud Rate |
| :--- | :---: | :---: | :---: |
| Original | 8,213 | 6,362,620 | 0.13% |
| After Fraud Simulation | ~87,500 | ~6,940,000 | 1.26% |
| After SMOTE (train folds) | ~1,600,000 | ~6,930,000 | **23.07%** |

### Feature Engineering

12 features used for model training (6 raw + 2 engineered + 4 one-hot):

| Feature | Type | Signal |
| :--- | :--- | :--- |
| `step` | Raw | Temporal position (0–743 hours) |
| `amount` | Raw | Transaction size — higher in fraud |
| `oldbalanceOrg` | Raw | Origin balance before |
| `newbalanceOrig` | Raw | Drops to 0 in full-drain fraud |
| `oldbalanceDest` | Raw | Mule accounts often start at 0 |
| `newbalanceDest` | Raw | Spikes in fraud cases |
| `balanceDiff` | **Engineered** | `oldbalanceOrg − newbalanceOrig − amount` — detects inconsistencies |
| `amount_ratio` | **Engineered** | `amount / (oldbalanceOrg + 1)` — approaches 1.0 in drain attacks |
| `type_CASH_OUT` | One-hot | Binary type indicator |
| `type_DEBIT` | One-hot | Binary type indicator |
| `type_PAYMENT` | One-hot | Binary type indicator |
| `type_TRANSFER` | One-hot | Binary type indicator |

### Models Trained

All four models use `ImbPipeline(SMOTE → StandardScaler → Classifier)` with 5-fold stratified CV:

| Model | Role | Key Hyperparameters |
| :--- | :--- | :--- |
| **XGBoost** ⭐ | Deployed model | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6` |
| Random Forest | Ensemble baseline | `max_depth=12`, `n_estimators=30`, `max_samples=0.5` |
| Neural Network | Non-linear baseline | `hidden_layer_sizes=(32,16)`, `batch_size=256`, `alpha=0.03` |
| Logistic Regression | Linear baseline | `C=1.0`, `solver=saga` |

### XAI — SHAP Explanations

SHAP `TreeExplainer` decomposes each prediction into per-feature additive contributions. Top drivers:

1. **`amount_ratio`** — values near 1.0 (full drain) are the strongest fraud signal
2. **`newbalanceOrig`** — near-zero post-transaction balance confirms drain
3. **`balanceDiff`** — inconsistencies between amount and balance change
4. **`type_CASH_OUT` / `type_TRANSFER`** — transaction type as a gate
5. **`amount`** — absolute size, though weaker than the ratio

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/ai-fraud-detection/main/Images/images_final/feature_importance_comparison.png" width="85%" alt="Feature Importance Comparison across all 4 models"/>
<br><em>Feature importance comparison across all 4 trained models</em>
</div>

### RAG Pipeline

| Component | Technology |
| :--- | :--- |
| Vector store | ChromaDB (`~100 chunks`, pre-built in `chroma_db/`) |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Lexical retrieval | BM25 (`rank-bm25`) |
| Reranker | `ms-marco-MiniLM-L-6-v2` CrossEncoder |
| Generator | `command-r-plus-08-2024` (Cohere) |
| Regulatory corpus | 5 SBP PDFs (C1-Annex, C2-Annex-A, CL33-Annex-B, C10-Branchless-Banking, SME-PRs-Jan-2025) |

> [!NOTE]
> The ChromaDB vector store is pre-built and included in `chroma_db/`. You do **not** need to rebuild it unless you add new documents to the corpus.

---

## 📈 Results

### Model Comparison — Test Set

| Model | Precision | Recall | F1 | AUC-ROC | Avg Precision |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** ⭐ | **0.3981** | **0.9976** | **0.5691** | **0.9995** | **0.9358** |
| Random Forest | 0.1035 | 0.9976 | 0.1875 | 0.9995 | 0.8870 |
| Neural Network | 0.1437 | 0.9732 | 0.2505 | 0.9983 | 0.7081 |
| Logistic Regression | 0.0217 | 0.9860 | 0.0425 | 0.9946 | 0.5567 |

### Cross-Validation — All 4 Models (5-fold, mean ± std)

| Model | CV Precision | CV Recall | CV F1 | CV AUC-ROC | CV Avg Prec |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **XGBoost** ⭐ | 0.905 ± 0.035 | 1.000 ± 0.000 | **0.949 ± 0.020** | 1.000 ± 0.000 | 0.985 ± 0.004 |
| Random Forest | 0.553 ± 0.009 | 0.995 ± 0.001 | 0.711 ± 0.007 | 0.999 ± 0.000 | 0.934 ± 0.002 |
| Neural Network | 0.667 ± 0.089 | 0.990 ± 0.005 | 0.793 ± 0.061 | 0.999 ± 0.000 | 0.969 ± 0.003 |
| Logistic Regression | 0.142 ± 0.002 | 0.997 ± 0.001 | 0.249 ± 0.003 | 0.977 ± 0.001 | 0.337 ± 0.006 |

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/ai-fraud-detection/main/Images/images_final/model_comparison_all_metrics.png" width="85%" alt="Model Comparison — All Metrics"/>
<br><em>All four models evaluated on the same hold-out test set</em>
</div>

### ROC & Precision-Recall Curves

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/ai-fraud-detection/main/Images/images_final/roc_pr_curves.png" width="85%" alt="ROC and Precision-Recall Curves"/>
<br><em>ROC and Precision-Recall curves for the deployed XGBoost model</em>
</div>

### Confusion Matrices — All Models

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/ai-fraud-detection/main/Images/images_final/confusion_matrices_all_models.png" width="85%" alt="Confusion Matrices for all 4 models"/>
<br><em>Confusion matrices for all 4 trained models on the held-out test set</em>
</div>

### Cost-Benefit Analysis

<div align="center">
<img src="https://raw.githubusercontent.com/whozahm3d/ai-fraud-detection/main/Images/images_final/cost_benefit_analysis.png" width="85%" alt="Cost-Benefit Analysis"/>
<br><em>Cost-benefit analysis at varying fraud probability thresholds</em>
</div>

### Ablation Study — All 7 Conditions

<details>
<summary><strong>📊 Click to expand full ablation results table</strong></summary>

<br>

| ID | Condition | CV F1 | Test Recall | Test F1 | Test AP |
| :--- | :--- | :---: | :---: | :---: | :---: |
| A1 | No Fraud Simulation | 0.671 | 0.9976 | 0.6247 | 0.9363 |
| **A2** | **Full Pipeline** ⭐ | **0.947** | **0.9976** | **0.5533** | **0.7317** |
| B1 | No SMOTE | 0.947 | 0.9951 | 0.9132 | 0.9639 |
| **B2** | **SMOTE ratio=0.3** ⭐ | **0.947** | **0.9976** | **0.5557** | **0.7322** |
| B3 | SMOTE ratio=0.5 | 0.947 | 0.9976 | 0.5280 | 0.7180 |
| C1 | No Engineered Features | 0.636 | 0.9976 | 0.1538 | 0.6061 |
| **C2** | **With Features** ⭐ | **0.947** | **0.9976** | **0.5533** | **0.7317** |

</details>

**Key findings:**

- Removing engineered features (C1) drops Test F1 by **73%** — the single most critical component
- Removing Fraud Simulation (A1) drops CV F1 by **29%** — SMOTE alone is insufficient at 0.13% fraud rate
- SMOTE ratio=0.3 (B2) slightly outperforms 0.5 (B3) on Test AP

### RAG Evaluation

| Transaction | Fraud Prob | Risk Tier | Latency | Hallucination |
| :--- | :---: | :---: | :---: | :---: |
| TXN-CRITICAL-001 | 97% | 🔴 CRITICAL | 6.45s | ✅ None |
| TXN-HIGH-001 | 94% | 🔴 CRITICAL | 4.67s | ✅ None |
| TXN-HIGH-002 | 88% | 🔴 CRITICAL | 3.40s | ✅ None |
| TXN-MEDIUM-001 | 73% | 🟠 HIGH | 3.64s | ✅ None |

Retrieval: **Avg Precision@5 = 0.855** · Term Hit Rate = 0.90 · Expected doc found: **10/10**

---

## 🖥️ Live Dashboard

**URL:** [[NEW_DEPLOYMENT_URL]]([NEW_DEPLOYMENT_URL])

| Page | What it does |
| :--- | :--- |
| **① Predict Transaction** | Enter transaction attributes → fraud probability, risk tier, SHAP waterfall, RAG regulatory report |
| **② Batch CSV Analysis** | Upload a CSV (up to 200MB) → score all rows, download results |
| **③ Dataset & Imbalance** | Interactive EDA: class distribution, fraud patterns, imbalance pipeline visualisation |
| **④ Model Performance** | ROC/PR curves, confusion matrices, threshold optimisation, cost-benefit analysis, bias check |
| **⑤ Ablation Study** | All 7 conditions across all metrics — bar chart, heatmap, delta plot, PR scatter |
| **⑥ About** | Author info, architecture overview, links |

### Risk Tier Thresholds

| Fraud Probability | Risk Tier | Recommended Action |
| :--- | :--- | :--- |
| < 30% | 🟢 LOW | Allow — log for audit |
| 30% – 60% | 🟡 MEDIUM | Flag for manual review |
| 60% – 85% | 🟠 HIGH | Hold pending investigation |
| > 85% | 🔴 CRITICAL | Block + escalate immediately |

### Offline Availability

> ✅ **Pages ①–⑥ work fully offline** — SHAP explanations, batch scoring, EDA, model metrics, and ablation results require no internet connection. Only the RAG regulatory report on Page ① requires an active Cohere API key and internet access.

### Batch CSV Format (Page ②)

Your CSV must include these exact column names (order does not matter):

```csv
step,type,amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest
1,TRANSFER,181.0,181.0,0.0,0.0,0.0
1,CASH_OUT,9839.64,170136.0,160296.36,0.0,0.0
1,PAYMENT,4024.36,53860.0,49835.64,0.0,0.0
```

**Valid values for `type`:** `CASH_OUT`, `TRANSFER`, `PAYMENT`, `DEBIT`, `CASH_IN`

> [!NOTE]
> The RAG regulatory report on Page 1 requires a Cohere API key. All other pages are fully functional without one.

---

## ⚡ Quick Start

```bash
# 1. Clone
git clone https://github.com/whozahm3d/ai-fraud-detection.git
cd ai-fraud-detection

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch
streamlit run app.py
```

The dashboard loads the **pre-trained XGBoost model** from `outputs/deployment/model.pkl` — no data download or retraining required.

**Alternatively, using Make:**

```bash
make install   # install dependencies
make run       # launch dashboard
make help      # see all available commands
```

---

## 🔧 Full Setup Guide

### Prerequisites

- Python 3.10+
- pip
- (Optional) Cohere API key for the RAG regulatory report feature

### 1. Clone the Repository

```bash
git clone https://github.com/whozahm3d/ai-fraud-detection.git
cd ai-fraud-detection
```

Or open instantly in a zero-setup cloud environment:

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/whozahm3d/ai-fraud-detection)

### 2. Set Up a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For development (notebooks, linting, testing):

```bash
pip install -r requirements-dev.txt
# or: make install-dev
```

<details>
<summary><strong>📦 Full dependency list (click to expand)</strong></summary>

<br>

| Package | Version | Purpose |
| :--- | :--- | :--- |
| `streamlit` | 1.45.0 | Dashboard UI |
| `xgboost` | 2.1.4 | Primary deployed model |
| `scikit-learn` | 1.5.2 | ML pipeline & baselines |
| `imbalanced-learn` | 0.12.4 | SMOTE / ImbPipeline |
| `shap` | ≥0.45, <0.48 | XAI explanations |
| `chromadb` | 0.5.23 | RAG vector store |
| `sentence-transformers` | 3.0.1 | RAG dense embeddings |
| `rank-bm25` | 0.2.2 | RAG lexical retrieval |
| `cohere` | ≥5.0, <6.0 | RAG generation (command-r-plus-08-2024) |
| `pandas` | 2.2.3 | Data processing |
| `numpy` | 1.26.4 | Numerics |
| `scipy` | 1.13.1 | Statistical utilities |
| `matplotlib` | 3.9.4 | Plotting |
| `seaborn` | 0.13.2 | Statistical visualisation |
| `joblib` | 1.4.2 | Model serialisation |
| `psutil` | 6.1.1 | System resource monitoring |
| `tqdm` | 4.66.4 | Progress bars |
| `protobuf` | <6 | ChromaDB compatibility |
| `rich` | <14 | Streamlit compatibility |

</details>

### 4. Configure Cohere API Key (Optional — RAG only)

> [!WARNING]
> **Never commit `.streamlit/secrets.toml`** — it contains your Cohere API key. The file is already in `.gitignore`. Always use `secrets.example.toml` as your template.

```bash
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
# Open secrets.toml and replace sk-proj-YOUR_KEY_HERE with your actual key
```

Or set it as an environment variable:

```bash
export COHERE_API_KEY="your-cohere-key-here"
```

**For Streamlit Cloud deployment:** go to **App Settings → Secrets** and paste the contents of your `secrets.toml`.

### 5. Download the Dataset (Retraining Only)

> [!NOTE]
> The pre-trained model is already included in `outputs/deployment/`. **Skip this step** unless you want to retrain from scratch.

1. Download from [Kaggle — PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1)
2. Place the CSV at `Data/original_dataset/`
3. Run the full training notebook:

```bash
jupyter notebook notebooks/notebooks_final/ai_fraud_detection_final.ipynb
```

### 6. Run the Dashboard

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

### 7. Run Individual Notebooks

```bash
# EDA and preprocessing
jupyter notebook notebooks/notebooks_phase1/exploratory_data_analysis.ipynb

# Full preprocessing + model training pipeline
jupyter notebook notebooks/notebooks_phase2/training_pipeline.ipynb

# Final model, ablation, XAI
jupyter notebook notebooks/notebooks_final/ai_fraud_detection_final.ipynb

# RAG pipeline build + evaluation
jupyter notebook notebooks/notebooks_final/rag_system.ipynb

# Model export + dashboard preparation
jupyter notebook notebooks/notebooks_final/deployment.ipynb
```

---

## 📁 Project Structure

```
ai-fraud-detection/
│
├── app.py                              # Streamlit dashboard (6 pages)
├── rag_module.py                       # RAG pipeline: retrieval + reranking + generation
├── requirements.txt                    # All runtime dependencies with pinned versions
├── requirements-dev.txt                # Dev-only dependencies (jupyter, flake8, pytest)
├── Makefile                            # Common commands: make install, make run, make lint
├── .python-version                     # Python 3.10 specifier
├── LICENSE                             # MIT License
├── CONTRIBUTING.md                     # Contribution guidelines
├── SECURITY.md                         # Security policy and API key guidance
│
├── .streamlit/
│   ├── config.toml                     # Theme config (primary=#49B6E5), 200MB upload limit
│   └── secrets.example.toml           # API key template — copy to secrets.toml (never commit)
│
├── .devcontainer/
│   └── devcontainer.json              # GitHub Codespaces / VS Code dev container config
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                     # CI: dependency install check + flake8 lint
│   ├── ISSUE_TEMPLATE/
│   │   ├── bug_report.md              # Structured bug report form
│   │   └── feature_request.md         # Structured feature request form
│   └── PULL_REQUEST_TEMPLATE.md       # PR checklist template
│
├── Data/
│   ├── original_dataset/
│   │   └── Paysim_dataset             # Raw PaySim CSV (not included — download from Kaggle)
│   └── processed_data/
│       ├── processed_paysim           # Cleaned + feature-engineered dataset
│       ├── train_original             # Pre-SMOTE training split
│       ├── train_after_simulation     # Post fraud simulation
│       ├── train_final_smote          # Post SMOTE (final training data)
│       └── test_original              # Held-out test set
│
├── notebooks/
│   ├── notebooks_phase1/
│   │   ├── eda_pipeline.ipynb                  # Data loading, cleaning, EDA
│   │   └── exploratory_data_analysis.ipynb     # Full EDA with 8+ visualisations
│   ├── notebooks_phase2/
│   │   └── training_pipeline.ipynb             # Feature engineering + model training
│   └── notebooks_final/
│       ├── ai_fraud_detection_final.ipynb      # Full pipeline, ablation, XAI
│       ├── rag_system.ipynb                    # RAG build + evaluation
│       └── deployment.ipynb                    # Model export + dashboard prep
│
├── scripts/
│   ├── scripts_phase1/
│   │   ├── load_dataset.py
│   │   ├── basic_inspection.py
│   │   ├── data_cleaning.py
│   │   ├── feature_engineering.py
│   │   ├── fraud_simulation_engine.py
│   │   ├── preprocessing_pipeline.py
│   │   ├── scaling_dataset.py
│   │   ├── smote.py
│   │   ├── train_test_split.py
│   │   ├── save_cleaned_dataset.py
│   │   ├── libraries_loaded.py
│   │   └── eda_analysis_and_visualization.py
│   └── scripts_phase2/
│       ├── logistic_regression.py
│       ├── random_forest.py
│       ├── neural_networks.py
│       ├── xgboost_regression.py
│       ├── compare_models.py
│       ├── hyperparameter_config.py
│       ├── kfold_validation.py
│       ├── feature_importance.py
│       ├── roc_curves.py
│       ├── visualize_confusion_matrix.py
│       ├── error_analysis.py
│       ├── bias_check.py
│       └── ablation_study.py
│
├── outputs/
│   ├── deployment/                     # ⬅ Loaded by app.py at runtime — required
│   │   ├── model.pkl                  # Trained XGBoost model
│   │   ├── scaler.pkl                 # Fitted StandardScaler
│   │   ├── model_meta.json            # Model metadata + test metrics
│   │   ├── feature_names.json         # Feature list in training order
│   │   └── model_comparison_deploy.png
│   ├── models/                         # All 4 trained model pkl files
│   │   ├── logistic_regression.pkl
│   │   ├── neural_network.pkl
│   │   ├── random_forest.pkl
│   │   ├── xgboost.pkl
│   │   └── scaler.pkl
│   ├── metrics/
│   │   ├── model_comparison.{json,csv}
│   │   ├── hyperparameter_table.csv
│   │   └── {model}_metrics.{json,csv}  # Per-model metrics (4 models × 2 formats)
│   ├── ablation/
│   │   ├── ablation_results.csv
│   │   ├── ablation_study.png
│   │   ├── ablation_heatmap.png
│   │   ├── ablation_delta.png
│   │   ├── ablation_pr_scatter.png
│   │   └── ablation_smote_trend.png
│   ├── plots/                          # Full set of EDA, model, and evaluation plots
│   ├── experiments/
│   │   ├── experiment_results.csv
│   │   └── cleaned_experiment_results.csv
│   └── rag/
│       ├── retrieval_evaluation.csv
│       ├── response_evaluation.csv
│       ├── TXN-CRITICAL-001.txt
│       ├── TXN-HIGH-001.txt
│       ├── TXN-HIGH-002.txt
│       └── TXN-MEDIUM-001.txt
│
├── Images/
│   ├── images_phase1/                  # EDA & preprocessing plots
│   ├── images_phase2/                  # Model training & evaluation plots
│   └── images_final/                   # Final report plots (used in this README)
│
├── SBP_Documents/                      # State Bank of Pakistan regulatory PDFs
│   ├── C1-Annex.pdf
│   ├── C2-Annex-A.pdf
│   ├── CL33-Annex-B.pdf
│   ├── C10-Branchless-Banking-Regulations.pdf
│   └── SME-PRs-Updtd-Jan-2025.pdf
│
├── chroma_db/                          # Pre-built ChromaDB vector store (no rebuild needed)
│   └── chroma.sqlite3
│
└── reports/
    └── Fraud_Detection_Final_Report.tex   # LaTeX source (regenerate PDF if needed)
```

---

## ⚠️ Known Limitations

This project is built on synthetic data. Understand these constraints before drawing any production conclusions:

| Limitation | Detail |
| :--- | :--- |
| **Synthetic data only** | PaySim is calibrated against real data but is not real. Fraud patterns in production are more diverse and adversarial. |
| **Low precision at default threshold** | XGBoost Test Precision = 0.40 — for every 10 flagged transactions, ~6 are false positives. Threshold tuning is required for any real deployment. |
| **RAG latency** | Average report generation takes 4–6 seconds. Not suitable for real-time inline transaction blocking. |
| **No authentication** | The Streamlit dashboard has no login or access control. Do not deploy publicly with live API keys embedded. |
| **File-based vector store** | ChromaDB in this project does not scale to concurrent users. A production RAG system would use a managed vector DB (Pinecone, Weaviate, etc.). |
| **Partial SBP corpus** | Only 5 SBP documents are indexed. Regulatory coverage is incomplete. |
| **No concept drift handling** | No retraining pipeline exists. Fraud patterns shift over time; production systems require ongoing monitoring and retraining. |

---

## 🔧 Troubleshooting

<details>
<summary><strong>❌ <code>chromadb</code> installation fails on Windows</strong></summary>

ChromaDB requires C++ build tools on Windows:

1. Download [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. During install, select **"Desktop development with C++"**
3. Re-run `pip install -r requirements.txt`

Alternatively, use [GitHub Codespaces](https://codespaces.new/whozahm3d/ai-fraud-detection) for a zero-setup Linux environment.

</details>

<details>
<summary><strong>❌ <code>shap</code> version conflict with scikit-learn</strong></summary>

The `requirements.txt` pins `shap>=0.45.0,<0.48.0` for compatibility with `scikit-learn==1.5.2`. If you encounter a conflict, force the correct versions:

```bash
pip install "shap>=0.45.0,<0.48.0" "scikit-learn==1.5.2"
```

</details>

<details>
<summary><strong>❌ <code>FileNotFoundError: outputs/deployment/model.pkl</code></strong></summary>

The pre-trained model is included in the repository. If it is missing, re-clone the repo, or retrain by running:

```bash
jupyter notebook notebooks/notebooks_final/deployment.ipynb
```

</details>

<details>
<summary><strong>❌ RAG report not generating / Cohere error</strong></summary>

1. Confirm your Cohere API key is set in `.streamlit/secrets.toml` or as `COHERE_API_KEY` in your environment.
2. Verify the key is valid and has available credits at [dashboard.cohere.com](https://dashboard.cohere.com).
3. If you see a `RateLimitError`, wait a moment and retry.

</details>

<details>
<summary><strong>❌ Batch CSV upload fails or times out</strong></summary>

The dashboard supports CSVs up to **200MB** (configured in `.streamlit/config.toml`). For larger files, split into chunks and upload separately. Ensure column names match those in the training data.

</details>

---

## 🏆 Hackathon Context

This project was built as a personal hackathon entry exploring explainable AI and regulatory-grounded fraud detection for mobile money systems. The PaySim dataset is synthetic. Results have **not** been validated on real-world production data and should not be used in any live financial system without further validation.

---

## 📖 How to Cite

If you reference this project in academic work, please cite it as:

```bibtex
@misc{aifrauddetection2026,
  title  = {AI Fraud Detection: Explainable AI and RAG-Based Policy Assistance},
  author = {Ahmad, Ali},
  year   = {2026},
  note   = {Hackathon project},
  url    = {https://github.com/whozahm3d/ai-fraud-detection}
}
```

---

## 🤝 Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

**Quick guide:**

```bash
# 1. Fork the repo and create your branch
git checkout -b feature/your-feature-name

# 2. Make your changes and commit
git commit -m "feat: describe your change"

# 3. Push and open a Pull Request
git push origin feature/your-feature-name
```

**Areas where contributions are especially welcome:**

- Additional regulatory document ingestion (FATF, Basel III)
- Docker / containerisation setup
- Unit tests for the preprocessing pipeline
- Additional XAI methods (LIME, Integrated Gradients)
- Real-time streaming inference optimisation

> [!WARNING]
> Do not commit large data files (`Data/original_dataset/`) or your secrets file (`.streamlit/secrets.toml`). Both are already in `.gitignore`.

---

## 👤 Author

| Name | GitHub |
|:---|:---|
| Ali Ahmad | [@whozahm3d](https://github.com/whozahm3d) |

---

## 📄 License

This project is licensed under the [MIT License](LICENSE). Academic use, learning, and extension are encouraged. Attribution appreciated.

---

## 📚 References

- Lopez-Rojas, E.A. et al. (2016). *PaySim: A financial mobile money simulator for fraud detection.* EMSS.
- Chawla, N.V. et al. (2002). *SMOTE: Synthetic Minority Over-sampling Technique.* JAIR, 16, 321–357.
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD 2016.
- Lundberg, S.M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS 30.
- Robertson, S. & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond.* Foundations and Trends in Information Retrieval.
- State Bank of Pakistan. (2023–2025). *AML/CFT Regulations, Branchless Banking Regulations, SME Payment Regulations.* [sbp.org.pk](https://www.sbp.org.pk)

---

<div align="center">

**🤖 AI Fraud Detection** — Built with purpose. Explained with transparency. Grounded in regulation.

<br>

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)]([NEW_DEPLOYMENT_URL])
&nbsp;&nbsp;
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/whozahm3d/ai-fraud-detection)
&nbsp;&nbsp;
[![GitHub Stars](https://img.shields.io/github/stars/whozahm3d/ai-fraud-detection?style=social)](https://github.com/whozahm3d/ai-fraud-detection)

</div>
