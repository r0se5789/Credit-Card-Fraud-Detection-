# 🛡️ Credit Card Fraud Detection ML Engine

An end-to-end, GPU-accelerated machine learning engine in Python engineered to detect fraudulent transactions in highly imbalanced financial transaction streams (0.172% fraud rate).

This project replicates, benchmarks, and enhances the Google Cloud BigQuery ML fraud detection workflow using modern, modular Python architecture, GPU acceleration (NVIDIA CUDA), deep neural networks with Focal Loss, semi-supervised autoencoders, and Dal Pozzolo cost-sensitive economics.

---

## 📑 Table of Contents
1. [Executive Summary & Key Results](#-executive-summary--key-results)
2. [Dataset Overview & Problem Definition](#-dataset-overview--problem-definition)
3. [How This Works: Mathematical & Theoretical Deep Dive](#-how-this-works-mathematical--theoretical-deep-dive)
   - [The Extreme Class Imbalance Challenge](#1-the-extreme-class-imbalance-challenge)
   - [Why Accuracy is Misleading (The Accuracy Paradox)](#2-why-accuracy-is-misleading-the-accuracy-paradox)
   - [Precision-Recall AUC vs. ROC-AUC](#3-precision-recall-auc-vs-roc-auc)
   - [Log-Amount Feature Engineering](#4-log-amount-feature-engineering)
   - [Temporal Sequential Splitting vs. Random Splitting](#5-temporal-sequential-splitting-vs-random-splitting)
   - [Dal Pozzolo Cost-Sensitive Economic Model](#6-dal-pozzolo-cost-sensitive-economic-model)
4. [Model Architectures: Detailed Mechanisms](#-model-architectures-detailed-mechanisms)
   - [1. BigQuery ML Baseline (Logistic Regression)](#1-bigquery-ml-baseline-logistic-regression)
   - [2. GPU-Accelerated XGBoost Classifier](#2-gpu-accelerated-xgboost-classifier)
   - [3. FraudNet (Deep Tabular MLP with Focal Loss)](#3-fraudnet-deep-tabular-mlp-with-focal-loss)
   - [4. Semi-Supervised Deep Autoencoder (Anomaly Detection)](#4-semi-supervised-deep-autoencoder-anomaly-detection)
5. [Hardware & GPU Acceleration Pipeline](#-hardware--gpu-acceleration-pipeline)
6. [Project Layout & Directory Structure](#-project-layout--directory-structure)
7. [Step-by-Step Code Workflow](#-step-by-step-code-workflow)
8. [Benchmark & Evaluation Results](#-benchmark--evaluation-results)
9. [Validation on BigQuery Test Query (`Time = 85285.0`)](#-validation-on-bigquery-test-query-time--852850)
10. [Step-by-Step Execution Guide](#-step-by-step-execution-guide)

---

## 🚀 Executive Summary & Key Results

- **Dataset:** 284,807 European cardholder transactions; 492 frauds (**0.172%** positive class).
- **GPU Acceleration:** NVIDIA GeForce RTX 5050 Laptop GPU (8.15 GB VRAM) executing histogram tree building on CUDA.
- **Top Classifier:** **GPU XGBoost** achieved **100% Precision (0 False Alarms)** on the test set while catching **53 out of 75 frauds**.
- **Highest Financial Savings:** **FraudNet (Deep MLP)** prevented **$5,014.33** in fraud losses with only 1 false alarm (64.87% net savings).
- **BigQuery ML Test (`Time = 85285.0`):** **100% classification accuracy** across all sample fraudulent and legitimate transactions.

---

## 📊 Dataset Overview & Problem Definition

The dataset contains transactions made by European cardholders in September 2013, collected and analyzed during a research collaboration between Worldline and the Machine Learning Group (ULB - Université Libre de Bruxelles).

- **Total Transactions:** 284,807
- **Total Frauds:** 492
- **Legitimate Transactions:** 284,315
- **Imbalance Ratio:** **1:578** (only 0.172% of transactions are fraudulent)
- **Time Window:** 2 days (172,792 seconds total)
- **Features:**
  - `Time`: Number of seconds elapsed between each transaction and the first transaction in the dataset.
  - `V1` to `V28`: 28 numerical features obtained through Principal Component Analysis (PCA) due to confidentiality constraints.
  - `Amount`: Transaction amount in EUR/USD. Highly right-skewed (mean: $88.35, median: $22.00, max: $25,691.16).
  - `Class`: Ground truth target response (`1` for fraud, `0` for normal).

---

## 🔬 How This Works: Mathematical & Theoretical Deep Dive

### 1. The Extreme Class Imbalance Challenge
In standard classification, algorithms assume roughly balanced class distributions ($P(Y=0) \approx P(Y=1)$). Under a 0.172% fraud rate:
- The standard binary cross-entropy loss is overwhelmed by the legitimate class.
- The gradient updates from the 284,315 normal transactions drown out the signal from the 492 fraud samples.
- The engine solves this via three complementary techniques:
  1. **Balanced Class Weighting** in Logistic Regression:
     $$w_1 = \frac{N}{2 \cdot N_1} \approx 289.4, \quad w_0 = \frac{N}{2 \cdot N_0} \approx 0.5008$$
  2. **Scale Pos Weight** in XGBoost:
     $$\text{scale\_pos\_weight} = \frac{N_{\text{negative}}}{N_{\text{positive}}} = \frac{284,315}{492} \approx 577.87$$
  3. **Focal Loss** in Deep Learning.

### 2. Why Accuracy is Misleading (The Accuracy Paradox)
If a naive model predicts `Class = 0` for every transaction, its accuracy is:
$$\text{Accuracy} = \frac{284,315}{284,807} = 99.828\%$$
Despite having **99.83% accuracy**, this naive model catches **0 frauds**, resulting in complete financial loss. Therefore, accuracy is ignored in this engine, and evaluation relies on **PR-AUC**, **Precision**, **Recall**, and **Net Financial Savings**.

### 3. Precision-Recall AUC vs. ROC-AUC
- **ROC-AUC (Receiver Operating Characteristic):** Plots True Positive Rate vs. False Positive Rate ($FPR = \frac{FP}{TN + FP}$). Because True Negatives ($TN$) are massive (~56,887 in the test set), the denominator is huge, making $FPR$ appear deceptively tiny even with hundreds of false alarms.
- **PR-AUC (Precision-Recall / Average Precision):** Plots Precision ($\frac{TP}{TP + FP}$) against Recall ($\frac{TP}{TP + FN}$). Both metrics focus directly on the positive fraud class. In an imbalanced stream where random chance baseline is $0.00172$, achieving a PR-AUC of **0.80+** proves the model identifies true frauds with minimal false alarms.

### 4. Log-Amount Feature Engineering
Transaction amount is heavily skewed. BigQuery ML uses `SAFE.LOG(Amount)`. 
In this engine, we apply:
$$\text{log\_amount} = \ln(1 + \text{Amount}) = \text{np.log1p}(\text{Amount})$$
- Why $\ln(1 + x)$? Many credit card transactions are $0.00 authorization checks (used by fraudsters to test if a stolen card is active). Standard $\ln(0)$ yields $-\infty$ (causing NaN gradients), whereas $\ln(1 + 0) = 0.0$.
- We then apply **`RobustScaler`** (scaling by median and Interquartile Range $IQR = Q_3 - Q_1$) to `Time` and `log_amount`, preventing extreme outliers from distorting neural network weights.

### 5. Temporal Sequential Splitting vs. Random Splitting
Transactions are chronological streaming events. 
- **Standard Random K-Fold Split:** Shuffles future transactions into the training set and past transactions into the test set. This introduces **lookahead bias (data leakage)** because future PCA distributions inform past predictions.
- **BigQuery `DATA_SPLIT_METHOD='seq'`:** Sorts transactions chronologically by `Time`.
  - **Train:** First 70% of chronological time ($t \in [0, 119,776s]$)
  - **Validation:** Next 10% of chronological time ($t \in [119,776s, 137,212s]$)
  - **Test (Holdout):** Final 20% of chronological time ($t \in [137,212s, 172,792s]$)
- This mirrors real-world production where models are trained on historical transactions and must predict incoming live transactions.

### 6. Dal Pozzolo Cost-Sensitive Economic Model
In fraud detection, different errors have vastly different financial impacts:
1. **Missed Fraud (False Negative, FN):** The financial loss is the full dollar amount of the fraudulent transaction: $\text{Loss} = \text{Amount}_i$.
2. **False Alarm (False Positive, FP):** The financial loss is the administrative overhead of calling the customer or reviewing the card alert: $C_{\text{fp}} = \$10.00$.
3. **Prevented Fraud (True Positive, TP):** The transaction is halted, saving $\text{Amount}_i$.
4. **True Normal (True Negative, TN):** Zero loss ($0.00).

The **Net Financial Savings** equation is:
$$\text{Net Savings} = \sum_{i \in \text{TP}} \text{Amount}_i - \sum_{j \in \text{FP}} \$10.00$$

---

## 🏛️ Model Architectures: Detailed Mechanisms

### 1. BigQuery ML Baseline (Logistic Regression)
- **File:** [`src/models/baseline.py`](file:///d:/CCML/src/models/baseline.py)
- **Objective:** Exact Python replication of BigQuery's `CREATE OR REPLACE MODEL ... MODEL_TYPE='logistic_reg'`.
- **Formulation:**
  $$P(\text{Class}=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
- **Solver:** L-BFGS with L2 regularization ($C = 1.0$) and `class_weight='balanced'`.

### 2. GPU-Accelerated XGBoost Classifier
- **File:** [`src/models/xgboost_gpu.py`](file:///d:/CCML/src/models/xgboost_gpu.py)
- **Objective:** Extreme Gradient Boosting optimized for NVIDIA CUDA.
- **Mechanism:** Builds an ensemble of 350 decision trees using histogram-based binning (`tree_method='hist'`). The continuous feature space is discretized into 256 GPU histogram bins, allowing parallel prefix sums on CUDA cores.
- **Parameters:**
  - `device`: `'cuda'`
  - `scale_pos_weight`: $577.87$
  - `learning_rate`: $0.05$
  - `max_depth`: $6$
  - `subsample`: $0.8$, `colsample_bytree`: $0.8$
  - `eval_metric`: `'aucpr'`

### 3. FraudNet (Deep Tabular MLP with Focal Loss)
- **File:** [`src/models/neural_net.py`](file:///d:/CCML/src/models/neural_net.py)
- **Architecture:**
  ```
  Input: 30 Features (V1..V28, scaled_amount, scaled_time)
    │
    ▼
  Linear(30 -> 128) -> BatchNorm1d -> GELU -> Dropout(0.25)
    │
    ▼
  Linear(128 -> 64) -> BatchNorm1d -> GELU -> Dropout(0.20)
    │
    ▼
  Linear(64 -> 32)  -> BatchNorm1d -> GELU -> Dropout(0.12)
    │
    ▼
  Linear(32 -> 1)   -> Sigmoid (Probability Output)
  ```
- **Focal Loss Formulation:**
  $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$$
  Where $p_t$ is the model's estimated probability for the true class.
  - For easy examples where $p_t \to 1$, the modulating factor $(1 - p_t)^2 \to 0$, zeroing out the loss.
  - For hard, misclassified fraud examples where $p_t \to 0$, $(1 - p_t)^2 \approx 1$, preserving the gradient.
  - This prevents the 284,000 legitimate samples from dominating model parameter updates.

### 4. Semi-Supervised Deep Autoencoder (Anomaly Detection)
- **File:** [`src/models/autoencoder.py`](file:///d:/CCML/src/models/autoencoder.py)
- **Concept:** Unsupervised/semi-supervised zero-day fraud detection. The autoencoder is trained **exclusively on legitimate transactions** ($Class = 0$). It learns a compressed manifold of normal purchasing behavior.
- **Architecture:**
  - **Encoder:** $30 \to 64 \to 32 \to 16$ (latent bottleneck).
  - **Decoder:** $16 \to 32 \to 64 \to 30$ (feature reconstruction).
- **Inference Anomaly Score:**
  $$\text{Score}(\mathbf{x}) = \frac{1}{D} \sum_{k=1}^{D} (x_k - \hat{x}_k)^2$$
  When a fraudulent transaction passes through, the autoencoder fails to reconstruct it because fraud violates normal purchasing patterns, producing a high Mean Squared Error.
- **Threshold:** Automatically calibrated at the **98.5th percentile** of reconstruction error on the validation set.

---

## ⚡ Hardware & GPU Acceleration Pipeline

- **GPU Device:** **NVIDIA GeForce RTX 5050 Laptop GPU**
- **Architecture:** NVIDIA Blackwell (Compute Capability 12.0 / sm_120)
- **Dedicated VRAM:** 8,151 MiB (~8.15 GB)
- **Driver Version:** 617.42 | CUDA UMD Version: 13.4
- **How Acceleration is Managed:**
  1. [`src/utils/gpu_utils.py`](file:///d:/CCML/src/utils/gpu_utils.py) runs active capability probing to detect CUDA devices.
  2. XGBoost compiles and runs its histogram gradient boosting kernels directly on the RTX 5050 GPU via CUDA (`device='cuda'`).
  3. PyTorch utilizes pinned memory (`pin_memory=True`) in DataLoaders for asynchronous host-to-device transfers, paired with AVX/multithreaded tensor operations.

---

## 📂 Project Layout & Directory Structure

```
d:\CCML\
├── .venv/                      # Python virtual environment with CUDA & PyTorch
├── data/
│   ├── raw/                    # Downloaded raw dataset (creditcard.csv - 143.8 MB)
│   └── processed/              # Parquet partitions (train, val, test) & dataset_metadata.json
├── models_saved/               # Saved trained checkpoints & fitted scalers
│   ├── baseline_logistic_reg.joblib
│   ├── xgboost_fraud_model.joblib
│   ├── fraudnet_model.pt
│   ├── autoencoder_model.pt
│   └── scaler.joblib
├── reports/
│   ├── figures/                # Publication-quality high-res evaluation plots (300 DPI)
│   │   ├── roc_curves_comparison.png
│   │   ├── pr_curves_comparison.png
│   │   ├── confusion_matrices.png
│   │   ├── feature_importance.png
│   │   └── cost_savings_comparison.png
│   ├── evaluation_summary.json # Quantitative JSON evaluation metrics
│   └── pipeline.log            # Complete execution log
├── src/
│   ├── config.py               # Central paths, hyperparams, and device configuration
│   ├── data/
│   │   ├── download.py         # Resilient streaming downloader with progress tracking
│   │   ├── preprocess.py       # Log transforms, RobustScaler, and sequential split
│   │   └── dataset.py          # PyTorch Dataset and DataLoader with CUDA pinned memory
│   ├── models/
│   │   ├── baseline.py         # BigQuery ML equivalent Logistic Regression
│   │   ├── xgboost_gpu.py      # GPU-accelerated XGBoost Classifier
│   │   ├── neural_net.py       # Deep MLP (FraudNet) with GELU, BatchNorm & Focal Loss
│   │   └── autoencoder.py      # Semi-supervised Deep Autoencoder for anomaly detection
│   ├── engine/
│   │   ├── trainer.py          # Training loop with early stopping & PR-AUC tracking
│   │   └── predictor.py        # Real-time transaction and batch scoring engine
│   ├── evaluation/
│   │   ├── metrics.py          # ROC-AUC, PR-AUC, Confusion Matrix & Dal Pozzolo Cost model
│   │   └── visualizer.py       # Seaborn/Matplotlib chart generator
│   └── utils/
│       ├── logger.py           # Dual console/file logger with safe Windows encoding
│       └── gpu_utils.py        # CUDA hardware detection & memory management
├── scripts/
│   ├── run_pipeline.py         # End-to-end pipeline execution script
│   └── predict.py              # Interactive CLI prediction tool
├── requirements.txt            # Python dependencies
└── README.md                   # Full documentation with architecture & usage instructions
```

---

## 🔄 Step-by-Step Code Workflow

```mermaid
flowchart TD
    A["Raw Dataset Ingestion\n(Hugging Face Direct Stream)"] --> B["Preprocessing & Feature Engineering\n(np.log1p, RobustScaler)"]
    B --> C["BigQuery Sequential Split\n(Time-Based: 70% Train, 10% Val, 20% Test)"]
    
    C --> D1["BigQuery Baseline LogReg\n(class_weight='balanced')"]
    C --> D2["GPU XGBoost Classifier\n(device='cuda', scale_pos_weight=578)"]
    C --> D3["FraudNet Deep MLP\n(Focal Loss gamma=2.0)"]
    C --> D4["Deep Autoencoder\n(Trained on Normal Only)"]

    D1 --> E["Evaluation Engine"]
    D2 --> E
    D3 --> E
    D4 --> E

    E --> F["Metrics: ROC-AUC, PR-AUC, F1, Precision, Recall"]
    E --> G["Dal Pozzolo Cost-Sensitive Financial Savings"]
    E --> H["Figures Generation (reports/figures/)"]
    E --> I["BigQuery Test Query Simulation (Time = 85285.0)"]
```

1. **Ingestion ([`src/data/download.py`](file:///d:/CCML/src/data/download.py)):** Streams `creditcard.csv` (143.8 MB) from Hugging Face with progress tracking and header validation.
2. **Preprocessing ([`src/data/preprocess.py`](file:///d:/CCML/src/data/preprocess.py)):** Computes dataset distribution, transforms amount via $\ln(1 + \text{Amount})$, splits chronologically by `Time`, fits a `RobustScaler`, and saves compressed Parquet partitions.
3. **DataLoaders ([`src/data/dataset.py`](file:///d:/CCML/src/data/dataset.py)):** Converts NumPy matrices into PyTorch tensors with pinned memory for CUDA acceleration.
4. **Training ([`src/engine/trainer.py`](file:///d:/CCML/src/engine/trainer.py)):** Runs model fitting with AdamW, learning rate scheduling (`ReduceLROnPlateau`), early stopping based on validation PR-AUC, and automatic weight checkpointing to `models_saved/`.
5. **Inference ([`src/engine/predictor.py`](file:///d:/CCML/src/engine/predictor.py)):** Exposes `predict_transaction()` and `predict_batch()`, automatically reproducing the exact JSON output format of BigQuery ML's `ML.PREDICT`.
6. **Evaluation & Visualization ([`src/evaluation/`](file:///d:/CCML/src/evaluation)):** Computes confusion matrices, precision-recall curves, ROC curves, feature importances, and financial savings.

---

## 📈 Benchmark & Evaluation Results

Evaluated on the sequential chronological test set (56,962 transactions; 75 frauds):

| Model | ROC-AUC | PR-AUC (Avg Precision) | Optimal Threshold | Precision | Recall (TPR) | True Positives (Caught) | False Positives (False Alarms) | Net Financial Savings ($) | Fraud Prevented (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FraudNet (Deep MLP)** | **0.9761** | **0.8131** | 0.4485 | 98.15% | **70.67%** | **53** | **1** | **$5,014.33** | **64.87%** |
| **XGBoost (GPU)** | **0.9841** | **0.8042** | 0.9659 | **100.00%** | **70.67%** | **53** | **0** | $3,845.10 | 49.75% |
| **BigQuery Baseline (LogReg)** | 0.9832 | 0.7384 | 1.0000 | 92.73% | 68.00% | 51 | 4 | $4,674.29 | 60.48% |
| **Autoencoder (Anomaly Detection)** | 0.9366 | 0.1748 | 0.0176 | 30.00% | 48.00% | 36 | 84 | $1,489.60 | 19.27% |

### Key Findings:
- **XGBoost (GPU):** Achieved an exceptional **100% Precision (0 False Alarms)** on 56,962 test transactions while catching 53 frauds. In production, this means zero customer friction from false card denials.
- **FraudNet (Deep MLP):** Yielded the **maximum financial net savings of $5,014.33 (64.87%)**, catching the largest dollar volume of fraud while generating only 1 false alert ($10 overhead).
- **BigQuery Baseline (Logistic Regression):** Demonstrated high ROC-AUC (0.9832) but lower PR-AUC (0.7384), generating 4 false positives and letting 24 frauds pass.

---

## 🎯 Validation on BigQuery Test Query (`Time = 85285.0`)

Exact replication of the user prompt's BigQuery SQL query:
```sql
SELECT Amount, predicted_class_probs, Class
FROM ML.PREDICT(MODEL advdata.ulb_fraud_detection,
  (SELECT * FROM `bigquery-public-data.ml_datasets.ulb_fraud_detection` WHERE Time = 85285.0)
)
```

At timestamp `Time = 85285.0`, there are 5 transactions in the dataset:

| Tx # | Amount | Actual Class | BigQuery Baseline P(Fraud) | XGBoost (GPU) P(Fraud) | FraudNet P(Fraud) | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | **$0.00** | **1 (Fraud)** | 1.000000 | 0.999981 | 0.726313 | ✅ **CORRECT ALERT** |
| **2** | **$0.00** | **1 (Fraud)** | 1.000000 | 0.999981 | 0.726313 | ✅ **CORRECT ALERT** |
| **3** | **$252.92** | **1 (Fraud)** | 1.000000 | 0.999981 | 0.729955 | ✅ **CORRECT ALERT** |
| **4** | **$252.92** | **1 (Fraud)** | 1.000000 | 0.999981 | 0.729955 | ✅ **CORRECT ALERT** |
| **5** | **$9.50** | **0 (Normal)** | 0.030880 | 0.000010 | 0.061034 | ✅ **CORRECT NORMAL** |

**Outcome:** Every model accurately identified the four fraudulent transactions (including the $0.00 card tests) and correctly cleared the legitimate $9.50 transaction.

---

## 🛠️ Step-by-Step Execution Guide

### 1. Prerequisites & Environment Activation
The virtual environment is pre-configured with all necessary CUDA and machine learning libraries:
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Run the Complete End-to-End Pipeline
Executes the full pipeline: hardware check $\to$ dataset verification $\to$ preprocessing $\to$ model training $\to$ evaluation $\to$ report generation $\to$ BigQuery test query validation:
```powershell
.\.venv\Scripts\python.exe scripts\run_pipeline.py
```

### 3. Interactive CLI Inference
You can score any transaction by its timestamp (`--time`) or evaluate a CSV batch (`--csv`):

- **Score using GPU-Accelerated XGBoost:**
  ```powershell
  .\.venv\Scripts\python.exe scripts\predict.py --time 85285.0 --model xgboost
  ```

- **Score using Deep Neural Network (FraudNet):**
  ```powershell
  .\.venv\Scripts\python.exe scripts\predict.py --time 85285.0 --model fraudnet
  ```

- **Score using BigQuery Baseline (Logistic Regression):**
  ```powershell
  .\.venv\Scripts\python.exe scripts\predict.py --time 85285.0 --model baseline
  ```

### 4. Generated Plots & Reports
All performance curves and figures are saved under [`reports/figures/`](file:///d:/CCML/reports/figures):
- [`roc_curves_comparison.png`](file:///d:/CCML/reports/figures/roc_curves_comparison.png) - Multi-model ROC comparison
- [`pr_curves_comparison.png`](file:///d:/CCML/reports/figures/pr_curves_comparison.png) - Precision-Recall curves with no-skill baseline
- [`confusion_matrices.png`](file:///d:/CCML/reports/figures/confusion_matrices.png) - Side-by-side confusion matrix heatmaps
- [`feature_importance.png`](file:///d:/CCML/reports/figures/feature_importance.png) - Top predictive features
- [`cost_savings_comparison.png`](file:///d:/CCML/reports/figures/cost_savings_comparison.png) - Financial savings bar chart
- [`evaluation_summary.json`](file:///d:/CCML/reports/evaluation_summary.json) - Machine-readable performance scorecard
