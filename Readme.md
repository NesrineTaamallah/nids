# 🛡️ NetGuard — Real-Time IoT Network Malware Detection

> **Hybrid Machine Learning Platform** · Supervised + Unsupervised · UNSW-NB15 & Kitsune/Mirai

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Results](#key-results)
- [System Architecture](#system-architecture)
- [Machine Learning Pipelines](#machine-learning-pipelines)
  - [Pipeline A — Supervised (UNSW-NB15)](#pipeline-a--supervised-unsw-nb15)
  - [Pipeline B — Unsupervised / Zero-Day (KitNET)](#pipeline-b--unsupervised--zero-day-kitnet)
- [Dataset Overview](#dataset-overview)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [API Reference](#api-reference)
- [NetGuard Interface](#netguard-interface)
- [Methodology](#methodology)
- [Team](#team)

---

## Overview

**NetGuard** is a real-time network intrusion detection system (NIDS) designed for IoT environments. It combines two complementary machine learning approaches in a sequential decision architecture:

- **Supervised learning** — identifies and classifies 9 known attack categories from labeled network flow data (UNSW-NB15 dataset).
- **Unsupervised learning** — detects unknown/zero-day threats by learning the statistical profile of normal traffic and flagging anomalies (KitNET on Mirai/Kitsune dataset).

The platform was developed as an end-to-end project following the **CRISP-DM** methodology, covering data understanding, preprocessing, dimensionality reduction, modeling, evaluation, and deployment into a functional demo interface called **NetGuard**.

### Why This Matters

Traditional IDS solutions (signature-based, static rule sets) are blind to zero-day attacks and unable to scale to the volume and heterogeneity of IoT traffic. Key figures from IBM's 2024 Cost of a Data Breach report:

| Metric | Value |
|---|---|
| Average cost of a data breach | **$4.88 million** |
| Mean time to identify a breach | **204 days** |
| Mean time to contain a breach | **73 days** |
| Cost savings with AI-driven security | **$2.22 million** |

NetGuard addresses these gaps directly through automated, AI-powered, real-time detection.

---

## Key Results

### Supervised Pipeline (UNSW-NB15 · 2,540,044 flows)

| Model | Task | Test F1 | ROC-AUC |
|---|---|---|---|
| Random Forest | Binary (Normal vs Attack) | **0.979** | **1.000** |
| XGBoost | Multi-class (9 attack types) | — | — |
| **Hierarchical Pipeline** | **Binary + Multi-class** | **F1-macro +0.02 vs direct** | — |

**Gains from hierarchical approach over direct multi-class classification:**

| Attack Class | Direct F1 | Hierarchical F1 | Δ |
|---|---|---|---|
| Analysis | ~0.72 | ~0.78 | **+0.06** |
| Backdoor | ~0.68 | ~0.74 | **+0.06** |
| Rare_Attack (Worms/Shellcode) | ~0.62 | ~0.69 | **+0.07** |
| Macro Average | ~0.82 | ~0.84 | **+0.02** |

### Unsupervised Pipeline (Kitsune/Mirai · 764,137 packets)

- Detects **zero-day anomalies** with no labeled attack examples during training.
- Detection threshold φ = `exp(μ + 3σ)` guarantees **99.7% normal traffic coverage**.
- Trained exclusively on 70,000 benign packets; successfully flags 694,000 Mirai attack packets.

---

## System Architecture

```
Incoming Network Flow
        │
        ▼
┌─────────────────────┐
│  Capture Layer       │
│  PyShark / TShark   │
│  CSV Import         │
└────────┬────────────┘
         │  49 UNSW-NB15 features
         ▼
┌─────────────────────────────────────┐
│         PIPELINE A (Supervised)     │
│                                     │
│  ┌─────────────────────────────┐    │
│  │  Random Forest Binary Filter │    │
│  │  Normal vs. Attack           │    │
│  └──────────┬──────────────────┘    │
│             │ Attack?               │
│    NO ──────┘                       │
│    YES ──► XGBoost Multi-Class      │
│            (9 attack categories)    │
└──────────────────────┬──────────────┘
                       │
                       ▼
┌─────────────────────────────────────┐
│        PIPELINE B (Unsupervised)    │
│                                     │
│   AfterImage → 115 temporal         │
│   features → KitNET ensemble        │
│   of autoencoders                   │
│                                     │
│   RMSE > φ  →  Zero-Day Alert       │
│   RMSE ≤ φ  →  Normal               │
└──────────────────────┬──────────────┘
                       │
                       ▼
              ┌────────────────┐
              │  NetGuard UI   │
              │  Alert Feed    │
              │  Live Stats    │
              └────────────────┘
```

---

## Machine Learning Pipelines

### Pipeline A — Supervised (UNSW-NB15)

#### 1. Exploratory Data Analysis

The UNSW-NB15 dataset poses three major challenges, identified during EDA:

- **Extreme skewness** — `dur` skewness ≈ 590; 28 of 35 numerical features have |skewness| > 2.
- **Double class imbalance** — 6.9:1 between Normal and Attack; 1,238:1 within attack types (Generic vs. Worms).
- **High multicollinearity** — `ct_*` features exhibit Pearson correlations of 0.82–0.96; `swin`/`dwin` are perfectly collinear (r = 1.00).

Most discriminant features identified: `sttl` (r = 0.90 with Label), `ct_state_ttl` (r = 0.87).

#### 2. Preprocessing Pipeline

| Step | Strategy |
|---|---|
| Missing values (MNAR) | Context-aware imputation per column (FTP/HTTP features → 0, `service` → `Unknown`) |
| High-cardinality categoricals (`proto`, `sport`, `dsport`) | Frequency encoding |
| Nominal categoricals (`state`, `service`) | One-Hot Encoding |
| Skewed numerical features (31 of 35) | `log1p` transformation |
| Residual skewness | Yeo-Johnson power transform |
| Normalization | `StandardScaler` (fit on train only — no data leakage) |
| Class imbalance | `class_weight='balanced'` (RF), `scale_pos_weight` (XGBoost), SMOTE for rare classes |

**Final feature matrix: 70 features**
- 4 non-skewed numerical
- 31 log-transformed
- 1 frequency-encoded (proto)
- 16 OHE state columns
- 13 OHE service columns
- 3 binary/ordinal
- 2 frequency-encoded ports

#### 3. Dimensionality Reduction (Diagnostic Only)

PCA and Factor Analysis were performed for diagnostic purposes (KMO ≥ 0.8, Bartlett p < 0.001), confirming preprocessing quality and identifying 4 semantic feature groups:

| Factor | Features |
|---|---|
| Volume | `sbytes`, `dbytes`, `Spkts`, `Dpkts`, `Sload`, `Dload` |
| Repetitiveness | `ct_srv_src`, `ct_srv_dst`, `ct_dst_ltm`, `ct_src_ltm` |
| TCP parameters | `synack`, `ackdat`, `tcprtt`, `swin`, `dwin` |
| TTL / Network state | `sttl`, `dttl`, `ct_state_ttl` |

> **Neither PCA nor FA was integrated into the classification pipeline.** Random Forest and XGBoost are intrinsically robust to high dimensionality and multicollinearity — dimensionality reduction would sacrifice interpretability with no performance gain.

#### 4. Hierarchical Modeling

**Stage 1 — Binary Filter (Random Forest)**
- 200 estimators, `class_weight='balanced'`
- 5-fold stratified cross-validation
- Result: **Test F1 = 0.979, AUC = 1.000**
- 1,028 false positives · 1,648 false negatives (threshold tunable below 0.5)

**Stage 2 — Multi-class Classifier (XGBoost)**
- Trained exclusively on attack-labeled samples
- SMOTE applied to minority classes (ratio < 0.10), capped at 5× amplification
- Worms (174 samples) + Shellcode (1,511) merged into `Rare_Attack_Worms_or_Shellcode`
- Hyperparameters: `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`, `reg_alpha=0.1`, `reg_lambda=1.0`, `early_stopping_rounds=20`

---

### Pipeline B — Unsupervised / Zero-Day (KitNET)

#### AfterImage Feature Extraction

AfterImage transforms each raw network packet into a **115-dimensional feature vector** using incremental damped statistics across 5 network channels and 5 temporal windows:

| Channel | Description |
|---|---|
| Source IP | Stats on all packets from source IP |
| Destination IP | Stats on all packets to destination IP |
| IP↔IP | Traffic between a specific IP pair |
| IP↔Port | Source IP ↔ destination port communication |
| Full socket | Complete TCP/UDP flow (src IP, src port, dst IP, dst port) |

For each channel × window (λ ∈ {5, 3, 1, 0.1, 0.01}), mean, variance, and (for select channels) correlation are computed:

```
x⃗ = [μ^λ1_1, σ^λ1_1, ..., μ^λ5_5, σ^λ5_5] ∈ ℝ^115
```

The five λ values cover detection timescales from ~100ms (DDoS burst detection) to ~1 minute (slow covert scans).

#### KitNET Architecture

KitNET is an **online ensemble of autoencoders** operating packet-by-packet without memory storage:

1. **Feature Mapper (FM)** — Correlated Group Discovery algorithm partitions 115 features into m correlated subgroups (maxAE = 10 features per group).
2. **Ensemble Layer** — Each subgroup is processed by a dedicated autoencoder; local RMSE `r_k` measures reconstruction error per group.
3. **Output Autoencoder** — Receives the RMSE vector `[r_1, ..., r_m]` and produces a final anomaly score.

#### Detection Threshold

RMSE scores of normal traffic follow a log-normal distribution. The threshold φ applies the 3σ rule:

```
φ = exp(μ_ln + 3σ_ln)
```

This guarantees 99.7% of normal packets score below φ (minimizing false positives while maintaining high sensitivity).

**Decision rule:**
```
RMSE(x⃗) > φ  →  ANOMALY
RMSE(x⃗) ≤ φ  →  NORMAL
```

---

## Dataset Overview

### UNSW-NB15 (Supervised)

| Property | Value |
|---|---|
| Total flows | 2,540,044 |
| Features | 49 |
| Attack categories | 9 |
| Normal / Attack ratio | 87.4% / 12.6% |
| Source | UNSW Canberra — ACCS Cyber Range |

**Attack categories:** Generic · Exploits · Fuzzers · DoS · Reconnaissance · Analysis · Backdoor · Shellcode · Worms

### Kitsune/Mirai (Unsupervised)

| Property | Value |
|---|---|
| Total packets | 764,137 |
| Benign packets | ~70,000 (9.2%) |
| Mirai attack packets | ~694,000 (90.8%) |
| Format | PCAP (Packet Capture) |
| Source | UCI ML Repository — ID 516 (Mirsky et al., NDSS 2018) |

---

## Project Structure

```
netguard/
├── app.py                          # Flask backend API
├── index.html                      # NetGuard frontend (single-page app)
├── requirements.txt                # Python dependencies
├── .gitignore
│
├── models/                         # Pre-trained model artifacts
│   ├── best_binary_model.pkl       # Random Forest binary classifier
│   ├── scaler_binary.pkl           # StandardScaler (binary pipeline)
│   ├── powertransformer_binary.pkl # Yeo-Johnson transformer (optional)
│   ├── xgb_hierarchical_multiclass.pkl
│   ├── scaler_hierarchical.pkl
│   ├── powertransformer_hierarchical.pkl
│   ├── label_encoder_hierarchical.pkl
│   ├── kitsune_mirai_model.pkl     # Pre-trained KitNET model
│   └── metadata.json               # Feature names & pipeline config
│
└── KitNET-py/                      # KitNET library (clone separately)
    ├── KitNET.py
    ├── AfterImage.py
    └── ...
```

---

## Installation & Setup

### Prerequisites

- Python 3.8+
- TShark / Wireshark (for live capture mode)
- Git

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/netguard.git
cd netguard
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` includes:
```
flask>=3.0.0
flask-cors>=4.0.0
numpy>=1.26.0
pandas>=2.1.0
scikit-learn>=1.4.0
xgboost>=2.0.0
joblib>=1.3.0
pyshark>=0.6
cython
```

### 3. Clone the KitNET library

```bash
git clone https://github.com/ymirsky/KitNET-py.git
```

Place the `KitNET-py/` folder in the same directory as `app.py`.

### 4. Add trained model files

Place all `.pkl` and `metadata.json` files into a `models/` directory next to `app.py`. The backend auto-loads them at startup. If model files are absent, the system runs in **stub mode** (returns mock predictions) — useful for UI development.

### 5. Start the backend

```bash
python app.py
```

The server starts at `http://0.0.0.0:5050`. On startup, the console confirms:

```
🛡  NetGuard backend  →  http://0.0.0.0:5050
   XGBoost ready : True
   KitNET  ready : True
   Feature cols  : 70
```

### 6. Open the interface

Open `index.html` in your browser, or navigate to `http://localhost:5050`.

---

## Usage

### Live Capture Mode

1. Open the NetGuard interface.
2. Click **Live** and select your network interface from the dropdown.
3. Click **Start Capture**.
4. Network flows appear in real-time in the flow table, color-coded by classification:
   - 🟢 Normal
   - 🔴 Attack (with category label)
   - 🟡 Zero-Day / Anomaly (KitNET flag)

> **Note:** Live capture requires TShark to be installed and the application to run with sufficient network permissions.

### CSV Import Mode

1. Click **CSV** in the control panel.
2. Upload a CSV file with UNSW-NB15 compatible columns.
3. Click **Analyze CSV**.
4. Results are displayed in an interactive table with:
   - Predicted label (Normal / Attack)
   - Attack category
   - Confidence score (%)
   - KitNET RMSE score
   - Anomaly flag

The results table (up to 200 rows) can be exported as an enriched CSV with `Predicted_Label` and `Confidence_Score` columns appended.

---

## API Reference

### Health Check

```
GET /health
```
Returns model readiness status.

### List Network Interfaces

```
GET /api/interfaces
```
Returns available network interfaces for live capture.

### Start/Stop Capture

```
POST /api/capture/start   { "interface": "eth0" }
POST /api/capture/stop
```

### Real-Time Event Stream (SSE)

```
GET /api/events
```
Server-Sent Events stream. Each event includes flow analysis results, KitNET phase/progress, alerts, and statistics.

### Analyze CSV

```
POST /api/analyze/csv
Content-Type: multipart/form-data
Body: file=<csv_file>
```

Returns detection results for all rows, summary statistics, and category distribution.

### Alerts & History

```
GET /api/alerts?limit=50
GET /api/stats/history
```

### Debug Endpoints

```
GET /api/debug/features   # Feature pipeline diagnostics
GET /api/debug/capture    # Active flow state
```

---

## NetGuard Interface

The **NetGuard** dashboard provides:

| Component | Description |
|---|---|
| **Control Panel** | Mode selection (Live / CSV), interface picker, start/stop |
| **Network Radar** | Real-time flow visualization |
| **Live Flow Table** | Per-flow predictions with color coding, confidence, and RMSE |
| **Attack Distribution** | Donut chart of detected attack categories |
| **Live Statistics** | Packet count, attack rate, KitNET threshold |
| **Alert Feed** | Chronological list of detected threats |

---

## Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

| Phase | Work Done |
|---|---|
| **Business Understanding** | IoT threat landscape analysis; UNSW-NB15 + Kitsune dataset selection |
| **Data Understanding** | Full EDA: distributions, missing values, correlations, separability analysis |
| **Data Preparation** | Context-aware imputation, encoding, log1p + Yeo-Johnson transforms, SMOTE |
| **Modeling** | Random Forest binary filter + XGBoost multi-class; KitNET unsupervised |
| **Evaluation** | 5-fold stratified CV; F1, AUC, confusion matrices; hierarchical vs. direct comparison |
| **Deployment** | Flask REST API + NetGuard single-page interface; two input modes |

---




---

## References

1. Antonakakis et al. — *Understanding the Mirai Botnet*, USENIX Security 2017
2. Chapman et al. — *CRISP-DM 1.0*, SPSS Inc., 2000
3. Chawla et al. — *SMOTE: Synthetic Minority Over-sampling Technique*, JAIR 2002
4. IBM Security — *Cost of a Data Breach Report 2024*
5. Meidan et al. — *N-BaIoT: Network-based Detection of IoT Botnet Attacks*, IEEE 2018
6. Mirsky et al. — *Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection*, NDSS 2018
7. Moustafa & Slay — *UNSW-NB15: A Comprehensive Dataset for Network IDS*, MilCIS 2015

---

<div align="center">

**NetGuard** · Real-Time IoT Malware Detection · EPT 2024–2025

</div>