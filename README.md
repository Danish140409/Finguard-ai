# FinGuard AI — Explainable AI for Financial Anomaly Detection in Enterprise Accounting Systems

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111.0-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35.0-FF4B4B.svg)](https://streamlit.io/)
[![Docker Compose](https://img.shields.io/badge/Docker-Compose-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Authors:** Shreya Das, Danish, Mayank  
**Target Role:** ML / Fintech (Finance AI, Order-to-Cash Analytics, Internal Audit Analytics)

---

## 1. Executive Summary

**FinGuard AI** is an enterprise-grade anomaly detection and explainability platform built for corporate accounting and Accounts Payable (AP) departments. It monitors invoices, vendor master data, purchase orders, payments, and general ledger journal entries from ERP-style schemas (SAP, Oracle, NetSuite). 

Every transaction is evaluated using an **ensemble of unsupervised and supervised machine learning models** (Isolation Forest, Local Outlier Factor, One-Class SVM, and XGBoost) and scored from 0 to 100 for anomaly/fraud risk. 

Critically, FinGuard AI provides an **auditable explainability layer** backed by game-theoretic **SHAP (Shapley Additive exPlanations)** and **LIME**, coupled with a **deterministic natural-language generator (NLG)** that translates mathematical attributions into plain English for controllers, internal auditors, and Sarbanes-Oxley (SOX) Section 404 compliance.

---

## 2. Core Capabilities & AP Fraud Types Detected

FinGuard AI is pre-configured with feature extractors specifically engineered to catch the most costly accounts payable fraud and error schemes:

| Anomaly Scheme | Risk Pattern & Mathematical Signal | Regulatory / Business Impact |
| :--- | :--- | :--- |
| **Split Invoices** | Frequency spikes of transactions just below approval thresholds (e.g. $49,800 vs $50k limit) | Circumvention of dual-signature authorization controls |
| **Duplicate Payments** | Identical amount paid to the same vendor within a 30-day rolling window | Working-capital leakage (estimated 1-2% of annual corporate spend) |
| **Weekend / Holiday Payments** | Transactions disbursed on non-business days to newly onboarded vendors | Unauthorized weekend treasury activity |
| **Fictitious Vendors** | Invoices submitted within 7–14 days of supplier registration with round-dollar amounts | Shell company procurement fraud |
| **Inflated Invoices** | Invoice amount > 3 standard deviations from vendor historical baseline ($z > 3.0$) | Vendor overbilling or unauthorized price increases |
| **Purchase Order Discrepancies** | Invoice amount exceeds matched PO amount by > 20% | Three-way matching process failure |
| **Round Dollar Anomalies** | Payments with exact zero cents / thousands, lacking itemized line charges | Correlated with unverified manual journal entry overrides |

---

## 3. Architecture & Tech Stack

FinGuard AI adopts a decoupled 3-tier enterprise architecture:

```
┌────────────────────────────────────────────────────────┐
│               Streamlit Multi-Page UI                  │
│   (KPI Dashboard • Transaction Explorer • Vendor Risk  │
│    Explainability Studio • SOX Audit Trail • Reports)  │
└───────────────────────────▲────────────────────────────┘
                            │ REST (JSON / JWT)
┌───────────────────────────▼────────────────────────────┐
│                    FastAPI REST API                    │
│   (Auth • On-demand /predict • Batch Scoring Queue     │
│    Vendor Analytics • SOX-Compliant Audit Logging)     │
└──────────────┬──────────────────────────┬──────────────┘
               │                          │
┌──────────────▼─────────────┐ ┌──────────▼──────────────┐
│     PostgreSQL Database    │ │    Machine Learning &   │
│   (8 Core Tables, Analytical│ │   Explainability Core   │
│   Views, SOX Immutability   │ │ (IF + LOF + OCSVM +    │
│   Audit Triggers)           │ │  XGBoost, SHAP, LIME)  │
└────────────────────────────┘ └─────────────────────────┘
```

- **Database Layer:** PostgreSQL 16 with schema DDL, analytical summary views, and triggers preventing `UPDATE` or `DELETE` on audit tables.
- **Backend API:** FastAPI with Pydantic v2 schemas, JWT authentication, and background task runners.
- **ML & Explainability:** Scikit-learn, XGBoost, SHAP `TreeExplainer`, LIME `LimeTabularExplainer`, and deterministic rule-based NLG.
- **Reporting Engine:** ReportLab generating publication-grade multi-page PDF audit workpapers with controller sign-off blocks.
- **Frontend Dashboard:** Streamlit with interactive Plotly visualisations.

---

## 4. Quickstart — Deployment with Docker (Recommended)

The easiest way to launch the entire FinGuard AI stack (Database, API, and UI) is via Docker Compose:

```bash
# 1. Clone the repository and navigate into the project directory
cd finguard-ai

# 2. Build and start all three containers
docker compose up --build
```

Once running:
- **Streamlit Web UI:** Open [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation (Swagger UI):** Open [http://localhost:8000/docs](http://localhost:8000/docs)
- **PostgreSQL Database:** Accessible on `localhost:5432` (`user: finguard`, `password: finguard_secret`, `db: finguard_db`)

---

## 5. Local Development Setup (Without Docker)

### 1. Environment & Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment (Windows)
venv\Scripts\activate
# Activate virtual environment (macOS/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

```bash
copy .env.example .env
```

### 3. Generate Synthetic ERP Data & Train Models

```bash
# 1. Synthesize 10,000 invoices with labeled AP anomalies
python training/generate_data.py

# 2. Train the 4-model ensemble (IF, LOF, OCSVM, XGBoost) and save artifacts
python training/train.py

# 3. (Optional) Run batch prediction
python training/predict.py
```

### 4. Start the Application Servers

In Terminal 1 (FastAPI Backend):
```bash
uvicorn backend.main:app --reload --port 8000
```

In Terminal 2 (Streamlit Frontend):
```bash
streamlit run frontend/Home.py
```

---

## 6. Running the Automated Test Suite

FinGuard AI includes full unit and integration tests across feature extraction, explainability determinism, and API endpoints:

```bash
pytest -v tests/
```

Test coverage includes:
- `tests/test_feature_pipeline.py`: Validates all 12 AP financial risk features.
- `tests/test_explainability.py`: Ensures SHAP/NLG narratives are deterministic and auditable.
- `tests/test_api.py`: Tests API health checks and access controls.

---

## 7. Project Structure

```
finguard-ai/
├── backend/
│   ├── config.py              # Application configuration & Pydantic settings
│   ├── deps.py                # JWT auth, RBAC, and model cache dependencies
│   ├── main.py                # FastAPI entrypoint, middleware, & lifespan
│   ├── routers/
│   │   ├── auth.py            # Login, logout, user profile
│   │   ├── predict.py         # On-demand & batch scoring, retraining, stats
│   │   ├── reports.py         # CSV & ReportLab PDF export, audit log
│   │   └── transactions.py    # Transaction & vendor inspection, analyst actions
│   └── schemas/
│       └── __init__.py        # Pydantic v2 request/response models
├── database/
│   ├── schema.sql             # PostgreSQL DDL for 8 tables + SOX trigger
│   └── views.sql              # Analytical reporting SQL views
├── explainability/
│   ├── lime_engine.py         # LIME local tabular surrogate explainer
│   ├── nlg_templater.py       # Deterministic natural-language audit narratives
│   └── shap_engine.py         # SHAP TreeExplainer engine (local & global)
├── frontend/
│   ├── app.py                 # Streamlit main dashboard & KPI overview
│   ├── pages/
│   │   ├── 1_Transaction_Explorer.py  # Filterable exception table & adjudication
│   │   ├── 2_Vendor_Risk.py           # Supplier risk directory & scorecard
│   │   ├── 3_Explainability.py        # SHAP waterfall, LIME, & simulator
│   │   ├── 4_Audit_Log.py             # SOX 404 immutable audit trail
│   │   └── 5_Reports.py               # Audit workpaper PDF & CSV downloads
│   └── utils/
│       └── api_client.py      # HTTP client communicating with FastAPI
├── models/
│   └── production/            # Serialized model artifacts (.joblib, metadata.json)
├── reports/
│   └── pdf_generator.py       # ReportLab PDF executive audit workpaper builder
├── tests/
│   ├── test_api.py
│   ├── test_explainability.py
│   └── test_feature_pipeline.py
├── training/
│   ├── generate_data.py       # ERP data generator with 7 injected fraud types
│   ├── predict.py             # Memory-efficient batch prediction pipeline
│   └── train.py               # Ensemble model trainer & evaluation
├── utils/
│   ├── db.py                  # SQLAlchemy session factory & ORM models
│   ├── feature_pipeline.py    # 12-feature AP extraction pipeline
│   └── logger.py              # Structured JSON logging & audit helper
├── docker-compose.yml         # Multi-container orchestration (DB, API, UI)
├── Dockerfile.backend         # Production Dockerfile for FastAPI
├── Dockerfile.frontend        # Production Dockerfile for Streamlit
└── requirements.txt           # Python package specifications
```

---

## 8. License & Governance

This software is developed for enterprise demonstration purposes, aligning with Sarbanes-Oxley (SOX) Section 404 requirements for automated internal controls over financial reporting. Released under the MIT License.
