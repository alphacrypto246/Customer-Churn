# 📊 Customer Churn Predictor — End-to-End Machine Learning System

An enterprise-grade, end-to-end Machine Learning web application designed to predict customer churn probability in the telecommunications sector. Built with **FastAPI**, **Scikit-Learn / XGBoost / CatBoost**, and an interactive **Amber & Onyx high-contrast frontend**.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Feature Telemetry Dictionary](#-feature-telemetry-dictionary)
- [Tech Stack](#-tech-stack)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Model Training & Pipelines](#-model-training--pipelines)
- [Running the Application](#-running-the-application)
- [API Reference](#-api-reference)
- [Frontend Experience & Rule-Based Risk Indicators](#-frontend-experience--rule-based-risk-indicators)

---

## 🌟 Overview

Customer attrition (churn) directly impacts subscription recurring revenues. This project provides:
1. **Automated ML Pipelines**: Complete lifecycle from data ingestion (MongoDB/CSV), data transformation with preprocessing scalers, and multi-model training/evaluation.
2. **High-Performance Inference**: FastAPI REST backend exposing real-time prediction and probability scoring endpoints.
3. **Interactive UI**: Sleek frontend designed with a high-impact **Amber (`#FFBE0B`) & Onyx Black (`#101010`)** aesthetic, real-time probability progress animations, sample profile presets, and risk assessment insights.

---

## ✨ Key Features

- **End-to-End Pipeline**: Modular architecture separating Ingestion, Transformation, Model Training, and Prediction.
- **Ensemble ML Support**: Evaluates multiple algorithms including **CatBoost**, **XGBoost**, **LightGBM**, and **Random Forest**.
- **Real-Time Probability Estimation**: Computes both class verdict (`Churn` / `No Churn`) and calibrated probability percentages.
- **Interactive Preset Simulator**: Load pre-configured customer profiles (*Loyal Customer*, *Standard User*, *High Churn Risk*) with one click.
- **Dynamic Heuristic Insights**: Automatically highlights key churn risk drivers (excess customer support friction, unrenewed contracts, overage bill shock).
- **Embedded API Inspector**: Live view of outgoing JSON request payloads and incoming backend responses.

---

## 🏗 System Architecture

```
[ Customer Data / Presets ]
           │
           ▼
[ Interactive Frontend UI ] (HTML5 / Vanilla CSS / JavaScript)
           │
           │  POST /predict (JSON Payload)
           ▼
[ FastAPI Application ] (main.py)
           │
           ▼
[ Predict Pipeline ] (src/pipeline/predict_pipeline.py)
   ├── Preprocessor (artifacts/preprocessor.pkl) ──> Feature Scaling & Encoding
   └── Trained Model (artifacts/model.pkl)        ──> Classification & Probabilities
           │
           ▼
[ JSON Prediction & Risk Score Response ]
```

---

## 📋 Feature Telemetry Dictionary

The model accepts 10 core telemetry variables:

| Feature Name | Type | Description | Example Range |
| :--- | :---: | :--- | :---: |
| `AccountWeeks` | `int` | Duration in weeks customer has held an active account | `1 - 500` |
| `ContractRenewal`| `int` | Whether customer renewed contract recently (`1` = Yes, `0` = No) | `0` or `1` |
| `DataPlan` | `int` | Cellular data subscription status (`1` = Active, `0` = None) | `0` or `1` |
| `DataUsage` | `float` | Monthly cellular data traffic consumed in Gigabytes (GB) | `0.0 - 50.0` |
| `CustServCalls` | `int` | Number of calls placed to customer support | `0 - 25` |
| `DayMins` | `float` | Average daytime voice call minutes per month | `0.0 - 800.0` |
| `DayCalls` | `int` | Total count of daytime voice calls made | `0 - 300` |
| `MonthlyCharge` | `float` | Base average monthly bill in USD ($) | `$10.0 - $300.0` |
| `OverageFee` | `float` | Peak overage fee charged in last 12 months in USD ($) | `$0.0 - $100.0` |
| `RoamMins` | `float` | Average monthly roaming minutes outside home network | `0.0 - 100.0` |

---

## 🛠 Tech Stack

- **Backend**: Python 3.10+, [FastAPI](https://fastapi.tiangolo.com/), [Uvicorn](https://www.uvicorn.org/), Pydantic
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/), [XGBoost](https://xgboost.readthedocs.io/), [CatBoost](https://catboost.ai/), [LightGBM](https://lightgbm.readthedocs.io/), Joblib, Pandas, NumPy
- **Frontend**: HTML5, Vanilla CSS3 (Custom Design System), JavaScript (ES6+ Fetch API), Google Fonts (*Outfit* & *JetBrains Mono*)
- **Database / Data Source**: MongoDB ([PyMongo](https://pymongo.readthedocs.io/)) / CSV storage

---

## 📂 Project Directory Structure

```
├── artifacts/                  # Serialized artifacts (model.pkl, preprocessor.pkl, raw data)
│   ├── model.pkl
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── test.csv
│   └── train.csv
├── logs/                       # Application and training log files
├── notebooks/                  # EDA and experimentation Jupyter notebooks
├── src/                        # Core application package
│   ├── components/             # Pipeline components
│   │   ├── data_ingestions.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── database/               # Database connection utilities
│   ├── pipeline/               # Execution pipelines
│   │   ├── predict_pipeline.py
│   │   └── train_pipeline.py
│   ├── exception.py            # Custom exception handling
│   ├── logger.py               # Centralized logging configuration
│   └── utils.py                # Helper functions (save/load objects)
├── static/                     # Frontend static assets
│   ├── script.js               # Frontend controller & API integration
│   └── style.css               # Amber & Onyx design system stylesheet
├── templates/                  # Frontend HTML templates
│   └── index.html              # Main dashboard view
├── .env                        # Environment variables (MongoDB URI, etc.)
├── main.py                     # FastAPI entry point & route definitions
├── pyproject.toml              # Build configuration
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone Repository
```bash
git clone https://github.com/your-username/Customer-Churn.git
cd Customer-Churn
```

### 2. Create and Activate a Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🔄 Model Training & Pipelines

To execute data ingestion, preprocessing, and model training from scratch:

```bash
# Run training pipeline
python -m src.pipeline.train_pipeline
```

Upon completion, updated artifacts (`model.pkl` and `preprocessor.pkl`) will be saved in the `artifacts/` folder.

---

## 🌐 Running the Application

Start the FastAPI server with Uvicorn:

```bash
# Start server with auto-reload
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

Once started:
- 🖥️ **Web Dashboard**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- 📖 **Interactive API Documentation (Swagger UI)**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- 📑 **ReDoc Documentation**: [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

---

## 🔌 API Reference

### 1. Root / UI Route
- **URL**: `/`
- **Method**: `GET`
- **Description**: Serves the frontend web dashboard (`templates/index.html`).

---

### 2. Churn Prediction Route
- **URL**: `/predict`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`

#### Request Body Example:
```json
{
  "AccountWeeks": 38,
  "ContractRenewal": 0,
  "DataPlan": 0,
  "DataUsage": 0.0,
  "CustServCalls": 5,
  "DayMins": 295.4,
  "DayCalls": 130,
  "MonthlyCharge": 89.50,
  "OverageFee": 19.80,
  "RoamMins": 15.6
}
```

#### Successful Response (`200 OK`):
```json
{
  "prediction": "Churn",
  "churn_probability": 0.9716
}
```

---

## 🎯 Frontend Experience & Rule-Based Risk Indicators

The frontend UI evaluates key customer risk indicators in real-time alongside model output:
- 🚨 **High Support Activity**: Flags customers exceeding 3+ support calls signaling potential friction.
- ⚠️ **Unrenewed Contract**: Highlights increased flight risk when contracts lapse without renewal.
- 💸 **Overage Surcharges**: Identifies potential bill shock when peak overage fees exceed thresholds.
- 🟢 **Account Tenure**: Identifies high-loyalty accounts with long tenure.

---

## 📜 License
This project is licensed under the MIT License.
