# Hospital Queue Management – AI & ABM

A professional AI-powered Hospital Queue Management System built for Final Year Project (FYP) presentation.

## 🏥 Overview

This system combines:
- **Artificial Neural Network (ANN)** – Predicts patient wait times using real hospital ER data
- **Agent-Based Model (ABM)** – Simulates patient flow and doctor interactions using Mesa framework
- **Streamlit Dashboard** – Real-time analytics, KPI monitoring, and AI-powered decision support

## 🚀 Features

- 🔴 Live Hospital Congestion Monitoring (Critical / Heavy / Normal)
- 🤖 AI Wait-Time Prediction (ANN Regression Model)
- 📊 Actual vs Predicted Performance Charts (MAE, RMSE, R²)
- 🧠 SHAP Explainability – Feature Importance Visualization
- 🏃 Patient Abandonment Simulation (Patience Threshold)
- 💊 Doctor Fatigue Modeling (Efficiency Decay)
- 🇵🇰 Localized for Pakistan – Pakistani Hospital Names & Triage Labels
- 📥 Downloadable Simulation Report (CSV)

## 🏗️ Tech Stack

| Technology | Purpose |
|---|---|
| Streamlit | Dashboard UI |
| Plotly | Interactive Charts |
| TensorFlow / Keras | ANN Model |
| Mesa | Agent-Based Simulation |
| Scikit-learn | Model Evaluation |
| SHAP | Explainability |
| Pandas / NumPy | Data Processing |

## 📁 Project Structure

```
hospital-management-project/
├── app.py                  # Main Streamlit Dashboard
├── requirements.txt        # Python Dependencies
├── abm/
│   ├── agents.py           # Patient & Doctor Agents
│   └── model.py            # Hospital ABM Model
├── ann/
│   ├── train_model.py      # Classification ANN Training
│   ├── train_regression.py # Regression ANN Training
│   └── predict.py          # Inference Module
├── utils/
│   ├── preprocessing.py    # Data Preprocessing Pipeline
│   └── helper_functions.py # Utility Functions
├── models/                 # Saved Models (not in repo – generate locally)
└── data/                   # Dataset CSVs (not in repo – place locally)
```

## ⚙️ Setup & Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the ANN model (if models/ not present)
python ann/train_regression.py

# 3. Run the Streamlit dashboard
python -m streamlit run app.py
```

## 🇵🇰 Triage Priority Levels

| Label | Meaning |
|---|---|
| 🔴 Fori (Immediate / Critical) | Life-threatening – treat immediately |
| 🟡 Zaruri (Urgent) | Urgent – within 30 minutes |
| 🟢 Kam Zaruri (Semi-Urgent) | Can wait – 1 to 2 hours |
| ⚪ Mamooli (Non-Urgent) | Routine / walk-in |

## 👥 Team

Developed as a Final Year Project (FYP).
