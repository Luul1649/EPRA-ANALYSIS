# ⚡ Smart Grid Demand Forecasting: A Framework for EPRA Compliance

This project is a high-performance **Proof of Concept (PoC)** designed to demonstrate how Machine Learning can optimize national grid dispatch operations. Built using a comprehensive energy dataset from **Kaggle**, the core architecture is engineered to align with the infrastructure forecasting needs of energy regulatory bodies like Kenya's **Energy and Petroleum Regulatory Authority (EPRA)**.

## 📈 Methodology & Data Strategy
Due to the confidentiality of local grid utility metrics, this system was developed using a world-class historical load dataset from Kaggle to build and validate the predictive logic. The underlying machine learning pipeline is fully decoupled and ready to plug directly into EPRA's real-time SCADA or API infrastructure.

## 📊 Project Performance Summary
* **Source Dataset:** Kaggle Hourly Energy & Weather Data
* **Algorithm:** Extreme Gradient Boosting (XGBoost Regressor)
* **Model Baseline Accuracy:** Mean Absolute Error (MAE) of **1,682.67 MW**
* **Primary Drivers Identified:** Historical Load Lag (24 Hours Prior), Time of Day Peaks, and Weekend Industrial Drops.
* **Interface Deployment:** Streamlit Web Application Framework

## ⚙️ Core Architecture & Pipeline
1. **Time-Series Engineering:** Transformed raw text timestamps into highly structural datetime properties (`hour`, `day_of_week`, `month`, `is_weekend`).
2. **Lag Parameter Analytics:** Ingested historical 24-hour lag periods (`load_lag_24h`) to provide the model with a baseline memory of grid behavior.
3. **Data Robustness:** Integrated forward-fill (`ffill`) and backward-fill (`bfill`) methods to seamlessly recover missing utility sensor metrics without data loss.
4. **Chronological Splitting:** Enforced an 80/20 chronological dataset split (`shuffle=False`) to guarantee zero data leakage and preserve historical timeline integrity.

## 🚀 How to Run the App Locally

### 1. Prerequisites
Ensure you have Python installed, then set up the required dependencies:
```bash
pip install pandas numpy xgboost scikit-learn matplotlib seaborn streamlit
```

### 2. File Organization
Place your source files and dataset in the same directory:
```text
📂 Project-Folder/
├── energy_dataset.csv
└── app.py
```

### 3. Execution
Navigate to the directory in your Command Prompt/Terminal and fire up the dashboard:
```bash
streamlit run app.py
```

