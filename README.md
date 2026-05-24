# EPRA-ANALYSIS
# ⚡ EPRA Smart Grid: Predictive Dispatch Optimization Engine

An end-to-end Machine Learning and Time-Series forecasting architecture built to optimize national electricity grids. This application enables energy regulatory bodies like the **Energy and Petroleum Regulatory Authority (EPRA)** to predict peak load demands, maximize renewable energy utility (Geothermal/Hydro), and securely mitigate the risks of regional load-shedding.

## 📊 Project Performance Summary
* **Algorithm:** Extreme Gradient Boosting (XGBoost Regressor)
* **Target Metric Accuracy:** Mean Absolute Error (MAE) of **1,682.67 MW**
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

## 📋 Operational Decision Framework Built-In
The dashboard houses an automated recommendation engine that interprets predictions for grid operations:
* **High-Demand Quantile (>85th percentile):** Formulates direct alerts to ramp up secondary base-loads instantly to stabilize peak cooking/lighting hour loads.
* **Low-Valley Quantile (<15th percentile):** Instructs control centers to dial back expensive thermal generation, saving millions by relying 100% on eco-friendly geothermal baseloads.
