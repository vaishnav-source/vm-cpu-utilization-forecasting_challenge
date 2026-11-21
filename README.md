# VM Cluster CPU Utilization Forecasting

A machine learning time-series regression project that predicts aggregate cluster CPU utilization at 5-minute intervals using historical lag features and active virtual machine (VM) counts.

## 📌 Project Overview
Efficient resource allocation in cloud environments requires accurate short-term workload forecasting. In this project, raw multi-VM telemetry logs are aggregated into 5-minute intervals to predict cluster-wide CPU utilization for the next time step. 

To ensure realistic evaluation and strictly prevent **data leakage**, all Exploratory Data Analysis (EDA) and feature transformations respect chronological boundaries.

## 🛠️ Complete End-to-End Workflow

1. **Data Aggregation (`GROUP BY`):**
   Grouped raw VM logs by `timestamp` into 5-minute intervals to calculate `aggregate_cpu_utilization` and `active_vm_count` (creating 122 chronological time steps), verified the output, and visualized initial cluster trends.
2. **Leakage-Free Exploratory Data Analysis (EDA):**
   Filtered raw CPU logs to isolate only the training timestamps (`train_cpu`) before running detailed visualizations and distributions, ensuring zero test data leakage.
3. **Baseline Benchmarking:**
   Built **Naive** (previous 5-min value) and **Moving Average** baselines and calculated their MAE and RMSE to establish minimum performance benchmarks before training ML models.
4. **Feature Engineering (5-Min Interval Lags):**
   Created past CPU lag features at 5, 10, 15, and 30-minute intervals (`lag_1`, `lag_2`, `lag_3`, `lag_6`) so models can capture short-term temporal patterns, then removed initial `NaN` rows.
5. **Chronological Train-Test Split:**
   Split the dataset sequentially without shuffling (**first 98 rows for training, last 24 rows for testing**) to preserve time order and mirror real-world forecasting.
6. **Model Training & Selection:**
   Trained and compared **Linear Regression** (to capture direct linear continuity without overfitting on a small dataset) against **Random Forest Regressor** (to test non-linear tree-based pattern detection).
7. **Evaluation & Visualization:**
   Compared all models using MAE and RMSE metrics, plotted **Actual vs. Predicted** CPU usage on the test set to visually verify trend tracking, and saved the winning model (`Linear Regression`) using `joblib`.



## 📊 Model Performance Comparison 
>  Note:  The values below represent the actual evaluation results obtained during my training and testing runs.

| Model | MAE | RMSE | Notes |
| :--- | :---: | :---: | :--- |
| **Naive Baseline** | `0.1467` | `0.1847` | Predicts previous 5-min value |
| **Moving Average Baseline** | `0.1315` | `0.1592` | Rolling mean of recent steps |
| **Random Forest Regressor** | `0.1386` | `0.1726` | Slightly overfits due to compact dataset |
| **Linear Regression (Winner)** | **`0.0751`** | **`0.1116`** | **Best generalization & lowest error** |

> **Key Takeaway:** **Linear Regression** achieved the lowest error (`RMSE: 0.1116`), outperforming both baseline heuristics and Random Forest by capturing the strong linear continuity in short-term CPU workloads.

---

## 📁 Repository Structure

```text
vm-cpu-utilization-forecasting/
├── notebook.ipynb                   # Complete EDA, feature engineering, modeling & plots
├── cpu_forecast_linear_model.pkl    # Serialized winning Linear Regression model
├── predict.py                       # Sample inference script using the saved model
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```

## Setup

1. Clone the repository

2. Create virtual environment: ``` python -m venv venv ```

3. Activate virtual environment:``` venv\Scripts\activate ```

4. Install dependencies:``` pip install -r requirements.txt ```

5. Run the application:```python predict.py```