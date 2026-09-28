# Model Card: Predictive Vehicle Maintenance

## Model Details
- **Model Name:** Predictive Vehicle Maintenance Model
- **Algorithm:** Random Forest Classifier (Scikit-Learn)
- **Date:** 2026-09-28
- **Purpose:** Predict elevated maintenance risk based on vehicle sensor telemetry.

## Intended Use
- **Primary Use Case:** Identifying vehicles with a high probability of requiring maintenance before failure occurs.
- **Out-of-Scope Use:** This model is a **portfolio/research prototype and is not a certified automotive diagnostic or safety system.** It must not be used to replace professional mechanical inspection or to make safety-critical decisions.

## Training Data
- **Dataset:** Synthetically generated vehicle telemetry.
- **Size:** 8,000 training records.
- **Features Used:** 
  - `vehicle_type`
  - `operating_hours`
  - `engine_temperature`
  - `rpm`
  - `vehicle_speed`
  - `oil_pressure`
  - `vibration_level`
  - `battery_voltage`
- **Target Variable:** `maintenance_risk` (0: Low, 1: High)
- **Data Preprocessing:** Median/Mode imputation, Standard Scaling for continuous variables, One-Hot Encoding for categorical variables.

## Evaluation
- **Methodology:** Evaluated on a 20% holdout test set (2,000 records).
- **Actual Metrics (Test Set):**
  - **ROC-AUC:** 0.992
  - **Precision (High Risk):** 0.839
  - **Recall (High Risk):** 0.925
  - **F1-Score (High Risk):** 0.880
  - **Accuracy:** 97.15%

*A high recall score was targeted to ensure potential failures are not missed (false negatives).*

## Feature Importance Findings
Based on the Random Forest feature importances, the top contributing features for predicting maintenance risk were:
1. `operating_hours`
2. `engine_temperature`
3. `oil_pressure`

*Note: Correlation does not imply causation. High engine temperature is a predictor, not necessarily the sole mechanical cause of failure.*

## Ethical & Deployment Considerations
- **Limitations:** The model assumes independent observations and does not capture complex sequential time-series patterns (e.g., how a sensor behaves over consecutive minutes).
- **Potential Biases:** Since data is synthetic, biases present in real-world data (e.g., unequal maintenance records across different regions or vehicle owners) are not captured.
- **Failure Modes:** Anomalous sensor values caused by broken sensors (rather than vehicle failure) may trigger false positives.
