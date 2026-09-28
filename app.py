import streamlit as st
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.predict import MaintenancePredictor

# Set page config
st.set_page_config(page_title="Predictive Vehicle Maintenance", page_icon="🚗", layout="wide")

@st.cache_resource
def load_predictor():
    return MaintenancePredictor()

predictor = load_predictor()

# App layout
st.title("🚗 Predictive Vehicle Maintenance Analytics System")
st.markdown("This dashboard demonstrates a machine learning pipeline that predicts elevated vehicle maintenance risk based on sensor telemetry data.")

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["Overview & Insights", "Prediction Interface", "Model Performance"])

st.sidebar.markdown("---")
st.sidebar.info(
    "**Disclaimer:**\n"
    "This model is a portfolio/research prototype and is not a certified automotive diagnostic or safety system. "
    "It predicts elevated failure risk from the available sensor features."
)

if page == "Overview & Insights":
    st.header("Project Overview")
    st.markdown("""
    Given historical vehicle sensor and operating data, this system identifies vehicles or operating conditions associated with elevated maintenance risk.
    
    **Dataset:** Synthetic vehicle telemetry data simulating normal operation and potential fault conditions.
    **Target:** Binary classification of maintenance risk (0: Low, 1: High).
    """)
    
    st.subheader("Feature Importances")
    feature_importance_path = 'results/figures/feature_importance.png'
    if os.path.exists(feature_importance_path):
        st.image(feature_importance_path, caption='Top Features Influencing Model Predictions')
    else:
        st.warning("Feature importance plot not found. Run training pipeline first.")

elif page == "Prediction Interface":
    st.header("Live Prediction Interface")
    st.markdown("Enter simulated sensor readings below to predict maintenance risk.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        vehicle_type = st.selectbox("Vehicle Type", ["Sedan", "SUV", "Truck", "Van"])
        operating_hours = st.number_input("Operating Hours", min_value=0, max_value=20000, value=5000, step=100)
        engine_temp = st.slider("Engine Temperature (°C)", min_value=60.0, max_value=130.0, value=90.0)
        rpm = st.number_input("RPM", min_value=0, max_value=8000, value=2500, step=100)
        
    with col2:
        speed = st.slider("Vehicle Speed (km/h)", min_value=0.0, max_value=200.0, value=60.0)
        oil_pressure = st.slider("Oil Pressure (psi)", min_value=0.0, max_value=100.0, value=40.0)
        vibration = st.slider("Vibration Level", min_value=0.0, max_value=2.0, value=0.3, step=0.1)
        battery = st.slider("Battery Voltage (V)", min_value=9.0, max_value=16.0, value=13.5, step=0.1)
        
    if st.button("Predict Maintenance Risk", type="primary"):
        input_data = {
            'vehicle_type': vehicle_type,
            'operating_hours': operating_hours,
            'engine_temperature': engine_temp,
            'rpm': rpm,
            'vehicle_speed': speed,
            'oil_pressure': oil_pressure,
            'vibration_level': vibration,
            'battery_voltage': battery
        }
        
        with st.spinner("Analyzing data..."):
            result = predictor.predict(input_data)
            
        if result['status'] == 'success':
            st.markdown("### Results")
            if result['risk_category'] == 'HIGH':
                st.error(f"⚠️ **HIGH RISK** (Probability: {result['probability']:.1%})")
                st.markdown("Recommendation: Schedule maintenance inspection immediately.")
            else:
                st.success(f"✅ **LOW RISK** (Probability: {result['probability']:.1%})")
                st.markdown("Recommendation: Continue normal operation.")
        else:
            st.error(f"Error during prediction: {result.get('message', 'Unknown error')}")

elif page == "Model Performance":
    st.header("Model Evaluation Metrics")
    
    metrics_path = 'results/metrics/evaluation_metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
            
        # Display best model
        best_model_name = max(metrics, key=lambda k: metrics[k]['f1_class_1'])
        st.subheader(f"Selected Model: {best_model_name}")
        
        best_metrics = metrics[best_model_name]
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("ROC-AUC", f"{best_metrics['roc_auc']:.3f}")
        mcol2.metric("Precision (Risk=1)", f"{best_metrics['precision_class_1']:.3f}")
        mcol3.metric("Recall (Risk=1)", f"{best_metrics['recall_class_1']:.3f}")
        mcol4.metric("F1-Score (Risk=1)", f"{best_metrics['f1_class_1']:.3f}")
        
        st.markdown("### Confusion Matrix")
        cm_path = f'results/figures/{best_model_name}_confusion_matrix.png'
        if os.path.exists(cm_path):
            st.image(cm_path, caption=f'{best_model_name} Confusion Matrix')
            
        st.markdown("---")
        st.markdown("### All Models Comparison")
        comp_df = pd.DataFrame.from_dict(metrics, orient='index')[['roc_auc', 'precision_class_1', 'recall_class_1', 'f1_class_1']]
        st.dataframe(comp_df.style.highlight_max(axis=0, color='lightgreen'))
        
    else:
        st.warning("Metrics file not found. Run training and evaluation pipeline.")
