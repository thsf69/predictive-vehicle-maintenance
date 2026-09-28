import pandas as pd
import joblib
import os
import logging
from src.preprocessing import load_preprocessor

logger = logging.getLogger(__name__)

class MaintenancePredictor:
    def __init__(self, models_dir='models/'):
        self.models_dir = models_dir
        self.preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
        self.model_path = os.path.join(models_dir, 'best_model.joblib')
        
        # Load components
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        
    def predict(self, input_data: dict) -> dict:
        """
        Predicts maintenance risk for a given set of sensor readings.
        """
        from src.feature_engineering import apply_feature_engineering
        try:
            # Convert input to DataFrame
            df = pd.DataFrame([input_data])
            
            # Apply feature engineering
            df = apply_feature_engineering(df)
            
            # Apply preprocessing
            X_processed = self.preprocessor.transform(df)
            
            # Predict
            prediction = self.model.predict(X_processed)[0]
            probability = self.model.predict_proba(X_processed)[0][1]
            
            return {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_category': 'HIGH' if prediction == 1 else 'LOW',
                'status': 'success'
            }
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
