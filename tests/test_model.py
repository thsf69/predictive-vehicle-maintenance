import pytest
import os
import sys
import joblib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_model_can_be_loaded():
    model_path = 'models/best_model.joblib'
    if not os.path.exists(model_path):
        pytest.skip("Model file not found. Run training pipeline first.")
        
    model = joblib.load(model_path)
    assert model is not None
    assert hasattr(model, 'predict')
    assert hasattr(model, 'predict_proba')
