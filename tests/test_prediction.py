import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import MaintenancePredictor

def test_predictor_initialization():
    try:
        predictor = MaintenancePredictor(models_dir='models/')
        assert predictor is not None
    except Exception as e:
        pytest.fail(f"Initialization failed: {e}")

def test_predict_valid_input():
    predictor = MaintenancePredictor(models_dir='models/')
    input_data = {
        'vehicle_type': 'SUV',
        'operating_hours': 5000,
        'engine_temperature': 90.0,
        'rpm': 2500,
        'vehicle_speed': 60.0,
        'oil_pressure': 40.0,
        'vibration_level': 0.3,
        'battery_voltage': 13.5
    }
    result = predictor.predict(input_data)
    assert result['status'] == 'success'
    assert 'prediction' in result
    assert result['prediction'] in [0, 1]
    assert 0.0 <= result['probability'] <= 1.0

def test_predict_invalid_input():
    predictor = MaintenancePredictor(models_dir='models/')
    input_data = {
        'missing_column': 'SUV'
    }
    result = predictor.predict(input_data)
    # The preprocessor will fail because required columns are missing
    assert result['status'] == 'error'
