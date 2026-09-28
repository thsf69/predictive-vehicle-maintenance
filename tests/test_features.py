import pytest
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.feature_engineering import apply_feature_engineering

def test_apply_feature_engineering():
    df = pd.DataFrame({
        'engine_temperature': [90.0, 105.0],
        'rpm': [2000, 3000],
        'battery_voltage': [14.0, 12.0],
        'vibration_level': [0.3, 0.9]
    })
    
    engineered = apply_feature_engineering(df)
    
    assert 'temp_rpm_ratio' in engineered.columns
    assert 'voltage_drop' in engineered.columns
    assert 'is_high_vibration' in engineered.columns
    
    # Check values
    assert engineered['voltage_drop'].iloc[0] == 0.0
    assert engineered['voltage_drop'].iloc[1] == 2.0
    assert engineered['is_high_vibration'].iloc[0] == 0
    assert engineered['is_high_vibration'].iloc[1] == 1
