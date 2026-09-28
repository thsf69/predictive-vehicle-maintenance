import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocessing import build_preprocessor

def test_build_preprocessor():
    num_cols = ['engine_temperature', 'rpm']
    cat_cols = ['vehicle_type']
    
    preprocessor = build_preprocessor(num_cols, cat_cols)
    
    # Create dummy dataframe with missing values
    df = pd.DataFrame({
        'engine_temperature': [90.5, np.nan, 105.0],
        'rpm': [2500, 3000, np.nan],
        'vehicle_type': ['SUV', 'Sedan', np.nan]
    })
    
    processed = preprocessor.fit_transform(df)
    
    # Check that there are no NaNs in processed data
    assert not np.isnan(processed).any()
    
    # Original shape: 3 rows, 3 columns.
    # After one-hot encoding 'vehicle_type' (2 known categories + imputed one), shape will increase.
    # Actually, imputed 'vehicle_type' will be the most frequent ('SUV' or 'Sedan').
    # Let's just check shape is consistent.
    assert processed.shape[0] == 3
    assert processed.shape[1] > 2
