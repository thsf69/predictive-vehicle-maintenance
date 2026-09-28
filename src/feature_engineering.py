import pandas as pd
import logging

logger = logging.getLogger(__name__)

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering to the raw dataframe.
    """
    df = df.copy()
    
    # Example feature 1: Temperature to RPM ratio
    # Add a small epsilon to avoid division by zero
    df['temp_rpm_ratio'] = df['engine_temperature'] / (df['rpm'] + 1e-5)
    
    # Example feature 2: Voltage Drop (Assuming 14.0V is nominal max)
    df['voltage_drop'] = 14.0 - df['battery_voltage']
    
    # Example feature 3: High vibration flag
    df['is_high_vibration'] = (df['vibration_level'] > 0.8).astype(int)
    
    logger.info(f"Feature engineering applied. New shape: {df.shape}")
    return df
