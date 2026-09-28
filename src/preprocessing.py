import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging
import joblib

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def build_preprocessor(numerical_cols, categorical_cols):
    """
    Builds a Scikit-Learn preprocessor.
    
    Args:
        numerical_cols (list): List of numerical column names.
        categorical_cols (list): List of categorical column names.
        
    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    return preprocessor

def save_preprocessor(preprocessor, filepath):
    """Saves the preprocessor pipeline."""
    joblib.dump(preprocessor, filepath)
    logger.info(f"Preprocessor saved to {filepath}")

def load_preprocessor(filepath):
    """Loads the preprocessor pipeline."""
    preprocessor = joblib.load(filepath)
    logger.info(f"Preprocessor loaded from {filepath}")
    return preprocessor
