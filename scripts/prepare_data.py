import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_data
from src.preprocessing import build_preprocessor, save_preprocessor
from src.feature_engineering import apply_feature_engineering

def prepare_data(raw_data_path, processed_dir, models_dir):
    """
    Loads raw data, applies feature engineering, splits it, fits preprocessor, transforms data, and saves.
    """
    print(f"Loading raw data from {raw_data_path}...")
    df = load_data(raw_data_path)
    
    # Drop duplicates
    initial_shape = df.shape
    df = df.drop_duplicates(subset=['vehicle_id'])
    print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate records.")
    
    # Apply feature engineering
    print("Applying feature engineering...")
    df = apply_feature_engineering(df)
    
    # Drop identifier columns for modeling
    df = df.drop(columns=['vehicle_id'])
    
    # Separate features and target
    X = df.drop(columns=['maintenance_risk'])
    y = df['maintenance_risk']
    
    print(f"Splitting data... (80% train, 20% test)")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    print(f"Numerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    
    print("Fitting preprocessor on training data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Get column names after preprocessing
    num_cols = preprocessor.transformers_[0][1].get_feature_names_out(numerical_cols)
    cat_cols = preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols)
    all_cols = list(num_cols) + list(cat_cols)
    
    # Create DataFrames
    X_train_df = pd.DataFrame(X_train_processed, columns=all_cols)
    X_test_df = pd.DataFrame(X_test_processed, columns=all_cols)
    
    # Save processed data
    os.makedirs(processed_dir, exist_ok=True)
    
    X_train_df['maintenance_risk'] = y_train.values
    X_test_df['maintenance_risk'] = y_test.values
    
    train_path = os.path.join(processed_dir, 'train.csv')
    test_path = os.path.join(processed_dir, 'test.csv')
    
    X_train_df.to_csv(train_path, index=False)
    X_test_df.to_csv(test_path, index=False)
    print(f"Processed training data saved to {train_path}")
    print(f"Processed test data saved to {test_path}")
    
    # Save preprocessor
    os.makedirs(models_dir, exist_ok=True)
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    save_preprocessor(preprocessor, preprocessor_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=str, default='data/raw/synthetic_vehicle_data.csv')
    parser.add_argument('--processed', type=str, default='data/processed/')
    parser.add_argument('--models', type=str, default='models/')
    args = parser.parse_args()
    
    prepare_data(args.raw, args.processed, args.models)
