import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
import joblib
import os
import argparse
import time

def train_models(processed_data_dir, models_dir):
    """Trains multiple models and selects the best one based on ROC-AUC."""
    print("Loading processed training data...")
    train_df = pd.read_csv(os.path.join(processed_data_dir, 'train.csv'))
    
    X_train = train_df.drop(columns=['maintenance_risk'])
    y_train = train_df['maintenance_risk']
    
    # Define models
    models = {
        'LogisticRegression': LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        'RandomForest': RandomForestClassifier(random_state=42, class_weight='balanced', n_estimators=100),
        'HistGradientBoosting': HistGradientBoostingClassifier(random_state=42, max_iter=100)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        print(f"{name} trained in {end_time - start_time:.2f} seconds.")
        trained_models[name] = model
        
        # Save each model
        joblib.dump(model, os.path.join(models_dir, f"{name}.joblib"))
    
    print("All models trained and saved.")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='data/processed/')
    parser.add_argument('--models', type=str, default='models/')
    args = parser.parse_args()
    
    train_models(args.processed, args.models)
