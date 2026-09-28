import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc
import joblib
import os
import argparse
import json
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_models(processed_data_dir, models_dir, results_dir):
    """Evaluates trained models and saves metrics/plots."""
    print("Loading test data...")
    test_df = pd.read_csv(os.path.join(processed_data_dir, 'test.csv'))
    
    X_test = test_df.drop(columns=['maintenance_risk'])
    y_test = test_df['maintenance_risk']
    
    models = ['LogisticRegression', 'RandomForest', 'HistGradientBoosting']
    metrics = {}
    
    os.makedirs(os.path.join(results_dir, 'metrics'), exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'figures'), exist_ok=True)
    
    for name in models:
        print(f"Evaluating {name}...")
        model = joblib.load(os.path.join(models_dir, f"{name}.joblib"))
        
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        roc_auc = roc_auc_score(y_test, y_prob)
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = auc(recall, precision)
        
        report = classification_report(y_test, y_pred, output_dict=True)
        
        metrics[name] = {
            'accuracy': report['accuracy'],
            'precision_class_1': report['1']['precision'],
            'recall_class_1': report['1']['recall'],
            'f1_class_1': report['1']['f1-score'],
            'roc_auc': roc_auc,
            'pr_auc': pr_auc
        }
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'{name} - Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig(os.path.join(results_dir, 'figures', f'{name}_confusion_matrix.png'))
        plt.close()

        # Feature Importance for Random Forest
        if name == 'RandomForest':
            importances = model.feature_importances_
            features = X_test.columns
            importance_df = pd.DataFrame({'Feature': features, 'Importance': importances})
            importance_df = importance_df.sort_values(by='Importance', ascending=False).head(10)
            
            plt.figure(figsize=(10,6))
            sns.barplot(x='Importance', y='Feature', data=importance_df)
            plt.title('Top 10 Feature Importances (Random Forest)')
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, 'figures', 'feature_importance.png'))
            plt.close()
    
    # Save metrics
    with open(os.path.join(results_dir, 'metrics', 'evaluation_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
        
    print("Evaluation completed. Metrics and figures saved.")
    
    # Select best model based on F1 score for class 1 (maintenance risk)
    best_model_name = max(metrics, key=lambda k: metrics[k]['f1_class_1'])
    print(f"\nBest Model based on F1-score (Class 1): {best_model_name}")
    print(json.dumps(metrics[best_model_name], indent=4))
    
    # Copy best model to 'best_model.joblib'
    best_model_path = os.path.join(models_dir, f"{best_model_name}.joblib")
    final_model_path = os.path.join(models_dir, "best_model.joblib")
    os.system(f"cp {best_model_path} {final_model_path}")
    print(f"Saved {best_model_name} as best_model.joblib")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='data/processed/')
    parser.add_argument('--models', type=str, default='models/')
    parser.add_argument('--results', type=str, default='results/')
    args = parser.parse_args()
    
    evaluate_models(args.processed, args.models, args.results)
