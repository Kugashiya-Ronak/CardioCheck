"""
Model Evaluation Module for CVD Prediction
Computes accuracy of different models and k-fold cross-validation results
"""
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression as SKLogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import pickle

class ModelEvaluator:
    """Evaluate multiple models and their cross-validation performance"""
    
    def __init__(self, X, y, test_size=0.2, random_state=42):
        self.X = X
        self.y = y
        self.test_size = test_size
        self.random_state = random_state
        self.models = {}
        self.cv_results = {}
        self.accuracies = {}
        
    def train_models(self):
        """Train multiple models"""
        from sklearn.model_selection import train_test_split
        
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state
        )
        
        # Initialize models
        models_dict = {
            'Logistic Regression': SKLogisticRegression(max_iter=1000, random_state=self.random_state),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=self.random_state),
            # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=self.random_state),
            'Support Vector Machine': SVC(kernel='rbf', probability=True, random_state=self.random_state)
        }
        
        # Train and evaluate each model
        for name, model in models_dict.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            if hasattr(model, 'predict_proba'):
                roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
            else:
                roc_auc = 'N/A'
            
            self.accuracies[name] = {
                'accuracy': round(accuracy * 100, 2),
                'precision': round(precision * 100, 2),
                'recall': round(recall * 100, 2),
                'f1_score': round(f1 * 100, 2),
                'roc_auc': round(roc_auc * 100, 2) if roc_auc != 'N/A' else 'N/A'
            }
        
        return X_train, X_test, y_train, y_test
    
    def kfold_validation(self, k=5):
        """Perform k-fold cross-validation"""
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=self.random_state)
        
        for name, model in self.models.items():
            # Cross-validate with multiple metrics
            scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
            cv_results = cross_validate(
                model, self.X, self.y, cv=kfold, scoring=scoring
            )
            
            # Store results
            self.cv_results[name] = {
                'folds': k,
                'accuracy': {
                    'scores': [round(s * 100, 2) for s in cv_results['test_accuracy']],
                    'mean': round(np.mean(cv_results['test_accuracy']) * 100, 2),
                    'std': round(np.std(cv_results['test_accuracy']) * 100, 2)
                },
                'precision': {
                    'scores': [round(s * 100, 2) for s in cv_results['test_precision']],
                    'mean': round(np.mean(cv_results['test_precision']) * 100, 2),
                    'std': round(np.std(cv_results['test_precision']) * 100, 2)
                },
                'recall': {
                    'scores': [round(s * 100, 2) for s in cv_results['test_recall']],
                    'mean': round(np.mean(cv_results['test_recall']) * 100, 2),
                    'std': round(np.std(cv_results['test_recall']) * 100, 2)
                },
                'f1': {
                    'scores': [round(s * 100, 2) for s in cv_results['test_f1']],
                    'mean': round(np.mean(cv_results['test_f1']) * 100, 2),
                    'std': round(np.std(cv_results['test_f1']) * 100, 2)
                },
                'roc_auc': {
                    'scores': [round(s * 100, 2) for s in cv_results['test_roc_auc']],
                    'mean': round(np.mean(cv_results['test_roc_auc']) * 100, 2),
                    'std': round(np.std(cv_results['test_roc_auc']) * 100, 2)
                }
            }
    
    def get_model_comparison(self):
        """Get dictionary for model comparison"""
        return self.accuracies
    
    def get_cv_results(self):
        """Get cross-validation results"""
        return self.cv_results


def load_sample_data():
    """
    Load sample CVD data (you'll need to load your actual dataset)
    Returns X, y arrays
    """
    # This is a placeholder - replace with your actual data loading
    try:
        # Try loading from a pickle or CSV file
        import pandas as pd
        # Replace 'your_cvd_data.csv' with actual path
        df = pd.read_csv('ds_cvd_w3.csv')
        X = df.drop('target', axis=1).values
        y = df['target'].values
        return X, y
    except:
        # Fallback: generate synthetic data for demonstration
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=1000, n_features=11, n_informative=8, 
                                   n_redundant=2, random_state=42)
        return X, y


if __name__ == "__main__":
    # Load data
    X, y = load_sample_data()
    
    # Create evaluator
    evaluator = ModelEvaluator(X, y)
    
    # Train models
    X_train, X_test, y_train, y_test = evaluator.train_models()
    
    # Perform k-fold validation
    evaluator.kfold_validation(k=5)
    
    # Print results
    print("\n=== Model Accuracies ===")
    for model, metrics in evaluator.get_model_comparison().items():
        print(f"\n{model}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value}%")
    
    print("\n=== K-Fold Cross-Validation Results ===")
    for model, results in evaluator.get_cv_results().items():
        print(f"\n{model} ({results['folds']}-Fold):")
        print(f"  Accuracy: {results['accuracy']['mean']}% ± {results['accuracy']['std']}%")
        print(f"  Precision: {results['precision']['mean']}% ± {results['precision']['std']}%")