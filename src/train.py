"""
Model training module for AI Traffic Accident Severity Prediction System.
Handles training, validation, and testing of multiple machine learning models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import joblib
import os
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_selection, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Handles training and evaluation of ML models for accident severity prediction."""

    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer.

        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.trained_models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_names = None

    def split_data_temporal(self, X: pd.DataFrame, y: pd.Series,
                           test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Split data using temporal strategy (train on older data, test on newer data).
        This prevents temporal leakage.

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Proportion for test set
            val_size: Proportion for validation set (from training data)

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # Assuming we have a temporal column like 'accident_year' or 'Start_Time'
        temporal_col = None
        for col in ['accident_year', 'Start_Time']:
            if col in X.columns:
                temporal_col = col
                break

        if temporal_col is None:
            print("Warning: No temporal column found. Using random split.")
            return self._split_data_random(X, y, test_size, val_size)

        # Sort by temporal column
        sorted_indices = X[temporal_col].argsort()
        X_sorted = X.iloc[sorted_indices]
        y_sorted = y.iloc[sorted_indices]

        # Calculate split indices
        n_total = len(X_sorted)
        n_test = int(n_total * test_size)
        n_val = int((n_total - n_test) * val_size)

        # Split: train (oldest) -> validation -> test (newest)
        X_train = X_sorted.iloc[:n_total - n_test - n_val]
        X_val = X_sorted.iloc[n_total - n_test - n_val:n_total - n_test]
        X_test = X_sorted.iloc[n_total - n_test:]

        y_train = y_sorted.iloc[:n_total - n_test - n_val]
        y_val = y_sorted.iloc[n_total - n_test - n_val:n_total - n_test]
        y_test = y_sorted.iloc[n_total - n_test:]

        print(f"Temporal split completed:")
        print(f"  Train: {len(X_train)} samples (oldest)")
        print(f"  Validation: {len(X_val)} samples")
        print(f"  Test: {len(X_test)} samples (newest)")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _split_data_random(self, X: pd.DataFrame, y: pd.Series,
                          test_size: float = 0.2, val_size: float = 0.1) -> Tuple:
        """
        Fallback to random stratified split.

        Args:
            X: Feature DataFrame
            y: Target series
            test_size: Proportion for test set
            val_size: Proportion for validation set

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state,
            stratify=y
        )

        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted,
            random_state=self.random_state, stratify=y_temp
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def initialize_models(self):
        """Initialize the models to be trained."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=self.random_state,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=10
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1,
                verbose=-1
            )
        }

        print(f"Initialized {len(self.models)} models for training.")

    def train_model(self, model_name: str, model: Any,
                   X_train: pd.DataFrame, y_train: np.ndarray,
                   X_val: pd.DataFrame = None, y_val: np.ndarray = None) -> Dict:
        """
        Train a single model and evaluate its performance.

        Args:
            model_name: Name of the model
            model: Scikit-learn compatible model
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)

        Returns:
            Dictionary with training results
        """
        start_time = time.time()

        # Train the model
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        # Predictions
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)

        results = {
            'model': model,
            'train_time': train_time,
            'train_accuracy': train_accuracy,
            'predictions': {}
        }

        # Validation evaluation
        if X_val is not None and y_val is not None:
            val_start = time.time()
            y_val_pred = model.predict(X_val)
            val_time = time.time() - val_start

            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_precision = precision_score(y_val, y_val_pred, average='weighted', zero_division=0)
            val_recall = recall_score(y_val, y_val_pred, average='weighted', zero_division=0)
            val_f1 = f1_score(y_val, y_val_pred, average='weighted', zero_division=0)

            # Try ROC-AUC for multiclass
            try:
                if hasattr(model, "predict_proba"):
                    y_val_proba = model.predict_proba(X_val)
                    val_roc_auc = roc_auc_score(y_val, y_val_proba, multi_class='ovr', average='weighted')
                else:
                    val_roc_auc = None
            except Exception:
                val_roc_auc = None

            results.update({
                'val_accuracy': val_accuracy,
                'val_precision': val_precision,
                'val_recall': val_recall,
                'val_f1': val_f1,
                'val_roc_auc': val_roc_auc,
                'val_time': val_time,
                'predictions': {
                    'train': y_train_pred,
                    'val': y_val_pred
                }
            })

            print(f"{model_name:20} | Train Acc: {train_accuracy:.4f} | "
                  f"Val Acc: {val_accuracy:.4f} | Val F1: {val_f1:.4f} | "
                  f"Time: {train_time:.1f}s")
        else:
            results['predictions']['train'] = y_train_pred
            print(f"{model_name:20} | Train Acc: {train_accuracy:.4f} | "
                  f"Time: {train_time:.1f}s")

        return results

    def train_all_models(self, X_train: pd.DataFrame, y_train: np.ndarray,
                        X_val: pd.DataFrame = None, y_val: np.ndarray = None):
        """
        Train all initialized models.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (optional)
            y_val: Validation targets (optional)
        """
        self.initialize_models()

        print("\nTraining models...")
        print("-" * 80)

        for name, model in self.models.items():
            results = self.train_model(name, model, X_train, y_train, X_val, y_val)
            self.trained_models[name] = results['model']
            self.model_scores[name] = results

        print("-" * 80)

    def hyperparameter_tuning(self, model_name: str, model: Any,
                             X_train: pd.DataFrame, y_train: np.ndarray,
                             cv: int = 3, n_iter: int = 20) -> Any:
        """
        Perform hyperparameter tuning for a specific model.

        Args:
            model_name: Name of the model to tune
            model: Base model to tune
            X_train: Training features
            y_train: Training targets
            cv: Number of cross-validation folds
            n_iter: Number of iterations for RandomizedSearchCV

        Returns:
            Best model from hyperparameter tuning
        """
        print(f"\nPerforming hyperparameter tuning for {model_name}...")

        # Define parameter grids for each model
        param_grids = {
            'logistic_regression': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga']
            },
            'decision_tree': {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0],
                'num_leaves': [31, 50, 100]
            }
        }

        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}. Using default model.")
            return model

        param_grid = param_grids[model_name]

        # Use RandomizedSearchCV for efficiency
        random_search = RandomizedSearchCV(
            model, param_grid, n_iter=n_iter, cv=cv,
            scoring='f1_weighted', random_state=self.random_state,
            n_jobs=-1, verbose=0
        )

        start_time = time.time()
        random_search.fit(X_train, y_train)
        tune_time = time.time() - start_time

        print(f"Tuning completed in {tune_time:.1f}s")
        print(f"Best parameters: {random_search.best_params_}")
        print(f"Best CV score: {random_search.best_score_:.4f}")

        return random_search.best_estimator_

    def evaluate_model(self, model: Any, X_test: pd.DataFrame, y_test: np.ndarray) -> Dict:
        """
        Evaluate a model on the test set.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary with evaluation metrics
        """
        # Predictions
        y_test_pred = model.predict(X_test)
        y_test_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Metrics
        test_accuracy = accuracy_score(y_test, y_test_pred)
        test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
        test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

        # ROC-AUC for multiclass
        try:
            if y_test_proba is not None:
                test_roc_auc = roc_auc_score(y_test, y_test_proba, multi_class='ovr', average='weighted')
            else:
                test_roc_auc = None
        except Exception:
            test_roc_auc = None

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)

        # Classification report
        class_report = classification_report(y_test, y_test_pred, output_dict=True)

        results = {
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_test_pred,
            'probabilities': y_test_proba
        }

        return results

    def select_best_model(self, metric: str = 'val_f1') -> Tuple[str, Any]:
        """
        Select the best model based on validation performance.

        Args:
            metric: Metric to use for selection (val_f1, val_accuracy, etc.)

        Returns:
            Tuple of (best_model_name, best_model)
        """
        if not self.model_scores:
            raise ValueError("No models have been trained yet.")

        # Filter scores that have the requested metric
        valid_scores = {}
        for name, scores in self.model_scores.items():
            if metric in scores and scores[metric] is not None:
                valid_scores[name] = scores[metric]

        if not valid_scores:
            # Fallback to train accuracy if validation metrics not available
            metric = 'train_accuracy'
            valid_scores = {name: scores['train_accuracy'] for name, scores in self.model_scores.items()}

        # Select best model
        self.best_model_name = max(valid_scores, key=valid_scores.get)
        self.best_model = self.trained_models[self.best_model_name]

        print(f"Best model selected: {self.best_model_name} ({metric}: {valid_scores[self.best_model_name]:.4f})")

        return self.best_model_name, self.best_model

    def save_model(self, model: Any, filename: str = "best_model.joblib"):
        """
        Save a trained model.

        Args:
            model: Model to save
            filename: Output filename
        """
        os.makedirs("models/", exist_ok=True)
        filepath = os.path.join("models/", filename)
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filename: str = "best_model.joblib") -> Any:
        """
        Load a trained model.

        Args:
            filename: Model filename

        Returns:
            Loaded model
        """
        filepath = os.path.join("models/", filename)
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            raise FileNotFoundError(f"Model file not found: {filepath}")

    def save_training_results(self, filename: str = "training_results.joblib"):
        """
        Save training results and scores.

        Args:
            filename: Output filename
        """
        os.makedirs("models/", exist_ok=True)
        filepath = os.path.join("models/", filename)
        results_dict = {
            'model_scores': self.model_scores,
            'best_model_name': self.best_model_name,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        joblib.dump(results_dict, filepath)
        print(f"Training results saved to {filepath}")


def main():
    """Main function for testing the model trainer."""
    print("Model Trainer Module - Testing with sample data")

    # Create sample data for testing
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Generate synthetic data
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )
    # Add a temporal feature for testing temporal split
    X['accident_year'] = np.random.choice([2018, 2019, 2020, 2021, 2022], n_samples)
    y = np.random.randint(0, 4, n_samples)  # 4 severity classes

    print(f"Generated sample data: {X.shape}")

    # Initialize trainer
    trainer = ModelTrainer(random_state=42)

    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.split_data_temporal(X, y)
    trainer.X_train, trainer.X_val, trainer.X_test = X_train, X_val, X_test
    trainer.y_train, trainer.y_val, trainer.y_test = y_train, y_val, y_test

    # Train all models
    trainer.train_all_models(X_train, y_train, X_val, y_val)

    # Select best model
    best_name, best_model = trainer.select_best_model(metric='val_f1')

    # Evaluate best model on test set
    test_results = trainer.evaluate_model(best_model, X_test, y_test)
    print(f"\nBest model ({best_name}) test performance:")
    print(f"  Accuracy: {test_results['test_accuracy']:.4f}")
    print(f"  F1-Score: {test_results['test_f1']:.4f}")

    # Save model and results
    trainer.save_model(best_model)
    trainer.save_training_results()


if __name__ == "__main__":
    main()